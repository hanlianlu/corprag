# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Ingestion pipeline for RAG document processing.

Provides document ingestion with content filtering to prevent
pollution from MinerU's discarded blocks (headers, footers, page numbers, etc.).

Flow: parse_document() -> policy filter -> insert_content_list()
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field
from raganything import RAGAnything

from corprag.converters.office import create_converter
from corprag.ingestion.cleanup import collect_deletion_context
from corprag.ingestion.hash_index import HashIndex
from corprag.ingestion.policy import IngestionPolicy, PolicyStats

if TYPE_CHECKING:
    from corprag.config import CorpragConfig

logger = logging.getLogger(__name__)


class IngestionCancelledError(Exception):
    """Raised when ingestion task is cancelled by the caller."""


class IngestionResult(BaseModel):
    """Ingestion result model."""

    status: Literal["success", "error"] = "success"
    processed: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    total_files: int | None = Field(default=None, ge=0)
    source_type: str | None = None
    doc_id: str | None = None
    source_path: str | None = None
    stats: PolicyStats | None = None
    error: str | None = None
    skipped_files: list[str] | None = None

    # Optional metadata
    folder: str | None = None
    container: str | None = None
    prefix: str | None = None
    blob_path: str | None = None


class IngestionPipeline:
    """Document ingestion with content filtering.

    Uses parse_document -> policy filter -> insert_content_list flow
    to filter out MinerU discarded blocks before indexing.

    Public API:
      - aingest_from_local(...)
      - aingest_from_azure_blob(...)
      - aingest_from_snowflake(...)
      - aingest_content_list(...)
    """

    def __init__(
        self,
        rag_instance: RAGAnything,
        config: CorpragConfig,
        max_concurrent: int = 4,
        policy: IngestionPolicy | None = None,
        mineru_backend: str | None = None,
        cancel_checker: Callable[[], Awaitable[bool]] | None = None,
    ) -> None:
        self.rag = rag_instance
        self.config = config
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # MinerU backend for parse_document kwargs (None if using docling)
        self.mineru_backend = mineru_backend

        # Content filtering policy
        self.policy = policy or IngestionPolicy()

        # LibreOffice converter for Excel-to-PDF auto-conversion
        self.converter = create_converter(config)

        # Hash index for content deduplication
        self._hash_index = HashIndex(
            self.config.working_dir_path,
            self.config.sources_dir,
        )

        # Callback for cancellation checking (replaces Redis task registry)
        self._cancel_checker = cancel_checker
        self._cancel_check_interval = 5  # Check every N files

    async def _check_cancelled(self) -> None:
        """Check if this ingestion task has been cancelled.

        Uses the cancel_checker callback provided at construction time.

        Raises:
            IngestionCancelledError: If task was cancelled
            asyncio.CancelledError: Re-raised for asyncio cancellation
        """
        # Check asyncio cancellation first
        if asyncio.current_task() and asyncio.current_task().cancelled():  # type: ignore[union-attr]
            raise asyncio.CancelledError()

        # Check via caller-provided callback
        if self._cancel_checker and await self._cancel_checker():
            logger.info("Ingestion task cancelled by caller")
            raise IngestionCancelledError("Task cancelled by caller")

    # ─────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────

    def _get_storage_dir(
        self,
        base_dir: Path,
        source_type: str,
        *path_parts: str,
    ) -> Path:
        """Get storage directory for a specific source type."""
        storage_dir = base_dir / source_type / Path(*path_parts)
        storage_dir.mkdir(parents=True, exist_ok=True)
        return storage_dir

    def _get_source_dir(self, source_type: str, *path_parts: str) -> Path:
        """Get source directory for a specific source type."""
        return self._get_storage_dir(
            self.config.sources_dir,
            source_type,
            *path_parts,
        )

    def _get_artifacts_dir(self, source_type: str, *path_parts: str) -> Path:
        """Get artifacts directory for a specific source type."""
        return self._get_storage_dir(
            self.config.artifacts_dir,
            source_type,
            *path_parts,
        )

    async def _maybe_convert_excel_to_pdf(
        self,
        file_path: Path,
        output_dir: Path,
    ) -> Path:
        """Auto-convert Excel to PDF if configured, otherwise return original path."""
        if not self.converter.should_convert(file_path):
            return file_path

        try:
            pdf_path = await asyncio.to_thread(
                self.converter.convert_to_pdf,
                source_path=file_path,
                output_dir=output_dir,
            )
            logger.info(f"Auto-converted Excel to PDF: {file_path.name} -> {pdf_path.name}")
            return pdf_path
        except Exception as e:
            logger.warning(
                f"Excel-to-PDF conversion failed for {file_path.name}, using original: {e}"
            )
            return file_path

    def _copy_to_sources_local(self, src_path: Path) -> Path:
        """Copy file to sources/local/, auto-converting Excel to PDF if configured."""
        source_dir = self._get_source_dir("local")
        dest_path = source_dir / src_path.name

        # Handle filename conflicts
        if dest_path.exists() and not dest_path.samefile(src_path):
            base = src_path.stem
            suffix = src_path.suffix
            counter = 1
            while dest_path.exists():
                dest_path = source_dir / f"{base}_{counter}{suffix}"
                counter += 1

        # Copy if not already in sources/local
        if not dest_path.exists():
            shutil.copy2(src_path, dest_path)
            logger.debug(f"Copied {src_path} to {dest_path}")

        # Auto-convert Excel to PDF (synchronous version)
        if self.converter.should_convert(dest_path):
            try:
                pdf_path = self.converter.convert_to_pdf(
                    source_path=dest_path,
                    output_dir=source_dir,
                )
                logger.info(f"Auto-converted Excel to PDF: {src_path.name} -> {pdf_path.name}")
                return pdf_path
            except Exception as e:
                logger.warning(
                    f"Excel-to-PDF conversion failed for {src_path.name}, using original: {e}"
                )

        return dest_path

    async def _download_blob_to_storage_async(
        self,
        source: Any,  # AzureBlobDataSource
        container_name: str,
        blob_path: str,
    ) -> Path:
        """Download Azure Blob to sources directory, auto-converting Excel to PDF."""
        source_dir = self._get_source_dir("azure_blobs", container_name)
        target_path = source_dir / blob_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        content = await source.aload_document(blob_path)
        await asyncio.to_thread(target_path.write_bytes, content)

        logger.info("Downloaded blob to: %s", target_path)

        # Auto-convert Excel to PDF using centralized method
        return await self._maybe_convert_excel_to_pdf(target_path, source_dir)

    def _extract_relative_source_path(self, file_path: str) -> str | None:
        """Extract relative path within sources/ from a full file path.

        E.g., "/abs/path/corprag_storage/sources/local/file.pdf" -> "local/file.pdf"
        """
        sources_marker = "/sources/"
        idx = file_path.find(sources_marker)
        if idx != -1:
            return file_path[idx + len(sources_marker) :]
        return None

    def _resolve_source_file(self, file_path: str) -> Path | None:
        """Resolve a file path to its actual location in the sources directory."""
        # Try as absolute path first
        candidate = Path(file_path)
        if candidate.exists() and candidate.is_file():
            return candidate

        # Try relative to sources directory
        rel = self._extract_relative_source_path(file_path)
        if rel:
            candidate = self.config.sources_dir / rel
            if candidate.exists():
                return candidate

        # Try as basename in any subdirectory of sources
        basename = Path(file_path).name
        for match in self.config.sources_dir.rglob(basename):
            if match.is_file():
                return match

        return None

    # ─────────────────────────────────────────────────────────────────
    # Core ingestion with policy filtering
    # ─────────────────────────────────────────────────────────────────

    async def _ingest_single_file_with_policy(
        self,
        file_path: Path,
        artifacts_dir: Path,
        content_hash: str | None = None,
    ) -> IngestionResult:
        """Ingest a single file with content filtering.

        Flow: parse_document() -> policy.apply() -> insert_content_list()
        """
        async with self._semaphore:
            try:
                # Check for cancellation before heavy processing
                await self._check_cancelled()

                # Step 1: Parse document (returns raw content_list)
                parse_method = self.config.parse_method

                # Build kwargs for parse_document (backend only for MinerU parser)
                parse_kwargs: dict[str, Any] = {}
                if self.mineru_backend:
                    parse_kwargs["backend"] = self.mineru_backend

                content_list, doc_id = await self.rag.parse_document(
                    file_path=str(file_path),
                    output_dir=str(artifacts_dir),
                    parse_method=parse_method,
                    **parse_kwargs,
                )

                # Step 2: Apply policy to filter discarded/noise content
                result = self.policy.apply(content_list)

                # Log stats with drop rate
                logger.info(
                    f"Policy filter [{file_path.name}]: "
                    f"total={result.stats.total}, indexed={result.stats.indexed}, "
                    f"dropped={result.stats.dropped_by_type} ({result.stats.drop_rate:.1f}%)"
                )

                # Step 3: Insert filtered content
                if result.index_stream:
                    await self.rag.insert_content_list(  # type: ignore[misc]
                        content_list=result.index_stream,
                        file_path=str(file_path),
                        doc_id=doc_id,
                    )

                # Step 4: Register hash for deduplication (if hash was provided)
                if content_hash and doc_id:
                    self._hash_index.register(content_hash, doc_id, str(file_path))

                return IngestionResult(
                    status="success",
                    processed=1,
                    doc_id=doc_id,
                    source_path=str(file_path),
                    stats=result.stats,
                )

            except Exception as e:
                logger.error(f"Failed to ingest {file_path}: {e}")
                return IngestionResult(
                    status="error",
                    processed=0,
                    source_path=str(file_path),
                    error=str(e),
                )

    # ─────────────────────────────────────────────────────────────────
    # Public API: Local ingestion
    # ─────────────────────────────────────────────────────────────────

    async def aingest_from_local(
        self,
        path: Path,
        replace: bool = False,
        recursive: bool = True,
        sync_hashes: bool = False,
    ) -> IngestionResult:
        """Ingest from local filesystem (file or directory).

        RAGAnything's parser automatically handles all supported file types
        (PDF, images, Office documents, etc.) based on file extension.

        Pre-parse deduplication: Files with identical content (by SHA256 hash)
        are skipped unless replace=True.

        Args:
            path: Path to file or directory to ingest
            replace: If True, delete existing docs with same basename before ingesting
            recursive: If True, recursively process directories
            sync_hashes: If True, sync hashes for existing processed documents
                        before deduplication check
        """
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        # Sync hashes for existing documents if requested
        if sync_hashes:
            await self._hash_index.sync_existing()

        artifacts_dir = self._get_artifacts_dir("local")

        # Single file
        if path.is_file():
            # Check for duplicate BEFORE copying/parsing
            should_skip, content_hash, reason = await self._hash_index.should_skip_file(
                path, replace
            )

            if should_skip:
                logger.info(f"Skipped (dedup): {path.name} - {reason}")
                return IngestionResult(
                    status="success",
                    processed=0,
                    skipped=1,
                    total_files=1,
                    source_type="local",
                    source_path=str(path),
                    skipped_files=[path.name],
                )

            if replace:
                await self.adelete_files(filenames=[path.name], delete_source=False)

            # Copy to sources/local/
            source_path = self._copy_to_sources_local(path)

            logger.info(
                "Ingesting local file: %s -> %s",
                path.resolve(),
                source_path,
            )

            result = await self._ingest_single_file_with_policy(
                source_path, artifacts_dir, content_hash=content_hash
            )
            result.source_type = "local"
            result.total_files = 1
            return result

        # Directory
        if path.is_dir():
            # Collect files to process with dedup check
            pattern = "**/*" if recursive else "*"
            files_to_process: list[tuple[Path, str]] = []
            skipped_count = 0
            skipped_files: list[str] = []

            for file_path in path.glob(pattern):
                if not file_path.is_file():
                    continue

                # Check for duplicate BEFORE copying
                (
                    should_skip,
                    content_hash,
                    reason,
                ) = await self._hash_index.should_skip_file(file_path, replace)

                if should_skip:
                    skipped_count += 1
                    skipped_files.append(file_path.name)
                    logger.info(f"Skipped (dedup): {file_path.name}")
                    continue

                if replace:
                    await self.adelete_files(filenames=[file_path.name], delete_source=False)

                copied_path = self._copy_to_sources_local(file_path)
                files_to_process.append((copied_path, content_hash or ""))

            total_files = len(files_to_process) + skipped_count

            if not files_to_process:
                if skipped_count > 0:
                    logger.info(
                        f"No new files to ingest from {path} "
                        f"({skipped_count} skipped as duplicates)"
                    )
                else:
                    logger.warning(f"No matching files found in {path}")
                return IngestionResult(
                    status="success",
                    processed=0,
                    skipped=skipped_count,
                    total_files=total_files,
                    source_type="local",
                    folder=str(path),
                    skipped_files=skipped_files if skipped_files else None,
                )

            logger.info(
                "Ingesting %d local files from %s (%d skipped, max %d concurrent)",
                len(files_to_process),
                path,
                skipped_count,
                self.max_concurrent,
            )

            # Process files concurrently with semaphore control
            tasks = [
                self._ingest_single_file_with_policy(
                    fp, artifacts_dir, content_hash=ch if ch else None
                )
                for fp, ch in files_to_process
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Count successes
            success_count = sum(
                1 for r in results if isinstance(r, IngestionResult) and r.status == "success"
            )

            # Invalidate hash index cache after batch ingestion
            self._hash_index.invalidate()

            return IngestionResult(
                status="success",
                processed=success_count,
                skipped=skipped_count,
                total_files=total_files,
                source_type="local",
                folder=str(path),
                skipped_files=skipped_files if skipped_files else None,
            )

        raise ValueError(f"Invalid path (neither file nor directory): {path}")

    # ─────────────────────────────────────────────────────────────────
    # Public API: Azure Blob ingestion
    # ─────────────────────────────────────────────────────────────────

    async def aingest_from_azure_blob(
        self,
        source: Any,  # AzureBlobDataSource
        container_name: str,
        blob_path: str | None = None,
        prefix: str | None = None,
        replace: bool = False,
        sync_hashes: bool = False,
    ) -> IngestionResult:
        """Ingest from Azure Blob Storage.

        Pre-parse deduplication: Files with identical content (by SHA256 hash)
        are skipped unless replace=True.

        Args:
            source: Azure Blob data source (from corprag.sourcing.azure_blob)
            container_name: Name of the Azure Blob container
            blob_path: Specific blob path to ingest (mutually exclusive with prefix)
            prefix: Prefix to filter blobs (mutually exclusive with blob_path)
            replace: If True, delete existing docs with same basename before ingesting
            sync_hashes: If True, sync hashes for existing processed documents
        """
        if blob_path and prefix:
            raise ValueError("blob_path and prefix are mutually exclusive")

        # Sync hashes for existing documents if requested
        if sync_hashes:
            await self._hash_index.sync_existing()

        artifacts_dir = self._get_artifacts_dir("azure_blobs", container_name)

        # Single blob
        if blob_path:
            basename = Path(blob_path).name

            # Download first (needed to compute hash)
            source_path = await self._download_blob_to_storage_async(
                source,
                container_name,
                blob_path,
            )

            # Check for duplicate AFTER download
            should_skip, content_hash, reason = await self._hash_index.should_skip_file(
                source_path, replace
            )

            if should_skip:
                logger.info(f"Skipped (dedup): {blob_path} - {reason}")
                return IngestionResult(
                    status="success",
                    processed=0,
                    skipped=1,
                    total_files=1,
                    source_type="azure_blobs",
                    container=container_name,
                    blob_path=blob_path,
                    skipped_files=[basename],
                )

            if replace:
                await self.adelete_files(filenames=[basename], delete_source=False)

            logger.info("Ingesting Azure blob: %s", blob_path)

            result = await self._ingest_single_file_with_policy(
                source_path, artifacts_dir, content_hash=content_hash
            )
            result.source_type = "azure_blobs"
            result.container = container_name
            result.blob_path = blob_path
            result.total_files = 1
            return result

        # Prefix batch
        if prefix is not None:
            blob_ids = await source.alist_documents(prefix=prefix)
            if not blob_ids:
                logger.warning("No blobs found in %s with prefix %s", container_name, prefix)
                return IngestionResult(
                    status="success",
                    processed=0,
                    total_files=0,
                    source_type="azure_blobs",
                    container=container_name,
                    prefix=prefix,
                )

            logger.info(
                "Preparing to ingest %d blobs from %s/%s (max %d concurrent)",
                len(blob_ids),
                container_name,
                prefix,
                self.max_concurrent,
            )

            # Phase 1: Download all blobs concurrently
            async def download_one(blob_id: str) -> Path | None:
                async with self._semaphore:
                    try:
                        return await self._download_blob_to_storage_async(
                            source,
                            container_name,
                            blob_id,
                        )
                    except Exception as exc:
                        logger.error("Failed to download blob %s: %s", blob_id, exc)
                        return None

            download_tasks = [download_one(bid) for bid in blob_ids]
            downloaded_paths = await asyncio.gather(*download_tasks)
            valid_paths = [p for p in downloaded_paths if p is not None]

            if not valid_paths:
                logger.warning(
                    "All downloads failed for prefix %s in container %s",
                    prefix,
                    container_name,
                )
                return IngestionResult(
                    status="success",
                    processed=0,
                    total_files=len(blob_ids),
                    source_type="azure_blobs",
                    container=container_name,
                    prefix=prefix,
                )

            # Phase 2: Check for duplicates and collect files to process
            files_to_process: list[tuple[Path, str]] = []
            skipped_count = 0
            skipped_files: list[str] = []

            for i, downloaded_path in enumerate(valid_paths):
                # Check for cancellation periodically
                if i % self._cancel_check_interval == 0:
                    await self._check_cancelled()

                (
                    should_skip,
                    content_hash,
                    reason,
                ) = await self._hash_index.should_skip_file(downloaded_path, replace)

                if should_skip:
                    skipped_count += 1
                    skipped_files.append(downloaded_path.name)
                    logger.info(f"Skipped (dedup): {downloaded_path.name}")
                    continue

                if replace:
                    await self.adelete_files(filenames=[downloaded_path.name], delete_source=False)

                files_to_process.append((downloaded_path, content_hash or ""))

            total_files = len(files_to_process) + skipped_count

            if not files_to_process:
                if skipped_count > 0:
                    logger.info(
                        f"No new blobs to ingest from {container_name}/{prefix} "
                        f"({skipped_count} skipped as duplicates)"
                    )
                return IngestionResult(
                    status="success",
                    processed=0,
                    skipped=skipped_count,
                    total_files=total_files,
                    source_type="azure_blobs",
                    container=container_name,
                    prefix=prefix,
                    skipped_files=skipped_files if skipped_files else None,
                )

            # Phase 3: Process downloaded files concurrently
            await self._check_cancelled()

            logger.info(
                "Processing %d blobs from %s/%s (%d skipped)",
                len(files_to_process),
                container_name,
                prefix,
                skipped_count,
            )

            ingest_tasks = [
                self._ingest_single_file_with_policy(
                    fp, artifacts_dir, content_hash=ch if ch else None
                )
                for fp, ch in files_to_process
            ]
            results = await asyncio.gather(*ingest_tasks, return_exceptions=True)

            success_count = sum(
                1 for r in results if isinstance(r, IngestionResult) and r.status == "success"
            )

            # Invalidate hash index cache after batch ingestion
            self._hash_index.invalidate()

            return IngestionResult(
                status="success",
                processed=success_count,
                skipped=skipped_count,
                total_files=total_files,
                source_type="azure_blobs",
                container=container_name,
                prefix=prefix,
                skipped_files=skipped_files if skipped_files else None,
            )

        raise ValueError("Must provide either blob_path or prefix for Azure Blob ingestion")

    # ─────────────────────────────────────────────────────────────────
    # Public API: Content list ingestion
    # ─────────────────────────────────────────────────────────────────

    async def aingest_content_list(
        self,
        content_list: list[dict[str, Any]],
        file_path: str = "content_list",
        display_stats: bool = True,
    ) -> IngestionResult:
        """Ingest an in-memory list of structured content with policy filtering."""
        if not content_list:
            logger.warning("Empty content list provided for ingestion")
            return IngestionResult(status="success", processed=0)

        # Generate doc_id from content
        from lightrag.utils import compute_mdhash_id

        content_str = str(content_list[:10])
        doc_id = compute_mdhash_id(content_str, prefix="doc-")

        # Apply policy
        result = self.policy.apply(content_list)

        if display_stats:
            logger.info(
                f"Policy filter [{file_path}]: "
                f"total={result.stats.total}, indexed={result.stats.indexed}, "
                f"dropped={result.stats.dropped_by_type} ({result.stats.drop_rate:.1f}%)"
            )

        if result.index_stream:
            await self.rag.insert_content_list(  # type: ignore[misc]
                content_list=result.index_stream,
                file_path=file_path,
                display_stats=display_stats,
            )

        return IngestionResult(
            status="success",
            processed=len(result.index_stream),
            doc_id=doc_id,
            source_path=file_path,
            stats=result.stats,
        )

    # ─────────────────────────────────────────────────────────────────
    # Public API: Snowflake ingestion
    # ─────────────────────────────────────────────────────────────────

    async def aingest_from_snowflake(
        self,
        query: str,  # noqa: ARG002
        table: str | None = None,
        content_columns: list[str] | None = None,  # noqa: ARG002
        metadata_columns: list[str] | None = None,  # noqa: ARG002
        **snowflake_kwargs: Any,
    ) -> IngestionResult:
        """Ingest structured data from Snowflake into RAG via content_list."""
        from corprag.sourcing.snowflake import SnowflakeDataSource

        config = self.config
        source = SnowflakeDataSource(
            account=snowflake_kwargs.get("account") or config.snowflake_account or "",
            user=snowflake_kwargs.get("user") or config.snowflake_user or "",
            password=snowflake_kwargs.get("password") or config.snowflake_password or "",
            warehouse=snowflake_kwargs.get("warehouse") or config.snowflake_warehouse,
            database=snowflake_kwargs.get("database") or config.snowflake_database,
            schema=snowflake_kwargs.get("schema") or config.snowflake_schema,
        )

        try:
            content_list: list[dict[str, Any]] = []

            for doc_id in await source.alist_documents():
                try:
                    content = await asyncio.to_thread(source.load_document, doc_id)
                    if isinstance(content, bytes):
                        content = content.decode("utf-8", errors="ignore")

                    content_list.append(
                        {
                            "type": "text",
                            "text": str(content),
                            "metadata": {
                                "doc_id": doc_id,
                                "table": table or "query",
                            },
                        }
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to load Snowflake document %s: %s",
                        doc_id,
                        exc,
                    )
                    continue

            if content_list:
                result = await self.aingest_content_list(
                    content_list=content_list,
                    file_path=f"snowflake://{table or 'query'}",
                    display_stats=True,
                )
                result.source_type = "snowflake"
                return result

            return IngestionResult(
                status="success",
                processed=0,
                source_type="snowflake",
            )
        finally:
            source.close()

    def _find_all_by_basename(self, basename: str) -> list[dict[str, str | None]]:
        """Find all matching entries by basename from kv_store_doc_status.

        Returns list of dicts with doc_id, file_path for each match.
        Exact name matches are returned first, then stem matches.
        """
        kv_path = self.config.working_dir_path / "kv_store_doc_status.json"
        if not kv_path.exists():
            return []

        try:
            data = json.loads(kv_path.read_text())
        except Exception as exc:
            logger.warning("Could not read doc status store: %s", exc)
            return []

        exact_matches: list[dict[str, str | None]] = []
        stem_matches: list[dict[str, str | None]] = []
        file_stem = Path(basename).stem

        for doc_id, meta in data.items():
            file_path = meta.get("file_path") or ""
            stored_name = Path(str(file_path)).name
            stored_stem = Path(str(file_path)).stem

            if stored_name == basename:
                exact_matches.append({"doc_id": doc_id, "file_path": file_path})
            elif stored_stem == file_stem:
                stem_matches.append({"doc_id": doc_id, "file_path": file_path})

        # Return exact matches first, then stem matches as fallback
        return exact_matches + stem_matches

    # ─────────────────────────────────────────────────────────────────
    # Public API: File management
    # ─────────────────────────────────────────────────────────────────

    def list_ingested_files(self) -> list[dict[str, Any]]:
        """List all ingested files from hash index."""
        return self._hash_index.list_all()

    async def adelete_files(
        self,
        *,
        file_paths: list[str] | None = None,
        filenames: list[str] | None = None,
        delete_source: bool = True,
    ) -> list[dict[str, Any]]:
        """Delete files from RAG storage.

        Finds doc_id(s) via hash index / KV doc_status, then delegates
        to LightRAG's adelete_by_doc_id for comprehensive cleanup across
        all storage layers.

        Args:
            file_paths: List of full file paths to delete
            filenames: List of filenames (basenames) to delete
            delete_source: Whether to delete source files (default True)

        Returns:
            List of deletion results with detailed cleanup status
        """
        if not file_paths and not filenames:
            return []

        results: list[dict[str, Any]] = []

        # Ensure LightRAG is initialized before deletion
        if hasattr(self.rag, "_ensure_lightrag_initialized"):
            await self.rag._ensure_lightrag_initialized()

        lightrag = getattr(self.rag, "lightrag", None)
        if not lightrag:
            logger.warning("LightRAG not initialized - some cleanup may be incomplete")

        # Collect all identifiers to process
        identifiers: list[str] = []
        if file_paths:
            identifiers.extend(file_paths)
        if filenames:
            identifiers.extend(filenames)

        for identifier in identifiers:
            # Phase 1: Multi-strategy context collection
            ctx = await collect_deletion_context(
                identifier=identifier,
                rag_working_dir=self.config.working_dir_path,
                hash_index=self._hash_index,
                lightrag=lightrag,
            )

            # Initialize result with context info
            deletion_result: dict[str, Any] = {
                "identifier": identifier,
                "doc_ids_found": list(ctx.doc_ids),
                "sources_used": ctx.sources_used,
                "cleanup_results": {},
                "status": "deleted",
            }

            # Check if we have any data to delete
            if not ctx.doc_ids:
                deletion_result["status"] = "not_found"
                deletion_result["file_path"] = identifier
                deletion_result["doc_id"] = None
                results.append(deletion_result)
                continue

            # Phase 2: Delete via LightRAG adelete_by_doc_id for each doc_id
            for doc_id in ctx.doc_ids:
                if lightrag and hasattr(lightrag, "adelete_by_doc_id"):
                    try:
                        result = await lightrag.adelete_by_doc_id(doc_id, delete_llm_cache=True)
                        deletion_result["cleanup_results"][f"lightrag_{doc_id}"] = (
                            result.status if hasattr(result, "status") else "completed"
                        )
                        logger.info(f"LightRAG deletion for {doc_id}: completed")
                    except Exception as exc:
                        logger.warning(f"LightRAG deletion failed for {doc_id}: {exc}")
                        deletion_result["cleanup_results"][f"lightrag_{doc_id}"] = f"error: {exc}"

            # Phase 3a: Remove from hash index
            for content_hash in ctx.content_hashes:
                if self._hash_index.remove(content_hash):
                    deletion_result["cleanup_results"]["hash_index"] = "removed"
                    logger.info(f"Removed hash index entry: {content_hash[:30]}...")

            # Phase 3b: Delete source files and artifacts
            if delete_source and ctx.file_paths:
                artifacts_dir = self.config.artifacts_dir
                source_deleted = False

                for fp in ctx.file_paths:
                    # Delete source file
                    resolved = self._resolve_source_file(fp)
                    if resolved and resolved.exists():
                        try:
                            resolved.unlink()
                            logger.info(f"Deleted source file: {resolved}")
                            source_deleted = True
                        except Exception as exc:
                            logger.warning(f"Failed to delete source file {resolved}: {exc}")

                    # Delete artifacts
                    rel = self._extract_relative_source_path(fp)
                    if rel:
                        rel_path = Path(rel)
                        artifact_base = artifacts_dir / rel_path.parent
                        file_stem = rel_path.stem

                        # Delete directory
                        artifact_dir = artifact_base / file_stem
                        if artifact_dir.exists() and artifact_dir.is_dir():
                            shutil.rmtree(artifact_dir)
                            logger.info(f"Deleted artifacts directory: {artifact_dir}")

                        # Delete artifact files
                        for artifact_file in artifact_base.glob(f"{file_stem}.*"):
                            if artifact_file.is_file():
                                artifact_file.unlink()
                                logger.info(f"Deleted artifact file: {artifact_file}")

                if source_deleted:
                    deletion_result["cleanup_results"]["source_files"] = "deleted"

            # Set file_path and doc_id for backward compatibility
            deletion_result["file_path"] = list(ctx.file_paths)[0] if ctx.file_paths else identifier
            deletion_result["doc_id"] = list(ctx.doc_ids)[0] if ctx.doc_ids else None

            results.append(deletion_result)

        return results


__all__ = [
    "IngestionPipeline",
    "IngestionResult",
    "IngestionCancelledError",
]
