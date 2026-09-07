# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""DlightRAG offline vector storage rebuild command."""

import argparse
import asyncio
import logging
import os
from types import SimpleNamespace
from typing import Any, Literal

from lightrag.base import DocStatus
from lightrag.constants import DEFAULT_COSINE_THRESHOLD
from lightrag.kg import STORAGE_ENV_REQUIREMENTS
from lightrag.namespace import NameSpace
from lightrag.tools.rebuild_vdb import DEFAULT_BATCH_SIZE, RebuildTool
from lightrag.utils import get_env_value

from dlightrag.adapters.observability import LangfuseTelemetry
from dlightrag.adapters.postgres.corpus.corpus import (
    apply_lightrag_environment,
    verify_lightrag_storage_configuration,
)
from dlightrag.application.config import DlightragConfig, get_config, load_config, set_config
from dlightrag.application.settings import rag_settings
from dlightrag.engine.ai.embedding import create_embedding_model
from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.ai.telemetry import Telemetry
from dlightrag.engine.rag.corpus.ingestion.document_embedding import (
    build_document_embedder,
    resolve_direct_image_embedding_enabled,
)
from dlightrag.engine.rag.corpus.ingestion.engine import UnifiedIngestionEngine
from dlightrag.engine.rag.corpus.rebuild_bm25 import run_rebuild_bm25
from dlightrag.engine.rag.lightrag.models import build_lightrag_embedding
from dlightrag.engine.rag.lightrag.stores import LightRAGStores
from dlightrag.engine.rag.workspace.settings import RagSettings

logger = logging.getLogger(__name__)

RebuildTarget = Literal["check", "graph", "chunks", "all"]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for `dlightrag-rebuild-vdb`."""
    parser = argparse.ArgumentParser(
        description="Offline DlightRAG vector storage rebuild command",
        suggest_on_error=True,
    )
    parser.add_argument("--env-file", help="Path to .env configuration file")
    parser.add_argument(
        "--target",
        choices=("check", "graph", "chunks", "all"),
        default="check",
        help=(
            "What to run: check graph->VDB consistency, rebuild graph vectors, "
            "rebuild chunk vectors, or rebuild all vector storages."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm that DlightRAG and other writers are stopped for destructive rebuilds.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Source records per rebuild batch (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--no-restore-sidecar-alignment",
        action="store_false",
        dest="restore_sidecar_alignment",
        help="Skip DlightRAG direct image-vector restoration after chunks rebuild.",
    )
    parser.set_defaults(restore_sidecar_alignment=True)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate destructive rebuild flags."""
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.target != "check" and not args.yes:
        raise SystemExit("--yes is required for rebuild targets; stop DlightRAG first")


class DlightRAGRebuildTool(RebuildTool):
    """LightRAG rebuild tool configured from DlightRAG settings."""

    def __init__(self, config: DlightragConfig, *, embedding_func: Any) -> None:
        super().__init__()
        self.config = config
        self.embedding_func = embedding_func
        self.full_docs = None
        self.doc_status = None
        self.workspace = config.deployment.workspace

    def resolve_storage_names(self) -> dict[str, str]:
        return {
            "graph": self.config.storage.lightrag.graph_storage,
            "vector": self.config.storage.lightrag.vector_storage,
            "kv": self.config.storage.lightrag.kv_storage,
            "doc_status": self.config.storage.lightrag.doc_status_storage,
        }

    def build_embedding_func(self) -> Any:
        return self.embedding_func

    def build_global_config(self) -> dict[str, Any]:
        if not self.storage_names:
            self.storage_names = self.resolve_storage_names()
        global_config: dict[str, Any] = {
            "working_dir": self.config.deployment.working_dir,
            "kv_storage": self.storage_names["kv"],
            "vector_storage": self.storage_names["vector"],
            "graph_storage": self.storage_names["graph"],
            "embedding_batch_num": self.config.models.embedding.batch_size,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": get_env_value(
                    "COSINE_THRESHOLD",
                    DEFAULT_COSINE_THRESHOLD,
                    float,
                )
            },
            "embedding_func": self.embedding_func,
        }
        global_config["vector_db_storage_cls_kwargs"].update(
            self.config.storage.lightrag.vector_db_kwargs
        )

        max_token_size = getattr(self.embedding_func, "max_token_size", None)
        if max_token_size:
            try:
                from lightrag.utils import TiktokenTokenizer

                global_config["tokenizer"] = TiktokenTokenizer(
                    get_env_value("TIKTOKEN_MODEL_NAME", "gpt-4o-mini", str)
                )
                global_config["embedding_token_limit"] = max_token_size
            except Exception as exc:  # noqa: BLE001
                logger.warning("Tokenizer unavailable, skipping truncation: %s", exc)
        return global_config

    async def setup_storages(self) -> bool:
        """Instantiate LightRAG storages plus full_docs/doc_status for DlightRAG restore."""
        from lightrag.kg.factory import get_storage_class

        self.storage_names = self.resolve_storage_names()
        self.workspace = self.config.deployment.workspace
        apply_lightrag_environment(self.config)
        verify_lightrag_storage_configuration(self.config)

        print("\nChecking configuration...")
        for storage_name in set(self.storage_names.values()):
            required_vars = STORAGE_ENV_REQUIREMENTS.get(storage_name, [])
            missing_vars = [var for var in required_vars if var not in os.environ]
            if missing_vars:
                print(f"Warning: {storage_name} normally requires: {', '.join(missing_vars)}")

        self.global_config = self.build_global_config()
        graph_cls = get_storage_class(self.storage_names["graph"])
        vector_cls = get_storage_class(self.storage_names["vector"])
        kv_cls = get_storage_class(self.storage_names["kv"])
        doc_status_cls = get_storage_class(self.storage_names["doc_status"])

        self.graph = graph_cls(
            namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
            workspace=self.workspace,
            global_config=self.global_config,
            embedding_func=self.embedding_func,
        )
        self.entities_vdb = vector_cls(
            namespace=NameSpace.VECTOR_STORE_ENTITIES,
            workspace=self.workspace,
            global_config=self.global_config,
            embedding_func=self.embedding_func,
            meta_fields={"entity_name", "source_id", "content", "file_path"},
        )
        self.relationships_vdb = vector_cls(
            namespace=NameSpace.VECTOR_STORE_RELATIONSHIPS,
            workspace=self.workspace,
            global_config=self.global_config,
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id", "source_id", "content", "file_path"},
        )
        self.chunks_vdb = vector_cls(
            namespace=NameSpace.VECTOR_STORE_CHUNKS,
            workspace=self.workspace,
            global_config=self.global_config,
            embedding_func=self.embedding_func,
            meta_fields={"full_doc_id", "content", "file_path"},
        )
        self.text_chunks = kv_cls(
            namespace=NameSpace.KV_STORE_TEXT_CHUNKS,
            workspace=self.workspace,
            global_config=self.global_config,
            embedding_func=self.embedding_func,
        )
        self.full_docs = kv_cls(
            namespace=NameSpace.KV_STORE_FULL_DOCS,
            workspace=self.workspace,
            global_config=self.global_config,
            embedding_func=self.embedding_func,
        )
        self.doc_status = doc_status_cls(
            namespace=NameSpace.DOC_STATUS,
            workspace=self.workspace,
            global_config=self.global_config,
            embedding_func=None,
        )

        print("\nInitializing storages...")
        try:
            for storage in self.all_storages():
                await storage.initialize()
        except Exception as exc:  # noqa: BLE001
            print(f"Storage initialization failed: {exc}")
            return False

        print(f"- Graph Storage:  {self.storage_names['graph']}")
        print(f"- Vector Storage: {self.storage_names['vector']}")
        print(f"- KV Storage:     {self.storage_names['kv']}")
        print(f"- Doc Status:     {self.storage_names['doc_status']}")
        print(f"- Workspace:      {self.workspace}")
        print(f"- Working Dir:    {self.global_config['working_dir']}")
        print("- Connection Status: Success")
        return True

    def all_storages(self) -> list[Any]:
        return [
            self.graph,
            self.entities_vdb,
            self.relationships_vdb,
            self.chunks_vdb,
            self.text_chunks,
            self.full_docs,
            self.doc_status,
        ]


async def restore_sidecar_image_vectors(
    *,
    workspace_id: str,
    settings: RagSettings,
    lightrag: Any,
    stores: Any,
    multimodal_embedder: Any,
    telemetry: Telemetry,
) -> dict[str, int]:
    """Restore DlightRAG direct image vectors after a chunks VDB rebuild."""
    document_embedder = build_document_embedder(
        settings,
        multimodal_embedder,
        image_enabled=True,
    )
    engine = UnifiedIngestionEngine(
        lightrag=lightrag,
        stores=stores,
        metadata_index=None,
        document_embedder=document_embedder,
        workspace=workspace_id,
        parser_rules=settings.parser_rules,
        chunk_options={},
        telemetry=telemetry,
    )

    stats = {"processed_docs": 0, "skipped_docs": 0}
    async for processed in stores.iter_doc_status_pages((DocStatus.PROCESSED,)):
        doc_ids = list(processed)
        full_statuses = await stores.get_full_doc_statuses(doc_ids)
        missing = set(doc_ids).difference(full_statuses)
        if missing:
            raise RuntimeError(
                f"document-status rows disappeared during sidecar restore: {sorted(missing)}"
            )
        for doc_id in doc_ids:
            chunks = _status_chunks(full_statuses[doc_id])
            if not chunks:
                stats["skipped_docs"] += 1
                continue
            full_doc = await stores.get_full_doc(doc_id)
            sidecar_location = _mapping_get(full_doc, "sidecar_location")
            if not sidecar_location:
                stats["skipped_docs"] += 1
                continue
            await engine._overwrite_sidecar_image_vectors(
                doc_id=doc_id,
                sidecar_location=str(sidecar_location),
                chunk_ids=set(chunks),
            )
            stats["processed_docs"] += 1
    return stats


async def run_rebuild(
    *,
    config: DlightragConfig | None = None,
    target: RebuildTarget = "check",
    assume_yes: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    restore_sidecar_alignment: bool = True,
) -> int:
    """Run a non-interactive DlightRAG rebuild target."""
    if target != "check" and not assume_yes:
        raise SystemExit("--yes is required for rebuild targets; stop DlightRAG first")

    resolved_config = config or get_config()
    if (
        restore_sidecar_alignment
        and target in {"chunks", "all"}
        and resolved_config.storage.lightrag.vector_storage != "PGVectorStorage"
    ):
        raise ValueError(
            "sidecar fused-vector restoration requires PGVectorStorage; "
            "use --no-restore-sidecar-alignment for MilvusVectorDBStorage"
        )
    resolved_embedding = resolved_config.models.embedding
    model_scheduler = ModelScheduler(max_concurrency=resolved_config.models.max_concurrency)
    multimodal_embedder = create_embedding_model(
        resolved_embedding,
        scheduler=model_scheduler,
        telemetry=LangfuseTelemetry(),
    )
    embedding_func = build_lightrag_embedding(resolved_embedding, multimodal_embedder)

    tool = DlightRAGRebuildTool(resolved_config, embedding_func=embedding_func)
    tool.batch_size = batch_size
    exit_code = 0

    from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

    initialize_share_data(workers=1)
    try:
        if not await tool.setup_storages():
            return 1
        if target == "check":
            await tool.run_check()
            return 0

        all_stats: list[dict[str, Any]] = []
        if target in {"graph", "all"}:
            graph_stats = await tool.run_rebuild_entities_relations()
            all_stats.extend(graph_stats or [])
        if target in {"chunks", "all"}:
            chunk_stats = await tool.run_rebuild_chunks()
            all_stats.extend(chunk_stats or [])

        if all_stats and tool.report_rebuild(all_stats):
            exit_code = 1

        if exit_code == 0 and target in {"chunks", "all"}:
            if resolved_config.corpus.retrieval.bm25_enabled:
                bm25_stats = await run_rebuild_bm25(
                    config=resolved_config,
                    assume_yes=True,
                    batch_size=batch_size,
                )
                print(
                    "BM25 language labels: "
                    f"{bm25_stats['processed_chunks']} scanned, "
                    f"{bm25_stats['updated_chunks']} updated"
                )

            if restore_sidecar_alignment:
                lightrag_surface = _lightrag_surface(tool)
                from dlightrag.adapters.postgres.corpus.corpus_chunks import PGCorpusChunkStore
                from dlightrag.adapters.postgres.corpus.lightrag_contract import (
                    PGLightRAGContractGuard,
                )

                PGLightRAGContractGuard(lightrag_surface).verify_surface()
                stores = LightRAGStores(
                    lightrag_surface,
                    chunk_store=PGCorpusChunkStore(lightrag_surface),
                )
                direct_enabled = await resolve_direct_image_embedding_enabled(
                    multimodal_embedder,
                    startup_probe=resolved_embedding.startup_probe,
                    require_image_support=resolved_embedding.input_modality == "multimodal",
                )
                if direct_enabled:
                    stats = await restore_sidecar_image_vectors(
                        workspace_id=resolved_config.deployment.workspace,
                        settings=rag_settings(resolved_config),
                        lightrag=lightrag_surface,
                        stores=stores,
                        multimodal_embedder=multimodal_embedder,
                        telemetry=LangfuseTelemetry(),
                    )
                    print(
                        "Sidecar fused visual-vector alignment: "
                        f"{stats['processed_docs']} processed, {stats['skipped_docs']} skipped"
                    )
    finally:
        for storage in tool.all_storages():
            if storage is not None:
                try:
                    await storage.finalize()
                except Exception:  # noqa: BLE001
                    logger.warning("Failed to finalize rebuild storage", exc_info=True)
        try:
            finalize_share_data()
        except Exception:  # noqa: BLE001
            logger.warning("Failed to finalize LightRAG shared storage", exc_info=True)
        await multimodal_embedder.aclose()
    return exit_code


def main() -> None:
    """Entry point for dlightrag-rebuild-vdb."""
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    config = load_config(args.env_file) if args.env_file else get_config()
    set_config(config)
    logging.basicConfig(
        level=getattr(logging, config.observability.log_level.upper(), logging.INFO)
    )

    exit_code = asyncio.run(
        run_rebuild(
            config=config,
            target=args.target,
            assume_yes=args.yes,
            batch_size=args.batch_size,
            restore_sidecar_alignment=args.restore_sidecar_alignment,
        )
    )
    if exit_code:
        raise SystemExit(exit_code)


def _lightrag_surface(tool: DlightRAGRebuildTool) -> Any:
    return SimpleNamespace(
        chunks_vdb=tool.chunks_vdb,
        text_chunks=tool.text_chunks,
        full_docs=tool.full_docs,
        doc_status=tool.doc_status,
        entities_vdb=tool.entities_vdb,
        relationships_vdb=tool.relationships_vdb,
        chunk_entity_relation_graph=tool.graph,
        full_entities=object(),
        full_relations=object(),
        entity_chunks=object(),
        relation_chunks=object(),
        llm_response_cache=object(),
    )


def _mapping_get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _status_chunks(status_info: Any) -> list[str]:
    raw = _mapping_get(status_info, "chunks_list") or []
    return [str(chunk_id) for chunk_id in raw if chunk_id]


__all__ = [
    "DlightRAGRebuildTool",
    "build_parser",
    "main",
    "restore_sidecar_image_vectors",
    "run_rebuild",
    "validate_args",
]
