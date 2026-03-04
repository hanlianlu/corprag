# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""LibreOffice-based Office document to PDF converter.

Converts Excel files to PDF during ingestion with:
- Landscape orientation
- Fit-to-width scaling (all columns visible on one page)
- Accurate numeric preservation (percentages, decimals, dates)

Strategy: Excel -> ODS (modify styles.xml only) -> PDF
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dlightrag.config import DlightragConfig

logger = logging.getLogger(__name__)

EXCEL_EXTENSIONS = {".xls", ".xlsx"}
SKIP_CONVERSION_EXTENSIONS = {".csv"}


class OfficeConverterError(Exception):
    """Exception raised when office document conversion fails."""


class LibreOfficeConverter:
    """Converts Office documents (Excel) to PDF using LibreOffice.

    Runs during the ingestion copy stage, transforming Excel files to PDFs
    with preserved layout before RAGAnything processes them.
    """

    def __init__(self, config: DlightragConfig) -> None:
        self.config = config
        self.timeout = config.libreoffice_timeout
        self.pdf_quality = config.libreoffice_pdf_quality
        self.delete_original = config.excel_pdf_delete_original
        self.rag_storage_path = Path(config.working_dir).resolve()

    def _is_safe_to_delete(self, file_path: Path) -> bool:
        """Check if file is safe to delete (must be within storage dir)."""
        try:
            file_path.relative_to(self.rag_storage_path)
            return True
        except ValueError:
            return False

    def should_convert(self, file_path: Path) -> bool:
        """Check if file should be converted to PDF."""
        if not self.config.excel_auto_convert_to_pdf:
            return False
        if self.config.parser == "docling":
            return False
        suffix = file_path.suffix.lower()
        if suffix in SKIP_CONVERSION_EXTENSIONS:
            return False
        return suffix in EXCEL_EXTENSIONS

    def convert_to_pdf(self, source_path: Path, output_dir: Path) -> Path:
        """Convert Excel file to PDF using LibreOffice.

        1. Converts Excel -> ODS -> PDF with landscape + fit-to-width
        2. Optionally deletes the copied Excel file (NOT the original source)
        3. Returns the path to the generated PDF
        """
        if not source_path.exists():
            raise OfficeConverterError(f"Source file not found: {source_path}")
        if not self.should_convert(source_path):
            raise OfficeConverterError(f"File type {source_path.suffix} should not be converted")

        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_name = source_path.stem + ".pdf"
        expected_pdf_path = output_dir / pdf_name

        if expected_pdf_path.exists():
            counter = 1
            while expected_pdf_path.exists():
                pdf_name = f"{source_path.stem}_{counter}.pdf"
                expected_pdf_path = output_dir / pdf_name
                counter += 1

        logger.info(f"Converting {source_path.name} to PDF using LibreOffice")

        try:
            final_pdf_path = self._convert_excel_to_pdf(source_path, output_dir, expected_pdf_path)
            logger.info(f"Successfully converted {source_path.name} -> {final_pdf_path.name}")

            if self.delete_original and source_path.exists():
                try:
                    source_resolved = source_path.resolve()
                    if self._is_safe_to_delete(source_resolved):
                        source_path.unlink()
                    else:
                        logger.warning(
                            f"Skipping deletion of {source_path} - not within storage (safety check)"
                        )
                except Exception as e:
                    logger.warning(f"Failed to delete copied Excel file {source_path.name}: {e}")

            return final_pdf_path

        except subprocess.TimeoutExpired as e:
            raise OfficeConverterError(
                f"LibreOffice conversion timed out after {self.timeout}s"
            ) from e
        except FileNotFoundError as e:
            raise OfficeConverterError(
                "LibreOffice not found. Install: apt-get install libreoffice"
            ) from e
        except OfficeConverterError:
            raise
        except Exception as e:
            raise OfficeConverterError(f"Unexpected conversion error: {e}") from e

    def _convert_excel_to_pdf(
        self, source_path: Path, output_dir: Path, expected_pdf_path: Path
    ) -> Path:
        """Convert Excel to PDF via ODS intermediate with page setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Step 1: Excel -> ODS
            ods_path = temp_path / (source_path.stem + ".ods")
            result = subprocess.run(
                [
                    "libreoffice",
                    "--headless",
                    "--convert-to",
                    "ods",
                    "--outdir",
                    str(temp_path),
                    str(source_path),
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                encoding="utf-8",
                errors="ignore",
            )
            if result.returncode != 0 or not ods_path.exists():
                raise OfficeConverterError(f"Excel->ODS conversion failed: {result.stderr}")

            # Step 2: Modify ODS page setup
            self._set_ods_landscape_fit(ods_path)

            # Step 3: ODS -> PDF
            result = subprocess.run(
                [
                    "libreoffice",
                    "--headless",
                    "--convert-to",
                    "pdf:calc_pdf_Export",
                    "--outdir",
                    str(output_dir),
                    str(ods_path),
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                encoding="utf-8",
                errors="ignore",
            )
            if result.returncode != 0:
                raise OfficeConverterError(f"ODS->PDF conversion failed: {result.stderr}")

            pdf_output = output_dir / (ods_path.stem + ".pdf")
            if not pdf_output.exists():
                raise OfficeConverterError("PDF file not created")

            if pdf_output != expected_pdf_path:
                pdf_output.rename(expected_pdf_path)
                pdf_output = expected_pdf_path

            return pdf_output

    def _set_ods_landscape_fit(self, ods_path: Path) -> None:
        """Modify ODS XML to set landscape orientation and fit-to-width scaling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            extract_dir = temp_path / "ods_content"

            with zipfile.ZipFile(ods_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            styles_xml = extract_dir / "styles.xml"
            if not styles_xml.exists():
                logger.warning("styles.xml not found in ODS, skipping page setup")
                return

            ET.register_namespace("", "urn:oasis:names:tc:opendocument:xmlns:office:1.0")
            ET.register_namespace("style", "urn:oasis:names:tc:opendocument:xmlns:style:1.0")
            ET.register_namespace(
                "fo", "urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"
            )
            ET.register_namespace("table", "urn:oasis:names:tc:opendocument:xmlns:table:1.0")

            tree = ET.parse(styles_xml)
            root = tree.getroot()

            ns = {
                "style": "urn:oasis:names:tc:opendocument:xmlns:style:1.0",
                "fo": "urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0",
            }

            for page_layout in root.findall(".//style:page-layout", ns):
                props = page_layout.find("style:page-layout-properties", ns)
                if props is not None:
                    props.set(
                        "{urn:oasis:names:tc:opendocument:xmlns:style:1.0}print-orientation",
                        "landscape",
                    )
                    props.set(
                        "{urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0}page-width",
                        "29.7cm",
                    )
                    props.set(
                        "{urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0}page-height",
                        "21cm",
                    )
                    props.set(
                        "{urn:oasis:names:tc:opendocument:xmlns:style:1.0}scale-to-pages", "1"
                    )
                    props.set("{urn:oasis:names:tc:opendocument:xmlns:style:1.0}scale-to-X", "1")
                    props.set("{urn:oasis:names:tc:opendocument:xmlns:style:1.0}scale-to-Y", "0")

            tree.write(styles_xml, encoding="utf-8", xml_declaration=True)
            self._repack_ods(ods_path, extract_dir)

    def _repack_ods(self, ods_path: Path, extract_dir: Path) -> None:
        """Repack ODS ZIP file with proper structure (mimetype first, uncompressed)."""
        ods_path.unlink()

        with zipfile.ZipFile(ods_path, "w") as zip_ref:
            mimetype_file = extract_dir / "mimetype"
            if mimetype_file.exists():
                zip_ref.write(mimetype_file, "mimetype", compress_type=zipfile.ZIP_STORED)

            for file_path in sorted(extract_dir.rglob("*")):
                if file_path.is_file():
                    arcname = str(file_path.relative_to(extract_dir))
                    if arcname == "mimetype":
                        continue
                    zip_ref.write(file_path, arcname, compress_type=zipfile.ZIP_DEFLATED)

    def convert_bytes_to_pdf(
        self,
        file_data: bytes,
        mime_type: str,
        *,
        apply_page_setup: bool = False,
    ) -> bytes | None:
        """Convert Office document bytes to PDF bytes (in-memory)."""
        ext_map = {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "application/msword": ".doc",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
            "application/vnd.ms-powerpoint": ".ppt",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
            "application/vnd.ms-excel": ".xls",
        }
        ext = ext_map.get(mime_type, ".docx")
        is_excel = ext in {".xlsx", ".xls"}

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                input_file = tmp_path / f"input{ext}"
                input_file.write_bytes(file_data)

                if is_excel and apply_page_setup:
                    pdf_path = self._convert_excel_to_pdf(
                        input_file, tmp_path, tmp_path / "output.pdf"
                    )
                else:
                    result = subprocess.run(
                        [
                            "libreoffice",
                            "--headless",
                            "--convert-to",
                            "pdf",
                            "--outdir",
                            str(tmp_path),
                            str(input_file),
                        ],
                        capture_output=True,
                        timeout=self.timeout,
                        encoding="utf-8",
                        errors="ignore",
                    )
                    pdf_path = tmp_path / "input.pdf"
                    if result.returncode != 0 or not pdf_path.exists():
                        logger.error(f"LibreOffice conversion failed: {result.stderr}")
                        return None

                return pdf_path.read_bytes() if pdf_path.exists() else None

        except subprocess.TimeoutExpired:
            logger.error(f"LibreOffice conversion timed out after {self.timeout}s")
            return None
        except FileNotFoundError:
            logger.error("LibreOffice not found. Please install libreoffice.")
            return None
        except Exception as e:
            logger.error(f"Office to PDF conversion failed: {e}")
            return None

    def convert_with_fallback(self, source_path: Path, output_dir: Path) -> Path:
        """Convert with fallback: return original if conversion fails."""
        try:
            return self.convert_to_pdf(source_path, output_dir)
        except (OfficeConverterError, Exception) as e:
            logger.warning(f"Conversion failed for {source_path.name}, using original: {e}")
            return source_path


def create_converter(config: DlightragConfig | None = None) -> LibreOfficeConverter:
    """Factory function to create LibreOffice converter instance."""
    if config is None:
        from dlightrag.config import get_config

        config = get_config()
    return LibreOfficeConverter(config)


def convert_office_bytes_to_pdf(
    file_data: bytes,
    mime_type: str,
    config: DlightragConfig | None = None,
) -> bytes | None:
    """Convenience function to convert Office document bytes to PDF."""
    converter = create_converter(config)
    return converter.convert_bytes_to_pdf(file_data, mime_type)


__all__ = [
    "LibreOfficeConverter",
    "OfficeConverterError",
    "create_converter",
    "convert_office_bytes_to_pdf",
    "EXCEL_EXTENSIONS",
    "SKIP_CONVERSION_EXTENSIONS",
]
