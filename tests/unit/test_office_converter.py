# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for LibreOffice converter."""

from __future__ import annotations

from pathlib import Path

from corprag.config import CorpragConfig
from corprag.converters.office import (
    EXCEL_EXTENSIONS,
    SKIP_CONVERSION_EXTENSIONS,
    LibreOfficeConverter,
)


class TestLibreOfficeConverter:
    """Test converter configuration and should_convert logic."""

    def _make_converter(self, test_config: CorpragConfig) -> LibreOfficeConverter:
        return LibreOfficeConverter(test_config)

    def test_should_convert_xlsx(self, test_config: CorpragConfig) -> None:
        """Test that .xlsx triggers conversion."""
        converter = self._make_converter(test_config)
        assert converter.should_convert(Path("test.xlsx"))
        assert converter.should_convert(Path("test.xls"))

    def test_should_not_convert_pdf(self, test_config: CorpragConfig) -> None:
        """Test that .pdf does not trigger conversion."""
        converter = self._make_converter(test_config)
        assert not converter.should_convert(Path("test.pdf"))
        assert not converter.should_convert(Path("test.docx"))

    def test_should_skip_csv(self, test_config: CorpragConfig) -> None:
        """Test that .csv is skipped."""
        converter = self._make_converter(test_config)
        assert not converter.should_convert(Path("test.csv"))

    def test_respects_config_flag(self, tmp_path: Path) -> None:
        """Test that excel_auto_convert_to_pdf=False disables conversion."""
        config = CorpragConfig(  # type: ignore[call-arg]
            working_dir=str(tmp_path),
            openai_api_key="test-key",
            excel_auto_convert_to_pdf=False,
            kv_storage="JsonKVStorage",
            doc_status_storage="JsonDocStatusStorage",
            vector_storage="NanoVectorDBStorage",
            graph_storage="NetworkXStorage",
        )
        converter = LibreOfficeConverter(config)
        assert not converter.should_convert(Path("test.xlsx"))

    def test_docling_parser_skips_conversion(self, tmp_path: Path) -> None:
        """Test that docling parser disables Excel conversion."""
        config = CorpragConfig(  # type: ignore[call-arg]
            working_dir=str(tmp_path),
            openai_api_key="test-key",
            parser="docling",
            kv_storage="JsonKVStorage",
            doc_status_storage="JsonDocStatusStorage",
            vector_storage="NanoVectorDBStorage",
            graph_storage="NetworkXStorage",
        )
        converter = LibreOfficeConverter(config)
        assert not converter.should_convert(Path("test.xlsx"))

    def test_is_safe_to_delete(self, test_config: CorpragConfig) -> None:
        """Test safety check for file deletion."""
        converter = self._make_converter(test_config)

        # File within storage is safe
        safe_path = test_config.working_dir_path / "sources" / "test.xlsx"
        assert converter._is_safe_to_delete(safe_path)

        # File outside storage is not safe
        assert not converter._is_safe_to_delete(Path("/tmp/outside.xlsx"))

    def test_constants(self) -> None:
        """Test module constants."""
        assert ".xlsx" in EXCEL_EXTENSIONS
        assert ".xls" in EXCEL_EXTENSIONS
        assert ".csv" in SKIP_CONVERSION_EXTENSIONS
