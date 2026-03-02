# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for content filtering utilities."""

from __future__ import annotations

from corprag.utils.content_filters import (
    extract_table_row_text,
    filter_content_for_snippet,
    is_table_separator_line,
)

# ---------------------------------------------------------------------------
# is_table_separator_line
# ---------------------------------------------------------------------------


class TestIsTableSeparatorLine:
    def test_standard_separator(self) -> None:
        assert is_table_separator_line("|---|---|") is True

    def test_separator_with_colons(self) -> None:
        assert is_table_separator_line("|:---:|:---:|") is True

    def test_separator_with_spaces(self) -> None:
        assert is_table_separator_line("| --- | --- |") is True

    def test_missing_trailing_pipe(self) -> None:
        assert is_table_separator_line("|---|---") is True

    def test_text_with_pipes_is_not_separator(self) -> None:
        assert is_table_separator_line("| hello | world |") is False

    def test_empty_string(self) -> None:
        assert is_table_separator_line("") is False

    def test_whitespace_only(self) -> None:
        assert is_table_separator_line("   ") is False

    def test_dashes_without_pipes(self) -> None:
        assert is_table_separator_line("---") is False


# ---------------------------------------------------------------------------
# extract_table_row_text
# ---------------------------------------------------------------------------


class TestExtractTableRowText:
    def test_standard_row(self) -> None:
        assert extract_table_row_text("| A | B | C |") == "A, B, C"

    def test_empty_cells_filtered(self) -> None:
        assert extract_table_row_text("| | data | |") == "data"

    def test_all_empty_cells(self) -> None:
        assert extract_table_row_text("| | |") == ""

    def test_no_pipes(self) -> None:
        assert extract_table_row_text("plain text") == "plain text"

    def test_single_cell(self) -> None:
        assert extract_table_row_text("| only |") == "only"

    def test_extra_whitespace_trimmed(self) -> None:
        assert extract_table_row_text("|  A  |  B  |") == "A, B"


# ---------------------------------------------------------------------------
# filter_content_for_snippet
# ---------------------------------------------------------------------------


class TestFilterContentForSnippet:
    def test_empty_content(self) -> None:
        assert filter_content_for_snippet("") == ""

    def test_plain_text_passthrough(self) -> None:
        assert filter_content_for_snippet("Hello world") == "Hello world"

    def test_filters_image_metadata(self) -> None:
        content = "Some text\nImage Path: /foo/bar.png\nMore text"
        result = filter_content_for_snippet(content, max_chars=200)
        assert "Image Path" not in result
        assert "Some text" in result
        assert "More text" in result

    def test_filters_table_separator_extracts_rows(self) -> None:
        content = "| Name | Age |\n|---|---|\n| Alice | 30 |"
        result = filter_content_for_snippet(content, max_chars=200)
        assert "---|" not in result
        assert "Alice" in result
        assert "Name" in result

    def test_mixed_content(self) -> None:
        content = (
            "Introduction\n"
            "Image Path: /img.png\n"
            "Caption: A photo\n"
            "| Col1 | Col2 |\n"
            "|---|---|\n"
            "| val1 | val2 |\n"
            "Conclusion"
        )
        result = filter_content_for_snippet(content, max_chars=500)
        assert "Introduction" in result
        assert "Conclusion" in result
        assert "val1" in result
        assert "Image Path" not in result
        assert "Caption:" not in result
        assert "---|" not in result

    def test_truncation_at_max_chars(self) -> None:
        content = "A" * 150
        result = filter_content_for_snippet(content, max_chars=100)
        assert len(result) == 103  # 100 + "..."
        assert result.endswith("...")

    def test_no_truncation_at_exact_max(self) -> None:
        content = "A" * 100
        result = filter_content_for_snippet(content, max_chars=100)
        assert result == "A" * 100
        assert "..." not in result

    def test_whitespace_collapse(self) -> None:
        content = "hello    world\n\n\nfoo   bar"
        result = filter_content_for_snippet(content, max_chars=200)
        assert "  " not in result
        assert "hello world" in result

    def test_all_lines_filtered(self) -> None:
        content = "Image Path: /a.png\nCaption: foo\n|---|---|"
        result = filter_content_for_snippet(content, max_chars=200)
        assert result == ""
