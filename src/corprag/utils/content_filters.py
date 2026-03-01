# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Content filtering utilities for text processing.

Provides common patterns and functions for filtering metadata, processing
table content, and truncating text. Used by retrieval for snippet generation.
"""

from __future__ import annotations

import re

# Markers for image/table metadata lines that should be filtered
IMAGE_METADATA_MARKERS = (
    "Image Path:",
    "Caption:",
    "Table Analysis:",
    "Image Content Analysis:",
    "Visual Analysis:",
)

# Redundant placeholder values that should be filtered
REDUNDANT_PLACEHOLDERS = (
    "Captions: None",
    "Caption: None",
    "Footnotes: None",
    "Footnote: None",
)

# Regex pattern to match markdown table separator lines
TABLE_SEPARATOR_PATTERN = re.compile(r"^\|[\s:\-|]+\|?\s*$")


def is_table_separator_line(line: str) -> bool:
    """Check if a line is a markdown table separator (e.g., |---|---|)."""
    stripped = line.strip()
    if not stripped:
        return False
    return bool(TABLE_SEPARATOR_PATTERN.match(stripped))


def extract_table_row_text(line: str) -> str:
    """Extract meaningful text from a markdown table row.

    Converts '| Col1 | Col2 | Col3 |' -> 'Col1, Col2, Col3'
    """
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = [cell.strip() for cell in stripped.split("|")]
    cells = [cell for cell in cells if cell]
    return ", ".join(cells) if cells else ""


def filter_content_for_snippet(content: str, max_chars: int = 100) -> str:
    """Filter metadata and transform tables for compact snippet display.

    Removes:
    - Lines starting with image metadata markers
    - Markdown table separator lines

    Transforms:
    - Table data rows: extracts cell text, joins with commas
    """
    lines = content.split("\n")
    filtered = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(marker) for marker in IMAGE_METADATA_MARKERS):
            continue
        if is_table_separator_line(stripped):
            continue
        if "|" in stripped:
            extracted = extract_table_row_text(stripped)
            if extracted:
                filtered.append(extracted)
            continue
        filtered.append(stripped)

    result = " ".join(filtered).strip()
    result = re.sub(r"\s+", " ", result)

    if len(result) > max_chars:
        result = result[:max_chars] + "..."
    return result


__all__ = [
    "IMAGE_METADATA_MARKERS",
    "REDUNDANT_PLACEHOLDERS",
    "TABLE_SEPARATOR_PATTERN",
    "is_table_separator_line",
    "extract_table_row_text",
    "filter_content_for_snippet",
]
