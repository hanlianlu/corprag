# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Custom VLM prompts for RAGAnything ingestion.

Provides smart content type detection (table vs image) with differentiated
analysis strategies. Injects custom prompts into RAGAnything's global PROMPTS dict.

Key design decisions:
- VLM first classifies content type (table vs image) - don't trust parser labels
- For tables: Output Markdown format + key data summary in detailed_description
- For images: Standard visual description in detailed_description
- All content goes in detailed_description (only field extracted by RAGAnything)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

SMART_IMAGE_ANALYSIS_SYSTEM = (
    "You are an expert visual content analyst specializing in technical documents. "
    "First determine the content type, then provide comprehensive analysis. "
    "ACCURACY IS CRITICAL: Double-check all numerical values - read each digit carefully. "
    "Common OCR errors to watch for: 6↔8, 5↔6, 0↔6, 9↔0."
)

SMART_IMAGE_ANALYSIS_PROMPT = """Analyze this image and provide a JSON response.

**Step 1: Content Type Classification**
Examine the image and determine its PRIMARY content type:
- "table": Contains tabular data with rows/columns, spreadsheets, data grids
- "image": General photos, diagrams, charts, illustrations, screenshots

**Step 2: Provide Analysis in detailed_description**

CRITICAL: Put ALL content in the detailed_description field. Do NOT create extra fields.

If content_type is "table", your detailed_description MUST contain (in this order):
1. Brief intro: "Technical specification table with X rows × Y columns."
2. The COMPLETE table in Markdown format - include ALL data, not a subset:
   | Header1 | Header2 | Header3 |
   |---------|---------|---------|
   | Value1  | Value2  | Value3  |
   ACCURACY CHECK: Verify each numerical value carefully before writing.
3. Key insights: Summarize the most important data points in natural language.
   MUST include units for all measurements (e.g., "wheelbase: 2960 mm", "angle: 17.8°").

If content_type is "image", your detailed_description should include:
- Overall composition and layout
- All objects, people, text, and visual elements
- Relationships between elements
- Colors, lighting, and visual style
- Technical details if applicable (charts, diagrams)

Return ONLY this JSON structure (no extra fields):
{{
    "detailed_description": "<ALL analysis here - for tables: intro + full Markdown table + key insights>",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "<table or image>",
        "summary": "Concise summary (max 100 words)"
    }}
}}

Image Information:
- Path: {image_path}
- Captions: {captions}
- Footnotes: {footnotes}

Respond with valid JSON only. Do NOT add fields like "table_markdown" - everything goes in detailed_description."""

SMART_IMAGE_ANALYSIS_PROMPT_WITH_CONTEXT = """Analyze this image considering the surrounding context, and provide a JSON response.

**Step 1: Content Type Classification**
Examine the image and determine its PRIMARY content type:
- "table": Contains tabular data with rows/columns, spreadsheets, data grids
- "image": General photos, diagrams, charts, illustrations, screenshots

**Step 2: Provide Analysis in detailed_description**

CRITICAL: Put ALL content in the detailed_description field. Do NOT create extra fields.

If content_type is "table", your detailed_description MUST contain (in this order):
1. Brief intro: "Technical specification table with X rows × Y columns."
2. The COMPLETE table in Markdown format - include ALL data, not a subset:
   | Header1 | Header2 | Header3 |
   |---------|---------|---------|
   | Value1  | Value2  | Value3  |
   ACCURACY CHECK: Verify each numerical value carefully before writing.
3. Key insights: Summarize the most important data points and relationship to context.
   MUST include units for all measurements (e.g., "wheelbase: 2960 mm", "angle: 17.8°").

If content_type is "image", your detailed_description should include:
- Overall composition and layout
- All objects, people, text, and visual elements
- Relationships between elements
- Colors, lighting, and visual style
- How the image relates to surrounding context

Return ONLY this JSON structure (no extra fields):
{{
    "detailed_description": "<ALL analysis here - for tables: intro + full Markdown table + key insights>",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "<table or image>",
        "summary": "Concise summary including relationship to context (max 100 words)"
    }}
}}

Context from surrounding content:
{context}

Image Information:
- Path: {image_path}
- Captions: {captions}
- Footnotes: {footnotes}

Respond with valid JSON only. Do NOT add fields like "table_markdown" - everything goes in detailed_description."""


def inject_custom_prompts() -> None:
    """Inject custom prompts into RAGAnything's PROMPTS dict.

    Must be called BEFORE RAGAnything is instantiated, as processors
    read from PROMPTS during initialization.
    """
    from raganything.prompt import PROMPTS

    PROMPTS["IMAGE_ANALYSIS_SYSTEM"] = SMART_IMAGE_ANALYSIS_SYSTEM
    PROMPTS["vision_prompt"] = SMART_IMAGE_ANALYSIS_PROMPT
    PROMPTS["vision_prompt_with_context"] = SMART_IMAGE_ANALYSIS_PROMPT_WITH_CONTEXT

    logger.info("Injected custom VLM prompts into RAGAnything")


__all__ = [
    "inject_custom_prompts",
    "SMART_IMAGE_ANALYSIS_SYSTEM",
    "SMART_IMAGE_ANALYSIS_PROMPT",
    "SMART_IMAGE_ANALYSIS_PROMPT_WITH_CONTEXT",
]
