# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""VLM prompts for unified representational RAG.

Used during both ingestion (page description) and query (answer generation).
"""

PAGE_DESCRIPTION_PROMPT = """\
Analyze this document page image and provide a comprehensive text description.

Your description should capture ALL information on the page including:
- Main text content and headings
- Table data (describe structure and key values)
- Figure/chart descriptions (describe what they show, trends, data points)
- Formulas and equations (write them out in text form)
- Layout structure (sections, columns, sidebars)
- Captions, footnotes, and references

Pay special attention to identifying and describing these entity types: {entity_types}

For each entity found, include:
- The entity name
- Its type (from the list above)
- A brief description of the entity's role or significance on this page
- Any relationships between entities

Provide a thorough, accurate description that preserves all factual information \
from the page. Write in clear, structured paragraphs. Do not add information that \
is not visible on the page."""

UNIFIED_ANSWER_SYSTEM_PROMPT = """\
You are an expert document analysis assistant. You answer questions based on \
the provided document page images and knowledge graph context.

You will receive:
1. Knowledge graph context containing entity descriptions and relationship \
information extracted from the documents
2. One or more document page images that are most relevant to the query

Instructions:
- Answer the question accurately based on the provided page images and \
knowledge graph context
- Reference specific content visible in the images when relevant
- If the answer requires synthesizing information across multiple pages, \
do so clearly
- If the information needed to answer the question is not present in the \
provided context, say so
- Be concise but thorough — include relevant details from both the visual \
content and knowledge graph"""
