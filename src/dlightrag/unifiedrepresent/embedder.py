# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Visual embedding for unified representational RAG.

Embeds page images into visual vectors via OpenAI-compatible multimodal
embedding API (e.g., qwen3-vl-embedding).
"""

from __future__ import annotations

import base64
import io
import logging

import httpx
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class VisualEmbedder:
    """Embed page images (and text queries) via an OpenAI-compatible multimodal embedding API."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        dim: int,
        batch_size: int = 4,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.dim = dim
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def embed_pages(self, images: list[Image.Image]) -> np.ndarray:
        """Embed a list of PIL images into visual vectors.

        Returns:
            np.ndarray of shape ``(len(images), dim)`` with dtype float32.
        """
        if not images:
            return np.empty((0, self.dim), dtype=np.float32)

        all_embeddings: list[list[float]] = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            embeddings = await self._embed_image_batch(batch)
            all_embeddings.extend(embeddings)

        return np.asarray(all_embeddings, dtype=np.float32)

    async def embed_query(self, query: str) -> np.ndarray:
        """Embed a text query into the same vector space as images.

        Returns:
            np.ndarray of shape ``(dim,)`` with dtype float32.
        """
        url = f"{self.base_url}/embeddings"
        payload = {
            "model": self.model,
            "input": query,
            "encoding_format": "float",
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                url,
                json=payload,
                headers=self._auth_headers(),
            )
            resp.raise_for_status()

        data = resp.json()
        embedding = data["data"][0]["embedding"]
        self._validate_dim(embedding)
        return np.asarray(embedding, dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert a PIL Image to a ``data:image/png;base64,...`` URI string."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    async def _embed_image_batch(
        self, images: list[Image.Image]
    ) -> list[list[float]]:
        """Call the embedding endpoint for a single batch of images."""
        url = f"{self.base_url}/embeddings"
        input_items = [
            {
                "type": "image_url",
                "image_url": {"url": self._image_to_base64(img)},
            }
            for img in images
        ]
        payload = {
            "model": self.model,
            "input": input_items,
            "encoding_format": "float",
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                url,
                json=payload,
                headers=self._auth_headers(),
            )
            resp.raise_for_status()

        data = resp.json()
        embeddings: list[list[float]] = []
        for item in data["data"]:
            emb = item["embedding"]
            self._validate_dim(emb)
            embeddings.append(emb)
        return embeddings

    def _auth_headers(self) -> dict[str, str]:
        """Return authorization headers for the API request."""
        return {"Authorization": f"Bearer {self.api_key}"}

    def _validate_dim(self, embedding: list[float]) -> None:
        """Raise if the returned embedding dimension does not match ``self.dim``."""
        if len(embedding) != self.dim:
            msg = (
                f"Expected embedding dimension {self.dim}, "
                f"got {len(embedding)}"
            )
            raise ValueError(msg)
