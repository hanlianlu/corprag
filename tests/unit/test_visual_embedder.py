# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for VisualEmbedder (visual embedding via OpenAI-compatible API)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from dlightrag.unifiedrepresent.embedder import VisualEmbedder

DIM = 128


def _make_embedder(**overrides: object) -> VisualEmbedder:
    """Create a VisualEmbedder with sensible defaults."""
    defaults: dict[str, object] = {
        "model": "test-model",
        "base_url": "https://api.example.com/v1",
        "api_key": "sk-test",
        "dim": DIM,
        "batch_size": 4,
    }
    defaults.update(overrides)
    return VisualEmbedder(**defaults)  # type: ignore[arg-type]


def _tiny_image() -> Image.Image:
    """Return a 2x2 red PIL image for testing."""
    return Image.new("RGB", (2, 2), color="red")


def _mock_response(dim: int, n: int = 1) -> MagicMock:
    """Build a mock httpx response returning *n* embeddings of size *dim*."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"data": [{"embedding": [0.1] * dim} for _ in range(n)]}
    return resp


# ---------------------------------------------------------------------------
# TestVisualEmbedderInit
# ---------------------------------------------------------------------------


class TestVisualEmbedderInit:
    def test_stores_parameters(self) -> None:
        emb = _make_embedder()
        assert emb.model == "test-model"
        assert emb.api_key == "sk-test"
        assert emb.dim == DIM
        assert emb.batch_size == 4

    def test_strips_trailing_slash(self) -> None:
        emb = _make_embedder(base_url="https://api.example.com/v1/")
        assert emb.base_url == "https://api.example.com/v1"

    def test_strips_multiple_trailing_slashes(self) -> None:
        emb = _make_embedder(base_url="https://api.example.com/v1///")
        assert emb.base_url == "https://api.example.com/v1"

    def test_no_trailing_slash_unchanged(self) -> None:
        emb = _make_embedder(base_url="https://api.example.com/v1")
        assert emb.base_url == "https://api.example.com/v1"


# ---------------------------------------------------------------------------
# TestImageToBase64
# ---------------------------------------------------------------------------


class TestImageToBase64:
    def test_returns_data_uri(self) -> None:
        emb = _make_embedder()
        uri = emb._image_to_base64(_tiny_image())
        assert uri.startswith("data:image/png;base64,")

    def test_base64_is_decodable(self) -> None:
        import base64 as b64

        emb = _make_embedder()
        uri = emb._image_to_base64(_tiny_image())
        b64_part = uri.split(",", 1)[1]
        decoded = b64.b64decode(b64_part)
        # Should be valid PNG bytes (PNG magic number).
        assert decoded[:4] == b"\x89PNG"


# ---------------------------------------------------------------------------
# TestValidateDim
# ---------------------------------------------------------------------------


class TestValidateDim:
    def test_matching_dim_passes(self) -> None:
        emb = _make_embedder(dim=DIM)
        # Should not raise.
        emb._validate_dim([0.0] * DIM)

    def test_mismatching_dim_raises(self) -> None:
        emb = _make_embedder(dim=DIM)
        with pytest.raises(ValueError, match="Expected embedding dimension"):
            emb._validate_dim([0.0] * (DIM + 1))

    def test_error_message_contains_both_dims(self) -> None:
        emb = _make_embedder(dim=DIM)
        with pytest.raises(ValueError, match=rf"{DIM}.*{DIM + 5}"):
            emb._validate_dim([0.0] * (DIM + 5))


# ---------------------------------------------------------------------------
# TestEmbedPages
# ---------------------------------------------------------------------------


class TestEmbedPages:
    async def test_empty_list_returns_zero_rows(self) -> None:
        emb = _make_embedder()
        result = await emb.embed_pages([])
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, DIM)
        assert result.dtype == np.float32

    async def test_single_image(self) -> None:
        emb = _make_embedder()
        mock_resp = _mock_response(DIM, n=1)
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp

        with patch("httpx.AsyncClient") as MockClient:
            ctx = AsyncMock()
            ctx.__aenter__.return_value = mock_client
            MockClient.return_value = ctx

            result = await emb.embed_pages([_tiny_image()])

        assert result.shape == (1, DIM)
        assert result.dtype == np.float32

    async def test_batching_five_images_batch_size_two(self) -> None:
        emb = _make_embedder(batch_size=2)
        images = [_tiny_image() for _ in range(5)]

        mock_client = AsyncMock()

        # Batches: [2, 2, 1] -> 3 HTTP calls with matching response sizes.
        responses = [
            _mock_response(DIM, n=2),
            _mock_response(DIM, n=2),
            _mock_response(DIM, n=1),
        ]
        mock_client.post.side_effect = responses

        with patch("httpx.AsyncClient") as MockClient:
            ctx = AsyncMock()
            ctx.__aenter__.return_value = mock_client
            MockClient.return_value = ctx

            result = await emb.embed_pages(images)

        assert mock_client.post.call_count == 3
        assert result.shape == (5, DIM)

    async def test_result_dtype_is_float32(self) -> None:
        emb = _make_embedder()
        mock_resp = _mock_response(DIM, n=2)
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp

        with patch("httpx.AsyncClient") as MockClient:
            ctx = AsyncMock()
            ctx.__aenter__.return_value = mock_client
            MockClient.return_value = ctx

            result = await emb.embed_pages([_tiny_image(), _tiny_image()])

        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# TestEmbedQuery
# ---------------------------------------------------------------------------


class TestEmbedQuery:
    async def test_returns_1d_array(self) -> None:
        emb = _make_embedder()
        mock_resp = _mock_response(DIM, n=1)
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp

        with patch("httpx.AsyncClient") as MockClient:
            ctx = AsyncMock()
            ctx.__aenter__.return_value = mock_client
            MockClient.return_value = ctx

            result = await emb.embed_query("test query")

        assert result.shape == (DIM,)
        assert result.dtype == np.float32

    async def test_calls_embeddings_endpoint(self) -> None:
        emb = _make_embedder(base_url="https://api.example.com/v1")
        mock_resp = _mock_response(DIM, n=1)
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp

        with patch("httpx.AsyncClient") as MockClient:
            ctx = AsyncMock()
            ctx.__aenter__.return_value = mock_client
            MockClient.return_value = ctx

            await emb.embed_query("hello")

        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://api.example.com/v1/embeddings"

    async def test_dim_mismatch_raises(self) -> None:
        emb = _make_embedder(dim=DIM)
        # Return embedding with wrong dimension.
        mock_resp = _mock_response(DIM + 10, n=1)
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp

        with patch("httpx.AsyncClient") as MockClient:
            ctx = AsyncMock()
            ctx.__aenter__.return_value = mock_client
            MockClient.return_value = ctx

            with pytest.raises(ValueError, match="Expected embedding dimension"):
                await emb.embed_query("query")


# ---------------------------------------------------------------------------
# TestAuthHeaders
# ---------------------------------------------------------------------------


class TestAuthHeaders:
    def test_returns_bearer_token(self) -> None:
        emb = _make_embedder(api_key="sk-secret-key")
        headers = emb._auth_headers()
        assert headers == {"Authorization": "Bearer sk-secret-key"}

    def test_different_api_key(self) -> None:
        emb = _make_embedder(api_key="my-other-key")
        headers = emb._auth_headers()
        assert headers["Authorization"] == "Bearer my-other-key"
