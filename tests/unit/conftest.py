# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unit-test fixtures — isolate from the operator's .env and config.yaml.

Both are deployment inputs, not product contracts: a unit test that reads them
asserts whatever this checkout happens to be tuned to, so retuning config.yaml
breaks CI. Tests that mean to exercise a YAML config build their own file.
"""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dlightrag.application import config as config_module
from dlightrag.application.config import DlightragConfig
from dlightrag.engine.ai.capacity import CONTEXT_POLICY_REVISION, ModelProfile
from dlightrag.engine.ai.catalog import current_model_catalog_revision
from dlightrag.engine.ai.fingerprints import ModelFingerprint
from dlightrag.engine.ai.media import MODEL_IMAGE_MAX_PIXELS
from dlightrag.engine.answer.capabilities import AnswerCapabilities
from dlightrag.engine.answer.execution.input import (
    AnswerRunInput,
    AnswerRunRequest,
    PinnedModelProfile,
)
from dlightrag.engine.answer.image_capability import AnswerImageCapability
from dlightrag.engine.answer.images import AnswerImagePolicy
from dlightrag.engine.answer.resources.models import ResourceInput

_REPO_CONFIG_YAML = Path(__file__).resolve().parents[2] / "config.yaml"
# Bound before the fixture patches the name, otherwise the wrapper recurses.
_FIND_YAML_CONFIG = config_module._find_yaml_config


def _yaml_config_ignoring_repo_file() -> Path | None:
    """Resolve config.yaml as production does, minus this checkout's own file."""
    found = _FIND_YAML_CONFIG()
    if found is not None and found.resolve() == _REPO_CONFIG_YAML:
        return None
    return found


def answer_image_policy(**overrides: int) -> AnswerImagePolicy:
    """Shipped answer transport policy for tests; images off unless opted in."""
    fields: dict[str, int] = {
        "max_images": 0,
        "max_total_bytes": 24_000_000,
        "max_bytes_per_image": 3_000_000,
        "max_pixels": MODEL_IMAGE_MAX_PIXELS,
        "max_px": 1536,
        "min_px": 1024,
        "quality": 89,
        "min_quality": 79,
    }
    return AnswerImagePolicy(**(fields | overrides))


def answer_model_profile(**overrides: int | bool | None) -> ModelProfile:
    """Resolved answer-model facts for tests that do not exercise the catalog."""
    fields: dict[str, int | bool | None] = {
        "context_window_tokens": 1_000_000,
        "max_input_tokens": None,
        "max_output_tokens": 128_000,
        "supports_images": True,
    }
    return ModelProfile(**(fields | overrides))  # type: ignore[arg-type]


def answer_capability_view(
    answer: AnswerImageCapability | None = None,
) -> SimpleNamespace:
    """Read-only capability-view double for transport tests."""
    snapshot = AnswerCapabilities(answer=answer, vlm_status="unknown")
    return SimpleNamespace(read=AsyncMock(return_value=snapshot))


async def prepare_test_answer_run_input(
    request: AnswerRunRequest,
    *,
    resources: list[ResourceInput] | None,  # noqa: ARG001
    idempotency_fingerprint: str,
) -> AnswerRunInput:
    """Pin one normalized request for tests that do not exercise model resolution."""
    return AnswerRunInput(
        query=request.query,
        workspaces=request.workspaces,
        history=request.history,
        retrieval=request.retrieval,
        filters=request.filters,
        semantic_highlights=request.semantic_highlights,
        links=request.links,
        attachments=request.attachments,
        history_attachments=request.history_attachments,
        pinned_models=(
            PinnedModelProfile(
                role="query",
                fingerprint=ModelFingerprint("openai", "test-model", None),
                profile=answer_model_profile(),
            ),
        ),
        context_policy_revision=CONTEXT_POLICY_REVISION,
        model_catalog_revision=current_model_catalog_revision(),
        idempotency_fingerprint=idempotency_fingerprint,
    )


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent .env and the repo's config.yaml from polluting unit tests."""
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", None)
    monkeypatch.setattr(config_module, "_find_yaml_config", _yaml_config_ignoring_repo_file)
    for key in list(os.environ):
        if key.startswith("DLIGHTRAG_"):
            monkeypatch.delenv(key, raising=False)
