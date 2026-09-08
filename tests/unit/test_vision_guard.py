# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the answer-image capability guard."""

import pytest

from dlightrag.engine.ai.vision import ImageCapabilityStatus
from dlightrag.engine.answer.errors import (
    ANSWER_IMAGE_CAPABILITY_UNKNOWN,
    CURRENT_IMAGES_UNSUPPORTED,
    AnswerImageError,
    CurrentImagePayloadError,
)
from dlightrag.engine.answer.image_capability import (
    AnswerImageCapability,
    check_answer_image_capability,
    check_answer_image_count,
)


def _capability(status: ImageCapabilityStatus) -> AnswerImageCapability:
    return AnswerImageCapability(
        status=status,
        configured_ceiling=8,
        effective_max_images=8 if status == "supported" else 0,
        provider="test",
        base_url=None,
        model="m",
        failure_kind=None,
    )


class TestAnswerImageCapabilityGuard:
    def test_raises_when_images_are_unsupported(self) -> None:
        with pytest.raises(AnswerImageError) as exc:
            check_answer_image_capability(
                image_count=1,
                capability=_capability("unsupported"),
            )
        assert exc.value.error_kind == CURRENT_IMAGES_UNSUPPORTED
        assert "[IMAGES_NOT_SUPPORTED_BY_MODEL]" in str(exc.value)

    def test_passes_when_images_are_supported(self) -> None:
        check_answer_image_capability(
            image_count=1,
            capability=_capability("supported"),
        )

    def test_passes_when_no_images_at_all(self) -> None:
        check_answer_image_capability(
            image_count=0,
            capability=_capability("unsupported"),
        )

    def test_unknown_fails_closed(self) -> None:
        # Fail-closed: an unconfirmed capability rejects with a clear kind rather
        # than a late provider or transport-budget failure.
        with pytest.raises(AnswerImageError) as exc:
            check_answer_image_capability(
                image_count=1,
                capability=_capability("unknown"),
            )
        assert exc.value.error_kind == ANSWER_IMAGE_CAPABILITY_UNKNOWN

    def test_unprobed_none_fails_closed(self) -> None:
        with pytest.raises(AnswerImageError) as exc:
            check_answer_image_capability(
                image_count=1,
                capability=None,
            )
        assert exc.value.error_kind == ANSWER_IMAGE_CAPABILITY_UNKNOWN

    def test_configured_ceiling_is_independent_of_query_model_support(self) -> None:
        check_answer_image_count(image_count=1, configured_ceiling=1)
        with pytest.raises(CurrentImagePayloadError, match="at most 1"):
            check_answer_image_count(image_count=2, configured_ceiling=1)
