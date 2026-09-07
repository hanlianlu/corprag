# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Closed transient dependency classification for durable execution."""

import httpx
import pytest

from dlightrag.application.errors import CorpusUnavailableError
from dlightrag.engine.dependencies import (
    ParserUnavailableError,
    ProviderUnavailableError,
    classify_transient_dependency,
)


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://provider.example")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("provider rejected request", request=request, response=response)


@pytest.mark.parametrize("status", [401, 403, 400, 413, 422])
def test_auth_input_and_context_rejections_are_not_transient(status: int) -> None:
    assert classify_transient_dependency(_http_error(status)) is None


@pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504])
def test_known_provider_status_interruptions_are_transient(status: int) -> None:
    assert classify_transient_dependency(_http_error(status)) == "providers"


def test_unknown_exception_is_not_transient() -> None:
    assert classify_transient_dependency(RuntimeError("unknown")) is None


def test_explicit_typed_dependency_boundaries_are_transient() -> None:
    assert classify_transient_dependency(CorpusUnavailableError()) == "corpus_storage"
    assert classify_transient_dependency(ProviderUnavailableError()) == "providers"


def test_typed_parser_and_hinted_transport_failures_are_transient() -> None:
    assert classify_transient_dependency(ParserUnavailableError()) == "parser"
    assert classify_transient_dependency(ConnectionError("down")) is None
    assert (
        classify_transient_dependency(ConnectionError("down"), component_hint="corpus_storage")
        == "corpus_storage"
    )
    assert (
        classify_transient_dependency(TimeoutError(), component_hint="corpus_storage")
        == "corpus_storage"
    )


def test_authentication_cause_wins_over_a_transient_wrapper() -> None:
    try:
        raise RuntimeError("password authentication failed")
    except RuntimeError as auth:
        wrapper = CorpusUnavailableError()
        wrapper.__cause__ = auth

    assert classify_transient_dependency(wrapper) is None
