# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the shared client contracts across REST, Web, and MCP surfaces."""

import pytest
from pydantic import ValidationError

from dlightrag.adapters.http.rest.models import AnswerRequest, RetrievalResponse, RetrieveRequest
from dlightrag.adapters.mcp.contracts import AnswerInput, RetrieveInput
from dlightrag.application.corpus_admin import (
    IngestSpec,
    ingest_kwargs_from_spec,
    ingest_spec_from_payload,
)
from dlightrag.engine.answer.citations.contracts import SourceReference, SourceReferencePayload
from dlightrag.engine.answer.client_contracts import (
    MAX_HISTORY_CONTENT_CHARS,
    MAX_HISTORY_MESSAGES,
    AnswerAttachmentLink,
    ConversationMessage,
    conversation_history_as_dicts,
)
from dlightrag.engine.rag.corpus.contracts import IngestDocument


def test_per_interface_current_image_admission() -> None:
    images = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,image-{index}"}}
        for index in range(4)
    ]

    # Retrieve interfaces gate current query images through their own contract.
    for model in (RetrieveRequest, RetrieveInput):
        with pytest.raises(ValidationError):
            model.model_validate({"query": "four images", "query_images": images})

    # The Answer contract exposes no public current-image field: images arrive as
    # ordered attachments/resources at the request boundary, never a query field.
    for model in (AnswerRequest, AnswerInput):
        assert "query_images" not in set(model.model_fields)


@pytest.mark.parametrize("model", [RetrieveRequest, RetrieveInput, AnswerRequest, AnswerInput])
@pytest.mark.parametrize("field", ["top_k", "chunk_top_k"])
def test_query_limits_must_be_positive(model, field: str) -> None:
    with pytest.raises(ValidationError):
        model.model_validate({"query": "q", field: 0})


def test_public_requests_reject_conversation_fields() -> None:
    for model in (RetrieveRequest, AnswerRequest, RetrieveInput, AnswerInput):
        fields = set(model.model_fields)
        assert "conversation_history" not in fields
        assert "session_id" not in fields
        assert "referenced_image_ids" not in fields

        for field in ("conversation_history", "session_id", "referenced_image_ids"):
            with pytest.raises(ValidationError):
                model.model_validate(
                    {"query": "standalone", field: [] if field != "session_id" else "s"}
                )


def test_answer_contracts_accept_history_retrieve_rejects() -> None:
    valid = {"role": "user", "content": "Earlier turn"}
    for model in (AnswerRequest, AnswerInput):
        assert "history" in model.model_fields
        parsed = model.model_validate({"query": "follow up", "history": [valid]})
        assert parsed.history is not None
        assert parsed.history[0].role == "user"
        assert parsed.history[0].content == "Earlier turn"
    for model in (RetrieveRequest, RetrieveInput):
        assert "history" not in model.model_fields
        with pytest.raises(ValidationError):
            model.model_validate({"query": "standalone", "history": [valid]})
    with pytest.raises(ValidationError):
        AnswerRequest.model_validate(
            {"query": "q", "history": [valid] * (MAX_HISTORY_MESSAGES + 1)}
        )


def test_conversation_message_validation_and_projection() -> None:
    with pytest.raises(ValidationError):
        ConversationMessage.model_validate({"role": "system", "content": "x"})
    with pytest.raises(ValidationError):
        ConversationMessage.model_validate({"role": "user", "content": ""})
    with pytest.raises(ValidationError):
        ConversationMessage.model_validate(
            {"role": "user", "content": "x" * (MAX_HISTORY_CONTENT_CHARS + 1)}
        )
    messages = [
        ConversationMessage(role="user", content="hi"),
        ConversationMessage(role="assistant", content="hello"),
    ]
    assert conversation_history_as_dicts(messages) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert conversation_history_as_dicts(None) is None
    assert conversation_history_as_dicts([]) is None


def test_public_retrieval_response_has_no_session_image_ids() -> None:
    assert "current_image_ids" not in RetrievalResponse.model_fields


def test_answer_links_accept_http_and_https() -> None:
    https_link = AnswerAttachmentLink(url="https://example.com/report.pdf")
    http_link = AnswerAttachmentLink(url="http://example.com/report.pdf")
    assert https_link.filename is None
    assert http_link.url == "http://example.com/report.pdf"
    with pytest.raises(ValidationError):
        AnswerAttachmentLink(url="ftp://example.com/report.pdf")


def test_answer_links_reject_embedded_credentials() -> None:
    for url in (
        "https://user:pass@example.com/report.pdf",
        "https://user@example.com/report.pdf",
    ):
        with pytest.raises(ValidationError):
            AnswerAttachmentLink(url=url)


def test_retrieve_accepts_query_images_but_rejects_attachments() -> None:
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
    for model in (RetrieveRequest, RetrieveInput):
        parsed = model.model_validate({"query": "q", "query_images": [image]})
        assert parsed.query_images is not None
        with pytest.raises(ValidationError):
            model.model_validate({"query": "q", "attachments": []})


@pytest.mark.parametrize("model", [AnswerRequest, AnswerInput])
def test_answer_contracts_reject_query_images(model) -> None:
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
    assert "query_images" not in model.model_fields
    with pytest.raises(ValidationError):
        model.model_validate({"query": "q", "query_images": [image]})


@pytest.mark.parametrize("model", [AnswerRequest, AnswerInput])
def test_answer_contracts_accept_attachment_links(model) -> None:
    assert "attachments" in model.model_fields
    parsed = model.model_validate(
        {
            "query": "q",
            "attachments": [{"url": "https://example.com/report.pdf", "filename": "report.pdf"}],
        }
    )
    assert parsed.attachments is not None
    assert parsed.attachments[0].url == "https://example.com/report.pdf"
    assert parsed.attachments[0].filename == "report.pdf"
    http_parsed = model.model_validate(
        {"query": "q", "attachments": [{"url": "http://example.com/x.pdf"}]}
    )
    assert http_parsed.attachments is not None
    assert http_parsed.attachments[0].url == "http://example.com/x.pdf"
    with pytest.raises(ValidationError):
        model.model_validate({"query": "q", "attachments": [{"url": "ftp://example.com/x.pdf"}]})


@pytest.mark.parametrize("model", [AnswerRequest, AnswerInput])
def test_answer_attachments_reject_local_and_base64_fields(model) -> None:
    for descriptor in (
        {"path": "/etc/passwd"},
        {"url": "https://example.com/x.pdf", "path": "/etc/passwd"},
        {"url": "https://example.com/x.pdf", "content": "aGVsbG8="},
        {"url": "https://example.com/x.pdf", "data": "aGVsbG8="},
    ):
        with pytest.raises(ValidationError):
            model.model_validate({"query": "q", "attachments": [descriptor]})


def test_public_source_contract_exposes_only_public_locators() -> None:
    internal_fields = set(SourceReference.model_fields)
    public_fields = set(SourceReferencePayload.model_fields)

    assert {"path", "url", "download_url"}.isdisjoint(internal_fields)
    assert {"workspace", "download_locator", "source_uri"} <= internal_fields
    assert {"path", "url", "workspace", "download_locator"}.isdisjoint(public_fields)
    assert {"source_uri", "download_url"} <= public_fields


def test_ingest_spec_from_payload_preserves_s3_manifest_fields() -> None:
    spec = ingest_spec_from_payload(
        {
            "source_type": "s3",
            "bucket": "my-bucket",
            "s3_region": "eu-north-1",
            "metadata": {"source_system": "s3-prod"},
            "retain_source_file": True,
            "documents": [
                {
                    "key": "docs/a.pdf",
                    "title": "A",
                    "metadata": {"department": "Legal", "asset_id": "a"},
                }
            ],
        }
    )

    assert spec == IngestSpec(
        source_type="s3",
        bucket="my-bucket",
        s3_region="eu-north-1",
        metadata={"source_system": "s3-prod"},
        retain_source_file=True,
        documents=[
            IngestDocument(
                key="docs/a.pdf",
                title="A",
                metadata={"department": "Legal", "asset_id": "a"},
            )
        ],
    )


def test_ingest_spec_from_payload_preserves_url_identity_fields() -> None:
    spec = ingest_spec_from_payload(
        {
            "source_type": "url",
            "url": "https://cdn.example.com/download?id=asset-1&signature=secret",
            "filename": "asset.pdf",
            "source_uri": "bynder://asset/asset-1",
        }
    )

    assert spec == IngestSpec(
        source_type="url",
        url="https://cdn.example.com/download?id=asset-1&signature=secret",
        filename="asset.pdf",
        source_uri="bynder://asset/asset-1",
    )


def test_url_ingest_projects_download_uri_fields() -> None:
    spec = IngestSpec(
        source_type="url",
        urls=["https://fetch.example.com/a.pdf", "https://fetch.example.com/b.pdf"],
        source_uris=["cms://a", "cms://b"],
        download_uris=[
            "https://cdn.example.com/a.pdf",
            "https://cdn.example.com/b.pdf",
        ],
    )

    kwargs = ingest_kwargs_from_spec(spec)

    assert kwargs["download_uris"] == [
        "https://cdn.example.com/a.pdf",
        "https://cdn.example.com/b.pdf",
    ]


def test_url_ingest_preserves_explicit_empty_download_uri_for_canonical_validation() -> None:
    kwargs = ingest_kwargs_from_spec(
        ingest_spec_from_payload(
            {
                "source_type": "url",
                "url": "https://fetch.example.com/a.pdf",
                "download_uri": "",
            }
        )
    )

    assert kwargs["download_uri"] == ""


def test_url_ingest_download_uri_cardinality_is_strict() -> None:
    with pytest.raises(ValidationError, match="download_uris"):
        IngestSpec(
            source_type="url",
            urls=["https://example.com/a.pdf", "https://example.com/b.pdf"],
            download_uris=["https://cdn.example.com/a.pdf"],
        )


def test_url_ingest_single_download_uri_requires_single_url() -> None:
    with pytest.raises(ValidationError, match="single url"):
        IngestSpec(
            source_type="url",
            urls=["https://example.com/a.pdf", "https://example.com/b.pdf"],
            download_uri="https://cdn.example.com/a.pdf",
        )


def test_url_ingest_download_uri_forms_are_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        IngestSpec(
            source_type="url",
            url="https://example.com/a.pdf",
            download_uri="https://cdn.example.com/a.pdf",
            download_uris=["https://cdn.example.com/a.pdf"],
        )


def test_url_manifest_projects_per_document_download_uri() -> None:
    spec = IngestSpec(
        source_type="url",
        documents=[
            IngestDocument(
                url="https://fetch.example.com/download?sig=secret",
                source_uri="cms://asset/a",
                download_uri="https://cdn.example.com/a.pdf",
            )
        ],
    )

    assert ingest_kwargs_from_spec(spec)["documents"] == [
        {
            "url": "https://fetch.example.com/download?sig=secret",
            "source_uri": "cms://asset/a",
            "download_uri": "https://cdn.example.com/a.pdf",
        }
    ]


def test_url_manifest_rejects_top_level_download_uri() -> None:
    with pytest.raises(ValidationError, match="documents.*mutually exclusive"):
        IngestSpec(
            source_type="url",
            documents=[IngestDocument(url="https://fetch.example.com/a.pdf")],
            download_uri="https://cdn.example.com/a.pdf",
        )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "source_type": "local",
            "documents": [{"path": "a.pdf", "download_uri": "https://cdn.example.com/a.pdf"}],
        },
        {
            "source_type": "azure_blob",
            "container_name": "container",
            "documents": [{"key": "a.pdf", "download_uri": "azure://container/a.pdf"}],
        },
        {
            "source_type": "s3",
            "bucket": "bucket",
            "documents": [{"key": "a.pdf", "download_uri": "s3://bucket/a.pdf"}],
        },
        {
            "source_type": "local",
            "path": "a.pdf",
            "download_uri": "https://cdn.example.com/a.pdf",
        },
    ],
)
def test_non_url_ingest_rejects_download_uri_fields_before_manifest_returns(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="only valid for URL ingestion"):
        IngestSpec.model_validate(payload)


def test_retrieve_request_accepts_bm25_query() -> None:
    from dlightrag.adapters.http.rest.models import RetrieveRequest

    body = RetrieveRequest(query="q", bm25_query="alpha beta")

    assert body.bm25_query == "alpha beta"


def test_retrieve_request_rejects_overlong_bm25_query() -> None:
    import pytest
    from pydantic import ValidationError

    from dlightrag.adapters.http.rest.models import RetrieveRequest

    with pytest.raises(ValidationError):
        RetrieveRequest(query="q", bm25_query="x" * 2000)


def test_retrieve_input_accepts_bm25_query() -> None:
    from dlightrag.adapters.mcp.contracts import RetrieveInput

    args = RetrieveInput(query="q", bm25_query="alpha beta")

    assert args.bm25_query == "alpha beta"


@pytest.mark.parametrize("model", [AnswerRequest, AnswerInput])
def test_answer_contracts_reject_bm25_query(model) -> None:
    with pytest.raises(ValidationError):
        model.model_validate({"query": "q", "bm25_query": "alpha beta"})


@pytest.mark.parametrize("model", [AnswerRequest, AnswerInput])
def test_answer_contracts_reject_answer_context_top_k(model) -> None:
    assert "answer_context_top_k" not in model.model_fields
    with pytest.raises(ValidationError):
        model.model_validate({"query": "q", "answer_context_top_k": 3})


def test_mcp_query_images_stay_non_nullable_with_list_default() -> None:
    parsed = RetrieveInput.model_validate({"query": "q"})
    assert parsed.query_images == []
    with pytest.raises(ValidationError):
        RetrieveInput.model_validate({"query": "q", "query_images": None})
    schema = RetrieveInput.model_json_schema()
    assert schema["properties"]["query_images"]["type"] == "array"
    assert "query_images" not in AnswerInput.model_fields
