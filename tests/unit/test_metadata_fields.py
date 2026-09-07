# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for storage.metadata_fields — field registry."""

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from dlightrag.engine.rag.retrieval import MetadataFilter
from dlightrag.engine.rag.retrieval.metadata_fields import (
    FILTER_FIELD_COLUMNS,
    INGEST_FINALIZATION_COMPLETE_FIELD,
    METADATA_FIELD_IDS,
    NormalizedUserMetadata,
    extract_system_metadata,
    normalize_user_metadata,
)


def _writer_settings() -> Any:
    return SimpleNamespace(read_only=False)


class TestMetadataFields:
    """Storage-neutral metadata field ids."""

    def test_has_filename(self) -> None:
        assert "filename" in METADATA_FIELD_IDS

    def test_field_ids_unique(self) -> None:
        assert len(METADATA_FIELD_IDS) == len(set(METADATA_FIELD_IDS))


def test_metadata_registry_has_source_identity_and_download_locator() -> None:
    ids = set(METADATA_FIELD_IDS)

    assert {"source_uri", "download_locator"} <= ids


def test_extract_system_metadata_stores_distinct_source_and_download_fields() -> None:
    metadata = extract_system_metadata(
        "https://cdn.example.com/assets/1.pdf",
        display_filename="report.pdf",
        source_uri="bynder://asset/1",
        download_locator="https://cdn.example.com/assets/1.pdf",
    )

    assert metadata["source_uri"] == "bynder://asset/1"
    assert metadata["download_locator"] == "https://cdn.example.com/assets/1.pdf"


class TestDerivedFunctions:
    """Derived helper functions built from neutral metadata ids."""

    def test_filter_fields_map_to_real_columns(self) -> None:
        from dlightrag.engine.rag.retrieval.metadata_fields import FILTER_FIELD_COLUMNS

        columns = set(METADATA_FIELD_IDS)
        backing = {column for cols in FILTER_FIELD_COLUMNS.values() for column in cols}
        assert backing <= columns
        # Every filter the planner may emit resolves to a column.
        assert set(MetadataFilter.model_fields) - {"custom"} == set(FILTER_FIELD_COLUMNS)
        # A named file is matched against the stored name and its stem.
        assert FILTER_FIELD_COLUMNS["filename"] == ("filename", "filename_stem")
        # One column backs both ends of the range the planner emits.
        assert (
            FILTER_FIELD_COLUMNS["creation_date_from"] == FILTER_FIELD_COLUMNS["creation_date_to"]
        )


def test_user_metadata_is_stored_verbatim() -> None:
    """Case folding is applied by the SQL comparison, so storage stays lossless."""
    norm = normalize_user_metadata({"department": " Finance ", "sku": "AbC-123"})

    assert norm.custom_metadata == {"department": " Finance ", "sku": "AbC-123"}


def test_any_key_is_accepted_without_declaring_it() -> None:
    norm = normalize_user_metadata({"project": "Analytical Engine"})

    assert norm.custom_metadata == {"project": "Analytical Engine"}


def test_non_string_values_survive_untouched() -> None:
    norm = normalize_user_metadata({"pages": 42, "reviewed": True})

    assert norm.custom_metadata == {"pages": 42, "reviewed": True}


async def test_metadata_update_stores_without_reindexing() -> None:
    from unittest.mock import AsyncMock

    from dlightrag.engine.rag.workspace.workspace_rag import WorkspaceRag

    service = object.__new__(WorkspaceRag)
    service.settings = _writer_settings()
    service._metadata_index = AsyncMock()
    service._metadata_index.is_visible.return_value = True
    service._lightrag = AsyncMock()

    await service.aupdate_metadata("doc-1", {"reviewer": " Ada Lovelace "})

    _, saved = service._metadata_index.merge_custom_metadata.await_args.args
    assert saved["custom_metadata"]["reviewer"] == " Ada Lovelace "
    service._lightrag.apipeline_enqueue_documents.assert_not_called()


async def test_metadata_update_reports_an_unknown_document() -> None:
    from unittest.mock import AsyncMock

    from dlightrag.engine.rag.workspace.workspace_rag import WorkspaceRag

    service = object.__new__(WorkspaceRag)
    service.settings = _writer_settings()
    service._metadata_index = AsyncMock()
    service._metadata_index.is_visible.return_value = True
    service._metadata_index.merge_custom_metadata.return_value = False

    # Updating a document that was never ingested must not conjure one.
    with pytest.raises(KeyError):
        await service.aupdate_metadata("ghost", {"reviewer": "Ada"})


async def test_metadata_update_rejects_an_unpublished_document() -> None:
    from unittest.mock import AsyncMock

    from dlightrag.engine.rag.workspace.workspace_rag import WorkspaceRag

    service = object.__new__(WorkspaceRag)
    service.settings = _writer_settings()
    service._metadata_index = AsyncMock()
    service._metadata_index.is_visible.return_value = False

    with pytest.raises(KeyError):
        await service.aupdate_metadata("hidden", {"reviewer": "Ada"})

    service._metadata_index.merge_custom_metadata.assert_not_awaited()


class TestCallerSettableColumns:
    """creation_date is the one filterable document attribute with no ingest parameter."""

    @staticmethod
    def _normalize(value: object) -> NormalizedUserMetadata:
        return normalize_user_metadata({"creation_date": value})

    def test_iso_date_reaches_the_column_not_jsonb(self) -> None:
        norm = self._normalize("2024-03-05")

        assert norm.system == {"creation_date": datetime(2024, 3, 5, 0, 0)}
        assert norm.custom_metadata == {}

    def test_offset_is_converted_to_the_same_instant_the_filter_uses(self) -> None:
        norm = self._normalize("2024-01-01T08:00:00+08:00")

        assert norm.system == {"creation_date": datetime(2024, 1, 1, 0, 0)}

    def test_bare_timestamp_is_read_as_utc(self) -> None:
        norm = self._normalize("2024-01-01T12:30:00")

        assert norm.system == {"creation_date": datetime(2024, 1, 1, 12, 30)}

    def test_datetime_object_is_accepted(self) -> None:
        norm = self._normalize(datetime(2024, 1, 1, 9, 0, tzinfo=UTC))

        assert norm.system == {"creation_date": datetime(2024, 1, 1, 9, 0)}

    @pytest.mark.parametrize("value", ["2024/01/01", "not-a-date", 1704067200])
    def test_unparseable_value_is_rejected_loudly(self, value: object) -> None:
        with pytest.raises(ValueError, match="creation_date must be an ISO 8601"):
            self._normalize(value)


class TestReservedMetadataKeys:
    """A filter name in `metadata` would be stored where no filter ever reads it."""

    @pytest.mark.parametrize("key", sorted(FILTER_FIELD_COLUMNS))
    def test_filter_names_are_rejected_rather_than_stored_in_jsonb(self, key: str) -> None:
        with pytest.raises(ValueError, match="built-in metadata field"):
            normalize_user_metadata({key: "x"})

    def test_finalization_journal_is_reserved_from_user_metadata(self) -> None:
        with pytest.raises(ValueError, match="built-in metadata field"):
            normalize_user_metadata({INGEST_FINALIZATION_COMPLETE_FIELD: True})
