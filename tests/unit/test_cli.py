# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for CLI argument validation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Load scripts/cli.py as a module (it's a script, not a package)
_cli_path = Path(__file__).resolve().parents[2] / "scripts" / "cli.py"
_spec = importlib.util.spec_from_file_location("cli", _cli_path)
_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cli)

build_parser = _cli.build_parser
_validate_ingest_args = _cli._validate_ingest_args


def _parse_ingest(args: list[str]):
    """Parse CLI args for the ingest subcommand."""
    return build_parser().parse_args(["ingest", *args])


# ---------------------------------------------------------------------------
# TestValidateIngestArgs — local source
# ---------------------------------------------------------------------------


class TestValidateLocal:
    """Validation for local source (default)."""

    def test_valid_local(self) -> None:
        args = _parse_ingest(["./docs"])
        _validate_ingest_args(args)  # should not raise

    def test_valid_local_with_flags(self) -> None:
        args = _parse_ingest(["./docs", "--replace", "--sync-hashes"])
        _validate_ingest_args(args)

    def test_local_requires_path(self) -> None:
        args = _parse_ingest([])
        with pytest.raises(SystemExit):
            _validate_ingest_args(args)

    def test_local_rejects_container(self) -> None:
        args = _parse_ingest(["./docs", "--container", "c"])
        with pytest.raises(SystemExit):
            _validate_ingest_args(args)

    def test_local_rejects_query(self) -> None:
        args = _parse_ingest(["./docs", "--query", "SELECT 1"])
        with pytest.raises(SystemExit):
            _validate_ingest_args(args)


# ---------------------------------------------------------------------------
# TestValidateIngestArgs — azure_blob source
# ---------------------------------------------------------------------------


class TestValidateAzureBlob:
    """Validation for azure_blob source."""

    def test_valid_azure_container_only(self) -> None:
        args = _parse_ingest(["--source", "azure_blob", "--container", "c"])
        _validate_ingest_args(args)

    def test_valid_azure_with_prefix(self) -> None:
        args = _parse_ingest(["--source", "azure_blob", "--container", "c", "--prefix", "docs/"])
        _validate_ingest_args(args)

    def test_valid_azure_with_blob_path(self) -> None:
        args = _parse_ingest(["--source", "azure_blob", "--container", "c", "--blob-path", "f.pdf"])
        _validate_ingest_args(args)

    def test_azure_requires_container(self) -> None:
        args = _parse_ingest(["--source", "azure_blob"])
        with pytest.raises(SystemExit):
            _validate_ingest_args(args)

    def test_azure_rejects_positional_path(self) -> None:
        args = _parse_ingest(["./docs", "--source", "azure_blob", "--container", "c"])
        with pytest.raises(SystemExit):
            _validate_ingest_args(args)

    def test_azure_blob_path_and_prefix_mutually_exclusive(self) -> None:
        args = _parse_ingest(
            [
                "--source",
                "azure_blob",
                "--container",
                "c",
                "--blob-path",
                "f.pdf",
                "--prefix",
                "docs/",
            ]
        )
        with pytest.raises(SystemExit):
            _validate_ingest_args(args)

    def test_azure_rejects_query(self) -> None:
        args = _parse_ingest(
            [
                "--source",
                "azure_blob",
                "--container",
                "c",
                "--query",
                "SELECT 1",
            ]
        )
        with pytest.raises(SystemExit):
            _validate_ingest_args(args)


# ---------------------------------------------------------------------------
# TestValidateIngestArgs — snowflake source
# ---------------------------------------------------------------------------


class TestValidateSnowflake:
    """Validation for snowflake source."""

    def test_valid_snowflake(self) -> None:
        args = _parse_ingest(["--source", "snowflake", "--query", "SELECT 1"])
        _validate_ingest_args(args)

    def test_valid_snowflake_with_table(self) -> None:
        args = _parse_ingest(
            [
                "--source",
                "snowflake",
                "--query",
                "SELECT 1",
                "--table",
                "reports",
            ]
        )
        _validate_ingest_args(args)

    def test_snowflake_requires_query(self) -> None:
        args = _parse_ingest(["--source", "snowflake"])
        with pytest.raises(SystemExit):
            _validate_ingest_args(args)

    def test_snowflake_rejects_replace(self) -> None:
        args = _parse_ingest(["--source", "snowflake", "--query", "SELECT 1", "--replace"])
        with pytest.raises(SystemExit):
            _validate_ingest_args(args)

    def test_snowflake_rejects_sync_hashes(self) -> None:
        args = _parse_ingest(["--source", "snowflake", "--query", "SELECT 1", "--sync-hashes"])
        with pytest.raises(SystemExit):
            _validate_ingest_args(args)

    def test_snowflake_rejects_container(self) -> None:
        args = _parse_ingest(
            [
                "--source",
                "snowflake",
                "--query",
                "SELECT 1",
                "--container",
                "c",
            ]
        )
        with pytest.raises(SystemExit):
            _validate_ingest_args(args)
