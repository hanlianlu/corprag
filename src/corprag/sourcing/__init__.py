# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Data source adapters for document ingestion."""

from corprag.sourcing.base import DataSource

__all__ = ["DataSource"]


def __getattr__(name: str):
    """Lazy import optional sourcing adapters."""
    if name == "AzureBlobDataSource":
        from corprag.sourcing.azure_blob import AzureBlobDataSource

        return AzureBlobDataSource
    if name == "SnowflakeDataSource":
        from corprag.sourcing.snowflake import SnowflakeDataSource

        return SnowflakeDataSource
    if name == "LocalDataSource":
        from corprag.sourcing.local import LocalDataSource

        return LocalDataSource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
