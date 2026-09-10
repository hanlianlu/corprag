# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""YAML 1.2 configuration loading for DlightRAG."""

from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

from pydantic_settings import YamlConfigSettingsSource
from ruamel.yaml import YAML

_YAML_VERSION = (1, 2)


def _load_yaml12_mapping(
    file_path: Path | Traversable,
    *,
    encoding: str | None = "utf-8",
) -> dict[str, Any]:
    """Load one mapping with YAML 1.2 scalar resolution and unique keys."""
    parser = YAML(typ="safe")
    parser.version = _YAML_VERSION
    parser.allow_duplicate_keys = False
    with file_path.open(encoding=encoding) as yaml_file:
        loaded = parser.load(yaml_file)

    document_version = parser.doc_infos[0].doc_version if parser.doc_infos else None
    if (
        document_version is not None
        and (
            document_version.major,
            document_version.minor,
        )
        != _YAML_VERSION
    ):
        raise ValueError(f'YAML configuration in "{file_path}" must use YAML 1.2')
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(f'YAML configuration in "{file_path}" must be a mapping')
    return loaded


class Yaml12ConfigSettingsSource(YamlConfigSettingsSource):
    """Feed YAML 1.2 configuration into Pydantic's settings pipeline."""

    def _read_file(self, file_path: Path | Traversable) -> dict[str, Any]:
        return _load_yaml12_mapping(file_path, encoding=self.yaml_file_encoding)


__all__ = ["Yaml12ConfigSettingsSource"]
