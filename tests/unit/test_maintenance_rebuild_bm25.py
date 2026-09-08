# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the offline workspace BM25 rebuild command."""

from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest


def _config(*, enabled: bool = True, reader: bool = False) -> SimpleNamespace:
    retrieval = SimpleNamespace(
        bm25_enabled=enabled,
        bm25_profiles=[
            SimpleNamespace(name="en", text_config="english", languages=["en"], fallback=False),
            SimpleNamespace(name="simple", text_config="simple", languages=[], fallback=True),
        ],
        bm25_k1=1.4,
        bm25_b=0.65,
    )
    return SimpleNamespace(
        corpus=SimpleNamespace(retrieval=retrieval),
        deployment=SimpleNamespace(workspace="research"),
        is_reader=reader,
    )


def test_bm25_rebuild_parser_defaults() -> None:
    from dlightrag.adapters.postgres.rebuild_bm25 import build_parser

    args = build_parser().parse_args([])

    assert args.yes is False
    assert args.batch_size == 500


def test_bm25_rebuild_requires_yes() -> None:
    from dlightrag.adapters.postgres.rebuild_bm25 import build_parser, validate_args

    args = build_parser().parse_args([])

    with pytest.raises(SystemExit, match="--yes is required"):
        validate_args(args)


def test_pyproject_exposes_bm25_rebuild_console_script() -> None:
    import tomllib

    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"]["dlightrag-rebuild-bm25"] == (
        "dlightrag.adapters.postgres.rebuild_bm25:main"
    )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (_config(enabled=False), "bm25_enabled=true"),
        (_config(reader=True), "writer service role"),
    ],
)
async def test_bm25_rebuild_rejects_incompatible_config(
    config: SimpleNamespace,
    message: str,
) -> None:
    from dlightrag.adapters.postgres.rebuild_bm25 import run_rebuild_bm25

    with pytest.raises(SystemExit, match=message):
        await run_rebuild_bm25(config=cast(Any, config), assume_yes=True)


async def test_bm25_rebuild_provisions_indexes_then_relabels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.adapters.postgres import rebuild_bm25 as module

    config = _config()
    events: list[object] = []

    class Coordination:
        @asynccontextmanager
        async def workspace_initialization(self):
            events.append("prerequisites:enter")
            try:
                yield
            finally:
                events.append("prerequisites:exit")

    def build_backend(resolved_config: object) -> SimpleNamespace:
        assert resolved_config is config
        events.append("factory")
        return SimpleNamespace(coordination=Coordination())

    monkeypatch.setattr(module, "build_pg_corpus_backend", build_backend)

    async def rebuild(config: Any, **kwargs: Any) -> dict[str, int]:
        events.append(("rebuild", config, kwargs))
        return {"processed_chunks": 3, "updated_chunks": 3}

    monkeypatch.setattr(module, "rebuild_postgres_bm25", rebuild)
    fake_pool = SimpleNamespace(bind=MagicMock(), close=AsyncMock())
    monkeypatch.setattr(module, "pg_pool", fake_pool)

    stats = await module.run_rebuild_bm25(
        config=cast(Any, config),
        assume_yes=True,
        batch_size=25,
    )

    assert stats == {"processed_chunks": 3, "updated_chunks": 3}
    assert events[:2] == ["factory", "prerequisites:enter"]
    rebuild_event = cast(tuple[str, Any, dict[str, Any]], events[2])
    assert rebuild_event[0] == "rebuild"
    assert rebuild_event[1] is config
    assert rebuild_event[2] == {"batch_size": 25}
    assert events[3] == "prerequisites:exit"
    fake_pool.bind.assert_called_once_with(config)
    fake_pool.close.assert_awaited_once()
