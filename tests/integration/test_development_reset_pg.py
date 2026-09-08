# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration tests for the native development reset against an isolated database.

The suite derives its server connection from the shared integration-test
environment and creates a uniquely named database that it alone force-drops.
"""

import importlib.util
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import asyncpg
import pytest

from tests.integration.pg_conn import PG_CONN_KWARGS

_reset_path = Path(__file__).resolve().parents[2] / "scripts" / "reset_development.py"
_spec = importlib.util.spec_from_file_location("reset_development_cli_pg", _reset_path)
assert _spec is not None and _spec.loader is not None
_reset = importlib.util.module_from_spec(_spec)
sys.modules["reset_development_cli_pg"] = _reset
_spec.loader.exec_module(_reset)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_ADMIN: dict[str, Any] = PG_CONN_KWARGS
_TEST_DATABASE = f"dlightrag_reset_{os.getpid()}_{uuid.uuid4().hex[:12]}"
_TEST_CONN_KWARGS = {**PG_CONN_KWARGS, "database": _TEST_DATABASE}

_EXTENSIONS = ("vector", "pg_textsearch", "pg_jieba")


async def _pg_available() -> bool:
    try:
        conn = await asyncpg.connect(**_ADMIN)
        await conn.close()
        return True
    except OSError, asyncpg.PostgresError:
        return False


async def _create_test_database() -> None:
    conn = await asyncpg.connect(**_ADMIN)
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", _TEST_DATABASE)
        if not exists:
            await conn.execute(f'CREATE DATABASE "{_TEST_DATABASE}"')
    finally:
        await conn.close()


async def _drop_test_database() -> None:
    conn = await asyncpg.connect(**_ADMIN)
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", _TEST_DATABASE)
        if exists:
            await conn.execute(f'DROP DATABASE "{_TEST_DATABASE}" WITH (FORCE)')
    finally:
        await conn.close()


def _test_target() -> Any:
    return _reset.PostgresTarget(
        host=str(_TEST_CONN_KWARGS["host"]),
        port=int(_TEST_CONN_KWARGS["port"]),
        user=str(_TEST_CONN_KWARGS["user"]),
        password=str(_TEST_CONN_KWARGS["password"]),
        database=str(_TEST_CONN_KWARGS["database"]),
    )


@pytest.fixture(autouse=True)
async def _test_database(tmp_path: Path):
    if not await _pg_available():
        pytest.skip("PostgreSQL is not reachable")
    await _create_test_database()
    yield
    await _drop_test_database()


async def test_native_reset_replaces_public_schema_and_recreates_extensions(
    tmp_path: Path,
) -> None:
    working_dir = tmp_path / "dlightrag_storage"
    working_dir.mkdir()
    (working_dir / "old-file").write_text("stale")

    # Seed an application-shaped schema and a table.
    seed = await asyncpg.connect(**_TEST_CONN_KWARGS)
    try:
        await seed.execute("CREATE TABLE dlightrag_seed(value int)")
        await seed.execute("INSERT INTO dlightrag_seed VALUES (1)")
    finally:
        await seed.close()

    report = _reset.ResetReport(mode="native")
    await _reset._native_pg_work(
        _test_target(), working_dir, report, force_disconnect=False, dry_run=False
    )
    _reset.clear_working_dir_children(working_dir, report)

    assert report.ok, report.failures
    assert list(working_dir.iterdir()) == []
    assert _reset.verify_working_dir_empty(working_dir) == []

    conn = await asyncpg.connect(**_TEST_CONN_KWARGS)
    try:
        assert await _reset._verify_empty_postgres(conn) == []
        extensions = {
            row["extname"]
            for row in await conn.fetch(
                "SELECT extname FROM pg_extension WHERE extname = ANY($1::text[])", _EXTENSIONS
            )
        }
        assert extensions == set(_EXTENSIONS)
    finally:
        await conn.close()


async def test_rerun_after_success_converges_to_the_same_empty_state(tmp_path: Path) -> None:
    working_dir = tmp_path / "dlightrag_storage"
    working_dir.mkdir()
    (working_dir / "file-a").write_text("a")

    first = _reset.ResetReport(mode="native")
    await _reset._native_pg_work(
        _test_target(), working_dir, first, force_disconnect=False, dry_run=False
    )
    _reset.clear_working_dir_children(working_dir, first)
    assert first.ok, first.failures

    (working_dir / "file-b").write_text("b")
    second = _reset.ResetReport(mode="native")
    await _reset._native_pg_work(
        _test_target(), working_dir, second, force_disconnect=False, dry_run=False
    )
    _reset.clear_working_dir_children(working_dir, second)

    assert second.ok, second.failures
    assert list(working_dir.iterdir()) == []
    assert _reset.verify_working_dir_empty(working_dir) == []

    conn = await asyncpg.connect(**_TEST_CONN_KWARGS)
    try:
        assert await _reset._verify_empty_postgres(conn) == []
    finally:
        await conn.close()


async def test_active_sessions_are_refused_without_force_disconnect(tmp_path: Path) -> None:
    working_dir = tmp_path / "dlightrag_storage"
    working_dir.mkdir()

    other = await asyncpg.connect(**_TEST_CONN_KWARGS)
    try:
        refused = _reset.ResetReport(mode="native")
        await _reset._native_pg_work(
            _test_target(), working_dir, refused, force_disconnect=False, dry_run=False
        )
        assert not refused.ok
        assert any("sessions" in failure for failure in refused.failures)

        forced = _reset.ResetReport(mode="native")
        await _reset._native_pg_work(
            _test_target(), working_dir, forced, force_disconnect=True, dry_run=False
        )
        assert forced.ok, forced.failures
        # The other session was terminated.
        with pytest.raises((asyncpg.PostgresError, asyncpg.InterfaceError, ConnectionError)):
            await other.fetchval("SELECT 1")
    finally:
        try:
            await other.close()
        except Exception:
            pass


async def test_dry_run_reports_without_mutation(tmp_path: Path) -> None:
    working_dir = tmp_path / "dlightrag_storage"
    working_dir.mkdir()
    (working_dir / "file").write_text("keep")

    seed = await asyncpg.connect(**_TEST_CONN_KWARGS)
    try:
        await seed.execute("CREATE TABLE dlightrag_keep(value int)")
    finally:
        await seed.close()

    report = _reset.ResetReport(mode="native")
    await _reset._native_pg_work(
        _test_target(), working_dir, report, force_disconnect=False, dry_run=True
    )

    assert report.ok, report.failures
    assert any("dry-run-schemas" in step for step, _ in report.steps)
    assert any("dry-run-ddl" in step for step, _ in report.steps)
    # Nothing was mutated.
    assert (working_dir / "file").exists()
    conn = await asyncpg.connect(**_TEST_CONN_KWARGS)
    try:
        remains = await conn.fetchval("SELECT to_regclass('dlightrag_keep') IS NOT NULL")
        assert remains is True
    finally:
        await conn.close()


async def test_working_dir_verification_continues_after_pg_failure(tmp_path: Path) -> None:
    working_dir = tmp_path / "dlightrag_storage"
    working_dir.mkdir()
    (working_dir / "stale").write_text("x")

    unreachable = _reset.PostgresTarget(
        host="127.0.0.1", port=1, user="u", password="p", database="db"
    )
    report = _reset.ResetReport(mode="native")
    await _reset._native_pg_work(
        unreachable, working_dir, report, force_disconnect=False, dry_run=False
    )
    assert report.failures  # connection failure is reported

    # Independent file cleanup still runs and is verified.
    _reset.clear_working_dir_children(working_dir, report)
    assert _reset.verify_working_dir_empty(working_dir) == []
