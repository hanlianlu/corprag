#!/usr/bin/env python3
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Reset the complete DlightRAG development environment (repository host tool).

Two explicit modes, never auto-detected:

    docker  docker compose down -v on both DlightRAG volumes, then start only
            PostgreSQL and verify the empty extension-only database.
    native  replace the dedicated development database's ``public`` schema,
            recreate the required extensions, and clear only the children of
            the verified working-directory root.

Usage:
    uv run scripts/reset_development.py --mode docker
    uv run scripts/reset_development.py --mode native
    uv run scripts/reset_development.py --mode docker --dry-run
    uv run scripts/reset_development.py --mode native --yes
    uv run scripts/reset_development.py --mode native --force-disconnect
    uv run scripts/reset_development.py --mode native --allow-remote-reset

Safety:
    * a read-only ``--dry-run`` resolves the same configuration, performs only
      read-only target/session/path checks, and reports every database, Compose
      project, volume, and filesystem path a real run would affect;
    * interactive execution requires typing the exact database name; ``--yes``
      skips the prompt but never bypasses target validation;
    * native mode refuses a non-loopback host without ``--allow-remote-reset``
      and refuses other database sessions without ``--force-disconnect``;
    * every invocation targets the complete database, migration ledger,
      LightRAG/DlightRAG state, and local runtime/corpus files: there is no
      partial reset scope. Product workspace reset stays a separate capability
      in scripts/reset_workspace.py; neither command imports the other.

The tool imports no application modules: it owns target validation, confirmation,
orchestration, and result verification with its host-tool dependencies.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import asyncpg
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

_REPO_ROOT = Path(__file__).resolve().parent.parent
_COMPOSE_FILE = "docker-compose.yml"
_DEFAULT_COMPOSE_PROJECT = "dlightrag"
_DEFAULT_PG = {
    "host": "localhost",
    "port": 5432,
    "user": "dlightrag",
    "password": "dlightrag",
    "database": "dlightrag",
}
_REQUIRED_EXTENSIONS = ("vector", "pg_textsearch", "pg_jieba", "pg_trgm")
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})
_SYSTEM_SCHEMAS = ("pg_catalog", "information_schema", "pg_toast")
_COMPOSE_VOLUMES = ("pg18_data", "dlightrag_data", "dlightrag_agent_workspaces")


@dataclass(frozen=True, slots=True)
class PostgresTarget:
    host: str
    port: int
    user: str
    password: str
    database: str

    def identity(self) -> str:
        return f"{self.user}@{self.host}:{self.port}/{self.database}"


@dataclass(slots=True)
class ResetReport:
    mode: str
    steps: list[tuple[str, str]] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def record(self, step: str, detail: str) -> None:
        self.steps.append((step, detail))

    def fail(self, step: str, detail: str) -> None:
        self.failures.append(f"{step}: {detail}")

    @property
    def ok(self) -> bool:
        return not self.failures


def _read_env(repo_root: Path) -> dict[str, str]:
    """Read DLIGHTRAG_* values from the repository .env, overlaid on os.environ."""
    values: dict[str, str] = {}
    env_file = repo_root / ".env"
    if env_file.is_file():
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
    for key, value in os.environ.items():
        if key.startswith("DLIGHTRAG_") or key == "COMPOSE_PROJECT_NAME":
            values[key] = value
    return values


def _read_config(repo_root: Path) -> dict[str, object]:
    """Read the canonical YAML 1.2 without importing the application being reset."""
    config_file = repo_root / "config.yaml"
    if not config_file.is_file():
        return {}
    parser = YAML(typ="safe")
    parser.version = (1, 2)
    parser.allow_duplicate_keys = False
    with config_file.open(encoding="utf-8") as stream:
        loaded = parser.load(stream)
    document_version = parser.doc_infos[0].doc_version if parser.doc_infos else None
    if document_version is not None and (
        document_version.major,
        document_version.minor,
    ) != (1, 2):
        raise ValueError("config.yaml must use YAML 1.2")
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("config.yaml must contain a mapping")
    return loaded


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _working_dir_root(
    repo_root: Path,
    env: dict[str, str],
    config: dict[str, object] | None = None,
) -> Path:
    """Resolve the canonical working-directory root with env precedence."""
    config = _read_config(repo_root) if config is None else config
    deployment = _mapping(config.get("deployment"))
    configured = env.get("DLIGHTRAG_DEPLOYMENT__WORKING_DIR") or deployment.get("working_dir")
    path = Path(str(configured or "./dlightrag_storage")).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.absolute()


def _workspace_root(
    repo_root: Path,
    env: dict[str, str],
    config: dict[str, object] | None = None,
) -> Path | None:
    """Resolve the Agent Workspace root only when local execution is enabled."""
    config = _read_config(repo_root) if config is None else config
    answer = _mapping(config.get("answer"))
    agent = _mapping(answer.get("agent"))
    execution = env.get("DLIGHTRAG_ANSWER__AGENT__EXECUTION_ENVIRONMENT") or agent.get(
        "execution_environment"
    )
    if str(execution or "disabled").strip() not in {"trust", "sandbox"}:
        return None
    configured = env.get("DLIGHTRAG_ANSWER__AGENT__WORKSPACE_ROOT") or agent.get("workspace_root")
    if configured is None or str(configured).strip() in {"", "null", "None", "~"}:
        return (Path.home() / ".dlightrag" / "agent_workspaces").absolute()
    path = Path(str(configured)).expanduser()
    if not path.is_absolute():
        raise ValueError(
            "answer.agent.workspace_root must be an absolute path when local execution is enabled"
        )
    return path.absolute()


def resolve_postgres_target(
    env: dict[str, str],
    config: dict[str, object] | None = None,
) -> PostgresTarget:
    prefix = "DLIGHTRAG_STORAGE__POSTGRES__"
    storage = _mapping((config or {}).get("storage"))
    postgres = _mapping(storage.get("postgres"))

    def value(name: str) -> object:
        return env.get(f"{prefix}{name.upper()}") or postgres.get(name) or _DEFAULT_PG[name]

    return PostgresTarget(
        host=str(value("host")),
        port=int(str(value("port"))),
        user=str(value("user")),
        password=str(value("password")),
        database=str(value("database")),
    )


# ─────────────────────────────────────────────────────────────────
# Target validation
# ─────────────────────────────────────────────────────────────────


def _symlink_violations(root: Path, *, stop: Path | None = None) -> list[str]:
    violations: list[str] = []
    for candidate in (root, *root.parents):
        if candidate == stop or candidate == Path(candidate.anchor):
            break
        if candidate.is_symlink():
            violations.append(f"root or an ancestor is a symbolic link: {candidate}")
            break
    for child in root.iterdir() if root.is_dir() else ():
        if child.is_symlink():
            violations.append(f"first-level child is a symbolic link: {child}")
    return violations


def validate_working_dir(working_dir: Path, repo_root: Path) -> list[str]:
    """Require the corpus root to be a strict, symlink-free repo descendant."""
    raw = working_dir.absolute()
    root, repo, home = raw.resolve(), repo_root.resolve(), Path.home().resolve()
    violations: list[str] = []
    if root == Path("/") or root == home:
        violations.append(f"working-directory root {root} must not be / or the home directory")
    if root == repo:
        violations.append(f"working-directory root must not be the repository root {repo}")
    elif not root.is_relative_to(repo):
        violations.append(
            f"working-directory root {root} must live inside the repository root {repo}"
        )
    violations.extend(_symlink_violations(raw, stop=repo))
    return violations


def validate_workspace_root(workspace_root: Path, repo_root: Path) -> list[str]:
    """Reject Agent cleanup roots that could contain user or repository data."""
    raw = workspace_root.absolute()
    root, repo, home = raw.resolve(), repo_root.resolve(), Path.home().resolve()
    violations: list[str] = []
    if root == Path("/") or root == home:
        violations.append(f"Agent Workspace root {root} must not be / or the home directory")
    if root == repo or repo.is_relative_to(root):
        violations.append(
            f"Agent Workspace root {root} must not contain the repository root {repo}"
        )
    violations.extend(_symlink_violations(raw))
    return violations


def validate_runtime_roots(
    working_dir: Path,
    workspace_root: Path | None,
    repo_root: Path,
) -> list[str]:
    violations = validate_working_dir(working_dir, repo_root)
    if workspace_root is None:
        return violations
    violations.extend(validate_workspace_root(workspace_root, repo_root))
    if (
        working_dir == workspace_root
        or working_dir.is_relative_to(workspace_root)
        or workspace_root.is_relative_to(working_dir)
    ):
        violations.append("Agent Workspace root must not overlap the working-directory root")
    return violations


def validate_native_target(
    *,
    target: PostgresTarget,
    working_dir: Path,
    repo_root: Path,
    allow_remote_reset: bool,
) -> list[str]:
    """Return every safety violation; an empty list means the target is safe."""
    violations: list[str] = []
    if not allow_remote_reset and target.host not in _LOOPBACK_HOSTS:
        violations.append(
            f"native reset refuses non-loopback host {target.host!r}; "
            "pass --allow-remote-reset to override for a dedicated development host"
        )
    violations.extend(validate_working_dir(working_dir, repo_root))
    return violations


# ─────────────────────────────────────────────────────────────────
# Verification helpers
# ─────────────────────────────────────────────────────────────────


async def _verify_empty_postgres(
    conn: asyncpg.Connection,
    *,
    expected_schemas: tuple[str, ...] = ("public",),
) -> list[str]:
    """Return violations: app schemas, missing extensions, or a migration ledger."""
    violations: list[str] = []
    schemas = await conn.fetch(
        "SELECT schema_name FROM information_schema.schemata"
        " WHERE schema_name NOT LIKE 'pg_%' AND schema_name NOT IN ($1, $2, $3)",
        *_SYSTEM_SCHEMAS,
    )
    present = {row["schema_name"] for row in schemas}
    unexpected = present - set(expected_schemas)
    if unexpected:
        violations.append(f"unexpected schemas remain: {sorted(unexpected)}")
    missing = set(expected_schemas) - present
    if missing:
        violations.append(f"expected schemas are missing: {sorted(missing)}")

    installed = {
        row["extname"]
        for row in await conn.fetch(
            "SELECT extname FROM pg_extension WHERE extname = ANY($1::text[])", _REQUIRED_EXTENSIONS
        )
    }
    missing_extensions = set(_REQUIRED_EXTENSIONS) - installed
    if missing_extensions:
        violations.append(f"required extensions missing: {sorted(missing_extensions)}")

    ledger = await conn.fetchval("SELECT to_regclass('dlightrag_schema_migrations') IS NOT NULL")
    if ledger:
        violations.append("migration ledger dlightrag_schema_migrations still exists")
    return violations


async def _other_sessions(conn: asyncpg.Connection, database: str) -> list[dict[str, object]]:
    rows = await conn.fetch(
        "SELECT pid, usename, application_name FROM pg_stat_activity"
        " WHERE datname = $1 AND pid <> pg_backend_pid()",
        database,
    )
    return [dict(row) for row in rows]


# ─────────────────────────────────────────────────────────────────
# Docker mode
# ─────────────────────────────────────────────────────────────────


def _compose(
    *args: str,
    project: str = _DEFAULT_COMPOSE_PROJECT,
) -> subprocess.CompletedProcess[str]:
    argv = ["docker", "compose", "-p", project, "-f", _COMPOSE_FILE, *args]
    return subprocess.run(  # noqa: S603 - fixed docker compose argv, host ops tool
        argv,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )


def run_docker_reset(
    target: PostgresTarget,
    report: ResetReport,
    *,
    compose_project: str | None = None,
) -> None:
    """Delete all DlightRAG volumes, start PostgreSQL, and verify exact emptiness."""
    project = compose_project or _DEFAULT_COMPOSE_PROJECT
    report.record("compose", f"project {project}, file {_COMPOSE_FILE}")
    down = _compose("down", "-v", project=project)
    if down.returncode != 0:
        report.fail(
            "compose-down",
            f"docker compose down -v failed: {down.stderr.strip() or down.stdout.strip()}",
        )
        return
    report.record(
        "compose-down",
        f"deleted volumes {', '.join(_COMPOSE_VOLUMES)}; every service stopped",
    )

    up = _compose("up", "-d", "postgres", project=project)
    if up.returncode != 0:
        report.fail(
            "compose-up",
            f"docker compose up -d postgres failed: {up.stderr.strip() or up.stdout.strip()}",
        )
        return
    report.record("compose-up", "started only the postgres service")

    healthy = _wait_for_postgres_health(report, project=project)
    if not healthy:
        return

    psql = _psql(target, "SELECT 1", project=project)
    if psql.returncode != 0:
        report.fail("psql-check", f"psql verification query failed: {psql.stderr.strip()}")
        return
    report.record("postgres-health", "postgres is healthy and accepts connections")
    report.record("postgres-empty", "PGDATA reinitialized from empty; init.sql ran")

    verification_checks = [
        (
            "extensions",
            "SELECT count(*) FROM pg_extension WHERE extname = ANY('{vector,pg_textsearch,pg_jieba,pg_trgm}')",
            "4",
        ),
        (
            "no-app-schema",
            "SELECT count(*) FROM information_schema.schemata"
            " WHERE schema_name NOT IN ('pg_catalog','information_schema','pg_toast')"
            " AND schema_name NOT LIKE 'pg_%'",
            "1",
        ),
        (
            "no-ledger",
            "SELECT to_regclass('dlightrag_schema_migrations') IS NOT NULL",
            "f",
        ),
    ]
    for label, query, expected in verification_checks:
        result = _psql(target, f"SELECT ({query}) AS value", project=project)
        if result.returncode != 0:
            report.fail(f"verify-{label}", result.stderr.strip())
            continue
        actual = result.stdout.strip()
        if actual != expected:
            report.fail(f"verify-{label}", f"expected {expected!r}, got {actual!r}")
            continue
        report.record(f"verify-{label}", actual)

    services = _compose("ps", "--format", "{{.Service}}", project=project)
    if services.returncode != 0:
        report.fail(
            "compose-services",
            f"docker compose ps failed: {services.stderr.strip() or services.stdout.strip()}",
        )
        return
    running = {line.strip() for line in (services.stdout or "").splitlines() if line.strip()}
    unexpected = running - {"postgres"}
    if unexpected:
        report.fail("compose-services", f"unexpected services running: {sorted(unexpected)}")
    else:
        report.record("compose-services", "only postgres is running")


def _wait_for_postgres_health(
    report: ResetReport,
    *,
    timeout_seconds: int = 180,
    project: str = _DEFAULT_COMPOSE_PROJECT,
) -> bool:
    import time

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        argv = [
            "docker",
            "inspect",
            "--format",
            "{{.State.Health.Status}}",
            f"{project}-postgres-1",
        ]
        inspect = subprocess.run(  # noqa: S603 - fixed docker argv, host ops tool
            argv,
            capture_output=True,
            text=True,
        )
        if inspect.returncode == 0 and inspect.stdout.strip() == "healthy":
            return True
        time.sleep(3)
    report.fail("postgres-health", "postgres did not become healthy before timeout")
    return False


def _psql(
    target: PostgresTarget,
    query: str,
    *,
    project: str = _DEFAULT_COMPOSE_PROJECT,
) -> subprocess.CompletedProcess[str]:
    env = {"PGPASSWORD": target.password}
    argv = [
        "docker",
        "compose",
        "-p",
        project,
        "-f",
        _COMPOSE_FILE,
        "exec",
        "-T",
        "postgres",
        "psql",
        "-U",
        target.user,
        "-d",
        target.database,
        "-tAc",
        query,
    ]
    return subprocess.run(  # noqa: S603 - fixed docker compose exec argv, host ops tool
        argv,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, **env},
    )


# ─────────────────────────────────────────────────────────────────
# Native mode
# ─────────────────────────────────────────────────────────────────


async def _native_pg_work(
    target: PostgresTarget,
    working_dir: Path,
    report: ResetReport,
    *,
    force_disconnect: bool,
    dry_run: bool,
) -> None:
    try:
        conn = await asyncpg.connect(
            host=target.host,
            port=target.port,
            user=target.user,
            password=target.password,
            database=target.database,
        )
    except (OSError, asyncpg.PostgresError) as exc:
        report.fail("connect", f"cannot connect to {target.identity()}: {exc}")
        return
    try:
        sessions = await _other_sessions(conn, target.database)
        report.record(
            "sessions",
            f"{len(sessions)} other session(s) on database {target.database}",
        )
        for session in sessions:
            report.record(
                "session",
                f"pid={session['pid']} user={session['usename']}"
                f" application={session.get('application_name') or '-'}",
            )
        if sessions and not dry_run:
            if not force_disconnect:
                report.fail(
                    "sessions",
                    "other sessions are active; pass --force-disconnect to terminate them",
                )
                return
            for session in sessions:
                await conn.execute(
                    "SELECT pg_terminate_backend($1::int)",
                    int(session["pid"]),  # type: ignore[arg-type]
                )
            report.record("sessions", "terminated active sessions")

        if dry_run:
            schemas = await conn.fetch(
                "SELECT schema_name FROM information_schema.schemata"
                " WHERE schema_name NOT IN ('pg_catalog','information_schema','pg_toast')"
                " AND schema_name NOT LIKE 'pg_%'"
            )
            report.record(
                "dry-run-schemas",
                f"schemas that would be dropped: {sorted(row['schema_name'] for row in schemas)}",
            )
            report.record(
                "dry-run-ddl",
                "DROP SCHEMA public CASCADE; CREATE SCHEMA public; then"
                " CREATE EXTENSION IF NOT EXISTS vector, pg_textsearch, pg_jieba, pg_trgm",
            )
            return

        try:
            await conn.execute("DROP SCHEMA public CASCADE")
        except asyncpg.InvalidSchemaNameError:
            pass  # already empty: reruns converge
        except asyncpg.PostgresError as exc:
            report.fail("drop-schema", str(exc))
            return
        await conn.execute("CREATE SCHEMA public")
        report.record("drop-schema", "replaced the public schema in place")

        for extension in _REQUIRED_EXTENSIONS:
            try:
                await conn.execute(f"CREATE EXTENSION IF NOT EXISTS {extension}")
            except asyncpg.PostgresError as exc:
                report.fail("extensions", f"cannot create {extension}: {exc}")
                return
        report.record("extensions", f"recreated {', '.join(_REQUIRED_EXTENSIONS)}")

        violations = await _verify_empty_postgres(conn)
        for violation in violations:
            report.fail("verify", violation)
        if not violations:
            report.record("verify", "only public remains; ledger absent; extensions present")
    finally:
        await conn.close()


def clear_working_dir_children(working_dir: Path, report: ResetReport) -> None:
    """Delete only the children of the verified working-directory root."""
    if not working_dir.exists():
        working_dir.mkdir(parents=True, exist_ok=True)
        report.record("working-dir", f"recreated empty root {working_dir}")
        return
    if not working_dir.is_dir():
        report.fail("working-dir", f"root is not a directory: {working_dir}")
        return
    for child in sorted(working_dir.iterdir()):
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
            report.record("working-dir", f"removed {child}")
        except OSError as exc:
            report.fail("working-dir", f"cannot remove {child}: {exc}")


def clear_runtime_dirs(working_dir: Path, workspace_root: Path | None, report: ResetReport) -> None:
    """Clear the working directory and the optional Agent Workspace root."""
    clear_working_dir_children(working_dir, report)
    for violation in verify_working_dir_empty(working_dir):
        report.fail("verify-working-dir", violation)
    if workspace_root is not None:
        clear_working_dir_children(workspace_root, report)


def verify_working_dir_empty(working_dir: Path) -> list[str]:
    violations: list[str] = []
    if working_dir.exists():
        children = list(working_dir.iterdir())
        if children:
            violations.append(f"working directory not empty: {[c.name for c in children]}")
    else:
        violations.append(f"working-directory root missing: {working_dir}")
    return violations


# ─────────────────────────────────────────────────────────────────
# Preview and CLI
# ─────────────────────────────────────────────────────────────────


def preview(
    *,
    mode: str,
    target: PostgresTarget,
    working_dir: Path,
    workspace_root: Path | None,
    repo_root: Path,
    compose_project: str,
) -> None:
    print("Development reset preview")
    print(f"  mode:                 {mode}")
    if mode == "docker":
        print(f"  compose project:      {compose_project}")
        print(f"  compose file:         {_REPO_ROOT / _COMPOSE_FILE}")
        print(f"  volumes to delete:    {', '.join(_COMPOSE_VOLUMES)}")
        print("  post-reset services:  postgres only (healthy, extensions, empty schema)")
    else:
        print(f"  database:             {target.identity()}")
        print("  ddl:                  DROP SCHEMA public CASCADE; CREATE SCHEMA public")
        print(
            f"  extensions:           {', '.join(_REQUIRED_EXTENSIONS)}"
            " (CREATE EXTENSION IF NOT EXISTS)"
        )
    print(f"  working-dir root:     {working_dir}")
    print(f"  Agent Workspace root: {workspace_root or '(disabled)'}")
    if working_dir.is_dir():
        children = list(working_dir.iterdir())
        print(f"  working-dir children: {len(children)}")
        for child in sorted(children):
            print(f"    - {child}")
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reset_development.py",
        description=(
            "Reset the complete DlightRAG development environment. Two explicit modes: "
            "docker (delete both Compose volumes, start only PostgreSQL) or native "
            "(replace the dedicated database's public schema and empty the working "
            "directory). There is no partial reset scope; workspace reset is the "
            "separate scripts/reset_workspace.py product command."
        ),
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["docker", "native"],
        help="reset mode; never auto-detected",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="read-only preview: resolve, validate, and report without any mutation",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the interactive confirmation prompt; target validation still runs",
    )
    parser.add_argument(
        "--force-disconnect",
        action="store_true",
        help="native only: terminate other sessions on the target database",
    )
    parser.add_argument(
        "--allow-remote-reset",
        action="store_true",
        help="native only: allow a non-loopback database host",
    )
    parser.add_argument("--verbose", action="store_true", help="verbose step reporting")
    return parser


def _print_report(report: ResetReport, *, verbose: bool) -> None:
    for step, detail in report.steps:
        if verbose or step.startswith("verify"):
            print(f"  [{step}] {detail}")
    if report.failures:
        print("\nFailures:")
        for failure in report.failures:
            print(f"  - {failure}")


def confirm_database_name(
    database: str,
    *,
    yes: bool,
    input_fn=input,  # noqa: A002 - injected prompt source for tests
) -> bool:
    """Confirm a destructive reset by typing the exact database name.

    ``--yes`` skips the prompt but never bypasses target validation.
    """
    if yes:
        return True
    confirmed = input_fn(f'Type the database name "{database}" to confirm the reset: ').strip()
    return confirmed == database


async def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_root = _REPO_ROOT
    env = _read_env(repo_root)
    report = ResetReport(mode=args.mode)
    try:
        config = _read_config(repo_root)
        target = resolve_postgres_target(env, config)
        working_dir = _working_dir_root(repo_root, env, config)
        workspace_root = _workspace_root(repo_root, env, config)
    except (OSError, TypeError, ValueError, YAMLError) as exc:
        report.fail("configuration", str(exc))
        _print_report(report, verbose=True)
        return 1
    compose_project = env.get("COMPOSE_PROJECT_NAME") or _DEFAULT_COMPOSE_PROJECT

    preview(
        mode=args.mode,
        target=target,
        working_dir=working_dir,
        workspace_root=workspace_root,
        repo_root=repo_root,
        compose_project=compose_project,
    )

    violations = validate_runtime_roots(working_dir, workspace_root, repo_root)
    if args.mode == "native":
        for violation in validate_native_target(
            target=target,
            working_dir=working_dir,
            repo_root=repo_root,
            allow_remote_reset=args.allow_remote_reset,
        ):
            if violation not in violations:
                violations.append(violation)
    for violation in violations:
        report.fail("target", violation)
    if report.failures:
        _print_report(report, verbose=True)
        return 1

    if args.mode == "native":
        if args.dry_run:
            await _native_pg_work(
                target,
                working_dir,
                report,
                force_disconnect=args.force_disconnect,
                dry_run=True,
            )
            _print_report(report, verbose=True)
            return 1 if report.failures else 0
        if report.failures:
            _print_report(report, verbose=args.verbose)
            return 1
        if not confirm_database_name(target.database, yes=args.yes):
            report.fail("confirmation", "database name did not match; aborted")
            _print_report(report, verbose=args.verbose)
            return 1
        report.record("confirmation", "target confirmed")
        # Destructive work starts only after confirmation.
        await _native_pg_work(
            target,
            working_dir,
            report,
            force_disconnect=args.force_disconnect,
            dry_run=False,
        )
        # Independent cleanup steps continue past earlier failures so one
        # failure never hides the rest; every failure is reported.
        clear_runtime_dirs(working_dir, workspace_root, report)
        _print_report(report, verbose=args.verbose)
        print("Development reset complete." if report.ok else "Development reset FAILED.")
        return 0 if report.ok else 1

    if args.dry_run:
        _print_report(report, verbose=True)
        return 0

    if not confirm_database_name(target.database, yes=args.yes):
        report.fail("confirmation", "database name did not match; aborted")
        _print_report(report, verbose=args.verbose)
        return 1
    report.record("confirmation", "target confirmed")

    run_docker_reset(target, report, compose_project=compose_project)
    # Every invocation also clears the pre-validated host runtime roots so no
    # half-reset environment can survive.
    if not report.failures:
        clear_runtime_dirs(working_dir, workspace_root, report)
    _print_report(report, verbose=args.verbose)
    print("Development reset complete." if report.ok else "Development reset FAILED.")
    return 0 if report.ok else 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("Aborted by user.", file=sys.stderr)
        sys.exit(130)
