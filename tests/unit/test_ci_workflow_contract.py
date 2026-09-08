# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Contracts for automatic pull-request CI gates."""

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

_CI_WORKFLOW = Path(".github/workflows/ci.yml")
_DURABLE_PG_SUITES = {
    "tests/integration/test_agent_session_pg.py",
    "tests/integration/test_answer_runs_pg.py",
    "tests/integration/test_answer_run_api_pg.py",
    "tests/integration/test_answer_run_coordinator_pg.py",
    "tests/integration/test_memory_pg.py",
    "tests/integration/test_pg_storage.py",
    "tests/integration/test_run_cancellation_pg.py",
    "tests/integration/test_retrieval_runs_pg.py",
    "tests/integration/test_corpus_mutation_runs_pg.py",
    "tests/integration/test_corpus_mutation_phase_faults_pg.py",
    "tests/integration/test_run_runtime_lane_independence_pg.py",
    "tests/integration/test_run_runtime_query_plans_pg.py",
}
_REQUIRED_ENV = {
    "PGHOST": "localhost",
    "PGPORT": "5432",
    "PGUSER": "dlightrag",
    "PGPASSWORD": "dlightrag",
    "PGDATABASE": "dlightrag",
}
_PG_CONNECTION_SUITES = {
    "tests.integration.test_development_reset_pg": (
        "tests/integration/test_development_reset_pg.py",
        "_ADMIN",
    ),
    "tests.integration.test_metadata_scope_pg": (
        "tests/integration/test_metadata_scope_pg.py",
        "_DEFAULT_KWARGS",
    ),
    "tests.integration.test_model_catalogue_pg": (
        "tests/integration/test_model_catalogue_pg.py",
        "_PG_CONN_KWARGS",
    ),
    "tests.integration.test_partition_foundation_pg": (
        "tests/integration/test_partition_foundation_pg.py",
        "_DEFAULT_KWARGS",
    ),
    "tests.integration.test_promotion_foundation_pg": (
        "tests/integration/test_promotion_foundation_pg.py",
        "_DEFAULT_KWARGS",
    ),
    "tests.integration.test_promotion_worker_pg": (
        "tests/integration/test_promotion_worker_pg.py",
        "_DEFAULT_KWARGS",
    ),
    "tests.integration.test_reader_role_pg": (
        "tests/integration/test_reader_role_pg.py",
        "_PG_CONN_KWARGS",
    ),
    "tests.integration.test_web_answer_runs_pg": (
        "tests/integration/test_web_answer_runs_pg.py",
        "_PG_CONN_KWARGS",
    ),
}
_PG_OVERRIDE_ENV = {
    "PGHOST": "198.51.100.7",
    "PGPORT": "6543",
    "PGUSER": "isolation_user",
    "PGPASSWORD": "<redacted>",
    "PGDATABASE": "isolation_admin",
}


def _integration_job() -> dict[str, Any]:
    workflow = yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))
    return workflow["jobs"]["integration"]


def _browser_e2e_job() -> dict[str, Any]:
    workflow = yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))
    return workflow["jobs"]["browser-e2e"]


def _workflow_triggers() -> set[str]:
    workflow = yaml.load(_CI_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    return set(workflow["on"])


def _named_step(job: dict[str, Any], name: str) -> dict[str, Any]:
    return next(step for step in job["steps"] if step.get("name") == name)


def test_pg_integration_job_is_automatic_and_uses_an_ephemeral_pinned_database() -> None:
    job = _integration_job()
    start = _named_step(job, "Start clean PostgreSQL")
    cleanup = _named_step(job, "Remove PostgreSQL container")
    start_command = start["run"]

    assert {"pull_request", "push", "workflow_dispatch"} <= _workflow_triggers()
    assert "if" not in job
    assert re.fullmatch(
        r"ghcr\.io/\$\{\{ github\.repository_owner \}\}/"
        r"dlightrag-postgres@sha256:[0-9a-f]{64}",
        job["env"]["POSTGRES_IMAGE"],
    )
    assert {key: str(job["env"][key]) for key in _REQUIRED_ENV} == _REQUIRED_ENV
    assert "docker run --detach --name dlightrag-ci-postgres" in start_command
    assert "--rm" not in start_command
    assert "--volume" not in start_command
    assert "postgres -c shared_preload_libraries=pg_textsearch,pg_jieba" in start_command
    assert "docker inspect" in start_command
    assert ".State.Status" in start_command
    assert ".State.Health.Status" in start_command
    assert "docker logs dlightrag-ci-postgres || true" in start_command
    assert cleanup["if"] == "always()"
    assert "docker rm --force dlightrag-ci-postgres" in cleanup["run"]


def test_pg_integration_suites_share_environment_connection_contract() -> None:
    """Every normal live-PG target comes from the one environment-aware mapping."""
    suite_attributes = {
        module_name: attribute for module_name, (_path, attribute) in _PG_CONNECTION_SUITES.items()
    }
    script = f"""
import importlib

from tests.integration.pg_conn import PG_CONN_KWARGS

expected = {{
    "host": "{_PG_OVERRIDE_ENV["PGHOST"]}",
    "port": {_PG_OVERRIDE_ENV["PGPORT"]},
    "user": "{_PG_OVERRIDE_ENV["PGUSER"]}",
    "password": "{_PG_OVERRIDE_ENV["PGPASSWORD"]}",
    "database": "{_PG_OVERRIDE_ENV["PGDATABASE"]}",
}}
for module_name, attribute in {suite_attributes!r}.items():
    module = importlib.import_module(module_name)
    connection = getattr(module, attribute)
    if connection is not PG_CONN_KWARGS:
        raise AssertionError(f"{{module_name}}.{{attribute}} does not consume PG_CONN_KWARGS")
    for key, value in expected.items():
        if connection[key] != value:
            env_name = "PG" + key.upper()
            raise AssertionError(f"{{module_name}}.{{attribute}} ignores {{env_name}}")

development_reset = importlib.import_module(
    "tests.integration.test_development_reset_pg"
)
target = development_reset._test_target()
for key in ("host", "port", "user", "password"):
    if getattr(target, key) != expected[key]:
        env_name = "PG" + key.upper()
        raise AssertionError(f"development reset target ignores {{env_name}}")
if target.database != development_reset._TEST_DATABASE:
    raise AssertionError("development reset target ignores its isolated database")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        env={**os.environ, **_PG_OVERRIDE_ENV},
        capture_output=True,
        text=True,
        check=False,
    )
    stderr = completed.stderr.replace(_PG_OVERRIDE_ENV["PGPASSWORD"], "<redacted>")
    assert completed.returncode == 0, stderr

    for path, _attribute in _PG_CONNECTION_SUITES.values():
        tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)
        literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)}
        assert "localhost" not in literals, path
        assert "dlightrag" not in literals, path
        assert 5432 not in literals, path


def test_pg_integration_job_rejects_skips_without_external_evaluation() -> None:
    job = _integration_job()
    test_step = _named_step(job, "Run deterministic durable PostgreSQL integration tests")
    skip_guard = _named_step(job, "Reject skipped PostgreSQL integration tests")
    test_command = test_step["run"]
    guard_command = skip_guard["run"]
    selected_suites = set(re.findall(r"tests/integration/test_[a-z0-9_]+\.py", test_command))
    serialized_job = json.dumps(job).lower()

    assert selected_suites == _DURABLE_PG_SUITES
    assert "--junitxml=.test-results/postgres.xml" in test_command
    assert ".test-results/postgres.xml" in guard_command
    assert "tests > 0" in guard_command
    assert "skipped == 0" in guard_command
    for forbidden in (
        "ragas",
        "scripts/ragas_eval.py",
        "credentials",
        "secrets.",
        "github_token",
        "api_key",
        "openai",
        "anthropic",
    ):
        assert forbidden not in serialized_job


def test_browser_e2e_job_is_automatic_local_and_builds_the_frontend() -> None:
    job = _browser_e2e_job()
    steps = job["steps"]
    commands = [step["run"] for step in steps if "run" in step]
    test_step = _named_step(job, "Run mocked browser E2E tests")
    test_command = test_step["run"]
    serialized_job = json.dumps(job).lower()
    expected_setup = [
        "make sync-dev",
        "make frontend-install",
        "uv run playwright install --with-deps chromium",
        "make frontend-build",
    ]

    assert {"pull_request", "push", "workflow_dispatch"} <= _workflow_triggers()
    assert "if" not in job
    assert "needs" not in job
    assert "services" not in job
    assert "env" not in job
    assert any(str(step.get("uses", "")).startswith("actions/setup-node@") for step in steps)
    assert [commands.index(command) for command in expected_setup] == sorted(
        commands.index(command) for command in expected_setup
    )
    assert "tests/e2e" in test_command
    assert re.search(r"(?:^|\s)-m\s+e2e(?:\s|$)", test_command)
    assert "e2e_pg18" not in test_command
    assert "--junitxml=.test-results/browser-e2e.xml" in test_command
    assert test_step["env"] == {"DLIGHTRAG_E2E_ARTIFACT_DIR": ".test-results/browser-e2e"}
    for forbidden in (
        "postgres",
        "e2e_pg18",
        "dlightrag_run_e2e_pg18",
        "credentials",
        "secrets.",
        "github_token",
        "api_key",
        "openai",
        "anthropic",
    ):
        assert forbidden not in serialized_job


def test_browser_e2e_job_rejects_empty_or_skipped_results_and_uploads_diagnostics() -> None:
    job = _browser_e2e_job()
    guard = _named_step(job, "Reject empty or skipped browser E2E results")
    upload = _named_step(job, "Upload browser E2E diagnostics")
    guard_command = guard["run"]

    assert guard["if"] == "${{ always() && steps.browser-tests.outcome != 'skipped' }}"
    assert ".test-results/browser-e2e.xml" in guard_command
    assert "tests > 0" in guard_command
    assert "skipped == 0" in guard_command
    assert upload["if"] == "always()"
    assert str(upload["uses"]).startswith("actions/upload-artifact@")
    assert upload["with"]["path"] == ".test-results"
    assert upload["with"]["if-no-files-found"] == "warn"
    assert upload["with"]["include-hidden-files"] is True
    assert upload["with"]["retention-days"] == 14


def test_browser_e2e_harness_retains_failed_context_diagnostics() -> None:
    harness = Path("tests/e2e/conftest.py").read_text(encoding="utf-8")
    theme_tests = Path("tests/e2e/test_web_theme.py").read_text(encoding="utf-8")
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    assert "def e2e_browser_context(" in harness
    assert 'os.getenv("DLIGHTRAG_E2E_ARTIFACT_DIR")' in harness
    assert "context.tracing.start(" in harness
    assert "candidate.screenshot(" in harness
    assert "context.tracing.stop(" in harness
    assert "e2e_browser_context: BrowserContext" in theme_tests
    assert "browser.new_context(" not in theme_tests
    assert ".test-results/" in gitignore
