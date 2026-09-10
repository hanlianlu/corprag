# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the full development reset command (host-side, repository-owned)."""

import importlib.util
import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

_reset_path = Path(__file__).resolve().parents[2] / "scripts" / "reset_development.py"
_spec = importlib.util.spec_from_file_location("reset_development_cli", _reset_path)
assert _spec is not None and _spec.loader is not None
_reset = importlib.util.module_from_spec(_spec)
sys.modules["reset_development_cli"] = _reset
_spec.loader.exec_module(_reset)


class TestParser:
    def test_mode_is_required_and_never_auto_detected(self) -> None:
        with pytest.raises(SystemExit):
            _reset.build_parser().parse_args([])

    def test_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(SystemExit):
            _reset.build_parser().parse_args(["--mode", "auto"])

    def test_help_exposes_no_workspace_selector(self, capsys: pytest.CaptureFixture[str]) -> None:
        import re

        with pytest.raises(SystemExit) as exc:
            _reset.build_parser().parse_args(["--help"])

        assert exc.value.code == 0
        help_text = capsys.readouterr().out
        assert "--workspace" not in help_text
        assert re.search(r"\s--all\s", help_text) is None
        assert "workspace reset is the separate" in help_text
        assert "--dry-run" in help_text
        assert "--force-disconnect" in help_text
        assert "--allow-remote-reset" in help_text


class TestConfirmation:
    def test_requires_the_exact_database_name(self) -> None:
        assert not _reset.confirm_database_name("dlightrag", yes=False, input_fn=lambda _: "wrong")
        assert _reset.confirm_database_name("dlightrag", yes=False, input_fn=lambda _: "dlightrag")
        assert _reset.confirm_database_name(
            "dlightrag",
            yes=False,
            input_fn=lambda _: " dlightrag ",  # stripped
        )

    def test_yes_skips_the_prompt_but_validation_still_runs(self) -> None:
        # --yes never bypasses target validation; it only skips the prompt.
        assert _reset.confirm_database_name("dlightrag", yes=True, input_fn=lambda _: "nope")


class TestNativeTargetValidation:
    def _target(self, **overrides: Any) -> Any:
        values = {"host": "localhost", "port": 5432, "user": "u", "password": "p", "database": "db"}
        values.update(overrides)
        return _reset.PostgresTarget(**values)

    def test_loopback_host_passes(self, tmp_path: Path) -> None:
        violations = _reset.validate_native_target(
            target=self._target(),
            working_dir=tmp_path / "dlightrag_storage",
            repo_root=tmp_path,
            allow_remote_reset=False,
        )
        assert violations == []

    def test_remote_host_requires_dangerous_override(self, tmp_path: Path) -> None:
        violations = _reset.validate_native_target(
            target=self._target(host="db.example.com"),
            working_dir=tmp_path / "dlightrag_storage",
            repo_root=tmp_path,
            allow_remote_reset=False,
        )
        assert any("non-loopback" in violation for violation in violations)
        assert (
            _reset.validate_native_target(
                target=self._target(host="db.example.com"),
                working_dir=tmp_path / "dlightrag_storage",
                repo_root=tmp_path,
                allow_remote_reset=True,
            )
            == []
        )

    def test_refuses_root_home_and_outside_repository(self, tmp_path: Path) -> None:
        root = tmp_path / "repo"
        root.mkdir()
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        violations = _reset.validate_native_target(
            target=self._target(),
            working_dir=Path("/"),
            repo_root=root,
            allow_remote_reset=False,
        )
        assert any("must not be /" in violation for violation in violations)
        violations = _reset.validate_native_target(
            target=self._target(),
            working_dir=Path.home(),
            repo_root=root,
            allow_remote_reset=False,
        )
        assert any("home" in violation for violation in violations)
        violations = _reset.validate_native_target(
            target=self._target(),
            working_dir=outside,
            repo_root=root,
            allow_remote_reset=False,
        )
        assert any("inside the repository root" in violation for violation in violations)
        violations = _reset.validate_native_target(
            target=self._target(),
            working_dir=root,
            repo_root=root,
            allow_remote_reset=False,
        )
        assert any("repository root" in violation for violation in violations)

    def test_refuses_symlink_root_and_children(self, tmp_path: Path) -> None:
        root = tmp_path / "repo"
        root.mkdir()
        real_dir = tmp_path / "real-storage"
        real_dir.mkdir()
        symlink = root / "dlightrag_storage"
        symlink.symlink_to(real_dir)
        violations = _reset.validate_native_target(
            target=self._target(),
            working_dir=symlink,
            repo_root=root,
            allow_remote_reset=False,
        )
        assert any("symbolic link" in violation for violation in violations)

        plain_root = tmp_path / "plain"
        plain_root.mkdir()
        (plain_root / "child-link").symlink_to(real_dir)
        violations = _reset.validate_native_target(
            target=self._target(),
            working_dir=plain_root,
            repo_root=tmp_path,
            allow_remote_reset=False,
        )
        assert any("first-level child is a symbolic link" in violation for violation in violations)


class TestWorkingDirectory:
    def test_clears_children_but_never_the_root(self, tmp_path: Path) -> None:
        root = tmp_path / "storage"
        root.mkdir()
        (root / "keep").write_text("data")
        (root / "sub").mkdir()
        (root / "sub" / "file").write_text("nested")

        report = _reset.ResetReport(mode="native")
        _reset.clear_working_dir_children(root, report)

        assert report.ok
        assert root.exists()
        assert list(root.iterdir()) == []
        assert _reset.verify_working_dir_empty(root) == []

    def test_clearing_failures_are_reported_and_verification_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "storage"
        root.mkdir()
        (root / "stuck").mkdir()
        (root / "file").write_text("x")

        def raising_rmtree(path, **kwargs):
            raise OSError("simulated removal failure")

        monkeypatch.setattr(_reset.shutil, "rmtree", raising_rmtree)
        report = _reset.ResetReport(mode="native")
        _reset.clear_working_dir_children(root, report)

        assert report.failures
        assert _reset.verify_working_dir_empty(root) != []

    def test_missing_root_is_recreated(self, tmp_path: Path) -> None:
        root = tmp_path / "new-storage"
        report = _reset.ResetReport(mode="native")
        _reset.clear_working_dir_children(root, report)
        assert root.is_dir()
        assert _reset.verify_working_dir_empty(root) == []


class TestSettingsResolution:
    def test_read_env_overlays_dotenv_and_environment(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text(
            "DLIGHTRAG_STORAGE__POSTGRES__HOST=env-host\n"
            "DLIGHTRAG_STORAGE__POSTGRES__PORT=5544\n"
            "# a comment\n"
            "DLIGHTRAG_STORAGE__POSTGRES__DATABASE=env-db\n"
        )
        env = _reset._read_env(tmp_path)
        assert env["DLIGHTRAG_STORAGE__POSTGRES__HOST"] == "env-host"
        assert env["DLIGHTRAG_STORAGE__POSTGRES__PORT"] == "5544"
        assert env["DLIGHTRAG_STORAGE__POSTGRES__DATABASE"] == "env-db"

    def test_postgres_target_uses_canonical_nested_environment(self) -> None:
        target = _reset.resolve_postgres_target(
            {
                "DLIGHTRAG_STORAGE__POSTGRES__HOST": "db.internal",
                "DLIGHTRAG_STORAGE__POSTGRES__PORT": "5544",
                "DLIGHTRAG_STORAGE__POSTGRES__USER": "operator",
                "DLIGHTRAG_STORAGE__POSTGRES__PASSWORD": "secret",
                "DLIGHTRAG_STORAGE__POSTGRES__DATABASE": "corpus",
            }
        )

        assert target == _reset.PostgresTarget(
            host="db.internal",
            port=5544,
            user="operator",
            password="secret",
            database="corpus",
        )

    def test_postgres_target_falls_back_to_canonical_yaml(self) -> None:
        target = _reset.resolve_postgres_target(
            {},
            {
                "storage": {
                    "postgres": {
                        "host": "yaml.internal",
                        "port": 6432,
                        "user": "yaml-user",
                        "password": "yaml-secret",
                        "database": "yaml-db",
                    }
                }
            },
        )

        assert target == _reset.PostgresTarget(
            host="yaml.internal",
            port=6432,
            user="yaml-user",
            password="yaml-secret",
            database="yaml-db",
        )

    def test_disabled_agent_execution_never_returns_a_cleanup_root(self, tmp_path: Path) -> None:
        config = {
            "answer": {
                "agent": {
                    "execution_environment": "disabled",
                    "workspace_root": "/",
                }
            }
        }
        assert _reset._workspace_root(tmp_path, {}, config) is None

    def test_agent_cleanup_root_rejects_root_home_and_repo_ancestors(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        for unsafe in (Path("/"), Path.home(), tmp_path, repo):
            violations = _reset.validate_workspace_root(unsafe, repo)
            assert violations, unsafe

    def test_agent_cleanup_root_requires_an_absolute_configured_path(self, tmp_path: Path) -> None:
        config = {
            "answer": {
                "agent": {
                    "execution_environment": "trust",
                    "workspace_root": "relative/workspaces",
                }
            }
        }

        with pytest.raises(ValueError, match="must be an absolute path"):
            _reset._workspace_root(tmp_path, {}, config)

    def test_read_config_uses_yaml_1_2_scalar_resolution(self, tmp_path: Path) -> None:
        (tmp_path / "config.yaml").write_text("deployment:\n  workspace: off\n")

        config = _reset._read_config(tmp_path)

        assert config["deployment"] == {"workspace": "off"}

    def test_working_dir_root_falls_back_to_repo_default(self, tmp_path: Path) -> None:
        (tmp_path / "config.yaml").write_text("answer:\n  max_images: 4\n")
        root = _reset._working_dir_root(tmp_path, {})
        assert root == (tmp_path / "dlightrag_storage").resolve()

    def test_working_dir_root_resolves_config_value(self, tmp_path: Path) -> None:
        (tmp_path / "config.yaml").write_text("deployment:\n  working_dir: ./custom_storage\n")
        root = _reset._working_dir_root(tmp_path, {})
        assert root == (tmp_path / "custom_storage").resolve()


class TestDockerMode:
    def test_docker_reset_verifies_empty_database(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []

        def fake_compose(*args: str, **_kwargs):
            calls.append(" ".join(args))
            result = Mock()
            result.returncode = 0
            if args[:2] == ("ps", "--format"):
                result.stdout = "postgres\n"
            else:
                result.stdout = ""
                result.stderr = ""
            return result

        def fake_psql(target, query, **_kwargs):
            calls.append(f"psql:{query}")
            result = Mock()
            result.returncode = 0
            if "extname" in query:
                result.stdout = "4"
            elif "schemata" in query:
                result.stdout = "1"
            elif "to_regclass" in query:
                result.stdout = "f"
            else:
                result.stdout = "1"
            result.stderr = ""
            return result

        monkeypatch.setattr(_reset, "_compose", fake_compose)
        monkeypatch.setattr(_reset, "_psql", fake_psql)
        monkeypatch.setattr(_reset, "_wait_for_postgres_health", lambda report, **_kwargs: True)

        report = _reset.ResetReport(mode="docker")
        _reset.run_docker_reset(
            _reset.PostgresTarget(
                host="localhost", port=5432, user="u", password="p", database="db"
            ),
            report,
        )

        assert report.ok
        assert any("down -v" in call for call in calls)
        assert any("up -d postgres" in call for call in calls)
        assert any("verify-extensions" in step for step, _ in report.steps)
        assert any("verify-no-app-schema" in step for step, _ in report.steps)
        assert any("verify-no-ledger" in step for step, _ in report.steps)

    def test_docker_reset_fails_when_verification_values_are_wrong(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_compose(*args: str, **_kwargs):
            result = Mock(returncode=0, stdout="", stderr="")
            if args[:2] == ("ps", "--format"):
                result.stdout = "postgres\n"
            return result

        def fake_psql(_target, query, **_kwargs):
            result = Mock(returncode=0, stderr="")
            result.stdout = "1" if "SELECT 1" in query else "wrong"
            return result

        monkeypatch.setattr(_reset, "_compose", fake_compose)
        monkeypatch.setattr(_reset, "_psql", fake_psql)
        monkeypatch.setattr(_reset, "_wait_for_postgres_health", lambda report, **_kwargs: True)

        report = _reset.ResetReport(mode="docker")
        _reset.run_docker_reset(
            _reset.PostgresTarget(
                host="localhost", port=5432, user="u", password="p", database="db"
            ),
            report,
        )

        assert not report.ok
        assert any("expected" in failure for failure in report.failures)

    def test_docker_reset_preserves_compose_ps_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_compose(*args: str, **_kwargs):
            if args[:2] == ("ps", "--format"):
                return Mock(returncode=1, stdout="", stderr="daemon unavailable")
            return Mock(returncode=0, stdout="", stderr="")

        def fake_psql(_target, query, **_kwargs):
            if "extname" in query:
                value = "3"
            elif "schemata" in query:
                value = "1"
            elif "to_regclass" in query:
                value = "f"
            else:
                value = "1"
            return Mock(returncode=0, stdout=value, stderr="")

        monkeypatch.setattr(_reset, "_compose", fake_compose)
        monkeypatch.setattr(_reset, "_psql", fake_psql)
        monkeypatch.setattr(_reset, "_wait_for_postgres_health", lambda report, **_kwargs: True)
        report = _reset.ResetReport(mode="docker")

        _reset.run_docker_reset(
            _reset.PostgresTarget(
                host="localhost", port=5432, user="u", password="p", database="db"
            ),
            report,
        )

        assert not report.ok
        assert any("daemon unavailable" in failure for failure in report.failures)

    def test_docker_reset_reports_unexpected_services(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_compose(*args: str, **_kwargs):
            result = Mock()
            result.returncode = 0
            result.stdout = "postgres\ndlightrag-api\n"
            result.stderr = ""
            return result

        def fake_psql(target, query, **_kwargs):
            result = Mock()
            result.returncode = 0
            result.stdout = "0"
            result.stderr = ""
            return result

        monkeypatch.setattr(_reset, "_compose", fake_compose)
        monkeypatch.setattr(_reset, "_psql", fake_psql)
        monkeypatch.setattr(_reset, "_wait_for_postgres_health", lambda report, **_kwargs: True)

        report = _reset.ResetReport(mode="docker")
        _reset.run_docker_reset(
            _reset.PostgresTarget(
                host="localhost", port=5432, user="u", password="p", database="db"
            ),
            report,
        )
        assert not report.ok
        assert any("dlightrag-api" in failure for failure in report.failures)

    async def test_docker_mode_validates_paths_before_compose_mutation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called = False

        def destructive_reset(*_args, **_kwargs):
            nonlocal called
            called = True

        monkeypatch.setattr(_reset, "_REPO_ROOT", tmp_path)
        monkeypatch.setattr(_reset, "_working_dir_root", lambda *_args: tmp_path)
        monkeypatch.setattr(_reset, "run_docker_reset", destructive_reset)

        assert await _reset.main(["--mode", "docker", "--yes"]) == 1
        assert called is False

    def test_docker_preview_lists_every_deleted_volume(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _reset.preview(
            mode="docker",
            target=_reset.PostgresTarget(
                host="localhost", port=5432, user="u", password="p", database="db"
            ),
            working_dir=tmp_path / "storage",
            workspace_root=tmp_path / "agent-workspaces",
            repo_root=tmp_path,
            compose_project="custom-project",
        )
        output = capsys.readouterr().out
        assert "custom-project" in output
        assert "dlightrag_agent_workspaces" in output


class TestSeparationFromWorkspaceReset:
    def test_neither_reset_module_imports_the_other(self) -> None:
        development = _reset_path.read_text(encoding="utf-8")
        workspace = (
            Path(__file__).resolve().parents[2] / "scripts" / "reset_workspace.py"
        ).read_text(encoding="utf-8")
        assert "import reset_workspace" not in development
        assert "from reset_workspace" not in development
        assert "import reset_development" not in workspace
        assert "from reset_development" not in workspace
        # The development tool imports no product code at all.
        assert "from dlightrag" not in development
        assert "import dlightrag" not in development
