# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Startup validation for optional trusted local execution."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from dlightrag.application.config import AgentExecutionConfig, DlightragConfig
from dlightrag.engine.agent.environment import SandboxUnavailableError
from dlightrag.engine.answer.execution_settings import (
    default_local_workspace_root,
    validate_agent_execution,
)


def test_execution_environment_defaults_to_trust() -> None:
    assert AgentExecutionConfig().execution_environment == "trust"


def test_disabled_ignores_workspace_root(tmp_path: Path) -> None:
    config = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        deployment={
            "working_dir": str(tmp_path / "corpus"),
        },
        answer={
            "agent": AgentExecutionConfig(
                execution_environment="disabled", workspace_root=str(tmp_path / "ws")
            ),
        },
    )
    assert (
        validate_agent_execution(
            execution_environment=config.answer.agent.execution_environment,
            workspace_root=config.answer.agent.workspace_root,
            working_dir=config.deployment.working_dir,
        )
        is None
    )


def test_trust_without_root_uses_home_default(tmp_path: Path) -> None:
    resolved = validate_agent_execution(
        execution_environment="trust",
        workspace_root=None,
        working_dir=str(tmp_path / "corpus"),
    )
    assert resolved == default_local_workspace_root()
    assert resolved is not None
    assert resolved.is_dir()


def test_trust_rejects_a_relative_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute path"):
        validate_agent_execution(
            execution_environment="trust",
            workspace_root="relative/workspaces",
            working_dir=str(tmp_path / "corpus"),
        )


def test_sandbox_without_adapter_fails_explicitly(tmp_path: Path) -> None:
    with pytest.raises(SandboxUnavailableError, match="requires a configured sandbox adapter"):
        validate_agent_execution(
            execution_environment="sandbox",
            workspace_root=str(tmp_path / "sandbox"),
            working_dir=str(tmp_path / "corpus"),
        )


def test_workspace_root_must_not_overlap_working_dir(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    config = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        deployment={
            "working_dir": str(corpus),
        },
        answer={
            "agent": AgentExecutionConfig(
                execution_environment="trust", workspace_root=str(corpus / "nested")
            ),
        },
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_agent_execution(
            execution_environment=config.answer.agent.execution_environment,
            workspace_root=config.answer.agent.workspace_root,
            working_dir=config.deployment.working_dir,
        )


def test_outbound_mcp_requires_explicit_transport_shape() -> None:
    config = AgentExecutionConfig.model_validate(
        {
            "outbound_mcp": [
                {
                    "name": "docs",
                    "transport": "streamable-http",
                    "url": "https://mcp.example.test",
                    "tools": ["search"],
                }
            ]
        }
    )
    assert config.outbound_mcp[0].tools == ("search",)
    with pytest.raises(ValidationError, match="requires url"):
        AgentExecutionConfig.model_validate(
            {
                "outbound_mcp": [
                    {
                        "name": "docs",
                        "transport": "streamable-http",
                        "tools": ["search"],
                    }
                ]
            }
        )


def test_unknown_agent_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        DlightragConfig.model_validate(
            {"agent": {"execution_environment": "disabled", "sandbox": True}}
        )


def test_skills_root_defaults_to_none_and_accepts_absolute() -> None:
    assert AgentExecutionConfig().skills_root is None
    config = AgentExecutionConfig.model_validate({"skills_root": "/opt/dlightrag/skills"})
    assert config.skills_root == "/opt/dlightrag/skills"


def test_skills_root_rejects_a_relative_path() -> None:
    with pytest.raises(ValidationError, match="skill roots must be absolute paths"):
        AgentExecutionConfig.model_validate({"skills_root": "skills"})


def test_owner_skills_root_defaults_to_none_and_accepts_absolute() -> None:
    assert AgentExecutionConfig().owner_skills_root is None
    config = AgentExecutionConfig.model_validate(
        {"owner_skills_root": "/opt/dlightrag/owner_skills"}
    )
    assert config.owner_skills_root == "/opt/dlightrag/owner_skills"


def test_owner_skills_root_rejects_a_relative_path() -> None:
    with pytest.raises(ValidationError, match="skill roots must be absolute paths"):
        AgentExecutionConfig.model_validate({"owner_skills_root": "owner_skills"})


def test_search_tool_configuration_defaults_and_absolute_paths() -> None:
    default = AgentExecutionConfig()
    assert default.fd_path == "fd"
    assert default.ripgrep_path == "rg"
    assert default.search_tool_cache_root is None
    assert default.search_tool_auto_install is False

    configured = AgentExecutionConfig.model_validate(
        {
            "fd_path": "/opt/bin/fd",
            "ripgrep_path": "/opt/bin/rg",
            "search_tool_cache_root": "/var/cache/dlightrag-tools",
            "search_tool_auto_install": True,
        }
    )
    assert configured.search_tool_auto_install is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fd_path", "relative/bin/fd"),
        ("ripgrep_path", "relative/bin/rg"),
        ("search_tool_cache_root", "relative-cache"),
    ],
)
def test_search_tool_directories_must_be_absolute(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        AgentExecutionConfig.model_validate({field: value})
