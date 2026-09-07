# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Behavioral contract for the transport-neutral Access module."""

from collections.abc import Sequence
from typing import Any

import pytest

from dlightrag.application.access import (
    DEPLOYMENT_OWNER_ID,
    AccessAction,
    AccessDeniedError,
    AccessGate,
    UserContext,
    WorkspaceRecord,
    access_control_from_settings,
    corpus_mutation_access_action,
    owner_id_from_user,
)
from dlightrag.application.config import (
    AccessControlConfig,
    AccessControlRuleConfig,
    DlightragConfig,
)
from dlightrag.application.settings import access_settings
from tests.config_helpers import mutate_config, replace_config


class _WorkspaceCatalog:
    async def alist_workspace_records(self) -> list[WorkspaceRecord]:
        return [
            {"workspace": "finance", "display_name": "Finance"},
            {"workspace": "legal", "display_name": "Legal"},
        ]


class _FinanceOnlyAccess:
    async def check(self, subject: Any, action: str, *, workspace: str | None = None) -> None:
        raise AssertionError("all-workspace expansion must use the filtered catalog")

    async def filter_workspaces(
        self,
        subject: Any,
        action: str,
        workspaces: Sequence[str],
    ) -> list[str]:
        assert action == AccessAction.WORKSPACE_QUERY
        return [workspace for workspace in workspaces if workspace == "finance"]


async def test_all_workspaces_expands_only_to_authorized_catalog_entries() -> None:
    gate = AccessGate(
        _FinanceOnlyAccess(),
        UserContext(user_id="alice", auth_mode="jwt"),
    )
    resolved = await gate.resolve_query_workspaces(
        _WorkspaceCatalog(),
        default_workspace="legal",
        workspaces=None,
        all_workspaces=True,
    )

    assert resolved == ["finance"]


async def test_allow_all_access_control_is_default(test_config: DlightragConfig) -> None:
    access_control = access_control_from_settings(access_settings(test_config))

    await access_control.check(
        UserContext(user_id="anonymous", auth_mode="none"),
        AccessAction.WORKSPACE_RESET,
        workspace="finance",
    )


async def test_jwt_claims_access_control_matches_claim_workspace_and_action(
    test_config: DlightragConfig,
) -> None:
    mutate_config(test_config, "access.auth_mode", "jwt")
    mutate_config(test_config, "access.jwt_verification_key", "test-key")
    test_config = replace_config(
        test_config,
        "access.control",
        AccessControlConfig(
            mode="jwt_claims",
            rules=[
                AccessControlRuleConfig(
                    claim="groups",
                    value="finance-rag-readers",
                    workspaces=["Finance Reports"],
                    actions=["workspace.query", "workspace.list_files"],
                )
            ],
        ),
    )
    access_control = access_control_from_settings(access_settings(test_config))
    user = UserContext(
        user_id="alice",
        auth_mode="jwt",
        claims={"groups": ["finance-rag-readers"]},
    )

    await access_control.check(user, AccessAction.WORKSPACE_QUERY, workspace="finance_reports")
    assert await access_control.filter_workspaces(
        user,
        AccessAction.WORKSPACE_QUERY,
        ["finance_reports", "legal"],
    ) == ["finance_reports"]

    with pytest.raises(AccessDeniedError):
        await access_control.check(user, AccessAction.WORKSPACE_RESET, workspace="finance_reports")


def _preset_access_control(preset: str, test_config: DlightragConfig):
    mutate_config(test_config, "access.auth_mode", "jwt")
    mutate_config(test_config, "access.jwt_verification_key", "test-key")
    test_config = replace_config(
        test_config,
        "access.control",
        AccessControlConfig(
            mode="jwt_claims",
            rules=[
                AccessControlRuleConfig(
                    claim="roles",
                    value=f"finance.{preset}",
                    workspaces=["finance"],
                    actions=[preset],
                )
            ],
        ),
    )
    user = UserContext(
        user_id="alice",
        auth_mode="jwt",
        claims={"roles": [f"finance.{preset}"]},
    )
    return access_control_from_settings(access_settings(test_config)), user


async def test_reader_preset_allows_reads_and_denies_writes(
    test_config: DlightragConfig,
) -> None:
    access_control, user = _preset_access_control("reader", test_config)

    await access_control.check(user, AccessAction.WORKSPACE_QUERY, workspace="finance")
    await access_control.check(user, AccessAction.WORKSPACE_READ_METADATA, workspace="finance")
    with pytest.raises(AccessDeniedError):
        await access_control.check(user, AccessAction.WORKSPACE_INGEST, workspace="finance")
    with pytest.raises(AccessDeniedError):
        await access_control.check(user, AccessAction.WORKSPACE_RESET, workspace="finance")


async def test_editor_preset_allows_corpus_edits_but_not_workspace_admin(
    test_config: DlightragConfig,
) -> None:
    access_control, user = _preset_access_control("editor", test_config)

    await access_control.check(user, AccessAction.WORKSPACE_QUERY, workspace="finance")
    await access_control.check(user, AccessAction.WORKSPACE_INGEST, workspace="finance")
    await access_control.check(user, AccessAction.WORKSPACE_DELETE_FILES, workspace="finance")
    with pytest.raises(AccessDeniedError):
        await access_control.check(user, AccessAction.WORKSPACE_RESET, workspace="finance")
    with pytest.raises(AccessDeniedError):
        await access_control.check(user, AccessAction.MODEL_CATALOGUE_WRITE)


async def test_admin_preset_allows_every_action(test_config: DlightragConfig) -> None:
    access_control, user = _preset_access_control("admin", test_config)

    for action in (
        AccessAction.WORKSPACE_QUERY,
        AccessAction.WORKSPACE_INGEST,
        AccessAction.WORKSPACE_CREATE,
        AccessAction.WORKSPACE_RESET,
        AccessAction.MODEL_CATALOGUE_WRITE,
    ):
        await access_control.check(user, action, workspace="finance")


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("ingest", AccessAction.WORKSPACE_INGEST),
        ("replace", AccessAction.WORKSPACE_INGEST),
        ("retry", AccessAction.WORKSPACE_INGEST),
        ("delete", AccessAction.WORKSPACE_DELETE_FILES),
        ("reset", AccessAction.WORKSPACE_RESET),
        ("unknown", AccessAction.WORKSPACE_RESET),
    ],
)
def test_corpus_mutation_action_mapping_fails_closed(mutation: str, expected: str) -> None:
    assert corpus_mutation_access_action(mutation) == expected


async def test_workspace_wildcard_rule_matches_any_canonical_workspace(
    test_config: DlightragConfig,
) -> None:
    test_config = replace_config(
        test_config,
        "access.control",
        AccessControlConfig(
            mode="jwt_claims",
            rules=[
                AccessControlRuleConfig(
                    claim="roles",
                    value="reader",
                    workspaces=["*"],
                    actions=[AccessAction.WORKSPACE_QUERY],
                )
            ],
        ),
    )
    access_control = access_control_from_settings(access_settings(test_config))
    user = UserContext(user_id="alice", auth_mode="jwt", claims={"roles": ["reader"]})

    await access_control.check(user, AccessAction.WORKSPACE_QUERY, workspace="finance_reports")
    await access_control.check(user, AccessAction.WORKSPACE_QUERY, workspace="legal")


def test_jwt_principal_uses_trust_domain_and_subject() -> None:
    alice = UserContext(
        user_id="alice",
        auth_mode="jwt",
        claims={"iss": "https://issuer.example"},
    )
    bob = UserContext(
        user_id="bob",
        auth_mode="jwt",
        claims={"iss": "https://issuer.example"},
    )

    assert owner_id_from_user(alice) == owner_id_from_user(alice)
    assert owner_id_from_user(alice) != owner_id_from_user(bob)


def test_simple_and_none_principals_are_deployment_scoped() -> None:
    first = UserContext(user_id="header-a", auth_mode="simple")
    second = UserContext(user_id="header-b", auth_mode="simple")

    assert owner_id_from_user(first) == owner_id_from_user(second)
    assert owner_id_from_user(None) != owner_id_from_user(first)


def test_direct_in_process_calls_share_the_deployment_owner() -> None:
    assert owner_id_from_user(None) == DEPLOYMENT_OWNER_ID
    assert owner_id_from_user(UserContext(user_id="anonymous", auth_mode="none")) == (
        DEPLOYMENT_OWNER_ID
    )
