# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Authorization policy for DlightRAG product resources."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol


class Principal(Protocol):
    """Authenticated facts required by authorization policy."""

    @property
    def auth_mode(self) -> str: ...

    @property
    def claims(self) -> Mapping[str, object]: ...


type AccessSubject = Principal | None


@dataclass(frozen=True, slots=True)
class AccessRule:
    claim: str
    value: str
    workspaces: tuple[str, ...]
    actions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AccessSettings:
    mode: Literal["allow_all", "jwt_claims"] = "allow_all"
    rules: tuple[AccessRule, ...] = ()


class AccessAction:
    WORKSPACE_QUERY = "workspace.query"
    WORKSPACE_INGEST = "workspace.ingest"
    WORKSPACE_LIST_FILES = "workspace.list_files"
    WORKSPACE_DELETE_FILES = "workspace.delete_files"
    WORKSPACE_DOWNLOAD_SOURCE = "workspace.download_source"
    WORKSPACE_READ_METADATA = "workspace.read_metadata"
    WORKSPACE_UPDATE_METADATA = "workspace.update_metadata"
    WORKSPACE_READ_VISUAL_ASSET = "workspace.read_visual_asset"
    WORKSPACE_CREATE = "workspace.create"
    WORKSPACE_RESET = "workspace.reset"
    # Storage/promotion facts are operator-facing: only the admin preset (and
    # explicitly granted rules) carry this action; ordinary readers/editors
    # never see tier, promotion state, or retry details.
    WORKSPACE_STORAGE_STATUS = "workspace.storage_status"
    MODEL_CATALOGUE_WRITE = "model_catalogue.write"


_READER_ACTIONS: tuple[str, ...] = (
    AccessAction.WORKSPACE_QUERY,
    AccessAction.WORKSPACE_LIST_FILES,
    AccessAction.WORKSPACE_DOWNLOAD_SOURCE,
    AccessAction.WORKSPACE_READ_METADATA,
    AccessAction.WORKSPACE_READ_VISUAL_ASSET,
)
_EDITOR_ACTIONS: tuple[str, ...] = (
    *_READER_ACTIONS,
    AccessAction.WORKSPACE_INGEST,
    AccessAction.WORKSPACE_UPDATE_METADATA,
    AccessAction.WORKSPACE_DELETE_FILES,
)
ACTION_PRESETS: dict[str, tuple[str, ...]] = {
    "reader": _READER_ACTIONS,
    "editor": _EDITOR_ACTIONS,
    "admin": ("*",),
}


def corpus_mutation_access_action(action: object) -> str:
    """Map one mutation's accepted action to its authorization boundary.

    Unknown values fail closed to the strongest corpus mutation permission.
    """
    return {
        "ingest": AccessAction.WORKSPACE_INGEST,
        "replace": AccessAction.WORKSPACE_INGEST,
        "retry": AccessAction.WORKSPACE_INGEST,
        "delete": AccessAction.WORKSPACE_DELETE_FILES,
        "reset": AccessAction.WORKSPACE_RESET,
    }.get(str(action or ""), AccessAction.WORKSPACE_RESET)


class AccessDeniedError(PermissionError):
    """Raised when an authenticated user is not authorized for a resource."""


class AccessControl(Protocol):
    async def check(
        self,
        subject: AccessSubject,
        action: str,
        *,
        workspace: str | None = None,
    ) -> None: ...

    async def filter_workspaces(
        self,
        subject: AccessSubject,
        action: str,
        workspaces: Sequence[str],
    ) -> list[str]: ...


class AllowAllAccessControl:
    async def check(
        self,
        subject: AccessSubject,
        action: str,
        *,
        workspace: str | None = None,
    ) -> None:
        return None

    async def filter_workspaces(
        self,
        subject: AccessSubject,
        action: str,
        workspaces: Sequence[str],
    ) -> list[str]:
        return list(workspaces)


class JwtClaimsAccessControl:
    def __init__(self, settings: AccessSettings) -> None:
        self._rules = settings.rules

    async def check(
        self,
        subject: AccessSubject,
        action: str,
        *,
        workspace: str | None = None,
    ) -> None:
        if self._allows(subject, action, workspace):
            return
        target = f" workspace={workspace}" if workspace else ""
        raise AccessDeniedError(f"Access denied for action={action}{target}")

    async def filter_workspaces(
        self,
        subject: AccessSubject,
        action: str,
        workspaces: Sequence[str],
    ) -> list[str]:
        return [workspace for workspace in workspaces if self._allows(subject, action, workspace)]

    def _allows(self, subject: AccessSubject, action: str, workspace: str | None) -> bool:
        if subject is None or subject.auth_mode != "jwt":
            return False
        return any(
            _claim_matches(subject.claims, rule.claim, rule.value)
            and _action_matches(rule.actions, action)
            and _workspace_matches(rule.workspaces, workspace)
            for rule in self._rules
        )


def access_control_from_settings(settings: AccessSettings) -> AccessControl:
    if settings.mode == "jwt_claims":
        return JwtClaimsAccessControl(settings)
    return AllowAllAccessControl()


def _claim_matches(claims: Mapping[str, object], claim_name: str, expected: str) -> bool:
    raw = claims.get(claim_name)
    if isinstance(raw, str):
        return raw == expected
    if isinstance(raw, Iterable) and not isinstance(raw, (bytes, Mapping)):
        return expected in {str(value) for value in raw}
    return str(raw) == expected if raw is not None else False


def _action_matches(patterns: Sequence[str], action: str) -> bool:
    return any(_pattern_allows_action(pattern, action) for pattern in patterns)


def _pattern_allows_action(pattern: str, action: str) -> bool:
    preset = ACTION_PRESETS.get(pattern)
    if preset is not None:
        return any(_pattern_allows_action(entry, action) for entry in preset)
    return (
        pattern == "*"
        or pattern == action
        or (pattern.endswith(".*") and action.startswith(pattern[:-1]))
    )


def _workspace_matches(patterns: Sequence[str], workspace: str | None) -> bool:
    return any(pattern == "*" or pattern == workspace for pattern in patterns)


__all__ = [
    "ACTION_PRESETS",
    "AccessRule",
    "AccessSettings",
    "AccessAction",
    "AccessControl",
    "AccessDeniedError",
    "AccessSubject",
    "AllowAllAccessControl",
    "JwtClaimsAccessControl",
    "Principal",
    "access_control_from_settings",
]
