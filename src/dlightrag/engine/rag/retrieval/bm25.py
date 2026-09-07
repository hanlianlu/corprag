# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Language-aware BM25 profile routing and result fusion."""

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from dlightrag.engine.rag.retrieval import ContextRow, MetadataScope, format_bm25_top, rrf_fuse
from dlightrag.engine.rag.retrieval.filtering import current_filter_stats
from dlightrag.engine.rag.retrieval.language import BM25LanguageClassifier, normalize_language_code
from dlightrag.engine.rag.retrieval.ports import BM25ProfileSearch

logger = logging.getLogger(__name__)
_PROFILE_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_LANGUAGE_CODE_RE = re.compile(r"[a-z]{2,8}")


@dataclass(frozen=True)
class BM25Profile:
    """One language bucket and backend text configuration."""

    name: str
    text_config: str
    languages: tuple[str, ...] = ()
    fallback: bool = False

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not _PROFILE_NAME_RE.fullmatch(name):
            raise ValueError(f"unsafe BM25 profile name: {self.name!r}")
        text_config = str(self.text_config).strip()
        if not text_config:
            raise ValueError("BM25 text_config cannot be empty")
        normalized_languages = tuple(
            code for language in self.languages if (code := normalize_language_code(language))
        )
        invalid_languages = [
            language
            for language in normalized_languages
            if not _LANGUAGE_CODE_RE.fullmatch(language)
        ]
        if invalid_languages:
            raise ValueError(f"unsafe BM25 language code: {invalid_languages[0]!r}")
        if self.fallback and normalized_languages:
            raise ValueError("BM25 fallback profile must not declare languages")
        if not self.fallback and len(normalized_languages) != 1:
            raise ValueError("BM25 language profiles must declare exactly one language")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "text_config", text_config)
        object.__setattr__(self, "languages", normalized_languages)

    @property
    def language_bucket(self) -> str | None:
        return None if self.fallback or not self.languages else self.languages[0]


BM25_PROFILE_FALLBACK = BM25Profile(name="simple", text_config="simple", fallback=True)


def profiles_from_config(config_profiles: Iterable[Any]) -> tuple[BM25Profile, ...]:
    return tuple(
        BM25Profile(
            name=profile.name,
            text_config=profile.text_config,
            languages=tuple(profile.languages),
            fallback=profile.fallback,
        )
        for profile in config_profiles
    )


def profile_languages(profiles: Iterable[BM25Profile]) -> tuple[str, ...]:
    """Return the language buckets routed by non-fallback profiles."""
    return tuple(
        language for profile in profiles if not profile.fallback for language in profile.languages
    )


class ProfiledBM25Search:
    """Select configured language profiles and fuse their adapter results."""

    def __init__(
        self,
        searcher: BM25ProfileSearch,
        *,
        workspace: str,
        profiles: tuple[BM25Profile, ...],
        top_k: int = 40,
    ) -> None:
        if not any(profile.fallback for profile in profiles):
            raise ValueError("At least one BM25 profile must be marked fallback")
        self._searcher = searcher
        self._workspace = workspace
        self._profiles = profiles
        self._top_k = top_k
        self._language_classifier = BM25LanguageClassifier(profile_languages(profiles))

    async def search(
        self,
        query: str,
        *,
        scope: MetadataScope | None,
        top_k: int | None = None,
    ) -> list[ContextRow]:
        if scope is not None and not scope:
            logger.info(
                "[BM25] search: workspace=%s query=%r profiles=none candidate_scope=0 "
                "top_k=%s returned=0 top=none",
                self._workspace,
                query,
                top_k or self._top_k,
            )
            return []
        limit = self._top_k if top_k is None else top_k
        profiles = self._profiles_for_query(query)
        rankings = [
            await self._searcher.search_profile(
                query,
                profile_name=profile.name,
                language=profile.language_bucket,
                scope=scope,
                limit=int(limit),
            )
            for profile in profiles
        ]
        result = rankings[0] if len(rankings) == 1 else rrf_fuse(rankings)[: int(limit)]
        stats = current_filter_stats()
        if stats is not None:
            stats.bm25_strategy = True
            stats.visibility_strategy = "pushdown" if scope is not None else "bounded_pushdown"
            if scope is not None and scope.candidate_count_exact:
                shortfall = max(0, min(int(limit), scope.candidate_count) - len(result))
                if shortfall:
                    stats.bm25_candidate_shortfall = shortfall
        logger.info(
            "[BM25] search: workspace=%s query=%r profiles=%s candidate_scope=%s "
            "top_k=%d returned=%d top=%s",
            self._workspace,
            query,
            ",".join(profile.name for profile in profiles) or "none",
            f"{scope.render_candidate_count()}chunk" if scope is not None else "all",
            int(limit),
            len(result),
            format_bm25_top(result),
        )
        return result

    def _profiles_for_query(self, query: str) -> tuple[BM25Profile, ...]:
        language_profiles = tuple(
            profile for profile in self._profiles if not profile.fallback and profile.languages
        )
        selected: list[BM25Profile] = []
        if language_profiles:
            language = self._language_classifier.detect(query)
            selected.extend(
                profile for profile in language_profiles if language in profile.languages
            )
        if not selected:
            selected.extend(profile for profile in self._profiles if profile.fallback)
        return tuple(dict.fromkeys(selected))


__all__ = [
    "BM25_PROFILE_FALLBACK",
    "BM25Profile",
    "ProfiledBM25Search",
    "profile_languages",
    "profiles_from_config",
]
