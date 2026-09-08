# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Startup shim for the MinerU sidecar interpreter (and its spawned workers).

MinerU loads document scans with Pillow, whose default decompression-bomb guard
warns at ~89.5MP and raises ``DecompressionBombError`` at ~179MP. Large
multi-page composite scans (e.g. a ~205MP certificate) therefore fail to load
*before* MinerU can parse them, surfacing as::

    MinerU local parse failed: Failed to load file <name>: Image size (N pixels)
    exceeds limit of 178956970 pixels, could be decompression bomb DOS attack.

DlightRAG targets 32GB+ hosts and intentionally accepts large scans, so raise
the ceiling to match DlightRAG's own ``MAX_DECODE_IMAGE_PIXELS`` (250MP: no
warning up to 250MP; hard error only above 500MP). CPython's ``site`` machinery
imports this module automatically because ``scripts/mineru/api.sh`` puts this
directory on ``PYTHONPATH`` — which MinerU's spawned worker processes inherit.
A failure here cannot break the interpreter: ``site`` catches sitecustomize
errors, warns on stderr, and continues startup with Pillow's default ceiling.

It also repairs MinerU's title-leveling prompts. Both builders show the model an
example dict with unquoted integer keys and then tell it not to format the
output, so a model that obliges returns compact pseudo-JSON like ``{0:2,1:3}``.
``json_repair`` mis-splits that into keys such as ``"3,2"`` and the following
``int(k)`` raises, so every title group burns three streamed LLM calls and ends
with no levels at all. Requesting strict JSON with quoted keys — matching the
shape MinerU already sends — removes the ambiguity. The directive is inserted
before the input block because the prompt ends on the model's answer cue.

MinerU threads a ``prompt_builder`` through ``_request_title_levels`` but its
public ``llm_aided_title`` entry point does not expose it, so there is no
configuration path; the builders are read from module globals at call time,
which makes rebinding them sufficient.

The title-aided call is also a best-effort parser refinement, not a reason to
hold the serial parse lane forever. MinerU's streamed OpenAI-compatible request
has only an inactivity timeout: a provider that keeps emitting partial bytes can
stay alive indefinitely, and each upstream retry can repeat that stall. Replace
that request with the same public protocol under both an inactivity timeout and
a wall-clock deadline. Failure returns no title levels, which is already
MinerU's supported fallback, and logs only the exception type so credentials in
the title-aided config never appear in diagnostic tracebacks.

Finally it exposes the hybrid parse effort. ``medium`` force-disables
image/chart analysis and feeds the VLM pipeline-YOLO layout boxes instead of
letting it detect blocks itself; ``high`` returns whole figures with bound
captions and non-empty chart content, at roughly five times the parse time.
Which trade is right depends on the corpus, but MinerU hardcodes the constant
with no environment override. Its ``effort`` form field binds it as a default at
function-definition time, so rebinding it here -- before MinerU's API module is
imported -- moves the choice into ``.env.mineru``.

It also widens the HTTP keep-alive. Uvicorn closes idle connections after 5s and
MinerU exposes no flag for it, while LightRAG's MinerU client polls a *pooled*
connection every ``poll_interval_seconds`` (DlightRAG default: 5). The two
deadlines coincide, so each poll is a coin flip on whether the server tears the
connection down just as the client reuses it; httpx then raises
``RemoteProtocolError: Server disconnected without sending a response`` and the
whole ingest fails while MinerU keeps parsing, unaware. Longer parses poll more
often and so fail more reliably — a ~470s parse gets ~94 chances.
The value matches docling-serve, which ships ``timeout_keep_alive = 60``: both
sidecars then behave alike, and the margin covers any sane poll interval instead
of breaking again the moment an operator raises a knob that has no upper bound.
The cost is one idle socket per client held longer, which a loopback sidecar
with a single client does not notice.
"""

import os
import sys
import time
from functools import wraps

import uvicorn.config
from PIL import Image

Image.MAX_IMAGE_PIXELS = 250_000_000

# docling-serve's own default; must exceed parser_sidecars.mineru.poll_interval_seconds.
_KEEP_ALIVE_SECONDS = 60
_uvicorn_config_init = uvicorn.config.Config.__init__


@wraps(_uvicorn_config_init)
def _config_with_keep_alive(self, *args, **kwargs):
    kwargs.setdefault("timeout_keep_alive", _KEEP_ALIVE_SECONDS)
    _uvicorn_config_init(self, *args, **kwargs)


uvicorn.config.Config.__init__ = _config_with_keep_alive

import mineru.utils.llm_aided as _llm_aided  # noqa: E402  # type: ignore[import-not-found]

_PROMPT_INPUT_ANCHOR = "Input title list:"
_STRICT_JSON_DIRECTIVE = (
    "严格要求：只返回合法 JSON。key 必须是带双引号的字符串，与输入字典的 key 完全一致；"
    'value 必须是整数。例如：{"0": 1, "1": 2, "2": 2, "3": 3}\n\n'
)


def _request_strict_json(builder):
    @wraps(builder)
    def build(title_dict):
        prompt = builder(title_dict)
        return prompt.replace(
            _PROMPT_INPUT_ANCHOR, _STRICT_JSON_DIRECTIVE + _PROMPT_INPUT_ANCHOR, 1
        )

    return build


for _name in ("_build_title_optimize_prompt", "_build_relative_title_optimize_prompt"):
    setattr(_llm_aided, _name, _request_strict_json(getattr(_llm_aided, _name)))


def _positive_float_env(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = 0
    if value > 0:
        return value
    print(f"sitecustomize: ignoring invalid {name}", file=sys.stderr)
    return default


def _positive_int_env(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value > 0:
        return value
    print(f"sitecustomize: ignoring invalid {name}", file=sys.stderr)
    return default


def _close_quietly(value):
    close = getattr(value, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception as exc:
        _llm_aided.logger.debug("Title-aided cleanup failed: {}", type(exc).__name__)


_TITLE_AIDED_ATTEMPT_TIMEOUT_SECONDS = _positive_float_env(
    "MINERU_TITLE_AIDED_ATTEMPT_TIMEOUT_SECONDS", 60.0
)
_TITLE_AIDED_MAX_ATTEMPTS = _positive_int_env("MINERU_TITLE_AIDED_MAX_ATTEMPTS", 2)
_TITLE_AIDED_READ_TIMEOUT_SECONDS = min(_TITLE_AIDED_ATTEMPT_TIMEOUT_SECONDS, 10.0)


def _bounded_request_title_levels(title_aided_config, title_dict, prompt_builder=None):
    if not title_dict:
        return {}

    builder = prompt_builder or _llm_aided._build_title_optimize_prompt
    prompt = builder(title_dict)
    api_params = {
        "model": title_aided_config["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "stream": True,
    }
    if "enable_thinking" in title_aided_config:
        api_params["extra_body"] = {"enable_thinking": title_aided_config["enable_thinking"]}

    try:
        client = _llm_aided.OpenAI(
            api_key=title_aided_config["api_key"],
            base_url=title_aided_config["base_url"],
            timeout=_TITLE_AIDED_READ_TIMEOUT_SECONDS,
            max_retries=0,
        )
    except Exception as exc:
        _llm_aided.logger.warning("Title-aided LLM client setup failed: {}", type(exc).__name__)
        return None

    expected_keys = set(range(len(title_dict)))
    try:
        for attempt in range(1, _TITLE_AIDED_MAX_ATTEMPTS + 1):
            completion = None
            started = time.monotonic()
            try:
                completion = client.chat.completions.create(**api_params)
                content_pieces = []
                for chunk in completion:
                    if time.monotonic() - started >= _TITLE_AIDED_ATTEMPT_TIMEOUT_SECONDS:
                        raise TimeoutError("title-aided wall-clock deadline exceeded")
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        content_pieces.append(chunk.choices[0].delta.content)

                content = "".join(content_pieces).strip()
                if "</think>" in content:
                    content = content.split("</think>", 1)[1].strip()
                parsed = _llm_aided.json_repair.loads(content)
                levels = {int(key): int(value) for key, value in parsed.items()}
                if set(levels) == expected_keys:
                    return levels
                raise ValueError("title keys did not match")
            except Exception as exc:
                _llm_aided.logger.warning(
                    "Title-aided LLM attempt {}/{} failed: {}",
                    attempt,
                    _TITLE_AIDED_MAX_ATTEMPTS,
                    type(exc).__name__,
                )
            finally:
                _close_quietly(completion)
    finally:
        _close_quietly(client)

    _llm_aided.logger.warning(
        "Title-aided correction skipped after {} bounded attempt(s)",
        _TITLE_AIDED_MAX_ATTEMPTS,
    )
    return None


_llm_aided._request_title_levels = _bounded_request_title_levels

import mineru.cli.backend_options as _backend_options  # noqa: E402  # type: ignore[import-not-found]

_effort = os.environ.get("MINERU_HYBRID_EFFORT", "").strip()
if _effort in _backend_options.HYBRID_EFFORT_CHOICES:
    _backend_options.DEFAULT_HYBRID_EFFORT = _effort
elif _effort:
    # A typo must not quietly leave the corpus on the other trade-off.
    print(
        f"sitecustomize: ignoring MINERU_HYBRID_EFFORT={_effort!r}; "
        f"expected one of {_backend_options.HYBRID_EFFORT_CHOICES}",
        file=sys.stderr,
    )
