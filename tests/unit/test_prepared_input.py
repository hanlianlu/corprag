# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The accepted execution envelope has one canonical bounded serializer."""

import pytest

from dlightrag.engine.agent.session.effects import canonical_json
from dlightrag.engine.runtime import (
    MAX_PREPARED_INPUT_BYTES,
    PreparedInputTooLargeError,
    require_prepared_input_bounds,
)


def test_exactly_8mib_passes_and_one_byte_over_fails() -> None:
    base = {"payload": ""}
    overhead = len(canonical_json(base).encode("utf-8"))
    exact = {"payload": "x" * (MAX_PREPARED_INPUT_BYTES - overhead)}
    require_prepared_input_bounds(exact)

    with pytest.raises(PreparedInputTooLargeError) as raised:
        require_prepared_input_bounds({"payload": exact["payload"] + "x"})

    assert raised.value.encoded_bytes == MAX_PREPARED_INPUT_BYTES + 1
