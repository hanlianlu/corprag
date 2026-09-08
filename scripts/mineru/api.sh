#!/usr/bin/env bash
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
set -euo pipefail

mineru_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/mineru/env.sh
source "$mineru_script_dir/env.sh"

load_mineru_env_key MINERU_API_HOST
load_mineru_env_key MINERU_API_PORT
load_mineru_env_key MINERU_SERVICE_VENV
load_mineru_env_key MINERU_HYBRID_EFFORT
load_mineru_env_key MINERU_TITLE_AIDED_ATTEMPT_TIMEOUT_SECONDS
load_mineru_env_key MINERU_TITLE_AIDED_MAX_ATTEMPTS

host="${MINERU_API_HOST:-127.0.0.1}"
port="${MINERU_API_PORT:-8210}"
venv="${MINERU_SERVICE_VENV:-.venv-mineru}"

if [[ -x "$venv/bin/mineru-api" ]]; then
  mineru_api="$venv/bin/mineru-api"
elif command -v mineru-api >/dev/null 2>&1; then
  mineru_api="$(command -v mineru-api)"
else
  cat >&2 <<'EOF'
mineru-api was not found on PATH.

Install the dedicated local MinerU service env first:
  make mineru-install

Override MinerU install extras when the machine needs them:
  MINERU_INSTALL_EXTRAS=core,mlx make mineru-install
  MINERU_INSTALL_EXTRAS=core,vllm make mineru-install
  MINERU_INSTALL_EXTRAS=core,lmdeploy make mineru-install
EOF
  exit 127
fi

# Load scripts/mineru/sitecustomize.py at interpreter startup (also inherited by
# MinerU's spawned worker processes) to apply the supported parser guards,
# including the image ceiling and bounded title-aided LLM requests.
export PYTHONPATH="${mineru_script_dir}${PYTHONPATH:+:${PYTHONPATH}}"

exec "$mineru_api" --host "$host" --port "$port" "$@"
