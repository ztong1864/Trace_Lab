#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/env_paths.sh"

export PYTHONPATH="$(compose_pythonpath "${PROJECT_LOCAL_PYTHONPATH_DEFAULT}")"

exec "${AGENT_PYTHON_DEFAULT}" "${PROJECT_ROOT}/experiments/run_lab_api.py" "$@"

