#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/env_paths.sh"

"${AGENT_PYTHON_DEFAULT}" -m pip install --upgrade pip
"${AGENT_PYTHON_DEFAULT}" -m pip install -r "${PROJECT_ROOT}/requirements.txt"

