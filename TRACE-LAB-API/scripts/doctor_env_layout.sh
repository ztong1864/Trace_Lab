#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/env_paths.sh"

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Python default: ${AGENT_PYTHON_DEFAULT}"
echo "[INFO] local atlas src: ${ATLAS_LOCAL_SRC_DEFAULT}"
echo "[INFO] local olympus src: ${CHEMBO_LOCAL_OLYMPUS_SRC_DEFAULT}"
echo "[INFO] project local pythonpath: ${PROJECT_LOCAL_PYTHONPATH_DEFAULT}"
echo
"${AGENT_PYTHON_DEFAULT}" - <<'PY'
import sys
print("[INFO] sys.executable:", sys.executable)
PY

