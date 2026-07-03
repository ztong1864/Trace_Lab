#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"

if [[ ! -x "$(command -v "${PYTHON_BIN}")" ]]; then
  echo "[ERROR] ${PYTHON_BIN} was not found. Install Python 3.10 or set PYTHON_BIN=/path/to/python." >&2
  exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
bash "${PROJECT_ROOT}/scripts/install_release_deps.sh"
bash "${PROJECT_ROOT}/scripts/check_release_env.sh"

echo
echo "[OK] TRACE Lab virtual environment is ready."
echo "Activate it with:"
echo "  source ${VENV_DIR}/bin/activate"

