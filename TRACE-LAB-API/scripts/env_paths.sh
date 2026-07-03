#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${BASH_SOURCE:-}" ]]; then
  _TRACE_ENV_SOURCE="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  _TRACE_ENV_SOURCE="${(%):-%x}"
else
  _TRACE_ENV_SOURCE="$0"
fi

PROJECT_ROOT="$(cd "$(dirname "${_TRACE_ENV_SOURCE}")/.." && pwd)"

export AGENT_PYTHON_DEFAULT="${AGENT_PYTHON_DEFAULT:-python}"
export QLOGEI_PYTHON_DEFAULT="${QLOGEI_PYTHON_DEFAULT:-python}"

export ATLAS_LOCAL_SRC_DEFAULT="${PROJECT_ROOT}/third_party/atlas/src"
export CHEMBO_LOCAL_OLYMPUS_SRC_DEFAULT="${PROJECT_ROOT}/third_party/olympus/src"

_trace_path_join() {
  local joined=""
  local item=""
  for item in "$@"; do
    if [[ -z "${item}" || ! -d "${item}" ]]; then
      continue
    fi
    if [[ -z "${joined}" ]]; then
      joined="${item}"
    else
      joined="${joined}:${item}"
    fi
  done
  printf '%s\n' "${joined}"
}

export PROJECT_LOCAL_PYTHONPATH_DEFAULT="$(_trace_path_join \
  "${ATLAS_LOCAL_SRC_DEFAULT}" \
  "${CHEMBO_LOCAL_OLYMPUS_SRC_DEFAULT}")"

compose_pythonpath() {
  local local_path="${1:-}"
  local existing_path="${2:-${PYTHONPATH:-}}"
  if [[ -n "${local_path}" && -n "${existing_path}" ]]; then
    printf '%s:%s\n' "${local_path}" "${existing_path}"
  elif [[ -n "${local_path}" ]]; then
    printf '%s\n' "${local_path}"
  else
    printf '%s\n' "${existing_path}"
  fi
}

