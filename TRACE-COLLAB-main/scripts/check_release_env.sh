#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/env_paths.sh"

export PYTHONPATH="$(compose_pythonpath "${PROJECT_LOCAL_PYTHONPATH_DEFAULT}")"

"${AGENT_PYTHON_DEFAULT}" - <<'PY'
import importlib
import sys

required = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "yaml",
    "fastapi",
    "uvicorn",
    "openai",
    "langchain",
    "langchain_openai",
    "torch",
    "gpytorch",
    "botorch",
    "rich",
    "deap",
    "pymoo",
    "sobol_seq",
    "pyDOE",
    "chimera",
    "seaborn",
    "sqlalchemy",
    "atlas",
    "olympus",
    "chem_agent_bo",
]
optional_on_windows = ["golem"]
if sys.platform != "win32":
    required.extend(optional_on_windows)

failed = []
optional_missing = []
for name in required:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        failed.append((name, f"{type(exc).__name__}: {exc}"))
        continue
    version = getattr(module, "__version__", "unknown")
    print(f"[OK] {name} {version}")

if sys.platform == "win32":
    for name in optional_on_windows:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            optional_missing.append((name, f"{type(exc).__name__}: {exc}"))
            continue
        version = getattr(module, "__version__", "unknown")
        print(f"[OK] {name} {version}")

if failed:
    print("\n[ERROR] Missing or broken dependencies:", file=sys.stderr)
    for name, error in failed:
        print(f"  - {name}: {error}", file=sys.stderr)
    raise SystemExit(1)

if optional_missing:
    print("\n[WARN] Optional Windows dependencies unavailable:")
    for name, error in optional_missing:
        print(f"  - {name}: {error}")

print("\n[OK] TRACE Lab release environment import check passed.")
PY
