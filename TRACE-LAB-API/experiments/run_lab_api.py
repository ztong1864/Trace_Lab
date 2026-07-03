"""Run the optional TRACE real-lab local web API."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
VENDOR_DIR = PROJECT_ROOT / ".vendor_py"
if VENDOR_DIR.exists() and str(VENDOR_DIR) not in sys.path:
    # Keep environment packages first to avoid version shadowing.
    sys.path.append(str(VENDOR_DIR))

from chem_agent_bo.lab.api import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve TRACE lab web UI/API.")
    parser.add_argument("--projects-root", default="runs/lab_projects")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Lab API requires uvicorn. Install fastapi and uvicorn first.") from exc
    app = create_app(args.projects_root)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
