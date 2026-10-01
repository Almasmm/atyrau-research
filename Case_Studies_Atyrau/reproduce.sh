#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"
cd -- "$PACKAGE_ROOT"
"$PYTHON_EXECUTABLE" scripts/reproduce.py --mode "${1:-full}" --case "${2:-all}"
