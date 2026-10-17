#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
task_python="${PYTHON:-python3}"
"$task_python" evidence_20261017/recompute.py --controls
