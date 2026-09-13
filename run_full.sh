#!/usr/bin/env bash
# Backward-compatible entry point.
# The v1.0 release uses run_reproduce_core.sh for the executable core mechanism chain.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
echo "run_full.sh is retained for compatibility; delegating to run_reproduce_core.sh"
exec bash "$HERE/run_reproduce_core.sh"
