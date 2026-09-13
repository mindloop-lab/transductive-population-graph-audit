#!/usr/bin/env bash
# Synthetic smoke test: exercises the same code path as the real runs, no ABIDE data.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

# Resolve an interpreter: honour $PY or $PYTHON, otherwise take python3 then python.
PY=${PY:-${PYTHON:-}}
if [ -z "$PY" ]; then
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then PY=$candidate; break; fi
  done
fi
if [ -z "$PY" ]; then
  echo "SMOKE FAIL: no python interpreter found; set PYTHON=/path/to/python3"
  exit 1
fi
echo "interpreter: $PY ($($PY -V 2>&1))"

export PYTHONPATH=src
export CUBLAS_WORKSPACE_CONFIG=:4096:8

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "[1/2] building a synthetic fixture (60 subjects, 4 sites)"
PAPERM_OUT="$WORK" "$PY" scripts/_make_synthetic_fixture.py
CACHE="$WORK/cache_abide_data.npz"
PHENO="$WORK/Phenotypic_V1_0b_preprocessed1.csv"
if [ ! -f "$CACHE" ] || [ ! -f "$PHENO" ]; then
  echo "SMOKE FAIL: fixture not produced"; exit 1
fi

echo "[2/2] running the leave-one-site-out harness on the fixture"
set +e
ABIDE_CACHE="$CACHE" ABIDE_PHENO="$PHENO" PAPERM_OUT="$WORK/out" \
  "$PY" scripts/train_loso.py > "$WORK/loso.log" 2>&1
RC=$?
set -e
tail -3 "$WORK/loso.log"
if [ "$RC" -ne 0 ]; then
  echo "SMOKE FAIL: train_loso.py exited $RC"; exit 1
fi
if ! grep -q "MODEL LOSO pooled AUC" "$WORK/loso.log"; then
  echo "SMOKE FAIL: no result line in the output"; exit 1
fi
echo "SMOKE PASS"
