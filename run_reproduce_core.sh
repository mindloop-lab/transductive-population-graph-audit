#!/usr/bin/env bash
# Reproduce the core mechanism chain reported in the paper.
# This is intentionally narrower than the complete frozen 69-row registry:
# archived/derived rows are verified from shipped artifacts by validate_registry.py.
set -euo pipefail

PY=${PY:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
export PYTHONPATH=src
export CUBLAS_WORKSPACE_CONFIG=:4096:8

: "${ABIDE_CACHE:?set ABIDE_CACHE to the C-PAC cache npz}"
: "${ABIDE_PHENO:?set ABIDE_PHENO to the phenotype csv}"

OUT=${PAPERM_OUT:-$HERE/results/reproduced}
mkdir -p "$OUT"
export PAPERM_OUT="$OUT"

echo "[1/10] main transductive cohort model"
$PY scripts/train_transductive_10fold_fast.py --seed 666

echo "[2/10] canonical sex+site control"
$PY scripts/train_controls.py --mode full --edge sex_site --seed 666

echo "[3/10] site-only graph"
$PY scripts/train_controls.py --mode full --edge site --seed 666

echo "[4/10] subject-only control"
$PY scripts/train_controls.py --mode subject_only --edge sex_site --seed 666

echo "[5/10] edge-count/random graph control"
$PY scripts/train_controls.py --mode full --edge random --seed 666

echo "[6/10] label-permutation null"
$PY scripts/train_controls.py --mode perm --edge sex_site --seed 666

echo "[7/10] remove train-test population edges"
$PY scripts/mechanism_controls.py --feat real --mask_tt --seed 666

echo "[8/10] degree-preserving rewiring"
$PY scripts/mechanism_controls.py --feat real --rewire --seed 666

echo "[9/10] constant-feature topology control"
$PY scripts/mechanism_controls.py --feat const --seed 666

echo "[10/10] C-PAC unseen-site LOSO"
$PY scripts/train_loso.py

if [[ -n "${ABIDE_CACHE_NIAK:-}" ]]; then
  echo "[optional] NIAK unseen-site LOSO"
  ABIDE_PIPELINE=NIAK PAPERM_OUT="$OUT/loso_niak" \
    $PY scripts/train_loso.py --cache "$ABIDE_CACHE_NIAK"
fi

echo "core reproduction complete -> $OUT"
echo "Run: $PY scripts/validate_registry.py"
