#!/usr/bin/env python3
"""Major 1 campaign runner — resumable.

528 fits = 88 feasible cells x 2 arms x 3 training seeds. A run is skipped when its report file
already exists, so an interrupted campaign resumes from the first missing cell/arm/seed. Every run
writes its own report + OOF arrays (with masks, digests, curve and environment), and this runner
appends a row to MAJOR1_LEDGER.csv after each fit. Smoke runs live in major1_runs/smoke and are not
counted here.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

PAPERM = Path("/mnt/disk_8tb/codex/2609-wan/paperM")
R3 = PAPERM / "PAPERM_CMIG_R3_20261011"
TREE = "/mnt/disk_8tb/codex/2609-wan/del/CR-MAGP-AHGNN-transductive-model"
PY = "/mnt/disk_8tb/codex/del/Wan-2608/magp_ahgnn_v0.9.2_uv/.venv/bin/python"
RUNS = R3 / "major1_runs"
RUNS.mkdir(parents=True, exist_ok=True)
LEDGER = R3 / "MAJOR1_LEDGER.csv"
SEEDS = [666, 2025, 2026]
EPOCHS, PATIENCE = 400, 200

man = json.loads((R3 / "SPLIT_AND_MASK_MANIFEST.json").read_text(encoding="utf-8"))
cells = [(i, c) for i, c in enumerate(man["cells"]) if c["feasible"]]
jobs = [(i, c, arm, s) for i, c in cells for arm in ("ON", "OFF") for s in SEEDS]
if not LEDGER.exists():
    LEDGER.write_text("cell,site,fold,arm,seed,rc,seconds,report\n", encoding="utf-8")
done = {p.name[:-len("_report.json")] for p in RUNS.glob("*_report.json")}
print(f"campaign: {len(cells)} cells x 2 arms x {len(SEEDS)} seeds = {len(jobs)} fits "
      f"| already present: {len(done)}", flush=True)

env = {"CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": "src", "PATH": "/usr/bin:/bin", "HOME": "/home/king"}
t_start = time.time()
for n, (idx, cell, arm, seed) in enumerate(jobs, 1):
    tag = f"cell{idx:03d}_{cell['site']}_f{cell['fold']}_{arm}_s{seed}"
    rep = RUNS / f"{tag}_report.json"
    if rep.exists():
        continue
    t0 = time.time()
    r = subprocess.run([PY, str(R3 / "major1_arms.py"), "--cell", str(idx), "--arm", arm,
                        "--seed", str(seed), "--epochs", str(EPOCHS), "--patience", str(PATIENCE),
                        "--out", str(RUNS)], cwd=TREE, env=env, capture_output=True, text=True,
                       timeout=7200)
    dt = round(time.time() - t0, 1)
    ok = rep.exists()
    with LEDGER.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([idx, cell["site"], cell["fold"], arm, seed, r.returncode, dt,
                                str(rep.relative_to(R3)) if ok else ""])
    tail = (r.stdout or "").strip().splitlines()[-1:] or [""]
    print(f"[{n}/{len(jobs)}] {tag} rc={r.returncode} {dt}s :: {tail[0][:110]}", flush=True)
    if not ok:
        print((r.stderr or "")[-400:], flush=True)
        # keep the campaign going: the failure is recorded, the original configuration is retried
        # only after the whole campaign pass (no seed or schedule changes to force success)
print(f"pass complete in {(time.time() - t_start) / 3600:.2f} h", flush=True)
missing = [j for j in jobs if not (RUNS / f"cell{j[0]:03d}_{j[1]['site']}_f{j[1]['fold']}_{j[2]}_s{j[3]}_report.json").exists()]
print(f"fits still missing after this pass: {len(missing)}", flush=True)
if missing:
    for m in missing[:10]:
        print("   MISSING", m[0], m[1]["site"], m[1]["fold"], m[2], m[3], flush=True)