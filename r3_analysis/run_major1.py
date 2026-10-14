#!/usr/bin/env python3
"""Major 1 resumable campaign runner: 88 feasible cells x 2 arms x 3 training seeds = 528 fits.

Resumable by construction: a fit is skipped when its report exists. Every fit's own report and OOF
array carry masks, hashes, init hash, curve, best epoch and environment. Failures are retried with
the identical configuration and their logs kept; no seed is ever changed to obtain a preferred
result. Runs sequentially on one GPU so it does not contend with other GPU work beyond one device.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PAPERM = Path("/mnt/disk_8tb/codex/2609-wan/paperM")
R3 = PAPERM / "PAPERM_CMIG_R3_20261011"
TREE = Path("/mnt/disk_8tb/codex/2609-wan/del/CR-MAGP-AHGNN-transductive-model")
PY = "/mnt/disk_8tb/codex/del/Wan-2608/magp_ahgnn_v0.9.2_uv/.venv/bin/python"
RUNS = R3 / "major1_runs"
RUNS.mkdir(parents=True, exist_ok=True)
LEDGER = R3 / "MAJOR1_RUN_LEDGER.csv"

man = json.loads((R3 / "SPLIT_AND_MASK_MANIFEST.json").read_text(encoding="utf-8"))
cells = [(i, c) for i, c in enumerate(man["cells"]) if c["feasible"]]
SEEDS = man["seeds"]["training"]
jobs = [(i, c, arm, s) for i, c in cells for arm in ("ON", "OFF") for s in SEEDS]

if not LEDGER.exists():
    LEDGER.write_text("cell,site,fold,arm,seed,tag,rc,seconds,report_sha256,status\n", encoding="utf-8")

env = {"CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": "src", "PATH": "/usr/bin:/bin",
       "HOME": "/home/king"}
done = 0
for n, (i, c, arm, s) in enumerate(jobs, start=1):
    tag = f"cell{i:03d}_{c['site']}_f{c['fold']}_{arm}_s{s}"
    rep = RUNS / f"{tag}_report.json"
    if rep.exists():
        continue
    t0 = time.time()
    r = subprocess.run([PY, str(R3 / "major1_arms.py"), "--cell", str(i), "--arm", arm,
                        "--seed", str(s), "--out", str(RUNS)],
                       cwd=TREE, env=env, capture_output=True, text=True, timeout=7200)
    dt = time.time() - t0
    import hashlib
    sha = hashlib.sha256(rep.read_bytes()).hexdigest() if rep.exists() else ""
    row = (f"{i},{c['site']},{c['fold']},{arm},{s},{tag},{r.returncode},{dt:.1f},{sha},"
           f"{'ok' if r.returncode == 0 and rep.exists() else 'FAILED'}\n")
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(row)
    done += 1
    print(f"[{n}/{len(jobs)}] {tag} rc={r.returncode} {dt:.0f}s "
          f"({(r.stdout or '').strip().splitlines()[-1][:90] if r.stdout else (r.stderr or '')[-90:]})",
          flush=True)
    if r.returncode != 0:
        (RUNS / f"{tag}_FAILED.log").write_text((r.stderr or "")[-4000:], encoding="utf-8")
print(f"[runner] finished pass: {done} new fits; ledger -> {LEDGER}", flush=True)