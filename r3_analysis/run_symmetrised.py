#!/usr/bin/env python3
"""Full symmetrised-graph rerun of Major 1 (528 fits), for the author's decision on graph orientation.

Driver: `major1_arms_sym.py` (identical to the canonical driver except that every edge pair is written
in both directions, which makes the propagation matrix symmetric). Everything else -- masks, arms,
seeds, epochs, patience -- is identical, so any difference is attributable to the orientation handling.

Resumable, separate output directory and ledger. The canonical runs, the canonical manifest and every
delivered artefact are never written. Intended to run AFTER the P1 campaign releases the GPU, so the two
campaigns do not contend for the same device.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

PAPERM = Path("/mnt/disk_8tb/codex/2609-wan/paperM")
R3 = PAPERM / "PAPERM_CMIG_R3_20261011"
TREE = Path("/mnt/disk_8tb/codex/2609-wan/del/CR-MAGP-AHGNN-transductive-model")
PY = "/mnt/disk_8tb/codex/del/Wan-2608/magp_ahgnn_v0.9.2_uv/.venv/bin/python"
RUNS = R3 / "major1_runs_sym"
RUNS.mkdir(parents=True, exist_ok=True)
LEDGER = R3 / "SYM_RUN_LEDGER.csv"

man = json.loads((R3 / "SPLIT_AND_MASK_MANIFEST.json").read_text(encoding="utf-8"))
SEEDS = man["seeds"]["training"]
jobs = [(i, c, arm, s) for i, c in enumerate(man["cells"]) if c["feasible"]
        for arm in ("ON", "OFF") for s in SEEDS]

if not LEDGER.exists():
    LEDGER.write_text("cell,site,fold,arm,seed,tag,rc,seconds,status\n", encoding="utf-8")

env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONPATH="src")
done = 0
for n, (i, c, arm, s) in enumerate(jobs, start=1):
    tag = f"cell{i:03d}_{c['site']}_f{c['fold']}_{arm}_s{s}"
    rep = RUNS / f"{tag}_report.json"
    if rep.exists():
        continue
    t0 = time.time()
    r = subprocess.run([PY, str(R3 / "major1_arms_sym.py"), "--cell", str(i), "--arm", arm,
                        "--seed", str(s), "--out", str(RUNS)], cwd=TREE, env=env,
                       capture_output=True, text=True, timeout=7200)
    dt = time.time() - t0
    ok = r.returncode == 0 and rep.exists()
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(f"{i},{c['site']},{c['fold']},{arm},{s},{tag},{r.returncode},{dt:.1f},{'ok' if ok else 'FAILED'}\n")
    if not ok:
        (RUNS / f"{tag}_FAILED.log").write_text((r.stderr or "")[-4000:], encoding="utf-8")
    done += 1
    if done % 10 == 0 or not ok:
        print(f"[{n}/{len(jobs)}] {tag} rc={r.returncode} {dt:.0f}s "
              f"{(r.stdout or '').strip().splitlines()[-1][:80] if r.stdout else ''}", flush=True)
print(f"[runner] symmetrised pass complete: {done} new fits of {len(jobs)}; ledger -> {LEDGER}", flush=True)