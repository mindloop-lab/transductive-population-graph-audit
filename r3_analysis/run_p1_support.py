#!/usr/bin/env python3
"""P1 support-sampling sensitivity runner (frozen design, 270 fits).

Pre-declared: sites LEUVEN_1, PITT, YALE; support-sampling replicates r in {1,2,3} with pre-fixed
seeds 20261013 + r; training seeds {666, 2025, 2026}. Only the support draw changes relative to the
canonical protocol (T/S/V verified identical; R and its complement C change by construction).

Resumable: a fit is skipped when its report exists. Outputs go to P1_support_runs/ and a separate
ledger; the canonical runs, the canonical manifest and every delivered artefact are never touched.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

PAPERM = Path("/mnt/disk_8tb/codex/2609-wan/paperM")
R3 = PAPERM / "PAPERM_CMIG_R3_20261011"
TREE = Path("/mnt/disk_8tb/codex/2609-wan/del/CR-MAGP-AHGNN-transductive-model")
PY = "/mnt/disk_8tb/codex/del/Wan-2608/magp_ahgnn_v0.9.2_uv/.venv/bin/python"
RUNS = R3 / "P1_support_runs"
RUNS.mkdir(parents=True, exist_ok=True)
LEDGER = R3 / "P1_RUN_LEDGER.csv"

REPLICATES = {1: 20261014, 2: 20261015, 3: 20261016}
SEEDS = [666, 2025, 2026]

jobs = []
for r, seed in REPLICATES.items():
    man = json.loads((R3 / f"P1_SUPPORT_MANIFEST_r{r}.json").read_text(encoding="utf-8"))
    for i, c in enumerate(man["cells"]):
        if not c.get("feasible"):
            continue
        for arm in ("ON", "OFF"):
            for s in SEEDS:
                jobs.append((r, seed, i, c["site"], c["fold"], arm, s))

if not LEDGER.exists():
    LEDGER.write_text("replicate,support_seed,cell,site,fold,arm,seed,tag,rc,seconds,status\n", encoding="utf-8")

env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONPATH="src")
env.pop("P1_MANIFEST", None)
done = 0
for n, (r, seed, i, site, fold, arm, s) in enumerate(jobs, start=1):
    tag = f"r{r}_cell{i:03d}_{site}_f{fold}_{arm}_s{s}"   # must match the driver tag under P1_RUN_TAG
    rep = RUNS / f"{tag}_report.json"
    if rep.exists():
        continue
    env["P1_MANIFEST"] = f"P1_SUPPORT_MANIFEST_r{r}.json"
    env["P1_RUN_TAG"] = f"r{r}"
    t0 = time.time()
    res = subprocess.run([PY, str(R3 / "p1_arms.py"), "--cell", str(i), "--arm", arm, "--seed", str(s),
                          "--out", str(RUNS)], cwd=TREE, env=env, capture_output=True, text=True, timeout=7200)
    dt = time.time() - t0
    ok = res.returncode == 0 and rep.exists()
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(f"{r},{seed},{i},{site},{fold},{arm},{s},{tag},{res.returncode},{dt:.1f},{'ok' if ok else 'FAILED'}\n")
    if not ok:
        (RUNS / f"{tag}_FAILED.log").write_text((res.stderr or "")[-4000:], encoding="utf-8")
    done += 1
    if done % 10 == 0 or not ok:
        print(f"[{n}/{len(jobs)}] {tag} rc={res.returncode} {dt:.0f}s "
              f"{(res.stdout or '').strip().splitlines()[-1][:80] if res.stdout else ''}", flush=True)
print(f"[runner] pass complete: {done} new fits of {len(jobs)} jobs; ledger -> {LEDGER}", flush=True)