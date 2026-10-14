#!/usr/bin/env python3
"""Append the population-endpoint block to MAJOR1_RESULTS.md (idempotent)."""
import json
from pathlib import Path

R = Path("/mnt/disk_8tb/codex/2609-wan/paperM/PAPERM_CMIG_R3_20261011")
d = json.loads((R / "major1_population_inference.json").read_text(encoding="utf-8"))
per_seed = ", ".join(f"{k}:{v:+.4f}" for k, v in d["population_macro_per_seed"].items())
block = (
    "\n## Population endpoint (pre-specified endpoint, implemented post hoc)\n\n"
    f"* population mean **{d['population_macro']:+.4f}**; per seed {per_seed}\n"
    f"* site-resample bootstrap 95% CI **[{d['site_resample_ci95'][0]:+.4f}, {d['site_resample_ci95'][1]:+.4f}]**\n"
    f"* null-centred site bootstrap two-sided p **< {1/(d['bootstrap']['b']+1):.1e}** "
    "(floor; an exact zero is never reported)\n"
    f"* exact two-sided sign test **{d['sign_test']['k_positive']}/{d['sign_test']['n']}, "
    f"p = {d['sign_test']['two_sided_p']:.3g}**\n"
    "* per-site p values and their Holm correction are **withheld** (invalid resampling unit); the "
    "withdrawn tail-ratio p values must not be cited. See MAJOR1_ANALYSIS_REPORT.md.\n"
)
f = R / "MAJOR1_RESULTS.md"
txt = f.read_text(encoding="utf-8")
if "Population endpoint (pre-specified" in txt:
    print("population block already present; no change")
else:
    f.write_text(txt + block, encoding="utf-8")
    print("appended;", f.stat().st_size, "B")
