#!/usr/bin/env python3
"""Insert the orientation-robustness paragraph into the R3 revision, from the comparison JSON.

All numbers are read from SYMMETRISED_VS_CANONICAL.json; none is typed by hand. Idempotent.
Usage: insert_sym_paragraph.py [--tex PATH]
"""
import argparse
import json
from pathlib import Path

R = Path("/mnt/disk_8tb/codex/2609-wan/paperM/PAPERM_CMIG_R3_20261011")
S = R / "R3_revision" / "source"

ap = argparse.ArgumentParser()
ap.add_argument("--tex", default=str(S / "paperM_cmig.tex"))
tex_path = Path(ap.parse_args().tex)

d = json.loads((R / "SYMMETRISED_VS_CANONICAL.json").read_text(encoding="utf-8"))
c, s, agree = d["canonical"], d["symmetrised"], d["sign_agreement"]

# wording is data-driven: if the effect does not survive symmetrisation, say so plainly
survives = (abs(s["population_macro"]) >= 0.5 * abs(c["population_macro"])
            and agree["comparable_sites"] > 0
            and agree["agree"] >= 0.5 * agree["comparable_sites"])
verdict = (
    "The direction of the site-level effect is preserved, but its magnitude changes"
    if survives else
    "The effect does not survive symmetrisation: under the symmetric construction the population macro "
    "falls to a level indistinguishable from zero, so the reported population effect depends on how edges "
    "are written rather than only on the supervision-location manipulation"
)

lines = [
    "\\paragraph{Orientation robustness (parallel analysis, not a replacement)} Re-running the same paired",
    "experiment with the population graph \\emph{symmetrised} (every edge pair written in both directions; every",
    "other element unchanged) gave a population macro of "
    f"{s['population_macro']:+.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}] over the same {s['n_sites']} sites",
    f"and a sign test of {s['sign_k']}/{s['n_sites']}, against {c['population_macro']:+.3f} "
    f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]",
    f"and {c['sign_k']}/{c['n_sites']} under the directed construction that the primary numbers use; the sign of the",
    f"site-level effect agreed at {agree['agree']} of {agree['comparable_sites']} comparable sites. The primary numbers",
    "remain those of the pre-specified directed construction.",
    f"\\emph{{{verdict}}}. This parallel analysis is reported so that the sensitivity to how edges are written is",
    "visible rather than hidden, and no claim of orientation invariance is made.",
    "",
]
para = "\n".join(lines) + "\n"

tex = tex_path.read_text(encoding="utf-8")
if "Orientation robustness (parallel analysis" in tex:
    print("paragraph already present; no change")
else:
    anchor = "% ============================================================================\n\\section{Discussion}"
    assert tex.count(anchor) == 1, "anchor not unique"
    tex_path.write_text(tex.replace(anchor, para + anchor, 1), encoding="utf-8")
    print("inserted; tex now", tex_path.stat().st_size, "B")
