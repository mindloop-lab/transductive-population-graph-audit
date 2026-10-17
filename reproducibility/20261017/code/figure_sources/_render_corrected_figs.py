#!/usr/bin/env python
"""Render corrected versions of the two ChatGPT artwork files with cairosvg.

Corrections applied to Figure 1 v3 (found by reading the rendered figure):
  * the "Test-test edges" row is only meaningful for the in-cohort protocol (in LOSO
    there are no test-test edges), so the label is qualified;
  * "HO (context)" -> "HO-111 (context)" to match the atlas list used in the paper;
  * the footnote is extended with the LOSO caveat.
FigureS1 v2 is rendered as supplied (it fixes an ambiguous symbol).
"""
import pathlib
import cairosvg

ROOT = pathlib.Path("/mnt/disk_8tb/codex/2609-wan/paperM/MAGP-AHGNN")
SRC = ROOT / "paperM_gptupdated_0911/figures"
OUT = pathlib.Path("/tmp/corrected_figs")
OUT.mkdir(exist_ok=True)

# ---------------- Figure 1 --------------------------------------------------------
svg = (SRC / "Figure1_study_design_v3.svg").read_text()
edits = [
    (">Test-test edges<", ">Test-test edges (in-cohort)<"),
    (">HO (context)<", ">HO-111 (context)<"),
    ("Key distinction: transductive context visibility is permitted;",
     "Key distinction: transductive context visibility is permitted;"),
    ("test-label access is not.", "test-label access is not. Test-test edges exist only in the in-cohort protocol."),
]
report = []
for old, new in edits:
    n = svg.count(old)
    report.append(f"  {'ok ' if n == 1 else f'MISS({n})'} {old[:52]}")
    if n == 1 and old != new:
        svg = svg.replace(old, new)
print("Figure 1 edits:")
print("\n".join(report))
f1 = OUT / "Figure1_study_design_v3_corrected.svg"
f1.write_text(svg)
cairosvg.svg2pdf(url=str(f1), write_to=str(OUT / "Figure1_study_design_v3_corrected.pdf"))
print("rendered:", OUT / "Figure1_study_design_v3_corrected.pdf")

# ---------------- Figure S1 -------------------------------------------------------
s2 = SRC / "FigureS1_protocol_schematic_v2.svg"
cairosvg.svg2pdf(url=str(s2), write_to=str(OUT / "FigureS1_protocol_schematic_v2.pdf"))
print("rendered:", OUT / "FigureS1_protocol_schematic_v2.pdf")

# ---------------- report fonts ----------------------------------------------------
print("\nfont check (must be Type 1 / TrueType, not Type 3):")
import subprocess
for f in sorted(OUT.glob("*.pdf")):
    out = subprocess.run(["pdffonts", str(f)], capture_output=True, text=True).stdout
    types = sorted({ln.split()[1] + " " + ln.split()[2] for ln in out.splitlines()[2:] if ln.split()})
    print(f"  {f.name}: {types or 'no embedded fonts'}")
