#!/usr/bin/env python3
"""R2 figure step (instruction file item P3).

  * redraws the E-M1a distribution as a dot plot (20 permutation samples + null median +
    observed line + no-permutation control) so the figure cannot be misread as a two-bar
    histogram, with every annotation placed INSIDE the canvas;
  * writes the standalone SVG and converts it to a vector PDF (rsvg-convert), asserting the PDF
    contains no raster images;
  * installs Figure S8 at the END of the supplement (existing S1-S7 untouched) with a caption file
    and a cross-reference from Supplementary Note S4.
Idempotent and fail-closed like the other chain steps.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess

PKG = pathlib.Path("/mnt/disk_8tb/codex/2609-wan/paperM/PAPERM_CMIG_ADAPTATION_20260930")
BUILD = PKG / "build_cmig"
FIGS = BUILD / "figures"
EM1A = PKG / "em1a"
SUP_P = BUILD / "supplement_cmig.tex"

res = json.loads((EM1A / "em1a_results.json").read_text(encoding="utf-8"))
vals = [float(v) for v in res["per_seed_pooled_oof_auc"].values()]
assert len(vals) == 20, f"expected 20 permutation values, got {len(vals)}"
MED = float(res["median"])
OBS = float(res["observed_condition"]["released_bundle"])
CTRL = float(res["identity_check"]["pooled_oof_auc"])
LO, HI = 0.45, 0.99
W, H, ML, MR, MT, MB = 760, 330, 64, 28, 46, 62


def x_of(v: float) -> float:
    return ML + (v - LO) / (HI - LO) * (W - ML - MR)


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# deterministic vertical stacking of the 20 samples so overlapping points stay visible
order = sorted(range(len(vals)), key=lambda i: vals[i])
stack: dict[int, int] = {}
seen: list[float] = []
for i in order:
    lvl = sum(1 for s in seen if abs(s - vals[i]) < 0.006)
    stack[i] = lvl
    seen.append(vals[i])
BASE = H - MB - 16

parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{ML}" y="22" font-size="13.5" font-weight="bold" '
         f'font-family="Helvetica, Arial, sans-serif">'
         f'{esc("Within-site training-label permutation null (B = 20) vs the observed condition")}</text>']
# axis
parts.append(f'<line x1="{ML}" y1="{BASE + 14}" x2="{W - MR}" y2="{BASE + 14}" stroke="#333" stroke-width="1"/>')
for t in (0.50, 0.60, 0.70, 0.80, 0.90):
    x = x_of(t)
    parts.append(f'<line x1="{x:.1f}" y1="{BASE + 14}" x2="{x:.1f}" y2="{BASE + 19}" stroke="#333"/>')
    parts.append(f'<text x="{x:.1f}" y="{BASE + 34}" font-size="11.5" text-anchor="middle" '
                 f'font-family="Helvetica, Arial, sans-serif">{t:.2f}</text>')
parts.append(f'<text x="{(ML + W - MR) / 2:.0f}" y="{H - 12}" font-size="12" text-anchor="middle" '
             f'font-family="Helvetica, Arial, sans-serif">pooled out-of-fold AUC</text>')
# 20 permutation samples
for i, v in enumerate(vals):
    x = x_of(v)
    y = BASE - stack[i] * 11
    parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="#3182bd" stroke="#08519c" '
                 f'stroke-width="0.7"/>')
parts.append(f'<text x="{ML}" y="{MT - 6}" font-size="11.5" fill="#08519c" '
             f'font-family="Helvetica, Arial, sans-serif">'
             f'{esc(f"20 permutation runs (median {MED:.4f})")}</text>')
# median line
xm = x_of(MED)
parts.append(f'<line x1="{xm:.1f}" y1="{MT}" x2="{xm:.1f}" y2="{BASE + 14}" stroke="#08519c" '
             f'stroke-width="1.6"/>')
parts.append(f'<text x="{xm - 6:.1f}" y="{MT + 14}" font-size="11.5" fill="#08519c" '
             f'text-anchor="end" font-family="Helvetica, Arial, sans-serif">'
             f'{esc(f"null median {MED:.4f}")}</text>')
# observed + no-permutation control: annotations to the LEFT of the lines (inside the canvas)
xo = x_of(OBS)
parts.append(f'<line x1="{xo:.1f}" y1="{MT}" x2="{xo:.1f}" y2="{BASE + 14}" stroke="#cb181d" '
             f'stroke-width="1.8" stroke-dasharray="5 3"/>')
parts.append(f'<text x="{xo - 7:.1f}" y="{MT + 34}" font-size="11.5" fill="#cb181d" text-anchor="end" '
             f'font-family="Helvetica, Arial, sans-serif">'
             f'{esc(f"observed {OBS:.4f}")}</text>')
xc = x_of(CTRL)
parts.append(f'<circle cx="{xc:.1f}" cy="{MT + 52:.1f}" r="3.6" fill="#cb181d"/>')
parts.append(f'<text x="{xc - 7:.1f}" y="{MT + 60:.1f}" font-size="11.5" fill="#cb181d" '
             f'text-anchor="end" font-family="Helvetica, Arial, sans-serif">'
             f'{esc(f"no-permutation control {CTRL:.4f}")}</text>')
parts.append(f'<text x="{W - MR}" y="{MT - 6}" font-size="11.5" fill="#555" text-anchor="end" '
             f'font-family="Helvetica, Arial, sans-serif">'
             f'{esc(f"2.5-97.5 percentile interval of the permuted runs: [{res['p2_5']:.4f}, {res['p97_5']:.4f}]")}</text>')
parts.append("</svg>")
svg = "\n".join(parts) + "\n"

svg_out = EM1A / "em1a_distribution.svg"
svg_sub = FIGS / "FigureS8_em1a_null.svg"
pdf_out = FIGS / "FigureS8_em1a_null.pdf"
FIGS.mkdir(parents=True, exist_ok=True)
svg_out.write_text(svg, encoding="utf-8")
svg_sub.write_text(svg, encoding="utf-8")

r = subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(pdf_out), str(svg_sub)],
                   capture_output=True, text=True, timeout=120)
assert r.returncode == 0 and pdf_out.exists(), f"rsvg-convert failed: {r.stderr[:200]}"
rast = subprocess.run(["pdfimages", "-list", str(pdf_out)], capture_output=True, text=True).stdout
n_rast = len([l for l in rast.splitlines()[2:] if l.strip()])
assert n_rast == 0, f"figure PDF contains {n_rast} raster images (must stay vector)"
print(f"figure: svg {svg_out.stat().st_size}B / pdf {pdf_out.stat().st_size}B / raster images {n_rast}")

# ------------------------------------------------------------------ caption + figure environment
CAPTION = (r"""\caption{Within-site training-label permutation null (Supplementary Note~S4).
Each blue point is the pooled out-of-fold AUC of one of the 20 independent permutation runs, in
which the labels of the \emph{training} subset were permuted among subjects of the same site; the
solid blue line marks the median of the permuted runs. The dashed red line marks the observed
release condition (0.9604) and the red point the no-permutation control run of the same driver
(0.9627). Validation-subset labels were kept true (they could still influence model selection) and
test labels were never permuted. The shaded-free interval quoted in the text is the empirical
2.5--97.5 percentile range of the 20 permuted runs; it is a range of permutation results, not a
confidence interval for a population AUC, and the one-sided tail estimate is limited by the Monte
Carlo resolution of 20 seeds.}
""")
(FIGS / "figureS8_caption.tex").write_text(CAPTION, encoding="utf-8")
assert "\\label{fig:s8em1a}" not in CAPTION, "label must live in the figure environment"

sup = SUP_P.read_text(encoding="utf-8")
marker = "\\label{fig:s8em1a}"
if marker in sup:
    print("figure S8 environment already present")
else:
    env = ("\n\\begin{figure}[!tbp]\n\\centering\n"
           "\\includegraphics[width=0.98\\textwidth]{FigureS8_em1a_null.pdf}\n"
           "\\input{figures/figureS8_caption.tex}\n\\label{fig:s8em1a}\n\\end{figure}\n")
    assert sup.count("\\end{document}") == 1
    sup = sup.replace("\\end{document}", env + "\n\\end{document}", 1)
    assert marker in sup and sup.index(marker) < sup.index("\\end{document}")
    print("figure S8 environment inserted at end of supplement")

# cross-reference from S4 (right after the null-distribution paragraph's percentile statement)
REF_MARK = "The distribution is shown in Fig.~\\ref{fig:s8em1a}."
if REF_MARK not in sup:
    anchor = "not a confidence interval for a population value of AUC."
    pat = re.compile(r"\s+".join(re.escape(w) for w in anchor.split()))
    n = len(pat.findall(sup))
    assert n == 1, f"S4 cross-reference anchor occurs {n}x"
    sup = pat.sub(lambda m: m.group(0) + " " + REF_MARK, sup, count=1)
    assert REF_MARK in sup

SUP_P.write_text(sup, encoding="utf-8")

_inserted = CAPTION + REF_MARK
allowed = sorted(set(re.findall(r"(?<!\d)(?!10\.\d{4})\d+\.\d+", _inserted)))
# the renderer prints these as x-axis tick labels; they are figure furniture, not results
_TICKS = [f"{v:.2f}" for v in (0.50, 0.60, 0.70, 0.80, 0.90)]
allowed = sorted(set(allowed) | set(_TICKS))
axis_ticks = _TICKS
(PKG / "r2_fig_log.json").write_text(json.dumps(
    {"step": "R2 figure (Figure S8)", "svg": str(svg_out), "svg_submission": str(svg_sub),
     "pdf": str(pdf_out), "raster_images_in_pdf": n_rast,
     "caption_adds_no_new_numbers": not allowed, "allowed_new_numbers": allowed,
     "placement": "end of supplement; S1-S7 identifiers unchanged",
     "axis_tick_labels": axis_ticks,
     "axis_tick_labels_are_figure_furniture": True},
    ensure_ascii=False, indent=2), encoding="utf-8")
print("allowed new numbers from caption/ref:", allowed)
print("r2_fig_log.json written")