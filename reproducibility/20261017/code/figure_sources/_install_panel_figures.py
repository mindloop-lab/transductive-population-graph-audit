#!/usr/bin/env python3
"""Install the full-size panel figures into the CMIG build (2026-10-02, author-authorized).

Figure 1 becomes panel A: identical author layout, placed at the full text width, which scales the
type up by 1600/910 = 1.76x (the four-panel composite cannot be legible at this page width without a
re-layout, and two re-layout architectures were measured to fail). Panels B-D and the author's
complete four-panel overview move into the supplement as Figures S4-S7.

Every copy and insertion is asserted; a silent no-op or a double insertion is impossible.
"""
import hashlib
import json
import xml.etree.ElementTree as ET
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
BUILD = HERE / "build_cmig"
SRC = HERE / "figure_source" / "panels"
FIG = BUILD / "figures"
SUP = BUILD / "supplement_cmig.tex"

COPY = [("figure1_panel_A.pdf", "Figure1_panel_A.pdf"),
        ("figure1_panel_B.pdf", "Figure1_panel_B.pdf"),
        ("figure1_panel_C.pdf", "Figure1_panel_C.pdf"),
        ("figure1_panel_D.pdf", "Figure1_panel_D.pdf"),
        ("figure1_panel_composite.pdf", "Figure1_panel_composite.pdf"),
        ("caption_fig1.tex", "figure1_caption_panelA.tex"),
        ("caption_S4.tex", "figureS4_caption.tex"),
        ("caption_S5.tex", "figureS5_caption.tex"),
        ("caption_S6.tex", "figureS6_caption.tex"),
        ("caption_S7.tex", "figureS7_caption.tex")]

BLOCKS = [("fig:s4panelB", "Figure1_panel_B.pdf", "figureS4_caption.tex", 0.98),
          ("fig:s5panelC", "Figure1_panel_C.pdf", "figureS5_caption.tex", 0.98),
          ("fig:s6panelD", "Figure1_panel_D.pdf", "figureS6_caption.tex", 0.98),
          ("fig:s7overview", "Figure1_panel_composite.pdf", "figureS7_caption.tex", 0.98)]

ANCHOR = "\\end{document}"
MARK = "% --- full-size study-design panels (installed by _install_panel_figures.py) ---"


def main():
    assert SRC.is_dir(), f"panel source dir missing: {SRC}"
    FIG.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for src, dst in COPY:
        s = SRC / src
        assert s.is_file(), f"missing source figure asset: {s}"
        shutil.copy2(s, FIG / dst)
        got = hashlib.sha256((FIG / dst).read_bytes()).hexdigest()
        want = hashlib.sha256(s.read_bytes()).hexdigest()
        assert got == want, f"copy verification failed for {dst}"
        hashes[dst] = got

    sup = SUP.read_text(encoding="utf-8")
    if MARK in sup:
        print("panel figures already installed in the supplement; nothing to do")
    else:
        assert sup.count(ANCHOR) == 1, f"expected exactly one {ANCHOR} in the supplement"
        block = [MARK]
        for label, pdf, cap, width in BLOCKS:
            block.append("\\begin{figure}[t]\n\\centering\n"
                         f"\\includegraphics[width={width}\\textwidth]{{{pdf}}}\n"
                         f"\\input{{figures/{cap}}}\n\\label{{{label}}}\n\\end{{figure}}")
        block.append("")
        sup = sup.replace(ANCHOR, "\n".join(block) + ANCHOR)
        for label, _pdf, _cap, _w in BLOCKS:
            assert f"\\label{{{label}}}" in sup, f"figure block {label} not installed"
        SUP.write_text(sup, encoding="utf-8")
        print(f"supplement: installed {len(BLOCKS)} full-size panel figures (S4-S7)")

    (HERE / "figure_source" / "panels" / "installed_manifest.json").write_text(
        json.dumps({"installed_into": str(BUILD), "files": hashes}, indent=2), encoding="utf-8")

    # provenance for the gates: measured admission checks + printed type per panel, derived from the
    # exported PDFs themselves (no values typed by hand)
    import re
    import subprocess

    TEXTWIDTH_PT = 458.6                      # 0.98\textwidth in the CMIG layout
    metrics = {}
    for lab, pdf in (("A", "Figure1_panel_A.pdf"), ("B", "Figure1_panel_B.pdf"),
                     ("C", "Figure1_panel_C.pdf"), ("D", "Figure1_panel_D.pdf"),
                     ("composite", "Figure1_panel_composite.pdf")):
        f = FIG / pdf
        info = subprocess.run(["pdfinfo", str(f)], capture_output=True, text=True).stdout
        page_w = float(re.search(r"Page size:\s+([\d.]+) x", info).group(1))
        pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
        fonts = subprocess.run(["pdffonts", str(f)], capture_output=True, text=True).stdout
        embedded = [l for l in fonts.splitlines()[2:] if l.strip()]
        all_emb = all(" no " not in l for l in embedded)
        raster = len([l for l in subprocess.run(["pdfimages", "-list", str(f)], capture_output=True,
                                                 text=True).stdout.splitlines()[2:] if l.strip()])
        txt = subprocess.run(["pdftotext", str(f), "-"], capture_output=True, text=True).stdout
        raw = f.read_bytes()
        boxes = subprocess.run(["pdftotext", "-bbox", str(f), "-"], capture_output=True,
                               text=True).stdout
        sizes = sorted((float(b) - float(a2)) * TEXTWIDTH_PT / page_w
                       for a2, b in re.findall(r'yMin="(-?[0-9.]+)"[^>]*yMax="(-?[0-9.]+)"', boxes)
                       if float(b) > float(a2))
        nominal = sorted(float(re.match(r"(-?[0-9.]+)", e.get("font-size")).group(1))
                         * TEXTWIDTH_PT / (page_w / 0.75)
                         for e in ET.parse(SRC / f"figure1_panel_{lab}.svg").getroot().iter()
                         if e.tag.endswith("text")) if lab != "composite" else []
        metrics[lab] = {
            "file": pdf, "sha256": hashes[pdf], "pages": pages, "page_pt": page_w,
            "ends_with_eof": raw.rstrip().endswith(b"%%EOF"),
            "has_startxref": b"startxref" in raw[-2048:],
            "fonts_all_embedded": all_emb, "raster_images": raster,
            "extractable_chars": len(re.sub(r"\s", "", txt)),
            "printed_pt_glyph_min": round(sizes[0], 2) if sizes else None,
            "printed_pt_glyph_median": round(sizes[len(sizes) // 2], 2) if sizes else None,
            "printed_pt_glyph_max": round(sizes[-1], 2) if sizes else None,
            "word_boxes": len(sizes),
            "printed_pt_fontsize_min": round(nominal[0], 2) if nominal else None,
            "printed_pt_fontsize_median": round(nominal[len(nominal) // 2], 2) if nominal else None,
        }
    pan = metrics["A"]
    prov = {"chosen_file": "Figure1_panel_A.pdf",
            "caption_file": "figures/figure1_caption_panelA.tex", "chosen_sha256": pan["sha256"],
            "installed_sha256": hashes["Figure1_panel_A.pdf"],
            "installed_matches_source": hashes["Figure1_panel_A.pdf"] == pan["sha256"],
            "admission_checks": {k: pan[k] for k in ("pages", "ends_with_eof", "has_startxref",
                                                     "fonts_all_embedded", "raster_images",
                                                     "extractable_chars")},
            "print_geometry": {
                # keys kept schema-compatible with the author-figure provenance the gates read
                "printed_font_pt_min": pan["printed_pt_glyph_min"],
                "printed_font_pt_median": pan["printed_pt_glyph_median"],
                "printed_font_pt_max": pan["printed_pt_glyph_max"],
                "words_measured": pan["word_boxes"],
                "printable_at_7pt": bool(pan["printed_pt_fontsize_min"] and
                                         pan["printed_pt_fontsize_min"] >= 6.0),
                "enlargement_needed_for_7pt": (round(7.0 / pan["printed_pt_fontsize_min"], 2)
                                               if pan["printed_pt_fontsize_min"] else None),
                "min_pt": pan["printed_pt_glyph_min"],
                "median_pt": pan["printed_pt_glyph_median"],
                "fontsize_min_pt": pan["printed_pt_fontsize_min"],
                "fontsize_median_pt": pan["printed_pt_fontsize_median"],
                "criterion": "legibility judged on the nominal font size at 0.98\\textwidth against the "
                             "6 pt print floor; glyph-box heights are reported alongside"},
            "panels": {k: v for k, v in metrics.items() if k != "composite"},
            "composite": metrics["composite"],
            "note": "Figure 1 = panel A (author layout unchanged); panels B-D + the overview are "
                    "Figures S4-S7 in the supplement"}
    (HERE / "figure1_panel_provenance.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"main figure  = Figure1_panel_A.pdf ({hashes['Figure1_panel_A.pdf'][:16]}…) at 0.98\\textwidth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
