#!/usr/bin/env python3
"""Overlap solver for the enlarged Figure 1 parts (author-authorized 2026-10-02).

Loop: enlarge (with per-element spacing pads) -> measure exact text-text overlaps with
`inkscape --query-all` -> add space below the upper element of every overlapping pair -> repeat.
Exits non-zero unless the final layout has zero introduced overlaps.

Baseline: the author's original and the un-enlarged split both measure 0 text-text overlaps,
so the target is objective.
"""
import argparse
import json
import pathlib
import subprocess
import sys


def query_all(svg):
    out = subprocess.run(["inkscape", "--query-all", str(svg)], capture_output=True, text=True).stdout
    boxes = {}
    for line in out.strip().splitlines():
        p = line.split(",")
        if len(p) == 5:
            try:
                boxes[p[0]] = tuple(float(v) for v in p[1:])
            except ValueError:
                pass
    return boxes


def text_overlaps(svg):
    boxes = query_all(svg)
    tids = sorted(k for k in boxes if k.startswith("text"))
    pairs = []
    for i in range(len(tids)):
        ax, ay, aw, ah = boxes[tids[i]]
        for j in range(i + 1, len(tids)):
            bx, by, bw, bh = boxes[tids[j]]
            ix = min(ax + aw, bx + bw) - max(ax, bx)
            iy = min(ay + ah, by + bh) - max(ay, by)
            if ix > 0.5 and iy > 0.5:
                upper, lower = (tids[i], tids[j]) if ay <= by else (tids[j], tids[i])
                pairs.append({"upper": upper, "lower": lower, "overlap_y": round(iy, 2),
                              "overlap_x": round(ix, 2)})
    return boxes, pairs


def run_transform(in_svg, out_svg, report, pads, mult, max_height, label, extra):
    cmd = ["python3", "_fig1_enlarge.py", "--in-svg", str(in_svg), "--out-svg", str(out_svg),
           "--report", str(report), "--mult", str(mult), "--max-height", str(max_height),
           "--label", label] + extra
    if pads:
        padfile = pathlib.Path(str(out_svg) + ".pads.json")
        padfile.write_text(json.dumps(pads))
        cmd += ["--pads", str(padfile)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r


def solve(in_svg, out_svg, mult, max_height, label, gap=3.0, max_iter=20, pad_cap=None):
    pads, history = {}, []
    for it in range(1, max_iter + 1):
        r = run_transform(in_svg, out_svg, pathlib.Path(str(out_svg) + ".report.json"),
                          pads, mult, max_height, label, [])
        if r.returncode != 0:
            print(f"  [{label}] iter {it}: transform failed rc={r.returncode}")
            print("   ", (r.stdout or "")[-300:].replace("\n", " | "))
            return {"label": label, "ok": False, "reason": f"transform rc={r.returncode}", "pads": pads}
        boxes, pairs = text_overlaps(out_svg)
        history.append(len(pairs))
        print(f"  [{label}] iter {it:2d}: text-text overlaps = {len(pairs)}"
              + (f" (pads on {len(pads)} elements)" if pads else ""))
        if not pairs:
            rep = json.loads(pathlib.Path(str(out_svg) + ".report.json").read_text())
            return {"label": label, "ok": True, "iterations": it, "overlap_history": history,
                    "pads": pads, "canvas_after_px": rep["canvas_after"],
                    "type_pt_min": rep["type_pt_min"], "wrapped": len(rep["wrapped"]),
                    "cards_grown": rep["cards_grown"]}
        for p in pairs:
            new = round(pads.get(p["upper"], 0.0) + p["overlap_y"] + gap, 2)
            pads[p["upper"]] = new if pad_cap is None else min(new, pad_cap)
    return {"label": label, "ok": False, "reason": "max iterations", "overlap_history": history, "pads": pads}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/tmp/fig1_work")
    ap.add_argument("--mult", type=float, default=2.07)
    ap.add_argument("--out", default="overlap_solve_report.json")
    a = ap.parse_args()
    res = []
    # panel-level solve: only panels A and C need a type bump (B and D already exceed 7 pt at full width)
    for label, src, mult, budget, cap in (("panelA", "panels/figure1_panel_A.svg", a.mult, 3000.0, 40.0),
                                          ("panelC", "panels/figure1_panel_C.svg", a.mult, 3000.0, 40.0)):
        print(f"--- 求解 {label} (mult={mult}) ---")
        res.append(solve(pathlib.Path(a.work) / src, pathlib.Path(a.work) / f"{label}_bumped.svg",
                         mult, budget, label, pad_cap=cap))
    pathlib.Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("\n结果:", json.dumps([{k: v for k, v in r.items() if k != "pads"} for r in res],
                                ensure_ascii=False, indent=2))
    sys.exit(0 if all(r["ok"] for r in res) else 9)
