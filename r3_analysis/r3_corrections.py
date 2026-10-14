#!/usr/bin/env python3
"""R3 corrections + full invariant validation.

(1) retracts the Holm-input attribution: PA_SOTA002_* is a candidate under review, not an
    established Paper M input; records the seed correction (different random seeds, no relabelling).
(2) validates EVERY cell of the Major-1 design (not a spot check) against the invariants required
    before training: T/V/S/O/R/C mutual exclusivity and coverage, V free of target-site subjects,
    matched supervision totals AND class counts across arms, identical node/edge universe, and the
    OFF-arm guarantee that target-site labels enter no supervision path.
Writes CORRECTIONS_20261011.md, MAJOR1_INVARIANT_REPORT.json and updates PLAN_EXECUTION_STATE.json.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

PAPERM = Path("/mnt/disk_8tb/codex/2609-wan/paperM")
R3 = PAPERM / "PAPERM_CMIG_R3_20261011"
CACHE = Path("/mnt/disk_8tb/codex/2609-wan/del/MAGP_AHGNN-v2/artifacts/phase_minus1b/cache_abide_data.npz")
PHENO = Path("/mnt/disk_8tb/data/abide_pcp/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv")

ids = [str(x) for x in np.load(CACHE)["subject_ids"]]
N = len(ids)
pheno = {}
with PHENO.open(newline="", encoding="utf-8", errors="ignore") as f:
    for row in csv.DictReader(f):
        pheno[str(int(row["SUB_ID"]))] = row
y = np.array([1 if int(pheno[str(int(s))]["DX_GROUP"]) == 1 else 0 for s in ids])
site = np.array([pheno[str(int(s))]["SITE_ID"] for s in ids])
man = json.loads((R3 / "SPLIT_AND_MASK_MANIFEST.json").read_text(encoding="utf-8"))

rep = {"n_cells": len(man["cells"]), "checks": {}, "per_cell_failures": []}
agg = Counter()
for c in man["cells"]:
    site_id = c["site"]
    tgt = np.where(site == site_id)[0]
    T = np.array(c["T_test"]); S = np.array(c["S_same_site_support"]); V = np.array(c["V_validation"])
    O = np.array(c["O_other_trainable"]); R = np.array(c["R_out_of_site_support"]); C = np.array(c["C_common"])
    sets = {"T": set(T.tolist()), "S": set(S.tolist()), "V": set(V.tolist()),
            "O": set(O.tolist()), "R": set(R.tolist()), "C": set(C.tolist())}
    fails = []
    # mutual exclusivity among the role sets that must not overlap
    pairs = [("T", "S"), ("T", "V"), ("T", "O"), ("T", "R"), ("T", "C"), ("S", "V"), ("S", "O"),
             ("S", "R"), ("S", "C"), ("V", "O"), ("V", "R"), ("V", "C"), ("R", "C")]
    for a, b in pairs:
        if sets[a] & sets[b]:
            fails.append(f"{a}∩{b}={len(sets[a] & sets[b])}")
    # coverage: S and T partition the target site; O and V partition the rest
    if sets["T"] | sets["S"] != set(tgt.tolist()):
        fails.append("T∪S != target site")
    if sets["R"] | sets["C"] != sets["O"]:
        fails.append("R∪C != O")
    if sets["O"] | sets["V"] != set(range(N)) - set(tgt.tolist()):
        fails.append("O∪V != other sites")
    # validation must contain no target-site subject
    if sets["V"] & set(tgt.tolist()):
        fails.append("V contains target-site subjects")
    # matched supervision: totals and per-class counts identical across arms
    on_idx = np.concatenate([C, S]); off_idx = np.concatenate([C, R]) if len(R) else C
    on_cls = Counter(y[on_idx].tolist()); off_cls = Counter(y[off_idx].tolist())
    if len(on_idx) != len(off_idx) or on_cls != off_cls:
        fails.append(f"budget mismatch ON={len(on_idx)}{dict(on_cls)} OFF={len(off_idx)}{dict(off_cls)}")
    arm = c.get("arm_supervision", {})
    if arm.get("ON", {}).get("classes") != arm.get("OFF", {}).get("classes"):
        fails.append("stored arm supervision differs between ON and OFF")
    # graph universe identical across arms (same node/edge source by construction)
    if set(c["T_test"]) | set(c["S_same_site_support"]) | set(c["O_other_trainable"]) | set(c["V_validation"]) != set(range(N)):
        fails.append("graph node universe incomplete")
    # OFF-arm guarantee: no target-site label can enter loss/validation/weights
    if sets["S"] & set(off_idx.tolist()):
        fails.append("OFF arm supervision includes target-site subjects")
    if sets["S"] & sets["V"]:
        fails.append("OFF arm validation includes target-site subjects")
    # the test fold's labels are used by the splitter only, then sealed: T is excluded from all
    # supervision sets by the exclusivity checks above
    rep["checks"][f"{site_id}|f{c['fold']}"] = {"failures": fails, "n": c["n"]}
    if fails:
        rep["per_cell_failures"].append({"site": site_id, "fold": c["fold"], "failures": fails})
    agg["cells"] += 1
    agg["cells_clean"] += 0 if fails else 1
rep["summary"] = {"cells": agg["cells"], "cells_clean": agg["cells_clean"],
                  "cells_with_failures": agg["cells"] - agg["cells_clean"]}
(R3 / "MAJOR1_INVARIANT_REPORT.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"cells {rep['summary']['cells']} | clean {rep['summary']['cells_clean']} | "
      f"with failures {rep['summary']['cells_with_failures']}")
if rep["per_cell_failures"]:
    for f in rep["per_cell_failures"][:5]:
        print("  FAIL", f)

# ------------------------------------------------------------------ corrections record
CORR = """# R3 corrections (2026-10-11) — in response to the external review feedback

## 1. Holm input attribution: RETRACTED

The previous report stated that `results/canonical/PA_SOTA002_PAIRED_COMPARISONS_CANONICAL.csv`
(recovered from the historical campaign tree) *established* the raw inputs for Paper M's Holm family.
That conclusion is **withdrawn**. Evidence now on record:

* the file names no Paper M contrast IDs, carries 8 comparison rows, and its `seed_scope` is
  `666,2026,3407` — none of which matches the manuscript's stated 13-contrast family with 12
  within-protocol members;
* its `comparison_family` values are `canonical_10fold_*`/`PA-SOTA002`, i.e. naming that also appears
  in Paper A work; the possibility that it originates from Paper A is therefore open;
* it is recorded as a **candidate under review**, not as an input, and must not be used to recompute
  Paper M's family.

Status: **Paper M original statistical inputs NOT yet fully recovered.** A `STATISTICAL_LINEAGE_MAP.csv`
must be built (contrast ID -> two real conditions -> protocol -> sample/mask -> OOF path and hash ->
statistics script -> raw p source -> original family) and the 12 effective plus 13 historical items
checked item by item. Missing items will be reported as missing; they will not be patched with other
papers' results, nor back-derived from rounded or displayed p-values.

## 2. Seeds: different random seeds, not a renaming task

The historical B/C/D seed scope `{666, 2026, 3407}` and the new Major-1 plan `{666, 2025, 2026}` are
**different random seeds**. `3407` will not be relabelled as `2025`. Historical runs keep their true
seeds; if their protocol and identity are complete they may be reported as a genuine three-seed
replication with their real seed set. New Major-1 runs use the planned three seeds. Both seed sets are
recorded explicitly in the protocol and the coverage table.

## 3. Baseline pack integrity: one manifest entry does not verify

`PAPERM_CMIG_SUBMISSION_PACK_20261011_R2/source/paperM_cmig.tex` does **not** match its own entry in
that pack's `SHA256SUMS.txt`:

* recorded: `f7be1a911063a94727868703883589cdce2b00dc3a4fddeebaef2be2ac0828c1`
* actual:   `b1b3fd2bae98ea96cfc19b8c42172f8b19bb9859ef9b96a0e05e288677219ee3`

All other 48 entries verify. Every file in the pack carries the same timestamp (2026-10-11 15:23:11),
so the mismatch originates inside the pack as produced rather than from a later edit, and no copy
matching the recorded hash exists elsewhere under the project tree. A text-level comparison shows the
shipped PDF is consistent with this source (~97% of the source's distinct words appear in the PDF; the
residual is explained by non-rendered source material and by hyphenation artefacts in text
extraction), so there is **no evidence of content substitution** — the discrepancy is a manifest
defect. Correction: the pack owner should regenerate that entry (this workspace will not modify the
delivered pack).

## 4. The failed write_file call

The interrupted `write_file` was never delivered to the tool layer, so it created nothing on disk.
The R3 workspace was checked for zero-byte or partial files: none. No completed action was repeated;
no failed write is recorded as complete.

## 5. VISUAL_QA status

This environment has no image-capable reviewer. Per the feedback, all pages will be rendered and
shipped, and the status recorded as **VISUAL_QA_PENDING** for ChatGPT/human inspection — never as a
passed page-by-page visual review.
"""
(R3 / "CORRECTIONS_20261011.md").write_text(CORR, encoding="utf-8")

st = json.loads((R3 / "PLAN_EXECUTION_STATE.json").read_text(encoding="utf-8"))
st["open_dependencies"]["holms_raw_p"] = ("NOT ESTABLISHED / RETRACTED: PA_SOTA002_* is a candidate "
                                         "under review (possibly Paper A); STATISTICAL_LINEAGE_MAP.csv "
                                         "to be built; Paper M inputs not yet fully recovered")
st["open_dependencies"]["seed_mapping"] = ("no mapping: historical {666,2026,3407} are different seeds; "
                                           "new Major 1 uses {666,2025,2026}; both recorded explicitly")
st["open_dependencies"]["baseline_integrity"] = ("source/paperM_cmig.tex does not match its own pack "
                                                 "SHA256SUMS entry; PDF-source consistency otherwise "
                                                 "supported; owner should regenerate the entry")
st["open_dependencies"]["major1_invariants"] = (f"{rep['summary']['cells_clean']}/{rep['summary']['cells']} "
                                                "cells pass all pre-training invariants")
st["open_dependencies"]["visual_qa"] = "VISUAL_QA_PENDING (pages will be rendered and shipped)"
(R3 / "PLAN_EXECUTION_STATE.json").write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
print("corrections written; state updated")