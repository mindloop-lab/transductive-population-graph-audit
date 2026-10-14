# R3 corrections (2026-10-11) — in response to the external review feedback

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
