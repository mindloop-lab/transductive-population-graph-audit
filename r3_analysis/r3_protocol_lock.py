#!/usr/bin/env python3
"""PROTOCOL_LOCK + hash-lock cancellation record.

The author states that files in the delivery directory were edited by hand, and instructs that hash
locking be CANCELLED. Accordingly the protocol is locked by content definition and by named
references, and every recorded digest is marked as an observational identifier only -- not an
immutability or integrity claim.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

PAPERM = Path("/mnt/disk_8tb/codex/2609-wan/paperM")
R3 = PAPERM / "PAPERM_CMIG_R3_20261011"
man = json.loads((R3 / "SPLIT_AND_MASK_MANIFEST.json").read_text(encoding="utf-8"))
inv = json.loads((R3 / "MAJOR1_INVARIANT_REPORT.json").read_text(encoding="utf-8"))

CANCEL = """# Hash locking: CANCELLED (author instruction, 2026-10-11)

The author states that part of the delivery directory was modified by hand, and instructs that hash
locking be cancelled. Effect on this workspace:

* **No digest in this workspace is an enforcement or immutability claim.** Digests are kept only as
  observational identifiers for the state observed at the recorded time.
* The earlier baseline finding is now explained: `PAPERM_CMIG_SUBMISSION_PACK_20261011_R2/source/
  paperM_cmig.tex` (hash `b1b3fd2b…`) does not match that pack's own `SHA256SUMS.txt` entry
  (`f7be1a91…`) because the file was hand-edited before the pack's manifest was finalised. This is a
  provenance note, not a claim of corruption, and no content substitution was detected
  (text-level PDF/source agreement ~97% of distinct words).
* Verified during this session: the baseline pack's directory-level content digest was unchanged from
  the record taken when work started, so nothing changed while this workspace was reading it.
* Consequence for the protocol lock: the lock below binds the design **by content definition and by
  named reference**, not by digest immutability. Any later change to a locked input must be reported
  as an amendment rather than being detected by a failing checksum.
"""
(R3 / "HASH_LOCK_CANCELLED.md").write_text(CANCEL, encoding="utf-8")

LOCK = f"""# PROTOCOL_LOCK — Paper M R3, Major 1 (supervision-location experiment)

Locked at: {time.strftime('%Y-%m-%d %H:%M:%S %z')}
Status: **PROTOCOL_LOCKED** (hash locking cancelled by author instruction — see HASH_LOCK_CANCELLED.md)

## What this experiment estimates

Whether, with the training-label budget (total and per class) held equal and the graph held
identical, the *location* of supervised support — same-site versus out-of-site — changes the
predictions for target-site test subjects. It is a new, protocol-internal paired experiment. It is
not a re-analysis of the cohort 10-fold or half-site results, and its numbers are not comparable to
those.

Estimand: sensitivity of predictions to the location of supervised support under this protocol.
NOT claimed: literal label propagation, cross-architecture generality, unseen-site validity.

## Design (locked)

| Element | Value |
|---|---|
| Pipeline | C-PAC (primary). NIAK not required for this experiment. |
| Sites | all 20; eligible = both classes >= 5 -> **20 eligible** |
| Target-site split | stratified 5-fold within the target site (splitter seed {man['seeds']['split']}) |
| Validation V | fixed per target site, drawn only from **other sites**, identical across folds and arms |
| Support S | target-site subjects not in the test fold |
| Support R | sampled from other-site trainable subjects with **per-class counts equal to S** |
| Common C | O \\ R |
| Arms | SAME_SITE_ON (loss labels from C ∪ S) and SAME_SITE_OFF (loss labels from C ∪ R) |
| Matched | total supervised count **and** per-class counts identical across arms |
| Graph | identical nodes and edges in both arms (same graph construction and same node set) |
| Preprocessing | fitted on the common eligible training pool S ∪ O; V and T are transform-only |
| Test labels | used by the splitter only, then sealed; passed to the evaluator only |
| Training seeds | {man['seeds']['training']} (three paired seeds; the same seeds in both arms) |
| Support sampling seed | {man['seeds']['support_sampling']} (one prespecified draw; composition sensitivity not covered unless repeated later) |
| Folds x cells | {man['n_cells_total']} cells, **{man['n_cells_feasible']} feasible** |
| Design cap | {man['design_cap_fits']} fits (20 x 5 x 2 arms x 3 seeds) |
| Primary metric | site-macro paired AUC difference per seed, averaged over the three seeds |
| Secondary metric | pooled OOF AUC (sites fitted independently; pooling limits apply) |
| Reporting | per seed and per site; one OOF per subject per seed (never 3N independent subjects) |
| Intervals | site-clustered bootstrap sharing one resampling index per paired difference; subject bootstrap as secondary sensitivity; B = 10000 |
| Inferential family | to be defined in the protocol for the new experiment; it is NOT added to the historical Holm table |

## Pre-training invariants (verified for every cell, not sampled)

{inv['summary']['cells_clean']} / {inv['summary']['cells']} cells pass all of: T/S/V/O/R/C mutual
exclusivity and coverage; V contains no target-site subject; matched supervision totals and class
counts across arms; complete graph node universe; **and the OFF-arm guarantee that no target-site
subject label can enter the loss, the validation path, class weights, sampling weights or any
auxiliary training objective**. Report: `MAJOR1_INVARIANT_REPORT.json`.

## Training rules (locked before the first fit)

* Identical initialisation, algorithm RNG and training schedule in both arms of a paired cell;
  optimiser state is stored separately per arm and **no arm is warm-started from the other**.
* Class weights (if used) are computed **only** from the matched training-loss labels, never from
  the full cohort.
* Node weights, loss normalisation, batch/step counts and the schedule are identical across arms.
* The two arms may stop at different epochs; the actual stop epoch and update count are recorded and
  reported.
* Model selection uses the fixed validation set V only. Target-site labels never re-enter through
  validation.
* No seed may be changed to obtain a preferred result; failed runs are retried with the original
  configuration and log.

## Explicitly not inferable from this experiment

* It is not a single-variable causal isolation: the training-label composition necessarily differs
  between arms, and matching counts does not remove every covariate difference.
* It cannot exclude training or site-population sampling uncertainty (intervals are conditional on
  the fitted models).
* Missing/unrecoverable items are recorded as missing; nothing is imputed or borrowed.
"""

(R3 / "PROTOCOL_LOCK.md").write_text(LOCK, encoding="utf-8")

st = json.loads((R3 / "PLAN_EXECUTION_STATE.json").read_text(encoding="utf-8"))
st["state"] = "PROTOCOL_LOCKED"
st["entered"] = time.time()
st["open_dependencies"]["hash_locking"] = "CANCELLED by author instruction (hand-edited delivery dir)"
st["open_dependencies"]["protocol_lock"] = ("PROTOCOL_LOCK.md written; lock binds content by definition "
                                            "and reference, not by digest immutability")
st["open_dependencies"]["major1_smoke_budget"] = "not measured yet (next step)"
(R3 / "PLAN_EXECUTION_STATE.json").write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
print("state:", st["state"])
for f in ("HASH_LOCK_CANCELLED.md", "PROTOCOL_LOCK.md"):
    print(f"  {f}: {(R3 / f).stat().st_size} B")