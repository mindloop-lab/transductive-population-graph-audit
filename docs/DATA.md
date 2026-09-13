# Data statement

* **ABIDE-I** (Autism Brain Imaging Data Exchange) is a public, de-identified dataset;
  request access from the ABIDE consortium. The study uses 871 subjects (403 ASD) across
  20 sites, with two preprocessing pipelines (C-PAC and NIAK `filt_noglobal`) and
  atlas-derived features (AAL-116, CC200; Harvard-Oxford context on C-PAC).
* **What this repository redistributes:** derived quantities only - per-subject
  out-of-fold predictions, the frozen split manifest, the result registry, and aggregate
  statistics. **No primary imaging data are redistributed.**
* **Cache format:** `subject_ids` (length N), `fused_x` (N, 116, 141) containing the
  Fisher-z connectome rows plus per-node context features, and `adj` (N, 116, 116) with
  the connectivity matrices.
* **Ethics:** secondary analysis of a public de-identified dataset; no new participant
  data were collected.
