# Reproducibility limits (read before using this bundle)

1. Training inputs are rebuilt from a local cache and absolute server paths; the raw-data rebuild
   pipeline is **not** bundled. What is bundled is everything needed to **recompute the reported
   statistics from saved predictions**.
2. The paired experiment, the symmetrised parallel analysis and the support-sampling sensitivity use
   different edge-construction paths from the original cohort model. Orientation findings are **not**
   transferable between them, in either direction.
3. Per-site intervals are conditional on the fitted model and the fixed seed set; they exclude
   training-seed variability and are narrower than total uncertainty.
4. No claim of orientation invariance, of a pure causal effect of supervision location, or of a
   general property of population-GNNs is made anywhere in these materials.
