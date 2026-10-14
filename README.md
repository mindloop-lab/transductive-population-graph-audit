# Policy-audit evaluation of a transductive population graph (R3 revision)

What this repository contains
  The code and result records behind a study that separates **cohort-conditioned discrimination** from
  **unseen-site generalization** for a site-aware population graph neural network, and that reports an
  **implementation-sensitivity finding** for a paired supervision-location experiment.

Headline results (each traceable to a named file)
  * cohort pooled out-of-fold AUC ~0.94 under the site-aware graph, with leave-one-site-out falling to
    0.522 (C-PAC) / 0.532 (NIAK) and intervals spanning 0.5;
  * a paired supervision-location experiment (18 sites, 88 site-fold cells, 3 fixed seeds, matched
    supervision totals and class counts) gave a population macro of +0.252 [+0.184, +0.322] with 17/18
    sites positive under the **directed** edge construction used for the primary analysis;
  * re-running the same experiment with the graph **symmetrised** gave -0.003 [-0.020, +0.015] with 8/18
    sites positive. The paired effect is therefore **implementation-conditional** and is reported as
    such in the text, the abstract and the limitations.
  * a pre-declared support-sampling sensitivity shows supervision location and support composition are
    **not separable** at two of three tested sites.

Reproducibility: what is and is not included
  * **Recompute from saved predictions: yes.** Every per-fit report and OOF array is in the evidence
    pack; the aggregation, population-endpoint and comparison scripts are included and refuse to emit
    numbers from an incomplete run.
  * **Rebuild training inputs from ABIDE-I raw data: not included.** The training tree expects a local
    data cache and absolute server paths; that pipeline is documented but not bundle-ready here. Do not
    describe this bundle as full end-to-end reproduction.

Known corrections (see CHANGELOG.md)
  Withdrawn: tail-ratio bootstrap p values and per-site p/Holm; a graph-semantics description that named
  the wrong module; a support-sampling explanation about the graph that the code does not support.
