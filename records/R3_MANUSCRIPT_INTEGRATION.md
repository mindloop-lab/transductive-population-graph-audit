# R3 manuscript integration delta (prepared 2026-10-13, for one-pass application)

Baseline of record: `PAPERM_CMIG_SUBMISSION_PACK_20261011_R2/source/*.tex` (author's, untouched).
Everything below is a proposed delta with the exact text; nothing has been applied to the author's
files. Canonical numeric sources are named so no number is retyped from a rendered table.

## A. New results subsection (methods/results, new)

Insert as a new subsection after the existing intervention results, titled
`\subsection{Paired supervision-location experiment}`:

> We ran a paired experiment inside a single protocol. For every target site and fold, the supervised
> set had an identical total number of subjects and identical per-class counts in both arms; the graph
> construction, validation split, initialisation and training schedule were identical, and only the
> *location* of the supervision support differed: supplied by the target site (ON) or by other sites
> (OFF). Across 18 sites (88 site-fold cells, three fixed training seeds), the equal-weight mean of the
> within-site fold $\Delta$AUC was $+0.252$ with a site-resample 95\% interval $[+0.184, +0.322]$; the
> effect was positive at 17 of 18 sites (exact two-sided sign test $p = 1.5\times10^{-4}$), and the
> three training seeds gave $+0.217$, $+0.256$ and $+0.283$. Effect sizes varied substantially across
> sites and seeds. Because the identity of the support subjects and their covariate composition change
> together with location, this is not a clean causal isolation of supervision location; it shows that
> under this pairing the target site's own support is generally more useful to the target site.

Numbers come from `major1_population_inference.json`; per-site values and conditional intervals from
`major1_site_level.csv` (labelled conditional on the fixed model and seeds).

## B. Figures

* Insert `r3_figures/F1_onoff_protocol_schematic.pdf` (vector) as the protocol schematic for A.
* Insert `r3_figures/F2_site_delta_auc_forest.pdf` (vector) as the per-site forest plot; its footnote
  text already records: fewer than 5 folds -> SDSU; no feasible cell -> the sites listed in the caption
  of `MAJOR1_GRAPH_SEMANTICS.md` (CALTECH, CMU among them).
* Figure 4: the review is right that it currently mixes protocols. Split it by protocol, or move its
  core claim to the new ON/OFF figure. Do **not** leave the current mixed rendering as the primary
  mechanism figure.

## C. Scope sentences (add verbatim)

* Cross-architecture: "The comparison reflects the **evaluated implementations** under this harness
  and is not a claim about architecture families."
* LOSO: "'The LOSO value did not fall' is **not** by itself evidence of robustness."
* Parisot node structure (supplement): "The Parisot reference row is an independent re-implementation
  of the published design under this harness (no upstream code executed); each node carries the
  subject's reduced imaging representation, and message passing follows the shared directed Chebyshev
  propagation with self-loops and row normalisation described in `MAJOR1_GRAPH_SEMANTICS.md`."

## D. Statistics wording (blocked items stay blocked)

* Do not insert or re-derive any p value for the historical 13-item family; 6 of 13 raw p remain
  unlocatable (`STATISTICAL_FAMILY_13.md`).
* Do not cite the withdrawn tail-ratio p values or the per-site p values
  (`MAJOR1_ANALYSIS_REPORT.md`).
* Replace blanket "descriptive" wording with the three-state scheme per item once the family closes.

## E. Protocol deviation statement (must appear)

State that the pre-specified endpoint was a site-clustered bootstrap, that the delivered first
implementation resampled within sites, that the pre-specified endpoint was implemented afterwards
(`major1_population_inference.py`) and is therefore post-hoc, and that per-site intervals are
conditional. Record it as a deviation, never as pre-registration.

## F. Rebuild both PDFs from the revised source

Rebuild main and supplement together, ship both PDFs with their source digests, and record that the
Oct-11 PDF is a stale rendering (see `PDF_REBUILD_VERIFICATION.md`). No number may change without a
named canonical source file and digest.
