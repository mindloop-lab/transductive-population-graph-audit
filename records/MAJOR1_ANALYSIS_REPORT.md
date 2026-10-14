# Major 1 analysis report (2026-10-13) — endpoints, deviation, withdrawn tests

## Primary endpoint (as pre-specified in PROTOCOL_LOCK.md)

Equal-weight mean over the 18 sites of the within-site fold macro of (AUC_ON - AUC_OFF), per training
seed, averaged over the three fixed seeds. Implemented in `major1_population_inference.py`, which is a
**post-hoc** implementation of the pre-specified endpoint (it was specified before the runs; it was
implemented after them). It is not a pre-registered analysis write-up.

* population mean **+0.2519**; per seed **+0.2168 / +0.2555 / +0.2832**
* site-resample bootstrap 95% CI **[+0.1841, +0.3222]** (B = 10,000, seed 20261012, sites resampled)
* null-centred site bootstrap, two-sided p **< 1e-4** (floor 1/(B+1); an exact zero is never reported)
* exact two-sided sign test **17/18, p = 1.45e-4**
* independent recomputation by the external reviewer agrees to four decimals

## Secondary endpoint

Within-site conditional intervals (`major1_site_level.csv`): the uncertainty of a site's test-subject
sample **conditional on the fixed trained model and the fixed seed set**. These are the correct
intervals for describing a single site; they are **not** population intervals and must not be labelled
as such.

## Withdrawn (do not cite, do not correct)

* **tail-ratio bootstrap p values** and their Holm adjustments: the ordinary bootstrap distribution is
  centred on the observed effect, so tail proportions are not a calibrated hypothesis test. Withdrawn
  from any inference; a column carrying them must not be quoted even descriptively as a p value.
* **per-site p values / per-site Holm**: computing them from the across-site distribution conflates two
  resampling units and flags all 18 sites. Withheld until a valid per-site test (that site's own
  subject-level unit with a stated null) is defined; multiplicity correction is then applied to that
  valid family only.

## Scheme deviation, recorded as required

`PROTOCOL_LOCK.md` pre-specified a **site-clustered** bootstrap. The first implementation
(`major1_aggregate.py`) resampled **subjects within sites**, which answers a different question. The
pre-specified endpoint is now implemented in `major1_population_inference.py`; the earlier per-site
intervals are retained but relabelled as conditional. Terminology is fixed as: **site macro** =
within-site fold mean; **population macro** = equal-weight mean over sites. The two are never mixed.

## Recommended manuscript sentence (subject to the covariates caveat)

> Under this paired protocol, replacing other-site supervision support with an equal-sized,
> class-matched support set drawn from the target site generally improved target-site discrimination,
> with substantial between-site and between-seed variation.

The caveat stays attached: the identity of the support subjects and their covariate composition change
together, so this is not a clean causal isolation of supervision *location*.
