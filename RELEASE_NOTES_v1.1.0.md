# v1.1.0

## Title metadata

Project title unified to: **Auditing Site-Dependent Performance in Transductive Population Graph Neural Networks for Multisite Autism fMRI** (README, `.zenodo.json`, `CITATION.cff`).

## Documentation accuracy

* README performance wording no longer uses "falls to chance"/"localise": it now reports the
  leave-one-site-out values (0.522 C-PAC, 0.532 NIAK) with confidence intervals including 0.5 and no
  clear evidence of unseen-site discrimination, and attributes the cohort-level dependence to
  site-structured connectivity plus the availability of same-site supervision under the tested
  interventions.
* The `.zenodo.json` description and the CITATION abstract were aligned with the same wording.

## Tooling

* `scripts/scan_secrets.sh` extracts archives with `--no-same-owner`, so the scan no longer depends on
  the extracting user being able to reproduce archived owner ids.

## Artefacts

* Frozen results, the result registry and the redistributed predictions are unchanged from v1.0.0
  except for the documentation/metadata edits listed above.
* Concept DOI 10.5281/zenodo.22748743 is reused; v1.0.0 (10.5281/zenodo.22748744) is not modified.
* Version DOI **10.5281/zenodo.22892320** (reserved in the Zenodo draft, registered when the release is published).

## Author metadata

* ORCID iDs added for Ning Zhong (0000-0001-7882-8340) and Hongzhi Kuai (0000-0002-2746-648X) in
  `.zenodo.json`, `CITATION.cff` and the manuscript ORCID section; the existing ORCIDs for Zongqing
  Chen and Fanghong Zhang are retained. No ORCID value was invented for the other four authors.
