# Paper M: code and aggregate-result candidate

STATUS: review candidate; publication rights have not been approved.

This tree contains exactly 31 byte-identical code/aggregate source files selected
by explicit relative path and SHA256 from the reviewed private candidate. It adds
three scope/inventory documents and a final hash manifest. It includes no NPZ,
no individual-score CSV, no individual membership JSON and no nested manuscript ZIP.
The four included CSVs are seed/site-level summaries. JSON is not inherently public:
the subject-membership manifest was deliberately excluded.

## Reproducibility limits

This public candidate ALONE cannot run the complete individual-level numerical
recomputation. code/run_recompute.sh and code/evidence_20261017/recompute.py retain
their original paths and behavior, but require excluded inputs. Their presence is
not a promise of a self-contained executable public release.

Missing categories include E3 participant predictions, main/control OOF archives,
site-prior per-participant tables, split masks and membership sets. No mechanism
for obtaining those inputs, release permission or author-mediated access is asserted.
The original evidence README describes the full private numerical bundle and must
be read as an input inventory, not as proof all inputs are included in this subset.

Some historical plotting scripts have external dependencies. The E3 original driver
is not a portable training entry point. No new fit, new model claim, clinical claim,
E1 reinstatement or E2 rerun is introduced here.

## Publication review

Verify source ownership and applicable licenses for the actual 31-path inventory.
No Apache/MIT/data license is newly assigned by this packaging operation. Do not
publish this tree as "fully reproducible" or publish the original private ZIP as a
release asset. If authorized for publication, make the repository README and journal
availability statements agree with these actual input-access limits.

## Version

This layer is released as **v1.2.1**. The earlier v1.2.0 draft release was never published and is not
replaced by this one. A version-specific DOI is minted by the archive at release time and is not quoted here
before it exists; the previous version DOI describes the earlier scope only.

## Counts and authorisation

- Original candidate allowlist: **31** files.
- Published in this subpackage: **21** source files (+ scope notes and manifests).
- Excluded for portability: **10** scripts that embedded machine-local paths (listed with their original
  hashes in EXCLUDED_FROM_PUBLIC_TREE.md); their private originals are unchanged.
- Separately private and not part of any public layer: **21** subject-level or collection files.

Author confirmation (2026-10-17): code and documentation may be published, and the derived aggregate
summaries were verified as publishable. The project licence is not extended to third-party material. The
authorisation concerns the 31 candidates; the 10 excluded files are excluded rather than unauthorised.
