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
