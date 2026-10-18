# Paper M: code and aggregate-result public layer

STATUS: author-authorized public layer for proposed release v1.2.1; not yet released.

The original author-approved candidate allowlist contained 31 source files. This
subpackage includes 21 source files, five scope/provenance/inventory documents and
one hash manifest (27 files total). Ten non-portable scripts were excluded; their
original hashes and reasons are recorded in EXCLUDED_FROM_PUBLIC_TREE.md and .json.
Private originals are unchanged. The four included CSVs contain seed/site summaries.
No NPZ, individual-score CSV, individual-membership JSON or manuscript ZIP is included
in this added layer. Existing public repository artifacts outside it are unchanged.

## Reproducibility limits

This layer alone cannot perform the complete individual-level numerical recomputation.
The retained code/run_recompute.sh and code/evidence_20261017/recompute.py require
excluded participant predictions, main/control OOF archives, empirical-prior tables,
split masks and membership sets. No author-mediated access arrangement is asserted.
The evidence README inventories the full private numerical bundle; it does not mean
all listed inputs are included here. Some retained historical figure utilities also
require assets or helper scripts not included in this subset. Their presence is not
a claim that every retained script is a standalone executable entry point.

No E1 reinstatement, E2 rerun, new training or new clinical claim is introduced.
E3 is a scoring-time population-edge access comparison after the same source training
procedure; it does not establish equivalence or complete subject isolation.

## Authorization and provenance

The recorded author confirmation of 2026-10-17 covers publication of code/documentation
and the verified aggregate summaries in the original 31-file candidate list. The 21
retained source files are authorized; the ten others are excluded for portability,
not for lack of authorization. SOURCE_ALLOWLIST.csv distinguishes the historical
permission status from the current authorization status and records original and
public-copy hashes. Two aggregate CSVs underwent CRLF-to-LF normalization; they are
not byte-identical to their private originals. No project licence is extended to
third-party material.

The separately private 21 subject-level or collection files, two mixed submission ZIPs
and internal audit history are not release assets. The public layer is not advertised
as fully reproducible. Manuscript availability statements must match these limits.

## Version and archive

The proposed release is v1.2.1. The existing v1.2.0 draft is left untouched.
No v1.2.1 release or version-specific DOI is asserted before publication. The older
version DOI represents its stated historical scope. A GitHub/Zenodo repository
snapshot can include existing files outside this layer; this inventory describes
only reproducibility/20261017/.
