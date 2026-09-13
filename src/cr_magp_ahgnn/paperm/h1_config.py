"""H1 frozen representation config (docs/H1_representation_freeze.md).

Freezes the canonical input contract and the producing artifact SHA so any
change to canonical features / ordering / split is detectable. This is the
INPUT contract; SubjectEncoder (B) turns it into 64-D subject embeddings at
train time.
"""
from __future__ import annotations

REPRESENTATION_ID = "B-SubjectEncoder/64/AAL7+CC200HO20@fold0-seed666"

# --- feature contract ---
NODE_FEATURES_DIM = 7      # pure AAL: [deg_w,deg_c,eig,clus (AAL) ... wait 5 topo? see below]
# NOTE: node_features = 5 AAL topology metrics (deg_w,deg_c,eig,clus + bet)
#       + row mean/std = 7. 'clus' named; bet is col index 3 (exact, deterministic).
CONTEXT_FEATURES_DIM = 20  # CC200(mean,std x5) + HO(mean,std x5)
HIDDEN = 32
EMBEDDING_DIM = 64         # concat(AAL pool 32, context 32)
THRESHOLD = 0.5            # arctanh-correlation edge / topology threshold
BETWEENNESS = "exact"      # deterministic; recovered legacy k=20 was unseeded

# --- split / order authority ---
SEED = 666
FOLD = 0
SPLIT_SHA256 = "45539f252a822f5267face1b52d7c3bb2c5fed6b90020442f3340aaff7d7cce4"
SUBJECT_IDS_SHA256 = "7cb85ff412ed55532c04d6576dba04a8a7dd73206f351c0bf0aae09dc01a16a5"
# artifact over canonical-derived inputs (node feats + context + edges), 871 subjects fold-0
REP_ARTIFACT_SHA256 = "3d8415a3b89ae367a355b627507aa5c1df11b36874149fd7d98afd5e83454e81"
REP_ARTIFACT_PATH = "artifacts/paper_m/h1_rep_fold0_seed666.pt"
