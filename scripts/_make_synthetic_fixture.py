#!/usr/bin/env python
"""Create a tiny synthetic ABIDE-style fixture so the kit runs end-to-end without the
(public, but large) real data.

The fixture carries a weak, site-independent label signal so that every stage returns a
finite AUC; it is a *plumbing* test only and must never be used for scientific numbers.
"""
import os
import numpy as np
from pathlib import Path

OUT = Path(os.environ.get("PAPERM_OUT", Path(__file__).resolve().parents[1] / "results" / "synthetic"))
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
N, R, F = 60, 116, 141
ids = np.array([f"{50000+i:07d}" for i in range(N)], dtype="U32")
labels = np.array([1 if (i % 8) < 4 else 2 for i in range(N)], dtype=int)  # 1 = ASD; both classes in every site
sex = np.array([1 if i % 3 else 2 for i in range(N)], dtype=int)
sites = [f"SYNTH_{i % 4}" for i in range(N)]

adj = rng.random((N, R, R)).astype(np.float32)
adj = (adj + adj.transpose(0, 2, 1)) / 2
for i in range(N):
    np.fill_diagonal(adj[i], 1.0)
fused = rng.random((N, R, F)).astype(np.float32)
# weak label signal: ASD subjects get a small positive shift on a subset of features,
# and the connectivity of same-site subjects is correlated, so both graph and features
# carry usable (synthetic) structure.
for i in range(N):
    if labels[i] == 1:
        fused[i, :, :40] += 0.35
for s in set(sites):
    idx = [i for i, x in enumerate(sites) if x == s]
    for a in idx:
        for b in idx:
            if a < b:
                adj[a, b] = adj[b, a] = np.clip(adj[a, b] + 0.25, 0, 1)

np.savez(OUT / "cache_abide_data.npz", subject_ids=ids, adj=adj, fused_x=fused)
with open(OUT / "Phenotypic_V1_0b_preprocessed1.csv", "w") as f:
    f.write("SUB_ID,SITE_ID,DX_GROUP,SEX\n")
    for i in range(N):
        f.write(f"{50000+i},{sites[i]},{labels[i]},{sex[i]}\n")
print("synthetic fixture ->", OUT)
print("cache keys: subject_ids/adj/fused_x;  phenotype: SUB_ID,SITE_ID,DX_GROUP,SEX")
