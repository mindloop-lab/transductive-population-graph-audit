"""H1 B-path shared inductive M3 machinery (NO legacy SubjectGNN).

Refactored from the preflight-verified smoke. SubjectEncoder(7,20)->64-D is
trained end-to-end with H1PopulationHead(in_dim=64) over SITE-free SEX
same/diff edges. Reference = optimization-train; val/test subjects are attached
one-way as single queries (query never a source); test yields exact-once OOF.
Nothing here runs the formal campaign by itself; callers pass the fold/seed and
write run-specific artifacts.
"""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path
import numpy as np
import torch

from cr_magp_ahgnn.models.subject_encoder import SubjectEncoder
from cr_magp_ahgnn.paperm import h1_config, h1_edges
from cr_magp_ahgnn.paperm.h1_model import H1PopulationHead, LabelSmoothCE, attach_query

CHUNK = 128

def load_fold_inputs(root: Path, seed: int, fold: int, verify_sha: bool = True):
    """Load frozen canonical-input artifact + aligned sex/labels; return dict."""
    root = Path(root)
    art = torch.load(root / h1_config.REP_ARTIFACT_PATH, map_location="cpu", weights_only=True)
    if verify_sha:
        h = hashlib.sha256()
        for c in iter(lambda: (root / h1_config.REP_ARTIFACT_PATH).open("rb").read(1 << 20), b""):
            h.update(c)
        assert h.hexdigest() == h1_config.REP_ARTIFACT_SHA256, "artifact SHA mismatch"
    assert art["fold"] == fold and art["seed"] == seed
    fix = torch.load(root / "artifacts/phase0b/real_fold0_fixture.pt",
                     map_location="cpu", weights_only=True)
    return {
        "X": art["node_features"], "C": art["topology_context"], "EI": art["edge_index_list"],
        "tr": np.asarray(art["indices"]["optimization_train"]),
        "va": np.asarray(art["indices"]["validation"]),
        "te": np.asarray(art["indices"]["test"]),
        "sex": fix["sex"].numpy(), "labels": fix["labels"].numpy(),
    }

def build_models():
    enc = SubjectEncoder(h1_config.NODE_FEATURES_DIM, h1_config.CONTEXT_FEATURES_DIM,
                         h1_config.HIDDEN)
    head = H1PopulationHead(in_dim=h1_config.EMBEDDING_DIM)
    return enc, head

def embed(enc, idx, F, device="cpu"):
    """Embed subjects (batchable) -> (k, 64). idx: np array (int)."""
    idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
    outs = []
    X, C, EI = F["X"], F["C"], F["EI"]
    for s in range(0, len(idx), CHUNK):
        part = idx[s:s + CHUNK]; k = len(part)
        x = torch.cat([X[i] for i in part], dim=0).to(device)
        ei = torch.cat([EI[i] + 116 * p for p, i in enumerate(part)], dim=1).to(device)
        batch = torch.arange(k, device=device).repeat_interleave(116)
        ctx = C[part].to(device)
        outs.append(enc(x, ei, batch, ctx))
    return torch.cat(outs, dim=0)

def _sq_acc(enc, head, ref_emb, ref_edges, ref_sex, va_idx, labels, F, device):
    """Val single-query accuracy over a sample of val subjects (frozen reference)."""
    with torch.no_grad():
        correct = 0
        for qi in va_idx:
            qe = embed(enc, [qi], F, device)[0]
            q_edges = h1_edges.query_attachment(ref_sex, int(F["sex"][qi]), query_index=ref_emb.shape[0])
            out = attach_query(head, ref_emb, ref_edges, qe, q_edges)
            correct += int(out.argmax(1)) == int(labels[qi])
    return correct

def train_fold(F, enc, head, tr_idx, va_idx, te_idx, *, seed=666, epochs=20, device="cpu",
               lr=1e-3, wd=5e-5, val_sample=30, out_dir: Path | None = None,
               representation_id: str | None = None, seed_rng: bool = True):
    """Inductive M3: train enc+head on reference(=tr), select by val single-query,
    then exact-once single-query OOF over te. Writes artifacts if out_dir given."""
    if seed_rng:
        torch.manual_seed(seed); np.random.seed(seed)
    device = "cpu"
    enc = enc.to(device); head = head.to(device)
    tr_idx = np.asarray(tr_idx, dtype=np.int64); va_idx = np.asarray(va_idx, dtype=np.int64)
    te_idx = np.asarray(te_idx, dtype=np.int64)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=lr, weight_decay=wd)
    lossf = LabelSmoothCE(smoothing=0.1, classes=2)
    ref_sex = F["sex"][tr_idx]
    ref_edges = h1_edges.reference_relations(ref_sex)
    same = torch.from_numpy(ref_edges["same"]).long().to(device)
    diff = torch.from_numpy(ref_edges["diff"]).long().to(device)
    y_tr = torch.from_numpy(F["labels"][tr_idx]).long().to(device)
    va_sample = va_idx[:val_sample]
    best = -1.0; best_enc = None; best_head = None
    for ep in range(epochs):
        enc.train(); head.train(); opt.zero_grad()
        emb = embed(enc, tr_idx, F, device)
        logits = head(emb, same, diff)
        loss = lossf(logits, y_tr); loss.backward(); opt.step()
        enc.eval(); head.eval()
        with torch.no_grad():
            ref_emb = embed(enc, tr_idx, F, device)
            acc = _sq_acc(enc, head, ref_emb, ref_edges, ref_sex, va_sample, F["labels"], F, device)
        if acc > best:
            best = acc
            best_enc = {k: v.detach().clone() for k, v in enc.state_dict().items()}
            best_head = {k: v.detach().clone() for k, v in head.state_dict().items()}
        if ep in (0, epochs - 1) or ep % 5 == 4:
            print(f"[H1-M3] ep {ep+1}/{epochs} loss {loss.item():.4f} val-sq {acc}/{len(va_sample)}", flush=True)
    # ---- exact-once single-query OOF over test ----
    if best_enc is not None:
        enc.load_state_dict(best_enc); head.load_state_dict(best_head)
    enc.eval(); head.eval()
    with torch.no_grad():
        ref_emb = embed(enc, tr_idx, F, device)
        oof_ids, logits = [], []
        for qi in te_idx:
            qe = embed(enc, [qi], F, device)[0]
            q_edges = h1_edges.query_attachment(ref_sex, int(F["sex"][qi]), query_index=ref_emb.shape[0])
            out = attach_query(head, ref_emb, ref_edges, qe, q_edges)
            oof_ids.append(int(qi)); logits.append(out)
    O = torch.cat(logits, dim=0).detach().cpu()
    assert len(oof_ids) == len(set(oof_ids)) == len(te_idx), "exact-once OOF violated"
    rep_id = representation_id or h1_config.REPRESENTATION_ID
    result = {"representation_id": rep_id, "seed": seed, "oof_subjects": len(oof_ids),
              "val_best_sq_acc": best, "oof_ids": oof_ids,
              "leakage_quantification": "reserved: cohort/site-visible contrast (post-gate)"}
    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "oof_predictions.npz", subject_ids=np.asarray(oof_ids),
                 logits=O.numpy())
        (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result, O
