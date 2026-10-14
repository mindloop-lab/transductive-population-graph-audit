# Major 2 — Parisot minimum checks (2026-10-13), completing the items left open

## Layer implementation, normalisation, self-loops, output activation

`parisot_gcn_adapter.py` imports the shared machinery
(`from .canonical_model import CanonicalModelConfig, CanonicalPopulationModel`, line 19) and builds it
with its own configuration (`CanonicalPopulationModel(cls.config(**overrides))`, lines 98-99). So the
evaluated Parisot row runs on the **same** `SpectralLayer` as the audited model:

* `SpectralLayer`: dense Chebyshev message passing with **explicit directed propagation**
  (`terms = [x, P@x, 2P@(P@x) - x, ...]`, canonical_model.py lines 44-50);
* `P`: `matrix[target, source] += w`, then `+ I` (**self-loops present**), then **row normalisation**
  (canonical_model.py lines 40-46);
* output: `nn.Linear(dim, 2)` + `F.cross_entropy` (line 114) and scoring via
  `softmax(logits)[:,1]` (line 141) -- no activation or scoring mismatch.

Configuration actually used: `input_dim 64, hidden_dim 16, layers 1, chebyshev_order 3, dropout 0.3,
learned_edges false, residual false, layer_concat false, lr 0.01, weight_decay 0.0005, max_epochs 120,
patience 25` (`data_contract/configs/parisot_gcn_c.json`).

## Training ran normally (from the archived unit manifests)

Read from `.../canonical_10fold_strict_single_query/Parisot-GCN-C/seed_2026/fold_00/unit_manifest.json`
and siblings:

* `status: PASS`; `strict_reload_prediction_equal: True` (the checkpoint reproduces its predictions);
* `checkpoint_epoch: 22` (best-validation epoch), `elapsed_seconds: 226`;
* `optimization_train_count: 705`, `validation_count: 78`, `test_count: 88`,
  `test_evaluation_count: 1` -- the test set is scored once, so no test-based model selection;
* digests recorded for config, graph, split and preprocessing;
* `upstream_code_executed: False`, `abide2_accessed: False`.

## Learning curves: NOT archived

`unit_manifest.json` carries no epoch-by-epoch curve, and no curve file exists for these units on this
host. This is recorded as **unavailable**, not reconstructed or inferred. The evidence that training
was healthy is the PASS status, the single test evaluation, the best-epoch record and the reload
equality check.

## Scope consequence for the manuscript (must be stated)

The Parisot comparison is against an **independent re-implementation of the published design** on this
harness (`upstream_code_executed: False`), sharing this harness's message-passing machinery. It is
therefore evidence about *evaluated implementations under this harness* -- which is exactly the scope
the manuscript should claim, and not a claim about the original authors' code or about architecture
families.
