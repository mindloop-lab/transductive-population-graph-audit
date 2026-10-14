# Graph semantics along the ACTUAL load path (2026-10-14, corrected)

Supersedes the earlier version of this file, which described the wrong module. The paired experiment
does **not** use `baselines/canonical_model.py`; it uses the trained model path below.

## What the driver actually loads

`major1_arms.py` loads its dependencies by file path (`load_module`), so the runtime sources were read
and hashed directly:

| module | path | bytes | sha256 |
|---|---|---|---|
| helpers (`mm`) | `CR-MAGP-AHGNN-transductive-model/scripts/mechanism_controls.py` | 11061 | `05ead33e4c700905ad416fd309ce041508c14383ea7eb999c8669b1b53440984` |
| model (`het`) | `.../src/cr_magp_ahgnn/models/heterogeneous_magp.py` | 8478 | `d0c24f80556b71e892ebccc37c851a267a55a74e0b46ece61b4dca888ad8384e` |
| Parisot comparator (separate path) | `released_repo/src/cr_magp_ahgnn/baselines/canonical_model.py` | 5964 | `543010419960df3661a5cb2eae707e92e086945c9fb8a0d998b79b4d9e907ddc` |

The model file is **byte-identical** between the training tree and `released_repo`, so the released
copy describes the trained model.

## The paired experiment's population head

`PopulationHead.forward(x, same_idx, diff_idx)` — three layers, each a pair of PyG `TransformerConv`
modules (channel `same` and channel `diff`) combined with softmax-normalised per-layer weights,
BatchNorm, leaky-ReLU, then a linear classifier. The driver calls it as
`population_gnn(emb, e_t, torch.zeros((2, 0)))`, i.e. **`diff_idx` is empty**.

Consequences, stated from the installed PyG implementation (verified by reading the class):

* `TransformerConv` has **`add_self_loops = False`**: no graph self-loops are added.
* It has **`root_weight = True`**: a node's own feature enters through a learnable root/skip transform.
  That is a *different* operation from adding a self-loop or from multiplying by a fixed propagation
  matrix.
* There is **no row normalisation** and no fixed Chebyshev propagation in this path; those descriptors
  belong to `canonical_model.py::SpectralLayer`, used by the Parisot comparator. The earlier claim that
  "all comparison rows share the same propagation code" is **retracted**.

## Edge list, direction, node set

* The driver builds pairs `(nodes[i], nodes[j])` with `i < j`, passed as PyG `edge_index = [source,
  target]`, so messages flow from the lower to the higher index. The `i<j` construction is a **code
  convention, not a graph property**; the reversed and symmetrised variants change exactly this.
* The node set is **every subject except the fixed validation set V** (documented in the driver).
  It does **not** depend on the support draw.

## Correction to the support-sampling interpretation (retraction)

The earlier explanation — that the same-site arm changes across support draws because the graph
contains the varying other-site support `R` — is **wrong and retracted**: the graph is fixed across
draws. What changes is the **supervised target set**: `C = O \ R` and the OFF arm is supervised on
`C ∪ R`, so for the ON arm the supervised set is `C ∪ S`, whose **composition** (and hence whose
labels) changes with the draw. Cross-replicate differences in the ON arm therefore reflect a changing
supervised target set, not a changing graph, and the ON arm is **not** a valid determinism control.

## Figure caption correction

The protocol figure's box list omits `C` and its note said edges are "same-site pairs among
S ∪ R ∪ T". Correct description: edges are same-site pairs among **all nodes except V**, i.e.
`S ∪ R ∪ T ∪ C`.

## Code check requested by the reviewer: does the original 0.94 result share this edge construction?

No. The two paths build edges in **different code**:

* the paired experiment's population edges are built by `major1_arms.py` as `(nodes[i], nodes[j]), i < j`
  — one directed write per pair;
* the canonical / mechanism-control path builds its graph from the **stored adjacency matrices**
  (`mechanism_controls.py` line 69: `a = np.abs(adjs[i]); r, c = np.where(a > 0.5)`) through a
  **networkx undirected graph** (line 86: `G.add_edges_from(zip(edges[0], edges[1]))`), and
  `paperm/h1_data.py` produces `edge_index` from stored adjacency as well.

Therefore the paired experiment's orientation sensitivity **cannot be extrapolated** to the 0.94
cohort result, and the 0.94 result's insensitivity **cannot be assumed** either: they are separate
constructions and each needs its own check. Stated as a scope boundary, not as a result.
