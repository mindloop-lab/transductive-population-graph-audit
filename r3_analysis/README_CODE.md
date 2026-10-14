# Public R3 statistical entry points

1. `unpack_inputs.py`: verify and unpack the committed compact test-prediction archive.
2. `major1_population_inference.py`: validate directed runs and recompute the equal-site endpoint.
3. `sym_compare.py`: validate both edge constructions and report them side by side.
4. `p1_aggregate.py`: require all three support draws and report descriptive variability.
5. `major1_aggregate.py`: recompute within-site conditional intervals; no per-site tests or Holm values.

`_common.py` defines identity, completeness, prediction and pairing guards plus shared inference.
Every official unit is derived from its manifest, including units absent from the input directory.
Historical training, packaging and one-shot editing scripts were removed from this public R3 directory;
the original release's model and core training scripts remain unchanged. See `docs/R3_RECOMPUTE.md`.
