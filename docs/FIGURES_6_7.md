# Reproducing manuscript Figures 6 and 7

The generator is `figure_sources/make_r3_figures.py`. It reads only the frozen
summaries and split manifest in `figure_sources/inputs/fig6_fig7/`; provenance and
input SHA256 values are recorded in `PROVENANCE.json`. These inputs refer to the
reviewed R3 analysis commit, whose tree matches the original R3 merge into main.
The compact prediction bundle hash is unchanged. This is plotting, not refitting
or new statistical inference.

```bash
python -m pip install -r figure_sources/requirements_fig6_fig7.txt
python figure_sources/make_r3_figures.py --output-dir reproduced/figures
```

Commands may run from another directory when the script is addressed by its path.
Without `--output-dir`, outputs go to the repository's `figures/` directory.
PDF, SVG and PNG are exported together. The repository tracks the editable SVGs;
the generated PDF/PNG files need not be committed. Matplotlib is pinned at 3.10.8.
SVG text is editable, SVG ids are stable, and SVG/PDF timestamps are omitted.
Repeated SVG/PNG exports were byte-identical in the validation environment;
PDF byte identity across processes or environments is not guaranteed.

| Manuscript figure | Generated basename | Design |
|---|---|---|
| Figure 6 | `F1_onoff_protocol_schematic` | Subject roles; ON = C union S and OFF = C union R; matched budget; fixed graph and information-access conditions |
| Figure 7 | `F2_site_delta_auc_forest` | Directed and symmetrised site effects; short legend outside the plotting axes |

The F1/F2 names are retained for compatibility with the manuscript figure includes.
Figure 7 has 18 sites and 88 eligible site-fold cells: SDSU contributes three folds;
CALTECH and CMU have no eligible cells. The script checks the expected site set,
fold count, seed averages, population average, and legend/axes separation.

Site intervals condition on fixed fitted models and seeds; the population interval
resamples sites. They are different uncertainty summaries. Formal interpretation
and interval definitions belong in the manuscript caption; plotting does not
restore the withdrawn per-site p values or Holm adjustments.

This update changes presentation and reproduction tooling only. Prediction inputs,
training, numerical effects, release tags, DOI metadata and published preprints are
unchanged. The `figure-render` CI job exercises this generator on each push/PR.
