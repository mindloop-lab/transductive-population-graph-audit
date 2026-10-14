> Historical snapshot retained for traceability. It is not the current release contract.
> Per-site tail-ratio p values/Holm results and the former graph-change explanation are withdrawn.
> Current input checks and executable results are documented in docs/R3_RECOMPUTE.md.

# PDF rebuild verification (2026-10-12)

Purpose: test, rather than assert, the claim in `R3_MANUSCRIPT_TEXT_DELTAS.md` item 0 that the
shipped Oct-11 PDF is a stale build of its own source.

Method: copy the delivered `source/` (tex, cls, sty, bst, bib, figures) into `R3_pdf_rebuild/` and run
`latexmk -pdf -interaction=nonstopmode -halt-on-error paperM_cmig.tex`. Nothing in the delivered pack
was modified; the rebuild is a separate directory.

## Result

| artefact | bytes | pages | sha256 |
|---|---|---|---|
| shipped `paperM_cmig.pdf` | 1835223 | 25 | `84aaca45e8ebb4f18c5c3489e253a0dc7270459529e10dcd464b388b5146cc6b` |
| rebuilt from delivered source | 1832151 | 25 | `5fd72d358cac57232255b0f663dfb6bb9f43bbc3c2a2abd5087a202e3d975a18` |

`latexmk` exit status 0; no LaTeX error. The two artefacts are different files (different bytes and
digests), so they cannot both be renderings of the same source revision.

## Text probes (rendered text, whitespace-normalised)

* `does not denote a newly proposed model` -- **present in shipped**, **absent in rebuilt**.
* `We developed a site-aware population graph neural network` -- present in rebuilt.
* `Here, site-aware refers to the use of site metadata` -- present in rebuilt.

## Consequence

* The divergence is now **experimentally established**, not inferred: the shipped PDF predates the
  source it was shipped with.
* R3 must publish a PDF built from the frozen R3 source together with the source digests, and must
  not present the Oct-11 PDF as the rendering of the Oct-11 source.
* No R3 text change is needed for the two items previously thought missing: the source already
  carries the required opening paragraph and has already removed the withdrawn sentence from the
  rendered text.
