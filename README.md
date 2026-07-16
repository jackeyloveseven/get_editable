# cherry — paper materials

Self-contained snapshot of the paper-writing materials for the cherry
project (frozen anime T2I DiT → identity conditioning via two mechanisms:
training-free attention K/V injection, and a 0.06%-param training-light
band-LoRA). Mirrored out of the main `cherry` project repo so this folder
can be reviewed/edited without the full project checkout.

## Layout

- `draft_v0.md` — the training-free-line paper draft (Abstract through
  Conclusion, §4.3/§4.4 now have embedded qualitative figures).
- `section_trainlight_band_ablation.md` — the training-light band-LoRA
  keystone result (layer-position ablation), written as a standalone
  paper-ready section pending a scope decision on whether it merges into
  `draft_v0.md` as a new section or becomes part of a two-mechanism paper.
- `idea_thesis.md` — the locked paper thesis as a Title/Motivation/Method
  file, scored with the `idea-quality` skill (83/100, strong).
- `figures/` — every figure, each in its own folder containing the final
  image (svg/pdf/png, tiff where applicable), the generating script or
  image-generation prompt, and a provenance file (`INTENT.md`/`PROMPT.md`
  + `data_sources.json`) documenting intent, target caption, and exactly
  which judge-verdict entries or reference images the figure is built
  from — so a future agent or collaborator can regenerate or modify any
  figure without reverse-engineering it.
- `data/` — a snapshot of every `data/*.json`/`.jsonl` file any number or
  table cell in the above traces back to, with its own `README.md` mapping
  file → what it backs → where it's cited.
- `aaai_latex_submission/` — the compilable AAAI-27 LaTeX source (ports
  `draft_v0.md` + `section_trainlight_band_ablation.md` into
  `main_aaai.tex` + `aaai_draft/*.tex`, both mechanisms in one submission
  under the stronger locked-thesis framing). Compiles to exactly 7 pages
  of body + 1 page of references — AAAI-27's hard cap. Includes the
  already-compiled `main_aaai.pdf`.

## Provenance

Every figure and every cited number traces to a real experiment run in
the `cherry` project (band-position ablation, held-out validation,
baseline comparisons against IP-Adapter-SDXL and krea2-identity-edit) —
nothing here is illustrative/mock data except the AI-generated mechanism
diagram in `figures/mechanism_diagram_kv_injection/`, which is explicitly
marked as a conceptual draft, not a data figure.
