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
  file, scored with the `idea-quality` skill (75/100, strong; A/C-gated,
  Axis A capped by "single-metric blind spots" not being deeply novel on
  its own). Includes a Method step on the GPT-5.5 cross-judge check (see
  Provenance below).
- `scripts/60_crossjudge_gpt55.py` — the independent cross-judge script
  (GPT-5.5 via Kaon router) used to stress-test the band-ablation defect
  gradient with a judge disjoint from the project's standard Gemini 3.1
  Pro; its output is `data/crossjudge_gpt55_band_mid_vs_fulldepth.json`.
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

**Judge-sensitivity disclosure.** The band-ablation defect gradient's
headline number (mid 3.3% vs full-depth 28.3%, 8.6x, CI-decisive) is
reported throughout under the project's standard judge (Gemini 3.1 Pro).
A same-judge/same-project independence objection was raised during
review; the response was not to reword the objection away but to run an
actual third-party check (`scripts/60_crossjudge_gpt55.py`, a different
model family, GPT-5.5). The honest result: the *direction* replicates
(full-depth still higher than mid, every defect flag under GPT-5.5 falls
in the full-depth arm) but the *magnitude* does not (CIs overlap under
the less-sensitive judge). This is stated consistently in
`idea_thesis.md`, `section_trainlight_band_ablation.md`, and
`aaai_latex_submission/aaai_draft/40_limits_conclusion.tex` — the
headline claim in the abstract/intro/results is the primary-judge
finding, and its judge-sensitivity is disclosed in Limitations, not
buried or omitted.
