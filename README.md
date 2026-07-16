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
- `scripts/` — every analysis/generation script any number in this folder
  depends on, so a claim can be reproduced without the full project
  checkout: `38_gemini_judge.py` (the shared judge harness, all `--set`
  variants), `60_crossjudge_gpt55.py` (cross-judge independence check),
  `61_defect_only_diagnostic.py` + `62_widerband_decompose.py` (the
  lowfreq-collapse mechanism decomposition, see Provenance below),
  `63_spectral_flatness_predictor.py` (exploratory cheap defect proxy).
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

**lowfreq's 82% defect collapse: mechanism decomposition (2026-07-16).**
`draft_v0.md` originally called this "unexplained." A single-variable
decomposition (`scripts/62_widerband_decompose.py`) tested the most
likely fix — strip position-encoding out of the widened band entirely —
and it does not work: position-free content injected into the same wide
band (10-25) is just as defective (75%) as the frequency-gated version
(71%), statistically indistinguishable on the same dev set. The driver is
injecting reference content into layers 10-20 at all, not position
precision. This is now reported as a clean negative result (no cheap fix
found) plus a real mechanistic finding (ruled out one specific
hypothesis, narrowed the explanation), not as an open problem. A
secondary, narrower observation (`scripts/61_defect_only_diagnostic.py`)
shows position precision *does* matter at the original narrow band
(21-25): full-precision repositioning is already bad there (70% defect)
while low-frequency-gated or absent position info stays clean (~5-20%) —
a different instrument/set from the widerband result, reported
separately, not stitched into one combined story. An exploratory,
judge-free defect proxy (spectral flatness, `scripts/
63_spectral_flatness_predictor.py`) was also checked against both
datasets; it holds moderately on the larger set (r=-0.49) and weakly on
the smaller one (r=-0.21) — reported as a preliminary signal, not a
validated tool.
