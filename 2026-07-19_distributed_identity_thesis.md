# Reframe thesis: identity in a frozen T2I DiT is a *distributed* representation
*(2026-07-19 — the strongest positive, AAAI-shaped framing extractable from the existing evidence, no new GPU experiment required. Honest viability assessment at the end.)*

> **ADOPTED 2026-07-20.** This reframe is now the paper's framing: the LaTeX
> submission (`aaai_latex_submission/main_aaai.tex`), `draft_v0.md`, and
> `idea_thesis.md` all lead with the distributed, composition-entangled thesis.
> The head-null is now quantified (n=12 head-gate) and a fifth confirmatory
> repair was added (n=60 identity-supervised training objective). Per strategic
> review the venue call is: AAAI-27 as a low-cost submission (odds ~15–20% best-packaged), TMLR as the paper's home.

## The one-sentence contribution
In a frozen, pure-T2I diffusion transformer, the identity information that
attention-level reference injection activates is **not localized** — not to a
layer band, not to a set of attention heads — and this distributed structure
**causally explains** why the identity gain and the composition (collage/PiP)
defect it induces cannot be separated by any targeted, injection-side
intervention.

## Why this is a positive contribution, not just our pile of negatives
The field's personalization/injection literature *assumes* localization and
exploits it: FreeCus injects into "vital layers"; HeadRouter / head-level
methods route by specialized heads. Our evidence says that assumption does not
hold for identity in this model, and we show it **causally**, with matched
controls — which reframes five negative experiments into one coherent finding.

**Evidence (all already collected, all with controls):**
1. **Layer distribution.** Band-localization ablation (early/mid/late/full-depth,
   n=60, Wilson CIs): clean-yes identity is statistically uniform across depth —
   identity is not carried by a "vital" band. (Cross-set replication caveat is
   already documented; report as a null, which is what it is.)
2. **Head distribution (new, 2026-07-19).** Restricting ref-KV injection to the
   top-*k* heads by reference affinity vs. a *random* *k* vs. the *bottom* *k*
   (matched count): indistinguishable. Identity is not carried by a selectable
   head subset — the apparent "fewer heads = cleaner" effect is the
   injection-strength count, not a head selection.
3. **The mechanism consequence.** Three separate targeted defect-fixes —
   background isolation (content-leakage hypothesis), head-localized injection,
   and high-norm outlier-token suppression — each with a matched control, all
   failed (§4.6). Under a *distributed* identity representation this is the
   *predicted* outcome: there is no local handle to grab that separates the
   identity signal from the composition damage it does.

## What the paper claims (analysis-first framing)
- **Question**: where and how does the identity signal live in a frozen T2I DiT,
  and can training-free injection exploit locality to get identity without the
  collage defect?
- **Method-as-instrument**: our training-free ref-KV injection (with the
  RoPE-collision handling and frequency-gating mechanics) is the *probe* that
  lets us ask the causal question at all.
- **Answer**: identity is distributed across depth and heads; the
  identity↔composition tradeoff is therefore a *structural* property of
  attention-level injection into this model, not a tuning failure — and we
  demonstrate this by ruling out the three natural localization-based fixes.
- **Secondary results kept**: 0.06%-param training-free parity with a same-class
  trained adapter (IP-Adapter) on identity; the shuffled-ref attribution showing
  our conditioning is genuinely reference-driven while a purpose-trained baseline
  (krea2) is partly a prompt floor.

## Honest viability assessment (do not skip)
- **Strength**: it is a *positive* claim (a causal characterization), it is
  counter-intuitive, it contradicts named prior assumptions, and it unifies the
  negatives into a mechanism rather than a list of failures. It needs **no new
  experiment** — only reframing and tightening.
- **Weakness for AAAI main track**: the core is still *null-based* ("identity is
  NOT localized"). Nulls-overturning-assumptions get into AAAI less often than
  positive SOTA/novelty; reviewers may want the localization claim tested on more
  than one model (single-model evidence). The head-null is currently
  *qualitative* (matched-control eyeball), not a quantified benchmark — for an
  AAAI submission this should be upgraded to the defect-aware judge on the
  holdout set (one bounded run, not a drill).
- **Verdict**: this is the **best AAAI-shaped shot** the evidence supports, and
  it is genuinely stronger than "audit trail is the contribution." It is still a
  **reach** for the main track and a **strong fit** for TMLR. The one cheap thing
  that would materially raise its AAAI odds is quantifying the head-null (top-*k*
  vs random-*k*, defect-aware judge, holdout n) so the central claim is not
  eyeball-only.

## If pursued: the single bounded experiment worth running
Quantify the head matched-count null: top-7 vs random-7 vs bottom-7 vs full
(baseline) on the holdout set with the defect-aware judge — clean-yes and defect
rate with CIs. Pre-registered prediction under the thesis: **no clean-yes
difference among the head-selection arms** (identity distributed), while defect
rate tracks injection *count*. This converts the headline claim from qualitative
to quantitative. It is one bounded run + one judge pass — not open-ended drilling.
