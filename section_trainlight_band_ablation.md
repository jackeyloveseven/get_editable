# Training-light band-LoRA: a layer-position ablation

> **⚠️ SUPERSEDED IN PART (2026-07-16).** The defect-gradient claim this
> section presents as its keystone ("mid 6.7% vs full-depth 43.3%
> clustered, CI-decisive") was **refuted by the project's own audit** the
> day after it was written — see the *Replication and confound audit*
> addendum at the end of this file. The clean-yes null (identity is
> depth-uniform) **survives and is now cross-set replicated**. The body
> below is preserved as the honest historical record; do not quote its
> defect-gradient numbers without the addendum's context. The AAAI
> submission text (`aaai_draft/30_results_traininglight.tex`,
> `40_limits_conclusion.tex`, abstract, intro) was rewritten 2026-07-16 to
> the audited state, then updated again 2026-07-17 for the closed 2x2
> confound factorial and the step6000 operating-point upgrade (addendum
> points 5-6 below).

*Status: originally the keystone result for the Wave-3 (training-light)
line, completed 2026-07-15; merged into the two-mechanism AAAI-27
submission. Scope note at the end predates that merge decision.*

## Setup

The training-free ref-KV injection mechanism (the subject of the rest of this
paper) has one structural failure mode: because it injects raw reference K/V
as extra attention slots, the model can degenerate into copying reference
pixels wholesale rather than reconstructing identity, producing collage/paste
artifacts. The training-light line asks whether a small LoRA, trained to
*route* to the right identity information rather than parrot it, avoids this
without needing full fine-tuning cost.

A band-LoRA is a standard LoRA (rank `r`, `alpha=r`) applied to
`to_q/to_k/to_v/to_out.0` in a contiguous *band* of the 30 transformer layers
of the frozen Z-Image DiT (dim 3840, 30 heads, head_dim 128, RMSNorm QK). An
earlier, informally-tuned run had settled on band **10–25** (16 layers, r=8,
~3.93M trainable params, 0.06% of the backbone) as the production recipe,
without ever testing whether that specific depth range was actually special
or just the first thing that worked. This ablation closes that gap with four
arms trained to a matched budget:

| arm | layers | r | alpha | trainable params |
|---|---|---|---|---|
| early | 0–15 | 8 | 8 | 3,932,160 |
| mid (reference) | 10–25 | 8 | 8 | 3,932,160 |
| late | 14–29 | 8 | 8 | 3,932,160 |
| full-depth | 0–29 | 4 | 4 | 3,686,400 (~94% of the 16-layer arms, r lowered to keep the budget comparable) |

All four arms: `train_manifest_mix3.jsonl` (5,936 rows / 3,339 unique
subjects), 4,000 steps, batch size 4, checkpointed every 250 steps, identical
optimizer/schedule. Evaluation: a held-out 30-subject dev set
(`repprobe30_tags.json`, excludes all train/capstone/holdout uids) × 2 seeds
= n=60 generations per arm, judged blind by Gemini 3.1 Pro (CoT, structured
JSON) for `same_character` (yes/partial/no) and `visible_defect` (yes/no).
`clean-yes` = same_character=yes AND no defect; `clean-usable` = same_character
in {yes,partial} AND no defect. All intervals are Wilson 95% CIs.

**Pre-registered decision rule** (locked before the three missing arms were
run): band 10–25 counts as *specially favored* only if its clean-yes point
estimate is highest of the four **and** its CI does not overlap the
worst-performing arm's CI. If all four CIs overlap, the finding is "band
position insensitive for identity match" — not a failure, a valid result
either way.

## Results

| arm | clean-yes | defect | clean-usable |
|---|---|---|---|
| early 0–15 (r8) | 15.0% [8,26] | 8.3% [4,18] | 45.0% [33,58] |
| mid 10–25 (r8) | 21.7% [13,34] | **3.3% [1,11]** | 55.0% [42,67] |
| late 14–29 (r8) | 21.7% [13,34] | 18.3% [11,30] | 46.7% [35,59] |
| full-depth 0–29 (r4) | 23.3% [14,35] | **28.3% [19,41]** | 43.3% [32,56] |

n=60 per arm (30 subjects × 2 seeds — see caveats below on effective sample size).

**Primary metric (clean-yes): the pre-registered rule returns "not specially
favored."** All four CIs overlap (best = full-depth at 23.3% [14,35], worst =
early at 15.0% [8,26]). Identity-match capability is roughly uniform across
depth; mid 10–25 is not uniquely good at *recognizing/reconstructing* identity.
This is worth stating plainly rather than burying: the original hypothesis
("10–25 is the identity band") does not survive this ablation on its primary
metric.

**Secondary metric (defect rate): a strong, likely real, monotonic gradient.**
3.3% → 8.3% → 18.3% → 28.3%, ordered early < mid < late < full-depth is *not*
what the raw ordering shows (mid is lowest, not an endpoint) — the actual
pattern is that mid 10–25 is the outlier-low point, and moving away from it in
either direction (earlier, later, or touching all layers) increases defect
rate. mid's CI [1,11] and full-depth's CI [19,41] do not overlap (8.6× point
estimate gap); mid and late are close to non-overlapping as well ([1,11] vs
[11,30], touching at the boundary). Under the same CI-non-overlap logic the
pre-registered rule used for clean-yes, **mid is specially favored on defect
rate**, just not on identity match.

### Qualitative confirmation (not just the judge's word for it)

Gemini's `visible_defect` field is a judge call, not ground truth, and the
project's standing rule is that no statistical claim about "collapse" or
"defect" gets written up without a human looking at actual pixels first (see
`README.md` §"实操雷区" item 14). Three full-depth defect-flagged samples were
pulled and compared against the mid-arm generation of the *same subject and
seed*:

![qualitative comparison](figures/band_ablation_defect_qualcheck/band_ablation_defect_qualcheck.png)

All three hold up under inspection: u5369_s0's full-depth arm/jacket geometry
is physically incoherent (an arm reaching into an ambiguous jacket/limb
shape) where mid's is clean; u5369_s1's full-depth hand-near-face and chest
strap routing are genuinely garbled where mid's is a coherent portrait;
u1740_s1's full-depth hand-at-collar is a blob with indeterminate finger
count where mid's hand is anatomically normal. This was a small, non-random
sample (3 of 17 full-depth defect-flagged cases, selected for having the most
specific judge notes — a real selection bias worth disclosing), but it is
enough to rule out "the whole gradient is judge noise" as the explanation.

## Interpretation

The result reframes the design rationale for the production band-LoRA recipe.
The original (undocumented, ad hoc) choice of layers 10–25 was defensible not
because that depth range is where identity lives — the ablation shows identity
signal is roughly depth-uniform — but because **that range happens to be
where LoRA training does the least collateral damage to rendering coherence**.
Touching early layers, late layers, or all layers trades a small (statistically
non-significant here) identity-match gain for a large, real increase in
visible anatomical/geometric breakage. For a paper, this is a more specific
and more defensible claim than "we ablated band position and it matters" — it
is "band position doesn't matter for what the LoRA learns, it matters for
how much damage learning it does," which is itself a useful practical finding
for anyone building similarly small identity-routing adapters on frozen DiTs.

## Caveats to carry into the write-up

- **n=60 is not 60 independent samples — resolved.** It is 30 subjects × 2
  seeds. Re-computed as a subject-clustered bootstrap (collapse each
  subject's 2 seeds to one "any-defect" indicator, resample at the subject
  level, n=30): mid 6.7% [0,17] vs full-depth 43.3% [27,60] — CIs still do
  not overlap, and mid vs late (36.7% [20,53]) doesn't overlap either. The
  finding survives the pseudo-replication fix. clean-yes stays null under
  clustering too (all four arms' CIs still overlap), consistent with the
  per-image result. Use the clustered numbers, not the per-image Wilson
  numbers, as the headline CIs in any final write-up — they are the
  honest ones.
- **The qualitative spot-check was 3 of 17 flagged cases, hand-picked for
  clarity of the judge's note text.** It supports "the defect gradient is not
  pure noise," not "100% of full-depth defect flags are real." A larger
  random (not cherry-picked) sample should be pulled before this becomes a
  headline claim.
- **Absolute numbers are modest regardless of arm**: even mid's 55%
  clean-usable / 21.7% clean-yes is not a number that wins on raw accuracy.
  The contribution here is the mechanism finding (position-dependent damage,
  not position-dependent identity capacity) and the honest ablation
  methodology, not state-of-the-art accuracy.
- **Cross-judge check (2026-07-16, `probe/60_crossjudge_gpt55.py`), added
  after a re-review flagged that both this finding and the training-free
  reversal were validated by the same judge (Gemini 3.1 Pro) and the same
  project — a fair "how independent is this really" objection.** Re-judged
  the mid-vs-full-depth pairs (n=30/arm, s0 only) with GPT-5.5 (Kaon
  router, a different model family from Gemini), identical
  `HOLDOUT_PROMPT`. Result is a **real but weak** replication, reported
  honestly rather than rounded up: **direction replicates** (full-depth
  defect 6.7% [2,21] vs mid 0.0% [0,11], both defect-flagged cases are in
  the full-depth arm, zero in mid) but **magnitude is much smaller and the
  CIs overlap** — GPT-5.5 flagged only 2/60 images as defective total,
  vs. Gemini's much higher flag rate on the identical images (Gemini
  s0-only subset on the same 30 subjects: mid 6.7% [2,21], full-depth
  26.7% [14,44]). GPT-5.5 is evidently a much less sensitive judge for
  this specific defect category (composition collage/anatomical
  garbling) at this prompt/temperature — a judge-calibration difference
  worth noting as its own small finding, not just noise. **Honest
  reading**: this is weak-to-moderate independent support for the
  *direction* of the mid-vs-full-depth defect claim, not a strong
  confirmation of the *magnitude* — do not claim more than that in the
  write-up. Full defect-flagged notes and raw judge output:
  `data/crossjudge_gpt55_band_mid_vs_fulldepth.json`.
- **Open scope question**: this section currently lives outside
  `draft_v0.md` because that draft is titled and scoped as the training-free
  KV-injection mechanism specifically. Decide before the abstract-registration
  deadline whether this becomes (a) a new major section of the same paper
  (recasting it as "two complementary identity-conditioning mechanisms for a
  frozen anime DiT: training-free and training-light"), or (b) a separate,
  shorter submission. This is a real content decision, not just a formatting
  one — it changes the abstract, the title, and the related-work framing.
  *(Resolved 2026-07-15: option (a), merged into the AAAI-27 two-mechanism
  submission.)*

## Replication and confound audit (2026-07-16) — the gradient does not survive

Run the day after the section above was written, before abstract
registration. Full data: `data/gemini_judge_ho30_band_*.json`,
`data/gemini_judge_cleanmid_*.json`, `data/ho30_band_replication_summary.json`,
`data/cleanmid_summary.json`, `data/random_defect_audit_band_ablation.json`;
analysis scripts `probe/64` (selftest reproduces every published clustered
number before touching fresh data) and `probe/65`.

1. **Cross-set replication fails.** All four checkpoints, zero retraining,
   re-run on holdout30 (disjoint 30 subjects, same judge): interior
   ranking inverts (late best 8.3%, mid second-worst 20.0%);
   mid-vs-full-depth clustered CIs overlap (36.7% [20,53] vs 46.7%
   [30,63]). The rep30 separation does not generalize.
2. **The mid arm was double-confounded.** It reused a pre-existing
   checkpoint (`run_mix1`) trained on the mix2 manifest (no `ref_caption`
   field → generic ref-slot caption; the other three arms trained on mix3
   with per-subject captions), and its published generations carried the
   `"In an anime_2d style, "` prefix that no other arm used. Same-pixel
   re-judging quantifies the prefix at ~10pp defect suppression; the
   manifest/checkpoint effect is statistically inseparable from
   run-to-run training variance (same-protocol checkpoint comparison CIs
   overlap); a residual remains that two unrun 2×2 protocol cells would
   be needed to settle.
3. **A confound-free mid erases the gradient on its home set.**
   `run_mix2/step_4000` (same recipe, mix3 manifest, verified twin — only
   `save_every` differs) generated prefix-free on rep30: defect 30.0%
   [20,43] per-image, 53.3% [37,70] clustered — **disjoint from the
   published 6.7% [0,17] on the same 30 subjects**, and cross-set stable
   (26.7% on holdout30). Every prefix-free mid measurement clusters in
   the 20–30% band; the published 3.3% is the lone outlier and the lone
   measurement with the unique protocol.
4. **Seeded random flag audit** (replacing the cherry-picked 3/17
   spot-check): full-depth 3 CONFIRMED / 4 BORDERLINE / 3 FALSE-POSITIVE
   of 10 sampled; late 1/1/3 of 5; mid 0/2/0 of 2; negative controls 0/4
   missed. Raw judge flag rates overstate severe-defect rates by
   ~1/3–2/3 per arm; judge failure modes documented (identity complaints
   absorbed into `visible_defect`, one outright hallucination).
5. **Confound now closed, not bounded (2026-07-17).** The checkpoint
   (`run_mix1`/`run_mix2`) × prompt-prefix (yes/raw) factorial that point 2
   above flagged as needing two more cells is complete
   (`data/cells3_summary.json`, rep30, n=60/cell, defect
   per-image/clustered [CI]): published `m1`+prefix 3.3%/6.7% [0,17],
   `m1`+raw 16.7%/30.0% [13,47], `m2`+prefix 30.0%/56.7% [40,73], `m2`+raw
   (=cleanmid) 30.0%/53.3% [37,70]. The prefix effect turns out to be
   checkpoint-specific: +13.3pp within `run_mix1` but exactly 0pp within
   `run_mix2` — an interaction, not the portable ~10pp correction point 2
   estimated. The only clustered-CI-disjoint marginal in the whole
   factorial is the checkpoint swap at prefix-on (+26.7pp); the same swap
   at prefix-off (+13.3pp) has overlapping CIs and stays inseparable from
   run-to-run training variance. clean-yes is flat across all four cells
   (16.7–21.7%), matching the null reported above.
6. **step6000 upgrades the flagship operating point (2026-07-17).** A
   clean-protocol run (`run_mix1`/`step_6000`, matched manifest, no
   prefix) on holdout30
   (`gemini_judge_m1s6k_ho30_band_10-25_r8_step6000.json`; step6000
   pre-registered as the identity-max checkpoint,
   `docs/design/2026-07-11_traininglight_design.md` §11.1) scores
   clean-yes 41.7% [30,54] — disjoint from the same-protocol step4000's
   18.3% [11,30] — defect 13.3% [7,24], clean-usable 76.7% [65,86].
   Against krea2 (33.3% [23,46] / 1.7% [0,9] / 58.3% [46,70]) every
   metric's CI still overlaps: tie, point-ahead, not a win. step4000
   remains reported alongside for transparency.

**Net result.** Under a uniform protocol every arm sits in an 8–30%
defect band with no stable position ordering; full-depth is numerically
worst on both sets but never CI-separated from mid. The claims that
survive: (a) clean-yes identity is depth-uniform — replicated across two
held-out sets, two protocols, and all five arm-measurements; (b) the
band-LoRA remains statistically tied with krea2 on usable-output, now at
a clean-protocol step6000 operating point (76.7% [65,86] vs 58.3%
[46,70], tie point-ahead — step4000 scores 53.3% [41,65] under the same
protocol). The defect gradient, and the "band 10-25 minimizes collateral
damage" interpretation built on it, are withdrawn.
