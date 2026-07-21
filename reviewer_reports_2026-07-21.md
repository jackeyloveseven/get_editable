# Review setup

- **Input scope:** `main_aaai.pdf` (8 pages), `main_aaai.tex`, six included section files, three manuscript figures, and five tables.
- **Assessment boundary:** This review assesses the claims and evidence visible in the supplied manuscript. No supplementary material, code, raw generations, per-subject labels, judge prompts/outputs, human-validation records, preregistration record, or training manifests were supplied. Novelty relative to the literature is therefore assessed only from the manuscript's own positioning, not by an external literature search.
- **Shared manuscript claim summary:** The manuscript claims that a frozen anime text-to-image DiT can be turned into a one-shot identity conditioner by training-free reference-KV injection or a small band-LoRA; that identity is encoded in a distributed, composition-entangled manner rather than at localized layers, heads, tokens, or denoising phases; and that identity-only evaluation systematically overestimates usable performance.
- **Visible evidence base:** A development ablation and held-out reversal for reference-KV injection; a four-arm cross-backbone baseline table for the training-free method; summarized intervention tests over background, heads, tokens, and time; a four-band LoRA comparison on two subject sets; a confound audit; and a three-arm auxiliary-loss pilot.
- **Missing materials affecting confidence:** Full results for the shuffled-reference control, the cross-architecture attention-selectivity diagnostic, the claimed band-LoRA-versus-krea2 comparison, the injection-side four-band localization test, and most of the four repair probes are absent from the paper body. These are nevertheless invoked in the abstract, introduction, and conclusion.

# Reviewer 1

## Overall assessment

The manuscript is unusually candid about failed hypotheses, confounds, and metric reversals, and the held-out reversal is potentially valuable. However, the central mechanistic conclusion is not established by the current experimental design. The data support a narrower claim: several coarse interventions failed to identify a performant local control point under one model, one domain, and one evaluation stack. They do not yet demonstrate that the underlying identity representation is distributed, nor that identity and composition are structurally entangled.

## Who would be interested in the results, and why

Researchers working on personalization, diffusion-transformer interpretability, reference-conditioned generation, and evaluation of generative models would care about the negative result and the documented failure of identity-only scoring. Practitioners using attention injection may also benefit from the concrete collage failure mode and the warning that apparent identity gains can reduce usable output.

## Major strengths

1. The held-out comparison of `strip` and `lowfreq` uses the same subjects, references, and seeds and explicitly evaluates a joint identity-and-defect outcome.
2. The authors disclose that the initially attractive defect gradient failed cross-set replication and was affected by a protocol-confounded checkpoint.
3. The manuscript distinguishes development tuning from held-out evaluation and states important scope limitations.
4. The qualitative figures make the collage failure readily visible and motivate a defect-aware metric.

## Major concerns

1. **The layer experiment cannot identify a distributed representation.** The LoRA bands are early 0--15, mid 10--25, late 14--29, and full 0--29. All four arms include layers 14--15. Thus an identity-relevant mechanism localized to those shared layers is fully compatible with the reported null. In addition, the full-depth arm changes rank from 8 to 4, so band position is confounded with per-layer rank and optimization geometry even when total parameter count is similar. The phrase "identity capacity is depth-uniform" is therefore not warranted.
2. **Intervention performance is being equated with representational location.** Similar end-task accuracy after training different LoRA bands can arise from redundancy, compensatory learning, optimization limits, metric noise, or a performance floor. It does not directly locate the frozen model's identity code. A representational claim requires more direct evidence, such as non-overlapping narrow/disjoint interventions, activation or causal patching analyses, and tests that can distinguish redundancy from distribution.
3. **The premise linking localization to repairability is not logically necessary.** A localized identity feature need not provide a handle that removes a separate composition defect while preserving identity. Conversely, a failed repair may reflect an ineffective intervention or hyperparameter choice rather than structural entanglement. The late-gating observation on 5 subjects is consistent with early timesteps affecting both outcomes, but does not show that the two are represented by the same code.
4. **Several central experiments are only asserted.** The paper gives no full numerical results for the shuffled-reference control, cross-architecture selectivity signature, injection-side four-band result, or most head/token/time/background probes. The head test is reported at `n=12`, late gating at `n=5`, and other sample sizes are unspecified. These summaries cannot carry abstract- or title-level claims.
5. **The statistical analysis does not match the design.** Many experiments use 30 subjects times 2 seeds, but Wilson intervals treat 60 images as independent. The manuscript acknowledges clustering for part of the LoRA defect audit but continues to use per-image Wilson intervals elsewhere. Paired arms should be analyzed with paired/clustered procedures. Non-overlap of marginal 95% confidence intervals is not a paired significance test, and overlap is not evidence of equivalence or "parity."
6. **The primary judge is insufficiently validated.** The same VLM call supplies identity and defect fields, so the defect field is not statistically independent merely because it is a separate output key. The authors' own audit reports that only 30--70% of flagged defects correspond to real damage depending on threshold. This finding materially weakens all defect-rate estimates and should trigger a systematic blinded human validation with agreement statistics, not only a small post hoc audit.
7. **Table 5 appears arithmetically inconsistent.** With `n=60`, binary rates must be multiples of 1.67 percentage points. The reported 10.2% and 40.7% correspond naturally to fractions over 59, not 60. The true denominator, missing-output policy, counts, and confidence intervals need correction.
8. **The auxiliary-loss pilot does not show the claimed trade-off.** Both auxiliary arms lower `clean-yes` and raise defects. That is joint degradation, not trading identity for coherence. With one training seed and one weight per loss, failure also cannot distinguish structural entanglement from poor loss scaling, optimization, or decoding effects.

## Technical failings that need to be addressed before the case is established

1. Replace the overlapping-band test with disjoint or sufficiently narrow bands and keep rank, steps, data, prompts, and initialization protocol-matched; report all sets and raw counts.
2. Separate the empirical statement "we did not find a local handle" from the mechanistic hypothesis "the code is distributed and composition-entangled." Test plausible alternatives such as redundancy, compensation, and judge insensitivity.
3. Report the missing shuffled-reference, cross-architecture, injection-band, head, token, timestep, and background experiments in full, including denominators, controls, uncertainty, and selection rules.
4. Reanalyze paired repeated-measure data at the subject level, using clustered bootstrap, a hierarchical model, or paired tests as appropriate. Use an equivalence/non-inferiority design before claiming parity.
5. Validate identity and defect labels against blinded human annotations on a preregistered sample; report agreement, sensitivity/specificity, uncertainty, and how ambiguous cases are handled.
6. Correct Table 5's denominator and avoid mechanistic inference from a one-seed, one-weight pilot.

## Assessment against Nature-style criteria

- **Originality:** Potentially original as a negative mechanistic/evaluation study, but the central representational inference is not yet identified by the experiments.
- **Scientific importance:** The evaluation failure is useful; the broader mechanistic importance remains conditional on stronger causal evidence.
- **Interdisciplinary readership:** Primarily of interest to diffusion and generative-model researchers in the current form.
- **Technical soundness:** Not yet sufficient for the title-level claim, chiefly because of overlapping interventions, dependent observations, judge validity, and missing central results.
- **Readability for nonspecialists:** The basic failure mode is understandable, but the manuscript moves too quickly from intervention nulls to representational conclusions.

## Recommendation posture

**Currently not established from the provided evidence; major revision is required.**

# Reviewer 2

## Overall assessment

The most convincing contribution is not the proposed conditioner but the empirical lesson that identity-only evaluation can select a visibly worse recipe. The manuscript's willingness to report reversals and confounds is a real strength. By contrast, the novelty and significance case is diluted by three different stories: a low-cost conditioning method, a mechanistic claim about distributed identity, and an evaluation protocol. The first is not competitive on usable quality, the second is overclaimed, and the third is promising but supported by only a narrow empirical base.

## Who would be interested in the results, and why

The work should interest researchers evaluating personalized generation and those studying when frozen-model activation patching fails. The failure-analysis angle may be more broadly useful than the specific anime-model recipe because it exposes how a plausible identity metric can reward composition collapse.

## Major strengths

1. The manuscript does not hide that the tuned `lowfreq` method fails badly on held-out usable output.
2. The explicit distinction among identity match, visible defect, `clean-yes`, and `clean-usable` is directionally valuable.
3. Reporting a null hypothesis and then auditing a confound is more informative than presenting only a selected successful band.
4. The limitations section already acknowledges the single-domain, single-main-model scope and the correlational nature of an `n=2` architecture comparison.

## Major concerns

1. **Contribution 1 is incompletely supported.** The training-free comparison shows overlapping confidence intervals with IP-Adapter, but "statistically indistinguishable" is not equivalence. More importantly, the manuscript never presents the claimed band-LoRA-versus-krea2 comparison. The conclusion's phrase "the band-LoRA with krea2 at 0.06% of its parameters" is also ambiguous: 0.06% refers to the adapted Z-Image backbone, not to krea2, and trainable adapter parameters are not directly comparable to a baseline model's total parameters.
2. **The baseline story is not controlled.** IP-Adapter-SDXL and krea2 differ in backbone, scale, training objective, domain, and conditioning interface. The manuscript admits this, but then uses "same-class" and "parity" to imply a cleaner comparison than exists. These baselines are useful external reference points, not evidence isolating the value of the proposed mechanism.
3. **The paper claims evidence that it does not show.** The shuffled-reference result and `n=2` selectivity signature appear in the abstract and conclusion but not in Results. A central claim cannot be supported only by an assertion in framing and limitations.
4. **The novelty positioning is too narrow and partly rhetorical.** "First ... pure anime T2I DiT" may be true, but the scientific novelty should not depend on the application domain alone. The relationship to prior attention injection, RoPE manipulation, and in-context conditioning needs a precise statement of what is technically new. Phrases such as "folk knowledge" and an unreported literature search do not substitute for this distinction.
5. **The main method result is negative for deployment.** The novel frequency-gated/wide-band recipe has 82% defect and 15% usable output. The recommended `strip` recipe appears closer to an existing position-free injection baseline. The paper should be framed primarily as a failure-analysis and evaluation study unless a technically novel, usable method is demonstrated.
6. **"Systematically optimistic" is broader than the evidence.** One held-out recipe reversal strongly supports a warning for this setup. The second reversal concerns a defect-gradient audit rather than a clean replication of identity-only ranking failure. Two episodes in one model/data family do not establish systematic bias across the method class.
7. **Qualitative claims are under-sampled.** Figure 3 uses two selected subjects to infer three distinct failure signatures. Without a selection protocol or coded sample-level analysis, statements about characteristic failure modes and generic priors remain illustrative rather than established.

## Technical failings that need to be addressed before the case is established

1. Choose a primary paper claim and align the title, abstract, figures, and experiments around it. The strongest current option is defect-aware evaluation plus a bounded non-localization result.
2. Add complete results for every abstract-level claim or remove those claims from the abstract, contributions, and conclusion.
3. Replace "parity," "tied," and "statistically indistinguishable" with correctly powered equivalence/non-inferiority analyses, or use the narrower wording "no difference was detected at this sample size."
4. Clarify which method constitutes the novel contribution and why it differs from prior position stripping, coordinate rebinding, and RoPE manipulation.
5. Add at least one architecture/domain-matched baseline or substantially narrow all comparative claims.
6. Quantify failure-mode prevalence over a systematically sampled set rather than inferring categories from two examples.

## Assessment against Nature-style criteria

- **Originality:** The self-audited negative-result narrative is distinctive; technical novelty over prior injection/RoPE work remains insufficiently isolated.
- **Scientific importance:** The metric warning is potentially important, but broad importance is not yet demonstrated across methods or models.
- **Interdisciplinary readership:** The current evidence is field-specific; a wider evaluation study could broaden the audience.
- **Technical soundness:** Comparative and mechanistic claims exceed the controls shown.
- **Readability for nonspecialists:** The reader can see that identity scores can be misleading, but the manuscript's three competing contribution narratives obscure the central message.

## Recommendation posture

**Promising as a carefully scoped failure-analysis paper, but the present novelty and significance claims require major revision.**

# Reviewer 3

## Overall assessment

The manuscript has an energetic and unusually transparent voice, but it often reads like a compressed research log rather than a finished scientific argument. Strong causal language, informal metaphors, undefined experimental labels, and dense audit details make it difficult to distinguish observation from interpretation. The paper would become clearer and more credible if each paragraph explicitly separated result, uncertainty, and inference.

## Who would be interested in the results, and why

Nonspecialists in generative AI could understand the central warning that a model may look correct by identity score while producing unusable compositions. Specialists will value the disclosed reversals, provided that the protocol and missing data are made reproducible.

## Major strengths

1. The prose frequently states limitations and avoids claiming that the proposed recipe beats all baselines.
2. Figures 2 and 3 communicate the qualitative failure immediately.
3. The paper has a recognizable narrative arc: activate identity conditioning, discover a measurement failure, probe the failure, and revise the interpretation.

## Major concerns

1. **The title is more absolute than the limitations.** "Subject Identity Is a Distributed, Composition-Entangled Code" states an ontological conclusion, whereas the limitations say only that several natural handles failed in one anime model. A scoped title such as "Coarse Intervention Tests Suggest Non-localized, Composition-Coupled Identity Conditioning in a Frozen Anime T2I Transformer" would match the evidence more closely.
2. **The abstract is overloaded.** It contains method, baseline, control, localization, five repairs, cross-architecture diagnostic, and two metric reversals in one paragraph. Several items have no corresponding Results presentation. The reader cannot tell which result is primary.
3. **Observation and causal interpretation are repeatedly fused.** Examples include "because identity and composition co-form," "the outcome ... predicts," "injecting ... is what destabilizes," and "five ... repairs each confirm." These should be recast as bounded consistency statements unless direct causal tests are added.
4. **Terminology is inconsistent or insufficiently defined.** `yes`, `no`, `partial`, raw identity, strict identity, `clean-yes`, and `clean-usable` are used in different comparisons. "Defect," "composition collapse," "collage," "ghosting," and "anatomical distortion" also need a stable taxonomy. "Same-class" and "independent mechanisms" require operational definitions.
5. **Reproducibility details are too compressed.** The paper does not provide dataset construction, subject inclusion, prompt templates, judge prompt/version/settings, generation settings, LoRA training data and optimizer, missing-output policy, or the exact protocol for blinding and preregistration.
6. **Some statements are internally contradictory.** A separate defect field returned by the same judge call is described as "independent"; a set is called both "held-out" and "dev"; all five repairs are said to trade identity for coherence even though the auxiliary-loss arms worsen both; and Table 5's percentages do not match its stated denominator.

## Language and presentation issues

1. **Abstract:** "identity preservation is normally bought with" is colloquial. Use "typically requires."
2. **Abstract:** "We turn such a model ... two cheap ways" is missing a preposition and sounds informal. Use "We adapt such a model in two low-cost ways."
3. **Abstract:** "no localized handle to grab" is conversational. Use "we find no evidence of localization under the tested interventions."
4. **Introduction, line 20:** "Both mechanisms reach that parity" has no clear antecedent because parity has not yet been introduced in the preceding paragraph.
5. **Introduction, lines 23--24:** "Within every handle we can test, it cannot" has an unclear pronoun and awkward logic. State exactly what was not localized.
6. **Introduction:** "the one defect that dogs it," "the paper's spine," and "would have shipped" are vivid but too informal for the main evidentiary chain.
7. **Contribution 2:** "a locality we do not find" is unidiomatic. Use "we find no localization across the tested layer bands..."
8. **Contribution 3:** The sentence beginning "A peaked-versus-diffuse..." becomes grammatically incomplete after the semicolon: the second coordinated noun phrase ("a paired ... protocol") has no finite main verb.
9. **Related Work:** "close to folk knowledge" should be replaced by a sourced, neutral statement.
10. **Method:** "the real generation's forward pass" should be "the target-generation pass."
11. **Training-free Results, lines 19--23:** "Every row is measured on the same 10 subjects band width and..." is grammatically malformed. A possible revision is: "All rows use the same 10 subjects on which band width and `lowfreq_k` were tuned."
12. **Evaluation protocol:** "a second, independent ... field to the same judge call" overstates independence. Use "a separate output field from the same judge call."
13. **Held-out protocol:** "single blind judgment pass" should be "single-blind evaluation pass," followed by a definition of what was blinded.
14. **Results:** "CI-decisive," "strongest ... bars," "outright miss," and "three audits then killed it" are nonstandard or colloquial. Prefer "non-overlapping confidence intervals," "reference baselines," "identity failure," and "three audits did not support this pattern."
15. **Statistics:** "statistically indistinguishable" and "statistically tied" should not be used for a nonsignificant difference. Use "no difference was detected" unless an equivalence test was specified.
16. **Training-light Results:** "Identity capacity is depth-uniform" is stronger than the result. Use "we detected no clean-yes difference among the tested, overlapping band interventions."
17. **Repair interpretation:** Replace "because identity and composition co-form" with "consistent with early-step injection affecting both identity and composition."
18. **Limitations:** The isolated mention of "GPT-5.5" is undefined and introduces an unreported evaluation instrument. Explain the model, protocol, and result in Methods/Results or remove it.
19. **Conclusion:** "the band-LoRA with krea2 at 0.06% of its parameters" has an ambiguous possessive and makes unlike parameter counts sound directly comparable.
20. **Throughout:** Use one notation for ranges (`10--25` in LaTeX prose), hyphenate compound modifiers consistently (`training-free`, `held-out`, `single-blind`), and report numerator/denominator alongside percentages for small samples.

## Technical failings that need to be addressed before the case is established

1. Rebuild the narrative around one primary claim and move protocol-debugging detail that is not essential to that claim into an appendix.
2. Define all outcomes once in a compact table and use the same labels throughout.
3. Add a reproducibility subsection covering data, prompts, generation, training, judge settings, blinding, missing outputs, and preregistration provenance.
4. Replace causal/confirmatory verbs with evidence-calibrated language throughout.
5. Ensure every abstract and conclusion claim points to a visible result, table, or figure.

## Assessment against Nature-style criteria

- **Originality:** The transparent negative-result narrative is engaging, but overstatement obscures the precise novelty.
- **Scientific importance:** The evaluation message could matter beyond this exact model if presented as a testable, replicated warning rather than a universal conclusion.
- **Interdisciplinary readership:** The visual failure mode is accessible; the transformer-specific mechanism discussion is not yet explained for nonspecialists.
- **Technical soundness:** Readability problems currently mask several unsupported inferences and missing methods.
- **Readability for nonspecialists:** Substantial revision is needed to reduce jargon, define metrics, and distinguish facts from interpretations.

## Recommendation posture

**Major revision focused on claim calibration, reproducibility, and language precision.**

# Cross-review synthesis

## Consensus strengths

All three reports view the held-out `lowfreq` reversal and the authors' willingness to disclose nulls, confounds, and judge failures as the paper's strongest features. The qualitative collage examples are clear, and the attempt to combine identity and defect assessment addresses a real evaluation risk.

## Consensus technical risks

1. The distributed-code conclusion is not identified by overlapping layer bands, and the full-depth rank change adds a confound.
2. Failure of a few coarse repairs does not establish structural identity--composition entanglement.
3. Multiple abstract-level claims have no complete Results evidence.
4. Repeated measurements are treated as independent in several tables, paired designs are not analyzed as paired, and nonsignificance is misread as equivalence.
5. The single VLM judge is central to the conclusions but the manuscript's own audit shows substantial label unreliability.
6. Table 5's percentages conflict with `n=60`, and its one-seed pilot cannot support a structural conclusion.

## Where emphasis differs across reviewers

- Reviewer 1 gives greatest weight to causal identification, band design, statistical dependence, and judge validity.
- Reviewer 2 gives greatest weight to novelty, baseline comparability, and whether the broad significance follows from this narrow setting.
- Reviewer 3 gives greatest weight to argument structure, reproducibility, terminology, and language calibration.

## Broad-interest / significance readout

The paper has a potentially strong, field-relevant message: identity-only scoring can favor visibly unusable reference-conditioned outputs, and negative localization results deserve reporting. The current evidence does not yet support the broader title-level statement that one-shot subject identity *is* a distributed, composition-entangled code. With a narrower claim and complete reporting, the work could become a credible failure-analysis/evaluation paper; establishing the mechanistic claim would require redesigned causal experiments.

## Claim-support matrix

| Manuscript claim | Evidence currently shown | Assessment |
|---|---|---|
| `strip` activates reference-conditioned identity at zero training cost | Four-arm held-out table and examples | **Partially supported**, but identity equivalence to IP-Adapter is not established by overlapping CIs and the backbone/domain are unmatched. |
| band-LoRA is competitive with krea2 | No corresponding comparison in Results | **Unsupported in the supplied manuscript.** |
| conditioning is genuinely reference-driven | Shuffled-reference control is asserted but not reported | **Unsupported as presented.** |
| identity is distributed across depth | Four overlapping LoRA bands; injection-side four-band data absent | **Not established**; a localized shared-layer or compensatory explanation remains possible. |
| no head/token/timestep handle separates identity and defect | Brief prose summaries, some with `n=12` or `n=5` | **Suggestive but underpowered and incompletely reported.** |
| identity and composition are structurally entangled | Five attempted repairs plus late gating and auxiliary loss | **Consistent with, but not diagnostic of, entanglement**; several alternatives remain. |
| cross-architecture selectivity distinguishes structured identity from mean color | `n=2` claim appears only in framing/limitations | **Not assessable from the shown evidence.** |
| identity-only evaluation is systematically optimistic | One strong held-out reversal plus one audit narrative | **Supported as a concrete risk in this setup; "systematically" and method-class generalization are overbroad.** |

## Most important issues to resolve before a strong case is established

1. Decide whether the paper's primary contribution is mechanistic localization or defect-aware evaluation; the present experiments support the latter more strongly.
2. Redesign the layer test with disjoint, protocol-matched interventions and use direct causal/representation probes if retaining the distributed-code claim.
3. Put every shuffled-control, repair, diagnostic, and baseline result invoked by the abstract into Results with full counts and uncertainty.
4. Revalidate the VLM metrics against blinded human labels and reanalyze all 30-subject-times-2-seed experiments with paired, subject-clustered methods.
5. Correct Table 5 and distinguish degradation from a genuine identity--coherence trade-off.
6. Scope the title, abstract, and conclusion to one anime model and the specific handles tested; replace confirmatory causal language with calibrated statements.

## Overall recommendation posture

The shared posture is **Weak Reject / Major Revision in the current form**, not because the negative results lack value, but because the strongest mechanistic and comparative claims are not yet supported by the experiments presented. A substantially revised paper centered on the held-out metric reversal and rigorously bounded non-localization findings would have a clearer and more defensible contribution.

# Risk / unsupported claims

1. **Missing evidence:** shuffled-reference control, band-LoRA/krea2 parity, cross-architecture selectivity diagnostic, injection-side four-band null, and full repair-probe results.
2. **Over-inference from nulls:** "distributed," "depth-uniform," "statistically tied," and "parity."
3. **Over-inference from interventions:** "composition-entangled," "co-form," "what a localized one does not predict," and "each confirm."
4. **Statistical risk:** pseudo-replication, unpaired analysis of paired arms, CI-overlap heuristics, no equivalence margin, unclear multiple-testing handling, and a probable denominator error in Table 5.
5. **Measurement risk:** one shared VLM judge, no full human validation, post hoc audit revealing low positive predictive value, and insufficiently specified prompts/settings.
6. **Comparative risk:** architecture/domain/training mismatches across baselines and no visible architecture-matched trained control.
7. **Scope risk:** one main anime model and one data family cannot support an unqualified claim about one-shot subject identity in frozen T2I transformers generally.
