# Title
Identity-match metrics hide composition-collapse risk: evidence from two unrelated conditioning mechanisms on a frozen anime T2I DiT

## Motivation
Reference-image identity conditioning (feed a reference portrait, get the
same character in a new scene/pose) is normally bought either with a
purpose-trained adapter on real-face data (IP-Adapter, InstantID — built
and evaluated on ArcFace/FaceNet-style face recognizers that do not detect
faces on anime images at all, so their own methodology cannot transfer to
the anime domain), or with a base model that was already pretrained for
fill/edit tasks (FreeGraftor, UNO, OminiControl — all built on FLUX.1-dev
or similar fill-capable backbones, not a virgin pure-T2I model). Whether a
frozen, never-fill/edit-trained, anime-domain T2I diffusion transformer
already has latent reference-conditioning capability that a training-free
or ultra-light-parameter intervention can unlock has not been shown — but
that domain-combination gap is not this work's central claim; it is the
setting in which a more portable methodological finding surfaced twice,
independently, by mechanisms that share no components.

Standard practice reports identity-match rate (does the generated
character look like the reference?) as the headline metric, treating
visible composition defects (collage, tiling, garbled anatomy) as a
secondary footnote if reported at all. This work finds identity-match
alone is not just incomplete but actively misleading, in two mechanistically
unrelated settings: (1) a dev-set-tuned training-free recipe with a
*higher* raw identity-match rate than the production baseline turns out to
have an 82% held-out composition-defect rate (vs. 20% for the baseline) —
identity-match alone would have recommended the worse recipe; (2) a
pre-registered layer-position ablation on an unrelated training-light
LoRA mechanism finds identity-match statistically flat across all tested
positions (CI-overlapping, including under subject-clustering), while the
*same* positions differ by 8.6x on training-induced defect rate
(CI-decisive) — a real, large effect entirely invisible to the identity
metric a naive ablation would have reported alone. Two independent
mechanisms, two independent evaluation protocols, the same blind spot: a
design choice's practical consequence can live almost entirely on a
defect axis that identity-only scoring never touches. That is the gap
this work is actually positioned to close — not "nobody tried this on
anime," but "identity-match-only evaluation is a systematically
optimistic proxy for reference-conditioning quality, and this isn't a
one-off artifact of a single method."

A fair objection: both replications above ran under the same judge
(Gemini 3.1 Pro) and the same project, so "independent" is doing less
work than it sounds like. A third check (GPT-5.5, a different model
family, same prompt, n=30/arm on the training-light mid-vs-full-depth
pairs) gives a real but modest answer, reported honestly rather than
rounded up: the *direction* replicates (full-depth's defect rate is
higher than mid's, and every defect flag under this judge falls in the
full-depth arm, none in mid) but the *magnitude* shrinks sharply and the
CIs overlap — GPT-5.5 flags this defect category far less readily than
Gemini does on the identical images. This is weak-to-moderate
independent support for the direction of the claim, not a strong
replication of its size, and we say so rather than claim more.

## Method
1. **Training-free path.** During a "capture" forward pass, run the
   reference image through the frozen DiT restricted to a contiguous band
   of transformer layers; at each active layer, store the post-RoPE
   key/value for reference tokens. During the real generation forward
   pass, concatenate the stored key/value onto the target sequence's own
   key/value at the same layers, scaled by a taper schedule that decays
   across denoising steps (value-scaling, not key-scaling, because
   QK-RMSNorm makes any key scalar multiple a no-op).
2. **The position-encoding fix.** Directly inspecting the transformer's
   coordinate construction shows every image slot's spatial axes share the
   same coordinate origin — naively injecting the reference's own
   position-encoded key makes it attend as if physically located in the
   target's top-left sub-block, producing literal copy-paste ghosting.
   Re-centering fixes the ghosting but collides with the model's freedom
   to choose a new pose, causing anatomical distortion instead. The fix
   re-applies RoPE at the target-centered coordinate but nulls all but the
   lowest-k frequency-pair channels — conveying coarse spatial layout
   without exact pixel-position binding.
3. **Training-light path.** A rank-8 LoRA (0.06% of backbone parameters)
   on the Q/K/V/O projections within the same layer band, trained
   end-to-end on ~6k reference→target pairs, learning to route to the
   right identity information in-context rather than being handed raw
   reference K/V — targeting the training-free path's structural
   copy-paste failure mode from a different angle.
4. **Defect-aware held-out evaluation.** Any recipe tuned on a dev set is
   re-validated on a disjoint held-out set, judged by a VLM on identity
   match AND visible composition defects jointly, not identity alone —
   because a recipe can win on identity-only scoring on a dev set while
   its composition collapses on fresh data (observed: an 82% held-out
   defect rate for a dev-set-preferred recipe, invisible to the dev-set's
   identity-only protocol).
5. **Pre-registered layer-position ablation (training-light path).**
   Before running it, lock the decision rule: a specific layer band counts
   as "the identity band" only if its identity-match confidence interval
   doesn't overlap the other tested bands'. Result, reported regardless of
   direction: it does not hold (all four tested bands' identity-match CIs
   overlap, including under subject-level bootstrap clustering) — but an
   unplanned, statistically distinct finding survives clustering: the same
   band is uniquely low on training-induced rendering-defect rate. Layer
   position governs how much collateral rendering damage LoRA training
   does, not how much identity-conditioning capacity it learns — a
   different and more specific claim than the one the ablation was
   designed to test.
6. **Cross-judge check.** Re-judge the mid-vs-full-depth pairs with a
   second model family (GPT-5.5, not Gemini), same prompt, disjoint from
   the judge used to produce every other number in this work. Report the
   result as-is: direction replicates, magnitude and CI width do not —
   a real independence check, not a rhetorical one, and its limits are
   part of the reported result rather than omitted.
