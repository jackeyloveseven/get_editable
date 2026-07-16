# baseline_comparison_qualcheck

## Intent
Support §4.4's claim that the four compared arms (this work's `strip` and
`lowfreq`, IP-Adapter-SDXL, krea2-identity-edit) don't just differ in
aggregate defect percentage but fail in three visually distinct, mechanism-
consistent ways: composition collage (lowfreq), garbled hand anatomy
(IP-Adapter), and accessory normalization/drop (krea2, on subjects the
prose already names, e.g. u5648's mask).

## Goal / target caption
"Two held-out subjects across all four arms. Top row (u7122): identity
matches in all four, but only `strip` and krea2 are defect-free — `lowfreq`
collages a duplicate inset, IP-Adapter garbles the hand near the face.
Bottom row (u5648): a hard subject (an iconic mask/accessory) that defeats
every arm's identity match, but not identically — `strip`, IP-Adapter, and
krea2 all drop the mask and render a normalized face rather than garbling
it, while `lowfreq` additionally collages small mask-motif fragments on top
of losing the accessory. This 'drops/normalizes vs. garbles' distinction is
exactly the failure-signature split §4.4's per-subject analysis argues for,
made visible rather than asserted from aggregate defect percentages alone."

Placed in `draft_v0.md` §4.4, right after the "three qualitatively
different failure mechanisms" paragraph.

## Selection method
Row 1 (u7122): chosen because judge verdicts exist and are `visible_defect
=yes` for both lowfreq (note explicitly says "collaged") and IP-Adapter
(note explicitly says "garbled hand anatomy") on the same subject/seed,
while `strip` and krea2 are both clean (`visible_defect=no`) on that same
subject/seed — a genuine four-way apples-to-apples case, not composited
from different subjects.
Row 2 (u5648): chosen because the prose in §4.4 (per-subject analysis
paragraph) already names this subject by uid as krea2's characteristic
failure pattern ("subject 5648's mask is discarded entirely rather than
mis-rendered, and its goggles are replaced with ordinary eyes") — the
figure makes an existing textual claim checkable rather than introducing a
new, unverified example.

## Archetype (nature-figure contract)
Image plate + quant. Core conclusion: three failure signatures are visually
distinct. Evidence chain: 2 rows, each isolating one comparison axis
(identity-match-but-different-defects; identity-miss-but-different-miss-style).

## Regenerating
`python3 make_baseline_comparison_qualcheck.py` (Python/matplotlib, no R).
See `data_sources.json` for resolved paths and judge verdicts.
