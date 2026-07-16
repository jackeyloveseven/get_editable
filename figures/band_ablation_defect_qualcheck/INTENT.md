# band_ablation_defect_qualcheck

## Intent
Support the training-light section's claim that the defect-rate gradient
found by the 4-arm band-position ablation (mid 3.3% vs full-depth 28.3%,
CI-decisive both per-image and after subject-clustering) is a real
rendering-damage difference, not a Gemini judge artifact. This was
explicitly requested by the user ("肉眼核实" -- eyeball-verify before
writing it up) before the finding was allowed into the paper's core
narrative.

## Goal / target caption
"Three held-out subjects, mid 10-25 (r8) vs full-depth 0-29 (r4), same
subject/seed. All three full-depth cells are flagged `visible_defect=yes`
by the judge; inspection confirms each is a genuine anatomical/geometric
break (arm/jacket geometry, garbled hand + strap routing, mangled hand at
the collar) absent in the matched mid-band generation. Selected from 17
full-depth defect flags project-wide; not exhaustive, but rules out 'the
whole gradient is judge noise' as an explanation."

Placed in `docs/paper/section_trainlight_band_ablation.md`, in the
"Qualitative confirmation" subsection.

## Selection method
3 of 17 full-depth `visible_defect=yes` entries in
`data/gemini_judge_band_0-29_r4_step4000.json`, chosen for having the most
specific/checkable judge notes (not a random sample -- this selection bias
is disclosed in the section text itself).

## Archetype (nature-figure contract)
Image plate + quant.

## Regenerating
`python3 make_band_ablation_defect_qualcheck.py` (Python/matplotlib, no R).
See `data_sources.json` for resolved paths and judge verdicts.
