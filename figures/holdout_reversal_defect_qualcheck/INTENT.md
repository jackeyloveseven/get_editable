# holdout_reversal_defect_qualcheck

## Intent
Support §4.3's claim that the `lowfreq`/band10-25 recipe's 82% held-out
defect rate is a real, visible collage/tiling failure, not a measurement
artifact of the Gemini judge. The quantitative table already states the
number; this figure is the reader's chance to see the actual failure mode
on the same subjects where the old recipe stays clean.

## Goal / target caption
"Three held-out subjects, `strip` (old) vs `lowfreq`/band10-25 (new), same
subject/seed. All three cases show `same_character=yes` for the new recipe
— the identity signal is there — but a collaged or tiled sub-image
(consistently top-left or stacked) corrupts the composition, matching the
`visible_defect` judge notes verbatim. This is the failure mode behind the
82% defect rate: not an occasional artifact, but a near-default outcome of
the wide-band frequency-gated recipe."

Placed in `draft_v0.md` §4.3, right after the defect-rate table and before
the CI-decisiveness caveat paragraph.

## Selection method
3 subjects pulled from `data/gemini_judge_holdout.json` where: new-arm
entry has `visible_defect=yes` AND `same_character=yes`, and the matching
old-arm entry (same uid+seed) has `visible_defect=no` AND
`same_character=yes` — i.e. a clean apples-to-apples pair where only the
recipe changed the outcome. Not cherry-picked beyond this filter; the
filter itself returned only 4 qualifying pairs project-wide (see
data_sources.json), of which 3 were used.

## Archetype (nature-figure contract)
Image plate + quant. Core conclusion: recipe reversal is visible, not
just tabulated. Evidence chain: 3 rows x (ref, old, new) with column-level
aggregate defect rate as the population-level anchor for each single
visible instance.

## Regenerating
`python3 make_holdout_reversal_defect_qualcheck.py` (Python/matplotlib,
no R). Reads raw project artifacts directly — see `data_sources.json` for
resolved paths and judge verdicts. Safe to re-run any time; will reflect
whatever is currently on disk at those paths.
