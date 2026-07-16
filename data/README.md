# docs/paper/data — snapshot of every data file cited in the paper

This is a **snapshot copy** (not a symlink, not the live pipeline output)
of every `data/*.json` / `data/*.jsonl` file that a number or table cell
in `draft_v0.md`, `section_trainlight_band_ablation.md`, or a figure's
`data_sources.json` traces back to. Copied here so the paper materials are
self-contained and don't require the full `cherry/` project checkout to
audit a number. Copied 2026-07-16.

The **generation scripts** (`probe/58_band_ablation_run.sh`,
`probe/59_band_ablation_report.py`, the figure `make_*.py` scripts) still
read from the live project `../../../data/` — this folder is a mirror for
paper-package portability, not a redirect target. If you regenerate a
figure or re-run the report, the live `data/` is the source of truth;
re-copy here afterward if the numbers changed.

| File | What it backs | Cited in |
|---|---|---|
| `gemini_judge_holdout.json` | §4.3 held-out reversal (old=strip/band21-25 vs new=lowfreq/band10-25, n=60 each, 120 total) | draft_v0.md §4.3, `figures/holdout_reversal_defect_qualcheck/` |
| `gemini_judge_ipadapter.json` | §4.4 baseline: IP-Adapter-SDXL on the same 30 held-out subjects | draft_v0.md §4.4, `figures/baseline_comparison_qualcheck/` |
| `gemini_judge_krea2.json` | §4.4 baseline: krea2-identity-edit (12.9B) on the same 30 held-out subjects | draft_v0.md §4.4, `figures/baseline_comparison_qualcheck/` |
| `gemini_judge_devprobe_base_rep30_step4000.json` | training-light band ablation, mid 10-25 (r8) arm — the pre-existing reference arm | section_trainlight_band_ablation.md, `figures/band_ablation_defect_qualcheck/` |
| `gemini_judge_band_0-15_r8_step4000.json` | training-light band ablation, early 0-15 (r8) arm | section_trainlight_band_ablation.md |
| `gemini_judge_band_14-29_r8_step4000.json` | training-light band ablation, late 14-29 (r8) arm | section_trainlight_band_ablation.md, `figures/band_ablation_defect_qualcheck/` |
| `gemini_judge_band_0-29_r4_step4000.json` | training-light band ablation, full-depth 0-29 (r4) arm | section_trainlight_band_ablation.md, `figures/band_ablation_defect_qualcheck/` |
| `band_ablation_summary.txt` | orchestrator log: per-arm train/gen/judge timestamps + the auto-generated 4-arm Wilson-CI comparison table | section_trainlight_band_ablation.md |
| `repprobe30_tags.json` | the 30-subject held-out dev set (rep30) subject tags/prompts used for all training-light arm generations | referenced by `probe/46_devprobe.py`, `probe/58_band_ablation_run.sh` |
| `holdout30_tags.json` | the 30-subject held-out set subject tags/prompts used for §4.3/§4.4 (training-free line + baselines) | referenced by `probe/41_holdout_eval.py`, `probe/42_baseline_ipadapter.py`, `probe/43_baseline_krea2.py` |
| `train_manifest_mix3.jsonl` | the 5,936-row / 3,339-subject training manifest used for all four band-ablation LoRA training runs | referenced by `probe/45_train_lora.py --manifest`, `probe/58_band_ablation_run.sh` |

## Subject-clustered CI recompute (not a stored file — reproducible one-liner)

The subject-clustered bootstrap CIs quoted in
`section_trainlight_band_ablation.md`'s caveats section (mid 6.7% [0,17]
vs full-depth 43.3% [27,60] defect rate) were computed ad hoc from the
four `gemini_judge_band_*` / `gemini_judge_devprobe_base_rep30_step4000`
files above — collapse each subject's 2 seeds to one any-defect
indicator, bootstrap-resample at the subject level (n=30, 20000
resamples). No separate output file was saved; re-run against the files
in this folder to reproduce.
