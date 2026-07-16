#!/usr/bin/env python3
"""
make_holdout_reversal_defect_qualcheck.py

Regenerates holdout_reversal_defect_qualcheck.{svg,pdf,tiff,png} from raw
generated images already on disk (nothing here is a mock/placeholder).

Figure contract (nature-figure skill):
  Core conclusion: the held-out 82% defect rate for the lowfreq/band10-25
    recipe is a real, visible collage/tiling failure on the same subjects
    where the old strip/band21-25 recipe stays clean -- not a measurement
    artifact of the Gemini judge.
  Evidence chain: 3 held-out subjects x 3 columns (reference, old-recipe
    output, new-recipe output). Each row is one piece of evidence: same
    subject/seed, only the injection recipe differs, defect only appears
    in the new column. Column headers carry the arm-level aggregate defect
    rate (Wilson 95% CI, n=60) so the reader can connect the single visible
    instance to the population-level claim.
  Archetype: image plate + quant (Pattern 13 adapted to a white plate --
    these are anime portraits, not microscopy, so the "black background"
    convention does not apply; see stance.md).
  Backend: Python / matplotlib (selected explicitly, no R involved).

Run:
  python3 make_holdout_reversal_defect_qualcheck.py
Requires:
  matplotlib, Pillow (both already present in the project's base env)
Reads (raw, unmodified project artifacts -- see data_sources.json in this
folder for the resolved judge-verdict provenance):
  /mnt/local/smoke_anime/anime_260623/img/u{uid}_ref.png
  /mnt/local/cherry_out/holdout30/u{uid}_holdout_{old,new}_s{seed}.png
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.linewidth": 0.8,
})

PALETTE = {
    "neutral_dark": "#4D4D4D",
    "neutral_mid": "#767676",
    "neutral_light": "#CFCECE",
    "red_strong": "#B64342",
    "blue_main": "#0F4D92",
}

HERE = os.path.dirname(os.path.abspath(__file__))
REF = "/mnt/local/smoke_anime/anime_260623/img/u{uid}_ref.png"
HOLD = "/mnt/local/cherry_out/holdout30/u{uid}_holdout_{arm}_s{s}.png"

ROWS = [
    ("7122", "0", "Severe collaged inset,\nheadless torso"),
    ("7285", "0", "Severe collage,\nstacked duplicated images"),
    ("752", "1", "Collaged inset,\ntop-left"),
]

COL_TITLES = [
    ("reference", ""),
    ("strip / band21-25 (old)", "defect 20% [12,32]"),
    ("lowfreq / band10-25 (new)", "defect 82% [70,89]"),
]


def load_fit(path, target_aspect=1.0):
    im = Image.open(path).convert("RGB")
    return im


def place_image(ax, path):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(PALETTE["neutral_light"])
        s.set_linewidth(0.8)
    if os.path.exists(path):
        im = load_fit(path)
        ax.imshow(im, aspect="equal")
    else:
        ax.set_facecolor("#fbeaea")
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=6, color=PALETTE["red_strong"])


def main():
    n_rows, n_cols = len(ROWS), 3
    fig = plt.figure(figsize=(7.2, 7.6))
    gs = gridspec.GridSpec(
        n_rows, n_cols + 1,
        width_ratios=[1, 1, 1, 0.62],
        hspace=0.20, wspace=0.06,
        left=0.03, right=0.98, top=0.88, bottom=0.02,
    )

    fig.suptitle(
        "Held-out reversal ($\\S$4.3) — the wide-band recipe collages, it doesn't just miss identity",
        x=0.03, ha="left", fontsize=10.5, fontweight="bold", y=0.985,
    )

    for r, (uid, s, note) in enumerate(ROWS):
        paths = [
            REF.format(uid=uid),
            HOLD.format(uid=uid, arm="old", s=s),
            HOLD.format(uid=uid, arm="new", s=s),
        ]
        for c, p in enumerate(paths):
            ax = fig.add_subplot(gs[r, c])
            place_image(ax, p)
            if r == 0:
                title, sub = COL_TITLES[c]
                ax.text(0.0, 1.22, title, transform=ax.transAxes,
                        fontsize=8, fontweight="bold", color=PALETTE["neutral_dark"],
                        ha="left", va="bottom")
                if sub:
                    ax.text(0.0, 1.05, sub, transform=ax.transAxes,
                            fontsize=6.3, color=PALETTE["neutral_mid"], ha="left", va="bottom")
        # row label + caption panel
        ax_lab = fig.add_subplot(gs[r, n_cols])
        ax_lab.set_axis_off()
        ax_lab.text(0.0, 0.85, f"u{uid}_s{s}", fontsize=7, fontweight="bold", color=PALETTE["neutral_dark"], va="top")
        ax_lab.text(0.0, 0.60, "judge note\n(new arm):", fontsize=6.3, color=PALETTE["neutral_mid"], va="top")
        ax_lab.text(0.0, 0.40, note, fontsize=6.3, color=PALETTE["red_strong"], va="top", wrap=True)

    out_stub = os.path.join(HERE, "holdout_reversal_defect_qualcheck")
    fig.savefig(out_stub + ".svg", bbox_inches="tight")
    fig.savefig(out_stub + ".pdf", bbox_inches="tight")
    fig.savefig(out_stub + ".png", dpi=300, bbox_inches="tight")
    try:
        fig.savefig(out_stub + ".tiff", dpi=600, bbox_inches="tight")
    except Exception as e:
        print("tiff export skipped:", e)
    print("saved", out_stub, "+ .svg/.pdf/.png(/.tiff)")


if __name__ == "__main__":
    main()
