#!/usr/bin/env python3
"""
make_band_ablation_defect_qualcheck.py

Figure contract (nature-figure skill):
  Core conclusion: the defect-rate gradient across band positions (mid
    3.3% vs full-depth 28.3%, CI-decisive even after subject-clustering)
    is a real, visible rendering-damage difference, not a judge artifact --
    on the same subject/seed, mid stays clean where full-depth garbles
    anatomy.
  Evidence chain: 3 held-out subjects x (reference, mid 10-25, full-depth
    0-29), all three full-depth cells flagged visible_defect=yes by the
    judge and independently confirmed by inspection.
  Archetype: image plate + quant.
  Backend: Python / matplotlib (selected explicitly, no R involved).

Run:
  python3 make_band_ablation_defect_qualcheck.py
Reads (raw, unmodified project artifacts -- see data_sources.json):
  /mnt/local/smoke_anime/anime_260623/img/u{uid}_ref.png
  /mnt/local/cherry_out/devprobe/base_rep30_step4000/u{uid}_lora_s{seed}.png
  /mnt/local/cherry_out/devprobe/lora_ckptlink_band0-29_r4/u{uid}_lora_s{seed}.png
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
}

HERE = os.path.dirname(os.path.abspath(__file__))
REF = "/mnt/local/smoke_anime/anime_260623/img/u{uid}_ref.png"
MID = "/mnt/local/cherry_out/devprobe/base_rep30_step4000/u{uid}_lora_s{s}.png"
FULLDEPTH = "/mnt/local/cherry_out/devprobe/lora_ckptlink_band0-29_r4/u{uid}_lora_s{s}.png"

ROWS = [
    ("5369", "0", "arm/jacket geometry broken"),
    ("5369", "1", "garbled hand + strap anatomy"),
    ("1740", "1", "mangled hand at collar"),
]

COL_TITLES = [
    ("reference", ""),
    ("mid 10-25 (r8)", "clean-yes 21.7% · defect 3.3%"),
    ("full-depth 0-29 (r4)", "clean-yes 23.3% · defect 28.3%"),
]


def place_image(ax, path):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(PALETTE["neutral_light"])
        s.set_linewidth(0.8)
    if os.path.exists(path):
        im = Image.open(path).convert("RGB")
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
        "Band-position ablation — qualitative spot check confirms the defect-rate gradient is real",
        x=0.03, ha="left", fontsize=10.5, fontweight="bold", y=0.985,
    )

    for r, (uid, s, note) in enumerate(ROWS):
        paths = [REF.format(uid=uid), MID.format(uid=uid, s=s), FULLDEPTH.format(uid=uid, s=s)]
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
        ax_lab = fig.add_subplot(gs[r, n_cols])
        ax_lab.set_axis_off()
        ax_lab.text(0.0, 0.85, f"u{uid}_s{s}", fontsize=7, fontweight="bold", color=PALETTE["neutral_dark"], va="top")
        ax_lab.text(0.0, 0.60, "judge note\n(full-depth):", fontsize=6.3, color=PALETTE["neutral_mid"], va="top")
        ax_lab.text(0.0, 0.40, note, fontsize=6.3, color=PALETTE["red_strong"], va="top", wrap=True)

    out_stub = os.path.join(HERE, "band_ablation_defect_qualcheck")
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
