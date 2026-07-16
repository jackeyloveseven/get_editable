#!/usr/bin/env python3
"""
make_baseline_comparison_qualcheck.py

Figure contract (nature-figure skill):
  Core conclusion: the four arms in Section 4.4 fail in three qualitatively
    different, visually distinguishable ways -- not just different
    aggregate defect percentages.
  Evidence chain: 2 held-out subjects x 5 columns (reference, strip,
    lowfreq, IP-Adapter-SDXL, krea2-identity-edit). Row 1 (u7122) shows all
    four arms matching identity but only two staying defect-free, with the
    other two failing geometrically differently (collage vs garbled hand).
    Row 2 (u5648) shows a hard subject where every arm drops the
    identity-defining accessory, but krea2 normalizes it away rather than
    garbling it -- the "drops/normalizes vs garbles" distinction the prose
    argues for from the per-subject analysis.
  Archetype: image plate + quant (white plate; these are anime portraits,
    not microscopy).
  Backend: Python / matplotlib (selected explicitly, no R involved).

Run:
  python3 make_baseline_comparison_qualcheck.py
Reads (raw, unmodified project artifacts -- see data_sources.json for the
resolved judge-verdict provenance):
  /mnt/local/smoke_anime/anime_260623/img/u{uid}_ref.png
  /mnt/local/cherry_out/holdout30/u{uid}_holdout_{old,new}_s{seed}.png
  /mnt/local/cherry_out/baseline_ipadapter/u{uid}_ipadapter_s{seed}.png
  /mnt/local/cherry_out/baseline_krea2/u{uid}_krea2_s{seed}.png
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

PATHS = {
    "ref": "/mnt/local/smoke_anime/anime_260623/img/u{uid}_ref.png",
    "strip": "/mnt/local/cherry_out/holdout30/u{uid}_holdout_old_s{s}.png",
    "lowfreq": "/mnt/local/cherry_out/holdout30/u{uid}_holdout_new_s{s}.png",
    "ipadapter": "/mnt/local/cherry_out/baseline_ipadapter/u{uid}_ipadapter_s{s}.png",
    "krea2": "/mnt/local/cherry_out/baseline_krea2/u{uid}_krea2_s{s}.png",
}

ROWS = [
    ("7122", "0",
     "all 4 match identity; only strip/krea2\nare defect-free — lowfreq collages,\nIP-Adapter garbles the hand"),
    ("5648", "0",
     "hard subject: iconic mask dropped or\nmis-rendered by every arm — krea2\nnormalizes it away instead of garbling it"),
]

COL_TITLES = [
    ("reference", ""),
    ("strip / band21-25", "defect 20% [12,32]"),
    ("lowfreq / band10-25", "defect 82% [70,89]"),
    ("IP-Adapter-SDXL", "defect 33%"),
    ("krea2-identity-edit", "defect 2% [0,9]"),
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
    n_rows, n_cols = len(ROWS), 5
    fig = plt.figure(figsize=(9.6, 4.4))
    gs = gridspec.GridSpec(
        n_rows, n_cols + 1,
        width_ratios=[1, 1, 1, 1, 1, 0.95],
        hspace=0.22, wspace=0.06,
        left=0.02, right=0.99, top=0.80, bottom=0.02,
    )

    fig.suptitle(
        "Baseline comparison ($\\S$4.4) — four arms, three qualitatively different failure signatures",
        x=0.02, ha="left", fontsize=10.5, fontweight="bold", y=0.98,
    )

    for r, (uid, s, note) in enumerate(ROWS):
        keys = ["ref", "strip", "lowfreq", "ipadapter", "krea2"]
        for c, key in enumerate(keys):
            p = PATHS[key].format(uid=uid, s=s)
            ax = fig.add_subplot(gs[r, c])
            place_image(ax, p)
            if r == 0:
                title, sub = COL_TITLES[c]
                ax.text(0.0, 1.22, title, transform=ax.transAxes,
                        fontsize=7.6, fontweight="bold", color=PALETTE["neutral_dark"],
                        ha="left", va="bottom")
                if sub:
                    ax.text(0.0, 1.05, sub, transform=ax.transAxes,
                            fontsize=6.0, color=PALETTE["neutral_mid"], ha="left", va="bottom")
        ax_lab = fig.add_subplot(gs[r, n_cols])
        ax_lab.set_axis_off()
        ax_lab.text(0.0, 0.85, f"u{uid}_s{s}", fontsize=7, fontweight="bold", color=PALETTE["neutral_dark"], va="top")
        ax_lab.text(0.0, 0.68, note, fontsize=6.0, color=PALETTE["neutral_mid"], va="top")

    out_stub = os.path.join(HERE, "baseline_comparison_qualcheck")
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
