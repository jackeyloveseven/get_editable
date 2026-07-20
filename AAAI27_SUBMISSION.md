# AAAI-27 Submission Sheet — cherry paper

*Copy-paste-ready title / abstract / keywords for the submission system. Generated 2026-07-20 after the positive-thesis reframe.*

---

## Key facts

| | |
|---|---|
| **Venue** | AAAI-27 Main Technical Track |
| **Full-paper (modification) deadline** | **July 28, 2026** (Phase-1; official CFP — "no exceptions") |
| **Page limit** | ≤ 7 pages non-reference content; references on pages 8–9 (max 9 total) |
| **Current status** | Reframed to positive thesis, compiles clean, body ends p7 / refs p8 — **compliant** |
| **Submission source** | `aaai_latex_submission/main_aaai.tex` (canonical) · working draft `draft_v0.md` |
| **Repo** | `github.com/jackeyloveseven/get_editable` (commit `cbdbd37`) |

---

## Title

```
One-Shot Subject Identity Is a Distributed, Composition-Entangled Code in a Frozen Text-to-Image Transformer
```

---

## Abstract

Reference-conditioned identity preservation is normally bought with a purpose-trained adapter or an edit-pretrained base; neither transfers to a frozen, pure text-to-image (T2I) anime diffusion transformer that must stay unmodified. We turn such a model into a one-shot identity conditioner two cheap ways — training-free attention key/value injection from a reference image, and a 0.06%-parameter band-LoRA — each competitive with a same-class trained adapter (IP-Adapter) on identity match, with a shuffled-reference control confirming the effect is genuinely reference-driven, not a prompt-only floor. We then ask *where* this identity signal lives, and find no localized handle to grab: identity match is statistically flat across four transformer layer bands (a null that cross-set replicates and holds for both the injection and the trained mechanism), across reference-affinity-selected head subsets (matched-count controls), and across denoising-time gates. Under this **distributed, composition-entangled** account the persistent tiled/"collage" defect is predicted, not incidental: five targeted repairs — background isolation, head-gating, outlier-token suppression, late-timestep gating, and an identity-supervised training loss — each with a matched control, all trade identity for coherence rather than separating them. A second architecture reproduces the account only as a diagnostic: a peaked-versus-diffuse attention-selectivity signature separates a model that transfers structured identity from one that transfers only mean color. Throughout, we document a measurement hazard twice over — identity-match-only evaluation is systematically optimistic, recommending recipes that a joint identity-and-defect metric rejects (usable output 53%→15% in one mechanism; a CI-decisive 8.6× "defect gradient" our own audit refutes in the other).

---

## TL;DR

**EN —** A frozen text-to-image diffusion transformer can be turned into a one-shot identity conditioner (training-free, tying a trained adapter), but the identity it carries is *distributed and entangled with composition* — no layer, head, or timestep handle isolates it, so all five matched-control repairs of its "collage" defect trade identity away. Identity-match-only evaluation hides this, twice.

**ZH —** 冻结的文生图 DiT 能被免训练激活成一次性身份生成器(平了训练过的 adapter),但它承载的身份是**分布式、且与构图纠缠**的——没有层/头/时间步层面的局部把手,所以 5 个带对照的缺陷修复都只能拿身份换连贯。只看"身份匹配率"的评测会两次掩盖这一点。

---

## Submission keywords (AAAI-27 areas & topics — verbatim)

Rationale: the paper is a **mechanistic anatomy of a generative model + evaluation methodology**, not a SOTA method. Put the domain-accurate keyword primary (to reach reviewers who understand DiT / KV injection / RoPE), and use interpretability + evaluation keywords to signal that the contribution is *characterization*, not benchmark-chasing.

| Slot | Keyword |
|---|---|
| **Primary** | `CV: Diffusion & Generative Models for Vision` |
| **Secondary** | `ML: Transparent, Interpretable & Explainable ML` |
| **Additional** | `ML: Evaluation, Benchmarking, Datasets & Analysis` |
| Alt for the additional slot | `CV: Bias, Fairness, Privacy & Interpretability` |

---

## One-sentence thesis + three pillars (for reference)

**Thesis.** In frozen text-to-image DiTs, one-shot subject identity is carried as a *distributed, composition-entangled* attention code — strong enough that zero-training KV injection matches a trained adapter, yet with no layer-band, head-subset, or timestep handle, which predicts (and we confirm with matched controls) that every localized repair trades identity for coherence.

1. **Capability + control** — two cheap mechanisms reach trained-adapter parity; shuffled-reference control shows it is genuinely reference-driven.
2. **Anatomy (core)** — identity is non-localized (layer/head/timestep; cross-set-replicated, both mechanisms) and composition-entangled; five matched-control repairs confirm; RoPE-coordinate mechanism for ghosting; early co-forming.
3. **Diagnostic + protocol** — peaked-vs-diffuse selectivity signature (diagnostic, n=2); paired identity-and-defect cross-set protocol, motivated by two dev→held-out reversals.
