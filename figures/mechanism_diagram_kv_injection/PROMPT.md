# mechanism_diagram_kv_injection — generation prompt

## Intent
§3 (Method) of `draft_v0.md` describes the core training-free ref-KV
injection mechanism entirely in prose across four subsections (3.1 base
mechanism, 3.2 the position problem, 3.3 the frequency-gated fix, 3.4 why
band width and position-gating interact) with zero supporting figure. This
is the one figure in the whole project with no raw pixels to reuse — it
has to be authored as a concept illustration. Target: a single wide
mechanism schematic a reader can grasp before reading the prose in detail.

## Target caption
"The ref-KV injection mechanism. (a) A capture pass runs the reference
image through a frozen T2I DiT restricted to layers 10-25; at each active
layer the post-RoPE key/value for reference tokens is stored. (b) Naively
injecting the reference's own position-encoded key causes it to attend as
if physically located in the target's top-left sub-block (same coordinate
origin), producing copy-paste ghosting; our fix re-centers the reference's
RoPE coordinates to the target's middle and nulls all but the lowest-k
frequency channels, conveying coarse position only. (c) During the real
generation pass, the captured key/value is concatenated onto the target
sequence's own key/value at the same layers before attention, scaled by a
taper schedule w(t) that decays across denoising steps -- producing an
output that preserves identity without ghosting or composition collapse."

## Selection method / honesty note
This is an AI-generated conceptual illustration, not a rendering of real
data -- unlike every other figure in this project. No quantitative values
appear in it. Treat it as a comprehension aid only; if any generated label
is misspelled or a shape reads ambiguously, that part should be redrawn as
a vector overlay (matplotlib/Illustrator/Inkscape) before submission, not
shipped as-is. See QA notes at the bottom of this file after generation.

## Reference style research (before writing the prompt)
Looked at how this genre of mechanism is conventionally drawn in
IP-Adapter / InstantID / ControlNet-style figures: a transformer block
stack with a highlighted layer subset, K/V tensors flowing as small
labeled arrows between a "capture" pass and an "inject" pass, RoPE/position
grids shown as small 2D coordinate-origin diagrams, taper/weight schedules
shown as a small decaying-curve or dial icon. That vocabulary is reused
below, adapted to this project's specific mechanism (frozen DiT, band of
layers not the whole stack, frequency-gated RoPE rather than a generic
position embedding).

## Final prompt (sent to the image model verbatim)

```
A scientific schematic illustration in the style of a NeurIPS/ICML/AAAI
method figure -- flat vector illustration, white background, no
photorealism except for two small anime-portrait thumbnails described
below, no drop shadows, no 3D bevels, clean thin 1-1.5pt strokes,
sans-serif publication font for all labels, generous white space.

CORE CONCEPT
The figure illustrates how a frozen text-to-image diffusion transformer is
turned into a reference-image identity conditioner via attention
key/value injection, including the position-handling fix that avoids
copy-paste artifacts.

LAYOUT
Single wide horizontal composite with three clearly separated panels,
each with a small lowercase italic sub-label (a, b, c) in its top-left
corner, thin vertical divider lines or generous whitespace gaps between
panels. Aspect ratio 21:9.

PANEL A (left, ~30% width) -- "capture pass"
A small reference anime portrait thumbnail (simple, clean anime style,
NOT photorealistic) feeds into a tall vertical stack of ~30 thin
horizontal bars representing frozen transformer blocks. A contiguous
subset of about 16 of these bars, roughly in the lower-middle of the
stack, is filled in deep blue while all other bars are filled light gray
outline-only. Small "K" and "V" tensor icons (simple labeled rounded
rectangles) are shown being extracted from the blue-highlighted bars only,
flowing sideways into a small tray/cache icon labeled "captured K, V" in
warm amber.

PANEL B (center, ~25% width) -- "position problem and fix", stacked
vertically as two small sub-diagrams
Top sub-diagram: two overlapping squares sharing the exact same top-left
corner (a small square labeled "ref" inside a larger square labeled
"target"), with a small red X and a tiny inset thumbnail showing a
duplicated ghost face in the top-left corner of an otherwise clean
portrait -- this represents a failure case.
Bottom sub-diagram: the same two squares, but now the small "ref" square
is centered in the middle of the larger "target" square, with a soft
radial gradient (concentric rings, dense color at center fading to gray
at the edges) drawn inside the small square -- representing coarse,
low-frequency-only position information. No ghost artifact shown here.

PANEL C (right, ~45% width) -- "generation (inject) pass"
A second, taller vertical transformer block stack (same visual convention
as panel A: ~30 thin bars, the same contiguous subset highlighted in deep
blue). Thin arrows flow from panel A's "captured K, V" tray into the
blue-highlighted bars of this second stack, merging at each one with a
small "+" icon (representing concatenation with the stack's own local
key/value). Next to the merge points, a small dial or decaying-curve icon
labeled "taper w(t)" shows a value decreasing from high to near-zero
across the vertical extent of the stack. An arrow leads from the top of
the stack to a final small clean anime-portrait output thumbnail on the
far right, drawn in a new pose and background but with visibly matching
hair color and eye color to panel A's reference thumbnail -- representing
successful, ghost-free identity transfer.

ANNOTATIONS (keep every label short, 1-4 words, correctly spelled, do not
overcrowd -- roughly 10 labels total across all three panels):
- "reference" under panel A's input thumbnail
- "frozen T2I, layers 10-25" near panel A's highlighted block subset
- "captured K, V" on the amber tray
- "same origin -> ghost" under panel B's top sub-diagram
- "recentered + low-freq only" under panel B's bottom sub-diagram
- "generation pass" above panel C's stack
- "concat + w(t)" near the merge icons
- "taper w(t) -> 0" near the dial icon
- "generated, identity preserved" under panel C's output thumbnail
- small panel labels "a", "b", "c"

COLOR PALETTE (must remain legible in grayscale -- rely on fill vs.
outline, not color alone, for the highlighted-vs-not distinction):
- white background
- deep blue #0F4D92 for highlighted "band" transformer blocks (filled)
- medium gray #767676 for non-highlighted transformer blocks (outline
  only, no fill)
- warm amber #D97706 for the reference path (input thumbnail border,
  captured-K/V tray)
- red #B64342 used only for the single X mark in panel B's failure case
- no other colors

CONSTRAINTS
No fabricated numbers, no equations, no logos, no institution marks, no
author names. No text longer than 4 words anywhere. This must read as a
rigorous machine-learning method figure, not a marketing graphic.
```

## Model / call details
- Endpoint: Kaon Router (`https://kaon-router.kaonai.com/v1/images/generations`),
  model `huskyi/gpt-image-2`, `size=1536x1024` (closest supported size to
  the requested 21:9 aspect ratio), `n=1`.
- Not OpenRouter -- the project's available credential is a Kaon Router
  key; verified working against this base URL/model pair before spending
  the real generation call (see `generate.py`'s inline comment).
- Regenerate with: `python3 generate.py`

## QA notes (2026-07-16, first generation, no regeneration needed)
- [x] All labels legible and correctly spelled: "reference", "frozen T2I,
      layers 10-25", "captured K, V", "same origin -> ghost", "recentered
      + low-freq only", "generation pass", "concat + w(t)", "taper w(t) ->
      0", "generated, identity preserved", panel labels a/b/c. No
      misspellings.
- [x] Highlighted-band distinction readable in grayscale: the band-10-25
      blocks are solid-filled deep blue, all other blocks are
      outline-only light gray -- fill-vs-outline holds independent of hue,
      not a color-only encoding.
- [x] No hallucinated numbers, logos, author marks, or institution marks.
- [ ] Minor, optional before final submission: the "high"/"low" axis
      labels next to the taper decay curve in panel (c) are placed a bit
      loosely relative to the curve -- would benefit from a light vector
      touch-up (nudge label position) in Illustrator/Inkscape, but is not
      a blocker for using this as-is in a draft.
- Overall: usable as the §3 method figure without redrawing anything
  structural. Treat as a conceptual illustration per the safety note
  above (no data encoded in it).
