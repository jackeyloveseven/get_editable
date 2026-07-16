# aaai_latex_submission

Compilable AAAI-27 LaTeX source, mirrored from the `cherry` project's
`paper/` directory. Modeled on the `image-edit-lens` sibling project's
file layout and prose conventions (numbered `aaai_draft/NN_*.tex`
sections, `\input`-assembled `main_aaai.tex`, unnumbered AAAI sections —
do not add `Sec.~\ref{}` cross-references, they render empty under this
template's `secnumdepth=0`).

Compiles to exactly **7 pages of body + 1 page of references** (AAAI-27's
7-page hard cap on body content). Build with:

```
pdflatex -interaction=nonstopmode main_aaai.tex
bibtex main_aaai
pdflatex -interaction=nonstopmode main_aaai.tex
pdflatex -interaction=nonstopmode main_aaai.tex
```

`main_aaai.pdf` in this folder is the already-compiled output (2026-07-16)
so it's viewable without a LaTeX toolchain. Build artifacts (`.aux`,
`.log`, `.bbl`, `.blg`) are intentionally not included — regenerate them
with the commands above if rebuilding.

Known cosmetic issue, not fixed yet: each figure in `figs/` carries its
own title bar baked into the PNG (from the source figure-generation
pipeline in `../figures/`) *and* LaTeX adds its own `Figure N:` caption
below it — mildly redundant, does not affect page count or correctness.
