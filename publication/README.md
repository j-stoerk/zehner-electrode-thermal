# Publication package

Complete, self-contained manuscript draft. **Verified to compile with MiKTeX pdfLaTeX (12 pages, 0 errors, 0 undefined references).**

## Compile in TeXworks

1. Open `main.tex` in TeXworks
2. Select **pdfLaTeX** in the dropdown
3. Compile **twice** (for cross-references). No BibTeX run needed — the bibliography is embedded.

## Contents

| File | Purpose |
|---|---|
| `main.tex` | Full manuscript (12 pp. compiled), structured per NeurIPS best practices: explicit contributions C1-C5, related-work section comparing approach families (effective-medium closures / microstructure-resolved simulation / rarefied-gas theory / measurements), assumptions A1-A5, numbered equations (1)-(11), baseline comparison (Wiener bounds, Maxwell-Eucken, Bruggeman), ablation M0->M1->M2, identifiability + transfer analysis, discussion of approach trade-offs, explicit Limitations section, Reproducibility statement, proposed experiments with falsification criteria; embedded bibliography (20 refs) |
| `main.pdf` | Compiled output (regenerate after edits) |
| `references.bib` | BibTeX database for journal submission (switch `main.tex` to `\bibliography{references}` + natbib then) |
| `figures/fig1_knudsen.pdf` … `fig6_qc.pdf` | Vector figures, journal styling |
| `make_figures.py` | Regenerates all six figures from `../src` + `../data` — **no hard-coded numbers**; run from repo root: `.venv/Scripts/python.exe publication/make_figures.py` |

## Figure map

| Fig | Content | Manuscript section |
|---|---|---|
| 1 | Knudsen reduction vs. pore size (air + He), system markers, separator band | 4.1 |
| 2 | Zero-fit λ_eff(ψ) maps: anode/cathode/separator × wet/dry-continuum/dry-Knudsen | 4.1 |
| 3 | Zero-fit validation vs. Gandert calendering data (4 panels, λ_s bands) | 4.2 |
| 4 | Calibrated φ(Π) closure vs. measured (u-shape reproduced) | 4.3 |
| 5 | Error analysis: identifiability valley + transfer-failure decomposition | 4.5 |
| 6 | Inverse porosity QC: parity + error histogram (200-sheet batch) | 4.6 |

## Before submission — checklist

- [ ] Author list, affiliations, corresponding e-mail (placeholders in `main.tex`)
- [x] Acknowledgments + funding (BMBF project GranuGoIn, grant no. 03XP0635A, Projektträger Jülich)
- [ ] Repository URL/DOI in Data Availability (consider Zenodo archive of the repo)
- [x] Bibliography verified (Steinhardt 2022, Sangrós Giménez 2019, Eucken 1932, McMasters 1999 author list confirmed; Bauer/Burheim ranges accepted)
- [ ] Decision: submit modeling-only vs. add in-house campaign data (Section 5 of the paper specifies the experiments; PUBLICATION_METHODS.md §4 in the repo root has the full protocol)
- [ ] Journal choice: drafted with *Energy Technology* in mind (same venue as the primary dataset); for *J. Power Sources*, add the experimental campaign first
- [ ] Reformat to journal template (current layout is a generic article for review/circulation)
- [ ] Language/consistency pass; spell out any remaining "to our knowledge" claims after a final literature check

## Provenance

Every number in the manuscript traces to executed notebook output (`../Electrode.ipynb`, sections 2–7.3) or to the cited primary sources. Methods detail beyond the manuscript: `../PUBLICATION_METHODS.md`.
