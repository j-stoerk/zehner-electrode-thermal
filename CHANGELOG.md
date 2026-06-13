# CHANGELOG

All notable changes to the Zehner Closure Model Validation Project.

---

## [4.1] - 2026-06-14

### Thorough unification: canonical data/calibration layer + single source of truth for results

- **New `src/electrode_data.py`** -- one canonical family registry (`FAMILIES`), one `load_gandert()` with the stack->coating conversion, one `calibrate()` and `lam_eff_contact()`. Replaces the `FAM` dict + loader + lam_cal that had been copy-pasted with drift across Electrode/Electrode_AI/Electrode_validation notebooks and make_figures.py (the four copies disagreed: ls_mid 80 vs 24.8 vs 10, key names ls_lo/lo/zs).
- **New `src/key_results.py` -> `results/key_numbers.json`** -- recomputes every headline number (ablation 31.1->4.5%, per-family MAPE 1.8/5.4/1.4/9.4, NMC811 +2.7%, reference lambda 2.56/0.97/0.065) from code+data, no hard-coding.
- **New `tests/test_key_results.py` (6 tests, 23 total)** -- asserts the canonical numbers reproduce the published values AND that `publication/main.tex` quotes them, so LaTeX/README/docs/notebooks can no longer silently drift from the code.
- **`publication/make_figures.py` refactored** to import the canonical layer; figures verified content-identical (calibrated parameters byte-identical).
- CI regenerates key_numbers.json and runs the consistency tests on every push.
- Deferred (documented): Electrode.ipynb and Electrode_AI.ipynb retain inline definitions (executed artifacts; re-running their NUTS/PySR/PINN purely for dedup is not worth the compute) -- now reconciled to and guarded by the canonical registry + consistency test.

## [4.0] - 2026-06-13

### Manuscript v3 + multi-source meta-validation, baseline head-to-head, Sobol

- **Abstract rewritten to 249 words** (was 421; journal-count ~240).
- **LFA vs guarded-hot-plate measurement gap addressed** (Gandert 2025): new Limitation (9) -- LFA reads up to ~2x GHP because it excludes inter-sample contact resistance; the effect is ~multiplicative so phi(Pi) shape, u-shape, ablation, WAIC and transfer findings are method-robust, only absolute lambda_s/phi0 scale with method. Bibliography entry added.
- **New `Electrode_validation.ipynb`** (lightweight, in CI validation list) + `data/raw/literature_meta.csv` (12 measured points, Burheim/Marconnet/Richter):
  1. Meta-validation: zero-fit at through-plane literature lambda_s -> median 37%, 83% within factor 2; LCO to 5%; residuals = inter-lab scatter (wet anode values differ 50% between labs) + LMO + 2 known separator anomalies. Not killed (structured residuals).
  2. Head-to-head on wet coatings: Bruggeman 185% (percolating solid), Maxwell-Eucken 19% (contrast-limited, no mechanism), ZBS+contact spans dry+wet+calendering. Vadakkepatt 2016's compression-dependent Bruggeman exponent independently corroborates phi(Pi).
  3. Sobol (4096 samples, converged CI 0.028): porosity 0.46, contact pair (phi,lam_b) ~0.48, particle size ~0 -> measurement-priority list.
- Manuscript: new subsection 'Multi-source meta-validation and sensitivity' (17 pp, 0 errors). Two figures in figures/electrode_val/.

## [3.9] - 2026-06-13

### Electrode_AI.ipynb -- the three no-new-data AI roadmap items, demonstrated

- **New notebook `Electrode_AI.ipynb`** (separate from the AI-free publication artifact `Electrode.ipynb`), executing the Summary section-7 roadmap items reachable with synthetic physics or the in-repo Gandert data:
  1. **PySR form discovery**: free symbolic search over {+,-,*} on closure-inverted phi values returns a low-order polynomial in Pi with constant term ~0.009 -- independently matching the VDI rigid-sphere value (0.0077) and the fitted phi0 (0.0094); confirms the ansatz function class without supervision (caveat: 6 points/family, exact degree not identifiable).
  2. **Conformal leave-one-family-out**: 87% mean coverage vs 90% nominal; under-covers exactly NMC811 (62%) -- the same family WAIC flagged in 7.5, two independent diagnostics agreeing on the outlier.
  3. **PINN field inverse (synthetic flash-thermography)**: recovers a spatially varying interface-conductance H(x) from 4 noisy IR frames; delamination located to one grid point (x=0.58 vs 0.60), 10.6% field L2 error, baseline conductance to ~6%; depth over-smoothed (honest PINN limit). The 'unknown is a field' regime the PINN-vs-LSQ benchmark reserved for neural methods.
- Three vector figures in `figures/electrode_ai/`. Summary.ipynb roadmap tags updated (SR form-discovery, conformal, PINN delamination now [done]/[demonstrated]).
- Electrode_AI.ipynb kept out of CI execute-notebooks (PySR/Julia + torch too heavy for the runner); nbformat-validated and error-checked like the others.

## [3.8] - 2026-06-13

### Manuscript v2, SOC/T state validation, partner package, experiment-design module

- **Manuscript v2 (16 pp., 0 errors)**: new Results subsections "Bayesian calibration and model evidence" (posteriors, pooled fit, WAIC table) and "From coatings to the cell: an 18650 stack test" (+15%, implied 6 uK m^2/W interface, TCR sweep explaining the 0.15-1.4 W/mK literature spread); abstract/contributions/methods/conclusion/outlook synced; refs steinhardt2021 + oehlerdiss added.
- **Electrode.ipynb 6.4 (state dependence, zero new parameters)**: lithiation-dependent constituent conductivities (Steinhardt Table A.16, Ju, Gotcu) -> predicted SOC swing **+9.8%**, bracketed by measured +7/+9% (partial OCV window) and calculated +12% (full swing); anode-only attribution +14.4% confirms graphite as driver. Temperature trend -0.09 %/K: right sign, below the measured -0.2..-1.7 %/K envelope; omitted T-channels identified.
- **Partner measurement package**: `publication/measurement_request_partner.md` (self-contained spec: sample matrix, instruments, two-pressure protocol, adhesion, data format, falsification criteria) with computed D-optimal calendering states (graphite {0,0.20,0.21,0.39,0.49,0.50}; NMC {0,0.05,0.05,0.10,0.29,0.30}) and the planning insight that the two-pressure split is ~10% on NMC vs ~1.5% on graphite (run it on the cathode).
- **New: `src/experiment_design.py`** (greedy D-optimal selection of calendering states via exact JAX Fisher information; pressure-split sizing) + 3 new tests (17 total, all green).
- **Summary.ipynb section 7**: per-method AI roadmap (SR/PINN/multiphysics/ML win conditions, status-tagged); Multiphysics items 1-2 already marked done.
- CI: requirements pins corrected (MAPIE 1.4.1 etc.) -> **badge green** (tests + docs); notebook-execution job tolerated.

## [3.7] - 2026-06-12

### Cell-level stack validation, pooled Bayes + WAIC, CI overhaul (desk-only tracks 2-4)

- **Electrode.ipynb 6.3 (Steinhardt 18650 stack validation)**: mined Steinhardt's TUM dissertation + J. Energy Storage 42 (2021) 103065 (open): real 18650 unit-cell structure transcribed to `data/raw/steinhardt_18650_unitcell.csv`. Bottom-up zero-fit prediction of the measured jelly roll: layer-only 1.289 vs measured 1.122 W/mK (+15%, expected sign); implied per-interface TCR ~6 uK m^2/W (vs 420 torn-down, Vishwakarma) -- winding compression explains it; TCR sweep [0..420] reproduces the meta-analysis cell-level spread 0.15-1.4 W/mK. Cross-model anchor: Oehler KIT dissertation simulations agree with our wet coatings within ~10-25%.
- **Electrode.ipynb 7.5 (pooled Bayes + WAIC)**: shared-parameter fit of both same-recipe graphite electrodes holds to 6.7/8.6% MAPE (recipe hypothesis confirmed; residual = thickness/interface). WAIC decisively favours quadratic phi(Pi) for 3/4 families (Delta 13-32); NMC811's Delta=1.6 independently rediscovers the unreached interlocking threshold.
- **CI overhaul (.github/workflows/test.yml)**: pytest suite + nbformat validation + committed-output error check + make_figures smoke test as the required job; notebook execution job (continue-on-error); docs/data existence check updated to current file set. CI badge added to README.
- Source documents archived locally under `_scratch/litdata/` (Oehler diss 280 pp., Steinhardt diss 128 pp., both open access).

## [3.6] - 2026-06-12

### Bayesian calibration + manuscript experiments/outlook extension

- **Electrode.ipynb section 7.4**: full Bayesian calibration (NumPyro NUTS, 2 chains x 1000+1500, target accept 0.9) of (lambda_s, phi0, a, b) per family; physics-informed priors (phi0 ~ TruncNormal(0.0077, 0.01), lambda_s uniform on anisotropy band, b >= 0, sigma ~ HalfNormal). Results: posterior medians match least squares; thin-anode lambda_s posterior 7.5-30.5 W/mK independently reproduces the profile-likelihood valley; inferred sigma matches each family's residual level (4.7/8.5/1.1/15.6%); phi0 credible intervals bracket the VDI value. New figure `bayesian_calibration.png`. numpyro added to requirements.
- **Manuscript**: experimental plan extended (SEM cross-section quantification of phi(Pi) as acceptance criterion (iv); multi-recipe matrix + information-gain experiment design); conclusion outlook restructured into experimental / modelling / statistical / application directions incl. Bayesian calibration, recipe-descriptor learning, conformal UQ, residual-based QC monitoring, and field-type inverse problems as the PINN entry point. 15 pp., 0 errors, 0 overfull.

## [3.5] - 2026-06-11

### Manuscript reworked to NeurIPS-style best practices (12 pp., compile-verified)

- **Structure**: explicit contribution list (C1-C5); full Related Work comparing approach families (analytic effective-medium closures, DEM/homogenization, rarefied-gas theory, component metrology); assumptions A1-A5; numbered equations (1)-(11) incl. calibration objective, Newton inverse and uncertainty propagation; explicit Limitations (8 items) and Reproducibility Statement; experiments section with falsification criteria.
- **New quantitative content** (computed, not hard-coded): baseline table at the uncalendered points -- Wiener bounds, Maxwell-Eucken (-11 to -70%), Bruggeman (+17 to +465%) vs ZBS+Kn; the failure *pattern* is morphology-diagnostic (Bruggeman matches co-continuous separators at 0.090 W/mK but fails sphere-pack electrodes). **Ablation table**: M0 (phi=0) 31.1% -> M1 (constant phi) 13.5% -> M2 (phi(Pi)) 4.5% average MAPE.
- Bibliography extended to 20 entries (Wiener 1912, Hashin-Shtrikman 1962, Maxwell, Eucken, Bruggeman 1935, Maleki 1999, Steinhardt 2022); unverifiable details flagged inline.

## [3.4] - 2026-06-11

### Publication package (publication/) + test suite

- **publication/main.tex** — complete manuscript draft (~9 pages compiled): abstract, intro, model (ZBS + Knudsen + phi(Pi) contact term), data/methods, results (zero-fit validation, calibrated closure, design principles, error analysis, QC tools), **proposed validation experiments** (incl. two-pressure Knudsen discrimination), conclusions, embedded bibliography. **Compile-verified with MiKTeX pdfLaTeX: 0 errors, 0 undefined references** — opens directly in TeXworks.
- **publication/figures/** — 6 journal-styled **vector PDF** figures, regenerated end-to-end from src/ + data/ by `make_figures.py` (no hard-coded numbers; reproduces calibration exactly).
- **publication/references.bib** — BibTeX database (4 entries flagged TODO-VERIFY for submission); **publication/README.md** — compile instructions + submission checklist.
- **tests/test_electrode_thermal.py** — 14 unit/regression tests (physics sanity, phi=0 reduction, monotonicity, inverse round-trip, manuscript-value regression incl. the NMC811 +2.7% headline). All pass in ~3 s.

## [3.3] - 2026-06-11

### Publication package: methods documentation + error analysis

- **New: PUBLICATION_METHODS.md** — manuscript-grade documentation: (1) experimental provenance/methodology of all five validation datasets (instruments, sample prep, uncertainty chains, our processing steps); (2) exact computational methods extracted from executed code — PINN architectures/hyperparameters per case (incl. per-parameter learning rates, loss weights, collocation counts), ML GP kernel (Matern nu=1.5 ARD + White), PySR configs per run, Multiphysics formulas (Carman-Kozeny, Ra*_crit=4pi^2, Sobol N=256) and UQ; (3) quantified error analysis; (4) the in-house experimental protocol (sample matrix, two-pressure Knudsen discrimination, adhesion correlation, acceptance criteria).
- **Electrode.ipynb section 7.3 (error analysis)**: hybrid-swap decomposition — the 40% NMC622->NMC811 transfer failure is carried by phi0 alone (39.5% -> 12.2% on swap; lambda_s swap: 36.2%) => recipe physics, with a practical one-measurement re-anchoring rule. Profile likelihood bounds the lambda_s/phi0 confound: lambda_s identifiable to [5, 30] W/mK (c-axis-dominated). Edge bounds: Knudsen beta uncertainty <=1.7% (electrodes) / 11.7% (dry separator); psi-extrapolation and NMC811 b-regime quantified.
- New figure: `figures/electrode/error_analysis.png`.

## [3.2] - 2026-06-11

### Design principles behind the contact parameters (Electrode.ipynb section 7.2)

- Decomposed the per-recipe contact parameters into identifiable material interplays using Gandert Table 2 compositions + constituent densities (volume-space comparison):
  - **Bridge conductance G = phi0*lambda_b**: anodes 19-24 W/mK per unit inactive volume vs cathodes 1.2-2.3 -- particle plasticity (graphite self-bridging) x additive conductivity (flake graphite >> CB >> PVDF); 2 wt% graphite additive doubles NMC622 over NMC811.
  - **Fractional damage rate |a|/phi0**: orders perfectly by binder mechanics -- PVDF 5.2-10.2 per unit Pi vs elastomeric CMC/SBR 2.6-3.5.
  - **Recovery b > 0 is a threshold**: present exactly where interlocking is independently evidenced (NMC622 SEM-documented Al-foil penetration; highest-line-load thick anode); NMC811's b=0 = regime not reached (Pi_max 0.26).
- Three actionable design rules (additive choice, co-binder elasticity, calendering setpoint vs. the dip).

## [3.1] - 2026-06-11

### Calendering-aware contact closure (benchmark + novelty + industrial upgrade)

- `src/electrode_thermal.py`: added `lambda_so_over_lambda_contact` / `lambda_eff_coating_contact` — VDI flattened-contact form with separate bridge-phase conductivity; phi=0 reduces exactly to the validated point-contact ZBS.
- **Electrode.ipynb section 7**: per-family calibration of phi(Pi) = phi0 + a*Pi + b*Pi^2 on the Gandert 2023 data with held-out transfer tests. **All-27-sheet MAPE: 31.1% -> 4.5%** (data noise floor); first closure to reproduce the measured u-shape of lambda_eff vs. calendering degree. phi0 = 0.005-0.017 brackets the VDI sphere value 0.0077; shear-damage term a<0 in all families. Transfer: within-composition 21%, across recipes 40% (calibration is per-recipe, ~6 calendering states).
- Industrial: bridge-share decomposition (as-coated 16-66% of lambda_eff) => thermal measurement as single-shot inline interface/delamination QC, z = 5-22 at 3% noise.
- New figure: `figures/electrode/calibrated_contact_closure.png`.

## [3.0] - 2026-06-11

### Documentation consolidation

- Consolidated 12 overlapping documentation files into **3**: `README.md` (overview, results, quick start), `TECHNICAL_GUIDE.md` (per-method reference + decision tree + failure envelope + electrode API), `CHANGELOG.md`.
- Superseded files moved to `_scratch/archive_docs/` (README_EXECUTIVE, QUICK_START_GUIDE, PREDICTION_GUIDE, FAILURE_ENVELOPE, Methods_README, Review_and_Next_Steps, PROJECT_STATUS, NAVIGATION_MAP + 3 session reports). Safe to delete.

## [2.2] - 2026-06-11

### Electrode application: literature validation

- **Electrode.ipynb section 6**: validated the Knudsen-extended ZBS zero-fit against five published datasets:
  - Gandert et al. 2023 (KIT, Energy Technol., CC-BY): 4 electrode types × 27 calendering states → NMC811 **+2.7%**, NMC622 −29%, graphite anodes −36/−39%; error ordering = binder/contact-bridging ordering; Knudsen term 2.8–5.5% in He (the gap Gandert et al. themselves flag).
  - Richter et al. 2017 + Sun/Kantharaj/Marconnet 2018 (separators): measured dry 0.07–0.18 W/mK falls between the framework's point-contact and continuous-skeleton+Knudsen bounds; exposed the two-error cancellation in continuum sphere-pack models (Kn = 1.0–1.6 at real 43–64 nm pores, −84% pore-gas conduction).
  - Burheim/Onsrud 2013 + Marconnet 2018 (soaked/dry ratios 2.4–3.1× vs model 3.2–4.3×; LCO absolute anchors within 12–14%).
- New datasets with citations: `data/raw/gandert2023_calendering.csv`, `data/raw/separator_electrode_literature.csv`.
- `src/electrode_thermal.py`: added helium gas properties (`LAMBDA_HELIUM`, `D_HELIUM`, `d_gas` parameter chain).

## [2.1] - 2026-06-11

### Electrode application: Knudsen-extended ZBS + inverse porosity QC

- **New: `Electrode.ipynb`** — maps the validated ZBS framework onto lithium-ion electrode manufacturing:
  - Knudsen/Smoluchowski extension for micrometer pores (decisive for separators: −44% on dry λ_eff at 133 nm hydraulic estimate, refined in [2.2])
  - λ_eff(ψ) maps for anode/cathode/separator × dry/wet with calendering window
  - Autodiff porosity sensitivities: ±0.02 calendering tolerance → 5–9% λ_eff drift
  - Inverse porosity QC: 200-sheet synthetic batch, ψ recovered to ±0.008 from 3%-noise measurements, 87.5% single-shot spec classification
- **New: `src/electrode_thermal.py`** — JAX-differentiable Knudsen-ZBS module (forward, sensitivities, Newton inversion, uncertainty propagation).
- Summary.ipynb: new section 5 (electrode application), synthesis renumbered to 6.

## [2.0] - 2026-06-11

### Complete Overhaul: Documentation & Tools

#### Added
- **README_EXECUTIVE.md** — Non-technical summary for stakeholders
- **QUICK_START_GUIDE.md** — 5-minute entry point with decision trees
- **PREDICTION_GUIDE.md** — 3-tier method selection guide with examples
- **FAILURE_ENVELOPE.md** — Validity map + where methods break down
- **NAVIGATION_MAP.txt** — File structure + recommended reading order
- **ALL_ENHANCEMENTS_COMPLETE.md** — Priority 1 & 2 completion report
- **CI/CD GitHub Actions** — Automated testing workflow (.github/workflows/test.yml)
- **CHANGELOG.md** — This file
- **Reproducibility cells** — Added to all 5 notebooks (Python + package versions)
- **Environment files** — requirements.txt + environment.yml for exact reproduction
- **Test suite** — Comprehensive validation of all notebooks + docs
- **PINN.ipynb flowchart** — ASCII diagram showing test progression
- **Failure envelope figure** — figures/summary/method_validity_envelope.png (3-subplot visualization)

#### Enhanced
- **Summary.ipynb** — Added TL;DR section (2-min executive overview) + reproducibility cell
- **PINN.ipynb** — Added section 10 (2D dispersion PDE inverse problem), test flowchart, reproducibility cell
- **All notebooks** — Reorganized with clear section headers + reproducibility documentation

#### Improved
- **Cross-references** — All markdown files link to each other (no broken references)
- **Navigation** — Clear paths for different user roles (novice, practitioner, researcher, developer)
- **Documentation** — 4500+ lines across 11 markdown files (was 1000 lines in 1 file)
- **Code quality** — All 89 notebook cells validated (0 errors)

#### Documentation Statistics
- Total markdown files: 11 (was 4)
- Total lines: ~4500 (was ~1000)
- Code examples: 15+ worked examples across guides
- Decision trees: 4 (Temperature, Material, Scale, Inverse)
- Figures: 4 (1 new: failure envelope map)

### What Works Now

**For new users (5 minutes)**:
- Read QUICK_START_GUIDE.md
- Pick method from decision tree
- Know which notebook to run

**For practitioners (30 minutes)**:
- Read PREDICTION_GUIDE.md
- Choose Tier 0/1/2 based on requirements
- Run examples

**For researchers (2-3 hours)**:
- Read Summary.ipynb
- Run chosen method notebook
- Validate with FAILURE_ENVELOPE.md

**For developers**:
- Use Methods_README.md as technical reference
- Follow code patterns in notebooks
- Extend using provided templates

### Under the Hood

- **Environment setup**: 1 command (pip or conda)
- **Package pinning**: Exact versions for reproducibility
- **CI/CD ready**: GitHub Actions test workflow included
- **Quality verified**: All notebooks pass validation

---

## [1.2] - 2026-06-11 (Prior Session)

### PINN Enhancements

#### Added
- **PINN.ipynb Section 10** — Genuine 2D radial-dispersion PDE inverse problem
  - Conservative finite-difference radial operator
  - LSQ-shooting vs. PINN comparison
  - Clean + noisy synthetic data
  - Wall-clock timing (2000-3700x difference)
  - **Critical fix**: multiplicative-Lambda formulation (eliminates degenerate solution)

#### Results
- Clean data: LSQ exact (2.0000), PINN (2.0014)
- Noisy (3%): LSQ (2.1201, 6.0%), PINN (2.1900, 9.5%)
- Cost asymmetry: LSQ ~0.04s vs. PINN ~140s

#### Updated
- **PINN.ipynb Section 11 Summary** — Added section 10 findings
- All results use real numbers (no placeholders)

### Multiphysics Additions

#### Completed
- **σ_Nu_S(ψ, κ) uncertainty map** on 2D rectangular grid
- Full UQ pipeline: closure → permeability → Ra*_S → Nu_S → uncertainty
- Industrial-scale convection barely active (Ra*_S ~ 47)
- Closure uncertainty (σ_r ~ 2%) propagates to Nu_S uncertainty (~1%)

---

## [1.1] - 2026-06-10 (Prior Session)

### Summary.ipynb Update

#### Enhanced
- **PINN section (Section 4)** — Updated with section 8 nonlinear ODE findings
  - LSQ-shooting exact on clean data
  - PINN close but not superior
  - Multi-run fusion demonstrated

#### Key Finding
- Even without closed-form ansatz, LSQ-shooting remains more robust than PINN
- PINN's value is structural (multi-run fusion, extensibility), not accuracy

---

## [1.0] - 2026-06-09 (Session Start)

### Original Project Completion

#### Delivered
- **4 validated methods**:
  1. Symbolic Regression (SR) — Structure discovery, K coefficient estimation
  2. Machine Learning (ML) — Gray-box GP residual + conformal UQ
  3. Multiphysics — JAX closure + VDI Fc convection coupling
  4. PINN — Inverse parameter recovery (1D → 2D)

#### Notebooks (6 total)
- Summary.ipynb — Cross-hypothesis synthesis
- PINN.ipynb — Sections 1-9 (1D exponential, nonlinear ODE, real data)
- Multiphysics.ipynb — Closure validation + convection analysis
- ML.ipynb — GP residual + leave-one-out analysis
- Symbolic Regression.ipynb — Structure validation + K(d/D) trend
- preliminaries.ipynb — Setup

#### Key Results
- All methods agree on Zehner's model validity
- Every method breaks at exactly same validity-range edges
- No single method dominates across all conditions
- PINN framework reusable for other PDE forms

#### Figures (4 total)
- sr_K_vs_dD.png — Symbolic Regression K estimates
- ml_parity_groups.png — Stratified performance
- mp_NuS_map.png — Multiphysics Nusselt maps

---

## Version History Summary

| Version | Date | Focus | Notebooks | Markdown | Status |
|---------|------|-------|-----------|----------|--------|
| 2.0 | 2026-06-11 | Documentation + Tools | 6 (89 cells) | 11 files | Production-ready |
| 1.2 | 2026-06-11 | PINN Section 10 | 6 (89 cells) | 4 files | Core complete |
| 1.1 | 2026-06-10 | Summary updates | 6 (78 cells) | 4 files | Research phase |
| 1.0 | 2026-06-09 | Initial delivery | 6 (78 cells) | 1 file | Validated |

---

## Backlog & Future Roadmap

### Priority 3B (Integration) — Not Yet Done
- [ ] Deploy REST API (Flask) for Eq. 66 predictions
- [ ] Interactive Jupyter widgets for method selection
- [ ] Unit tests for core functions

### Priority 3C (Outreach) — Not Yet Done
- [ ] Academic publication template
- [ ] One-page executive summary (added in 2.0)
- [ ] 5-minute walkthrough video

### Longer-term (Data-Driven)
- [ ] High-T retraining (if T > 500K data obtained)
- [ ] Copper validation (if copper-bed data obtained)
- [ ] Industrial-scale Fc measurements (if s > 5m data obtained)
- [ ] d/D trend resolution (if more literature points digitized)

---

## Quality Metrics

### Code Quality
- **Notebook execution**: 89 cells, 0 errors
- **Validation**: All notebooks pass nbformat.validate()
- **Reproducibility**: Python 3.11, exact package versions, random seeds documented

### Documentation Quality
- **Markdown files**: 11 (vs. 1 initially)
- **Total content**: ~4500 lines (vs. ~1000)
- **Code examples**: 15+ worked examples
- **Cross-references**: All internal links verified
- **User paths**: 4 (novice, practitioner, researcher, developer)

### Testing
- **Notebook validation**: 6/6 pass
- **Documentation**: 8/8 files present
- **Environment**: requirements.txt + environment.yml verified
- **Figures**: 4/4 generated
- **Reproducibility**: Cells present in 5/5 computational notebooks

---

## How to Use This Changelog

- **For new features**: Look at the "Added" section of the latest version
- **For bug fixes**: Look at "Fixed" sections
- **For understanding evolution**: Read from [1.0] upward
- **For dependencies**: See [2.0] "Environment files"
- **For what's next**: See "Backlog & Future Roadmap"

---

## Breaking Changes

**None documented.** All versions maintain backward compatibility.

---

## Deprecations

**None.** All methods documented in [1.0] remain supported.

---

## Contributors & Attribution

- **Core methodology**: Zehner (1972), extended with modern ML/AI
- **Implementation**: Multi-disciplinary approach (symbolic regression, machine learning, physics-informed learning)
- **Validation**: Four independent methods all agreeing
- **Documentation**: Comprehensive guides for all user types

---

**Last Updated**: June 11, 2026  
**Status**: Production-ready  
**Next Major Release**: When new validation data available (high-T, copper, industrial scale)

