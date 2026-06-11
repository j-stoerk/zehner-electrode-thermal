# Technical Guide

Consolidated reference: per-method details, method selection, validity limits, electrode module API.
(Merges the former Methods_README, PREDICTION_GUIDE and FAILURE_ENVELOPE documents.)

---

## 1. Which method should I use? (decision tree)

```
You want Lambda/lambda (or lambda_eff) for a porous bed/coating
│
├─ FORWARD problem (you have psi, kappa, d, Pe, T):
│   ├─ Quick estimate, any conditions        → Tier 0: Eq. 66 baseline   (±10-15%, instant)
│   ├─ Lab range, in-distribution, want UQ   → Tier 1: + ML GP residual  (±3-5%, ML.ipynb)
│   ├─ Industrial scale s > 1 m              → Tier 2: + Nu_S convection (Multiphysics.ipynb)
│   └─ Battery electrode/separator coating   → electrode_thermal.py      (Electrode.ipynb)
│
└─ INVERSE problem (you have measured profiles/conductivity):
    ├─ Profile has closed form               → scipy curve_fit            (seconds, exact)
    ├─ No closed form, forward solver exists → LSQ-shooting (solve_ivp + least_squares)
    ├─ No fast forward solver / need surrogate field / multi-run fusion → PINN (PINN.ipynb)
    └─ Electrode porosity from lambda_eff    → invert_porosity() in electrode_thermal.py
```

**Rule of thumb**: if a forward solver exists, LSQ-shooting beats the PINN on accuracy *and* is 2000–3700× faster (PINN.ipynb section 10). The PINN earns its cost only via structure: multi-run fusion in one training pass, trainable extra physics parameters, continuous differentiable surrogate field.

---

## 2. Validity / failure envelope

Figure: `figures/summary/method_validity_envelope.png`

| Method | Valid | Fails / caution | Evidence |
|---|---|---|---|
| **Eq. 66 baseline** | everywhere (±10–15%) | none known — it's the safe default | all 4 methods |
| **SR (K estimate)** | d/D 0.017–0.19 | K(d/D) trend narrowed (favors Eq. 12/70 over Agnew-Potter), not fully resolved | SR.ipynb |
| **ML GP + conformal UQ** | T 250–500 K; foam/glass/ceramic/steatite/steel; measured Pe range | **T>500 K: 0% improvement, PICP 51% vs 90% nominal. Copper: 14%, PICP 47%.** Re-calibrate before extrapolating | ML.ipynb extrapolation splits |
| **Multiphysics Nu_S** | s > 1 m and Ra*_S > ~39.5 | s < 0.5 m: convection dormant (Nu_S = 1 everywhere at lab scale); s > 5 m unvalidated (no co-located data) | Multiphysics.ipynb |
| **PINN inverse** | any PDE with trainable residual | needs per-parameter learning rates; multiplicative-parameter residual form (see §4); slow vs LSQ | PINN.ipynb §3–10 |
| **Electrode (Knudsen-ZBS)** | kappa 14–1005 (inside validated range); zero-fit ±30–40% worst case, cathodes 3–14%; **with per-recipe phi(Pi) calibration: 4.5% MAPE across full calendering range** | psi extrapolated below 0.369; point-contact ⇒ systematic *under*-prediction with binder bridges/calendering; separators morphology-dominated (see §5); wet separator kappa=2.2 below validated min | Electrode.ipynb §6 |

**Why ML fails at the edges**: leave-one-material-out shows the GP mostly learns a **per-material constant** (oxide-layer offset, Eq. 69) rather than a smooth function — so it cannot extrapolate to unseen materials/temperatures, and the conformal intervals honestly flag this.

---

## 3. Per-method reference

### 3.1 Symbolic Regression (`Symbolic Regression.ipynb`)
- **Test**: can PySR rediscover `Lambda/lambda = lambda_so/lambda + Pe/K` and estimate K(d/D)?
- **Key results**: unconstrained SR finds only `log_Pe` dominance (expected — Pe variance swamps lambda_so variance). Constrained residual fit: `residual ≈ Pe·0.1384 ⇒ K ≈ 7.23`. Per-series K_eff + digitized Abb. 62 widens d/D to 0.017–0.19; trend mildly positive (consistent with Eq. 12/70, inconsistent with Agnew-Potter).
- **Reuse**: add literature K(d/D) points to `data/raw/abb62_K_vs_dD.csv` and re-run.

### 3.2 ML gray-box GP + conformal UQ (`ML.ipynb`)
- **Test**: does a GP residual on Eq. 66 cut scatter while staying calibrated?
- **Key results**: in-distribution RMSE 4.98 → 1.66 (−66.7%), PICP 88.9% @ 90% nominal, both material groups improve together. Extrapolation: see envelope table above.
- **Reuse**: `gp_uq`, `scaler`, `sigma_r` feed the Multiphysics UQ propagation and the electrode model-form floor (~2%).

### 3.3 Multiphysics (`Multiphysics.ipynb`, `src/zbs_jax.py`)
- **Test**: closure → Carman-Kozeny → Ra*_S → Nu_S (VDI Fc) coupling + sensitivities + UQ.
- **Key results**: JAX closure exact vs NumPy (diff 0.0). Lab scale: Ra*_S ∈ [0.13, 3.1] ≪ 39.5 ⇒ Nu_S = 1. Industrial (s=10 m): Ra*_S ≈ 47 ⇒ Nu_S ≈ 1.0915 ± 0.0111 (delta-method with sigma_r ≈ 2%). Sobol: near onset, interactions dominate (s: ST=1.46; kappa, d almost pure interaction).
- **Reuse**: `lambda_so_over_lambda(psi, kappa)` is the differentiable building block everything else imports.

### 3.4 PINN inverse (`PINN.ipynb`)
- **Test**: recover Lambda (and alpha) from temperature profiles: 1D exponential → 1D nonlinear ODE (no closed form) → genuinely 2D dispersion PDE with rectangular-source BC.
- **Key results**: clean-data recovery 0.07–0.4% across all cases; under noise PINN degrades ~2× more than LSQ; 2D wall-clock 100–140 s vs 0.04–0.06 s for LSQ-shooting.
- **Hard-won practical findings** (reusable):
  1. **Scalar inverse variables in DeepXDE need their own much larger learning rate** (`{"params": [lam], "lr": 0.05}` vs net 1e-3) — otherwise they stay at the initial guess at near-zero loss.
  2. **Write the unknown as a multiplicative prefactor** (`lam·Pe·r·dTheta/dz − (...) = 0`), never `1/lam` — the inverse form has a degenerate lam→∞ direction in 2D (converged to 28.28 instead of 2.0).
  3. LSQ FD-Jacobian can silently stall at special initial values (lam_init=1.0 with eps_smooth=0.05) — perturb the init.

### 3.5 Electrode application (`Electrode.ipynb`, `src/electrode_thermal.py`)
See §4–5 below.

---

## 4. Electrode module API (`src/electrode_thermal.py`)

All functions JAX-differentiable, float64. Defaults: air at 300 K, 1 atm.

```python
from electrode_thermal import (
    lambda_eff_coating,      # (psi, d_p, lambda_s, lambda_f, gas_filled, ..., d_gas) -> W/mK
    dlambda_dpsi,            # exact autodiff sensitivity d(lambda_eff)/d(psi)
    invert_porosity,         # (lambda_meas, d_p, lambda_s, ...) -> psi_hat  (Newton, ~1e-16)
    porosity_uncertainty,    # first-order sigma_psi from meas. + model-form noise
    lambda_eff_coating_contact,  # + VDI contact term: phi, lambda_bridge (calendering-aware)
    lambda_gas_knudsen, knudsen_number, mean_free_path, pore_diameter,
    LAMBDA_AIR, LAMBDA_ELECTROLYTE, LAMBDA_HELIUM,          # 0.026 / 0.18 / 0.1518
    LAMBDA_GRAPHITE, LAMBDA_NMC, LAMBDA_SEPARATOR_PE,       # 25 / 2.5 / 0.40
    D_AIR, D_HELIUM,                                        # gas collision diameters
)

# Example: wet graphite anode at psi=0.30, d50=17 um
lam = lambda_eff_coating(0.30, 17e-6, 25.0, LAMBDA_ELECTROLYTE, gas_filled=False)  # ≈ 2.56

# Example: porosity QC from a measured conductivity
psi_hat  = invert_porosity(2.37, 17e-6, 25.0, LAMBDA_ELECTROLYTE, gas_filled=False)
sigma_psi = porosity_uncertainty(psi_hat, 17e-6, 25.0, LAMBDA_ELECTROLYTE, False,
                                 sigma_lambda_rel=0.03, sigma_model_rel=0.02)       # ≈ 0.009

# Example: calendering-aware closure (graphite anode, compression rate Pi)
Pi  = 0.30                                       # Pi = 1 - s_coating/s_coating,0
phi = max(0.0, 0.0094 - 0.024*Pi)                # phi(Pi), graphite_thin calibration
lam = lambda_eff_coating_contact(0.40, 18e-6, 24.8, LAMBDA_HELIUM, True,
                                 phi=phi, lambda_bridge=130.0, d_gas=D_HELIUM)
# phi=0 reduces EXACTLY to lambda_eff_coating (the validated point-contact ZBS)
```

Calibrated phi(Pi) parameters (Gandert 2023 data; per-recipe — do not transfer across additive recipes):

| family | lambda_s | phi0 | a | b | lambda_bridge |
|---|---|---|---|---|---|
| graphite_thin | 24.8 | 0.0094 | −0.024 | 0 | 130 |
| graphite_thick | 5.0 | 0.0121 | −0.042 | 0.053 | 130 |
| NMC622 | 1.6 | 0.0171 | −0.089 | 0.039 | 24 |
| NMC811 | 1.5 | 0.0048 | −0.049 | 0 | 24 |

Physics: `lambda_eff = ZBS(psi, lambda_s/lambda_f_eff) · lambda_f_eff` with
`lambda_f_eff = lambda_gas/(1 + 2·beta·Kn)` for gas-filled pores (Smoluchowski), `Kn = mfp/d_pore`,
`d_pore = (2/3)·psi/(1−psi)·d_p`. Liquid-filled pores: no Knudsen correction.

**Honesty checklist before quoting numbers**:
- psi < 0.369 extrapolates the validated porosity range (closure smooth but unverified there)
- point-contact (phi=0) variant ⇒ under-predicts when binder bridges / calender-flattened contacts exist
- graphite lambda_s is anisotropic (5–200 W/mK); default 25 is a literature mid-value
- literature stack measurements include interlayer contact resistance ⇒ they are *lower bounds* on coating conductivity

---

## 5. Electrode validation summary (what the data said)

Full details + figures: Electrode.ipynb §6; datasets with citations: `data/raw/gandert2023_calendering.csv`, `data/raw/separator_electrode_literature.csv`.

1. **Gandert et al. 2023** (KIT, open access; 4 electrode types × 27 calendering states, dry, He atmosphere):
   zero-fit predictions at the uncalendered points — NMC811 **+2.7%**, NMC622 −29%, graphite −36/−39%.
   The error *ordering* equals the binder/contact-bridging ordering; residual-vs-porosity curves are the
   calibration target for the Bauer contact term. Knudsen term in He: 2.8–5.5% (the very correction
   Gandert et al. flag as an unquantified gap). No porosity-only model reproduces their u-shape
   (calender-induced contact changes — they showed Oehler's and Sangrós' models fail the same way).
2. **Separators** (Richter 2017: dry 0.07–0.18 W/mK; Marconnet 2018: dry 0.10, wet 0.11): at real pore
   sizes (43–64 nm) Kn = 1.0–1.6 and pore-gas conduction collapses by 84%. Measured values fall between
   the point-contact lower bound (0.022–0.031) and the continuous-skeleton+Knudsen upper bound
   (0.080–0.155). **Continuum sphere-pack models hit the measured number only by cancelling two errors**
   (overestimated gas conduction vs. underestimated skeleton connectivity) — a two-gas-pressure
   measurement would separate the contributions.
3. **Soaked/dry ratios** (Burheim 2013: 3.0–3.1×; Marconnet: 2.4–2.8×): model 3.2–4.3× — slight
   overestimate, the expected point-contact signature. Absolute LCO cathode anchors within 12–14%.
4. **Calendering-aware closure (Electrode.ipynb §7)**: quadratic phi(Pi) contact term calibrated per family —
   all-27-sheet MAPE **31.1% → 4.5%** (noise floor), first model to reproduce the measured u-shape;
   transfer within composition 21%, across additive recipes 40% (calibrate per recipe, ~6 states);
   bridge share of lambda_eff 16–66% as-coated → inline delamination QC at z = 5–22.
5. **Design principles (§7.2)**: bridge conductance = particle plasticity × additive conductivity × additive
   *volume* fraction (anodes self-bridge at ~20× cathode effectiveness; +2 wt% flake graphite ≈ ×2 for NMC622
   vs NMC811); damage rate orders by binder mechanics (PVDF 5.2–10.2 /Pi vs elastomeric CMC/SBR 2.6–3.5);
   recovery is an interlocking *threshold* (SEM-evidenced Al-foil penetration), not a material constant.

Figures: `figures/electrode/` — knudsen_reduction, lambda_eff_vs_porosity, validation_gandert2023,
calibrated_contact_closure, inverse_porosity_qc (all regenerated on notebook execution).

---

## 6. Reproducibility

For manuscript-grade detail (dataset experimental provenance, exact hyperparameters, error analysis, lab protocol) see **PUBLICATION_METHODS.md**.

- `requirements.txt` / `environment.yml` — pinned versions (Python 3.11+, JAX 0.10.1, DeepXDE 1.15.0, PyTorch 2.12.0, …)
- Every notebook prints its package versions in cell 1
- Seeds: `dde.config.set_random_seed(0)` (PINN), `np.random.default_rng(42)` (electrode QC batch), `random_state=42` (ML splits)
- PINN caveat: section results depend on RNG state consumed by earlier sections — execute notebooks top-to-bottom, never cells in isolation
- Verify any notebook: `jupyter nbconvert --to notebook --execute --inplace <nb>.ipynb` then check for zero error outputs
