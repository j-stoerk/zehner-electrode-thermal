# Publication Methods & Supplementary Documentation

Manuscript-grade documentation: (1) experimental provenance of all validation data, (2) exact computational architectures and hyperparameters, (3) error analysis and edge cases, (4) the proposed in-house experimental protocol. All hyperparameters below were extracted from the executed notebook code, not from memory.

---

## 1. Experimental data provenance (validation datasets)

No in-house physical experiments were performed in this work; all validation uses published measurements whose experimental methodology is documented here in the detail needed to reproduce our processing. §4 specifies the in-house campaign that would complete the experimental story.

### 1.1 Gandert et al. 2023 — primary calendering dataset
*Energy Technology* 11, 2300259 (open access CC-BY; KIT). 4 electrode types × 6–8 calendering states.

**Electrode preparation**: single-side coated sheets, each family from one batch. Calendered on a GKL 200 two-roll laboratory calender (Saueressig; roll Ø 267 mm, width 400 mm, circumferential speed 1 m/min, roll temperature 50 °C). Dried in air at 50–60 °C overnight, stored in dry glovebox under inert gas (graphite: ambient/desiccator; NMC811: inert atmosphere throughout). Compositions (their Table 2): graphite anodes 96/1.25/1.25/1.5 wt% (AM/CMC/SBR/CB) on 10.3–11.0 µm Cu; NMC622 92/3/2/3 (AM/PVDF/graphite/CB) and NMC811 96/2/2 (AM/PVDF/CB) on 16.2–16.7 µm Al.

**Measurement chain** (λ = κ·ρ·c_p, their Eq. 1/3):
- Thermal diffusivity κ: laser-flash analysis (NETZSCH LFA), evaluated with the McMasters penetration model for porous samples (Proteus software); **helium purge atmosphere** — hence pore gas = He (λ_He = 0.1518 W/mK at 20 °C, the value the authors themselves use in their model comparison).
- Specific heat c_p: DSC (TA Instruments Q2000), ±0.05%, −40…60 °C at 20 K/min in N₂ (50 mL/min); coating scraped off the collector, 3 samples × 5 ramps.
- Solid density ρ: gas pycnometry (Quantachrome Ultrapyc 1200e), ±0.03%, N₂ at 25 °C, 3 samples.
- Thickness: micrometer screw (Mahr; flat anvil 33.18 mm², 5–10 N → ≤ 0.30 N/mm²), 5 samples/sheet × 10 readings. The dial-gauge alternative (Mitutoyo, ball tip, up to 21 N/mm² locally) reads up to 23% lower by compressing the sample and was rejected by the authors — we therefore use only the micrometer-screw-based porosities.
- Porosity: from areal mass, pycnometric solid density, and thickness; uncertainties per GUM (notably ±0.04–0.11 absolute for graphite porosities).

**Our processing** (`data/raw/gandert2023_calendering.csv`): per-sheet porosity and thicknesses transcribed from their Table 1; λ_stack values evaluated from their published quadratic fit functions (Figs. 2/5) at the Table-1 porosities. Stack → coating conversion by series resistance, R_co = s_stack/λ_stack − s_cc/λ_cc with λ_Cu = 400, λ_Al = 237 W/mK (the metal term is < 0.1% of the total; the coating/collector **contact resistance remains lumped** into the coating value — unavoidable, as the authors state). Compression rate per sheet: Π = 1 − s_co/s_co,0.

### 1.2 Richter, Kjelstrup, Vie & Burheim 2017 — separator anchors
*J. Power Sources* 359, 592–600. Through-plane steady-state measurements using the Burheim-group ex-situ stacked-sample method (apparatus established in *J. Power Sources* 195 (2010) 249: sample stack between steel cylinders with heat-flux metering, under controlled compaction pressure), dry and with electrolyte solvent, at several compaction pressures. Values used here (from the published abstract): dry separators **0.07 ± 0.01 to 0.18 ± 0.02 W/mK** (range across separator types and compaction); dry electrode coatings 0.13–0.61 W/mK. Used as a bounds test, not point calibration.

### 1.3 Sun, Kantharaj & Marconnet 2018 — dry/wet component table
TFAWS 2018 presentation (Purdue; NASA TFAWS public archive, read in full). Steady-state 1D copper-block reference-bar rig: Kapton film heater, water-cooled sink, embedded T-type thermocouples in both blocks; samples = multi-layer stacks from a commercial 26650 cell teardown (cathode ~15 layers, anode ~15, separator ~34). Dry and electrolyte-soaked. Values (slide 14): separator (ceramic-coated) 0.10 ± 0.01 dry / 0.11 ± 0.01 wet; CMS-graphite anode 0.57 ± 0.12 / 1.35 ± 0.49; LMO cathode 0.16 ± 0.06 / 0.45 ± 0.09 W/mK.

### 1.4 Burheim, Onsrud, Pharoah, Vullum-Bruer & Vie 2013 — soaked/dry ratios
224th ECS Meeting, Abstract 1190 (read in full). Same ex-situ method as 1.2. SLP50 graphite anode: dry 0.30 ± 0.01 → soaked 0.89 ± 0.04; LCO cathode: dry 0.36 ± 0.003 → soaked 1.10 ± 0.06 W/mK ("at least a factor of three" on soaking).

### 1.5 Vishwakarma et al. 2015 — interface-resistance context
*J. Power Sources* 300, 123–131 (read in full). Steady-state copper-block sandwich with extrapolated thermocouple arrays; cathode-side half-cell decomposition of a 26650 cell. Key numbers: LCO cathode material resistance 11 µK·m²/W; separator–cathode interface TCR 420 µK·m²/W ≈ **88% of total stack resistance**. Used as the quantitative caveat that all stack-derived literature values are *lower bounds* on single-coating conductivity.

**Caveat summary**: 1.1 pins λ(ψ, Π) per recipe (dry, He) but lumps the collector contact; 1.2–1.4 anchor separators and wet states but are stack measurements at partially unreported porosities; 1.5 quantifies why. The §4 campaign is designed to remove exactly these gaps.

---

## 2. Computational methods (exact architectures and hyperparameters)

### 2.1 ZBS closure (`src/zbs.py`, `src/zbs_jax.py`)
Zehner Eq. 37a/38 with deformation parameter B = C·((1−ψ)/ψ)^(10/9) (Eq. 42a), shape factor C = 1.25 (spheres). JAX implementation in float64 (`jax_enable_x64`); verified identical to the NumPy reference to 0.0 over ψ ∈ [0.369, 0.429], κ ∈ [4, 14094].

### 2.2 Knudsen extension (`src/electrode_thermal.py`)
λ_gas,eff = λ_gas/(1 + 2βKn), Kn = Λ_mfp/d_pore, Λ_mfp = k_B T/(√2 π d_gas² p) with d_air = 3.7×10⁻¹⁰ m, d_He = 2.18×10⁻¹⁰ m; β = 1.64 default (sensitivity in §3). Hydraulic pore size d_pore = (2/3)·ψ/(1−ψ)·d_p; for separators, measured pore diameters (43–64 nm, Celgard-type) are used directly by inverting this relation. Liquid-filled pores: no correction.

### 2.3 Contact-augmented closure
k_bed = 1 − √(1−ψ) + √(1−ψ)·[φ·κ_bridge + (1−φ)·λ_pm(ψ, κ)] (VDI flattened-contact form); φ(Π) = max(0, φ0 + a·Π + b·Π²); bridge conductivities: graphite families 130, NMC families 24 W/mK (carbon black). φ = 0 reduces exactly to 2.1+2.2 (regression-tested).

### 2.4 Calibration protocol
`scipy.optimize.least_squares` (trust-region reflective), **relative** residuals (model−meas)/meas; x0 = [λ_s_init, 0.008, −0.02, 0.1]; bounds: λ_s ∈ [5, 139] (graphite) / [1.5, 5] (NMC), φ0 ∈ [0, 0.08], a ∈ [−0.2, 0.3], b ∈ [0, 0.8]. Per-family fits (6–8 points, 4 parameters); honesty checks = held-out transfer (graphite_thin→thick; NMC622→811) + parameter plausibility + §3 identifiability profile.

### 2.5 Gray-box GP + conformal UQ (`ML.ipynb`)
450 rows; features [log_κ, ψ, d/D, log_Pe, Nu_r, T_K, ε] standardized (StandardScaler). Kernel: ConstantKernel(1.0, (1e-2,1e2)) × Matern(ARD length scales, bounds (1e-2,1e2), **ν = 1.5**) + WhiteKernel(0.01, (1e-5,1.0)); `GaussianProcessRegressor(normalize_y=False, n_restarts_optimizer=2, random_state=0)`. CV: KFold(5, shuffle, rs=0). Conformal: MAPIE split-conformal, 60/20/20 train/calibration/test (rs=0), 90% nominal.

### 2.6 Symbolic regression (`Symbolic Regression.ipynb`, PySR)
All runs: `model_selection="best"`, deterministic, serial. Synthetic recovery: niterations=40, populations=24, ops {+,−,×,÷,log,square}, maxsize=22. Real-data unconstrained: 60/30/26, same ops. **Constrained residual fit** (the K-estimate): niterations=60, populations=30, binary ops {+,−,×,÷} only, maxsize=18. Ablation holdouts: 30/24/26, random_state=0.

### 2.7 Multiphysics coupling (`Multiphysics.ipynb`)
Permeability: Carman-Kozeny K = (d²/150)·ψ³/(1−ψ)². Modified Rayleigh-Darcy: Ra*_S = gβΔT·s·K/(ν·α_eff) with ν = Pr·α_f, Pr_N₂ = 0.71. Onset: Ra*_crit = 4π² (Horton-Rogers-Lapwood, free boundaries); Nu_S = 1 below onset, √(Ra*/Ra*_crit) above. Representative lab point: ψ = 0.421, κ = 45.5, d = 9.6 mm, s = 204 mm (an actual Zehner test series row); industrial point s = 10 m. Sobol: SALib, sample N = 256 over (ψ, log₁₀κ, d, s) → 2560 model evaluations via `jax.vmap`. UQ: delta method, σ_NuS = |dNu_S/dλ_Sch|·λ_Sch·σ_r with σ_r = 0.0204 from 2.5.

### 2.8 PINN inverse problems (`PINN.ipynb`, DeepXDE 1.15.0 / PyTorch backend, seed `dde.config.set_random_seed(0)`)

| Case | Network | Collocation | Optimization | Inverse variables |
|---|---|---|---|---|
| 1D linear (§3–5) | FNN [1,24,24,24,1], tanh, Glorot uniform | num_domain=40 + 7 data anchors (PointSetBC) | Adam 8000 iters (net lr 1e-3, **Λ lr 5.0**) → L-BFGS | Λ = dde.Variable |
| Multi-run fusion (§5c) | FNN [2,32,32,32,1] | num_domain=80, both runs anchored | Adam 8000 (1e-3 / 5.0) → L-BFGS | shared Λ |
| 1D nonlinear (§8) | FNN [1,24,24,24,1] | num_domain=40 + anchors | Adam 8000 (net 1e-3, Λ 5.0, α 1e-3) → L-BFGS | Λ, α |
| 2D dispersion (§10) | FNN [2,24,24,24,1] | num_domain=2000, num_boundary=200, num_initial=200 + 60 anchors | Adam 6000 (net 1e-3, **Λ lr 0.05**) → L-BFGS; loss weights [1,10,1,1,10] (PDE, IC, axis BC, wall BC, data) | Λ (multiplicative form) |

2D residual: Λ·Pe·r·∂Θ/∂z − (∂Θ/∂r + r·∂²Θ/∂r²); IC Θ(r,0) = ½(1−tanh((r−0.3)/0.05)); Neumann at r=0,1. Synthetic truth: conservative cylindrical FD operator (41 radial nodes, L'Hôpital + ghost at axis, Neumann ghost at wall) marched by `solve_ivp` RK45 (rtol 1e-7, atol 1e-9, 21 z-stations). Data: 60 points, r~U(0,1), z~U(0.05,0.5) (`default_rng(0)`); noise 3% additive (2D) / 5% multiplicative ×5 realizations (1D). Classical baselines: `curve_fit` (1D linear); LSQ-shooting = solve_ivp + least_squares, x0=[Λ_init,α_init], bounds ([1,−1],[1e5,1]) (1D nonlinear) and x0=1.5, bounds [0.1,20] (2D).
Reproducibility pitfalls (documented in the notebook): scalar inverse variables need their own ~10³× larger learning rate; the unknown must enter the residual multiplicatively (the 1/Λ form has a degenerate Λ→∞ direction: converged to 28.28 vs true 2.0); LSQ FD-Jacobians can silently stall at special initial values.

### 2.9 Inverse QC tools
Porosity inversion: Newton iteration via `jax.lax.scan` (25 steps, exact autodiff Jacobian, clip ψ ∈ [0.05, 0.75]); round-trip exact to 5.6×10⁻¹⁷. Uncertainty: σ_ψ = √(σ_meas² + σ_model²)·λ_eff/|dλ_eff/dψ| (first order), σ_meas = 0.03, σ_model = 0.02 (the packed-bed GP floor). Batch demonstrator: 200 sheets, ψ ~ N(0.30, 0.012), `default_rng(42)`, spec window [0.28, 0.32].

---

## 3. Error analysis and edge cases (quantified; Electrode.ipynb §7.3)

**Transfer-failure decomposition (NMC622 → NMC811, hybrid swaps)**: full transfer 39.5% MAPE; swapping λ_s only → 36.2%; swapping **φ0 only → 12.2%**; full contact set → 11.9%; self-fit floor 9.4%. The cross-recipe failure is carried (~90% of the recoverable error) by the as-coated bridge conductance φ0 — i.e., by the additive recipe (NMC622's 2 wt% flake graphite + 3% CB vs. NMC811's 2% CB), not by particle properties or damage dynamics. Mitigation: re-anchor φ0 with a single as-coated measurement when changing recipes; (a, b) transfer far better within a chemistry.

**Identifiability (profile likelihood, graphite_thin)**: with λ_s fixed on [5, 139] and (φ0, a, b) refit, MAPE stays within 2 points of optimum only for **λ_s ∈ [5, 30] W/mK** (φ0 compensating 0.0073–0.0120). The data constrain λ_s to the c-axis-dominated end of the anisotropy band — physically expected for through-plane transport across flakes — but individual values inside the valley must not be over-interpreted; the robust deliverables are the grouped quantities G = φ0·λ_b and |a|/φ0 plus the valley bounds as parameter uncertainty.

**Edge-case bounds**: Knudsen β ∈ [1.5, 2.0] → ≤1.7% on dry electrodes (negligible), 11.7% on the dry separator (resolved by the two-pressure experiment, §4). Porosity extrapolation: 6/27 Gandert sheets lie below the validated ψ_min = 0.369 (to 0.211); calibrated residuals there are indistinguishable from the in-range branch. NMC811's recovery term b is unconstrained beyond its tested Π = 0.26. All stack-derived values lump the coating/collector contact resistance (cf. 1.5). Framework-level edges (T > 500 K, copper, s > 5 m convection) are documented in TECHNICAL_GUIDE §2.

---

## 4. Proposed in-house experimental protocol (per electrode recipe)

**Objective**: convert the transferred/calibrated model into an in-house validated one by pinning (C, φ(Π), λ_s) independently and separating Knudsen from skeleton conduction.

**Sample matrix** (~26 sheets/recipe): 6 calendering states (Π = 0 … production maximum, evenly spaced; same batch, single-side coating) × [dry, electrolyte-soaked] + 2 extra as-coated sheets for the φ0 re-anchor across recipes.

**Measurements per sheet**:
1. **Porosity ground truth**: areal mass (analytical balance) + thickness (flat-anvil micrometer ≤ 0.3 N/mm² contact pressure — *not* a ball-tip dial gauge, cf. 1.1) + He-pycnometric solid density. GUM uncertainty budget.
2. **λ_eff through-plane**: LFA with penetration-model evaluation (state purge gas!) or guarded hot plate, 20 °C; n = 4 repeats (σ/√n brings 3% single-shot noise to ~1.5%).
3. **Two-pressure discrimination** (dry sheets; mandatory for separator, recommended for the most porous electrode state): repeat λ_eff at ~1000 mbar and ≤ 100 mbar chamber pressure. Knudsen conduction scales with pressure (Kn ∝ 1/p); skeleton/bridge conduction does not → the difference isolates the pore-gas term and pins β.
4. **Adhesion pull-off strength** per calendering state, to correlate directly with the fitted φ(Π) (testing the §7.2 mechanism, not just the fit).

**Analysis & acceptance**: refit per §2.4; accept if (i) residuals ≤ measurement uncertainty across all Π, (ii) the two-pressure split matches the Knudsen prediction within its β band, (iii) φ(Π) correlates with adhesion (sign and dip location). Report grouped quantities (G, |a|/φ0, b-threshold) with profile-likelihood bounds. Expected effort: one lab afternoon of calendering + ~2 days of measurements per recipe.

---

*All numbers in this document trace to executed notebook output (Electrode.ipynb §§2–7.3, PINN.ipynb, ML.ipynb, Multiphysics.ipynb, Symbolic Regression.ipynb) or to the cited primary sources read in full.*
