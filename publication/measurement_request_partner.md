# Measurement Request: Thermal Validation Campaign for a Calendering-Aware Conductivity Closure

**Purpose of this document**: a self-contained specification that a partner laboratory can execute without further context. Background and model: see the accompanying manuscript (`main.pdf`) and repository (https://github.com/j-stoerk/zehner-electrode-thermal). Contact: Julius Störk, VARTA Microbattery GmbH.

---

## 1. Objective

Validate three mechanisms of a thermal-conductivity closure for calendered electrode coatings, each with a stated falsification criterion:

| # | Mechanism | Test | Falsified if |
|---|---|---|---|
| M-A | Knudsen (rarefied-gas) pore conduction | two-pressure measurement | pressure split deviates from prediction beyond the β-band |
| M-B | Calendering-dependent contact/bridge fraction φ(Π) | conductivity across calendering states + adhesion correlation | φ(Π) and adhesion disagree in sign structure or dip location |
| M-C | Parameter transfer between recipes via single re-anchor | as-coated measurement of a second recipe | re-anchored prediction misses by ≫ measurement uncertainty |

## 2. Sample matrix (per recipe)

- **One coating batch**, single-side coated, on production-representative foil.
- **Six calendering states**: Π = 0 (uncalendered) to the production maximum. *Information-optimal values (greedy D-optimal, `src/experiment_design.py`): graphite-class recipes ≈ {0, 0.20, 0.21, 0.39, 0.49, 0.50}; NMC-class ≈ {0, 0.05, 0.05, 0.10, 0.29, 0.30} — i.e. the anchor, a replicated pair in the damage region, and a replicated pair at maximum compression. Even spacing is an acceptable fallback.*
- **Duplicates**: 2 extra sheets at Π = 0 (re-anchor reference and spare).
- **States per sheet**: dry and electrolyte-solvent-soaked (see §4.4).
- Total: ≈ 26 sheets ≙ one calendering afternoon + ~2 measurement days.

## 3. Per-sheet characterization (porosity ground truth)

1. **Areal mass**: analytical balance, coating mass via foil-tare or post-measurement stripping.
2. **Thickness**: flat-anvil micrometer at contact pressure **≤ 0.3 N/mm²** (e.g., Mahr micrometer screw). *Do not use ball-tip dial gauges* — they locally compress the coating and read up to 23% low (Gandert et al. 2023).
3. **Solid (skeletal) density**: helium pycnometry on scraped coating material (≥ 3 samples).
4. Porosity from 1–3 with a GUM uncertainty budget; report Π = 1 − s_co/s_co,0 per sheet.

## 4. Thermal conductivity measurement

### 4.1 Method
Through-plane laser-flash analysis (LFA) with penetration-model evaluation for porous samples (McMasters model; standard in NETZSCH Proteus), **or** a guarded-hot-plate/steady-state reference-bar method. 20 °C reference temperature.

### 4.2 Repeats and noise
n = 4 repeats per sheet and state (brings ~3% single-shot reproducibility to ~1.5%). Report individual repeats, not only means.

### 4.3 Critical: record the chamber/purge gas
For dry porous samples **the purge gas is the pore gas** and enters the result via rarefied-gas conduction (helium ≈ 3× the effect of air/nitrogen at electrode pore sizes). Without the gas identity the dry values cannot be modeled.

### 4.4 Wet state
Soak in electrolyte **solvent** (EC/DMC class, no conducting salt — avoids HF handling; thermally equivalent within ~5%): immerse/vacuum-infiltrate, verify full saturation by mass uptake vs. pore volume, blot, and measure in a **sealed** liquid sample holder (solvent evaporation otherwise drifts the result and contaminates the instrument).

### 4.5 Two-pressure discrimination (mechanism M-A; mandatory for separators, recommended for the most porous electrode state)
Repeat the **dry** measurement at ~1000 mbar and at ≤ 100 mbar chamber pressure on the same sample. Pore-gas conduction scales with pressure (Kn ∝ 1/p); solid-skeleton and bridge conduction do not. Report both values with chamber pressure and gas. *Predicted split in air (as-coated): ≈ 10% for NMC-class electrodes (easily measurable) but only ≈ 1.5% for graphite-class (marginal against noise) — if electrode beam time is limited, run the two-pressure test on the cathode.*

## 5. Adhesion (mechanism M-B)

Pull-off adhesion strength (e.g., 90°-peel or stamp pull-off, partner's standard method) at **every** calendering state, n ≥ 3. The model predicts the conductivity-relevant contact fraction and the adhesion share sign structure and dip location across Π.

## 6. Optional add-ons (high value, partner's discretion)

- **Ion-milled SEM cross-sections** at 3 calendering states (Π = 0, dip region, maximum): enables image-based quantification of the contact/bridge fraction as an independent observable.
- A second recipe varying **one** factor (binder system or conductive-additive content) — activates the transfer test M-C and starts the recipe→parameter map.

## 7. Data return format

One CSV per recipe, one row per (sheet, state, repeat):

```
recipe_id, sheet_id, Pi, porosity, porosity_unc, thickness_um, areal_mass_mg_cm2,
state(dry|wet), gas(air|N2|He|...), pressure_mbar, T_C, lambda_W_mK, method, repeat_no,
adhesion_N_mm2 (per sheet), notes
```

Raw LFA shots/thermograms appreciated but not required.

## 8. Acceptance criteria (what "success" means)

(i) Calibrated model residuals ≤ measurement uncertainty at all Π; (ii) two-pressure split consistent with the Knudsen prediction within the β ∈ [1.5, 2.0] band; (iii) φ(Π)–adhesion correlation as predicted; (iv) where SEM is available, image-derived contact fraction consistent with the thermally fitted one. Analysis is performed with the open repository code; the partner receives the full calibration report.
