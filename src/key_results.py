"""Single source of truth for the manuscript's headline numbers.

Recomputes every quoted result from the canonical closure + data (no notebook
execution, no hard-coded answers) and writes results/key_numbers.json. A test
(tests/test_key_results.py) asserts the manuscript matches this file, so the
LaTeX, README, docs and notebooks cannot silently drift from the code.

Run:  .venv/Scripts/python.exe src/key_results.py
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from electrode_data import FAMILIES, load_gandert, calibrate, zero_fit_mape
from electrode_thermal import (lambda_eff_coating, LAMBDA_AIR, LAMBDA_ELECTROLYTE,
                               LAMBDA_HELIUM, D_HELIUM, LAMBDA_GRAPHITE, LAMBDA_NMC,
                               LAMBDA_SEPARATOR_PE)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compute():
    gd = load_gandert()
    R = {}

    # per-family calibrated and zero-fit MAPE; the ablation endpoints
    cal, z0 = {}, {}
    for fam in FAMILIES:
        theta, mape = calibrate(fam, gd)
        cal[fam] = dict(theta=[round(float(x), 4) for x in theta], mape=round(mape, 1))
        z0[fam] = round(zero_fit_mape(fam, gd), 1)
    R["calibrated"] = cal
    R["zero_fit_mape"] = z0
    R["mape_zero_fit_avg"] = round(np.mean(list(z0.values())), 1)
    R["mape_calibrated_avg"] = round(np.mean([v["mape"] for v in cal.values()]), 1)

    # NMC811 uncalendered zero-fit headline (+2.7%)
    sub = gd[gd.system == "NMC811"].sort_values("porosity", ascending=False).iloc[0]
    pred = float(lambda_eff_coating(sub.porosity, 10e-6, FAMILIES["NMC811"]["ls_mid"],
                                    LAMBDA_HELIUM, True, d_gas=D_HELIUM))
    R["nmc811_uncal_err_pct"] = round((pred - sub.lam_co_meas) / sub.lam_co_meas * 100, 1)

    # reference coating/separator values (W/mK) at the documented reference points
    R["reference_values"] = dict(
        anode_wet=round(float(lambda_eff_coating(0.30, 17e-6, LAMBDA_GRAPHITE, LAMBDA_ELECTROLYTE, False)), 2),
        cathode_wet=round(float(lambda_eff_coating(0.30, 8e-6, LAMBDA_NMC, LAMBDA_ELECTROLYTE, False)), 2),
        separator_dry_knudsen=round(float(lambda_eff_coating(0.40, 0.3e-6, LAMBDA_SEPARATOR_PE, LAMBDA_AIR, True)), 3),
        separator_dry_continuum=round(float(lambda_eff_coating(0.40, 0.3e-6, LAMBDA_SEPARATOR_PE, LAMBDA_AIR, False)), 3),
    )

    # Knudsen pore-gas reduction at separator reference (the -84%)
    from electrode_thermal import lambda_gas_knudsen
    lam_g = float(lambda_gas_knudsen(0.40, 0.3e-6))
    R["separator_knudsen_reduction_pct"] = round((1 - lam_g / LAMBDA_AIR) * 100, 0)

    return R


def main():
    R = compute()
    os.makedirs(os.path.join(_ROOT, "results"), exist_ok=True)
    out = os.path.join(_ROOT, "results", "key_numbers.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2)
    print("wrote", out)
    print(json.dumps(R, indent=2))


if __name__ == "__main__":
    main()
