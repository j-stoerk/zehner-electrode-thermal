"""Canonical data, family registry, and calibration layer for the electrode work.

Single source of truth replacing the FAM dict / stack->coating conversion /
lam_cal helper that had been copy-pasted (with drift) across Electrode.ipynb,
Electrode_AI.ipynb, Electrode_validation.ipynb and publication/make_figures.py.

Every notebook and the figure pipeline should import from here:

    from electrode_data import FAMILIES, load_gandert, calibrate, lam_eff_contact
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from electrode_thermal import lambda_eff_coating_contact, LAMBDA_HELIUM, D_HELIUM

# repo root, so the loaders work regardless of the caller's working directory
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Canonical electrode-family registry.
#   d_p     particle diameter [m]
#   lam_b   bridge-phase conductivity [W/mK] (flake graphite / carbon black)
#   ls_lo/ls_hi  physical bound on through-plane solid conductivity (anisotropy band)
#   ls_init      least-squares start value
#   ls_mid       representative zero-fit (phi=0) solid conductivity used in maps
#   collector    current-collector metal
# ---------------------------------------------------------------------------
FAMILIES = {
    "graphite_thin":  dict(d_p=18e-6, lam_b=130.0, ls_lo=5.0, ls_hi=139.0, ls_init=10.0, ls_mid=80.0, collector="Cu"),
    "graphite_thick": dict(d_p=18e-6, lam_b=130.0, ls_lo=5.0, ls_hi=139.0, ls_init=10.0, ls_mid=80.0, collector="Cu"),
    "NMC622":         dict(d_p=10e-6, lam_b=24.0,  ls_lo=1.5, ls_hi=5.0,   ls_init=2.5,  ls_mid=2.5,  collector="Al"),
    "NMC811":         dict(d_p=10e-6, lam_b=24.0,  ls_lo=1.5, ls_hi=5.0,   ls_init=2.5,  ls_mid=2.5,  collector="Al"),
}
LAMBDA_COLLECTOR = {"Cu": 400.0, "Al": 237.0}
TRANSFER_PAIRS = [("graphite_thin", "graphite_thick"), ("NMC622", "NMC811")]


def load_gandert():
    """Gandert 2023 calendering data with coating conductivity and compression rate.

    Adds per-sheet columns:
      lam_co_meas  coating conductivity, foil removed in series (Eq. stack)
      Pi           compression rate 1 - s_co/s_co,0
    """
    gd = pd.read_csv(os.path.join(_ROOT, "data", "raw", "gandert2023_calendering.csv"))
    s_stack = gd.s_co_um + gd.s_cc_um
    gd["lam_co_meas"] = gd.s_co_um / (
        s_stack / gd.lambda_stack_W_mK - gd.s_cc_um / gd.collector.map(LAMBDA_COLLECTOR)
    )
    gd["Pi"] = 0.0
    for sysname in gd.system.unique():
        m = gd.system == sysname
        gd.loc[m, "Pi"] = 1 - gd.loc[m, "s_co_um"] / gd.loc[m, "s_co_um"].iloc[0]
    return gd


def lam_eff_contact(family, psi, Pi, theta):
    """Calibrated contact-closure prediction for a family at (psi, Pi).

    theta = (lam_s, phi0, a, b); dry, helium pore gas (Gandert measurement state).
    """
    lam_s, phi0, a, b = theta
    fam = FAMILIES[family]
    phi = max(0.0, phi0 + a * Pi + b * Pi**2)
    return float(lambda_eff_coating_contact(psi, fam["d_p"], lam_s, LAMBDA_HELIUM, True,
                                            phi=phi, lambda_bridge=fam["lam_b"], d_gas=D_HELIUM))


def calibrate(family, gd=None):
    """Least-squares calibration of theta=(lam_s, phi0, a, b) for one family.

    Returns (theta, mape_percent). Mirrors the protocol used throughout the
    notebooks (relative residuals, trust-region, physical bounds).
    """
    if gd is None:
        gd = load_gandert()
    sub = gd[gd.system == family]
    fam = FAMILIES[family]
    x0 = [fam["ls_init"], 0.008, -0.02, 0.1]
    bounds = ([fam["ls_lo"], 0, -0.2, 0], [fam["ls_hi"], 0.08, 0.3, 0.8])

    def resid(p):
        return [(lam_eff_contact(family, r.porosity, r.Pi, p) - r.lam_co_meas) / r.lam_co_meas
                for r in sub.itertuples()]

    fit = least_squares(resid, x0=x0, bounds=bounds)
    mape = float(np.mean(np.abs(fit.fun)) * 100)
    return fit.x, mape


def zero_fit_mape(family, gd=None):
    """MAPE of the phi=0 closure at the family's representative ls_mid (the M0 baseline)."""
    if gd is None:
        gd = load_gandert()
    sub = gd[gd.system == family]
    theta0 = (FAMILIES[family]["ls_mid"], 0.0, 0.0, 0.0)
    errs = [abs(lam_eff_contact(family, r.porosity, r.Pi, theta0) - r.lam_co_meas) / r.lam_co_meas
            for r in sub.itertuples()]
    return float(np.mean(errs) * 100)
