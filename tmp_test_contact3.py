import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from electrode_thermal import lambda_eff_coating_contact, LAMBDA_HELIUM, D_HELIUM

gd = pd.read_csv("data/raw/gandert2023_calendering.csv")
LAM_CC = {"Cu": 400.0, "Al": 237.0}
s_stack = gd.s_co_um + gd.s_cc_um
gd["lambda_co"] = gd.s_co_um / (s_stack / gd.lambda_stack_W_mK - gd.s_cc_um / gd.collector.map(LAM_CC))
gd["Pi"] = 0.0
for sysname in gd.system.unique():
    m = gd.system == sysname
    gd.loc[m, "Pi"] = 1 - gd.loc[m, "s_co_um"] / gd.loc[m, "s_co_um"].iloc[0]

FAM = {"graphite_thin":  dict(d_p=18e-6, lam_b=130.0, ls_lo=5.0, ls_hi=139.0, ls_init=10.0),
       "graphite_thick": dict(d_p=18e-6, lam_b=130.0, ls_lo=5.0, ls_hi=139.0, ls_init=10.0),
       "NMC622":         dict(d_p=10e-6, lam_b=24.0,  ls_lo=1.5, ls_hi=5.0,   ls_init=2.5),
       "NMC811":         dict(d_p=10e-6, lam_b=24.0,  ls_lo=1.5, ls_hi=5.0,   ls_init=2.5)}

def model(psi, Pi, fam, lam_s, phi0, a, b):
    phi = max(0.0, phi0 + a*Pi + b*Pi**2)   # adhesion dip-then-rise (Gandert et al.)
    return float(lambda_eff_coating_contact(psi, fam["d_p"], lam_s, LAMBDA_HELIUM, True,
                                            phi=phi, lambda_bridge=fam["lam_b"], d_gas=D_HELIUM))

def fit_family(sysname):
    sub = gd[gd.system == sysname]
    fam = FAM[sysname]
    def res(p):
        return [(model(r.porosity, r.Pi, fam, *p) - r.lambda_co) / r.lambda_co for r in sub.itertuples()]
    fit = least_squares(res, x0=[fam["ls_init"], 0.008, -0.02, 0.1],
                        bounds=([fam["ls_lo"], 0, -0.2, 0], [fam["ls_hi"], 0.08, 0.3, 0.8]))
    return fit.x, np.mean(np.abs(fit.fun))*100

def eval_family(sysname, params):
    sub = gd[gd.system == sysname]
    fam = FAM[sysname]
    errs = [(model(r.porosity, r.Pi, fam, *params) - r.lambda_co)/r.lambda_co for r in sub.itertuples()]
    return np.mean(np.abs(errs))*100, [f"{e*100:+.0f}%" for e in errs]

print("=== quadratic phi(Pi): fit -> transfer ===")
for cal, hold in [("graphite_thin","graphite_thick"), ("NMC622","NMC811")]:
    p, mi = fit_family(cal)
    mo, errs = eval_family(hold, p)
    print(f"fit {cal:15}: lam_s={p[0]:6.2f} phi0={p[1]:.5f} a={p[2]:+.4f} b={p[3]:.4f} | in {mi:5.1f}% | -> {hold}: {mo:5.1f}%  {errs}")
print("\n=== self-fits (in-sample benchmark) ===")
for sysname in FAM:
    p, mi = fit_family(sysname)
    _, errs = eval_family(sysname, p)
    print(f"{sysname:15}: lam_s={p[0]:6.2f} phi0={p[1]:.5f} a={p[2]:+.4f} b={p[3]:.4f} MAPE={mi:5.1f}%  {errs}")
