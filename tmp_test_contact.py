import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from electrode_thermal import (lambda_eff_coating, lambda_eff_coating_contact,
                               LAMBDA_HELIUM, D_HELIUM)

# 1) phi=0 must reduce to the validated model exactly
a = float(lambda_eff_coating(0.45, 18e-6, 80.0, LAMBDA_HELIUM, True, d_gas=D_HELIUM))
b = float(lambda_eff_coating_contact(0.45, 18e-6, 80.0, LAMBDA_HELIUM, True, phi=0.0, d_gas=D_HELIUM))
print(f"phi=0 reduction check: {a:.6f} vs {b:.6f}  diff={abs(a-b):.2e}")
assert abs(a-b) < 1e-12

# 2) feasibility: fit phi(Pi) = phi0 + a*Pi on Gandert graphite_thin coating data
gd = pd.read_csv("data/raw/gandert2023_calendering.csv")
LAM_CC = {"Cu": 400.0, "Al": 237.0}
s_stack = gd.s_co_um + gd.s_cc_um
gd["lambda_co"] = gd.s_co_um / (s_stack / gd.lambda_stack_W_mK - gd.s_cc_um / gd.collector.map(LAM_CC))
gd["Pi"] = 0.0
for sysname in gd.system.unique():
    m = gd.system == sysname
    s0 = gd.loc[m, "s_co_um"].iloc[0]
    gd.loc[m, "Pi"] = 1 - gd.loc[m, "s_co_um"] / s0

FAM = {"graphite_thin":  dict(d_p=18e-6, lam_s=80.0, lam_b=80.0),
       "graphite_thick": dict(d_p=18e-6, lam_s=80.0, lam_b=80.0),
       "NMC622":         dict(d_p=10e-6, lam_s=2.5,  lam_b=24.0),
       "NMC811":         dict(d_p=10e-6, lam_s=2.5,  lam_b=24.0)}

def model(psi, Pi, fam, phi0, slope):
    phi = phi0 + slope * Pi
    return float(lambda_eff_coating_contact(psi, fam["d_p"], fam["lam_s"], LAMBDA_HELIUM,
                                            True, phi=phi, lambda_bridge=fam["lam_b"], d_gas=D_HELIUM))

for sysname in ["graphite_thin", "NMC622"]:
    sub = gd[gd.system == sysname]
    fam = FAM[sysname]
    def res(p):
        return [model(r.porosity, r.Pi, fam, p[0], p[1]) - r.lambda_co for r in sub.itertuples()]
    fit = least_squares(res, x0=[0.005, 0.02], bounds=([0, 0], [0.08, 0.3]))
    mape = np.mean(np.abs(fit.fun / sub.lambda_co.values)) * 100
    print(f"{sysname}: phi0={fit.x[0]:.4f}, slope={fit.x[1]:.4f}, in-sample MAPE={mape:.1f}%")
