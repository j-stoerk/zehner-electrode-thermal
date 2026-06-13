"""Regenerate all manuscript figures as vector PDFs (publication styling).

Run from the repository root:
    .venv/Scripts/python.exe publication/make_figures.py
Outputs: publication/figures/fig1_knudsen.pdf ... fig6_qc.pdf
All numbers are recomputed from src/ + data/raw/ (no hard-coded results).
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from electrode_thermal import (
    mean_free_path, pore_diameter, knudsen_number, lambda_gas_knudsen,
    lambda_eff_coating, lambda_eff_coating_contact, invert_porosity,
    porosity_uncertainty,
    LAMBDA_AIR, LAMBDA_ELECTROLYTE, LAMBDA_HELIUM, D_HELIUM,
    LAMBDA_GRAPHITE, LAMBDA_NMC, LAMBDA_SEPARATOR_PE,
)

OUT = os.path.join("publication", "figures")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------- styling --
plt.rcParams.update({
    "font.size": 8, "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 6.8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "lines.linewidth": 1.2, "axes.linewidth": 0.6,
    "grid.linewidth": 0.4, "grid.alpha": 0.3,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
UNIT = r"$\lambda_{\mathrm{eff}}$ [W m$^{-1}$ K$^{-1}$]"

SYSTEMS = {
    "anode (graphite)":  dict(lam_s=LAMBDA_GRAPHITE,     d_p=17e-6,  psi_ref=0.30),
    "cathode (NMC)":     dict(lam_s=LAMBDA_NMC,          d_p=8e-6,   psi_ref=0.30),
    "separator (PE)":    dict(lam_s=LAMBDA_SEPARATOR_PE, d_p=0.3e-6, psi_ref=0.40),
}

# ------------------------------------------------------ data & calibration --
# Canonical layer (single source of truth): family registry, loader, calibration.
from electrode_data import FAMILIES, load_gandert, calibrate, lam_eff_contact
gd = load_gandert()
gd["lambda_co_meas"] = gd["lam_co_meas"]          # alias kept for plot code below
# `zs` is the representative zero-fit (phi=0) solid conductivity = ls_mid
FAM = {k: dict(v, zs=v["ls_mid"]) for k, v in FAMILIES.items()}
TITLES = {"graphite_thin": "graphite (thin)", "graphite_thick": "graphite (thick)",
          "NMC622": "NMC622", "NMC811": "NMC811"}

def lam_cal(psi, Pi, fam, lam_s, phi0, a, b):
    phi = max(0.0, phi0 + a * Pi + b * Pi**2)
    return float(lambda_eff_coating_contact(psi, fam["d_p"], lam_s, LAMBDA_HELIUM, True,
                                            phi=phi, lambda_bridge=fam["lam_b"], d_gas=D_HELIUM))

def mape_family(name, p):
    sub = gd[gd.system == name]
    fam = FAM[name]
    return np.mean([abs(lam_cal(r.porosity, r.Pi, fam, *p) - r.lambda_co_meas) / r.lambda_co_meas
                    for r in sub.itertuples()]) * 100

params_all = {name: calibrate(name, gd)[0] for name in FAM}
print("calibrated:", {k: np.round(v, 4).tolist() for k, v in params_all.items()})

# ------------------------------------------------------------------- Fig 1 --
d_pore_range = np.logspace(-8, -4.3, 200)
mfp_air = float(mean_free_path())
red_air = 1 / (1 + 2 * 1.64 * mfp_air / d_pore_range)
mfp_he = float(mean_free_path(d_gas=D_HELIUM))
red_he = 1 / (1 + 2 * 1.64 * mfp_he / d_pore_range)

fig, ax = plt.subplots(figsize=(3.4, 2.5))
ax.semilogx(d_pore_range * 1e6, red_air * 100, "-", color="C0", label="air (293 K, 1 atm)")
ax.semilogx(d_pore_range * 1e6, red_he * 100, "--", color="C2", label="helium")
for name, s in SYSTEMS.items():
    dp = float(pore_diameter(s["psi_ref"], s["d_p"]))
    ax.plot(dp * 1e6, 100 / (1 + 2 * 1.64 * mfp_air / dp), "o", ms=4.5,
            label=name.split(" ")[0])
ax.axvspan(0.043, 0.064, color="C3", alpha=0.15, lw=0)
ax.text(0.052, 72, "separator\npores", fontsize=6, color="C3", ha="center", va="bottom")
ax.set_xlabel(r"pore diameter $d_{\mathrm{pore}}$ [$\mu$m]")
ax.set_ylabel("gas conductivity retained [%]")
ax.grid(True, which="both")
ax.legend(loc="lower right", framealpha=0.9)
fig.savefig(os.path.join(OUT, "fig1_knudsen.pdf"))
plt.close(fig)

# ------------------------------------------------------------------- Fig 2 --
psi_grid = np.linspace(0.20, 0.55, 120)
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
for ax, (name, s) in zip(axes, SYSTEMS.items()):
    wet = [float(lambda_eff_coating(p, s["d_p"], s["lam_s"], LAMBDA_ELECTROLYTE, False)) for p in psi_grid]
    dkn = [float(lambda_eff_coating(p, s["d_p"], s["lam_s"], LAMBDA_AIR, True)) for p in psi_grid]
    dno = [float(lambda_eff_coating(p, s["d_p"], s["lam_s"], LAMBDA_AIR, False)) for p in psi_grid]
    ax.plot(psi_grid, wet, "-", color="C0", label="electrolyte-filled")
    ax.plot(psi_grid, dno, ":", color="C3", label="dry, continuum")
    ax.plot(psi_grid, dkn, "-", color="C3", label="dry, Knudsen")
    ax.axvspan(0.25, 0.35, color="green", alpha=0.08, lw=0)
    ax.set_xlabel(r"porosity $\varepsilon$")
    ax.grid(True)
for ax, lbl, name in zip(axes, "abc", SYSTEMS):
    ax.set_title(f"({lbl}) {name}")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
axes[0].set_ylabel(UNIT)
axes[0].legend(loc="upper right")
fig.savefig(os.path.join(OUT, "fig2_maps.pdf"))
plt.close(fig)

# ------------------------------------------------------------------- Fig 3 --
psi_plot = np.linspace(0.20, 0.62, 80)
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.1), sharex=False)
for ax, (name, fam) in zip(axes, FAM.items()):
    sub = gd[gd.system == name]
    lo_s, hi_s = fam["ls_lo"], fam["ls_hi"]
    mid = [lam_cal(p, 0, fam, fam["zs"], 0, 0, 0) for p in psi_plot]
    lo = [lam_cal(p, 0, fam, lo_s, 0, 0, 0) for p in psi_plot]
    hi = [lam_cal(p, 0, fam, hi_s, 0, 0, 0) for p in psi_plot]
    ax.fill_between(psi_plot, lo, hi, alpha=0.15, color="C0", lw=0)
    ax.plot(psi_plot, mid, "-", color="C0", label=r"ZBS+Kn, $\varphi=0$")
    ax.plot(sub.porosity, sub.lambda_co_meas, "ks", ms=3.2, label="measured")
    ax.invert_xaxis()
    ax.set_xlabel(r"$\varepsilon$")
    ax.grid(True)
for ax, lbl, name in zip(axes, "abcd", FAM):
    ax.set_title(f"({lbl}) {TITLES[name]}")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
axes[0].set_ylabel(UNIT)
axes[0].legend(loc="upper left")
fig.savefig(os.path.join(OUT, "fig3_zerofit.pdf"))
plt.close(fig)

# ------------------------------------------------------------------- Fig 4 --
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.1))
for ax, (name, fam) in zip(axes, FAM.items()):
    sub = gd[gd.system == name]
    psi0 = sub.porosity.iloc[0]
    psi_tr = np.linspace(sub.porosity.min() - 0.02, psi0, 80)
    Pi_tr = 1 - (1 - psi0) / (1 - psi_tr)
    p = params_all[name]
    cal = [lam_cal(ps, Pi, fam, *p) for ps, Pi in zip(psi_tr, Pi_tr)]
    zf = [lam_cal(ps, 0, fam, fam["zs"], 0, 0, 0) for ps in psi_tr]
    ax.plot(psi_tr, zf, ":", color="C0", label=r"zero-fit ($\varphi=0$)")
    ax.plot(psi_tr, cal, "-", color="C3", label=r"calibrated $\varphi(\Pi)$")
    ax.plot(sub.porosity, sub.lambda_co_meas, "ks", ms=3.2, label="measured")
    ax.invert_xaxis()
    ax.set_xlabel(r"$\varepsilon$")
    ax.grid(True)
for ax, lbl, name in zip(axes, "abcd", FAM):
    ax.set_title(f"({lbl}) {TITLES[name]}")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
axes[0].set_ylabel(UNIT)
axes[0].legend(loc="upper left")
fig.savefig(os.path.join(OUT, "fig4_calibrated.pdf"))
plt.close(fig)

# ------------------------------------------------------------------- Fig 5 --
sub = gd[gd.system == "graphite_thin"]
fam = FAM["graphite_thin"]
ls_grid = np.array([5, 10, 15, 20, 25, 30, 40, 60, 80, 100, 120, 139], float)
prof_mape, prof_phi0 = [], []
for ls in ls_grid:
    def res(p):
        return [(lam_cal(r.porosity, r.Pi, fam, ls, *p) - r.lambda_co_meas) / r.lambda_co_meas
                for r in sub.itertuples()]
    f = least_squares(res, x0=[0.008, -0.02, 0.1], bounds=([0, -0.2, 0], [0.08, 0.3, 0.8]))
    prof_mape.append(np.mean(np.abs(f.fun)) * 100)
    prof_phi0.append(f.x[0])

p622, p811 = params_all["NMC622"], params_all["NMC811"]
hybrids = [
    (r"622 cal." + "\n(transfer)",        [p622[0], p622[1], p622[2], p622[3]]),
    (r"+811 $\lambda_s$",                  [p811[0], p622[1], p622[2], p622[3]]),
    (r"+811 $\varphi_0$",                  [p622[0], p811[1], p622[2], p622[3]]),
    (r"+811 $(\varphi_0,a,b)$",            [p622[0], p811[1], p811[2], p811[3]]),
    ("811 self-fit",                       [p811[0], p811[1], p811[2], p811[3]]),
]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.45})
ax = axes[0]
ax.plot(ls_grid, prof_mape, "o-", color="C3", ms=3.5)
ax.axhline(5, color="gray", ls=":", lw=0.8)
ax.text(50, 3.4, r"$\approx$5% measurement scatter", fontsize=6.5, color="gray", va="top")
ax.set_xlabel(r"fixed $\lambda_s$ [W m$^{-1}$ K$^{-1}$]")
ax.set_ylabel("best achievable MAPE [%]")
ax.grid(True)
ax2 = ax.twinx()
ax2.plot(ls_grid, prof_phi0, "s--", color="C0", ms=3, alpha=0.6)
ax2.set_ylabel(r"compensating $\varphi_0$", color="C0")
ax2.tick_params(axis="y", colors="C0", labelsize=7)
ax = axes[1]
vals = [mape_family("NMC811", p) for _, p in hybrids]
ax.bar(range(len(vals)), vals, color=["C3", "C1", "C0", "C0", "C2"], width=0.65)
ax.set_xticks(range(len(vals)))
ax.set_xticklabels([h[0] for h in hybrids], rotation=16, ha="right", fontsize=6.5)
ax.set_ylabel("MAPE on NMC811 [%]")
ax.grid(True, axis="y")
fig.savefig(os.path.join(OUT, "fig5_error.pdf"))
plt.close(fig)

# ------------------------------------------------------------------- Fig 6 --
rng = np.random.default_rng(42)
anode = SYSTEMS["anode (graphite)"]
N = 200
psi_true = rng.normal(0.30, 0.012, N)
lam_true = np.array([float(lambda_eff_coating(p, anode["d_p"], anode["lam_s"],
                                              LAMBDA_ELECTROLYTE, False)) for p in psi_true])
lam_meas = lam_true * (1 + rng.normal(0, 0.03, N))
psi_hat = np.array([float(invert_porosity(l, anode["d_p"], anode["lam_s"],
                                          LAMBDA_ELECTROLYTE, False)) for l in lam_meas])
sig = float(porosity_uncertainty(0.30, anode["d_p"], anode["lam_s"], LAMBDA_ELECTROLYTE,
                                 False, 0.03, 0.02))
rmse = float(np.sqrt(np.mean((psi_hat - psi_true) ** 2)))

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
ax = axes[0]
ax.errorbar(psi_true, psi_hat, yerr=sig, fmt="o", ms=2.5, alpha=0.5,
            ecolor="lightgray", elinewidth=0.5)
lims = [psi_true.min() - 0.01, psi_true.max() + 0.01]
ax.plot(lims, lims, "k--", lw=0.8)
ax.axvspan(0.28, 0.32, color="green", alpha=0.07, lw=0)
ax.axhspan(0.28, 0.32, color="green", alpha=0.07, lw=0)
ax.set_xlabel(r"true porosity $\varepsilon$")
ax.set_ylabel(r"recovered porosity $\hat{\varepsilon}$")
ax.set_title("(a) parity, 200-sheet batch")
ax.grid(True)
ax = axes[1]
ax.hist(psi_hat - psi_true, bins=25, color="steelblue", alpha=0.85)
ax.axvline(0, color="k", lw=0.8)
ax.axvline(sig, color="r", ls="--", lw=0.8, label=rf"predicted $\pm\sigma_\varepsilon$ = {sig:.4f}")
ax.axvline(-sig, color="r", ls="--", lw=0.8)
ax.set_xlabel(r"$\hat{\varepsilon}-\varepsilon$")
ax.set_ylabel("count")
ax.set_title("(b) recovery error")
ax.legend()
ax.grid(True, axis="y")
fig.savefig(os.path.join(OUT, "fig6_qc.pdf"))
plt.close(fig)
print(f"QC: RMSE={rmse:.4f}, sigma_psi={sig:.4f}")

print("All figures written to", OUT)
for f in sorted(os.listdir(OUT)):
    print("  ", f, f"({os.path.getsize(os.path.join(OUT, f))//1024} KB)")
