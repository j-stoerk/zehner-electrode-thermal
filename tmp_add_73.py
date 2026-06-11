import sys
sys.stdout.reconfigure(encoding='utf-8')
import nbformat
from nbformat.v4 import new_markdown_cell, new_code_cell

nb = nbformat.read("Electrode.ipynb", as_version=4)

intro_md = new_markdown_cell(source="""##### 7.3 Error analysis: transfer-failure decomposition, identifiability, and edge-case bounds

Three quantitative questions a reviewer will (rightly) ask:

1. **Which parameter mismatch causes the 40% NMC622 -> NMC811 transfer failure?** Answered by *hybrid swaps*: apply NMC622's calibration to NMC811 but replace one parameter group at a time with NMC811's own value, and watch the MAPE collapse (or not). If the bridge parameters dominate, the failure is recipe physics (additive content); if lambda_s dominates, it would be a particle-property issue.
2. **How well-determined are the fitted parameters?** Answered by a *profile likelihood*: fix lambda_s on a grid spanning the full graphite anisotropy band, refit (phi0, a, b) at each value, and plot the resulting MAPE. A flat valley = the lambda_s/phi0 confound quantified; the *width* of the valley is the honest parameter uncertainty.
3. **How large are the remaining edge-case exposures?** Quantified bounds for: the Knudsen jump-coefficient uncertainty (beta in [1.5, 2.0]), the porosity extrapolation below the validated range, and the untested high-compression regime for NMC811."""
)

code_cell = new_code_cell(source='''# --- 7.3: transfer decomposition, identifiability profile, edge-case bounds ---

# (1) Hybrid-swap decomposition of the NMC622 -> NMC811 transfer failure
p622, p811 = params_all["NMC622"], params_all["NMC811"]
hybrids = {
    "full NMC622 params (the failing transfer)":      [p622[0], p622[1], p622[2], p622[3]],
    "+ NMC811 lambda_s only":                          [p811[0], p622[1], p622[2], p622[3]],
    "+ NMC811 phi0 only":                              [p622[0], p811[1], p622[2], p622[3]],
    "+ NMC811 contact set (phi0, a, b)":               [p622[0], p811[1], p811[2], p811[3]],
    "full NMC811 params (self-fit reference)":         [p811[0], p811[1], p811[2], p811[3]],
}
print("(1) NMC622 -> NMC811 transfer decomposition (MAPE on NMC811 sheets):")
for label, p in hybrids.items():
    print(f"    {mape_family('NMC811', p):5.1f}%  {label}")

# (2) Identifiability: profile likelihood over lambda_s for graphite_thin
sub = gd[gd.system == "graphite_thin"]
fam = FAM["graphite_thin"]
ls_grid = np.array([5, 10, 15, 20, 25, 30, 40, 60, 80, 100, 120, 139], dtype=float)
prof_mape, prof_phi0 = [], []
for ls in ls_grid:
    def res(p):
        return [(lam_cal(r.porosity, r.Pi, fam, ls, *p) - r.lambda_co_meas) / r.lambda_co_meas
                for r in sub.itertuples()]
    fit = least_squares(res, x0=[0.008, -0.02, 0.1],
                        bounds=([0, -0.2, 0], [0.08, 0.3, 0.8]))
    prof_mape.append(np.mean(np.abs(fit.fun)) * 100)
    prof_phi0.append(fit.x[0])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
ax = axes[0]
ax.plot(ls_grid, prof_mape, "o-", color="C3")
ax.axhline(5, color="gray", ls=":", lw=1)
ax.text(100, 5.3, "~5% measurement scatter", fontsize=8, color="gray")
ax.set_xlabel("fixed lambda_s [W/(m K)]"); ax.set_ylabel("best achievable MAPE [%]")
ax.set_title("Profile likelihood, graphite_thin:\\nthe lambda_s/phi0 identifiability valley")
ax.grid(alpha=0.3)
ax2 = ax.twinx()
ax2.plot(ls_grid, prof_phi0, "s--", color="C0", alpha=0.6)
ax2.set_ylabel("compensating phi0 [-]", color="C0")

ax = axes[1]
labels = ["622 params\\n(transfer)", "+811\\nlambda_s", "+811\\nphi0", "+811 contact\\n(phi0,a,b)", "811 self-fit"]
vals = [mape_family("NMC811", p) for p in hybrids.values()]
ax.bar(labels, vals, color=["C3", "C1", "C0", "C0", "C2"])
ax.set_ylabel("MAPE on NMC811 [%]")
ax.set_title("Transfer-failure decomposition:\\nwhich parameter group carries the 40%?")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("figures/electrode/error_analysis.png", dpi=150)
plt.show()

ident_range = (ls_grid[np.array(prof_mape) <= min(prof_mape) + 2.0])
print(f"\\n(2) Identifiability: MAPE stays within 2 pts of optimum for lambda_s in "
      f"[{ident_range.min():.0f}, {ident_range.max():.0f}] W/mK "
      f"(phi0 compensates from {max(prof_phi0):.4f} down to {min(prof_phi0):.4f})")

# (3) Edge-case bounds
print("\\n(3) Edge-case exposure bounds:")
# (3a) Knudsen beta uncertainty
for name, s in SYSTEMS.items():
    lo = float(lambda_eff_coating(s["psi_ref"], s["d_p"], s["lam_s"], LAMBDA_AIR, True, beta=1.5))
    hi = float(lambda_eff_coating(s["psi_ref"], s["d_p"], s["lam_s"], LAMBDA_AIR, True, beta=2.0))
    print(f"    beta in [1.5, 2.0] -> dry {name:22}: lambda_eff {hi:.4f}-{lo:.4f} W/mK "
          f"(spread {abs(lo-hi)/((lo+hi)/2)*100:.1f}%)")
# (3b) psi extrapolation exposure: fraction of Gandert sheets below validated psi range
n_extrap = int((gd.porosity < 0.369).sum())
print(f"    psi extrapolation: {n_extrap}/{len(gd)} Gandert sheets lie below the validated "
      f"psi_min = 0.369 (down to {gd.porosity.min():.3f}); calibrated residuals there are "
      f"indistinguishable from the high-psi branch (section 7) -- extrapolation absorbed by phi(Pi)")
# (3c) NMC811 untested regime
print(f"    NMC811 recovery term: b unconstrained for Pi > {gd[gd.system=='NMC811'].Pi.max():.2f} "
      f"-- do not use its calibration beyond the tested compression range")''')

results_md = new_markdown_cell(source="""**Section 7.3 results** -- PLACEHOLDER (filled after execution)""")

idx = next(i for i, c in enumerate(nb.cells)
           if c.cell_type == "markdown" and c.source.startswith("**Section 7.2 results**"))
nb.cells[idx+1:idx+1] = [intro_md, code_cell, results_md]
nbformat.validate(nb)
nbformat.write(nb, "Electrode.ipynb")
print(f"OK - inserted 7.3, total {len(nb.cells)} cells")
