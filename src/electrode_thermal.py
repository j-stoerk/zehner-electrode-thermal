"""Knudsen-extended ZBS effective thermal conductivity for battery electrode coatings.

Maps the validated Zehner-Bauer-Schlünder (ZBS) stagnant-bed closure (src/zbs_jax.py)
onto lithium-ion electrode manufacturing:

  - electrode coatings (anode graphite, cathode NMC) and separators are porous
    particle/fiber beds at micrometer scale;
  - at these pore sizes the gas mean free path (~68 nm for air at ambient) is no
    longer negligible, so gas conduction in the pores is reduced by the
    Smoluchowski/Knudsen effect:  lambda_gas_eff = lambda_gas / (1 + 2*beta*Kn);
  - the corrected fluid conductivity feeds the unchanged, validated ZBS closure.

Everything is JAX-differentiable, so porosity sensitivities (d lambda_eff / d psi)
and the inverse problem (measured lambda_eff -> porosity, for calendering QC)
come from autodiff rather than finite differences.

Validity notes (be honest when using this):
  - The ZBS closure was validated in this project for kappa in [4, 14094] and
    psi in [0.369, 0.429] on mm-scale packed beds. Electrode porosities
    (0.20-0.55) extrapolate the psi range; the closure is analytic and smooth
    there, but treat absolute values with a model-form uncertainty floor
    (~2 %, from the packed-bed GP residual study) plus a contact/binder caveat:
    this is the point-contact (phi = 0) ZBS variant, which tends to UNDER-predict
    conduction when sintered necks / binder bridges are present.
  - The Knudsen correction applies to GAS-filled pores only (dry electrodes,
    drying ovens, dry rooms). Liquid electrolyte has no Knudsen effect.
"""

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

from zbs_jax import lambda_so_over_lambda

# ---------------------------------------------------------------------------
# Reference material properties (ambient, ~300 K). All in W/(m K).
# These are defensible literature mid-values; treat as defaults, not gospel.
# ---------------------------------------------------------------------------
LAMBDA_AIR = 0.026            # dry air, 300 K, 1 atm
LAMBDA_ELECTROLYTE = 0.18     # LiPF6 in EC/DMC-type carbonate mixtures (0.16-0.20)
LAMBDA_GRAPHITE = 25.0        # spheroidized graphite particle, isotropic effective (5-200 anisotropic)
LAMBDA_NMC = 2.5              # NMC secondary particle (1.5-5)
LAMBDA_SEPARATOR_PE = 0.40    # polyethylene/polypropylene solid phase

KB = 1.380649e-23             # Boltzmann constant, J/K
D_AIR = 3.7e-10               # effective collision diameter of air molecules, m
D_HELIUM = 2.18e-10           # effective collision diameter of He, m (mfp ~190 nm ambient)
LAMBDA_HELIUM = 0.1518        # helium, 293 K (LFA purge gas; value used by Gandert et al. 2023)
BETA_AIR = 1.64               # Knudsen jump coefficient for air on technical surfaces (1.5-2.0)


# ---------------------------------------------------------------------------
# Knudsen / Smoluchowski gas-conduction correction
# ---------------------------------------------------------------------------
def mean_free_path(T=300.0, p=101325.0, d_gas=D_AIR):
    """Mean free path of gas molecules [m]. Air: ~68 nm; helium: ~190 nm (300 K, 1 atm)."""
    return KB * T / (jnp.sqrt(2.0) * jnp.pi * d_gas**2 * p)


def pore_diameter(psi, d_p):
    """Hydraulic pore diameter of a packed bed of spheres [m].

    d_pore = (2/3) * psi/(1-psi) * d_p   (standard hydraulic-diameter estimate)
    """
    return (2.0 / 3.0) * psi / (1.0 - psi) * d_p


def knudsen_number(psi, d_p, T=300.0, p=101325.0, d_gas=D_AIR):
    """Kn = mean free path / pore diameter."""
    return mean_free_path(T, p, d_gas) / pore_diameter(psi, d_p)


def lambda_gas_knudsen(psi, d_p, lambda_gas=LAMBDA_AIR, T=300.0, p=101325.0,
                       beta=BETA_AIR, d_gas=D_AIR):
    """Knudsen-reduced gas conductivity in the pore space [W/(m K)].

    lambda_gas_eff = lambda_gas / (1 + 2*beta*Kn)   (Smoluchowski form)
    """
    kn = knudsen_number(psi, d_p, T, p, d_gas)
    return lambda_gas / (1.0 + 2.0 * beta * kn)


# ---------------------------------------------------------------------------
# Effective coating conductivity = (Knudsen-corrected fluid) x ZBS closure
# ---------------------------------------------------------------------------
def lambda_eff_coating(psi, d_p, lambda_s, lambda_f=LAMBDA_AIR, gas_filled=True,
                       T=300.0, p=101325.0, beta=BETA_AIR, d_gas=D_AIR):
    """Effective through-plane conductivity of a porous coating [W/(m K)].

    psi        : coating porosity (pore volume fraction)
    d_p        : particle diameter [m] (sets the pore size for the Knudsen term)
    lambda_s   : solid-phase (particle) conductivity [W/(m K)]
    lambda_f   : pore-fluid conductivity [W/(m K)] (air or electrolyte)
    gas_filled : True -> apply the Knudsen correction (dry coating);
                 False -> liquid-filled pores, no correction.
    """
    lam_f_eff = jnp.where(
        gas_filled,
        lambda_gas_knudsen(psi, d_p, lambda_f, T, p, beta, d_gas),
        lambda_f,
    )
    kappa_eff = lambda_s / lam_f_eff
    return lambda_so_over_lambda(psi, kappa_eff) * lam_f_eff


# Porosity sensitivity, d lambda_eff / d psi, via autodiff (exact, no FD step)
dlambda_dpsi = jax.grad(lambda_eff_coating, argnums=0)


# ---------------------------------------------------------------------------
# Contact-augmented closure (VDI Heat Atlas flattened-contact form)
#
# k_bed = 1 - sqrt(1-psi) + sqrt(1-psi) * [ phi*kappa_bridge + (1-phi)*lam_pm ]
#
# phi is the flattened-contact / bridge area fraction (phi ~ 0.0077 for rigid
# spheres; larger for plastically deformed flakes and binder/carbon-black
# bridges). kappa_bridge = lambda_bridge / lambda_f allows the bridge phase to
# differ from the particle (e.g. carbon-black bridges between NMC particles).
# phi = 0 recovers the validated point-contact ZBS exactly.
# ---------------------------------------------------------------------------
from zbs_jax import C_SPHERE


def lambda_so_over_lambda_contact(psi, kappa, phi=0.0, kappa_bridge=None,
                                  C=C_SPHERE):
    """Relative bed conductivity with the VDI flattened-contact term."""
    if kappa_bridge is None:
        kappa_bridge = kappa
    psi = jnp.asarray(psi, dtype=jnp.float64)
    kappa = jnp.asarray(kappa, dtype=jnp.float64)
    B = C * ((1 - psi) / psi) ** (10 / 9)
    B_over_kappa = B / kappa
    one_minus = 1 - B_over_kappa
    term1 = (B * (kappa - 1)) / (kappa * one_minus**2) * jnp.log(kappa / B)
    term2 = -(B + 1) / 2
    term3 = -(B - 1) / one_minus
    lam_pm = (2 / one_minus) * (term1 + term2 + term3)
    return (1 - jnp.sqrt(1 - psi)
            + jnp.sqrt(1 - psi) * (phi * kappa_bridge + (1 - phi) * lam_pm))


def lambda_eff_coating_contact(psi, d_p, lambda_s, lambda_f=LAMBDA_AIR,
                               gas_filled=True, phi=0.0, lambda_bridge=None,
                               T=300.0, p=101325.0, beta=BETA_AIR, d_gas=D_AIR):
    """Effective coating conductivity with contact/bridge conduction [W/(m K)].

    phi           : contact/bridge area fraction (0 = validated point-contact ZBS)
    lambda_bridge : bridge-phase conductivity [W/(m K)]; defaults to lambda_s.
                    For binder/carbon-black-bridged cathodes use the additive
                    conductivity (e.g. ~24 for carbon black).
    """
    if lambda_bridge is None:
        lambda_bridge = lambda_s
    lam_f_eff = jnp.where(
        gas_filled,
        lambda_gas_knudsen(psi, d_p, lambda_f, T, p, beta, d_gas),
        lambda_f,
    )
    return lambda_so_over_lambda_contact(
        psi, lambda_s / lam_f_eff, phi, lambda_bridge / lam_f_eff
    ) * lam_f_eff


# ---------------------------------------------------------------------------
# Inverse problem: measured lambda_eff -> porosity (calendering QC)
# ---------------------------------------------------------------------------
def invert_porosity(lambda_meas, d_p, lambda_s, lambda_f=LAMBDA_AIR,
                    gas_filled=True, psi_init=0.35, n_newton=25,
                    psi_min=0.05, psi_max=0.75):
    """Recover porosity from a measured effective conductivity (Newton iteration).

    Uses the exact autodiff derivative of the forward model, so convergence is
    quadratic. lambda_eff(psi) is strictly monotonic (decreasing) over the
    electrode range, so the root is unique.

    Returns the estimated porosity psi_hat (clipped to [psi_min, psi_max]).
    """
    def newton_step(psi, _):
        f = lambda_eff_coating(psi, d_p, lambda_s, lambda_f, gas_filled) - lambda_meas
        df = dlambda_dpsi(psi, d_p, lambda_s, lambda_f, gas_filled)
        psi_new = psi - f / df
        return jnp.clip(psi_new, psi_min, psi_max), None

    psi_hat, _ = jax.lax.scan(newton_step, jnp.asarray(psi_init, dtype=jnp.float64),
                              None, length=n_newton)
    return psi_hat


def porosity_uncertainty(psi_hat, d_p, lambda_s, lambda_f=LAMBDA_AIR,
                         gas_filled=True, sigma_lambda_rel=0.03,
                         sigma_model_rel=0.02):
    """First-order propagated porosity uncertainty for the inverse QC estimate.

    sigma_psi = sqrt(sigma_meas^2 + sigma_model^2) * lambda_eff / |d lambda/d psi|

    sigma_lambda_rel : relative measurement noise of the conductivity method
                       (transient plane source / laser flash: typically 2-5 %)
    sigma_model_rel  : relative model-form uncertainty floor (~2 % from the
                       packed-bed GP residual study; treat as a lower bound
                       outside the validated psi range)
    """
    lam = lambda_eff_coating(psi_hat, d_p, lambda_s, lambda_f, gas_filled)
    dlam = dlambda_dpsi(psi_hat, d_p, lambda_s, lambda_f, gas_filled)
    sigma_lam = jnp.sqrt(sigma_lambda_rel**2 + sigma_model_rel**2) * lam
    return sigma_lam / jnp.abs(dlam)
