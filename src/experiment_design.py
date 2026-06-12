"""Information-optimal experiment design for the calendering campaign.

Chooses which calendering states (and chamber pressures) to measure so that
the campaign constrains the contact-closure parameters (lambda_s, phi0, a, b)
as tightly as possible: greedy D-optimal selection on the Fisher information
of the relative-residual likelihood, with exact JAX jacobians of the forward
model (no finite differences).

Usage (see also publication/measurement_request_partner.md):

    from experiment_design import select_calendering_states, pressure_split_gain
    Pis = select_calendering_states(theta0, fam, n_select=6)
"""
from jax import config as _cfg

_cfg.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np

from electrode_thermal import (
    lambda_so_over_lambda_contact, lambda_gas_knudsen,
    LAMBDA_HELIUM, D_HELIUM, D_AIR, LAMBDA_AIR,
)


def _forward(theta, psi, Pi, d_p, lam_b, lam_gas=LAMBDA_HELIUM, d_gas=D_HELIUM,
             p_mbar=1013.0):
    """Dry-coating forward model lambda_eff(theta; psi, Pi) at gas pressure p."""
    lam_s, phi0, a, b = theta
    lam_f = lambda_gas_knudsen(psi, d_p, lam_gas, p=p_mbar * 100.0, d_gas=d_gas)
    phi = jnp.maximum(0.0, phi0 + a * Pi + b * Pi**2)
    return lambda_so_over_lambda_contact(psi, lam_s / lam_f, phi, lam_b / lam_f) * lam_f


def trajectory(psi0, Pi):
    """Porosity along the mass-conserving calendering trajectory."""
    return 1.0 - (1.0 - psi0) / (1.0 - Pi)


def fisher_information(theta, design_Pis, psi0, d_p, lam_b, sigma_rel=0.03,
                       **gas_kw):
    """Fisher information matrix for relative-residual Gaussian noise."""
    theta = jnp.asarray(theta, dtype=jnp.float64)

    def rel_model(th, Pi):
        psi = trajectory(psi0, Pi)
        lam = _forward(th, psi, Pi, d_p, lam_b, **gas_kw)
        lam0 = _forward(theta, psi, Pi, d_p, lam_b, **gas_kw)
        return lam / lam0  # relative sensitivity around the reference point

    J = jnp.stack([jax.jacobian(rel_model)(theta, Pi) for Pi in design_Pis])
    return np.asarray(J.T @ J) / sigma_rel**2


def select_calendering_states(theta, psi0, d_p, lam_b, n_select=6,
                              Pi_max=0.5, n_candidates=51, sigma_rel=0.03,
                              ridge=1e-9, **gas_kw):
    """Greedy D-optimal selection of calendering states Pi.

    Returns the selected Pi values (sorted). Pi=0 (the as-coated anchor that
    re-anchors phi0 across recipes) is always included first.
    """
    candidates = list(np.linspace(0.0, Pi_max, n_candidates))
    selected = [0.0]
    candidates.remove(0.0)
    for _ in range(n_select - 1):
        best, best_logdet = None, -np.inf
        for c in candidates:
            F = fisher_information(theta, selected + [c], psi0, d_p, lam_b,
                                   sigma_rel=sigma_rel, **gas_kw)
            s, logdet = np.linalg.slogdet(F + ridge * np.eye(4))
            if logdet > best_logdet:
                best, best_logdet = c, logdet
        selected.append(best)
        candidates.remove(best)
    return sorted(selected)


def pressure_split_gain(theta, Pi, psi0, d_p, lam_b, lam_gas=LAMBDA_AIR,
                        d_gas=D_AIR, p_lo_mbar=100.0, p_hi_mbar=1013.0):
    """Relative conductivity change between two chamber pressures.

    Pure pore-gas (Knudsen) effect: skeleton and bridge conduction are
    pressure-independent, so this split isolates mechanism M-A. Returns the
    relative drop (lam(p_hi) - lam(p_lo)) / lam(p_hi); compare against the
    measurement noise to size the experiment.
    """
    psi = trajectory(psi0, Pi)
    hi = float(_forward(jnp.asarray(theta), psi, Pi, d_p, lam_b,
                        lam_gas=lam_gas, d_gas=d_gas, p_mbar=p_hi_mbar))
    lo = float(_forward(jnp.asarray(theta), psi, Pi, d_p, lam_b,
                        lam_gas=lam_gas, d_gas=d_gas, p_mbar=p_lo_mbar))
    return (hi - lo) / hi
