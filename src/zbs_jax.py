"""JAX implementation of the Zehner-Bauer-Schlünder stagnant-bed closure (Eq. 37a/38/42a).

Mirrors `src/zbs.py` (numpy) but uses jax.numpy so it is differentiable
(jax.grad / jax.vmap) for the Multiphysics coupling and sensitivity analysis.
"""

import jax.numpy as jnp

C_SPHERE = 1.25  # shape factor C from Tab. 2, for spherical particles


def b_factor(psi, C: float = C_SPHERE):
    """Deformation parameter B (Eq. 42a)."""
    return C * ((1 - psi) / psi) ** (10 / 9)


def lambda_so_over_lambda(psi, kappa, C: float = C_SPHERE):
    """Relative stagnant-bed conductivity lambda_so/lambda (Eq. 37a/38).

    psi: bed porosity
    kappa: lambda_s / lambda, solid-to-fluid conductivity ratio (0 <= kappa <= 1e4)
    """
    psi = jnp.asarray(psi, dtype=jnp.float64)
    kappa = jnp.asarray(kappa, dtype=jnp.float64)
    B = b_factor(psi, C)
    B_over_kappa = B / kappa
    one_minus = 1 - B_over_kappa

    term1 = (B * (kappa - 1)) / (kappa * one_minus**2) * jnp.log(kappa / B)
    term2 = -(B + 1) / 2
    term3 = -(B - 1) / one_minus

    lambda_pm = (2 / one_minus) * (term1 + term2 + term3)
    return 1 - jnp.sqrt(1 - psi) + jnp.sqrt(1 - psi) * lambda_pm
