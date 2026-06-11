"""Zehner-Bauer-Schlünder (ZBS) stagnant-bed conductivity closure (Eq. 37a/38/42a).

lambda_so / lambda is the conductivity of the *stagnant* (non-flowed) packed
bed relative to the fluid. The flowed model (Eq. 66) adds a Peclet-dependent
dispersion term: Lambda/lambda = lambda_so/lambda + Pe / K.
"""

import numpy as np

C_SPHERE = 1.25  # shape factor C from Tab. 2, for spherical particles


def b_factor(psi: float, C: float = C_SPHERE) -> float:
    """Deformation parameter B (Eq. 42a)."""
    return C * ((1 - psi) / psi) ** (10 / 9)


def lambda_so_over_lambda(psi, kappa, C: float = C_SPHERE):
    """Relative stagnant-bed conductivity lambda_so/lambda (Eq. 37a/38).

    psi: bed porosity
    kappa: lambda_s / lambda, solid-to-fluid conductivity ratio (0 <= kappa <= 1e4)
    """
    psi = np.asarray(psi, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    B = b_factor(psi, C)
    B_over_kappa = B / kappa
    one_minus = 1 - B_over_kappa

    term1 = (B * (kappa - 1)) / (kappa * one_minus**2) * np.log(kappa / B)
    term2 = -(B + 1) / 2
    term3 = -(B - 1) / one_minus

    lambda_pm = (2 / one_minus) * (term1 + term2 + term3)
    return 1 - np.sqrt(1 - psi) + np.sqrt(1 - psi) * lambda_pm
