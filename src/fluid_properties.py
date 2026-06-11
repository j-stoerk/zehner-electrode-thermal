"""Fluid (N2) properties used to non-dimensionalize the measurement table.

lambda_N2 follows Eq. (64) of the dissertation (a Sutherland-type fit to
nitrogen thermal conductivity, fitted in 1972 -- accurate to within a few
percent around 300 K and within ~15% at 900 K). cp and rho come from
CoolProp; Nu_r is the radiation Nusselt number used in Eq. 37a/38.
"""

import numpy as np
from CoolProp.CoolProp import PropsSI

P_ATM = 101325.0  # Pa, experiments run at ambient pressure
STEFAN_BOLTZMANN_CS = 5.67  # W/(m^2 K^4), with T expressed in units of 100 K


def lambda_n2(T_K: float) -> float:
    """Nitrogen thermal conductivity [W/(m K)], Eq. (64)."""
    numerator = 2.094 * (1 + 5e-5 * T_K) * np.sqrt(T_K)
    denominator = 1 + 123.6 / T_K
    return numerator / denominator * 1e-3


def cp_n2(T_K: float, p_Pa: float = P_ATM) -> float:
    """Nitrogen specific heat capacity [J/(kg K)] via CoolProp."""
    return PropsSI("CPMASS", "T", T_K, "P", p_Pa, "Nitrogen")


def rho_n2(T_K: float, p_Pa: float = P_ATM) -> float:
    """Nitrogen density [kg/m^3] via CoolProp."""
    return PropsSI("DMASS", "T", T_K, "P", p_Pa, "Nitrogen")


def thermal_diffusivity(T_K: float, p_Pa: float = P_ATM) -> float:
    """Thermal diffusivity a = lambda / (rho * cp) [m^2/s]."""
    return lambda_n2(T_K) / (rho_n2(T_K, p_Pa) * cp_n2(T_K, p_Pa))


def nu_radiation(T_K: float, d_m: float, epsilon: float, lambda_fluid: float) -> float:
    """Radiation Nusselt number Nu_r = [0.04*Cs/(2/eps - 1)] * (T/100)^3 * (d/lambda)."""
    return (0.04 * STEFAN_BOLTZMANN_CS / (2 / epsilon - 1)) * (T_K / 100) ** 3 * (d_m / lambda_fluid)
