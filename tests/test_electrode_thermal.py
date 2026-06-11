"""Unit + regression tests for src/electrode_thermal.py.

Run from the repository root:
    .venv/Scripts/python.exe -m pytest tests/ -v
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from electrode_thermal import (
    mean_free_path, pore_diameter, knudsen_number, lambda_gas_knudsen,
    lambda_eff_coating, lambda_eff_coating_contact, dlambda_dpsi,
    invert_porosity, porosity_uncertainty,
    LAMBDA_AIR, LAMBDA_ELECTROLYTE, LAMBDA_HELIUM, D_HELIUM,
    LAMBDA_GRAPHITE, LAMBDA_NMC, LAMBDA_SEPARATOR_PE,
)


# ---------------------------------------------------------------- physics --
def test_mean_free_path_air_ambient():
    # ~67 nm at 300 K, 1 atm (textbook value 65-70 nm)
    assert 60e-9 < float(mean_free_path()) < 75e-9


def test_mean_free_path_helium_larger_than_air():
    assert float(mean_free_path(d_gas=D_HELIUM)) > 2.5 * float(mean_free_path())


def test_knudsen_reduces_gas_conductivity():
    lam = float(lambda_gas_knudsen(0.40, 0.3e-6))  # separator-scale pores
    assert lam < 0.5 * LAMBDA_AIR  # strong rarefaction


def test_knudsen_negligible_for_large_pores():
    lam = float(lambda_gas_knudsen(0.40, 1e-3))    # mm-scale packed bed
    assert lam > 0.999 * LAMBDA_AIR


def test_pore_diameter_hydraulic():
    # (2/3) * psi/(1-psi) * d_p at psi=0.5 equals (2/3) d_p
    assert float(pore_diameter(0.5, 18e-6)) == pytest.approx(12e-6, rel=1e-12)


# ----------------------------------------------------------- consistency --
def test_contact_phi0_reduces_to_point_contact():
    a = float(lambda_eff_coating(0.45, 18e-6, 80.0, LAMBDA_HELIUM, True, d_gas=D_HELIUM))
    b = float(lambda_eff_coating_contact(0.45, 18e-6, 80.0, LAMBDA_HELIUM, True,
                                         phi=0.0, d_gas=D_HELIUM))
    assert a == pytest.approx(b, abs=1e-12)


def test_contact_term_increases_conductivity():
    base = float(lambda_eff_coating_contact(0.40, 18e-6, 25.0, LAMBDA_AIR, True, phi=0.0))
    bridged = float(lambda_eff_coating_contact(0.40, 18e-6, 25.0, LAMBDA_AIR, True,
                                               phi=0.01, lambda_bridge=130.0))
    assert bridged > base


def test_monotonic_decreasing_in_porosity():
    psis = np.linspace(0.20, 0.55, 12)
    lams = [float(lambda_eff_coating(p, 17e-6, LAMBDA_GRAPHITE, LAMBDA_ELECTROLYTE, False))
            for p in psis]
    assert all(lams[i] > lams[i + 1] for i in range(len(lams) - 1))


def test_wet_exceeds_dry():
    for lam_s, d_p in [(LAMBDA_GRAPHITE, 17e-6), (LAMBDA_NMC, 8e-6),
                       (LAMBDA_SEPARATOR_PE, 0.3e-6)]:
        wet = float(lambda_eff_coating(0.35, d_p, lam_s, LAMBDA_ELECTROLYTE, False))
        dry = float(lambda_eff_coating(0.35, d_p, lam_s, LAMBDA_AIR, True))
        assert wet > dry


def test_sensitivity_negative():
    assert float(dlambda_dpsi(0.30, 17e-6, LAMBDA_GRAPHITE, LAMBDA_ELECTROLYTE, False)) < 0


# --------------------------------------------------------------- inverse --
def test_inverse_round_trip_exact():
    lam = float(lambda_eff_coating(0.32, 17e-6, LAMBDA_GRAPHITE, LAMBDA_ELECTROLYTE, False))
    psi = float(invert_porosity(lam, 17e-6, LAMBDA_GRAPHITE, LAMBDA_ELECTROLYTE, False))
    assert psi == pytest.approx(0.32, abs=1e-10)


def test_porosity_uncertainty_magnitude():
    # ~0.009 for 3% meas + 2% model noise at the anode reference point
    sig = float(porosity_uncertainty(0.30, 17e-6, LAMBDA_GRAPHITE, LAMBDA_ELECTROLYTE,
                                     False, 0.03, 0.02))
    assert 0.005 < sig < 0.015


# ------------------------------------------------- published-value regression --
def test_regression_reference_values():
    """Values quoted in the manuscript and docs; guards against silent drift."""
    anode_wet = float(lambda_eff_coating(0.30, 17e-6, LAMBDA_GRAPHITE, LAMBDA_ELECTROLYTE, False))
    cathode_wet = float(lambda_eff_coating(0.30, 8e-6, LAMBDA_NMC, LAMBDA_ELECTROLYTE, False))
    sep_dry_kn = float(lambda_eff_coating(0.40, 0.3e-6, LAMBDA_SEPARATOR_PE, LAMBDA_AIR, True))
    assert anode_wet == pytest.approx(2.559, abs=0.01)
    assert cathode_wet == pytest.approx(0.970, abs=0.01)
    assert sep_dry_kn == pytest.approx(0.0647, abs=0.002)


def test_regression_nmc811_zero_fit():
    """NMC811 uncalendered: model 0.505 vs measured 0.492 (+2.7%) -- the headline."""
    lam = float(lambda_eff_coating(0.507, 10e-6, 2.5, LAMBDA_HELIUM, True, d_gas=D_HELIUM))
    assert lam == pytest.approx(0.505, abs=0.005)
