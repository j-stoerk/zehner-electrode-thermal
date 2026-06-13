"""Single-source-of-truth guard: the canonical computed numbers must match the
manuscript and the family registry, so code/notebooks/LaTeX/docs cannot drift.
"""
import os, sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from electrode_data import FAMILIES, TRANSFER_PAIRS, load_gandert, calibrate
import key_results

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def R():
    return key_results.compute()


# --- registry ---------------------------------------------------------------
def test_registry_has_four_families():
    assert set(FAMILIES) == {"graphite_thin", "graphite_thick", "NMC622", "NMC811"}
    assert FAMILIES["graphite_thin"]["lam_b"] == 130.0
    assert FAMILIES["NMC622"]["lam_b"] == 24.0
    assert FAMILIES["graphite_thin"]["collector"] == "Cu"
    assert FAMILIES["NMC811"]["collector"] == "Al"
    assert TRANSFER_PAIRS == [("graphite_thin", "graphite_thick"), ("NMC622", "NMC811")]


# --- canonical numbers reproduce the published values -----------------------
def test_ablation_endpoints(R):
    assert R["mape_zero_fit_avg"] == pytest.approx(31.1, abs=0.5)
    assert R["mape_calibrated_avg"] == pytest.approx(4.5, abs=0.5)


def test_per_family_calibrated_mape(R):
    exp = {"graphite_thin": 1.8, "graphite_thick": 5.4, "NMC622": 1.4, "NMC811": 9.4}
    for fam, m in exp.items():
        assert R["calibrated"][fam]["mape"] == pytest.approx(m, abs=0.6), fam


def test_nmc811_headline(R):
    assert R["nmc811_uncal_err_pct"] == pytest.approx(2.7, abs=0.5)


def test_reference_values(R):
    rv = R["reference_values"]
    assert rv["anode_wet"] == pytest.approx(2.56, abs=0.03)
    assert rv["cathode_wet"] == pytest.approx(0.97, abs=0.03)
    assert rv["separator_dry_knudsen"] == pytest.approx(0.065, abs=0.003)


# --- manuscript must quote the canonical numbers ----------------------------
def test_manuscript_matches_canonical(R):
    tex = open(os.path.join(_ROOT, "publication", "main.tex"), encoding="utf-8").read()
    must_contain = [
        f'{R["mape_zero_fit_avg"]}',        # 31.1
        f'{R["mape_calibrated_avg"]}',      # 4.5
        f'{R["nmc811_uncal_err_pct"]}',     # 2.7
        f'{R["reference_values"]["anode_wet"]}',   # 2.56
        f'{R["reference_values"]["cathode_wet"]}', # 0.97
    ]
    missing = [v for v in must_contain if v not in tex]
    assert not missing, f"manuscript missing canonical numbers: {missing}"
