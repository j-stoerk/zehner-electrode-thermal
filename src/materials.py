"""Join tables for packing material (Tab. 5) and series geometry (Tab. 6)."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Temperatures (K) at which Tab. 5 lists lambda_s
LAMBDA_S_TEMPS = np.array([320, 600, 800, 900, 1000])
LAMBDA_S_COLS = [
    "lambda_s_320K",
    "lambda_s_600K",
    "lambda_s_800K",
    "lambda_s_900K",
    "lambda_s_1000K",
]

# IK09 (Schaumsilikat) only has a fixed lambda_s/lambda ratio (footnote 7),
# not an absolute conductivity.
IK09_LAMBDA_RATIO = 4.0


def load_tab5(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return pd.read_csv(raw_dir / "tab5_materials.csv")


def load_tab6(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return pd.read_csv(raw_dir / "tab6_series.csv")


def parse_vers_nr(vers_nr: str) -> tuple[str, int, int]:
    """Split 'PREFIX/NNN' into (prefix, suffix, block).

    block = (suffix // 10) * 10, used to look up Tab. 6 series ranges.
    """
    prefix, suffix_str = vers_nr.split("/")
    suffix = int(suffix_str)
    block = (suffix // 10) * 10
    return prefix, suffix, block


def lookup_series_geometry(prefix: str, block: int, tab6: pd.DataFrame) -> pd.Series:
    """Find the Tab. 6 row for `prefix` whose [block_lo, block_hi] contains `block`."""
    candidates = tab6[
        (tab6["prefix"] == prefix)
        & (tab6["block_lo"] <= block)
        & (tab6["block_hi"] >= block)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one Tab. 6 match for prefix={prefix!r}, block={block}, "
            f"got {len(candidates)}"
        )
    return candidates.iloc[0]


def lambda_s_of_T(prefix: str, T_K: float, tab5: pd.DataFrame) -> float:
    """Return lambda_s(T) [W/(m K)] for a packing material, linearly interpolated
    (and extrapolated at the edges) over the temperatures listed in Tab. 5.

    Raises for IK09, whose Tab. 5 entry is a fixed lambda_s/lambda ratio rather
    than an absolute conductivity -- use `lambda_s_over_lambda` for that material.
    """
    if prefix == "IK09":
        raise ValueError("IK09 has no absolute lambda_s; use lambda_s_over_lambda")

    row = tab5.loc[tab5["prefix"] == prefix].iloc[0]
    values = row[LAMBDA_S_COLS].astype(float)
    mask = values.notna()
    temps = LAMBDA_S_TEMPS[mask.values]
    vals = values[mask].values

    if len(vals) == 1:
        return float(vals[0])
    return float(np.interp(T_K, temps, vals))


def lambda_s_over_lambda(prefix: str, T_K: float, lambda_fluid: float, tab5: pd.DataFrame) -> float:
    """Dimensionless solid/fluid conductivity ratio lambda_s/lambda at T_K."""
    if prefix == "IK09":
        return IK09_LAMBDA_RATIO
    return lambda_s_of_T(prefix, T_K, tab5) / lambda_fluid


def build_join_table(meas_vers_nr: pd.Series, tab5: pd.DataFrame, tab6: pd.DataFrame) -> pd.DataFrame:
    """For each `vers_nr`, return prefix/material/geometry/material-property columns."""
    records = []
    for vers_nr in meas_vers_nr:
        prefix, suffix, block = parse_vers_nr(vers_nr)
        mat_row = tab5.loc[tab5["prefix"] == prefix].iloc[0]
        geo_row = lookup_series_geometry(prefix, block, tab6)
        records.append(
            {
                "vers_nr": vers_nr,
                "prefix": prefix,
                "material": mat_row["material"],
                "d_mm": mat_row["d_mm"],
                "epsilon": mat_row["epsilon"],
                "gamma_s_kg_m3": mat_row["gamma_s_kg_m3"],
                "psi": geo_row["psi"],
                "D_mm": geo_row["D_mm"],
                "L_mm": geo_row["L_mm"],
            }
        )
    return pd.DataFrame.from_records(records)
