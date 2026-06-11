"""Assemble the tidy, dimensionless dataset from the transcribed Messwerttabellen.

Reads data/raw/messwerttabellen.csv plus the Tab. 5 / Tab. 6 join tables,
attaches fluid properties at each measurement temperature, and writes a tidy
parquet frame to data/processed/zehner_dataset.parquet.
"""

from pathlib import Path

import pandas as pd

import fluid_properties as fp
import materials as mat

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def build_dataset() -> pd.DataFrame:
    meas = pd.read_csv(RAW_DIR / "messwerttabellen.csv")
    tab5 = mat.load_tab5()
    tab6 = mat.load_tab6()

    join = mat.build_join_table(meas["vers_nr"], tab5, tab6)
    df = meas.merge(join, on="vers_nr", validate="one_to_one")

    df["lambda_n2"] = df["T_K"].apply(fp.lambda_n2)
    df["cp_n2"] = df["T_K"].apply(fp.cp_n2)
    df["rho_n2"] = df["T_K"].apply(fp.rho_n2)
    df["thermal_diffusivity"] = df["T_K"].apply(fp.thermal_diffusivity)

    df["lambda_s_over_lambda"] = df.apply(
        lambda r: mat.lambda_s_over_lambda(r["prefix"], r["T_K"], r["lambda_n2"], tab5),
        axis=1,
    )
    df["d_over_D"] = (df["d_mm"] / df["D_mm"]).astype(float)
    df["Nu_r"] = df.apply(
        lambda r: fp.nu_radiation(r["T_K"], r["d_mm"] * 1e-3, r["epsilon"], r["lambda_n2"]),
        axis=1,
    )

    cols = [
        "vers_nr",
        "prefix",
        "material",
        "T_K",
        "mdot_kg_h",
        "z",
        "d_mm",
        "D_mm",
        "L_mm",
        "epsilon",
        "psi",
        "d_over_D",
        "lambda_n2",
        "cp_n2",
        "rho_n2",
        "thermal_diffusivity",
        "lambda_s_over_lambda",
        "PE_eff",
        "Pe",
        "Nu_r",
        "lambda_ratio_meas",
        "lambda_ratio_calc",
        "fehl",
    ]
    return df[cols]


def main() -> None:
    df = build_dataset()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "zehner_dataset.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df.dtypes)
    print(df.describe())


if __name__ == "__main__":
    main()
