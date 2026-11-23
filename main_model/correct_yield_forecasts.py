"""Add predicted yield errors to baselines and export the corrected curves.

Output layout follows the beta correction files. Lambda records the parameter
used for the NS baseline; ML errors are added directly to the yield values.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main_model.ar_forecast import forecast_betas
from main_model.nelson_siegel import COLUMNS, MATURITIES, LAMBDA, estimate_betas, ns_loadings


DEFAULT_DIR = ROOT / "data" / "yield_error_correction"
DEFAULT_OUTPUT_DIR = DEFAULT_DIR / "corrected_yield_forcasts"
DEFAULT_DATABASE = ROOT / "data" / "MLyielddatabase.csv"
DEFAULT_YIELDS = ROOT / "data/error_correction/corrected_forecasts/baseline_yield_history.csv"
KEYS = ["origin", "k", "target"]
IDENTIFIERS = [*KEYS, "config_id", "training_cutoff"]
BASELINE_COLUMNS = [f"forecast_yield_{m}m" for m in MATURITIES]
ERROR_COLUMNS = [f"predicted_error_yield_{m}m" for m in MATURITIES]
CORRECTED_COLUMNS = [f"corrected_yield_{m}m" for m in MATURITIES]


def corrected_yield_outputs(matched, horizon=12):
    rows = matched.copy()
    for name in ("origin", "target", "training_cutoff"):
        rows[name] = pd.to_datetime(rows[name])
    if rows.empty or not np.isfinite(rows[BASELINE_COLUMNS + ERROR_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Every yield error needs a finite baseline yield.")
    for (origin, config), group in rows.groupby(["origin", "config_id"], dropna=False):
        targets = pd.date_range(origin + pd.offsets.MonthEnd(1), periods=horizon, freq="ME")
        if (len(group) != horizon or set(group["k"]) != set(range(1, horizon + 1))
                or not pd.DatetimeIndex(group.sort_values("k")["target"]).equals(targets)
                or pd.isna(config) or not str(config).strip()
                or group["training_cutoff"].isna().any()
                or group["training_cutoff"].nunique() != 1
                or group["training_cutoff"].gt(origin).any()):
            raise ValueError(f"{origin.date()} / {config}: require horizons 1..{horizon} and one training cutoff no later than origin.")

    # The trained error is actual - baseline, so add it in percentage points.
    rows[CORRECTED_COLUMNS] = rows[BASELINE_COLUMNS].to_numpy() + rows[ERROR_COLUMNS].to_numpy()
    rows["lambda"] = LAMBDA
    rows = rows[IDENTIFIERS + ["baseline_source", "lambda"] + BASELINE_COLUMNS + ERROR_COLUMNS + CORRECTED_COLUMNS]
    rows = rows.sort_values(["origin", "config_id", "k"]).reset_index(drop=True)
    curves = rows[IDENTIFIERS + ["lambda"]].copy()
    curves[COLUMNS] = rows[CORRECTED_COLUMNS].to_numpy()
    return rows, curves


def run_yield_correction(predictions_path=DEFAULT_DIR / "predictions.csv",
                         output_dir=DEFAULT_OUTPUT_DIR, horizon=12,
                         csv_path=DEFAULT_DATABASE, yields_csv=DEFAULT_YIELDS):
    database = pd.read_csv(csv_path, parse_dates=["origin", "target"])
    predictions = pd.read_csv(predictions_path, parse_dates=["origin", "target", "training_cutoff"])
    matched = predictions.merge(database[KEYS + BASELINE_COLUMNS], on=KEYS,
                                how="left", validate="many_to_one")
    matched["baseline_source"] = np.where(
        matched[BASELINE_COLUMNS].notna().all(axis=1), "MLyielddatabase", "AR_recomputed",
    )

    # Recent origins have unobserved horizons absent from the collector's CSV.
    # Regenerate those baseline yields using history through the origin only.
    missing = matched[BASELINE_COLUMNS].isna().any(axis=1)
    yield_history = None
    if missing.any():
        yield_history = pd.read_csv(yields_csv, index_col="date", parse_dates=True)
        for origin in matched.loc[missing, "origin"].unique():
            factors, _ = estimate_betas(yield_history.loc["1996-01-31":origin, COLUMNS])
            forecast = forecast_betas(factors, horizon)
            baselines = pd.DataFrame(forecast.to_numpy() @ ns_loadings(MATURITIES, LAMBDA).T,
                                     index=forecast.index, columns=BASELINE_COLUMNS)
            mask = missing & matched["origin"].eq(origin)
            matched.loc[mask, BASELINE_COLUMNS] = baselines.loc[matched.loc[mask, "target"]].to_numpy()

    rows, curves = corrected_yield_outputs(matched, horizon)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_dir / "yield_correction_details.csv", index=False, date_format="%Y-%m-%d")
    curves.to_csv(output_dir / "final_yield_curves.csv", index=False, date_format="%Y-%m-%d")
    if yield_history is not None:
        yield_history.loc["1996-01-31":rows["origin"].max(), COLUMNS].to_csv(
            output_dir / "baseline_yield_history.csv", index_label="date", date_format="%Y-%m-%d",
        )
    print(f"Saved {len(curves)} corrected yield curves to {output_dir}")
    return rows, curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_DIR / "predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--yields-csv", type=Path, default=DEFAULT_YIELDS)
    parser.add_argument("--horizon", type=int, default=12)
    args = parser.parse_args()
    run_yield_correction(args.predictions, args.output_dir, args.horizon, args.data, args.yields_csv)


if __name__ == "__main__":
    main()
