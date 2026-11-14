"""Add predicted beta errors to baseline forecasts and export Nelson-Siegel curves.

Match on (origin, k, target), keeping each XGBoost configuration separate.
Reuse MLdatabase's baseline betas; regenerate missing horizons with the existing
AR model and yield observations through the origin only.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .ar_forecast import forecast_betas
    from .nelson_siegel import COLUMNS, LAMBDA, MATURITIES, estimate_betas, ns_loadings
else:
    from ar_forecast import forecast_betas
    from nelson_siegel import COLUMNS, LAMBDA, MATURITIES, estimate_betas, ns_loadings


ROOT = Path(__file__).resolve().parents[1]
KEYS = ["origin", "k", "target"]
BASELINE_COLUMNS = [f"forecast_beta{i}" for i in range(1, 4)]
ERROR_COLUMNS = [f"predicted_error_beta{i}" for i in range(1, 4)]
CORRECTED_COLUMNS = [f"corrected_beta{i}" for i in range(1, 4)]
IDENTIFIERS = [*KEYS, "config_id", "training_cutoff"]
# The collector rounds MLdatabase to three decimals.
ROUNDING_TOLERANCE = 0.000501
HISTORY_START = pd.Timestamp("1996-01-31")


# Validate inputs and attach stored basline betas to each prediction row
def match_baselines(database: pd.DataFrame, predictions: pd.DataFrame, horizon=12) -> pd.DataFrame:
    if not isinstance(horizon, int) or not 1 <= horizon <= 12:
        raise ValueError("horizon must be an integer from 1 to 12.")
    database = database.copy()
    predictions = predictions.copy()
    for frame in (database, predictions):
        frame["origin"] = pd.to_datetime(frame["origin"])
        frame["target"] = pd.to_datetime(frame["target"])
    for frame, columns, name in (
        (database, BASELINE_COLUMNS, "MLdatabase"),
        (predictions, ["config_id", "training_cutoff", *ERROR_COLUMNS], "predictions"),
    ):
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    for frame, columns in ((database, BASELINE_COLUMNS), (predictions, ERROR_COLUMNS)):
        frame[columns] = frame[columns].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
            raise ValueError(f"Missing or non-finite values in {columns}.")
    if predictions["config_id"].isna().any() or predictions["config_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Every prediction needs a config_id.")
    if database.duplicated(KEYS).any() or predictions.duplicated([*KEYS, "config_id"]).any():
        raise ValueError("Duplicate forecast keys; refusing an ambiguous baseline/error match.")
    cutoff = pd.to_datetime(predictions["training_cutoff"], errors="raise")
    if cutoff.isna().any() or cutoff.dt.tz is not None or (cutoff > predictions["origin"]).any():
        raise ValueError("training_cutoff must be present and cannot be after the forecast origin.")
    predictions["training_cutoff"] = cutoff
    for (origin, config), rows in predictions.groupby(["origin", "config_id"]):
        if set(rows["k"]) != set(range(1, horizon + 1)):
            raise ValueError(f"{origin.date()} / {config} must have horizons 1..{horizon} exactly.")
        if rows["training_cutoff"].nunique() != 1:
            raise ValueError(f"{origin.date()} / {config} has inconsistent training cutoffs.")

    # Only baseline forecasts enter this merge; realized betas/errors are not inputs.
    matched = predictions[IDENTIFIERS + ERROR_COLUMNS].merge(
        database[KEYS + BASELINE_COLUMNS], on=KEYS, how="left", validate="many_to_one",
    )
    matched["baseline_source"] = np.where(matched[BASELINE_COLUMNS[0]].notna(), "MLdatabase", "AR_recomputed")
    return matched.sort_values(["origin", "config_id", "k"]).reset_index(drop=True)


# Regenerate absent horizons, checking against any stored forecasts at that origin.
def fill_missing_baselines(matched: pd.DataFrame, yields: pd.DataFrame) -> pd.DataFrame:
    matched = matched.copy()
    yields = yields.copy()
    yields.columns = yields.columns.map(str)
    yields.index = pd.to_datetime(yields.index, errors="raise")
    if yields.index.tz is not None or yields.index.hasnans or yields.index.has_duplicates:
        raise ValueError("Yield history must have unique dates without missing values or timezones.")
    for origin in matched.loc[matched["baseline_source"] == "AR_recomputed", "origin"].unique():
        # Match the collector's January 1996 start and exclude all future yields.
        history = yields.loc[(yields.index >= HISTORY_START) & (yields.index <= origin), COLUMNS].sort_index()
        expected = pd.date_range(HISTORY_START, origin, freq="ME")
        if history.empty or not history.index.equals(expected):
            raise ValueError(f"Complete monthly yield history from 1996-01 through {origin:%Y-%m} is required.")
        history = history.apply(pd.to_numeric, errors="raise")
        if not np.isfinite(history.to_numpy(dtype=float)).all():
            raise ValueError(f"Missing or non-finite yields through {origin:%Y-%m}.")
        factors, _ = estimate_betas(history)
        origin_mask = matched["origin"].eq(origin)
        regenerated = forecast_betas(factors, int(matched.loc[origin_mask, "k"].max()))
        for index, row in matched.loc[origin_mask].iterrows():
            baseline = regenerated.loc[row["target"]].to_numpy(dtype=float)
            if row["baseline_source"] == "MLdatabase":
                if not np.allclose(row[BASELINE_COLUMNS].to_numpy(dtype=float), baseline,
                                   atol=ROUNDING_TOLERANCE, rtol=0):
                    raise ValueError(
                        f"Regenerated baseline disagrees with MLdatabase at {origin.date()}, k={row['k']}. "
                        "Supply the original yield snapshot with --yields-csv; data revisions or "
                        "different history must not be mixed with saved predictions."
                    )
            else:
                matched.loc[index, BASELINE_COLUMNS] = baseline
    return matched


# Correct each beta for each horizon, then evaluate its curve at the project's eight maturities.
# The equation keeps the same form, while each horizon has its own corrected coefficients.
def corrected_outputs(matched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not np.isfinite(matched[BASELINE_COLUMNS + ERROR_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Every predicted error needs a baseline beta before correction.")
    betas = matched.copy()
    # Error was trained as actual_beta - forecast_beta, so ADD it without rescaling.
    betas[CORRECTED_COLUMNS] = (
        betas[BASELINE_COLUMNS].to_numpy(dtype=float) + betas[ERROR_COLUMNS].to_numpy(dtype=float)
    )
    betas["lambda"] = LAMBDA
    betas = betas[IDENTIFIERS + ["baseline_source", "lambda"] + BASELINE_COLUMNS + ERROR_COLUMNS + CORRECTED_COLUMNS]
    curves = betas[IDENTIFIERS + ["lambda"]].copy()
    curves[COLUMNS] = betas[CORRECTED_COLUMNS].to_numpy(dtype=float) @ ns_loadings(MATURITIES, LAMBDA).T
    return betas, curves


def run_correction(
    csv_path=ROOT / "data" / "MLdatabase.csv",
    predictions_path=ROOT / "data" / "error_correction" / "predictions.csv",
    output_dir=ROOT / "data" / "error_correction" / "corrected_forecasts",
    config_id=None,
    yields_csv=None,
    database_only=False,
    horizon=12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    database = pd.read_csv(csv_path)
    predictions = pd.read_csv(predictions_path)
    if config_id is not None:
        predictions = predictions.loc[predictions["config_id"] == config_id]
        if predictions.empty:
            raise ValueError(f"No predictions for config_id {config_id!r}.")
    matched = match_baselines(database, predictions, horizon=horizon)
    missing = matched.loc[matched["baseline_source"] == "AR_recomputed", KEYS].drop_duplicates()
    yield_history = None
    if not missing.empty:
        print(f"{len(missing)} baseline horizons are absent from MLdatabase; regenerating through each origin only.")
        if database_only:
            raise ValueError("Missing baseline horizons. Omit --database-only and provide --yields-csv or a FRED_API_KEY in .env.")
        if yields_csv is not None:
            yield_history = pd.read_csv(yields_csv, index_col="date", parse_dates=True)
        else:
            from dotenv import load_dotenv

            if __package__:
                from .baselinemodel_run import pull_yields
            else:
                from baselinemodel_run import pull_yields

            load_dotenv(ROOT / ".env")
            api_key = os.getenv("FRED_API_KEY")
            if not api_key:
                raise ValueError("Missing baseline horizons: supply --yields-csv or set FRED_API_KEY in .env.")
            print("Downloading baseline yield history from FRED.")
            try:
                yield_history = pull_yields(START="1996-01-01", api_key=api_key)
            except Exception:
                # Network exceptions may include the API key in their URL.
                raise ValueError("FRED download failed. Check network access/API key or supply --yields-csv.") from None
        matched = fill_missing_baselines(matched, yield_history)
    betas, curves = corrected_outputs(matched)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in (("corrected_betas.csv", betas), ("corrected_yields.csv", curves)):
        frame.to_csv(output_dir / filename, index=False, date_format="%Y-%m-%d")
    if yield_history is not None:
        # Save the observations used so this run can be repeated without a download.
        yield_history.loc[HISTORY_START:betas["origin"].max(), COLUMNS].to_csv(
            output_dir / "baseline_yield_history.csv", index_label="date", date_format="%Y-%m-%d",
        )
    print(f"Saved {len(betas)} corrected curves across {betas['config_id'].nunique()} configurations to {output_dir.resolve()}")
    return betas, curves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "MLdatabase.csv")
    parser.add_argument("--predictions", type=Path, default=ROOT / "data" / "error_correction" / "predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "error_correction" / "corrected_forecasts")
    parser.add_argument("--config-id", help="Export one configuration; by default export every configuration separately.")
    parser.add_argument("--yields-csv", type=Path, help="Original month-end yields with date and the eight maturity columns.")
    parser.add_argument("--database-only", action="store_true", help="Require all baseline horizons to exist in MLdatabase.")
    parser.add_argument("--horizon", type=int, default=12, help="Require prediction horizons 1..horizon (default: 12).")
    args = parser.parse_args()
    try:
        run_correction(
            csv_path=args.data, predictions_path=args.predictions, output_dir=args.output_dir,
            config_id=args.config_id, yields_csv=args.yields_csv, database_only=args.database_only,
            horizon=args.horizon,
        )
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
