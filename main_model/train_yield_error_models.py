"""Train yield-error models using the same layout as train_error_models.py.

Read MLyielddatabase.csv directly and save six configurations x eight maturity
models in data/yield_error_correction. Only errors realized by the forecast
origin enter training. Each model uses k, origin yields, and macro features.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import product
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "MLyielddatabase.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "yield_error_correction"
MATURITIES = (3, 6, 12, 24, 36, 60, 84, 120)
FEATURES = (
    "k", *[f"yield_{m}m_origin" for m in MATURITIES],
    "inflation", "unemployment_rate", "unemployment_chg",
    "fed_funds_rate", "fed_funds_chg_3m", "fed_funds_chg_6m",
    "sentiment_score",
)
TARGETS = tuple(f"error_yield_{m}m" for m in MATURITIES)


def predict_yield_errors(config_id, feature_rows, output_dir=DEFAULT_OUTPUT_DIR):
    """Return eight yield errors, with the same identifiers as beta predictions."""
    output_dir = Path(output_dir)
    metadata = json.loads((output_dir / "metadata.json").read_text())
    cutoff = pd.Timestamp(metadata["training"]["cutoff"])
    if "origin" in feature_rows and (pd.to_datetime(feature_rows["origin"]) < cutoff).any():
        raise ValueError("Prediction origins cannot precede the saved models' training cutoff.")

    identifiers = [name for name in ("origin", "k", "target") if name in feature_rows]
    predictions = feature_rows[identifiers].copy()
    predictions["config_id"] = config_id
    predictions["training_cutoff"] = cutoff.date().isoformat()
    features = feature_rows[metadata["features"]]
    for target, filename in metadata["configurations"][config_id]["models"].items():
        model = XGBRegressor()
        model.load_model(output_dir / filename)
        predictions[f"predicted_{target}"] = model.predict(features).astype(float)
    return predictions


def train_yield_error_models(
    csv_path=DEFAULT_DATA_PATH, output_dir=DEFAULT_OUTPUT_DIR,
    forecast_origin=None, max_depths=(2, 3), n_estimators=(100, 150, 200),
    learning_rate=0.05, horizon=12, seed=42, max_seconds=1800,
):
    """Train all configurations and export wide prediction and parameter CSVs."""
    started = time.monotonic()
    csv_path, output_dir = Path(csv_path), Path(output_dir)
    snapshot = csv_path.read_bytes()
    data = pd.read_csv(io.BytesIO(snapshot), parse_dates=["origin", "target"])
    data = data.sort_values(["origin", "k"]).reset_index(drop=True)
    origin = pd.Timestamp(forecast_origin) if forecast_origin is not None else data["origin"].max()
    origin_rows = data.loc[data["origin"] == origin]
    training = data.loc[data["target"] <= origin]

    # Basic input checks; the collector already prepares the database layout.
    if pd.isna(origin) or not origin.is_month_end or origin != origin.normalize() or origin_rows.empty:
        raise ValueError("Use a forecast origin that is a month-end in MLyielddatabase.csv.")
    if training.empty:
        raise ValueError("No realized yield errors are available by the forecast origin.")
    if (not max_depths or not n_estimators or any(x < 1 for x in (*max_depths, *n_estimators))
            or not 0 < learning_rate <= 1 or not 1 <= horizon <= 12 or max_seconds <= 0):
        raise ValueError("Use positive model parameters, 0 < learning_rate <= 1, and horizon 1..12.")

    # All inputs are known at the origin; only k changes across future horizons.
    feature_rows = pd.DataFrame([
        origin_rows.iloc[0][list(FEATURES)].to_dict() for _ in range(horizon)
    ])
    feature_rows["k"] = range(1, horizon + 1)
    feature_rows["origin"] = origin
    feature_rows["target"] = pd.date_range(origin + pd.offsets.MonthEnd(1), periods=horizon, freq="ME")

    common_params = {
        "objective": "reg:squarederror", "booster": "gbtree",
        "tree_method": "hist", "device": "cpu",
        "learning_rate": learning_rate, "random_state": seed, "n_jobs": 1,
        "subsample": 1.0, "colsample_bytree": 1.0,
        "min_child_weight": 1.0, "gamma": 0.0,
        "reg_alpha": 0.0, "reg_lambda": 1.0, "max_bin": 256, "base_score": 0.0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    parameter_rows, configurations = [], {}
    for depth, count in product(sorted(set(max_depths)), sorted(set(n_estimators))):
        rate_label = format(learning_rate, ".12g").replace(".", "")
        config_id = f"d{depth}_n{count}_lr{rate_label}"
        params = {**common_params, "max_depth": depth, "n_estimators": count}
        models = {}
        for target in TARGETS:
            if time.monotonic() - started >= max_seconds:
                raise TimeoutError("Yield-model training exceeded the runtime budget.")
            model = XGBRegressor(**params)
            model.fit(training[list(FEATURES)], training[target])
            filename = f"{config_id}_{target}.json"
            model.save_model(output_dir / filename)
            models[target] = filename
        configurations[config_id] = {"parameters": params, "models": models}
        parameter_rows.append({
            "config_id": config_id, **params,
            "training_cutoff": origin.date().isoformat(), "n_training_rows": len(training),
        })
        print(f"Trained {config_id}: eight yield-error models on {len(training)} realized rows.", flush=True)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "file": str(csv_path.resolve()), "sha256": hashlib.sha256(snapshot).hexdigest(),
            "n_rows": len(data),
        },
        "features": list(FEATURES), "targets": list(TARGETS),
        "error_definition": "actual_yield - Nelson_Siegel_AR_forecast_yield",
        "prediction_units": "percentage points",
        "training": {
            "cutoff": origin.date().isoformat(), "rule": "target <= forecast_origin",
            "n_rows": len(training),
            "first_realized_target": training["target"].min().date().isoformat(),
            "last_realized_target": training["target"].max().date().isoformat(),
        },
        "prediction": {
            "origin": origin.date().isoformat(), "horizon": horizon,
            "first_target": feature_rows["target"].min().date().isoformat(),
            "last_target": feature_rows["target"].max().date().isoformat(),
        },
        "seed": seed, "n_models": len(configurations) * len(TARGETS),
        "training_seconds": time.monotonic() - started,
        "versions": {"python": platform.python_version(), **{
            package: version(package) for package in ("numpy", "pandas", "xgboost", "scikit-learn")
        }},
        "configurations": configurations,
        "data_note": "Historical features and targets come from the existing revised-data snapshot.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(parameter_rows).to_csv(output_dir / "parameters.csv", index=False)
    predictions = pd.concat([
        predict_yield_errors(config_id, feature_rows, output_dir) for config_id in configurations
    ], ignore_index=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False, date_format="%Y-%m-%d")
    print(f"Forecast origin: {origin.date()}; saved {len(predictions)} rows to {output_dir.resolve()}", flush=True)
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--forecast-origin")
    parser.add_argument("--max-depths", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--n-estimators", nargs="+", type=int, default=[100, 150, 200])
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seconds", type=float, default=1800)
    args = parser.parse_args()
    train_yield_error_models(
        csv_path=args.data, output_dir=args.output_dir, forecast_origin=args.forecast_origin,
        max_depths=args.max_depths, n_estimators=args.n_estimators,
        learning_rate=args.learning_rate, horizon=args.horizon, seed=args.seed,
        max_seconds=args.max_seconds,
    )


if __name__ == "__main__":
    main()
