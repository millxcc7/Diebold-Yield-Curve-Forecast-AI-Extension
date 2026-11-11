"""Train XGBoost beta-error models and output errors for later forecast correction.

Reads data/MLdatabase.csv directly. The forecast origin defaults to its latest
origin month; only errors realized by that origin are used for training.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import product
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "MLdatabase.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "error_correction"
FEATURES = (
    "k",
    "beta1_origin", "beta2_origin", "beta3_origin",
    "inflation", "unemployment_rate", "unemployment_chg",
    "fed_funds_rate", "fed_funds_chg_3m", "fed_funds_chg_6m",
    "sentiment_score",
)
TARGETS = ("error_beta1", "error_beta2", "error_beta3")


def predict_errors(
    config_id: str,
    feature_rows: pd.DataFrame,
    output_dir=DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    """Return three predicted errors from a saved configuration and its 11 inputs.

    Optional origin/target dates are carried into the output. Historical origins
    preceding the saved model's training cutoff require an earlier trained model.
    """
    output_dir = Path(output_dir)
    metadata = json.loads((output_dir / "metadata.json").read_text())
    if config_id not in metadata["configurations"]:
        raise ValueError(f"No trained models saved for config_id {config_id!r}.")
    cutoff = pd.Timestamp(metadata["training"]["cutoff"])
    if "origin" in feature_rows and (pd.to_datetime(feature_rows["origin"]) < cutoff).any():
        raise ValueError("Prediction origins cannot precede the saved models' training cutoff.")

    identifiers = [name for name in ("origin", "k", "target") if name in feature_rows]
    predictions = feature_rows.loc[:, identifiers].copy()
    predictions["config_id"] = config_id
    predictions["training_cutoff"] = cutoff.date().isoformat()
    features = feature_rows.loc[:, metadata["features"]]
    for target, filename in metadata["configurations"][config_id]["models"].items():
        model = XGBRegressor()
        model.load_model(output_dir / filename)
        predictions[f"predicted_{target}"] = model.predict(features).astype(float)
    return predictions


def train_error_models(
    csv_path=DEFAULT_DATA_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
    forecast_origin=None,
    max_depths=(2, 3),
    n_estimators=(100, 150, 200),
    learning_rate=0.05,
    horizon=12,
    seed=42,
) -> pd.DataFrame:
    """Train each specified configuration and return future beta-error predictions.

    Parameters are supplied by the caller. Every configuration is saved without
    ranking or choosing one. Predictions cover months 1..horizon after the origin.
    Leave forecast_origin=None to use the latest origin in the CSV, or pass a
    month-end date when calling this function to train at a historical origin.
    Each run replaces predictions.csv and the parameter/metadata files; prediction
    rows are not appended across origins.
    """
    csv_path, output_dir = Path(csv_path), Path(output_dir)
    snapshot = csv_path.read_bytes()
    data = pd.read_csv(io.BytesIO(snapshot), parse_dates=["origin", "target"])
    data = data.sort_values(["origin", "k"]).reset_index(drop=True)
    origin = pd.Timestamp(forecast_origin) if forecast_origin is not None else data["origin"].max()
    if pd.isna(origin) or not origin.is_month_end or origin != origin.normalize():
        raise ValueError("forecast_origin must be a calendar month-end without a time component.")
    origin_rows = data.loc[data["origin"] == origin]
    if origin_rows.empty:
        raise ValueError("The requested forecast origin must exist in MLdatabase.csv.")
    training = data.loc[data["target"] <= origin]
    if training.empty:
        raise ValueError("No realized training errors are available by the requested origin.")
    if not max_depths or not n_estimators or any(x < 1 for x in (*max_depths, *n_estimators)):
        raise ValueError("Supply positive max_depth and n_estimators values.")
    if not 0 < learning_rate <= 1 or not 1 <= horizon <= 12:
        raise ValueError("Use 0 < learning_rate <= 1 and a horizon from 1 to 12 months.")

    # Origin features are shared across horizons; only k changes for future months.
    feature_rows = pd.DataFrame([
        origin_rows.iloc[0].loc[list(FEATURES)].to_dict() for _ in range(horizon)
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
        "reg_alpha": 0.0, "reg_lambda": 1.0, "max_bin": 256,
        "base_score": 0.0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    parameter_rows, configurations = [], {}
    for depth, count in product(sorted(set(max_depths)), sorted(set(n_estimators))):
        rate_label = format(learning_rate, ".12g").replace(".", "")
        config_id = f"d{depth}_n{count}_lr{rate_label}"
        params = {**common_params, "max_depth": depth, "n_estimators": count}
        models = {}
        for target in TARGETS:
            model = XGBRegressor(**params)
            model.fit(training.loc[:, list(FEATURES)], training[target])
            filename = f"{config_id}_{target}.json"
            model.save_model(output_dir / filename)
            models[target] = filename
        configurations[config_id] = {"parameters": params, "models": models}
        parameter_rows.append({
            "config_id": config_id, **params,
            "training_cutoff": origin.date().isoformat(), "n_training_rows": len(training),
        })
        print(f"Trained {config_id}: three beta-error models on {len(training)} realized errors.")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "file": str(csv_path.resolve()), "sha256": hashlib.sha256(snapshot).hexdigest(),
            "n_rows": len(data),
        },
        "features": list(FEATURES), "targets": list(TARGETS),
        "error_definition": "actual_beta - AR_forecast_beta",
        "prediction_units": "beta factor units (percentage points)",
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
        "seed": seed,
        "versions": {"python": platform.python_version(), **{
            package: version(package) for package in ("numpy", "pandas", "xgboost", "scikit-learn")
        }},
        "configurations": configurations,
        "data_note": "Historical macro features come from the existing revised-data snapshot.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(parameter_rows).to_csv(output_dir / "parameters.csv", index=False)
    predictions = pd.concat([
        predict_errors(config_id, feature_rows, output_dir) for config_id in configurations
    ], ignore_index=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False, date_format="%Y-%m-%d")
    print(f"Forecast origin: {origin.date()}; saved {len(predictions)} error-prediction rows to {output_dir.resolve()}")
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--forecast-origin", help="Origin month-end; defaults to the latest origin in the CSV.")
    parser.add_argument("--max-depths", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--n-estimators", nargs="+", type=int, default=[100, 150, 200])
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_error_models(
        csv_path=args.data, output_dir=args.output_dir, forecast_origin=args.forecast_origin,
        max_depths=args.max_depths, n_estimators=args.n_estimators,
        learning_rate=args.learning_rate, horizon=args.horizon, seed=args.seed,
    )


if __name__ == "__main__":
    main()
