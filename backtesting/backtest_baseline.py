
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(
    0, os.path.join(PROJECT_ROOT, "main_model")
)

from baselinemodel_run import baseline_main, pull_yields


# Replace with the FRED API key when running locally.
API_KEY = "5eb22df42d3f41aefa70fc73d4dd7e50"


MATURITIES = ["3", "6", "12", "24", "36", "60", "84", "120"]

# Historical forecasting experiments.
# Each entry contains (base_month, forecast_horizon).
TEST_CASES = {
    "holdout_2025": ("2025-09-30", 12),
    "financial_crisis_2008": ("2008-08-31", 12),
    "covid_2020": ("2020-02-29", 12),
    "rate_hikes_2022": ("2021-12-31", 12),
    # Non-crisis periods
    "economic_recovery_2014": ("2013-12-31", 12),
    "low_volatility_2017": ("2016-12-31", 12),
    "normal_policy_adjust_2019": ("2018-12-31", 12),
}

RESULTS_DIR = os.path.join(
    PROJECT_ROOT, "backtesting", "results"
)

MODEL_NAME = "baseline"

# Cleans downloaded Treasury yield data for backtesting
def prepare_data(data):

    # Copy the data to avoid modifying the original.
    data = data.copy()
    data.columns = data.columns.map(str)
    data = data[MATURITIES]
    data = data.apply(pd.to_numeric, errors="coerce")
    data.index = pd.to_datetime(data.index)
    data.index = (
        data.index.to_period("M").to_timestamp("M")
    )

    data = data.sort_index()

    # Check for duplicate months.
    if data.index.has_duplicates:
        raise ValueError("Duplicate monthly observations found.")

    # Remove incomplete observations.
    data = data.dropna()

    if data.empty:
        raise ValueError("No complete monthly yield data found.")

    # Check whether any months are missing after cleaning.
    expected = pd.date_range(
        start=data.index.min(),
        end=data.index.max(),
        freq="ME"
    )

    if not data.index.equals(expected):
        raise ValueError(
            "Missing months in yield data. "
            "Review the observations before forecasting."
        )

    return data


# Predicts future yields by assuming they stay the same as the last observed yields.
def no_change_benchmark(training, forecast_dates):

    # Assumption: yields remain unchanged from the base month.
    last_yields = training.iloc[-1]

    # Repeat the base-month yields for all forecast dates.
    benchmark = pd.DataFrame(
        [last_yields.to_numpy()] * len(forecast_dates),
        index=forecast_dates,
        columns=training.columns
    )

    return benchmark


# Simply calculate the difference between predicted and actual yields.
def calculate_errors(predicted, actual):

    errors_bps = (predicted - actual) * 100

    return errors_bps

# Calculatess RMSE, MAE, and bias to measure forecasting accuracy.
def calculate_metrics(errors_bps):

    # RMSE: penalizes large errors more heavily.
    rmse = np.sqrt(
        (errors_bps ** 2).mean(axis=0)
    )

    # MAE: average absolute forecasting error.
    mae = errors_bps.abs().mean(axis=0)

    # Bias: average signed error.
    # Positive means overprediction.
    # Negative means underprediction.
    bias = errors_bps.mean(axis=0)

    metrics = pd.DataFrame({
        "RMSE_bps": rmse,
        "MAE_bps": mae,
        "Bias_bps": bias
    })

    metrics.index.name = "maturity_months"

    return metrics


# Compares the baseline model's performance against the no-change benchmark.
def compare_models(model_errors, benchmark_errors):

    model_metrics = calculate_metrics(model_errors)
    benchmark_metrics = calculate_metrics(benchmark_errors)

    comparison = pd.DataFrame(index=model_metrics.index)

    # RMSE comparison
    comparison["model_RMSE_bps"] = model_metrics["RMSE_bps"]
    comparison["benchmark_RMSE_bps"] = benchmark_metrics["RMSE_bps"]

    # MAE comparison
    comparison["model_MAE_bps"] = model_metrics["MAE_bps"]
    comparison["benchmark_MAE_bps"] = benchmark_metrics["MAE_bps"]

    # Bias comparison
    comparison["model_Bias_bps"] = model_metrics["Bias_bps"]
    comparison["benchmark_Bias_bps"] = benchmark_metrics["Bias_bps"]

    # Positive improvement: baseline performs better.
    # Negative improvement: benchmark performs better.
    comparison["RMSE_improvement_pct"] = (
        (comparison["benchmark_RMSE_bps"] - comparison["model_RMSE_bps"])
        / comparison["benchmark_RMSE_bps"].replace(0, np.nan)
        * 100
    )

    comparison.index.name = "maturity_months"

    return comparison

# Runs one historical forecasting experiment, evaluates accuracy, and saves CSV results.
def run_backtest(data, name, base_month, h):

    print(f"\nRunning backtest: {name}")

    base_date = pd.Timestamp(base_month)

  
    training = data.loc[:base_date].copy()

    if training.empty or training.index[-1] != base_date:
        raise ValueError(
            f"Base month unavailable: {base_month}"
        )

    # Run the existing Nelson-Siegel + AR forecasting model.
    # This function receives historical data and does not
    # need to download anything from FRED.
    predicted = baseline_main(training, base_month, h)

    # Standardize the prediction table.
    predicted = predicted.copy()

    predicted.index = pd.to_datetime(predicted.index)
    predicted.index = (predicted.index.to_period("M").to_timestamp("M"))

    predicted.columns = predicted.columns.map(str)
    predicted = predicted[MATURITIES]

    # Verify that we forecast the requested future months.
    expected_dates = pd.date_range(start=base_date + pd.offsets.MonthEnd(1), periods=h, freq="ME")

    if not predicted.index.equals(expected_dates):
        raise ValueError(f"Unexpected forecast dates for {name}")

    if predicted.isna().any().any():
        raise ValueError(f"Missing predicted yields for {name}")

    # Retrieve actual future yields from the downloaded data.

    actual = data.reindex(predicted.index)[MATURITIES]

    if actual.isna().any().any():
        raise ValueError(f"Actual future yields unavailable for {name}")

    # Generate no-change benchmark predictions.
    benchmark = no_change_benchmark(training, predicted.index)

    # Calculate forecast errors.
    model_errors = calculate_errors(predicted, actual)

    benchmark_errors = calculate_errors(benchmark, actual)

    # Compare model performance.
    comparison = compare_models(model_errors, benchmark_errors)

    # Create a results folder for this experiment.
    folder = os.path.join(RESULTS_DIR, name)
    os.makedirs(folder, exist_ok=True)

    # Save forecasted, actual, and benchmark yields.
    predicted.to_csv(f"{folder}/{MODEL_NAME}_predicted_yields.csv")

    actual.to_csv(f"{folder}/{MODEL_NAME}_actual_yields.csv")

    benchmark.to_csv(f"{folder}/{MODEL_NAME}_benchmark_yields.csv")

    # Save forecasting errors.
    model_errors.to_csv(f"{folder}/{MODEL_NAME}_model_errors_bps.csv")

    benchmark_errors.to_csv(f"{folder}/{MODEL_NAME}_benchmark_errors_bps.csv")

    # Save performance comparison.
    comparison.to_csv(f"{folder}/{MODEL_NAME}_metrics_comparison.csv")

    print("\nPerformance metrics (bps):")
    print(comparison.round(2))

    # Add identifiers for future LLM and ML comparisons.
    comparison = comparison.reset_index()

    comparison.insert(0, "model", MODEL_NAME)
    comparison.insert(1, "experiment", name)

    return comparison

def main():

    data = pull_yields(api_key=API_KEY)
    data = prepare_data(data)

    # Store each experiment's performance results.
    all_results = []

    # Loop through the four historical events.
    for name, (base_month, h) in TEST_CASES.items():

        comparison = run_backtest(data=data, name=name, base_month=base_month, h=h)

        all_results.append(comparison)

    # Combine performance across all events.
    summary = pd.concat(all_results, ignore_index=True)

    # Create results folder.
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save combined baseline results.
    summary.to_csv(f"{RESULTS_DIR}/baseline_all_experiments_summary.csv", index=False)

    print("\nAll baseline backtests completed.")

if __name__ == "__main__":
    main()
    