import os
import pandas as pd
from backtest_baseline import calculate_errors, compare_models, MATURITIES, RESULTS_DIR

# Purpose: to score every ML version against the actual yields 
# This saves the results to <scenario>/ml/ml_analysis.csv
def ml_analysis(scenarios=("financial_crisis_2008", "economic_recovery_2014", "low_volatility_2017",
                           "normal_policy_adjust_2019", "covid_2020", "rate_hikes_2022", "holdout_2025")):
    # input: list of scenarios to analyze  

    # Loop through the scenarios and compare each ML version to baseline
    for k in scenarios:

        folder = os.path.join(RESULTS_DIR, k)
        corrected_path = os.path.join(folder, "ml", "corrected", "corrected_yields.csv")

        # Skip scenarios that don't have ML results yet
        if not os.path.exists(corrected_path):
            print(f"Skipping {k}: no ml/corrected/corrected_yields.csv")
            continue

        # Actual and baseline yields (rows = dates, columns = maturities)
        actual = pd.read_csv(os.path.join(folder, "baseline_actual_yields.csv"), index_col=0, parse_dates=True)
        baseline = pd.read_csv(os.path.join(folder, "baseline_predicted_yields.csv"), index_col=0, parse_dates=True)
        baseline_errors = calculate_errors(baseline, actual)

        # All 6 ML versions stacked in one file
        corrected = pd.read_csv(corrected_path, parse_dates=["target"])

        all_versions = []  # temp list to store each ML version's metrics table

        # Loop through the different ML versions
        for version in corrected["config_id"].unique():

            # The 12 rows for this version, shaped like actual (rows = target dates, columns = maturities)
            predicted = corrected.loc[corrected["config_id"] == version].set_index("target")[MATURITIES]
            predicted.index.name = None # removing the label row

            # Check the forecast months match the actual months before comparing
            if not predicted.index.equals(actual.index):
                raise ValueError(f"{k} / {version}: forecast dates don't match the actual yield dates")

            errors = calculate_errors(predicted, actual)

            # compare_models labels its two inputs "model" and "benchmark"; rename to "ml" and "baseline"
            metrics = compare_models(errors, baseline_errors)
            metrics = metrics.rename(columns=lambda c: c.replace("model_", "ml_").replace("benchmark_", "baseline_"))


            # Preparing to stack the ML version metrics table 
            metrics = metrics.reset_index() # add index back
            metrics.insert(0, "model_version", version) #add the ml version index
            all_versions.append(metrics) # save it

        # Combine the 6 versions and save
        analysis = pd.concat(all_versions, ignore_index=True)
        output_path = os.path.join(folder, "ml", "ml_analysis.csv")
        analysis.to_csv(output_path, index=False)

        print(f"\n{k}: saved {output_path}")
        print(analysis.round(2).to_string(index=False))


if __name__ == "__main__":
    ml_analysis()
