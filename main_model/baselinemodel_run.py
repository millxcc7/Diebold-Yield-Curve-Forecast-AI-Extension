import pandas as pd

from nelson_siegel import COLUMNS, load_yields, estimate_betas, calculate_errors, MATURITIES, LAMBDA
from ar_forecast import fit_ar_coef, forecast_betas, forecast_yield

# Building the Runable Function First 
# orecast the yield curve for the next h months from one base month
def baseline_main(yields: pd.DataFrame, base_month: str, h: int) -> pd.DataFrame:
    # Inputs:
    # yields: DataFrame containing monthly yield data (the CSV file from FRED)
    # base_month: base month as a month-end date "YYYY-MM-DD" (e.g. "2015-12-31")
    #    notes: base month must have existed (cannot be a future month)
    # h: forecast horizon in months 
    
    # Returns:
    # table of forecasted yields: rows = target months (base + 1 ... base + h),
    # columns = maturities ("3", "6", ..., "120")


    # CHECKS BEFORE THE FORECAST------------------
    # Check if h is at least 1 month
    if h < 1: 
        raise ValueError(f"Forecast horizon h must be at least 1 month, please adjust h parameter")

    # Check if base month is a month-end date that exists in the data before running forecast
    base = pd.Timestamp(base_month)
    if not base.is_month_end: # Checking if it is actually a valid month-end date
        raise ValueError(f"{base_month} is not a month-end date, please adjust base_month parameter")
    if base not in yields.index: # Checking if the base month is in the yield data
        raise ValueError(f"{base_month} is not in the csv_path data, please add or adjust csv_path/base_month parameter")

    
    # FORECASTING STEPS -----------------------
    # Estimate the 3 betas for every month (using OLS method)
    # factors is a DataFrame with rows = month-end dates, columns = est. beta1, est. beta2, est. beta3
    factors, fitted = estimate_betas(yields)

    # Only care about betas up to and including the base month (no look-ahead)
    est_betadf = factors.loc[:base]

    # Forecast betas for base month + 1 month ... base month + h months
    # forc_betadf is a DataFrame with rows = target months, columns = forc. beta1, forc. beta2, forc. beta3
    forc_betadf = forecast_betas(est_betadf, h)

    # Convert forecasted betas into forecasted yields using Diebold-Li Yield Curve Equation for each maturity
    # forecasted_yielddf is a DataFrame with rows = target months, columns = forecasted yields for each maturity
    forecasted_yielddf = pd.DataFrame(columns=COLUMNS, dtype=float) # empty DataFrame to store results

    # For loop through each of the h months to get the forecasted yields for each maturity
    for k in range(h):
        target_date = forc_betadf.index[k] # the target month-end date

        # For loop through each of the 8 maturities
        for m in range(len(MATURITIES)):    
            forc_h_yield = forecast_yield(forc_betadf, target_date, MATURITIES[m]) # Diebold-Li yield curve equation to get the forecasted yield for the target month and maturity
            forecasted_yielddf.loc[target_date, COLUMNS[m]] = forc_h_yield #adding the forecasted yield to the forecasted_yielddf dataframe

    return forecasted_yielddf


def main():

    # Read the csv
     data = pd.read_csv(r"C:\Users\milly\OneDrive\文档\Research\Fall 2026 FARMSA\treasury_yields_1996_2026.csv", index_col="date", parse_dates=True)

     # CHECKS and CLEANING DATA ----------------------
     # Check if all required columns are present in the data
     missing_columns = [col for col in COLUMNS if col not in data.columns]
     if missing_columns:
         raise ValueError(f"Missing columns in the data: {missing_columns}")

     data = data[COLUMNS]

     # check for missing values in the data
     if data.isna().any().any():
        raise ValueError("Yield data has missing values")

     # ACTUAL MODEL RUN ----------------------
     result = baseline_main(data, "2025-09-30", 12)
     current = data.loc["2025-10-31":"2026-09-30"]
     return result, current

results, current = main()
print("Forecasted yields:")
print(results.round(3))
print("\nActual yields:")
print(current.round(3))
