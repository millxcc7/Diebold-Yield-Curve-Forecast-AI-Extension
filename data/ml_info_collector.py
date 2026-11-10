import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main_model"))
from nelson_siegel import COLUMNS, estimate_betas
from ar_forecast import forecast_betas
from baselinemodel_run import pull_yields
import os
from dotenv import load_dotenv
load_dotenv()


# Steps 1 to 3 of the ML process combined:
# AR forecast errors for every origin month + the features known at that origin
def ml_info_collector(base_month: str, api_key: str, first_origin: str = "2001-01-31", h: int = 12) -> pd.DataFrame:
    # Inputs:
    # base_month: baseline date x as a month-end date "YYYY-MM-DD" (e.g. "2015-01-31")
    #    notes: only data up to and including base_month is used (no look-ahead)
    # api_key: FRED API key (used to pull the yields)
    # first_origin: first forecast origin month (default "2001-01-31")
    # h: forecast horizon in months (default 12)

    # Returns:
    # table with one row per (origin, k):
    # columns are: origin, k, target, forecast/actual/error for beta1-3, betas at origin,
    # inflation, unemployment change, unemployment rate,fed funds change (3m, 6m), sentiment


    # CHECKS BEFORE COLLECTING ------------------
    # Check if base month is a month-end date
    base = pd.Timestamp(base_month)
    if not base.is_month_end:
        raise ValueError(f"{base_month} is not a month-end date, please adjust base_month parameter")


    # LOADING THE DATA -----------------------
    # Pull all yields from 1996 (the first origins need earlier history to fit the AR)
    yields = pull_yields(START="1996-01-01", api_key=api_key)
    yields = yields[COLUMNS]

    # Only keep yields up to and including the base month (no look-ahead)
    yields = yields.loc[:base]

    # Check if base month is in the yield data and nothing is missing
    if base not in yields.index:
        raise ValueError(f"{base_month} is not in the yield data, please adjust base_month parameter")
    if yields.isna().any().any():
        raise ValueError("Yield data has missing values")

    # Macro data (inflation, unemployment, fed funds) and monthly FOMC sentiment
    macro = pd.read_csv("data/macro.csv", index_col="date", parse_dates=True)
    sentiment = pd.read_csv("data/fomc_monthly_sentiment.csv", index_col="month", parse_dates=True)


    # FORECASTING STEPS -----------------------
    # Estimate the 3 betas for every month-end once (each month is its own OLS)
    # factors is a DataFrame with rows = month-end dates, columns = beta1, beta2, beta3
    factors, fitted = estimate_betas(yields)

    rows = [] # temp list to store every (origin, k) row, turned into a DataFrame at the end

    # Loop through every origin month from first_origin to the base month (using the factors df)
    for current in factors.loc[first_origin:base].index:

        # Forecast betas for current + 1 ... current + h using only betas up to current
        # forc_betadf is a DataFrame with rows = target months, columns = forc. beta1, forc. beta2, forc. beta3
        forc_betadf = forecast_betas(factors.loc[:current], h)

        # FEATURES AT THE ORIGIN (same for all k rows of this origin) ----
        # Macro data for month current is not released yet at current -> use last month's values
        prev = current - pd.offsets.MonthEnd(1)       # 1 month before current (used bc some this month data haven't been released)
        prev_3m = current - pd.offsets.MonthEnd(4)    # 3 months before prev
        prev_6m = current - pd.offsets.MonthEnd(7)    # 6 months before prev

        inflation = macro.loc[prev, "inflation"] # not looking at change becasue because already core CPI YoY %
        unemployment_chg = macro.loc[prev, "unemployment"] - macro.loc[prev_3m, "unemployment"]  # 3m change of unemployment
        unemployment = macro.loc[prev, "unemployment"] # current unemployment rate
        fed_funds_chg_3m = macro.loc[prev, "fed_funds"] - macro.loc[prev_3m, "fed_funds"] # 3m change of fed funds
        fed_funds_chg_6m = macro.loc[prev, "fed_funds"] - macro.loc[prev_6m, "fed_funds"] # 6m change of fed funds
        fed_funds = macro.loc[prev, "fed_funds"] # current fed funds rate

        # Sentiment is already assigned to the first month-end after each statement -> use current
        sentiment_score = sentiment.loc[current, "sentiment_score"]

        # Betas at the origin (also an input to the ML model)
        origin_beta = factors.loc[current]

        # Loop for each of the h months ahead (this is to update the target month column)
        # between 1 to h (12)
        for k in range(1, h + 1):
            target = forc_betadf.index[k - 1] # the target month-end date (current + k months)

            # Stop if the target month is after the base month (has not happened yet)
            if target > base:
                break

            forecast = forc_betadf.loc[target] # forecasted betas for the target month
            actual = factors.loc[target]       # actual betas for the target month

            # Adding the row: error = actual - forecast
            rows.append({
                "origin": current, "k": k, "target": target,
                "forecast_beta1": forecast["beta1"], "actual_beta1": actual["beta1"], "error_beta1": actual["beta1"] - forecast["beta1"],
                "forecast_beta2": forecast["beta2"], "actual_beta2": actual["beta2"], "error_beta2": actual["beta2"] - forecast["beta2"],
                "forecast_beta3": forecast["beta3"], "actual_beta3": actual["beta3"], "error_beta3": actual["beta3"] - forecast["beta3"],
                "beta1_origin": origin_beta["beta1"], "beta2_origin": origin_beta["beta2"], "beta3_origin": origin_beta["beta3"],
                "inflation": inflation,
                "unemployment_chg": unemployment_chg,
                "unemployment_rate": unemployment,
                "fed_funds_chg_3m": fed_funds_chg_3m,
                "fed_funds_chg_6m": fed_funds_chg_6m,
                "fed_funds_rate": fed_funds,
                "sentiment_score": sentiment_score,
            })

    # Convert all rows into the final DataFrame
    final = pd.DataFrame(rows).round(3) # round to 3 decimals
    return final


if __name__ == "__main__":
    result = ml_info_collector("2015-01-31", api_key = os.getenv("FRED_API_KEY")) # PLEASE INPUT THE FRED API KEY
    result.to_csv("data/MLdatabase.csv", index=False) # printing out to csv  

    print(result.head(13).round(3))   # first origin's 12 rows + the start of the next
    print("\nRows:", len(result))

