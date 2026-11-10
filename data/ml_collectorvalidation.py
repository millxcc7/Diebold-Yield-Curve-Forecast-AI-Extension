import pandas as pd


# Load the ML database and the source CSVs
db = pd.read_csv("data/MLdatabase.csv", parse_dates=["origin"])
macro = pd.read_csv("data/macro.csv", index_col="date", parse_dates=True)
sentiment = pd.read_csv("data/fomc_monthly_sentiment.csv", index_col="month", parse_dates=True)

# Same dates the collector used: macro is lagged one month from the origin
prev = db["origin"] - pd.offsets.MonthEnd(1)
prev_3m = db["origin"] - pd.offsets.MonthEnd(4)
prev_6m = db["origin"] - pd.offsets.MonthEnd(7)


# MACRO CHECK: macro.csv - MLdatabase (should all be around 0) ---------------
diff = pd.DataFrame({
    "inflation":         macro.loc[prev, "inflation"].values - db["inflation"],
    "unemployment_rate": macro.loc[prev, "unemployment"].values - db["unemployment_rate"],
    "unemployment_chg":  (macro.loc[prev, "unemployment"].values - macro.loc[prev_3m, "unemployment"].values) - db["unemployment_chg"],
    "fed_funds_rate":    macro.loc[prev, "fed_funds"].values - db["fed_funds_rate"],
    "fed_funds_chg_3m":  (macro.loc[prev, "fed_funds"].values - macro.loc[prev_3m, "fed_funds"].values) - db["fed_funds_chg_3m"],
    "fed_funds_chg_6m":  (macro.loc[prev, "fed_funds"].values - macro.loc[prev_6m, "fed_funds"].values) - db["fed_funds_chg_6m"],
})



# SENTIMENT CHECK: fomc_monthly_sentiment.csv - MLdatabase (should be ~0) ----
sentiment_diff = sentiment.loc[db["origin"], "sentiment_score"].values - db["sentiment_score"]



# FORECAST BETA CHECK: rerun the AR forecast - MLdatabase (should be ~0) -----
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main_model"))
from nelson_siegel import COLUMNS, estimate_betas
from ar_forecast import forecast_betas
from baselinemodel_run import pull_yields

load_dotenv()
base = pd.Timestamp("2015-01-31") # same base month used to build MLdatabase.csv

# Estimate the betas for every month-end up to the base month (same as the collector)
yields = pull_yields(START="1996-01-01", api_key=os.getenv("FRED_API_KEY"))[COLUMNS].loc[:base] # MAKE SURE TO UPDATE THIS WHEN RUNNING THE CHECK
factors, fitted = estimate_betas(yields)

# Empty table to store every origin's differences (rows = origin + target month)
fccheckdf = pd.DataFrame()

# Check one origin date at a time
for origin in ["2005-03-31", "2014-10-31", "2012-01-31"]:
    o = pd.Timestamp(origin)

    # Forecast betas for the next 12 months using only betas up to the origin
    forc_betadf = forecast_betas(factors.loc[:o], 12)

    # The MLdatabase rows for this origin (rows = target months)
    db_rows = db[db["origin"] == o].set_index(pd.to_datetime(db.loc[db["origin"] == o, "target"]))

    # Only compare target months that are in MLdatabase (targets after the base month are not)
    forc_betadf = forc_betadf.loc[db_rows.index]

    # Difference: rerun forecast - MLdatabase forecast
    beta_diff = pd.DataFrame({
        "beta1": forc_betadf["beta1"] - db_rows["forecast_beta1"],
        "beta2": forc_betadf["beta2"] - db_rows["forecast_beta2"],
        "beta3": forc_betadf["beta3"] - db_rows["forecast_beta3"],
    })

    # Add this origin's differences to fccheckdf
    beta_diff.insert(0, "origin", o)
    fccheckdf = pd.concat([fccheckdf, beta_diff])

print(fccheckdf.round(4))

