import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main_model.nelson_siegel import COLUMNS, MATURITIES, estimate_betas
from main_model.ar_forecast import forecast_betas, forecast_yield
from main_model.baselinemodel_run import pull_yields

DEFAULT_YIELDS = ROOT / "data/error_correction/corrected_forecasts/baseline_yield_history.csv"
DEFAULT_YIELD_DATABASE = ROOT / "data/MLyielddatabase.csv"


# Same steps as ml_info_collector, with yield errors instead of beta errors.
def ml_yield_info_collector(base_month: str, api_key: str = None,
                           first_origin: str = "2001-01-31", h: int = 12,
                           yields_csv=DEFAULT_YIELDS) -> pd.DataFrame:
    # Each row is one (origin, k, target), containing eight maturities and
    # the yield/macro features known at the origin. Yields are percentage points.
    base = pd.Timestamp(base_month)
    if not base.is_month_end:
        raise ValueError(f"{base_month} is not a month-end date.")
    if not 1 <= h <= 12:
        raise ValueError("h must be between 1 and 12 months.")

    # LOADING THE DATA
    # Reuse the local yield snapshot; pass yields_csv=None to download from FRED.
    if yields_csv is None:
        yields = pull_yields(START="1996-01-01", api_key=api_key)
    else:
        yields = pd.read_csv(yields_csv, index_col="date", parse_dates=True)
    yields = yields.loc["1996-01-31":base, COLUMNS]

    if base not in yields.index:
        raise ValueError(f"{base_month} is not in the yield data.")
    if yields.isna().any().any():
        raise ValueError("Yield data has missing values.")

    macro = pd.read_csv(ROOT / "data/macro.csv", index_col="date", parse_dates=True)
    sentiment = pd.read_csv(ROOT / "data/fomc_monthly_sentiment.csv", index_col="month", parse_dates=True)

    # FORECASTING STEPS
    factors, _ = estimate_betas(yields)
    rows = []

    for current in factors.loc[first_origin:base].index:
        if current == base:
            break  # No future targets have been observed yet at the final month.
        forecast = forecast_betas(factors.loc[:current], h)

        # Same feature timing as ml_info_collector.
        prev = current - pd.offsets.MonthEnd(1)
        prev_3m = current - pd.offsets.MonthEnd(4)
        inflation = macro.loc[prev, "inflation"]
        unemployment_chg = macro.loc[prev, "unemployment"] - macro.loc[prev_3m, "unemployment"]
        unemployment = macro.loc[prev, "unemployment"]
        fed_funds_chg_3m = macro.loc[current, "fed_funds"] - macro.loc[current - pd.offsets.MonthEnd(3), "fed_funds"]
        fed_funds_chg_6m = macro.loc[current, "fed_funds"] - macro.loc[current - pd.offsets.MonthEnd(6), "fed_funds"]
        fed_funds = macro.loc[current, "fed_funds"]
        sentiment_score = sentiment.loc[current, "sentiment_score"]

        for k in range(1, h + 1):
            target = forecast.index[k - 1]
            if target > base:
                break

            row = {"origin": current, "k": k, "target": target}
            for maturity in MATURITIES:
                # Apply the Nelson-Siegel equation to the forecast betas.
                predicted = forecast_yield(forecast, target, maturity)
                actual = yields.loc[target, str(maturity)]
                row[f"forecast_yield_{maturity}m"] = predicted
                row[f"actual_yield_{maturity}m"] = actual
                row[f"error_yield_{maturity}m"] = actual - predicted

            for maturity in MATURITIES:
                row[f"yield_{maturity}m_origin"] = yields.loc[current, str(maturity)]

            row.update({
                "inflation": inflation,
                "unemployment_chg": unemployment_chg,
                "unemployment_rate": unemployment,
                "fed_funds_chg_3m": fed_funds_chg_3m,
                "fed_funds_chg_6m": fed_funds_chg_6m,
                "fed_funds_rate": fed_funds,
                "sentiment_score": sentiment_score,
            })
            rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    result = ml_yield_info_collector("2026-09-30")
    result.to_csv(DEFAULT_YIELD_DATABASE, index=False)
    print(result.head(13).round(3))
    print("\nRows:", len(result))
