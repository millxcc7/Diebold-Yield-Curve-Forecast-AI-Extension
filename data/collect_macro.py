import os
import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred


# inflation, unemployment and fed funds rates -> FRED series id
SERIES = {
    "inflation":    "CPILFESL",   # core CPI (excludes food and energy)
    "unemployment": "UNRATE",
    "fed_funds":    "FEDFUNDS",
}

START = "1996-01-01"


def main():
    load_dotenv()
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))

    # empty table: rows = every month-end from 1996 to the last finished month,
    # columns = one per series
    today = pd.Timestamp.today().normalize()
    last_month_end = today if today.is_month_end else today - pd.offsets.MonthEnd(1)
    months = pd.date_range(START, last_month_end, freq="ME")
    macro_df = pd.DataFrame(index=months, columns=list(SERIES), dtype=float)
    macro_df.index.name = "date"

    for name, sid in SERIES.items():
        if name == "inflation":
            # start in 1995 so the year-over-year % has a value for 1996-01
            s = fred.get_series(sid, observation_start="1995-01-01")
        else:
            s = fred.get_series(sid, observation_start=START)

        # FRED labels months by the 1st -> move to month-end to match the yields
        s.index = s.index + pd.offsets.MonthEnd(0)

        # convert core CPI -> year over year inflation rate (in %)
        if name == "inflation":
            s = round(s.pct_change(12) * 100, 2)

        # fill this series' column (dates outside 1996-now, like 1995, are dropped)
        macro_df[name] = s

    # end the table at the most recent month-end where every series has a value
    complete = macro_df.dropna().index
    macro_df = macro_df.loc[:complete[-1]]

    # list every missing value as (date, series)
    # missing = [(date.strftime("%Y-%m-%d"), col)
    #           for col in macro_df.columns
    #           for date in macro_df.index[macro_df[col].isna()]]
    #if missing:
    #    print("Missing values (filled with the month before):", missing)

    # fill each missing value with the month before (usually missing for special reasons)
    macro_df = macro_df.ffill()

    os.makedirs("data", exist_ok=True)
    macro_df.to_csv("data/macro.csv")
    print(f"Saved data/macro.csv: {macro_df.index[0].date()} to {macro_df.index[-1].date()}")

if __name__ == "__main__":
    main()