import os
import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()  # reads FRED_API_KEY from the .env file
api_key = os.getenv("FRED_API_KEY")
if not api_key:
    raise ValueError("FRED_API_KEY not found. Check your .env file.")

fred = Fred(api_key=api_key)

START, END = "1996-01-01", "2026-09-30"

# maturity in months -> FRED series id
SERIES = {
    3: "DGS3MO", 6: "DGS6MO", 12: "DGS1",
    24: "DGS2", 36: "DGS3", 60: "DGS5", 84: "DGS7",
    120: "DGS10", 240: "DGS20", 360: "DGS30",
}

# pull daily data for every maturity
daily = pd.DataFrame({
    m: fred.get_series(sid, observation_start=START, observation_end=END)
    for m, sid in SERIES.items()
})
daily.index.name = "date"

# convert to month-end snapshots
monthly = daily.resample("ME").last()  # if you get an error, change "ME" to "M"
monthly.index.name = "date"

# save
os.makedirs("data", exist_ok=True)
daily.to_csv("data/yields_daily.csv")
monthly.to_csv("data/treasury_yields_1996_2026.csv")

# tests
print("CSV saved. Monthly shape (rows, maturities):", monthly.shape)  # expect (369, 10)
print("\nMissing values per maturity:")
print(monthly.isna().sum())
print("\nFirst available date per maturity:")
print(monthly.apply(lambda s: s.first_valid_index()))
print("\nLast 3 rows:")
print(monthly.tail(3))
