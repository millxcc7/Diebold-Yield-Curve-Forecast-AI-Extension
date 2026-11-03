
"""
validate_rates.py
-----------------
Compare Gemini FOMC sentiment scores against actual
Federal Reserve policy-rate changes.

Run from the repository root:
    python3 -m fomc_sentiment_analysis.validate_rates

Output:
    data/fomc_rate_validation.csv
"""

from io import StringIO
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
SCORES_PATH = ROOT / "data" / "fomc_scores.csv"
OUTPUT_PATH = ROOT / "data" / "fomc_rate_validation.csv"

# FRED series:
# Before Dec 16, 2008: single federal funds target.
# From Dec 16, 2008: upper and lower bounds of target range.
SERIES = ["DFEDTAR", "DFEDTARU", "DFEDTARL"]
SWITCH_DATE = pd.Timestamp("2008-12-16")


# 1. Download official policy-rate data

def download_fred(series_id):
    """Download one daily policy-rate series from FRED."""

    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    response = requests.get(
        url,
        params={"id": series_id},
        timeout=60,
    )
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    df.columns = ["date", series_id]

    df["date"] = pd.to_datetime(df["date"])
    df[series_id] = pd.to_numeric(
        df[series_id], errors="coerce"
    )

    return df.set_index("date").sort_index()


def load_policy_rates():
    """Construct a consistent historical policy-rate series."""

    rates = None

    for series_id in SERIES:
        data = download_fred(series_id)

        if rates is None:
            rates = data
        else:
            rates = rates.join(data, how="outer")

    # Target-range midpoint after December 2008
    midpoint = (
        rates["DFEDTARU"] + rates["DFEDTARL"]
    ) / 2

    rates["policy_rate"] = rates["DFEDTAR"]

    rates.loc[
        rates.index >= SWITCH_DATE, "policy_rate"
    ] = midpoint.loc[rates.index >= SWITCH_DATE]

    # Fill calendar dates using the last published target.
    full_index = pd.date_range(
        rates.index.min(),
        rates.index.max(),
        freq="D",
    )

    rates = rates.reindex(full_index)
    rates.index.name = "date"
    rates["policy_rate"] = rates["policy_rate"].ffill()

    return rates[["policy_rate"]]


# 2. Match policy-rate changes to FOMC statements

def match_rate_changes(scores, rates):
    """Calculate rate changes around each statement release."""

    scores = scores.copy()
    scores["release_date"] = pd.to_datetime(
        scores["release_date"]
    )

    policy = rates["policy_rate"]

    # Compare the day before the announcement with the day
    # after, since some target changes take effect next day.
    before_dates = scores["release_date"] - pd.Timedelta(days=1)
    after_dates = scores["release_date"] + pd.Timedelta(days=1)

    scores["rate_before"] = policy.reindex(
        pd.DatetimeIndex(before_dates)
    ).to_numpy()

    scores["rate_after"] = policy.reindex(
        pd.DatetimeIndex(after_dates)
    ).to_numpy()

    scores["rate_change_bps"] = (
        (scores["rate_after"] - scores["rate_before"]) * 100
    ).round(2)

    # December 2008 changed from a point target to a range.
    # Its midpoint change is not a standard-sized rate cut.
    scores["target_type_transition"] = (
        scores["release_date"] == SWITCH_DATE
    )

    return scores


# 3. Compare sentiment with rate decisions

def classify_direction(row):
    """Compare Gemini sentiment with the rate-change direction."""

    score = row["score"]
    change = row["rate_change_bps"]

    if pd.isna(change) or pd.isna(score):
        return "missing_data"

    if change == 0:
        return "unchanged_rate"

    if score == 0:
        return "neutral_on_rate_change"

    if score * change > 0:
        return "aligned"

    return "opposite"


def validate(scores):
    """Report agreement and correlation statistics."""

    print("\n=== 1. RATE VALIDATION SUMMARY ===")

    print("Statements:", len(scores))
    print("Missing rate data:",
          scores["rate_change_bps"].isna().sum())

    print("\nDirection classifications:")
    print(scores["comparison"].value_counts())

    changed = scores[
        scores["rate_change_bps"].notna()
        & (scores["rate_change_bps"] != 0)
        & ~scores["target_type_transition"]
    ]

    nonneutral = changed[changed["score"] != 0]

    print("\n=== 2. DIRECTION AGREEMENT ===")

    print("Meetings with rate changes:", len(changed))
    print("With non-neutral sentiment:", len(nonneutral))

    if len(nonneutral):
        agreement = (
            nonneutral["comparison"] == "aligned"
        ).mean()

        print(f"Direction agreement: {agreement:.1%}")
    else:
        print("Direction agreement: N/A")

    print("\n=== 3. CORRELATION ===")

    # Exclude the Dec 2008 target-definition transition.
    correlation_data = scores[
        scores["rate_change_bps"].notna()
        & ~scores["target_type_transition"]
    ]

    if (
        len(correlation_data) >= 3
        and correlation_data["score"].nunique() > 1
        and correlation_data["rate_change_bps"].nunique() > 1
    ):
        correlation = correlation_data["score"].corr(
            correlation_data["rate_change_bps"]
        )
        print(f"Pearson correlation: {correlation:.3f}")
    else:
        print("Pearson correlation: N/A")

    print("\n=== 4. DISAGREEMENTS ===")

    disagreements = scores[
        scores["comparison"] == "opposite"
    ]

    if len(disagreements):
        print(disagreements[
            ["release_date", "score", "rate_change_bps"]
        ].to_string(index=False))
    else:
        print("No opposite-direction cases.")

    print("\n=== 5. UNCHANGED-RATE MEETINGS ===")

    unchanged = scores[
        scores["comparison"] == "unchanged_rate"
    ]

    print("Count:", len(unchanged))
    print("\nSentiment distribution:")
    print(unchanged["score"].value_counts().sort_index())

    print("\n=== 6. EXAMPLE STATEMENTS ===")

    examples = [
        "2008-12-16",
        "2019-07-31",
        "2022-06-15",
    ]

    selected = scores[
        scores["release_date"].isin(
            pd.to_datetime(examples)
        )
    ]

    print(selected[
        [
            "release_date",
            "score",
            "rate_change_bps",
            "comparison",
        ]
    ].to_string(index=False))


def main():
    print("Loading Gemini scores...")
    scores = pd.read_csv(SCORES_PATH)

    print("Downloading Federal Reserve rate data...")
    rates = load_policy_rates()

    print("Matching statements with rate changes...")
    results = match_rate_changes(scores, rates)

    results["comparison"] = results.apply(
        classify_direction, axis=1
    )

    validate(results)

    # Save detailed results without changing original scores.
    results["release_date"] = results[
        "release_date"
    ].dt.strftime("%Y-%m-%d")

    results.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved validation results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
