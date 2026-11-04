"""Convert FOMC scores to sentiment available at each Treasury month-end.

Run from the repository root:
    python3 -B -m fomc_sentiment_analysis.monthly_sentiment
    python3 -B -m fomc_sentiment_analysis.monthly_sentiment --yields path/to/yields.csv
    python3 -B -m fomc_sentiment_analysis.monthly_sentiment --start 1995-12-01 --end 2026-09-30

"""

import argparse
from bisect import bisect_right
import calendar
import csv
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCORES_PATH = ROOT / "data" / "fomc_scores.csv"
YIELDS_PATH = ROOT / "data" / "treasury_yields_1996_2026.csv"
OUTPUT_PATH = ROOT / "data" / "fomc_monthly_sentiment.csv"
# Match START and END in data/pull_yields.py if the yield CSV is unavailable.
DEFAULT_START = date(1996, 1, 1)
DEFAULT_END = date(2026, 9, 30)
VALID_SCORES = {-1.0, -0.5, 0.0, 0.5, 1.0}
COLUMNS = [
    "month",
    "sentiment_score",
    "statement_date",
    "new_statement",
    "months_since_statement",
]


def read_scores(path):
    # Read release dates and scores without invoking the scoring model.
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not {"release_date", "score"}.issubset(reader.fieldnames or []):
            raise ValueError("Scores CSV must contain release_date and score.")
        return [
            (date.fromisoformat(row["release_date"]), float(row["score"]))
            for row in reader
        ]


def read_yield_months(path):
    # Read the yield CSV's date column, retaining its exact set of months.
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if "date" not in (reader.fieldnames or []):
            raise ValueError("Monthly yields CSV must contain a date column.")
        return [date.fromisoformat(row["date"]) for row in reader]


def calendar_months(start, end):
    # Return month-ends within an inclusive date range.
    if start > end:
        raise ValueError("Start date must be on or before end date.")
    months = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        month_end = date(year, month, calendar.monthrange(year, month)[1])
        if start <= month_end <= end:
            months.append(month_end)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def build_monthly_sentiment(scores, months):
    """Select only releases <= each month-end, carrying the latest forward.

    scores is an iterable of (release_date, score) pairs; 
    months contains month-end dates. 
    Dates are datetime.date objects. 
    Age is the number of calendar months since release (zero in the release month), not row count.
    Missing scores and ambiguous duplicate release dates fail validation
    rather than silently substituting an older or future statement.
    """
    scores = sorted(scores)
    release_dates = [release for release, _ in scores]
    if len(set(release_dates)) != len(release_dates):
        raise ValueError("Duplicate release dates: select one score per statement.")
    if any(score not in VALID_SCORES for _, score in scores):
        raise ValueError("Scores must be non-missing and one of -1, -0.5, 0, 0.5, 1.")

    months = sorted(months)
    if not months or len(set(months)) != len(months):
        raise ValueError("Monthly dates must be non-empty and unique.")
    if any(day.day != calendar.monthrange(day.year, day.month)[1] for day in months):
        raise ValueError("Monthly dates must all be calendar month-ends.")

    rows = []
    for month in months:
        # bisect_right includes releases on month-end and excludes every later one.
        position = bisect_right(release_dates, month) - 1
        statement_date, score = scores[position] if position >= 0 else (None, None)
        age = None
        if statement_date is not None:
            age = (month.year - statement_date.year) * 12 + month.month - statement_date.month
        rows.append({
            "month": month,
            "sentiment_score": score,
            "statement_date": statement_date,
            "new_statement": age == 0,
            "months_since_statement": age,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=SCORES_PATH)
    parser.add_argument("--yields", type=Path, help="Monthly yields CSV with a date column.")
    parser.add_argument("--start", type=date.fromisoformat, help="Calendar start (YYYY-MM-DD).")
    parser.add_argument("--end", type=date.fromisoformat, help="Calendar end (YYYY-MM-DD).")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    if args.yields is not None and (args.start is not None or args.end is not None):
        parser.error("Use either --yields or --start/--end.")

    if args.start is not None or args.end is not None:
        months = calendar_months(args.start or DEFAULT_START, args.end or DEFAULT_END)
    elif args.yields is not None or YIELDS_PATH.exists():
        months = read_yield_months(args.yields if args.yields is not None else YIELDS_PATH)
    else:
        print(
            "Monthly yield CSV unavailable; using the configured yield calendar "
            f"{DEFAULT_START} through {DEFAULT_END}."
        )
        months = calendar_months(DEFAULT_START, DEFAULT_END)

    rows = build_monthly_sentiment(read_scores(args.scores), months)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} months ({rows[0]['month']} to {rows[-1]['month']}) to {args.output}")


if __name__ == "__main__":
    main()
