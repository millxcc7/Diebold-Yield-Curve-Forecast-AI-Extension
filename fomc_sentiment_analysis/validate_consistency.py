
"""
Re-score 20 historical statements and compare with original scores.

Run:
    python3 -m fomc_sentiment_analysis.validate_consistency
"""

import time
from pathlib import Path

import pandas as pd

from .fomc_lib import score_statement

ROOT = Path(__file__).resolve().parents[1]
STATEMENTS_PATH = ROOT / "data" / "fomc_statements.csv"
SCORES_PATH = ROOT / "data" / "fomc_scores.csv"
OUTPUT_PATH = ROOT / "data" / "fomc_consistency.csv"


def main():
    statements = pd.read_csv(STATEMENTS_PATH)
    scores = pd.read_csv(SCORES_PATH)

    # Reproducible random sample of 20 statements
    sample = scores.sample(n=20, random_state=42)

    sample = sample.merge(
        statements[["release_date", "text"]],
        on="release_date",
        validate="one_to_one",
    )

    if OUTPUT_PATH.exists():
        results = pd.read_csv(OUTPUT_PATH).to_dict("records")
    else:
        results = []

    completed = {row["release_date"] for row in results}

    for _, row in sample.iterrows():
        date = row["release_date"]

        if date in completed:
            continue

        print(f"Re-scoring {date}...")

        try:
            result = score_statement(row["text"], date)
        except Exception as e:
            print(f"Stopped at {date}: {e}")
            break

        results.append({
            "release_date": date,
            "original_score": row["score"],
            "new_score": result["score"],
            "original_rationale": row["rationale"],
            "new_rationale": result["rationale"],
        })

        pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
        completed.add(date)
        time.sleep(5)

    if not results:
        print("No results yet.")
        return

    df = pd.DataFrame(results)
    differences = (df["original_score"] - df["new_score"]).abs()

    print("\nStatements compared:", len(df))
    print("Exact agreement:", round((differences == 0).mean(), 3))
    print("Mean absolute difference:", round(differences.mean(), 3))
    print("Largest difference:", differences.max())

    print("\nDisagreements:")
    print(df.loc[differences > 0, [
        "release_date", "original_score", "new_score"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
