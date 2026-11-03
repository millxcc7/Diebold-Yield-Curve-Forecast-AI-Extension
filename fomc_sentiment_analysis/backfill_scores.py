"""
Score historical FOMC statements using Gemini.
Read existing statements, call score_statement() for each one, and save results to data/fomc_scores.csv.

Run from the repo root:
    python3 -m fomc_sentiment_analysis.backfill_scores
"""

import csv
import os
import time
from pathlib import Path

import pandas as pd
from google import genai

from .fomc_lib import MODEL, PROMPT_VERSION, score_statement

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "fomc_statements.csv"
OUTPUT_PATH = ROOT / "data" / "fomc_scores.csv"

COLUMNS = [
    "release_date",
    "score",
    "rationale",
    "model",
    "prompt_version",
]

PAUSE = 5


def main():
    statements = pd.read_csv(INPUT_PATH)
    statements = statements.sort_values("release_date")

    # Load existing scores so reruns skip completed statements.
    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)

        if list(existing.columns) != COLUMNS:
            raise ValueError("Existing scores CSV has unexpected columns.")

        completed = set(
            zip(
                existing["release_date"],
                existing["model"],
                existing["prompt_version"],
            )
        )
    else:
        completed = set()

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    with OUTPUT_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)

        if f.tell() == 0:
            writer.writeheader()

        for _, row in statements.iterrows():
            date = row["release_date"]
            key = (date, MODEL, PROMPT_VERSION)

            if key in completed:
                print(f"Skipping {date}: already scored")
                continue

            print(f"Scoring {date}...")

            try:
                result = score_statement(
                    source=row["text"],
                    baseline_date=date,
                    client=client,
                )
            except Exception as e:
                print(f"FAILED {date}: {e}")
                print("Stopping. Rerun later to resume.")
                break

            writer.writerow({
                "release_date": date,
                "score": result["score"],
                "rationale": result["rationale"],
                "model": result["model"],
                "prompt_version": result["prompt_version"],
            })

            f.flush()
            completed.add(key)

            print(f"  Score: {result['score']}")
            time.sleep(PAUSE)

    print(f"\nCompleted {len(completed)} statement scores.")


if __name__ == "__main__":
    main()
