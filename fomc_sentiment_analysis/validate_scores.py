"""
Validate historical FOMC sentiment scores.

Run:
    python3 -m fomc_sentiment_analysis.validate_scores
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STATEMENTS_PATH = ROOT / "data" / "fomc_statements.csv"
SCORES_PATH = ROOT / "data" / "fomc_scores.csv"

VALID_SCORES = {-1, -0.5, 0, 0.5, 1}


def main():
    statements = pd.read_csv(STATEMENTS_PATH)
    scores = pd.read_csv(SCORES_PATH)

    print("=== 1. DATASET INTEGRITY ===")

    print("Statements:", len(statements))
    print("Scores:", len(scores))
    print("Missing scores:", scores["score"].isna().sum())
    print("Missing rationales:", (
        scores["rationale"].isna()
        | scores["rationale"].fillna("").str.strip().eq("")
    ).sum())
    print("Duplicate dates:", scores["release_date"].duplicated().sum())

    invalid = scores[~scores["score"].isin(VALID_SCORES)]
    print("Invalid scores:", len(invalid))

    statement_dates = set(statements["release_date"])
    scored_dates = set(scores["release_date"])

    missing_dates = sorted(statement_dates - scored_dates)
    extra_dates = sorted(scored_dates - statement_dates)

    print("Unscored statement dates:", missing_dates)
    print("Unexpected score dates:", extra_dates)

    print("\nModels used:")
    print(scores["model"].value_counts())

    print("\nPrompt versions:")
    print(scores["prompt_version"].value_counts())

    print("\n=== 2. SCORE DISTRIBUTION ===")
    print(scores["score"].value_counts().sort_index())
    print("Mean score:", round(scores["score"].mean(), 3))
    print("Standard deviation:", round(scores["score"].std(), 3))

    print("\n=== 3. HISTORICAL EPISODES ===")

    episodes = {
        "2008 financial crisis": ("2008-09-01", "2009-12-31"),
        "2020 pandemic response": ("2020-03-01", "2020-04-30"),
        "2022 tightening cycle": ("2022-03-01", "2022-12-31"),
    }

    for name, (start, end) in episodes.items():
        subset = scores[
            scores["release_date"].between(start, end)
        ]

        print(f"\n{name}")
        print("Statements:", len(subset))
        print("Average score:", round(subset["score"].mean(), 3))

        print(subset[
            ["release_date", "score"]
        ].to_string(index=False))

    print("\n=== 4. OVERALL CHECK ===")

    passed = (
        len(statements) == 232
        and len(scores) == len(statements)
        and scores["release_date"].is_unique
        and scores["score"].notna().all()
        and len(invalid) == 0
        and len(missing_dates) == 0
        and len(extra_dates) == 0
        and (
            scores["rationale"].notna()
            & scores["rationale"].str.strip().ne("")
        ).all()
        and scores["model"].nunique() == 1
        and scores["prompt_version"].nunique() == 1
    )

    if passed:
        print("PASS: Dataset integrity checks passed.")
    else:
        print("FAIL: Review the problems listed above.")


if __name__ == "__main__":
    main()
