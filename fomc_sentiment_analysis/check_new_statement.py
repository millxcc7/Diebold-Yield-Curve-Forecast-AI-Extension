"""Check the Fed's RSS feed and update the statement and score CSVs.

Run one check from the repository root (locally or through GitHub Actions):
    python -B -m fomc_sentiment_analysis.check_new_statement

Direct execution also works:
    python -B fomc_sentiment_analysis/check_new_statement.py

Run monthly_sentiment.py afterward to update the monthly sentiment CSV:
    python -B -m fomc_sentiment_analysis.monthly_sentiment

Reuses the historical collector and Gemini scorer. The
existing GEMINI_API_KEY environment variable or .env configuration is used only
when scoring is needed. GitHub Actions will control the schedule.
Check regularly because RSS contains only recent announcements.
"""

from pathlib import Path
import sys
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET

import pandas as pd
import requests

# Support both module execution and running this file directly.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.collect_fomc import BASE, HEADERS, date_from_url, get_statement_text

ROOT = Path(__file__).resolve().parents[1]
FEED_URL = f"{BASE}/feeds/press_monetary.xml"
STATEMENT_COLUMNS = ["release_date", "url", "text"]
SCORE_COLUMNS = ["release_date", "score", "rationale", "model", "prompt_version"]


def read_csv(path, columns):
    if not path.exists():
        return []
    return pd.read_csv(
        path, usecols=columns, dtype=str, keep_default_na=False, encoding="utf-8-sig"
    ).to_dict("records")


def write_csv(path, columns, rows):
    """Write a complete CSV before replacing the existing file."""
    temporary = path.with_suffix(".csv.tmp")
    try:
        pd.DataFrame(rows, columns=columns).to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def feed_statements(content):
    """Select statement releases, excluding minutes and economic projections."""
    found = {}
    for item in ET.fromstring(content).findall("./channel/item"):
        title = " ".join((item.findtext("title") or "").split())
        if title.casefold().rstrip(".") != "federal reserve issues fomc statement":
            continue
        link = item.findtext("link") or item.findtext("guid")
        if not link or not link.strip():
            continue
        url = link.strip()
        if urlsplit(url).hostname != "www.federalreserve.gov":
            continue
        # Reuse date_from_url() from data/collect_fomc.py.
        released = date_from_url(url)
        if not pd.isna(released):
            found[url] = {
                "release_date": released.strftime("%Y-%m-%d"), "url": url
            }
    return list(found.values())


def check_new_statements(data_dir):
    # 1. Read the statements and scores already saved.
    statements_path = data_dir / "fomc_statements.csv"
    scores_path = data_dir / "fomc_scores.csv"
    statements = read_csv(statements_path, STATEMENT_COLUMNS)
    scores = read_csv(scores_path, SCORE_COLUMNS)
    by_url = {row["url"]: row for row in statements}
    by_date = {row["release_date"]: row for row in statements}
    scored_dates = {row["release_date"] for row in scores}

    # 2. Check the Fed's Monetary Policy RSS feed for statement announcements.
    response = requests.get(FEED_URL, headers=HEADERS, timeout=30)
    response.raise_for_status()
    candidates = {row["url"]: row for row in feed_statements(response.content)}

    # 3. Compare the Fed's URLs directly and skip completed releases.
    #    Retry saved, unscored text.
    candidates.update({url: row for url, row in by_url.items()
                       if row["release_date"] not in scored_dates})
    pending = [row for url, row in candidates.items()
               if url not in by_url or row["release_date"] not in scored_dates]
    for item in sorted(pending, key=lambda row: row["release_date"]):
        url, released = item["url"], item["release_date"]
        saved = by_url.get(url)
        if saved is None and released in by_date:
            raise ValueError(f"A different statement URL already exists for {released}.")

        # 4. Reuse get_statement_text() from data/collect_fomc.py. It downloads
        #    with retries and calls the existing clean_text() HTML cleanup.
        statement = saved or {**item, "text": get_statement_text(item["url"])}
        new_scores = scores
        if released not in scored_dates:
            # 5. Reuse score_statement() from fomc_lib.py, including its Gemini
            #    call_llm(), model, prompt, text normalization and validation.
            from fomc_sentiment_analysis.fomc_lib import score_statement

            print(f"Scoring {released}...")
            result = score_statement(source=statement["text"], baseline_date=released)
            new_scores = sorted(scores + [{
                "release_date": released,
                **{key: str(result[key]) for key in SCORE_COLUMNS if key != "release_date"},
            }], key=lambda row: row["release_date"])
        new_statements = statements if saved else sorted(
            statements + [statement], key=lambda row: row["release_date"]
        )

        # 6. Save the statement and score CSVs.
        #    Save scores first so an interrupted update can reuse paid results.
        if new_scores != scores:
            write_csv(scores_path, SCORE_COLUMNS, new_scores)
        if new_statements != statements:
            write_csv(statements_path, STATEMENT_COLUMNS, new_statements)
        statements, scores = new_statements, new_scores
        by_url[url], by_date[released] = statement, statement
        scored_dates.add(released)
        print(f"Updated statement and score for {released}.")

    if not pending:
        print("No new FOMC statements to score.")
    return len(pending)


def main():
    # 7. Run one check and exit. GitHub Actions will handle scheduled runs.
    try:
        check_new_statements(ROOT / "data")
    except Exception as error:
        print(f"Statement check failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
