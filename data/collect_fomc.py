"""
Collect historical FOMC statements (1996 - Sep 2026) into ONE csv.

Output:  data/fomc_statements.csv   columns: release_date, url, text
Also:    data/fomc_missing.csv      meetings with no statement (mostly 1996-1999)

Re-running only downloads statements that aren't in the csv yet.
Run from the repo root:  python3 data/collect_fomc.py
"""
import os
import re
import time
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://www.federalreserve.gov"
HEADERS = {"User-Agent": "Mozilla/5.0 (university course project)"}
PAUSE = 1.0                                   
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(DATA_DIR, "fomc_statements.csv")
MISSING_PATH = os.path.join(DATA_DIR, "fomc_missing.csv")

FIRST_YEAR, LAST_HIST_YEAR = 1996, 2020        
INCLUDE_2021_PLUS = True                       
START, END = pd.Timestamp("1996-01-01"), pd.Timestamp("2026-09-30")

# text cleaning: stop at the first of these lines (once real text has started)
STOP_PATTERN = re.compile(
    r"^(\d{4} Monetary policy|Home$|Implementation Note|Last update|"
    r"Back to Top|Return to top|For media inquiries|Stay Connected|"
    r"Board of Governors of the Federal Reserve System|"
    r"Related Information|Information on Related Actions|"
    r"Information on Actions Taken by Other Central Banks|"
    r"Statements by Other Central Banks|Frequently Asked Questions|"
    r"Statement Regarding|Statement from Federal Reserve Bank|\d+\.$)",
    re.I
)

def get_soup(url, tries=3):
    for attempt in range(tries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            time.sleep(PAUSE)
            return BeautifulSoup(resp.content, "html.parser")
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def date_from_url(url):
    m = re.search(r"(\d{8})", url)
    if not m:
        return pd.NaT

    date = m.group(1)

    # The Fed's URL has an incorrect date for this statement.
    if url.endswith("/20070618a.htm"):
        date = "20070628"
    return pd.to_datetime(date, format="%Y%m%d", errors="coerce")

def meeting_end_date(title):
    # "June 29-30 Meeting - 1999" -> 1999-06-30 ; "January 31-February 1 ..." works too
    m = re.match(r"([A-Za-z]+)\s+(\d+)(?:\s*-\s*(?:([A-Za-z]+)\s+)?(\d+))?\s+"
                 r"(?:Meeting|Conference Call)\s*-\s*(\d{4})", title)
    if not m:
        return pd.NaT
    month1, day1, month2, day2, year = m.groups()
    return pd.to_datetime(f"{month2 or month1} {day2 or day1} {year}", errors="coerce")


def links_from_year_page(year):
    """Return [(meeting title, statement url or None)] for one year page."""
    soup = get_soup(f"{BASE}/monetarypolicy/fomchistorical{year}.htm")
    rows = []
    for h in soup.find_all("h5"):
        title = h.get_text(" ", strip=True)
        if "Meeting" not in title and "Conference" not in title:
            continue
        link = None
        for el in h.find_all_next(["h5", "a"]):
            if el.name == "h5":
                break
            txt = el.get_text(" ", strip=True).lower()
            if txt.startswith("statement") and "longer-run" not in txt:
                link = urljoin(BASE, el.get("href"))
                break
        rows.append((title, link))
    return rows


def clean_text(text):
    text = text.replace("\u2011", "-").replace("\u2013", "-").replace("\xa0", " ").replace("\u0100", " ")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if "Share" in lines[:5]:
        lines = lines[lines.index("Share") + 1:]
    while lines and lines[0].lower() in {
        "for immediate release",
         "share",
         "press release",
    }:
         lines.pop(0)

    kept = []
    for line in lines:
        if kept and STOP_PATTERN.match(line):
            break
        kept.append(line)

    return "\n".join(kept)


def get_statement_text(url):
    soup = get_soup(url)
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    main = soup.find(id="article") or soup.find(id="content") or soup.body
    return clean_text(main.get_text("\n", strip=True))


# ---------- 1. find statement urls ----------
found, missing = [], []

for year in range(FIRST_YEAR, LAST_HIST_YEAR + 1):
    print("Year", year)
    for title, link in links_from_year_page(year):
        if link is None:
            missing.append({"meeting": title, "meeting_end_date": meeting_end_date(title)})
        else:
            d = date_from_url(link)
            if pd.isna(d):
                d = meeting_end_date(title)
            found.append({"release_date": d, "url": link})

if INCLUDE_2021_PLUS:
    print("Calendars page (2021+)")
    soup = get_soup(f"{BASE}/monetarypolicy/fomccalendars.htm")
    for a in soup.find_all("a", href=True):
        m = re.search(r"monetary(\d{8})a\.htm", a["href"])
        if m and m.group(1) != "20250822":
            found.append({"release_date": pd.to_datetime(m.group(1), format="%Y%m%d"),
                          "url": urljoin(BASE, a["href"])})

urls = pd.DataFrame(found).drop_duplicates("url")
urls = urls[(urls["release_date"] >= START) & (urls["release_date"] <= END)]

# Exclude liquidity-only announcements from sentiment scoring
EXCLUDE_DATES = pd.to_datetime(["2007-08-10", "2008-03-11", "2010-05-09"])
urls = urls[~urls["release_date"].isin(EXCLUDE_DATES)]

# ---------- 2. download only what we don't have yet ----------
if os.path.exists(OUT_PATH):
    old = pd.read_csv(OUT_PATH, parse_dates=["release_date"])
else:
    old = pd.DataFrame(columns=["release_date", "url", "text"])

todo = urls[~urls["url"].isin(old["url"])]
print(f"\n{len(urls)} statements found, {len(todo)} new to download")

new_rows = []
for _, row in todo.iterrows():
    try:
        text = get_statement_text(row["url"])
    except Exception as e:
        print("FAILED:", row["url"], e)
        continue
    new_rows.append({"release_date": row["release_date"], "url": row["url"], "text": text})
    print("  got", row["release_date"].date())

frames = [f for f in (old, pd.DataFrame(new_rows)) if len(f)]
df = pd.concat(frames, ignore_index=True) if frames else old
df = df.drop_duplicates("url").sort_values("release_date").reset_index(drop=True)
df.to_csv(OUT_PATH, index=False)
pd.DataFrame(missing).to_csv(MISSING_PATH, index=False)

# ---------- 3. summary ----------
words = df["text"].fillna("").str.split().str.len()
print("\nStatements saved:", len(df))
print("Meetings with no statement:", len(missing))
print("Shortest / longest (words):", words.min(), "/", words.max())
print("\nStatements per year:")
print(df["release_date"].dt.year.value_counts().sort_index().to_string())