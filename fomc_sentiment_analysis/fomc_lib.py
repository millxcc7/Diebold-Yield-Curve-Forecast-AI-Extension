"""
fomc_lib.py
-----------
Turns an FOMC statement into a hawkish/dovish sentiment score.

Contains:
    1. Scoring prompt
    2. Gemini LLM function
    3. Statement-text extraction
    4. Reusable score_statement() function
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
import time
from google.genai import errors

MODEL = "gemini-3.5-flash-lite"
PROMPT_VERSION = "v1"
VALID_SCORES = {-1, -0.5, 0, 0.5, 1}

# Load API key from the repository's .env file
ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


# 1. Scoring prompt

PROMPT_TEMPLATE = """
You are scoring the monetary policy stance communicated in a historical
Federal Reserve FOMC statement.

Baseline Date: {baseline_date}

Assign exactly one of the following scores:

Score | Meaning           | Interpretation
 1    | Very Hawkish      | Strong tightening or clear expectations
                           of substantially tighter monetary policy
 0.5  | Somewhat Hawkish  | Moderately restrictive or leaning toward tightening
 0    | Neutral           | No clear directional monetary policy bias
-0.5  | Somewhat Dovish   | Moderately accommodative or leaning toward easing
-1    | Very Dovish       | Strong easing or clear expectations
                           of substantially more monetary accommodation

Evaluate the overall communicated monetary policy stance, considering:
- Interest-rate decisions
- Asset purchases and other monetary policy actions
- Forward guidance and future policy intentions
- Inflation, employment, and economic risks

Do not classify a statement as hawkish solely because inflation is high,
or as dovish solely because economic growth is weak.

Evaluate the statement as of {baseline_date}, considering only information
available on or before that date.

Base the score ONLY on the supplied FOMC statement text.
Do not use knowledge of later economic events or policy decisions.

Return one of the five allowed scores and a one-sentence rationale
grounded in the statement.

FOMC statement:
\"\"\"
{statement_text}
\"\"\"
"""


# 2. Gemini LLM function

def call_llm(prompt: str, client=None) -> dict:
    """Send a prompt to Gemini, retrying temporary server errors."""

    if client is None:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing from .env")

        client = genai.Client(api_key=api_key)

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {
                            "score": {"type": "NUMBER"},
                            "rationale": {"type": "STRING"},
                        },
                        "required": ["score", "rationale"],
                    },
                ),
            )

            if not response.text:
                raise ValueError("Gemini returned an empty response.")

            return json.loads(response.text)

        except errors.ServerError as e:
            if e.code != 503 or attempt == 2:
                raise

            delay = 2 ** (attempt + 1)
            print(f"Gemini unavailable (503). Retrying in {delay}s...")
            time.sleep(delay)


# 3. Statement-text extraction

def extract_statement_text(source: str) -> str:
    """
    Extract and lightly normalize statement text.

    Expects the cleaned text from data/fomc_statements.csv.
    Does not download or re-scrape the statement.
    """

    if not isinstance(source, str) or not source.strip():
        raise ValueError("Statement text is empty or invalid.")

    text = source.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


# 4. Reusable scoring function

def score_statement(source: str, baseline_date: str, client=None) -> dict:
    """
    Score the hawkish/dovish stance of one FOMC statement.

    Args:
        source: Cleaned FOMC statement text.
        baseline_date: Statement release date, e.g. "2008-12-16".
        client: Optional reusable Gemini client.

    Returns:
        Dictionary containing the date, score, rationale,
        model name, and prompt version.
    """

    statement_text = extract_statement_text(source)

    prompt = PROMPT_TEMPLATE.format(
        baseline_date=str(baseline_date),
        statement_text=statement_text,
    )

    result = call_llm(prompt, client=client)

    # Validate that Gemini returned one of our five allowed scores
    if isinstance(result.get("score"), bool):
        raise ValueError("Invalid sentiment score: boolean")

    try:
        score = float(result["score"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError("Gemini returned an invalid score.") from e

    if score not in VALID_SCORES:
        raise ValueError(f"Invalid sentiment score: {score}")

    rationale = result.get("rationale")

    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("Gemini returned an empty rationale.")

    return {
        "baseline_date": str(baseline_date),
        "score": score,
        "rationale": rationale.strip(),
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
    }
