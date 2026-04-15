"""
agents/emotion/emotion_agent.py

Claude-powered Glassdoor sentiment analysis agent.

Workflow:
  1. Claude calls run_emotion_analysis → NLP pipeline labels every Glassdoor
     review by topic (management / salary / workload / career) and sentiment.
  2. Claude calls get_emotion_summary → aggregated topic-level statistics.
  3. Claude calls get_high_risk_reviews → concrete review excerpts with the
     highest negative scores for qualitative grounding.
  4. Claude synthesizes the data into a company-level sentiment report and
     saves it to the Emotion collection.

Output (MongoDB Emotion collection):
  One document per company per month:
    { company, month, company_sentiment_report, created_at }

All tool implementations live in tools/attrition_tools.py.

Usage:
  python -m agents.emotion.emotion_agent \\
      --company Apple \\
      --csv data/mock_reviews.csv \\
      --month 2026-04
"""

import copy
import os
import argparse
from datetime import datetime, timezone

import anthropic
import certifi
from dotenv import load_dotenv
from pathlib import Path
from pymongo import MongoClient

from tools.attrition_tools import TOOLS, execute_tool

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / "config.env")

EMOTION_TOOLS = [t for t in TOOLS if t["name"] in {
    "run_emotion_analysis",
    "get_emotion_summary",
    "get_high_risk_reviews",
}]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior HR analyst specializing in employee sentiment and attrition prevention.

You have access to three tools that let you analyze Glassdoor employee reviews:

  1. run_emotion_analysis  — run the NLP pipeline on the raw Glassdoor CSV for a company
  2. get_emotion_summary   — retrieve aggregated sentiment and topic-issue rates from MongoDB
  3. get_high_risk_reviews — fetch the most negatively-scored review samples

Follow this sequence strictly:
  Step 1: Call run_emotion_analysis to label and store the reviews.
  Step 2: Call get_emotion_summary to get the quantitative breakdown.
  Step 3: Call get_high_risk_reviews to read actual employee language.
  Step 4: Write the final report.

Your final report must follow this structure:

## Overall Sentiment Assessment
[State the overall negative sentiment rate and what it signals about attrition risk.
 Cite the exact numbers returned by the tools.]

## Top Attrition Drivers (from Employee Voice)
[For each issue category that exceeds 20% of reviews, explain:
 - What percentage of reviews flag it
 - What the actual review excerpts reveal about employee frustration
 - Why this driver leads to voluntary turnover]

## Predicted Attrition Risk by Driver
[Rank the drivers from highest to lowest attrition impact. Justify the ranking
 using both the sentiment score and the review language.]

## Recommended HR Interventions
[Numbered list. For each recommendation:
 - State which attrition driver it addresses
 - Describe the specific action (not generic advice)
 - Identify which employee group is the priority target]

Be concrete. Quote employee language from the reviews. Every claim must be backed
by a number or a direct quote from the data you retrieved."""


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_emotion_agent(
    company_name: str,
    csv_path: str,
    month: str | None = None,
) -> str:
    """
    Run the Glassdoor sentiment analysis agent and save the report to MongoDB.

    Args:
        company_name: Company to analyze. Must match the 'firm' field in the CSV.
        csv_path:     Path to the Glassdoor reviews CSV file.
        month:        Analysis month in YYYY-MM format. Defaults to current month.

    Returns:
        The final structured sentiment report as a string.
    """
    if month is None:
        month = datetime.now(timezone.utc).strftime("%Y-%m")

    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise EnvironmentError("CLAUDE_API_KEY is not set in config.env")

    client = anthropic.Anthropic(api_key=api_key)

    # Strip csv_path from the tool schema — injected by Python, not passed by Claude
    emotion_tools = copy.deepcopy(EMOTION_TOOLS)
    for t in emotion_tools:
        if t["name"] == "run_emotion_analysis":
            t["input_schema"]["properties"].pop("csv_path", None)
            t["input_schema"]["required"] = [
                r for r in t["input_schema"].get("required", []) if r != "csv_path"
            ]

    def _execute(name: str, inputs: dict) -> str:
        if name == "run_emotion_analysis":
            inputs = {**inputs, "csv_path": csv_path}
        return execute_tool(name, inputs)

    messages = [{"role": "user", "content": (
        f"Please analyze employee sentiment and attrition risk for "
        f"**{company_name}**, period **{month}**.\n\n"
        f"Run the NLP pipeline first, then retrieve the aggregated sentiment "
        f"statistics, then sample the most negative reviews. "
        f"Identify the primary attrition drivers from the employee voice data "
        f"and provide specific HR interventions for each driver."
    )}]
    final_report = ""

    print(f"\n{'='*62}")
    print(f"  Emotion Agent  |  Sentiment Analysis")
    print(f"  Company : {company_name}")
    print(f"  Period  : {month}")
    print(f"  CSV     : {csv_path}")
    print(f"{'='*62}\n")

    turn = 0
    while True:
        turn += 1
        print(f"[Turn {turn}] Sending request to Claude...\n")

        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=8000,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=emotion_tools,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
            response = stream.get_final_message()

        print()

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        if response.stop_reason == "end_turn" or not tool_use_blocks:
            final_report = "".join(
                b.text for b in response.content if b.type == "text"
            )
            break

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in tool_use_blocks:
            result = _execute(block.name, block.input)
            print(f"[Tool] {block.name}: {result[:300]}\n")
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})

        messages.append({"role": "user", "content": tool_results})

    print(f"\n{'='*62}")
    print("  Analysis complete.")
    print(f"{'='*62}\n")

    _save_report(final_report, company_name, month)

    return final_report


def _save_report(report: str, company_name: str, month: str) -> None:
    """
    Save the company-level sentiment report to the Emotion collection.
    One document per company per month — upserts on re-run.
    """
    uri             = os.getenv("MONGODB_URI")
    db_name         = os.getenv("MONGODB_NAME", "MarketInformation")
    collection_name = os.getenv("COLLECTION_NAME_EMOTION", "Emotion")

    client = MongoClient(uri, serverSelectionTimeoutMS=5000, tlsCAFile=certifi.where())
    db     = client[db_name]

    if collection_name not in db.list_collection_names():
        db.create_collection(collection_name)
        print(f"[MongoDB] Created collection '{collection_name}'")

    doc = {
        "company":                  company_name,
        "month":                    month,
        "company_sentiment_report": report,
        "created_at":               datetime.now(timezone.utc).isoformat(),
    }

    db[collection_name].update_one(
        {"company": company_name, "month": month},
        {"$set": doc},
        upsert=True,
    )
    print(f"[MongoDB] Emotion report saved → collection='{collection_name}' "
          f"company='{company_name}' month='{month}'")
    client.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Claude-powered Glassdoor sentiment analysis agent"
    )
    parser.add_argument("--company", default="Apple",
                        help="Company name (must match 'firm' column in CSV)")
    parser.add_argument("--csv",     default="data/mock_reviews.csv",
                        help="Path to the Glassdoor reviews CSV file")
    parser.add_argument("--month",   default=None,
                        help="Analysis month in YYYY-MM format (default: current month)")
    args = parser.parse_args()

    run_emotion_agent(
        company_name=args.company,
        csv_path=args.csv,
        month=args.month,
    )