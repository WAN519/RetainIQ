"""
agents/recommendation/claude_recommendation_agent.py

Claude-powered retention recommendation agent with adversarial audit gate.

Workflow:
  1. Claude calls get_employee_profiles → combined Emotion + employee_comment data.
  2. Claude analyzes each profile and calls save_recommendations.
  3. Agent intercepts the call and runs a two-stage audit:
       a. Pre-flight — deterministic rule checks (weights, score ranges, generic text)
       b. Claude audit  — adversarial critic reviews quality and routing
  4. APPROVED  → recommendations saved to MongoDB with audit metadata.
     REJECTED/NEEDS_REVIEW → audit feedback returned to Claude for one revision.
  5. On second attempt the result is always saved regardless of verdict.

Usage:
  python -m agents.recommendation.claude_recommendation_agent --month 2026-04
"""

import os
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from tools.recommendation_tools import (
    TOOLS, execute_tool,
    preflight_recommendations, mongo_save,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / "config.env")


# ---------------------------------------------------------------------------
# Audit — structured output schema
# ---------------------------------------------------------------------------

_AUDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["APPROVED", "NEEDS_REVIEW", "REJECTED"],
        },
        "quality_score": {
            "type": "number",
            "description": "Overall quality 0-10 (10 = perfect).",
        },
        "overall_assessment": {"type": "string"},
        "flagged": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "employee_id":   {"type": ["integer", "string"]},
                    "issues":        {"type": "array", "items": {"type": "string"}},
                    "severity":      {"type": "string", "enum": ["Critical", "Warning"]},
                    "suggested_fix": {"type": "string"},
                },
                "required": ["employee_id", "issues", "severity", "suggested_fix"],
                "additionalProperties": False,
            },
        },
        "revision_instructions": {
            "type": "string",
            "description": "Specific instructions for what to fix if verdict is not APPROVED.",
        },
    },
    "required": ["verdict", "quality_score", "overall_assessment",
                 "flagged", "revision_instructions"],
    "additionalProperties": False,
}

_AUDIT_SYSTEM_PROMPT = """You are an adversarial HR audit critic. \
Review AI-generated retention recommendations BEFORE they are saved.
You are looking for reasons to reject, not to approve.

Check every recommendation for:

1. WEIGHT VALIDITY
   - weight_factors must sum to ~1.0.
   - Weights must be justified by the available signals (risk_score, comment, tenure).
   - Flag Critical if weights are arbitrary or unsupported by the data.

2. SCORE CONSISTENCY
   - weighted_score must align with risk_bucket.
   - A High-bucket employee with weighted_score < 0.50 is a contradiction — Critical.
   - A Medium-priority recommendation for a High-bucket employee needs justification.

3. RECOMMENDATION SPECIFICITY
   - Generic phrases ("have a conversation", "monitor the situation", "consider improving")
     add zero value — flag Warning.
   - Recommendations must name specific actions, timelines, or roles.

4. ROUTING CORRECTNESS
   - HR: compensation, pay gap, promotion process, policy, internal mobility.
   - Management: manager behavior, team culture, workload, day-to-day environment.
   - Wrong routing is a Warning.

5. PRIORITY CALIBRATION
   - Urgent is only valid for risk_bucket=High AND weighted_score >= 0.80.
   - Over-use of Urgent (> 40% of records) is priority inflation — Warning.

Verdict rules:
  APPROVED     — zero Critical issues, quality_score >= 7.0
  NEEDS_REVIEW — no Critical issues, quality_score 5.0–6.9, or <= 3 Warnings
  REJECTED     — any Critical issue, > 3 Warnings, or quality_score < 5.0

If verdict is not APPROVED, revision_instructions must be specific and actionable — \
tell the model exactly what to fix, for which employees, and why."""


# ---------------------------------------------------------------------------
# Audit runner
# ---------------------------------------------------------------------------

def _run_audit(recommendations: list[dict], preflight: dict,
               client: anthropic.Anthropic) -> dict:
    """Run the Claude adversarial audit on a batch of recommendations."""
    user_content = (
        "## Pre-flight Rule Check Results\n"
        f"{json.dumps(preflight, indent=2, default=str)}\n\n"
        "## Recommendations to Audit\n"
        f"{json.dumps(recommendations, indent=2, default=str)}\n\n"
        "Audit these recommendations. Be specific, name employee IDs, give your verdict."
    )

    print("[Audit] Sending recommendations to Claude for audit review...")

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=4000,
        system=_AUDIT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
        output_config={"format": {"type": "json_schema", "schema": _AUDIT_SCHEMA}},
    ) as stream:
        final = stream.get_final_message()

    text_block = next(b for b in final.content if b.type == "text")
    return json.loads(text_block.text)


# ---------------------------------------------------------------------------
# Generation system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior HR strategy analyst specializing in employee retention.

You have two tools:
  1. get_employee_profiles — fetches combined risk and comment data per employee
  2. save_recommendations  — submits your recommendations for audit and saving

Follow this sequence strictly:
  Step 1: Call get_employee_profiles to retrieve employee data for the requested month.
  Step 2: For EVERY employee in the profiles list, generate a recommendation.
  Step 3: Call save_recommendations with the full list.
  Step 4: If the audit returns revision_required, fix the flagged issues and call
          save_recommendations again with the corrected full list.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA SOURCES (only these two MongoDB collections)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

From Emotion collection:
  - risk_score / risk_pct / risk_bucket
  - months_since_promotion
  - risk_reasons

From employee_comment collection:
  - comment (employee's own words)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANALYSIS STEPS PER EMPLOYEE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step A — Set weights (must sum to 1.0):
  Default: risk_score=0.40, comment_sentiment=0.35, tenure_stagnation=0.25
  Adjust based on evidence strength in the actual data.

Step B — Calculate weighted_score (0–1).

Step C — Identify key_concerns (1–3):
  management, workload, career_growth, compensation,
  culture, work_life_balance, recognition, team_dynamics

Step D — Write recommendation (2–4 sentences, specific, no generic advice).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROUTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  HR         — compensation, promotion, policy, internal mobility
  Management — manager relationship, team culture, workload, environment

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRIORITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Urgent — risk_bucket=High AND weighted_score >= 0.80
  High   — risk_bucket=High OR weighted_score >= 0.65
  Medium — everything else"""


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_recommendation_agent(month: str) -> None:
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise EnvironmentError("CLAUDE_API_KEY is not set in config.env")

    client = anthropic.Anthropic(api_key=api_key)

    messages = [{"role": "user", "content": (
        f"Generate retention recommendations for all High and Mid risk employees "
        f"for month **{month}**.\n\n"
        f"Fetch their profiles, analyze each one with appropriate factor weights, "
        f"then submit the full recommendations list."
    )}]

    print(f"\n{'='*62}")
    print(f"  Retention Recommendation Agent  |  Month: {month}")
    print(f"{'='*62}\n")

    _save_attempt = 0

    turn = 0
    while True:
        turn += 1
        print(f"[Turn {turn}] Sending request to Claude...\n")

        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
            response = stream.get_final_message()

        print()

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        if response.stop_reason == "end_turn" or not tool_use_blocks:
            break

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in tool_use_blocks:

            if block.name == "save_recommendations":
                recommendations = block.input["recommendations"]
                _save_attempt += 1

                # Stage 1 — deterministic pre-flight
                preflight = preflight_recommendations(recommendations)
                print(f"[Audit] Pre-flight: {preflight['critical_count']} critical  "
                      f"{preflight['warning_count']} warnings")

                # Stage 2 — Claude adversarial audit
                audit = _run_audit(recommendations, preflight, client)
                print(f"[Audit] Verdict: {audit['verdict']}  "
                      f"quality_score={audit['quality_score']:.1f}")

                if audit["verdict"] == "APPROVED" or _save_attempt >= 2:
                    # Save to MongoDB (always save on second attempt)
                    saved = mongo_save(recommendations, audit["verdict"],
                                      audit["quality_score"])
                    print(f"[MongoDB] {saved['inserted_count']} recommendations saved "
                          f"(verdict={audit['verdict']})\n")
                    result = json.dumps({**saved, "audit": audit}, default=str)
                else:
                    # Return feedback — Claude will revise and retry
                    print(f"[Audit] Revision required — returning feedback to Claude\n")
                    result = json.dumps({
                        "status":               "revision_required",
                        "verdict":              audit["verdict"],
                        "quality_score":        audit["quality_score"],
                        "flagged":              audit["flagged"],
                        "revision_instructions": audit["revision_instructions"],
                    }, default=str)

            else:
                result = execute_tool(block.name, block.input)
                print(f"[Tool] {block.name}: {result[:300]}\n")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})

    print(f"\n{'='*62}")
    print("  Recommendation agent complete.")
    print(f"{'='*62}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Claude-powered retention recommendation agent"
    )
    parser.add_argument(
        "--month",
        default=datetime.now(timezone.utc).strftime("%Y-%m"),
        help="Analysis month in YYYY-MM format (default: current month)",
    )
    args = parser.parse_args()
    run_recommendation_agent(month=args.month)