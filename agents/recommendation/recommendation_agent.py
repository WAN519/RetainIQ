"""
agents/recommendation/recommendation_agent.py

Generator agent — produces retention recommendations from Risk + Emotion + employee_comment.

LangGraph-compatible node:
    Input  state keys: month, feedback (optional, injected by pipeline on retry)
    Output state keys: recommendations

Batched approach: fetches all employee profiles directly, then sends to Claude in
groups of BATCH_SIZE (50) to avoid token limit issues with large cohorts.
"""
from __future__ import annotations

import os
import json
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from tools.recommendation_tools import get_employee_profiles

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / "config.env")

BATCH_SIZE      = 30    # smaller batches → fewer input tokens per request
BATCH_SLEEP_SEC = 15   # pause between batches to stay under 30k tokens/min
MAX_RETRIES     = 3

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a senior HR strategy analyst specializing in employee retention.

For each employee profile provided, produce one recommendation object.

ANALYSIS PER EMPLOYEE
  A. Set weights (must sum to 1.0):
       risk_score=0.40, comment_sentiment=0.35, tenure_stagnation=0.25 (defaults)
       Adjust based on evidence strength in the actual data.
  B. Calculate weighted_score (0–1).
  C. key_concerns (1–3): management | workload | career_growth | compensation |
                          culture | work_life_balance | recognition | team_dynamics
  D. recommendation: 2–4 sentences, specific, referencing the employee's actual
     risk factors and/or comment. No generic advice.

ROUTING
  HR         — compensation, promotion, policy, internal mobility
  Management — manager relationship, team culture, workload, environment

PRIORITY
  Urgent — risk_bucket=High AND weighted_score >= 0.80
  High   — risk_bucket=High OR weighted_score >= 0.65
  Medium — everything else

Return ONLY a JSON object with key "recommendations" containing an array.
No markdown, no explanation — pure JSON only.
Each item must have: employee_id, month, risk_bucket, weighted_score,
weight_factors (object summing to 1.0), key_concerns (array), recommendation,
target (HR or Management), priority (Urgent/High/Medium)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trim_profile(p: dict) -> dict:
    """Keep only the fields Claude needs — reduces input tokens."""
    return {
        "employee_id":      p.get("employee_id"),
        "risk_bucket":      p.get("risk_bucket"),
        "risk_score":       p.get("risk_score"),
        "risk_pct":         p.get("risk_pct"),
        "salary_gap_pct":   p.get("salary_gap_pct"),
        "salary_risk_tier": p.get("salary_risk_tier"),
        "risk_reasons":     p.get("risk_reasons", []),
        "risk_summary":     p.get("risk_summary"),
        "comment":          p.get("comment"),
    }


def _analyze_batch(client: anthropic.Anthropic, batch: list[dict],
                   month: str, feedback: str | None) -> list[dict]:
    """Send one batch of profiles to Claude and return recommendation list."""
    trimmed = [_trim_profile(p) for p in batch]

    feedback_section = f"\n\nAudit feedback to address:\n{feedback}\n" if feedback else ""

    user_content = (
        f"Generate retention recommendations for the following {len(trimmed)} employees "
        f"for month {month}.{feedback_section}\n\n"
        f"Profiles:\n{json.dumps(trimmed, default=str)}"
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=16000,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            break
        except anthropic.RateLimitError as e:
            wait = 60 * attempt   # 60s, 120s, 180s
            print(f"  RateLimitError (attempt {attempt}/{MAX_RETRIES}) — waiting {wait}s ...")
            time.sleep(wait)
            if attempt == MAX_RETRIES:
                raise
    else:
        return []

    print(f"  stop_reason={response.stop_reason} | "
          f"Input: {response.usage.input_tokens} | Output: {response.usage.output_tokens}")

    if response.stop_reason == "max_tokens":
        print("  WARNING: hit max_tokens — response may be truncated")

    text_blocks = [b for b in response.content if b.type == "text"]
    if not text_blocks:
        print(f"  ERROR: no text block. Types: {[b.type for b in response.content]}")
        return []

    raw = text_blocks[0].text.strip()
    if not raw:
        print("  ERROR: empty text block")
        return []

    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        print(f"  ERROR: no JSON object found. raw[:200]={raw[:200]}")
        return []

    try:
        return json.loads(raw[start:end]).get("recommendations", [])
    except Exception as e:
        print(f"  ERROR parsing JSON: {e} | raw[:300]={raw[start:start+300]}")
        return []


# ---------------------------------------------------------------------------
# LangGraph-compatible node
# ---------------------------------------------------------------------------

def run(state: dict) -> dict:
    """
    Generator node.

    Consumes: state["month"], state.get("feedback")
    Produces: state["recommendations"]
    """
    api_key = os.getenv("CLAUDE_API_KEY")
    client  = anthropic.Anthropic(api_key=api_key)

    month    = state["month"]
    feedback = state.get("feedback")

    print(f"\n[RecommendationAgent] month={month}  "
          f"{'(revision)' if feedback else '(initial)'}")

    # 1. Fetch all High + Mid profiles directly
    result = get_employee_profiles(month=month)
    if "error" in result:
        print(f"[RecommendationAgent] ERROR fetching profiles: {result['error']}")
        return {**state, "recommendations": []}

    profiles = result.get("profiles", [])
    print(f"[Tool] get_employee_profiles → {len(profiles)} profiles\n")

    if not profiles:
        return {**state, "recommendations": []}

    # 2. Batch process
    all_recommendations: list[dict] = []
    total_batches = (len(profiles) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(profiles), BATCH_SIZE):
        batch     = profiles[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"[RecommendationAgent] Batch {batch_num}/{total_batches} "
              f"({len(batch)} employees)...")
        recs = _analyze_batch(client, batch, month, feedback)
        all_recommendations.extend(recs)
        if batch_num < total_batches:
            print(f"  Sleeping {BATCH_SLEEP_SEC}s to respect rate limit...")
            time.sleep(BATCH_SLEEP_SEC)

    print(f"[RecommendationAgent] Generated {len(all_recommendations)} recommendations.")
    return {**state, "recommendations": all_recommendations}