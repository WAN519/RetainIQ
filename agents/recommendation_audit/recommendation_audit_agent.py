"""
agents/recommendation_audit/recommendation_audit_agent.py

Critic agent — audits retention recommendations before they are saved.

LangGraph-compatible node:
    Input  state keys: recommendations
    Output state keys: audit_result, audit_attempts (incremented)

Two-stage audit:
  Stage 1 — Pre-flight (deterministic rules, no API call)
  Stage 2 — Claude adversarial critique (structured JSON output)

Verdict:
  APPROVED     — zero Critical issues, quality_score >= 7.0
  NEEDS_REVIEW — no Critical, quality_score 5.0–6.9 or <= 3 Warnings
  REJECTED     — any Critical issue, > 3 Warnings, or quality_score < 5.0
"""

import os
import json
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from tools.recommendation_tools import preflight_recommendations

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / "config.env")

# ---------------------------------------------------------------------------
# Structured output schema
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
            "description": "Overall quality 0–10 (10 = perfect).",
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
            "description": (
                "Specific instructions for the generator: which employees to fix, "
                "what exactly is wrong, and what the corrected output should look like. "
                "Empty string if verdict is APPROVED."
            ),
        },
    },
    "required": ["verdict", "quality_score", "overall_assessment",
                 "flagged", "revision_instructions"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Audit system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an adversarial HR audit critic. \
Review AI-generated retention recommendations before they are saved. \
You are looking for reasons to reject, not to approve.

Check every recommendation:

1. WEIGHT VALIDITY
   - weight_factors must sum to ~1.0. Flag Critical if they don't.
   - Weights must be justified by the data signals (risk_score, comment, tenure).

2. SCORE CONSISTENCY
   - weighted_score must align with risk_bucket.
   - High-bucket employee with weighted_score < 0.50 is a contradiction → Critical.

3. RECOMMENDATION SPECIFICITY
   - Generic phrases ("have a conversation", "monitor", "consider improving") → Warning.
   - Must reference specific signals from the employee's risk_reasons or comment.

4. ROUTING CORRECTNESS
   - HR: compensation, promotion process, policy, internal mobility.
   - Management: manager behavior, team culture, workload, day-to-day environment.
   - Wrong routing → Warning.

5. PRIORITY CALIBRATION
   - Urgent only valid for risk_bucket=High AND weighted_score >= 0.80 → Warning otherwise.
   - If > 40% of records are Urgent: priority inflation → Warning.

Verdict rules:
  APPROVED     — zero Critical, quality_score >= 7.0
  NEEDS_REVIEW — no Critical, quality_score 5.0–6.9, or <= 3 Warnings
  REJECTED     — any Critical, > 3 Warnings, or quality_score < 5.0

revision_instructions must be specific — name the employee IDs, quote the exact problem, \
state what the corrected output should look like. Leave empty if APPROVED."""


# ---------------------------------------------------------------------------
# LangGraph-compatible node
# ---------------------------------------------------------------------------

def run(state: dict) -> dict:
    """
    Critic node.

    Consumes: state["recommendations"]
    Produces: state["audit_result"], state["audit_attempts"] (incremented)
    """
    api_key = os.getenv("CLAUDE_API_KEY")
    client  = anthropic.Anthropic(api_key=api_key)

    recommendations = state["recommendations"]
    attempt         = state.get("audit_attempts", 0) + 1

    print(f"\n[RecommendationAuditAgent] Auditing {len(recommendations)} recommendations "
          f"(attempt {attempt})...")

    # Stage 1 — deterministic pre-flight
    preflight = preflight_recommendations(recommendations)
    print(f"[RecommendationAuditAgent] Pre-flight: {preflight['critical_count']} critical  "
          f"{preflight['warning_count']} warnings")

    # Stage 2 — Claude adversarial critique
    user_content = (
        "## Pre-flight Results\n"
        f"{json.dumps(preflight, indent=2, default=str)}\n\n"
        "## Recommendations\n"
        f"{json.dumps(recommendations, indent=2, default=str)}\n\n"
        "Audit these recommendations. Name employee IDs. Give your verdict."
    )

    user_content += (
        "\n\nReturn ONLY a JSON object matching this schema — no markdown, no explanation:\n"
        f"{json.dumps(_AUDIT_SCHEMA, indent=2)}"
    )

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4000,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

    text_blocks = [b for b in response.content if b.type == "text"]
    if not text_blocks:
        raise RuntimeError(f"Audit: no text block. Types: {[b.type for b in response.content]}")

    raw = text_blocks[0].text.strip()
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise RuntimeError(f"Audit: no JSON object in response. raw[:200]={raw[:200]}")
    audit_result = json.loads(raw[start:end])

    print(f"[RecommendationAuditAgent] Verdict: {audit_result['verdict']}  "
          f"score={audit_result['quality_score']:.1f}  "
          f"flagged={len(audit_result['flagged'])}")

    return {**state, "audit_result": audit_result, "audit_attempts": attempt}