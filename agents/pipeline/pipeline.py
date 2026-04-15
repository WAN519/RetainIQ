"""
agents/pipeline/pipeline.py

Orchestrator for the retention recommendation pipeline.

Coordinates RecommendationAgent (generator) and RecommendationAuditAgent (critic)
with a retry loop of up to MAX_AUDIT_ATTEMPTS before forcing a save.

Designed to be LangGraph-ready: each step maps directly to a node,
and the conditional logic maps to edges.

Current (plain Python):
    run_pipeline(month) → loops manually

Future (LangGraph):
    graph = StateGraph(PipelineState)
    graph.add_node("generate", recommendation_agent.run)
    graph.add_node("audit",    recommendation_audit_agent.run)
    graph.add_node("save",     save_node)
    graph.add_conditional_edges("audit", should_continue, {...})

Usage:
    python -m agents.pipeline.pipeline --month 2026-04
"""

import os
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import certifi
from dotenv import load_dotenv
from pymongo import MongoClient

import agents.recommendation.recommendation_agent as recommendation_agent
import agents.recommendation_audit.recommendation_audit_agent as recommendation_audit_agent

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / "config.env")

MAX_AUDIT_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# State definition  (maps directly to LangGraph StateGraph schema)
# ---------------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    month:           str
    recommendations: list[dict]
    audit_result:    dict
    audit_attempts:  int
    feedback:        str


# ---------------------------------------------------------------------------
# Save node
# ---------------------------------------------------------------------------

def _save_node(state: PipelineState) -> PipelineState:
    """Persist recommendations to MongoDB with audit metadata."""
    uri             = os.getenv("MONGODB_URI")
    db_name         = os.getenv("MONGODB_NAME", "MarketInformation")
    collection_name = "retention_recommendations"

    client = MongoClient(uri, serverSelectionTimeoutMS=5000, tlsCAFile=certifi.where())
    db     = client[db_name]

    if collection_name not in db.list_collection_names():
        db.create_collection(collection_name)

    audit  = state.get("audit_result") or {}
    now    = datetime.now(timezone.utc).isoformat()
    docs   = [{
        **r,
        "audit_verdict":  audit.get("verdict", "UNKNOWN"),
        "audit_score":    audit.get("quality_score"),
        "audit_attempts": state.get("audit_attempts", 0),
        "created_at":     now,
    } for r in state["recommendations"]]

    result = db[collection_name].insert_many(docs)
    print(f"\n[Pipeline] Saved {len(result.inserted_ids)} recommendations "
          f"→ collection='{collection_name}'  "
          f"audit_verdict={audit.get('verdict')}")
    client.close()
    return state


# ---------------------------------------------------------------------------
# Conditional edge  (maps to LangGraph conditional_edges)
# ---------------------------------------------------------------------------

def _should_continue(state: PipelineState) -> str:
    """
    Routing logic after each audit.

    Returns:
        "save"       — approved or max attempts reached
        "regenerate" — audit failed, retry allowed
    """
    verdict  = state["audit_result"]["verdict"]
    attempts = state["audit_attempts"]

    if verdict == "APPROVED":
        print(f"[Pipeline] APPROVED after {attempts} audit(s) → saving")
        return "save"

    if attempts >= MAX_AUDIT_ATTEMPTS:
        print(f"[Pipeline] Max audit attempts ({MAX_AUDIT_ATTEMPTS}) reached "
              f"(verdict={verdict}) → force saving")
        return "save"

    print(f"[Pipeline] {verdict} — attempt {attempts}/{MAX_AUDIT_ATTEMPTS} → regenerating")
    return "regenerate"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(month: str) -> None:
    print(f"\n{'='*62}")
    print(f"  Recommendation Pipeline  |  Month: {month}")
    print(f"  Max audit attempts : {MAX_AUDIT_ATTEMPTS}")
    print(f"{'='*62}")

    state: PipelineState = {
        "month":           month,
        "recommendations": [],
        "audit_result":    {},
        "audit_attempts":  0,
        "feedback":        "",
    }

    # ── Generate ──────────────────────────────────────────────────────────
    state = recommendation_agent.run(state)

    while True:
        # ── Audit ─────────────────────────────────────────────────────────
        state = recommendation_audit_agent.run(state)

        decision = _should_continue(state)

        if decision == "save":
            _save_node(state)
            break

        # ── Inject feedback and regenerate ────────────────────────────────
        state["feedback"] = state["audit_result"]["revision_instructions"]
        state = recommendation_agent.run(state)

    print(f"\n{'='*62}")
    print("  Pipeline complete.")
    print(f"{'='*62}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retention recommendation pipeline (generator + audit loop)"
    )
    parser.add_argument(
        "--month",
        default=datetime.now(timezone.utc).strftime("%Y-%m"),
        help="Analysis month in YYYY-MM format (default: current month)",
    )
    args = parser.parse_args()
    run_pipeline(month=args.month)