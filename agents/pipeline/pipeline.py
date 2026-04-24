"""
agents/pipeline/pipeline.py

Full end-to-end retention pipeline using LangGraph.

Graph topology:
    START → equity → retention ─┐
    START → emotion            ─┴→ generate → audit → [_should_continue]
                                                          "save"       → save → END
                                                          "regenerate" → prepare_feedback → generate

Nodes:
    equity           — LightGBM salary fair-value  → MongoDB: Equity_Predictions
    retention        — Cox survival + Claude HR     → MongoDB: Risk
    emotion          — NLP sentiment + Claude       → MongoDB: Emotion
    generate         — Claude recommendation gen    (reads Risk + Emotion + employee_comment)
    audit            — Adversarial Claude critic    (up to MAX_AUDIT_ATTEMPTS rounds)
    prepare_feedback — injects revision_instructions into state before regeneration
    save             — persist approved recs        → MongoDB: retention_recommendations

Usage:
    python -m agents.pipeline.pipeline --month 2026-04
"""

import os
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, TypedDict

import certifi
from dotenv import load_dotenv
from pymongo import MongoClient
from langgraph.graph import StateGraph, START, END

import agents.recommendation.recommendation_agent as recommendation_agent
import agents.recommendation_audit.recommendation_audit_agent as recommendation_audit_agent

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / "config.env")

MAX_AUDIT_ATTEMPTS = 3
_DEFAULT_HR_CSV      = str(_PROJECT_ROOT / "data" / "ibm_enhanced_test.csv")
_DEFAULT_REVIEWS_CSV = str(_PROJECT_ROOT / "data" / "mock_reviews.csv")
_MODEL_DIR           = _PROJECT_ROOT / "models"


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

def _keep(a, b):
    """Reducer for read-only config fields: keep existing value on fan-in merge."""
    return a if b is None else b


class PipelineState(TypedDict, total=False):
    # Config — read-only after init; Annotated reducer prevents fan-in conflicts
    month:           Annotated[str, _keep]
    company:         Annotated[str, _keep]
    hr_csv:          Annotated[str, _keep]
    reviews_csv:     Annotated[str, _keep]
    # Recommendation pipeline state
    recommendations: list[dict]
    audit_result:    dict
    audit_attempts:  int
    feedback:        str


# ---------------------------------------------------------------------------
# Stage nodes (1–3)
# ---------------------------------------------------------------------------

def _equity_node(state: PipelineState) -> dict:
    """Stage 1 — LightGBM salary equity scoring → MongoDB: Equity_Predictions."""
    from agents.equity.equity_agent import EquityAgent

    hr_csv       = state.get("hr_csv", _DEFAULT_HR_CSV)
    equity_model = str(_MODEL_DIR / "agent_salary_regressor.pkl")

    if not Path(equity_model).exists():
        print(f"[equity] SKIP — model not found: {equity_model}")
        return {}
    if not Path(hr_csv).exists():
        print(f"[equity] SKIP — HR CSV not found: {hr_csv}")
        return {}

    EquityAgent(model_path=equity_model).run_analysis_pipeline(hr_csv)
    print("[equity] Complete → MongoDB: Equity_Predictions")
    return {}


def _retention_node(state: PipelineState) -> dict:
    """Stage 2 — Cox survival model + Claude HR insights → MongoDB: Risk."""
    from agents.retention.retention_agent import RetentionAgent

    hr_csv       = state.get("hr_csv", _DEFAULT_HR_CSV)
    model_pkl    = str(_MODEL_DIR / "cox_retention_v1.pkl")
    feature_json = str(_MODEL_DIR / "cox_retention_v1_features.json")

    if not Path(model_pkl).exists():
        print(f"[retention] SKIP — model not found: {model_pkl}")
        return {}
    if not Path(feature_json).exists():
        print(f"[retention] SKIP — features JSON not found: {feature_json}")
        return {}
    if not Path(hr_csv).exists():
        print(f"[retention] SKIP — HR CSV not found: {hr_csv}")
        return {}

    agent = RetentionAgent(model_path=model_pkl, feature_json_path=feature_json)
    docs  = agent.run(hr_csv)
    print(f"[retention] Complete — {len(docs)} records → MongoDB: Risk")
    return {}


def _emotion_node(state: PipelineState) -> dict:
    """Stage 3 — NLP sentiment + Claude report → MongoDB: Emotion."""
    from agents.emotion.emotion_agent import run_emotion_agent

    reviews_csv = state.get("reviews_csv", _DEFAULT_REVIEWS_CSV)
    company     = state.get("company", "Apple")
    month       = state.get("month", "")

    if not Path(reviews_csv).exists():
        print(f"[emotion] SKIP — reviews CSV not found: {reviews_csv}")
        return {}

    run_emotion_agent(company_name=company, csv_path=reviews_csv, month=month)
    print("[emotion] Complete → MongoDB: Emotion")
    return {}


# ---------------------------------------------------------------------------
# Stage 4 nodes — recommendation + audit loop
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
    return {}


def _prepare_feedback_node(state: PipelineState) -> dict:
    """Extract revision instructions from audit result into the feedback field."""
    feedback = state.get("audit_result", {}).get("revision_instructions", "")
    return {"feedback": feedback}


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------

def _should_continue(state: PipelineState) -> str:
    """
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
# Build and compile the LangGraph
# ---------------------------------------------------------------------------

def _build_graph():
    graph = StateGraph(PipelineState)

    # Stage 1–3 nodes
    graph.add_node("equity",            _equity_node)
    graph.add_node("retention",         _retention_node)
    graph.add_node("emotion",           _emotion_node)
    # Stage 4 nodes
    graph.add_node("generate",          recommendation_agent.run)
    graph.add_node("audit",             recommendation_audit_agent.run)
    graph.add_node("prepare_feedback",  _prepare_feedback_node)
    graph.add_node("save",              _save_node)

    # equity → retention (sequential, retention reads equity output)
    graph.add_edge(START,               "equity")
    graph.add_edge("equity",            "retention")
    # emotion is independent — starts from START in parallel with equity
    graph.add_edge(START,               "emotion")
    # generate waits for both retention and emotion to finish (fan-in)
    graph.add_edge("retention",         "generate")
    graph.add_edge("emotion",           "generate")
    graph.add_edge("generate",          "audit")
    graph.add_conditional_edges(
        "audit",
        _should_continue,
        {"save": "save", "regenerate": "prepare_feedback"},
    )
    graph.add_edge("prepare_feedback",  "generate")
    graph.add_edge("save",              END)

    return graph.compile()


_compiled_graph = _build_graph()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    month:       str,
    company:     str = "Apple",
    hr_csv:      str = _DEFAULT_HR_CSV,
    reviews_csv: str = _DEFAULT_REVIEWS_CSV,
) -> None:
    print(f"\n{'='*62}")
    print(f"  RetainIQ Pipeline  |  {company}  |  Month: {month}")
    print(f"  HR CSV      : {hr_csv}")
    print(f"  Reviews CSV : {reviews_csv}")
    print(f"  Max audits  : {MAX_AUDIT_ATTEMPTS}")
    print(f"{'='*62}")

    initial_state: PipelineState = {
        "month":           month,
        "company":         company,
        "hr_csv":          hr_csv,
        "reviews_csv":     reviews_csv,
        "recommendations": [],
        "audit_result":    {},
        "audit_attempts":  0,
        "feedback":        "",
    }

    _compiled_graph.invoke(initial_state)

    print(f"\n{'='*62}")
    print("  Pipeline complete.")
    print(f"{'='*62}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full retention pipeline: equity → retention → emotion → recommend → audit → save"
    )
    parser.add_argument("--month",       default=datetime.now(timezone.utc).strftime("%Y-%m"),
                        help="Analysis month YYYY-MM (default: current month)")
    parser.add_argument("--company",     default="Apple",
                        help="Company name (must match 'firm' column in reviews CSV)")
    parser.add_argument("--hr-csv",      default=_DEFAULT_HR_CSV,
                        help="Path to employee HR data CSV")
    parser.add_argument("--reviews-csv", default=_DEFAULT_REVIEWS_CSV,
                        help="Path to Glassdoor reviews CSV")
    args = parser.parse_args()
    run_pipeline(
        month=args.month,
        company=args.company,
        hr_csv=args.hr_csv,
        reviews_csv=args.reviews_csv,
    )