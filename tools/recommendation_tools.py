"""
tools/recommendation_tools.py

Tool implementations, JSON schemas, and dispatcher for the retention
recommendation agent (agents/retention/recommendation_agent.py).

Tools:
  get_employee_profiles   — join Retention_Predictions + Emotion + employee_comment
  save_recommendations    — persist Claude's recommendations to MongoDB
"""

import os
import json
from datetime import datetime, timezone

import certifi
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv("config.env")


def _get_db():
    uri     = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_NAME", "MarketInformation")
    client  = MongoClient(uri, serverSelectionTimeoutMS=5000, tlsCAFile=certifi.where())
    return client[db_name]


# ---------------------------------------------------------------------------
# Tool 1 — fetch combined employee profiles
# ---------------------------------------------------------------------------

def get_employee_profiles(month: str, risk_buckets: list[str] | None = None) -> dict:
    """
    Join three collections to build employee profiles for recommendation generation.

    Sources:
      - Risk             (employee-level scores from ai_salary_risk_agent — field-mapped)
      - Emotion          (company-level sentiment report, by month)
      - employee_comment (latest comment per employee_id)

    Field mapping from Risk → canonical profile schema:
      combined_risk_bucket → risk_bucket
      cox_risk_score       → risk_score
      cox_risk_pct         → risk_pct
      risk_factors         → risk_reasons

    Args:
        month:        Analysis month in YYYY-MM format.
        risk_buckets: Risk tiers to include. Defaults to ["High", "Mid"].

    Returns:
        dict with company_sentiment_report and a list of combined employee profiles.
    """
    if risk_buckets is None:
        risk_buckets = ["High", "Mid"]

    db = _get_db()

    # 1. Employee-level risk data from Risk collection (written by ai_salary_risk_agent)
    #    Filter by month using scoring_date date range
    from datetime import datetime as _dt
    year, mon = map(int, month.split("-"))
    month_start = _dt(year, mon, 1)
    month_end   = _dt(year + (1 if mon == 12 else 0), 1 if mon == 12 else mon + 1, 1)

    risk_docs = list(db[os.getenv("COLLECTION_NAME_RISK", "Risk")].find(
        {
            "scoring_date":         {"$gte": month_start, "$lt": month_end},
            "combined_risk_bucket": {"$in": risk_buckets},
        },
        {"_id": 0,
         "employee_id": 1, "scoring_date": 1,
         "combined_risk_bucket": 1, "cox_risk_score": 1, "cox_risk_pct": 1,
         "risk_factors": 1, "claude_analysis": 1,
         "salary_gap_pct": 1, "salary_risk_tier": 1},
        sort=[("cox_risk_pct", -1)],
        limit=199,
    ))

    if not risk_docs:
        return {"error": f"No Risk records found for month={month}, buckets={risk_buckets}"}

    # 2. Company-level sentiment report from Emotion (one doc per month)
    emotion_doc = db["Emotion"].find_one({"month": month}, {"_id": 0, "company_sentiment_report": 1})
    company_sentiment_report = emotion_doc.get("company_sentiment_report") if emotion_doc else None

    # 3. Latest comment per employee from employee_comment
    emp_ids = [d["employee_id"] for d in risk_docs]
    comment_docs = list(db["employee_comment"].find(
        {"employee_id": {"$in": emp_ids}},
        {"_id": 0, "employee_id": 1, "comment": 1, "created_at": 1}
    ))

    comment_map: dict = {}
    for c in comment_docs:
        eid = c["employee_id"]
        if eid not in comment_map or c.get("created_at", "") > comment_map[eid].get("created_at", ""):
            comment_map[eid] = c

    # 4. Merge — normalise Risk field names to canonical profile schema
    profiles = []
    for doc in risk_docs:
        eid = doc["employee_id"]
        ai  = doc.get("claude_analysis") or {}
        profiles.append({
            "employee_id":           eid,
            "month":                 month,
            "risk_score":            doc.get("cox_risk_score"),
            "risk_pct":              doc.get("cox_risk_pct"),
            "risk_bucket":           doc.get("combined_risk_bucket"),
            "salary_gap_pct":        doc.get("salary_gap_pct"),
            "salary_risk_tier":      doc.get("salary_risk_tier"),
            "risk_reasons":          doc.get("risk_factors", []),
            "risk_summary":          ai.get("risk_summary"),
            "comment":               comment_map.get(eid, {}).get("comment"),
        })

    return {
        "month":                    month,
        "count":                    len(profiles),
        "company_sentiment_report": company_sentiment_report,
        "profiles":                 profiles,
    }


# ---------------------------------------------------------------------------
# Tool 2 — save recommendations
# ---------------------------------------------------------------------------

def save_recommendations(recommendations: list[dict]) -> dict:
    """
    Persist Claude's retention recommendations to the retention_recommendations collection.

    Each recommendation must include employee_id, weighted_score, weight_factors,
    key_concerns, recommendation text, target (HR/Management), and priority.

    Args:
        recommendations: List of recommendation dicts produced by Claude.

    Returns:
        dict with inserted count or error.
    """
    if not recommendations:
        return {"error": "recommendations list is empty"}

    db = _get_db()
    collection_name = "retention_recommendations"

    if collection_name not in db.list_collection_names():
        db.create_collection(collection_name)

    now = datetime.now(timezone.utc).isoformat()
    docs = [{**r, "created_at": now} for r in recommendations]

    result = db[collection_name].insert_many(docs)
    return {
        "status":         "saved",
        "inserted_count": len(result.inserted_ids),
        "collection":     collection_name,
    }


# ---------------------------------------------------------------------------
# Pre-flight checks  (called by the agent before Claude audit)
# ---------------------------------------------------------------------------

_GENERIC_PHRASES = [
    "have a conversation", "consider improving", "monitor the situation",
    "discuss with hr", "discuss with manager", "could consider",
    "may want to", "look into", "address concerns",
]


def preflight_recommendations(recommendations: list[dict]) -> dict:
    """
    Deterministic rule checks run before Claude's audit critique.
    Returns a summary injected into the audit prompt.
    """
    issues = []

    for rec in recommendations:
        eid = rec.get("employee_id", "UNKNOWN")

        # 1. Required fields
        for field in ("employee_id", "weighted_score", "weight_factors",
                      "key_concerns", "recommendation", "target", "priority"):
            if not rec.get(field):
                issues.append({"employee_id": eid, "severity": "Critical",
                               "issue": f"Missing required field: {field}"})

        # 2. weighted_score range
        score = rec.get("weighted_score")
        if score is not None and not (0 <= score <= 1):
            issues.append({"employee_id": eid, "severity": "Critical",
                           "issue": f"weighted_score={score} out of range [0, 1]"})

        # 3. weight_factors sum ~1.0
        wf = rec.get("weight_factors") or {}
        if wf:
            total = sum(wf.values())
            if abs(total - 1.0) > 0.05:
                issues.append({"employee_id": eid, "severity": "Warning",
                               "issue": f"weight_factors sum to {total:.2f}, expected ~1.0"})

        # 4. Priority / score / bucket consistency
        priority = rec.get("priority")
        bucket   = rec.get("risk_bucket")
        if priority == "Urgent" and bucket != "High":
            issues.append({"employee_id": eid, "severity": "Warning",
                           "issue": f"priority=Urgent but risk_bucket={bucket}"})
        if priority == "Urgent" and score is not None and score < 0.80:
            issues.append({"employee_id": eid, "severity": "Warning",
                           "issue": f"priority=Urgent but weighted_score={score:.2f} < 0.80"})

        # 5. Generic recommendation detection
        text = rec.get("recommendation", "").lower()
        if any(p in text for p in _GENERIC_PHRASES):
            issues.append({"employee_id": eid, "severity": "Warning",
                           "issue": "recommendation contains generic phrases"})

    return {
        "total":          len(recommendations),
        "issues":         issues,
        "critical_count": sum(1 for i in issues if i["severity"] == "Critical"),
        "warning_count":  sum(1 for i in issues if i["severity"] == "Warning"),
    }


def mongo_save(recommendations: list[dict], audit_verdict: str, audit_score: float) -> dict:
    """
    Persist recommendations to MongoDB with audit metadata attached.
    Called only after the audit gate (APPROVED or forced on second attempt).
    """
    if not recommendations:
        return {"error": "empty list"}

    db = _get_db()
    collection_name = "retention_recommendations"

    if collection_name not in db.list_collection_names():
        db.create_collection(collection_name)

    now  = datetime.now(timezone.utc).isoformat()
    docs = [{**r, "audit_verdict": audit_verdict,
                  "audit_score":   audit_score,
                  "created_at":    now} for r in recommendations]

    result = db[collection_name].insert_many(docs)
    return {
        "status":         "saved",
        "inserted_count": len(result.inserted_ids),
        "collection":     collection_name,
        "audit_verdict":  audit_verdict,
    }


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "get_employee_profiles",
        "description": (
            "Fetch combined employee profiles for a given month by joining three collections: "
            "Risk (employee-level Cox risk scores, salary gap, and Claude HR analysis), "
            "Emotion (company-level Glassdoor sentiment report for the month), and "
            "employee_comment (each employee's own words). "
            "Only returns High and Mid risk employees by default. "
            "Call this first before generating recommendations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "month": {
                    "type": "string",
                    "description": "Analysis month in YYYY-MM format.",
                },
                "risk_buckets": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["High", "Mid", "Low"]},
                    "description": "Risk tiers to include. Defaults to [High, Mid].",
                },
            },
            "required": ["month"],
        },
    },
    {
        "name": "save_recommendations",
        "description": (
            "Submit your retention recommendations for audit review and saving. "
            "An adversarial audit runs automatically — if approved the recommendations "
            "are saved to MongoDB immediately. If rejected you will receive specific "
            "revision instructions and must call this tool again with corrected output. "
            "Call this after you have analyzed all employee profiles."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "recommendations": {
                    "type": "array",
                    "description": "List of recommendation objects, one per employee.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "employee_id":    {"type": ["integer", "string"]},
                            "month":          {"type": "string"},
                            "risk_bucket":    {"type": "string"},
                            "weighted_score": {
                                "type": "number",
                                "description": "Final weighted attrition likelihood 0–1.",
                            },
                            "weight_factors": {
                                "type": "object",
                                "description": "Factor name → weight you assigned (must sum to 1).",
                            },
                            "key_concerns":   {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Top 1–3 root causes driving this employee's risk.",
                            },
                            "recommendation": {
                                "type": "string",
                                "description": "Specific, actionable retention action (2–4 sentences).",
                            },
                            "target": {
                                "type": "string",
                                "enum": ["HR", "Management"],
                                "description": "Who should act on this recommendation.",
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["Urgent", "High", "Medium"],
                            },
                        },
                        "required": ["employee_id", "month", "risk_bucket",
                                     "weighted_score", "weight_factors",
                                     "key_concerns", "recommendation", "target", "priority"],
                    },
                },
            },
            "required": ["recommendations"],
        },
    },
]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "get_employee_profiles": get_employee_profiles,
    "save_recommendations":  save_recommendations,
}


def execute_tool(name: str, inputs: dict) -> str:
    fn = TOOL_MAP.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return json.dumps(fn(**inputs), default=str, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})