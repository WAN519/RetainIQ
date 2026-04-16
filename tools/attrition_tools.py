"""
tools/attrition_tools.py

Tool implementations, JSON schemas, and dispatcher for the Claude attrition
analysis agent (agents/emotion/claude_attrition_agent.py).

Each function is callable by Claude via tool_use. The module also exposes:
  TOOLS         — list of tool schema dicts passed directly to the Anthropic API
  execute_tool  — dispatcher that routes a tool_use block to the right function

Tools:
  run_emotion_analysis   — run NLP pipeline on raw Glassdoor CSV → store in MongoDB
  get_emotion_summary    — aggregate per-review labels into company-level stats
  get_high_risk_reviews  — sample the most negatively-scored reviews
  get_retention_risks    — Cox model risk scores + distribution from Retention_Predictions
  get_equity_gaps        — employees underpaid vs. 2026 market benchmarks
"""

import os
import json
from datetime import datetime

import certifi
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv("config.env")


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _get_mongo_db():
    """Return a MongoDB database handle using credentials from config.env."""
    uri     = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_NAME", "MarketInformation")
    client  = MongoClient(uri, serverSelectionTimeoutMS=5000, tlsCAFile=certifi.where())
    return client[db_name]


# ---------------------------------------------------------------------------
# Tool 1 — run NLP sentiment pipeline
# ---------------------------------------------------------------------------

def run_emotion_analysis(company_name: str, csv_path: str, month: str | None = None) -> dict:
    """
    Run the two-stage Glassdoor NLP pipeline for a company and persist results
    to the MongoDB `reviews_analysis` collection.

    Internally calls tools.emotion_tool.run_emotion_analysis, which uses
    SentenceTransformer topic labeling and RoBERTa sentiment scoring.

    Args:
        company_name: Company name matching the 'firm' column in the CSV.
        csv_path:     Path to the Glassdoor reviews CSV file.
        month:        Analysis month (YYYY-MM). Defaults to current month.

    Returns:
        dict with status, review count, and guidance for the next tool to call.
    """
    from tools.emotion_tool import run_emotion_analysis as _run_nlp
    return _run_nlp(company_name=company_name, csv_path=csv_path, month=month)


# ---------------------------------------------------------------------------
# Tool 2 — aggregate sentiment stats
# ---------------------------------------------------------------------------

def get_emotion_summary(company_name: str, month: str) -> dict:
    """
    Aggregate per-review NLP labels for a company into company-level metrics.

    Queries the MongoDB `reviews_analysis` collection written by run_emotion_analysis.

    Args:
        company_name: Target company name.
        month:        Analysis month in YYYY-MM format.

    Returns:
        dict with total review count, average sentiment, negative rate, and
        per-topic issue rates (management, salary, workload, career).
    """
    db = _get_mongo_db()
    pipeline = [
        {"$match": {"firm": company_name, "analysis_month": month}},
        {"$group": {
            "_id":            "$firm",
            "total_reviews":  {"$sum": 1},
            "avg_sentiment":  {"$avg": "$roberta_score"},
            "negative_count": {"$sum": {"$cond": [{"$eq": ["$roberta_label", "Negative"]}, 1, 0]}},
            "mgmt_issues":    {"$sum": "$sem_management"},
            "salary_issues":  {"$sum": "$sem_salary"},
            "workload_issues":{"$sum": "$sem_workload"},
            "career_issues":  {"$sum": "$sem_career"},
        }}
    ]
    result = list(db["reviews_analysis"].aggregate(pipeline))
    if not result:
        return {"error": f"No emotion data found for {company_name} in {month}. Run run_emotion_analysis first."}

    r     = result[0]
    total = r["total_reviews"]
    return {
        "company":              company_name,
        "month":                month,
        "total_reviews":        total,
        "avg_sentiment_score":  round(r["avg_sentiment"], 3),
        "negative_rate":        round(r["negative_count"] / total, 3),
        "mgmt_issue_rate":      round(r["mgmt_issues"]     / total, 3),
        "salary_issue_rate":    round(r["salary_issues"]   / total, 3),
        "workload_issue_rate":  round(r["workload_issues"] / total, 3),
        "career_issue_rate":    round(r["career_issues"]   / total, 3),
    }


# ---------------------------------------------------------------------------
# Tool 3 — high-risk review samples
# ---------------------------------------------------------------------------

def get_high_risk_reviews(company_name: str, month: str, limit: int = 8) -> dict:
    """
    Fetch the most negatively-scored employee reviews for a company.

    Provides Claude with concrete review text to ground its attrition analysis
    in specific employee language rather than aggregate numbers alone.

    Args:
        company_name: Target company.
        month:        Analysis month in YYYY-MM format.
        limit:        Maximum number of reviews to return (default 8).

    Returns:
        dict with a list of review snippets plus their topic flags.
    """
    db = _get_mongo_db()
    cursor = (
        db["reviews_analysis"]
        .find(
            {"firm": company_name, "analysis_month": month, "roberta_label": "Negative"},
            {"_id": 0, "cons": 1, "roberta_score": 1,
             "sem_management": 1, "sem_salary": 1, "sem_workload": 1, "sem_career": 1}
        )
        .sort("roberta_score", -1)
        .limit(limit)
    )
    reviews = list(cursor)
    if not reviews:
        return {"error": "No negative reviews found. Ensure run_emotion_analysis has been called."}

    return {"company": company_name, "month": month, "sample_reviews": reviews}


# ---------------------------------------------------------------------------
# Tool 4 — Cox model retention risk scores
# ---------------------------------------------------------------------------

def get_retention_risks(limit: int = 15) -> dict:
    """
    Retrieve employee-level attrition risk scores from the Cox survival model.

    Returns the risk bucket distribution (High / Mid / Low) and the top
    high-risk employees with their contributing risk factors.

    Args:
        limit: Maximum number of high-risk employees to return (default 15).

    Returns:
        dict with risk distribution summary and high-risk employee records.
    """
    db = _get_mongo_db()

    # Company-wide distribution by risk bucket
    pipeline = [
        {"$group": {
            "_id":        "$risk_bucket",
            "count":      {"$sum": 1},
            "avg_gap":    {"$avg": "$pay_gap"},
            "avg_months": {"$avg": "$months_since_promotion"},
        }}
    ]
    distribution = {r["_id"]: r for r in db["Retention_Predictions"].aggregate(pipeline)}

    # Individual high-risk employees
    high_risk = list(
        db["Retention_Predictions"]
        .find(
            {"risk_bucket": "High"},
            {"_id": 0, "employee_id": 1, "risk_score": 1, "risk_pct": 1,
             "pay_gap": 1, "months_since_promotion": 1, "rule_flag": 1, "risk_reasons": 1}
        )
        .sort("risk_score", -1)
        .limit(limit)
    )

    return {
        "risk_distribution":   distribution,
        "high_risk_employees": high_risk,
        "total_high_risk":     distribution.get("High", {}).get("count", 0),
    }


# ---------------------------------------------------------------------------
# Tool 5 — salary equity gaps
# ---------------------------------------------------------------------------

def get_equity_gaps(limit: int = 10) -> dict:
    """
    Fetch employees who are significantly underpaid vs. the 2026 market benchmark.

    A negative external_gap_pct indicates underpayment. Values below -10% signal
    a meaningful retention risk that should be correlated with attrition scores.

    Args:
        limit: Maximum number of records to return (default 10).

    Returns:
        dict with underpaid employee records and total count.
    """
    db = _get_mongo_db()
    cursor = (
        db["Equity_Predictions"]
        .find(
            {"equity_gaps.external_gap_pct": {"$lt": -10}},
            {"_id": 0, "employee_id": 1, "role_name": 1,
             "actual_salary": 1, "benchmarks": 1, "equity_gaps": 1}
        )
        .sort("equity_gaps.external_gap_pct", 1)
        .limit(limit)
    )
    underpaid = list(cursor)
    return {
        "underpaid_employees": underpaid,
        "count":               len(underpaid),
        "note":                "external_gap_pct < -10 means employee earns >10% below market rate",
    }


# ---------------------------------------------------------------------------
# Tool schemas — passed directly to the Anthropic messages API
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "run_emotion_analysis",
        "description": (
            "Run the Glassdoor NLP sentiment pipeline for a specific company. "
            "Reads the raw CSV, filters by company name, applies SentenceTransformer "
            "topic labeling and RoBERTa sentiment scoring, then stores the enriched "
            "reviews in MongoDB. ALWAYS call this tool first before calling "
            "get_emotion_summary or get_high_risk_reviews."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {
                    "type": "string",
                    "description": "Company name exactly as it appears in the 'firm' column of the CSV.",
                },
                "csv_path": {
                    "type": "string",
                    "description": "Path to the Glassdoor reviews CSV file.",
                },
                "month": {
                    "type": "string",
                    "description": "Analysis month in YYYY-MM format. Omit to use the current month.",
                },
            },
            "required": ["company_name", "csv_path"],
        },
    },
    {
        "name": "get_emotion_summary",
        "description": (
            "Retrieve aggregated Glassdoor sentiment statistics for a company in a given month. "
            "Returns total reviews, average sentiment score, negative review rate, and "
            "issue rates for management, salary, workload, and career growth topics. "
            "Call this after run_emotion_analysis has completed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string"},
                "month":        {"type": "string", "description": "Month in YYYY-MM format."},
            },
            "required": ["company_name", "month"],
        },
    },
    {
        "name": "get_high_risk_reviews",
        "description": (
            "Fetch sample employee reviews with the highest negative sentiment scores. "
            "Use these to identify specific language patterns and grievances driving attrition."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string"},
                "month":        {"type": "string", "description": "Month in YYYY-MM format."},
                "limit":        {"type": "integer", "description": "Max reviews to return (default 8).", "default": 8},
            },
            "required": ["company_name", "month"],
        },
    },
    {
        "name": "get_retention_risks",
        "description": (
            "Retrieve employee-level attrition risk scores from the Cox survival model. "
            "Returns the High / Mid / Low risk bucket distribution and individual high-risk "
            "employee records with contributing factors (pay gap, promotion lag, overtime)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max high-risk employees to return (default 15).", "default": 15},
            },
            "required": [],
        },
    },
    {
        "name": "get_equity_gaps",
        "description": (
            "Fetch employees whose salary is significantly below the 2026 market benchmark. "
            "Use this to correlate pay inequity with attrition risk signals from get_retention_risks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max records to return (default 10).", "default": 10},
            },
            "required": [],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "run_emotion_analysis":  run_emotion_analysis,
    "get_emotion_summary":   get_emotion_summary,
    "get_high_risk_reviews": get_high_risk_reviews,
    "get_retention_risks":   get_retention_risks,
    "get_equity_gaps":       get_equity_gaps,
}


def execute_tool(name: str, inputs: dict) -> str:
    """
    Dispatch a Claude tool_use block to the correct function.

    Args:
        name:   Tool name from the tool_use block.
        inputs: Tool arguments from the tool_use block.

    Returns:
        JSON string with the tool result or an error message.
    """
    fn = TOOL_MAP.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = fn(**inputs)
        return json.dumps(result, default=str, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})