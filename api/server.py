"""
api/server.py

FastAPI backend for the RetentionAgent UI.

Data flow:
  retention_recommendations (target=HR)         → HR dashboard
  retention_recommendations (target=Management)  → Manager dashboard (dept-filtered)

Join strategy:
  retention_recommendations.employee_id → Risk (latest run_id) for scores + salary data

Fallback:
  If retention_recommendations is empty, serve from Risk collection directly.

Endpoints:
  POST /api/login
  GET  /api/employees?role=hr|manager&month=YYYY-MM
  GET  /api/health

Usage:
  uvicorn api.server:app --reload --port 8000
"""

import os
import certifi
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / "config.env")

app = FastAPI(title="RetentionAgent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://wan519.github.io",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Demo credentials
# dept must match _infer_dept() output below
# ---------------------------------------------------------------------------
USERS: dict[str, dict] = {
    "hr_admin": {"password": "hr2026",  "role": "hr",      "dept": None},
    "mgr_lead": {"password": "mgr2026", "role": "manager", "dept": "Sales"},
    "mgr_eng":  {"password": "mgr2026", "role": "manager", "dept": "Engineering"},
}

EMPLOYEE_LIMIT = 199   # total employees shown across the platform

# ---------------------------------------------------------------------------
# MongoDB helpers
# ---------------------------------------------------------------------------

def _get_db():
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_NAME", "MarketInformation")
    client = MongoClient(uri, serverSelectionTimeoutMS=5000, tlsCAFile=certifi.where())
    return client[db_name]


def _latest_run_id(db) -> str | None:
    doc = db["Risk"].find_one({}, {"run_id": 1, "_id": 0}, sort=[("scoring_date", -1)])
    return doc["run_id"] if doc else None


# ---------------------------------------------------------------------------
# Department & role helpers
# ---------------------------------------------------------------------------

_CITIES = {"Atlanta", "Austin", "Boston", "Chicago", "Denver", "Miami", "NYC", "Seattle"}


def _infer_dept(role_name: str) -> str:
    r = role_name.upper()
    if "SALES" in r:
        return "Sales"
    if any(x in r for x in ("SOFTWARE", "ENGINEERING_MANAGER", "BUSINESS_ANALYST",
                             "PRODUCT_MANAGER", "OPERATIONS_MANAGER")):
        return "Engineering"
    if any(x in r for x in ("DATA_SCIENTIST", "RESEARCH", "LABORATORY",
                             "MANUFACTURING", "HEALTHCARE")):
        return "Research & Ops"
    if any(x in r for x in ("HR_", "HUMAN_RESOURCE")):
        return "HR"
    return "General"


def _format_role(role_name: str) -> str:
    """'Engineering_Manager_Austin' → 'Engineering Manager'"""
    parts = role_name.split("_")
    if parts and parts[-1] in _CITIES:
        parts = parts[:-1]
    return " ".join(parts)


def _city(role_name: str) -> str:
    parts = role_name.split("_")
    return parts[-1] if parts and parts[-1] in _CITIES else ""


# ---------------------------------------------------------------------------
# Document mapping helpers
# ---------------------------------------------------------------------------

def _make_flag(salary_risk_tier: str | None, risk_factors: list) -> str:
    text = " ".join((risk_factors or [])).lower()
    if "salary" in text and "internal" in text:
        return "Salary + Equity"
    if "salary" in text and ("career" in text or "promotion" in text):
        return "Salary + Career"
    if "salary" in text:
        return "Salary"
    if "career" in text or "promotion" in text:
        return "Career"
    if salary_risk_tier in ("Critical", "High"):
        return "Salary"
    return salary_risk_tier or "Monitor"


_URGENCY_TO_SENTIMENT = {
    "Immediate": "Frustrated",
    "Near-term": "Concerned",
    "Monitor":   "Neutral",
}

_PRIORITY_TO_URGENCY = {
    "Urgent": "Immediate",
    "High":   "Near-term",
    "Medium": "Monitor",
}


def _build_employee(risk_doc: dict, rec_doc: dict | None) -> dict:
    """
    Merge Risk doc + retention_recommendations doc into the frontend schema.

    HR view  : compensation plan from rec_doc, salary data from risk_doc
    Manager view: behavioral recommendation from rec_doc, risk data from risk_doc
    Both share: attrition score, role, dept, flag
    """
    role_name = risk_doc.get("role_name", "")
    ai        = risk_doc.get("claude_analysis") or {}

    # Attrition: cox_risk_pct is 0-1 → convert to 0-100
    risk_raw  = risk_doc.get("cox_risk_pct", 0) or 0
    attrition = round(risk_raw * 100) if risk_raw <= 1.0 else round(risk_raw)

    # salary_gap_pct: positive = below market → negate for UI ("−22%")
    sgap      = risk_doc.get("salary_gap_pct")
    salary_gap = round(-sgap, 1) if sgap is not None else 0.0

    pgap      = risk_doc.get("pay_gap_from_equity")
    int_gap   = round(-pgap, 1) if pgap is not None else 0.0

    # Recommendation fields — prefer rec_doc when available
    if rec_doc:
        plan          = rec_doc.get("recommendation", "")
        key_concerns  = rec_doc.get("key_concerns", [])
        priority      = rec_doc.get("priority", "Medium")
        urgency       = _PRIORITY_TO_URGENCY.get(priority, "Monitor")
        target        = rec_doc.get("target", "HR")
        weighted_score = rec_doc.get("weighted_score")
    else:
        # Fallback: derive from Risk claude_analysis
        recs          = ai.get("recommendations") or []
        plan          = recs[0] if recs else "No recommendation available."
        key_concerns  = ai.get("root_causes") or []
        urgency       = ai.get("urgency", "Monitor")
        priority      = {"Immediate": "Urgent", "Near-term": "High", "Monitor": "Medium"}.get(urgency, "Medium")
        target        = "HR"   # unknown without rec_doc
        weighted_score = None

    return {
        "id":            risk_doc.get("employee_id", ""),
        "name":          risk_doc.get("employee_id", ""),
        "role":          _format_role(role_name),
        "dept":          _infer_dept(role_name),
        "city":          _city(role_name),
        "attrition":     attrition,
        "salaryGap":     salary_gap,
        "internalGap":   int_gap,
        "sentiment":     _URGENCY_TO_SENTIMENT.get(urgency, "Neutral"),
        "flag":          _make_flag(risk_doc.get("salary_risk_tier"), risk_doc.get("risk_factors", [])),
        "plan":          plan,
        "managerNote":   ai.get("risk_summary", ""),
        "keyConcerns":   key_concerns,
        "riskBucket":    risk_doc.get("combined_risk_bucket", "Low"),
        "urgency":       urgency,
        "priority":      priority,
        "target":        target,
        "weightedScore": weighted_score,
        "priorityScore": ai.get("priority_score", 0),
        "rootCauses":    ai.get("root_causes", []),
        "currentSalary": risk_doc.get("current_salary"),
        "marketMedian":  risk_doc.get("market_median_2026"),
        "salaryRiskTier":risk_doc.get("salary_risk_tier"),
        "riskFactors":   risk_doc.get("risk_factors", []),
    }


# ---------------------------------------------------------------------------
# Core query logic
# ---------------------------------------------------------------------------

def _fetch_from_recommendations(db, target_role: str, dept: str | None,
                                  month: str, run_id: str) -> list[dict]:
    """
    Primary path: join retention_recommendations → Risk.

    target_role: "HR" or "Management"
    dept:        filter employees by department (manager only), None = all
    """
    # 1. Get latest Risk doc per employee across all runs
    all_risk = list(db["Risk"].find({}, {"_id": 0}, sort=[("scoring_date", -1)]))
    seen_ids: set = set()
    risk_docs = []
    for d in all_risk:
        eid = d.get("employee_id")
        if eid and eid not in seen_ids:
            seen_ids.add(eid)
            risk_docs.append(d)
        if len(risk_docs) >= EMPLOYEE_LIMIT:
            break
    risk_map  = {d["employee_id"]: d for d in risk_docs}

    # Build dept map  employee_id → dept
    dept_map  = {eid: _infer_dept(d.get("role_name", "")) for eid, d in risk_map.items()}

    # 2. Determine which employee_ids belong to this manager's dept
    if dept:
        allowed_ids = {eid for eid, d in dept_map.items() if d == dept}
    else:
        allowed_ids = None   # HR sees all

    # 3. Query retention_recommendations
    rec_query: dict = {"target": target_role}
    if month:
        rec_query["month"] = month
    if allowed_ids is not None:
        rec_query["employee_id"] = {"$in": list(allowed_ids)}

    rec_docs = list(db["retention_recommendations"].find(rec_query, {"_id": 0}))

    if not rec_docs:
        return []

    # 4. Merge — skip if no matching Risk doc
    results = []
    for rec in rec_docs:
        eid      = rec.get("employee_id", "")
        risk_doc = risk_map.get(eid)
        if not risk_doc:
            continue
        results.append(_build_employee(risk_doc, rec))

    return results


def _fetch_from_risk_fallback(db, role: str, dept: str | None, run_id: str) -> list[dict]:
    """
    Fallback path when retention_recommendations is empty.
    HR: top EMPLOYEE_LIMIT by risk.  Manager: their dept slice of those 199.
    """
    # Always pull top EMPLOYEE_LIMIT employees by cox_risk_pct (highest risk first)
    docs = list(db["Risk"].find(
        {"run_id": run_id},
        {"_id": 0},
        sort=[("cox_risk_pct", -1)],
        limit=EMPLOYEE_LIMIT,
    ))
    results = [_build_employee(d, None) for d in docs]

    if role == "manager":
        # Managers only see High+Mid risk within their dept
        results = [e for e in results
                   if e["riskBucket"] in ("High", "Mid") and
                      (dept is None or e["dept"] == dept)]

    return results


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/login")
def login(body: LoginRequest):
    user = USERS.get(body.username)
    if not user or user["password"] != body.password:
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    return {"role": user["role"], "dept": user.get("dept"), "username": body.username}


@app.get("/api/employees")
def get_employees(role: str = "hr", dept: str | None = None, month: str = "2026-04"):
    """
    role=hr      → recommendations with target=HR (all depts)
    role=manager → recommendations with target=Management, filtered by dept
    Falls back to Risk collection when retention_recommendations is empty.
    """
    db     = _get_db()
    run_id = _latest_run_id(db)
    if not run_id:
        raise HTTPException(status_code=404, detail="No Risk data found.")

    target_role = "HR" if role == "hr" else "Management"

    # Try primary source: retention_recommendations
    employees = _fetch_from_recommendations(db, target_role, dept, month, run_id)
    source    = "recommendations"

    # Fallback to Risk if empty
    if not employees:
        employees = _fetch_from_risk_fallback(db, role, dept, run_id)
        source    = "risk_fallback"

    # Sort by attrition desc
    employees.sort(key=lambda e: e["attrition"], reverse=True)

    return {
        "employees": employees,
        "total":     len(employees),
        "source":    source,
        "run_id":    run_id,
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}