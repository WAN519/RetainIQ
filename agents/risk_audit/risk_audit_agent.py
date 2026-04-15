"""
agents/risk_audit/risk_audit_agent.py

Risk Audit Agent — adversarial quality gate for the retention risk scoring pipeline.

Audits AI-generated employee risk reports (Risk collection) before they reach
HR teams or managers, and rejects or flags anything unreliable or inconsistent.

Pipeline:
  Tool  →  fetch latest Risk run from MongoDB
  API   →  Claude claude-opus-4-6 (critic persona) — structured audit review
  Store →  save audit result to MongoDB: Audit_Reports

Audit verdicts:
  APPROVED      — report is ready for HR / manager distribution
  NEEDS_REVIEW  — minor issues found; human should review flagged records
  REJECTED      — critical data problems; report must NOT be distributed

MongoDB output schema (Audit_Reports):
    {
        audit_id, run_id, audit_date, verdict,
        audit_summary: { total_reviewed, approved_count, flagged_count,
                         critical_count, overall_assessment },
        quality_metrics: { risk_calibration_score, recommendation_quality_score,
                           urgency_alignment_score, data_completeness_score,
                           overall_quality_score },
        flagged_employees: [{ employee_id, issues, severity, suggested_correction }],
        audit_notes
    }

Usage:
  python -m agents.risk_audit.risk_audit_agent
  python -m agents.risk_audit.risk_audit_agent --run-id <uuid>
"""

import os
import json
import uuid
import datetime
from pathlib import Path

import certifi
import anthropic
from dotenv import load_dotenv
from pymongo import MongoClient, DESCENDING

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / "config.env")

# ──────────────────────────────────────────────────────────────────────────────
# Structured output schema for Claude's audit response
# ──────────────────────────────────────────────────────────────────────────────
_AUDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["APPROVED", "NEEDS_REVIEW", "REJECTED"],
        },
        "audit_summary": {
            "type": "object",
            "properties": {
                "total_reviewed":    {"type": "integer"},
                "approved_count":    {"type": "integer"},
                "flagged_count":     {"type": "integer"},
                "critical_count":    {"type": "integer"},
                "overall_assessment":{"type": "string"},
            },
            "required": ["total_reviewed", "approved_count", "flagged_count",
                         "critical_count", "overall_assessment"],
            "additionalProperties": False,
        },
        "quality_metrics": {
            "type": "object",
            "properties": {
                "risk_calibration_score":       {"type": "number"},
                "recommendation_quality_score": {"type": "number"},
                "urgency_alignment_score":      {"type": "number"},
                "data_completeness_score":      {"type": "number"},
                "overall_quality_score":        {"type": "number"},
            },
            "required": [
                "risk_calibration_score", "recommendation_quality_score",
                "urgency_alignment_score", "data_completeness_score",
                "overall_quality_score",
            ],
            "additionalProperties": False,
        },
        "flagged_employees": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "employee_id":         {"type": "string"},
                    "issues":              {"type": "array", "items": {"type": "string"}},
                    "severity":            {"type": "string", "enum": ["Critical", "Warning", "Info"]},
                    "suggested_correction":{"type": "string"},
                },
                "required": ["employee_id", "issues", "severity", "suggested_correction"],
                "additionalProperties": False,
            },
        },
        "audit_notes": {"type": "string"},
    },
    "required": ["verdict", "audit_summary", "quality_metrics",
                 "flagged_employees", "audit_notes"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """You are a rigorous HR audit specialist and adversarial critic. \
Your job is to review AI-generated employee risk reports BEFORE they reach HR teams \
or managers, and reject or flag anything that is unreliable, inconsistent, or \
potentially harmful.

You approach every report with healthy skepticism. You are NOT looking for reasons \
to approve — you are looking for reasons to reject or flag.

Apply the following checks to every employee record:

1. RISK LEVEL CONSISTENCY
   - risk_level must be coherent with cox_risk_bucket and salary_risk_tier.
   - A "High" risk_level with a "Low" cox_risk_bucket and "Low" salary_risk_tier \
is a contradiction — flag it Critical.

2. PRIORITY SCORE CALIBRATION
   - High risk_level → priority_score MUST be ≥ 7.
   - Low risk_level  → priority_score MUST be ≤ 4.
   - Violations are Critical issues.

3. URGENCY ALIGNMENT
   - "Immediate" urgency is only valid for High risk employees.
   - "Monitor" urgency on a High risk employee is a red flag — flag Warning.

4. RECOMMENDATION QUALITY
   - Reject generic advice like "have a conversation", "consider improving",
     "monitor the situation", "discuss with HR". These add no value.
   - Recommendations must be specific, actionable, and role-aware.
   - Flag as Warning if > 50% of an employee's recommendations are generic.

5. SALARY GAP PLAUSIBILITY
   - salary_gap_pct > 80% is suspicious — flag Warning and note it.
   - salary_gap_pct > 150% is almost certainly a data error — flag Critical.

6. ROOT CAUSE COVERAGE
   - High-risk employees must have at least 2 distinct root causes.
   - An empty root_causes list on any employee is a Critical issue.

7. PRIORITY INFLATION
   - If more than 40% of the batch has priority_score ≥ 8, the model is likely
     over-alarming. Flag the batch-level issue as Warning.

8. DATA COMPLETENESS
   - employee_id, risk_level, claude_analysis, analysis_date are mandatory.
   - Any missing or null required field is a Critical issue.

Scoring (0–10, where 10 = perfect):
  risk_calibration_score:        how well risk_level matches underlying quantitative signals
  recommendation_quality_score:  specificity and actionability of recommendations
  urgency_alignment_score:       how well urgency matches risk level
  data_completeness_score:       fraction of records with all required fields

Verdict rules:
  APPROVED      — zero Critical issues, overall_quality_score ≥ 7.0
  NEEDS_REVIEW  — no Critical issues but quality_score < 7.0, OR ≤ 3 Warnings
  REJECTED      — any Critical issue OR > 3 Warnings OR quality_score < 5.0

Be direct, specific, and unsparing. Name the employee IDs. Quote the exact problem. \
This report protects real employees from bad AI decisions."""


# ──────────────────────────────────────────────────────────────────────────────
# MongoDB helpers
# ──────────────────────────────────────────────────────────────────────────────
def _get_mongo_db():
    uri     = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_NAME", "MarketInformation")
    client  = MongoClient(uri, serverSelectionTimeoutMS=8000, tlsCAFile=certifi.where())
    return client[db_name]


def _fetch_run_docs(run_id: str | None) -> tuple[list[dict], str]:
    """
    Fetch all Risk documents for a given run_id.
    If run_id is None, fetches the most recent run.
    """
    db   = _get_mongo_db()
    coll = db[os.getenv("COLLECTION_NAME_RISK", "Risk")]

    if run_id is None:
        latest = coll.find_one(
            {"run_id": {"$exists": True}},
            sort=[("analysis_date", DESCENDING)],
        )
        if not latest:
            raise RuntimeError("No documents with run_id found in Risk collection.")
        run_id = latest["run_id"]

    docs = list(coll.find({"run_id": run_id}, {"_id": 0}))
    if not docs:
        raise RuntimeError(f"No documents found for run_id={run_id}")

    print(f"[MongoDB] Fetched {len(docs)} records for run_id={run_id}")
    return docs, run_id


# ──────────────────────────────────────────────────────────────────────────────
# Pre-flight rule checks (deterministic, run before Claude)
# ──────────────────────────────────────────────────────────────────────────────
_GENERIC_PHRASES = [
    "have a conversation", "consider improving", "monitor the situation",
    "discuss with hr", "discuss with manager", "review the situation",
    "look into", "may want to", "could consider",
]


def _preflight_checks(docs: list[dict]) -> dict:
    """Deterministic rule checks injected into the Claude audit prompt."""
    issues   = []
    high_pri = 0

    for doc in docs:
        eid      = doc.get("employee_id", "UNKNOWN")
        analysis = doc.get("claude_analysis") or {}
        rl       = (doc.get("risk_level") or "").strip()
        cox_bkt  = (doc.get("cox_risk_bucket") or "").strip()
        sal_tier = (doc.get("salary_risk_tier") or "").strip()
        urgency  = (analysis.get("urgency") or "").strip()
        prio     = analysis.get("priority_score")
        causes   = analysis.get("root_causes") or []
        recs     = analysis.get("recommendations") or []

        for field in ("employee_id", "risk_level", "claude_analysis", "analysis_date"):
            if not doc.get(field):
                issues.append({"employee_id": eid, "severity": "Critical",
                               "issue": f"Missing required field: {field}"})

        if rl == "High" and cox_bkt == "Low" and sal_tier == "Low":
            issues.append({"employee_id": eid, "severity": "Critical",
                           "issue": "risk_level=High contradicts both cox_risk_bucket=Low and salary_risk_tier=Low"})
        if rl == "Low" and cox_bkt == "High":
            issues.append({"employee_id": eid, "severity": "Critical",
                           "issue": "risk_level=Low contradicts cox_risk_bucket=High"})

        if prio is not None:
            if rl == "High" and prio < 7:
                issues.append({"employee_id": eid, "severity": "Critical",
                               "issue": f"High risk employee has priority_score={prio} (must be ≥ 7)"})
            if rl == "Low" and prio > 4:
                issues.append({"employee_id": eid, "severity": "Critical",
                               "issue": f"Low risk employee has priority_score={prio} (must be ≤ 4)"})
            if prio >= 8:
                high_pri += 1

        if urgency == "Immediate" and rl != "High":
            issues.append({"employee_id": eid, "severity": "Warning",
                           "issue": f"urgency=Immediate but risk_level={rl}"})
        if urgency == "Monitor" and rl == "High":
            issues.append({"employee_id": eid, "severity": "Warning",
                           "issue": "urgency=Monitor on a High risk employee"})

        gap_pct = abs(doc.get("salary_gap_pct") or 0)
        if gap_pct > 150:
            issues.append({"employee_id": eid, "severity": "Critical",
                           "issue": f"salary_gap_pct={gap_pct:.1f}% is likely a data error (> 150%)"})
        elif gap_pct > 80:
            issues.append({"employee_id": eid, "severity": "Warning",
                           "issue": f"salary_gap_pct={gap_pct:.1f}% is unusually large (> 80%)"})

        if not causes:
            issues.append({"employee_id": eid, "severity": "Critical",
                           "issue": "root_causes is empty"})
        elif rl == "High" and len(causes) < 2:
            issues.append({"employee_id": eid, "severity": "Warning",
                           "issue": f"High risk employee has only {len(causes)} root cause(s)"})

        generic_count = sum(
            1 for r in recs
            if any(phrase in r.lower() for phrase in _GENERIC_PHRASES)
        )
        if recs and generic_count / len(recs) > 0.5:
            issues.append({"employee_id": eid, "severity": "Warning",
                           "issue": f"{generic_count}/{len(recs)} recommendations use generic language"})

    inflation_rate = high_pri / len(docs) if docs else 0
    if inflation_rate > 0.40:
        issues.append({"employee_id": "BATCH", "severity": "Warning",
                       "issue": f"Priority inflation: {high_pri}/{len(docs)} ({inflation_rate:.0%}) employees have priority_score ≥ 8"})

    return {
        "total_records":    len(docs),
        "preflight_issues": issues,
        "critical_count":   sum(1 for i in issues if i["severity"] == "Critical"),
        "warning_count":    sum(1 for i in issues if i["severity"] == "Warning"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Risk Audit Agent
# ──────────────────────────────────────────────────────────────────────────────
class RiskAuditAgent:
    """
    Adversarial critic that reviews RetentionAgent output (Risk collection)
    before distribution to HR or managers.
    """

    MODEL = "claude-opus-4-6"

    def __init__(self):
        api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("CLAUDE_API_KEY (or ANTHROPIC_API_KEY) not set in config.env")
        self.client = anthropic.Anthropic(api_key=api_key)

    def _run_claude_audit(self, docs: list[dict], preflight: dict) -> dict:
        flagged_ids  = {i["employee_id"] for i in preflight["preflight_issues"]}
        payload_docs = []
        for doc in docs:
            eid = doc.get("employee_id", "")
            if eid in flagged_ids or doc.get("risk_level") == "High":
                payload_docs.append(doc)
            else:
                payload_docs.append({
                    "employee_id":           eid,
                    "risk_level":            doc.get("risk_level"),
                    "cox_risk_bucket":       doc.get("cox_risk_bucket"),
                    "salary_risk_tier":      doc.get("salary_risk_tier"),
                    "urgency":               (doc.get("claude_analysis") or {}).get("urgency"),
                    "priority_score":        (doc.get("claude_analysis") or {}).get("priority_score"),
                    "root_causes_count":     len((doc.get("claude_analysis") or {}).get("root_causes") or []),
                    "recommendations_count": len((doc.get("claude_analysis") or {}).get("recommendations") or []),
                })

        user_content = (
            "## Pre-flight Rule Check Results\n"
            f"{json.dumps(preflight, indent=2, default=str)}\n\n"
            "## Employee Risk Records\n"
            f"{json.dumps(payload_docs, indent=2, default=str)}\n\n"
            "Perform your audit. Be specific, name employee IDs, and give your verdict."
        )

        print(f"[RiskAuditAgent] Sending {len(payload_docs)} records to Claude...")

        with self.client.messages.stream(
            model=self.MODEL,
            max_tokens=8000,
            thinking={"type": "adaptive"},
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
            output_config={"format": {"type": "json_schema", "schema": _AUDIT_SCHEMA}},
        ) as stream:
            final = stream.get_final_message()

        text_block = next(b for b in final.content if b.type == "text")
        return json.loads(text_block.text)

    def _save_audit(self, run_id: str, audit_result: dict) -> str:
        audit_id   = str(uuid.uuid4())
        audit_date = datetime.datetime.now(datetime.timezone.utc)

        doc = {"audit_id": audit_id, "run_id": run_id, "audit_date": audit_date, **audit_result}

        db = _get_mongo_db()
        db["Audit_Reports"].insert_one(doc)
        print(f"[MongoDB] Audit report saved → Audit_Reports (audit_id={audit_id})")
        return audit_id

    def run(self, run_id: str | None = None) -> dict:
        """
        Audit the latest (or specified) RetentionAgent run.

        Args:
            run_id: Specific run_id to audit. If None, audits the most recent run.
        """
        docs, resolved_run_id = _fetch_run_docs(run_id)

        print(f"[RiskAuditAgent] Running pre-flight checks on {len(docs)} records...")
        preflight = _preflight_checks(docs)
        print(f"[RiskAuditAgent] Pre-flight: {preflight['critical_count']} critical | "
              f"{preflight['warning_count']} warnings")

        audit_result = self._run_claude_audit(docs, preflight)
        audit_result["audit_summary"]["preflight_critical"] = preflight["critical_count"]
        audit_result["audit_summary"]["preflight_warnings"]  = preflight["warning_count"]

        audit_id = self._save_audit(resolved_run_id, audit_result)
        self._print_report(audit_result, resolved_run_id, audit_id)
        return audit_result

    @staticmethod
    def _print_report(result: dict, run_id: str, audit_id: str):
        verdict = result["verdict"]
        summary = result["audit_summary"]
        metrics = result["quality_metrics"]

        LABEL = {
            "APPROVED":     "APPROVED — ready for HR / manager distribution",
            "NEEDS_REVIEW": "NEEDS_REVIEW — human review required before distribution",
            "REJECTED":     "REJECTED — do NOT distribute this report",
        }

        print("\n" + "=" * 70)
        print(f"  RISK AUDIT REPORT  |  run_id: {run_id}")
        print(f"  audit_id: {audit_id}")
        print("=" * 70)
        print(f"\n  {LABEL.get(verdict, verdict)}\n")
        print(f"  Records reviewed : {summary['total_reviewed']}")
        print(f"  Flagged          : {summary['flagged_count']}")
        print(f"  Critical issues  : {summary['critical_count']}")
        print()
        print("  Quality scores (0–10):")
        print(f"    Risk calibration      : {metrics['risk_calibration_score']:.1f}")
        print(f"    Recommendation quality: {metrics['recommendation_quality_score']:.1f}")
        print(f"    Urgency alignment     : {metrics['urgency_alignment_score']:.1f}")
        print(f"    Data completeness     : {metrics['data_completeness_score']:.1f}")
        print(f"    Overall quality       : {metrics['overall_quality_score']:.1f}")
        print()

        for f in result.get("flagged_employees") or []:
            print(f"  [{f['severity']:8}] {f['employee_id']}")
            for issue in f["issues"]:
                print(f"             • {issue}")
            print(f"             → {f['suggested_correction']}")

        print()
        print("  Audit notes:")
        for line in result.get("audit_notes", "").split(". "):
            if line.strip():
                print(f"    {line.strip()}.")
        print("=" * 70 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Module-level entry point
# ──────────────────────────────────────────────────────────────────────────────
def run_risk_audit_agent(run_id: str | None = None) -> dict:
    """Instantiate RiskAuditAgent and audit the specified (or latest) risk run."""
    agent = RiskAuditAgent()
    return agent.run(run_id=run_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Risk Audit Agent — retention risk report quality gate")
    parser.add_argument("--run-id", default=None, help="run_id to audit (default: most recent run)")
    args = parser.parse_args()
    run_risk_audit_agent(run_id=args.run_id)