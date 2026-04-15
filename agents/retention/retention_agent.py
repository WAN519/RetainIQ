"""
agents/retention/retention_agent.py

Retention Agent — Cox survival model + salary gap analysis + Claude HR insights.

Pipeline:
  Tool  →  RiskScorer.run()         scores each employee (Cox + salary gap)
  API   →  Claude claude-opus-4-6   interprets scores, produces HR analysis JSON
  Store →  RiskScorer.save_results() inserts enriched JSON to MongoDB: Risk

MongoDB output schema (Risk collection):
    {
        run_id, employee_id, scoring_date, role_name,
        current_salary, market_median_2026,
        salary_gap_abs, salary_gap_pct, salary_risk_tier,
        internal_salary_rank, performance_consistency,
        pay_gap_from_equity,
        cox_risk_score, cox_risk_pct, cox_risk_bucket,
        combined_risk_bucket, risk_factors,
        model_version, analysis_type,
        risk_level,
        claude_analysis: {
            risk_summary, root_causes, recommendations, urgency, priority_score
        },
        analysis_date,
    }

Usage:
  python -m agents.retention.retention_agent
"""

import os
import json
import uuid
import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from agents.retention.risk_scorer import RiskScorer

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / "config.env")

_SYSTEM_PROMPT = """You are a senior HR analyst specialising in employee retention and \
compensation equity. You receive quantitative risk scores produced by a Cox proportional \
hazard survival model combined with market salary gap analysis.

For each employee, provide:
- risk_level: overall retention risk classification — "High", "Mid", or "Low".
  Use combined_risk_bucket as the primary signal; override upward if salary gap or Cox \
score clearly warrants it.
- risk_summary: 1–2 sentences explaining the main risk drivers.
- root_causes: key factors behind the risk (be specific, not generic).
- recommendations: prioritised HR actions to address the risk.
- urgency: "Immediate" (act within 2 weeks), "Near-term" (1–3 months), "Monitor" (3+ months).
- priority_score: 1–10 across the whole cohort (10 = most urgent).

Be concise and actionable. Avoid generic advice."""


# ---------------------------------------------------------------------------
# Retention Agent
# ---------------------------------------------------------------------------

class RetentionAgent:
    """
    Orchestrates Cox-based risk scoring (RiskScorer) → Claude API analysis → MongoDB.

    Attributes:
        scorer (RiskScorer): Scoring tool — runs Cox model + salary gap analysis.
        client (anthropic.Anthropic): Anthropic API client.
    """

    MODEL = "claude-opus-4-6"

    def __init__(self, model_path: str, feature_json_path: str):
        self.scorer = RiskScorer(model_path, feature_json_path)
        api_key = os.getenv("CLAUDE_API_KEY")
        if not api_key:
            raise EnvironmentError("CLAUDE_API_KEY is not set in config.env")
        self.client = anthropic.Anthropic(api_key=api_key)

    def _run_scoring_tool(self, csv_path: str) -> tuple[list, list]:
        """Run RiskScorer and return (clean_docs_for_claude, raw_docs_for_merge)."""
        print("[RetentionAgent] RiskScorer — running Cox + salary gap scoring...")
        raw_docs = self.scorer.run(csv_path)
        print(f"[RetentionAgent] Scored {len(raw_docs)} employees.")
        clean = [{k: v for k, v in doc.items() if not isinstance(v, datetime.datetime)}
                 for doc in raw_docs]
        return clean, raw_docs

    @staticmethod
    def _trim_for_claude(doc: dict) -> dict:
        """Keep only the fields Claude needs for HR analysis — reduces input tokens."""
        return {
            "employee_id":          doc.get("employee_id"),
            "combined_risk_bucket": doc.get("combined_risk_bucket"),
            "cox_risk_score":       doc.get("cox_risk_score"),
            "cox_risk_pct":         doc.get("cox_risk_pct"),
            "salary_gap_pct":       doc.get("salary_gap_pct"),
            "salary_risk_tier":     doc.get("salary_risk_tier"),
            "pay_gap_from_equity":  doc.get("pay_gap_from_equity"),
            "risk_factors":         doc.get("risk_factors", []),
        }

    def _analyze_batch(self, batch: list) -> list:
        """Send one batch of employees to Claude and return employee_analyses list."""
        trimmed = [self._trim_for_claude(d) for d in batch]
        user_content = (
            "Analyse the following employee salary risk records.\n"
            "Return ONLY a JSON object with key \"employee_analyses\" containing an array. "
            "No markdown, no explanation — pure JSON only.\n"
            "Each item must have: employee_id, risk_level (High/Mid/Low), "
            "risk_summary (1-2 sentences), root_causes (array of strings), "
            "recommendations (array of strings), "
            "urgency (Immediate/Near-term/Monitor), priority_score (integer 1-10).\n\n"
            f"Records:\n{json.dumps(trimmed, default=str)}"
        )

        response = self.client.messages.create(
            model=self.MODEL,
            max_tokens=16000,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )

        print(f"[RetentionAgent] Batch done. "
              f"stop_reason={response.stop_reason} | "
              f"Input: {response.usage.input_tokens} | Output: {response.usage.output_tokens}")

        text_blocks = [b for b in response.content if b.type == "text"]
        if not text_blocks:
            print(f"[RetentionAgent] ERROR: no text block. Types: {[b.type for b in response.content]}")
            return []

        raw   = text_blocks[0].text.strip()
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            print(f"[RetentionAgent] ERROR: no JSON object found. raw[:200]={raw[:200]}")
            return []
        raw = raw[start:end]
        try:
            return json.loads(raw).get("employee_analyses", [])
        except Exception as e:
            print(f"[RetentionAgent] ERROR parsing JSON: {e} | raw[:300]={raw[:300]}")
            return []

    def _analyze_with_claude(self, scored_docs: list) -> dict:
        """
        Analyse only High + Mid risk employees in batches of 50.
        Low risk employees get a default analysis (no API call needed).
        """
        BATCH_SIZE = 50

        # Split by risk level — only send High/Mid to Claude
        to_analyze = [d for d in scored_docs if d.get("combined_risk_bucket") in ("High", "Mid")]
        low_risk   = [d for d in scored_docs if d.get("combined_risk_bucket") not in ("High", "Mid")]

        print(f"[RetentionAgent] High/Mid: {len(to_analyze)} → Claude  |  "
              f"Low: {len(low_risk)} → default")

        all_analyses = []

        # Batch process High/Mid employees
        for i in range(0, len(to_analyze), BATCH_SIZE):
            batch = to_analyze[i:i + BATCH_SIZE]
            print(f"[RetentionAgent] Batch {i // BATCH_SIZE + 1} "
                  f"({len(batch)} employees)...")
            results = self._analyze_batch(batch)
            all_analyses.extend(results)

        # Default analysis for Low risk employees
        for doc in low_risk:
            all_analyses.append({
                "employee_id":    doc.get("employee_id", ""),
                "risk_level":     "Low",
                "risk_summary":   "Employee shows low attrition risk based on Cox model and salary gap analysis.",
                "root_causes":    [],
                "recommendations": ["Continue regular check-ins; no immediate retention action required."],
                "urgency":        "Monitor",
                "priority_score": 2,
            })

        return {"employee_analyses": all_analyses}

    def _merge_and_save(self, raw_docs: list, analysis: dict) -> list:
        """Merge Claude analysis into scored docs and insert into MongoDB: Risk."""
        employee_map  = {item["employee_id"]: item for item in analysis.get("employee_analyses", [])}
        analysis_date = datetime.datetime.now(datetime.timezone.utc)
        run_id        = str(uuid.uuid4())

        risk_coll = self.scorer.db.db[os.getenv("COLLECTION_NAME_RISK", "Risk")]
        print(f"[RetentionAgent] Inserting {len(raw_docs)} docs into Risk (run_id={run_id}) ...")

        enriched_docs = []
        for doc in raw_docs:
            emp_id = doc.get("employee_id", "")
            ai     = employee_map.get(emp_id, {})

            doc_to_insert = {
                "run_id":      run_id,
                "employee_id": emp_id,
                **{k: v for k, v in doc.items() if k != "employee_id"},
                "risk_level": ai.get("risk_level", doc.get("combined_risk_bucket", "Low")),
                "claude_analysis": {
                    "risk_summary":    ai.get("risk_summary", ""),
                    "root_causes":     ai.get("root_causes", []),
                    "recommendations": ai.get("recommendations", []),
                    "urgency":         ai.get("urgency", "Monitor"),
                    "priority_score":  ai.get("priority_score", 0),
                },
                "analysis_date": analysis_date,
            }

            risk_coll.insert_one(doc_to_insert)
            print(f"  [INSERT] {emp_id}")
            enriched_docs.append(doc_to_insert)

        if enriched_docs:
            sample_id = enriched_docs[0].get("employee_id")
            verify = risk_coll.find_one({"run_id": run_id, "employee_id": sample_id}, {"_id": 0})
            if verify:
                print(f"[Verify] {sample_id} — stored fields: {list(verify.keys())}")
            else:
                print(f"[Verify] WARNING: {sample_id} not found after write!")

        self.scorer.db.close()
        print(f"[RetentionAgent] Inserted {len(enriched_docs)} docs → MongoDB: Risk")
        return enriched_docs

    def run(self, csv_path: str) -> list:
        """
        Execute the full retention risk pipeline.

          1. Score   — RiskScorer (Cox model + salary gap analysis)
          2. Analyse — Claude API (structured HR insights)
          3. Save    — MongoDB Risk collection (enriched JSON)

        Returns:
            list[dict]: Enriched documents written to MongoDB.
        """
        clean_docs, raw_docs = self._run_scoring_tool(csv_path)
        analysis             = self._analyze_with_claude(clean_docs)
        enriched_docs        = self._merge_and_save(raw_docs, analysis)

        levels = {}
        for d in enriched_docs:
            lvl = d.get("risk_level", "Low")
            levels[lvl] = levels.get(lvl, 0) + 1

        print(f"\n[RetentionAgent] Complete — {len(enriched_docs)} records → MongoDB: Risk")
        print(f"  High: {levels.get('High', 0)}  Mid: {levels.get('Mid', 0)}  Low: {levels.get('Low', 0)}")
        return enriched_docs


# ---------------------------------------------------------------------------
# Module-level entry point
# ---------------------------------------------------------------------------

def run_retention_agent():
    """Resolve project-root paths and run the retention risk pipeline."""
    base_dir          = str(_PROJECT_ROOT)
    model_path        = os.path.join(base_dir, "models", "cox_retention_v1.pkl")
    feature_json_path = os.path.join(base_dir, "models", "cox_retention_v1_features.json")
    csv_path          = os.path.join(base_dir, "data",   "ibm_enhanced_test.csv")

    agent = RetentionAgent(model_path, feature_json_path)
    agent.run(csv_path)


if __name__ == "__main__":
    run_retention_agent()