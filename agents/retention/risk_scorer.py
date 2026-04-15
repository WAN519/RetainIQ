"""
agents/retention/retention_agent_v1.py

Agent 2 — Salary Risk Scorer (Cox Proportional Hazard + Market Pay Analysis).

This agent combines Cox survival model predictions with market salary gap analysis
to produce a comprehensive salary risk assessment per employee.

Data sources:
  - ibm_enhanced_test.csv  : employee snapshot (primary HR data)
  - MongoDB › Salary        : market salary benchmarks per role
  - MongoDB › Equity_Predictions : pay gap signals from the Equity Agent

Pipeline:
  1. Load employee snapshot from ibm_enhanced_test.csv.
  2. Pull market salary benchmarks from MongoDB collection: Salary.
  3. Pull equity gap data from MongoDB collection: Equity_Predictions.
  4. Engineer Cox model features (current_salary / 12 → MonthlyIncome, etc.).
  5. Run Cox model inference → partial hazard score per employee.
  6. Rank scores into percentiles and bucket into Low / Mid / High risk tiers.
  7. Compute salary gap vs. market median (absolute and percentage).
  8. Derive salary risk tier from gap severity.
  9. Combine Cox risk + salary gap risk into a final combined risk bucket.
  10. Upsert results as JSON to MongoDB collection: Risk (created if absent).

MongoDB output schema (Risk):
    {
        employee_id, analysis_date, role_name,
        current_salary, market_median_2026,
        salary_gap_abs, salary_gap_pct, salary_risk_tier,
        internal_salary_rank, performance_consistency,
        pay_gap_from_equity,
        cox_risk_score, cox_risk_pct, cox_risk_bucket,
        combined_risk_bucket, risk_factors,
        model_version, analysis_type
    }

Salary risk tier thresholds:
    gap_pct ≥ 15% below market  → High
    gap_pct ≥  5% below market  → Mid
    gap_pct  < 5% below market  → Low
    no market data available     → Unknown
"""

import os
import sys
import json
import pickle
import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from tools.mongoDB import MarketDB

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from database_mysql import DatabaseManager

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Salary gap thresholds (employee paid below market by this percentage)
# ---------------------------------------------------------------------------
SALARY_HIGH_RISK_THRESHOLD = 15.0   # ≥ 15% below market → High salary risk
SALARY_MID_RISK_THRESHOLD  =  5.0   # ≥  5% below market → Mid salary risk


def ensure_columns(df: pd.DataFrame, needed_cols: list, fill_value=0) -> pd.DataFrame:
    """Add any missing columns to *df*, filled with *fill_value*."""
    for col in needed_cols:
        if col not in df.columns:
            df[col] = fill_value
    return df


class RiskScorer:
    """
    Salary risk scorer combining Cox survival analysis with market pay gap analysis.

    Cox model background:
      - Trained on IBM HR data (Attrition.csv).
      - duration = YearsAtCompany  |  event = Attrition == 'Yes'
      - Partial hazard output is a relative risk score; higher = higher attrition risk.

    Salary risk component:
      - Compares current_salary against the market median sourced from either
        the CSV's Market_Median_2026 column or the MongoDB Salary collection.
      - Gaps are expressed as (market − current) / market × 100.
        Positive values indicate the employee is underpaid relative to market.

    Attributes:
        cph            : Loaded CoxPH model (lifelines, deserialized from pickle).
        feature_columns: Ordered feature list loaded from JSON metadata.
        db             : Connected MongoDB handler (MarketDB).
    """

    def __init__(self, model_path: str, feature_json_path: str):
        """
        Load the Cox model and feature metadata, then connect to MongoDB.

        Args:
            model_path       : Path to the serialized CoxPH model (.pkl).
            feature_json_path: Path to JSON with {"feature_columns": [...]}.

        Raises:
            FileNotFoundError: If either file does not exist.
            RuntimeError     : If the MongoDB connection fails.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(feature_json_path):
            raise FileNotFoundError(f"Feature JSON not found: {feature_json_path}")

        with open(model_path, "rb") as f:
            self.cph = pickle.load(f)

        with open(feature_json_path, "r") as f:
            meta = json.load(f)
            self.feature_columns = meta["feature_columns"]

        self.db = MarketDB()
        if not self.db.is_connected:
            raise RuntimeError("MongoDB not connected. Check config.env.")

    # ------------------------------------------------------------------
    # MongoDB loaders
    # ------------------------------------------------------------------

    def _load_salary_benchmarks(self) -> dict:
        """
        Load market salary benchmarks from the MongoDB Salary collection.

        Supports several field naming conventions used by the Equity Agent's
        BLS benchmark documents.

        Returns:
            dict: {role_name: annual_market_median} for all stored roles.
        """
        benchmarks = {}
        try:
            docs = list(self.db.db["Salary"].find({}, {"_id": 0}))
            for doc in docs:
                role = doc.get("role_name")
                median = (
                    doc.get("market_2026_target")
                    or doc.get("Market_Median_2026")
                    or doc.get("median_salary")
                    or doc.get("predicted_salary")
                )
                if role and median:
                    benchmarks[role] = float(median)
        except Exception as e:
            print(f"Warning: Could not load Salary benchmarks from MongoDB: {e}")
        return benchmarks

    def _load_equity_predictions(self) -> pd.DataFrame:
        """
        Load pay gap signals from the MongoDB Equity_Predictions collection.

        Returns:
            pd.DataFrame: Columns [employee_id, pay_gap_from_equity].
                          Empty DataFrame if the collection has no documents.
        """
        try:
            docs = list(self.db.db["Equity_Predictions"].find({}, {"_id": 0}))
        except Exception:
            return pd.DataFrame()

        if not docs:
            print("Info: Equity_Predictions collection is empty — skipping equity merge.")
            return pd.DataFrame()

        eq_df = pd.DataFrame(docs)

        if "employee_id" not in eq_df.columns and "EmployeeNumber" in eq_df.columns:
            eq_df["employee_id"] = eq_df["EmployeeNumber"]

        # Unwrap nested equity_gaps dict if stored by the Equity Agent
        if "equity_gaps" in eq_df.columns:
            def _extract_ext_gap(x):
                try:
                    return float(x.get("external_gap_pct", np.nan))
                except Exception:
                    return np.nan
            eq_df["pay_gap_from_equity"] = eq_df["equity_gaps"].apply(_extract_ext_gap)
        elif "pay_gap" in eq_df.columns:
            eq_df["pay_gap_from_equity"] = pd.to_numeric(eq_df["pay_gap"], errors="coerce")
        else:
            eq_df["pay_gap_from_equity"] = np.nan

        keep = [c for c in ["employee_id", "pay_gap_from_equity"] if c in eq_df.columns]
        return eq_df[keep].copy()

    # ------------------------------------------------------------------
    # MySQL loader
    # ------------------------------------------------------------------

    def _load_from_mysql(self) -> pd.DataFrame:
        """
        Load all employee records from MySQL employee_risk_profiles table.
        Missing Cox features (YearsSinceLastPromotion, OverTime_flag,
        WorkLifeBalance, JobSatisfaction, EnvironmentSatisfaction) are
        filled with 0 by _prepare_data — same as the CSV path.
        """
        mysql_db = DatabaseManager()
        rows = mysql_db.fetch_all("SELECT * FROM employee_risk_profiles")
        if not rows:
            raise RuntimeError("No rows returned from employee_risk_profiles")

        col_rows = mysql_db.fetch_all("DESCRIBE employee_risk_profiles")
        col_names = [r[0] for r in col_rows]

        df = pd.DataFrame(rows, columns=col_names)
        print(f"[RiskScorer] Loaded {len(df)} employees from MySQL")
        return df

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _prepare_data_core(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer Cox model features from a raw DataFrame.

        Column mapping:
          current_salary / 12   → MonthlyIncome
          YearsAtCompany        → duration  (0 replaced by 0.5)

        Missing Cox columns filled with 0:
          YearsSinceLastPromotion, OverTime_flag, WorkLifeBalance,
          JobSatisfaction, EnvironmentSatisfaction
        """
        # Normalise employee_id
        if "employee_id" not in df.columns:
            if "EmployeeNumber" in df.columns:
                df["employee_id"] = df["EmployeeNumber"].astype(str)
            else:
                df["employee_id"] = [f"EMP{i:04d}" for i in range(len(df))]
        else:
            df["employee_id"] = df["employee_id"].astype(str)

        # annual current_salary → monthly MonthlyIncome for Cox features
        if "current_salary" in df.columns:
            df["current_salary"] = pd.to_numeric(df["current_salary"], errors="coerce").fillna(0)
            if "MonthlyIncome" not in df.columns:
                df["MonthlyIncome"] = df["current_salary"] / 12.0
        elif "MonthlyIncome" in df.columns:
            df["MonthlyIncome"] = pd.to_numeric(df["MonthlyIncome"], errors="coerce").fillna(0)
            df["current_salary"] = df["MonthlyIncome"] * 12.0
        else:
            df["MonthlyIncome"] = 0.0
            df["current_salary"] = 0.0

        # Survival duration (min 0.5 years to avoid degenerate t=0 behaviour)
        if "YearsAtCompany" in df.columns:
            df["duration"] = (
                pd.to_numeric(df["YearsAtCompany"], errors="coerce")
                .fillna(1)
                .replace(0, 0.5)
            )
        else:
            df["duration"] = 1.0

        # Columns absent from ibm_enhanced_test.csv — fill with conservative defaults
        for col in ["YearsSinceLastPromotion", "OverTime_flag", "WorkLifeBalance",
                    "JobSatisfaction", "EnvironmentSatisfaction"]:
            if col not in df.columns:
                df[col] = 0

        # Ensure all Cox feature columns are numeric
        for col in self.feature_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Market / equity enrichment columns from CSV
        if "Market_Median_2026" in df.columns:
            df["market_median_csv"] = pd.to_numeric(df["Market_Median_2026"], errors="coerce").fillna(0)
        else:
            df["market_median_csv"] = 0.0

        for col_csv, col_out in [("Internal_Salary_Rank", "internal_salary_rank"),
                                 ("Performance_Consistency", "performance_consistency")]:
            if col_csv in df.columns:
                df[col_out] = pd.to_numeric(df[col_csv], errors="coerce")
            else:
                df[col_out] = np.nan

        return df

    def _prepare_data_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Same feature engineering as _prepare_data but takes a DataFrame directly."""
        return self._prepare_data_core(df)

    def _prepare_data(self, csv_path: str) -> pd.DataFrame:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        return self._prepare_data_core(pd.read_csv(csv_path))

    def _build_model_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Construct the aligned feature matrix for Cox model inference.

        Handles optional one-hot encoding of Department / JobRole when those
        dummy columns appear in the saved feature list.

        Returns:
            pd.DataFrame: Numeric matrix with exactly self.feature_columns columns.
        """
        X = df.copy()

        need_department = any(c.startswith("Department_") for c in self.feature_columns)
        need_jobrole    = any(c.startswith("JobRole_")    for c in self.feature_columns)

        dummy_cols = []
        if need_department and "Department" in X.columns:
            dummy_cols.append("Department")
        if need_jobrole and "JobRole" in X.columns:
            dummy_cols.append("JobRole")
        if dummy_cols:
            X = pd.get_dummies(X, columns=dummy_cols, drop_first=True)

        X = ensure_columns(X, self.feature_columns, fill_value=0)
        X_final = X[self.feature_columns].copy()
        for col in X_final.columns:
            X_final[col] = pd.to_numeric(X_final[col], errors="coerce")
        return X_final.fillna(0)

    # ------------------------------------------------------------------
    # Risk classification helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _salary_risk_tier(gap_pct) -> str:
        """
        Classify salary risk from market gap percentage.

        Args:
            gap_pct: (market − current) / market × 100.
                     Positive = underpaid (current < market).
                     None / NaN = no market data available.

        Returns:
            "High" | "Mid" | "Low" | "Unknown"
        """
        try:
            val = float(gap_pct)
        except (TypeError, ValueError):
            return "Unknown"
        if np.isnan(val):
            return "Unknown"
        if val >= SALARY_HIGH_RISK_THRESHOLD:
            return "High"
        if val >= SALARY_MID_RISK_THRESHOLD:
            return "Mid"
        return "Low"

    @staticmethod
    def _combined_risk(cox_bucket: str, salary_tier: str) -> str:
        """Escalate to the higher of the two independent risk signals."""
        rank  = {"Low": 0, "Mid": 1, "High": 2, "Unknown": -1}
        tiers = ["Low", "Mid", "High"]
        a = rank.get(cox_bucket, 0)
        b = rank.get(salary_tier, 0)
        return tiers[max(a, b)] if max(a, b) >= 0 else "Low"

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(val):
        """Convert a value to float; return None for NaN / non-numeric."""
        try:
            return float(val) if pd.notna(val) else None
        except Exception:
            return None

    def save_results(self, docs: list) -> None:
        """
        Insert scored documents as new records into MongoDB: Risk.

        Every call appends fresh records — existing documents are never
        modified. Use employee_id + scoring_date to identify a specific run.

        Args:
            docs: List of dicts produced by run() or enriched by the AI agent.
        """
        risk_coll = self.db.db[os.getenv("COLLECTION_NAME_RISK", "Risk")]
        db_name   = self.db.db.name
        print(f"[MongoDB] Inserting {len(docs)} new records into {db_name}.Risk ...")

        inserted = 0
        last_result = None
        for doc in docs:
            emp_id = doc.get("employee_id")
            last_result = risk_coll.insert_one(doc)
            inserted += 1
            print(f"  [INSERT] {emp_id}")

        # Read back the last inserted document to verify the write reached Atlas
        if last_result:
            verify = risk_coll.find_one({"_id": last_result.inserted_id}, {"_id": 0})
            if verify:
                print(f"[Verify] Read-back OK — fields: {list(verify.keys())}")
            else:
                print(f"[Verify] WARNING: read-back returned nothing!")

        self.db.close()
        print(f"[MongoDB] Done — {inserted} new records inserted.")

    def run(self, csv_path: str | None = None) -> list:
        """
        Execute the salary risk scoring pipeline and return scored documents.

        Does NOT write to MongoDB — call save_results(docs) afterwards.

        Data source priority:
          1. MySQL employee_risk_profiles  (preferred — full 1000-employee dataset)
          2. csv_path fallback             (used only if MySQL is unavailable)

        Steps:
          1. Load employee data from MySQL (or CSV fallback).
          2. Merge equity pay gap signals from MongoDB.
          3. Resolve market median per employee (table column → MongoDB fallback).
          4. Compute salary gap (abs + %) and salary risk tier.
          5. Run Cox model → partial hazard scores → percentile buckets.
          6. Derive combined risk bucket.
          7. Build human-readable risk factor list.
          8. Return list of scored dicts (one per employee).

        Returns:
            list[dict]: JSON-serialisable scored documents ready for MongoDB upsert.
        """
        try:
            df = self._load_from_mysql()
            df = self._prepare_data_from_df(df)
        except Exception as e:
            print(f"[RiskScorer] MySQL load failed ({e}), falling back to CSV: {csv_path}")
            if not csv_path:
                raise
            df = self._prepare_data(csv_path)

        # ── MongoDB data ────────────────────────────────────────────────
        benchmarks = self._load_salary_benchmarks()
        equity_df  = self._load_equity_predictions()

        # Merge equity pay gap signals
        df["employee_id"] = df["employee_id"].astype(str)
        if not equity_df.empty and "employee_id" in equity_df.columns:
            equity_df["employee_id"] = equity_df["employee_id"].astype(str)
            df = df.merge(equity_df, on="employee_id", how="left")
        if "pay_gap_from_equity" not in df.columns:
            df["pay_gap_from_equity"] = np.nan

        # ── Salary gap analysis ─────────────────────────────────────────
        def resolve_market(row):
            """CSV value first; MongoDB benchmark as fallback."""
            csv_val = row.get("market_median_csv", 0)
            if csv_val and float(csv_val) > 0:
                return float(csv_val)
            role = str(row.get("role_name", ""))
            if role and role in benchmarks:
                return benchmarks[role]
            return None

        df["market_median_2026"] = df.apply(resolve_market, axis=1)

        def calc_salary_gap(row):
            market  = row["market_median_2026"]
            current = row["current_salary"]
            if market and float(market) > 0 and current and float(current) > 0:
                gap_abs = float(market) - float(current)
                gap_pct = (gap_abs / float(market)) * 100.0
                return gap_abs, gap_pct
            return None, None

        gaps = df.apply(calc_salary_gap, axis=1, result_type="expand")
        df["salary_gap_abs"] = gaps[0]
        df["salary_gap_pct"] = gaps[1]
        df["salary_risk_tier"] = df["salary_gap_pct"].apply(self._salary_risk_tier)

        # ── Cox model inference ─────────────────────────────────────────
        X_final = self._build_model_input(df)
        df["cox_risk_score"] = self.cph.predict_partial_hazard(X_final).values.ravel()
        df["cox_risk_pct"]   = df["cox_risk_score"].rank(pct=True)
        df["cox_risk_bucket"] = pd.cut(
            df["cox_risk_pct"],
            bins=[0, 0.5, 0.9, 1.0],
            labels=["Low", "Mid", "High"],
            include_lowest=True
        ).astype(str)

        # ── Combined risk ───────────────────────────────────────────────
        df["combined_risk_bucket"] = df.apply(
            lambda r: self._combined_risk(r["cox_risk_bucket"], r["salary_risk_tier"]),
            axis=1
        )

        # ── Risk factor narration ───────────────────────────────────────
        def build_risk_factors(row):
            factors = []
            gap_pct = row.get("salary_gap_pct")
            if gap_pct is not None and not (isinstance(gap_pct, float) and np.isnan(gap_pct)):
                if gap_pct >= SALARY_HIGH_RISK_THRESHOLD:
                    factors.append(f"salary {gap_pct:.1f}% below market (high salary risk)")
                elif gap_pct >= SALARY_MID_RISK_THRESHOLD:
                    factors.append(f"salary {gap_pct:.1f}% below market")
                elif gap_pct < 0:
                    factors.append(f"salary {abs(gap_pct):.1f}% above market")

            eq_gap = row.get("pay_gap_from_equity")
            if pd.notna(eq_gap) and float(eq_gap) <= -10:
                factors.append(f"internal pay gap {float(eq_gap):.1f}% (equity agent signal)")

            cox_bucket = row.get("cox_risk_bucket", "")
            if cox_bucket == "High":
                factors.append("high Cox attrition risk score (top 10th percentile)")
            elif cox_bucket == "Mid":
                factors.append("elevated Cox attrition risk score (50th–90th percentile)")

            rank = row.get("internal_salary_rank")
            if pd.notna(rank) and float(rank) <= 1:
                factors.append("low internal salary rank")

            if not factors:
                factors.append("no significant risk factors detected")
            return factors

        df["risk_factors"] = df.apply(build_risk_factors, axis=1)

        # ── Build scored document list (no MongoDB write here) ──────────
        sf = self._safe_float
        scored_at = datetime.datetime.now(datetime.timezone.utc)
        docs = []
        for _, employee_row in df.iterrows():
            emp_id = str(employee_row["employee_id"])
            doc = {
                "employee_id":             emp_id,
                "scoring_date":            scored_at,
                "role_name":               str(employee_row.get("role_name", "")),
                "current_salary":          sf(employee_row.get("current_salary")),
                "market_median_2026":      sf(employee_row.get("market_median_2026")),
                "salary_gap_abs":          sf(employee_row.get("salary_gap_abs")),
                "salary_gap_pct":          sf(employee_row.get("salary_gap_pct")),
                "salary_risk_tier":        str(employee_row.get("salary_risk_tier", "Unknown")),
                "internal_salary_rank":    sf(employee_row.get("internal_salary_rank")),
                "performance_consistency": sf(employee_row.get("performance_consistency")),
                "pay_gap_from_equity":     sf(employee_row.get("pay_gap_from_equity")),
                "cox_risk_score":          sf(employee_row.get("cox_risk_score")),
                "cox_risk_pct":            sf(employee_row.get("cox_risk_pct")),
                "cox_risk_bucket":         str(employee_row.get("cox_risk_bucket", "Low")),
                "combined_risk_bucket":    str(employee_row.get("combined_risk_bucket", "Low")),
                "risk_factors":            employee_row.get("risk_factors", []),
                "model_version":           "cox_retention_v1",
                "analysis_type":           "salary_risk",
            }
            docs.append(doc)

        print(f"Scoring complete — {len(docs)} employees ready for analysis.")
        return docs


# ---------------------------------------------------------------------------
# Module-level entry point used by main.py
# ---------------------------------------------------------------------------

def run_risk_scorer():
    """
    Standalone entry point: score → save (no AI enrichment).

    For AI-enriched results use AISalaryRiskAgent in ai_salary_risk_agent.py.
    """
    base_dir          = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path        = os.path.join(base_dir, "models", "cox_retention_v1.pkl")
    feature_json_path = os.path.join(base_dir, "models", "cox_retention_v1_features.json")
    csv_path          = os.path.join(base_dir, "data",   "ibm_enhanced_test.csv")

    print(f"Model   : {model_path}")
    print(f"Features: {feature_json_path}")
    print(f"CSV     : {csv_path}")

    agent = RiskScorer(model_path, feature_json_path)
    docs = agent.run(csv_path)
    agent.save_results(docs)
    print(f"Salary Risk Agent complete → MongoDB: Risk ({len(docs)} records)")


if __name__ == "__main__":
    run_salary_risk_agent()