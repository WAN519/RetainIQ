"""
run_all_agents.py

End-to-end integration test for the full RetentionAgent pipeline.

Two independent tracks that converge at Stage 4:

  Track A — Salary
    Stage 1  Equity Agent       →  MongoDB: Equity_Predictions
                 LightGBM fair-value + external/internal pay gap
    Stage 2  Retention Agent    →  MongoDB: Risk
                 Cox model + salary gap  (reads Equity_Predictions)

  Track B — Sentiment  (independent of Track A)
    Stage 3  Emotion Agent      →  MongoDB: reviews_analysis, Emotion
                 NLP pipeline + Claude sentiment report

  Converge
    Stage 4  Recommendation Pipeline  →  MongoDB: retention_recommendations
                 Generator → Adversarial Audit (max 3 rounds) → Save
                 reads: Risk + Emotion + employee_comment

Prerequisites
─────────────
  • config.env  —  CLAUDE_API_KEY, MONGODB_URI, MONGODB_NAME
  • data/ibm_enhanced_test.csv
  • models/agent_salary_regressor.pkl
  • models/cox_retention_v1.pkl  +  models/cox_retention_v1_features.json
  • data/mock_reviews.csv  (or --reviews-csv)
  • MongoDB employee_comment collection  (run test.py first if empty)

Usage
─────
  python run_all_agents.py                               # all stages
  python run_all_agents.py --stage 1                    # equity only
  python run_all_agents.py --stage 2                    # retention only
  python run_all_agents.py --stage 3 --company Apple --month 2026-04
  python run_all_agents.py --stage 4 --month 2026-04
"""

import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

# ── Default configuration ─────────────────────────────────────────────────────
COMPANY     = "Apple"
MONTH       = datetime.now(timezone.utc).strftime("%Y-%m")
REVIEWS_CSV = "data/mock_reviews.csv"
HR_CSV      = "data/ibm_enhanced_test.csv"
MODEL_DIR   = Path(__file__).parent / "models"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")


def _stage_header(n: int, title: str) -> None:
    print(f"\n{'─'*62}")
    print(f"  Stage {n} — {title}")
    print(f"{'─'*62}\n")


# ── Stage 1: Equity Agent ─────────────────────────────────────────────────────

def run_stage_1(hr_csv: str) -> None:
    """
    LightGBM salary fair-value model + external/internal pay gap.
    Writes per-employee equity scores to MongoDB: Equity_Predictions.
    Must run before Stage 2 — Retention Agent reads Equity_Predictions.
    """
    _stage_header(1, "Equity Agent  (LightGBM fair-value + pay gap)")

    equity_model = str(MODEL_DIR / "agent_salary_regressor.pkl")

    if not Path(equity_model).exists():
        print(f"[Stage 1] SKIP — equity model not found: {equity_model}")
        return
    if not Path(hr_csv).exists():
        print(f"[Stage 1] SKIP — HR CSV not found: {hr_csv}")
        return

    from agents.equity.equity_agent import EquityAgent

    agent = EquityAgent(model_path=equity_model)
    agent.run_analysis_pipeline(hr_csv)
    print(f"\n[Stage 1] Complete — equity gaps → MongoDB: Equity_Predictions")


# ── Stage 2: Retention Agent ──────────────────────────────────────────────────

def run_stage_2(hr_csv: str) -> None:
    """
    Cox survival model + salary gap analysis + Claude HR insights.
    Reads Equity_Predictions for pay gap signals.
    Writes enriched risk documents to MongoDB: Risk.
    """
    _stage_header(2, "Retention Agent  (Cox + Salary Gap + Claude Analysis)")

    model_pkl    = str(MODEL_DIR / "cox_retention_v1.pkl")
    feature_json = str(MODEL_DIR / "cox_retention_v1_features.json")

    if not Path(model_pkl).exists():
        print(f"[Stage 2] SKIP — Cox model not found: {model_pkl}")
        return
    if not Path(feature_json).exists():
        print(f"[Stage 2] SKIP — features JSON not found: {feature_json}")
        return
    if not Path(hr_csv).exists():
        print(f"[Stage 2] SKIP — HR CSV not found: {hr_csv}")
        return

    from agents.retention.retention_agent import RetentionAgent

    agent = RetentionAgent(model_path=model_pkl, feature_json_path=feature_json)
    docs  = agent.run(hr_csv)
    print(f"\n[Stage 2] Complete — {len(docs)} records → MongoDB: Risk")


# ── Stage 3: Emotion Agent ────────────────────────────────────────────────────

def run_stage_3(company: str, month: str, reviews_csv: str) -> None:
    """
    SentenceTransformer topic labeling + RoBERTa sentiment + Claude report.
    Independent of Stages 1-2.
    Writes per-review labels to MongoDB: reviews_analysis.
    Writes company sentiment report to MongoDB: Emotion.
    """
    _stage_header(3, f"Emotion Agent  (NLP + Claude Sentiment Report)  [{company} / {month}]")

    if not Path(reviews_csv).exists():
        print(f"[Stage 3] SKIP — reviews CSV not found: {reviews_csv}")
        return

    from agents.emotion.emotion_agent import run_emotion_agent

    run_emotion_agent(company_name=company, csv_path=reviews_csv, month=month)
    print(f"\n[Stage 3] Complete — sentiment report → MongoDB: Emotion")


# ── Stage 4: Recommendation Pipeline ─────────────────────────────────────────

def run_stage_4(month: str) -> None:
    """
    Recommendation Generator + Adversarial Audit loop (max 3 attempts).
    Reads from MongoDB: Risk + Emotion + employee_comment.
    Writes approved recommendations to MongoDB: retention_recommendations.
    """
    _stage_header(4, f"Recommendation Pipeline  (Generator → Audit × 3 → Save)  [{month}]")

    from agents.pipeline.pipeline import run_pipeline

    run_pipeline(month=month)
    print(f"\n[Stage 4] Complete — recommendations → MongoDB: retention_recommendations")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all RetentionAgent stages end-to-end",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--company",     default=COMPANY,     help="Company name (must match 'firm' column in reviews CSV)")
    parser.add_argument("--month",       default=MONTH,       help="Analysis month in YYYY-MM format")
    parser.add_argument("--reviews-csv", default=REVIEWS_CSV, help="Path to Glassdoor reviews CSV")
    parser.add_argument("--hr-csv",      default=HR_CSV,      help="Path to employee HR data CSV")
    parser.add_argument("--stage",       type=int, choices=[1, 2, 3, 4], default=None,
                        help="Run only this stage (omit to run all)")
    args = parser.parse_args()

    _banner(f"RetentionAgent Pipeline  |  {args.company}  |  {args.month}")
    print(f"  HR CSV      : {args.hr_csv}")
    print(f"  Reviews CSV : {args.reviews_csv}")
    print(f"  Model dir   : {MODEL_DIR}")

    run_all = args.stage is None
    t0 = time.time()
    timings: dict[int, float] = {}

    if run_all:
        # Run the full LangGraph pipeline (all 4 stages in one compiled graph)
        from agents.pipeline.pipeline import run_pipeline
        t = time.time()
        run_pipeline(
            month=args.month,
            company=args.company,
            hr_csv=args.hr_csv,
            reviews_csv=args.reviews_csv,
        )
        timings[0] = time.time() - t
    else:
        if args.stage == 1:
            t = time.time()
            run_stage_1(hr_csv=args.hr_csv)
            timings[1] = time.time() - t

        if args.stage == 2:
            t = time.time()
            run_stage_2(hr_csv=args.hr_csv)
            timings[2] = time.time() - t

        if args.stage == 3:
            t = time.time()
            run_stage_3(company=args.company, month=args.month, reviews_csv=args.reviews_csv)
            timings[3] = time.time() - t

        if args.stage == 4:
            t = time.time()
            run_stage_4(month=args.month)
            timings[4] = time.time() - t

    _banner("Pipeline Complete")
    for n, elapsed in timings.items():
        label = "Full (LangGraph)" if n == 0 else f"Stage {n}"
        print(f"  {label} : {elapsed:.1f}s")
    print(f"  Total   : {time.time() - t0:.1f}s")
    print()


if __name__ == "__main__":
    main()