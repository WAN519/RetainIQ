"""
main.py — RetentionAgent Pipeline Entry Point

RetentionAgent is an AI-powered employee attrition prevention system built for MLH.
It combines three specialized agents with machine learning models and LLM-generated
insights to give HR teams and managers an early warning system for employee flight risk.

Three-agent pipeline:
  1. Equity Agent (agents/equity/equity_agent.py)
       Input : ibm_enhanced_test.csv
       Model : LightGBM salary regression (models/agent_salary_regressor.pkl)
       Output: MongoDB › Equity_Predictions
       What  : Scores each employee's salary against both the market benchmark
               (BLS CPI-adjusted) and an internal fair-valuation model.

  2. Salary Risk Agent (agents/retention/retention_agent_v1.py)
       Input : ibm_enhanced_test.csv + MongoDB › Salary + MongoDB › Equity_Predictions
       Model : Cox Proportional Hazard survival model (models/cox_retention_v1.pkl)
       Output: MongoDB › Risk
       What  : Scores each employee's salary risk by combining Cox attrition hazard
               scores with market pay gap analysis; writes JSON results to Risk collection.

  3. Emotion Agent (agents/emotion/emotion_agent.py + emotion_tool.py)
       Input : glassdoor_reviews.csv (not committed — ~280 MB)
       Models: Sentence-Transformers (all-MiniLM-L6-v2) + RoBERTa sentiment
       Output: MongoDB › reviews_analysis → Gemini LLM report
       What  : Labels Glassdoor reviews by attrition topic, aggregates sentiment
               statistics, and generates an AI HR advisory report via Gemini Pro.

Prerequisites:
  1. Copy config.env.example → config.env and fill in your credentials.
  2. Run `python scripts/MockData.py` to seed the MySQL demo database.
  3. Run `python -m agents.equity.run` to sync BLS market salary benchmarks.
  4. Then run this file: `python main.py`

Frontend:
  cd retention-ui && npm install && npm run dev
"""

import os
from dotenv import load_dotenv

load_dotenv("config.env")

MODEL_DIR = "models"
DATA_DIR  = "data"

EQUITY_MODEL       = os.path.join(MODEL_DIR, "agent_salary_regressor.pkl")
RETENTION_MODEL    = os.path.join(MODEL_DIR, "cox_retention_v1.pkl")
RETENTION_FEATURES = os.path.join(MODEL_DIR, "cox_retention_v1_features.json")
HR_CSV             = os.path.join(DATA_DIR,  "Attrition.csv")
HR_TEST_CSV        = os.path.join(DATA_DIR,  "ibm_enhanced_test.csv")


def run_equity_agent():
    """
    Run Agent 1: score salary equity for all employees in the test CSV.
    Results are written to MongoDB › Equity_Predictions.
    """
    from agents.equity.equity_agent import EquityAgent
    print("\n=== Step 1: Equity Agent ===")
    agent = EquityAgent(model_path=EQUITY_MODEL)
    agent.run_analysis_pipeline(HR_TEST_CSV)


def run_retention_agent(ai_mode: bool = True):
    """
    Run Agent 2: salary risk scoring using Cox survival model + market pay gap analysis.
    Reads from ibm_enhanced_test.csv + MongoDB › Salary + Equity_Predictions.
    Writes enriched JSON results to MongoDB › Risk.

    Args:
        ai_mode: If True (default), use AISalaryRiskAgent — scores via Cox model
                 then enriches results through the Claude API before saving.
                 If False, use SalaryRiskAgent directly (no Claude call).
    """
    if ai_mode:
        from agents.retention.ai_salary_risk_agent import AISalaryRiskAgent
        print("\n=== Step 2: AI Salary Risk Agent (Claude-enriched) ===")
        agent = AISalaryRiskAgent(RETENTION_MODEL, RETENTION_FEATURES)
        agent.run(HR_TEST_CSV)
    else:
        from agents.retention.retention_agent_v1 import SalaryRiskAgent, run_salary_risk_agent
        print("\n=== Step 2: Salary Risk Agent (scoring only) ===")
        run_salary_risk_agent()


def run_emotion_agent(company_name: str, data_path: str):
    """
    Run Agent 3: analyze Glassdoor reviews and generate an LLM HR report.

    Args:
        company_name (str): Company to analyze (must match 'firm' column in the CSV).
        data_path (str): Path to the Glassdoor reviews CSV file.
    """
    from tools.emotion_tool import GlassdoorEmotionAgent
    from agents.emotion.emotion_agent import EmotionOrchestrator
    print("\n=== Step 3: Emotion Agent ===")
    agent = GlassdoorEmotionAgent()
    orchestrator = EmotionOrchestrator(agent)
    report = orchestrator.check_and_analyze(company_name, data_path)
    print(report)


if __name__ == "__main__":
    run_equity_agent()
    run_retention_agent()
    # Uncomment to run the Glassdoor emotion analysis (requires the large CSV):
    # run_emotion_agent("Apple", "data/glassdoor_reviews.csv")