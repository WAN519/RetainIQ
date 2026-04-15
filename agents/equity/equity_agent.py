"""
agents/equity/equity_agent.py

Agent 1 — Salary Equity Analyzer.

For each employee in the input CSV this agent:
  1. Retrieves the 2026 market salary benchmark for the employee's role from MongoDB.
  2. Constructs the LightGBM feature vector (must exactly match training-time column order).
  3. Runs inference to produce an internal fair-valuation estimate (log-transformed monthly income).
  4. Computes two equity gap metrics:
       - external_gap_pct: how far the actual salary deviates from the market benchmark
       - internal_gap_pct: how far the actual salary deviates from the model's fair valuation
  5. Writes results to MongoDB collection `Equity_Predictions` for downstream agents.

MongoDB output schema (Equity_Predictions):
    {
        employee_id, role_name, analysis_date, actual_salary,
        benchmarks: { market_2026_target, internal_fair_valuation },
        equity_gaps: { external_gap_pct, internal_gap_pct },
        status: "READY_FOR_ORCHESTRATOR"
    }
"""

import os
import joblib
import pandas as pd
import numpy as np
import datetime
import urllib.parse
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from tools.mongoDB import MarketDB


class EquityAgent:
    """
    Salary fairness scoring agent powered by a pre-trained LightGBM regression model.

    The model was trained to predict an employee's fair monthly income (log-transformed)
    given their role, performance, and tenure features. Comparing this prediction against
    the employee's actual salary reveals internal pay inequities.

    Attributes:
        model: Loaded LightGBM model (joblib).
        db (MarketDB): Connected MongoDB handler for benchmark lookups and result storage.
        feature_columns (list[str]): Ordered feature list matching the training schema.
    """

    # Categorical columns — must be cast to 'category' dtype for LightGBM
    CAT_COLS = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus', 'Over18']

    # Continuous float columns injected at inference time
    FLOAT_COLS = ['Market_Median_2026', 'Internal_Salary_Rank', 'Performance_Consistency']

    def __init__(self, model_path: str, config_path: str = "config.env"):
        """
        Load the LightGBM model and initialize the database connection.

        Args:
            model_path (str): Path to the serialized LightGBM model (.pkl).
            config_path (str): Path to the environment file containing DB credentials.

        Raises:
            Exception: Re-raises any exception thrown by joblib if the model cannot be loaded.
        """
        try:
            self.model = joblib.load(model_path)
            print(f"LightGBM model loaded: {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

        self.db = MarketDB()

        # Feature order must be identical to the order used during model training
        self.feature_columns = [
            'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education',
            'EducationField', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole',
            'MaritalStatus', 'MonthlyRate', 'NumCompaniesWorked', 'Over18',
            'PercentSalaryHike', 'PerformanceRating', 'StockOptionLevel', 'TotalWorkingYears',
            'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole',
            'YearsWithCurrManager', 'Market_Median_2026', 'Internal_Salary_Rank',
            'Performance_Consistency'
        ]

    def run_analysis_pipeline(self, csv_input_path: str):
        """
        Run the full equity scoring pipeline for every employee in the input CSV.

        For each employee:
          - Fetches the market benchmark salary from MongoDB.
          - Builds the model input DataFrame with strict column ordering and dtypes.
          - Predicts the fair internal valuation using the LightGBM model.
          - Calculates external and internal pay gap percentages.
          - Upserts the result into the `Equity_Predictions` MongoDB collection.

        Args:
            csv_input_path (str): Path to the employee CSV file.
                Required columns: employee_id, role_name, current_salary,
                                  plus all LightGBM feature columns.
        """
        if not self.db.is_connected:
            print("Database not connected. Aborting pipeline.")
            return

        df = pd.read_csv(csv_input_path)
        print(f"Processing {len(df)} employees...")

        for index, row in df.iterrows():
            emp_id = row.get('employee_id', f"UNKNOWN_{index}")
            role = row.get('role_name')
            actual_salary = row.get('current_salary')

            # --- Step A: Retrieve external market benchmark (Perception layer) ---
            market_data = self.db.get_benchmark(role)
            if not market_data:
                print(f"Skipping {emp_id}: no market benchmark found for role '{role}'.")
                continue

            market_val = market_data['predicted_salary_2026']

            # --- Step B: Build the model feature vector ---
            input_dict = row.to_dict()

            # Inject dynamic features not present in the raw CSV
            input_dict['Internal_Salary_Rank'] = row.get('Internal_Salary_Rank', 0.5)
            input_dict['Performance_Consistency'] = row.get('Performance_Consistency', row['PerformanceRating'])

            # Convert annual market benchmark to monthly to match model training units
            input_dict['Market_Median_2026'] = market_val / 12
            input_dict['current_salary'] = actual_salary / 12

            # Select and order columns exactly as seen during training
            final_input = pd.DataFrame([input_dict])[self.feature_columns]

            # Cast to dtypes expected by LightGBM
            for col in self.CAT_COLS:
                final_input[col] = final_input[col].astype('category')

            for col in self.FLOAT_COLS:
                final_input[col] = final_input[col].astype('float64')

            int_cols = [c for c in self.feature_columns if c not in self.CAT_COLS + self.FLOAT_COLS]
            for col in int_cols:
                final_input[col] = pd.to_numeric(final_input[col], errors='coerce').fillna(0).astype('int64')

            # --- Step C: Run LightGBM inference ---
            # Model output is log1p(MonthlyIncome); reverse with expm1 to get monthly salary
            pred_log = self.model.predict(final_input)[0]
            internal_monthly_valuation = np.expm1(pred_log)

            # Annualize the monthly prediction for fair comparison against actual_salary
            internal_valuation_annual = internal_monthly_valuation * 12
            print(f"  {emp_id} | Actual: {actual_salary:,.0f} | Fair Value: {internal_valuation_annual:,.0f}")

            # --- Step D: Compute equity gap metrics ---
            # Negative values indicate the employee is underpaid relative to the benchmark
            external_gap = (actual_salary - market_val) / market_val
            internal_gap = (actual_salary - internal_valuation_annual) / internal_valuation_annual

            # --- Step E: Persist results to MongoDB for the Orchestrator ---
            analysis_result = {
                "employee_id": emp_id,
                "role_name": role,
                "analysis_date": datetime.datetime.utcnow(),
                "actual_salary": round(actual_salary, 2),
                "benchmarks": {
                    "market_2026_target": round(market_val, 2),
                    "internal_fair_valuation": round(internal_valuation_annual, 2)
                },
                "equity_gaps": {
                    "external_gap_pct": round(external_gap * 100, 2),
                    "internal_gap_pct": round(internal_gap * 100, 2)
                },
                "status": "READY_FOR_ORCHESTRATOR"
            }

            try:
                self.db.db["Equity_Predictions"].update_one(
                    {"employee_id": emp_id},
                    {"$set": analysis_result},
                    upsert=True
                )
                print(f"  Saved — IntGap: {analysis_result['equity_gaps']['internal_gap_pct']}% | "
                      f"ExtGap: {analysis_result['equity_gaps']['external_gap_pct']}%")
            except PyMongoError as e:
                print(f"  DB write error for {emp_id}: {e}")

        print("\n--- Equity Agent: pipeline complete ---")
        self.db.close()


if __name__ == "__main__":
    MODEL_FILE = "models/agent_salary_regressor.pkl"
    TEST_DATA = "data/ibm_enhanced_test.csv"

    if os.path.exists(MODEL_FILE) and os.path.exists(TEST_DATA):
        agent = EquityAgent(model_path=MODEL_FILE)
        agent.run_analysis_pipeline(TEST_DATA)
    else:
        print("Model or test CSV not found. Check paths and try again.")