
"""
tools/emotion_tool.py

Glassdoor Review Sentiment & Topic Analyzer.

Processes raw Glassdoor employee review text through a two-stage NLP pipeline:

  Stage A — Semantic topic labeling (Sentence-Transformers / all-MiniLM-L6-v2):
      Each review's "cons" text is embedded and compared against four topic
      definitions using cosine similarity. Reviews exceeding a 0.35 threshold
      are labeled 1 for that topic:
          sem_management  — management / leadership issues
          sem_salary      — compensation dissatisfaction
          sem_workload    — burnout / work-life balance
          sem_career      — lack of growth or promotion

  Stage B — Sentiment polarity scoring (RoBERTa):
      cardiffnlp/twitter-roberta-base-sentiment-latest classifies each review
      as Positive / Neutral / Negative and outputs a confidence score.

Results are stored in MongoDB collection `reviews_analysis` for the
EmotionOrchestrator to aggregate and forward to the LLM report generator.

Hardware: automatically uses Apple MPS acceleration when available, falls back to CPU.
"""
from __future__ import annotations

import torch
import pandas as pd
from tqdm import tqdm
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline


class GlassdoorEmotionAgent:
    """
    Batch NLP processor for Glassdoor employee reviews.

    Attributes:
        device (str): Inference device — 'mps' on Apple Silicon, 'cpu' otherwise.
        sem_model (SentenceTransformer): Embedding model for semantic topic labeling.
        target_definitions (dict[str, str]): Maps column names to natural-language
            topic descriptions used as embedding anchors.
        emotion_pipe: HuggingFace sentiment-analysis pipeline (RoBERTa).
        client (MongoClient): MongoDB client.
        db: MongoDB database handle.
        collection: MongoDB collection for storing analysis results.
    """

    # Cosine similarity threshold for assigning a semantic topic label
    SEMANTIC_THRESHOLD = 0.35

    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/", db_name: str = "hr_analysis"):
        """
        Initialize models and database connection.

        Args:
            mongo_uri (str): MongoDB connection URI.
            db_name (str): Target database name.
        """
        # Prefer Apple MPS for faster inference on Mac; fall back to CPU
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Emotion Agent initializing on device: {self.device}")

        # Semantic embedding model — lightweight but effective for topic similarity
        self.sem_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

        # Natural-language definitions for each attrition topic.
        # These are embedded once at pipeline start and reused for every batch.
        self.target_definitions = {
            'sem_management': "Issues with management, toxic leadership, or poor company culture.",
            'sem_salary':     "Dissatisfaction with pay, low salary, or lack of benefits.",
            'sem_workload':   "Long working hours, burnout, and bad work-life balance.",
            'sem_career':     "Lack of career growth, no promotion opportunities, and poor training.",
        }

        # RoBERTa fine-tuned on Twitter data; generalizes well to short review text
        self.emotion_pipe = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=self.device
        )

        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db["reviews_analysis"]

    def _clean_and_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare raw Glassdoor data for NLP processing.

        - Fills missing "cons" text with empty string.
        - Truncates "cons" to 1000 characters to prevent RoBERTa token overflow.
        - Converts the 'recommend' flag to a numeric score.
        - Creates a binary 'is_former' flag for former vs. current employees.

        Args:
            df (pd.DataFrame): Raw Glassdoor review DataFrame.
                Expected columns: cons, recommend, current.

        Returns:
            pd.DataFrame: Cleaned copy of the input DataFrame.
        """
        df = df.copy()

        # Truncate to avoid exceeding RoBERTa's 512-token limit
        df['cons'] = df['cons'].fillna("").str.slice(0, 1000)

        # Map Glassdoor recommendation codes: 'v'=yes(1), 'x'=no(-1), 'o'/'r'=neutral(0)
        mapping = {'v': 1, 'x': -1, 'o': 0, 'r': 0}
        df['recommend_num'] = df['recommend'].map(mapping).fillna(0)

        df['is_former'] = df['current'].str.contains('Former', na=False).astype(int)
        return df

    def run_pipeline(self, df: pd.DataFrame, batch_size: int = 100):
        """
        Execute the full NLP pipeline and persist results to MongoDB.

        Processing order per batch:
          1. Embed all texts with SentenceTransformer.
          2. Compute cosine similarity against topic definition embeddings.
          3. Assign binary topic labels using the 0.35 cosine threshold.
          4. Run RoBERTa sentiment classification.
          5. Insert the enriched batch into MongoDB.

        Args:
            df (pd.DataFrame): Preprocessed (or raw) review DataFrame.
                Must contain a 'cons' column.
            batch_size (int): Number of reviews to process per iteration.
                Smaller batches use less memory; larger batches run faster.
        """
        df = self._clean_and_preprocess(df)
        total_rows = len(df)

        # Pre-compute topic definition embeddings once — reused for every batch
        def_embeddings = self.sem_model.encode(
            list(self.target_definitions.values()),
            convert_to_tensor=True
        )

        print(f"Starting pipeline: {total_rows} reviews, batch size {batch_size}")

        for i in tqdm(range(0, total_rows, batch_size)):
            batch_df = df.iloc[i: i + batch_size].copy()
            texts = batch_df['cons'].tolist()

            # Stage A: Semantic topic labeling
            text_embeddings = self.sem_model.encode(
                texts, convert_to_tensor=True, show_progress_bar=False
            )
            # Shape: (batch_size, num_topics)
            cosine_scores = util.cos_sim(text_embeddings, def_embeddings)

            for idx, col in enumerate(self.target_definitions.keys()):
                # Binary label: 1 if cosine similarity exceeds threshold, else 0
                batch_df[col] = (cosine_scores[:, idx] > self.SEMANTIC_THRESHOLD).int().tolist()

            # Stage B: RoBERTa sentiment polarity
            roberta_results = self.emotion_pipe(texts)
            batch_df['roberta_label'] = [res['label'] for res in roberta_results]
            batch_df['roberta_score'] = [res['score'] for res in roberta_results]

            # Persist enriched batch to MongoDB
            self.collection.insert_many(batch_df.to_dict('records'))

        print(f"Pipeline complete. Results saved to {self.db.name}.{self.collection.name}")


def run_emotion_analysis(company_name: str, csv_path: str, month: str | None = None) -> dict:
    """
    Top-level tool entry point: run the full NLP pipeline for one company and
    persist the enriched reviews to MongoDB.

    This function is designed to be called as a Claude tool. It loads the CSV,
    filters by company, runs GlassdoorEmotionAgent, and returns a summary dict
    that Claude can use to decide whether to proceed with downstream queries.

    Args:
        company_name (str): Company name matching the 'firm' column in the CSV.
        csv_path (str): Absolute or relative path to the Glassdoor reviews CSV.
        month (str | None): Analysis month in YYYY-MM format. Defaults to current month.

    Returns:
        dict: Status summary with review count, or an error key on failure.
    """
    import os
    from datetime import datetime
    from dotenv import load_dotenv

    load_dotenv("config.env")

    if month is None:
        month = datetime.now().strftime("%Y-%m")

    try:
        raw_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return {"error": f"CSV not found: {csv_path}"}
    except Exception as e:
        return {"error": f"Failed to read CSV: {e}"}

    company_df = raw_df[raw_df["firm"] == company_name].copy()
    if company_df.empty:
        return {"error": f"No reviews found for '{company_name}' in {csv_path}"}

    company_df["analysis_month"] = month

    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    db_name   = os.getenv("MONGODB_NAME", "MarketInformation")

    agent = GlassdoorEmotionAgent(mongo_uri=mongo_uri, db_name=db_name)
    agent.run_pipeline(company_df)

    return {
        "status":            "analysis_complete",
        "company":           company_name,
        "month":             month,
        "reviews_processed": len(company_df),
        "next_step":         "Call get_emotion_summary and get_high_risk_reviews to query the results.",
    }


if __name__ == "__main__":
    raw_df = pd.read_csv("data/glassdoor_reviews.csv")

    agent = GlassdoorEmotionAgent(mongo_uri="mongodb://localhost:27017/")

    # Start with a small sample to verify the pipeline before running on the full dataset
    agent.run_pipeline(raw_df.head(500), batch_size=50)