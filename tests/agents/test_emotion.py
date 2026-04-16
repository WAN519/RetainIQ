"""
tests/agents/test_emotion.py

Unit tests for GlassdoorEmotionAgent and EmotionOrchestrator.
All ML models, MongoDB, and the Gemini API are mocked — no GPU or network needed.

Heavy ML packages (sentence_transformers, transformers, torch) are stubbed out
in sys.modules before import so the test environment doesn't need them installed.
"""

import sys
from unittest.mock import MagicMock, patch

# --- Stub out heavy ML dependencies before importing the agents ---
for mod_name in [
    "torch", "torch.backends", "torch.backends.mps",
    "sentence_transformers", "sentence_transformers.util",
    "transformers",
    "tqdm", "tqdm.auto",
    "google", "google.generativeai",
]:
    sys.modules.setdefault(mod_name, MagicMock())

# Ensure torch.backends.mps.is_available() returns False (CPU mode)
sys.modules["torch"].backends.mps.is_available.return_value = False

# tqdm must be a passthrough so pipeline for-loops actually execute
_tqdm_module = MagicMock()
_tqdm_module.tqdm = lambda iterable, **kwargs: iterable
sys.modules["tqdm"] = _tqdm_module

import pytest
import pandas as pd
from tools.emotion_tool import GlassdoorEmotionAgent
from agents.emotion.emotion_agent import EmotionOrchestrator


# ---------------------------------------------------------------------------
# Fixture: emotion agent with mocked models and DB
# ---------------------------------------------------------------------------

@pytest.fixture
def emotion_agent():
    with patch("tools.emotion_tool.SentenceTransformer"), \
         patch("tools.emotion_tool.pipeline"), \
         patch("tools.emotion_tool.MongoClient"):
        agent = GlassdoorEmotionAgent(mongo_uri="mongodb://localhost:27017/")
    return agent


# ---------------------------------------------------------------------------
# _clean_and_preprocess
# ---------------------------------------------------------------------------

def test_clean_fills_missing_cons(emotion_agent, sample_review_df):
    result = emotion_agent._clean_and_preprocess(sample_review_df)
    # Row 2 has empty cons — should remain empty string, not NaN
    assert result["cons"].iloc[2] == ""


def test_clean_truncates_long_cons(emotion_agent):
    long_text = "x" * 2000
    df = pd.DataFrame([{"cons": long_text, "recommend": "v", "current": "Current Employee"}])
    result = emotion_agent._clean_and_preprocess(df)
    assert len(result["cons"].iloc[0]) == 1000


def test_clean_recommend_mapping(emotion_agent, sample_review_df):
    result = emotion_agent._clean_and_preprocess(sample_review_df)
    # 'x' → -1, 'v' → 1, 'o' → 0
    assert result["recommend_num"].tolist() == [-1, 1, 0]


def test_clean_is_former_flag(emotion_agent, sample_review_df):
    result = emotion_agent._clean_and_preprocess(sample_review_df)
    assert result["is_former"].iloc[0] == 1   # "Former Employee"
    assert result["is_former"].iloc[1] == 0   # "Current Employee"


# ---------------------------------------------------------------------------
# run_pipeline — mock model outputs, verify MongoDB insert is called
# ---------------------------------------------------------------------------

def test_run_pipeline_inserts_to_mongo(emotion_agent, sample_review_df):
    # Build a cos_sim mock where [:, idx] > threshold returns a proper tolist()
    col_mock = MagicMock()
    col_mock.__gt__ = MagicMock(
        return_value=MagicMock(
            int=MagicMock(return_value=MagicMock(tolist=MagicMock(return_value=[0, 0, 0])))
        )
    )
    scores_mock = MagicMock()
    scores_mock.__getitem__ = MagicMock(return_value=col_mock)

    emotion_agent.emotion_pipe.return_value = [
        {"label": "Negative", "score": 0.9},
        {"label": "Positive", "score": 0.8},
        {"label": "Neutral",  "score": 0.6},
    ]
    emotion_agent.collection.insert_many = MagicMock()

    with patch("tools.emotion_tool.util.cos_sim", return_value=scores_mock):
        emotion_agent.run_pipeline(sample_review_df, batch_size=10)

    emotion_agent.collection.insert_many.assert_called_once()


# ---------------------------------------------------------------------------
# EmotionOrchestrator
# ---------------------------------------------------------------------------

@pytest.fixture
def orchestrator():
    with patch("agents.emotion.emotion_agent.genai"), \
         patch("tools.mongoDB.MarketDB") as mock_db_cls:
        mock_db = MagicMock()
        mock_db.collection.count_documents.return_value = 0
        mock_db_cls.return_value = mock_db

        orch = EmotionOrchestrator(agent_instance=MagicMock())
        orch.db = mock_db
    return orch


def test_generate_report_returns_no_data_message(orchestrator):
    result = orchestrator._generate_gemini_report("Apple", None)
    assert result == "No data available for this company and month."


def test_generate_report_includes_company_name_in_prompt(orchestrator):
    mock_response = MagicMock()
    mock_response.text = '{"risk_level": "High"}'
    orchestrator.llm = MagicMock()
    orchestrator.llm.generate_content.return_value = mock_response

    stats = {
        "total_count": 100, "avg_sentiment": 0.75,
        "mgmt_issues": 30, "salary_issues": 20, "workload_issues": 10,
    }
    orchestrator._generate_gemini_report("Google", stats)

    prompt_text = orchestrator.llm.generate_content.call_args[0][0]
    assert "Google" in prompt_text

def test_generate_report_returns_llm_text(orchestrator):
    orchestrator.llm = MagicMock()
    orchestrator.llm.generate_content.return_value.text = '{"risk_level": "Low"}'

    stats = {
        "total_count": 50, "avg_sentiment": 0.4,
        "mgmt_issues": 5, "salary_issues": 3, "workload_issues": 2,
    }
    result = orchestrator._generate_gemini_report("Acme", stats)
    assert result == '{"risk_level": "Low"}'