import random
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from database_mysql import DatabaseManager

import certifi
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv("config.env")

# ── MySQL table name ─────────────────────────────────────────────────────────
EMPLOYEE_TABLE  = "employee_risk_profiles"
EMPLOYEE_ID_COL = "employee_id"
# ─────────────────────────────────────────────────────────────────────────────

# Sentence pools by topic and tone — comments are assembled by randomly
# combining 2–5 sentences so each employee gets a unique, natural-length review.

_SENTENCES = {
    "management_neg": [
        "Management is completely disconnected from what engineers actually do.",
        "Senior leadership changes priorities every quarter, making it impossible to ship anything meaningful.",
        "My manager consistently takes credit for the team's work during review cycles and never advocates for us.",
        "There are too many layers of management — getting a simple decision approved takes weeks of escalations.",
        "Directors have no technical background yet override every engineering decision, which is deeply frustrating.",
        "Leadership promotes based on politics rather than performance; if you are not in the right circle you will not advance.",
        "One-on-ones are cancelled more often than they happen and feedback is almost never constructive.",
        "The constant reorganisations mean you report to a different manager every six months and nothing ever gets prioritised.",
    ],
    "management_pos": [
        "My direct manager is genuinely supportive and makes time for regular, useful one-on-ones.",
        "Leadership communicates the company vision clearly and I always understand how my work fits in.",
        "Senior management listens to feedback and has actually changed course based on employee input.",
        "The executive team is transparent about business performance and does not hide bad news.",
    ],
    "workload_neg": [
        "Constant on-call rotations with no compensation or time off in lieu are burning people out fast.",
        "I am expected to be available on weekends and evenings; work-life balance is essentially a myth here.",
        "Headcount keeps shrinking but the scope of work keeps growing — the team is stretched impossibly thin.",
        "Product deadlines are always unrealistic and crunch mode has become the default rather than the exception.",
        "There is no real PTO culture; people stay connected even when officially on vacation and nobody pushes back.",
        "Meetings fill the entire day and there is no protected focus time to actually complete deep work.",
        "The culture glorifies overwork — leaving at a normal hour is seen as a sign of low commitment.",
    ],
    "workload_pos": [
        "The company respects personal time and I have never felt pressured to work outside normal hours.",
        "Meeting hygiene is good here — agendas are shared in advance and sessions end on time.",
        "I have genuine flexibility in how and when I structure my workday, which makes a huge difference.",
    ],
    "career_neg": [
        "Promotion criteria are vague and inconsistently applied — I have been at the same level for three years with no clear path.",
        "There is no real internal mobility program; switching teams means going through a full external-style interview process.",
        "The training and development budget is practically zero and you are expected to upskill entirely on your own time.",
        "The organisation is flat at senior levels so there is a hard ceiling you hit relatively quickly.",
        "Performance reviews are a black box — scores arrive with no explanation and calibration feels arbitrary.",
        "Career ladder documentation is outdated and every manager interprets the levels differently.",
        "High performers leave because there is simply no room to grow beyond a certain point.",
    ],
    "career_pos": [
        "There are genuine opportunities to move into new roles and the internal job board is actively used.",
        "The company invested in my professional development and fully funded a certification I wanted to pursue.",
        "Promotion timelines are clearly communicated and I always know what I need to demonstrate to reach the next level.",
        "I have been able to switch teams twice and each move was supported by both my old and new managers.",
    ],
    "culture_neg": [
        "The culture has shifted dramatically since the last round of layoffs — trust is at an all-time low.",
        "There is a strong in-group/out-group dynamic and if you are not visible to the right people your work goes unnoticed.",
        "Collaboration across teams is nearly impossible because everyone is protective of their own metrics.",
        "The stated values look great on the website but bear little resemblance to how decisions are actually made.",
    ],
    "culture_pos": [
        "The team genuinely cares about each other and collaboration feels effortless most of the time.",
        "There is a strong culture of psychological safety — I feel comfortable raising concerns without fear.",
        "Diversity is taken seriously here and it shows in who gets promoted and who is given stretch assignments.",
        "The work is genuinely interesting and I am proud of the impact we have on customers.",
        "Despite the challenges, I still believe in the mission and that keeps me engaged day to day.",
    ],
    "compensation_neg": [
        "Compensation has not kept pace with inflation and colleagues at competitors earn significantly more for equivalent roles.",
        "Equity refreshes are small and the total comp story told during recruiting does not match reality.",
        "Bonuses are discretionary and almost always below target — the variable component feels meaningless.",
        "No salary bands are disclosed internally, so you only find out you are underpaid when a new hire earns more than you.",
    ],
    "compensation_pos": [
        "The base salary is competitive and the equity package has been a meaningful part of my total compensation.",
        "Benefits are genuinely good — health coverage is comprehensive and the parental leave policy is among the best I have seen.",
    ],
}

def _build_comment() -> str:
    """Assemble a varied comment by sampling sentences from different topic pools."""
    # Pick 2–4 topic pools randomly, weighted toward negative topics for realism
    all_pools = list(_SENTENCES.keys())
    neg_pools = [p for p in all_pools if p.endswith("_neg")]
    pos_pools = [p for p in all_pools if p.endswith("_pos")]

    # 60% chance of mostly-negative, 20% mixed, 20% mostly-positive
    roll = random.random()
    if roll < 0.60:
        chosen = random.sample(neg_pools, k=min(random.randint(2, 3), len(neg_pools)))
        if random.random() < 0.4:
            chosen += random.sample(pos_pools, k=1)
    elif roll < 0.80:
        chosen = random.sample(neg_pools, k=1) + random.sample(pos_pools, k=random.randint(1, 2))
    else:
        chosen = random.sample(pos_pools, k=min(random.randint(2, 3), len(pos_pools)))
        if random.random() < 0.3:
            chosen += random.sample(neg_pools, k=1)

    sentences = []
    for pool in chosen:
        sentences.append(random.choice(_SENTENCES[pool]))

    random.shuffle(sentences)
    return " ".join(sentences)


def get_employee_ids(db: DatabaseManager) -> list[int]:
    rows = db.fetch_all(f"SELECT {EMPLOYEE_ID_COL} FROM {EMPLOYEE_TABLE}")
    if not rows:
        raise RuntimeError(f"No rows returned from {EMPLOYEE_TABLE}.{EMPLOYEE_ID_COL}")
    return [row[0] for row in rows]


def mock_comment(employee_id: int) -> dict:
    return {
        "employee_id": employee_id,
        "comment":     _build_comment(),
        "created_at":  datetime.now(timezone.utc).isoformat(),
    }


def save_to_mongo(docs: list[dict]) -> None:
    uri             = os.getenv("MONGODB_URI")
    db_name         = os.getenv("MONGODB_NAME", "MarketInformation")
    collection_name = "employee_comment"

    client = MongoClient(uri, serverSelectionTimeoutMS=5000, tlsCAFile=certifi.where())
    db = client[db_name]

    if collection_name not in db.list_collection_names():
        db.create_collection(collection_name)
        print(f"[MongoDB] Created collection '{collection_name}'")

    result = db[collection_name].insert_many(docs)
    print(f"[MongoDB] Saved {len(result.inserted_ids)} comments → collection='{collection_name}'")
    client.close()


if __name__ == "__main__":
    db = DatabaseManager()

    print("[MySQL] Fetching employee IDs ...")
    employee_ids = get_employee_ids(db)
    print(f"[MySQL] Found {len(employee_ids)} employees")

    docs = [mock_comment(eid) for eid in employee_ids]

    save_to_mongo(docs)