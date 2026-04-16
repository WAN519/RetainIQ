"""
scripts/generate_mock_reviews.py

Generate a small mock Glassdoor reviews CSV for debugging the emotion analysis
pipeline without loading the full 22k-row dataset.

Output: data/mock_reviews.csv  (30 reviews, firm = "Apple")

Usage:
    python scripts/generate_mock_reviews.py
"""

import csv
import random
from pathlib import Path

random.seed(42)

CONS_TEMPLATES = [
    # Management issues
    "Management is completely disconnected from what engineers actually do. Directors micromanage every decision and there is no trust.",
    "Senior leadership changes priorities every quarter. It is impossible to ship anything because strategy keeps shifting.",
    "My manager takes credit for the team's work in every review cycle. Feedback is never constructive.",
    "Too many layers of management. Getting a simple decision approved takes weeks and multiple escalations.",
    "Leadership is toxic. Managers yell in meetings and nobody in HR seems to care.",
    "Management promotes based on politics not performance. If you are not in the right clique you will not advance.",
    "Directors have no technical background yet override every engineering decision. Very frustrating.",

    # Salary / compensation issues
    "Pay is way below market for the Bay Area. I got a 2% raise after a strong performance review which is insulting.",
    "Compensation has not kept up with inflation. Colleagues at competitors earn 40% more for the same role.",
    "Equity refreshes are tiny compared to what is offered at other big tech companies.",
    "Bonuses are discretionary and almost always below target. The total comp story they tell during recruiting is misleading.",
    "No salary bands are disclosed. You only find out you are underpaid when a new hire gets more than you.",
    "Benefits have been cut three years in a row. Health insurance coverage is now worse than most competitors.",

    # Workload / burnout issues
    "Constant on-call rotations with no compensation or time off in lieu. Burnout is a real problem here.",
    "Expected to be available on weekends and evenings. Work-life balance is a myth at this company.",
    "Headcount keeps shrinking but the work keeps growing. The team is stretched way too thin.",
    "The culture glorifies overwork. If you leave at 6pm people assume you are not committed.",
    "Product deadlines are always unrealistic. Crunch mode is the default not the exception.",
    "No real PTO culture. People are pressured to stay connected even when officially on vacation.",
    "Meetings all day with no heads-down focus time. Impossible to actually get work done during business hours.",

    # Career / growth issues
    "Promotion criteria are never clearly explained. I have been at the same level for three years with no path forward.",
    "There is no internal mobility program. If you want a new role you basically have to re-interview from scratch.",
    "Training budget is practically zero. You are expected to grow on your own time and own dime.",
    "The org is very flat at senior levels so there is a ceiling you hit very quickly.",
    "Performance reviews are a black box. You never know how you are actually being assessed.",
    "Career ladder documentation is outdated and managers interpret it differently. There is no consistency.",

    # Mixed issues
    "Pay is decent but management issues and lack of growth make it hard to stay motivated.",
    "Great brand name but the actual day-to-day is exhausting. Work-life balance is terrible and salary is stagnant.",
    "I liked my team but upper management is out of touch and there is no path to promotion.",
    "The company is coasting on its reputation. Internal tools are broken, processes are slow, and pay is falling behind.",
]

RECOMMEND  = ['x', 'x', 'x', 'o', 'v', 'x', 'o']
CURRENT    = [
    "Current Employee, more than 3 years",
    "Current Employee, more than 1 year",
    "Former Employee, more than 2 years",
    "Former Employee, more than 5 years",
    "Current Employee, less than 1 year",
]
JOB_TITLES = [
    "Software Engineer", "Senior Software Engineer", "Product Manager",
    "Data Scientist", "Engineering Manager", "UX Designer",
    "Site Reliability Engineer", "Technical Program Manager",
]

rows = []
for i, cons in enumerate(CONS_TEMPLATES):
    rows.append({
        "firm":            "Apple",
        "date_review":     f"2026-0{(i % 3) + 1}-{(i % 28) + 1:02d}",
        "job_title":       random.choice(JOB_TITLES),
        "current":         random.choice(CURRENT),
        "location":        "Cupertino, CA",
        "overall_rating":  random.choice([1, 2, 2, 3]),
        "work_life_balance": random.choice([1, 2, 2, 3]),
        "culture_values":  random.choice([2, 3, 3]),
        "career_opp":      random.choice([1, 2, 2]),
        "comp_benefits":   random.choice([1, 2, 3]),
        "senior_mgmt":     random.choice([1, 2, 2]),
        "recommend":       random.choice(RECOMMEND),
        "headline":        "Review " + str(i + 1),
        "pros":            "Good brand name and smart colleagues.",
        "cons":            cons,
    })

output_path = Path(__file__).parent.parent / "data" / "mock_reviews.csv"
output_path.parent.mkdir(exist_ok=True)

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {len(rows)} mock reviews → {output_path}")