"""
scripts/check_mysql_schema.py

检查 employee_risk_profiles 表的列，与 Cox 模型所需特征对比。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from database_mysql import DatabaseManager

# Cox 模型所需的全部特征
COX_REQUIRED = [
    "MonthlyIncome",           # 或 current_salary / 12
    "PercentSalaryHike",
    "YearsSinceLastPromotion",
    "YearsAtCompany",
    "OverTime_flag",
    "WorkLifeBalance",
    "JobInvolvement",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "JobLevel",
    "TotalWorkingYears",
]

# 薪资对比所需（非 Cox 特征，但 risk_scorer 要用）
SALARY_REQUIRED = [
    "employee_id",
    "current_salary",   # 或 MonthlyIncome * 12
    "role_name",
]

ALIASES = {
    "MonthlyIncome": ["monthly_income", "salary", "current_salary"],
}

def main():
    db = DatabaseManager()

    # 查询表的所有列名
    rows = db.fetch_all("DESCRIBE employee_risk_profiles")
    if not rows:
        print("[ERROR] 无法查询 employee_risk_profiles 表，请检查连接或表名")
        return

    actual_cols = {row[0].lower(): row[0] for row in rows}  # lower → original

    print(f"\n{'='*55}")
    print(f"  employee_risk_profiles 实际列 ({len(actual_cols)} 个)")
    print(f"{'='*55}")
    for orig in actual_cols.values():
        print(f"  • {orig}")

    print(f"\n{'='*55}")
    print("  Cox 特征对比")
    print(f"{'='*55}")

    missing = []
    for col in COX_REQUIRED + SALARY_REQUIRED:
        found = col.lower() in actual_cols
        # 检查别名
        if not found:
            for alias in ALIASES.get(col, []):
                if alias.lower() in actual_cols:
                    found = True
                    print(f"  [别名] {col:30s} → {actual_cols[alias.lower()]}")
                    break
        if found and col not in [a for aliases in ALIASES.values() for a in aliases]:
            print(f"  [OK]   {col}")
        elif not found:
            missing.append(col)
            print(f"  [缺少] {col}")

    print(f"\n{'='*55}")
    if missing:
        print(f"  缺少 {len(missing)} 个列，无法直接用 MySQL 数据跑 Cox 模型：")
        for m in missing:
            print(f"    - {m}")
        print("\n  建议：生成 mock 数据补齐缺失列，或直接用 ibm_enhanced_test.csv")
    else:
        print("  所有必要列都存在，可以直接从 MySQL 读取数据！")
    print(f"{'='*55}\n")

    # 顺便看一行样本数据
    sample = db.fetch_one("SELECT * FROM employee_risk_profiles LIMIT 1")
    if sample:
        col_names = [row[0] for row in rows]
        print("  样本数据（第一行）：")
        for name, val in zip(col_names, sample):
            print(f"    {name}: {val}")

if __name__ == "__main__":
    main()