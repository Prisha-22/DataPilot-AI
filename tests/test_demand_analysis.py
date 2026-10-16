import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.analysis.demand_analysis import (
    calculate_inventory_coverage,
    classify_inventory_risk,
    high_risk_inventory
)


# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_retail_data.csv")


print("\n========== INVENTORY COVERAGE TEST ==========")


# 1. Inventory Coverage
coverage = calculate_inventory_coverage(df)

print("\n--- INVENTORY COVERAGE SAMPLE ---")
print(coverage.head(20).to_string(index=False))


# 2. Risk Classification
risk = classify_inventory_risk(df)

print("\n--- RISK DISTRIBUTION ---")
print(risk["Risk"].value_counts())


# 3. High Risk Items
high_risk = high_risk_inventory(df)

print("\n--- HIGH STOCK-OUT RISK ITEMS ---")

if high_risk.empty:
    print("No high-risk items found.")
else:
    print(high_risk.head(20).to_string(index=False))