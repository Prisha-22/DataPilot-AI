import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.cleaning.clean import clean_dataset


df = pd.read_csv("data/raw/retail_store_inventory.csv")

cleaned_df, cleaning_log = clean_dataset(df)

print("\n--- CLEANING TEST ---")

print("\nOriginal rows:", len(df))
print("Cleaned rows:", len(cleaned_df))

print("\nNegative Demand Forecast BEFORE:",
      (df["Demand Forecast"] < 0).sum())

print("Negative Demand Forecast AFTER:",
      (cleaned_df["Demand Forecast"] < 0).sum())

print("\nDate data type AFTER:",
      cleaned_df["Date"].dtype)

print("\n--- CLEANING LOG ---")
for entry in cleaning_log:
    print(entry)

cleaned_df.to_csv(
    "data/processed/cleaned_retail_data.csv",
    index=False
)

print("\nCleaned dataset saved successfully.")