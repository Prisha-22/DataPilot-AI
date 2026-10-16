import pandas as pd

df = pd.read_csv("data/raw/retail_store_inventory.csv")

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- UNIQUE VALUES ---")
for column in df.columns:
    print(f"\n{column}:")
    print(df[column].unique()[:20])

print("\n--- BASIC STATISTICS ---")
print(df.describe(include="all"))