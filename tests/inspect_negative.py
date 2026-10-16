import pandas as pd


df = pd.read_csv("data/raw/retail_store_inventory.csv")

negative_rows = df[df["Demand Forecast"] < 0]

print("Number of negative Demand Forecast rows:", len(negative_rows))

print("\n--- NEGATIVE DEMAND FORECAST VALUES ---")
print(negative_rows["Demand Forecast"].describe())

print("\n--- SAMPLE NEGATIVE ROWS ---")
print(negative_rows.head(20).to_string(index=False))