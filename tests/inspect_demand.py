import pandas as pd


# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_retail_data.csv")


print("\n========== DEMAND INSPECTION ==========")


print("\n--- BASIC DEMAND STATISTICS ---")
print(df["Demand Forecast"].describe())


print("\n--- UNITS SOLD STATISTICS ---")
print(df["Units Sold"].describe())


print("\n--- INVENTORY STATISTICS ---")
print(df["Inventory Level"].describe())


print("\n--- AVERAGE VALUES ---")
print("Average Units Sold:",
      round(df["Units Sold"].mean(), 2))

print("Average Demand Forecast:",
      round(df["Demand Forecast"].mean(), 2))

print("Average Inventory Level:",
      round(df["Inventory Level"].mean(), 2))


print("\n--- DAILY TOTALS ---")

daily = (
    df.groupby("Date")
    .agg(
        Total_Units_Sold=("Units Sold", "sum"),
        Total_Demand_Forecast=("Demand Forecast", "sum"),
        Total_Inventory=("Inventory Level", "sum")
    )
)

print(daily.head(10))


print("\n--- CORRELATION ---")

print(
    df[
        ["Units Sold", "Demand Forecast", "Inventory Level"]
    ].corr()
)