import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import pandas as pd

from src.forecasting.demand_forecast import (
    prepare_daily_demand,
    moving_average_forecast,
    evaluate_moving_average,
    forecast_summary
)

# Load cleaned dataset
df = pd.read_csv(
    "data/processed/cleaned_retail_data.csv"
)


print("\n========== DEMAND FORECAST TEST ==========")


# Test 1: Prepare daily demand
daily_demand = prepare_daily_demand(
    df,
    store_id="S001",
    product_id="P0001"
)

print("\n--- DAILY DEMAND ---")
print(daily_demand.tail(10).to_string(index=False))


# Test 2: Forecast next 7 days
forecast = moving_average_forecast(
    df,
    store_id="S001",
    product_id="P0001",
    window=7,
    forecast_days=7
)

print("\n--- 7-DAY DEMAND FORECAST ---")
print(forecast.to_string(index=False))


print("\nForecast completed successfully.")


# Test 3: Evaluate forecast accuracy
mae = evaluate_moving_average(
    df,
    store_id="S001",
    product_id="P0001",
    window=7
)

print("\n--- FORECAST EVALUATION ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} units")

# Test 4: Forecast Summary
summary = forecast_summary(
    df,
    store_id="S001",
    product_id="P0001",
    window=7,
    forecast_days=7
)

print("\n--- FORECAST SUMMARY ---")
print("Status:", summary["status"])
print(
    "Historical Average Demand:",
    round(summary["historical_average_demand"], 2)
)
print(
    "Forecast Average Demand:",
    round(summary["forecast_average_demand"], 2)
)
print(
    "Forecast Days:",
    summary["forecast_days"]
)