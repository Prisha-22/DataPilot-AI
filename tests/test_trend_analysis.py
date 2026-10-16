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

from src.analysis.trend_analysis import (
    calculate_daily_sales,
    calculate_sales_trend
)


# Load cleaned dataset
df = pd.read_csv(
    "data/processed/cleaned_retail_data.csv"
)


print("\n========== TREND ANALYSIS TEST ==========")


# Test 1: Daily Sales
daily_sales = calculate_daily_sales(df)

print("\n--- DAILY SALES ---")
print(
    daily_sales.tail(10).to_string(
        index=False
    )
)


# Test 2: Sales Trend
trend = calculate_sales_trend(df)

print("\n--- SALES TREND ---")

if trend is None:
    print("Not enough data for trend analysis.")
else:
    print(
        "Recent 7-Day Average Sales: ₹"
        f"{trend['recent_average_sales']:,.2f}"
    )

    print(
        "Previous 7-Day Average Sales: ₹"
        f"{trend['previous_average_sales']:,.2f}"
    )

    print(
        "Percentage Change: "
        f"{trend['percentage_change']:.2f}%"
    )

    print(
        "Trend:",
        trend["trend"]
    )


print("\nTrend analysis completed successfully.")