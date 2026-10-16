import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from src.tools.analysis_tools import (
    get_total_sales,
    get_total_inventory,
    get_sales_trend,
    get_high_risk_inventory,
    get_demand_forecast
)


print("\n========== ANALYSIS TOOLS TEST ==========")


# Test 1: Total Sales
total_sales = get_total_sales()

print("\n--- TOTAL SALES TOOL ---")
print(f"Total Sales: ₹{total_sales:,.2f}")


# Test 2: Total Inventory
total_inventory = get_total_inventory()

print("\n--- TOTAL INVENTORY TOOL ---")
print(f"Total Inventory: {total_inventory:,}")


# Test 3: Sales Trend
trend = get_sales_trend()

print("\n--- SALES TREND TOOL ---")

if trend:
    print(
        f"Trend: {trend['trend']}"
    )

    print(
        f"Percentage Change: "
        f"{trend['percentage_change']:.2f}%"
    )


# Test 4: High-Risk Inventory
high_risk = get_high_risk_inventory()

print("\n--- HIGH-RISK INVENTORY TOOL ---")
print(
    f"High-risk items found: {len(high_risk)}"
)

print(
    high_risk.head(5).to_string(
        index=False
    )
)


# Test 5: Demand Forecast
forecast = get_demand_forecast(
    store_id="S001",
    product_id="P0001",
    window=7,
    forecast_days=7
)

print("\n--- DEMAND FORECAST TOOL ---")
print(
    forecast.to_string(
        index=False
    )
)


print("\nAll analysis tools completed successfully.")