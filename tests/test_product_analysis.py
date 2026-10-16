import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.analysis.product_analysis import (
    product_sales,
    product_units_sold,
    product_demand,
    product_performance
)


# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_retail_data.csv")


print("\n========== PRODUCT ANALYSIS TEST ==========")


# 1. Sales by Product
sales = product_sales(df)

print("\n--- SALES BY PRODUCT ---")
print(sales)


# 2. Units Sold by Product
units = product_units_sold(df)

print("\n--- UNITS SOLD BY PRODUCT ---")
print(units)


# 3. Demand Forecast by Product
demand = product_demand(df)

print("\n--- DEMAND FORECAST BY PRODUCT ---")
print(demand)


# 4. Complete Product Performance
performance = product_performance(df)

print("\n--- PRODUCT PERFORMANCE ---")
print(performance)