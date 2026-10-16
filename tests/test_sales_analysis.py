import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.analysis.sales_analysis import (
    calculate_total_sales,
    sales_by_category,
    sales_by_region,
    sales_by_month
)


# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_retail_data.csv")


print("\n========== SALES ANALYSIS TEST ==========")


# 1. Total Sales
total_sales = calculate_total_sales(df)

print("\n--- TOTAL SALES ---")
print(f"Total Sales: ₹{total_sales:,.2f}")


# 2. Sales by Category
category_sales = sales_by_category(df)

print("\n--- SALES BY CATEGORY ---")
print(category_sales)


# 3. Sales by Region
region_sales = sales_by_region(df)

print("\n--- SALES BY REGION ---")
print(region_sales)


# 4. Sales by Month
monthly_sales = sales_by_month(df)

print("\n--- SALES BY MONTH ---")
print(monthly_sales)