import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.analysis.inventory_analysis import (
    total_inventory,
    inventory_by_category,
    inventory_by_region,
    low_inventory_products,
    inventory_vs_sales
)


# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_retail_data.csv")


print("\n========== INVENTORY ANALYSIS TEST ==========")


# 1. Total Inventory
inventory = total_inventory(df)

print("\n--- TOTAL INVENTORY ---")
print(f"Total Inventory: {inventory:,}")


# 2. Inventory by Category
category_inventory = inventory_by_category(df)

print("\n--- INVENTORY BY CATEGORY ---")
print(category_inventory)


# 3. Inventory by Region
region_inventory = inventory_by_region(df)

print("\n--- INVENTORY BY REGION ---")
print(region_inventory)


# 4. Low Inventory Products
low_inventory = low_inventory_products(df)

print("\n--- LOW INVENTORY PRODUCTS ---")

if low_inventory.empty:
    print("No products found below the threshold.")
else:
    print(low_inventory)


# 5. Inventory vs Sales
inventory_sales = inventory_vs_sales(df)

print("\n--- INVENTORY VS SALES ---")
print(inventory_sales)