import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.profiling.profile import profile_dataset


df = pd.read_csv("data/raw/retail_store_inventory.csv")

profile = profile_dataset(df)

print("Rows:", profile["rows"])
print("Columns:", profile["columns"])
print("Column Names:", profile["column_names"])
print("Duplicate Rows:", profile["duplicate_rows"])
print("Missing Values:", profile["missing_values"])
print("Data Types:", profile["data_types"])