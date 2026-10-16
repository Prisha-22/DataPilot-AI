import pandas as pd

from src.tools.analysis_tools import (
    load_cleaned_data,
    get_sales_by_category,
    get_sales_by_region,
    get_product_performance,
    get_high_risk_inventory,
    get_sales_trend,
)


def build_dataset_knowledge():
    """
    Convert important information from the retail dataset
    into text chunks that can later be embedded and retrieved.
    """

    df = load_cleaned_data()

    knowledge_chunks = []

    # ---------------------------------------------------------
    # 1. Dataset overview
    # ---------------------------------------------------------

    df["Date"] = pd.to_datetime(df["Date"])

    date_min = df["Date"].min().date()
    date_max = df["Date"].max().date()

    dataset_overview = {
        "rows": len(df),
        "columns": len(df.columns),
        "stores": df["Store ID"].nunique(),
        "products": df["Product ID"].nunique(),
        "categories": df["Category"].nunique(),
        "regions": df["Region"].nunique(),
        "date_start": str(date_min),
        "date_end": str(date_max),
    }

    knowledge_chunks.append({
        "id": "dataset_overview",
        "text": f"""
DataPilot AI retail dataset overview:

The dataset contains {dataset_overview["rows"]:,} rows
and {dataset_overview["columns"]} columns.

It covers:
- {dataset_overview["stores"]} stores
- {dataset_overview["products"]} products
- {dataset_overview["categories"]} product categories
- {dataset_overview["regions"]} regions

Date range:
{dataset_overview["date_start"]} to {dataset_overview["date_end"]}.

The dataset contains sales, inventory, demand forecast,
pricing, discount, weather, promotion, competitor pricing,
and seasonality information.
""".strip()
    })

    # ---------------------------------------------------------
    # 2. Data quality knowledge
    # ---------------------------------------------------------

    missing_values = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    knowledge_chunks.append({
        "id": "data_quality",
        "text": f"""
Data quality information for the cleaned retail dataset:

Total missing values: {missing_values}
Duplicate rows: {duplicate_rows}

The cleaning pipeline converts the Date column to a
datetime format as part of preprocessing.

During quality checking, negative Demand Forecast values
were identified as invalid because retail demand cannot be
negative. These values were replaced with 0.
""".strip()
    })

    # ---------------------------------------------------------
    # 3. Sales by category
    # ---------------------------------------------------------

    category_sales = get_sales_by_category()

    for category, sales in category_sales.items():

        knowledge_chunks.append({
            "id": f"category_sales_{category}",
            "text": (
                f"Retail category sales information: "
                f"The {category} category generated "
                f"₹{sales:,.2f} in total sales across the dataset."
            )
        })

    # ---------------------------------------------------------
    # 4. Sales by region
    # ---------------------------------------------------------

    region_sales = get_sales_by_region()

    for region, sales in region_sales.items():

        knowledge_chunks.append({
            "id": f"region_sales_{region}",
            "text": (
                f"Retail regional sales information: "
                f"The {region} region generated "
                f"₹{sales:,.2f} in total sales across the dataset."
            )
        })

    # ---------------------------------------------------------
    # 5. Product performance
    # ---------------------------------------------------------

    product_performance = get_product_performance()

    top_products = product_performance.head(20)

    for product_id, row in top_products.iterrows():

        knowledge_chunks.append({
            "id": f"product_{product_id}",
            "text": f"""
Product performance information:

Product ID: {product_id}

Total sales: ₹{row["Total_Sales"]:,.2f}
Total units sold: {row["Total_Units_Sold"]:,.0f}
Average inventory: {row["Average_Inventory"]:,.2f}
Total demand forecast: {row["Total_Demand_Forecast"]:,.2f}
""".strip()
        })

    # ---------------------------------------------------------
    # 6. Sales trend
    # ---------------------------------------------------------

    trend = get_sales_trend()

    if trend:

        knowledge_chunks.append({
            "id": "sales_trend",
            "text": f"""
Recent sales trend information:

Recent 7-day average daily sales:
₹{trend["recent_average_sales"]:,.2f}

Previous 7-day average daily sales:
₹{trend["previous_average_sales"]:,.2f}

Percentage change:
{trend["percentage_change"]:.2f}%

Observed trend:
{trend["trend"]}

This is a descriptive comparison of two recent periods,
not a future sales prediction.
""".strip()
        })

    # ---------------------------------------------------------
    # 7. Inventory risk
    # ---------------------------------------------------------

    high_risk = get_high_risk_inventory(
        recent_days=7
    )

    knowledge_chunks.append({
        "id": "inventory_risk_summary",
        "text": f"""
Inventory risk information:

Using a 7-day recent demand window,
{len(high_risk)} store-product combinations are classified
as High Stock-Out Risk.

The project's High Stock-Out Risk definition is:
inventory coverage below 3 days.

The most critical store-product combinations are:

{high_risk.head(10).to_string(index=False)}
""".strip()
    })

    return knowledge_chunks