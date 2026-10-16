import pandas as pd
import os

from src.analysis.sales_analysis import (
    calculate_total_sales,
    sales_by_category,
    sales_by_region,
    sales_by_month
)

from src.analysis.inventory_analysis import (
    total_inventory,
    inventory_by_category,
    inventory_by_region,
    low_inventory_products,
    inventory_vs_sales
)

from src.analysis.product_analysis import (
    product_sales,
    product_units_sold,
    product_demand,
    product_performance
)

from src.analysis.demand_analysis import (
    calculate_inventory_coverage,
    classify_inventory_risk,
    high_risk_inventory
)

from src.analysis.trend_analysis import (
    calculate_daily_sales,
    calculate_sales_trend
)

from src.forecasting.demand_forecast import (
    prepare_daily_demand,
    moving_average_forecast,
    evaluate_moving_average,
    forecast_summary
)


def load_cleaned_data():
    """
    Load the currently active DataPilot dataset.

    If DATAPILOT_DATASET_PATH is set, use that dataset.
    Otherwise, fall back to the default processed dataset.
    """

    dataset_path = os.getenv(
        "DATAPILOT_DATASET_PATH",
        "data/processed/cleaned_retail_data.csv"
    )

    return pd.read_csv(dataset_path)


def get_total_sales():
    """
    Tool for calculating total sales.
    """

    df = load_cleaned_data()

    return calculate_total_sales(df)


def get_sales_by_category():
    """
    Tool for category-wise sales analysis.
    """

    df = load_cleaned_data()

    return sales_by_category(df)


def get_sales_by_region():
    """
    Tool for region-wise sales analysis.
    """

    df = load_cleaned_data()

    return sales_by_region(df)


def get_sales_by_month():
    """
    Tool for monthly sales analysis.
    """

    df = load_cleaned_data()

    return sales_by_month(df)


def get_total_inventory():
    """
    Tool for calculating total inventory.
    """

    df = load_cleaned_data()

    return total_inventory(df)


def get_inventory_by_category():
    """
    Tool for category-wise inventory analysis.
    """

    df = load_cleaned_data()

    return inventory_by_category(df)


def get_inventory_by_region():
    """
    Tool for region-wise inventory analysis.
    """

    df = load_cleaned_data()

    return inventory_by_region(df)


def get_low_inventory_products(
    threshold=20
):
    """
    Tool for finding products with low inventory.
    """

    df = load_cleaned_data()

    return low_inventory_products(
        df,
        threshold=threshold
    )


def get_inventory_vs_sales():
    """
    Tool for comparing inventory and sales.
    """

    df = load_cleaned_data()

    return inventory_vs_sales(df)


def get_product_sales():
    """
    Tool for product-wise sales analysis.
    """

    df = load_cleaned_data()

    return product_sales(df)


def get_product_units_sold():
    """
    Tool for product-wise units sold.
    """

    df = load_cleaned_data()

    return product_units_sold(df)


def get_product_demand():
    """
    Tool for product-wise demand analysis.
    """

    df = load_cleaned_data()

    return product_demand(df)


def get_product_performance():
    """
    Tool for complete product performance.
    """

    df = load_cleaned_data()

    return product_performance(df)


def get_inventory_coverage(
    recent_days=30
):
    """
    Tool for calculating inventory coverage.
    """

    df = load_cleaned_data()

    return calculate_inventory_coverage(
        df,
        recent_days=recent_days
    )


def get_inventory_risk(
    recent_days=30
):
    """
    Tool for classifying inventory risk.
    """

    df = load_cleaned_data()

    return classify_inventory_risk(
        df,
        recent_days=recent_days
    )


def get_high_risk_inventory(
    recent_days=30
):
    """
    Tool for finding high stock-out risk items.
    """

    df = load_cleaned_data()

    return high_risk_inventory(
        df,
        recent_days=recent_days
    )


def get_sales_trend():
    """
    Tool for sales trend analysis.
    """

    df = load_cleaned_data()

    return calculate_sales_trend(df)


def get_demand_forecast(
    store_id=None,
    product_id=None,
    window=7,
    forecast_days=7
):
    """
    Tool for generating demand forecasts.
    """

    df = load_cleaned_data()

    return moving_average_forecast(
        df,
        store_id=store_id,
        product_id=product_id,
        window=window,
        forecast_days=forecast_days
    )


def get_forecast_evaluation(
    store_id=None,
    product_id=None,
    window=7
):
    """
    Tool for evaluating forecast accuracy.
    """

    df = load_cleaned_data()

    return evaluate_moving_average(
        df,
        store_id=store_id,
        product_id=product_id,
        window=window
    )


def get_forecast_summary(
    store_id=None,
    product_id=None,
    window=7,
    forecast_days=7
):
    """
    Tool for generating a complete forecast summary.
    """

    df = load_cleaned_data()

    return forecast_summary(
        df,
        store_id=store_id,
        product_id=product_id,
        window=window,
        forecast_days=forecast_days
    )