import pandas as pd


def total_inventory(df):
    """
    Calculate total inventory available.
    """

    return int(df["Inventory Level"].sum())


def inventory_by_category(df):
    """
    Calculate total inventory for each category.
    """

    result = (
        df.groupby("Category")["Inventory Level"]
        .sum()
        .sort_values(ascending=False)
    )

    return result


def inventory_by_region(df):
    """
    Calculate total inventory for each region.
    """

    result = (
        df.groupby("Region")["Inventory Level"]
        .sum()
        .sort_values(ascending=False)
    )

    return result


def low_inventory_products(df, threshold=20):
    """
    Identify products whose inventory is at or below the given threshold.
    """

    result = (
        df.groupby("Product ID")["Inventory Level"]
        .mean()
        .sort_values()
    )

    return result[result <= threshold]


def inventory_vs_sales(df):
    """
    Compare average inventory with total units sold for each product.
    """

    result = (
        df.groupby("Product ID")
        .agg(
            Average_Inventory=("Inventory Level", "mean"),
            Total_Units_Sold=("Units Sold", "sum")
        )
        .sort_values("Total_Units_Sold", ascending=False)
    )

    return result