import pandas as pd


def product_sales(df):
    """
    Calculate total sales for each product.
    """

    data = df.copy()

    data["Sales"] = data["Units Sold"] * data["Price"]

    result = (
        data.groupby("Product ID")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    return result


def product_units_sold(df):
    """
    Calculate total units sold for each product.
    """

    result = (
        df.groupby("Product ID")["Units Sold"]
        .sum()
        .sort_values(ascending=False)
    )

    return result


def product_demand(df):
    """
    Calculate total demand forecast for each product.
    """

    result = (
        df.groupby("Product ID")["Demand Forecast"]
        .sum()
        .sort_values(ascending=False)
    )

    return result


def product_performance(df):
    """
    Create a product-level performance summary.
    """

    data = df.copy()

    data["Sales"] = data["Units Sold"] * data["Price"]

    result = (
        data.groupby("Product ID")
        .agg(
            Total_Sales=("Sales", "sum"),
            Total_Units_Sold=("Units Sold", "sum"),
            Average_Inventory=("Inventory Level", "mean"),
            Total_Demand_Forecast=("Demand Forecast", "sum")
        )
        .sort_values("Total_Sales", ascending=False)
    )

    return result