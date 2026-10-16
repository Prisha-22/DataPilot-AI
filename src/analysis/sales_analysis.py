import pandas as pd


def calculate_total_sales(df):
    """
    Calculate total sales value.
    """

    sales = df["Units Sold"] * df["Price"]

    return float(sales.sum())


def sales_by_category(df):
    """
    Calculate total sales for each product category.
    """

    data = df.copy()

    data["Sales"] = data["Units Sold"] * data["Price"]

    result = (
        data.groupby("Category")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    return result


def sales_by_region(df):
    """
    Calculate total sales for each region.
    """

    data = df.copy()

    data["Sales"] = data["Units Sold"] * data["Price"]

    result = (
        data.groupby("Region")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    return result


def sales_by_month(df):
    """
    Calculate total sales month-wise.
    """

    data = df.copy()

    data["Date"] = pd.to_datetime(data["Date"])

    data["Sales"] = data["Units Sold"] * data["Price"]

    data["Month"] = data["Date"].dt.to_period("M").astype(str)

    result = (
        data.groupby("Month")["Sales"]
        .sum()
        .sort_index()
    )

    return result