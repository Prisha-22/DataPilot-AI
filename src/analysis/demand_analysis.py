import pandas as pd


def calculate_inventory_coverage(df, recent_days=30):
    """
    Calculate inventory coverage using recent demand.

    Inventory coverage tells us approximately how many days
    the current inventory can support based on recent demand.
    """

    data = df.copy()

    data["Date"] = pd.to_datetime(data["Date"])

    # Find the latest date in the dataset
    latest_date = data["Date"].max()

    # Keep only recent data
    recent_data = data[
        data["Date"] >= latest_date - pd.Timedelta(days=recent_days - 1)
    ].copy()

    # Calculate average daily demand for each store-product pair
    demand = (
        recent_data.groupby(["Store ID", "Product ID"])
        .agg(
            Average_Daily_Demand=("Units Sold", "mean"),
            Current_Inventory=("Inventory Level", "last")
        )
        .reset_index()
    )

    # Avoid division by zero
    demand["Inventory_Coverage_Days"] = demand.apply(
        lambda row:
        row["Current_Inventory"] / row["Average_Daily_Demand"]
        if row["Average_Daily_Demand"] > 0
        else float("inf"),
        axis=1
    )

    return demand


def classify_inventory_risk(df, recent_days=30):
    """
    Classify inventory risk based on inventory coverage days.
    """

    result = calculate_inventory_coverage(
        df,
        recent_days=recent_days
    )

    result["Risk"] = "Normal"

    result.loc[
        result["Inventory_Coverage_Days"] < 3,
        "Risk"
    ] = "High Stock-Out Risk"

    result.loc[
        (result["Inventory_Coverage_Days"] >= 3) &
        (result["Inventory_Coverage_Days"] < 7),
        "Risk"
    ] = "Medium Stock-Out Risk"

    result.loc[
        result["Inventory_Coverage_Days"] >= 30,
        "Risk"
    ] = "Potential Overstock"

    return result


def high_risk_inventory(df, recent_days=30):
    """
    Return only store-product combinations with high
    stock-out risk.
    """

    result = classify_inventory_risk(
        df,
        recent_days=recent_days
    )

    return result[
        result["Risk"] == "High Stock-Out Risk"
    ].sort_values(
        "Inventory_Coverage_Days"
    )