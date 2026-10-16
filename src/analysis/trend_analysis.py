import pandas as pd


def calculate_daily_sales(df):
    """
    Calculate total sales for each day.
    """

    data = df.copy()

    data["Date"] = pd.to_datetime(data["Date"])

    data["Sales"] = (
        data["Units Sold"] * data["Price"]
    )

    daily_sales = (
        data.groupby("Date")["Sales"]
        .sum()
        .reset_index()
        .sort_values("Date")
    )

    return daily_sales


def calculate_sales_trend(df):
    """
    Compare recent sales with previous-period sales.

    Uses two equal periods at the end of the dataset.
    """

    daily_sales = calculate_daily_sales(df)

    if len(daily_sales) < 14:
        return None

    # Use the last 7 days and the 7 days before them
    recent_period = daily_sales.tail(7)
    previous_period = daily_sales.iloc[-14:-7]

    recent_average = recent_period["Sales"].mean()
    previous_average = previous_period["Sales"].mean()

    if previous_average == 0:
        percentage_change = 0
    else:
        percentage_change = (
            (recent_average - previous_average)
            / previous_average
        ) * 100

    if percentage_change > 5:
        trend = "Increasing"
    elif percentage_change < -5:
        trend = "Decreasing"
    else:
        trend = "Stable"

    return {
        "recent_average_sales": float(recent_average),
        "previous_average_sales": float(previous_average),
        "percentage_change": float(percentage_change),
        "trend": trend
    }