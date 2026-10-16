import pandas as pd


def prepare_daily_demand(df, store_id=None, product_id=None):
    """
    Prepare daily demand data for forecasting.
    """

    data = df.copy()

    data["Date"] = pd.to_datetime(data["Date"])

    # Filter by store if provided
    if store_id is not None:
        data = data[data["Store ID"] == store_id]

    # Filter by product if provided
    if product_id is not None:
        data = data[data["Product ID"] == product_id]

    daily_demand = (
        data.groupby("Date")["Units Sold"]
        .sum()
        .reset_index()
        .sort_values("Date")
    )

    return daily_demand


def moving_average_forecast(
    df,
    store_id=None,
    product_id=None,
    window=7,
    forecast_days=7
):
    """
    Forecast future demand using a moving average.

    The forecast is based on the average demand
    from the most recent days.
    """

    daily_demand = prepare_daily_demand(
        df,
        store_id=store_id,
        product_id=product_id
    )

    if daily_demand.empty:
        return pd.DataFrame(
            columns=["Date", "Forecast_Demand"]
        )

    recent_demand = daily_demand["Units Sold"].tail(window)

    average_demand = recent_demand.mean()

    last_date = daily_demand["Date"].max()

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D"
    )

    forecast = pd.DataFrame({
        "Date": future_dates,
        "Forecast_Demand": average_demand
    })

    return forecast

def evaluate_moving_average(
    df,
    store_id=None,
    product_id=None,
    window=7
):
    """
    Evaluate the moving average forecasting method
    using historical data.

    MAE = Mean Absolute Error
    Lower MAE means better forecasting accuracy.
    """

    daily_demand = prepare_daily_demand(
        df,
        store_id=store_id,
        product_id=product_id
    )

    if len(daily_demand) <= window:
        return None

    actual_values = []
    predicted_values = []

    for i in range(window, len(daily_demand)):

        previous_values = daily_demand[
            "Units Sold"
        ].iloc[i - window:i]

        predicted = previous_values.mean()

        actual = daily_demand[
            "Units Sold"
        ].iloc[i]

        predicted_values.append(predicted)
        actual_values.append(actual)

    errors = [
        abs(actual - predicted)
        for actual, predicted
        in zip(actual_values, predicted_values)
    ]

    mae = sum(errors) / len(errors)

    return float(mae)

def forecast_summary(
    df,
    store_id=None,
    product_id=None,
    window=7,
    forecast_days=7
):
    """
    Generate a simple forecast summary.
    """

    daily_demand = prepare_daily_demand(
        df,
        store_id=store_id,
        product_id=product_id
    )

    forecast = moving_average_forecast(
        df,
        store_id=store_id,
        product_id=product_id,
        window=window,
        forecast_days=forecast_days
    )

    if daily_demand.empty:
        return {
            "status": "No data found",
            "forecast": forecast
        }

    average_historical_demand = (
        daily_demand["Units Sold"].mean()
    )

    average_forecast_demand = (
        forecast["Forecast_Demand"].mean()
    )

    return {
        "status": "success",
        "historical_average_demand":
            float(average_historical_demand),
        "forecast_average_demand":
            float(average_forecast_demand),
        "forecast_days": forecast_days,
        "forecast": forecast
    }