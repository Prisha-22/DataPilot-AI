import pandas as pd


def clean_dataset(df):
    cleaned_df = df.copy()

    cleaning_log = []

    # Convert Date to datetime
    if "Date" in cleaned_df.columns:
        cleaned_df["Date"] = pd.to_datetime(
            cleaned_df["Date"],
            errors="coerce"
        )

    # Correct negative demand forecasts
    if "Demand Forecast" in cleaned_df.columns:
        negative_count = int(
            (cleaned_df["Demand Forecast"] < 0).sum()
        )

        if negative_count > 0:
            cleaned_df.loc[
                cleaned_df["Demand Forecast"] < 0,
                "Demand Forecast"
            ] = 0

            cleaning_log.append({
                "column": "Demand Forecast",
                "issue": "Negative values",
                "rows_affected": negative_count,
                "action": "Replaced negative values with 0",
                "reason": "Demand cannot be negative"
            })

    return cleaned_df, cleaning_log