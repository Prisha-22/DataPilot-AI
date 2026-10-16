import pandas as pd


def check_data_quality(df):
    quality_report = {
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "negative_values": {},
        "invalid_values": {}
    }

    # Check negative values in numeric columns
    numeric_columns = df.select_dtypes(include="number").columns

    for column in numeric_columns:
        negative_count = int((df[column] < 0).sum())

        if negative_count > 0:
            quality_report["negative_values"][column] = negative_count

    # Retail-specific validation rules

    if "Price" in df.columns:
        invalid_price = int((df["Price"] <= 0).sum())

        if invalid_price > 0:
            quality_report["invalid_values"]["Price"] = invalid_price

    if "Discount" in df.columns:
        invalid_discount = int(
            ((df["Discount"] < 0) | (df["Discount"] > 100)).sum()
        )

        if invalid_discount > 0:
            quality_report["invalid_values"]["Discount"] = invalid_discount

    if "Holiday/Promotion" in df.columns:
        invalid_holiday = int(
            (~df["Holiday/Promotion"].isin([0, 1])).sum()
        )

        if invalid_holiday > 0:
            quality_report["invalid_values"]["Holiday/Promotion"] = invalid_holiday

    if "Inventory Level" in df.columns:
        invalid_inventory = int((df["Inventory Level"] < 0).sum())

        if invalid_inventory > 0:
            quality_report["invalid_values"]["Inventory Level"] = invalid_inventory

    if "Units Sold" in df.columns:
        invalid_units_sold = int((df["Units Sold"] < 0).sum())

        if invalid_units_sold > 0:
            quality_report["invalid_values"]["Units Sold"] = invalid_units_sold

    if "Units Ordered" in df.columns:
        invalid_units_ordered = int((df["Units Ordered"] < 0).sum())

        if invalid_units_ordered > 0:
            quality_report["invalid_values"]["Units Ordered"] = invalid_units_ordered

    return quality_report