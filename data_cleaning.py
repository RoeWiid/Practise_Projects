"""
Assignment 1: Data Cleaning

Purpose:
    Clean a deliberately messy customer dataset and produce a validated,
    analysis-ready CSV.

"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "data" / "customer_data_raw.csv"
CLEAN_PATH = BASE_DIR / "data" / "customer_data_clean.csv"


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean and validate the raw customer data."""
    report = {
        "rows_before": len(df),
        "duplicate_rows_before": int(df.duplicated().sum()),
    }

    # Standardise column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    # Strip whitespace from all text columns
    text_columns = df.select_dtypes(include=["object", "str"]).columns
    for column in text_columns:
        df[column] = df[column].str.strip()

    # Standardise names and cities
    df["name"] = df["name"].str.replace(r"\s+", " ", regex=True).str.title()
    df["city"] = df["city"].str.replace(r"\s+", " ", regex=True).str.title()

    # Standardise gender values
    gender_map = {
        "m": "Male",
        "male": "Male",
        "f": "Female",
        "female": "Female",
    }
    df["gender"] = (
        df["gender"]
        .str.lower()
        .map(gender_map)
    )

    # Convert age to numeric and flag impossible ages as missing
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    invalid_age_count = int((~df["age"].between(18, 100)).sum())
    df.loc[~df["age"].between(18, 100), "age"] = np.nan
    report["invalid_age_values"] = invalid_age_count

    # Convert income to numeric
    df["income"] = pd.to_numeric(df["income"], errors="coerce")

    # Negative income is treated as an invalid value
    invalid_income_count = int((df["income"] < 0).sum())
    df.loc[df["income"] < 0, "income"] = np.nan
    report["invalid_income_values"] = invalid_income_count

    # Convert dates; unparseable dates become NaT
    df["signup_date"] = pd.to_datetime(
        df["signup_date"],
        errors="coerce",
        dayfirst=True,
        format="mixed",
    )

    report["invalid_dates"] = int(df["signup_date"].isna().sum())

    # Remove exact duplicate rows
    df = df.drop_duplicates()
    report["duplicate_rows_removed"] = report["rows_before"] - len(df) - 0

    # If the same customer appears more than once, keep the most recent
    # valid signup record. This is a business rule and should be documented.
    df = df.sort_values(["customer_id", "signup_date"])
    duplicate_customer_ids = int(df["customer_id"].duplicated().sum())
    df = df.drop_duplicates(subset="customer_id", keep="last")
    report["duplicate_customer_ids_resolved"] = duplicate_customer_ids

    # A signup date is required for this dataset's analysis.
    df = df.dropna(subset=["customer_id", "signup_date"])

    # Impute numeric missing values using the median.
    # Median is robust to potential outliers.
    df["age"] = df["age"].fillna(df["age"].median())
    df["income"] = df["income"].fillna(df["income"].median())

    # Impute missing categorical values using the mode.
    df["gender"] = df["gender"].fillna(df["gender"].mode().iloc[0])
    df["city"] = df["city"].fillna(df["city"].mode().iloc[0])

    # Clean data types
    df["customer_id"] = pd.to_numeric(df["customer_id"], 
                                      errors="raise").astype(int)
    df["age"] = df["age"].round().astype(int)
    df["income"] = df["income"].round(2)

    # Consistent ordering
    df = df.sort_values("customer_id").reset_index(drop=True)

    # Final validation checks
    assert not df["customer_id"].duplicated().any(),"Duplicate customer IDs remain."
    assert df["age"].between(18, 100).all(), "Invalid ages remain."
    assert (df["income"] >= 0).all(), "Negative income remains."
    assert df["signup_date"].notna().all(), "Invalid dates remain."
    assert df["gender"].isin(["Male", "Female"]).all(), "Unexpected gender values remain."

    report["rows_after"] = len(df)
    report["missing_values_after"] = int(df.isna().sum().sum())

    return df, report


def main() -> None:
    df_raw = pd.read_csv(RAW_PATH)

    print("=" * 60)
    print("ASSIGNMENT 1 — DATA CLEANING")
    print("=" * 60)

    print("\nRaw dataset shape:", df_raw.shape)
    print("\nRaw missing values:")
    print(df_raw.isna().sum())
    print("\nRaw duplicate rows:", df_raw.duplicated().sum())

    cleaned_df, report = clean_data(df_raw)

    cleaned_df.to_csv(CLEAN_PATH, index=False)

    print("\nCleaning report:")
    for key, value in report.items():
        print(f"  {key}: {value}")

    print("\nCleaned dataset:")
    print(cleaned_df.to_string(index=False))

    print("\nFinal data types:")
    print(cleaned_df.dtypes)

    print("\nFinal missing values:")
    print(cleaned_df.isna().sum())

    print(f"\nSaved cleaned dataset to: {CLEAN_PATH}")


if __name__ == "__main__":
    main()
