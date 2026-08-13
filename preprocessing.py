import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from load_data import load_data


# =====================================================
# Numeric columns used for preprocessing
# =====================================================

NUMERIC_COLUMNS = [
    "CGPA",
    "AttendancePercent",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "CodingTestScore",
    "MockInterviewScore",
    "Salary Package",
    "Projects"
]


# =====================================================
# Task 1 : Outlier Fix
# =====================================================

def fix_outliers(data):

    data = data.copy()

    numeric_cols = [
        col for col in NUMERIC_COLUMNS
        if col in data.columns
    ]

    outlier_counts = {}

    for col in numeric_cols:

        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)

        IQR = Q3 - Q1

        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        outliers = (
            (data[col] < lower_limit) |
            (data[col] > upper_limit)
        )

        outlier_counts[col] = int(outliers.sum())

        # Fix outliers using clipping
        data[col] = data[col].clip(
            lower=lower_limit,
            upper=upper_limit
        )

    return data, outlier_counts


# =====================================================
# Task 2 : Min-Max Scaling
# =====================================================

def minmax_scaling(data):

    data = data.copy()

    numeric_cols = [
        col for col in NUMERIC_COLUMNS
        if col in data.columns
    ]

    scaler = MinMaxScaler()

    data[numeric_cols] = scaler.fit_transform(
        data[numeric_cols]
    )

    return data


# =====================================================
# Task 3 : Standard Scaling
# =====================================================

def standard_scaling(data):

    data = data.copy()

    numeric_cols = [
        col for col in NUMERIC_COLUMNS
        if col in data.columns
    ]

    scaler = StandardScaler()

    data[numeric_cols] = scaler.fit_transform(
        data[numeric_cols]
    )

    return data


# =====================================================
# Main Preprocessing Function
# =====================================================

def run_preprocessing():

    data = load_data()

    # -------------------------------------------------
    # Original data
    # -------------------------------------------------

    original_rows = len(data)
    original_columns = len(data.columns)

    # -------------------------------------------------
    # Step 1 : Outlier Fix
    # -------------------------------------------------

    outlier_fixed_data, outlier_counts = fix_outliers(data)

    # -------------------------------------------------
    # Step 2 : Min-Max Scaling
    # -------------------------------------------------

    minmax_data = minmax_scaling(outlier_fixed_data)

    # -------------------------------------------------
    # Step 3 : Standard Scaling
    # -------------------------------------------------

    standard_data = standard_scaling(outlier_fixed_data)

    # -------------------------------------------------
    # Preview
    # -------------------------------------------------

    preview_columns = [
        col for col in NUMERIC_COLUMNS
        if col in data.columns
    ]

    return {
        "n_rows": original_rows,
        "n_cols": original_columns,

        "outlier_counts": outlier_counts,

        "original_preview":
            data[preview_columns].head(10).to_dict(
                orient="records"
            ),

        "minmax_preview":
            minmax_data[preview_columns].head(10).to_dict(
                orient="records"
            ),

        "standard_preview":
            standard_data[preview_columns].head(10).to_dict(
                orient="records"
            )
    }