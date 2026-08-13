import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from load_data import load_data

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "static", "charts")


def _chart_path(filename):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)

#
def run_eda():

    data = load_data()
    charts = []
    sns.set_style("whitegrid")

    # =====================================================
    # Task 1 & 2 : Dataset Summary
    # =====================================================

    n_rows = len(data)
    n_cols = len(data.columns)

    # =====================================================
    # Task 3 : Missing Values
    # =====================================================

    missing = data.isnull().sum()
    missing = missing[missing > 0]

    if not missing.empty:

        plt.figure(figsize=(10,5))
        sns.barplot(x=missing.index, y=missing.values)

        plt.xticks(rotation=45)
        plt.title("Missing Values")

        plt.tight_layout()
        plt.savefig(_chart_path("missing_values.png"))
        plt.close()

        charts.append("missing_values.png")

        plt.figure(figsize=(12,6))
        sns.heatmap(data.isnull(), cbar=False)

        plt.title("Missing Value Heatmap")

        plt.tight_layout()
        plt.savefig(_chart_path("missing_heatmap.png"))
        plt.close()

        charts.append("missing_heatmap.png")

    # =====================================================
    # Task 4 : Duplicate Rows
    # =====================================================

    duplicates = int(data.duplicated().sum())

    # =====================================================
    # Task 5 : Placement Status
    # =====================================================

    target_counts = data["PlacementStatus"].value_counts().to_dict()
    plt.figure(figsize=(6,5))
    sns.countplot(
        x="PlacementStatus",
        data=data
    )

    plt.title("Placement Status Distribution")
    plt.tight_layout()
    plt.savefig(_chart_path("placement_status.png"))
    plt.close()

    charts.append("placement_status.png")

    # =====================================================
    # Task 6 : Numeric Feature Distribution
    # =====================================================

    hist_cols = [
        "CGPA",
        "AttendancePercent",
        "AptitudeTestScore",
        "SoftSkillsRating",
        "CodingTestScore",
        "MockInterviewScore",
        "Salary Package",
        "Projects"
    ]

    hist_cols = [c for c in hist_cols if c in data.columns]

    if hist_cols:
        data[hist_cols].hist(figsize=(14,10), bins=20)
        plt.tight_layout()
        plt.savefig(_chart_path("numeric_distribution.png"))
        plt.close()
        charts.append("numeric_distribution.png")

    if "CGPA" in data.columns:
        plt.figure(figsize=(7,5))
        sns.histplot(data["CGPA"], kde=True)

        plt.axvline(
            data["CGPA"].mean(),
            color="green",
            linestyle="--",
            label="Mean"
        )

        plt.legend()
        plt.title("CGPA Distribution")
        plt.tight_layout()
        plt.savefig(_chart_path("cgpa_distribution.png"))
        plt.close()
        charts.append("cgpa_distribution.png")

    # =====================================================
    # Task 7 : Boxplots
    # =====================================================

    box_cols = [
        "CGPA",
        "AttendancePercent",
        "AptitudeTestScore",
        "SoftSkillsRating",
        "CodingTestScore",
        "MockInterviewScore",
        "Salary Package",
        "Projects"
    ]

    box_cols = [c for c in box_cols if c in data.columns]

    for col in box_cols:

        plt.figure(figsize=(9,4))
        sns.boxplot(x=data[col], color="red")
        plt.title(f"Boxplot - {col}")
        plt.tight_layout()
        filename = f"boxplot_{col}.png"
        plt.savefig(_chart_path(filename))
        plt.close()
        charts.append(filename)

    # =====================================================
    # Task 8 : Correlation
    # =====================================================

    corr = data.select_dtypes(include="number").corr()

    plt.figure(figsize=(16,12))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(_chart_path("correlation_heatmap.png"))
    plt.close()
    charts.append("correlation_heatmap.png")

    # ---------- TASK 9 STARTS HERE ----------

    # =====================================================
    # Task 9 : Relationship Plots
    # =====================================================

    if "CGPA" in data.columns and "Salary Package" in data.columns:

        plt.figure(figsize=(10,6))

        sns.regplot(
            x="CGPA",
            y="Salary Package",
            data=data,
            color="red"
        )

        plt.title("CGPA vs Salary Package")
        plt.tight_layout()
        plt.savefig(_chart_path("cgpa_salary.png"))
        plt.close()
        charts.append("cgpa_salary.png")


    if "AptitudeTestScore" in data.columns and "CodingTestScore" in data.columns:

        plt.figure(figsize=(10,6))

        sns.regplot(
            x="AptitudeTestScore",
            y="CodingTestScore",
            data=data,
            scatter_kws={"alpha":0.6},
            line_kws={"color":"red"}
        )

        plt.title("Aptitude vs Coding Test Score")
        plt.tight_layout()
        plt.savefig(_chart_path("aptitude_coding.png"))
        plt.close()
        charts.append("aptitude_coding.png")


    # =====================================================
    # Task 10 : Categorical Count Plots
    # =====================================================

    cat_cols = [
        "Gender",
        "City",
        "CollegeTier",
        "Stream",
        "Specialisation",
        "Hostel",
        "HistoryOfBacklogs",
        "CGPA_Tier"
    ]

    cat_cols = [c for c in cat_cols if c in data.columns]

    for col in cat_cols:

        plt.figure(figsize=(10,5))

        order = data[col].value_counts().index

        sns.countplot(
            x=col,
            data=data,
            order=order
        )

        plt.xticks(rotation=45)
        plt.title(col)
        plt.tight_layout()
        filename = f"{col}_count.png"
        plt.savefig(_chart_path(filename))
        plt.close()
        charts.append(filename)


    # =====================================================
    # Task 11 : Gender vs Placement
    # =====================================================

    if "Gender" in data.columns:

        plt.figure(figsize=(7,5))

        sns.countplot(
            x="Gender",
            hue="PlacementStatus",
            data=data
        )

        plt.title("Gender vs Placement Status")
        plt.tight_layout()
        plt.savefig(_chart_path("gender_placement.png"))
        plt.close()
        charts.append("gender_placement.png")


    # =====================================================
    # Task 12 : College Tier & Stream
    # =====================================================

    if "CollegeTier" in data.columns:

        plt.figure(figsize=(8,5))

        sns.countplot(
            x="CollegeTier",
            hue="PlacementStatus",
            data=data
        )

        plt.title("College Tier vs Placement")
        plt.tight_layout()
        plt.savefig(_chart_path("college_tier_placement.png"))
        plt.close()
        charts.append("college_tier_placement.png")


    if "Stream" in data.columns:

        plt.figure(figsize=(12,5))

        sns.countplot(
            x="Stream",
            hue="PlacementStatus",
            data=data
        )

        plt.xticks(rotation=45)
        plt.title("Stream vs Placement")
        plt.tight_layout()
        plt.savefig(_chart_path("stream_placement.png"))
        plt.close()
        charts.append("stream_placement.png")


    # =====================================================
    # Task 13 : SGPA Trend
    # =====================================================

    sgpa_cols = [f"SGPA_Sem{i}" for i in range(1,9)]
    sgpa_cols = [c for c in sgpa_cols if c in data.columns]

    if sgpa_cols:

        avg_sgpa = data[sgpa_cols].mean()

        plt.figure(figsize=(8,5))

        plt.plot(
            avg_sgpa.index,
            avg_sgpa.values,
            marker="o"
        )

        plt.title("Average SGPA Across Semesters")
        plt.xlabel("Semester")
        plt.ylabel("Average SGPA")
        plt.tight_layout()
        plt.savefig(_chart_path("sgpa_trend.png"))
        plt.close()
        charts.append("sgpa_trend.png")


    # =====================================================
    # Task 14 : Salary Analysis
    # =====================================================

    if "Salary Package" in data.columns:

        placed = data[data["PlacementStatus"] == 1]

        plt.figure(figsize=(8,5))

        sns.histplot(
            placed["Salary Package"],
            kde=True
        )

        plt.title("Salary Distribution")
        plt.tight_layout()
        plt.savefig(_chart_path("salary_distribution.png"))
        plt.close()
        charts.append("salary_distribution.png")


        if "CollegeTier" in data.columns:

            plt.figure(figsize=(8,5))

            sns.boxplot(
                x="CollegeTier",
                y="Salary Package",
                data=placed
            )
            plt.title("Salary by College Tier")
            plt.tight_layout()
            plt.savefig(_chart_path("salary_college_tier.png"))
            plt.close()
            charts.append("salary_college_tier.png")


    # =====================================================
    # Task 15 : Pairplot
    # =====================================================

    pair_cols = [
        "CGPA",
        "AptitudeTestScore",
        "CodingTestScore",
        "MockInterviewScore",
        "PlacementStatus"
    ]

    pair_cols = [c for c in pair_cols if c in data.columns]
    if len(pair_cols) == 5:

        sample = data[pair_cols].sample(
            n=min(1000, len(data)),
            random_state=42
        )
        g = sns.pairplot(
            sample,
            hue="PlacementStatus",
            diag_kind="hist",
            plot_kws={"s":10,"alpha":0.6}
        )
        g.savefig(_chart_path("pairplot.png"))
        plt.close("all")
        charts.append("pairplot.png")
    # =====================================================
    # Return
    # =====================================================

    return {
        "n_rows": len(data),
        "n_cols": len(data.columns),
        "duplicate_count": duplicates,
        "missing": missing.to_dict(),
        "target_counts": target_counts,
        "charts": charts

    }