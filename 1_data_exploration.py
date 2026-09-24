from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sd_data = pd.read_csv("smoking_drinking.csv", header=0, sep=",")

# 1a
# First rows
print("\nFirst rows:")
print(sd_data.head())
print("\n\n\n")

# Summary statistics
print("\nSummary statistics:")
print(sd_data.describe())
print("\n\n\n")

# Data types
print("\nData types:")
print(sd_data.dtypes)
print("\n\n\n")


# 1b
# Missing values
print("\nMissing values:")
print(sd_data.isnull().sum())
print("\n\n\n")

# Unique values
print("\nUnique values:")
categorical_columns = [
    "sex",
    "DRK_YN",
    "SMK_stat_type_cd",
    "hear_left",
    "hear_right",
    "urine_protein",
]

print("\nCategorical values and frequencies:")
for column in categorical_columns:
    print(f"\n{column}:")
    print(sd_data[column].value_counts(dropna=False).sort_index())

# Duplicate records (also recommended in lecture slide 57)
print(f"\nDuplicate rows: {sd_data.duplicated().sum()}")

# IQR screening: numerical category codes are not continuous measurements.
numerical_columns = sd_data.select_dtypes(include="number").columns
numerical_columns = numerical_columns.drop(categorical_columns, errors="ignore")

outlier_results = []
for column in numerical_columns:
    values = sd_data[column].dropna()
    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    mask = (values < lower) | (values > upper)
    count = int(mask.sum())
    outlier_results.append({
        "column": column,
        "lower_bound": lower,
        "upper_bound": upper,
        "outlier_count": count,
        "outlier_percent": 100 * count / len(values) if len(values) else float("nan"),
        "minimum": values.min(),
        "median": values.median(),
        "99th_percentile": values.quantile(0.99),
        "maximum": values.max(),
    })

outlier_summary = pd.DataFrame(outlier_results).set_index("column")
print("\nPotential outliers using 1.5 × IQR:")
print(outlier_summary.round(2).to_string())
print("Counts are per column; the same row may be flagged in several columns.")

# Save results for the report. Nothing is removed or replaced in sd_data.
output_dir = Path("output/exploration")
output_dir.mkdir(parents=True, exist_ok=True)
outlier_summary.to_csv(output_dir / "outlier_summary.csv")

# Inspect representative measurements, including the strongly skewed columns.
plot_columns = ["height", "SBP", "waistline", "triglyceride", "gamma_GTP", "SGOT_AST"]
fig, axes = plt.subplots(len(plot_columns), 2, figsize=(14, 18))
for row, column in enumerate(plot_columns):
    values = sd_data[column].dropna()
    lower = outlier_summary.loc[column, "lower_bound"]
    upper = outlier_summary.loc[column, "upper_bound"]

    axes[row, 0].hist(values, bins=80, color="steelblue", edgecolor="white")
    # Log counts make rare extreme values visible; measurement values stay unchanged.
    axes[row, 0].set_yscale("log")
    axes[row, 0].set_title(f"{column}: histogram")
    axes[row, 0].set_xlabel(column)
    axes[row, 0].set_ylabel("Count (log scale)")

    axes[row, 1].boxplot(values, orientation="horizontal", whis=1.5,
                         flierprops={"markersize": 2, "alpha": 0.3})
    axes[row, 1].set_title(f"{column}: boxplot")
    axes[row, 1].set_xlabel(column)
    axes[row, 1].set_yticks([])
    axes[row, 1].axvline(lower, color="orange", linestyle="--", label="IQR boundaries")
    axes[row, 1].axvline(upper, color="orange", linestyle="--")
    axes[row, 1].legend()

fig.tight_layout()
fig.savefig(output_dir / "outlier_distributions.png", dpi=150)
plt.close(fig)
print(f"\nSummary and plots saved in: {output_dir.resolve()}")

# IQR flags unusual values, not confirmed errors. Check the dataset documentation
# for special codes (e.g. 999) before treating them as missing or invalid.
# Inspect category frequencies above for inconsistent labels or unexpected codes.
# Removal, capping, or transformation requires a separate decision in task 3.
