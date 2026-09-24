from pathlib import Path

import pandas as pd

# Start with original readings so imputed values do not affect our thresholds.
data_path = Path(__file__).resolve().parent / "paraguay_energy" / "electricity-consumption-raw.csv"
data = pd.read_csv(data_path, parse_dates=["datetime"])

# Examine I1 separately because feeders can have different consumption ranges.
feeder = data[data["feeder"] == "I1"].sort_values("datetime")
consumption = feeder["consumption"].dropna()

print("Summary of observed consumption for I1:")
print(consumption.describe())

# IQR measures the spread of the middle 50% of observed readings.
q1 = consumption.quantile(0.25)
q3 = consumption.quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Flag readings strictly outside the bounds; missing values are not outliers.
outlier_mask = feeder["consumption"].notna() & (
    (feeder["consumption"] < lower_bound)
    | (feeder["consumption"] > upper_bound)
)
outliers = feeder.loc[outlier_mask, ["datetime", "consumption"]]

print(f"\nQ1: {q1}, Q3: {q3}, IQR: {iqr}")
print(f"Lower bound: {lower_bound}, upper bound: {upper_bound}")
print(f"Below lower bound: {(consumption < lower_bound).sum()}")
print(f"Above upper bound: {(consumption > upper_bound).sum()}")
print(f"Potential outliers: {len(outliers)} of {len(consumption)} observed readings "
      f"({100 * len(outliers) / len(consumption):.2f}%)")
print("\nFirst 10 flagged readings in time order:")
print(outliers.head(10).to_string(index=False))
print("\nFive largest flagged readings:")
print(outliers.nlargest(5, "consumption").to_string(index=False))

# Detection only: no readings have been removed, capped, or transformed.
