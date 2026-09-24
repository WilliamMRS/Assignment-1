from pathlib import Path

import pandas as pd

# Use raw data until we choose our final imputation and outlier treatments.
data_path = Path(__file__).resolve().parent / "paraguay_energy" / "electricity-consumption-raw.csv"
data = pd.read_csv(data_path, parse_dates=["datetime"])
data = data.sort_values(["datetime", "feeder"]).reset_index(drop=True)

# 1. One-hot encoding: one 0/1 column per feeder, with no artificial ranking.
# Substation is omitted because each feeder already identifies its substation.
feeder_columns = pd.get_dummies(data["feeder"], prefix="feeder", dtype=int)

# 2. Reserve the last 20% of timestamps for later evaluation (Task 5).
# All feeders at a given timestamp stay on the same side of the boundary.
timestamps = data["datetime"].drop_duplicates().sort_values()
cutoff = timestamps.iloc[int(len(timestamps) * 0.8)]
training_rows = data["datetime"] < cutoff

# Fit Z-score parameters to observed TRAINING readings only.
# ddof=0 uses population standard deviation, as standard scalers do.
training_consumption = data.loc[training_rows, "consumption"]
mean = training_consumption.mean()
std = training_consumption.std(ddof=0)
if pd.isna(std) or std == 0:
    raise ValueError("Cannot standardize consumption without nonzero variation.")

# Apply the same parameters to all rows. Missing values stay missing.
# If predicting consumption, this is target scaling, not input-feature scaling.
consumption_z = (data["consumption"] - mean) / std
transformed = pd.concat([
    data[["datetime"]],  # Retain timestamps as metadata, not numeric features.
    consumption_z.rename("consumption_z"),
    feeder_columns,
], axis=1)

print(f"One-hot columns: {len(feeder_columns.columns)}")
print(f"Training timestamps precede: {cutoff}")
print(f"Training consumption mean: {mean:.4f}")
print(f"Training consumption standard deviation: {std:.4f}")
print("\nEncoding example:")
example = data["feeder"].isin(["A1", "A2", "I1"])
example = data.loc[example].drop_duplicates("feeder").index
print(pd.concat([data.loc[example, ["feeder"]],
                 feeder_columns.loc[example, ["feeder_A1", "feeder_A2", "feeder_I1"]]],
                axis=1).to_string(index=False))
print("\nConsumption before and after scaling:")
print(pd.DataFrame({"consumption": data["consumption"],
                    "consumption_z": consumption_z}).dropna().head(10).to_string(index=False))
print(f"\nScaled training mean: {consumption_z[training_rows].mean():.6f}")
print(f"Scaled training standard deviation: {consumption_z[training_rows].std(ddof=0):.6f}")
print(f"Missing consumption values retained: {consumption_z.isna().sum()}")

# 'transformed' is the resulting DataFrame. Original 'data' is unchanged.
# Keep the binary columns as 0/1; do not Z-score them.
