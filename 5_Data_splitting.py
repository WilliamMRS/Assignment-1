from pathlib import Path
import runpy

# Run Task 4 to reuse its transformed data and training-only scaling parameters.
# runpy lets us load a script whose filename starts with a number.
task4 = runpy.run_path(str(Path(__file__).resolve().parent / "4_Data_transformation.py"))
transformed = task4["transformed"]
cutoff = task4["cutoff"]

# Split by time, not randomly: learn from the past and evaluate on the future.
# This is 80/20 of unique timestamps; row percentages can differ because
# some feeders have absent rows. Never put the same timestamp in both sets.
train_data = transformed.loc[transformed["datetime"] < cutoff].copy()
test_data = transformed.loc[transformed["datetime"] >= cutoff].copy()

assert len(train_data) + len(test_data) == len(transformed)
assert train_data["datetime"].max() < test_data["datetime"].min()
assert train_data.columns.equals(test_data.columns)

print("\nChronological train/test split:")
for name, subset in [("Training", train_data), ("Testing", test_data)]:
    print(f"{name}: {len(subset):,} rows ({100 * len(subset) / len(transformed):.2f}%)")
    print(f"  From {subset['datetime'].min()} to {subset['datetime'].max()}")
    print(f"  Missing consumption values: {subset['consumption_z'].isna().sum():,}")

# No model is fitted here. These two DataFrames are the split datasets.
# Task 4 still uses raw readings: final cleaning is not yet connected.
# Missing targets must not be used as evaluation ground truth. If cleaning
# is added, it must not use test observations to fill or transform training data.
