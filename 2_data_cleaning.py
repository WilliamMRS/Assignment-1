from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# 1. Load the raw CSV. Parse datetime as dates instead of plain text.
data_path = Path(__file__).resolve().parent / "paraguay_energy" / "electricity-consumption-raw.csv"
data = pd.read_csv(data_path, parse_dates=["datetime"])

# 2. Plot one feeder so we do not mix different electricity consumption series.
feeder = data[data["feeder"] == "I1"].sort_values("datetime")
week = feeder[
    (feeder["datetime"] >= "2017-01-01")
    & (feeder["datetime"] < "2017-01-08")
]
plt.figure(figsize=(12, 4))
plt.plot(week["datetime"], week["consumption"], marker=".", linewidth=0.5)
plt.title("Raw electricity consumption: feeder I1")
plt.xlabel("Datetime")
plt.ylabel("Consumption (dataset units)")
plt.tight_layout()
plt.show()

# 3. Count missing cells in each column of the entire raw dataset.
print("Missing values per column:")
print(data.isnull().sum())

missing_by_feeder = data.groupby("feeder")["consumption"].agg(
    total_rows="size",
    available_values="count",
)

missing_by_feeder["missing_values"] = (
    missing_by_feeder["total_rows"] - missing_by_feeder["available_values"]
)

print(missing_by_feeder)

