"""Compare the supplied script's linear and weekly filling steps on raw data.

Both methods use the same hourly grid and preserve observed consumption.
Outlier processing and whole-day masking are excluded from this comparison.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "output" / "imputation_comparison"
OUTPUT.mkdir(parents=True, exist_ok=True)
raw = pd.read_csv(BASE / "paraguay_energy/electricity-consumption-raw.csv",
                  parse_dates=["datetime"])

# One column per feeder, with all hours present so 168 rows equals one week.
assert not raw.duplicated(["datetime", "substation", "feeder"]).any()
hours = pd.date_range(raw.datetime.min(), raw.datetime.max(), freq="h")
original = raw.pivot(index="datetime", columns=["substation", "feeder"],
                     values="consumption").reindex(hours)
original.index.name = "datetime"

# Method 1: the supplied script interpolates forward, then backward.
# Inside gaps this is linear interpolation; at the ends it carries the nearest
# available value outward, which is an additional assumption, not interpolation.
linear = original.interpolate(method="linear", limit_direction="both")

# Method 2: supplied weekly averaging, corrected to average donor columns only.
# Use original observations at the same hour/weekday up to six weeks each way.
weekly = original.copy()
for key in original.columns:
    donors = pd.concat([original[key].shift(168 * step)
                        for step in range(-6, 7) if step != 0], axis=1)
    weekly[key] = original[key].fillna(donors.mean(axis=1))

# Export both estimates alongside the original values for every feeder/hour.
result = original.stack(["substation", "feeder"]).rename("consumption_raw").to_frame()
result["consumption_linear"] = linear.stack(["substation", "feeder"])
result["consumption_weekly"] = weekly.stack(["substation", "feeder"])
result.reset_index().to_csv(OUTPUT / "electricity-consumption-comparison.csv", index=False)
assert len(result) == original.size
assert linear.where(original.notna()).equals(original)
assert weekly.where(original.notna()).equals(original)

# Compare the same I1 week in two panels with identical axes.
key = ("I", "I1")
observed = original.loc["2017-01-01":"2017-01-07", key]
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, sharey=True)
for ax, name, estimates in zip(axes, ["Linear interpolation", "Weekly averaging"],
                               [linear, weekly]):
    values = estimates.loc[observed.index, key]
    imputed = observed.isna() & values.notna()
    ax.plot(values.index, values, color="steelblue", linewidth=0.8,
            label="After filling")
    ax.plot(observed.index, observed, color="black", marker=".",
            linewidth=0.5, markersize=3, label="Original observations")
    ax.scatter(values.index[imputed], values[imputed], color="darkorange",
               s=24, zorder=3, label="Imputed values")
    ax.set_title(f"I1: {name}")
    ax.set_ylabel("Consumption (dataset units)")
    ax.legend(loc="upper left")
    print(f"{name}: {imputed.sum()} filled, {values.isna().sum()} still missing in I1 week")
axes[-1].set_xlabel("Datetime (January 1–7, 2017)")
fig.tight_layout()
fig.savefig(OUTPUT / "I1_linear_vs_weekly.png", dpi=150)
print(f"\nFull-grid missing before: {int(original.isna().sum().sum()):,}")
print(f"Missing after linear filling: {int(linear.isna().sum().sum()):,}")
print(f"Missing after weekly filling: {int(weekly.isna().sum().sum()):,}")
print(f"Saved dataset and plot to {OUTPUT}")
plt.show()
