from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "output" / "pca_smoking"
OUTPUT.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(BASE / "smoking_drinking.csv")

# DRK_YN is the outcome we would predict, so it must not be used to construct
# PCA components. Coded categories are encoded before PCA so their numbers do
# not imply a false numerical order.
target = "DRK_YN"
categorical = [
    "sex",
    "hear_left",
    "hear_right",
    "urine_protein",
    "SMK_stat_type_cd",
]
continuous = [
    column for column in data.columns
    if column not in categorical + [target]
]
if data[continuous + categorical + [target]].isnull().any().any():
    raise ValueError("This PCA run expects the smoking/drinking data to be complete.")

encoded_categories = pd.get_dummies(data[categorical].astype(str),
                                    prefix=categorical, dtype=float)
features = pd.concat([data[continuous], encoded_categories], axis=1)

# A health-record table is treated as independent rows here, so use a fixed
# random 80/20 split. PCA parameters are learned from training rows only.
rng = np.random.default_rng(42)
train_mask = rng.random(len(data)) < 0.8
X_train = features.loc[train_mask].to_numpy(dtype=float)
X_test = features.loc[~train_mask].to_numpy(dtype=float)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0, ddof=0)
if np.any(std == 0):
    raise ValueError("A PCA input column has no variation in the training data.")
X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

# PCA via the covariance matrix: each component is an orthogonal combination
# of the standardized input columns, ordered by explained variance.
covariance = (X_train_scaled.T @ X_train_scaled) / (len(X_train_scaled) - 1)
eigenvalues, components = np.linalg.eigh(covariance)
order = np.argsort(eigenvalues)[::-1]
eigenvalues = np.maximum(eigenvalues[order], 0)
components = components[:, order].T
explained = eigenvalues / eigenvalues.sum()
cumulative = np.cumsum(explained)
components_95 = int(np.searchsorted(cumulative, 0.95) + 1)

# Produce a compact, inspectable two-component representation. The variance
# report records all components, including the number needed for 95% retention.
train_scores = X_train_scaled[:10_000] @ components[:2].T
test_scores = X_test_scaled[:10_000] @ components[:2].T
train_output = pd.DataFrame(train_scores, columns=["PC1", "PC2"])
test_output = pd.DataFrame(test_scores, columns=["PC1", "PC2"])
train_output.to_csv(OUTPUT / "train_pca_sample.csv", index=False)
test_output.to_csv(OUTPUT / "test_pca_sample.csv", index=False)

variance_report = pd.DataFrame({
    "component": [f"PC{i}" for i in range(1, len(explained) + 1)],
    "explained_variance_ratio": explained,
    "cumulative_explained_variance": cumulative,
})
variance_report.to_csv(OUTPUT / "variance_report.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(cumulative) + 1), cumulative, marker=".", linewidth=1)
ax.axhline(0.95, color="darkorange", linestyle="--", label="95% variance")
ax.axvline(components_95, color="darkorange", linestyle=":")
ax.set(xlabel="Number of principal components",
       ylabel="Cumulative explained variance",
       title="PCA variance retention: smoking/drinking data")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT / "cumulative_variance.png", dpi=150)
plt.close(fig)

print(f"Rows: {len(data):,}; training: {train_mask.sum():,}; testing: {(~train_mask).sum():,}")
print(f"Target excluded from PCA: {target}")
print(f"Continuous inputs: {len(continuous)}; categorical inputs: {len(categorical)}")
print(f"Numeric columns after encoding: {features.shape[1]}")
print(f"Components needed for at least 95% variance: {components_95}")
print("\nFirst 10 components:")
print(variance_report.head(10).round(4).to_string(index=False))
print(f"\nPC1 + PC2 variance: {cumulative[1]:.4f}")
print(f"Saved PCA outputs to {OUTPUT}")
