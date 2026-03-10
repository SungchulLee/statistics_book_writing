# Data Handling with pandas

pandas provides labeled data structures (`Series` and `DataFrame`) for loading, cleaning, transforming, and summarizing structured data. It is the primary tool for data preparation throughout this book.

## Definition

A **Series** is a 1-D labeled array (like a named column). A **DataFrame** is a 2-D labeled table where each column is a Series, possibly of different dtypes. DataFrames support SQL-style operations: selection, filtering, grouping, joining, and pivoting.

## Explanation

**Loading and inspection**: `pd.read_csv()` reads tabular files. Inspect with `df.head()`, `df.shape`, `df.dtypes`, `df.describe()`.

**Selection**: Column access via `df["col"]` or `df[["col1","col2"]]`. Row access via `df.iloc[i]` (integer position) or `df.loc[label]`. Boolean filtering: `df[df["x"] > 5]`.

**Cleaning**: `df.isna().sum()` counts missing values per column. `df.dropna()` removes rows with NaN; `df.fillna(value)` imputes.

**Grouping**: `df.groupby("key")["value"].agg(["mean","std","count"])` implements split-apply-combine, the workhorse of exploratory analysis.

**Descriptive statistics**: `df.mean()`, `df.std()` (ddof=1 by default), `df.var()`, `df.quantile()`, `df.corr()` map directly to statistical concepts.

## Examples

```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)

# Create a DataFrame and compute grouped statistics
df = pd.DataFrame({
    "group": rng.choice(["A", "B", "C"], size=200),
    "value": rng.normal(50, 10, size=200)
})

summary = df.groupby("group")["value"].agg(["count", "mean", "std"])
print(summary.round(2))

# Boolean filtering
high = df[df["value"] > 60]
print(f"\nValues > 60: {len(high)} out of {len(df)}")

# Correlation matrix
df2 = pd.DataFrame({
    "X": rng.normal(0, 1, 100),
    "Y": rng.normal(0, 1, 100)
})
df2["Z"] = 2 * df2["X"] + rng.normal(0, 0.5, 100)
print("\nCorrelation matrix:")
print(df2.corr().round(3))
```
