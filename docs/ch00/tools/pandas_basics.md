# Data Handling with pandas

## Overview

**pandas** is the primary library for structured data manipulation in Python. Built on top of NumPy, it provides two core data structures—`Series` (1-D) and `DataFrame` (2-D)—along with a rich set of tools for loading, cleaning, transforming, and summarizing data. Throughout this book, pandas is used to prepare datasets for statistical analysis.

```python
import pandas as pd
import numpy as np
```

## Core Data Structures

### Series

A `Series` is a one-dimensional labeled array. It behaves like a NumPy array with an attached index.

```python
s = pd.Series([10, 20, 30, 40], index=["a", "b", "c", "d"])
print(s)
# a    10
# b    20
# c    30
# d    40

print(s["b"])       # 20
print(s[s > 15])    # b, c, d entries
print(s.dtype)      # int64
```

### DataFrame

A `DataFrame` is a two-dimensional labeled table where each column is a `Series`. Columns can have different data types.

```python
data = {
    "Name":   ["Alice", "Bob", "Charlie"],
    "Age":    [25, 30, 35],
    "Salary": [70_000, 80_000, 90_000]
}
df = pd.DataFrame(data)
print(df)
#       Name  Age  Salary
# 0    Alice   25   70000
# 1      Bob   30   80000
# 2  Charlie   35   90000
```

## Loading Data

```python
# CSV
df = pd.read_csv("data.csv")

# Excel
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")

# From a URL
url = "https://example.com/dataset.csv"
df = pd.read_csv(url)

# Tab-separated
df = pd.read_csv("data.tsv", sep="\t")
```

## Inspecting a DataFrame

```python
df.head()            # first 5 rows
df.tail(3)           # last 3 rows
df.shape             # (rows, columns)
df.dtypes            # data type of each column
df.info()            # concise summary (types, non-null counts, memory)
df.describe()        # descriptive statistics for numeric columns
df.columns           # column names
df.index             # row index
```

## Selecting Data

### Column Selection

```python
# Single column → Series
df["Age"]

# Multiple columns → DataFrame
df[["Name", "Salary"]]
```

### Row Selection

```python
# By integer position
df.iloc[0]          # first row as Series
df.iloc[0:2]        # first two rows as DataFrame

# By label
df.loc[0]           # row with index label 0
df.loc[0:1, "Name":"Age"]   # label-based slicing (inclusive on both ends)
```

### Conditional Filtering

```python
# Rows where Salary > 75000
high_salary = df[df["Salary"] > 75_000]
print(high_salary)
#       Name  Age  Salary
# 1      Bob   30   80000
# 2  Charlie   35   90000

# Multiple conditions (use & for AND, | for OR; parentheses required)
subset = df[(df["Age"] >= 30) & (df["Salary"] > 75_000)]
```

## Adding and Modifying Columns

```python
# New column from arithmetic
df["Bonus"] = df["Salary"] * 0.10

# New column from a function
df["Senior"] = df["Age"].apply(lambda x: x >= 30)

# Rename columns
df = df.rename(columns={"Salary": "Annual_Salary"})
```

## Handling Missing Data

```python
# Detect missing values
df.isna().sum()              # count NaNs per column

# Drop rows with any NaN
df_clean = df.dropna()

# Drop rows where specific column is NaN
df_clean = df.dropna(subset=["Age"])

# Fill missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
```

## Sorting

```python
# Sort by a single column
df.sort_values("Age")

# Sort descending
df.sort_values("Salary", ascending=False)

# Sort by multiple columns
df.sort_values(["Age", "Salary"], ascending=[True, False])
```

## Grouping and Aggregation

`groupby` splits the data by one or more keys, applies an aggregation function, and combines the results—a cornerstone of exploratory data analysis.

```python
data = {
    "Department": ["Sales", "Sales", "Engineering", "Engineering", "HR"],
    "Employee":   ["A", "B", "C", "D", "E"],
    "Salary":     [60_000, 70_000, 90_000, 95_000, 65_000]
}
df = pd.DataFrame(data)

# Mean salary by department
df.groupby("Department")["Salary"].mean()
# Department
# Engineering    92500
# HR             65000
# Sales          65000

# Multiple aggregations
df.groupby("Department")["Salary"].agg(["mean", "std", "count"])
```

## Merging and Joining

```python
# Two DataFrames sharing a key column
orders = pd.DataFrame({
    "OrderID":    [1, 2, 3],
    "CustomerID": [101, 102, 103],
    "Amount":     [250, 450, 300]
})

customers = pd.DataFrame({
    "CustomerID": [101, 102, 104],
    "Name":       ["Alice", "Bob", "Diana"]
})

# Inner join (only matching keys)
merged = pd.merge(orders, customers, on="CustomerID", how="inner")

# Left join (keep all orders)
merged_left = pd.merge(orders, customers, on="CustomerID", how="left")
```

## Pivot Tables

```python
sales = pd.DataFrame({
    "Region":  ["East", "East", "West", "West"],
    "Quarter": ["Q1", "Q2", "Q1", "Q2"],
    "Revenue": [100, 150, 200, 180]
})

pivot = sales.pivot_table(values="Revenue", index="Region", columns="Quarter", aggfunc="sum")
print(pivot)
# Quarter   Q1   Q2
# Region
# East     100  150
# West     200  180
```

## Descriptive Statistics with pandas

pandas provides convenient methods that align with the statistical concepts covered in this book.

```python
rng = np.random.default_rng(42)
df = pd.DataFrame({
    "X": rng.normal(50, 10, 200),
    "Y": rng.normal(100, 25, 200)
})

# Central tendency
df.mean()
df.median()

# Spread
df.std()          # sample standard deviation (ddof=1 by default)
df.var()          # sample variance
df.quantile([0.25, 0.5, 0.75])

# Shape
df.skew()
df.kurt()         # excess kurtosis

# Correlation
df.corr()         # Pearson correlation matrix
df.corr(method="spearman")
```

## Working with Dates and Times

Statistical data frequently includes timestamps. pandas has first-class support for datetime operations.

```python
# Parse dates during CSV read
df = pd.read_csv("timeseries.csv", parse_dates=["date"])

# Manual conversion
df["date"] = pd.to_datetime(df["date"])

# Set as index for time-series operations
df = df.set_index("date")

# Resample to monthly frequency
monthly = df.resample("M").mean()

# Rolling statistics
df["rolling_mean_7d"] = df["value"].rolling(window=7).mean()
```

## Applying Functions

```python
# Element-wise via apply
df["log_Salary"] = df["Salary"].apply(np.log)

# Row-wise via apply with axis=1
df["Total"] = df.apply(lambda row: row["Base"] + row["Bonus"], axis=1)

# Vectorized string methods
df["Name_upper"] = df["Name"].str.upper()
```

## Exporting Data

```python
df.to_csv("output.csv", index=False)
df.to_excel("output.xlsx", index=False)
```

## Summary

| Concept | Key Takeaway |
|---|---|
| `Series` / `DataFrame` | Core 1-D and 2-D labeled data structures |
| Selection | `loc` for labels, `iloc` for positions, boolean masks for filtering |
| Missing data | `isna`, `dropna`, `fillna` |
| Grouping | `groupby` + aggregation for split-apply-combine analysis |
| Merging | `pd.merge` for SQL-style joins on key columns |
| Descriptive stats | `mean`, `std`, `var`, `quantile`, `corr` map directly to statistical concepts |
| Time series | `pd.to_datetime`, `resample`, `rolling` for temporal data |
