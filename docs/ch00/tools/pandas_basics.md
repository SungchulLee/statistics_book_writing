# Data Handling with pandas

pandas provides labeled data structures — `Series` (1-D) and `DataFrame` (2-D) — for loading, cleaning, transforming, and summarizing structured data. It is the bridge between raw files (CSV, parquet, SQL) and the array-level computations of NumPy or the statistical models of statsmodels and scikit-learn. Almost every empirical workflow in this book starts with a pandas import and ends only after a final summary table is rendered.

## Definition

A **`Series`** is a 1-D labeled array — values plus an index. A **`DataFrame`** is a 2-D table where each column is a `Series`, possibly of a different dtype, but all sharing the same row index. Conceptually, a DataFrame is a dictionary of columns; mechanically, each column is backed by a NumPy array.

The mental model is SQL on top of NumPy: the array layer gives speed; the labeling layer gives selection, alignment, group-by, join, and pivot semantics that align with statistical analysis.

## Explanation

### Loading and inspecting

```python
import pandas as pd

df = pd.read_csv("data.csv")
df.head()        # first 5 rows
df.tail()        # last 5 rows
df.shape         # (n_rows, n_cols)
df.dtypes        # type of each column
df.info()        # dtypes, non-null counts, memory usage
df.describe()    # numeric summary: count, mean, std, min, quartiles, max
```

`read_csv` accepts `parse_dates`, `dtype`, `na_values`, `usecols`, and `chunksize` — most data-quality issues are best addressed at load time, not later.

### Selecting rows and columns

There are three orthogonal operations you must distinguish:

| Pattern | Meaning |
|---|---|
| `df["col"]`, `df[["col1", "col2"]]` | Column selection by label |
| `df.loc[row_label, col_label]` | Label-based, both axes |
| `df.iloc[row_pos, col_pos]` | Integer-position-based, both axes |
| `df[df["x"] > 5]` | Boolean row filter on a column |

`loc` and `iloc` differ exactly when the row index is not the default `0, 1, 2, ...`. Use the explicit form whenever the index has been sorted, set, or filtered.

### Cleaning missing values

```python
df.isna().sum()             # count of missing per column
df.dropna()                 # drop rows with any NaN
df.dropna(subset=["x"])     # drop rows with NaN in 'x' only
df.fillna(df.median(numeric_only=True))   # impute with column median
```

There is no "right" imputation strategy; the choice (drop, mean, median, model-based, multiple imputation) depends on the missingness mechanism. Pandas gives you the tools but leaves the decision to you.

### Grouping: split–apply–combine

The single most powerful pattern in pandas:

```python
df.groupby("treatment")["outcome"].agg(["count", "mean", "std"])
```

`groupby` splits the data by the unique values of `"treatment"`, applies the listed aggregations to `"outcome"` within each group, and combines the results into a tidy DataFrame. This is the workhorse of exploratory and confirmatory analysis. Multiple keys (`groupby(["a", "b"])`) and custom aggregations (`agg(my_func)`) generalize the pattern.

### Descriptive statistics mapped to formulas

| pandas call | Computes | ddof default |
|---|---|---|
| `df["x"].mean()` | $\bar{x}$ | n/a |
| `df["x"].var()` | $s^2$ | **`ddof=1`** |
| `df["x"].std()` | $s$ | **`ddof=1`** |
| `df["x"].quantile(0.5)` | sample median | n/a |
| `df.corr()` | Pearson correlation matrix | n/a |
| `df.cov()` | sample covariance matrix | `ddof=1` |

Note the difference from NumPy, which defaults to `ddof=0`. When numbers from pandas and NumPy disagree by a factor of $(n-1)/n$, this is why.

## Examples

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# Build a DataFrame mixing categorical and numeric data
df = pd.DataFrame({
    "group": rng.choice(["A", "B", "C"], size=200),
    "value": rng.normal(50, 10, size=200),
    "score": rng.integers(0, 100, size=200),
})

# Per-group summary
summary = df.groupby("group")["value"].agg(["count", "mean", "std"]).round(2)
print(summary)

# Boolean filtering
high = df[df["value"] > 60]
print(f"\nValues > 60: {len(high)} / {len(df)}")

# Correlation between numeric columns
print("\nCorrelation:\n", df.corr(numeric_only=True).round(3))

# Pivot: mean value by group × score quartile
df["score_q"] = pd.qcut(df["score"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
print("\nPivot:\n", df.pivot_table(values="value", index="group",
                                   columns="score_q", aggfunc="mean").round(1))
```

## Exercises

**Exercise 1.**
Create a `DataFrame` with columns `name` (str), `score` (int), and `passed` (bool) for five students. Filter to show only rows where `passed` is `True` **and** `score` is above 80.

??? success "Solution to Exercise 1"
    ```python
    import pandas as pd
    df = pd.DataFrame({
        "name":   ["Alice", "Bob", "Carol", "Dave", "Eve"],
        "score":  [92, 65, 88, 73, 95],
        "passed": [True, False, True, False, True],
    })
    print(df[df["passed"] & (df["score"] > 80)])
    ```
    The bit-wise `&` is required (not `and`) because the operands are pandas boolean Series, not Python scalars. Parenthesize each comparison — operator precedence places `&` higher than `>`.

---

**Exercise 2.**
Given the DataFrame
```python
df = pd.DataFrame({"group": ["A","A","B","B","B"], "x": [1, 3, 2, 8, 5]})
```
compute, for each group, the count, sample mean, sample variance, and the maximum minus minimum. Return a single DataFrame.

??? success "Solution to Exercise 2"
    ```python
    summary = df.groupby("group")["x"].agg(
        n="count",
        mean="mean",
        var="var",
        range=lambda s: s.max() - s.min(),
    )
    print(summary)
    ```
    `agg` accepts keyword arguments mapping output column names to either string aggregation names or callables. The lambda lets you express "range" as a single expression without writing a named function.

---

**Exercise 3.**
Explain the difference between `df.loc[]` and `df.iloc[]`. Provide an example where they return different rows for the same DataFrame.

??? success "Solution to Exercise 3"
    `loc` is **label-based**: `df.loc[0]` returns the row whose index label is `0`. `iloc` is **integer-position-based**: `df.iloc[0]` returns the first row regardless of its index label.

    ```python
    df = pd.DataFrame({"A": [10, 20, 30]}, index=[2, 0, 1])
    print(df.loc[0])    # row labeled 0   → A = 20
    print(df.iloc[0])   # row at position 0 → A = 10
    ```

    The two coincide whenever the index is `RangeIndex(0, n)` (the default) and no rows have been reordered. After `df.sort_values()`, `df.set_index()`, or boolean filtering, the two diverge — and silent confusion of the two is a common source of bugs.

---

**Exercise 4.**
Load the following CSV-formatted string, fill missing values with the column median, and compute the correlation matrix:

```text
x,y,z
1.0,2.0,3.0
2.0,,4.0
3.0,6.0,
4.0,8.0,6.0
5.0,10.0,7.0
```

??? success "Solution to Exercise 4"
    ```python
    from io import StringIO
    csv = """x,y,z
    1.0,2.0,3.0
    2.0,,4.0
    3.0,6.0,
    4.0,8.0,6.0
    5.0,10.0,7.0"""

    df = pd.read_csv(StringIO(csv))
    df = df.fillna(df.median(numeric_only=True))
    print(df.corr().round(3))
    ```

    Median imputation is more robust to outliers than mean imputation but still distorts variance and correlation when missingness is informative. For real analyses, model-based or multiple imputation should be preferred — see Chapter 12.

---

**Exercise 5.**
The default `pd.Series.var()` uses `ddof=1`; NumPy's `np.var()` uses `ddof=0`. Construct a Series of five values, compute the variance both ways, and explain which one is the unbiased estimator and why.

??? success "Solution to Exercise 5"
    ```python
    import numpy as np
    import pandas as pd
    s = pd.Series([2, 4, 4, 4, 5])
    print("pandas s.var()  :", s.var())            # ddof=1, divides by n-1=4
    print("numpy  np.var() :", np.var(s.values))   # ddof=0, divides by n=5
    ```

    With `ddof=1`, $S^2 = \frac{1}{n-1}\sum(x_i - \bar x)^2$ is unbiased: $\mathbb{E}[S^2] = \sigma^2$. With `ddof=0`, $\tilde S^2 = \frac{1}{n}\sum(x_i - \bar x)^2$ is the **maximum-likelihood** variance for normal data but is biased downward by a factor of $(n-1)/n$. The two libraries' opposite defaults are a perennial source of confusion; always check which is being used.

---

**Exercise 6.**
Two DataFrames are joined by `df1.merge(df2, on="id", how="left")`. Explain what `how="left"` does, what determines the number of rows in the result, and one diagnostic you should run immediately after merging.

??? success "Solution to Exercise 6"
    `how="left"` keeps every row of `df1` and attaches matching columns from `df2` by `id`. Rows of `df1` with no match in `df2` get `NaN` in the new columns. Rows of `df2` with no match in `df1` are dropped.

    The output row count equals the row count of `df1` **only if** `id` is unique in `df2`. If `id` repeats in `df2`, each `df1` row matches multiple `df2` rows and the result has more rows than `df1` — a common surprise.

    Diagnostics to run immediately:

    ```python
    assert df2["id"].is_unique, "right-side join key not unique — row count will inflate"
    merged = df1.merge(df2, on="id", how="left", indicator=True)
    print(merged["_merge"].value_counts())   # left_only / both / right_only
    ```

    The `indicator=True` flag adds a `_merge` column reporting where each row originated, exposing silent join failures.
