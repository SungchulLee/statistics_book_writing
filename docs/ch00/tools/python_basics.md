# Python and Jupyter Basics

Python is the primary computational language used throughout this book. We chose it for three reasons: a low-friction syntax that keeps statistical ideas in the foreground, a mature ecosystem of numerical libraries (NumPy, SciPy, pandas, statsmodels, scikit-learn, PyMC), and a literate-programming workflow via Jupyter notebooks where code, output, equations, and prose live in one document. This page covers environment setup, package management, the language features used most often in statistical scripts, and the conventions adopted in every code example in this book.

## Definition

### What "Python" means in this book

**Python** is a dynamically typed, garbage-collected, interpreted language with first-class functions and a "batteries included" standard library. Statistical workflows rely on:

- The CPython interpreter (version 3.11+ recommended).
- A scientific stack: NumPy (arrays), SciPy (algorithms, distributions, optimization), pandas (data frames), Matplotlib (plotting), statsmodels (regression, time series), scikit-learn (machine learning).
- A package/environment manager: `conda` (Anaconda/Miniconda) or `pip` + `venv` / `uv`.

The **Anaconda** distribution bundles Python with 1,500+ scientific packages and the `conda` package/environment manager — the path of least resistance for new users. **Jupyter Notebook** (and its successor JupyterLab) provides an interactive, cell-based interface where Markdown narrative, Python code, plots, and LaTeX equations coexist.

### Reproducible environments

Each project should pin its dependencies to a specific Python version and package versions. With conda:

```bash
conda create --name stats_env python=3.11
conda activate stats_env
conda install numpy scipy pandas matplotlib statsmodels
conda env export > environment.yml
```

The `environment.yml` file commits to source control and makes the analysis reproducible on another machine.

## Explanation

### Built-in containers and when to use each

| Container | Mutable | Ordered | Use case |
|---|---|---|---|
| `list`  | ✓ | ✓ | General-purpose sequence (e.g., a column of values before vectorizing) |
| `tuple` | ✗ | ✓ | Fixed-size record (e.g., return multiple values from a function) |
| `dict`  | ✓ | ✓ (insertion order) | Lookup by key — parameter bundles, JSON-shaped data |
| `set`   | ✓ | ✗ | Membership tests, deduplication |

Lists are the default. Reach for dictionaries when keys are meaningful labels rather than integer indices. Tuples shine as lightweight return values: `mean, std = summarize(data)` unpacks the result.

### Comprehensions

List comprehensions express the pattern "compute $f(x)$ for each $x$ in a collection, optionally filtered by a predicate" in one line:

```python
squares = [x**2 for x in data if x > 0]
```

Equivalents exist for dictionaries (`{k: f(k) for k in keys}`), sets (`{f(x) for x in xs}`), and generators (`(f(x) for x in xs)`). Generators yield lazily — important when the sequence is large or infinite.

### Functions and docstrings

```python
def sample_mean(data):
    """Return the arithmetic mean of an iterable of numbers."""
    return sum(data) / len(data)
```

A docstring is the function's contract: what it computes, what it expects, what it returns. The triple-quoted string immediately after the `def` line is accessible via `help(fn)` and is the basis of automated documentation.

Lambdas (`lambda x: x**2`) are unnamed single-expression functions, handy as arguments to `map`, `filter`, or `sorted(..., key=...)`. Prefer named `def`s for anything longer than a single expression.

### Module structure and the `__main__` guard

Every Python file is a module. To make a file safely both **importable** (its functions reused elsewhere) and **runnable** (executes a demonstration when invoked directly), use:

```python
"""Module docstring describing what this script does."""

# === Imports ===
import numpy as np

# === Function definitions ===
def my_function(x):
    return x + 1

# === Demonstration / entry point ===
if __name__ == "__main__":
    print(my_function(3))
```

The `if __name__ == "__main__":` guard ensures the demonstration block runs only on direct invocation (`python my_script.py`), not when another module does `import my_script`. This is the educational style used in every `.py` file in this book.

### Standard import aliases

The first cell of nearly every notebook is:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
```

These aliases (`np`, `pd`, `plt`, `stats`) are de facto standard and aid readability.

### Jupyter essentials

Launch with `jupyter notebook` or `jupyter lab`. Key shortcuts in command mode (press `Esc` to enter):

- `Shift+Enter` — run the current cell.
- `A` / `B` — insert a cell above / below.
- `M` / `Y` — convert the cell to Markdown / Code.
- `D D` — delete the current cell.

Markdown cells support LaTeX between `$...$` (inline) and `$$...$$` (display), making notebooks a natural place to develop statistical arguments.

## Examples

```python
"""Demonstrate core Python idioms used in statistics."""

# === Compute a summary without NumPy ===
def summarize(data):
    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / (n - 1)
    return {"mean": mean, "variance": variance, "n": n}

# === Comprehensions and unpacking ===
values = [4, 7, 13, 2, 9]
squares = [v ** 2 for v in values]
above_average = [v for v in values if v > sum(values) / len(values)]

result = summarize(values)
mean, variance, n = result["mean"], result["variance"], result["n"]

print(f"Values:           {values}")
print(f"Squares:          {squares}")
print(f"Above average:    {above_average}")
print(f"Mean = {mean:.3f}, Var = {variance:.3f}, n = {n}")

# === Hand-off to NumPy ===
import numpy as np
data = np.array(values)
print(f"NumPy mean:       {data.mean():.3f}")
print(f"NumPy var (n-1):  {data.var(ddof=1):.3f}")
```

## Exercises

**Exercise 1.**
Write a function `summary_stats(data)` that takes a list of numbers and returns a dictionary with keys `"mean"`, `"variance"`, `"n"`, and `"se_mean"` (the standard error of the mean, $s / \sqrt{n}$), computed without using NumPy.

??? success "Solution to Exercise 1"
    ```python
    from math import sqrt

    def summary_stats(data):
        n = len(data)
        mean = sum(data) / n
        variance = sum((x - mean) ** 2 for x in data) / (n - 1)
        se_mean = sqrt(variance / n)
        return {"mean": mean, "variance": variance, "n": n, "se_mean": se_mean}

    print(summary_stats([2, 4, 4, 4, 5, 5, 7, 9]))
    # {'mean': 5.0, 'variance': 4.0, 'n': 8, 'se_mean': 0.7071...}
    ```

    The variance uses $n - 1$ (Bessel's correction) for an unbiased estimate. The standard error of the mean is $s/\sqrt{n}$ where $s = \sqrt{s^2}$.

---

**Exercise 2.**
Explain the difference between a Python `list` and a NumPy `ndarray` for element-wise arithmetic. Show what `[1, 2, 3] * 2` and `np.array([1, 2, 3]) * 2` produce, and why.

??? success "Solution to Exercise 2"
    ```python
    import numpy as np

    print([1, 2, 3] * 2)            # [1, 2, 3, 1, 2, 3]
    print(np.array([1, 2, 3]) * 2)   # [2 4 6]
    ```

    Python's `list` defines `*` as repeated concatenation, not elementwise arithmetic. NumPy's `ndarray` overloads `*` to broadcast the scalar to every element. Vectorized arithmetic (`arr + 1`, `arr ** 2`, `np.sqrt(arr)`) is both faster (delegated to compiled C/BLAS routines) and shorter (no explicit loops). Reaching into Python `for` loops for numerical work is almost always a sign that the work belongs in NumPy.

---

**Exercise 3.**
Write a list comprehension that generates all pairs $(i, j)$ with $1 \le i < j \le 5$ in a single expression. Verify that the count equals $\binom{5}{2}$.

??? success "Solution to Exercise 3"
    ```python
    pairs = [(i, j) for i in range(1, 6) for j in range(i + 1, 6)]
    print(pairs)
    print(f"Count: {len(pairs)}, C(5,2) = {5 * 4 // 2}")
    # [(1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)]
    # Count: 10, C(5,2) = 10
    ```

    The double-`for` comprehension iterates `i` in the outer loop and `j` in the inner loop, mirroring nested `for` statements but in expression form.

---

**Exercise 4.**
Explain what `if __name__ == "__main__":` does. Give one example where omitting it causes unwanted behavior on import.

??? success "Solution to Exercise 4"
    `__name__` is a special module attribute. It equals `"__main__"` when the file is executed as a script (`python myscript.py`) and equals the module's import name otherwise (`mypackage.myscript`).

    Code inside `if __name__ == "__main__":` runs only on direct invocation; on import the block is skipped. Three reasons this matters:

    1. **No side effects on import**: top-level `print` statements or expensive computations would run every time the module is loaded.
    2. **Reusability**: functions and classes are available for import; demonstration code is hidden.
    3. **Convention**: every script in this book follows the pattern.

    **Example of failure without the guard:**
    ```python
    # bad_module.py
    def util(x): return x + 1
    print("Loading...")            # runs every time someone imports bad_module
    print(util(10))
    ```
    Any other module that does `import bad_module` triggers the prints — clutter at best, side-effect-driven errors at worst.

---

**Exercise 5.**
The arithmetic mean can overflow when summing many large floats. Write a one-pass online mean using the update rule

$$
\bar{x}_n = \bar{x}_{n-1} + \frac{x_n - \bar{x}_{n-1}}{n}
$$

Compare it numerically against `sum(data)/len(data)` on a list of $10^6$ values drawn from a Gaussian, and explain when the online form is preferable.

??? success "Solution to Exercise 5"
    ```python
    import numpy as np

    def online_mean(data):
        m = 0.0
        for n, x in enumerate(data, start=1):
            m += (x - m) / n
        return m

    rng = np.random.default_rng(0)
    data = rng.normal(1e6, 1, size=10**6).tolist()

    naive = sum(data) / len(data)
    online = online_mean(data)
    print(f"naive  = {naive:.6f}")
    print(f"online = {online:.6f}")
    ```

    Both agree to many digits here, but the online formulation has two real advantages: (1) it processes a stream — useful when the data does not fit in memory — and (2) it avoids accumulating a sum near $10^{12}$ for `n = 10^6, mean = 10^6`, which can lose precision in 64-bit floats. The streaming algorithm extends to variance (Welford), regression, and quantile estimation.

---

**Exercise 6.**
Why is a Python `dict` (since 3.7) a better choice than a `list` of `(key, value)` tuples for storing distribution parameters that get accessed by name? Discuss in terms of asymptotic complexity and code clarity.

??? success "Solution to Exercise 6"
    **Complexity**: dictionaries are hash tables. Lookup, insertion, and deletion by key are $O(1)$ expected time. A list of $(k, v)$ tuples requires a linear scan: $O(n)$ per lookup. For parameter bundles with a handful of entries, the constant matters more than the asymptotic, but the principle stays the same once datasets grow.

    **Clarity**: `params["mu"]` says exactly what is being read. `next(v for k, v in params if k == "mu")` says the same thing but obscures intent and is more brittle. Named access also pairs cleanly with `**kwargs` unpacking when passing the bundle to a function:

    ```python
    params = {"loc": 0.0, "scale": 1.0}
    samples = rng.normal(size=100, **params)  # equivalent to loc=0.0, scale=1.0
    ```

    The list-of-tuples form is appropriate only when keys can repeat or insertion order is the only semantic.
