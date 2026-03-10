# Python and Jupyter Basics

Python is the primary language for statistical computing in this book. This page covers environment setup, package management, and the core language features needed for data analysis.

## Definition

**Python** is a general-purpose programming language. The **Anaconda** distribution bundles Python with 1,500+ scientific packages (NumPy, pandas, SciPy, Matplotlib) and the `conda` package/environment manager. **Jupyter Notebook** provides an interactive cell-based interface where code, output, and narrative coexist.

## Explanation

**Environment setup**: Install Anaconda, then create isolated environments per project with `conda create --name stats_env python=3.11`. Use `conda install` for packages in the Anaconda repository; fall back to `pip` for others. Export with `conda env export > environment.yml` for reproducibility.

**Jupyter workflow**: Launch with `jupyter notebook`. Key shortcuts: `Shift+Enter` to run a cell, `Esc+A/B` to insert cells, `Esc+M/Y` to toggle Markdown/Code. Markdown cells render LaTeX equations inline.

**Core Python for statistics**: Lists, tuples, dicts, and sets are the built-in collections. List comprehensions (`[x**2 for x in data]`) replace explicit loops. Functions use `def`; lambdas handle simple one-liners. The standard import convention is:

```python
import numpy as np
import pandas as pd
from scipy import stats
```

## Examples

```python
# Define a function and use it
def sample_mean(data):
    """Return the arithmetic mean of a list."""
    return sum(data) / len(data)

values = [4, 7, 13, 2, 9]
print(f"Mean: {sample_mean(values)}")

# List comprehension for squares
squares = [v ** 2 for v in values]
print(f"Squares: {squares}")

# Dictionary for storing parameters
params = {"mu": 0.0, "sigma": 1.0, "n": 100}
print(f"Parameters: {params}")

# Verify numpy is available
import numpy as np
data = np.array(values)
print(f"NumPy mean: {data.mean():.2f}")
```
