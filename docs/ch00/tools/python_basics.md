# Python and Jupyter Basics

## Why Python for Statistics?

Python is a general-purpose programming language that has become the dominant tool for data analysis, scientific computing, and machine learning. Its readability, extensive library ecosystem, and active community make it an ideal choice for statistical work ranging from exploratory analysis to production-grade modeling.

## Choosing a Python Distribution

For statistical and data-science work the **Anaconda** distribution is the recommended starting point.

| Feature | Anaconda | Standard Python |
|---|---|---|
| Pre-installed packages | 1,500+ (NumPy, Pandas, Matplotlib, SciPy, …) | Standard library only |
| Package manager | `conda` + `pip` | `pip` only |
| Environment management | Built-in (`conda env`) | Requires `venv` or `virtualenv` |
| Jupyter Notebook | Included | Manual install |
| Platforms | Windows, macOS, Linux | Windows, macOS, Linux |

### Installing Anaconda

1. Download the installer from [anaconda.com](https://www.anaconda.com/products/distribution).
2. Run the installer and (optionally) add Anaconda to your system `PATH`.
3. Verify the installation:

```bash
conda --version
```

## Package Management

### conda

`conda` is Anaconda's native package and environment manager.

```bash
# Install a package
conda install numpy

# Install a specific version
conda install pandas=2.1.0

# Update a package
conda update matplotlib

# List installed packages
conda list
```

### pip

`pip` is the standard Python package installer and is useful for packages not available through `conda`.

```bash
# Install a package
pip install seaborn

# Install from a requirements file
pip install -r requirements.txt
```

!!! tip "Best Practice"
    Use `conda` for packages available in the Anaconda repository and fall back to `pip` for everything else. Mixing the two carelessly can cause dependency conflicts.

## Virtual Environments

A virtual environment isolates a project's dependencies from the system-wide Python installation, preventing version conflicts across projects.

### Creating and Managing Environments

```bash
# Create a new environment with a specific Python version
conda create --name stats_env python=3.11

# Activate the environment
conda activate stats_env

# Install packages inside the environment
conda install numpy pandas matplotlib scipy

# Deactivate when finished
conda deactivate

# List all environments
conda env list

# Remove an environment
conda env remove --name stats_env
```

### Exporting and Reproducing Environments

```bash
# Export environment specification
conda env export > environment.yml

# Recreate environment from file
conda env create -f environment.yml
```

## Integrated Development Environments

### Jupyter Notebook

Jupyter Notebook is the primary tool used throughout this book. It provides an interactive, cell-based interface where code, output, and narrative text coexist in a single document.

**Launching Jupyter:**

```bash
jupyter notebook
```

This opens the Jupyter interface in your default web browser. From there, create a new notebook and select the Python kernel.

**Key features:**

- Execute code cells independently and see results inline.
- Mix Markdown cells for documentation with code cells for analysis.
- Render $\LaTeX$ equations directly in Markdown cells.
- Export notebooks to HTML, PDF, or slides.

**Useful keyboard shortcuts:**

| Shortcut | Action |
|---|---|
| `Shift + Enter` | Run cell and move to next |
| `Ctrl + Enter` | Run cell in place |
| `Esc + A` | Insert cell above |
| `Esc + B` | Insert cell below |
| `Esc + M` | Convert cell to Markdown |
| `Esc + Y` | Convert cell to Code |
| `Esc + D D` | Delete cell |

### Other IDEs

- **Spyder** — Ships with Anaconda; MATLAB-like layout with variable explorer, editor, and console panes. Well-suited for interactive scientific computing.
- **PyCharm** — Full-featured IDE with intelligent code completion, debugging, and project management. The Professional edition includes Jupyter support.
- **VS Code** — Lightweight editor with excellent Python and Jupyter extensions; a popular all-purpose choice.

## Essential Python Refresher

The subsections below review core Python constructs that appear throughout the book.

### Data Types and Variables

```python
# Numeric types
x_int   = 42          # int
x_float = 3.14        # float
x_bool  = True        # bool (subclass of int)

# Strings
name = "statistics"

# Type checking
print(type(x_float))  # <class 'float'>
```

### Collections

```python
# List — ordered, mutable
values = [1, 2, 3, 4, 5]

# Tuple — ordered, immutable
point = (3.0, 4.0)

# Dictionary — key-value pairs
params = {"mu": 0.0, "sigma": 1.0}

# Set — unordered, unique elements
unique = {1, 2, 3, 3, 2}  # {1, 2, 3}
```

### Control Flow

```python
# Conditional
if x > 0:
    print("positive")
elif x == 0:
    print("zero")
else:
    print("negative")

# For loop
total = 0
for v in values:
    total += v

# List comprehension
squares = [v ** 2 for v in values]

# While loop
n = 10
while n > 0:
    n -= 1
```

### Functions

```python
def sample_mean(data):
    """Return the arithmetic mean of a list of numbers."""
    return sum(data) / len(data)

# Lambda (anonymous) function
square = lambda x: x ** 2
```

### Importing Libraries

```python
# Standard import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import specific objects
from scipy.stats import norm
```

## Installing the Libraries Used in This Book

Run the following once inside your environment to install all core dependencies:

```python
# From a notebook cell
!pip install numpy pandas matplotlib seaborn scipy scikit-learn statsmodels
```

Or from the terminal:

```bash
conda install numpy pandas matplotlib seaborn scipy scikit-learn statsmodels
```

## Recommended Directory Structure

Keeping your projects organized makes reproducibility straightforward:

```
project/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_modeling.ipynb
├── src/
│   └── utils.py
├── environment.yml
└── README.md
```

## Summary

| Concept | Key Takeaway |
|---|---|
| Distribution | Use Anaconda for a batteries-included setup |
| Package manager | Prefer `conda`; use `pip` as fallback |
| Environments | Always isolate projects with virtual environments |
| IDE | Jupyter Notebook for interactive analysis; Spyder / VS Code for scripts |
| Organization | Maintain a clean directory structure and export `environment.yml` |
