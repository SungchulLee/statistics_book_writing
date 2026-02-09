# 18.7 Exercises

## Correlation

### Exercise 1: Compute and Compare Correlation Coefficients

Using the height-weight dataset, compute the Pearson correlation coefficient for:

1. Males only
2. Females only
3. Combined dataset

```python
import pandas as pd

url = "https://raw.githubusercontent.com/beccadsouza/Machine-Learning-Python/master/Datasets/height-weight.csv"
data = pd.read_csv(url)

# Your code here
```

Discuss why the combined correlation might differ from the within-group correlations.

---

### Exercise 2: Visualize Correlation Across a Range of $\rho$

Write a function that generates bivariate normal samples for $\rho \in \{-0.99, -0.8, -0.5, 0, 0.5, 0.8, 0.99\}$ and displays them in a single row of subplots. For each subplot, also print the sample Pearson $r$ in the title.

---

### Exercise 3: Anscombe's Quartet

Reproduce Anscombe's quartet using `scipy` or manually. Compute the Pearson $r$ for each of the four datasets and verify that they are nearly identical despite very different scatter plot patterns.

```python
# Hint: Anscombe's quartet is available in seaborn
import seaborn as sns
anscombe = sns.load_dataset("anscombe")
```

---

## Ecological Correlation and Simpson's Paradox

### Exercise 4: Simpson's Paradox Simulation

Create a synthetic dataset with two groups where a trend reverses when the groups are combined:

1. Generate Group A: $x \sim U(0, 5)$, $y = -0.5x + 10 + \epsilon$
2. Generate Group B: $x \sim U(5, 10)$, $y = -0.5x + 5 + \epsilon$
3. Plot each group separately and combined
4. Compute the Pearson $r$ within each group and for the combined data

Explain why the combined correlation can be positive even though both within-group correlations are negative.

---

### Exercise 5: UC Berkeley Admissions

Using the UC Berkeley admissions data from Section 18.2, compute:

1. The overall admission rate for men and women
2. The admission rate for men and women in each department
3. Identify which departments contribute most to the paradox

---

## Survivorship Bias

### Exercise 6: Simulating Survivorship Bias

Simulate an investment scenario:

1. Generate 1000 "companies" with random annual returns drawn from $N(0.05, 0.3)$ over 10 years
2. A company "survives" if its cumulative return never drops below $-90\%$
3. Compute the average annual return for survivors vs. all companies
4. Discuss how focusing only on survivors inflates perceived returns

```python
import numpy as np

np.random.seed(42)
n_companies = 1000
n_years = 10

# Your simulation here
```

---

### Exercise 7: Identify Survivorship Bias

For each scenario below, identify the survivorship bias and explain what data is missing:

1. A study finds that people who take a particular supplement live longer on average.
2. An analysis of successful restaurants finds they all have outdoor seating.
3. A review of top-performing mutual funds over 20 years shows consistent market-beating returns.

---

## Confounding

### Exercise 8: Identifying Confounders

For each of the following correlations, identify at least one plausible confounding variable:

1. Countries with more chocolate consumption per capita win more Nobel Prizes.
2. Students who eat breakfast perform better on exams.
3. Cities with more police officers have higher crime rates.
4. People who own more books tend to have higher incomes.

---

### Exercise 9: Housing Data Analysis

Using the California housing dataset:

1. Compute the full correlation matrix
2. Identify the variable most strongly correlated with `median_house_value`
3. Discuss potential confounders in the relationship between `median_income` and `median_house_value`
4. Create a scatter plot matrix for the four most correlated variables

```python
import os, tarfile, urllib.request
import pandas as pd

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data():
    if not os.path.isdir(HOUSING_PATH):
        os.makedirs(HOUSING_PATH)
    tgz_path = os.path.join(HOUSING_PATH, "housing.tgz")
    urllib.request.urlretrieve(HOUSING_URL, tgz_path)
    with tarfile.open(tgz_path) as f:
        f.extractall(path=HOUSING_PATH)

def load_housing_data():
    return pd.read_csv(os.path.join(HOUSING_PATH, "housing.csv"))

# Your analysis here
```

---

## Correlation Tests

### Exercise 10: Pearson vs. Spearman vs. Kendall

Generate data with a monotonic but nonlinear relationship:

$$y = e^{0.1x} + \epsilon, \quad x \sim U(0, 30), \quad \epsilon \sim N(0, 1)$$

1. Compute Pearson's $r$, Spearman's $\rho_s$, and Kendall's $\tau$
2. Explain why Spearman and Kendall detect the relationship more effectively than Pearson

---

### Exercise 11: Correlation Test with Real Data

Using the age-income data from Section 18.6:

```python
age    = [18, 25, 57, 45, 26, 64, 37, 40, 24, 33]
income = [15000, 29000, 68000, 52000, 32000, 80000, 41000, 45000, 26000, 33000]
```

1. Compute all three correlation coefficients and their p-values
2. At $\alpha = 0.01$, can you reject $H_0: \rho = 0$?
3. Add an outlier (age=20, income=200000) and recompute. Which test is most affected?

---

### Exercise 12: Effect of Sample Size on p-values

For a true $\rho = 0.3$:

1. Generate bivariate normal samples with $n \in \{10, 30, 100, 500, 1000\}$
2. For each $n$, compute Pearson's $r$ and its p-value
3. Plot p-value vs. sample size and discuss the relationship between sample size and statistical significance

```python
import numpy as np
from scipy import stats

np.random.seed(0)
rho = 0.3
sample_sizes = [10, 30, 100, 500, 1000]

# Your code here
```

---

## Causation

### Exercise 13: Criteria for Causation

For each of the following claims, evaluate which of the five criteria for causation (temporal precedence, covariation, elimination of confounders, plausibility, experimental evidence) are met:

1. "Smoking causes lung cancer"
2. "Wearing a seatbelt prevents death in car accidents"
3. "Eating organic food causes better health"
4. "Social media use causes depression in teenagers"

---

### Exercise 14: Study Design

Design a study to test the causal relationship between sleep duration and academic performance. Specify:

1. The type of study (observational, RCT, longitudinal)
2. How you would control for confounders
3. What variables you would measure
4. Potential ethical constraints
5. How you would interpret the results
