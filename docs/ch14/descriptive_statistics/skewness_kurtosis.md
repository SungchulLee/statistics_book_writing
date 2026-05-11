# Skewness and Kurtosis


Descriptive statistics quantify the shape of a distribution and assess how closely it resembles a normal distribution. Two key measures for evaluating normality are **skewness** and **kurtosis**. These metrics describe the asymmetry and peakedness of the data distribution, respectively.

## Skewness

**Skewness** measures the asymmetry of the distribution. For a perfectly normal distribution, skewness is 0. A positive skewness indicates a long right tail (data skewed to the right), while a negative skewness indicates a long left tail (data skewed to the left).

The formula for skewness is:

$$
\text{Skewness} = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{x_i - \bar{x}}{s} \right)^3
$$

where

- $n$ is the number of data points,
- $x_i$ is each data point,
- $\bar{x}$ is the sample mean,
- $s$ is the sample standard deviation.

If the skewness is close to zero, the population distribution will likely be symmetric, indicating normality. Significant deviations from zero suggest non-normality.

```python
import numpy as np
from scipy import stats

np.random.seed(0)

# Generate a sample dataset
data = np.random.normal(0, 3, 1000)
# data = np.random.exponential(1, 1000)

# Calculate skewness
skewness_value = stats.skew(data)
print(f"Skewness: {skewness_value:.4f}")
```

## Kurtosis

**Kurtosis** describes the "tailedness" of the distribution. A normal distribution has a kurtosis value of 3 (also called **mesokurtic**). Kurtosis values above 3 indicate a distribution with heavy tails (**leptokurtic**), while values below 3 indicate lighter tails (**platykurtic**).

The formula for kurtosis is:

$$
\text{Kurtosis} = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{x_i - \bar{x}}{s} \right)^4
$$

and the formula for **excess kurtosis** is:

$$
\text{Excess Kurtosis} = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{x_i - \bar{x}}{s} \right)^4 - 3
$$

The subtraction of 3 ensures that a normal distribution has an excess kurtosis of 0 (for easier comparison). The `scipy.stats.kurtosis` function computes this **excess kurtosis**, not the raw kurtosis.

A kurtosis value (computed by `scipy.stats.kurtosis`) near zero suggests a normal distribution. Larger values indicate heavier tails, while smaller values suggest lighter tails than normal.

```python
import numpy as np
from scipy import stats

np.random.seed(0)

# Generate a sample dataset
# data = np.random.normal(0, 1, 1000)
data = np.random.exponential(1, 1000)

# Calculate skewness
skewness_value = stats.skew(data)
print(f"Skewness: {skewness_value:.4f}")

# Calculate kurtosis
kurtosis_value = stats.kurtosis(data)
print(f"Kurtosis: {kurtosis_value:.4}")
```

## Exercises

**Exercise 1.**
Compute the sample skewness and excess kurtosis for the data: $\{1, 2, 2, 3, 3, 3, 4, 4, 5, 100\}$. What does the result tell you?

??? success "Solution to Exercise 1"
    $\bar{x} = 12.7$, $n = 10$. The single outlier (100) will dominate the higher moments.

    The sample skewness will be large and positive (the outlier pulls the right tail), and the excess kurtosis will be very large (the outlier creates an extreme fourth-moment contribution).

    Computing: The skewness is approximately $2.87$ and the excess kurtosis is approximately $8.9$. These values indicate severe right-skewness and extremely heavy tails. A single outlier can dramatically inflate both measures, illustrating their sensitivity to extreme values.

---

**Exercise 2.**
For the standard normal distribution, state the theoretical values of skewness and kurtosis (both regular and excess). Why is excess kurtosis often preferred?

??? success "Solution to Exercise 2"
    For $N(0,1)$: skewness $= 0$ (symmetric), kurtosis $= 3$, excess kurtosis $= 3 - 3 = 0$.

    Excess kurtosis subtracts 3 (the normal benchmark) so that the normal distribution has excess kurtosis of zero. This makes interpretation easier: positive excess kurtosis (leptokurtic) means heavier tails than normal; negative (platykurtic) means lighter tails. Without the adjustment, a kurtosis of 4 could be misinterpreted without context.

---

**Exercise 3.**
The uniform distribution has excess kurtosis $= -1.2$. Explain what this means in terms of tail behavior compared to the normal.

??? success "Solution to Exercise 3"
    Negative excess kurtosis means the uniform distribution has **lighter tails** (and a flatter peak) than the normal distribution. The uniform distribution is bounded -- it has no tails at all beyond its support -- so extreme values are impossible.

    Kurtosis is sometimes misinterpreted as "peakedness," but it is primarily a measure of tail heaviness. The uniform distribution has the lightest possible tails (completely bounded), which produces strongly negative excess kurtosis. Any observation from a uniform distribution lies within a fixed range, whereas normal observations can theoretically be any real number.

---

**Exercise 4.**
A financial analyst reports that daily stock returns have excess kurtosis of 5. Interpret this in the context of risk management.

??? success "Solution to Exercise 4"
    Excess kurtosis of 5 means the return distribution has much heavier tails than the normal distribution. Specifically, extreme returns (both large gains and large losses) occur far more frequently than a normal model would predict.

    For risk management, this has critical implications:

    - **VaR underestimation:** A normal-based 99% VaR (1% tail probability) will underestimate potential losses because the actual 1st percentile is more extreme.
    - **More frequent tail events:** Events that a normal model considers once-in-a-century may occur once-in-a-decade with kurtosis = 8 (kurtosis + 3).
    - **Model choice:** Use heavy-tailed distributions (Student's $t$, generalized Pareto) for risk models instead of the normal.
