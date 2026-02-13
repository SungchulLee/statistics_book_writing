# 18.1 Correlation

## Understanding the Relationship Between Variables

Correlation is a fundamental concept in statistics that describes the strength and direction of the relationship between two variables. It is a crucial tool for data analysis and helps in understanding how changes in one variable are associated with changes in another.

---

## What Is Correlation?

Correlation quantifies the degree to which two variables move in relation to each other. If two variables are correlated, it means that changes in one variable tend to be associated with changes in another. The correlation coefficient, a numerical value ranging from $-1$ to $1$, measures this relationship.

- **Positive Correlation**: When two variables increase or decrease together, they exhibit a positive correlation. For instance, there is a positive correlation between education level and income; as educational attainment increases, income generally rises.

- **Negative Correlation**: When one variable increases while the other decreases, a negative correlation exists. An example is the relationship between the amount of time spent watching TV and academic performance; generally, more TV time correlates with lower academic achievement.

- **Zero Correlation**: When there is no discernible relationship between the variables, the correlation is zero. For example, the relationship between shoe size and intelligence is typically zero; changes in shoe size do not predict changes in intelligence.

---

## Visualizing Positive Correlation

The following code generates scatter plots of bivariate normal samples with increasing positive correlation coefficients, illustrating how the point cloud tightens around a line as $\rho$ increases.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(0)

def generate_samples(mu_1, mu_2, sigma_1, sigma_2, rho, n):
    """
    Generates samples from a bivariate normal distribution.

    Parameters:
        mu_1, mu_2: Means of the two variables.
        sigma_1, sigma_2: Standard deviations.
        rho: Correlation coefficient.
        n: Number of samples.

    Returns:
        np.ndarray of shape (n, 2).
    """
    covariance_matrix = [
        [sigma_1**2, rho * sigma_1 * sigma_2],
        [rho * sigma_1 * sigma_2, sigma_2**2]
    ]
    return stats.multivariate_normal([mu_1, mu_2], covariance_matrix).rvs(size=n)

def plot_correlations():
    fig, axes = plt.subplots(1, 6, figsize=(15, 4))
    correlation_coefficients = (0.00, 0.40, 0.60, 0.80, 0.90, 0.95)

    for ax, rho in zip(axes, correlation_coefficients):
        xy = generate_samples(0, 0, 1, 1, rho, 100)
        ax.plot(xy[:, 0], xy[:, 1], 'ok')
        ax.set_title(f'ρ = {rho}')
        ax.axis('off')
        ax.axis('equal')
        for loc in ('left', 'right', 'top', 'bottom'):
            ax.spines[loc].set_visible(False)
    plt.show()

if __name__ == "__main__":
    plot_correlations()
```

---

## Visualizing Negative Correlation

Similarly, negative correlation coefficients produce point clouds that slope downward.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(0)

def generate_samples(mu_1, mu_2, sigma_1, sigma_2, rho, n):
    covariance_matrix = [
        [sigma_1**2, rho * sigma_1 * sigma_2],
        [rho * sigma_1 * sigma_2, sigma_2**2]
    ]
    return stats.multivariate_normal([mu_1, mu_2], covariance_matrix).rvs(size=n)

def plot_negative_correlations():
    fig, axes = plt.subplots(1, 6, figsize=(15, 4))
    correlation_coefficients = (0.00, -0.40, -0.60, -0.80, -0.90, -0.95)

    for ax, rho in zip(axes, correlation_coefficients):
        xy = generate_samples(0, 0, 1, 1, rho, 100)
        ax.plot(xy[:, 0], xy[:, 1], 'ok')
        ax.set_title(f'ρ = {rho}')
        ax.axis('off')
        ax.axis('equal')
        for loc in ('left', 'right', 'top', 'bottom'):
            ax.spines[loc].set_visible(False)
    plt.show()

if __name__ == "__main__":
    plot_negative_correlations()
```

---

## Full Spectrum: From Strong Negative to Strong Positive

Placing all correlation values on a single row reveals the full continuum.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(0)

def generate_samples(mu_1, mu_2, sigma_1, sigma_2, rho, n):
    covariance_matrix = [
        [sigma_1**2, rho * sigma_1 * sigma_2],
        [rho * sigma_1 * sigma_2, sigma_2**2]
    ]
    return stats.multivariate_normal([mu_1, mu_2], covariance_matrix).rvs(size=n)

def plot_all_correlations():
    fig, axes = plt.subplots(1, 11, figsize=(20, 2))
    rhos = (-0.95, -0.90, -0.80, -0.60, -0.40,
             0.00,  0.40,  0.60,  0.80,  0.90, 0.95)

    for ax, rho in zip(axes, rhos):
        xy = generate_samples(0, 0, 1, 1, rho, 100)
        ax.plot(xy[:, 0], xy[:, 1], 'ok')
        ax.set_title(f'ρ = {rho}')
        ax.axis('off')
        ax.axis('equal')
        for loc in ('left', 'right', 'top', 'bottom'):
            ax.spines[loc].set_visible(False)
    plt.show()

if __name__ == "__main__":
    plot_all_correlations()
```

---

## Definition: Pearson Correlation Coefficient

The strength and direction of the linear relationship between two variables $X$ and $Y$ are captured by the **Pearson correlation coefficient**:

$$
\rho = \rho_{X,Y} = \frac{\text{Cov}(X, Y)}{\sqrt{\text{Var}(X)}\;\sqrt{\text{Var}(Y)}}
$$

where:

$$
\begin{aligned}
\mathbb{E}[X] &\approx \bar{x} = \frac{\sum_{i=1}^n x_i}{n} \\[6pt]
\mathbb{E}[Y] &\approx \bar{y} = \frac{\sum_{i=1}^n y_i}{n} \\[6pt]
\text{Var}(X) &= \mathbb{E}\!\left[(X - \mathbb{E}[X])^2\right] \approx S_x^2 = \frac{\sum_{i=1}^n (x_i - \bar{x})^2}{n-1} \\[6pt]
\text{Var}(Y) &= \mathbb{E}\!\left[(Y - \mathbb{E}[Y])^2\right] \approx S_y^2 = \frac{\sum_{i=1}^n (y_i - \bar{y})^2}{n-1} \\[6pt]
\text{Cov}(X,Y) &= \mathbb{E}\!\left[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])\right] \approx S_{xy} = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{n-1}
\end{aligned}
$$

The sample correlation coefficient can equivalently be written as:

$$
r = \frac{n\left(\sum_{i=1}^n x_i y_i\right) - \left(\sum_{i=1}^n x_i\right)\left(\sum_{i=1}^n y_i\right)}{\sqrt{\left[n\sum_{i=1}^n x_i^2 - \left(\sum_{i=1}^n x_i\right)^2\right]\left[n\sum_{i=1}^n y_i^2 - \left(\sum_{i=1}^n y_i\right)^2\right]}}
$$

### Interpretation of Pearson's $\rho$

| Range | Interpretation |
|-------|---------------|
| $\rho > 0.7$ | Strong positive correlation |
| $0.3 < \rho < 0.7$ | Moderate positive correlation |
| $0 < \rho < 0.3$ | Weak positive correlation |
| $\rho = 0$ | No correlation |
| $-0.3 < \rho < 0$ | Weak negative correlation |
| $-0.7 < \rho < -0.3$ | Moderate negative correlation |
| $\rho < -0.7$ | Strong negative correlation |

Special values:

- $\rho = 1$: Perfect positive correlation — the variables move together perfectly in the same direction.
- $\rho = -1$: Perfect negative correlation — the variables move together perfectly in opposite directions.
- $\rho = 0$: No linear correlation.

---

## Properties of Correlation

$$
\begin{aligned}
(1) &\quad \rho_{Y,X} = \rho_{X,Y} & \text{(symmetry)} \\[4pt]
(2) &\quad \rho_{X+a,\,Y} = \rho_{X,Y} & \text{(translation invariance)} \\
    &\quad \rho_{X,\,Y+a} = \rho_{X,Y} \\[4pt]
(3) &\quad \rho_{aX,\,Y} = \rho_{X,Y} \quad \text{for } a > 0 & \text{(scale invariance)} \\
    &\quad \rho_{X,\,aY} = \rho_{X,Y} \quad \text{for } a > 0
\end{aligned}
$$

These properties tell us that correlation measures the *shape* of the relationship, not its location or scale.

---

## Example: Height and Weight

A classic example of positive correlation in real data.

```python
import matplotlib.pyplot as plt
import pandas as pd

def plot_height_weight_scatter():
    url = "https://raw.githubusercontent.com/beccadsouza/Machine-Learning-Python/master/Datasets/height-weight.csv"
    data = pd.read_csv(url)
    filtered = data[data.Gender == "Male"][:300]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(filtered.Height, filtered.Weight, '.k')
    ax.set_xlabel('Height', fontsize=15)
    ax.set_ylabel('Weight', fontsize=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

if __name__ == "__main__":
    plot_height_weight_scatter()
```

### Exercise 1: Scatter Plot for Women

**Objective**: Modify the code to filter for female individuals and plot height vs. weight.

```python
import matplotlib.pyplot as plt
import pandas as pd

def plot_height_weight_scatter_for_women():
    url = "https://raw.githubusercontent.com/beccadsouza/Machine-Learning-Python/master/Datasets/height-weight.csv"
    data = pd.read_csv(url)
    filtered = data[data.Gender == "Female"][:300]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(filtered.Height, filtered.Weight, '.r')
    ax.set_xlabel('Height', fontsize=15)
    ax.set_ylabel('Weight', fontsize=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

if __name__ == "__main__":
    plot_height_weight_scatter_for_women()
```

### Exercise 2: Scatter Plot for All Individuals

**Objective**: Create a scatter plot distinguishing males and females with different colors.

```python
import matplotlib.pyplot as plt
import pandas as pd

def plot_height_weight_scatter_for_all():
    url = "https://raw.githubusercontent.com/beccadsouza/Machine-Learning-Python/master/Datasets/height-weight.csv"
    data = pd.read_csv(url)
    subset = data[:300]
    males = subset[subset.Gender == "Male"]
    females = subset[subset.Gender == "Female"]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(males.Height, males.Weight, '.k', label='Male')
    ax.plot(females.Height, females.Weight, '.r', label='Female')
    ax.set_xlabel('Height', fontsize=15)
    ax.set_ylabel('Weight', fontsize=15)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

if __name__ == "__main__":
    plot_height_weight_scatter_for_all()
```

### Exercise 3: Analyze Mixed Gender Data

**Objective**: Investigate why combining male and female data might weaken the observed linear relationship.

```python
import pandas as pd

url = "https://raw.githubusercontent.com/beccadsouza/Machine-Learning-Python/master/Datasets/height-weight.csv"
data = pd.read_csv(url)

males = data[data.Gender == "Male"]
females = data[data.Gender == "Female"]

male_corr = males[['Height', 'Weight']].corr().iloc[0, 1]
female_corr = females[['Height', 'Weight']].corr().iloc[0, 1]
combined_corr = data[['Height', 'Weight']].corr().iloc[0, 1]

print(f"Correlation for Males:         {male_corr:.4f}")
print(f"Correlation for Females:       {female_corr:.4f}")
print(f"Correlation for Combined Data: {combined_corr:.4f}")
```

**Discussion Points**: The within-group correlations may differ from the combined correlation because males and females form distinct clusters. The combined correlation reflects both the within-group relationship and the between-group separation, which can either strengthen or weaken the overall observed association.

### Exercise 4: Scatter Plot with Regression Line

**Objective**: Add a regression line to the scatter plot.

```python
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

def plot_scatter_with_regression():
    url = "https://raw.githubusercontent.com/beccadsouza/Machine-Learning-Python/master/Datasets/height-weight.csv"
    data = pd.read_csv(url)
    subset = data[:300]

    X = subset[['Height']].values
    y = subset['Weight'].values

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(subset.Height, subset.Weight, 'o', label='Data Points')
    ax.plot(subset.Height, predictions, 'r-', label='Regression Line')
    ax.set_xlabel('Height', fontsize=15)
    ax.set_ylabel('Weight', fontsize=15)
    ax.grid(True)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

if __name__ == "__main__":
    plot_scatter_with_regression()
```

---

## Importance of Correlation

Correlation analysis is invaluable across many fields:

- **Business**: Understanding market trends, customer behavior, and marketing effectiveness.
- **Healthcare**: Investigating relationships between lifestyle factors and health outcomes, such as smoking and lung cancer.
- **Education**: Analyzing the impact of teaching methods on student performance and the link between study habits and academic success.
- **Social Sciences**: Exploring relationships between social factors and behaviors, such as the connection between income and education levels.

---

## Limitations of Correlation

While correlation is a powerful tool, it has important limitations:

1. **Correlation Does Not Imply Causation**: A correlation between two variables does not mean that one causes the other. A correlation between ice cream sales and drowning incidents does not imply that ice cream consumption causes drowning—both are driven by warm weather.

2. **Confounding Variables**: Correlations can be influenced by third factors that affect both variables. A correlation between hours of sleep and academic performance might be confounded by stress levels or study habits.

3. **Non-linear Relationships**: The Pearson correlation coefficient measures *linear* relationships only. Non-linear associations may produce a Pearson $\rho$ near zero despite a strong relationship.

### Correlation Measures Linear Association, Not Association in General

Anscombe's quartet famously illustrates that very different data patterns can produce nearly identical correlation coefficients. The four datasets have the same mean, variance, correlation, and regression line, yet their scatter plots reveal fundamentally different structures—including non-linear relationships and the influence of outliers.

See: [Anscombe's Quartet (Wikipedia)](https://en.wikipedia.org/wiki/Anscombe%27s_quartet)

### Correlation Is Not Causation

This principle is so important that it warrants its own section. See [Section 17.3: Correlation, Causation, and Confounding](../confounding/confounding.md) for a thorough treatment.

---

## Summary

Correlation is a foundational concept in statistics that provides insight into linear relationships between variables. Understanding how to measure and interpret correlation helps identify patterns and inform decisions across disciplines. However, it is crucial to recognize its limitations—particularly that correlation does not imply causation—and to complement correlation analysis with other statistical methods to gain a comprehensive understanding of the data.
