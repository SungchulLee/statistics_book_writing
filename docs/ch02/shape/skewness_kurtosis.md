# Skewness and Kurtosis

## Overview

**Skewness** and **kurtosis** are numerical measures that quantify the shape of a distribution beyond what the mean and variance capture. Skewness measures asymmetry, while kurtosis measures the heaviness of the tails relative to a normal distribution.

---

## 1. Symmetric and Skewed Distributions

### Symmetric Distribution

A distribution is **symmetric** if its left and right sides mirror each other. The most common example is the normal distribution (bell curve), where mean, median, and mode are equal and located at the center.

**Example:** Heights of people often follow a symmetric distribution.

#### Symmetric Distribution: Mixture of Gaussians

A symmetric shape can also arise from a mixture of distributions, provided the components are centered at the same location.

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def generate_and_plot_mixed_distribution(seed: int = 0):
    np.random.seed(seed)
    main_data = stats.norm().rvs(1_000)
    minor_1 = stats.norm(scale=2).rvs(200)
    minor_2 = stats.norm(scale=4).rvs(100)
    combined = np.concatenate((main_data, minor_1, minor_2))

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.hist(combined, bins=30)
    plt.show()

if __name__ == "__main__":
    generate_and_plot_mixed_distribution()
```

### Skewed Distributions

A **skewed** distribution has data that stretches more on one side than the other.

**Right-Skewed (Positively Skewed):** The tail extends to the right. Mean > Median > Mode. Example: income distribution.

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def generate_and_plot_right_skewed_distribution(seed: int = 0):
    np.random.seed(seed)
    main_data = stats.norm().rvs(1_000)
    right_1 = stats.norm(loc=2).rvs(200)
    right_2 = stats.norm(loc=4).rvs(100)
    combined = np.concatenate((main_data, right_1, right_2))

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.hist(combined, bins=30)
    plt.show()

if __name__ == "__main__":
    generate_and_plot_right_skewed_distribution()
```

**Left-Skewed (Negatively Skewed):** The tail extends to the left. Mean < Median < Mode. Example: age at retirement.

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def generate_and_plot_left_skewed_distribution(seed: int = 0):
    np.random.seed(seed)
    main_data = stats.norm().rvs(1_000)
    left_1 = stats.norm(loc=-2).rvs(200)
    left_2 = stats.norm(loc=-4).rvs(100)
    combined = np.concatenate((main_data, left_1, left_2))

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.hist(combined, bins=30)
    plt.show()

if __name__ == "__main__":
    generate_and_plot_left_skewed_distribution()
```

---

## 2. Detecting Skewness via Box Plots

Box plots provide a quick visual diagnostic for skewness:

$$
\begin{array}{lll}
\text{Left\_Box} > \text{Right\_Box} &\Rightarrow& \text{Skew to Left} \\
\text{Left\_Box} < \text{Right\_Box} &\Rightarrow& \text{Skew to Right} \\
\text{Left\_Box} = \text{Right\_Box},\; \text{Left\_Whisker} > \text{Right\_Whisker} &\Rightarrow& \text{Skew to Left} \\
\text{Left\_Box} = \text{Right\_Box},\; \text{Left\_Whisker} < \text{Right\_Whisker} &\Rightarrow& \text{Skew to Right} \\
\text{Left\_Box} = \text{Right\_Box},\; \text{Left\_Whisker} = \text{Right\_Whisker} &\Rightarrow& \text{Symmetric} \\
\end{array}
$$

### Box Plot: Symmetric Distribution

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def generate_and_plot_histogram_and_box_plot_mixed_distribution(seed: int = 0):
    np.random.seed(seed)
    main_data = stats.norm().rvs(1_000)
    minor_1 = stats.norm(scale=2).rvs(200)
    minor_2 = stats.norm(scale=4).rvs(100)
    combined = np.concatenate((main_data, minor_1, minor_2))

    fig, (ax_hist, ax_box) = plt.subplots(2, 1, figsize=(12, 6))
    ax_hist.hist(combined, density=True, bins=30)
    ax_hist.set_title('Histogram of Combined Data (Density)')
    ax_box.boxplot(combined, vert=False)
    ax_box.set_title('Boxplot of Combined Data')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_and_plot_histogram_and_box_plot_mixed_distribution()
```

### Box Plot: Right-Skewed Distribution

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def generate_and_plot_histogram_and_box_plot_right_skewed(seed: int = 0):
    np.random.seed(seed)
    main_data = stats.norm().rvs(1_000)
    right_1 = stats.norm(loc=2).rvs(200)
    right_2 = stats.norm(loc=4).rvs(100)
    combined = np.concatenate((main_data, right_1, right_2))

    fig, (ax_hist, ax_box) = plt.subplots(2, 1, figsize=(12, 6))
    ax_hist.hist(combined, density=True, bins=30)
    ax_hist.set_title('Histogram of Combined Data (Density)')
    ax_box.boxplot(combined, vert=False)
    ax_box.set_title('Boxplot of Combined Data')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_and_plot_histogram_and_box_plot_right_skewed()
```

### Box Plot: Left-Skewed Distribution

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def generate_and_plot_histogram_and_box_plot_left_skewed(seed: int = 0):
    np.random.seed(seed)
    main_data = stats.norm().rvs(1_000)
    left_1 = stats.norm(loc=-2).rvs(200)
    left_2 = stats.norm(loc=-4).rvs(100)
    combined = np.concatenate((main_data, left_1, left_2))

    fig, (ax_hist, ax_box) = plt.subplots(2, 1, figsize=(12, 6))
    ax_hist.hist(combined, density=True, bins=30)
    ax_hist.set_title('Histogram of Combined Data (Density)')
    ax_box.boxplot(combined, vert=False)
    ax_box.set_title('Boxplot of Combined Data')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_and_plot_histogram_and_box_plot_left_skewed()
```

---

## 3. Skewness: Definition and Computation

### Definition

$$
\text{Skewness}(X) = E\left(\frac{X - \mu}{\sigma}\right)^3 \approx \frac{1}{n}\sum_{i=1}^{n}\left(\frac{x_i - \bar{x}}{s}\right)^3
$$

- **Skewness = 0:** Symmetric distribution.
- **Skewness > 0:** Right-skewed (positive skew).
- **Skewness < 0:** Left-skewed (negative skew).

### Skewness Simulation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def generate_samples(main_size, right_size, left_size):
    main_sample = np.random.normal(0, 1, main_size)
    right_sample = np.random.normal(2, 1, right_size)
    left_sample = np.random.normal(-2, 1, left_size)
    return np.concatenate([main_sample, right_sample, left_sample])

def calculate_statistics(data):
    n = data.shape[0]
    mean = data.sum() / n
    std_dev = np.sqrt(np.sum((data - mean) ** 2) / n)
    skewness = stats.describe(data).skewness
    return mean, std_dev, skewness

def plot_distribution_with_normal_fit(data, mean, std_dev, skewness, title):
    fig, ax = plt.subplots(figsize=(12, 3))
    _, bins, _ = ax.hist(data, density=True, bins=100, label="Samples")
    normal_pdf = stats.norm(loc=mean, scale=std_dev).pdf(bins)
    ax.plot(bins, normal_pdf, "--r", label="Normal PDF")
    ax.set_title(f"{title}\nSkewness = {skewness:.4f}")
    ax.legend()
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.show()

def main():
    np.random.seed(0)
    main_size = 10_000
    right_size = 3_000
    left_size = 3_000

    samples = generate_samples(main_size, right_size, left_size)
    mean, std_dev, skewness = calculate_statistics(samples)

    if right_size > left_size:
        title = "Right-Skewed Distribution"
    elif right_size < left_size:
        title = "Left-Skewed Distribution"
    else:
        title = "Symmetric Distribution"

    plot_distribution_with_normal_fit(samples, mean, std_dev, skewness, title)

if __name__ == "__main__":
    main()
```

---

## 4. Kurtosis

### Definition

Kurtosis measures the "tailedness" of a distribution—how much probability mass is in the tails relative to the center.

$$
\text{Kurtosis}(X) = E\left(\frac{X - \mu}{\sigma}\right)^4 \approx \frac{1}{n}\sum_{i=1}^{n}\left(\frac{x_i - \bar{x}}{s}\right)^4
$$

**Excess Kurtosis** subtracts the kurtosis of the normal distribution (which equals 3):

$$
\text{Excess Kurtosis}(X) = \text{Kurtosis}(X) - 3
$$

- **Excess Kurtosis = 0 (Mesokurtic):** Normal-like tails.
- **Excess Kurtosis > 0 (Leptokurtic):** Heavier tails than normal; more extreme outliers.
- **Excess Kurtosis < 0 (Platykurtic):** Lighter tails than normal; fewer extreme values.

### Kurtosis Simulation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def generate_samples(main_size, peak_size):
    main_sample = np.random.normal(0, 1, main_size)
    peak_sample = np.random.normal(0, 0.2, peak_size)
    return np.concatenate([main_sample, peak_sample])

def calculate_statistics(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    skewness = stats.describe(data).skewness
    kurtosis = np.mean(((data - mean) / std_dev) ** 4)
    excess_kurtosis = kurtosis - 3
    return mean, std_dev, skewness, kurtosis, excess_kurtosis

def plot_distribution_with_normal_fit(data, mean, std_dev, excess_kurtosis, title):
    fig, ax = plt.subplots(figsize=(12, 3))
    _, bins, _ = ax.hist(data, density=True, bins=100, label="Sample Data")
    normal_pdf = stats.norm(loc=mean, scale=std_dev).pdf(bins)
    ax.plot(bins, normal_pdf, "--r", label="Normal PDF")
    ax.set_title(f"{title}\nExcess Kurtosis = {excess_kurtosis:.4f}")
    ax.legend()
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.show()

def main():
    np.random.seed(0)
    main_size = 10_000
    peak_size = 500

    data = generate_samples(main_size, peak_size)
    mean, std_dev, skewness, kurtosis, excess_kurtosis = calculate_statistics(data)

    if excess_kurtosis > 0:
        title = "Leptokurtic Distribution"
    elif excess_kurtosis < 0:
        title = "Platykurtic Distribution"
    else:
        title = "Mesokurtic Distribution"

    plot_distribution_with_normal_fit(data, mean, std_dev, excess_kurtosis, title)

if __name__ == "__main__":
    main()
```

### Computing Kurtosis in Python

SciPy provides convenient functions that compute excess kurtosis directly:

```python
from scipy import stats
import numpy as np

data = np.random.normal(0, 1, 10000)

# All three return excess kurtosis (kurtosis - 3)
print(stats.kurtosis(data))
print(stats.describe(data).kurtosis)
```

---

## Summary

Skewness and kurtosis extend the description of a distribution beyond its center and spread. Skewness reveals directional asymmetry, guiding the choice between mean and median as a representative center. Kurtosis quantifies tail behavior, which is critical in risk management and finance where extreme events (heavy tails) have outsized consequences.

## Exercises

**Exercise 1.**
For $\{1, 2, 3, 4, 10\}$: compute (a) sample mean and SD; (b) population skewness; (c) interpret the sign.

??? success "Solution to Exercise 1"
    (a) $\bar x = 4$. Squared deviations $9, 4, 1, 0, 36$; sum $= 50$; $m_2 = 50/5 = 10$, $s_{\text{pop}} = \sqrt{10} \approx 3.162$.

    (b) Cubed deviations $-27, -8, -1, 0, 216$; sum $= 180$; $m_3 = 180/5 = 36$. Then

    $$
    g_1 = \frac{m_3}{m_2^{3/2}} = \frac{36}{10^{3/2}} \approx 1.138
    $$

    (c) $g_1 > 0$ → right-skewed. The single value 10 produces a cubed deviation $+216$ that dominates the numerator. The four smaller values contribute only $-36$ collectively. One long right tail produces positive skewness.

---

**Exercise 2.**
Two datasets: A = $\{4,5,5,6,6,6,7,7,8\}$, B = $\{1,2,5,6,6,6,7,10,11\}$. Verify equal means, then compute the population excess kurtosis of each. Both turn out equal — explain why.

??? success "Solution to Exercise 2"
    Both means: $54/9 = 6$.

    **Dataset A:** $m_2 = 12/9 = 4/3$, $m_4 = 36/9 = 4$. Excess kurtosis $= 4/(4/3)^2 - 3 = 4 \cdot 9/16 - 3 = 2.25 - 3 = -0.75$.

    **Dataset B:** $m_2 = 84/9 = 28/3$, $m_4 = 1764/9 = 196$. Excess kurtosis $= 196 / (28/3)^2 - 3 = 196 \cdot 9 / 784 - 3 = 2.25 - 3 = -0.75$.

    **Why equal:** kurtosis is the ratio $m_4 / m_2^2$ — a *standardized* fourth moment. Dataset B has values farther from the mean (range 1 to 11 vs. 4 to 8), but its $m_2$ is also proportionally larger. Kurtosis measures tail heaviness *relative to the distribution's own variance*, so multiplying every observation by a constant leaves kurtosis unchanged. The two datasets have the same *shape* up to rescaling.

---

**Exercise 3.**
Define the **third standardized central moment** as $\mu_3 / \sigma^3$. Show that this is **scale-invariant** (rescaling all observations leaves it unchanged) and **shift-invariant** (adding a constant leaves it unchanged). What does this say about the population skewness coefficient?

??? success "Solution to Exercise 3"
    Let $Y = a X + b$ with $a > 0$. Then $\mu_Y = a\mu_X + b$, $\sigma_Y = a \sigma_X$, and

    $$
    \mu_3(Y) = \mathbb{E}[(Y - \mu_Y)^3] = \mathbb{E}[(a(X - \mu_X))^3] = a^3 \mu_3(X)
    $$

    Therefore

    $$
    \frac{\mu_3(Y)}{\sigma_Y^3} = \frac{a^3 \mu_3(X)}{a^3 \sigma_X^3} = \frac{\mu_3(X)}{\sigma_X^3}
    $$

    The skewness is **affine-invariant**: it depends only on the shape of the distribution, not its location or scale. Two distributions with the same shape (e.g., all normal distributions) have the same skewness regardless of their parameters $(\mu, \sigma)$. This invariance is what allows skewness to compare across data sets with different units.

    Similarly, the **fourth standardized moment** (kurtosis) is also affine-invariant, which is why both A and B in Exercise 2 produce identical excess kurtosis despite the different spreads.

---

**Exercise 4.**
The **Pearson median skewness** is $\gamma = (\mu - \tilde\mu) / \sigma$ where $\tilde\mu$ is the median. Why is this measure more robust than the classical skewness, and what is the typical relationship between $\mu$, $\tilde\mu$, and the mode in a unimodal right-skewed distribution?

??? success "Solution to Exercise 4"
    **More robust:** the classical skewness involves cubed deviations, so a single outlier far from the mean contributes $|x - \bar x|^3$, which can be enormous. Pearson's median skewness uses the median (breakdown 50%) and is less sensitive to outliers. The standard deviation in the denominator is still sensitive — for a fully robust skewness measure, replace $\sigma$ with a robust scale like MAD.

    **Typical ordering in a unimodal right-skewed distribution:**

    $$
    \text{Mode} < \text{Median} < \text{Mean}
    $$

    Intuition: the mode is at the peak of the density; the median is at the half-mass cutoff; the mean is pulled toward the long right tail. For a left-skewed distribution the ordering reverses: $\text{Mean} < \text{Median} < \text{Mode}$. For a perfectly symmetric unimodal distribution, all three coincide.

    This ordering is widely used as a heuristic skewness diagnostic but it can fail in multimodal or pathologically skewed distributions.

---

**Exercise 5.**
The kurtosis of a standard normal is 3 (excess kurtosis 0). Heavy-tailed distributions like the $t$-distribution with $\nu$ degrees of freedom have excess kurtosis $6/(\nu - 4)$ (for $\nu > 4$). Compute and interpret this for $\nu = 5, 10, 30, 100$.

??? success "Solution to Exercise 5"
    Excess kurtosis $= 6/(\nu - 4)$:

    | $\nu$ | excess kurtosis |
    |---|---|
    | 5 | 6 |
    | 10 | 1 |
    | 30 | 0.231 |
    | 100 | 0.0625 |

    **Interpretation:** as $\nu$ grows, the $t$-distribution approaches the normal, so its excess kurtosis $\to 0$. At $\nu = 5$, the tails are far heavier than normal (excess kurtosis 6 is enormous). At $\nu = 30$, the tails are nearly normal (excess $\approx 0.23$). At $\nu = 100$, the $t$ is practically indistinguishable from normal in kurtosis.

    **Practical implications:**

    - Stock returns are often modeled with $t$-distribution at $\nu \approx 4$–$8$ (heavy tails matching observed crashes).
    - Hypothesis tests using $t$ critical values converge to $z$ critical values for $\nu \gtrsim 30$ — this is the rule of thumb for using the normal approximation.
    - When sample kurtosis is large (e.g., $g_2 > 2$), suspect that the data has heavier tails than normal and standard CIs based on normality may have under-coverage.

---

**Exercise 6.**
**Sample skewness and kurtosis are themselves random**, and their sampling variability is large for small samples. What is approximately the standard error of the sample skewness for an i.i.d. sample of size $n$ from a normal distribution? Use this to discuss when a "nonzero" sample skewness is statistically meaningful.

??? success "Solution to Exercise 6"
    For an i.i.d. normal sample, the asymptotic standard error of the sample skewness is

    $$
    \mathrm{SE}(g_1) \approx \sqrt{\frac{6 n (n-1)}{(n-2)(n+1)(n+3)}} \approx \sqrt{\frac{6}{n}}
    $$

    A sample skewness $|g_1| > 2 \cdot \mathrm{SE}$ is needed for evidence of departure from normal skewness. Values:

    | $n$ | approximate SE | "significant" threshold |
    |---|---|---|
    | 20 | 0.55 | $\pm 1.10$ |
    | 50 | 0.35 | $\pm 0.69$ |
    | 100 | 0.24 | $\pm 0.49$ |
    | 1000 | 0.077 | $\pm 0.15$ |

    **Implication:** with $n = 50$, a sample skewness of 0.4 is *not* statistically distinguishable from zero — it could easily arise from normal data by chance. Practitioners who report "skewness = 0.4, so the data is right-skewed" without considering sampling variability are over-interpreting. Always (a) plot the data, (b) report SE alongside the point estimate, (c) prefer goodness-of-fit tests (Shapiro–Wilk, Anderson–Darling) for formal normality assessment. Similar caution applies to sample kurtosis, whose SE is even larger.
