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
