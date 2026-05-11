# Robust Estimators Comparison

## Overview

Robust estimators resist the influence of outliers and departures from distributional assumptions. While the sample mean and standard deviation are optimal for normal data, they can be severely distorted by even a few extreme observations. This page compares robust alternatives for both location (trimmed mean, weighted mean, median) and scale (MAD, IQR) estimation, demonstrating their behavior under clean and contaminated data.

## Location Estimators

### Trimmed Mean

The **$\alpha$-trimmed mean** removes the lowest and highest $\alpha$ fraction of the sorted data and averages the remainder:

$$\bar{X}_\alpha = \frac{1}{n - 2k}\sum_{i=k+1}^{n-k} X_{(i)}$$

where $k = \lfloor n\alpha \rfloor$ and $X_{(i)}$ denotes the $i$-th order statistic.

```python
import numpy as np

def trimmed_mean(data, proportion=0.1):
    x = np.sort(data)
    n = len(x)
    k = int(np.floor(n * proportion))
    if k == 0:
        return x.mean()
    return x[k:-k].mean()
```

### Weighted Mean and Weighted Median

The **weighted mean** assigns different importances to observations:

$$\bar{X}_w = \frac{\sum_{i=1}^n w_i X_i}{\sum_{i=1}^n w_i}$$

The **weighted median** is the value $m$ such that the cumulative weight on each side is at most 50%. It is more robust than the weighted mean.

```python
def weighted_mean(data, weights):
    return np.sum(data * weights) / np.sum(weights)

def weighted_median(data, weights):
    order = np.argsort(data)
    sorted_data = data[order]
    sorted_w = weights[order]
    cum_w = np.cumsum(sorted_w) / np.sum(sorted_w)
    idx = np.searchsorted(cum_w, 0.5)
    return sorted_data[idx]
```

## Scale Estimators

### Median Absolute Deviation

The **MAD** (Median Absolute Deviation) is a robust measure of spread:

$$\text{MAD} = \text{median}(|X_i - \text{median}(X)|)$$

For normal data, $\text{MAD} \approx 0.6745\sigma$, so $\hat{\sigma}_{\text{MAD}} = 1.4826 \times \text{MAD}$ is a consistent estimator of $\sigma$.

```python
def mad(data):
    med = np.median(data)
    return np.median(np.abs(data - med))
```

### Interquartile Range

The **IQR** (Interquartile Range) is another robust scale measure:

$$\text{IQR} = Q_3 - Q_1$$

For normal data, $\text{IQR} \approx 1.349\sigma$, so $\hat{\sigma}_{\text{IQR}} = \text{IQR}/1.349$ estimates $\sigma$.

## Comparison Under Contamination

The following code compares location and scale estimators on clean normal data and data contaminated with extreme outliers.

```python
np.random.seed(42)

# Clean data
clean = np.random.normal(loc=50, scale=10, size=100)

# Contaminated data: add 5 extreme outliers
outliers = np.array([200, 250, 300, -100, -150])
contaminated = np.concatenate([clean, outliers])

for label, data in [("Clean", clean), ("Contaminated", contaminated)]:
    print(f"\n{label} data (n = {len(data)}):")
    print(f"  Mean             = {data.mean():.2f}")
    print(f"  Median           = {np.median(data):.2f}")
    print(f"  Trimmed mean 10% = {trimmed_mean(data, 0.10):.2f}")
    print(f"  Trimmed mean 20% = {trimmed_mean(data, 0.20):.2f}")
    print(f"  Std dev          = {data.std(ddof=1):.2f}")
    print(f"  IQR              = {np.percentile(data, 75) - np.percentile(data, 25):.2f}")
    print(f"  MAD              = {mad(data):.2f}")
```

!!! note "Impact of outliers"
    Five outliers out of 105 observations shift the mean by about 5 units (from ~50 to ~55) and nearly double the standard deviation. The median, trimmed mean, IQR, and MAD are barely affected.

## Robustness Under Progressive Contamination

To visualize the breakdown, we progressively add outliers (value = 300) to 100 clean observations and track each estimator.

```python
import matplotlib.pyplot as plt

n_out_range = range(0, 21)
means, medians, trims = [], [], []
stds, iqrs, mads_list = [], [], []

for n_out in n_out_range:
    extra = np.full(n_out, 300.0)
    data = np.concatenate([clean, extra])
    means.append(data.mean())
    medians.append(np.median(data))
    trims.append(trimmed_mean(data, 0.10))
    stds.append(data.std(ddof=1))
    iqrs.append(np.percentile(data, 75) - np.percentile(data, 25))
    mads_list.append(mad(data))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(list(n_out_range), means, 'o-', label='Mean', markersize=4)
ax1.plot(list(n_out_range), medians, 's-', label='Median', markersize=4)
ax1.plot(list(n_out_range), trims, 'D-', label='Trimmed Mean (10%)', markersize=4)
ax1.set_xlabel('Number of outliers added (value = 300)')
ax1.set_ylabel('Estimated location')
ax1.set_title('Location Estimators vs Outlier Count')
ax1.legend()

ax2.plot(list(n_out_range), stds, 'o-', label='Std Dev', markersize=4)
ax2.plot(list(n_out_range), iqrs, 's-', label='IQR', markersize=4)
ax2.plot(list(n_out_range), mads_list, 'D-', label='MAD', markersize=4)
ax2.set_xlabel('Number of outliers added (value = 300)')
ax2.set_ylabel('Estimated scale')
ax2.set_title('Scale Estimators vs Outlier Count')
ax2.legend()

plt.tight_layout()
plt.show()
```

## Breakdown Point

The **breakdown point** of an estimator is the maximum fraction of arbitrary contamination it can tolerate before giving an arbitrarily bad result.

| Estimator | Breakdown Point |
|-----------|----------------|
| Mean | $0\%$ (a single extreme value can make it arbitrary) |
| Median | $50\%$ |
| $\alpha$-Trimmed Mean | $\alpha$ (e.g., 10% for 10% trimming) |
| Standard Deviation | $0\%$ |
| MAD | $50\%$ |
| IQR | $25\%$ |

!!! info "Robustness vs efficiency"
    Robust estimators sacrifice some efficiency under the assumed model (e.g., the median is only 63.7% as efficient as the mean for normal data) in exchange for protection against model violations. The trimmed mean provides a useful middle ground: nearly as efficient as the mean under normality while offering meaningful robustness.

## Interpretation

- The **mean and standard deviation** are optimal for clean normal data but are arbitrarily sensitive to outliers (zero breakdown point).
- The **median and MAD** have 50% breakdown points — they remain informative even when nearly half the data is contaminated.
- The **trimmed mean** provides a tunable compromise: small trimming fractions preserve efficiency while providing modest robustness.
- **Weighted estimators** are useful when observation quality varies, but the weighted mean is still sensitive to outliers unless the weights are themselves robustly determined.
- In practice, it is good practice to compute both classical and robust estimators. If they differ substantially, further investigation of the data is warranted.

## Exercises

**Exercise 1.**
Compute the mean, median, and 10% trimmed mean for the dataset $\{1, 2, 3, 4, 5, 6, 7, 8, 9, 100\}$. Which estimator best represents the "typical" value?

??? success "Solution to Exercise 1"
    **Mean:** $\frac{1+2+3+4+5+6+7+8+9+100}{10} = \frac{145}{10} = 14.5$

    **Median:** The sorted data has 10 values, so the median is the average of the 5th and 6th: $(5+6)/2 = 5.5$.

    **10% Trimmed Mean:** With $n = 10$ and $\alpha = 0.10$, we trim $k = \lfloor 10 \times 0.10 \rfloor = 1$ observation from each end. Remaining: $\{2, 3, 4, 5, 6, 7, 8, 9\}$. Mean: $44/8 = 5.5$.

    The mean (14.5) is pulled far above the bulk of the data by the single outlier at 100. The median (5.5) and trimmed mean (5.5) both ignore this outlier and better represent the typical value. $\square$

---

**Exercise 2.**
Prove that the breakdown point of the sample median is $\lfloor(n-1)/2\rfloor / n$, which approaches 50% for large $n$.

??? success "Solution to Exercise 2"
    Consider $n$ observations $x_1 \leq x_2 \leq \cdots \leq x_n$. The median is approximately $x_{(\lceil n/2 \rceil)}$.

    To make the median arbitrarily large, we need to replace enough observations so that more than half the data is extreme. Specifically, we need to replace $\lceil n/2 \rceil$ observations with values approaching $+\infty$. Then the new median will be one of the extreme values.

    With only $\lfloor (n-1)/2 \rfloor$ replacements, at least $\lceil (n+1)/2 \rceil$ original observations remain. The median must lie among these original values and hence remains bounded.

    Therefore the breakdown point is $\lfloor (n-1)/2 \rfloor / n$. For $n$ odd, this is $(n-1)/(2n)$; for $n$ even, this is $(n-2)/(2n)$. In both cases, as $n \to \infty$, the breakdown point approaches $1/2 = 50\%$. $\square$

---

**Exercise 3.**
For $X \sim N(\mu, \sigma^2)$, show that $\text{MAD} = \mathcal{N}^{-1}(3/4) \cdot \sigma \approx 0.6745\sigma$, and hence that $1.4826 \times \text{MAD}$ is a consistent estimator of $\sigma$.

??? success "Solution to Exercise 3"
    For $X \sim N(\mu, \sigma^2)$, the deviations $|X - \mu|$ follow a half-normal distribution. The median of $|X - \mu|$ is the value $m$ such that:

    $$P(|X - \mu| \leq m) = \frac{1}{2}$$

    This means $P(-m \leq X - \mu \leq m) = 1/2$, i.e., $\mathcal{N}(m/\sigma) - \mathcal{N}(-m/\sigma) = 1/2$, so $2\mathcal{N}(m/\sigma) - 1 = 1/2$, giving $\mathcal{N}(m/\sigma) = 3/4$.

    Therefore $m = \sigma \mathcal{N}^{-1}(3/4) \approx 0.6745\sigma$.

    The population MAD (using the true median $\mu$) equals $0.6745\sigma$. By the consistency of the sample median and the continuous mapping theorem, the sample MAD converges to the population MAD.

    Therefore $\hat{\sigma} = \text{MAD}/0.6745 = 1.4826 \times \text{MAD}$ is a consistent estimator of $\sigma$. $\square$

---

**Exercise 4.**
A dataset of 200 observations has mean 50, standard deviation 10, median 49, and MAD 6.5. Is there evidence of outliers or non-normality? Justify your answer using the ratio of classical to robust estimators.

??? success "Solution to Exercise 4"
    Under normality, we expect:

    - Mean $\approx$ Median: Here $50 \approx 49$, close agreement. Slight right skew.
    - $\text{MAD} \approx 0.6745\sigma$: Expected MAD $= 0.6745 \times 10 = 6.745$. Observed MAD $= 6.5$. Ratio: $6.5/6.745 = 0.964$.
    - $\hat{\sigma}_{\text{MAD}} = 1.4826 \times 6.5 = 9.637$ vs $s = 10$. Ratio: $s/\hat{\sigma}_{\text{MAD}} = 10/9.637 = 1.038$.

    The classical standard deviation is only about 3.8% higher than the robust estimate. If there were significant outliers, we would expect $s$ to be much larger than $\hat{\sigma}_{\text{MAD}}$ (ratios of 1.2 or more would be concerning).

    The near-agreement between all pairs of estimators (mean/median, $s$/MAD-based $\hat{\sigma}$) suggests the data is approximately normal with no major outlier contamination. The slight discrepancies are within normal sampling variation for $n = 200$. $\square$

---

**Exercise 5.**
Explain the robustness-efficiency tradeoff for the $\alpha$-trimmed mean. For normal data with $n = 100$, what is the asymptotic relative efficiency of the 10% trimmed mean compared to the sample mean?

??? success "Solution to Exercise 5"
    **Tradeoff:** Increasing $\alpha$ improves robustness (higher breakdown point) but decreases efficiency under the normal model (discarding good data). At $\alpha = 0$ we get the mean (fully efficient, zero robustness); at $\alpha \to 0.5$ we approach the median (maximum robustness, 63.7% efficiency for normal data).

    For the $\alpha$-trimmed mean with normal data, the asymptotic variance is:

    $$\text{Var}(\bar{X}_\alpha) \approx \frac{\sigma^2}{n(1 - 2\alpha)^2}\left[(1-2\alpha) + 2\alpha\phi(\mathcal{N}^{-1}(\alpha))^2/\alpha^2 - \text{correction}\right]$$

    A simpler approximation for the asymptotic relative efficiency (ARE) of the $\alpha$-trimmed mean relative to the sample mean under normality is:

    $$\text{ARE} = \frac{\text{Var}(\bar{X})}{\text{Var}(\bar{X}_\alpha)}$$

    For $\alpha = 0.10$ (10% trimming), the ARE is approximately **0.95** (the trimmed mean uses about 95% of the information in the data).

    This means we lose only about 5% efficiency under normality while gaining a breakdown point of 10% (the estimator can tolerate up to 10% contamination). This is generally considered an excellent tradeoff, which is why the 10% trimmed mean is a popular default in many applications. $\square$
