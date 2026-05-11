# Robust Variance Estimators

## Overview

A single outlier can inflate the sample variance by orders of magnitude, making it unreliable for contaminated data. Consider a dataset of exam scores $\{70, 72, 74, 76, 78\}$ with sample variance $s^2 = 10$. Replacing the last observation with $780$ yields $s^2 = 100{,}490$ --- a ten-thousand-fold increase driven entirely by one corrupted value. This fragility motivates the study of **robust** variance estimators: measures of spread that remain stable even when a substantial fraction of the data is contaminated.

Two key concepts help us evaluate robustness. The **breakdown point** of an estimator is the largest proportion of observations that can be arbitrarily corrupted before the estimator produces unbounded (or completely misleading) results. The **asymptotic relative efficiency** (ARE) measures how much information the estimator extracts relative to the best possible estimator under a specific model --- typically the normal distribution. An ARE of 100% means no information is lost; lower values indicate a price paid for robustness.

## Median Absolute Deviation (MAD)

The sample variance measures average squared deviations from the mean, but both the mean and squaring amplify the effect of outliers. A natural fix is to replace the mean with the median --- itself a robust location estimator --- and to use absolute deviations instead of squared deviations. This leads to the **Median Absolute Deviation** (MAD).

For observations $X_1, X_2, \ldots, X_n$, define

$$
\text{MAD} = \text{Median}\bigl(|X_i - \text{Median}(X)|\bigr)
$$

The MAD first computes the median of the data, then finds the absolute deviation of each observation from that median, and finally takes the median of those deviations.

To use MAD as an estimator of the population standard deviation $\sigma$ under normality, we apply a **consistency factor**:

$$
\hat{\sigma}_{\text{MAD}} = \frac{1}{\mathcal{N}^{-1}(3/4)} \cdot \text{MAD} \approx 1.4826 \cdot \text{MAD}
$$

where $\mathcal{N}^{-1}(3/4) \approx 0.6745$ is the 75th percentile of the standard normal distribution. This scaling ensures that $\hat{\sigma}_{\text{MAD}}$ is a consistent estimator of $\sigma$ when the data are truly normal.

The MAD achieves a **breakdown point of 50%**, meaning that up to half the observations can be arbitrarily corrupted before the estimator breaks down. This is the highest possible breakdown point for any translation-equivariant estimator.

## IQR-based Estimator

The interquartile range (IQR) measures the spread of the middle 50% of the data, which is unaffected by extreme values in either tail. This provides another route to a robust scale estimator.

The IQR-based estimator of $\sigma$ is

$$
\hat{\sigma}_{\text{IQR}} = \frac{\text{IQR}}{2\,\mathcal{N}^{-1}(3/4)} \approx \frac{\text{IQR}}{1.3490}
$$

where $\text{IQR} = Q_3 - Q_1$ is the difference between the 75th and 25th percentiles. Under normality, $Q_3 - Q_1 = 2\,\mathcal{N}^{-1}(3/4)\,\sigma$, so dividing by $2\,\mathcal{N}^{-1}(3/4)$ recovers $\sigma$.

The IQR-based estimator has a **breakdown point of 25%**, since corrupting more than a quarter of the data from either end can shift a quartile arbitrarily.

## Worked Example

Consider the dataset $\{2, 3, 4, 5, 100\}$, where $100$ is an obvious outlier.

**Sample standard deviation:**

The sample mean is $\bar{x} = (2 + 3 + 4 + 5 + 100)/5 = 22.8$, so

$$
s = \sqrt{\frac{1}{4}\sum_{i=1}^{5}(x_i - 22.8)^2} = \sqrt{\frac{(20.8)^2 + (19.8)^2 + (18.8)^2 + (17.8)^2 + (77.2)^2}{4}} \approx 42.15
$$

The outlier inflates the estimate far beyond the spread of the clean observations.

**MAD-based estimate:**

The median is $\text{Median} = 4$. The absolute deviations from the median are $|2-4|, |3-4|, |4-4|, |5-4|, |100-4| = 2, 1, 0, 1, 96$. Sorting gives $\{0, 1, 1, 2, 96\}$, so $\text{MAD} = 1$. Then

$$
\hat{\sigma}_{\text{MAD}} = 1.4826 \times 1 = 1.4826
$$

**IQR-based estimate:**

The quartiles are $Q_1 = 2.5$ and $Q_3 = 52.5$ (using linear interpolation with $n=5$), giving $\text{IQR} = 50.0$. With only five observations, the IQR is still influenced by the outlier because $Q_3$ is pulled toward $100$.

Using simple quartile positions ($Q_1 = 3$, $Q_3 = 5$), we get $\text{IQR} = 2$, so

$$
\hat{\sigma}_{\text{IQR}} = \frac{2}{1.349} \approx 1.48
$$

!!! tip "Practical Guidance"
    With small samples, the quartile computation method matters significantly. The MAD is generally more robust than the IQR-based estimator for small samples with outliers, because the median of absolute deviations is less sensitive to the specific interpolation method used for quartiles.

## Comparison

The following table summarizes the key properties of each estimator.

| Estimator | Breakdown Point | ARE at Normal |
|---|---|---|
| Sample standard deviation $s$ | $1/n \to 0\%$ | 100% |
| MAD-based $\hat{\sigma}_{\text{MAD}}$ | 50% | 37% |
| IQR-based $\hat{\sigma}_{\text{IQR}}$ | 25% | 37% |

The sample standard deviation is the most efficient estimator when the data are truly normal, but it has effectively zero breakdown point --- a single extreme outlier can make it arbitrarily large. The MAD-based estimator sacrifices 63% of efficiency at the normal model in exchange for the maximum possible breakdown point of 50%. The IQR-based estimator offers a compromise, though in practice the MAD is often preferred because of its higher breakdown point.

!!! warning "Robustness--Efficiency Tradeoff"
    No estimator can simultaneously achieve maximum breakdown (50%) and full efficiency (100%) at the normal model. Choosing a robust estimator always involves accepting some loss of efficiency under ideal conditions in exchange for protection against contaminated data.

## Exercises

**Exercise 1.**
**Ledoit-Wolf shrinkage** for covariance. Simulate $p = 30$ assets, $n = 60$. Compare with sample covariance.

??? success "Solution to Exercise 1"
    ```python
    import numpy as np
    from sklearn.covariance import LedoitWolf
    rng = np.random.default_rng(0)
    p, n, R = 30, 60, 500
    Sigma = 0.5**np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    err_s = err_lw = 0.0; shr = []
    for _ in range(R):
        X = rng.multivariate_normal(np.zeros(p), Sigma, n)
        S = np.cov(X, rowvar=False)
        lw = LedoitWolf().fit(X)
        err_s += np.linalg.norm(S - Sigma, "fro")**2
        err_lw += np.linalg.norm(lw.covariance_ - Sigma, "fro")**2
        shr.append(lw.shrinkage_)
    print(f"MSE: sample={err_s/R:.1f}, LW={err_lw/R:.1f}, shrinkage={np.mean(shr):.2f}")
    ```

    Expected: LW MSE roughly 40-60% lower; shrinkage intensity $\sim 0.3$ at $p/n = 0.5$. Higher $p/n$ → more shrinkage. In high-dim regimes ($p \approx n$), sample covariance becomes singular; LW provides invertibility plus better MSE.

---

**Exercise 2.**
**MAD as robust scale.** For $\{2, 4, 6, 8, 10, 100\}$, compute $\text{MAD}$ and the scaled MAD ($1.4826 \cdot \mathrm{MAD}$). Compare with sample SD.

??? success "Solution to Exercise 2"
    Median = 7. Absolute deviations: $|2-7|, |4-7|, |6-7|, |8-7|, |10-7|, |100-7| = 5, 3, 1, 1, 3, 93$.

    Sorted: 1, 1, 3, 3, 5, 93. Median (MAD) = (3 + 3)/2 = 3.

    Scaled MAD = $1.4826 \cdot 3 \approx 4.45$.

    Sample SD = $\sqrt{(\sum(x_i - \bar x)^2)/(n-1)}$ with $\bar x = 21.67$. $\sum(x_i - \bar x)^2 \approx 7188$. $\mathrm{SD} \approx 37.9$.

    SD is dominated by the outlier; MAD captures the typical scale of the bulk of the data.

---

**Exercise 3.**
**Breakdown point of variance estimators.** Compare sample variance, MAD, IQR.

??? success "Solution to Exercise 3"
    **Sample variance:** breakdown 0% (one corrupted observation moves it arbitrarily, especially through squaring).

    **MAD:** breakdown 50%. Median of absolute deviations from the median — both layers of medians are robust.

    **IQR:** breakdown 25%. One quartile can be moved by corrupting 25% of one tail.

    For outlier-prone data, MAD is the most robust; sample variance is the least.

---

**Exercise 4.**
**Trimmed variance.** Define and compute for $\{1, 3, 5, 7, 9, 11, 100\}$ at 20% trimming.

??? success "Solution to Exercise 4"
    20% trimming with $n = 7$: $k = 1$. Trim smallest and largest: remaining $\{3, 5, 7, 9, 11\}$.

    Trimmed mean: 7 (from earlier exercise).

    Trimmed variance: $\frac{1}{n-2k}\sum_{i=k+1}^{n-k}(X_{(i)} - \bar X_{\text{trim}})^2 = (1/5)[(3-7)^2 + (5-7)^2 + (7-7)^2 + (9-7)^2 + (11-7)^2] = 40/5 = 8$.

    Sample variance of original: dominated by 100. Trimmed variance captures the spread of the typical observations.

    Scaling: for normal data, trimmed variance is a biased estimator of $\sigma^2$; scaling constants exist (analog of $1.4826$ for MAD).

---

**Exercise 5.**
**Robust covariance.** Why is the sample covariance matrix sensitive to multivariate outliers, and what are alternative estimators?

??? success "Solution to Exercise 5"
    Sample covariance: $S = (1/(n-1))\sum (X_i - \bar X)(X_i - \bar X)^T$. Squared deviations amplify any outlier in any dimension.

    **Alternatives:**

    - **Minimum Covariance Determinant (MCD):** find subset of $\lceil n/2 \rceil$ observations with smallest determinant of sample covariance. Robust but computationally expensive.
    - **Minimum Volume Ellipsoid (MVE):** find smallest ellipsoid containing $\lceil n/2 \rceil$ observations. Highly robust.
    - **Tukey's bisquare M-estimator:** down-weight observations based on Mahalanobis distance from the center.
    - **Ledoit-Wolf:** shrinks toward a structured target (diagonal). Not robust to outliers but addresses high-dimensional noise.

    For multivariate outlier detection (financial portfolios, multivariate quality control), MCD is the standard. Implemented in `sklearn.covariance.MinCovDet`.

---

**Exercise 6.**
**When to use robust covariance.** Application contexts.

??? success "Solution to Exercise 6"
    **Use robust covariance when:**

    - **Outlier detection:** want to identify points far from the bulk (via Mahalanobis distance using robust $\hat\boldsymbol\Sigma$).
    - **Mahalanobis distance for classification:** outliers shouldn't drive the covariance estimate used in QDA, LDA, or k-NN.
    - **PCA on contaminated data:** robust covariance gives robust principal components.
    - **Multivariate quality control:** detecting unusual patterns (Hotelling's $T^2$).

    **Use sample covariance when:**

    - Data is clean (controlled experiments, simulation).
    - High-dimensional / low-noise regime where shrinkage estimators are preferred.
    - Computational efficiency required.

    **High-dimensional caveat:** when $p > n$, sample covariance is singular regardless of outliers. Need shrinkage (Ledoit-Wolf) or sparse methods (graphical lasso) — and possibly both robust and shrinkage components.
