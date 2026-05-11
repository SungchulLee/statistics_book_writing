# Trimmed and Winsorized Means

The sample mean is the most efficient estimator of the population mean under Normality, but a single extreme observation can shift it arbitrarily far from the true center. Robust estimators sacrifice a small amount of efficiency under ideal conditions in exchange for stability when the data deviate from the assumed model. This section introduces two such estimators — the trimmed mean and the Winsorized mean — and compares their trade-offs.

## Trimmed Mean

The idea behind trimming is simple: remove the most extreme observations on both ends before averaging. By discarding the values most likely to be outliers or heavy-tail artifacts, the estimator resists contamination while still using the bulk of the data.

Given a sample $X_1, \ldots, X_n$, let $X_{(1)} \leq X_{(2)} \leq \cdots \leq X_{(n)}$ denote the **order statistics** (the observations sorted from smallest to largest). The $\alpha$-trimmed mean removes the smallest and largest $\lfloor n\alpha \rfloor$ observations and averages the rest:

$$
\bar{X}_{\text{trim}(\alpha)} = \frac{1}{n - 2\lfloor n\alpha \rfloor} \sum_{i=\lfloor n\alpha \rfloor + 1}^{n - \lfloor n\alpha \rfloor} X_{(i)}
$$

Setting $\alpha = 0$ recovers the ordinary sample mean, while $\alpha = 0.5$ yields the median.

!!! example "Trimmed Mean with 20% Trimming"
    Consider the sample $\{1, 3, 5, 7, 100\}$ with $n = 5$ and $\alpha = 0.2$. Since $\lfloor 5 \times 0.2 \rfloor = 1$, the trimmed mean removes the smallest value ($1$) and the largest value ($100$), then averages the remaining three: $\bar{X}_{\text{trim}(0.2)} = (3 + 5 + 7)/3 = 5.0$. The ordinary sample mean is $(1 + 3 + 5 + 7 + 100)/5 = 23.2$, showing how a single outlier inflates the untrimmed estimate.

## Winsorized Mean

Rather than discarding extreme observations, the Winsorized mean replaces them with the nearest non-extreme value. This retains the original sample size, which can simplify variance estimation.

For trimming fraction $\alpha$, let $k = \lfloor n\alpha \rfloor$. The $\alpha$-Winsorized mean replaces the $k$ smallest observations with $X_{(k+1)}$ and the $k$ largest with $X_{(n-k)}$, then averages all $n$ values:

$$
\bar{X}_{\text{win}(\alpha)} = \frac{1}{n}\left( k \cdot X_{(k+1)} + \sum_{i=k+1}^{n-k} X_{(i)} + k \cdot X_{(n-k)} \right)
$$

!!! example "Winsorized Mean with 20% Winsorizing"
    Using the same sample $\{1, 3, 5, 7, 100\}$ with $\alpha = 0.2$ ($k = 1$), the Winsorized mean replaces $1$ with $3$ and $100$ with $7$: $\bar{X}_{\text{win}(0.2)} = (3 + 3 + 5 + 7 + 7)/5 = 5.0$. The result matches the trimmed mean in this case, though they generally differ for larger samples.

## Breakdown Point and Efficiency

The **breakdown point** of an estimator is the largest fraction of observations that can be replaced by arbitrary values before the estimator produces an unbounded result. A higher breakdown point indicates greater robustness.

| Estimator | Breakdown Point | ARE vs Sample Mean (Normal) |
|---|---|---|
| Sample mean | 0% | 100% |
| 10% trimmed mean | 10% | ~97% |
| 20% trimmed mean | 20% | ~93% |
| Median (50% trimmed) | 50% | ~64% |

The efficiency column reports the asymptotic relative efficiency compared to the sample mean under a Normal distribution. The 10% trimmed mean loses only about 3% efficiency under Normality while gaining substantial protection against outliers. The median achieves the maximum breakdown point of 50% but at a cost of roughly 36% efficiency loss under Normality.

!!! tip "Choosing a Trimming Fraction"
    In practice, $\alpha = 0.1$ (10% trimming) or $\alpha = 0.2$ (20% trimming) offer a good balance between robustness and efficiency. When the distribution is expected to be heavy-tailed, the efficiency loss relative to the sample mean under Normality is more than offset by the gain in stability.

## Exercises

**Exercise 1.**
Bootstrap SEs for $n = 30$ Gamma(3, 2). Compare with theoretical/asymptotic formulas for mean and median.

??? success "Solution to Exercise 1"
    ```python
    import numpy as np
    from scipy import stats
    rng = np.random.default_rng(0)
    n, B = 30, 5000
    x = rng.gamma(shape=3, scale=2, size=n)
    mean_boot, med_boot = np.empty(B), np.empty(B)
    for b in range(B):
        xb = rng.choice(x, size=n, replace=True)
        mean_boot[b] = xb.mean(); med_boot[b] = np.median(xb)
    print(f"SE(mean): boot={mean_boot.std(ddof=1):.3f}, theory={x.std(ddof=1)/np.sqrt(n):.3f}")
    m = np.median(x); f_hat = stats.gaussian_kde(x)(m)[0]
    print(f"SE(med): boot={med_boot.std(ddof=1):.3f}, asymp={1/(2*f_hat*np.sqrt(n)):.3f}")
    ```

    Mean: bootstrap and theory agree closely.

    Median: bootstrap typically exceeds asymptotic formula by 5–15% at $n = 30$ — asymptotic formula assumes large-sample regime not fully reached. Bootstrap captures finite-sample behavior more accurately.

---

**Exercise 2.**
**Breakdown point** of trimmed and Winsorized means. For 20% trimming, what fraction of data must be corrupted to push the estimator arbitrarily far?

??? success "Solution to Exercise 2"
    With 20% trimming ($\alpha = 0.2$, $k = \lfloor 0.2n \rfloor$): the trimmed mean averages observations $X_{(k+1)}$ through $X_{(n-k)}$. To move this average arbitrarily, you must corrupt at least one observation *inside* the trimmed range — i.e., make one of the middle observations extreme.

    More than $k$ corruptions in one tail might be needed to shift the order statistic into the middle. Practical breakdown: $k/n = \alpha = 0.2$ or 20%.

    **General rule:** $\alpha$-trimmed mean has breakdown $\alpha$. Higher trimming = higher robustness = lower efficiency under normal.

---

**Exercise 3.**
**Compute the 20% trimmed mean** of $\{1, 3, 5, 7, 9, 11, 100\}$. Compare with sample mean and median.

??? success "Solution to Exercise 3"
    $n = 7$, $\alpha = 0.2$, $k = \lfloor 1.4 \rfloor = 1$.

    Trim smallest 1 and largest 1: remaining $\{3, 5, 7, 9, 11\}$. Trimmed mean = $(3+5+7+9+11)/5 = 7$.

    Sample mean: $(1+3+5+7+9+11+100)/7 \approx 19.4$. Dominated by outlier.

    Median: $7$ (middle value).

    The trimmed mean and median both give 7; sample mean is misleading. Robust estimators recover the "typical" value of the clean part of the data.

---

**Exercise 4.**
**Asymptotic relative efficiency** of trimmed mean vs sample mean for normal data. Why does efficiency *decrease* with more trimming?

??? success "Solution to Exercise 4"
    Trimming discards information. For normal data, all observations carry mean information; discarding any reduces effective sample size.

    Specifically, the $\alpha$-trimmed mean has asymptotic variance approximately $\sigma^2 \cdot c(\alpha)/n$ where $c(\alpha) > 1$ for $\alpha > 0$. ARE = $1/c(\alpha) < 1$.

    Typical values: $\alpha = 0.1$ gives ARE $\approx 0.97$. $\alpha = 0.2$: $\approx 0.93$. Median ($\alpha = 0.5$): $\approx 0.64$.

    **Conclusion:** small trimming (10-20%) costs little efficiency under normal but provides substantial robustness. Aggressive trimming (toward median) loses efficiency without much robustness gain (breakdown $\alpha = 0.5$ vs 0.2 is rarely needed).

---

**Exercise 5.**
**M-estimators** generalize trimmed means. Briefly describe Huber's $M$-estimator and its breakdown/efficiency.

??? success "Solution to Exercise 5"
    Huber's M-estimator solves $\sum \psi((X_i - \hat\mu)/s) = 0$ where $\psi(u) = u$ for $|u| \le c$ and $\psi(u) = c \cdot \text{sign}(u)$ for $|u| > c$. Combines OLS (in the middle) with median-like behavior (in the tails).

    With $c$ small: behaves like the median. With $c$ large: behaves like the sample mean.

    Standard choice $c = 1.345 \sigma$: 95% efficiency under normal, breakdown point $\approx \min(\alpha, 0.5)$ where $\alpha$ depends on tuning.

    **Why M-estimators dominate trimmed means in practice:**

    - Smooth transition between OLS and median (no sharp cutoff).
    - Can incorporate covariates (robust regression).
    - Better efficiency at intermediate contamination levels.

    The default robust estimator in `statsmodels.RLM` and most modern statistical software.

---

**Exercise 6.**
**When to use trimmed mean vs median vs mean.** Decision flowchart.

??? success "Solution to Exercise 6"
    **Use sample mean** when:

    - Data is approximately symmetric and roughly normal.
    - You need full efficiency (large $n$ or precise estimates needed).
    - Outliers are believed absent or have been pre-processed.

    **Use median** when:

    - Data has extreme outliers or heavy tails.
    - You want maximum breakdown (50%).
    - You don't care about efficiency (large $n$ available).

    **Use trimmed mean** when:

    - Moderate outliers expected (5-20% contamination).
    - Want a compromise between efficiency and robustness.
    - Need a simple, well-known estimator.

    **Use M-estimator** when:

    - Need flexibility (regression context, custom loss).
    - Want better efficiency than trimmed mean at intermediate contamination.
    - Have access to specialized software.

    In practice, default to mean for clean data, median for "outlier-prone" data, trimmed mean (10-20%) when in doubt.
