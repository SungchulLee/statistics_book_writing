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
