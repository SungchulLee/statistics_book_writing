# Testing Kendall's tau

Kendall's tau measures monotonic association by comparing concordant and discordant pairs. To determine whether the observed value of $\tau$ is statistically significant, we test the null hypothesis that the two variables are independent. This section presents the hypothesis test, its null distribution for both small and large samples, and its implementation.

---

## Hypotheses

The standard test is:

$$
H_0\!: \tau = 0 \quad \text{vs} \quad H_1\!: \tau \neq 0
$$

where $\tau$ is the population Kendall tau coefficient. Under $H_0$, the two variables are independent, meaning all orderings of one variable relative to the other are equally likely.

One-sided alternatives ($H_1\!: \tau > 0$ or $H_1\!: \tau < 0$) are used when the direction of the monotonic trend is specified in advance.

---

## The Test Statistic S

The test is based on the Kendall $S$ statistic:

$$
S = C - D
$$

where $C$ is the number of concordant pairs and $D$ is the number of discordant pairs. The relationship to tau-a is

$$
\tau_a = \frac{S}{\binom{n}{2}}
$$

---

## Exact Distribution for Small Samples

Under $H_0$, all $n!$ permutations of the $Y$-ranks are equally likely. The exact distribution of $S$ can be computed by enumerating all permutations or using a recursive algorithm.

For small samples (typically $n \le 10$), the exact p-value is computed by finding the proportion of permutations that yield a value of $|S|$ at least as large as the observed value:

$$
p = P(|S| \ge |s_{\text{obs}}| \mid H_0)
$$

Exact critical values of $S$ are available in statistical tables for small $n$.

---

## Normal Approximation for Large Samples

For larger samples, $S$ is approximately normal under $H_0$. The expected value and variance of $S$ are:

$$
\mathbb{E}[S] = 0
$$

$$
\text{Var}(S) = \frac{n(n-1)(2n+5)}{18}
$$

When there are no ties, the standardized test statistic is

$$
Z = \frac{S}{\sqrt{\frac{n(n-1)(2n+5)}{18}}}
$$

Under $H_0$, $Z$ follows approximately a standard normal distribution for large $n$ (typically $n \ge 10$ is sufficient).

### Correction for Ties

When ties are present, the variance is adjusted:

$$
\text{Var}(S) = \frac{n(n-1)(2n+5) - \sum_t t_x(t_x-1)(2t_x+5) - \sum_u t_y(t_y-1)(2t_y+5)}{18}
$$

$$

+ \frac{\sum_t t_x(t_x-1)(t_x-2) \cdot \sum_u t_y(t_y-1)(t_y-2)}{9n(n-1)(n-2)}
$$

$$

+ \frac{\sum_t t_x(t_x-1) \cdot \sum_u t_y(t_y-1)}{2n(n-1)}
$$

where $t_x$ is the size of each tied group in $X$ and $t_y$ is the size of each tied group in $Y$. Most software handles this automatically.

---

## Decision Rule

For a two-sided test at significance level $\alpha$:

- **Small samples**: reject $H_0$ if the exact p-value $< \alpha$.
- **Large samples**: reject $H_0$ if $|Z| > z_{\alpha/2}$.

---

## Example

Consider the five observations from the [Kendall's Tau](../correlation/kendall.md) section, where we computed $S = C - D = 3 - 7 = -4$ with $n = 5$.

Using the normal approximation:

$$
\text{Var}(S) = \frac{5 \times 4 \times 15}{18} = \frac{300}{18} = 16.67
$$

$$
Z = \frac{-4}{\sqrt{16.67}} = \frac{-4}{4.083} = -0.980
$$

The two-sided p-value is $2 \times P(Z < -0.980) = 2 \times 0.164 = 0.327$. We do not reject $H_0$ at $\alpha = 0.05$; there is insufficient evidence of a monotonic association.

For $n = 5$, the exact test would be more appropriate. The exact two-sided p-value (from permutation enumeration) is approximately $0.483$, confirming the conclusion.

---

## Power Comparison with Spearman's Test

Kendall's test and Spearman's test are both nonparametric tests for monotonic association. Their relative power depends on the alternative:

- For **bivariate normal** data, the **asymptotic relative efficiency** (ARE) of Kendall's test relative to Spearman's test is

    $$
    \text{ARE} = \frac{9}{4\pi^2 - 36} \approx 0.98
    $$

    meaning both tests have nearly identical power.

- For **small samples**, Kendall's test often performs slightly better due to the more regular distribution of $\tau$ compared to $r_s$.

- In practice, the choice between the two tests matters little for the hypothesis testing conclusion. Kendall's tau is sometimes preferred for its clearer probabilistic interpretation ($P(\text{concordant}) - P(\text{discordant})$).

---

## Computation in Python

```python
import numpy as np
from scipy import stats

x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 4, 2, 1])

# Kendall's tau test
tau, p_value = stats.kendalltau(x, y)
print(f"Kendall tau = {tau:.4f}")
print(f"p-value     = {p_value:.4f}")

# Manual computation for verification
n = len(x)
S = 0
for i in range(n):
    for j in range(i + 1, n):
        S += np.sign(x[j] - x[i]) * np.sign(y[j] - y[i])

var_S = n * (n - 1) * (2 * n + 5) / 18
Z = S / np.sqrt(var_S)
p_manual = 2 * (1 - stats.norm.cdf(abs(Z)))
print(f"S = {S}, Z = {Z:.4f}, Manual p = {p_manual:.4f}")
```

The `scipy.stats.kendalltau` function computes tau-b and handles ties in the variance formula automatically.

---

## Summary

The hypothesis test for Kendall's $\tau$ determines whether the observed number of concordant minus discordant pairs is significantly different from zero. For small samples, the exact permutation distribution is used; for larger samples, a normal approximation based on $\text{Var}(S) = n(n-1)(2n+5)/18$ is employed. The test is distribution-free and has power comparable to Spearman's test. Kendall's test is preferred in small samples or when the probabilistic interpretation of $\tau$ is important.

## Exercises

**Exercise 1.**
For $n = 8$ observations with Kendall's $\tau = 0.43$, test $H_0: \tau = 0$ at $\alpha = 0.05$ using the normal approximation.

??? success "Solution to Exercise 1"
    The standard error under $H_0$ is:

    $$
    \text{SE}(\tau) = \sqrt{\frac{2(2n+5)}{9n(n-1)}} = \sqrt{\frac{2(21)}{9 \times 8 \times 7}} = \sqrt{\frac{42}{504}} = \sqrt{0.08333} = 0.2887
    $$

    The test statistic is:

    $$
    Z = \frac{\tau}{\text{SE}(\tau)} = \frac{0.43}{0.2887} = 1.489
    $$

    For a two-sided test at $\alpha = 0.05$, the critical value is 1.96. Since $|Z| = 1.49 < 1.96$, we fail to reject $H_0$. There is insufficient evidence of monotonic association.

    Note: with $n = 8$, the normal approximation may be rough. An exact permutation test would be more reliable.

---

**Exercise 2.**
Explain why the exact distribution of Kendall's $\tau$ under $H_0$ is distribution-free and describe how it can be computed.

??? success "Solution to Exercise 2"
    Under $H_0: \tau = 0$ (independence of $X$ and $Y$), the $Y$-ranks are equally likely to be any permutation of $(1, 2, \dots, n)$, regardless of the distribution of $X$ or $Y$. This means the distribution of the number of concordant pairs $C$ (and hence $\tau$) depends only on $n$, not on the underlying distributions.

    The exact distribution can be computed by enumerating all $n!$ permutations of the $Y$-ranks and computing $\tau$ for each. For small $n$, this is feasible (and implemented in statistical software). For large $n$, the normal approximation $Z = \tau/\text{SE}(\tau)$ is used, where the variance formula under $H_0$ involves $n$ alone.

---

**Exercise 3.**
Compare the power of Kendall's $\tau$ test versus Spearman's $\rho$ test for detecting monotonic association. Which is generally more powerful?

??? success "Solution to Exercise 3"
    In most settings, the Spearman test is slightly more powerful than the Kendall test for detecting monotonic associations, because Spearman's $\rho$ uses rank values (which carry more information than pairwise ordinal comparisons) and has a larger variance separation between $H_0$ and $H_a$.

    However, the differences are typically small (asymptotic relative efficiency of Kendall vs. Spearman is close to 1). Kendall's $\tau$ has advantages in other respects: simpler variance formula, cleaner probabilistic interpretation, and better performance with ties.

    The choice between them is often based on convention within a field or the specific properties desired rather than power considerations.

---

**Exercise 4.**
A researcher computes Kendall's $\tau = -0.12$ with $n = 200$ and obtains $p = 0.03$. Interpret this result in terms of both statistical and practical significance.

??? success "Solution to Exercise 4"
    **Statistical significance:** With $p = 0.03 < 0.05$, we reject $H_0: \tau = 0$. There is statistically significant evidence of a negative monotonic association.

    **Practical significance:** $\tau = -0.12$ is a weak association. The probability interpretation: for a randomly chosen pair of observations, the probability of concordance exceeds the probability of discordance by only $0.12$ (i.e., $P(\text{concordant}) - P(\text{discordant}) = -0.12$, meaning discordance is slightly more common). The effect is detectable only because $n = 200$ provides high power.

    As with Pearson's $r$, large samples can detect trivially small associations. The researcher should report the effect size ($\tau = -0.12$) alongside the p-value and discuss whether this magnitude is meaningful in the applied context.
