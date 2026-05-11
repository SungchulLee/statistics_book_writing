# Mean Squared Error

## Introduction

The **Mean Squared Error (MSE)** is the most widely used criterion for evaluating the quality of a statistical estimator. It measures the average squared deviation of an estimator from the true parameter value, capturing both systematic error (bias) and random fluctuation (variance) in a single quantity.

MSE serves as the default loss function in estimation theory, regression analysis, and many optimization problems throughout statistics and quantitative finance.

## Definition

Let $\hat{\theta}$ be an estimator of parameter $\theta$. The **Mean Squared Error** is:

$$\text{MSE}(\hat{\theta}) = E\left[(\hat{\theta} - \theta)^2\right]$$

This is the expected value of the squared difference between the estimator and the true parameter, averaged over all possible samples.

### Equivalent Expressions

The MSE can be computed in several equivalent ways:

$$\text{MSE}(\hat{\theta}) = E[\hat{\theta}^2] - 2\theta E[\hat{\theta}] + \theta^2$$

$$= \text{Var}(\hat{\theta}) + [E[\hat{\theta}]]^2 - 2\theta E[\hat{\theta}] + \theta^2$$

$$= \text{Var}(\hat{\theta}) + (E[\hat{\theta}] - \theta)^2$$

$$= \text{Var}(\hat{\theta}) + [\text{Bias}(\hat{\theta})]^2$$

This last form is the **bias–variance decomposition**.

## Properties of MSE

### Non-negativity

MSE is always non-negative: $\text{MSE}(\hat{\theta}) \geq 0$, with equality only if $\hat{\theta} = \theta$ with probability 1 (the estimator is perfect).

### MSE of Unbiased Estimators

If $\hat{\theta}$ is unbiased ($\text{Bias}(\hat{\theta}) = 0$), then:

$$\text{MSE}(\hat{\theta}) = \text{Var}(\hat{\theta})$$

For unbiased estimators, MSE and variance are identical. Comparing unbiased estimators by MSE is equivalent to comparing them by variance.

### Consistency and MSE

An estimator is **MSE-consistent** if $\text{MSE}(\hat{\theta}_n) \to 0$ as $n \to \infty$. By the decomposition, this requires both:

- $\text{Bias}(\hat{\theta}_n) \to 0$
- $\text{Var}(\hat{\theta}_n) \to 0$

MSE-consistency implies consistency in probability (convergence in probability to $\theta$), by Chebyshev's inequality.

## MSE Comparisons Between Estimators

### Relative Efficiency

The **relative efficiency** of estimator $\hat{\theta}_1$ compared to $\hat{\theta}_2$ is:

$$\text{RE}(\hat{\theta}_1, \hat{\theta}_2) = \frac{\text{MSE}(\hat{\theta}_2)}{\text{MSE}(\hat{\theta}_1)}$$

If $\text{RE} > 1$, then $\hat{\theta}_1$ is more efficient (lower MSE).

For unbiased estimators, this simplifies to:

$$\text{RE}(\hat{\theta}_1, \hat{\theta}_2) = \frac{\text{Var}(\hat{\theta}_2)}{\text{Var}(\hat{\theta}_1)}$$

### Admissibility

An estimator $\hat{\theta}$ is **inadmissible** under MSE if there exists another estimator $\hat{\theta}'$ such that:

$$\text{MSE}(\hat{\theta}') \leq \text{MSE}(\hat{\theta}) \quad \text{for all } \theta$$

with strict inequality for at least one $\theta$. An estimator that is not inadmissible is **admissible**.

**James-Stein result:** When estimating a multivariate normal mean $\mu \in \mathbb{R}^p$ with $p \geq 3$, the sample mean $\bar{X}$ is inadmissible — the James-Stein estimator dominates it uniformly in MSE.

## Worked Examples

### Example 1: MSE of the Sample Mean

Let $X_1, \ldots, X_n \sim \text{iid}$ with mean $\mu$ and variance $\sigma^2$. The sample mean is $\bar{X} = \frac{1}{n}\sum X_i$.

**Bias:** $E[\bar{X}] = \mu$, so $\text{Bias}(\bar{X}) = 0$ (unbiased).

**Variance:** $\text{Var}(\bar{X}) = \sigma^2/n$.

**MSE:** $\text{MSE}(\bar{X}) = 0 + \sigma^2/n = \sigma^2/n$.

### Example 2: MSE of the Naive Variance Estimator

The naive variance estimator is $\tilde{S}^2 = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2$.

For a normal population:

**Bias:** $E[\tilde{S}^2] = \frac{n-1}{n}\sigma^2$, so $\text{Bias}(\tilde{S}^2) = -\sigma^2/n$.

**Variance:** $\text{Var}(\tilde{S}^2) = \frac{2(n-1)}{n^2}\sigma^4$.

**MSE:**

$$\text{MSE}(\tilde{S}^2) = \frac{2(n-1)}{n^2}\sigma^4 + \frac{\sigma^4}{n^2} = \frac{2n-1}{n^2}\sigma^4$$

### Example 3: Comparing Biased vs Unbiased Variance Estimators

The Bessel-corrected estimator is $S^2 = \frac{1}{n-1}\sum (X_i - \bar{X})^2$.

For a normal population:

**MSE of unbiased $S^2$:** $\text{MSE}(S^2) = \text{Var}(S^2) = \frac{2\sigma^4}{n-1}$

**MSE of biased $\tilde{S}^2$:** $\text{MSE}(\tilde{S}^2) = \frac{(2n-1)\sigma^4}{n^2}$

Compare: $\frac{2n-1}{n^2}$ vs $\frac{2}{n-1}$

Cross-multiplying: $(2n-1)(n-1)$ vs $2n^2$, i.e., $2n^2 - 3n + 1$ vs $2n^2$.

Since $-3n + 1 < 0$ for $n > 0$, we have $\text{MSE}(\tilde{S}^2) < \text{MSE}(S^2)$.

**The biased estimator has lower MSE than the unbiased one!** This is a concrete illustration of the bias–variance tradeoff. The optimal estimator (minimizing MSE among estimators of the form $c \cdot \sum(X_i - \bar{X})^2$) divides by $n+1$, not $n$ or $n-1$.

### Example 4: MSE-Optimal Variance Estimator

Consider $\hat{\sigma}^2_c = \frac{1}{c}\sum_{i=1}^n(X_i - \bar{X})^2$ for constant $c > 0$.

For normal populations:

$$\text{MSE}(\hat{\sigma}^2_c) = \left(\frac{n-1}{c} - 1\right)^2 \sigma^4 + \frac{2(n-1)}{c^2}\sigma^4$$

Differentiating with respect to $c$ and setting to zero:

$$c^* = n + 1$$

So the MSE-optimal estimator divides by $n+1$:

$$\hat{\sigma}^2_{n+1} = \frac{1}{n+1}\sum_{i=1}^n (X_i - \bar{X})^2$$

This is biased (underestimates $\sigma^2$) but has lower MSE than both $\tilde{S}^2$ (divide by $n$) and $S^2$ (divide by $n-1$).

## Connections to Other Loss Functions

### Mean Absolute Error (MAE)

$$\text{MAE}(\hat{\theta}) = E\left[|\hat{\theta} - \theta|\right]$$

MAE is less sensitive to outliers than MSE. However, MSE is mathematically more tractable and directly connects to the bias-variance decomposition.

### Risk Function

In decision theory, $\text{MSE}(\hat{\theta})$ is the **risk** of $\hat{\theta}$ under squared error loss $L(\hat{\theta}, \theta) = (\hat{\theta} - \theta)^2$:

$$R(\hat{\theta}, \theta) = E[L(\hat{\theta}, \theta)] = \text{MSE}(\hat{\theta})$$

### Cramér-Rao Lower Bound

For unbiased estimators, the MSE (= variance) is bounded below by the **Cramér-Rao bound**:

$$\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}$$

where $I(\theta) = -E\left[\frac{\partial^2}{\partial\theta^2}\log f(X;\theta)\right]$ is the Fisher information. An unbiased estimator achieving this bound is called **efficient**.

## MSE in Finance

MSE appears throughout quantitative finance:

- **Forecast evaluation**: MSE is the standard metric for comparing return, volatility, or risk forecasts. RMSE = $\sqrt{\text{MSE}}$ puts the error in the same units as the target.
- **Tracking error**: The MSE between a portfolio's returns and its benchmark captures both systematic deviation (bias) and random deviation (variance).
- **Model calibration**: MSE between model-implied and market-observed option prices is the objective function in calibrating volatility models.
- **Regression**: OLS minimizes $\sum(y_i - \hat{y}_i)^2/n$, the in-sample MSE.

## Summary

MSE is the fundamental criterion for evaluating estimator quality. Its decomposition into variance and squared bias reveals the inherent tradeoff in estimation and provides a principled framework for choosing between competing estimators. While unbiasedness is desirable, MSE reminds us that the best estimator minimizes total error — and a little bias can be worth a lot of variance reduction.

## Key Formulas

| Quantity | Formula |
|----------|---------|
| MSE | $E[(\hat{\theta} - \theta)^2]$ |
| Decomposition | $\text{Var}(\hat{\theta}) + [\text{Bias}(\hat{\theta})]^2$ |
| MSE of $\bar{X}$ | $\sigma^2 / n$ |
| Relative Efficiency | $\text{MSE}(\hat{\theta}_2) / \text{MSE}(\hat{\theta}_1)$ |
| Cramér-Rao Bound | $\text{Var}(\hat{\theta}) \geq 1/I(\theta)$ |

## Exercises

**Exercise 1.**
$X_i \sim \mathrm{Uniform}(0, \theta)$ i.i.d. Two estimators: $\hat\theta_1 = 2\bar X$ and $\hat\theta_2 = ((n+1)/n) X_{(n)}$. (a) Both unbiased? (b) Variances? (c) Which has smaller MSE?

??? success "Solution to Exercise 1"
    (a) $\mathbb{E}[X] = \theta/2$, so $\mathbb{E}[\hat\theta_1] = \theta$.

    $\mathbb{E}[X_{(n)}] = n\theta/(n+1)$, so $\mathbb{E}[\hat\theta_2] = ((n+1)/n) \cdot n\theta/(n+1) = \theta$.

    Both unbiased. ✓

    (b) $\mathrm{Var}(\hat\theta_1) = 4 \mathrm{Var}(\bar X) = 4 \theta^2/(12n) = \theta^2/(3n)$.

    $\mathrm{Var}(X_{(n)}) = n\theta^2/[(n+1)^2(n+2)]$. So $\mathrm{Var}(\hat\theta_2) = \theta^2/[n(n+2)]$.

    (c) Both unbiased: MSE = Var. $\hat\theta_2$ has variance $\theta^2/[n(n+2)] = O(1/n^2)$, vs $\hat\theta_1$ at $\theta^2/(3n) = O(1/n)$.

    **$\hat\theta_2$ has lower MSE for every $n \ge 2$, with a faster (super-)linear convergence rate**. This is a notable example: the MLE-based estimator (max-based) is $n$-consistent rather than $\sqrt n$-consistent. For uniform endpoints, sample extremes are far more informative than sample means.

---

**Exercise 2.**
**MSE-optimal scaling of $S^2$.** Among estimators of $\sigma^2$ of the form $c \sum(X_i - \bar X)^2$ for $X_i \sim N(\mu, \sigma^2)$, find the $c$ that minimizes MSE.

??? success "Solution to Exercise 2"
    Let $T = \sum(X_i - \bar X)^2$. Then $T \sim \sigma^2 \chi^2_{n-1}$, so $\mathbb{E}[T] = (n-1)\sigma^2$, $\mathrm{Var}(T) = 2(n-1)\sigma^4$.

    For $\hat\sigma^2_c = c T$:

    $\mathbb{E}[\hat\sigma^2_c] = c(n-1)\sigma^2$, $\mathrm{Bias} = (c(n-1) - 1)\sigma^2$, $\mathrm{Var} = 2 c^2 (n-1) \sigma^4$.

    $\mathrm{MSE}(c) = 2 c^2(n-1) \sigma^4 + (c(n-1) - 1)^2 \sigma^4$.

    Minimize: $d \mathrm{MSE}/dc = 4c(n-1)\sigma^4 + 2(c(n-1) - 1)(n-1)\sigma^4 = 0$.

    $4c + 2(c(n-1) - 1) = 0 \Rightarrow 2c + c(n-1) = 1 \Rightarrow c(n+1) = 1 \Rightarrow c^* = 1/(n+1)$.

    So **$\hat\sigma^2_{\text{MSE}} = (1/(n+1))\sum(X_i - \bar X)^2$** minimizes MSE — between MLE ($c = 1/n$) and unbiased ($c = 1/(n-1)$), tilted toward smaller denominator for more shrinkage.

    Rarely used in practice because the "unbiased" $s^2$ has the cleaner interpretation. But it demonstrates that the optimal-MSE estimator may differ from both MLE and unbiased estimators.

---

**Exercise 3.**
**MSE for biased estimator.** $\hat\theta$ is biased with $\mathbb{E}[\hat\theta] = \theta + b/n$ and $\mathrm{Var}(\hat\theta) = v/n$. Compute MSE and the asymptotic behavior.

??? success "Solution to Exercise 3"
    $\mathrm{MSE} = \mathrm{Var} + \mathrm{Bias}^2 = v/n + b^2/n^2$.

    For large $n$: $\mathrm{MSE} \approx v/n + O(1/n^2)$. The variance dominates; bias contributes only the lower-order term.

    Asymptotic consistency: $\mathrm{MSE} \to 0 \Rightarrow \hat\theta \to \theta$ in $L^2$ and hence in probability. The estimator is consistent despite finite-sample bias.

    **Insight:** $O(1/n)$ bias is "invisible" asymptotically — it disappears faster than the noise. This is why MLEs (typically with $O(1/n)$ bias) are asymptotically efficient.

    Estimators with $O(1)$ bias (constant, like $\hat\theta = c$) are inconsistent — the bias term in MSE doesn't shrink.

---

**Exercise 4.**
**Cramér-Rao + MSE.** For an unbiased estimator, MSE = Var. CRLB gives Var $\ge 1/(n I(\theta))$. State the analog for biased estimators.

??? success "Solution to Exercise 4"
    **Biased CRLB:** for any estimator $\hat\theta$ (biased or not):

    $$
    \mathrm{Var}(\hat\theta) \ge \frac{(1 + b'(\theta))^2}{n I(\theta)}
    $$

    where $b(\theta) = \mathbb{E}[\hat\theta] - \theta$ is the bias function.

    For unbiased estimators, $b' = 0$, recovering the standard CRLB.

    For biased estimators: if $b'(\theta) = -1$ (e.g., a constant estimator $\hat\theta = c$), the bound is 0 — trivially achieved by zero-variance constant estimators.

    More generally, biased CRLB gives: MSE = Var + bias$^2 \ge (1 + b')^2/(nI) + b^2$.

    Practical use: shrinkage estimators (ridge, James-Stein) deliberately introduce bias to reduce variance, lowering MSE below the unbiased CRLB. This is provably impossible for unbiased estimators but routine for biased ones.

---

**Exercise 5.**
**Practical shrinkage example.** A poll of $n$ people gives $\hat p = X/n$. Consider the shrunk estimator $\hat p_{\text{shr}} = w \hat p + (1 - w) p_0$ for some target $p_0$ (e.g., 0.5). Find the optimal $w$ (assume the true $p$ equals $p_0$).

??? success "Solution to Exercise 5"
    If true $p = p_0$: bias of $\hat p_{\text{shr}} = w p_0 + (1 - w) p_0 - p_0 = 0$. Variance: $w^2 \mathrm{Var}(\hat p) = w^2 p_0(1 - p_0)/n$.

    MSE $= w^2 p_0(1-p_0)/n$. Minimized at $w = 0$ (i.e., always estimate $p_0$).

    **More realistic case:** true $p$ is uncertain (random with $\mathbb{E}[p] = p_0$, $\mathrm{Var}(p) = \tau^2$). Bias squared = $(1 - w)^2(p - p_0)^2$. Expected MSE over $p$:

    $w^2 p_0(1-p_0)/n + (1-w)^2 \tau^2$.

    Minimize: $w^* = \tau^2/(\tau^2 + p_0(1-p_0)/n)$. Closer to 1 (less shrinkage) when $\tau^2$ is large (prior uncertain); closer to 0 (more shrinkage) when $\tau^2$ is small (prior confident).

    This is the **empirical Bayes** shrinkage estimator. Used in election polling aggregation, A/B testing with many small experiments, and James-Stein-style multivariate shrinkage.

---

**Exercise 6.**
**MSE for $\bar X$ under different population distributions.** Compute MSE of $\bar X$ as estimator of $\mu$ for: (a) $N(\mu, \sigma^2)$; (b) $\mathrm{Exp}(1/\mu)$; (c) population with infinite second moment.

??? success "Solution to Exercise 6"
    (a) Normal: $\mathrm{MSE}(\bar X) = \mathrm{Var}(\bar X) = \sigma^2/n$ (unbiased). Achieves CRLB; UMVUE.

    (b) Exponential with mean $\mu$, variance $\mu^2$: $\bar X$ unbiased; $\mathrm{Var}(\bar X) = \mu^2/n$. So $\mathrm{MSE} = \mu^2/n$. Compare to CRLB: $I(\mu) = 1/\mu^2$, CRLB $= \mu^2/n$. $\bar X$ is efficient for exponential mean.

    (c) Infinite variance (e.g., Pareto with shape $\le 2$): $\mathrm{Var}(\bar X) = \infty$, $\mathrm{MSE} = \infty$.

    For Cauchy (no finite mean), $\bar X$ is not even a meaningful estimator of "center" — better to use the sample median, which is consistent for the Cauchy median.

    **General lesson:** $\bar X$ is the canonical estimator, optimal for normal/exponential, but breaks down for heavy-tailed populations. Always verify finiteness of moments before using mean-based estimators.
