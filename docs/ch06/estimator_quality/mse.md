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
