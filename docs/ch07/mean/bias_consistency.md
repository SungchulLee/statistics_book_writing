# Bias and Consistency of the Sample Mean

## Introduction

Two fundamental questions about any estimator are: (1) Does it systematically over- or underestimate the true parameter? (**bias**) and (2) Does it converge to the true value as the sample size grows? (**consistency**). For the sample mean $\bar{X}$, the answers are reassuringly simple — it is unbiased and consistent under very mild conditions — but the precise statements and their implications are worth studying carefully.

## Bias of the Sample Mean

### Unbiasedness

The sample mean $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$ is **unbiased** for $\mu = E[X]$:

$$E[\bar{X}] = \mu \quad \text{for all } n \geq 1$$

**Proof:** By linearity of expectation:

$$E[\bar{X}] = E\left[\frac{1}{n}\sum_{i=1}^n X_i\right] = \frac{1}{n}\sum_{i=1}^n E[X_i] = \frac{1}{n} \cdot n\mu = \mu$$

This holds under minimal conditions:
- Observations need not be identically distributed (only requires $E[X_i] = \mu$ for all $i$)
- Observations need not be independent
- No distributional assumptions are needed
- Valid for any sample size $n \geq 1$

### Finite Sample Bias is Zero

Unlike many estimators (e.g., the MLE of variance), the sample mean has **exactly zero bias** for every finite sample size. This is a strong property — most estimators have nonzero bias for finite samples and are only asymptotically unbiased.

### Comparison with Biased Alternatives

Some estimators of the population mean are intentionally biased:

| Estimator | Bias | MSE |
|-----------|------|-----|
| $\bar{X}$ (sample mean) | $0$ | $\sigma^2/n$ |
| $\lambda\bar{X}$ (shrinkage, $\lambda < 1$) | $(\lambda-1)\mu$ | $\lambda^2\sigma^2/n + (1-\lambda)^2\mu^2$ |
| $c$ (constant) | $c - \mu$ | $(c-\mu)^2$ |

As discussed in the bias-variance tradeoff, biased estimators can sometimes have lower MSE, especially when $|\mu|$ is small relative to $\sigma/\sqrt{n}$.

## Consistency

### Consistency in Probability

The sample mean is **consistent** for $\mu$:

$$\bar{X}_n \xrightarrow{p} \mu \quad \text{as } n \to \infty$$

This means: for any $\epsilon > 0$,

$$\lim_{n \to \infty} P\left(|\bar{X}_n - \mu| > \epsilon\right) = 0$$

### Proof via Chebyshev's Inequality

Using Chebyshev's inequality with the known variance of $\bar{X}$:

$$P(|\bar{X} - \mu| > \epsilon) \leq \frac{\text{Var}(\bar{X})}{\epsilon^2} = \frac{\sigma^2}{n\epsilon^2} \to 0$$

This requires only that $\sigma^2 < \infty$ (finite variance).

### Proof via the Weak Law of Large Numbers (WLLN)

The consistency of $\bar{X}$ is precisely the statement of the **Weak Law of Large Numbers**: if $X_1, X_2, \ldots$ are iid with $E[X_i] = \mu$ and $\text{Var}(X_i) = \sigma^2 < \infty$, then:

$$\bar{X}_n \xrightarrow{p} \mu$$

**Khintchine's WLLN** weakens the requirement: only $E[|X|] < \infty$ is needed (no finite variance requirement).

### Almost Sure Convergence (Strong Consistency)

Under the same conditions, the **Strong Law of Large Numbers (SLLN)** provides a stronger result:

$$P\left(\lim_{n \to \infty} \bar{X}_n = \mu\right) = 1$$

This means $\bar{X}_n \to \mu$ almost surely, not just in probability.

### Rate of Convergence

How fast does $\bar{X}_n$ converge to $\mu$?

**MSE convergence rate:**
$$\text{MSE}(\bar{X}_n) = \frac{\sigma^2}{n} = O(1/n)$$

**Standard error convergence rate:**
$$\text{SE}(\bar{X}_n) = \frac{\sigma}{\sqrt{n}} = O(1/\sqrt{n})$$

This $O(1/\sqrt{n})$ rate is fundamental — it means:
- Doubling accuracy requires 4× the data
- For 10× accuracy, you need 100× the data
- This rate cannot be improved (in general) without additional assumptions

### MSE Consistency

An estimator is **MSE-consistent** if $\text{MSE}(\hat{\theta}_n) \to 0$. For $\bar{X}$:

$$\text{MSE}(\bar{X}_n) = \underbrace{[\text{Bias}(\bar{X}_n)]^2}_{= 0} + \underbrace{\text{Var}(\bar{X}_n)}_{= \sigma^2/n \to 0} \to 0$$

MSE-consistency implies consistency in probability (by Markov's inequality).

## Conditions for Consistency

### When $\bar{X}$ is Consistent

The sample mean is consistent under various relaxations of the iid assumption:

1. **Independent, not identically distributed**: If $E[X_i] = \mu$ for all $i$ and $\frac{1}{n^2}\sum_{i=1}^n \text{Var}(X_i) \to 0$, then $\bar{X}_n \xrightarrow{p} \mu$.

2. **Dependent observations**: For stationary ergodic processes, $\bar{X}_n \to \mu$ a.s. by the Ergodic Theorem.

3. **Weakly dependent time series**: If autocorrelations decay fast enough (e.g., $\sum_{k=0}^\infty |\rho_k| < \infty$), then $\bar{X}_n$ is consistent.

### When $\bar{X}$ Fails to be Consistent

1. **Infinite variance** (e.g., Cauchy distribution): $\bar{X}_n$ is still consistent if $E[|X|] < \infty$ (by Khintchine's WLLN), even though $\text{Var}(X)$ doesn't exist.

2. **Infinite mean** (e.g., Cauchy): $E[X]$ doesn't exist, so there is no $\mu$ for $\bar{X}$ to converge to. The sample mean fluctuates wildly and does not converge.

3. **Non-stationary data**: If the mean changes over time, $\bar{X}$ converges to the average of the changing means, not to any single "true" value.

## Asymptotic Distribution

Beyond consistency, the CLT provides the asymptotic distribution:

$$\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} N(0, \sigma^2)$$

This allows construction of confidence intervals and hypothesis tests:

$$\bar{X} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}} \quad \text{(known } \sigma\text{)}$$

$$\bar{X} \pm t_{n-1,\alpha/2} \frac{S}{\sqrt{n}} \quad \text{(unknown } \sigma\text{)}$$

## Connections to Finance

Understanding bias and consistency of the mean is critical in finance:

- **Return estimation**: While $\bar{X}$ is consistent, the convergence rate $O(1/\sqrt{n})$ is too slow for practical return prediction. With 30 years of monthly data ($n = 360$), $\text{SE} \approx \sigma_{\text{monthly}} / 19$, which is still substantial.

- **Stationarity concerns**: Financial return distributions change over time (regime changes, structural breaks), violating the stationarity assumption. The sample mean of historical returns may not estimate the *current* expected return.

- **Mean reversion testing**: Testing whether asset prices are mean-reverting requires careful attention to the convergence properties of $\bar{X}$ under various dependency structures.

- **High-frequency estimation**: With high-frequency data, microstructure noise introduces bias. The "realized" mean of tick-by-tick prices is biased by bid-ask bounce effects.

## Summary

The sample mean is unbiased (zero bias for all $n$) and consistent (converges to $\mu$ as $n \to \infty$) under very mild conditions. The convergence rate is $O(1/\sqrt{n})$, which is optimal but practically slow. Almost sure convergence (SLLN) provides a stronger guarantee than convergence in probability (WLLN). These properties make $\bar{X}$ the default estimator for population means, but the slow convergence rate and sensitivity to distributional assumptions must be recognized, especially in financial applications.

## Key Formulas

| Property | Result | Condition |
|----------|--------|-----------|
| Unbiasedness | $E[\bar{X}] = \mu$ | $E[X_i] = \mu$ |
| Consistency (WLLN) | $\bar{X}_n \xrightarrow{p} \mu$ | iid, $E[|X|] < \infty$ |
| Strong consistency (SLLN) | $\bar{X}_n \to \mu$ a.s. | iid, $E[|X|] < \infty$ |
| MSE rate | $O(1/n)$ | $\text{Var}(X) < \infty$ |
| SE rate | $O(1/\sqrt{n})$ | $\text{Var}(X) < \infty$ |
| CLT | $\sqrt{n}(\bar{X}-\mu)/\sigma \to N(0,1)$ | iid, $\text{Var}(X) < \infty$ |
