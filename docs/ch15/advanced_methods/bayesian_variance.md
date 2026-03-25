# Bayesian Variance Testing

The frequentist tests for variance rely on sampling distributions and $p$-values. The Bayesian approach takes a fundamentally different perspective: it treats the population variance $\sigma^2$ as a random variable with a prior distribution, updates this distribution using the observed data, and makes inferences through the resulting posterior distribution. This framework provides a natural way to quantify uncertainty about $\sigma^2$ and to compare variances across groups.

## The Conjugate Prior for Variance

When the data are normally distributed with known mean $\mu$, the conjugate prior for the variance $\sigma^2$ is the **inverse-gamma** distribution:

$$
\sigma^2 \sim \text{Inv-Gamma}(\alpha_0, \beta_0)
$$

with density

$$
p(\sigma^2) = \frac{\beta_0^{\alpha_0}}{\Gamma(\alpha_0)} (\sigma^2)^{-\alpha_0 - 1} \exp\!\left(-\frac{\beta_0}{\sigma^2}\right), \quad \sigma^2 > 0
$$

The hyperparameters $\alpha_0 > 0$ and $\beta_0 > 0$ encode prior beliefs:

- $\alpha_0$ controls the strength of the prior (larger $\alpha_0$ means more confidence in the prior)
- $\beta_0 / \alpha_0$ approximates the prior expected value of $\sigma^2$ (for large $\alpha_0$)
- The prior mean is $E[\sigma^2] = \beta_0 / (\alpha_0 - 1)$ for $\alpha_0 > 1$

A common weakly informative choice is $\alpha_0 = \beta_0 = 0.01$ (or even $\alpha_0 = \beta_0 = 0.001$), which produces a diffuse prior that lets the data dominate.

## Posterior Distribution

Given observations $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$ with known $\mu$, the posterior for $\sigma^2$ is also inverse-gamma:

$$
\sigma^2 \mid X_1, \ldots, X_n \sim \text{Inv-Gamma}(\alpha_n, \beta_n)
$$

where

$$
\alpha_n = \alpha_0 + \frac{n}{2}, \qquad \beta_n = \beta_0 + \frac{1}{2}\sum_{i=1}^{n}(X_i - \mu)^2
$$

When $\mu$ is unknown (the usual case), we replace $\mu$ with $\bar{X}$ and use $n - 1$ degrees of freedom:

$$
\alpha_n = \alpha_0 + \frac{n - 1}{2}, \qquad \beta_n = \beta_0 + \frac{(n-1)S^2}{2}
$$

This conjugacy means the posterior has a closed-form solution, making computation straightforward.

## Bayesian Credible Interval

A $100(1-\alpha)\%$ credible interval for $\sigma^2$ is an interval $[L, U]$ such that

$$
P(L \le \sigma^2 \le U \mid \text{data}) = 1 - \alpha
$$

For the inverse-gamma posterior, the credible interval is obtained from the quantiles of the $\text{Inv-Gamma}(\alpha_n, \beta_n)$ distribution.

!!! note "Credible Interval vs. Confidence Interval"
    A Bayesian credible interval has a direct probability interpretation: given the data and prior, there is a $(1-\alpha)$ probability that $\sigma^2$ lies in the interval. A frequentist confidence interval does not make this claim; it says that the procedure captures $\sigma^2$ in $(1-\alpha)$ of repeated samples. With a diffuse prior, the two intervals are numerically similar.

## Bayesian Hypothesis Testing

To test $H_0\colon \sigma^2 = \sigma_0^2$ against $H_1\colon \sigma^2 \neq \sigma_0^2$, the Bayesian approach computes the posterior probability of $H_0$ or uses the **Bayes factor**:

$$
\text{BF}_{01} = \frac{p(\text{data} \mid H_0)}{p(\text{data} \mid H_1)}
$$

where $p(\text{data} \mid H_0)$ is the marginal likelihood under $H_0$ (with $\sigma^2$ fixed at $\sigma_0^2$) and $p(\text{data} \mid H_1)$ integrates the likelihood over the prior for $\sigma^2$.

A simpler approach for comparing two variances uses the posterior distribution of the variance ratio. Given independent posteriors for $\sigma_1^2$ and $\sigma_2^2$, compute

$$
R = \frac{\sigma_1^2}{\sigma_2^2}
$$

If the $95\%$ credible interval for $R$ includes 1, the data are consistent with equal variances.

## Comparing Two Variances

For two independent groups with inverse-gamma posteriors:

$$
\sigma_1^2 \mid \text{data}_1 \sim \text{Inv-Gamma}(\alpha_{n_1}, \beta_{n_1})
$$

$$
\sigma_2^2 \mid \text{data}_2 \sim \text{Inv-Gamma}(\alpha_{n_2}, \beta_{n_2})
$$

The ratio $R = \sigma_1^2 / \sigma_2^2$ does not have a simple closed-form distribution, but it can be estimated by Monte Carlo sampling: draw $\sigma_1^{2(b)}$ from the first posterior and $\sigma_2^{2(b)}$ from the second, then compute $R^{(b)} = \sigma_1^{2(b)} / \sigma_2^{2(b)}$ for $b = 1, \ldots, B$.

The posterior probability that $\sigma_1^2 > \sigma_2^2$ is

$$
P(\sigma_1^2 > \sigma_2^2 \mid \text{data}) \approx \frac{1}{B}\sum_{b=1}^{B} \mathbf{1}(R^{(b)} > 1)
$$

## Example

A sample of $n = 20$ observations has sample variance $S^2 = 15.3$. Using a weakly informative prior $\sigma^2 \sim \text{Inv-Gamma}(0.01, 0.01)$:

$$
\alpha_n = 0.01 + \frac{19}{2} = 9.51
$$

$$
\beta_n = 0.01 + \frac{19 \times 15.3}{2} = 145.36
$$

The posterior is $\sigma^2 \mid \text{data} \sim \text{Inv-Gamma}(9.51, 145.36)$.

The posterior mean is $\beta_n / (\alpha_n - 1) = 145.36 / 8.51 = 17.08$, which is close to the sample variance of 15.3 (the small discrepancy reflects the prior's mild influence).

## Python Implementation

```python
import numpy as np
from scipy import stats

# Data
n = 20
s_squared = 15.3

# Weakly informative prior
alpha_0, beta_0 = 0.01, 0.01

# Posterior parameters
alpha_n = alpha_0 + (n - 1) / 2
beta_n = beta_0 + (n - 1) * s_squared / 2

# Posterior summary
# Inv-Gamma(alpha, beta) is related to 1/Gamma(alpha, 1/beta)
post_mean = beta_n / (alpha_n - 1)
print(f"Posterior mean of sigma^2: {post_mean:.2f}")

# 95% credible interval using inverse-gamma quantiles
# scipy's invgamma uses scale parameter
ci_lower = stats.invgamma.ppf(0.025, a=alpha_n, scale=beta_n)
ci_upper = stats.invgamma.ppf(0.975, a=alpha_n, scale=beta_n)
print(f"95% credible interval: ({ci_lower:.2f}, {ci_upper:.2f})")

# Monte Carlo comparison of two variances
rng = np.random.default_rng(42)
alpha_n1, beta_n1 = 9.51, 145.36   # Group 1 posterior
alpha_n2, beta_n2 = 12.01, 120.10  # Group 2 posterior

sigma1_samples = stats.invgamma.rvs(a=alpha_n1, scale=beta_n1, size=10000, random_state=rng)
sigma2_samples = stats.invgamma.rvs(a=alpha_n2, scale=beta_n2, size=10000, random_state=rng)

ratio = sigma1_samples / sigma2_samples
prob_greater = np.mean(ratio > 1)
print(f"P(sigma1^2 > sigma2^2 | data) = {prob_greater:.3f}")
print(f"95% credible interval for ratio: ({np.percentile(ratio, 2.5):.2f}, {np.percentile(ratio, 97.5):.2f})")
```

## Advantages and Limitations

**Advantages:**

- Direct probability statements about $\sigma^2$ (credible intervals have intuitive interpretation)
- Prior information from domain knowledge can be incorporated
- No reliance on asymptotic approximations
- Natural framework for comparing multiple variances through posterior sampling

**Limitations:**

- Requires specifying a prior, which may be controversial
- Conjugate analysis assumes normality; non-normal data require more complex models
- Computational cost increases with model complexity (though MCMC makes this manageable)
