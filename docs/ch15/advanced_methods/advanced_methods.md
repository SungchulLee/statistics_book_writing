# Advanced Methods for Variance Testing


In modern statistics, especially when traditional parametric tests fail due to violated assumptions or limited sample sizes, advanced methods for variance testing provide more flexibility and robustness. These methods often use computational techniques such as bootstrapping, or Bayesian approaches, to make inferences when normality and homoscedasticity assumptions are violated.

## Bootstrapping Methods for Variance Testing

**Bootstrapping** is a resampling technique that generates many simulated samples from the original data by sampling with replacement. It is especially useful when we do not wish to assume normality or when dealing with small samples. Bootstrapping provides a way to estimate the sampling distribution of a statistic, such as variance, without relying on a theoretical distribution.

### Steps for Bootstrapping Variance Testing

1. **Resampling:** Generate a large number of new datasets by randomly sampling with replacement from the original data. Each bootstrap sample should have the same size as the original dataset.
2. **Calculate Variance:** For each bootstrap sample, calculate the sample variance $s_i^2$.
3. **Construct the Sampling Distribution:** The variances from the bootstrapped samples approximate the sampling distribution of the variance. This can be used to estimate confidence intervals or test hypotheses.
4. **Hypothesis Testing:** Use the distribution of the bootstrapped variances to compare with the null hypothesis (e.g., equal variances across groups).

### Hypotheses

**Null Hypothesis ($H_0$):** The variances are equal across the groups:

$$
H_0: \sigma_1^2 = \sigma_2^2 = \dots = \sigma_k^2
$$

**Alternative Hypothesis ($H_1$):** At least one group has a variance different from the others:

$$
H_1: \sigma_i^2 \neq \sigma_j^2 \quad \text{for at least one pair} \quad i \neq j
$$

### Python Implementation

```python
import numpy as np

# Original data samples from two groups
sample1 = np.array([10, 12, 14, 16, 18])
sample2 = np.array([22, 24, 26, 28, 30])

# Number of bootstrap samples
B = 1000
bootstrapped_diffs = []

# Bootstrapping process
for _ in range(B):
    sample1_resampled = np.random.choice(sample1, size=len(sample1), replace=True)
    sample2_resampled = np.random.choice(sample2, size=len(sample2), replace=True)

    # Calculate the variance for each resampled sample
    var_sample1 = np.var(sample1_resampled, ddof=1)
    var_sample2 = np.var(sample2_resampled, ddof=1)

    # Store the difference in variances
    bootstrapped_diffs.append(var_sample1 - var_sample2)

# Observed difference in variances
observed_diff = np.var(sample1, ddof=1) - np.var(sample2, ddof=1)

# Calculate p-value by comparing the observed difference to the bootstrap distribution
p_value = np.mean(np.abs(bootstrapped_diffs) >= np.abs(observed_diff))

print(f"Observed difference in variances: {observed_diff}")
print(f"Bootstrap p-value: {p_value}")
```

### Interpretation

The bootstrapped distribution of the differences in variances can be used to compute a p-value. If the p-value is below a chosen threshold (e.g., $\alpha = 0.05$), we reject the null hypothesis and conclude that the variances differ between the two groups.

### Advantages of Bootstrapping

- **Distribution-Free:** Bootstrapping does not rely on the assumption of normality, making it ideal for data that deviate from this assumption.
- **Small Samples:** It can be applied to small datasets, where parametric tests may lack power.
- **Flexible:** Bootstrapping can be applied to any statistic, including variance, making it highly versatile.

---

## Bayesian Methods for Variance Testing

**Bayesian methods** offer a probabilistic approach to variance testing by incorporating prior information about the variances and updating these beliefs with observed data. Instead of relying solely on the observed sample (as in classical statistics), Bayesian methods combine the data with prior distributions to yield **posterior distributions** of the variances.

### Bayesian Framework for Variance Testing

In Bayesian variance testing, we estimate the posterior distribution of variances using prior beliefs and the likelihood of the data. This results in a posterior probability distribution for each variance, allowing for hypothesis testing or credible intervals for the variances.

### Hypotheses

**Null Hypothesis ($H_0$):** The variances are equal across groups:

$$
H_0: \sigma_1^2 = \sigma_2^2 = \dots = \sigma_k^2
$$

**Alternative Hypothesis ($H_1$):** At least one group has a different variance:

$$
H_1: \sigma_i^2 \neq \sigma_j^2 \quad \text{for at least one pair} \quad i \neq j
$$

### Bayesian Inference Process

**1. Prior Distribution:** Choose a prior distribution for the variance parameters $\sigma_i^2$. A common choice is the inverse-gamma distribution:

$$
\sigma_i^2 \sim \text{InverseGamma}(\alpha, \beta)
$$

**2. Likelihood:** Model the likelihood of the data given the variances. For normally distributed data:

$$
X_i \sim \mathcal{N}(0, \sigma_i^2)
$$

**3. Posterior Distribution:** Use Bayes' theorem to combine the prior and likelihood to obtain the posterior distribution for the variances:

$$
P(\sigma_i^2 \mid \text{data}) \propto P(\text{data} \mid \sigma_i^2) \, P(\sigma_i^2)
$$

**4. Bayes Factor:** To test for equality of variances, calculate the Bayes Factor which quantifies the strength of evidence for the null hypothesis vs. the alternative:

$$
BF = \frac{P(\text{data} \mid H_0)}{P(\text{data} \mid H_1)}
$$

A Bayes Factor greater than 1 supports the null hypothesis, while a value less than 1 supports the alternative hypothesis.

### Python Implementation

Using the `PyMC3` package for Bayesian inference:

```python
import pymc3 as pm

# Data for two groups
sample1 = [10, 12, 14, 16, 18]
sample2 = [22, 24, 26, 28, 30]

# Bayesian model for variance comparison
with pm.Model() as model:
    # Priors for variances
    sigma1 = pm.InverseGamma('sigma1', alpha=2, beta=1)
    sigma2 = pm.InverseGamma('sigma2', alpha=2, beta=1)

    # Likelihood based on sample data
    obs1 = pm.Normal('obs1', mu=0, sigma=sigma1, observed=sample1)
    obs2 = pm.Normal('obs2', mu=0, sigma=sigma2, observed=sample2)

    # Sampling from the posterior
    trace = pm.sample(2000)

    # Summary of the posterior
    pm.summary(trace)
```

### Interpretation

The posterior distribution provides a probability distribution for each variance. By comparing the posterior distributions, we can make probabilistic statements about whether the variances differ across the groups. The Bayes Factor provides a measure of evidence for the null hypothesis compared to the alternative — a high Bayes Factor indicates stronger support for the null hypothesis of equal variances.

### Advantages of Bayesian Methods

1. **Prior Information:** Incorporates prior knowledge or expert opinion into the analysis, which is useful in fields with well-established prior information.
2. **Probabilistic Inference:** Provides full posterior distributions for parameters, allowing for more flexible and informative conclusions than traditional hypothesis testing.
3. **Handling of Complex Models:** Bayesian methods can be applied to hierarchical models and other complex data structures.

---

## When to Use Advanced Methods

**Bootstrapping** and **Bayesian methods** are particularly useful in the following situations:

- **Small Sample Sizes:** When the sample size is small, bootstrapping provides a flexible and non-parametric alternative to traditional tests.
- **Non-Normal Data:** When the assumption of normality is violated, bootstrapping or Bayesian methods are more reliable than parametric tests like the F-test or Bartlett's test.
- **Incorporating Prior Knowledge:** Bayesian methods are ideal when prior knowledge about the population variances is available and can be incorporated into the analysis.


## Exercises

**Exercise 1.**
Compare the likelihood ratio test, Wald test, and score test for testing hypotheses about variances. Under what conditions do they give similar results?

??? success "Solution to Exercise 1"
    All three tests are asymptotically equivalent under the null hypothesis: as $n \to \infty$, their test statistics converge to the same $\chi^2$ distribution, and their p-values agree.

    In finite samples, they can differ: the likelihood ratio test is generally most reliable (invariant to parameterization), the Wald test can be unreliable when the parameter is near a boundary, and the score test only requires estimation under $H_0$.

    They give similar results when: (1) the sample size is large, (2) the data are approximately normal, and (3) the true parameter is not near the boundary of the parameter space.

---

**Exercise 2.**
Describe a scenario where advanced variance testing methods (bootstrap, Bayesian) are preferable to the classical chi-squared test.

??? success "Solution to Exercise 2"
    When the data are heavily non-normal (e.g., financial returns with excess kurtosis of 5+), the chi-squared test for variance is unreliable because its derivation requires exact normality.

    The **bootstrap** approach resamples the data to build a reference distribution for the variance statistic without assuming normality. The **Bayesian** approach places a prior on $\sigma^2$ and computes the posterior, allowing probability statements about the variance.

    Both methods are preferred when: (1) normality is clearly violated, (2) the sample size is too small for asymptotic methods to be reliable, or (3) the analyst wants to incorporate prior information (Bayesian) or avoid distributional assumptions entirely (bootstrap).

---

**Exercise 3.**
The bootstrap test for equal variances resamples under the null. Describe the resampling procedure.

??? success "Solution to Exercise 3"
    Under $H_0: \sigma_1^2 = \sigma_2^2$, the two samples come from populations with equal variance. The bootstrap procedure:

    1. Pool all observations from both groups.
    2. For each bootstrap replicate, draw $n_1$ observations (with replacement) for group 1 and $n_2$ for group 2 from the pooled sample.
    3. Compute the test statistic (e.g., ratio of sample variances $s_1^{*2}/s_2^{*2}$) for each bootstrap replicate.
    4. The p-value is the proportion of bootstrap test statistics as extreme as or more extreme than the observed statistic.

    By resampling from the pooled data, the null hypothesis of equal variances is enforced.

---

**Exercise 4.**
In a Bayesian test for $\sigma^2$, the conjugate prior for the variance of normal data is the inverse-gamma distribution. If the prior is $\sigma^2 \sim \text{Inv-Gamma}(\alpha_0, \beta_0)$ and we observe $n$ data points, state the posterior distribution.

??? success "Solution to Exercise 4"
    The posterior is:

    $$
    \sigma^2 \mid \mathbf{x} \sim \text{Inv-Gamma}\!\left(\alpha_0 + \frac{n}{2},\; \beta_0 + \frac{1}{2}\sum_{i=1}^n(x_i - \bar{x})^2\right)
    $$

    The posterior incorporates both prior information ($\alpha_0, \beta_0$) and data ($n$ and the sum of squared deviations). A 95% credible interval for $\sigma^2$ is obtained from the 2.5th and 97.5th percentiles of this inverse-gamma distribution. Unlike the frequentist confidence interval, the Bayesian credible interval has a direct probability interpretation: there is a 95% posterior probability that $\sigma^2$ lies in the interval.
