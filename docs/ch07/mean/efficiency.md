# Efficiency of the Sample Mean

After establishing that an estimator is unbiased and consistent, a natural next question is: how precise can it be? Among all unbiased estimators of the same parameter, some achieve smaller variance than others. An estimator that attains the smallest possible variance — the Cramér–Rao lower bound — is called **efficient**. This section examines when and why the sample mean earns that title, and what happens when Normality fails.

## Definition of Efficiency

An unbiased estimator $\hat{\theta}$ of a parameter $\theta$ is **efficient** if its variance equals the Cramér–Rao lower bound (CRLB):

$$
\operatorname{Var}(\hat{\theta}) = \frac{1}{n \, I(\theta)}
$$

where $I(\theta)$ is the Fisher information for a single observation. Any unbiased estimator satisfying this equality has the smallest variance achievable among all unbiased estimators of $\theta$.

## CRLB for the Normal Mean

The Normal family $X \sim N(\mu, \sigma^2)$ satisfies the regularity conditions required for the CRLB to hold (the support does not depend on $\mu$, and the log-likelihood is twice differentiable with interchangeable expectation and differentiation). The Fisher information for a single observation is $I(\mu) = 1/\sigma^2$, so the CRLB gives:

$$
\operatorname{Var}(\hat{\mu}) \geq \frac{1}{n \, I(\mu)} = \frac{\sigma^2}{n}
$$

The sample mean $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$ has $\operatorname{Var}(\bar{X}) = \sigma^2 / n$, which matches the bound exactly. Because $\bar{X}$ is both the maximum likelihood estimator and the uniformly minimum variance unbiased estimator (UMVUE) for $\mu$ under Normality, it is efficient among all unbiased estimators of $\mu$.

## Asymptotic Relative Efficiency

The sample mean is efficient under Normality, but real data often deviate from the Normal model. Heavy-tailed or skewed distributions cause the sample mean to lose its efficiency advantage. The **asymptotic relative efficiency (ARE)** provides a way to compare two estimators by measuring how many observations one estimator needs relative to another to achieve the same precision.

For two estimators $T_1$ and $T_2$ of the same parameter, the ARE of $T_1$ relative to $T_2$ is defined as:

$$
\operatorname{ARE}(T_1, T_2) = \frac{\operatorname{Var}(T_2)}{\operatorname{Var}(T_1)}
$$

When $\operatorname{ARE}(T_1, T_2) > 1$, estimator $T_1$ is more efficient (needs fewer observations). When $\operatorname{ARE}(T_1, T_2) < 1$, estimator $T_2$ is more efficient.

### ARE of the Sample Mean vs the Median

Under a Normal distribution, the asymptotic variances are $\operatorname{Var}(\bar{X}) = \sigma^2/n$ and $\operatorname{Var}(\text{Median}) = \pi\sigma^2/(2n)$, giving:

$$
\operatorname{ARE}(\bar{X}, \text{Median}) = \frac{\operatorname{Var}(\text{Median})}{\operatorname{Var}(\bar{X})} = \frac{\pi}{2} \approx 1.57
$$

This means the sample mean is about 57% more efficient than the median under Normality: the median would require roughly 1.57 times as many observations to match the precision of $\bar{X}$.

However, the ranking reverses for heavy-tailed distributions. The following table summarizes the ARE of the sample mean relative to the median for several distributions:

| Distribution | $\operatorname{ARE}(\bar{X}, \text{Median})$ | Interpretation |
|---|---|---|
| Normal | $\pi/2 \approx 1.57$ | Mean is 57% more efficient |
| Double exponential (Laplace) | $2/3 \approx 0.67$ | Median is 50% more efficient |
| Cauchy | $0$ (mean has infinite variance) | Median is strictly preferred |

For the Laplace distribution, the median requires only two-thirds as many observations as the sample mean. For the Cauchy distribution, the sample mean has infinite variance, so it provides no useful information regardless of sample size — the median is the clear choice.

!!! tip "Practical Guidance"
    When the underlying distribution is approximately Normal, the sample mean is the best choice. When heavy tails or outliers are present, robust alternatives such as the trimmed mean or median offer better precision despite sacrificing some efficiency under Normality.
