# Likelihood Ratio Test for Variances

The likelihood ratio test (LRT) provides a general-purpose framework for hypothesis testing based on comparing the maximized likelihoods under the null and alternative hypotheses. When applied to the problem of testing equal variances across $k$ normal populations, the LRT yields a statistic that is closely related to Bartlett's test. This section derives the LRT for variance homogeneity and explains the connection.

## The Likelihood Ratio Framework

For a general hypothesis test $H_0$ versus $H_1$, the likelihood ratio statistic is

$$
\Lambda = \frac{\sup_{\theta \in \Theta_0} L(\theta)}{\sup_{\theta \in \Theta} L(\theta)}
$$

where $\Theta_0$ is the parameter space under $H_0$ and $\Theta$ is the full parameter space. Since $\Theta_0 \subseteq \Theta$, we always have $0 \le \Lambda \le 1$. Small values of $\Lambda$ indicate that $H_0$ restricts the parameter space in a way that significantly reduces the likelihood.

By Wilks' theorem, under regularity conditions and as $n \to \infty$:

$$
-2\ln\Lambda \stackrel{d}{\to} \chi^2_r
$$

where $r$ is the difference in the number of free parameters between $\Theta$ and $\Theta_0$.

## One-Sample LRT for Variance

For a single sample $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$ with $\mu$ unknown, consider testing $H_0\colon \sigma^2 = \sigma_0^2$.

**Under $H_0$:** The MLE of $\mu$ is $\bar{X}$, and $\sigma^2$ is fixed at $\sigma_0^2$. The maximized log-likelihood is

$$
\ell_0 = -\frac{n}{2}\ln(2\pi\sigma_0^2) - \frac{1}{2\sigma_0^2}\sum_{i=1}^{n}(X_i - \bar{X})^2
$$

**Under the full model:** The MLEs are $\hat{\mu} = \bar{X}$ and $\hat{\sigma}^2 = \frac{1}{n}\sum(X_i - \bar{X})^2$. The maximized log-likelihood is

$$
\ell_1 = -\frac{n}{2}\ln(2\pi\hat{\sigma}^2) - \frac{n}{2}
$$

The log-likelihood ratio statistic is

$$
-2\ln\Lambda = -2(\ell_0 - \ell_1) = n\ln\!\left(\frac{\hat{\sigma}^2}{\sigma_0^2}\right) + \frac{n\hat{\sigma}^2}{\sigma_0^2} - n
$$

Under $H_0$, this converges in distribution to $\chi^2_1$.

## Multi-Sample LRT for Equal Variances

Consider $k$ independent samples from normal populations $N(\mu_i, \sigma_i^2)$ with sample sizes $n_i$ and sample variances $S_i^2$.

**Under $H_0\colon \sigma_1^2 = \cdots = \sigma_k^2 = \sigma^2$:** The MLE of the common variance is the pooled variance

$$
\hat{\sigma}^2 = S_p^2 = \frac{\sum_{i=1}^{k}(n_i - 1)S_i^2}{N - k}
$$

**Under the alternative:** Each group has its own MLE $\hat{\sigma}_i^2 = \frac{n_i - 1}{n_i}S_i^2 \approx S_i^2$ for large $n_i$.

The log-likelihood ratio statistic (using degrees of freedom $\nu_i = n_i - 1$) simplifies to

$$
-2\ln\Lambda = \sum_{i=1}^{k} \nu_i \ln\!\left(\frac{S_p^2}{S_i^2}\right) = (N - k)\ln S_p^2 - \sum_{i=1}^{k}\nu_i \ln S_i^2
$$

This is exactly the numerator of Bartlett's test statistic before the correction factor.

## Connection to Bartlett's Test

Bartlett's test statistic is the corrected version of the LRT:

$$
T_{\text{Bartlett}} = \frac{-2\ln\Lambda}{C}
$$

where

$$
C = 1 + \frac{1}{3(k-1)}\left(\sum_{i=1}^{k}\frac{1}{\nu_i} - \frac{1}{N-k}\right)
$$

The correction factor $C > 1$ improves the finite-sample chi-square approximation. Without it, the raw LRT statistic $-2\ln\Lambda$ tends to reject too frequently in small samples.

!!! note "LRT as the Foundation for Bartlett's Test"
    Bartlett's test is not an independent invention; it is the likelihood ratio test with a small-sample correction. Understanding the LRT derivation explains why Bartlett's test has its particular form and why it requires normality (the likelihood is Gaussian).

## Degrees of Freedom

Under $H_0$, there are $k + 1$ free parameters ($\mu_1, \ldots, \mu_k, \sigma^2$). Under the alternative, there are $2k$ free parameters ($\mu_1, \ldots, \mu_k, \sigma_1^2, \ldots, \sigma_k^2$). The difference is $r = 2k - (k+1) = k - 1$.

Therefore, by Wilks' theorem:

$$
-2\ln\Lambda \stackrel{d}{\to} \chi^2_{k-1}
$$

This confirms that Bartlett's test uses the correct reference distribution.

## Example

Two groups with $n_1 = 15$, $S_1^2 = 22.4$ and $n_2 = 18$, $S_2^2 = 35.1$.

**Step 1.** Pooled variance:

$$
S_p^2 = \frac{14(22.4) + 17(35.1)}{31} = \frac{313.6 + 596.7}{31} = 29.365
$$

**Step 2.** LRT statistic:

$$
-2\ln\Lambda = 14\ln\!\left(\frac{29.365}{22.4}\right) + 17\ln\!\left(\frac{29.365}{35.1}\right)
$$

$$
= 14(0.2710) + 17(-0.1786) = 3.794 - 3.036 = 0.758
$$

**Step 3.** Correction factor:

$$
C = 1 + \frac{1}{3}\left(\frac{1}{14} + \frac{1}{17} - \frac{1}{31}\right) = 1 + \frac{1}{3}(0.0714 + 0.0588 - 0.0323) = 1 + 0.0326 = 1.033
$$

**Step 4.** Bartlett's statistic: $T = 0.758 / 1.033 = 0.734$.

**Step 5.** Compare with $\chi^2_{0.95, 1} = 3.841$. Since $0.734 < 3.841$, we fail to reject $H_0$.

## Python Implementation

```python
import numpy as np
from scipy import stats

# Group statistics
n = np.array([15, 18])
s2 = np.array([22.4, 35.1])
k = len(n)
nu = n - 1
N = n.sum()

# Pooled variance
s2_pooled = np.sum(nu * s2) / np.sum(nu)

# LRT statistic (uncorrected)
lrt = np.sum(nu * np.log(s2_pooled / s2))

# Bartlett correction
C = 1 + (1 / (3 * (k - 1))) * (np.sum(1 / nu) - 1 / np.sum(nu))

# Corrected statistic
T = lrt / C

# p-value
p_value = stats.chi2.sf(T, k - 1)

print(f"LRT statistic (uncorrected): {lrt:.3f}")
print(f"Correction factor: {C:.4f}")
print(f"Bartlett statistic (corrected): {T:.3f}")
print(f"P-value: {p_value:.4f}")
```
