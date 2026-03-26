# One-Sample t-Test for the Mean


!!! warning "Incomplete page"
    This page is missing the required five-section structure (Concept Definition, Explanation, Diagram / Example). Content needs to be reorganized and expanded.

## Overview

In practice, the population standard deviation $\sigma$ is almost never known. The $z$-test for the mean requires $\sigma$, so it cannot be applied directly to most real-world problems. The one-sample $t$-test resolves this by replacing $\sigma$ with the sample standard deviation $S$. This substitution introduces additional uncertainty --- $S$ is itself a random variable --- and the $t$-distribution accounts for this extra variability through heavier tails than the standard normal. As the sample size grows, $S$ converges to $\sigma$, the $t$-distribution converges to the standard normal, and the $t$-test and $z$-test become equivalent.

## Hypotheses

Let $X_1, X_2, \ldots, X_n$ be a random sample from a population with mean $\mu$ and unknown variance $\sigma^2$. The null hypothesis specifies a particular value for the mean:

$$
H_0\colon \mu = \mu_0
$$

The alternative takes one of three forms:

| Alternative | Interpretation |
|---|---|
| $H_1\colon \mu \neq \mu_0$ | Two-sided: the mean differs from $\mu_0$ |
| $H_1\colon \mu > \mu_0$ | Right-sided: the mean exceeds $\mu_0$ |
| $H_1\colon \mu < \mu_0$ | Left-sided: the mean is below $\mu_0$ |

## Test Statistic

The test statistic measures how many estimated standard errors the sample mean $\bar{X}$ falls from the hypothesized value $\mu_0$. Its structure mirrors the $z$-statistic, but with $S$ in place of $\sigma$:

$$
t = \frac{\bar{X} - \mu_0}{S / \sqrt{n}}
$$

where $S = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2}$ is the sample standard deviation.

Under $H_0$ and normality of the population, this statistic follows Student's $t$-distribution with $n - 1$ degrees of freedom:

$$
t = \frac{\bar{X} - \mu_0}{S / \sqrt{n}} \sim t_{n-1}
$$

The $t_{n-1}$ distribution has heavier tails than the standard normal $N(0,1)$, reflecting the additional uncertainty from estimating $\sigma$. As $n \to \infty$, $t_{n-1} \to N(0,1)$, and the $t$-test reduces to the $z$-test.

!!! note "Why n - 1 Degrees of Freedom?"
    The sample standard deviation $S$ uses $n - 1$ in the denominator because one degree of freedom is consumed by estimating $\bar{X}$. The $n$ deviations $X_i - \bar{X}$ satisfy the constraint $\sum(X_i - \bar{X}) = 0$, leaving only $n - 1$ free pieces of information about the spread.

## Rejection Regions

At significance level $\alpha$, the rejection region depends on the alternative:

**Two-sided** ($H_1\colon \mu \neq \mu_0$): Reject $H_0$ if

$$
|t| > t_{\alpha/2,\, n-1}
$$

**Right-sided** ($H_1\colon \mu > \mu_0$): Reject $H_0$ if

$$
t > t_{\alpha,\, n-1}
$$

**Left-sided** ($H_1\colon \mu < \mu_0$): Reject $H_0$ if

$$
t < -t_{\alpha,\, n-1}
$$

Here $t_{\alpha,\, n-1}$ denotes the upper $\alpha$ critical value of the $t_{n-1}$ distribution, satisfying $P(t_{n-1} > t_{\alpha,\, n-1}) = \alpha$.

## Worked Example

A food manufacturer claims that its cereal boxes contain an average of $\mu_0 = 500$ grams. A consumer group suspects the boxes are underfilled and collects a random sample of $n = 16$ boxes, finding $\bar{x} = 496.2$ grams and $s = 4.8$ grams. Test whether the mean fill weight is less than 500 grams at $\alpha = 0.05$.

**Step 1.** State the hypotheses:

$$
H_0\colon \mu = 500 \qquad H_1\colon \mu < 500
$$

**Step 2.** Compute the test statistic:

$$
t = \frac{496.2 - 500}{4.8 / \sqrt{16}} = \frac{-3.8}{1.2} = -3.167
$$

**Step 3.** Find the critical value. For a left-sided test at $\alpha = 0.05$ with $15$ degrees of freedom:

$$
-t_{0.05,\, 15} = -1.753
$$

**Step 4.** Make the decision. Since $t = -3.167 < -1.753$, we reject $H_0$. There is sufficient evidence at the 5% level to conclude that the mean fill weight is less than 500 grams.

The $p$-value is $P(t_{15} < -3.167) \approx 0.003$, providing strong evidence against $H_0$.

## Assumptions

The one-sample $t$-test requires:

- **Random sampling**: The observations are independent and identically distributed.
- **Normality**: The population is normally distributed, or $n$ is large enough for the Central Limit Theorem to apply.

## Robustness

The $t$-test is fairly robust to departures from normality, particularly for moderate to large sample sizes. Simulation studies show that for $n \geq 30$, the actual Type I error rate stays close to the nominal $\alpha$ even for moderately skewed or heavy-tailed distributions. For smaller samples ($n < 15$), the test can be unreliable if the population is strongly skewed or has heavy tails.

Guidelines for when the normality assumption matters most:

- **$n < 15$**: Normality is important. Check with a Q-Q plot or Shapiro-Wilk test. Consider nonparametric alternatives (e.g., the Wilcoxon signed-rank test) if normality is questionable.
- **$15 \leq n < 30$**: Mild departures from normality are tolerable. The test is unreliable only for strongly skewed or heavy-tailed distributions.
- **$n \geq 30$**: The CLT ensures the sampling distribution of $\bar{X}$ is approximately normal, making the $t$-test reliable for most practical distributions.

!!! warning "Outliers Remain Problematic"
    Even with large $n$, individual outliers can inflate $S$ and shift $\bar{X}$, potentially masking a real effect or creating a spurious one. Always inspect the data for outliers before applying the $t$-test.
