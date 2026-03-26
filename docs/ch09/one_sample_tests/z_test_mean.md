# One-Sample Z-Test for the Mean

## Overview

The one-sample $z$-test is the simplest hypothesis test for a population mean. It applies when the population standard deviation $\sigma$ is known --- a situation that arises in quality control when historical process data provides a reliable estimate, or in standardized testing where the test's standard deviation is established from prior administrations. Although the known-$\sigma$ scenario is uncommon in practice, the $z$-test serves as the conceptual foundation for the $t$-test, which handles the more common case of unknown $\sigma$. Understanding the $z$-test first makes the transition to the $t$-test straightforward: the only change is replacing $\sigma$ with $S$ and the normal distribution with the $t$-distribution.

## Hypotheses

Let $X_1, X_2, \ldots, X_n$ be a random sample from a population with mean $\mu$ and known variance $\sigma^2$. The null hypothesis specifies a particular value for the mean:

$$
H_0\colon \mu = \mu_0
$$

The alternative hypothesis takes one of three forms:

| Alternative | Interpretation |
|---|---|
| $H_1\colon \mu \neq \mu_0$ | Two-sided: the mean differs from $\mu_0$ |
| $H_1\colon \mu > \mu_0$ | Right-sided: the mean exceeds $\mu_0$ |
| $H_1\colon \mu < \mu_0$ | Left-sided: the mean is below $\mu_0$ |

## Test Statistic

The test statistic measures how many standard errors the sample mean $\bar{X}$ falls from the hypothesized value $\mu_0$. A large absolute value indicates that the observed sample mean is far from what we would expect under $H_0$, providing evidence against the null.

$$
Z = \frac{\bar{X} - \mu_0}{\sigma / \sqrt{n}}
$$

The denominator $\sigma / \sqrt{n}$ is the standard error of $\bar{X}$, so $Z$ expresses the discrepancy between $\bar{X}$ and $\mu_0$ in units of the sampling variability.

Under $H_0$ and either of the following conditions, $Z$ follows a standard normal distribution:

- **Exact**: The population is normally distributed, or
- **Approximate**: The sample size $n$ is large enough for the Central Limit Theorem to apply.

$$
Z \sim N(0, 1) \quad \text{under } H_0
$$

## Rejection Regions

At significance level $\alpha$, the rejection region depends on the alternative:

**Two-sided** ($H_1\colon \mu \neq \mu_0$): Reject $H_0$ if

$$
|Z| > z_{\alpha/2}
$$

**Right-sided** ($H_1\colon \mu > \mu_0$): Reject $H_0$ if

$$
Z > z_\alpha
$$

**Left-sided** ($H_1\colon \mu < \mu_0$): Reject $H_0$ if

$$
Z < -z_\alpha
$$

Here $z_\alpha$ denotes the upper $\alpha$ critical value of the standard normal, satisfying $P(Z > z_\alpha) = \alpha$.

## P-Values

The $p$-value measures the strength of evidence against $H_0$ by computing the probability of observing a test statistic as extreme as (or more extreme than) the observed value $z_{\text{obs}}$, assuming $H_0$ is true.

| Alternative | P-value formula |
|---|---|
| $H_1\colon \mu \neq \mu_0$ | $p = 2\,P(Z > \|z_{\text{obs}}\|) = 2\bigl[1 - \Phi(\|z_{\text{obs}}\|)\bigr]$ |
| $H_1\colon \mu > \mu_0$ | $p = P(Z > z_{\text{obs}}) = 1 - \Phi(z_{\text{obs}})$ |
| $H_1\colon \mu < \mu_0$ | $p = P(Z < z_{\text{obs}}) = \Phi(z_{\text{obs}})$ |

Reject $H_0$ whenever $p < \alpha$.

## Worked Example

A battery manufacturer uses a production line whose fill weights have a known standard deviation of $\sigma = 3$ grams (established from years of quality control data). The target mean weight is $\mu_0 = 50$ grams. An inspector takes a random sample of $n = 36$ batteries and finds $\bar{x} = 49.1$ grams. Test whether the mean weight differs from 50 grams at $\alpha = 0.05$.

**Step 1.** State the hypotheses:

$$
H_0\colon \mu = 50 \qquad H_1\colon \mu \neq 50
$$

**Step 2.** Compute the test statistic:

$$
Z = \frac{49.1 - 50}{3 / \sqrt{36}} = \frac{-0.9}{0.5} = -1.80
$$

**Step 3.** Find the critical values. For a two-sided test at $\alpha = 0.05$:

$$
z_{\alpha/2} = z_{0.025} = 1.96
$$

**Step 4.** Make the decision. Since $|Z| = 1.80 < 1.96$, we fail to reject $H_0$. The data do not provide sufficient evidence at the 5% level to conclude that the mean weight differs from 50 grams.

**Step 5.** Compute the $p$-value:

$$
p = 2[1 - \Phi(1.80)] = 2(0.0359) = 0.0718
$$

Since $p = 0.072 > 0.05$, this confirms the failure to reject $H_0$.

!!! tip "Interpreting Marginal Results"
    With $p = 0.072$, the result is not statistically significant at $\alpha = 0.05$ but would be significant at $\alpha = 0.10$. In practice, the inspector might recommend increasing the sample size to obtain a more definitive conclusion, since the observed departure ($\bar{x} = 49.1$ vs $\mu_0 = 50$) could represent a real but small shift.

## Assumptions

The one-sample $z$-test requires:

- **Known $\sigma$**: The population standard deviation is a known constant, not estimated from the current sample.
- **Independence**: The observations $X_1, \ldots, X_n$ are independent.
- **Normality or large $n$**: Either the population is normally distributed (for exact inference) or $n$ is large enough (typically $n \geq 30$) for the CLT to provide a good approximation.

!!! warning "Known sigma Is Rare"
    The assumption that $\sigma$ is known is strong and seldom met outside of controlled industrial settings with extensive historical data. When $\sigma$ must be estimated from the sample, use the one-sample $t$-test instead, which accounts for the additional uncertainty in estimating $\sigma$.
