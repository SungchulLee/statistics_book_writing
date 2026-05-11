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
| $H_1\colon \mu \neq \mu_0$ | $p = 2\,P(Z > \|z_{\text{obs}}\|) = 2\bigl[1 - \mathcal{N}(\|z_{\text{obs}}\|)\bigr]$ |
| $H_1\colon \mu > \mu_0$ | $p = P(Z > z_{\text{obs}}) = 1 - \mathcal{N}(z_{\text{obs}})$ |
| $H_1\colon \mu < \mu_0$ | $p = P(Z < z_{\text{obs}}) = \mathcal{N}(z_{\text{obs}})$ |

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
p = 2[1 - \mathcal{N}(1.80)] = 2(0.0359) = 0.0718
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

## Exercises

**Exercise 1.**
Manufacturer claims mean lifetime 1200 hrs. Sample: $n = 36$, $\bar X = 1150$, $s = 200$. Test at $\alpha = 0.05$.

??? success "Solution to Exercise 1"
    $H_0: \mu = 1200$ vs $H_1: \mu \ne 1200$.

    $z = (1150 - 1200)/(200/\sqrt{36}) = -50/33.3 \approx -1.5$.

    Critical: $\pm 1.96$. $|z| = 1.5 < 1.96$. **Fail to reject.**

    Not enough evidence to dispute the claim. P-value: $2 \cdot \Phi(-1.5) = 2 \cdot 0.067 = 0.134$.

---

**Exercise 2.**
**Power calculation.** For Exercise 1, what is the power if the true mean is 1100?

??? success "Solution to Exercise 2"
    Under $H_0$: rejection regions $\bar X < 1200 - 1.96 \cdot 33.3 = 1134.7$ or $> 1265.3$.

    Under $H_1: \mu = 1100$, $\bar X \sim N(1100, 33.3^2)$.

    Power = $P(\bar X < 1134.7 \mid \mu = 1100) + P(\bar X > 1265.3 \mid \mu = 1100)$.

    First: $P(Z < (1134.7 - 1100)/33.3) = P(Z < 1.04) \approx 0.85$.

    Second: negligible (huge $Z$).

    Power $\approx 85\%$. Reasonable — if the true mean is 100 hrs below claim, we have 85% chance of detecting.

---

**Exercise 3.**
**One-sided alternative.** Modify Exercise 1 to test $H_1: \mu < 1200$ (suspect mean is lower). Recompute decision.

??? success "Solution to Exercise 3"
    $H_0: \mu \ge 1200$ vs $H_1: \mu < 1200$.

    Critical region: $z < -z_{0.05} = -1.645$.

    $z = -1.5 > -1.645$. **Fail to reject** but just barely.

    P-value: $\Phi(-1.5) = 0.067$ — close to 0.05.

    One-sided test gives different decision at the margin. If pre-specified directional hypothesis is justified, one-sided is more powerful.

---

**Exercise 4.**
**Required sample size.** What $n$ provides 80% power to detect $\mu = 1100$ at $\alpha = 0.05$ (two-sided)?

??? success "Solution to Exercise 4"
    $n = ((z_{\alpha/2} + z_\beta) \sigma / \Delta)^2$ where $\Delta = |\mu_1 - \mu_0|$.

    $\sigma = 200$, $\Delta = 100$. $z_{0.025} = 1.96$, $z_{0.20} = 0.84$.

    $n = ((1.96 + 0.84) \cdot 200/100)^2 = (5.60)^2 \approx 31.4$. Round up: $n = 32$.

    With $n = 36$ from Exercise 1, we exceed this — that's why the power is 85% rather than 80%.

---

**Exercise 5.**
**z-test vs t-test.** When is each appropriate?

??? success "Solution to Exercise 5"
    **z-test:** $\sigma$ known OR sample large enough that $s$ is essentially exact. Rare in practice.

    **t-test:** $\sigma$ unknown, estimated by $s$. Default for one-sample tests with continuous data.

    Practical guide:

    - If $n \ge 30$: $t$-test critical values $\approx z$ values. Difference negligible.
    - If $n < 30$: $t$ values are larger than $z$ — test is less likely to reject for the same data. Reflects extra uncertainty from estimating $\sigma$.

    Most software automatically uses $t$ even when "z-test" is requested by intuition. SciPy's `ttest_1samp` is the standard.

---

**Exercise 6.**
**Multiple comparisons.** A quality engineer tests 20 production lines for $\mu = $ target. At least how many would she expect to falsely reject at $\alpha = 0.05$?

??? success "Solution to Exercise 6"
    Under all-true $H_0$: expected false rejections = $20 \cdot 0.05 = 1$. Family-wise probability of $\ge 1$ false rejection: $1 - (0.95)^{20} \approx 0.642$.

    Even with all lines truly on target, ~64% chance of finding "evidence" of a problem somewhere. This is the multiple-testing problem.

    Corrections:

    - **Bonferroni:** test at $\alpha/20 = 0.0025$. Family-wise error $\le 0.05$.
    - **FDR control:** Benjamini-Hochberg, controls expected proportion of false rejections among declared positives.

    Without correction: 20 tests at $\alpha = 0.05$ give substantial false-discovery risk. Document the number of tests performed; preregister hypotheses.
