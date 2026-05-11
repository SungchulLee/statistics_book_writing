# One-Sample t-Test for the Mean

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

## Exercises

**Exercise 1.**
Cereal: claimed mean 500g. Sample $n = 25$, $\bar X = 490$, $s = 15$. Test at $\alpha = 0.01$.

??? success "Solution to Exercise 1"
    $H_0: \mu = 500$ vs $H_1: \mu \ne 500$.

    $t = (490 - 500)/(15/\sqrt{25}) = -10/3 = -3.33$.

    Critical: $t_{0.005, 24} = \pm 2.797$. $|t| = 3.33 > 2.797$. **Reject.**

    Strong evidence the mean is below 500g.

---

**Exercise 2.**
**Conditions for $t$-test.** State and explain.

??? success "Solution to Exercise 2"
    1. **Random sample / independence:** $X_i$ are independent (or at least exchangeable). For sampling without replacement, the sampling fraction should be < 10%.

    2. **Normality:** the underlying population is approximately normal, OR $n$ is large enough for the CLT to give $\bar X$ approximate normality.

    For small $n$ (< 30) with non-normal data, $t$-test may have wrong size. Check normality via Q-Q plot.

    For very heavy-tailed or skewed data: use Wilcoxon signed-rank (non-parametric) or bootstrap test.

---

**Exercise 3.**
**One-sided test.** Same data as Exercise 1, but test $H_1: \mu < 500$ (suspect mean below claim).

??? success "Solution to Exercise 3"
    Critical: $t_{0.01, 24} = -2.492$. $t = -3.33 < -2.492$. **Reject.**

    P-value: $P(T_{24} < -3.33) \approx 0.0014$.

    Decision unchanged: still reject. But the p-value is half of the two-sided value, reflecting that we put all $\alpha$ on one tail.

---

**Exercise 4.**
**Effect size.** Compute Cohen's $d$ for the cereal exercise.

??? success "Solution to Exercise 4"
    Cohen's $d = (\bar X - \mu_0)/s = (490 - 500)/15 = -0.67$.

    Interpretation:

    - $|d| = 0.2$: small.
    - $|d| = 0.5$: medium.
    - $|d| = 0.8$: large.

    $d = -0.67$ is "medium-large." Statistical significance + meaningful effect size — both signal a real problem with cereal weights.

    Always report effect size alongside p-value. P-value alone (especially with large $n$) can flag trivial effects.

---

**Exercise 5.**
**Sample size for power.** What $n$ gives 90% power to detect a 5g decrease at $\alpha = 0.01$, assuming $\sigma \approx 15$?

??? success "Solution to Exercise 5"
    $n = ((z_{\alpha/2} + z_\beta) \sigma/\Delta)^2$ (approximate using $z$ since $n$ will be moderate).

    $z_{0.005} = 2.576$, $z_{0.10} = 1.282$. $\Delta = 5$, $\sigma = 15$.

    $n = ((2.576 + 1.282) \cdot 15/5)^2 = (11.57)^2 \approx 134$.

    Detecting smaller effects with high power requires larger samples. To detect 10g: $n \approx 34$. Quadratic in $1/\Delta$.

---

**Exercise 6.**
**Multiple testing.** A QA engineer runs $t$-tests on 20 production lines. At $\alpha = 0.05$, what's the family-wise false-positive rate under all-true nulls?

??? success "Solution to Exercise 6"
    $P(\text{at least one false rejection}) = 1 - (1 - 0.05)^{20} \approx 0.642$.

    Expected number of false rejections: $20 \cdot 0.05 = 1$.

    To control family-wise error at $\alpha = 0.05$: use Bonferroni — test each at $\alpha/20 = 0.0025$. Very conservative.

    Alternative — FDR (Benjamini-Hochberg): controls expected proportion of false positives among declared positives. Less conservative; standard in high-throughput testing.

    Without correction, false alarms are nearly guaranteed in multi-test scenarios.
