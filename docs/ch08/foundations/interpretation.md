# Interpretation and Common Misconceptions

## The Repeated-Sampling Interpretation

A 95% confidence interval does **not** mean "there is a 95% probability that $\mu$ is in this interval." The parameter $\mu$ is a fixed (but unknown) number — it either is or is not in the interval.

The correct interpretation: if we were to repeat the sampling process many times, each time constructing a 95% CI, then **approximately 95% of those intervals would contain the true parameter**.

### Formal Statement

Let $X_1, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$ with $\sigma$ known. The interval

$$\bar{X} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

satisfies:

$$P\left(\bar{X} - z_{\alpha/2}\frac{\sigma}{\sqrt{n}} \leq \mu \leq \bar{X} + z_{\alpha/2}\frac{\sigma}{\sqrt{n}}\right) = 1 - \alpha$$

The probability statement is about the **random endpoints** $\bar{X} \pm z_{\alpha/2}\sigma/\sqrt{n}$, not about $\mu$.

### Simulation Demonstration

```python
import numpy as np
np.random.seed(42)

mu, sigma, n = 50, 10, 30
alpha = 0.05
z = 1.96
n_simulations = 1000

covers = 0
for _ in range(n_simulations):
    sample = np.random.normal(mu, sigma, n)
    xbar = sample.mean()
    me = z * sigma / np.sqrt(n)
    lower, upper = xbar - me, xbar + me
    if lower <= mu <= upper:
        covers += 1

print(f"Coverage: {covers}/{n_simulations} = {covers/n_simulations:.3f}")
# ≈ 0.950
```

## Common Misconceptions

### Misconception 1: "95% probability that μ is in this interval"

After computing $[48.2, 51.8]$, the statement "there is a 95% probability that $\mu$ is between 48.2 and 51.8" is **wrong**. Either $\mu$ is in that interval or it is not — there is no randomness left.

The 95% refers to the **procedure**, not any single interval.

### Misconception 2: "95% of the data falls in the interval"

A CI estimates a **parameter** (like the population mean), not the range of individual observations. The interval $\bar{X} \pm z_{\alpha/2}\sigma/\sqrt{n}$ shrinks with $n$, while the range of data does not.

### Misconception 3: "If two CIs overlap, the difference is not significant"

Two 95% CIs can overlap even when the difference between parameters is statistically significant. The proper comparison uses a CI for the **difference** $\mu_1 - \mu_2$.

## Width, Confidence Level, and Sample Size

The margin of error for a z-interval is:

$$E = z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

Three relationships follow:

1. **Higher confidence → wider interval.** Increasing from 95% to 99% increases $z_{\alpha/2}$ from 1.96 to 2.576, widening the interval by 31%.

2. **Larger sample → narrower interval.** The margin of error decreases as $1/\sqrt{n}$. To halve the width, you need 4 times the sample size.

3. **Larger variance → wider interval.** More variability in the population makes estimation harder.

### Sample Size Determination

To achieve a desired margin of error $E$ at confidence level $1 - \alpha$:

$$n = \left(\frac{z_{\alpha/2} \cdot \sigma}{E}\right)^2$$

**Example:** To estimate a population mean within $\pm 2$ units with 95% confidence, given $\sigma = 10$:

$$n = \left(\frac{1.96 \times 10}{2}\right)^2 = 96.04 \implies n = 97$$

### Common Confidence Levels

| Confidence Level | $\alpha$ | $z_{\alpha/2}$ |
|---|---|---|
| 90% | 0.10 | 1.645 |
| 95% | 0.05 | 1.960 |
| 99% | 0.01 | 2.576 |

## One-Sided Confidence Intervals (Confidence Bounds)

Sometimes we only need a bound in one direction:

- **Upper bound:** $\mu \leq \bar{X} + z_\alpha \cdot \sigma/\sqrt{n}$ (with confidence $1 - \alpha$)
- **Lower bound:** $\mu \geq \bar{X} - z_\alpha \cdot \sigma/\sqrt{n}$ (with confidence $1 - \alpha$)

Note that one-sided bounds use $z_\alpha$ (not $z_{\alpha/2}$). A 95% one-sided bound uses $z_{0.05} = 1.645$.

**Financial example:** A risk manager may want an upper bound on portfolio loss: "We are 95% confident that the expected loss does not exceed $X."

## Exercises

**Exercise 1.**
95% CI for cholesterol = $(188.3, 205.7)$. (a) Which interpretation is correct? (b) Point estimate and ME?

??? success "Solution to Exercise 1"
    (a) Only **(ii)** is correct: "If we repeated this study many times, about 95% of the resulting intervals would contain the true mean."

    (i) is wrong: it assigns probability to the fixed parameter (frequentist error). The parameter is fixed; the interval is random.

    (iii) is wrong: confuses CI for the mean with the spread of data. The CI is about the population mean, not the range of individual cholesterol levels.

    (b) $\bar X = (188.3 + 205.7)/2 = 197.0$. ME = $(205.7 - 188.3)/2 = 8.7$.

---

**Exercise 2.**
**Why the parameter doesn't have a 95% probability.** Explain why the frequentist CI doesn't allow probability statements about $\mu$.

??? success "Solution to Exercise 2"
    In frequentist statistics, $\mu$ is a **fixed constant**, not a random variable. Random variables have probability distributions; constants do not. A statement like "$P(\mu \in (188, 206)) = 0.95$" is meaningless because $\mu$ is either inside the interval (probability 1) or outside (probability 0) — there is no randomness in $\mu$.

    What IS random: the **interval** itself (its endpoints depend on the sample). The 95% refers to the **procedure**: among all 95% CIs constructed from many samples, 95% will trap $\mu$.

    For a probability statement about $\mu$, you need Bayesian inference, where $\mu$ has a prior distribution and the posterior gives "$P(\mu \in \text{interval} \mid \text{data}) = 0.95$" — a **credible interval**.

---

**Exercise 3.**
**Coverage probability.** Define coverage and explain why a 95% CI might have actual coverage lower than 95%.

??? success "Solution to Exercise 3"
    **Coverage probability:** $P(\theta \in \text{CI})$ — the probability the procedure traps the true parameter. Designed to equal $1 - \alpha$.

    **Reasons for actual coverage < nominal:**

    - **Approximation error:** CI based on asymptotic theory (CLT) at finite $n$. Wald CI for proportion has coverage well below 95% near $p = 0$ or $p = 1$.
    - **Distribution misspecification:** $t$-interval assumes normality; for skewed data with small $n$, coverage can be 85-90% instead of 95%.
    - **Garden of forking paths:** if the procedure is chosen after seeing data, "CI" is not a proper CI — coverage may be far from nominal.
    - **Multiple-testing:** the CI's $1 - \alpha$ guarantee is per CI; making 20 CIs and reporting the "interesting" one inflates miscoverage.

    Wilson CI for proportion was developed specifically to fix Wald's poor coverage. Always check coverage via simulation when in doubt.

---

**Exercise 4.**
**Width and confidence level.** How does CI width change with (a) higher confidence level, (b) larger sample size, (c) higher population variance?

??? success "Solution to Exercise 4"
    CI width = $2 \cdot z_{\alpha/2} \cdot \mathrm{SE} = 2 z_{\alpha/2} \sigma/\sqrt n$.

    (a) **Higher confidence** ($1 - \alpha = 0.99$ vs 0.95): $z_{0.005} = 2.576$ vs $z_{0.025} = 1.96$. Width grows by 31%. More confidence = wider interval.

    (b) **Larger $n$:** width shrinks as $1/\sqrt n$. Quadruple $n$ → halve width.

    (c) **Higher $\sigma$:** width grows linearly with $\sigma$. Doubling SD → doubling width.

    Practical: to achieve target ME, $n = (z_{\alpha/2} \sigma/\mathrm{ME})^2$. Quadratic in both $z$ and $\sigma$, inverse-square in ME.

---

**Exercise 5.**
**One-sided CI.** When is a one-sided CI appropriate? Give one example.

??? success "Solution to Exercise 5"
    One-sided CI: $(-\infty, \hat\theta + z_\alpha \mathrm{SE})$ or $(\hat\theta - z_\alpha \mathrm{SE}, \infty)$, depending on direction of interest.

    Appropriate when only one direction matters:

    - **Quality control:** "guaranteed at most 5% defect rate" — upper one-sided CI for $p$.
    - **Equivalence/non-inferiority:** new treatment is *no worse* than old — lower one-sided CI for treatment difference.
    - **Drug toxicity:** dose limit guaranteed to be exceeded with <1% probability.

    Advantage: tighter bound on the side of interest (uses entire $\alpha$ on one tail). Disadvantage: no information about the other side.

    Default for scientific reporting is two-sided unless one-sided is well-justified. Regulatory contexts often require one-sided.

---

**Exercise 6.**
**Bayesian credible intervals.** What is a Bayesian 95% credible interval and how does it differ from a frequentist CI?

??? success "Solution to Exercise 6"
    **Bayesian 95% credible interval:** $(\theta_L, \theta_U)$ such that posterior probability $P(\theta_L \le \theta \le \theta_U \mid x) = 0.95$.

    **Frequentist CI:** interval whose construction procedure has 95% coverage over repeated sampling.

    **Differences:**

    - **Interpretation:** Bayesian directly assigns probability to $\theta$; frequentist's probability is about the procedure.
    - **Subjectivity:** Bayesian requires a prior; frequentist requires the sampling distribution.
    - **Asymptotic agreement:** with informative data, both converge to similar intervals (Bernstein-von Mises theorem).
    - **Small-sample:** can differ significantly. Bayesian CI may be much narrower if the prior is informative.

    Which to report depends on philosophy and audience. Modern practice often uses both: Bayesian for inference, frequentist for testing hypothesis $H_0: \theta = \theta_0$. Many physicists report both in major experiments.
