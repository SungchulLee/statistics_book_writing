# Type I and Type II Errors

## Overview

When conducting a hypothesis test, two potential types of errors can occur. Understanding these errors is essential for correctly interpreting the results of statistical tests and for designing studies that minimize the risk of incorrect conclusions.

## Type I Error (False Positive)

A **Type I error** occurs when the null hypothesis $H_0$ is true, but we mistakenly reject it in favor of the alternative hypothesis $H_a$. This is analogous to convicting an innocent person in a trial.

$$\alpha = P(\text{Type I Error}) = P(\text{Reject } H_0 \mid H_0 \text{ is true})$$

The **significance level** $\alpha$ represents the maximum tolerable probability of committing a Type I error. Common choices are $\alpha = 0.05$, $0.01$, and $0.10$.

**Example**: A pharmaceutical company tests a new drug that has no real effect. If the study incorrectly concludes the drug is effective (rejects $H_0$), that is a Type I error.

## Type II Error (False Negative)

A **Type II error** occurs when the null hypothesis $H_0$ is false, but we fail to reject it. This is analogous to acquitting a guilty person in a trial.

$$\beta = P(\text{Type II Error}) = P(\text{Fail to reject } H_0 \mid H_a \text{ is true})$$

A lower $\beta$ implies a lower risk of retaining a false null hypothesis.

**Example**: A drug truly works, but the study fails to detect the effect and concludes there is no significant difference from placebo. That is a Type II error.

## Summary Table

| | $H_0$ is true | $H_0$ is false |
|---|---|---|
| **Reject $H_0$** | Type I Error ($\alpha$) | Correct Decision (Power = $1 - \beta$) |
| **Fail to reject $H_0$** | Correct Decision | Type II Error ($\beta$) |

## The Tradeoff Between alpha and beta
There is an inherent tradeoff between the two error types:

- **Decreasing $\alpha$** (making it harder to reject $H_0$) reduces the chance of a Type I error but increases the chance of a Type II error.
- **Increasing $\alpha$** makes it easier to reject $H_0$, reducing $\beta$ but increasing the chance of a false positive.

The appropriate balance depends on the context:

- In medical trials, a Type I error (approving an ineffective drug) may be very costly, so $\alpha$ is set low.
- In screening tests, a Type II error (missing a disease) may be more costly, so higher $\alpha$ (greater sensitivity) is preferred.

## Factors Affecting Error Rates

Several factors influence the probability of committing each type of error:

- **Sample size ($n$)**: Larger samples reduce both types of errors by providing more precise estimates.
- **Effect size**: Larger true effects are easier to detect, reducing $\beta$.
- **Significance level ($\alpha$)**: Directly controls Type I error rate.
- **Variability in the data**: Higher variance makes it harder to detect true effects, increasing $\beta$.

## Example: Unemployment Rate

The mayor tests $H_0: p = 0.09$ vs $H_1: p \neq 0.09$.

- **Type I Error**: The town's unemployment rate truly is 9%, but the mayor incorrectly concludes it is different from 9%.
- **Type II Error**: The town's unemployment rate truly differs from 9%, but the mayor fails to detect this difference.

## Exercises

**Exercise 1.**
**Definitions.** State Type I, Type II errors. Provide a contingency table.

??? success "Solution to Exercise 1"
    |  | $H_0$ true | $H_0$ false |
    |---|---|---|
    | Reject $H_0$ | **Type I error** ($\alpha$) | Correct ($1 - \beta$, power) |
    | Fail to reject | Correct ($1 - \alpha$) | **Type II error** ($\beta$) |

    Type I: false positive ($\alpha$ = significance level).
    Type II: false negative ($\beta = 1 - $ power).

---

**Exercise 2.**
Mayor tests town's unemployment rate vs national 9%: $H_0: p = 0.09$. Describe Type I and Type II errors in this context.

??? success "Solution to Exercise 2"
    **Type I:** town's rate is actually 9% ($H_0$ true), but the mayor concludes it's different (rejects $H_0$). Costs: incorrect press release, unnecessary policy response.

    **Type II:** town's rate differs from 9% ($H_0$ false), but mayor fails to detect (fails to reject). Costs: missed insight, mistaken belief that town matches national average.

---

**Exercise 3.**
**Type I/II trade-off.** Why can't we set both $\alpha$ and $\beta$ to zero?

??? success "Solution to Exercise 3"
    With fixed sample size, decreasing $\alpha$ (making rejection harder) typically *increases* $\beta$ (more failures to reject true alternatives).

    To decrease both simultaneously: increase sample size $n$. The trade-off is unavoidable for fixed $n$.

    Extreme cases:

    - Never reject: $\alpha = 0$, $\beta = 1$ (always miss true effects).
    - Always reject: $\alpha = 1$, $\beta = 0$ (always detect; but always false alarm).

    Practical balance: pick $\alpha = 0.05$ first, then design $n$ to achieve desired $\beta$ (typically 0.20 for 80% power).

---

**Exercise 4.**
**Power.** Define power. For drug trial $H_0: \mu = 0$ vs $H_1: \mu = 5$, $\sigma = 10$, $\alpha = 0.05$ (one-sided), $n = 30$: compute power.

??? success "Solution to Exercise 4"
    Power = $1 - \beta = P(\text{reject} \mid H_1$ true).

    Under $H_0$: $\bar X \sim N(0, 100/30)$, $\mathrm{SE} = \sqrt{100/30} \approx 1.83$. Reject if $\bar X > z_{0.05} \cdot 1.83 = 3.00$.

    Under $H_1$ ($\mu = 5$): $\bar X \sim N(5, 100/30)$. Power = $P(\bar X > 3 \mid \mu = 5) = P(Z > (3 - 5)/1.83) = P(Z > -1.09) = 0.862$.

    Power ≈ 86%. Reasonable but not great. Conventional target: 80%.

---

**Exercise 5.**
**Sample-size for target power.** What $n$ achieves 90% power in Exercise 4's setup?

??? success "Solution to Exercise 5"
    For power $1 - \beta$ at one-sided test, sample size:

    $$
    n = \left(\frac{(z_\alpha + z_\beta)\sigma}{\mu - \mu_0}\right)^2
    $$

    Here $z_{0.05} = 1.645$, $z_{0.10} = 1.282$, $\sigma = 10$, effect = 5:

    $n = ((1.645 + 1.282) \cdot 10/5)^2 = (5.854)^2 \approx 34.3$. Round up: $n = 35$.

    Slightly larger than the $n = 30$ in Exercise 4 (which gave 86% power). The formula confirms the trade-off: more power requires more data.

---

**Exercise 6.**
**Multiple testing.** If you run 100 tests at $\alpha = 0.05$, what is the expected number of false rejections under all-true $H_0$?

??? success "Solution to Exercise 6"
    Under all-true $H_0$, each test rejects with probability $0.05$ independently. Expected number of rejections: $100 \cdot 0.05 = 5$.

    Family-wise probability of at least one false rejection: $1 - (0.95)^{100} \approx 0.994$ — almost certain to see "significant" results from pure noise.

    **Corrections:**

    - **Bonferroni:** test at $\alpha/m = 0.0005$. Conservative; family-wise error $\le 0.05$.
    - **Holm-Bonferroni:** step-down, slightly more powerful.
    - **Benjamini-Hochberg (FDR):** controls expected proportion of false rejections among declared positives. Less conservative.

    Pre-registration and Bonferroni-like corrections are essential for valid inference in high-dimensional settings (genomics, A/B testing platforms).
