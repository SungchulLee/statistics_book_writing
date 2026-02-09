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

## The Tradeoff Between $\alpha$ and $\beta$

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
