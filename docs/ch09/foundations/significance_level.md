# Significance Level and Decision Rules

## Overview

After formulating null and alternative hypotheses, the next step is to decide how much Type I error risk to tolerate. The **significance level** $\alpha$ is the maximum probability of rejecting $H_0$ when it is actually true. This threshold is chosen *before* any data is collected, and it controls the long-run false positive rate. Choosing $\alpha$ involves balancing the cost of a false rejection against the cost of failing to detect a real effect.

## Decision Rules

Once $\alpha$ is fixed, there are two equivalent ways to reach a reject-or-fail-to-reject decision. Both approaches always produce the same conclusion.

- **P-value approach.** Compute the p-value from the data and reject $H_0$ if $p \leq \alpha$. The p-value measures how extreme the observed data are under $H_0$, so comparing it to $\alpha$ directly answers whether the evidence exceeds the pre-set threshold.

- **Critical value approach.** Determine the critical value(s) that mark the boundary of the rejection region at level $\alpha$, then reject $H_0$ if the test statistic falls in that region. Rejecting via the critical value is equivalent to finding $p \leq \alpha$.

??? example "Worked Example: Two Approaches, Same Decision"

    Suppose we test $H_0: \mu = 50$ versus $H_1: \mu \neq 50$ at $\alpha = 0.05$ with known $\sigma = 10$ and a sample of $n = 25$ yielding $\bar{x} = 53.5$.

    **Critical value approach.** The rejection region is $|Z| > z_{0.025} = 1.96$. The test statistic is

    $$
    Z = \frac{53.5 - 50}{10/\sqrt{25}} = \frac{3.5}{2} = 1.75
    $$

    Since $|1.75| < 1.96$, we fail to reject $H_0$.

    **P-value approach.** The two-sided p-value is $2\,P(Z > 1.75) \approx 2(0.0401) = 0.0802$. Since $0.0802 > 0.05$, we again fail to reject $H_0$. Both approaches agree.

## Common Significance Levels

The choice of $\alpha$ depends on the practical consequences of a Type I error. More serious consequences call for a smaller $\alpha$, which makes the test more stringent but harder to reject $H_0$.

| $\alpha$ | Stringency | Typical use |
|---|---|---|
| 0.10 | More lenient | Exploratory studies, screening |
| 0.05 | Moderate | Most scientific research (conventional default) |
| 0.01 | Stringent | Studies where false positives are costly |
| 0.001 | Very stringent | Genomics, particle physics |

!!! warning "Alpha is not evidence"

    The significance level $\alpha$ is a pre-specified error-rate control, not a measure of evidence. It is the p-value, computed from the data, that quantifies how incompatible the observed results are with $H_0$. A common mistake is to describe $\alpha = 0.01$ as requiring "stronger evidence," when in fact it simply sets a more demanding threshold that the p-value must clear.
