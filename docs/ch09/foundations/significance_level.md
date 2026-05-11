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

## Exercises

**Exercise 1.**
A researcher sets $\alpha = 0.05$ and obtains a p-value of 0.03. Another researcher tests the same hypothesis at $\alpha = 0.01$. What does each conclude, and why can the same data lead to different conclusions?

??? success "Solution to Exercise 1"
    At $\alpha = 0.05$: since $p = 0.03 < 0.05$, the researcher rejects $H_0$. The result is statistically significant at the 5% level.

    At $\alpha = 0.01$: since $p = 0.03 > 0.01$, the researcher fails to reject $H_0$. The result is not significant at the 1% level.

    The same data lead to different conclusions because the significance level is a threshold chosen by the researcher before collecting data, reflecting how much Type I error risk they are willing to tolerate. A more conservative researcher (smaller $\alpha$) requires stronger evidence to reject $H_0$. The data themselves do not change -- only the decision rule changes.

---

**Exercise 2.**
Explain the difference between a Type I error and a Type II error. Which one is controlled by the significance level $\alpha$?

??? success "Solution to Exercise 2"

    - **Type I error (false positive):** Rejecting $H_0$ when $H_0$ is actually true. Probability = $\alpha$.
    - **Type II error (false negative):** Failing to reject $H_0$ when $H_0$ is actually false. Probability = $\beta$.

    The significance level $\alpha$ directly controls the Type I error rate: by setting $\alpha = 0.05$, we guarantee that if $H_0$ is true, we will falsely reject it at most 5% of the time (across repeated experiments).

    The Type II error rate $\beta$ depends on the true parameter value (effect size), the sample size, and $\alpha$. It is not directly set by the researcher but can be reduced by increasing $n$ or increasing $\alpha$. Power = $1 - \beta$ is the probability of correctly detecting a true effect.

---

**Exercise 3.**
A pharmaceutical company tests 20 independent drugs, each with a true null hypothesis ($H_0$ is true for all 20). If each test uses $\alpha = 0.05$, what is the expected number of false positives?

??? success "Solution to Exercise 3"
    Each test independently has a $\alpha = 0.05$ probability of a false positive. With 20 tests:

    $$
    E[\text{false positives}] = 20 \times 0.05 = 1.0
    $$

    The probability of at least one false positive is:

    $$
    P(\text{at least one FP}) = 1 - (1 - 0.05)^{20} = 1 - 0.95^{20} \approx 1 - 0.358 = 0.642
    $$

    So there is a 64.2% chance of at least one false discovery, even though every null hypothesis is true. This illustrates the multiple testing problem: the per-test error rate of 5% does not protect against the accumulation of errors across many tests.

---

**Exercise 4.**
A journal requires $\alpha = 0.05$ for publication. Explain the "file drawer problem" and how it can inflate the observed rate of false positives in the published literature beyond 5%.

??? success "Solution to Exercise 4"
    The **file drawer problem** (publication bias) refers to the tendency for studies with significant results ($p < 0.05$) to be published while studies with non-significant results ($p \geq 0.05$) remain unpublished ("filed away").

    This inflates the false positive rate in published literature because:

    1. If 20 labs independently study a null effect, on average 1 will get $p < 0.05$ by chance.
    2. That 1 lab publishes; the other 19 do not.
    3. Readers see a 100% hit rate (1 published study, 1 significant) instead of the true 5% rate (1 out of 20).

    The result is that the published literature over-represents false positives. Pre-registration (committing to publish regardless of results) and registered reports (peer review before data collection) are designed to mitigate this problem.
