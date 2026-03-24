# Family-Wise Error Rate

## Why Multiple Tests Inflate False Positives

When a researcher performs a single hypothesis test at significance level $\alpha = 0.05$, there is a 5% chance of incorrectly rejecting a true null hypothesis. This error rate is acceptable for one test in isolation. However, modern studies frequently test many hypotheses simultaneously -- a genome-wide association study may test hundreds of thousands of genetic variants, and a clinical trial may evaluate a treatment across multiple endpoints. As the number of tests grows, so does the probability that **at least one** true null hypothesis is incorrectly rejected, even when every individual test is conducted at the same $\alpha$. The family-wise error rate (FWER) formalizes this cumulative risk.

## Definition

Consider a family of $m$ hypothesis tests $H_1, H_2, \ldots, H_m$, each tested at individual significance level $\alpha$. Let $V$ denote the number of true null hypotheses that are incorrectly rejected (false positives). The **family-wise error rate** is the probability of committing at least one Type I error across the entire family:

$$
\text{FWER} = P(V \geq 1)
$$

In words, FWER is the probability that at least one of the $m$ tests produces a false rejection. Strong control of FWER at level $\alpha$ means that $\text{FWER} \leq \alpha$ regardless of which hypotheses are true and which are false.

## FWER Under Independence

To see how rapidly the false positive risk grows, consider the special case where all $m$ tests are independent and all $m$ null hypotheses are true. Each test has probability $\alpha$ of a false rejection and probability $1 - \alpha$ of a correct non-rejection. Because the tests are independent, the probability that **none** of them produces a false rejection is:

$$
P(V = 0) = (1 - \alpha)^m
$$

Therefore the FWER is:

$$
\text{FWER} = 1 - (1 - \alpha)^m
$$

This expression grows quickly with $m$. The following table illustrates the FWER when each test uses $\alpha = 0.05$:

| Number of tests $m$ | FWER |
|---|---|
| 1 | 0.050 |
| 5 | 0.226 |
| 10 | 0.401 |
| 20 | 0.642 |
| 50 | 0.923 |
| 100 | 0.994 |

With just 20 independent tests, there is a 64% chance of at least one false positive. With 100 tests, the chance exceeds 99%.

## The Bonferroni (Union) Bound

The formula $1 - (1 - \alpha)^m$ requires the tests to be independent. In practice, test statistics are often correlated -- for example, when testing overlapping sets of variables or when the same data set is used for all tests. A more general result comes from Boole's inequality (the union bound).

For **any** dependency structure among the $m$ tests:

$$
\text{FWER} = P\!\left(\bigcup_{i=1}^{m} A_i\right) \leq \sum_{i=1}^{m} P(A_i)
$$

where $A_i$ is the event that test $i$ falsely rejects a true null hypothesis. If each test is conducted at level $\alpha$, then $P(A_i) \leq \alpha$ for each $i$, giving:

$$
\text{FWER} \leq m\alpha
$$

This bound holds without any assumption about the dependence among tests. It is the mathematical foundation of the Bonferroni correction: to control FWER at level $\alpha$, test each hypothesis at the adjusted level $\alpha / m$.

??? note "Relationship between the exact formula and the bound"
    For small $\alpha$, the first-order Taylor expansion of $(1 - \alpha)^m$ gives $(1 - \alpha)^m \approx 1 - m\alpha$, so the exact independent-case FWER $1 - (1 - \alpha)^m \approx m\alpha$. The Bonferroni bound $m\alpha$ is therefore a good approximation when $m\alpha$ is small. However, when $m$ is large, $m\alpha$ can exceed 1, whereas the true FWER is always at most 1. The bound is loose precisely when the number of tests is large relative to $\alpha$.

## Strong vs Weak Control

There is an important distinction between two modes of FWER control:

- **Weak control**: FWER $\leq \alpha$ under the **global null hypothesis**, the configuration where every null hypothesis is true. Weak control guards against false positives only when there are no real effects.
- **Strong control**: FWER $\leq \alpha$ under **every possible configuration** of true and false nulls. Strong control is the standard requirement because in practice we do not know which hypotheses are true.

Procedures such as the Bonferroni correction and [Holm's step-down method](bonferroni_holm.md) provide strong FWER control.

## Example: Testing Five Fertilizers

A researcher tests whether each of five fertilizers increases crop yield relative to a control, using a separate t-test for each fertilizer at $\alpha = 0.05$. If none of the fertilizers actually works (all five null hypotheses are true), the FWER is:

$$
\text{FWER} = 1 - (1 - 0.05)^5 = 1 - 0.9510 \approx 0.226
$$

There is roughly a 23% chance that at least one fertilizer appears to work purely by chance. To control FWER at 0.05, the Bonferroni correction tests each hypothesis at $\alpha / 5 = 0.01$. The resulting FWER under independence becomes:

$$
\text{FWER}_{\text{Bonferroni}} = 1 - (1 - 0.01)^5 \approx 0.049
$$

which is below the target of 0.05.

## Example: Genomics Screening

In a genome-wide association study, a researcher tests $m = 500{,}000$ genetic variants for association with a disease. At the individual level $\alpha = 0.05$:

$$
\text{FWER} \leq m\alpha = 500{,}000 \times 0.05 = 25{,}000
$$

The Bonferroni bound exceeds 1, so it is not informative here, but the exact FWER under independence is effectively 1.0. To control FWER at 0.05, the Bonferroni-adjusted threshold becomes:

$$
\frac{0.05}{500{,}000} = 1 \times 10^{-7}
$$

This extremely stringent threshold is the reason genomics studies report "genome-wide significance" at $p < 5 \times 10^{-8}$ (which accounts for the roughly 1 million effective independent tests after adjusting for linkage disequilibrium). Such conservatism motivates the [false discovery rate](fdr.md) as a less stringent alternative.

## When FWER Control Is Appropriate

FWER control is most appropriate when:

- The cost of even a single false positive is high (e.g., approving an ineffective drug)
- The number of tests is moderate (roughly $m \leq 100$)
- The researcher needs to make definitive claims about individual hypotheses

When the number of tests is very large and some false positives are tolerable, the [false discovery rate (FDR)](fdr.md) provides a less conservative framework that retains more statistical power. For the specific procedures used to control FWER, see [Bonferroni and Holm Corrections](bonferroni_holm.md).
