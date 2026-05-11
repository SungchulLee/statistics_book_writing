# Bonferroni and Holm Corrections

## Overview

When a study tests multiple hypotheses simultaneously, the chance of at least one false rejection grows rapidly. For example, testing $m = 20$ independent hypotheses at level $\alpha = 0.05$ yields a probability of roughly $1 - (1 - 0.05)^{20} \approx 0.64$ that at least one true null hypothesis is incorrectly rejected. Multiple testing corrections address this inflation by adjusting either the significance thresholds or the p-values so that a global error rate remains controlled. This section introduces two widely used corrections — Bonferroni and Holm — that control the **family-wise error rate (FWER)**.

!!! info "Family-Wise Error Rate (FWER)"
    Suppose we test $m$ null hypotheses $H_1, H_2, \ldots, H_m$ simultaneously. Let $V$ denote the number of true null hypotheses that are incorrectly rejected (false positives). The FWER is defined as

    $$
    \text{FWER} = P(V \geq 1)
    $$

    A multiple testing procedure **controls the FWER at level** $\alpha$ if $\text{FWER} \leq \alpha$ regardless of which hypotheses are true and which are false.

## Bonferroni Correction

The simplest approach to controlling the FWER is due to Bonferroni. The key idea is to divide the overall significance level $\alpha$ equally among all $m$ tests, so that each individual test uses a more stringent threshold. This ensures that even in the worst case, the total probability of any false rejection stays below $\alpha$.

### Rejection Rule

Let $p_1, p_2, \ldots, p_m$ denote the p-values from $m$ hypothesis tests, and let $\alpha$ be the desired FWER. The Bonferroni correction rejects $H_i$ whenever

$$
p_i \leq \frac{\alpha}{m}
$$

### Why It Controls FWER

The justification follows immediately from Boole's inequality (the union bound). Let $\mathcal{M}_0 \subseteq \{1, \ldots, m\}$ denote the set of indices for which $H_i$ is true. Then

$$
\text{FWER} = P\!\Bigl(\bigcup_{i \in \mathcal{M}_0} \{p_i \leq \alpha/m\}\Bigr) \leq \sum_{i \in \mathcal{M}_0} P(p_i \leq \alpha/m) \leq |\mathcal{M}_0| \cdot \frac{\alpha}{m} \leq \alpha
$$

The last inequality uses $|\mathcal{M}_0| \leq m$. This bound holds without any assumption about the dependence structure among the tests, which is both the strength and the limitation of the Bonferroni correction.

!!! warning "Conservativeness of Bonferroni"
    The Bonferroni correction is **conservative**: the actual FWER is often well below $\alpha$, especially when $m$ is large or the test statistics are positively correlated. This conservativeness reduces statistical power — truly false null hypotheses may fail to be rejected because the per-test threshold $\alpha/m$ is too stringent.

## Holm's Step-Down Procedure

While Bonferroni applies the same stringent threshold $\alpha/m$ to every test, Holm (1979) observed that once the most significant hypotheses have been rejected, the remaining tests can use less stringent thresholds. This step-down approach is **uniformly more powerful** than Bonferroni — it rejects at least as many hypotheses in every possible configuration — while still controlling the FWER at level $\alpha$.

### Algorithm

1. **Order the p-values** from smallest to largest: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$, with corresponding hypotheses $H_{(1)}, H_{(2)}, \ldots, H_{(m)}$.
2. **Start with** $i = 1$. Compare $p_{(i)}$ to the adjusted threshold $\alpha / (m - i + 1)$.
3. **If** $p_{(i)} \leq \alpha / (m - i + 1)$, reject $H_{(i)}$ and increment $i$ by one.
4. **If** $p_{(i)} > \alpha / (m - i + 1)$, **stop**. Retain $H_{(i)}, H_{(i+1)}, \ldots, H_{(m)}$ (all remaining hypotheses).

The adjusted thresholds form an increasing sequence:

$$
\frac{\alpha}{m}, \quad \frac{\alpha}{m-1}, \quad \frac{\alpha}{m-2}, \quad \ldots, \quad \alpha
$$

Because later thresholds are less stringent than $\alpha/m$, Holm rejects at least as many hypotheses as Bonferroni. At the same time, the step-down structure ensures that the FWER remains controlled at level $\alpha$.

### Why Holm Dominates Bonferroni

Under Bonferroni, every hypothesis faces the threshold $\alpha/m$. Under Holm, only the most significant p-value faces this threshold; the second most significant faces $\alpha/(m-1)$, and so on. Since $\alpha/(m-i+1) \geq \alpha/m$ for all $i \geq 1$, any hypothesis rejected by Bonferroni is also rejected by Holm, but the converse is not necessarily true. This makes Holm uniformly more powerful.

!!! example "Numerical Example"
    Suppose $m = 4$ hypotheses are tested at FWER $\alpha = 0.05$, yielding p-values:

    | Hypothesis | p-value |
    |:----------:|:-------:|
    | $H_A$ | 0.003 |
    | $H_B$ | 0.012 |
    | $H_C$ | 0.042 |
    | $H_D$ | 0.130 |

    **Bonferroni:** The adjusted threshold is $0.05/4 = 0.0125$. Only $H_A$ ($p = 0.003$) is rejected. $H_B$ ($p = 0.012 < 0.0125$? No, $0.012 < 0.0125$ is false) is not rejected.

    **Holm:** Order the p-values and compare sequentially:

    | Step $i$ | $H_{(i)}$ | $p_{(i)}$ | Threshold $\alpha/(m-i+1)$ | Decision |
    |:--------:|:---------:|:---------:|:--------------------------:|:--------:|
    | 1 | $H_A$ | 0.003 | $0.05/4 = 0.0125$ | Reject |
    | 2 | $H_B$ | 0.012 | $0.05/3 = 0.0167$ | Reject |
    | 3 | $H_C$ | 0.042 | $0.05/2 = 0.025$ | Stop (retain) |
    | 4 | $H_D$ | 0.130 | $0.05/1 = 0.05$ | Retain |

    Holm rejects both $H_A$ and $H_B$, while Bonferroni rejects only $H_A$. This illustrates how Holm recovers power without sacrificing FWER control.

## Comparison

| Property | Bonferroni | Holm |
|:---------|:-----------|:-----|
| FWER control | Yes (any dependence) | Yes (any dependence) |
| Power | Conservative | Uniformly more powerful |
| Simplicity | Very simple | Slightly more involved |
| Assumption on dependence | None required | None required |
| Best use case | Quick adjustment, few tests | Default choice over Bonferroni |

In practice, there is rarely a reason to prefer Bonferroni over Holm, since Holm provides the same FWER guarantee with strictly greater (or equal) power.

## Exercises

**Exercise 1.**
A researcher performs 5 hypothesis tests with p-values 0.008, 0.025, 0.040, 0.060, 0.120. Apply the Bonferroni correction at $\alpha = 0.05$ and state which hypotheses are rejected.

??? success "Solution to Exercise 1"
    The Bonferroni adjusted threshold is $\alpha/m = 0.05/5 = 0.01$.

    Compare each p-value to 0.01:

    | Test | p-value | $< 0.01$? | Decision |
    |---|---|---|---|
    | 1 | 0.008 | Yes | Reject |
    | 2 | 0.025 | No | Fail to reject |
    | 3 | 0.040 | No | Fail to reject |
    | 4 | 0.060 | No | Fail to reject |
    | 5 | 0.120 | No | Fail to reject |

    Only test 1 is rejected. Bonferroni is conservative: test 2 with $p = 0.025$ is not rejected despite being significant at the unadjusted level.

---

**Exercise 2.**
Apply the Holm (step-down) procedure to the same p-values from Exercise 1 at $\alpha = 0.05$. Compare the results with Bonferroni.

??? success "Solution to Exercise 2"
    Sort the p-values: $p_{(1)} = 0.008, p_{(2)} = 0.025, p_{(3)} = 0.040, p_{(4)} = 0.060, p_{(5)} = 0.120$.

    Holm thresholds: $\alpha/(m - j + 1)$ for the $j$-th ordered test:

    | Step $j$ | $p_{(j)}$ | Threshold $\alpha/(m-j+1)$ | $p_{(j)} < $ threshold? | Decision |
    |---|---|---|---|---|
    | 1 | 0.008 | 0.05/5 = 0.010 | Yes | Reject |
    | 2 | 0.025 | 0.05/4 = 0.0125 | No | Stop |

    Once we fail to reject at step 2, we stop and do not reject any remaining tests.

    Result: only test 1 is rejected -- the same as Bonferroni in this case. However, Holm is uniformly more powerful than Bonferroni (it can never reject fewer hypotheses), and with different p-value configurations, Holm would reject more.

---

**Exercise 3.**
Prove that the Bonferroni correction controls the family-wise error rate (FWER) at level $\alpha$. Use the union bound.

??? success "Solution to Exercise 3"
    Let $\mathcal{H}_0$ denote the set of true null hypotheses, with $|\mathcal{H}_0| = m_0 \leq m$. The FWER is:

    $$
    \text{FWER} = P\!\left(\bigcup_{i \in \mathcal{H}_0} \{p_i < \alpha/m\}\right) \leq \sum_{i \in \mathcal{H}_0} P(p_i < \alpha/m) = m_0 \cdot \frac{\alpha}{m} \leq m \cdot \frac{\alpha}{m} = \alpha
    $$

    The first inequality is Boole's inequality (union bound). The second equality uses the fact that under the true null, p-values are uniformly distributed on $(0,1)$, so $P(p_i < \alpha/m) = \alpha/m$. The final inequality uses $m_0 \leq m$.

    This proof makes no assumptions about the dependence structure of the tests, which is why Bonferroni is valid under arbitrary dependence but potentially conservative. $\square$

---

**Exercise 4.**
Explain why Bonferroni is conservative (has FWER $\ll \alpha$) when the number of true nulls $m_0$ is much less than $m$, or when the tests are positively correlated.

??? success "Solution to Exercise 4"
    **When $m_0 \ll m$:** The proof uses $m_0 \leq m$, so the actual FWER is at most $m_0 \alpha/m$. If only 10 out of 1000 nulls are true ($m_0 = 10$), the true FWER is at most $10 \times 0.05/1000 = 0.0005$, far below the nominal $\alpha = 0.05$. The Bonferroni threshold is calibrated for the worst case ($m_0 = m$).

    **Under positive correlation:** The union bound $P(\cup A_i) \leq \sum P(A_i)$ is tight only when events are mutually exclusive. When tests are positively correlated (which is common when test statistics share data), the events $\{p_i < \alpha/m\}$ tend to occur together, making the union much smaller than the sum. The actual FWER can be much less than $\alpha$.

    Both sources of conservatism reduce power: true effects are harder to detect. The Holm procedure partially addresses the first issue; resampling-based methods (e.g., permutation FWER) address the second.
