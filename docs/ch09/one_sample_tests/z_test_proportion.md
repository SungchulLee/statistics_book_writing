# Z-Test for a Proportion

## Overview

Many practical questions involve population proportions: Has a manufacturing defect rate changed? Does voter support exceed 50%? Is the response rate for a new drug different from the known rate? The one-sample Z-test for proportions provides a formal framework for answering such questions. It tests whether a population proportion $p$ equals a specific hypothesized value $p_0$, using the normal approximation to the binomial distribution.

## Hypotheses

$$
H_0: p = p_0 \quad \text{vs} \quad H_1: p \neq p_0
$$

For one-sided tests, the alternative is either $H_1: p > p_0$ or $H_1: p < p_0$, depending on the research question.

## Test Statistic

The test statistic measures how many standard errors the sample proportion $\hat{p}$ lies from the hypothesized value $p_0$. Under $H_0$, the standard error of $\hat{p}$ is $\sqrt{p_0(1-p_0)/n}$, so we standardize:

$$
Z = \frac{\hat{p} - p_0}{\sqrt{p_0(1-p_0)/n}}
$$

where $\hat{p} = X/n$ is the sample proportion and $X$ is the number of successes in $n$ independent trials. Under $H_0$, $Z$ is approximately standard normal when the sample size conditions below are met.

## Decision Rule

- **Two-sided** ($H_1: p \neq p_0$): Reject $H_0$ if $|Z| > z_{\alpha/2}$
- **Right-sided** ($H_1: p > p_0$): Reject $H_0$ if $Z > z_{\alpha}$
- **Left-sided** ($H_1: p < p_0$): Reject $H_0$ if $Z < -z_{\alpha}$

Equivalently, compute the p-value and reject $H_0$ if $p\text{-value} \leq \alpha$.

## Conditions

The normal approximation to the binomial is adequate when both

$$
np_0 \geq 10 \quad \text{and} \quad n(1-p_0) \geq 10
$$

as a rule of thumb. These conditions ensure that the sampling distribution of $\hat{p}$ is approximately normal under $H_0$, so the Z-statistic has an approximate $N(0,1)$ distribution. Some references use a threshold of 5 instead of 10.

??? example "Worked Example: Testing a Defect Rate"

    A factory historically produces items with a 4% defect rate ($p_0 = 0.04$). After a process change, a random sample of $n = 500$ items contains 28 defectives, giving $\hat{p} = 28/500 = 0.056$. Test whether the defect rate has changed at the $\alpha = 0.05$ level.

    **Check conditions.** $np_0 = 500 \times 0.04 = 20 \geq 10$ and $n(1 - p_0) = 500 \times 0.96 = 480 \geq 10$. Both satisfied.

    **Compute the test statistic.**

    $$
    Z = \frac{0.056 - 0.04}{\sqrt{0.04 \times 0.96 / 500}} = \frac{0.016}{\sqrt{0.0000768}} = \frac{0.016}{0.00877} \approx 1.83
    $$

    **Decision.** For a two-sided test, the critical value is $z_{0.025} = 1.96$. Since $|1.83| < 1.96$, we fail to reject $H_0$. The two-sided p-value is $2P(Z > 1.83) \approx 0.067 > 0.05$, confirming the decision. There is not sufficient evidence at the 5% level to conclude that the defect rate has changed.

## Exercises

**Exercise 1.**
Pennies: 100 spins, 59 heads. Test $H_0: p = 0.5$ vs $H_1: p > 0.5$ at $\alpha = 0.05$.

??? success "Solution to Exercise 1"
    $\hat p = 0.59$. $z = (0.59 - 0.5)/\sqrt{0.25/100} = 0.09/0.05 = 1.8$.

    P-value (one-sided): $P(Z > 1.8) \approx 0.036$. Since $0.036 < 0.05$, **reject $H_0$**.

    Evidence that pennies show heads more than 50% — consistent with biased coin.

---

**Exercise 2.**
Vehicle inspection: 74/100 pass. Test claim of 80% at $\alpha = 0.05$.

??? success "Solution to Exercise 2"
    $H_0: p = 0.80$ vs $H_1: p \ne 0.80$.

    $z = (0.74 - 0.80)/\sqrt{0.80 \cdot 0.20/100} = -0.06/0.04 = -1.5$. $|z| < 1.96$. **Fail to reject.**

    Not enough evidence to contradict 80%.

---

**Exercise 3.**
**Defect rate change.** 28/500 defective, historical 4%. Test for change.

??? success "Solution to Exercise 3"
    $\hat p = 28/500 = 0.056$. $z = (0.056 - 0.04)/\sqrt{0.04 \cdot 0.96/500} = 0.016/0.00877 \approx 1.83$.

    Two-sided p-value $= 2 \cdot P(Z > 1.83) \approx 0.067$. Fail to reject at $\alpha = 0.05$.

    Conditions check: $np_0 = 20 \ge 10$, $n(1-p_0) = 480 \ge 10$. Valid.

---

**Exercise 4.**
**One-sided vs two-sided.** Recompute Exercise 3 as $H_1: p > 0.04$ (suspect defect rate increased).

??? success "Solution to Exercise 4"
    $z = 1.83$, one-sided p-value $= P(Z > 1.83) \approx 0.034$. **Reject at $\alpha = 0.05$.**

    Different decision: two-sided says no, one-sided says yes. If you had prior reason to suspect *increase only*, one-sided is appropriate.

    Caution: choosing one-sided after seeing the data inflates Type I error. Pre-specify direction.

---

**Exercise 5.**
**Conditions for $z$-test on proportion.** State and verify for Exercise 1.

??? success "Solution to Exercise 5"
    Conditions: $np_0 \ge 10$ and $n(1 - p_0) \ge 10$ (some use 5).

    Exercise 1: $np_0 = 100 \cdot 0.5 = 50 \ge 10$, $n(1-p_0) = 50 \ge 10$. Valid.

    When conditions fail (small $n$ or extreme $p_0$): use exact binomial test instead. Available in scipy: `binomtest`.

---

**Exercise 6.**
**Sample size planning.** What $n$ detects a 2pp difference ($p = 0.42$ vs claimed 0.40) with 80% power at $\alpha = 0.05$ (two-sided)?

??? success "Solution to Exercise 6"
    Formula: $n = (z_{\alpha/2}\sqrt{p_0(1-p_0)} + z_\beta\sqrt{p_1(1-p_1)})^2/(p_1 - p_0)^2$.

    $= (1.96 \sqrt{0.24} + 0.84 \sqrt{0.2436})^2/(0.02)^2$

    $= (1.96 \cdot 0.490 + 0.84 \cdot 0.494)^2/0.0004$

    $= (0.960 + 0.415)^2/0.0004 = 1.89/0.0004 \approx 4730$.

    Need ~4700 observations to detect a 2pp difference with 80% power. Small differences require large samples.
