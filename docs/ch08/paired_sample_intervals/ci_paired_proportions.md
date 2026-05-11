# Paired Interval for Proportions (McNemar)

## Overview

When the same subjects are measured under two conditions --- such as before and after a treatment --- the resulting binary outcomes are dependent. The standard two-proportion confidence interval assumes independence between groups and is therefore invalid for paired data. McNemar's approach resolves this by recognizing that only the **discordant pairs** (subjects whose outcomes changed between conditions) carry information about the difference in proportions. This section develops the McNemar confidence interval for the difference in paired proportions.

## The Concordance--Discordance Table

Suppose $n$ subjects are each observed under two conditions, producing a pair of binary outcomes $(Y_{i1}, Y_{i2})$ for each subject $i$. We classify the $n$ pairs into four cells:

|  | Condition 2: Success | Condition 2: Failure |
|---|---|---|
| **Condition 1: Success** | $a$ | $b$ |
| **Condition 1: Failure** | $c$ | $d$ |

Here $a + b + c + d = n$. The cells $a$ and $d$ are **concordant pairs** --- subjects whose outcome did not change. The cells $b$ and $c$ are **discordant pairs** --- subjects whose outcome differs between conditions.

The marginal proportions of success under each condition are

$$
\hat{p}_1 = \frac{a + b}{n}, \qquad \hat{p}_2 = \frac{a + c}{n}
$$

so the difference in marginal proportions depends only on the discordant counts:

$$
\hat{p}_1 - \hat{p}_2 = \frac{(a + b) - (a + c)}{n} = \frac{b - c}{n}
$$

This is the key insight: concordant pairs ($a$ and $d$) cancel out. A subject who succeeds under both conditions, or fails under both, tells us nothing about whether one condition is better than the other. All information about the difference resides in the $b + c$ subjects who switched.

## McNemar Confidence Interval

To construct a confidence interval for the population difference $p_1 - p_2$, we need the standard error of $(b - c)/n$. Conditional on $n$, the discordant pair counts $b$ and $c$ satisfy $b + c = m$ (the total number of discordant pairs), and under the null hypothesis $p_1 = p_2$, $b \sim \text{Binomial}(m, 1/2)$.

The Wald-type standard error for the difference in marginal proportions is

$$
\text{SE} = \frac{1}{n}\sqrt{b + c - \frac{(b - c)^2}{n}}
$$

and the $(1 - \alpha)$ confidence interval is

$$
\hat{p}_1 - \hat{p}_2 \pm z_{\alpha/2} \cdot \text{SE} = \frac{b - c}{n} \pm z_{\alpha/2} \cdot \frac{1}{n}\sqrt{b + c - \frac{(b - c)^2}{n}}
$$

where $z_{\alpha/2}$ is the upper $\alpha/2$ quantile of the standard normal distribution.

!!! note "Alternative Standard Error"
    A simpler and commonly used form of the standard error treats $b$ and $c$ as independent binomial counts:

    $$
    \text{SE}_{\text{simple}} = \frac{\sqrt{b + c}}{n}
    $$

    This yields a slightly wider interval than the Wald form above and is often preferred in practice for its simplicity. The resulting CI is:

    $$
    \frac{b - c}{n} \pm z_{\alpha/2} \cdot \frac{\sqrt{b + c}}{n}
    $$

## Worked Example

A pharmaceutical company tests a new allergy medication on $n = 200$ patients. Each patient is evaluated for symptom relief both before and after treatment. The results are:

|  | After: Relief | After: No Relief |
|---|---|---|
| **Before: Relief** | $a = 20$ | $b = 50$ |
| **Before: No Relief** | $c = 30$ | $d = 100$ |

**Step 1.** Compute the difference in proportions:

$$
\hat{p}_1 - \hat{p}_2 = \frac{b - c}{n} = \frac{50 - 30}{200} = 0.10
$$

The proportion with relief was 10 percentage points higher before treatment (i.e., $\hat{p}_{\text{before}} = 70/200 = 0.35$ versus $\hat{p}_{\text{after}} = 50/200 = 0.25$).

**Step 2.** Compute the standard error:

$$
\text{SE} = \frac{1}{200}\sqrt{50 + 30 - \frac{(50 - 30)^2}{200}} = \frac{1}{200}\sqrt{80 - 2} = \frac{\sqrt{78}}{200} \approx 0.0441
$$

**Step 3.** Construct the 95% confidence interval ($z_{0.025} = 1.96$):

$$
0.10 \pm 1.96 \times 0.0441 = 0.10 \pm 0.0865 = (0.0135,\; 0.1865)
$$

Since the interval does not contain zero, we have evidence at the 95% level that the marginal proportions differ between the two conditions.

!!! tip "Interpreting the Result"
    The confidence interval $(0.014, 0.187)$ means we are 95% confident that the true difference in population proportions (before minus after) lies between 1.4% and 18.7%. This suggests a statistically significant but potentially modest reduction in relief rates after treatment.

## Assumptions

The McNemar confidence interval relies on the following conditions:

- **Paired design**: Each observation under Condition 1 is matched with an observation under Condition 2 on the same subject (or a matched pair).
- **Binary outcomes**: Each observation is classified as success or failure.
- **Large sample**: The Wald-type CI requires a sufficient number of discordant pairs. A common guideline is $b + c \geq 10$. For small discordant counts, an exact binomial CI on $b/(b+c)$ is preferred.

!!! warning "Small Discordant Counts"
    When $b + c$ is small (say, fewer than 10), the normal approximation underlying the Wald CI is unreliable. In such cases, use an exact confidence interval based on the binomial distribution for $b$ given $b + c$.

## Exercises

**Exercise 1.**
In a study of 150 patients tested before and after a new drug, the results are: $a = 60$ (positive both times), $b = 25$ (positive then negative), $c = 15$ (negative then positive), $d = 50$ (negative both times). Construct a 95% confidence interval for the difference in marginal proportions $p_1 - p_2$.

??? success "Solution to Exercise 1"
    The point estimate is:

    $$
    \hat{p}_1 - \hat{p}_2 = \frac{b - c}{n} = \frac{25 - 15}{150} = \frac{10}{150} \approx 0.0667
    $$

    The standard error is:

    $$
    \text{SE} = \frac{1}{150}\sqrt{25 + 15 - \frac{(25-15)^2}{150}} = \frac{1}{150}\sqrt{40 - \frac{100}{150}} = \frac{1}{150}\sqrt{39.33} \approx \frac{6.272}{150} \approx 0.0418
    $$

    The 95% CI is:

    $$
    0.0667 \pm 1.96 \times 0.0418 = 0.0667 \pm 0.0819 = (-0.015, 0.149)
    $$

    Since the interval contains zero, we do not have sufficient evidence at the 95% level to conclude that the marginal proportions differ.

---

**Exercise 2.**
In the data from Exercise 1, verify that the large-sample condition ($b + c \geq 10$) is satisfied.

??? success "Solution to Exercise 2"
    The number of discordant pairs is $b + c = 25 + 15 = 40$, which is well above the minimum of 10. The normal approximation for the McNemar confidence interval is therefore appropriate.

---

**Exercise 3.**
Explain why only the discordant pairs ($b$ and $c$) are relevant for the McNemar CI and why the concordant pairs ($a$ and $d$) provide no information about the difference in marginal proportions.

??? success "Solution to Exercise 3"
    The concordant pairs ($a$ and $d$) have the same outcome under both conditions, so they provide no evidence of a change. A subject who is positive both times or negative both times does not help distinguish whether one condition produces a higher proportion of positives than the other.

    Only the discordant pairs — where the outcome changes between conditions — carry information about the difference. If $b > c$, more subjects changed from positive to negative than from negative to positive, suggesting $p_1 > p_2$. The McNemar test and CI formalize this by focusing exclusively on the ratio $b/(b+c)$ and whether it differs from 0.5.

---

**Exercise 4.**
If $b = 3$ and $c = 2$ in a study of $n = 100$ subjects, explain why the Wald-type McNemar CI should not be used. What alternative would you recommend?

??? success "Solution to Exercise 4"
    The number of discordant pairs is $b + c = 3 + 2 = 5$, which is below the recommended minimum of 10 for the normal approximation. With so few discordant pairs, the Wald CI will be unreliable — the normal approximation to the binomial is poor for small counts.

    The recommended alternative is an **exact binomial confidence interval** for the probability $\pi = P(\text{discordant pair is type } b) = b/(b+c)$. Under $H_0$: $p_1 = p_2$, the number of type-$b$ discordant pairs follows a Binomial$(b+c, 0.5)$ distribution. The exact CI for $\pi$ can then be transformed back to a CI for $p_1 - p_2$.
