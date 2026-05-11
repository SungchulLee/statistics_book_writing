# Sample Size for Comparing Two Groups

## Overview

Before collecting data for a two-group comparison, a researcher must determine how many subjects to enroll in each group. Too few subjects yield a test with low power --- the study may fail to detect a real treatment effect, wasting time and resources. Too many subjects are costly and may expose unnecessary participants to an inferior treatment. Sample size formulas translate the desired power, significance level, and minimum meaningful effect size into a concrete number of observations per group. These formulas connect directly to the Type I error rate $\alpha$, the Type II error rate $\beta$, and the power $1 - \beta$ introduced in previous chapters.

## Two-Sample Mean Comparison

### Intuition

Three factors determine how many observations each group needs. First, **larger effects are easier to detect**: if the true difference between group means is large relative to the noise, fewer observations suffice. Second, **more variable data requires more observations**: when $\sigma$ is large, the signal is harder to distinguish from noise. Third, **stricter error control costs more observations**: demanding a smaller $\alpha$ (fewer false positives) or higher power $1 - \beta$ (fewer missed effects) both increase the required sample size.

### Formula

Consider a two-sided test of $H_0\colon \mu_1 = \mu_2$ against $H_1\colon \mu_1 \neq \mu_2$, assuming:

- Equal population variances: $\sigma_1^2 = \sigma_2^2 = \sigma^2$
- Equal group sizes: $n_1 = n_2 = n$
- Normal populations (or large enough samples for the CLT)

The required sample size per group to detect a difference $\delta = |\mu_1 - \mu_2|$ with power $1 - \beta$ at significance level $\alpha$ is

$$
n = \frac{2(z_{\alpha/2} + z_\beta)^2 \sigma^2}{\delta^2}
$$

where $z_{\alpha/2} = \mathcal{N}^{-1}(1 - \alpha/2)$ is the upper $\alpha/2$ quantile of the standard normal, and $z_\beta = \mathcal{N}^{-1}(1 - \beta)$ is the upper $\beta$ quantile. The numerator reflects the combined stringency of the Type I and Type II error requirements, while the denominator is the squared effect size --- the signal we want to detect.

### Worked Example

A clinical trial compares a new drug to a placebo for reducing blood pressure. Prior studies suggest a common standard deviation of $\sigma = 12$ mmHg. The investigators want to detect a difference of $\delta = 5$ mmHg with 80% power ($\beta = 0.20$) at the 5% significance level ($\alpha = 0.05$).

The critical values are $z_{0.025} = 1.96$ and $z_{0.20} = 0.842$. Substituting:

$$
n = \frac{2(1.96 + 0.842)^2 (12)^2}{(5)^2} = \frac{2(2.802)^2 (144)}{25} = \frac{2(7.851)(144)}{25} = \frac{2261.1}{25} \approx 90.4
$$

Rounding up, the trial needs **91 subjects per group** (182 total).

!!! tip "Always Round Up"
    Since the formula gives a real number, always round up to the next integer. Rounding down would yield power slightly below the target.

## Two-Sample Proportion Comparison

### Intuition

Comparing two proportions follows the same logic, but the variance of a proportion $p(1-p)$ depends on $p$ itself. This introduces a subtlety: the variance under the null hypothesis (when $p_1 = p_2$) differs from the variance under the alternative (when $p_1 \neq p_2$). The sample size formula must account for both.

### Formula

For a two-sided test of $H_0\colon p_1 = p_2$ against $H_1\colon p_1 \neq p_2$ with equal group sizes $n_1 = n_2 = n$:

$$
n = \frac{\bigl(z_{\alpha/2}\sqrt{2\bar{p}(1-\bar{p})} + z_\beta\sqrt{p_1(1-p_1) + p_2(1-p_2)}\bigr)^2}{(p_1 - p_2)^2}
$$

where $\bar{p} = (p_1 + p_2)/2$ is the average of the two hypothesized proportions. The first term under the square root uses the pooled variance under $H_0$ (where both proportions equal $\bar{p}$), while the second term uses the separate variances under $H_1$.

### Worked Example

A marketing team wants to compare conversion rates between a new website design ($p_1 = 0.12$) and the current design ($p_2 = 0.08$). They want 90% power ($\beta = 0.10$) at the 5% significance level.

First, compute the pooled proportion: $\bar{p} = (0.12 + 0.08)/2 = 0.10$.

The critical values are $z_{0.025} = 1.96$ and $z_{0.10} = 1.282$. The numerator is

$$
\bigl(1.96\sqrt{2(0.10)(0.90)} + 1.282\sqrt{(0.12)(0.88) + (0.08)(0.92)}\bigr)^2
$$

$$
= \bigl(1.96\sqrt{0.18} + 1.282\sqrt{0.1056 + 0.0736}\bigr)^2
$$

$$
= \bigl(1.96 \times 0.4243 + 1.282 \times 0.4234\bigr)^2
$$

$$
= (0.8316 + 0.5428)^2 = (1.3744)^2 = 1.889
$$

The denominator is $(0.12 - 0.08)^2 = 0.0016$. Therefore:

$$
n = \frac{1.889}{0.0016} \approx 1180.6
$$

The study needs **1181 subjects per group** (2362 total).

## Practical Considerations

!!! warning "Adjust for Dropout"
    The formulas give the number of subjects needed for analysis. If a dropout rate $d$ is expected, enroll $n / (1 - d)$ subjects per group. For example, with 15% expected dropout and $n = 91$, enroll $\lceil 91 / 0.85 \rceil = 108$ per group.

**Unequal group sizes.** When $n_1 \neq n_2$, the mean comparison formula generalizes to

$$
n_1 = \frac{(1 + 1/k)(z_{\alpha/2} + z_\beta)^2 \sigma^2}{\delta^2}
$$

where $k = n_2/n_1$ is the allocation ratio and $n_2 = k \cdot n_1$. Setting $k = 1$ recovers the equal-size formula (since $1 + 1/1 = 2$).

**Effect size formulation.** Dividing both sides of the mean comparison formula by $\sigma^2$ shows that the required sample size depends on the **standardized effect size** $d = \delta/\sigma$:

$$
n = \frac{2(z_{\alpha/2} + z_\beta)^2}{d^2}
$$

Common benchmarks are $d = 0.2$ (small), $d = 0.5$ (medium), and $d = 0.8$ (large).

## Exercises

**Exercise 1.**
A researcher wants to detect a difference of $\delta = 5$ units between two groups with $\sigma = 10$, using $\alpha = 0.05$ (two-sided) and power $= 0.80$. Compute the required sample size per group.

??? success "Solution to Exercise 1"
    The critical values are $z_{0.025} = 1.96$ and $z_{0.20} = 0.842$ (since power $= 1 - \beta = 0.80$ gives $\beta = 0.20$).

    $$
    n = \frac{2(z_{\alpha/2} + z_\beta)^2 \sigma^2}{\delta^2} = \frac{2(1.96 + 0.842)^2 \times 100}{25} = \frac{2 \times 7.8505 \times 100}{25} = \frac{1570.1}{25} = 62.8
    $$

    Rounding up, $n = 63$ per group (126 total).

---

**Exercise 2.**
Using the effect size formulation, compute the required sample size per group to detect a "medium" effect ($d = 0.5$) with 80% power at $\alpha = 0.05$.

??? success "Solution to Exercise 2"
    $$
    n = \frac{2(z_{\alpha/2} + z_\beta)^2}{d^2} = \frac{2(1.96 + 0.842)^2}{0.25} = \frac{2 \times 7.8505}{0.25} = \frac{15.701}{0.25} = 62.8
    $$

    Rounding up, $n = 63$ per group. This is a widely cited benchmark: detecting a medium effect size at 80% power requires about 63 subjects per group.

---

**Exercise 3.**
If the sample size is fixed at $n = 30$ per group and $\sigma = 10$ with $\alpha = 0.05$, what is the minimum detectable difference $\delta$ at 80% power?

??? success "Solution to Exercise 3"
    Rearranging the formula:

    $$
    \delta = \sqrt{\frac{2(z_{\alpha/2} + z_\beta)^2 \sigma^2}{n}} = \sqrt{\frac{2(1.96 + 0.842)^2 \times 100}{30}} = \sqrt{\frac{1570.1}{30}} = \sqrt{52.34} \approx 7.23
    $$

    With 30 subjects per group, the study can detect a difference of about 7.23 units (or a standardized effect of $d = 7.23/10 = 0.72$) with 80% power.

---

**Exercise 4.**
Explain intuitively why doubling the sample size does not double the detectable effect size. What is the precise relationship between $n$ and $\delta$?

??? success "Solution to Exercise 4"
    From the formula $n = 2(z_{\alpha/2} + z_\beta)^2 \sigma^2 / \delta^2$, solving for $\delta$ gives $\delta \propto 1/\sqrt{n}$. Doubling $n$ reduces the minimum detectable difference by a factor of $\sqrt{2} \approx 1.414$, not by a factor of 2.

    Intuitively, the standard error of the difference in means is $\sigma\sqrt{2/n}$, which decreases as $1/\sqrt{n}$. Since the power to detect $\delta$ depends on the ratio $\delta / \text{SE}$, improving precision by a factor of $\sqrt{2}$ (from doubling $n$) allows detecting a difference that is $\sqrt{2}$ times smaller. This diminishing-returns relationship means that halving the detectable effect size requires **quadrupling** the sample size.
