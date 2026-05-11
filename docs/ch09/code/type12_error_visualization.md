# Type I/II Error Visualization

## Overview

Every hypothesis test involves two kinds of mistakes: a Type I error (rejecting $H_0$ when it is true) and a Type II error (failing to reject $H_0$ when $H_1$ is true). This page visualizes both errors as shaded areas under two overlapping distributions -- the null and the alternative -- and shows how the significance level, effect size, and sample size jointly determine the power of a test.

## Definitions

| | $H_0$ true | $H_1$ true |
|---|---|---|
| **Reject $H_0$** | Type I error ($\alpha$) | Correct (Power $= 1 - \beta$) |
| **Fail to reject $H_0$** | Correct | Type II error ($\beta$) |

- **Type I error rate** ($\alpha$): the probability of rejecting $H_0$ when $H_0$ is actually true. This is the significance level, set by the researcher (commonly 0.05).
- **Type II error rate** ($\beta$): the probability of failing to reject $H_0$ when $H_1$ is actually true.
- **Power** ($1 - \beta$): the probability of correctly rejecting $H_0$ when $H_1$ is true.

## Geometry of the Errors

Consider a one-sided test with null distribution $N(\mu_0, 1)$ and alternative distribution $N(\mu_1, 1)$ where $\mu_1 > \mu_0$. The critical value for a right-tailed test at level $\alpha$ is

$$
z_{\text{crit}} = \mu_0 + z_{1-\alpha}.
$$

Then:

$$
\alpha = P(Z \geq z_{\text{crit}} \mid H_0) = 1 - \mathcal{N}(z_{1-\alpha}) = \alpha,
$$

$$
\beta = P(Z < z_{\text{crit}} \mid H_1) = \mathcal{N}\!\left(z_{\text{crit}} - \mu_1\right),
$$

$$
\text{Power} = 1 - \beta = 1 - \mathcal{N}\!\left(z_{1-\alpha} - (\mu_1 - \mu_0)\right).
$$

## Code

### Plotting the Two Distributions

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

null_loc = 0
alt_loc = 3
alpha = 0.05

x = np.linspace(null_loc - 4, alt_loc + 4, 400)
y_null = stats.norm.pdf(x, loc=null_loc)
y_alt = stats.norm.pdf(x, loc=alt_loc)

z_crit = stats.norm.ppf(1 - alpha, loc=null_loc)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x, y_null, "b-", lw=2, label="Null distribution")
ax.plot(x, y_alt, "r-", lw=2, label="Alternative distribution")

# Type I error: null distribution beyond critical value
mask_t1 = x >= z_crit
ax.fill_between(x[mask_t1], y_null[mask_t1], alpha=0.4, color="blue",
                label="Type I error (alpha)")

# Type II error: alternative distribution below critical value
mask_t2 = x <= z_crit
ax.fill_between(x[mask_t2], y_alt[mask_t2], alpha=0.3, color="red",
                label="Type II error (beta)")

ax.axvline(z_crit, color="black", linestyle="--", alpha=0.6,
           label=f"Critical value = {z_crit:.2f}")
ax.set_xlabel("Test statistic")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
plt.show()
```

### Computing Power for Different Separations

```python
for sep in [1, 2, 3, 4, 5]:
    z_c = stats.norm.ppf(0.95)
    power = 1 - stats.norm.cdf(z_c, loc=sep)
    beta = 1 - power
    print(f"Separation = {sep}: beta = {beta:.4f}, Power = {power:.4f}")
```

## Interpretation

With $\mu_0 = 0$, $\mu_1 = 3$, and $\alpha = 0.05$:

- The **critical value** is $z_{\text{crit}} \approx 1.645$.
- The **Type I error** (blue shaded area under the null curve to the right of 1.645) equals exactly $\alpha = 0.05$.
- The **Type II error** (red shaded area under the alternative curve to the left of 1.645) is $\beta = \mathcal{N}(1.645 - 3) = \mathcal{N}(-1.355) \approx 0.088$.
- **Power** is $1 - 0.088 = 0.912$, meaning there is a 91.2% chance of detecting an effect of size 3.

As the separation $\mu_1 - \mu_0$ increases (larger effect size), the alternative distribution shifts right, reducing the overlap with the null and decreasing $\beta$. Conversely, smaller effects produce greater overlap and lower power.

## Factors Affecting Power

Power increases when:

1. **Effect size** ($\mu_1 - \mu_0$) increases -- the distributions separate further.
2. **Sample size** ($n$) increases -- the standard error $\sigma/\sqrt{n}$ shrinks, making both distributions narrower.
3. **Significance level** ($\alpha$) increases -- the critical value moves left, enlarging the rejection region (but at the cost of more Type I errors).
4. **Variance** ($\sigma^2$) decreases -- narrower distributions mean less overlap.

## Exercises

**Exercise 1.** For a one-sided test with $\mu_0 = 0$, $\mu_1 = 2$, $\sigma = 1$, and $\alpha = 0.05$, compute $\beta$ and the power analytically.

??? success "Solution to Exercise 1"

    The critical value is $z_{\text{crit}} = z_{0.95} = 1.645$. Under $H_1$, the test statistic has distribution $N(2, 1)$. Therefore

    $$
    \beta = P(Z < 1.645 \mid Z \sim N(2,1)) = \mathcal{N}(1.645 - 2) = \mathcal{N}(-0.355) \approx 0.3613.
    $$

    Power is

    $$
    1 - \beta = 1 - 0.3613 = 0.6387.
    $$

    There is approximately a 64% chance of detecting an effect of size 2. $\square$

---

**Exercise 2.** Show that increasing the sample size from $n$ to $4n$ doubles the "effective separation" between the null and alternative distributions (in standard-error units). What does this imply about the required sample size to achieve a target power?

??? success "Solution to Exercise 2"

    With sample size $n$, the test is based on $\bar{X} \sim N(\mu, \sigma^2/n)$. The separation in standard-error units is

    $$
    \delta = \frac{\mu_1 - \mu_0}{\sigma / \sqrt{n}} = \frac{(\mu_1 - \mu_0)\sqrt{n}}{\sigma}.
    $$

    Replacing $n$ by $4n$:

    $$
    \delta' = \frac{(\mu_1 - \mu_0)\sqrt{4n}}{\sigma} = 2\delta.
    $$

    So quadrupling the sample size doubles the effective separation. More generally, to achieve a target power with a fixed effect size, the required sample size is

    $$
    n = \left(\frac{(z_{1-\alpha} + z_{1-\beta})\sigma}{\mu_1 - \mu_0}\right)^2.
    $$

    This formula shows that $n$ is inversely proportional to the squared effect size. $\square$

---

**Exercise 3.** Explain the trade-off between $\alpha$ and $\beta$. If we lower $\alpha$ from 0.05 to 0.01 while keeping everything else fixed, what happens to $\beta$ and power?

??? success "Solution to Exercise 3"

    Lowering $\alpha$ moves the critical value to the right (for a right-tailed test): $z_{0.99} = 2.326 > z_{0.95} = 1.645$. This makes it harder to reject $H_0$, so:

    - $\alpha$ decreases (fewer Type I errors).
    - $\beta$ increases (more Type II errors), because more of the alternative distribution now falls below the stricter critical value.
    - Power $= 1 - \beta$ decreases.

    For the $\mu_1 = 3$ example:

    - At $\alpha = 0.05$: $\beta = \mathcal{N}(1.645 - 3) = \mathcal{N}(-1.355) \approx 0.088$, Power $\approx 0.912$.
    - At $\alpha = 0.01$: $\beta = \mathcal{N}(2.326 - 3) = \mathcal{N}(-0.674) \approx 0.250$, Power $\approx 0.750$.

    The only way to reduce both error types simultaneously is to increase the sample size. $\square$

---

**Exercise 4.** In the visualization, the two distributions have equal variance. What changes if the alternative distribution has a larger variance? Sketch or describe the new picture.

??? success "Solution to Exercise 4"

    If the alternative distribution is $N(\mu_1, \sigma_1^2)$ with $\sigma_1 > 1$, it is wider and flatter than the null $N(\mu_0, 1)$. The key consequences are:

    - The alternative curve spreads out, so more of its area falls below the critical value. This increases $\beta$ and decreases power.
    - The overlap between the two distributions increases even if the means are well separated.
    - The power formula becomes

    $$
    \text{Power} = 1 - \mathcal{N}\!\left(\frac{z_{\text{crit}} - \mu_1}{\sigma_1}\right),
    $$

    which is smaller than the equal-variance case when $\sigma_1 > 1$.

    Visually, the red curve is shorter and broader, with a larger red-shaded Type II error region to the left of the critical value. $\square$

---

**Exercise 5.** A clinical trial requires 80% power to detect an effect of $\delta = 0.5$ (Cohen's $d$) at $\alpha = 0.05$ (two-sided). Derive the minimum sample size per group.

??? success "Solution to Exercise 5"

    For a two-sided test, the power equation is

    $$
    1 - \beta = \mathcal{N}\!\left(\delta\sqrt{\frac{n}{2}} - z_{1-\alpha/2}\right).
    $$

    Setting $1 - \beta = 0.80$ gives $z_{0.80} = 0.8416$, and $z_{0.975} = 1.960$. Solving:

    $$
    0.8416 = \delta\sqrt{\frac{n}{2}} - 1.960,
    $$

    $$
    \delta\sqrt{\frac{n}{2}} = 2.8016,
    $$

    $$
    \sqrt{\frac{n}{2}} = \frac{2.8016}{0.5} = 5.6032,
    $$

    $$
    \frac{n}{2} = 31.40, \qquad n = 62.8.
    $$

    We need at least $n = 63$ subjects per group (126 total) to achieve 80% power for detecting a medium effect of $d = 0.5$ at the two-sided 5% level. $\square$
