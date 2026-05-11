# When to Use Paired vs Independent Designs

## Overview

The choice between a paired design and an independent design directly affects the width of confidence intervals and the power of hypothesis tests. A paired design can produce substantially narrower intervals when subjects exhibit natural variation that is large relative to the treatment effect. Conversely, misapplying an independent-sample method to paired data wastes information about within-subject correlation, while misapplying a paired method to independent data violates the dependence assumption. Understanding when each design is appropriate --- and why --- is essential for efficient statistical practice.

## Paired Designs

In a paired design, each observational unit provides measurements under both conditions, or two units are explicitly matched on key characteristics before being assigned to different conditions. The classic example is a before/after study: a physician records each patient's blood pressure before medication and again six weeks later. Because both measurements come from the same individual, much of the person-to-person variability cancels when we compute the within-subject difference $D_i = X_{i,\text{after}} - X_{i,\text{before}}$.

Common paired designs include:

- **Repeated measures**: The same subjects are measured twice (e.g., before and after treatment).
- **Matched pairs**: Subjects are matched on confounding variables (age, sex, baseline severity) and one member of each pair is assigned to each group.
- **Crossover trials**: Each subject receives both treatments in sequence, with a washout period between them.

The central advantage is variance reduction. By focusing on within-subject differences, we eliminate inter-subject variability, which is often the dominant source of noise.

## Independent Designs

In an independent design, different subjects appear in each group with no explicit pairing. A clinical trial that randomly assigns 50 patients to a drug group and 50 different patients to a placebo group is a typical example.

Independent designs have practical advantages:

- **Simpler logistics**: There is no need to match subjects or administer multiple treatments to the same individual.
- **No carryover effects**: Each subject experiences only one condition, so there is no concern about the first treatment affecting the second.
- **Flexibility**: Sample sizes need not be equal across groups.

The tradeoff is that all subject-to-subject variability enters the standard error, potentially producing wider confidence intervals when individual differences are large.

## The Variance Comparison

The mathematical basis for choosing between designs rests on a variance comparison. Suppose each subject $i$ in a paired design yields a difference $D_i = X_{i1} - X_{i2}$, and we estimate $\mu_D = \mu_1 - \mu_2$ with $\bar{D}$.

**Paired design variance:**

$$
\text{Var}(\bar{D}) = \frac{\sigma_1^2 + \sigma_2^2 - 2\rho\,\sigma_1 \sigma_2}{n}
$$

where $\rho$ is the within-pair correlation between $X_{i1}$ and $X_{i2}$, and $n$ is the number of pairs.

**Independent design variance** (with equal group sizes $n$):

$$
\text{Var}(\bar{X}_1 - \bar{X}_2) = \frac{\sigma_1^2 + \sigma_2^2}{n}
$$

The paired design has smaller variance whenever

$$
\frac{\sigma_1^2 + \sigma_2^2 - 2\rho\,\sigma_1\sigma_2}{n} < \frac{\sigma_1^2 + \sigma_2^2}{n}
$$

which simplifies to $\rho > 0$. In words, **pairing helps whenever the within-pair correlation is positive**, which is the typical case in before/after studies, matched experiments, and crossover trials.

When $\sigma_1 = \sigma_2 = \sigma$, the paired variance becomes $2\sigma^2(1 - \rho)/n$, and the independent variance is $2\sigma^2/n$. The variance ratio is

$$
\frac{\text{Var}_{\text{paired}}}{\text{Var}_{\text{independent}}} = 1 - \rho
$$

So a within-pair correlation of $\rho = 0.5$ cuts the variance in half, and $\rho = 0.8$ reduces it by 80%.

## Worked Example

A researcher measures reaction time (in milliseconds) for $n = 5$ subjects under two conditions: with and without caffeine.

| Subject | Without ($X_{i1}$) | With ($X_{i2}$) | Difference ($D_i$) |
|---|---|---|---|
| 1 | 250 | 230 | 20 |
| 2 | 310 | 290 | 20 |
| 3 | 280 | 260 | 20 |
| 4 | 340 | 325 | 15 |
| 5 | 220 | 205 | 15 |

**Paired analysis.** The differences are $\{20, 20, 20, 15, 15\}$ with $\bar{D} = 18$ and $s_D = 2.74$. The standard error is

$$
\text{SE}_{\text{paired}} = \frac{s_D}{\sqrt{n}} = \frac{2.74}{\sqrt{5}} \approx 1.22
$$

**Independent analysis (ignoring the pairing).** Treating the groups as independent samples: $\bar{X}_1 = 280$, $\bar{X}_2 = 262$, with $s_1 = 46.37$ and $s_2 = 46.69$. The pooled standard error is

$$
\text{SE}_{\text{indep}} = s_p\sqrt{\frac{1}{n_1} + \frac{1}{n_2}} = 46.53 \times \sqrt{\frac{2}{5}} \approx 29.43
$$

The paired standard error (1.22) is roughly 24 times smaller than the independent standard error (29.43). The large subject-to-subject variability in baseline reaction times (220 ms to 340 ms) dominates when pairing is ignored, but it cancels almost entirely in the within-subject differences.

!!! tip "Practical Guidance"
    The gains from pairing are largest when between-subject variability is large relative to within-subject variability. In the example above, subjects differ by over 100 ms in baseline reaction time, but the treatment effect is remarkably consistent (15--20 ms), producing a very high within-pair correlation.

## Decision Guidelines

| Factor | Favors Paired | Favors Independent |
|---|---|---|
| Within-pair correlation $\rho$ | High ($\rho > 0$) | Near zero or negative |
| Carryover effects | Absent or controllable | Present and uncontrollable |
| Logistics | Feasible to measure twice | Impractical to re-measure |
| Matching variables | Good predictors available | No strong predictors |
| Sample size | Limited (pairing extracts more information) | Abundant |

!!! warning "Do Not Mix Methods"
    Applying an independent-sample confidence interval to paired data ignores the correlation and produces intervals that are too wide, wasting statistical power. Applying a paired-sample interval to truly independent data violates the assumption of dependence and can produce intervals that are too narrow or too wide depending on the structure of the data.

## Exercises

**Exercise 1.**
A study measures blood pressure before and after a medication in 30 patients. The within-pair correlation is $\rho = 0.85$. Explain why a paired design is more powerful than an independent-samples design in this case.

??? success "Solution to Exercise 1"
    With $\rho = 0.85$, the variance of the paired differences is:

    $$
    \text{Var}(D) = \sigma_1^2 + \sigma_2^2 - 2\rho\sigma_1\sigma_2
    $$

    The high positive correlation means $2\rho\sigma_1\sigma_2$ is large, which substantially reduces $\text{Var}(D)$ compared to the independent-sample variance $\sigma_1^2 + \sigma_2^2$. A smaller variance of differences leads to a smaller standard error, narrower confidence intervals, and more statistical power. Each patient serves as their own control, removing between-patient variability from the comparison.

---

**Exercise 2.**
For each scenario, determine whether a paired or independent design is more appropriate: (a) comparing test scores of students before and after a tutoring program, (b) comparing average heights of randomly selected men and women.

??? success "Solution to Exercise 2"
    **(a) Paired design.** Each student is measured twice (before and after), creating natural pairs. The paired design accounts for individual differences in baseline ability, isolating the effect of the tutoring program.

    **(b) Independent design.** Men and women are separate, unrelated groups with no natural pairing. There is no meaningful way to match a specific man to a specific woman. An independent two-sample approach is appropriate.

---

**Exercise 3.**
A researcher has paired data but accidentally analyzes it as independent samples. If the true within-pair correlation is $\rho = 0.7$ and both groups have $\sigma = 10$, compare the standard error of the difference under both analyses when $n = 25$ pairs.

??? success "Solution to Exercise 3"
    **Paired analysis:** $\text{Var}(D) = \sigma_1^2 + \sigma_2^2 - 2\rho\sigma_1\sigma_2 = 100 + 100 - 2(0.7)(100) = 60$. So $\text{SE}_{\text{paired}} = \sqrt{60/25} = \sqrt{2.4} \approx 1.549$.

    **Independent analysis (incorrect):** $\text{SE}_{\text{indep}} = \sqrt{\sigma_1^2/n_1 + \sigma_2^2/n_2} = \sqrt{100/25 + 100/25} = \sqrt{8} \approx 2.828$.

    The incorrect independent analysis produces a standard error that is $2.828/1.549 \approx 1.83$ times larger. The confidence interval would be 83% wider than necessary, substantially reducing the chance of detecting a real effect.

---

**Exercise 4.**
Describe a scenario where carryover effects make a paired (crossover) design inappropriate despite the availability of repeated measurements.

??? success "Solution to Exercise 4"
    In a drug trial comparing a new analgesic to a placebo, each patient receives both treatments in sequence. If the new drug has a long-lasting biological effect (e.g., it permanently alters pain receptor sensitivity), then the response to the placebo in the second period is contaminated by the residual effect of the drug from the first period. This **carryover effect** means the second measurement does not reflect the true placebo response.

    In this case, even though repeated measurements are available, the paired analysis is invalid because the difference $D_i = X_{i,\text{drug}} - X_{i,\text{placebo}}$ is biased. An independent-samples design with separate groups (one receiving the drug, the other the placebo) would avoid this problem entirely.
