# Sample Surveys and Sampling Methods

A sample survey collects data from a representative subset of a population to support inference about the whole. The choice of sampling design — how the subset is selected — has at least as much impact on accuracy as the sample size: a small probability sample reliably outperforms a large convenience sample. This page covers the four standard probability-sampling designs, their precision properties, and the considerations that drive design choice in practice.

## Definition

A **sample survey** selects units from a population according to a defined sampling design and measures variables of interest on each selected unit. A **probability sample** is one where every unit has a known, nonzero selection probability — the property that makes valid inference possible. The four canonical designs:

| Method | Procedure | Advantage | Risk |
|---|---|---|---|
| Simple random | Each unit equally likely | Simplest theory, unbiased | May under-represent small subgroups |
| Stratified | Divide into strata, sample within each | Guaranteed subgroup coverage; can improve precision | Requires stratum knowledge in the frame |
| Cluster | Randomly select intact groups | Cost-effective for dispersed populations | Higher variance if clusters are internally homogeneous |
| Systematic | Every $k$-th unit from a list | Easy to implement; spreads sample across the frame | Bias if list has hidden periodicity |

## Explanation

### Sampling frame

Every probability design starts with a **frame** — an enumerable list of population units. Frame defects propagate directly into sampling bias:

- **Undercoverage**: units exist but are absent from the frame (homeless population in a telephone frame).
- **Overcoverage**: ineligible units appear in the frame (deceased voters in registration lists).
- **Duplicates**: a unit appears more than once with different selection probabilities.

Frame quality, not sample size, is the binding constraint on survey accuracy. The 1936 *Literary Digest* poll, with 2.4 million respondents, predicted the wrong U.S. presidential winner because its frame (telephone directories, automobile registrations) excluded the Depression-era poor.

### Simple random sampling (SRS)

Every subset of size $n$ is equally likely. Estimators are unbiased, and the variance of the sample mean is

$$
\mathrm{Var}(\bar y) = \frac{\sigma^2}{n}\left(1 - \frac{n}{N}\right)
$$

where the finite-population correction $(1 - n/N)$ becomes negligible whenever $n/N$ is small. SRS is theoretically clean but operationally awkward for large dispersed populations.

### Stratified sampling

Partition the population into strata (e.g., age groups, regions) and sample within each. With **proportional allocation** $n_h = n \cdot N_h / N$,

$$
\mathrm{Var}(\bar y_{\text{strat}}) = \sum_h \left(\frac{N_h}{N}\right)^{\!2} \frac{\sigma_h^2}{n_h} \le \mathrm{Var}(\bar y_{\text{SRS}})
$$

Strict equality when strata are identical; the gain is largest when within-stratum variances $\sigma_h^2$ are small and between-stratum means differ a lot. **Neyman allocation** $n_h \propto N_h \sigma_h$ minimizes variance further when stratum variances differ markedly.

### Cluster sampling

Randomly select intact clusters (city blocks, schools) and survey all units within selected clusters. Drastically cheaper to administer (only a few sites to visit), but units within a cluster tend to be similar — the **intracluster correlation** $\rho$ inflates variance by the **design effect**

$$
\mathrm{DEFF} \approx 1 + (m - 1)\rho
$$

where $m$ is cluster size. Cluster sampling pays for cost savings with reduced statistical efficiency; in practice it is almost always combined with stratification.

### Systematic sampling

Choose a random start in $\{1, \ldots, k\}$ and sample every $k$-th unit. Equivalent to SRS when the list order is random; biased when the list has periodicity matching $k$ (the textbook example: visiting every 7th house misses one day of the week of trash collection).

### Diminishing returns of sample size

For SRS, $\mathrm{SE}(\bar y) = \sigma/\sqrt{n}$. Quadrupling $n$ halves the SE. To halve the margin of error you need *four times* the data — a basic economics that shapes survey budgets.

## Examples

```python
"""Compare simple random and stratified sampling on a synthetic population."""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# === Population with two strata ===
n_pop = 10_000
stratum = rng.choice(["Young", "Old"], size=n_pop, p=[0.7, 0.3])
income = np.where(
    stratum == "Young",
    rng.normal(40_000, 10_000, n_pop),
    rng.normal(70_000, 15_000, n_pop),
)
pop = pd.DataFrame({"stratum": stratum, "income": income})
true_mean = pop["income"].mean()

# === SRS of size 200 ===
srs = pop.sample(200, random_state=1)

# === Stratified sample (proportional) ===
strat = pop.groupby("stratum", group_keys=False).apply(
    lambda x: x.sample(int(round(200 * len(x) / n_pop)), random_state=1)
)

print(f"True mean:        ${true_mean:,.0f}")
print(f"SRS estimate:     ${srs['income'].mean():,.0f}")
print(f"Stratified est.:  ${strat['income'].mean():,.0f}")
```

## Exercises

**Exercise 1.**
A university wants to survey student satisfaction. Identify the sampling method in each scenario.

**(a)** The registrar generates a list of all 20,000 students and uses a random-number generator to select 500.
**(b)** The university divides students into freshmen, sophomores, juniors, and seniors, then randomly selects 125 from each class.
**(c)** The university randomly selects 10 dormitories and surveys every resident in those dormitories.
**(d)** A researcher stands outside the library and surveys the first 200 students who walk by.
**(e)** Starting from a randomly chosen position in the alphabetized roster, the university picks every 40th student.

??? success "Solution to Exercise 1"
    (a) Simple random sampling.
    (b) Stratified random sampling, strata = class year.
    (c) Cluster sampling, clusters = dormitories.
    (d) Convenience sampling — not a probability sample; the resulting estimates have no valid measure of uncertainty.
    (e) Systematic sampling. Equivalent to SRS *if* the roster order is uncorrelated with satisfaction; biased if (say) alphabetical order proxies for ethnic background and satisfaction varies by ethnicity.

---

**Exercise 2.**
Derive the variance of the **stratified mean** under proportional allocation and show that it is at most the SRS variance. Use the law of total variance to identify when the gain is largest.

??? success "Solution to Exercise 2"
    Let $w_h = N_h/N$. Under proportional allocation $n_h = w_h n$, the stratified estimator is $\bar y_{\text{strat}} = \sum_h w_h \bar y_h$ with

    $$
    \mathrm{Var}(\bar y_{\text{strat}}) = \sum_h w_h^2 \cdot \frac{\sigma_h^2}{w_h n} = \frac{1}{n}\sum_h w_h \sigma_h^2
    $$

    The SRS variance is $\mathrm{Var}(\bar y_{\text{SRS}}) = \sigma^2 / n$. By the law of total variance,

    $$
    \sigma^2 = \underbrace{\sum_h w_h \sigma_h^2}_{\text{within}} + \underbrace{\sum_h w_h (\mu_h - \mu)^2}_{\text{between}}
    $$

    Therefore

    $$
    \mathrm{Var}(\bar y_{\text{strat}}) = \frac{\sigma^2 - \sum_h w_h (\mu_h - \mu)^2}{n} \le \frac{\sigma^2}{n} = \mathrm{Var}(\bar y_{\text{SRS}})
    $$

    The gain — the second term in the numerator — equals the **between-stratum variance**. Stratification helps most when stratum means $\mu_h$ differ markedly and within-stratum variance is small. $\square$

---

**Exercise 3.**
A health survey uses cluster sampling of schools (clusters), with $m = 30$ students per school and intracluster correlation $\rho = 0.10$ on the outcome. Compute the design effect and the effective sample size if 50 schools are sampled.

??? success "Solution to Exercise 3"
    $$
    \mathrm{DEFF} = 1 + (m - 1)\rho = 1 + 29 \times 0.10 = 3.9
    $$

    Nominal sample size: $n = 50 \times 30 = 1500$.

    Effective sample size: $n_{\text{eff}} = n / \mathrm{DEFF} = 1500 / 3.9 \approx 385$.

    Despite collecting data on 1,500 students, the precision is equivalent to an SRS of only $\sim 385$. To match the precision of an SRS of 1,500, the cluster design would need to sample roughly $1500 \times 3.9 \approx 5,850$ students — illustrating why cluster designs are chosen only when per-unit costs are dramatically lower.

---

**Exercise 4.**
Explain why a sample size of 2,400,000 from telephone directories (the 1936 *Literary Digest* poll) gave a worse estimate of voting intention than a probability sample of 50,000 (Gallup's poll the same year).

??? success "Solution to Exercise 4"
    The *Literary Digest* frame consisted of telephone subscribers and automobile owners. In 1936, both were luxury items concentrated among the wealthy — a group disproportionately Republican. The bias was not in the size of the sample but in the frame: respondents were systematically *unlike* the broader voting population.

    Sampling theory gives standard error $\sigma/\sqrt{n}$ around the *frame mean*, not the *population mean*. With $n = 2.4$ million, the SE is essentially zero, but the estimator is consistent for the wrong target. Gallup's smaller probability sample drew from a frame matching the voting population and was therefore unbiased; the modest sampling variability around an unbiased target beat near-zero variability around a biased one.

    **Big data does not fix bad data**: this is the modern restatement of the *Literary Digest* lesson, increasingly relevant as analysts work with massive but non-representative web and administrative datasets.

---

**Exercise 5.**
**Nonresponse** afflicts every real survey. A telephone poll achieves a 25% response rate; respondents are slightly older and more educated than the population. State two distinct strategies — one design-side and one analysis-side — for addressing the resulting bias.

??? success "Solution to Exercise 5"
    **Design-side (improve the response):**

    - Multiple callback attempts at different times of day.
    - Mixed-mode follow-up (mail, then phone, then in-person).
    - Modest incentives for completion (e.g., $5 gift cards).
    - Shorter questionnaires.

    Higher response rates reduce nonresponse bias even when respondents and non-respondents differ.

    **Analysis-side (post-survey adjustment):**

    - **Post-stratification weighting**: weight respondents to match known population marginals (age, education, region) drawn from the census. A 30-year-old female respondent in an oversampled cell receives less weight than one in an undersampled cell.
    - **Inverse-probability weighting (IPW)** when response propensities can be modeled from auxiliary data.
    - **Multiple imputation** of missing outcomes under explicit missingness assumptions.

    All analysis-side methods rest on assumptions (the response model, conditional ignorability of the outcome given the weighting variables). They reduce but do not eliminate bias; the only fully reliable defense is high response rate by design.

---

**Exercise 6.**
For a survey of a binary outcome with proportion $p$, you want a margin of error (95% CI half-width) of at most $0.03$. What is the minimum sample size under SRS without using any prior knowledge of $p$? Where does the formula come from?

??? success "Solution to Exercise 6"
    The 95% margin of error is approximately

    $$
    \mathrm{ME} = 1.96 \sqrt{\frac{p(1-p)}{n}}
    $$

    Worst case is at $p = 1/2$, where $p(1-p) = 0.25$. Setting $\mathrm{ME} \le 0.03$:

    $$
    n \ge \frac{(1.96)^2 \cdot 0.25}{(0.03)^2} \approx \frac{0.9604}{0.0009} \approx 1067
    $$

    So a sample of about **1,068** suffices regardless of the true $p$. This is the origin of the "$n \approx 1{,}000$" rule of thumb for national opinion polls.

    Note that this is *sampling* error only. Total error in real surveys also includes nonresponse bias, coverage bias, and measurement error — all of which can dwarf the $\pm 3$ pp sampling margin reported in the press.
