# Populations and Samples

The distinction between population and sample is the foundation of all inferential statistics. Every confidence interval, hypothesis test, and regression model in this book is a statement about *the population* based on what we observe in *the sample*. Mastering this distinction — and the related ideas of representativeness, randomness, and sampling error — is the first step to interpreting any statistical analysis honestly.

## Definition

The **population** is the complete set of individuals or observations of interest. A **sample** is a subset of the population selected for measurement. Population quantities are **parameters** (denoted $\mu$, $\sigma^2$, $p$, $\beta$); sample quantities are **statistics** (denoted $\bar{x}$, $s^2$, $\hat{p}$, $\hat\beta$).

$$
\mu = \frac{1}{N}\sum_{i=1}^{N} x_i \qquad \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i
$$

$$
\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2 \qquad s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2
$$

A **target population** is what the analyst wants to learn about; the **study population** is what the sampling frame actually covers; the **sample** is what gets measured. Any gap between the target and study populations is a source of bias that no sample size can eliminate.

## Explanation

### Why we sample

A complete census of a population is rarely feasible. Reasons include:

- **Cost**: measuring every U.S. household for a survey would consume billions of dollars.
- **Time**: by the time a census is complete, the population has changed.
- **Practicality**: testing every light bulb's lifetime would destroy the product.
- **Definition**: the "population" may be implicit (all possible measurements of a physical constant, all future patients of a hospital). Only a sample is ever observable.

### What makes a sample useful

A useful sample lets the analyst quantify (a) what's true on average and (b) how uncertain that estimate is. The minimum requirement is **probability sampling**: every unit in the population has a known, nonzero probability of selection. This requirement supports:

- **Unbiasedness** in expectation: $\mathbb{E}[\bar x] = \mu$ under simple random sampling.
- **Quantifiable uncertainty**: standard errors and confidence intervals have valid coverage.
- **No invisible omissions**: probability sampling guarantees every part of the population is reachable.

Convenience samples (people who walked by, friends of the analyst, online volunteers) violate this and produce statistics with no valid measure of uncertainty.

### Sampling error vs. bias

**Sampling error** is the random variation in a statistic across different samples of the same size. It shrinks with $\sqrt{n}$ and is captured by the standard error.

**Bias** is systematic error — the difference between $\mathbb{E}[\hat\theta]$ and $\theta$. It does not shrink with $n$. A million-sample biased survey is more confidently wrong than a thousand-sample biased survey.

The two are independent quantities; a survey can be precise (low SE) and biased, or unbiased and imprecise.

### Bessel's correction

The sample variance uses $n - 1$ in the denominator, not $n$. The reason: $\bar x$ minimizes $\sum (x_i - c)^2$ over $c$, which means $\sum (x_i - \bar x)^2 \le \sum (x_i - \mu)^2$ in any sample. Dividing by $n$ would give $\mathbb{E}[\tilde s^2] = \frac{n-1}{n}\sigma^2$ — a systematic downward bias. Dividing by $n - 1$ exactly compensates, producing the unique scalar multiple of the centered sum of squares that is unbiased for $\sigma^2$.

## Examples

```python
"""Sampling error decreases with sqrt(n)."""

import numpy as np

rng = np.random.default_rng(42)

mu_true, sigma_true = 170, 10
population = rng.normal(mu_true, sigma_true, size=100_000)

for n in [10, 100, 1_000, 10_000]:
    sample = rng.choice(population, size=n, replace=False)
    print(f"n = {n:>5d}: x_bar = {sample.mean():.3f}, "
          f"empirical SE = {sample.std(ddof=1)/np.sqrt(n):.3f}, "
          f"theoretical = {sigma_true/np.sqrt(n):.3f}")
```

The standard error drops by roughly a factor of 10 each time $n$ multiplies by 100 — the $\sqrt{n}$ law in action.

## Exercises

**Exercise 1.**
For each scenario, identify the **population** and the **sample**.

**(a)** A polling firm calls 1,200 randomly selected registered voters in a state to estimate the proportion who support a ballot measure.
**(b)** A hospital reviews the medical records of 500 patients admitted in 2024 to study post-surgical complications.
**(c)** An online retailer analyzes click data from 10,000 randomly selected user sessions to estimate the overall conversion rate.
**(d)** A physicist measures the gravitational constant 50 times in a lab to estimate its value.

??? success "Solution to Exercise 1"
    (a) Population: all registered voters in the state. Sample: the 1,200 voters called.
    (b) Population: all patients who could have been admitted to the hospital under similar conditions. Sample: the 500 reviewed.
    (c) Population: all sessions on the retailer's website. Sample: the 10,000 selected sessions.
    (d) Population: the conceptual ensemble of all possible measurements under the same experimental setup. Sample: the 50 measurements. (When the "population" is conceptual, every observation is in some sense a sample from an implicit superpopulation.)

---

**Exercise 2.**
Prove that the sample variance $s^2 = \frac{1}{n-1}\sum (X_i - \bar X)^2$ is unbiased for the population variance $\sigma^2$ when $X_1, \ldots, X_n$ are i.i.d. with mean $\mu$ and variance $\sigma^2$.

??? success "Solution to Exercise 2"
    Use $\sum (X_i - \bar X)^2 = \sum X_i^2 - n \bar X^2$. Taking expectations:

    $$
    \mathbb{E}\!\sum X_i^2 = n(\sigma^2 + \mu^2), \qquad \mathbb{E}[n \bar X^2] = n\!\left(\frac{\sigma^2}{n} + \mu^2\right) = \sigma^2 + n\mu^2
    $$

    Subtracting:

    $$
    \mathbb{E}\!\sum (X_i - \bar X)^2 = n\sigma^2 + n\mu^2 - \sigma^2 - n\mu^2 = (n-1)\sigma^2
    $$

    Hence $\mathbb{E}[s^2] = \sigma^2$. $\square$

---

**Exercise 3.**
A polling firm reports that "75% of Americans approve of policy X, based on 800 phone interviews conducted between 5 pm and 7 pm on weekdays." Critique the sampling design and list two distinct sources of bias.

??? success "Solution to Exercise 3"
    Two sources of bias:

    - **Coverage / frame bias**: the frame is phone subscribers. Non-subscribers and individuals without phones are excluded. Even within phone owners, those who answer unknown numbers (i.e., respond to a polling call) skew older and more retired.
    - **Time-of-day bias**: 5–7 pm weekday calls miss working adults, shift workers, and people with evening commitments. Respondents skew toward retirees, the unemployed, and people working from home — groups that may differ systematically on policy preferences.

    The combination produces a sample that is unrepresentative regardless of how large $n$ is. The reported 75% has a sharp confidence interval *around the wrong target population*.

---

**Exercise 4.**
Distinguish between **sampling error** and **bias**. For each of the following, identify which is at play (or both): (a) a 95% CI for $\mu$ is wider than the analyst wants; (b) an exit poll consistently overstates Democratic support across many elections; (c) a thermometer reads 2 degrees high regardless of how many readings are averaged.

??? success "Solution to Exercise 4"
    (a) **Sampling error.** Wide CIs are a precision problem; they shrink with larger $n$.
    (b) **Bias.** Repeated systematic error in the same direction across many independent samples is bias, not random sampling error. The likely mechanism is differential response (Democrats more willing to participate in exit polls). Increasing $n$ does not reduce it.
    (c) **Bias** (measurement bias). The thermometer is unbiased in the **statistical** sense — it always reads 2 degrees high. No averaging fixes it; only recalibration does.

---

**Exercise 5.**
A new manufacturing line produces 10 million widgets per year. Quality control samples 1,000 widgets monthly and tests them. Why is this a *sample* of the production population? What changes when the test is destructive?

??? success "Solution to Exercise 5"
    The 10 million widgets per year are the population of interest. The 1,000 tested per month form a sample — typically 12,000 per year — selected to estimate population defect rate, dimension distributions, etc.

    Sample inference applies: defect rate estimated as $\hat p = (\text{defects in sample})/1000$ with SE $\sqrt{\hat p(1 - \hat p)/1000}$, and 95% CI roughly $\hat p \pm 2\sqrt{\hat p(1 - \hat p)/1000}$.

    **Destructive testing:** the test breaks the widget (crash test, breaking strength, lifetime test). A census is now impossible by definition — testing every widget would leave nothing to sell. The sample-based inference is the only option. This is a common reason to choose between *non-destructive* approximations (X-ray, optical inspection — every unit; less informative) and *destructive* tests (a sample; more informative but unrepeatable on that unit).

---

**Exercise 6.**
The **finite-population correction (FPC)** factor $\sqrt{1 - n/N}$ multiplies the standard error of the sample mean for sampling without replacement from a finite population. When does it matter, and when is it ignored?

??? success "Solution to Exercise 6"
    The variance of $\bar X$ for an SRS without replacement from a population of size $N$ is

    $$
    \mathrm{Var}(\bar X) = \frac{\sigma^2}{n}\left(1 - \frac{n}{N}\right)
    $$

    The factor $(1 - n/N)$ goes to 1 when $n \ll N$ (the FPC is negligible) and to 0 as $n \to N$ (when you've sampled the whole population, there is no sampling uncertainty).

    **Matters** in finite populations with non-trivial sampling fractions: a small audit sampling 50 invoices from a population of 200, where $n/N = 0.25$. Ignoring the FPC overestimates the SE by a factor of $\sqrt{1/(1-0.25)} = 1.15$ — a 15% inflation.

    **Ignored** in surveys of very large populations ($n/N \ll 0.05$) — the FPC ≈ 1 and standard formulas suffice. National polls of 1,000 from a population of 200 million use $n/N = 5 \times 10^{-6}$; the FPC is indistinguishable from 1. This is why infinite-population formulas dominate textbooks but the FPC reappears in audit, quality-control, and election-recount contexts.
