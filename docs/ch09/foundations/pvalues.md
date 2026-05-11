# Test Statistics and P-values

## Test Statistic

A **test statistic** is a calculated value from the sample data that, when compared to a threshold from a theoretical distribution, helps decide whether to reject the null hypothesis, $H_0$. The choice of test statistic depends on the data type and the hypothesis we test. Standard test statistics include the z-statistic, t-statistic, and chi-square statistic.

- **$z$-statistic**: This statistic can be used when the population variance is known, the sample size is large, or the data follow a normal distribution.
- **$t$-statistic**: This statistic can be used when the population variance is unknown and the sample size is small. It assumes that the data come from a normally distributed population.
- **$\chi^2$-statistic**: Typically used for categorical data to test the independence or goodness of fit.
- **$f$-statistic**: Used primarily in the analysis of variance (ANOVA) and regression analysis to compare variances between groups and test if the group means are significantly different. We can compute this statistic by dividing the variance explained by the model by the unexplained variance.

The formula for calculating a test statistic varies based on the test. For example, a common formula for the z-statistic in testing population means is

$$ z = \frac{\bar{x} - \mu_0}{\sigma/\sqrt{n}} $$

where $\bar{x}$ is the sample mean, $\mu_0$ is the mean under the null hypothesis, $\sigma$ is the population standard deviation, and $n$ is the sample size.

---

## Significance Level (alpha) and Error Types
Understanding errors in hypothesis testing is essential for correctly interpreting the results of statistical tests. When conducting a hypothesis test, two potential types of errors can occur:

- **Type I Error**: A Type I error, often called a "false positive," occurs when the null hypothesis $H_0$ is true, but we mistakenly reject it in favor of the alternative hypothesis $H_a$. This Type I error is analogous to convicting an innocent person in a trial. The **significance level** $\alpha$ represents the threshold we set for rejecting the null hypothesis but does not directly represent the probability of committing a Type I error. Instead, $\alpha$ is the probability of rejecting $H_0$ when true, which is the potential for a Type I error. For example, if we set $\alpha = 0.05$, there is a 5% risk of incorrectly rejecting $H_0$.

$$\alpha=P(\text{Type I Error})=P(\text{Reject } H_0 \mid H_0)$$

- **Type II Error**: A Type II error, also known as a "false negative," occurs when the null hypothesis $H_0$ is false, but we fail to reject it, thereby incorrectly retaining it. This Type II error is akin to acquitting a guilty person in a trial. The probability of committing a Type II error is denoted by $\beta$. A lower $\beta$ value implies a lower risk of retaining $H_0$ when it is false. The complement of $\beta$ is the **power of the test**, which represents the probability of correctly rejecting $H_0$ when it is false. In other words, power is the test's ability to detect an effect when there is one.

$$\beta=P(\text{Type II Error})=P(\text{Accept } H_0 \mid H_a)$$

The balance between $\alpha$ and $\beta$ is crucial in hypothesis testing. Lowering $\alpha$ decreases the likelihood of a Type I error but increases the risk of a Type II error, and vice versa. Therefore, selecting an appropriate significance level depends on the context of the test and the consequences of making either type of error.

---

## P-value and Its Interpretation

The **p-value** is a fundamental concept in hypothesis testing that quantifies the evidence against the null hypothesis ($H_0$). Specifically, the p-value represents the probability of observing a test statistic at least as extreme as the one computed from the sample data, assuming that the null hypothesis is true. It helps statisticians determine whether the observed data are consistent with $H_0$ or whether the data provide enough evidence to reject it in favor of the alternative hypothesis ($H_a$).

The p-value measures how unusual the observed data are under the assumption that $H_0$ holds:

- A **small p-value** ($p \leq \alpha$) indicates that the observed data are doubtful under the null hypothesis. This small p-value suggests strong evidence against $H_0$, prompting its **rejection**. In other words, the smaller the p-value, the less plausible the null hypothesis could explain the observed data. For instance, if we set the significance level $\alpha = 0.05$, and the p-value from the test is $p = 0.01$, there is only a 1% chance of observing data as extreme as this sample, assuming $H_0$ is true. As a result, we reject $H_0$.

- A **large p-value** ($p > \alpha$) suggests that the data are not sufficiently inconsistent with the null hypothesis. In this case, there is insufficient evidence to reject $H_0$; thus, we **fail to reject** it. This large p-value does not imply that $H_0$ is true; instead, the data does not provide strong enough evidence against it. For example, if the p-value is $p = 0.20$ and $\alpha = 0.05$, we conclude that the data are plausible under $H_0$, so we do not reject the null hypothesis.

In summary, the p-value serves as a tool for assessing the compatibility of the observed data with the null hypothesis. A lower p-value indicates stronger evidence against $H_0$. A higher p-value indicates weaker evidence against it, guiding us toward either rejecting or retaining the null hypothesis based on the significance level $\alpha$.

---

## Test Names

### Test Names — Hypothesis

$$\begin{array}{lll}
\text{Two-Sided}&&\displaystyle\text{$H_0$ : $\mu=\mu_0$ vs $H_1$ : $\mu\neq\mu_0$}\\
\text{Less}&&\displaystyle\text{$H_0$ : $\mu=\mu_0$ vs $H_1$ : $\mu<\mu_0$}\\
\text{Greater}&&\displaystyle\text{$H_0$ : $\mu=\mu_0$ vs $H_1$ : $\mu>\mu_0$}\\
\end{array}$$

### Test Names — Data

$$\begin{array}{lll}
\text{One Sample}&&\displaystyle\{x_1,\cdots,x_n\}\\
\text{Two Sample}&&\displaystyle\{x_1,\cdots,x_n\}\ \text{and}\ \{y_1,\cdots,y_m\}\\
\text{Paired Sample}&&\displaystyle\{(x_1,y_1),\cdots,(x_n,y_n)\}\\
\end{array}$$

### Test Names — Sampling Distribution

$$\begin{array}{lll}
\text{$z$ Test}&&\displaystyle\text{sampling distribution is expressed in terms of $z$}\\
\text{$t$ Test}&&\displaystyle\text{sampling distribution is expressed in terms of $t$}\\
\text{$f$ Test}&&\displaystyle\text{sampling distribution is expressed in terms of $f$}\\
\text{$\chi^2$ Test}&&\displaystyle\text{sampling distribution is expressed in terms of $\chi^2$}\\
\end{array}$$

### Naming Convention

$$\begin{array}{ccc}
\text{Two-Sided}&\text{Two Sample}&\text{$z$ Test}\\
\uparrow&\uparrow&\uparrow\\
\text{Hypothesis}&\text{Data}&\text{Sampling Distribution}\\
\end{array}$$

---

## Steps in Hypothesis Testing

Hypothesis testing is a systematic process used to evaluate assumptions about a population parameter based on sample data. The steps involved in hypothesis testing are critical to ensure the integrity and accuracy of the conclusions drawn.

### Step 1: Formulating Hypotheses

The first step in hypothesis testing is to formulate two opposing hypotheses:

- **Null Hypothesis ($H_0$)**: Assumes no effect or difference in the population. It serves as the baseline hypothesis that the test seeks to challenge.
- **Alternative Hypothesis ($H_a$)**: Posits that there is an effect or a difference. This hypothesis is a choice if the evidence suggests we can reject the null hypothesis.

For example, if investigating whether a new drug lowers blood pressure, the hypotheses might be:

- $H_0$: The drug's mean decrease in blood pressure is zero.
- $H_a$: The mean decrease in blood pressure by the drug is greater than zero.

### Step 2: Choosing the Appropriate Test

Selecting the correct statistical test is crucial and depends on the type of data and the hypothesis. The choice of the test affects how we compute the test statistic and how we evaluate the hypotheses. Factors include the data's measurement level, the sample size, and whether the data follows a normal distribution. Standard tests include the z-test, t-test, chi-square test, and ANOVA.

### Step 3: Deciding on the Significance Level

The significance level ($\alpha$) is the threshold at which we reject the null hypothesis. It reflects the probability of committing a Type I error — rejecting the null hypothesis when it is true. Typical values for $\alpha$ are 0.05, 0.01, and 0.10. We must decide on this level before analyzing the data to avoid bias.

### Step 4: Calculating the Test Statistic

The test statistic is a value computed from the sample data that, assuming the null hypothesis is true, follows a specific probability distribution. The test statistic compares the observed data to the sampling distribution under the null hypothesis. For example, in a z-test for a mean, the test statistic is calculated as:

$$ z = \frac{\bar{x} - \mu_0}{\sigma/\sqrt{n}} $$

where $\bar{x}$ is the sample mean, $\mu_0$ is the mean under the null hypothesis, $\sigma$ is the population standard deviation, and $n$ is the sample size.

### Step 5: Determining the P-value

The p-value is the probability of observing a test statistic as extreme as, or more extreme than, the observed statistic under the null hypothesis. It is a crucial measure in deciding whether to reject the null hypothesis. A p-value less than $\alpha$ suggests strong evidence against the null hypothesis.

### Step 6: Making the Decision

Based on the p-value and the predetermined significance level, the decision is made as follows:

- If the p-value $\leq \alpha$, reject the null hypothesis.
- If the p-value $> \alpha$, do not reject the null hypothesis.

### Step 7: Concluding the Hypothesis Test

The final step involves interpreting the results in the context of the research question. This final step includes considering the implications of the decision, discussing potential errors, and suggesting further research if needed.

## Exercises

**Exercise 1.**
**Define p-value formally** and explain its relationship to the test statistic distribution under $H_0$.

??? success "Solution to Exercise 1"
    **P-value:** the probability, under $H_0$, of obtaining a test statistic at least as extreme as the observed value.

    Formally: $p = P_{H_0}(T \ge t_{\text{obs}})$ (one-sided) or $p = 2 \min[P_{H_0}(T \ge t_{\text{obs}}), P_{H_0}(T \le t_{\text{obs}})]$ (two-sided).

    Computed as the tail area of the null distribution beyond the observed statistic.

    Under $H_0$, the p-value itself is **uniformly distributed** on $[0, 1]$ (probability integral transform). This justifies the decision rule "reject if $p \le \alpha$" — it gives Type I error rate exactly $\alpha$.

---

**Exercise 2.**
**Vegetarian teens.** Evie samples 25 students, finds 20% vegetarian. Tests $H_0: p = 0.06$ vs $H_1: p > 0.06$. Compute the exact p-value.

??? success "Solution to Exercise 2"
    Observed: $X = 5$ vegetarians in $n = 25$. Under $H_0$: $X \sim \mathrm{Binomial}(25, 0.06)$.

    P-value: $P(X \ge 5) = 1 - P(X \le 4) = 1 - \sum_{k=0}^4 \binom{25}{k}(0.06)^k (0.94)^{25-k}$.

    Numerical: $P(X \le 4) \approx 0.9979$. So $p \approx 0.0021$.

    Strong evidence at school's veggie rate exceeds 6%. Reject $H_0$ at $\alpha = 0.01$.

    Normal approximation would give: $z = (0.20 - 0.06)/\sqrt{0.06 \cdot 0.94/25} \approx 2.95$, $p \approx 0.0016$ — close but slightly different (exact is preferred for small $n$ with small $p$).

---

**Exercise 3.**
**Multilingual Americans.** $\hat p = 40/120 \approx 0.333$, test $H_0: p = 0.26$ vs $H_1: p > 0.26$. Compute p-value via normal approximation.

??? success "Solution to Exercise 3"
    $\mathrm{SE} = \sqrt{0.26 \cdot 0.74/120} \approx 0.0400$. $z = (0.333 - 0.26)/0.0400 \approx 1.83$.

    P-value: $P(Z > 1.83) = 1 - \Phi(1.83) \approx 0.034$. Reject $H_0$ at $\alpha = 0.05$.

    Conclusion: evidence that more than 26% of Americans speak multiple languages.

    Check: $np_0 = 31.2 \ge 10$, $n(1-p_0) = 88.8 \ge 10$. Normal approximation valid.

---

**Exercise 4.**
**Common p-value misinterpretations.** List three.

??? success "Solution to Exercise 4"
    1. **"$p$-value is the probability $H_0$ is true."** WRONG. P-value is a probability assuming $H_0$, not about $H_0$. To get $P(H_0 \mid \text{data})$, need Bayes' theorem with a prior.

    2. **"$1 - p$ is the probability $H_1$ is true."** WRONG. Same confusion.

    3. **"$p < 0.05$ means a large effect."** WRONG. P-value depends on both effect size and sample size. With $n = 10^6$, a tiny irrelevant effect can have $p < 10^{-10}$. Always report effect size and CI alongside p-value.

    Other misconceptions: "p = 0.05 means 5% chance of error" (probability statement about hypothesis, not procedure); "if $p > 0.05$, $H_0$ is true" (failing to reject ≠ accepting).

---

**Exercise 5.**
**Simulation-based p-value.** Demonstrate for the vegetarian example.

??? success "Solution to Exercise 5"
    ```python
    import numpy as np
    rng = np.random.default_rng(42)
    p0, n, x_obs = 0.06, 25, 5
    n_sim = 10_000
    samples = rng.binomial(n, p0, size=n_sim)
    p_value = (samples >= x_obs).mean()
    print(f"Simulated p-value: {p_value:.4f}")
    ```

    Expected: $p \approx 0.002$, matching exact binomial.

    **Advantages of simulation:**

    - Works for any test statistic, not just standard ones.
    - Automatically handles continuity correction.
    - Reveals the *shape* of the null distribution (histogram).

    **Disadvantages:**

    - Monte Carlo error: $\mathrm{SE} \approx \sqrt{p(1-p)/n_{\text{sim}}}$. For $p = 0.002$, need $n_{\text{sim}} \approx 10^6$ for 3-digit accuracy.
    - Slower than analytic when both available.

---

**Exercise 6.**
**P-value vs effect size.** Why is the p-value alone insufficient?

??? success "Solution to Exercise 6"
    P-value confounds **effect size** and **sample size**:

    - Tiny effect + huge $n$: $p$ small, but practical significance negligible.
    - Large effect + small $n$: $p$ large (no significance), but effect may be important.

    Two A/B tests:
    - Test 1: 10000 users, 51% vs 50% conversion. $p = 0.045$. Effect: 1pp.
    - Test 2: 100 users, 70% vs 50% conversion. $p = 0.003$. Effect: 20pp.

    Test 1 is "more significant" by p-value but Test 2 has a far larger effect.

    **Always report:**

    - Effect size (raw or standardized like Cohen's $d$).
    - Confidence interval (range of plausible effects).
    - P-value (evidence against $H_0$).

    Together these tell the full story. P-value alone is impoverished.
