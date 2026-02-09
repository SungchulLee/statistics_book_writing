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

## Significance Level ($\alpha$) and Error Types

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
