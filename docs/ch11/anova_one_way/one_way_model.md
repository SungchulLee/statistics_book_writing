# One-Way ANOVA: Model and Assumptions


## 1. One-Way ANOVA (Analysis of Variance)

### A. One-Way ANOVA

One-Way ANOVA (Analysis of Variance) is a statistical technique employed to determine if there are significant differences between the means of three or more independent groups. Unlike a two-sample t-test, which only allows for comparisons between two groups, One-Way ANOVA provides a method to simultaneously evaluate multiple groups, thereby reducing the likelihood of committing a Type I error—an incorrect rejection of a true null hypothesis—when conducting numerous pairwise comparisons.

The key concept in One-Way ANOVA is to partition the total variability observed in the data into two components: the variability **between** the groups and the variability **within** each group. By comparing these sources of variability, the test determines if the observed group means differ more than would be expected by chance.

In detail, One-Way ANOVA uses the **F-statistic**, which is calculated as the ratio of the variance between groups to the variance within groups. If the F-statistic is significantly larger than 1, it suggests that the differences between group means are greater than what might occur by random variation alone, implying that at least one group mean is significantly different from the others.

One of the assumptions of One-Way ANOVA is that the data in each group are normally distributed, and that the variances across these groups are approximately equal (homogeneity of variances). Moreover, the observations should be independent of each other. Violations of these assumptions can lead to inaccurate conclusions, and alternative approaches or adjustments (such as Welch's ANOVA or non-parametric tests) may be necessary.

Overall, One-Way ANOVA is particularly useful in a wide range of fields, such as medicine, psychology, agriculture, and business, where researchers or analysts want to compare the effects of different treatments, conditions, or groups. It provides a streamlined and efficient way to assess differences among multiple groups while controlling the risk of error associated with multiple hypothesis testing.

If the result of One-Way ANOVA is significant, further analysis is typically required to pinpoint which specific groups differ from each other. This follow-up process is often done using **post-hoc tests** (e.g., Tukey's HSD, Bonferroni correction), which help determine which pairs of group means are significantly different while controlling for Type I error.

### B. One-Way ANOVA vs. t-Test When Comparing Two Groups

When comparing two groups, One-Way ANOVA and the t-test share some similarities but also have significant differences in terms of their assumptions and applications.

#### What They Have in Common

1. **Purpose**: Both One-Way ANOVA and the t-test are used to compare group means and determine if there is a statistically significant difference between them. They are both designed to test whether the observed differences are likely to have occurred by chance.

2. **Null Hypothesis**: Both tests evaluate the null hypothesis that there is no difference between the group means. In other words, for two groups, both tests assume that the means are equal unless proven otherwise.

3. **Interpretation**: Both One-Way ANOVA and the t-test provide a p-value, which is used to determine whether to reject the null hypothesis. If the p-value is less than the specified significance level (e.g., 0.05), we reject the null hypothesis and conclude that there is a significant difference between the means.

4. **Equivalent Results for Two Groups**: When comparing only two groups with equal variances, One-Way ANOVA and the Student's t-test yield equivalent results. The F-statistic from One-Way ANOVA is equal to the square of the t-statistic from the t-test, meaning both tests will produce the same p-value and lead to the same conclusion.

#### Key Differences

1. **Assumptions About Variance**: The t-test comes in two forms: the Student's t-test and Welch's t-test. The Student's t-test assumes that the variances of the two groups are equal (homogeneity of variance). Welch's t-test, on the other hand, does not assume equal variances, making it more suitable when the group variances are different. One-Way ANOVA assumes equal variances across groups. This assumption, called homoscedasticity, is crucial for the reliability of the test. If this assumption is violated, the standard One-Way ANOVA may not be appropriate, and a version like Welch's ANOVA should be used instead.

2. **Use Case**: The t-test is specifically designed for comparing the means of two groups. It is simpler and more efficient when there are only two groups to compare. One-Way ANOVA is generally used to compare the means of three or more groups, but it can also be used for two groups. When there are only two groups, the t-test is typically more straightforward, though One-Way ANOVA will give the same result if variances are equal.

3. **Statistical Output**: The t-test provides a t-statistic and a p-value. The t-statistic reflects the difference between group means relative to the variability within the groups. One-Way ANOVA provides an F-statistic and a p-value. The F-statistic represents the ratio of between-group variance to within-group variance. For two groups, the F-statistic is simply the square of the t-statistic.

4. **Flexibility and Scalability**: The t-test is limited to comparing two groups. If there are more than two groups, performing multiple t-tests can increase the risk of a Type I error (false positive). One-Way ANOVA is more flexible, as it can compare two or more groups at once. It is designed to control the Type I error rate, making it more appropriate when comparing multiple groups.

#### Example of When to Use Each Test

- **t-Test**: Suppose you want to compare the mean scores of students who used two different study methods. If the two groups (e.g., Method A vs. Method B) have similar variances, a Student's t-test would be appropriate. If the variances are unequal, Welch's t-test should be used.

- **One-Way ANOVA**: Suppose you want to compare the mean scores of students who used three different study methods (Method A, Method B, and Method C). In this case, One-Way ANOVA would be the appropriate choice. If you only have two groups, One-Way ANOVA can still be used, but a t-test is generally simpler.

#### Summary

One-Way ANOVA and the t-test share the common goal of comparing group means to determine if there are significant differences. They both test the null hypothesis of equal means and provide a p-value for interpretation. However, they differ in their assumptions about variance and in their intended use cases.

- When comparing **two groups with equal variances**, a **t-test** (Student's t-test) is often the better choice due to its simplicity and efficiency.
- When **variances are unequal**, **Welch's t-test** is preferred for comparing two groups.
- When comparing **more than two groups**, **One-Way ANOVA** is more appropriate, as it controls for the Type I error rate better than performing multiple t-tests.

Thus, while One-Way ANOVA and the t-test can produce equivalent results for two groups under certain conditions, differences in variance assumptions and flexibility make each test suitable for different situations.

### C. Comparison of Null of One-Way ANOVA to Linear Regression

$$\begin{array}{cccccccc}
&&\text{Linear Regression}&&\text{One Way ANOVA}\\
\text{Model}&&y=\alpha+\beta x+\text{noise}
&&
y=\alpha+\sum_{i=1}^{k}\beta_i 1_{x\in C_i}+\text{noise}
\\
\text{H}_0&&\beta=0
&&
\quad\beta_1=\cdots=\beta_k=0\\
\end{array}$$

One-Way ANOVA (Analysis of Variance) can be viewed as a specific form of a linear regression model where the independent variable is categorical, representing distinct groups. In a typical linear regression, the predictor variable $x$ is continuous, and the relationship between the dependent variable $y$ and $x$ is expressed as:

$$y = \alpha + \beta x + \text{noise}$$

where $\alpha$ is the intercept, $\beta$ is the slope, and "noise" refers to the random error. The null hypothesis, $H_0$, in this model is $\beta = 0$, meaning that $x$ has no effect on $y$.

$$H_0: \beta = 0$$

In contrast, One-Way ANOVA models the relationship by comparing the means of several groups. The categorical variable defines different groups $C_1, C_2, \dots, C_k$, and the model is written as:

$$y = \alpha + \sum_{i=1}^{k} \beta_i 1_{x \in C_i} + \text{noise}$$

where $1_{x \in C_i}$ is an indicator function that equals 1 if $x$ belongs to group $C_i$ and 0 otherwise. Here, $\alpha$ is the overall mean, and each $\beta_i$ represents the deviation of group $C_i$ from the overall mean.

The null hypothesis in One-Way ANOVA is that the means of all the groups are equal, which is expressed as:

$$H_0: \beta_1 = \beta_2 = \cdots = \beta_k = 0$$

This means there are no significant differences among the group means. Rejecting this hypothesis suggests that at least one group mean differs significantly from the others.

## 2. Assumptions of One-Way ANOVA

### When to Use One-Way ANOVA

- **Independent Variable**: One-Way ANOVA is used when you have a single independent variable (also called a factor) that has three or more levels or groups. For example, if you're comparing the effects of three different diets on weight loss, the diets would be the groups.

- **Dependent Variable**: The dependent variable (what you're measuring) should be continuous. This means it can take on any value within a range, such as height, weight, or test scores.

### Assumptions of One-Way ANOVA

Before conducting a One-Way ANOVA, certain assumptions must be met:

- **Independence of Observations**: Each sample is independent of the others. The data should be collected through a random process, ensuring that the observations in each group are independent of one another.

- **Normality**: The dependent variable should be approximately normally distributed within each group. This is critical for small sample sizes and can be verified using visual methods like Q-Q plots or statistical tests like the Shapiro-Wilk test. For larger samples, the Central Limit Theorem helps approximate normality.

- **Homogeneity of Variance (Homoscedasticity)**: The variances across the groups should be similar (homogeneous). This assumption can be tested using Levene's test or Bartlett's Test. If this assumption is violated, alternative methods like Welch's ANOVA may be considered.

## 3. Limitations

- **Assumption Sensitivity**: The results of a One-Way ANOVA can be sensitive to violations of assumptions, particularly homogeneity of variance and normality.
- **Only Detects Overall Difference**: While One-Way ANOVA can detect if there is a significant difference between groups, it does not specify which groups are different. Post-hoc tests are needed for detailed comparisons.
## Exercises

**Exercise 1.**
A researcher plans to compare average customer satisfaction scores across four different retail store layouts. Each layout will be tested with an independent group of 20 customers. State the three assumptions required for a valid one-way ANOVA and, for each assumption, name one diagnostic method that can be used to verify it.

??? success "Solution to Exercise 1"
    The three assumptions are:

    1. **Independence of observations.** Verified by reviewing the study design: customers must be randomly sampled and randomly assigned to layouts, with no customer appearing in more than one group.

    2. **Normality.** The satisfaction scores within each group should be approximately normally distributed. This can be checked using a Q-Q plot of the residuals or the Shapiro-Wilk test.

    3. **Homogeneity of variance (homoscedasticity).** The variances of the satisfaction scores should be approximately equal across the four groups. This can be checked using Levene's test.

---

**Exercise 2.**
Explain why performing six separate two-sample $t$-tests to compare all pairs among four group means inflates the family-wise Type I error rate. If each test uses $\alpha = 0.05$, compute an upper bound on the probability of at least one false positive across all six tests.

??? success "Solution to Exercise 2"
    With six independent tests each at $\alpha = 0.05$, the probability of making at least one Type I error is bounded by:

    $$
    1 - (1 - 0.05)^6 = 1 - 0.95^6 \approx 1 - 0.735 = 0.265
    $$

    So the family-wise error rate can be as high as approximately $26.5\%$, far exceeding the nominal $5\%$ level. One-way ANOVA avoids this inflation by testing all group means simultaneously with a single F-test.

---

**Exercise 3.**
Show that for $k = 2$ groups with equal variances, the one-way ANOVA F-statistic equals the square of the two-sample (pooled) $t$-statistic. Start from the definitions of SSB, SSW, and the pooled $t$-statistic.

??? success "Solution to Exercise 3"
    For $k = 2$ groups of sizes $n_1$ and $n_2$ with pooled variance $s_p^2 = \text{SSW}/(n_1 + n_2 - 2)$:

    $$
    \text{SSB} = \frac{n_1 n_2}{n_1 + n_2}(\bar{Y}_1 - \bar{Y}_2)^2
    $$

    $$
    F = \frac{\text{SSB}/1}{\text{SSW}/(N-2)} = \frac{n_1 n_2 (\bar{Y}_1 - \bar{Y}_2)^2}{(n_1+n_2) s_p^2}
    $$

    The pooled two-sample $t$-statistic is:

    $$
    t = \frac{\bar{Y}_1 - \bar{Y}_2}{s_p \sqrt{1/n_1 + 1/n_2}}
    $$

    Squaring:

    $$
    t^2 = \frac{(\bar{Y}_1 - \bar{Y}_2)^2}{s_p^2 (1/n_1 + 1/n_2)} = \frac{n_1 n_2 (\bar{Y}_1 - \bar{Y}_2)^2}{(n_1 + n_2) s_p^2} = F
    $$

    Therefore $F = t^2$ when $k = 2$. $\square$

---

**Exercise 4.**
A marketing team observes group variances of $s_1^2 = 4.2$, $s_2^2 = 18.7$, and $s_3^2 = 5.1$ across three advertising strategies with sample sizes $n_1 = 15$, $n_2 = 10$, $n_3 = 20$. Explain why the standard one-way ANOVA may produce misleading results and recommend a more appropriate test.

??? success "Solution to Exercise 4"
    The largest variance ($18.7$) is more than four times the smallest ($4.2$), and the sample sizes are unequal. When the group with the largest variance also has the smallest sample size ($n_2 = 10$), the pooled variance in standard ANOVA underestimates the true variability for group 2 and overestimates it for the others. This makes the F-test liberal, rejecting $H_0$ more often than the nominal $\alpha$ level.

    The appropriate alternative is **Welch's one-way ANOVA**, which uses separate variance estimates for each group and adjusts the degrees of freedom via the Welch-Satterthwaite equation, maintaining the correct Type I error rate under heteroscedasticity.
