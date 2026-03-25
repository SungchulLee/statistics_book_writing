# A/B Testing and Experimental Design

## Overview

In practice, researchers and analysts frequently need to compare the effectiveness of different treatments: a pharmaceutical company tests two drug formulations, a tech company evaluates three website layouts, or an economist compares policy interventions across regions. These comparisons require a principled statistical framework that controls error rates while providing clear decision rules. A/B testing formalizes this process by combining experimental design principles with hypothesis testing, and when the number of treatment groups exceeds two, the analysis reduces to the one-way ANOVA framework developed earlier in this chapter.

This section establishes the statistical foundations of A/B testing, connects it explicitly to the ANOVA F-test, and addresses the practical considerations of sample size determination and multiple comparisons.

## Design Principles

A well-designed A/B test rests on three foundational principles, each of which has a direct statistical justification.

**Randomization.** Subjects are assigned to treatment groups using a random mechanism (e.g., a pseudorandom number generator). Randomization ensures that the treatment assignment is independent of all potential confounders, both observed and unobserved. Without randomization, observed differences between groups may reflect pre-existing differences rather than genuine treatment effects.

**Control.** Every experiment includes a baseline group (the control) that receives either no treatment or the current standard. The control group provides the reference against which treatment effects are measured. In the ANOVA model, the control group mean serves as the benchmark $\mu_1$ (or equivalently, the intercept $\mu$ when treatment effects are parameterized as deviations $\alpha_i$).

**Replication.** Each treatment group must contain enough independent observations to detect a practically meaningful effect with adequate statistical power. Replication reduces the within-group variance estimate and narrows the confidence intervals for treatment differences. The next subsection formalizes the sample size requirement.

These principles ensure that the resulting data satisfy the independence and identically distributed assumptions required by the ANOVA F-test.

## Connection to ANOVA

### Two-Group Case

When an A/B test compares exactly two groups (control vs. treatment), the analyst tests

$$
H_0: \mu_1 = \mu_2 \quad \text{vs.} \quad H_1: \mu_1 \neq \mu_2
$$

where $\mu_1$ and $\mu_2$ denote the population means of the control and treatment groups, respectively. Under the equal-variance assumption, the test statistic is the two-sample $t$-statistic

$$
t = \frac{\bar{Y}_1 - \bar{Y}_2}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}
$$

where $s_p$ is the pooled standard deviation and $n_1, n_2$ are the group sizes. This $t$-statistic follows a $t$-distribution with $n_1 + n_2 - 2$ degrees of freedom under $H_0$.

### Multi-Group Case

When the experiment involves $k \geq 2$ treatment groups (including the control), the hypothesis test generalizes to

$$
H_0: \mu_1 = \mu_2 = \cdots = \mu_k \quad \text{vs.} \quad H_1: \text{at least one } \mu_i \text{ differs}
$$

This is precisely the one-way ANOVA hypothesis. Under the model

$$
Y_{ij} = \mu + \alpha_i + \varepsilon_{ij}, \quad \varepsilon_{ij} \overset{\text{iid}}{\sim} N(0, \sigma^2)
$$

where $i = 1, \ldots, k$ indexes groups, $j = 1, \ldots, n_i$ indexes observations within group $i$, $\mu$ is the overall mean, and $\alpha_i$ is the effect of treatment $i$ (with the constraint $\sum_{i=1}^k n_i \alpha_i = 0$ for identifiability), the test statistic is the F-ratio

$$
F = \frac{\text{MSB}}{\text{MSW}} = \frac{\text{SSB} / (k - 1)}{\text{SSW} / (N - k)}
$$

where $N = \sum_{i=1}^k n_i$ is the total sample size, SSB is the between-group sum of squares, and SSW is the within-group sum of squares. Under $H_0$, this statistic follows an $F(k-1, \, N-k)$ distribution. The null hypothesis is rejected at significance level $\alpha$ when $F > F_{\alpha, \, k-1, \, N-k}$.

!!! note "Two groups as a special case of ANOVA"
    When $k = 2$, the F-statistic equals $t^2$, where $t$ is the two-sample $t$-statistic. The one-way ANOVA with two groups and the two-sample $t$-test always produce identical $p$-values.

### Post-Hoc Comparisons

A significant F-test tells us that at least one group mean differs from the others, but it does not identify which specific pairs differ. To determine which treatments outperform the control (or each other), post-hoc pairwise comparison methods are required. These methods control the family-wise error rate (FWER) to avoid inflating the Type I error rate across multiple comparisons. Common choices include Tukey HSD for all pairwise comparisons and Dunnett's test when the goal is to compare each treatment to a single control. See [Post-Hoc Comparisons](../post_hoc/tukey.md) for a full treatment of these methods.

## Sample Size Determination

Before running an A/B test, the analyst must determine how many observations are needed per group to detect a meaningful effect. The required sample size depends on three quantities: the significance level $\alpha$, the desired power $1 - \beta$, and the minimum detectable effect size.

For the two-group case with equal group sizes, the per-group sample size required to detect an effect size $\delta = \mu_1 - \mu_2$ with power $1 - \beta$ at significance level $\alpha$ is approximately

$$
n \geq \frac{2\sigma^2 (z_{\alpha/2} + z_\beta)^2}{\delta^2}
$$

where $z_{\alpha/2}$ and $z_\beta$ are the standard normal critical values corresponding to the two-tailed significance level and the desired power, respectively, and $\sigma^2$ is the common within-group variance.

!!! example "Sample size calculation"
    Suppose a website A/B test aims to detect a difference of $\delta = 0.5$ percentage points in conversion rate (on the probability scale, $\delta = 0.005$), with baseline rate $p = 0.05$ so that $\sigma^2 \approx p(1-p) = 0.0475$, at $\alpha = 0.05$ and power $1 - \beta = 0.80$. The critical values are $z_{0.025} = 1.96$ and $z_{0.20} = 0.84$. Then

    $$
    n \geq \frac{2(0.0475)(1.96 + 0.84)^2}{0.005^2} = \frac{2(0.0475)(7.84)}{0.000025} \approx 29{,}792
    $$

    Each group requires approximately 30,000 users — a reminder that detecting small effects demands large samples.

For multi-group designs, power analysis extends to the F-test framework, and the effect size is typically expressed using Cohen's $f$, defined as

$$
f = \frac{\sigma_\alpha}{\sigma}
$$

where $\sigma_\alpha = \sqrt{\frac{1}{k}\sum_{i=1}^k \alpha_i^2}$ measures the standard deviation of the treatment effects. Conventional benchmarks are $f = 0.10$ (small), $f = 0.25$ (medium), and $f = 0.40$ (large).

## Common Pitfalls

!!! warning "Peeking and early stopping"
    A frequent mistake in A/B testing is to check results repeatedly during data collection and stop the experiment as soon as a significant result appears. This practice inflates the Type I error rate well beyond the nominal $\alpha$ because each interim look is an additional hypothesis test. Sequential testing methods (e.g., group sequential designs or always-valid $p$-values) provide principled alternatives that allow early stopping while controlling the overall error rate.

!!! warning "Multiple comparisons without correction"
    When testing $k$ groups, the number of pairwise comparisons is $\binom{k}{2}$. Performing each at level $\alpha$ without correction inflates the family-wise error rate to $1 - (1 - \alpha)^{\binom{k}{2}}$, which can be substantial. For example, with $k = 5$ groups and $\alpha = 0.05$, the FWER reaches approximately $0.40$. Always apply a multiple comparison correction such as Bonferroni or Tukey HSD.
