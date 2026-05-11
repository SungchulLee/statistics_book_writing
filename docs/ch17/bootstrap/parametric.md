# Parametric Bootstrap

## Motivation

The nonparametric bootstrap makes no assumptions about the population distribution $F$ and resamples directly from the data. When we have good reason to believe the data come from a specific parametric family — say, a normal, exponential, or Poisson distribution — we can exploit this knowledge. The **parametric bootstrap** fits a parametric model to the data and then generates bootstrap samples from the fitted distribution rather than from the empirical distribution.

This approach typically produces more efficient estimates (smaller standard errors, tighter confidence intervals) when the model is correct, but it carries the risk of being invalid when the model is misspecified.

## The Algorithm

Given an observed sample $x_1, \ldots, x_n$ and a parametric model $F_\psi$ indexed by parameter $\psi$:

1. **Fit** the parametric model: compute the MLE $\hat{\psi}$ from the observed data
2. **Set** the number of bootstrap replicates $B$
3. **For** $b = 1, 2, \ldots, B$:
    - Generate a bootstrap sample $x_1^*, x_2^*, \ldots, x_n^*$ by drawing $n$ observations iid from $F_{\hat{\psi}}$
    - Compute the bootstrap replicate $\hat{\theta}^{*(b)} = g(x_1^*, \ldots, x_n^*)$
4. **Use** the bootstrap distribution $\{\hat{\theta}^{*(1)}, \ldots, \hat{\theta}^{*(B)}\}$ for inference

The key difference from the nonparametric bootstrap is in Step 3: instead of resampling the observed data with replacement, we simulate fresh data from the fitted model $F_{\hat{\psi}}$.

!!! note "Parametric vs Nonparametric Resampling"
    In the nonparametric bootstrap, $x_i^* \sim \hat{F}_n$ (draw from the data). In the parametric bootstrap, $x_i^* \sim F_{\hat{\psi}}$ (draw from the fitted model). The bootstrap samples in the parametric case are genuinely new values that may never have appeared in the original data.

## Comparison with the Nonparametric Bootstrap

| Aspect | Nonparametric Bootstrap | Parametric Bootstrap |
|---|---|---|
| **Assumption** | None beyond iid | Data follow $F_\psi$ |
| **Resampling from** | $\hat{F}_n$ (observed data) | $F_{\hat{\psi}}$ (fitted model) |
| **Bootstrap values** | Subset of observed values | New simulated values |
| **Efficiency** | Lower (more variable) | Higher (when model is correct) |
| **Robustness** | High | Low (sensitive to misspecification) |

The parametric bootstrap is more efficient because $F_{\hat{\psi}}$ is a smoother estimate of $F$ than $\hat{F}_n$. Smoothness reduces the variance of the bootstrap approximation.

## When to Use the Parametric Bootstrap

The parametric bootstrap is appropriate when:

- **The parametric model is well-justified** by theory or extensive prior analysis
- **Goodness-of-fit tests** (Shapiro-Wilk, Anderson-Darling, Q-Q plots) support the model
- **The statistic of interest involves the model parameters** directly, such as MLEs or likelihood ratio statistics
- **The sample size is small**, where the empirical distribution is a poor approximation to $F$ but a parametric model may still capture the essential structure

!!! warning "Model Misspecification"
    If the assumed parametric model $F_\psi$ is wrong, the parametric bootstrap can produce misleading results. Confidence intervals may have incorrect coverage, and hypothesis tests may have inflated or deflated Type I error rates. When in doubt, use the nonparametric bootstrap or compare results from both approaches.

## Example: Normal Model

Suppose $x_1, \ldots, x_n$ appear to come from a normal distribution, and we want the standard error of the sample variance $s^2$.

**Parametric bootstrap procedure:**

1. Compute $\hat{\mu} = \bar{x}$ and $\hat{\sigma}^2 = s^2$ from the data
2. For $b = 1, \ldots, B$: generate $x_1^*, \ldots, x_n^* \overset{\text{iid}}{\sim} N(\hat{\mu}, \hat{\sigma}^2)$ and compute $s^{2*(b)}$
3. The bootstrap standard error is $\widehat{\text{SE}}_{\text{boot}} = \text{sd}(s^{2*(1)}, \ldots, s^{2*(B)})$

Under the normal model, the exact standard error of $s^2$ is $\sigma^2\sqrt{2/(n-1)}$. The parametric bootstrap should closely approximate this known result, providing a useful check on the procedure.

## Example: Exponential Model

Suppose we model waiting times as $x_1, \ldots, x_n \overset{\text{iid}}{\sim} \text{Exp}(\lambda)$ and want a confidence interval for the mean $\mu = 1/\lambda$.

**Parametric bootstrap procedure:**

1. Compute the MLE $\hat{\lambda} = 1/\bar{x}$
2. For $b = 1, \ldots, B$: generate $x_1^*, \ldots, x_n^* \overset{\text{iid}}{\sim} \text{Exp}(\hat{\lambda})$ and compute $\bar{x}^{*(b)}$
3. Use the quantiles of $\{\bar{x}^{*(1)}, \ldots, \bar{x}^{*(B)}\}$ to form a confidence interval for $\mu$

Because the exponential distribution is right-skewed, the sampling distribution of $\bar{x}$ is also skewed for moderate $n$. The parametric bootstrap captures this skewness, producing asymmetric confidence intervals that respect the positive support of $\mu$.

## Parametric Bootstrap for Likelihood Ratio Tests

A particularly powerful application is testing nested parametric models. Consider testing $H_0: \psi \in \Psi_0$ against $H_1: \psi \in \Psi$ where $\Psi_0 \subset \Psi$.

The observed likelihood ratio statistic is:

$$
\Lambda_{\text{obs}} = 2\left[\ell(\hat{\psi}) - \ell(\hat{\psi}_0)\right]
$$

where $\hat{\psi}$ and $\hat{\psi}_0$ are the MLEs under $H_1$ and $H_0$, respectively.

**Parametric bootstrap $p$-value:**

1. Fit the model under $H_0$: obtain $\hat{\psi}_0$
2. For $b = 1, \ldots, B$: generate data from $F_{\hat{\psi}_0}$, compute $\Lambda^{*(b)}$
3. The $p$-value is the proportion of $\Lambda^{*(b)} \ge \Lambda_{\text{obs}}$

This approach avoids relying on the $\chi^2$ approximation for $\Lambda$, which can be inaccurate for small samples or when the models are close to the boundary of the parameter space.

!!! tip "Practical Recommendation"
    When feasible, run both the parametric and nonparametric bootstrap and compare the results. If they agree closely, the parametric model assumption is likely reasonable and the parametric bootstrap estimates are preferred for their greater precision. If they disagree, investigate the model assumption before trusting either result.

## Limitations

The parametric bootstrap inherits the limitations of the assumed model:

- **Model selection uncertainty** is not accounted for — the bootstrap conditions on the chosen model
- **Boundary parameters** (e.g., testing $\sigma^2 = 0$) can cause bootstrap failure because the fitted model degenerates
- **Multivariate models** require correct specification of the joint distribution, not just the marginals

For complex models with many parameters, the nonparametric bootstrap is often safer and only marginally less efficient.

## Summary

The parametric bootstrap replaces the empirical distribution $\hat{F}_n$ with a fitted parametric distribution $F_{\hat{\psi}}$ as the resampling mechanism. When the parametric model is correct, this yields more efficient inference — tighter confidence intervals and more powerful tests. The tradeoff is sensitivity to model misspecification: if the assumed distribution is wrong, parametric bootstrap results can be misleading. A practical strategy is to compare parametric and nonparametric bootstrap results as a diagnostic for model adequacy.


## Exercises

**Exercise 1.**
Describe the main concept of Parametric Bootstrap and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Parametric Bootstrap is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

---

**Exercise 2.**
State the key assumptions required by the method discussed here. How can each assumption be checked?

??? success "Solution to Exercise 2"
    The main assumptions typically include: (1) independence of observations -- verified by understanding the data collection process and checking for serial correlation; (2) distributional requirements (e.g., normality) -- checked with Q-Q plots and formal tests like Shapiro-Wilk; (3) equal variances (if applicable) -- assessed with boxplots and Levene's test. When assumptions are violated, consider robust alternatives, transformations, or nonparametric methods.

---

**Exercise 3.**
Work through a small numerical example illustrating the application of the technique from this section.

??? success "Solution to Exercise 3"
    A structured approach to applying this technique involves: (1) clearly stating the hypotheses or estimation goal; (2) verifying that the data meet the required assumptions; (3) computing the relevant test statistic, estimate, or model fit; (4) obtaining the p-value, confidence interval, or posterior distribution; (5) interpreting the result in the context of the original question. Following these steps systematically ensures a rigorous and reproducible analysis.

---

**Exercise 4.**
Compare the approach from this section with an alternative method. When would you choose each?

??? success "Solution to Exercise 4"
    The method discussed here is appropriate when its assumptions hold and the sample size is sufficient for the asymptotic approximations to be accurate. Alternative approaches include: (1) nonparametric methods -- preferred when distributional assumptions are suspect; (2) bootstrap methods -- useful when analytical reference distributions are unavailable; (3) Bayesian methods -- valuable when incorporating prior information or when direct probability statements about parameters are desired. Running multiple approaches and comparing results provides a useful robustness check.
