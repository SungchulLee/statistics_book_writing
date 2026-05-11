# Feature Selection via Penalized Likelihood

## Why Feature Selection Matters

Including irrelevant predictors in a logistic regression model increases
variance without reducing bias, degrades interpretability, and can harm
predictive performance.  **Feature selection** identifies the subset of
predictors that contribute meaningfully to the model.  The L1 (lasso) penalty
provides an automated, principled approach: as the penalty strength increases,
coefficient estimates are driven to exactly zero, effectively removing features
from the model.

## The L1 Regularization Path

Recall from the [regularization page](regularization.md) that the L1-penalized
log-likelihood is

$$
\ell_{\text{lasso}}(\boldsymbol{\theta})
= \ell(\boldsymbol{\theta})

  - \lambda\sum_{j=1}^{p-1}|\theta_j|
$$

As $\lambda$ varies from large to small, each coefficient traces a path from
zero to its unpenalized MLE value.  This **regularization path** is piecewise
linear for the lasso.

### Key properties of the L1 path

1. **At large $\lambda$:** All coefficients are zero (null model).
2. **As $\lambda$ decreases:** Coefficients enter the model one at a time
   at specific threshold values $\lambda_j^*$.
3. **At $\lambda = 0$:** All coefficients equal the unpenalized MLE
   (assuming it exists).

The order in which features enter the path provides a natural **ranking** of
predictor importance.  Features that enter early (at large $\lambda$) have the
strongest marginal association with the response.

## Selecting the Penalty Strength

The regularization path shows which features survive at each $\lambda$, but
we still need to choose a specific $\lambda$.  Cross-validation is the standard
approach.

### Procedure

1. Define a grid $\lambda_1 > \lambda_2 > \cdots > \lambda_m$.
2. For each $\lambda_k$, compute the $K$-fold cross-validated log-likelihood
   (or equivalently, the cross-validated deviance):

$$
\text{CV}(\lambda_k) = \frac{1}{K}\sum_{k=1}^{K}D^{(-k)}(\lambda_k)
$$

where $D^{(-k)}$ is the deviance on fold $k$ using the model trained on the
remaining $K-1$ folds.

3. Select $\hat{\lambda} = \arg\min_{\lambda_k}\text{CV}(\lambda_k)$.

!!! tip "One-Standard-Error Rule"
    A common conservative choice is $\hat{\lambda}_{\text{1se}}$: the largest
    $\lambda$ whose CV score is within one standard error of the minimum.
    This produces a sparser model with nearly the same predictive performance.

## Stability Selection

A single L1 path can be sensitive to the specific training sample: small
perturbations may cause different features to be selected.  **Stability
selection** (Meinshausen and Buhlmann, 2010) addresses this instability.

### Algorithm

1. For $b = 1, \ldots, B$ (e.g., $B = 100$):
    - Draw a random subsample of size $\lfloor n/2 \rfloor$ without
      replacement.
    - Fit the L1 logistic regression across a grid of $\lambda$ values.
    - Record which features have non-zero coefficients at each $\lambda$.
2. For each feature $j$, compute the **selection probability**:

$$
\hat{\Pi}_j(\lambda) = \frac{1}{B}\sum_{b=1}^{B}\mathbf{1}\{\hat{\theta}_j^{(b)}(\lambda) \neq 0\}
$$

3. Select feature $j$ if $\max_\lambda \hat{\Pi}_j(\lambda) \ge \pi_{\text{thr}}$,
   where a typical threshold is $\pi_{\text{thr}} = 0.6$ to $0.9$.

### Advantages

- Controls the expected number of **false selections** (variables incorrectly
  included).
- Robust to the choice of $\lambda$ — the selection probability is aggregated
  over the entire path.
- Works well in high-dimensional settings ($p \gg n$).

## Comparison with Stepwise Methods

Traditional stepwise procedures (forward selection, backward elimination) have
been widely used for feature selection but have important limitations compared
to the L1 approach.

| Criterion | L1 (Lasso) | Forward Stepwise | Backward Stepwise |
|---|---|---|---|
| Objective | Penalized likelihood | Sequential testing (AIC, BIC, or $p$-values) | Sequential testing |
| Search strategy | Continuous shrinkage path | Greedy, one-at-a-time addition | Greedy, one-at-a-time removal |
| Handles $p > n$ | Yes | Stops at $n$ features | Cannot start (requires $n > p$) |
| Coefficient shrinkage | Yes (toward zero) | No (unpenalized MLE) | No (unpenalized MLE) |
| Selection stability | Moderate (improved by stability selection) | Low | Low |
| Computational cost | Single path via coordinate descent | $O(p^2)$ model fits | $O(p^2)$ model fits |

!!! warning "Pitfalls of Stepwise Selection"
    Stepwise procedures inflate Type I error rates because they perform
    multiple implicit hypothesis tests without proper correction.  The
    $p$-values from a stepwise-selected model are generally too small, and
    confidence intervals are too narrow.  Penalized methods avoid this by
    treating selection and estimation as a single optimization problem.

??? example "Worked Example: L1 Path for Feature Selection"
    Consider a logistic regression with $n = 200$ observations and $p = 10$
    predictors, of which only $x_1$, $x_3$, and $x_7$ are truly associated
    with the response.

    | $\log(\lambda)$ | Non-zero coefficients | CV deviance |
    |---|---|---|
    | 2.0 | none | 277 |
    | 1.0 | $x_1$ | 245 |
    | 0.5 | $x_1, x_3$ | 218 |
    | 0.0 | $x_1, x_3, x_7$ | 195 |
    | -0.5 | $x_1, x_3, x_5, x_7$ | 194 |
    | -1.0 | $x_1, x_2, x_3, x_5, x_7, x_8$ | 198 |

    The CV deviance is minimized near $\log(\lambda) = -0.5$, but the
    one-standard-error rule selects $\log(\lambda) = 0.0$, recovering the
    three true predictors without the noise variables.


## Exercises

**Exercise 1.**
Describe the main concept of Feature Selection via Penalized Likelihood and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Feature Selection via Penalized Likelihood is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

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
