# Generalized Method of Moments

## From MoM to GMM

The Method of Moments works by equating $p$ population moments to their sample counterparts and solving for $p$ unknown parameters. In many applications, however, economic theory, financial models, or physical constraints provide **more moment conditions than parameters**. For example, a model with 2 parameters might generate 5 testable moment conditions. The standard MoM cannot use all 5 simultaneously — it would pick 2 and discard the rest, wasting potentially valuable information. The Generalized Method of Moments (GMM) solves this problem by combining all available moment conditions through a weighted quadratic objective, extracting maximum information from the data without requiring the full distributional specification that MLE demands.

## Moment Conditions and Overidentification

The starting point for GMM is a set of **moment conditions**. Let $\theta \in \mathbb{R}^p$ be the parameter vector of interest, and suppose that for the true parameter value $\theta_0$, we have:

$$
E[g(X, \theta_0)] = \mathbf{0}
$$

where $g : \mathbb{R}^d \times \mathbb{R}^p \to \mathbb{R}^r$ maps each observation and parameter value to an $r$-dimensional vector. Each component of $g$ represents one moment condition.

Three cases arise depending on the relationship between $r$ (number of moment conditions) and $p$ (number of parameters):

- **Exactly identified** ($r = p$): The system has the same number of equations as unknowns. The sample analogue $\bar{g}_n(\theta) = \mathbf{0}$ can typically be solved exactly, recovering standard MoM.
- **Overidentified** ($r > p$): More equations than unknowns. No $\theta$ can set all sample moment conditions to exactly zero, so we minimize a weighted sum of squared violations.
- **Underidentified** ($r < p$): Fewer equations than unknowns. The parameters are not point-identified — additional information is needed.

GMM is designed for the overidentified case, though it reduces to MoM when $r = p$.

## The GMM Objective

Define the sample moment vector:

$$
\bar{g}_n(\theta) = \frac{1}{n} \sum_{i=1}^{n} g(X_i, \theta)
$$

The GMM estimator minimizes the quadratic form:

$$
\hat{\theta}_{\text{GMM}} = \arg\min_{\theta} \; \bar{g}_n(\theta)^\top \mathbf{W} \, \bar{g}_n(\theta)
$$

where $\mathbf{W} \in \mathbb{R}^{r \times r}$ is a positive-definite **weighting matrix** that determines how the different moment conditions are weighted in the objective. Different choices of $\mathbf{W}$ yield consistent estimators, but the choice affects efficiency.

!!! note "Intuition for the GMM Objective"
    Think of $\bar{g}_n(\theta)$ as a vector of "residuals" — each component measures how badly a particular moment condition is violated at parameter value $\theta$. The GMM estimator finds the $\theta$ that makes these residuals as small as possible in a weighted least-squares sense. The weighting matrix $\mathbf{W}$ determines the relative importance of each residual.

## Optimal Weighting Matrix

Any positive-definite $\mathbf{W}$ produces a consistent GMM estimator, but efficiency depends on the choice of $\mathbf{W}$. The optimal weighting matrix is the inverse of the asymptotic covariance of the moment conditions:

$$
\mathbf{W}_{\text{opt}} = \mathbf{S}^{-1}
$$

where:

$$
\mathbf{S} = \text{Var}\!\left(\sqrt{n}\,\bar{g}_n(\theta_0)\right) = E\!\left[g(X, \theta_0)\,g(X, \theta_0)^\top\right]
$$

The second equality holds when the observations are i.i.d. This choice minimizes the asymptotic variance of the GMM estimator in the matrix (Loewner) ordering.

!!! tip "Why the Inverse Covariance?"
    Moment conditions with high variance are less informative — they fluctuate more from sample to sample. By weighting by $\mathbf{S}^{-1}$, the optimal GMM estimator downweights noisy moment conditions and upweights precise ones, analogous to weighted least squares in regression.

## Two-Step GMM

In practice, $\mathbf{S}$ depends on the unknown $\theta_0$, so the optimal weighting matrix cannot be computed directly. The standard solution is **two-step GMM**:

1. **Step 1**: Choose an initial weighting matrix (typically $\mathbf{W}_1 = \mathbf{I}_r$, the identity matrix). Minimize the GMM objective to obtain a preliminary consistent estimate $\hat{\theta}_1$.

2. **Step 2**: Using $\hat{\theta}_1$, estimate the optimal weighting matrix:

$$
\hat{\mathbf{S}} = \frac{1}{n} \sum_{i=1}^{n} g(X_i, \hat{\theta}_1)\,g(X_i, \hat{\theta}_1)^\top
$$

Set $\mathbf{W}_2 = \hat{\mathbf{S}}^{-1}$ and re-minimize the GMM objective to obtain the efficient estimate $\hat{\theta}_2$.

The two-step estimator $\hat{\theta}_2$ achieves the same asymptotic efficiency as the infeasible estimator that uses the true $\mathbf{S}^{-1}$.

## Asymptotic Properties

Under regularity conditions, the efficient (optimal-weight) GMM estimator satisfies:

$$
\sqrt{n}\left(\hat{\theta}_{\text{GMM}} - \theta_0\right) \xrightarrow{d} N\!\left(\mathbf{0}, \left(\mathbf{G}^\top \mathbf{S}^{-1} \mathbf{G}\right)^{-1}\right)
$$

where:

$$
\mathbf{G} = E\!\left[\frac{\partial g(X, \theta_0)}{\partial \theta^\top}\right] \in \mathbb{R}^{r \times p}
$$

is the Jacobian of the moment conditions. The asymptotic covariance matrix $(\mathbf{G}^\top \mathbf{S}^{-1} \mathbf{G})^{-1}$ is estimated by replacing $\mathbf{G}$ and $\mathbf{S}$ with their sample analogues evaluated at $\hat{\theta}_{\text{GMM}}$.

## Hansen's J-Test for Overidentifying Restrictions

When the model is overidentified ($r > p$), not all moment conditions can be satisfied simultaneously even at the GMM estimate. This provides a natural specification test: if the model is correctly specified, the minimized objective should be small. Hansen's J-statistic is:

$$
J = n \, \bar{g}_n(\hat{\theta}_{\text{GMM}})^\top \hat{\mathbf{S}}^{-1} \bar{g}_n(\hat{\theta}_{\text{GMM}})
$$

Under $H_0$ (model correctly specified):

$$
J \xrightarrow{d} \chi^2(r - p)
$$

A large $J$ (relative to $\chi^2(r-p)$ critical values) suggests that the moment conditions are mutually inconsistent, indicating model misspecification.

## Worked Example: Normal Distribution with Three Moments

Suppose $X_1, \ldots, X_n \stackrel{\text{i.i.d.}}{\sim} N(\mu, \sigma^2)$ and we estimate $\theta = (\mu, \sigma^2)^\top$ using three moment conditions ($r = 3 > p = 2$):

$$
g(X, \theta) = \begin{pmatrix} X - \mu \\ X^2 - (\mu^2 + \sigma^2) \\ X^3 - \mu^3 - 3\mu\sigma^2 \end{pmatrix}
$$

These correspond to equating the first three population moments to their sample counterparts. With $p = 2$ parameters and $r = 3$ conditions, the model is overidentified by 1.

**Step 1** (with $\mathbf{W} = \mathbf{I}_3$): Minimize $\bar{g}_n(\theta)^\top \bar{g}_n(\theta)$ to get $\hat{\theta}_1 = (\hat{\mu}_1, \hat{\sigma}^2_1)^\top$.

**Step 2**: Estimate $\hat{\mathbf{S}}$ from the residuals at $\hat{\theta}_1$, set $\mathbf{W}_2 = \hat{\mathbf{S}}^{-1}$, and re-minimize.

The third moment condition provides an additional equation that constrains the estimator. For symmetric distributions like the normal, the third central moment is zero, so the third condition is essentially $E[X^3] = \mu^3 + 3\mu\sigma^2$. The J-test checks whether this symmetry constraint is consistent with the data — a significant J-statistic would suggest the data are not normally distributed.

??? example "GMM Reduces to MoM When Exactly Identified"
    When $r = p$, the moment conditions $\bar{g}_n(\theta) = \mathbf{0}$ can typically be solved exactly, and the weighting matrix $\mathbf{W}$ becomes irrelevant (any positive-definite $\mathbf{W}$ yields the same solution). In this case, GMM reduces to standard MoM. For instance, estimating $(\mu, \sigma^2)$ with the two conditions $E[X - \mu] = 0$ and $E[(X - \mu)^2 - \sigma^2] = 0$ gives $\hat{\mu} = \bar{X}$ and $\hat{\sigma}^2 = \frac{1}{n}\sum(X_i - \bar{X})^2$, regardless of $\mathbf{W}$.

## Connection to Other Estimation Methods

GMM occupies a central position in the estimation landscape:

- **MoM** is GMM with $r = p$ (exactly identified).
- **MLE** can be viewed as a special case of GMM where the moment conditions are the score equations $g(X, \theta) = \frac{\partial}{\partial \theta}\log f(X \mid \theta)$. When the model is correctly specified, MLE is efficient among all GMM estimators that use moment conditions derived from the likelihood.
- **Instrumental Variables (IV)** regression is a special case of GMM where the moment conditions take the form $E[Z_i(Y_i - X_i^\top \beta)] = \mathbf{0}$, with instruments $Z_i$.

!!! warning "When to Use GMM"
    GMM is most valuable when: (1) the full likelihood is unknown or intractable, but moment conditions are available from theory; (2) the model is overidentified, providing a testable restriction via the J-test; (3) robustness to distributional misspecification is desired. If the full likelihood is known and tractable, MLE is generally preferred for its higher efficiency.
