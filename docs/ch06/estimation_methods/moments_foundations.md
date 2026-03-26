# Method of Moments Foundations

## Motivation

Before Ronald Fisher formalized Maximum Likelihood Estimation in the 1920s, statisticians needed a systematic method for fitting parametric models to data. Karl Pearson introduced the **Method of Moments (MoM)** in 1894, establishing one of the oldest and most intuitive approaches to parameter estimation. The core idea is strikingly simple: since sample moments converge to population moments by the Law of Large Numbers, we can estimate parameters by equating theoretical moments (which depend on the unknown parameters) to their observed sample counterparts and solving the resulting equations. While MLE has largely supplanted MoM as the default estimation method, MoM remains valuable as a quick, closed-form starting point and serves as the foundation for the Generalized Method of Moments (GMM), which is the dominant estimation framework in econometrics and empirical finance.

## Population and Sample Moments

Moments are numerical summaries that characterize the shape of a probability distribution. There are two types relevant to MoM:

**Raw moments** (moments about zero). The $k$-th raw moment of a random variable $X$ is:

$$
\mu_k' = E[X^k]
$$

The first raw moment is the mean: $\mu_1' = E[X] = \mu$.

**Central moments** (moments about the mean). The $k$-th central moment is:

$$
\mu_k = E[(X - \mu)^k]
$$

The second central moment is the variance: $\mu_2 = \text{Var}(X) = \sigma^2$. The third and fourth central moments relate to skewness and kurtosis.

The corresponding **sample moments** replace the expectation with the sample average:

$$
M_k' = \frac{1}{n}\sum_{i=1}^n X_i^k \quad \text{(sample raw moment)}
$$

$$
M_k = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^k \quad \text{(sample central moment)}
$$

By the Law of Large Numbers, $M_k' \xrightarrow{p} \mu_k'$ and $M_k \xrightarrow{p} \mu_k$ as $n \to \infty$, ensuring that sample moments consistently estimate their population counterparts.

## The MoM Procedure

Suppose a distribution has $p$ unknown parameters $\theta = (\theta_1, \ldots, \theta_p)^\top$, and the first $p$ raw population moments can be written as functions of these parameters:

$$
\mu_k'(\theta) = E_\theta[X^k], \quad k = 1, \ldots, p
$$

The Method of Moments estimator $\hat{\theta}_{\text{MoM}}$ is obtained by solving the system:

$$
\mu_k'(\hat{\theta}) = M_k', \quad k = 1, \ldots, p
$$

That is, we set each population moment equal to its sample counterpart and solve for the $p$ unknown parameters.

!!! note "The MoM Algorithm"
    **Step 1.** Express the first $p$ population moments as functions of the parameters: $\mu_1'(\theta), \mu_2'(\theta), \ldots, \mu_p'(\theta)$.

    **Step 2.** Compute the first $p$ sample moments: $M_1' = \bar{X}$, $M_2' = \frac{1}{n}\sum X_i^2$, etc.

    **Step 3.** Equate: $\mu_k'(\theta) = M_k'$ for $k = 1, \ldots, p$.

    **Step 4.** Solve this system of $p$ equations in $p$ unknowns for $\hat{\theta}_{\text{MoM}}$.

??? example "MoM for the Exponential Distribution"
    For $X \sim \text{Exp}(\lambda)$, there is $p = 1$ parameter. The first population moment is:

    $$
    \mu_1' = E[X] = \frac{1}{\lambda}
    $$

    Setting $\mu_1' = M_1'$:

    $$
    \frac{1}{\lambda} = \bar{X} \implies \hat{\lambda}_{\text{MoM}} = \frac{1}{\bar{X}}
    $$

    For data $\{2.1, 0.8, 1.5, 3.2, 0.4\}$: $\bar{X} = 1.6$, so $\hat{\lambda}_{\text{MoM}} = 1/1.6 = 0.625$.

## Raw Moments versus Central Moments

The procedure above uses raw moments, but one can equivalently use central moments. For a two-parameter location-scale family, a common approach is:

- First equation: $E[X] = M_1'$ (raw first moment)
- Second equation: $\text{Var}(X) = M_2$ (central second moment)

This mixed approach is convenient because the variance formula is often simpler than the second raw moment formula (recall $\text{Var}(X) = E[X^2] - (E[X])^2$). For location-scale families like the normal, both approaches yield identical estimators. For other parameterizations, the choice can matter, and the raw-moment approach is the canonical one.

## Properties of MoM Estimators

### Consistency

The MoM estimator is consistent under mild regularity conditions. The argument proceeds in two steps:

1. By the Law of Large Numbers, each sample moment converges in probability to the corresponding population moment: $M_k' \xrightarrow{p} \mu_k'(\theta_0)$.

2. If the mapping from moments to parameters $h: (\mu_1', \ldots, \mu_p') \mapsto (\theta_1, \ldots, \theta_p)$ is continuous at the true moment values, then by the **continuous mapping theorem**:

$$
\hat{\theta}_{\text{MoM}} = h(M_1', \ldots, M_p') \xrightarrow{p} h(\mu_1'(\theta_0), \ldots, \mu_p'(\theta_0)) = \theta_0
$$

The continuity condition is satisfied whenever the system of moment equations has a unique solution that varies smoothly with the moments — a condition met by most standard distributions.

### Asymptotic Normality

Under additional smoothness conditions, the MoM estimator is asymptotically normal. By the multivariate Central Limit Theorem:

$$
\sqrt{n}\begin{pmatrix} M_1' - \mu_1' \\ \vdots \\ M_p' - \mu_p' \end{pmatrix} \xrightarrow{d} N(\mathbf{0}, \boldsymbol{\Sigma})
$$

where $\boldsymbol{\Sigma}$ is the covariance matrix of $(X, X^2, \ldots, X^p)$. Applying the delta method through the mapping $h$:

$$
\sqrt{n}(\hat{\theta}_{\text{MoM}} - \theta_0) \xrightarrow{d} N\!\left(\mathbf{0}, \nabla h \cdot \boldsymbol{\Sigma} \cdot \nabla h^\top\right)
$$

This result enables the construction of confidence intervals and hypothesis tests based on MoM estimators.

### Simplicity

MoM often yields closed-form estimators that require only basic algebra. For the exponential distribution, $\hat{\lambda} = 1/\bar{X}$. For the normal, $\hat{\mu} = \bar{X}$ and $\hat{\sigma}^2 = M_2$. No numerical optimization is needed, making MoM especially attractive as a starting point for iterative procedures like MLE when closed-form MLEs are unavailable.

### Potential Drawbacks

Despite its simplicity, MoM has several limitations that the practitioner should be aware of:

- **Not necessarily efficient.** MoM estimators generally do not achieve the Cramer-Rao lower bound. They can have substantially larger variance than MLE, especially when the distribution's shape is primarily determined by higher moments rather than the first few.

- **May produce inadmissible estimates.** Because MoM solves algebraic equations without enforcing parameter constraints, it can yield estimates outside the parameter space. For example, MoM can produce a negative variance estimate for certain distributions when the sample is small or unusual.

- **Sensitive to higher moments.** When $p$ is large, MoM requires matching higher sample moments ($M_3', M_4', \ldots$), which have high sampling variability. The estimation error in $M_k'$ grows rapidly with $k$, degrading the precision of the corresponding parameter estimates.

!!! warning "When MoM Fails"
    MoM requires that the first $p$ population moments exist and that the mapping from moments to parameters is invertible. For heavy-tailed distributions like the Cauchy (whose mean does not exist), MoM cannot be applied. In such cases, MLE or other methods must be used.

## Existence and Uniqueness

The MoM estimator exists and is unique when:

1. The first $p$ population moments exist (i.e., $E[|X|^p] < \infty$).
2. The mapping $\theta \mapsto (\mu_1'(\theta), \ldots, \mu_p'(\theta))$ is one-to-one (injective) on the parameter space.
3. The sample moments $(M_1', \ldots, M_p')$ fall within the range of this mapping.

Condition 2 is crucial: if two different parameter values produce the same first $p$ moments, the system is not identifiable through moments alone. For most standard distributions (normal, gamma, beta, Poisson, exponential), conditions 1 and 2 are satisfied, guaranteeing a well-defined MoM estimator.
