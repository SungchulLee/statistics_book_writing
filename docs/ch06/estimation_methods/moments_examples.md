# MoM for Common Distributions

The Method of Moments provides a systematic way to estimate parameters for any distribution whose moments can be expressed in terms of those parameters. By equating the first $p$ population moments to their sample counterparts and solving the resulting system of equations, we obtain closed-form estimators that are intuitive and easy to compute. This page derives MoM estimators for three important distribution families — normal, gamma, and beta — illustrating the general procedure in increasing order of algebraic complexity.

Throughout this page, we define the sample moments as:

$$
M_1 = \bar{X} = \frac{1}{n}\sum_{i=1}^n X_i, \quad M_2 = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2
$$

Note that $M_2$ uses the $1/n$ divisor (not $1/(n-1)$), because MoM matches **population** moments to their **sample analogues**, and the population second central moment $E[(X - \mu)^2] = \sigma^2$ corresponds to $M_2$, not to the Bessel-corrected sample variance $S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$.

---

## Normal Distribution

The normal distribution is the simplest case for MoM because its parameters **are** the first two moments. For $X \sim N(\mu, \sigma^2)$, the population moments are:

$$
E[X] = \mu, \quad E[(X - \mu)^2] = \sigma^2
$$

**Setting up the moment equations.** Equate the population moments to their sample counterparts:

$$
\mu = M_1 = \bar{X}
$$

$$
\sigma^2 = M_2 = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2
$$

**Solving.** The system is already solved — each equation directly gives an estimator:

$$
\hat{\mu}_{\text{MoM}} = \bar{X}, \quad \hat{\sigma}^2_{\text{MoM}} = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2
$$

!!! note "MoM vs MLE for the Normal"
    The MoM estimators for the normal distribution are identical to the MLEs. Both yield $\hat{\mu} = \bar{X}$ and $\hat{\sigma}^2 = \frac{1}{n}\sum(X_i - \bar{X})^2$. The MoM variance estimator is biased (its expectation is $\frac{n-1}{n}\sigma^2$), but consistent as $n \to \infty$.

??? example "Numerical Example"
    Suppose we observe $n = 5$ values: $2, 4, 6, 8, 10$.

    $$
    \hat{\mu}_{\text{MoM}} = \bar{X} = \frac{2 + 4 + 6 + 8 + 10}{5} = 6
    $$

    $$
    \hat{\sigma}^2_{\text{MoM}} = \frac{(2-6)^2 + (4-6)^2 + (6-6)^2 + (8-6)^2 + (10-6)^2}{5} = \frac{16 + 4 + 0 + 4 + 16}{5} = 8
    $$

    The MoM estimates are $\hat{\mu} = 6$ and $\hat{\sigma}^2 = 8$ (compared to the Bessel-corrected $S^2 = 10$).

---

## Gamma Distribution

The gamma distribution is widely used to model positive, right-skewed data such as waiting times, rainfall amounts, and insurance claim sizes. Its MoM estimation requires solving a two-equation system, making it a more instructive example than the normal case.

We use the **shape-scale parameterization**: $X \sim \text{Gamma}(\alpha, \beta)$ where $\alpha > 0$ is the shape parameter and $\beta > 0$ is the scale parameter. The population moments are:

$$
E[X] = \alpha\beta, \quad \text{Var}(X) = \alpha\beta^2
$$

**Setting up the moment equations.** Equate population moments to sample moments:

$$
\alpha\beta = M_1 = \bar{X}
$$

$$
\alpha\beta^2 = M_2 = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2
$$

**Solving.** Divide the second equation by the first to isolate $\beta$:

$$
\frac{\alpha\beta^2}{\alpha\beta} = \frac{M_2}{M_1} \implies \beta = \frac{M_2}{M_1} = \frac{M_2}{\bar{X}}
$$

Substitute back into the first equation to find $\alpha$:

$$
\alpha = \frac{M_1}{\beta} = \frac{\bar{X}}{M_2/\bar{X}} = \frac{\bar{X}^2}{M_2}
$$

The MoM estimators are:

$$
\hat{\beta}_{\text{MoM}} = \frac{M_2}{\bar{X}}, \quad \hat{\alpha}_{\text{MoM}} = \frac{\bar{X}^2}{M_2}
$$

!!! tip "Interpreting the Estimators"
    The shape estimate $\hat{\alpha}$ is the squared coefficient of variation inverted: $\hat{\alpha} = (\bar{X}/\sqrt{M_2})^2 = 1/\hat{CV}^2$. Distributions with larger shape parameters are less variable relative to their mean.

??? example "Numerical Example"
    Suppose we observe waiting times (in minutes): $3.2, 5.1, 4.7, 6.3, 2.8, 4.9$.

    $$
    \bar{X} = \frac{3.2 + 5.1 + 4.7 + 6.3 + 2.8 + 4.9}{6} = \frac{27.0}{6} = 4.5
    $$

    $$
    M_2 = \frac{(3.2-4.5)^2 + (5.1-4.5)^2 + (4.7-4.5)^2 + (6.3-4.5)^2 + (2.8-4.5)^2 + (4.9-4.5)^2}{6}
    $$

    $$
    = \frac{1.69 + 0.36 + 0.04 + 3.24 + 2.89 + 0.16}{6} = \frac{8.38}{6} \approx 1.397
    $$

    $$
    \hat{\beta}_{\text{MoM}} = \frac{1.397}{4.5} \approx 0.310, \quad \hat{\alpha}_{\text{MoM}} = \frac{4.5^2}{1.397} = \frac{20.25}{1.397} \approx 14.50
    $$

    The estimates suggest a gamma distribution with shape $\hat{\alpha} \approx 14.5$ and scale $\hat{\beta} \approx 0.31$.

---

## Beta Distribution

The beta distribution models data on the interval $(0, 1)$, making it useful for proportions, rates, and Bayesian prior specification. Its MoM estimation involves a slightly more intricate algebraic manipulation.

For $X \sim \text{Beta}(a, b)$ with $a, b > 0$, the population moments are:

$$
E[X] = \frac{a}{a + b}, \quad \text{Var}(X) = \frac{ab}{(a+b)^2(a+b+1)}
$$

**Setting up the moment equations.** Let $\mu = E[X]$ and $\sigma^2 = \text{Var}(X)$. Equate to sample moments:

$$
\frac{a}{a+b} = M_1 = \bar{X}
$$

$$
\frac{ab}{(a+b)^2(a+b+1)} = M_2
$$

**Solving.** From the first equation, $a = \bar{X}(a + b)$, so $b = a(1 - \bar{X})/\bar{X}$ and $a + b = a/\bar{X}$. Define $s = a + b$ for convenience. Then $a = s\bar{X}$ and $b = s(1 - \bar{X})$.

Substituting into the variance equation:

$$
\frac{s\bar{X} \cdot s(1 - \bar{X})}{s^2(s + 1)} = M_2 \implies \frac{\bar{X}(1 - \bar{X})}{s + 1} = M_2
$$

Solving for $s$:

$$
s + 1 = \frac{\bar{X}(1 - \bar{X})}{M_2} \implies s = \frac{\bar{X}(1 - \bar{X})}{M_2} - 1
$$

Since $a = s\bar{X}$ and $b = s(1 - \bar{X})$:

$$
\hat{a}_{\text{MoM}} = \bar{X}\left(\frac{\bar{X}(1 - \bar{X})}{M_2} - 1\right)
$$

$$
\hat{b}_{\text{MoM}} = (1 - \bar{X})\left(\frac{\bar{X}(1 - \bar{X})}{M_2} - 1\right)
$$

!!! warning "Validity Condition"
    The MoM estimators for the beta distribution require $M_2 < \bar{X}(1 - \bar{X})$, which ensures $s > 0$ and hence $\hat{a}, \hat{b} > 0$. If the sample variance exceeds $\bar{X}(1 - \bar{X})$, the MoM procedure fails — the data are more dispersed than any beta distribution with these mean parameters can accommodate.

??? example "Numerical Example"
    Suppose we observe proportions: $0.3, 0.5, 0.4, 0.6, 0.35, 0.55$.

    $$
    \bar{X} = \frac{0.3 + 0.5 + 0.4 + 0.6 + 0.35 + 0.55}{6} = \frac{2.70}{6} = 0.45
    $$

    $$
    M_2 = \frac{(0.3-0.45)^2 + \cdots + (0.55-0.45)^2}{6} = \frac{0.0225 + 0.0025 + 0.0025 + 0.0225 + 0.01 + 0.01}{6} = \frac{0.07}{6} \approx 0.01167
    $$

    Check validity: $\bar{X}(1-\bar{X}) = 0.45 \times 0.55 = 0.2475 > 0.01167$ (satisfied).

    $$
    s = \frac{0.2475}{0.01167} - 1 = 21.21 - 1 = 20.21
    $$

    $$
    \hat{a}_{\text{MoM}} = 0.45 \times 20.21 \approx 9.09, \quad \hat{b}_{\text{MoM}} = 0.55 \times 20.21 \approx 11.12
    $$

    The MoM estimates suggest a $\text{Beta}(9.1, 11.1)$ distribution, which is unimodal and slightly left-skewed (mean 0.45).

## Exercises

**Exercise 1.**
Gamma claims data: 2.1, 0.8, 3.5, 1.2, 5.7, 0.4, 2.8, 1.9, 4.3, 0.6. (a) MoM estimates of $\alpha, \beta$. (b) MLE via scipy. (c) Visual comparison.

??? success "Solution to Exercise 1"
    (a) Sample moments: $\bar x = 2.33$, $m_2 \approx 2.77$.

    Gamma$(\alpha, \beta)$ with $\mathbb{E}[X] = \alpha\beta$, $\mathrm{Var}(X) = \alpha\beta^2$:

    $\hat\beta = m_2/\bar x \approx 1.19$, $\hat\alpha = \bar x^2/m_2 \approx 1.96$.

    (b) MLE via `scipy.stats.gamma.fit(x, floc=0)`: typical output $\hat\alpha \approx 1.69$, $\hat\beta \approx 1.38$.

    (c) Plot histogram + overlaid densities. With $n = 10$, both estimators have large variance and the curves are similar but not identical. Distinguishing requires more data.

---

**Exercise 2.**
**MoM for Uniform$(a, b)$.** Derive the MoM estimators using $\mathbb{E}[X], \mathbb{E}[X^2]$.

??? success "Solution to Exercise 2"
    Population moments: $\mathbb{E}[X] = (a+b)/2$. $\mathrm{Var}(X) = (b-a)^2/12$. So $\mathbb{E}[X^2] = (a+b)^2/4 + (b-a)^2/12$.

    Set sample equal to population: $\bar X = (a+b)/2$ and $m_2 = (b-a)^2/12$ where $m_2 = $ sample variance (using $n$).

    From the second: $b - a = \sqrt{12 m_2}$. Combined with $a + b = 2\bar X$:

    $\hat a_{\text{MoM}} = \bar X - \sqrt{3 m_2}$, $\hat b_{\text{MoM}} = \bar X + \sqrt{3 m_2}$.

    **Issue:** if any observation falls outside $[\hat a, \hat b]$, the estimate is impossible. MLE handles this with order statistics: $\hat a_{\text{MLE}} = \min X_i$, $\hat b_{\text{MLE}} = \max X_i$ — guaranteed feasible.

    Demonstrates that MoM can fail boundary constraints that MLE respects.

---

**Exercise 3.**
**MoM for Beta$(\alpha, \beta)$.** Derive from $\mathbb{E}[X], \mathrm{Var}(X)$.

??? success "Solution to Exercise 3"
    Beta$(\alpha, \beta)$: $\mathbb{E}[X] = \alpha/(\alpha+\beta)$, $\mathrm{Var}(X) = \alpha\beta/[(\alpha+\beta)^2(\alpha+\beta+1)]$.

    Let $m = \bar X$, $v = m_2$ (sample variance). Solve:

    $\alpha + \beta = m(1 - m)/v - 1$. Then $\hat\alpha = m \cdot (m(1-m)/v - 1)$, $\hat\beta = (1-m)(m(1-m)/v - 1)$.

    **Validity:** requires $v < m(1-m)$ (the variance of a Beta is bounded above by the variance of a Bernoulli with the same mean). If sample $v > m(1-m)$, MoM fails (no Beta fits). This rules out using MoM on data with extreme overdispersion.

---

**Exercise 4.**
**MoM for log-normal.** Given $X = e^Y$ where $Y \sim N(\mu, \sigma^2)$, derive MoM for $\mu, \sigma$.

??? success "Solution to Exercise 4"
    $\mathbb{E}[X] = e^{\mu + \sigma^2/2}$, $\mathrm{Var}(X) = (e^{\sigma^2} - 1) e^{2\mu + \sigma^2}$.

    Compute squared coefficient of variation: $\mathrm{CV}^2 = \mathrm{Var}(X)/(\mathbb{E}[X])^2 = e^{\sigma^2} - 1$.

    So $\hat\sigma^2_{\text{MoM}} = \ln(1 + \mathrm{CV}^2_{\text{sample}}) = \ln(1 + s^2/\bar X^2)$.

    $\hat\mu_{\text{MoM}} = \ln \bar X - \hat\sigma^2_{\text{MoM}}/2$.

    **Easier alternative (MLE-style):** transform first: $Y_i = \ln X_i$ are i.i.d. $N(\mu, \sigma^2)$. Standard $\hat\mu = \bar Y$, $\hat\sigma^2 = s_Y^2$. Cleaner and more efficient than MoM. MoM on the original scale forgoes the convenient log-transformation.

---

**Exercise 5.**
**Why MoM can be inefficient.** Compare MoM and MLE for Uniform$(0, \theta)$.

??? success "Solution to Exercise 5"
    MoM: $\bar X = \theta/2 \Rightarrow \hat\theta_{\text{MoM}} = 2\bar X$. $\mathrm{Var}(\hat\theta_{\text{MoM}}) = 4 \mathrm{Var}(\bar X) = 4 \theta^2/(12n) = \theta^2/(3n)$.

    MLE: $\hat\theta_{\text{MLE}} = X_{(n)}$. Bias-corrected $((n+1)/n) X_{(n)}$ has $\mathrm{Var} = \theta^2/[n(n+2)]$.

    Ratio: $\mathrm{Var}(\hat\theta_{\text{MoM}})/\mathrm{Var}(\hat\theta_{\text{MLE}}) = (n+2)/3 \to \infty$.

    MoM is *infinitely* worse than MLE as $n \to \infty$ for this problem. The sample mean discards information about the maximum; the MLE exploits it.

    **General lesson:** MoM is naive and uses only low-order moments. MLE uses the full likelihood and can be far more efficient when the distribution has "structure" beyond moments — particularly distributions with bounded support (uniform endpoints) or heavy tails.

    MoM remains useful when MLE is intractable or as a starting point for iterative MLE optimization.

---

**Exercise 6.**
**Generalized Method of Moments (GMM).** When you have more moment conditions than parameters, propose a weighting scheme to combine them.

??? success "Solution to Exercise 6"
    Suppose $\theta \in \mathbb{R}^k$ and $m \ge k$ moment conditions $\mathbb{E}[g_j(X; \theta)] = 0$ for $j = 1, \ldots, m$.

    **GMM estimator:**

    $$
    \hat\theta = \arg\min_\theta \left[\sum_i \mathbf g(X_i; \theta)\right]^T W \left[\sum_i \mathbf g(X_i; \theta)\right]
    $$

    where $W$ is a positive-definite $m \times m$ weight matrix.

    **Optimal $W$:** $W^* = \Omega^{-1}$, the inverse covariance matrix of the moment conditions. This minimizes asymptotic variance.

    **Two-step GMM:**

    1. Use $W = I$ (identity) for initial $\hat\theta^{(1)}$.
    2. Estimate $\Omega$ at $\hat\theta^{(1)}$, set $W = \hat\Omega^{-1}$.
    3. Reoptimize.

    GMM is the foundation of empirical economics (Hansen's 1982 paper won the Nobel) and underlies instrumental-variables estimation when more instruments than endogenous variables are available.
