# Efficiency and the Cramer-Rao Lower Bound

## Why a Lower Bound on Variance?

Given a parametric model, many different unbiased estimators of a parameter $\theta$ may exist. Some will have smaller variance than others, and a natural question arises: how small can the variance of an unbiased estimator possibly be? The **Cramer-Rao Lower Bound (CRLB)** answers this question by establishing a fundamental floor on the variance of any unbiased estimator. This result is central to estimation theory because it provides a benchmark against which we can measure the quality of any particular estimator.

## Fisher Information

Before stating the CRLB, we need the concept of Fisher information, which quantifies how much information a random observation carries about the unknown parameter.

Let $X$ be a random variable with probability density or mass function $f(x; \theta)$, where $\theta$ is the parameter of interest. The **score function** is the partial derivative of the log-likelihood with respect to $\theta$:

$$
s(X; \theta) = \frac{\partial}{\partial \theta} \log f(X; \theta)
$$

Under regularity conditions (the support of $f$ does not depend on $\theta$, and differentiation under the integral sign is valid), the score function has mean zero: $E[s(X; \theta)] = 0$.

The **Fisher information** for a single observation is defined as the variance of the score function:

$$
I(\theta) = E\left[\left(\frac{\partial}{\partial \theta} \log f(X;\theta)\right)^2\right]
$$

Under the same regularity conditions that permit interchanging differentiation and integration, this equals the negative expected second derivative of the log-likelihood:

$$
I(\theta) = -E\left[\frac{\partial^2}{\partial\theta^2} \log f(X;\theta)\right]
$$

The second form is often easier to compute. Intuitively, a large Fisher information means the log-likelihood is sharply curved around the true parameter value, making $\theta$ easier to estimate precisely.

For a random sample $X_1, \ldots, X_n \overset{\text{iid}}{\sim} f(x; \theta)$, the total Fisher information is $n I(\theta)$, reflecting the fact that independent observations contribute additively to information.

## The Cramer-Rao Inequality

With Fisher information in hand, we can state the fundamental result.

!!! info "Theorem (Cramer-Rao Lower Bound)"

    Let $X_1, \ldots, X_n$ be iid with density or mass function $f(x; \theta)$ satisfying the regularity conditions:

    1. The support $\{x : f(x; \theta) > 0\}$ does not depend on $\theta$.
    2. The derivatives $\frac{\partial}{\partial \theta} f(x; \theta)$ and $\frac{\partial^2}{\partial \theta^2} f(x; \theta)$ exist and are continuous.
    3. Differentiation and integration (or summation) can be interchanged.

    Then for any unbiased estimator $\hat{\theta} = \hat{\theta}(X_1, \ldots, X_n)$:

    $$
    \text{Var}(\hat{\theta}) \geq \frac{1}{n I(\theta)}
    $$

The quantity $1 / (nI(\theta))$ is the Cramer-Rao lower bound. No unbiased estimator can have variance below this threshold.

!!! example "CRLB for the Normal Mean"

    Let $X_1, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$ with $\sigma^2$ known. The log-likelihood for a single observation is

    $$
    \log f(x; \mu) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x - \mu)^2}{2\sigma^2}
    $$

    Taking the second derivative with respect to $\mu$:

    $$
    \frac{\partial^2}{\partial \mu^2} \log f(x; \mu) = -\frac{1}{\sigma^2}
    $$

    Therefore $I(\mu) = 1/\sigma^2$, and the CRLB for $n$ observations is

    $$
    \text{Var}(\hat{\mu}) \geq \frac{1}{n \cdot (1/\sigma^2)} = \frac{\sigma^2}{n}
    $$

    Since $\text{Var}(\bar{X}) = \sigma^2 / n$, the sample mean $\bar{X}$ achieves the CRLB exactly.

## Efficiency

The CRLB naturally leads to a way of ranking unbiased estimators. An unbiased estimator that achieves the lower bound is the best possible — it extracts all the information from the data.

An unbiased estimator $\hat{\theta}$ is called **efficient** if

$$
\text{Var}(\hat{\theta}) = \frac{1}{n I(\theta)}
$$

for all $\theta$. The **efficiency** of an unbiased estimator is the ratio of the CRLB to its actual variance:

$$
e(\hat{\theta}) = \frac{1 / (nI(\theta))}{\text{Var}(\hat{\theta})} \leq 1
$$

An efficiency of 1 means the estimator is efficient; values less than 1 indicate how much variance is "wasted" relative to the theoretical optimum.

!!! example "Efficiency of the Sample Mean (Normal)"

    From the example above, $\bar{X}$ has variance $\sigma^2/n$ and the CRLB is $\sigma^2/n$, so

    $$
    e(\bar{X}) = \frac{\sigma^2/n}{\sigma^2/n} = 1
    $$

    The sample mean is an efficient estimator of the normal mean (with known variance).

!!! example "Efficiency of the Sample Median (Normal)"

    For $X_1, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$, the sample median is also an unbiased estimator of $\mu$, but its asymptotic variance is $\pi \sigma^2 / (2n)$. Therefore its asymptotic efficiency is

    $$
    e(\text{median}) = \frac{\sigma^2/n}{\pi\sigma^2/(2n)} = \frac{2}{\pi} \approx 0.637
    $$

    The sample median uses only about 64% of the information that the sample mean extracts from normally distributed data.

## When the CRLB Cannot Be Achieved

Not every parametric model admits an efficient estimator. The CRLB is achievable if and only if the score function can be written in the form

$$
\frac{\partial}{\partial \theta} \log f(x; \theta) = a(\theta)\left[T(x) - \theta\right]
$$

for some function $a(\theta)$ and statistic $T(x)$. This condition is satisfied by exponential family distributions but fails for many other models.

!!! warning "Uniform Distribution"

    Let $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Uniform}(0, \theta)$. The support $\{x : 0 < x < \theta\}$ depends on $\theta$, violating the first regularity condition. The CRLB does not apply, and indeed the MLE $\hat{\theta} = X_{(n)}$ (the sample maximum) has variance that decreases at rate $1/n^2$, faster than any $1/n$ rate that the CRLB would allow.

## Exercises

**Exercise 1.**
$X_1, \ldots, X_n \sim N(\mu, \sigma^2)$. Estimator $\hat\mu_w = wX_1 + (1-w)\bar X$. (a) Unbiased? (b) $w$ minimizing $\mathrm{Var}$? (c) Variance at $w = 0$ and $w = 1$?

??? success "Solution to Exercise 1"
    (a) Linearity: $\mathbb{E}[\hat\mu_w] = w\mu + (1-w)\mu = \mu$. Unbiased for all $w$.

    (b) Writing $\bar X = X_1/n + (n-1)\bar X_{-1}/n$ where $\bar X_{-1}$ is mean of $X_2, \ldots, X_n$ (independent of $X_1$):

    $\hat\mu_w = (w + (1-w)/n) X_1 + ((1-w)(n-1)/n) \bar X_{-1}$

    Variance has positive coefficients on both terms; differentiating w.r.t. $w$ and setting to 0 gives $w^* = 0$.

    (c) $w = 0$: $\hat\mu = \bar X$, $\mathrm{Var} = \sigma^2/n$. $w = 1$: $\hat\mu = X_1$, $\mathrm{Var} = \sigma^2$. Penalty for $w = 1$: variance multiplied by $n$.

    Sample mean is more efficient than relying on a single observation, by factor $n$.

---

**Exercise 2.**
**Cramér-Rao lower bound (CRLB).** State and prove the CRLB for unbiased estimators.

??? success "Solution to Exercise 2"
    **CRLB:** if $\hat\theta$ is an unbiased estimator of $\theta$ based on $n$ i.i.d. observations from density $f(x; \theta)$:

    $$
    \mathrm{Var}(\hat\theta) \ge \frac{1}{n I(\theta)}
    $$

    where $I(\theta) = -\mathbb{E}[\partial^2 \log f/\partial \theta^2]$ is the Fisher information per observation.

    **Proof sketch:** by Cauchy-Schwarz, $|\mathrm{Cov}(\hat\theta, S)|^2 \le \mathrm{Var}(\hat\theta) \mathrm{Var}(S)$ where $S = \partial \log L/\partial \theta$ is the **score function**. Compute $\mathrm{Cov}(\hat\theta, S) = 1$ (from unbiasedness, by differentiating $\mathbb{E}[\hat\theta] = \theta$) and $\mathrm{Var}(S) = n I(\theta)$. So $1 \le \mathrm{Var}(\hat\theta) \cdot n I(\theta)$, giving the bound.

    Equality iff $\hat\theta$ is a linear function of the score — i.e., the estimator is **efficient**.

---

**Exercise 3.**
**Efficiency for normal mean.** Show that $\bar X$ achieves the CRLB for estimating $\mu$ in $X \sim N(\mu, \sigma^2)$.

??? success "Solution to Exercise 3"
    Fisher information for normal mean (Exercise from earlier section): $I(\mu) = 1/\sigma^2$. CRLB: $\mathrm{Var}(\hat\mu) \ge \sigma^2/n$.

    Sample mean: $\mathrm{Var}(\bar X) = \sigma^2/n$ — achieves CRLB exactly.

    So $\bar X$ is the **uniformly minimum-variance unbiased estimator (UMVUE)** for $\mu$. No other unbiased estimator has lower variance for any $\mu$.

    This is a strong optimality result: $\bar X$ is not just consistent and unbiased — it is the *best* such estimator under the normal model.

---

**Exercise 4.**
**Asymptotic efficiency of MLE.** State and explain the result that MLEs achieve the CRLB asymptotically.

??? success "Solution to Exercise 4"
    **Result:** under regularity conditions, $\hat\theta_{\text{MLE}}$ is **asymptotically normal** with variance equal to the CRLB:

    $$
    \sqrt n (\hat\theta_{\text{MLE}} - \theta) \xrightarrow{d} N(0, 1/I(\theta))
    $$

    So $\mathrm{Var}(\hat\theta_{\text{MLE}}) \to 1/(n I(\theta))$ — the inverse Fisher information per observation. This is the CRLB for unbiased estimators.

    **Implication:** MLEs are **asymptotically efficient**. Among all consistent estimators, the MLE has the smallest possible asymptotic variance.

    **Caveat:** finite-sample efficiency can be poor. MLE may have bias, may not exist, or may converge slowly. Asymptotic efficiency is a long-run guarantee, not a finite-sample one.

    For small samples, alternative estimators (MoM, bias-corrected MLE, James-Stein-style shrinkage) can outperform MLE. For $n$ large, MLE is essentially optimal.

---

**Exercise 5.**
**Inefficient unbiased estimator.** Construct an unbiased estimator of $\mu$ for $N(\mu, 1)$ that has variance $> 1/n$ (the CRLB). Explain why such estimators exist.

??? success "Solution to Exercise 5"
    Example: $\hat\mu = X_1$ (first observation only). Unbiased: $\mathbb{E}[X_1] = \mu$. Variance: $\mathrm{Var}(X_1) = 1$.

    For $n \ge 2$: $\mathrm{Var}(X_1) = 1 > 1/n = $ CRLB. So this estimator is inefficient by factor $n$.

    **Why such estimators exist:** the CRLB is the *lower bound*; many estimators are above it. The CRLB just says no unbiased estimator can be **below** it. Estimators that throw away information (like $X_1$ ignoring $X_2, \ldots, X_n$) are clearly suboptimal.

    Other examples: $(X_1 + X_2)/2$ (uses only 2 observations), $X_{(n)}$ (the max — unbiased only for certain distributions).

    **Efficiency** measures how close an unbiased estimator's variance is to the CRLB. $\bar X$ has efficiency 1 (achieves CRLB); $X_1$ has efficiency $1/n$ for $n$-sample normal data.

---

**Exercise 6.**
**Trade-off in real applications.** When might you prefer a biased but lower-MSE estimator (e.g., ridge regression, James-Stein) to an unbiased estimator?

??? success "Solution to Exercise 6"
    **MSE decomposition:** $\mathrm{MSE}(\hat\theta) = \mathrm{Var}(\hat\theta) + [\mathrm{Bias}(\hat\theta)]^2$.

    Unbiased estimators have $\mathrm{Bias} = 0$, so $\mathrm{MSE} = \mathrm{Var}$. CRLB lower-bounds this.

    By accepting some bias, we may reduce variance more than the bias squared adds — **MSE goes down**.

    **Examples:**

    - **Ridge regression:** biased toward zero, but reduces variance by penalizing large coefficients. Lower MSE in correlated-feature regimes.
    - **James-Stein** (estimating $p \ge 3$ normal means): combines individual sample means with shrinkage toward 0; dominates the MLE in MSE for any true parameters.
    - **MAP with informative prior:** biased toward prior mean, but lower variance from regularization. Lower MSE when prior is approximately correct.

    **When to prefer:**

    - Small sample size (high variance dominates).
    - Many parameters relative to data (high-dimensional regression).
    - Prediction is the goal (MSE matters more than unbiasedness).
    - Prior information is available (Bayesian setting).

    **When to prefer unbiased:**

    - Interpretation of the estimate matters (e.g., causal effect).
    - Sample size is large (variance shrinks; bias persists).
    - Regulatory or scientific norms require unbiasedness.

    The bias-variance trade-off is the fundamental tension; modern statistical learning routinely accepts bias to win on MSE.
