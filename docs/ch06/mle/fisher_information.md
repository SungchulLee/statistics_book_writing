# Fisher Information and Standard Errors

## Why Quantify Information?

When estimating a parameter $\theta$ from data, we want to know: how precisely can $\theta$ be estimated? The answer depends on how much "information" the data carry about $\theta$. Fisher information formalizes this notion. It determines the sharpest possible precision for any unbiased estimator and directly provides standard errors for maximum likelihood estimates. In short, Fisher information is the bridge between a parametric model and the quality of the inferences we can draw from it.

## The Score Function

To define Fisher information, we first introduce the **score function**. Let $X$ be a random variable with probability density or mass function $f(x; \theta)$. The score function is the derivative of the log-likelihood with respect to $\theta$:

$$
s(x; \theta) = \frac{\partial}{\partial \theta} \log f(x; \theta)
$$

The score function captures how sensitive the log-likelihood is to changes in $\theta$. When the log-likelihood changes sharply as $\theta$ varies, the data are highly informative about $\theta$; when it is flat, the data carry little information.

Under regularity conditions — specifically, that the support $\{x : f(x; \theta) > 0\}$ does not depend on $\theta$ and that differentiation and integration can be interchanged — the score function has a key property:

$$
E[s(X; \theta)] = 0
$$

This means the score function is centered at zero, and its variance measures the typical magnitude of the log-likelihood's slope.

## Definition of Fisher Information

The **Fisher information** for a single observation is the variance of the score function:

$$
I(\theta) = E\left[\left(\frac{\partial}{\partial \theta} \log f(X;\theta)\right)^2\right] = \text{Var}(s(X; \theta))
$$

Under the same regularity conditions (interchange of differentiation and integration), this is equivalent to the negative expected curvature of the log-likelihood:

$$
I(\theta) = -E\left[\frac{\partial^2}{\partial\theta^2} \log f(X;\theta)\right]
$$

The second form is often easier to compute. It also provides geometric intuition: a sharply curved log-likelihood (large $I(\theta)$) means the data strongly constrain the parameter, while a flat log-likelihood (small $I(\theta)$) means the data are uninformative.

For $n$ independent and identically distributed observations $X_1, \ldots, X_n$, the total Fisher information is additive:

$$
I_n(\theta) = n \cdot I(\theta)
$$

## Worked Examples

### Bernoulli Distribution

Let $X \sim \text{Bernoulli}(p)$, so $f(x; p) = p^x (1-p)^{1-x}$ for $x \in \{0, 1\}$. The log-likelihood for a single observation is

$$
\log f(x; p) = x \log p + (1-x) \log(1-p)
$$

Taking the second derivative:

$$
\frac{\partial^2}{\partial p^2} \log f(x; p) = -\frac{x}{p^2} - \frac{1-x}{(1-p)^2}
$$

Since $E[X] = p$:

$$
I(p) = -E\left[-\frac{X}{p^2} - \frac{1-X}{(1-p)^2}\right] = \frac{1}{p} \cdot \frac{1}{1} + \frac{1}{(1-p)} = \frac{1}{p(1-p)}
$$

Notice that Fisher information is largest when $p$ is near 0 or 1 (where each observation is most informative about $p$) and smallest when $p = 1/2$.

### Normal Distribution (Mean)

Let $X \sim N(\mu, \sigma^2)$ with $\sigma^2$ known. The log-likelihood for a single observation is

$$
\log f(x; \mu) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x - \mu)^2}{2\sigma^2}
$$

The second derivative with respect to $\mu$ is

$$
\frac{\partial^2}{\partial \mu^2} \log f(x; \mu) = -\frac{1}{\sigma^2}
$$

This is a constant (does not depend on $x$), so

$$
I(\mu) = \frac{1}{\sigma^2}
$$

The Fisher information increases as the noise $\sigma^2$ decreases, which matches intuition: less noisy data carry more information about the mean.

### Poisson Distribution

Let $X \sim \text{Poisson}(\lambda)$, so $f(x; \lambda) = e^{-\lambda}\lambda^x / x!$. The log-likelihood is

$$
\log f(x; \lambda) = -\lambda + x \log \lambda - \log(x!)
$$

The second derivative is

$$
\frac{\partial^2}{\partial \lambda^2} \log f(x; \lambda) = -\frac{x}{\lambda^2}
$$

Since $E[X] = \lambda$:

$$
I(\lambda) = \frac{1}{\lambda}
$$

## Standard Errors from Fisher Information

One of the most important applications of Fisher information is providing standard errors for the MLE. Under regularity conditions, the MLE $\hat{\theta}_{\text{MLE}}$ is asymptotically normal:

$$
\hat{\theta}_{\text{MLE}} \overset{d}{\to} N\left(\theta, \frac{1}{nI(\theta)}\right) \quad \text{as } n \to \infty
$$

In practice, we estimate the standard error by evaluating the Fisher information at the MLE (the plug-in principle):

$$
\widehat{\text{SE}}(\hat{\theta}_{\text{MLE}}) = \frac{1}{\sqrt{nI(\hat{\theta}_{\text{MLE}})}}
$$

This approximation improves with sample size and forms the basis for constructing confidence intervals and Wald test statistics.

!!! example "Standard Error for the Bernoulli MLE"

    For $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Bernoulli}(p)$, the MLE is $\hat{p} = \bar{X}$. The estimated standard error is

    $$
    \widehat{\text{SE}}(\hat{p}) = \frac{1}{\sqrt{n \cdot \frac{1}{\hat{p}(1-\hat{p})}}} = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
    $$

    This is the familiar formula for the standard error of a sample proportion.

## Summary of Common Fisher Information Values

| Distribution | Parameter | Fisher Information $I(\theta)$ |
|---|---|---|
| $\text{Bernoulli}(p)$ | $p$ | $\dfrac{1}{p(1-p)}$ |
| $N(\mu, \sigma^2)$ (known $\sigma^2$) | $\mu$ | $\dfrac{1}{\sigma^2}$ |
| $\text{Poisson}(\lambda)$ | $\lambda$ | $\dfrac{1}{\lambda}$ |
| $\text{Exp}(\lambda)$ | $\lambda$ | $\dfrac{1}{\lambda^2}$ |

Each entry in this table can be verified using the second-derivative method demonstrated in the worked examples above.

## Exercises

**Exercise 1.**
Pareto: $f(x; \alpha) = \alpha x^{-(\alpha+1)}$ for $x \ge 1$. (a) Derive $\hat\alpha_{\text{MLE}}$. (b) MoM using $\mathbb{E}[X] = \alpha/(\alpha-1)$. (c) Fisher info and CRLB. (d) Is MLE efficient?

??? success "Solution to Exercise 1"
    (a) $\ell(\alpha) = n\ln\alpha - (\alpha + 1)\sum \ln x_i$. $\ell'(\alpha) = n/\alpha - \sum\ln x_i = 0 \Rightarrow \hat\alpha_{\text{MLE}} = n/\sum \ln x_i$.

    (b) $\bar x = \alpha/(\alpha - 1) \Rightarrow \hat\alpha_{\text{MoM}} = \bar x/(\bar x - 1)$.

    (c) $\partial^2 \ln f/\partial\alpha^2 = -1/\alpha^2 \Rightarrow I(\alpha) = 1/\alpha^2$. CRLB = $\alpha^2/n$.

    (d) $Y = \ln X \sim \mathrm{Exp}(\alpha)$, so $\sum \ln X_i \sim \mathrm{Gamma}(n, 1/\alpha)$ and $\hat\alpha_{\text{MLE}} = n/\sum \ln X_i$ has $\mathbb{E}[\hat\alpha_{\text{MLE}}] = n\alpha/(n-1)$ (biased upward). Bias-corrected version $\tilde\alpha = (n-1)/n \cdot \hat\alpha_{\text{MLE}}$ achieves CRLB asymptotically. MLE asymptotically efficient.

---

**Exercise 2.**
Poisson$(\lambda)$: $f(x; \lambda) = \lambda^x e^{-\lambda}/x!$. (a) Compute $I(\lambda)$. (b) CRLB. (c) Show $\bar X$ is efficient. (d) CRLB for $g(\lambda) = e^{-\lambda}$.

??? success "Solution to Exercise 2"
    (a) $\ln f = x\ln\lambda - \lambda - \ln x!$. $\partial^2/\partial\lambda^2 = -x/\lambda^2$. $I(\lambda) = \mathbb{E}[X]/\lambda^2 = 1/\lambda$.

    (b) CRLB = $\lambda/n$.

    (c) $\mathrm{Var}(\bar X) = \lambda/n$ = CRLB. Efficient.

    (d) Delta-method CRLB: $\mathrm{Var}(\widehat{g}) \ge [g'(\lambda)]^2/(n I(\lambda)) = e^{-2\lambda} \cdot \lambda/n = \lambda e^{-2\lambda}/n$.

---

**Exercise 3.**
**Define Fisher information** and prove the equivalence of the two definitions: $I(\theta) = \mathbb{E}[(\partial \log f/\partial\theta)^2] = -\mathbb{E}[\partial^2 \log f/\partial\theta^2]$.

??? success "Solution to Exercise 3"
    **Definition 1 (score variance):** $I(\theta) = \mathbb{E}[(\partial \log f/\partial\theta)^2]$.

    **Definition 2 (Hessian):** $I(\theta) = -\mathbb{E}[\partial^2 \log f/\partial\theta^2]$.

    **Equivalence proof:** Differentiate the identity $\int f \, dx = 1$ once: $\int \partial f/\partial\theta \, dx = 0 \Rightarrow \int f \cdot \partial \log f/\partial\theta \, dx = 0 \Rightarrow \mathbb{E}[\partial \log f/\partial\theta] = 0$.

    Differentiate again: $\int [\partial^2 f/\partial\theta^2] dx = 0$. Express $\partial^2 f/\partial\theta^2 = f \partial^2 \log f/\partial\theta^2 + f (\partial \log f/\partial\theta)^2$:

    $\mathbb{E}[\partial^2 \log f/\partial\theta^2] + \mathbb{E}[(\partial \log f/\partial\theta)^2] = 0$.

    So $I(\theta) = \mathbb{E}[(\partial \log f/\partial\theta)^2] = -\mathbb{E}[\partial^2 \log f/\partial\theta^2]$. $\square$

    The two definitions are computationally interchangeable; choose whichever is easier for the specific likelihood.

---

**Exercise 4.**
**Additivity of Fisher info.** For i.i.d. data, $I_n(\theta) = n I(\theta)$ where $I$ is the per-observation Fisher info. Prove this and interpret.

??? success "Solution to Exercise 4"
    Joint log-likelihood: $\log L(\theta) = \sum_i \log f(X_i; \theta)$.

    Score: $\partial \log L/\partial\theta = \sum_i \partial \log f(X_i; \theta)/\partial\theta$. Variance of sum of independent random variables is sum of variances:

    $\mathrm{Var}(\partial \log L/\partial\theta) = \sum_i \mathrm{Var}(\partial \log f(X_i; \theta)/\partial\theta) = n I(\theta)$.

    So $I_n(\theta) = n I(\theta)$. $\square$

    **Interpretation:** information accumulates linearly with sample size. Doubling $n$ doubles the Fisher info, halves the CRLB, and tightens the asymptotic variance of the MLE.

---

**Exercise 5.**
**Information and reparameterization.** Show that under $\eta = g(\theta)$ with $g$ differentiable, $I(\eta) = I(\theta)/[g'(\theta)]^2$.

??? success "Solution to Exercise 5"
    By chain rule: $\partial \log f/\partial\eta = (\partial \log f/\partial\theta) \cdot (\partial\theta/\partial\eta) = (\partial \log f/\partial\theta)/g'(\theta)$.

    Variance: $I(\eta) = \mathbb{E}[(\partial \log f/\partial\eta)^2] = \mathbb{E}[(\partial \log f/\partial\theta)^2]/[g'(\theta)]^2 = I(\theta)/[g'(\theta)]^2$.

    $\square$

    **Implication:** Fisher information **depends on the parameterization**. Reparameterizing changes the information content quantitatively. This is one reason Jeffreys' prior $\pi(\theta) \propto \sqrt{I(\theta)}$ is appealing — it's invariant under reparameterization (whereas a flat prior is not).

---

**Exercise 6.**
**Observed vs. expected information.** Define both. When are they equal, and when do they differ?

??? success "Solution to Exercise 6"
    **Expected (Fisher) information:** $I(\theta) = -\mathbb{E}_\theta[\partial^2 \log f/\partial\theta^2]$. Expectation under the true distribution.

    **Observed information:** $J(\hat\theta) = -\partial^2 \log L/\partial\theta^2 \big|_{\hat\theta}$. Evaluated at the *sample* MLE, no expectation taken.

    **Equality:** at $\theta = \hat\theta$ and asymptotically, $J(\hat\theta)/n \to I(\theta)$. They agree in expectation.

    **Differences:**

    - For finite samples, observed info uses the actual data and may give a different point estimate.
    - Expected info requires integrating over the distribution, which can be analytically hard or unavailable.
    - Observed info is computationally simpler: just compute the second derivative at the MLE.

    **Recommendation:** Efron and Hinkley (1978) argue that **observed information** is preferable for finite-sample inference (gives more accurate CIs in many cases). Statistical software (R, Python's `statsmodels`) typically uses observed info by default.
