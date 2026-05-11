# Asymptotic Properties of MLE

## Why Asymptotics Matter

In finite samples, the behavior of the MLE depends on the specific model and data. However, under general regularity conditions, the MLE possesses three powerful properties as the sample size $n$ grows: it converges to the true parameter value (consistency), it becomes approximately normally distributed (asymptotic normality), and it achieves the best possible precision (asymptotic efficiency). These properties justify using MLE as a default estimation strategy and provide the theoretical basis for confidence intervals, hypothesis tests, and model comparison tools built on the likelihood.

Throughout this page, let $\theta_0$ denote the **true parameter value** — the value that actually generated the data. We assume $X_1, \ldots, X_n \overset{\text{iid}}{\sim} f(x; \theta_0)$.

## Regularity Conditions

The asymptotic results below rely on a set of regularity conditions. These conditions ensure that the likelihood function is well-behaved enough for the standard theory to apply.

1. **Identifiability.** Different parameter values produce different distributions: if $\theta_1 \neq \theta_2$, then $f(x; \theta_1) \neq f(x; \theta_2)$ for at least one $x$.

2. **Common support.** The set $\{x : f(x; \theta) > 0\}$ does not depend on $\theta$. This excludes distributions like $\text{Uniform}(0, \theta)$ whose support changes with the parameter.

3. **Interior parameter.** The true value $\theta_0$ lies in the interior of the parameter space $\Theta$, not on the boundary.

4. **Smoothness.** The log-density $\log f(x; \theta)$ is at least three times continuously differentiable with respect to $\theta$, and the derivatives can be passed under the integral (or sum).

5. **Positive Fisher information.** The Fisher information $I(\theta_0) > 0$, ensuring that the data actually carry information about $\theta$.

!!! warning "When Regularity Fails"

    When regularity conditions are violated, the standard asymptotic results may break down entirely. For example, the MLE of $\theta$ in $\text{Uniform}(0, \theta)$ is $X_{(n)}$ (the sample maximum), which converges at rate $1/n$ rather than the usual $1/\sqrt{n}$ and has an exponential (not normal) limiting distribution.

## Consistency

A good estimator should converge to the truth as we collect more data. **Consistency** formalizes this minimal requirement.

Under the regularity conditions above, the MLE is consistent:

$$
\hat{\theta}_{\text{MLE}} \xrightarrow{P} \theta_0 \quad \text{as } n \to \infty
$$

This means that for any $\epsilon > 0$,

$$
P(|\hat{\theta}_{\text{MLE}} - \theta_0| > \epsilon) \to 0 \quad \text{as } n \to \infty
$$

The proof relies on the fact that the log-likelihood ratio $\ell(\theta) - \ell(\theta_0)$ converges uniformly to the Kullback-Leibler divergence $-\text{KL}(f_{\theta_0} \| f_\theta)$, which is uniquely maximized at $\theta = \theta_0$ by identifiability.

!!! tip "Consistency Is Necessary but Not Sufficient"

    Consistency alone does not tell us how fast the estimator converges or what its distribution looks like. Two consistent estimators can have very different finite-sample behavior. The next two properties address rate and distributional shape.

## Asymptotic Normality

The most practically important asymptotic property of the MLE is that its sampling distribution becomes approximately normal for large $n$.

Under the regularity conditions, the MLE satisfies

$$
\sqrt{n}(\hat{\theta}_{\text{MLE}} - \theta_0) \xrightarrow{d} N\left(0, \frac{1}{I(\theta_0)}\right)
$$

Equivalently, for large $n$:

$$
\hat{\theta}_{\text{MLE}} \overset{\text{approx}}{\sim} N\left(\theta_0, \frac{1}{nI(\theta_0)}\right)
$$

This result has immediate practical consequences. An approximate $(1 - \alpha)$ confidence interval for $\theta_0$ is

$$
\hat{\theta}_{\text{MLE}} \pm z_{\alpha/2} \cdot \frac{1}{\sqrt{nI(\hat{\theta}_{\text{MLE}})}}
$$

where $z_{\alpha/2}$ is the $(1 - \alpha/2)$ quantile of the standard normal distribution. In practice, we evaluate the Fisher information at $\hat{\theta}_{\text{MLE}}$ since $\theta_0$ is unknown.

!!! example "Asymptotic CI for Bernoulli Parameter"

    For $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Bernoulli}(p)$, the MLE is $\hat{p} = \bar{X}$ and the Fisher information is $I(p) = 1/(p(1-p))$. An approximate 95% confidence interval is

    $$
    \hat{p} \pm 1.96 \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
    $$

    For $n = 100$ and $\hat{p} = 0.3$, this gives $0.3 \pm 1.96\sqrt{0.21/100} = 0.3 \pm 0.090$, or approximately $(0.210, 0.390)$.

## Asymptotic Efficiency

Asymptotic normality tells us the MLE converges at rate $1/\sqrt{n}$ with asymptotic variance $1/(nI(\theta_0))$. But is this variance the best we can achieve? The answer is yes.

Among regular estimators, the MLE achieves the **Cramer-Rao lower bound** asymptotically:

$$
\text{Var}_{\text{asy}}(\hat{\theta}_{\text{MLE}}) = \frac{1}{nI(\theta_0)}
$$

This means no other regular consistent estimator can have a smaller asymptotic variance. Any alternative estimator $\tilde{\theta}$ that is also consistent and asymptotically normal must satisfy

$$
\text{Var}_{\text{asy}}(\tilde{\theta}) \geq \frac{1}{nI(\theta_0)}
$$

with equality if and only if $\tilde{\theta}$ is asymptotically equivalent to the MLE.

!!! note "Superefficiency"

    It is possible to construct estimators that have smaller variance than the MLE at isolated points in the parameter space (Hodges' estimator is the classical example). However, such estimators necessarily have larger variance at other points. The MLE is efficient uniformly over the parameter space among the class of regular estimators.

## Summary of the Three Properties

| Property | Statement | Practical Implication |
|---|---|---|
| Consistency | $\hat{\theta}_{\text{MLE}} \xrightarrow{P} \theta_0$ | Estimates converge to the truth |
| Asymptotic normality | $\sqrt{n}(\hat{\theta}_{\text{MLE}} - \theta_0) \xrightarrow{d} N(0, 1/I(\theta_0))$ | Confidence intervals and tests via normal approximation |
| Asymptotic efficiency | $\text{Var}_{\text{asy}} = 1/(nI(\theta_0))$ | No regular estimator can do better |

These three results, taken together, make the MLE the default choice in parametric estimation whenever the regularity conditions hold and the sample size is large enough for the asymptotic approximation to be reliable.

## Exercises

**Exercise 1.**
State the three main asymptotic properties of the MLE under regularity conditions: consistency, asymptotic normality, and asymptotic efficiency.

??? success "Solution to Exercise 1"
    Under standard regularity conditions:

    1. **Consistency:** $\hat{\theta}_{\text{MLE}} \xrightarrow{p} \theta_0$ as $n \to \infty$.

    2. **Asymptotic normality:** $\sqrt{n}(\hat{\theta}_{\text{MLE}} - \theta_0) \xrightarrow{d} N(0, I(\theta_0)^{-1})$, where $I(\theta_0)$ is the Fisher information for a single observation.

    3. **Asymptotic efficiency:** The MLE achieves the Cramer-Rao lower bound asymptotically. That is, no consistent estimator can have a smaller asymptotic variance than $1/(nI(\theta_0))$.

    These properties make the MLE the default choice for parametric estimation in large samples, though in small samples the MLE may be biased or less efficient than other estimators.

---

**Exercise 2.**
For the exponential distribution with rate parameter $\lambda$, the MLE is $\hat{\lambda} = 1/\bar{X}$. Compute the Fisher information $I(\lambda)$ and verify that the asymptotic variance of $\hat{\lambda}$ is $\lambda^2/n$.

??? success "Solution to Exercise 2"
    The log-likelihood for one observation is $\ell(\lambda) = \log\lambda - \lambda x$. The second derivative is:

    $$
    \frac{d^2\ell}{d\lambda^2} = -\frac{1}{\lambda^2}
    $$

    The Fisher information is:

    $$
    I(\lambda) = -E\!\left[\frac{d^2\ell}{d\lambda^2}\right] = \frac{1}{\lambda^2}
    $$

    The asymptotic variance of the MLE is:

    $$
    \text{Var}(\hat{\lambda}) \approx \frac{1}{nI(\lambda)} = \frac{\lambda^2}{n}
    $$

    This means $\hat{\lambda} \approx N(\lambda, \lambda^2/n)$ for large $n$, which gives the standard error $\text{SE}(\hat{\lambda}) \approx \lambda/\sqrt{n}$.

---

**Exercise 3.**
The invariance property states that if $\hat{\theta}$ is the MLE of $\theta$, then $g(\hat{\theta})$ is the MLE of $g(\theta)$ for any function $g$. Use this to find the MLE of the mean $1/\lambda$ of an exponential distribution.

??? success "Solution to Exercise 3"
    The MLE of the rate parameter is $\hat{\lambda} = 1/\bar{X}$. By the invariance property, the MLE of $g(\lambda) = 1/\lambda$ (the population mean) is:

    $$
    \widehat{1/\lambda} = g(\hat{\lambda}) = \frac{1}{\hat{\lambda}} = \frac{1}{1/\bar{X}} = \bar{X}
    $$

    This confirms the intuitive result: the MLE of the exponential mean is the sample mean. The invariance property is powerful because it works for any transformation, including non-linear ones, without re-deriving the MLE from scratch.

---

**Exercise 4.**
Explain why the MLE can be biased in finite samples despite being asymptotically unbiased. Give a specific example.

??? success "Solution to Exercise 4"
    Asymptotic unbiasedness means $E[\hat{\theta}_n] \to \theta_0$ as $n \to \infty$, but for any fixed $n$, the bias $E[\hat{\theta}_n] - \theta_0$ may be nonzero.

    **Example:** For the normal variance, the MLE is $\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum(X_i - \bar{X})^2$, which has:

    $$
    E[\hat{\sigma}^2_{\text{MLE}}] = \frac{n-1}{n}\sigma^2 \neq \sigma^2
    $$

    The bias is $-\sigma^2/n$, which vanishes as $n \to \infty$ but is nonzero for every finite $n$. This is why the unbiased estimator $S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$ uses the Bessel correction $n - 1$.

    In general, if $\hat{\theta}$ is the MLE of $\theta$ and $g$ is nonlinear, then $g(\hat{\theta})$ is the MLE of $g(\theta)$ by invariance, but Jensen's inequality implies $E[g(\hat{\theta})] \neq g(\theta)$ in finite samples, introducing bias.
