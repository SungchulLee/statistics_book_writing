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
