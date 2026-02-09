# Likelihood Function

## Introduction

The **likelihood function** is the cornerstone of parametric statistical inference. Given observed data, the likelihood function measures how "likely" each candidate parameter value is to have generated that data. Unlike a probability distribution, which assigns probabilities to outcomes given fixed parameters, the likelihood function fixes the data and treats the parameter as the variable.

Maximum Likelihood Estimation (MLE), built on the likelihood function, is the most widely used estimation method in statistics, machine learning, and quantitative finance — from fitting distribution parameters to calibrating option pricing models.

## Definition

### Likelihood Function

Let $X_1, X_2, \ldots, X_n$ be a random sample from a distribution with probability density (or mass) function $f(x; \theta)$, where $\theta \in \Theta$ is an unknown parameter (possibly a vector). After observing data $x_1, x_2, \ldots, x_n$, the **likelihood function** is:

$$L(\theta) = L(\theta; x_1, \ldots, x_n) = \prod_{i=1}^n f(x_i; \theta)$$

**Key distinction:** $f(x; \theta)$ viewed as a function of $x$ (with $\theta$ fixed) is a density. The same expression viewed as a function of $\theta$ (with $x$ fixed at the observed data) is the likelihood.

**Important properties:**
- The likelihood is **not** a probability density over $\theta$ — it does not integrate to 1 over $\Theta$
- Only **ratios** of likelihoods are meaningful; the absolute scale is arbitrary
- The likelihood function summarizes all the information in the data about $\theta$ (by the sufficiency principle)

### Log-Likelihood Function

Because the likelihood is a product of many terms, it is almost always more convenient to work with the **log-likelihood**:

$$\ell(\theta) = \log L(\theta) = \sum_{i=1}^n \log f(x_i; \theta)$$

**Advantages of the log-likelihood:**
- Converts products to sums (computationally stable, analytically simpler)
- Preserves the location of the maximum (log is monotonically increasing)
- Connects directly to information-theoretic quantities (KL divergence, entropy)
- Avoids numerical underflow for large samples

## Maximum Likelihood Estimation

### Definition

The **Maximum Likelihood Estimator (MLE)** is the value of $\theta$ that maximizes the likelihood function:

$$\hat{\theta}_{\text{MLE}} = \arg\max_{\theta \in \Theta} L(\theta) = \arg\max_{\theta \in \Theta} \ell(\theta)$$

The MLE answers the question: "Which parameter value makes the observed data most probable?"

### Finding the MLE

For differentiable log-likelihoods, the MLE is typically found by solving the **score equation**:

$$\frac{\partial \ell(\theta)}{\partial \theta} = 0$$

The **score function** $s(\theta) = \frac{\partial \ell(\theta)}{\partial \theta}$ is the gradient of the log-likelihood. The MLE satisfies $s(\hat{\theta}_{\text{MLE}}) = 0$.

For vector parameters $\theta = (\theta_1, \ldots, \theta_k)^T$, we solve the system:

$$\frac{\partial \ell}{\partial \theta_j} = 0, \quad j = 1, \ldots, k$$

One must verify that the solution is a maximum (not a minimum or saddle point), typically by checking that the Hessian matrix is negative definite at the solution.

## Worked Examples

### Example 1: Normal Distribution — Unknown Mean

Let $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$ with $\sigma^2$ known. Find the MLE of $\mu$.

**Likelihood:**

$$L(\mu) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)$$

**Log-likelihood:**

$$\ell(\mu) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2$$

**Score equation:**

$$\frac{d\ell}{d\mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = 0$$

$$\sum_{i=1}^n x_i - n\mu = 0$$

$$\hat{\mu}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n x_i = \bar{x}$$

The MLE of the mean is the sample mean — unbiased and efficient.

### Example 2: Normal Distribution — Both Parameters Unknown

Let $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$ with both $\mu$ and $\sigma^2$ unknown.

**Log-likelihood:**

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2$$

**Score equations:**

$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = 0 \implies \hat{\mu} = \bar{x}$$

$$\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^n (x_i - \mu)^2 = 0 \implies \hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2$$

**Note:** The MLE of $\sigma^2$ divides by $n$, not $n-1$. It is biased: $E[\hat{\sigma}^2_{\text{MLE}}] = \frac{n-1}{n}\sigma^2$. This illustrates that MLEs are not always unbiased.

### Example 3: Bernoulli Distribution

Let $X_1, \ldots, X_n \sim \text{Bernoulli}(p)$.

**Log-likelihood:**

$$\ell(p) = \sum_{i=1}^n \left[x_i \log p + (1 - x_i)\log(1 - p)\right]$$

$$= k \log p + (n - k)\log(1 - p)$$

where $k = \sum_{i=1}^n x_i$ is the number of successes.

**Score equation:**

$$\frac{d\ell}{dp} = \frac{k}{p} - \frac{n-k}{1-p} = 0$$

$$\hat{p}_{\text{MLE}} = \frac{k}{n} = \bar{x}$$

The MLE is the sample proportion — intuitively natural and unbiased.

### Example 4: Exponential Distribution

Let $X_1, \ldots, X_n \sim \text{Exp}(\lambda)$ with density $f(x; \lambda) = \lambda e^{-\lambda x}$ for $x > 0$.

**Log-likelihood:**

$$\ell(\lambda) = n\log\lambda - \lambda \sum_{i=1}^n x_i$$

**Score equation:**

$$\frac{d\ell}{d\lambda} = \frac{n}{\lambda} - \sum_{i=1}^n x_i = 0$$

$$\hat{\lambda}_{\text{MLE}} = \frac{n}{\sum_{i=1}^n x_i} = \frac{1}{\bar{x}}$$

**Second derivative check:** $\frac{d^2\ell}{d\lambda^2} = -n/\lambda^2 < 0$, confirming a maximum.

### Example 5: Poisson Distribution

Let $X_1, \ldots, X_n \sim \text{Poisson}(\lambda)$.

**Log-likelihood:**

$$\ell(\lambda) = \sum_{i=1}^n \left[x_i \log\lambda - \lambda - \log(x_i!)\right] = \left(\sum x_i\right)\log\lambda - n\lambda - \sum\log(x_i!)$$

**Score equation:**

$$\frac{d\ell}{d\lambda} = \frac{\sum x_i}{\lambda} - n = 0$$

$$\hat{\lambda}_{\text{MLE}} = \bar{x}$$

## Properties of MLEs

### Asymptotic Properties

Under regularity conditions, MLEs possess several powerful asymptotic properties:

**1. Consistency:** $\hat{\theta}_{\text{MLE}} \xrightarrow{P} \theta_0$ as $n \to \infty$, where $\theta_0$ is the true parameter value.

**2. Asymptotic Normality:** 

$$\sqrt{n}(\hat{\theta}_{\text{MLE}} - \theta_0) \xrightarrow{d} N(0, I(\theta_0)^{-1})$$

where $I(\theta_0)$ is the Fisher information. Equivalently:

$$\hat{\theta}_{\text{MLE}} \dot{\sim} N\left(\theta_0, \frac{1}{nI(\theta_0)}\right) \quad \text{for large } n$$

**3. Asymptotic Efficiency:** The MLE achieves the Cramér-Rao lower bound asymptotically — no consistent estimator has smaller asymptotic variance.

**4. Invariance Property:** If $\hat{\theta}$ is the MLE of $\theta$, then $g(\hat{\theta})$ is the MLE of $g(\theta)$ for any function $g$. This is extremely useful: for example, if $\hat{\sigma}^2$ is the MLE of $\sigma^2$, then $\sqrt{\hat{\sigma}^2}$ is the MLE of $\sigma$.

### Fisher Information

The **Fisher information** quantifies how much information a sample carries about the parameter:

$$I(\theta) = -E\left[\frac{\partial^2 \ell(\theta)}{\partial \theta^2}\right] = E\left[\left(\frac{\partial \ell(\theta)}{\partial \theta}\right)^2\right]$$

For $n$ iid observations, the total Fisher information is $nI_1(\theta)$, where $I_1(\theta)$ is the Fisher information from a single observation.

**Key role:** Fisher information determines the precision of the MLE — higher Fisher information means a sharper likelihood function and a more precise estimate.

### Observed Fisher Information

In practice, the **observed Fisher information** is often used instead of the expected Fisher information:

$$J(\hat{\theta}) = -\frac{\partial^2 \ell(\theta)}{\partial \theta^2}\bigg|_{\theta = \hat{\theta}}$$

This avoids computing expectations and is evaluated at the MLE. Standard errors are then estimated as $\text{SE}(\hat{\theta}) = 1/\sqrt{J(\hat{\theta})}$.

## Likelihood-Based Inference

### Likelihood Ratio

The **likelihood ratio** for comparing two parameter values $\theta_0$ and $\theta_1$ is:

$$\Lambda = \frac{L(\theta_0)}{L(\theta_1)}$$

Small values of $\Lambda$ indicate that the data favor $\theta_1$ over $\theta_0$.

### Likelihood Ratio Test

For testing $H_0: \theta = \theta_0$ against $H_1: \theta \neq \theta_0$, the **likelihood ratio test statistic** is:

$$\Lambda = \frac{L(\theta_0)}{L(\hat{\theta}_{\text{MLE}})}$$

Under $H_0$ and regularity conditions:

$$-2\log\Lambda = 2[\ell(\hat{\theta}_{\text{MLE}}) - \ell(\theta_0)] \xrightarrow{d} \chi^2_k$$

where $k$ is the number of constrained parameters. This is the foundation of many hypothesis tests in applied statistics.

### Confidence Intervals from the Likelihood

**Wald confidence interval:** Using asymptotic normality:

$$\hat{\theta} \pm z_{\alpha/2} \cdot \text{SE}(\hat{\theta})$$

where $\text{SE}(\hat{\theta}) = 1/\sqrt{nI(\hat{\theta})}$ or $1/\sqrt{J(\hat{\theta})}$.

**Profile likelihood interval:** The set of $\theta$ values satisfying:

$$2[\ell(\hat{\theta}) - \ell(\theta)] \leq \chi^2_{1, 1-\alpha}$$

This interval is generally more accurate than the Wald interval for small samples or skewed likelihoods.

## Computational Methods

### Newton-Raphson Method

When the score equation cannot be solved analytically, the MLE is found numerically. Newton-Raphson iterates:

$$\theta^{(t+1)} = \theta^{(t)} - \left[\frac{\partial^2 \ell}{\partial \theta^2}\bigg|_{\theta^{(t)}}\right]^{-1} \frac{\partial \ell}{\partial \theta}\bigg|_{\theta^{(t)}}$$

This is equivalent to iteratively fitting a quadratic approximation to the log-likelihood.

### Fisher Scoring

A variant replaces the observed Hessian with the expected Fisher information:

$$\theta^{(t+1)} = \theta^{(t)} + I(\theta^{(t)})^{-1} \cdot s(\theta^{(t)})$$

Fisher scoring is more stable when the observed Hessian is poorly conditioned.

### EM Algorithm

For models with latent variables (e.g., mixture models, hidden Markov models), the **Expectation-Maximization (EM) algorithm** iterates between:
- **E-step**: Compute the expected log-likelihood given current parameters and observed data
- **M-step**: Maximize this expected log-likelihood to update parameters

The EM algorithm guarantees monotone increase of the likelihood at each step and is widely used in financial applications such as regime-switching models.

## Connections to Finance

The likelihood function and MLE are pervasive in quantitative finance:

- **GARCH models**: Parameters $(\omega, \alpha, \beta)$ are estimated by maximizing the conditional log-likelihood of returns, typically assuming normal or Student-$t$ innovations.
- **Option pricing**: The Black-Scholes implied volatility is the MLE of volatility given observed option prices under the model's assumed dynamics.
- **Regime-switching models**: Hamilton's regime-switching model uses the EM algorithm to maximize the likelihood with respect to transition probabilities and regime-specific parameters.
- **Risk modeling**: Fitting fat-tailed distributions (Student-$t$, generalized Pareto) to loss data via MLE for Value-at-Risk and Expected Shortfall estimation.
- **Term structure models**: Parameters of Vasicek, CIR, and affine term structure models are calibrated by maximizing the likelihood of observed yield curves.
- **Copula models**: Dependence parameters in copula-based portfolio risk models are estimated via pseudo-maximum likelihood.

## Limitations of MLE

While MLE is powerful, it has important limitations:

- **Small-sample bias**: MLEs can be substantially biased in small samples (e.g., variance estimator divides by $n$ instead of $n-1$)
- **Boundary problems**: When the true parameter is on the boundary of the parameter space, standard asymptotic theory breaks down
- **Model misspecification**: MLE is optimal when the model is correct; under misspecification, it converges to the parameter value that minimizes KL divergence from the true distribution to the model family (quasi-MLE or pseudo-MLE)
- **Multimodal likelihoods**: Numerical optimization may find local rather than global maxima, especially in mixture models
- **Sensitivity to outliers**: For some models, a single outlier can dramatically shift the MLE

## Summary

The likelihood function transforms observed data into a measure of support for parameter values, and the MLE selects the most supported value. MLE is consistent, asymptotically efficient, and invariant to reparameterization. Through the likelihood ratio and Fisher information, the likelihood function also provides a unified framework for hypothesis testing and confidence intervals. These properties make MLE the default estimation method across statistics and quantitative finance.

## Key Formulas

| Quantity | Formula |
|----------|---------|
| Likelihood | $L(\theta) = \prod_{i=1}^n f(x_i; \theta)$ |
| Log-likelihood | $\ell(\theta) = \sum_{i=1}^n \log f(x_i; \theta)$ |
| Score function | $s(\theta) = \partial \ell(\theta) / \partial \theta$ |
| Fisher information | $I(\theta) = -E[\partial^2 \ell / \partial \theta^2]$ |
| Asymptotic variance of MLE | $1 / [nI(\theta)]$ |
| Likelihood ratio test | $-2\log\Lambda \sim \chi^2_k$ |
| Invariance | $\widehat{g(\theta)} = g(\hat{\theta}_{\text{MLE}})$ |
