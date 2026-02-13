# Exercises: General Statistical Estimation

## Conceptual

**Exercise 1.** Let $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Uniform}(0, \theta)$. Consider two estimators of $\theta$:

- $\hat{\theta}_1 = 2\bar{X}$
- $\hat{\theta}_2 = \frac{n+1}{n}X_{(n)}$, where $X_{(n)} = \max(X_1, \ldots, X_n)$

(a) Show that both estimators are unbiased.

(b) Compute the variance of each estimator. (Hint: the PDF of $X_{(n)}$ is $f_{X_{(n)}}(x) = \frac{n}{\theta^n}x^{n-1}$ for $0 \leq x \leq \theta$.)

(c) Which estimator has smaller MSE? Does the answer depend on $n$?

**Exercise 2.** Prove that if $\hat{\theta}$ is an unbiased estimator of $\theta$ and $\text{Var}(\hat{\theta}) = 0$, then $\hat{\theta} = \theta$ with probability 1.

**Exercise 3.** Let $X_1, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$. Consider the weighted estimator $\hat{\mu}_w = w X_1 + (1-w)\bar{X}$ for $0 \leq w \leq 1$.

(a) Show that $\hat{\mu}_w$ is unbiased for all $w$.

(b) Find the value of $w$ that minimizes $\text{Var}(\hat{\mu}_w)$.

(c) What happens when $w = 0$? When $w = 1$?

**Exercise 4.** Consider estimating $\theta$ from $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Bernoulli}(\theta)$ with the Bayesian-inspired estimator:

$$
\hat{\theta}_B = \frac{\sum X_i + a}{n + a + b}
$$

(a) Compute $\text{Bias}(\hat{\theta}_B)$ and $\text{Var}(\hat{\theta}_B)$.

(b) For $a = b = \sqrt{n}/2$, show that $\hat{\theta}_B$ is biased but consistent.

(c) Find the MSE and compare with the MLE $\hat{\theta} = \bar{X}$ when $\theta = 0.5$ and $n = 10$.

## Computation

**Exercise 5.** Derive the MLE for the Geometric distribution $P(X = k) = (1-p)^{k-1}p$ for $k = 1, 2, \ldots$

(a) Write the log-likelihood for a sample $x_1, \ldots, x_n$.

(b) Find $\hat{p}_{\text{MLE}}$.

(c) Find the MoM estimator $\hat{p}_{\text{MoM}}$ using $E[X] = 1/p$.

(d) Are they the same?

**Exercise 6.** For the Pareto distribution with PDF $f(x; \alpha) = \alpha x^{-(\alpha+1)}$ for $x \geq 1$:

(a) Find the MLE of $\alpha$.

(b) Find the MoM estimator using $E[X] = \frac{\alpha}{\alpha - 1}$ (for $\alpha > 1$).

(c) Compute the Fisher information $I(\alpha)$ and the CRLB.

(d) Is the MLE efficient?

**Exercise 7.** Write a simulation that:

(a) Generates 10,000 samples of size $n = 30$ from $\text{Gamma}(\alpha = 2, \beta = 3)$.

(b) For each sample, computes the MLE and MoM estimates of $\alpha$ and $\beta$.

(c) Compares the empirical bias, variance, and MSE of MLE vs MoM.

(d) Repeats for $n = 5, 10, 30, 100, 500$ and plots MSE as a function of $n$.

**Exercise 8 (Fisher Information).** For the Poisson distribution $f(x; \lambda) = \frac{\lambda^x e^{-\lambda}}{x!}$:

(a) Compute the Fisher information $I(\lambda)$.

(b) State the CRLB for estimating $\lambda$.

(c) Show that the MLE $\hat{\lambda} = \bar{X}$ is efficient.

(d) Compute the Fisher information for estimating $g(\lambda) = e^{-\lambda} = P(X = 0)$ and find the CRLB for this function.

## Applied

**Exercise 9 (Finance).** Daily log-returns of a stock are often modeled as $r_t \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$.

(a) Given 252 trading days with $\bar{r} = 0.0004$ and $s = 0.015$, compute the MLE of the annualized mean return $\mu_{\text{annual}} = 252\mu$ and annualized volatility $\sigma_{\text{annual}} = \sigma\sqrt{252}$.

(b) Construct approximate 95% confidence intervals for both using the asymptotic normality of MLE.

(c) The sample mean of daily returns is very noisy. Compute the standard error of $\hat{\mu}_{\text{annual}}$ and comment on the difficulty of estimating expected returns compared to volatility.

**Exercise 10.** Insurance claim amounts often follow a Gamma distribution.

(a) Given the following claim data (in thousands): 2.1, 0.8, 3.5, 1.2, 5.7, 0.4, 2.8, 1.9, 4.3, 0.6, compute MoM estimates of the Gamma parameters $\alpha$ and $\beta$.

(b) Use `scipy.stats.gamma.fit()` to compute MLE estimates.

(c) Compare the fitted distributions visually by plotting histograms with overlaid densities.
