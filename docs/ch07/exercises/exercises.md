# Chapter 7 Exercises: Estimation of Mean and Variance

## Section 7.1: Estimation of the Mean

### Exercise 7.1.1 — Unbiasedness of the Sample Mean
Let $X_1, \ldots, X_n$ be independent with $E[X_i] = \mu$ for all $i$. Show $\bar{X}$ is unbiased. Does this require independence?

### Exercise 7.1.2 — Variance with Correlated Data
Suppose $X_1, \ldots, X_n$ have common mean $\mu$, variance $\sigma^2$, and pairwise correlation $\rho$.

(a) Show $\text{Var}(\bar{X}) = \frac{\sigma^2}{n}[1 + (n-1)\rho]$.

(b) What happens as $n \to \infty$? When is $\bar{X}$ still consistent?

(c) A fund equally weights 20 hedge funds with 15% vol and 0.4 pairwise correlation. What is SE of estimated mean return from one year?

### Exercise 7.1.3 — Relative Efficiency
(a) Show the asymptotic relative efficiency of median to mean for Normal data is $2/\pi$.

(b) Verify via simulation for $n = 50$. Repeat for $t_3$ data. Which estimator wins for heavy tails?

### Exercise 7.1.4 — Shrinkage Estimator
For $\hat{\mu}_\lambda = \lambda\bar{X}$:

(a) Derive MSE as a function of $\lambda, \mu, \sigma^2, n$.

(b) Find MSE-optimal $\lambda^*$. Show $\lambda^* < 1$ when $|\mu| < \sigma/\sqrt{n}$.

(c) Why can't we use $\lambda^*$ directly in practice?

### Exercise 7.1.5 — Estimation Horizon
Strategy: 5% expected annual return, 18% volatility.

(a) SE of estimated mean from 10 years of annual data?

(b) How many years until 95% CI excludes zero?

(c) Does switching to monthly data change the answer to (b)?

---

## Section 7.2: Estimation of the Variance

### Exercise 7.2.1 — Deriving the Bias
Prove $E[\frac{1}{n}\sum(X_i - \bar{X})^2] = \frac{n-1}{n}\sigma^2$ using the identity $\sum(X_i - \bar{X})^2 = \sum(X_i - \mu)^2 - n(\bar{X} - \mu)^2$.

### Exercise 7.2.2 — MSE-Optimal Divisor
For $\hat{\sigma}^2_c = \frac{1}{c}\sum(X_i - \bar{X})^2$, Normal data:

(a) Derive $\text{MSE}(\hat{\sigma}^2_c)$.

(b) Find $c^*$ minimizing MSE. Verify $c^* = n+1$.

(c) For $n=10$, compute MSE at $c=9,10,11$ and verify ordering.

### Exercise 7.2.3 — Known Mean Advantage
When $\mu$ is known, $\hat{\sigma}^2 = \frac{1}{n}\sum(X_i - \mu)^2$:

(a) Show it is unbiased.

(b) For Normal, show $\text{Var}(\hat{\sigma}^2) = 2\sigma^4/n$.

(c) What is the relative efficiency gain from knowing $\mu$?

### Exercise 7.2.4 — Chi-Squared Distribution
For Normal data:

(a) Show $(n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$.

(b) Construct 95% CI for $\sigma^2$ when $n=20, S^2=16$.

(c) Explain why the CI is asymmetric about $S^2$.

### Exercise 7.2.5 — Standard Deviation Bias
(a) Use Jensen's inequality to explain why $E[S] < \sigma$.

(b) For $n=5$, compute the correction factor $c_4$.

(c) Is this correction typically applied in practice?

### Exercise 7.2.6 — Realized Volatility
Analyst estimates daily vol from 78 five-minute returns.

(a) Bias of $1/n$ estimator as fraction of true variance?

(b) Does Bessel's correction matter when $n=78$?

(c) Impact of using a 21-day rolling window instead?

---

## Section 7.3: Gaussian MLE

### Exercise 7.3.1 — Deriving the MLE
From $\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum(x_i - \mu)^2$:

(a) Derive $\hat{\mu}_{\text{MLE}} = \bar{X}$.

(b) Derive $\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum(X_i - \bar{X})^2$.

(c) Verify second-order conditions.

### Exercise 7.3.2 — Fisher Information
(a) Compute the Fisher information matrix $I(\mu, \sigma^2)$. Show off-diagonals are zero.

(b) What does this imply about estimating $\mu$ and $\sigma^2$?

(c) Verify $\hat{\mu}$ achieves the CRLB for all $n$.

### Exercise 7.3.3 — Invariance Property
Using MLE invariance, find the MLE of:

(a) $\sigma$ (standard deviation)

(b) $\text{CV} = \sigma/\mu$ (coefficient of variation)

(c) The 99th percentile $\mu + 2.326\sigma$

### Exercise 7.3.4 — Constrained MLE ($\mu = 0$)
(a) Derive the constrained MLE of $\sigma^2$ and show it is unbiased.

(b) Compare its variance with the unrestricted MLE.

(c) For $n=30$ daily returns with $\sum r_i^2 = 0.0048$, compute $\hat{\sigma}_{\text{annual}}$.

### Exercise 7.3.5 — Parametric VaR
Given $\hat{\mu} = 0.0003, \hat{\sigma} = 0.012$ from 252 daily observations:

(a) Compute 1-day 99% parametric VaR.

(b) Compute 10-day 99% VaR (square-root-of-time rule).

(c) If true excess kurtosis is 3, does normal VaR overestimate or underestimate risk?

### Exercise 7.3.6 — Monte Carlo Verification
Write Python code to generate 10,000 samples of $n=20$ from $N(5, 9)$, compute $\hat{\mu}$, $\hat{\sigma}^2_{\text{MLE}}$, $S^2$, and verify their expected values.

---

## Challenge Problems

### Challenge 7.1 — James-Stein Estimator
For $X \sim N(\mu, I_p)$ with $p \geq 3$, implement and compare the James-Stein estimator $\hat{\mu}^{JS} = (1 - (p-2)/\|X\|^2)X$ with $X$ across simulations for $p = 10$.

### Challenge 7.2 — Ledoit-Wolf Shrinkage
Simulate $p=30$ assets, $n=60$ observations. Compare sample covariance vs Ledoit-Wolf shrinkage estimator in Frobenius norm. Report MSE reduction and how optimal shrinkage intensity depends on $p/n$.

### Challenge 7.3 — Bootstrap Standard Errors
Generate $n=30$ from Gamma(3,2). Estimate mean and median, compute bootstrap SEs (B=5000), and compare with theoretical/asymptotic formulas.
