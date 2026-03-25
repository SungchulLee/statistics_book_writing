# MSE of Variance Estimators

## Motivation

In previous sections, we encountered two natural estimators for the population variance: dividing by $n$ (the MLE) and dividing by $n-1$ (Bessel's correction). The MLE is biased but has smaller variance, while Bessel's estimator is unbiased but more variable. Mean squared error (MSE) provides a single criterion that balances bias and variance, letting us ask: is there a divisor that minimizes overall estimation error?

## Setup

Assume $X_1, X_2, \ldots, X_n \overset{iid}{\sim} N(\mu, \sigma^2)$. Define the sum of squared deviations

$$
Q = \sum_{i=1}^n (X_i - \bar{X})^2
$$

Under normality, $Q / \sigma^2 \sim \chi^2_{n-1}$. This distributional result gives us the moments we need:

$$
E[Q] = (n-1)\sigma^2, \quad \text{Var}(Q) = 2(n-1)\sigma^4
$$

Any estimator of $\sigma^2$ that takes the form $\hat{\sigma}^2 = cQ$ for a constant $c > 0$ has

$$
\text{Bias}(cQ) = E[cQ] - \sigma^2 = [c(n-1) - 1]\,\sigma^2
$$

$$
\text{Var}(cQ) = c^2 \,\text{Var}(Q) = 2c^2(n-1)\,\sigma^4
$$

$$
\text{MSE}(cQ) = \text{Var}(cQ) + \text{Bias}^2(cQ) = \left[2c^2(n-1) + \bigl(c(n-1)-1\bigr)^2\right]\sigma^4
$$

## Three Estimators

Substituting the three common choices of $c$ into the formulas above produces the following comparison:

| Estimator | Divisor | Bias | MSE |
|---|---|---|---|
| MLE | $n$ | $-\sigma^2/n$ | $\dfrac{2n-1}{n^2}\,\sigma^4$ |
| Bessel's correction | $n-1$ | $0$ | $\dfrac{2}{n-1}\,\sigma^4$ |
| MSE-optimal | $n+1$ | $-\dfrac{2\sigma^2}{n+1}$ | $\dfrac{2}{n+1}\,\sigma^4$ |

Each entry follows from plugging $c = 1/n$, $c = 1/(n-1)$, or $c = 1/(n+1)$ into the general MSE formula derived above.

## Deriving the Optimal Divisor

Among all estimators of the form $cQ$, which value of $c$ minimizes MSE? Expanding the MSE expression:

$$
\text{MSE}(c) = \left[2c^2(n-1) + c^2(n-1)^2 - 2c(n-1) + 1\right]\sigma^4
$$

Differentiating with respect to $c$ and setting the result to zero:

$$
\frac{d}{dc}\,\text{MSE}(c) = \left[4c(n-1) + 2c(n-1)^2 - 2(n-1)\right]\sigma^4 = 0
$$

Factoring out $2(n-1)$:

$$
2(n-1)\left[2c + c(n-1) - 1\right] = 0
$$

Since $n \geq 2$, we can divide by $2(n-1)$ to get $c(n+1) = 1$, so

$$
c^* = \frac{1}{n+1}
$$

This confirms that dividing by $n+1$ yields the smallest MSE among all estimators of this form.

## Numerical Example

To see the practical difference, consider $n = 10$ and $\sigma^2 = 1$. The MSE values for the three estimators are:

| Estimator | MSE formula | MSE value |
|---|---|---|
| MLE ($c = 1/10$) | $(2 \cdot 10 - 1)/100$ | $0.190$ |
| Bessel ($c = 1/9$) | $2/9$ | $0.222$ |
| MSE-optimal ($c = 1/11$) | $2/11$ | $0.182$ |

The MSE-optimal estimator reduces MSE by about 4% compared to the MLE and by about 18% compared to Bessel's correction. The improvement over the MLE is modest because the MLE is already nearly optimal -- it is only slightly too biased. Bessel's correction, while unbiased, pays a meaningful variance penalty.

## Key Insight

The MSE-optimal divisor $n+1$ demonstrates the bias-variance tradeoff in a clean, closed-form setting. By accepting a small bias of $-2\sigma^2/(n+1)$, the estimator achieves a variance reduction that more than compensates. As the sample size $n$ grows, all three estimators converge to the same value, and the differences in MSE become negligible -- but for small samples, the choice of divisor matters.

!!! tip "When to use which estimator"
    In practice, Bessel's correction ($n-1$ divisor) remains the default because unbiasedness simplifies theoretical analysis and is expected by downstream procedures like $t$-tests. The MSE-optimal estimator is primarily of theoretical interest, illustrating that unbiasedness is not always the best criterion.
