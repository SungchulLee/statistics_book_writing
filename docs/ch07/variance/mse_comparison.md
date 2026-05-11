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

## Exercises

**Exercise 1.**
For $\hat\sigma^2_c = (1/c)\sum(X_i - \bar X)^2$ on normal data: (a) derive $\mathrm{MSE}$; (b) find optimal $c^*$; (c) verify for $n = 10$.

??? success "Solution to Exercise 1"
    (a) With $W = \sum(X_i - \bar X)^2 \sim \sigma^2 \chi^2_{n-1}$: $\mathbb{E}[\hat\sigma^2_c] = (n-1)\sigma^2/c$, $\mathrm{Var}(\hat\sigma^2_c) = 2(n-1)\sigma^4/c^2$.

    $\mathrm{MSE}(c) = (\sigma^4/c^2)[(n-1-c)^2 + 2(n-1)]$.

    (b) Differentiate and set to 0: $c^* = n + 1$.

    (c) $n = 10$, $\sigma = 1$: $\mathrm{MSE}(9) \approx 0.222$, $\mathrm{MSE}(10) = 0.190$, $\mathrm{MSE}(11) \approx 0.182$. $c^* = 11$ wins.

---

**Exercise 2.**
**Compare the three divisors** (n, n-1, n+1) for variance estimation.

??? success "Solution to Exercise 2"
    | Estimator | Divisor | Bias | $\mathrm{Var}/\sigma^4$ | $\mathrm{MSE}/\sigma^4$ |
    |---|---|---|---|---|
    | $\hat\sigma^2_{\text{MLE}}$ | $n$ | $-\sigma^2/n$ | $2(n-1)/n^2$ | $(2n-1)/n^2$ |
    | $S^2$ (unbiased) | $n-1$ | 0 | $2/(n-1)$ | $2/(n-1)$ |
    | $\hat\sigma^2_{c^*}$ | $n+1$ | $-2\sigma^2/(n+1)$ | $2(n-1)/(n+1)^2$ | $2/(n+1)$ |

    **Minimum MSE:** divisor $n+1$ beats both MLE and unbiased. The trade-off: a small bias is accepted to gain larger variance reduction.

    Despite winning on MSE, the $n+1$ divisor is rarely used because $S^2$ has the cleaner interpretation (unbiased) and the MSE difference is small for moderate $n$.

---

**Exercise 3.**
**MSE depends on the loss function.** Why is MSE the standard, and what alternatives exist?

??? success "Solution to Exercise 3"
    MSE = squared-error loss: $L(\hat\theta, \theta) = (\hat\theta - \theta)^2$. Popular because:

    - Mathematically tractable (linearity of expectation, decomposition into bias + variance).
    - Differentiable everywhere — analytic optimization.
    - Penalizes large errors heavily (squared).

    **Alternatives:**

    - **Absolute error:** $|\hat\theta - \theta|$. Minimized by posterior median, more robust to outliers.
    - **Quantile loss:** $\rho_\tau(\hat\theta - \theta) = (\hat\theta - \theta)(\tau - \mathbf 1\{\hat\theta < \theta\})$. Used in quantile regression.
    - **0-1 loss:** $\mathbf 1\{\hat\theta \ne \theta\}$. For classification.
    - **Log-loss / KL divergence:** for density estimation, probabilistic forecasting.

    Choice of loss reflects what kinds of errors are penalized. MSE is the *default* but not always the *right* choice.

---

**Exercise 4.**
**Shrinkage and James-Stein.** The MLE for multivariate normal mean is the sample mean for each component. James-Stein shows this is **inadmissible** when $p \ge 3$. Sketch the idea.

??? success "Solution to Exercise 4"
    Estimate $\boldsymbol\mu \in \mathbb{R}^p$ from $\mathbf X \sim N(\boldsymbol\mu, I_p)$ (single observation). MLE: $\hat{\boldsymbol\mu} = \mathbf X$. MSE: $\mathbb{E}[\|\mathbf X - \boldsymbol\mu\|^2] = p$.

    **James-Stein estimator:** $\hat{\boldsymbol\mu}_{\text{JS}} = (1 - (p-2)/\|\mathbf X\|^2) \mathbf X$ (shrinkage toward 0).

    Stein (1956) showed: $\mathrm{MSE}(\hat{\boldsymbol\mu}_{\text{JS}}) < p$ for **every** $\boldsymbol\mu$ when $p \ge 3$. The MLE is **dominated** — there's always a better estimator.

    Intuition: even when shrinking toward an *arbitrary* point, the JS estimator does better than MLE in MSE. The bias introduced is more than compensated by variance reduction in high dimensions.

    Modern statistics: shrinkage estimators (ridge, lasso, James-Stein) routinely beat MLEs on MSE for high-dimensional problems. Empirical Bayes and hierarchical models formalize this.

---

**Exercise 5.**
**Sample size and MSE.** Show that quadrupling $n$ approximately halves $\sqrt{\mathrm{MSE}}$ for $S^2$.

??? success "Solution to Exercise 5"
    $\mathrm{MSE}(S^2) = 2\sigma^4/(n-1)$. Square root: $\sqrt{\mathrm{MSE}} = \sigma^2 \sqrt{2/(n-1)}$.

    For $n' = 4n$: $\sqrt{\mathrm{MSE}'} = \sigma^2 \sqrt{2/(4n - 1)} \approx \sigma^2 \sqrt{2/(n-1)}/2 = \sqrt{\mathrm{MSE}}/2$.

    Halving as $n$ quadruples. This is the same $\sqrt n$ scaling as for $\bar X$, applied to RMSE of variance estimation.

---

**Exercise 6.**
**Asymptotic optimality.** Show that as $n \to \infty$, all three estimators ($n$, $n-1$, $n+1$ divisors) have the same asymptotic MSE.

??? success "Solution to Exercise 6"
    All three:

    - $\mathrm{MSE}(\hat\sigma^2_{\text{MLE}})/\sigma^4 = (2n-1)/n^2 \to 2/n$.
    - $\mathrm{MSE}(S^2)/\sigma^4 = 2/(n-1) \to 2/n$.
    - $\mathrm{MSE}(\hat\sigma^2_{c^*})/\sigma^4 = 2/(n+1) \to 2/n$.

    Asymptotically equivalent — to first order in $1/n$, all three have MSE $\approx 2\sigma^4/n$.

    **Implication:** the choice of divisor matters only for finite samples. For large $n$, just use whichever is most convenient. For small $n$ where the difference is non-negligible, choose based on loss function: unbiasedness ($n-1$), MLE convenience ($n$), or minimum MSE ($n+1$).
