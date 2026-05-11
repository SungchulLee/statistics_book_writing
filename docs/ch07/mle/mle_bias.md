# Bias of the Gaussian MLE for Variance

## Why This Example Matters

Maximum likelihood estimation has many desirable asymptotic properties — consistency, normality, and efficiency. However, in finite samples, the MLE can be biased. The normal variance estimator is the classic illustration: dividing by $n$ instead of $n-1$ systematically underestimates the true variance. This example teaches two important lessons — that optimality in the MLE sense does not guarantee unbiasedness, and that the trade-off between bias and variance is a recurring theme in estimation theory.

## The MLE for Normal Variance

Let $X_1, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$ with both $\mu$ and $\sigma^2$ unknown. Maximizing the log-likelihood with respect to $\sigma^2$ yields the MLE:

$$
\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2
$$

This is the average squared deviation from the sample mean. Intuitively, it seems like a natural estimator, but it turns out to be biased downward.

## Deriving the Bias

To compute $E[\hat{\sigma}^2_{\text{MLE}}]$, we use a standard algebraic decomposition. Start by adding and subtracting the population mean $\mu$:

$$
\sum_{i=1}^n (X_i - \bar{X})^2 = \sum_{i=1}^n \left[(X_i - \mu) - (\bar{X} - \mu)\right]^2
$$

Expanding the square:

$$
= \sum_{i=1}^n (X_i - \mu)^2 - 2(\bar{X} - \mu)\sum_{i=1}^n (X_i - \mu) + n(\bar{X} - \mu)^2
$$

Since $\sum_{i=1}^n (X_i - \mu) = n(\bar{X} - \mu)$, the middle term equals $-2n(\bar{X} - \mu)^2$, giving

$$
\sum_{i=1}^n (X_i - \bar{X})^2 = \sum_{i=1}^n (X_i - \mu)^2 - n(\bar{X} - \mu)^2
$$

Now take expectations of both sides. For the first term, each $(X_i - \mu)^2$ has expectation $\sigma^2$, so the sum has expectation $n\sigma^2$. For the second term, $\bar{X} \sim N(\mu, \sigma^2/n)$, so $E[(\bar{X} - \mu)^2] = \sigma^2/n$ and $E[n(\bar{X} - \mu)^2] = \sigma^2$. Therefore:

$$
E\left[\sum_{i=1}^n (X_i - \bar{X})^2\right] = n\sigma^2 - \sigma^2 = (n-1)\sigma^2
$$

Dividing by $n$:

$$
E[\hat{\sigma}^2_{\text{MLE}}] = \frac{n-1}{n}\sigma^2
$$

The bias of the MLE is

$$
\text{Bias}(\hat{\sigma}^2_{\text{MLE}}) = E[\hat{\sigma}^2_{\text{MLE}}] - \sigma^2 = -\frac{\sigma^2}{n}
$$

The MLE systematically underestimates the true variance. The bias is negative and has magnitude $\sigma^2/n$, which decreases with sample size. However, vanishing bias alone does not guarantee consistency — we also need the variance of the estimator to vanish, which it does (at rate $1/n$), so the MLE is indeed consistent.

## Bessel's Correction

The derivation above reveals exactly where the bias comes from: using $\bar{X}$ instead of the true mean $\mu$ costs one degree of freedom. Estimating $\mu$ from the data means the deviations $X_i - \bar{X}$ satisfy the constraint $\sum_{i=1}^n (X_i - \bar{X}) = 0$, so only $n - 1$ of them are free.

**Bessel's correction** divides by $n - 1$ instead of $n$ to produce an unbiased estimator:

$$
S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2
$$

By the calculation above, $E[S^2] = \sigma^2$ exactly. This is the sample variance used in most statistical software.

## MSE Comparison

Unbiasedness is not the only criterion for a good estimator. The **mean squared error (MSE)** balances bias and variance:

$$
\text{MSE}(\hat{\sigma}^2) = \text{Bias}^2(\hat{\sigma}^2) + \text{Var}(\hat{\sigma}^2)
$$

For the normal distribution, the MSE of estimators of the form $\hat{\sigma}^2_c = \frac{1}{c}\sum_{i=1}^n (X_i - \bar{X})^2$ can be computed in closed form.

**Unbiased estimator** ($c = n - 1$):

$$
\text{MSE}(S^2) = \frac{2\sigma^4}{n-1}
$$

**MLE** ($c = n$):

$$
\text{MSE}(\hat{\sigma}^2_{\text{MLE}}) = \frac{2n - 1}{n^2}\,\sigma^4
$$

**MSE-optimal estimator** ($c = n + 1$):

$$
\text{MSE}\!\left(\frac{1}{n+1}\sum(X_i - \bar{X})^2\right) = \frac{2\sigma^4}{n+1}
$$

!!! example "Numerical Comparison for n = 10"

    With $n = 10$ and $\sigma^2 = 1$:

    | Estimator | Divisor | Bias | MSE |
    |---|---|---|---|
    | $S^2$ | $n - 1 = 9$ | 0 | $2/9 \approx 0.222$ |
    | $\hat{\sigma}^2_{\text{MLE}}$ | $n = 10$ | $-0.1$ | $19/100 = 0.190$ |
    | MSE-optimal | $n + 1 = 11$ | $-2/11 \approx -0.182$ | $2/11 \approx 0.182$ |

    The MLE has lower MSE than $S^2$ despite being biased. The MSE-optimal estimator (dividing by $n + 1$) does even better, accepting more bias in exchange for substantially lower variance.

!!! tip "Bias-Variance Trade-Off"

    This example illustrates a general principle: a small amount of bias can be worth accepting if it comes with a sufficient reduction in variance. The MSE-optimal estimator is neither the MLE nor the unbiased estimator — it strikes the best balance between the two.

## Exercises

**Exercise 1.**
Show that the MLE of the normal variance $\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n(X_i - \bar{X})^2$ is biased. Compute the exact bias.

??? success "Solution to Exercise 1"
    We know $\sum_{i=1}^n(X_i - \bar{X})^2 / \sigma^2 \sim \chi^2_{n-1}$, which has mean $n - 1$. Therefore:

    $$
    E\!\left[\sum_{i=1}^n(X_i - \bar{X})^2\right] = (n-1)\sigma^2
    $$

    $$
    E[\hat{\sigma}^2_{\text{MLE}}] = \frac{1}{n}(n-1)\sigma^2 = \frac{n-1}{n}\sigma^2
    $$

    The bias is:

    $$
    \text{Bias} = E[\hat{\sigma}^2_{\text{MLE}}] - \sigma^2 = -\frac{\sigma^2}{n}
    $$

    The MLE underestimates $\sigma^2$ on average. The bias vanishes as $n \to \infty$ but is noticeable for small $n$.

---

**Exercise 2.**
The unbiased estimator $S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$ corrects the bias. Show that $\text{MSE}(\hat{\sigma}^2_{\text{MLE}}) < \text{MSE}(S^2)$ for $n \geq 2$, so the biased MLE actually has smaller mean squared error.

??? success "Solution to Exercise 2"
    For any estimator $\hat{\theta}$, $\text{MSE} = \text{Bias}^2 + \text{Var}$.

    Since $\sum(X_i - \bar{X})^2/\sigma^2 \sim \chi^2_{n-1}$ has variance $2(n-1)$:

    $$
    \text{Var}(S^2) = \frac{\sigma^4}{(n-1)^2} \cdot 2(n-1) = \frac{2\sigma^4}{n-1}
    $$

    $$
    \text{MSE}(S^2) = 0 + \frac{2\sigma^4}{n-1}
    $$

    For the MLE:

    $$
    \text{Var}(\hat{\sigma}^2_{\text{MLE}}) = \frac{\sigma^4}{n^2} \cdot 2(n-1) = \frac{2(n-1)\sigma^4}{n^2}
    $$

    $$
    \text{MSE}(\hat{\sigma}^2_{\text{MLE}}) = \frac{\sigma^4}{n^2} + \frac{2(n-1)\sigma^4}{n^2} = \frac{(2n-1)\sigma^4}{n^2}
    $$

    Comparing: $\frac{2n-1}{n^2} < \frac{2}{n-1}$ simplifies to $(2n-1)(n-1) < 2n^2$, i.e., $2n^2 - 3n + 1 < 2n^2$, i.e., $-3n + 1 < 0$, which holds for all $n \geq 1$. $\square$

---

**Exercise 3.**
For $n = 5$ and $\sigma^2 = 10$, compute the bias, variance, and MSE of both $\hat{\sigma}^2_{\text{MLE}}$ and $S^2$.

??? success "Solution to Exercise 3"
    For $\hat{\sigma}^2_{\text{MLE}}$ with $n = 5$, $\sigma^2 = 10$:

    - Bias: $-10/5 = -2.0$
    - Variance: $2(4)(100)/25 = 32.0$
    - MSE: $4 + 32 = 36.0$

    For $S^2$:

    - Bias: $0$
    - Variance: $2(100)/4 = 50.0$
    - MSE: $0 + 50 = 50.0$

    The MLE has MSE = 36, which is 28% smaller than the unbiased $S^2$ with MSE = 50. The bias-variance trade-off favors the biased estimator here: the reduction in variance (from 50 to 32) more than compensates for the introduced bias.

---

**Exercise 4.**
The estimator $\hat{\sigma}^2_c = \frac{1}{n+1}\sum(X_i - \bar{X})^2$ has even smaller MSE than $\hat{\sigma}^2_{\text{MLE}}$ for the normal distribution. Verify this by computing its MSE and comparing with the MLE.

??? success "Solution to Exercise 4"
    For the general estimator $\hat{\sigma}^2_c = \frac{1}{c}\sum(X_i - \bar{X})^2$ with $c = n + 1$:

    $$
    E[\hat{\sigma}^2_c] = \frac{n-1}{n+1}\sigma^2, \quad \text{Bias} = \frac{n-1}{n+1}\sigma^2 - \sigma^2 = -\frac{2\sigma^2}{n+1}
    $$

    $$
    \text{Var}(\hat{\sigma}^2_c) = \frac{2(n-1)\sigma^4}{(n+1)^2}
    $$

    $$
    \text{MSE}(\hat{\sigma}^2_c) = \frac{4\sigma^4}{(n+1)^2} + \frac{2(n-1)\sigma^4}{(n+1)^2} = \frac{(2n+2)\sigma^4}{(n+1)^2} = \frac{2\sigma^4}{n+1}
    $$

    Comparing with the MLE's MSE = $(2n-1)\sigma^4/n^2$: for $n = 5$, the MLE gives $9 \times 100/25 = 36$ while $c = 6$ gives $200/6 \approx 33.3$. The MSE-optimal divisor is actually $c = n + 1$ (for normal data), which minimizes MSE at $2\sigma^4/(n+1)$.
