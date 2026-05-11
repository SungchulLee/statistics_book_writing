# Efficiency of the Sample Mean

After establishing that an estimator is unbiased and consistent, a natural next question is: how precise can it be? Among all unbiased estimators of the same parameter, some achieve smaller variance than others. An estimator that attains the smallest possible variance — the Cramér–Rao lower bound — is called **efficient**. This section examines when and why the sample mean earns that title, and what happens when Normality fails.

## Definition of Efficiency

An unbiased estimator $\hat{\theta}$ of a parameter $\theta$ is **efficient** if its variance equals the Cramér–Rao lower bound (CRLB):

$$
\operatorname{Var}(\hat{\theta}) = \frac{1}{n \, I(\theta)}
$$

where $I(\theta)$ is the Fisher information for a single observation. Any unbiased estimator satisfying this equality has the smallest variance achievable among all unbiased estimators of $\theta$.

## CRLB for the Normal Mean

The Normal family $X \sim N(\mu, \sigma^2)$ satisfies the regularity conditions required for the CRLB to hold (the support does not depend on $\mu$, and the log-likelihood is twice differentiable with interchangeable expectation and differentiation). The Fisher information for a single observation is $I(\mu) = 1/\sigma^2$, so the CRLB gives:

$$
\operatorname{Var}(\hat{\mu}) \geq \frac{1}{n \, I(\mu)} = \frac{\sigma^2}{n}
$$

The sample mean $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$ has $\operatorname{Var}(\bar{X}) = \sigma^2 / n$, which matches the bound exactly. Because $\bar{X}$ is both the maximum likelihood estimator and the uniformly minimum variance unbiased estimator (UMVUE) for $\mu$ under Normality, it is efficient among all unbiased estimators of $\mu$.

## Asymptotic Relative Efficiency

The sample mean is efficient under Normality, but real data often deviate from the Normal model. Heavy-tailed or skewed distributions cause the sample mean to lose its efficiency advantage. The **asymptotic relative efficiency (ARE)** provides a way to compare two estimators by measuring how many observations one estimator needs relative to another to achieve the same precision.

For two estimators $T_1$ and $T_2$ of the same parameter, the ARE of $T_1$ relative to $T_2$ is defined as:

$$
\operatorname{ARE}(T_1, T_2) = \frac{\operatorname{Var}(T_2)}{\operatorname{Var}(T_1)}
$$

When $\operatorname{ARE}(T_1, T_2) > 1$, estimator $T_1$ is more efficient (needs fewer observations). When $\operatorname{ARE}(T_1, T_2) < 1$, estimator $T_2$ is more efficient.

### ARE of the Sample Mean vs the Median

Under a Normal distribution, the asymptotic variances are $\operatorname{Var}(\bar{X}) = \sigma^2/n$ and $\operatorname{Var}(\text{Median}) = \pi\sigma^2/(2n)$, giving:

$$
\operatorname{ARE}(\bar{X}, \text{Median}) = \frac{\operatorname{Var}(\text{Median})}{\operatorname{Var}(\bar{X})} = \frac{\pi}{2} \approx 1.57
$$

This means the sample mean is about 57% more efficient than the median under Normality: the median would require roughly 1.57 times as many observations to match the precision of $\bar{X}$.

However, the ranking reverses for heavy-tailed distributions. The following table summarizes the ARE of the sample mean relative to the median for several distributions:

| Distribution | $\operatorname{ARE}(\bar{X}, \text{Median})$ | Interpretation |
|---|---|---|
| Normal | $\pi/2 \approx 1.57$ | Mean is 57% more efficient |
| Double exponential (Laplace) | $2/3 \approx 0.67$ | Median is 50% more efficient |
| Cauchy | $0$ (mean has infinite variance) | Median is strictly preferred |

For the Laplace distribution, the median requires only two-thirds as many observations as the sample mean. For the Cauchy distribution, the sample mean has infinite variance, so it provides no useful information regardless of sample size — the median is the clear choice.

!!! tip "Practical Guidance"
    When the underlying distribution is approximately Normal, the sample mean is the best choice. When heavy tails or outliers are present, robust alternatives such as the trimmed mean or median offer better precision despite sacrificing some efficiency under Normality.

## Exercises

**Exercise 1.**
**ARE of median vs mean.** (a) Show ARE = $2/\pi$ for normal data. (b) For $t_3$ data, which wins?

??? success "Solution to Exercise 1"
    (a) Asymptotic variance of median: $1/[4 f(\mu)^2 n]$ where $f$ is density at the median. For $N(\mu, \sigma^2)$, $f(\mu) = 1/(\sigma\sqrt{2\pi})$, giving $\mathrm{AVar}(\text{median}) = \pi\sigma^2/(2n)$. Mean: $\sigma^2/n$. ARE = $2/\pi \approx 0.637$. Mean wins by factor 1.57.

    (b) For $t_3$: density at 0 is $\Gamma(2)/(\sqrt{3\pi}\Gamma(3/2)) \approx 0.367$. Mean variance is *infinite* (since $t_3$ has heavy tails barely beyond having a finite second moment). Median variance: finite. **Median wins dramatically for heavy tails.**

    Simulation confirms: for normal, mean variance $\approx 1/n$ vs median $\approx \pi/(2n)$. For $t_3$, mean variance is large (variance is $3$, so $\mathrm{Var}(\bar X) = 3/n$); median variance much smaller — and the difference grows with sample size.

---

**Exercise 2.**
**Shrinkage estimator.** $\hat\mu_\lambda = \lambda \bar X$. (a) Derive MSE. (b) Optimal $\lambda^*$. (c) Why can't we use $\lambda^*$ directly?

??? success "Solution to Exercise 2"
    (a) Bias = $(\lambda - 1)\mu$. Var = $\lambda^2 \sigma^2/n$. MSE = $(\lambda-1)^2 \mu^2 + \lambda^2 \sigma^2/n$.

    (b) $d\mathrm{MSE}/d\lambda = 2(\lambda - 1)\mu^2 + 2\lambda\sigma^2/n = 0 \Rightarrow \lambda^* = \mu^2/(\mu^2 + \sigma^2/n)$.

    Always $\lambda^* < 1$. Small when $\sigma^2/n$ is large (high noise) — shrink aggressively. Large (close to 1) when signal dominates.

    (c) $\lambda^*$ depends on unknown $\mu$. Plug-in $\hat\lambda$ has its own variability. James-Stein addresses this: an empirical-Bayes shrinkage factor that uses sample data adaptively and dominates the MLE for $p \ge 3$.

---

**Exercise 3.**
**James-Stein estimator.** For $\mathbf X \sim N(\boldsymbol\mu, I_p)$ with $p \ge 3$, compare $\hat{\boldsymbol\mu}_{\text{JS}} = (1 - (p-2)/\|\mathbf X\|^2)\mathbf X$ with MLE.

??? success "Solution to Exercise 3"
    ```python
    import numpy as np
    rng = np.random.default_rng(0)
    p, R = 10, 20_000
    mu = np.ones(p) * 0.5
    mse_mle = mse_js = 0.0
    for _ in range(R):
        x = mu + rng.standard_normal(p)
        js = (1 - (p - 2)/np.dot(x, x)) * x
        mse_mle += np.sum((x - mu)**2)
        mse_js += np.sum((js - mu)**2)
    print(f"MSE MLE={mse_mle/R:.3f}  JS={mse_js/R:.3f}")
    ```

    Expected: MSE(JS) < MSE(MLE) for every $\boldsymbol\mu$ when $p \ge 3$ (Stein, 1956). The MLE is **inadmissible** in dimension 3 or higher — there's always a better estimator.

    Practical impact: foundation of modern shrinkage (ridge regression, hierarchical Bayes, lasso). Even when components $\mu_i$ are unrelated, joint shrinkage improves total MSE.

---

**Exercise 4.**
**Efficient = achieving CRLB.** Show $\bar X$ for $N(\mu, \sigma^2)$ is efficient (CRLB-attaining) at every $n$, not just asymptotically.

??? success "Solution to Exercise 4"
    Fisher info for normal mean: $I(\mu) = 1/\sigma^2$ per observation, $nI(\mu) = n/\sigma^2$ total.

    CRLB: $\mathrm{Var}(\hat\mu) \ge 1/(nI(\mu)) = \sigma^2/n$.

    $\mathrm{Var}(\bar X) = \sigma^2/n$ — **achieves CRLB exactly**.

    Equality in CRLB is rare — typically MLEs achieve CRLB only asymptotically. $\bar X$ for normal mean is one of the few cases of exact efficiency for any $n$. This is because the score function $\partial \log f/\partial\mu = (X - \mu)/\sigma^2$ is linear in $X$, so the Cauchy-Schwarz inequality (which underlies CRLB derivation) holds with equality.

---

**Exercise 5.**
**Efficiency and sufficient statistics.** Connect efficiency to sufficiency: $\bar X$ is efficient because it is sufficient for $\mu$.

??? success "Solution to Exercise 5"
    By **Rao-Blackwell theorem:** any unbiased estimator can be improved by conditioning on a sufficient statistic.

    $\bar X$ is sufficient for $\mu$ (with $\sigma^2$ known): the likelihood factors as $L(\mu) = f(\bar X, \sigma^2/n) \cdot h(X_1, \ldots, X_n)$, with $h$ not depending on $\mu$.

    By **Lehmann-Scheffé:** the unique UMVUE is a function of a complete sufficient statistic. $\bar X$ is complete + sufficient + unbiased → UMVUE.

    **In words:** efficiency follows from sufficient (using all the data) + unbiased (no systematic error). $\bar X$ ticks both boxes.

    Inefficient estimators (like $X_1$) throw away information by not using the full sample.

---

**Exercise 6.**
**Trade-off in high dimensions.** Why do shrinkage estimators dominate MLE in high dimensions, but not in low dimensions?

??? success "Solution to Exercise 6"
    Risk of MLE in $p$-dim: $\mathrm{Risk}(\hat{\boldsymbol\mu}_{\text{MLE}}) = p\sigma^2$.

    JS shrinkage: $\mathrm{Risk}(\hat{\boldsymbol\mu}_{\text{JS}}) = p\sigma^2 - (p-2)^2 \mathbb{E}[1/\|\mathbf X\|^2]$.

    For $p \ge 3$, $(p-2)^2 > 0$, so JS dominates uniformly. For $p \le 2$, the correction is 0 or negative — MLE remains optimal.

    **Intuitive explanation:** in high dim, the MLE "explores" more directions, accumulating error. Even random shrinkage toward 0 (or any fixed point) reduces this excess error because most of the high-dim space is far from the truth.

    **Stein's phenomenon:** the MLE is inadmissible *only* in dimension 3+. The cutoff at $p = 3$ is precise — at $p = 2$, MLE is admissible.

    Practical consequence: in high-dimensional regression ($p$ predictors), shrinkage methods (ridge, lasso, elastic net) routinely outperform OLS, by similar mechanisms.
