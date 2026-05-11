# MoM vs MLE Comparison

## Why Compare These Two Methods?

Having developed the Method of Moments (MoM) and Maximum Likelihood Estimation (MLE) as separate frameworks, a natural question arises: when should we prefer one method over the other? Both approaches produce consistent estimators under suitable conditions, yet they differ in computational complexity, statistical efficiency, and sensitivity to model assumptions. Understanding these trade-offs is essential for making principled choices in practice.

## Side-by-Side Overview

The following table summarizes the key differences between MoM and MLE. Each criterion is discussed in detail in the sections that follow.

| Criterion | MoM | MLE |
|---|---|---|
| Computation | Often closed-form | May require numerical optimization |
| Efficiency | Generally less efficient | Asymptotically efficient (achieves CRLB) |
| Consistency | Yes (identifiability + finite moments) | Yes (under regularity conditions) |
| Invariance | Not in general | Yes (functional invariance) |
| Robustness | Less sensitive to distributional misspecification | Sensitive to model misspecification |

## Detailed Comparison

### Computation

MoM estimators are obtained by solving a system of equations that match sample moments to population moments. For many standard distributions, this yields closed-form expressions. MLE, by contrast, requires maximizing the log-likelihood function, which often demands iterative numerical methods such as Newton-Raphson or the EM algorithm.

In practice, MoM estimates frequently serve as starting values for iterative MLE algorithms, combining the computational ease of MoM with the statistical optimality of MLE.

### Efficiency

The most important theoretical distinction is **asymptotic efficiency**. Under standard regularity conditions, the MLE achieves the Cramer-Rao lower bound (CRLB) as $n \to \infty$, meaning no other consistent estimator has smaller asymptotic variance. In contrast, the MoM estimator generally does not achieve this bound.

The **asymptotic relative efficiency (ARE)** quantifies this gap. For estimators $\hat{\theta}_{\text{MoM}}$ and $\hat{\theta}_{\text{MLE}}$, the ARE is defined as

$$
\text{ARE}(\hat{\theta}_{\text{MoM}}, \hat{\theta}_{\text{MLE}}) = \frac{\text{Var}_{\text{asy}}(\hat{\theta}_{\text{MLE}})}{\text{Var}_{\text{asy}}(\hat{\theta}_{\text{MoM}})}
$$

An ARE less than 1 indicates that MLE is more efficient.

!!! example "Exponential Distribution: ARE Comparison"

    Let $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Exp}(\lambda)$. Both MoM and MLE yield the same estimator $\hat{\lambda} = 1/\bar{X}$, so the ARE equals 1. This is one case where MoM is fully efficient.

!!! example "Gamma Distribution: ARE Comparison"

    Let $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Gamma}(\alpha, \beta)$. The MoM estimators are

    $$
    \hat{\alpha}_{\text{MoM}} = \frac{\bar{X}^2}{S^2}, \qquad \hat{\beta}_{\text{MoM}} = \frac{S^2}{\bar{X}}
    $$

    where $S^2$ is the sample variance. The MLE requires solving a system involving the digamma function and has no closed form. For the shape parameter $\alpha$, the ARE of MoM relative to MLE can be substantially less than 1, especially when $\alpha$ is small. For instance, when $\alpha = 1$, the ARE for the shape parameter is approximately 0.64, meaning MoM requires roughly 56% more observations than MLE to achieve the same precision.

### Consistency

Both methods produce consistent estimators, but under different conditions:

- **MoM** requires that the moment equations uniquely identify the parameters (identifiability) and that the relevant population moments are finite.
- **MLE** requires a different set of regularity conditions, including that the parameter space is identifiable, the log-likelihood is sufficiently smooth, and the true parameter lies in the interior of the parameter space.

### Invariance

MLE has a powerful **functional invariance** property: if $\hat{\theta}$ is the MLE of $\theta$, then $g(\hat{\theta})$ is the MLE of $g(\theta)$ for any function $g$. This means we can freely reparametrize without re-deriving the estimator.

MoM does not enjoy this property in general. If $\hat{\theta}_{\text{MoM}}$ is the MoM estimator of $\theta$, applying a nonlinear transformation $g(\hat{\theta}_{\text{MoM}})$ does not necessarily produce the MoM estimator of $g(\theta)$.

### Robustness to Model Misspecification

MoM uses only a finite number of population moments and does not require specifying the entire distribution. If the assumed model is wrong but the first few moments are still correctly modeled, MoM can still yield reasonable estimates.

MLE, on the other hand, uses the full likelihood function and is therefore more sensitive to misspecification. When the assumed model is incorrect, the MLE converges to the parameter value that minimizes the Kullback-Leibler divergence from the true distribution to the assumed model family — which may not correspond to a meaningful quantity.

## When to Use Each Method

### Prefer MoM When

- **The likelihood is intractable.** Some models (e.g., certain mixture models or models defined via moment conditions) do not have a closed-form likelihood. MoM provides a viable alternative.
- **A quick initial estimate is needed.** MoM estimators serve as excellent starting values for iterative MLE algorithms.
- **Robustness to misspecification is important.** When confidence in the parametric model is limited, MoM estimates based on low-order moments may be more reliable.

### Prefer MLE When

- **Statistical efficiency is the priority.** For correctly specified models with large samples, MLE provides the most precise estimates.
- **Asymptotic inference is needed.** The well-developed asymptotic theory of MLE (Wald tests, likelihood ratio tests, score tests) makes it the natural choice when formal hypothesis testing or confidence intervals are required.
- **The invariance property is useful.** When interest lies in a transformed parameter $g(\theta)$, MLE avoids the need to re-derive the estimator.

!!! tip "Practical Strategy"

    A common workflow combines both methods: use MoM for a fast initial estimate, then refine with MLE for optimal efficiency. This hybrid approach leverages the strengths of each method while avoiding their individual weaknesses.

## Exercises

**Exercise 1.**
Geometric: $P(X = k) = (1-p)^{k-1} p$. (a) Log-likelihood for $x_1, \ldots, x_n$. (b) $\hat p_{\text{MLE}}$. (c) MoM. (d) Same?

??? success "Solution to Exercise 1"
    (a) $\ell(p) = \sum_i [(x_i - 1)\ln(1-p) + \ln p] = (T - n)\ln(1-p) + n\ln p$ where $T = \sum x_i$.

    (b) $\ell'(p) = -(T-n)/(1-p) + n/p = 0 \Rightarrow \hat p_{\text{MLE}} = n/T = 1/\bar x$.

    (c) $\bar x = 1/p \Rightarrow \hat p_{\text{MoM}} = 1/\bar x$.

    (d) **Same.** MLE and MoM coincide for the Geometric — both are reciprocals of the sample mean.

    This happens whenever the parameter $\theta$ is in 1-to-1 correspondence with the first moment and there are no higher-moment constraints. Most "single-parameter mean-determined" distributions (Bernoulli, Poisson, Exponential, Geometric) have MLE = MoM.

---

**Exercise 2.**
**Gamma simulation: MLE vs MoM.** Simulate from Gamma$(2, 3)$ at $n = 5, 10, 30, 100, 500$; compare MSE.

??? success "Solution to Exercise 2"
    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    R = 10_000
    for n in [5, 10, 30, 100, 500]:
        a_mom, b_mom, a_mle, b_mle = [], [], [], []
        for _ in range(R):
            x = rng.gamma(2.0, 3.0, n)
            a_mom.append(x.mean()**2 / x.var()); b_mom.append(x.var() / x.mean())
            am, _, bm = stats.gamma.fit(x, floc=0); a_mle.append(am); b_mle.append(bm)
        print(f"n={n}: MSE(α) MoM={np.mean((np.array(a_mom)-2)**2):.3f}, MLE={np.mean((np.array(a_mle)-2)**2):.3f}")
    ```

    Expected finding: MLE has smaller MSE at every $n$, with the gap widest for small $n$. Both decrease at rate $1/n$. MLE is asymptotically efficient; MoM is consistent but not efficient.

---

**Exercise 3.**
**Why MLE generally beats MoM.** State the asymptotic relative efficiency (ARE) result and the intuition.

??? success "Solution to Exercise 3"
    **ARE:** $\mathrm{ARE}(\hat\theta_{\text{MoM}}, \hat\theta_{\text{MLE}}) = $ asymptotic variance ratio $\le 1$.

    Equality (ARE = 1) iff MoM accidentally equals MLE (e.g., Bernoulli, Poisson, Exponential, where MLE and MoM coincide).

    Inequality (ARE < 1) is generic: MoM uses only low-order moments, MLE uses the full likelihood.

    **Intuition:** the Fisher information is the maximum possible information in the data about $\theta$, and MLE achieves it asymptotically. MoM throws away information by reducing the data to a few moments. The "thrown-away" information shows up as larger variance.

    **Practical implication:** prefer MLE when computationally feasible. MoM is useful for:

    - Quick initial estimates (start MLE optimization from MoM).
    - Distributions where MLE is intractable.
    - Robustness considerations (MoM may be less sensitive to model misspecification).

---

**Exercise 4.**
**When MoM is preferred.** Give two concrete cases where MoM might be chosen over MLE.

??? success "Solution to Exercise 4"
    **Case 1: MLE intractable.** For some distributions (e.g., Cauchy, generalized hyperbolic, certain copulas), the likelihood has no closed-form maximizer or is computationally expensive. MoM gives a quick closed-form estimate.

    **Case 2: Robustness to misspecification.** If the true distribution is a slight perturbation of the assumed model, MoM may degrade more gracefully than MLE. MLE is "locked into" the assumed likelihood; MoM only uses the assumption that population and sample moments match. A misspecified MLE can have arbitrarily large bias if the wrong likelihood is used; misspecified MoM at least matches the true population moments.

    **Case 3: Aggregating partial data.** When only sample moments are available (no individual data points), MoM works while MLE doesn't. Insurance claim reserves are sometimes computed from aggregate moments.

    These cases drive the continued use of MoM despite its asymptotic inefficiency.

---

**Exercise 5.**
**Pareto MoM vs MLE.** For $X \sim \mathrm{Pareto}(\alpha)$ on $x \ge 1$: derive both and compute the asymptotic relative efficiency.

??? success "Solution to Exercise 5"
    Pareto$(\alpha)$ on $[1, \infty)$: $f(x; \alpha) = \alpha x^{-(\alpha+1)}$, $\mathbb{E}[X] = \alpha/(\alpha - 1)$ for $\alpha > 1$.

    **MoM:** $\bar X = \alpha/(\alpha - 1) \Rightarrow \hat\alpha_{\text{MoM}} = \bar X/(\bar X - 1)$.

    **MLE:** $\hat\alpha_{\text{MLE}} = n/\sum \ln X_i$.

    Asymptotic variances (large $n$):

    $\mathrm{Var}(\hat\alpha_{\text{MoM}}) \approx \alpha^2(\alpha-1)^2(\alpha-2)/[n(\alpha-2)]$... actually this gets messy. The MoM only exists when $\mathbb{E}[X^2] < \infty$, i.e., $\alpha > 2$.

    $\mathrm{Var}(\hat\alpha_{\text{MLE}}) \to \alpha^2/n$ (CRLB).

    **ARE:** for $\alpha = 3$ (a typical case where MoM is defined), MoM variance is much larger than MLE. For $\alpha \le 2$, MoM doesn't even exist (sample variance is infinite).

    **Implication:** for heavy-tailed distributions, MLE is essential — MoM may not even be defined.

---

**Exercise 6.**
**MoM as starting value for MLE.** Why is using $\hat\theta_{\text{MoM}}$ as an initial value for iterative MLE optimization a good idea?

??? success "Solution to Exercise 6"
    Iterative MLE (Newton-Raphson, BFGS, EM) requires a starting value. Good starting values:

    1. **Converge faster** (fewer iterations to reach optimum).
    2. **Avoid local optima** (multimodal likelihoods can trap iterations at a non-global maximum).
    3. **Avoid numerical issues** (extreme parameter values can cause overflow/underflow).

    **MoM as starting point** is appealing because:

    - **Closed-form** (no iteration needed for the initial estimate).
    - **Often consistent** (close to the true $\theta$ in large samples).
    - **Generally near the MLE** (both estimators target the same $\theta$).

    Hybrid procedure:

    ```python
    theta_init = mom_estimate(data)
    theta_mle = scipy.optimize.minimize(neg_loglik, theta_init, ...).x
    ```

    This is the standard recipe in `scipy.stats.fit` and similar libraries.
