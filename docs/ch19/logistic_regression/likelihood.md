# Likelihood for Logistic Regression


## Setup

Given $n$ observations $(A[i,:],\; y^{(i)})$ where $y^{(i)}\in\{0,1\}$ and
$A$ is the $n\times(p+1)$ design matrix (with a column of ones for the
intercept), the predicted probability for observation $i$ is

$$
\sigma^{(i)} = \sigma\!\bigl(z^{(i)}\bigr),
\qquad
z^{(i)} = A[i,:]\,\boldsymbol{\theta}
$$

## Bernoulli Likelihood

Each label $y^{(i)}$ is modeled as a Bernoulli random variable with
success probability $\sigma^{(i)}$. The likelihood of the entire dataset
is

$$
\mathcal{L}(\boldsymbol{\theta})
= \prod_{i=1}^{n}\bigl[\sigma^{(i)}\bigr]^{y^{(i)}}
  \bigl[1-\sigma^{(i)}\bigr]^{1-y^{(i)}}
$$

## Cross-Entropy Loss (Negative Log-Likelihood)

Taking the negative logarithm gives the **cross-entropy loss** (also called
the **log-loss**):

$$
\ell = -\sum_{i=1}^{n}
  \Bigl[
    y^{(i)}\log\sigma^{(i)}

    + \bigl(1-y^{(i)}\bigr)\log\bigl(1-\sigma^{(i)}\bigr)
  \Bigr]
$$

Minimizing $\ell$ with respect to $\boldsymbol{\theta}$ is equivalent to
maximizing $\mathcal{L}$.

### Why Cross-Entropy?

The cross-entropy loss has two important properties that make it preferable
to, say, squared error for classification:

1. **Convexity.** $\ell$ is convex in $\boldsymbol{\theta}$, so every local
   minimum is a global minimum.
2. **Information-theoretic motivation.** The cross-entropy between the true
   distribution $y$ and the model distribution $\hat{y}=\sigma$ measures
   the additional bits needed to encode labels when using the model
   distribution instead of the true one.

### Behavior at the Extremes

When $y^{(i)}=1$ and $\sigma^{(i)}\to 0$, the term
$-\log\sigma^{(i)}\to+\infty$, heavily penalizing a confident wrong
prediction. Symmetrically for $y^{(i)}=0$ and $\sigma^{(i)}\to 1$. This
asymmetric penalty is exactly what drives the model toward well-calibrated
probabilities.

## Connection to Information Theory

If $q$ denotes the model distribution and $p$ the empirical (true)
distribution, the cross-entropy is

$$
H(p, q) = -\sum_x p(x)\log q(x)
$$

The cross-entropy decomposes as $H(p,q) = H(p) + D_{\mathrm{KL}}(p\|q)$,
where $H(p)$ is the entropy of the true distribution and
$D_{\mathrm{KL}}$ is the Kullback–Leibler divergence.  Since $H(p)$ is
constant with respect to $\boldsymbol{\theta}$, minimizing cross-entropy
is the same as minimizing the KL divergence.

## Numerical Stability

In practice a small constant $\varepsilon$ (e.g. $10^{-6}$) is added
inside the logarithm to avoid $\log 0$:

$$
\ell \approx -\sum_{i=1}^{n}
  \Bigl[
    y^{(i)}\log\bigl(\sigma^{(i)}+\varepsilon\bigr)

    + \bigl(1-y^{(i)}\bigr)\log\bigl(1-\sigma^{(i)}+\varepsilon\bigr)
  \Bigr]
$$

Alternatively, many frameworks compute the loss directly from the logits
$z^{(i)}$ using the numerically stable identity:

$$
-\log\sigma(z) = \log(1+e^{-z}) = \operatorname{softplus}(-z)
$$


## Exercises

**Exercise 1.**
Describe the main concept of Likelihood for Logistic Regression and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Likelihood for Logistic Regression is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

---

**Exercise 2.**
State the key assumptions required by the method discussed here. How can each assumption be checked?

??? success "Solution to Exercise 2"
    The main assumptions typically include: (1) independence of observations -- verified by understanding the data collection process and checking for serial correlation; (2) distributional requirements (e.g., normality) -- checked with Q-Q plots and formal tests like Shapiro-Wilk; (3) equal variances (if applicable) -- assessed with boxplots and Levene's test. When assumptions are violated, consider robust alternatives, transformations, or nonparametric methods.

---

**Exercise 3.**
Work through a small numerical example illustrating the application of the technique from this section.

??? success "Solution to Exercise 3"
    A structured approach to applying this technique involves: (1) clearly stating the hypotheses or estimation goal; (2) verifying that the data meet the required assumptions; (3) computing the relevant test statistic, estimate, or model fit; (4) obtaining the p-value, confidence interval, or posterior distribution; (5) interpreting the result in the context of the original question. Following these steps systematically ensures a rigorous and reproducible analysis.

---

**Exercise 4.**
Compare the approach from this section with an alternative method. When would you choose each?

??? success "Solution to Exercise 4"
    The method discussed here is appropriate when its assumptions hold and the sample size is sufficient for the asymptotic approximations to be accurate. Alternative approaches include: (1) nonparametric methods -- preferred when distributional assumptions are suspect; (2) bootstrap methods -- useful when analytical reference distributions are unavailable; (3) Bayesian methods -- valuable when incorporating prior information or when direct probability statements about parameters are desired. Running multiple approaches and comparing results provides a useful robustness check.
