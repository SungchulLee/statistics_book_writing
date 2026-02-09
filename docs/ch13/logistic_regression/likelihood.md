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
