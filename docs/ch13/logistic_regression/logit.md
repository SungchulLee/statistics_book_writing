# Logit Link and Odds

## From Linear Models to Classification

In linear regression the response variable $y$ is continuous. When the response
is binary — taking values 0 or 1 — we need a model that maps $\mathbb{R}$
into the interval $(0,1)$.  Logistic regression achieves this by passing the
linear predictor through the **sigmoid (logistic) function**.

## The Sigmoid Function

$$
\sigma(z) = \frac{1}{1+e^{-z}}
$$

The sigmoid maps every real number to $(0,1)$ and is therefore a valid model
for the conditional probability $P(Y=1 \mid \mathbf{x})$.

### Derivative of the Sigmoid

The derivative has a remarkably clean form that simplifies gradient
computations throughout logistic regression:

$$
\sigma'(z) = \frac{e^{-z}}{(1+e^{-z})^2} = \sigma(z)\bigl(1-\sigma(z)\bigr)
$$

??? note "Derivation"
    Write $\sigma = (1+e^{-z})^{-1}$ and apply the chain rule:

    $$
    \sigma' = -\,(1+e^{-z})^{-2}\cdot(-e^{-z})
            = \frac{e^{-z}}{(1+e^{-z})^2}
    $$

    Factor as $\frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}}
    = \sigma(1-\sigma)$.

## The Logit (Log-Odds) Link

Define the **logit** of a probability $p$:

$$
\operatorname{logit}(p) = \log\frac{p}{1-p}
$$

The ratio $p/(1-p)$ is the **odds** of the event, and the logit is the
**log-odds**.  Logistic regression assumes a linear relationship in the
log-odds:

$$
\operatorname{logit}\bigl(P(Y=1\mid\mathbf{x})\bigr) = \mathbf{x}^T\boldsymbol{\theta}
$$

Equivalently, for the $i$-th observation with feature vector
$A[i,:]$ (the design-matrix row, including a leading 1 for the intercept):

$$
z^{(i)} = A[i,:]\,\boldsymbol{\theta},
\qquad
\sigma^{(i)} = \sigma\!\bigl(z^{(i)}\bigr)
$$

## Interpretation via Odds

Because $\operatorname{logit}(p) = \mathbf{x}^T\boldsymbol{\theta}$, a
unit increase in feature $x_j$ multiplies the odds by $e^{\theta_j}$,
holding all other features constant.  This multiplicative interpretation
is one of the key reasons logistic regression remains popular in applied
statistics and finance.

## Key Properties

| Property | Value |
|---|---|
| Domain | $z \in (-\infty, +\infty)$ |
| Range | $\sigma(z) \in (0, 1)$ |
| Symmetry | $\sigma(-z) = 1 - \sigma(z)$ |
| Midpoint | $\sigma(0) = 0.5$ |
| Maximum slope | $\sigma'(0) = 0.25$ |
