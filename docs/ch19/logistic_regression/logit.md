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

## Confounding in Logistic Regression

As with linear regression, logistic regression can suffer from **confounding**: a third variable may be associated with both the predictor and the outcome, distorting the apparent relationship between them.

### Example: Student Status Confounds Balance and Default

Consider predicting credit card default ($Y$) using account balance ($X_1$). A bivariate logistic regression shows a negative relationship:

$$
\log\frac{P(\text{Default}=1)}{P(\text{Default}=0)} = \beta_0 + \beta_1 \cdot \text{Balance}
$$

You might find $\beta_1 < 0$, suggesting that higher balance reduces default risk. However, this may be **misleading**.

The confounder is **student status** ($X_2$). In the data:

- **Students** tend to have:
  - Lower account balances (younger, less established financially)
  - Higher default rates (lower income, less stable employment)

- **Non-students** tend to have:
  - Higher account balances (older, more established)
  - Lower default rates (higher income, better job stability)

### The Paradox

**Bivariate model** (ignoring student status):
$$
\log\frac{P(\text{Default}=1)}{1-P(\text{Default}=1)} = \beta_0 + \beta_1 \cdot \text{Balance}
$$

This shows $\beta_1 < 0$: higher balance → lower default probability.

**But this is confounded!** Student status is driving both variables in opposite directions.

**Multivariate model** (including student status):
$$
\log\frac{P(\text{Default}=1)}{1-P(\text{Default}=1)} = \beta_0 + \beta_1 \cdot \text{Balance} + \beta_2 \cdot \text{Student}
$$

Once we control for student status, the relationship between balance and default may reverse or change dramatically:

- $\beta_1$ might become **positive** (higher balance → higher default for students AND non-students)
- The coefficient magnitude may change substantially

This reversal is an example of **Simpson's paradox** in classification: an association observed in the marginal (bivariate) model disappears or reverses when you control for a confounder.

### Interpretation of Odds Ratios with Confounders

In the bivariate model:
$$
\text{Odds Ratio}_{X_1} = e^{\beta_1} \approx 0.99
$$
("For every $1 increase in balance, odds decrease by 1%")

In the multivariate model:
$$
\text{Odds Ratio}_{X_1 | X_2} = e^{\hat{\beta}_1} \approx 1.01
$$
("For every $1 increase in balance, *controlling for student status*, odds increase by 1%")

The different interpretations reflect that:
- The first is a **marginal** (unconditional) effect
- The second is a **conditional** (partial) effect

### Checking for Confounding

To detect confounding in logistic regression:

1. **Fit a bivariate model:** $\log\frac{p}{1-p} = \beta_0 + \beta_1 X_1$
2. **Fit a multivariate model:** $\log\frac{p}{1-p} = \beta_0 + \beta_1 X_1 + \beta_2 X_2$
3. **Compare coefficients:**
   - If $|\hat{\beta}_1^{\text{multivariate}} - \hat{\beta}_1^{\text{bivariate}}| / |\hat{\beta}_1^{\text{bivariate}}| > 0.10$, confounding is likely present
   - A change in sign is strong evidence of confounding

### Example Output

```
Bivariate Model (Balance only):
  Intercept:  -10.65
  Balance:     -0.00555  (Odds Ratio: 0.9945)

Multivariate Model (Balance + Student):
  Intercept:  -11.10
  Balance:     +0.00265  (Odds Ratio: 1.0027)
  Student:     +0.71    (Odds Ratio: 2.03)
```

The balance coefficient changes from negative to positive once student status is included—a clear sign of confounding.

### Implications for Prediction and Inference

- **For prediction:** Including confounders improves predictive accuracy by capturing true relationships
- **For inference:** Ignoring confounders leads to biased coefficient estimates and incorrect odds ratio interpretations
- **Best practice:** Always consider domain knowledge and collect data on potential confounders

See also: [Confounding and Association vs. Causation](../../ch01/classical/confounding_causation.md) for a broader discussion of confounding across statistical methods.


