# Logistic vs Linear Visualization


## Overview

When the response variable is binary, applying ordinary linear regression
produces predicted values that can fall outside the interval $[0,1]$, making
them invalid as probabilities.  Logistic regression solves this by passing the
linear predictor through the sigmoid function.  This page contrasts the two
approaches on a simulated credit-card default dataset and explains why logistic
regression is the appropriate choice for binary classification.

## Synthetic Data

We simulate $n = 300$ credit-card holders.  The probability of default
increases with account balance according to a logistic relationship:

$$
P(\text{Default} = 1 \mid \text{Balance}) = \frac{1}{1 + \exp\!\bigl(-({\text{Balance}} - 1250)/300\bigr)}
$$

```python
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

np.random.seed(42)
n_samples = 300
balance = np.random.uniform(0, 2500, n_samples)

true_prob = 1 / (1 + np.exp(-(balance - 1250) / 300))
default = np.random.binomial(1, true_prob)

X = balance.reshape(-1, 1)
y = default
X_test = np.linspace(balance.min(), balance.max(), 300).reshape(-1, 1)
```

## Linear Regression on Binary Data

Linear regression treats the binary outcome as a continuous variable and fits

$$
\hat{y} = \hat\beta_0 + \hat\beta_1 \cdot \text{Balance}
$$

by ordinary least squares.  Because a straight line extends indefinitely, the
predicted values inevitably fall outside $[0,1]$ for sufficiently large or
small values of Balance.

```python
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X_test)

print(f"Linear predictions range: "
      f"[{y_pred_linear.min():.3f}, {y_pred_linear.max():.3f}]")
```

These predictions are not valid probabilities whenever they are negative or
exceed 1.

## Logistic Regression on Binary Data

Logistic regression models the probability through the sigmoid function:

$$
P(Y = 1 \mid x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x)}}
$$

This guarantees that $\hat{p} \in (0,1)$ for any input.

```python
logistic_model = LogisticRegression(solver='lbfgs')
logistic_model.fit(X, y)
y_pred_logistic = logistic_model.predict_proba(X_test)[:, 1]

print(f"Logistic predictions range: "
      f"[{y_pred_logistic.min():.3f}, {y_pred_logistic.max():.3f}]")
```

## Side-by-Side Visualization

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: Linear regression
axes[0].scatter(X[y == 0], y[y == 0], alpha=0.6, s=30,
                color='steelblue', label='No Default (y=0)')
axes[0].scatter(X[y == 1], y[y == 1], alpha=0.6, s=30,
                color='coral', label='Default (y=1)')
axes[0].plot(X_test, y_pred_linear, 'g-', linewidth=2.5,
             label='Linear Fit')
axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
axes[0].axhline(y=1, color='black', linestyle='--', linewidth=0.8)
axes[0].set_xlabel('Credit Card Balance')
axes[0].set_ylabel('Predicted Probability')
axes[0].set_title('Linear Regression on Binary Data')
axes[0].legend(); axes[0].set_ylim(-0.5, 1.5)

# Right panel: Logistic regression
axes[1].scatter(X[y == 0], y[y == 0], alpha=0.6, s=30,
                color='steelblue', label='No Default (y=0)')
axes[1].scatter(X[y == 1], y[y == 1], alpha=0.6, s=30,
                color='coral', label='Default (y=1)')
axes[1].plot(X_test, y_pred_logistic, 'purple', linewidth=2.5,
             label='Logistic Fit')
axes[1].axhline(y=0.5, color='red', linestyle=':', linewidth=1.5,
                alpha=0.7, label='Decision boundary (0.5)')
axes[1].set_xlabel('Credit Card Balance')
axes[1].set_ylabel('Predicted Probability')
axes[1].set_title('Logistic Regression on Binary Data')
axes[1].legend(); axes[1].set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.show()
```

## Odds Ratio Interpretation

The logistic model provides an interpretable summary via the odds ratio.
A unit increase in Balance multiplies the odds of default by $e^{\hat\beta_1}$:

$$
\text{Odds Ratio} = e^{\hat\beta_1}
$$

```python
odds_ratio = np.exp(logistic_model.coef_[0][0])
print(f"Odds ratio per $1 increase: {odds_ratio:.4f}")
print(f"Percentage increase in odds per $100: "
      f"{(np.exp(100 * logistic_model.coef_[0][0]) - 1) * 100:.2f}%")
```

Linear regression has no analogous probabilistic interpretation: its slope
gives the change in $\hat{y}$ per unit increase in $x$, but $\hat{y}$ is not
constrained to be a probability.

## Interpretation

The key differences between the two approaches are:

| Property | Linear Regression | Logistic Regression |
|---|---|---|
| Predicted range | $(-\infty, +\infty)$ | $(0, 1)$ |
| Link function | Identity | Sigmoid |
| Loss function | Squared error | Cross-entropy |
| Coefficient meaning | Change in $\hat{y}$ | Change in log-odds |
| Valid for probabilities | No | Yes |

Linear regression is inappropriate for binary outcomes because it violates the
fundamental requirement that probabilities lie in $[0,1]$.  The sigmoid curve
of logistic regression respects this constraint and provides a natural
probabilistic interpretation through odds ratios.

## Exercises

**Exercise 1.**
For the logistic model $\log\frac{p}{1-p} = \beta_0 + \beta_1 x$, show that
the predicted probability at $x = -\beta_0/\beta_1$ is exactly 0.5.

??? success "Solution to Exercise 1"

    At $x = -\beta_0/\beta_1$:

    $$
    \log\frac{p}{1-p} = \beta_0 + \beta_1\Bigl(-\frac{\beta_0}{\beta_1}\Bigr) = \beta_0 - \beta_0 = 0
    $$

    Therefore $p/(1-p) = e^0 = 1$, which gives $p = 0.5$.  This is the
    **decision boundary**: the value of $x$ where the model is equally likely
    to predict either class. $\square$

---

**Exercise 2.**
Suppose a linear regression fitted to binary data produces $\hat{y} = -0.1$
for a certain observation.  Explain why this is problematic and describe two
ways to fix it.

??? success "Solution to Exercise 2"

    A predicted value of $\hat{y} = -0.1$ is a negative probability, which is
    undefined.  This violates the axioms of probability.

    Two ways to address this:

    1. **Use logistic regression.** The sigmoid function maps any real-valued
       linear predictor to $(0,1)$, guaranteeing valid probabilities.

    2. **Clip the predictions.** After fitting linear regression, truncate
       predictions to $[0,1]$: $\hat{p} = \max(0, \min(1, \hat{y}))$.
       However, this is an ad hoc fix and does not address the underlying
       model misspecification.  Logistic regression is strongly preferred. $\square$

---

**Exercise 3.**
Derive the sigmoid function by solving $\log\frac{p}{1-p} = z$ for $p$.

??? success "Solution to Exercise 3"

    Starting from $\log\frac{p}{1-p} = z$:

    $$
    \frac{p}{1-p} = e^z
    $$

    $$
    p = e^z(1 - p) = e^z - p\,e^z
    $$

    $$
    p + p\,e^z = e^z
    $$

    $$
    p(1 + e^z) = e^z
    $$

    $$
    p = \frac{e^z}{1 + e^z} = \frac{1}{1 + e^{-z}} = \sigma(z)
    $$

    The last step uses the identity $\frac{e^z}{1+e^z} = \frac{1}{1+e^{-z}}$,
    obtained by dividing numerator and denominator by $e^z$. $\square$

---

**Exercise 4.**
A linear regression on binary data yields $\hat{y} = 0.2 + 0.0003 \cdot \text{Balance}$.
At what Balance does the prediction exceed 1?  At what Balance does it become
negative?

??? success "Solution to Exercise 4"

    Setting $\hat{y} = 1$:

    $$
    0.2 + 0.0003 \cdot \text{Balance} = 1
    \implies \text{Balance} = \frac{0.8}{0.0003} \approx 2667
    $$

    Setting $\hat{y} = 0$:

    $$
    0.2 + 0.0003 \cdot \text{Balance} = 0
    \implies \text{Balance} = \frac{-0.2}{0.0003} \approx -667
    $$

    Since balance cannot be negative, the prediction becomes negative for
    unrealistic inputs.  However, the prediction exceeds 1 at Balance
    $\approx$ \$2667, which is a plausible value, demonstrating that linear
    regression produces invalid probabilities even within realistic input
    ranges. $\square$

---

**Exercise 5.**
Prove that the cross-entropy loss for logistic regression is convex in the
parameters $\boldsymbol\beta$.

??? success "Solution to Exercise 5"

    The negative log-likelihood (cross-entropy) for one observation is

    $$
    L_i(\boldsymbol\beta) = -y_i \log \sigma(\mathbf{x}_i^T\boldsymbol\beta)

      - (1-y_i)\log\bigl(1-\sigma(\mathbf{x}_i^T\boldsymbol\beta)\bigr)
    $$

    Using $\sigma(z) = 1/(1+e^{-z})$ and $1-\sigma(z) = \sigma(-z)$, this
    simplifies to

    $$
    L_i(\boldsymbol\beta) = -y_i\,\mathbf{x}_i^T\boldsymbol\beta

      + \log\bigl(1 + e^{\mathbf{x}_i^T\boldsymbol\beta}\bigr)
    $$

    The first term is linear in $\boldsymbol\beta$ (hence convex).  The second
    term is $\log(1+e^z)$ evaluated at $z = \mathbf{x}_i^T\boldsymbol\beta$.
    Since $\frac{d^2}{dz^2}\log(1+e^z) = \sigma(z)(1-\sigma(z)) > 0$ for all
    $z$, the function $\log(1+e^z)$ is convex.  A convex function composed
    with a linear map is convex.  The sum $L(\boldsymbol\beta) = \sum_i L_i$
    is a sum of convex functions and is therefore convex. $\square$
