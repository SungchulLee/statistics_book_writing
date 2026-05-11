# Logistic Regression Demonstrations


## Overview

This page demonstrates logistic regression in Python using both scikit-learn and
statsmodels.  We generate synthetic data relating hours studied to exam outcome
(pass/fail), fit the model, compute odds ratios and confidence intervals, perform
a likelihood ratio test, and evaluate predictions using a confusion matrix, ROC
curve, precision-recall curve, and threshold selection via Youden's J statistic.

## Data Generation

A binary outcome $y_i \in \{0,1\}$ is generated from a latent linear model
passed through the sigmoid function:

$$
z_i = -3 + 0.7\,x_i + 0.3\,\varepsilon_i, \qquad
p_i = \frac{1}{1+e^{-z_i}}, \qquad
y_i \sim \operatorname{Bernoulli}(p_i)
$$

where $x_i$ is hours studied drawn uniformly on $[1,10]$ and
$\varepsilon_i \sim N(0,1)$.

```python
import numpy as np

np.random.seed(42)
n = 300
hours_studied = np.random.uniform(1, 10, n)
noise = np.random.normal(0, 1, n)
logit = -3 + 0.7 * hours_studied + 0.3 * noise
prob = 1 / (1 + np.exp(-logit))
passed = np.random.binomial(1, prob)

X = hours_studied.reshape(-1, 1)
y = passed
```

## Fitting with scikit-learn

Scikit-learn's `LogisticRegression` maximizes the penalized log-likelihood by
default (C = 1.0, L2 penalty).  The estimated intercept and slope map directly
to the logistic model:

$$
\log\frac{P(Y=1\mid x)}{1-P(Y=1\mid x)} = \hat\beta_0 + \hat\beta_1\,x
$$

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

print(f"Intercept: {model.intercept_[0]:.4f}")
print(f"Coefficient (hours): {model.coef_[0][0]:.4f}")
print(f"Train accuracy: {model.score(X_train, y_train):.3f}")
print(f"Test accuracy:  {model.score(X_test, y_test):.3f}")

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)
```

## Inference with statsmodels

Statsmodels provides standard errors, Wald tests, and confidence intervals via
maximum likelihood estimation (no regularization by default).

```python
import statsmodels.api as sm
from scipy import stats

X_sm = sm.add_constant(hours_studied)
logit_model = sm.Logit(y, X_sm)
result = logit_model.fit(disp=0)
print(result.summary())
```

### Odds Ratios

Because the logit link is the log of the odds, exponentiating the coefficients
yields odds ratios:

$$
\text{OR}_j = e^{\hat\beta_j}
$$

A one-unit increase in hours studied multiplies the odds of passing by
$e^{\hat\beta_1}$.

```python
import numpy as np

print("Odds Ratios:")
print(np.exp(result.params))

print("95% CI for Odds Ratios:")
print(np.exp(result.conf_int()))
```

### Likelihood Ratio Test

The likelihood ratio test compares the fitted model to the null (intercept-only)
model:

$$
\Lambda = -2\bigl[\ell(\hat{\boldsymbol\beta}_0) - \ell(\hat{\boldsymbol\beta})\bigr]
\;\sim\; \chi^2_1
$$

```python
null_model = sm.Logit(y, sm.add_constant(np.ones(n))).fit(disp=0)
lr_stat = -2 * (null_model.llf - result.llf)
lr_pvalue = stats.chi2.sf(lr_stat, df=1)
print(f"Likelihood Ratio Test: chi2 = {lr_stat:.4f}, p = {lr_pvalue:.6f}")
```

## Confusion Matrix and Classification Report

At the default threshold $\tau = 0.5$ the confusion matrix is

$$
\begin{pmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{pmatrix}
$$

and the standard metrics are:

$$
\text{Accuracy} = \frac{\text{TP}+\text{TN}}{n}, \qquad
\text{Precision} = \frac{\text{TP}}{\text{TP}+\text{FP}}, \qquad
\text{Recall} = \frac{\text{TP}}{\text{TP}+\text{FN}}
$$

$$
F_1 = \frac{2\,\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}
$$

```python
from sklearn.metrics import (confusion_matrix, classification_report,
                              accuracy_score, precision_score,
                              recall_score, f1_score)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print(f"TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))
```

## ROC Curve and AUC

The ROC curve plots TPR against FPR as the threshold varies.  The area under
this curve (AUC) summarizes discriminative ability:

$$
\text{AUC} = \int_0^1 \text{TPR}\bigl(\text{FPR}\bigr)\,d(\text{FPR})
$$

```python
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
print(f"AUC = {auc:.4f}")
```

## Precision-Recall Curve

When the positive class is rare, the precision-recall curve is often more
informative than the ROC curve.  Average precision (AP) summarizes this curve:

$$
\text{AP} = \sum_{k} (R_k - R_{k-1})\,P_k
$$

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)
print(f"Average Precision = {ap:.4f}")
```

## Threshold Selection

The default threshold $\tau = 0.5$ is not always optimal.  **Youden's J
statistic** selects the threshold that maximizes $J = \text{TPR} - \text{FPR}$:

```python
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold (Youden's J): {optimal_threshold:.3f}")
print(f"  TPR = {tpr[optimal_idx]:.3f}, FPR = {fpr[optimal_idx]:.3f}")
```

The table below shows how accuracy, precision, recall, and $F_1$ change with
the threshold:

```python
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred_t = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_test, y_pred_t)
    prec = precision_score(y_test, y_pred_t, zero_division=0)
    rec = recall_score(y_test, y_pred_t, zero_division=0)
    f1 = f1_score(y_test, y_pred_t, zero_division=0)
    print(f"tau={threshold:.1f}  Acc={acc:.3f}  Prec={prec:.3f}  "
          f"Rec={rec:.3f}  F1={f1:.3f}")
```

## Interpretation

- The coefficient on hours studied is positive, confirming that more study time
  increases the log-odds (and therefore the probability) of passing.
- Exponentiating the coefficient gives the odds ratio: each additional hour
  multiplies the odds of passing by $e^{\hat\beta_1}$.
- The likelihood ratio test rejects the null hypothesis that hours studied has
  no effect.
- AUC provides a single-number summary of how well the model separates the two
  classes across all thresholds.
- Youden's J statistic offers a principled way to choose a threshold when false
  positives and false negatives are equally costly.

## Exercises

**Exercise 1.**
Generate a synthetic dataset of size $n = 500$ with two features $x_1$ and $x_2$ and true logistic model $\log\frac{p}{1-p} = -1 + 0.5\,x_1 - 0.3\,x_2$.  Fit a logistic regression with scikit-learn, report the estimated coefficients, and compare them to the true values.

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    np.random.seed(0)
    n = 500
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    logit = -1 + 0.5 * x1 - 0.3 * x2
    p = 1 / (1 + np.exp(-logit))
    y = np.random.binomial(1, p)

    X = np.column_stack([x1, x2])
    model = LogisticRegression(penalty=None, solver='lbfgs')
    model.fit(X, y)
    print(f"Intercept: {model.intercept_[0]:.4f} (true: -1)")
    print(f"Coef x1:   {model.coef_[0][0]:.4f} (true: 0.5)")
    print(f"Coef x2:   {model.coef_[0][1]:.4f} (true: -0.3)")
    ```

    With $n = 500$ the estimates are close to the true values but not exact
    due to sampling variability.  Increasing $n$ brings them closer. $\square$

---

**Exercise 2.**
Show algebraically that maximizing Youden's $J = \text{TPR} - \text{FPR}$ is
equivalent to finding the threshold where the ROC curve has the greatest
vertical distance from the diagonal.

??? success "Solution to Exercise 2"

    The diagonal of the ROC plot is the line $\text{TPR} = \text{FPR}$.
    The vertical distance from a point $(\text{FPR}, \text{TPR})$ on the ROC
    curve to the diagonal is

    $$
    d = \text{TPR} - \text{FPR}
    $$

    because the diagonal value at $x = \text{FPR}$ is $\text{FPR}$ itself.
    Therefore maximizing $J = \text{TPR} - \text{FPR}$ is exactly maximizing
    the vertical distance from the curve to the diagonal. $\square$

---

**Exercise 3.**
A logistic regression outputs $\hat{p} = 0.72$ for a patient.  The confusion
matrix at threshold 0.5 shows $\text{TP}=80$, $\text{FP}=15$, $\text{FN}=20$,
$\text{TN}=85$.  Compute accuracy, precision, recall, and $F_1$.

??? success "Solution to Exercise 3"

    $$
    \text{Accuracy} = \frac{80+85}{200} = 0.825
    $$

    $$
    \text{Precision} = \frac{80}{80+15} = \frac{80}{95} \approx 0.842
    $$

    $$
    \text{Recall} = \frac{80}{80+20} = \frac{80}{100} = 0.800
    $$

    $$
    F_1 = \frac{2 \times 0.842 \times 0.800}{0.842 + 0.800}
        = \frac{1.347}{1.642} \approx 0.821
    $$

    $\square$

---

**Exercise 4.**
Prove that the $F_1$ score is the harmonic mean of precision and recall.

??? success "Solution to Exercise 4"

    The harmonic mean of two positive numbers $a$ and $b$ is

    $$
    H = \frac{2}{\frac{1}{a} + \frac{1}{b}} = \frac{2ab}{a+b}
    $$

    Setting $a = \text{Precision}$ and $b = \text{Recall}$:

    $$
    H = \frac{2\,\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}} = F_1
    $$

    This shows that $F_1$ penalizes extreme imbalances between precision and
    recall more than the arithmetic mean would.  If either precision or recall
    is close to zero, the harmonic mean is pulled sharply downward. $\square$

---

**Exercise 5.**
The likelihood ratio statistic $\Lambda = -2[\ell_0 - \ell_1]$ follows
$\chi^2_p$ under $H_0$, where $p$ is the number of additional parameters in
the full model.  Suppose the null model has log-likelihood $-180$ and the full
model (with 3 predictors) has log-likelihood $-160$.  Compute $\Lambda$ and
state whether you reject $H_0$ at $\alpha = 0.01$.

??? success "Solution to Exercise 5"

    $$
    \Lambda = -2\bigl[(-180) - (-160)\bigr] = -2(-20) = 40
    $$

    Under $H_0$, $\Lambda \sim \chi^2_3$.  The critical value at
    $\alpha = 0.01$ is $\chi^2_{3,0.99} \approx 11.34$.  Since
    $40 \gg 11.34$ we reject $H_0$ and conclude that at least one of the
    three predictors has a statistically significant effect on the outcome.

    Equivalently, $p\text{-value} = P(\chi^2_3 \geq 40) \approx 1.0\times 10^{-8}$,
    far below 0.01. $\square$
