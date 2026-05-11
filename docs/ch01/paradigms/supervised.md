# Supervised Learning

Supervised learning trains a model on labeled input-output pairs so that it can predict the output for new, unlabeled inputs. It is the dominant paradigm in modern data analysis because the goal — *predict $y$ from $\mathbf{x}$* — is precise, the loss is observable, and the success criterion is unambiguous. Credit scoring, demand forecasting, fraud detection, medical diagnosis, machine translation, and most image-recognition systems are supervised learners under the hood.

## Definition

Let $(\mathbf{X}, Y)$ be jointly distributed random variables with $\mathbf{X} \in \mathcal{X} \subseteq \mathbb{R}^p$ and $Y \in \mathcal{Y}$. We observe a training sample $\mathcal{D}_n = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ drawn i.i.d. from this joint distribution and seek a function $\hat{f} : \mathcal{X} \to \mathcal{Y}$ that minimizes the **expected loss** (or **risk**)

$$
R(f) = \mathbb{E}_{(\mathbf{X}, Y)}\!\left[L(Y, f(\mathbf{X}))\right]
$$

where $L : \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}_{\ge 0}$ is a problem-specific loss function. The two canonical tasks differ in the type of $Y$:

- **Regression**: $\mathcal{Y} = \mathbb{R}$. Standard loss is squared error $L(y, \hat{y}) = (y - \hat{y})^2$, whose risk minimizer is the conditional mean $f^*(\mathbf{x}) = \mathbb{E}[Y \mid \mathbf{X} = \mathbf{x}]$.
- **Classification**: $\mathcal{Y} = \{1, \ldots, K\}$. Standard loss is 0–1 loss $L(y, \hat{y}) = \mathbf{1}\{y \ne \hat{y}\}$, whose risk minimizer is the **Bayes classifier** $f^*(\mathbf{x}) = \arg\max_k P(Y = k \mid \mathbf{X} = \mathbf{x})$.

Because $R(f)$ depends on the unknown joint distribution, we replace it with the **empirical risk**

$$
\hat{R}_n(f) = \frac{1}{n} \sum_{i=1}^n L(y_i, f(\mathbf{x}_i))
$$

and minimize over a chosen hypothesis class $\mathcal{F}$ — a procedure called **empirical risk minimization** (ERM).

## Explanation

### The bias–variance decomposition

The fundamental tension in supervised learning is captured by the bias–variance decomposition. For squared-error regression at a fixed point $\mathbf{x}_0$,

$$
\mathbb{E}\!\left[(Y - \hat{f}(\mathbf{x}_0))^2\right] = \underbrace{\sigma^2}_{\text{irreducible}} + \underbrace{\left(\mathbb{E}[\hat{f}(\mathbf{x}_0)] - f^*(\mathbf{x}_0)\right)^2}_{\text{bias}^2} + \underbrace{\mathrm{Var}(\hat{f}(\mathbf{x}_0))}_{\text{variance}}
$$

Simple models (linear regression, shallow trees) have high bias but low variance; flexible models (deep neural nets, large ensembles) have low bias but high variance. The art of supervised learning is choosing model complexity so that bias and variance trade off favorably for the available sample size.

### The workflow

1. **Split the data**: train / validation / test (e.g., 60/20/20), or use $k$-fold cross-validation on the train+validation portion.
2. **Choose a model family** $\mathcal{F}$: linear, tree-based, kernel-based, neural, ensemble.
3. **Fit by ERM**: minimize empirical risk, often with a regularization penalty $\lambda \cdot \Omega(f)$ to control complexity.
4. **Tune hyperparameters** (penalty strength, tree depth, learning rate) on the validation set.
5. **Evaluate on the test set** — used exactly once, after all model decisions are locked.
6. **Deploy and monitor** for distribution drift.

### Why supervised learning is easier than its cousins

Compared to unsupervised and reinforcement learning, the supervised problem enjoys three structural advantages:

- **Observable loss**: every prediction has a ground-truth label to compare against, so $\hat{R}_n(f)$ is computable.
- **Independent supervision**: each $(\mathbf{x}_i, y_i)$ provides its own learning signal; there are no temporal credit-assignment problems as in reinforcement learning.
- **Honest evaluation**: a held-out test set yields an unbiased estimate of generalization error, provided no information leaks during training.

These same advantages mean supervised learning fails silently when the test distribution differs from training (covariate shift, label shift) or when labels are themselves systematically biased.

## Examples

```python
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # logistic function

np.random.seed(42)
n = 1000

# Simulate binary classification: loan default
income = np.random.normal(60, 20, n).clip(10)
dti = np.random.normal(0.3, 0.15, n).clip(0.01, 1.0)
log_odds = -3 + 0.01 * (50 - income) + 5 * (dti - 0.3)
prob = expit(log_odds)
default = np.random.binomial(1, prob)

# Train/test split
train, test = np.arange(700), np.arange(700, n)
X = np.column_stack([np.ones(n), income, dti])

# Fit logistic regression by minimizing negative log-likelihood
def neg_log_lik(beta):
    z = X[train] @ beta
    return -np.sum(default[train] * z - np.log1p(np.exp(z)))

result = minimize(neg_log_lik, np.zeros(3), method="BFGS")
beta_hat = result.x

probs_test = expit(X[test] @ beta_hat)
preds = (probs_test > 0.5).astype(int)
accuracy = np.mean(preds == default[test])
print(f"Test accuracy:        {accuracy:.3f}")
print(f"Default rate (test):  {default[test].mean():.3f}")
print(f"Coefficients:         {beta_hat.round(4)}")
```

The accuracy alone is misleading when classes are imbalanced — a model that always predicts "no default" can be 90% accurate when only 10% of loans default. Real evaluations require precision, recall, ROC-AUC, or a cost-weighted loss reflecting the asymmetric cost of false positives versus false negatives.

## Exercises

**Exercise 1.**
Classify each task into supervised, unsupervised, or reinforcement learning, and (for supervised tasks) say whether it is regression or classification.

**(a)** Predicting tomorrow's stock closing price given historical price and volume data.
**(b)** Grouping customers into segments based on purchasing behavior, without predefined categories.
**(c)** Teaching a robot to navigate a maze by rewarding it for reaching the exit and penalizing collisions.
**(d)** Training a model to flag emails as spam or not spam using a labeled inbox.
**(e)** Detecting anomalous credit-card transactions when no labeled fraud examples are available.
**(f)** Estimating the number of hospital readmissions a patient will have in the next year.

??? success "Solution to Exercise 1"
    (a) Supervised — regression (continuous target).
    (b) Unsupervised — clustering, no labels.
    (c) Reinforcement learning — reward-driven sequential decisions.
    (d) Supervised — binary classification.
    (e) Unsupervised — anomaly detection without labeled examples (could be semi-supervised if some labels exist).
    (f) Supervised — count regression (Poisson regression is a natural choice).

---

**Exercise 2.**
Let $L(y, \hat{y}) = (y - \hat{y})^2$ be squared-error loss. Show that the function $f^*$ minimizing the risk $R(f) = \mathbb{E}[L(Y, f(\mathbf{X}))]$ over all measurable $f$ is $f^*(\mathbf{x}) = \mathbb{E}[Y \mid \mathbf{X} = \mathbf{x}]$.

??? success "Solution to Exercise 2"
    Condition on $\mathbf{X} = \mathbf{x}$ and minimize pointwise. For fixed $\mathbf{x}$ we seek $c \in \mathbb{R}$ minimizing $\mathbb{E}[(Y - c)^2 \mid \mathbf{X} = \mathbf{x}]$. Expanding,

    $$
    \mathbb{E}[(Y - c)^2 \mid \mathbf{X} = \mathbf{x}] = \mathrm{Var}(Y \mid \mathbf{X} = \mathbf{x}) + (\mathbb{E}[Y \mid \mathbf{X} = \mathbf{x}] - c)^2
    $$

    The first term does not depend on $c$, and the second is minimized at $c = \mathbb{E}[Y \mid \mathbf{X} = \mathbf{x}]$. Since this holds for every $\mathbf{x}$, the global minimizer is $f^*(\mathbf{x}) = \mathbb{E}[Y \mid \mathbf{X} = \mathbf{x}]$. $\square$

---

**Exercise 3.**
For 0–1 loss in $K$-class classification, show that the Bayes classifier $f^*(\mathbf{x}) = \arg\max_k P(Y = k \mid \mathbf{X} = \mathbf{x})$ minimizes the expected misclassification rate.

??? success "Solution to Exercise 3"
    For any classifier $f$,

    $$
    \mathbb{E}[\mathbf{1}\{Y \ne f(\mathbf{X})\} \mid \mathbf{X} = \mathbf{x}] = 1 - P(Y = f(\mathbf{x}) \mid \mathbf{X} = \mathbf{x})
    $$

    To minimize this pointwise we maximize $P(Y = f(\mathbf{x}) \mid \mathbf{X} = \mathbf{x})$ over $f(\mathbf{x}) \in \{1, \ldots, K\}$, which selects $f^*(\mathbf{x}) = \arg\max_k P(Y = k \mid \mathbf{X} = \mathbf{x})$. Integrating over $\mathbf{X}$ gives the global minimum. $\square$

---

**Exercise 4.**
A bank trains a credit-scoring model on applicants from 2015–2019 and observes 92% test accuracy on a held-out 2019 sample. Deployed in 2024, it achieves only 71% accuracy. List three distinct mechanisms that could explain the drop, and a diagnostic for each.

??? success "Solution to Exercise 4"
    - **Covariate shift**: the distribution of $\mathbf{X}$ has changed (e.g., post-pandemic income distributions). Diagnostic: compare marginal distributions of features between training and 2024 data using KS tests or PSI (population stability index).
    - **Label shift / concept drift**: the conditional $P(Y \mid \mathbf{X})$ has changed (default behavior responds to new economic conditions). Diagnostic: refit the model on recent labeled data and compare coefficients or feature importances.
    - **Data leakage in training**: a feature unavailable at deployment time was used during training (e.g., a "current balance" that was actually post-default). Diagnostic: audit the feature pipeline and re-train with strictly pre-decision features.

    Other valid mechanisms: feedback loops (the model's own decisions changed the applicant pool), sampling bias in the original training set, or label-quality changes (definition of "default" was revised).

---

**Exercise 5.**
You fit two regression models to the same data:
- Model A: linear regression, training MSE $= 12$, test MSE $= 15$.
- Model B: deep neural network, training MSE $= 2$, test MSE $= 25$.

Using the bias–variance decomposition, characterize each model. Which would you deploy, and what would you try next?

??? success "Solution to Exercise 5"
    Model A has high training error and only slightly higher test error: it is **underfit** (high bias, low variance). The hypothesis class is too restrictive.

    Model B fits the training set far better than the test set: it is **overfit** (low bias, high variance). It has memorized noise.

    Deploy Model A — its 15 MSE generalizes, while Model B's training MSE of 2 is illusory. Next steps: try a model of intermediate flexibility (gradient-boosted trees, regularized neural net, kernel ridge regression), or apply regularization / early stopping / data augmentation to Model B to reduce its variance.

---

**Exercise 6.**
Why does evaluating a model on its own training data give an overly optimistic estimate of risk? Formalize the answer by relating $\hat{R}_n(\hat{f})$ to $R(\hat{f})$ when $\hat{f}$ was chosen to minimize $\hat{R}_n$.

??? success "Solution to Exercise 6"
    The training data plays two roles: it determines the estimator $\hat{f}$ (which is selected to minimize $\hat{R}_n$ over $\mathcal{F}$), and it is then re-used to compute $\hat{R}_n(\hat{f})$. Because $\hat{f}$ was chosen to make $\hat{R}_n$ as small as possible, the empirical risk underestimates the true risk:

    $$
    \mathbb{E}\!\left[\hat{R}_n(\hat{f})\right] \le \mathbb{E}\!\left[R(\hat{f})\right]
    $$

    with equality only when $\mathcal{F}$ contains a single function. The gap is the **optimism** of the training error, and it grows with the effective complexity of $\mathcal{F}$. A held-out test set breaks the dependence: the test data was not used to choose $\hat{f}$, so $\hat{R}_{\text{test}}(\hat{f})$ is an unbiased estimate of $R(\hat{f})$. This is why train/test splitting (or cross-validation) is non-negotiable for honest evaluation. $\square$
