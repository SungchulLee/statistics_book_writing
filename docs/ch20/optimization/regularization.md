# Regularization for Softmax Regression

## Why Regularize Softmax Models

Softmax regression with $C$ classes and $p$ features estimates a weight matrix
$\mathbf{W} \in \mathbb{R}^{p \times C}$ containing $pC$ parameters (plus $C$
bias terms).  As the number of classes or features grows, the model can overfit:
it memorizes training-set idiosyncrasies by pushing logits to extreme values,
producing overconfident predictions that generalize poorly.  Regularization
constrains the weight magnitudes, encouraging the model to rely on robust
patterns rather than noise.

## Penalized Cross-Entropy Objective

The unregularized softmax objective for $n$ training examples is the negative
log-likelihood (cross-entropy loss):

$$
\mathcal{L}(\mathbf{W}, \mathbf{b})
= -\frac{1}{n}\sum_{i=1}^{n}\log\operatorname{softmax}(\mathbf{W}^T\mathbf{x}_i + \mathbf{b})_{y_i}
$$

where $y_i \in \{1, \ldots, C\}$ is the true class.  Adding an L2 penalty:

$$
\mathcal{L}_{\text{reg}}(\mathbf{W}, \mathbf{b})
= \mathcal{L}(\mathbf{W}, \mathbf{b})
  + \frac{\lambda}{2}\lVert\mathbf{W}\rVert_F^2
$$

Here $\lVert\mathbf{W}\rVert_F^2 = \sum_{j=1}^{p}\sum_{c=1}^{C}W_{jc}^2$
is the squared **Frobenius norm** of the weight matrix.  As in logistic
regression, the bias vector $\mathbf{b}$ is typically **not penalized**.

## Gradient with L2 Penalty

The gradient of the regularized loss with respect to $\mathbf{W}$ adds a
simple correction to the unpenalized gradient:

$$
\frac{\partial\mathcal{L}_{\text{reg}}}{\partial \mathbf{W}}
= \frac{\partial\mathcal{L}}{\partial \mathbf{W}}
  + \lambda\,\mathbf{W}
$$

Each gradient descent step therefore becomes

$$
\mathbf{W}^{(t+1)}
= \mathbf{W}^{(t)}
  - \eta\frac{\partial\mathcal{L}}{\partial \mathbf{W}}\biggr|_{\mathbf{W}^{(t)}}
  - \eta\lambda\,\mathbf{W}^{(t)}
= (1 - \eta\lambda)\,\mathbf{W}^{(t)}
  - \eta\frac{\partial\mathcal{L}}{\partial \mathbf{W}}\biggr|_{\mathbf{W}^{(t)}}
$$

The factor $(1 - \eta\lambda)$ multiplies the current weights by a number
slightly less than one at every step, which is why L2 regularization in the
context of gradient descent is called **weight decay**.

## Weight Decay Interpretation

Weight decay and L2 regularization are mathematically equivalent for
standard gradient descent.  However, for adaptive optimizers (Adam, RMSProp)
the two formulations can differ because the adaptive learning rate scales the
penalty differently.  **Decoupled weight decay** (AdamW) applies the decay
directly to the weights rather than through the gradient, preserving the
intended regularization effect.

## Preventing Overconfident Predictions

Without regularization, the softmax model can drive the logit for the correct
class arbitrarily high, producing predicted probabilities close to 1.  This
**overconfidence** has two practical consequences:

1. **Poor calibration:** The predicted probabilities no longer match observed
   frequencies (see [Calibration and Brier Score](../../ch19/evaluation/calibration.md)).
2. **Sensitivity to distribution shift:** An overconfident model assigns near-zero
   probability to plausible alternative classes, making it brittle when the test
   distribution differs from training.

L2 regularization limits the norm of $\mathbf{W}$, which in turn bounds the
logit magnitudes.  Smaller logits produce softer probability distributions
(closer to uniform), improving calibration and robustness.

!!! tip "Label Smoothing as Implicit Regularization"
    An alternative to explicit weight penalties is **label smoothing**: replace
    the hard target $\mathbf{e}_{y_i}$ with
    $(1-\epsilon)\,\mathbf{e}_{y_i} + \frac{\epsilon}{C}\,\mathbf{1}$,
    where $\epsilon \in (0,1)$ is a small constant (e.g., $\epsilon = 0.1$).
    This discourages the model from driving any single class probability to 1.

## Choosing the Regularization Strength

The hyperparameter $\lambda$ (or equivalently $C = 1/\lambda$ in scikit-learn)
controls the bias-variance trade-off:

| $\lambda$ | Effect on weights | Effect on predictions |
|---|---|---|
| Too large | Weights near zero | Under-confident, high bias |
| Too small | Weights unconstrained | Over-confident, high variance |
| Optimal | Moderate magnitudes | Well-calibrated, good generalization |

Cross-validation on a held-out set is the standard method for selecting
$\lambda$.

??? example "Effect of Regularization on a 3-Class Problem"
    Consider softmax regression on a 3-class dataset with $p = 50$ features and
    $n = 300$ training examples.  We fit models with different $\lambda$ values
    and evaluate on a held-out test set of 100 examples.

    | $\lambda$ | Train accuracy | Test accuracy | Max predicted prob (mean) |
    |---|---|---|---|
    | 0.0 | 100% | 78% | 0.997 |
    | 0.01 | 97% | 84% | 0.92 |
    | 0.1 | 93% | 86% | 0.81 |
    | 1.0 | 85% | 82% | 0.62 |
    | 10.0 | 68% | 65% | 0.45 |

    At $\lambda = 0$ the model memorizes the training data (100% train
    accuracy) but generalizes poorly and produces overconfident predictions.
    At $\lambda = 0.1$ the test accuracy peaks and the mean maximum predicted
    probability drops to a more realistic 0.81.

## L1 Regularization and Elastic Net

While L2 is the most common penalty for softmax, L1 regularization is also
applicable:

$$
\mathcal{L}_{\text{L1}}
= \mathcal{L}(\mathbf{W}, \mathbf{b})
  + \lambda\sum_{j,c}|W_{jc}|
$$

L1 encourages **sparsity** in the weight matrix, setting entire rows of
$\mathbf{W}$ to zero when a feature is irrelevant to all classes.  The elastic
net combines both penalties and is available in scikit-learn via
`LogisticRegression(penalty='elasticnet', multi_class='multinomial')`.
