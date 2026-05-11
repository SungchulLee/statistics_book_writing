# One-vs-Rest Approach

## Idea

Many classification algorithms — logistic regression, SVMs, perceptrons — are
designed for **binary** problems.  The **One-vs-Rest** (OVR, also called
One-vs-All) strategy extends any binary classifier to $C$ classes by
decomposing the multiclass problem into $C$ independent binary sub-problems.
Each sub-problem asks: "Is this observation class $c$, or is it something else?"

## Training Procedure

Given training data $\{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$ with
$y_i \in \{1, \ldots, C\}$, OVR trains $C$ binary classifiers
$f_1, \ldots, f_C$.  For classifier $f_c$:

1. Relabel the targets:

$$
\tilde{y}_i^{(c)} =
\begin{cases}
1 & \text{if } y_i = c \\
0 & \text{otherwise}
\end{cases}
$$

2. Train a binary classifier on the relabeled dataset
   $\{(\mathbf{x}_i, \tilde{y}_i^{(c)})\}_{i=1}^{n}$.
3. The result is a scoring function $f_c(\mathbf{x})$ — for logistic
   regression, this is $P(Y = c \mid \mathbf{x})$ from the $c$-th binary
   model.

## Prediction Rule

At test time, compute all $C$ scores and predict the class with the highest
score:

$$
\hat{y} = \arg\max_{c \in \{1,\ldots,C\}} f_c(\mathbf{x})
$$

For logistic regression each $f_c$ outputs a probability, but these
probabilities are estimated from $C$ separate models and generally **do not sum
to one**.

## Class Imbalance in OVR

Each binary sub-problem is typically **imbalanced**: if the $C$ classes are
roughly equal in size, the "rest" class has about $(C-1)/C$ of the data.  For
example, with $C = 10$ balanced classes, each binary classifier sees a 1:9
positive-to-negative ratio.

This artificial imbalance can cause:

- Bias toward predicting "rest" (class 0 in the binary sub-problem).
- Poorly calibrated probability estimates.

!!! warning "Handling Artificial Imbalance"
    Standard remedies include class-weight adjustment
    (`class_weight='balanced'` in scikit-learn), oversampling the positive
    class, or using a calibration step after training.

## Advantages and Disadvantages

| Aspect | Assessment |
|---|---|
| Number of classifiers | $C$ (linear in the number of classes) |
| Training data per classifier | All $n$ examples |
| Interpretability | Each classifier has its own coefficients |
| Parallelization | All $C$ classifiers can be trained independently |
| Probability calibration | Scores do not sum to 1 without recalibration |
| Decision boundaries | Piecewise — can produce ambiguous regions |

### Ambiguous Regions

Because the $C$ classifiers are trained independently, it is possible for two
or more classifiers to assign high scores to the same input, or for all
classifiers to assign low scores.  In the first case the $\arg\max$ resolves
the tie; in the second the prediction is unreliable.  These **ambiguous
regions** do not arise in native softmax regression, which always produces a
valid probability distribution.

## OVR with Logistic Regression

When the base classifier is logistic regression, each binary model estimates

$$
f_c(\mathbf{x}) = \sigma(\mathbf{w}_c^T\mathbf{x} + b_c)
= \frac{1}{1 + e^{-(\mathbf{w}_c^T\mathbf{x} + b_c)}}
$$

The collection of weight vectors $\mathbf{w}_1, \ldots, \mathbf{w}_C$ and
biases $b_1, \ldots, b_C$ defines $C$ linear decision boundaries in feature
space.

??? example "Worked Example: 3-Class OVR"
    Consider three classes (A, B, C) with $n = 300$ training examples (100 per
    class) and $p = 2$ features.

    **Training.** Three binary logistic regressions are fitted:

    | Classifier | Positive class | Negative class | $n_+$ | $n_-$ |
    |---|---|---|---|---|
    | $f_A$ | A | B $\cup$ C | 100 | 200 |
    | $f_B$ | B | A $\cup$ C | 100 | 200 |
    | $f_C$ | C | A $\cup$ B | 100 | 200 |

    **Prediction.** For a new point $\mathbf{x}_*$, suppose:

    - $f_A(\mathbf{x}_*) = 0.72$
    - $f_B(\mathbf{x}_*) = 0.35$
    - $f_C(\mathbf{x}_*) = 0.18$

    The predicted class is A because $0.72 = \max(0.72, 0.35, 0.18)$.  Note
    that the scores sum to $1.25 \neq 1$ — they are not a valid probability
    distribution.

## Scikit-learn Usage

Scikit-learn's `LogisticRegression` uses OVR by default when
`multi_class='ovr'` (the default for solvers that do not support multinomial).
The `OneVsRestClassifier` wrapper applies OVR to any binary classifier:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Explicit OVR with logistic regression
model = OneVsRestClassifier(LogisticRegression())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

See [Comparison with Softmax](comparison.md) for a discussion of when OVR is
preferred over native multinomial softmax regression.

## Exercises

**Exercise 1.**
One-vs-Rest versus Softmax

Consider a 4-class problem where a one-vs-rest (OvR) approach trains 4 binary classifiers, and a softmax model is trained natively.

**(a)** The OvR classifiers produce the following scores for a test example:

| Classifier | P(class $k$ vs rest) |
|:---:|:---:|
| Class 1 vs rest | 0.80 |
| Class 2 vs rest | 0.65 |
| Class 3 vs rest | 0.40 |
| Class 4 vs rest | 0.55 |

Do these probabilities form a valid probability distribution? Explain.

**(b)** How would you make a prediction from the OvR scores? What is the predicted class?

**(c)** A softmax model produces $\hat{\mathbf{p}} = (0.45, 0.30, 0.10, 0.15)^\top$ for the same example. How do these probabilities differ fundamentally from the OvR scores?

**(d)** State two advantages of native softmax regression over the OvR approach.

??? success "Solution to Exercise 1"

    **(a)** The scores sum to $0.80 + 0.65 + 0.40 + 0.55 = 2.40 \ne 1$. They do **not** form a valid probability distribution. Each OvR classifier independently estimates the probability that the example belongs to its class versus all others. These individual binary probabilities are not constrained to sum to 1, and their magnitudes depend on the decision boundary of each separate binary model.

    **(b)** The standard OvR prediction assigns the class with the highest score: $\arg\max_k \text{score}_k = 1$, since Class 1 has the highest score (0.80). While simple, this rule can lead to ties and does not produce calibrated probabilities. One could normalize the scores by dividing by their sum ($0.80/2.40 = 0.333$), but this is ad hoc and does not guarantee well-calibrated probabilities.

    **(c)** The softmax probabilities $\hat{\mathbf{p}} = (0.45, 0.30, 0.10, 0.15)^\top$ sum to exactly 1 and are jointly estimated by a single model. They represent a coherent probability distribution over all classes, calibrated through the softmax function. The OvR scores, by contrast, come from independently trained classifiers that do not communicate during training.

    **(d)** Two advantages of native softmax over OvR:

    1. **Calibrated probabilities**: Softmax inherently produces a valid probability distribution (non-negative, sums to 1) without post-hoc normalization. This makes the outputs directly interpretable as class probabilities and suitable for downstream probabilistic reasoning.

    2. **Joint training**: The softmax model's weight matrix is optimized to separate all classes simultaneously, allowing the model to share information across class boundaries. OvR trains each classifier independently, so it cannot exploit the structure of the multiclass problem. For example, features useful for distinguishing Class 1 from Class 2 might also help distinguish Class 3 from Class 4, but OvR cannot leverage this.
