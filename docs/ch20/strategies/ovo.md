# One-vs-One Approach

## Idea

The **One-vs-One** (OVO) strategy decomposes a $C$-class problem into
$\binom{C}{2} = C(C-1)/2$ binary sub-problems, one for every pair of
classes.  Each binary classifier is trained on only the examples from the two
classes it distinguishes, making each sub-problem smaller and often easier to
solve.  At prediction time the classifiers vote, and the class with the most
votes wins.

## Training Procedure

For each pair $(c, c')$ with $1 \le c < c' \le C$:

1. Extract the subset of training data belonging to class $c$ or class $c'$:

$$
\mathcal{D}_{c,c'} = \{(\mathbf{x}_i, y_i) : y_i \in \{c, c'\}\}
$$

2. Train a binary classifier $f_{c,c'}$ on $\mathcal{D}_{c,c'}$, where class
   $c$ is treated as positive and class $c'$ as negative.

The total number of classifiers is

$$
\binom{C}{2} = \frac{C(C-1)}{2}
$$

For $C = 10$ classes this gives 45 classifiers; for $C = 26$ (e.g., letter
recognition) it gives 325.

## Prediction by Majority Voting

At test time, each of the $C(C-1)/2$ classifiers casts a vote for one of its
two classes.  Let $v_c$ denote the number of votes received by class $c$:

$$
v_c = \sum_{\substack{c' = 1 \\ c' \neq c}}^{C}
\mathbf{1}\{f_{c,c'}(\mathbf{x}) \text{ predicts class } c\}
$$

The predicted class is

$$
\hat{y} = \arg\max_{c \in \{1,\ldots,C\}} v_c
$$

Each class participates in $C - 1$ pairwise contests, so the maximum possible
vote count for any class is $C - 1$.

### Tie Breaking

Ties can occur when two or more classes receive the same number of votes.
Common tie-breaking strategies include:

- Predict the class with the **highest total confidence** (sum of predicted
  probabilities from all pairwise classifiers involving that class).
- Predict the class with the **smallest index** (arbitrary but deterministic).

## Computational Cost

| Quantity | OVR | OVO |
|---|---|---|
| Number of classifiers | $C$ | $C(C-1)/2$ |
| Training samples per classifier | $n$ | $\approx 2n/C$ (on average) |
| Total training cost | $C \cdot T(n, p)$ | $\frac{C(C-1)}{2}\cdot T(2n/C, p)$ |

Here $T(m, p)$ is the cost of training one binary classifier on $m$ examples
with $p$ features.  Because each OVO classifier uses only a fraction of the
data, the per-classifier training time is much smaller.  For algorithms whose
training cost is super-linear in $n$ (e.g., SVMs with $O(n^2)$ to $O(n^3)$
cost), OVO can be **faster** overall than OVR despite having more classifiers.

For logistic regression, where training is roughly $O(np)$, the total OVO cost
is approximately $C(C-1)/2 \cdot O(2np/C) = O((C-1)np)$, which is comparable
to OVR's $O(Cnp)$.

## Advantages and Disadvantages

| Aspect | Assessment |
|---|---|
| Sub-problem size | Small (only two classes per classifier) |
| Training speed for super-linear algorithms | Often faster than OVR |
| Number of classifiers | Quadratic in $C$ — can be large |
| Prediction speed | Must evaluate $C(C-1)/2$ classifiers |
| Class imbalance | Less severe than OVR (balanced binary sub-problems) |
| Probability estimates | Not directly available; require aggregation |

### When OVO Shines

OVO is the default strategy for SVMs in many libraries (e.g., libsvm and
scikit-learn's `SVC`) because SVMs scale poorly with $n$, and OVO's smaller
sub-problems offset the quadratic number of classifiers.  For logistic
regression the benefit is less pronounced, and native softmax regression is
typically preferred.

??? example "Worked Example: 4-Class OVO"
    Consider $C = 4$ classes (A, B, C, D).  The number of classifiers is
    $\binom{4}{2} = 6$:

    | Classifier | Classes | Training subset size |
    |---|---|---|
    | $f_{A,B}$ | A vs B | $n_A + n_B$ |
    | $f_{A,C}$ | A vs C | $n_A + n_C$ |
    | $f_{A,D}$ | A vs D | $n_A + n_D$ |
    | $f_{B,C}$ | B vs C | $n_B + n_C$ |
    | $f_{B,D}$ | B vs D | $n_B + n_D$ |
    | $f_{C,D}$ | C vs D | $n_C + n_D$ |

    For a new point $\mathbf{x}_*$, suppose the votes are:

    | Classifier | Winner |
    |---|---|
    | $f_{A,B}$ | A |
    | $f_{A,C}$ | A |
    | $f_{A,D}$ | D |
    | $f_{B,C}$ | B |
    | $f_{B,D}$ | B |
    | $f_{C,D}$ | D |

    Vote tallies: A = 2, B = 2, C = 0, D = 2.  Three classes are tied at 2
    votes.  A tie-breaking rule (e.g., highest total confidence) is needed to
    produce a final prediction.

## Scikit-learn Usage

Scikit-learn provides the `OneVsOneClassifier` wrapper:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier

model = OneVsOneClassifier(LogisticRegression())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

For SVMs, OVO is the default behavior of `SVC` — no explicit wrapper is
needed.

See [Comparison with Softmax](comparison.md) for guidance on choosing between
OVO, OVR, and native softmax regression.
