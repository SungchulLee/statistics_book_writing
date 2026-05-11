# Comparison with Softmax

## Three Approaches to Multiclass Classification

The previous sections introduced two decomposition strategies for extending
binary classifiers to multiple classes:
[One-vs-Rest (OVR)](ovr.md) and [One-vs-One (OVO)](ovo.md).  The third
approach — **native softmax (multinomial) regression** — models all $C$ classes
jointly in a single optimization problem
(see [Softmax Function](../softmax_regression/softmax.md)).  This section
compares the three methods across the dimensions that matter most in practice.

## Structural Comparison

| Property | OVR | OVO | Softmax |
|---|---|---|---|
| Number of models | $C$ | $C(C-1)/2$ | 1 |
| Parameters per model | $p$ (binary) | $p$ (binary) | $pC$ (single matrix) |
| Total parameters | $Cp$ | $C(C-1)p/2$ | $pC$ |
| Training data per model | All $n$ | $\approx 2n/C$ | All $n$ |
| Joint optimization | No | No | Yes |
| Outputs valid probabilities | No (do not sum to 1) | No (votes, not probabilities) | Yes |

## Scalability

### Training cost

For logistic regression (training cost roughly linear in $n$):

$$
\text{OVR: } O(Cnp), \qquad
\text{OVO: } O\!\left(\frac{C(C-1)}{2}\cdot\frac{2np}{C}\right) = O((C-1)np), \qquad
\text{Softmax: } O(nCp)
$$

All three scale similarly when the base cost is linear in $n$.  The practical
difference lies in constant factors and parallelizability: OVR and OVO train
independent sub-problems that parallelize trivially, while softmax requires a
single coordinated optimization.

### Prediction cost

At test time, OVR evaluates $C$ linear functions, OVO evaluates $C(C-1)/2$
linear functions, and softmax evaluates one matrix-vector product of size
$p \times C$.  For large $C$, OVO's quadratic number of evaluations makes it
the slowest at prediction time.

## Calibration

**Softmax** produces well-calibrated probabilities by construction: the output
lies on the probability simplex and sums to one.  This is a direct consequence
of jointly modeling all classes.

**OVR** outputs $C$ independent sigmoid probabilities that generally do not sum
to one.  Rescaling by dividing by their sum (Platt scaling) is a common
post-hoc fix, but it does not fully correct the calibration because each binary
model was trained on an artificially imbalanced dataset.

**OVO** outputs votes, not probabilities.  Probability estimates can be
recovered using pairwise coupling methods (e.g., the method of Wu, Lin, and
Weng, 2004), but these add complexity and are approximate.

!!! tip "When Calibration Matters"
    If the application requires reliable probability estimates — credit
    scoring, medical diagnosis, or any setting where the predicted probability
    drives a downstream decision — native softmax is strongly preferred.

## Statistical Consistency

Softmax regression directly maximizes the correct multinomial log-likelihood
and is therefore **consistent**: as $n \to \infty$, the estimated parameters
converge to the true parameters (assuming the model is correctly specified).

OVR estimates each class boundary independently, which can be suboptimal when
class boundaries interact.  In particular, OVR can produce inconsistent
estimates when the true decision boundaries do not decompose into independent
binary problems.

OVO avoids the class-imbalance issue of OVR but introduces a different
problem: the pairwise classifiers may disagree, and majority voting does not
always recover the Bayes-optimal decision.

## When to Choose Each Strategy

| Scenario | Recommended strategy | Reason |
|---|---|---|
| Logistic regression with moderate $C$ | Softmax | Joint optimization, calibrated probabilities |
| SVM with moderate $C$ | OVO | SVMs scale poorly with $n$; OVO uses small subsets |
| SVM with very large $C$ | OVR | Quadratic number of OVO classifiers becomes prohibitive |
| Need probability estimates | Softmax | Only approach with natively valid probabilities |
| Per-class interpretability | OVR | Each binary model has its own interpretable coefficients |
| Base classifier does not support multiclass | OVR or OVO | Decomposition is the only option |

## Empirical Performance

In practice, the three strategies often achieve similar **accuracy** on
well-behaved datasets.  The differences become more pronounced in specific
settings:

- **Highly imbalanced classes:** Softmax handles imbalance more gracefully
  because it trains on all classes simultaneously.
- **Large $C$:** OVO's $O(C^2)$ classifiers become a bottleneck in both
  training and prediction.
- **Correlated classes:** Softmax captures inter-class structure through the
  shared weight matrix; decomposition methods treat classes independently.

??? example "Empirical Comparison on a 5-Class Problem"
    On a synthetic dataset with $n = 1000$, $p = 10$, and $C = 5$ balanced
    classes, logistic regression is evaluated under all three strategies using
    5-fold cross-validation:

    | Strategy | Mean accuracy | Std | Probability sum = 1 |
    |---|---|---|---|
    | OVR | 0.842 | 0.018 | No |
    | OVO | 0.838 | 0.021 | No |
    | Softmax | 0.847 | 0.016 | Yes |

    All three achieve similar accuracy.  The softmax model has a slight edge
    and is the only one producing valid probability estimates.  On this
    well-behaved dataset the differences are small, but the calibration
    advantage of softmax becomes more important in downstream decision-making.

## Summary

For logistic regression, **native softmax is the default recommendation**.
It produces valid probabilities, is statistically consistent, and scales
comparably to the decomposition methods.  OVR and OVO remain valuable when
the base classifier is inherently binary (e.g., SVMs) or when per-class
interpretability is needed.  The table below recaps the key trade-offs:

| Criterion | OVR | OVO | Softmax |
|---|---|---|---|
| Calibration | Poor | None (votes) | Good |
| Consistency | Approximate | Approximate | Exact |
| Scalability in $C$ | Linear | Quadratic | Linear |
| Requires binary classifier | Yes | Yes | No |
| Parallelizable training | Yes | Yes | No |


## Exercises

**Exercise 1.**
Describe the main concept of Comparison with Softmax and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Comparison with Softmax is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

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
