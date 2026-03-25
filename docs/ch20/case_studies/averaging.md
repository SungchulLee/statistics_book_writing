# Macro, Micro, and Weighted Averaging

## From Binary to Multiclass Metrics

In binary classification, precision and recall each produce a single number.
For $C > 2$ classes, these metrics are computed **per class**, yielding $C$
precision values and $C$ recall values.  To obtain a single summary number we
must **average** across classes.  The choice of averaging strategy can
dramatically affect the reported performance, especially when class sizes are
imbalanced.  This section defines the three standard strategies and explains
when each is appropriate.

## Per-Class Metrics

For class $c \in \{1, \ldots, C\}$, define the confusion-matrix quantities:

- $\text{TP}_c$: true positives (correctly predicted as class $c$)
- $\text{FP}_c$: false positives (incorrectly predicted as class $c$)
- $\text{FN}_c$: false negatives (class $c$ examples predicted as another class)

Per-class precision and recall are

$$
P_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c},
\qquad
R_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}
$$

and the per-class F1-score is their harmonic mean:

$$
F_{1,c} = \frac{2\,P_c\,R_c}{P_c + R_c}
$$

## Macro-Average

The **macro-average** computes the metric for each class independently and then
takes the unweighted mean:

$$
P_{\text{macro}} = \frac{1}{C}\sum_{c=1}^{C}P_c,
\qquad
R_{\text{macro}} = \frac{1}{C}\sum_{c=1}^{C}R_c,
\qquad
F_{1,\text{macro}} = \frac{1}{C}\sum_{c=1}^{C}F_{1,c}
$$

Every class contributes equally regardless of its size.  Macro-averaging is
therefore sensitive to performance on **rare classes**: a model that performs
poorly on a small minority class will see its macro score drop even if it
handles the majority classes well.

!!! tip "When to Use Macro-Average"
    Use macro-averaging when all classes are equally important, regardless of
    their frequency.  This is common in medical diagnosis (every disease
    matters) and document classification with balanced topic importance.

## Micro-Average

The **micro-average** pools the TP, FP, and FN counts across all classes before
computing the metric:

$$
P_{\text{micro}} = \frac{\sum_{c=1}^{C}\text{TP}_c}{\sum_{c=1}^{C}(\text{TP}_c + \text{FP}_c)},
\qquad
R_{\text{micro}} = \frac{\sum_{c=1}^{C}\text{TP}_c}{\sum_{c=1}^{C}(\text{TP}_c + \text{FN}_c)}
$$

For multiclass classification (where every sample receives exactly one label),
the micro-averaged precision, recall, and F1-score all equal the overall
**accuracy**:

$$
P_{\text{micro}} = R_{\text{micro}} = F_{1,\text{micro}} = \frac{\text{total correct}}{n}
$$

Micro-averaging gives more weight to **larger classes** because they contribute
more counts to the pooled totals.

!!! tip "When to Use Micro-Average"
    Use micro-averaging when you want a metric that reflects overall
    performance across all samples.  It is appropriate when class imbalance
    mirrors the true deployment distribution and you care about per-sample
    accuracy.

## Weighted Average

The **weighted average** computes a weighted mean of the per-class metrics,
with weights proportional to class size (support):

$$
P_{\text{weighted}} = \sum_{c=1}^{C}\frac{n_c}{n}\,P_c,
\qquad
R_{\text{weighted}} = \sum_{c=1}^{C}\frac{n_c}{n}\,R_c
$$

where $n_c = \text{TP}_c + \text{FN}_c$ is the number of true examples in
class $c$ and $n = \sum_c n_c$.  Weighted averaging is a compromise: it
accounts for class imbalance (like micro) while still computing per-class
metrics (like macro).

## Comparison

| Strategy | Weight per class | Sensitive to rare classes | Equals accuracy |
|---|---|---|---|
| Macro | $1/C$ (equal) | Yes | No |
| Micro | Proportional to support | No | Yes (single-label) |
| Weighted | $n_c / n$ | Intermediate | No |

??? example "Worked Example: 3-Class Imbalanced Problem"
    Consider a classifier evaluated on $n = 1000$ test samples with three
    classes:

    | Class | $n_c$ | $\text{TP}_c$ | $\text{FP}_c$ | $\text{FN}_c$ | $P_c$ | $R_c$ | $F_{1,c}$ |
    |---|---|---|---|---|---|---|---|
    | A | 800 | 760 | 30 | 40 | 0.962 | 0.950 | 0.956 |
    | B | 150 | 100 | 40 | 50 | 0.714 | 0.667 | 0.690 |
    | C | 50 | 20 | 10 | 30 | 0.667 | 0.400 | 0.500 |

    **Macro-average:**

    $$
    P_{\text{macro}} = \frac{0.962 + 0.714 + 0.667}{3} = 0.781
    $$

    $$
    F_{1,\text{macro}} = \frac{0.956 + 0.690 + 0.500}{3} = 0.715
    $$

    **Micro-average:**

    $$
    P_{\text{micro}} = \frac{760 + 100 + 20}{760 + 100 + 20 + 30 + 40 + 10} = \frac{880}{960} = 0.917
    $$

    **Weighted average:**

    $$
    P_{\text{weighted}} = \frac{800}{1000}(0.962) + \frac{150}{1000}(0.714) + \frac{50}{1000}(0.667) = 0.910
    $$

    The macro-average (0.781 precision) is much lower than the micro-average
    (0.917) because it gives equal weight to class C, where the model performs
    poorly.  Which number to report depends on whether rare-class performance
    is a priority.

## Practical Recommendations

1. **Report all three** in research papers so readers can assess performance
   from different perspectives.
2. **Macro** for applications where every class matters equally.
3. **Micro** when overall per-sample correctness is the primary goal.
4. **Weighted** as a balanced default in scikit-learn's `classification_report`.
5. When classes are approximately balanced, all three averages converge to
   similar values, and the choice matters less.
