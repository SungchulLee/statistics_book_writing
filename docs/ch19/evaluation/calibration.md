# Calibration and Brier Score

## Why Calibration Matters

A classifier may have high accuracy yet produce misleading probability
estimates.  For example, a model that assigns $\hat{p} = 0.90$ to a group of
applicants should see roughly 90% of them default if the model is well
calibrated.  **Calibration** measures whether predicted probabilities match
observed frequencies.  Good calibration is essential in applications such as
credit scoring and medical diagnosis where the probabilities themselves — not
just the binary decisions — drive downstream actions.

## Definition of Calibration

A model is **perfectly calibrated** if, for every predicted probability $q$,

$$
P(Y = 1 \mid \hat{p} = q) = q
$$

In words: among all observations that receive a predicted probability of $q$,
the observed proportion of positives equals $q$.

## Reliability Diagrams (Calibration Curves)

Because predicted probabilities are continuous, we cannot check the condition
above for every $q$ individually.  Instead, we bin the predictions into $G$
groups (typically $G = 10$ deciles) and compare the average predicted
probability in each bin to the observed proportion of positives.

### Construction

1. Sort the $n$ predictions $\hat{p}_1, \ldots, \hat{p}_n$.
2. Divide into $G$ bins $B_1, \ldots, B_G$ of roughly equal size.
3. For each bin $g$, compute:
    - Mean predicted probability: $\bar{p}_g = \frac{1}{|B_g|}\sum_{i \in B_g}\hat{p}_i$
    - Observed fraction of positives: $\bar{y}_g = \frac{1}{|B_g|}\sum_{i \in B_g}y_i$
4. Plot $\bar{y}_g$ against $\bar{p}_g$.

A perfectly calibrated model lies on the **diagonal** $\bar{y} = \bar{p}$.
Points above the diagonal indicate under-prediction (the model is under-confident);
points below indicate over-prediction (the model is over-confident).

## Brier Score

The **Brier score** provides a single-number summary of calibration and
predictive accuracy:

$$
\text{BS} = \frac{1}{n}\sum_{i=1}^{n}(\hat{p}_i - y_i)^2
$$

The Brier score ranges from 0 (perfect) to 1 (worst possible).  It equals the
**mean squared error** of the predicted probabilities treated as point forecasts
for the binary outcomes.

### Brier Score Decomposition

The Brier score can be decomposed into three components:

$$
\text{BS} = \underbrace{\frac{1}{n}\sum_{g=1}^{G}|B_g|\,(\bar{p}_g - \bar{y}_g)^2}_{\text{calibration (reliability)}}
- \underbrace{\frac{1}{n}\sum_{g=1}^{G}|B_g|\,\bar{y}_g(1-\bar{y}_g)}_{\text{resolution}}
+ \underbrace{\bar{y}(1-\bar{y})}_{\text{uncertainty}}
$$

- **Calibration (reliability):** Measures how far the calibration curve deviates
  from the diagonal.  Lower is better.
- **Resolution:** Measures how much the model's predictions vary across bins.
  Higher resolution is better (it is subtracted).
- **Uncertainty:** Depends only on the base rate $\bar{y}$ and is the same for
  all models on the same dataset.

## Hosmer-Lemeshow Test

The **Hosmer-Lemeshow test** formalizes the visual check of the reliability
diagram.  It tests the null hypothesis that the model is well calibrated.

### Procedure

1. Sort observations by $\hat{p}_i$ and form $G$ groups (usually $G = 10$).
2. For each group $g$, let $O_g = \sum_{i \in B_g} y_i$ be the observed count
   of positives and $E_g = \sum_{i \in B_g} \hat{p}_i$ be the expected count.
3. Compute the test statistic:

$$
\hat{C} = \sum_{g=1}^{G}\frac{(O_g - E_g)^2}{E_g(1 - E_g/|B_g|)}
$$

Under the null hypothesis of adequate fit, $\hat{C}$ follows approximately a
$\chi^2_{G-2}$ distribution.  A large value of $\hat{C}$ (small $p$-value)
indicates lack of fit.

!!! warning "Sensitivity to Binning"
    The Hosmer-Lemeshow test is sensitive to the number of groups $G$ and the
    binning strategy.  Different choices of $G$ can lead to different
    conclusions.  It is best used alongside the reliability diagram rather than
    as a standalone verdict.

## Calibration Techniques

When a model is poorly calibrated, post-hoc **recalibration** can improve the
probability estimates without retraining the entire model.

### Platt Scaling

Fit a logistic regression with the original model's log-odds as the single
predictor:

$$
\hat{p}_{\text{cal}} = \sigma(a \cdot f(\mathbf{x}) + b)
$$

where $f(\mathbf{x})$ is the original model's output and $a, b$ are learned on
a held-out calibration set.

### Isotonic Regression

A non-parametric alternative that fits a monotone non-decreasing function
mapping raw predictions to calibrated probabilities.  Isotonic regression is
more flexible than Platt scaling but requires more calibration data.

??? example "Worked Example"
    A logistic model produces the following binned results on a test set of
    $n = 1000$:

    | Bin | $\bar{p}_g$ | $\bar{y}_g$ | $|B_g|$ |
    |---|---|---|---|
    | 1 | 0.05 | 0.04 | 100 |
    | 2 | 0.15 | 0.12 | 100 |
    | 3 | 0.25 | 0.28 | 100 |
    | 4 | 0.35 | 0.33 | 100 |
    | 5 | 0.45 | 0.47 | 100 |
    | 6 | 0.55 | 0.52 | 100 |
    | 7 | 0.65 | 0.68 | 100 |
    | 8 | 0.75 | 0.73 | 100 |
    | 9 | 0.85 | 0.88 | 100 |
    | 10 | 0.95 | 0.93 | 100 |

    The predictions are close to the diagonal, indicating good calibration.
    The Brier score is

    $$
    \text{BS} = \frac{1}{1000}\sum_{i=1}^{1000}(\hat{p}_i - y_i)^2 \approx 0.21
    $$

    The reliability component is small because $\bar{p}_g \approx \bar{y}_g$
    in every bin.
