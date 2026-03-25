# Log-Rank Test

The Kaplan--Meier estimator provides a separate survival curve for each group.
The natural next question is whether the observed difference between two (or
more) survival curves reflects a genuine difference in the underlying survival
distributions or is simply due to sampling variability.  The **log-rank test**
is the standard non-parametric hypothesis test for this comparison.

This section states the hypotheses, derives the test statistic, and works
through a two-group example.

## Hypotheses

For two groups (e.g., treatment vs control), the log-rank test evaluates

$$
H_0 : S_1(t) = S_2(t) \quad \text{for all } t \geq 0
$$

$$
H_1 : S_1(t) \neq S_2(t) \quad \text{for some } t \geq 0
$$

Under $H_0$, the two groups share the same survival distribution, meaning
any observed difference in their Kaplan--Meier curves is due to chance.

## Setup

Pool the event times from both groups and order the $K$ distinct event times:
$t_{(1)} < t_{(2)} < \cdots < t_{(K)}$.  At each event time $t_{(j)}$, record:

| Quantity | Group 1 | Group 2 | Total |
|:---------|:-------:|:-------:|:-----:|
| Events | $d_{1j}$ | $d_{2j}$ | $d_j$ |
| At risk | $n_{1j}$ | $n_{2j}$ | $n_j$ |

## Expected Events Under the Null

Under $H_0$, events at each time $t_{(j)}$ are allocated to the two groups in
proportion to their risk sets.  The expected number of events in group 1 at
time $t_{(j)}$ is

$$
e_{1j} = d_j \cdot \frac{n_{1j}}{n_j}
$$

This is the expectation of a hypergeometric distribution: $d_j$ events are
drawn without replacement from a risk set of $n_j$ subjects, of which $n_{1j}$
belong to group 1.

The total expected number of events in group 1 is

$$
E_1 = \sum_{j=1}^{K} e_{1j} = \sum_{j=1}^{K} d_j \cdot \frac{n_{1j}}{n_j}
$$

## Test Statistic

The log-rank test statistic compares the observed total events in group 1,
$O_1 = \sum_{j=1}^{K} d_{1j}$, with the expected total $E_1$.  The
variance under $H_0$ is

$$
V_1 = \sum_{j=1}^{K} \frac{n_{1j} \, n_{2j} \, d_j \, (n_j - d_j)}{n_j^2 \, (n_j - 1)}
$$

The test statistic is

$$
\chi^2_{\text{LR}} = \frac{(O_1 - E_1)^2}{V_1}
$$

Under $H_0$, this statistic follows approximately a chi-squared distribution
with 1 degree of freedom:

$$
\chi^2_{\text{LR}} \;\xrightarrow{d}\; \chi^2_1
$$

The p-value is $P(\chi^2_1 \geq \chi^2_{\text{LR}})$.

!!! note "Equivalent Formulation"

    Because $O_1 + O_2 = d$ and $E_1 + E_2 = d$ (the totals are fixed), the
    test statistic based on group 2 gives the same result.  The two-group
    log-rank test uses only 1 degree of freedom regardless of which group is
    chosen.

## Worked Example

Two groups of patients are followed after diagnosis.

**Group A** (treatment): times 2, 4+, 6, 8+, 10 (+ denotes censored).

**Group B** (control): times 1, 3, 5+, 7, 9.

Pooled event times (excluding censorings): 1, 2, 3, 6, 7, 9, 10.

| $t_{(j)}$ | $n_{Aj}$ | $n_{Bj}$ | $n_j$ | $d_{Aj}$ | $d_{Bj}$ | $d_j$ | $e_{Aj}$ |
|:----------:|:--------:|:--------:|:-----:|:--------:|:--------:|:-----:|:--------:|
| 1 | 5 | 5 | 10 | 0 | 1 | 1 | 0.500 |
| 2 | 5 | 4 | 9 | 1 | 0 | 1 | 0.556 |
| 3 | 4 | 4 | 8 | 0 | 1 | 1 | 0.500 |
| 6 | 3 | 2 | 5 | 1 | 0 | 1 | 0.600 |
| 7 | 2 | 2 | 4 | 0 | 1 | 1 | 0.500 |
| 9 | 1 | 1 | 2 | 0 | 1 | 1 | 0.500 |
| 10 | 1 | 0 | 1 | 1 | 0 | 1 | 1.000 |

Totals: $O_A = 3$, $E_A = 4.156$.

The observed count for group A (3 events) is fewer than expected (4.156),
suggesting group A may have better survival.  Computing the variance $V_A$
and the test statistic yields the p-value for formal inference.

## Extension to More Than Two Groups

For $G$ groups, the log-rank test generalizes to a multivariate chi-squared
test with $G - 1$ degrees of freedom.  Define the vector of observed-minus-expected
counts for groups $1, \ldots, G-1$:

$$
\mathbf{U} = \begin{pmatrix} O_1 - E_1 \\ O_2 - E_2 \\ \vdots \\ O_{G-1} - E_{G-1} \end{pmatrix}
$$

and let $\mathbf{V}$ be the $(G-1) \times (G-1)$ variance-covariance matrix
of $\mathbf{U}$.  The test statistic is

$$
\chi^2_{\text{LR}} = \mathbf{U}^\top \mathbf{V}^{-1} \mathbf{U} \;\xrightarrow{d}\; \chi^2_{G-1}
$$

## Assumptions and Limitations

1. **Independent censoring.** The censoring mechanism must not depend on the
   event time, conditional on covariates.
2. **Non-informative censoring.** Censored subjects at any time are
   representative of all subjects at risk.
3. **Proportional hazards.** The log-rank test has optimal power when the
   hazard ratio between groups is constant over time.  If hazards cross
   (e.g., one treatment is better early but worse late), the log-rank test
   may fail to detect the difference.

!!! warning "Crossing Hazards"

    When survival curves cross, the log-rank test can yield a non-significant
    p-value even when the curves differ substantially.  In such cases, consider
    the Wilcoxon (Gehan--Breslow) test, which gives more weight to early
    event times, or a stratified analysis.

## Weighted Log-Rank Tests

The standard log-rank test weights all event times equally.  **Weighted
variants** apply time-dependent weights $w_j$ at each event time:

$$
\chi^2_w = \frac{\left(\sum_{j=1}^K w_j (d_{1j} - e_{1j})\right)^2}{\sum_{j=1}^K w_j^2 \, v_{1j}}
$$

Common choices include:

| Test Name | Weight $w_j$ | Sensitivity |
|:----------|:------------|:------------|
| Log-rank (Mantel--Haenszel) | $1$ | Late differences |
| Wilcoxon (Gehan--Breslow) | $n_j$ | Early differences |
| Tarone--Ware | $\sqrt{n_j}$ | Moderate balance |
| Peto--Peto | $\hat{S}(t_{(j)})$ | Early-to-mid differences |

The choice of weights should be guided by the scientific question, not by
which test gives the smallest p-value.
