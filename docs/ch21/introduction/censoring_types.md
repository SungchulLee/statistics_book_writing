# Types of Censoring

The previous section introduced censoring as the defining challenge of survival
data: some subjects' true event times are only partially observed.  Not all
censoring is alike, however.  The mechanism that hides the event time determines
what information remains and which statistical methods are appropriate.

This section classifies the three main censoring types---right, left, and
interval---and discusses the independent censoring assumption that underpins
valid inference.

## Right Censoring

Right censoring is the most common type in practice.  A subject is
**right-censored** when the event is known to occur *after* the observed time,
but the exact event time is unknown.

Formally, let $T$ denote the true event time and $C$ the censoring time.  The
observed data for subject $i$ are

$$
t_i = \min(T_i, C_i), \qquad \delta_i = \mathbf{1}(T_i \leq C_i)
$$

When $\delta_i = 0$, the subject was censored: all we know is $T_i > t_i$.

Right censoring arises in several ways:

- **Type I censoring.** The study has a fixed end date.  All subjects who have
  not experienced the event by that date are censored at the same calendar time
  (though their observed durations may differ if they entered the study at
  different times).
- **Type II censoring.** The study continues until a predetermined number of
  events $r$ have been observed, then stops.  The remaining $n - r$ subjects are
  censored.
- **Random censoring.** Each subject has an independent censoring time $C_i$
  (e.g., loss to follow-up).  This is the most general and most common setting.

!!! example "Right Censoring in a Loan Portfolio"

    A bank tracks 1,000 loans originated in January 2020.  By December 2023,
    200 loans have defaulted (events) and 800 remain active (right-censored).
    For each active loan, the bank knows the borrower has survived *at least*
    $t_i$ months, but the eventual default time is unknown.

## Left Censoring

A subject is **left-censored** when the event is known to have occurred *before*
the observation time, but the exact event time is unknown.  The observed
information is $T_i < t_i$.

Left censoring is less common than right censoring but arises in specific
settings:

- **Delayed detection.** A medical test at time $t$ reveals that a disease is
  already present.  The onset occurred before $t$, but the exact time is
  unknown.
- **Detection limits.** An environmental measurement falls below the instrument's
  detection threshold, indicating the true concentration is somewhere between
  zero and the threshold.

!!! example "Left Censoring in Disease Screening"

    A routine blood test at age 50 reveals elevated PSA levels indicating
    prostate cancer.  The cancer developed at some unknown time $T < 50$, so
    the observation is left-censored at 50.

## Interval Censoring

A subject is **interval-censored** when the event is known to have occurred
within a time window $(L_i, R_i]$, but the exact event time within that interval
is unknown.

$$
L_i < T_i \leq R_i
$$

Interval censoring occurs naturally when subjects are examined at periodic
visits rather than monitored continuously.

!!! example "Interval Censoring in a Clinical Trial"

    A patient is examined at months 6 and 12.  At month 6 the patient is
    disease-free; at month 12 the disease is present.  The event time satisfies
    $6 < T \leq 12$, but the exact month is unknown.

Note that right censoring and left censoring are special cases of interval
censoring:

- Right censoring: $L_i = t_i$ and $R_i = \infty$.
- Left censoring: $L_i = 0$ and $R_i = t_i$.

## Truncation vs Censoring

Truncation is sometimes confused with censoring but describes a different
phenomenon.  A subject is **truncated** when its existence is entirely unknown
to the analyst unless a condition is met.

- **Left truncation (delayed entry).** A subject enters the study at time $a_i$
  and is only observed if $T_i > a_i$.  Subjects who experienced the event
  before entering the study are never recorded.
- **Right truncation.** Only subjects who have already experienced the event
  are included (e.g., a registry of confirmed cases).

The key distinction is that censoring provides partial information about the
event time, while truncation provides no information at all for excluded
subjects.

| Mechanism | What is observed | What is unknown |
|:----------|:-----------------|:----------------|
| Right censoring | $T > t$ | Exact $T$ |
| Left censoring | $T < t$ | Exact $T$ |
| Interval censoring | $L < T \leq R$ | Exact $T$ |
| Left truncation | Subject exists only if $T > a$ | Subjects with $T \leq a$ |

## Independent Censoring Assumption

Valid inference in survival analysis requires that censoring be **non-informative**:
the censoring mechanism must not depend on the unobserved event time.  Formally,

$$
T \perp C \mid \mathbf{x}
$$

where $\mathbf{x}$ is the vector of observed covariates.  This means that at any
time $t$, a censored subject is representative of all subjects still at risk,
conditional on covariates.

!!! warning "Informative Censoring Invalidates Standard Methods"

    If patients who are deteriorating rapidly are more likely to drop out of a
    clinical trial, the censoring depends on the (unobserved) event time.
    Standard Kaplan--Meier and Cox model estimates will be biased in this
    setting.  Sensitivity analyses or joint models for the event and dropout
    processes are needed.

## Summary

| Type | Notation | Typical Setting |
|:-----|:---------|:----------------|
| Right censoring | $T > t$ | End of study, loss to follow-up |
| Left censoring | $T < t$ | Delayed detection, threshold instruments |
| Interval censoring | $L < T \leq R$ | Periodic examinations |
| Left truncation | Only observe if $T > a$ | Delayed entry into study |

Right censoring dominates in practice and is the default assumption throughout
the remainder of this chapter.  The Kaplan--Meier estimator (Section 21.2),
parametric models (Section 21.3), and the Cox model (Section 21.4) all assume
right-censored data unless stated otherwise.


## Exercises

**Exercise 1.**
Describe the main concept of Types of Censoring and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Types of Censoring is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

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
