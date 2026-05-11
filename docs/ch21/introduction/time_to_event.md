# Time-to-Event Data and Censoring

In many applications the outcome of interest is not a binary label or a continuous
measurement, but the **time until an event occurs**.  A hospital may track how long
a patient survives after surgery; a bank may record the number of months until a
borrower defaults on a loan; an engineer may measure the operating hours before a
turbine blade fails.  These durations, called *survival times* or *event times*,
require their own statistical framework because the data carry a complication that
ordinary regression ignores: **censoring**.

This section defines survival data, introduces the censoring problem, and
establishes the notation used throughout the chapter.

## What Is Survival Data?

A survival dataset consists of $n$ subjects, each associated with two quantities:

1. An observed time $t_i > 0$.
2. An event indicator $\delta_i \in \{0, 1\}$, where $\delta_i = 1$ means the
   event of interest was observed at time $t_i$, and $\delta_i = 0$ means the
   observation was **censored** at time $t_i$.

The random variable of interest is the *true event time* $T_i$, but for censored
subjects we only know that $T_i > t_i$.

!!! example "Survival Data Layout"

    | Subject | Time (months) | Event ($\delta$) | Interpretation |
    |:-------:|:-------------:|:----------------:|:---------------|
    | 1       | 8             | 1                | Defaulted at month 8 |
    | 2       | 14            | 0                | Still active at month 14 (censored) |
    | 3       | 3             | 1                | Defaulted at month 3 |
    | 4       | 12            | 0                | Lost to follow-up at month 12 (censored) |

## The Censoring Problem

Censoring arises whenever the event has not yet been observed for some subjects
by the time the study ends or the subject leaves the study.  Ignoring censored
observations---either by dropping them or by treating them as events---introduces
bias.

- **Dropping censored subjects** discards information: subject 2 in the table
  above survived *at least* 14 months, which is informative.
- **Treating censored times as event times** underestimates the true survival
  time because it records the censoring time as if the event happened then.

Survival analysis solves this problem by incorporating censored observations
into the likelihood function, using the fact that a censored subject contributes
the information $P(T > t_i)$ rather than $P(T = t_i)$.

## Right Censoring

The most common form of censoring in practice is **right censoring**, where the
event is known to occur *after* the observed time.  Right censoring occurs in
three main settings:

1. **End-of-study censoring.** The study ends at a fixed calendar date, and some
   subjects have not yet experienced the event.
2. **Loss to follow-up.** A subject leaves the study before the event occurs
   (e.g., a patient moves to another city).
3. **Competing events.** The subject experiences a different event that prevents
   observation of the event of interest (e.g., death from an unrelated cause in a
   cancer study).

Formally, each subject has a true event time $T_i$ and a censoring time $C_i$.
The observed data are

$$
t_i = \min(T_i, C_i), \qquad \delta_i = \mathbf{1}(T_i \leq C_i)
$$

where $\mathbf{1}(\cdot)$ is the indicator function.

## Independent Censoring Assumption

Almost all standard survival methods require the **independent censoring
assumption**: the censoring mechanism carries no information about the event
time.  Formally, for every subject $i$,

$$
T_i \perp C_i
$$

or, in the presence of covariates $\mathbf{x}_i$,

$$
T_i \perp C_i \mid \mathbf{x}_i
$$

This assumption means that, given the covariates, a subject who is censored at
time $t$ is representative of all subjects still at risk at time $t$.

!!! warning "When Independent Censoring Fails"

    If sicker patients are more likely to drop out, the remaining subjects are
    healthier on average, and the estimated survival curve is biased upward.
    Checking this assumption is difficult because the true event times for
    censored subjects are unobserved.

## Examples Across Domains

Survival analysis applies wherever the outcome is a duration:

- **Medicine.** Time from diagnosis to death, relapse, or recovery.
- **Finance.** Time from loan origination to default; time from account opening
  to customer churn; trade duration in market microstructure.
- **Engineering.** Time from deployment to component failure (reliability
  analysis).
- **Social science.** Duration of unemployment spells; time to recidivism after
  release from prison.

In each case, censoring is the rule rather than the exception.  Clinical trials
end on a calendar date, loan portfolios contain active accounts, and machines
are still running when the maintenance report is written.

## Notation Summary

The following notation is used throughout the chapter.

| Symbol | Meaning |
|:------:|:--------|
| $T$ | True (possibly unobserved) event time |
| $C$ | Censoring time |
| $t_i$ | Observed time for subject $i$: $\min(T_i, C_i)$ |
| $\delta_i$ | Event indicator: 1 = event observed, 0 = censored |
| $n$ | Total number of subjects |
| $d$ | Total number of observed events: $d = \sum_{i=1}^{n} \delta_i$ |
| $S(t)$ | Survival function: $P(T > t)$ |
| $h(t)$ | Hazard function: instantaneous event rate at time $t$ |
| $H(t)$ | Cumulative hazard function: $\int_0^t h(u)\,du$ |

??? tip "Connection to Previous Chapters"

    The likelihood-based reasoning developed in Chapter 6 (Statistical
    Estimation) carries over directly.  The key difference is that censored
    observations contribute a survival probability $S(t_i)$ rather than a
    density value $f(t_i)$ to the likelihood.  Section 21.3 develops this
    idea formally.


## Exercises

**Exercise 1.**
Describe the main concept of Time-to-Event Data and Censoring and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Time-to-Event Data and Censoring is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

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
