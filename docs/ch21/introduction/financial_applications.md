# Financial Applications of Survival Analysis

Survival analysis originated in medical research and reliability engineering,
but its tools apply wherever the outcome is a duration subject to censoring.
Finance is rich with such outcomes: the time until a borrower defaults, the
time until a customer churns, the duration of a trade, and the time until an
insurance claim is filed.  In each case, the portfolio or customer base contains
active (right-censored) observations alongside completed events.

This section surveys four financial applications where survival methods provide
insights that conventional regression cannot.

## Credit Default Modeling

A bank holds a portfolio of $n$ loans.  For each loan $i$, the event of interest
is **default**---the borrower fails to meet payment obligations.  The survival
time $T_i$ is the duration from origination to default.

**Why survival analysis?**  At any reporting date, most loans are still
performing (right-censored).  Logistic regression can model the probability of
default within a fixed horizon (e.g., 12 months), but it discards timing
information and cannot produce a full default-time distribution.

The hazard function $h(t)$ for credit default often exhibits a characteristic
shape:

- **Early period (0--12 months).** Low hazard as borrowers have recently been
  screened and approved.
- **Seasoning period (12--36 months).** Rising hazard as financial shocks
  accumulate and weak borrowers begin to default.
- **Mature period (36+ months).** Declining or stable hazard among surviving
  borrowers, who have demonstrated creditworthiness.

This hump-shaped hazard is well captured by log-normal or log-logistic models
(Section 21.3) or by a Cox model with time-varying covariates (Section 21.4).

!!! example "Survival Curve for a Loan Portfolio"

    Suppose a Kaplan--Meier estimate yields $\hat{S}(12) = 0.97$,
    $\hat{S}(24) = 0.93$, and $\hat{S}(36) = 0.90$.  This means that an
    estimated 97% of loans survive past 12 months, 93% past 24 months, and
    90% past 36 months.  The conditional default probability between months
    24 and 36, given survival to month 24, is
    $1 - \hat{S}(36)/\hat{S}(24) = 1 - 0.90/0.93 \approx 0.032$.

## Customer Churn Analysis

Subscription-based businesses (banks, telecom providers, SaaS platforms) track
the time until a customer **churns**---cancels the service or closes the account.
The survival time is the duration of the customer relationship.

Key features of churn data:

- **Right censoring is pervasive.** Active customers are censored at the
  analysis date.
- **Covariates evolve over time.** Usage patterns, complaint frequency, and
  payment behavior change monthly.  The Cox model accommodates time-varying
  covariates naturally.
- **Competing risks.** A customer may leave voluntarily (churn) or be
  terminated by the company (involuntary closure).  These are distinct events
  that require competing-risk models for rigorous analysis.

The hazard function for churn often decreases with tenure: customers who
survive the first few months tend to stay longer.  This suggests a Weibull
model with shape parameter $k < 1$ (decreasing hazard).

!!! tip "Hazard Ratios for Churn Drivers"

    A Cox model fitted to churn data might yield a hazard ratio of
    $\text{HR} = 1.45$ for customers who contacted support more than three
    times in a quarter.  This means such customers churn at 1.45 times the
    rate of those with fewer contacts, holding other covariates constant.

## Trade Duration Analysis

In market microstructure, the **duration** between consecutive transactions is
itself the variable of interest.  The Autoregressive Conditional Duration (ACD)
model, introduced by Engle and Russell (1998), adapts survival analysis ideas to
model the time between trades.

Let $x_i$ denote the duration between trade $i-1$ and trade $i$.  The
conditional hazard of the next trade arriving, given the history of past
durations, captures the intensity of trading activity.

- **High hazard** periods correspond to rapid trading (high liquidity, volatile
  markets).
- **Low hazard** periods correspond to slow trading (low liquidity, calm
  markets).

The exponential model serves as a baseline (constant arrival rate), while the
Weibull model allows the arrival intensity to depend on elapsed time since the
last trade.

!!! note "Connection to Point Processes"

    Trade arrival times form a point process on the positive real line.  The
    hazard function $h(t)$ is the conditional intensity of the process, linking
    survival analysis to the broader theory of counting processes and
    martingales.

## Insurance Claims and Duration

Insurance companies model the time until a claim is filed after a policy is
issued.  Survival analysis is essential because:

- **Policies lapse.** A policyholder who cancels before filing a claim is
  right-censored.
- **Claim frequency varies with time.** Auto insurance claims often cluster in
  the first year (inexperience), then decline.
- **Loss reserving.** Actuaries use survival models to estimate the number of
  claims that have been incurred but not yet reported (IBNR).

The hazard function for claim filing depends on the insurance line:

| Insurance Line | Typical Hazard Shape | Model Choice |
|:---------------|:---------------------|:-------------|
| Auto liability | Decreasing after initial peak | Log-logistic, Weibull ($k < 1$) |
| Life insurance | Increasing with age | Weibull ($k > 1$), Gompertz |
| Property (catastrophe) | Constant between events | Exponential |

## Why Not Just Use Logistic Regression?

Logistic regression models a binary outcome (event vs no event) within a fixed
time window.  Survival analysis extends this in three ways:

1. **Uses all available follow-up time.** A subject observed for 6 months
   contributes 6 months of information, even if the study window is 12 months.
2. **Handles variable follow-up.** Subjects enter the study at different times
   and are observed for different durations.
3. **Models the timing of the event.** The survival and hazard functions
   describe *when* the event occurs, not just *whether* it occurs.

!!! warning "Bias from Ignoring Censoring"

    Fitting a logistic regression to predict 12-month default while excluding
    loans that are only 6 months old throws away useful data.  Including them
    as "no default" biases the default probability downward because these loans
    have not yet had the opportunity to default in months 7--12.  Survival
    analysis avoids this bias by explicitly modeling the censoring mechanism.

## Summary

| Application | Event | Censoring Source | Typical Hazard Shape |
|:------------|:------|:-----------------|:---------------------|
| Credit default | Loan default | Active loans | Hump-shaped |
| Customer churn | Service cancellation | Active customers | Decreasing |
| Trade duration | Next trade arrival | End of trading day | Varies with market regime |
| Insurance claims | Claim filing | Policy lapse | Depends on insurance line |

Survival methods unify these applications under a common mathematical framework.
The tools developed in the remainder of this chapter---Kaplan--Meier estimation,
parametric models, and the Cox proportional hazards model---apply directly to
all four settings.


## Exercises

**Exercise 1.**
Describe the main concept of Financial Applications of Survival Analysis and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Financial Applications of Survival Analysis is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

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
