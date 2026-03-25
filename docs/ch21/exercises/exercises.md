# Exercises

## Survival Models

Work through the following exercises to test your understanding of survival
analysis concepts from this chapter.

## Exercise 1: Survival and Hazard Function Relationships

A continuous random variable $T$ has hazard function $h(t) = 0.05$ for all
$t \geq 0$.

**(a)** Identify the distribution of $T$ and state its parameter.

**(b)** Compute the survival function $S(t)$ and evaluate $S(10)$.

**(c)** Compute the cumulative hazard $H(20)$.

**(d)** What is the median survival time?

**Solution:**

**(a)** The constant hazard $h(t) = 0.05$ implies $T \sim \text{Exp}(\lambda = 0.05)$.

**(b)** $S(t) = e^{-0.05t}$, so $S(10) = e^{-0.5} \approx 0.607$.

**(c)** $H(20) = 0.05 \times 20 = 1.0$.

**(d)** Solve $S(t_{0.5}) = 0.5$: $e^{-0.05 t_{0.5}} = 0.5$, so
$t_{0.5} = \ln(2)/0.05 = 13.86$.

## Exercise 2: Kaplan-Meier Estimation

Ten patients are followed after diagnosis. Their observed times (in months) and
event indicators ($\delta$: 1 = death, 0 = censored) are:

| Subject | Time | $\delta$ |
|:-------:|:----:|:--------:|
| 1 | 3 | 1 |
| 2 | 5 | 0 |
| 3 | 7 | 1 |
| 4 | 8 | 1 |
| 5 | 10 | 0 |
| 6 | 12 | 1 |
| 7 | 12 | 0 |
| 8 | 15 | 1 |
| 9 | 18 | 0 |
| 10 | 20 | 1 |

**(a)** Compute the Kaplan--Meier estimate $\hat{S}(t)$ at each event time.

**(b)** What is the estimated median survival time?

**Solution:**

**(a)** Distinct event times: 3, 7, 8, 12, 15, 20.

| $t_{(j)}$ | $n_j$ | $d_j$ | $1 - d_j/n_j$ | $\hat{S}(t_{(j)})$ |
|:----------:|:-----:|:-----:|:--------------:|:-------------------:|
| 3 | 10 | 1 | 0.900 | 0.900 |
| 7 | 8 | 1 | 0.875 | 0.788 |
| 8 | 7 | 1 | 0.857 | 0.675 |
| 12 | 5 | 1 | 0.800 | 0.540 |
| 15 | 3 | 1 | 0.667 | 0.360 |
| 20 | 1 | 1 | 0.000 | 0.000 |

**(b)** $\hat{S}(12) = 0.540 > 0.5$ and $\hat{S}(15) = 0.360 < 0.5$, so
the median survival time is $\hat{t}_{0.5} = 15$ months.

## Exercise 3: Nelson-Aalen Estimator

Using the data from Exercise 2, compute the Nelson--Aalen estimate of the
cumulative hazard $\hat{H}(t)$ at each event time and compare
$\exp(-\hat{H}(12))$ with $\hat{S}_{\text{KM}}(12)$.

**Solution:**

| $t_{(j)}$ | $d_j/n_j$ | $\hat{H}(t_{(j)})$ |
|:----------:|:---------:|:-------------------:|
| 3 | 0.100 | 0.100 |
| 7 | 0.125 | 0.225 |
| 8 | 0.143 | 0.368 |
| 12 | 0.200 | 0.568 |
| 15 | 0.333 | 0.901 |
| 20 | 1.000 | 1.901 |

$\exp(-\hat{H}(12)) = \exp(-0.568) = 0.567$ vs
$\hat{S}_{\text{KM}}(12) = 0.540$. The Nelson--Aalen-based estimate is
slightly higher, as expected.

## Exercise 4: Log-Rank Test

Two groups of patients (A: new treatment, B: standard care) have the following
survival data:

**Group A:** 4, 7+, 10, 14+, 18 (+ denotes censored).

**Group B:** 2, 5, 9+, 11, 16.

**(a)** State the null and alternative hypotheses.

**(b)** At event time $t = 2$, compute the expected events $e_{A1}$ for group A.

**(c)** Compute $O_A$ and $E_A$ across all event times and comment on the
direction of the difference.

**Solution:**

**(a)** $H_0: S_A(t) = S_B(t)$ for all $t$ vs $H_1: S_A(t) \neq S_B(t)$ for
some $t$.

**(b)** At $t = 2$: $n_A = 5$, $n_B = 5$, $n = 10$, $d = 1$.
$e_{A1} = 1 \times 5/10 = 0.5$.

**(c)** $O_A = 3$ events observed in group A. Computing expected events at
each event time and summing yields $E_A \approx 3.5$. Since $O_A < E_A$,
group A has fewer events than expected, suggesting better survival with the
new treatment.

## Exercise 5: Exponential Model MLE

A reliability study tracks 30 components. By the study end, 18 have failed and
12 are still operating. The total observed time (events + censored) is
$\sum t_i = 4{,}200$ hours.

**(a)** Compute the MLE $\hat{\lambda}$ assuming an exponential model.

**(b)** Estimate the mean time to failure.

**(c)** Compute a 95% confidence interval for $\lambda$.

**Solution:**

**(a)** $\hat{\lambda} = d / \sum t_i = 18/4200 = 0.00429$ per hour.

**(b)** $1/\hat{\lambda} = 233.3$ hours.

**(c)** $\text{se}(\hat{\lambda}) = \hat{\lambda}/\sqrt{d} = 0.00429/\sqrt{18} = 0.00101$.
CI: $0.00429 \pm 1.96 \times 0.00101 = (0.00231, 0.00627)$.

## Exercise 6: Weibull Shape Parameter

A Weibull model fitted to time-to-default data yields $\hat{k} = 0.75$ and
$\hat{\lambda} = 36$ months.

**(a)** Is the hazard increasing or decreasing over time?  Explain.

**(b)** Compute the median time to default.

**(c)** Compute $\hat{S}(24)$, the probability of surviving past 24 months.

**Solution:**

**(a)** Since $\hat{k} = 0.75 < 1$, the hazard is **decreasing** over time.
Default risk is highest early and declines among surviving borrowers.

**(b)** $t_{0.5} = \lambda(\ln 2)^{1/k} = 36 \times (0.693)^{1/0.75} = 36 \times 0.693^{1.333} = 36 \times 0.627 = 22.6$ months.

**(c)** $\hat{S}(24) = \exp(-(24/36)^{0.75}) = \exp(-0.667^{0.75}) = \exp(-0.740) = 0.477$.

## Exercise 7: Cox Model Interpretation

A Cox model for employee turnover includes three covariates. The estimated
coefficients and standard errors are:

| Covariate | $\hat{\beta}$ | $\text{se}(\hat{\beta})$ |
|:----------|:-------------:|:------------------------:|
| Salary (per \$10k) | $-0.18$ | $0.06$ |
| Remote work (1 = yes) | $-0.42$ | $0.15$ |
| Manager (1 = yes) | $0.31$ | $0.12$ |

**(a)** Compute and interpret the hazard ratio for each covariate.

**(b)** Compute a 95% confidence interval for the hazard ratio of remote work.

**(c)** Which covariates are significant at the 5% level?

**Solution:**

**(a)**

- Salary: $\text{HR} = e^{-0.18} = 0.835$. Each \$10k increase in salary is
  associated with a 16.5% reduction in turnover hazard.
- Remote work: $\text{HR} = e^{-0.42} = 0.657$. Remote workers have a 34.3%
  lower turnover hazard than on-site workers.
- Manager: $\text{HR} = e^{0.31} = 1.363$. Managers have a 36.3% higher
  turnover hazard than non-managers.

**(b)** CI for $\beta$: $-0.42 \pm 1.96 \times 0.15 = (-0.714, -0.126)$.
CI for HR: $(e^{-0.714}, e^{-0.126}) = (0.490, 0.882)$. Since the interval
excludes 1, remote work significantly reduces turnover.

**(c)** Wald test: $|z| = |\hat{\beta}|/\text{se}$. Salary: $3.00$;
Remote: $2.80$; Manager: $2.58$. All three exceed $z_{0.025} = 1.96$, so
all three are significant at the 5% level.

## Exercise 8: Proportional Hazards Assumption

A Schoenfeld test for a Cox model with two covariates (age and treatment)
yields:

| Covariate | $\chi^2$ | p-value |
|:----------|:--------:|:-------:|
| Age | 1.23 | 0.267 |
| Treatment | 6.89 | 0.009 |
| GLOBAL | 7.54 | 0.023 |

**(a)** For which covariate(s) does the PH assumption appear to be violated?

**(b)** Suggest two remedies for the violation.

**Solution:**

**(a)** The PH assumption is violated for treatment ($p = 0.009 < 0.05$) but
not for age ($p = 0.267$). The global test also rejects ($p = 0.023$),
confirming at least one violation.

**(b)** Two remedies:

1. **Stratification:** Fit a stratified Cox model with separate baseline
   hazards for each treatment group, estimating only the age effect as a
   regression coefficient.
2. **Time-varying coefficient:** Include a treatment $\times \ln(t)$
   interaction term to allow the treatment effect to change over time.

## Exercise 9: Greenwood's Formula

Using the Kaplan--Meier estimates from Exercise 2, compute the standard error
of $\hat{S}(8)$ using Greenwood's formula and construct a 95% pointwise
confidence interval using the linear method.

**Solution:**

$\hat{S}(8) = 0.675$. Greenwood's sum through $t = 8$:

$$
\frac{1}{10 \times 9} + \frac{1}{8 \times 7} + \frac{1}{7 \times 6} = 0.0111 + 0.0179 + 0.0238 = 0.0528
$$

$$
\text{se}(\hat{S}(8)) = 0.675 \times \sqrt{0.0528} = 0.675 \times 0.2298 = 0.155
$$

95% CI (linear): $0.675 \pm 1.96 \times 0.155 = (0.371, 0.979)$.

## Exercise 10: Model Selection

Three parametric models are fitted to the same dataset of 200 subjects with
120 events:

| Model | Parameters ($p$) | Log-Likelihood |
|:------|:----------------:|:--------------:|
| Exponential | 1 | $-458.2$ |
| Weibull | 2 | $-441.6$ |
| Log-logistic | 2 | $-443.1$ |

**(a)** Compute the AIC for each model.

**(b)** Perform a likelihood ratio test of exponential vs Weibull at
$\alpha = 0.05$.

**(c)** A Cox model fitted to the same data yields a concordance index of
$C = 0.72$, while the Weibull model yields $C = 0.74$. Which model
discriminates better?

**Solution:**

**(a)** AIC = $-2\ell + 2p$:

- Exponential: $-2(-458.2) + 2(1) = 918.4$
- Weibull: $-2(-441.6) + 2(2) = 887.2$
- Log-logistic: $-2(-443.1) + 2(2) = 890.2$

The Weibull model has the lowest AIC (887.2).

**(b)** $\Lambda = 2(-441.6 - (-458.2)) = 2 \times 16.6 = 33.2$. Under $H_0$,
$\Lambda \sim \chi^2_1$. Since $33.2 \gg 3.84 = \chi^2_{1, 0.05}$, we reject
$H_0$. The Weibull model fits significantly better than the exponential.

**(c)** The Weibull model ($C = 0.74$) discriminates slightly better than the
Cox model ($C = 0.72$), suggesting that the Weibull distributional assumption
is appropriate for this dataset and provides a small gain in predictive accuracy.
