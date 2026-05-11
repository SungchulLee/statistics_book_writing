# Confounding and Causation Demo

## Overview

This page demonstrates how confounding variables create spurious correlations and how Simpson's paradox can reverse the apparent direction of a treatment effect. We simulate two scenarios: a confounded regression where a hidden common cause produces a misleading association, and a treatment effect analysis where the naive estimate has the wrong sign.

---

## Part 1: Confounded Regression

### The Data-Generating Process

Consider the DAG $T \leftarrow C \rightarrow Y$, where $C$ is a confounder, $T$ is the treatment, and $Y$ is the outcome. The treatment $T$ has **no causal effect** on $Y$; all association flows through $C$.

We generate data from:

$$
\begin{pmatrix} T \\ C \end{pmatrix} \sim \mathcal{N}\!\left(\begin{pmatrix} 0 \\ 0 \end{pmatrix}, \begin{pmatrix} 1 & \rho_{TC} \\ \rho_{TC} & 1 \end{pmatrix}\right), \qquad Y = C + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, 1)
$$

Since $Y$ depends only on $C$, the true causal effect of $T$ on $Y$ is zero.

```python
import numpy as np
from scipy import stats

np.random.seed(42)


def simulate_confounded_data(n=500, rho_tc=0.8):
    mean = [0, 0]
    cov = [[1, rho_tc], [rho_tc, 1]]
    tc = np.random.multivariate_normal(mean, cov, n)
    t = tc[:, 0]
    c = tc[:, 1]
    y = c + np.random.normal(0, 1, n)
    return t, c, y
```

### Short vs Long Regression

The **short regression** (omitting $C$) regresses $Y$ on $T$ alone:

$$
Y = \alpha + \beta_T^{\text{short}} T + u
$$

The **long regression** (controlling for $C$) includes the confounder:

$$
Y = \alpha + \beta_T^{\text{long}} T + \beta_C C + u
$$

```python
def compute_regressions(t, c, y):
    # Short regression
    slope_short, _, r_short, p_short, _ = stats.linregress(t, y)

    # Long regression via OLS
    X = np.column_stack([np.ones(len(t)), t, c])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    return {
        "short_slope": slope_short,
        "short_p": p_short,
        "long_beta_T": beta[1],
        "long_beta_C": beta[2],
    }
```

When $\rho_{TC}$ is large, the short regression produces a significant $\beta_T^{\text{short}}$ even though $T$ has no causal effect. The long regression correctly estimates $\beta_T^{\text{long}} \approx 0$.

### Partial Regression (Frisch-Waugh-Lovell)

An equivalent approach residualizes both $T$ and $Y$ on $C$, then regresses the residuals:

$$
e_T = T - \hat{\gamma}_1 C, \qquad e_Y = Y - \hat{\gamma}_2 C
$$

$$
\beta_T^{\text{long}} = \frac{\text{Cov}(e_T, e_Y)}{\text{Var}(e_T)}
$$

```python
t_resid = t - stats.linregress(c, t).slope * c
y_resid = y - stats.linregress(c, y).slope * c
slope_partial = stats.linregress(t_resid, y_resid).slope
```

This is the **Frisch-Waugh-Lovell theorem** in action: the coefficient on $T$ in the long regression equals the slope from regressing $e_Y$ on $e_T$.

---

## Part 2: Simpson's Paradox and the Average Treatment Effect

### Setup

Patients have a binary severity indicator (mild = 0, severe = 1). Severe patients are more likely to receive treatment (confounding by indication):

$$
P(\text{treatment} = 1 \mid \text{severe}) = 0.7, \qquad P(\text{treatment} = 1 \mid \text{mild}) = 0.3
$$

The outcome depends on both severity and treatment:

$$
Y = 50 - 20 \cdot \text{severity} + 5 \cdot \text{treatment} + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, 5^2)
$$

The true treatment effect is $+5$, but severe patients have worse outcomes overall.

```python
def simpson_paradox_demo(n=1000):
    severity = np.random.binomial(1, 0.5, n)
    p_treat = np.where(severity == 1, 0.7, 0.3)
    treatment = np.random.binomial(1, p_treat)

    y = (50 - 20 * severity + 5 * treatment
         + np.random.normal(0, 5, n))

    # Naive ATE
    ate_naive = y[treatment == 1].mean() - y[treatment == 0].mean()

    # Adjusted ATE (stratified)
    ate_mild = (y[(treatment == 1) & (severity == 0)].mean()
                - y[(treatment == 0) & (severity == 0)].mean())
    ate_severe = (y[(treatment == 1) & (severity == 1)].mean()
                  - y[(treatment == 0) & (severity == 1)].mean())
    p_severe = severity.mean()
    ate_adjusted = (1 - p_severe) * ate_mild + p_severe * ate_severe

    return ate_naive, ate_mild, ate_severe, ate_adjusted
```

### The Paradox

The **naive ATE** compares treated and untreated groups without adjusting for severity. Because severe patients (who have worse outcomes) are overrepresented in the treatment group, the naive ATE can be **negative** -- making it appear that the treatment is harmful.

The **adjusted ATE** stratifies by severity and computes a weighted average:

$$
\text{ATE}_{\text{adj}} = P(\text{mild}) \cdot \text{ATE}_{\text{mild}} + P(\text{severe}) \cdot \text{ATE}_{\text{severe}}
$$

Both subgroup ATEs are close to the true effect of $+5$, and their weighted average correctly recovers it.

---

## Interpretation

These demonstrations illustrate two central themes of causal inference:

1. **Omitted variable bias.** Failing to control for a confounder produces a biased estimate of the causal effect. The short regression attributes to $T$ the variation that actually belongs to $C$. The bias equals $\beta_T^{\text{short}} - \beta_T^{\text{long}} = \hat{\delta} \cdot \hat{\gamma}$, where $\hat{\delta}$ is the coefficient of $C$ in the long regression and $\hat{\gamma}$ is the coefficient from regressing $T$ on $C$.

2. **Simpson's paradox.** Aggregating heterogeneous subgroups can reverse the direction of an effect. The naive ATE is confounded by severity; the stratified ATE removes this confounding. In observational studies, randomization is absent, so explicit adjustment for confounders is essential.

---

## Exercises

**Exercise 1.**
In the confounded regression, derive the omitted variable bias formula. Show that:

$$
\beta_T^{\text{short}} = \beta_T^{\text{long}} + \beta_C^{\text{long}} \cdot \hat{\gamma}
$$

where $\hat{\gamma}$ is the coefficient from regressing $C$ on $T$.

??? success "Solution to Exercise 1"

    The short regression estimates:

    $$
    \beta_T^{\text{short}} = \frac{\text{Cov}(T, Y)}{\text{Var}(T)}
    $$

    Since $Y = \alpha + \beta_T^{\text{long}} T + \beta_C^{\text{long}} C + u$:

    $$
    \text{Cov}(T, Y) = \beta_T^{\text{long}} \text{Var}(T) + \beta_C^{\text{long}} \text{Cov}(T, C)
    $$

    Dividing by $\text{Var}(T)$:

    $$
    \beta_T^{\text{short}} = \beta_T^{\text{long}} + \beta_C^{\text{long}} \cdot \frac{\text{Cov}(T, C)}{\text{Var}(T)} = \beta_T^{\text{long}} + \beta_C^{\text{long}} \cdot \hat{\gamma}
    $$

    where $\hat{\gamma} = \text{Cov}(T, C) / \text{Var}(T)$ is the regression coefficient of $C$ on $T$. In our simulation $\beta_T^{\text{long}} \approx 0$ and $\beta_C^{\text{long}} \approx 1$, so $\beta_T^{\text{short}} \approx \hat{\gamma} = \rho_{TC}$. The entire short regression slope is bias. $\square$

---

**Exercise 2.**
Simulate the confounded regression with $\rho_{TC} \in \{-0.9, -0.5, 0, 0.5, 0.9\}$ and $n = 500$. For each value, report $\beta_T^{\text{short}}$ and $\beta_T^{\text{long}}$. Plot $\beta_T^{\text{short}}$ as a function of $\rho_{TC}$ and confirm it is approximately linear.

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from scipy import stats

    rhos = [-0.9, -0.5, 0, 0.5, 0.9]
    short_slopes = []

    for rho in rhos:
        np.random.seed(42)
        n = 500
        tc = np.random.multivariate_normal([0, 0],
                                           [[1, rho], [rho, 1]], n)
        t, c = tc[:, 0], tc[:, 1]
        y = c + np.random.normal(0, 1, n)

        short_slope = stats.linregress(t, y).slope
        X = np.column_stack([np.ones(n), t, c])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

        short_slopes.append(short_slope)
        print(f"rho={rho:+.1f}: short={short_slope:.3f}, "
              f"long_T={beta[1]:.3f}")

    import matplotlib.pyplot as plt
    plt.plot(rhos, short_slopes, 'o-')
    plt.xlabel('rho(T, C)')
    plt.ylabel('Short regression slope')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title('Omitted Variable Bias')
    plt.show()
    ```

    The plot shows an approximately linear relationship: $\beta_T^{\text{short}} \approx \rho_{TC}$, confirming the omitted variable bias formula. The long regression slope stays near zero regardless of $\rho_{TC}$. $\square$

---

**Exercise 3.**
In the Simpson's paradox demo, what happens if treatment assignment is independent of severity (i.e., $P(\text{treatment}) = 0.5$ for everyone)? Simulate this and show that the naive ATE now correctly estimates the true effect.

??? success "Solution to Exercise 3"

    ```python
    import numpy as np

    np.random.seed(42)
    n = 1000
    severity = np.random.binomial(1, 0.5, n)
    treatment = np.random.binomial(1, 0.5, n)  # independent of severity

    y = 50 - 20 * severity + 5 * treatment + np.random.normal(0, 5, n)

    ate_naive = y[treatment == 1].mean() - y[treatment == 0].mean()
    print(f"Naive ATE = {ate_naive:.2f} (true = 5)")
    ```

    When treatment is randomized (independent of the confounder), the naive ATE is an unbiased estimator of the true ATE. The treated and control groups have the same distribution of severity, so there is no confounding. This is exactly why randomized controlled trials are the gold standard for causal inference. $\square$

---

**Exercise 4.**
Extend the Simpson's paradox example to three severity levels (mild, moderate, severe) with treatment probabilities 0.2, 0.5, and 0.8 respectively. Show that the paradox still occurs and compute the stratified ATE.

??? success "Solution to Exercise 4"

    ```python
    import numpy as np

    np.random.seed(42)
    n = 1500
    severity = np.random.choice([0, 1, 2], size=n, p=[1/3, 1/3, 1/3])
    p_treat = np.where(severity == 0, 0.2,
                       np.where(severity == 1, 0.5, 0.8))
    treatment = np.random.binomial(1, p_treat)

    y = 50 - 15 * severity + 5 * treatment + np.random.normal(0, 5, n)

    ate_naive = y[treatment == 1].mean() - y[treatment == 0].mean()

    ates = []
    for s in [0, 1, 2]:
        mask_t = (treatment == 1) & (severity == s)
        mask_c = (treatment == 0) & (severity == s)
        ate_s = y[mask_t].mean() - y[mask_c].mean()
        ates.append(ate_s)
        print(f"ATE (severity={s}): {ate_s:.2f}")

    p_sev = [np.mean(severity == s) for s in [0, 1, 2]]
    ate_adjusted = sum(p * a for p, a in zip(p_sev, ates))

    print(f"Naive ATE:    {ate_naive:.2f}")
    print(f"Adjusted ATE: {ate_adjusted:.2f}")
    print(f"True effect:  5.00")
    ```

    The naive ATE is biased negative because the most severely ill patients (who have worse outcomes) disproportionately receive treatment. Each subgroup ATE is close to 5, and the weighted average correctly recovers the true effect. The paradox generalizes to any number of confounding strata. $\square$

---

**Exercise 5.**
Prove that if $T$ is randomized (independent of all confounders), then the naive difference in means $\mathbb{E}[Y \mid T = 1] - \mathbb{E}[Y \mid T = 0]$ equals the average treatment effect $\mathbb{E}[Y(1) - Y(0)]$ under the potential outcomes framework.

??? success "Solution to Exercise 5"

    Let $Y(1)$ and $Y(0)$ denote the potential outcomes under treatment and control. The observed outcome is $Y = T \cdot Y(1) + (1 - T) \cdot Y(0)$.

    The ATE is defined as:

    $$
    \tau = \mathbb{E}[Y(1) - Y(0)]
    $$

    If $T$ is independent of $(Y(0), Y(1))$ (randomization), then:

    $$
    \mathbb{E}[Y \mid T = 1] = \mathbb{E}[Y(1) \mid T = 1] = \mathbb{E}[Y(1)]
    $$

    where the last equality uses independence. Similarly:

    $$
    \mathbb{E}[Y \mid T = 0] = \mathbb{E}[Y(0) \mid T = 0] = \mathbb{E}[Y(0)]
    $$

    Therefore:

    $$
    \mathbb{E}[Y \mid T = 1] - \mathbb{E}[Y \mid T = 0] = \mathbb{E}[Y(1)] - \mathbb{E}[Y(0)] = \tau
    $$

    Randomization ensures that the treatment and control groups are comparable in expectation, eliminating selection bias. Without randomization, $\mathbb{E}[Y(1) \mid T = 1] \ne \mathbb{E}[Y(1)]$ in general, and the naive difference is biased. $\square$
