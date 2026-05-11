# Weibull Density Function and Hazard

## Overview

The **Weibull distribution** is a flexible model for lifetimes, failure times, and survival data. Its shape parameter $k$ controls whether the hazard rate is increasing, decreasing, or constant:

$$
f(x; k, \lambda) = \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1} \exp\!\left(-\left(\frac{x}{\lambda}\right)^k\right), \qquad x \ge 0
$$

| $k$ | Hazard behavior | Special case |
|---|---|---|
| $k < 1$ | Decreasing hazard (infant mortality) | — |
| $k = 1$ | Constant hazard | Exponential($\lambda$) |
| $k > 1$ | Increasing hazard (aging/wear-out) | — |
| $k \approx 3.6$ | Approximately normal shape | — |

---

## Survival and Hazard Functions

$$
S(x) = \exp\!\left(-\left(\frac{x}{\lambda}\right)^k\right), \qquad h(x) = \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1}
$$

The hazard function $h(x) = f(x)/S(x)$ gives the instantaneous failure rate at time $x$, conditional on survival to $x$.

---

## Code

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

x = np.linspace(0.01, 3.0, 500)
lam = 1.0

params = [
    (0.5, "k=0.5 (decreasing hazard)"),
    (1.0, "k=1.0 (exponential)"),
    (1.5, "k=1.5 (increasing hazard)"),
    (3.0, "k=3.0 (near-normal shape)"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for k, desc in params:
    pdf = (k / lam) * (x / lam) ** (k - 1) * np.exp(-(x / lam) ** k)
    sf = np.exp(-(x / lam) ** k)
    haz = (k / lam) * (x / lam) ** (k - 1)

    axes[0].plot(x, pdf, lw=2, label=f'k = {k}')
    axes[1].plot(x, sf, lw=2, label=f'k = {k}')
    axes[2].plot(x, haz, lw=2, label=f'k = {k}')

axes[0].set_title('Weibull PDF')
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[1].set_title('Survival Function S(x)')
axes[1].set_xlabel('x')
axes[2].set_title('Hazard Function h(x)')
axes[2].set_xlabel('x')
axes[2].set_ylim(0, 5)

for ax in axes:
    ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
```

---

## Exercises

**Exercise 1.**
Show that the Weibull distribution with $k = 1$ reduces to the exponential distribution with rate $1/\lambda$.

??? success "Solution to Exercise 1"
    Setting $k = 1$:

    $$
    f(x) = \frac{1}{\lambda}\exp\!\left(-\frac{x}{\lambda}\right)
    $$

    This is the PDF of $\text{Exponential}(\text{rate} = 1/\lambda)$ with mean $\lambda$. The hazard becomes $h(x) = 1/\lambda$, a constant — consistent with the memoryless property. $\square$

---

**Exercise 2.**
Derive the CDF and median of the Weibull distribution.

??? success "Solution to Exercise 2"
    **CDF:**

    $$
    F(x) = 1 - S(x) = 1 - \exp\!\left(-\left(\frac{x}{\lambda}\right)^k\right)
    $$

    **Median:** Solve $F(m) = 0.5$:

    $$
    \exp\!\left(-\left(\frac{m}{\lambda}\right)^k\right) = 0.5 \implies \left(\frac{m}{\lambda}\right)^k = \ln 2 \implies m = \lambda(\ln 2)^{1/k}
    $$

---

**Exercise 3.**
A component has a Weibull lifetime with $k = 2$ and $\lambda = 1000$ hours. What is the probability it survives past 500 hours? Past 1500 hours?

??? success "Solution to Exercise 3"
    $$
    S(500) = \exp\!\left(-\left(\frac{500}{1000}\right)^2\right) = e^{-0.25} \approx 0.779
    $$

    $$
    S(1500) = \exp\!\left(-\left(\frac{1500}{1000}\right)^2\right) = e^{-2.25} \approx 0.105
    $$

    The component has a 77.9% chance of surviving 500 hours but only 10.5% chance of surviving 1500 hours. The increasing hazard ($k = 2 > 1$) means failure becomes more likely as the component ages.

---

**Exercise 4.**
Explain why the Weibull distribution is widely used in reliability engineering despite the availability of more flexible distributions.

??? success "Solution to Exercise 4"
    The Weibull's popularity comes from several practical advantages:

    1. **Interpretable hazard:** The single parameter $k$ directly controls whether failure rate increases, decreases, or stays constant.
    2. **Closed-form functions:** The PDF, CDF, survival function, and hazard all have simple closed-form expressions — no numerical integration needed.
    3. **Linearizable:** Taking $\ln(-\ln(S(x)))$ vs $\ln(x)$ gives a straight line, enabling graphical parameter estimation (Weibull probability plots).
    4. **Nests the exponential:** Setting $k=1$ recovers the simplest lifetime model, making the Weibull a natural generalization.
