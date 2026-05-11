# Capture-Recapture Maximum Likelihood

## Overview

The **capture-recapture method** is a classic technique for estimating the size of a population that cannot be directly counted. By capturing, tagging, and releasing a subset of individuals, then recapturing another subset and counting how many are tagged, we can derive a maximum likelihood estimate of the total population size. This page develops the hypergeometric likelihood and the MLE for the capture-recapture model.

## The Capture-Recapture Setup

The method proceeds in two stages:

1. **Capture phase:** Capture $c$ individuals from a population of unknown size $N$, tag them, and release them.
2. **Recapture phase:** Capture $r$ individuals. Of these, $t$ are found to be tagged.

The key question: what is $N$?

!!! info "Assumptions"

    - The population is closed (no births, deaths, immigration, or emigration between phases).
    - Every individual has an equal probability of being captured.
    - Tags are not lost and are correctly identified.
    - Capture in the second phase is independent of capture in the first phase.

## The Hypergeometric Model

Given $N$ total individuals with $c$ tagged, the number of tagged individuals $T$ in a recapture sample of size $r$ follows a **hypergeometric distribution**:

$$
P(T = t \mid N) = \frac{\binom{c}{t}\binom{N - c}{r - t}}{\binom{N}{r}}
$$

for $\max(0, r + c - N) \leq t \leq \min(r, c)$.

The parameter of interest is $N$, and the likelihood function is $L(N) = P(T = t \mid N)$ viewed as a function of $N$ for fixed data $(c, r, t)$.

## The Maximum Likelihood Estimator

The MLE of $N$ is the value that maximizes $L(N)$. Since $N$ is a discrete parameter (a positive integer), we search over integers $N \geq c + r - t$.

The MLE has a well-known closed form:

$$
\hat{N}_{\text{MLE}} = \left\lfloor \frac{cr}{t} \right\rfloor
$$

This is the **Lincoln-Petersen estimate** (rounded down to the nearest integer).

### Intuition

The MLE arises from the proportionality argument: if the recapture is representative, then the proportion of tagged individuals in the recapture should approximate the proportion in the population:

$$
\frac{t}{r} \approx \frac{c}{N} \quad \Rightarrow \quad N \approx \frac{cr}{t}
$$

## Implementation

```python
from scipy import special

def prob(n, c, r, t):
    """
    Hypergeometric probability: P(T = t | N = n).
    
    Parameters
    ----------
    n : Total population size
    c : Number tagged in capture phase
    r : Number in recapture sample
    t : Number of tagged in recapture
    """
    return special.comb(n - c, r - t) * special.comb(c, t) / special.comb(n, r)


def capture_recapture_mle(c, r, t):
    """
    Compute the MLE of population size N via exhaustive search.
    """
    n_min = c + r - t  # minimum possible N
    n_max = 10 * n_min  # search range

    prob_list = [prob(n, c, r, t) for n in range(n_min, n_max)]
    mle_idx = max(range(len(prob_list)), key=lambda i: prob_list[i])
    mle_n = mle_idx + n_min

    return mle_n, prob_list


# Example: c=10 tagged, r=10 recaptured, t=3 tagged in recapture
c, r, t = 10, 10, 3
mle_n, probs = capture_recapture_mle(c, r, t)
print(f"Capture: {c} tagged, Recapture: {r} caught, {t} tagged")
print(f"MLE of N: {mle_n}")
print(f"Lincoln-Petersen estimate: {c * r // t}")
```

## A Worked Example

Suppose a wildlife biologist captures and tags $c = 5$ birds, releases them, and later recaptures $r = 6$ birds, of which $t = 2$ are tagged.

```python
c, r, t = 5, 6, 2
mle_n, probs = capture_recapture_mle(c, r, t)
print(f"MLE of N: {mle_n}")
print(f"Lincoln-Petersen: {c * r // t}")
```

The Lincoln-Petersen estimate gives $\hat{N} = \lfloor 5 \times 6 / 2 \rfloor = 15$.

The likelihood function shows a clear peak at $N = 15$, with the probability declining for both smaller and larger values of $N$.

## Properties of the Estimator

!!! note "Bias of the Lincoln-Petersen Estimator"
    The basic Lincoln-Petersen estimator $cr/t$ is biased, tending to overestimate $N$ especially when $t$ is small. Chapman's corrected estimator reduces this bias:

    $$
    \hat{N}_{\text{Chapman}} = \frac{(c+1)(r+1)}{t+1} - 1
    $$

The likelihood function $L(N)$ for this problem is **unimodal** (has a single peak), which ensures the MLE is unique and grid search is reliable.

## Sensitivity Analysis

The quality of the estimate depends heavily on the number of recaptured tagged individuals $t$:

- When $t$ is large (relative to $r$ and $c$), the estimate is precise.
- When $t$ is small (e.g., $t = 1$), the estimate is unreliable and the likelihood function is flat.
- When $t = 0$, the MLE is undefined (the population could be arbitrarily large).

```python
from scipy import special

def sensitivity_analysis():
    """Show how MLE changes with different values of t."""
    c, r = 10, 10
    print(f"c = {c}, r = {r}")
    print(f"{'t':>4} {'MLE':>6} {'cr/t':>8}")
    print("-" * 20)
    for t in range(1, min(c, r) + 1):
        n_min = c + r - t
        n_max = 10 * n_min
        probs = [special.comb(n - c, r - t) * special.comb(c, t) / special.comb(n, r)
                 for n in range(n_min, n_max)]
        mle_idx = max(range(len(probs)), key=lambda i: probs[i])
        mle_n = mle_idx + n_min
        print(f"{t:>4} {mle_n:>6} {c*r/t:>8.1f}")

sensitivity_analysis()
```

## Interpretation

- The capture-recapture MLE provides a principled way to estimate population size from mark-and-recapture data.
- The method relies on the hypergeometric distribution, which models sampling without replacement from a finite population.
- The Lincoln-Petersen formula $\hat{N} = cr/t$ has an elegant proportionality interpretation but can be biased for small $t$.
- Real ecological applications must account for violations of the closed-population and equal-catchability assumptions.

## Exercises

**Exercise 1.** A marine biologist tags $c = 20$ fish and releases them. In a later sample of $r = 25$ fish, $t = 5$ are tagged. Compute the MLE of the total population size using both the grid search method and the Lincoln-Petersen formula.

??? success "Solution to Exercise 1"
    Lincoln-Petersen: $\hat{N} = \lfloor cr/t \rfloor = \lfloor 20 \times 25/5 \rfloor = 100$.

    Grid search:
    ```python
    from scipy import special
    c, r, t = 20, 25, 5
    n_min = c + r - t  # = 40
    probs = [special.comb(n - c, r - t) * special.comb(c, t) / special.comb(n, r)
             for n in range(n_min, 500)]
    mle_idx = max(range(len(probs)), key=lambda i: probs[i])
    print(f"MLE: N = {mle_idx + n_min}")
    ```

    Both methods give $\hat{N} = 100$. $\square$

---

**Exercise 2.** Show that the Lincoln-Petersen estimator $\hat{N} = cr/t$ is the value that maximizes the hypergeometric likelihood when we treat $N$ as continuous. (Hint: show that $L(N)/L(N-1) > 1$ if and only if $N < cr/t$.)

??? success "Solution to Exercise 2"
    The likelihood ratio is:

    $$
    \frac{L(N)}{L(N-1)} = \frac{\binom{N-c}{r-t}}{\binom{N-1-c}{r-t}} \cdot \frac{\binom{N-1}{r}}{\binom{N}{r}}
    $$

    Using $\binom{n}{k}/\binom{n-1}{k} = n/(n-k)$:

    $$
    \frac{L(N)}{L(N-1)} = \frac{N - c}{N - c - (r-t)} \cdot \frac{N - r}{N} = \frac{(N-c)(N-r)}{N(N-c-r+t)}
    $$

    This ratio exceeds 1 when $(N-c)(N-r) > N(N-c-r+t)$, i.e., $N^2 - (c+r)N + cr > N^2 - (c+r-t)N$, which simplifies to $cr > tN$, or $N < cr/t$.

    So $L(N)$ is increasing for $N < cr/t$ and decreasing for $N > cr/t$, confirming the maximum is at $N = \lfloor cr/t \rfloor$ (since $N$ must be an integer). $\square$

---

**Exercise 3.** Chapman's corrected estimator is $\hat{N}_C = (c+1)(r+1)/(t+1) - 1$. Compute $\hat{N}_C$ for $c = 10, r = 10, t = 3$ and compare to the MLE. Why is the correction useful?

??? success "Solution to Exercise 3"
    Chapman's estimate: $\hat{N}_C = (11)(11)/4 - 1 = 121/4 - 1 = 30.25 - 1 = 29.25$.

    The MLE (Lincoln-Petersen) gives $\hat{N} = \lfloor 100/3 \rfloor = 33$.

    Chapman's estimator is lower because it corrects for the positive bias of the Lincoln-Petersen estimator. The bias arises because $E[cr/T] > cr/E[T]$ by Jensen's inequality (since $1/T$ is convex). Chapman's correction approximately removes this bias, making it especially useful when $t$ is small relative to $r$ and $c$. $\square$

---

**Exercise 4.** If no tagged individuals are found in the recapture ($t = 0$), explain why the MLE does not exist. What does this imply for the design of capture-recapture studies?

??? success "Solution to Exercise 4"
    When $t = 0$, the likelihood function is:

    $$
    L(N) = \frac{\binom{N-c}{r}}{\binom{N}{r}}
    $$

    This is a decreasing function of... actually, for $t = 0$, $L(N)$ is increasing in $N$: larger populations make it more likely that none of the recaptured individuals are tagged. As $N \to \infty$, $L(N) \to 1$. There is no finite maximizer, so the MLE does not exist.

    **Design implication:** The study must be designed so that $t > 0$ is likely. This requires:

    - Tagging a sufficiently large number $c$.
    - Recapturing a sufficiently large number $r$.
    - The product $cr/N$ should be large enough that $P(T > 0)$ is high. As a rule of thumb, $cr \gg N$ or at least $cr/N > 5$ to ensure a reasonable probability of recapturing tagged individuals. $\square$

---

**Exercise 5.** Derive the variance of the Lincoln-Petersen estimator $\hat{N} = cr/T$ using the delta method. The variance of the hypergeometric is $\text{Var}(T) = r \cdot \frac{c}{N} \cdot \frac{N-c}{N} \cdot \frac{N-r}{N-1}$.

??? success "Solution to Exercise 5"
    Let $g(T) = cr/T$ so $\hat{N} = g(T)$. By the delta method:

    $$
    \text{Var}(\hat{N}) \approx [g'(E[T])]^2 \, \text{Var}(T)
    $$

    We have $g'(T) = -cr/T^2$ and $E[T] = rc/N$. So:

    $$
    g'(E[T]) = \frac{-cr}{(rc/N)^2} = \frac{-N^2}{cr}
    $$

    Substituting:

    $$
    \text{Var}(\hat{N}) \approx \frac{N^4}{c^2 r^2} \cdot r \cdot \frac{c}{N} \cdot \frac{N-c}{N} \cdot \frac{N-r}{N-1}
    $$

    $$
    = \frac{N^2(N-c)(N-r)}{cr(N-1)}
    $$

    This shows the variance decreases as $c$ and $r$ increase, and increases with $N$. For large populations with small capture fractions, the variance can be very large, underscoring the need for substantial capture effort. $\square$
