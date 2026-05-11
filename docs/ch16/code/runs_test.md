# Runs Test

## Overview

The **Wald--Wolfowitz runs test** is a nonparametric procedure for deciding whether a sequence
of binary observations was generated at random. It counts the number of *runs*---maximal
consecutive subsequences of identical symbols---and compares that count to the distribution
expected under the null hypothesis of independence. Too few runs suggest clustering; too many
suggest systematic alternation.

## The Runs Statistic

Suppose a binary sequence of length $N$ contains $N_+$ symbols of one type and
$N_- = N - N_+$ of the other. A **run** is any maximal block of consecutive identical
symbols.

Under the null hypothesis $H_0$ that the sequence is independently and identically
distributed, the number of runs $R$ has mean and standard deviation

$$
\mu_R = \frac{2\,N_+\,N_-}{N} + 1, \qquad
\sigma_R = \sqrt{\frac{(\mu_R - 1)(\mu_R - 2)}{N - 1}}.
$$

For moderate-to-large $N$ the standardized statistic

$$
Z = \frac{R - \mu_R}{\sigma_R}
$$

is approximately standard normal, giving a two-sided $p$-value

$$
p = 2\,\mathcal{N}\!\bigl(-|Z|\bigr),
$$

where $\mathcal{N}$ is the standard normal CDF.

## Counting Runs Efficiently

Given a sequence $x_1, x_2, \dots, x_N$ coded as $\pm 1$, the product $x_i\,x_{i+1}$
equals $+1$ when consecutive elements agree and $-1$ when they differ. Each sign change
starts a new run, so

$$
R = \frac{N_+ + N_- + 1 - \displaystyle\sum_{i=1}^{N-1} x_i\,x_{i+1}}{2}.
$$

## Implementation

The following Python function implements the runs test with the normal approximation.

```python
import numpy as np
import scipy.stats as stats


def runs_test(data):
    """Wald-Wolfowitz runs test for a +1/-1 sequence."""
    data = np.asarray(data)
    N = data.shape[0]
    N_plus = (data == 1).sum()
    N_minus = N - N_plus

    mu = 2 * N_plus * N_minus / N + 1
    sigma = np.sqrt((mu - 1) * (mu - 2) / (N - 1))
    R = (N_plus + N_minus + 1 - np.sum(data[1:] * data[:-1])) / 2

    statistic = (R - mu) / sigma
    p_value = 2 * stats.norm().cdf(-abs(statistic))
    return statistic, p_value
```

### Example: Clustered Sequence

A sequence with heavy clustering has very few runs.

```python
data = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
z, p = runs_test(data * 2 - 1)
# z ≈ -2.97, p ≈ 0.003 → reject randomness
```

### Example: Well-Mixed Sequence

A sequence with frequent alternation looks compatible with randomness.

```python
data = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0])
z, p = runs_test(data * 2 - 1)
# z ≈ 0.29, p ≈ 0.77 → fail to reject
```

## Interpretation

| Outcome | Meaning |
|---|---|
| $Z \ll 0$ (few runs) | Observations are **clustered**---successive values tend to be the same. |
| $Z \gg 0$ (many runs) | Observations **alternate** more than chance predicts. |
| $\lvert Z \rvert$ small | No evidence against randomness at the chosen significance level. |

The test is two-sided by default: both unusually few and unusually many runs are evidence
against independence.

## Exercises

**Exercise 1.** A coin is flipped 20 times, yielding the sequence
HHHHTTTTHHHHTTTTTTHH. Code each H as $+1$ and each T as $-1$, count the number of runs
$R$, and compute $\mu_R$ and $\sigma_R$ by hand.

??? success "Solution to Exercise 1"

    The sequence is HHHH TTTT HHHH TTTTTT HH, giving $R = 5$ runs.
    We have $N = 20$, $N_+ = 10$, $N_- = 10$.

    $$
    \mu_R = \frac{2 \cdot 10 \cdot 10}{20} + 1 = 11.
    $$

    $$
    \sigma_R = \sqrt{\frac{(11 - 1)(11 - 2)}{20 - 1}}
             = \sqrt{\frac{90}{19}}
             \approx 2.176.
    $$

    Therefore $Z = (5 - 11)/2.176 \approx -2.76$, giving strong evidence of clustering.
    $\square$

---

**Exercise 2.** Prove that $R = \dfrac{N_+ + N_- + 1 - \sum_{i=1}^{N-1} x_i\,x_{i+1}}{2}$
when each $x_i \in \{-1, +1\}$.

??? success "Solution to Exercise 2"

    Define the indicator $d_i = \mathbf{1}[x_i \neq x_{i+1}]$ for $i = 1,\dots,N-1$.
    Each sign change starts a new run, so $R = 1 + \sum_{i=1}^{N-1} d_i$.

    Since $x_i \in \{-1,+1\}$, we have $x_i\,x_{i+1} = 1 - 2\,d_i$, hence
    $d_i = (1 - x_i\,x_{i+1})/2$. Summing:

    $$
    R = 1 + \sum_{i=1}^{N-1} \frac{1 - x_i\,x_{i+1}}{2}
      = 1 + \frac{(N-1) - \sum_{i=1}^{N-1} x_i\,x_{i+1}}{2}
      = \frac{N + 1 - \sum_{i=1}^{N-1} x_i\,x_{i+1}}{2}.
    $$

    Since $N = N_+ + N_-$, the result follows. $\square$

---

**Exercise 3.** Using the data in the "Well-Mixed Sequence" example above, compute the
runs test statistic and $p$-value in Python. Verify that the null hypothesis is not rejected
at $\alpha = 0.05$.

??? success "Solution to Exercise 3"

    ```python
    import numpy as np
    import scipy.stats as stats

    data = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0])
    seq = data * 2 - 1  # convert to +1/-1

    N = len(seq)
    N_plus = (seq == 1).sum()   # 9
    N_minus = N - N_plus        # 8

    mu = 2 * N_plus * N_minus / N + 1   # ≈ 9.47
    sigma = np.sqrt((mu - 1) * (mu - 2) / (N - 1))  # ≈ 1.93
    R = (N - np.sum(seq[1:] * seq[:-1]) + 1) / 2     # 10

    z = (R - mu) / sigma
    p = 2 * stats.norm.cdf(-abs(z))
    print(f"Z = {z:.4f}, p = {p:.4f}")
    # p > 0.05 → fail to reject H₀
    ```

    The $p$-value is well above $0.05$, so there is no evidence against randomness. $\square$

---

**Exercise 4.** Explain why the runs test is inappropriate if the sequence is not binary.
Describe one common approach to convert a continuous sequence into a binary one suitable for
the test.

??? success "Solution to Exercise 4"

    The derivation of $\mu_R$ and $\sigma_R$ assumes exactly two symbol types with
    fixed counts $N_+$ and $N_-$. With more than two categories the combinatorics
    change and the normal approximation no longer applies.

    A standard remedy is to **dichotomize** the continuous sequence about its sample
    median: values above the median are coded $+1$ and values below are coded $-1$
    (values equal to the median are handled by convention, e.g., dropped or assigned
    to one group). The resulting binary sequence can then be tested for randomness
    using the Wald--Wolfowitz procedure. $\square$

---

**Exercise 5.** Show that $\operatorname{E}[R] = \mu_R = \dfrac{2\,N_+\,N_-}{N} + 1$ by a
counting argument over all permutations of the sequence.

??? success "Solution to Exercise 5"

    Under $H_0$ all $\binom{N}{N_+}$ arrangements of the sequence are equally likely.
    Write $R = 1 + \sum_{i=1}^{N-1} d_i$ where $d_i = \mathbf{1}[x_i \neq x_{i+1}]$.
    By linearity of expectation,

    $$
    \operatorname{E}[R] = 1 + \sum_{i=1}^{N-1} P(x_i \neq x_{i+1}).
    $$

    For any adjacent pair under a uniform random permutation,

    $$
    P(x_i \neq x_{i+1})
      = \frac{N_+\,N_-}{N(N-1)/2 \cdot 2/(N-1)}
      = \frac{2\,N_+\,N_-}{N(N-1)}.
    $$

    More directly, $P(x_i = +1, x_{i+1} = -1) = N_+\,N_- / [N(N-1)]$ and similarly for
    the reverse order, so $P(x_i \neq x_{i+1}) = 2\,N_+\,N_- / [N(N-1)]$.

    Summing over $N-1$ adjacent pairs:

    $$
    \operatorname{E}[R] = 1 + (N-1) \cdot \frac{2\,N_+\,N_-}{N(N-1)}
                        = 1 + \frac{2\,N_+\,N_-}{N}
                        = \mu_R.
    $$

    $\square$
