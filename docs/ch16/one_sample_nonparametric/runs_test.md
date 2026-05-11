# Runs Test for Randomness

Before testing whether a sample comes from a particular distribution or whether two groups differ, it is often important to verify a more basic property: **randomness**. If the observations in a sequence exhibit systematic patterns -- clusters of similar values or excessive alternation -- then the independence assumption underlying most statistical tests is violated. The **Wald-Wolfowitz runs test** detects such departures from randomness by counting the number of "runs" in a binary-coded sequence.

## Definition of a Run

A **run** is a maximal consecutive subsequence of identical elements. Given a sequence of two types of symbols (e.g., $+$ and $-$), each contiguous block of the same symbol constitutes one run. For example, in the sequence

$$
\underbrace{+ + +}_{\text{run 1}} \; \underbrace{- -}_{\text{run 2}} \; \underbrace{+}_{\text{run 3}} \; \underbrace{- - -}_{\text{run 4}}
$$

there are $R = 4$ runs. A truly random sequence should have neither too few runs (which indicates clustering) nor too many (which indicates systematic alternation).

## Hypotheses

$$
H_0 \colon \text{The elements of the sequence are mutually independent (random order)}
$$

$$
H_a \colon \text{The sequence is not random}
$$

The test is typically two-sided: both clustering (too few runs) and alternation (too many runs) are considered departures from randomness.

## Constructing the Binary Sequence

In practice, the input data are usually continuous observations rather than binary symbols. The standard approach is to compare each observation to the sample median $\tilde{x}$ and encode values above the median as $+$ and values below as $-$. Observations exactly equal to the median are typically dropped from the sequence.

## Test Statistic

Let the sequence have length $N$, with $n_1$ elements of one type and $n_2 = N - n_1$ of the other. The number of runs $R$ is the test statistic.

**Expected number of runs under $H_0$:**

$$
\mu_R = \frac{2 \, n_1 \, n_2}{N} + 1
$$

**Variance of the number of runs under $H_0$:**

$$
\sigma_R^2 = \frac{2 \, n_1 \, n_2 \, (2 \, n_1 \, n_2 - N)}{N^2 (N - 1)}
$$

This can equivalently be written as

$$
\sigma_R^2 = \frac{(\mu_R - 1)(\mu_R - 2)}{N - 1}
$$

## Normal Approximation

For large $N$ (typically $n_1 \ge 10$ and $n_2 \ge 10$), the standardized statistic

$$
Z = \frac{R - \mu_R}{\sigma_R}
$$

is approximately standard normal under $H_0$. The two-sided $p$-value is

$$
p = 2 \, \mathcal{N}(-|Z|)
$$

where $\mathcal{N}$ is the standard normal CDF.

!!! note "Exact distribution for small samples"
    For small $N$, exact critical values can be obtained from tables of the runs distribution. The exact null distribution is computed by enumerating all $\binom{N}{n_1}$ equally likely arrangements and counting the number of runs in each.

## Worked Example

Consider the following sequence of 15 stock returns classified as positive ($+$) or negative ($-$):

$$
+, +, +, -, -, +, +, -, +, -, -, -, +, +, -
$$

**Step 1.** Count the elements: $n_1 = 8$ (positive), $n_2 = 7$ (negative), $N = 15$.

**Step 2.** Count the runs: $+\!+\!+$, $-\!-$, $+\!+$, $-$, $+$, $-\!-\!-$, $+\!+$, $-$, giving $R = 8$.

**Step 3.** Compute the null mean and standard deviation:

$$
\mu_R = \frac{2(8)(7)}{15} + 1 = \frac{112}{15} + 1 \approx 8.467
$$

$$
\sigma_R^2 = \frac{(8.467 - 1)(8.467 - 2)}{15 - 1} = \frac{(7.467)(6.467)}{14} \approx 3.448
$$

$$
\sigma_R \approx 1.857
$$

**Step 4.** Compute the $Z$-statistic:

$$
Z = \frac{8 - 8.467}{1.857} \approx -0.251
$$

**Step 5.** Compute the $p$-value: $p = 2\,\mathcal{N}(-0.251) \approx 0.802$.

Since $p = 0.802 \gg 0.05$, we fail to reject $H_0$. The data are consistent with a random sequence.

??? example "Detecting clustering"
    If the same 15 observations were arranged as $+,+,+,+,+,+,+,+,-,-,-,-,-,-,-$, there would be only $R = 2$ runs. With $\mu_R \approx 8.467$ and $\sigma_R \approx 1.857$, the $Z$-statistic would be $(2 - 8.467)/1.857 \approx -3.48$, giving $p \approx 0.0005$. This provides strong evidence that the sequence is not random (the values are clustered).

## Interpretation

| Outcome | Meaning |
|:--------|:--------|
| Too few runs ($R \ll \mu_R$) | Observations are clustered -- positive autocorrelation |
| Too many runs ($R \gg \mu_R$) | Observations alternate excessively -- negative autocorrelation |
| $R \approx \mu_R$ | No evidence against randomness |

## Applications

- **Random walk hypothesis.** Testing whether successive stock returns are independent by coding each return as above or below zero (or above/below the median).
- **Quality control.** Checking whether defects on a production line occur randomly or in clusters.
- **Residual diagnostics.** After fitting a regression model, applying the runs test to the sequence of positive and negative residuals checks the independence assumption.

## Summary

The Wald-Wolfowitz runs test provides a simple, distribution-free method for detecting departures from randomness in a sequence. By counting the number of maximal consecutive blocks of identical elements and comparing to the expected count under independence, the test identifies both clustering and alternation patterns. The normal approximation is reliable for sequences with at least 10 elements of each type; for smaller samples, exact tables should be consulted.

## Exercises

**Exercise 1.**
A coin is flipped 20 times, producing the sequence:

$$
H, H, T, T, T, H, H, H, H, T, T, H, T, H, H, T, T, T, H, H
$$

**(a)** Count the number of runs $R$ in this sequence.

**(b)** Let $n_H = 11$ and $n_T = 9$. Under the null hypothesis of randomness, compute the expected number of runs:

$$
E[R] = \frac{2 n_H n_T}{n_H + n_T} + 1
$$

and the variance:

$$
\text{Var}(R) = \frac{2 n_H n_T (2 n_H n_T - n_H - n_T)}{(n_H + n_T)^2 (n_H + n_T - 1)}
$$

**(c)** Compute the $z$-statistic and perform a two-sided test at $\alpha = 0.05$.

??? success "Solution to Exercise 1"

    **(a)** Identifying the runs (maximal consecutive identical elements):

    - Run 1: HH
    - Run 2: TTT
    - Run 3: HHHH
    - Run 4: TT
    - Run 5: H
    - Run 6: T
    - Run 7: HH
    - Run 8: TTT
    - Run 9: HH

    The number of runs is $R = 9$.

    **(b)** With $n_H = 11$ and $n_T = 9$:

    $$
    E[R] = \frac{2(11)(9)}{20} + 1 = \frac{198}{20} + 1 = 9.9 + 1 = 10.9
    $$

    $$
    \text{Var}(R) = \frac{2(11)(9)(2 \times 11 \times 9 - 11 - 9)}{20^2 \times 19} = \frac{198(198 - 20)}{400 \times 19} = \frac{198 \times 178}{7600} = \frac{35244}{7600} \approx 4.637
    $$

    **(c)**

    $$
    z = \frac{R - E[R]}{\sqrt{\text{Var}(R)}} = \frac{9 - 10.9}{\sqrt{4.637}} = \frac{-1.9}{2.153} \approx -0.883
    $$

    Two-sided p-value:

    $$
    p = 2 \times P(Z \le -0.883) = 2 \times \mathcal{N}(-0.883) \approx 2 \times 0.189 = 0.377
    $$

    Since $p = 0.377 \gg 0.05$, we fail to reject $H_0$. The sequence does not show statistically significant evidence of non-randomness. The observed number of runs (9) is close to the expected value (10.9), consistent with a random sequence.
