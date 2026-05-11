# Mann-Whitney U Test

The **Mann-Whitney U test** is an alternative formulation of the [Wilcoxon rank-sum test](rank_sum.md) that is mathematically equivalent but offers a different -- and often more intuitive -- interpretation. While the rank-sum test focuses on the sum of ranks in one group, the Mann-Whitney test counts the number of times an observation from one group exceeds an observation from the other. This count, the $U$ statistic, directly estimates the probability $P(X > Y)$, giving the test a clear probabilistic meaning.

## Relationship to the Rank-Sum Test

If $W$ is the Wilcoxon rank-sum statistic for group 1 (with sample size $n_1$), then

$$
U_1 = W - \frac{n_1(n_1 + 1)}{2}
$$

and

$$
U_2 = n_1 n_2 - U_1
$$

The Mann-Whitney $U$ statistic is $U = \min(U_1, U_2)$. The rank-sum test and the Mann-Whitney test always yield the same $p$-value.

## Intuition: Counting Pairwise Wins

Consider all $n_1 \times n_2$ pairs $(X_i, Y_j)$ formed by taking one observation from each group. Define

$$
U_1 = \sum_{i=1}^{n_1} \sum_{j=1}^{n_2} \mathbf{1}(X_i > Y_j)
$$

This counts the number of pairs in which the observation from group 1 exceeds the observation from group 2. Conversely,

$$
U_2 = \sum_{i=1}^{n_1} \sum_{j=1}^{n_2} \mathbf{1}(Y_j > X_i)
$$

Note that $U_1 + U_2 = n_1 n_2$ when there are no ties.

The ratio $U_1 / (n_1 n_2)$ estimates the probability $P(X > Y)$. Under $H_0$ (identical distributions), $P(X > Y) = 0.5$, so $U_1 \approx n_1 n_2 / 2$.

## Hypotheses

$$
H_0 \colon P(X > Y) = 0.5 \quad \text{(the two distributions are identical)}
$$

$$
H_a \colon P(X > Y) \ne 0.5 \quad \text{(two-sided)}
$$

One-sided alternatives: $H_a \colon P(X > Y) > 0.5$ (group 1 tends to be larger) or $P(X > Y) < 0.5$.

## Test Statistic

The test statistic is

$$
U = \min(U_1, U_2)
$$

Small values of $U$ indicate a large difference between the groups.

## Null Distribution

Under $H_0$, the mean and variance of $U_1$ are

$$
\mu_U = \frac{n_1 \, n_2}{2}
$$

$$
\sigma_U^2 = \frac{n_1 \, n_2 \,(n_1 + n_2 + 1)}{12}
$$

With a tie correction for $g$ groups of tied observations of sizes $t_1, \ldots, t_g$ (in the combined sample of size $N = n_1 + n_2$):

$$
\sigma_U^2 = \frac{n_1 \, n_2}{12}\left(N + 1 - \frac{\sum_{j=1}^{g}(t_j^3 - t_j)}{N(N - 1)}\right)
$$

## Normal Approximation

For $n_1, n_2 \ge 10$:

$$
Z = \frac{U_1 - \mu_U}{\sigma_U}
$$

is approximately $\mathcal{N}(0, 1)$ under $H_0$.

## Worked Example

An investor compares daily returns (in %) of two stocks over 7 and 6 trading days.

**Stock A** ($n_1 = 7$): 1.2, 0.5, $-0.3$, 2.1, 0.8, 1.5, 0.1

**Stock B** ($n_2 = 6$): $-0.5$, 0.3, $-1.0$, 0.6, $-0.2$, 0.9

**Step 1.** Form all $7 \times 6 = 42$ pairs and count how many times a Stock A return exceeds a Stock B return:

Sorted Stock A: $-0.3, 0.1, 0.5, 0.8, 1.2, 1.5, 2.1$

Sorted Stock B: $-1.0, -0.5, -0.2, 0.3, 0.6, 0.9$

For each Stock A value, count how many Stock B values it exceeds:

| Stock A | Stock B values exceeded | Count |
|:-------:|:-----------------------:|:-----:|
| $-0.3$ | $-1.0, -0.5$ | 2 |
| 0.1 | $-1.0, -0.5, -0.2$ | 3 |
| 0.5 | $-1.0, -0.5, -0.2, 0.3$ | 4 |
| 0.8 | $-1.0, -0.5, -0.2, 0.3, 0.6$ | 5 |
| 1.2 | $-1.0, -0.5, -0.2, 0.3, 0.6, 0.9$ | 6 |
| 1.5 | all 6 | 6 |
| 2.1 | all 6 | 6 |

$$
U_1 = 2 + 3 + 4 + 5 + 6 + 6 + 6 = 32
$$

$$
U_2 = 42 - 32 = 10
$$

**Step 2.** Probability estimate: $\hat{P}(X > Y) = 32/42 \approx 0.762$.

**Step 3.** Normal approximation:

$$
\mu_U = \frac{7 \times 6}{2} = 21
$$

$$
\sigma_U = \sqrt{\frac{7 \times 6 \times 14}{12}} = \sqrt{49} = 7
$$

$$
Z = \frac{32 - 21}{7} \approx 1.571
$$

$$
p = 2\,\mathcal{N}(-1.571) \approx 0.116
$$

At $\alpha = 0.05$, we fail to reject $H_0$. Although Stock A's returns tend to be higher (76.2% of pairwise comparisons), the small sample sizes do not provide sufficient evidence at the 5% level.

## Handling Ties

When tied observations occur between groups, each tie contributes 0.5 (instead of 0 or 1) to the pairwise comparison count:

$$
U_1 = \sum_{i=1}^{n_1} \sum_{j=1}^{n_2} \left[\mathbf{1}(X_i > Y_j) + \tfrac{1}{2}\,\mathbf{1}(X_i = Y_j)\right]
$$

The tie-corrected variance formula given above should then be used for the normal approximation.

## Effect Size

The **common language effect size** is simply the probability estimate:

$$
\hat{P}(X > Y) = \frac{U_1}{n_1 n_2}
$$

Values near 0.5 indicate no effect; values near 0 or 1 indicate strong stochastic dominance.

## Summary

The Mann-Whitney $U$ test counts pairwise wins between two independent groups and tests whether one group systematically produces larger values. The $U$ statistic is a linear function of the Wilcoxon rank-sum statistic, so the two tests are equivalent. The key advantage of the Mann-Whitney formulation is its interpretability: $U_1 / (n_1 n_2)$ directly estimates $P(X > Y)$, providing a natural effect size measure alongside the hypothesis test.

## Exercises

**Exercise 1.**
A researcher compares reaction times (in milliseconds) between two independent groups:

- **Group A** (caffeine): 210, 230, 215, 240, 225
- **Group B** (placebo): 250, 260, 235, 270, 245, 255

**(a)** Combine both groups and rank all observations from smallest to largest.

**(b)** Compute the rank sum $R_A$ for Group A, and then compute the Mann-Whitney $U$ statistic for Group A:

$$
U_A = R_A - \frac{n_A(n_A + 1)}{2}
$$

**(c)** Using the normal approximation with $E[U] = n_A n_B / 2$ and $\text{Var}(U) = n_A n_B (n_A + n_B + 1)/12$, compute the $z$-statistic and the two-sided p-value.

**(d)** Interpret the result in the context of the study.

??? success "Solution to Exercise 1"

    **(a)** Combined and sorted:

    | Value | Group | Rank |
    |:---:|:---:|:---:|
    | 210 | A | 1 |
    | 215 | A | 2 |
    | 225 | A | 3 |
    | 230 | A | 4 |
    | 235 | B | 5 |
    | 240 | A | 6 |
    | 245 | B | 7 |
    | 250 | B | 8 |
    | 255 | B | 9 |
    | 260 | B | 10 |
    | 270 | B | 11 |

    **(b)** $R_A = 1 + 2 + 3 + 4 + 6 = 16$ with $n_A = 5$, $n_B = 6$.

    $$
    U_A = R_A - \frac{n_A(n_A + 1)}{2} = 16 - \frac{5 \times 6}{2} = 16 - 15 = 1
    $$

    A small value of $U_A$ indicates that Group A observations tend to have low ranks (fast reaction times).

    **(c)**

    $$
    E[U] = \frac{n_A n_B}{2} = \frac{5 \times 6}{2} = 15
    $$

    $$
    \text{Var}(U) = \frac{n_A n_B(n_A + n_B + 1)}{12} = \frac{5 \times 6 \times 12}{12} = 30
    $$

    $$
    z = \frac{U_A - E[U]}{\sqrt{\text{Var}(U)}} = \frac{1 - 15}{\sqrt{30}} = \frac{-14}{5.477} \approx -2.556
    $$

    Two-sided p-value:

    $$
    p = 2 \times P(Z \le -2.556) = 2 \times \mathcal{N}(-2.556) \approx 2 \times 0.0053 = 0.0106
    $$

    **(d)** With $p \approx 0.011 < 0.05$, we reject $H_0$ at the 5% level. There is statistically significant evidence that the caffeine and placebo groups have different reaction time distributions. The caffeine group tends to have faster reaction times, consistent with the stimulant effect of caffeine.

- **Group A** (caffeine): 210, 230, 215, 240, 225
- **Group B** (placebo): 250, 260, 235, 270, 245, 255

**(a)** Combine both groups and rank all observations from smallest to largest.

**(b)** Compute the rank sum $R_A$ for Group A, and then compute the Mann-Whitney $U$ statistic for Group A:

$$
U_A = R_A - \frac{n_A(n_A + 1)}{2}
$$

**(c)** Using the normal approximation with $E[U] = n_A n_B / 2$ and $\text{Var}(U) = n_A n_B (n_A + n_B + 1)/12$, compute the $z$-statistic and the two-sided p-value.

**(d)** Interpret the result in the context of the study.

??? success "Solution to Exercise 1"

    **(a)** Combined and sorted:

    | Value | Group | Rank |
    |:---:|:---:|:---:|
    | 210 | A | 1 |
    | 215 | A | 2 |
    | 225 | A | 3 |
    | 230 | A | 4 |
    | 235 | B | 5 |
    | 240 | A | 6 |
    | 245 | B | 7 |
    | 250 | B | 8 |
    | 255 | B | 9 |
    | 260 | B | 10 |
    | 270 | B | 11 |

    **(b)** $R_A = 1 + 2 + 3 + 4 + 6 = 16$ with $n_A = 5$, $n_B = 6$.

    $$
    U_A = R_A - \frac{n_A(n_A + 1)}{2} = 16 - \frac{5 \times 6}{2} = 16 - 15 = 1
    $$

    A small value of $U_A$ indicates that Group A observations tend to have low ranks (fast reaction times).

    **(c)**

    $$
    E[U] = \frac{n_A n_B}{2} = \frac{5 \times 6}{2} = 15
    $$

    $$
    \text{Var}(U) = \frac{n_A n_B(n_A + n_B + 1)}{12} = \frac{5 \times 6 \times 12}{12} = 30
    $$

    $$
    z = \frac{U_A - E[U]}{\sqrt{\text{Var}(U)}} = \frac{1 - 15}{\sqrt{30}} = \frac{-14}{5.477} \approx -2.556
    $$

    Two-sided p-value:

    $$
    p = 2 \times P(Z \le -2.556) = 2 \times \mathcal{N}(-2.556) \approx 2 \times 0.0053 = 0.0106
    $$

    **(d)** With $p \approx 0.011 < 0.05$, we reject $H_0$ at the 5% level. There is statistically significant evidence that the caffeine and placebo groups have different reaction time distributions. The caffeine group tends to have faster reaction times, consistent with the stimulant effect of caffeine.
