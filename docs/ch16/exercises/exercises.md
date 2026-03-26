# Exercises

These exercises cover the non-parametric testing methods from Chapter 16, including the sign test, Wilcoxon signed-rank test, Mann-Whitney U test, Kruskal-Wallis test, Friedman test, Spearman and Kendall correlation, and the runs test for randomness.

---

## Exercise 1: Sign Test

A nutrition study claims that a new diet reduces cholesterol. The cholesterol levels (mg/dL) of 10 patients are measured before and after the diet:

| Patient | Before | After | Difference (Before $-$ After) |
|:---:|:---:|:---:|:---:|
| 1 | 220 | 210 | 10 |
| 2 | 240 | 235 | 5 |
| 3 | 195 | 200 | $-5$ |
| 4 | 260 | 245 | 15 |
| 5 | 230 | 228 | 2 |
| 6 | 215 | 210 | 5 |
| 7 | 250 | 240 | 10 |
| 8 | 205 | 208 | $-3$ |
| 9 | 235 | 220 | 15 |
| 10 | 245 | 235 | 10 |

**(a)** State the null and alternative hypotheses for a one-sided sign test.

**(b)** Count the number of positive and negative signs (ignoring zeros). Compute the test statistic.

**(c)** Under $H_0$, what distribution does the number of positive signs follow? Compute the p-value.

**(d)** At $\alpha = 0.05$, what is your conclusion?

??? success "Solution"

    **(a)** Let $\tilde{\mu}_d$ denote the population median of the paired differences.

    - $H_0$: $\tilde{\mu}_d = 0$ (the diet has no effect)
    - $H_1$: $\tilde{\mu}_d > 0$ (the diet reduces cholesterol)

    **(b)** Counting signs of the differences: 8 positive ($+$), 2 negative ($-$), 0 zeros. The test statistic for the sign test is the number of positive signs: $S^+ = 8$.

    **(c)** Under $H_0$, each difference is equally likely to be positive or negative, so $S^+ \sim \text{Binomial}(n = 10, p = 0.5)$.

    For a one-sided test ($H_1$: median $> 0$), the p-value is:

    $$
    p = P(S^+ \ge 8) = P(S^+ = 8) + P(S^+ = 9) + P(S^+ = 10)
    $$

    $$
    = \binom{10}{8}(0.5)^{10} + \binom{10}{9}(0.5)^{10} + \binom{10}{10}(0.5)^{10}
    $$

    $$
    = \frac{45 + 10 + 1}{1024} = \frac{56}{1024} \approx 0.0547
    $$

    **(d)** Since $p \approx 0.055 > 0.05$, we fail to reject $H_0$ at the 5% significance level. The evidence for a cholesterol reduction is suggestive but not statistically significant by the sign test. Note that the sign test is conservative because it discards magnitude information — the Wilcoxon signed-rank test (see Exercise 2) may yield a different conclusion.

---

## Exercise 2: Wilcoxon Signed-Rank Test

Using the same cholesterol data from Exercise 1, perform a Wilcoxon signed-rank test.

**(a)** Compute the absolute differences $|d_i|$ and rank them from smallest to largest. Assign signed ranks.

**(b)** Compute the test statistic $W^+ = \sum \text{(ranks of positive differences)}$.

**(c)** For $n = 10$, under $H_0$ the expected value of $W^+$ is $n(n+1)/4$ and the variance is $n(n+1)(2n+1)/24$. Compute the $z$-statistic and the approximate p-value.

**(d)** Compare the conclusion with the sign test result from Exercise 1.

??? success "Solution"

    **(a)** Absolute differences and ranks:

    | Patient | $d_i$ | $|d_i|$ | Rank | Signed rank |
    |:---:|:---:|:---:|:---:|:---:|
    | 5 | 2 | 2 | 1 | $+1$ |
    | 3 | $-5$ | 5 | 3 | $-3$ |
    | 8 | $-3$ | 3 | 2 | $-2$ |
    | 2 | 5 | 5 | 3 | $+3$ |
    | 6 | 5 | 5 | 3 | $+3$ |
    | 1 | 10 | 10 | 6.5 | $+6.5$ |
    | 7 | 10 | 10 | 6.5 | $+6.5$ |
    | 10 | 10 | 10 | 6.5 | $+6.5$ |
    | 4 | 15 | 15 | 9.5 | $+9.5$ |
    | 9 | 15 | 15 | 9.5 | $+9.5$ |

    Tied values of $|d_i|$ receive the average of the ranks they would occupy. The three values of 5 occupy ranks 2, 3, 4, so each gets rank 3. The three values of 10 occupy ranks 6, 7, 8, so each gets rank $(6+7+8)/3 = 7$. Let me re-rank properly:

    Sorted $|d_i|$: 2, 3, 5, 5, 5, 10, 10, 10, 15, 15.

    - 2 → rank 1
    - 3 → rank 2
    - 5, 5, 5 → ranks 3, 4, 5 → average rank 4
    - 10, 10, 10 → ranks 6, 7, 8 → average rank 7
    - 15, 15 → ranks 9, 10 → average rank 9.5

    | Patient | $d_i$ | $|d_i|$ | Rank | Signed rank |
    |:---:|:---:|:---:|:---:|:---:|
    | 5 | 2 | 2 | 1 | $+1$ |
    | 8 | $-3$ | 3 | 2 | $-2$ |
    | 3 | $-5$ | 5 | 4 | $-4$ |
    | 2 | 5 | 5 | 4 | $+4$ |
    | 6 | 5 | 5 | 4 | $+4$ |
    | 1 | 10 | 10 | 7 | $+7$ |
    | 7 | 10 | 10 | 7 | $+7$ |
    | 10 | 10 | 10 | 7 | $+7$ |
    | 4 | 15 | 15 | 9.5 | $+9.5$ |
    | 9 | 15 | 15 | 9.5 | $+9.5$ |

    **(b)**

    $$
    W^+ = 1 + 4 + 4 + 7 + 7 + 7 + 9.5 + 9.5 = 49
    $$

    $$
    W^- = 2 + 4 = 6
    $$

    Check: $W^+ + W^- = 49 + 6 = 55 = n(n+1)/2 = 10 \times 11/2$.

    **(c)** Under $H_0$:

    $$
    E[W^+] = \frac{n(n+1)}{4} = \frac{10 \times 11}{4} = 27.5
    $$

    $$
    \text{Var}(W^+) = \frac{n(n+1)(2n+1)}{24} = \frac{10 \times 11 \times 21}{24} = \frac{2310}{24} = 96.25
    $$

    $$
    z = \frac{W^+ - E[W^+]}{\sqrt{\text{Var}(W^+)}} = \frac{49 - 27.5}{\sqrt{96.25}} = \frac{21.5}{9.811} \approx 2.19
    $$

    For a one-sided test, the p-value is:

    $$
    p = P(Z \ge 2.19) = 1 - \Phi(2.19) \approx 0.0143
    $$

    **(d)** The Wilcoxon signed-rank test gives $p \approx 0.014$, which is significant at $\alpha = 0.05$, whereas the sign test gave $p \approx 0.055$, which was not significant. The Wilcoxon test reaches significance because it uses the magnitudes of the differences — the larger positive differences (10 and 15) receive high ranks that contribute substantially to $W^+$. The sign test treats all positive differences equally, wasting this information.

---

## Exercise 3: Mann-Whitney U Test

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

??? success "Solution"

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
    p = 2 \times P(Z \le -2.556) = 2 \times \Phi(-2.556) \approx 2 \times 0.0053 = 0.0106
    $$

    **(d)** With $p \approx 0.011 < 0.05$, we reject $H_0$ at the 5% level. There is statistically significant evidence that the caffeine and placebo groups have different reaction time distributions. The caffeine group tends to have faster reaction times, consistent with the stimulant effect of caffeine.

---

## Exercise 4: Kruskal-Wallis Test

Three fertilizers are tested on plant growth (height in cm after 4 weeks):

- **Fertilizer A**: 15, 18, 20, 17
- **Fertilizer B**: 22, 25, 19, 23
- **Fertilizer C**: 12, 14, 16, 13

**(a)** Combine all observations and assign ranks.

**(b)** Compute the mean rank for each group.

**(c)** The Kruskal-Wallis test statistic is:

$$
H = \frac{12}{N(N+1)} \sum_{i=1}^{k} n_i (\bar{R}_i - \bar{R})^2
$$

where $N$ is the total sample size, $n_i$ is the size of group $i$, $\bar{R}_i$ is the mean rank for group $i$, and $\bar{R} = (N+1)/2$. Compute $H$.

**(d)** Under $H_0$, $H$ approximately follows a $\chi^2$ distribution with $k - 1$ degrees of freedom. Find the p-value and state your conclusion at $\alpha = 0.05$.

??? success "Solution"

    **(a)** Combined and sorted with ranks:

    | Value | Group | Rank |
    |:---:|:---:|:---:|
    | 12 | C | 1 |
    | 13 | C | 2 |
    | 14 | C | 3 |
    | 15 | A | 4 |
    | 16 | C | 5 |
    | 17 | A | 6 |
    | 18 | A | 7 |
    | 19 | B | 8 |
    | 20 | A | 9 |
    | 22 | B | 10 |
    | 23 | B | 11 |
    | 25 | B | 12 |

    **(b)** Mean ranks:

    $$
    \bar{R}_A = \frac{4 + 6 + 7 + 9}{4} = \frac{26}{4} = 6.5
    $$

    $$
    \bar{R}_B = \frac{8 + 10 + 11 + 12}{4} = \frac{41}{4} = 10.25
    $$

    $$
    \bar{R}_C = \frac{1 + 2 + 3 + 5}{4} = \frac{11}{4} = 2.75
    $$

    The overall mean rank is $\bar{R} = (N+1)/2 = 13/2 = 6.5$.

    **(c)**

    $$
    H = \frac{12}{12 \times 13} \sum_{i=1}^{3} 4(\bar{R}_i - 6.5)^2
    $$

    $$
    = \frac{12}{156}\left[4(6.5 - 6.5)^2 + 4(10.25 - 6.5)^2 + 4(2.75 - 6.5)^2\right]
    $$

    $$
    = \frac{1}{13}\left[4(0) + 4(14.0625) + 4(14.0625)\right] = \frac{1}{13}(0 + 56.25 + 56.25) = \frac{112.5}{13} \approx 8.654
    $$

    **(d)** Under $H_0$, $H \sim \chi^2(k-1) = \chi^2(2)$. The critical value for $\chi^2(2)$ at $\alpha = 0.05$ is 5.991.

    Since $H = 8.654 > 5.991$, we reject $H_0$. The p-value is $P(\chi^2(2) > 8.654) \approx 0.013$.

    There is significant evidence that the three fertilizers produce different growth distributions. Post-hoc analysis (e.g., Dunn's test) would be needed to determine which pairs differ.

---

## Exercise 5: Runs Test for Randomness

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

??? success "Solution"

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
    p = 2 \times P(Z \le -0.883) = 2 \times \Phi(-0.883) \approx 2 \times 0.189 = 0.377
    $$

    Since $p = 0.377 \gg 0.05$, we fail to reject $H_0$. The sequence does not show statistically significant evidence of non-randomness. The observed number of runs (9) is close to the expected value (10.9), consistent with a random sequence.

---

## Exercise 6: Friedman Test

Four panelists rate three brands of coffee on a 1--10 scale:

| Panelist | Brand A | Brand B | Brand C |
|:---:|:---:|:---:|:---:|
| 1 | 7 | 5 | 8 |
| 2 | 6 | 8 | 7 |
| 3 | 5 | 4 | 9 |
| 4 | 8 | 6 | 7 |

**(a)** Rank the brands within each panelist (block) from 1 to 3.

**(b)** Compute the rank sums $R_j$ for each brand across all panelists.

**(c)** Compute the Friedman test statistic:

$$
\chi_F^2 = \frac{12}{bk(k+1)} \sum_{j=1}^{k} R_j^2 - 3b(k+1)
$$

where $b = 4$ (panelists) and $k = 3$ (brands).

**(d)** Under $H_0$, $\chi_F^2 \sim \chi^2(k-1)$. Test at $\alpha = 0.05$.

??? success "Solution"

    **(a)** Ranking within each panelist (1 = lowest, 3 = highest):

    | Panelist | Brand A | Brand B | Brand C |
    |:---:|:---:|:---:|:---:|
    | 1 | 2 | 1 | 3 |
    | 2 | 1 | 3 | 2 |
    | 3 | 2 | 1 | 3 |
    | 4 | 3 | 1 | 2 |

    **(b)** Rank sums:

    $$
    R_A = 2 + 1 + 2 + 3 = 8
    $$

    $$
    R_B = 1 + 3 + 1 + 1 = 6
    $$

    $$
    R_C = 3 + 2 + 3 + 2 = 10
    $$

    Check: $R_A + R_B + R_C = 24 = bk(k+1)/2 = 4 \times 3 \times 4/2 = 24$.

    **(c)**

    $$
    \chi_F^2 = \frac{12}{4 \times 3 \times 4}(8^2 + 6^2 + 10^2) - 3 \times 4 \times 4
    $$

    $$
    = \frac{12}{48}(64 + 36 + 100) - 48 = \frac{12 \times 200}{48} - 48 = 50 - 48 = 2.0
    $$

    **(d)** Under $H_0$, $\chi_F^2 \sim \chi^2(2)$. The critical value at $\alpha = 0.05$ is 5.991.

    Since $\chi_F^2 = 2.0 < 5.991$, we fail to reject $H_0$. There is no statistically significant difference in the ratings of the three coffee brands. The p-value is $P(\chi^2(2) > 2.0) \approx 0.368$. With only 4 panelists, the test has limited power to detect moderate differences.

---

## Exercise 7: Spearman Rank Correlation

A teacher ranks 8 students by their performance on a math test and an English test:

| Student | Math rank | English rank |
|:---:|:---:|:---:|
| 1 | 1 | 3 |
| 2 | 2 | 1 |
| 3 | 3 | 2 |
| 4 | 4 | 5 |
| 5 | 5 | 4 |
| 6 | 6 | 8 |
| 7 | 7 | 6 |
| 8 | 8 | 7 |

**(a)** Compute the rank differences $d_i$ and $d_i^2$.

**(b)** Compute Spearman's rank correlation coefficient:

$$
r_s = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}
$$

**(c)** Test whether $r_s$ is significantly different from zero using the $t$-approximation:

$$
t = r_s \sqrt{\frac{n-2}{1 - r_s^2}}
$$

with $n - 2$ degrees of freedom.

??? success "Solution"

    **(a)**

    | Student | Math | English | $d_i$ | $d_i^2$ |
    |:---:|:---:|:---:|:---:|:---:|
    | 1 | 1 | 3 | $-2$ | 4 |
    | 2 | 2 | 1 | 1 | 1 |
    | 3 | 3 | 2 | 1 | 1 |
    | 4 | 4 | 5 | $-1$ | 1 |
    | 5 | 5 | 4 | 1 | 1 |
    | 6 | 6 | 8 | $-2$ | 4 |
    | 7 | 7 | 6 | 1 | 1 |
    | 8 | 8 | 7 | 1 | 1 |

    $$
    \sum d_i^2 = 4 + 1 + 1 + 1 + 1 + 4 + 1 + 1 = 14
    $$

    **(b)**

    $$
    r_s = 1 - \frac{6 \times 14}{8(64 - 1)} = 1 - \frac{84}{504} = 1 - 0.1667 = 0.8333
    $$

    **(c)** The $t$-statistic:

    $$
    t = 0.8333 \sqrt{\frac{6}{1 - 0.6944}} = 0.8333 \sqrt{\frac{6}{0.3056}} = 0.8333 \sqrt{19.63} = 0.8333 \times 4.431 = 3.692
    $$

    With 6 degrees of freedom, the critical value for a two-sided test at $\alpha = 0.05$ is $t_{0.025, 6} = 2.447$.

    Since $|t| = 3.69 > 2.447$, we reject $H_0$. The p-value is approximately 0.010. There is a statistically significant positive monotonic association between math and English performance ($r_s = 0.83$).

---

## Exercise 8: Kendall's Tau

Using the same data from Exercise 7, compute Kendall's tau.

**(a)** For each pair of students $(i, j)$ with $i < j$, determine whether the pair is concordant or discordant. A pair is concordant if the math ranks and English ranks are ordered in the same direction.

**(b)** Compute Kendall's tau:

$$
\tau = \frac{C - D}{\binom{n}{2}}
$$

where $C$ is the number of concordant pairs and $D$ is the number of discordant pairs.

**(c)** Compare Kendall's tau with Spearman's rho from Exercise 7. Why is $|\tau| < |r_s|$ in general?

??? success "Solution"

    **(a)** With $n = 8$, there are $\binom{8}{2} = 28$ pairs. Since math ranks are already $1, 2, \ldots, 8$, a pair $(i,j)$ with $i < j$ is concordant if the English rank of student $j$ exceeds that of student $i$.

    English ranks in order of math rank: 3, 1, 2, 5, 4, 8, 6, 7.

    For each student, count how many subsequent English ranks are larger (concordant):

    - Student 1 (English 3): compared with 1, 2, 5, 4, 8, 6, 7 → larger: 5, 4, 8, 6, 7 → $C_1 = 5$, $D_1 = 2$
    - Student 2 (English 1): compared with 2, 5, 4, 8, 6, 7 → larger: all 6 → $C_2 = 6$, $D_2 = 0$
    - Student 3 (English 2): compared with 5, 4, 8, 6, 7 → larger: all 5 → $C_3 = 5$, $D_3 = 0$
    - Student 4 (English 5): compared with 4, 8, 6, 7 → larger: 8, 6, 7 → $C_4 = 3$, $D_4 = 1$
    - Student 5 (English 4): compared with 8, 6, 7 → larger: all 3 → $C_5 = 3$, $D_5 = 0$
    - Student 6 (English 8): compared with 6, 7 → larger: none → $C_6 = 0$, $D_6 = 2$
    - Student 7 (English 6): compared with 7 → larger: 7 → $C_7 = 1$, $D_7 = 0$

    Totals: $C = 5 + 6 + 5 + 3 + 3 + 0 + 1 = 23$, $D = 2 + 0 + 0 + 1 + 0 + 2 + 0 = 5$.

    Check: $C + D = 28 = \binom{8}{2}$.

    **(b)**

    $$
    \tau = \frac{23 - 5}{28} = \frac{18}{28} \approx 0.643
    $$

    **(c)** For these data, $\tau = 0.643$ while $r_s = 0.833$. Kendall's tau is generally smaller in magnitude than Spearman's rho for the same data because they use different scales. Spearman's rho squares the rank differences (amplifying large discrepancies), while Kendall's tau counts pairwise concordances (a binary classification). Roughly, $\tau \approx \frac{2}{\pi}\arcsin(r_s)$ for bivariate normal data, which gives $|\tau| < |r_s|$ except at the extremes $\pm 1$, where they agree. Despite the different magnitudes, both measures indicate a strong positive monotonic association.

---

## Exercise 9: Kolmogorov-Smirnov Two-Sample Test

Two groups of students take different preparation courses, and their exam scores are:

- **Course 1**: 72, 78, 85, 90, 65
- **Course 2**: 80, 88, 92, 95, 85, 76

**(a)** Compute the empirical CDF $\hat{F}_1(x)$ and $\hat{F}_2(x)$ for each group.

**(b)** Find the Kolmogorov-Smirnov test statistic $D = \max_x |\hat{F}_1(x) - \hat{F}_2(x)|$.

**(c)** Explain what a large value of $D$ indicates about the two distributions.

??? success "Solution"

    **(a)** Sort each sample and compute the ECDF (each jump has size $1/n_i$):

    **Course 1** ($n_1 = 5$): 65, 72, 78, 85, 90. ECDF jumps by $1/5 = 0.2$ at each value.

    **Course 2** ($n_2 = 6$): 76, 80, 85, 88, 92, 95. ECDF jumps by $1/6 \approx 0.167$ at each value.

    **(b)** To find $D$, evaluate $|\hat{F}_1(x) - \hat{F}_2(x)|$ at every observed value:

    | $x$ | $\hat{F}_1(x)$ | $\hat{F}_2(x)$ | $|\hat{F}_1 - \hat{F}_2|$ |
    |:---:|:---:|:---:|:---:|
    | 65 | 0.2 | 0 | 0.200 |
    | 72 | 0.4 | 0 | 0.400 |
    | 76 | 0.4 | 1/6 | 0.233 |
    | 78 | 0.6 | 1/6 | 0.433 |
    | 80 | 0.6 | 2/6 | 0.267 |
    | 85 | 0.8 | 3/6 | 0.300 |
    | 88 | 0.8 | 4/6 | 0.133 |
    | 90 | 1.0 | 4/6 | 0.333 |
    | 92 | 1.0 | 5/6 | 0.167 |
    | 95 | 1.0 | 1.0 | 0.000 |

    $$
    D = \max_x |\hat{F}_1(x) - \hat{F}_2(x)| = 0.433 \text{ (at } x = 78\text{)}
    $$

    **(c)** A large value of $D$ indicates that the two empirical CDFs differ substantially, suggesting the two samples come from different underlying distributions. The KS test is sensitive to differences in both location (shift) and shape. For these data, $D = 0.433$ suggests Course 2 students tend to score higher, but with $n_1 = 5$ and $n_2 = 6$, the critical value at $\alpha = 0.05$ is approximately $c(\alpha)\sqrt{(n_1 + n_2)/(n_1 n_2)} = 1.36\sqrt{11/30} \approx 0.823$, so $D = 0.433 < 0.823$ and we would not reject $H_0$ at this sample size.

---

## Exercise 10: Choosing Between Parametric and Non-Parametric Tests

For each scenario below, state whether you would use a parametric or non-parametric test, name the specific test, and justify your choice.

**(a)** You want to compare the mean blood pressure of two groups (drug vs. placebo). Both groups have $n = 50$ observations, and Q-Q plots suggest approximate normality.

**(b)** You have 8 observations of customer satisfaction ratings (on a 1--5 Likert scale) from two store locations and want to test if the locations differ.

**(c)** You have paired before/after measurements for 12 subjects, but the differences are heavily right-skewed with one extreme outlier.

**(d)** You want to test whether three teaching methods produce different exam score distributions. Group sizes are 8, 10, and 7, and Shapiro-Wilk tests reject normality in two of the three groups.

??? success "Solution"

    **(a)** **Parametric: two-sample $t$-test** (or Welch's $t$-test). With $n = 50$ per group and approximate normality confirmed by Q-Q plots, the conditions for a parametric test are well satisfied. The $t$-test will have higher power than a non-parametric alternative under these conditions.

    **(b)** **Non-parametric: Mann-Whitney U test** (Wilcoxon rank-sum test). Likert-scale data are ordinal, not continuous, so means and standard deviations are not meaningful. The small sample size ($n = 8$) and discrete nature of the data make non-parametric methods more appropriate.

    **(c)** **Non-parametric: Wilcoxon signed-rank test** (or even the sign test if symmetry of differences is in doubt). The heavy skewness and extreme outlier violate the normality assumption of the paired $t$-test. With only 12 observations, the CLT does not provide reliable normal approximations for highly skewed data. The Wilcoxon signed-rank test, based on ranks, is resistant to the outlier.

    **(d)** **Non-parametric: Kruskal-Wallis test**. Since normality is rejected in two of three groups, one-way ANOVA assumptions are violated. The sample sizes are relatively small (7--10), offering insufficient data for the CLT to compensate. The Kruskal-Wallis test does not require normality and is the appropriate multi-group comparison. If the Kruskal-Wallis test is significant, follow up with Dunn's test for pairwise comparisons.
