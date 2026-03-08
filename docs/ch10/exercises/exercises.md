# Exercises

## Exercise 1: Normality Test Using Goodness-of-Fit Test

A researcher divided independently obtained data into 4 intervals to check whether the data follows a normal distribution and counted the frequencies for each interval. The data is as follows:

- **Observed frequencies**: $[10, 30, 50, 10]$
- **Expected frequencies (based on normal distribution)**: $[20, 25, 40, 15]$

Using a chi-square test, determine whether the data can be modeled as following a normal distribution.
The significance level is 5%, and the critical value is $\chi^2_{0.05,3} = 7.815$.
If the null hypothesis is not rejected, the data is considered to follow normality.

**(a)** Check the conditions required to perform a chi-square test.

**(b)** State the null hypothesis ($H_0$) and the alternative hypothesis ($H_a$).

**(c)** Calculate the test statistic.

**(d)** Describe the test result.

### Solution

**(a)**

- Since the data was obtained independently, the independence of the data is ensured.
- The expected frequency in each interval is greater than or equal to 5, so the frequency condition is satisfied.

**(b)**

1. $H_0$: The data follows a normal distribution.
2. $H_a$: The data does not follow a normal distribution.

**(c)**

$$
\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i} = \frac{(10-20)^2}{20} + \frac{(30-25)^2}{25} + \frac{(50-40)^2}{40} + \frac{(10-15)^2}{15}
$$

$$
= 5.00 + 1.00 + 2.50 + 1.67 = 10.17
$$

**Note**: The original source states $\chi^2 = 5.63$, which may use different observed/expected values. Using the values as given above, $\chi^2 \approx 10.17$.

**(d)**

- If $\chi^2 = 5.63 < 7.815$: We do not reject the null hypothesis. The data follows normality.
- If $\chi^2 = 10.17 > 7.815$: We reject the null hypothesis. The data does not follow a normal distribution.

---

## Exercise 2: Assessing the Suitability of a Chi-Square Test for Plant Height Data

In a study, the heights of a specific plant were measured under two conditions (A and B):

- Condition A: $[15, 20, 25]$
- Condition B: $[10, 15, 35]$

Test whether the difference in distribution between the two conditions is significant. Is this data suitable for a chi-square test? Explain why.

### Solution

To evaluate whether the chi-square test is suitable, we check the assumptions:

1. **The data must be categorical**: The chi-square test is used for categorical data (nominal or ordinal).
2. **The data must represent frequencies**: The chi-square test compares observed and expected frequencies. The data must be counts, not continuous measurements.
3. **Expected frequencies must be sufficiently large**: Each category's expected frequency should be at least 5.

**Conclusion**: The given data is continuous (plant heights), not categorical, and thus is not suitable for a chi-square test. Instead, it would be more appropriate to perform a **t-test** or a **non-parametric test** (such as the Mann-Whitney U test) to analyze the difference in height distributions between conditions A and B.

---

## Exercise 3: Testing the Independence Between Payment Methods and Days of the Week

In a shopping mall, a study was conducted to investigate the relationship between customers' preferred payment methods (cash, card, mobile payment) and the day of the week (weekend, weekday). The data is as follows:

|         | Cash | Card | Mobile Payment |
|---------|------|------|----------------|
| Weekend | 30   | 50   | 20             |
| Weekday | 40   | 60   | 30             |

Test whether the payment method and day of the week are independent. The significance level ($\alpha$) is 1%.

**(a)** State the null hypothesis ($H_0$) and the alternative hypothesis ($H_a$).

**(b)** Calculate the expected frequencies.

**(c)** Calculate the test statistic.

**(d)** Calculate the degrees of freedom ($df$).

**(e)** Express the critical value using the significance level ($\alpha$) and the PPF $G$, the inverse of the CDF.

### Solution

**(a)**

- $H_0$: Payment method and day of the week are independent.
- $H_a$: Payment method and day of the week are not independent.

**(b)**

$$
E_{ij} = \frac{\text{Row Total} \times \text{Column Total}}{\text{Grand Total}}
$$

$$
\begin{array}{|c|c|c|c|c|}
\hline
  & \text{Cash} & \text{Card} & \text{Mobile} & \text{Row Total} \\
\hline
\text{Weekend} & \frac{100 \times 70}{230} = 30.43 & \frac{100 \times 110}{230} = 47.83 & \frac{100 \times 50}{230} = 21.74 & 100 \\
\hline
\text{Weekday} & \frac{130 \times 70}{230} = 39.57 & \frac{130 \times 110}{230} = 62.17 & \frac{130 \times 50}{230} = 28.26 & 130 \\
\hline
\text{Col Total} & 70 & 110 & 50 & 230 \\
\hline
\end{array}
$$

**(c)**

$$
\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}} \approx 2.34
$$

**(d)**

$$
df = (2 - 1)(3 - 1) = 2
$$

**(e)**

$$
\text{Critical value} = G(1 - \alpha)
$$

where $G$ is the inverse CDF (percent point function) of the $\chi^2$ distribution with 2 degrees of freedom.

---

## Exercise 4: Comparing the Effectiveness of Two Drugs

An experiment was conducted to determine whether two drugs are effective in treating a disease. The data obtained is as follows:

- **Drug A**: 60 successes, 40 failures.
- **Drug B**: 55 successes, 45 failures.

**(a)** Explain how to prevent distortions in results if other factors (e.g., patient age, gender, disease severity) that could influence the comparison are not properly controlled.

**(b)** Use a $z$-test to determine if the performance of Drug A and Drug B is the same. Let $\alpha = 5\%$ and the critical value $= 1.96$.

**(c)** Use a chi-square test to determine if the performance of Drug A and Drug B is the same. Let $\alpha = 5\%$ and the critical value $= 3.841$.

### Solution

**(a)**

Participants should be randomly assigned to Drug A and Drug B to eliminate confounding variables unrelated to the drug's effects. Random assignment helps reduce bias by ensuring that characteristics of the participants (e.g., age, health status) do not cause differences between the groups.

**(b)**

1. **Hypotheses**:
   - $H_0$: The success rates of Drug A and Drug B are the same.
   - $H_a$: The success rates of Drug A and Drug B are different.

2. **Pooled Proportion**:

$$
\hat{p} = \frac{x_1 + x_2}{n_1 + n_2} = \frac{60 + 55}{100 + 100} = 0.575
$$

3. **Test Statistic**:

$$
z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1 - \hat{p}) \left( \frac{1}{n_1} + \frac{1}{n_2} \right)}}
$$

4. **Result**: The calculated $z$-value is approximately $0.715$. Since $|z| < 1.96$, we do not reject the null hypothesis. There is no significant difference in the performance of Drug A and Drug B.

**(c)**

1. **Contingency Table**:

$$
\begin{array}{|c|c|c|}
\hline
 & \text{Success} & \text{Failure} \\
\hline
\text{Drug A} & 60 & 40 \\
\text{Drug B} & 55 & 45 \\
\hline
\end{array}
$$

2. **Expected Frequencies**:

$$
\begin{array}{|c|c|c|c|}
\hline
 & \text{Success} & \text{Failure} & \text{Row Total} \\
\hline
\text{Drug A} & 57.5 & 42.5 & 100 \\
\hline
\text{Drug B} & 57.5 & 42.5 & 100 \\
\hline
\text{Col Total} & 115 & 85 & 200 \\
\hline
\end{array}
$$

3. **Test Statistic**:

$$
\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}} = 0.512
$$

```python
# Observed frequencies
observed = [[60, 40], [55, 45]]

# Expected frequencies (precomputed)
expected = [[57.5, 42.5], [57.5, 42.5]]

# Calculate chi-square statistic
chi_square_stat = sum(
    (obs - exp) ** 2 / exp
    for row_obs, row_exp in zip(observed, expected)
    for obs, exp in zip(row_obs, row_exp)
)

print(f"{chi_square_stat = :.3f}")
```

4. **Result**: The calculated $\chi^2$-value is approximately $0.512$. Since $\chi^2 < 3.841$, we do not reject the null hypothesis. There is no significant difference in the performance of Drug A and Drug B.

**Note**: For a $2 \times 2$ table, the relationship between the $z$-test and chi-square test is $\chi^2 = z^2$. Indeed, $0.715^2 \approx 0.511 \approx 0.512$, confirming the equivalence of the two approaches.
