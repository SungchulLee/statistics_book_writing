# Exercises

## 1. Writing Null and Alternative Hypotheses

[Writing null and alternative hypotheses (Khan Academy)](https://www.khanacademy.org/math/ap-statistics/xfb5d8e68:inference-categorical-proportions/idea-significance-tests/e/writing-null-and-alternative-hypotheses-informal)

## 2. Writing Hypotheses for a Test about a Mean

[Writing Hypotheses for a Test about a Mean (Khan Academy)](https://www.khanacademy.org/math/ap-statistics/xfb5d8e68:inference-quantitative-means/one-sample-t-test-mean/e/writing-hypotheses-one-sample-t-test-mean)

## 3. Automated Drink Machine

We are told a restaurant owner installed a new automated drink machine. The machine is designed to dispense 530 milliliters of liquid on the medium size setting. The owner suspects that the machine may be dispensing too much in medium drinks. They decide to take a sample of 30 medium drinks to see if the average amount is significantly greater than 530 milliliters. What are appropriate hypotheses for their significance test?

**Solution:**

$$H_0: \mu = 530 \quad \text{vs} \quad H_1: \mu > 530$$

## 4. Type I vs Type II Error

[Type I vs Type II Error (Khan Academy)](https://www.khanacademy.org/math/ap-statistics/xfb5d8e68:inference-categorical-proportions/error-probabilities-power/e/type-i-error-type-ii-error-power)

## 5. Unemployment Rate (Type I Error)

The mayor of a local town is interested in determining if the national unemployment rate of 9% applies to her town. She tests:

$$H_0: p = 0.09 \quad \text{vs} \quad H_1: p \neq 0.09$$

Under what conditions would the mayor commit a Type I error?

**Solution:** The town's unemployment rate is 9%. However, she concludes otherwise.

## 6. Estimating p-values from Simulations

[Estimating p-values from Simulations (Khan Academy)](https://www.khanacademy.org/math/ap-statistics/xfb5d8e68:inference-categorical-proportions/idea-significance-tests/e/estimating-p-values-and-making-conclusions)

## 7. Using P-values to Make Conclusions

[Using P-values to make conclusions (Khan Academy)](https://www.khanacademy.org/math/ap-statistics/xfb5d8e68:inference-categorical-proportions/idea-significance-tests/a/p-value-conclusions)

## 8. Estimation of p-value using Simulation

Evie read that 6% of teenagers are vegetarians. She believes the percentage at her school is higher. She sampled 25 students and found 20% are vegetarians.

$$H_0: p = 0.06 \quad \text{vs} \quad H_1: p > 0.06$$

She simulated 40 samples of $n=25$ from a population where 6% are vegetarian.

**Solution:**

```python
import numpy as np

statistic = 0.2
np.random.seed(0)

p = 0.06
num_samples = 40
n = 25
random_samples = np.random.binomial(1, p, size=(num_samples, n)).sum(axis=1) / n

zero_one = np.zeros_like(random_samples)
zero_one[random_samples >= statistic] = 1
estimated_p_value = zero_one.sum() / num_samples
print(f"{estimated_p_value = }")
```

## 9. Multilingual Americans (p-value)

Fay read that 26% of Americans can speak more than one language. She tests $H_0: p = 0.26$ vs $H_1: p > 0.26$. She found 40 of 120 people could speak more than one language.

**Solution:**

```python
import numpy as np
from scipy import stats

p = 0.26
n = 120
p_hat = 40 / 120
sigma = np.sqrt(p * (1 - p) / n)
z_score = (p_hat - p) / sigma
sf = stats.norm().sf(z_score)
print(f"p-value: {sf:.4f}")
```

## 10. Mean Lifetime of Light Bulbs (One Sample z Test)

A manufacturer claims mean lifetime is 1,200 hours. A sample of 36 bulbs has $\bar{X} = 1150$ hours, $s = 200$ hours. Test at $\alpha = 0.05$.

**Solution:**

$$H_0: \mu = 1200 \quad \text{vs} \quad H_1: \mu \neq 1200$$

$$z = \frac{1150 - 1200}{200/\sqrt{36}} = \frac{-50}{33.33} \approx -1.5$$

Critical values: $z = \pm 1.96$. Since $-1.96 < -1.5 < 1.96$, **fail to reject** $H_0$.

## 11. Conditions for a t Test about a Mean

[Conditions for a t Test about a Mean (Khan Academy)](https://www.khanacademy.org/math/ap-statistics/xfb5d8e68:inference-quantitative-means/one-sample-t-test-mean/e/checking-conditions-one-sample-t-test-mean)

## 12. Average Weight of Cereal (One Sample t Test)

Claimed average weight is 500g. Sample of 25 boxes: $\bar{X} = 490$g, $s = 15$g. Test at $\alpha = 0.01$.

**Solution:**

$$H_0: \mu = 500 \quad \text{vs} \quad H_1: \mu \neq 500$$

$$t = \frac{490 - 500}{15/\sqrt{25}} = \frac{-10}{3} \approx -3.33$$

Critical value: $t_{0.005, 24} \approx \pm 2.797$. Since $|-3.33| > 2.797$, **reject** $H_0$. The average weight is significantly different from 500g.

## 13. Pennies (One Sample Proportion Test)

Rahim tests $H_0: p = 0.5$ vs $H_1: p > 0.5$. In 100 spins, 59 showed heads ($\hat{p} = 0.59$, p-value $\approx 0.036$). At $\alpha = 0.05$: since $0.036 \leq 0.05$, reject $H_0$.

```python
import numpy as np
from scipy import stats

alpha = 0.05
p_hat = 0.59
n = 100
p_0 = 0.5

statistic = (p_hat - p_0) / np.sqrt(p_0 * (1 - p_0) / n)
p_value = stats.norm().sf(statistic)
print(f"{p_value = :.3f}")

if p_value < alpha:
    print("We choose H_1: p > 0.5")
else:
    print("We choose H_0: p = 0.5")
```

## 14. Proportion of Vehicles Passing Inspection

Claimed: 80% pass. Sample: 74 of 100 pass. Test at $\alpha = 0.05$.

**Solution:**

$$H_0: p = 0.80 \quad \text{vs} \quad H_1: p \neq 0.80$$

$$z = \frac{0.74 - 0.80}{\sqrt{0.80 \times 0.20 / 100}} = \frac{-0.06}{0.04} = -1.5$$

Critical values: $z = \pm 1.96$. Since $|-1.5| < 1.96$, **fail to reject** $H_0$.

## 15. Difference in Mean Salaries (Two Sample t Test)

Program A: $\bar{X}_1 = 55{,}000$, $s_1 = 7{,}500$, $n_1 = 14$. Program B: $\bar{X}_2 = 60{,}000$, $s_2 = 8{,}000$, $n_2 = 16$. Equal variances assumed. Test at $\alpha = 0.05$.

**Solution:**

$$S_p^2 = \frac{13 \times 56{,}250{,}000 + 15 \times 64{,}000{,}000}{28} \approx 60{,}402{,}679$$

$$t = \frac{55{,}000 - 60{,}000}{\sqrt{60{,}402{,}679 \times 0.1339}} \approx -1.76$$

With $df = 28$ and $t_{0.025,28} \approx \pm 2.048$: since $|-1.76| < 2.048$, **fail to reject** $H_0$.

## 16. Average Annual Income: Norway vs US

| | Norway | US |
|:---:|:---:|:---:|
| Mean | 64.3 | 53.4 |
| Std Dev | 18.2 | 23.9 |
| n | 65 | 75 |

Test $H_0: \mu_A = \mu_B$ vs $H_1: \mu_A \neq \mu_B$ at $\alpha = 0.05$.

```python
import numpy as np
from scipy import stats

X_1_bar, X_2_bar = 64.3, 53.4
s_1, s_2 = 18.2, 23.9
n_1, n_2 = 65, 75

statistic = (X_1_bar - X_2_bar) / np.sqrt(s_1**2 / n_1 + s_2**2 / n_2)
top = (s_1**2 / n_1 + s_2**2 / n_2)**2
bottom = (s_1**2 / n_1)**2 / n_1 + (s_2**2 / n_2)**2 / n_2
df = top / bottom
p_value = 2 * stats.t(df).cdf(-abs(statistic))

print(f"{df = :.4f}")
print(f"{statistic = :.4f}")
print(f"{p_value   = :.4f}")
```

## 17. Mean First Marriage Age: US vs Canada

| | US | Canada |
|:---:|:---:|:---:|
| Mean | 25.5 | 26.3 |
| Std Dev | 3.8 | 3.2 |
| n | 108 | 102 |

Test $H_0: \mu_{US} = \mu_{CA}$ vs $H_1: \mu_{US} \neq \mu_{CA}$ at $\alpha = 0.05$.

```python
X_1_bar, X_2_bar = 25.5, 26.3
s_1, s_2 = 3.8, 3.2
n_1, n_2 = 108, 102

s_p_square = ((n_1-1)*s_1**2 + (n_2-1)*s_2**2) / (n_1 + n_2 - 2)
statistic = (X_1_bar - X_2_bar) / np.sqrt(s_p_square/n_1 + s_p_square/n_2)
df = n_1 + n_2 - 2
p_value = 2 * stats.t(df).cdf(-abs(statistic))

print(f"{statistic = :.4f}, {p_value = :.4f}")
```

## 18. Mean Driving Distance: Electric Car Model A vs B

| | Model A | Model B |
|:---:|:---:|:---:|
| Mean | 168km | 172km |
| Std Dev | 5.4km | 7.5km |
| n | 5 | 5 |

Test $H_0: \mu_A = \mu_B$ vs $H_1: \mu_A \neq \mu_B$ at $\alpha = 0.05$.

```python
X_1_bar, X_2_bar = 168, 172
s_1, s_2 = 5.4, 7.5
n_1, n_2 = 5, 5

statistic = (X_1_bar - X_2_bar) / np.sqrt(s_1**2/n_1 + s_2**2/n_2)
top = (s_1**2/n_1 + s_2**2/n_2)**2
bottom = (s_1**2/n_1)**2/n_1 + (s_2**2/n_2)**2/n_2
df = top / bottom
p_value = 2 * stats.t(df).cdf(-abs(statistic))

print(f"{statistic = :.4f}, {p_value = :.4f}")
```

## 19. Mean School Years: Italy vs France

| | Italy | France |
|:---:|:---:|:---:|
| Mean | 10.7 | 10.4 |
| Std Dev | 2.3 | 2.5 |
| n | 46 | 58 |

Test $H_0: \mu_I = \mu_F$ vs $H_1: \mu_I \neq \mu_F$ at $\alpha = 0.05$ using pooled variance.

```python
X_1_bar, X_2_bar = 10.7, 10.4
s_1, s_2 = 2.3, 2.5
n_1, n_2 = 46, 58

s_p_square = ((n_1-1)*s_1**2 + (n_2-1)*s_2**2) / (n_1 + n_2 - 2)
statistic = (X_1_bar - X_2_bar) / np.sqrt(s_p_square/n_1 + s_p_square/n_2)
df = n_1 + n_2 - 2
p_value = 2 * stats.t(df).cdf(-abs(statistic))

print(f"{statistic = :.4f}, {p_value = :.4f}")
```

## 20. Mean Salaries of Two Departments (Welch's t Test)

Dept A: $\bar{X}_1 = 60{,}000$, $s_1 = 8{,}000$, $n_1 = 12$. Dept B: $\bar{X}_2 = 65{,}000$, $s_2 = 10{,}000$, $n_2 = 15$. Unequal variances. Test at $\alpha = 0.05$.

**Solution:**

$$t = \frac{60{,}000 - 65{,}000}{\sqrt{\frac{8{,}000^2}{12} + \frac{10{,}000^2}{15}}} = \frac{-5{,}000}{3{,}464.1} \approx -1.443$$

With $df \approx 24$ and $t_{0.025, 24} \approx \pm 2.064$: since $|-1.443| < 2.064$, **fail to reject** $H_0$.

## 21. Difference in Proportions of Smoking Cessation

Treatment A: 60/200 quit. Treatment B: 54/180 quit. Test at $\alpha = 0.05$.

**Solution:**

$$\hat{p}_1 = 0.30, \quad \hat{p}_2 = 0.30, \quad \hat{p}_{\text{pool}} = 114/380 = 0.30$$

$$z = \frac{0.30 - 0.30}{\sqrt{0.30 \times 0.70 \times (1/200 + 1/180)}} = 0$$

Since $|0| < 1.96$, **fail to reject** $H_0$. No significant difference.

## Paired Sample Exercises

### Effectiveness of Workout Program

| Participant | Before (%) | After (%) |
|---|---|---|
| 1 | 25 | 23 |
| 2 | 28 | 26 |
| 3 | 30 | 28 |
| 4 | 27 | 26 |
| 5 | 32 | 30 |
| 6 | 29 | 27 |
| 7 | 26 | 25 |
| 8 | 31 | 29 |
| 9 | 24 | 23 |
| 10 | 30 | 28 |

**(1)** Data type: **(c) Paired Sample** (same participants measured before and after).

**(2)** Sampling distribution: **(b) $t_9$** (paired differences with $n=10$, $df = 9$).

**Full test:** $\bar{d} = 1.7$, $s_d \approx 0.483$, $t = 1.7 / (0.483/\sqrt{10}) \approx 11.11$. Critical value $t_{0.05,9} \approx 1.833$. Since $11.11 > 1.833$, **reject** $H_0$. The workout program significantly reduces body fat.

### Dietary Plan and Cholesterol Levels

12 participants, before and after a 6-week program. $\bar{d} = 7.5$, $s_d \approx 2.61$.

$$t = \frac{7.5}{2.61/\sqrt{12}} = \frac{7.5}{0.754} \approx 9.95$$

With $df = 11$ and $t_{0.05, 11} \approx 1.796$: since $9.95 > 1.796$, **reject** $H_0$. The dietary plan significantly reduces cholesterol.
