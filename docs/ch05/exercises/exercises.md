# Exercises

## Exercise 1: Capture-Recapture Method

We aim to estimate the total number of fish, $N$, in a pond using the **capture-recapture method**.

1. **First Capture**: Researchers captured $M = 50$ fish, marked them, and released them back into the pond.
2. **Second Capture**: A week later, they captured $n = 40$ fish. Among these, $m = 10$ fish were found to be marked.

**(a)** Derive the probability $P(N)$ of observing the given outcome as a function of $N$.

**(b)** Determine the maximum likelihood estimate (MLE) of $N$.

### Solution

**Part (a): Deriving the Probability $P(N)$**

The number of marked fish in the second sample follows a **hypergeometric distribution**:

$$
P(N) = \frac{\binom{M}{m} \binom{N-M}{n-m}}{\binom{N}{n}}
$$

Substituting the given values ($M = 50$, $n = 40$, $m = 10$):

$$
P(N) = \frac{\binom{50}{10} \binom{N-50}{30}}{\binom{N}{40}}
$$

**Part (b): Finding the MLE of $N$**

The log-likelihood is:

$$
\ln P(N) = \ln \binom{50}{10} + \ln \binom{N-50}{30} - \ln \binom{N}{40}
$$

The term $\ln \binom{50}{10}$ is constant. Expanding the remaining terms and differentiating:

$$
\frac{d}{dN} \ln P(N) = 0 \quad \Rightarrow \quad \hat{N} = \frac{M \cdot n}{m} = \frac{50 \cdot 40}{10} = 200
$$

---

## Exercise 2: Estimation of Probability of Heads

A coin with probability $p$ of landing heads is tossed 100 times. The coin lands on heads 40 times.

**(a)** Express the likelihood function $P(p)$.

**(b)** Find the value of $p$ that maximizes $P(p)$.

### Solution

**Part (a):** The likelihood (ignoring $\binom{100}{40}$):

$$
P(p) = p^{40}(1-p)^{60}
$$

**Part (b):** The log-likelihood:

$$
\ln P(p) = 40 \ln p + 60 \ln(1-p)
$$

Derivative:

$$
\frac{d}{dp} \ln P(p) = \frac{40}{p} - \frac{60}{1-p} = 0
$$

$$
40(1-p) = 60p \quad \Rightarrow \quad \hat{p} = 0.4
$$

---

## Exercise 3: Mean of the Sample Mean

If a random sample of 9 is taken from a population with mean 75 and standard deviation 18, what is the mean of the sampling distribution of the sample mean?

### Solution

The mean of the sampling distribution is the same as the population mean: **75**.

---

## Exercise 4: Standard Error of the Sample Mean

A random sample of 9 observations is taken from a population with a mean of 75 and a standard deviation of 18. What is the standard error?

### Solution

$$
\sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}} = \frac{18}{\sqrt{9}} = \frac{18}{3} = 6
$$

---

## Exercise 5: Probability of Sample Mean Exceeding a Given Weight

The weights of apples in a large orchard follow a normal distribution with a mean of 150 grams and a standard deviation of 20 grams. A random sample of 25 apples is selected. What is the probability that the sample mean weight exceeds 155 grams?

### Solution

$$
\sigma_{\bar{X}} = \frac{20}{\sqrt{25}} = 4
$$

$$
Z = \frac{155 - 150}{4} = 1.25
$$

$$
P(\bar{X} > 155) = 1 - \Phi(1.25) = 1 - 0.8944 = 0.1056
$$

The probability is approximately **10.56%**.

$$
\text{prob} = 1 - F(1.25)
$$

```python
from scipy import stats

print(f"{stats.norm().sf(1.25) = :.4f}")
```

---

## Exercise 6: Probability of Sample Mean Falling Within a Range

The average number of hours a student sleeps per night is 7 hours, with a standard deviation of 1.5 hours. A random sample of 49 students is selected. What is the probability that the sample mean sleep duration is between 6.8 and 7.2 hours?

### Solution

$$
\sigma_{\bar{X}} = \frac{1.5}{\sqrt{49}} = \frac{1.5}{7} = 0.2143
$$

$$
Z_1 = \frac{6.8 - 7}{0.2143} \approx -0.93, \qquad
Z_2 = \frac{7.2 - 7}{0.2143} \approx 0.93
$$

$$
P(6.8 < \bar{X} < 7.2) = \Phi(0.93) - \Phi(-0.93) = 0.8238 - 0.1762 = 0.6476
$$

The probability is approximately **64.76%**.

$$
\text{prob} = F(0.93) - F(-0.93)
$$

```python
from scipy import stats

print(f"{stats.norm().cdf(0.93) - stats.norm().cdf(-0.93) = :.4f}")
```

---

## Exercise 7: Probability of Sample Mean Weight Exceeding 72 kg

The weights in a population are normally distributed with a mean of 70 kg and a standard deviation of 10 kg. If a random sample of 5 individuals is selected, what is the probability that the sample mean weight will be greater than 72 kg?

### Solution

Since the population weight is normally distributed, we can use the exact result even for $n = 5$.

$$
\sigma_{\bar{X}} = \frac{10}{\sqrt{5}} \approx 4.47
$$

$$
Z = \frac{72 - 70}{4.47} \approx 0.447
$$

$$
P(\bar{X} > 72) = 1 - \Phi(0.447) \approx 1 - 0.6726 = 0.3274
$$

The probability is approximately **32.74%**.

$$
\text{prob} = 1 - F(0.447)
$$

```python
from scipy import stats

print(f"{stats.norm().sf(0.447) = :.4f}")
```

---

## Exercise 8: Uncertainty of Sample Mean Exceeding 810 Hours Without Normality Assumption

The average lifespan of a certain type of lightbulb is 800 hours, with a standard deviation of 100 hours. For a sample of 5 lightbulbs, what can be said about the probability that the sample mean lifespan will be greater than 810 hours, without assuming that the population distribution is normal?

### Solution

$$
\sigma_{\bar{X}} = \frac{100}{\sqrt{5}} \approx 44.72
$$

However, since the sample size is small ($n = 5$) and we do not assume that the population distribution is normal, the Central Limit Theorem (CLT) does not apply. Without assuming normality, **we cannot accurately determine the probability** that the sample mean will be greater than 810 hours. Further information about the population's shape would be needed.

---

## Exercise 9: Probability of Sample Proportion Exceeding 65% Preference

In a large population, 60% prefer brand A over brand B. If a sample of 100 individuals is selected, what is the probability that more than 65% of the sample will prefer brand A?

### Solution

$$
\sigma_{\hat{p}} = \sqrt{\frac{0.60 \times 0.40}{100}} = \sqrt{0.0024} \approx 0.049
$$

$$
Z = \frac{0.65 - 0.60}{0.049} \approx 1.02
$$

$$
P(\hat{p} > 0.65) = 1 - \Phi(1.02) = 1 - 0.8461 = 0.1539
$$

The probability is approximately **15.39%**.

```python
from scipy import stats

print(f"{stats.norm().sf(1.02) = :.4f}")
```

---

## Exercise 10: Comparing Exact and Normal Approximation for Sample Proportion

A survey indicates that 30% of a town's population prefers public transport. If a sample of 10 residents is selected, what can be said about the probability that more than 35% of the sample will prefer public transport?

### Solution

**Exact Binomial Calculation**

Since $\hat{p} > 0.35$ with $n = 10$ means $X \geq 4$ (where $X \sim \text{Binomial}(10, 0.3)$):

$$
P(X = 0) \approx 0.0282, \quad P(X = 1) \approx 0.1211, \quad P(X = 2) \approx 0.2335, \quad P(X = 3) \approx 0.2668
$$

$$
P(X < 4) = 0.0282 + 0.1211 + 0.2335 + 0.2668 = 0.6496
$$

$$
P(X \geq 4) = 1 - 0.6496 = 0.3504
$$

**Normal Approximation (questionable since $np = 3 < 5$)**

$$
\sigma_{\hat{p}} = \sqrt{\frac{0.30 \times 0.70}{10}} \approx 0.1449
$$

$$
Z = \frac{0.35 - 0.30}{0.1449} \approx 0.345
$$

$$
P(\hat{p} > 0.35) = 1 - \Phi(0.345) \approx 0.3650
$$

**Comparison:**

| Method | Result |
|--------|--------|
| Exact binomial | 0.3504 |
| Normal approximation | 0.3650 |

The normal approximation is reasonably close despite the small sample.

```python
from scipy import stats

print(f"{stats.norm().cdf(0.345) = :.4f}")
print(f"{stats.norm().sf(0.345) = :.4f}")
```

---

## Exercise 11: Distribution of the Sample Variance

A sample of size 10 is drawn from a normal distribution with a known variance of 25. What is the expected value and variance of the sample variance?

Use the fact that for a chi-square distribution with $k$ degrees of freedom: mean is $\mu = k$ and variance is $\sigma^2 = 2k$.

### Solution

For $Y \sim \chi^2_{n-1}$, we have $EY = n - 1$ and $\text{Var}(Y) = 2(n-1)$.

$$
E\frac{(n-1)S^2}{\sigma^2} = n - 1
\quad\Rightarrow\quad
ES^2 = \sigma^2 = 25
$$

$$
\text{Var}\!\left(\frac{(n-1)S^2}{\sigma^2}\right) = 2(n-1)
\quad\Rightarrow\quad
\text{Var}(S^2) = \frac{2\sigma^4}{n-1} = \frac{2(25^2)}{9} = \frac{1250}{9} \approx 138.89
$$

---

## Exercise 12: Probability of Sample Variance Exceeding 30

A sample of size 10 is drawn from a normal distribution with a known population variance of 25. What is the probability that the sample variance will be greater than 30?

### Solution

$$
\chi^2 = \frac{(n-1)s^2}{\sigma^2} = \frac{9 \times 30}{25} = 10.8
$$

$$
P(S^2 > 30) = P(\chi^2_9 > 10.8) \approx 0.2897
$$

The probability is approximately **28.97%**.

$$
\text{prob} = 1 - F(10.8)
$$

```python
from scipy import stats

print(f"{stats.chi2(9).sf(10.8) = :.4f}")
```

---

## Exercise 13: Assessing Probability of Sample Variance Without Normality Assumption

A sample of size 10 is drawn from a population with a known variance of 25. What can be said about the probability that the sample variance will be greater than 30, without assuming that the population distribution is normal?

### Solution

Without normality, $(n-1)S^2/\sigma^2$ does **not** follow a chi-square distribution. We know $E(S^2) = 25$, but we cannot determine $P(S^2 > 30)$.

Chebyshev's Inequality could provide a bound:

$$
P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}
$$

However, to use it we need $\text{Var}(S^2)$, which depends on higher moments of the non-normal population that we don't have.

**Conclusion:** Without assuming normality, we cannot determine the probability using typical methods.

---

## Exercise 14: Standard Error of the Difference in Sample Means

We draw two independent samples: population A with $\sigma_A = 15$ ($n_A = 36$) and population B with $\sigma_B = 20$ ($n_B = 49$). Find the standard error of $\bar{X}_A - \bar{X}_B$.

### Solution

$$
\sigma_{\bar{X}_A - \bar{X}_B} = \sqrt{\frac{15^2}{36} + \frac{20^2}{49}} = \sqrt{6.25 + 8.16} \approx 3.80
$$

---

## Exercise 15: Probability of Sample Mean Exceeding \$45,000 Using $t$-Distribution

The annual salaries of employees in a company are normally distributed with a mean of \$40,000. A random sample of 9 employees is selected, and the sample standard deviation is \$8,000. What is the probability that the sample mean annual salary will be \$45,000 or more?

### Solution

$$
\sigma_{\bar{X}} = \frac{s}{\sqrt{n}} = \frac{8{,}000}{\sqrt{9}} = \frac{8{,}000}{3} \approx 2{,}666.67
$$

$$
t = \frac{45{,}000 - 40{,}000}{2{,}666.67} \approx 1.875
$$

With $\text{df} = 8$:

$$
P(T_8 \geq 1.875) \approx 0.048
$$

The probability is approximately **4.8%**.

```python
from scipy import stats

print(f"{stats.t(8).sf(1.875) = :.4f}")
```

---

## Exercise 16: Probability of $F$-Ratio Exceeding 1.5

Two independent samples from two normal populations with common variance $\sigma^2 = 20$. First sample: $n_1 = 15$. Second sample: $n_2 = 10$. What is $P(S_1^2 / S_2^2 > 1.5)$?

### Solution

Under equal variances, $F = S_1^2 / S_2^2 \sim F_{n_1-1, \, n_2-1} = F_{14, 9}$.

Starting from $S_i^2 \sim \frac{\sigma^2 \chi^2_{n_i-1}}{n_i - 1}$:

$$
F = \frac{S_1^2 / \sigma^2}{S_2^2 / \sigma^2} = \frac{\chi^2_{n_1-1}/(n_1-1)}{\chi^2_{n_2-1}/(n_2-1)} \sim F_{n_1-1, n_2-1}
$$

$$
P(F_{14,9} > 1.5) \approx 0.274
$$

The probability is approximately **27.4%**.

```python
from scipy import stats

n1 = 15
n2 = 10
print(f"{stats.f(n1-1, n2-1).sf(1.5) = :.4f}")
```

---

## Exercise 17: Probability of Sample Mean Between 48 and 52

A population has a mean of 50 and a standard deviation of 12. If a sample of size 36 is drawn, what is the probability that the sample mean will be between 48 and 52?

### Solution

$$
\sigma_{\bar{X}} = \frac{12}{\sqrt{36}} = 2
$$

$$
Z_1 = \frac{48 - 50}{2} = -1, \qquad Z_2 = \frac{52 - 50}{2} = 1
$$

$$
P(48 < \bar{X} < 52) = \Phi(1) - \Phi(-1) = 0.6827
$$

The probability is approximately **68.27%**.

```python
from scipy import stats

print(f"{stats.norm().cdf(1) - stats.norm().cdf(-1) = :.4f}")
```

---

## Exercise 18: The Law of Large Numbers

A population has a mean of 100 with a standard deviation of 20. If you take increasingly larger samples, what happens to the mean of the sampling distribution?

### Solution

According to the Law of Large Numbers, as the sample size increases, the mean of the sampling distribution approaches the population mean of **100**.

---

## Exercise 19: Shape of Sampling Distribution for a Skewed Population

A population has a skewed distribution with a mean of 50 and a standard deviation of 10. If a sample of size 100 is drawn, what is the approximate shape of the sampling distribution of the sample mean?

### Solution

By the Central Limit Theorem (CLT), since $n = 100$ is large (greater than 30), the sampling distribution of the sample mean will be approximately **normal**, even though the original population distribution is skewed.

---

## Exercise 20: Probability of Sample Mean Exceeding \$2100 for Skewed Sales

A company's daily sales are skewed to the right with a mean of \$2000 and a standard deviation of \$500. If a random sample of 100 days is selected, what is the probability that the sample mean daily sales will exceed \$2100?

### Solution

$$
\sigma_{\bar{X}} = \frac{500}{\sqrt{100}} = 50
$$

$$
Z = \frac{2100 - 2000}{50} = 2
$$

$$
P(\bar{X} > 2100) = P(Z > 2) = 1 - 0.9772 = 0.0228
$$

The probability is approximately **2.28%**.

```python
from scipy import stats

print(f"{stats.norm().sf(2) = :.4f}")
```

---

## Exercise 21: Effect of Increasing Sample Size on Standard Error

If the standard deviation of a population is 50 and the sample size is increased from 25 to 100, what happens to the standard error?

### Solution

$$
\text{SE}_{n=25} = \frac{50}{\sqrt{25}} = 10
$$

$$
\text{SE}_{n=100} = \frac{50}{\sqrt{100}} = 5
$$

The standard error is **reduced by half** when the sample size increases from 25 to 100.

---

## Exercise 22: Standard Error and Sample Size

If the standard error of the mean for a sample of size 16 is 5, what would be the standard error if the sample size increased to 64?

### Solution

From $\sigma / \sqrt{16} = 5$, we get $\sigma = 20$. For a sample size of 64:

$$
\sigma_{\bar{X}} = \frac{20}{\sqrt{64}} = 2.5
$$

---

## Exercise 23: Expected Value and Variance of the Sample Variance (Repeat for Emphasis)

A sample of size 10 is drawn from a normal distribution with a known variance of 25. What is the expected value and variance of the sample variance?

Use the fact that for a chi-square distribution with $k$ degrees of freedom: mean is $\mu = k$ and variance is $\sigma^2 = 2k$.

### Solution

$$
E\frac{(n-1)S^2}{\sigma^2} = n - 1
\quad\Rightarrow\quad
ES^2 = \sigma^2 = 25
$$

$$
\text{Var}\!\left(\frac{(n-1)S^2}{\sigma^2}\right) = 2(n-1)
\quad\Rightarrow\quad
\text{Var}(S^2) = \frac{2\sigma^4}{n-1} = \frac{2(25^2)}{9} = \frac{1250}{9} \approx 138.89
$$
