# Chapter 8: Exercises

## Exercise 1: Interpreting Confidence Intervals

A researcher computes a 95% CI for the mean cholesterol level and obtains $(188.3, 205.7)$.

**(a)** Which of the following interpretations is correct?

- (i) "There is a 95% probability that the population mean is between 188.3 and 205.7."
- (ii) "If we repeated this study many times, about 95% of the resulting intervals would contain the true mean."
- (iii) "95% of all cholesterol levels fall between 188.3 and 205.7."

**(b)** What is the point estimate $\bar{X}$ and margin of error?

??? note "Solution"

    **(a)** Only (ii) is correct. Statement (i) incorrectly assigns probability to the fixed parameter. Statement (iii) confuses a CI for the mean with the range of data.

    **(b)** $\bar{X} = (188.3 + 205.7)/2 = 197.0$, margin of error $= (205.7 - 188.3)/2 = 8.7$.

---

## Exercise 2: CI for the Mean

A random sample of $n = 25$ light bulbs has mean lifetime $\bar{X} = 1200$ hours and standard deviation $s = 150$ hours.

**(a)** Construct a 95% CI for the population mean lifetime.

**(b)** Would a 99% CI be wider or narrower? Compute it.

**(c)** How large a sample would be needed to achieve a margin of error of 20 hours at 95% confidence?

??? note "Solution"

    **(a)** $t_{0.025, 24} = 2.064$. CI: $1200 \pm 2.064 \times 150/\sqrt{25} = 1200 \pm 61.9 = (1138.1, 1261.9)$

    **(b)** Wider. $t_{0.005, 24} = 2.797$. CI: $1200 \pm 2.797 \times 30 = 1200 \pm 83.9 = (1116.1, 1283.9)$

    **(c)** Using z-approximation: $n = (1.96 \times 150 / 20)^2 = (14.7)^2 = 216.09 \implies n = 217$

---

## Exercise 3: CI for a Proportion

In a poll of 500 voters, 280 support a ballot measure.

**(a)** Compute the Wald 95% CI for the true proportion of support.

**(b)** Compute the Wilson 95% CI. Compare with the Wald interval.

**(c)** Is there evidence that more than half the voters support the measure?

??? note "Solution"

    **(a)** $\hat{p} = 280/500 = 0.56$. $\text{ME} = 1.96\sqrt{0.56 \times 0.44/500} = 1.96 \times 0.0222 = 0.0435$

    Wald CI: $(0.517, 0.604)$

    **(b)** Wilson CI: center $= (0.56 + 1.96^2/(2\times500))/(1 + 1.96^2/500) = 0.5596/1.00768 = 0.5554$

    Wilson CI $\approx (0.517, 0.603)$. Very similar here since $n$ is large and $\hat{p}$ is not extreme.

    **(c)** Since the entire 95% CI is above 0.5, we have evidence at the 5% level that more than half support the measure.

---

## Exercise 4: CI for Variance

A manufacturer measures the diameter of 15 ball bearings (in mm): the sample variance is $s^2 = 0.0025$.

**(a)** Construct a 95% CI for $\sigma^2$.

**(b)** Construct a 95% CI for $\sigma$.

??? note "Solution"

    **(a)** $\chi^2_{0.025, 14} = 26.119$, $\chi^2_{0.975, 14} = 5.629$

    CI for $\sigma^2$: $\left(\frac{14 \times 0.0025}{26.119}, \frac{14 \times 0.0025}{5.629}\right) = (0.00134, 0.00622)$

    **(b)** CI for $\sigma$: $(\sqrt{0.00134}, \sqrt{0.00622}) = (0.0366, 0.0789)$ mm

---

## Exercise 5: Two-Sample CI

Two teaching methods are compared. Method A: $n_1 = 30$, $\bar{X}_1 = 78$, $s_1 = 8$. Method B: $n_2 = 35$, $\bar{X}_2 = 82$, $s_2 = 10$.

**(a)** Construct a 95% CI for $\mu_A - \mu_B$ using Welch's method.

**(b)** Does the interval contain 0? What does this imply?

??? note "Solution"

    **(a)** $\text{SE} = \sqrt{64/30 + 100/35} = \sqrt{2.133 + 2.857} = \sqrt{4.990} = 2.234$

    Welch df $= \frac{(2.133 + 2.857)^2}{2.133^2/29 + 2.857^2/34} = \frac{24.90}{0.157 + 0.240} = 62.7$

    $t_{0.025, 62} \approx 2.000$

    CI: $(78 - 82) \pm 2.000 \times 2.234 = -4 \pm 4.47 = (-8.47, 0.47)$

    **(b)** The interval contains 0, so we cannot conclude a statistically significant difference at the 5% level. However, the interval is mostly negative, suggesting Method B may be slightly better.

---

## Exercise 6: Paired CI

Ten students take a pre-test and post-test after a training program. The differences (post − pre) are: $5, 3, 8, 2, 7, 1, 6, 4, 9, 5$.

**(a)** Construct a 95% CI for the mean improvement $\mu_D$.

**(b)** Is the training program effective at the 5% level?

??? note "Solution"

    **(a)** $\bar{D} = 5.0$, $s_D = 2.582$, $n = 10$

    $t_{0.025, 9} = 2.262$

    CI: $5.0 \pm 2.262 \times 2.582/\sqrt{10} = 5.0 \pm 1.85 = (3.15, 6.85)$

    **(b)** Since the entire CI is positive (does not contain 0), the improvement is statistically significant at the 5% level. The training program appears effective.
