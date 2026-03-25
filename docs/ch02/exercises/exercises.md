# Exercises

These exercises test your understanding of descriptive statistics concepts from Chapter 2, including exploratory data analysis, measures of center and spread, distributional shape, and visualization techniques. Work through each problem carefully and compare your reasoning with the provided solutions.

---

## Exercise 1: Histogram Bin Width

A researcher collects the following 20 exam scores:

$$
55, 62, 67, 70, 71, 73, 74, 75, 76, 78, 80, 81, 83, 85, 87, 88, 90, 92, 95, 98
$$

**(a)** Using Sturges' rule, how many bins should a histogram of these data have?

**(b)** If you instead use 4 bins of equal width spanning the range of the data, specify the bin edges and count the number of observations in each bin.

**(c)** Explain why very few bins can hide important features of the distribution, while too many bins can introduce spurious features.

??? success "Solution"

    **(a)** Sturges' rule gives the number of bins as:

    $$
    k = \lceil \log_2 n \rceil + 1
    $$

    With $n = 20$:

    $$
    k = \lceil \log_2 20 \rceil + 1 = \lceil 4.32 \rceil + 1 = 5 + 1 = 6
    $$

    Sturges' rule recommends 6 bins.

    **(b)** The range is $98 - 55 = 43$. With 4 bins the width is $43 / 4 = 10.75$. Rounding up to a convenient width of 11:

    - Bin 1: $[55, 66)$ — 2 observations (55, 62)
    - Bin 2: $[66, 77)$ — 8 observations (67, 70, 71, 73, 74, 75, 76)
    - Bin 3: $[77, 88)$ — 6 observations (78, 80, 81, 83, 85, 87)
    - Bin 4: $[88, 99]$ — 4 observations (88, 90, 92, 95, 98)

    Note: the count in Bin 2 is 7 if 76 falls in $[66, 77)$, and the exact assignment depends on the convention for the boundary. With the left-closed convention above, 76 is in Bin 2 (7 values: 67, 70, 71, 73, 74, 75, 76) and Bin 3 has 5 values (78, 80, 81, 83, 85, 87). Recounting carefully with 4 equal-width bins of width 10.75:

    - $[55, 65.75)$: 2 values
    - $[65.75, 76.5)$: 7 values
    - $[76.5, 87.25)$: 6 values
    - $[87.25, 98]$: 5 values

    **(c)** Too few bins merge distinct features — a bimodal distribution might appear unimodal if both modes fall in the same bin. Too many bins produce noisy histograms where random sampling variation creates artificial peaks and valleys that do not reflect the true underlying distribution.

---

## Exercise 2: ECDF Interpretation

Consider the dataset $\{2, 5, 5, 7, 10\}$.

**(a)** Write the empirical CDF $\hat{F}(x)$ as a piecewise function.

**(b)** What is $\hat{F}(5)$? What is $\hat{F}(6)$?

**(c)** Using the ECDF, determine the 50th percentile (median) of this dataset.

??? success "Solution"

    **(a)** The ECDF is defined as $\hat{F}(x) = \frac{1}{n} \sum_{i=1}^n \mathbf{1}(X_i \le x)$. With $n = 5$:

    $$
    \hat{F}(x) = \begin{cases} 0 & x < 2 \\ 1/5 & 2 \le x < 5 \\ 3/5 & 5 \le x < 7 \\ 4/5 & 7 \le x < 10 \\ 1 & x \ge 10 \end{cases}
    $$

    Note that $\hat{F}$ jumps by $1/5$ at each unique value, and by $2/5$ at $x = 5$ because that value appears twice.

    **(b)** $\hat{F}(5) = 3/5 = 0.6$, since 3 of the 5 observations are $\le 5$. $\hat{F}(6) = 3/5 = 0.6$ as well, since no observations fall between 5 and 7.

    **(c)** The 50th percentile is the smallest $x$ such that $\hat{F}(x) \ge 0.5$. Since $\hat{F}(2) = 0.2 < 0.5$ and $\hat{F}(5) = 0.6 \ge 0.5$, the median is 5. This matches the middle value when the data are sorted: $2, 5, \mathbf{5}, 7, 10$.

---

## Exercise 3: Mean, Median, and Robustness

A small company has five employees with annual salaries (in thousands of dollars):

$$
35, \; 40, \; 42, \; 45, \; 250
$$

**(a)** Compute the sample mean and the sample median.

**(b)** The CEO's salary of \$250k is replaced by \$500k. Recompute the mean and median. Which measure changed more?

**(c)** Explain why the median is considered a robust measure of central tendency while the mean is not.

??? success "Solution"

    **(a)** The sample mean is:

    $$
    \bar{x} = \frac{35 + 40 + 42 + 45 + 250}{5} = \frac{412}{5} = 82.4
    $$

    To find the median, sort the data: $35, 40, 42, 45, 250$. The middle value is the 3rd observation, so the median is $42$.

    **(b)** With the CEO salary changed to 500:

    $$
    \bar{x}_{\text{new}} = \frac{35 + 40 + 42 + 45 + 500}{5} = \frac{662}{5} = 132.4
    $$

    The sorted data become $35, 40, 42, 45, 500$ and the median remains $42$.

    The mean increased by $132.4 - 82.4 = 50.0$ (a 60.7% increase), while the median did not change at all.

    **(c)** The median depends only on the rank ordering of observations, not their magnitudes. Changing a single extreme value does not affect the middle rank, so the median has a high breakdown point (approximately 50%). The mean, by contrast, uses every value in its computation, so a single extreme observation can shift the mean arbitrarily far from the center of the bulk of the data.

---

## Exercise 4: Variance and Standard Deviation

Suppose a dataset consists of the values $\{3, 7, 7, 9, 14\}$.

**(a)** Compute the sample mean $\bar{x}$.

**(b)** Compute the sample variance $s^2$ using Bessel's correction.

**(c)** Compute the sample standard deviation $s$.

**(d)** If every observation is increased by a constant $c = 10$, what happens to the mean, variance, and standard deviation?

??? success "Solution"

    **(a)**

    $$
    \bar{x} = \frac{3 + 7 + 7 + 9 + 14}{5} = \frac{40}{5} = 8
    $$

    **(b)** The sample variance with Bessel's correction divides by $n - 1$:

    $$
    s^2 = \frac{1}{n-1} \sum_{i=1}^{n}(x_i - \bar{x})^2
    $$

    Computing the squared deviations:

    | $x_i$ | $x_i - \bar{x}$ | $(x_i - \bar{x})^2$ |
    |:---:|:---:|:---:|
    | 3 | $-5$ | 25 |
    | 7 | $-1$ | 1 |
    | 7 | $-1$ | 1 |
    | 9 | 1 | 1 |
    | 14 | 6 | 36 |

    $$
    s^2 = \frac{25 + 1 + 1 + 1 + 36}{4} = \frac{64}{4} = 16
    $$

    **(c)** $s = \sqrt{16} = 4$.

    **(d)** Adding a constant $c$ to every observation shifts the mean by $c$ but does not change the variance or standard deviation. The new mean is $\bar{x} + c = 18$, the new variance is still $s^2 = 16$, and the new standard deviation is still $s = 4$. This follows because the deviations $(x_i + c) - (\bar{x} + c) = x_i - \bar{x}$ are unchanged.

---

## Exercise 5: IQR and Outlier Detection

The following dataset contains 12 measurements:

$$
4, \; 7, \; 8, \; 12, \; 14, \; 15, \; 16, \; 18, \; 19, \; 22, \; 25, \; 55
$$

**(a)** Find the first quartile $Q_1$, the median $Q_2$, and the third quartile $Q_3$.

**(b)** Compute the interquartile range (IQR).

**(c)** Using the $1.5 \times \text{IQR}$ rule, determine the lower and upper fences. Identify any outliers.

**(d)** How would this dataset appear in a boxplot? Describe the key features.

??? success "Solution"

    **(a)** With $n = 12$ observations already sorted, we split the data into a lower half (positions 1--6) and an upper half (positions 7--12).

    - $Q_1$ is the median of $\{4, 7, 8, 12, 14, 15\}$: $Q_1 = (8 + 12)/2 = 10$.
    - $Q_2$ is the median of the full dataset: $(15 + 16)/2 = 15.5$.
    - $Q_3$ is the median of $\{16, 18, 19, 22, 25, 55\}$: $Q_3 = (19 + 22)/2 = 20.5$.

    **(b)** The interquartile range is:

    $$
    \text{IQR} = Q_3 - Q_1 = 20.5 - 10 = 10.5
    $$

    **(c)** The fences are:

    $$
    \text{Lower fence} = Q_1 - 1.5 \times \text{IQR} = 10 - 15.75 = -5.75
    $$

    $$
    \text{Upper fence} = Q_3 + 1.5 \times \text{IQR} = 20.5 + 15.75 = 36.25
    $$

    Any observation below $-5.75$ or above $36.25$ is flagged as an outlier. The value $55$ exceeds the upper fence, so it is an outlier. No values fall below the lower fence.

    **(d)** The boxplot would show:

    - The box spans from $Q_1 = 10$ to $Q_3 = 20.5$, with a line at the median $Q_2 = 15.5$.
    - The lower whisker extends from $Q_1$ down to the smallest observation within the fences, which is 4.
    - The upper whisker extends from $Q_3$ up to the largest non-outlier, which is 25.
    - The value 55 would appear as an individual point (outlier marker) beyond the upper whisker.
    - The box would appear roughly symmetric around the median, but the outlier at 55 reveals right-skewed behavior in the tail.

---

## Exercise 6: Skewness Computation

Consider the dataset $\{1, 2, 3, 4, 10\}$.

**(a)** Compute the sample mean and sample standard deviation.

**(b)** Compute the sample skewness using the formula:

$$
g_1 = \frac{\frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^3}{\left(\frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2\right)^{3/2}}
$$

**(c)** Is this distribution left-skewed, symmetric, or right-skewed? Explain how the outlier at 10 drives the skewness value.

??? success "Solution"

    **(a)** The sample mean is:

    $$
    \bar{x} = \frac{1 + 2 + 3 + 4 + 10}{5} = \frac{20}{5} = 4
    $$

    The deviations and their squares:

    | $x_i$ | $x_i - \bar{x}$ | $(x_i - \bar{x})^2$ |
    |:---:|:---:|:---:|
    | 1 | $-3$ | 9 |
    | 2 | $-2$ | 4 |
    | 3 | $-1$ | 1 |
    | 4 | 0 | 0 |
    | 10 | 6 | 36 |

    $$
    \frac{1}{n}\sum(x_i - \bar{x})^2 = \frac{50}{5} = 10
    $$

    $$
    s_{\text{pop}} = \sqrt{10} \approx 3.162
    $$

    **(b)** The cubed deviations are:

    | $x_i$ | $(x_i - \bar{x})^3$ |
    |:---:|:---:|
    | 1 | $-27$ |
    | 2 | $-8$ |
    | 3 | $-1$ |
    | 4 | 0 |
    | 10 | 216 |

    $$
    \frac{1}{n}\sum(x_i - \bar{x})^3 = \frac{-27 - 8 - 1 + 0 + 216}{5} = \frac{180}{5} = 36
    $$

    $$
    g_1 = \frac{36}{10^{3/2}} = \frac{36}{31.623} \approx 1.138
    $$

    **(c)** Since $g_1 > 0$, the distribution is right-skewed. The single large value at 10 produces a cubed deviation of $+216$, which dominates the numerator. The four smaller values contribute cubed deviations that sum to only $-36$. This asymmetry — one long right tail — is exactly what positive skewness measures.

---

## Exercise 7: Comparing MAD and Standard Deviation

A quality control process records the following measurements of a component's diameter (in mm):

$$
10.1, \; 10.0, \; 9.9, \; 10.2, \; 10.0, \; 9.8, \; 10.1, \; 10.0, \; 15.3, \; 10.0
$$

**(a)** Compute the sample standard deviation $s$.

**(b)** Compute the median absolute deviation (MAD):

$$
\text{MAD} = \text{median}(|x_i - \tilde{x}|)
$$

where $\tilde{x}$ is the sample median.

**(c)** Compute the scaled MAD using the consistency constant $1.4826$, which makes it comparable to the standard deviation under normality. Compare the scaled MAD to the standard deviation and explain the difference.

??? success "Solution"

    **(a)** First, the mean:

    $$
    \bar{x} = \frac{10.1 + 10.0 + 9.9 + 10.2 + 10.0 + 9.8 + 10.1 + 10.0 + 15.3 + 10.0}{10} = \frac{105.4}{10} = 10.54
    $$

    The squared deviations sum to:

    $$
    \sum (x_i - \bar{x})^2 = 0.1936 + 0.2916 + 0.4096 + 0.1156 + 0.2916 + 0.5476 + 0.1936 + 0.2916 + 22.6576 + 0.2916 = 25.284
    $$

    $$
    s^2 = \frac{25.284}{9} = 2.809, \quad s = \sqrt{2.809} \approx 1.676
    $$

    **(b)** Sorted data: $9.8, 9.9, 10.0, 10.0, 10.0, 10.0, 10.1, 10.1, 10.2, 15.3$. The median of 10 values is the average of the 5th and 6th values: $\tilde{x} = (10.0 + 10.0)/2 = 10.0$.

    Absolute deviations from the median: $0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.2, 5.3$.

    Sorted absolute deviations: $0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.2, 0.2, 5.3$.

    $$
    \text{MAD} = \frac{0.1 + 0.1}{2} = 0.1
    $$

    **(c)** The scaled MAD is $1.4826 \times 0.1 = 0.14826$.

    The standard deviation ($s \approx 1.676$) is more than 11 times larger than the scaled MAD ($0.148$). This enormous discrepancy arises because the outlier at 15.3 inflates the standard deviation dramatically — the squared deviation $(15.3 - 10.54)^2 = 22.66$ accounts for about 90% of the total sum of squares. The MAD, by contrast, is virtually unaffected by this single extreme value because it depends on the median of the absolute deviations, not their sum. This example illustrates why MAD is preferred as a measure of spread when outliers may be present.

---

## Exercise 8: Kurtosis and Tail Behavior

Consider two distributions:

- **Dataset A**: $\{4, 5, 5, 6, 6, 6, 7, 7, 8\}$ (concentrated near center)
- **Dataset B**: $\{1, 2, 5, 6, 6, 6, 7, 10, 11\}$ (heavier tails)

**(a)** Verify that both datasets have the same mean.

**(b)** Compute the sample excess kurtosis for each dataset using:

$$
g_2 = \frac{\frac{1}{n}\sum_{i=1}^n(x_i - \bar{x})^4}{\left(\frac{1}{n}\sum_{i=1}^n(x_i - \bar{x})^2\right)^2} - 3
$$

**(c)** Which dataset has heavier tails? How does the excess kurtosis capture this?

??? success "Solution"

    **(a)** Both datasets have 9 observations.

    Dataset A: $\bar{x}_A = (4+5+5+6+6+6+7+7+8)/9 = 54/9 = 6$.

    Dataset B: $\bar{x}_B = (1+2+5+6+6+6+7+10+11)/9 = 54/9 = 6$.

    Both means equal 6.

    **(b)** **Dataset A** (deviations from 6):

    | $x_i$ | $x_i - 6$ | $(x_i - 6)^2$ | $(x_i - 6)^4$ |
    |:---:|:---:|:---:|:---:|
    | 4 | $-2$ | 4 | 16 |
    | 5 | $-1$ | 1 | 1 |
    | 5 | $-1$ | 1 | 1 |
    | 6 | 0 | 0 | 0 |
    | 6 | 0 | 0 | 0 |
    | 6 | 0 | 0 | 0 |
    | 7 | 1 | 1 | 1 |
    | 7 | 1 | 1 | 1 |
    | 8 | 2 | 4 | 16 |

    $$
    m_2^A = \frac{12}{9} = \frac{4}{3}, \quad m_4^A = \frac{36}{9} = 4
    $$

    $$
    g_2^A = \frac{4}{(4/3)^2} - 3 = \frac{4}{16/9} - 3 = \frac{36}{16} - 3 = 2.25 - 3 = -0.75
    $$

    **Dataset B** (deviations from 6):

    | $x_i$ | $x_i - 6$ | $(x_i - 6)^2$ | $(x_i - 6)^4$ |
    |:---:|:---:|:---:|:---:|
    | 1 | $-5$ | 25 | 625 |
    | 2 | $-4$ | 16 | 256 |
    | 5 | $-1$ | 1 | 1 |
    | 6 | 0 | 0 | 0 |
    | 6 | 0 | 0 | 0 |
    | 6 | 0 | 0 | 0 |
    | 7 | 1 | 1 | 1 |
    | 10 | 4 | 16 | 256 |
    | 11 | 5 | 25 | 625 |

    $$
    m_2^B = \frac{84}{9} = \frac{28}{3}, \quad m_4^B = \frac{1764}{9} = 196
    $$

    $$
    g_2^B = \frac{196}{(28/3)^2} - 3 = \frac{196}{784/9} - 3 = \frac{196 \times 9}{784} - 3 = \frac{1764}{784} - 3 = 2.25 - 3 = -0.75
    $$

    Interestingly, both datasets yield the same excess kurtosis of $-0.75$, meaning both are platykurtic (lighter-tailed than a normal distribution in the kurtosis sense). Despite Dataset B having more extreme values, its variance is also proportionally larger, and the ratio of fourth to squared-second moments turns out identical.

    **(c)** Although Dataset B has observations farther from the mean (values 1 and 11 are 5 units away versus 2 units for Dataset A), the excess kurtosis is the same because kurtosis measures tail heaviness relative to the distribution's own variance. Dataset B's larger variance "accounts for" the more extreme values. This illustrates that kurtosis is not simply about range or extremes — it captures the shape of the distribution relative to its own spread. A distribution with heavy tails relative to its variance (like a $t$-distribution with few degrees of freedom) would show positive excess kurtosis.

---

## Exercise 9: Boxplot versus Violin Plot

A biology experiment measures the growth (in cm) of plants under two treatments:

- **Treatment 1**: $\{5, 6, 6, 7, 7, 7, 8, 8, 9\}$
- **Treatment 2**: $\{3, 5, 7, 7, 7, 7, 7, 9, 11\}$

**(a)** Compute the five-number summary (minimum, $Q_1$, median, $Q_3$, maximum) for each treatment.

**(b)** Based on the five-number summaries alone, would boxplots for the two treatments look similar or different? Explain.

**(c)** Describe how the violin plots would differ from the boxplots for these two datasets, and what additional information the violin plots reveal.

??? success "Solution"

    **(a)** Both datasets have 9 observations.

    **Treatment 1** (sorted: 5, 6, 6, 7, 7, 7, 8, 8, 9):

    - Minimum: 5
    - $Q_1$: median of $\{5, 6, 6, 7\}$ = $(6+6)/2 = 6$
    - Median: 7 (5th value)
    - $Q_3$: median of $\{7, 8, 8, 9\}$ = $(8+8)/2 = 8$
    - Maximum: 9

    **Treatment 2** (sorted: 3, 5, 7, 7, 7, 7, 7, 9, 11):

    - Minimum: 3
    - $Q_1$: median of $\{3, 5, 7, 7\}$ = $(5+7)/2 = 6$
    - Median: 7 (5th value)
    - $Q_3$: median of $\{7, 7, 9, 11\}$ = $(7+9)/2 = 8$
    - Maximum: 11

    **(b)** The five-number summaries are almost identical: both have $Q_1 = 6$, median $= 7$, $Q_3 = 8$. The boxes in the boxplots would be the same. The only visible difference would be in the whisker lengths: Treatment 2 has a wider range ($3$ to $11$) compared to Treatment 1 ($5$ to $9$). The two boxplots would look quite similar, with Treatment 2 having slightly longer whiskers.

    **(c)** Violin plots would reveal a crucial difference that boxplots hide:

    - **Treatment 1** has a roughly bell-shaped density — values are spread fairly evenly around the median with a gradual taper in both tails.
    - **Treatment 2** has a sharp spike at 7 (five of nine values equal 7), giving it a strongly peaked, leptokurtic shape. The density would show a pronounced narrow peak at 7 with thin tails extending to 3 and 11.

    This difference in modality and concentration is invisible in the boxplots because the five-number summary cannot distinguish between a smooth distribution and one with a sharp peak. Violin plots, by showing the full kernel density estimate, reveal whether data are uniformly spread within the box or concentrated at specific values.

---

## Exercise 10: Weighted Mean and Grouped Data

A class of 30 students takes an exam. The score distribution is summarized in a frequency table:

| Score range | Midpoint $m_i$ | Frequency $f_i$ |
|:---:|:---:|:---:|
| 50--59 | 54.5 | 3 |
| 60--69 | 64.5 | 5 |
| 70--79 | 74.5 | 10 |
| 80--89 | 84.5 | 8 |
| 90--99 | 94.5 | 4 |

**(a)** Estimate the mean score using the weighted mean formula:

$$
\bar{x} = \frac{\sum_{i=1}^k f_i \, m_i}{\sum_{i=1}^k f_i}
$$

**(b)** Estimate the variance using the grouped data formula:

$$
s^2 = \frac{\sum_{i=1}^k f_i (m_i - \bar{x})^2}{\left(\sum_{i=1}^k f_i\right) - 1}
$$

**(c)** Explain why these are estimates rather than exact values.

??? success "Solution"

    **(a)** Computing the weighted sum:

    $$
    \sum f_i \, m_i = 3(54.5) + 5(64.5) + 10(74.5) + 8(84.5) + 4(94.5)
    $$

    $$
    = 163.5 + 322.5 + 745 + 676 + 378 = 2285
    $$

    $$
    \bar{x} = \frac{2285}{30} \approx 76.17
    $$

    **(b)** Computing the weighted squared deviations:

    | Midpoint $m_i$ | $m_i - \bar{x}$ | $(m_i - \bar{x})^2$ | $f_i(m_i - \bar{x})^2$ |
    |:---:|:---:|:---:|:---:|
    | 54.5 | $-21.67$ | 469.4 | 1408.3 |
    | 64.5 | $-11.67$ | 136.1 | 680.6 |
    | 74.5 | $-1.67$ | 2.8 | 27.8 |
    | 84.5 | 8.33 | 69.4 | 555.6 |
    | 94.5 | 18.33 | 336.1 | 1344.4 |

    $$
    s^2 = \frac{1408.3 + 680.6 + 27.8 + 555.6 + 1344.4}{29} = \frac{4016.7}{29} \approx 138.5
    $$

    $$
    s \approx \sqrt{138.5} \approx 11.77
    $$

    **(c)** These are estimates because we use the midpoint of each interval to represent all observations in that interval. The true individual scores are unknown — a student in the 70--79 range could have scored anywhere from 70 to 79, not necessarily 74.5. The grouped mean and variance therefore involve an approximation error that depends on how uniformly scores are distributed within each bin. With narrower bins, the midpoint approximation improves.
