# Test of Homogeneity

## Homogeneity Test vs Independence Test

The **Chi-Square Test of Independence** and the **Chi-Square Test of Homogeneity** use **the exact same computational procedure** — but they **differ in purpose, experimental design, and interpretation**.

### The Core Similarity

Both tests use the same **χ² test statistic**:

$$
\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
$$

and the same **sampling distribution** (χ² with $(r-1)(c-1)$ degrees of freedom).

The **expected counts** are computed the same way:

$$
E_{ij} = \frac{(\text{row total})(\text{column total})}{\text{grand total}}
$$

So if you only looked at the calculations, you could not tell which test you were doing. The difference lies in **how the data were collected** and **what question you are answering**.

### The Conceptual Difference

| Feature               | **Test of Independence**                                                | **Test of Homogeneity**                                                                                |
|-----------------------|-------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| **Research question** | Are two categorical variables **associated** (statistically dependent)? | Are **two or more populations** similar (homogeneous) in their distribution of a categorical variable? |
| **Data source**       | One single random sample, classified by **two variables**.              | Multiple independent random samples, one from each population.                                         |
| **Example question**  | Is **smoking status** related to **gender** in a population?            | Do **men, women, and teenagers** have the **same distribution** of smoking habits?                     |
| **Sampling design**   | One sample → cross-classify by both variables.                          | Separate samples from each group (or treatment).                                                       |
| **Interpretation**    | Tests for **association** or **independence** between two variables.    | Tests for **similarity (homogeneity)** of distributions across populations.                            |

### Example Comparison

#### Independence Example

A health researcher surveys **300 people** and records:

- Variable 1: Smoking status (smoker/non-smoker)
- Variable 2: Gender (male/female)

→ One sample, two variables. We test: "Are smoking and gender independent?"

#### Homogeneity Example

A different researcher surveys **100 men**, **100 women**, and **100 teenagers**, asking each whether they smoke.

→ Separate samples from each group. We test: "Are the proportions of smokers the same across the three groups?"

### The Subtle Connection

Mathematically, both tests analyze a **contingency table** of observed counts, compare them to expected counts under $H_0$, and use the same χ² statistic.

- In the **independence test**, "rows" and "columns" represent two variables from a *single* population.
- In the **homogeneity test**, "rows" represent different *populations* or *treatments*, while columns represent categories of one variable.

Under the null hypothesis:

- **Independence test:** the two variables are independent.
- **Homogeneity test:** all populations share the same distribution.

Those are equivalent statements when expressed probabilistically.

### Summary

| Aspect                 | **Independence Test**                  | **Homogeneity Test**                           |
|------------------------|----------------------------------------|------------------------------------------------|
| Data collection        | One sample → two categorical variables | Two or more samples → one categorical variable |
| Null hypothesis        | The two variables are independent      | All populations have the same distribution     |
| Alternative hypothesis | The variables are associated           | At least one population differs                |
| Test statistic & df    | Identical                              | Identical                                      |
| Interpretation         | Association within a single population | Consistency across populations                 |

> **In short:** The **procedure** is the same, but the **context** differs:
> Independence → relationship *within* one sample.
> Homogeneity → consistency *across* multiple samples.

---

## Example A: Hospital Quality

### Question

For each country, we asked how people in the country feel about the hospital quality from five stars to one star. Here is the data.

**Observed:**

$$
\begin{array}{cccc}
\text{Hospital Quality} & \text{US} & \text{Canada} & \text{Mexico} \\ \hline
\text{5 Star} & 541 & 75 & 231 \\
\text{4 Star} & 498 & 71 & 213 \\
\text{3 Star} & 779 & 96 & 321 \\
\text{2 Star} & 282 & 50 & 345 \\
\text{1 Star} & 65 & 19 & 120
\end{array}
$$

Is the hospital satisfaction level distribution homogeneous among the countries, or do some differ?

### Python Implementation (Without `scipy.stats.chi2_contingency`)

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def compute_expected(observed):
    row_sum = observed.sum(axis=1)
    row_pmf = row_sum.reshape((-1, 1)) / row_sum.sum()

    column_sum = observed.sum(axis=0)
    column_pmf = column_sum.reshape((1, -1)) / column_sum.sum()

    joint_pmf = row_pmf * column_pmf
    expected = joint_pmf * row_sum.sum()
    return expected

def main():
    observed = np.array([[541, 75, 231], [498, 71, 213],
                         [779, 96, 321], [282, 50, 345], [65, 19, 120]])
    expected = compute_expected(observed)
    df = (observed.shape[0] - 1) * (observed.shape[1] - 1)

    statistic = np.sum((observed - expected)**2 / expected)
    p_value = stats.chi2(df).sf(statistic)
    print(f"{statistic = :.02f}")
    print(f"{p_value   = :.02%}")

    _, ax = plt.subplots(figsize=(12, 4))

    x = np.linspace(0, statistic, 1000)
    y = stats.chi2(df).pdf(x)
    ax.plot(x, y, color='b', linewidth=3)

    x = np.concatenate([[0], x, [statistic], [0]])
    y = np.concatenate([[0], y, [0], [0]])
    ax.fill(x, y, color='b', alpha=0.1)

    x = np.linspace(statistic, 300, 100)
    y = stats.chi2(df).pdf(x)
    ax.plot(x, y, color='r', linewidth=3)

    x = np.concatenate([[statistic], x, [20], [statistic]])
    y = np.concatenate([[0], y, [0], [0]])
    ax.fill(x, y, color='r', alpha=0.1)

    xy = (250, 0.01)
    xytext = (250, 0.08)
    arrowprops = dict(color='k', width=0.2, headwidth=8)
    ax.annotate(f'{p_value = :.02%}', xy, xytext=xytext, fontsize=15, arrowprops=arrowprops)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position("zero")
    ax.spines['left'].set_position("zero")

    plt.show()

if __name__ == "__main__":
    main()
```

### Python Implementation (With `scipy.stats.chi2_contingency`)

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def main():
    observed = np.array([[541, 75, 231], [498, 71, 213],
                         [779, 96, 321], [282, 50, 345], [65, 19, 120]])

    statistic, p_value, df, expected = stats.chi2_contingency(observed)
    print(f"{statistic = :.02f}")
    print(f"{p_value   = :.02%}")

    _, ax = plt.subplots(figsize=(12, 4))

    x = np.linspace(0, statistic, 1000)
    y = stats.chi2(df).pdf(x)
    ax.plot(x, y, color='b', linewidth=3)

    x = np.concatenate([[0], x, [statistic], [0]])
    y = np.concatenate([[0], y, [0], [0]])
    ax.fill(x, y, color='b', alpha=0.1)

    x = np.linspace(statistic, 300, 100)
    y = stats.chi2(df).pdf(x)
    ax.plot(x, y, color='r', linewidth=3)

    x = np.concatenate([[statistic], x, [20], [statistic]])
    y = np.concatenate([[0], y, [0], [0]])
    ax.fill(x, y, color='r', alpha=0.1)

    xy = (250, 0.01)
    xytext = (250, 0.08)
    arrowprops = dict(color='k', width=0.2, headwidth=8)
    ax.annotate(f'{p_value = :.02%}', xy, xytext=xytext, fontsize=15, arrowprops=arrowprops)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position("zero")
    ax.spines['left'].set_position("zero")

    plt.show()

if __name__ == "__main__":
    main()
```

### Homogeneous Case Comparison

To illustrate how a homogeneous distribution looks, compare the original data with a case where the distributions are similar across countries:

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def compute_expected(observed):
    row_sum = observed.sum(axis=1)
    row_pmf = row_sum.reshape((-1, 1)) / row_sum.sum()
    column_sum = observed.sum(axis=0)
    column_pmf = column_sum.reshape((1, -1)) / column_sum.sum()
    joint_pmf = row_pmf * column_pmf
    expected = joint_pmf * row_sum.sum()
    return expected

def main():
    # Homogeneous case — distributions are similar across countries
    observed = np.array([[541, 530, 550], [498, 490, 503],
                         [779, 750, 760], [282, 270, 265], [65, 60, 58]])
    expected = compute_expected(observed)
    df = (observed.shape[0] - 1) * (observed.shape[1] - 1)

    statistic = np.sum((observed - expected)**2 / expected)
    p_value = stats.chi2(df).sf(statistic)
    print(f"{statistic = :.02f}")
    print(f"{p_value   = :.02%}")

if __name__ == "__main__":
    main()
```

### Two-Country Comparison (US vs Canada)

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def main():
    observed = np.array([[541, 75], [498, 71], [779, 96], [282, 50], [65, 19]])

    statistic, p_value, df, expected = stats.chi2_contingency(observed)
    print(f"{statistic = :.02f}")
    print(f"{p_value   = :.02%}")
    print("expected")
    print(expected)

if __name__ == "__main__":
    main()
```

---

## Example B: Favorite Subject vs. Dominant Hand

> **Source**: [Khan Academy — Chi-Square Test Homogeneity](https://www.khanacademy.org/math/ap-statistics/chi-square-tests/chi-square-tests-two-way-tables/v/chi-square-test-homogeneity)

We want to determine whether left-handed and right-handed individuals exhibit similar inclinations towards science, technology, engineering, mathematics, humanities, or none of the above.

- **Null Hypothesis**: There is no difference in the distribution of subject preferences between left-handed and right-handed individuals.
- **Alternative Hypothesis**: There is a difference in the distribution of subject preferences between left- and right-handed individuals.

We gather a random sample of 60 right-handed individuals and another random sample of 40 left-handed individuals:

|            | Right | Left | Total   |
|:----------:|:-----:|:----:|:-------:|
| STEM       | 30    | 10   | **40**  |
| Humanities | 15    | 25   | **40**  |
| Equal      | 15    | 5    | **20**  |
| Total      | **60**| **40** | **100** |

### Python Implementation

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def main():
    observed = np.array([[30, 10], [15, 25], [15, 5]])

    statistic, p_value, df, expected = stats.chi2_contingency(observed)
    print(f"{statistic = :.02f}")
    print(f"{p_value   = :.04f}")
    print(f"\nExpected frequencies:")
    print(expected)

    _, ax = plt.subplots(figsize=(12, 4))

    x = np.linspace(0, statistic)
    y = stats.chi2(df).pdf(x)
    ax.plot(x, y, color='b', linewidth=3)

    x = np.concatenate([[0], x, [statistic], [0]])
    y = np.concatenate([[0], y, [0], [0]])
    ax.fill(x, y, color='b', alpha=0.1)

    x = np.linspace(statistic, 20, 100)
    y = stats.chi2(df).pdf(x)
    ax.plot(x, y, color='r', linewidth=3)

    x = np.concatenate([[statistic], x, [20], [statistic]])
    y = np.concatenate([[0], y, [0], [0]])
    ax.fill(x, y, color='r', alpha=0.1)

    xy = (15.0, 0.01)
    xytext = (16.5, 0.10)
    arrowprops = dict(color='k', width=0.2, headwidth=8)
    ax.annotate(f'{p_value = :.04f}', xy, xytext=xytext, fontsize=15, arrowprops=arrowprops)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position("zero")
    ax.spines['left'].set_position("zero")

    plt.show()

if __name__ == "__main__":
    main()
```

## Exercises

**Exercise 1.**
Drug A: 60/100 successes. B: 55/100. (a) Controlling confounders? (b) $z$-test. (c) $\chi^2$ test. Show equivalence.

??? success "Solution to Exercise 1"
    (a) **Randomize** participants to A or B. Random assignment balances confounders (age, gender, severity) across groups in expectation.

    (b) Pooled $\hat p = 115/200 = 0.575$. $\mathrm{SE} = \sqrt{0.575 \cdot 0.425 \cdot (1/100 + 1/100)} \approx 0.0699$.

    $z = (0.60 - 0.55)/0.0699 \approx 0.715$. $|z| < 1.96$. Fail to reject.

    (c) Expected: all cells 57.5 (success) or 42.5 (failure). $\chi^2 = 4 \cdot (2.5)^2/57.5 + 4 \cdot (2.5)^2/42.5 \approx 0.512$. $0.512 < 3.84$. Fail to reject.

    Equivalence: $z^2 = 0.715^2 = 0.511 \approx \chi^2$. For $2 \times 2$ tables, $\chi^2$ test ≡ two-sided $z$-test for proportions.

---

**Exercise 2.**
**Test of homogeneity** vs independence. What's the difference?

??? success "Solution to Exercise 2"
    Both use chi-square with same statistic and df. Difference is in the **sampling design**:

    **Homogeneity:** fixed margins for one variable (e.g., $n_A = n_B = 100$ pre-specified). Test whether the distribution of the other variable is the same across rows.

    **Independence:** total $n$ is fixed; cell counts randomly distributed. Test whether two variables are independent.

    **Same math, different interpretation.** A drug trial with pre-assigned sample sizes is homogeneity. An observational study of customer preferences is independence.

    Practically: indistinguishable in computation; conceptually distinct because of design assumptions.

---

**Exercise 3.**
**Multi-population homogeneity.** Three drugs compared: A (60/100), B (55/100), C (45/100). Test if all three have the same success rate.

??? success "Solution to Exercise 3"
    Table: success row = (60, 55, 45), failure row = (40, 45, 55). Total = 300; success total = 160.

    Pooled $\hat p_{\text{success}} = 160/300 \approx 0.533$.

    Expected per group: success 53.33, failure 46.67.

    $\chi^2 = \sum (O - E)^2/E$:

    - A: $(60-53.33)^2/53.33 + (40-46.67)^2/46.67 \approx 0.834 + 0.953 = 1.787$.
    - B: $(55-53.33)^2/53.33 + (45-46.67)^2/46.67 \approx 0.052 + 0.060 = 0.112$.
    - C: $(45-53.33)^2/53.33 + (55-46.67)^2/46.67 \approx 1.302 + 1.488 = 2.790$.

    Total $\chi^2 \approx 4.69$. df = $(3-1)(2-1) = 2$. Critical $\chi^2_{2, 0.05} = 5.99$. **Fail to reject** at 5%.

    Although Drug C has visibly lower success rate (45% vs 60%), the test doesn't reach significance.

---

**Exercise 4.**
**Post-hoc analysis** after rejecting homogeneity. What's recommended?

??? success "Solution to Exercise 4"
    After omnibus chi-square rejects $H_0$, find which groups differ.

    **Options:**

    - **Pairwise chi-square** with Bonferroni correction: 3 groups → 3 pairwise tests at $\alpha/3$.
    - **Adjusted residuals:** $r_{ij} = (O - E)/\sqrt{E \cdot (1 - p_i)(1 - p_j)}$. $|r| > 2$ indicates significant cell.
    - **Z-test for two proportions** between specific groups of interest.

    Important: control family-wise error or false discovery rate when making multiple comparisons.

---

**Exercise 5.**
**McNemar's test** for paired/matched binary data. Define and contrast with chi-square.

??? success "Solution to Exercise 5"
    Setup: same subjects measured twice (pre/post, two raters). Binary outcomes.

    Matched table:
    | | After + | After - |
    |---|---|---|
    | Before + | $a$ | $b$ |
    | Before - | $c$ | $d$ |

    **McNemar's statistic:** $\chi^2 = (b - c)^2/(b + c)$. df = 1.

    Tests whether the marginal proportions changed (e.g., "did treatment shift success rate?").

    **Contrast with chi-square:** chi-square for independence assumes independent observations. McNemar accounts for pairing — uses only the discordant pairs ($b$, $c$).

    Example: agree-disagree pairs in survey, before-after improvements in treatment.

---

**Exercise 6.**
**Power analysis** for chi-square homogeneity.

??? success "Solution to Exercise 6"
    Effect size: $w = \sqrt{\sum (p_{ij} - p_{ij,0})^2/p_{ij,0}}$ where $p_{ij,0}$ is expected under $H_0$.

    Cohen's conventions: $w = 0.1$ (small), 0.3 (medium), 0.5 (large).

    Required $n$ for 80% power, $\alpha = 0.05$, df = 2: $\lambda \approx 9.63$, $n = \lambda/w^2$.

    Small effect: $n \approx 963$. Medium: $n \approx 107$. Large: $n \approx 39$.

    Use `statsmodels.stats.power.GofChisquarePower` or formal computation. Sample-size planning is essential — underpowered chi-square tests are common in applied research.
