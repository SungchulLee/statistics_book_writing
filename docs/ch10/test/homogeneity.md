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
