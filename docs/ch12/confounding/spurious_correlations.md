# Spurious Correlations

Two variables can exhibit a strong statistical correlation despite having no causal connection whatsoever. Such **spurious correlations** arise from confounding variables, common trends, coincidence in finite samples, or data dredging. Recognizing spurious correlations is essential for avoiding false conclusions and is at the heart of the principle that **correlation does not imply causation**.

---

## What Makes a Correlation Spurious

A correlation between $X$ and $Y$ is **spurious** when the association does not reflect a direct causal relationship in either direction. The correlation exists, but it is misleading if interpreted as evidence of a causal link. Spurious correlations arise from several distinct mechanisms.

---

## Mechanism 1: Common Cause (Confounding)

The most frequent source of spurious correlation is a **confounding variable** $Z$ that influences both $X$ and $Y$:

$$
X \leftarrow Z \rightarrow Y
$$

The correlation between $X$ and $Y$ exists only because both are driven by $Z$. Once $Z$ is controlled for, the association disappears.

??? example "Ice cream and drowning"
    Ice cream sales ($X$) and drowning incidents ($Y$) are positively correlated. The confounder is temperature ($Z$): hot weather increases both ice cream consumption and swimming activity. Ice cream does not cause drowning.

??? example "Shoe size and reading ability"
    Among schoolchildren, shoe size correlates positively with reading ability. The confounder is age: older children have both larger feet and better reading skills. Shoe size does not cause literacy.

---

## Mechanism 2: Common Trends

Two time series that both increase (or decrease) over time will be positively correlated, even if they are completely unrelated. This is a pervasive source of spurious correlations in observational data.

Common trends arise from:

- **Secular trends**: population growth, inflation, and technological change cause many time series to trend upward together.
- **Seasonality**: any two variables with seasonal patterns will correlate if measured over the same time period.

!!! warning "Trending time series produce spurious correlations"
    The correlation between U.S. spending on science and suicides by hanging ($r \approx 0.99$ over a particular decade) is entirely spurious. Both series happen to trend upward over time. This is a statistical artifact, not a meaningful relationship.

When analyzing time series, detrending or differencing the data before computing correlations can help distinguish genuine associations from artifacts of shared trends.

---

## Mechanism 3: Coincidence and Data Dredging

With enough variables, some pairs will be strongly correlated by pure chance. If a researcher tests thousands of variable pairs and reports only the ones with high correlations, many of those will be spurious.

This problem is exacerbated by:

- **Multiple comparisons**: testing many hypotheses without adjusting for multiplicity.
- **Data dredging** (p-hacking): searching through data for any significant result.
- **Publication bias**: journals preferentially publish surprising, positive findings.

The website "Spurious Correlations" by Tyler Vigen catalogs absurd but statistically strong correlations (e.g., per capita cheese consumption and deaths by bedsheet entanglement), illustrating how easily meaningless patterns emerge from large datasets.

---

## Mechanism 4: Collider Bias (Selection Bias)

A subtler form of spurious correlation arises from conditioning on a **collider** -- a variable caused by both $X$ and $Y$:

$$
X \rightarrow Z \leftarrow Y
$$

When we condition on $Z$ (e.g., by selecting only individuals with a particular value of $Z$), a spurious association between $X$ and $Y$ is created even if they are marginally independent.

??? example "Talent and attractiveness in Hollywood"
    Among the general population, acting talent ($X$) and physical attractiveness ($Y$) may be unrelated. But among successful actors ($Z$), they appear negatively correlated: less talented actors tend to be more attractive, and vice versa. The conditioning on success (which requires either talent or attractiveness) creates a spurious negative association.

---

## Distinguishing Spurious from Genuine Correlations

No purely statistical test can determine whether a correlation is spurious. However, several strategies help:

1. **Check for confounders.** Identify plausible third variables and compute [partial correlations](../correlation/partial.md). If the correlation disappears after conditioning, it was likely confounded.

2. **Consider the mechanism.** Is there a plausible causal pathway? A correlation without a plausible mechanism is more likely spurious.

3. **Replicate.** Does the correlation persist in independent datasets? Spurious correlations due to chance or data dredging typically do not replicate.

4. **Examine temporal ordering.** If $X$ occurs after $Y$ in time, $X$ cannot cause $Y$ (though a common cause could still explain both).

5. **Look for dose-response.** Does the association strengthen as $X$ increases? Dose-response relationships are more consistent with causation than spurious correlation.

6. **Control for common trends.** For time series, detrend or difference the data before computing correlations.

---

## Why Correlation Does Not Imply Causation

The existence of spurious correlations is the fundamental reason behind the maxim **"correlation does not imply causation."** An observed correlation between $X$ and $Y$ is consistent with multiple causal structures:

- $X$ causes $Y$
- $Y$ causes $X$
- A common cause $Z$ drives both
- The correlation is a statistical artifact (coincidence, data dredging, collider bias)

Only by ruling out alternative explanations -- through randomized experiments, careful observational study design, or formal causal inference methods -- can we move from correlation to causation. See [Criteria for Causal Inference](../causation/causal_criteria.md).

---

## Summary

Spurious correlations are statistical associations that do not reflect direct causal relationships. They arise from confounding variables, shared trends, coincidence, data dredging, and collider bias. The existence of spurious correlations is the primary reason that correlation does not imply causation. Identifying and ruling out alternative explanations for an observed correlation is a necessary step before drawing causal conclusions.

## Exercises

**Exercise 1.**
Simulate an investment scenario to illustrate survivorship bias:

1. Generate 1000 "companies" with random annual returns drawn from $N(0.05, 0.3)$ over 10 years
2. A company "survives" if its cumulative return never drops below $-90\%$
3. Compute the average annual return for survivors vs. all companies
4. Discuss how focusing only on survivors inflates perceived returns

```python
import numpy as np

np.random.seed(42)
n_companies = 1000
n_years = 10

# Your simulation here
```

??? success "Solution to Exercise 1"

    Companies that experience a cumulative return drop below $-90\%$ are removed from the survivor set. The surviving companies are a biased sample — they include companies that happened to have favorable return sequences while excluding those that suffered catastrophic losses. The average annual return for survivors will be higher than the average for all companies because the worst performers have been removed. This is survivorship bias: analyzing only the survivors creates a spuriously positive picture of investment performance. In practice, mutual fund databases and stock indices suffer from this bias because delisted or merged funds disappear from the historical record.

---

**Exercise 2.**
For each scenario below, identify the survivorship bias and explain what data is missing:

1. A study finds that people who take a particular supplement live longer on average.
2. An analysis of successful restaurants finds they all have outdoor seating.
3. A review of top-performing mutual funds over 20 years shows consistent market-beating returns.

??? success "Solution to Exercise 2"

    1. **Supplement study**: People who take supplements may be healthier to begin with (healthy user bias). Those who became too ill to continue the supplement or died early are not included in the "supplement user" group. Missing data: people who started and stopped the supplement due to illness or death.

    2. **Restaurant analysis**: Only surviving (successful) restaurants are studied. Restaurants that had outdoor seating but failed are not in the dataset. Missing data: all restaurants that opened with outdoor seating but subsequently closed. The analysis cannot determine whether outdoor seating contributes to success.

    3. **Mutual funds**: Funds that performed poorly over 20 years were likely closed, merged, or renamed. Only the funds that survived (and therefore tended to perform well) remain in the database. Missing data: the full universe of funds that existed at the start of the 20-year period, including those that were subsequently liquidated.
