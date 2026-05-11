# Modality

## Overview

The **modality** of a distribution describes the number of distinct peaks (modes) in its shape. Identifying modality is a critical first step in exploratory data analysis because it reveals whether the data comes from a single population or is a mixture of distinct subgroups.

## Unimodal Distributions

A **unimodal** distribution has a single peak. The most familiar example is the normal (bell curve) distribution, where data clusters around one central value.

**Example:** Heights of adult women in a single country typically form a unimodal distribution centered near the population mean.

## Bimodal Distributions

A **bimodal** distribution has two distinct peaks, indicating that the data likely contains two separate groups or processes.

**Example:** Test scores in a class might be bimodal if one group of students studied extensively and another did not, producing peaks at high and low scores with a valley in between.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(0)

# Two normal distributions centered at different locations
data_normal_1 = stats.norm().rvs(1_000)
data_normal_2 = stats.norm(loc=6).rvs(1_000)
combined_data = np.concatenate((data_normal_1, data_normal_2))

fig, ax = plt.subplots(figsize=(12, 3))
ax.hist(combined_data, bins=30, color='skyblue', edgecolor='black')
ax.set_title("Histogram of Bimodal Distribution")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
```

The two peaks are clearly visible, each corresponding to one of the component normal distributions.

## Multimodal Distributions

A **multimodal** distribution has more than two peaks. This often arises from mixing three or more subpopulations.

**Example:** The distribution of commute times in a large metropolitan area might show peaks at walking distance, short drive, and long commute durations.

## Why Modality Matters

Detecting modality has practical consequences for analysis:

- A bimodal or multimodal distribution signals that **summary statistics like the mean may be misleading**, as the mean could fall in a valley between peaks where few observations actually lie.
- It suggests that the data should be **disaggregated** into subgroups before further analysis.
- Standard parametric methods assuming unimodality (e.g., t-tests, normal-based confidence intervals) may be inappropriate for multimodal data.

## Detecting Modality

Common approaches include visual inspection of histograms and density plots, kernel density estimation (KDE) with varying bandwidths, and formal tests such as the dip test of unimodality. In practice, the histogram with a reasonable number of bins is the simplest and most effective first check.

## Summary

Modality provides essential information about the structure of a dataset. Unimodal distributions suggest a single homogeneous population, while bimodal or multimodal shapes point to underlying subgroups that deserve separate investigation.

## Exercises

**Exercise 1.**
A dataset of adult heights from a mixed-gender population shows two peaks: one near 163 cm and another near 176 cm. Is this distribution unimodal, bimodal, or multimodal? What underlying subgroups likely explain the shape?

??? success "Solution to Exercise 1"
    The distribution is **bimodal** because it has two distinct peaks. The two subgroups are almost certainly **males and females**, whose height distributions overlap but have different means (approximately 176 cm for males and 163 cm for females in many populations). When the two groups are combined, the histogram shows two modes corresponding to the typical heights of each sex.

---

**Exercise 2.**
A researcher computes the mean of a dataset and finds it is 50. The histogram shows two peaks at approximately 30 and 70 with a valley near 50. Explain why the mean is a misleading summary in this case.

??? success "Solution to Exercise 2"
    The mean of 50 falls in the valley between the two peaks, a region where very few observations actually lie. This means the "average" value is not representative of any typical observation in the dataset. In a bimodal distribution, reporting only the mean hides the fact that the data contains two distinct clusters. A more informative summary would report the location and spread of each mode separately, or at minimum note that the distribution is bimodal with peaks near 30 and 70.

---

**Exercise 3.**
Give an example of a real-world dataset that you would expect to be: (a) unimodal, (b) bimodal, (c) multimodal with three or more modes. Justify each choice.

??? success "Solution to Exercise 3"
    **(a) Unimodal:** The distribution of exam scores in a well-designed test given to a homogeneous group of students. Most students score near the class average, with fewer scoring very high or very low, producing a single peak.

    **(b) Bimodal:** The distribution of commute times in a city where most people either walk to work (peak around 10 minutes) or drive on a highway (peak around 40 minutes), with few people in between.

    **(c) Multimodal (three or more modes):** The distribution of eruption durations at Old Faithful geyser, which has been observed to have multiple clusters of short, medium, and long eruptions. Another example: the distribution of prices at a grocery store, where items cluster around common price points like \$1, \$3, and \$5.

---

**Exercise 4.**
Suppose you observe a histogram that appears bimodal. Describe two different strategies for further analysis, and explain what each would reveal.

??? success "Solution to Exercise 4"
    **Strategy 1: Disaggregate by a grouping variable.** If a plausible categorical variable is available (e.g., sex, treatment group, geographic region), plot separate histograms for each group. If the bimodality disappears within each group and each subgroup shows a unimodal distribution, this confirms that the overall bimodality arises from mixing distinct subpopulations.

    **Strategy 2: Fit a mixture model.** Use a Gaussian mixture model (GMM) with two components to estimate the means, variances, and mixing proportions of the two underlying distributions. This reveals the center, spread, and relative size of each cluster without requiring an explicit grouping variable. The Bayesian Information Criterion (BIC) can be used to compare the two-component model against a one-component model to assess whether the bimodality is statistically justified.

---

**Exercise 5.**
The **dip test** of Hartigan and Hartigan (1985) formally tests the null hypothesis that a distribution is unimodal. Briefly describe how it works and one alternative non-parametric test for multimodality.

??? success "Solution to Exercise 5"
    **Dip test:** computes the maximum vertical distance between the empirical CDF and the closest *unimodal* CDF (one with a single inflection in its derivative). Under the null of unimodality this dip statistic is small; under the alternative of bimodality or multimodality it grows because no unimodal CDF can closely approximate the empirical one. A reference distribution (computed by simulation from a uniform null) provides $p$-values.

    **Alternative — Silverman's bandwidth test:** searches for the smallest bandwidth $h$ such that a Gaussian KDE with that bandwidth yields at most $k$ modes. The null hypothesis of "at most $k$ modes" is rejected when the critical bandwidth is significantly larger than expected under unimodality. Bootstrap calibration provides the $p$-value.

    Both tests have low power for sample sizes below 100 and can be sensitive to the smoothness assumption. Visual inspection of KDE plots at multiple bandwidths is often more informative in practice.

---

**Exercise 6.**
**Mixture models** can produce distributions that are bimodal *or* unimodal depending on how separated the components are. Describe the relationship between component separation and observed modality for a two-component normal mixture with equal mixing weights and equal variances.

??? success "Solution to Exercise 6"
    Consider $f(x) = 0.5 \cdot \phi(x; -\mu, \sigma) + 0.5 \cdot \phi(x; +\mu, \sigma)$ for some $\mu > 0$. The mixture is symmetric around 0; the question is whether 0 is a local maximum (unimodal) or a local minimum (bimodal).

    The result (Behboodian 1970): the mixture is **bimodal iff $\mu/\sigma > 1$**. In words:

    - If component means are less than one $\sigma$ apart, the mixture is **unimodal** — the components overlap so heavily that no valley appears.
    - If component means are more than one $\sigma$ apart, the mixture is **bimodal** — a valley appears at the midpoint.

    This has a practical consequence: even when data genuinely comes from two distinct populations, a histogram may look unimodal if the populations are close together. Modality is therefore a *lower bound* on the number of subpopulations, not an exact count. Fitting a mixture model with model selection (BIC, AIC) is more reliable than counting peaks for inferring the number of components.
