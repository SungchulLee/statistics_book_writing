# Sample Surveys and Sampling Methods

## Overview

**Sample surveys** are studies in which researchers gather data by selecting a representative sample from a larger population. By focusing on a sample instead of the entire population, surveys allow for efficient and cost-effective data collection while still enabling inferences about the population.

## The Importance of Representative Sampling

A **representative sample** accurately reflects the characteristics of the population from which it is drawn. Every individual in the population should have a known, non-zero chance of being selected. When a sample is representative, results can be generalized to the entire population with a high degree of confidence. If the sample is not representative, results will be biased.

## Types of Sampling Methods

### Simple Random Sampling

Every member of the population has an **equal chance** of being selected. This method is straightforward but may not always be practical for large or dispersed populations.

### Stratified Sampling

The population is divided into **subgroups (strata)** based on certain characteristics (e.g., age, income, education level), and samples are taken from each stratum. This ensures that each subgroup is represented proportionally in the sample, often improving precision compared to simple random sampling.

### Cluster Sampling

The population is divided into **clusters** (e.g., geographical regions), and entire clusters are randomly selected. All members of the selected clusters are surveyed. This method is useful when a population is large and geographically spread out, reducing travel and administrative costs.

### Systematic Sampling

A random starting point is chosen, and every $k$-th individual is selected from a list or sequence. This method is easy to implement but can introduce bias if there is a hidden periodic pattern in the list.

### Comparison

| Method | Procedure | Advantage | Risk |
|---|---|---|---|
| Simple Random | Select each unit with equal probability | Unbiased, simple theory | May miss small subgroups |
| Stratified | Divide into strata, sample within each | Guarantees subgroup representation | Requires knowledge of strata |
| Cluster | Randomly select entire clusters | Cost-effective for dispersed populations | Higher variance if clusters are homogeneous |
| Systematic | Every $k$-th unit from a list | Easy to implement | Bias if list has periodicity |

## Designing a Sample Survey

Designing a rigorous sample survey involves several steps:

1. **Define the population**: Clearly identify the group you want to study. The population can be broad (all adults in a country) or narrow (employees in a specific company).

2. **Choose the sampling method**: Select the method best suited to the research goals, population characteristics, and available resources.

3. **Determine the sample size**: Larger samples generally provide more reliable results but are more costly. The required sample size depends on the desired precision (margin of error) and confidence level.

4. **Develop the survey instrument**: Create clear, unbiased questions. The format (online, phone, in-person) affects response rates and data quality.

5. **Conduct the survey**: Administer consistently to the selected sample to avoid introducing variability.

6. **Analyze the data**: Use statistical techniques to estimate population parameters and measure the reliability of the estimates.

## Python Example: Stratified vs. Simple Random Sampling

```python
import numpy as np
import pandas as pd

np.random.seed(42)

# Simulate a population with two strata
n_pop = 10_000
stratum = np.random.choice(["Young", "Old"], size=n_pop, p=[0.7, 0.3])
income = np.where(stratum == "Young",
                  np.random.normal(40_000, 10_000, n_pop),
                  np.random.normal(70_000, 15_000, n_pop))
pop = pd.DataFrame({"stratum": stratum, "income": income})
true_mean = pop["income"].mean()

# Simple random sample (n=200)
srs = pop.sample(200, random_state=1)
srs_mean = srs["income"].mean()

# Stratified sample (n=200, proportional allocation)
strat = pop.groupby("stratum", group_keys=False).apply(
    lambda x: x.sample(int(200 * len(x) / n_pop), random_state=1)
)
strat_mean = strat["income"].mean()

print(f"True population mean:    ${true_mean:,.0f}")
print(f"Simple random sample:    ${srs_mean:,.0f}  (error: ${abs(srs_mean - true_mean):,.0f})")
print(f"Stratified sample:       ${strat_mean:,.0f}  (error: ${abs(strat_mean - true_mean):,.0f})")
```

## Key Takeaways

- Sample surveys are a practical way to learn about populations without studying every individual.
- The choice of sampling method has a direct impact on the accuracy and generalizability of results.
- Stratified sampling often outperforms simple random sampling when the population has identifiable subgroups with different characteristics.
- Good survey design—from defining the population to crafting unbiased questions—is essential for obtaining reliable data.
