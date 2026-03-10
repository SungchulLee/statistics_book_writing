# Sample Surveys and Sampling Methods

Sample surveys collect data from a representative subset of a population, enabling inference about the whole without studying every individual. The choice of sampling method directly affects accuracy and generalizability.

## Definition

A **sample survey** selects units from a population according to a probabilistic sampling design and measures variables of interest. The four main methods are:

| Method | Procedure | Advantage | Risk |
|---|---|---|---|
| Simple Random | Each unit equally likely | Unbiased, simple theory | May miss small subgroups |
| Stratified | Divide into strata, sample within each | Guarantees subgroup coverage | Requires stratum knowledge |
| Cluster | Randomly select entire clusters | Cost-effective for dispersed populations | Higher variance if clusters are homogeneous |
| Systematic | Every $k$-th unit from a list | Easy to implement | Bias if list has periodicity |

## Explanation

A representative sample requires every individual to have a known, nonzero selection probability. Simple random sampling (SRS) is the baseline: every subset of size $n$ is equally likely. Stratified sampling improves precision by ensuring each subgroup (stratum) is represented proportionally. Cluster sampling reduces costs when the population is geographically dispersed. Systematic sampling is operationally simple but vulnerable to hidden periodicities in the sampling frame.

Sample size determination balances precision (margin of error), confidence level, and cost. Larger samples reduce sampling error but with diminishing returns: the standard error of $\bar{x}$ decreases as $1/\sqrt{n}$, so quadrupling the sample size only halves the error.

## Examples

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n_pop = 10_000
stratum = np.random.choice(["Young", "Old"], size=n_pop, p=[0.7, 0.3])
income = np.where(stratum == "Young",
                  np.random.normal(40_000, 10_000, n_pop),
                  np.random.normal(70_000, 15_000, n_pop))
pop = pd.DataFrame({"stratum": stratum, "income": income})
true_mean = pop["income"].mean()

# Simple random sample
srs = pop.sample(200, random_state=1)

# Stratified sample (proportional allocation)
strat = pop.groupby("stratum", group_keys=False).apply(
    lambda x: x.sample(int(200 * len(x) / n_pop), random_state=1)
)

print(f"True mean:      ${true_mean:,.0f}")
print(f"SRS mean:       ${srs['income'].mean():,.0f}  "
      f"(error: ${abs(srs['income'].mean() - true_mean):,.0f})")
print(f"Stratified mean: ${strat['income'].mean():,.0f}  "
      f"(error: ${abs(strat['income'].mean() - true_mean):,.0f})")
```
