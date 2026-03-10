# Bias and Nonresponse

Even with careful survey design, systematic errors called biases can distort results regardless of sample size. Understanding these sources of error is essential for both designing studies and interpreting their findings.

## Definition

**Bias** is a systematic tendency for a sample statistic to overestimate or underestimate the corresponding population parameter. The main types are:

- **Sampling bias**: The sampling frame does not cover the full population.
- **Nonresponse bias**: Non-respondents differ systematically from respondents.
- **Response bias**: Respondents answer inaccurately (social desirability, question wording).
- **Selection bias**: The inclusion process systematically favors certain outcomes (survivorship bias, healthy worker effect, volunteer bias).

## Explanation

The Literary Digest's 1936 prediction that Landon would defeat Roosevelt is the canonical lesson. The magazine surveyed 2.4 million people drawn from telephone directories and automobile registrations, which in Depression-era America skewed heavily toward wealthy Landon supporters. George Gallup, using a scientifically selected sample of 50,000, correctly predicted Roosevelt's victory. **Sample quality matters far more than sample size.**

Nonresponse bias arises when certain groups systematically refuse to participate. If satisfaction surveys are completed mainly by very happy or very angry customers, the sample mean deviates from the true population mean. The direction of bias depends on which subgroup is over-represented among respondents.

Mitigation strategies include: stratified sampling to cover subgroups, follow-up contacts to reduce nonresponse, anonymous surveys to reduce response bias, and poll aggregation to average out individual firms' house effects.

## Examples

```python
import numpy as np

np.random.seed(42)
n_pop = 10_000
satisfaction = np.random.normal(5.5, 2.0, n_pop).clip(1, 10)

# Nonresponse: higher satisfaction -> more likely to respond
response_prob = 0.1 + 0.08 * (satisfaction - 1)
responded = np.random.binomial(1, response_prob).astype(bool)

true_mean = satisfaction.mean()
biased_mean = satisfaction[responded].mean()
print(f"True population mean: {true_mean:.2f}")
print(f"Biased survey mean:   {biased_mean:.2f}")
print(f"Bias (overestimate):  {biased_mean - true_mean:+.2f}")
print(f"Response rate:        {responded.mean():.1%}")
```
