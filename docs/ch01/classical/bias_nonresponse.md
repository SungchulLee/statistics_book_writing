# Bias and Nonresponse

## Overview

Even with careful survey design, various **biases** can distort results and lead to incorrect conclusions. Understanding the sources of bias—and the historical disasters they have caused—is essential for anyone working with data.

## Types of Bias in Sample Surveys

### Sampling Bias (Under-Coverage Bias)

Occurs when the sample is **not representative** of the population because certain groups are overrepresented or underrepresented. This often arises when the sampling frame (the list from which subjects are drawn) does not cover the full population.

### Nonresponse Bias

Occurs when a significant portion of the selected sample **does not respond** to the survey. If non-respondents differ systematically from respondents, the results will be skewed.

### Response Bias

Arises when respondents **do not answer truthfully** or accurately, often due to poorly worded questions, social desirability pressure, or misunderstanding of the question.

### Selection Bias

Occurs when the process of **selecting or including** subjects in the study systematically favors certain outcomes. This includes survivorship bias, volunteer bias, and healthy worker bias.

## Famous Historical Examples

### The Literary Digest Disaster — FDR vs. Landon (1936)

| | FDR Percentage | Sample Size |
|---|---|---|
| Election Outcome | 62% | — |
| Digest's Prediction | 43% | 2,400,000 |
| Gallup's Prediction of Digest | 44% | 3,000 |
| Gallup's Prediction | 56% | 50,000 |

The *Literary Digest* surveyed 2.4 million people—an enormous sample—yet predicted the wrong winner by nearly 20 percentage points. The magazine drew its sample from **telephone directories and automobile registrations**, which in the 1930s skewed heavily toward wealthier individuals who favored Landon. Meanwhile, George Gallup used a scientifically selected random sample of just 50,000 and correctly predicted Roosevelt's victory. Gallup even predicted, using only 3,000 respondents, how wrong the *Digest* would be.

**Lesson:** Sample **quality** matters far more than sample **size**.

### Dewey Defeats Truman (1948)

| | Truman % | Dewey % |
|---|---|---|
| Election Outcome | 50 | 45 |
| Crossley's Prediction | 45 | 50 |
| Gallup's Prediction | 44 | 50 |
| Roper's Prediction | 38 | 53 |

Every major polling organization predicted a Dewey victory. The polls relied on **telephone surveys** that over-sampled middle-class, urban, and Republican-leaning voters. Lower-income and rural voters—who disproportionately supported Truman—were underrepresented because many lacked telephones. Additionally, pollsters **stopped collecting data weeks before** the election, missing Truman's late-campaign momentum from his whistle-stop tour.

The *Chicago Daily Tribune* went to press early with the headline "Dewey Defeats Truman." Truman famously held up the erroneous paper the next morning, producing one of the most iconic photographs in American political history.

### The 2016 U.S. Election Polling Errors

Many polls predicted a Clinton victory but **over-sampled** urban and college-educated voters who leaned toward Clinton, while **under-sampling** rural and non-college-educated voters who favored Trump. Despite sophisticated methods, the failure to accurately capture key demographic segments led to widespread polling errors across swing states.

## Selection Bias: Memorable Examples

### Survivorship Bias in World War II

During WWII, military researchers examined returning planes to decide where to add armor. The planes had more bullet holes in the wings and tail than in the cockpit or engine area. The intuitive conclusion—armor the wings and tail—was **wrong**. The planes that were hit in the cockpit or engine *never returned*. The correct approach, identified by statistician Abraham Wald, was to armor the areas with **few** bullet holes on surviving planes, because hits there were fatal.

### The Healthy Worker Effect

Studies of employed people often show that workers are healthier than the general population. This is not because work makes you healthy; rather, people who are too sick or disabled to work are excluded from the "employed" sample, creating a systematic bias.

### Online Product Reviews

Products with overwhelmingly positive reviews may not reflect the average consumer's experience. People who are extremely satisfied or extremely dissatisfied are more likely to leave reviews, while those with moderate opinions stay silent. Companies sometimes encourage only satisfied customers to review, further skewing the picture.

### Diet Program Testimonials

Diet programs showcase their most successful participants. The many people who did not lose weight or who dropped out are invisible, giving a biased impression of the program's effectiveness.

## The House Effect in Polling

The **House Effect** refers to systematic differences between polling firms that arise from methodological choices rather than random error. Sources include:

- **Question wording and ordering**: Leading or ambiguous phrasing influences responses.
- **Sampling frame**: Different databases yield different demographic mixes.
- **Weighting**: Statistical adjustments to correct demographic imbalances can themselves introduce bias.
- **Mode of interview**: Phone, online, and in-person surveys attract different respondent profiles.

### Addressing the House Effect

1. **Poll aggregation**: Averaging results across many firms (as done by FiveThirtyEight and RealClearPolitics) smooths out individual biases.
2. **Standardizing question wording**: Consistent phrasing across firms improves comparability.
3. **Weighting by historical accuracy**: Giving more weight to firms with better track records.
4. **Transparency in methodology**: Public disclosure of methods enables expert scrutiny.
5. **Mixed-mode polling**: Combining online, phone, and in-person surveys to capture a broader respondent base.

## Python Example: Simulating Nonresponse Bias

```python
import numpy as np

np.random.seed(42)

# Population: 10,000 people, satisfaction score 1-10
# Higher-satisfaction people are more likely to respond
n_pop = 10_000
satisfaction = np.random.normal(5.5, 2.0, n_pop).clip(1, 10)

# Response probability increases with satisfaction
response_prob = 0.1 + 0.08 * (satisfaction - 1)  # higher satisfaction → more likely to respond
responded = np.random.binomial(1, response_prob).astype(bool)

true_mean = satisfaction.mean()
biased_mean = satisfaction[responded].mean()
response_rate = responded.mean()

print(f"True population mean satisfaction: {true_mean:.2f}")
print(f"Survey mean (with nonresponse):    {biased_mean:.2f}")
print(f"Response rate:                     {response_rate:.1%}")
print(f"Bias (overestimate):               {biased_mean - true_mean:+.2f}")
```

## Key Takeaways

- Bias can enter a study through sampling, nonresponse, response, or selection mechanisms.
- A large sample size **does not** compensate for a biased sampling method—the Literary Digest's 2.4 million responses were less accurate than Gallup's 50,000.
- Survivorship bias, the healthy worker effect, and review bias are pervasive in everyday reasoning.
- Awareness of these biases is the first step toward designing better studies and interpreting data more critically.
- Poll aggregation and methodological transparency are practical strategies for mitigating the House Effect in public opinion research.
