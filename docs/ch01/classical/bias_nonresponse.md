# Bias and Nonresponse

Even with careful survey design, systematic errors called **biases** can distort results no matter how large the sample. Understanding these errors is essential for designing surveys, interpreting their findings, and challenging headline numbers in the press. This page covers the four most important types of bias in survey work and the strategies that partially mitigate each.

## Definition

**Bias** is a systematic tendency for a sample statistic to over- or under-estimate the corresponding population parameter. Formally, $\mathrm{bias}(\hat\theta) = \mathbb{E}[\hat\theta] - \theta$, with the expectation taken over the *repeated sampling process*, not over the data. Bias persists no matter how many additional samples are drawn.

The four canonical types in survey research:

| Type | Mechanism | Direction |
|---|---|---|
| **Sampling (coverage) bias** | The sampling frame does not match the target population | Unpredictable |
| **Nonresponse bias** | Non-respondents differ systematically from respondents | Depends on who skips |
| **Response (measurement) bias** | Respondents answer inaccurately | Often toward socially desirable answers |
| **Selection bias** | Inclusion in the data is correlated with the outcome | Surveys "successes" |

## Explanation

### Sampling (coverage) bias

The **sampling frame** is the enumerable list from which a sample is drawn. Anyone missing from the frame has zero probability of selection — invisible to the analysis. The classic example: the 1936 *Literary Digest* poll surveyed 2.4 million people drawn from telephone directories and automobile registrations. In Depression-era America these lists skewed wealthy, who tended Republican. The result confidently predicted Landon. Gallup's much smaller probability sample correctly predicted Roosevelt. **Sample quality matters more than sample size.**

### Nonresponse bias

Even with a perfect frame, some selected individuals don't respond. Bias arises when respondents differ from non-respondents on the variable of interest. If a customer-satisfaction survey is completed mainly by very happy or very angry customers, the sample mean misses everyone in between. The direction of bias depends on which subgroup over-responds.

Response rates have collapsed in modern surveys — once 70–80%, now often below 10% for phone surveys. Inference relies on the assumption that, after weighting on observed covariates, response is "missing at random." This is unverifiable from the data alone.

### Response (measurement) bias

Respondents may answer inaccurately due to:

- **Social desirability**: under-reporting alcohol use, drug use, or unethical behavior; over-reporting voting, charity, exercise.
- **Question wording**: "Should the government waste money on X?" vs. "Should the government invest in X?" — produce different answers about the same X.
- **Mode effects**: face-to-face interviews elicit different answers than anonymous online surveys, especially on sensitive topics.
- **Anchoring and recall bias**: questions that prompt specific examples bias subsequent answers.

### Selection (survivorship) bias

The sample consists of entities that passed through some filter, while those that failed are invisible. Mutual-fund returns, startup-success traits, surgical-outcome studies, and many others suffer this — covered in detail in the survivorship-bias section.

### Mitigation strategies

| Bias | Mitigation |
|---|---|
| Coverage | Improve the frame; auxiliary data to detect undercoverage |
| Nonresponse | Multiple callbacks; mixed-mode follow-up; post-stratification weighting; IPW |
| Response | Anonymity; piloting question wording; objective measurements when feasible |
| Selection | Track exits; include all originally enrolled subjects (intention-to-treat) |

All mitigation is partial. Bias is the asymptotic floor that more data cannot lower.

## Examples

```python
"""Nonresponse correlated with the outcome distorts the estimate."""

import numpy as np

rng = np.random.default_rng(42)
n_pop = 10_000

# True satisfaction (Gaussian, mean 5.5)
satisfaction = np.clip(rng.normal(5.5, 2.0, n_pop), 1, 10)

# Response probability increases with satisfaction
response_prob = 0.1 + 0.08 * (satisfaction - 1)
responded = rng.binomial(1, response_prob).astype(bool)

true_mean = satisfaction.mean()
biased_mean = satisfaction[responded].mean()
print(f"True population mean:    {true_mean:.2f}")
print(f"Biased survey mean:      {biased_mean:.2f}")
print(f"Bias (overestimate):     {biased_mean - true_mean:+.2f}")
print(f"Response rate:           {responded.mean():.1%}")
```

The reported mean overstates true satisfaction because happier customers respond more. This is irreducible without modeling the response mechanism.

## Exercises

**Exercise 1.**
For each scenario, identify the **type of bias** (sampling, nonresponse, response, or selection/survivorship).

**(a)** A magazine mails a political opinion survey to its subscribers. Only 15% respond, and respondents hold stronger opinions than non-respondents.
**(b)** A mutual-fund company advertises strong performance of its current funds, not mentioning funds that were closed for poor performance.
**(c)** A telephone survey conducted during weekday business hours misses working adults.
**(d)** In a face-to-face interview, respondents under-report their alcohol consumption.
**(e)** A LinkedIn poll about job satisfaction is shared widely on Twitter, where unhappy professionals are more vocal.

??? success "Solution to Exercise 1"
    (a) Nonresponse bias — 85% non-respondents differ systematically.
    (b) Survivorship (selection) bias — failed funds excluded from the average.
    (c) Sampling (coverage) bias — the frame excludes workday-occupied people during the call window.
    (d) Response bias — social desirability lowers reported drinking.
    (e) Multiple: a selection bias for who sees the poll (Twitter-active LinkedIn users), and a nonresponse bias for who chooses to vote (the loudest voices over-vote).

---

**Exercise 2.**
A clinic surveys patients about pain. The questionnaire asks: "How bad is your pain today, on a scale of 0–10?" Forty percent of patients do not respond. The clinic computes a mean pain score of 4.3 over the responders and reports it as the patient population's average pain. What can you say about the direction of bias *without* additional information?

??? success "Solution to Exercise 2"
    Nothing definite about the direction. Either tail of the distribution might over- or under-respond:

    - Very high-pain patients may be too uncomfortable to fill out forms → biased downward.
    - Very high-pain patients may be most motivated to communicate to clinicians → biased upward.
    - Low-pain patients may skip the survey because they don't think it matters → biased upward.

    Without auxiliary information (chart-recorded pain medications, observed grimacing, follow-up contact) the direction is unidentifiable. The point estimate is *biased* in some direction but the analyst cannot say which without an extra-statistical model.

    Practical fix: improve response rate to above 80%, or use the chart data to estimate response propensities and weight accordingly.

---

**Exercise 3.**
Distinguish **bias** from **variance**. A poll of 50 voters and a poll of 50,000 voters from the same biased frame produce estimates with very different precisions. Which is "more accurate"?

??? success "Solution to Exercise 3"
    Define **accuracy** as small total error from the truth: $\mathrm{MSE} = \mathrm{bias}^2 + \mathrm{variance}$. A large biased poll has tiny variance (sharp confidence interval) but the *bias* term dominates total error. A small unbiased poll has high variance and zero bias.

    Without measuring the bias, neither poll is unambiguously more "accurate" — but the large biased poll gives a *more confident wrong answer*. The 50,000-voter poll will routinely produce estimates outside its own reported confidence intervals over many repeated polls of the same biased process — proof that the CI is misleading.

    A meta-analysis of biased polls is even worse: averaging biased polls converges to the bias asymptote, not to the truth. The fix is not more polling — it is fixing the frame.

---

**Exercise 4.**
**Post-stratification** is the practice of re-weighting respondents so that the weighted demographic composition matches known population marginals (e.g., from the census). Explain when post-stratification reduces bias and when it does not.

??? success "Solution to Exercise 4"
    **Reduces bias** when the variables determining response are also the variables being post-stratified on. If young people respond less and you have age data, post-stratification by age corrects the bias *under the assumption* that respondents of a given age are representative of all people of that age.

    **Does not reduce bias** when there is residual non-response within strata. A 30-year-old male respondent in an urban area may still differ on the outcome from a 30-year-old male non-respondent in an urban area — they are not interchangeable just because they share demographics. Some unobserved characteristic (income, lifestyle, opinion intensity) may be both correlated with response and the outcome.

    The assumption underlying post-stratification is "missing at random conditional on the post-stratification variables." This is *unverifiable* from the data. Stronger weighting variables (more covariates, finer cells) generally reduce bias but increase variance — the bias-variance trade-off resurfaces in survey weighting.

---

**Exercise 5.**
The **2016 U.S. presidential polls** systematically underestimated Trump support, especially in rust-belt swing states. Identify two distinct mechanisms that contributed and explain why they were not caught before the election.

??? success "Solution to Exercise 5"
    **Mechanism 1 — differential nonresponse (the "shy Trump voter"):** Trump supporters may have been less willing to answer polls or to admit their preference, especially in regions where Trump support was socially stigmatized. Without an objective indicator, polls treated their lower expressed preference as the truth.

    **Mechanism 2 — education-level weighting:** rust-belt polls under-weighted non-college-educated voters (who responded to phone polls at lower rates and were under-represented after standard demographic weighting). Trump's support among non-college voters was historically high; under-weighting them produced an underestimate.

    **Why not caught:** the 2016 election was the first time these specific dynamics manifested at scale. Polls were calibrated on 2012 patterns where Democratic-Republican response rates and demographic correlations were different. Post-2016, most major pollsters added education-level weighting and adjusted for response biases. The 2020 polls were better — but still overstated Biden's margin in many states, suggesting residual issues remain.

---

**Exercise 6.**
A pharmaceutical study reports a 30% improvement in symptoms for patients who completed the 12-week regimen. About 40% of enrolled patients dropped out before week 12. Why is this conclusion potentially misleading, and how does **intention-to-treat (ITT)** analysis address it?

??? success "Solution to Exercise 6"
    Patients who dropped out are excluded from the "completers" sample. Dropout is non-random: side effects (drug intolerance), lack of perceived benefit, or worsening health are all common reasons. The completers are a *selected* subset — generally those for whom the drug works or who can tolerate it. Reporting only completers' improvement overestimates the drug's effect on the general population that would be prescribed it.

    **Intention-to-treat (ITT) analysis** includes every patient in the group they were originally assigned to, regardless of compliance or completion. Patients who dropped out are kept in the analysis with their last available outcome (or imputed under a specified missing-data model). The estimate is conservative — it dilutes the effect by including patients who didn't fully receive the treatment — but it answers the policy question "what is the expected benefit of prescribing this drug to the patient population?" rather than "how well does this drug work in those who tolerate it for 12 weeks?".

    ITT is the standard for regulatory submissions exactly because it prevents selection bias from inflating reported effects.
