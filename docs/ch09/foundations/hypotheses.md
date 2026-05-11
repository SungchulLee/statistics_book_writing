# Null and Alternative Hypotheses

## Introduction to Hypothesis Testing

Hypothesis testing is a cornerstone of statistical analysis, providing a systematic, data-grounded approach to decision-making. It involves making an assumption about a population parameter and then determining whether the data provide sufficient evidence to reject this assumption. This process is used across various fields to test theories and hypotheses, enabling researchers and analysts to make informed decisions based on empirical data.

### Definition and Purpose

A statistical hypothesis is a claim or assumption about a population parameter, such as the mean, proportion, or standard deviation. **Hypothesis testing** is the formal procedure statisticians use to accept or reject these hypotheses. It is a powerful tool for determining whether we can generalize the evidence from a sample to the broader population, similar to how evidence is used in a criminal trial to reach a verdict.

The primary purposes of hypothesis testing are:

- **To infer about a population**: Hypothesis testing allows statisticians to use data from a sample to make conclusions about an entire population.
- **To determine the statistical significance of evidence**: By evaluating hypotheses, statisticians can determine if observed results in a sample are meaningful in the context of the larger population or simply due to random chance.

This process involves two opposing hypotheses, akin to the defense and prosecution in a courtroom:

1. **Null Hypothesis ($H_0$)**: The null hypothesis is like the presumption of innocence in a trial, assuming that nothing unusual is happening. Statistically, it claims there is no significant effect or difference. For example, in a drug trial, $H_0$ might state that the new drug has no effect compared to a placebo.

2. **Alternative Hypothesis ($H_a$)**: The alternative hypothesis represents the "prosecutor's case." It suggests that there is enough evidence to reject the null hypothesis and claim that there is an effect or difference. In the drug trial analogy, $H_a$ would argue that the new drug has an effect.

Like in a court case, the burden of proof lies with the sample data to provide strong enough evidence to reject $H_0$. Suppose the evidence (data) is not convincing enough. In that case, we accept $H_0$. In statistician's jargon, we fail to reject $H_0$. Much like a "not guilty" verdict does not necessarily prove innocence, but instead that there is insufficient evidence to convict. Conversely, rejecting $H_0$ is akin to a "guilty" verdict, suggesting that the evidence supports $H_a$.

---

## Types of Hypotheses

We classify hypotheses into two main categories:

- **Simple Hypothesis**: A simple hypothesis specifies the population distribution completely, meaning that every population parameter is fully defined. For example, a simple hypothesis might state that the population mean $\mu$ is 5, and the population standard deviation $\sigma$ is 2. In this case, there is no ambiguity about the population parameters.

- **Composite Hypothesis**: On the other hand, a composite hypothesis does not fully specify the population distribution. Instead, it suggests that the population parameter lies within a range of values. For instance, a composite hypothesis might claim $\mu > 5$ without specifying a precise value for $\mu$. This composite hypothesis introduces flexibility, as the hypothesis does not restrict the population to a single set of parameters.

Additionally, depending on the research question and the nature of the hypothesis, tests can be either **directional** (one-tailed) or **non-directional** (two-tailed):

- **One-tailed Test**: We use a one-tailed test when the research hypothesis claims the direction of the effect or relationship between variables. For example, if we hypothesize that a new teaching method leads to higher test scores than the traditional method, we would conduct a one-tailed test to determine whether the mean score of the new method is *greater than* that of the traditional method, i.e.,

$$H_0: \mu \leq \mu_0 \quad \text{versus} \quad H_a: \mu > \mu_0$$

- **Two-tailed Test**: A two-tailed test is appropriate when the research hypothesis does not specify a direction but only claims that the effect exists. This test checks for differences in either direction. For instance, if we are testing whether a new medication has a different effect compared to a placebo, without specifying whether it is better or worse, we would use a two-tailed test to examine whether the population mean differs from the control group's mean, i.e.,

$$H_0: \mu = \mu_0 \quad \text{versus} \quad H_a: \mu \neq \mu_0$$

In summary, the choice between a simple and composite hypothesis depends on whether we fully specify the population parameters or allow flexibility in our assumptions. The decision between one-tailed and two-tailed tests depends on whether we look for effects in a specific direction or test for any difference. Both concepts are crucial for designing valid hypothesis tests that answer specific research questions.

---

## Importance in Statistical Inference

Hypothesis testing plays a critical role in validating research findings. It helps determine whether conclusions from sample data are robust and applicable to a larger population. The hypothesis testing process helps to control the likelihood of incorrectly rejecting a true null hypothesis (Type I error) or failing to reject a false null hypothesis (Type II error). These errors are critical considerations that affect the credibility and replicability of research findings.

- **Type I Error**: This error occurs when the null hypothesis is true but is rejected. It represents a false positive result.
- **Type II Error**: This error occurs when the null hypothesis is false but not rejected. It represents a false negative result.

The choice of significance level ($\alpha$), the threshold for rejecting the null hypothesis, balances these errors. A typical value for $\alpha$ is 0.05, which indicates a 5% risk of committing a Type I error.

---

## Application and Scope

Hypothesis testing is ubiquitous in scientific research, policy-making, medicine, economics, and business, among other fields. It provides a framework for decision-making where stakes are high, and we must make decisions under uncertainty. For instance, in clinical trials, hypothesis testing helps ascertain new drugs' effectiveness. In economics, it can evaluate the impact of policy changes or economic factors on certain variables of interest.

Understanding hypothesis testing is, therefore, essential for students and professionals who engage in data-driven decision-making. It provides the tools needed to test assumptions and validate theories against real-world data.

## Exercises

**Exercise 1.**
Restaurant owner suspects drink machine dispenses too much (target 530 mL). Sample of 30. State hypotheses.

??? success "Solution to Exercise 1"
    $H_0: \mu = 530$ vs $H_1: \mu > 530$ (one-sided, suspect overflow).

    Alternative direction comes from the research question. Null always contains equality. Statement is about the population parameter $\mu$, not the sample mean.

---

**Exercise 2.**
Write hypotheses for: (a) drug reduces blood pressure vs. placebo; (b) coin biased toward heads; (c) two processes have different variances.

??? success "Solution to Exercise 2"
    (a) $H_0: \mu_{\text{drug}} = \mu_{\text{placebo}}$ vs $H_1: \mu_{\text{drug}} < \mu_{\text{placebo}}$ (one-sided reduction).

    (b) $H_0: p = 0.5$ vs $H_1: p > 0.5$ (one-sided).

    (c) $H_0: \sigma_1^2 = \sigma_2^2$ vs $H_1: \sigma_1^2 \ne \sigma_2^2$ (two-sided).

    Directional claim → one-sided; "different/not equal" → two-sided.

---

**Exercise 3.**
**Type I and Type II errors** for the drink machine. Describe each.

??? success "Solution to Exercise 3"
    Type I: reject true $H_0$ — conclude machine overfills when it doesn't. Probability $\alpha$. Cost: unnecessary recalibration.

    Type II: fail to reject false $H_0$ — miss that machine overfills. Probability $\beta$. Cost: continued waste, customer overcharge.

    Trade-off: smaller $\alpha$ → larger $\beta$ (less power). Setting $\alpha$ requires weighing costs.

---

**Exercise 4.**
**One-sided vs two-sided.** When appropriate? Cost of one-sided?

??? success "Solution to Exercise 4"
    One-sided: only one direction is of interest. More power in that direction (uses full $\alpha$ on one tail).

    Two-sided: any departure matters. Critical value $z_{\alpha/2}$ — slightly harder to reject.

    **Cost of one-sided:** if truth is in the unchosen direction, you'll never detect it. Critical in safety contexts.

    Modern default: two-sided unless prior knowledge rules out one direction. Journals often require justification for one-sided.

---

**Exercise 5.**
**Why $\alpha = 0.05$?** When to depart?

??? success "Solution to Exercise 5"
    Convention from Fisher (1925) — not a deep justification. "1 in 20" is psychologically manageable.

    Depart when:

    - High-stakes (drug approval): $\alpha = 0.01$ or smaller.
    - Multiple testing: $\alpha/m$ (Bonferroni).
    - Exploratory: $\alpha = 0.10$ to find weak signals.
    - Physics "discovery": $5\sigma$ ($\alpha \approx 6 \times 10^{-7}$).

    2019 ASA statement: don't blindly use 0.05. Report exact p-values, effect sizes, CIs.

---

**Exercise 6.**
**Hypothesis test vs CI.** When to use each?

??? success "Solution to Exercise 6"
    Test answers: does data contradict $H_0$?
    CI answers: what's the plausible range of $\theta$?

    Mathematical equivalence: two-sided test at level $\alpha$ rejects $H_0: \theta = \theta_0$ iff $\theta_0 \notin$ $(1-\alpha)$ CI.

    Use test for specific hypotheses (regulatory, science). Use CI for reporting uncertainty. **Modern preference: CIs primary, tests supplementary.**

    Effect size + CI is more informative than a binary "significant or not."
