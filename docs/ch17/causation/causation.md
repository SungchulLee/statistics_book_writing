# 18.5 Causation

## Understanding Causation

Causation is a fundamental concept in statistics, science, and everyday reasoning. It refers to a relationship where one event (the cause) directly produces or influences another event (the effect). While correlations can suggest a relationship, causation goes further by implying that changes in one variable are **responsible** for changes in another.

Understanding causation is crucial in fields like medicine, economics, and social sciences, where the goal is often to identify and influence factors that lead to specific outcomes.

---

## Correlation vs. Causation

It is essential to distinguish between correlation and causation. While correlation indicates that two variables move together, it does not imply that one causes the other. For example, ice cream sales and drowning incidents both increase during summer, but eating ice cream does not cause drowning. This illustrates the classic principle: **correlation does not imply causation**.

Causation requires more rigorous evidence. Establishing a causal relationship often involves controlled experiments, longitudinal studies, or other methods that can rule out confounding variables.

### Correlation (or Association) Does Not Imply Causation

See: [Correlation vs Causation in Data Science](https://sundaskhalid.medium.com/correlation-vs-causation-in-data-science-66b6cfa702f0)

**Khan Academy Reference**: [Invalid Conclusions from Studies Example](https://www.khanacademy.org/math/ap-statistics/gathering-data-ap/sampling-observational-studies/v/invalid-conclusions-studies-example)

---

## Criteria for Causation

Several criteria help determine whether a causal relationship exists between two variables:

1. **Temporal Precedence**: The cause must occur before the effect. If a study claims that smoking causes lung cancer, smoking must precede the diagnosis of lung cancer in study participants.

2. **Covariation of Cause and Effect**: The cause and effect must show a consistent relationship. Changes in the cause should be associated with changes in the effect. Higher levels of physical activity should consistently be associated with lower rates of heart disease if physical activity is indeed protective.

3. **Elimination of Confounding Variables**: Confounding variables are external factors that can affect both the cause and the effect. Controlling for these variables is essential to establish a direct link.

4. **Plausibility**: There should be a scientifically reasonable mechanism or explanation for the causal relationship. The causal link between smoking and lung cancer is supported by biological evidence that tobacco smoke contains carcinogens.

5. **Experimental Evidence**: Controlled experiments provide the strongest evidence for causation. Randomized controlled trials (RCTs) are the gold standard in clinical research because they minimize biases and confounding factors.

---

## Challenges in Establishing Causation

- **Confounding Variables**: Identifying and controlling for confounders is one of the biggest challenges. Confounders can obscure the true relationship between variables, leading to incorrect conclusions.

- **Reverse Causality**: Sometimes the direction of causality is unclear. Poor health might lead to lower income, but lower income can also cause poor health. Establishing which factor is the cause requires careful study design.

- **Ethical and Practical Constraints**: Conducting experiments to establish causation is often impractical or unethical. It would be unethical to expose people to harmful substances to observe the effects. In such cases, researchers rely on observational studies and sophisticated statistical methods to infer causation.

---

## Example: Smoking and Lung Cancer — A Landmark Study

The causal link between smoking and lung cancer is one of the most well-documented examples in medical research.

In the 1950s, British researchers Richard Doll and Austin Bradford Hill conducted a landmark **cohort study** involving thousands of British doctors. The study followed the doctors over several years, collecting data on their smoking habits and health outcomes. The results showed that doctors who smoked had a significantly higher incidence of lung cancer compared to non-smokers.

To rule out confounding variables, the researchers controlled for factors such as age, gender, and socioeconomic status. The study design ensured that smoking preceded the development of lung cancer, and consistent findings across multiple studies and populations provided strong evidence for a causal link.

This research contributed to public health campaigns and policy changes aimed at reducing smoking rates, ultimately saving millions of lives.

---

## Example: Vaccines and Disease Eradication — The Case of Polio

Polio was once a widespread and devastating disease, causing paralysis and death in millions of people worldwide. In the mid-20th century, the introduction of the polio vaccine marked a turning point.

Public health officials conducted extensive vaccination campaigns, and the incidence of polio dropped dramatically in countries where the vaccine was administered. The causal link was established through rigorous scientific research, including **randomized controlled trials** that demonstrated vaccinated individuals were significantly less likely to contract polio compared to unvaccinated individuals.

The success of the polio vaccine led to widespread immunization efforts, ultimately bringing the world close to eradicating the disease. This example underscores the importance of establishing causation to guide effective public health interventions.

---

## Example: Double-Blind Tests and Drug Efficacy

One of the most rigorous methods for establishing causation in medicine is the **double-blind test**. In this design, neither the participants nor the researchers know who is receiving the treatment or the placebo. This eliminates bias and ensures that observed effects can be attributed directly to the treatment.

**Example**: Consider a clinical trial testing a new blood pressure drug. Participants are randomly assigned to two groups: one receives the actual drug, the other receives a placebo. After several months, the drug group shows a significant reduction in blood pressure compared to the placebo group. The double-blind design ensures that this reduction is caused by the drug, not by expectations or other confounding factors.

---

## Example: FDA Three-Phase Drug Approval Process

The U.S. Food and Drug Administration (FDA) uses a stringent three-phase process to establish the safety and efficacy of new drugs:

- **Phase 1**: A small group of healthy volunteers (20–100) receives the drug to assess safety, dosage range, and side effects.

- **Phase 2**: Several hundred participants who have the target condition receive the drug to determine efficacy and further evaluate safety.

- **Phase 3**: Thousands of participants provide a larger sample size to confirm efficacy, monitor side effects, and compare to existing treatments. **Randomized controlled trials** are used to minimize confounding variables and biases.

**Example**: A new vaccine must pass all three phases. In Phase 3, a large-scale RCT randomly assigns participants to receive either the vaccine or a placebo. If the vaccine significantly reduces infection incidence compared to the placebo, a causal relationship is established. Only after successfully passing all three phases does the FDA approve the vaccine for public use.

---

## Example: Education and Income — Longitudinal Studies

Longitudinal studies, which follow individuals over time, are essential for establishing causation in social sciences.

A long-term study tracks students from various socioeconomic backgrounds over several decades, collecting data on educational attainment, job history, and income. The analysis reveals that individuals who obtain higher levels of education consistently earn more over their lifetimes.

To establish causation, researchers control for confounding variables such as family background, intelligence, and early life opportunities. By showing that higher education levels lead to increased earning potential while holding these factors constant, the research provides strong evidence for a causal relationship between education and income.

---

## Summary

Understanding causation is essential for making informed decisions and advancing scientific knowledge. While correlation can suggest a relationship between variables, establishing causation requires more rigorous evidence and careful consideration of:

- Confounding factors
- Reverse causality
- Temporal precedence
- Experimental design

By applying the criteria for causation and using appropriate research methods—including RCTs, longitudinal studies, and statistical controls—we can better understand the causes of various phenomena and develop effective interventions in health, economics, and social policy.
