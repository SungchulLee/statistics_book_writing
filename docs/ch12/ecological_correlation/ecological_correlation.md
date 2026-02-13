# 18.2 Ecological Correlation

Ecological correlation refers to the statistical relationship between variables that are **aggregated over groups or regions** rather than being measured at the individual level. The phenomenon where an ecological correlation misrepresents the relationship at the individual level is known as the **ecological fallacy**.

---

## Understanding Ecological Correlation

Ecological correlation often arises in studies that use data aggregated by regions, countries, or other groupings. For instance, researchers may study the relationship between average income and average health outcomes across different countries. Although such studies can reveal important trends, the correlation observed at the group level may not reflect the true relationship at the individual level.

**Example**: If you analyze the average education level and average income across different cities, you might find a strong positive correlation, indicating that cities with higher average education levels tend to have higher average incomes. However, this does not necessarily imply that individuals within each city with higher education levels earn more than those with lower education levels.

---

## Simpson's Paradox

Simpson's paradox occurs when a trend that appears in several different groups of data reverses or disappears when these groups are combined. It is one of the most striking illustrations of why aggregate data can be misleading.

### Example 1: Derek Jeter vs. David Justice (Baseball)

| | 1995 | 1996 | Combined |
|---|---|---|---|
| **Derek Jeter** | 12/48 = **.250** | 183/582 = **.314** | 195/630 = **.310** |
| **David Justice** | 104/411 = **.253** | 45/140 = **.321** | 149/551 = **.270** |

David Justice had a higher batting average than Derek Jeter in **both** 1995 and 1996 individually, yet Jeter had a higher **combined** average. The paradox arises because Jeter had far more at-bats in 1996 (his better year), while Justice had more at-bats in 1995 (his worse year). The unequal weighting across years reverses the overall comparison.

### Example 2: Kidney Stone Treatment

| | Small Stones | Large Stones | Combined |
|---|---|---|---|
| **Treatment A** | 81/87 = **.93** | 192/263 = **.73** | 273/350 = .78 |
| **Treatment B** | 234/270 = .87 | 55/80 = .69 | 289/350 = **.83** |

Treatment A is better for both small and large stones individually, yet Treatment B appears better when the data is combined. The confounding factor is that Treatment A was preferentially assigned to the more difficult (large stone) cases, dragging down its combined rate.

### Example 3: UC Berkeley Graduate Admissions Gender Bias

| Major | Men Applicants | Men % Admitted | Women Applicants | Women % Admitted |
|-------|---------------|---------------|-----------------|-----------------|
| A | 825 | 62% | 108 | **82%** |
| B | 560 | 63% | 25 | **68%** |
| C | 325 | **37%** | 593 | 34% |
| D | 417 | 33% | 375 | **35%** |
| E | 191 | **28%** | 393 | 24% |
| F | 373 | 6% | 341 | **7%** |

```python
import pandas as pd

def main():
    major = ["A", "B", "C", "D", "E", "F"]
    n_male = [825, 560, 325, 417, 191, 373]
    p_male = [0.62, 0.63, 0.37, 0.33, 0.28, 0.06]
    n_female = [108, 25, 593, 375, 393, 341]
    p_female = [0.82, 0.68, 0.34, 0.35, 0.24, 0.07]

    data = {
        "major": major, "n_male": n_male, "p_male": p_male,
        "n_female": n_female, "p_female": p_female
    }
    df = pd.DataFrame(data).set_index("major")

    prob_male_admitted = (df.n_male * df.p_male).sum() / df.n_male.sum()
    prob_female_admitted = (df.n_female * df.p_female).sum() / df.n_female.sum()

    print(f"Overall male admission rate:   {prob_male_admitted:.2%}")   # ~44%
    print(f"Overall female admission rate: {prob_female_admitted:.2%}") # ~30%

if __name__ == "__main__":
    main()
```

At the aggregate level, men appear to be admitted at a significantly higher rate (44% vs. 30%). However, examining each department individually reveals that women were admitted at equal or higher rates in most departments. The paradox arises because women disproportionately applied to more competitive departments (C, D, E, F) with lower overall admission rates.

---

## Real-World Examples of Ecological Correlation

### Alcohol Consumption and Coronary Heart Disease

An ecological study examined the relationship between per capita alcohol consumption and death rates from coronary heart disease (CHD) across different countries. The analysis revealed a notably strong **negative** correlation at the country level.

However, a cohort study analyzing **individual** alcohol consumption revealed a **J-shaped** relationship: individuals who drank moderately had lower mortality rates compared to abstainers, but as consumption increased beyond moderate levels, there was a significant linear rise in mortality.

The ecological correlation (linear, negative) dramatically oversimplifies the true individual-level relationship (non-linear, J-shaped).

Source: [Ecological Studies — Boston University](https://sphweb.bumc.bu.edu/otlt/MPH-Modules/PH717-QuantCore/PH717-Module1B-DescriptiveStudies_and_Statistics/PH717-Module1B-DescriptiveStudies_and_Statistics6.html)

### Smoking and Lung Cancer

Countries with higher average cigarette consumption per capita tend to have higher lung cancer rates—a strong positive ecological correlation. However, assuming this directly reflects individual risk commits the ecological fallacy. In a country with high cigarette consumption, other factors (like air pollution or occupational exposures) may also contribute to high lung cancer rates.

### Cholesterol Levels and Heart Disease Across Countries

Countries with higher average cholesterol levels tend to have higher rates of heart disease (positive ecological correlation). But this does not necessarily mean that individuals with higher cholesterol in those countries are more likely to develop heart disease. Dietary habits, healthcare access, genetic predispositions, and lifestyle differences across countries can influence both variables.

### Education and Crime Rates in Urban Areas

Neighborhoods with higher average education levels often have lower crime rates (negative ecological correlation). However, assuming this correlation applies to individuals within each neighborhood can be misleading. Economic opportunities, policing, community programs, and social cohesion may be influencing both education levels and crime rates at the neighborhood level.

### Voting Patterns and Income Levels

Wealthier districts might show a tendency to vote for conservative candidates, leading to an ecological correlation between income and conservative voting behavior. But individuals with lower incomes in those districts might also vote conservatively, and some wealthy individuals might vote for liberal candidates. The ecological correlation may reflect broader regional trends rather than a direct individual-level relationship.

### Income and Health Outcomes

Regions with higher average incomes tend to have better health outcomes and longer life expectancies (positive ecological correlation). However, individuals in lower-income regions might still enjoy good health due to strong community health programs, and individuals in higher-income regions might face health challenges due to stress or lifestyle choices.

### Literacy Rates and Economic Development

Countries with higher literacy rates tend to have higher GDP per capita (positive ecological correlation). But this does not mean that increasing literacy alone will lead to higher individual incomes. Economic development is a complex process influenced by government policies, natural resources, trade, and industrialization.

---

## The Ecological Fallacy

The ecological fallacy occurs when conclusions about **individual behavior** are drawn from **aggregate data**. This fallacy can lead to incorrect inferences because the correlation observed in group-level data does not necessarily apply to individuals within those groups.

**Key Mechanisms**:

1. **Aggregation bias**: Averaging across a group removes individual-level variation, potentially creating or masking associations.
2. **Confounding at the group level**: Group-level variables may be correlated with other group characteristics that drive the observed pattern.
3. **Unequal weighting**: As Simpson's paradox demonstrates, the composition of groups can reverse overall trends.

---

## Implications and Caution

Ecological correlations are useful for identifying broad trends and generating hypotheses for further study, but they should be interpreted with caution:

1. **Do not assume individual-level relationships** from group-level data without additional evidence.
2. **Complement ecological studies** with individual-level data analysis whenever possible.
3. **Be aware of Simpson's paradox** — always check whether aggregate trends hold within meaningful subgroups.
4. **Consider confounders** at both the group and individual level.

---

## Summary

These examples highlight the importance of understanding the context when interpreting ecological correlations. While such correlations can provide valuable insights into group-level patterns, they should not be assumed to apply directly to individuals within those groups. To avoid the ecological fallacy, it is crucial to consider additional data and analysis at the individual level whenever possible.
