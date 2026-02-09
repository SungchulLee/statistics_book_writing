# Two-Way ANOVA: Interaction Effects

## 1. Procedure for Conducting Two-Way ANOVA

[kor|](https://www.youtube.com/watch?v=i4NHIGvTB-g) [eng|](https://www.youtube.com/playlist?list=PLWtoq-EhUJe2TjJYfZUQtuq7a0dQCnOWp) [wiki|](https://en.wikipedia.org/wiki/Two-way_analysis_of_variance)

### Step 1: Formulate the Hypotheses

- For **Main Effect A**:
    - $H_0: \mu^A_1=\mu^A_2=\cdots=\mu^A_a$
    - $H_A$: At least one mean is different across the levels of Factor A.
- For **Main Effect B**:
    - $H_0: \mu^B_1=\mu^B_2=\cdots=\mu^B_b$
    - $H_A$: At least one mean is different across the levels of Factor B.
- For **Interaction Effect (A x B)**:
    - $H_0: (\mu_{ij} - \mu_{i\cdot} - \mu_{\cdot j} + \mu_{\cdot \cdot}) = 0$ for all $i$ and $j$
    - $H_A$: There is an interaction between Factor A and Factor B.

where $\mu_{ij}$ is the mean at level $i$ of Factor A and level $j$ of Factor B, $\mu_{i\cdot}$ is the marginal mean of Factor A at level $i$, $\mu_{\cdot j}$ is the marginal mean of Factor B at level $j$, and $\mu_{\cdot \cdot}$ is the overall mean.

### Step 2: Calculate the Overall Mean and Group Means

$$ \bar{y}_{\cdot\cdot\cdot} = \frac{1}{abc}\sum_{i=1}^{a}\sum_{j=1}^{b}\sum_{k=1}^{c} y_{ijk} $$

$$\begin{array}{lll}
\bar{y}_{i\cdot\cdot}&=&\displaystyle \frac{1}{bc}\sum_{j=1}^{b}\sum_{k=1}^{c} y_{ijk}\\
\bar{y}_{\cdot j\cdot}&=&\displaystyle \frac{1}{ac}\sum_{i=1}^{a}\sum_{k=1}^{c} y_{ijk}\\
\bar{y}_{i j\cdot}&=&\displaystyle \frac{1}{k}\sum_{k=1}^{c} y_{ijk}\\
\end{array}$$

### Step 3: Calculate Total Sum of Squares SST

$$SST =\sum_{i=1}^{a}\sum_{j=1}^{b}\sum_{k=1}^{c} \left( y_{ijk} - \bar{y}_{\cdot\cdot\cdot} \right)^2$$

### Step 4: Calculate SSA and SSB

$$\begin{array}{lll}
SSA&=&\displaystyle \sum_{i=1}^{a} bc \left( \bar{y}_{i\cdot\cdot} - \bar{y}_{\cdot\cdot\cdot} \right)^2\\
SSB&=&\displaystyle \sum_{j=1}^{b} ac \left( \bar{y}_{\cdot j\cdot} - \bar{y}_{\cdot\cdot\cdot} \right)^2\\
\end{array}$$

### Step 5: Calculate Interaction Variation SSAB

$$SSAB = \sum_{i=1}^{a} \sum_{j=1}^{b} c \left( \bar{y}_{ij\cdot} - \bar{y}_{i\cdot\cdot} - \bar{y}_{\cdot j\cdot} + \bar{y}_{\cdot\cdot\cdot} \right)^2$$

### Step 6: Calculate Residual Variation SSE

$$SSE = \sum_{i=1}^{a} \sum_{j=1}^{b} \sum_{k=1}^{c} \left( y_{ijk} - \bar{y}_{ij\cdot} \right)^2$$

### Step 7: Check the Partition

$$SST = SSA + SSB + SSAB + SSE$$

### Step 8: Calculate the F-Statistic

$$\begin{array}{cccccccccc}
\text{Factor}&\text{df}&SS&MS&F&H_0&\text{Sampling Dist under $H_0$}\\
\hline
\text{Factor A}&a-1&SSA&MSA=\frac{SSA}{a-1}&F_A=\frac{MSA}{MSE}&\text{all }\beta^{(1)}_{i}=0&F_A\sim F_{a-1,ab(c-1)}\\
\text{Factor B}&b-1&SSB&MSB=\frac{SSB}{b-1}&F_B=\frac{MSB}{MSE}&\text{all }\beta^{(2)}_{j}=0&F_B\sim F_{b-1,ab(c-1)}\\
\text{Interaction}&(a-1)(b-1)&SSAB&MSAB=\frac{SSAB}{(a-1)(b-1)}&F_{AB}=\frac{MSAB}{MSE}&\text{all }\beta_{ij}=0&F_{AB}\sim F_{(a-1)(b-1),ab(c-1)}\\
\text{Error}&ab(c-1)&SSE&MSE=\frac{SSE}{ab(c-1)}&\\
\hline
\text{Total}&abc-1&SST&&\\
\end{array}$$

### Step 9: Determine the Critical Value or P-Value

Compare the F-statistics to the critical values from the F-distribution table, or use p-values.

### Step 10: Make Decisions

If the F-statistic for a main effect or interaction is greater than the critical value (or if the p-value is less than the chosen significance level), reject the null hypothesis for that effect.

## 2. Two-Way ANOVA Packages

### statsmodels: Interaction Plot and ANOVA

```python
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.factorplots import interaction_plot

def load_data():
    url = 'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/ToothGrowth.csv'
    df = pd.read_csv(url, usecols=[1, 2, 3])
    return df

def plot_interaction(df):
    fig, ax = plt.subplots(figsize=(12, 3))
    interaction_plot(df.dose, df.supp, df.len,
                     colors=['red', 'blue'],
                     markers=['*', 'P'],
                     markersize=7, ax=ax,
                     legendloc='lower right',
                     linestyles=["--", "--"])
    ax.set_title("Interaction Plot: Dose vs. Supplement on Tooth Length")
    ax.set_xlabel("Dose Level")
    ax.set_ylabel("Tooth Length")
    plt.show()

def perform_two_way_anova(df):
    model = ols('len ~ C(supp) + C(dose) + C(supp):C(dose)', data=df).fit()
    anova_results = anova_lm(model)
    print("\nTwo-Way ANOVA Results:")
    print(anova_results)

df = load_data()
plot_interaction(df)
perform_two_way_anova(df)
```

### Output Interpretation

**Interaction Plot**: As the dose level increases from 0.5 to 2.0, tooth length generally increases for both supplements. At lower dose levels (0.5 and 1.0), "OJ" results in noticeably higher tooth length than "VC". At the highest dose level (2.0), the difference is much smaller. The non-parallel lines suggest a potential **interaction effect**.

**ANOVA Results Table**:

| Term               | df  | sum_sq    | mean_sq   | F          | PR(>F)         |
|--------------------|-----|-----------|-----------|------------|----------------|
| **C(supp)**        | 1.0 | 205.350   | 205.350   | 15.572     | 2.31e-04       |
| **C(dose)**        | 2.0 | 2426.434  | 1213.217  | 91.999     | 4.05e-18       |
| **C(supp):C(dose)**| 2.0 | 108.319   | 54.160    | 4.107      | 2.19e-02       |
| **Residual**       | 54.0| 712.106   | 13.187    | NaN        | NaN            |

Both **supplement type** and **dose level** have statistically significant effects on tooth length. The interaction between them is also significant, suggesting that the effectiveness of each supplement varies depending on the dose level.

### Post-Hoc with Tukey's HSD

```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Step 1: Two-Way ANOVA
url = 'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/ToothGrowth.csv'
df = pd.read_csv(url, usecols=[1, 2, 3])
model = ols('len ~ C(supp) + C(dose) + C(supp):C(dose)', data=df).fit()
anova_results = anova_lm(model)
print(anova_results, end="\n\n")

# Step 2: Tukey's HSD for Main Effects
tukey_dose = pairwise_tukeyhsd(endog=df['len'], groups=df['dose'], alpha=0.05)
print(tukey_dose, end="\n\n")

tukey_supp = pairwise_tukeyhsd(endog=df['len'], groups=df['supp'], alpha=0.05)
print(tukey_supp, end="\n\n")

# Step 3: Post-Hoc for Interaction Effect
df['supp_dose'] = df['supp'].astype(str) + "_" + df['dose'].astype(str)
tukey_interaction = pairwise_tukeyhsd(endog=df['len'], groups=df['supp_dose'], alpha=0.05)
print(tukey_interaction, end="\n\n")
```

**Key findings from interaction post-hoc**: At lower doses, OJ tends to produce significantly longer tooth growth than VC. At the highest dose (2.0), both supplements are similarly effective (OJ_2.0 vs VC_2.0: meandiff = 0.08, p = 1.0).

## 3. Example: Test Score Based On Teaching Method And Study Time

### Question

Two factors on test scores: **Factor A (Teaching Method)** with levels Traditional and Online, **Factor B (Study Time)** with levels 1 Hour and 2 Hours.

| Teaching Method | Study Time | Score 1 | Score 2 | Average |
|-----------------|------------|---------|---------|---------|
| Traditional     | 1 Hour     | 60      | 62      | 61      |
| Traditional     | 2 Hours    | 68      | 70      | 69      |
| Online          | 1 Hour     | 65      | 63      | 64      |
| Online          | 2 Hours    | 72      | 74      | 73      |

### Step 1: Calculate the Grand Mean

$$\bar{X} = \frac{534}{8} = 66.75$$

### Step 2: Calculate Factor and Interaction Means

- Traditional: 65, Online: 68.5
- 1 Hour: 62.5, 2 Hours: 71
- Cell means: Traditional/1Hr = 61, Traditional/2Hr = 69, Online/1Hr = 64, Online/2Hr = 73

### Step 3: Calculate Sum of Squares

- $SS_{\text{Total}} = 177.5$
- $SS_A = 4 \times ((65 - 66.75)^2 + (68.5 - 66.75)^2) = 24.5$
- $SS_B = 4 \times ((62.5 - 66.75)^2 + (71 - 66.75)^2) = 144.5$
- $SS_{AB} = 0.5$ (each cell contributes 0.125)
- $SS_E = 177.5 - 24.5 - 144.5 - 0.5 = 8$

### Step 4: Degrees of Freedom, Mean Squares, and F-Statistics

| Source         | SS     | df | MS      | F      | PR(>F)   |
|----------------|--------|----|---------|--------|----------|
| Factor A       | 24.5   | 1  | 24.5    | 12.25  | 0.024896 |
| Factor B       | 144.5  | 1  | 144.5   | 72.25  | 0.001051 |
| Interaction AB | 0.5    | 1  | 0.5     | 0.25   | 0.643330 |
| Error          | 8.0    | 4  | 2.0     |        |          |
| Total          | 177.5  | 7  |         |        |          |

### Interpretation

- **Factor A (Teaching Method)**: $F_A = 12.25$, $p = 0.0249$, significant at 0.05. Teaching method has a significant effect on test scores.
- **Factor B (Study Time)**: $F_B = 72.25$, $p = 0.0011$, significant at 0.01. Study time has a significant effect on test scores.
- **Interaction (A x B)**: $F_{AB} = 0.25$, $p = 0.6433$, not significant. No significant interaction between teaching method and study time.

### Python Implementation

```python
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

data = {
    'Teaching_Method': ['Traditional', 'Traditional', 'Traditional', 'Traditional',
                        'Online', 'Online', 'Online', 'Online'],
    'Study_Time': ['1 Hour', '1 Hour', '2 Hours', '2 Hours',
                   '1 Hour', '1 Hour', '2 Hours', '2 Hours'],
    'Score': [60, 62, 68, 70, 65, 63, 72, 74]
}
df = pd.DataFrame(data)

model = ols('Score ~ C(Teaching_Method) + C(Study_Time) + C(Teaching_Method):C(Study_Time)', data=df).fit()
anova_results = anova_lm(model)
print("Two-Way ANOVA Results:")
print(anova_results)
```

### R Code

```r
# Load necessary libraries
library(dplyr)
library(stats)

data <- data.frame(
  Teaching_Method = factor(c('Traditional', 'Traditional', 'Traditional', 'Traditional',
                             'Online', 'Online', 'Online', 'Online')),
  Study_Time = factor(c('1 Hour', '1 Hour', '2 Hours', '2 Hours',
                        '1 Hour', '1 Hour', '2 Hours', '2 Hours')),
  Score = c(60, 62, 68, 70, 65, 63, 72, 74)
)

model <- aov(Score ~ Teaching_Method + Study_Time + Teaching_Method:Study_Time, data = data)
anova_results <- summary(model)
print("Two-Way ANOVA Results:")
print(anova_results)
```

### R Output Interpretation

The R output provides the same ANOVA table with Df, Sum Sq, Mean Sq, F value, and Pr(>F) columns. The significance codes (`*` for p < 0.05, `**` for p < 0.01) confirm:

- Both `Teaching_Method` and `Study_Time` have statistically significant main effects on `Score`.
- The interaction between `Teaching_Method` and `Study_Time` is not statistically significant, suggesting that the effect of teaching method on scores does not depend on study time.
