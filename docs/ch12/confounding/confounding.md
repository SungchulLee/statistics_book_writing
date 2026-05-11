# 18.4 Correlation, Causation, and Confounding


## Correlation vs. Causation

**Correlation** measures the strength and direction of a linear relationship between two variables, quantified by the correlation coefficient ($-1$ to $1$). **Causation** implies that a change in one variable directly causes a change in another.

**Key Distinction**: Correlation alone does not imply causation. Two variables might be correlated due to a direct causal relationship, a reverse causal relationship, or because of an underlying third variable influencing both.

---

## The Role of Confounding

A **confounder** is an extraneous variable that is related to both the independent variable and the dependent variable. It can create a false impression of a relationship between them, thereby misleading interpretations of causality.

**Classic Example — Ice Cream and Drowning**:
There is a correlation between ice cream sales and drowning incidents. Initially, it might seem like eating ice cream causes drowning. The true confounder is **temperature**: higher temperatures increase both ice cream sales and swimming activity, which leads to more drowning incidents. The correlation between ice cream sales and drowning is spurious, driven by the confounding effect of temperature.

### Identifying and Addressing Confounding

1. **Control Variables**: In statistical analyses, researchers use techniques like multiple regression to control for confounding variables and isolate the effect of the independent variable on the dependent variable.

2. **Experimental Design**: Randomized controlled trials (RCTs) mitigate confounding by randomly assigning subjects to groups, balancing out confounder effects.

3. **Longitudinal Studies**: Tracking variables over time helps observe how changes in one variable affect another, helping to establish causality while accounting for confounding.

---

## Example: Education and Income

- **Observation**: A study finds a strong correlation between higher levels of education and higher income.
- **Initial Interpretation**: Higher education causes higher income.
- **Confounding Factor**: **Socioeconomic background** — individuals from higher socioeconomic backgrounds may have access to better education and also tend to have higher incomes.
- **How to Address**: Control for socioeconomic status in the analysis. Research controlling for family background shows that while education has a positive impact on income, the effect is moderated by the family's socioeconomic status.

---

## Example: Vitamin Intake and Cancer Risk

- **Observation**: People who take vitamin supplements have a lower risk of cancer.
- **Initial Interpretation**: Vitamin supplements reduce cancer risk.
- **Confounding Factor**: **Health-conscious behavior** — individuals who take vitamins are often more health-conscious overall, engaging in other healthy behaviors (exercise, balanced diet, avoiding smoking).
- **How to Address**: Control for other health behaviors to isolate the effect of vitamin intake on cancer risk. Studies that account for overall health behaviors often find that the relationship between vitamins alone and cancer risk is not as strong as initially observed.

---

## Example: Coffee Consumption and Heart Disease

- **Observation**: High coffee consumption correlates with increased incidence of heart disease.
- **Initial Interpretation**: Coffee causes heart disease.
- **Confounding Factor**: **Smoking** — coffee drinkers may be more likely to smoke, which is a known risk factor for heart disease.
- **How to Address**: Control for smoking in the analysis to determine whether coffee consumption independently affects heart disease risk.

---

## Example: Exercise and Weight Loss

- **Observation**: People who exercise regularly tend to lose weight.
- **Initial Interpretation**: Exercise causes weight loss.
- **Confounding Factor**: **Diet** — individuals who exercise regularly might also follow healthier diets.
- **How to Address**: Control for dietary habits (caloric intake and dietary quality). When controlled, the relationship between exercise and weight loss becomes more clearly defined, showing that both exercise and diet play significant roles.

---

## Example: Job Performance and Salary

- **Observation**: High-performing employees earn higher salaries.
- **Initial Interpretation**: Better job performance leads to higher salaries.
- **Confounding Factor**: **Experience and tenure** — employees with more experience may both perform better and receive higher salaries due to seniority.
- **How to Address**: Control for years of experience and length of employment to accurately assess the direct impact of performance on salary.

---

## Example: Ice Cream Sales and Drowning Incidents

- **Observation**: There is a correlation between higher ice cream sales and increased drowning incidents.
- **Initial Interpretation**: Eating ice cream may be associated with a higher risk of drowning.
- **Confounding Factor**: **Temperature** — both variables increase during hot weather.
- **How to Address**: By analyzing and controlling for temperature, the apparent correlation between ice cream sales and drowning incidents disappears, revealing that the real link is with temperature, not ice cream.

---

## Example: Housing Data

The California housing dataset provides a practical example of exploring correlations and potential confounding relationships.

```python
import os
import tarfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path)

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def plot_housing_data(df):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 4))

    ax0.plot(df.median_income, df.median_house_value, ',')
    ax0.set_xlabel("Median Income")

    ax1.plot(df.population, df.median_house_value, ',')
    ax1.set_xlabel("Population")

    ax2.plot(df.housing_median_age, df.median_house_value, ',')
    ax2.set_xlabel("Housing Median Age")

    for ax in (ax0, ax1, ax2):
        ax.set_ylabel("Median House Value")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

def main():
    fetch_housing_data()
    df = load_housing_data()
    plot_housing_data(df)

if __name__ == "__main__":
    main()
```

### Exercise 1: Calculate Correlation Coefficients

**Objective**: Calculate Pearson correlation coefficients between `median_house_value` and other variables.

```python
import pandas as pd

def calculate_correlations(df):
    income_corr = df['median_income'].corr(df['median_house_value'])
    population_corr = df['population'].corr(df['median_house_value'])
    age_corr = df['housing_median_age'].corr(df['median_house_value'])

    print(f"Correlation (Median Income vs House Value):      {income_corr:.4f}")
    print(f"Correlation (Population vs House Value):         {population_corr:.4f}")
    print(f"Correlation (Housing Median Age vs House Value): {age_corr:.4f}")

if __name__ == "__main__":
    df = load_housing_data()
    calculate_correlations(df)
```

### Exercise 2: Correlation Heatmap

**Objective**: Visualize the correlations between all numerical variables.

```python
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of Housing Data")
    plt.show()

if __name__ == "__main__":
    df = load_housing_data()
    plot_correlation_heatmap(df)
```

### Exercise 3: Scatter Plot Matrix

**Objective**: Create a pair plot to visualize pairwise relationships.

```python
import seaborn as sns
import matplotlib.pyplot as plt

def plot_scatter_matrix(df):
    selected = ['median_income', 'median_house_value', 'population', 'housing_median_age']
    sns.pairplot(df[selected], diag_kind="kde")
    plt.suptitle("Scatter Plot Matrix of Selected Housing Variables", y=1.02)
    plt.show()

if __name__ == "__main__":
    df = load_housing_data()
    plot_scatter_matrix(df)
```

### Exercise 4: Impact of Outliers on Correlation

**Objective**: Explore how outliers affect correlation.

```python
import pandas as pd
import matplotlib.pyplot as plt

def add_outliers(df):
    outliers = pd.DataFrame({
        'median_income': [20, 22],
        'median_house_value': [1000000, 1200000]
    })
    df_with_outliers = pd.concat([df, outliers], ignore_index=True)

    original_corr = df['median_income'].corr(df['median_house_value'])
    new_corr = df_with_outliers['median_income'].corr(df_with_outliers['median_house_value'])

    print(f"Original Correlation: {original_corr:.4f}")
    print(f"New Correlation with Outliers: {new_corr:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(df['median_income'], df['median_house_value'], 'o', label='Original Data')
    plt.plot(outliers['median_income'], outliers['median_house_value'], 'ro', label='Outliers')
    plt.xlabel('Median Income')
    plt.ylabel('Median House Value')
    plt.legend()
    plt.title("Impact of Outliers on Correlation")
    plt.show()

if __name__ == "__main__":
    df = load_housing_data()
    add_outliers(df)
```

### Exercise 5: Correlation vs. Causation Discussion

**Objective**: Discuss why a high correlation between `median_income` and `median_house_value` does not necessarily imply causation.

**Discussion Points**:

- **Confounding variables**: Location desirability, job opportunities, and educational facilities might influence both income and house values.
- **Reverse causality**: Do higher incomes drive up house values, or do expensive areas attract higher-income residents?
- **Third variable problem**: Regional economic development, local policies, and geographic factors can affect both variables simultaneously.

---

## Summary

Confounding variables can obscure the true nature of relationships between variables, leading to misleading conclusions about causation. By identifying and controlling for confounders, researchers can better isolate and understand the true effects of the variables of interest.

---

## Additional Exercises

### Exercise: Identifying Confounders

Present research findings with apparent correlations (e.g., screen time and sleep quality, social media usage and mental health). Identify potential confounding variables and discuss how they might be controlled.

### Exercise: Designing an Experiment

Given a correlation between reading books and academic performance, design an experiment to test the causal relationship, controlling for confounders such as socioeconomic status and prior academic performance.

### Exercise: Analyzing Real Data

Using a dataset with variables like exercise, diet, weight loss, and stress levels, perform a correlation analysis and then use multiple regression to control for potential confounders.
## Exercises

**Exercise 1.**
A study finds that ice cream sales and drowning deaths are positively correlated ($r = 0.85$). Identify the most likely confounder and explain the mechanism by which it creates a spurious association.

??? success "Solution to Exercise 1"
    The most likely confounder is **temperature (or season)**. During summer months, higher temperatures cause both increased ice cream consumption and more people swimming (leading to more drowning deaths). The correlation between ice cream sales and drowning is **spurious** -- it arises because both variables are driven by the same underlying cause (temperature), not because one causes the other.

    Formally, if $Z$ = temperature, $X$ = ice cream sales, and $Y$ = drowning deaths, then $Z \to X$ and $Z \to Y$, but there is no direct path $X \to Y$.

---

**Exercise 2.**
In a multiple regression model, adding a control variable for education reduces the coefficient of income on health outcomes from $\hat{\beta} = 0.45$ to $\hat{\beta} = 0.12$. Interpret this change in terms of confounding.

??? success "Solution to Exercise 2"
    The large reduction (from 0.45 to 0.12) indicates that **education is a confounder** in the relationship between income and health. Education affects both income (more education leads to higher income) and health (more education leads to better health behaviors and outcomes).

    The unadjusted coefficient (0.45) overestimates the direct effect of income on health because it captures both the direct effect and the indirect effect through the income-education pathway. After controlling for education, the remaining coefficient (0.12) represents a more accurate estimate of the effect of income on health that operates through channels other than education.

---

**Exercise 3.**
Explain the difference between a confounding variable and a mediating variable. Why is it important to distinguish between the two when deciding which variables to include in a regression model?

??? success "Solution to Exercise 3"
    A **confounder** is a variable that causes both the treatment and the outcome, creating a spurious association. Controlling for a confounder removes bias and reveals the true causal effect.

    A **mediator** is a variable that lies on the causal pathway between treatment and outcome ($X \to M \to Y$). Controlling for a mediator removes part of the true causal effect, leading to an underestimate of the total effect.

    The distinction matters because: controlling for a confounder is **necessary** for valid causal inference, while controlling for a mediator is **harmful** if the goal is to estimate the total effect. For example, if exercise affects weight loss partly through increased metabolism ($\text{exercise} \to \text{metabolism} \to \text{weight loss}$), controlling for metabolism would remove a real causal pathway and underestimate the benefit of exercise.

---

**Exercise 4.**
Simpson's paradox occurs when a trend in aggregated data reverses when the data are disaggregated by a confounding variable. Construct a simple numerical example with two treatments and two subgroups where Treatment A has a higher overall success rate but Treatment B has a higher success rate within every subgroup.

??? success "Solution to Exercise 4"
    | | Subgroup 1 (easy cases) | Subgroup 2 (hard cases) | Overall |
    |---|---|---|---|
    | Treatment A | 90/100 (90%) | 5/10 (50%) | 95/110 (86.4%) |
    | Treatment B | 9/10 (90%) | 50/100 (50%) | 59/110 (53.6%) |

    Wait -- this gives equal rates within subgroups. Let me adjust:

    | | Subgroup 1 (easy cases) | Subgroup 2 (hard cases) | Overall |
    |---|---|---|---|
    | Treatment A | 87/100 (87%) | 4/10 (40%) | 91/110 (82.7%) |
    | Treatment B | 9/10 (90%) | 55/100 (55%) | 64/110 (58.2%) |

    Within Subgroup 1: B (90%) > A (87%). Within Subgroup 2: B (55%) > A (40%). But overall: A (82.7%) > B (58.2%).

    The paradox arises because Treatment A is disproportionately applied to easy cases (100 out of 110), inflating its overall rate. The subgroup (case difficulty) is the confounder.
