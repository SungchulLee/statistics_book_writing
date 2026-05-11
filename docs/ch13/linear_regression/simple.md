# Simple Linear Regression


Simple linear regression models the relationship between a single independent variable $X$ and a dependent variable $Y$ as a straight line:

$$
Y = \beta_0 + \beta_1 X + \varepsilon
$$

where:

- $Y$ is the dependent variable (the outcome we are trying to predict).
- $X$ is the independent variable (the predictor).
- $\beta_0$ is the y-intercept of the regression line.
- $\beta_1$ is the slope of the regression line (the amount by which $Y$ changes for a one-unit change in $X$).
- $\varepsilon$ represents the error term, accounting for the difference between the observed and predicted values.

**Key assumptions**:

- **Linearity**: The model assumes a linear relationship between $X$ and $Y$.
- **Deterministic vs. Stochastic**: The deterministic part of the model is $\beta_0 + \beta_1 X$, while $\varepsilon$ captures the stochastic (random) part, representing the variability in $Y$ not explained by $X$.

---

## 1. Visualizing the Relationship: Height and Weight

We begin with a concrete example — the relationship between height and weight among males — to build geometric intuition for regression.

### Scatter Plot

A scatter plot reveals the overall pattern: taller individuals tend to weigh more, suggesting a positive association.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load dataset from URL
data_url = "https://raw.githubusercontent.com/beccadsouza/Machine-Learning-Python/master/Datasets/height-weight.csv"
dataframe = pd.read_csv(data_url)

# Filter for male entries and select only the first 300 rows
male_height_weight_data = dataframe[dataframe.Gender == "Male"].loc[:300, ["Height", "Weight"]]

# Calculate means of height and weight
mean_height = male_height_weight_data.Height.mean()
mean_weight = male_height_weight_data.Weight.mean()

# Calculate standard deviations of height and weight
std_height = male_height_weight_data.Height.std()
std_weight = male_height_weight_data.Weight.std()

# Calculate correlation between height and weight
height_weight_corr = male_height_weight_data.corr().loc["Height", "Weight"]

# Plot scatter plot of height vs. weight
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(male_height_weight_data.Height, male_height_weight_data.Weight, '.k', label='Data Points')

ax.set_xlabel('Height (inches)', fontsize=15)
ax.set_ylabel('Weight (pounds)', fontsize=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()
plt.show()
```

### Point of Averages

The **point of averages** $(\bar{x}, \bar{y})$ is the center of the scatter plot. Every regression line passes through this point.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_url = "https://raw.githubusercontent.com/beccadsouza/Machine-Learning-Python/master/Datasets/height-weight.csv"
dataframe = pd.read_csv(data_url)

male_height_weight_data = dataframe[dataframe.Gender == "Male"].loc[:300, ["Height", "Weight"]]

mean_height = male_height_weight_data.Height.mean()
mean_weight = male_height_weight_data.Weight.mean()
std_height = male_height_weight_data.Height.std()
std_weight = male_height_weight_data.Weight.std()
height_weight_corr = male_height_weight_data.corr().loc["Height", "Weight"]

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(male_height_weight_data.Height, male_height_weight_data.Weight, '.k', label='Data Points')
ax.plot([mean_height], [mean_weight], 'ro', markersize=15, label="Point of Averages")

ax.set_xlabel('Height (inches)', fontsize=15)
ax.set_ylabel('Weight (pounds)', fontsize=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()
plt.show()
```

### Standard Deviation Bands

The **2 SD bands** mark the interval $[\bar{x} - 2\sigma_x,\; \bar{x} + 2\sigma_x]$ or $[\bar{y} - 2\sigma_y,\; \bar{y} + 2\sigma_y]$. Approximately 95% of the data falls within these bands (under normality).

#### 2 SD x-Band (Height)

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def add_vertical_reference_line(axis, x_position, y_min, y_max, line_style, line_color='k', line_label=None):
    """Draws a vertical reference line on the given axis."""
    axis.plot([x_position, x_position], [y_min, y_max], linestyle=line_style, color=line_color, label=line_label)

data_url = "https://raw.githubusercontent.com/beccadsouza/Machine-Learning-Python/master/Datasets/height-weight.csv"
data = pd.read_csv(data_url)

male_height_weight_data = data[data.Gender == "Male"].loc[:300, ["Height", "Weight"]]

mean_height = male_height_weight_data.Height.mean()
mean_weight = male_height_weight_data.Weight.mean()
std_dev_height = male_height_weight_data.Height.std()
std_dev_weight = male_height_weight_data.Weight.std()
height_weight_corr = male_height_weight_data.corr().loc["Height", "Weight"]

fig, axis = plt.subplots(figsize=(6, 6))
axis.plot(male_height_weight_data.Height, male_height_weight_data.Weight, '.k', label="Data Points")
axis.plot([mean_height], [mean_weight], 'ro', markersize=15, label="Point of Averages")

add_vertical_reference_line(axis, mean_height - 2 * std_dev_height,
    male_height_weight_data.Weight.min(), male_height_weight_data.Weight.max(),
    '--', 'k', "2 SD Band - Height")
add_vertical_reference_line(axis, mean_height + 2 * std_dev_height,
    male_height_weight_data.Weight.min(), male_height_weight_data.Weight.max(),
    '--', 'k')

axis.set_xlabel('Height (inches)', fontsize=15)
axis.set_ylabel('Weight (pounds)', fontsize=15)
axis.spines['top'].set_visible(False)
axis.spines['right'].set_visible(False)
axis.legend()
plt.show()
```

#### 2 SD y-Band (Weight)

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def add_horizontal_reference_line(axis, y_position, x_min, x_max, line_style, line_color='k', line_label=None):
    """Draws a horizontal reference line on the given axis."""
    axis.plot([x_min, x_max], [y_position, y_position], linestyle=line_style, color=line_color, label=line_label)

data_url = "https://raw.githubusercontent.com/beccadsouza/Machine-Learning-Python/master/Datasets/height-weight.csv"
data = pd.read_csv(data_url)

male_height_weight_data = data[data.Gender == "Male"].loc[:300, ["Height", "Weight"]]

mean_height = male_height_weight_data.Height.mean()
mean_weight = male_height_weight_data.Weight.mean()
std_dev_height = male_height_weight_data.Height.std()
std_dev_weight = male_height_weight_data.Weight.std()
height_weight_corr = male_height_weight_data.corr().loc["Height", "Weight"]

fig, axis = plt.subplots(figsize=(6, 6))
axis.plot(male_height_weight_data.Height, male_height_weight_data.Weight, '.k', label="Data Points")
axis.plot([mean_height], [mean_weight], 'ro', markersize=15, label="Point of Averages")

add_horizontal_reference_line(axis, mean_weight - 2 * std_dev_weight,
    male_height_weight_data.Height.min(), male_height_weight_data.Height.max(),
    '--', 'k', "2 SD Band - Weight")
add_horizontal_reference_line(axis, mean_weight + 2 * std_dev_weight,
    male_height_weight_data.Height.min(), male_height_weight_data.Height.max(),
    '--', 'k')

axis.set_xlabel('Height (inches)', fontsize=15)
axis.set_ylabel('Weight (pounds)', fontsize=15)
axis.spines['top'].set_visible(False)
axis.spines['right'].set_visible(False)
axis.legend()
plt.show()
```

### The SD Line

The **SD line** passes through the point of averages with slope $\pm \sigma_y / \sigma_x$. It connects points that are the same number of standard deviations away from the mean in both variables. When the correlation is positive, the positive SD line is more relevant; when negative, the negative SD line applies.

#### Positive SD Line

The positive SD line has slope $+\sigma_y / \sigma_x$. For every increase of $\sigma_x$ in height, weight increases by $\sigma_y$.

```python
import matplotlib.pyplot as plt
import pandas as pd

data_url = "https://raw.githubusercontent.com/beccadsouza/Machine-Learning-Python/master/Datasets/height-weight.csv"
data = pd.read_csv(data_url)

male_height_weight_data = data[data.Gender == "Male"].loc[:300, ["Height", "Weight"]]

mean_height = male_height_weight_data.Height.mean()
mean_weight = male_height_weight_data.Weight.mean()
std_dev_height = male_height_weight_data.Height.std()
std_dev_weight = male_height_weight_data.Weight.std()

fig, axis = plt.subplots(figsize=(8, 8))
axis.plot(male_height_weight_data.Height, male_height_weight_data.Weight, '.k', label="Data Points")
axis.plot([mean_height], [mean_weight], 'ro', markersize=15, label="Point of Averages")

# Positive SD Line spanning ±3 SD
axis.plot(
    [mean_height - 3 * std_dev_height, mean_height + 3 * std_dev_height],
    [mean_weight - 3 * std_dev_weight, mean_weight + 3 * std_dev_weight],
    linestyle="--", color='k', label="Positive SD Line"
)

# Annotate the SD triangle
axis.plot([mean_height, mean_height + std_dev_height], [mean_weight, mean_weight], '-r')
axis.plot([mean_height + std_dev_height, mean_height + std_dev_height],
          [mean_weight, mean_weight + std_dev_weight], '-r')
axis.plot([mean_height, mean_height + std_dev_height],
          [mean_weight, mean_weight + std_dev_weight], '-r')

axis.annotate("$\\sigma_x$",
    [mean_height + 0.4 * std_dev_height, mean_weight - 0.3 * std_dev_weight],
    fontsize=35, color="red")
axis.annotate("$\\sigma_y$",
    [mean_height + 1.1 * std_dev_height, mean_weight + 0.3 * std_dev_weight],
    fontsize=35, color="red")

axis.set_xlabel('Height (inches)', fontsize=15)
axis.set_ylabel('Weight (pounds)', fontsize=15)
axis.spines['top'].set_visible(False)
axis.spines['right'].set_visible(False)
axis.legend(fontsize=15)
plt.show()
```

#### Negative SD Line

The negative SD line has slope $-\sigma_y / \sigma_x$. For every increase of $\sigma_x$ in height, weight *decreases* by $\sigma_y$.

```python
import matplotlib.pyplot as plt
import pandas as pd

data_url = "https://raw.githubusercontent.com/beccadsouza/Machine-Learning-Python/master/Datasets/height-weight.csv"
data = pd.read_csv(data_url)

male_height_weight_data = data[data.Gender == "Male"].loc[:300, ["Height", "Weight"]]

mean_height = male_height_weight_data.Height.mean()
mean_weight = male_height_weight_data.Weight.mean()
std_dev_height = male_height_weight_data.Height.std()
std_dev_weight = male_height_weight_data.Weight.std()

fig, axis = plt.subplots(figsize=(8, 8))
axis.plot(male_height_weight_data.Height, male_height_weight_data.Weight, '.k', label="Data Points")
axis.plot([mean_height], [mean_weight], 'ro', markersize=15, label="Point of Averages")

# Negative SD Line spanning ±3 SD
axis.plot(
    [mean_height - 3 * std_dev_height, mean_height + 3 * std_dev_height],
    [mean_weight + 3 * std_dev_weight, mean_weight - 3 * std_dev_weight],
    linestyle="--", color='k', label="Negative SD Line"
)

axis.plot([mean_height, mean_height + std_dev_height], [mean_weight, mean_weight], '-r')
axis.plot([mean_height + std_dev_height, mean_height + std_dev_height],
          [mean_weight, mean_weight - std_dev_weight], '-r')
axis.plot([mean_height, mean_height + std_dev_height],
          [mean_weight, mean_weight - std_dev_weight], '-r')

axis.annotate("$\\sigma_x$",
    [mean_height + 0.3 * std_dev_height, mean_weight + 0.2 * std_dev_weight],
    fontsize=35, color="red")
axis.annotate("$\\sigma_y$",
    [mean_height + 1.1 * std_dev_height, mean_weight - 0.5 * std_dev_weight],
    fontsize=35, color="red")

axis.set_xlabel('Height (inches)', fontsize=15)
axis.set_ylabel('Weight (pounds)', fontsize=15)
axis.spines['top'].set_visible(False)
axis.spines['right'].set_visible(False)
axis.legend(fontsize=15)
plt.show()
```

### The Regression Line vs. the SD Line

The **regression line** has slope $r \cdot \sigma_y / \sigma_x$, which is the SD line's slope multiplied by the correlation coefficient $r$. Since $|r| \leq 1$, the regression line is always flatter than (or equal to) the SD line. This flattening is the **regression effect** — predictions regress toward the mean.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_url = "https://raw.githubusercontent.com/beccadsouza/Machine-Learning-Python/master/Datasets/height-weight.csv"
data = pd.read_csv(data_url)

male_data = data[data.Gender == "Male"].loc[:300, ["Height", "Weight"]]

mean_height = male_data.Height.mean()
mean_weight = male_data.Weight.mean()
std_dev_height = male_data.Height.std()
std_dev_weight = male_data.Weight.std()
correlation_coefficient = male_data[["Height", "Weight"]].corr().loc["Height", "Weight"]

# Regression predictions
x_values = np.array(male_data.Height)
y_pred = mean_weight + correlation_coefficient * (std_dev_weight / std_dev_height) * (x_values - mean_height)

fig, axis = plt.subplots(figsize=(8, 8))
axis.plot(male_data.Height, male_data.Weight, '.k', label="Data Points")
axis.plot([mean_height], [mean_weight], 'ro', markersize=15, label="Point of Averages")
axis.plot(x_values, y_pred, 'b', label="Regression Line")

# Annotate the regression triangle
axis.plot([mean_height, mean_height + std_dev_height], [mean_weight, mean_weight], '-b')
axis.plot([mean_height + std_dev_height, mean_height + std_dev_height],
          [mean_weight, mean_weight + correlation_coefficient * std_dev_weight], '-b')
axis.plot([mean_height, mean_height + std_dev_height],
          [mean_weight, mean_weight + correlation_coefficient * std_dev_weight], '-b')

axis.annotate("$\\sigma_x$",
    [mean_height + 0.4 * std_dev_height, mean_weight - 0.3 * std_dev_weight],
    fontsize=35, color="b")
axis.annotate("$r\\ \\sigma_y$",
    [mean_height + 1.1 * std_dev_height, mean_weight + 0.3 * std_dev_weight],
    fontsize=35, color="b")

axis.set_xlabel('Height (inches)', fontsize=15)
axis.set_ylabel('Weight (pounds)', fontsize=15)
axis.spines['top'].set_visible(False)
axis.spines['right'].set_visible(False)
axis.legend(fontsize=15)
plt.show()
```

---

## 2. The Two Regression Lines

There are two distinct regression lines depending on which variable is being predicted:

- **Regression of $Y$ on $X$** (predicting weight from height): $y = \alpha + \beta x$
- **Regression of $X$ on $Y$** (predicting height from weight): $x = \alpha' + \beta' y$

These two lines coincide only when $|r| = 1$ (perfect correlation). Otherwise they form a "V" shape opening around the point of averages.

### Both Regression Lines

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_url = "https://raw.githubusercontent.com/beccadsouza/Machine-Learning-Python/master/Datasets/height-weight.csv"
data = pd.read_csv(data_url)

male_height_weight_data = data[data.Gender == "Male"].loc[:300, ["Height", "Weight"]]

mean_height = male_height_weight_data.Height.mean()
mean_weight = male_height_weight_data.Weight.mean()
std_dev_height = male_height_weight_data.Height.std()
std_dev_weight = male_height_weight_data.Weight.std()
correlation_coefficient = male_height_weight_data.corr().loc["Height", "Weight"]

# Regression of Y on X
heights = np.array(male_height_weight_data.Height)
predicted_weights = mean_weight + correlation_coefficient * (std_dev_weight / std_dev_height) * (heights - mean_height)

# Regression of X on Y
weights = np.array(male_height_weight_data.Weight)
predicted_heights = mean_height + correlation_coefficient * (std_dev_height / std_dev_weight) * (weights - mean_weight)

fig, axis = plt.subplots(figsize=(8, 8))
axis.plot([mean_height], [mean_weight], 'ro', markersize=15, label="Point of Averages")
axis.plot(male_height_weight_data.Height, male_height_weight_data.Weight, '.k', label="Data Points")

# Y on X regression line (blue)
axis.plot(heights, predicted_weights, '-b', label="y = alpha + beta * x")
axis.plot([mean_height, mean_height + std_dev_height], [mean_weight, mean_weight], '-b')
axis.plot([mean_height + std_dev_height, mean_height + std_dev_height],
          [mean_weight, mean_weight + correlation_coefficient * std_dev_weight], '-b')
axis.plot([mean_height, mean_height + std_dev_height],
          [mean_weight, mean_weight + correlation_coefficient * std_dev_weight], '-b')

axis.annotate("$\\sigma_x$",
    [mean_height + 0.4 * std_dev_height, mean_weight - 0.3 * std_dev_weight],
    fontsize=35, color="b")
axis.annotate("$r\\ \\sigma_y$",
    [mean_height + 1.1 * std_dev_height, mean_weight + 0.3 * std_dev_weight],
    fontsize=35, color="b")

# X on Y regression line (red)
axis.plot(predicted_heights, weights, '-r', label="x = alpha + beta * y")
axis.plot([mean_height, mean_height], [mean_weight, mean_weight + std_dev_weight], '-r')
axis.plot([mean_height, mean_height + correlation_coefficient * std_dev_height],
          [mean_weight + std_dev_weight, mean_weight + std_dev_weight], '-r')
axis.plot([mean_height, mean_height + correlation_coefficient * std_dev_height],
          [mean_weight, mean_weight + std_dev_weight], '-r')

axis.annotate("$\\sigma_y$",
    [mean_height - 0.3 * std_dev_height, mean_weight + 0.4 * std_dev_weight],
    fontsize=35, color="r")
axis.annotate("$r\\ \\sigma_x$",
    [mean_height + 0.1 * std_dev_height, mean_weight + 1.2 * std_dev_weight],
    fontsize=35, color="r")

axis.set_xlabel('Height (inches)', fontsize=15)
axis.set_ylabel('Weight (pounds)', fontsize=15)
axis.legend(fontsize=15)
axis.spines['top'].set_visible(False)
axis.spines['right'].set_visible(False)
plt.show()
```

### Regression of Y on X with Vertical Strip
Selecting a narrow vertical strip of data (at a fixed height value) and examining the average weight within that strip illustrates how the regression line predicts the conditional mean.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_url = "https://raw.githubusercontent.com/beccadsouza/Machine-Learning-Python/master/Datasets/height-weight.csv"
data = pd.read_csv(data_url)

male_data = data[data.Gender == "Male"].loc[:300, ["Height", "Weight"]]

mean_height = male_data.Height.mean()
mean_weight = male_data.Weight.mean()
std_dev_height = male_data.Height.std()
std_dev_weight = male_data.Weight.std()
correlation_coefficient = male_data[["Height", "Weight"]].corr().loc["Height", "Weight"]

x_values = np.array(male_data.Height)
y_pred = mean_weight + correlation_coefficient * (std_dev_weight / std_dev_height) * (x_values - mean_height)

fig, axis = plt.subplots(figsize=(8, 8))
axis.plot(male_data.Height, male_data.Weight, '.k', label="Data Points")

# Vertical strip at mean + ~1 SD
axis.plot(
    [mean_height + 0.9 * std_dev_height, mean_height + 0.9 * std_dev_height],
    [male_data.Weight.min(), male_data.Weight.max()], '--b')
axis.plot(
    [mean_height + 1.1 * std_dev_height, mean_height + 1.1 * std_dev_height],
    [male_data.Weight.min(), male_data.Weight.max()], '--b')

axis.plot([mean_height], [mean_weight], 'ro', markersize=15, label="Point of Averages")
axis.plot(x_values, y_pred, 'b', label="Regression Line")

axis.plot([mean_height, mean_height + std_dev_height], [mean_weight, mean_weight], '-b')
axis.plot([mean_height + std_dev_height, mean_height + std_dev_height],
          [mean_weight, mean_weight + correlation_coefficient * std_dev_weight], '-b')
axis.plot([mean_height, mean_height + std_dev_height],
          [mean_weight, mean_weight + correlation_coefficient * std_dev_weight], '-b')

axis.annotate("$\\sigma_x$",
    [mean_height + 0.4 * std_dev_height, mean_weight - 0.3 * std_dev_weight],
    fontsize=35, color="b")
axis.annotate("$r\\ \\sigma_y$",
    [mean_height + 1.1 * std_dev_height, mean_weight + 0.3 * std_dev_weight],
    fontsize=35, color="b")

axis.set_xlabel('Height (inches)', fontsize=15)
axis.set_ylabel('Weight (pounds)', fontsize=15)
axis.spines['top'].set_visible(False)
axis.spines['right'].set_visible(False)
axis.legend(fontsize=15)
plt.show()
```

### Regression of X on Y with Horizontal Strip
Similarly, a horizontal strip at a fixed weight value shows the conditional mean of height.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_url = "https://raw.githubusercontent.com/beccadsouza/Machine-Learning-Python/master/Datasets/height-weight.csv"
data = pd.read_csv(data_url)

male_height_weight_data = data[data.Gender == "Male"].loc[:300, ["Height", "Weight"]]

mean_height = male_height_weight_data.Height.mean()
mean_weight = male_height_weight_data.Weight.mean()
std_dev_height = male_height_weight_data.Height.std()
std_dev_weight = male_height_weight_data.Weight.std()
correlation_coefficient = male_height_weight_data.corr().loc["Height", "Weight"]

heights = np.array(male_height_weight_data.Height)
predicted_weights = mean_weight + correlation_coefficient * (std_dev_weight / std_dev_height) * (heights - mean_height)
weights = np.array(male_height_weight_data.Weight)
predicted_heights = mean_height + correlation_coefficient * (std_dev_height / std_dev_weight) * (weights - mean_weight)

fig, axis = plt.subplots(figsize=(8, 8))
axis.plot([mean_height], [mean_weight], 'ro', markersize=15, label="Point of Averages")
axis.plot(male_height_weight_data.Height, male_height_weight_data.Weight, '.k', label="Data Points")

# Horizontal strip at mean + ~1 SD weight
axis.plot(
    [male_height_weight_data.Height.min(), male_height_weight_data.Height.max()],
    [mean_weight + 0.9 * std_dev_weight, mean_weight + 0.9 * std_dev_weight], '--r')
axis.plot(
    [male_height_weight_data.Height.min(), male_height_weight_data.Height.max()],
    [mean_weight + 1.1 * std_dev_weight, mean_weight + 1.1 * std_dev_weight], '--r')

axis.plot(predicted_heights, weights, '-r', label="x = alpha + beta * y")
axis.plot([mean_height, mean_height], [mean_weight, mean_weight + std_dev_weight], '-r')
axis.plot([mean_height, mean_height + correlation_coefficient * std_dev_height],
          [mean_weight + std_dev_weight, mean_weight + std_dev_weight], '-r')
axis.plot([mean_height, mean_height + correlation_coefficient * std_dev_height],
          [mean_weight, mean_weight + std_dev_weight], '-r')

axis.annotate("$\\sigma_y$",
    [mean_height - 0.3 * std_dev_height, mean_weight + 0.4 * std_dev_weight],
    fontsize=35, color="r")
axis.annotate("$r\\ \\sigma_x$",
    [mean_height + 0.1 * std_dev_height, mean_weight + 1.2 * std_dev_weight],
    fontsize=35, color="r")

axis.set_xlabel('Height (inches)', fontsize=15)
axis.set_ylabel('Weight (pounds)', fontsize=15)
axis.legend(fontsize=15)
axis.spines['top'].set_visible(False)
axis.spines['right'].set_visible(False)
plt.show()
```

---

## 3. Closed-Form Solution

[Reference: Khan Academy — Calculating the Equation of a Regression Line](https://www.khanacademy.org/math/ap-statistics/bivariate-data-ap/least-squares-regression/v/calculating-the-equation-of-a-regression-line)

### L2 Loss

The objective is to find parameters $\alpha$ and $\beta$ that minimize the mean squared error:

$$
l = \frac{1}{n} \sum_{i=1}^n (\alpha + \beta x_i - y_i)^2
$$

### Normal Equations

Setting the partial derivatives to zero:

$$
\begin{array}{lll}
\displaystyle \frac{\partial l}{\partial \alpha} = \frac{2}{n} \sum_{i=1}^n \left((\alpha + \beta x_i) - y_i\right) = 0
& \Rightarrow &
2\alpha + 2\beta \bar{x} - 2\bar{y} = 0 \\[6pt]
\displaystyle \frac{\partial l}{\partial \beta} = \frac{2}{n} \sum_{i=1}^n \left((\alpha + \beta x_i) - y_i\right) x_i = 0
& \Rightarrow &
2\alpha \bar{x} + 2\beta \overline{x^2} - 2\overline{xy} = 0
\end{array}
$$

### Solution

Solving the normal equations yields:

$$
\begin{array}{lll}
\beta
&=&
\displaystyle \frac{\overline{xy} - \bar{x}\bar{y}}{\overline{x^2} - (\bar{x})^2}
= \frac{\text{Cov}(X,Y)}{\text{Var}(X)}
= \frac{\rho \sqrt{\text{Var}(X)} \sqrt{\text{Var}(Y)}}{\text{Var}(X)}
= \frac{\rho \sqrt{\text{Var}(Y)}}{\sqrt{\text{Var}(X)}}
= \rho \frac{\sigma_y}{\sigma_x} \\[8pt]
\alpha &=& \displaystyle -\rho \frac{\sigma_y}{\sigma_x} \bar{x} + \bar{y}
\end{array}
$$

### Regression Line Equation

Substituting back:

$$
\begin{array}{lll}
y &=& \alpha + \beta x \\[4pt]
  &=& \displaystyle -\rho \frac{\sigma_y}{\sigma_x}\bar{x} + \bar{y} + \rho \frac{\sigma_y}{\sigma_x} x \\[4pt]
  &=& \displaystyle \rho \frac{\sigma_y}{\sigma_x}(x - \bar{x}) + \bar{y}
\end{array}
$$

In standardized form:

$$
\frac{y - \bar{y}}{\sigma_y} = \rho \frac{x - \bar{x}}{\sigma_x}
$$

This elegant result says: the predicted value of $y$ in standard units equals $r$ times the observed value of $x$ in standard units.

---

## 4. Application: Linear Regression in Finance

In finance, simple linear regression is used extensively to model the relationship between an individual asset's returns and a market benchmark's returns. The slope coefficient $\beta$ in this context is the **market beta** — a measure of systematic risk.

### AAPL vs SPY Daily Returns

```python
from sklearn.linear_model import LinearRegression
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

tickers = ["SPY", "AAPL"]

spy_data = yf.Ticker(tickers[0]).history(period='max')
aapl_data = yf.Ticker(tickers[1]).history(period='max')

spy_data[tickers[0]] = spy_data['Close'].pct_change()
aapl_data[tickers[1]] = aapl_data['Close'].pct_change()

daily_returns = pd.merge(spy_data[[tickers[0]]], aapl_data[[tickers[1]]],
                         left_index=True, right_index=True).dropna()

# Train/test split
x_train = daily_returns.iloc[-200:-100, 0:1].values
y_train = daily_returns.iloc[-200:-100, 1].values
x_test = daily_returns.iloc[-100:, 0:1].values
y_test = daily_returns.iloc[-100:, 1].values

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}\n")

model = LinearRegression()
model.fit(x_train, y_train)

print('Slope (Beta) of regression line:', f"{model.coef_[0]:.4f}")
print('Intercept (Alpha) of regression line:', f"{model.intercept_:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 3))

for ax, x, y, title in zip(axes, (x_train, x_test), (y_train, y_test), ("Training Data", "Testing Data")):
    y_pred = model.predict(x)
    ax.plot(x, y, "o", label="Actual Data")
    ax.plot(x, y_pred, "--r", alpha=0.3, label="Linear Regression")
    ax.set_title(title)
    ax.set_xlabel(tickers[0] + " Daily Return", loc='right')
    ax.set_ylabel(tickers[1] + " Daily Return", loc='top')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

plt.tight_layout()
plt.show()
```

### WMT vs SPY Daily Returns

This example includes joint histograms to visualize the marginal distributions of both return series alongside the scatter plot and regression line.

```python
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

ticker_wmt = "WMT"
ticker_spy = "SPY"

wmt_data = yf.Ticker(ticker_wmt).history(period='max')
spy_data = yf.Ticker(ticker_spy).history(period='max')

wmt_data[ticker_wmt] = wmt_data['Close'].pct_change()
spy_data[ticker_spy] = spy_data['Close'].pct_change()

daily_returns = wmt_data[[ticker_wmt]].join(spy_data[[ticker_spy]], how='inner').dropna()

spy_returns = daily_returns[ticker_spy].to_numpy().reshape(-1, 1)
wmt_returns = daily_returns[ticker_wmt].to_numpy()

model = LinearRegression()
model.fit(spy_returns, wmt_returns)
regression_line = model.predict(spy_returns)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))

# SPY histogram
axes[0, 0].hist(spy_returns, density=True, bins=50, color='skyblue', edgecolor='black')
axes[0, 0].set_xlim(-0.2, 0.2)
axes[0, 0].set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
axes[0, 0].set_title(f"{ticker_spy} Daily Returns Histogram")

axes[0, 1].axis('off')

# Scatter + regression
axes[1, 0].plot(spy_returns, wmt_returns, '.', color='purple', label='Data Points')
axes[1, 0].plot(spy_returns, regression_line, 'r-', label='Regression Line')
axes[1, 0].set_xlabel(f"{ticker_spy} Daily Returns")
axes[1, 0].set_ylabel(f"{ticker_wmt} Daily Returns")
axes[1, 0].set_xlim(-0.2, 0.2)
axes[1, 0].set_ylim(-0.2, 0.2)
axes[1, 0].set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
axes[1, 0].set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
axes[1, 0].legend()

# WMT histogram (horizontal)
axes[1, 1].hist(wmt_returns, density=True, bins=50, orientation='horizontal',
                color='lightgreen', edgecolor='black')
axes[1, 1].set_ylim(-0.2, 0.2)
axes[1, 1].set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
axes[1, 1].set_title(f"{ticker_wmt} Daily Returns Histogram")

plt.tight_layout()
plt.show()
```

## Exercises

**Exercise 1.**
Statistical information on height and weight of 100 men sampled from a specific group:

- **Height**: Mean = 173 cm, Standard Deviation = 6 cm
- **Weight**: Mean = 70 kg, Standard Deviation = 7 kg
- **Correlation between Height and Weight**: 0.59

Since the sample size is large, use the normal distribution instead of the $t$-distribution.

**(a)** What is the predicted weight of a man whose height is 179 cm?

**(b)** Suppose the predicted weight obtained above represents the weight of a certain man. What is the predicted height of this man?

**(c)** What is the 95% confidence interval for the mean predicted weight when the height is 179 cm?

**(d)** What is the 95% confidence interval for the predicted weight when the height is 179 cm?

??? success "Solution to Exercise 1"

    **(a)** Predict the weight for a height of 179 cm:

    $$
    X = \mu_X + \sigma_X \quad \rightarrow \quad \hat{Y} = \mu_Y + r\sigma_Y = 70 + 0.59 \times 7 = 74.13
    $$

    **(b)** Predict the height for a weight of 74.13 kg:

    $$
    Y = \mu_Y + r\sigma_Y \quad \rightarrow \quad \hat{X} = \mu_X + r^2\sigma_X = 173 + 0.59^2 \times 6 = 175.09
    $$

    **(c)** 95% Confidence Interval for the Mean Predicted Weight:

    $$
    \hat{Y} \pm z_{0.025} \cdot SE(\hat{Y}) = (72.13, \; 76.13)
    $$

    where:

    $$
    SE(\hat{Y}) = s_Y \sqrt{\frac{1}{n} + \frac{(X - \bar{X})^2}{\sum (X_i - \bar{X})^2}}
    $$

    **(d)** 95% Confidence Interval for the Predicted Weight:

    $$
    \hat{Y} \pm z_{0.025} \cdot SE_{\text{prediction}} = (60.10, \; 88.16)
    $$

    where:

    $$
    SE_{\text{prediction}} = s_Y \sqrt{1 + \frac{1}{n} + \frac{(X - \bar{X})^2}{\sum (X_i - \bar{X})^2}}
    $$

---

**Exercise 2.**
Use housing data and do the following:

**(a)** Plot the regression line using $x = \text{df.median\_income}$ and $y = \text{df.median\_house\_value}$.

**(b)** Compute the regression prediction of `median_house_value` when `median_income` is 8.

```python
import os
import tarfile
import urllib
from sklearn import metrics
from sklearn.linear_model import LinearRegression

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def main():
    fetch_housing_data()
    pass

if __name__ == "__main__":
    main()
```

??? success "Solution to Exercise 2"

    ```python
    import os
    import tarfile
    import urllib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression

    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
    HOUSING_PATH = os.path.join("datasets", "housing")
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
        if not os.path.isdir(housing_path):
            os.makedirs(housing_path)
        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()

    def load_housing_data(housing_path=HOUSING_PATH):
        csv_path = os.path.join(housing_path, "housing.csv")
        return pd.read_csv(csv_path)

    def main():
        fetch_housing_data()

        print("(a)")
        df = load_housing_data()
        x = np.array(df.median_income).reshape((-1, 1))
        y = np.array(df.median_house_value)
        model = LinearRegression()
        model.fit(x, y)
        y_pred = model.predict(x)
        fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(x, y, ',')
        ax.plot(x, y_pred, '--r')
        plt.show()

        print("(b)")
        print(model.predict([[8]])[0])

    if __name__ == "__main__":
        main()
    ```
