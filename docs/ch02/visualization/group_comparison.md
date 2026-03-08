# Group Comparisons

## Overview

Effective data visualization often requires comparing distributions, frequencies, or relationships across groups. This section covers the major visualization tools for group comparisons: scatter plots, line plots, bar plots, pie charts, pair plots, stem-and-leaf plots, dot plots, frequency tables, and mosaic plots.

---

## 1. Line Plots

Line plots connect data points in sequence, making them ideal for time series and trends.

### Stock Price Example

```python
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def download_stock_prices(ticker, start='2023-01-01', end='2023-12-31'):
    return yf.download(ticker, start=start, end=end)

def display_stock_prices(data, ticker, ticker_name):
    data.index = data.index.tz_localize(None)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(data['Close'], label=ticker_name, color='blue')

    date_to_mark = pd.to_datetime('2023-10-19').tz_localize(None)
    if date_to_mark in data.index:
        ax.plot([date_to_mark], [data.loc[date_to_mark, 'Close']],
                'or', label=f'{ticker_name} on {date_to_mark.date()}')

    ax.set_xlabel('Date')
    ax.set_ylabel(f'{ticker_name} Price (KRW)')
    ax.set_title(f'{ticker_name} Prices in 2023')
    ax.legend()
    plt.show()

ticker = '019170.KS'
data = download_stock_prices(ticker)
display_stock_prices(data, ticker, "Shinpoong")
```

---

## 2. Scatter Plots

Scatter plots display the relationship between two continuous variables. Matplotlib offers two methods with different capabilities.

### `ax.plot` vs. `ax.scatter`

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(0)
num_samples = 10
x = stats.norm().rvs(size=num_samples)
noise = 0.7 * stats.norm().rvs(size=num_samples)
y = 1 + 2 * x + noise

fig, (ax_plot, ax_scatter) = plt.subplots(1, 2, figsize=(12, 3))

point_sizes = 100 * stats.norm().rvs(size=num_samples) ** 2
color_values = stats.uniform().rvs(size=num_samples)

# ax.plot: Fixed marker properties
ax_plot.plot(x, y, 'o', markersize=10, mec="red", mfc="blue", mew=3)
ax_plot.set_title("Standard Plot\nFixed Marker Size")

# ax.scatter: Variable marker properties
ax_scatter.scatter(x, y, s=point_sizes, c=color_values)
ax_scatter.set_title("Scatter Plot\nVariable Marker Size")

for ax in (ax_plot, ax_scatter):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine].set_visible(False)

plt.show()
```

**Key differences:** `ax.plot` uses uniform marker size and color—ideal for simple point displays. `ax.scatter` allows each point to have individual size and color, enabling visualization of additional data dimensions.

---

## 3. Bar Plots

### Single Group Bar Plot

```python
import matplotlib.pyplot as plt
import pandas as pd

data = {
    'Courses': ('Language', 'History', 'Geometry', 'Chemistry', 'Physics'),
    'Number of Teachers': (7, 3, 9, 1, 2)
}
df = pd.DataFrame(data).set_index('Courses')

fig, ax = plt.subplots(figsize=(12, 3))
ax.bar(x=range(len(df)), height=df["Number of Teachers"],
       tick_label=df.index, width=0.5)
ax.set_xlabel('Courses')
ax.set_ylabel('Number of Teachers')
ax.set_title("Favorite Courses of Teachers")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()
```

### Grouped Bar Plot

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = {
    'Student': ['Brandon', 'Vanessa', 'Daniel', 'Kevin', 'William'],
    'Midterm': [85, 60, 60, 65, 100],
    'Final': [90, 90, 65, 80, 95]
}
df = pd.DataFrame(data).set_index('Student')

positions = np.arange(len(df))
width = 0.3

fig, ax = plt.subplots(figsize=(12, 3))
ax.bar(positions - width / 2, df['Midterm'], width=width, label="Midterm")
ax.bar(positions + width / 2, df['Final'], width=width, label="Final")
ax.set_xticks(positions)
ax.set_xticklabels(df.index)
ax.set_xlabel("Student")
ax.set_ylabel("Scores")
ax.set_title("Midterm and Final Scores")
ax.legend(title="Exam Type")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()
```

### Segmented (Stacked) Bar Plot

Segmented bar plots show the composition of each category.

```python
import matplotlib.pyplot as plt
import numpy as np

labels = ("Yes", "No")
counts = (np.array([95, 90, 40]), np.array([5, 10, 60]))
age_groups = ("Adults", "Children", "Infants")

fig, ax = plt.subplots(figsize=(6, 3))
bottom = np.zeros(3)

for label, count in zip(labels, counts):
    ax.bar(np.arange(3), count, width=0.5, bottom=bottom,
           tick_label=age_groups, label=label)
    bottom += count

ax.set_title("Has Antibodies?")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(title="Response", loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
plt.show()
```

---

## 4. Pie Charts

Pie charts show proportions of a whole. They work best with a small number of categories.

```python
import matplotlib.pyplot as plt

labels = 'Apples', 'Bananas', 'Cherries', 'Dates'
sizes = [215, 130, 245, 210]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0, 0)

fig, ax = plt.subplots()
ax.pie(sizes, explode=explode, labels=labels, colors=colors,
       autopct='%1.1f%%', shadow=True, startangle=140,
       radius=1.5, counterclock=True)
ax.axis('equal')
ax.set_title('Fruit Distribution in Basket')
plt.show()
```

The `autopct='%1.1f%%'` format string displays percentages to one decimal place on each slice.

---

## 5. Pair Plots

Pair plots create a matrix of scatter plots for every pair of variables, with histograms on the diagonal. They are invaluable for multivariate exploration.

```python
import seaborn as sns
import pandas as pd

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url, index_col='PassengerId')
df['Sex_int'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)

sns.pairplot(df[["Survived", "Age", "Sex_int"]])
```

---

## 6. Stem-and-Leaf Plots

Stem-and-leaf plots preserve individual data values while showing the distribution shape.

```python
import stemgraphic

data = [65, 93, 45, 73, 99, 70, 88, 46, 75, 34, 83, 100, 88, 72, 70]
fig, ax = stemgraphic.stem_graphic(data, scale=10,
                                    title="Stem-and-Leaf Plot of Student Scores")
```

---

## 7. Dot Plots and Frequency Tables

### Dot Plot

```python
import matplotlib.pyplot as plt

data = [5, 7, 5, 9, 7, 7, 6, 9, 9, 9, 10, 12, 12, 7]

age_freq = {}
for age in data:
    age_freq[age] = age_freq.get(age, 0) + 1

fig, ax = plt.subplots(figsize=(12, 3))
for age, freq in age_freq.items():
    ax.plot([age] * freq, range(1, freq + 1), 'ok')

ax.set_xlabel('Ages')
ax.set_ylabel('Number of Students')
ax.set_title("Ages of Students in Class")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticks([0, 1, 2, 3, 4])
ax.spines["bottom"].set_position("zero")
plt.show()
```

---

## 8. Two-Way Frequency Tables

### Frequency Table

```python
import pandas as pd

data = {'SUV': 28*['yes'] + 35*['no'] + 97*['yes'] + 104*['no'],
        'Accident': 28*['yes'] + 35*['yes'] + 97*['no'] + 104*['no']}
df = pd.DataFrame(data)

dg = pd.crosstab(df.SUV, df.Accident, rownames=['SUV'], colnames=['Accident'])
dg.loc['TOTAL', :] = dg.sum()
dg.loc[:, 'TOTAL'] = dg.sum(axis=1)
dg = dg.astype(int)
print(dg)
```

### Relative Frequency Table

```python
dh = dg / dg.loc['TOTAL', 'TOTAL']
print(dh)
```

### Connection to Probability

Two-way frequency tables directly relate to joint, marginal, and conditional distributions:

$$
\begin{array}{ll}
\text{Chain rule:} & p(x, y) = p(x) \, p(y|x) \\
\text{Marginalization:} & p(x) = \sum_y p(x, y) \\
\text{Conditioning:} & p(y|x) = \frac{p(x, y)}{p(x)}
\end{array}
$$

---

## 9. Data Types and Appropriate Visualizations

$$
\text{Data} \begin{cases}
\text{Categorical Data: Pie Chart, Bar Chart, Mosaic Plot, \ldots} \\
\text{Quantitative Data: Histogram, Box Plot, Stem Plot, Time Plot, \ldots}
\end{cases}
$$

## Summary

Different group comparison tasks call for different visualization tools. Bar plots and pie charts work for categorical data; histograms, box plots, and violin plots reveal the shape of continuous distributions; scatter plots and pair plots expose bivariate relationships; and frequency tables bridge visualization with probability. Choosing the right tool depends on the data type, the number of groups, and the specific aspect of the comparison you want to emphasize.
