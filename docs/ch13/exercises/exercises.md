# Exercises

## Exercise 1: Statsmodels Package Interpretation

The following is the result of performing a linear regression analysis using the `statsmodels` package. The analysis predicts **sales** based on advertising expenses allocated to **TV**, **radio**, and **newspaper** media:

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  Sales   R-squared:                       0.894
Model:                            OLS   Adj. R-squared:                  0.891
Method:                 Least Squares   F-statistic:                     381.2
Date:                Mon, 11 Nov 2024   Prob (F-statistic):           5.60e-66
Time:                        02:39:45   Log-Likelihood:                -273.89
No. Observations:                 140   AIC:                             555.8
Df Residuals:                     136   BIC:                             567.5
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      3.0451      0.391      7.782      0.000       2.271       3.819
TV             0.0470      0.002     27.653      0.000       0.044       0.050
Radio          0.1797      0.011     16.665      0.000       0.158       0.201
Newspaper     -0.0030      0.007     -0.428      0.669      -0.017       0.011
==============================================================================
Omnibus:                       50.782   Durbin-Watson:                   2.089
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              131.355
Skew:                          -1.459   Prob(JB):                     3.00e-29
Kurtosis:                       6.741   Cond. No.                         457.
==============================================================================
```

**(a)** If advertising expenses for TV, radio, and newspapers are denoted by $x_1$, $x_2$, and $x_3$, respectively, and sales are denoted by $y$, what is the predicted value $\hat{y}$ based on the regression results?

**(b)** The coefficient for newspaper advertising is $-0.0030$. Can this be interpreted as suggesting that newspaper advertising decreases sales? Discuss its validity based on the $p$-value.

**(c)** What do the Jarque-Bera (JB) statistic (131.355) and its $p$-value (3.00e-29) signify?

??? note "Solution"

    **(a)** The predicted value of sales is:

    $$
    \hat{y} = 3.0451 + 0.0470 \cdot x_1 + 0.1797 \cdot x_2 - 0.0030 \cdot x_3
    $$

    This equation combines the intercept and coefficients for each advertising medium to predict sales.

    **(b)** The $p$-value for the newspaper coefficient is $0.669$, which is much greater than the standard significance level of 0.05. This means we fail to reject the null hypothesis that the coefficient is zero. While the coefficient is negative, its large $p$-value implies this result is not statistically meaningful. It would be more appropriate to conclude that newspaper advertising does not have a statistically significant impact on sales, rather than interpreting it as having a negative effect.

    **(c)** The Jarque-Bera test assesses whether the residuals follow a normal distribution. The extremely small $p$-value ($3.00 \times 10^{-29}$) indicates that the null hypothesis (residuals are normally distributed) is strongly rejected. This suggests the residuals are likely **not normally distributed**, which could imply model issues such as non-normal errors or the presence of outliers.

---

## Exercise 2: Male Weight and Height

Statistical information on height and weight of 100 men sampled from a specific group:

- **Height**: Mean = 173 cm, Standard Deviation = 6 cm
- **Weight**: Mean = 70 kg, Standard Deviation = 7 kg
- **Correlation between Height and Weight**: 0.59

Since the sample size is large, use the normal distribution instead of the $t$-distribution.

**(a)** What is the predicted weight of a man whose height is 179 cm?

**(b)** Suppose the predicted weight obtained above represents the weight of a certain man. What is the predicted height of this man?

**(c)** What is the 95% confidence interval for the mean predicted weight when the height is 179 cm?

**(d)** What is the 95% confidence interval for the predicted weight when the height is 179 cm?

??? note "Solution"

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

## Exercise 3: Education in Italy and France

A researcher studying education in Italy and France wanted to compare how many years, on average, men in each country spent in school. The researcher obtained a random sample of men from each country:

| | Italy | France |
|:---:|:---:|:---:|
| Mean | 10.7 | 10.4 |
| Standard Deviation | 2.3 | 2.5 |
| Number of Samples | 46 | 58 |

Test whether there is a significant difference between the two countries' mean school years with significance level $\alpha = 0.05$.

??? note "Solution"

    Two Sample $t$-Test (pooled variance):

    $$
    H_0: \mu_A = \mu_B \quad \text{vs} \quad H_1: \mu_A \neq \mu_B
    $$

    ```python
    import numpy as np
    from scipy import stats

    def main():
        X_1_bar, X_2_bar = 10.7, 10.4
        s_1, s_2 = 2.3, 2.5
        n_1, n_2 = 46, 58

        s_p_square = ((n_1-1) * s_1**2 + (n_2-1) * s_2**2) / (n_1 + n_2 - 2)
        statistic = (X_1_bar - X_2_bar) / np.sqrt(s_p_square / n_1 + s_p_square / n_2)
        df = n_1 + n_2 - 2
        p_value = 2 * stats.t(df).cdf(-abs(statistic))
        print(f"{df = :.4f}")
        print(f"{statistic = :.4f}")
        print(f"{p_value   = :.4f}")

        alpha = 0.05
        if p_value <= alpha:
            print("We choose H_1, or using statistician's jargon, reject H_0")
        else:
            print("We choose H_0, or using statistician's jargon, fail to reject H_0")

    if __name__ == "__main__":
        main()
    ```

---

## Exercise 4: Housing Data

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

??? note "Solution"

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

---

## Exercise 5: Mean Personal Incomes of Norway and US

An economist studying income in Norway and the United States wanted to compare the average annual income between the two countries. The economist obtained incomes for a random sample of people from each country denominated in thousands of US dollars:

| | Norway | US |
|:---:|:---:|:---:|
| Mean | 64.3 | 53.4 |
| Standard Deviation | 18.2 | 23.9 |
| Number of Samples | 65 | 75 |

Test whether there is a significant difference between the two countries' mean annual income with significance level $\alpha = 0.05$.

??? note "Solution"

    Two Sample $t$-Test (Welch's, unpooled variance):

    $$
    H_0: \mu_A = \mu_B \quad \text{vs} \quad H_1: \mu_A \neq \mu_B
    $$

    ```python
    import numpy as np
    from scipy import stats

    def main():
        X_1_bar, X_2_bar = 64.3, 53.4
        s_1, s_2 = 18.2, 23.9
        n_1, n_2 = 65, 75

        statistic = (X_1_bar - X_2_bar) / np.sqrt(s_1**2 / n_1 + s_2**2 / n_2)
        top = (s_1**2 / n_1 + s_2**2 / n_2)**2
        bottom = (s_1**2 / n_1)**2 / n_1 + (s_2**2 / n_2)**2 / n_2
        df = top / bottom
        p_value = 2 * stats.t(df).cdf(-abs(statistic))
        print(f"{df = :.4f}")
        print(f"{statistic = :.4f}")
        print(f"{p_value   = :.4f}")

        alpha = 0.05
        if p_value <= alpha:
            print("We choose H_1, or using statistician's jargon, reject H_0")
        else:
            print("We choose H_0, or using statistician's jargon, fail to reject H_0")

    if __name__ == "__main__":
        main()
    ```

---

## Exercise 6: Age at First Marriage

A sociologist studying marriages in the United States and Canada wanted to compare how old, on average, women in each country were when they first got married. The sociologist obtained a random sample of married women from each country:

| | US | Canada |
|:---:|:---:|:---:|
| Mean | 25.5 | 26.3 |
| Standard Deviation | 3.8 | 3.2 |
| Number of Samples | 108 | 102 |

Test whether there is a significant difference between the two countries' mean age at first marriage with significance level $\alpha = 0.05$.

??? note "Solution"

    Two Sample $t$-Test (pooled variance):

    $$
    H_0: \mu_A = \mu_B \quad \text{vs} \quad H_1: \mu_A \neq \mu_B
    $$

    ```python
    import numpy as np
    from scipy import stats

    def main():
        X_1_bar, X_2_bar = 25.5, 26.3
        s_1, s_2 = 3.8, 3.2
        n_1, n_2 = 108, 102

        s_p_square = ((n_1-1) * s_1**2 + (n_2-1) * s_2**2) / (n_1 + n_2 - 2)
        statistic = (X_1_bar - X_2_bar) / np.sqrt(s_p_square / n_1 + s_p_square / n_2)
        df = n_1 + n_2 - 2
        p_value = 2 * stats.t(df).cdf(-abs(statistic))
        print(f"{df = :.4f}")
        print(f"{statistic = :.4f}")
        print(f"{p_value   = :.4f}")

        alpha = 0.05
        if p_value <= alpha:
            print("We choose H_1, or using statistician's jargon, reject H_0")
        else:
            print("We choose H_0, or using statistician's jargon, fail to reject H_0")

    if __name__ == "__main__":
        main()
    ```

---

## Exercise 7: Two Different Electric Car Models

Julie was testing how far two new electric cars—models A and B—could drive on a full charge. She obtained a sample of 5 new cars of each model, charged them fully, and drove them as far as she could along a controlled route:

| | Model A | Model B |
|:---:|:---:|:---:|
| Mean | 168 km | 172 km |
| Standard Deviation | 5.4 km | 7.5 km |
| Number of Samples | 5 | 5 |

Test whether there is a significant difference between the two models' mean distance with significance level $\alpha = 0.05$.

??? note "Solution"

    Two Sample $t$-Test (Welch's, unpooled variance):

    $$
    H_0: \mu_A = \mu_B \quad \text{vs} \quad H_1: \mu_A \neq \mu_B
    $$

    ```python
    import numpy as np
    from scipy import stats

    def main():
        X_1_bar, X_2_bar = 168, 172
        s_1, s_2 = 5.4, 7.5
        n_1, n_2 = 5, 5

        statistic = (X_1_bar - X_2_bar) / np.sqrt(s_1**2 / n_1 + s_2**2 / n_2)
        top = (s_1**2 / n_1 + s_2**2 / n_2)**2
        bottom = (s_1**2 / n_1)**2 / n_1 + (s_2**2 / n_2)**2 / n_2
        df = top / bottom
        p_value = 2 * stats.t(df).cdf(-abs(statistic))
        print(f"{df = :.4f}")
        print(f"{statistic = :.4f}")
        print(f"{p_value   = :.4f}")

        alpha = 0.05
        if p_value <= alpha:
            print("We choose H_1, or using statistician's jargon, reject H_0")
        else:
            print("We choose H_0, or using statistician's jargon, fail to reject H_0")

    if __name__ == "__main__":
        main()
    ```

---

## Exercise 8: Model Selection — Comparing $R^2$ and AIC

Two linear regression models, **Model A** and **Model B**, were fitted to the same dataset. The results show that:

- Model A has a higher $R^2$ value.
- Model B has a lower AIC (Akaike Information Criterion).

**(a)** Between Model A (higher $R^2$) and Model B (lower AIC), which model should be selected?

**(b)** Why is AIC preferred over $R^2$ for model selection?

??? note "Solution"

    **(a)** It is generally recommended to prioritize the model with the **lower AIC** (Model B in this case).

    **(b)** Three key reasons:

    1. **Limitations of $R^2$**: $R^2$ measures the proportion of variance explained but does not penalize for model complexity. A model with more predictors can artificially increase $R^2$, even if those predictors do not improve true predictive performance, leading to overfitting.

    2. **Strengths of AIC**: AIC balances model fit (how well the model explains the data) and model simplicity (penalizing additional predictors). This helps select the model likely to have better predictive performance on unseen data.

    3. **Key Difference**: $R^2$ focuses solely on explanatory power, while AIC accounts for both explanation and complexity. AIC is therefore a more reliable criterion for comparing models in terms of prediction accuracy.
