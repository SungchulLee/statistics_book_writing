# Mean Absolute Percentage Error and Other Relative Metrics

MAE, MSE, and RMSE all measure errors in the original units of $Y$. This makes them difficult to compare across datasets with different scales: an RMSE of 5 means something very different when the response ranges from 0 to 10 versus 0 to 10,000. Relative metrics address this by expressing errors as fractions or percentages of the observed values, enabling comparison across different contexts.

---

## 1. Mean Absolute Percentage Error

The **mean absolute percentage error (MAPE)** expresses each prediction error as a percentage of the observed value:

$$
\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100\%
$$

A MAPE of 5% means the model's predictions are off by an average of 5% relative to the true values.

### Advantages

- **Scale-free**: MAPE allows direct comparison of model performance across datasets with different units or magnitudes.
- **Intuitive**: Percentage errors are easy for non-technical audiences to interpret.

### Limitations

- **Undefined when $y_i = 0$**: The division by $y_i$ makes MAPE undefined whenever an observed value is zero. In practice, this limits MAPE to datasets where all observations are strictly positive.
- **Asymmetric penalty**: MAPE penalizes over-predictions and under-predictions unequally. An over-prediction from $y = 100$ to $\hat{y} = 150$ yields a 50% error, but an under-prediction from $y = 100$ to $\hat{y} = 50$ also yields 50%. However, if we reverse the roles — an observation of $y = 50$ with prediction $\hat{y} = 100$ yields 100% error, while $y = 150$ with $\hat{y} = 100$ yields only 33%. This asymmetry biases MAPE toward models that systematically under-predict.
- **Not bounded above**: Individual percentage errors can exceed 100%, and MAPE has no finite upper bound.

---

## 2. Symmetric Mean Absolute Percentage Error

The **symmetric mean absolute percentage error (sMAPE)** addresses the asymmetry of MAPE by using the average of the observed and predicted values in the denominator:

$$
\text{sMAPE} = \frac{1}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|) / 2} \times 100\%
$$

This can be simplified to:

$$
\text{sMAPE} = \frac{2}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i|} \times 100\%
$$

### Properties

- **Bounded**: Each term lies between 0% and 200%, so $\text{sMAPE} \in [0\%, 200\%]$.
- **More symmetric**: Over-predictions and under-predictions of the same magnitude receive more comparable penalties than under MAPE.
- **Still undefined when $y_i = \hat{y}_i = 0$**: The denominator vanishes when both the observed and predicted values are zero, though this case is typically handled by defining the contribution as zero.

!!! warning "sMAPE is not truly symmetric"
    Despite its name, sMAPE is not perfectly symmetric in all cases. When $y_i$ and $\hat{y}_i$ have opposite signs, the behavior can be counterintuitive. sMAPE works best when all values are positive.

---

## 3. Mean Absolute Scaled Error

The **mean absolute scaled error (MASE)** scales prediction errors relative to the in-sample MAE of a naive baseline model. For cross-sectional regression, the naive baseline is typically the sample mean $\bar{y}$:

$$
\text{MASE} = \frac{\displaystyle \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|}{\displaystyle \frac{1}{n}\sum_{i=1}^{n} |y_i - \bar{y}|}
$$

This simplifies to the ratio of the model's MAE to the baseline MAE:

$$
\text{MASE} = \frac{\text{MAE}_{\text{model}}}{\text{MAE}_{\text{baseline}}}
$$

### Interpretation

- $\text{MASE} < 1$: the model outperforms the naive baseline.
- $\text{MASE} = 1$: the model performs no better than the baseline.
- $\text{MASE} > 1$: the model performs worse than the baseline.

MASE is well-defined even when $y_i = 0$ (as long as the denominator is nonzero), making it a practical alternative to MAPE.

---

## 4. Relative Squared Error and Relative Absolute Error

Two additional relative metrics normalize errors against the baseline model (predicting $\bar{y}$ for every observation):

**Relative Squared Error (RSE)**:

$$
\text{RSE} = \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} = \frac{\text{SSE}}{\text{SST}} = 1 - R^2
$$

**Relative Absolute Error (RAE)**:

$$
\text{RAE} = \frac{\sum_{i=1}^{n}|y_i - \hat{y}_i|}{\sum_{i=1}^{n}|y_i - \bar{y}|}
$$

Both RSE and RAE yield values less than 1 when the model outperforms the mean baseline. Note that RSE is simply the complement of $R^2$.

---

## 5. Comparison of Relative Metrics

| Metric | Handles $y_i = 0$ | Bounded | Symmetric | Scale-free |
|--------|-------------------|---------|-----------|------------|
| MAPE   | No                | No      | No        | Yes        |
| sMAPE  | Partially         | Yes (0--200%) | Approximately | Yes |
| MASE   | Yes               | No      | Yes       | Yes        |
| RSE    | Yes               | No (but typically $\in [0,1]$) | Yes | Yes |
| RAE    | Yes               | No (but typically $\in [0,1]$) | Yes | Yes |

!!! tip "Choosing a relative metric"
    Use MAPE when communicating with non-technical stakeholders and all observed values are strictly positive. Use MASE when zeros are present or when you want a principled comparison against a baseline. Use RSE when you want a direct complement to $R^2$.

---

## 6. Numerical Example

Consider a model with $n = 4$ observations where all values are positive:

| $i$ | $y_i$ | $\hat{y}_i$ | $|e_i|$ | $|e_i|/y_i$ | $2|e_i|/(y_i + \hat{y}_i)$ |
|-----|--------|--------------|----------|--------------|-----------------------------|
| 1   | 100    | 110          | 10       | 0.100        | 0.095                       |
| 2   | 200    | 190          | 10       | 0.050        | 0.051                       |
| 3   | 50     | 45           | 5        | 0.100        | 0.105                       |
| 4   | 150    | 160          | 10       | 0.067        | 0.065                       |

The sample mean is $\bar{y} = 125$.

$$
\text{MAPE} = \frac{0.100 + 0.050 + 0.100 + 0.067}{4} \times 100\% = 7.9\%
$$

$$
\text{sMAPE} = \frac{0.095 + 0.051 + 0.105 + 0.065}{4} \times 100\% = 7.9\%
$$

For MASE, the baseline MAE is:

$$
\text{MAE}_{\text{baseline}} = \frac{|100-125| + |200-125| + |50-125| + |150-125|}{4} = \frac{25+75+75+25}{4} = 50
$$

$$
\text{MAE}_{\text{model}} = \frac{10+10+5+10}{4} = 8.75
$$

$$
\text{MASE} = \frac{8.75}{50} = 0.175
$$

A MASE of 0.175 indicates the model's average error is only 17.5% of the naive baseline's error, confirming strong predictive performance relative to predicting the mean.
