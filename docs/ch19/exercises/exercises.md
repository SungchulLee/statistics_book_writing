# Chapter 13: Exercises

## Exercise 1: Logit and Odds

A logistic regression model for loan default ($Y = 1$) has: $\log\frac{p}{1-p} = -2.5 + 0.8\,\text{DTI} - 0.03\,\text{Credit Score}$

where DTI is debt-to-income ratio and Credit Score is FICO score.

**(a)** For a borrower with DTI = 3.0 and Credit Score = 700, compute the predicted probability of default.

**(b)** Interpret the coefficient 0.8 for DTI in terms of odds ratios.

**(c)** What DTI ratio gives a 50% default probability for a borrower with Credit Score = 650?

??? note "Solution"

    **(a)** $\text{logit} = -2.5 + 0.8(3) - 0.03(700) = -2.5 + 2.4 - 21 = -21.1$

    $\hat{p} = \frac{1}{1 + e^{21.1}} \approx 0$. Very low default probability.

    Actually, let's reconsider: perhaps the model uses Credit Score / 100.

    With credit_score_units: $\text{logit} = -2.5 + 0.8(3) - 0.03(700) = -2.5 + 2.4 - 21 = -21.1$

    $\hat{p} = 1/(1+e^{21.1}) \approx 6.7 \times 10^{-10}$

    **(b)** $e^{0.8} \approx 2.23$. A one-unit increase in DTI multiplies the odds of default by 2.23 (a 123% increase), holding credit score constant.

    **(c)** Set $p = 0.5 \Rightarrow \text{logit} = 0$: $0 = -2.5 + 0.8\,\text{DTI} - 0.03(650)$

    $0.8\,\text{DTI} = 2.5 + 19.5 = 22$, so $\text{DTI} = 27.5$.

---

## Exercise 2: MLE for Logistic Regression

Given 4 observations: $(x_1, y_1) = (1, 0)$, $(x_2, y_2) = (2, 0)$, $(x_3, y_3) = (3, 1)$, $(x_4, y_4) = (4, 1)$ with model $\log\frac{p}{1-p} = \beta_0 + \beta_1 x$.

**(a)** Write the log-likelihood as a function of $\beta_0$ and $\beta_1$.

**(b)** Explain why there is no closed-form solution and numerical optimization is needed.

**(c)** Use Python to find $\hat{\beta}_0$ and $\hat{\beta}_1$.

??? note "Solution"

    **(a)** $p_i = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_i)}}$

    $\ell(\beta_0, \beta_1) = \sum_{i=1}^4 \left[y_i \log p_i + (1-y_i)\log(1-p_i)\right]$

    $= \log(1-p_1) + \log(1-p_2) + \log p_3 + \log p_4$

    **(b)** The score equations $\frac{\partial \ell}{\partial \beta_0} = 0$ and $\frac{\partial \ell}{\partial \beta_1} = 0$ involve $p_i$, which is a nonlinear function of $\beta$. The equations cannot be solved algebraically.

    **(c)**

    ```python
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression(penalty=None, solver='lbfgs')
    model.fit(X, y)
    print(f"β₀ = {model.intercept_[0]:.4f}")
    print(f"β₁ = {model.coef_[0][0]:.4f}")
    ```

---

## Exercise 3: Model Evaluation

A spam classifier produces the following confusion matrix on a test set of 1000 emails:

|  | Predicted Spam | Predicted Not Spam |
|---|---|---|
| **Actual Spam** | 85 | 15 |
| **Actual Not Spam** | 30 | 870 |

**(a)** Compute accuracy, precision, recall, specificity, and F1 score.

**(b)** If the cost of a missed spam is 3× the cost of a false alarm, should you lower or raise the classification threshold?

**(c)** What is the False Positive Rate?

??? note "Solution"

    **(a)** TP=85, FN=15, FP=30, TN=870

    Accuracy $= (85+870)/1000 = 0.955$

    Precision $= 85/(85+30) = 0.739$

    Recall $= 85/(85+15) = 0.850$

    Specificity $= 870/(870+30) = 0.967$

    F1 $= 2 \times 0.739 \times 0.850/(0.739+0.850) = 0.791$

    **(b)** Since missed spam (FN) is costlier, we should **lower** the threshold to increase recall (catch more spam), accepting more false positives.

    **(c)** FPR $= FP/(FP+TN) = 30/900 = 0.033$

---

## Exercise 4: ROC and AUC

**(a)** Explain why the ROC curve of any reasonable classifier lies above the diagonal.

**(b)** A model has AUC = 0.85. Give the probabilistic interpretation.

**(c)** Model A has AUC = 0.92 and Model B has AUC = 0.88. Can you always conclude Model A is better for deployment? Why or why not?

??? note "Solution"

    **(a)** The diagonal represents random classification (TPR = FPR at every threshold). A reasonable classifier assigns higher probabilities to positives than negatives, so at any FPR, it achieves a higher TPR than random guessing.

    **(b)** If we randomly pick one positive and one negative example, there is an 85% chance the model assigns a higher predicted probability to the positive example.

    **(c)** Not necessarily. AUC summarizes performance across all thresholds. At the specific operating threshold relevant to the application, Model B might outperform Model A. Also, for imbalanced data, the PR-AUC might be more informative. AUC also doesn't account for different misclassification costs.
