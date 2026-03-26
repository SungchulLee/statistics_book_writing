# Exercises

These exercises cover softmax regression and multiclass classification from Chapter 20, including the softmax function and its properties, cross-entropy loss, gradient derivations, numerical stability, confusion matrices, multiclass averaging schemes, and comparisons between softmax regression and OvR/OvO strategies.

---

## Exercise 1: Softmax Function Computation

Consider a 3-class classification problem with logit vector $\mathbf{z} = (2, 1, -1)^\top$.

**(a)** Compute the softmax probabilities $\hat{p}_k = \text{softmax}(\mathbf{z})_k$ for $k = 1, 2, 3$ using the formula:

$$
\hat{p}_k = \frac{e^{z_k}}{\sum_{j=1}^C e^{z_j}}
$$

**(b)** Verify that the probabilities sum to 1 and that each is non-negative.

**(c)** Show that the softmax function is shift-invariant: for any constant $c$, $\text{softmax}(\mathbf{z} + c\mathbf{1}) = \text{softmax}(\mathbf{z})$. Why is this property important for numerical stability?

??? success "Solution"

    **(a)** First compute the exponentials:

    $$
    e^{z_1} = e^2 \approx 7.389, \quad e^{z_2} = e^1 \approx 2.718, \quad e^{z_3} = e^{-1} \approx 0.368
    $$

    The normalizing constant is:

    $$
    \sum_{j=1}^3 e^{z_j} = 7.389 + 2.718 + 0.368 = 10.475
    $$

    The softmax probabilities are:

    $$
    \hat{p}_1 = \frac{7.389}{10.475} \approx 0.705, \quad \hat{p}_2 = \frac{2.718}{10.475} \approx 0.259, \quad \hat{p}_3 = \frac{0.368}{10.475} \approx 0.035
    $$

    **(b)** $\hat{p}_1 + \hat{p}_2 + \hat{p}_3 = 0.705 + 0.259 + 0.035 \approx 1.0$. Each probability is positive since exponentials are always positive. This confirms that softmax maps any real-valued logit vector to a valid probability distribution on the simplex.

    **(c)** For any constant $c$:

    $$
    \text{softmax}(\mathbf{z} + c\mathbf{1})_k = \frac{e^{z_k + c}}{\sum_j e^{z_j + c}} = \frac{e^c \cdot e^{z_k}}{e^c \cdot \sum_j e^{z_j}} = \frac{e^{z_k}}{\sum_j e^{z_j}} = \text{softmax}(\mathbf{z})_k
    $$

    The $e^c$ factors cancel. This property is crucial for the **log-sum-exp trick**: by choosing $c = -\max_k z_k$, we ensure the largest exponent is $e^0 = 1$, preventing numerical overflow when logits are large.

---

## Exercise 2: Cross-Entropy Loss

Consider a single training example with true class label $y = 2$ (out of $C = 4$ classes) and predicted probability vector $\hat{\mathbf{p}} = (0.1, 0.6, 0.2, 0.1)^\top$.

**(a)** Write the one-hot encoding $\mathbf{y}$ for this example.

**(b)** Compute the cross-entropy loss:

$$
L = -\sum_{k=1}^C y_k \log \hat{p}_k
$$

**(c)** Suppose another model predicts $\hat{\mathbf{p}}' = (0.05, 0.85, 0.05, 0.05)^\top$. Compute its cross-entropy loss and explain which model is better.

**(d)** What is the minimum possible cross-entropy loss for a correctly classified example? When is it achieved?

??? success "Solution"

    **(a)** The one-hot encoding for class 2 (using 1-based indexing) is:

    $$
    \mathbf{y} = (0, 1, 0, 0)^\top
    $$

    **(b)** Since only $y_2 = 1$, the sum reduces to a single term:

    $$
    L = -\log \hat{p}_2 = -\log(0.6) \approx 0.511
    $$

    **(c)** For the second model:

    $$
    L' = -\log(0.85) \approx 0.163
    $$

    Since $L' < L$, the second model is better — it assigns higher probability to the correct class. Cross-entropy penalizes low confidence in the true class. The closer $\hat{p}_y$ is to 1, the lower the loss.

    **(d)** The minimum cross-entropy loss is 0, achieved when $\hat{p}_y = 1$ (i.e., the model assigns probability 1 to the correct class). Since $-\log(1) = 0$, a perfectly confident and correct prediction incurs zero loss.

---

## Exercise 3: Cross-Entropy Gradient

Consider softmax regression with $C$ classes. The predicted probability for class $k$ is $\hat{p}_k = \text{softmax}(\mathbf{z})_k$ where $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$.

**(a)** Show that the derivative of the softmax function satisfies:

$$
\frac{\partial \hat{p}_k}{\partial z_j} = \hat{p}_k(\delta_{kj} - \hat{p}_j)
$$

where $\delta_{kj}$ is the Kronecker delta.

**(b)** Using the result from (a), derive the gradient of the cross-entropy loss with respect to the logits:

$$
\frac{\partial L}{\partial z_j} = \hat{p}_j - y_j
$$

**(c)** Interpret this gradient. What happens when the model is very confident and correct? What about when it is confident and wrong?

??? success "Solution"

    **(a)** The softmax function is $\hat{p}_k = e^{z_k} / S$ where $S = \sum_m e^{z_m}$.

    **Case $k = j$:** Using the quotient rule:

    $$
    \frac{\partial \hat{p}_k}{\partial z_k} = \frac{e^{z_k} \cdot S - e^{z_k} \cdot e^{z_k}}{S^2} = \frac{e^{z_k}}{S} - \left(\frac{e^{z_k}}{S}\right)^2 = \hat{p}_k - \hat{p}_k^2 = \hat{p}_k(1 - \hat{p}_k)
    $$

    **Case $k \ne j$:**

    $$
    \frac{\partial \hat{p}_k}{\partial z_j} = \frac{0 - e^{z_k} \cdot e^{z_j}}{S^2} = -\hat{p}_k \hat{p}_j
    $$

    Both cases are unified by $\frac{\partial \hat{p}_k}{\partial z_j} = \hat{p}_k(\delta_{kj} - \hat{p}_j)$.

    **(b)** The cross-entropy loss is $L = -\sum_k y_k \log \hat{p}_k$. By the chain rule:

    $$
    \frac{\partial L}{\partial z_j} = -\sum_k y_k \frac{1}{\hat{p}_k} \frac{\partial \hat{p}_k}{\partial z_j} = -\sum_k y_k \frac{1}{\hat{p}_k} \hat{p}_k(\delta_{kj} - \hat{p}_j)
    $$

    $$
    = -\sum_k y_k (\delta_{kj} - \hat{p}_j) = -y_j + \hat{p}_j \sum_k y_k
    $$

    Since $\mathbf{y}$ is one-hot, $\sum_k y_k = 1$, giving:

    $$
    \frac{\partial L}{\partial z_j} = \hat{p}_j - y_j
    $$

    In vector form: $\nabla_{\mathbf{z}} L = \hat{\mathbf{p}} - \mathbf{y}$.

    **(c)** The gradient $\hat{p}_j - y_j$ has a clean interpretation:

    - **Correct and confident** ($y_j = 1$, $\hat{p}_j \approx 1$): gradient $\approx 0$. The model is already correct, so little update is needed.
    - **Correct but uncertain** ($y_j = 1$, $\hat{p}_j \approx 0.3$): gradient $\approx -0.7$. The negative gradient pushes $z_j$ upward to increase $\hat{p}_j$.
    - **Wrong and confident** ($y_j = 0$, $\hat{p}_j \approx 0.9$): gradient $\approx 0.9$. The large positive gradient pushes $z_j$ downward to decrease $\hat{p}_j$.

    This "residual" form $(\hat{\mathbf{p}} - \mathbf{y})$ parallels the gradient of squared error in linear regression $(\hat{\mathbf{y}} - \mathbf{y})$, making gradient descent updates intuitive.

---

## Exercise 4: Numerical Stability and Log-Sum-Exp

Consider logits $\mathbf{z} = (1000, 1001, 999)^\top$.

**(a)** Explain why naively computing $e^{1000}$, $e^{1001}$, $e^{999}$ will cause numerical overflow in standard 64-bit floating point.

**(b)** Apply the log-sum-exp trick: subtract $m = \max_k z_k = 1001$ and compute $\text{softmax}(\mathbf{z} - m\mathbf{1})$.

**(c)** Derive the log-sum-exp formula:

$$
\log\sum_{k=1}^C e^{z_k} = m + \log\sum_{k=1}^C e^{z_k - m}
$$

and explain why the right-hand side is numerically stable.

??? success "Solution"

    **(a)** In 64-bit floating point (double precision), the maximum representable value is approximately $1.8 \times 10^{308}$, which corresponds to $e^{709.8}$. Since $e^{999} \approx 10^{434}$ already far exceeds this limit, computing $e^{1000}$ or $e^{1001}$ directly would produce `inf` (infinity), making the softmax calculation $\hat{p}_k = e^{z_k}/\sum e^{z_j}$ return `nan` (the ratio `inf/inf` is undefined).

    **(b)** Shifting by $m = 1001$:

    $$
    \mathbf{z} - m\mathbf{1} = (-1, 0, -2)^\top
    $$

    $$
    e^{-1} \approx 0.368, \quad e^{0} = 1, \quad e^{-2} \approx 0.135
    $$

    $$
    S = 0.368 + 1 + 0.135 = 1.503
    $$

    $$
    \hat{p}_1 = \frac{0.368}{1.503} \approx 0.245, \quad \hat{p}_2 = \frac{1}{1.503} \approx 0.665, \quad \hat{p}_3 = \frac{0.135}{1.503} \approx 0.090
    $$

    All computations involve exponentials of small numbers, avoiding overflow.

    **(c)** Starting from the definition:

    $$
    \log\sum_{k=1}^C e^{z_k} = \log\sum_{k=1}^C e^{(z_k - m) + m} = \log\left(e^m \sum_{k=1}^C e^{z_k - m}\right)
    $$

    $$
    = m + \log\sum_{k=1}^C e^{z_k - m}
    $$

    The right-hand side is stable because: (1) the largest term in the sum is $e^{z_{\max} - m} = e^0 = 1$, so no term overflows; (2) all other terms are $e^{\text{negative}} < 1$, so underflow to zero is benign (it just means those terms are negligible); (3) the overall log-sum is then at least $\log(1) = 0$, so the final result $m + \text{(something} \ge 0)$ is well-defined.

---

## Exercise 5: Confusion Matrix and Per-Class Metrics

A 3-class classifier produces the following confusion matrix on a test set of 100 examples:

|  | Predicted A | Predicted B | Predicted C |
|:---:|:---:|:---:|:---:|
| **Actual A** | 25 | 5 | 0 |
| **Actual B** | 3 | 32 | 5 |
| **Actual C** | 2 | 3 | 25 |

**(a)** Compute the overall accuracy.

**(b)** Compute the precision, recall, and F1-score for each class.

**(c)** Compute the macro-averaged and micro-averaged F1-scores.

**(d)** Which class has the worst performance? What does the confusion matrix reveal about common misclassifications?

??? success "Solution"

    **(a)** Overall accuracy is the fraction of correct predictions (diagonal sum):

    $$
    \text{Accuracy} = \frac{25 + 32 + 25}{100} = \frac{82}{100} = 0.82
    $$

    **(b)** For each class:

    **Class A:**

    - Precision $= 25 / (25 + 3 + 2) = 25/30 \approx 0.833$
    - Recall $= 25 / (25 + 5 + 0) = 25/30 \approx 0.833$
    - $F_1 = 2 \times 0.833 \times 0.833 / (0.833 + 0.833) = 0.833$

    **Class B:**

    - Precision $= 32 / (5 + 32 + 3) = 32/40 = 0.800$
    - Recall $= 32 / (3 + 32 + 5) = 32/40 = 0.800$
    - $F_1 = 0.800$

    **Class C:**

    - Precision $= 25 / (0 + 5 + 25) = 25/30 \approx 0.833$
    - Recall $= 25 / (2 + 3 + 25) = 25/30 \approx 0.833$
    - $F_1 = 0.833$

    **(c)** **Macro-averaged F1** (unweighted mean across classes):

    $$
    F_1^{\text{macro}} = \frac{0.833 + 0.800 + 0.833}{3} = \frac{2.467}{3} \approx 0.822
    $$

    **Micro-averaged F1**: In the micro approach, we sum all true positives, false positives, and false negatives across classes.

    - Total TP $= 25 + 32 + 25 = 82$
    - Total FP $= (3+2) + (5+3) + (0+5) = 5 + 8 + 5 = 18$
    - Total FN $= (5+0) + (3+5) + (2+3) = 5 + 8 + 5 = 18$

    $$
    \text{Precision}_{\text{micro}} = \frac{82}{82 + 18} = 0.82, \quad \text{Recall}_{\text{micro}} = \frac{82}{82 + 18} = 0.82
    $$

    $$
    F_1^{\text{micro}} = 0.82
    $$

    For this balanced dataset (30, 40, 30 examples per class), micro and macro F1 are similar.

    **(d)** Class B has the worst F1-score (0.800). The confusion matrix shows that Class B loses 3 examples to Class A and 5 examples to Class C. The most common error is predicting Class B when the true class is A (5 misclassifications) and predicting Class C when the true class is B (5 misclassifications). This suggests that the decision boundaries between B and its neighbors may need refinement.

---

## Exercise 6: Macro, Micro, and Weighted Averaging

Consider a test set with 3 classes of highly imbalanced size:

| Class | Support | Precision | Recall |
|:---:|:---:|:---:|:---:|
| 0 | 900 | 0.95 | 0.98 |
| 1 | 80 | 0.70 | 0.50 |
| 2 | 20 | 0.40 | 0.30 |

**(a)** Compute the macro-averaged precision and recall.

**(b)** Compute the weighted-averaged precision and recall (weighted by support).

**(c)** Explain why macro-averaging is preferred when minority class performance matters, and why micro-averaging can be misleading with imbalanced data.

??? success "Solution"

    **(a)** Macro-averaged (simple unweighted mean):

    $$
    \text{Precision}_{\text{macro}} = \frac{0.95 + 0.70 + 0.40}{3} = \frac{2.05}{3} \approx 0.683
    $$

    $$
    \text{Recall}_{\text{macro}} = \frac{0.98 + 0.50 + 0.30}{3} = \frac{1.78}{3} \approx 0.593
    $$

    **(b)** Weighted-averaged (weighted by support, total $= 1000$):

    $$
    \text{Precision}_{\text{weighted}} = \frac{900(0.95) + 80(0.70) + 20(0.40)}{1000} = \frac{855 + 56 + 8}{1000} = 0.919
    $$

    $$
    \text{Recall}_{\text{weighted}} = \frac{900(0.98) + 80(0.50) + 20(0.30)}{1000} = \frac{882 + 40 + 6}{1000} = 0.928
    $$

    **(c)** Macro-averaging treats all classes equally regardless of size. The poor performance on Class 2 (precision 0.40, recall 0.30) pulls down the macro-averaged metrics to 0.68 and 0.59, flagging that the model struggles on rare classes. Weighted-averaging (and similarly micro-averaging, which is equivalent to accuracy for single-label classification) is dominated by the majority class (Class 0, 90% of the data), producing metrics above 0.90 that obscure the near-failure on minority classes. When the cost of misclassifying minority classes is high (e.g., rare disease detection, fraud), macro-averaging provides a more honest assessment of model quality.

---

## Exercise 7: Softmax Regression Weight Matrix

In softmax regression with $C = 3$ classes and $d = 2$ input features (plus a bias), the model computes:

$$
\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}
$$

where $\mathbf{W} \in \mathbb{R}^{3 \times 2}$ and $\mathbf{b} \in \mathbb{R}^3$.

**(a)** Suppose the learned parameters are:

$$
\mathbf{W} = \begin{pmatrix} 2 & -1 \\ -1 & 2 \\ 0 & 0 \end{pmatrix}, \quad \mathbf{b} = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}
$$

For input $\mathbf{x} = (1, 1)^\top$, compute the logits and the predicted class.

**(b)** Explain the geometric meaning of the weight matrix. How does each row of $\mathbf{W}$ define a linear classifier?

**(c)** The model has a redundant parameterization: for any vector $\mathbf{v}$, the parameters $(\mathbf{W}', \mathbf{b}')$ where $\mathbf{W}'$ has rows $\mathbf{w}_k - \mathbf{v}^\top$ and $\mathbf{b}' = \mathbf{b} - \mathbf{v}$ produce the same softmax output. Use this to reduce the model to $C - 1 = 2$ free weight vectors by setting the last row to zero.

??? success "Solution"

    **(a)** Computing the logits:

    $$
    \mathbf{z} = \begin{pmatrix} 2 & -1 \\ -1 & 2 \\ 0 & 0 \end{pmatrix} \begin{pmatrix} 1 \\ 1 \end{pmatrix} + \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix} = \begin{pmatrix} 1 \\ 1 \\ 0 \end{pmatrix} + \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix} = \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix}
    $$

    Since all logits are equal, $\text{softmax}(\mathbf{z}) = (1/3, 1/3, 1/3)^\top$. No class is preferred — the model is maximally uncertain for this input. Any class could be predicted (e.g., class 1 by convention).

    **(b)** Each row $\mathbf{w}_k^\top$ of $\mathbf{W}$ defines a linear scoring function $z_k = \mathbf{w}_k^\top \mathbf{x} + b_k$. Geometrically, $\mathbf{w}_k$ is a direction in feature space along which class $k$ scores increase. The decision boundary between class $i$ and class $j$ is the hyperplane where $z_i = z_j$, i.e., $(\mathbf{w}_i - \mathbf{w}_j)^\top \mathbf{x} + (b_i - b_j) = 0$. The weight vector $\mathbf{w}_i - \mathbf{w}_j$ is normal to this hyperplane.

    **(c)** Choose $\mathbf{v} = \mathbf{w}_3 = (0, 0)^\top$ (the third row). The reduced parameters have rows $\mathbf{w}_k' = \mathbf{w}_k - \mathbf{w}_3$:

    $$
    \mathbf{W}' = \begin{pmatrix} 2 & -1 \\ -1 & 2 \\ 0 & 0 \end{pmatrix}, \quad \mathbf{b}' = \begin{pmatrix} -1 \\ -1 \\ 0 \end{pmatrix}
    $$

    In this case, the third row is already zero, so $\mathbf{W}' = \mathbf{W}$ and $\mathbf{b}' = \mathbf{b} - (1,1,1)^\top \cdot 0$. Actually, we subtract $\mathbf{v} = (0,0)$ from rows and $v_b = b_3 = 1$ from biases:

    $$
    \mathbf{b}' = \begin{pmatrix} 0 - 1 \\ 0 - 1 \\ 1 - 1 \end{pmatrix} = \begin{pmatrix} -1 \\ -1 \\ 0 \end{pmatrix}
    $$

    The third class now serves as a reference with $z_3' = 0$. The model has $2 \times 2 + 2 = 6$ free parameters instead of $3 \times 2 + 3 = 9$, with no change in the softmax output (by shift invariance). This reduction makes the parameterization identifiable.

---

## Exercise 8: One-vs-Rest versus Softmax

Consider a 4-class problem where a one-vs-rest (OvR) approach trains 4 binary classifiers, and a softmax model is trained natively.

**(a)** The OvR classifiers produce the following scores for a test example:

| Classifier | P(class $k$ vs rest) |
|:---:|:---:|
| Class 1 vs rest | 0.80 |
| Class 2 vs rest | 0.65 |
| Class 3 vs rest | 0.40 |
| Class 4 vs rest | 0.55 |

Do these probabilities form a valid probability distribution? Explain.

**(b)** How would you make a prediction from the OvR scores? What is the predicted class?

**(c)** A softmax model produces $\hat{\mathbf{p}} = (0.45, 0.30, 0.10, 0.15)^\top$ for the same example. How do these probabilities differ fundamentally from the OvR scores?

**(d)** State two advantages of native softmax regression over the OvR approach.

??? success "Solution"

    **(a)** The scores sum to $0.80 + 0.65 + 0.40 + 0.55 = 2.40 \ne 1$. They do **not** form a valid probability distribution. Each OvR classifier independently estimates the probability that the example belongs to its class versus all others. These individual binary probabilities are not constrained to sum to 1, and their magnitudes depend on the decision boundary of each separate binary model.

    **(b)** The standard OvR prediction assigns the class with the highest score: $\arg\max_k \text{score}_k = 1$, since Class 1 has the highest score (0.80). While simple, this rule can lead to ties and does not produce calibrated probabilities. One could normalize the scores by dividing by their sum ($0.80/2.40 = 0.333$), but this is ad hoc and does not guarantee well-calibrated probabilities.

    **(c)** The softmax probabilities $\hat{\mathbf{p}} = (0.45, 0.30, 0.10, 0.15)^\top$ sum to exactly 1 and are jointly estimated by a single model. They represent a coherent probability distribution over all classes, calibrated through the softmax function. The OvR scores, by contrast, come from independently trained classifiers that do not communicate during training.

    **(d)** Two advantages of native softmax over OvR:

    1. **Calibrated probabilities**: Softmax inherently produces a valid probability distribution (non-negative, sums to 1) without post-hoc normalization. This makes the outputs directly interpretable as class probabilities and suitable for downstream probabilistic reasoning.

    2. **Joint training**: The softmax model's weight matrix is optimized to separate all classes simultaneously, allowing the model to share information across class boundaries. OvR trains each classifier independently, so it cannot exploit the structure of the multiclass problem. For example, features useful for distinguishing Class 1 from Class 2 might also help distinguish Class 3 from Class 4, but OvR cannot leverage this.

---

## Exercise 9: Regularized Softmax Regression

Consider softmax regression with weight matrix $\mathbf{W} \in \mathbb{R}^{C \times d}$ and L2 regularization. The regularized loss is:

$$
\mathcal{L}(\mathbf{W}) = -\frac{1}{N}\sum_{i=1}^N \sum_{k=1}^C y_{ik} \log \hat{p}_{ik} + \frac{\lambda}{2}\|\mathbf{W}\|_F^2
$$

where $\|\mathbf{W}\|_F^2 = \sum_{j,k} W_{jk}^2$ is the squared Frobenius norm.

**(a)** Write the gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{W}}$ including the regularization term.

**(b)** How does the regularization term affect the weight update in gradient descent?

**(c)** As $\lambda \to \infty$, what happens to the weight matrix and the predicted probabilities? As $\lambda \to 0$?

**(d)** Explain why regularization is especially important when the input dimension $d$ is much larger than the number of training examples $N$.

??? success "Solution"

    **(a)** The cross-entropy gradient for a single example is $\hat{\mathbf{p}}_i - \mathbf{y}_i$ with respect to the logits, which translates to $(\hat{\mathbf{p}}_i - \mathbf{y}_i)\mathbf{x}_i^\top$ with respect to $\mathbf{W}$. Averaging over the dataset and adding the regularization gradient:

    $$
    \frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \frac{1}{N}\sum_{i=1}^N (\hat{\mathbf{p}}_i - \mathbf{y}_i)\mathbf{x}_i^\top + \lambda \mathbf{W}
    $$

    **(b)** The gradient descent update becomes:

    $$
    \mathbf{W} \leftarrow \mathbf{W} - \eta\left[\frac{1}{N}\sum_{i=1}^N (\hat{\mathbf{p}}_i - \mathbf{y}_i)\mathbf{x}_i^\top + \lambda \mathbf{W}\right]
    $$

    $$
    = (1 - \eta\lambda)\mathbf{W} - \frac{\eta}{N}\sum_{i=1}^N (\hat{\mathbf{p}}_i - \mathbf{y}_i)\mathbf{x}_i^\top
    $$

    The regularization term shrinks all weights toward zero by the factor $(1 - \eta\lambda)$ at each step (weight decay). This prevents any single weight from becoming excessively large.

    **(c)**

    - **$\lambda \to \infty$**: The penalty dominates, forcing $\mathbf{W} \to \mathbf{0}$. With $\mathbf{z} = \mathbf{b}$ (only biases remain), the model predicts the same class distribution for every input — it degeneralizes to always predicting the prior class distribution. If biases are also regularized, predictions approach $(1/C, \ldots, 1/C)$.
    - **$\lambda \to 0$**: No regularization. The model fits the training data as closely as possible, potentially overfitting by learning weights that perfectly separate training examples but generalize poorly.

    **(d)** When $d \gg N$, the model has far more parameters ($Cd$) than training constraints ($N$). This creates a highly underdetermined system where many weight configurations fit the training data perfectly (zero training loss). Without regularization, gradient descent can converge to any of these solutions, and the one found may have large weights that produce extreme, poorly-calibrated predictions on new data. L2 regularization constrains the solution to have small weights, acting as an implicit prior that favors simpler models and reduces overfitting.

---

## Exercise 10: MNIST-Style Classification

A softmax classifier with $C = 10$ digit classes and $d = 784$ input features (28 x 28 pixel images) is trained on MNIST.

**(a)** How many parameters does this model have (weights and biases)?

**(b)** After training, the confusion matrix reveals that digits 4 and 9 are frequently confused (high off-diagonal entries in the (4,9) and (9,4) cells). Propose two strategies to reduce this confusion.

**(c)** The test accuracy is 92%. A two-layer network with 256 hidden units and ReLU activation achieves 97%. Explain the source of this improvement in terms of the model's representational capacity.

**(d)** For a test image that the model classifies as digit 3 with $\hat{p}_3 = 0.52$ and $\hat{p}_5 = 0.35$, should we trust this prediction? How could you use the softmax output to flag uncertain predictions?

??? success "Solution"

    **(a)** The weight matrix has $C \times d = 10 \times 784 = 7{,}840$ entries, and the bias vector has $C = 10$ entries. Total: $7{,}840 + 10 = 7{,}850$ parameters.

    **(b)** Two strategies to reduce 4/9 confusion:

    1. **Feature engineering or data augmentation**: Digits 4 and 9 share structural features (a vertical stroke on the right). Adding slightly rotated, scaled, or thickened versions of 4s and 9s to the training set would help the model learn the distinguishing features (the closed vs. open top loop).

    2. **Increase model capacity**: A two-layer network or CNN can learn nonlinear feature combinations (e.g., detecting the presence of a closed loop at the top for 9 vs. the open angular junction for 4) that a single linear layer cannot represent. Convolutional layers are particularly effective because they detect local spatial patterns.

    **(c)** A single-layer softmax model computes $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$, which is a linear function of the raw pixels. It can only learn linear decision boundaries in the 784-dimensional pixel space. A two-layer network computes $\mathbf{h} = \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$ followed by $\mathbf{z} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2$. The hidden layer with ReLU activation learns a nonlinear feature representation $\mathbf{h}$ where digits are more linearly separable. With 256 hidden units, the model can detect and combine stroke patterns, curves, and intersections — intermediate features that are more discriminative than raw pixel values.

    **(d)** The prediction $\hat{p}_3 = 0.52$ should not be trusted with high confidence. The maximum probability is barely above $1/C = 0.10$, and the second-most-likely class ($\hat{p}_5 = 0.35$) is close behind, indicating substantial uncertainty.

    A simple uncertainty flagging strategy: define a confidence threshold $\tau$ (e.g., $\tau = 0.80$) and flag any prediction where $\max_k \hat{p}_k < \tau$ as "uncertain." Alternatively, use the **entropy** of the predicted distribution:

    $$
    H(\hat{\mathbf{p}}) = -\sum_k \hat{p}_k \log \hat{p}_k
    $$

    High entropy indicates high uncertainty. For a 10-class problem, maximum entropy is $\log(10) \approx 2.30$ (uniform prediction). A threshold on entropy provides a principled way to route uncertain examples to human review or a more powerful model.
