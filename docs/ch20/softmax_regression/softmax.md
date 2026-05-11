# Softmax Function and Probability Simplex


## Definition

The **softmax** function maps a vector of $C$ real-valued logits
$\mathbf{z}=(z_1,\ldots,z_C)$ to a probability distribution:

$$
\operatorname{softmax}(\mathbf{z})_c
= \frac{e^{z_c}}{\sum_{c'=1}^{C}e^{z_{c'}}},
\qquad c=1,\ldots,C
$$

### Properties

1. **Non-negative:** Every output is strictly positive.
2. **Sums to one:** $\sum_c\operatorname{softmax}(\mathbf{z})_c = 1$.
3. **Monotone:** A larger logit $z_c$ produces a larger probability.
4. **Shift invariance:** $\operatorname{softmax}(\mathbf{z}+\alpha\mathbf{1})
   = \operatorname{softmax}(\mathbf{z})$ for any scalar $\alpha$.

Property 4 is exploited for numerical stability: before exponentiation
we subtract $\max_c z_c$.

## The Probability Simplex

The output of softmax lies on the **probability simplex**

$$
\Delta^{C-1} = \Bigl\{\mathbf{p}\in\mathbb{R}^C : p_c\ge 0,\;
\sum_c p_c = 1\Bigr\}
$$

For $C=3$ this is a triangle in 3D space; for $C=10$ (MNIST) it is a
9-dimensional simplex.

## Softmax as a Generalization of the Sigmoid

When $C=2$ with logits $(z_1,z_2)$:

$$
\operatorname{softmax}(\mathbf{z})_1
= \frac{e^{z_1}}{e^{z_1}+e^{z_2}}
= \frac{1}{1+e^{-(z_1-z_2)}}
= \sigma(z_1-z_2)
$$

Thus the binary softmax is exactly the sigmoid applied to the difference
of the two logits.

## Temperature Scaling

A common variant introduces a temperature parameter $\tau>0$:

$$
\operatorname{softmax}(\mathbf{z}/\tau)_c
= \frac{e^{z_c/\tau}}{\sum_{c'}e^{z_{c'}/\tau}}
$$

As $\tau\to 0$ the distribution collapses to a point mass on
$\arg\max_c z_c$ (hard decision); as $\tau\to\infty$ it approaches the
uniform distribution.  Temperature scaling is used in model calibration
and in generative models (e.g. controlling the "creativity" of language
models).

## NumPy Implementation

```python
import numpy as np

def softmax(z):
    """Numerically stable softmax."""
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
```

The `np.max` subtraction prevents overflow in `np.exp` without
changing the result (shift invariance).

## Exercises

**Exercise 1.**
Softmax Function Computation

Consider a 3-class classification problem with logit vector $\mathbf{z} = (2, 1, -1)^\top$.

**(a)** Compute the softmax probabilities $\hat{p}_k = \text{softmax}(\mathbf{z})_k$ for $k = 1, 2, 3$ using the formula:

$$
\hat{p}_k = \frac{e^{z_k}}{\sum_{j=1}^C e^{z_j}}
$$

**(b)** Verify that the probabilities sum to 1 and that each is non-negative.

**(c)** Show that the softmax function is shift-invariant: for any constant $c$, $\text{softmax}(\mathbf{z} + c\mathbf{1}) = \text{softmax}(\mathbf{z})$. Why is this property important for numerical stability?

??? success "Solution to Exercise 1"

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

**Exercise 2.**
Softmax Regression Weight Matrix

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

??? success "Solution to Exercise 2"

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
