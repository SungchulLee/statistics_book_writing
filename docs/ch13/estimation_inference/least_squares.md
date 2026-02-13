# Least Squares Estimation

This section derives the optimal parameters for linear regression from three perspectives: the ordinary least squares (OLS) criterion, the maximum likelihood principle under Gaussian errors, and the normal equation with its vector calculus proof.

---

## 1. Maximum Likelihood Estimation for Linear Regression

### Data and Model

Given a dataset $\{(x^{(i)}, y^{(i)}): i = 1, \ldots, m\}$ following the linear regression model:

$$
y^{(i)} = \alpha + \beta x^{(i)} + \varepsilon^{(i)}
$$

where $\varepsilon^{(i)} \sim N(0, \sigma^2)$ with fixed $\sigma^2$.

### Likelihood Function

Since each $y^{(i)}$ is normally distributed as $y^{(i)} \sim N(\alpha + \beta x^{(i)}, \sigma^2)$, the likelihood function is:

$$
L(\alpha, \beta) = \prod_{i=1}^m \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{1}{2\sigma^2} \left(y^{(i)} - \alpha - \beta x^{(i)}\right)^2\right)
$$

### Log-Likelihood Function

Taking the natural logarithm:

$$
l(\alpha, \beta) = -\frac{1}{2\sigma^2} \sum_{i=1}^m \left(y^{(i)} - \alpha - \beta x^{(i)}\right)^2 + \text{Constant}
$$

### Cost Function (Square Loss)

Define the cost function:

$$
J(\alpha, \beta) = \frac{1}{m} \sum_{i=1}^m \left(y^{(i)} - \alpha - \beta x^{(i)}\right)^2
$$

### MLE–OLS Equivalence

Maximizing the likelihood is equivalent to minimizing the squared loss, since the constant and $\sigma^2$ do not depend on $\alpha$ or $\beta$:

$$
\text{argmax}_{\alpha, \beta}\ L \quad \Leftrightarrow \quad \text{argmax}_{\alpha, \beta}\ l \quad \Leftrightarrow \quad \text{argmin}_{\alpha, \beta}\ J
$$

This is a foundational result: **under Gaussian error assumptions, OLS and MLE produce the same parameter estimates.**

---

## 2. Normal Equation (Simple Case)

Setting the partial derivatives of the L2 loss to zero:

$$
l = \frac{1}{n} \sum_{i=1}^n (\alpha + \beta x_i - y_i)^2
$$

$$
\begin{array}{lll}
\displaystyle \frac{\partial l}{\partial \alpha} = \frac{2}{n} \sum_{i=1}^n \left((\alpha + \beta x_i) - y_i\right) = 0
& \Rightarrow &
2\alpha + 2\beta \bar{x} - 2\bar{y} = 0 \\[8pt]
\displaystyle \frac{\partial l}{\partial \beta} = \frac{2}{n} \sum_{i=1}^n \left((\alpha + \beta x_i) - y_i\right) x_i = 0
& \Rightarrow &
2\alpha \bar{x} + 2\beta \overline{x^2} - 2\overline{xy} = 0
\end{array}
$$

Solving:

$$
\begin{array}{lll}
\beta &=& \displaystyle \frac{\overline{xy} - \bar{x}\bar{y}}{\overline{x^2} - (\bar{x})^2}
= \frac{s_{xy}}{s_x^2}
= \frac{\rho\, s_x\, s_y}{s_x^2}
= \rho \frac{s_y}{s_x} \\[8pt]
\alpha &=& \displaystyle -\rho \frac{s_y}{s_x} \bar{x} + \bar{y}
\end{array}
$$

---

## 3. Normal Equation (Matrix Form)

For the general case with $d$ predictors:

$$
\text{argmin}_{\boldsymbol{\theta}} \; J(\boldsymbol{\theta})
\quad \Rightarrow \quad
\mathbf{X}^T \mathbf{X} \boldsymbol{\theta} = \mathbf{X}^T \mathbf{y}
\quad \Rightarrow \quad
\hat{\boldsymbol{\theta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

### Vector Calculus Identities

The following identities are used in the derivation:

$$
\begin{aligned}
(1) \quad & \frac{\partial (\mathbf{a}^T \mathbf{b})}{\partial \mathbf{a}} = \mathbf{b} \\[4pt]
(2) \quad & \frac{\partial (\mathbf{a}^T \mathbf{b})}{\partial \mathbf{b}} = \mathbf{a} \\[4pt]
(3) \quad & \frac{\partial \text{tr}(\mathbf{A}\mathbf{B})}{\partial \mathbf{A}} = \mathbf{B}^T \\[4pt]
(4) \quad & \frac{\partial \text{tr}(\mathbf{A}\mathbf{B})}{\partial \mathbf{B}} = \mathbf{A}^T \\[4pt]
(5) \quad & \frac{\partial |\mathbf{A}|}{\partial \mathbf{A}} = \mathbf{C} \quad \text{(where $\mathbf{C}$ is the cofactor matrix of $\mathbf{A}$)} \\[4pt]
(6) \quad & \frac{\partial \log|\mathbf{A}|}{\partial \mathbf{A}} = \mathbf{A}^{-T} := (\mathbf{A}^{-1})^T
\end{aligned}
$$

### Proof of the Normal Equation

Starting from the cost function:

$$
\begin{aligned}
J(\boldsymbol{\theta})
&= \frac{1}{2m} \| \mathbf{X}\boldsymbol{\theta} - \mathbf{y} \|^2 \\[4pt]
&= \frac{1}{2m} (\mathbf{X}\boldsymbol{\theta} - \mathbf{y})^T (\mathbf{X}\boldsymbol{\theta} - \mathbf{y}) \\[4pt]
&= \frac{1}{2m} \left( \boldsymbol{\theta}^T \mathbf{X}^T \mathbf{X} \boldsymbol{\theta} - \boldsymbol{\theta}^T \mathbf{X}^T \mathbf{y} - \mathbf{y}^T \mathbf{X} \boldsymbol{\theta} + \mathbf{y}^T \mathbf{y} \right)
\end{aligned}
$$

Taking the derivative with respect to $\boldsymbol{\theta}$:

$$
\begin{aligned}
\frac{\partial J}{\partial \boldsymbol{\theta}}
&= \frac{1}{2m} \left( \mathbf{X}^T \mathbf{X} \boldsymbol{\theta} + (\boldsymbol{\theta}^T \mathbf{X}^T \mathbf{X})^T - \mathbf{X}^T \mathbf{y} - (\mathbf{y}^T \mathbf{X})^T \right) \\[4pt]
&= \frac{1}{m} \left( \mathbf{X}^T \mathbf{X} \boldsymbol{\theta} - \mathbf{X}^T \mathbf{y} \right) \\[4pt]
&= \mathbf{0}
\end{aligned}
$$

This gives the **normal equation** $\mathbf{X}^T \mathbf{X} \boldsymbol{\theta} = \mathbf{X}^T \mathbf{y}$, with the closed-form solution:

$$
\hat{\boldsymbol{\theta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

!!! note "When Does the Inverse Exist?"
    The matrix $\mathbf{X}^T\mathbf{X}$ is invertible if and only if $\mathbf{X}$ has full column rank, i.e., no predictor is a perfect linear combination of the others. When this condition fails (multicollinearity), regularization techniques such as ridge regression can be used.

---

## 4. Prediction

### General Case

Given a new input $\mathbf{x}$:

$$
\mathbf{x}
\quad \Rightarrow \quad
\mathbf{x} = [1, \mathbf{x}]
\quad \Rightarrow \quad
\hat{y} = \mathbf{x} \hat{\boldsymbol{\theta}}
$$

### Simple Linear Regression

In the special case of a single predictor, the prediction formula reduces to the elegant standardized form:

$$
\frac{y - \bar{y}}{s_y} = \rho \frac{x - \bar{x}}{s_x}
$$

**Proof**: From the normal equations of the simple case, we derived $\beta = \rho\, s_y / s_x$ and $\alpha = \bar{y} - \beta \bar{x}$, so:

$$
\begin{aligned}
y &= \alpha + \beta x \\
  &= \bar{y} - \rho \frac{s_y}{s_x} \bar{x} + \rho \frac{s_y}{s_x} x \\
  &= \rho \frac{s_y}{s_x}(x - \bar{x}) + \bar{y}
\end{aligned}
$$

Rearranging: $\displaystyle \frac{y - \bar{y}}{s_y} = \rho \frac{x - \bar{x}}{s_x}$.

---

## 5. Exercises

### Exercise 1: MLE Derivation

You are given a dataset $\{(x^{(i)}, y^{(i)}): i = 1, \ldots, m\}$ that follows the linear regression model:

$$
y^{(i)} = \alpha + \beta x^{(i)} + \varepsilon^{(i)},
$$

where $\varepsilon^{(i)} \sim N(0, \sigma^2)$ with fixed $\sigma^2$.

**(a)** Derive the likelihood function $L(\alpha, \beta)$ for the given model.

**(b)** Derive the log-likelihood function $l(\alpha, \beta)$ for the given model.

**(c)** When the cost function (square loss) is defined as:

$$
J(\alpha, \beta) = \frac{1}{2} \sum_{i=1}^m \left(y^{(i)} - \alpha - \beta x^{(i)}\right)^2,
$$

explain the relationship between minimizing the square loss and maximizing the likelihood function.

**(d)** Derive the normal equations that must be satisfied by the parameters $\hat{\alpha}$ and $\hat{\beta}$ that minimize the square loss.

**(e)** Solve the normal equations to compute the parameters $\hat{\alpha}$ and $\hat{\beta}$ that minimize the square loss.

**(f)** Extend the model to the case of multiple independent variables, where $d$ predictors are available:

$$
\mathbf{x}^{(i)} = (x_1^{(i)}, x_2^{(i)}, \ldots, x_d^{(i)}), \quad 1 \leq i \leq m.
$$

Represent the square loss function in terms of matrix operations using the design matrix $\mathbf{X}$, parameter vector $\boldsymbol{\theta}$, and response vector $\mathbf{y}$.

??? success "Solution"

    **(a) Likelihood Function**

    Assuming $\varepsilon^{(i)} \sim N(0, \sigma^2)$, each $y^{(i)}$ is distributed as $y^{(i)} \sim N(\alpha + \beta x^{(i)}, \sigma^2)$. The likelihood function is:

    $$
    L(\alpha, \beta) = \prod_{i=1}^m \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{1}{2\sigma^2} \left(y^{(i)} - \alpha - \beta x^{(i)}\right)^2\right)
    $$

    **(b) Log-Likelihood Function**

    $$
    l(\alpha, \beta) = \ln L(\alpha, \beta) = -\frac{1}{2\sigma^2} \sum_{i=1}^m \left(y^{(i)} - \alpha - \beta x^{(i)}\right)^2 + \text{Constant}
    $$

    **(c) MLE–OLS Equivalence**

    The square loss function is:

    $$
    J(\alpha, \beta) = \frac{1}{2} \sum_{i=1}^m \left(y^{(i)} - \alpha - \beta x^{(i)}\right)^2
    $$

    Maximizing the log-likelihood is equivalent to minimizing the residual sum of squares (since the constant and $\sigma^2$ do not depend on $\alpha$ or $\beta$):

    $$
    \text{argmax}_{\alpha, \beta}\ L(\alpha, \beta) \quad \Leftrightarrow \quad \text{argmax}_{\alpha, \beta}\ l(\alpha, \beta) \quad \Leftrightarrow \quad \text{argmin}_{\alpha, \beta}\ J(\alpha, \beta)
    $$

    **(d) Normal Equations**

    Taking partial derivatives and setting them to zero:

    1. $\displaystyle \frac{\partial J}{\partial \alpha} = \sum_{i=1}^m \left(\alpha + \beta x^{(i)} - y^{(i)}\right) = 0$

    2. $\displaystyle \frac{\partial J}{\partial \beta} = \sum_{i=1}^m \left(\alpha + \beta x^{(i)} - y^{(i)}\right) x^{(i)} = 0$

    These lead to the normal equations:

    $$
    \begin{aligned}
    \alpha + \beta \bar{x} &= \bar{y} \\
    \beta \overline{x^2} + \alpha \bar{x} &= \overline{xy}
    \end{aligned}
    $$

    **(e) Solving the Normal Equations**

    $$
    \beta = \frac{\overline{xy} - \bar{x}\bar{y}}{\overline{x^2} - (\bar{x})^2} = \frac{\text{Cov}(X, Y)}{\text{Var}(X)} = \rho \frac{\sigma_y}{\sigma_x}
    $$

    $$
    \alpha = \bar{y} - \beta \bar{x}
    $$

    **(f) Matrix Representation**

    Construct the design matrix $\mathbf{X}$, parameter vector $\boldsymbol{\theta}$, and output vector $\mathbf{y}$:

    $$
    \mathbf{X} =
    \begin{bmatrix}
    1 & x_1^{(1)} & \cdots & x_d^{(1)} \\
    \vdots & \vdots & \ddots & \vdots \\
    1 & x_1^{(m)} & \cdots & x_d^{(m)}
    \end{bmatrix}, \quad
    \boldsymbol{\theta} =
    \begin{bmatrix}
    \beta_0 \\ \beta_1 \\ \vdots \\ \beta_d
    \end{bmatrix}, \quad
    \mathbf{y} =
    \begin{bmatrix}
    y^{(1)} \\ \vdots \\ y^{(m)}
    \end{bmatrix}
    $$

    The square loss function is:

    $$
    J(\boldsymbol{\theta}) = \frac{1}{2m} \| \mathbf{X}\boldsymbol{\theta} - \mathbf{y} \|^2 = \frac{1}{2m} (\mathbf{X}\boldsymbol{\theta} - \mathbf{y})^T (\mathbf{X}\boldsymbol{\theta} - \mathbf{y})
    $$
