# Multinomial Logistic Regression


## From Binary to Multiclass

Logistic regression models a binary response.  When the response has
$C>2$ categories, we generalize to **multinomial logistic regression**
(also called **softmax regression**).  Instead of a single weight vector
$\boldsymbol{\theta}$, we learn a weight matrix $\mathbf{W}$ and a bias
vector $\mathbf{b}$ that map each input to a vector of $C$ real-valued
scores (logits), one per class.

## Model Architecture (Single Layer)

For a dataset with $n$ observations and $p$ features, the single-layer
softmax model computes:

$$
\underset{n \times C}{\mathbf{Z}}
= \underset{n \times p}{\mathbf{X}}\;
  \underset{p \times C}{\mathbf{W}}

  + \underset{1 \times C}{\mathbf{b}}
$$

$$
\underset{n \times C}{\hat{\mathbf{Y}}}
= \operatorname{softmax}(\mathbf{Z})
$$

where each row of $\hat{\mathbf{Y}}$ is a probability distribution over
the $C$ classes.

## Two-Layer Model (Hidden Layer + Softmax)

Adding a hidden layer with the logistic activation gives a shallow
neural network — the architecture used in the MNIST examples below.

$$
\begin{aligned}
\underset{n \times 100}{\mathbf{Z}^h}
  &= \underset{n \times 784}{\mathbf{X}}\;
     \underset{784 \times 100}{\mathbf{W}^h}

     + \underset{1 \times 100}{\mathbf{b}^h} \\[4pt]
\underset{n \times 100}{\mathbf{H}}
  &= \operatorname{logistic}\!\bigl(\mathbf{Z}^h\bigr) \\[4pt]
\underset{n \times 10}{\mathbf{Z}^o}
  &= \underset{n \times 100}{\mathbf{H}}\;
     \underset{100 \times 10}{\mathbf{W}^o}

     + \underset{1 \times 10}{\mathbf{b}^o} \\[4pt]
\underset{n \times 10}{\hat{\mathbf{Y}}}
  &= \operatorname{softmax}\!\bigl(\mathbf{Z}^o\bigr)
\end{aligned}
$$

The logistic (sigmoid) activation is

$$
\operatorname{logistic}(x) = \frac{1}{1+e^{-x}},
\qquad
\operatorname{logistic}'(x) = \operatorname{logistic}(x)\bigl(1-\operatorname{logistic}(x)\bigr)
$$

## MNIST Data

The MNIST dataset is the canonical benchmark for this model family:

$$
\mathbf{X} \in \mathbb{R}^{n\times 784},\quad
\mathbf{Y} \in \{0,1\}^{n\times 10}\;\text{(one-hot)},\quad
\mathbf{y}_{\text{cls}} \in \{0,\ldots,9\}^n
$$

Each image is $28\times 28$ pixels, flattened to a 784-dimensional
vector.  Pixel values are scaled to $[0,1]$.

## Relationship to Binary Logistic Regression

When $C=2$, multinomial logistic regression reduces to ordinary logistic
regression.  The two-class softmax produces the same decision boundary
as the sigmoid model because the log-ratio of class probabilities is
linear in the features:

$$
\log\frac{P(Y=1\mid\mathbf{x})}{P(Y=0\mid\mathbf{x})}
= (\mathbf{w}_1-\mathbf{w}_0)^T\mathbf{x} + (b_1-b_0)
$$


## Exercises

**Exercise 1.**
Describe the main concept of Multinomial Logistic Regression and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Multinomial Logistic Regression is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

---

**Exercise 2.**
State the key assumptions required by the method discussed here. How can each assumption be checked?

??? success "Solution to Exercise 2"
    The main assumptions typically include: (1) independence of observations -- verified by understanding the data collection process and checking for serial correlation; (2) distributional requirements (e.g., normality) -- checked with Q-Q plots and formal tests like Shapiro-Wilk; (3) equal variances (if applicable) -- assessed with boxplots and Levene's test. When assumptions are violated, consider robust alternatives, transformations, or nonparametric methods.

---

**Exercise 3.**
Work through a small numerical example illustrating the application of the technique from this section.

??? success "Solution to Exercise 3"
    A structured approach to applying this technique involves: (1) clearly stating the hypotheses or estimation goal; (2) verifying that the data meet the required assumptions; (3) computing the relevant test statistic, estimate, or model fit; (4) obtaining the p-value, confidence interval, or posterior distribution; (5) interpreting the result in the context of the original question. Following these steps systematically ensures a rigorous and reproducible analysis.

---

**Exercise 4.**
Compare the approach from this section with an alternative method. When would you choose each?

??? success "Solution to Exercise 4"
    The method discussed here is appropriate when its assumptions hold and the sample size is sufficient for the asymptotic approximations to be accurate. Alternative approaches include: (1) nonparametric methods -- preferred when distributional assumptions are suspect; (2) bootstrap methods -- useful when analytical reference distributions are unavailable; (3) Bayesian methods -- valuable when incorporating prior information or when direct probability statements about parameters are desired. Running multiple approaches and comparing results provides a useful robustness check.
