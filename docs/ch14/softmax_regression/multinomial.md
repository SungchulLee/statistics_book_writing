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
