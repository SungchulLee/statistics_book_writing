# Chapter 20: Softmax Regression

## Overview

This chapter extends binary logistic regression to the multiclass setting by developing softmax regression (multinomial logistic regression). Starting from the softmax function and its geometric interpretation on the probability simplex, we derive the cross-entropy loss and its gradients, explore regularization and optimization strategies, and evaluate multiclass models using confusion matrices and averaging schemes. The chapter also covers alternative decomposition strategies---one-vs-rest and one-vs-one---and concludes with a hands-on MNIST case study.

---

## Chapter Structure

### 20.1 Softmax Regression

The foundations of multiclass classification via the softmax model:

- **Multinomial Logistic Regression** --- Generalizes binary logistic regression to $C > 2$ classes by learning a weight matrix and bias vector that produce per-class logits, establishing the single-layer softmax architecture.
- **Softmax Function and Probability Simplex** --- Defines the softmax mapping from real-valued logits to a valid probability distribution on the $(C{-}1)$-simplex, covering key properties such as non-negativity, normalization, monotonicity, and shift invariance.
- **Numerical Stability (Log-Sum-Exp Trick)** --- Addresses the overflow and underflow issues that arise when exponentiating large logits, introducing the log-sum-exp trick to ensure stable computation.

### 20.2 Estimation and Optimization

Training the softmax model via gradient-based methods:

- **Cross-Entropy Loss** --- Derives the categorical cross-entropy objective for one-hot encoded labels and shows that its gradient with respect to the logits has the elegant form $\hat{\mathbf{Y}} - \mathbf{Y}$.
- **Gradient-Based Optimization** --- Presents full gradient derivations for both single-layer and two-layer networks (with logistic hidden activations), including output-layer and hidden-layer backpropagation steps.
- **Regularization for Softmax** --- Discusses L1 and L2 penalty terms added to the cross-entropy loss to prevent overfitting and improve generalization in multiclass models.

### 20.3 Model Evaluation and Case Studies

Assessing multiclass classifier performance and applying the theory to real data:

- **Multiclass Metrics (Accuracy, Confusion Matrix)** --- Introduces accuracy and the $C \times C$ confusion matrix for diagnosing per-class misclassification patterns, with guidance on extracting precision and recall from the matrix.
- **Macro, Micro, and Weighted Averaging** --- Explains the three principal strategies for aggregating per-class precision, recall, and F1-score into a single summary metric, and when each is appropriate.
- **MNIST Case Study** --- Applies softmax regression and deeper architectures (two-layer network, simple CNN) to the MNIST handwritten digit dataset, comparing model complexity, training procedures, and test-set performance.

### 20.4 One-vs-Rest and One-vs-One Strategies

Alternative approaches to multiclass classification that reduce it to a collection of binary problems:

- **OvR (One-vs-Rest) Approach** --- Trains $C$ binary classifiers, each distinguishing one class from all others, and combines their outputs for prediction.
- **OvO (One-vs-One) Approach** --- Trains $\binom{C}{2}$ binary classifiers, one for each pair of classes, and aggregates predictions via majority voting.
- **Comparison with Softmax** --- Contrasts the OvR and OvO decomposition strategies with native softmax regression in terms of computational cost, calibration, and scalability.

### 20.5 Code

Complete Python implementations:

- **Softmax Regression Implementation** --- End-to-end softmax regression with gradient descent.
- **Multiclass Evaluation Metrics** --- Functions for confusion matrices, per-class precision/recall, and averaging schemes.
- **MNIST Classification** --- Full MNIST pipeline comparing single-layer, two-layer, and CNN models in PyTorch.

### 20.6 Exercises

Practice problems covering the softmax function, cross-entropy loss derivations, multiclass evaluation metrics, and comparisons between softmax regression and OvR/OvO strategies.

---

## Prerequisites

This chapter builds on:

- **Chapter 19** (Logistic Regression) --- The logit link, odds ratios, maximum likelihood estimation for binary classification, and evaluation metrics such as ROC curves and the confusion matrix.
- **Chapter 18** (Regularization Techniques) --- L1 and L2 penalties, the bias--variance tradeoff, and cross-validation for hyperparameter tuning.
- **Chapter 13** (Linear Regression) --- Least squares estimation, gradient computation, and model evaluation, which underpin the linear scoring layer in softmax regression.
- **Chapter 6** (Statistical Estimation) --- Maximum likelihood estimation principles and Fisher information, used throughout the optimization derivations.

---

## Key Takeaways

1. Softmax regression is the natural multiclass generalization of logistic regression, mapping $C$ logits through the softmax function to produce a valid probability distribution over classes.
2. Cross-entropy loss, combined with the softmax function, yields a clean gradient ($\hat{\mathbf{Y}} - \mathbf{Y}$) that makes gradient-based optimization straightforward and efficient.
3. Numerical stability requires the log-sum-exp trick: subtracting the maximum logit before exponentiation prevents overflow without changing the result.
4. Multiclass evaluation goes beyond accuracy---confusion matrices, per-class metrics, and macro/micro/weighted averaging reveal performance disparities across classes.
5. One-vs-rest and one-vs-one are viable alternatives to native softmax regression, but softmax provides naturally calibrated probabilities and scales more gracefully to many classes.
6. Regularization (L1/L2 penalties on the weight matrix) is essential when the feature dimension is large relative to the number of training examples, as in image classification tasks like MNIST.
