# Chapter 0: Prerequisites

## Overview

This chapter establishes the mathematical and computational foundations needed for the rest of the book. It begins with a review of core mathematical concepts (logic, sets, sequences, limits, and linear algebra notation), introduces the essential Python tools used throughout (NumPy, pandas, Matplotlib), and then develops the advanced linear algebra of square matrices and their connections to statistical theory via quadratic forms and OLS sampling distributions.

---

## Chapter Structure

### 0.1 Mathematical Background

A review of the foundational mathematical language used throughout the book:

- **Sets, Functions, and Logic** -- Covers propositional logic (statements, connectives, truth tables), set operations (union, intersection, complement), functions, and quantifiers that provide the precise language needed for probability and inference.
- **Sequences, Limits, and Asymptotics** -- Reviews sequences, convergence, monotone sequences, series, and asymptotic notation, forming the mathematical backbone for the Law of Large Numbers, the Central Limit Theorem, and estimator consistency.
- **Linear Algebra Notation and Conventions** -- Establishes vector and matrix notation (column vectors, dot products, norms), matrix operations, the design matrix, and key results (rank, inverse, determinant) used in regression and multivariate statistics.

### 0.2 Computational Tools

An introduction to the Python ecosystem for statistical computing:

- **Python and Jupyter Basics** -- Covers choosing a Python distribution (Anaconda), package management, Jupyter notebooks, and the core language features needed for statistical work.
- **NumPy Arrays** -- Introduces the ndarray, array creation, vectorized arithmetic, broadcasting, indexing, and the numerical operations that underpin every scientific Python library.
- **Data Handling with pandas** -- Covers the Series and DataFrame data structures, data loading, cleaning, transformation, grouping, and summarization for structured data analysis.
- **Basic Visualization with Matplotlib** -- Introduces the Figure-Axes model, line plots, scatter plots, histograms, bar charts, customization, and the plotting patterns used throughout the book.

### 0.3 Square Matrices

A systematic treatment of special matrix types that arise in regression and multivariate statistics:

- **Similar Matrices** -- Defines matrix similarity and its invariants (eigenvalues, trace, determinant), establishing the foundation for matrix decompositions.
- **Diagonal Form of Diagonalizable Matrices** -- Covers eigendecomposition and the conditions under which a matrix can be diagonalized, enabling simplified computation of matrix powers and functions.
- **Jordan Canonical Form** -- Extends diagonalization to non-diagonalizable matrices via Jordan blocks, providing a complete canonical form for square matrices.
- **Trace and Eigenvalues** -- Establishes the relationship between the trace of a matrix and the sum of its eigenvalues, a key identity used in quadratic form calculations.
- **Idempotent Matrices** -- Defines matrices satisfying A^2 = A and derives their eigenvalue properties, providing the algebraic foundation for projection and hat matrices.
- **Symmetric Matrices** -- Covers the spectral theorem for real symmetric matrices, guaranteeing real eigenvalues and orthogonal eigenvectors, with direct applications to covariance matrices.
- **Positive Definite Matrices** -- Defines positive definiteness via quadratic forms and eigenvalue conditions, essential for understanding covariance matrices and optimization in statistics.
- **Gram Matrices** -- Introduces X^T X matrices, their positive semi-definiteness, and their central role in least squares estimation and normal equations.
- **Projection Matrices** -- Covers general projection matrices and their geometric interpretation as mappings onto subspaces, key to understanding least squares residuals.
- **Orthogonal Projection Matrices** -- Specializes projections to the orthogonal case, deriving the hat matrix H = X(X^T X)^{-1} X^T used in regression diagnostics.

### 0.4 Linear Algebra and Statistics

Connects matrix theory to statistical inference:

- **Chi-Squared Distribution and Quadratic Forms** -- Links quadratic forms of normal random vectors to the chi-squared distribution, a result used extensively in hypothesis testing and confidence intervals.
- **Sampling Distributions (Simple OLS)** -- Derives the exact sampling distributions of the slope and intercept estimators in simple linear regression using projection matrices.
- **Sampling Distributions (General OLS)** -- Extends the sampling distribution results to the general multiple regression setting, establishing the distributional theory for the OLS estimator vector.

### 0.5 Exercises

Practice problems covering mathematical foundations, computational tools, matrix algebra, and the connections between linear algebra and statistical theory.

---

## Prerequisites

This is the foundational chapter of the book. It assumes:

- **High school algebra and basic calculus** -- Familiarity with functions, limits, derivatives, and integrals.
- **Introductory linear algebra** -- Basic exposure to vectors, matrices, and systems of linear equations.

---

## Key Takeaways

1. Precise mathematical language (logic, sets, functions) prevents ambiguity in later work with probability and statistical inference.
2. Sequences, limits, and asymptotic notation provide the deterministic framework that underpins the convergence theorems of statistics.
3. Linear algebra notation and the design matrix formulation are the language of multivariate statistics and regression.
4. Python, NumPy, pandas, and Matplotlib form the computational toolkit used for data analysis throughout the book.
5. Special matrix types (idempotent, symmetric, positive definite, projection) have direct statistical interpretations in regression and hypothesis testing.
6. Quadratic forms of normal vectors follow chi-squared distributions, connecting matrix algebra to the sampling distributions used in inference.
