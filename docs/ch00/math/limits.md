# Sequences, Limits, and Asymptotics

Sequences and their limiting behavior are the mathematical backbone of statistical inference. The Law of Large Numbers, the Central Limit Theorem, and the consistency of estimators are all statements about limits of sequences of random variables. This section reviews the deterministic foundations; the probabilistic extensions appear in Chapter 3.

## Sequences

### Definition

A **sequence** is an ordered list of real numbers indexed by the natural numbers:

$$
(a_n)_{n=1}^{\infty} = a_1,\, a_2,\, a_3,\, \dots
$$

Equivalently, a sequence is a function $a: \mathbb{N} \to \mathbb{R}$.

**Examples:**

- $a_n = 1/n$: the harmonic sequence $1, 1/2, 1/3, \dots$
- $a_n = (-1)^n / n$: an alternating sequence $-1, 1/2, -1/3, \dots$
- $a_n = (1 + 1/n)^n$: converges to $e \approx 2.718$

### Monotone and Bounded Sequences

- $(a_n)$ is **increasing** if $a_n \leq a_{n+1}$ for all $n$, and **decreasing** if $a_n \geq a_{n+1}$.
- $(a_n)$ is **bounded above** if there exists $M$ such that $a_n \leq M$ for all $n$; **bounded below** similarly.

!!! info "Monotone Convergence Theorem"
    Every bounded monotone sequence converges. This theorem is used implicitly whenever we assert that a non-decreasing sequence of probabilities or expectations has a limit.

## Limits of Sequences

### Definition ($\varepsilon$-$N$ Definition)

A sequence $(a_n)$ **converges** to a limit $L \in \mathbb{R}$, written $\lim_{n \to \infty} a_n = L$ or $a_n \to L$, if:

$$
\forall\, \varepsilon > 0,\;\; \exists\, N \in \mathbb{N} \text{ such that } n > N \implies |a_n - L| < \varepsilon.
$$

If no such $L$ exists, the sequence **diverges**.

### Limit Laws

If $a_n \to L$ and $b_n \to M$, then:

| Rule | Statement |
|---|---|
| Sum | $a_n + b_n \to L + M$ |
| Product | $a_n \cdot b_n \to L \cdot M$ |
| Quotient | $a_n / b_n \to L / M$ (provided $M \neq 0$) |
| Scalar multiple | $c \cdot a_n \to c \cdot L$ |
| Power | $a_n^k \to L^k$ for fixed $k \in \mathbb{N}$ |

### Squeeze Theorem

If $a_n \leq c_n \leq b_n$ for all $n$ and $a_n \to L$ and $b_n \to L$, then $c_n \to L$.

This technique is used to establish bounds on tail probabilities and approximation errors.

## Series

A **series** is the sequence of partial sums of a sequence:

$$
S_N = \sum_{n=1}^{N} a_n
$$

The series **converges** if $\lim_{N \to \infty} S_N$ exists and is finite.

### Key Series

| Series | Convergence | Limit |
|---|---|---|
| Geometric: $\sum_{n=0}^{\infty} r^n$ | $\lvert r \rvert < 1$ | $\dfrac{1}{1-r}$ |
| Harmonic: $\sum_{n=1}^{\infty} \frac{1}{n}$ | Diverges | — |
| $p$-series: $\sum_{n=1}^{\infty} \frac{1}{n^p}$ | $p > 1$ | Finite (depends on $p$) |
| Exponential: $\sum_{n=0}^{\infty} \frac{x^n}{n!}$ | All $x \in \mathbb{R}$ | $e^x$ |

The geometric series identity is used to derive the PMF normalizations for geometric and negative binomial distributions. The exponential series underpins the Poisson distribution and moment generating functions.

## Limits of Functions

### Definition

$$
\lim_{x \to a} f(x) = L \quad \Longleftrightarrow \quad \forall\, \varepsilon > 0,\; \exists\, \delta > 0 \text{ s.t. } 0 < |x - a| < \delta \implies |f(x) - L| < \varepsilon.
$$

### Continuity

A function $f$ is **continuous at** $a$ if $\lim_{x \to a} f(x) = f(a)$.

$f$ is **continuous on** an interval if it is continuous at every point in that interval.

Continuous functions preserve limits: if $a_n \to L$ and $f$ is continuous at $L$, then $f(a_n) \to f(L)$. This **continuous mapping theorem** has a probabilistic analog that is central to deriving the asymptotic distributions of estimators.

## Differentiation Essentials

### Derivative

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

### Key Rules

| Rule | Formula |
|---|---|
| Power | $(x^n)' = n x^{n-1}$ |
| Exponential | $(e^x)' = e^x$ |
| Logarithm | $(\ln x)' = 1/x$ |
| Chain | $(f \circ g)'(x) = f'(g(x)) \cdot g'(x)$ |
| Product | $(fg)' = f'g + fg'$ |
| Quotient | $(f/g)' = (f'g - fg')/g^2$ |

### Partial Derivatives

For $f: \mathbb{R}^n \to \mathbb{R}$, the **partial derivative** with respect to $x_i$ is

$$
\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \dots, x_i + h, \dots, x_n) - f(x_1, \dots, x_n)}{h}
$$

The **gradient** is the vector of all partial derivatives:

$$
\nabla f = \left(\frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_n}\right)
$$

Gradients are used extensively in maximum likelihood estimation and gradient-based optimization (Chapters 6, 7, 13, 14).

## Integration Essentials

### Definite Integral

$$
\int_a^b f(x)\, dx
$$

represents the signed area under $f$ from $a$ to $b$. For probability, this computes $P(a \leq X \leq b)$ when $f$ is a probability density function.

### Fundamental Theorem of Calculus

If $F'(x) = f(x)$, then

$$
\int_a^b f(x)\, dx = F(b) - F(a)
$$

### Key Integrals

| Integral | Result | Statistical Use |
|---|---|---|
| $\int_0^{\infty} e^{-\lambda x}\, dx$ | $1/\lambda$ | Exponential distribution |
| $\int_{-\infty}^{\infty} e^{-x^2/2}\, dx$ | $\sqrt{2\pi}$ | Normal distribution normalizing constant |
| $\int_0^{\infty} x^{n-1} e^{-x}\, dx$ | $\Gamma(n) = (n-1)!$ | Gamma function |

## Taylor Series and Approximations

### Taylor Expansion

The Taylor series of $f$ about $x = a$ is

$$
f(x) = \sum_{k=0}^{\infty} \frac{f^{(k)}(a)}{k!}(x - a)^k = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots
$$

### Important Expansions

$$
e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots
$$

$$
\ln(1 + x) = x - \frac{x^2}{2} + \frac{x^3}{3} - \cdots \qquad (|x| \leq 1,\; x \neq -1)
$$

$$
(1 + x)^n \approx 1 + nx + \frac{n(n-1)}{2}x^2 + \cdots
$$

Taylor approximations are the workhorse behind the delta method, the derivation of the Central Limit Theorem via moment generating functions, and asymptotic expansions of test statistics.

## Asymptotic Notation

Asymptotic notation describes the growth rate of functions and sequences, which is essential for characterizing how fast estimators converge.

### Big-$O$ and Little-$o$

| Notation | Definition | Intuition |
|---|---|---|
| $f(n) = O(g(n))$ | $\exists\, C, N$ s.t. $\lvert f(n) \rvert \leq C \lvert g(n) \rvert$ for $n > N$ | $f$ grows **no faster** than $g$ |
| $f(n) = o(g(n))$ | $\lim_{n \to \infty} f(n)/g(n) = 0$ | $f$ grows **strictly slower** than $g$ |
| $f(n) \sim g(n)$ | $\lim_{n \to \infty} f(n)/g(n) = 1$ | $f$ and $g$ are **asymptotically equivalent** |

**Examples:**

- $n^2 + 3n = O(n^2)$
- $1/n = o(1)$ — converges to zero.
- $n! \sim \sqrt{2\pi n}\,(n/e)^n$ — Stirling's approximation.

### Convergence Rates

In statistics, we often write:

$$
\hat{\theta}_n - \theta = O_p(n^{-1/2})
$$

This means the estimation error shrinks at rate $1/\sqrt{n}$, which is the standard rate for many estimators (e.g., the sample mean). The subscript $p$ indicates **convergence in probability**, formalized in Chapter 3.

## Modes of Convergence (Preview)

The following modes of convergence for sequences of random variables are developed fully in Chapter 3 but previewed here for context:

| Mode | Notation | Informal Meaning |
|---|---|---|
| Almost sure | $X_n \xrightarrow{\text{a.s.}} X$ | $X_n(\omega) \to X(\omega)$ for almost every outcome |
| In probability | $X_n \xrightarrow{p} X$ | $P(\lvert X_n - X \rvert > \varepsilon) \to 0$ |
| In distribution | $X_n \xrightarrow{d} X$ | CDFs converge: $F_{X_n}(x) \to F_X(x)$ |
| In $L^p$ | $X_n \xrightarrow{L^p} X$ | $E[\lvert X_n - X \rvert^p] \to 0$ |

**Hierarchy**: a.s. $\Rightarrow$ in probability $\Rightarrow$ in distribution. The Law of Large Numbers is a statement about convergence in probability (or a.s.), while the Central Limit Theorem is a statement about convergence in distribution.

## Summary

| Concept | Where It Appears |
|---|---|
| Limits of sequences | Consistency of estimators, LLN |
| Series convergence | PMF normalization, MGFs |
| Continuity and continuous mapping | Asymptotic distributions of transformed estimators |
| Derivatives and gradients | MLE, score functions, optimization |
| Integration | CDF/PDF relationship, expected values |
| Taylor series | Delta method, CLT derivation, asymptotic expansions |
| Big-$O$ / little-$o$ | Convergence rates of estimators |
