# Exercises

These exercises cover the probability distributions and dependence concepts from Chapter 4, including the binomial, geometric, Poisson, uniform, exponential, and normal distributions, as well as joint distributions, marginal and conditional distributions, covariance, and correlation.

---

## Exercise 1: Binomial Distribution

A quality inspector examines 10 items from a production line where each item independently has a 15% probability of being defective.

**(a)** Let $X$ be the number of defective items. What distribution does $X$ follow? State its parameters.

**(b)** Compute $P(X = 2)$.

**(c)** Compute $P(X \ge 3)$.

**(d)** Find $E[X]$ and $\text{Var}(X)$.

??? success "Solution"

    **(a)** $X \sim \text{Binomial}(n = 10, p = 0.15)$, since we have 10 independent trials, each with a 15% probability of success (defective).

    **(b)** The binomial PMF gives:

    $$
    P(X = 2) = \binom{10}{2}(0.15)^2(0.85)^8
    $$

    $$
    = 45 \times 0.0225 \times 0.2725 = 45 \times 0.006131 \approx 0.2759
    $$

    **(c)** Using the complement:

    $$
    P(X \ge 3) = 1 - P(X \le 2) = 1 - [P(X=0) + P(X=1) + P(X=2)]
    $$

    $$
    P(X=0) = (0.85)^{10} \approx 0.1969
    $$

    $$
    P(X=1) = \binom{10}{1}(0.15)(0.85)^9 = 10 \times 0.15 \times 0.2316 \approx 0.3474
    $$

    $$
    P(X \ge 3) = 1 - (0.1969 + 0.3474 + 0.2759) = 1 - 0.8202 = 0.1798
    $$

    **(d)**

    $$
    E[X] = np = 10 \times 0.15 = 1.5
    $$

    $$
    \text{Var}(X) = np(1-p) = 10 \times 0.15 \times 0.85 = 1.275
    $$

---

## Exercise 2: Geometric Distribution

A salesperson makes cold calls. Each call independently results in a sale with probability $p = 0.1$.

**(a)** Let $Y$ be the number of calls until the first sale (including the successful call). What distribution does $Y$ follow?

**(b)** Compute $P(Y = 5)$ and $P(Y > 10)$.

**(c)** The salesperson has already made 8 calls without a sale. What is the probability the first sale occurs after 15 total calls? Use the memoryless property.

**(d)** Find $E[Y]$ and $\text{Var}(Y)$.

??? success "Solution"

    **(a)** $Y \sim \text{Geometric}(p = 0.1)$, where $Y$ counts the number of trials until the first success.

    **(b)** The geometric PMF is $P(Y = k) = (1-p)^{k-1} p$ for $k = 1, 2, 3, \ldots$

    $$
    P(Y = 5) = (0.9)^4(0.1) = 0.6561 \times 0.1 = 0.0656
    $$

    $$
    P(Y > 10) = (1 - p)^{10} = (0.9)^{10} \approx 0.3487
    $$

    **(c)** By the memoryless property of the geometric distribution:

    $$
    P(Y > 15 \mid Y > 8) = P(Y > 7) = (0.9)^7 \approx 0.4783
    $$

    The fact that 8 calls have already failed provides no information about future calls. The conditional distribution of remaining calls is the same as if the salesperson were starting fresh.

    **(d)**

    $$
    E[Y] = \frac{1}{p} = \frac{1}{0.1} = 10
    $$

    $$
    \text{Var}(Y) = \frac{1-p}{p^2} = \frac{0.9}{0.01} = 90
    $$

---

## Exercise 3: Poisson Distribution

A call center receives an average of 4 calls per minute during peak hours.

**(a)** Let $N$ be the number of calls in a 1-minute interval. What distribution does $N$ follow and what is its parameter?

**(b)** Compute $P(N = 0)$ and $P(N \ge 6)$.

**(c)** What is the probability of receiving more than 10 calls in a 2-minute interval?

**(d)** Approximate the Poisson probability $P(N \ge 8)$ using a normal approximation. Compare with the exact Poisson probability.

??? success "Solution"

    **(a)** $N \sim \text{Poisson}(\lambda = 4)$, since calls arrive at a constant average rate of 4 per minute and are assumed to occur independently.

    **(b)** The Poisson PMF is $P(N = k) = \frac{e^{-\lambda}\lambda^k}{k!}$:

    $$
    P(N = 0) = e^{-4} \approx 0.0183
    $$

    $$
    P(N \ge 6) = 1 - \sum_{k=0}^{5} \frac{e^{-4} \cdot 4^k}{k!}
    $$

    Computing the partial sum:

    | $k$ | $4^k/k!$ | $e^{-4} \cdot 4^k/k!$ |
    |:---:|:---:|:---:|
    | 0 | 1 | 0.0183 |
    | 1 | 4 | 0.0733 |
    | 2 | 8 | 0.1465 |
    | 3 | 10.667 | 0.1954 |
    | 4 | 10.667 | 0.1954 |
    | 5 | 8.533 | 0.1563 |

    $$
    P(N \le 5) \approx 0.7852, \quad P(N \ge 6) \approx 0.2148
    $$

    **(c)** In a 2-minute interval, the number of calls follows $N_2 \sim \text{Poisson}(\lambda = 8)$ (by the additive property of Poisson processes). We need:

    $$
    P(N_2 > 10) = 1 - P(N_2 \le 10) = 1 - \sum_{k=0}^{10} \frac{e^{-8} \cdot 8^k}{k!} \approx 1 - 0.8159 = 0.1841
    $$

    **(d)** For the normal approximation, $N \sim \text{Poisson}(4)$ has $\mu = 4$ and $\sigma = 2$. Using a continuity correction:

    $$
    P(N \ge 8) \approx P\!\left(Z \ge \frac{7.5 - 4}{2}\right) = P(Z \ge 1.75) = 1 - \Phi(1.75) \approx 0.0401
    $$

    The exact Poisson probability is:

    $$
    P(N \ge 8) = 1 - P(N \le 7) \approx 1 - 0.9489 = 0.0511
    $$

    The normal approximation gives 0.040, while the exact value is 0.051. The approximation is reasonable but somewhat off because $\lambda = 4$ is not very large and the Poisson distribution at this parameter is noticeably right-skewed.

---

## Exercise 4: Exponential Distribution

The lifetime (in years) of a certain electronic component follows an exponential distribution with rate $\lambda = 0.5$ (mean lifetime of 2 years).

**(a)** Compute the probability that a component lasts more than 3 years.

**(b)** Given that a component has already lasted 2 years, what is the probability it lasts at least 1 more year? Verify using the memoryless property.

**(c)** A system requires both of two independent components to function. If each has lifetime $T_i \sim \text{Exp}(0.5)$, find the distribution and expected value of $T_{\min} = \min(T_1, T_2)$, the system's lifetime.

??? success "Solution"

    **(a)** The survival function of the exponential is $P(T > t) = e^{-\lambda t}$:

    $$
    P(T > 3) = e^{-0.5 \times 3} = e^{-1.5} \approx 0.2231
    $$

    **(b)** By the memoryless property, $P(T > s + t \mid T > s) = P(T > t)$ for all $s, t \ge 0$:

    $$
    P(T > 3 \mid T > 2) = P(T > 1) = e^{-0.5} \approx 0.6065
    $$

    Direct verification:

    $$
    P(T > 3 \mid T > 2) = \frac{P(T > 3)}{P(T > 2)} = \frac{e^{-1.5}}{e^{-1}} = e^{-0.5} \approx 0.6065
    $$

    Both methods agree, confirming the memoryless property.

    **(c)** For independent exponentials, the minimum is also exponential with rate equal to the sum of rates:

    $$
    T_{\min} = \min(T_1, T_2) \sim \text{Exp}(\lambda_1 + \lambda_2) = \text{Exp}(1.0)
    $$

    To derive this, note that:

    $$
    P(T_{\min} > t) = P(T_1 > t) \cdot P(T_2 > t) = e^{-0.5t} \cdot e^{-0.5t} = e^{-t}
    $$

    This is the survival function of $\text{Exp}(1)$. The expected system lifetime is:

    $$
    E[T_{\min}] = \frac{1}{\lambda_1 + \lambda_2} = \frac{1}{1} = 1 \text{ year}
    $$

    The system's expected lifetime is half that of each individual component.

---

## Exercise 5: Normal Distribution

Let $X \sim N(70, 100)$ represent exam scores (mean 70, variance 100, standard deviation 10).

**(a)** Compute $P(60 < X < 80)$.

**(b)** Find the 90th percentile of the score distribution.

**(c)** If a class has 200 students, approximately how many score above 85?

**(d)** Let $Y = \frac{X - 70}{10}$. What is the distribution of $Y$? Show that $P(X > 85) = P(Y > 1.5)$.

??? success "Solution"

    **(a)** Standardize: $Z = (X - 70)/10$.

    $$
    P(60 < X < 80) = P\!\left(\frac{60 - 70}{10} < Z < \frac{80 - 70}{10}\right) = P(-1 < Z < 1)
    $$

    $$
    = \Phi(1) - \Phi(-1) = 0.8413 - 0.1587 = 0.6827
    $$

    About 68.3% of scores fall within one standard deviation of the mean.

    **(b)** The 90th percentile corresponds to $z_{0.90} = 1.2816$:

    $$
    x_{0.90} = \mu + z_{0.90} \cdot \sigma = 70 + 1.2816 \times 10 = 82.82
    $$

    A score of approximately 82.8 marks the 90th percentile.

    **(c)**

    $$
    P(X > 85) = P\!\left(Z > \frac{85 - 70}{10}\right) = P(Z > 1.5) = 1 - \Phi(1.5) \approx 1 - 0.9332 = 0.0668
    $$

    Expected number of students: $200 \times 0.0668 \approx 13.4$, so approximately 13 students.

    **(d)** Since $Y = (X - 70)/10$ is a linear transformation of a normal random variable:

    $$
    Y \sim N\!\left(\frac{70 - 70}{10}, \frac{100}{100}\right) = N(0, 1)
    $$

    $Y$ is standard normal. The equivalence follows directly:

    $$
    P(X > 85) = P\!\left(\frac{X - 70}{10} > \frac{85 - 70}{10}\right) = P(Y > 1.5)
    $$

---

## Exercise 6: Joint Distribution (Discrete)

Two random variables $X$ and $Y$ have the following joint PMF:

| | $Y = 0$ | $Y = 1$ | $Y = 2$ |
|:---:|:---:|:---:|:---:|
| $X = 0$ | 0.10 | 0.15 | 0.05 |
| $X = 1$ | 0.10 | 0.20 | 0.10 |
| $X = 2$ | 0.05 | 0.10 | 0.15 |

**(a)** Verify that this is a valid joint PMF.

**(b)** Find the marginal PMFs $p_X(x)$ and $p_Y(y)$.

**(c)** Compute $E[X]$, $E[Y]$, and $E[XY]$.

**(d)** Compute $\text{Cov}(X, Y)$ and the correlation $\rho(X, Y)$.

**(e)** Are $X$ and $Y$ independent? Justify your answer.

??? success "Solution"

    **(a)** Sum of all entries: $0.10 + 0.15 + 0.05 + 0.10 + 0.20 + 0.10 + 0.05 + 0.10 + 0.15 = 1.00$. All entries are non-negative, so this is a valid joint PMF.

    **(b)** Marginal PMF of $X$ (row sums):

    $$
    p_X(0) = 0.30, \quad p_X(1) = 0.40, \quad p_X(2) = 0.30
    $$

    Marginal PMF of $Y$ (column sums):

    $$
    p_Y(0) = 0.25, \quad p_Y(1) = 0.45, \quad p_Y(2) = 0.30
    $$

    **(c)**

    $$
    E[X] = 0(0.30) + 1(0.40) + 2(0.30) = 1.0
    $$

    $$
    E[Y] = 0(0.25) + 1(0.45) + 2(0.30) = 1.05
    $$

    $$
    E[XY] = \sum_{x,y} xy \cdot p(x,y) = 0 + 0 + 0 + 0 + 1(1)(0.20) + 1(2)(0.10) + 0 + 2(1)(0.10) + 2(2)(0.15)
    $$

    $$
    = 0.20 + 0.20 + 0.20 + 0.60 = 1.20
    $$

    **(d)**

    $$
    \text{Cov}(X, Y) = E[XY] - E[X]E[Y] = 1.20 - (1.0)(1.05) = 0.15
    $$

    For the correlation, we need the variances:

    $$
    E[X^2] = 0(0.30) + 1(0.40) + 4(0.30) = 1.60, \quad \text{Var}(X) = 1.60 - 1.0 = 0.60
    $$

    $$
    E[Y^2] = 0(0.25) + 1(0.45) + 4(0.30) = 1.65, \quad \text{Var}(Y) = 1.65 - 1.05^2 = 1.65 - 1.1025 = 0.5475
    $$

    $$
    \rho(X, Y) = \frac{\text{Cov}(X,Y)}{\sqrt{\text{Var}(X)\text{Var}(Y)}} = \frac{0.15}{\sqrt{0.60 \times 0.5475}} = \frac{0.15}{\sqrt{0.3285}} = \frac{0.15}{0.5731} \approx 0.262
    $$

    **(e)** For independence, we need $p(x,y) = p_X(x) \cdot p_Y(y)$ for all $(x,y)$. Check one cell: $p(0,0) = 0.10$ but $p_X(0) \cdot p_Y(0) = 0.30 \times 0.25 = 0.075 \ne 0.10$. Since the factorization fails, $X$ and $Y$ are **not** independent. The positive correlation $\rho \approx 0.26$ also confirms dependence.

---

## Exercise 7: Marginal and Conditional Distributions (Continuous)

Let $(X, Y)$ have joint PDF:

$$
f(x, y) = \begin{cases} 6(1 - y) & 0 \le x \le y \le 1 \\ 0 & \text{otherwise} \end{cases}
$$

**(a)** Verify that $f$ integrates to 1.

**(b)** Find the marginal PDF $f_Y(y)$.

**(c)** Find the conditional PDF $f_{X \mid Y}(x \mid y)$ and identify the distribution of $X \mid Y = y$.

**(d)** Compute $E[X \mid Y = y]$.

??? success "Solution"

    **(a)** Integrate over the region $0 \le x \le y \le 1$:

    $$
    \int_0^1 \int_0^y 6(1-y)\,dx\,dy = \int_0^1 6(1-y) \cdot y\,dy = 6\int_0^1 (y - y^2)\,dy
    $$

    $$
    = 6\left[\frac{y^2}{2} - \frac{y^3}{3}\right]_0^1 = 6\left(\frac{1}{2} - \frac{1}{3}\right) = 6 \cdot \frac{1}{6} = 1
    $$

    **(b)** Integrate out $x$:

    $$
    f_Y(y) = \int_0^y 6(1-y)\,dx = 6y(1-y), \quad 0 \le y \le 1
    $$

    This is a $\text{Beta}(2, 2)$ distribution (up to normalization — indeed $B(2,2) = 1/6$ and $6y(1-y) = y^{2-1}(1-y)^{2-1}/B(2,2)$).

    **(c)** The conditional PDF is:

    $$
    f_{X|Y}(x \mid y) = \frac{f(x,y)}{f_Y(y)} = \frac{6(1-y)}{6y(1-y)} = \frac{1}{y}, \quad 0 \le x \le y
    $$

    This is the $\text{Uniform}(0, y)$ distribution. Given $Y = y$, $X$ is uniformly distributed on $[0, y]$.

    **(d)** Since $X \mid Y = y \sim \text{Uniform}(0, y)$:

    $$
    E[X \mid Y = y] = \frac{0 + y}{2} = \frac{y}{2}
    $$

    The conditional expectation of $X$ is half the value of $Y$, which makes geometric sense: given $Y = y$, $X$ is equally likely to be anywhere from 0 to $y$.

---

## Exercise 8: Covariance and Correlation

Let $X$ be a continuous random variable with $E[X] = 0$, $E[X^2] = 1$, and $E[X^3] = 0$ (symmetric distribution with unit variance). Define $Y = X^2$.

**(a)** Compute $\text{Cov}(X, Y)$.

**(b)** What is $\rho(X, Y)$?

**(c)** Are $X$ and $Y$ independent? Explain why this example demonstrates that zero correlation does not imply independence.

??? success "Solution"

    **(a)** Using $\text{Cov}(X, Y) = E[XY] - E[X]E[Y]$:

    $$
    E[XY] = E[X \cdot X^2] = E[X^3] = 0
    $$

    $$
    E[X] = 0
    $$

    $$
    \text{Cov}(X, Y) = 0 - 0 \cdot E[Y] = 0
    $$

    **(b)** Since $\text{Cov}(X, Y) = 0$:

    $$
    \rho(X, Y) = 0
    $$

    **(c)** $X$ and $Y$ are **not** independent. Knowing $X$ completely determines $Y = X^2$: if $X = 3$, then $Y = 9$ with certainty. Independence would require $P(Y \le y \mid X = x) = P(Y \le y)$ for all $x, y$, which clearly fails.

    This is the canonical counterexample showing that zero correlation does not imply independence. Correlation measures only *linear* association. Here, $X$ and $Y$ have a perfect *nonlinear* (quadratic) relationship that produces zero covariance because the positive and negative deviations cancel perfectly due to the symmetry of $X$.

---

## Exercise 9: Poisson Approximation to the Binomial

A book has 500 pages. Suppose each page independently contains a misprint with probability $p = 0.004$.

**(a)** Let $X$ be the total number of misprints. What is the exact distribution of $X$?

**(b)** Argue why a Poisson approximation is appropriate here. What is the Poisson parameter?

**(c)** Use the Poisson approximation to compute $P(X = 0)$, $P(X = 1)$, and $P(X \ge 4)$.

**(d)** Compare $P(X = 2)$ under the exact binomial and the Poisson approximation.

??? success "Solution"

    **(a)** $X \sim \text{Binomial}(n = 500, p = 0.004)$, since each of 500 pages independently has a misprint with probability 0.004.

    **(b)** The Poisson approximation is appropriate because $n$ is large (500) and $p$ is small (0.004), so the product $\lambda = np = 500 \times 0.004 = 2$ is moderate. Under these conditions, $\text{Binomial}(n, p) \approx \text{Poisson}(\lambda = np)$.

    **(c)** With $\lambda = 2$:

    $$
    P(X = 0) \approx e^{-2} \approx 0.1353
    $$

    $$
    P(X = 1) \approx 2e^{-2} \approx 0.2707
    $$

    $$
    P(X \ge 4) = 1 - P(X \le 3) = 1 - e^{-2}\left(1 + 2 + 2 + \frac{4}{3}\right) = 1 - e^{-2} \cdot \frac{19}{3}
    $$

    $$
    \approx 1 - 0.1353 \times 6.333 = 1 - 0.8571 = 0.1429
    $$

    **(d)** Exact binomial:

    $$
    P(X = 2) = \binom{500}{2}(0.004)^2(0.996)^{498}
    $$

    $$
    = 124750 \times 0.000016 \times (0.996)^{498}
    $$

    Now $(0.996)^{498} = e^{498 \ln(0.996)} \approx e^{498 \times (-0.004008)} = e^{-1.996} \approx 0.13614$:

    $$
    P_{\text{Binom}}(X = 2) \approx 124750 \times 0.000016 \times 0.13614 \approx 0.2718
    $$

    Poisson approximation:

    $$
    P_{\text{Poisson}}(X = 2) = \frac{e^{-2} \cdot 4}{2} = 2e^{-2} \approx 0.2707
    $$

    The difference is about 0.001 — the Poisson approximation is excellent when $n$ is large and $p$ is small.

---

## Exercise 10: Uniform Distribution and Simulation

Let $U \sim \text{Uniform}(0, 1)$ and define $X = -\frac{1}{\lambda}\ln(1 - U)$ for $\lambda > 0$.

**(a)** Find the CDF of $X$ by computing $P(X \le x)$ for $x \ge 0$.

**(b)** Identify the distribution of $X$.

**(c)** This result is the foundation of the **inverse transform method** for simulation. Explain how you would use it to generate exponential random variables from uniform random numbers.

**(d)** If $U \sim \text{Uniform}(0,1)$, show that $1 - U \sim \text{Uniform}(0,1)$ as well, so the transform $X = -\frac{1}{\lambda}\ln(U)$ works equally well.

??? success "Solution"

    **(a)** For $x \ge 0$:

    $$
    P(X \le x) = P\!\left(-\frac{1}{\lambda}\ln(1 - U) \le x\right)
    $$

    Since $-1/\lambda < 0$, dividing by it reverses the inequality:

    $$
    = P(\ln(1 - U) \ge -\lambda x) = P(1 - U \ge e^{-\lambda x}) = P(U \le 1 - e^{-\lambda x})
    $$

    Since $U \sim \text{Uniform}(0,1)$, $P(U \le u) = u$ for $u \in [0,1]$:

    $$
    P(X \le x) = 1 - e^{-\lambda x}
    $$

    **(b)** The CDF $F(x) = 1 - e^{-\lambda x}$ for $x \ge 0$ is the CDF of the $\text{Exponential}(\lambda)$ distribution. Therefore $X \sim \text{Exp}(\lambda)$.

    **(c)** The inverse transform method works as follows:

    1. Generate $u$ from $\text{Uniform}(0,1)$ (available in any programming language).
    2. Compute $x = -\frac{1}{\lambda}\ln(1 - u)$.
    3. The resulting $x$ is a sample from $\text{Exp}(\lambda)$.

    More generally, for any distribution with invertible CDF $F$, setting $X = F^{-1}(U)$ produces a random variable with CDF $F$.

    **(d)** If $U \sim \text{Uniform}(0,1)$, then for any $t \in [0,1]$:

    $$
    P(1 - U \le t) = P(U \ge 1 - t) = 1 - (1 - t) = t
    $$

    This is the CDF of $\text{Uniform}(0,1)$, so $1 - U \sim \text{Uniform}(0,1)$. Therefore we can replace $1 - U$ with $U$ in the transform, yielding the computationally simpler formula $X = -\frac{1}{\lambda}\ln(U)$.
