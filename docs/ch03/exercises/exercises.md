# Exercises

These exercises cover the foundations of probability from Chapter 3, including sample spaces, probability axioms, conditional probability, Bayes' theorem, independence, random variables, expectation, variance, moment generating functions, and limit theorems.

---

## Exercise 1: Sample Spaces and Events

A fair six-sided die is rolled twice. Let $A$ be the event that the sum is 7, and let $B$ be the event that the first roll is a 3.

**(a)** Write out the sample space $\Omega$ for this experiment. How many outcomes does it contain?

**(b)** List the outcomes in $A$, $B$, and $A \cap B$.

**(c)** Compute $P(A)$, $P(B)$, and $P(A \cap B)$.

**(d)** Are $A$ and $B$ independent? Justify your answer.

??? success "Solution"

    **(a)** The sample space consists of all ordered pairs $(i, j)$ where $i, j \in \{1, 2, 3, 4, 5, 6\}$:

    $$
    \Omega = \{(i,j) : i,j \in \{1,2,3,4,5,6\}\}
    $$

    There are $6 \times 6 = 36$ equally likely outcomes.

    **(b)**

    - $A = \{(1,6), (2,5), (3,4), (4,3), (5,2), (6,1)\}$ — the 6 outcomes summing to 7.
    - $B = \{(3,1), (3,2), (3,3), (3,4), (3,5), (3,6)\}$ — the 6 outcomes with first roll 3.
    - $A \cap B = \{(3,4)\}$ — the unique outcome in both events.

    **(c)** Since all 36 outcomes are equally likely:

    $$
    P(A) = \frac{6}{36} = \frac{1}{6}, \quad P(B) = \frac{6}{36} = \frac{1}{6}, \quad P(A \cap B) = \frac{1}{36}
    $$

    **(d)** Events $A$ and $B$ are independent if and only if $P(A \cap B) = P(A) \cdot P(B)$. We check:

    $$
    P(A) \cdot P(B) = \frac{1}{6} \cdot \frac{1}{6} = \frac{1}{36} = P(A \cap B)
    $$

    Since equality holds, $A$ and $B$ are independent. Intuitively, knowing the first roll is 3 does not change the probability of the sum being 7 — exactly one of the six second-roll values (namely 4) produces a sum of 7, giving a conditional probability of $1/6$.

---

## Exercise 2: Conditional Probability and Bayes' Theorem

A medical test for a rare disease has the following characteristics:

- Prevalence: $P(\text{Disease}) = 0.001$ (1 in 1000 people are affected)
- Sensitivity: $P(\text{Positive} \mid \text{Disease}) = 0.99$
- Specificity: $P(\text{Negative} \mid \text{No Disease}) = 0.95$

**(a)** If a randomly selected person tests positive, what is the probability they actually have the disease? Compute $P(\text{Disease} \mid \text{Positive})$ using Bayes' theorem.

**(b)** Explain intuitively why the result in (a) is surprisingly low despite the test's high sensitivity and specificity.

**(c)** If the prevalence increases to $P(\text{Disease}) = 0.05$, recompute $P(\text{Disease} \mid \text{Positive})$. How does the prior probability affect the posterior?

??? success "Solution"

    **(a)** By Bayes' theorem:

    $$
    P(\text{Disease} \mid \text{Positive}) = \frac{P(\text{Positive} \mid \text{Disease}) \cdot P(\text{Disease})}{P(\text{Positive})}
    $$

    First compute the total probability of testing positive:

    $$
    P(\text{Positive}) = P(\text{Pos} \mid \text{Dis}) \cdot P(\text{Dis}) + P(\text{Pos} \mid \text{No Dis}) \cdot P(\text{No Dis})
    $$

    $$
    = (0.99)(0.001) + (0.05)(0.999) = 0.00099 + 0.04995 = 0.05094
    $$

    Therefore:

    $$
    P(\text{Disease} \mid \text{Positive}) = \frac{0.00099}{0.05094} \approx 0.0194
    $$

    Only about 1.94% of those who test positive actually have the disease.

    **(b)** The low posterior probability arises because the disease is very rare ($P(\text{Disease}) = 0.001$). Even with a 5% false positive rate, the number of false positives from the 999 healthy people ($\approx 50$) vastly outnumbers the true positives from the 1 sick person ($\approx 1$). The base rate of the disease is so low that false positives dominate the pool of positive results.

    **(c)** With $P(\text{Disease}) = 0.05$:

    $$
    P(\text{Positive}) = (0.99)(0.05) + (0.05)(0.95) = 0.0495 + 0.0475 = 0.097
    $$

    $$
    P(\text{Disease} \mid \text{Positive}) = \frac{(0.99)(0.05)}{0.097} = \frac{0.0495}{0.097} \approx 0.510
    $$

    The posterior probability jumps from about 2% to 51%. A higher prior dramatically increases the posterior because the true positive count grows relative to the false positive count. This illustrates how strongly the prior (base rate) influences the posterior in Bayesian reasoning.

---

## Exercise 3: Law of Total Probability

A factory has three machines producing widgets. Machine 1 produces 50% of output, Machine 2 produces 30%, and Machine 3 produces 20%. Their defect rates are 2%, 3%, and 5%, respectively.

**(a)** What is the overall probability that a randomly selected widget is defective?

**(b)** Given that a widget is defective, what is the probability it came from Machine 3?

??? success "Solution"

    **(a)** Let $D$ denote the event "defective" and $M_i$ denote the event "produced by Machine $i$". By the law of total probability:

    $$
    P(D) = \sum_{i=1}^3 P(D \mid M_i) \cdot P(M_i)
    $$

    $$
    = (0.02)(0.50) + (0.03)(0.30) + (0.05)(0.20) = 0.010 + 0.009 + 0.010 = 0.029
    $$

    The overall defect rate is 2.9%.

    **(b)** By Bayes' theorem:

    $$
    P(M_3 \mid D) = \frac{P(D \mid M_3) \cdot P(M_3)}{P(D)} = \frac{(0.05)(0.20)}{0.029} = \frac{0.010}{0.029} \approx 0.345
    $$

    About 34.5% of defective widgets come from Machine 3. Although Machine 3 produces only 20% of the output, its high defect rate (5%) means it contributes a disproportionate share of defectives.

---

## Exercise 4: PMF and CDF of a Discrete Random Variable

Let $X$ be the number of heads in 3 independent flips of a fair coin.

**(a)** Write the probability mass function $p(x) = P(X = x)$ for all possible values of $X$.

**(b)** Write the cumulative distribution function $F(x) = P(X \le x)$ as a piecewise function.

**(c)** Compute $P(1 \le X \le 2)$ using both the PMF and the CDF.

??? success "Solution"

    **(a)** $X \sim \text{Binomial}(3, 1/2)$. The sample space for 3 coin flips has $2^3 = 8$ equally likely outcomes:

    | $x$ | Outcomes | $p(x)$ |
    |:---:|:---:|:---:|
    | 0 | TTT | $1/8$ |
    | 1 | HTT, THT, TTH | $3/8$ |
    | 2 | HHT, HTH, THH | $3/8$ |
    | 3 | HHH | $1/8$ |

    Equivalently, $p(x) = \binom{3}{x} (1/2)^3$ for $x \in \{0, 1, 2, 3\}$.

    **(b)** The CDF accumulates the PMF values:

    $$
    F(x) = \begin{cases} 0 & x < 0 \\ 1/8 & 0 \le x < 1 \\ 4/8 & 1 \le x < 2 \\ 7/8 & 2 \le x < 3 \\ 1 & x \ge 3 \end{cases}
    $$

    **(c)** Using the PMF:

    $$
    P(1 \le X \le 2) = p(1) + p(2) = \frac{3}{8} + \frac{3}{8} = \frac{6}{8} = \frac{3}{4}
    $$

    Using the CDF:

    $$
    P(1 \le X \le 2) = F(2) - F(0) = \frac{7}{8} - \frac{1}{8} = \frac{6}{8} = \frac{3}{4}
    $$

    Both methods yield $3/4$.

---

## Exercise 5: Continuous Random Variable — PDF and CDF

Let $X$ have the probability density function:

$$
f(x) = \begin{cases} cx^2 & 0 \le x \le 2 \\ 0 & \text{otherwise} \end{cases}
$$

**(a)** Find the constant $c$ that makes $f$ a valid PDF.

**(b)** Find the CDF $F(x)$.

**(c)** Compute $P(1 \le X \le 2)$.

**(d)** Find the median of $X$.

??? success "Solution"

    **(a)** For $f$ to be a valid PDF, it must integrate to 1:

    $$
    \int_{-\infty}^{\infty} f(x)\,dx = c \int_0^2 x^2\,dx = c \left[\frac{x^3}{3}\right]_0^2 = c \cdot \frac{8}{3} = 1
    $$

    Therefore $c = 3/8$.

    **(b)** For $0 \le x \le 2$:

    $$
    F(x) = \int_0^x \frac{3}{8}t^2\,dt = \frac{3}{8} \cdot \frac{x^3}{3} = \frac{x^3}{8}
    $$

    The full CDF is:

    $$
    F(x) = \begin{cases} 0 & x < 0 \\ x^3/8 & 0 \le x \le 2 \\ 1 & x > 2 \end{cases}
    $$

    **(c)**

    $$
    P(1 \le X \le 2) = F(2) - F(1) = 1 - \frac{1}{8} = \frac{7}{8}
    $$

    **(d)** The median $m$ satisfies $F(m) = 1/2$:

    $$
    \frac{m^3}{8} = \frac{1}{2} \implies m^3 = 4 \implies m = \sqrt[3]{4} \approx 1.587
    $$

---

## Exercise 6: Expectation and Variance

Let $X$ be a random variable with PMF:

| $x$ | 1 | 2 | 3 | 4 |
|:---:|:---:|:---:|:---:|:---:|
| $P(X=x)$ | $0.1$ | $0.3$ | $0.4$ | $0.2$ |

**(a)** Compute $E[X]$.

**(b)** Compute $E[X^2]$ and use it to find $\text{Var}(X)$ via the formula $\text{Var}(X) = E[X^2] - (E[X])^2$.

**(c)** Let $Y = 3X + 5$. Compute $E[Y]$ and $\text{Var}(Y)$ without computing the PMF of $Y$.

??? success "Solution"

    **(a)**

    $$
    E[X] = \sum_x x \cdot P(X = x) = 1(0.1) + 2(0.3) + 3(0.4) + 4(0.2)
    $$

    $$
    = 0.1 + 0.6 + 1.2 + 0.8 = 2.7
    $$

    **(b)**

    $$
    E[X^2] = \sum_x x^2 \cdot P(X = x) = 1(0.1) + 4(0.3) + 9(0.4) + 16(0.2)
    $$

    $$
    = 0.1 + 1.2 + 3.6 + 3.2 = 8.1
    $$

    $$
    \text{Var}(X) = E[X^2] - (E[X])^2 = 8.1 - (2.7)^2 = 8.1 - 7.29 = 0.81
    $$

    **(c)** By linearity of expectation:

    $$
    E[Y] = E[3X + 5] = 3E[X] + 5 = 3(2.7) + 5 = 13.1
    $$

    By the variance scaling property (constants shift and scale):

    $$
    \text{Var}(Y) = \text{Var}(3X + 5) = 3^2 \cdot \text{Var}(X) = 9(0.81) = 7.29
    $$

    Adding a constant does not affect variance, and multiplying by a constant $a$ scales variance by $a^2$.

---

## Exercise 7: Moment Generating Function

Let $X \sim \text{Exponential}(\lambda)$ with PDF $f(x) = \lambda e^{-\lambda x}$ for $x \ge 0$.

**(a)** Derive the moment generating function $M_X(t) = E[e^{tX}]$ and state for which values of $t$ it is defined.

**(b)** Use the MGF to find $E[X]$ and $E[X^2]$.

**(c)** Verify that $\text{Var}(X) = 1/\lambda^2$.

??? success "Solution"

    **(a)**

    $$
    M_X(t) = E[e^{tX}] = \int_0^{\infty} e^{tx} \lambda e^{-\lambda x}\,dx = \lambda \int_0^{\infty} e^{-({\lambda - t})x}\,dx
    $$

    This integral converges if and only if $\lambda - t > 0$, i.e., $t < \lambda$. In that case:

    $$
    M_X(t) = \lambda \cdot \frac{1}{\lambda - t} = \frac{\lambda}{\lambda - t}
    $$

    for $t < \lambda$.

    **(b)** The $n$-th moment is obtained by differentiating the MGF $n$ times and evaluating at $t = 0$.

    $$
    M_X'(t) = \frac{\lambda}{(\lambda - t)^2} \implies E[X] = M_X'(0) = \frac{\lambda}{\lambda^2} = \frac{1}{\lambda}
    $$

    $$
    M_X''(t) = \frac{2\lambda}{(\lambda - t)^3} \implies E[X^2] = M_X''(0) = \frac{2\lambda}{\lambda^3} = \frac{2}{\lambda^2}
    $$

    **(c)**

    $$
    \text{Var}(X) = E[X^2] - (E[X])^2 = \frac{2}{\lambda^2} - \frac{1}{\lambda^2} = \frac{1}{\lambda^2}
    $$

    This confirms the well-known result that the exponential distribution has both mean and standard deviation equal to $1/\lambda$.

---

## Exercise 8: Independence versus Conditional Independence

Let $C$ be the event "it is cloudy," $R$ the event "it rains," and $U$ the event "someone carries an umbrella." Suppose:

- $P(C) = 0.4$
- $P(R \mid C) = 0.6$, $P(R \mid C^c) = 0.1$
- $P(U \mid R) = 0.9$, $P(U \mid R^c) = 0.2$

**(a)** Are $C$ and $U$ independent? Compute $P(U)$ and $P(U \mid C)$ to check.

**(b)** Are $C$ and $U$ conditionally independent given $R$? Formally, does $P(C \cap U \mid R) = P(C \mid R) \cdot P(U \mid R)$?

??? success "Solution"

    **(a)** First compute $P(R)$ by the law of total probability:

    $$
    P(R) = P(R \mid C)P(C) + P(R \mid C^c)P(C^c) = (0.6)(0.4) + (0.1)(0.6) = 0.24 + 0.06 = 0.30
    $$

    Next compute $P(U)$:

    $$
    P(U) = P(U \mid R)P(R) + P(U \mid R^c)P(R^c) = (0.9)(0.30) + (0.2)(0.70) = 0.27 + 0.14 = 0.41
    $$

    Now compute $P(U \mid C)$ by conditioning on $R$:

    $$
    P(U \mid C) = P(U \mid R, C)P(R \mid C) + P(U \mid R^c, C)P(R^c \mid C)
    $$

    Assuming that umbrella-carrying depends only on rain (not directly on clouds), so $P(U \mid R, C) = P(U \mid R) = 0.9$ and $P(U \mid R^c, C) = P(U \mid R^c) = 0.2$:

    $$
    P(U \mid C) = (0.9)(0.6) + (0.2)(0.4) = 0.54 + 0.08 = 0.62
    $$

    Since $P(U \mid C) = 0.62 \ne 0.41 = P(U)$, events $C$ and $U$ are **not** independent. Knowing it is cloudy increases the probability of carrying an umbrella.

    **(b)** Under the assumption that umbrella-carrying depends only on rain (i.e., $P(U \mid R, C) = P(U \mid R)$), we have:

    $$
    P(U \mid R, C) = P(U \mid R) = 0.9
    $$

    This means $C$ provides no additional information about $U$ once we know $R$. Formally:

    $$
    P(C \cap U \mid R) = P(U \mid R, C) \cdot P(C \mid R) = P(U \mid R) \cdot P(C \mid R)
    $$

    So $C$ and $U$ are conditionally independent given $R$. This illustrates a key concept: $C$ and $U$ are marginally dependent (clouds predict umbrellas) but conditionally independent given $R$ (once we know whether it rains, clouds add no further information about umbrellas). Rain "screens off" the association between clouds and umbrellas.

---

## Exercise 9: Law of Large Numbers Simulation

Let $X_1, X_2, \ldots$ be i.i.d. random variables with $X_i \sim \text{Uniform}(0, 1)$.

**(a)** State the expected value and variance of $X_i$.

**(b)** Write the sample mean $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$. What does the Weak Law of Large Numbers say about $\bar{X}_n$ as $n \to \infty$?

**(c)** Using Chebyshev's inequality, find an upper bound on $P(|\bar{X}_n - 1/2| \ge 0.05)$ when $n = 100$.

**(d)** For $n = 10{,}000$, how does the bound change? Explain why this bound, while valid, is typically very loose.

??? success "Solution"

    **(a)** For $X \sim \text{Uniform}(0,1)$:

    $$
    E[X] = \frac{1}{2}, \quad \text{Var}(X) = \frac{1}{12}
    $$

    **(b)** The Weak Law of Large Numbers (WLLN) states that for any $\epsilon > 0$:

    $$
    P\!\left(|\bar{X}_n - \mu| \ge \epsilon\right) \to 0 \quad \text{as } n \to \infty
    $$

    where $\mu = E[X_i] = 1/2$. The sample mean converges in probability to the population mean.

    **(c)** Chebyshev's inequality gives:

    $$
    P\!\left(|\bar{X}_n - \mu| \ge \epsilon\right) \le \frac{\text{Var}(\bar{X}_n)}{\epsilon^2}
    $$

    Since $\text{Var}(\bar{X}_n) = \text{Var}(X)/n = 1/(12n)$:

    $$
    P\!\left(|\bar{X}_{100} - 1/2| \ge 0.05\right) \le \frac{1/(12 \cdot 100)}{0.05^2} = \frac{1/1200}{0.0025} = \frac{1}{3} \approx 0.333
    $$

    **(d)** For $n = 10{,}000$:

    $$
    P\!\left(|\bar{X}_{10000} - 1/2| \ge 0.05\right) \le \frac{1/(12 \cdot 10000)}{0.0025} = \frac{1}{300} \approx 0.00333
    $$

    The bound decreases by a factor of 100 when $n$ increases by a factor of 100, confirming the $O(1/n)$ convergence rate. However, Chebyshev's inequality uses only the mean and variance — it makes no assumptions about the distribution shape. For the uniform distribution, the actual probability is much smaller than the bound suggests. The CLT-based approximation would give a far tighter estimate.

---

## Exercise 10: Central Limit Theorem Application

A machine fills bottles with a mean of $\mu = 500$ ml and a standard deviation of $\sigma = 10$ ml. The fill amounts are independent but **not** normally distributed (they follow a slightly right-skewed distribution).

**(a)** For a sample of $n = 36$ bottles, what does the Central Limit Theorem say about the distribution of the sample mean $\bar{X}_{36}$?

**(b)** Approximate $P(\bar{X}_{36} > 503)$.

**(c)** How large must $n$ be so that $P(|\bar{X}_n - 500| < 2) \ge 0.95$?

??? success "Solution"

    **(a)** By the Central Limit Theorem, for large $n$, the sample mean is approximately normal regardless of the underlying distribution:

    $$
    \bar{X}_n \stackrel{\text{approx}}{\sim} N\!\left(\mu, \frac{\sigma^2}{n}\right)
    $$

    For $n = 36$:

    $$
    \bar{X}_{36} \stackrel{\text{approx}}{\sim} N\!\left(500, \frac{100}{36}\right) = N(500, 2.778)
    $$

    The standard error is $\sigma/\sqrt{n} = 10/6 \approx 1.667$ ml.

    **(b)** Standardize:

    $$
    Z = \frac{\bar{X}_{36} - 500}{10/6} = \frac{503 - 500}{1.667} = 1.8
    $$

    $$
    P(\bar{X}_{36} > 503) = P(Z > 1.8) = 1 - \Phi(1.8) \approx 1 - 0.9641 = 0.0359
    $$

    There is approximately a 3.6% chance of observing a sample mean above 503 ml.

    **(c)** We need:

    $$
    P\!\left(|\bar{X}_n - 500| < 2\right) \ge 0.95
    $$

    Standardizing: $P\!\left(|Z| < \frac{2}{\sigma/\sqrt{n}}\right) \ge 0.95$, which requires:

    $$
    \frac{2}{\sigma/\sqrt{n}} \ge z_{0.025} = 1.96
    $$

    $$
    \frac{2\sqrt{n}}{10} \ge 1.96 \implies \sqrt{n} \ge 9.8 \implies n \ge 96.04
    $$

    Therefore $n \ge 97$ bottles are needed (rounding up to the next integer).
