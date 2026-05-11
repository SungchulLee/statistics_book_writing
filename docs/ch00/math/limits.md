# Sequences, Limits, and Asymptotics

Sequences and their limiting behavior are the mathematical backbone of statistical inference. The Law of Large Numbers, Central Limit Theorem, consistency of estimators, and almost every asymptotic result in this book are statements about limits. Before lifting these ideas to random sequences, we must be fluent with deterministic sequences — the analytic skeleton that the probabilistic machinery hangs on.

## Definition

### Convergence of a sequence

A sequence $(a_n)_{n \ge 1}$ in $\mathbb{R}$ **converges** to $L \in \mathbb{R}$, written $a_n \to L$ or $\lim_{n \to \infty} a_n = L$, if

$$
\forall\, \varepsilon > 0,\;\; \exists\, N \in \mathbb{N} \text{ such that } n > N \implies |a_n - L| < \varepsilon
$$

A sequence is **Cauchy** if $\forall\, \varepsilon > 0,\, \exists\, N$ such that $m, n > N \implies |a_m - a_n| < \varepsilon$. In $\mathbb{R}$, Cauchy and convergent are equivalent (completeness).

### Convergence of a series

A series $\sum_{n=1}^\infty a_n$ converges to $S$ if the partial sums $S_N = \sum_{n=1}^N a_n$ converge to $S$. Tests: comparison, ratio, root, integral, alternating-series.

### Big-O and little-o

For sequences (and functions of $n$):

$$
f(n) = O(g(n)) \;\Longleftrightarrow\; \exists\, C > 0,\, N \text{ s.t. } |f(n)| \le C |g(n)| \text{ for } n > N
$$

$$
f(n) = o(g(n)) \;\Longleftrightarrow\; \lim_{n \to \infty} \frac{f(n)}{g(n)} = 0
$$

$f(n) \sim g(n)$ means $f(n)/g(n) \to 1$ (asymptotic equivalence). The probabilistic counterparts $O_p$ and $o_p$ appear in Chapter 3.

## Explanation

### Limit laws

If $a_n \to L$ and $b_n \to M$, then $a_n + b_n \to L + M$, $a_n b_n \to LM$, and $a_n / b_n \to L/M$ when $M \ne 0$. The squeeze theorem: if $a_n \le b_n \le c_n$ eventually and $a_n, c_n \to L$, then $b_n \to L$. Continuous functions preserve limits: $a_n \to L \Rightarrow g(a_n) \to g(L)$ for $g$ continuous at $L$.

### Series used throughout the book

| Series | Sum | Where it appears |
|---|---|---|
| $\sum_{n=0}^\infty r^n = \dfrac{1}{1-r}$, $|r|<1$ | Geometric | Geometric / negative binomial PMFs |
| $\sum_{n=0}^\infty \dfrac{x^n}{n!} = e^x$ | Exponential | Poisson PMF, MGFs |
| $-\sum_{n=1}^\infty \dfrac{(-1)^n x^n}{n} = \ln(1+x)$, $|x|<1$ | Logarithmic | Log-likelihood expansions |
| $\sum_{n=1}^\infty \dfrac{1}{n^s}$ converges iff $s > 1$ | $p$-series | Tail bound diagnostics |

A separate but constantly used integral is Gauss's $\int_{-\infty}^\infty e^{-x^2/2}\,dx = \sqrt{2\pi}$, which normalizes the standard normal density.

### Taylor expansion

For $f$ sufficiently smooth at $a$,

$$
f(x) = f(a) + f'(a)(x - a) + \tfrac{1}{2} f''(a)(x - a)^2 + \cdots + \tfrac{1}{k!} f^{(k)}(a)(x - a)^k + R_k(x)
$$

with remainder $R_k(x) = o((x - a)^k)$ as $x \to a$. Two consequences are used over and over:

- **Delta method**: if $\sqrt{n}(\hat{\theta}_n - \theta) \xrightarrow{d} N(0, \sigma^2)$ and $g$ is differentiable at $\theta$, then $\sqrt{n}(g(\hat{\theta}_n) - g(\theta)) \xrightarrow{d} N(0, [g'(\theta)]^2 \sigma^2)$.
- **CLT derivations** via MGFs: expand $M_X(t/\sqrt{n})$ around $0$ to second order; the surviving term is the variance contribution.

### Asymptotic notation in statistics

Statements like "$\hat{\theta}_n - \theta = O_p(n^{-1/2})$" mean the estimation error shrinks at the standard $\sqrt{n}$ rate — the typical rate of consistent regular estimators. Faster rates ($n^{-1}$) appear in superefficiency or boundary problems; slower rates ($n^{-1/4}$, $\log n$) appear in nonparametric estimation. Tracking the rate is how we compare estimators when both are consistent.

### Modes of convergence (a preview)

Random sequences can converge in several non-equivalent senses, developed in Chapter 3:

1. **Almost surely**: $P(\lim_n X_n = X) = 1$.
2. **In probability**: $\forall \varepsilon, P(|X_n - X| > \varepsilon) \to 0$.
3. **In distribution**: $F_{X_n}(x) \to F_X(x)$ at continuity points of $F_X$.
4. **In $L^p$**: $\mathbb{E}|X_n - X|^p \to 0$.

The hierarchy: (1) $\Rightarrow$ (2) $\Rightarrow$ (3), and (4) $\Rightarrow$ (2). None of the reverse implications hold in general.

## Examples

```python
import math
import numpy as np

# === (1 + 1/n)^n → e ===
ns = [10, 100, 1_000, 10_000, 100_000]
for n in ns:
    approx = (1 + 1/n)**n
    print(f"n={n:>7d}: (1+1/n)^n = {approx:.8f}, error = {abs(approx - math.e):.2e}")

# === Geometric series partial sums ===
r = 0.5
partial = np.cumsum(r ** np.arange(20))
exact = 1 / (1 - r)
print(f"\nGeometric r=0.5: S_19 = {partial[-1]:.10f}, exact = {exact}")

# === Taylor approximation of e^x at x = 0.3 ===
x = 0.3
for k in range(1, 7):
    taylor = sum(x**n / math.factorial(n) for n in range(k + 1))
    print(f"order {k}: {taylor:.8f}, exact: {math.exp(x):.8f}")

# === Demonstrating o(1/n) vs O(1/n) ===
n = np.arange(1, 50)
print("\nlog(n) / n   (o(1)? yes):", (np.log(n) / n)[-1])
print("sin(n) / n^2 (O(1/n^2)):", (np.sin(n) / n**2)[-1])
```

## Exercises

**Exercise 1.**
Prove directly from the $\varepsilon$–$N$ definition that

$$
\lim_{n \to \infty} \frac{3n + 1}{n + 2} = 3
$$

??? success "Solution to Exercise 1"
    Compute

    $$
    \left| \frac{3n+1}{n+2} - 3 \right| = \left| \frac{3n+1 - 3(n+2)}{n+2} \right| = \frac{5}{n+2}
    $$

    Given $\varepsilon > 0$, choose $N = \lceil 5/\varepsilon - 2 \rceil$. Then for all $n > N$,

    $$
    \frac{5}{n+2} < \frac{5}{N+2} \le \varepsilon
    $$

    Hence $|a_n - 3| < \varepsilon$. $\square$

---

**Exercise 2.**
**(a)** Show $\sum_{k=0}^\infty r^k$ converges iff $|r| < 1$, and find its sum.
**(b)** Use part (a) to evaluate $\displaystyle\sum_{k=1}^\infty \frac{3}{4^k}$.

??? success "Solution to Exercise 2"
    (a) The partial sums are

    $$
    S_n = \sum_{k=0}^n r^k = \frac{1 - r^{n+1}}{1 - r} \qquad (r \ne 1)
    $$

    If $|r| < 1$, $r^{n+1} \to 0$, so $S_n \to 1/(1 - r)$. If $|r| \ge 1$, $|r^k|$ does not tend to zero, violating the divergence test, so the series diverges.

    (b)

    $$
    \sum_{k=1}^\infty \frac{3}{4^k} = 3 \sum_{k=1}^\infty \left(\tfrac{1}{4}\right)^{\!k} = 3 \cdot \frac{1/4}{1 - 1/4} = 3 \cdot \tfrac{1}{3} = 1
    $$

---

**Exercise 3.**
Show that $(1 + x/n)^n \to e^x$ as $n \to \infty$ for every fixed $x \in \mathbb{R}$.

??? success "Solution to Exercise 3"
    Take logarithms. For fixed $x$ and $n$ large enough that $x/n$ lies in the domain of $\ln(1 + \cdot)$,

    $$
    n \ln\!\left(1 + \frac{x}{n}\right) = n \left[\frac{x}{n} - \frac{1}{2}\!\left(\frac{x}{n}\right)^{\!2} + O\!\left(\tfrac{1}{n^3}\right) \right] = x - \frac{x^2}{2n} + O\!\left(\tfrac{1}{n^2}\right)
    $$

    using the Taylor expansion of $\ln(1 + u)$ around $u = 0$. The right side tends to $x$, and since $\exp$ is continuous,

    $$
    (1 + x/n)^n = \exp\!\left(n \ln(1 + x/n)\right) \to e^x
    $$

    $\square$

---

**Exercise 4.**
Prove the squeeze theorem: if $a_n \le b_n \le c_n$ for all $n$ sufficiently large and $a_n, c_n \to L$, then $b_n \to L$.

??? success "Solution to Exercise 4"
    Let $\varepsilon > 0$. Choose $N_1$ such that $n > N_1 \Rightarrow |a_n - L| < \varepsilon$, and $N_2$ such that $n > N_2 \Rightarrow |c_n - L| < \varepsilon$. Let $N_3$ be a bound past which $a_n \le b_n \le c_n$ holds, and set $N = \max(N_1, N_2, N_3)$.

    For $n > N$:

    $$
    L - \varepsilon < a_n \le b_n \le c_n < L + \varepsilon
    $$

    so $|b_n - L| < \varepsilon$. $\square$

---

**Exercise 5.**
Use a Taylor expansion to show that for $X \sim \mathrm{Bernoulli}(p)$ with sample mean $\bar{X}_n$, the variance-stabilizing transform $g(p) = 2\arcsin(\sqrt{p})$ satisfies

$$
\sqrt{n}\!\left(g(\bar{X}_n) - g(p)\right) \xrightarrow{d} N(0, 1)
$$

so that $g(\bar{X}_n)$ has approximately constant variance $1/n$ regardless of $p$.

??? success "Solution to Exercise 5"
    By the CLT, $\sqrt{n}(\bar{X}_n - p) \xrightarrow{d} N(0, p(1 - p))$. The delta method gives

    $$
    \sqrt{n}\!\left(g(\bar{X}_n) - g(p)\right) \xrightarrow{d} N\!\left(0, [g'(p)]^2 \, p(1 - p)\right)
    $$

    Differentiating $g(p) = 2 \arcsin(\sqrt{p})$:

    $$
    g'(p) = 2 \cdot \frac{1}{\sqrt{1 - p}} \cdot \frac{1}{2\sqrt{p}} = \frac{1}{\sqrt{p(1-p)}}
    $$

    Therefore $[g'(p)]^2 \cdot p(1-p) = 1$ for every $p \in (0, 1)$, so the limit is $N(0, 1)$. $\square$

---

**Exercise 6.**
Show that big-O is not symmetric: give sequences with $f(n) = O(g(n))$ but $g(n) \ne O(f(n))$. Then state the natural equivalence relation defined by "$f \asymp g$" meaning $f = O(g)$ **and** $g = O(f)$, and give an example of two sequences with $f \asymp g$ but $f \not\sim g$.

??? success "Solution to Exercise 6"
    **Asymmetric example:** $f(n) = 1$, $g(n) = n$. Then $f(n) = O(g(n))$ (take $C = 1$), but $g(n)/f(n) = n \to \infty$, so $g(n) \ne O(f(n))$.

    **Equivalence relation:** $f \asymp g$ if and only if there exist $0 < c_1 \le c_2 < \infty$ and $N$ such that $c_1 |g(n)| \le |f(n)| \le c_2 |g(n)|$ for $n > N$. This is reflexive, symmetric (by definition), and transitive. It captures "same order of growth."

    **$f \asymp g$ but $f \not\sim g$:** take $f(n) = n$ and $g(n) = 2n$. Then $c_1 = 1/2$, $c_2 = 2$ work, so $f \asymp g$. But $f(n)/g(n) = 1/2 \ne 1$, so they are not asymptotically equivalent. $\square$
