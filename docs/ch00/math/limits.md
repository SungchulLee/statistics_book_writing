# Sequences, Limits, and Asymptotics

Sequences and their limiting behavior are the mathematical backbone of statistical inference. The LLN, CLT, and consistency of estimators are all statements about limits; this section reviews the deterministic foundations.

## Definition

A sequence $(a_n)$ **converges** to $L$, written $a_n \to L$, if

$$
\forall\, \varepsilon > 0,\;\; \exists\, N \in \mathbb{N} \text{ such that } n > N \implies |a_n - L| < \varepsilon
$$

A series $\sum_{n=1}^\infty a_n$ converges if the partial sums $S_N = \sum_{n=1}^N a_n$ converge. Asymptotic notation describes growth rates: $f(n) = O(g(n))$ means $|f(n)| \leq C|g(n)|$ for large $n$; $f(n) = o(g(n))$ means $f(n)/g(n) \to 0$.

## Explanation

**Limit laws** preserve convergence: if $a_n \to L$ and $b_n \to M$, then $a_n + b_n \to L + M$, $a_nb_n \to LM$, and $a_n/b_n \to L/M$ (when $M \neq 0$). The squeeze theorem establishes convergence by bounding from above and below.

**Key series** used in statistics:

- Geometric $\sum r^n = 1/(1-r)$ for $|r|<1$ -- normalizes geometric and negative binomial PMFs.
- Exponential $\sum x^n/n! = e^x$ -- underpins the Poisson distribution and MGFs.
- The Gaussian integral $\int_{-\infty}^{\infty} e^{-x^2/2}\,dx = \sqrt{2\pi}$ normalizes the normal density.

**Taylor expansion** $f(x) \approx f(a) + f'(a)(x-a) + f''(a)(x-a)^2/2$ is the workhorse behind the delta method and CLT derivations via MGFs.

**Asymptotic notation in statistics**: $\hat{\theta}_n - \theta = O_p(n^{-1/2})$ means the estimation error shrinks at rate $1/\sqrt{n}$, the standard rate for consistent estimators.

**Modes of convergence** (developed in Chapter 3): almost sure, in probability, in distribution, and in $L^p$. The hierarchy is: a.s. $\Rightarrow$ in probability $\Rightarrow$ in distribution.

## Examples

```python
import numpy as np

# Sequence convergence: (1 + 1/n)^n -> e
ns = [10, 100, 1_000, 10_000, 100_000]
for n in ns:
    approx = (1 + 1/n)**n
    print(f"n={n:>7d}: (1+1/n)^n = {approx:.8f}, error = {abs(approx - np.e):.2e}")

# Geometric series partial sums
r = 0.5
partial_sums = np.cumsum(r ** np.arange(20))
exact = 1 / (1 - r)
print(f"\nGeometric series (r=0.5): S_19 = {partial_sums[-1]:.10f}, exact = {exact}")

# Taylor approximation of e^x at x=0.3
x = 0.3
for k in range(1, 7):
    taylor = sum(x**n / np.math.factorial(n) for n in range(k + 1))
    print(f"Taylor order {k}: {taylor:.8f}, exact: {np.exp(x):.8f}")
```
