# Numerical Stability (Log-Sum-Exp Trick)

## The Overflow Problem

Computing the softmax function requires exponentiating the logits $z_1, \ldots,
z_C$.  In 64-bit floating point, $e^{z}$ overflows to infinity when
$z \gtrsim 709$ and underflows to zero when $z \lesssim -745$.  Raw logits
from a neural network or a large linear model can easily exceed these
thresholds, so a naive implementation of softmax produces `inf` or `0/0`.

This section develops the **log-sum-exp trick**, a simple algebraic identity
that eliminates overflow and greatly reduces underflow without changing the
mathematical result.

## Shift Invariance of Softmax

Recall from the [softmax page](softmax.md) that softmax is shift-invariant:
for any scalar $c$,

$$
\operatorname{softmax}(\mathbf{z})_k
= \frac{e^{z_k}}{\sum_{j=1}^{C}e^{z_j}}
= \frac{e^{z_k - c}}{\sum_{j=1}^{C}e^{z_j - c}}
$$

This identity holds because multiplying numerator and denominator by $e^{-c}$
cancels.  By choosing $c$ wisely, we keep all exponents in a safe range.

## The Log-Sum-Exp Identity

The **log-sum-exp** (LSE) of a vector $\mathbf{z}$ is

$$
\operatorname{LSE}(\mathbf{z})
= \log\sum_{j=1}^{C}e^{z_j}
$$

Applying the shift with $c = \max_j z_j$:

$$
\operatorname{LSE}(\mathbf{z})
= c + \log\sum_{j=1}^{C}e^{z_j - c}
$$

where $c = \max_j z_j$.  After the shift, every exponent satisfies
$z_j - c \le 0$, so $e^{z_j - c} \le 1$.  This guarantees:

- **No overflow:** The largest exponent is $e^0 = 1$.
- **Reduced underflow:** Only logits far below the maximum underflow, and
  they contribute negligibly to the sum anyway.

## Stable Softmax via Log-Sum-Exp

Combining the two ideas, the numerically stable softmax is

$$
\log\operatorname{softmax}(\mathbf{z})_k
= z_k - \operatorname{LSE}(\mathbf{z})
= (z_k - c) - \log\sum_{j=1}^{C}e^{z_j - c}
$$

To obtain the probabilities themselves, exponentiate: $\operatorname{softmax}(\mathbf{z})_k = e^{\log\operatorname{softmax}(\mathbf{z})_k}$.  Working in log space as long as possible further improves stability.

## Stable Cross-Entropy Loss

The cross-entropy loss for a single observation with true class $y$ is

$$
\mathcal{L} = -\log\operatorname{softmax}(\mathbf{z})_y
= -z_y + \operatorname{LSE}(\mathbf{z})
$$

Using the stable LSE:

$$
\mathcal{L} = -(z_y - c) + \log\sum_{j=1}^{C}e^{z_j - c}
$$

This formulation avoids ever computing $e^{z_j}$ directly for large $z_j$.
Frameworks such as PyTorch and TensorFlow implement this in fused operations
like `cross_entropy_with_logits` that accept raw logits rather than
probabilities.

!!! warning "Never Pass Probabilities to the Loss"
    Computing `softmax` first and then taking `log` reintroduces numerical
    issues: if a probability underflows to zero, `log(0)` produces `-inf`.
    Always use the fused log-softmax or cross-entropy-from-logits functions
    provided by your framework.

## Step-by-Step Algorithm

| Step | Operation | Purpose |
|---|---|---|
| 1 | $c \leftarrow \max_j z_j$ | Find the shift constant |
| 2 | $s_j \leftarrow z_j - c$ for all $j$ | Shift logits to non-positive range |
| 3 | $e_j \leftarrow e^{s_j}$ for all $j$ | Safe exponentiation ($e_j \le 1$) |
| 4 | $S \leftarrow \sum_j e_j$ | Denominator of softmax |
| 5 | $p_j \leftarrow e_j / S$ | Softmax probabilities |
| 6 | $\operatorname{LSE} \leftarrow c + \log S$ | For the loss function |

??? example "Worked Example"
    Let $C = 3$ with logits $\mathbf{z} = (1000,\; 1001,\; 999)$.

    **Naive computation.** $e^{1000}$ overflows to `inf` in float64, producing
    `inf / inf = NaN`.

    **Stable computation.**

    1. $c = 1001$.
    2. Shifted logits: $(-1, 0, -2)$.
    3. Exponentials: $(e^{-1}, e^0, e^{-2}) = (0.3679, 1.0, 0.1353)$.
    4. Sum: $S = 1.5032$.
    5. Probabilities: $(0.2447, 0.6652, 0.0900)$.
    6. $\operatorname{LSE} = 1001 + \log(1.5032) = 1001.407$.

    All arithmetic stays in a safe range, and the probabilities are identical to
    the exact mathematical values.

## Log-Sum-Exp for Two Terms (Sigmoid)

In binary logistic regression ($C = 2$), the LSE reduces to

$$
\log(1 + e^z) = \max(0, z) + \log(1 + e^{-|z|})
$$

This identity keeps the exponent $-|z| \le 0$, preventing overflow.  It is the
standard implementation of the **softplus** function in numerical libraries.

## Summary

| Problem | Naive approach | Stable approach |
|---|---|---|
| Softmax overflow | $e^{z_j}$ for large $z_j$ | Subtract $c = \max z_j$ first |
| Log-softmax underflow | $\log(0)$ when $p_k \approx 0$ | Compute in log space via LSE |
| Cross-entropy loss | Softmax then log | Use fused `cross_entropy_with_logits` |
| Softplus ($C = 2$) | $\log(1 + e^z)$ overflows | $\max(0,z) + \log(1 + e^{-|z|})$ |
