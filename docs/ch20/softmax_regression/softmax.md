# Softmax Function and Probability Simplex

## Definition

The **softmax** function maps a vector of $C$ real-valued logits
$\mathbf{z}=(z_1,\ldots,z_C)$ to a probability distribution:

$$
\operatorname{softmax}(\mathbf{z})_c
= \frac{e^{z_c}}{\sum_{c'=1}^{C}e^{z_{c'}}},
\qquad c=1,\ldots,C
$$

### Properties

1. **Non-negative:** Every output is strictly positive.
2. **Sums to one:** $\sum_c\operatorname{softmax}(\mathbf{z})_c = 1$.
3. **Monotone:** A larger logit $z_c$ produces a larger probability.
4. **Shift invariance:** $\operatorname{softmax}(\mathbf{z}+\alpha\mathbf{1})
   = \operatorname{softmax}(\mathbf{z})$ for any scalar $\alpha$.

Property 4 is exploited for numerical stability: before exponentiation
we subtract $\max_c z_c$.

## The Probability Simplex

The output of softmax lies on the **probability simplex**

$$
\Delta^{C-1} = \Bigl\{\mathbf{p}\in\mathbb{R}^C : p_c\ge 0,\;
\sum_c p_c = 1\Bigr\}
$$

For $C=3$ this is a triangle in 3D space; for $C=10$ (MNIST) it is a
9-dimensional simplex.

## Softmax as a Generalization of the Sigmoid

When $C=2$ with logits $(z_1,z_2)$:

$$
\operatorname{softmax}(\mathbf{z})_1
= \frac{e^{z_1}}{e^{z_1}+e^{z_2}}
= \frac{1}{1+e^{-(z_1-z_2)}}
= \sigma(z_1-z_2)
$$

Thus the binary softmax is exactly the sigmoid applied to the difference
of the two logits.

## Temperature Scaling

A common variant introduces a temperature parameter $\tau>0$:

$$
\operatorname{softmax}(\mathbf{z}/\tau)_c
= \frac{e^{z_c/\tau}}{\sum_{c'}e^{z_{c'}/\tau}}
$$

As $\tau\to 0$ the distribution collapses to a point mass on
$\arg\max_c z_c$ (hard decision); as $\tau\to\infty$ it approaches the
uniform distribution.  Temperature scaling is used in model calibration
and in generative models (e.g. controlling the "creativity" of language
models).

## NumPy Implementation

```python
import numpy as np

def softmax(z):
    """Numerically stable softmax."""
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
```

The `np.max` subtraction prevents overflow in `np.exp` without
changing the result (shift invariance).
