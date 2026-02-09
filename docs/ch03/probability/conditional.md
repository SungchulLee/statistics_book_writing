# Conditional Probability

## Overview

**Conditional probability** quantifies how the probability of an event changes when we learn that another event has occurred. It is one of the most important concepts in probability theory, forming the basis for Bayesian reasoning, statistical inference, and decision-making under uncertainty.

---

## Definition

The **conditional probability** of event $A$ given event $B$ (where $P(B) > 0$) is:

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)}
$$

**Interpretation:** Out of the total weight (probability) allocated to outcomes in $B$, $P(A \mid B)$ is the fraction of that weight which also belongs to $A$. Conditioning on $B$ effectively **restricts the sample space** from $\Omega$ to $B$ and renormalizes the probabilities.

---

## Intuition: Updating the Sample Space

When we condition on $B$, we discard all outcomes outside $B$ and rescale the remaining probabilities so they sum to 1:

$$
P(A \mid B) = \frac{\text{Weight of bricks in } A \cap B}{\text{Weight of bricks in } B}
$$

This is equivalent to saying: "If we know $B$ happened, what fraction of $B$'s probability belongs to $A$?"

---

## The Multiplication Rule

Rearranging the definition gives the **multiplication rule**:

$$
P(A \cap B) = P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A)
$$

This extends to chains of events:

$$
P(A \cap B \cap C) = P(A) \cdot P(B \mid A) \cdot P(C \mid A \cap B)
$$

---

## The Law of Total Probability

If $B_1, B_2, \ldots, B_n$ form a **partition** of $\Omega$ (i.e., they are mutually exclusive and their union is $\Omega$), then for any event $A$:

$$
P(A) = \sum_{i=1}^{n} P(A \mid B_i) \cdot P(B_i)
$$

This decomposes the probability of $A$ by considering each scenario $B_i$ separately.

---

## Examples

### Example: Drawing Cards

A card is drawn from a standard 52-card deck. Let $A$ = "the card is a king" and $B$ = "the card is a face card (J, Q, K)."

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)} = \frac{4/52}{12/52} = \frac{4}{12} = \frac{1}{3}
$$

Since all kings are face cards, $A \cap B = A$, so knowing the card is a face card narrows the possibilities to 12, of which 4 are kings.

### Example: Two Dice

Roll two fair dice. Let $A$ = "the sum is 8" and $B$ = "the first die shows 3."

- $P(B) = 1/6$
- $A \cap B = \{(3, 5)\}$, so $P(A \cap B) = 1/36$

$$
P(A \mid B) = \frac{1/36}{1/6} = \frac{1}{6}
$$

### Example: Medical Testing (Total Probability)

A disease affects 1% of a population. A test has 95% sensitivity ($P(\text{positive} \mid \text{disease}) = 0.95$) and 90% specificity ($P(\text{negative} \mid \text{no disease}) = 0.90$).

The probability of testing positive:

$$
\begin{aligned}
P(\text{positive}) &= P(\text{positive} \mid \text{disease}) \cdot P(\text{disease}) + P(\text{positive} \mid \text{no disease}) \cdot P(\text{no disease}) \\
&= 0.95 \times 0.01 + 0.10 \times 0.99 \\
&= 0.0095 + 0.099 = 0.1085
\end{aligned}
$$

About 10.85% of the population would test positive, even though only 1% actually has the disease.

---

## Python Exploration

```python
import numpy as np

def conditional_probability_simulation(n_simulations=100_000):
    """Simulate conditional probability with two dice."""
    np.random.seed(42)

    die1 = np.random.randint(1, 7, size=n_simulations)
    die2 = np.random.randint(1, 7, size=n_simulations)
    total = die1 + die2

    # P(sum=8 | die1=3)
    mask_B = die1 == 3
    mask_A_and_B = (die1 == 3) & (total == 8)

    p_conditional = mask_A_and_B.sum() / mask_B.sum()
    print(f"Simulated P(sum=8 | die1=3) = {p_conditional:.4f}")
    print(f"Theoretical P(sum=8 | die1=3) = {1/6:.4f}")

conditional_probability_simulation()
```

```python
import numpy as np

def medical_test_simulation(n_people=1_000_000):
    """Simulate the medical testing example using total probability."""
    np.random.seed(42)

    prevalence = 0.01
    sensitivity = 0.95
    false_positive_rate = 0.10

    has_disease = np.random.rand(n_people) < prevalence
    test_positive = np.where(
        has_disease,
        np.random.rand(n_people) < sensitivity,
        np.random.rand(n_people) < false_positive_rate
    )

    p_positive = test_positive.mean()
    print(f"Simulated P(positive) = {p_positive:.4f}")
    print(f"Theoretical P(positive) = {0.1085:.4f}")

medical_test_simulation()
```

---

## Key Takeaways

- Conditional probability $P(A \mid B)$ updates our belief about $A$ after observing $B$.
- Conditioning restricts the sample space to $B$ and renormalizes probabilities.
- The multiplication rule connects joint and conditional probabilities.
- The law of total probability decomposes $P(A)$ across a partition of the sample space.
