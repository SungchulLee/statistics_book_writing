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

## Exercises

**Exercise 1.**
A jar contains 4 red and 6 blue marbles. Two marbles are drawn without replacement. What is the probability that the second marble is red given that the first marble is blue?

??? success "Solution to Exercise 1"
    Let $B_1$ = "first marble is blue" and $R_2$ = "second marble is red."

    After drawing one blue marble, the jar contains 4 red and 5 blue marbles (9 total). Therefore:

    $$
    P(R_2 \mid B_1) = \frac{4}{9}
    $$

---

**Exercise 2.**
In a factory, Machine A produces 60% of the items and Machine B produces 40%. Machine A has a defect rate of 2%, while Machine B has a defect rate of 5%. An item is selected at random and found to be defective. Using the law of total probability, compute the probability that the item is defective.

??? success "Solution to Exercise 2"
    Let $A$ = "produced by Machine A", $B$ = "produced by Machine B", and $D$ = "defective." We have:

    $$
    P(A) = 0.60, \quad P(B) = 0.40
    $$

    $$
    P(D \mid A) = 0.02, \quad P(D \mid B) = 0.05
    $$

    By the law of total probability:

    $$
    P(D) = P(D \mid A) P(A) + P(D \mid B) P(B) = 0.02 \times 0.60 + 0.05 \times 0.40 = 0.012 + 0.020 = 0.032
    $$

    The overall defect rate is 3.2%.

---

**Exercise 3.**
Prove that if $P(B) > 0$, then $P(\cdot \mid B)$ satisfies the three axioms of probability. That is, show that conditional probability is itself a valid probability measure on the restricted sample space.

??? success "Solution to Exercise 3"
    We verify the three axioms for $P(\cdot \mid B)$:

    **Non-negativity:** For any event $A$, $P(A \cap B) \geq 0$ and $P(B) > 0$, so:

    $$
    P(A \mid B) = \frac{P(A \cap B)}{P(B)} \geq 0
    $$

    **Normalization:**

    $$
    P(\Omega \mid B) = \frac{P(\Omega \cap B)}{P(B)} = \frac{P(B)}{P(B)} = 1
    $$

    **Countable additivity:** If $A_1, A_2, \ldots$ are mutually disjoint events, then $A_1 \cap B, A_2 \cap B, \ldots$ are also mutually disjoint, so:

    $$
    P\!\left(\bigcup_i A_i \mid B\right) = \frac{P\!\left(\bigcup_i (A_i \cap B)\right)}{P(B)} = \frac{\sum_i P(A_i \cap B)}{P(B)} = \sum_i P(A_i \mid B)
    $$

    All three axioms hold, so $P(\cdot \mid B)$ is a valid probability measure. $\square$

---

**Exercise 4.**
Two fair dice are rolled. Let $A$ = "the sum is at least 10" and $B$ = "both dice show 5 or higher." Compute $P(A \mid B)$.

??? success "Solution to Exercise 4"
    First, identify the event $B$ = "both dice show 5 or higher." Each die can be 5 or 6, so $B = \{(5,5),(5,6),(6,5),(6,6)\}$ with $|B| = 4$ and $P(B) = 4/36$.

    Next, $A \cap B$ = outcomes in $B$ where the sum is at least 10:

    - $(5,5)$: sum $= 10$ (yes)
    - $(5,6)$: sum $= 11$ (yes)
    - $(6,5)$: sum $= 11$ (yes)
    - $(6,6)$: sum $= 12$ (yes)

    All four outcomes in $B$ have sum $\geq 10$, so $A \cap B = B$ and $P(A \cap B) = 4/36$.

    $$
    P(A \mid B) = \frac{P(A \cap B)}{P(B)} = \frac{4/36}{4/36} = 1
    $$

    Given that both dice show 5 or higher, the sum is guaranteed to be at least 10.

---

**Exercise 5.**
**The chain rule** for joint probabilities: $P(A_1, A_2, A_3) = P(A_1) P(A_2 \mid A_1) P(A_3 \mid A_1, A_2)$. Use this to compute $P(\text{3 hearts in a row})$ when drawing 3 cards from a standard 52-card deck without replacement.

??? success "Solution to Exercise 5"
    Let $H_i$ be "the $i$-th card is a heart." There are 13 hearts in a 52-card deck.

    $$
    P(H_1, H_2, H_3) = P(H_1) P(H_2 \mid H_1) P(H_3 \mid H_1, H_2)
    $$

    $P(H_1) = 13/52 = 1/4$.

    After drawing one heart, the deck has 51 cards including 12 hearts: $P(H_2 \mid H_1) = 12/51$.

    After drawing two hearts, the deck has 50 cards including 11 hearts: $P(H_3 \mid H_1, H_2) = 11/50$.

    Joint: $P(H_1, H_2, H_3) = (1/4)(12/51)(11/50) = 132/10200 = 11/850 \approx 0.0129$.

    About 1.3% probability of drawing 3 hearts in a row.

    **General chain rule:** $P(A_1, \ldots, A_n) = \prod_{i=1}^n P(A_i \mid A_1, \ldots, A_{i-1})$. This is the foundation of sequential probability models — Markov chains, hidden Markov models, sequential Bayesian updating.

---

**Exercise 6.**
**The Monty Hall problem.** Three doors; one hides a car, two hide goats. You pick door 1. The host (who knows what's behind each door) opens door 3, revealing a goat, and offers you a chance to switch. What is the probability of winning if you switch vs. stay?

??? success "Solution to Exercise 6"
    Let $C_i$ be "car is behind door $i$" (uniform prior $P(C_i) = 1/3$ for $i = 1, 2, 3$). Let $H_3$ be "host opens door 3."

    The host's behavior: if you picked the car (door 1, $C_1$), the host picks randomly between doors 2 and 3, so $P(H_3 \mid C_1) = 1/2$. If the car is behind door 2 ($C_2$), the host must open door 3, so $P(H_3 \mid C_2) = 1$. If the car is behind door 3, the host cannot open it, so $P(H_3 \mid C_3) = 0$.

    By Bayes:

    $P(C_1 \mid H_3) = (1/2)(1/3) / P(H_3) = (1/6)/P(H_3)$.

    $P(C_2 \mid H_3) = (1)(1/3) / P(H_3) = (1/3)/P(H_3)$.

    $P(C_3 \mid H_3) = (0)(1/3) / P(H_3) = 0$.

    Normalizing: $P(H_3) = 1/6 + 1/3 + 0 = 1/2$. So $P(C_1 \mid H_3) = 1/3$, $P(C_2 \mid H_3) = 2/3$, $P(C_3 \mid H_3) = 0$.

    **Stay**: win with probability $P(C_1 \mid H_3) = 1/3$.
    **Switch**: win with probability $P(C_2 \mid H_3) = 2/3$.

    Switching doubles your winning probability. The intuition: your initial pick had probability 1/3 of being correct; the door the host doesn't open carries the remaining 2/3 probability because the host's choice gives information. This problem famously confused even mathematicians when it was popularized — the answer feels wrong until you formalize it with Bayes' theorem.
