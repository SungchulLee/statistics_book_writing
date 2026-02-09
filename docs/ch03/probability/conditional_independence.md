# Conditional Independence

## Overview

**Conditional independence** extends the notion of independence by introducing a conditioning event. Two events may be dependent overall but become independent once we condition on additional information—or vice versa. This concept is central to graphical models, Bayesian networks, and causal reasoning.

---

## Definition

Events $A$ and $B$ are **conditionally independent** given event $C$ (with $P(C) > 0$) if:

$$
P(A \cap B \mid C) = P(A \mid C) \cdot P(B \mid C)
$$

Equivalently, if $P(B \cap C) > 0$:

$$
P(A \mid B \cap C) = P(A \mid C)
$$

**Interpretation:** Once we know $C$ has occurred, learning that $B$ has occurred provides no additional information about $A$.

We write $A \perp\!\!\!\perp B \mid C$ to denote conditional independence.

---

## Independence Does Not Imply Conditional Independence

Two events can be (unconditionally) independent but become dependent after conditioning. This is known as **Berkson's paradox** or the **explaining away** effect.

### Example: Two Causes of a Shared Effect

Suppose a fire alarm ($C$) can be triggered by either a fire ($A$) or burnt toast ($B$). The two causes are independent:

$$
P(A \cap B) = P(A) \cdot P(B)
$$

But given the alarm went off ($C$), learning there is no fire makes burnt toast more likely:

$$
P(B \mid A^c \cap C) > P(B \mid C)
$$

So $A \perp\!\!\!\perp B$ but $A \not\perp\!\!\!\perp B \mid C$.

---

## Conditional Independence Does Not Imply Independence

Conversely, events can be conditionally independent given $C$ but not unconditionally independent.

### Example: Drawing from a Mixture

A coin is chosen at random: coin 1 has $P(\text{H}) = 0.3$ and coin 2 has $P(\text{H}) = 0.7$. Let $C$ indicate which coin is chosen, and let $A$ and $B$ be the outcomes of two flips.

Given the coin ($C$), the flips are independent:

$$
P(A \cap B \mid C) = P(A \mid C) \cdot P(B \mid C)
$$

But without knowing the coin, the flips are dependent—if the first flip is heads, it is more likely that the biased-toward-heads coin was chosen, which increases the probability of the second flip being heads.

---

## Examples

### Example: Students and a Shared Exam

Two students, $A$ and $B$, take the same exam. Let event $A_{\text{pass}}$ and $B_{\text{pass}}$ denote each passing. Let $C$ = "the exam was easy."

Given $C$, whether student $A$ passes provides little information about student $B$ (their abilities are separate). But unconditionally, learning $A$ passed makes it more likely the exam was easy, which in turn makes $B$ passing more likely.

$$
A_{\text{pass}} \perp\!\!\!\perp B_{\text{pass}} \mid C \quad \text{but} \quad A_{\text{pass}} \not\perp\!\!\!\perp B_{\text{pass}}
$$

### Example: Dice with Known Sum

Roll two fair dice. Let $A$ = "die 1 shows 4" and $B$ = "die 2 shows 3." These are independent. But condition on $C$ = "the sum is 7":

$$
P(A \mid C) = \frac{1}{6}, \quad P(B \mid C) = \frac{1}{6}, \quad P(A \cap B \mid C) = \frac{1}{6}
$$

Here $P(A \cap B \mid C) = 1/6 \neq (1/6)(1/6)$, so $A$ and $B$ are **not** conditionally independent given $C$. In fact, knowing die 1 is 4 and the sum is 7 determines die 2 is 3 with certainty.

---

## Summary of Relationships

| Scenario | $A \perp\!\!\!\perp B$ | $A \perp\!\!\!\perp B \mid C$ |
|:---|:---:|:---:|
| Independent, and remains so after conditioning | ✓ | ✓ |
| Independent, but dependent after conditioning (Berkson's) | ✓ | ✗ |
| Dependent, but independent after conditioning | ✗ | ✓ |
| Dependent, and remains so after conditioning | ✗ | ✗ |

All four scenarios are possible. There is **no logical implication** in either direction between independence and conditional independence.

---

## Python Exploration

```python
import numpy as np

def mixture_coin_simulation(n_simulations=200_000):
    """Demonstrate conditional independence in a mixture model."""
    np.random.seed(42)

    # Choose coin: coin 0 has P(H)=0.3, coin 1 has P(H)=0.7
    coin = np.random.randint(0, 2, size=n_simulations)
    p_heads = np.where(coin == 0, 0.3, 0.7)

    flip1 = np.random.rand(n_simulations) < p_heads
    flip2 = np.random.rand(n_simulations) < p_heads

    # Unconditional: P(flip2=H | flip1=H) vs P(flip2=H)
    p_f2 = flip2.mean()
    p_f2_given_f1 = flip2[flip1].mean()
    print("=== Unconditional (marginal) ===")
    print(f"P(flip2=H) = {p_f2:.4f}")
    print(f"P(flip2=H | flip1=H) = {p_f2_given_f1:.4f}")
    print(f"Not independent: {abs(p_f2 - p_f2_given_f1) > 0.01}\n")

    # Conditional on coin 0
    mask_c0 = coin == 0
    p_f2_c0 = flip2[mask_c0].mean()
    p_f2_given_f1_c0 = flip2[mask_c0 & flip1].mean()
    print("=== Conditional on coin 0 (P(H)=0.3) ===")
    print(f"P(flip2=H | coin=0) = {p_f2_c0:.4f}")
    print(f"P(flip2=H | flip1=H, coin=0) = {p_f2_given_f1_c0:.4f}")
    print(f"Conditionally independent: {abs(p_f2_c0 - p_f2_given_f1_c0) < 0.02}")

mixture_coin_simulation()
```

```python
import numpy as np

def berkson_paradox_simulation(n_simulations=200_000):
    """Demonstrate Berkson's paradox: independent events become
    dependent after conditioning on a shared effect."""
    np.random.seed(42)

    # A = fire (rare), B = burnt toast (common), C = alarm
    p_fire = 0.01
    p_toast = 0.10

    fire = np.random.rand(n_simulations) < p_fire
    toast = np.random.rand(n_simulations) < p_toast
    alarm = fire | toast  # alarm if either occurs

    # Unconditional independence
    p_fire_given_toast = fire[toast].mean()
    print(f"P(fire) = {fire.mean():.4f}")
    print(f"P(fire | toast) = {p_fire_given_toast:.4f}")
    print(f"Unconditionally independent: {abs(fire.mean() - p_fire_given_toast) < 0.005}\n")

    # Conditional on alarm: explaining away
    p_fire_given_alarm = fire[alarm].mean()
    p_fire_given_alarm_no_toast = fire[alarm & ~toast].mean()
    print(f"P(fire | alarm) = {p_fire_given_alarm:.4f}")
    print(f"P(fire | alarm, no toast) = {p_fire_given_alarm_no_toast:.4f}")
    print(f"Conditionally dependent (explaining away): "
          f"{abs(p_fire_given_alarm - p_fire_given_alarm_no_toast) > 0.01}")

berkson_paradox_simulation()
```

---

## Key Takeaways

- Conditional independence means that once $C$ is known, $A$ and $B$ carry no information about each other.
- Independence does **not** imply conditional independence, and vice versa.
- Berkson's paradox shows how conditioning on a common effect induces dependence between its independent causes.
- Conditional independence is the structural assumption behind Bayesian networks and Markov models.
