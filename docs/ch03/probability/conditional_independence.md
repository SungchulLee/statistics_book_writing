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

## Exercises

**Exercise 1.**
Clouds $C$, rain $R$, umbrella $U$: $P(C) = 0.4$, $P(R \mid C) = 0.6$, $P(R \mid C^c) = 0.1$, $P(U \mid R) = 0.9$, $P(U \mid R^c) = 0.2$, with $U \perp\!\!\!\perp C \mid R$ assumed. (a) Are $C$ and $U$ independent? (b) Verify $C \perp\!\!\!\perp U \mid R$.

??? success "Solution to Exercise 1"
    (a) $P(R) = 0.6 \cdot 0.4 + 0.1 \cdot 0.6 = 0.30$. $P(U) = 0.9 \cdot 0.3 + 0.2 \cdot 0.7 = 0.41$.

    Using $P(U \mid C) = P(U \mid R) P(R \mid C) + P(U \mid R^c) P(R^c \mid C) = 0.9 \cdot 0.6 + 0.2 \cdot 0.4 = 0.62$.

    Since $0.62 \ne 0.41$, $C$ and $U$ are **not** unconditionally independent.

    (b) By assumption, $P(U \mid R, C) = P(U \mid R)$. So

    $$
    P(C \cap U \mid R) = P(U \mid R, C) P(C \mid R) = P(U \mid R) P(C \mid R)
    $$

    Hence $C \perp\!\!\!\perp U \mid R$. Rain "screens off" the association between clouds and umbrellas: once you know whether it rains, clouds add no further information about umbrellas.

---

**Exercise 2.**
**Berkson's paradox.** Let $A, B$ be two independent risk factors for hospitalization $C$. Show that conditional on $C$, $A$ and $B$ become negatively correlated, even though they are independent overall.

??? success "Solution to Exercise 2"
    Suppose $A, B \in \{0, 1\}$ are independent Bernoulli($p$) and hospitalization happens if $A = 1$ or $B = 1$ (so $C = A \cup B$).

    $P(A = 1 \mid C) = P(A = 1, C)/P(C) = P(A = 1)/P(C) = p/(2p - p^2)$, which for $p = 0.5$ gives $0.5/0.75 = 2/3$.

    $P(A = 1 \mid B = 1, C) = P(A = 1 \mid B = 1) = p = 0.5$ (since $B = 1$ implies $C$).

    $P(A = 1 \mid B = 0, C) = P(A = 1 \mid B = 0, A = 1) = 1$ (because among hospitalized people with $B = 0$, $A$ must be 1).

    So given $C$, knowing $B = 1$ *decreases* $P(A = 1)$ from 2/3 to 1/2 (and knowing $B = 0$ raises it to 1). The two are negatively associated conditional on $C$, despite being unconditionally independent. **Conditioning on a common effect (collider) creates a spurious association.**

    Real-world impact: medical studies confined to hospitalized patients often find negative associations between independent risk factors — Berkson's hospital-bias.

---

**Exercise 3.**
**Mixture distribution.** A coin is randomly chosen: coin 1 has $P(H) = 0.3$, coin 2 has $P(H) = 0.7$. Two flips $X_1, X_2$ are made of the chosen coin. Show $X_1 \perp\!\!\!\perp X_2$ fails but $X_1 \perp\!\!\!\perp X_2 \mid \text{coin}$ holds.

??? success "Solution to Exercise 3"
    Given the coin choice, the flips are conditionally independent (the coin is the same and flips are independent given which coin).

    Unconditionally:

    $P(X_1 = H) = 0.5 \cdot 0.3 + 0.5 \cdot 0.7 = 0.5$. Same for $X_2$.

    $P(X_1 = H, X_2 = H) = 0.5 \cdot 0.3^2 + 0.5 \cdot 0.7^2 = 0.045 + 0.245 = 0.29 \ne 0.25 = 0.5 \cdot 0.5$.

    So unconditionally, $X_1$ and $X_2$ are *positively correlated* — if the first flip is heads, the biased-toward-heads coin was more likely chosen, making the second flip more likely heads.

    The positive correlation $\rho(X_1, X_2) > 0$ reflects information about the latent coin choice. Once the coin choice is known (i.e., conditioning on it), the correlation disappears. This is the basis of **latent variable models** in statistics: an observed correlation can be explained by an unobserved common cause.

---

**Exercise 4.**
**Markov chains.** A Markov chain $X_0, X_1, X_2, \ldots$ satisfies $X_n \perp\!\!\!\perp \{X_0, \ldots, X_{n-2}\} \mid X_{n-1}$. Verbalize this property and explain how it enables tractable inference.

??? success "Solution to Exercise 4"
    **Markov property** (verbal): given the present ($X_{n-1}$), the future ($X_n$) is conditionally independent of the past ($X_0, \ldots, X_{n-2}$). The present "screens off" the past from the future.

    **Tractable inference:** the joint distribution of $(X_0, \ldots, X_T)$ factors as

    $$
    P(X_0, X_1, \ldots, X_T) = P(X_0) \prod_{t=1}^T P(X_t \mid X_{t-1})
    $$

    Each conditional only depends on the previous state, not the entire history. With a finite state space of size $k$, this requires only $k$ initial-state probabilities and $k^2$ transition probabilities, rather than the exponentially many parameters of a fully general joint over $T + 1$ variables.

    Markov chains underlie language models, hidden Markov models, MCMC sampling, queuing theory, and PageRank. The conditional independence structure is what makes these tractable.

---

**Exercise 5.**
**Common-cause structure.** Three variables $X, Y, Z$ form a "chain" $X \to Z \to Y$. Verify $X \perp\!\!\!\perp Y \mid Z$ from the factorization $P(X, Y, Z) = P(X) P(Z \mid X) P(Y \mid Z)$.

??? success "Solution to Exercise 5"
    $P(Y \mid X, Z) = P(X, Y, Z)/P(X, Z) = P(X) P(Z \mid X) P(Y \mid Z)/(P(X) P(Z \mid X)) = P(Y \mid Z)$.

    Since $P(Y \mid X, Z) = P(Y \mid Z)$ doesn't depend on $X$, $Y$ is conditionally independent of $X$ given $Z$.

    Equivalent way: $P(X \cap Y \mid Z) = P(X \mid Z) P(Y \mid Z, X) = P(X \mid Z) P(Y \mid Z)$.

    **Causal interpretation:** if $X$ affects $Y$ only *through* $Z$ (no direct $X \to Y$ arrow), then once $Z$ is known, $X$ provides no further information about $Y$. This is the "chain" pattern in d-separation rules for causal graphs.

    By contrast, in a **collider** $X \to Z \leftarrow Y$, conditioning on $Z$ creates a spurious dependence between $X$ and $Y$ — opposite of the chain pattern. Distinguishing chains, forks, and colliders is the core of graphical causal modeling.

---

**Exercise 6.**
**The "explaining away" effect.** In a Bayesian network with two causes $A, B$ for effect $E$, suppose $A$ and $B$ are independent priors. After observing $E$, learning $A$ occurred *decreases* the posterior probability that $B$ also occurred. Demonstrate with $P(A) = P(B) = 0.1$, $P(E \mid A, B) = 1$, $P(E \mid A, B^c) = 0.8$, $P(E \mid A^c, B) = 0.8$, $P(E \mid A^c, B^c) = 0$.

??? success "Solution to Exercise 6"
    Compute joint probabilities:

    $P(A, B, E) = 0.1 \cdot 0.1 \cdot 1 = 0.01$.
    $P(A, B^c, E) = 0.1 \cdot 0.9 \cdot 0.8 = 0.072$.
    $P(A^c, B, E) = 0.9 \cdot 0.1 \cdot 0.8 = 0.072$.
    $P(A^c, B^c, E) = 0.9 \cdot 0.9 \cdot 0 = 0$.

    $P(E) = 0.01 + 0.072 + 0.072 + 0 = 0.154$.

    Posteriors:

    $P(B \mid E) = (0.01 + 0.072)/0.154 = 0.082/0.154 \approx 0.532$.

    $P(B \mid A, E) = P(A, B, E)/P(A, E) = 0.01/(0.01 + 0.072) = 0.01/0.082 \approx 0.122$.

    So after observing the effect, knowing that $A$ also occurred *drops* the probability of $B$ from 53% to 12%. The observed effect was "explained away" by $A$ — once we know $A$ caused $E$, the alternative cause $B$ becomes less likely.

    **Real-world example:** an alarm goes off; you initially suspect either a burglar or an earthquake. Hearing on the radio that an earthquake just occurred explains the alarm and makes the burglar hypothesis less likely. This is the formal mechanism of "competing explanations" in Bayesian reasoning.
