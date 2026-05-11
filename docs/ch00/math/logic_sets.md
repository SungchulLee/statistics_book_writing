# Sets, Functions, and Logic

This section establishes the foundational mathematical language used throughout the book. Precise definitions of logic, sets, and functions prevent ambiguity when we later introduce probability spaces (which are triples of a set, a $\sigma$-algebra, and a measure), random variables (which are measurable functions), and statistical inference (which manipulates assertions about populations using logical structure).

## Definition

### Propositions and connectives

A **proposition** is a declarative sentence that is either true or false. The fundamental connectives are:

| Symbol | Name | Read as | Truth condition |
|---|---|---|---|
| $\neg P$ | Negation | "not $P$" | true iff $P$ is false |
| $P \land Q$ | Conjunction | "$P$ and $Q$" | true iff both are true |
| $P \lor Q$ | Disjunction | "$P$ or $Q$" (inclusive) | true iff at least one is true |
| $P \Rightarrow Q$ | Implication | "if $P$ then $Q$" | false only when $P$ true and $Q$ false |
| $P \Leftrightarrow Q$ | Biconditional | "$P$ if and only if $Q$" | true iff $P$ and $Q$ have the same truth value |

The contrapositive $\neg Q \Rightarrow \neg P$ is logically equivalent to $P \Rightarrow Q$. The converse $Q \Rightarrow P$ is **not** equivalent and must be proved separately.

### Sets and set operations

A **set** is an unordered collection of distinct objects. We write $x \in A$ for "$x$ is an element of $A$". The empty set is $\emptyset$. Subset is $A \subseteq B$. Standard operations include:

$$
A \cup B = \{x : x \in A \text{ or } x \in B\}, \qquad A \cap B = \{x : x \in A \text{ and } x \in B\}
$$

$$
A \setminus B = \{x : x \in A \text{ and } x \notin B\}, \qquad A^c = \Omega \setminus A
$$

The **power set** $2^A$ is the set of all subsets of $A$. The **Cartesian product** is $A \times B = \{(a, b) : a \in A, b \in B\}$.

### Functions

A **function** $f: A \to B$ assigns to each $x \in A$ exactly one $f(x) \in B$. Call $A$ the domain and $B$ the codomain. The **image** is $f(A) = \{f(x) : x \in A\} \subseteq B$. A function is:

- **Injective** (one-to-one) if $f(x_1) = f(x_2) \Rightarrow x_1 = x_2$.
- **Surjective** (onto) if $f(A) = B$.
- **Bijective** if both.

A bijection is the rigorous version of "the same number of elements" and is the basis for defining cardinality (countability).

## Explanation

### Quantifiers and their negation

**Quantifiers** formalize "for all" ($\forall$) and "there exists" ($\exists$). Negation swaps them and negates the inner predicate:

$$
\neg(\forall\, x \in A,\; P(x)) \;\Leftrightarrow\; \exists\, x \in A \text{ s.t. } \neg P(x)
$$

$$
\neg(\exists\, x \in A,\; P(x)) \;\Leftrightarrow\; \forall\, x \in A,\; \neg P(x)
$$

The order of quantifiers matters: $\forall x \exists y\, P(x, y)$ (for each $x$ some $y$ may work) is logically weaker than $\exists y \forall x\, P(x, y)$ (one $y$ works for all $x$). The convergence definition $\forall \varepsilon \, \exists N \, \forall n > N$ has this nested structure, and reversing the first two quantifiers gives uniform convergence — a strictly stronger condition.

### De Morgan's laws

For sets,

$$
(A \cup B)^c = A^c \cap B^c, \qquad (A \cap B)^c = A^c \cup B^c
$$

These generalize to arbitrary (even uncountable) collections:

$$
\left(\bigcup_{\alpha} A_\alpha\right)^{\!c} = \bigcap_{\alpha} A_\alpha^c, \qquad \left(\bigcap_{\alpha} A_\alpha\right)^{\!c} = \bigcup_{\alpha} A_\alpha^c
$$

This is the workhorse for moving between "at least one" and "all" — a constant move in probability when computing complements of unions of events.

### Countability and its consequences

A set is **countably infinite** if it bijects onto $\mathbb{N}$ (e.g., $\mathbb{Z}$, $\mathbb{Q}$, the rationals in any interval). It is **uncountable** otherwise. Cantor's diagonal argument shows $\mathbb{R}$ is uncountable.

The distinction is foundational for probability:

- For countable sample spaces, a probability measure is determined by a PMF — the probability of any event is a sum.
- For uncountable sample spaces (e.g., real-valued measurements), individual outcomes have probability zero and probabilities are defined via integrals of a density.

### Functions used throughout the book

- **Indicator function** $\mathbf{1}_A(x) = 1$ if $x \in A$, $0$ otherwise. The bridge between sets (events) and numbers (random variables): $\mathbb{E}[\mathbf{1}_A] = P(A)$.
- **Exponential** $e^x$ and **logarithm** $\ln x$ — appear in MGFs, log-likelihoods, and entropy.
- **Logistic / sigmoid** $\sigma(x) = 1/(1 + e^{-x})$ — maps the real line to $(0, 1)$; inverse of the logit; foundational for logistic regression.
- **Gamma function** $\Gamma(\alpha) = \int_0^\infty t^{\alpha - 1} e^{-t}\, dt$ — generalizes the factorial; normalizes gamma, beta, $t$, and $\chi^2$ densities.

!!! note "Vacuous truth"
    When $P$ is false, $P \Rightarrow Q$ is true regardless of $Q$. This matters when conditioning on probability-zero events: any statement is "true" on a null event, so probability statements must be interpreted as holding almost surely.

## Examples

```python
import numpy as np

# === De Morgan's law with finite sets ===
omega = set(range(1, 11))
A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7}

A_c, B_c = omega - A, omega - B
lhs = omega - (A | B)   # (A ∪ B)^c
rhs = A_c & B_c          # A^c ∩ B^c
print(f"(A ∪ B)^c = {lhs}")
print(f"A^c ∩ B^c = {rhs}")
print(f"Equal: {lhs == rhs}")

# === Indicator function and its connection to probability ===
rng = np.random.default_rng(42)
samples = rng.integers(low=1, high=11, size=10_000)
prob_A = np.mean([s in A for s in samples])
print(f"P(A) estimate: {prob_A:.3f} (true 1/2 since |A|=5 of 10)")

# === Bijection demonstration: N <-> Z ===
def bijection_N_to_Z(n):
    # n=1 -> 0, n=2 -> 1, n=3 -> -1, n=4 -> 2, n=5 -> -2, ...
    return n // 2 if n % 2 == 0 else -(n // 2)

print([bijection_N_to_Z(n) for n in range(1, 11)])
```

## Exercises

**Exercise 1.**
Write the formal negation of each statement.

**(a)** $\forall\, x \in \mathbb{R},\; x^2 \geq 0$
**(b)** $\exists\, \varepsilon > 0 \text{ such that } \forall\, n \in \mathbb{N},\; a_n > \varepsilon$
**(c)** $\forall\, \varepsilon > 0,\; \exists\, N \in \mathbb{N} \text{ such that } n > N \implies |a_n - L| < \varepsilon$

??? success "Solution to Exercise 1"
    (a) $\exists\, x \in \mathbb{R} \text{ such that } x^2 < 0$.
    (b) $\forall\, \varepsilon > 0,\; \exists\, n \in \mathbb{N} \text{ such that } a_n \leq \varepsilon$.
    (c) Negate from outside in. The negation is

    $$
    \exists\, \varepsilon > 0 \text{ such that } \forall\, N \in \mathbb{N},\; \exists\, n > N \text{ with } |a_n - L| \geq \varepsilon
    $$

    This is exactly the statement "$a_n$ does **not** converge to $L$".

---

**Exercise 2.**
Let $A = \{1, 2, 3, 4, 5\}$, $B = \{3, 4, 5, 6, 7\}$, $\Omega = \{1, 2, 3, 4, 5, 6, 7, 8\}$.

**(a)** Compute $A \cap B$, $A \cup B$, $A \setminus B$, $A^c$, and $A \triangle B := (A \setminus B) \cup (B \setminus A)$.
**(b)** Verify De Morgan's law $(A \cup B)^c = A^c \cap B^c$.

??? success "Solution to Exercise 2"
    (a) $A \cap B = \{3, 4, 5\}$, $A \cup B = \{1, 2, 3, 4, 5, 6, 7\}$, $A \setminus B = \{1, 2\}$, $A^c = \{6, 7, 8\}$, $A \triangle B = \{1, 2, 6, 7\}$.

    (b) $(A \cup B)^c = \{8\}$, and $A^c \cap B^c = \{6, 7, 8\} \cap \{1, 2, 8\} = \{8\}$. Both equal $\{8\}$. $\square$

---

**Exercise 3.**
Prove De Morgan's law $(A \cup B)^c = A^c \cap B^c$ from first principles using the definitions of $\cup$, $\cap$, and complement.

??? success "Solution to Exercise 3"
    We show set equality by double inclusion.

    ($\subseteq$): Let $x \in (A \cup B)^c$. Then $x \notin A \cup B$, so $x \notin A$ **and** $x \notin B$. Hence $x \in A^c$ and $x \in B^c$, i.e. $x \in A^c \cap B^c$.

    ($\supseteq$): Let $x \in A^c \cap B^c$. Then $x \notin A$ and $x \notin B$, so $x$ is in neither set and therefore $x \notin A \cup B$, giving $x \in (A \cup B)^c$.

    Both inclusions hold, so the sets are equal. $\square$

---

**Exercise 4.**
Let $f: A \to B$. Prove that $f$ is injective if and only if for every pair of subsets $S_1, S_2 \subseteq A$,

$$
f(S_1 \cap S_2) = f(S_1) \cap f(S_2)
$$

??? success "Solution to Exercise 4"
    ($\Rightarrow$) Suppose $f$ is injective. The inclusion $f(S_1 \cap S_2) \subseteq f(S_1) \cap f(S_2)$ holds for any function (if $y = f(x)$ with $x \in S_1 \cap S_2$, then $y$ lies in both $f(S_1)$ and $f(S_2)$). For the reverse, let $y \in f(S_1) \cap f(S_2)$. Then $y = f(x_1) = f(x_2)$ for some $x_1 \in S_1$, $x_2 \in S_2$. By injectivity, $x_1 = x_2$, so $x_1 \in S_1 \cap S_2$ and $y \in f(S_1 \cap S_2)$.

    ($\Leftarrow$) Suppose the identity holds for all subsets. Pick any $x_1, x_2 \in A$ with $f(x_1) = f(x_2) = y$ and let $S_1 = \{x_1\}$, $S_2 = \{x_2\}$. Then $f(S_1) \cap f(S_2) = \{y\}$, so by hypothesis $f(S_1 \cap S_2) = \{y\}$, which is nonempty only if $S_1 \cap S_2 \ne \emptyset$, forcing $x_1 = x_2$. $\square$

---

**Exercise 5.**
Show that $\mathbb{Q}$, the set of rational numbers, is countable.

??? success "Solution to Exercise 5"
    It suffices to exhibit an injection from $\mathbb{Q}$ into $\mathbb{N}$ (then $\mathbb{Q}$ is at most countable, and it is clearly infinite). Every positive rational can be written uniquely as $p/q$ in lowest terms with $p, q \in \mathbb{N}$. Define

    $$
    \phi\!\left(\tfrac{p}{q}\right) = 2^p\, 3^q
    $$

    Unique prime factorization makes $\phi$ injective on $\mathbb{Q}_{>0}$. Composing with a bijection from $\mathbb{Q}$ to $\{0\} \cup \mathbb{Q}_{>0} \cup \mathbb{Q}_{<0}$ (e.g., interleave positive and negative rationals as $0, q_1, -q_1, q_2, -q_2, \ldots$) yields an injection $\mathbb{Q} \hookrightarrow \mathbb{N}$. $\square$

---

**Exercise 6.**
A common error in probability is to claim that "if $P(A \mid B) > P(A)$ then $A$ caused $B$." Formalize the **direction-of-inference** issue by computing $P(B \mid A)$ in terms of $P(A \mid B)$, $P(A)$, and $P(B)$, and explain in plain language why the original statement is unfounded.

??? success "Solution to Exercise 6"
    By Bayes' rule,

    $$
    P(B \mid A) = \frac{P(A \mid B)\, P(B)}{P(A)}
    $$

    The hypothesis $P(A \mid B) > P(A)$ is symmetric in $A$ and $B$ — multiplying both sides by $P(B)/P(A)$ shows $P(B \mid A) > P(B)$ as well. So "$A$ and $B$ are positively associated" tells us nothing about which (if either) causes the other. Causation is a statement about counterfactuals or interventions and cannot be inferred from conditional-probability statements alone (this is the formal counterpart of "correlation does not imply causation"). $\square$
