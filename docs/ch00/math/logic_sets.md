# Sets, Functions, and Logic

This section reviews the foundational language of mathematics used throughout the book. Precise definitions here prevent ambiguity in later chapters on probability, random variables, and statistical inference.

## Propositional Logic

### Statements and Connectives

A **proposition** (or **statement**) is a declarative sentence that is either true or false.

| Symbol | Name | Read as | Example |
|---|---|---|---|
| $\neg P$ | Negation | "not $P$" | $\neg$(it is raining) |
| $P \land Q$ | Conjunction | "$P$ and $Q$" | It is raining **and** cold |
| $P \lor Q$ | Disjunction | "$P$ or $Q$" (inclusive) | It is raining **or** cold |
| $P \Rightarrow Q$ | Implication | "if $P$ then $Q$" | If it rains, the ground is wet |
| $P \Leftrightarrow Q$ | Biconditional | "$P$ if and only if $Q$" | Triangle is equilateral **iff** all angles are 60° |

### Truth Tables

| $P$ | $Q$ | $P \land Q$ | $P \lor Q$ | $P \Rightarrow Q$ | $P \Leftrightarrow Q$ |
|---|---|---|---|---|---|
| T | T | T | T | T | T |
| T | F | F | T | F | F |
| F | T | F | T | T | F |
| F | F | F | F | T | T |

!!! note "Vacuous Truth"
    When $P$ is false, $P \Rightarrow Q$ is true regardless of $Q$. This convention is essential in probability when conditioning on events of measure zero.

### Quantifiers

- **Universal quantifier**: $\forall\, x \in S,\; P(x)$ — "$P(x)$ holds for every $x$ in $S$."
- **Existential quantifier**: $\exists\, x \in S \text{ such that } P(x)$ — "there exists at least one $x$ in $S$ for which $P(x)$ holds."

**Negation of quantifiers:**

$$\neg\bigl(\forall\, x,\; P(x)\bigr) \;\Leftrightarrow\; \exists\, x \text{ s.t. } \neg P(x)$$

$$\neg\bigl(\exists\, x,\; P(x)\bigr) \;\Leftrightarrow\; \forall\, x,\; \neg P(x)$$

## Sets

### Definitions and Notation

A **set** is an unordered collection of distinct objects called **elements** (or **members**).

| Notation | Meaning |
|---|---|
| $x \in A$ | $x$ is an element of $A$ |
| $x \notin A$ | $x$ is not an element of $A$ |
| $\emptyset$ | The empty set (contains no elements) |
| $\{1, 2, 3\}$ | Roster notation |
| $\{x \in \mathbb{R} : x > 0\}$ | Set-builder notation |

### Standard Number Sets

| Symbol | Name | Description |
|---|---|---|
| $\mathbb{N}$ | Natural numbers | $\{0, 1, 2, 3, \dots\}$ (or $\{1,2,3,\dots\}$ by convention) |
| $\mathbb{Z}$ | Integers | $\{\dots, -2, -1, 0, 1, 2, \dots\}$ |
| $\mathbb{Q}$ | Rational numbers | $\{p/q : p \in \mathbb{Z},\, q \in \mathbb{Z} \setminus \{0\}\}$ |
| $\mathbb{R}$ | Real numbers | The complete ordered field |
| $\mathbb{R}^n$ | $n$-dimensional Euclidean space | Vectors $(x_1, \dots, x_n)$ |

### Subsets and Equality

- $A \subseteq B$ — $A$ is a **subset** of $B$: every element of $A$ belongs to $B$.
- $A \subset B$ — $A$ is a **proper subset** of $B$: $A \subseteq B$ and $A \neq B$.
- $A = B$ iff $A \subseteq B$ and $B \subseteq A$.

### Set Operations

| Operation | Notation | Definition |
|---|---|---|
| Union | $A \cup B$ | $\{x : x \in A \text{ or } x \in B\}$ |
| Intersection | $A \cap B$ | $\{x : x \in A \text{ and } x \in B\}$ |
| Difference | $A \setminus B$ | $\{x : x \in A \text{ and } x \notin B\}$ |
| Complement | $A^c$ | $\{x \in \Omega : x \notin A\}$ (relative to a universal set $\Omega$) |
| Cartesian product | $A \times B$ | $\{(a, b) : a \in A,\, b \in B\}$ |
| Power set | $\mathcal{P}(A)$ | The set of all subsets of $A$ |

### De Morgan's Laws

$$
(A \cup B)^c = A^c \cap B^c, \qquad (A \cap B)^c = A^c \cup B^c
$$

These laws generalize to arbitrary (even uncountable) collections of sets and are used frequently in probability theory when manipulating events.

### Countability

- A set is **finite** if it has a finite number of elements.
- A set is **countably infinite** if its elements can be put in one-to-one correspondence with $\mathbb{N}$ (e.g., $\mathbb{Z}$, $\mathbb{Q}$).
- A set is **uncountable** if no such correspondence exists (e.g., $\mathbb{R}$, any interval $[a, b]$ with $a < b$).

The distinction between countable and uncountable sets is critical when defining probability measures: discrete distributions sum over countable sets, while continuous distributions integrate over uncountable ones.

## Functions

### Definition

A **function** $f: A \to B$ is a rule that assigns to each element $x \in A$ exactly one element $f(x) \in B$.

- $A$ is the **domain**.
- $B$ is the **codomain**.
- The **range** (or **image**) is $\{f(x) : x \in A\} \subseteq B$.

### Types of Functions

| Property | Definition | Example |
|---|---|---|
| **Injective** (one-to-one) | $f(x_1) = f(x_2) \Rightarrow x_1 = x_2$ | $f(x) = 2x$ |
| **Surjective** (onto) | For every $y \in B$, there exists $x \in A$ with $f(x) = y$ | $f: \mathbb{R} \to \mathbb{R},\; f(x) = x^3$ |
| **Bijective** | Both injective and surjective | $f: \mathbb{R} \to \mathbb{R},\; f(x) = 2x + 1$ |

A bijection has an **inverse** $f^{-1}: B \to A$ satisfying $f^{-1}(f(x)) = x$ for all $x \in A$.

### Indicator Functions

The **indicator function** of a set $A$ is defined as

$$
\mathbf{1}_A(x) = \begin{cases} 1 & \text{if } x \in A, \\ 0 & \text{if } x \notin A. \end{cases}
$$

Indicator functions appear throughout probability and statistics—for instance, in defining Bernoulli random variables and in expressing likelihoods.

### Composition

Given $f: A \to B$ and $g: B \to C$, the **composition** $g \circ f: A \to C$ is defined by $(g \circ f)(x) = g(f(x))$.

### Important Function Classes

The following function types are referenced frequently in this book:

- **Linear**: $f(x) = ax + b$ — regression models.
- **Polynomial**: $f(x) = a_n x^n + \cdots + a_1 x + a_0$ — Taylor approximations.
- **Exponential**: $f(x) = e^x$ — moment generating functions, growth/decay.
- **Logarithmic**: $f(x) = \ln x$ — log-likelihoods, entropy.
- **Logistic (sigmoid)**: $\sigma(x) = \dfrac{1}{1 + e^{-x}}$ — logistic regression.

## Proof Techniques (Reference)

Statistical theorems rely on a small set of proof strategies. The following are encountered most frequently in this book:

- **Direct proof**: Assume premises, derive conclusion through logical steps.
- **Proof by contradiction**: Assume the negation of the conclusion and derive a contradiction.
- **Proof by induction**: Prove a base case and an inductive step ($P(n) \Rightarrow P(n+1)$).
- **Proof by contrapositive**: To prove $P \Rightarrow Q$, prove $\neg Q \Rightarrow \neg P$.

## Summary

| Concept | Why It Matters |
|---|---|
| Logic and quantifiers | Precisely state theorems, hypotheses, and conditions |
| Sets and operations | Define sample spaces, events, and parameter spaces |
| De Morgan's laws | Manipulate complements of unions/intersections of events |
| Countability | Distinguish discrete (summation) from continuous (integration) settings |
| Functions | Model relationships; indicator, exponential, and logistic functions are ubiquitous |
| Bijections and inverses | Underpin transformations of random variables |
