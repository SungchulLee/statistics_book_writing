# Directed Acyclic Graphs

Throughout this chapter, we have encountered confounders, mediators, colliders, and lurking variables. **Directed acyclic graphs** (DAGs) provide a visual and mathematical framework for representing causal relationships among variables, making it possible to determine which variables to control for and which to leave alone. DAGs are the foundation of the modern graphical approach to causal inference, developed primarily by Judea Pearl.

---

## What Is a DAG

A **directed acyclic graph** is a graph consisting of:

- **Nodes** (vertices): each node represents a variable.
- **Directed edges** (arrows): an arrow from $A$ to $B$ means that $A$ is a direct cause of $B$.
- **Acyclic**: there are no directed cycles (you cannot follow the arrows from a node back to itself).

The absence of an arrow between two nodes means there is no direct causal effect between them (though there may be an indirect effect through other nodes).

---

## Three Fundamental Structures

Every path in a DAG is composed of three elementary building blocks. Understanding these three structures is sufficient to determine the statistical implications of any DAG.

### 1. Chain (Mediation)

$$
A \rightarrow B \rightarrow C
$$

$B$ is a **mediator** on the causal path from $A$ to $C$. Information flows from $A$ to $C$ through $B$.

- **Unconditional**: $A$ and $C$ are associated.
- **Conditioning on $B$**: blocks the path. $A$ and $C$ become conditionally independent given $B$.

### 2. Fork (Common Cause)

$$
A \leftarrow B \rightarrow C
$$

$B$ is a **common cause** (confounder) of $A$ and $C$. Information flows between $A$ and $C$ through their shared cause $B$.

- **Unconditional**: $A$ and $C$ are associated.
- **Conditioning on $B$**: blocks the path. $A$ and $C$ become conditionally independent given $B$.

### 3. Collider (Common Effect)

$$
A \rightarrow B \leftarrow C
$$

$B$ is a **collider** -- it is caused by both $A$ and $C$.

- **Unconditional**: $A$ and $C$ are **not** associated (the path is blocked).
- **Conditioning on $B$**: **opens** the path, creating a spurious association between $A$ and $C$.

!!! warning "Conditioning on a collider creates bias"
    Unlike chains and forks, where conditioning blocks the path, conditioning on a collider opens a previously blocked path. This is a frequent source of selection bias and is the reason why indiscriminate inclusion of variables in a regression can make estimates worse, not better.

---

## d-Separation

The concept of **d-separation** (directional separation) generalizes the three fundamental structures to determine whether two variables are conditionally independent given a set of conditioning variables.

Two nodes $X$ and $Y$ are **d-separated** by a set $S$ if, for every undirected path between $X$ and $Y$:

1. The path contains a chain $A \rightarrow B \rightarrow C$ or a fork $A \leftarrow B \rightarrow C$ where $B \in S$ (the path is blocked by conditioning on a non-collider), **or**

2. The path contains a collider $A \rightarrow B \leftarrow C$ where $B \notin S$ and no descendant of $B$ is in $S$ (the path is blocked because the collider is not conditioned on).

If $X$ and $Y$ are d-separated by $S$, then $X$ and $Y$ are conditionally independent given $S$ in any probability distribution compatible with the DAG:

$$
X \perp\!\!\!\perp Y \mid S
$$

---

## The Backdoor Criterion

The **backdoor criterion** provides a practical rule for identifying which variables to control for when estimating the causal effect of $X$ on $Y$.

A set of variables $S$ satisfies the backdoor criterion relative to $(X, Y)$ if:

1. No variable in $S$ is a descendant of $X$.
2. $S$ blocks every path between $X$ and $Y$ that contains an arrow into $X$ (a "backdoor path").

If such a set $S$ exists, the causal effect of $X$ on $Y$ is identified by the **adjustment formula**:

$$
P(Y \mid \text{do}(X = x)) = \sum_s P(Y \mid X = x, S = s) \, P(S = s)
$$

The $\text{do}(\cdot)$ notation, introduced by Pearl, distinguishes interventional distributions (what happens when we set $X$ to a value) from observational distributions (what we observe when $X$ takes a value).

---

## Example: Applying the Backdoor Criterion

Consider the DAG:

$$
Z \rightarrow X \rightarrow Y, \quad Z \rightarrow Y
$$

Here $Z$ is a confounder (it has a backdoor path $X \leftarrow Z \rightarrow Y$). To estimate the causal effect of $X$ on $Y$:

- **Control for $Z$**: $S = \{Z\}$ satisfies the backdoor criterion because $Z$ is not a descendant of $X$ and it blocks the backdoor path.
- **Do not control for nothing**: $S = \emptyset$ fails because the backdoor path $X \leftarrow Z \rightarrow Y$ remains open.

Now consider:

$$
X \rightarrow M \rightarrow Y, \quad X \rightarrow Y
$$

Here $M$ is a mediator. To estimate the **total** causal effect of $X$ on $Y$:

- **Do not control for $M$**: there are no backdoor paths (no arrows into $X$), so $S = \emptyset$ suffices.
- **Controlling for $M$** would block the indirect path and estimate only the direct effect, not the total effect.

---

## Practical Steps for Using DAGs

1. **Draw the DAG.** List all relevant variables and draw arrows based on subject-matter knowledge about causal relationships. This step requires domain expertise, not statistical analysis.

2. **Identify the causal question.** Which effect are you estimating? The total effect of $X$ on $Y$? The direct effect?

3. **Find all backdoor paths.** List all paths from $X$ to $Y$ that have an arrow pointing into $X$.

4. **Apply the backdoor criterion.** Find a set $S$ that blocks all backdoor paths without including descendants of $X$ (for total effects) or colliders.

5. **Adjust for $S$ in the analysis.** Include the variables in $S$ as covariates in a regression, stratification, or matching procedure.

---

## Common Mistakes

| Mistake | Why it is wrong |
|:---|:---|
| Controlling for a mediator when estimating the total effect | Blocks part of the causal path |
| Controlling for a collider | Opens a spurious path |
| Controlling for a descendant of a collider | Also opens the collider path |
| Including every available variable in the regression | May introduce collider bias or block causal paths |

The principle is that more control variables are not always better. The DAG tells you which variables to include and which to exclude.

---

## Summary

Directed acyclic graphs (DAGs) represent causal relationships using nodes (variables) and directed edges (causal arrows). The three fundamental structures -- chains, forks, and colliders -- determine how conditioning on a variable affects the flow of information between other variables. The d-separation criterion generalizes these rules to complex graphs, and the backdoor criterion identifies which variables to control for when estimating causal effects. DAGs provide a principled framework for avoiding common errors such as conditioning on colliders or blocking causal pathways, and they are essential tools for any researcher working with observational data.
