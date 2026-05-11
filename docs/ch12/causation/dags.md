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

## Exercises

**Exercise 1.**
Draw a DAG representing the following causal structure: smoking causes lung cancer, smoking causes yellow teeth, and lung cancer does not cause yellow teeth. Identify whether yellow teeth is a confounder, mediator, or collider in the relationship between smoking and lung cancer.

??? success "Solution to Exercise 1"
    The DAG is: Smoking $\to$ Lung Cancer, Smoking $\to$ Yellow Teeth. There is no arrow between Lung Cancer and Yellow Teeth.

    Yellow teeth is **neither** a confounder, mediator, nor collider for the smoking-cancer relationship. It is a separate effect of smoking (a "descendant" of smoking on a different causal path). Conditioning on yellow teeth is unnecessary and generally harmless, though it slightly reduces efficiency.

    A confounder would be a common cause of both smoking and lung cancer. A mediator would lie on the causal path between them. A collider would be a common effect of both.

---

**Exercise 2.**
In the DAG: $X \to Z \to Y$, explain why $Z$ is a mediator and what happens to the association between $X$ and $Y$ when you condition on $Z$.

??? success "Solution to Exercise 2"
    $Z$ is a **mediator** because it lies on the causal path from $X$ to $Y$. The effect of $X$ on $Y$ operates through $Z$: $X$ causes $Z$, which in turn causes $Y$.

    When you condition on $Z$ (e.g., include $Z$ as a covariate in regression), you block the causal path $X \to Z \to Y$. This means the association between $X$ and $Y$ disappears (or is reduced to only the direct effect if there is also a direct arrow $X \to Y$).

    This is important in practice: if you want to estimate the **total** causal effect of $X$ on $Y$, you should not condition on a mediator. If you want only the **direct** effect (not through $Z$), then conditioning on $Z$ is appropriate.

---

**Exercise 3.**
Consider the DAG: $X \to Z \leftarrow Y$. Explain why $Z$ is a collider and what paradoxical effect conditioning on $Z$ has.

??? success "Solution to Exercise 3"
    $Z$ is a **collider** because two arrows point into it from $X$ and $Y$. In this structure, $X$ and $Y$ are marginally independent (no causal connection between them).

    However, conditioning on the collider $Z$ creates a **spurious association** between $X$ and $Y$. This is called "collider bias" or "Berkson's paradox." Intuitively: if you know $Z$ occurred, then knowing $X$ was not the cause makes $Y$ more likely to be the cause (and vice versa), inducing a negative correlation.

    **Example:** Talent ($X$) and Beauty ($Y$) may be independent in the population, but among actors ($Z = $ became an actor, which requires either talent or beauty), they appear negatively correlated. Conditioning on $Z$ opened a path that was blocked unconditionally.

---

**Exercise 4.**
Given the DAG: $U \to X$, $U \to Y$, $X \to Y$, where $U$ is unobserved, explain the problem of confounding and state the back-door criterion for identifying the causal effect of $X$ on $Y$.

??? success "Solution to Exercise 4"
    The variable $U$ is an unobserved common cause (confounder) of $X$ and $Y$. The causal path $X \to Y$ gives the true causal effect, but the back-door path $X \leftarrow U \to Y$ creates a spurious association. Without adjusting for $U$, the observed association between $X$ and $Y$ conflates the causal effect with confounding.

    The **back-door criterion** (Pearl, 1993) states: a set of variables $\mathbf{Z}$ is sufficient for identifying the causal effect of $X$ on $Y$ if (1) $\mathbf{Z}$ blocks every back-door path from $X$ to $Y$, and (2) no variable in $\mathbf{Z}$ is a descendant of $X$.

    In this DAG, since $U$ is unobserved and no observed variable blocks the path $X \leftarrow U \to Y$, the causal effect of $X$ on $Y$ is **not identifiable** from observational data without additional assumptions (e.g., an instrumental variable).
