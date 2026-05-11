# Bayes' Theorem

## Overview

**Bayes' theorem** provides a systematic way to update probabilities when new evidence is observed. It reverses the direction of conditioning: given $P(B \mid A)$, it computes $P(A \mid B)$. This theorem is the foundation of Bayesian statistics and has widespread applications in medical diagnosis, spam filtering, machine learning, and finance.

---

## Statement

For events $A$ and $B$ with $P(B) > 0$:

$$
P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}
$$

Each term has a specific name:

$$
\underbrace{P(A \mid B)}_{\text{Posterior}} = \frac{\overbrace{P(B \mid A)}^{\text{Likelihood}} \cdot \overbrace{P(A)}^{\text{Prior}}}{\underbrace{P(B)}_{\text{Evidence}}}
$$

---

## Derivation

Starting from the definition of conditional probability:

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)} = \frac{P(B \mid A) \cdot P(A)}{P(B)}
$$

The denominator is often expanded using the law of total probability:

$$
P(B) = P(B \mid A) \cdot P(A) + P(B \mid A^c) \cdot P(A^c)
$$

This gives the expanded form:

$$
P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B \mid A) \cdot P(A) + P(B \mid A^c) \cdot P(A^c)}
$$

---

## General Form (Multiple Hypotheses)

If $A_1, A_2, \ldots, A_n$ partition the sample space $\Omega$:

$$
P(A_i \mid B) = \frac{P(B \mid A_i) \cdot P(A_i)}{\sum_{j=1}^{n} P(B \mid A_j) \cdot P(A_j)}
$$

---

## Examples

### Example: Medical Diagnosis

A disease affects 1% of a population. A test has 95% sensitivity and 90% specificity. If a person tests positive, what is the probability they have the disease?

$$
\begin{aligned}
P(\text{disease} \mid \text{positive}) &= \frac{P(\text{positive} \mid \text{disease}) \cdot P(\text{disease})}{P(\text{positive})} \\[6pt]
&= \frac{0.95 \times 0.01}{0.95 \times 0.01 + 0.10 \times 0.99} \\[6pt]
&= \frac{0.0095}{0.1085} \approx 0.0876
\end{aligned}
$$

Despite the seemingly accurate test, a positive result only yields an 8.76% probability of actually having the disease. This counterintuitive result arises because the disease is rare—most positives are false positives.

### Example: Drawing Balls from Urns

Two urns: Urn A has 3 red and 7 blue balls; Urn B has 8 red and 2 blue balls. An urn is chosen at random (50/50), and a red ball is drawn. What is the probability it came from Urn B?

$$
\begin{aligned}
P(B \mid \text{red}) &= \frac{P(\text{red} \mid B) \cdot P(B)}{P(\text{red} \mid A) \cdot P(A) + P(\text{red} \mid B) \cdot P(B)} \\[6pt]
&= \frac{0.8 \times 0.5}{0.3 \times 0.5 + 0.8 \times 0.5} = \frac{0.40}{0.55} \approx 0.727
\end{aligned}
$$

### Example: Spam Filtering

Suppose 40% of emails are spam. The word "free" appears in 80% of spam emails and 10% of non-spam emails. Given an email contains "free," what is the probability it is spam?

$$
P(\text{spam} \mid \text{"free"}) = \frac{0.80 \times 0.40}{0.80 \times 0.40 + 0.10 \times 0.60} = \frac{0.32}{0.38} \approx 0.842
$$

---

## Python Exploration

```python
import numpy as np

def bayes_theorem(prior, likelihood, evidence):
    """Apply Bayes' theorem."""
    posterior = (likelihood * prior) / evidence
    return posterior

# Medical diagnosis example
prior_disease = 0.01
sensitivity = 0.95
specificity = 0.90
false_positive_rate = 1 - specificity

p_positive = sensitivity * prior_disease + false_positive_rate * (1 - prior_disease)
posterior = bayes_theorem(prior_disease, sensitivity, p_positive)

print(f"P(disease | positive) = {posterior:.4f}")
print(f"Despite a 95% sensitive test, only {posterior*100:.1f}% of positives truly have the disease.")
```

```python
import numpy as np
import matplotlib.pyplot as plt

def bayes_update_visualization():
    """Visualize how the posterior changes with prevalence."""
    prevalences = np.linspace(0.001, 0.5, 200)
    sensitivity = 0.95
    specificity = 0.90

    posteriors = []
    for prev in prevalences:
        p_pos = sensitivity * prev + (1 - specificity) * (1 - prev)
        post = (sensitivity * prev) / p_pos
        posteriors.append(post)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(prevalences * 100, np.array(posteriors) * 100, lw=2)
    ax.set_xlabel('Prevalence (%)')
    ax.set_ylabel('P(Disease | Positive) (%)')
    ax.set_title("Bayes' Theorem: Posterior vs. Prevalence")
    ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% threshold')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()

bayes_update_visualization()
```

---

## Key Takeaways

- Bayes' theorem **reverses conditioning**: it computes $P(A \mid B)$ from $P(B \mid A)$.
- The **prior** reflects initial beliefs; the **posterior** reflects updated beliefs after observing evidence.
- Low base rates (rare events) can dominate: even with a good test, most positives may be false positives.
- Bayes' theorem is the foundation of Bayesian inference, where parameters are treated as random variables with prior distributions updated by data.

## Exercises

**Exercise 1.**
A test for a rare disease has prevalence $P(D) = 0.001$, sensitivity $P(+ \mid D) = 0.99$, specificity $P(- \mid D^c) = 0.95$. (a) Compute $P(D \mid +)$. (b) Explain why it is so low. (c) Repeat with prevalence 0.05.

??? success "Solution to Exercise 1"
    (a) $P(+) = 0.99 \cdot 0.001 + 0.05 \cdot 0.999 = 0.00099 + 0.04995 = 0.05094$. So $P(D \mid +) = 0.00099/0.05094 \approx 0.019$ (about 1.9%).

    (b) The disease is rare; the false-positive *rate* (5%) applied to the 999 healthy per 1000 produces $\approx 50$ false positives — vastly more than the $\approx 1$ true positive. The pool of positives is dominated by false positives.

    (c) With prevalence 0.05: $P(+) = 0.99 \cdot 0.05 + 0.05 \cdot 0.95 = 0.097$. $P(D \mid +) = 0.0495/0.097 \approx 0.510$. The posterior jumps from 2% to 51% — a stark demonstration of how strongly the prior governs the posterior, even when the likelihood ratio is identical.

---

**Exercise 2.**
A factory has 3 machines producing 50%, 30%, 20% of output with defect rates 2%, 3%, 5%. (a) Compute the overall defect rate. (b) Given a defective item, what's the probability it came from Machine 3?

??? success "Solution to Exercise 2"
    (a) $P(D) = 0.02 \cdot 0.50 + 0.03 \cdot 0.30 + 0.05 \cdot 0.20 = 0.010 + 0.009 + 0.010 = 0.029$. The overall defect rate is 2.9%.

    (b) $P(M_3 \mid D) = (0.05 \cdot 0.20)/0.029 = 0.010/0.029 \approx 0.345$. Machine 3 produces only 20% of output but contributes 34.5% of defects, because its defect rate is 2.5× the average.

---

**Exercise 3.**
**Prove Bayes' theorem** from the definition of conditional probability. Then state the **odds form**: $P(H \mid E)/P(H^c \mid E) = [P(E \mid H)/P(E \mid H^c)] \cdot [P(H)/P(H^c)]$.

??? success "Solution to Exercise 3"
    From conditional probability, $P(A \mid B) = P(A \cap B)/P(B)$ and $P(B \mid A) = P(A \cap B)/P(A)$. Therefore $P(A \cap B) = P(A \mid B) P(B) = P(B \mid A) P(A)$, giving

    $$
    P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}
    $$

    **Odds form:** divide Bayes for $H$ by Bayes for $H^c$ at the same $E$:

    $$
    \frac{P(H \mid E)}{P(H^c \mid E)} = \frac{P(E \mid H) P(H) / P(E)}{P(E \mid H^c) P(H^c) / P(E)} = \underbrace{\frac{P(E \mid H)}{P(E \mid H^c)}}_{\text{likelihood ratio}} \cdot \underbrace{\frac{P(H)}{P(H^c)}}_{\text{prior odds}}
    $$

    **Posterior odds = likelihood ratio × prior odds.** This form avoids computing $P(E)$ and is the workhorse of Bayesian inference, evidence assessment in courts, and medical decision-making.

---

**Exercise 4.**
A coin is **either** fair ($P = 0.5$) **or** biased ($P = 0.7$ for heads). Equal prior. You observe 8 heads in 10 flips. Compute the posterior probability the coin is biased.

??? success "Solution to Exercise 4"
    Let $B$ = biased, $F$ = fair, $E$ = observe 8 heads in 10 flips.

    Likelihoods: $P(E \mid F) = \binom{10}{8} 0.5^8 \cdot 0.5^2 = 45 \cdot (0.5)^{10} \approx 0.0439$.

    $P(E \mid B) = \binom{10}{8} 0.7^8 \cdot 0.3^2 = 45 \cdot 0.0576 \cdot 0.09 \approx 0.2335$.

    By Bayes (or odds form):

    $$
    P(B \mid E) = \frac{P(E \mid B) P(B)}{P(E \mid B) P(B) + P(E \mid F) P(F)} = \frac{0.2335 \cdot 0.5}{0.2335 \cdot 0.5 + 0.0439 \cdot 0.5} = \frac{0.2335}{0.2774} \approx 0.842
    $$

    Roughly 84% posterior probability the coin is biased.

    Note the odds form gives the same result directly: prior odds 1:1, likelihood ratio $0.2335/0.0439 \approx 5.32$, posterior odds 5.32:1, posterior probability $5.32/(5.32 + 1) \approx 0.842$.

---

**Exercise 5.**
**Sequential updating.** A new piece of evidence $E_2$ arrives after $E_1$. Show that the Bayesian update after observing $E_1$ and then $E_2$ (when treated as conditionally independent given the hypothesis) is mathematically equivalent to updating once with their joint likelihood.

??? success "Solution to Exercise 5"
    Let posterior after $E_1$: $P(H \mid E_1) \propto P(E_1 \mid H) P(H)$.

    After observing $E_2$, treat the post-$E_1$ posterior as the new prior:

    $$
    P(H \mid E_1, E_2) \propto P(E_2 \mid H, E_1) P(H \mid E_1)
    $$

    Assuming conditional independence $P(E_2 \mid H, E_1) = P(E_2 \mid H)$:

    $$
    P(H \mid E_1, E_2) \propto P(E_2 \mid H) P(E_1 \mid H) P(H) = P(E_1, E_2 \mid H) P(H)
    $$

    which is exactly Bayes applied to the joint likelihood. So sequential and batch updating give the same posterior — the Bayesian framework is internally consistent under conditional independence.

    This justifies online updating: process data one observation at a time, no need to recompute from scratch. The post-update posterior is sufficient information.

---

**Exercise 6.**
**Base rate neglect.** The **defendant's fallacy** states: "The prosecutor's DNA matched the defendant's; the chance of a random match is 1 in a million. So the defendant is guilty beyond reasonable doubt." Why is this argument flawed, and how does Bayes' theorem expose the error?

??? success "Solution to Exercise 6"
    The argument confuses $P(\text{match} \mid \text{innocent})$ (1 in 1 million) with $P(\text{innocent} \mid \text{match})$ — the inverse conditional that actually matters.

    **Bayes' theorem reveals the gap.** Let $G$ = guilty, $M$ = DNA match. Suppose the suspect was identified from a database of $10^6$ people (or, equivalently, the prior is that the defendant is one of $10^6$ candidates with prior $P(G) = 10^{-6}$). Then $P(M \mid G^c) = 10^{-6}$, $P(M \mid G) \approx 1$. By Bayes,

    $$
    P(G \mid M) = \frac{P(M \mid G) P(G)}{P(M \mid G) P(G) + P(M \mid G^c) P(G^c)} = \frac{1 \cdot 10^{-6}}{1 \cdot 10^{-6} + 10^{-6} \cdot (1 - 10^{-6})} \approx 0.5
    $$

    Only about 50% probability of guilt — far from "beyond reasonable doubt." The 1-in-a-million figure ignores the prior odds; once the prior odds are incorporated, the posterior is much weaker. Additional evidence beyond the DNA match is needed to push the posterior to a confident verdict.

    This is the same error pattern as the rare-disease example: forgetting the base rate. Real-world juries and policy-makers commit this error routinely. The Bayesian framework provides the correction.
