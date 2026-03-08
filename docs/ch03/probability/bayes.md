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
