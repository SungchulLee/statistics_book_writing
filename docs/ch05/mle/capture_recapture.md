# Capture-Recapture Method

> **Reference:** [Wikipedia — Mark and Recapture](https://en.wikipedia.org/wiki/Mark_and_recapture)

## Overview

Mark and recapture is a method commonly used in ecology to estimate an animal population's size where it is impractical to count every individual. A portion of the population is captured, marked, and released. Later, another portion will be captured and the number of marked individuals within the sample is counted. Since the number of marked individuals within the second sample should be proportional to the number of marked individuals in the whole population, an estimate of the total population size can be obtained by dividing the number of marked individuals by the proportion of marked individuals in the second sample.

Other names for this method include capture-recapture, capture-mark-recapture, mark-release-recapture, multiple systems estimation, band recovery, the Petersen method, and the Lincoln method.

## Steps of the Capture-Recapture Method

1. **Capture and Mark**: A random sample of individuals from the population is captured. These individuals are marked in a way that allows them to be identified if recaptured (e.g., tags, bands, or harmless dyes). Let the number of marked individuals be denoted as $M$.

2. **Release**: The marked individuals are released back into the population and allowed to mix thoroughly.

3. **Recapture**: A second random sample is taken from the population. Some of the individuals in this sample will already be marked. Let the total number of individuals in the second sample be $n$, and the number of marked individuals in the second sample be $m$.

4. **Estimation**: Using the assumption that the proportion of marked individuals in the second sample represents the proportion of the entire population that was marked in the first sample, the population size $N$ can be estimated:
$$
\hat{N} = \frac{M \cdot n}{m}
$$

## Assumptions

1. **Closed Population**: The population size remains constant during the study (no births, deaths, immigration, or emigration).
2. **Equal Capture Probability**: Every individual has an equal chance of being captured in each sampling event.
3. **Marking Doesn't Affect Behavior**: The marking process does not influence the likelihood of being recaptured.
4. **Marks are Durable and Visible**: Marks are not lost, and they remain identifiable throughout the study.

## Simple Example

Imagine a pond where we want to estimate the fish population:

1. **First Capture**: 50 fish are caught, marked, and released ($M = 50$).
2. **Second Capture**: 40 fish are caught ($n = 40$), and 10 of them are found to be marked ($m = 10$).
3. **Estimation**:

$$
\hat{N} = \frac{M \cdot n}{m} = \frac{50 \cdot 40}{10} = 200
$$

The estimated fish population in the pond is 200.

## MLE Derivation

### Hypergeometric Distribution

The probability of observing $m$ marked individuals in a recapture sample of size $n$, given the population size $N$, is:

$$
P(m \mid N) = \frac{\binom{M}{m} \binom{N-M}{n-m}}{\binom{N}{n}}
$$

where:

- $\binom{M}{m}$: Ways to select $m$ marked individuals from $M$.
- $\binom{N-M}{n-m}$: Ways to select $n-m$ unmarked individuals from the remaining $N-M$.
- $\binom{N}{n}$: Total ways to select $n$ individuals from $N$.

### Likelihood Function

The likelihood function $L(N)$ is proportional to this probability:

$$
L(N) = P(m \mid N) \propto \frac{\binom{M}{m} \binom{N-M}{n-m}}{\binom{N}{n}}
$$

where $M$, $n$, and $m$ are known from the experiment, and $N$ is the parameter to be estimated.

### Simplified Likelihood

Ignoring constants that do not depend on $N$, the likelihood becomes:

$$
L(N) \propto \frac{(N-M)!}{(N-n)!(N)!}
$$

### Maximizing the Likelihood

Taking the natural logarithm of the likelihood $\ell(N) = \log L(N)$ and differentiating with respect to $N$, we get the MLE of $N$:

$$
\hat{N} = \frac{M \cdot n}{m}
$$

### Intuition Behind the MLE

The estimate $\hat{N}$ is based on the idea that the proportion of marked individuals in the recaptured sample ($m/n$) reflects the proportion of marked individuals in the entire population ($M/N$):

$$
\frac{m}{n} \approx \frac{M}{N}
\quad\Longrightarrow\quad
\hat{N} = \frac{M \cdot n}{m}
$$

## Properties of the MLE

### Bias

For small sample sizes, $\hat{N}$ can be slightly biased. The **Chapman estimator** provides a bias-corrected alternative:

$$
\hat{N}_{\text{Chapman}} = \frac{(M+1)(n+1)}{m+1} - 1
$$

### Variance

The variance of the MLE can be approximated as:

$$
\text{Var}(\hat{N}) \approx \frac{M^2 \cdot n \cdot (n-m)}{m^3}
$$

## Extensions and Variations

- **Multiple Recapture Events**: Involves capturing and marking over multiple events to refine estimates.
- **Open Populations**: Extensions like the Jolly-Seber model can handle populations where individuals may enter or leave the system.
- **Unequal Capture Probability**: Models like the Lincoln-Petersen estimator or logistic regression can adjust for variation in capture probability.

## Python Implementation

```python
import matplotlib.pyplot as plt
from scipy import special

def prob(n, c, r, t):
    """
    Calculate the probability of capturing 't' tagged birds in a recapture
    sample of size 'r', given that there are 'n' birds in total.

    Parameters:
    - n: Total number of birds in the population
    - c: Number of birds captured and tagged in the first stage
    - r: Number of birds recaptured in the second stage
    - t: Number of tagged birds in the recapture stage

    Returns:
    - Probability of observing 't' tagged birds in the recapture sample.
    """
    return special.comb(n - c, r - t) * special.comb(c, t) / special.comb(n, r)

def capture_recapture(c=10, r=10, t=3):
    """
    Calculate the probability distribution over possible total population sizes
    and determine the MLE (Maximum Likelihood Estimate) for the population size.

    Parameters:
    - c: Number of birds captured and tagged in the first stage
    - r: Number of birds recaptured in the second stage
    - t: Number of tagged birds in the recapture stage

    Returns:
    - prob_list: List of probabilities for each population size
    - mle_n: MLE for the total population size
    """
    prob_list = []

    # Calculate probability for each possible population size n
    for n in range(c + r - t, 10 * (c + r - t)):
        prob_list.append(prob(n, c, r, t))

    # Determine the MLE for the population size
    prob_max = max(prob_list)
    idx = prob_list.index(prob_max)
    mle_n = idx + (c + r - t)
    print(f'MLE n: {mle_n}')

    return prob_list, mle_n

def draw(prob_list, mle_n, c=10, r=10, t=3):
    """
    Plot the probability distribution of the total population size
    and highlight the MLE.

    Parameters:
    - prob_list: List of probabilities for each population size
    - mle_n: MLE for the total population size
    - c, r, t: Parameters for the capture-recapture model
    """
    idx = mle_n - (c + r - t)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(range(c + r - t, 10 * (c + r - t)), prob_list, label='Probability')
    ax.plot([mle_n, mle_n], [0, prob_list[idx]], 'o--r', label=f'MLE: {mle_n}')

    # Customize plot
    ax.set_xlabel('Total Population Size (n)')
    ax.set_ylabel('Probability')
    ax.set_title('Capture-Recapture MLE for Population Size')
    ax.legend()
    plt.show()

# Parameters for capture-recapture model
c = 5   # Birds captured and tagged in the first stage
r = 6   # Birds recaptured in the second stage
t = 2   # Tagged birds in the recapture stage

# Calculate probabilities and MLE
prob_list, mle_n = capture_recapture(c, r, t)

# Plot the probability distribution and highlight the MLE
draw(prob_list, mle_n, c, r, t)
```
