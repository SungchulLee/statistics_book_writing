# Reinforcement Learning (Sequential Decisions)

## Overview

In **reinforcement learning (RL)**, an **agent** learns to make decisions by interacting with an **environment**. The agent receives **rewards** or **penalties** based on its actions and adjusts its behavior over time to maximize cumulative rewards. Unlike supervised learning, there are no labeled input–output pairs; the agent must discover which actions lead to the best outcomes through trial and error.

## Key Characteristics

- **Agent–environment interaction**: The agent takes actions, the environment responds with a new state and a reward signal.
- **Sequential decision-making**: Actions affect not only immediate rewards but also future states and future rewards.
- **Exploration vs. exploitation**: The agent must balance trying new actions (exploration) with leveraging what it already knows works (exploitation).
- **Delayed rewards**: The consequences of an action may not be immediately apparent; the agent must learn to associate current actions with future outcomes.

## The RL Framework

At each time step $t$, the agent:

1. Observes the current **state** $s_t$.
2. Chooses an **action** $a_t$ according to its **policy** $\pi$.
3. Receives a **reward** $r_t$ and transitions to a new state $s_{t+1}$.

The agent's goal is to learn a policy $\pi^*$ that maximizes the expected cumulative (discounted) reward:

$$
\pi^* = \arg\max_\pi \; E\left[\sum_{t=0}^{\infty} \gamma^t \, r_t \right]
$$

where $\gamma \in [0, 1)$ is the **discount factor** that controls how much the agent values future rewards relative to immediate ones.

```
    ┌─────────┐
    │  Agent  │
    │  (π)    │
    └──┬───▲──┘
 action│   │ state, reward
       │   │
    ┌──▼───┴──┐
    │Environment│
    └─────────┘
```

## Comparison with Other Paradigms

| Aspect | Supervised Learning | Unsupervised Learning | Reinforcement Learning |
|---|---|---|---|
| **Feedback** | Correct label for each input | No labels | Reward signal (delayed, scalar) |
| **Goal** | Learn input→output mapping | Discover structure | Maximize cumulative reward |
| **Data** | Fixed dataset | Fixed dataset | Generated through interaction |
| **Temporal aspect** | Usually i.i.d. samples | Usually i.i.d. samples | Sequential, non-i.i.d. |

## Examples

**Game playing:** Teaching an agent to play chess, Go, or Atari games. AlphaGo famously learned to defeat the world champion in Go through self-play reinforcement learning.

**Robotics:** Training a robot to navigate a maze by rewarding it for getting closer to the exit and penalizing it for hitting walls.

**Autonomous driving:** An RL agent learns to control a vehicle by receiving rewards for safe driving and penalties for collisions or traffic violations.

**Finance applications:**

- **Portfolio management:** An agent allocates capital across assets, receiving rewards proportional to risk-adjusted returns.
- **Order execution:** An agent learns to split a large order into smaller trades to minimize market impact.
- **Market making:** An agent sets bid and ask prices to maximize profit while managing inventory risk.

## Simple Example: Multi-Armed Bandit

```python
import numpy as np

np.random.seed(42)

# 5-armed bandit: each arm has a different true mean reward
true_means = [1.0, 1.5, 2.0, 1.2, 0.8]
n_arms = len(true_means)
n_steps = 1000
epsilon = 0.1  # exploration rate

# Epsilon-greedy strategy
Q = np.zeros(n_arms)       # estimated value of each arm
N = np.zeros(n_arms)       # number of times each arm was pulled
rewards = []

for t in range(n_steps):
    if np.random.rand() < epsilon:
        action = np.random.randint(n_arms)  # explore
    else:
        action = np.argmax(Q)               # exploit

    reward = np.random.normal(true_means[action], 1.0)
    N[action] += 1
    Q[action] += (reward - Q[action]) / N[action]  # incremental mean update
    rewards.append(reward)

print("Estimated values:", np.round(Q, 2))
print("True means:      ", true_means)
print(f"Average reward:   {np.mean(rewards):.2f}")
print(f"Best arm chosen:  {np.argmax(N)} (pulled {int(N[np.argmax(N)])} times)")
```

## Key Takeaways

- Reinforcement learning is designed for **sequential decision-making** problems where an agent learns from interaction with an environment.
- The agent balances **exploration** (trying new actions) and **exploitation** (using known good actions).
- RL has achieved remarkable results in games, robotics, and is increasingly applied in quantitative finance for portfolio optimization, execution, and trading.
- Unlike supervised and unsupervised learning, RL generates its own training data through interaction, making it suitable for dynamic, evolving environments.

## Exercises

**Exercise 1.**
An agent uses an epsilon-greedy strategy with $\epsilon = 0.1$ and has estimated action values $Q = [2.0, 3.5, 1.0, 4.0]$ for a 4-armed bandit. On the next step, what is the probability that the agent selects arm 4 (the greedy choice)? What is the probability of selecting arm 2?

??? success "Solution to Exercise 1"
    With probability $1 - \epsilon = 0.9$ the agent selects the greedy arm (the one with the highest $Q$-value, which is arm 4 with $Q = 4.0$). With probability $\epsilon = 0.1$ the agent explores uniformly at random among all 4 arms.

    The probability of selecting arm 4 is:

    $$
    P(\text{arm 4}) = (1 - \epsilon) + \frac{\epsilon}{4} = 0.9 + 0.025 = 0.925
    $$

    The probability of selecting arm 2 (which is not the greedy choice) is:

    $$
    P(\text{arm 2}) = \frac{\epsilon}{4} = \frac{0.1}{4} = 0.025
    $$

---

**Exercise 2.**
Explain the exploration-exploitation tradeoff in the context of a restaurant recommendation system. Give one concrete scenario where pure exploitation fails and one where pure exploration is wasteful.

??? success "Solution to Exercise 2"
    In a restaurant recommendation system, **exploitation** means always recommending the restaurant the user has rated highest so far, while **exploration** means recommending a restaurant the user has not tried or has limited experience with.

    **Pure exploitation fails** when the user has only tried a few restaurants early on and rated a mediocre one highest by chance. The system would keep recommending that mediocre restaurant forever, never discovering a much better option nearby.

    **Pure exploration is wasteful** because the system would keep recommending random untried restaurants even after the user has clearly established preferences. If the user loves Italian food and has given high ratings to several Italian restaurants, pure exploration would still send them to random cuisines, leading to many unsatisfying recommendations.

    An effective system must balance the two: mostly recommending known good options while occasionally suggesting something new to refine its understanding of the user's preferences.

---

**Exercise 3.**
In the discounted reward formulation, the agent maximizes $\sum_{t=0}^{\infty} \gamma^t r_t$. If the discount factor is $\gamma = 0.9$ and the agent receives a constant reward of $r = 1$ at every time step, what is the total discounted return? What happens as $\gamma \to 1$?

??? success "Solution to Exercise 3"
    The total discounted return is a geometric series:

    $$
    \sum_{t=0}^{\infty} \gamma^t \cdot 1 = \frac{1}{1 - \gamma} = \frac{1}{1 - 0.9} = 10
    $$

    As $\gamma \to 1$, the sum $\frac{1}{1-\gamma} \to \infty$. This means the agent values future rewards almost as much as immediate rewards, and the total return diverges. In practice, $\gamma < 1$ is required for the infinite-horizon formulation to be well-defined (finite total return). A higher $\gamma$ makes the agent more "far-sighted," while a lower $\gamma$ makes it prioritize short-term rewards.

---

**Exercise 4.**
Compare reinforcement learning, supervised learning, and unsupervised learning across the following dimensions: (a) the type of feedback available, (b) the role of temporal ordering in the data, and (c) the goal of the learning algorithm.

??? success "Solution to Exercise 4"
    **(a) Type of feedback:**

    - **Supervised learning** receives explicit correct labels for each input (e.g., "this image is a cat").
    - **Unsupervised learning** receives no feedback at all — only the raw data.
    - **Reinforcement learning** receives a scalar reward signal that is often delayed and does not directly indicate the correct action.

    **(b) Role of temporal ordering:**

    - **Supervised and unsupervised learning** typically assume data points are independent and identically distributed (i.i.d.) — the order does not matter.
    - **Reinforcement learning** is inherently sequential: the agent's current action affects future states and rewards, creating temporal dependencies.

    **(c) Goal:**

    - **Supervised learning** aims to learn a mapping from inputs to outputs that generalizes to unseen data.
    - **Unsupervised learning** aims to discover hidden structure (clusters, latent factors, density) in the data.
    - **Reinforcement learning** aims to learn a policy that maximizes cumulative reward over time through interaction with the environment.

---

**Exercise 5.**
The **multi-armed bandit** is a special case of RL where the state never changes. State the regret of an algorithm that always picks the empirically best arm after one initial pull of each (no further exploration). When does this fail?

??? success "Solution to Exercise 5"
    "Pick the empirically best arm after one pull each" almost surely fails. After one trial each, the empirical mean of each arm has variance comparable to the reward noise itself; with positive probability, a suboptimal arm has the highest sample mean. The algorithm then commits to that arm forever, accumulating linear regret $T \cdot (\mu^* - \mu_{\text{chosen}})$ where $T$ is the horizon.

    A correct algorithm explores enough that the empirical mean concentrates around the true mean. The $\varepsilon$-greedy algorithm achieves $O(T)$ regret with constant $\varepsilon$ (still linear, but better), and the optimal **UCB1** (upper confidence bound) algorithm achieves $O(\log T)$ regret — exponentially better. The asymptotic lower bound is also $\Omega(\log T)$ (Lai & Robbins, 1985), so UCB is optimal up to constants.

---

**Exercise 6.**
A trading agent is trained on historical market data using reinforcement learning. List three distinct ways this can fail when the agent is deployed, even if it performs well in backtest.

??? success "Solution to Exercise 6"
    **Distribution shift / non-stationarity:** market regimes change. The patterns the agent learned (volatility, momentum, mean reversion) may not hold in the deployed period. RL agents trained on a single regime are particularly brittle when conditions shift.

    **Market impact:** in backtest the agent's trades are infinitesimal — historical prices are taken as given. Deployed at scale, the agent's own orders move the price against itself (slippage). The estimated reward function ignored this and the policy chooses sizes that no longer make sense.

    **Overfitting to the backtest path:** RL on a single historical path is essentially overfitting one realization of a stochastic process. The policy may exploit specific sequences ("on day 17 of the test set, momentum kicked in") that will not recur. Defenses include training on multiple bootstrap-resampled paths, walk-forward validation, and conservative ensembling.

    Additional valid answers: **action constraints** that were ignored in simulation but matter in production (margin limits, regulatory holding periods); **adversarial agents** (other algorithmic traders adapt to the agent's behavior); **operational issues** (latency, data quality, order failure rates).
