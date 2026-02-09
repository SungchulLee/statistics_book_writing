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
