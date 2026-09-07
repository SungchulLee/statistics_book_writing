# 이산확률변수

## 개요

**확률변수**는 표본공간의 결과를 실수로 보내는 함수다. **이산확률변수**는 가산개의 서로 다른 값을 취한다.

---

## 정의

**확률변수** $X$는 형식적으로 다음 함수로 정의된다.

$$
X : \Omega \longrightarrow \mathbb{R}
$$

여기서 $\Omega$는 표본공간이고 $\mathbb{R}$은 실수의 집합이다.

**이산확률변수**는 서로 다른 값들의 가산집합 $\{x_1, x_2, x_3, \ldots\}$을 취한다. 주사위를 굴린 결과나 여러 번 동전을 던졌을 때 앞면의 개수 등이 그 예다.

---

## 이산확률변수의 분포

각 결과 $\omega \in \Omega$에 그 결과의 확률을 나타내는 무게를 가진 "벽돌"이 붙어 있다고 상상하자. 확률변수 $X$를 적용하면 벽돌을 $\omega$에서 실직선 위의 위치 $X(\omega)$로 옮긴다.

모든 벽돌을 옮기고 나면 $\mathbb{R}$ 위에 놓인 무게의 배치가 **$X$의 분포**를 정의한다.

$$
\begin{aligned}
\mathbb{P}(X = a) &= \text{Weight of the bricks at } a \\
\mathbb{P}(X \in A) &= \text{Weight of the bricks in the set } A
\end{aligned}
$$

---

## 확률질량함수 (PMF)

이산확률변수 $X$에 대해 **확률질량함수**는 각 값에 확률을 부여한다.

$$
p_{x_i} = P(X = x_i) = \text{Weight of the brick at } x_i
$$

확률질량함수는 다음을 만족해야 한다.

1. 모든 $i$에 대해 $p_{x_i} \geq 0$
2. $\sum_i p_{x_i} = 1$

---

## 예제

### 예: 공정한 주사위의 확률질량함수

$X$를 공정한 육면체 주사위를 굴린 결과라 하자.

$$
P(X = x) = \frac{1}{6}, \quad \text{for } x = 1, 2, 3, 4, 5, 6
$$

### 예: 동전 3번 던지기에서 앞면의 개수

$X$를 공정한 동전을 3번 던졌을 때 앞면의 개수라 하자. 가능한 값은 $\{0, 1, 2, 3\}$이다.

$$
\begin{aligned}
P(X = 0) &= \frac{1}{8} \\
P(X = 1) &= \frac{3}{8} \\
P(X = 2) &= \frac{3}{8} \\
P(X = 3) &= \frac{1}{8}
\end{aligned}
$$

### 예: 야구 카드

휴고는 좋아하는 선수의 카드를 얻을 때까지 야구 카드 팩을 살 계획이다. 최대 네 팩까지 살 수 있고 각 팩에 그 카드가 들어 있을 확률은 0.2다. $X$를 휴고가 사는 팩의 수라 하자.

**풀이:**

$$
\begin{aligned}
P(X=1) &= 0.2 \\
P(X=2) &= 0.8 \times 0.2 = 0.16 \\
P(X=3) &= 0.8^2 \times 0.2 = 0.128 \\
P(X=4) &= 1 - P(X=1) - P(X=2) - P(X=3) = 0.512
\end{aligned}
$$

따라서

$$
\begin{aligned}
P(X \geq 2) &= 1 - P(X=1) = 0.8 \\
P(X = 4) &= 0.512
\end{aligned}
$$

이다. $P(X=4) = 0.512$에는 네 번째 팩에서 카드를 찾을 확률과 끝내 찾지 못할 확률이 모두 포함된다. 어느 쪽이든 휴고는 4팩에서 멈추기 때문이다.

### 예: 삼면체 주사위 두 개의 차이

$D_1, D_2$가 삼면체 주사위를 굴린 결과일 때 $D = |D_1 - D_2|$라 하자. 똑같이 일어날 법한 아홉 결과로부터 다음을 얻는다.

| $D_1 \backslash D_2$ | 1 | 2 | 3 |
|:---:|:---:|:---:|:---:|
| **1** | 0 | 1 | 2 |
| **2** | 1 | 0 | 1 |
| **3** | 2 | 1 | 0 |

$$
P(D=0) = \frac{3}{9}, \quad P(D=1) = \frac{4}{9}, \quad P(D=2) = \frac{2}{9}
$$

---

## 파이썬 구현

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_pmf(values, probabilities, title="PMF"):
    """Plot the probability mass function."""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(values, probabilities, width=0.4, alpha=0.7, edgecolor='black')
    ax.set_xlabel('x')
    ax.set_ylabel('P(X = x)')
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

# Fair die PMF
values = [1, 2, 3, 4, 5, 6]
probs = [1/6] * 6
plot_pmf(values, probs, "PMF of a Fair Die")

# Coin flip PMF (3 flips, counting heads)
from math import comb
n = 3
values = list(range(n + 1))
probs = [comb(n, k) * (0.5**k) * (0.5**(n-k)) for k in values]
plot_pmf(values, probs, "PMF: Number of Heads in 3 Coin Flips")

# Baseball cards PMF
values = [1, 2, 3, 4]
probs = [0.2, 0.16, 0.128, 0.512]
plot_pmf(values, probs, "PMF: Baseball Card Packs Purchased")
```

---

## 핵심 요약

- 이산확률변수는 결과를 실수의 가산집합으로 보낸다.
- 확률질량함수는 가능한 각 값의 확률을 주며 그 합이 1이어야 한다.
- "벽돌" 비유가 직관을 제공한다. 각 결과가 무게(확률)를 지니고 있고 확률변수가 이 무게들을 실직선 위로 옮긴다.

## 연습문제

**연습문제 1.**
이산확률변수 $X$의 확률질량함수가 $P(X=0) = 0.1$, $P(X=1) = 0.3$, $P(X=2) = c$, $P(X=3) = 0.2$다. $c$의 값을 구하고 $P(X \geq 2)$를 계산하라.

??? success "풀이"
    확률질량함수의 합이 1이어야 하므로

    $$
    0.1 + 0.3 + c + 0.2 = 1 \implies c = 0.4
    $$

    이다. 따라서

    $$
    P(X \geq 2) = P(X=2) + P(X=3) = 0.4 + 0.2 = 0.6
    $$

    이다.

---

**연습문제 2.**
공정한 사면체 주사위(면이 1, 2, 3, 4) 두 개를 굴린다. $S$를 두 주사위의 합이라 하자. $S$의 확률질량함수를 모두 쓰고 확률의 합이 1임을 확인하라.

??? success "풀이"
    똑같이 일어날 법한 결과가 $4 \times 4 = 16$개다. 가능한 합은 2에서 8까지다.

    | $s$ | 결과 | $P(S = s)$ |
    |:---:|:---|:---:|
    | 2 | $(1,1)$ | $1/16$ |
    | 3 | $(1,2),(2,1)$ | $2/16$ |
    | 4 | $(1,3),(2,2),(3,1)$ | $3/16$ |
    | 5 | $(1,4),(2,3),(3,2),(4,1)$ | $4/16$ |
    | 6 | $(2,4),(3,3),(4,2)$ | $3/16$ |
    | 7 | $(3,4),(4,3)$ | $2/16$ |
    | 8 | $(4,4)$ | $1/16$ |

    확인: $1 + 2 + 3 + 4 + 3 + 2 + 1 = 16$이므로 $\sum P(S=s) = 16/16 = 1$이다. $\square$

---

**연습문제 3.**
치우친 동전의 $P(\text{앞면}) = 0.7$이다. 이 동전을 3번 던진다. $X$를 앞면의 개수라 할 때 $X$의 확률질량함수를 쓰라.

??? success "풀이"
    각 던지기는 독립이고 $p = 0.7$(앞면), $q = 0.3$(뒷면)이다. 3번 던졌을 때 앞면의 개수는 이항분포를 따른다.

    $$
    P(X = k) = \binom{3}{k} (0.7)^k (0.3)^{3-k}
    $$

    각 값을 계산하면

    $$
    P(X=0) = \binom{3}{0}(0.7)^0(0.3)^3 = 0.027
    $$

    $$
    P(X=1) = \binom{3}{1}(0.7)^1(0.3)^2 = 3 \times 0.063 = 0.189
    $$

    $$
    P(X=2) = \binom{3}{2}(0.7)^2(0.3)^1 = 3 \times 0.147 = 0.441
    $$

    $$
    P(X=3) = \binom{3}{3}(0.7)^3(0.3)^0 = 0.343
    $$

    이다. 확인: $0.027 + 0.189 + 0.441 + 0.343 = 1.000$. $\square$

---

**연습문제 4.**
연속확률변수(예: 무작위로 고른 사람의 정확한 키)를 확률질량함수로 기술할 수 없는 이유를 설명하라. 연속인 경우에는 무엇이 확률질량함수를 대신하는가?

??? success "풀이"
    확률질량함수는 개별 값에 양의 확률을 부여한다. 받침의 각 값에 대해 $P(X = x) > 0$이다. 연속확률변수에서는 받침이 비가산 구간이다(예: $[150, 200]$ cm 안의 모든 실수). 개별 값마다 양의 확률을 가진다면 비가산개의 값에 대한 합(또는 적분)이 무한대로 발산하여 정규화 공리 $P(\Omega) = 1$을 위반한다.

    대신 연속확률변수는 **확률밀도함수** $f(x)$로 기술하며, $f(x) \geq 0$이고 $\int_{-\infty}^{\infty} f(x)\,dx = 1$이다. 확률밀도함수는 확률이 아니라 밀도를 준다. 어떤 한 값에 대해서도 $P(X = x) = 0$이지만 $P(a \leq X \leq b) = \int_a^b f(x)\,dx$가 구간에 대한 확률을 준다.

---

**연습문제 5.**
**기하분포.** $X$ = i.i.d. Bernoulli($p$)에서 첫 성공까지의 시행 횟수. 확률질량함수, $\mathbb{E}[X]$, $\mathrm{Var}(X)$를 유도하라.

??? success "풀이"
    **확률질량함수:** $X = k$이려면 실패 $k - 1$번 뒤에 성공해야 하므로 $k = 1, 2, \ldots$에 대해 $P(X = k) = (1 - p)^{k-1} p$이다.

    정규화: $\sum_{k=1}^\infty (1-p)^{k-1} p = p/p = 1$. ✓

    **기댓값:** 꼬리합 공식에 의해 $\mathbb{E}[X] = \sum_{n=0}^\infty P(X > n) = \sum_{n=0}^\infty (1-p)^n = 1/p$이다.

    **분산:** $\mathrm{Var}(X) = (1-p)/p^2$이다($\mathbb{E}[X(X-1)]$을 이용한 비슷한 계산으로 유도한다).

    $p = 0.5$이면 평균 2회, 분산 2, 표준편차 $\sqrt 2$다. 기하분포는 지수분포의 이산판이며 무기억 성질을 물려받는다: $P(X > m + n \mid X > m) = P(X > n)$.

---

**연습문제 6.**
**이항분포의 포아송 근사.** $np \to \lambda$가 상수인 채로 $n \to \infty$이면 Binomial$(n, p)$ → Poisson$(\lambda)$임을 보여라.

??? success "풀이"
    이항 확률질량함수에 $p = \lambda/n$을 대입한다.

    $$
    P(X = k) = \binom{n}{k}\left(\frac{\lambda}{n}\right)^k \left(1 - \frac{\lambda}{n}\right)^{n-k}
    $$

    정리하면

    $$
    = \frac{\lambda^k}{k!} \cdot \underbrace{\frac{n!}{(n-k)! n^k}}_{\to 1} \cdot \underbrace{\left(1 - \frac{\lambda}{n}\right)^n}_{\to e^{-\lambda}} \cdot \underbrace{\left(1 - \frac{\lambda}{n}\right)^{-k}}_{\to 1}
    $$

    이다. $n \to \infty$이면 $P(X = k) \to \frac{\lambda^k e^{-\lambda}}{k!}$로, 포아송 확률질량함수다.

    **용도:** 드문 사건($p$가 작고 $n$이 클 때)에는 포아송이 이항보다 훨씬 간단하다. 응용: 도시의 하루 교통사고 수, 칩당 결함 수, 분당 통화 수, 유전체당 돌연변이 수.

    어림법칙: $n \ge 20$, $p \le 0.05$, $np \le 10$일 때 포아송이 잘 맞는다. 그렇지 않으면 이항을 쓰거나, $np \ge 10$이면 정규근사를 쓴다.
