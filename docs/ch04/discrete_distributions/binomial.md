# Bernoulli 분포와 Binomial 분포

## 개요

**Bernoulli 분포**는 두 가지 결과(성공/실패)를 갖는 단일 시행을 모형화하고, **Binomial 분포**는 이를 확장하여 $n$번의 독립 시행에서 성공 횟수를 센다. 이 둘은 이산확률모형의 기초를 이룬다.

---

## Bernoulli 분포

### 정의

확률변수 $X$가 확률 $p$로 값 1(성공)을, 확률 $1 - p$로 값 0(실패)을 가지면 $X$는 Bernoulli 분포를 따른다:

$$
X \sim \text{Bernoulli}(p), \qquad P(X = x) = p^x (1 - p)^{1-x}, \quad x \in \{0, 1\}
$$

### 성질

$$
\begin{aligned}
E[X] &= p \\
\text{Var}(X) &= p(1 - p) \\
\text{SD}(X) &= \sqrt{p(1 - p)}
\end{aligned}
$$

### 분산의 유도

$$
E[X^2] = 0^2 \cdot (1-p) + 1^2 \cdot p = p
$$

$$
\text{Var}(X) = E[X^2] - (E[X])^2 = p - p^2 = p(1 - p)
$$

---

## Binomial 분포

### 정의

$X_1, X_2, \ldots, X_n$이 독립인 $\text{Bernoulli}(p)$ 확률변수이면, $Y = \sum_{i=1}^n X_i$는 **Binomial 분포**를 따른다:

$$
Y \sim \text{Binomial}(n, p), \qquad P(Y = k) = \binom{n}{k} p^k (1 - p)^{n-k}, \quad k = 0, 1, \ldots, n
$$

이항계수 $\binom{n}{k} = \frac{n!}{k!(n-k)!}$는 $n$번의 시행에서 $k$번의 성공을 고르는 경우의 수를 센다.

### 성질

$$
\begin{aligned}
E[Y] &= np \\
\text{Var}(Y) &= np(1 - p) \\
\text{SD}(Y) &= \sqrt{np(1 - p)}
\end{aligned}
$$

### 평균과 분산의 유도

$X_i \overset{\text{iid}}{\sim} \text{Bernoulli}(p)$에 대해 $Y = \sum_{i=1}^n X_i$이므로:

$$
E[Y] = \sum_{i=1}^n E[X_i] = np
$$

독립성에 의해:

$$
\text{Var}(Y) = \sum_{i=1}^n \text{Var}(X_i) = np(1 - p)
$$

### PMF의 합이 1임을 확인하기

이항정리에 의해:

$$
\sum_{k=0}^n \binom{n}{k} p^k (1-p)^{n-k} = (p + (1-p))^n = 1^n = 1
$$

---

## 이항계수 항등식

Binomial 분포를 다룰 때 유용한 항등식이 여럿 있다:

$$
\begin{aligned}
(1) &\quad \binom{n}{k} = \binom{n}{n-k} \quad \text{(대칭성)} \\[4pt]
(2) &\quad \binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k} \quad \text{(Pascal 규칙)} \\[4pt]
(3) &\quad k\binom{n}{k} = n\binom{n-1}{k-1} \quad \text{(흡수 항등식)}
\end{aligned}
$$

흡수 항등식은 PMF로부터 $E[Y]$를 직접 계산할 때 특히 유용하다:

$$
E[Y] = \sum_{k=0}^n k \binom{n}{k} p^k (1-p)^{n-k} = np \sum_{k=1}^n \binom{n-1}{k-1} p^{k-1} (1-p)^{n-k} = np
$$

---

## 예제

**문제:** 어떤 주식이 하루에 상승할 확률이 60%이고 날짜별로 독립이라 하자. 10 거래일 동안 정확히 7일 상승할 확률은 얼마인가?

**풀이:**

$$
P(Y = 7) = \binom{10}{7} (0.6)^7 (0.4)^3 = 120 \cdot 0.0280 \cdot 0.064 = 0.2150
$$

상승일 수의 기댓값: $E[Y] = 10 \times 0.6 = 6$.

---

## Python: PMF, CDF, 표본추출

### PMF와 CDF

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

n, p = 10, 0.6
x = np.arange(0, n + 1)

fig, ax = plt.subplots(figsize=(12, 3))
ax.bar(x - 0.15, stats.binom(n, p).pmf(x), width=0.3, label='PMF', alpha=0.7)
ax.bar(x + 0.15, stats.binom(n, p).cdf(x), width=0.3, label='CDF', alpha=0.7)
ax.set_xlabel('k')
ax.set_xticks(x)
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
plt.show()
```

### 모수에 따른 비교

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

fig, ax = plt.subplots(figsize=(12, 3))
for n, p in [(10, 0.5), (20, 0.5), (20, 0.7)]:
    x = np.arange(0, n + 1)
    ax.plot(x, stats.binom(n, p).pmf(x), 'o-', label=f'n={n}, p={p}', markersize=4)
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel('k')
ax.legend()
plt.show()
```

### 표본추출과 검증

```python
import numpy as np
from scipy import stats

np.random.seed(42)
n, p = 10, 0.6
samples = stats.binom(n, p).rvs(100_000)

print(f"Theoretical mean: {n*p:.4f},  Sample mean: {samples.mean():.4f}")
print(f"Theoretical var:  {n*p*(1-p):.4f},  Sample var:  {samples.var():.4f}")
```

---

## Binomial 분포의 정규근사

$n$이 크면 Binomial 분포는 정규분포로 잘 근사된다:

$$
Y \sim \text{Binomial}(n, p) \approx N(np, \, np(1-p)) \quad \text{when } np \geq 5 \text{ and } n(1-p) \geq 5
$$

연속성 수정을 적용하면 $P(Y \leq k) \approx \mathcal{N}\left(\frac{k + 0.5 - np}{\sqrt{np(1-p)}}\right)$이다.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

n, p = 50, 0.4
x_disc = np.arange(0, n + 1)
x_cont = np.linspace(0, n, 200)

fig, ax = plt.subplots(figsize=(12, 3))
ax.bar(x_disc, stats.binom(n, p).pmf(x_disc), alpha=0.5, label='Binomial PMF')
ax.plot(x_cont, stats.norm(n*p, np.sqrt(n*p*(1-p))).pdf(x_cont),
        'r-', lw=2, label='Normal approx.')
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
plt.show()
```

---

## 핵심 요약

- Bernoulli 분포는 단일 이항 시행을 모형화하고, Binomial 분포는 $n$번의 독립 시행에서 성공 횟수를 센다.
- Binomial PMF는 성공이 나타날 수 있는 모든 순서를 반영하기 위해 이항계수를 사용한다.
- 평균 $np$와 분산 $np(1-p)$는 독립 Bernoulli 확률변수의 합이라는 표현에서 곧바로 따라 나온다.
- $n$이 크면 Binomial 분포는 정규분포로 잘 근사되어 이산확률과 연속확률을 이어 준다.

## 연습문제

**연습문제 1.**
10개의 제품이 각각 독립적으로 확률 $p = 0.15$로 불량이다. (a) 불량품 개수 $X$의 분포는? (b) $P(X = 2)$. (c) $P(X \ge 3)$. (d) 평균과 분산.

??? success "연습문제 1 풀이"
    (a) $X \sim \mathrm{Binomial}(10, 0.15)$.

    (b) $P(X = 2) = \binom{10}{2}(0.15)^2(0.85)^8 = 45 \cdot 0.0225 \cdot 0.2725 \approx 0.276$.

    (c) $P(X \ge 3) = 1 - P(X \le 2)$. $P(X = 0) = (0.85)^{10} \approx 0.197$, $P(X = 1) = 10 \cdot 0.15 \cdot (0.85)^9 \approx 0.347$, $P(X = 2) \approx 0.276$를 계산하면 $P(X \ge 3) = 1 - 0.820 = 0.180$.

    (d) $\mathbb{E}[X] = np = 1.5$. $\mathrm{Var}(X) = np(1-p) = 10 \cdot 0.15 \cdot 0.85 = 1.275$.

---

**연습문제 2.**
$Y \sim \mathrm{Binomial}(n, p)$에 대해 $X_i \sim \mathrm{Bernoulli}(p)$인 지시함수 표현 $Y = \sum_{i=1}^n X_i$를 사용하여 **$\mathbb{E}[Y] = np$와 $\mathrm{Var}(Y) = np(1-p)$를 증명하라.**

??? success "연습문제 2 풀이"
    **평균.** 기댓값의 선형성에 의해:

    $$
    \mathbb{E}[Y] = \mathbb{E}\!\sum_{i=1}^n X_i = \sum_{i=1}^n \mathbb{E}[X_i] = \sum_{i=1}^n p = np
    $$

    **분산.** 독립성에 의해:

    $$
    \mathrm{Var}(Y) = \sum_{i=1}^n \mathrm{Var}(X_i) = \sum_{i=1}^n p(1-p) = np(1-p)
    $$

    지시함수의 합으로 나타내는 표현이 가장 깔끔한 유도이다. PMF로부터 직접 계산해도 되지만 흡수 항등식 $k\binom{n}{k} = n\binom{n-1}{k-1}$이 필요하다. $\square$

---

**연습문제 3.**
**독립인 두 Binomial 확률변수의 합.** $X \sim \mathrm{Binomial}(n_1, p)$와 $Y \sim \mathrm{Binomial}(n_2, p)$가 독립이라 하자. $X + Y \sim \mathrm{Binomial}(n_1 + n_2, p)$임을 보여라.

??? success "연습문제 3 풀이"
    각 Binomial 확률변수는 그 자체가 i.i.d. Bernoulli($p$) 시행의 합이다. $X$는 $n_1$개의 Bernoulli($p$)의 합이고, $Y$는 $n_2$개의 합이다. $X$와 $Y$가 독립이라는 것은 두 그룹에 속한 Bernoulli 확률변수들이 서로 독립임을 뜻한다.

    따라서 $X + Y$는 $n_1 + n_2$개의 i.i.d. Bernoulli($p$) 시행의 합이므로 Binomial$(n_1 + n_2, p)$이다. $\square$

    **MGF를 통한 확인:** $M_{X+Y}(t) = M_X(t) M_Y(t) = (1 - p + pe^t)^{n_1}(1 - p + pe^t)^{n_2} = (1 - p + pe^t)^{n_1 + n_2}$이며, 이는 Binomial$(n_1 + n_2, p)$의 MGF이다.

    **주의:** *$p$가 공통이라는 점*이 본질적이다. $p$가 다르면 합은 Binomial이 아니다(Poisson-binomial 분포를 따른다).

---

**연습문제 4.**
**연속성 수정을 적용한 정규근사.** Binomial(100, 0.4)에 대해 연속성 수정을 적용한 경우와 적용하지 않은 경우 각각 정규근사로 $P(35 \le Y \le 45)$를 구하라. 정확한 Binomial 값(0.7287)과 비교하라.

??? success "연습문제 4 풀이"
    $\mu = 40$, $\sigma = \sqrt{100 \cdot 0.4 \cdot 0.6} = \sqrt{24} \approx 4.899$.

    **연속성 수정 없이:**

    $$
    P(35 \le Y \le 45) \approx \Phi\!\left(\frac{45 - 40}{4.899}\right) - \Phi\!\left(\frac{35 - 40}{4.899}\right) = \Phi(1.021) - \Phi(-1.021) = 0.8463 - 0.1537 = 0.6926
    $$

    오차: $|0.6926 - 0.7287| = 0.036$.

    **연속성 수정을 적용하면:**

    $$
    P(35 \le Y \le 45) \approx \Phi\!\left(\frac{45.5 - 40}{4.899}\right) - \Phi\!\left(\frac{34.5 - 40}{4.899}\right) = \Phi(1.122) - \Phi(-1.122) = 0.8691 - 0.1309 = 0.7382
    $$

    오차: $|0.7382 - 0.7287| = 0.010$ — 세 배 작다.

    이산분포를 연속분포로 근사할 때는 항상 연속성 수정을 사용하라.

---

**연습문제 5.**
**Bernoulli 분산은 $p = 1/2$에서 최대가 된다.** 이를 해석적으로 증명하고 신뢰구간 계산에서 갖는 실용적 의미를 설명하라.

??? success "연습문제 5 풀이"
    $\mathrm{Var}(X) = p(1 - p)$를 $p$에 대해 미분하면:

    $$
    \frac{d}{dp}\, p(1-p) = 1 - 2p
    $$

    0으로 두면 $p = 1/2$이다. 이계도함수가 $-2 < 0$이므로 최대점이며, 최대 분산은 $1/4$이다.

    **실용적 의미:** 이항 비율의 신뢰구간에서 최악의 경우 분산은 $p(1-p) \le 1/4$이다. 보수적인 표준오차는 $\sqrt{1/(4n)} = 1/(2\sqrt n)$이므로, 95% 오차한계는 최대 $1.96/(2\sqrt n) \approx 1/\sqrt n$이다.

    오차한계를 $\le 0.03$으로 두면 $n \ge 1/(0.03)^2 \approx 1111$이 되는데, 이것이 전국 여론조사에서 "n ≈ 1000" 규칙이 나온 배경이다. 실제 $p$는 대개 0.5에서 떨어져 있으므로 이 보수적 한계는 다소 느슨하지만, $p$가 무엇이든 통하는 표본크기 추정치를 제공한다.

---

**연습문제 6.**
**역문제: 표본으로부터 $p$ 구하기.** $n = 100$번의 시행에서 $Y = 35$번의 성공을 관측했다. 두 가지 방법으로 $p$에 대한 근사 95% 신뢰구간을 구성하라: (a) **Wald** ($\hat p \pm 1.96 \sqrt{\hat p(1 - \hat p)/n}$); (b) **Wilson 점수 구간**. 둘을 비교하라.

??? success "연습문제 6 풀이"
    $\hat p = 35/100 = 0.35$.

    **(a) Wald 구간:** $\hat p \pm 1.96 \sqrt{\hat p(1 - \hat p)/n} = 0.35 \pm 1.96 \sqrt{0.35 \cdot 0.65 / 100} = 0.35 \pm 1.96 \cdot 0.0477 = 0.35 \pm 0.094 = (0.256, 0.444)$.

    **(b) Wilson 구간** (부등식 $|\hat p - p|/\sqrt{p(1-p)/n} \le 1.96$을 $p$에 대해 푼다):

    $$
    p_{\text{Wilson}} = \frac{\hat p + z^2/(2n) \pm z\sqrt{\hat p(1-\hat p)/n + z^2/(4n^2)}}{1 + z^2/n}
    $$

    $z = 1.96$, $\hat p = 0.35$, $n = 100$일 때:

    분자의 중심: $0.35 + 0.0192 = 0.3692$. 분자의 반폭: $1.96 \sqrt{0.002275 + 9.6e-5} = 1.96 \sqrt{0.002371} \approx 0.0954$.

    분모: $1 + 0.0384 = 1.0384$.

    신뢰구간: $((0.3692 - 0.0954)/1.0384, (0.3692 + 0.0954)/1.0384) = (0.264, 0.448)$.

    **비교:** Wilson 구간은 $\hat p$를 중심으로 비대칭이며(0.5 쪽으로 약간 이동), $\hat p$가 0이나 1에 가까울 때도 포함확률이 보장된다. Wald 구간은 극단적인 $\hat p$에서 퇴화할 수 있지만(0 아래나 1 위로 뻗어 나간다) Wilson 구간은 결코 그렇지 않다. 현대적 관행에서는 특히 작은 표본에서 이항 신뢰구간으로 Wald보다 Wilson을 선호한다.
