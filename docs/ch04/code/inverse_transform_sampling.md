# 역변환 표본추출

## 개요

**역변환 표본추출**은 역 CDF(분위수 함수)를 구할 수 있는 임의의 분포에서 확률표본을 생성하는 방법이다. 핵심 결과는 다음과 같다.

$U \sim \text{Uniform}(0, 1)$이고 $F$가 역함수 $F^{-1}$을 갖는 CDF이면:

$$
X = F^{-1}(U) \sim F
$$

이 하나의 착상이 계산통계학과 Monte Carlo 모의실험의 상당 부분을 떠받친다.

---

## 증명

임의의 $x$에 대해:

$$
P(X \le x) = P(F^{-1}(U) \le x) = P(U \le F(x)) = F(x)
$$

마지막 등식은 $U \sim \text{Uniform}(0,1)$이므로 $P(U \le p) = p$라는 사실을 사용한다. $\square$

---

## 예 1: Exponential 분포

Exponential 분포의 CDF는 $F(x) = 1 - e^{-\lambda x}$이다. 역함수를 구하면:

$$
F^{-1}(u) = -\frac{\ln(1 - u)}{\lambda}
$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
n = 10_000
lam = 1.0

u = np.random.uniform(0, 1, n)
x_exp = -np.log(1 - u) / lam

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(u, bins=50, density=True, color="lightgray", edgecolor="black")
axes[0].set_title("Step 1: U ~ Uniform(0,1)")

t = np.linspace(0, np.max(x_exp), 300)
axes[1].hist(x_exp, bins=50, density=True, color="steelblue",
             edgecolor="white", alpha=0.7, label="Transformed")
axes[1].plot(t, stats.expon.pdf(t, scale=1/lam), "r-", lw=2.5, label="Exp PDF")
axes[1].set_title("Step 2: X = -ln(1-U)/λ")
axes[1].legend()

u_grid = np.linspace(0.001, 0.999, 300)
axes[2].plot(u_grid, -np.log(1 - u_grid) / lam, "b-", lw=2)
axes[2].set_title("Inverse CDF: F⁻¹(u)")
axes[2].set_xlabel("u")
axes[2].set_ylabel("x")

plt.tight_layout()
plt.show()
```

---

## 예 2: Cauchy 분포

표준 Cauchy 분포의 CDF는 $F(x) = \frac{1}{2} + \frac{1}{\pi}\arctan(x)$이다. 역함수를 구하면:

$$
F^{-1}(u) = \tan\!\left(\pi\!\left(u - \frac{1}{2}\right)\right)
$$

```python
u = np.random.uniform(0, 1, n)
x_cauchy = np.tan(np.pi * (u - 0.5))

fig, ax = plt.subplots(figsize=(10, 4))
x_clipped = np.clip(x_cauchy, -20, 20)
ax.hist(x_clipped, bins=80, density=True, color="coral",
        edgecolor="white", alpha=0.7, label="Transformed")
t2 = np.linspace(-20, 20, 500)
ax.plot(t2, stats.cauchy.pdf(t2), "k-", lw=2.5, label="Cauchy PDF")
ax.set_title("Cauchy via Inverse Transform")
ax.set_xlim(-20, 20)
ax.legend()
plt.tight_layout()
plt.show()
```

Cauchy 예는 역변환 표본추출이 유한한 평균조차 없는 두꺼운 꼬리 분포에서도 작동함을 보여 준다.

---

## 언제 사용하는가

| 상황 | 권장 방법 |
|---|---|
| $F^{-1}$이 닫힌 형태로 주어짐 | 역변환 (빠르고 정확) |
| $F^{-1}$ 계산 비용이 큼 | 기각표본추출이나 MCMC 고려 |
| 이산분포 | 누적 PMF 문턱값 사용 |
| 다변량 분포 | 조건부 분해나 전용 알고리즘 사용 |

---

## 연습문제

**연습문제 1.**
$\text{Uniform}(a, b)$ 분포의 역 CDF를 유도하고 역변환 공식을 쓰라.

??? success "풀이"
    CDF는 $F(x) = (x - a)/(b - a)$이다. $u = F(x)$로 두고 풀면:

    $$
    x = a + (b - a)u = F^{-1}(u)
    $$

    따라서 $U \sim \text{Uniform}(0,1)$이면 $X = a + (b-a)U \sim \text{Uniform}(a, b)$이다.

---

**연습문제 2.**
역변환 표본추출로 $\text{Bernoulli}(p)$ 분포에서 표본을 생성하라. 같은 착상이 임의의 이산분포로 어떻게 확장되는지 설명하라.

??? success "풀이"
    $U \sim \text{Uniform}(0,1)$을 생성한다. $U \le p$이면 $X = 1$로, 그렇지 않으면 $X = 0$으로 둔다.

    값이 $x_1, x_2, \ldots$이고 확률이 $p_1, p_2, \ldots$인 일반적인 이산분포에서는 누적확률 $c_k = \sum_{i=1}^k p_i$를 계산한다. $U \le c_k$를 만족하는 가장 작은 첨자 $k$에 대해 $X = x_k$로 둔다. 이는 $[0,1]$을 길이 $p_k$인 구간들로 나누고 각 구간을 $x_k$에 대응시키는 것이다.

---

**연습문제 3.**
역변환 결과를 증명하라. $U \sim \text{Uniform}(0,1)$이고 $F$가 연속이고 순증가하는 CDF이면 $F^{-1}(U) \sim F$이다.

??? success "풀이"
    임의의 $x \in \mathbb{R}$에 대해:

    $$
    P(F^{-1}(U) \le x) = P(U \le F(x))
    $$

    이 등식은 $F$가 순증가하므로 $F^{-1}(u) \le x \iff u \le F(x)$라는 사실을 사용한다. $U \sim \text{Uniform}(0,1)$이므로:

    $$
    P(U \le F(x)) = F(x)
    $$

    따라서 $P(F^{-1}(U) \le x) = F(x)$이고, 이는 $F^{-1}(U)$의 CDF가 $F$임을 뜻한다. $\square$

---

**연습문제 4.**
Rayleigh 분포는 $x \ge 0$에 대해 CDF가 $F(x) = 1 - e^{-x^2/(2\sigma^2)}$이다. 역 CDF를 유도하고 역변환 표본추출로 Rayleigh 표본을 생성하는 코드를 작성하라.

??? success "풀이"
    $u = 1 - e^{-x^2/(2\sigma^2)}$의 역함수를 구하면:

    $$
    e^{-x^2/(2\sigma^2)} = 1 - u \implies x = \sigma\sqrt{-2\ln(1-u)}
    $$

    코드:

    ```python
    sigma = 1.0
    u = np.random.uniform(0, 1, 10000)
    x = sigma * np.sqrt(-2 * np.log(1 - u))
    ```

    이는 Box-Muller 변환의 한 성분과 밀접하게 관련된다. $R = \sqrt{-2\ln U}$가 Rayleigh 분포를 따른다.

---

**연습문제 5.**
모든 역변환 공식에서 $U$를 $1 - U$로 바꾸어도 출력의 분포가 달라지지 않는 이유를 설명하라. 실무에서 이것이 왜 유용한가?

??? success "풀이"
    $U \sim \text{Uniform}(0,1)$이면 $1 - U$도 $\text{Uniform}(0,1)$이다(균등분포는 0.5를 중심으로 대칭이다). 따라서 $F^{-1}(U)$에서 $U$를 $1 - U$로 바꾸어도 같은 분포가 나온다.

    이는 공식을 간단하게 만들어 주므로 유용하다. Exponential 분포에서는 $-\ln(1-U)/\lambda$를 $-\ln(U)/\lambda$로 바꿀 수 있어 뺄셈 한 번을 아낀다. 실무에서는 $\ln(0) = -\infty$를 주는 $U = 0$이라는 경계 사례도 피하게 되는데, $1 - U = 0$이 될 확률은 0이기 때문이다.
