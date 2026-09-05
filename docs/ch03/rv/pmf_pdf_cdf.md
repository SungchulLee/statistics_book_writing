# 확률질량함수, 확률밀도함수, 누적분포함수

## 개요

확률변수의 분포를 특징짓는 세 가지 기본 함수는 이산변수에 대한 확률질량함수(PMF), 연속변수에 대한 확률밀도함수(PDF), 그리고 둘 다에 적용되는 누적분포함수(CDF)다.

---

## 확률질량함수와 확률밀도함수

$$
\begin{aligned}
\textbf{PMF:} \quad & p_{x_i} = \text{The weight of the brick assigned to the discrete value } x_i \\[8pt]
\textbf{PDF:} \quad & f(x)\,dx = \text{The weight of the bricks within the continuous interval } [x, x + dx]
\end{aligned}
$$

---

## 누적분포함수 (CDF)

누적분포함수 $F(x)$는 확률변수 $X$가 $x$ 이하의 값을 취할 누적 확률을 준다.

$$
F(x) = \mathbb{P}(X \leq x) =
\begin{cases}
\displaystyle\sum_{x_i \leq x} p_{x_i}, & \text{if } X \text{ is discrete} \\[10pt]
\displaystyle\int_{-\infty}^x f(s)\,ds, & \text{if } X \text{ is continuous}
\end{cases}
$$

벽돌 비유로 말하면 $F(x)$는 **$-\infty$부터 $x$까지 쌓인 모든 벽돌의 총 무게**다.

### 누적분포함수의 성질

- $F(x)$는 비감소함수다
- $\lim_{x \to -\infty} F(x) = 0$
- $\lim_{x \to +\infty} F(x) = 1$
- 연속인 $X$에 대해 $F'(x) = f(x)$ (확률밀도함수는 누적분포함수의 도함수다)

---

## 확률밀도함수와 누적분포함수의 관계

확률밀도함수와 누적분포함수는 적분과 미분으로 이어져 있다.

$$
\text{CDF} = \int \text{PDF} \qquad \text{and} \qquad \text{PDF} = \frac{d}{dx} \text{CDF}
$$

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

fig, (ax_pdf, ax_arrow, ax_cdf) = plt.subplots(1, 3, figsize=(12, 3))

x = np.linspace(-3, 3, 100)

# PDF
ax_pdf.set_title("PDF", fontsize=16)
ax_pdf.plot(x, stats.norm().pdf(x))

# Arrows showing relationship
ax_arrow.arrow(0.1, 0.6, 0.8, 0, width=0.05, length_includes_head=True)
ax_arrow.arrow(0.9, 0.4, -0.8, 0, width=0.05, length_includes_head=True)
ax_arrow.annotate("Integrate", (0.38, 0.75), fontsize=14)
ax_arrow.annotate("Differentiate", (0.30, 0.2), fontsize=14)
for spine in ax_arrow.spines.values():
    spine.set_visible(False)
ax_arrow.set_xticks([])
ax_arrow.set_yticks([])

# CDF
ax_cdf.set_title("CDF", fontsize=16)
ax_cdf.plot(x, stats.norm().cdf(x))

for ax in (ax_pdf, ax_cdf):
    ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()
```

---

## 분위수함수 (PPF)

**분위수함수**는 누적분포함수의 역함수다. 누적확률 $p$가 주어지면 $P(X \leq x) = p$가 되는 값 $x$를 돌려준다.

$$
\text{PPF}(p) = F^{-1}(p) = \inf\{x : F(x) \geq p\}
$$

### 예: 표준정규분포의 95번째 백분위수

$Z \sim N(0, 1)$에 대해 $P(Z \leq z) = 0.95$가 되는 값 $z$는 약 1.645다.

```python
import scipy.stats as stats

z_95 = stats.norm(0, 1).ppf(0.95)
print(f"95th percentile of N(0,1): {z_95:.4f}")
```

### 예: 97.5번째 백분위수

$P(Z \leq z) = 0.975$가 되는 값 $z$는 약 1.96이며 신뢰구간에 널리 쓰인다.

```python
z_975 = stats.norm(0, 1).ppf(0.975)
print(f"97.5th percentile of N(0,1): {z_975:.4f}")
```

---

## 누적분포함수와 분위수함수의 시각화

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

fig, ax = plt.subplots(figsize=(12, 3))
ax.set_xlim(-3, 3)
ax.set_ylim(-0.2, 1.1)

# CDF curve
x = np.linspace(-3, 3, 100)
ax.plot(x, stats.norm().cdf(x), label='CDF')

# PPF demonstration at 0.975
u = 0.975
z = stats.norm().ppf(u)

ax.plot(0, u, 'or', markersize=8)
ax.plot(z, 0, 'or', markersize=8)
ax.annotate(f"U = {u}", (-1.2, u + 0.02), fontsize=14)
ax.annotate(f"Z = {z:.3f}", (z - 0.3, -0.12), fontsize=14)
ax.annotate("PPF →", (0.3, u + 0.03), fontsize=14)
ax.annotate("↓ CDF", (z + 0.1, 0.5), fontsize=14)

ax.spines[['right', 'top']].set_visible(False)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.legend(fontsize=14)
plt.show()
```

---

## 분위수함수로 난수 표본 생성하기 (역변환 표집)

분위수함수의 강력한 응용이 있다. $U \sim \text{Uniform}(0,1)$이면 $X = F^{-1}(U)$는 누적분포함수가 $F$인 분포를 따른다.

```python
import scipy.stats as stats
import matplotlib.pyplot as plt

u = stats.uniform().rvs(10_000)
z = stats.norm().ppf(u)

plt.figure(figsize=(12, 3))
plt.hist(z, bins=100, density=True, alpha=0.7, label='Inverse Transform Samples')
x = np.linspace(-4, 4, 200)
plt.plot(x, stats.norm().pdf(x), 'r--', lw=2, label='N(0,1) PDF')
plt.legend()
plt.show()
```

---

## 예: 정규분포 누적분포함수 계산

```python
from scipy import stats

mean, std_dev = 50, 10

# P(40 ≤ X ≤ 60) for X ~ N(50, 10²)
prob = stats.norm(mean, std_dev).cdf(60) - stats.norm(mean, std_dev).cdf(40)
print(f"P(40 ≤ X ≤ 60) = {prob * 100:.2f}%")

# P(X ≤ 55)
prob_55 = stats.norm(mean, std_dev).cdf(55)
print(f"P(X ≤ 55) = {prob_55 * 100:.2f}%")
```

---

## 경험적 확률질량함수/확률밀도함수와 누적분포함수

실무에서는 히스토그램과 경험적 누적분포함수로 확률밀도함수와 누적분포함수를 자료에서 추정한다.

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

np.random.seed(42)
data = stats.norm.rvs(size=200)

fig, ax = plt.subplots(figsize=(12, 4))

# Empirical PDF (histogram)
counts, bin_edges, _ = ax.hist(data, bins=20, density=True, alpha=0.6, label="Empirical PDF")

# Empirical CDF
empirical_cdf = np.cumsum(counts) / np.sum(counts)
ax.step(bin_edges[1:], empirical_cdf, where='mid', label="Empirical CDF", lw=2)

# Theoretical CDF
ax.plot(bin_edges, stats.norm.cdf(bin_edges), 'r', lw=2, label="Theoretical CDF")

ax.legend()
ax.spines[['right', 'top']].set_visible(False)
plt.show()
```

---

## 요약: scipy.stats 메서드

| 메서드 | 설명 |
|:---|:---|
| `rvs` | 난수 표본 생성 |
| `pdf` | 확률밀도함수 계산 |
| `cdf` | 누적분포함수 계산: $P(X \leq x)$ |
| `sf` | 생존함수: $1 - \text{cdf}(x) = P(X > x)$ |
| `ppf` | 분위수함수(역 누적분포함수) |

---

## 핵심 요약

- **확률질량함수**는 이산변수의 점 확률을, **확률밀도함수**는 연속변수의 확률밀도를 준다.
- **누적분포함수**는 $-\infty$부터 $x$까지 확률을 누적하며 두 유형 모두에 적용된다.
- **분위수함수**는 누적분포함수를 뒤집는다. 확률이 주어지면 대응하는 분위수를 돌려준다.
- 적분이 확률밀도함수 → 누적분포함수를, 미분이 누적분포함수 → 확률밀도함수를 잇는다.

## 연습문제

**연습문제 1.**
$X$ = 공정한 동전을 3번 던졌을 때 앞면의 개수. (a) 확률질량함수를 쓰라. (b) 누적분포함수를 쓰라. (c) $P(1 \le X \le 2)$를 두 가지 방법으로 계산하라.

??? success "연습문제 1 풀이"
    (a) $X \sim \mathrm{Binomial}(3, 1/2)$:

    | $x$ | $p(x)$ |
    |:---:|:---:|
    | 0 | $1/8$ |
    | 1 | $3/8$ |
    | 2 | $3/8$ |
    | 3 | $1/8$ |

    (b) 구간 $(-\infty, 0), [0, 1), [1, 2), [2, 3), [3, \infty)$에서 $F(x) = 0, 1/8, 4/8, 7/8, 1$이다.

    (c) 확률질량함수로: $p(1) + p(2) = 3/8 + 3/8 = 3/4$.

    누적분포함수로: $X$가 정숫값을 가지므로 $P(1 \le X \le 2) = F(2) - F(1^-) = F(2) - F(0) = 7/8 - 1/8 = 6/8 = 3/4$.

    두 방법이 일치한다.

---

**연습문제 2.**
$[0, 1]$에서 확률밀도함수가 $f(x) = c \cdot x^2$이고 그 밖에서는 0인 연속확률변수 $X$에 대해 (a) $c$를 구하라. (b) $F(x)$를 계산하라. (c) $P(0.3 < X < 0.7)$을 구하라.

??? success "연습문제 2 풀이"
    (a) $\int_0^1 c x^2 \, dx = c/3 = 1$이므로 $c = 3$이다.

    (b) $x \in [0, 1]$에서 $F(x) = \int_0^x 3 t^2 \, dt = x^3$이고, $x < 0$이면 $F(x) = 0$, $x > 1$이면 $F(x) = 1$이다.

    (c) $P(0.3 < X < 0.7) = F(0.7) - F(0.3) = 0.343 - 0.027 = 0.316$.

---

**연습문제 3.**
**누적분포함수를 미분해 확률밀도함수 구하기.** $x \ge 0$에서 $F(x) = 1 - e^{-\lambda x}$인 연속확률변수 $X$에 대해 확률밀도함수 $f(x)$를 계산하라. 이것은 어떤 분포인가?

??? success "연습문제 3 풀이"
    $x \ge 0$에서 $f(x) = F'(x) = \lambda e^{-\lambda x}$이다.

    이는 비율 $\lambda$인 **지수분포**다. 성질은 다음과 같다.
    - 평균 $1/\lambda$, 분산 $1/\lambda^2$.
    - 무기억성: $P(X > s + t \mid X > s) = P(X > t)$.
    - 비율이 $\lambda$인 포아송 과정에서 사건 사이의 대기시간.

---

**연습문제 4.**
**역변환 표집.** $U \sim \mathrm{Uniform}(0, 1)$이고 $F$가 연속인 순증가 누적분포함수이면 $X = F^{-1}(U)$의 누적분포함수가 $F$임을 보여라.

??? success "연습문제 4 풀이"
    $P(X \le x) = P(F^{-1}(U) \le x)$를 계산한다. 양변에 (증가함수라 부등호를 보존하는) $F$를 적용하면

    $$
    P(F^{-1}(U) \le x) = P(F(F^{-1}(U)) \le F(x)) = P(U \le F(x)) = F(x)
    $$

    이다. 여기서 가역인 $F$에 대해 $F \circ F^{-1} = \mathrm{id}$이고, 균등분포의 누적분포함수가 $u \in [0, 1]$에 대해 $P(U \le u) = u$라는 사실을 썼다.

    따라서 $X$의 누적분포함수는 $F$다. $\square$

    **용도:** $F^{-1}$을 아는 어떤 분포에서든 표본을 생성하려면 $U$를 균등하게 뽑아 $F^{-1}$을 적용하면 된다. 자명하지 않은 분포에 대해 여러 난수 생성 루틴이 내부적으로 이렇게 작동한다.

---

**연습문제 5.**
**분위수, 백분위수, 분위수함수.** 이 세 용어를 예를 들어 명확히 구분하라. 분위수함수와 생존함수의 관계를 진술하라.

??? success "연습문제 5 풀이"
    **분위수(quantile):** $p \in [0, 1]$에 대해 $p$-분위수 $q_p = F^{-1}(p)$는 $P(X \le q_p) = p$가 되는 값이다. 분위수 = 분위수함수의 값.

    **백분위수(percentile):** $p \in [0, 100]$에 대한 $p$-백분위수는 $q_{p/100}$이다. 단위 관례일 뿐이다. "95번째 백분위수"는 $q_{0.95}$를 뜻한다.

    **분위수함수(PPF):** 역 누적분포함수 그 자체, 즉 $\mathrm{PPF}(p) = F^{-1}(p)$.

    **생존함수:** $S(x) = 1 - F(x) = P(X > x)$. 역생존함수(ISF)는 "주어진 확률 질량이 그 위에 놓이는 값"을 준다: $\mathrm{ISF}(p) = S^{-1}(p) = F^{-1}(1 - p) = \mathrm{PPF}(1 - p)$.

    scipy.stats에서 `dist.ppf(0.95)`는 95번째 백분위수를 주고, `dist.isf(0.05)`는 위쪽 꼬리 확률이 5%인 값을 주는데 이는 95번째 백분위수와 같다. 꼬리 분위수를 계산할 때 ISF가 수치적 정밀도 손실을 피해 준다($1 - F$가 0에 가까우면 정밀도가 나쁘므로 $S$를 직접 쓰는 편이 낫다).

---

**연습문제 6.**
**이상적분.** $x \ge 2$에서 $f(x) = 1/(x \ln^2 x)$을 확률밀도함수로 제안한다. 이것이 타당한 분포를 정의하는가? $\mathbb{E}[X]$를 계산하라.

??? success "연습문제 6 풀이"
    **정규화:** $\int_2^\infty \frac{1}{x \ln^2 x} dx$를 계산한다. $u = \ln x$, $du = dx/x$로 치환하면

    $$
    \int_{\ln 2}^\infty \frac{1}{u^2} du = \left[-\frac{1}{u}\right]_{\ln 2}^\infty = \frac{1}{\ln 2} \approx 1.443
    $$

    이다. 따라서 $f(x)$는 정규화되어 있지 **않다**. $c = \ln 2$로 두고 $f(x) = c/(x \ln^2 x)$로 다시 정의하면 $\int f = 1$이 되어 타당한 확률밀도함수를 얻는다.

    **기댓값:** $\mathbb{E}[X] = \int_2^\infty x \cdot \frac{c}{x \ln^2 x} dx = c \int_2^\infty \frac{1}{\ln^2 x} dx$이다.

    피적분함수가 $1/\ln^2 x$처럼 감쇠하는데 이는 무한대에서 적분 가능하지 *않다*(적분이 발산한다). 따라서 $\mathbb{E}[X] = \infty$로, 이 분포는 정규화는 유한하지만 평균은 무한하다.

    **교훈:** "타당한 분포"(누적분포함수의 성질이 성립함)는 "유한한 기댓값"보다 약한 요건이다. 이런 꼬리가 두꺼운 분포에는 평균 기반 요약 대신 분위수 기반 요약이 필요하다. 이것이 큰수의 법칙 절 연습문제 5의 주제였다. 평균이 무한한 분포는 큰수의 법칙을 깨뜨린다.
