# 확률질량함수, 확률밀도함수, 누적분포함수

앞의 두 절에서 분포를 적는 방법을 두 가지 배웠다. 이산이면 **확률질량함수**, 연속이면 **확률밀도함수**다. 둘은 성격이 달라 함께 쓰기 불편하다. 하나는 확률이고 하나는 밀도이며, 하나는 더하고 하나는 적분한다.

두 세계를 하나로 묶는 함수가 있다. **누적분포함수** $F(x) = P(X \le x)$다. 이산이든 연속이든 똑같이 정의되고, 이것 하나로 분포가 완전히 결정된다. 그리고 이 함수를 뒤집으면 분위수가 나오고, 분위수에서 난수 생성이 나온다.

이 절은 세 개의 정리로 이루어진다. 누적분포함수의 정의와 성질(정리 1), 확률밀도함수와의 미적분 관계(정리 2), 그리고 역함수인 분위수함수와 그 응용(정리 3)이다.

## 1. 왼쪽부터 무게를 쌓아 나간다

확률질량함수와 확률밀도함수는 "그 지점에" 무게가 얼마인지를 말한다. 누적분포함수는 "그 지점까지" 쌓인 무게가 얼마인지를 말한다. 이 작은 차이가 두 세계를 통일한다.

### 정리 1. 누적분포함수 — 왼쪽부터 쌓은 총 무게

확률변수 $X$의 **누적분포함수(CDF)** 는

$$
F(x) = \mathbb{P}(X \leq x) =
\begin{cases}
\displaystyle\sum_{x_i \leq x} p_{x_i}, & X \text{가 이산일 때} \\[10pt]
\displaystyle\int_{-\infty}^x f(s)\,ds, & X \text{가 연속일 때}
\end{cases}
$$

이며 다음 성질을 갖는다.

- $F$는 **비감소**다.
- $\displaystyle\lim_{x \to -\infty} F(x) = 0$, $\displaystyle\lim_{x \to +\infty} F(x) = 1$
- $F$는 오른쪽 연속이다.

벽돌 비유로 말하면 $F(x)$는 $-\infty$부터 $x$까지 쌓인 **모든 벽돌의 총 무게**다. 왼쪽 끝에서는 아무것도 없어 0이고, 오른쪽 끝에서는 전부 쌓여 1이다. 무게가 음수일 수 없으므로 결코 줄어들지 않는다.

이산과 연속의 차이는 **모양**으로 나타난다. 이산이면 벽돌이 있는 곳에서 계단처럼 뛰고, 연속이면 매끄럽게 오른다. 뛰는 높이가 그 점의 확률이므로, 연속확률변수에서 $F$가 연속이라는 사실이 곧 $P(X = a) = 0$이다.

**분포를 완전히 결정한다.** $F$를 알면 어떤 구간의 확률이든 계산할 수 있다.

$$
P(a < X \leq b) = F(b) - F(a)
$$

확률질량함수와 확률밀도함수는 각각 이산과 연속에서만 쓸 수 있지만, 누적분포함수는 언제나 존재하고 언제나 통한다. 그래서 이론적 논의에서는 $F$를 기본 대상으로 삼는 경우가 많다.

## 2. 밀도와 누적은 미적분으로 이어져 있다

연속인 경우 두 함수는 서로를 완전히 결정한다. 하나를 알면 다른 하나는 미분하거나 적분해서 얻는다.

### 정리 2. 미분과 적분 — 확률밀도함수와 누적분포함수의 왕복

$X$가 연속확률변수이고 $f$가 연속이면

$$
F(x) = \int_{-\infty}^{x} f(s)\,ds
\qquad\Longleftrightarrow\qquad
f(x) = \frac{d}{dx}F(x)
$$

이다. 적분이 밀도에서 누적으로 가고, 미분이 누적에서 밀도로 돌아온다.

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

두 그림의 관계를 눈으로 확인하라. 밀도가 가장 높은 0 부근에서 누적분포함수의 기울기가 가장 가파르다. 밀도가 0에 가까운 양 끝에서는 누적분포함수가 거의 평평하다.

**예: 정규분포에서 구간의 확률.** $X \sim N(50, 10^2)$일 때 $P(40 \le X \le 60)$은 $F(60) - F(40)$이다.

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

## 3. 누적분포함수를 뒤집으면 분위수가 나온다

지금까지는 "값을 주면 확률을 돌려주는" 방향이었다. 실무에서는 반대 방향이 더 자주 필요하다. "확률 95%에 해당하는 값은 얼마인가?"

### 정리 3. 분위수함수 — 누적분포함수의 역함수

**분위수함수(PPF)** 는 누적분포함수의 역함수다.

$$
\text{PPF}(p) = F^{-1}(p) = \inf\{x : F(x) \geq p\}
$$

하한(inf)으로 정의하는 이유는 $F$가 계단이거나 평평한 구간이 있어 엄밀한 역함수가 없을 수 있기 때문이다. 연속이고 순증가하는 $F$에서는 보통의 역함수와 같다.

이 함수가 통계학 전체에서 쓰이는 곳이 신뢰구간과 임계값이다.

```python
import scipy.stats as stats

z_95 = stats.norm(0, 1).ppf(0.95)
print(f"95th percentile of N(0,1): {z_95:.4f}")

z_975 = stats.norm(0, 1).ppf(0.975)
print(f"97.5th percentile of N(0,1): {z_975:.4f}")
```

$1.96$이라는 익숙한 수가 여기서 나온다. 8장의 95% 신뢰구간에 등장하는 그 값이다.

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

세로축에서 출발해 곡선을 만나 가로축으로 내려오는 것이 분위수함수, 그 반대가 누적분포함수다.

!!! tip "역변환 표집: 균등난수 하나로 어떤 분포든 만든다"
    분위수함수에는 아름다운 응용이 하나 있다. $U \sim \text{Uniform}(0,1)$일 때

    $$
    X = F^{-1}(U)
    $$

    로 두면 $X$의 누적분포함수가 정확히 $F$가 된다. **균등난수만 만들 수 있으면 어떤 분포의 난수든 만들 수 있다**는 뜻이다.

    증명은 한 줄이다. $P(X \le x) = P(F^{-1}(U) \le x) = P(U \le F(x)) = F(x)$. 마지막 등식은 $U$가 $[0,1]$ 균등분포이기 때문이다.

    이것이 **역변환 표집**이며, 난수 생성기의 기본 원리다. 4장에서 다시 다룬다.

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

**자료에서 추정하기.** 실무에서는 참 분포를 모르므로 자료에서 추정한다. 히스토그램이 확률밀도함수의 추정값이고, 경험적 누적분포함수가 누적분포함수의 추정값이다. 2장의 탐색적 자료분석에서 다시 만난다.

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

**`scipy.stats` 메서드 대응표.** 이 절의 네 함수가 그대로 메서드 이름이 된다.

| 메서드 | 대응하는 개념 |
|:---|:---|
| `pdf` / `pmf` | 확률밀도함수 / 확률질량함수 |
| `cdf` | 누적분포함수 $F(x) = P(X \le x)$ |
| `sf` | 생존함수 $1 - F(x) = P(X > x)$ |
| `ppf` | 분위수함수 $F^{-1}(p)$ |
| `rvs` | 난수 표본 생성 |

## 연습문제

**연습문제 1.**
$X$ = 공정한 동전을 3번 던졌을 때 앞면의 개수. (a) 확률질량함수를 쓰라. (b) 누적분포함수를 쓰라. (c) $P(1 \le X \le 2)$를 두 가지 방법으로 계산하라.

??? success "풀이"
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

??? success "풀이"
    (a) $\int_0^1 c x^2 \, dx = c/3 = 1$이므로 $c = 3$이다.

    (b) $x \in [0, 1]$에서 $F(x) = \int_0^x 3 t^2 \, dt = x^3$이고, $x < 0$이면 $F(x) = 0$, $x > 1$이면 $F(x) = 1$이다.

    (c) $P(0.3 < X < 0.7) = F(0.7) - F(0.3) = 0.343 - 0.027 = 0.316$.

---

**연습문제 3.**
**누적분포함수를 미분해 확률밀도함수 구하기.** $x \ge 0$에서 $F(x) = 1 - e^{-\lambda x}$인 연속확률변수 $X$에 대해 확률밀도함수 $f(x)$를 계산하라. 이것은 어떤 분포인가?

??? success "풀이"
    $x \ge 0$에서 $f(x) = F'(x) = \lambda e^{-\lambda x}$이다.

    이는 비율 $\lambda$인 **지수분포**다. 성질은 다음과 같다.
    - 평균 $1/\lambda$, 분산 $1/\lambda^2$.
    - 무기억성: $P(X > s + t \mid X > s) = P(X > t)$.
    - 비율이 $\lambda$인 포아송 과정에서 사건 사이의 대기시간.

---

**연습문제 4.**
**역변환 표집.** $U \sim \mathrm{Uniform}(0, 1)$이고 $F$가 연속인 순증가 누적분포함수이면 $X = F^{-1}(U)$의 누적분포함수가 $F$임을 보여라.

??? success "풀이"
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

??? success "풀이"
    **분위수(quantile):** $p \in [0, 1]$에 대해 $p$-분위수 $q_p = F^{-1}(p)$는 $P(X \le q_p) = p$가 되는 값이다. 분위수 = 분위수함수의 값.

    **백분위수(percentile):** $p \in [0, 100]$에 대한 $p$-백분위수는 $q_{p/100}$이다. 단위 관례일 뿐이다. "95번째 백분위수"는 $q_{0.95}$를 뜻한다.

    **분위수함수(PPF):** 역 누적분포함수 그 자체, 즉 $\mathrm{PPF}(p) = F^{-1}(p)$.

    **생존함수:** $S(x) = 1 - F(x) = P(X > x)$. 역생존함수(ISF)는 "주어진 확률 질량이 그 위에 놓이는 값"을 준다: $\mathrm{ISF}(p) = S^{-1}(p) = F^{-1}(1 - p) = \mathrm{PPF}(1 - p)$.

    scipy.stats에서 `dist.ppf(0.95)`는 95번째 백분위수를 주고, `dist.isf(0.05)`는 위쪽 꼬리 확률이 5%인 값을 주는데 이는 95번째 백분위수와 같다. 꼬리 분위수를 계산할 때 ISF가 수치적 정밀도 손실을 피해 준다($1 - F$가 0에 가까우면 정밀도가 나쁘므로 $S$를 직접 쓰는 편이 낫다).

---

**연습문제 6.**
**이상적분.** $x \ge 2$에서 $f(x) = 1/(x \ln^2 x)$을 확률밀도함수로 제안한다. 이것이 타당한 분포를 정의하는가? $\mathbb{E}[X]$를 계산하라.

??? success "풀이"
    **정규화:** $\int_2^\infty \frac{1}{x \ln^2 x} dx$를 계산한다. $u = \ln x$, $du = dx/x$로 치환하면

    $$
    \int_{\ln 2}^\infty \frac{1}{u^2} du = \left[-\frac{1}{u}\right]_{\ln 2}^\infty = \frac{1}{\ln 2} \approx 1.443
    $$

    이다. 따라서 $f(x)$는 정규화되어 있지 **않다**. $c = \ln 2$로 두고 $f(x) = c/(x \ln^2 x)$로 다시 정의하면 $\int f = 1$이 되어 타당한 확률밀도함수를 얻는다.

    **기댓값:** $\mathbb{E}[X] = \int_2^\infty x \cdot \frac{c}{x \ln^2 x} dx = c \int_2^\infty \frac{1}{\ln^2 x} dx$이다.

    피적분함수가 $1/\ln^2 x$처럼 감쇠하는데 이는 무한대에서 적분 가능하지 *않다*(적분이 발산한다). 따라서 $\mathbb{E}[X] = \infty$로, 이 분포는 정규화는 유한하지만 평균은 무한하다.

    **교훈:** "타당한 분포"(누적분포함수의 성질이 성립함)는 "유한한 기댓값"보다 약한 요건이다. 이런 꼬리가 두꺼운 분포에는 평균 기반 요약 대신 분위수 기반 요약이 필요하다. 이것이 큰수의 법칙 절 연습문제 5의 주제였다. 평균이 무한한 분포는 큰수의 법칙을 깨뜨린다.

## 정리하며

누적분포함수는 이산과 연속을 하나로 묶는 함수다.

- **정리 1**은 $F(x) = P(X \le x)$를 정의했다. 왼쪽부터 쌓은 총 무게이며, 이산이면 계단으로 뛰고 연속이면 매끄럽게 오른다. 어느 쪽이든 분포를 완전히 결정한다.
- **정리 2**는 연속인 경우 밀도와 누적이 미분·적분으로 왕복함을 보였다.
- **정리 3**은 $F$를 뒤집어 **분위수함수**를 얻었다. 신뢰구간의 $1.96$이 여기서 나오고, 역변환 표집으로 난수 생성까지 이어진다.

3.3절 전체를 한 줄로 줄이면 이렇다. **확률변수는 결과를 수로 옮기고, 그 결과 실직선 위에 무게 배치가 생기며, 그것을 적는 방법이 세 가지 함수다.** 값마다의 무게(확률질량함수), 구간당 무게(확률밀도함수), 왼쪽부터 쌓은 무게(누적분포함수).

이제 분포를 적을 수 있게 되었으니 다음 물음이 자연스럽다. **분포를 몇 개의 수로 요약할 수 없을까?** 주사위 눈의 분포 전체를 말하는 대신 "평균 3.5"라고 말하는 것처럼.

다음 절의 **기댓값**이 그 첫 번째 요약값이고, 이어지는 **분산**이 두 번째다. 그리고 이 요약이 통계학이 자료를 다루는 방식 전체의 출발점이 된다.
