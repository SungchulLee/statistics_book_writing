# Normal 분포

## 개요

**Normal 분포**(Gaussian 분포라고도 한다)는 통계학에서 가장 근본적인 확률분포 중 하나이다. 중심값 주위에 모여 양쪽으로 대칭적으로 잦아드는 특유의 "종 모양 곡선"을 이루는 연속 자료를 기술한다.

---

## 정규분포와 표준정규분포

<div class="defn" markdown>

### 정의 1. 정규분포 { .dfn }

정규분포는 평균 $\mu$(중심)와 분산 $\sigma^2$(퍼짐)로 규정된다. PDF는 다음과 같다:

$$
f(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

이를 $X \sim N(\mu, \sigma^2)$로 쓴다.

</div>

### 표준정규분포

$\mu = 0$, $\sigma = 1$인 특수한 경우를 **표준정규분포**라 한다:

$$
Z \sim N(0, 1), \qquad f(z) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{z^2}{2}\right)
$$

---

## 표준화

모든 정규확률변수는 **Z-점수 변환**을 통해 표준정규확률변수로 바꿀 수 있다:

$$
\begin{aligned}
\textbf{Standardization:} \quad & X \sim N(\mu, \sigma^2) \implies Z = \frac{X - \mu}{\sigma} \sim N(0, 1) \\[6pt]
\textbf{Reverse:} \quad & Z \sim N(0, 1) \implies X = Z\sigma + \mu \sim N(\mu, \sigma^2)
\end{aligned}
$$

---

## 정규분포의 성질

### 닫힘 성질

$$
\begin{aligned}
(1) &\quad X \sim \text{Normal} \implies aX + b \sim \text{Normal} \\[4pt]
(2) &\quad X \sim \text{Normal}, \; Y \sim \text{Normal}, \; X \perp Y \implies X + Y \sim \text{Normal} \\[4pt]
(3) &\quad (X, Y) \sim \text{Multivariate Normal} \implies X + Y \sim \text{Normal}
\end{aligned}
$$

**주의:** $X \sim \text{Normal}$이고 $Y \sim \text{Normal}$이라고 해서 $X + Y \sim \text{Normal}$인 것은 **아니다**. 독립성이나 결합정규성이 있어야 한다.

### 성질 (1)의 증명

$a > 0$이고 $X \sim N(\mu, \sigma^2)$일 때:

$$
P(aX + b \leq x) = P\left(X \leq \frac{x - b}{a}\right) = \int_{-\infty}^{(x-b)/a} \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(s-\mu)^2}{2\sigma^2}} ds
$$

$x$에 대해 미분하면:

$$
f_{aX+b}(x) = \frac{1}{\sqrt{2\pi(a\sigma)^2}} \exp\left(-\frac{(x - (a\mu + b))^2}{2a^2\sigma^2}\right)
$$

따라서 $aX + b \sim N(a\mu + b, \, a^2\sigma^2)$이다.

### 주요 기하적 성질

- **대칭성:** $\mu$를 중심으로 완전히 대칭이며, 평균 = 중앙값 = 최빈값 = $\mu$이다.
- **종 모양:** 대부분의 자료가 평균 근처에 몰려 있다.
- **무한한 꼬리:** 꼬리는 $\pm\infty$까지 뻗지만 확률은 빠르게 감소한다.

### 68–95–99.7 규칙

$$
\begin{aligned}
P(\mu - \sigma < X < \mu + \sigma) &\approx 68\% \\
P(\mu - 2\sigma < X < \mu + 2\sigma) &\approx 95\% \\
P(\mu - 3\sigma < X < \mu + 3\sigma) &\approx 99.7\%
\end{aligned}
$$

---

## 표준정규분포의 PDF: 주요 성질 확인

$N(0, 1)$의 PDF는 $f(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$이다. 다음을 확인한다:

### (1) 전체 질량이 1이다

$I = \int_{-\infty}^{\infty} e^{-x^2/2}\,dx$라 하자. 그러면:

$$
I^2 = \int\!\!\int e^{-(x^2+y^2)/2}\,dx\,dy = \int_0^{2\pi}\!\int_0^{\infty} e^{-r^2/2}\,r\,dr\,d\theta = 2\pi
$$

따라서 $I = \sqrt{2\pi}$이고 $\int f(x)\,dx = 1$임이 확인된다.

### (2) 평균이 0이다

피적분함수 $x \cdot e^{-x^2/2}$는 **기함수**이므로 $(-\infty, \infty)$ 위의 적분은 0이다.

### (3) 분산이 1이다

부분적분에 의해:

$$
\frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty} x^2 e^{-x^2/2}\,dx = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty} e^{-x^2/2}\,dx = 1
$$

---

## 표준정규분포의 CDF

CDF는 닫힌 형태가 없어 수치적으로 계산한다:

$$
\mathcal{N}(x) = N(x) = \int_{-\infty}^x \frac{1}{\sqrt{2\pi}} e^{-s^2/2}\,ds
$$

### Phi의 성질

$$
\begin{aligned}
(1) &\quad P(a \leq Z \leq b) = \mathcal{N}(b) - \mathcal{N}(a) \\
(2) &\quad P(Z \geq x) = P(Z \leq -x) = \mathcal{N}(-x) \\
(3) &\quad P(Z \geq x) = 1 - \mathcal{N}(x) \\
(4) &\quad P(Z \leq 0) = P(Z \geq 0) = 0.5
\end{aligned}
$$

---

## 정규 PDF와 관련된 적분 요령

<div class="probox" markdown>

**문제:** <span class="diff med" title="중간"></span> $\int_{-\infty}^{\infty} e^{-x^2 - 2x}\,dx$를 계산하라.

</div>

??? success "풀이"
    완전제곱식으로 만든다: $-x^2 - 2x = -(x+1)^2 + 1$. 그러면:

    $$
    \int_{-\infty}^{\infty} e^{-x^2-2x}\,dx = e \int_{-\infty}^{\infty} e^{-(x+1)^2}\,dx = e\sqrt{2\pi \cdot \tfrac{1}{2}} \cdot \underbrace{\int \frac{1}{\sqrt{\pi}} e^{-(x+1)^2}\,dx}_{=1 \text{ (PDF of } N(-1, 1/2))} = e\sqrt{\pi}
    $$
---

## Python: PDF, CDF 그리기와 표본추출

### PDF와 CDF

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

mu, sigma = 0, 1                  # 표준정규분포
# 평균에서 좌우 3 표준편차. 확률의 99.7%가 이 안에 있다.
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 200)

fig, ax = plt.subplots(figsize=(12, 3))
# 두 함수를 같은 축에 겹쳐 관계를 본다.
#   PDF는 평균에서 가장 높고 좌우로 떨어진다.
#   CDF는 0에서 1로 단조 증가하며, PDF가 가장 높은 곳에서 가장 가파르다.
# CDF의 기울기가 곧 PDF이기 때문이다.
ax.plot(x, stats.norm(mu, sigma).pdf(x), label='PDF')
ax.plot(x, stats.norm(mu, sigma).cdf(x), label='CDF')
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
plt.show()
```

![Normal 분포](./img/normal_155.png)

### 표본추출과 추정된 PDF

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(0)
data = stats.norm(loc=0, scale=1).rvs(10_000)     # 참 모수는 (0, 1)

fig, ax = plt.subplots(figsize=(12, 3))
# density=True 로 넓이를 1로 맞춰야 밀도곡선과 같은 눈금에 놓인다
_, bins, _ = ax.hist(data, bins=100, density=True, color='blue', alpha=0.7, label="Samples")
# 참 모수 (0, 1)이 아니라 **표본에서 추정한** 평균과 표준편차로 곡선을 그린다.
# 실제 분석에서는 참값을 모르기 때문이다.
ax.plot(bins, stats.norm(data.mean(), data.std()).pdf(bins),
        '--r', lw=3, label="Estimated Normal PDF")
ax.legend()
plt.show()
```

![Normal 분포](./img/normal_173.png)

### 68–95–99.7 규칙 확인

```python
import pandas as pd

url = 'https://raw.githubusercontent.com/gedeck/practical-statistics-for-data-scientists/master/data/loans_income.csv'
df = pd.read_csv(url)
mean, std, n = df.x.mean(), df.x.std(), len(df.x)

n1 = len(df.x[(mean - std < df.x) & (df.x < mean + std)])
n2 = len(df.x[(mean - 2*std < df.x) & (df.x < mean + 2*std)])
n3 = len(df.x[(mean - 3*std < df.x) & (df.x < mean + 3*std)])

print(f"Within 1σ: {n1/n*100:.2f}%")   # ≈ 68%
print(f"Within 2σ: {n2/n*100:.2f}%")   # ≈ 95%
print(f"Within 3σ: {n3/n*100:.2f}%")   # ≈ 99.7%
```

출력:

```
Within 1σ: 72.69%
Within 2σ: 95.00%
Within 3σ: 98.66%
```

---

## 표준정규곡선 아래의 넓이

### scipy.stats 메서드

| 메서드 | 설명 |
|:---|:---|
| `rvs` | 확률표본 생성 |
| `pdf` | PDF 계산 |
| `cdf` | $P(X \leq x)$ 계산 |
| `sf` | 생존함수: $1 - \text{cdf}(x)$ |
| `ppf` | 백분위점 함수(CDF의 역함수) |

### 왼쪽 꼬리, 오른쪽 꼬리, 가운데 넓이

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def shade_area(z_bounds, side='left', ax=None):
    """Shade a region under the standard normal curve."""
    x = np.linspace(-4, 4, 200)
    ax.plot(x, stats.norm().pdf(x), color='k', alpha=0.9)

    if side == 'left':
        x_shade = np.linspace(-4, z_bounds, 200)
    elif side == 'right':
        x_shade = np.linspace(z_bounds, 4, 200)
    else:  # center
        x_shade = np.linspace(z_bounds[0], z_bounds[1], 200)

    ax.fill_between(x_shade, stats.norm().pdf(x_shade), alpha=0.2, color='k')
    ax.spines[['left', 'right', 'top']].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.set_yticks([])

# Left area
z = -1.2
print(f"P(Z ≤ {z}) = {stats.norm().cdf(z):.4f}")

# Right area
z = 1.2
print(f"P(Z ≥ {z}) = {stats.norm().sf(z):.4f}")

# Center area
z1, z2 = -2.1, 1.2
print(f"P({z1} ≤ Z ≤ {z2}) = {stats.norm().cdf(z2) - stats.norm().cdf(z1):.4f}")
```

출력:

```
P(Z ≤ -1.2) = 0.1151
P(Z ≥ 1.2) = 0.1151
P(-2.1 ≤ Z ≤ 1.2) = 0.8671
```

---

## 왜 정규분포인가?

중심극한정리는 정규분포가 어디에나 나타나는 이유를 설명한다. 원래 모집단의 분포가 무엇이든, $n$이 크면 표본평균의 분포는 근사적으로 정규분포이다:

$$
\bar{X} \sim N\left(\mu, \frac{\sigma^2}{n}\right) \quad \text{as } n \to \infty
$$

이 때문에 정규분포는 신뢰구간, 가설검정, 품질관리의 기초가 된다.

---

## 난수 시드 고정하기

`scipy.stats`는 NumPy의 난수 생성기를 사용하므로 `np.random.seed()`를 설정하면 재현성이 보장된다:

```python
import numpy as np
import scipy.stats as stats

np.random.seed(42)
samples = stats.norm.rvs(size=10)
print(samples)  # Same output every time with seed 42
```

출력:

```
[ 0.49671415 -0.1382643   0.64768854  1.52302986 -0.23415337 -0.23413696
  1.57921282  0.76743473 -0.46947439  0.54256004]
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
점수가 $X \sim N(70, 100)$이다. (a) $P(60 < X < 80)$. (b) 90 백분위수. (c) $n = 200$명 중 85점을 넘는 학생 수의 기댓값. (d) $(X - 70)/10$의 분포.

</div>

??? success "풀이"
    (a) 표준화하면 $P(-1 < Z < 1) = 0.8413 - 0.1587 = 0.6827$.

    (b) $x_{0.90} = 70 + 1.2816 \cdot 10 = 82.82$.

    (c) $P(X > 85) = P(Z > 1.5) = 0.0668$. 기댓값은 $200 \cdot 0.0668 \approx 13.4 \approx 13$명.

    (d) 표준화에 의해 $Y = (X - 70)/10 \sim N(0, 1)$. 따라서 $P(X > 85) = P(Y > 1.5)$.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
**경험적 (68-95-99.7) 규칙.** $Z \sim N(0, 1)$일 때 $k = 1, 2, 3$에 대해 $P(|Z| \le k) \approx 0.683, 0.954, 0.997$임을 보여라.

</div>

??? success "풀이"
    표준정규분포표에서 $P(Z \le 1) = 0.8413$이므로 $P(|Z| \le 1) = 2 \cdot 0.8413 - 1 = 0.6827$.

    같은 방식으로 $P(|Z| \le 2) = 2 \cdot 0.9772 - 1 = 0.9545$.

    $P(|Z| \le 3) = 2 \cdot 0.9987 - 1 = 0.9973$.

    **함의:**

    - "2시그마 사건"의 확률은 $\approx 5\%$ — 유의성의 기준.
    - "3시그마 사건"의 확률은 $\approx 0.3\%$ — 강한 증거.
    - "5시그마"(물리학의 기준): $P(|Z| > 5) \approx 5.7 \times 10^{-7}$.

    이 문턱값들은 "귀무가설 아래에서 이 관측이 얼마나 드문가"에 대한 질적 기준을 이룬다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
**독립인 정규확률변수의 선형결합.** $X_1 \sim N(\mu_1, \sigma_1^2)$, $X_2 \sim N(\mu_2, \sigma_2^2)$이 독립이다. $aX_1 + bX_2 + c$의 분포를 구하라.

</div>

??? success "풀이"
    MGF를 이용하면 $M_{aX_1 + bX_2 + c}(t) = e^{ct} M_{X_1}(at) M_{X_2}(bt) = e^{ct} \exp(a\mu_1 t + a^2\sigma_1^2 t^2/2) \exp(b\mu_2 t + b^2\sigma_2^2 t^2/2)$.

    $= \exp\!\left((c + a\mu_1 + b\mu_2)t + (a^2\sigma_1^2 + b^2\sigma_2^2) t^2/2\right)$.

    이는 $N(a\mu_1 + b\mu_2 + c, a^2\sigma_1^2 + b^2\sigma_2^2)$의 MGF이다.

    따라서 $aX_1 + bX_2 + c \sim N(a\mu_1 + b\mu_2 + c, a^2\sigma_1^2 + b^2\sigma_2^2)$이다. 정규분포족은 선형결합에 대해 **닫혀 있으며**, 이는 정규분포를 규정하는 성질이다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
**표준화**는 임의의 정규확률변수를 표준정규확률변수로 바꾼다. $X \sim N(\mu, \sigma^2)$에 대해 $\Phi^{-1}(F_X(x)) = (x - \mu)/\sigma$임을 증명하라.

</div>

??? success "풀이"
    $X \sim N(\mu, \sigma^2)$에 대해:

    $F_X(x) = P(X \le x) = P((X - \mu)/\sigma \le (x - \mu)/\sigma) = \Phi((x - \mu)/\sigma)$.

    양변에 $\Phi^{-1}$을 적용하면 $\Phi^{-1}(F_X(x)) = (x - \mu)/\sigma$. $\square$

    **활용:** **분위수-분위수(Q-Q) 그림**은 표본분위수를 대응하는 표준정규분위수에 대해 그린다. 자료가 어떤 평균과 분산을 갖든 정규분포를 따른다면 점들은 기울기 $\sigma$, 절편 $\mu$인 직선 위에 놓인다. 정규성을 시각적으로 점검하면서 모수까지 읽어 낼 수 있다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
**정규분포의 최대가능도추정.** 두 모수가 모두 미지인 i.i.d. 표본 $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$이 주어졌을 때 MLE를 유도하라.

</div>

??? success "풀이"
    로그가능도:

    $$
    \ell(\mu, \sigma^2) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum (X_i - \mu)^2
    $$

    $\mu$에 대한 편미분: $\partial \ell/\partial \mu = \sum(X_i - \mu)/\sigma^2 = 0 \Rightarrow \hat\mu = \bar X$.

    $\sigma^2$에 대한 편미분: $\partial \ell/\partial \sigma^2 = -n/(2\sigma^2) + \sum(X_i - \mu)^2/(2\sigma^4) = 0 \Rightarrow \hat\sigma^2 = (1/n)\sum(X_i - \hat\mu)^2$.

    두 MLE 모두 닫힌 형태로 주어진다. 분산의 MLE는 $n - 1$이 아니라 $n$으로 나누므로 편향되어 있다($\mathbb{E}[\hat\sigma^2_{\text{MLE}}] = \frac{n-1}{n}\sigma^2$). 불편추정을 하려면 분모를 $n - 1$로 하는 Bessel 수정을 사용한다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
**Q-Q 그림의 해석.** 어떤 표본을 표준정규분포에 대해 그린 Q-Q 그림에서 오른쪽 꼬리의 점들이 기준선 아래에 놓인다. 이 양상을 해석하라.

</div>

??? success "풀이"
    Q-Q 그림의 $y$축은 표본분위수이고 $x$축은 표준정규분위수이다. 기준선 $y = \mu + \sigma x$는 자료가 정규분포를 따를 때 점들이 놓일 위치를 나타낸다.

    **"오른쪽 꼬리에서 선 아래"**라는 것은 $x$가 큰 양수일 때(표준정규분위수가 클 때) 표본의 분위수가 선이 예측하는 값보다 *작다*는 뜻이다. 다시 말해 표본의 상단 극단값들이 정규분포에서 기대되는 것만큼 극단적이지 않으며, **오른쪽 꼬리가 정규분포보다 얇다**.

    이는 **가벼운 꼬리** 분포(예: 균등분포, 유계 지지집합 위의 베타분포, 절단정규분포)를 시사한다. 반대 양상, 즉 오른쪽 꼬리에서 점들이 선 위에 놓이면 **두꺼운 꼬리**(예: $t$ 분포, 로그정규분포)를 뜻한다.

    진단 양상:

    | 양상 | 분포 |
    |---|---|
    | 직선 | 정규분포 |
    | S자 곡선 | 양쪽 꼬리가 가벼움 |
    | 역 S자 | 양쪽 꼬리가 두꺼움 |
    | 아래로 볼록 | 오른쪽으로 치우침 |
    | 위로 볼록 | 왼쪽으로 치우침 |

    Q-Q 그림은 모형이 *어디서* 잘 맞고 어디서 어긋나는지를 보여 주므로 적합도 검정의 $p$ 값 하나보다 훨씬 많은 정보를 준다.

---

## 정리하며

- 정규분포 $N(\mu, \sigma^2)$는 종 모양, 대칭성, 68–95–99.7 규칙으로 특징지어진다.
- 표준정규분포 $N(0,1)$은 Z-점수 표준화를 통해 보편적인 기준 역할을 한다.
- 독립인 정규확률변수의 선형변환과 합은 여전히 정규분포이다.
- CDF는 닫힌 형태가 없지만 수치적으로 효율적으로 계산된다.
- 중심극한정리는 정규분포가 자연과 통계학에서 그토록 자주 나타나는 이유를 설명한다.
