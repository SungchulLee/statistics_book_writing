# 독립성과 무상관성의 차이

## 개요

무상관인 확률변수는 독립이라는 오해가 흔하다. **독립이면 상관계수가 0이지만**, 그 역은 일반적으로 **성립하지 않는다**. 이 절에서는 증명과 반례, 그리고 두 개념이 일치하는 특수한 경우를 통해 이 구별을 명확히 한다.

---

## 독립성과 무상관성


<div class="defn" markdown>

### 정의 1. 독립성 { .dfn }

$X$와 $Y$가 **독립**($X \perp Y$)이라는 것은 다음을 뜻한다:

$$
P(X \in A, Y \in B) = P(X \in A) \cdot P(Y \in B) \quad \text{for all sets } A, B
$$

동등하게, 결합 밀도/PMF가 인수분해된다: $f_{X,Y}(x,y) = f_X(x) \cdot f_Y(y)$.

</div>

<div class="defn" markdown>

### 정의 2. 무상관성 { .dfn }

$X$와 $Y$가 **무상관**이라는 것은 다음을 뜻한다:

$$
\text{Cov}(X, Y) = E[XY] - E[X]E[Y] = 0
$$

동등하게, $\rho(X,Y) = 0$ 또는 $E[XY] = E[X]E[Y]$이다.

---

</div>

## 독립이면 상관계수가 0이다

**정리:** $X \perp Y$이면 $\text{Cov}(X,Y) = 0$이다.

??? proof "증명"


    $X$와 $Y$가 독립이면 $E[XY] = E[X] \cdot E[Y]$이다(곱의 기댓값이 기댓값의 곱과 같다). 따라서:

    $$
    \text{Cov}(X,Y) = E[XY] - E[X]E[Y] = E[X]E[Y] - E[X]E[Y] = 0
    $$

    더 일반적으로, 독립성은 **모든** 가측함수 $g, h$에 대해 $E[g(X)h(Y)] = E[g(X)]E[h(Y)]$임을 함의한다. 무상관성은 $g(x) = x$와 $h(y) = y$인 경우만 요구한다.

    ---

## 상관계수가 0이어도 독립은 아니다

### 반례 1: 대칭적인 비선형 의존성

$X \sim N(0, 1)$이고 $Y = X^2$이라 하자. 그러면 $Y$는 $X$에 의해 완전히 결정되지만(최대한의 의존성), 다음이 성립한다:

$$
\text{Cov}(X, Y) = E[XY] - E[X]E[Y] = E[X^3] - 0 \cdot E[X^2] = 0
$$

표준정규분포의 대칭성에 의해 $E[X^3] = 0$이기 때문이다.

**왜 이렇게 되는가:** 상관계수는 **선형** 의존성만 측정한다. $Y = X^2$이라는 관계는 완전히 비선형이면서 대칭이므로, 공분산 계산에서 양의 편차와 음의 편차가 상쇄된다.

### 반례 2: 이산형 예

$X \sim \text{Uniform}\{-1, 0, 1\}$이고 $Y = |X|$라 하자:

| $X$ | $Y = \|X\|$ | $P$ |
|:---:|:---:|:---:|
| $-1$ | $1$ | $1/3$ |
| $0$ | $0$ | $1/3$ |
| $1$ | $1$ | $1/3$ |

$$
E[X] = 0, \quad E[Y] = \tfrac{2}{3}, \quad E[XY] = (-1)(1)\tfrac{1}{3} + 0 + (1)(1)\tfrac{1}{3} = 0
$$

$$
\text{Cov}(X,Y) = 0 - 0 \cdot \tfrac{2}{3} = 0
$$

그러나 $X$와 $Y$는 **독립이 아니다**: $P(Y = 0 \mid X = 0) = 1 \neq P(Y = 0) = 1/3$이다.

### 반례 3: 단위원

$(X, Y)$가 단위원 위에 균등하게 분포한다고 하자. 대칭성에 의해 $\text{Cov}(X,Y) = 0$이지만, $X^2 + Y^2 = 1$이므로 둘은 완전히 의존적이다.

---

## 언제 두 개념이 동등한가?

### 결합정규 확률변수

**정리:** $(X, Y)$가 **이변량 정규분포**를 따르면:

$$
\text{Cov}(X, Y) = 0 \iff X \perp Y
$$

이는 다변량 정규분포의 특별하고도 매우 중요한 성질이다. 이변량 정규 PDF는 다음과 같다:

$$
f(x,y) = \frac{1}{2\pi\sigma_X\sigma_Y\sqrt{1-\rho^2}} \exp\left(-\frac{1}{2(1-\rho^2)}\left[\frac{(x-\mu_X)^2}{\sigma_X^2} - \frac{2\rho(x-\mu_X)(y-\mu_Y)}{\sigma_X\sigma_Y} + \frac{(y-\mu_Y)^2}{\sigma_Y^2}\right]\right)
$$

$\rho = 0$이면 교차항이 사라지고 결합 PDF가 두 주변 정규 PDF의 곱으로 인수분해된다.

### 이항 확률변수

각각 두 값만 취하는 확률변수에 대해서도 공분산이 0이면 독립이다.

---

## 요약 도식

$$
\boxed{
\text{Independence} \implies \text{Zero Correlation} \implies E[XY] = E[X]E[Y]
}
$$

$$
\text{Zero Correlation} \;\not\!\!\!\implies \text{Independence} \quad \text{(일반적으로)}
$$

$$
\text{Zero Correlation} \iff \text{Independence} \quad \text{(결합정규일 때)}
$$

---

## 의존성 개념의 위계

강한 것부터 약한 것 순으로:

$$
\begin{aligned}
&\textbf{Functional dependence:} \quad Y = g(X) \\[4pt]
&\textbf{Statistical dependence:} \quad f_{X,Y} \neq f_X \cdot f_Y \\[4pt]
&\textbf{Correlation:} \quad \rho \neq 0 \\[4pt]
&\textbf{Uncorrelated:} \quad \rho = 0 \\[4pt]
&\textbf{Independence:} \quad f_{X,Y} = f_X \cdot f_Y
\end{aligned}
$$

상관계수는 선형적인 양상을 잡아낸다. 의존성은 어떤 형태든 취할 수 있다. 어떤 변수가 다른 변수의 함수로 완전히 결정되면서도 무상관일 수 있다(위의 반례들이 보여 준 대로).

---

## Python: 차이를 보이기

### 무상관이지만 의존적인 경우: Y = X의 제곱

<div class="codebox" markdown>

#### 예제 1. 무상관인데 독립이 아닌 경우 { .eg }

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 100_000
X = np.random.normal(0, 1, n)
Y = X**2        # X만 알면 Y가 완전히 결정된다. 이보다 강한 종속은 없다.

# 그런데 상관계수는 0이 나온다.
# Cov(X, X^2) = E[X^3] - E[X]E[X^2] 인데, X가 0을 중심으로 대칭이면
# E[X] = 0 이고 E[X^3] = 0 이므로 공분산이 정확히 0이다.
# 상관은 **직선 관계만** 재기 때문에 포물선 관계를 전혀 보지 못한다.
corr = np.corrcoef(X, Y)[0, 1]
print(f"Correlation(X, X²) = {corr:.6f}")
print(f"But Y is completely determined by X!")

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(X[:2000], Y[:2000], s=2, alpha=0.3)
ax.set_xlabel('X')
ax.set_ylabel('Y = X²')
ax.set_title(f'Uncorrelated (ρ={corr:.4f}) but Dependent')
ax.spines[['top', 'right']].set_visible(False)
plt.show()
```

출력:

```
Correlation(X, X²) = 0.000122
But Y is completely determined by X!
```

![독립성과 무상관성의 차이](./img/independence_vs_zero_corr_153.png)

</div>

### 실제 자료에서: 태양 흑점

$Y = X^2$은 만들어 낸 예다. 실제로 관측한 자료에서도 같은 일이 일어나는지 보는 편이 설득력이 있다. 1700년부터 2008년까지 기록된 **연평균 태양 흑점 수**를 쓴다. 천문학에서 가장 오래 이어진 관측 기록 가운데 하나이고, 대략 11년을 주기로 오르내리는 것으로 잘 알려져 있다.

올해의 흑점 수와 $k$년 뒤의 흑점 수를 짝지어 상관계수를 재 보자.

<div class="codebox" markdown>

#### 예제 2. 무상관이지만 독립이 아닌 실제 자료 { .eg }

```python
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 1700~2008년 연평균 태양 흑점 수. statsmodels에 함께 배포되는 실제 관측 자료다.
sun = sm.datasets.sunspots.load_pandas().data
x = sun["SUNACTIVITY"].values

# 올해 값과 k년 뒤 값을 짝지어 상관계수를 잰다.
print(f"{'시차(년)':>8}{'상관계수':>10}")
for lag in (1, 3, 5, 11):
    print(f"{lag:>8}{np.corrcoef(x[:-lag], x[lag:])[0, 1]:>10.3f}")

# 시차 3년에서 상관이 사실상 0이다. 독립일까?
lag = 3
today, later = x[:-lag], x[lag:]

# 오늘의 흑점 수를 5분위로 나누고, 각 구간에서 3년 뒤 평균을 본다.
edges = np.quantile(today, [0, .2, .4, .6, .8, 1.0])
print(f"\n{'오늘의 흑점 수':>18}{'3년 뒤 평균':>12}")
for i in range(5):
    lo, hi = edges[i], edges[i + 1]
    m = (today >= lo) & (today <= hi) if i == 4 else (today >= lo) & (today < hi)
    print(f"{f'{lo:6.1f} ~ {hi:6.1f}':>18}{later[m].mean():12.1f}")
print(f"{'전체 평균':>18}{later.mean():12.1f}")
```

출력:

```
   시차(년)      상관계수
       1     0.824
       3     0.040
       5    -0.430
      11     0.672

          오늘의 흑점 수     3년 뒤 평균
      0.0 ~   12.4        54.0
     12.4 ~   30.6        55.8
     30.6 ~   52.2        42.7
     52.2 ~   82.9        38.1
     82.9 ~  190.2        60.1
             전체 평균        50.1
```

![태양 흑점: 무상관이지만 독립이 아니다](./img/independence_vs_zero_corr_sunspots.png)

</div>

시차별 상관계수부터 읽어 보자. 1년 뒤와는 $0.824$로 강하게 붙어 있고, 11년 뒤와는 $0.672$로 다시 붙는다. 한 주기를 돌아 같은 국면으로 돌아왔기 때문이다. 5년 뒤와는 $-0.430$인데, 반주기쯤 지나 극대가 극소와 마주 보는 자리다.

문제는 **시차 3년**이다. 상관계수가 $0.040$으로 사실상 0이다. 여기서 "3년 뒤 흑점 수는 올해와 무관하다"고 말하고 싶어지는데, 태양이 11년 주기로 돈다는 것을 아는 이상 그럴 리가 없다.

오른쪽 그림의 붉은 선이 답을 준다. 오늘의 흑점 수로 자료를 다섯 구간으로 나누고 각 구간에서 3년 뒤 평균을 찍은 것인데, **아래로 볼록한 U자**를 그린다. 오늘이 아주 적으면($0 \sim 12$) 3년 뒤는 $54.0$으로 평균보다 높고, 오늘이 중간쯤이면($52 \sim 83$) 3년 뒤는 $38.1$로 평균보다 낮으며, 오늘이 아주 많으면($83$ 이상) 3년 뒤는 $60.1$로 다시 높다.

이유는 주기에 있다. 흑점이 아주 적다는 것은 **골짜기 근처**라는 뜻이고 3년 뒤면 올라가는 중이다. 아주 많다는 것은 **봉우리 근처**라는 뜻인데 3년 뒤면 아직 높은 수준이 남아 있다. 중간값은 오르는 길일 수도 내리는 길일 수도 있어서, 두 경우가 섞이며 평균이 낮게 나온다.

$Y = X^2$에서 본 것과 **정확히 같은 구조**다. 관계가 U자라서 직선으로 요약하면 올라가는 쪽과 내려가는 쪽이 서로 상쇄되고, 상관계수가 0에 가깝게 나온다. 상관계수가 못 보는 것이지 관계가 없는 것이 아니다.

!!! warning "시계열에서 특히 조심할 것"

    "상관이 0이니 독립이다"라는 논리는 시계열에서 자주 깨진다. 주기가 있거나 변동성이 몰려다니는 자료에서는 값 자체의 자기상관이 0에 가까운데도 **크기**나 **제곱**의 자기상관이 뚜렷하게 남는 일이 흔하다. 금융 수익률이 대표적인 예이며, 14장과 15장에서 다시 만난다. 흑점 자료를 쓴 것은 원인이 분명해서 그림으로 바로 납득되기 때문이다.


### 독립성 검정: 결합분포와 주변분포의 곱 비교

독립의 정의는 **모든** 사건 쌍에 대해 $P(A \cap B) = P(A)P(B)$가 성립하는 것이다. 따라서 사건을 하나 골라 확인하는 것으로는 독립을 증명할 수 없고, **반례를 하나 찾으면 종속을 증명할 수 있다.**

<div class="codebox" markdown>

#### 예제 3. 결합분포와 주변분포를 견주어 독립성 확인 { .eg }

```python
import numpy as np

np.random.seed(42)
n = 100_000
X = np.random.normal(0, 1, n)
Y = X**2


def check(name, A, B):
    """P(A ∩ B) 와 P(A)P(B) 를 비교한다. 다르면 독립이 아니다."""
    p_joint = np.mean(A & B)
    p_prod = np.mean(A) * np.mean(B)
    same = np.isclose(p_joint, p_prod, atol=0.01)
    print(f"{name}\n  P(A∩B) = {p_joint:.4f},  P(A)P(B) = {p_prod:.4f}"
          f"  ->  {'같다' if same else '다르다'}")


# 사건을 잘못 고르면 종속인데도 통과한다.
# X>0 과 X^2>1 은 X의 대칭성 때문에 우연히 곱셈 규칙을 만족한다.
check("A: X>0,      B: Y>1", X > 0, Y > 1)

# 반례를 제대로 고르면 곧바로 드러난다.
# |X| < 0.5 이면 Y = X^2 < 0.25 이므로 Y > 1 일 수가 없다. 결합확률이 0이다.
check("A: |X|<0.5,  B: Y>1", np.abs(X) < 0.5, Y > 1)
```

출력:

```
A: X>0,      B: Y>1
  P(A∩B) = 0.1599,  P(A)P(B) = 0.1596  ->  같다
A: |X|<0.5,  B: Y>1
  P(A∩B) = 0.0000,  P(A)P(B) = 0.1218  ->  다르다
```

**첫 번째 쌍이 통과했다고 독립인 것이 아니다.** 두 번째 쌍이 곱셈 규칙을 깨뜨리므로 $X$와 $Y$는 독립이 아니다. 반례 하나면 충분하다.

</div>

이것이 상관계수만 보는 것의 위험과 같은 구조다. 상관은 사실상 "한 가지 방식으로만" 관계를 확인하는 것이고, 위의 첫 번째 검사도 한 가지 사건 쌍만 확인한 것이다. 어느 쪽이든 **통과했다는 사실은 아무것도 보장하지 않는다.**

### 결합정규일 때: 무상관 ↔ 독립

<div class="codebox" markdown>

#### 예제 4. 결합정규에서는 무상관이 곧 독립 { .eg }

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 100_000

# 결합정규분포에서는 앞의 반례가 통하지 않는다.
# 이 경우에만 "무상관 = 독립"이 성립하기 때문이다.
# 주의: 두 변수가 **각각** 정규분포인 것으로는 부족하고,
#       둘의 **결합분포**가 정규여야 한다.
rho = 0.8
cov = [[1, rho], [rho, 1]]
corr_data = np.random.multivariate_normal([0, 0], cov, n)

# rho = 0 인 결합정규. 무상관이면서 동시에 독립이다.
indep_data = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], n)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, data, title in zip(axes,
    [corr_data, indep_data],
    [f'ρ={rho} (correlated, dependent)', 'ρ=0 (uncorrelated, independent)']):
    ax.scatter(data[:2000, 0], data[:2000, 1], s=2, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()

# 정규분포에서는 무상관이 곧 독립임을 확인한다
p_joint = np.mean((indep_data[:, 0] > 1) & (indep_data[:, 1] > 1))
p_prod = np.mean(indep_data[:, 0] > 1) * np.mean(indep_data[:, 1] > 1)
print(f"Jointly normal, ρ=0:")
print(f"  P(X>1,Y>1) = {p_joint:.4f}, P(X>1)P(Y>1) = {p_prod:.4f}")
print(f"  Independent? {np.isclose(p_joint, p_prod, atol=0.005)}")
```

출력:

```
Jointly normal, ρ=0:
  P(X>1,Y>1) = 0.0255, P(X>1)P(Y>1) = 0.0253
  Independent? True
```

![독립성과 무상관성의 차이](./img/independence_vs_zero_corr_198.png)

</div>

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
$X$가 대칭이고 $\mathbb{E}[X] = 0$, $\mathbb{E}[X^2] = 1$, $\mathbb{E}[X^3] = 0$이다. $Y = X^2$이라 하자. (a) $\mathrm{Cov}(X, Y)$. (b) $\rho(X, Y)$. (c) 둘은 독립인가?

</div>

??? success "풀이"
    (a) $\mathrm{Cov}(X, Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y] = \mathbb{E}[X^3] - 0 = 0$.

    (b) $\rho(X, Y) = 0/(\sigma_X \sigma_Y) = 0$.

    (c) **독립이 아니다.** $Y = X^2$은 $X$의 결정론적 함수이다. $X = 3$을 알면 $Y = 9$가 결정된다. 상관계수는 관계의 *선형* 성분만 잡아내는데, 양수와 음수인 $X$가 같은 $Y$ 값을 주기 때문에 이 이차 의존성의 선형 성분은 0이다.

    **교훈:** 무상관성은 독립성의 필요조건이지 충분조건이 아니다. 항상 자료를 그려 보라.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
**무상관성이 독립성을 함의하는 경우.** 무상관성이 독립성을 보장하는 경우를 밝히고 증명하라.

</div>

??? success "풀이"
    **특수한 경우:** $(X, Y)$가 **결합정규**이면 무상관성이 독립성을 함의한다.

    **증명:** 이변량 정규분포에 대해

    $$
    f(x, y) = \frac{1}{2\pi\sigma_X\sigma_Y\sqrt{1-\rho^2}}\exp\!\left(-\frac{Q(x, y)}{2(1-\rho^2)}\right)
    $$

    이며 $Q$는 교차항 $-2\rho(x - \mu_X)(y - \mu_Y)/(\sigma_X \sigma_Y)$를 포함한다. $\rho = 0$이면 이 교차항이 사라지고 $Q$는 $x$만 포함하는 항과 $y$만 포함하는 항의 합으로 갈라진다. 그러면 밀도가 인수분해된다:

    $$
    f(x, y) = f_X(x) \cdot f_Y(y)
    $$

    이것이 독립성의 정의이다. $\square$

    **참고:** *결합* 정규성 가정이 결정적이다. $X$와 $Y$가 각각 주변적으로 정규인 것만으로는 충분하지 않다. $X, Y$가 각각 $N(0, 1)$이지만 결합정규가 아니어서 무상관성이 독립성을 함의하지 않는 반례들이 존재한다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
**거리 상관계수**는 독립일 때 그리고 그때만 0이 되어 Pearson 상관계수의 한계를 보완한다. 거리 상관계수를 개념적으로 정의하고 주된 장점을 서술하라.

</div>

??? success "풀이"
    **거리 상관계수**(Székely, Rizzo, Bakirov 2007)는 다음을 만족하는 의존성 측도이다:

    $$
    \mathrm{dCor}(X, Y) = 0 \iff X \perp\!\!\!\perp Y
    $$

    **정의(비형식적):** $X$ 표본들 사이의 쌍별 거리와 $Y$ 표본들 사이의 쌍별 거리를 계산하고, 각 거리행렬을 이중 중심화한 다음, 중심화된 두 거리행렬 사이의 "상관계수"를 계산한다.

    **Pearson 상관계수에 대한 장점:**

    - **비선형** 의존성을 탐지한다(예: $Y = X^2$).
    - $X$와 $Y$의 차원이 달라도 의존성을 탐지한다.
    - 항상 $[0, 1]$에 속한다(Pearson의 절댓값과 비슷하지만 언제나 음이 아니다).
    - $\mathrm{dCor} = 0$일 필요충분조건이 독립성이다(Pearson과 달리 완전한 진단이 된다).

    **비용:** 표본크기에 대해 계산복잡도가 $O(n^2)$이며, Pearson의 $O(n)$과 대비된다. $n < 10^4$인 탐색적 분석에서는 거리 상관계수가 점점 더 권장되는 의존성 측도가 되고 있다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
**상호정보량** $I(X; Y) = \mathbb{E}\!\left[\log \frac{p(X, Y)}{p(X) p(Y)}\right]$에 대해, $I(X; Y) \ge 0$이고 $I(X; Y) = 0$일 필요충분조건이 $X \perp\!\!\!\perp Y$임을 보여라.

</div>

??? success "풀이"
    상호정보량은 결합분포 $p(X, Y)$와 주변분포의 곱 $p(X) p(Y)$ 사이의 **Kullback-Leibler 발산**이다:

    $$
    I(X; Y) = D_{KL}(p(X, Y) \| p(X) p(Y))
    $$

    KL 발산은 음이 아니며(볼록함수 $-\log$에 Jensen 부등식을 적용하면 $D_{KL}(p \| q) \ge 0$), 등호는 거의 어디서나 $p = q$일 때만 성립한다.

    따라서 $I(X; Y) \ge 0$이고, 등호는 $p(X, Y) = p(X) p(Y)$일 때만 성립하는데 이것이 바로 독립성이다.

    $\square$

    **활용:** 상호정보량은 정보이론적 의존성 측도이다. 임의의 비선형 의존성과 고차 의존성을 탐지한다. 자료로부터 $I$를 추정하는 것은 (특히 연속변수에서) 까다롭지만 방법들이 존재한다(k-최근접이웃 추정량, 커널밀도추정, 신경망 기반 상호정보량 추정).

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
**순위 상관계수.** Spearman의 $\rho_S$는 순위들 사이의 Pearson 상관계수이다. Spearman의 $\rho_S$가 (선형 의존성만 잡는 Pearson과 달리) *단조* 의존성을 포착하며 임의의 단조변환에 불변임을 보여라.

</div>

??? success "풀이"
    각 $X_i$를 그 순위 $R_i^X$(1부터 $n$까지)로 바꾸고 $Y_i$에도 같은 작업을 한다. Spearman 상관계수는

    $$
    \rho_S = \frac{\mathrm{Cov}(R^X, R^Y)}{\sigma_{R^X} \sigma_{R^Y}}
    $$

    **단조 의존성:** 단조증가함수 $g$에 대해 $Y = g(X)$이면 순위가 보존되어 $R^Y = R^X$이므로 $\rho_S = 1$이다. $g$가 단조감소이면 $R^Y = n + 1 - R^X$이므로 $\rho_S = -1$이다. *어떤* 단조 관계든 포착한다.

    **Pearson과의 대비:** Pearson의 $\rho$는 $g$가 선형일 때만 1이 된다. $X$가 $[-1, 1]$ 위의 균등분포일 때 $Y = X^3$이면 Pearson $\rho < 1$이지만, 관계가 완전히 단조이므로 Spearman $\rho_S = 1$이다.

    **불변성:** $X$에 임의의 단조변환 $f$를 적용해도 $R^X$가 보존되므로 $\rho_S$는 그런 변환에 불변이다. 이 때문에 이상점이 선형 상관계수를 왜곡할 수 있을 때 Spearman이 로버스트한 상관 측도가 된다.

    Spearman은 (1) 관계가 비선형이지만 단조일 수 있을 때, (2) 이상점이 있을 때, (3) 변수가 순서형일 때 선호된다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
**독립성에 대한 실용적 검정.** 표본 $(X_i, Y_i)_{i=1}^n$이 주어졌을 때 서로 보완하는 두 가지 독립성 검정을 제안하고 각각이 언제 적절한지 논하라.

</div>

??? success "풀이"
    **검정 1 — Pearson 상관 검정:** $H_0$ 아래에서(그리고 결합정규성 아래에서) 통계량 $t = \rho\sqrt{n-2}/\sqrt{1 - \rho^2}$는 $t_{n-2}$를 따른다. $|t|$가 임계값을 넘으면 기각한다.

    *적절한 경우:* 관계가 선형이라고 볼 만하고 자료가 근사적으로 이변량 정규일 때.

    **검정 2 — 거리 상관 검정:** 표본에 대해 $\mathrm{dCor}^2$를 계산하고 $n$을 곱해 척도를 맞춘다. $H_0$ 아래에서 점근분포는 가중된 카이제곱 확률변수들의 합을 포함하며, 검정은 순열로 보정한다($Y$ 값을 무작위로 섞어 $\mathrm{dCor}^2$를 다시 계산하여 귀무분포를 만든다).

    *적절한 경우:* 관계가 비선형일 수 있을 때, 주변분포에 대한 가정을 두지 않을 때, 또는 거리 상관계수가 의미를 갖는 충분히 큰 $n$일 때.

    **로버스트성을 위한 병용:** Pearson은 강한 선형 신호를 값싸게 잡아내고, 거리 상관계수는 더 미묘한 비선형 신호를 잡아내지만 계산 비용이 크다. 표준적인 작업 흐름은 다음과 같다. 먼저 Pearson으로 많은 변수 쌍을 훑고, "무상관"으로 보이는 쌍들을 거리 상관계수나 상호정보량으로 다시 살핀다.

    실용적 고려사항: 두 검정 모두 검정력이 $n$에 따라 커진다. $n < 30$이면 상당한 의존성이 있어도 통계적 유의성을 얻기 어렵다. 산점도로 시각화하는 것이 어떤 형식적 검정보다 의존성을 빨리 진단해 주는 경우가 많다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
**평균독립**을 $E[Y \mid X] = E[Y]$로 정의한다. 다음을 보여라.

(가) 독립이면 평균독립이다.
(나) 평균독립이면 무상관이다.
(다) 두 함의 모두 역이 성립하지 않는다.

</div>

??? success "풀이"
    **(가)** 독립이면 조건부분포가 주변분포와 같으므로 $E[Y\mid X] = E[Y]$이다.

    **(나)** 전체기대값 정리를 쓰면

    $$
    E[XY] = E\{X\,E[Y\mid X]\} = E\{X\,E[Y]\} = E[X]E[Y]
    $$

    이므로 $\operatorname{Cov}(X,Y) = 0$이다.

    **(다-1) 평균독립이지만 독립이 아닌 예.** $X \sim \text{Uniform}(-1,1)$이고 $\varepsilon$이 $X$와 독립이며 평균 0, 분산 1이라 하자. $Y = |X|\varepsilon$로 두면

    $$
    E[Y\mid X] = |X|E[\varepsilon] = 0 = E[Y]
    $$

    로 평균독립이지만, $\operatorname{Var}(Y\mid X) = X^2\operatorname{Var}(\varepsilon)$이 $X$에 의존하므로 독립이 아니다. 이분산 회귀모형이 정확히 이 구조다.

    **(다-2) 무상관이지만 평균독립이 아닌 예.** 본문의 $Y = X^2$($X$가 0 대칭)이 그렇다. $\operatorname{Cov}(X,Y) = E[X^3] = 0$이지만 $E[Y\mid X] = X^2$은 $X$에 의존한다. $\square$

    **위계.**

    $$
    \text{독립} \implies \text{평균독립} \implies \text{무상관}
    $$

    이고 어느 화살표도 뒤집히지 않는다. 세 개념이 서로 다른 층위에 있다는 점이 중요하다. 회귀에서 오차항에 요구하는 조건이 보통 **평균독립** $E[\varepsilon\mid X] = 0$인데, 이것이 무상관 $\operatorname{Cov}(X,\varepsilon)=0$보다 강하다는 사실이 최소제곱 추정량의 불편성을 보장한다. 무상관만으로는 부족하다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
$X, Y, Z$에 대해 (가) $X \perp Y$이면서 $Z$가 주어진 조건 아래 $X \not\perp Y$인 예와, (나) $X \not\perp Y$이면서 $Z$가 주어진 조건 아래 $X \perp Y$인 예를 각각 들어라.

</div>

??? success "풀이"
    **(가) 주변독립이지만 조건부 종속.** $X, Y$가 독립인 동전 던지기이고 $Z = X \oplus Y$(배타적 논리합)라 하자. $X$와 $Y$는 분명히 독립이다. 그런데 $Z = 0$임을 알면 $X = Y$가 되므로, $X$를 아는 순간 $Y$가 완전히 결정된다. 조건을 걸자 없던 의존성이 생겼다.

    인과 그래프로는 $X \to Z \leftarrow Y$ 구조이고 $Z$를 **충돌부**라 부른다. 충돌부에 조건을 걸면 부모들 사이에 가짜 연관이 생긴다. 대학 합격자만 놓고 보면 내신과 수능 점수가 음의 상관을 보이는 벅슨의 역설, 그리고 선택 편향 일반이 모두 이 구조다.

    **(나) 주변 종속이지만 조건부 독립.** $Z$를 기온, $X$를 아이스크림 판매량, $Y$를 익사 사고 수라 하고 $Z$가 주어지면 $X$와 $Y$가 독립이라 하자. 기온을 모르면 $X$와 $Y$는 강하게 상관되지만, 같은 기온대끼리 비교하면 관계가 사라진다.

    인과 그래프로는 $X \leftarrow Z \to Y$ 구조이고 $Z$가 **혼란변수**다. 이 경우에는 조건을 거는 것이 옳다.

    **교훈.** 조건부 독립과 주변 독립은 **어느 쪽도 다른 쪽을 함의하지 않는다.** 그리고 어느 변수에 조건을 걸어야 하는지는 자료만으로는 알 수 없고 인과구조에 대한 가정이 필요하다. 혼란변수에는 조건을 걸어야 하고 충돌부에는 걸면 안 되는데, 둘 다 "상관을 바꾸는 제3의 변수"로 보인다는 점이 어렵다. "통제할 수 있는 변수는 모두 통제하라"는 흔한 조언이 위험한 이유다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
다음 중 $X$와 $Y$의 **무상관**만으로 성립하는 것과 **독립**이 필요한 것을 가려라.

(가) $\operatorname{Var}(X+Y) = \operatorname{Var}(X)+\operatorname{Var}(Y)$
(나) $E[g(X)h(Y)] = E[g(X)]E[h(Y)]$ (모든 $g, h$에 대해)
(다) $M_{X+Y}(t) = M_X(t)M_Y(t)$
(라) $\operatorname{Var}(XY) = \operatorname{Var}(X)\operatorname{Var}(Y) + \operatorname{Var}(X)(E[Y])^2 + \operatorname{Var}(Y)(E[X])^2$

</div>

??? success "풀이"
    **(가) 무상관으로 충분.** $\operatorname{Var}(X+Y) = \operatorname{Var}(X)+\operatorname{Var}(Y)+2\operatorname{Cov}(X,Y)$이므로 공분산이 0이기만 하면 된다. 이것이 무상관의 가장 쓸모 있는 귀결이며, 표본평균의 분산 공식이 여기에 기댄다.

    **(나) 독립이 필요.** 사실 이 조건은 독립과 **동치**다. 무상관은 $g(x)=x$, $h(y)=y$인 한 가지 경우만 보장한다. 본문의 $Y=X^2$ 예에서 $g(x)=x^2$, $h(y)=y$로 두면 등식이 깨진다.

    **(다) 독립이 필요.** 적률생성함수의 곱셈성은 $E[e^{tX}e^{tY}] = E[e^{tX}]E[e^{tY}]$를 요구하는데, 이는 (나)의 특수한 경우다. 무상관만으로는 성립하지 않는다. 정규분포처럼 무상관이 독립을 함의하는 경우에만 안심하고 쓸 수 있다.

    **(라) 독립이 필요.** $E[X^2Y^2] = E[X^2]E[Y^2]$를 써야 유도되는데, 이는 다시 (나)의 한 경우다.

    **정리.** 무상관은 오직 **일차·이차 적률의 교차항** 하나에 대한 조건이다. 그래서 분산의 가법성처럼 공분산만 관계되는 곳에서는 충분하지만, 변수의 함수가 끼어드는 순간 부족해진다. "독립 대신 무상관만 가정해도 된다"는 말이 통하는 범위는 생각보다 좁다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
주변분포가 둘 다 $\text{Uniform}(0,1)$이면서 의존구조가 서로 다른 결합분포를 세 가지 만들어라. 이것이 뜻하는 바를 코퓰라의 언어로 설명하라.

</div>

??? success "풀이"
    $U \sim \text{Uniform}(0,1)$이라 하자.

    1. **독립.** $(U, V)$에서 $V$를 $U$와 독립인 균등확률변수로 둔다. 결합밀도가 $[0,1]^2$에서 1로 평평하다. $\rho = 0$.
    2. **완전 양의 의존.** $V = U$. 질량이 대각선 위에만 놓인다. $\rho = 1$.
    3. **완전 음의 의존.** $V = 1-U$. 질량이 반대 대각선 위에만 놓인다. $\rho = -1$.

    셋 모두 $U$와 $V$의 주변분포는 정확히 $\text{Uniform}(0,1)$이다. 더 만들 수도 있다. 예컨대 $V = 2U \bmod 1$로 두면 질량이 두 선분 위에 놓이면서 $\rho$가 0에 가깝지만 완전히 종속이다.

    **코퓰라의 언어.** 스클라의 정리에 따르면 임의의 결합 CDF $H$는

    $$
    H(x,y) = C\{F_X(x),\ F_Y(y)\}
    $$

    로 쪼개진다. 여기서 $C$가 **코퓰라**이며, $[0,1]^2$ 위에서 주변분포가 균등인 결합분포 그 자체다.

    이 분해가 말해 주는 것은 **주변분포와 의존구조가 완전히 분리된다**는 점이다. 위의 세 예는 각각 독립 코퓰라 $C(u,v)=uv$, 상계 코퓰라 $C(u,v)=\min(u,v)$, 하계 코퓰라 $C(u,v)=\max(u+v-1,0)$에 해당한다. 프레셰-회프딩 부등식에 따라 모든 코퓰라가 아래 둘 사이에 놓인다.

    실무적 함의가 크다. 첫째, **주변분포를 아무리 잘 맞춰도 의존구조는 따로 정해야 한다.** 각 자산의 수익률 분포를 정확히 추정했다 해도 포트폴리오 위험은 코퓰라가 정한다. 둘째, 상관계수 하나로는 의존구조를 담을 수 없다. 정규 코퓰라와 $t$ 코퓰라는 같은 $\rho$를 가져도 꼬리에서 전혀 다르게 움직인다. 정규 코퓰라는 꼬리의존성이 0이라 "같이 폭락할" 확률을 0으로 보는데, 2008년 금융위기 때 신용파생상품 평가에 이 코퓰라가 쓰인 것이 위험을 크게 과소평가한 한 원인으로 지목된다.

---

## 정리하며

- 독립성은 무상관성보다 강한 조건이다. 독립이면 상관계수가 0이지만 그 역은 성립하지 않는다.
- 상관계수는 **선형** 관계만 포착한다. 비선형 의존성을 갖는 변수들(예: $Y = X^2$)은 상관계수가 0일 수 있다.
- 결합정규 확률변수에서는 무상관성이 독립성을 **함의한다**. 정규분포만의 강력한 성질이다.
- 무상관을 독립과 동일시하기 전에 결합정규성 가정이 성립하는지 항상 따져 보아야 한다.
- 실무에서 독립성을 확인하려면 상관계수 하나가 아니라 결합분포 전체를 살펴야 한다.
