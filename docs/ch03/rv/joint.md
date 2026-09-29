# 결합분포와 주변분포

앞의 세 쪽에서 확률변수를 하나씩 보았다. 확률공간 $(\Omega, P)$ 위의 함수 $X$가 결과마다 수를 붙이면 그 수들이 실직선 위에 무게로 퍼지고, 그 퍼짐이 $X$의 분포였다.

그런데 실제로 궁금한 것은 대개 둘 이상이다. 키와 몸무게, 객실등급과 생존 여부, 첫 번째 동전과 두 번째 동전. 이때 $X$와 $Y$를 **따로따로** 아는 것으로는 부족하다. 결과를 $\omega \mapsto (X(\omega), Y(\omega))$로 보내면 무게가 실직선이 아니라 **평면** 위에 놓이며, 그 평면 위의 무게가 **결합분포**다.

이 절이 답하는 물음은 둘이다. 평면 위의 무게에서 각 변수 하나의 분포를 어떻게 되찾는가, 그리고 "$X$와 $Y$가 독립"이라는 말이 그 무게의 모양에 대해 무엇을 뜻하는가.

## 1. 확률변수 둘을 함께 보면 평면 위의 무게가 된다

<div class="thmbox" markdown>

### 정리 1. 결합분포 — 평면 위에 퍼진 무게 { .thm }

**이산형.** 결합확률질량함수는 $p_{X,Y}(x, y) = P(X = x,\; Y = y)$이며 두 조건을 만족한다.

$$
p_{X,Y}(x, y) \ge 0 \quad \text{(모든 } x, y\text{)},
\qquad
\sum_x \sum_y p_{X,Y}(x, y) = 1
$$

**연속형.** 결합확률밀도함수 $f_{X,Y}$는 임의의 영역 $A \subseteq \mathbb{R}^2$에 대해

$$
P((X, Y) \in A) = \iint_A f_{X,Y}(x, y)\,dx\,dy
$$

를 만족하며, 마찬가지로 $f_{X,Y} \ge 0$이고 전체 적분이 $1$이다.

**결합 누적분포함수.**

$$
F_{X,Y}(x, y) = P(X \leq x,\; Y \leq y),
\qquad
f_{X,Y}(x, y) = \frac{\partial^2}{\partial x \, \partial y} F_{X,Y}(x, y)
$$

</div>

**한 변수일 때와 달라진 것은 차원뿐이다.** 조건도 그대로다. 무게는 음수일 수 없고 전부 더하면 $1$이다. 이산이면 영역 $A$ 안의 칸을 더하고 연속이면 그 위에서 적분한다.

## 2. 한 축으로 눌러 모으면 주변분포가 남는다

결합분포를 손에 쥐고 있으면 각 변수 하나의 분포는 언제든 되찾을 수 있다. 다른 변수를 **더해 없애면** 된다.

<div class="thmbox" markdown>

### 정리 2. 주변분포 — 한 축에 비친 그림자 { .thm }

**이산형.**

$$
p_X(x) = \sum_y p_{X,Y}(x, y),
\qquad
p_Y(y) = \sum_x p_{X,Y}(x, y)
$$

**연속형.**

$$
f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x, y)\,dy,
\qquad
f_Y(y) = \int_{-\infty}^{\infty} f_{X,Y}(x, y)\,dx
$$

</div>

**주변분포는 그림자다.** 평면 위의 무게를 한 축 방향으로 눌러 모으면 그 축 위에 무게가 쌓인다. $y$를 모두 더해 없앤 것이 $x$축에 비친 그림자이고, 그것이 $X$의 분포다. "주변(marginal)"이라는 이름은 결합확률을 표로 적었을 때 행과 열의 합계를 **표의 가장자리**에 적던 관행에서 왔다.

**그런데 그림자에서 원래 무게로 돌아오는 길은 없다.** 이것이 이 절에서 가장 중요한 사실이며, 아래 §3에서 같은 그림자를 남기는 전혀 다른 무게 배치를 직접 만들어 확인한다.

!!! note "주변분포는 결합분포를 결정하지 못한다"
    위의 두 표는 같은 실험에서 나왔는데 하나는 독립이고 하나는 종속이다. 그렇다면 **주변분포가 완전히 같은데도** 독립 여부가 갈릴 수 있을까? 갈린다.

    $X$와 $Y$는 둘 다 $\text{Bernoulli}(p)$다. 이 주변분포를 그대로 둔 채 결합분포를 두 가지로 만들 수 있다.

    | 칸 | $(X, Y)$ — 독립 | $(X, X)$ — 완전 종속 |
    |:---|:---:|:---:|
    | $(0,0)$ | $q^2$ | $q$ |
    | $(0,1)$ | $qp$ | $0$ |
    | $(1,0)$ | $pq$ | $0$ |
    | $(1,1)$ | $p^2$ | $p$ |
    | **두 주변분포** | 모두 $\text{Bernoulli}(p)$ | 모두 $\text{Bernoulli}(p)$ |

    오른쪽은 "두 번째 던지기를 보는 대신 첫 번째를 한 번 더 보는" 짝이다. 두 결합분포의 가장자리는 한 칸도 다르지 않지만 안쪽은 전혀 다르다.

    **결론:** 주변분포는 결합분포의 **그림자**일 뿐이다. 결합분포에서 주변분포로 가는 길(주변화)은 언제나 열려 있지만, 반대 방향은 닫혀 있다. 의존 구조는 주변분포 바깥에 있는 정보이며, 그것을 담기 위해 결합분포가 따로 필요하다.

## 3. 결합분포가 있으면 기댓값을 잴 수 있다

무게가 어떻게 퍼져 있는지 알면 그 위에서 평균을 낼 수 있다. 한 변수일 때와 달라지는 것은 더하거나 적분할 축이 둘이라는 점뿐이다.

<div class="thmbox" markdown>

### 정리 3. 결합분포 위의 기댓값 { .thm }

함수 $g(X, Y)$에 대해

$$
E[g(X,Y)] = \sum_x \sum_y g(x,y)\, p_{X,Y}(x,y)
\quad \text{또는} \quad
\iint g(x,y)\, f_{X,Y}(x,y)\, dx\, dy
$$

이다. 특히 **선형성은 독립과 무관하게 언제나 성립한다.**

$$
E[aX + bY] = a\,E[X] + b\,E[Y]
$$

반면 **합의 분산에는 교차항이 붙는다.**

$$
\operatorname{Var}(X + Y) = \operatorname{Var}(X) + \operatorname{Var}(Y) + 2\operatorname{Cov}(X, Y)
$$

</div>

**선형성이 가정을 요구하지 않는다는 점이 중요하다.** $X$와 $Y$가 아무리 얽혀 있어도 평균은 그냥 더해진다. 그래서 기댓값 계산은 결합분포를 몰라도 주변분포만으로 된다.

**분산은 그렇지 않다.** 교차항 $2\operatorname{Cov}(X,Y)$가 남으며, 이 항은 주변분포만으로는 구할 수 없다. 두 변수가 어떻게 얽혀 있는지를 알아야 하고, 그 얽힘을 재는 양이 공분산이다. [3.4절](variance_covariance.md)이 그것을 다룬다.

**독립이면 교차항이 사라진다.** $X \perp Y$이면 $\operatorname{Cov}(X,Y) = 0$이므로

$$
\operatorname{Var}(X + Y) = \operatorname{Var}(X) + \operatorname{Var}(Y)
$$

가 된다. 분산이 그냥 더해지는 이 성질이 5장에서 $\operatorname{Var}(\bar X) = \sigma^2/n$을 얻는 근거이며, 독립이 깨지면 그 계산부터 어긋난다.

### 손으로 풀어 보기 — 이산

<div class="probox" markdown>

**문제:** <span class="diff easy" title="쉬움"></span> 두 자산 $X$와 $Y$의 결합 PMF가 다음과 같다:

| | $Y=0$ | $Y=1$ | $Y=2$ |
|:---|:---:|:---:|:---:|
| $X=0$ | 0.10 | 0.15 | 0.05 |
| $X=1$ | 0.10 | 0.25 | 0.10 |
| $X=2$ | 0.05 | 0.10 | 0.10 |

$P(X + Y \leq 2)$와 $E[XY]$를 계산하라.

</div>

??? success "풀이"

    $$
    P(X+Y \leq 2) = p(0,0) + p(0,1) + p(0,2) + p(1,0) + p(1,1) + p(2,0) = 0.10 + 0.15 + 0.05 + 0.10 + 0.25 + 0.05 = 0.70
    $$

    $$
    E[XY] = \sum_x \sum_y xy \cdot p(x,y) = 0 + 0 + 0 + 0 + 1(1)(0.25) + 1(2)(0.10) + 0 + 2(1)(0.10) + 2(2)(0.10) = 0.85
    $$
---

### 손으로 풀어 보기 — 연속

<div class="probox" markdown>

**문제:** <span class="diff med" title="중간"></span> $0 \leq x \leq y \leq 1$에서 $f_{X,Y}(x,y) = 6(1-y)$라 하자. 이것이 올바른 PDF임을 확인하고 $P(X < 1/2, Y < 1/2)$를 구하라.

</div>

??? success "풀이"
    먼저 확인한다:

    $$
    \int_0^1 \int_0^y 6(1-y)\,dx\,dy = \int_0^1 6y(1-y)\,dy = 6\left[\frac{y^2}{2} - \frac{y^3}{3}\right]_0^1 = 6\left(\frac{1}{2} - \frac{1}{3}\right) = 1 \checkmark
    $$

    이제 확률을 구한다:

    $$
    P\left(X < \tfrac{1}{2}, Y < \tfrac{1}{2}\right) = \int_0^{1/2}\!\int_0^y 6(1-y)\,dx\,dy = \int_0^{1/2} 6y(1-y)\,dy = 6\left[\frac{y^2}{2} - \frac{y^3}{3}\right]_0^{1/2} = 6\left(\frac{1}{8} - \frac{1}{24}\right) = 6 \cdot \frac{2}{24} = \frac{1}{2}
    $$

    따라서 $P(X < 1/2, Y < 1/2) = 1/2$이다.
---

### 코드로 확인하기

### 이산 결합 PMF 표

<div class="codebox" markdown>

#### 예제 2. 이산 결합 확률질량함수 표 { .eg }

```python
import numpy as np
import pandas as pd

# 결합 PMF를 2차원 배열로 적는다. pmf[i, j] = P(X=i, Y=j) 이고 전체 합이 1이다.
pmf = np.array([
    [0.10, 0.15, 0.05],
    [0.10, 0.25, 0.10],
    [0.05, 0.10, 0.10]
])

df = pd.DataFrame(pmf, index=['X=0', 'X=1', 'X=2'], columns=['Y=0', 'Y=1', 'Y=2'])

# 주변분포는 표의 "가장자리(margin)"에 놓인다. 이름의 유래가 그것이다.
#   행 방향으로 더하면(axis=1) Y를 지워 P(X=x)가 남고,
#   열 방향으로 더하면(axis=0) X를 지워 P(Y=y)가 남는다.
# 이것이 주변화 p(x) = sum_y p(x,y) 를 표에서 실행한 것이다.
df['P(X=x)'] = pmf.sum(axis=1)
df.loc['P(Y=y)'] = pmf.sum(axis=0).tolist() + [1.0]   # 맨 끝 1.0은 전체 합
print(df)
```

출력:

```
         Y=0   Y=1   Y=2  P(X=x)
X=0     0.10  0.15  0.05    0.30
X=1     0.10  0.25  0.10    0.45
X=2     0.05  0.10  0.10    0.25
P(Y=y)  0.25  0.50  0.25    1.00
```

</div>

### 연속 결합 PDF 시각화

<div class="codebox" markdown>

#### 예제 3. 연속 결합 밀도함수 시각화 { .eg }

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 200)
y = np.linspace(0, 1, 200)
X, Y = np.meshgrid(x, y)

# 이 결합밀도는 삼각형 영역 0 <= x <= y <= 1 위에서만 0이 아니다.
# where로 그 조건을 걸어 바깥을 0으로 만든다.
# **정의역이 사각형이 아니라는 점이 핵심이다.** X의 범위가 Y에 달려 있으므로
# 두 변수는 종속이며, 결합밀도를 주변밀도의 곱으로 쪼갤 수 없다.
Z = np.where(X <= Y, 6 * (1 - Y), 0)

fig, ax = plt.subplots(figsize=(6, 5))
c = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
fig.colorbar(c, ax=ax, label='f(x, y)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Joint PDF: f(x,y) = 6(1-y)')
plt.show()
```

![Joint PDF: f(x,y) = 6(1-y)](./img/joint_180.png)

</div>

### 이변량 정규분포 표본추출

<div class="codebox" markdown>

#### 예제 4. 이변량 정규분포 표본추출 { .eg }

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
mean = [0, 0]
# 분산이 둘 다 1이므로 비대각원소 0.7이 곧 상관계수다.
cov = [[1, 0.7], [0.7, 1]]
# 5000개의 (x, y) 쌍을 뽑는다. 결과는 (5000, 2) 모양이다.
samples = np.random.multivariate_normal(mean, cov, 5000)

fig, ax = plt.subplots(figsize=(6, 5))
# alpha를 낮춰 겹침을 푼다. 5000개를 그대로 찍으면 가운데가 뭉개진다.
ax.scatter(samples[:, 0], samples[:, 1], alpha=0.2, s=5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
# 가로세로 비를 맞춰야 타원의 기울기를 정직하게 볼 수 있다
ax.set_aspect('equal')
ax.spines[['top', 'right']].set_visible(False)
plt.show()
```

![결합분포](./img/joint_202.png)

</div>

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
결합 PMF가 $p(0,0)=0.10, p(0,1)=0.15, p(0,2)=0.05, p(1,0)=0.10, p(1,1)=0.20, p(1,2)=0.10, p(2,0)=0.05, p(2,1)=0.10, p(2,2)=0.15$이다. (a) 올바른 PMF인가? (b) 주변분포. (c) $\mathbb{E}[X], \mathbb{E}[Y], \mathbb{E}[XY]$. (d) $\mathrm{Cov}, \rho$. (e) 독립인가?

</div>

??? success "풀이"
    (a) 합 = 1이고 모두 음이 아니다 ✓.

    (b) 행의 합: $p_X(0,1,2) = (0.30, 0.40, 0.30)$. 열의 합: $p_Y(0,1,2) = (0.25, 0.45, 0.30)$.

    (c) $\mathbb{E}[X] = 1.0$, $\mathbb{E}[Y] = 1.05$, $\mathbb{E}[XY] = 0.20 + 0.20 + 0.20 + 0.60 = 1.20$.

    (d) $\mathrm{Cov} = 1.20 - 1.0 \cdot 1.05 = 0.15$. $\mathrm{Var}(X) = 0.60$, $\mathrm{Var}(Y) = 0.5475$. $\rho = 0.15/\sqrt{0.60 \cdot 0.5475} \approx 0.262$.

    (e) 독립이 아니다: $p(0,0) = 0.10 \ne p_X(0) p_Y(0) = 0.075$.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
**결합 CDF.** 결합 PDF가 $f(x, y)$인 연속확률변수 $(X, Y)$에 대해 결합 CDF를 $F(x, y) = P(X \le x, Y \le y)$로 정의한다. $f$를 $F$로 표현하라.

</div>

??? success "풀이"
    결합 CDF는 왼쪽 아래 사분면에서 결합 PDF를 적분한 것이다:

    $$
    F(x, y) = \int_{-\infty}^x \int_{-\infty}^y f(s, t) ds\, dt
    $$

    (충분한 매끄러움을 가정하고) 두 변수 모두에 대해 미분하면:

    $$
    f(x, y) = \frac{\partial^2 F(x, y)}{\partial x \partial y}
    $$

    이는 1차원 관계 $f = F'$의 결합분포 판이다.

    결합 CDF의 성질: 두 인수 각각에 대해 비감소이고, $F(-\infty, y) = F(x, -\infty) = 0$, $F(\infty, \infty) = 1$이며, 주변분포는 $F_X(x) = F(x, \infty)$와 $F_Y(y) = F(\infty, y)$로 되찾는다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
**조건부분포로부터 결합분포 구하기.** $f_{X|Y}(x \mid y) = (1/y) \mathbf 1\{0 \le x \le y\}$($[0, y]$ 위의 균등분포)이고 $y \ge 0$에 대해 $f_Y(y) = e^{-y}$이다. 결합분포와 $X$의 주변분포를 구하라.

</div>

??? success "풀이"
    결합분포: $f_{X, Y}(x, y) = f_{X|Y}(x|y) f_Y(y) = (1/y) e^{-y} \mathbf 1\{0 \le x \le y\}$.

    $X$의 주변분포: $y \ge x$인 영역에서 $y$에 대해 적분한다:

    $$
    f_X(x) = \int_x^\infty \frac{e^{-y}}{y} dy
    $$

    이 적분은 초등함수로 표현되지 않는다(**지수적분** $E_1(x)$와 같다). $x$가 작으면 $-\ln x$처럼 발산하고, $x$가 크면 $e^{-x}/x$처럼 행동한다. $X$는 알려져 있지만 초등함수가 아닌 분포를 갖는다.

    **교훈:** 주변분포가 복잡하더라도 조건부 구조는 단순할 수 있다. 이러한 인수분해 관점이 계층적 베이즈 모형의 토대이다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
**다항분포.** 이항분포를 $k$개의 범주로 일반화한다. $n$번의 시행에서 각 시행이 독립적으로 확률 $p_j$로 결과 $j$를 내고 $\sum_j p_j = 1$이다. $X_j$를 결과 $j$의 횟수라 할 때 결합 PMF를 유도하라.

</div>

??? success "풀이"
    특정한 결과 열 하나가 나올 확률은 $\prod_j p_j^{X_j}$이다. 계수 벡터 $(X_1, \ldots, X_k)$를 주는 열의 개수는 다항계수 $\binom{n}{X_1, X_2, \ldots, X_k} = n!/(X_1! X_2! \cdots X_k!)$이다.

    결합 PMF:

    $$
    P(X_1 = x_1, \ldots, X_k = x_k) = \frac{n!}{x_1! x_2! \cdots x_k!} \prod_{j=1}^k p_j^{x_j}
    $$

    이는 $x_j \ge 0$이고 $\sum_j x_j = n$일 때 성립한다.

    **$X_j$의 주변분포**는 $\mathrm{Binomial}(n, p_j)$이다($j$ 이외의 범주를 하나로 묶으면 된다).

    $i \ne j$일 때 $X_i, X_j$ 사이의 **공분산**은 $\mathrm{Cov}(X_i, X_j) = -np_i p_j$이다. 음수인 이유는 한 계수가 늘어나면 제약 $\sum = n$ 때문에 다른 계수가 줄어들어야 하기 때문이다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
**이변량 정규분포.** 평균이 $\mu_X, \mu_Y$, 분산이 $\sigma_X^2, \sigma_Y^2$, 상관계수가 $\rho$인 이변량 정규 $(X, Y)$의 결합 PDF 공식을 쓰라. 조건부분포는 무엇인가?

</div>

??? success "풀이"
    PDF:

    $$
    f(x, y) = \frac{1}{2\pi\sigma_X\sigma_Y\sqrt{1-\rho^2}} \exp\!\left(-\frac{1}{2(1-\rho^2)}\!\left[\frac{(x-\mu_X)^2}{\sigma_X^2} - \frac{2\rho(x-\mu_X)(y-\mu_Y)}{\sigma_X\sigma_Y} + \frac{(y-\mu_Y)^2}{\sigma_Y^2}\right]\right)
    $$

    **조건부분포:** $Y \mid X = x$는 정규분포이며

    $$
    \mathbb{E}[Y \mid X = x] = \mu_Y + \rho\frac{\sigma_Y}{\sigma_X}(x - \mu_X)
    $$

    $$
    \mathrm{Var}(Y \mid X = x) = \sigma_Y^2(1 - \rho^2)
    $$

    **주목할 특징:**

    - 조건부 평균이 $x$에 대해 **선형**이며, 이것이 *회귀직선*이다. 기울기 $\rho\sigma_Y/\sigma_X$는 정확히 OLS 회귀계수이다.
    - 조건부 분산이 $x$에 의존하지 *않는다*. 등분산성이다.
    - $\rho = 0$이면 독립이다(결합정규일 때의 특별한 성질이며, 일반적으로는 성립하지 않는다).

    이변량 정규분포는 상관분석의 토대이며, 닫힌 형태의 회귀 구조를 갖는 다변량 분포의 가장 단순하고 자명하지 않은 예이다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
$X$와 $Y$가 둘 다 $\text{Uniform}(0,1)$을 따른다는 것만 알고 있다. 이 정보만으로 $P(X + Y > 1.5)$를 답할 수 있는가? 서로 다른 답을 주는 결합분포 둘을 만들어 보여라.

</div>

??? success "풀이"
    **답할 수 없다.** 주변분포는 결합분포를 결정하지 않으므로(§2) 두 변수가 어떻게 얽혀 있는지에 따라 답이 달라진다.

    **경우 1 (독립).** 단위정사각형 위에 무게가 고르게 퍼져 있고, $x + y > 1.5$인 영역은 꼭짓점 $(0.5, 1)$, $(1, 0.5)$, $(1,1)$을 잇는 직각삼각형이다. 두 변의 길이가 $0.5$이므로

    $$
    P(X + Y > 1.5) = \tfrac12 \times 0.5 \times 0.5 = 0.125
    $$

    **경우 2 ($Y = X$).** 무게가 대각선 위에만 놓인다. 주변분포는 여전히 각각 $\text{Uniform}(0,1)$이다.

    $$
    P(X + Y > 1.5) = P(2X > 1.5) = P(X > 0.75) = 0.25
    $$

    **같은 그림자에서 $0.125$와 $0.25$가 나온다.** 두 배 차이다.

    **실무적 함의가 작지 않다.** 위험을 합산하는 일이 모두 이 구조다. 자산 둘의 수익률 분포를 각각 알아도 포트폴리오 손실의 분포는 정해지지 않으며, 둘이 함께 무너질 가능성을 따로 모형화해야 한다. 주변분포는 그대로 두고 얽힘만 갈아 끼우는 도구를 **코퓰러**라 부른다. 2008년 금융위기에서 문제가 된 것이 바로 이 얽힘을 너무 낙관적으로 잡은 모형이었고, [5장의 금융위기 사례](../../ch05/foundations/financial_crisis_clt.md)가 그 이야기를 다룬다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff easy" title="쉬움"></span>
결합 PMF가 아래 표와 같다. 두 주변분포를 구하고, $E[X]$와 $E[XY]$를 계산하라.

| $p_{X,Y}$ | $Y=0$ | $Y=1$ |
|:---|---:|---:|
| $X=0$ | $0.1$ | $0.2$ |
| $X=1$ | $0.3$ | $0.4$ |

</div>

??? success "풀이"
    행을 더해 $p_X(0) = 0.3$, $p_X(1) = 0.7$이고, 열을 더해 $p_Y(0) = 0.4$, $p_Y(1) = 0.6$이다.

    $$
    E[X] = 0 \cdot 0.3 + 1 \cdot 0.7 = 0.7
    $$

    $E[XY]$는 $xy$가 $0$이 아닌 칸이 하나뿐이라 간단하다.

    $$
    E[XY] = 1 \cdot 1 \cdot p_{X,Y}(1,1) = 0.4
    $$

    참고로 $E[X]E[Y] = 0.7 \times 0.6 = 0.42 \ne 0.4$이므로 $\operatorname{Cov}(X,Y) = -0.02$이고 두 변수는 독립이 아니다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
$[0,1]^2$ 위에서 $f_{X,Y}(x,y) = c\,(x + y)$라 하자. $c$를 구하고, 주변밀도와 $P(X > Y)$를 구하라.

</div>

??? success "풀이"
    **규격화.** $\int_0^1\!\!\int_0^1 c(x+y)\,dx\,dy = c\left(\tfrac12 + \tfrac12\right) = c = 1$이므로 $c = 1$이다.

    **주변밀도.**

    $$
    f_X(x) = \int_0^1 (x + y)\,dy = x + \tfrac12 \qquad (0 \le x \le 1)
    $$

    대칭이므로 $f_Y(y) = y + \tfrac12$다.

    **$P(X > Y)$.** 밀도가 $x$와 $y$에 대해 **대칭**이므로 직선 $y = x$의 양쪽 무게가 같다. 따라서

    $$
    P(X > Y) = \tfrac12
    $$

    적분으로 확인해도 $\int_0^1\!\!\int_0^x (x+y)\,dy\,dx = \int_0^1 \left(x^2 + \tfrac{x^2}{2}\right)dx = \tfrac12$다.

    **덧붙여 독립이 아니다.** $f_X(x)f_Y(y) = (x+\tfrac12)(y+\tfrac12)$에는 $xy$ 항이 생기는데 원래 밀도에는 없다. **합은 곱으로 쪼개지지 않는다.**

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
연습문제 $8$의 분포에서 $E[X + Y]$와 $\operatorname{Var}(X+Y)$를 구하려 한다. 둘 중 어느 쪽이 주변분포만으로 계산되는가? 그 차이가 왜 생기는지 설명하라.

</div>

??? success "풀이"
    **$E[X+Y]$는 주변분포만으로 된다.** 선형성이 독립을 요구하지 않기 때문이다.

    $$
    E[X] = \int_0^1 x\left(x + \tfrac12\right)dx = \tfrac13 + \tfrac14 = \tfrac{7}{12}
    $$

    대칭이므로 $E[Y]$도 같고, 따라서 $E[X+Y] = \tfrac{7}{6}$이다.

    **$\operatorname{Var}(X+Y)$는 주변분포만으로 안 된다.** 정리 3에서 보았듯

    $$
    \operatorname{Var}(X+Y) = \operatorname{Var}(X) + \operatorname{Var}(Y) + 2\operatorname{Cov}(X,Y)
    $$

    인데 마지막 항이 두 변수가 **어떻게 얽혀 있는지**에 달려 있다. 같은 주변분포를 갖는 다른 결합분포를 가져오면 이 값이 달라진다(연습문제 $11$이 그 극단적인 예다).

    **요약하면 이렇다.** 평균은 그림자만 보고도 더할 수 있지만, 퍼짐은 안쪽 무게를 보아야 한다. 이것이 공분산이 따로 필요한 이유이며 [3.4절](variance_covariance.md)이 그 양을 다룬다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
공정한 주사위 둘을 굴려 눈을 $X$, $Y$라 한다. $S = X + Y$의 분포를 구하라. 이 계산에 $X$와 $Y$의 **주변분포만으로 충분한가?**

</div>

??? success "풀이"
    결합분포는 $36$칸에 $1/36$씩 고르게 놓인다. $S = k$인 칸의 개수를 세면 된다.

    | $k$ | $2$ | $3$ | $4$ | $5$ | $6$ | $7$ | $8$ | $9$ | $10$ | $11$ | $12$ |
    |---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
    | 칸 수 | $1$ | $2$ | $3$ | $4$ | $5$ | $6$ | $5$ | $4$ | $3$ | $2$ | $1$ |

    따라서 $P(S=k)$는 위 칸 수를 $36$으로 나눈 값이고, $k=7$에서 $6/36$으로 가장 크다.

    **주변분포만으로는 충분하지 않다.** 위 계산에서 실제로 쓴 것은 "각 칸이 $1/36$"이라는 **결합분포**다. 주변분포가 똑같이 각각 균등한데도 결합분포가 다르면 답이 달라진다. 예컨대 $Y = X$로 짝지으면 두 주변분포는 그대로지만

    $$
    S = 2X \in \{2, 4, 6, 8, 10, 12\}
    $$

    이 되어 홀수가 아예 나오지 않는다. **합의 분포는 그림자가 아니라 안쪽 무게가 정한다.**

    다만 **독립이라는 조건이 주어지면** 주변분포만으로 계산이 닫힌다. 그때 결합분포가 곱으로 복원되기 때문이며, 그 계산을 **합성곱**이라 부른다.

    $$
    P(S = k) = \sum_x p_X(x)\, p_Y(k - x)
    $$

## 정리하며

- 결합분포는 여러 확률변수의 동시적 거동을 기술한다.
- 결합 PMF/PDF는 음이 아니어야 하고 지지집합 전체에서 합 또는 적분이 1이어야 한다.
- 독립성은 결합분포가 주변분포들의 곱으로 인수분해되는 것과 동치이다.
- 여러 변수의 함수에 대한 기댓값은 결합분포에 대해 합하거나 적분하여 계산한다.
- 합의 분산은 변수들 사이의 공분산에 의존하며, 독립이면 단순히 분산의 합이 된다.
