# 균등분포

## 개요

**균등분포**는 구간 $[a, b]$의 모든 값에 동일한 확률을 부여한다. 가장 단순한 연속분포이며, 난수 생성, 시뮬레이션, 확률적분변환의 토대가 된다.

---

## 연속 균등분포와 누적분포함수

<div class="defn" markdown>

### 정의 1. 연속 균등분포 { .dfn }

확률변수 $X$가 $[a, b]$ 위의 연속 균등분포를 따른다는 것은 다음을 뜻한다:

$$
X \sim \text{Uniform}(a, b), \qquad f(x) = \begin{cases} \frac{1}{b - a} & \text{if } a \leq x \leq b \\ 0 & \text{otherwise} \end{cases}
$$

PDF는 구간에서 상수이며, 이는 모든 값이 동일하게 나타날 수 있음을 반영한다.

</div>

### CDF

$$
F(x) = \begin{cases} 0 & x < a \\ \frac{x - a}{b - a} & a \leq x \leq b \\ 1 & x > b \end{cases}
$$

---

## 성질

$$
\begin{aligned}
E[X] &= \frac{a + b}{2} \\[4pt]
\text{Var}(X) &= \frac{(b - a)^2}{12} \\[4pt]
\text{SD}(X) &= \frac{b - a}{2\sqrt{3}}
\end{aligned}
$$

### 평균의 유도

$$
E[X] = \int_a^b x \cdot \frac{1}{b-a}\,dx = \frac{1}{b-a} \cdot \frac{x^2}{2}\bigg|_a^b = \frac{b^2 - a^2}{2(b-a)} = \frac{a+b}{2}
$$

### 분산의 유도

$$
E[X^2] = \int_a^b x^2 \cdot \frac{1}{b-a}\,dx = \frac{1}{b-a} \cdot \frac{x^3}{3}\bigg|_a^b = \frac{a^2 + ab + b^2}{3}
$$

$$
\text{Var}(X) = E[X^2] - (E[X])^2 = \frac{a^2 + ab + b^2}{3} - \frac{(a+b)^2}{4} = \frac{(b-a)^2}{12}
$$

---

## 표준 균등분포

특수한 경우인 $U \sim \text{Uniform}(0, 1)$을 **표준 균등분포**라 한다. 모든 균등확률변수는 이것과 다음과 같이 연결된다:

$$
X = a + (b - a)U \sim \text{Uniform}(a, b) \quad \text{where } U \sim \text{Uniform}(0, 1)
$$

역으로:

$$
U = \frac{X - a}{b - a} \sim \text{Uniform}(0, 1) \quad \text{where } X \sim \text{Uniform}(a, b)
$$

---

## 확률적분변환

균등분포는 **확률적분변환**을 통해 시뮬레이션에서 핵심적인 역할을 한다:

**정리:** $X$가 CDF $F$를 갖는 연속확률변수이면 $F(X) \sim \text{Uniform}(0, 1)$이다.

**역 (역변환 표본추출):** $U \sim \text{Uniform}(0, 1)$이면 $X = F^{-1}(U)$의 CDF는 $F$이다.

??? proof "증명"


    $$
    P(F(X) \leq u) = P(X \leq F^{-1}(u)) = F(F^{-1}(u)) = u
    $$

    이는 $\text{Uniform}(0, 1)$의 CDF이다.

    이 정리는 균등난수 생성기만으로 임의의 분포에서 확률표본을 생성할 수 있게 하는 근거이다.

    ---

## 이산 균등분포

이산형 대응물은 유한집합 $\{a, a+1, \ldots, b\}$의 각 값에 동일한 확률을 부여한다:

$$
P(X = k) = \frac{1}{b - a + 1}, \quad k = a, a+1, \ldots, b
$$

$$
E[X] = \frac{a + b}{2}, \qquad \text{Var}(X) = \frac{(b - a + 1)^2 - 1}{12}
$$

---

## 문제

<div class="probox" markdown>

**문제:** <span class="diff easy" title="쉬움"></span> 어떤 자산의 일간 수익률을 $-2\%$와 $+3\%$ 사이의 균등분포로 모형화한다. 수익률이 $1\%$를 넘을 확률은? 기대수익률은?

</div>

??? success "풀이"

    $$
    P(X > 1) = \frac{3 - 1}{3 - (-2)} = \frac{2}{5} = 0.40
    $$

    $$
    E[X] = \frac{-2 + 3}{2} = 0.5\%
    $$
---

## Python: PDF, CDF, 표본추출

### PDF와 CDF

<div class="codebox" markdown>

#### 예제 1. 균등분포의 밀도함수와 분포함수 { .eg }

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

a, b = 2, 8
# 구간 바깥까지 그려야 "밖에서는 0"이라는 사실이 그림에 드러난다
x = np.linspace(a - 1, b + 1, 300)

fig, ax = plt.subplots(figsize=(12, 3))
# scale은 폭 b-a 이지 b가 아니다(loc=2, scale=6 이 [2, 8]을 뜻한다).
# PDF는 구간 안에서 1/(b-a) = 1/6 로 평평하다.
# CDF는 그 평평한 값을 적분한 것이므로 **기울기 1/6 의 직선**이 된다.
ax.plot(x, stats.uniform(loc=a, scale=b-a).pdf(x), label='PDF', lw=2)
ax.plot(x, stats.uniform(loc=a, scale=b-a).cdf(x), label='CDF', lw=2)
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
plt.show()
```

![균등분포](./img/uniform_125.png)

</div>

### 표본추출과 히스토그램

<div class="codebox" markdown>

#### 예제 2. 균등 표본의 히스토그램 { .eg }

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
a, b = 2, 8
samples = stats.uniform(loc=a, scale=b-a).rvs(50_000)

fig, ax = plt.subplots(figsize=(12, 3))
# 5만 개를 60개 구간에 넣으면 구간마다 평균 833개다.
# 막대 높이가 들쭉날쭉한 것은 잡음이며, 표본을 늘리면 평평해진다.
ax.hist(samples, bins=60, density=True, alpha=0.7, label='Samples')
x = np.linspace(a - 1, b + 1, 300)
ax.plot(x, stats.uniform(loc=a, scale=b-a).pdf(x), 'r-', lw=2, label='PDF')
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
plt.show()
```

![균등분포](./img/uniform_143.png)

</div>

### 역변환 표본추출

<div class="codebox" markdown>

#### 예제 3. 균등난수로 지수 표본 만들기 { .eg }

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# 균등난수로 지수분포 표본을 만든다(역변환 표집).
# 지수분포의 CDF는 F(x) = 1 - e^{-lam x} 이므로 이를 x에 대해 풀면
#   u = 1 - e^{-lam x}  ->  x = -ln(1-u) / lam
# 이 역함수가 아래 한 줄이다. 균등난수만 있으면 어떤 분포든 만들 수 있다는
# 사실이 몬테카를로 방법의 출발점이다.
u = np.random.uniform(0, 1, 50_000)
lam = 2.0
x_exp = -np.log(1 - u) / lam

fig, ax = plt.subplots(figsize=(12, 3))
ax.hist(x_exp, bins=100, density=True, alpha=0.7, label='Inverse transform samples')
t = np.linspace(0, 4, 200)
ax.plot(t, stats.expon(scale=1/lam).pdf(t), 'r-', lw=2, label='Exponential PDF')
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
plt.show()
```

![균등분포](./img/uniform_163.png)

</div>

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
$U \sim \mathrm{Uniform}(0, 1)$이고 $X = -(1/\lambda)\ln(1 - U)$이다. (a) $X$의 CDF를 구하라. (b) 그 분포를 밝혀라. (c) 역변환 방법을 설명하라. (d) $1 - U \sim \mathrm{Uniform}(0, 1)$임을 보여라.

</div>

??? success "풀이"
    (a) $P(X \le x) = P(-(1/\lambda)\ln(1 - U) \le x) = P(U \le 1 - e^{-\lambda x}) = 1 - e^{-\lambda x}$.

    (b) 이는 $\mathrm{Exp}(\lambda)$의 CDF이다. 따라서 $X \sim \mathrm{Exp}(\lambda)$.

    (c) **역변환 방법:** CDF $F$가 역함수를 갖는 임의의 분포에 대해 $U \sim \mathrm{Uniform}(0, 1)$을 써서 $X = F^{-1}(U)$로 두면 $X$의 CDF는 $F$가 된다. 분위수 함수가 닫힌 형태로 주어지는 분포에서 표본을 뽑는 보편적인 방법이다.

    (d) $P(1 - U \le t) = P(U \ge 1 - t) = 1 - (1 - t) = t$. 따라서 $1 - U \sim \mathrm{Uniform}(0, 1)$이다. 그러므로 더 간단한 공식 $X = -(1/\lambda) \ln U$도 동등하다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
**Uniform$(a, b)$의 평균과 분산.** PDF로부터 둘 다 유도하라.

</div>

??? success "풀이"
    PDF: $[a, b]$ 위에서 $f(x) = 1/(b - a)$.

    $\mathbb{E}[X] = \int_a^b x/(b - a) dx = (b^2 - a^2)/(2(b - a)) = (a + b)/2$.

    $\mathbb{E}[X^2] = \int_a^b x^2/(b - a) dx = (b^3 - a^3)/(3(b - a)) = (a^2 + ab + b^2)/3$.

    $\mathrm{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = (a^2 + ab + b^2)/3 - (a + b)^2/4 = (b - a)^2/12$.

    **표준적인 경우:** Uniform(0, 1)의 평균은 1/2, 분산은 1/12이다. Uniform(-1, 1)의 평균은 0, 분산은 1/3이다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
**두 균등확률변수의 합.** $U_1, U_2 \sim \mathrm{Uniform}(0, 1)$이 독립이면 $U_1 + U_2$가 $[0, 2]$ 위의 **삼각분포**를 따름을 보여라.

</div>

??? success "풀이"
    PDF를 합성곱한다:

    $$
    f_{U_1 + U_2}(s) = \int_{-\infty}^\infty f_{U_1}(s - u) f_{U_2}(u) du
    $$

    $u \in [0, 1]$이고 $s - u \in [0, 1]$일 때 피적분함수가 1이고, 그 밖에서는 0이다. 적분 영역은:

    - $s \in [0, 1]$일 때: $u \in [0, s]$이므로 적분값 = $s$.
    - $s \in [1, 2]$일 때: $u \in [s - 1, 1]$이므로 적분값 = $2 - s$.

    결과: $s \in [0, 2]$에 대해 $f_{U_1 + U_2}(s) = \min(s, 2 - s)$이며, $s = 1$에서 높이 1로 정점을 이루는 삼각형이다.

    균등확률변수의 합에 대한 중심극한정리는 6개 이상만 더해도 근사적으로 정규분포가 됨을 알려 준다. 수렴이 빠른 것이다. 이는 정규난수 생성을 위한 Marsaglia-Bray 알고리즘의 바탕이 된다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
**균등확률변수의 순서통계량.** $U_1, \ldots, U_n$이 i.i.d. $\mathrm{Uniform}(0, 1)$일 때 $k$번째 순서통계량 $U_{(k)}$는 $\mathrm{Beta}(k, n - k + 1)$ 분포를 따른다. CDF를 유도하라.

</div>

??? success "풀이"
    $U_{(k)} \le u$일 필요충분조건은 $U_i$들 중 적어도 $k$개가 $u$ 이하인 것이다. $u$ 이하인 $U_i$의 개수는 $\mathrm{Binomial}(n, u)$를 따른다(각각 독립적으로 확률 $u$).

    $$
    P(U_{(k)} \le u) = P(\mathrm{Binomial}(n, u) \ge k) = \sum_{j=k}^n \binom{n}{j} u^j (1 - u)^{n - j}
    $$

    불완전 베타함수 항등식에 의해 이는 정규화된 불완전 베타함수 $I_u(k, n - k + 1)$과 같다. 따라서 $U_{(k)} \sim \mathrm{Beta}(k, n - k + 1)$이다.

    베타분포의 잘 알려진 평균 공식에 의해 $\mathbb{E}[U_{(k)}] = k/(n + 1)$이다. 기대 순서통계량은 $[0, 1]$을 $n + 1$개의 동일한 조각으로 나누며, 이것이 Q-Q 그림에서 쓰는 **작도 위치**가 된다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
**최대 엔트로피 성질.** 유계 구간 $[a, b]$ 위의 모든 분포 중에서 균등분포가 **미분 엔트로피**를 최대로 한다. 미분 엔트로피 공식을 쓰고 이를 확인하라.

</div>

??? success "풀이"
    미분 엔트로피: $h(X) = -\int f(x) \ln f(x) dx$.

    $X \sim \mathrm{Uniform}(a, b)$에 대해 $h(X) = -\int_a^b (1/(b-a)) \ln(1/(b-a)) dx = \ln(b - a)$.

    **주장:** $[a, b]$ 위의 다른 어떤 분포 $g$도 $h(g) \le \ln(b - a)$를 만족한다. KL 발산의 비음성으로 증명한다:

    $0 \le D_{KL}(g \| f) = \int g \ln(g/f) dx = -h(g) - \int g \ln f \, dx = -h(g) + \ln(b - a)$

    따라서 $h(g) \le \ln(b - a) = h(f)$이며, 등호는 거의 어디서나 $g = f$일 때만 성립한다.

    **해석:** 지지집합 외에 아무 정보가 없을 때 균등분포는 "가장 정보가 적은" 분포로, 모든 곳에 동일한 질량을 부여한다. 이 때문에 유계 모수에 대한 무정보 베이즈 분석에서 자연스러운 사전분포가 된다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
**확률적분변환.** $X$가 연속 CDF $F$를 가지면 $U = F(X) \sim \mathrm{Uniform}(0, 1)$임을 증명하라.

</div>

??? success "풀이"
    $F$의 연속성(이 덕분에 $F^{-1}$이 잘 정의되고 $F \circ F^{-1} = \mathrm{id}$이다)을 이용하면 $P(U \le u) = P(F(X) \le u) = P(X \le F^{-1}(u)) = F(F^{-1}(u)) = u$이다.

    따라서 $U$의 CDF는 $[0, 1]$ 위에서 $u$이며, 즉 $U \sim \mathrm{Uniform}(0, 1)$이다.

    **활용:** 확률적분변환은 여러 통계 검정의 바탕이 된다:

    - **Kolmogorov-Smirnov 검정:** 가설로 세운 $F$로 자료를 변환하면 귀무가설 아래에서 변환된 자료는 균등분포를 따른다.
    - **코퓰러 모형:** 결합분포를 주변분포(확률적분변환으로 균등분포화)와 균등확률변수들을 잇는 코퓰러로 분해한다.
    - **분포 예측의 검증:** 예측분포가 올바르게 설정되었다면 변환된 관측값들은 균등분포를 따라야 한다.

    이는 역변환 표본추출의 쌍대이다. 하나는 균등확률변수를 다른 분포로 바꾸고, 다른 하나는 다른 분포를 균등분포로 바꾼다. 둘 다 같은 CDF/역 CDF 장치에 의존한다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
$\{1, 2, \dots, n\}$ 위의 이산 균등분포의 평균과 분산을 구하라. $n = 6$(공정한 주사위)일 때 값을 구하고, 연속 균등분포 $\text{Uniform}(0.5, 6.5)$의 분산과 견주어라.

</div>

??? success "풀이"
    각 값의 확률이 $1/n$이므로

    $$
    E[X] = \frac1n\sum_{k=1}^n k = \frac1n\cdot\frac{n(n+1)}{2} = \frac{n+1}{2}
    $$

    이고, $\sum k^2 = n(n+1)(2n+1)/6$을 쓰면

    $$
    E[X^2] = \frac{(n+1)(2n+1)}{6}, \qquad \operatorname{Var}(X) = \frac{(n+1)(2n+1)}{6} - \frac{(n+1)^2}{4} = \frac{n^2-1}{12}
    $$

    이다. $n = 6$이면 평균 $3.5$, 분산 $35/12 \approx 2.917$이다.

    연속판 $\text{Uniform}(0.5, 6.5)$는 평균이 같은 3.5이지만 분산이 $6^2/12 = 3$이다. 두 값의 차이가 정확히

    $$
    3 - \frac{35}{12} = \frac{1}{12}
    $$

    인데, 이것이 **반올림 보정** $w^2/12$($w=1$)이다. 우연이 아니다. 이산 균등분포는 연속 균등분포를 폭 1의 눈금으로 반올림한 것이고, 반올림은 산포를 **줄인다**. 각 칸 안의 변동이 사라지기 때문이다. 거꾸로 이미 반올림된 자료에서 원래 산포를 되살리려면 $w^2/12$를 더해야 하는데, 이것이 다음 절에서 볼 셰퍼드 보정이다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
선형합동생성기 $x_{k+1} = (a x_k + c) \bmod m$이 만드는 $u_k = x_k/m$은 진짜 균등난수가 아니다. 어떤 점에서 그러한지 두 가지를 들고, 그럼에도 왜 쓸 만한지 설명하라.

</div>

??? success "풀이"
    **(1) 주기가 유한하다.** 상태가 $\{0, 1, \dots, m-1\}$ 안의 정수 하나이므로 많아야 $m$번 만에 반드시 반복된다. 32비트 생성기라면 주기가 최대 $2^{32} \approx 4.3\times10^9$인데, 요즘 모의실험은 이 개수를 쉽게 넘긴다. 주기를 넘어서면 같은 수열을 되풀이하므로 "독립 표본"이 아니게 되고, 추정량의 분산이 실제보다 작게 보이는 함정이 생긴다.

    **(2) 격자 구조가 있다.** 연속한 값들을 묶어 $(u_k, u_{k+1}, \dots, u_{k+d-1})$로 $d$차원 점을 만들면, 점들이 공간에 고르게 흩어지지 않고 **몇 개의 평행한 초평면 위에만 놓인다**(마사글리아 정리). 1차원 히스토그램은 완벽히 평평한데 2차원 산점도에 줄무늬가 보이는 식이다. 악명 높은 `RANDU`는 3차원에서 점들이 겨우 15개 평면 위에 놓여, 이를 쓴 1970년대의 모의실험 결과들을 의심스럽게 만들었다.

    그 밖에 하위 비트의 주기가 짧다는 문제도 있다. $m = 2^{32}$인 생성기의 최하위 비트는 주기가 2다.

    **그럼에도 쓸 만한 이유.** 애초에 필요한 것은 "진짜 무작위"가 아니라 **통계적 성질이 무작위와 구별되지 않는 수열**이다. 그리고 결정론적이라는 점이 오히려 장점이 된다. 씨앗만 기록하면 결과가 완벽히 재현되고, 뒤에 나올 공통난수 같은 분산감소 기법도 재현성 위에서만 가능하다. 속도가 빠르고 상태가 작다는 것도 실용적 미덕이다.

    다만 현대의 기본값은 선형합동생성기가 아니다. NumPy의 `default_rng`가 쓰는 PCG64는 주기가 $2^{128}$이고 격자 구조 문제를 출력 함수로 흐트러뜨린다. 오래된 `np.random.seed` 계열이 쓰던 메르센 트위스터는 주기가 $2^{19937}-1$로 넉넉하지만 상태가 크고 몇몇 통계 검정을 통과하지 못한다. **암호학 용도로는 어느 쪽도 쓰면 안 된다.** 출력 몇 개를 보면 상태를 복원할 수 있기 때문이며, 그때는 `secrets` 모듈을 써야 한다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
$\int_a^b g(x)\,dx$를 균등난수로 추정하는 몬테카를로 적분의 추정량과 그 표준오차를 구하라. 오차가 $O(n^{-1/2})$인데도 고차원에서 사다리꼴 공식보다 나은 이유를 설명하라.

</div>

??? success "풀이"
    **추정량.** $U_i \sim \text{Uniform}(a,b)$가 독립일 때 밀도가 $1/(b-a)$이므로

    $$
    E[g(U)] = \int_a^b g(x)\frac{1}{b-a}dx = \frac{I}{b-a}, \qquad I := \int_a^b g(x)dx
    $$

    이다. 따라서

    $$
    \hat I_n = \frac{b-a}{n}\sum_{i=1}^n g(U_i)
    $$

    이 불편추정량이다. $\sigma_g^2 = \operatorname{Var}\{g(U)\}$라 하면

    $$
    \operatorname{Var}(\hat I_n) = \frac{(b-a)^2\sigma_g^2}{n}, \qquad \operatorname{SE}(\hat I_n) = \frac{(b-a)\sigma_g}{\sqrt n}
    $$

    이다. $\sigma_g$는 표본표준편차로 추정하며, 중심극한정리로 신뢰구간까지 붙일 수 있다는 것이 이 방법의 큰 장점이다. **오차의 크기를 추정값과 함께 얻는** 수치적분은 흔치 않다.

    **고차원에서의 우위.** 1차원에서는 몬테카를로가 형편없다. 사다리꼴 공식의 오차가 $O(n^{-2})$, 심프슨 공식이 $O(n^{-4})$인데 몬테카를로는 $O(n^{-1/2})$에 지나지 않는다.

    문제는 차원이다. $d$차원에서 격자법은 각 축을 $m$등분하므로 점의 개수가 $n = m^d$이고, 오차가 축 방향 간격 $h = m^{-1} = n^{-1/d}$의 거듭제곱이므로

    $$
    \text{사다리꼴 오차} = O(h^2) = O\!\left(n^{-2/d}\right)
    $$

    이 된다. **차원이 오르면 수렴 속도가 나빠진다.** $d = 4$에서 $n^{-1/2}$로 몬테카를로와 같아지고, $d > 4$부터는 몬테카를로가 이긴다.

    몬테카를로의 $O(n^{-1/2})$에는 $d$가 **전혀 들어 있지 않다**. 차원이 100이든 1000이든 같은 속도다. 상수 $\sigma_g$는 차원에 따라 커질 수 있지만 수렴 지수는 변하지 않는다. 베이즈 사후분포의 적분이나 금융 파생상품 가격처럼 차원이 수십에서 수백인 문제에서 몬테카를로 말고는 선택지가 없는 이유다.

    실제로는 순수 몬테카를로보다 나은 것들이 있다. 준몬테카를로는 난수 대신 저불일치 수열을 써서 매끄러운 피적분함수에 대해 거의 $O(n^{-1})$을 낸다. 중요도추출은 $g$가 큰 곳을 더 자주 뽑아 $\sigma_g$ 자체를 줄인다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
"원에 내접하는 정삼각형의 한 변보다 무작위 현이 더 길 확률"을 묻는 베르트랑의 역설에서 세 가지 다른 답이 나오는 이유를 균등분포의 관점에서 설명하라.

</div>

??? success "풀이"
    반지름 1인 원에서 내접 정삼각형의 한 변 길이는 $\sqrt3$이고, 이는 중심에서 거리 $1/2$인 현에 해당한다. 세 가지 자연스러운 "무작위"가 서로 다른 답을 준다.

    **(가) 끝점을 균등하게.** 한 끝점을 고정하고 다른 끝점의 각도를 $[0, 2\pi)$에서 균등하게 뽑는다. 현이 $\sqrt3$보다 길려면 각도가 가운데 $120^\circ$ 구간에 들어야 하므로 확률은 $1/3$이다.

    **(나) 중심까지의 거리를 균등하게.** 현의 방향을 고정하고 중심에서의 거리를 $[0,1]$에서 균등하게 뽑는다. 거리가 $1/2$ 미만이면 되므로 확률은 $1/2$이다.

    **(다) 중점을 원판 위에 균등하게.** 현의 중점을 원판 전체에서 넓이에 비례해 뽑는다. 중점이 반지름 $1/2$인 안쪽 원판에 들어가야 하므로 확률은 넓이의 비 $(1/2)^2 = 1/4$이다.

    **무엇이 문제인가.** 세 계산 모두 옳다. 틀린 것은 **"무작위 현"이라는 말 자체가 확률모형을 지정하지 않는다**는 점을 눈치채지 못한 것이다. 균등분포는 "무엇에 대해 균등한가"를 정해야 비로소 정의된다. 각도에 대해 균등한 것, 거리에 대해 균등한 것, 중점의 위치에 대해 균등한 것은 서로 다른 분포이며, 한 척도에서 균등한 분포를 비선형 변환하면 다른 척도에서는 균등하지 않다.

    이 역설은 두 가지를 가르쳐 준다. 첫째, **모수화가 바뀌면 "무정보 사전분포"도 바뀐다.** $\theta$에 균등한 사전분포는 $\theta^2$이나 $\log\theta$에 대해 균등하지 않으므로, 균등 사전분포를 "아무 정보도 넣지 않은 것"이라고 부르는 것은 정확하지 않다. 이 문제를 피하려고 고안된 것이 모수화 불변인 제프리스 사전분포다.

    둘째, 실제 문제에서는 물리적 절차가 모형을 결정한다. 원 위에 막대를 무작위로 던지는 방식이라면 (나)가 맞고(이 경우만 병진·회전 불변성을 만족한다), 원둘레에서 두 점을 고르는 방식이라면 (가)가 맞다. **"무작위"라고만 말하지 말고 어떻게 무작위인지를 적어야 한다**는 것이 실무적 교훈이다.

---

## 정리하며

- 균등분포는 구간의 모든 값에 동일한 확률을 부여하므로, 유계 구간 위에서 "최대로 정보가 없는" 분포이다.
- 표준 균등분포 $U(0,1)$은 역변환 방법을 통한 난수 생성의 기본 구성요소이다.
- 확률적분변환은 임의의 연속확률변수에 그 CDF를 적용하면 균등분포가 됨을 말해 준다.
- 단순함에도 불구하고 균등분포는 Monte Carlo 시뮬레이션과 계산통계학의 기초를 이룬다.
