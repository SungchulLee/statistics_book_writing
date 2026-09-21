# 지수분포

## 개요

**지수분포**는 포아송 과정에서 사건 사이의 시간을 모형화한다. 기하분포의 연속형 대응물이며, 무기억성을 갖는 유일한 연속분포이다. 도착 간 시간, 대기 시간, 부품 수명 모형화에 흔히 쓰인다.

4.2절의 연속분포는 지수분포에서 시작하는 사슬을 이룬다.

$$
\text{Exp}(\lambda) \;\longrightarrow\; N(\mu, \sigma^2) \;\longrightarrow\; \chi^2_d \;\longrightarrow\; t_d \;\longrightarrow\; F_{d_1, d_2}
$$

첫 고리에서 둘째 고리로 넘어가는 다리는 **더하기**다. 지수분포는 심하게 치우친 분포인데도 여러 개를 더하면 정규분포로 다가간다. 그 뒤로는 정규분포를 제곱하고 나누는 것만으로 나머지 세 분포가 차례로 나온다.

---

## 지수분포와 누적분포함수

<div class="defn" markdown>

### 정의 1. 지수분포 { .dfn }

확률변수 $X$가 비율 모수 $\lambda > 0$인 지수분포를 따른다는 것은 다음을 뜻한다:

$$
X \sim \text{Exponential}(\lambda), \qquad f(x) = \lambda e^{-\lambda x}, \quad x \geq 0
$$

**다른 모수화:** 어떤 교재에서는 척도 모수 $\beta = 1/\lambda$를 사용하여 $f(x) = \frac{1}{\beta}e^{-x/\beta}$로 쓴다. SciPy는 척도 모수화를 사용한다.

</div>

### CDF

$$
F(x) = 1 - e^{-\lambda x}, \quad x \geq 0
$$

### 생존함수

$$
S(x) = P(X > x) = e^{-\lambda x}
$$

---

## 성질

$$
\begin{aligned}
E[X] &= \frac{1}{\lambda} \\[4pt]
\text{Var}(X) &= \frac{1}{\lambda^2} \\[4pt]
\text{SD}(X) &= \frac{1}{\lambda} \\[4pt]
\text{Median} &= \frac{\ln 2}{\lambda}
\end{aligned}
$$

$\text{평균} = \text{표준편차} = 1/\lambda$라는 점에 주목하라. 지수분포의 두드러진 특징이다.

### 평균의 유도

$$
E[X] = \int_0^{\infty} x \lambda e^{-\lambda x}\,dx = \left[-x e^{-\lambda x}\right]_0^{\infty} + \int_0^{\infty} e^{-\lambda x}\,dx = \frac{1}{\lambda}
$$

### 분산의 유도

$$
E[X^2] = \int_0^{\infty} x^2 \lambda e^{-\lambda x}\,dx = \frac{2}{\lambda^2}
$$

$$
\text{Var}(X) = E[X^2] - (E[X])^2 = \frac{2}{\lambda^2} - \frac{1}{\lambda^2} = \frac{1}{\lambda^2}
$$

---

## 무기억성

지수분포는 무기억성을 갖는 **유일한** 연속분포이다:

$$
P(X > s + t \mid X > s) = P(X > t) \quad \text{for all } s, t \geq 0
$$

??? proof "증명"


    $$
    P(X > s + t \mid X > s) = \frac{P(X > s + t)}{P(X > s)} = \frac{e^{-\lambda(s+t)}}{e^{-\lambda s}} = e^{-\lambda t} = P(X > t)
    $$

    **해석:** 이미 $s$만큼의 시간을 기다렸더라도 남은 대기 시간의 분포는 방금 시작했을 때와 같다. 이 과정은 자신의 이력을 "잊어버린다".

    ---

## 포아송 과정과의 연결

사건이 비율 $\lambda$인 포아송 과정에 따라 도착하면:

$$
\begin{aligned}
\text{Number of events in } [0, t] &\sim \text{Poisson}(\lambda t) \\
\text{Time between consecutive events} &\sim \text{Exponential}(\lambda) \\
\text{Time to the } n\text{-th event} &\sim \text{Gamma}(n, \lambda)
\end{aligned}
$$

---

## 지수 확률변수의 최솟값

$X_1 \sim \text{Exp}(\lambda_1)$과 $X_2 \sim \text{Exp}(\lambda_2)$가 독립이면:

$$
\min(X_1, X_2) \sim \text{Exp}(\lambda_1 + \lambda_2)
$$

??? proof "증명"


    $$
    P(\min(X_1, X_2) > t) = P(X_1 > t) \cdot P(X_2 > t) = e^{-\lambda_1 t} \cdot e^{-\lambda_2 t} = e^{-(\lambda_1 + \lambda_2)t}
    $$

    이는 $n$개의 독립인 지수 확률변수로 일반화된다: $\min(X_1, \ldots, X_n) \sim \text{Exp}\left(\sum_{i=1}^n \lambda_i\right)$.

    ---

## 다음 고리: 더하면 정규분포로 간다

최솟값을 취하면 지수분포가 그대로 남지만, **더하면** 이야기가 달라진다.

$$
X_1 + X_2 + \cdots + X_n \sim \text{Gamma}(n, \lambda)
$$

감마분포의 모양은 $n$이 커질수록 점점 대칭인 종 모양이 된다. 중심극한정리를 쓰면 그 이유가 곧바로 설명된다. $E[X_i] = 1/\lambda$, $\text{Var}(X_i) = 1/\lambda^2$이므로

$$
\frac{\sum_{i=1}^n X_i - n/\lambda}{\sqrt n/\lambda} \;\xrightarrow{\;n \to \infty\;}\; N(0, 1)
$$

이다. 즉 $\text{Gamma}(n, \lambda) \approx N(n/\lambda,\ n/\lambda^2)$이다.

### 얼마나 많이 더해야 하는가

지수분포는 왜도가 2로, 연속분포 가운데 상당히 치우친 편이다. $n$개를 더한 합의 왜도는 $2/\sqrt n$로 줄어들지만 그 속도가 빠르지 않다.

| $n$ | 합의 왜도 |
|---|---|
| 1 | 2.00 |
| 5 | 0.89 |
| 20 | 0.45 |
| 50 | 0.28 |
| 100 | 0.20 |

"$n \ge 30$이면 중심극한정리가 듣는다"는 흔한 규칙이 지수분포 같은 치우친 모집단에서는 부족하다는 것을 알 수 있다. 자료가 치우쳐 있을수록 더 큰 표본이 필요하며, 특히 **꼬리 확률**을 다룰 때는 훨씬 더 그렇다. 3장의 베리–에센 정리가 이 수렴 속도를 정량적으로 말해 준다.

이 다리를 건너면 정규분포에 닿고, 그다음은 정규분포를 제곱하거나 나누는 것만 남는다.

---

## 문제

<div class="probox" markdown>

**문제:** <span class="diff easy" title="쉬움"></span> 어떤 트레이딩 데스크에 주문이 시간당 평균 12건 도착한다. 연속한 주문 사이의 시간이 10분을 넘을 확률은?

</div>

??? success "풀이"
    비율은 시간당 $\lambda = 12$, 즉 분당 $0.2$이다.

    $$
    P(X > 10) = e^{-0.2 \times 10} = e^{-2} \approx 0.1353
    $$

    주문 사이의 기대 시간: $E[X] = 1/0.2 = 5$분.
---

## Python: PDF, CDF, 표본추출

### PDF와 CDF

<div class="codebox" markdown>

#### 예제 1. 지수분포의 밀도함수와 분포함수 { .eg }

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

lam = 2.0                      # 비율모수. 단위 시간당 평균 2회 발생.
x = np.linspace(0, 4, 200)     # 지수분포는 x >= 0 에서만 정의된다

fig, ax = plt.subplots(figsize=(12, 3))
# scipy는 rate가 아니라 scale = 1/rate 를 받는다. 이 책에서 반복되는 함정이다.
# PDF는 x=0 에서 lam(=2)으로 시작해 단조 감소한다.
#   -> 지수분포에서 가장 있을 법한 대기시간은 **0에 가까운 값**이다.
# CDF는 0에서 1로 오르며 1 - e^{-lam x} 다.
ax.plot(x, stats.expon(scale=1/lam).pdf(x), label='PDF')
ax.plot(x, stats.expon(scale=1/lam).cdf(x), label='CDF')
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
plt.show()
```

![지수분포](./img/exponential_132.png)

</div>

### 비율에 따른 비교

<div class="codebox" markdown>

#### 예제 2. 비율모수에 따른 지수분포 비교 { .eg }

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

fig, ax = plt.subplots(figsize=(12, 3))
# lam 하나가 모든 것을 정한다. 평균 = 표준편차 = 1/lam 이다.
# lam이 커질수록 시작 높이가 높아지고 더 빨리 0으로 떨어진다.
# 네 곡선 모두 x=0 에서의 높이가 정확히 lam 이라는 점을 확인해 보라.
for lam in [0.5, 1.0, 2.0, 5.0]:
    x = np.linspace(0, 6, 200)
    ax.plot(x, stats.expon(scale=1/lam).pdf(x), label=f'λ={lam}')
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel('x')
ax.legend()
plt.show()
```

![지수분포](./img/exponential_150.png)

</div>

### 표본추출과 검증

<div class="codebox" markdown>

#### 예제 3. 지수 표본의 평균과 표준편차 확인 { .eg }

```python
import numpy as np
from scipy import stats

np.random.seed(42)
lam = 3.0
samples = stats.expon(scale=1/lam).rvs(100_000)

# 지수분포의 특징: 평균과 **표준편차**가 같다(둘 다 1/lam).
# 분산은 1/lam^2 이므로 평균과 다르다. 포아송(평균 = 분산)과 헷갈리기 쉽다.
print(f"Theoretical mean: {1/lam:.4f},  Sample mean: {samples.mean():.4f}")
print(f"Theoretical var:  {1/lam**2:.4f},  Sample var:  {samples.var():.4f}")
print(f"Mean ≈ SD: {np.isclose(samples.mean(), samples.std(), atol=0.01)}")
```

출력:

```
Theoretical mean: 0.3333,  Sample mean: 0.3320
Theoretical var:  0.1111,  Sample var:  0.1096
Mean ≈ SD: True
```

</div>

### 무기억성 확인하기

<div class="codebox" markdown>

#### 예제 4. 무기억성 확인하기 { .eg }

```python
import numpy as np
from scipy import stats

np.random.seed(42)
lam = 2.0
samples = stats.expon(scale=1/lam).rvs(1_000_000)

s = 0.5
# 무기억성: P(X > s+t | X > s) = P(X > t).
# "이미 0.5만큼 기다렸다"는 사실이 앞으로의 대기시간에 아무 정보도 주지 않는다.
# 연속분포 중에서 이 성질을 갖는 것은 **지수분포뿐이다.**
# (이산분포에서는 기하분포가 유일하다.)
for t in [0.25, 0.5, 1.0]:
    # samples > s 로 "0.5를 넘긴 표본만" 골라 조건을 건다
    conditional = np.mean(samples[samples > s] > s + t)
    unconditional = np.mean(samples > t)
    print(f"P(X>{s}+{t}|X>{s}) = {conditional:.4f},  P(X>{t}) = {unconditional:.4f}")
```

출력:

```
P(X>0.5+0.25|X>0.5) = 0.6066,  P(X>0.25) = 0.6068
P(X>0.5+0.5|X>0.5) = 0.3681,  P(X>0.5) = 0.3682
P(X>0.5+1.0|X>0.5) = 0.1360,  P(X>1.0) = 0.1355
```

</div>

### 포아송 과정 모의실험

<div class="codebox" markdown>

#### 예제 5. 포아송 과정 모의실험 { .eg }

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
lam = 3.0        # 단위 시간당 평균 3회 발생
n_events = 50

# 포아송 과정을 만드는 가장 쉬운 방법: 사건 **사이의 간격**을 지수분포에서 뽑고
# 누적합을 취해 도착 시각으로 바꾼다.
# 지수 간격 <-> 포아송 계수는 같은 과정의 두 얼굴이다.
inter_arrivals = stats.expon(scale=1/lam).rvs(n_events)
arrival_times = np.cumsum(inter_arrivals)

fig, ax = plt.subplots(figsize=(12, 3))
# 계수과정 N(t)는 사건이 일어날 때만 1씩 뛰는 계단함수다.
# where='post' 로 "그 시각에 뛰어오른 뒤 다음까지 유지"를 나타낸다.
ax.step(arrival_times, range(1, n_events + 1), where='post', lw=1.5)
ax.set_xlabel('Time')
ax.set_ylabel('Cumulative events')
ax.spines[['top', 'right']].set_visible(False)
plt.show()
```

![지수분포](./img/exponential_199.png)

</div>

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
부품 수명이 $T \sim \mathrm{Exp}(0.5)$(단위: 년)이다. (a) $P(T > 3)$. (b) $P(T > 3 \mid T > 2)$. (c) 독립인 부품 두 개에 대해 $\min(T_1, T_2)$의 분포와 기댓값.

</div>

??? success "풀이"
    (a) $P(T > 3) = e^{-1.5} \approx 0.223$.

    (b) 무기억성에 의해 $P(T > 3 \mid T > 2) = P(T > 1) = e^{-0.5} \approx 0.607$. 직접 계산하면 $P(T > 3)/P(T > 2) = e^{-1.5}/e^{-1} = e^{-0.5}$.

    (c) $\min(T_1, T_2) \sim \mathrm{Exp}(\lambda_1 + \lambda_2) = \mathrm{Exp}(1.0)$. $\mathbb{E}[\min] = 1$년(부품 하나일 때의 절반).

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
지수분포의 **무기억성을 증명**하고, 이 성질을 갖는 연속분포가 *유일*함을 보여라.

</div>

??? success "풀이"
    생존함수는 $\bar F(t) = e^{-\lambda t}$이다. 그러면

    $$
    P(T > s + t \mid T > s) = \bar F(s + t)/\bar F(s) = e^{-\lambda(s+t)}/e^{-\lambda s} = e^{-\lambda t} = P(T > t)
    $$

    $\square$

    **유일성:** $\bar F$가 연속이고 감소하며 $\bar F(0) = 1$이고 무기억성 $\bar F(s + t) = \bar F(s) \bar F(t)$를 만족한다고 하자. (연속성 아래에서 풀린) 코시 함수방정식에 의해 이런 함수는 어떤 $\lambda > 0$에 대한 $\bar F(t) = e^{-\lambda t}$뿐이다.

    따라서 지수분포는 무기억성을 갖는 유일한 연속분포이며, 이는 기하분포가 유일한 이산 무기억 분포인 것과 정확히 대응된다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
$\mathrm{Exp}(\lambda)$의 **PDF, 평균, 분산을 유도하라.**

</div>

??? success "풀이"
    PDF: CDF $F(t) = 1 - e^{-\lambda t}$를 미분하면 $t \ge 0$에 대해 $f(t) = \lambda e^{-\lambda t}$.

    평균: $\mathbb{E}[T] = \int_0^\infty t \lambda e^{-\lambda t} dt$. 부분적분하거나 꼬리 공식을 쓰면 $\mathbb{E}[T] = \int_0^\infty e^{-\lambda t} dt = 1/\lambda$.

    2차 적률: $\mathbb{E}[T^2] = \int_0^\infty t^2 \lambda e^{-\lambda t} dt = 2/\lambda^2$(부분적분을 두 번 사용).

    분산: $\mathrm{Var}(T) = 2/\lambda^2 - 1/\lambda^2 = 1/\lambda^2$.

    참고: 평균과 표준편차가 모두 $1/\lambda$인 것이 지수분포의 두드러진 특징이다. 변동계수 CV = 1이다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
**포아송 과정과의 연결.** 사건이 비율 $\lambda$인 포아송 과정에서 발생할 때 $k$번째 도착 시각 $T_k$의 분포를 유도하라.

</div>

??? success "풀이"
    $T_k = \sum_{i=1}^k X_i$이며, 여기서 $X_i$는 i.i.d. $\mathrm{Exp}(\lambda)$(도착 간 시간)이다.

    $k$개의 i.i.d. 지수 확률변수의 합은 **감마(Erlang) 분포**이다:

    $$
    T_k \sim \mathrm{Gamma}(\text{shape} = k, \text{rate} = \lambda)
    $$

    PDF: $t \ge 0$에 대해 $f_{T_k}(t) = \lambda^k t^{k-1} e^{-\lambda t} / (k-1)!$.

    $\mathbb{E}[T_k] = k/\lambda$, $\mathrm{Var}(T_k) = k/\lambda^2$.

    이는 지수 도착 간 시간과 포아송 계수를 잇는 근본적인 연결이며, 대기행렬과 신뢰성 분석에서 재생이론의 토대가 된다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
**최대가능도추정.** i.i.d. $T_1, \ldots, T_n \sim \mathrm{Exp}(\lambda)$가 주어졌을 때 MLE $\hat\lambda$를 유도하라.

</div>

??? success "풀이"
    가능도: $L(\lambda) = \prod_i \lambda e^{-\lambda T_i} = \lambda^n e^{-\lambda \sum T_i}$.

    로그가능도: $\ell(\lambda) = n \ln \lambda - \lambda \sum T_i$.

    도함수: $\ell'(\lambda) = n/\lambda - \sum T_i = 0 \Rightarrow \hat\lambda = n/\sum T_i = 1/\bar T$.

    **성질:**

    - $\hat\lambda$는 표본평균의 역수이다. $\mathbb{E}[T] = 1/\lambda$이므로 자연스러운 결과이다.
    - 점근적으로 불편이지만 유한표본에서는 약간 편향되어 있다: $n \ge 2$에 대해 $\mathbb{E}[\hat\lambda] = n\lambda/(n - 1)$.
    - 점근적으로 정규이며 $\sqrt n (\hat\lambda - \lambda) \xrightarrow{d} N(0, \lambda^2)$이다.

    표본평균의 역수는 여러 분포에서 비율 모수를 추정하는 표준적인 추정량이다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
**위험함수.** 위험률은 $h(t) = f(t)/\bar F(t)$로 정의된다. 지수분포의 위험률이 *상수*임을 보이고, 이것이 물리적으로 무엇을 뜻하는지 논하라.

</div>

??? success "풀이"
    $h(t) = \lambda e^{-\lambda t} / e^{-\lambda t} = \lambda$.

    **상수 위험률:** 순간 고장률 $h(t) = \lambda$가 $t$에 의존하지 않는다. 해석하자면, 아직 고장 나지 않은 부품이 다음 짧은 구간에서 고장 날 확률은 나이와 무관하게 언제나 같다.

    이는 **무기억성의 직접적인 표현**이다. 미래의 위험은 과거의 생존에 의존하지 않는다.

    **상수가 아닌 위험률과의 비교:**

    - **증가하는 위험률** (예: $k > 1$인 Weibull): 부품이 마모된다. 오래된 부품일수록 고장 나기 쉽다. 기계 부품이 그렇다.
    - **감소하는 위험률** ($k < 1$): 부품이 길들여진다. 오래된 부품일수록 고장이 덜 난다. 초기 결함을 넘긴 전자 부품이 그렇다.
    - **욕조 곡선**: 초기에 높고(길들이기) 중간이 평평하며 이후 증가한다(마모). 위의 두 경우가 결합된 형태이다.

    현실의 신뢰성이 정확히 지수분포를 따르는 경우는 드물지만, 지수분포는 유용한 기준선이다. (a) 모수가 평균 수명이라는 직접적인 의미를 갖고, (b) 수학적으로 다루기 쉬우며, (c) 무기억성이 "완전히 무작위한" 고장에 대응하여 고장 모형화의 자연스러운 귀무가설이 되기 때문이다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
$X \sim \text{Exp}(\lambda)$일 때 $N = \lfloor X \rfloor + 1$의 분포를 구하라. 이 결과가 "기하분포는 지수분포의 이산판"이라는 말과 어떻게 이어지는가?

</div>

??? success "풀이"
    $N = k$($k = 1, 2, \dots$)일 필요충분조건은 $k-1 \le X < k$이므로

    $$
    P(N = k) = F(k) - F(k-1) = e^{-\lambda(k-1)} - e^{-\lambda k} = \left(e^{-\lambda}\right)^{k-1}\left(1 - e^{-\lambda}\right)
    $$

    이다. $p = 1 - e^{-\lambda}$로 두면

    $$
    P(N = k) = (1-p)^{k-1}p
    $$

    로 정확히 $\text{Geometric}(p)$이다. "첫 성공까지의 시행 횟수" 판이다.

    **왜 자연스러운가.** 지수분포의 시간축을 길이 1인 칸으로 자르고 "이 칸 안에 사건이 있었는가"만 기록한 것이 $N$이다. 각 칸에서 사건이 일어날 확률이 $p = 1-e^{-\lambda}$로 같고, 무기억성 덕분에 칸들이 서로 독립이다. 독립적인 동전 던지기의 첫 성공까지 세는 것이 곧 기하분포다.

    두 분포의 무기억성도 서로 대응한다. $P(X > s+t \mid X > s) = P(X > t)$가 $P(N > m+n \mid N > m) = P(N > n)$이 되고, 기하분포는 무기억성을 갖는 **유일한 이산분포**다. 반대 방향으로 $\lambda$를 작게 하면서 칸을 잘게 쪼개면 기하분포가 지수분포로 수렴한다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
$T_1, \dots, T_n$이 독립이고 $\text{Exp}(\lambda)$를 따를 때 $2\lambda\sum_i T_i \sim \chi^2_{2n}$임을 보이고, 이를 이용해 $\lambda$의 정확한 95% 신뢰구간을 만들어라. $n = 20$, $\sum T_i = 87.4$일 때 값을 구하라.

</div>

??? success "풀이"
    **분포.** $\sum_i T_i \sim \text{Gamma}(\text{형상}=n,\ \text{비율}=\lambda)$이다. 감마분포는 척도 변환에 대해 닫혀 있으므로 $2\lambda\sum T_i \sim \text{Gamma}(\text{형상}=n,\ \text{척도}=2)$이고, 이것이 바로 자유도 $2n$인 카이제곱분포의 정의이다.

    **추축량.** $Q = 2\lambda\sum T_i$는 $\lambda$를 담고 있으면서 그 분포가 $\lambda$에 의존하지 않는다. 따라서

    $$
    P\!\left(\chi^2_{2n,\,0.025} \le 2\lambda\sum T_i \le \chi^2_{2n,\,0.975}\right) = 0.95
    $$

    이고, $\lambda$에 대해 풀면

    $$
    \left(\frac{\chi^2_{2n,\,0.025}}{2\sum T_i},\ \frac{\chi^2_{2n,\,0.975}}{2\sum T_i}\right)
    $$

    가 정확한 95% 신뢰구간이다.

    **수치.** $n=20$이므로 자유도가 40이고 $\chi^2_{40,0.025} = 24.433$, $\chi^2_{40,0.975} = 59.342$이다. $2\sum T_i = 174.8$이므로

    $$
    \hat\lambda = \frac{20}{87.4} = 0.2288, \qquad \text{95\% CI} = (0.1398,\ 0.3395)
    $$

    이다. 평균 $1/\lambda$의 구간은 양끝을 뒤집어 $(2.95,\ 7.15)$이다.

    두 가지를 눈여겨본다. 첫째, 구간이 $\hat\lambda$를 중심으로 **대칭이 아니다**. 오른쪽으로 더 길다. 정규근사로 만든 $\hat\lambda \pm 1.96\hat\lambda/\sqrt n$은 이 비대칭을 놓친다. 둘째, 이 구간은 근사가 아니라 **정확**하다. $n$이 아무리 작아도 포함확률이 정확히 0.95다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
수명 시험을 시각 $C$에서 중단했다. $n$개 가운데 $d$개가 고장 났고 나머지는 아직 살아 있다(우측 중도절단). $\lambda$의 최대가능도추정량을 구하고, 중도절단된 관측값을 그냥 버리면 무엇이 잘못되는지 설명하라.

</div>

??? success "풀이"
    고장 난 개체는 밀도 $f(t_i) = \lambda e^{-\lambda t_i}$로, 살아 있는 개체는 생존확률 $S(C) = e^{-\lambda C}$로 가능도에 들어간다. 관측시간을 $t_i$(고장이면 고장시각, 중도절단이면 $C$)라 하고 $\delta_i$를 고장 지시자라 하면

    $$
    L(\lambda) = \prod_{i=1}^n \left\{\lambda e^{-\lambda t_i}\right\}^{\delta_i}\left\{e^{-\lambda t_i}\right\}^{1-\delta_i} = \lambda^{d}\exp\!\left(-\lambda\sum_i t_i\right)
    $$

    이다. 로그를 취해 미분하면

    $$
    \ell'(\lambda) = \frac{d}{\lambda} - \sum_i t_i = 0 \implies \hat\lambda = \frac{d}{\sum_i t_i} = \frac{\text{고장 횟수}}{\text{총 노출시간}}
    $$

    이다. 분자에는 **사건 수**가, 분모에는 살아 있던 개체까지 포함한 **총 관찰시간**이 들어간다는 점이 핵심이다.

    **중도절단 자료를 버리면.** 고장 난 $d$개만 써서 $\hat\lambda = d/\sum_{\delta_i=1}t_i$로 계산하게 되는데, 분모에서 살아남은 개체들의 노출시간이 통째로 빠진다. 분모가 작아지므로 $\lambda$를 **과대추정**하고 평균수명을 과소추정한다. 게다가 이 편향은 표본을 키워도 사라지지 않는다.

    직관적으로도 그렇다. 시험을 일찍 끊을수록 오래 사는 개체가 더 많이 잘려 나가고, 관측된 고장은 짧은 수명 쪽에 치우친다. 살아 있는 개체가 주는 정보는 "적어도 $C$는 넘었다"이며, 이것도 엄연한 정보다. 생존분석 전체가 이 정보를 버리지 않으려고 만들어진 분야이며, 21장에서 다시 다룬다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
비율 $\lambda$인 포아송 과정에서 $[0, t]$ 동안 $n$개의 사건이 일어났다는 조건이 주어졌을 때, 그 도착시각 $(S_1, \dots, S_n)$의 조건부 결합분포가 $\text{Uniform}(0,t)$에서 뽑은 $n$개의 순서통계량과 같음을 보여라.

</div>

??? success "풀이"
    도착 간 시간 $X_1, \dots, X_{n+1}$이 독립이고 $\text{Exp}(\lambda)$를 따르며 $S_k = X_1 + \cdots + X_k$이다. $(X_1,\dots,X_{n+1})$의 결합밀도는

    $$
    \lambda^{n+1}\exp\!\left(-\lambda\sum_{i=1}^{n+1}x_i\right)
    $$

    이다. 변환 $(x_1,\dots,x_{n+1}) \mapsto (s_1,\dots,s_n, x_{n+1})$은 선형이고 야코비안의 절댓값이 1이므로, $0 < s_1 < \cdots < s_n$ 영역에서

    $$
    f(s_1,\dots,s_n,x_{n+1}) = \lambda^{n+1}\exp\!\left\{-\lambda(s_n + x_{n+1})\right\}
    $$

    이다. 사건 $\{N(t) = n\}$은 $\{S_n \le t < S_n + X_{n+1}\}$, 즉 $x_{n+1} > t - s_n$과 같으므로 $x_{n+1}$을 적분해 없애면

    $$
    f(s_1,\dots,s_n,\, N(t)=n) = \lambda^{n+1}e^{-\lambda s_n}\int_{t-s_n}^{\infty}e^{-\lambda x}dx = \lambda^{n}e^{-\lambda t}
    $$

    를 얻는다. $s_n$이 깨끗이 사라졌다.

    $P(N(t)=n) = e^{-\lambda t}(\lambda t)^n/n!$로 나누면

    $$
    f(s_1,\dots,s_n \mid N(t)=n) = \frac{\lambda^n e^{-\lambda t}}{e^{-\lambda t}(\lambda t)^n/n!} = \frac{n!}{t^n}, \qquad 0 < s_1 < \cdots < s_n < t
    $$

    이다. 이것이 정확히 $\text{Uniform}(0,t)$ 확률변수 $n$개의 순서통계량의 결합밀도이다($n!$은 순서를 매기는 가짓수, $1/t^n$은 각 변수의 밀도). $\square$

    **뜻.** $\lambda$가 결과 식에서 완전히 사라진 것이 핵심이다. **몇 개가 일어났는지를 알고 나면, 그것들이 언제 일어났는지에 대해 포아송 과정은 "아무 선호도 없다".** 사건들이 구간 안에 완전히 무작위로 흩어져 있다는 뜻이고, 이것이 포아송 과정을 "완전 무작위"의 표준으로 삼는 이유다.

    쓸모도 많다. 첫째, 포아송 과정을 모의실험하는 가장 빠른 방법을 준다. $N \sim \text{Poisson}(\lambda t)$를 뽑은 뒤 균등난수 $N$개를 뽑아 정렬하면 끝이다. 도착 간 시간을 하나씩 누적하는 것보다 벡터화하기 좋다. 둘째, 시간에 따라 비율이 변하는 비동질 포아송 과정의 표집(잔기 방법)과 공간통계의 완전공간랜덤성 검정이 모두 이 성질에 기댄다.

---

## 정리하며

- 지수분포는 사건 사이의 대기 시간을 모형화하며, 비율 $\lambda$(또는 척도 $1/\lambda$)로 모수화된다.
- 무기억성을 갖는 유일한 연속분포이다. 남은 대기 시간은 이미 얼마나 기다렸는지와 무관하다.
- 포아송 과정과 직접 연결된다. 포아송 계수와 지수 도착 간 시간은 같은 현상을 보는 두 관점이다.
- 독립인 지수 확률변수들의 최솟값은 다시 지수분포이며, 비율은 합해진다.
- SciPy에서는 비율 모수화를 다루려면 `stats.expon(scale=1/lambda)`를 사용한다.
