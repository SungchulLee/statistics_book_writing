# 지수 밀도함수

## 개요

**지수분포**는 포아송 과정에서 첫 사건이 일어날 때까지의 대기 시간을 모형화한다. 기하분포의 연속형 대응물이며, **무기억성**을 갖는 유일한 연속분포이다.

비율 모수 $\lambda > 0$인 PDF는 다음과 같다:

$$
f(x) = \lambda\, e^{-\lambda x}, \qquad x \ge 0
$$

| 성질 | 값 |
|---|---|
| 평균 | $1/\lambda$ |
| 분산 | $1/\lambda^2$ |
| 중앙값 | $\ln(2)/\lambda$ |
| 최빈값 | 0 |

---

## SciPy 모수화

SciPy는 **척도** 모수화를 사용한다: `stats.expon(scale=1/lambda)`.

<div class="codebox" markdown>

### 예제 1. 비율모수에 따른 지수 밀도함수 { .eg }

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# 비율모수 lambda 를 셋 준비한다. lambda 가 클수록 사건이 자주 일어난다.
lambdas = [0.5, 1.0, 2.0]
x = np.linspace(0, 6, 300)

fig, ax = plt.subplots(figsize=(12, 4))
for lam in lambdas:
    # scipy는 비율(rate) lambda 가 아니라 **척도(scale) = 1/lambda** 를 받는다.
    # scale이 곧 평균이다. lambda=2 이면 평균 대기시간이 0.5다.
    rv = stats.expon(scale=1 / lam)
    # 밀도는 x=0에서 lambda 로 시작해 지수적으로 떨어진다.
    # lambda 가 클수록 시작점이 높고 더 가파르게 준다.
    ax.plot(x, rv.pdf(x), label=rf'$\lambda={lam}$')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Exponential Distribution — PDF')
ax.legend()
plt.tight_layout()
plt.show()
```

![Exponential Distribution — PDF](./img/exponential_pdf_26.png)

$\lambda$가 클수록 사건이 더 자주 일어나므로 분포가 0 근처에 더 몰린다.

</div>

---

## 무기억성

지수분포는 다음을 만족한다:

$$
P(X > s + t \mid X > s) = P(X > t) \qquad \text{for all } s, t \ge 0
$$

남은 대기 시간이 이미 얼마나 기다렸는지와 무관하다는 뜻이다. 이 과정에는 기억이 없다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
지수분포의 무기억성을 증명하라.

</div>

??? success "풀이"
    생존함수는 $S(x) = e^{-\lambda x}$이다. 그러면:

    $$
    P(X > s + t \mid X > s) = \frac{P(X > s + t)}{P(X > s)} = \frac{e^{-\lambda(s+t)}}{e^{-\lambda s}} = e^{-\lambda t} = P(X > t)
    $$

    $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
고객 도착이 시간당 $\lambda = 3$인 포아송 과정을 따를 때, 다음 고객을 30분 넘게 기다릴 확률은 얼마인가?

</div>

??? success "풀이"
    도착 간 시간은 (시간 단위로) $X \sim \text{Exp}(\lambda = 3)$이다. 30분은 $t = 0.5$시간이다.

    $$
    P(X > 0.5) = e^{-3 \times 0.5} = e^{-1.5} \approx 0.2231
    $$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
지수분포가 무기억성을 갖는 유일한 연속분포임을 보여라.

</div>

??? success "풀이"
    모든 $s, t \ge 0$에 대해 $P(X > s + t) = P(X > s) \cdot P(X > t)$라 하자. $g(t) = P(X > t)$로 두면 $g(0) = 1$이고 $g$는 감소하며 $g(s+t) = g(s)g(t)$이다.

    $g(0) = 1$인 함수방정식 $g(s+t) = g(s)g(t)$의 연속인 해는 어떤 $\lambda > 0$에 대한 $g(t) = e^{-\lambda t}$뿐이다. 이는 지수분포의 생존함수이다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
PDF로부터 지수분포의 CDF, 중앙값, 평균을 유도하라.

</div>

??? success "풀이"
    **CDF:**

    $$
    F(x) = \int_0^x \lambda e^{-\lambda t}\,dt = 1 - e^{-\lambda x}
    $$

    **중앙값:** $F(m) = 0.5$를 풀면:

    $$
    1 - e^{-\lambda m} = 0.5 \implies m = \frac{\ln 2}{\lambda}
    $$

    **평균:**

    $$
    E[X] = \int_0^{\infty} x\lambda e^{-\lambda x}\,dx = \frac{1}{\lambda}
    $$

    (부분적분으로 구한다.)

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
평균 대기시간이 8분인 지수분포를 SciPy로 만들려 한다. `stats.expon`에 어떤 인자를 넘겨야 하는가? 비율모수 $\lambda$는 얼마인가? 또 대기시간이 평균인 8분을 넘길 확률을 구하라.

</div>

??? success "풀이"
    SciPy는 척도 모수화를 쓰고 척도가 곧 평균이므로 `stats.expon(scale=8)`이다. 비율모수는 그 역수인 $\lambda = 1/8 = 0.125$(분당 0.125회)이다.

    $$
    P(X > 8) = e^{-\lambda \cdot 8} = e^{-1} \approx 0.3679
    $$

    평균을 넘길 확률이 절반이 아니라 0.368이다. 분포가 오른쪽으로 치우쳐 있어 중앙값 $8\ln 2 \approx 5.55$분이 평균보다 작기 때문이다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
$E[X^2]$을 직접 적분해 구하고, 이로부터 $\operatorname{Var}(X) = 1/\lambda^2$임을 보여라.

</div>

??? success "풀이"
    부분적분을 두 번 하거나 감마적분 $\int_0^\infty x^{n} e^{-\lambda x}\,dx = n!/\lambda^{n+1}$을 쓰면

    $$
    E[X^2] = \int_0^\infty x^2 \lambda e^{-\lambda x}\,dx = \lambda \cdot \frac{2!}{\lambda^3} = \frac{2}{\lambda^2}
    $$

    이다. 따라서

    $$
    \operatorname{Var}(X) = E[X^2] - (E[X])^2 = \frac{2}{\lambda^2} - \frac{1}{\lambda^2} = \frac{1}{\lambda^2}
    $$

    이다. 표준편차가 평균과 같다는 점이 지수분포의 특징이다. 변동계수가 항상 1이다. $\square$

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
$X_1 \sim \text{Exp}(\lambda_1)$과 $X_2 \sim \text{Exp}(\lambda_2)$가 독립이라 하자. $M = \min(X_1, X_2)$의 분포를 구하고, $P(X_1 < X_2)$를 구하라.

</div>

??? success "풀이"
    최솟값이 $t$를 넘으려면 둘 다 $t$를 넘어야 한다. 독립이므로

    $$
    P(M > t) = P(X_1 > t)\,P(X_2 > t) = e^{-\lambda_1 t} e^{-\lambda_2 t} = e^{-(\lambda_1 + \lambda_2) t}
    $$

    이다. 즉 $M \sim \text{Exp}(\lambda_1 + \lambda_2)$이다. 비율이 더해진다.

    다음으로 $X_1$의 밀도로 조건을 걸어 적분하면

    $$
    P(X_1 < X_2) = \int_0^\infty \lambda_1 e^{-\lambda_1 x} \, e^{-\lambda_2 x}\,dx = \frac{\lambda_1}{\lambda_1 + \lambda_2}
    $$

    이다. 두 창구 중 어느 쪽이 먼저 끝나는지가 비율에 비례한다는 뜻이다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
위험함수는 $h(x) = f(x)/S(x)$로 정의한다. 지수분포의 위험함수를 구하고, 그 결과가 무기억성과 어떻게 이어지는지 설명하라.

</div>

??? success "풀이"
    $f(x) = \lambda e^{-\lambda x}$이고 $S(x) = e^{-\lambda x}$이므로

    $$
    h(x) = \frac{\lambda e^{-\lambda x}}{e^{-\lambda x}} = \lambda
    $$

    이다. 위험이 $x$에 무관한 상수다.

    위험함수는 "지금까지 살아남았다는 조건 아래 바로 다음 순간에 고장 날 순간 비율"이다. 이것이 상수라는 말은 부품이 새것이든 1년을 썼든 앞으로의 고장 위험이 같다는 뜻이고, 그것이 곧 무기억성이다. 실제 기계 부품은 노화하므로 $h$가 증가하는데, 그런 경우를 담으려고 형상모수로 위험을 증가·감소시킬 수 있게 만든 것이 와이불분포다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
$U \sim \text{Uniform}(0, 1)$일 때 $X = -\ln(U)/\lambda$가 $\text{Exp}(\lambda)$를 따름을 보여라. 역변환 공식 $F^{-1}(u) = -\ln(1-u)/\lambda$와 견주어 보라.

</div>

??? success "풀이"
    $x \ge 0$에 대해

    $$
    P(X \le x) = P\!\left(-\frac{\ln U}{\lambda} \le x\right) = P(\ln U \ge -\lambda x) = P(U \ge e^{-\lambda x}) = 1 - e^{-\lambda x}
    $$

    이다. 이는 $\text{Exp}(\lambda)$의 CDF이다.

    $F(x) = 1 - e^{-\lambda x}$를 뒤집으면 $F^{-1}(u) = -\ln(1-u)/\lambda$이므로 정석은 $-\ln(1-U)/\lambda$이다. 그런데 $U$가 균등분포이면 $1 - U$도 같은 균등분포이므로 둘은 같은 분포를 낳는다. 뺄셈 한 번을 아끼려고 실제 구현에서는 $-\ln(U)/\lambda$를 쓴다. $\square$

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
$E[X \mid X > s] = s + 1/\lambda$임을 보여라. 또 $c > 0$에 대해 $cX \sim \text{Exp}(\lambda/c)$임을 보이고, 이것이 SciPy의 척도 모수화와 어떻게 맞아떨어지는지 설명하라.

</div>

??? success "풀이"
    **조건부 기대값.** 무기억성에 따라 모든 $t \ge 0$에서 $P(X - s > t \mid X > s) = e^{-\lambda t}$이므로, 조건부 초과분 $X - s \mid X > s$ 자체가 다시 $\text{Exp}(\lambda)$를 따른다. 따라서

    $$
    E[X \mid X > s] = s + E[X - s \mid X > s] = s + \frac{1}{\lambda}
    $$

    이다. 이미 $s$만큼 기다렸다는 사실이 앞으로 기다릴 기대시간을 전혀 줄여 주지 않는다.

    **척도 변환.** $c > 0$에 대해

    $$
    P(cX > t) = P\!\left(X > \frac{t}{c}\right) = e^{-\lambda t / c} = e^{-(\lambda/c) t}
    $$

    이므로 $cX \sim \text{Exp}(\lambda/c)$이다. 즉 지수분포족은 척도 변환에 대해 닫혀 있고, $\lambda$는 척도의 역수처럼 움직인다.

    그래서 $X = Z/\lambda$($Z \sim \text{Exp}(1)$) 꼴로 언제나 쓸 수 있고, SciPy가 모든 분포에 공통으로 제공하는 `scale` 인자 하나로 지수분포를 다룰 수 있다. `scale=1/lambda`가 붙는 이유가 여기에 있다. $\square$

---

## 정리하며

지수분포는 포아송 과정에서 **첫 사건까지의 대기 시간**이며, 기하분포의 연속형 대응물이다.

- **모수 하나가 전부다.** 평균 $1/\lambda$, 분산 $1/\lambda^2$, 중앙값 $\ln 2/\lambda$, 최빈값 $0$. **평균이 중앙값보다 크다**는 사실이 오른쪽으로 치우친 모양을 그대로 말해 준다.
- **최빈값이 0이라는 점을 놓치기 쉽다.** 평균 대기시간이 10분이어도 가장 흔한 대기시간은 0에 가깝다.
- **무기억성** $P(X>s+t\mid X>s)=P(X>t)$ **를 갖는 유일한 연속분포다.** 이미 기다린 시간이 앞으로 기다릴 시간에 아무 정보도 주지 않는다. 기계 부품처럼 노화하는 대상에는 이 가정이 맞지 않으며, 그때 쓰는 것이 뒤에 나올 **와이불분포**다.
- **SciPy 는 척도 모수화를 쓴다.** `stats.expon(scale=1/lambda)` 이며, 비율 $\lambda$ 를 그대로 넣으면 뜻이 뒤집힌다.

다음 절 **정규분포**로 넘어간다. 이 책에서 가장 많이 등장하는 분포이며, 중심극한정리 덕분에 원래 분포가 무엇이든 표본평균이 향하는 곳이다.
