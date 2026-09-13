# 정규분포의 생존함수

## 개요

**생존함수**(SF)는 CDF의 여집합에 해당한다:

$$
S(x) = P(X > x) = 1 - F(x)
$$

$X$가 주어진 문턱값을 넘을 확률을 준다. 생존함수는 신뢰성 공학, 보험계리학, 임상시험처럼 사건 발생까지의 시간이나 초과 확률이 관심 대상인 분야에서 널리 쓰인다.

---

## 코드

<div class="codebox" markdown>

### 예제 1. 생존함수로 오른쪽 꼬리 보기 { .eg }

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

mu, sigma = 0, 1
dist = stats.norm(loc=mu, scale=sigma)

x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 400)
# 생존함수 SF(x) = P(X > x) = 1 - CDF(x).
# 수학적으로는 CDF의 여집합일 뿐이지만, 계산 방식이 다르다.
# scipy는 sf를 1-cdf로 계산하지 않고 꼬리를 직접 적분하므로
# 아주 작은 확률에서도 정밀도를 잃지 않는다(아래 절 참고).
cdf = dist.cdf(x)
sf = dist.sf(x)

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(x, cdf, lw=2, label='CDF  P(X ≤ x)')
ax.plot(x, sf, lw=2, label='SF   P(X > x)')
# 두 곡선이 만나는 지점을 표시한다.
# 대칭분포에서는 평균에서 CDF = SF = 0.5 로 교차한다.
ax.axvline(0, ls=':', color='gray', alpha=0.6)
ax.axhline(0.5, ls=':', color='gray', alpha=0.6)
ax.annotate("CDF + SF = 1", xy=(1.2, 0.5), fontsize=12,
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray'))
ax.set_xlabel('x')
ax.set_ylabel('Probability')
ax.set_ylim(-0.03, 1.03)
ax.legend(loc='center left', frameon=False)
ax.set_title(f"Normal({mu}, {sigma}) — CDF vs Survival Function")
ax.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()
```

![정규분포의 생존함수](./img/normal_sf_17.png)

</div>

---

## 생존함수를 쓰는 이유

상단꼬리 확률이 극단적으로 작을 때 $1 - F(x)$를 직접 계산하면 $F(x)$가 1에 매우 가까워 부동소수점 상쇄가 일어날 수 있다. 전용 메서드 `sf()`는 꼬리 확률을 직접 계산하여 이 문제를 피한다.

<div class="codebox" markdown>

### 예제 2. 생존함수가 수치적으로 더 정확한 이유 { .eg }

```python
from scipy import stats

# 꼬리 확률을 두 가지 방법으로 구해 비교한다.
#   나쁜 방법: 1 - CDF.  CDF가 1에 아주 가까우면 뺄셈에서 유효숫자가 날아간다(상쇄).
#   좋은 방법: SF.       꼬리를 직접 계산하므로 상쇄가 일어나지 않는다.
for x in (6, 8, 10, 12):
    bad = 1 - stats.norm.cdf(x)
    good = stats.norm.sf(x)
    print(f"x={x:>3}:  1-cdf = {bad:.6e}   sf = {good:.6e}")
```

출력:

```
x=  6:  1-cdf = 9.865877e-10   sf = 9.865876e-10
x=  8:  1-cdf = 6.661338e-16   sf = 6.220961e-16
x= 10:  1-cdf = 0.000000e+00   sf = 7.619853e-24
x= 12:  1-cdf = 0.000000e+00   sf = 1.776482e-33
```

세 단계로 나빠지는 것이 보인다.

- **$x = 6$**: 아직 괜찮다. 마지막 자리만 다르다.
- **$x = 8$**: 유효숫자가 이미 두 자리 넘게 어긋났다($6.661$ 대 $6.221$).
- **$x \ge 10$**: `1 - cdf`가 **정확히 0**이 된다. 확률이 0이 아닌데 0이라고 답하는 것이다.

원인은 배정밀도 부동소수점이 1 근처에서 약 $10^{-16}$ 간격으로만 값을 구별할 수 있다는 데 있다. $\Phi(10) = 1 - 7.6 \times 10^{-24}$은 그 간격보다 훨씬 1에 가까우므로 **컴퓨터 안에서는 그냥 1로 저장된다.** 1에서 1을 빼면 0이다.

</div>

!!! danger "꼬리 확률에는 언제나 `sf`를 써라"
    $p$-값 계산이 대표적이다. $p$-값은 본질적으로 꼬리 확률이므로 `1 - cdf`로 구하면 아주 작은 $p$-값이 0으로 보고된다. 유전체학처럼 $p < 10^{-20}$을 다루는 분야에서는 치명적이다.

    같은 이유로 로그가 필요하면 `np.log(sf(x))`가 아니라 **`logsf(x)`** 를 쓴다. `sf`조차 언더플로로 0이 되는 극단적인 영역에서도 로그값은 정상적으로 나온다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
$Z \sim N(0,1)$에 대해 CDF와 SF를 각각 사용하여 $P(Z > 2)$를 계산하고 두 결과가 일치함을 확인하라.

</div>

??? success "풀이"
    CDF로: $P(Z > 2) = 1 - \mathcal{N}(2) = 1 - 0.9772 = 0.0228$.

    SF로: `stats.norm.sf(2) = 0.0228`.

    둘 다 같은 결과를 준다. $P(Z > 8)$처럼 극단적인 값에서는 SF 방식이 수치적으로 더 낫다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
임의의 연속확률변수에 대해 모든 $x$에서 $S(x) + F(x) = 1$임을 보여라.

</div>

??? success "풀이"
    정의에 의해:

    $$
    F(x) + S(x) = P(X \le x) + P(X > x) = P(X \in (-\infty, x]) + P(X \in (x, \infty))
    $$

    $(-\infty, x]$와 $(x, \infty)$는 $\mathbb{R}$의 분할이므로 두 확률의 합은 1이다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
어떤 부품은 응력 $X \sim N(500, 2500)$이 문턱값 600을 넘으면 고장 난다. 고장 확률은 얼마인가?

</div>

??? success "풀이"
    표준화하면 $Z = (600 - 500)/50 = 2$이다.

    $$
    P(X > 600) = P(Z > 2) = S(2) \approx 0.0228
    $$

    부품의 약 2.3%가 고장 난다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
**위험함수**는 $h(x) = f(x)/S(x)$로 정의된다. 표준정규분포에 대해 $h(0)$을 계산하고 $x > 0$에서 $h(x)$가 증가하는 이유를 설명하라.

</div>

??? success "풀이"
    $x = 0$에서 $f(0) = 1/\sqrt{2\pi} \approx 0.3989$이고 $S(0) = 0.5$이다.

    $$
    h(0) = \frac{0.3989}{0.5} \approx 0.7979
    $$

    $x > 0$에서는 $S(x)$가 $f(x)$보다 빠르게 감소한다. 분모는 ($x$ 위에 남은 값이 줄어들어) 작아지는 반면, 분자인 밀도도 감소하지만 상대적으로는 더 천천히 줄어들기 때문이다. 그래서 $h(x)$가 증가한다. $x$까지 생존했다는 조건 아래 $x$에서 "고장"이 날 조건부 확률이 $x$와 함께 커지는 것이다. 정규분포는 **증가하는 고장률**을 갖는다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
$x = 20, 40$에서 `np.log(stats.norm.sf(x))`와 `stats.norm.logsf(x)`를 견주어라. `sf`조차 부족해지는 지점은 어디이고 왜 그런가?

</div>

??? success "풀이"
    $x = 20$에서는 `sf(20) = 2.754e-89`이고 두 방법 모두 $-203.917$을 준다. 아직 문제가 없다.

    $x = 40$에서는 사정이 달라진다. 참값이 $S(40) \approx 10^{-350}$쯤인데, 배정밀도 부동소수점이 나타낼 수 있는 가장 작은 양수가 약 $5 \times 10^{-324}$이다. 그보다 작으므로 **언더플로**가 일어나 `sf(40)`이 정확히 0이 되고, 로그를 취하면 $-\infty$가 나온다. 반면 `logsf(40)`은 $-804.608$을 제대로 준다.

    `logsf`는 확률을 구한 뒤 로그를 취하는 것이 아니라 처음부터 로그 척도에서 계산한다. 지수 부분의 $-x^2/2$를 그대로 다루므로 언더플로가 생길 여지가 없다.

    정리하면 정밀도의 층이 세 겹이다. `1 - cdf`는 $x \approx 8$에서 무너지고, `sf`는 $x \approx 38$에서 언더플로하며, `logsf`는 그 너머에서도 버틴다. 가능도 계산이 로그 척도에서 이루어지는 이유도 같다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
$x > 0$에 대한 밀 비 부등식

$$
\frac{\varphi(x)}{x}\left(1 - \frac{1}{x^2}\right) < S(x) < \frac{\varphi(x)}{x}
$$

를 부분적분으로 유도하고, $x = 3$과 $x = 5$에서 상대오차를 확인하라.

</div>

??? success "풀이"
    **위쪽 경계.** $t > x > 0$에서 $t/x > 1$이므로

    $$
    S(x) = \int_x^\infty \varphi(t)\,dt < \int_x^\infty \frac{t}{x}\varphi(t)\,dt = \frac{1}{x}\left[-\varphi(t)\right]_x^\infty = \frac{\varphi(x)}{x}
    $$

    이다. $\varphi'(t) = -t\varphi(t)$를 쓴 것이다.

    **아래쪽 경계.** $\int_x^\infty t^{-2}\,t\varphi(t)\,dt$에 같은 요령을 쓰면 부분적분으로

    $$
    \int_x^\infty \frac{\varphi(t)}{t^2}dt = \frac{\varphi(x)}{x^3} - 3\int_x^\infty \frac{\varphi(t)}{t^4}dt < \frac{\varphi(x)}{x^3}
    $$

    를 얻는다. 한편 $\varphi(t)(1 - 3t^{-4})$를 적분하는 식으로 정리하면

    $$
    S(x) = \frac{\varphi(x)}{x} - \int_x^\infty \frac{\varphi(t)}{t^2}dt > \frac{\varphi(x)}{x} - \frac{\varphi(x)}{x^3} = \frac{\varphi(x)}{x}\left(1 - \frac{1}{x^2}\right)
    $$

    이다. $\square$

    **수치 확인.**

    | $x$ | 아래 경계 | 참값 $S(x)$ | 위 경계 | 위 경계의 상대오차 |
    |---|---|---|---|---|
    | 3 | $1.3131 \times 10^{-3}$ | $1.3499 \times 10^{-3}$ | $1.4773 \times 10^{-3}$ | 9.4% |
    | 5 | $2.8545 \times 10^{-7}$ | $2.8665 \times 10^{-7}$ | $2.9734 \times 10^{-7}$ | 3.7% |

    $x$가 커질수록 경계가 좁아지고, $S(x) \sim \varphi(x)/x$라는 점근식이 꼬리의 감소 속도를 알려 준다. 정규 꼬리가 $e^{-x^2/2}$ 꼴로 **초지수적으로** 줄어든다는 사실이 여기서 보이며, 이것이 극단값이 사실상 나타나지 않는 이유이자 실제 자료의 두꺼운 꼬리를 정규모형이 과소평가하는 이유이기도 하다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff easy" title="쉬움"></span>
검정통계량 $z = 2.5$를 얻었다. 단측 $p$-값과 양측 $p$-값을 각각 생존함수로 계산하라. 양측에서 왜 2를 곱하는가?

</div>

??? success "풀이"
    단측(오른쪽 꼬리) $p$-값은 관측값보다 극단적인 값이 나올 확률이므로

    $$
    p_{\text{단측}} = S(2.5) = 0.00621
    $$

    이다. 양측검정에서는 "극단적"이 $|Z| \ge 2.5$를 뜻하므로

    $$
    p_{\text{양측}} = P(|Z| \ge 2.5) = S(2.5) + F(-2.5) = 2\,S(2.5) = 0.01242
    $$

    이다. 표준정규분포가 대칭이라 두 꼬리의 확률이 같으므로 2를 곱하면 된다.

    대칭이 아닌 분포에서는 이 곱하기가 성립하지 않는다. 카이제곱 검정이나 $F$ 검정처럼 한쪽 꼬리만 쓰는 검정에 2를 곱하는 것은 명백한 오류이고, 이항검정처럼 이산이면서 비대칭인 경우에는 양측 $p$-값의 정의부터 따로 정해야 한다.

    코드로는 `2 * stats.norm.sf(abs(z))`로 쓴다. `2 * (1 - stats.norm.cdf(abs(z)))`는 $|z|$가 클 때 0을 준다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
응력이 $X \sim N(500, 50^2)$인 부품이 550을 견디고 있다는 사실을 알았다. 이 부품이 600도 견디지 못할 조건부 확률을 구하라. 같은 물음을 지수분포에 대해 답하면 무엇이 달라지는가?

</div>

??? success "풀이"
    조건부 생존확률은 생존함수의 비이다.

    $$
    P(X > 600 \mid X > 550) = \frac{S(600)}{S(550)} = \frac{0.02275}{0.15866} = 0.1434
    $$

    따라서 고장 날 확률은 $1 - 0.1434 = 0.857$이다.

    조건 없이 보면 $P(X > 600) = 0.0228$에 지나지 않는데, 이미 550을 넘었다는 정보가 더해지자 600을 넘을 확률이 0.143으로 여섯 배 넘게 올라갔다. 정규분포는 무기억성을 갖지 않으므로 과거 정보가 미래 예측을 바꾼다.

    지수분포라면 무기억성에 따라

    $$
    P(X > 600 \mid X > 550) = P(X > 50) = e^{-50\lambda}
    $$

    로, 550까지 버텼다는 사실이 아무 정보도 주지 않는다. 시작점이 어디든 남은 수명의 분포가 같다. 이 차이가 신뢰성 모형을 고를 때의 핵심 갈림길이며, 노화를 반영하려면 정규나 와이불처럼 위험함수가 증가하는 분포를 써야 한다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
표준정규분포의 위험함수 $h(x) = \varphi(x)/S(x)$가 모든 실수 $x$에서 순증가함을 증명하라.

</div>

??? success "풀이"
    미분하면 $\varphi'(x) = -x\varphi(x)$이고 $S'(x) = -\varphi(x)$이므로

    $$
    h'(x) = \frac{\varphi'(x)S(x) + \varphi(x)^2}{S(x)^2} = \frac{\varphi(x)\left\{\varphi(x) - x S(x)\right\}}{S(x)^2}
    $$

    이다. $\varphi(x) > 0$이고 $S(x)^2 > 0$이므로 $h'(x) > 0$은 다음과 동치이다.

    $$
    g(x) := \varphi(x) - x\,S(x) > 0
    $$

    $x \le 0$이면 $-xS(x) \ge 0$이므로 $g(x) > 0$이 자명하다. $x > 0$이면 연습문제 6의 위쪽 경계 $S(x) < \varphi(x)/x$에서 $xS(x) < \varphi(x)$이므로 역시 $g(x) > 0$이다.

    경계를 쓰지 않고 직접 보이려면 $g$를 미분한다.

    $$
    g'(x) = -x\varphi(x) - S(x) + x\varphi(x) = -S(x) < 0
    $$

    이므로 $g$는 순감소하고, $x \to \infty$에서 $\varphi(x) \to 0$이고 $xS(x) \to 0$(밀 비 부등식)이므로 $g(x) \to 0$이다. 순감소하면서 0으로 수렴하니 모든 $x$에서 $g(x) > 0$이다. 따라서 $h'(x) > 0$이다. $\square$

    더 일반적으로, **밀도가 로그오목이면 위험함수가 증가한다**는 정리가 있다. $\ln\varphi(x) = -x^2/2 + \text{상수}$는 오목하므로 정규분포가 이 조건을 만족한다. 지수분포는 $\ln f$가 선형이라 로그오목과 로그볼록의 경계에 있고, 그래서 위험함수가 증가도 감소도 하지 않는 상수가 된다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
부품 세 개의 수명이 독립이고 각각 생존함수 $S_i(t)$를 가질 때, 세 부품이 직렬로 연결된 시스템과 병렬로 연결된 시스템의 생존함수를 각각 구하라. 각 부품이 $S_i(t) = 0.9$일 때의 값을 견주어라.

</div>

??? success "풀이"
    **직렬.** 하나라도 고장 나면 시스템이 멈추므로 모든 부품이 살아 있어야 한다. 독립이므로

    $$
    S_{\text{직렬}}(t) = \prod_{i=1}^3 S_i(t)
    $$

    이다. $S_i = 0.9$이면 $0.9^3 = 0.729$이다.

    **병렬.** 하나만 살아 있어도 시스템이 돌아가므로, 여집합을 쓰면

    $$
    S_{\text{병렬}}(t) = 1 - \prod_{i=1}^3 \{1 - S_i(t)\}
    $$

    이다. $S_i = 0.9$이면 $1 - 0.1^3 = 0.999$이다.

    부품 하나의 신뢰도가 0.9로 같은데 시스템 신뢰도는 0.729와 0.999로 크게 갈린다. 직렬 구조는 부품이 늘수록 나빠지고 병렬 구조는 좋아진다. 이것이 중요한 시스템에 여분을 두는 이유다.

    직렬 구조에서는 위험함수가 더해진다는 점도 눈여겨볼 만하다. $S_{\text{직렬}} = \prod S_i$의 로그를 취하면

    $$
    h_{\text{직렬}}(t) = \sum_{i=1}^3 h_i(t)
    $$

    이므로, 부품이 모두 지수분포를 따르면 시스템도 비율 $\sum\lambda_i$인 지수분포를 따른다. 반면 병렬 구조에서는 이런 단순한 관계가 성립하지 않는다.

---

## 정리하며

생존함수 $S(x)=P(X>x)=1-F(x)$ 는 **오른쪽 꼬리**의 확률이다.

- **왜 따로 두는가.** 꼬리로 갈수록 $F(x)$ 가 $1$ 에 붙으므로 `1 - cdf(x)` 는 **자리올림 소실**로 정밀도를 잃는다. $x=10$ 에서 `1 - norm.cdf(10)` 은 $0$ 을 주지만 `norm.sf(10)` 은 $7.6\times10^{-24}$ 를 제대로 준다. **작은 $p$ 값을 계산할 때는 반드시 `sf` 를 쓴다.**
- 대응되는 역함수가 `isf` 이며, 상위 $q$ 를 가르는 값을 준다.
- 신뢰성 공학·보험계리·생존분석에서 기본 언어다. 21장의 생존함수와 위험함수가 이 개념의 확장이다.
- 단측 검정의 $p$ 값이 곧 관측된 통계량에서의 생존함수 값이다.

다음 절 **정규분포 난수 생성**으로 이 분포의 네 가지 창을 마무리한다. 이론적 밀도와 실제 표본이 얼마나 맞아떨어지는지 눈으로 확인한다.
