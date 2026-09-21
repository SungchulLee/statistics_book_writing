# 와이불분포

## 개요

**와이불분포**는 수명, 고장 시간, 생존 자료를 다루는 유연한 모형이다. 형상 모수 $k$가 위험률이 증가하는지, 감소하는지, 일정한지를 결정한다:

$$
f(x; k, \lambda) = \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1} \exp\!\left(-\left(\frac{x}{\lambda}\right)^k\right), \qquad x \ge 0
$$

| $k$ | 위험률 거동 | 특수한 경우 |
|---|---|---|
| $k < 1$ | 감소하는 위험률 (초기 고장) | — |
| $k = 1$ | 일정한 위험률 | Exponential($\lambda$) |
| $k > 1$ | 증가하는 위험률 (노화/마모) | — |
| $k \approx 3.6$ | 근사적으로 정규분포 모양 | — |

---

## 생존함수와 위험함수

$$
S(x) = \exp\!\left(-\left(\frac{x}{\lambda}\right)^k\right), \qquad h(x) = \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1}
$$

위험함수 $h(x) = f(x)/S(x)$는 시각 $x$까지 생존했다는 조건 아래 그 시점의 순간 고장률을 준다.

---

## 코드

<div class="codebox" markdown>

### 예제 1. 형상모수에 따른 와이불 밀도와 위험함수 { .eg }

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

x = np.linspace(0.01, 3.0, 500)
lam = 1.0        # 척도모수. 시간의 단위를 정할 뿐 모양은 바꾸지 않는다.

# 형상모수 k가 이 분포의 성격을 전부 결정한다.
# 아래 세 번째 패널(위험함수)을 보면 그 뜻이 분명해진다.
#   k < 1 : 위험률이 시간에 따라 **감소** — 초기 불량. 살아남을수록 안전해진다.
#   k = 1 : 위험률이 **일정** — 지수분포와 같아진다. 무기억성.
#   k > 1 : 위험률이 시간에 따라 **증가** — 마모. 오래될수록 위험해진다.
params = [
    (0.5, "k=0.5 (decreasing hazard)"),
    (1.0, "k=1.0 (exponential)"),
    (1.5, "k=1.5 (increasing hazard)"),
    (3.0, "k=3.0 (near-normal shape)"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for k, desc in params:
    # 세 함수를 정의대로 직접 계산한다(scipy를 쓰지 않고).
    #   pdf : 밀도함수
    #   sf  : 생존함수 S(t) = P(T > t) = exp(-(t/lam)^k)
    #   haz : 위험함수 h(t) = pdf/sf.  "여기까지 살아남았다는 조건 하에
    #         지금 당장 고장 날 순간적인 비율"이다.
    # 바일불에서는 pdf/sf 를 계산하면 지수항이 정확히 약분되어
    # 아래처럼 아주 단순한 멱함수만 남는다. 이것이 이 분포가 널리 쓰이는 이유다.
    pdf = (k / lam) * (x / lam) ** (k - 1) * np.exp(-(x / lam) ** k)
    sf = np.exp(-(x / lam) ** k)
    haz = (k / lam) * (x / lam) ** (k - 1)

    axes[0].plot(x, pdf, lw=2, label=f'k = {k}')
    axes[1].plot(x, sf, lw=2, label=f'k = {k}')
    axes[2].plot(x, haz, lw=2, label=f'k = {k}')

axes[0].set_title('Weibull PDF')
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[1].set_title('Survival Function S(x)')
axes[1].set_xlabel('x')
axes[2].set_title('Hazard Function h(x)')
axes[2].set_xlabel('x')
axes[2].set_ylim(0, 5)

for ax in axes:
    ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
```

![Weibull PDF](./img/weibull_pdf_32.png)

</div>

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
$k = 1$인 와이불분포가 비율 $1/\lambda$인 지수분포로 환원됨을 보여라.

</div>

??? success "풀이"
    $k = 1$로 두면:

    $$
    f(x) = \frac{1}{\lambda}\exp\!\left(-\frac{x}{\lambda}\right)
    $$

    이는 평균이 $\lambda$인 $\text{Exponential}(\text{rate} = 1/\lambda)$의 PDF이다. 위험률은 $h(x) = 1/\lambda$로 상수가 되며, 이는 무기억성과 일관된다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
와이불분포의 CDF와 중앙값을 유도하라.

</div>

??? success "풀이"
    **CDF:**

    $$
    F(x) = 1 - S(x) = 1 - \exp\!\left(-\left(\frac{x}{\lambda}\right)^k\right)
    $$

    **중앙값:** $F(m) = 0.5$를 풀면:

    $$
    \exp\!\left(-\left(\frac{m}{\lambda}\right)^k\right) = 0.5 \implies \left(\frac{m}{\lambda}\right)^k = \ln 2 \implies m = \lambda(\ln 2)^{1/k}
    $$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
어떤 부품의 수명이 $k = 2$, $\lambda = 1000$시간인 와이불분포를 따른다. 500시간을 넘겨 생존할 확률은? 1500시간은?

</div>

??? success "풀이"
    $$
    S(500) = \exp\!\left(-\left(\frac{500}{1000}\right)^2\right) = e^{-0.25} \approx 0.779
    $$

    $$
    S(1500) = \exp\!\left(-\left(\frac{1500}{1000}\right)^2\right) = e^{-2.25} \approx 0.105
    $$

    이 부품이 500시간을 견딜 확률은 77.9%이지만 1500시간을 견딜 확률은 10.5%에 불과하다. 위험률이 증가하므로($k = 2 > 1$) 부품이 나이를 먹을수록 고장이 더 잘 난다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
더 유연한 분포들이 있는데도 와이불분포가 신뢰성 공학에서 널리 쓰이는 이유를 설명하라.

</div>

??? success "풀이"
    와이불분포가 널리 쓰이는 데에는 몇 가지 실용적인 장점이 있다:

    1. **해석 가능한 위험률:** 모수 $k$ 하나가 고장률이 증가할지, 감소할지, 일정할지를 직접 결정한다.
    2. **닫힌 형태의 함수:** PDF, CDF, 생존함수, 위험함수가 모두 간단한 닫힌 형태로 주어져 수치적분이 필요 없다.
    3. **선형화 가능:** $\ln(-\ln(S(x)))$를 $\ln(x)$에 대해 그리면 직선이 되어 그래프로 모수를 추정할 수 있다(와이불 확률지).
    4. **지수분포를 포함:** $k=1$로 두면 가장 단순한 수명 모형이 복원되므로 와이불은 그 자연스러운 일반화가 된다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
와이불분포의 평균과 분산이

$$
E[X] = \lambda\,\Gamma\!\left(1 + \frac1k\right), \qquad \operatorname{Var}(X) = \lambda^2\left\{\Gamma\!\left(1+\frac2k\right) - \Gamma\!\left(1+\frac1k\right)^2\right\}
$$

임을 보이고, 연습문제 3의 $k=2$, $\lambda = 1000$에 대해 값을 구해 중앙값과 견주어라.

</div>

??? success "풀이"
    치환 $u = (x/\lambda)^k$를 쓴다. 그러면 $x = \lambda u^{1/k}$, $du = \frac{k}{\lambda}(x/\lambda)^{k-1}dx$이므로 밀도의 앞부분이 그대로 $du$로 흡수되어

    $$
    E[X^r] = \int_0^\infty x^r f(x)\,dx = \int_0^\infty \lambda^r u^{r/k} e^{-u}\,du = \lambda^r\,\Gamma\!\left(1 + \frac{r}{k}\right)
    $$

    를 얻는다. 감마함수의 정의 $\Gamma(z) = \int_0^\infty u^{z-1}e^{-u}du$를 쓴 것이다. $r = 1$이 평균이고, $r = 2$를 넣어 $\operatorname{Var} = E[X^2] - (E[X])^2$을 계산하면 위 식이 나온다. $\square$

    **수치.** $k=2$이면 $\Gamma(1.5) = \sqrt\pi/2 \approx 0.8862$이고 $\Gamma(2) = 1$이므로

    $$
    E[X] = 1000 \times 0.8862 = 886.2, \qquad \operatorname{Var}(X) = 10^6(1 - 0.7854) = 214{,}602
    $$

    으로 표준편차가 463.3이다. 중앙값은 연습문제 2에서 $1000(\ln 2)^{1/2} = 832.6$이다.

    평균 886.2 > 중앙값 832.6으로 오른쪽으로 치우쳐 있다. $k$가 커질수록 이 치우침이 줄어들어 $k \approx 3.6$에서 거의 대칭이 된다. $k=2$인 경우를 따로 **레일리분포**라 부르며, 2차원 등방 정규벡터의 길이가 이 분포를 따른다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
와이불 확률지의 원리를 유도하라. 자료에서 $S(300) = 0.9$, $S(800) = 0.5$로 읽혔다면 $k$와 $\lambda$는 얼마인가?

</div>

??? success "풀이"
    생존함수의 로그를 두 번 취한다.

    $$
    S(x) = \exp\!\left(-(x/\lambda)^k\right) \implies -\ln S(x) = \left(\frac{x}{\lambda}\right)^k \implies \ln\{-\ln S(x)\} = k\ln x - k\ln\lambda
    $$

    즉 가로축을 $\ln x$, 세로축을 $\ln(-\ln S)$로 두면 **기울기가 $k$이고 세로축 절편이 $-k\ln\lambda$인 직선**이 된다. 자료가 와이불분포를 따르는지를 직선성으로 눈으로 확인할 수 있고, 직선을 맞춰 모수도 얻는다.

    주어진 두 점에서

    $$
    y_1 = \ln(-\ln 0.9) = -2.250, \quad y_2 = \ln(-\ln 0.5) = -0.3665
    $$

    이고 $\ln 300 = 5.704$, $\ln 800 = 6.685$이므로

    $$
    k = \frac{y_2 - y_1}{\ln 800 - \ln 300} = \frac{1.884}{0.9808} \approx 1.92
    $$

    이다. 절편에서

    $$
    \ln\lambda = \ln 300 - \frac{y_1}{k} = 5.704 + \frac{2.250}{1.92} = 6.875 \implies \lambda \approx 968
    $$

    을 얻는다. $k \approx 1.92 > 1$이므로 마모형 고장이다.

    이 방법은 컴퓨터가 없던 시절의 유물이 아니다. 최대가능도 추정의 초기값을 주고, 무엇보다 **모형이 맞는지를 눈으로 진단하는** 수단이다. 점들이 한 직선이 아니라 꺾인 두 직선을 이루면 고장 원인이 둘이라는 신호이며, 그때는 혼합 와이불이나 경쟁위험 모형을 생각해야 한다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
$X_1, \dots, X_n$이 독립이고 모두 $\text{Weibull}(k, \lambda)$를 따를 때 $\min_i X_i$의 분포를 구하라. 이 성질이 재료의 강도 모형에서 왜 중요한가?

</div>

??? success "풀이"
    최솟값이 $t$를 넘으려면 모두가 $t$를 넘어야 하므로

    $$
    P(\min_i X_i > t) = \prod_{i=1}^n S(t) = \exp\!\left(-n\left(\frac{t}{\lambda}\right)^k\right) = \exp\!\left(-\left(\frac{t}{\lambda n^{-1/k}}\right)^k\right)
    $$

    이다. 즉 $\min_i X_i \sim \text{Weibull}(k,\ \lambda n^{-1/k})$이다. **형상은 그대로이고 척도만 줄어든다.** 와이불족은 최솟값 연산에 대해 닫혀 있다.

    **약한 고리 이론.** 쇠사슬은 가장 약한 고리에서 끊어지고, 섬유 다발은 가장 약한 가닥에서 먼저 갈라지며, 취성 재료는 가장 큰 결함에서 파괴된다. 재료 전체의 강도가 수많은 미세 요소 강도의 **최솟값**이라면, 요소 강도의 분포가 무엇이든(왼쪽 끝이 0이고 거듭제곱 꼴로 시작하기만 하면) 최솟값의 극한분포는 와이불이 된다. 최댓값의 극한분포가 굼벨·프레셰·와이불 세 가지뿐이라는 극단값 이론의 최솟값 판이다.

    실무적 귀결이 뚜렷하다. 위 식에서 척도가 $n^{-1/k}$로 줄므로 **시편이 클수록 평균 강도가 낮게 측정된다.** 작은 시편으로 잰 강도를 큰 구조물에 그대로 적용하면 위험하며, $k$가 작을수록(산포가 클수록) 이 크기 효과가 심하다. 세라믹이나 유리섬유 설계에서 반드시 고려하는 사항이다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
SciPy에서 $k = 2$, $\lambda = 1000$인 와이불분포를 만들고 $S(500)$과 $S(1500)$을 계산하는 코드를 적어라. `weibull_min`과 `weibull_max`, `exponweib`의 차이는 무엇인가?

</div>

??? success "풀이"
    형상 모수는 위치·척도가 아닌 별도 인자이므로 첫 번째 자리에 `c`로 들어간다.

    ```python
    from scipy import stats

    rv = stats.weibull_min(c=2, scale=1000)     # loc=0 이 기본값
    print(rv.sf(500), rv.sf(1500))              # 0.7788  0.10540
    print(rv.mean(), rv.std())                  # 886.23  463.25
    ```

    연습문제 3의 손계산 $e^{-0.25} = 0.779$, $e^{-2.25} = 0.105$와 일치한다.

    **세 이름의 차이.**

    - `weibull_min` — 이 문서가 다루는 보통의 와이불분포다. 지지집합이 $[0,\infty)$이고 최솟값의 극한분포에 해당한다. 수명 자료에는 언제나 이것을 쓴다.
    - `weibull_max` — 이를 좌우로 뒤집어 지지집합이 $(-\infty, 0]$인 분포다. 최댓값의 극단값 분포 가운데 위가 막힌 경우를 다룰 때 쓴다. 수명에 쓰면 음수 시간이 나온다.
    - `exponweib` — 형상 모수를 하나 더 붙인 지수화 와이불로, $F(x) = \{1 - e^{-(x/\lambda)^k}\}^a$ 꼴이다. $a = 1$이면 보통의 와이불이 된다.

    `loc`을 쓰면 3-모수 와이불이 된다. "최소 보증 수명"이 있는 부품을 모형화할 때 쓰지만, `loc`까지 자료에서 추정하면 가능도가 발산할 수 있어 주의가 필요하다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
$X \sim \text{Weibull}(k, \lambda)$와 $(X/\lambda)^k \sim \text{Exp}(1)$이 동치임을 보여라. 이를 이용해 (가) 난수 생성법, (나) 연습문제 5의 적률 공식을 다시 유도하라.

</div>

??? success "풀이"
    **동치.** $Y = (X/\lambda)^k$로 두면 $y > 0$에 대해

    $$
    P(Y > y) = P\!\left(X > \lambda y^{1/k}\right) = \exp\!\left(-\left(\frac{\lambda y^{1/k}}{\lambda}\right)^k\right) = e^{-y}
    $$

    이므로 $Y \sim \text{Exp}(1)$이다. 거꾸로 $Y \sim \text{Exp}(1)$이면 $X = \lambda Y^{1/k}$의 생존함수가 $\exp(-(x/\lambda)^k)$가 되어 와이불이다. $\square$

    이 한 줄이 와이불분포의 정체를 말해 준다. **와이불은 지수분포를 시간축에서 거듭제곱으로 늘이거나 줄인 것**이다. $k>1$이면 시간이 갈수록 압축되어 위험이 커지고, $k<1$이면 늘어나 위험이 작아진다.

    **(가) 난수 생성.** $U \sim \text{Uniform}(0,1)$일 때 $-\ln U \sim \text{Exp}(1)$이므로

    $$
    X = \lambda\left(-\ln U\right)^{1/k}
    $$

    가 와이불 난수를 준다. 로그 한 번과 거듭제곱 한 번이면 끝난다. 이것이 와이불분포의 역변환 공식 $F^{-1}(u) = \lambda\{-\ln(1-u)\}^{1/k}$과 같은 식임을 확인하라($U$와 $1-U$가 같은 분포이므로).

    **(나) 적률.** $X = \lambda Y^{1/k}$이므로

    $$
    E[X^r] = \lambda^r E\!\left[Y^{r/k}\right] = \lambda^r \int_0^\infty y^{r/k}e^{-y}dy = \lambda^r\,\Gamma\!\left(1+\frac rk\right)
    $$

    이다. 연습문제 5에서 한 치환이 사실은 이 변환을 계산 안에서 되풀이한 것이었다. 관계를 먼저 알아 두면 적률·분위수·난수 생성이 모두 표준지수분포 하나의 계산으로 환원된다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
신뢰성 공학의 "욕조곡선"은 위험함수가 처음에 감소하다가 평평해진 뒤 다시 증가하는 모양이다. 와이불분포 하나로 이 모양을 낼 수 있는가? 낼 수 없다면 어떻게 모형화하겠는가?

</div>

??? success "풀이"
    **낼 수 없다.** 와이불의 위험함수 $h(x) = (k/\lambda)(x/\lambda)^{k-1}$은 거듭제곱함수이므로 $k$의 값에 따라 단조감소($k<1$), 상수($k=1$), 단조증가($k>1$) 셋 중 하나일 뿐이다. 방향을 도중에 바꿀 수 없다.

    **대안 1 — 경쟁위험.** 고장 원인이 여럿이고 각각 독립이라면 위험함수가 더해진다. 초기 불량($k_1 < 1$), 우발 고장($k_2 = 1$), 마모($k_3 > 1$)를 각각 와이불로 두고

    $$
    h(x) = h_1(x) + h_2(x) + h_3(x)
    $$

    로 합치면 욕조 모양이 자연스럽게 나온다. 처음에는 감소하는 $h_1$이 지배하고, 중간에는 상수인 $h_2$가, 나중에는 커지는 $h_3$이 지배한다. 전체 생존함수는 $S(x) = S_1S_2S_3$이다. 물리적 해석이 분명해 가장 널리 쓰이는 방법이다.

    **대안 2 — 혼합 와이불.** 모집단이 "불량품 소수"와 "정상품 다수"로 나뉜다고 보고 두 와이불의 혼합으로 둔다. 이쪽은 개체마다 어느 부류인지가 다른 이질성 모형이라는 점에서 경쟁위험과 다르다.

    **대안 3 — 더 유연한 분포.** 위험함수가 욕조 모양을 낼 수 있는 분포를 직접 쓴다. 일반화 감마분포나 지수화 와이불이 그렇다. 다만 모수가 늘어 추정이 불안정해지기 쉽다.

    실무에서는 아예 구간을 나누어 다루는 경우도 많다. 제조 단계의 번인(burn-in)으로 초기 불량 구간을 미리 태워 없애고, 예방 정비로 마모 구간에 들어가기 전에 교체하면, 남는 사용 구간에서는 위험률이 거의 일정해져 지수분포 모형으로 충분해진다.

---

## 정리하며

와이불분포의 핵심은 형상 모수 $k$ 하나가 **위험률의 방향**을 정한다는 데 있다.

$$
S(x) = \exp\!\left(-\left(\frac{x}{\lambda}\right)^k\right), \qquad h(x) = \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1}
$$

| $k$ | 위험률 | 뜻 |
|---|---|---|
| $k<1$ | 감소 | 초기 고장 — 오래 버틸수록 안전해진다 |
| $k=1$ | 일정 | 지수분포, **무기억성** |
| $k>1$ | 증가 | 노화·마모 — 오래 쓸수록 위험해진다 |

- **$k=1$ 에서 지수분포가 되고, 그때에만 무기억성이 성립한다.** 앞 절의 지수분포가 "노화하지 않는 대상"에만 맞는다고 한 이유가 여기서 분명해진다.
- **위험함수 $h(x)=f(x)/S(x)$ 는 "지금까지 살아남았다는 조건 아래의 순간 고장률"** 이다. 밀도가 아니라 이 함수를 보아야 대상이 노화하는지가 드러난다.
- $k\approx3.6$ 이면 모양이 정규분포에 가까워진다. 하나의 형상 모수로 이만큼 다양한 모양을 낸다는 점이 이 분포가 신뢰성 공학의 기본 도구가 된 이유다.
- 21장의 생존분석이 이 위험함수 개념을 그대로 확장한다.

다음 절 **역변환 표본추출**로 4.2절을 마무리한다. 지금까지 본 분포들에서 실제로 난수를 뽑는 일반적인 방법이다.
