# 이표본 비율 검정

## 개요

이표본 비율 검정은 독립인 두 모집단의 비율을 비교한다. A/B 검정, 임상시험, 사회과학 연구에서 처리나 개입이 이진 결과의 비율을 바꾸는지 판단하는 데 널리 쓰인다. 이항분포에 대한 정규근사에 기반한 z-통계량을 쓴다.

## 검정의 구성

**가설:**

- 양측: $H_0\colon p_1 - p_2 = \delta_0$ 대 $H_1\colon p_1 - p_2 \neq \delta_0$

$\delta_0 = 0$(동일성 검정)일 때는 **합동** 표준오차를 쓴다:

$$
\hat{p}_{\text{pool}} = \frac{k_1 + k_2}{n_1 + n_2}, \qquad SE = \sqrt{\hat{p}_{\text{pool}}(1-\hat{p}_{\text{pool}})\left(\frac{1}{n_1}+\frac{1}{n_2}\right)}.
$$

$\delta_0 \neq 0$일 때는 **Wald** 표준오차를 쓴다:

$$
SE = \sqrt{\frac{\hat{p}_1(1-\hat{p}_1)}{n_1} + \frac{\hat{p}_2(1-\hat{p}_2)}{n_2}}.
$$

검정통계량은

$$
Z = \frac{(\hat{p}_1 - \hat{p}_2) - \delta_0}{SE} \;\dot\sim\; N(0,1).
$$

## 코드

```python
import math
from scipy.stats import norm

def test_diff_two_props(k1, n1, k2, n2, delta0=0.0,
                        method="pooled", alt="two-sided", alpha=0.05):
    """H0: p1 - p2 = delta0.

    delta0=0이고 method='pooled'이면 합동 표준오차를 쓴다.
    돌려주는 값은 (z, p, 기각 여부, 이름).
    """
    p1, p2 = k1 / n1, k2 / n2
    d_hat = p1 - p2
    if delta0 == 0.0 and method == "pooled":
        # H0가 "두 비율이 같다"이면 그 공통값의 최선의 추정은 전체를 합친 비율이다.
        # 검정은 H0 아래의 분포를 쓰므로 여기서 합동하는 것이 옳다.
        p_pool = (k1 + k2) / (n1 + n2)
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        label = "pooled z-test"
    else:
        # delta0이 0이 아니면 "두 비율이 같다"는 가정이 없으므로 합동할 수 없다.
        # 신뢰구간에서 쓰는 것과 같은 표준오차다.
        se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        label = "wald z-test"

    z = (d_hat - delta0) / se
    if alt == "two-sided":
        p = 2 * min(norm.cdf(z), 1 - norm.cdf(z))
    elif alt == "less":
        p = norm.cdf(z)
    else:
        p = 1 - norm.cdf(z)
    return z, p, (p < alpha), label
```

**보기 1.**

```python
z, p, reject, label = test_diff_two_props(
    k1=30, n1=80, k2=18, n2=60, delta0=0.0, method="pooled"
)
print(label, "z:", z, "p:", p, "reject:", reject)

# 같은 자료를 합동하지 않은 Wald 표준오차로 계산하면 어떻게 다른가?
z2, p2, reject2, label2 = test_diff_two_props(
    k1=30, n1=80, k2=18, n2=60, delta0=0.0, method="wald"
)
print(label2, "z:", z2, "p:", p2, "reject:", reject2)
```

출력:

```
pooled z-test z: 0.9251909321159419 p: 0.3548665994806586 reject: False
wald z-test z: 0.9353331581027243 p: 0.3496166322568355 reject: False
```

$\hat p_1 = 0.375$, $\hat p_2 = 0.30$으로 7.5%p 차이지만 표본이 80명과 60명뿐이라 기각하지 못한다.

두 방식의 z가 0.925와 0.935로 비슷하다. 두 비율이 0.375와 0.30으로 그리 멀지 않아 합동비율 0.343이 양쪽 모두를 그런대로 대신하기 때문이다. 두 비율이 크게 다르면 이 차이가 커진다. 신뢰구간을 만들 때는 합동하지 않은 쪽을 써야 하며, 그래서 "검정은 기각하지 않았는데 신뢰구간은 0을 아슬아슬하게 벗어난다" 같은 어긋남이 생길 수 있다.

### 해석

$\hat{p}_1 = 30/80 = 0.375$, $\hat{p}_2 = 18/60 = 0.300$으로 $H_0\colon p_1 = p_2$를 검정한다. 합동 비율은 $\hat{p}_{\text{pool}} = 48/140 \approx 0.343$이다. 검정통계량은

$$
Z = \frac{0.375 - 0.300}{\sqrt{0.343 \times 0.657 \times (1/80 + 1/60)}} = \frac{0.075}{\sqrt{0.343 \times 0.657 \times 0.0292}} \approx \frac{0.075}{0.0810} \approx 0.926.
$$

양측 p-값이 약 0.35이므로 $H_0$을 기각하지 못한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** 어떤 임상시험에서 약을 받은 환자 200명 중 45명이, 위약을 받은 200명 중 30명이 회복했다. $\alpha = 0.05$에서 $H_0\colon p_1 = p_2$ 대 $H_1\colon p_1 > p_2$를 검정하라.

</div>

??? success "풀이"

    $\hat{p}_1 = 0.225$, $\hat{p}_2 = 0.150$, $\hat{p}_{\text{pool}} = 75/400 = 0.1875$.

    $$
    SE = \sqrt{0.1875 \times 0.8125 \times (1/200 + 1/200)} = \sqrt{0.1875 \times 0.8125 \times 0.01} = \sqrt{0.001523} \approx 0.03903.
    $$

    $$
    Z = \frac{0.225 - 0.150}{0.03903} = \frac{0.075}{0.03903} \approx 1.921.
    $$

    단측 p-값은 $P(Z \geq 1.921) \approx 0.0274$이다. $0.0274 < 0.05$이므로 $H_0$을 기각한다. 이 약의 회복률이 유의하게 높다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** 합동 표준오차와 Wald 표준오차를 각각 언제 쓰는지 설명하라.

</div>

??? success "풀이"

    **합동 표준오차**는 $H_0$이 $p_1 = p_2$를 지정할 때(즉 $\delta_0 = 0$일 때) 쓴다. 이 귀무가설 아래에서 공통 비율의 최선의 추정값은 합동 비율 $\hat{p}_{\text{pool}}$이며, 이를 표준오차에 쓰면 더 정확한 검정이 된다.

    **Wald 표준오차**는 $\delta_0 \neq 0$일 때(차이가 0이 아닌 어떤 값인지 검정할 때) 쓴다. 이 경우 추정할 공통 비율이 없으므로 각 집단의 비율을 따로 쓴다. 귀무가설의 값과 무관하게 $p_1 - p_2$의 신뢰구간을 만들 때에도 Wald 표준오차를 쓴다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** 어떤 웹사이트 A/B 검정에서 변형 A는 방문자 5000명 중 120명이, 변형 B는 5000명 중 95명이 전환했다. $\alpha = 0.05$에서 차이를 검정하고 $p_A - p_B$의 95% 신뢰구간을 계산하라.

</div>

??? success "풀이"

    $\hat{p}_A = 0.024$, $\hat{p}_B = 0.019$, $\hat{p}_{\text{pool}} = 215/10000 = 0.0215$.

    **검정:**

    $$
    SE = \sqrt{0.0215 \times 0.9785 \times 2/5000} = \sqrt{0.0215 \times 0.9785 \times 0.0004} \approx 0.002898.
    $$

    $$
    Z = \frac{0.024 - 0.019}{0.002898} \approx 1.725.
    $$

    양측 p-값: $2 \times P(Z \geq 1.725) \approx 0.0845$. $\alpha = 0.05$에서 기각하지 못한다.

    **95% 신뢰구간** (Wald 표준오차 사용):

    $$
    SE_{\text{Wald}} = \sqrt{\frac{0.024 \times 0.976}{5000} + \frac{0.019 \times 0.981}{5000}} \approx \sqrt{0.000004685 + 0.000003728} \approx 0.002901.
    $$

    $$
    0.005 \pm 1.96 \times 0.002901 = 0.005 \pm 0.00569 = (-0.00069,\; 0.01069).
    $$

    신뢰구간이 0을 포함하여 검정 결과와 일관된다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** $H_0\colon p_1 = p_2$에 대한 가능도비 검정에서 합동 z-검정통계량을 유도하라.

</div>

??? success "풀이"

    $H_0\colon p_1 = p_2 = p$ 아래에서 MLE는 $\hat{p} = (k_1+k_2)/(n_1+n_2)$이다. 제약 없는 모형에서 MLE는 $\hat{p}_1 = k_1/n_1$과 $\hat{p}_2 = k_2/n_2$이다. 로그가능도비는

    $$
    \Lambda = 2\left[\ell(\hat{p}_1, \hat{p}_2) - \ell(\hat{p}, \hat{p})\right].
    $$

    점근이론에 의해 $H_0$ 아래에서 $\Lambda \xrightarrow{d} \chi^2_1$이다. (1차까지 동등한) Rao 점수검정은 다음 검정통계량을 준다:

    $$
    Z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})(1/n_1 + 1/n_2)}},
    $$

    그리고 $Z^2 \approx \Lambda$이다. 이것이 바로 합동 z-통계량이며, 합동 검정이 점수검정/가능도비 검정의 틀에서 자연스럽게 나옴을 확인해 준다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** 이항분포에 대한 정규근사가 $n_1\hat{p}_{\text{pool}} \geq 5$이고 $n_1(1-\hat{p}_{\text{pool}}) \geq 5$($n_2$에 대해서도 마찬가지)를 요구함을 보여라. 이 조건이 깨지면 어떤 대안을 쓸 수 있는가?

</div>

??? success "풀이"

    z-검정은 중심극한정리 근사 $\hat{p}_i \approx N(p_i, p_i(1-p_i)/n_i)$에 기댄다. $p$가 0이나 1에 가깝거나(이항분포가 심하게 치우친다) $n$이 작으면 이 근사가 나쁘다. 경험칙 $np \geq 5$, $n(1-p) \geq 5$는 이항분포가 정규근사를 쓸 만큼 충분히 대칭이 되도록 보장한다. 합동 검정에서는 $\hat{p}_{\text{pool}}$로 확인한다.

    이 조건이 깨지면 다음 대안이 있다:

    - **Fisher의 정확검정**: $H_0$ 아래 초기하분포로 정확한 p-값을 계산한다. 점근근사가 필요 없다.
    - **Barnard의 정확검정**: 조건을 두지 않는 정확검정으로 Fisher 검정보다 검정력이 클 수 있다.
    - **베이즈 방법**: Beta-Binomial 모형과 사후추론을 쓴다. $\square$

---

## 정리하며

비율 차 검정에서 **$\delta_0$ 이 $0$ 인지가 표준오차를 정한다.**

- **$\delta_0=0$ 이면 합동 표준오차를 쓴다.** $H_0$ 아래에서 두 비율이 같으므로 공통 $p$ 를 합쳐 추정하는 것이 옳다.
- **$\delta_0\ne0$ 이면 합동할 수 없다.** 두 비율이 다르다고 가정한 상태이므로 각자의 $\hat p_i$ 로 표준오차를 만든다. 비열등성 검정이 그런 경우다.
- **`statsmodels.stats.proportion.proportions_ztest` 가 합동 버전**을 제공하며, 손으로 구현할 때 이와 대조해 검산한다.
- **표본이 작으면 피셔의 정확검정으로 간다.** 기대도수가 작을 때 정규근사가 무너지며, 10장에서 다룬다.
- **A/B 검정 실무에서는 조기 중단이 더 큰 위험이다.** 유의해질 때까지 들여다보면 제1종 오류율이 명목값을 훨씬 넘어서며, 이 장 뒤의 $p$-해킹 절이 그 문제를 다룬다.

다음 절 **이표본 $t$ 검정 (대마 가격)** 에서 실제 자료로 전체 흐름을 밟는다.
