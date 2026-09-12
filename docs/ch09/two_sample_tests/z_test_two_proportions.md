# 비율 차이에 대한 이표본 Z-검정

## 집단 간 성공률의 비교

실무의 많은 질문이 두 모집단의 "성공" 비율을 비교하는 것이다. 새 웹사이트 디자인이 기존보다 높은 전환율로 이어지는가? 백신 접종군의 감염률이 위약군보다 낮은가? 두 인구집단의 지지율이 다른가? 비율에 대한 이표본 z-검정은 두 모비율 $p_1$과 $p_2$가 같은지 검정하여 이런 질문에 답하는 형식적 틀을 제공한다.

## 설정과 기호

이진(성공/실패) 결과의 독립인 두 확률표본을 관측한다:

- 표본 1: 독립 시행 $n_1$번에서 성공 $X_1$번, 표본비율 $\hat{p}_1 = X_1 / n_1$
- 표본 2: 독립 시행 $n_2$번에서 성공 $X_2$번, 표본비율 $\hat{p}_2 = X_2 / n_2$

여기서 $X_1 \sim \text{Binomial}(n_1, p_1)$, $X_2 \sim \text{Binomial}(n_2, p_2)$이고 두 표본은 서로 독립이다.

## 가설

귀무가설은 두 모비율이 같다는 것이다:

$$
H_0 : p_1 = p_2
$$

대립가설은 다음 세 형태 중 하나이다:

| 이름 | 대립가설 $H_a$ | 기각하는 경우 |
|---|---|---|
| 양측 | $p_1 \neq p_2$ | $\lvert Z \rvert$가 클 때 |
| 좌측 | $p_1 < p_2$ | $Z$가 매우 작을 때(음수) |
| 우측 | $p_1 > p_2$ | $Z$가 매우 클 때 |

??? note "0이 아닌 차이를 검정할 때"
    여기서 제시한 검정은 $H_0: p_1 - p_2 = 0$에만 적용된다. $\delta_0 \neq 0$인 $H_0: p_1 - p_2 = \delta_0$을 검정하려면 (합동하지 않는) 다른 표준오차 공식이 필요하다. 합동 비율은 귀무가설 아래에서 $p_1 = p_2$를 가정할 때에만 의미가 있기 때문이다.

## 합동 비율

$H_0: p_1 = p_2$ 아래에서 두 표본은 성공확률이 같은 모집단에서 나온다. 이 공통 비율의 최선의 추정값은 두 표본의 자료를 합친 것이다:

$$
\hat{p} = \frac{X_1 + X_2}{n_1 + n_2}
$$

이 합동 비율 $\hat{p}$는 성공 횟수와 시행 횟수를 모두 합쳐 하나의 추정값을 만든다. 귀무가설 아래에서 공통 표준오차를 추정하기 위해 검정통계량의 분모에 쓰인다.

## 검정통계량의 유도

$H_0$ 아래에서 차이 $\hat{p}_1 - \hat{p}_2$의 기댓값은 0이다. 두 표본이 독립이므로 차이의 분산은:

$$
\text{Var}(\hat{p}_1 - \hat{p}_2) = p(1 - p)\left(\frac{1}{n_1} + \frac{1}{n_2}\right)
$$

여기서 $p = p_1 = p_2$는 $H_0$ 아래의 공통 모비율이다. 미지의 $p$를 합동 추정값 $\hat{p}$로 바꾸고 표준화하면 검정통계량을 얻는다:

$$
Z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1 - \hat{p})\left(\dfrac{1}{n_1} + \dfrac{1}{n_2}\right)}}
$$

$H_0$ 아래에서 표본이 충분히 크면 $Z$는 근사적으로 표준정규이다: $Z \dot{\sim} N(0, 1)$.

## 기각역과 p-값

$z_{\alpha}$를 표준정규분포의 상위 $\alpha$ 분위수, $z_{\text{obs}}$를 검정통계량의 관측값이라 하자.

### 양측검정 (Hₐ: p₁ ≠ p₂)

**기각역**: $|z_{\text{obs}}| > z_{\alpha/2}$이면 $H_0$을 기각한다.

**p-값**:

$$
p\text{-value} = 2\bigl[1 - \mathcal{N}(|z_{\text{obs}}|)\bigr]
$$

### 좌측검정 (Hₐ: p₁ < p₂)

**기각역**: $z_{\text{obs}} < -z_{\alpha}$이면 $H_0$을 기각한다.

**p-값**:

$$
p\text{-value} = \mathcal{N}(z_{\text{obs}})
$$

### 우측검정 (Hₐ: p₁ > p₂)

**기각역**: $z_{\text{obs}} > z_{\alpha}$이면 $H_0$을 기각한다.

**p-값**:

$$
p\text{-value} = 1 - \mathcal{N}(z_{\text{obs}})
$$

## 가정

비율에 대한 이표본 z-검정에는 다음 조건이 필요하다:

1. **독립인 표본**: 두 표본이 각 모집단에서 독립적으로 뽑혔다.
2. **독립인 관측값**: 각 표본 안에서 이진 결과가 독립이다.
3. **충분히 큰 표본크기**: 이항분포에 대한 정규근사가 적절해야 한다. 표준적인 경험칙은 다음 네 가지를 모두 요구한다:

$$
n_1 \hat{p} \geq 5, \quad n_1(1 - \hat{p}) \geq 5, \quad n_2 \hat{p} \geq 5, \quad n_2(1 - \hat{p}) \geq 5
$$

어떤 교재는 이 조건을 각 표본비율($n_1\hat{p}_1 \geq 5$ 등)로 진술하지만, $\hat{p}$가 $H_0$ 아래에서 쓰는 추정값이므로 합동 비율 $\hat{p}$로 확인하는 것이 더 적절하다.

!!! warning "작은 표본이나 극단적인 비율"
    표본이 작거나 비율이 0 또는 1에 가까우면 정규근사가 나쁘다. 이런 경우에는 Fisher의 정확검정이나 순열검정이 더 믿을 만한 대안이다.

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> 전환율에 대한 A/B 검정. 어떤 전자상거래 회사가 기존 결제 페이지(A)와 새로 디자인한 버전(B)의 전환율을 비교하는 A/B 검정을 한 주 동안 진행했다:

- 페이지 A: 방문자 $n_1 = 500$명, 전환 $X_1 = 45$건, $\hat{p}_1 = 45/500 = 0.090$
- 페이지 B: 방문자 $n_2 = 480$명, 전환 $X_2 = 58$건, $\hat{p}_2 = 58/480 = 0.121$

$\alpha = 0.05$에서 전환율이 다른지 검정하라.

**1단계: 가설을 세운다.**

$$
H_0: p_1 = p_2 \quad \text{vs} \quad H_a: p_1 \neq p_2
$$

**2단계: 합동 비율을 계산한다.**

$$
\hat{p} = \frac{45 + 58}{500 + 480} = \frac{103}{980} \approx 0.1051
$$

**3단계: 표본크기 조건을 확인한다.** 가장 작은 기대도수는 $n_2(1 - \hat{p}) = 480 \times 0.8949 \approx 430 \geq 5$이다. 네 조건이 모두 만족된다.

**4단계: 표준오차와 검정통계량을 계산한다.**

$$
\text{SE} = \sqrt{0.1051 \times 0.8949 \times \left(\frac{1}{500} + \frac{1}{480}\right)} = \sqrt{0.09406 \times 0.004083} \approx \sqrt{0.000384} \approx 0.01960
$$

$$
z_{\text{obs}} = \frac{0.090 - 0.121}{0.01960} = \frac{-0.031}{0.01960} \approx -1.582
$$

**5단계: p-값을 계산한다.**

$$
p\text{-value} = 2\bigl[1 - \mathcal{N}(1.582)\bigr] = 2(1 - 0.9431) = 2(0.0569) \approx 0.114
$$

**6단계: 판정한다.** $p \approx 0.114 > 0.05 = \alpha$이므로 $H_0$을 기각하지 못한다. 5% 유의수준에서 두 페이지 디자인의 전환율이 다르다고 결론지을 증거가 부족하다.

</div>

## Python 구현

```python
import numpy as np
from scipy import stats

def two_proportion_z_test(x1, n1, x2, n2, alternative="two-sided"):
    """Two-sample z-test for the difference of proportions.

    Parameters
    ----------
    x1, x2 : int
        Number of successes in each sample.
    n1, n2 : int
        Sample sizes.
    alternative : str
        'two-sided', 'less', or 'greater'.

    Returns
    -------
    z_stat : float
        Test statistic.
    p_value : float
        P-value.
    """
    p1_hat = x1 / n1
    p2_hat = x2 / n2
    # H0가 "두 비율이 같다"이므로 그 공통값을 전체를 합쳐 추정한다.
    # 신뢰구간을 만들 때는 합동하지 않는다. 목적이 다르기 때문이다.
    p_hat = (x1 + x2) / (n1 + n2)

    se = np.sqrt(p_hat * (1 - p_hat) * (1 / n1 + 1 / n2))
    z_stat = (p1_hat - p2_hat) / se

    if alternative == "two-sided":
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    elif alternative == "less":
        p_value = stats.norm.cdf(z_stat)
    elif alternative == "greater":
        p_value = 1 - stats.norm.cdf(z_stat)
    else:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    return z_stat, p_value


# 예제: A/B 검정
z, p = two_proportion_z_test(45, 500, 58, 480, alternative="two-sided")
print(f"z = {z:.3f}, p-value = {p:.3f}")

# 같은 전환율 차이를 표본만 열 배로 늘려 다시 검정한다.
z10, p10 = two_proportion_z_test(450, 5000, 580, 4800, alternative="two-sided")
print(f"z = {z10:.3f}, p-value = {p10:.5f}")
```

출력:

```
z = -1.573, p-value = 0.116
z = -4.975, p-value = 0.00000
```

전환율 9.0%와 12.1%로 3.1%p 차이인데, 방문자 1,000명 남짓으로는 기각하지 못한다. 같은 차이를 10,000명으로 보면 $p$가 $10^{-6}$ 수준까지 떨어진다.

A/B 검정에서 표본크기 계획이 왜 중요한지 보여주는 예다. 실험을 너무 일찍 멈추면 실재하는 3%p 개선을 "차이 없음"으로 결론짓게 된다.

## 신뢰구간과의 관계

[가설검정과 신뢰구간의 쌍대성](../errors_and_power/duality.md)에 의해, 수준 $\alpha$에서 $H_0: p_1 = p_2$를 기각하지 못하는 것은 $p_1 - p_2$의 $(1 - \alpha)$ 신뢰구간이 0을 포함하는 것과 동등하다. 다만 신뢰구간은 보통 **합동하지 않은** 표준오차를 쓴다:

$$
\text{SE}_{\text{CI}} = \sqrt{\frac{\hat{p}_1(1 - \hat{p}_1)}{n_1} + \frac{\hat{p}_2(1 - \hat{p}_2)}{n_2}}
$$

신뢰구간은 $p_1 = p_2$를 가정하지 않기 때문이다. 합동 표준오차는 $H_0$ 아래의 가설검정에만 해당한다.

## 관련 주제

- [평균에 대한 이표본 Z-검정](z_test_two_means.md): 분산을 알 때의 평균 비교
- [이표본 t-검정](t_test_two_means.md): 분산을 모를 때의 평균 비교
- [두 분산에 대한 F-검정](f_test_two_variances.md): 모분산의 동일성 검정

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
치료 A: 200명 중 60명 금연. B: 180명 중 54명 금연. $\alpha = 0.05$에서 검정하라.

</div>

??? success "풀이"
    $\hat p_1 = 0.30$, $\hat p_2 = 0.30$. 합동: $\hat p = (60+54)/(200+180) = 114/380 = 0.30$.

    $z = (0.30 - 0.30)/\mathrm{SE} = 0$. 기각하지 못한다. 차이가 탐지되지 않았다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
A/B 검정: A는 2000명 중 240명, B는 2000명 중 270명이 전환했다. $\alpha = 0.05$에서 $H_0: p_A = p_B$를 검정하라.

</div>

??? success "풀이"
    $\hat p_A = 0.120$, $\hat p_B = 0.135$. 합동: $\hat p = 510/4000 = 0.1275$.

    $\mathrm{SE} = \sqrt{0.1275 \cdot 0.8725 \cdot (1/2000 + 1/2000)} = \sqrt{0.0001113} \approx 0.01055$.

    $z = (0.135 - 0.120)/0.01055 \approx 1.42$. 양측 p-값 $\approx 0.156$. 기각하지 못한다.

    상대적으로 12.5%의 상승인데도 이 표본크기에서는 통계적으로 유의하지 않다. 표본이 더 커야 한다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
**A/B 검정을 위한 표본크기.** $\alpha = 0.05$에서 12%에서 13.5%로의 상승을 검정력 80%로 탐지하려면 집단당 $n$이 얼마여야 하는가?

</div>

??? success "풀이"
    효과크기: $|p_1 - p_2| = 0.015$. 평균 분산: $\approx 0.1275 \cdot 0.8725 \approx 0.1112$.

    공식: $n \approx 2 \cdot ((z_{\alpha/2} + z_\beta)^2 \cdot p(1-p))/(\Delta)^2 = 2 \cdot ((1.96 + 0.84)^2 \cdot 0.1112)/0.000225 \approx 7750$.

    집단당 대략 8000명이다. 작은 효과에 대한 A/B 검정에는 큰 표본이 필요하며, 기술 업계에서 흔한 일이다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
**합동 표준오차와 비합동 표준오차.** 가설검정에는 합동, 신뢰구간에는 비합동 표준오차를 쓰는 이유는?

</div>

??? success "풀이"
    $H_0: p_1 = p_2 = p$ 아래에서는 합동 추정량 $\hat p_{\text{pool}}$이 공통 $p$의 최선의 추정값이다. 검정의 표준오차에는 이것을 쓴다.

    (동일성을 가정하지 않는) $p_1 - p_2$의 신뢰구간에는 $\hat p_1$과 $\hat p_2$를 따로 쓴다.

    현대의 A/B 검정 플랫폼은 검정에도 비합동 표준오차를 쓰기도 한다 — 보수적인 선택으로, 검정력이 약간 낮지만 비율이 다를 때에도 타당하다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
두 비율 $z$-검정의 **조건.**

</div>

??? success "풀이"
    - 각 모집단에서 독립적으로 뽑은 확률표본.
    - 충분히 큰 표본크기: $n_1 \hat p_1 \ge 10$, $n_1(1 - \hat p_1) \ge 10$, 표본 2도 마찬가지.
    - 합동 버전에서는 $n \hat p_{\text{pool}}$과 $n (1 - \hat p_{\text{pool}})$이 모두 $\ge 10$.

    조건이 깨지면 Fisher의 정확검정을 쓴다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
**두 비율의 효과크기.** 상대위험도와 오즈비를 정의하라.

</div>

??? success "풀이"
    **위험 차이:** $p_1 - p_2$. 절대적인 값이다.

    **상대위험도(RR):** $p_1/p_2$. 비율이다.

    **오즈비(OR):** $[p_1/(1-p_1)]/[p_2/(1-p_2)]$. 환자-대조군 연구와 로지스틱 회귀에서 쓴다.

    드문 사건($p$가 작을 때)에서는 $\mathrm{OR} \approx \mathrm{RR}$이다. 흔한 사건에서는 다르다.

    보고 방식: 위험 차이(임상적으로 해석 가능), 상대위험도(효과의 크기), 오즈비(통계적 관례)를 함께 제시하라. 각각 쓰임새가 있다.

---

## 정리하며

두 비율 비교에서 핵심은 **합동비율**이다.

$$
z=\frac{\hat p_1-\hat p_2}{\sqrt{\hat p(1-\hat p)\left(\frac1{n_1}+\frac1{n_2}\right)}},
\qquad \hat p=\frac{X_1+X_2}{n_1+n_2}
$$

- **$H_0:p_1=p_2$ 아래에서는 공통의 $p$ 가 하나뿐이므로** 두 표본을 합쳐 추정하는 것이 옳다. 그것이 합동비율이다.
- **신뢰구간에서는 합동하지 않는다.** 거기서는 $p_1=p_2$ 를 가정하지 않으므로 각자의 $\hat p_i$ 를 쓴다. **같은 자료에서 검정과 구간이 미세하게 다른 결론을 낼 수 있는 이유다.**
- **타당성 조건을 양쪽 모두에서 확인한다.** 각 집단에서 성공과 실패가 충분히 있어야 한다.
- **A/B 검정의 표준 도구**이며, 전환율·감염률·지지율 비교가 모두 이 형태다.
- **차이만이 답은 아니다.** 비율이 아주 작으면 절대차가 작아 보여도 상대위험이나 오즈비가 더 의미 있는 요약일 수 있다.

다음 절 **$\sigma_1^2/\sigma_2^2$ 에 대한 $F$ 검정**으로 넘어간다.
