# 평균 차이에 대한 이표본 Z-검정

## 모분산을 아는 경우

평균을 비교하는 대부분의 이표본 문제는 모분산 $\sigma_1^2$과 $\sigma_2^2$을 아는 경우가 드물기 때문에 [t-검정](t_test_two_means.md)을 쓴다. 그러나 방대한 과거 자료나 측정기기의 물리적 특성 등으로 이 분산들을 **아는** 경우에는, 이표본 z-검정이 자유도 조정 없이 표준정규분포에 기반한 정확한 검정을 제공한다. 분산을 모를 때에도 z-검정은 이표본 t-검정의 개념적 토대가 되므로 출발점으로 삼기에 유용하다.

## 설정과 기호

독립인 두 확률표본을 관측한다:

- 표본 1: 평균이 $\mu_1$이고 분산 $\sigma_1^2$을 **아는** 모집단에서 뽑은 $X_1, X_2, \ldots, X_{n_1}$
- 표본 2: 평균이 $\mu_2$이고 분산 $\sigma_2^2$을 **아는** 모집단에서 뽑은 $Y_1, Y_2, \ldots, Y_{n_2}$

표본평균은 $\bar{X} = \frac{1}{n_1}\sum_{i=1}^{n_1} X_i$와 $\bar{Y} = \frac{1}{n_2}\sum_{j=1}^{n_2} Y_j$이다. 차이 $\mu_1 - \mu_2$가 가설의 값 $\delta_0$(흔히 $\delta_0 = 0$)과 같은지 검정하고자 한다.

## 가설

귀무가설은:

$$
H_0 : \mu_1 - \mu_2 = \delta_0
$$

대립가설은 연구 질문에 따라 다음 세 형태 중 하나이다:

| 이름 | 대립가설 $H_a$ | 기각하는 경우 |
|---|---|---|
| 양측 | $\mu_1 - \mu_2 \neq \delta_0$ | $\lvert Z \rvert$가 클 때 |
| 좌측 | $\mu_1 - \mu_2 < \delta_0$ | $Z$가 매우 작을 때(음수) |
| 우측 | $\mu_1 - \mu_2 > \delta_0$ | $Z$가 매우 클 때 |

대부분의 응용에서 $\delta_0 = 0$이므로 검정은 두 모평균이 같은지를 묻는다.

## 검정통계량의 유도

귀무가설 아래에서 표본평균의 차이 $\bar{X} - \bar{Y}$가 $\delta_0$을 추정한다. 두 표본이 독립이므로 $\bar{X} - \bar{Y}$의 분산은 각 분산의 합이다:

$$
\text{Var}(\bar{X} - \bar{Y}) = \frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}
$$

두 모집단이 모두 정규이거나 $n_1$과 $n_2$가 중심극한정리를 적용할 만큼 크면 $\bar{X} - \bar{Y}$는 근사적으로 정규이다. $H_0$ 아래에서 표준화하면 검정통계량을 얻는다:

$$
Z = \frac{(\bar{X} - \bar{Y}) - \delta_0}{\sqrt{\dfrac{\sigma_1^2}{n_1} + \dfrac{\sigma_2^2}{n_2}}}
$$

$H_0$ 아래에서 이 통계량은 **표준정규분포**를 따른다: $Z \sim N(0, 1)$.

??? note "정확한 정규성과 근사적 정규성"
    두 모집단이 정확히 정규이면 표본크기와 무관하게 $Z$가 정확히 $N(0, 1)$을 따른다. 모집단이 정규가 아니면 중심극한정리에 의해 $n_1$과 $n_2$가 충분히 클 때 $Z$가 근사적으로 $N(0, 1)$이다. 흔한 지침은 $n_1 \geq 30$이고 $n_2 \geq 30$이지만, 이 문턱은 모집단 분포가 정규에서 얼마나 벗어나 있는지에 달려 있다.

## 기각역과 p-값

$z_{\alpha}$를 표준정규분포의 상위 $\alpha$ 분위수, $z_{\text{obs}}$를 검정통계량의 관측값이라 하자.

### 양측검정 (Hₐ: μ₁ - μ₂ ≠ δ₀)

**기각역**: $|z_{\text{obs}}| > z_{\alpha/2}$이면 $H_0$을 기각한다.

**p-값**:

$$
p = 2\,P(Z > |z_{\text{obs}}|) = 2\bigl[1 - \mathcal{N}(|z_{\text{obs}}|)\bigr]
$$

여기서 $\mathcal{N}$은 표준정규 누적분포함수이다.

### 좌측검정 (Hₐ: μ₁ - μ₂ < δ₀)

**기각역**: $z_{\text{obs}} < -z_{\alpha}$이면 $H_0$을 기각한다.

**p-값**:

$$
p = P(Z < z_{\text{obs}}) = \mathcal{N}(z_{\text{obs}})
$$

### 우측검정 (Hₐ: μ₁ - μ₂ > δ₀)

**기각역**: $z_{\text{obs}} > z_{\alpha}$이면 $H_0$을 기각한다.

**p-값**:

$$
p = P(Z > z_{\text{obs}}) = 1 - \mathcal{N}(z_{\text{obs}})
$$

## 가정

이표본 z-검정에는 네 가지 조건이 필요하다:

1. **분산을 안다**: 모분산 $\sigma_1^2$과 $\sigma_2^2$이 자료에서 추정한 값이 아니라 알려진 값이다. 추정해야 한다면 [이표본 t-검정](t_test_two_means.md)을 쓴다.
2. **표본 간 독립성**: 두 표본이 서로 독립적으로 뽑혔다. 한 표본의 관측값이 다른 표본의 관측값에 영향을 주거나 짝지어져 있지 않다.
3. **표본 내 독립성**: 각 표본 안의 관측값이 독립이다(예: 단순확률표본추출).
4. **정규성 또는 큰 표본**: 각 모집단이 정규분포를 따르거나, 표본크기가 중심극한정리로 충분한 정규근사를 얻을 만큼 크다($n_1 \geq 30$, $n_2 \geq 30$).

!!! warning "분산을 아는 경우는 현실적이지 않다"
    실무에서 모분산을 확실히 아는 경우는 거의 없다. 따라서 이표본 z-검정은 일상적인 분석 도구라기보다 이론적인 디딤돌에 가깝다. 분산을 모를 때 평균을 비교하는 표준 절차는 [이표본 t-검정](t_test_two_means.md)이다.

## 예제: 표준화 시험 점수의 비교

두 학군이 서로 다른 수학 교육과정을 채택했다. 과거 시험 자료에서 점수의 모표준편차가 $\sigma_1 = 12$(A 학군), $\sigma_2 = 15$(B 학군)로 알려져 있다. A 학군 학생 $n_1 = 50$명의 확률표본에서 평균 점수 $\bar{x} = 78$을, B 학군 학생 $n_2 = 60$명의 확률표본에서 평균 점수 $\bar{y} = 74$를 얻었다. $\alpha = 0.05$에서 두 학군의 평균 점수가 다른지 검정하라.

**1단계: 가설을 세운다.**

$$
H_0: \mu_1 - \mu_2 = 0 \quad \text{vs} \quad H_a: \mu_1 - \mu_2 \neq 0
$$

**2단계: 표준오차를 계산한다.**

$$
\text{SE} = \sqrt{\frac{12^2}{50} + \frac{15^2}{60}} = \sqrt{\frac{144}{50} + \frac{225}{60}} = \sqrt{2.88 + 3.75} = \sqrt{6.63} \approx 2.575
$$

**3단계: 검정통계량을 계산한다.**

$$
z_{\text{obs}} = \frac{(78 - 74) - 0}{2.575} = \frac{4}{2.575} \approx 1.553
$$

**4단계: p-값을 계산한다.**

$$
p = 2\bigl[1 - \mathcal{N}(1.553)\bigr] = 2(1 - 0.9398) = 2(0.0602) \approx 0.120
$$

**5단계: 판정한다.** $p \approx 0.120 > 0.05 = \alpha$이므로 $H_0$을 기각하지 못한다. 5% 유의수준에서 두 학군의 평균 수학 점수가 다르다고 결론지을 증거가 부족하다.

## Python 구현

```python
import numpy as np
from scipy import stats

def two_sample_z_test(x_bar, y_bar, sigma1, sigma2, n1, n2,
                      delta0=0, alternative="two-sided"):
    """Two-sample z-test for the difference of means.

    Parameters
    ----------
    x_bar, y_bar : float
        Sample means.
    sigma1, sigma2 : float
        Known population standard deviations.
    n1, n2 : int
        Sample sizes.
    delta0 : float
        Hypothesized difference (default 0).
    alternative : str
        'two-sided', 'less', or 'greater'.

    Returns
    -------
    z_stat : float
        Test statistic.
    p_value : float
        P-value.
    """
    # sigma를 알고 있으므로 표본에서 추정하지 않는다.
    # 그래서 자유도라는 개념이 없고 표준정규를 그대로 쓴다.
    se = np.sqrt(sigma1**2 / n1 + sigma2**2 / n2)
    z_stat = ((x_bar - y_bar) - delta0) / se

    if alternative == "two-sided":
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    elif alternative == "less":
        p_value = stats.norm.cdf(z_stat)
    elif alternative == "greater":
        p_value = 1 - stats.norm.cdf(z_stat)
    else:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    return z_stat, p_value


# 예제: A 학군 대 B 학군
z, p = two_sample_z_test(78, 74, 12, 15, 50, 60, alternative="two-sided")
print(f"z = {z:.3f}, p-value = {p:.3f}")

# 같은 차이 4점을 표본크기만 네 배로 늘려 다시 검정해 본다.
z4, p4 = two_sample_z_test(78, 74, 12, 15, 200, 240, alternative="two-sided")
print(f"z = {z4:.3f}, p-value = {p4:.3f}")
```

출력:

```
z = 1.553, p-value = 0.120
z = 3.107, p-value = 0.002
```

같은 4점 차이가 표본을 네 배로 늘리자 $p = 0.120$에서 $p = 0.002$로 바뀐다. $z$는 정확히 $\sqrt{4} = 2$배가 되었다. 표준오차가 $1/\sqrt{n}$로 줄기 때문이다.

p-값이 말해 주는 것은 효과의 크기가 아니라 "이 표본크기에서 이만한 차이를 우연으로 보기 어려운가"라는 점을 잘 보여준다. 두 경우의 효과크기는 완전히 같다.

## 이표본 t-검정과의 관계

이표본 z-검정과 [이표본 t-검정](t_test_two_means.md)은 구조가 같다. 차이는 분모뿐이다: z-검정은 알려진 분산 $\sigma_1^2$과 $\sigma_2^2$을 쓰고, t-검정은 이를 표본분산 $S_1^2$과 $S_2^2$으로 바꾼다. 이 대체가 추가적인 불확실성을 낳으므로 t-검정은 표준정규가 아니라 (꼬리가 더 두꺼운) $t$-분포를 쓴다. 표본크기가 커지면 $t$-분포가 표준정규에 가까워져 두 검정이 같아진다.

## 관련 주제

- [이표본 t-검정](t_test_two_means.md): 모분산을 모를 때의 표준 검정
- [비율에 대한 이표본 Z-검정](z_test_two_proportions.md): 두 모비율의 비교
- [두 분산에 대한 F-검정](f_test_two_variances.md): 모분산의 동일성 검정

## 연습문제

**연습문제 1.**
독립인 두 표본: $\bar{x}_1 = 82$, $\sigma_1 = 10$, $n_1 = 50$이고 $\bar{x}_2 = 78$, $\sigma_2 = 12$, $n_2 = 60$. 양측 z-검정으로 $\alpha = 0.05$에서 $H_0: \mu_1 - \mu_2 = 0$을 검정하라.

??? success "풀이"
    검정통계량은:

    $$
    Z = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\sigma_1^2/n_1 + \sigma_2^2/n_2}} = \frac{82 - 78}{\sqrt{100/50 + 144/60}} = \frac{4}{\sqrt{2 + 2.4}} = \frac{4}{\sqrt{4.4}} = \frac{4}{2.098} \approx 1.907
    $$

    양측검정의 임계값은 $z_{0.025} = 1.96$이다. $|Z| = 1.907 < 1.96$이므로 $H_0$을 기각하지 못한다.

    p-값은 $2P(Z > 1.907) = 2(0.0283) = 0.0566$이다. $\alpha = 0.05$에서 증거가 시사적이기는 하지만 통계적으로 유의하지는 않다.

---

**연습문제 2.**
연습문제 1의 자료로 $\mu_1 - \mu_2$의 95% 신뢰구간을 구성하라. 이 구간이 0을 포함하는가?

??? success "풀이"
    $\mu_1 - \mu_2$의 95% 신뢰구간은:

    $$
    (\bar{x}_1 - \bar{x}_2) \pm z_{0.025}\sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}} = 4 \pm 1.96 \times 2.098 = 4 \pm 4.112
    $$

    $$
    = (-0.112, 8.112)
    $$

    구간이 0을 포함하며, 연습문제 1에서 $H_0$을 기각하지 못한 것과 일관된다. 참고: 신뢰구간은 가설검정보다 많은 정보를 준다 — 참 차이가 작은 음수에서 약 8단위까지 걸쳐 있을 수 있음을 보여준다.

---

**연습문제 3.**
이표본 z-검정과 이표본 t-검정이 각각 언제 적절한지 설명하라. 모분산에 대한 어떤 가정이 결정적인가?

??? success "풀이"
    **이표본 z-검정**은 모분산 $\sigma_1^2$과 $\sigma_2^2$을 **알 때** 적절하다. 이때 검정통계량 $Z = (\bar{X}_1 - \bar{X}_2)/\sqrt{\sigma_1^2/n_1 + \sigma_2^2/n_2}$은 (정규성이나 큰 표본을 가정하면) $H_0$ 아래에서 정확히 표준정규분포를 따른다.

    **이표본 t-검정**은 모분산을 **모르고** 자료에서 추정해야 할 때 쓴다. 분산을 표본분산 $s_1^2, s_2^2$으로 바꾸며, 검정통계량은 (분산이 다를 때는 Welch 근사로) 근사적인 t-분포를 따르거나 ($\sigma_1^2 = \sigma_2^2$일 때 합동분산으로) 정확한 t-분포를 따른다.

    실무에서 모분산을 아는 경우는 거의 없으므로 t-검정이 훨씬 흔하다. z-검정은 주로 교육적인 디딤돌 역할을 하며, 표본이 아주 커서 $\sigma$와 $s$의 구분이 무시할 만할 때 적용할 수 있다.

---

**연습문제 4.**
"추정값 빼기 가설값을 표준오차로 나눈다"는 일반 원리에서 이표본 z-검정통계량을 유도할 수 있음을 보여라. $\bar{X}_1 - \bar{X}_2$의 표준오차는 무엇인가?

??? success "풀이"
    일반적인 검정통계량의 구조는:

    $$
    Z = \frac{\text{estimator} - \text{hypothesized value}}{\text{SE(estimator)}} = \frac{(\bar{X}_1 - \bar{X}_2) - \delta_0}{\text{SE}(\bar{X}_1 - \bar{X}_2)}
    $$

    $\bar{X}_1$과 $\bar{X}_2$가 독립이므로:

    $$
    \text{Var}(\bar{X}_1 - \bar{X}_2) = \text{Var}(\bar{X}_1) + \text{Var}(\bar{X}_2) = \frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}
    $$

    $$
    \text{SE}(\bar{X}_1 - \bar{X}_2) = \sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}
    $$

    $H_0: \mu_1 - \mu_2 = \delta_0$(보통 $\delta_0 = 0$) 아래에서 중심극한정리 또는 자료의 정확한 정규성에 의해 $Z \sim N(0, 1)$이다. 이 유도는 단일 평균에 대한 z-검정, 비율에 대한 z-검정, 이표본 z-검정을 하나의 틀로 통합한다.
