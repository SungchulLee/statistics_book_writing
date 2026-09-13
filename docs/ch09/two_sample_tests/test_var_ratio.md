# 두 분산에 대한 F-검정

## 개요

두 분산에 대한 F-검정은 독립인 두 정규모집단의 분산을 비교한다. 두 집단의 변동성이 같은지 평가하는 데 쓰이며, 이는 합동 이표본 t-검정의 전제 조건이다. 등분산이라는 귀무가설 아래에서 검정통계량은 F-분포를 따른다. 카이제곱 분산 검정과 마찬가지로 정규성에서 벗어나는 데 민감하다.

## 검정의 구성

**가설:** 분산이 $\sigma_1^2$과 $\sigma_2^2$인 독립인 두 정규모집단에 대해,

- 양측: $H_0\colon \sigma_1^2/\sigma_2^2 = \theta_0$ 대 $H_1\colon \sigma_1^2/\sigma_2^2 \neq \theta_0$
- 단측: $H_0\colon \sigma_1^2/\sigma_2^2 = \theta_0$ 대 $H_1\colon \sigma_1^2/\sigma_2^2 > \theta_0$

가장 흔한 경우는 $\theta_0 = 1$(등분산 검정)이다.

**검정통계량:** 크기 $n_1$, $n_2$인 표본의 표본표준편차 $S_1$, $S_2$가 주어졌을 때, $H_0$과 정규성 가정 아래에서

$$
F = \frac{S_1^2 / S_2^2}{\theta_0} \sim F_{n_1-1,\,n_2-1}
$$

이다.

<div class="codebox" markdown>

### 예제 1. 분산비 검정 계산기 { .eg }

```python
from scipy.stats import f

def test_ratio_two_variances(n1, s1, n2, s2, theta0=1.0,
                              alt="two-sided", alpha=0.05):
    """H0: sigma1^2 / sigma2^2 = theta0. **정규모집단**을 가정한다.

    F-검정은 정규성에서 벗어나는 데 특히 약하다. 꼬리가 조금만 두꺼워도
    제1종 오류율이 크게 부풀기 때문에, 실무에서 등분산 사전검정으로 쓰는 것은
    권장되지 않는다(그냥 Welch를 쓰는 편이 낫다).
    s1, s2에는 ddof=1로 계산한 표본표준편차를 넣는다.
    """
    # 자유도의 **순서**가 중요하다. 분자 쪽이 df1이다.
    # s1과 s2를 바꿔 넣으면 자유도도 함께 바꿔야 하며, 그러지 않으면
    # 오류 없이 조용히 틀린 p-값이 나온다.
    df1, df2 = n1 - 1, n2 - 1
    F_stat = (s1 ** 2 / s2 ** 2) / theta0
    if alt == "two-sided":
        p = 2 * min(f.cdf(F_stat, df1, df2),
                     1 - f.cdf(F_stat, df1, df2))
    elif alt == "less":
        p = f.cdf(F_stat, df1, df2)
    else:
        p = 1 - f.cdf(F_stat, df1, df2)
    return F_stat, p, (p < alpha)
```

</div>

<div class="codebox" markdown>

### 예제 2. 두 분산에 대한 F-검정 { .eg }

```python
F_stat, p, reject = test_ratio_two_variances(
    n1=15, s1=1.3, n2=12, s2=0.9, theta0=1.0, alt="greater"
)
print("F:", F_stat, "p:", p, "reject:", reject)

# 두 집단의 역할을 바꾸면 F는 역수가 되고 대립가설의 방향도 뒤집힌다.
# 제대로 바꾸면 p-값은 같아야 한다.
F2, p2, reject2 = test_ratio_two_variances(
    n1=12, s1=0.9, n2=15, s2=1.3, theta0=1.0, alt="less"
)
print("F:", F2, "p:", p2, "reject:", reject2)
```

출력:

```
F: 2.0864197530864197 p: 0.11291422151817565 reject: False
F: 0.47928994082840237 p: 0.11291422151817561 reject: False
```

표준편차가 1.3과 0.9로 1.4배 차이, 분산으로는 두 배가 넘는데도 기각하지 못한다. $n = 15$와 $12$로는 분산비를 가려낼 힘이 없다.

두 번째 줄은 집단의 순서를 바꾼 것으로, $F$가 정확히 역수($1/2.086 = 0.479$)가 되고 p-값도 (부동소수점 끝자리를 빼면) 같다. 순서를 바꿀 때 자유도와 대립가설의 방향까지 함께 바꿔야 이렇게 일치한다.

</div>

### 해석

$s_1 = 1.3$, $n_1 = 15$와 $s_2 = 0.9$, $n_2 = 12$로 $H_0\colon \sigma_1^2 = \sigma_2^2$ 대 $H_1\colon \sigma_1^2 > \sigma_2^2$을 검정한다. F-통계량은

$$
F = \frac{1.3^2}{0.9^2} = \frac{1.69}{0.81} \approx 2.086.
$$

자유도 $(14, 11)$에서 단측 p-값 $P(F_{14,11} \geq 2.086)$이 $H_0$의 기각 여부를 결정한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> 두 생산라인의 표본에서 $s_1 = 4.2$ ($n_1 = 20$), $s_2 = 3.1$ ($n_2 = 25$)을 얻었다. $\alpha = 0.05$에서 $H_0\colon \sigma_1^2 = \sigma_2^2$ 대 $H_1\colon \sigma_1^2 \neq \sigma_2^2$을 검정하라.

</div>

??? success "풀이"

    F-통계량은

    $$
    F = \frac{4.2^2}{3.1^2} = \frac{17.64}{9.61} \approx 1.836.
    $$

    자유도는 $(19, 24)$이다. 양측검정의 p-값은 $2 \times \min(P(F \leq 1.836),\, P(F \geq 1.836))$이다. 표나 소프트웨어로 구하면 $P(F_{19,24} \geq 1.836) \approx 0.082$이므로 양측 p-값은 약 $0.164$이다. $0.164 > 0.05$이므로 $H_0$을 기각하지 못한다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 독립인 두 카이제곱 확률변수의 비에서 F-검정통계량을 유도하라.

</div>

??? success "풀이"

    정규성 아래에서 $i = 1, 2$에 대해 $(n_i - 1)S_i^2/\sigma_i^2 \sim \chi^2_{n_i - 1}$이고 둘은 독립이다. F-분포는 독립인 두 카이제곱 변수를 각각의 자유도로 나눈 비로 정의된다:

    $$
    F = \frac{\chi^2_{n_1-1}/(n_1-1)}{\chi^2_{n_2-1}/(n_2-1)} = \frac{(n_1-1)S_1^2/\sigma_1^2/(n_1-1)}{(n_2-1)S_2^2/\sigma_2^2/(n_2-1)} = \frac{S_1^2/\sigma_1^2}{S_2^2/\sigma_2^2}.
    $$

    $H_0\colon \sigma_1^2/\sigma_2^2 = \theta_0$ 아래에서 이는 $F = (S_1^2/S_2^2)/\theta_0 \sim F_{n_1-1,\,n_2-1}$로 간단해진다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> 분산에 대한 F-검정이 평균에 대한 t-검정보다 비정규성에 민감한 이유를 설명하라. 어떤 대안이 있는가?

</div>

??? success "풀이"

    t-검정은 중심극한정리 덕분에 $n$이 적당하면 $\bar{X}$의 표본분포가 근사적으로 정규이므로 웬만한 비정규성에 로버스트하다. 그러나 $S^2$의 분포는 모집단의 첨도에 의존한다. 꼬리가 두꺼운 분포에서는 $S^2$의 변동이 카이제곱분포가 예측하는 것보다 훨씬 커서 F-비가 왜곡된다. 따라서 분산에 대한 F-검정은 비정규성 아래에서 제1종 오류율이 부풀려진다.

    대안:

    - **Levene 검정**: 평균으로부터의 절대편차를 쓰며 비정규성에 로버스트하다.
    - **Brown–Forsythe 검정**: Levene 검정에서 평균 대신 중앙값을 쓴다.
    - **Bartlett 검정**: 가능도비 검정이며 비정규성에는 여전히 민감하지만 분산분석 맥락에서 흔히 쓰인다.
    - **붓스트랩 방법**: 재표본추출에 기반한 비모수적 분산 비교. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> $F \sim F_{\nu_1, \nu_2}$이면 $1/F \sim F_{\nu_2, \nu_1}$임을 보여라.

</div>

??? success "풀이"

    정의에 의해 독립인 $U \sim \chi^2_{\nu_1}$, $V \sim \chi^2_{\nu_2}$에 대해 $F = (U/\nu_1)/(V/\nu_2)$이다. 그러면

    $$
    \frac{1}{F} = \frac{V/\nu_2}{U/\nu_1},
    $$

    이는 $\chi^2_{\nu_2}/\nu_2$와 $\chi^2_{\nu_1}/\nu_1$의 비이다. 정의에 의해 이것이 $F_{\nu_2,\nu_1}$이다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span> $\alpha = 0.10$, $n_1 = 10$, $n_2 = 8$인 양측 F-검정에서 $P(F_L < F < F_U) = 0.90$이 되는 임계값 $F_L$과 $F_U$를 구하라.

</div>

??? success "풀이"

    자유도는 $(\nu_1, \nu_2) = (9, 7)$이다. 위쪽 임계값은

    $$
    F_U = F_{0.95,\,9,\,7} \approx 3.677.
    $$

    아래쪽 임계값은 역수 성질을 쓴다:

    $$
    F_L = \frac{1}{F_{0.95,\,7,\,9}} \approx \frac{1}{3.293} \approx 0.304.
    $$

    채택역은 $0.304 < F < 3.677$이다. 관측된 F가 이 구간 밖이면 $H_0$을 기각한다. $\square$

---

## 정리하며

분산 비 검정의 **구현**에서 챙길 점들이다.

- **`ddof=1` 을 쓴다.** 통계량이 베셀 수정한 표본분산의 비이며, `numpy` 기본값은 `ddof=0` 이다.
- **양측 $p$ 값에 주의한다.** $F$ 분포가 비대칭이라 작은 쪽 꼬리 확률의 두 배를 쓰되 $1$ 을 넘지 않게 자른다.
- **자유도의 순서가 중요하다.** $F_{n_1-1,\,n_2-1}$ 에서 분자와 분모를 바꾸면 다른 분포이며, 뒤집기 성질 $F_{1-\alpha,d_1,d_2}=1/F_{\alpha,d_2,d_1}$ 로 서로 연결된다.
- **$\theta_0\ne1$ 도 검정할 수 있다.** "한 공정의 분산이 다른 공정의 두 배를 넘는가" 같은 물음이 그 형태다.
- **정규성 민감도를 다시 확인하라.** 카이제곱 분산 검정과 마찬가지이며, **합동 $t$ 검정의 사전 확인 용도로는 쓰지 않는다.**

다음 절 **이표본 평균 검정**으로 넘어간다.
