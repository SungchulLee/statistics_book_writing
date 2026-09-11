# 일표본 분산 검정

## 개요

일표본 분산 검정(분산에 대한 카이제곱 검정)은 모분산 $\sigma^2$이 가설의 값 $\sigma_0^2$과 같은지 평가한다. 이 검정은 바탕 모집단이 정규분포를 따른다고 가정한다. 제조 공정이 허용 가능한 변동성을 유지하는지 확인하는 품질관리에서 흔히 쓰인다.

## 검정의 구성

**가설:**

- 양측: $H_0\colon \sigma^2 = \sigma_0^2$ 대 $H_1\colon \sigma^2 \neq \sigma_0^2$
- 단측: $H_0\colon \sigma^2 = \sigma_0^2$ 대 $H_1\colon \sigma^2 > \sigma_0^2$ (또는 $< \sigma_0^2$)

**검정통계량:** 크기 $n$인 표본의 표본분산 $S^2$($\text{ddof}=1$)이 주어졌을 때, $H_0$과 정규성 가정 아래에서

$$
\chi^2 = \frac{(n-1)\,S^2}{\sigma_0^2} \sim \chi^2_{n-1}
$$

이다.

수준 $\alpha$의 **양측**검정에서는 다음이면 $H_0$을 기각한다.

$$
\chi^2 < \chi^2_{\alpha/2,\,n-1} \quad \text{or} \quad \chi^2 > \chi^2_{1-\alpha/2,\,n-1}.
$$

## 코드

```python
from scipy.stats import chi2

def test_variance_one_sample(n, s2, sigma0, alt="two-sided", alpha=0.05):
    """H0: sigma^2 = sigma0^2. **정규모집단**을 가정한다.

    이 가정은 형식적인 단서가 아니다. 평균 검정과 달리 여기서는
    중심극한정리가 도와주지 않아서 n을 키워도 비정규성이 상쇄되지 않는다.
    s2에는 ddof=1로 계산한 표본분산을 넣는다.
    """
    df = n - 1
    chi2_stat = df * s2 / (sigma0 ** 2)
    if alt == "two-sided":
        # 카이제곱분포는 비대칭이라 "양쪽 꼬리"를 나누는 방식이 여럿이다.
        # 여기서는 작은 쪽 꼬리를 두 배 하는 관례를 따랐다. 간단하고
        # 신뢰구간과 어긋나지 않지만, 확률이 반씩 나뉘지는 않는다.
        p = 2 * min(chi2.cdf(chi2_stat, df), 1 - chi2.cdf(chi2_stat, df))
    elif alt == "less":
        p = chi2.cdf(chi2_stat, df)
    else:
        p = 1 - chi2.cdf(chi2_stat, df)
    return chi2_stat, p, (p < alpha)
```

**보기 1.**

```python
stat, p, reject = test_variance_one_sample(
    n=12, s2=2.1**2, sigma0=2.0, alt="greater"
)
print("chi2:", stat, "p:", p, "reject:", reject)

# 표본분산은 그대로 두고 표본크기만 키우면 어떻게 되는지 본다.
for n in [12, 50, 200]:
    st, pv, rj = test_variance_one_sample(n=n, s2=2.1**2, sigma0=2.0, alt="greater")
    print(f"n={n:>4}: chi2={st:8.2f}  p={pv:.4f}  reject={rj}")
```

출력:

```
chi2: 12.127500000000001 p: 0.35413619761553705 reject: False
n=  12: chi2=   12.13  p=0.3541  reject=False
n=  50: chi2=   54.02  p=0.2885  reject=False
n= 200: chi2=  219.40  p=0.1532  reject=False
```

표준편차가 2.0이 아니라 2.1이라는 같은 증거를 놓고도 $n = 12$에서는 $p = 0.35$, $n = 200$에서도 $p = 0.15$다. 분산에서 5% 차이를 잡아내려면 이보다도 훨씬 큰 표본이 필요하다. 분산은 평균보다 추정하기 어렵고, 그래서 검정하기도 어렵다.

### 해석

$n=12$, $s^2 = 4.41$로 $H_0\colon \sigma^2 = 4.0$ 대 $H_1\colon \sigma^2 > 4.0$을 검정한다. 검정통계량은

$$
\chi^2 = \frac{11 \times 4.41}{4.0} = 12.1275.
$$

$H_0$ 아래에서 $\chi^2 \sim \chi^2_{11}$이다. 단측 p-값 $P(\chi^2_{11} \geq 12.1275) \approx 0.353$이므로 $H_0$을 기각하지 못한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** 어떤 기계가 목표 분산 $\sigma_0^2 = 0.01$ mL$^2$으로 병을 채운다. 병 $n = 25$개의 표본에서 $s^2 = 0.015$를 얻었다. $\alpha = 0.05$에서 $H_0\colon \sigma^2 = 0.01$ 대 $H_1\colon \sigma^2 > 0.01$을 검정하라.

</div>

??? success "풀이"

    검정통계량은

    $$
    \chi^2 = \frac{24 \times 0.015}{0.01} = 36.0.
    $$

    $H_0$ 아래에서 $\chi^2 \sim \chi^2_{24}$이다. 임계값은 $\chi^2_{0.05,\,24} = 36.415$이다. $36.0 < 36.415$이므로 아슬아슬하게 $H_0$을 기각하지 못한다. p-값은 $P(\chi^2_{24} \geq 36.0) \approx 0.055$이다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** 분산에 대한 카이제곱 검정이 정규성 가정에 민감한 이유를 설명하라. 모집단의 꼬리가 두꺼우면 어떻게 되는가?

</div>

??? success "풀이"

    검정통계량 $(n-1)S^2/\sigma_0^2 \sim \chi^2_{n-1}$은 자료가 정규분포에서 나올 때에만 정확히 성립한다. 표본분산의 카이제곱분포는 모집단의 4차 적률(첨도)에 의존한다. 꼬리가 두꺼운 분포(예: 자유도가 작은 $t$-분포)에서는 표본분산 $S^2$의 변동이 카이제곱분포가 예측하는 것보다 크다. 그 결과 실제 제1종 오류율이 명목 $\alpha$보다 훨씬 커져 검정을 믿을 수 없게 된다. 이런 경우에는 로버스트한 대안(예: Levene 검정이나 붓스트랩 방법)을 택한다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** 정규성 아래에서 $(n-1)S^2/\sigma^2$의 분포를 유도하라.

</div>

??? success "풀이"

    $X_1, \dots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$이라 하자. $Z_i = (X_i - \mu)/\sigma \overset{\text{iid}}{\sim} N(0,1)$로 두면 $\sum Z_i^2 \sim \chi^2_n$이다. 표본분산은

    $$
    S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2.
    $$

    상수 벡터의 직교여공간으로 사영하면 차원이 1 줄어들므로, Cochran 정리에 의해 $\sum (X_i - \bar{X})^2 / \sigma^2 \sim \chi^2_{n-1}$이다. 따라서

    $$
    \frac{(n-1)S^2}{\sigma^2} = \frac{\sum(X_i - \bar{X})^2}{\sigma^2} \sim \chi^2_{n-1}.
    $$

    $H_0\colon \sigma^2 = \sigma_0^2$ 아래에서 $\sigma^2$ 자리에 $\sigma_0^2$을 넣으면 검정통계량이 된다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** $n = 20$, $\alpha = 0.05$의 양측검정에서 임계값 $\chi^2_{L}$과 $\chi^2_{U}$ 및 $\chi^2$의 채택역을 구하라.

</div>

??? success "풀이"

    $\text{df} = 19$, $\alpha/2 = 0.025$일 때:

    $$
    \chi^2_L = \chi^2_{0.025,\,19} = 8.907, \qquad \chi^2_U = \chi^2_{0.975,\,19} = 32.852.
    $$

    채택역($H_0$을 기각하지 못하는 영역)은 $8.907 \leq \chi^2 \leq 32.852$이다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** 어떤 품질 엔지니어가 측정값 $n = 15$개를 모아 $s = 3.2$를 얻었다. $\sigma^2$의 95% 신뢰구간을 구성하고 이를 써서 $H_0\colon \sigma^2 = 9$를 검정하라.

</div>

??? success "풀이"

    $\sigma^2$의 $100(1-\alpha)\%$ 신뢰구간은

    $$
    \left(\frac{(n-1)S^2}{\chi^2_{1-\alpha/2,\,n-1}},\; \frac{(n-1)S^2}{\chi^2_{\alpha/2,\,n-1}}\right).
    $$

    $n=15$, $s^2 = 10.24$, $\text{df}=14$, $\chi^2_{0.975,14} = 26.119$, $\chi^2_{0.025,14} = 5.629$이므로:

    $$
    \left(\frac{14 \times 10.24}{26.119},\; \frac{14 \times 10.24}{5.629}\right) = \left(\frac{143.36}{26.119},\; \frac{143.36}{5.629}\right) = (5.49,\; 25.47).
    $$

    $\sigma_0^2 = 9$가 구간 $(5.49, 25.47)$ 안에 있으므로 $\alpha = 0.05$에서 $H_0\colon \sigma^2 = 9$를 기각하지 못한다. $\square$

---

## 정리하며

분산 검정의 구현에서 주의할 점은 **분포의 비대칭**이다.

- **양측 $p$ 값을 두 배로 적당히 만들 수 없다.** 카이제곱분포가 비대칭이므로 양쪽 꼬리 확률이 다르며, 관례적으로 작은 쪽 꼬리 확률의 두 배를 쓰되 $1$ 을 넘지 않게 자른다.
- **`ddof=1` 을 반드시 확인한다.** 통계량이 $(n-1)s^2/\sigma_0^2$ 이므로 베셀 수정한 $s^2$ 을 써야 한다. `numpy.var()` 의 기본값은 `ddof=0` 이다(7장).
- **자유도는 $n-1$ 이다.** 평균을 추정하느라 하나를 잃은 결과다.
- **품질관리가 대표적 응용이다.** 공정의 변동이 규격을 넘는지 검정하며, 대개 **단측**($\sigma^2>\sigma_0^2$)으로 쓴다.
- **정규성 확인이 검정 자체보다 중요하다.** 이 검정은 정규성에서 벗어나면 결과를 믿을 수 없으므로, 14장의 정규성 진단을 먼저 하거나 15장의 로버스트 대안으로 넘어가야 한다.

다음 절부터 **이표본 검정**으로 넘어간다. 두 집단을 비교하는 문제다.
