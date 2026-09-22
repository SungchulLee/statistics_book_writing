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

<div class="codebox" markdown>

### 예제 1. 일표본 분산 검정 계산기 { .eg }

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

</div>

<div class="codebox" markdown>

### 예제 2. 일표본 분산 검정 { .eg }

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

</div>

### 해석

$n=12$, $s^2 = 4.41$로 $H_0\colon \sigma^2 = 4.0$ 대 $H_1\colon \sigma^2 > 4.0$을 검정한다. 검정통계량은

$$
\chi^2 = \frac{11 \times 4.41}{4.0} = 12.1275.
$$

$H_0$ 아래에서 $\chi^2 \sim \chi^2_{11}$이다. 단측 p-값 $P(\chi^2_{11} \geq 12.1275) \approx 0.353$이므로 $H_0$을 기각하지 못한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> 어떤 기계가 목표 분산 $\sigma_0^2 = 0.01$ mL$^2$으로 병을 채운다. 병 $n = 25$개의 표본에서 $s^2 = 0.015$를 얻었다. $\alpha = 0.05$에서 $H_0\colon \sigma^2 = 0.01$ 대 $H_1\colon \sigma^2 > 0.01$을 검정하라.

</div>

??? success "풀이"

    검정통계량은

    $$
    \chi^2 = \frac{24 \times 0.015}{0.01} = 36.0.
    $$

    $H_0$ 아래에서 $\chi^2 \sim \chi^2_{24}$이다. 임계값은 $\chi^2_{0.05,\,24} = 36.415$이다. $36.0 < 36.415$이므로 아슬아슬하게 $H_0$을 기각하지 못한다. p-값은 $P(\chi^2_{24} \geq 36.0) \approx 0.055$이다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 분산에 대한 카이제곱 검정이 정규성 가정에 민감한 이유를 설명하라. 모집단의 꼬리가 두꺼우면 어떻게 되는가?

</div>

??? success "풀이"

    검정통계량 $(n-1)S^2/\sigma_0^2 \sim \chi^2_{n-1}$은 자료가 정규분포에서 나올 때에만 정확히 성립한다. 표본분산의 카이제곱분포는 모집단의 4차 적률(첨도)에 의존한다. 꼬리가 두꺼운 분포(예: 자유도가 작은 $t$-분포)에서는 표본분산 $S^2$의 변동이 카이제곱분포가 예측하는 것보다 크다. 그 결과 실제 제1종 오류율이 명목 $\alpha$보다 훨씬 커져 검정을 믿을 수 없게 된다. 이런 경우에는 로버스트한 대안(예: Levene 검정이나 붓스트랩 방법)을 택한다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> 정규성 아래에서 $(n-1)S^2/\sigma^2$의 분포를 유도하라.

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

**연습문제 4.** <span class="diff easy" title="쉬움"></span> $n = 20$, $\alpha = 0.05$의 양측검정에서 임계값 $\chi^2_{L}$과 $\chi^2_{U}$ 및 $\chi^2$의 채택역을 구하라.

</div>

??? success "풀이"

    $\text{df} = 19$, $\alpha/2 = 0.025$일 때:

    $$
    \chi^2_L = \chi^2_{0.025,\,19} = 8.907, \qquad \chi^2_U = \chi^2_{0.975,\,19} = 32.852.
    $$

    채택역($H_0$을 기각하지 못하는 영역)은 $8.907 \leq \chi^2 \leq 32.852$이다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span> 어떤 품질 엔지니어가 측정값 $n = 15$개를 모아 $s = 3.2$를 얻었다. $\sigma^2$의 95% 신뢰구간을 구성하고 이를 써서 $H_0\colon \sigma^2 = 9$를 검정하라.

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

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
분산 검정의 **검정력**을 계산하고 표본크기를 정하라. 평균 검정과 비교하면 얼마나 비싼가?

</div>

??? success "풀이"
    **검정력.** 참 분산비를 $r=\sigma^2/\sigma_0^2$라 하면 $W=(n-1)S^2/\sigma_0^2\sim r\,\chi^2_{n-1}$이므로

    $$
    \text{검정력}(r)=P\!\left(\chi^2_{\nu}<\frac{\chi^2_{\nu,\alpha/2}}{r}\right)
    +P\!\left(\chi^2_{\nu}>\frac{\chi^2_{\nu,1-\alpha/2}}{r}\right)
    $$

    ```python
    import numpy as np
    from scipy import stats

    def power_var(n, r, alpha=0.05):
        nu = n - 1
        lo = stats.chi2.ppf(alpha / 2, nu)
        hi = stats.chi2.ppf(1 - alpha / 2, nu)
        return stats.chi2.cdf(lo / r, nu) + stats.chi2.sf(hi / r, nu)

    rs = [0.5, 0.67, 1.5, 2.0, 3.0]
    print(f"{'n':>5s} " + " ".join(f"{'r='+str(r):>9s}" for r in rs))
    for n in [10, 20, 30, 50, 100, 200]:
        print(f"{n:5d} " + " ".join(f"{power_var(n, r):9.4f}" for r in rs))

    print()
    for r in [1.5, 2.0, 3.0, 0.5]:
        n = next(m for m in range(5, 5000) if power_var(m, r) >= 0.80)
        print(f"σ²/σ0² = {r}: 검정력 80%에 필요한 n = {n}")
    ```

    ```text
        n     r=0.5    r=0.67     r=1.5     r=2.0     r=3.0
       10    0.2020    0.0914    0.1833    0.3934    0.7057
       20    0.4650    0.1770    0.2911    0.6289    0.9255
       30    0.6842    0.2687    0.3910    0.7829    0.9830
       50    0.9152    0.4494    0.5623    0.9324    0.9993
      100    0.9987    0.7787    0.8289    0.9974    1.0000
      200    1.0000    0.9788    0.9806    1.0000    1.0000

    σ²/σ0² = 1.5: 검정력 80%에 필요한 n = 93
    σ²/σ0² = 2.0: 검정력 80%에 필요한 n = 32
    σ²/σ0² = 3.0: 검정력 80%에 필요한 n = 13
    σ²/σ0² = 0.5: 검정력 80%에 필요한 n = 38
    ```

    **분산이 50% 늘어난 것을 탐지하려면 93개가 필요하다.** 두 배가 되어야 32개다.

    **비대칭에 주목하라.** $r=2.0$(두 배)에는 32개, $r=0.5$(절반)에는 38개다. **같은 "두 배 차이"인데 표본이 다르다.** 카이제곱 분포의 비대칭 때문이며, 앞서 본 편향된 검정의 또 다른 표현이다.

    **평균 검정과 비교.**

    ```python
    za, zb = stats.norm.ppf(0.975), stats.norm.ppf(0.80)
    print(f"평균이 0.5σ 이동: n = {np.ceil((za + zb)**2 / 0.5**2):.0f}")
    print(f"평균이 1.0σ 이동: n = {np.ceil((za + zb)**2 / 1.0**2):.0f}")
    ```

    ```text
    평균이 0.5σ 이동: n = 32
    평균이 1.0σ 이동: n = 8
    ```

    **"분산이 두 배"와 "평균이 0.5σ 이동"이 비슷한 난이도**(32개)다. 그런데 분산이 두 배라는 것은 **표준편차가 1.41배**로, 눈에 띄게 큰 변화다. 반면 평균 0.5σ 이동은 앞서 본 대로 "중간 효과"에 불과하다.

    **결론 — 분산 검정은 비싸다.** 같은 "체감 크기"의 변화를 탐지하는 데 훨씬 많은 표본이 든다. 이유는

    1. $S^2$의 상대표준오차가 $\sqrt{2/(n-1)}$로 $\bar X$의 $1/\sqrt n$보다 크고,
    2. 분산 척도에서의 변화가 원 척도에서는 제곱근으로 압축되기 때문이다.

    **실무 함의.** 공정의 산포를 감시하려면 **평균 감시보다 훨씬 큰 표본**이나 **누적합 기반 방법**이 필요하다. 이것이 $\bar X$ 관리도와 $S$ 관리도를 함께 쓰되, 후자에 더 많은 자료를 쓰는 관행의 배경이다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span>
분산 검정의 **첨도 보정판**을 구현하고, 카이제곱 검정과 비교하라.

</div>

??? success "풀이"
    **착안.** $\log S^2$의 점근분산이 $(\gamma_2+2)/(n-1)$이므로, 표본첨도를 넣어 보정한다.

    $$
    Z=\frac{\log S^2-\log\sigma_0^2}{\sqrt{(\hat\gamma_2+2)/(n-1)}}
    $$

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(5)
    n, M = 30, 20_000
    nu = n - 1
    lo = stats.chi2.ppf(0.025, nu)
    hi = stats.chi2.ppf(0.975, nu)

    cases = [("정규", lambda s: rng.normal(0, 1, s), 1.0),
             ("t(5)", lambda s: rng.standard_t(5, s), 5 / 3),
             ("지수", lambda s: rng.exponential(1, s), 1.0)]
    print(f"{'분포':>7s} {'카이제곱':>10s} {'첨도 보정':>10s}")
    for name, gen, var in cases:
        x = gen((M, n))
        s2 = x.var(1, ddof=1)
        chi = nu * s2 / var
        rej_chi = (chi < lo) | (chi > hi)
        kur = stats.kurtosis(x, axis=1, bias=False)
        z = (np.log(s2) - np.log(var)) / np.sqrt((kur + 2) / nu)
        print(f"{name:>7s} {rej_chi.mean():10.4f} "
              f"{np.mean(np.abs(z) > 1.96):10.4f}")
    ```

    ```text
       분포      카이제곱     첨도 보정
       정규     0.0492     0.0733
     t(5)     0.1766     0.1323
       지수     0.2830     0.1711
    ```

    **개선되지만 충분하지 않다.** 지수분포에서 0.283 → 0.171, $t_5$에서 0.177 → 0.132다.

    **정규에서는 손해다.** 0.049 → 0.073. 첨도를 추정하느라 변동이 늘었다.

    **왜 완전히 고쳐지지 않는가.**

    1. **$\hat\gamma_2$의 추정오차가 크다.** $n=30$에서 표본첨도의 표준오차가 대략 $\sqrt{24/n}=0.89$인데, 지수분포의 참 $\gamma_2=6$을 그 정도 오차로 추정하는 것은 매우 부정확하다.

    2. **$\hat\gamma_2$ 자체가 8차 적률에 의존**한다. 두꺼운 꼬리에서는 그 적률이 거의 추정 불가능하다.

    3. **점근이론이 느리게 수렴**한다. $\log S^2$의 정규근사 자체가 $n=30$에서 부정확하다.

    **더 나은 대안.**

    | 방법 | 성격 |
    |---|---|
    | **부트스트랩(로그 척도)** | 첨도를 추정하지 않고 재표본이 반영 |
    | 보넷의 절사 첨도 | 이상치의 영향을 줄인 $\hat\gamma_2$ |
    | 순열 | 일표본에서는 적용이 어렵다 |
    | **분포를 모형화** | 지수라면 $\sigma=\mu$이므로 평균만 추정 |

    **가장 실용적인 결론.** **비정규가 의심되면 분산 검정 자체를 다시 생각한다.** 정말 궁금한 것이 "산포가 목표보다 큰가"라면, 사분위범위나 MAD 같은 **강건한 산포 측도**를 목표와 비교하는 것이 더 안정적이고 해석도 쉽다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
분산에 대한 **단측검정**을 구성하고, 공정관리에서 어느 방향이 중요한지 논하라.

</div>

??? success "풀이"
    **두 방향의 단측검정.**

    $$
    H_0:\sigma^2\le\sigma_0^2\ \text{대}\ H_1:\sigma^2>\sigma_0^2
    \quad\Rightarrow\quad W>\chi^2_{\nu,1-\alpha}\ \text{이면 기각}
    $$

    $$
    H_0:\sigma^2\ge\sigma_0^2\ \text{대}\ H_1:\sigma^2<\sigma_0^2
    \quad\Rightarrow\quad W<\chi^2_{\nu,\alpha}\ \text{이면 기각}
    $$

    ```python
    import numpy as np
    from scipy import stats

    n, s2, sigma0sq = 15, 0.0025, 0.0020
    nu = n - 1
    W = nu * s2 / sigma0sq
    print(f"W = {W:.4f}  (자유도 {nu})")
    print(f"양측 p = {2 * min(stats.chi2.cdf(W, nu), stats.chi2.sf(W, nu)):.4f}")
    print(f"상단 단측 p (σ² > σ0²) = {stats.chi2.sf(W, nu):.4f}")
    print(f"하단 단측 p (σ² < σ0²) = {stats.chi2.cdf(W, nu):.4f}")

    print(f"\n임계값: 상단 {stats.chi2.ppf(0.95, nu):.3f}, "
          f"하단 {stats.chi2.ppf(0.05, nu):.3f}")
    print(f"→ 상단 단측이면 s² 가 "
          f"{stats.chi2.ppf(0.95, nu) * sigma0sq / nu:.6f} 를 넘으면 기각")
    ```

    ```text
    W = 17.5000  (자유도 14)
    양측 p = 0.4610
    상단 단측 p (σ² > σ0²) = 0.2305
    하단 단측 p (σ² < σ0²) = 0.7695

    임계값: 상단 23.685, 하단 6.571
    → 상단 단측이면 s² 가 0.003384 를 넘으면 기각
    ```

    **어느 방향이 중요한가 — 대부분 "커지는 쪽"이다.**

    | 상황 | 관심 방향 | 이유 |
    |---|---|---|
    | 제조 공정의 산포 | **커지는 쪽** | 불량 증가 |
    | 측정기기의 정밀도 | **커지는 쪽** | 신뢰도 저하 |
    | 금융 변동성 | 양쪽 | 위험 관리와 기회 |
    | 시험 문제의 변별력 | **작아지는 쪽** | 변별이 안 됨 |
    | 공정 개선 검증 | **작아지는 쪽** | 개선을 입증 |

    **공정관리에서는 대개 상단 단측이 맞다.** 분산이 줄어드는 것은 좋은 일이므로 경보할 이유가 없다.

    **그런데 주의할 점.** 분산이 **갑자기 작아지는 것**이 이상 신호일 수 있다.

    - 측정기기가 고장 나 같은 값을 반복 출력.
    - 자료가 인위적으로 다듬어짐.
    - 표본이 실제로는 한 배치에서만 나옴.

    **따라서 감시 목적이라면 양쪽을 보되, 하한 경보를 "품질 개선"이 아니라 "이상 확인 필요"로 해석**하는 것이 실무적이다.

    **$S$ 관리도의 관행.** 하한 관리한계를 그리되, 그것을 벗어나면 **개선의 원인을 조사**한다. 진짜 개선이면 관리한계를 다시 계산한다.

    **단측검정의 이득.** 상단 단측이면 임계값이 23.685 대신(양측의 26.119) 낮아져 검정력이 오른다. 다만 앞서 본 대로 방향을 **사전에 확정**해야 한다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
분산 검정을 **부트스트랩**으로 수행하는 법을 보이고, 카이제곱 검정과 비교하라.

</div>

??? success "풀이"
    **방법.** $\log S^2$의 표집분포를 재표본으로 근사한다. $H_0$를 강제하기 위해 자료를 **척도 조정**한다.

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(88)
    x = rng.exponential(2.0, 40)          # 참 분산 4
    sigma0sq = 2.5
    n, B = len(x), 9_999
    s2 = x.var(ddof=1)

    # ① 카이제곱 검정
    W = (n - 1) * s2 / sigma0sq
    p_chi = 2 * min(stats.chi2.cdf(W, n - 1), stats.chi2.sf(W, n - 1))

    # ② 부트스트랩: H0 가 참이 되도록 자료를 척도 조정한 뒤 재표본
    xc = (x - x.mean()) * np.sqrt(sigma0sq / s2) + x.mean()
    idx = rng.integers(0, n, (B, n))
    s2b = xc[idx].var(1, ddof=1)
    t_obs = np.log(s2 / sigma0sq)
    tb = np.log(s2b / sigma0sq)
    p_boot = (np.sum(np.abs(tb) >= abs(t_obs)) + 1) / (B + 1)

    print(f"s² = {s2:.4f}  (σ0² = {sigma0sq})")
    print(f"카이제곱 p = {p_chi:.4f}")
    print(f"부트스트랩 p = {p_boot:.4f}")
    ```

    ```text
    s² = 3.0517  (σ0² = 2.5)
    카이제곱 p = 0.3245
    부트스트랩 p = 0.5945
    ```

    **두 $p$-값이 두 배 차이 난다**(0.32 대 0.59). 이 표본에서는 결론이 같지만, 경계 근처였다면 갈렸을 것이다.

    **부트스트랩 쪽이 옳다.** 자료가 지수분포라 첨도가 6이고, 카이제곱 검정은 $S^2$의 변동을 절반으로 과소평가해 $p$-값을 지나치게 작게 만든다. 앞서 여러 번 확인한 현상이다.

    **수준을 모의실험으로 확인하면.**

    ```python
    rng = np.random.default_rng(3)
    M, n, B = 1_000, 40, 999
    c_chi = c_boot = 0
    for _ in range(M):
        y = rng.exponential(1.0, n)        # 참 분산 1
        s2 = y.var(ddof=1)
        W = (n - 1) * s2 / 1.0
        c_chi += 2 * min(stats.chi2.cdf(W, n - 1),
                         stats.chi2.sf(W, n - 1)) < 0.05
        yc = (y - y.mean()) / np.sqrt(s2) + y.mean()
        ib = rng.integers(0, n, (B, n))
        s2b = yc[ib].var(1, ddof=1)
        c_boot += (np.sum(np.abs(np.log(s2b)) >= abs(np.log(s2))) + 1) \
            / (B + 1) < 0.05
    print(f"지수분포, n=40: 카이제곱 {c_chi / M:.4f}, "
          f"부트스트랩 {c_boot / M:.4f}")
    ```

    ```text
    지수분포, n=40: 카이제곱 0.2900, 부트스트랩 0.1310
    ```

    **카이제곱의 실제 수준이 0.290이다.** 명목의 여섯 배다. 부트스트랩은 0.131로 여전히 완벽하지 않지만 **절반 이하로 개선**된다.

    **부트스트랩도 완벽하지 않은 이유.** $n=40$에서 지수분포의 $S^2$ 표집분포는 여전히 심하게 치우쳐 있고, 백분위 부트스트랩은 **일차정확**이라 그 치우침을 완전히 보정하지 못한다. BCa나 부트스트랩-$t$를 쓰면 더 낫다.

    **권고.** 분산 검정이 꼭 필요하면 **정규성을 진단**하고, 위배가 보이면 부트스트랩이나 첨도 보정을 쓴다. 그러고도 수준이 완전하지 않다는 점을 인정하고 결과를 조심스럽게 해석한다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
분산 검정 결과를 **어떻게 보고하고 해석해야 하는지** 정리하라.

</div>

??? success "풀이"
    **보고할 것 일곱.**

    1. **$\sigma$ 척도로 변환.** 분산은 단위가 제곱이라 읽기 어렵다.
    2. **$\sigma_0$와 그 근거.** 규격, 과거 공정, 기기 사양.
    3. **검정통계량과 자유도.** $\chi^2(14)=17.5$.
    4. **$p$-값과 단측/양측.**
    5. **$\sigma$의 신뢰구간.** 비대칭 그대로.
    6. **정규성 진단 결과.** **이것이 없으면 결과를 믿을 수 없다.**
    7. **표본크기.** 분산 추정에 $n$이 결정적이다.

    **좋은 보고의 예.**

    > 볼베어링 15개의 지름 표준편차는 0.050 mm였다($s^2=0.0025$ mm²). 규격 상한인 0.045 mm와 비교하는 상단 단측 카이제곱 검정에서 $\chi^2(14)=17.28$, $p=0.241$이었다. 표준편차의 95% 신뢰구간은 0.037~0.079 mm로, 규격값 0.045를 담는다. 샤피로-윌크 검정과 Q-Q 그림에서 정규성 위배의 증거는 없었다($p=0.62$). 구간의 상한이 규격의 1.75배이므로, **규격 초과를 배제하지 못한다** — 표본을 늘려 재확인할 필요가 있다.

    **해석의 주의 다섯.**

    1. **"기각하지 못함"이 "규격을 만족함"이 아니다.** 앞서 본 대로 $n=15$에서는 구간이 매우 넓어 큰 초과도 배제하지 못한다.

    2. **$n$이 작으면 사실상 아무것도 말할 수 없다.** $n=10$이면 $\sigma$의 구간이 대략 $(0.69s,\ 1.83s)$로 2.7배 폭이다.

    3. **정규성이 깨지면 $p$-값이 무의미하다.** 앞 문제에서 본 대로 실제 수준이 0.25까지 간다.

    4. **분산이 관심사가 아닐 수 있다.** 진짜 질문이 "규격을 벗어나는 제품의 비율"이라면, 그것을 직접 추정하는 것이 낫다(공정능력지수, 허용구간).

    5. **$\sigma$와 $\sigma^2$의 혼동.** "분산이 두 배"는 "표준편차가 1.41배"다. 보고할 때 어느 척도인지 반드시 명시한다.

    **더 유용한 대안 지표.**

    | 목적 | 지표 |
    |---|---|
    | 규격 대비 산포 | **공정능력지수** $C_p=(\text{USL}-\text{LSL})/(6\sigma)$ |
    | 규격 이탈 비율 | 불량률 추정과 구간 |
    | 미래 개체의 범위 | **허용구간** |
    | 두 공정 비교 | 분산비와 그 구간 |

    **한 문장.** **분산 검정의 결과는 거의 언제나 "정밀도가 부족하다"로 요약된다.** 구간을 함께 보고하면 그 사실이 드러나고, 독자가 올바로 해석할 수 있다.

---

## 정리하며

분산 검정의 구현에서 주의할 점은 **분포의 비대칭**이다.

- **양측 $p$ 값을 두 배로 적당히 만들 수 없다.** 카이제곱분포가 비대칭이므로 양쪽 꼬리 확률이 다르며, 관례적으로 작은 쪽 꼬리 확률의 두 배를 쓰되 $1$ 을 넘지 않게 자른다.
- **`ddof=1` 을 반드시 확인한다.** 통계량이 $(n-1)s^2/\sigma_0^2$ 이므로 베셀 수정한 $s^2$ 을 써야 한다. `numpy.var()` 의 기본값은 `ddof=0` 이다(7장).
- **자유도는 $n-1$ 이다.** 평균을 추정하느라 하나를 잃은 결과다.
- **품질관리가 대표적 응용이다.** 공정의 변동이 규격을 넘는지 검정하며, 대개 **단측**($\sigma^2>\sigma_0^2$)으로 쓴다.
- **정규성 확인이 검정 자체보다 중요하다.** 이 검정은 정규성에서 벗어나면 결과를 믿을 수 없으므로, 14장의 정규성 진단을 먼저 하거나 15장의 로버스트 대안으로 넘어가야 한다.

다음 절부터 **이표본 검정**으로 넘어간다. 두 집단을 비교하는 문제다.
