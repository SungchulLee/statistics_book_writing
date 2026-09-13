# 분산에 대한 카이제곱 검정

!!! note "이 주제를 다루는 다른 곳"
    분산분석의 등분산성 사전확인이라는 좁은 맥락에서 같은 검정을 짧게 쓰는 예가
    **11.5 가정**에 있다.

## 개요

분산에 대한 카이제곱 검정은 정규분포를 따르는 모집단의 분산이 가설로 세운 값과 같은지 판정하는 일표본 가설검정이다. 카이제곱분포 위에 직접 세워져 있으며, 평균에 대한 일표본 $z$ 검정이나 $t$ 검정의 분산판에 해당한다. 검정통계량이 표본분산과 가설 분산의 비에 의존하므로, 공정이 지정된 변동성 목표를 충족해야 하는 품질관리 상황에서 특히 유용하다.

## 검정 설정

$X_1, X_2, \ldots, X_n$이 $N(\mu, \sigma^2)$에서 나온 독립 확률표본이라 하자. 다음을 검정한다.

$$
H_0 : \sigma^2 = \sigma_0^2 \quad \text{대} \quad H_1 : \sigma^2 \neq \sigma_0^2
$$

여기서 $\sigma_0^2$은 가설로 세운 모분산이다.

## 검정통계량

표본분산을

$$
S^2 = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})^2
$$

로 정의한다. $H_0$ 아래에서 검정통계량

$$
T = \frac{(n-1) S^2}{\sigma_0^2}
$$

은 자유도 $n - 1$인 카이제곱분포를 따른다.

$$
T \sim \chi^2(n-1).
$$

## 판정규칙

유의수준 $\alpha$의 양측검정에서 다음이면 $H_0$을 기각한다.

$$
T < \chi^2_{\alpha/2,\, n-1} \quad \text{또는} \quad T > \chi^2_{1-\alpha/2,\, n-1},
$$

여기서 $\chi^2_{q,\, n-1}$은 $\chi^2(n-1)$의 $q$ 분위수이다. 동등하게 양측 $p$값은

$$
p = 2 \min\!\bigl(F_{\chi^2}(T),\; 1 - F_{\chi^2}(T)\bigr),
$$

여기서 $F_{\chi^2}$은 $\chi^2(n-1)$의 CDF이다.

아래 함수는 분산에 대한 일표본 카이제곱 검정을 구현한다.

<div class="codebox" markdown>

### 예제 1. 분산에 대한 카이제곱 검정 구현 { .eg }

```python
import numpy as np
import scipy.stats as stats


def chi2_test_for_variance(data, sigma2_0=1.0):
    """모분산이 sigma2_0 인지 검정하는 일표본 카이제곱 검정.

    H0: sigma^2 = sigma2_0
    H1: sigma^2 != sigma2_0

    카이제곱 분포는 좌우가 대칭이 아니므로, 양측 p-값은 두 꼬리 넓이 중
    작은 쪽을 두 배 해서 만든다.
    """
    n = len(data)
    s2 = np.var(data, ddof=1)
    statistic = (n - 1) * s2 / sigma2_0
    p_value = 2 * min(
        stats.chi2(df=n - 1).cdf(statistic),
        stats.chi2(df=n - 1).sf(statistic),
    )
    return statistic, p_value
```

전형적인 사용법은 참 표준편차를 바꿔가며 자료를 생성하고 검정이 $\sigma_0^2 = 1$로부터의 이탈을 탐지하는지 확인하는 것이다.

</div>

<div class="codebox" markdown>

### 예제 2. 분산을 키워 가며 검정하기 { .eg }

```python
import matplotlib.pyplot as plt

# 참 표준편차를 1 에서 1.2 까지 올려 가며 검정이 언제부터 잡아내는지 본다.
size, seed = 100, 0

for scale in [1.00, 1.05, 1.10, 1.15, 1.20]:
    y = stats.norm(loc=1, scale=scale).rvs(size, random_state=seed)
    stat, pval = chi2_test_for_variance(y, sigma2_0=1.0)
    print(f"sigma={scale:.2f}  s={y.std(ddof=1):.4f}  "
          f"T={stat:.2f}  p={pval:.3f}")
```

출력:

```text
sigma=1.00  s=1.0130  T=101.58  p=0.819
sigma=1.05  s=1.0636  T=111.99  p=0.351
sigma=1.10  s=1.1143  T=122.92  p=0.104
sigma=1.15  s=1.1649  T=134.34  p=0.021
sigma=1.20  s=1.2156  T=146.28  p=0.003
```

</div>

## 해석

- 참 분산이 가설값과 같으면($\sigma = 1.00$) $p$값이 크게 나오므로 $H_0$을 기각하지 못한다.
- 참 표준편차가 커질수록 검정통계량이 커지고 $p$값이 작아져 $H_0$에 반하는 증거가 강해진다.
- 이 검정은 정규성을 가정한다. 꼬리가 두껍거나 치우친 자료에서는 실제 제1종 오류율이 명목 수준 $\alpha$에서 크게 벗어날 수 있다.

!!! note "같은 seed를 쓰면 자료가 겹친다"
    위 반복문은 모든 `scale`에 대해 `random_state=seed`를 고정한다. 그래서 다섯 표본이 **독립이 아니라 같은 표준정규 추출값을 척도만 바꾸어 재사용한 것**이다. 표본표준편차가 $1.0130, 1.0636, 1.1143, 1.1649, 1.2156$으로 정확히 $1.0130 \times \{1.00, 1.05, 1.10, 1.15, 1.20\}$이 되는 것이 그 증거이다.

    이는 교육적으로 오히려 유리하다. 표집변동이 제거되어 **척도 변화의 효과만** 순수하게 드러나기 때문이다. 다만 이 표를 "다섯 번의 독립적인 실험"으로 읽어서는 안 된다.

    검정력을 논하려면 각 `scale`마다 독립적인 표본을 여러 개 생성해야 한다.

$n = 100$에서 표준편차가 15%만 커져도($\sigma = 1.15$) 5% 수준에서 기각한다는 점이 눈에 띈다. 15.3절에서 본 F 검정의 낮은 검정력과 대비되는데, 이는 일표본 검정이 $\sigma_0^2$을 **알려진 상수**로 취급하여 비교 대상의 불확실성이 없기 때문이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> 어떤 제조공정은 부품 지름의 분산이 $\sigma_0^2 = 0.04\;\text{mm}^2$이 되도록 설계되었다. 부품 $n = 25$개의 표본에서 $S^2 = 0.06$을 얻었다. 카이제곱 검정통계량을 계산하고 Python으로 양측 $p$값을 구하라. $\alpha = 0.05$에서 결론을 서술하라.

</div>

??? success "풀이"

    검정통계량은

    $$
    T = \frac{(25 - 1)(0.06)}{0.04} = \frac{1.44}{0.04} = 36.
    $$

    ```python
    import scipy.stats as stats

    T = 24 * 0.06 / 0.04  # 36.0
    p = 2 * min(stats.chi2(df=24).cdf(T), stats.chi2(df=24).sf(T))
    print(f"T = {T:.1f}, right tail = {stats.chi2(df=24).sf(T):.4f}, "
          f"two-sided p = {p:.4f}")
    ```

    출력:

    ```text
    T = 36.0, right tail = 0.0549, two-sided p = 0.1098
    ```

    $\chi^2(24)$에서 오른쪽 꼬리확률 $P(\chi^2 > 36) = 0.0549$이므로 양측 $p = 0.110$이다. $\alpha = 0.05$에서 $H_0$을 기각하지 못한다. 분산이 $0.04$와 다르다고 결론지을 증거가 충분하지 않다.

    **다만 단측이라면 결론이 달라질 뻔했다.** 품질관리에서는 보통 "분산이 목표보다 **큰가**"만 문제가 되므로 단측검정이 적절하다. 그 경우 $p = 0.0549$로 여전히 기각하지 못하지만 경계선에 훨씬 가깝다.

    검정 방향은 자료를 보기 전에 정해야 한다. 양측으로 계획했다가 결과를 보고 단측으로 바꾸는 것은 유의수준을 두 배로 부풀리는 조작이다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 표본분산의 정의와, $H_0$ 아래에서 각 $(X_i - \mu)/\sigma_0 \sim N(0,1)$이라는 사실에서 출발하여 검정통계량 $T = (n-1)S^2 / \sigma_0^2$을 유도하라. 자유도가 왜 $n$이 아니라 $n - 1$인지 설명하라.

</div>

??? success "풀이"

    $Z_i = (X_i - \mu)/\sigma_0$이라 쓰면 $\sum_{i=1}^n Z_i^2 \sim \chi^2(n)$이다. 그러나 $\mu$는 알려져 있지 않아 $\bar{X}$로 대체된다. 제약 $\sum (X_i - \bar{X}) = 0$이 자유도 하나를 없애므로

    $$
    \sum_{i=1}^{n} \frac{(X_i - \bar{X})^2}{\sigma_0^2} = \frac{(n-1)S^2}{\sigma_0^2} \sim \chi^2(n-1).
    $$

    이는 $n$차원 표준정규 벡터를 모든 성분이 1인 벡터에 직교하는 $(n-1)$차원 부분공간으로 사영한 것이며, Cochran 정리에 의해 $\chi^2(n-1)$ 분포를 갖는다.

    **기하학적 그림.** $\mathbf{Z} = (Z_1,\ldots,Z_n)$의 분포는 회전에 불변이다. 이 벡터를 두 성분으로 분해한다.

    - $\mathbf{1}$ 방향 성분: $\sqrt{n}\bar{Z}$, 자유도 1
    - 그에 직교하는 성분: $\sum_i (Z_i - \bar{Z})^2$, 자유도 $n-1$

    직교 분해이므로 두 성분이 독립이고 자유도가 $1 + (n-1) = n$으로 더해진다. 우리가 관심 갖는 것은 두 번째 성분이며, 그것이 $\chi^2(n-1)$이다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span> $N(0,1)$에서 크기 $n = 30$인 표본을 뽑아 $\sigma_0^2 = 1$, $\alpha = 0.05$로 카이제곱 검정을 적용하는 모의실험(5,000회 이상)을 작성하라. 경험적 제1종 오류율을 추정하고 0.05에 가까운지 확인하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    import scipy.stats as stats

    rng = np.random.default_rng(42)
    n, sigma2_0, alpha, n_sims = 30, 1.0, 0.05, 10000
    rejections = 0

    for _ in range(n_sims):
        x = rng.normal(0, 1, size=n)
        s2 = np.var(x, ddof=1)
        T = (n - 1) * s2 / sigma2_0
        p = 2 * min(stats.chi2(df=n - 1).cdf(T), stats.chi2(df=n - 1).sf(T))
        if p < alpha:
            rejections += 1

    print(f"Empirical Type I error: {rejections / n_sims:.4f}")
    ```

    출력:

    ```text
    Empirical Type I error: 0.0500
    ```

    정확히 $0.0500$이다. 정규성 아래에서 이 검정이 **정확한**(exact) 검정이므로 당연한 결과이다. 점근근사가 아니라 정확한 표집분포를 쓰기 때문에 어떤 $n$에서도 크기가 정확히 $\alpha$이다.

    이는 14장과 15장에서 본 다른 검정들(D'Agostino $K^2$, Jarque-Bera, Levene 등)이 근사에 기대어 크기가 조금씩 어긋났던 것과 대조된다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> 연습문제 3의 모의실험을 정규분포 대신 $t(5)$ 분포(꼬리가 더 두꺼움)에서 뽑아 반복하라. 경험적 기각률을 $\alpha = 0.05$와 비교하고, 카이제곱 검정이 비정규 자료에 왜 문제가 되는지 설명하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    import scipy.stats as stats

    rng = np.random.default_rng(42)
    n, sigma2_0, alpha, n_sims = 30, 5 / 3, 0.05, 10000
    # Var(t(5)) = 5/(5-2) = 5/3
    rejections = 0

    for _ in range(n_sims):
        x = stats.t(df=5).rvs(size=n, random_state=rng)
        s2 = np.var(x, ddof=1)
        T = (n - 1) * s2 / sigma2_0
        p = 2 * min(stats.chi2(df=n - 1).cdf(T), stats.chi2(df=n - 1).sf(T))
        if p < alpha:
            rejections += 1

    print(f"Empirical rejection rate (t(5)): {rejections / n_sims:.4f}")
    ```

    출력:

    ```text
    Empirical rejection rate (t(5)): 0.1725
    ```

    기각률이 $0.173$으로 명목값의 **3.5배**이다. 연습문제 3의 정확한 $0.0500$과 극명하게 대비된다.

    **원인.** 카이제곱 분산 검정은 첨도에 민감하다. 15.1절 연습문제 1에서 유도했듯

    $$
    \operatorname{Var}(S^2) \approx \frac{\sigma^4(\gamma_2 + 2)}{n}
    $$

    이고 $t_5$는 $\gamma_2 = 6$이므로 $S^2$의 실제 분산이 정규 이론값의 **4배**이다. 카이제곱 기준분포는 정규 이론값을 쓰므로 실제보다 절반의 폭을 갖고, 그래서 $T$가 임계값 밖으로 자주 벗어난다.

    **표본을 키워도 나아지지 않는다.** 팽창 인자 $(\gamma_2+2)/2 = 4$가 $n$에 의존하지 않기 때문이다. 15.3절 연습문제 4에서 F 검정에 대해 확인한 것과 같은 구조이다.

    **대안.** 붓스트랩 신뢰구간(15.6절)이나, 분산 자체보다 로버스트 산포 측도(중앙값절대편차 등)를 쓰는 방법이 있다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> 단측검정 $H_0 : \sigma^2 \le \sigma_0^2$ 대 $H_1 : \sigma^2 > \sigma_0^2$을 구성하라. 기각규칙을 $\chi^2_{1-\alpha,\, n-1}$로 표현하고, Python 함수 `chi2_test_for_variance`를 단측 $p$값을 반환하도록 수정하라.

</div>

??? success "풀이"

    오른쪽 대립가설 $H_1: \sigma^2 > \sigma_0^2$에 대해 다음이면 $H_0$을 기각한다.

    $$
    T = \frac{(n-1)S^2}{\sigma_0^2} > \chi^2_{1-\alpha,\, n-1}.
    $$

    단측 $p$값은 $p = P(\chi^2(n-1) \ge T) = 1 - F_{\chi^2}(T)$이다.

    ```python
    import numpy as np
    import scipy.stats as stats

    def chi2_test_variance(data, sigma2_0=1.0, alternative='two-sided'):
        """모분산에 대한 일표본 카이제곱 검정.

        alternative: 'two-sided', 'greater', or 'less'
        """
        n = len(data)
        s2 = np.var(data, ddof=1)
        T = (n - 1) * s2 / sigma2_0
        dist = stats.chi2(df=n - 1)
        if alternative == 'greater':
            p = dist.sf(T)
        elif alternative == 'less':
            p = dist.cdf(T)
        elif alternative == 'two-sided':
            p = 2 * min(dist.cdf(T), dist.sf(T))
        else:
            raise ValueError(f"unknown alternative: {alternative}")
        return T, p
    ```

    **왜 $H_0$을 $\sigma^2 \le \sigma_0^2$(부등호)로 쓰는가.** 복합 귀무가설이지만 검정은 경계값 $\sigma^2 = \sigma_0^2$에서 수행한다. $\sigma^2 < \sigma_0^2$이면 $T$가 더 작아져 기각확률이 더 낮아지므로, 경계에서의 크기가 $H_0$ 전체에 대한 크기의 **상한**이 된다. 이런 성질을 갖는 검정을 수준 $\alpha$ 검정이라 한다.

    **품질관리에서의 의미.** 실무에서 관심은 대체로 "변동성이 규격을 넘는가"이므로 단측이 자연스럽다. 15.2절 연습문제 3에서 다룬 상단 신뢰한계와 쌍대 관계에 있다. $\square$

---

## 정리하며

카이제곱 분산 검정의 **구현**에서 챙길 점들이다.

- **`ddof=1` 을 확인한다.** 통계량이 베셀 수정한 $S^2$ 을 쓰므로 `numpy` 기본값(`ddof=0`)을 그대로 쓰면 틀린다.
- **양측 $p$ 값은 작은 쪽 꼬리의 두 배**로 계산하되 $1$ 을 넘지 않게 자른다. 비대칭 분포이므로 관례가 필요하다.
- **`scipy` 에 전용 함수가 없다.** `chi2.sf` 와 `chi2.cdf` 로 직접 만들어야 하며, 그래서 구현 실수가 생기기 쉽다.
- **분산비 시나리오로 검증한다.** 참 분산을 $\sigma_0^2$ 의 여러 배수로 두고 기각률을 재면 검정이 의도대로 작동하는지 확인된다.
- **정규가 아닌 자료로도 돌려 본다.** 제1종 오류율이 명목값에서 얼마나 벗어나는지 직접 보는 것이 이 장의 경고를 체감하는 가장 좋은 방법이다.

다음 절 **카이제곱분포**를 정리하고 $F$ 검정으로 넘어간다.
