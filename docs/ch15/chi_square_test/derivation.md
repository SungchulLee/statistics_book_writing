# 유도와 분포이론

모분산에 대한 카이제곱 검정은 하나의 분포이론적 결과 위에 서 있다. 자료가 정규분포를 따를 때 척도가 조정된 표본분산이 카이제곱분포를 따른다는 것이다. 이 절은 표준정규 변수와 카이제곱족의 연결에서 출발하여 그 결과를 처음부터 유도한다.

## 표준정규 제곱의 합

5장에서 보았듯 $Z_1, Z_2, \ldots, Z_n$이 독립인 표준정규 확률변수이면 그 제곱합은 자유도 $n$인 카이제곱분포를 따른다.

$$
\sum_{i=1}^{n} Z_i^2 \sim \chi^2_n
$$

이것이 카이제곱분포의 정의적 성질이다. 각 $Z_i^2$이 자유도 1을 기여하고, 독립성 덕분에 자유도가 더해진다.

## 관측값의 표준화

$X_1, X_2, \ldots, X_n$이 독립이고 동일하게 $X_i \sim N(\mu, \sigma^2)$을 따른다고 하자. 표준화된 관측값을

$$
Z_i = \frac{X_i - \mu}{\sigma}
$$

로 정의하면 각 $Z_i$가 독립적으로 $\mathcal{N}(0, 1)$을 따른다. 그러면

$$
\sum_{i=1}^{n} \left(\frac{X_i - \mu}{\sigma}\right)^2 = \sum_{i=1}^{n} Z_i^2 \sim \chi^2_n
$$

이 합은 독립인 표준정규 제곱 $n$개로 이루어지고 추정된 모수가 없으므로 자유도가 $n$이다.

## 모평균을 표본평균으로 바꾸기

실무에서 $\mu$는 알려져 있지 않으므로 표본평균 $\bar{X} = \frac{1}{n}\sum_{i=1}^{n} X_i$로 대체한다. 다음 분해를 생각하자.

$$
\sum_{i=1}^{n} (X_i - \mu)^2 = \sum_{i=1}^{n} (X_i - \bar{X})^2 + n(\bar{X} - \mu)^2
$$

양변을 $\sigma^2$으로 나누면

$$
\sum_{i=1}^{n} \left(\frac{X_i - \mu}{\sigma}\right)^2 = \sum_{i=1}^{n} \left(\frac{X_i - \bar{X}}{\sigma}\right)^2 + \left(\frac{\bar{X} - \mu}{\sigma / \sqrt{n}}\right)^2
$$

좌변은 $\chi^2_n$이다. 우변의 마지막 항은 표준정규 변수의 제곱이므로($\bar{X} \sim N(\mu, \sigma^2/n)$이기 때문) $\chi^2_1$이다. Cochran 정리에 의해 우변의 두 항은 독립이며, 따라서

$$
\sum_{i=1}^{n} \left(\frac{X_i - \bar{X}}{\sigma}\right)^2 \sim \chi^2_{n-1}
$$

자유도가 하나 줄어드는 것은 제약 $\sum_{i=1}^{n}(X_i - \bar{X}) = 0$을 반영하며, 이 제약이 $n$개의 제곱항에서 자유로운 차원 하나를 없앤다.

## 핵심 결과

표본분산은 다음과 같이 정의된다.

$$
S^2 = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})^2
$$

양변에 $(n-1)/\sigma^2$을 곱하면

$$
\frac{(n-1)S^2}{\sigma^2} = \sum_{i=1}^{n} \left(\frac{X_i - \bar{X}}{\sigma}\right)^2 \sim \chi^2_{n-1}
$$

이것이 $\sigma^2$에 관한 추론의 추축량(pivotal quantity)이다. $H_0\colon \sigma^2 = \sigma_0^2$ 아래에서 검정통계량은

$$
\chi^2 = \frac{(n-1)S^2}{\sigma_0^2} \sim \chi^2_{n-1}
$$

!!! note "Cochran 정리의 역할"
    Cochran 정리는 $\sum(X_i - \bar{X})^2/\sigma^2$과 $(\bar{X} - \mu)^2/(\sigma^2/n)$이 독립인 카이제곱 변수이고 그 자유도의 합이 $n$임을 보장한다. 이 독립성이 없으면 위 분해로부터 표본분산의 정확한 카이제곱분포를 얻을 수 없다.

## 카이제곱분포의 성질

자유도 $\nu$인 카이제곱분포는 분산 검정과 관련된 몇 가지 성질을 갖는다.

- **평균:** $E[\chi^2_\nu] = \nu$
- **분산:** $\operatorname{Var}(\chi^2_\nu) = 2\nu$
- **왜도:** $\sqrt{8/\nu}$, $\nu$가 커질수록 줄어든다
- **최빈값:** $\nu \ge 2$일 때 $\nu - 2$

카이제곱분포가 오른쪽으로 치우쳐 있으므로 $\sigma^2$의 신뢰구간은 비대칭이다. 자유도가 크면 중심극한정리에 의해 분포가 근사적으로 정규가 된다.

## 예제

정규 모집단에서 $n = 21$개의 확률표본을 뽑았고 표본분산이 $S^2 = 18.5$였다고 하자. 유의수준 $\alpha = 0.05$에서 $H_0\colon \sigma^2 = 15$를 $H_1\colon \sigma^2 \neq 15$에 대해 검정하자.

**1단계.** 검정통계량을 계산한다.

$$
\chi^2 = \frac{(21 - 1)(18.5)}{15} = \frac{20 \times 18.5}{15} = 24.667
$$

**2단계.** $\alpha = 0.05$(양측)에서 $\chi^2_{20}$의 임계값을 구한다.

- 하단 임계값: $\chi^2_{0.975, 20} = 9.591$
- 상단 임계값: $\chi^2_{0.025, 20} = 34.170$

**3단계.** $9.591 < 24.667 < 34.170$이므로 검정통계량이 채택역 안에 있다. $H_0$을 기각하지 못한다.

유의수준 5%에서 모분산이 15와 다르다고 결론지을 만한 증거가 충분하지 않다($p = 0.429$).

## Python 구현

```python
import numpy as np
from scipy import stats

# Sample data
n = 21
s_squared = 18.5
sigma_0_squared = 15
alpha = 0.05

# Test statistic
chi2_stat = (n - 1) * s_squared / sigma_0_squared

# Critical values (two-tailed)
df = n - 1
chi2_lower = stats.chi2.ppf(alpha / 2, df)
chi2_upper = stats.chi2.ppf(1 - alpha / 2, df)

# p-value (two-tailed)
p_value = 2 * min(stats.chi2.cdf(chi2_stat, df), stats.chi2.sf(chi2_stat, df))

print(f"Test statistic: {chi2_stat:.3f}")
print(f"Critical values: [{chi2_lower:.3f}, {chi2_upper:.3f}]")
print(f"P-value: {p_value:.4f}")

if p_value < alpha:
    print("Reject H0: variance differs from the hypothesized value.")
else:
    print("Fail to reject H0: insufficient evidence of a difference.")
```

출력:

```text
Test statistic: 24.667
Critical values: [9.591, 34.170]
P-value: 0.4290
Fail to reject H0: insufficient evidence of a difference.
```


## 연습문제

**연습문제 1.**
유도의 출발점이 되는 항등식

$$
\sum_{i=1}^{n} (X_i - \mu)^2 = \sum_{i=1}^{n} (X_i - \bar{X})^2 + n(\bar{X} - \mu)^2
$$

을 증명하라.

??? success "연습문제 1 풀이"
    $X_i - \mu = (X_i - \bar{X}) + (\bar{X} - \mu)$로 쓰고 제곱하여 합한다.

    $$
    \sum_i (X_i - \mu)^2 = \sum_i (X_i - \bar{X})^2 + 2(\bar{X} - \mu)\sum_i (X_i - \bar{X}) + \sum_i (\bar{X} - \mu)^2.
    $$

    두 번째 항에서 $(\bar{X} - \mu)$는 $i$에 의존하지 않으므로 합 밖으로 빼냈다. 그런데

    $$
    \sum_i (X_i - \bar{X}) = \sum_i X_i - n\bar{X} = n\bar{X} - n\bar{X} = 0
    $$

    이므로 교차항이 **사라진다**. 세 번째 항은 $i$에 무관한 값을 $n$번 더한 것이므로 $n(\bar{X}-\mu)^2$이다. 따라서

    $$
    \sum_i (X_i - \mu)^2 = \sum_i (X_i - \bar{X})^2 + n(\bar{X} - \mu)^2.
    $$

    이 항등식은 **평행축 정리** 또는 **편차제곱합 분해**로 불린다. 교차항이 0이 되는 것은 $\bar{X}$가 $\sum_i (X_i - c)^2$을 최소화하는 $c$라는 사실의 다른 표현이며, 이 때문에 $\bar{X}$가 최소제곱 추정량이 된다.

    부수적 결과로 $\mu \neq \bar{X}$인 한 $\sum_i (X_i - \bar{X})^2 < \sum_i (X_i - \mu)^2$이다. 표본평균으로 계산한 제곱합이 참 평균으로 계산한 것보다 항상 작으며, 이것이 $n$이 아니라 $n-1$로 나누어 편향을 보정하는 이유이다. $\square$

---

**연습문제 2.**
정규 자료에서 $\bar{X}$와 $S^2$이 독립임을 보여라. 이 독립성이 정규분포에만 특유한 성질임을 설명하라.

??? success "연습문제 2 풀이"
    **정규성 아래의 증명 개요.** $Z_i = (X_i - \mu)/\sigma$라 하고 벡터 $\mathbf{Z} = (Z_1,\ldots,Z_n)^T \sim \mathcal{N}(\mathbf{0}, I_n)$을 생각한다. 첫 행이 $\frac{1}{\sqrt{n}}(1,1,\ldots,1)$인 직교행렬 $Q$를 잡고 $\mathbf{Y} = Q\mathbf{Z}$라 하자.

    직교변환은 표준정규 벡터의 분포를 보존하므로 $\mathbf{Y} \sim \mathcal{N}(\mathbf{0}, I_n)$이고 성분들이 서로 독립이다. 그런데

    $$
    Y_1 = \frac{1}{\sqrt{n}}\sum_i Z_i = \sqrt{n}\,\frac{\bar{X}-\mu}{\sigma}, \qquad \sum_{j=2}^n Y_j^2 = \|\mathbf{Z}\|^2 - Y_1^2 = \frac{(n-1)S^2}{\sigma^2}.
    $$

    $Y_1$과 $(Y_2,\ldots,Y_n)$이 독립이므로 $\bar{X}$와 $S^2$도 독립이다. 동시에 $\sum_{j=2}^n Y_j^2$은 독립인 표준정규 제곱 $n-1$개의 합이므로 $\chi^2_{n-1}$을 따른다. 이것이 본문 결과의 엄밀한 증명이다.

    **왜 정규분포에만 특유한가.** 증명에서 결정적으로 쓴 성질은 **표준정규 벡터의 분포가 직교변환에 불변**이라는 것이다. 이 회전불변성을 갖는 유일한 독립 성분 분포가 정규분포이다(Maxwell의 정리).

    실제로 Lukacs의 정리는 더 강한 결과를 준다. $X_1,\ldots,X_n$이 i.i.d.일 때 $\bar{X}$와 $S^2$이 독립일 **필요충분조건**이 바탕 분포가 정규라는 것이다.

    **실무적 함의.** 비정규 자료에서는 $\bar{X}$와 $S^2$이 상관되어 있다. 예컨대 오른쪽으로 치우친 분포에서는 큰 $\bar{X}$가 큰 $S^2$과 함께 나타나는 경향이 있다. 이 상관이 $t$ 통계량의 분포를 왜곡하는 한 원인이며, 분산 검정에서는 카이제곱 근사를 무너뜨린다. $\square$

---

**연습문제 3.**
$\chi^2_\nu$의 왜도가 $\sqrt{8/\nu}$임을 이용하여, 왜 $\sigma^2$의 신뢰구간이 비대칭이며 $\nu$가 커질수록 대칭에 가까워지는지 설명하라. $\nu = 5, 20, 50, 200$에서 정규근사와 Wilson-Hilferty 근사의 정확도를 비교하라.

??? success "연습문제 3 풀이"
    **비대칭성의 근원.** $\chi^2_\nu$의 왜도는 $\sqrt{8/\nu}$이다.

    | $\nu$ | 5 | 20 | 50 | 200 |
    |---|---|---|---|---|
    | 왜도 | 1.265 | 0.632 | 0.400 | 0.200 |

    작은 자유도에서 분포가 강하게 오른쪽으로 치우쳐 있으므로 상단 분위수가 평균 $\nu$에서 하단 분위수보다 훨씬 멀리 떨어져 있다. 신뢰구간을 만들 때 $\sigma^2$을 이 분위수로 **나누므로** 결과 구간의 상한이 하한보다 점추정값에서 멀어진다.

    $\nu \to \infty$이면 왜도가 $0$으로 가고 분포가 정규에 접근하여 구간이 대칭에 가까워진다.

    **근사의 정확도.** 두 가지 근사를 비교하자.

    - **단순 정규근사:** $\chi^2_\nu \approx \nu + z\sqrt{2\nu}$
    - **Wilson-Hilferty 근사:** $\chi^2_\nu \approx \nu\left(1 - \frac{2}{9\nu} + z\sqrt{\frac{2}{9\nu}}\right)^3$

    ```python
    import numpy as np
    from scipy import stats

    z = 1.96
    print(f"{'df':>5} {'exact':>10} {'normal':>10} {'Wilson-H':>10}")
    for df in [5, 20, 50, 200]:
        exact = stats.chi2.ppf(0.975, df)
        norm_ap = df + z * np.sqrt(2 * df)
        wh = df * (1 - 2 / (9 * df) + z * np.sqrt(2 / (9 * df)))**3
        print(f"{df:>5} {exact:>10.3f} {norm_ap:>10.3f} {wh:>10.3f}")
    ```

    출력:

    ```text
       df      exact     normal   Wilson-H
        5     12.833     11.198     12.822
       20     34.170     32.396     34.172
       50     71.420     69.600     71.424
      200    241.058    239.200    241.061
    ```

    단순 정규근사는 $\nu = 5$에서 12.8을 11.2로 추정하여 **13% 오차**를 낸다. $\nu = 200$에서도 여전히 0.8% 부족하다. 근사가 치우침을 무시하므로 상단 분위수를 체계적으로 과소추정한다.

    Wilson-Hilferty 근사는 $\nu = 5$에서 이미 소수 둘째 자리까지 맞고 $\nu \geq 20$에서는 사실상 정확하다. $(\chi^2_\nu/\nu)^{1/3}$이 거의 정규라는 사실에 기반하며, 세제곱근 변환이 치우침을 제거한다. Box-Cox 계열의 변환이 왜 유용한지를 보여주는 좋은 예이다. $\square$

---

**연습문제 4.**
Cochran 정리 없이도 $\sum(X_i - \bar{X})^2/\sigma^2$의 **평균**은 $n-1$임을 정규성 가정 없이 보일 수 있다. 이를 증명하고, 왜 평균만으로는 카이제곱 검정을 정당화할 수 없는지 설명하라.

??? success "연습문제 4 풀이"
    **평균의 계산(정규성 불필요).** $X_i$가 i.i.d.이고 $\mathbb{E}[X_i] = \mu$, $\operatorname{Var}(X_i) = \sigma^2$이라고만 가정하자. 연습문제 1의 항등식에 기댓값을 취하면

    $$
    \mathbb{E}\left[\sum_i (X_i - \bar{X})^2\right] = \mathbb{E}\left[\sum_i (X_i - \mu)^2\right] - n\,\mathbb{E}\left[(\bar{X} - \mu)^2\right] = n\sigma^2 - n \cdot \frac{\sigma^2}{n} = (n-1)\sigma^2.
    $$

    따라서

    $$
    \mathbb{E}\left[\frac{(n-1)S^2}{\sigma^2}\right] = n - 1
    $$

    이며, 이는 $\chi^2_{n-1}$의 평균과 일치한다. 어떤 분포에서든 성립한다. 동시에 $\mathbb{E}[S^2] = \sigma^2$, 곧 $S^2$이 $\sigma^2$의 불편추정량임을 보여준다.

    **평균만으로는 부족한 이유.** 검정을 하려면 통계량의 **전체 분포**가 필요하지 평균만으로는 안 된다. 두 분포가 같은 평균을 가져도 임계값이 전혀 다를 수 있다.

    구체적으로 15.1절에서 보았듯

    $$
    \operatorname{Var}\left(\frac{(n-1)S^2}{\sigma^2}\right) \approx (n-1)(\gamma_2 + 2)
    $$

    인데 $\chi^2_{n-1}$의 분산은 $2(n-1)$이다. 두 값이 일치하는 것은 $\gamma_2 = 0$, 곧 정규분포일 때뿐이다.

    곧 **비정규 자료에서도 검정통계량의 평균은 맞지만 산포가 틀린다.** 기준분포의 중심은 제자리인데 폭이 좁으므로, 실제 통계량이 임계값 밖으로 자주 벗어나 제1종 오류가 팽창한다.

    이것이 Cochran 정리가 필요한 이유이다. 정규성 아래에서만 1차 적률뿐 아니라 **분포 전체**가 정확히 카이제곱이 된다. $\square$
