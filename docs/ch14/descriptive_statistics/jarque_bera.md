# Jarque-Bera 검정

## 개요

Jarque-Bera 검정은 자료의 두 핵심 특징인 왜도와 첨도를 평가하여 자료가 정규분포를 따르는지 판정하는 통계검정이다. 정규성이 많은 통계 모형과 방법의 결정적 가정인 계량경제학과 금융 응용에서 널리 쓰인다.

### 가설

- **귀무가설** ($H_0$): 자료가 정규분포를 따른다.
- **대립가설** ($H_1$): 자료가 정규분포를 따르지 않는다.

## Jarque-Bera 검정통계량의 계산

검정통계량 $JB$는 다음 공식으로 계산한다.

$$
JB = \frac{n}{6} \left( S^2 + \frac{(K - 3)^2}{4} \right)
$$

여기서

- $n$은 표본크기,
- $S$는 표본왜도,
- $K$는 표본첨도,
- 분모의 6은 왜도와 첨도의 기여를 표준화하는 척도 상수이다.

### 검정통계량 계산 단계

1. **표본평균 계산**:

    $$
    \bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i
    $$

2. **표본왜도 $S$ 계산**:

    $$
    S = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{X_i - \bar{X}}{\sigma} \right)^3
    $$

    여기서 $\sigma^2$은 표본분산이다.

    $$
    \sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (X_i - \bar{X})^2
    $$

3. **표본첨도 $K$ 계산**:

    $$
    K = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{X_i - \bar{X}}{\sigma} \right)^4
    $$

4. **Jarque-Bera 통계량 $JB$ 계산**:

    $$
    JB = \frac{n}{6} \left( S^2 + \frac{(K - 3)^2}{4} \right)
    $$

    이 통계량은 왜도의 제곱과 첨도가 3(정규분포의 첨도)에서 벗어난 정도의 제곱을 표본크기 $n$으로 가중하여 결합한다.

!!! note "적률에는 $1/n$을 쓴다"
    위의 $\sigma^2$은 $1/(n-1)$이 아니라 $1/n$로 나눈 편향된 표본분산이다. `scipy.stats.skew`, `scipy.stats.kurtosis`, `scipy.stats.jarque_bera`가 모두 이 규약을 쓰므로, 손으로 계산한 값과 라이브러리 값을 맞추려면 이 규약을 따라야 한다.

## p값 유도

Jarque-Bera 검정통계량 $JB$는 귀무가설 아래에서 자유도 2인 카이제곱($\chi^2$) 분포를 따른다(왜도와 첨도라는 두 성분으로 계산하기 때문이다). $p$값을 얻으려면

1. 계산한 $JB$ 통계량을 자유도 2인 카이제곱분포의 임계값과 비교한다.

2. **$p$값**은 귀무가설($H_0$)이 참이라는 가정 아래에서 검정통계량 $JB$가 관측값만큼 또는 그보다 더 극단적일 확률이다.

    - $p$값이 작으면(보통 유의수준 $\alpha = 0.05$ 미만) 자료가 정규성에서 유의하게 벗어남을 시사하며 귀무가설을 기각한다.
    - $p$값이 크면 귀무가설을 기각할 증거가 부족하다는 뜻이며, 자료가 정규분포에서 왔을 수 있다.

### 판정 규칙

- $p$값이 유의수준 $\alpha$(예: 0.05)보다 작으면 $H_0$을 기각하고 자료가 정규분포를 따르지 않는다고 결론짓는다.
- $p$값이 $\alpha$ 이상이면 $H_0$을 기각하지 못하며 자료가 정규분포를 따를 수 있다고 결론짓는다.

### 해석

- **$JB$ 통계량이 크면** 자료의 왜도나 첨도(또는 둘 다)가 정규분포에서 유의하게 벗어났다는 뜻이다.
- **$JB$ 통계량이 작으면** 표본자료의 왜도와 첨도가 정규분포와 일관된다는 뜻이다.

## Python 구현

```python
import numpy as np
from scipy import stats

np.random.seed(0)

n = 1000

# Generate a sample dataset
data = np.random.normal(0, 1, n)
# data = np.random.exponential(1, n)

skewness_value = stats.skew(data)
kurtosis_value = stats.kurtosis(data)
JB = n / 6 * (skewness_value**2 + kurtosis_value**2 / 4)
print(f"{JB = }")

# Perform Jarque-Bera test
stat, p_value = stats.jarque_bera(data)
print(f"Jarque-Bera Test: Statistic={stat}, p-value={p_value}")

# Interpretation
alpha = 0.05
if p_value <= alpha:
    print("Reject H_0: The data is not normally distributed.")
else:
    print("Fail to reject H_0: The data is normally distributed.")
```

출력:

```text
JB = 0.28220016508625234
Jarque-Bera Test: Statistic=0.28220016508625234, p-value=0.8684023954281485
Fail to reject H_0: The data is normally distributed.
```

손으로 계산한 값이 `scipy.stats.jarque_bera`와 정확히 일치한다. `stats.kurtosis`가 이미 **초과**첨도 $K-3$을 돌려주므로 코드에서 `kurtosis_value**2 / 4`가 공식의 $(K-3)^2/4$에 해당한다는 점에 유의하라.

---

## Jarque-Bera 검정과 D'Agostino의 K 제곱 검정

Jarque-Bera 검정은 D'Agostino의 $K^2$ 검정의 근사가 **아니다**. 두 검정 모두 왜도와 첨도를 살펴 정규성을 평가하지만 검정통계량을 계산하는 방식이 근본적으로 다르다.

### 핵심 차이

**Jarque-Bera 검정**:

- 표본의 **왜도**와 **첨도**를 직접 써서 검정통계량을 계산한다.
- 검정통계량은

    $$
    JB = \frac{n}{6} \left( S^2 + \frac{(K - 3)^2}{4} \right)
    $$

- 왜도와 첨도를 하나의 검정통계량으로 결합하며, 귀무가설 아래에서 자유도 2인 카이제곱분포를 따른다고 가정한다.

**D'Agostino의 $K^2$ 검정**:

- 왜도와 첨도를 각각 독립적인 **Z 점수**로 변환한다.
    - **$Z_{\text{skewness}}$**: 표본왜도를 정규화하는 변환.
    - **$Z_{\text{kurtosis}}$**: 표본첨도를 정규화하는 변환.
- 검정통계량은

    $$
    K^2 = Z_{\text{skewness}}^2 + Z_{\text{kurtosis}}^2
    $$

- Jarque-Bera와 마찬가지로 $K^2$도 자유도 2인 카이제곱분포를 따르지만, D'Agostino 검정은 왜도와 첨도에 각각 별도의 변환을 적용하므로 정규성 이탈에 더 로버스트하고 민감하다.

### 왜 서로 다른가

- **접근 방식의 차이**: Jarque-Bera 검정은 원래의 왜도와 첨도 값에 기초한 단순하고 직접적인 공식을 쓰는 반면, D'Agostino의 $K^2$ 검정은 표본크기를 반영하고 왜도·첨도의 분포를 정규화하는 변환을 적용한다.

- **검정통계량의 구성**: Jarque-Bera는 왜도와 첨도를 곧바로 하나의 통계량으로 결합하지만, D'Agostino 검정은 둘을 분리하여 각각을 개별 검정통계량으로 변환한 뒤 제곱하여 더한다.

- **민감도**: D'Agostino의 $K^2$ 검정이 특히 표본이 클 때 정규성 이탈에 더 민감하다. Z 변환 덕분에 표본크기가 커질수록 정확해지는 반면, Jarque-Bera는 작은 표본에서 성능이 나빠지거나 꼬리 이탈에 덜 민감할 수 있다.

### 결론

Jarque-Bera 검정은 D'Agostino의 $K^2$ 검정을 근사하는 것이 아니다. 둘은 서로 다른 통계적 토대를 가진 별개의 정규성 검정 방법이다. 두 검정 모두 왜도와 첨도를 쓰지만, D'Agostino 검정은 변환 기반 접근 덕분에 더 로버스트한 것으로 평가되고 Jarque-Bera는 더 단순하며 계량경제학에서 흔히 쓰인다.

## 연습문제

**연습문제 1.**
Jarque-Bera 검정으로 자료의 정규성을 검정하는 통계량 $JB$는 다음과 같이 정의된다.

$$
JB = \frac{n}{6} \left( S^2 + \frac{(K - 3)^2}{4} \right) \sim \chi^2_2
$$

여기서

- $n$: 표본크기
- $S$: 표본왜도
- $K$: 표본첨도
- 분모의 $6$은 왜도와 첨도의 기여를 표준화하는 척도 상수이다.

**(a)** 정규분포에서 $S$의 이론값을 구하라.

**(b)** 적분 $\int_{-\infty}^\infty x^4 e^{-x^2} dx$를 계산하라.

**(c)** $JB$를 계산할 때 왜 $K$에서 $3$을 빼는가?

**(d)** $JB = 3.2189$일 때 $\chi^2_2$의 누적분포함수 $F$로 $p$값을 표현하라.

**(e)** $p$값이 0.2이고 유의수준($\alpha$)이 5%라면 검정의 결론은 무엇인가?

??? success "풀이"

    **(a)** 정규분포에서 이론적 왜도 $S$는

    $$
    S = \mathbb{E}\left(\frac{X - \mu}{\sigma}\right)^3 = \mathbb{E}Z^3 = \int_{-\infty}^\infty x^3 \cdot \frac{1}{\sqrt{2\pi}} e^{-x^2/2} dx = 0
    $$

    피적분함수가 기함수이므로 적분이 0이다.

    **(b)**

    $$
    I
    = \int_{-\infty}^\infty x^4 e^{-x^2} dx
    = 2 \int_0^\infty x^4 e^{-x^2} dx
    $$

    $u = x^2$로 두면 $x = u^{1/2}$, $dx = \frac{1}{2} u^{-1/2} du$이므로 적분은

    $$
    I
    = 2 \int_0^\infty (u^{1/2})^4 e^{-u} \cdot \frac{1}{2} u^{-1/2} du
    = \int_0^\infty u^{\frac{5}{2}-1} e^{-u} du
    = \Gamma\left(\frac{5}{2}\right)
    = \frac{3}{2}\cdot\Gamma\left(\frac{3}{2}\right)
    = \frac{3}{2}\cdot\frac{1}{2}\cdot\Gamma\left(\frac{1}{2}\right)
    = \frac{3}{4}\sqrt{\pi} \approx 1.3293
    $$

    여기서 감마함수는 다음으로 정의된다.

    $$
    \Gamma(n) = \int_0^\infty x^{n-1} e^{-x} dx
    $$

    감마함수는 다음을 만족한다.

    $$
    \Gamma(n+1) = n \cdot \Gamma(n),\quad
    \Gamma\left(\frac{1}{2}\right)=\sqrt{\pi}
    $$

    **(c)** 정규분포의 이론적 첨도 $K$는 $3$이다. $3$을 빼면 정규 자료에서 $K - 3 = 0$이 된다. 그러면 정규성이라는 귀무가설 아래에서 검정통계량이 간단해진다.

    **(d)** $p$값은

    $$
    p = 1 - F(3.2189)
    $$

    $\chi^2_2$의 누적분포함수가 $F(x) = 1 - e^{-x/2}$이므로 $p = e^{-3.2189/2} = e^{-1.60945} = 0.2000$이다.

    **(e)** $p = 0.2 > \alpha = 0.05$이므로 귀무가설을 기각할 증거가 부족하다. 따라서 주어진 유의수준에서 자료가 정규성을 위배하지 않으며 정규성 가정을 유지한다고 결론짓는다.
