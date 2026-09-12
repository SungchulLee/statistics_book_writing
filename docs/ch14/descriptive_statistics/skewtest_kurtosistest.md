# 왜도 검정과 첨도 검정

## 왜도 검정

`scipy.stats.skewtest()`가 제공하는 **왜도 검정**은 자료의 왜도가 0에서 유의하게 벗어나는지, 곧 자료가 대칭인지 아닌지를 평가하는 형식적 통계검정이다.

### 가설

- **귀무가설** ($H_0$): 자료의 왜도가 0이다(대칭분포이다).
- **대립가설** ($H_1$): 자료의 왜도가 0이 아니다(비대칭이다).

왜도 검정은 검정통계량과 $p$값을 제공한다. $p$값이 작으면(보통 0.05 미만) 귀무가설을 기각하고 자료가 대칭분포가 아니라고 결론짓는다.

### 단계별 설명

1. **표본왜도 계산**: 먼저 다음 공식으로 표본왜도를 계산한다.

    $$
    \text{Skewness} = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{x_i - \bar{x}}{\sigma} \right)^3
    $$

    여기서 $n$은 자료점의 개수, $x_i$는 개별 자료점, $\bar{x}$는 평균, $\sigma$는 표준편차이다.

2. **z 점수 계산**: 검정통계량인 **z 점수**는 관측된 왜도를 그 표준오차로 나누어 계산한다. z 점수는 왜도가 기댓값(정규분포에서는 0)에서 얼마나 떨어져 있는지를 알려준다.

3. **z 점수를 p값으로 변환**: 표준정규분포의 누적분포함수로 p값을 계산한다. 앞 단계에서 얻은 z 점수를 표준정규분포와 비교하여 **양측 p값**을 얻는다. 이 p값은 왜도가 0(대칭분포)이라는 귀무가설 아래에서 관측된 값만큼 극단적인 왜도를 볼 가능성을 수량화한다.

### 응용

왜도 검정은 대칭성을 가정하는 모수적 통계 방법(특정 형태의 $t$ 검정이나 분산분석 등)으로 자료를 분석할 수 있는지 평가할 때 실용적이다. 자료가 치우쳤다고 결론지으면 자료를 변환하거나(예: 로그나 Box-Cox 변환) 대칭성을 가정하지 않는 비모수 방법을 써야 할 수 있다.

<div class="codebox" markdown>

**예제 1.** 왜도 검정

```python
import numpy as np
from scipy import stats

np.random.seed(0)

# 지금은 정규자료다. 위 주석을 바꿔 감마분포로 돌리면 검정이 기각되는 것을
# 볼 수 있다. 두 경우를 견주는 것이 이 코드의 쓰임이다.
# data = np.random.gamma(2, 2, 1000)
data = np.random.normal(2, 2, 1000)

skewness_value = stats.skew(data)
print(f"Skewness: {skewness_value:.4f}")

# 귀무가설은 "모집단의 왜도가 0" 이다. 이 검정은 치우침만 보므로,
# 대칭이면서 꼬리만 두꺼운 분포는 잡아내지 못한다.
stat, p_value = stats.skewtest(data)
print(f"Skewness Test: Statistic={stat:.4f}, p-value={p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value <= alpha:
    print("Reject H_0: The data is not symmetrically distributed (significant skewness).")
else:
    print("Fail to reject H_0: The data is symmetrically distributed (no significant skewness).")
```

출력:

```text
Skewness: 0.0339
Skewness Test: Statistic=0.4402, p-value=0.6598
Fail to reject H_0: The data is symmetrically distributed (no significant skewness).
```

</div>

정규 자료이므로 왜도가 $0.0339$로 0에 가깝고 $p = 0.66$으로 기각하지 못한다. 기대한 대로이다.

---

## 첨도 검정

`scipy.stats.kurtosistest()`가 제공하는 **첨도 검정**은 자료의 초과첨도가 정규분포의 첨도에서 유의하게 벗어나는지 평가하는 형식적 통계검정이다.

### 가설

- **귀무가설** ($H_0$): 자료의 첨도가 정규분포의 것과 같다.
- **대립가설** ($H_1$): 자료의 첨도가 정규분포의 것과 다르다.

첨도 검정은 검정통계량과 $p$값을 제공한다. $p$값이 작으면(보통 0.05 미만) 귀무가설을 기각하고 자료의 첨도가 정규가 아니라고 결론짓는다.

### 단계별 설명

1. **표본 초과첨도 계산**: 먼저 다음 공식으로 표본 초과첨도를 계산한다.

    $$
    \text{Excess Kurtosis} = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{x_i - \bar{x}}{\sigma} \right)^4 - 3
    $$

    여기서 $n$은 자료점의 개수, $x_i$는 개별 자료점, $\bar{x}$는 평균, $\sigma$는 표준편차이다.

2. **z 점수 계산**: 검정통계량인 **z 점수**는 관측된 초과첨도를 그 표준오차로 나누어 계산한다. z 점수는 초과첨도가 기댓값(정규분포에서는 0)에서 얼마나 떨어져 있는지를 알려준다.

3. **z 점수를 p값으로 변환**: 표준정규분포의 누적분포함수로 **양측 p값**을 계산한다. 이 p값은 귀무가설 아래에서 관측된 값만큼 극단적인 초과첨도를 볼 가능성을 수량화한다.

### 응용

정규 첨도를 가정하는 모수적 방법($t$ 검정, 분산분석 등)이 적절한지 평가할 때 첨도 검정을 쓴다. 첨도 검정이 자료의 꼬리가 유의하게 두껍거나 얇다고 나타내면 변환(로그나 Box-Cox 변환)이나 비모수 방법이 필요할 수 있다.

<div class="codebox" markdown>

**예제 2.** 첨도 검정

```python
import numpy as np
from scipy import stats

np.random.seed(0)

# data = np.random.gamma(2, 2, 1000)
data = np.random.normal(2, 2, 1000)

kurtosis_value = stats.kurtosis(data)
print(f"Kurtosis: {kurtosis_value:.4f}")

# 귀무가설은 "모집단의 초과첨도가 0" 이다. 앞의 왜도 검정과 짝을 이루며,
# 둘을 합친 것이 D'Agostino 의 K^2 이다.
stat, p_value = stats.kurtosistest(data)
print(f"Kurtosis Test: Statistic={stat:.4f}, p-value={p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value <= alpha:
    print("Reject H_0: The data does not have normal kurtosis.")
else:
    print("Fail to reject H_0: The data has normal kurtosis.")
```

출력:

```text
Kurtosis: -0.0468
Kurtosis Test: Statistic=-0.1980, p-value=0.8431
Fail to reject H_0: The data has normal kurtosis.
```

</div>

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`scipy.stats.skewtest`로 어떤 연구자가 z 점수 3.2와 p값 0.001을 얻었다. 이 결과를 해석하라.

</div>

??? success "풀이"
    왜도 검정은 $H_0$: 모집단 왜도가 0(정규성과 일관됨)을 검정한다. z 점수 3.2는 0에서 멀고 $p = 0.001$은 어떤 관행적 유의수준보다도 훨씬 작다.

    해석: 자료가 치우친(비대칭) 분포에서 왔다는 강한 증거가 있다. z 점수가 양수이므로 오른쪽 치우침(오른쪽 꼬리가 왼쪽보다 무겁다)을 나타낸다. 이는 비정규성의 한 요소이며, 자료에 변환이나 비모수 방법이 필요할 수 있다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`skewtest`와 단순히 표본왜도를 계산하는 것의 차이를 설명하라. 형식적 검정이 왜 필요한가?

</div>

??? success "풀이"
    표본왜도 $g_1 = \frac{m_3}{m_2^{3/2}}$는 점추정값이며, 자료가 진짜 정규여도 표집변동 때문에 언제나 0이 아니다. 검정이 없으면 관측된 왜도가 통계적으로 유의한지 단순한 표집 잡음인지 판단할 길이 없다.

    `skewtest`는 D'Agostino 변환으로 $g_1$을 z 통계량으로 바꾼다. 이 변환은 정규성 아래에서 왜도의 표집분포를 반영한다. 그러면 p값이 관측된 왜도가 귀무가설 아래에서 얼마나 있을 법하지 않은지를 수량화한다. 예를 들어 $g_1 = 0.3$은 $n = 500$에서는 유의하지만 $n = 20$에서는 유의하지 않을 수 있다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
`kurtosistest`에는 최소 표본크기 요건이 있다. 작은 표본이 첨도 추정에 문제가 되는 이유를 설명하라.

</div>

??? success "풀이"
    표본첨도는 편차의 4제곱 $(x_i - \bar{x})^4$을 쓰므로 개별 관측값에 극도로 민감하다. 작은 표본에서는

    1. **큰 분산:** 첨도 추정량의 표집분포가 매우 넓어 추정을 믿을 수 없다.
    2. **편향:** 작은 표본에서 표본첨도가 편향되어 있고 편향 보정 공식도 불확실성이 크다.
    3. **추정량의 비정규성:** `kurtosistest`가 쓰는 z 변환은 $n$이 근사적으로 표준정규가 될 만큼 커야 한다는 것을 가정한다. $n \approx 20$ 아래에서는 이 근사가 무너진다.

    scipy에서 `kurtosistest`는 $n < 20$이면 경고를 내고 $n < 5$이면 아예 오류를 낸다. 참고로 `skewtest`는 $n < 8$이면 오류를 낸다.

    작은 표본에서는 적률 기반 검정보다 시각적 방법(Q-Q 그림)과 Shapiro-Wilk 검정이 더 믿을 만하다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
`skewtest`가 $p = 0.15$를, `kurtosistest`가 $p = 0.03$을 준다면 비정규성의 성격에 대해 무엇을 결론지을 수 있는가?

</div>

??? success "풀이"
    왜도 검정은 기각하지 않으므로($p = 0.15$) 자료가 근사적으로 대칭임을 시사한다. 첨도 검정은 기각하므로($p = 0.03$) 자료의 꼬리 거동이 비정상임을(꼬리가 두껍거나 얇음을) 나타낸다.

    이 패턴은 분포가 **대칭이지만 고첨**(두꺼운 꼬리)이거나 **대칭이지만 저첨**(얇은 꼬리)임을 시사한다. 예로는 자유도가 작은 $t$ 분포(대칭, 두꺼운 꼬리)나 균등분포(대칭, 얇은 꼬리)가 있다.

    실질적 함의는 방향에 달려 있다. 두꺼운 꼬리(양의 초과첨도)는 기대보다 이상점이 많다는 뜻으로 평균 기반 추론에 영향을 준다. 얇은 꼬리(음의 초과첨도)는 표준적인 방법에 대체로 덜 문제가 된다.

---

## 정리하며

왜도와 첨도를 **각각 형식적으로 검정**할 수 있다.

- **$H_0$ 이 각각 $\gamma_1=0$, $\gamma_2=0$ 이다.** 표본 통계량을 표준화해 근사적으로 $N(0,1)$ 이 되도록 변환한다.
- **표적 검정이라는 것이 장점이다.** 어느 방향으로 이탈했는지 알려 주므로, 옴니버스 검정이 기각했을 때 **원인을 짚는 데 쓴다.**
- **최소 표본크기 요건이 있다.** `scipy` 의 `skewtest` 는 $n\ge8$, `kurtosistest` 는 $n\ge20$ 을 요구하며, 그보다 작으면 근사가 성립하지 않는다.
- **두 검정을 따로 하면 다중검정 문제가 생긴다.** 그래서 둘을 합친 다고스티노 $K^2$ 나 자크–베라를 쓴다.
- **표본이 크면 사소한 이탈도 기각한다.** 다른 모든 정규성 검정과 같은 문제이며, 통계량의 **크기**를 함께 보아야 한다.

다음 절 **D'Agostino $K^2$ 검정**으로 넘어간다.
