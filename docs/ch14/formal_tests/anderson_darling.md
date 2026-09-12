# Anderson-Darling 검정

## 개요

**Anderson-Darling 검정**은 Kolmogorov-Smirnov 검정을 개선한 것으로, 표본이 정규분포 같은 특정 분포에서 왔는지 평가하도록 설계되었다. 꼬리에서의 이탈에 특히 민감하여 작은 표본에서 정규성 이탈을 탐지하는 데 특히 유용하다.

### 가설

- **귀무가설** ($H_0$): 자료가 정규분포를 따른다.
- **대립가설** ($H_1$): 자료가 정규분포를 따르지 않는다.

## Anderson-Darling 검정통계량의 계산

Anderson-Darling 검정통계량 $A^2$은 다음 단계로 계산한다.

1. **자료 정렬**: 표본자료 $X_1, X_2, \dots, X_n$을 오름차순으로 정렬하여 $X_{(1)} \leq X_{(2)} \leq \dots \leq X_{(n)}$을 얻는다.

2. **자료 표준화**: 각 자료점을 평균 0, 분산 1이 되도록 표준화한다. 이 변환이 자료에 정규성을 강제하는 것은 아니라는 점에 유의하라. 각 자료점 $X_i$에 대해 표준화 값 $Z_i$를 계산한다.

    $$
    Z_i = \frac{X_i - \mu}{\sigma}
    $$

    여기서 $\mu$는 표본평균, $\sigma$는 표본표준편차이다.

3. **경험분포함수 계산**: 정렬된 각 표준화 값 $Z_{(i)}$에 대해 정규분포의 누적분포함수 $F(Z_{(i)})$를 계산한다.

4. **검정통계량 $A^2$ 계산**:

    $$
    A^2 = -n - \frac{1}{n} \sum_{i=1}^{n} \left[ (2i-1) \left( \ln(F(Z_{(i)})) + \ln(1 - F(Z_{(n+1-i)})) \right) \right]
    $$

    여기서 $n$은 표본크기이고 $F(Z_{(i)})$는 정렬된 각 표준화 자료점 $Z_{(i)}$에서의 정규분포 누적분포함수이다.

    이 공식은 아래쪽과 위쪽 꼬리를 함께 반영하여 분포 꼬리에서의 이탈에 검정이 더 민감해지게 한다.

## p값 유도

검정통계량 $A^2$을 계산한 뒤에는 대상 분포 유형과 표본크기에 맞는 Anderson-Darling 분포의 임계값과 비교한다. 임계값은 유의수준 $\alpha$(예: 0.01, 0.05, 0.10)에 따라 선택한다.

- 주어진 $\alpha$의 임계값보다 $A^2$이 크면 $p$값이 $\alpha$ 아래가 되어 귀무가설을 기각한다.
- $A^2$이 임계값보다 작으면 $p$값이 $\alpha$ 위가 되어 귀무가설을 기각할 증거가 부족하다.

### 판정 규칙

- 유의수준 $\alpha$에서 $A^2 > \text{임계값}$이면 $H_0$을 기각한다(자료가 정규분포를 따르지 않는다).
- $A^2 < \text{임계값}$이면 $H_0$을 기각하지 못한다(자료가 정규분포를 따를 수 있다).

## `stats.anderson`을 이용한 Python 구현

<div class="codebox" markdown>

**예제 1.** 기각값 표로 판정하기

```python
import numpy as np
from scipy import stats

np.random.seed(0)

# 평균과 표준편차를 바꿔도 결론은 같아야 한다. 정규성 검정은 위치와
# 척도가 아니라 모양을 묻기 때문이다.
# data = np.random.normal(0, 1, 1000)
data = np.random.normal(1, 10, 1000)

# scipy 의 anderson 은 p-값 대신 유의수준별 기각값을 돌려준다.
# 통계량이 기각값보다 크면 기각이다.
result = stats.anderson(data)
statistic = result.statistic
print(f"Anderson-Darling Test: Statistic={statistic}")

# 기각값은 유의수준이 낮아질수록 커진다. 통계량 하나로 여러 수준의
# 판정을 한꺼번에 읽을 수 있다.
for significance_level, critical_value in zip(result.significance_level, result.critical_values):
    if statistic >= critical_value:
        print(f"At {significance_level}% significance level: Reject H_0. The data is not normally distributed.")
    else:
        print(f"At {significance_level}% significance level: Fail to reject H_0. The data is normally distributed.")
```

출력:

```text
Anderson-Darling Test: Statistic=0.24321791746319832
At 15.0% significance level: Fail to reject H_0. The data is normally distributed.
At 10.0% significance level: Fail to reject H_0. The data is normally distributed.
At 5.0% significance level: Fail to reject H_0. The data is normally distributed.
At 2.5% significance level: Fail to reject H_0. The data is normally distributed.
At 1.0% significance level: Fail to reject H_0. The data is normally distributed.
```

</div>

임계값은 $[0.574, 0.653, 0.784, 0.914, 1.088]$이며 $A^2 = 0.243$은 그 가운데 가장 작은 값보다도 작다. 어떤 유의수준에서도 정규성을 기각하지 않는다.

---

## `stats.anderson`에서 p값을 얻을 수 있는가

SciPy의 `stats.anderson()` 함수는 $p$값을 직접 제공하지 **않는다**. 검정통계량과 특정 유의수준에서의 임계값만 돌려준다.

### 왜 직접적인 p값이 없는가

Anderson-Darling 검정은 각 유의수준(정규분포의 경우 15%, 10%, 5%, 2.5%, 1%)에 대해 모의실험이나 이론적 분포표에서 얻은 미리 정해진 임계값을 쓴다. Anderson-Darling 통계량의 분포는 표본크기와 검정 대상 분포에 따라 달라지므로 정확한 $p$값을 계산하는 일이 복잡하다.

### 근사 p값 (정규성 검정의 경우)

정규성 검정에서 근사 $p$값이 필요하다면 두 가지 선택지가 있다.

**선택지 1: `statsmodels` 사용**

`statsmodels` 라이브러리가 근사 $p$값을 함께 제공하는 구현을 제공한다.

<div class="codebox" markdown>

**예제 2.** p-값으로 판정하기

```python
import numpy as np
from statsmodels.stats.diagnostic import normal_ad

np.random.seed(0)
data = np.random.normal(0, 1, 1000)

# statsmodels 의 normal_ad 는 같은 통계량에 p-값까지 붙여 준다.
# 기각값 표를 읽는 대신 p-값으로 바로 판단하고 싶을 때 쓴다.
statistic, p_value = normal_ad(data)
print(f"Anderson-Darling Test: Statistic={statistic}, p-value={p_value}")
```

출력:

```
Anderson-Darling Test: Statistic=0.2432179174634257, p-value=0.7659878263029309
```

</div>

`scipy.stats.anderson`이 준 통계량 $0.2432$와 같은 값에 근사 $p$값 $0.766$이 붙었다. 앞의 임계값 비교에서 1% 수준까지 모두 기각하지 못한 결과와 일치한다.

**선택지 2: 임계값으로 해석하기**

추가 라이브러리 없이 대략적인 근사를 원한다면 검정통계량과 제공된 임계값의 비교로 $p$값의 범위를 해석할 수 있다.

- 어떤 유의수준의 임계값보다 검정통계량이 **작으면** $p$값이 그 유의수준보다 **크다**.
- 임계값보다 검정통계량이 **크면** $p$값이 그 유의수준보다 **작다**.

### `normal_ad`를 이용한 Python 구현

<div class="codebox" markdown>

**예제 3.** 치우친 자료에 적용

```python
import numpy as np
from statsmodels.stats.diagnostic import normal_ad

np.random.seed(0)

# data = np.random.normal(0, 1, 1000)
data = np.random.normal(1, 10, 1000)

# Anderson-Darling 은 꼬리 쪽 이탈에 특히 민감하다. 꼬리가 문제가 되는
# 금융 자료에서 이 검정을 즐겨 쓰는 까닭이다.
statistic, p_value = normal_ad(data)
print(f"Anderson-Darling Test: Statistic={statistic}, p-value={p_value}")

# 결과 해석
alpha = 0.05
if p_value <= alpha:
    print("Reject H_0: The data is not normally distributed.")
else:
    print("Fail to reject H_0: The data is normally distributed.")
```

출력:

```text
Anderson-Darling Test: Statistic=0.243217917463312, p-value=0.7659878263032931
Fail to reject H_0: The data is normally distributed.
```

</div>

통계량은 `stats.anderson`과 정확히 같고, 여기에 근사 $p$값 $0.766$이 더해진다.

---

## `stats.anderson`과 `normal_ad` 중 무엇을 쓸 것인가

1. **정규성에 국한되지 않는 일반적인 분포 검정**이 필요하고 $p$값이 필요 없다면 **`stats.anderson`**이 낫다. 정규, 지수, Weibull, 로지스틱, 극값 분포를 검정할 수 있고 각각의 임계값을 제공한다.

2. **정규성 검정이 목표이고 $p$값이 필요하다면** **`statsmodels`의 `normal_ad`**가 이상적이다. 정규성 검정을 위해 특별히 설계되었고 근사 $p$값을 포함하므로 전통적인 가설검정 틀에서 결과를 해석하기 쉽다.

### 권장 사항

**명확한 $p$값 해석**이 필요한 **정규성 검정**이 초점이면 `normal_ad`를 쓴다. 여러 분포에 대해 유연하게 검정하려면 `stats.anderson`을 쓰고 제공된 임계값으로 결과를 해석한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
Anderson-Darling 검정은 Kolmogorov-Smirnov 검정보다 분포의 꼬리에 더 큰 가중치를 준다. 가중함수가 어떻게 이를 달성하는지 설명하라.

</div>

??? success "풀이"
    Anderson-Darling 통계량은

    $$
    A^2 = -n - \frac{1}{n}\sum_{i=1}^n (2i-1)[\ln F(x_{(i)}) + \ln(1 - F(x_{(n+1-i)}))]
    $$

    이는 경험적 CDF와 이론적 CDF의 제곱차를 가중 적분한 것으로 표현할 수 있다: $A^2 = n\int_{-\infty}^{\infty} \frac{[F_n(x) - F(x)]^2}{F(x)(1-F(x))}dF(x)$.

    가중함수 $w(x) = 1/[F(x)(1-F(x))]$는 ($F(x)$가 0이나 1에 가까운) 꼬리에서 크고 중앙에서 작다. 그래서 Anderson-Darling 검정이 (분포의 모든 부분에 같은 가중치를 주는) KS 검정보다 꼬리 이탈에 더 민감해지고, 꼬리가 두껍거나 얇은 대립가설을 더 잘 탐지한다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
관측값 50개에 대한 Anderson-Darling 검정에서 $A^2 = 0.85$를 얻었다. 임계값이 $0.631$(10%), $0.752$(5%), $1.035$(1%)일 때 $\alpha = 0.05$에서의 결론을 정하라.

</div>

??? success "풀이"
    $A^2 = 0.85 > 0.752$(5% 임계값)이므로 유의수준 5%에서 $H_0$(정규성)을 기각한다. 다만 $A^2 = 0.85 < 1.035$(1% 임계값)이므로 1% 수준에서는 기각하지 않는다.

    결론: 5% 수준에서 비정규성의 증거가 있다($0.01 < p < 0.05$). 이탈의 성격을 파악하려면 시각적 점검(Q-Q 그림)을 함께 해야 한다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
정규성 이탈 탐지에서 Anderson-Darling, Shapiro-Wilk, Kolmogorov-Smirnov 검정의 검정력을 비교하라.

</div>

??? success "풀이"
    모의실험 연구는 대체로 다음을 보여준다.

    1. **Shapiro-Wilk**가 전반적으로 검정력이 가장 높다. 특히 작거나 중간 크기의 $n$에서, 그리고 폭넓은 대립가설(치우침, 두꺼운 꼬리, 얇은 꼬리)에 대해 그렇다.
    2. **Anderson-Darling**은 Shapiro-Wilk에 거의 견줄 만하고 KS보다 우수하다. 꼬리 가중 덕분에 두꺼운 꼬리 대립가설을 특히 잘 탐지한다.
    3. **Kolmogorov-Smirnov(Lilliefors)**는 검정력이 가장 낮다. 가중 없이 최대 편차만 쓰므로 (이탈이 작은) 중앙과 (가장 중요한) 꼬리에 같은 관심을 주기 때문이다.

    권장: Shapiro-Wilk를 주 검정으로 쓰고, (특히 $n$이 클 때) Anderson-Darling을 대안으로 쓰며, KS는 완전히 지정된 분포와 비교할 때에만 쓴다(복합 정규성 검정에는 쓰지 않는다).

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
Anderson-Darling 검정은 정규분포가 아닌 분포(지수, Weibull 등)에도 적용할 수 있다. 일반 원리를 설명하라.

</div>

??? success "풀이"
    Anderson-Darling 검정은 일반적인 적합도 검정이다. 경험적 CDF를 임의로 지정한 이론적 CDF $F_0(x)$와 비교한다. 정규성 검정에서는 추정된 모수로 $F_0 = \Phi((x-\hat{\mu})/\hat{\sigma})$를 쓴다.

    자료가 지수분포를 따르는지 검정하려면 $F_0(x) = 1 - e^{-x/\hat{\lambda}}$를 쓴다. Weibull이라면 추정된 모양·척도 모수를 가진 Weibull CDF를 쓴다.

    검정통계량 공식은 모든 경우에 같고 $F_0$만 바뀐다. 임계값은 분포족마다 다른데, $A^2$의 귀무분포가 추정한 모수의 개수와 기준분포의 모양에 의존하기 때문이다. 각 분포족에 대해 전용 임계값 표나 모의실험 기반 p값을 쓴다.

---

## 정리하며

앤더슨–달링은 **꼬리에 가중**을 둔 EDF 검정이다.

- **KS 를 개선한 것이다.** 적분 안에 $1/[F_0(x)(1-F_0(x))]$ 가중을 넣어 **꼬리에서의 차이를 크게 센다.**
- **꼬리 이탈 탐지에 강하다.** 금융 자료처럼 꼬리가 문제인 상황에서 KS 보다 훨씬 예민하다.
- **$p$ 값 대신 임계값 표를 준다.** `scipy.stats.anderson` 은 $A^2$ 통계량과 여러 유의수준의 임계값을 돌려주며, **$p$ 값을 직접 주지 않는다.**
- **분포마다 임계값이 다르다.** 정규·지수·로지스틱 등 지정한 분포에 따라 표가 달라지며, 모수를 추정했다는 사실이 이미 반영되어 있다.
- **소표본에서도 쓸 만하다.** 샤피로–윌크와 함께 실무의 주력이다.

다음 절 **Shapiro-Wilk 검정**으로 넘어간다.
