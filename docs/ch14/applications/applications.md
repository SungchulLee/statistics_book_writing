# 통계학에서 정규성 검정의 응용

정규성 검정은 통계 분석의 필수적인 부분이다. 흔히 쓰는 많은 통계 방법이 자료가 정규분포를 따른다는 가정에 기대기 때문이다.

## 언제 정규성 검정을 적용하는가

정규성 검정은 보통 정규성을 가정하는 모수적 통계 방법을 쓰기 전에 적용한다. 흔한 상황은 다음과 같다.

- **$t$ 검정**: 일표본과 이표본 $t$ 검정 모두 각 집단 안의 자료가 정규분포를 따른다고 가정한다.
- **분산분석**: 분산분석은 집단들에 걸쳐 잔차가 정규분포를 따른다고 가정한다.
- **선형회귀**: 회귀분석은 모형의 잔차(오차)가 정규분포를 따른다고 가정한다.
- **신뢰구간**: 모수의 신뢰구간을 만들 때, 특히 표본이 작을 때 정규성을 흔히 가정한다.

정규성 검정은 이런 방법을 쓰는 것이 적절한지, 아니면 대안(변수변환이나 비모수 검정)을 고려해야 하는지 판단하는 데 유용하다.

## 사례 1: t 검정 가정의 정규성 확인

이표본 $t$ 검정은 각 표본 안의 자료가 정규분포를 따른다고 가정한다. $t$ 검정을 수행하기 전에 두 집단의 정규성을 확인하는 것이 필수적이다.

```python
import numpy as np
from scipy.stats import ttest_ind, shapiro

np.random.seed(0)

# Generate two sample datasets
group1 = np.random.normal(0, 1, 50)
group2 = np.random.normal(0.5, 1, 50)

# Perform Shapiro-Wilk test for normality on both groups
_, p_value_group1 = shapiro(group1)
_, p_value_group2 = shapiro(group2)

if p_value_group1 > 0.05 and p_value_group2 > 0.05:
    # If both groups pass the normality test, perform a t-test
    stat, p_value = ttest_ind(group1, group2)
    print(f"Two-sample t-test: p-value={p_value}")
else:
    print("One or both groups fail the normality test. Consider using a non-parametric alternative.")
```

출력:

```text
Two-sample t-test: p-value=0.09856088...
```

두 집단의 Shapiro-Wilk $p$값은 각각 $0.877$과 $0.837$로 정규성 확인을 통과하며, 이어진 $t$ 검정의 $p$값은 $0.0986$이다. 참 평균 차이가 0.5인데도 집단당 50개로는 5% 수준에서 기각하지 못한다는 점이 흥미롭다. 검정력의 문제이다.

한 집단이라도 정규성 검정을 통과하지 못하면 **Mann-Whitney U 검정** 같은 비모수 대안을 써야 한다.

!!! warning "정규성 검정 결과로 분석을 분기하는 것의 위험"
    위 코드처럼 "정규성 검정을 통과하면 $t$ 검정, 아니면 비모수 검정"으로 자동 분기하는 것은 널리 쓰이는 관행이지만 문제가 있다. 최종 검정의 선택이 같은 자료에 의존하게 되어 실제 제1종 오류율이 명목값에서 벗어난다. 실무에서는 사전에 분석 방법을 정하거나, 두 결과를 모두 보고하는 편이 낫다.

## 사례 2: 선형회귀 잔차의 정규성

선형회귀에서는 잔차(관측값과 예측값의 차이)가 정규분포를 따른다고 가정한다. 잔차에 정규성 검정을 적용하여 이 가정이 성립하는지 확인할 수 있다.

```python
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import shapiro

# Generate example data
np.random.seed(0)
X = np.random.normal(0, 1, 100)
y = 2 * X + np.random.normal(0, 1, 100)

# Add a constant to X for the intercept
X = sm.add_constant(X)

# Fit the linear model
model = sm.OLS(y, X).fit()

# Get the residuals
residuals = model.resid

# Perform a Shapiro-Wilk test on the residuals
_, p_value = shapiro(residuals)

print(f"Shapiro-Wilk Test on Residuals: p-value={p_value}")

# Plot residuals
plt.hist(residuals, bins=20)
plt.title('Residuals Histogram')
plt.show()

if p_value > 0.05:
    print("Residuals are normally distributed.")
else:
    print("Residuals are not normally distributed.")
```

출력:

```text
Shapiro-Wilk Test on Residuals: p-value=0.11416...
Residuals are normally distributed.
```

(엄밀히 말하면 "잔차가 정규분포를 따른다"가 아니라 "잔차가 정규성과 일관된다"가 옳은 표현이다. 기각하지 못한 것이 정규성을 증명하지는 않는다.)

잔차가 정규분포를 따르지 않으면 회귀분석 결과를 믿기 어려워질 수 있으며, 변수변환이나 대안 회귀모형 같은 교정 조치가 필요할 수 있다.

## 사례 3: 분산분석의 정규성

**분산분석**은 집단들에 걸친 자료의 잔차가 정규분포를 따른다고 가정한다. 이 가정이 위배되면 분산분석의 결과가 오도할 수 있다.

```python
import numpy as np
from scipy.stats import f_oneway, shapiro

np.random.seed(0)

# Generate sample data for three groups
group1 = np.random.normal(0, 1, 30)
group2 = np.random.normal(0.5, 1, 30)
group3 = np.random.normal(1, 1, 30)

# Perform Shapiro-Wilk test on the residuals
_, p_value_group1 = shapiro(group1)
_, p_value_group2 = shapiro(group2)
_, p_value_group3 = shapiro(group3)

# Check if the data is normally distributed
if p_value_group1 > 0.05 and p_value_group2 > 0.05 and p_value_group3 > 0.05:
    # Perform ANOVA
    stat, p_value = f_oneway(group1, group2, group3)
    print(f"ANOVA test: p-value={p_value}")
else:
    print("One or more groups fail the normality test. Consider using a non-parametric alternative.")
```

출력:

```text
ANOVA test: p-value=0.03998...
```

세 집단의 Shapiro-Wilk $p$값은 각각 $0.525$, $0.909$, $0.720$으로 모두 정규성 확인을 통과하고, 분산분석은 $p = 0.040$으로 5% 수준에서 집단 평균의 차이를 탐지한다.

분산분석을 적용하기 전에 Shapiro-Wilk 검정으로 각 집단의 자료가 정규분포를 따르는지 확인한다. 한 집단 이상이 검정을 통과하지 못하면 **Kruskal-Wallis 검정** 같은 비모수 대안이 더 적절할 수 있다.

## 결론

정규성 검정은 $t$ 검정, 분산분석, 선형회귀 같은 모수적 방법을 쓰는 여러 응용에서 결정적이다. 자료(또는 잔차)가 정규분포를 따르는지 확인해야 이 방법들이 타당한 결과를 낸다. 정규성 가정이 위배되면 변수변환이나 비모수 대안을 적용할 수 있다. 실무에서는 정규성 검정을 시각적 평가와 결합할 때 바탕 자료 분포를 더 뚜렷하게 파악할 수 있다.

## 연습문제

**연습문제 1.**
어떤 연구자가 관측값 $n = 15$개에 일표본 t 검정을 수행하여 $t = 2.35$를 얻었다. 결과를 해석하기 전에 정규성을 확인해야 한다. 그 이유를 설명하고, 자료가 심하게 치우쳐 있다면 무엇이 잘못될 수 있는지 기술하라.

??? success "풀이"
    일표본 t 검정은 자료가 정규분포에서 온다고 가정한다(또는 $n$이 중심극한정리를 쓸 만큼 크다고 가정한다). $n = 15$이면 자료가 심하게 치우쳐 있을 때 중심극한정리가 충분한 근사를 제공하지 못할 수 있다.

    자료가 오른쪽으로 치우쳐 있으면 $\bar{X}$의 표집분포도 치우치며 t 분포가 나쁜 근사가 된다. 그러면 (1) p값이 틀리고(실제 제1종 오류율이 명목 $\alpha$와 다르다), (2) 신뢰구간의 포함확률이 틀리며, (3) 실제 효과를 탐지할 검정력이 떨어진다. 치우침이 심하고 $n = 15$라면 비모수 검정(Wilcoxon 부호순위)이나 붓스트랩 검정이 더 믿을 만하다.

---

**연습문제 2.**
정규성 가정에 기대는 통계 방법 세 가지를 들고 각각이 비정규성에 얼마나 로버스트한지 서술하라.

??? success "풀이"

    1. **평균에 대한 t 검정:** 중간 정도로 로버스트하다. $n \geq 30$이고 치우침이 중간 정도이면 중심극한정리가 근사적 타당성을 보장한다. 작은 표본에서 두꺼운 꼬리나 극단적 이상점에는 로버스트하지 않다.

    2. **분산의 동일성에 대한 F 검정:** 로버스트하지 않다. F 검정은 비정규성에 매우 민감하여 가벼운 이탈만으로도 제1종 오류율이 심각하게 부풀려질 수 있다. Levene 검정이나 Brown-Forsythe 검정이 선호되는 대안이다.

    3. **선형회귀(OLS):** OLS 추정값 자체는 정규성 없이도 타당하다(불편, BLUE). 그러나 추론(t 검정, F 검정, 신뢰구간)에는 오차의 정규성이나 큰 $n$이 필요하다. 예측구간은 비정규성에 특히 민감하다.

---

**연습문제 3.**
통계검정을 수행하기 전에 정규성을 확인하는 실용적 절차를 설명하라.

??? success "풀이"
    권장 절차:

    1. **먼저 시각적으로 점검한다:** 자료(회귀에서는 잔차)의 히스토그램이나 밀도 그림과 Q-Q 그림을 만든다. 치우침, 두꺼운 꼬리, 이상점, 다봉성을 살핀다.

    2. **형식적 검정:** 시각적 방법을 보완하기 위해 정규성 검정을 적용한다($n < 50$이면 Shapiro-Wilk, 더 큰 표본에는 Anderson-Darling이나 D'Agostino).

    3. **결과를 함께 해석한다:** Q-Q 그림이 대략 선형이고 형식적 검정이 합리적인 수준에서 기각하지 않으면 정규론 방법으로 진행한다. 둘 다 비정규성을 시사하면 대안을 고려한다.

    4. **필요하면 대책을 고른다:** 변환(로그, Box-Cox), 비모수 검정, 붓스트랩 방법을 적용한다.

    5. **평가를 보고한다:** 정규성이 뒷받침되더라도 어떤 정규성 확인을 수행했고 그 결과가 무엇이었는지 서술한다.

---

**연습문제 4.**
큰 표본($n = 5000$)에서 Shapiro-Wilk 검정이 $p < 0.001$로 정규성을 기각했지만 Q-Q 그림은 거의 선형으로 보인다. 어떻게 진행해야 하는가?

??? success "풀이"
    $n = 5000$이면 형식적 검정의 검정력이 극도로 높아, 추론에 실질적 영향이 없는 사소한 이탈까지 탐지한다. Shapiro-Wilk $p < 0.001$은 자료가 정규에서 "멀다"는 뜻이 아니라 그 이탈이 통계적으로 탐지 가능하다는 뜻이다.

    Q-Q 그림이 거의 선형이므로 이탈은 작을 가능성이 높다. 다음과 같이 진행한다.

    1. **정규론 방법으로 진행한다:** $n = 5000$이면 중심극한정리의 보호가 강력하여 약간의 비정규성이 있어도 t 검정, 분산분석, 회귀 추론이 매우 정확하다.
    2. **두 결과를 모두 보고한다:** 형식적 검정은 정규성을 기각했지만 시각적 점검은 근사적 정규성을 시사한다고 밝힌다.
    3. **효과 크기를 고려한다:** 이분법적 기각/비기각 판정에 기대는 대신 왜도와 첨도 계수로 비정규성의 정도를 수량화한다.
