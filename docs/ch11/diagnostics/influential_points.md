# 영향점

## 개요

분산분석에서 어떤 자료점은 결과에 지나치게 큰 영향을 주어 결론을 왜곡할 수 있다. 이런 영향점은 이상점(특이한 반응값)일 수도 있고 지렛점(특이한 설명변수값)일 수도 있으며, 추정된 집단 평균과 분산, 전체 F-통계량에 상당한 영향을 줄 수 있다. 이런 점들을 찾아 다루는 일은 분산분석 결과의 로버스트성을 확보하는 데 결정적이다.

## Cook의 거리

Cook의 거리는 각 관측값의 잔차와 지렛값을 결합하여 적합된 모형에 대한 전체적인 영향을 평가한다. 관측값 $i$를 제거했을 때 적합값이 얼마나 변하는지를 잰다:

$$
D_i = \frac{r_i^2}{p} \cdot \frac{h_{ii}}{1 - h_{ii}}
$$

여기서 $r_i$는 표준화 잔차, $h_{ii}$는 지렛값, $p$는 모형의 모수 개수(일원배치 분산분석에서는 집단의 수)이다.

```python
import numpy as np
import matplotlib.pyplot as plt

influence = model.get_influence()
cooks_d = influence.cooks_distance[0]

plt.stem(range(len(cooks_d)), cooks_d, markerfmt=",")
plt.xlabel("Observation Index")
plt.ylabel("Cook's Distance")
plt.title("Cook's Distance")
plt.axhline(y=4/len(cooks_d), color='r', linestyle='--', label=f'Threshold = {4/len(cooks_d):.3f}')
plt.legend()
plt.show()
```

영향점을 찾는 데 흔히 쓰는 문턱:

- $D_i > 4/n$: 흔히 쓰이는 경험 법칙.
- $D_i > 1$: 더 보수적인 문턱.
- $D_i > F_{0.50}(p, n-p)$: $F$-분포의 중앙값에 근거한 문턱.

## 지렛값

지렛값은 관측값의 설명변수값이 설명변수 평균에서 얼마나 떨어져 있는지를 잰다. 일원배치 분산분석에서 지렛값은 집단 크기에 의존한다:

$$
h_{ii} = \frac{1}{n_i}
$$

여기서 $n_i$는 관측값 $i$가 속한 집단의 크기이다. 작은 집단에 속한 점일수록 지렛값이 크다.

지렛값이 큰 점이 반드시 영향점인 것은 아니다. 잔차가 클 때에만 영향점이 된다.

```python
leverage = influence.hat_matrix_diag

plt.scatter(leverage, influence.resid_studentized_internal, alpha=0.6)
plt.xlabel("Leverage")
plt.ylabel("Studentized Residuals")
plt.title("Leverage vs. Studentized Residuals")
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
```

## DFFITS

DFFITS는 각 관측값이 자기 자신의 적합값에 주는 영향을 잰다:

$$
\text{DFFITS}_i = r_i^* \sqrt{\frac{h_{ii}}{1 - h_{ii}}}
$$

여기서 $r_i^*$는 외부 스튜던트화 잔차이다. 흔한 문턱은 $|\text{DFFITS}_i| > 2\sqrt{p/n}$이다.

## 영향점 다루기

영향점을 찾았을 때 고려할 수 있는 전략은 여럿이다:

**조사:**
어떤 조치를 취하기 전에 왜 그 점이 영향력이 큰지 조사한다. 자료 입력 오류인가? 측정 이상인가? 아니면 과학적으로 의미 있는 정말로 특이한 관측인가?

**민감도 분석:**
영향점을 포함한 경우와 제외한 경우로 분산분석을 각각 수행한다. 결론이 크게 달라지면 결과가 그 관측값에 로버스트하지 않다는 뜻이므로 이를 보고해야 한다.

**제거:**
실질적인 근거(예: 알려진 자료 오류)가 있을 때에만 영향점을 제거한다. 단지 불편하다는 이유로 점을 제거해서는 안 된다.

**변환:**
자료에 변환(예: 로그, 제곱근)을 적용하면 척도가 압축되어 극단값의 영향이 줄어들 수 있다.

**로버스트 분산분석 방법:**
절사평균, 윈저화 평균, M-추정량 같은 방법은 이상점의 영향을 낮추어 더 신뢰할 만한 결과를 준다. 붓스트랩 방법도 영향점에 대한 로버스트성을 제공한다.

## 예제: 완전한 영향 진단

```python
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Fit model
model = ols('response ~ group', data=data).fit()

# Influence diagnostics
influence = model.get_influence()
summary = influence.summary_frame()
print(summary[['hat_diag', 'cooks_d', 'dffits', 'student_resid']].describe())
```

## 연습문제

**연습문제 1.**
집단이 셋이고 각각 $n = 10$인 일원배치 분산분석에서 어떤 관측값의 Cook의 거리가 $D_i = 1.2$이다. 흔히 쓰는 문턱은 $D_i > 4/N$이다. 이 점이 영향점인지 판정하고 Cook의 거리가 무엇을 재는지 설명하라.

??? success "연습문제 1 풀이"
    문턱은 $4/N = 4/30 \approx 0.133$이다. $D_i = 1.2 \gg 0.133$이므로 이 관측값은 영향력이 매우 크다.

    Cook의 거리는 관측값 $i$가 모든 적합값에 동시에 주는 전체적인 영향을 잰다. 지렛값(설명변수값이 얼마나 특이한지)과 잔차의 크기(적합 모형에서 얼마나 떨어져 있는지)를 결합한다. Cook의 거리가 크다는 것은 그 관측값을 제거하면 추정된 집단 평균과 F-통계량이 상당히 달라진다는 뜻이다.

---

**연습문제 2.**
분산분석의 맥락에서 이상점, 지렛점, 영향점을 구별하라. 지렛값은 크지만 영향점이 아닌 예를 들어라.

??? success "연습문제 2 풀이"

    - **이상점:** 잔차가 유별나게 큰(자기 집단 평균에서 멀리 떨어진) 관측값.
    - **지렛점:** 설명변수값이 특이한 관측값. 분산분석에서는 보통 관측값이 아주 적은 집단에 속해 집단 평균에 더 큰 영향을 주는 경우를 뜻한다.
    - **영향점:** 제거하면 결과가 상당히 달라지는 관측값. 대체로 지렛값도 크고 잔차도 크다.

    **지렛값은 크지만 영향점이 아닌 예:** 일원배치 분산분석에서 한 집단만 $n = 3$이고 다른 집단은 $n = 30$이라면 작은 집단의 각 관측값은 지렛값(햇값)이 크다. 그러나 그 관측값들이 자기 집단 평균에 가까우면 잔차가 작아 영향점이 아니다(Cook의 거리가 낮게 유지된다).

---

**연습문제 3.**
집단 B의 어떤 관측값에서 DFFITS 값이 $2\sqrt{p/n}$을 넘었다. DFFITS가 무엇을 재는지, Cook의 거리와 어떻게 다른지 설명하라.

??? success "연습문제 3 풀이"
    DFFITS는 관측값 $i$를 삭제했을 때 그 관측값의 **적합값**이 얼마나 변하는지를 표준오차로 나누어 잰다. 형식적으로

    $$
    \text{DFFITS}_i = \frac{\hat{Y}_i - \hat{Y}_{i(i)}}{s_{(i)} \sqrt{h_{ii}}}
    $$

    이며 $\hat{Y}_{i(i)}$는 관측값 $i$를 제외했을 때의 적합값이다.

    Cook의 거리와의 핵심 차이는, DFFITS가 **하나의 적합값**(그 관측값 자신의 예측)에 대한 효과에 초점을 두는 반면 Cook의 거리는 **모든 적합값에 대한 효과를 동시에** 잰다는 점이다. 영향이 국소적이면 DFFITS는 크지만 Cook의 거리는 중간 정도일 수 있다.
