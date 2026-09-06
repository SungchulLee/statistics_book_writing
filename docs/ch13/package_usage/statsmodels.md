# statsmodels 최소제곱 인터페이스

`statsmodels` 라이브러리는 회귀에서 통계적 추론을 수행하는 Python의 대표적인 도구이다. 예측에 초점을 맞추는 기계학습 라이브러리와 달리 `statsmodels`는 계수 추정값, 표준오차, t 통계량, p값, 신뢰구간을 담은 상세한 요약표를 제공한다. 통계 분석에서 기대하는 표준적인 출력이다.

---

## 1. OLS 클래스

최소제곱 회귀의 핵심 인터페이스는 `statsmodels.api.OLS`이다. 이 클래스는 사용자가 설명변수 행렬에 상수(절편) 열을 명시적으로 추가할 것을 요구한다.

```python
import numpy as np
import statsmodels.api as sm

# Generate example data
np.random.seed(42)
n = 100
X = np.random.randn(n, 2)
beta_true = np.array([3.0, 1.5])
y = X @ beta_true + 2.0 + np.random.randn(n) * 0.5

# Add constant for intercept
X_with_const = sm.add_constant(X)

# Fit OLS model
model = sm.OLS(y, X_with_const)
results = model.fit()

print(results.summary())
```

`sm.add_constant(X)` 함수는 설명변수 행렬 앞에 1로 채운 열을 붙인다. 이는 모형 $Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \varepsilon$의 절편항 $\beta_0$에 대응한다.

!!! warning "상수를 빠뜨리면"
    `sm.add_constant()`를 호출하지 않고 `X`를 그대로 넘기면 모형이 원점을 지나는 회귀(절편 없음)를 적합한다. 이는 거의 언제나 원하는 바가 아니다. 절편을 의도적으로 없앨 특별한 이유가 없다면 항상 상수를 추가하라.

---

## 2. 요약표 읽기

`results.summary()` 출력은 세 개의 패널로 이루어진다. 가장 중요한 항목은 다음과 같다.

### 위 패널 (모형 정보)

| 항목 | 의미 |
|---|---|
| R-squared | 설명된 분산의 비율 ($R^2$) |
| Adj. R-squared | 설명변수 개수를 반영해 조정한 $R^2$ |
| F-statistic | 회귀의 유의성에 대한 전체 F 검정 |
| Prob (F-statistic) | F 검정의 p값 |
| AIC / BIC | 모형 비교를 위한 정보기준 |

### 가운데 패널 (계수)

| 열 | 의미 |
|---|---|
| coef | 추정된 회귀계수 $\hat{\beta}_j$ |
| std err | $\hat{\beta}_j$의 표준오차 |
| t | t 통계량: $t = \hat{\beta}_j / \text{SE}(\hat{\beta}_j)$ |
| P>\|t\| | $H_0: \beta_j = 0$에 대한 양측 p값 |
| [0.025, 0.975] | $\beta_j$의 95% 신뢰구간 |

p값이 0.05보다 작으면, 동등하게 95% 신뢰구간이 0을 포함하지 않으면 그 설명변수는 유의수준 5%에서 통계적으로 유의하다.

### 아래 패널 (진단)

| 항목 | 의미 |
|---|---|
| Omnibus / Prob(Omnibus) | 잔차의 정규성 검정 |
| Durbin-Watson | 잔차의 자기상관 검정(2에 가까우면 자기상관 없음) |
| Jarque-Bera / Prob(JB) | 왜도와 첨도에 기초한 또 다른 정규성 검정 |
| Cond. No. | 설계행렬의 조건수(값이 크면 다중공선성을 시사) |

---

## 3. 식(formula) API

`statsmodels`의 식 API는 `patsy` 식을 이용해 R과 비슷한 문법을 제공한다. 이 인터페이스는 절편을 자동으로 추가하고 범주형 변수를 알아서 처리한다.

```python
import pandas as pd
import statsmodels.formula.api as smf

# Create a DataFrame
df = pd.DataFrame({
    'y': y,
    'x1': X[:, 0],
    'x2': X[:, 1]
})

# Fit using formula API
results_formula = smf.ols('y ~ x1 + x2', data=df).fit()
print(results_formula.summary())
```

식 `'y ~ x1 + x2'`는 모형 $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \varepsilon$을 지정한다. 절편은 기본으로 포함된다. 없애려면 `'y ~ x1 + x2 - 1'`을 쓴다.

### 식 문법

| 식 | 모형 |
|---|---|
| `y ~ x1 + x2` | $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2$ |
| `y ~ x1 * x2` | $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1 x_2$ |
| `y ~ x1 + I(x1**2)` | $y = \beta_0 + \beta_1 x_1 + \beta_2 x_1^2$ |
| `y ~ C(group)` | 범주형 변수의 원핫 부호화 |
| `y ~ x1 + x2 - 1` | 절편 없음 |

`I()` 감싸기는 `patsy`에게 그 식을 식 연산자가 아니라 산술로 해석하라고 알린다. 이것이 없으면 `x1**2`가 "x1의 제곱"으로 인식되지 않는다.

---

## 4. 결과를 프로그램으로 다루기

적합된 `results` 객체는 이후 분석에 필요한 모든 양을 담고 있다.

```python
# Coefficient estimates
print("Coefficients:", results.params)

# Standard errors
print("Standard errors:", results.bse)

# P-values
print("P-values:", results.pvalues)

# Confidence intervals
print("95% CI:\n", results.conf_int(alpha=0.05))

# R-squared and adjusted R-squared
print("R-squared:", results.rsquared)
print("Adjusted R-squared:", results.rsquared_adj)

# Residuals
residuals = results.resid

# Fitted values
fitted = results.fittedvalues

# AIC and BIC
print("AIC:", results.aic)
print("BIC:", results.bic)
```

---

## 5. 진단 메서드

`results` 객체는 모형 진단을 위한 메서드를 제공한다.

```python
# Influence diagnostics (leverage, Cook's distance)
influence = results.get_influence()
cooks_d = influence.cooks_distance[0]
leverage = influence.hat_matrix_diag

# Heteroscedasticity tests
from statsmodels.stats.diagnostic import het_breuschpagan
bp_stat, bp_pval, _, _ = het_breuschpagan(results.resid, results.model.exog)
print(f"Breusch-Pagan p-value: {bp_pval:.4f}")

# Normality test on residuals
from statsmodels.stats.stattools import jarque_bera
jb_stat, jb_pval, skew, kurtosis = jarque_bera(results.resid)
print(f"Jarque-Bera p-value: {jb_pval:.4f}")

# Variance Inflation Factors (skip index 0: the constant column)
from statsmodels.stats.outliers_influence import variance_inflation_factor
for i in range(1, X_with_const.shape[1]):
    vif = variance_inflation_factor(X_with_const, i)
    print(f"VIF for variable {i}: {vif:.2f}")
```

!!! note "VIF 반복문에서 상수 열은 건너뛴다"
    `X_with_const`의 0번 열은 절편을 위한 상수이다. 여기에 `variance_inflation_factor`를 호출해도 오류는 나지 않지만 그 값에는 아무 의미가 없다. 위 코드처럼 `range(1, ...)`로 시작해 실제 설명변수만 다루어야 한다.

!!! tip "언제 statsmodels를 쓰는가"
    주된 목표가 통계적 추론일 때 `statsmodels`를 쓴다. 곧 계수에 대한 가설검정, 신뢰구간 구성, 모형 가정 진단이 목적일 때이다. 추론이 필요 없는 순수 예측 작업에는 `sklearn.linear_model.LinearRegression`이 더 간단한 인터페이스를 제공한다.

## 연습문제

**연습문제 1.**
다음은 `statsmodels` 패키지로 선형회귀 분석을 수행한 결과이다. **TV**, **라디오**, **신문** 매체에 배정한 광고비로 **매출**을 예측한다.

```text
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  Sales   R-squared:                       0.894
Model:                            OLS   Adj. R-squared:                  0.891
Method:                 Least Squares   F-statistic:                     381.2
Date:                Mon, 11 Nov 2024   Prob (F-statistic):           5.60e-66
Time:                        02:39:45   Log-Likelihood:                -273.89
No. Observations:                 140   AIC:                             555.8
Df Residuals:                     136   BIC:                             567.5
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      3.0451      0.391      7.782      0.000       2.271       3.819
TV             0.0470      0.002     27.653      0.000       0.044       0.050
Radio          0.1797      0.011     16.665      0.000       0.158       0.201
Newspaper     -0.0030      0.007     -0.428      0.669      -0.017       0.011
==============================================================================
Omnibus:                       50.782   Durbin-Watson:                   2.089
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              131.355
Skew:                          -1.459   Prob(JB):                     3.00e-29
Kurtosis:                       6.741   Cond. No.                         457.
==============================================================================
```

**(a)** TV, 라디오, 신문 광고비를 각각 $x_1$, $x_2$, $x_3$으로, 매출을 $y$로 나타낼 때, 회귀 결과에 근거한 예측값 $\hat{y}$은 무엇인가?

**(b)** 신문 광고의 계수는 $-0.0030$이다. 이를 신문 광고가 매출을 줄인다는 뜻으로 해석할 수 있는가? $p$값에 근거하여 그 타당성을 논하라.

**(c)** Jarque-Bera(JB) 통계량 131.355와 그 $p$값 3.00e-29는 무엇을 뜻하는가?

??? success "연습문제 1 풀이"

    **(a)** 매출의 예측값은

    $$
    \hat{y} = 3.0451 + 0.0470 \cdot x_1 + 0.1797 \cdot x_2 - 0.0030 \cdot x_3
    $$

    절편과 각 광고 매체의 계수를 결합하여 매출을 예측한다.

    **(b)** 신문 계수의 $p$값은 $0.669$로 표준 유의수준 0.05보다 훨씬 크다. 곧 계수가 0이라는 귀무가설을 기각하지 못한다. 계수가 음수이긴 하지만 $p$값이 크므로 이 결과는 통계적으로 의미가 없다. 신문 광고가 음의 효과를 갖는다고 해석하기보다는 매출에 통계적으로 유의한 영향을 주지 않는다고 결론짓는 것이 적절하다. 95% 신뢰구간 $(-0.017, 0.011)$이 0을 포함한다는 사실도 같은 이야기를 한다.

    **(c)** Jarque-Bera 검정은 잔차가 정규분포를 따르는지 평가한다. $p$값이 $3.00 \times 10^{-29}$로 극히 작으므로 귀무가설(잔차가 정규분포를 따른다)이 강하게 기각된다. 잔차가 정규분포를 따르지 **않을** 가능성이 높다는 뜻이며, 비정규 오차나 이상점의 존재 같은 모형 문제를 시사할 수 있다. 실제로 왜도 $-1.459$와 첨도 $6.741$은 정규분포의 값(0과 3)에서 크게 벗어나 있어, 왼쪽으로 치우치고 꼬리가 두꺼운 잔차 분포임을 보여준다.
