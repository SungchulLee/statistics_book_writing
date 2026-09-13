# statsmodels 최소제곱 인터페이스

`statsmodels` 라이브러리는 회귀에서 통계적 추론을 수행하는 Python의 대표적인 도구이다. 예측에 초점을 맞추는 기계학습 라이브러리와 달리 `statsmodels`는 계수 추정값, 표준오차, t 통계량, p값, 신뢰구간을 담은 상세한 요약표를 제공한다. 통계 분석에서 기대하는 표준적인 출력이다.

---

## 1. OLS 클래스

최소제곱 회귀의 핵심 인터페이스는 `statsmodels.api.OLS`이다. 이 클래스는 사용자가 설명변수 행렬에 상수(절편) 열을 명시적으로 추가할 것을 요구한다.

<div class="codebox" markdown>

**예제 1.** OLS 적합과 출력표

```python
import numpy as np
import statsmodels.api as sm

# 참 계수가 [3.0, 1.5], 절편이 2.0 인 자료다.
np.random.seed(42)
n = 100
X = np.random.randn(n, 2)
beta_true = np.array([3.0, 1.5])
y = X @ beta_true + 2.0 + np.random.randn(n) * 0.5

# statsmodels 는 절편을 자동으로 넣지 않는다. 이 줄을 빠뜨리면 원점을
# 지나는 회귀가 되므로 sklearn 과의 가장 흔한 차이가 여기서 생긴다.
X_with_const = sm.add_constant(X)

# OLS(y, X) 순서다. sklearn 의 fit(X, y) 와 반대이니 헷갈리기 쉽다.
model = sm.OLS(y, X_with_const)
results = model.fit()

def print_summary(res):
    """summary()에서 실행 날짜와 시각만 지우고 인쇄한다.

    statsmodels의 summary()는 표 머리에 Date와 Time을 함께 찍는다.
    그대로 두면 실행할 때마다 출력이 달라져 문서에 싣기 어렵다.
    같은 줄에 있는 다른 값(Prob (F-statistic), Log-Likelihood)은 남긴다.
    """
    lines = []
    for line in str(res.summary()).split("\n"):
        if line.startswith(("Date:", "Time:")):
            label, rest = line[:19], line[19:]
            lines.append(label.ljust(19) + " " * 19 + rest[19:])
        else:
            lines.append(line)
    print("\n".join(lines))

print_summary(results)
```

출력:

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.971
Model:                            OLS   Adj. R-squared:                  0.970
Method:                 Least Squares   F-statistic:                     1603.
Date:                                   Prob (F-statistic):           4.97e-75
Time:                                   Log-Likelihood:                -77.799
No. Observations:                 100   AIC:                             161.6
Df Residuals:                      97   BIC:                             169.4
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          2.0464      0.054     37.884      0.000       1.939       2.154
x1             3.0954      0.063     49.281      0.000       2.971       3.220
x2             1.4139      0.054     26.258      0.000       1.307       1.521
==============================================================================
Omnibus:                        4.136   Durbin-Watson:                   2.212
Prob(Omnibus):                  0.126   Jarque-Bera (JB):                3.956
Skew:                           0.266   Prob(JB):                        0.138
Kurtosis:                       3.817   Cond. No.                         1.23
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

요약표를 세 부분으로 나눠 읽는다.

- **위 블록**: 모형 수준의 적합도. $R^2 = 0.971$, $F = 1603$, AIC/BIC.
- **가운데 블록**: 계수별 추정값, 표준오차, $t$, p-값, 신뢰구간. 참값이 절편 2.0, 기울기 3.0과 1.5인데 추정값이 2.046, 3.095, 1.414로 잘 맞는다.
- **아래 블록**: 잔차 진단. Durbin-Watson 2.21은 자기상관 없음을, Jarque-Bera $p = 0.138$은 정규성 이탈의 증거 없음을 뜻한다.

`Cond. No.` 1.23도 눈여겨보라. 설명변수를 독립으로 만들었으므로 다중공선성이 없다. 이 값이 30을 넘으면 공선성을 의심한다.

`sm.add_constant(X)` 함수는 설명변수 행렬 앞에 1로 채운 열을 붙인다. 이는 모형 $Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \varepsilon$의 절편항 $\beta_0$에 대응한다.

</div>

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

<div class="codebox" markdown>

**예제 2.** 수식 API

```python
import pandas as pd
import statsmodels.formula.api as smf

# 수식 API 는 R 의 문법을 따른다. 이쪽에서는 절편이 자동으로 들어가고,
# 범주형 변수도 알아서 가변수로 바뀐다.
df = pd.DataFrame({
    'y': y,
    'x1': X[:, 0],
    'x2': X[:, 1]
})

results_formula = smf.ols('y ~ x1 + x2', data=df).fit()
print_summary(results_formula)
```

출력:

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.971
Model:                            OLS   Adj. R-squared:                  0.970
Method:                 Least Squares   F-statistic:                     1603.
Date:                                   Prob (F-statistic):           4.97e-75
Time:                                   Log-Likelihood:                -77.799
No. Observations:                 100   AIC:                             161.6
Df Residuals:                      97   BIC:                             169.4
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      2.0464      0.054     37.884      0.000       1.939       2.154
x1             3.0954      0.063     49.281      0.000       2.971       3.220
x2             1.4139      0.054     26.258      0.000       1.307       1.521
==============================================================================
Omnibus:                        4.136   Durbin-Watson:                   2.212
Prob(Omnibus):                  0.126   Jarque-Bera (JB):                3.956
Skew:                           0.266   Prob(JB):                        0.138
Kurtosis:                       3.817   Cond. No.                         1.23
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

식 API가 앞의 배열 API와 **완전히 같은 결과**를 준다. 계수 이름이 `const, x1, x2`에서 `Intercept, x1, x2`로 바뀐 것뿐이다.

식 API는 절편을 자동으로 넣어 준다. 배열 API에서 `add_constant`를 빠뜨리는 실수를 막아 준다는 점이 실용적인 장점이다.

식 `'y ~ x1 + x2'`는 모형 $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \varepsilon$을 지정한다. 절편은 기본으로 포함된다. 없애려면 `'y ~ x1 + x2 - 1'`을 쓴다.

</div>

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

<div class="codebox" markdown>

**예제 3.** 결과에서 값 꺼내기

```python
# 적합 결과에서 꺼낼 수 있는 것들을 한자리에 모았다.
print("Coefficients:", results.params)

# 표준오차
print("Standard errors:", results.bse)

# p-값
print("P-values:", results.pvalues)

# 신뢰구간
print("95% CI:\n", results.conf_int(alpha=0.05))

# 결정계수와 수정결정계수
print("R-squared:", results.rsquared)
print("Adjusted R-squared:", results.rsquared_adj)

# 잔차
residuals = results.resid

# 적합값
fitted = results.fittedvalues

# AIC and BIC
print("AIC:", results.aic)
print("BIC:", results.bic)
```

출력:

```
Coefficients: [2.04639669 3.09536017 1.41392895]
Standard errors: [0.05401682 0.06281005 0.05384654]
P-values: [6.18804800e-60 1.81607514e-70 7.11568582e-46]
95% CI:
 [[1.93918825 2.15360513]
 [2.9706996  3.22002073]
 [1.30705847 1.52079942]]
R-squared: 0.9706252436321817
Adjusted R-squared: 0.9700195785524329
AIC: 161.5973963941535
BIC: 169.41290695211777
```

`params`, `bse`, `pvalues`, `conf_int()`로 요약표의 각 열을 배열로 꺼낼 수 있다. 보고서를 자동 생성하거나 여러 모형을 비교할 때 이 접근이 필요하다.

</div>

---

## 5. 진단 메서드

`results` 객체는 모형 진단을 위한 메서드를 제공한다.

<div class="codebox" markdown>

**예제 4.** 진단 도구들

```python
# 진단 도구가 갖춰져 있다는 점이 statsmodels 를 쓰는 큰 이유다.
# sklearn 에는 이런 것이 아예 없다.
influence = results.get_influence()
cooks_d = influence.cooks_distance[0]
leverage = influence.hat_matrix_diag

# 등분산성 검정
from statsmodels.stats.diagnostic import het_breuschpagan
bp_stat, bp_pval, _, _ = het_breuschpagan(results.resid, results.model.exog)
print(f"Breusch-Pagan p-value: {bp_pval:.4f}")

# 잔차의 정규성 검정
from statsmodels.stats.stattools import jarque_bera
jb_stat, jb_pval, skew, kurtosis = jarque_bera(results.resid)
print(f"Jarque-Bera p-value: {jb_pval:.4f}")

# 다중공선성 점검. 상수항 열은 건너뛴다.
from statsmodels.stats.outliers_influence import variance_inflation_factor
for i in range(1, X_with_const.shape[1]):
    vif = variance_inflation_factor(X_with_const, i)
    print(f"VIF for variable {i}: {vif:.2f}")
```

출력:

```
Breusch-Pagan p-value: 0.7942
Jarque-Bera p-value: 0.1383
VIF for variable 1: 1.00
VIF for variable 2: 1.00
```

Breusch-Pagan과 Jarque-Bera 모두 기각하지 못하고 VIF도 1.00이다. 자료를 가정에 맞게 만들었으니 당연한 결과이며, 진단 도구가 제대로 작동한다는 확인이기도 하다.

</div>

!!! note "VIF 반복문에서 상수 열은 건너뛴다"
    `X_with_const`의 0번 열은 절편을 위한 상수이다. 여기에 `variance_inflation_factor`를 호출해도 오류는 나지 않지만 그 값에는 아무 의미가 없다. 위 코드처럼 `range(1, ...)`로 시작해 실제 설명변수만 다루어야 한다.

!!! tip "언제 statsmodels를 쓰는가"
    주된 목표가 통계적 추론일 때 `statsmodels`를 쓴다. 곧 계수에 대한 가설검정, 신뢰구간 구성, 모형 가정 진단이 목적일 때이다. 추론이 필요 없는 순수 예측 작업에는 `sklearn.linear_model.LinearRegression`이 더 간단한 인터페이스를 제공한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
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

</div>

??? success "풀이"

    **(a)** 매출의 예측값은

    $$
    \hat{y} = 3.0451 + 0.0470 \cdot x_1 + 0.1797 \cdot x_2 - 0.0030 \cdot x_3
    $$

    절편과 각 광고 매체의 계수를 결합하여 매출을 예측한다.

    **(b)** 신문 계수의 $p$값은 $0.669$로 표준 유의수준 0.05보다 훨씬 크다. 곧 계수가 0이라는 귀무가설을 기각하지 못한다. 계수가 음수이긴 하지만 $p$값이 크므로 이 결과는 통계적으로 의미가 없다. 신문 광고가 음의 효과를 갖는다고 해석하기보다는 매출에 통계적으로 유의한 영향을 주지 않는다고 결론짓는 것이 적절하다. 95% 신뢰구간 $(-0.017, 0.011)$이 0을 포함한다는 사실도 같은 이야기를 한다.

    **(c)** Jarque-Bera 검정은 잔차가 정규분포를 따르는지 평가한다. $p$값이 $3.00 \times 10^{-29}$로 극히 작으므로 귀무가설(잔차가 정규분포를 따른다)이 강하게 기각된다. 잔차가 정규분포를 따르지 **않을** 가능성이 높다는 뜻이며, 비정규 오차나 이상점의 존재 같은 모형 문제를 시사할 수 있다. 실제로 왜도 $-1.459$와 첨도 $6.741$은 정규분포의 값(0과 3)에서 크게 벗어나 있어, 왼쪽으로 치우치고 꼬리가 두꺼운 잔차 분포임을 보여준다.

---

## 정리하며

`statsmodels` 는 **추론**을 위한 도구다.

- **요약표가 핵심 산출물이다.** 계수·표준오차·$t$·$p$ 값·신뢰구간이 한 번에 나오며, 이 장에서 손으로 계산한 것들이 그대로 들어 있다.
- **절편을 직접 넣어야 한다.** `sm.add_constant()` 를 빠뜨리면 절편 없는 모형이 적합되며, **$R^2$ 의 정의까지 달라진다.** 가장 흔한 실수다.
- **수식 인터페이스가 편하다.** `ols("y ~ x1 + C(g)", data)` 는 절편을 자동으로 넣고 범주형을 더미로 바꿔 준다.
- **진단이 함께 제공된다.** 잔차, 영향 측도, 이분산 검정, 로버스트 표준오차(`cov_type='HC3'`)까지 모형 객체에서 바로 얻는다.
- **1장의 구분이 도구 선택으로 나타난다.** 계수를 해석할 생각이면 이쪽이다.

다음 절 **sklearn** 으로 넘어간다.
