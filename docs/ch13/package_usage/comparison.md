# sklearn과 statsmodels 비교

## 개요

Python에는 회귀 모형화를 위한 주요 라이브러리가 둘 있다. **scikit-learn**(`sklearn`)과 **statsmodels**이다. 둘은 목적이 다르며 서로 다른 작업 흐름에 적합하다.

---

## 한눈에 보기

| 기능 | `statsmodels` | `sklearn` |
|---|---|---|
| 주된 초점 | 통계적 추론 | 예측과 기계학습 |
| 모형 요약 | 상세함(계수, $p$값, $R^2$, AIC, BIC) | 최소한(직접 계산해야 함) |
| 가설검정 | 내장($t$ 검정, $F$ 검정, Wald 검정) | 없음 |
| 신뢰구간 | 계수와 예측에 대해 내장 | 없음 |
| 진단 | 풍부함(VIF, 영향 그림, 잔차 검정) | 제한적 |
| 교차검증 | 내장되어 있지 않음 | 내장(`cross_val_score`, 파이프라인) |
| 정칙화 | 제한적 | 릿지, 라쏘, 엘라스틱넷 내장 |
| 절편 처리 | `add_constant()`로 직접 추가해야 함 | 자동(기본 `fit_intercept=True`) |
| 식 인터페이스 | 있음(`smf.ols('y ~ x1 + x2', data=df)`) | 없음 |

---

## 각각을 언제 쓰는가

### `statsmodels`를 쓸 때:

- **계수 추론**이 필요할 때: $p$값, 신뢰구간, 유의성 검정.
- **모형 진단**을 수행할 때: 잔차 분석, 이분산 검정, 다중공선성 확인.
- **AIC나 BIC**로 모형을 비교하고 싶을 때.
- 보고를 위한 **상세한 회귀 요약표**가 필요할 때.
- **고전 통계학**이나 **계량경제학** 맥락에서 작업할 때.

### `sklearn`을 쓸 때:

- 주된 목표가 **예측**과 새 자료로의 일반화일 때.
- **교차검증**과 **훈련/검정 분할**이 필요할 때.
- 전처리 단계(척도화, 부호화, 특성 선택)를 포함한 **파이프라인**을 만들 때.
- **정칙화 모형**(릿지, 라쏘, 엘라스틱넷)이 필요할 때.
- 여러 모형 유형(회귀, 분류, 군집)에 걸쳐 일관된 API를 원할 때.

---

## 나란히 놓고 보기

### statsmodels

```python
import statsmodels.api as sm
import pandas as pd

X = sm.add_constant(df[['RM', 'LSTAT', 'PTRATIO']])
y = df['PRICE']

model = sm.OLS(y, X).fit()
print(model.summary())

# Access specific results
print(f"R²: {model.rsquared:.4f}")
print(f"Adj. R²: {model.rsquared_adj:.4f}")
print(f"AIC: {model.aic:.2f}")
print(f"BIC: {model.bic:.2f}")
print(model.pvalues)
print(model.conf_int())
```

### sklearn

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

X = df[['RM', 'LSTAT', 'PTRATIO']]
y = df['PRICE']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
print(f"R²: {r2_score(y, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_:.4f}")

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

---

## 둘을 함께 쓰기

실무에서 많은 분석가가 한 프로젝트에서 두 라이브러리를 모두 쓴다.

1. `statsmodels`로 **탐색하고 진단한다**: OLS 모형을 적합하고, 요약을 살피고, VIF를 확인하고, 잔차 가정을 검정한다.
2. `sklearn`으로 **예측하고 검증한다**: 교차검증, 정칙화 모형, 파이프라인으로 배포 가능한 예측을 만든다.

```python
# Step 1: Statistical analysis with statsmodels
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_sm = sm.add_constant(df[['RM', 'LSTAT', 'PTRATIO']])
model_sm = sm.OLS(df['PRICE'], X_sm).fit()
print(model_sm.summary())

# Check VIF (skip column 0: the constant)
vif = pd.DataFrame({
    'Feature': X_sm.columns[1:],
    'VIF': [variance_inflation_factor(X_sm.values, i) for i in range(1, X_sm.shape[1])]
})
print(vif)

# Step 2: Prediction pipeline with sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])

X_sk = df[['RM', 'LSTAT', 'PTRATIO']]
cv_scores = cross_val_score(pipe, X_sk, df['PRICE'], cv=5, scoring='r2')
print(f"Ridge CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

!!! note "VIF 계산에서 상수 열 제외하기"
    `sm.add_constant`가 만든 0번 열은 절편을 위한 상수이므로 그에 대한 VIF는 의미가 없다. 위 코드처럼 `range(1, ...)`로 실제 설명변수만 계산해야 한다.

---

## 요약

`statsmodels`는 통계적 추론과 진단에, `sklearn`은 예측과 모형 배포에 뛰어나다. 두 라이브러리는 상호보완적이며, 회귀 작업 흐름에서 둘을 함께 쓰면 가장 완전한 분석을 얻을 수 있다.

## 연습문제

**연습문제 1.**
어떤 데이터 과학자가 선형회귀 모형을 적합하고 각 계수의 p값, 95% 신뢰구간, 종합적인 모형 요약을 얻어야 한다. `sklearn`과 `statsmodels` 가운데 무엇을 써야 하는가? 답을 정당화하라.

??? success "풀이"
    **`statsmodels`**가 명백한 선택이다. `OLS` 클래스의 `.summary()` 메서드는 계수 추정값, 표준오차, $t$ 통계량, p값, 신뢰구간, $R^2$, 수정 $R^2$, $F$ 통계량, AIC, BIC, 잔차 진단을 하나의 출력에 담아 준다.

    `sklearn`의 `LinearRegression`은 어떤 추론 통계량도 제공하지 않는다(p값도, 표준오차도, 신뢰구간도 없다). 추론이 아니라 예측을 위해 설계되었기 때문이다.

---

**연습문제 2.**
교차검증과 파이프라인 같은 기법을 쓸 때, 예측 모형을 만드는 데 `sklearn`이 `statsmodels`보다 나은 점 하나를 설명하라.

??? success "풀이"
    `sklearn`은 `.fit()`, `.predict()`, `.score()` 메서드로 이루어진 일관된 API를 제공하며, 이는 그 생태계의 도구들과 매끄럽게 통합된다. 교차검증을 위한 `cross_val_score`, 전처리와 모형화 단계를 엮는 `Pipeline`, 초모수 조정을 위한 `GridSearchCV`, 특성 척도화를 위한 `StandardScaler` 등이 그것이다.

    `statsmodels`에는 이런 표준화된 인터페이스가 없고 교차검증 파이프라인을 기본으로 지원하지 않으므로, 예측 모형화 작업 흐름에서 모형선택과 평가를 하기에는 번거롭다.
