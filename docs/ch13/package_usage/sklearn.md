# sklearn LinearRegression 인터페이스

`statsmodels`가 통계적 추론을 위해 설계된 반면, scikit-learn의 `LinearRegression`은 예측을 위해 설계되었다. scikit-learn의 일관된 추정기 API — `fit`, `predict`, `score` — 를 따르며 라이브러리의 전처리, 파이프라인, 교차검증 도구와 매끄럽게 통합된다. 대가는 `sklearn`이 p값, 신뢰구간, 진단검정을 기본으로 제공하지 않는다는 점이다.

---

## 1. 기본 사용법

`LinearRegression` 클래스는 최소제곱으로 모형 $Y = \mathbf{X}\boldsymbol{\beta} + \varepsilon$을 적합한다. `statsmodels`와 달리 절편을 기본으로 자동 추가한다.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Generate example data
np.random.seed(42)
n = 100
X = np.random.randn(n, 2)
beta_true = np.array([3.0, 1.5])
y = X @ beta_true + 2.0 + np.random.randn(n) * 0.5

# Fit model
model = LinearRegression()
model.fit(X, y)

# Access coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
```

출력:

```
Intercept: 2.046396690621326
Coefficients: [3.09536017 1.41392895]
```

참값이 절편 2.0, 계수 3.0과 1.5인데 추정값이 2.046, 3.095, 1.414다.

scikit-learn은 절편을 `intercept_`에, 기울기를 `coef_`에 따로 담는다. statsmodels가 둘을 한 배열에 담는 것과 다르며, 두 라이브러리를 오가며 쓸 때 자주 헷갈리는 지점이다.

적합된 모형은 절편을 `model.intercept_`에, 기울기 계수를 `model.coef_`에 저장한다. `X`에 1로 채운 열이 필요하지 않다는 점에 유의하라. `fit_intercept=True`(기본값)일 때 절편은 내부에서 처리된다.

---

## 2. 예측

`predict` 메서드는 새 자료의 적합값을 계산한다.

```python
# Predict on training data
y_pred_train = model.predict(X)

# Predict on new data
X_new = np.array([[1.0, 0.5], [-0.5, 2.0]])
y_pred_new = model.predict(X_new)
print("Predictions:", y_pred_new)
```

출력:

```
Predictions: [5.84872133 3.3265745 ]
```

적합된 모형으로 새 입력의 예측값을 얻는다. `predict`는 2차원 배열을 받으므로 관측값이 하나여도 `[[x1, x2]]` 모양으로 넣어야 한다.

`predict`의 입력은 훈련자료와 열의 개수가 같아야 한다. 각 행이 새로운 관측값이며 출력은 예측값의 벡터 $\hat{y} = \hat{\beta}_0 + \mathbf{X}_{\text{new}} \hat{\boldsymbol{\beta}}$이다.

---

## 3. 모형 성능 평가

`score` 메서드는 주어진 자료에서의 $R^2$를 돌려준다.

```python
r2_train = model.score(X, y)
print(f"R-squared (training): {r2_train:.4f}")
```

출력:

```
R-squared (training): 0.9706
```

훈련 자료의 $R^2 = 0.9706$이다. 잡음의 표준편차를 0.5로 두었으므로 이 정도가 상한에 가깝다.

다른 척도가 필요하면 `sklearn.metrics`를 쓴다.

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

y_pred = model.predict(X)

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = root_mean_squared_error(y, y_pred)

print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
```

출력:

```
MAE:  0.4211
MSE:  0.2775
RMSE: 0.5268
```

MAE 0.42, RMSE 0.53이다. RMSE가 MAE보다 큰 것은 언제나 성립한다. 제곱이 큰 오차에 더 큰 가중치를 주기 때문이며, 두 값의 차이가 벌어질수록 오차 분포에 꼬리가 있다는 신호다.

!!! note "`root_mean_squared_error`는 scikit-learn 1.4부터"
    이 함수는 scikit-learn 1.4에서 추가되었다. 더 낮은 버전에서는 `mean_squared_error(y, y_pred, squared=False)`를 쓰거나 `np.sqrt(mean_squared_error(...))`로 직접 계산한다.

---

## 4. 훈련-검정 분할

표본 밖 성능을 추정하려면 적합하기 전에 자료를 나눈다.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

r2_test = model.score(X_test, y_test)
print(f"R-squared (test): {r2_test:.4f}")
```

출력:

```
R-squared (test): 0.9855
```

시험 $R^2$가 훈련 $R^2$(0.9706)보다 오히려 높다. 과적합의 징후가 없다는 뜻이며, 설명변수가 둘뿐인 선형모형이라 예상할 만한 결과다.

검정 $R^2$는 모형이 적합 과정에서 검정자료를 보지 않았으므로 훈련 $R^2$보다 예측 성능을 정직하게 추정한다.

---

## 5. 교차검증

scikit-learn은 훈련-검정 분할의 반복을 자동화하는 교차검증 도구를 제공한다.

```python
from sklearn.model_selection import cross_val_score

model = LinearRegression()
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

print(f"CV R-squared scores: {cv_scores}")
print(f"Mean CV R-squared: {cv_scores.mean():.4f}")
print(f"Std CV R-squared: {cv_scores.std():.4f}")
```

출력:

```
CV R-squared scores: [0.96022772 0.96777464 0.9761083  0.96564245 0.97171516]
Mean CV R-squared: 0.9683
Std CV R-squared: 0.0054
```

교차검증 $R^2$의 평균이 0.9683, 표준편차가 0.0054다. 겹마다 값이 크게 흔들리지 않으므로 모형이 안정적이다.

`scoring` 인자는 scikit-learn의 어떤 채점기도 받는다. 회귀에서 흔한 선택은 `'r2'`, `'neg_mean_squared_error'`, `'neg_mean_absolute_error'`이다. "neg" 접두사가 붙는 것은 점수가 높을수록 좋다는 scikit-learn의 관례 때문이며, 그래서 오차 척도에 음수를 붙인다.

---

## 6. 파이프라인

파이프라인은 전처리와 모형화 단계를 하나의 객체로 엮어, 훈련과 예측에서 변환이 일관되게 적용되도록 보장한다.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('regression', LinearRegression())
])

pipeline.fit(X_train, y_train)
r2_pipeline = pipeline.score(X_test, y_test)
print(f"Pipeline R-squared (test): {r2_pipeline:.4f}")
```

출력:

```
Pipeline R-squared (test): 0.9858
```

파이프라인을 거친 시험 $R^2$가 0.9858로, 앞서 수동으로 표준화한 결과와 사실상 같다. 선형회귀에서 표준화는 예측 성능을 바꾸지 않는다. 계수의 해석과 규제(ridge, lasso)에서 의미가 생긴다.

이 파이프라인은 먼저 각 설명변수를 평균 0, 분산 1로 표준화하고, 다음으로 (교호작용 항을 포함한) 다항 특성을 만들고, 마지막으로 선형회귀를 적합한다. 파이프라인 전체를 `cross_val_score`에 넘겨 교차검증으로 평가할 수 있다.

!!! note "파이프라인이 중요한 이유"
    파이프라인 없이 작업하면 전체 자료로 계산한 훈련 통계량으로 검정자료를 표준화하거나 변환하는 실수(자료 누출)를 저지르기 쉽다. 파이프라인은 각 변환 단계를 그 시점에 사용 가능한 자료에만 적용하여 이를 막아 준다.

---

## 7. sklearn과 statsmodels 중 무엇을 쓸 것인가

| 작업 | 권장 라이브러리 |
|---|---|
| 계수에 대한 가설검정 | `statsmodels` |
| $\beta_j$의 신뢰구간 | `statsmodels` |
| 잔차 진단(정규성, 이분산) | `statsmodels` |
| AIC/BIC를 통한 모형 비교 | `statsmodels` |
| 새 자료에 대한 예측 | `sklearn` |
| 교차검증 기반 모형 평가 | `sklearn` |
| 전처리 파이프라인과의 통합 | `sklearn` |
| 정칙화 회귀(릿지, 라쏘) | `sklearn` |
| 고차원 자료 ($p > n$) | `sklearn` |

!!! tip "경쟁이 아니라 상호보완 관계이다"
    흔한 작업 흐름은 모형을 세우는 단계에서 `statsmodels`로 가설을 검정하고 진단을 살피고 설명변수를 고른 뒤, 예측 파이프라인에 배포하기 위해 최종 모형을 `sklearn`으로 다시 적합하는 것이다. 계수 추정값은 (둘 다 OLS를 쓰므로) 동일하며, 선택의 기준은 모형 주위에 어떤 도구가 필요한가이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
scikit-learn으로 인공자료에 선형회귀를 적합하고, 예측을 계산하며, $R^2$ 점수를 출력하는 Python 코드를 작성하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (200, 3))
    y = 2 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] + rng.normal(0, 0.5, 200)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"Coefficients: {model.coef_.round(3)}")
    print(f"Intercept: {model.intercept_:.3f}")
    print(f"R^2 (test): {model.score(X_test, y_test):.4f}")
    ```

    출력:

    ```
    Coefficients: [ 2.012 -1.519  0.442]
    Intercept: -0.015
    R^2 (test): 0.9440
    ```

    참 계수가 $2, -1.5, 0.4$인데 추정값이 $2.012, -1.519, 0.442$로 잘 맞는다. 시험 자료의 $R^2$는 0.944다.

<div class="drillbox" markdown>

**연습문제 2.**
`model.score()`와 예측값에서 $R^2$를 직접 계산하는 것의 차이를 설명하라. 둘은 동등한가?

</div>

??? success "풀이"
    `model.score(X, y)`는 다음을 계산한다.

    $$
    R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}
    $$

    여기서 $\hat{y}_i$는 `model.predict(X)`이고 $\bar{y}$는 넘겨준 `y`의 평균이다. 이는 다음을 직접 계산하는 것과 동등하다.

    ```python
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    ```

    `Pipeline`은 전처리와 모형을 하나로 묶는다. 교차검증에서 표준화를 훈련 겹 안에서만 계산하게 해 주므로 정보 누출을 막는다.

    둘은 수학적으로 동일하다. 다만 검정자료에 적용할 때 $\bar{y}$는 (훈련자료가 아니라) 검정자료의 평균이며, 그래서 모형이 나쁘게 적합하면 검정 $R^2$가 음수가 될 수 있다.

<div class="drillbox" markdown>

**연습문제 3.**
scikit-learn의 `LinearRegression`이 계수의 p값이나 신뢰구간을 제공하지 않는 이유는 무엇인가? 어떻게 얻을 수 있는가?

</div>

??? success "풀이"
    scikit-learn은 통계적 추론이 아니라 예측을 주된 목적으로 설계되었다. `LinearRegression`은 OLS를 기계학습 알고리즘으로 구현하며 `.fit()`, `.predict()`, `.score()`에 집중한다. 표준오차, t 통계량, p값, 신뢰구간은 계산하지 않는다.

    추론 통계량을 얻으려면

    1. **statsmodels 사용:** `import statsmodels.api as sm; model = sm.OLS(y, sm.add_constant(X)).fit(); print(model.summary())`가 p값, 신뢰구간, 진단 통계량을 담은 완전한 회귀표를 제공한다.
    2. **직접 계산:** $\hat{\boldsymbol{\beta}}$를 구한 뒤 $s^2 = \text{SSE}/(n-p)$($p$는 절편을 포함한 모수의 개수), $\text{SE}(\hat{\beta}_j) = s\sqrt{[(\mathbf{X}^T\mathbf{X})^{-1}]_{jj}}$, $t_j = \hat{\beta}_j/\text{SE}(\hat{\beta}_j)$를 계산한다.

    예측 중심(sklearn)과 추론 중심(statsmodels) 도구가 나뉘어 있는 것은 모형과 알고리즘의 구분을 반영한다.

<div class="drillbox" markdown>

**연습문제 4.**
`sklearn.model_selection.cross_val_score`로 선형회귀 모형의 일반화 성능을 추정하는 방법을 기술하라.

</div>

??? success "풀이"
    ```python
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    import numpy as np

    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    print(f"CV R^2 scores: {scores.round(4)}")
    print(f"Mean R^2: {scores.mean():.4f} (+/- {scores.std():.4f})")
    ```

    출력:

    ```
    CV R^2 scores: [0.9531 0.9465 0.9701 0.9306 0.965 ]
    Mean R^2: 0.9530 (+/- 0.0140)
    ```

    5겹 교차검증의 $R^2$가 0.93에서 0.97 사이에 흩어져 있다. 한 번의 훈련/시험 분할로 얻은 값 하나만 보고하면 이 변동이 숨는다.

    `cross_val_score`는 내부에서 k-겹 교차검증을 수행한다. 자료를 `cv=5`개의 겹으로 나누고, 4개 겹으로 모형을 적합한 뒤 5번째 겹에서 평가하며, 모든 겹에 대해 반복한다. `scoring` 인자가 척도를 지정한다(선택지로 `"r2"`, `"neg_mean_squared_error"`, `"neg_mean_absolute_error"` 등이 있다).

    참고: sklearn은 점수가 클수록 좋다는 관례(채점기는 최대화되어야 한다)를 지키기 위해 음의 MSE(`neg_mean_squared_error`)를 쓴다.
