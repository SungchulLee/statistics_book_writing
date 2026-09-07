# 계단함수

## 개요

이 페이지는 비선형 모형화 기법으로서 계단함수 회귀(조각별 상수 적합)를 보인다. King County 주택 자료를 써서 구간화한 설명변수로 지시변수를 만들고, 계단함수 모형을 적합하고, 구간 개수를 달리해 비교하며, 계단함수를 선형·다항·스플라인 대안과 대조한다. 분류를 위한 로지스틱 회귀에 계단함수를 적용하는 방법도 다룬다.

## 수학적 배경

계단함수는 설명변수 $x$의 범위를 절단점 $c_1 < c_2 < \cdots < c_{K-1}$로 $K$개 구간으로 나누고 지시변수를 만든다.

$$
C_k(x) = \mathbf{1}(c_{k-1} < x \leq c_k), \qquad k = 1, \ldots, K.
$$

계단함수 회귀모형은

$$
y_i = \beta_0 + \beta_1 C_1(x_i) + \beta_2 C_2(x_i) + \cdots + \beta_K C_K(x_i) + \varepsilon_i.
$$

이는 각 구간 안에서 별도의 상수(구간 평균)를 적합하는 것과 같다. 구간 $k$에 속한 임의의 $x$에 대한 적합값은 (지시변수를 하나도 빼지 않고 모두 넣었을 때) 그 구간의 평균 반응 $\bar{y}_k$가 된다.

### 절단점 정하기

절단점은 다음 기준으로 정할 수 있다.

- **등간격 구간**: 범위를 $K$개의 같은 구간으로 나눈다
- **분위수 구간**(`pd.qcut`): 각 구간이 대략 같은 개수의 관측값을 담는다
- **분야 지식**: 의미 있는 문턱에 절단점을 둔다

## 코드

### pd.cut으로 계단함수 만들기

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

age = 2024 - house['YrBuilt'].values
price = house['AdjSalePrice'].values
df = pd.DataFrame({'age': age, 'price': price})

# Define breakpoints
knots = [0, 20, 40, 60, 80, 150]
df['age_bin'] = pd.cut(df['age'], bins=knots, include_lowest=True)

# Create dummy variables
df_dummies = pd.get_dummies(df['age_bin'], drop_first=False)

# Fit step function regression
X_step = df_dummies.values
step_model = LinearRegression()
step_model.fit(X_step, df['price'])
```

!!! note "지시변수를 모두 넣으면 설계행렬의 계수가 부족해진다"
    `drop_first=False`로 만든 $K$개의 지시변수는 합이 1이므로 절편과 완전 공선이다. `LinearRegression`은 내부적으로 최소제곱 최소노름 해를 쓰므로 **예측값 자체는 정확하지만** 개별 계수를 해석할 수는 없다. 계수를 해석하려면 `drop_first=True`로 기준 구간을 하나 두거나, 절편 없이(`fit_intercept=False`) $K$개 지시변수만 쓰면 된다. 뒤쪽 방식에서는 각 계수가 그 구간의 평균이 된다(연습문제 4 참조).

### 구간 개수 비교

```python
results = []
for n_bins in [3, 4, 5, 6, 8, 10]:
    df['bin_temp'] = pd.qcut(df['age'], q=n_bins, duplicates='drop')
    X_temp = pd.get_dummies(df['bin_temp'], drop_first=False).values
    model = LinearRegression().fit(X_temp, df['price'])
    pred = model.predict(X_temp)
    r2 = r2_score(df['price'], pred)
    rmse = np.sqrt(mean_squared_error(df['price'], pred))
    results.append({'n_bins': n_bins, 'R2': r2, 'RMSE': rmse})
```

### 다른 방법과의 비교

```python
# Linear
linear_model = LinearRegression()
linear_model.fit(df[['age']].values, df['price'])

# Polynomial (degree 3)
X_poly = np.column_stack([df['age'] ** i for i in range(1, 4)])
poly_model = LinearRegression().fit(X_poly, df['price'])
```

## 해석

- **계단함수**는 조각별 상수이다. 한 구간 안의 모든 관측값에 대해 예측값이 같다. 그래서 구간 경계에서 불연속인 "계단" 모양의 적합값이 나온다.
- **장점**: 해석이 단순하고(예측값이 곧 구간 평균이다), 구현이 쉬우며, 국면 전환을 포착할 수 있다.
- **단점**: 경계에서 불연속이고(매끄러운 전이가 없고), 자유도를 낭비하며(구간마다 상수 하나에 모수 하나를 쓴다), 구간 배치에 민감하다.
- **비교**: 모수 개수가 같을 때 계단함수는 대체로 다항이나 스플라인 모형보다 $R^2$가 낮다. 스플라인과 다항식은 매끄러움을 통해 이웃 구역끼리 정보를 공유하기 때문이다.
- **분류**: 로그오즈가 특정 설명변수 값에서 급격히 변할 때 계단함수는 로지스틱 회귀에서도 잘 작동한다.

## 연습문제

**연습문제 1.** 분위수 기반 구간($K = 5$)으로 계단함수를 적합하고 등간격 구간의 구간 평균과 비교하라. 어느 방식이 더 균일한 예측 품질을 주는가?

??? success "풀이"

    ```python
    # Quantile bins
    df['q_bin'] = pd.qcut(df['age'], q=5)
    q_means = df.groupby('q_bin')['price'].agg(['mean', 'count'])

    # Equal-width bins
    df['w_bin'] = pd.cut(df['age'], bins=5)
    w_means = df.groupby('w_bin')['price'].agg(['mean', 'count'])
    ```

    분위수 구간은 각 구간의 관측값 개수를 대략 같게 만들어 구간 평균의 표준오차를 더 균일하게 한다. 등간격 구간은 극단 구간(예: 아주 오래된 집)에 관측값이 아주 적어 그 구간의 추정을 믿을 수 없게 될 수 있다. $\square$

---

**연습문제 2.** 구간 개수를 늘리면 훈련 $R^2$가 항상 커지거나(적어도 줄지 않고) 검정오차는 커질 수 있는 이유를 설명하라.

??? success "풀이"

    구간이 하나 늘 때마다 모수도 하나 늘어난다. 구간이 많아지면 모형이 훈련자료의 더 세밀한 패턴을 맞출 수 있어 훈련 RSS가 줄고 $R^2$가 커진다. 그러나 구간이 너무 많으면(특히 관측값이 적은 구간에서) 구간 평균이 참 조건부 기댓값의 잡음 섞인 추정이 된다. 표본 밖에서는 이 잡음 섞인 추정이 예측오차를 키운다. 극단적으로 $K = n$개 구간이면 훈련자료에서 $R^2 = 1$이지만 표본 밖 성능은 형편없다. $\square$

---

**연습문제 3.** `pd.cut` 대신 `np.digitize`로 계단함수를 구현하라. 결과가 일치하는지 확인하라.

??? success "풀이"

    ```python
    bin_edges = [0, 20, 40, 60, 80, 150]
    bin_indices = np.digitize(df['age'], bin_edges)
    # Create dummy matrix
    K = len(bin_edges) - 1
    X_digit = np.zeros((len(df), K))
    for k in range(K):
        X_digit[:, k] = (bin_indices == k + 1).astype(float)
    model_digit = LinearRegression().fit(X_digit, df['price'])
    ```

    두 방법이 같은 지시행렬을 만들므로 예측값이 일치한다. 다만 경계값 처리가 미묘하게 다르다. `np.digitize`는 기본적으로 왼쪽이 닫힌 구간 $[c_{k-1}, c_k)$을 쓰고 `pd.cut`은 오른쪽이 닫힌 구간 $(c_{k-1}, c_k]$을 쓴다. 절단점과 정확히 같은 값이 자료에 있다면 두 방법의 배정이 달라진다. `pd.cut`과 맞추려면 `np.digitize(..., right=True)`를 쓴다. $\square$

---

**연습문제 4.** (모든 지시변수를 넣고 절편은 없는) 계단함수 모형에서 각 구간 계수의 OLS 추정값이 그 구간 반응변수의 표본평균과 같음을 수학적으로 보여라.

??? success "풀이"

    $\mathbf{X} = [C_1 \mid C_2 \mid \cdots \mid C_K]$라 하고 $C_k$를 구간 $k$의 지시벡터라 하자. 구간이 서로 겹치지 않으므로 $\mathbf{X}^\top\mathbf{X} = \mathrm{diag}(n_1, n_2, \ldots, n_K)$이고 $\mathbf{X}^\top\mathbf{y} = (\sum_{i \in B_1} y_i, \ldots, \sum_{i \in B_K} y_i)^\top$이다. 따라서

    $$
    \hat{\boldsymbol{\beta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y} = \left(\frac{\sum_{i \in B_1} y_i}{n_1}, \ldots, \frac{\sum_{i \in B_K} y_i}{n_K}\right)^\top = (\bar{y}_1, \ldots, \bar{y}_K)^\top.
    $$

    각 계수가 그 구간 안의 표본평균이다. $\square$

---

**연습문제 5.** "비싼" 주택을 예측하는 계단함수 로지스틱 회귀를, 연속형 age를 쓰는 로지스틱 회귀와 비교하라. 어느 쪽의 분류 정확도가 높으며 그 이유는 무엇인가?

??? success "풀이"

    ```python
    from sklearn.linear_model import LogisticRegression

    # Continuous predictor
    logit_cont = LogisticRegression().fit(df[['age']], df['expensive'])
    acc_cont = logit_cont.score(df[['age']], df['expensive'])

    # Step function predictor
    X_class = pd.get_dummies(pd.cut(df['age'], bins=5), drop_first=True).values
    logit_step = LogisticRegression().fit(X_class, df['expensive'])
    acc_step = logit_step.score(X_class, df['expensive'])
    ```

    참 로그오즈가 age에 대해 대략 선형이면 연속형 모형이 더 적은 모수로 비슷하거나 더 나은 성능을 낼 수 있다. 로그오즈가 급격히 변한다면(예: 60년이 넘은 집은 비쌀 가능성이 훨씬 낮다면) 계단함수가 이를 더 잘 포착한다. 상대적 성능은 참 관계가 어떤 모양인지에 달려 있다.

    다만 두 정확도 모두 **훈련자료**에서 계산된 것임에 유의하라. 계단함수 쪽이 모수가 더 많으므로 훈련 정확도에서 유리할 수밖에 없다. 공정한 비교를 하려면 교차검증을 써야 한다. $\square$
