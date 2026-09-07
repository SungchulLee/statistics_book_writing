# 교차검증 다항 모형선택

## 개요

이 페이지는 회귀에서 최적 다항 차수를 고르는 세 가지 재표본추출 접근 — 검증집합 방법, 하나 빼기 교차검증(LOOCV), $k$-겹 교차검증 — 을 보인다. 참 관계가 이차인 인공자료를 써서, 검증집합 방법은 변동이 크고, LOOCV는 편향이 작지만 계산 비용이 크며, $k$-겹 교차검증이 실용적인 균형을 준다는 점을 보인다.

## 수학적 배경

### 다항회귀

$d$차 다항 모형은

$$
y_i = \beta_0 + \beta_1 x_i + \beta_2 x_i^2 + \cdots + \beta_d x_i^d + \varepsilon_i.
$$

$d$를 키우면 훈련오차는 줄지만 검정오차는 커질 수 있다(과대적합). 모형선택의 목표는 기대 검정오차를 최소화하는 $d$를 찾는 것이다.

### 검증집합 방법

자료를 훈련집합과 검증집합으로 나눈다. 각 후보 모형을 훈련자료에 적합하고 검증자료에서 평가한다. 검증집합의 MSE가 검정오차를 추정한다. 단점은 추정값이 무작위 분할에 크게 좌우된다는 점이다.

### 하나 빼기 교차검증(LOOCV)

LOOCV는 관측값 하나씩을 빼는 $n$개의 겹을 쓴다.

$$
\mathrm{CV}_{(n)} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i^{(-i)})^2.
$$

선형모형에서는 모자행렬을 이용해 효율적으로 계산할 수 있다.

$$
\mathrm{CV}_{(n)} = \frac{1}{n}\sum_{i=1}^n \left(\frac{e_i}{1 - h_{ii}}\right)^2.
$$

### k-겹 교차검증

자료를 크기가 대략 같은 $K$개의 겹으로 나누고, $K-1$개 겹으로 훈련한 뒤 남겨 둔 겹에서 검정한다.

$$
\mathrm{CV}_{(K)} = \frac{1}{K}\sum_{k=1}^K \mathrm{MSE}_k.
$$

흔한 선택은 $K = 5$나 $K = 10$이다.

## 코드

### 검증집합 방법

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

np.random.seed(42)
n = 200
X = np.random.uniform(1, 10, n)
y = 5 + 2 * X - 0.3 * X**2 + np.random.normal(0, 2, n)
X_2d = X.reshape(-1, 1)

degrees = np.arange(1, 11)
n_validations = 20
val_mse_multiple = np.zeros((n_validations, len(degrees)))

for run in range(n_validations):
    val_size = int(0.2 * n)
    indices = np.random.permutation(n)
    train_idx, val_idx = indices[:-val_size], indices[-val_size:]
    for i, degree in enumerate(degrees):
        poly = PolynomialFeatures(degree)
        X_tr = poly.fit_transform(X_2d[train_idx])
        X_va = poly.transform(X_2d[val_idx])
        model = LinearRegression().fit(X_tr, y[train_idx])
        val_mse_multiple[run, i] = np.mean((y[val_idx] - model.predict(X_va)) ** 2)
```

20번의 무작위 분할이 고른 최적 차수는

```text
[5, 6, 3, 3, 2, 3, 8, 6, 2, 2, 9, 9, 2, 8, 9, 2, 3, 3, 3, 5]
```

로 2에서 9까지 흩어진다. 같은 자료인데도 분할 하나에 따라 결론이 완전히 달라진다. 검증집합 방법이 왜 불안정한지 그대로 보여준다.

### LOOCV

```python
from sklearn.model_selection import cross_val_score, LeaveOneOut

loo = LeaveOneOut()
loocv_mse = np.zeros(len(degrees))
for i, degree in enumerate(degrees):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X_2d)
    model = LinearRegression()
    scores = cross_val_score(model, X_poly, y, cv=loo,
                             scoring='neg_mean_squared_error')
    loocv_mse[i] = -scores.mean()
```

### k-겹 교차검증

```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
kfold_mse = np.zeros(len(degrees))
for i, degree in enumerate(degrees):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X_2d)
    model = LinearRegression()
    scores = cross_val_score(model, X_poly, y, cv=kfold,
                             scoring='neg_mean_squared_error')
    kfold_mse[i] = -scores.mean()
```

### 결과

| 차수 | LOOCV MSE | 10-겹 MSE |
|---|---|---|
| 1 | 6.5243 | 6.4595 |
| 2 | 3.8318 | 3.8361 |
| **3** | **3.8123** | **3.7948** |
| 4 | 3.8574 | 3.8162 |
| 5 | 3.8870 | 3.8284 |
| 6 | 3.9246 | 3.8642 |
| 7 | 3.9769 | 3.9221 |
| 8 | 4.0359 | 4.0216 |
| 9 | 4.0686 | 4.0074 |
| 10 | 4.1443 | 4.0964 |

두 방법 모두 형식적으로는 차수 3에서 최솟값을 갖지만, 차수 2와 3의 차이는 0.5%에 지나지 않아 사실상 동률이다. 참 차수가 2인 점을 생각하면 이는 자연스러운 결과이다. 삼차항이 잡음을 조금 더 맞춰 준 것뿐이며, 이런 상황에서는 [1 표준오차 규칙](../model_selection/cross_validation.md)에 따라 더 단순한 차수 2를 고르는 것이 옳다.

차수 1에서 2로 갈 때 MSE가 $6.52$에서 $3.83$으로 절반 가까이 떨어지는 반면 그 뒤로는 거의 평평하다가 서서히 올라간다는 점이 중요하다. 곡률을 포착하는 것이 결정적이고, 그 이상은 과대적합이라는 뜻이다.

## 해석

- **검증집합**: 빠르지만 불안정하다. 무작위 분할이 달라지면 다른 최적 차수를 고를 수 있어 한 번의 분할로는 신뢰할 수 없다.
- **LOOCV**: 거의 불편이지만(각 훈련집합이 $n-1$개의 관측값을 쓴다) 계산이 비싸고(후보마다 $n$번 적합) $n$개의 훈련집합이 거의 동일하므로 분산이 클 수 있다.
- **10-겹 교차검증**: 실용적인 절충이다. 각 훈련집합이 자료의 $90\%$를 쓰고(편향이 작고) 10개 겹의 평균이 분산을 줄인다.
- 참 관계가 이차($d = 2$)이므로 세 방법 모두 차수 2 또는 그 근처를 고른다. 차수가 높아지면 과대적합이 일어나 최솟값 이후 검정오차 곡선이 올라간다.

## 연습문제

**연습문제 1.** 모자행렬 지름길로 $n$개 모형을 다시 적합하지 않고 2차 다항회귀의 LOOCV MSE를 계산하라. 무차별 계산 결과와 일치하는지 확인하라.

??? success "풀이"

    ```python
    poly = PolynomialFeatures(2)
    X_poly = poly.fit_transform(X_2d)
    model = LinearRegression().fit(X_poly, y)
    H = X_poly @ np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T
    e = y - model.predict(X_poly)
    h = np.diag(H)
    loocv_shortcut = np.mean((e / (1 - h)) ** 2)
    print(f"LOOCV (shortcut): {loocv_shortcut:.4f}")
    print(f"LOOCV (brute):    {loocv_mse[1]:.4f}")
    ```

    두 값 모두 $3.8318$로 일치한다. 지름길 공식이 선형모형에서 하나 빼기 재적합과 대수적으로 동등하기 때문이다. $\square$

---

**연습문제 2.** $n$을 200에서 1000으로 늘려라. 검증집합 방법의 변동성과 LOOCV·10-겹 사이의 격차는 어떻게 달라지는가?

??? success "풀이"

    $n = 1000$이면 (1) 훈련집합과 검증집합이 모두 커져 특정 분할에 대한 민감도가 줄어들므로 검증집합 방법이 더 안정된다. (2) 10-겹 교차검증의 편향(자료의 $90\%$만 훈련에 쓰는 데서 오는)이 $n$이 클 때 무시할 만해지므로 LOOCV와의 격차가 좁아진다. 두 방법이 참 검정오차에 대해 비슷한 추정값으로 수렴한다. $\square$

---

**연습문제 3.** (10-겹 교차검증으로 얻은) 검정 MSE와 함께 훈련 MSE를 다항 차수의 함수로 그려라. 그림에 드러나는 편향-분산 절충을 설명하라.

??? success "풀이"

    ```python
    train_mse = []
    for degree in degrees:
        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(X_2d)
        model = LinearRegression().fit(X_poly, y)
        train_mse.append(np.mean((y - model.predict(X_poly)) ** 2))
    ```

    훈련 MSE는 차수에 따라 단조롭게 감소한다(모수가 많아지면 훈련자료를 언제나 더 잘 맞춘다). 검정 MSE는 처음에 감소했다가(편향 감소) 이후 증가한다(분산 증가). 최적 차수는 검정 MSE가 최소가 되는 지점이다. 이 U자 모양의 검정오차 곡선이 편향-분산 절충의 상징적인 모습이다. $\square$

---

**연습문제 4.** 계수 1의 갱신에 대한 Sherman-Morrison-Woodbury 공식에서 $\mathrm{CV}_{(n)} = \frac{1}{n}\sum_i \left(\frac{e_i}{1 - h_{ii}}\right)^2$을 유도하라.

??? success "풀이"

    관측값 $i$를 지우는 것은 $\mathbf{X}^\top\mathbf{X}$의 계수 1 갱신과 같다. Sherman-Morrison 공식에 의해

    $$
    (\mathbf{X}_{(-i)}^\top\mathbf{X}_{(-i)})^{-1} = (\mathbf{X}^\top\mathbf{X} - \mathbf{x}_i\mathbf{x}_i^\top)^{-1} = (\mathbf{X}^\top\mathbf{X})^{-1} + \frac{(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{x}_i\mathbf{x}_i^\top(\mathbf{X}^\top\mathbf{X})^{-1}}{1 - h_{ii}}.
    $$

    정리하면 관측값 $i$의 하나 빼기 예측오차가 $y_i - \hat{y}_i^{(-i)} = e_i/(1 - h_{ii})$로 간단해지고 공식이 나온다. $\square$

---

**연습문제 5.** $K$-겹 교차검증에서 $K$의 선택에 편향-분산 절충이 있는 이유를 설명하라. 극단인 $K = 2$와 $K = n$의 경우는 어떠한가?

??? success "풀이"

    $K$가 작으면(예: $K = 2$) 각 훈련집합이 자료의 $50\%$만 쓰므로 오차 추정값에 위쪽 편향이 생긴다(적은 자료로 훈련한 모형은 성능이 나쁘다). 그러나 $K$개의 추정값이 겹치지 않는 훈련집합에서 나오므로 분산은 작다. $K = n$(LOOCV)이면 각 훈련집합이 $n-1$개의 관측값을 쓰므로 편향이 최소이지만, $n$개의 훈련집합이 거의 완전히 겹쳐 추정값들이 강하게 상관되고 분산이 클 수 있다. $K = 5$나 $K = 10$이 이 양극단의 균형을 잡는다. $\square$
