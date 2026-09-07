# 주성분회귀와 부분최소제곱 예제

## 개요

주성분회귀(PCR)와 부분최소제곱(PLS)은 회귀에 대한 차원축소 접근법이다. PCR은 먼저
PCA(비지도 방법)로 설명변수 공간을 축소한 뒤 선행 주성분에 반응변수를 회귀시킨다. PLS는
반응변수와의 공분산이 큰 방향을 설명변수 공간에서 찾는다(지도 방법). 이 절에서는 두 방법을
주택 자료에 적용하고 OLS 및 능형회귀와 비교한다.

## 주성분회귀(PCR)

### 착상

PCR은 두 단계로 진행된다.

1. **차원축소.** $X$의 주성분 $Z_1, \dots, Z_M$을 계산한다. 여기서 $Z_m = X v_m$이고
   $v_m$은 $X^\top X$의 $m$번째 고유벡터다.
2. **회귀.** $Z_1, \dots, Z_M$ 위에 $y$를 OLS로 회귀시킨다.

성분의 개수 $M \le p$는 교차검증으로 고르는 조율모수다.

### 수학적 정식화

$X = U D V^\top$를 SVD라 하자. 주성분은 $Z = XV = UD$다. $M$개 성분을 쓰는 PCR은

$$
\hat{y}^{\text{PCR}} = Z_M (Z_M^\top Z_M)^{-1} Z_M^\top y = \sum_{m=1}^{M} z_m \frac{z_m^\top y}{\|z_m\|^2}
$$

를 적합한다. 여기서 $Z_M = [z_1 \mid \cdots \mid z_M]$은 처음 $M$개 주성분을 담는다.

### 능형회귀와의 관계

PCR과 능형회귀는 모두 주성분 방향을 따라 축소하지만 방식이 다르다.

- **능형회귀**는 $m$번째 성분을 $d_m^2/(d_m^2 + \lambda)$배로 축소하며, 이 인자는 연속이다.
- **PCR**은 성분을 살리거나(인자 1) 버리거나(인자 0) 둘 중 하나이며, 이는 이산적이다.

따라서 능형회귀는 PCR의 "매끄러운" 판본이라 할 수 있다.

## 부분최소제곱(PLS)

### 착상

PCA가 $X$만의 분산이 최대인 방향을 찾는 것과 달리, PLS는 $X$ 공간에서 $y$와 가장 상관이 큰
방향을 찾는다. PLS 성분 $T_1, \dots, T_M$은 반복적으로 구성된다.

1. 가중벡터 $w_m = X^\top y / \|X^\top y\|$를 계산한다($y$와의 공분산이 최대인 방향).
2. 성분 $t_m = X w_m$을 만든다.
3. 축소(deflate): $X$와 $y$를 $t_m$에 회귀시킨 잔차로 각각 대체한다.
4. 반복한다.

### PLS가 PCR보다 나은 경우

PLS는 다음과 같은 상황에서 PCR을 능가하는 경향이 있다.

- $X$에서 분산이 가장 큰 방향이 반응변수와 정렬되어 있지 않을 때.
- 적은 수의 지도 성분만으로 $X$--$y$ 관계를 충분히 포착할 수 있을 때.

## 코드: 자료 적재와 OLS 기준선

```python
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

DATA = Path(__file__).parent.parent.parent / 'data'
house = pd.read_csv(DATA / 'house_sales.csv', sep='\t')

numeric_features = [
    'SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms',
    'BldgGrade', 'NbrLivingUnits', 'SqFtFinBasement', 'YrBuilt', 'YrRenovated'
]
X = house[numeric_features].values
y = house['AdjSalePrice'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ols_model = LinearRegression().fit(X_scaled, y)
ols_r2 = r2_score(y, ols_model.predict(X_scaled))
ols_rmse = np.sqrt(mean_squared_error(y, ols_model.predict(X_scaled)))
```

## 코드: 교차검증을 곁들인 PCR

```python
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, KFold

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_var = pca.explained_variance_ratio_
cumsum_var = np.cumsum(explained_var)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
pcr_mse_scores = []

for M in range(1, X_scaled.shape[1] + 1):
    reg = LinearRegression()
    cv_scores = cross_val_score(
        reg, X_pca[:, :M], y,
        cv=kfold, scoring='neg_mean_squared_error'
    )
    pcr_mse_scores.append(-cv_scores.mean())

M_opt_pcr = np.argmin(pcr_mse_scores) + 1
pcr_cv_rmse = np.sqrt(pcr_mse_scores[M_opt_pcr - 1])
```

설명분산의 스크리 그림을 보면 $X$의 변동 대부분을 몇 개의 성분이 포착하는지 가늠할 수 있다.

!!! warning "이 코드의 자료 누설"
    위 코드는 `PCA()`를 전체 자료에 한 번 적합한 뒤 그 성분으로 교차검증한다. 주성분이 검증
    겹의 정보를 이미 반영하므로 CV 오차가 낙관적으로 편향된다. 엄밀하게 하려면 표준화와 PCA를
    모두 `Pipeline` 안에 넣어 각 겹의 훈련자료에서만 적합해야 한다.

## 코드: 교차검증을 곁들인 PLS

```python
from sklearn.cross_decomposition import PLSRegression

pls_mse_scores = []
for M in range(1, X_scaled.shape[1] + 1):
    pls = PLSRegression(n_components=M)
    cv_scores = cross_val_score(
        pls, X_scaled, y,
        cv=kfold, scoring='neg_mean_squared_error'
    )
    pls_mse_scores.append(-cv_scores.mean())

M_opt_pls = np.argmin(pls_mse_scores) + 1
pls_cv_rmse = np.sqrt(pls_mse_scores[M_opt_pls - 1])
```

## 코드: 비교용 능형회귀

```python
from sklearn.linear_model import RidgeCV

ridge_cv = RidgeCV(alphas=np.logspace(-2, 5, 100), cv=10)
ridge_cv.fit(X_scaled, y)
ridge_r2 = r2_score(y, ridge_cv.predict(X_scaled))
ridge_rmse = np.sqrt(mean_squared_error(y, ridge_cv.predict(X_scaled)))
```

## 모형 비교

| 모형 | 초모수 | 핵심 성질 |
|---|---|---|
| OLS | 없음 | 불편이지만 분산이 가장 큼 |
| 능형회귀 | $\lambda$ | 연속적 축소, 모든 특성 유지 |
| PCR | $M$(성분 수) | 비지도 차원축소 |
| PLS | $M$(성분 수) | 지도 차원축소 |

주택 자료에서 PCR과 PLS는 더 적은 유효모수로 OLS에 근접한 $R^2$를 낸다. PLS는 반응변수와
관련된 방향을 직접 겨냥하므로 대개 PCR보다 적은 성분을 필요로 한다.

## 해석

- **PCR**은 $X$의 분산을 거의 설명하지 못하는 주성분을 버리는데, 그 성분이 $y$와 관련이
  있을 수도 없을 수도 있다. 분산이 작은 성분이 반응변수를 잘 예측하는 경우도 가능하며, 그때
  PCR은 그 성분을 놓친다.
- **PLS**는 $y$와의 공분산을 직접 겨냥하므로 대개 더 적은 성분으로 충분하다. 그래서 $p \gg n$인
  화학계량학이나 분광학에서 특히 유용하다.
- **능형회귀**는 이산적인 성분 선택 대신 연속적 축소로 비슷한 결과를 얻는다. 닫힌 형태의 해가
  있어 계산도 더 싸다.
- **방법의 선택**은 목적에 달려 있다. 잠재성분 관점의 해석이 중요하면 PCR이나 PLS를, 원래
  특성 전부를 쓰는 예측이 좋다면 능형회귀를 쓴다.

## 연습문제

**연습문제 1.** $M = p$개 성분을 쓰는 PCR이 OLS와 동치임을 보여라.

??? success "풀이"

    주성분은 $Z = XV$이고, 여기서 $V$는 $X^\top X$의 고유벡터로 이루어진 $p \times p$
    직교행렬이다. $M = p$이면 PCR은 $Z$의 모든 열에 $y$를 회귀시킨다.

    $$
    \hat{\beta}^{\text{PCR}} = V (Z^\top Z)^{-1} Z^\top y.
    $$

    $Z = XV$이고 $V$가 직교($V^\top V = I$)이므로

    $$
    Z^\top Z = V^\top X^\top X V = D^2
    $$

    이며, $D^2 = \text{diag}(d_1^2, \dots, d_p^2)$는 고유값을 담는다. 또한
    $Z^\top y = V^\top X^\top y$이므로

    $$
    \hat{\beta}^{\text{PCR}} = V D^{-2} V^\top X^\top y = (V D^2 V^\top)^{-1} X^\top y = (X^\top X)^{-1} X^\top y = \hat{\beta}^{\text{OLS}}
    $$

    이다. 성분을 모두 남기면 버려지는 정보가 없다. $\square$

---

**연습문제 2.** 설명변수의 표준화가 PCR에는 필수적이지만 OLS에는 반드시 필요하지 않은 이유를
설명하라. 표준화하지 않으면 PCR에서 무엇이 잘못되는가?

??? success "풀이"

    PCA는 분산이 최대인 방향을 찾는다. 설명변수의 척도가 서로 다르면(예: 면적은 수천 단위,
    침실 수는 한 자릿수) 선행 주성분은 예측력과 무관하게 분산이 큰(척도가 큰) 변수에 지배된다.

    **예:** `SqFtLot`이 1,000에서 500,000까지, `Bedrooms`가 1에서 6까지 변한다면, 첫 주성분은
    단지 수치적 분산이 크다는 이유만으로 거의 전적으로 `SqFtLot`과 정렬된다. 그러면 PCR은
    침실 수가 더 예측력이 높더라도 대지면적에 근거해 회귀하게 된다.

    OLS에는 이런 문제가 없다. 중간에 분산 최대화 단계를 거치지 않고 잔차제곱합을 직접
    최소화하기 때문이다. OLS 계수는 각 설명변수의 척도에 맞추어 자동으로 조정된다. (수치
    안정성을 위해 표준화는 여전히 좋은 습관이지만, OLS의 적합값 자체는 바뀌지 않는다.)
    $\square$

---

**연습문제 3.** 주택 예제에서 최적 PCR이 $p = 9$개 중 $M = 7$개 성분을 쓴다고 하자. 이는 자료에
대해 무엇을 말해 주는가? PLS는 더 많은 성분을 필요로 할까, 더 적은 성분을 필요로 할까?

??? success "풀이"

    PCR이 9개 중 7개 성분을 필요로 한다면, 마지막 두 주성분이 $X$의 분산은 거의 설명하지
    못하면서도 $y$를 예측하는 데 유용한 정보를 담고 있다는 뜻이다. 이를 버리면 예측이 조금
    나빠진다. 즉 자료의 신호가 저차원 부분공간에 몰려 있지 않고 설명변수 공간의 여러 방향에
    퍼져 있음을 시사한다.

    PLS는 **더 적은** 성분을 필요로 할 가능성이 크다. $X$만의 분산이 아니라 $y$와의 공분산을
    최대화하는 방향을 만들기 때문이다. 어떤 방향이 $X$의 분산을 거의 설명하지 못하더라도
    $y$와 강하게 연관되어 있으면 PLS는 그 방향을 일찍 집어낸다. 경험적으로 PLS는 2--5개
    성분만으로 7개 성분을 쓴 PCR과 비슷하거나 더 나은 성능을 내는 경우가 많다. $\square$

---

**연습문제 4.** scikit-learn의 PCA를 쓰지 않고, 중심화·척도화한 계획행렬의 SVD를 이용해 PCR을
직접 구현하라. 작은 시험자료에서 구현 결과가 scikit-learn과 일치함을 확인하라.

??? success "풀이"

    ```python
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression

    np.random.seed(42)
    n, p = 50, 5
    X = np.random.randn(n, p)
    beta_true = np.array([3, -1, 2, 0, 0])
    y = X @ beta_true + np.random.randn(n) * 0.5

    # Standardize
    X_s = StandardScaler().fit_transform(X)

    # Manual PCR via SVD
    U, D, Vt = np.linalg.svd(X_s, full_matrices=False)
    M = 3  # number of components
    Z = U[:, :M] * D[:M]  # first M principal components
    y_bar = y.mean()                                  # intercept
    gamma = np.linalg.lstsq(Z, y - y_bar, rcond=None)[0]
    beta_pcr_manual = Vt[:M].T @ gamma
    y_pred_manual = X_s @ beta_pcr_manual + y_bar

    # Scikit-learn PCR
    pca = PCA(n_components=M)
    Z_sk = pca.fit_transform(X_s)
    reg = LinearRegression().fit(Z_sk, y)
    y_pred_sk = reg.predict(Z_sk)

    print(f"Max prediction difference: {np.max(np.abs(y_pred_manual - y_pred_sk)):.2e}")
    ```

    실행하면 최대 예측 차이는 $9.8 \times 10^{-15}$로 기계 엡실론 수준이며, 두 구현이 동치임을
    확인해 준다.

    !!! warning "절편을 빠뜨리면"
        `np.linalg.lstsq(Z, y)`처럼 $y$를 중심화하지 않고 그대로 회귀시키면 절편이 빠진다.
        반면 scikit-learn의 `LinearRegression`은 기본적으로 절편을 적합하므로, 두 예측값은
        정확히 $\bar{y}$만큼 어긋난다. 위 자료에서는 $\bar{y} = 0.3189$이고 실제로 최대 차이가
        $0.319$로 나온다. $Z$의 열은 중심화되어 있지만 $y$는 그렇지 않다는 점을 놓치기 쉽다.

    !!! note "부호 규약"
        `np.linalg.svd`와 `PCA`는 특이벡터의 부호 규약이 다를 수 있어 $v_m$과 $\gamma_m$의
        부호가 뒤집혀 나올 수 있다. 그러나 곱 $z_m \gamma_m$은 부호에 불변이므로 **예측값**은
        정확히 일치한다. 성분별 계수를 직접 비교할 때는 부호를 맞춰 주어야 한다.
    $\square$

---

**연습문제 5.** 첫 PLS 방향 $w_1$이 $\|w\| = 1$ 아래에서 $\text{Cov}(Xw, y)^2$을 최대화함을
증명하고, 이것이 $w_1 \propto X^\top y$와 동치임을 보여라.

??? success "풀이"

    구하고자 하는 것은

    $$
    w_1 = \arg\max_{\|w\|=1} \left[\text{Cov}(Xw, y)\right]^2
    $$

    이다. $X$와 $y$가 중심화되어 있다고 하면
    $\text{Cov}(Xw, y) = \frac{1}{n-1}(Xw)^\top y = \frac{1}{n-1}w^\top X^\top y$이다.
    $\|w\| = 1$ 아래에서 $[w^\top X^\top y]^2$을 최대화하는 것은 $|w^\top X^\top y|$를
    최대화하는 것과 같다.

    코시-슈바르츠 부등식에 의해

    $$
    |w^\top (X^\top y)| \le \|w\| \cdot \|X^\top y\| = \|X^\top y\|
    $$

    이고, 등호는 $w \propto X^\top y$일 때 성립한다. 따라서

    $$
    w_1 = \frac{X^\top y}{\|X^\top y\|}
    $$

    이다. 즉 첫 PLS 방향은 각 설명변수와 반응변수의 주변공분산을 모은 벡터를 정규화한 것에
    지나지 않는다. $\square$
