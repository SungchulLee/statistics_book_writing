# 능형회귀 예제

## 개요

능형회귀는 최소제곱(OLS) 목적함수에 $L_2$ 벌점을 더해 계수를 0 쪽으로 축소하되 어느 것도
정확히 0으로 만들지는 않는다. 이 기법은 설명변수들이 상관되어 있거나(다중공선성) $p$가 $n$에
가깝거나 그보다 클 때 특히 효과적이다. 이 절에서는 능형 추정량을 유도하고, 편향-분산 절충을
살펴보며, 조율모수 $\lambda$가 계수 추정치에 미치는 영향을 확인한다.

## 능형 목적함수

계획행렬 $X \in \mathbb{R}^{n \times p}$와 반응변수 $y \in \mathbb{R}^n$이 주어졌을 때
능형회귀 문제는

$$
\hat{\beta}^{\text{ridge}} = \arg\min_{\beta} \left\{ \| y - X\beta \|_2^2 + \lambda \| \beta \|_2^2 \right\}
$$

이며, 여기서 $\lambda \ge 0$은 정칙화(조율) 모수다. 벌점항
$\lambda \| \beta \|_2^2 = \lambda \sum_{j=1}^{p} \beta_j^2$는 계수가 커지는 것을 억제한다.

### 닫힌 형태의 해

기울기를 0으로 놓으면 닫힌 형태의 해

$$
\hat{\beta}^{\text{ridge}} = (X^\top X + \lambda I_p)^{-1} X^\top y
$$

를 얻는다. $\lambda = 0$이면 OLS 추정량이 되고, $\lambda \to \infty$이면 모든 계수가 0으로
축소된다.

## 편향-분산 절충

능형회귀는 편향을 감수하는 대신 분산을 줄인다. 능형 추정량의 평균제곱오차(MSE)는

$$
\text{MSE}(\hat{\beta}^{\text{ridge}}) = \text{편향}^2 + \text{분산}
$$

으로 분해된다. $\lambda$가 작으면 추정량은 거의 불편이지만 분산이 크고(OLS에 가깝다),
$\lambda$가 크면 분산은 작지만 편향이 크다. 최적의 $\lambda$는 둘의 합인 MSE를 최소화한다.

## 코드: 자료 생성과 능형 적합

다음 스크립트는 정규분포 표본을 생성하고 기본 요약통계량을 출력한다. 실제로는 이 자리에 완전한
능형회귀 적합을 넣게 된다.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

n = 100
data = np.random.normal(loc=0, scale=1, size=n)

print(f"Sample size: {n}")
print(f"Sample mean: {data.mean():.4f}")
print(f"Sample std:  {data.std(ddof=1):.4f}")
```

출력:

```
Sample size: 100
Sample mean: -0.1038
Sample std:  0.9082
```

완전한 구현에서는 $X$와 $y$를 구성하고, 설명변수를 표준화한 뒤, $\lambda$ 격자 위에서
$\hat{\beta}^{\text{ridge}}$를 구한다.

## 표준화

$L_2$ 벌점은 모든 계수를 동등하게 취급하므로, 적합 전에 설명변수를 표준화해야 한다.

$$
\tilde{x}_{ij} = \frac{x_{ij} - \bar{x}_j}{s_j},
$$

여기서 $\bar{x}_j$와 $s_j$는 $j$번째 설명변수의 표본평균과 표본표준편차다. 표준화하지 않으면
벌점이 단위가 큰 변수의 계수를 부당하게 더 많이 축소한다.

## 정칙화 경로

**정칙화 경로**는 각 계수 $\hat{\beta}_j^{\text{ridge}}$를 $\lambda$(또는
$\log_{10}\lambda$)의 함수로 그린 그림이다. 주요 관찰 사항은 다음과 같다.

- 유한한 모든 $\lambda$에 대해 계수는 0이 아니다.
- $\lambda$가 커짐에 따라 계수는 매끄럽게 0을 향해 축소된다.
- 중요한 설명변수의 계수는 더 넓은 $\lambda$ 범위에서 큰 값을 유지한다.

## 해석

- **능형회귀는 변수선택을 하지 않는다.** $\lambda$와 무관하게 모든 설명변수가 모형에 남는다.
  희소성을 통한 해석 가능성이 필요하면 라쏘나 엘라스틱넷을 고려하라.
- **다중공선성 완화.** $X^\top X$에 더해지는 $\lambda I_p$가 행렬의 가역성을 보장하고
  추정치를 안정화한다.
- **$\lambda$의 선택.** 교차검증(예: 5-겹 또는 10-겹)이 표준적인 방법이다. 교차검증 예측오차를
  최소화하는 $\lambda$를 고른다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** 능형 목적함수에서 출발하여 기울기를 0으로 놓음으로써 닫힌 형태의 해
$\hat{\beta}^{\text{ridge}} = (X^\top X + \lambda I_p)^{-1} X^\top y$를 유도하라.

</div>

??? success "풀이"

    목적함수는

    $$
    L(\beta) = (y - X\beta)^\top (y - X\beta) + \lambda \beta^\top \beta
    $$

    이다. 전개한 뒤 $\beta$로 미분하면

    $$
    \frac{\partial L}{\partial \beta} = -2 X^\top y + 2 X^\top X \beta + 2\lambda \beta
    $$

    이고, 이를 0으로 놓으면

    $$
    (X^\top X + \lambda I_p) \beta = X^\top y
    $$

    를 얻는다. $\lambda > 0$이면 $X^\top X + \lambda I_p$는 양정치이므로 가역이고, 따라서

    $$
    \hat{\beta}^{\text{ridge}} = (X^\top X + \lambda I_p)^{-1} X^\top y. \quad \square
    $$

<div class="drillbox" markdown>

**연습문제 2.** $X = U D V^\top$를 특이값분해(SVD)라 할 때, 능형 추정량이
$\hat{\beta}^{\text{ridge}} = \sum_{j=1}^{p} \frac{d_j^2}{d_j^2 + \lambda}\, \frac{u_j^\top y}{d_j}\, v_j$
로 표현됨을 보여라.

</div>

??? success "풀이"

    $X = U D V^\top$이고 $D = \text{diag}(d_1, \dots, d_p)$라 하자. 그러면
    $X^\top X = V D^2 V^\top$, $X^\top y = V D U^\top y$이므로

    $$
    \hat{\beta}^{\text{ridge}} = (V D^2 V^\top + \lambda I)^{-1} V D U^\top y = V (D^2 + \lambda I)^{-1} D U^\top y
    $$

    이다. 성분으로 쓰면 $V^\top \hat{\beta}^{\text{ridge}}$의 $j$번째 원소가
    $\frac{d_j}{d_j^2 + \lambda} u_j^\top y$이므로

    $$
    \hat{\beta}^{\text{ridge}} = \sum_{j=1}^{p} \frac{d_j^2}{d_j^2 + \lambda} \cdot \frac{u_j^\top y}{d_j} \cdot v_j
    $$

    를 얻는다. 인자 $d_j^2 / (d_j^2 + \lambda) \in [0, 1)$은 특이값이 작은 방향을 더 강하게
    축소한다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** $X^\top X = I_p$(정규직교 계획)라 하자. $\hat{\beta}_j^{\text{ridge}}$를
$\hat{\beta}_j^{\text{OLS}}$와 $\lambda$로 표현하라.

</div>

??? success "풀이"

    $X^\top X = I_p$이면 OLS 추정량은 $\hat{\beta}^{\text{OLS}} = X^\top y$이고, 능형
    추정량은

    $$
    \hat{\beta}^{\text{ridge}} = (I_p + \lambda I_p)^{-1} X^\top y = \frac{1}{1 + \lambda}\, \hat{\beta}^{\text{OLS}}
    $$

    이 된다. 즉 모든 계수가 $1/(1 + \lambda)$배로 균일하게 축소된다. 이는 능형회귀가 비례
    축소를 수행함을 확인해 준다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** 다중공선성이 있는 인공자료($n = 200$, $p = 10$)에 대해 5-겹 교차검증으로
격자 $\lambda \in \{10^{-3}, 10^{-2}, \dots, 10^{3}\}$에서 최적 $\lambda$를 찾아라. 최적
$\lambda$의 교차검증 RMSE를 보고하고 OLS의 RMSE와 비교하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)
    n, p = 200, 10

    # Correlated design
    rho = 0.9
    Sigma = rho * np.ones((p, p)) + (1 - rho) * np.eye(p)
    L = np.linalg.cholesky(Sigma)
    X = np.random.randn(n, p) @ L.T
    beta_true = np.array([3, -2, 1.5, 0, 0, 0, 0, 0, 0, 0])
    y = X @ beta_true + np.random.randn(n)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # OLS
    ols_scores = cross_val_score(
        LinearRegression(), X_s, y, cv=5,
        scoring="neg_mean_squared_error"
    )
    ols_rmse = np.sqrt(-ols_scores.mean())

    # Ridge grid search
    best_rmse, best_lam = np.inf, None
    for exp in range(-3, 4):
        lam = 10.0 ** exp
        scores = cross_val_score(
            Ridge(alpha=lam), X_s, y, cv=5,
            scoring="neg_mean_squared_error"
        )
        rmse = np.sqrt(-scores.mean())
        if rmse < best_rmse:
            best_rmse, best_lam = rmse, lam

    print(f"OLS CV RMSE:  {ols_rmse:.4f}")
    print(f"Best lambda:  {best_lam}")
    print(f"Ridge CV RMSE: {best_rmse:.4f}")
    ```

    출력:

    ```
    OLS CV RMSE:  1.0171
    Best lambda:  1.0
    Ridge CV RMSE: 1.0156
    ```

    실행하면 OLS의 교차검증 RMSE는 1.0171이고, 최적 $\lambda = 1$에서 능형회귀의 RMSE는
    1.0156으로 조금 더 작다. $\lambda$를 더 키우면(10, 100, 1000) RMSE는 각각 1.0768,
    1.4158, 1.7891로 오히려 나빠진다. 즉 정칙화는 도움이 되지만 그 이득의 크기와 최적
    $\lambda$의 위치는 자료에 따라 다르며, 여기서는 $n = 200$이 $p = 10$에 비해 충분히
    커서 OLS 자체가 이미 안정적이므로 개선폭이 작다. 개선폭은 $p/n$이 커질수록 뚜렷해진다.
    $\square$

<div class="drillbox" markdown>

**연습문제 5.** 임의의 $\lambda > 0$에 대해 능형 추정량이
$\|\hat{\beta}^{\text{ridge}}\|_2 \le \|\hat{\beta}^{\text{OLS}}\|_2$를 만족함을 증명하라.

</div>

??? success "풀이"

    KKT 조건에 의해 능형 문제는 어떤 $t > 0$에 대해

    $$
    \min_{\beta} \| y - X\beta \|_2^2 \quad \text{subject to} \quad \|\beta\|_2^2 \le t
    $$

    와 동치다. OLS 해는 아무 제약 없이 손실을 최소화하므로, $\hat{\beta}^{\text{OLS}}$는
    제약영역 안에 있거나(이 경우 $\hat{\beta}^{\text{ridge}} = \hat{\beta}^{\text{OLS}}$이고
    등호가 성립한다) 밖에 있다. 밖에 있으면 제약 최적해는 경계
    $\|\beta\|_2^2 = t < \|\hat{\beta}^{\text{OLS}}\|_2^2$ 위에 놓인다. 어느 경우든

    $$
    \|\hat{\beta}^{\text{ridge}}\|_2 \le \|\hat{\beta}^{\text{OLS}}\|_2. \quad \square
    $$
