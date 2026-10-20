# 능형회귀 보기

## 개요

능형회귀는 최소제곱(OLS) 목적함수에 $L_2$ 벌점을 더해 계수를 0 쪽으로 축소하되 어느 것도
정확히 0으로 만들지는 않는다. 이 기법은 설명변수들이 상관되어 있거나(다중공선성) $p$가 $n$에
가깝거나 그보다 클 때 특히 효과적이다. 이 절에서는 능형 추정량을 유도하고, 편향-분산 절충을
살펴보며, 조율모수 $\lambda$가 계수 추정치에 미치는 영향을 확인한다.

---

## 1. 능형 목적함수

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

---

## 2. 편향-분산 절충

능형회귀는 편향을 감수하는 대신 분산을 줄인다. 능형 추정량의 평균제곱오차(MSE)는

$$
\text{MSE}(\hat{\beta}^{\text{ridge}}) = \text{편향}^2 + \text{분산}
$$

으로 분해된다. $\lambda$가 작으면 추정량은 거의 불편이지만 분산이 크고(OLS에 가깝다),
$\lambda$가 크면 분산은 작지만 편향이 크다. 최적의 $\lambda$는 둘의 합인 MSE를 최소화한다.

---

## 3. 코드: 자료 생성과 능형 적합

$\lambda$를 키워 가며 계수가 어떻게 줄어드는지, 그리고 참값과의 거리가 어디에서 가장
작아지는지를 본다.

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> 람다에 따른 축소와 편향-분산 절충. $n = 40$, $p = 20$이고 참 계수는 앞의 셋만 $(3, -2, 1.5)$이며 잡음은 $N(0,1)$이다. $X$를 고정해 놓고 $\lambda$를 키워 간다.

**(1)** $\lVert\hat\beta^{\text{ridge}}(\lambda)\rVert_2$가 $\lambda$에 대해 **반드시** 감소함을 특이값분해로 보이시오. 참값과의 거리도 그러한가.

**(2)** $E\lVert\hat\beta(\lambda) - \beta\rVert^2$를 편향과 분산으로 나누어 적고, $\lambda = 0$에서의 도함수를 구해 **최소제곱보다 나은 $\lambda > 0$이 반드시 존재함**을 보이시오. 그 값을 코드로 확인하시오.

</div>

??? success "풀이"

    **(1) 해석적으로.** $X = UDV^\top$를 특이값분해라 하면 (연습문제 4에서 유도하듯)

    $$
    \hat\beta^{\text{ridge}}(\lambda) = \sum_{j=1}^p \frac{d_j}{d_j^2+\lambda}\,(u_j^\top y)\,v_j
    $$

    이고 $v_j$가 정규직교이므로

    $$
    \lVert\hat\beta^{\text{ridge}}(\lambda)\rVert_2^2
    = \sum_{j=1}^p \left(\frac{d_j\,u_j^\top y}{d_j^2+\lambda}\right)^2
    $$

    이다. 각 항의 분모가 $\lambda$에 대해 증가하므로 **모든 항이 따로따로 감소한다.** 합도 감소한다. 여기에 U자가 생길 수 없으며, 이것은 자료와 무관한 항등식 수준의 사실이다.

    참값과의 거리는 다르다. $\lambda$가 크면 $\hat\beta \to 0$이고 거리는 $\lVert\beta\rVert = \sqrt{9+4+2.25} = 3.9051$로 올라가므로, 거리가 끝까지 줄어들 수는 없다. (2)에서 보듯 중간에 바닥이 있다.

    **(2) 해석적으로.** $S = X^\top X$, $A_\lambda = (S+\lambda I)^{-1}$이라 두자. $X$를 고정하면 $\hat\beta = A_\lambda X^\top y$이고 $y = X\beta + \varepsilon$, $\operatorname{Var}(\varepsilon) = \sigma^2 I$이므로

    $$
    E[\hat\beta] = A_\lambda S\beta,
    \qquad
    E[\hat\beta] - \beta = (A_\lambda S - I)\beta = -\lambda A_\lambda \beta
    $$

    이다. 마지막 등식은 $A_\lambda S - I = A_\lambda(S + \lambda I - \lambda I) - I = -\lambda A_\lambda$에서 나온다. 분산 쪽은 $\operatorname{Var}(\hat\beta) = \sigma^2 A_\lambda S A_\lambda$이므로

    $$
    E\lVert\hat\beta(\lambda) - \beta\rVert^2
    = \underbrace{\lambda^2\,\beta^\top A_\lambda^2 \beta}_{\text{편향}^2}
    + \underbrace{\sigma^2\operatorname{tr}\!\left(A_\lambda S A_\lambda\right)}_{\text{분산}}
    $$

    다. $\lambda = 0$에서 값은 $\sigma^2\operatorname{tr}(S^{-1})$으로 최소제곱의 오차다.

    이제 $\lambda = 0$에서의 도함수를 본다. 편향항은 $\lambda^2$에 비례하므로 미분하면 $2\lambda(\cdots)$ 꼴이라 $\lambda = 0$에서 **0이다.** 분산항은 $\frac{d}{d\lambda}A_\lambda = -A_\lambda^2$이므로

    $$
    \frac{d}{d\lambda}\,\sigma^2\operatorname{tr}(A_\lambda S A_\lambda)
    = -2\sigma^2\operatorname{tr}(A_\lambda^2 S A_\lambda)
    \;\xrightarrow[\lambda\to 0]{}\;
    -2\sigma^2\operatorname{tr}(S^{-2}) < 0
    $$

    이다. **편향은 이차로 자라고 분산은 일차로 줄어든다.** 그러므로 $\lambda = 0$ 바로 오른쪽에서 오차가 반드시 감소하며, 최소제곱보다 나은 $\lambda > 0$이 언제나 존재한다. 이것이 능형회귀의 존재정리이고, 이 보기의 표에서 $\lambda = 1$이 $\lambda = 0$을 이기는 까닭이다.

    **(2) 수치적으로.**

    ```python
    import numpy as np
    from sklearn.linear_model import LinearRegression, Ridge

    rng = np.random.default_rng(42)

    # 설명변수 20개 중 참으로 쓰이는 것은 앞의 셋뿐이다. 관측은 40개로 변수 수에
    # 비해 적어, 최소제곱이 잡음까지 따라가기 좋은 상황이다.
    n, p = 40, 20
    X = rng.normal(size=(n, p))
    beta_true = np.zeros(p)
    beta_true[:3] = [3.0, -2.0, 1.5]
    y = X @ beta_true + rng.normal(0, 1, n)

    ols = LinearRegression().fit(X, y)

    # lambda 를 키울수록 계수가 0 쪽으로 줄어든다. 라쏘와 달리 정확히 0 이 되지는
    # 않고 작아지기만 한다. 그래서 능형은 변수를 고르지 못한다.
    print(f"{'lambda':>8}  {'계수의 L2 크기':>14}  {'참값과의 거리':>14}")
    print(f"{0.0:>8.1f}  {np.linalg.norm(ols.coef_):>14.3f}  "
          f"{np.linalg.norm(ols.coef_ - beta_true):>14.3f}")
    for lam in [0.1, 1.0, 10.0, 100.0]:
        ridge = Ridge(alpha=lam).fit(X, y)
        print(f"{lam:>8.1f}  {np.linalg.norm(ridge.coef_):>14.3f}  "
              f"{np.linalg.norm(ridge.coef_ - beta_true):>14.3f}")

    # 참값과의 거리가 lambda=0 일 때보다 중간 어딘가에서 작아진다. 편향을 조금
    # 받아들이는 대가로 분산을 크게 줄인 결과이며, 이것이 편향-분산 절충이다.

    # --- 유도한 기댓값 곡선과 맞춰 본다 (sigma = 1, X 는 고정) ---
    S = X.T @ X

    def expected_sq_error(lam):
        """E||beta_hat(lam) - beta||^2 = 편향^2 + 분산.  X 를 고정한 조건부 기댓값."""
        A = np.linalg.inv(S + lam * np.eye(p))
        bias2 = lam ** 2 * beta_true @ (A @ A) @ beta_true
        var = np.trace(A @ S @ A)
        return bias2, var

    lams = np.logspace(-2, 2, 400)
    tot = np.array([sum(expected_sq_error(l)) for l in lams])
    k = tot.argmin()
    b2, v = expected_sq_error(lams[k])
    print(f"이론: lambda = 0 에서 E||.||^2 = tr(S^-1) = {np.trace(np.linalg.inv(S)):.4f}")
    print(f"이론: 최솟값 {tot[k]:.4f} (편향^2 {b2:.4f} + 분산 {v:.4f}) at lambda = {lams[k]:.4f}")
    print(f"0 에서의 도함수 = -2*tr(S^-2) = {-2 * np.trace(np.linalg.inv(S) @ np.linalg.inv(S)):.4f}  (< 0)")

    grid = np.logspace(-2, 2, 2000)
    dist = np.array([np.linalg.norm(Ridge(alpha=l).fit(X, y).coef_ - beta_true) for l in grid])
    print(f"이 표본의 거리 곡선 최솟값 {dist.min():.4f} at lambda = {grid[dist.argmin()]:.4f}")
    ```

    출력:

    ```
      lambda       계수의 L2 크기         참값과의 거리
         0.0           4.202           1.081
         0.1           4.169           1.048
         1.0           3.929           0.880
        10.0           2.891           1.404
       100.0           1.091           3.026
    이론: lambda = 0 에서 E||.||^2 = tr(S^-1) = 1.3683
    이론: 최솟값 1.0621 (편향^2 0.1914 + 분산 0.8707) at lambda = 1.6810
    0 에서의 도함수 = -2*tr(S^-2) = -0.5116  (< 0)
    이 표본의 거리 곡선 최솟값 0.8499 at lambda = 1.7584
    ```

    ![계수 크기는 단조감소하지만 참값과의 거리는 U자를 그린다](./img/shrink_vs_accuracy.png)

    **유도와 코드가 맞는다.** 기댓값 곡선은 $\lambda = 1.6810$에서 바닥을 치고, 이 한 표본의 거리 곡선은 $\lambda = 1.7584$에서 바닥을 친다. 둘이 정확히 같을 이유는 없다. 앞의 것은 $y$에 대한 기댓값이고 뒤의 것은 $y$를 한 번 뽑은 실현값이기 때문이다. 바닥의 **위치**가 가까운 것으로 충분하며, 바닥의 **값**은 $1.0305$(기댓값의 제곱근) 대 $0.8499$로 더 벌어진다. 이 표본이 운 좋게 평균보다 잘 맞은 것이다.

    $\lambda = 0$에서의 도함수가 $-0.5116$으로 음수인 것이 (2)의 결론을 그대로 확인한다. 최적점에서 편향제곱 $0.1914$와 분산 $0.8707$의 합 $1.0621$이 최소제곱의 $1.3683$보다 $22\%$ 작다. **편향 $0.19$를 사서 분산 $0.50$을 깎은 거래다.**

    위 표의 다섯 줄을 촘촘한 격자로 채워 그린 것이 위 그림이다. 파란 곡선은 계수벡터의 크기 $\lVert\hat{\boldsymbol{\beta}}\rVert_2$, 주황 곡선은 참값과의 거리 $\lVert\hat{\boldsymbol{\beta}} - \boldsymbol{\beta}\rVert_2$다. **두 곡선의 모양이 다르다는 것이 이 그림의 전부다.**

    파란 곡선은 끝까지 단조감소한다. $4.202$에서 출발해 $\lambda = 100$에서 $1.091$까지, 그리고 그 뒤로도 계속 내려간다. (1)에서 항별로 보인 그대로이며 여기에는 U자가 있을 수 없다.

    주황 곡선은 다르다. $\lambda = 0.01$에서 $1.081$(OLS와 사실상 같다)로 시작해 $\lambda = 1.76$에서 $0.850$까지 내려갔다가 다시 올라간다. $\lambda = 10$에서 벌써 $1.404$로 OLS보다 나빠지고 $\lambda = 100$에서는 $3.026$이다. 회색 점선으로 그린 OLS 수준 $1.081$과 주황 곡선이 만나는 지점이 $\lambda = 5.51$인데, **그보다 큰 $\lambda$는 손해**라는 뜻이다. 이득이 나는 구간은 생각보다 좁다.

    여기서 실무적 함정 하나가 보인다. 계수가 작아지는 것은 눈에 잘 띄지만 정확도가 나빠지는 것은 눈에 띄지 않는다. $\lambda = 100$에서 계수의 크기는 $1.091$로 "아주 깔끔하게 정리된" 모형처럼 보이는데, 실제로는 참값에서 $3.026$만큼 떨어져 OLS보다 세 배 나쁘다. **계수가 작다는 것은 좋은 모형의 증거가 아니다.** 어디가 바닥인지는 오직 자료로 추정해야 하며, 그 방법이 교차검증이다.

---

## 4. 표준화

$L_2$ 벌점은 모든 계수를 동등하게 취급하므로, 적합 전에 설명변수를 표준화해야 한다.

$$
\tilde{x}_{ij} = \frac{x_{ij} - \bar{x}_j}{s_j},
$$

여기서 $\bar{x}_j$와 $s_j$는 $j$번째 설명변수의 표본평균과 표본표준편차다. 표준화하지 않으면
벌점이 단위가 큰 변수의 계수를 부당하게 더 많이 축소한다.

---

## 5. 정칙화 경로

**정칙화 경로**는 각 계수 $\hat{\beta}_j^{\text{ridge}}$를 $\lambda$(또는
$\log_{10}\lambda$)의 함수로 그린 그림이다. 주요 관찰 사항은 다음과 같다.

- 유한한 모든 $\lambda$에 대해 계수는 0이 아니다.
- $\lambda$가 커짐에 따라 계수는 매끄럽게 0을 향해 축소된다.
- 중요한 설명변수의 계수는 더 넓은 $\lambda$ 범위에서 큰 값을 유지한다.

---

## 6. 해석

- **능형회귀는 변수선택을 하지 않는다.** $\lambda$와 무관하게 모든 설명변수가 모형에 남는다.
  희소성을 통한 해석 가능성이 필요하면 라쏘나 엘라스틱넷을 고려하라.
- **다중공선성 완화.** $X^\top X$에 더해지는 $\lambda I_p$가 행렬의 가역성을 보장하고
  추정치를 안정화한다.
- **$\lambda$의 선택.** 교차검증(예: 5-겹 또는 10-겹)이 표준적인 방법이다. 교차검증 예측오차를
  최소화하는 $\lambda$를 고른다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span> 능형 목적함수에서 출발하여 기울기를 0으로 놓음으로써 닫힌 형태의 해
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

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> $X^\top X = I_p$(정규직교 계획)라 하자. $\hat{\beta}_j^{\text{ridge}}$를
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

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> 다중공선성이 있는 인공자료($n = 200$, $p = 10$)에 대해 5-겹 교차검증으로
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

    # 상관된 설계행렬
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

    # 능형회귀의 격자탐색
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

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span> $X = U D V^\top$를 특이값분해(SVD)라 할 때, 능형 추정량이
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

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span> 임의의 $\lambda > 0$에 대해 능형 추정량이
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

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span> 4절의 표준화가 왜 **필수**인지를 손으로 끝까지 계산해 확인한다. 절편을 두지 않는 다음 설계를 보자.

$$
X = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix},
\qquad y = \begin{pmatrix} 1 \\ 1 \end{pmatrix},
\qquad \lambda = 1
$$

**(1)** 능형 해 $\hat\beta^{\text{ridge}}$와 적합값 $\hat y = X\hat\beta^{\text{ridge}}$를 구하라.

**(2)** 둘째 변수의 **단위만** 바꾼다. 즉 $x_2' = 10\,x_2$로 재어 $X' = \begin{pmatrix} 1 & 0 \\ 0 & 10\end{pmatrix}$을 쓴다. 같은 $\lambda = 1$로 $\hat\beta'^{\text{ridge}}$와 적합값을 구하고, 원래 척도로 환산한 둘째 계수 $10\,\hat\beta_2'$를 (1)의 $\hat\beta_2$와 비교하라.

**(3)** 같은 비교를 OLS로 하면 적합값이 두 설계에서 **같음**을 확인하라. 그러므로 (2)의 어긋남은 오직 벌점에서 온다. 축소인자 $d^2/(d^2+\lambda)$로 그 어긋남을 설명하고, **열의 노름이 큰 변수는 벌점을 더 받는가 덜 받는가**를 답하라.

**(4)** 표준화가 이것을 어떻게 고치는지, 그리고 sklearn의 `Ridge`가 표준화를 해 주지 않는다는 사실이 왜 함정인지 한 줄로 적어라.

</div>

??? success "풀이"

    **(1)** $X^\top X = I_2$, $X^\top y = (1,1)^\top$이므로 연습문제 2의 비례축소가 그대로 적용된다.

    $$
    \hat\beta^{\text{ridge}} = \frac{1}{1+\lambda}\,X^\top y = \frac{1}{2}\begin{pmatrix} 1 \\ 1\end{pmatrix}
    = \begin{pmatrix} 0.5 \\ 0.5\end{pmatrix},
    \qquad
    \hat y = \begin{pmatrix} 0.5 \\ 0.5\end{pmatrix}
    $$

    **(2)** 이제 $X'^\top X' = \operatorname{diag}(1, 100)$, $X'^\top y = (1, 10)^\top$이다. 대각이므로 좌표마다 따로 풀린다.

    $$
    \hat\beta_1' = \frac{1}{1+1} = 0.5,
    \qquad
    \hat\beta_2' = \frac{10}{100+1} = \frac{10}{101} = 0.0990
    $$

    적합값은 $\hat y' = (0.5,\; 10 \cdot 10/101)^\top = (0.5,\; 100/101)^\top = (0.5,\; 0.9901)^\top$이다. 원래 척도로 환산한 둘째 계수는

    $$
    10\,\hat\beta_2' = \frac{100}{101} = 0.9901
    \;\ne\;
    0.5 = \hat\beta_2
    $$

    로, **자료를 하나도 바꾸지 않고 자(尺)만 바꾸었는데 둘째 변수의 기여가 두 배로 늘었다.** 첫째 계수는 $0.5$로 그대로이니 두 해는 서로 재조정으로 옮겨 갈 수조차 없다.

    **(3)** OLS는 $\hat\beta^{\text{OLS}} = (X^\top X)^{-1}X^\top y = (1,1)^\top$이고 $\hat\beta'^{\text{OLS}} = (1, 0.1)^\top$이다. 둘 다 적합값은 $\hat y = (1,1)^\top$으로 **같다.** 열을 $a$배 하면 OLS 계수는 정확히 $1/a$배가 되어 $x_j\beta_j$가 보존되기 때문이다(척도 등변성). 능형은 이 성질을 잃는다. 연습문제 4의 축소인자로 보면 둘째 방향의 특잇값이 $d_2 = 1 \to 10$으로 커지므로

    $$
    \frac{d_2^2}{d_2^2+\lambda} : \quad \frac{1}{1+1} = 0.5
    \;\longrightarrow\;
    \frac{100}{100+1} = 0.9901
    $$

    이다. 즉 **열의 노름이 큰 변수는 벌점을 덜 받는다.** 벌점 $\lambda\lVert\beta\rVert_2^2$는 $\beta$의 크기만 보는데, 단위를 키우면 같은 설명력을 더 작은 $\beta$로 낼 수 있어 벌점을 피해 가기 때문이다. $\lambda \to 0$에서는 두 축소인자가 모두 $1$로 가 어긋남이 사라지고, $\lambda \to \infty$에서는 둘 다 $0$으로 간다. 어긋남은 중간 $\lambda$에서 가장 크다.

    **(4)** 각 열을 같은 노름으로 맞추면(여기서는 $x_2'$를 다시 $10$으로 나누면) 모든 $d_j$가 단위에 무관해지고 해가 자(尺)에 불변이 된다. 그래서 4절의 $\tilde x_{ij} = (x_{ij}-\bar x_j)/s_j$가 선택이 아니라 **전제**다. 함정은 sklearn의 `Ridge`·`Lasso`가 이 표준화를 **대신 해 주지 않는다**는 데 있다. 키를 cm로 적었다가 m로 적는 것만으로 계수가 바뀌므로, 적합 전에 `StandardScaler`를 반드시 끼워야 한다(그것도 연습문제 9처럼 **폴드 안에서**). $\square$

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span> 능형의 **유효 자유도**를 연습문제 4의 특이값분해로 센다. $H_\lambda = X(X^\top X + \lambda I_p)^{-1}X^\top$을 능형의 모자행렬이라 하고 $\operatorname{df}(\lambda) = \operatorname{tr}(H_\lambda)$로 정의한다.

**(1)** $H_\lambda$의 고유분해를 구하고

$$
\operatorname{df}(\lambda) = \sum_{j=1}^{p} \frac{d_j^2}{d_j^2+\lambda}
$$

임을 보여라. 또 $\operatorname{df}(0) = p$이고 $\operatorname{df}$가 $\lambda$에 대해 단조감소하며 $\lambda \to \infty$에서 $0$으로 감을 확인하라.

**(2)** $\lambda > 0$이면 $H_\lambda$는 **사영행렬이 아님**을 보여라($\operatorname{tr}(H_\lambda^2) < \operatorname{tr}(H_\lambda)$를 쓰면 된다). OLS의 모자행렬과 무엇이 다른가.

**(3)** 특이값이 $d = (3, 1, 0.5)$인 계획행렬에서 $\lambda = 1$과 $\lambda = 3$의 축소인자와 $\operatorname{df}(\lambda)$를 구하라. **자유도를 먼저 잃는 방향은 어디인가.**

**(4)** 계수가 하나도 $0$이 아닌데(6절) 자유도가 $p$가 아닌 까닭을, 라쏘에서 "$0$이 아닌 계수의 개수"가 자유도의 불편추정이 되는 것과 대비해 설명하라.

</div>

??? success "풀이"

    **(1)** $X = UDV^\top$을 넣으면 $X^\top X + \lambda I = V(D^2+\lambda I)V^\top$이고 $V$가 직교이므로

    $$
    H_\lambda = UDV^\top \cdot V(D^2+\lambda I)^{-1}V^\top \cdot VDU^\top
    = U\,D(D^2+\lambda I)^{-1}D\,U^\top
    = U \operatorname{diag}\!\left(\frac{d_j^2}{d_j^2+\lambda}\right) U^\top
    $$

    이다. 즉 $H_\lambda$의 고유벡터는 $u_j$이고 고유값은 축소인자 $d_j^2/(d_j^2+\lambda) \in (0,1]$이다. 대각합은 고유값의 합이므로

    $$
    \operatorname{tr}(H_\lambda) = \sum_{j=1}^p \frac{d_j^2}{d_j^2+\lambda}
    $$

    를 얻는다. $\lambda = 0$이면 모든 항이 $1$이라 $\operatorname{df}(0) = p$(열이 선형독립일 때)이고, 각 항이 $\lambda$에 대해 감소하므로 합도 감소하며 $\lambda \to \infty$에서 각 항이 $0$으로 간다.

    **(2)** 고유값이 $h_j = d_j^2/(d_j^2+\lambda)$이므로 $H_\lambda^2$의 고유값은 $h_j^2$이다. $\lambda > 0$이면 모든 $h_j \in (0,1)$이라 $h_j^2 < h_j$이고 따라서

    $$
    \operatorname{tr}(H_\lambda^2) = \sum_j h_j^2 < \sum_j h_j = \operatorname{tr}(H_\lambda)
    $$

    이다. $H_\lambda^2 = H_\lambda$라면 대각합도 같아야 하니 모순이고, $H_\lambda$는 멱등이 아니다. OLS의 모자행렬은 고유값이 $1$ 아니면 $0$인 **직교사영**이어서 자유도가 정확히 정수 $p$다. 능형은 사영이 아니라 **방향마다 다른 비율로 누르는** 변환이므로 자유도가 $1.6$처럼 정수가 아닌 값이 된다.

    **(3)** $d^2 = (9, 1, 0.25)$다. $\lambda = 1$에서

    $$
    \frac{9}{10} = 0.9, \qquad \frac{1}{2} = 0.5, \qquad \frac{0.25}{1.25} = 0.2
    \qquad\Rightarrow\qquad
    \operatorname{df}(1) = 1.6
    $$

    이고, $\lambda = 3$에서는

    $$
    \frac{9}{12} = 0.75, \qquad \frac{1}{4} = 0.25, \qquad \frac{0.25}{3.25} = 0.0769
    \qquad\Rightarrow\qquad
    \operatorname{df}(3) = 1.0769
    $$

    이다. $3$에서 각각 $1.6$, $1.08$로 떨어졌다. **먼저 잃는 쪽은 특잇값이 작은 방향이다.** $d_3 = 0.5$인 방향은 $\lambda = 1$에서 벌써 $0.2$만 남고 $\lambda = 3$에서는 $0.08$로 사실상 지워졌지만, $d_1 = 3$인 방향은 $\lambda = 3$에서도 $0.75$를 지킨다. 축소인자를 $\lambda$와 비교하면 $d_j^2 \gg \lambda$인 방향은 거의 그대로이고 $d_j^2 \ll \lambda$인 방향은 거의 사라진다. 이것이 능형이 다중공선성(작은 $d_j$)을 다루는 방식이다 — 불안정한 방향만 골라 누른다.

    참고로 (2)의 계산도 해 보면 $\lambda = 1$에서 $\operatorname{tr}(H_1^2) = 0.81+0.25+0.04 = 1.10 < 1.6$이다.

    **(4)** 자유도는 "$0$이 아닌 계수를 몇 개 썼는가"가 아니라 "자료에 **얼마나 자유롭게** 맞췄는가"를 재는 양이다. 능형은 세 계수를 모두 $0$이 아니게 두지만 각 방향을 $0.9$, $0.5$, $0.2$만큼만 따라가므로, 자료를 따라가는 자유는 세 모수가 아니라 $1.6$개 모수에 해당한다. 라쏘는 다르다. 고른 변수 위에서는 벌점이 **상수**($\lambda$ 곱하기 부호의 합)로 바뀌어 그 부분공간 안에서는 사실상 OLS로 맞추므로, 선택의 비용까지 세면 $0$이 아닌 계수의 개수가 자유도의 불편추정이 된다. 정보기준(AIC·BIC)이나 $C_p$를 쓸 때 능형에는 $\operatorname{df}(\lambda)$를, 라쏘에는 $0$이 아닌 개수를 넣어야 하는 까닭이다. $\square$

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span> 정규직교 계획($X^\top X = I_3$, $n = 100$)에서 능형과 라쏘를 나란히 놓는다. $\hat\beta^{\text{OLS}} = (2.0,\, -0.6,\, 0.15)^\top$이고 벌점은 교과서 꼴

$$
\lVert y - X\beta\rVert_2^2 + \lambda \lVert\beta\rVert_2^2
\qquad\text{또는}\qquad
\lVert y - X\beta\rVert_2^2 + \lambda \lVert\beta\rVert_1
$$

로 쓴다.

**(1)** 열벌점의 열미분(소도함수)으로 라쏘 해가 **연성문턱** $\hat\beta_j = \operatorname{sign}(\hat\beta_j^{\text{OLS}})\bigl(\lvert\hat\beta_j^{\text{OLS}}\rvert - \lambda/2\bigr)_+$임을 유도하라. 문턱이 $\lambda$가 아니라 $\lambda/2$인 까닭을 밝혀라.

**(2)** $\lambda = 0.5$에서 능형 해와 라쏘 해를 모두 구하라.

**(3)** 각 계수를 **정확히 $0$**으로 만드는 가장 작은 $\lambda$를 세 좌표에 대해 구하라. 능형에서는 같은 물음의 답이 무엇인가.

**(4)** sklearn의 `Lasso`는 $\frac{1}{2n}\lVert y-X\beta\rVert_2^2 + \alpha\lVert\beta\rVert_1$을 최소화한다. (2)의 라쏘 해를 얻으려면 $\alpha$를 얼마로 주어야 하는가.

**(5)** $\lambda = 0.5$에서 두 해의 유효 자유도를 연습문제 7의 셈으로 비교하라. 무엇이 같고 무엇이 다른가.

</div>

??? success "풀이"

    **(1)** $X^\top X = I$이면 $\hat\beta^{\text{OLS}} = X^\top y$이고

    $$
    \lVert y - X\beta\rVert_2^2 = \lVert y\rVert^2 - 2\beta^\top X^\top y + \beta^\top\beta
    = \text{(상수)} + \sum_{j} \left(\beta_j - \hat\beta_j^{\text{OLS}}\right)^2
    $$

    이므로 목적함수가 좌표마다 분리된다. 좌표 하나의 문제는

    $$
    \min_{t} \; (t - z)^2 + \lambda\lvert t\rvert,
    \qquad z = \hat\beta_j^{\text{OLS}}
    $$

    다. $t \ne 0$에서 미분하면 $2(t-z) + \lambda\operatorname{sign}(t) = 0$, 즉 $t = z - (\lambda/2)\operatorname{sign}(t)$이다. $z > \lambda/2$이면 $t = z - \lambda/2 > 0$이 부호와 모순 없이 성립하고, $z < -\lambda/2$이면 $t = z+\lambda/2 < 0$이 성립한다. $\lvert z\rvert \le \lambda/2$이면 어느 쪽도 성립하지 않고, $t=0$에서 소도함수 집합 $-2z + \lambda[-1,1]$이 $0$을 포함하는지 보면 그 조건이 바로 $\lvert z \rvert \le \lambda/2$다. 따라서

    $$
    \hat\beta_j^{\text{lasso}} = \operatorname{sign}(z)\left(\lvert z\rvert - \lambda/2\right)_+
    $$

    이다. **문턱이 $\lambda/2$인 것은 제곱손실을 미분할 때 $2$가 나오기 때문이다.** 제곱손실을 $\frac{1}{2}\lVert y-X\beta\rVert^2$로 쓰는 책에서는 문턱이 $\lambda$가 된다 — 이 $2$ 하나가 (4)에서 숫자를 전부 바꾼다.

    **(2)** 능형은 연습문제 2의 비례축소로 $1/(1+0.5) = 2/3$배다.

    $$
    \hat\beta^{\text{ridge}} = \tfrac{2}{3}(2.0,\, -0.6,\, 0.15)^\top = (1.3333,\, -0.4000,\, 0.1000)^\top
    $$

    라쏘는 문턱 $\lambda/2 = 0.25$의 연성문턱이다.

    $$
    \hat\beta^{\text{lasso}} = (2.0-0.25,\; -(0.6-0.25),\; 0)^\top = (1.75,\, -0.35,\, 0)^\top
    $$

    $\lvert 0.15\rvert < 0.25$라 셋째 계수만 정확히 $0$이 되었다. 큰 계수는 $2.0 \to 1.75$로 $12.5\%$만 줄었는데 능형은 $2.0 \to 1.3333$으로 $33\%$를 깎았다. **라쏘는 작은 것을 죽이고 큰 것은 거의 건드리지 않으며(일정량 평행이동), 능형은 모두를 같은 비율로 깎는다.**

    **(3)** 라쏘에서 $\hat\beta_j = 0$이 되는 조건은 $\lvert\hat\beta_j^{\text{OLS}}\rvert \le \lambda/2$, 즉 $\lambda \ge 2\lvert\hat\beta_j^{\text{OLS}}\rvert$다. 따라서 임계값은

    $$
    \lambda_1 = 4.0, \qquad \lambda_2 = 1.2, \qquad \lambda_3 = 0.3
    $$

    이고 $\lambda \ge 4.0$에서 해가 완전히 $0$이 된다. 능형에서는 $\hat\beta_j = \hat\beta_j^{\text{OLS}}/(1+\lambda)$가 유한한 $\lambda$에서 절대 $0$이 아니므로 **그런 $\lambda$는 없다.** 이 한 줄이 "능형은 변수선택을 하지 않는다"(6절)의 전부다.

    **(4)** sklearn의 목적함수에 $2n$을 곱하면 $\lVert y - X\beta\rVert_2^2 + 2n\alpha\lVert\beta\rVert_1$이므로 $\lambda = 2n\alpha$, 곧

    $$
    \alpha = \frac{\lambda}{2n} = \frac{0.5}{200} = 0.0025
    $$

    다. 같은 말로 sklearn의 문턱은 $n\alpha$다. $\alpha = 0.5$를 넣으면 문턱이 $50$이 되어 모든 계수가 $0$으로 죽는다. **$\lambda$를 그대로 $\alpha$에 넣는 것이 이 장에서 가장 흔한 사고다.** (`Ridge`의 `alpha`는 사정이 달라서 교과서의 $\lambda$와 그대로 맞는다. 두 함수의 규약이 서로 다른 것도 함정이다.)

    **(5)** 능형의 유효 자유도는 $X^\top X = I$이므로 모든 $d_j = 1$이고

    $$
    \operatorname{df}(0.5) = \sum_{j=1}^{3} \frac{1}{1+0.5} = \frac{3}{1.5} = 2
    $$

    다. 라쏘는 $0$이 아닌 계수가 둘이니 자유도도 $2$다. **자유도는 똑같이 $2$인데 쓰는 방식이 정반대다.** 능형은 세 변수를 다 남기고 각각을 $2/3$만 신뢰하며, 라쏘는 둘을 거의 그대로 믿고 하나를 버린다. 같은 복잡도를 "조금씩 모두"와 "확실히 몇 개"로 나눠 쓰는 것이 $L_2$와 $L_1$의 차이이고, 예측이 목적이면 앞쪽이, 해석이 목적이면 뒤쪽이 흔히 낫다. $\square$

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span> 6절은 $\lambda$를 교차검증으로 고르라고 했다. 그런데 **전처리를 폴드 밖에서 하면** 그 교차검증 오차가 거짓이 된다. 이를 수로 보인다.

$n = 50$, $p = 500$이고 $y$를 $X$와 **완전히 무관하게** 생성한다(참 계수가 모두 $0$). $y$와의 $\lvert$상관계수$\rvert$가 큰 설명변수 $10$개를 골라 $\lambda = 1$의 능형을 적합하는 절차를 두 방식으로 교차검증하라.

**(가)** 전체 자료로 변수를 **한 번** 고른 뒤 5-겹 교차검증.
**(나)** 폴드마다 **훈련 부분만** 보고 변수를 고른 뒤 교차검증.

**(1)** 두 방식의 교차검증 RMSE와 $R^2$를 구하라. 참 신호가 없는데도 (가)가 어떤 수를 주는가.

**(2)** 왜 그런 일이 생기는지, $p = 500$에서 $\lvert$상관계수$\rvert$의 최댓값이 얼마였는지로 설명하라.

**(3)** 연습문제 3의 코드도 `StandardScaler`를 전체 자료로 적합한 뒤 교차검증했다. 그것도 누수인가. 같은 자료에서 폴드 안 표준화와 견주어 보고, 이 누수가 (가)보다 훨씬 작은 까닭을 말하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    n, p, k = 50, 500, 10

    # y 는 X 와 완전히 무관하다. 참 계수가 모두 0 이므로 어떤 모형도 y 의
    # 표준편차보다 잘 맞출 수 없다.
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)


    def top_k(Xa, ya, k):
        """|상관계수| 가 큰 설명변수 k 개의 번호."""
        Xc = Xa - Xa.mean(axis=0)
        yc = ya - ya.mean()
        num = Xc.T @ yc
        den = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())
        return np.argsort(-np.abs(num / den))[:k]


    sel_all = top_k(X, y, k)          # 전체 자료를 한 번 보고 고른다 (새는 쪽)


    def cv_rmse(select_inside):
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        sq = []
        for tr, te in kf.split(X):
            s = top_k(X[tr], y[tr], k) if select_inside else sel_all
            model = Ridge(alpha=1.0).fit(X[np.ix_(tr, s)], y[tr])
            sq.append((y[te] - model.predict(X[np.ix_(te, s)])) ** 2)
        return np.sqrt(np.concatenate(sq).mean())


    leaky = cv_rmse(select_inside=False)
    honest = cv_rmse(select_inside=True)
    base = np.sqrt(((y - y.mean()) ** 2).mean())

    cc = np.corrcoef(np.column_stack([X[:, sel_all], y]), rowvar=False)[-1, :-1]
    print(f"고른 10 개의 |상관계수|: 최대 {np.abs(cc).max():.4f}, 최소 {np.abs(cc).min():.4f}")
    print(f"기준선 (y 의 표준편차)            RMSE = {base:.4f}")
    print(f"(가) 전체 자료로 고른 뒤 교차검증  RMSE = {leaky:.4f}  R^2 = {1 - leaky ** 2 / base ** 2:+.4f}")
    print(f"(나) 폴드 안에서 고른 교차검증     RMSE = {honest:.4f}  R^2 = {1 - honest ** 2 / base ** 2:+.4f}")

    # --- 표준화만 밖에서 한 경우: 연습문제 3 의 자료로 두 방식을 견준다 ---
    np.random.seed(42)
    m, q = 200, 10
    rho = 0.9
    Sigma = rho * np.ones((q, q)) + (1 - rho) * np.eye(q)
    Xc = np.random.randn(m, q) @ np.linalg.cholesky(Sigma).T
    beta_true = np.array([3, -2, 1.5, 0, 0, 0, 0, 0, 0, 0])
    yc = Xc @ beta_true + np.random.randn(m)

    out = np.sqrt(-cross_val_score(Ridge(alpha=1.0), StandardScaler().fit_transform(Xc), yc,
                                   cv=5, scoring="neg_mean_squared_error").mean())
    inn = np.sqrt(-cross_val_score(make_pipeline(StandardScaler(), Ridge(alpha=1.0)), Xc, yc,
                                   cv=5, scoring="neg_mean_squared_error").mean())
    print(f"표준화를 밖에서   RMSE = {out:.4f}")
    print(f"표준화를 폴드 안에서 RMSE = {inn:.4f}  (차이 {inn - out:+.4f})")
    ```

    출력:

    ```
    고른 10 개의 |상관계수|: 최대 0.4095, 최소 0.3336
    기준선 (y 의 표준편차)            RMSE = 0.8953
    (가) 전체 자료로 고른 뒤 교차검증  RMSE = 0.6519  R^2 = +0.4699
    (나) 폴드 안에서 고른 교차검증     RMSE = 1.1828  R^2 = -0.7454
    표준화를 밖에서   RMSE = 1.0156
    표준화를 폴드 안에서 RMSE = 1.0158  (차이 +0.0003)
    ```

    **(1)** 참 계수가 전부 $0$이니 정직한 절차는 $y$의 표준편차 $0.8953$보다 나을 수 없고, 변수를 $10$개나 집어넣은 값은 그보다 **나빠야** 한다. (나)가 바로 그렇다. RMSE $1.1828$, $R^2 = -0.7454$로 "아무것도 하지 말라"는 정직한 답을 준다. 그런데 (가)는 RMSE $0.6519$, $R^2 = +0.4699$를 준다. **신호가 전혀 없는 자료에서 분산의 $47\%$를 설명했다고 보고하는 것이다.** 두 수의 차이는 모형이 아니라 절차에서 나왔다 — 모형과 $\lambda$와 폴드까지 전부 같고 변수를 고른 **시점**만 다르다.

    **(2)** $p = 500$개의 쓸모없는 변수 가운데 $n = 50$으로 재면 순전히 운으로 $y$와 꽤 닮은 것이 나온다. 출력의 $\lvert r\rvert$ 최댓값이 $0.4095$인데, 상관이 $0$일 때 $\lvert r\rvert$가 $0.33$을 넘을 확률은 한 변수당 대략 $2\%$ 정도이므로 $500$개 중에는 $10$개가 쉽게 나온다. 문제는 그 $10$개를 **전체 $50$개 관측을 다 보고** 골랐다는 데 있다. 그 안에는 나중에 검증 폴드로 쓰일 $10$개 관측의 잡음도 들어 있다. 검증 폴드는 "본 적 없는 자료"여야 하는데 변수선택이 이미 그것을 엿보았으므로, 선택된 변수는 그 폴드에서도 $y$와 닮아 있다. 교차검증은 이 공모를 성능으로 착각한다. 고를 때 쓴 정보가 평가할 때 다시 쓰이는 것, 이것이 **누수**다. (나)처럼 선택을 폴드 안으로 넣으면 훈련 부분에서 운 좋게 뽑힌 변수가 검증 부분에서는 아무 소용이 없어, 오차가 정직하게 커진다.

    **(3)** 그것도 원칙적으로는 누수다. `StandardScaler`가 쓰는 $\bar x_j$와 $s_j$에 검증 폴드의 관측이 들어가 있으니 모형이 검증 폴드를 간접적으로 본 셈이다. 하지만 크기가 전혀 다르다. 출력을 보면 $1.0156$과 $1.0158$로 차이가 $0.0003$, 넷째 자리에서 겨우 보인다. 까닭은 셋이다. 첫째, 표준화는 $y$를 **보지 않는다.** 새는 정보가 $X$의 열평균·열표준편차뿐이고 반응과의 관계는 전혀 담기지 않는다. 둘째, $n = 200$에서 $\bar x_j$와 $s_j$는 폴드를 $40$개 빼도 거의 그대로다. 셋째, 고르는 행동이 없다 — 누수가 가장 커지는 것은 **많은 후보 가운데 $y$를 보고 고를 때**이고, (가)는 $500$개에서 $10$개를 골랐다. 실무 규칙은 그래서 간단하다. **$y$를 보는 모든 조작(변수선택, $\lambda$ 선택, 변수 변환, 결측 대치)은 반드시 폴드 안에서 하라.** $\lambda$까지 고르려면 안쪽 교차검증을 하나 더 두는 **중첩 교차검증**이 필요하다. sklearn에서는 `make_pipeline`으로 전처리까지 추정량에 묶어 `cross_val_score`에 넘기는 것이 가장 안전하다. $\square$

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span> 능형과 라쏘는 모두 MAP 추정량이다. 이 대응이 **어디까지** 성립하는지 따진다. $y = X\beta + \varepsilon$, $\varepsilon \sim N(0, \sigma^2 I)$이고 좌표마다 독립인 사전분포를 둔다.

**(1)** 사전분포가 $\beta_j \sim N(0, \tau^2)$이면 MAP가 능형이고 $\lambda = \sigma^2/\tau^2$임을, 사전분포가 라플라스 $p(\beta_j) \propto e^{-\lvert\beta_j\rvert/b}$이면 MAP가 라쏘이고 $\lambda = 2\sigma^2/b$임을 보여라(교과서 꼴 $\lVert y-X\beta\rVert_2^2 + \lambda\lVert\beta\rVert$ 기준).

**(2)** 정규 사전분포에서는 사후분포가 정규이므로 **최빈값과 평균이 같다.** 즉 능형 추정량은 사후평균이기도 하다. 이를 확인하라.

**(3)** 라플라스 사전분포에서는 그렇지 않다. $p = 1$, $X^\top X = 1$, $\sigma = 1$, $b = 1$인 경우를 보자. 그러면 $z = \hat\beta^{\text{OLS}}$만 있으면 되고 $\lambda = 2$, 연성문턱은 $\lambda/2 = 1$이다. $z = 0.8$에서 MAP는 정확히 $0$이다. 그런데 **사후평균은 $0$이 아님**을 대칭성으로 보이고, 그 값을 구하라. 필요한 수는 $\Phi(-0.2) = 0.42074$, $\varphi(0.2) = 0.39104$, $\Phi(-1.8) = 0.035930$, $\varphi(1.8) = 0.078950$이다.

**(4)** (3)이 "라쏘의 희소성"에 대해 말해 주는 것을 한 줄로 적어라.

</div>

??? success "풀이"

    **(1)** 사후분포는 $p(\beta \mid y) \propto p(y\mid\beta)p(\beta)$이고 음의 로그를 취하면

    $$
    -\log p(\beta\mid y) = \frac{1}{2\sigma^2}\lVert y - X\beta\rVert_2^2 - \log p(\beta) + \text{(상수)}
    $$

    다. 정규 사전분포이면 $-\log p(\beta) = \frac{1}{2\tau^2}\lVert\beta\rVert_2^2 + \text{상수}$이므로 전체에 $2\sigma^2$을 곱해도 최소점이 변하지 않아

    $$
    \arg\max_\beta p(\beta\mid y)
    = \arg\min_\beta \left\{ \lVert y - X\beta\rVert_2^2 + \frac{\sigma^2}{\tau^2}\lVert\beta\rVert_2^2 \right\}
    $$

    이고 $\lambda = \sigma^2/\tau^2$다. 라플라스이면 $-\log p(\beta) = \frac{1}{b}\lVert\beta\rVert_1 + \text{상수}$이므로 같은 곱셈으로

    $$
    \arg\max_\beta p(\beta\mid y)
    = \arg\min_\beta \left\{ \lVert y - X\beta\rVert_2^2 + \frac{2\sigma^2}{b}\lVert\beta\rVert_1 \right\}
    $$

    이고 $\lambda = 2\sigma^2/b$다. 두 식에서 **$\lambda$가 크다는 것은 사전분포가 좁다는 것**($\tau$나 $b$가 작다)이고, $\sigma^2$이 크면(잡음이 크면) 같은 사전분포가 더 센 벌점으로 작동한다. $\lambda$를 교차검증으로 고르는 일은 베이즈 쪽에서 보면 사전분포의 폭을 자료로 정하는 경험적 베이즈다.

    **(2)** 정규 사전분포에서 음의 로그 사후밀도는 $\beta$의 **이차식**이다.

    $$
    \frac{1}{2\sigma^2}\left\{ \beta^\top (X^\top X + \lambda I)\beta - 2\beta^\top X^\top y \right\} + \text{(상수)},
    \qquad \lambda = \sigma^2/\tau^2
    $$

    이차형식의 지수는 정규밀도이므로 사후분포는 $N\!\left((X^\top X+\lambda I)^{-1}X^\top y,\; \sigma^2 (X^\top X+\lambda I)^{-1}\right)$이고, 정규분포는 최빈값과 평균이 같다. 그러므로

    $$
    \hat\beta^{\text{ridge}} = E[\beta \mid y]
    $$

    다. **능형은 MAP이면서 동시에 사후평균이다.** 그래서 베이즈 해석이 깔끔하고, 능형의 축소가 "사전분포로 끌어당기는 것"과 정확히 같은 말이 된다.

    **(3)** $X^\top X = 1$, $\sigma = 1$이면 사후밀도는 $z = \hat\beta^{\text{OLS}}$만 통해 자료에 의존한다.

    $$
    p(\beta \mid y) \propto \exp\!\left( -\tfrac{1}{2}(\beta - z)^2 - \lvert\beta\rvert \right)
    $$

    먼저 **평균이 $0$이 아닌 것**은 대칭성으로 바로 보인다. $\beta > 0$에 대해

    $$
    \frac{p(\beta\mid y)}{p(-\beta\mid y)}
    = \exp\!\left( -\tfrac{1}{2}(\beta-z)^2 + \tfrac{1}{2}(\beta+z)^2 \right)
    = e^{2z\beta} > 1
    \qquad (z = 0.8 > 0)
    $$

    이므로(절댓값 항은 $\pm\beta$에서 같아 지워진다) 사후분포는 양수 쪽에 더 많은 질량을 둔다. $\beta$와 $-\beta$를 짝지어 적분하면 $E[\beta\mid y] > 0$이다. **MAP가 정확히 $0$인데 사후평균은 결코 $0$이 아니다**($z \ne 0$인 모든 자료에서).

    값을 구하자. 지수를 $\beta$의 부호에 따라 완전제곱으로 묶으면, $m_\pm = z \mp 1$이라 둘 때

    $$
    -\tfrac{1}{2}(\beta-z)^2 - \lvert\beta\rvert
    = -\tfrac{1}{2}(\beta - m_+)^2 + \tfrac{1}{2}m_+^2 - \tfrac{1}{2}z^2
    \quad (\beta > 0)
    $$

    이고 $\beta<0$에서는 $m_+$ 대신 $m_- = z+1$이 들어간다. 즉 사후분포는 **평균이 $m_+ = -0.2$인 정규분포를 $\beta>0$으로 자른 조각**과 **평균이 $m_- = 1.8$인 정규분포를 $\beta<0$으로 자른 조각**을 붙인 것이다. 각 조각의 질량은 $\int_0^\infty$와 $\int_{-\infty}^0$을 계산해

    $$
    w_+ = e^{m_+^2/2}\,\Phi(m_+) = e^{0.02}\cdot 0.42074 = 0.4292,
    \qquad
    w_- = e^{m_-^2/2}\,\Phi(-m_-) = e^{1.62}\cdot 0.035930 = 0.1816
    $$

    에 비례한다($e^{0.02} = 1.02020$, $e^{1.62} = 5.05309$). 절단정규의 평균 공식 $E[\,\cdot\mid \beta>0] = m + \varphi(m)/\Phi(m)$, $E[\,\cdot\mid\beta<0] = m - \varphi(m)/\Phi(-m)$을 쓰면

    $$
    E_+ = -0.2 + \frac{0.39104}{0.42074} = 0.7294,
    \qquad
    E_- = 1.8 - \frac{0.078950}{0.035930} = -0.3973
    $$

    이고($\varphi$는 짝함수라 $\varphi(-0.2) = \varphi(0.2)$다) 따라서

    $$
    E[\beta\mid y] = \frac{w_+E_+ + w_-E_-}{w_+ + w_-}
    = \frac{0.4292 \cdot 0.7294 + 0.1816 \cdot (-0.3973)}{0.4292+0.1816}
    = \frac{0.2409}{0.6108}
    = 0.3945
    $$

    다(반올림하지 않고 셈하면 $0.39450$). 사후확률 $P(\beta>0\mid y) = w_+/(w_++w_-) = 0.7028$로 질량의 $70\%$가 양수 쪽에 있다. **MAP는 $0$, 사후평균은 $0.3945$, 심지어 $P(\beta = 0 \mid y) = 0$이다** — 사후분포는 연속이라 어느 한 점에도 질량을 주지 않는다.

    **(4)** 라쏘의 희소성은 사후분포의 성질이 아니라 **사후최빈값을 고르는 행동의 성질**이다. 라플라스 사전분포는 $\beta_j = 0$에 양의 확률을 주지 않으므로 "그 변수는 쓸모없다"는 베이즈적 결론을 담고 있지 않다. 밀도의 꼭짓점이 $0$에서 뾰족하다는 사실이 최적화의 해를 $0$으로 끌어당기는 것일 뿐이며, 같은 사전분포로 사후평균을 쓰면 계수는 전부 $0$이 아니다. 그래서 라쏘의 $0$을 "이 변수는 없다"로 읽으면 안 되고(연습문제 9의 누수와 겹쳐 생각하면 더욱 그렇다), 진짜로 변수를 골라내는 베이즈 모형이 필요하면 $0$에 점질량을 두는 **스파이크-앤-슬랩** 사전분포를 써야 한다. $\square$

---

## 정리하며

능형회귀를 **실제로 적합**해 보았다.

- **$\lambda$ 를 키우면 모든 계수가 $0$ 쪽으로 모인다.** 정칙화 경로를 그리면 그 수축이 한눈에 보이며, **어느 것도 $0$ 에 닿지 않는다.**
- **상관된 설명변수들을 비슷한 크기로 나눠 갖는다.** 라쏘가 하나를 고르고 나머지를 버리는 것과 대조되며, **공선 집단을 다룰 때 능형이 안정적인 이유**다.
- **$p>n$ 에서도 작동한다.** OLS 가 아예 정의되지 않는 상황에서 해를 준다.
- **$\lambda$ 선택이 남은 문제다.** 교차검증으로 고르며, 이 장 뒤에서 다룬다.
- **계수를 효과크기로 읽으면 안 된다.** 축소되어 있으므로 체계적으로 과소평가하며, **예측이 목적일 때 쓰는 도구**다(1장의 예측 대 추론).

다음 절 **라쏘 정식화와 희소성**으로 넘어간다. 벌점을 $\ell_2$ 에서 $\ell_1$ 로 바꾸는 순간 계수가 줄어드는 데 그치지 않고 **정확히 $0$ 이 된다.**
