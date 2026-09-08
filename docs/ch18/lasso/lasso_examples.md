# 라쏘 회귀 예제

## 개요

라쏘(Least Absolute Shrinkage and Selection Operator) 회귀는 최소제곱 목적함수에 $L_1$
벌점을 더한다. 이 벌점은 계수를 축소할 뿐 아니라 일부를 정확히 0으로 만든다. 이 절에서는
좌표하강으로 라쏘를 처음부터 구현하고, 정칙화 경로를 시각화하며, 교차검증으로 조율모수
$\lambda$를 선택한다.

## 라쏘 목적함수

$X \in \mathbb{R}^{n \times p}$와 $y \in \mathbb{R}^n$이 주어졌을 때 라쏘는

$$
\hat{\beta}^{\text{lasso}} = \arg\min_{\beta} \left\{ \frac{1}{2n}\| y - X\beta \|_2^2 + \lambda \sum_{j=1}^{p} |\beta_j| \right\}
$$

를 푼다. 능형회귀와 달리 $L_1$ 벌점 $\lambda \|\beta\|_1$은 희소한 해를 만든다. $\lambda$가
충분히 크면 일부 계수가 정확히 0이 된다.

## 연성 문턱 연산자

라쏘 좌표하강의 핵심 구성요소는 **연성 문턱**(근접) 연산자다.

$$
S(\rho,\, \lambda) = \text{sign}(\rho)\, \max(|\rho| - \lambda,\, 0) =
\begin{cases}
\rho - \lambda & \text{if } \rho > \lambda, \\
0 & \text{if } |\rho| \le \lambda, \\
\rho + \lambda & \text{if } \rho < -\lambda.
\end{cases}
$$

## 코드: 좌표하강 해법기

다음은 순환 좌표하강으로 라쏘를 푸는 순수 NumPy 구현이다.

```python
import numpy as np

def soft_threshold(rho, lam):
    """Soft-thresholding operator for coordinate descent."""
    if rho > lam:
        return rho - lam
    elif rho < -lam:
        return rho + lam
    return 0.0

def lasso_cd(X, y, lam, max_iter=1000, tol=1e-6):
    """
    Lasso regression via coordinate descent.

    Parameters
    ----------
    X   : (n, p) design matrix (should be standardised).
    y   : (n,)   response vector.
    lam : float  L1 penalty parameter.

    Returns
    -------
    beta : (p,) coefficient vector.
    """
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            r_j = y - X @ beta + X[:, j] * beta[j]
            rho_j = X[:, j] @ r_j / n
            beta[j] = soft_threshold(rho_j, lam)
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    return beta
```

각 단계에서 알고리즘은 부분잔차 $r_j = y - X\beta + X_j \beta_j$를 계산하고, 일변량
최소제곱 기울기 $\rho_j = X_j^\top r_j / n$을 구한 뒤 연성 문턱을 적용한다.

## 코드: 정칙화 경로

$\lambda$ 격자를 큰 값에서 작은 값으로 훑으면 계수들이 어떤 순서로 모형에 들어오는지 볼 수 있다.

```python
def lasso_path(X, y, lambdas):
    """Compute coefficient path over a grid of lambda values."""
    coefs = []
    for lam in lambdas:
        beta = lasso_cd(X, y, lam)
        coefs.append(beta.copy())
    return np.array(coefs)
```

## 코드: 교차검증으로 람다 선택

5-겹 교차검증으로 각 후보 $\lambda$의 예측오차를 추정한다.

```python
def cv_lasso(X, y, lambdas, folds=5):
    """K-fold cross-validated MSE for each lambda."""
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // folds
    cv_mse = np.zeros(len(lambdas))

    for k in range(folds):
        val_idx = indices[k * fold_size:(k + 1) * fold_size]
        train_idx = np.setdiff1d(indices, val_idx)
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        for i, lam in enumerate(lambdas):
            beta = lasso_cd(X_tr, y_tr, lam)
            pred = X_va @ beta
            cv_mse[i] += np.mean((y_va - pred) ** 2)
    return cv_mse / folds
```

## 코드: 전체 시연

설명변수 10개 중 3개만 실제로 관련 있는 인공자료를 만들어 전체 절차를 돌려 본다.

```python
import matplotlib.pyplot as plt

np.random.seed(42)

n, p = 150, 10
X_raw = np.random.randn(n, p)
beta_true = np.array([4.0, -3.0, 2.0, 0, 0, 0, 0, 0, 0, 0])
y = X_raw @ beta_true + np.random.randn(n) * 2

# Standardise columns
X_mean = X_raw.mean(axis=0)
X_std = X_raw.std(axis=0)
X = (X_raw - X_mean) / X_std

# Regularisation path
lambdas = np.logspace(1, -2, 60)
path = lasso_path(X, y, lambdas)

# Cross-validated lambda selection
lambdas_cv = np.logspace(1, -2, 30)
mse_cv = cv_lasso(X, y, lambdas_cv)
best_idx = int(np.argmin(mse_cv))
best_lam = lambdas_cv[best_idx]

beta_best = lasso_cd(X, y, best_lam)
n_nonzero = np.sum(np.abs(beta_best) > 1e-8)

print(f"Best lambda (5-fold CV):  {best_lam:.4f}")
print(f"Non-zero coefficients:    {n_nonzero}  (true: 3)")
print(f"Min CV MSE:               {mse_cv[best_idx]:.3f}")
```

출력:

```
Best lambda (5-fold CV):  0.2212
Non-zero coefficients:    3  (true: 3)
Min CV MSE:               4.774
```

실행하면 $\hat{\lambda} = 0.2212$, 0이 아닌 계수 3개(참값도 3개), 최소 CV MSE $4.774$를 얻는다.
추정된 계수는 $(3.546,\, -2.912,\, 1.643,\, 0, \dots, 0)$으로, 참값 $(4, -3, 2, 0, \dots, 0)$을
모두 0 쪽으로 축소한 값이다. 이 축소 편향이 라쏘가 분산을 줄이는 대가로 치르는 비용이다.

## 해석

- **희소성.** 최적 $\lambda$에서 라쏘는 실제로 관련 있는 설명변수 3개를 정확히 찾아내고 나머지
  7개를 0으로 만든다.
- **정칙화 경로.** $\lambda$가 작아짐에 따라 계수들이 하나씩 모형에 들어온다. 들어오는 순서는
  대개 변수의 중요도를 반영한다.
- **교차검증 곡선.** CV MSE 곡선은 보통 U자 모양이다. $\lambda$가 너무 크면 과소적합(편향이
  크고), 너무 작으면 과적합(분산이 크다). 최소점이 둘의 균형을 맞춘다.
- **능형회귀와의 비교.** 능형회귀라면 10개 설명변수를 모두 작지만 0이 아닌 계수로 남긴다.
  라쏘는 정확한 희소성을 얻는다.

## 연습문제

**연습문제 1.** $S(\rho, \lambda)$가 $\lambda |\cdot|$의 근접 연산자임을 손으로 확인하라. 즉
$S(\rho, \lambda) = \arg\min_{z} \left\{ \frac{1}{2}(z - \rho)^2 + \lambda |z| \right\}$
임을 보여라.

??? success "풀이"

    $g(z) = \frac{1}{2}(z - \rho)^2 + \lambda |z|$라 하고 세 경우로 나눈다.

    **경우 1: $z > 0$.** $g(z) = \frac{1}{2}(z-\rho)^2 + \lambda z$이고
    $g'(z) = z - \rho + \lambda = 0$에서 $z^* = \rho - \lambda$를 얻는다. 이 값이 양수인 것은
    $\rho > \lambda$일 때뿐이다.

    **경우 2: $z < 0$.** $g(z) = \frac{1}{2}(z-\rho)^2 - \lambda z$이고
    $g'(z) = z - \rho - \lambda = 0$에서 $z^* = \rho + \lambda$를 얻는다. 이 값이 음수인 것은
    $\rho < -\lambda$일 때뿐이다.

    **경우 3: $z = 0$.** 부분미분 조건 $0 \in \{-\rho\} + \lambda[-1, 1]$은
    $|\rho| \le \lambda$를 요구한다.

    셋을 합치면 $z^* = \text{sign}(\rho)\max(|\rho| - \lambda, 0) = S(\rho, \lambda)$이다.
    $\square$

---

**연습문제 2.** 좌표하강 알고리즘에서 $\beta_j$를 갱신하기 전에 부분잔차
$r_j = y - X\beta + X_j \beta_j$를 계산해야 하는 이유를 설명하라. 대신 전체 잔차
$r = y - X\beta$를 쓰면 무엇이 잘못되는가?

??? success "풀이"

    $\beta_j$에 대한 좌표하강 갱신은 다른 계수를 모두 고정한 채 라쏘 목적함수를 $\beta_j$에
    대해 최소화하는 것이다. 부분잔차 $r_j$는 현재 적합에서 $j$번째 변수의 기여를 제거하므로,
    갱신은 $X_j$와 **$X_j$ 자신의 기여를 뺀** 잔차 사이의 상관에만 의존하게 된다.

    전체 잔차 $r = y - X\beta$를 쓰면 $X_j \beta_j$가 이미 빠져 있는 상태이므로 $j$번째 변수의
    기여를 이중으로 차감하는 셈이 되어 갱신이 편향된다. 구체적으로 $\rho_j$가
    $X_j^\top(y - X_{-j}\beta_{-j})/n$이 아니라 $X_j^\top(y - X\beta)/n$이 되고, 반복열은 라쏘
    해로 수렴하지 않는다. $\square$

---

**연습문제 3.** `lasso_cd` 함수가 온기 시작(즉 항상 0에서 출발하는 대신 초기값 $\beta^{(0)}$을
받도록)을 지원하도록 수정하라. 온기 시작이 정칙화 경로 계산을 왜 빠르게 하는지 설명하라.

??? success "풀이"

    ```python
    def lasso_cd_warm(X, y, lam, beta_init=None, max_iter=1000, tol=1e-6):
        n, p = X.shape
        beta = beta_init.copy() if beta_init is not None else np.zeros(p)
        for _ in range(max_iter):
            beta_old = beta.copy()
            for j in range(p):
                r_j = y - X @ beta + X[:, j] * beta[j]
                rho_j = X[:, j] @ r_j / n
                beta[j] = soft_threshold(rho_j, lam)
            if np.max(np.abs(beta - beta_old)) < tol:
                break
        return beta
    ```

    $\lambda$를 큰 값에서 작은 값으로 훑으며 정칙화 경로를 계산할 때, 해 사상의 연속성에 의해
    $\lambda_k$에서의 해는 $\lambda_{k+1}$에서의 해와 가깝다. 직전 해를 초기값으로 쓰면 좌표하강
    반복 횟수가 수백에서 몇 회로 줄어드는 것이 보통이다. $\square$

---

**연습문제 4.** 관측치보다 설명변수가 많은 자료($n = 100$, $p = 200$)에서 계수 5개만 0이 아니게
생성하라. 교차검증으로 고른 $\lambda$에서 라쏘를 적합하고, 선택된 변수 중 참양성과 위양성의
개수를 보고하라.

??? success "풀이"

    ```python
    import numpy as np

    np.random.seed(0)
    n, p = 100, 200
    X = np.random.randn(n, p)
    X = (X - X.mean(0)) / X.std(0)
    beta_true = np.zeros(p)
    beta_true[:5] = [3, -2, 4, -1, 2]
    y = X @ beta_true + np.random.randn(n) * 1.5

    lambdas_cv = np.logspace(1, -2, 30)
    mse_cv = cv_lasso(X, y, lambdas_cv)
    best_lam = lambdas_cv[np.argmin(mse_cv)]
    beta_hat = lasso_cd(X, y, best_lam)

    selected = np.abs(beta_hat) > 1e-8
    true_support = np.abs(beta_true) > 0

    tp = np.sum(selected & true_support)
    fp = np.sum(selected & ~true_support)
    print(f"True positives:  {tp}/5")
    print(f"False positives: {fp}")
    ```

    출력:

    ```
    True positives:  5/5
    False positives: 11
    ```

    실행 결과는 $\hat{\lambda} = 0.2807$에서 참양성 5/5, 위양성 11개(선택된 변수 총 16개)다.
    즉 라쏘는 $p > n$인 고차원 희소 상황에서도 참
    신호를 **모두** 되찾지만, 예측오차를 최소화하는 $\lambda$는 잡음변수도 상당수 함께
    들여보낸다. 이는 라쏘 자체의 결함이라기보다 교차검증이 **예측**을 최적화하기 때문이다.
    지지집합을 정확히 복원하려면 더 큰 $\lambda$(1-표준오차 규칙)나 사후 라쏘, 안정성 선택
    같은 추가 절차가 필요하다. $\square$

---

**연습문제 5.** $X$의 열이 완전계수(full column rank)이면 라쏘 해가 유일하지만, $X$의 열이
일차종속이면 유일하지 않을 수 있음을 증명하라.

??? success "풀이"

    라쏘 목적함수는

    $$
    f(\beta) = \frac{1}{2n}\|y - X\beta\|_2^2 + \lambda \|\beta\|_1
    $$

    이다. 첫 항은 헤세행렬이 $\frac{1}{n}X^\top X$인 볼록 이차식이다. $X$가 완전계수이면
    $X^\top X$는 양정치이므로 $f$는 **강볼록**이다. 강볼록 함수는 최소점을 많아야 하나 가지므로
    해가 유일하다.

    $X$의 열이 일차종속이면 $X^\top X$는 양반정치일 뿐이다. 이차항은 볼록이지만 강볼록이 아니고
    $L_1$ 항 역시 볼록이지만 강볼록이 아니다. 합은 볼록이므로 최소점의 집합은 볼록집합이지만,
    한 점보다 많을 수 있다.

    **반례:** $X = [x \mid x]$(동일한 두 열)라 하자. 이때 $X\beta = x(\beta_1 + \beta_2)$이므로
    목적함수는 $s = \beta_1 + \beta_2$에만 의존하고, 주어진 $s$에 대해 $\|\beta\|_1$은
    $\beta_1, \beta_2$의 부호가 같을 때 최솟값 $|s|$를 갖는다. 따라서 문제는

    $$
    \min_{s}\ \frac{1}{2n}\|y - sx\|_2^2 + \lambda |s|
    $$

    로 환원되고, 그 최적해 $s^*$는 유일하다. 그러나 원래 문제의 해집합은
    $\{(\beta_1, \beta_2) : \beta_1 + \beta_2 = s^*,\ \beta_1\beta_2 \ge 0\}$이라는 선분 전체다.
    예컨대 $s^* = 1$이면 $(\alpha, 1-\alpha)$가 모든 $\alpha \in [0,1]$에 대해 최적이다.
    즉 해가 유일하지 않다. $\square$
