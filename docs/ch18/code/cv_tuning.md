# 교차검증과 람다 조율

## 개요

정칙화 모수 $\lambda$는 능형회귀, 라쏘, 엘라스틱넷에서 벌점의 강도를 조절한다. $\lambda$를 잘
고르는 일은 결정적이다. 너무 작으면 모형이 과적합하고, 너무 크면 과소적합한다. 교차검증(CV)은
표본 밖 예측오차를 추정함으로써 $\lambda$를 자료에 근거해 원리적으로 선택하는 방법이다.

## 조율 문제

어떤 벌점회귀에서든 표본 내 손실은 $\lambda$가 0으로 줄어들수록 단조적으로 감소한다(정칙화가
약할수록 훈련자료에 더 잘 맞는다). 그러나 표본 밖 예측오차는 대개 U자 곡선을 그린다.

$$
\text{CV}(\lambda) = \frac{1}{K}\sum_{k=1}^{K} \text{MSE}^{(-k)}(\lambda),
$$

여기서 $\text{MSE}^{(-k)}$는 나머지 $K - 1$개 겹으로 모형을 적합했을 때 겹 $k$에서의
평균제곱오차다. 최적 $\lambda$는 이 곡선을 최소화한다.

## K-겹 교차검증

표준 절차는 다음과 같다.

1. 자료를 크기가 거의 같은 $K$개의 겹으로 무작위 분할한다.
2. 격자 $\{\lambda_1, \dots, \lambda_M\}$의 각 후보 $\lambda$에 대해:
    - $k = 1, \dots, K$에 대해 겹 $k$를 남겨 두고 나머지 자료로 모형을 적합한 뒤 겹 $k$에서
      예측오차를 계산한다.
    - $K$개의 예측오차를 평균하여 $\text{CV}(\lambda)$를 얻는다.
3. $\hat{\lambda} = \arg\min_\lambda \text{CV}(\lambda)$를 선택한다.

$K = 5$ 또는 $K = 10$이 흔히 쓰인다.

## 람다 격자

후보 $\lambda$ 격자는 보통 로그 척도로 잡는다.

$$
\lambda_1 > \lambda_2 > \cdots > \lambda_M, \quad \text{where } \lambda_m = 10^{a + (b-a)\frac{m-1}{M-1}}
$$

여기서 $[a, b]$는 적당한 범위다(예: $a = -4$, $b = 2$). 실무에서는 다음과 같이 잡는다.

- $\lambda_{\max}$는 모든 라쏘 계수를 0으로 만드는 가장 작은 값
  $\lambda_{\max} = \frac{1}{n}\|X^\top y\|_\infty$이다.
- 격자는 $\lambda_{\max}$에서 시작해 그 작은 배수 $\epsilon \cdot \lambda_{\max}$까지 내려간다
  (예: $\epsilon = 10^{-4}$).

## 코드: 기본 교차검증 시연

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

완전한 구현에서는 자료를 겹으로 나누고, $\lambda$ 격자 위를 순회하며, 각 (겹, $\lambda$) 조합의
MSE를 기록한다.

## 1-표준오차 규칙

$\hat{\lambda}$를 CV 곡선의 정확한 최소점으로 잡는 대신, CV 오차가 최솟값으로부터 1 표준오차
이내인 $\lambda$ 중 가장 큰 것을 고르는 보수적 규칙이 널리 쓰인다.

$$
\hat{\lambda}_{1\text{SE}} = \max\left\{ \lambda : \text{CV}(\lambda) \le \text{CV}(\hat{\lambda}) + \text{SE}(\hat{\lambda}) \right\}.
$$

이 규칙은 예측오차를 거의 늘리지 않으면서 더 단순한(더 강하게 정칙화된) 모형을 선호한다.

## 실무상의 고려사항

- **표준화.** 교차검증 전에 항상 설명변수를 표준화하라. 단, 표준화에 쓰는 평균과 표준편차는
  전체 자료가 아니라 훈련 겹에서만 계산해야 자료 누설을 피할 수 있다.
- **중첩 교차검증.** 최종 모형의 일반화 오차까지 추정하려면 중첩(이중) 교차검증을 쓴다.
  바깥 루프가 검정오차를 추정하고 안쪽 루프가 $\lambda$를 선택한다.
- **계산 비용.** 온기 시작(직전 $\lambda$의 해에서 좌표하강을 시작하는 것)은 경로 계산을
  극적으로 빠르게 한다.
- **무작위 분할.** 분할 전에 섞거나 반복 교차검증을 쓰면 특정 분할에 대한 민감도가 줄어든다.

## 해석

- CV 곡선은 U자(적어도 비단조) 모양이어야 한다. 가장 작은 $\lambda$에서도 여전히 감소 중이라면
  격자를 더 작은 쪽으로 넓혀라.
- 가장 큰 $\lambda$에서도 여전히 감소 중이라면 그 자료에는 정칙화가 필요 없을 수도 있다.
- 1SE 규칙은 예측오차를 조금 희생하는 대신 더 희소하고 해석하기 좋은 모형을 준다.

## 연습문제

**연습문제 1.** 라쏘에 대해 $\lambda_{\max} = \frac{1}{n}\|X^\top y\|_\infty$를 유도하라. 즉
$\lambda \ge \lambda_{\max}$이면 라쏘 해가 $\hat{\beta} = 0$임을 보여라.

??? success "연습문제 1 풀이"

    라쏘 목적함수는 $f(\beta) = \frac{1}{2n}\|y - X\beta\|_2^2 + \lambda\|\beta\|_1$이다.
    $\beta = 0$에서의 부분미분은

    $$
    \partial f(0) = \left\{-\frac{1}{n}X^\top y + \lambda s : s \in \partial\|\cdot\|_1(0)\right\} = \left\{-\frac{1}{n}X^\top y + \lambda s : s_j \in [-1,1]\right\}
    $$

    이다. 최적성 조건 $0 \in \partial f(0)$은 $|s_j| \le 1$인 어떤 $s$에 대해
    $\frac{1}{n}X^\top y = \lambda s$가 성립할 것을 요구한다. 이는 모든 $j$에 대해
    $\frac{1}{n}|X_j^\top y| \le \lambda$일 때, 그리고 그때에만 가능하다. 즉
    $\lambda \ge \frac{1}{n}\|X^\top y\|_\infty = \lambda_{\max}$이다. $\square$

---

**연습문제 2.** 교차검증 전에 전체 자료로 설명변수를 표준화하면 왜 자료 누설이 생기는지
설명하고, 올바른 절차를 서술하라.

??? success "연습문제 2 풀이"

    검증 겹까지 포함한 전체 자료에서 $\bar{x}_j$와 $s_j$를 계산해 표준화하면, 검증 겹의 자료가
    자기 자신에게 적용되는 변환에 영향을 미친다. 결국 모형이 훈련 과정에서 검증 겹의 정보를
    간접적으로 "본" 셈이 되어 CV 오차 추정이 낙관적으로 편향된다.

    **올바른 절차:** 각 CV 반복 안에서 훈련 겹만으로 평균과 표준편차를 계산한 뒤, 같은 변환을
    남겨 둔 겹에 적용한다.

    ```python
    for k in range(K):
        X_train, X_val = X[train_idx], X[val_idx]
        mu = X_train.mean(axis=0)
        sigma = X_train.std(axis=0)
        X_train_s = (X_train - mu) / sigma
        X_val_s = (X_val - mu) / sigma
        # fit on X_train_s, evaluate on X_val_s
    ```

    이렇게 하면 검증 겹이 진정으로 미관측 상태로 남는다. $\square$

---

**연습문제 3.** 1-표준오차 규칙을 구현하라. 배열 `lambdas`, `cv_mean`, `cv_se`(각 $\lambda$의
CV MSE 평균과 표준오차)가 주어졌을 때 $\hat{\lambda}_{1\text{SE}}$를 반환하는 함수를 작성하라.

??? success "연습문제 3 풀이"

    ```python
    def one_se_rule(lambdas, cv_mean, cv_se):
        """
        Select the largest lambda whose CV error is within
        one SE of the minimum CV error.
        """
        best_idx = np.argmin(cv_mean)
        threshold = cv_mean[best_idx] + cv_se[best_idx]

        # Among lambdas with CV error <= threshold, pick the largest
        candidates = np.where(cv_mean <= threshold)[0]
        best_1se_idx = candidates[np.argmax(lambdas[candidates])]
        return lambdas[best_1se_idx]
    ```

    이 함수는 최소 CV 오차를 찾아 1 표준오차를 더해 문턱값을 만들고, 그 문턱값 이내에서 가장
    강하게 정칙화된 모형(가장 큰 $\lambda$)을 고른다.

    !!! warning "흔한 함정"
        후보 인덱스에서 곧바로 `candidates.max()`나 `candidates.min()`을 쓰면 안 된다.
        `lambdas` 배열이 오름차순인지 내림차순인지에 따라 정반대 결과가 나오기 때문이다.
        `sklearn.linear_model.lasso_path`가 돌려주는 `alphas`는 **내림차순**이라
        `candidates.max()`는 가장 작은 $\lambda$를 고르게 되어 규칙의 취지와 반대가 된다.
        위처럼 $\lambda$ 값 자체에 `argmax`를 취하면 정렬 순서와 무관하게 항상 옳다.
    $\square$

---

**연습문제 4.** 인공자료($n = 200$, $p = 30$, 참 계수 중 5개만 0이 아님)에 대해 로그 등간격
$\lambda$ 50개 위에서 라쏘의 5-겹 교차검증을 수행하라. 오차막대($\pm 1$ SE)를 포함한 CV 곡선을
그리고 $\hat{\lambda}_{\min}$과 $\hat{\lambda}_{1\text{SE}}$를 모두 표시하라.

??? success "연습문제 4 풀이"

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)
    n, p = 200, 30
    X = np.random.randn(n, p)
    beta_true = np.zeros(p)
    beta_true[:5] = [3, -2, 4, -1, 2]
    y = X @ beta_true + np.random.randn(n)

    X_s = StandardScaler().fit_transform(X)
    lambdas = np.logspace(1, -3, 50)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    mse_folds = np.zeros((len(lambdas), 5))
    for fold_idx, (tr, va) in enumerate(kf.split(X_s)):
        mu, sig = X_s[tr].mean(0), X_s[tr].std(0)
        Xtr = (X_s[tr] - mu) / sig
        Xva = (X_s[va] - mu) / sig
        for i, lam in enumerate(lambdas):
            m = Lasso(alpha=lam, max_iter=10000).fit(Xtr, y[tr])
            mse_folds[i, fold_idx] = np.mean((y[va] - m.predict(Xva))**2)

    cv_mean = mse_folds.mean(axis=1)
    cv_se = mse_folds.std(axis=1) / np.sqrt(5)

    best_idx = np.argmin(cv_mean)
    lam_min = lambdas[best_idx]
    lam_1se = one_se_rule(lambdas, cv_mean, cv_se)

    plt.errorbar(np.log10(lambdas), cv_mean, yerr=cv_se, fmt='o-', ms=4)
    plt.axvline(np.log10(lam_min), color='red', ls='--', label='lambda_min')
    plt.axvline(np.log10(lam_1se), color='blue', ls='--', label='lambda_1SE')
    plt.xlabel('log10(lambda)')
    plt.ylabel('CV MSE')
    plt.legend()
    plt.show()
    ```

    실행하면 $\hat{\lambda}_{\min} = 0.0518$에서 CV MSE는 $1.0154$(SE $= 0.0611$)이고,
    1SE 규칙은 $\hat{\lambda}_{1\text{SE}} = 0.1099$를 고르며 그때 CV MSE는 $1.0637$이다.
    전체 자료에 다시 적합하면 $\hat{\lambda}_{\min}$은 0이 아닌 계수를 16개 남기지만
    $\hat{\lambda}_{1\text{SE}}$는 10개만 남긴다. 즉 예측오차가 5% 가까이 늘어나는 대신
    모형이 훨씬 희소해진다. 곡선은 특징적인 U자 모양이며, 잡음변수 25개 중 상당수가 여전히
    선택된다는 점은 교차검증이 **예측**을 최적화할 뿐 **변수선택**을 최적화하지는 않음을
    보여준다. $\square$

---

**연습문제 5.** 능형회귀의 하나 남기기 교차검증(LOOCV)이 닫힌 형태
$\text{CV}_{\text{LOO}} = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{y_i - \hat{y}_i}{1 - h_{ii}}\right)^2$
로 계산됨을 증명하라. 여기서 $H = X(X^\top X + \lambda I)^{-1}X^\top$는 능형 모자행렬이고
$h_{ii}$는 그 $i$번째 대각원소다.

??? success "연습문제 5 풀이"

    관측치 $i$를 제거하면 나머지 $n-1$개로 적합한 능형 모형이 예측 $\hat{y}_{(-i), i}$를 준다.
    셔먼-모리슨 공식에 의해, $X$에서 $i$번째 행과 $y$에서 $i$번째 원소를 빼고 다시 적합하는
    것은

    $$
    \hat{y}_{(-i), i} = x_i^\top (X_{(-i)}^\top X_{(-i)} + \lambda I)^{-1} X_{(-i)}^\top y_{(-i)}
    $$

    와 같다. 선형모형의 표준 결과(OLS의 LOOCV 지름길을 확장한 것)에 따르면 능형회귀의 LOO
    잔차는

    $$
    y_i - \hat{y}_{(-i),i} = \frac{y_i - \hat{y}_i}{1 - h_{ii}}
    $$

    를 만족한다. 여기서 $\hat{y}_i = h_i^\top y$는 전체 자료로 적합한 능형 예측값이고
    $h_{ii}$는 모자행렬 $H = X(X^\top X + \lambda I)^{-1}X^\top$의 $(i,i)$ 원소다.

    따라서

    $$
    \text{CV}_{\text{LOO}} = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{y_i - \hat{y}_i}{1 - h_{ii}}\right)^2
    $$

    이다. 이 식은 전체 모형을 한 번 적합하고 $H$의 대각만 구하면 되므로, 순진한 방법의
    $O(n^2 p^2)$ 대신 $O(np^2)$에 계산된다. $\square$
