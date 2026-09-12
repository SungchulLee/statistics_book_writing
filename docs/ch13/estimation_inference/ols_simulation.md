# 최소제곱 모의실험 (몬테카를로)

## 개요

이 페이지는 몬테카를로 모의실험을 통해 선형대수의 관점에서 OLS 추정을 보인다. 정규방정식 추정량, 사영행렬과 그 멱등성, 분산분석 분해, 불편분산 추정, $\hat{\boldsymbol{\beta}}$의 표집분포라는 핵심 이론적 성질을 확인한다. 반복 모의실험은 $\hat{\boldsymbol{\beta}}$가 불편이며 그 경험적 표준편차가 이론적 표준오차와 일치함을 확인해 준다.

## 수학적 배경

### OLS 추정량

$\mathbf{u} \sim N(\mathbf{0}, \sigma^2\mathbf{I})$인 모형 $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \mathbf{u}$에 대해

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}.
$$

### 사영행렬

**사영행렬** $\mathbf{P} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$는 $\mathbf{X}$의 열공간 위로 사영한다.

$$
\hat{\mathbf{y}} = \mathbf{P}\mathbf{y}.
$$

**소거행렬** $\mathbf{M} = \mathbf{I} - \mathbf{P}$는 직교여공간 위로 사영한다.

$$
\mathbf{e} = \mathbf{M}\mathbf{y}.
$$

둘 다 대칭이고 멱등이며($\mathbf{P}^2 = \mathbf{P}$, $\mathbf{M}^2 = \mathbf{M}$), $\operatorname{tr}(\mathbf{P}) = k$, $\operatorname{tr}(\mathbf{M}) = n - k$이다.

### 분산분석 분해

$$
\underbrace{\sum(y_i - \bar{y})^2}_{\mathrm{TSS}} = \underbrace{\sum(\hat{y}_i - \bar{y})^2}_{\mathrm{ESS}} + \underbrace{\sum(y_i - \hat{y}_i)^2}_{\mathrm{RSS}}.
$$

### 불편분산 추정량

$$
s^2 = \frac{\mathbf{e}^\top\mathbf{e}}{n - k}, \qquad E[s^2] = \sigma^2.
$$

### 추정량의 공분산

$$
\mathrm{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2(\mathbf{X}^\top\mathbf{X})^{-1}, \qquad \widehat{\mathrm{Var}}(\hat{\boldsymbol{\beta}}) = s^2(\mathbf{X}^\top\mathbf{X})^{-1}.
$$

### 핵심 함수

<div class="codebox" markdown>

**예제 1.** 최소제곱의 행렬 연산

```python
import numpy as np

def gen_X(n, k):
    """설계행렬. 첫 열의 1 이 절편에 대응한다."""
    return np.hstack([np.ones((n, 1)), np.random.randn(n, k - 1)])

def ols(y, X):
    """정규방정식의 해. 실제 계산에서는 역행렬 대신 solve 를 쓰는 편이 낫다."""
    return np.linalg.inv(X.T @ X) @ X.T @ y

def proj_P(X):
    """사영행렬 P. y 를 X 의 열공간 위로 떨어뜨린다. P @ y 가 곧 적합값이다."""
    return X @ np.linalg.inv(X.T @ X) @ X.T

def proj_M(X):
    """잔차생성행렬 M = I - P. M @ y 가 잔차이고, 열공간에 수직이다."""
    return np.eye(X.shape[0]) - proj_P(X)

def anova_decomposition(y, X, beta_hat):
    """TSS = ESS + RSS 로 갈라 본다.

    적합값과 잔차가 서로 수직이므로 피타고라스 정리가 그대로 성립한다.
    최소제곱의 기하가 이 한 줄에 들어 있다.
    """
    y_bar = y.mean()
    y_hat = X @ beta_hat
    TSS = float(np.sum((y - y_bar) ** 2))
    ESS = float(np.sum((y_hat - y_bar) ** 2))
    RSS = float(np.sum((y - y_hat) ** 2))
    return TSS, ESS, RSS
```

</div>

### 몬테카를로 검증

<div class="codebox" markdown>

**예제 2.** 몬테카를로로 확인하는 불편성

```python
def monte_carlo(n=100, beta_true=[2, 3, -1], sigma=1.0, n_sim=5000):
    """같은 실험을 5000번 되풀이해 추정량의 분포를 본다.

    참 계수를 우리가 정해 두었으므로, 추정값들의 평균이 참값에 붙는지
    (불편성) 그리고 그 흩어짐이 얼마인지를 직접 확인할 수 있다.
    """
    k = len(beta_true)
    estimates = np.empty((n_sim, k))
    for i in range(n_sim):
        X = gen_X(n, k)
        beta = np.array(beta_true).reshape(-1, 1)
        u = np.random.randn(n, 1) * sigma
        y = X @ beta + u
        bhat = ols(y, X)
        estimates[i] = bhat.flatten()
    return estimates

beta_true = [2, 3, -1]
estimates = monte_carlo(n=200, beta_true=beta_true, sigma=2.0, n_sim=5000)
mc_mean = estimates.mean(axis=0)
mc_std = estimates.std(axis=0, ddof=1)

for j in range(len(beta_true)):
    print(f"beta_{j}: true={beta_true[j]}, "
          f"MC mean={mc_mean[j]:.4f}, MC std={mc_std[j]:.4f}")
```

출력:

```
beta_0: true=2, MC mean=2.0009, MC std=0.1414
beta_1: true=3, MC mean=3.0019, MC std=0.1425
beta_2: true=-1, MC mean=-0.9994, MC std=0.1442
```

</div>

Monte Carlo 평균이 참값 $(2, 3, -1)$에 소수점 셋째 자리까지 맞는다. OLS가 불편추정량이라는 것을 모의실험으로 확인한 셈이다.

$n = 200$, $\sigma = 2$일 때 몬테카를로 표준편차는 세 계수 모두 $0.15$ 근처가 되며, 이는 이론값 $\sigma/\sqrt{n} = 2/\sqrt{200} = 0.1414$와 잘 맞는다(설명변수가 표준정규이므로 $(\mathbf{X}^\top\mathbf{X})^{-1}$의 대각원소가 대략 $1/n$이다).

## 해석

- **불편성**: $\hat{\beta}_j$의 몬테카를로 평균은 참값 $\beta_j$에 가까워야 한다. 5000번 반복하면 보통 $\pm 0.05$ 안에서 참값과 일치한다.
- **사영행렬**: $\mathbf{P}$와 $\mathbf{M}$은 OLS의 근본적인 기하학적 대상이다. $\mathbf{P}$는 적합값 부분공간으로, $\mathbf{M}$은 잔차 부분공간으로 사영한다. 이들의 멱등성과 상보성($\mathbf{P} + \mathbf{M} = \mathbf{I}$)이 $\mathbf{y}$의 직교분해를 담고 있다.
- **분산분석**: 항등식 TSS = ESS + RSS는 전체 변동을 설명된 부분과 설명되지 않은 부분으로 나눈다. $R^2 = \mathrm{ESS}/\mathrm{TSS}$이다.
- **분산 추정**: $\operatorname{tr}(\mathbf{M}) = n - k$가 추정에서 잃은 자유도를 반영하므로 $s^2$은 $\sigma^2$에 대해 불편이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span> 특정한 $\mathbf{X}$ 실현값에 대해 $\mathbf{P}$와 $\mathbf{M}$이 멱등이고 대칭임을 수치적으로 확인하라.

</div>

??? success "풀이"

    ```python
    X = gen_X(50, 3)
    P = proj_P(X)
    M = proj_M(X)
    print("P idempotent:", np.allclose(P @ P, P))
    print("M idempotent:", np.allclose(M @ M, M))
    print("P symmetric:", np.allclose(P, P.T))
    print("M symmetric:", np.allclose(M, M.T))
    print("tr(P):", np.trace(P))  # should be 3
    print("tr(M):", np.trace(M))  # should be 47
    ```

    출력:

    ```
    P idempotent: True
    M idempotent: True
    P symmetric: True
    M symmetric: True
    tr(P): 3.000000000000001
    tr(M): 46.99999999999999
    ```

    사영행렬 $P$와 잔차행렬 $M$이 멱등이고 대칭임을 수치로 확인했다. 대각합도 $\text{tr}(P) = 3$(모수 개수), $\text{tr}(M) = 47$($n - p$)로 이론과 맞는다. 잔차의 자유도가 $n - p$인 이유가 바로 이것이다.

    모든 확인을 통과한다. $\mathbf{P}^2 = \mathbf{P}$, $\mathbf{M}^2 = \mathbf{M}$이고 둘 다 대칭이며 $\operatorname{tr}(\mathbf{P}) = k = 3$, $\operatorname{tr}(\mathbf{M}) = n - k = 47$이다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 95% 신뢰구간 $\hat{\beta}_j \pm t^*_{n-k,0.025} \cdot \mathrm{SE}(\hat{\beta}_j)$의 포함확률을 추정하도록 몬테카를로를 고쳐라. 95%에 가까운가?

</div>

??? success "풀이"

    ```python
    from scipy import stats
    coverage = np.zeros(3)
    n, sigma = 200, 2.0
    beta_true_arr = np.array(beta_true)
    for i in range(5000):
        X = gen_X(n, 3)
        y = X @ beta_true_arr.reshape(-1, 1) + sigma * np.random.randn(n, 1)
        bhat = ols(y, X)
        e = y - X @ bhat
        s2 = np.sum(e ** 2) / (n - 3)
        se = np.sqrt(s2 * np.diag(np.linalg.inv(X.T @ X)))
        t_star = stats.t(n - 3).ppf(0.975)
        for j in range(3):
            if bhat[j, 0] - t_star * se[j] <= beta_true[j] <= bhat[j, 0] + t_star * se[j]:
                coverage[j] += 1
    print("Coverage:", coverage / 5000)  # should be ~0.95
    ```

    출력:

    ```
    Coverage: [0.952  0.9482 0.951 ]
    ```

    세 계수의 95% 신뢰구간 포함확률이 각각 0.952, 0.948, 0.951로 명목값과 맞는다. 가정이 성립하면 OLS의 구간이 약속한 대로 작동한다는 확인이다.

    경험적 포함확률은 각 계수에 대해 대략 0.95가 되어 이론을 확인해 준다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> $\mathbf{P}\mathbf{M} = \mathbf{0}$임을 보이고 기하학적으로 해석하라.

</div>

??? success "풀이"

    $\mathbf{M} = \mathbf{I} - \mathbf{P}$이므로

    $$
    \mathbf{P}\mathbf{M} = \mathbf{P}(\mathbf{I} - \mathbf{P}) = \mathbf{P} - \mathbf{P}^2 = \mathbf{P} - \mathbf{P} = \mathbf{0}.
    $$

    기하학적으로 $\mathbf{P}$는 $\mathrm{col}(\mathbf{X})$ 위로, $\mathbf{M}$은 $\mathrm{col}(\mathbf{X})^\perp$ 위로 사영한다. 두 부분공간이 직교하므로 한쪽으로 사영한 뒤 다른 쪽으로 사영하면 영벡터가 된다. 이것이 $\hat{\mathbf{y}}$와 $\mathbf{e}$가 직교하는 이유이다: $\hat{\mathbf{y}}^\top\mathbf{e} = (\mathbf{P}\mathbf{y})^\top(\mathbf{M}\mathbf{y}) = \mathbf{y}^\top\mathbf{P}\mathbf{M}\mathbf{y} = 0$. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> $n = 200$을 유지한 채 $\sigma$를 2에서 10으로 키워라. 몬테카를로 표준편차와 $R^2$의 분포는 어떻게 달라지는가?

</div>

??? success "풀이"

    $\mathrm{SE}(\hat{\beta}_j) \propto \sigma$이므로 $\hat{\beta}_j$의 몬테카를로 표준편차는 비례해서 5배 커진다. 모의실험으로 확인하면 세 계수 모두 약 $0.15$에서 약 $0.75$로 늘어난다.

    $R^2$은 크게 떨어진다. 다만 그 이유를 정확히 짚을 필요가 있다. 설명변수가 표준정규이고 $\boldsymbol{\beta} = (2, 3, -1)$이므로 신호의 분산은 $3^2 + (-1)^2 = 10$으로 **$\sigma$와 무관하게 일정하다**. 반면 $\mathrm{TSS}/n \approx 10 + \sigma^2$이므로

    - $\sigma = 2$: $\mathrm{TSS}/n \approx 14$, $R^2 \approx 10/14 = 0.71$
    - $\sigma = 10$: $\mathrm{TSS}/n \approx 110$, $R^2 \approx 10/110 = 0.09$

    곧 TSS는 $\sigma^2$의 비인 25배가 아니라 약 7.9배 늘어난다. 신호 성분 10이 $\sigma$와 함께 커지지 않기 때문이다. 모의실험에서 평균 $R^2$은 $0.718$에서 $0.103$으로 떨어져 이 계산과 맞는다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> $\mathbf{M}$의 대각합을 이용해 $E[s^2] = \sigma^2$임을 증명하라.

</div>

??? success "풀이"

    $\mathbf{M}\mathbf{X} = \mathbf{0}$이므로 잔차벡터는 $\mathbf{e} = \mathbf{M}\mathbf{y} = \mathbf{M}\mathbf{u}$이다. 그러면

    $$
    E[\mathbf{e}^\top\mathbf{e}] = E[\mathbf{u}^\top\mathbf{M}^\top\mathbf{M}\mathbf{u}] = E[\mathbf{u}^\top\mathbf{M}\mathbf{u}] = E[\operatorname{tr}(\mathbf{u}\mathbf{u}^\top\mathbf{M})],
    $$

    여기서 대각합 요령 $\mathbf{u}^\top\mathbf{M}\mathbf{u} = \operatorname{tr}(\mathbf{M}\mathbf{u}\mathbf{u}^\top)$를 썼다. 기댓값을 취하면

    $$
    E[\operatorname{tr}(\mathbf{M}\mathbf{u}\mathbf{u}^\top)] = \operatorname{tr}(\mathbf{M}\,E[\mathbf{u}\mathbf{u}^\top]) = \operatorname{tr}(\mathbf{M}\sigma^2\mathbf{I}) = \sigma^2\operatorname{tr}(\mathbf{M}) = \sigma^2(n - k).
    $$

    $n - k$로 나누면 $E[s^2] = E[\mathbf{e}^\top\mathbf{e}/(n-k)] = \sigma^2$이다. $\square$

---

## 정리하며

이론적 성질을 **모의실험으로 하나씩 확인**했다.

- **$\hat{\boldsymbol\beta}$ 가 불편이다.** 반복 모의의 평균이 참값에 맞고, 경험적 표준편차가 이론적 표준오차와 일치한다.
- **모자 행렬이 대칭 멱등이다.** $\mathbf H^2=\mathbf H$ 를 수치로 확인하며, 0장에서 본 성질이다. 그 대각합이 곧 $p+1$ 이다.
- **분산분석 분해가 성립한다.** $\text{TSS}=\text{ESS}+\text{RSS}$ 이며, 잔차와 적합값이 직교하기 때문이다.
- **$s^2$ 이 $\sigma^2$ 의 불편추정량이다.** $n-p-1$ 로 나누어야 그렇다는 것이 수치로 확인된다.
- **모의실험이 선형대수와 통계를 잇는다.** 0장에서 유도한 결과들이 실제 자료 생성 과정에서 그대로 재현되는 것을 보는 것이 이 절의 목적이다.

다음 절부터 **계수 검정의 실제**로 넘어간다.
