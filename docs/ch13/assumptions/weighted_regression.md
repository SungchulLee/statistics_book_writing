# 가중최소제곱

## 개요

이 페이지는 이분산에 대한 대책으로 가중최소제곱(WLS) 회귀를 보인다. 오차분산이 설명변수에 따라 선형으로 커지는 자료를 생성하고, OLS와 WLS를 모두 적합한 뒤 계수 추정값, 표준오차, 잔차그림을 비교하여 WLS가 비상수 분산을 어떻게 바로잡는지 보인다.

## 수학적 배경

오차분산이 일정하지 않으면($\mathrm{Var}(\varepsilon_i) = \sigma_i^2$) OLS는 여전히 불편이지만 더 이상 효율적이지 않고 표준오차가 틀리게 된다. WLS는 가중된 잔차제곱합을 최소화하여 이에 대처한다.

$$
\hat{\boldsymbol{\beta}}_{\text{WLS}} = \arg\min_{\boldsymbol{\beta}} \sum_{i=1}^n w_i(y_i - \mathbf{x}_i^\top\boldsymbol{\beta})^2,
$$

여기서 $w_i = 1/\sigma_i^2$이다(분산이 큰 관측값이 작은 가중치를 받는다). 행렬 형태로는

$$
\hat{\boldsymbol{\beta}}_{\text{WLS}} = (\mathbf{X}^\top\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{W}\mathbf{y},
$$

여기서 $\mathbf{W} = \mathrm{diag}(w_1, \ldots, w_n)$이다.

$\hat{\boldsymbol{\beta}}_{\text{WLS}}$의 공분산행렬은

$$
\mathrm{Var}(\hat{\boldsymbol{\beta}}_{\text{WLS}}) = (\mathbf{X}^\top\mathbf{W}\mathbf{X})^{-1}.
$$

WLS는 변환된 모형 $\sqrt{w_i}\,y_i = \sqrt{w_i}\,\mathbf{x}_i^\top\boldsymbol{\beta} + \sqrt{w_i}\,\varepsilon_i$에 OLS를 적용하는 것과 동등하다. 변환된 오차는 분산이 일정하다.

### OLS와 WLS 구현

<div class="codebox" markdown>

#### 예제 1. OLS와 WLS 구현 { .eg }

```python
import numpy as np

def ols_fit(X, y):
    """보통최소제곱. 모든 관측값에 같은 무게를 준다."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta

def wls_fit(X, y, w):
    """가중최소제곱. 관측값마다 다른 무게 w 를 준다.

    정규방정식의 X'X 와 X'y 사이에 가중행렬 W 가 끼어드는 것이 전부다.
    분산이 큰 관측값에 작은 무게를 주면 추정의 분산이 줄어든다.
    """
    W = np.diag(w)
    XtW = X.T @ W
    beta = np.linalg.solve(XtW @ X, XtW @ y)
    return beta
```

</div>

### 이분산 자료 생성과 적합

<div class="codebox" markdown>

#### 예제 2. 이분산 자료로 견주기 { .eg }

```python
np.random.seed(42)
n = 120
x = np.random.uniform(1, 10, n)

# 잡음의 크기가 x 에 따라 커진다. 전형적인 이분산 자료다.
sigma = 0.5 + 1.5 * x
y = 3.0 + 2.0 * x + np.random.normal(0, sigma)

X = np.column_stack([np.ones(n), x])

# 두 방법 모두 불편추정량이라 계수 자체는 비슷하게 나온다.
beta_ols = ols_fit(X, y)

# 분산의 역수를 무게로 쓰는 것이 최적이다. 여기서는 참 sigma 를 알고 있어
# 그대로 썼지만, 실제로는 sigma 도 자료에서 추정해야 한다.
w = 1.0 / sigma ** 2
beta_wls = wls_fit(X, y, w)
```

</div>

### 표준오차 비교

<div class="codebox" markdown>

#### 예제 3. 표준오차의 차이 { .eg }

```python
# 여기서 갈린다. OLS 의 표준오차 공식은 등분산을 전제하므로, 이분산
# 자료에서는 그 값 자체를 믿을 수 없다.
resid_ols = y - X @ beta_ols
s2_ols = np.sum(resid_ols ** 2) / (n - 2)
se_ols = np.sqrt(np.diag(s2_ols * np.linalg.inv(X.T @ X)))

# WLS 의 표준오차는 이분산을 제대로 셈에 넣은 값이고, OLS 의 것보다 작다.
# 같은 자료에서 더 정확한 결론을 얻는다는 뜻이다.
W = np.diag(w)
XtWX_inv = np.linalg.inv(X.T @ W @ X)
se_wls = np.sqrt(np.diag(XtWX_inv))

print(f"OLS:  intercept={beta_ols[0]:.3f} (SE={se_ols[0]:.3f}), "
      f"slope={beta_ols[1]:.3f} (SE={se_ols[1]:.3f})")
print(f"WLS:  intercept={beta_wls[0]:.3f} (SE={se_wls[0]:.3f}), "
      f"slope={beta_wls[1]:.3f} (SE={se_wls[1]:.3f})")
```

출력(참값은 절편 3.0, 기울기 2.0):

```text
OLS:  intercept=4.083 (SE=1.851), slope=1.842 (SE=0.312)
WLS:  intercept=3.350 (SE=0.841), slope=2.011 (SE=0.259)
```

WLS 추정값이 참값에 훨씬 가깝고(절편 $3.350$ 대 $4.083$, 기울기 $2.011$ 대 $1.842$) 표준오차도 절편에서 절반 이하, 기울기에서 17% 작다.

</div>

## 해석

- **이분산 아래의 OLS**: OLS 추정값은 여전히 불편이지만, 등분산 가정 아래에서 계산한 표준오차는 틀리다. 잔차그림에는 $x$가 커질수록 흩어짐이 커지는 특징적인 "부채꼴"이 나타난다.
- **WLS 보정**: 각 관측값에 분산의 역수로 가중치를 주어, 정밀한 관측값($x$가 작은 쪽)에 더 큰 가중치를, 잡음이 큰 관측값($x$가 큰 쪽)에 더 작은 가중치를 준다. 가중 잔차그림에서는 분산이 안정된 모습이 보인다.
- **표준오차**: 이 예에서는 WLS 표준오차가 OLS보다 작아 더 검정력 있는 검정을 준다. 다만 일반적으로 이분산 아래에서 OLS 표준오차는 이분산의 패턴에 따라 너무 클 수도, 너무 작을 수도 있다는 점에 유의하라.
- **실무 주의**: 실제로는 참 분산함수 $\sigma_i^2$을 모른다. 흔한 접근은 제곱잔차를 설명변수에 회귀시킨 예비 회귀로 추정하거나, WLS 대신 이분산 일치(HC) 표준오차를 쓰는 것이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span> 반대 패턴, 곧 분산이 $x$에 따라 줄어드는 자료를 생성하라(예: $\sigma_i = 10 - 0.8x_i$). OLS와 WLS를 적합하라. 계수 정확도의 관점에서 WLS가 여전히 OLS보다 나은가?

</div>

??? success "풀이"

    ```python
    sigma_rev = 10 - 0.8 * x
    y_rev = 3.0 + 2.0 * x + np.random.normal(0, sigma_rev)
    w_rev = 1.0 / sigma_rev ** 2
    beta_ols_rev = ols_fit(X, y_rev)
    beta_wls_rev = wls_fit(X, y_rev, w_rev)
    ```

    그렇다. 이분산이 존재하는 한 방향과 무관하게 WLS가 OLS보다 낫다. WLS는 올바른 가중치를 쓰므로 이분산 오차로 일반화된 Gauss-Markov 정리 아래에서 최소분산 선형불편추정량(BLUE)이 된다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 모든 $i$에 대해 가중치가 같으면($w_i = c$) WLS가 OLS로 환원됨을 보여라. 이는 두 방법의 관계에 대해 무엇을 말해 주는가?

</div>

??? success "풀이"

    모든 $i$에 대해 $w_i = c$이면 $\mathbf{W} = c\mathbf{I}$이므로

    $$
    \hat{\boldsymbol{\beta}}_{\text{WLS}} = (\mathbf{X}^\top c\mathbf{I}\,\mathbf{X})^{-1}\mathbf{X}^\top c\mathbf{I}\,\mathbf{y} = (c\mathbf{X}^\top\mathbf{X})^{-1}c\mathbf{X}^\top\mathbf{y} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y} = \hat{\boldsymbol{\beta}}_{\text{OLS}}.
    $$

    OLS는 모든 관측값에 같은 가중치를 주는 WLS의 특수한 경우이며, 등분산 가정이 성립할 때 적절하다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> 실제로는 $\sigma_i^2$을 모른다. 먼저 OLS를 적합하고, $\ln(e_i^2)$을 $x_i$에 회귀시켜 분산함수를 추정한 뒤, 추정된 가중치로 WLS를 적용하는 실행가능 WLS를 구현하라.

</div>

??? success "풀이"

    ```python
    resid_ols = y - X @ beta_ols
    log_resid_sq = np.log(resid_ols ** 2 + 1e-10)
    gamma = np.linalg.lstsq(X, log_resid_sq, rcond=None)[0]
    sigma_hat = np.sqrt(np.exp(X @ gamma))
    w_feas = 1.0 / sigma_hat ** 2
    beta_fwls = wls_fit(X, y, w_feas)
    ```

    실행가능 WLS는 알려진 가중치 대신 추정된 가중치를 쓴다. 추정량은 일치성과 점근적 효율성을 갖지만, 가중치를 아는 WLS에 비해 작은 표본에서는 효율이 떨어질 수 있다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> 공분산 구조 $\boldsymbol{\Sigma} = \mathrm{diag}(\sigma_1^2, \ldots, \sigma_n^2)$을 알 때 $\hat{\boldsymbol{\beta}}_{\text{WLS}}$가 BLUE임을 증명하라.

</div>

??? success "풀이"

    일반화 Gauss-Markov 정리에 따르면 $\mathrm{Var}(\boldsymbol{\varepsilon}) = \boldsymbol{\Sigma}$일 때 $\boldsymbol{\beta}$의 BLUE는

    $$
    \hat{\boldsymbol{\beta}}_{\text{GLS}} = (\mathbf{X}^\top\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}\mathbf{X}^\top\boldsymbol{\Sigma}^{-1}\mathbf{y}.
    $$

    $\boldsymbol{\Sigma}$가 대각행렬이면 $\boldsymbol{\Sigma}^{-1} = \mathrm{diag}(1/\sigma_1^2, \ldots, 1/\sigma_n^2) = \mathbf{W}$이다. 따라서 $\hat{\boldsymbol{\beta}}_{\text{GLS}} = \hat{\boldsymbol{\beta}}_{\text{WLS}}$이다. "최선"이란 모든 선형불편추정량 가운데 분산이 가장 작다는 뜻으로, 임의의 선형불편 $\tilde{\boldsymbol{\beta}}$와 임의의 방향 $\mathbf{a}$에 대해 $\mathrm{Var}(\mathbf{a}^\top\hat{\boldsymbol{\beta}}_{\text{WLS}}) \leq \mathrm{Var}(\mathbf{a}^\top\tilde{\boldsymbol{\beta}})$이다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> WLS를 OLS + 이분산 일치(HC) 표준오차(White의 로버스트 표준오차)와 비교하라. 절충 관계는 무엇인가?

</div>

??? success "풀이"

    HC 표준오차는 계수 추정값은 그대로 두고 OLS의 표준오차만 교정한다.

    $$
    \widehat{\mathrm{Var}}_{\text{HC}}(\hat{\boldsymbol{\beta}}) = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\hat{\boldsymbol{\Omega}}\mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1},
    $$

    여기서 $\hat{\boldsymbol{\Omega}} = \mathrm{diag}(e_1^2, \ldots, e_n^2)$이다(HC0 형태). 절충 관계는 이렇다. (1) HC 표준오차는 분산함수를 지정하지 않고도 임의의 이분산 아래에서 타당하지만 OLS 계수 자체는 비효율적이다. (2) WLS는 효율적인 추정을 주지만 올바른 가중함수를 지정해야 한다. (3) 분산 모형이 잘못 설정되면 WLS는 표준오차에 편향을 들여올 수 있는 반면, HC 표준오차는 여전히 로버스트하다. 실무에서 분산함수를 모를 때는 HC 표준오차가 선호된다. $\square$

---

## 정리하며

가중최소제곱은 **분산 구조를 알 때** 이분산을 정면으로 다룬다.

$$
\hat{\boldsymbol\beta}_{\text{WLS}}=(\mathbf X^\top\mathbf W\mathbf X)^{-1}\mathbf X^\top\mathbf W\mathbf y,
\qquad w_i\propto\frac{1}{\sigma_i^2}
$$

- **정밀한 관측에 더 무게를 준다.** 7장의 역분산 가중과 같은 원리이며, 분산이 큰 관측의 영향을 줄인다.
- **가중치를 알아야 한다는 것이 실무의 걸림돌이다.** $\sigma_i^2$ 를 모르면 잔차에서 추정해야 하고(반복가중최소제곱), 그 추정이 틀리면 이득이 사라진다.
- **OLS 는 불편하지만 비효율적이다.** WLS 는 효율을 되찾으며, 가우스–마르코프의 최적성이 가중된 문제에서 회복된다.
- **로버스트 표준오차와 목적이 다르다.** HC3 은 계수를 그대로 두고 표준오차만 고치고, WLS 는 **추정값 자체를 바꾼다.** 분산 구조를 안다면 WLS 가 더 효율적이고, 모른다면 로버스트 표준오차가 안전하다.
- **잔차 그림으로 확인한다.** 가중 후 잔차의 퍼짐이 고르게 되었는지 보는 것이 성공 여부의 판단이다.

다음 절부터 **진단**으로 넘어간다.
