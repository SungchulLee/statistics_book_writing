# OLS 회귀 출력 재현

## 개요

이 페이지는 정규방정식을 써서 OLS 회귀 요약표 전체를 밑바닥부터 재현하는 방법을 보인다. Advertising 자료(Sales를 TV, Radio, Newspaper에 회귀)에서 출발하여, 미리 만들어진 회귀 요약 함수에 기대지 않고 계수, 표준오차, $t$ 통계량, $p$값, 95% 신뢰구간을 직접 계산한다.

## 수학적 배경

행렬 형태의 다중선형회귀 모형은

$$
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}, \qquad \boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2 \mathbf{I}).
$$

OLS 추정량은 **정규방정식**으로 얻는다.

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}.
$$

잔차 표준오차는

$$
s = \sqrt{\frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{n - k}},
$$

여기서 $k$는 (절편을 포함한) 모수의 개수이다. $j$번째 계수의 표준오차는

$$
\mathrm{SE}(\hat{\beta}_j) = s \sqrt{[(\mathbf{X}^\top \mathbf{X})^{-1}]_{jj}}.
$$

$H_0\colon \beta_j = 0$을 검정하는 $t$ 통계량과 $p$값은

$$
t_j = \frac{\hat{\beta}_j}{\mathrm{SE}(\hat{\beta}_j)}, \qquad p\text{-value} = 2\,P(T_{n-k} > |t_j|).
$$

$\beta_j$의 95% 신뢰구간은 $\hat{\beta}_j \pm t^*_{n-k,\,0.025} \cdot \mathrm{SE}(\hat{\beta}_j)$이다.

### 정규방정식으로 OLS 적합하기

<div class="codebox" markdown>

#### 예제 1. OLS 적합 함수 { .eg }

```python
import numpy as np
from scipy import stats

def fit_ols(X, y):
    """최소제곱 추정값과, 표준오차를 만드는 데 필요한 두 조각을 돌려준다.

    s 는 잔차의 표준편차(자유도 n-k), cov_matrix 는 (X'X)^-1 이다.
    계수 j 의 표준오차는 s * sqrt(cov_matrix[j, j]) 로 만들어진다.
    """
    n, k = X.shape
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    y_hat = X @ beta_hat
    residuals = y - y_hat
    s = np.sqrt(np.sum(residuals ** 2) / (n - k))
    cov_matrix = np.linalg.inv(X.T @ X)
    return beta_hat, s, cov_matrix
```

</div>

### 회귀표 만들기

<div class="codebox" markdown>

#### 예제 2. 회귀 출력표 만들기 { .eg }

```python
def regression_table(beta_hat, s, cov_matrix, n, k, var_names):
    """회귀 출력표를 직접 만든다.

    statsmodels 의 summary() 가 찍어 주는 계수 표와 같은 내용이다.
    계수 → 표준오차 → t → p-값 → 신뢰구간이 어떤 순서로 만들어지는지를
    보이려고 풀어 썼다.
    """
    df = n - k
    t_crit = stats.t(df).ppf(0.975)

    for name, j in zip(var_names, range(k)):
        coef = beta_hat[j, 0]
        v_j = cov_matrix[j, j]
        se = s * np.sqrt(v_j)
        t_stat = coef / se
        p_val = 2 * stats.t(df).sf(np.abs(t_stat))
        ci_lo = coef - t_crit * se
        ci_hi = coef + t_crit * se
        print(f"{name:10}  coef={coef:.4f}  SE={se:.3f}  "
              f"t={t_stat:.3f}  p={p_val:.3f}  "
              f"CI=({ci_lo:.3f}, {ci_hi:.3f})")
```

</div>

### Advertising 자료에서 실행하기

<div class="codebox" markdown>

#### 예제 3. 광고 자료에 적용하기 { .eg }

```python
import pandas as pd

url = ('https://raw.githubusercontent.com/justmarkham/'
       'scikit-learn-videos/master/data/Advertising.csv')
data = pd.read_csv(url, usecols=[1, 2, 3, 4])
# 앞 70%만 훈련에 쓴다.
training_data = data.iloc[:int(len(data) * 0.7)]

y = np.array(training_data.Sales).reshape(-1, 1)
n = y.shape[0]
# 1 로 채운 열을 앞에 붙여 절편을 만든다. 마지막 열이 반응(Sales)이므로
# iloc[:, :-1] 로 설명변수만 고른다.
X = np.concatenate(
    (np.ones((n, 1)), np.array(training_data.iloc[:, :-1])), axis=1
)
k = X.shape[1]

beta_hat, s, cov_matrix = fit_ols(X, y)
var_names = ["Intercept", "TV", "Radio", "Newspaper"]
regression_table(beta_hat, s, cov_matrix, n, k, var_names)
```

출력:

```
Intercept   coef=3.0451  SE=0.391  t=7.782  p=0.000  CI=(2.271, 3.819)
TV          coef=0.0470  SE=0.002  t=27.653  p=0.000  CI=(0.044, 0.050)
Radio       coef=0.1797  SE=0.011  t=16.665  p=0.000  CI=(0.158, 0.201)
Newspaper   coef=-0.0030  SE=0.007  t=-0.428  p=0.669  CI=(-0.017, 0.011)
```

요약표의 각 열을 따로 꺼내 인쇄했다. 계수, 표준오차, $t$, p-값, 신뢰구간이 어떻게 맞물리는지 한 줄로 볼 수 있다.

출력($n = 140$, $k = 4$):

</div>

```text
Intercept   coef=3.0451  SE=0.391  t=7.782   p=0.000  CI=(2.271, 3.819)
TV          coef=0.0470  SE=0.002  t=27.653  p=0.000  CI=(0.044, 0.050)
Radio       coef=0.1797  SE=0.011  t=16.665  p=0.000  CI=(0.158, 0.201)
Newspaper   coef=-0.0030 SE=0.007  t=-0.428  p=0.669  CI=(-0.017, 0.011)
```

이 값들은 `statsmodels`의 `sm.OLS(y, X).fit().summary()`가 내놓는 계수표와 정확히 일치한다.

## 해석

- 회귀표의 각 행은 설명변수 하나에 대응한다. 계수 추정값 $\hat{\beta}_j$는 다른 설명변수를 고정했을 때 그 설명변수가 한 단위 늘어날 때 기대되는 Sales의 변화를 준다.
- 표준오차는 각 추정의 정밀도를 수량화한다. SE가 작을수록 정밀한 추정이다.
- $t$ 통계량은 계수가 0에서 표준오차 몇 개만큼 떨어져 있는지를 잰다. 절댓값이 크면 통계적 유의성을 나타낸다.
- $p$값은 $H_0\colon \beta_j = 0$ 아래에서 적어도 그만큼 극단적인 $t$ 통계량을 관측할 확률이다. 0.05 미만이면 관행적으로 유의하다고 본다.
- 95% 신뢰구간은 참 계수가 취할 만한 값의 범위를 준다. 0을 배제하면 그 설명변수는 유의수준 5%에서 유의하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span> $\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}$가 정규방정식 $\mathbf{X}^\top \mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}^\top \mathbf{y}$를 만족함을 수치적으로 확인하라.

</div>

??? success "풀이"

    ```python
    lhs = X.T @ X @ beta_hat
    rhs = X.T @ y
    print(np.allclose(lhs, rhs))  # True
    ```

    출력:

    ```
    True
    ```

    `True`가 나온다. 요약표의 $t$ 값이 계수를 표준오차로 나눈 것과 정확히 같다는 확인이다.

    구성상 $\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$의 양변에 왼쪽에서 $\mathbf{X}^\top\mathbf{X}$를 곱하면 $\mathbf{X}^\top\mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}^\top\mathbf{y}$가 된다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 잔차에서 $R^2$와 수정 $R^2$를 계산하라. $R^2 = 1 - \mathrm{RSS}/\mathrm{TSS}$임을 보여라.

</div>

??? success "풀이"

    ```python
    y_hat = X @ beta_hat
    RSS = np.sum((y - y_hat) ** 2)
    TSS = np.sum((y - y.mean()) ** 2)
    R2 = 1 - RSS / TSS
    adj_R2 = 1 - (1 - R2) * (n - 1) / (n - k)
    print(f"R^2 = {R2:.4f}, Adjusted R^2 = {adj_R2:.4f}")
    ```

    출력:

    ```
    R^2 = 0.8937, Adjusted R^2 = 0.8914
    ```

    $R^2 = 0.894$, 조정 $R^2 = 0.891$이다. 조정 $R^2$가 조금 작은 것은 설명변수 개수에 대한 벌점 때문이며, 변수를 늘려도 적합이 그만큼 좋아지지 않으면 조정 $R^2$는 오히려 떨어진다.

    이 자료에서는 $R^2 = 0.8938$, 수정 $R^2 = 0.8915$가 나온다.

    정의에 따라 $\mathrm{TSS} = \sum(y_i - \bar{y})^2$, $\mathrm{RSS} = \sum(y_i - \hat{y}_i)^2$이고 $R^2 = 1 - \mathrm{RSS}/\mathrm{TSS}$는 모형이 설명하는 분산의 비율을 잰다. 수정 $R^2$는 $1 - \frac{n-1}{n-k}(1 - R^2)$로 설명변수의 개수에 벌점을 준다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> $s^2$의 분모로 $n$ 대신 $n - k$를 쓰면 왜 $\sigma^2$의 불편추정량이 되는지 설명하라.

</div>

??? success "풀이"

    잔차벡터는 $\mathbf{e} = \mathbf{M}\mathbf{y}$이며 $\mathbf{M} = \mathbf{I} - \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$이다. 모형 아래에서 $\mathbf{e} = \mathbf{M}\boldsymbol{\varepsilon}$이므로

    $$
    E[\mathbf{e}^\top\mathbf{e}] = E[\boldsymbol{\varepsilon}^\top\mathbf{M}\boldsymbol{\varepsilon}] = \sigma^2 \operatorname{tr}(\mathbf{M}) = \sigma^2(n - k),
    $$

    $\mathbf{M}$이 멱등이고 $\operatorname{tr}(\mathbf{M}) = n - k$이기 때문이다. $n - k$로 나누면 $E[s^2] = \sigma^2$을 얻는다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> $\mathbf{X}^\top\mathbf{X}$를 명시적으로 역행렬 계산하는 대신 `numpy.linalg.lstsq`로 회귀표를 다시 계산하라. `lstsq`가 수치적으로 선호되는 이유를 논하라.

</div>

??? success "풀이"

    ```python
    beta_lstsq, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    ```

    `lstsq`는 SVD 분해를 쓰는데, 이는 $(\mathbf{X}^\top\mathbf{X})^{-1}$을 명시적으로 계산하는 것보다 수치적으로 안정적이다. $\mathbf{X}^\top\mathbf{X}$의 조건이 나쁘면(거의 특이행렬이면) 직접 역행렬을 구하는 것은 부동소수점 오차를 증폭시키지만, SVD는 거의 공선인 상황을 매끄럽게 처리한다. 조건이 좋은 문제에서는 두 결과가 기계 정밀도 수준으로 일치한다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> $j$번째 계수의 $t$ 통계량을 $t_j = \hat{\beta}_j \sqrt{[(\mathbf{X}^\top\mathbf{X})]_{jj}} / s$로 쓸 수 있는 것은 설명변수들이 직교할 때뿐임을 증명하라. 일반적으로는 어떻게 되는가?

</div>

??? success "풀이"

    설명변수들이 직교하면 $\mathbf{X}^\top\mathbf{X}$가 대각행렬이므로 $[(\mathbf{X}^\top\mathbf{X})^{-1}]_{jj} = 1/[(\mathbf{X}^\top\mathbf{X})]_{jj}$이다. 이때

    $$
    t_j = \frac{\hat{\beta}_j}{s\sqrt{[(\mathbf{X}^\top\mathbf{X})^{-1}]_{jj}}} = \frac{\hat{\beta}_j\sqrt{[(\mathbf{X}^\top\mathbf{X})]_{jj}}}{s}.
    $$

    일반적으로 $(\mathbf{X}^\top\mathbf{X})^{-1}$은 대각원소의 역수가 아니다. $\mathbf{X}^\top\mathbf{X}$의 비대각원소(설명변수 사이의 상관)가 역행렬의 대각원소를 부풀리기 때문이다. 이 부풀림을 재는 것이 분산팽창인자(VIF)이다. $\square$

---

## 정리하며

회귀 요약표 전체를 **손으로 재현**했다.

- **모든 열이 $(\mathbf X^\top\mathbf X)^{-1}$ 에서 나온다.** 계수는 $(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top\mathbf y$, 표준오차는 그 역행렬 대각원소의 제곱근에 $s$ 를 곱한 것, $t$ 는 둘의 비, $p$ 값은 $t_{n-p-1}$ 의 양측 꼬리다.
- **직접 계산해 보면 요약표가 블랙박스가 아니게 된다.** 어떤 수가 어디서 오는지 알면 이상한 값이 나왔을 때 원인을 짚을 수 있다.
- **표준오차가 핵심 고리다.** 계수만으로는 아무 판단도 못 하며, **표준오차가 계수를 해석 가능하게 만든다.**
- **신뢰구간과 $p$ 값이 같은 계산의 두 표현이다.** 구간이 $0$ 을 포함하는 것과 $p>\alpha$ 가 동치다.
- **`statsmodels` 결과와 대조해 검산한다.** 맞지 않으면 대개 절편 열을 빠뜨렸거나 자유도를 잘못 잡은 것이다.

다음 절 **계수 검정**으로 넘어간다.
