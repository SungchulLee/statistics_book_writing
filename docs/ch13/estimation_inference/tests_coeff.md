# 일반 OLS 추정량의 표집분포

## 개요

이 절은 단순선형회귀의 추론 결과를 행렬 표기를 써서 **다중선형회귀** 상황으로 확장한다. OLS 계수벡터 $\hat{\beta}$, 잔차분산 추정량 $s^2$, 개별 계수를 검정하는 $t$ 통계량의 표집분포를 유도한다.

---

## 준비: 다중선형회귀 모형

행렬 형태의 모형을 생각하자.

$$
\mathbf{y} = \mathbf{X}\beta + \varepsilon
$$

여기서

- $\mathbf{y}$는 $N \times 1$ 반응벡터,
- $\mathbf{X}$는 $N \times (p+1)$ 설계행렬(절편 열을 포함한다),
- $\beta$는 $(p+1) \times 1$ 미지 계수벡터,
- $\varepsilon \sim N(\mathbf{0}, \sigma^2 I_N)$는 오차벡터이다.

OLS 추정량은

$$
\hat{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

---

## 1. 베타 추정량의 표집분포

### 정리

선형회귀 모형의 가정 아래에서

$$
\hat{\beta} \sim N\!\left(\beta,\; \sigma^2 (\mathbf{X}^T \mathbf{X})^{-1}\right)
$$

곧 $\hat{\beta}$는 다음의 정규분포를 따른다.

- **평균**: $E(\hat{\beta}) = \beta$ (불편),
- **공분산**: $\text{Var}(\hat{\beta}) = \sigma^2 (\mathbf{X}^T \mathbf{X})^{-1}$.

### 증명

OLS 공식에 $\mathbf{y} = \mathbf{X}\beta + \varepsilon$을 대입하면

$$
\hat{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T (\mathbf{X}\beta + \varepsilon) = \beta + (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \varepsilon
$$

**불편성**: $E(\varepsilon) = \mathbf{0}$이므로

$$
E(\hat{\beta}) = \beta + (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T E(\varepsilon) = \beta
$$

**공분산**: $\mathbf{A} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T$라 두면 $\hat{\beta} - \beta = \mathbf{A}\varepsilon$이고

$$
\text{Var}(\hat{\beta}) = \mathbf{A}\,\text{Var}(\varepsilon)\,\mathbf{A}^T = \mathbf{A}(\sigma^2 I_N)\mathbf{A}^T = \sigma^2 \mathbf{A}\mathbf{A}^T
$$

$\mathbf{A}\mathbf{A}^T$를 계산하면

$$
\mathbf{A}\mathbf{A}^T = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1} = (\mathbf{X}^T\mathbf{X})^{-1}
$$

따라서 $\text{Var}(\hat{\beta}) = \sigma^2(\mathbf{X}^T\mathbf{X})^{-1}$이다.

**정규성**: $\hat{\beta} - \beta = \mathbf{A}\varepsilon$은 다변량정규벡터 $\varepsilon$의 선형변환이므로 $\hat{\beta}$ 자체도 다변량정규이다. $\square$

### 함의

이 결과는 다중회귀의 모든 추론의 토대가 된다. 개별 계수의 신뢰구간, 결합 신뢰영역, 가설검정, 예측구간이 모두 이 분포 결과에서 따라 나온다.

---

## 2. s 제곱의 표집분포

### 정리

잔차분산 추정량

$$
s^2 = \frac{1}{N - p - 1} \sum_{i=1}^N (y^{(i)} - \hat{y}^{(i)})^2
$$

의 표집분포는 다음과 같다.

$$
\frac{(N - p - 1)\,s^2}{\sigma^2} \sim \chi^2_{N-p-1}
$$

동등하게

$$
s^2 \sim \sigma^2 \frac{\chi^2_{N-p-1}}{N - p - 1}
$$

### 주요 성질

**불편성**: $E(s^2) = \sigma^2$이므로 $s^2$은 오차분산의 불편추정량이다.

**자유도**: 자유도 $N - p - 1$은 관측값 $N$개에서 추정한 모수 $p + 1$개를 뺀 것을 반영한다.

**$\hat{\beta}$와의 독립성**: 정규성 가정 아래에서 $s^2$과 $\hat{\beta}$는 통계적으로 독립이다. 이 독립성은 $t$ 검정이 타당하기 위한 필수 조건이다.

### 증명

**1단계: 잔차를 사영으로 표현.** 모자행렬 $\mathbf{P} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$와 잔차생성행렬 $\mathbf{M} = I_N - \mathbf{P}$를 정의하자. 잔차벡터는

$$
\mathbf{e} = \mathbf{y} - \mathbf{X}\hat{\beta} = (I_N - \mathbf{P})\mathbf{y} = \mathbf{M}\varepsilon
$$

마지막 등식에는 $\mathbf{M}\mathbf{X} = \mathbf{0}$을 썼다.

**2단계: 잔차제곱합을 이차형식으로.**

$$
\text{RSS} = \mathbf{e}^T\mathbf{e} = \varepsilon^T \mathbf{M}^T \mathbf{M}\,\varepsilon = \varepsilon^T \mathbf{M}\,\varepsilon
$$

$\mathbf{M}$이 대칭이고 멱등($\mathbf{M}^2 = \mathbf{M}$)이기 때문이다.

**3단계: 카이제곱분포.** $\varepsilon \sim N(\mathbf{0}, \sigma^2 I_N)$이고 $\mathbf{M}$이 계수 $\text{tr}(\mathbf{M}) = N - (p+1) = N - p - 1$인 대칭 멱등행렬이므로

$$
\frac{\varepsilon^T \mathbf{M}\,\varepsilon}{\sigma^2} = \frac{\text{RSS}}{\sigma^2} \sim \chi^2_{N-p-1}
$$

**4단계: 결론.** 자유도로 나누면

$$
s^2 = \frac{\text{RSS}}{N - p - 1} \sim \sigma^2\frac{\chi^2_{N-p-1}}{N - p - 1} \qquad \square
$$

---

## 3. 개별 회귀계수의 t 통계량

### 정리

귀무가설 $H_0: \beta_j = 0$ 아래에서 검정통계량

$$
t_j = \frac{\hat{\beta}_j}{s\sqrt{v_j}} \sim t_{N-p-1}
$$

이다. 여기서 $v_j = \left((\mathbf{X}^T\mathbf{X})^{-1}\right)_{jj}$는 $(\mathbf{X}^T\mathbf{X})^{-1}$의 $j$번째 대각원소이다.

### 해석

$t$ 통계량 $t_j$는 **다른 모든 설명변수를 고려한 뒤에도** $j$번째 설명변수가 모형에 기여하는지를 검정한다. 절댓값이 크면 $\beta_j \neq 0$이라는 증거, 곧 $j$번째 변수가 $y$에 통계적으로 유의한 부분효과를 갖는다는 증거가 된다.

분모 $s\sqrt{v_j}$가 $\hat{\beta}_j$의 **표준오차**이다.

$$
\text{SE}(\hat{\beta}_j) = s\sqrt{v_j}
$$

### 증명

**1단계: $H_0$ 아래 $\hat{\beta}_j$의 분포.** $\hat{\beta}$의 표집분포에서 각 성분은

$$
\hat{\beta}_j \sim N(\beta_j, \sigma^2 v_j)
$$

$H_0: \beta_j = 0$ 아래에서는

$$
\frac{\hat{\beta}_j}{\sigma\sqrt{v_j}} \sim N(0, 1)
$$

**2단계: 분모의 독립 카이제곱.** $s^2$의 분포에서

$$
\frac{(N-p-1)s^2}{\sigma^2} \sim \chi^2_{N-p-1}
$$

이고 이는 $\hat{\beta}_j$와 독립이다($\hat{\beta}$는 $\mathbf{P}\varepsilon$에, $s^2$은 $\mathbf{M}\varepsilon$에 의존하며 $\mathbf{PM} = \mathbf{0}$이기 때문이다).

**3단계: $t$ 비 만들기.** $t$ 분포의 정의에 따라

$$
t_j = \frac{\hat{\beta}_j / (\sigma\sqrt{v_j})}{\sqrt{s^2/\sigma^2}} = \frac{\hat{\beta}_j}{s\sqrt{v_j}} \sim t_{N-p-1} \qquad \square
$$

### 일반 신뢰구간

$\beta_j$에 대한 $(1 - \alpha)$ 신뢰구간은

$$
\hat{\beta}_j \pm t_{N-p-1}(1 - \alpha/2)\; s\sqrt{v_j}
$$

---

## 예제: 선형회귀 출력 재현

### 문제

Advertising 자료를 써서 모형 $\text{Sales} \sim \text{TV} + \text{Radio} + \text{Newspaper}$의 주요 회귀 출력 — 계수, 표준오차, $t$ 통계량, $p$값, 신뢰구간 — 을 밑바닥부터 계산해 재현하라.

!!! info "참고"

    - [Khan Academy: Using Least-Squares Regression Output](https://www.khanacademy.org/math/ap-statistics/bivariate-data-ap/least-squares-regression/v/using-least-squares-regression-output)
    - [Khan Academy: Interpreting Computer Regression Data](https://www.khanacademy.org/math/ap-statistics/bivariate-data-ap/assessing-fit-least-squares-regression/v/interpreting-computer-regression-data)

### 구현

```python
import numpy as np
import pandas as pd
from scipy import stats

# Load the Advertising dataset
dataset_url = (
    'https://raw.githubusercontent.com/justmarkham/'
    'scikit-learn-videos/master/data/Advertising.csv'
)
advertising_data = pd.read_csv(dataset_url, usecols=[1, 2, 3, 4])

# Train/test split (70/30)
total_observations = advertising_data.shape[0]
test_set_ratio = 0.3
train_count = int(total_observations * (1 - test_set_ratio))
training_data = advertising_data.iloc[:train_count]

# Response vector y and design matrix X (with intercept column)
y = np.array(training_data.Sales).reshape(-1, 1)
n = y.shape[0]
X = np.concatenate(
    (np.ones((n, 1)), np.array(training_data.iloc[:, :-1])),
    axis=1
)
p_plus_1 = X.shape[1]  # number of parameters (including intercept)

# OLS coefficients: β̂ = (X'X)⁻¹X'y
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y

# Predicted values and residual standard error
y_hat = X @ beta_hat
s = np.sqrt(np.sum((y - y_hat) ** 2) / (n - p_plus_1))

# Variance-covariance matrix: (X'X)⁻¹
cov_matrix = np.linalg.inv(X.T @ X)

# Print regression table
print("=" * 100)
print("\t\t    coef    std err \t     t      P>|t|"
      "     [0.025      0.975] ")
print("-" * 100)

variable_names = ["Intercept", "TV", "Radio", "Newspaper"]

for name, j in zip(variable_names, range(p_plus_1)):
    coef = beta_hat[j, 0]
    v_j = cov_matrix[j, j]
    se = s * np.sqrt(v_j)
    t_stat = coef / se
    p_val = 2 * stats.t(n - p_plus_1).sf(np.abs(t_stat))
    ci_lower = coef - stats.t(n - p_plus_1).ppf(0.975) * se
    ci_upper = coef + stats.t(n - p_plus_1).ppf(0.975) * se
    print(f"{name:10}    {coef:10.4f} {se:10.3f} "
          f"{t_stat:10.3f} {p_val:10.3f} "
          f"{ci_lower:10.3f} {ci_upper:10.3f}")

print("=" * 100)
```

출력(훈련자료 140개 관측값):

```text
                coef    std err        t      P>|t|     [0.025      0.975]
Intercept     3.0451      0.391      7.782      0.000      2.271      3.819
TV            0.0470      0.002     27.653      0.000      0.044      0.050
Radio         0.1797      0.011     16.665      0.000      0.158      0.201
Newspaper    -0.0030      0.007     -0.428      0.669     -0.017      0.011
```

이 값들은 `statsmodels`의 `sm.ols('Sales ~ TV + Radio + Newspaper', train_data).fit().summary()`가 내놓는 표와 정확히 일치한다.

### 출력 읽기

회귀표의 각 행은 다음을 담고 있다.

- **coef**: OLS 추정값 $\hat{\beta}_j$.
- **std err**: 표준오차 $s\sqrt{v_j}$. 여기서 $v_j = ((\mathbf{X}^T\mathbf{X})^{-1})_{jj}$이다.
- **t**: $t$ 통계량 $t_j = \hat{\beta}_j / (s\sqrt{v_j})$.
- **P>|t|**: $t_{N-p-1}$에서 얻은 양측 $p$값.
- **[0.025, 0.975]**: 95% 신뢰구간 $\hat{\beta}_j \pm t_{N-p-1}(0.975) \cdot s\sqrt{v_j}$.

$p$값이 0.05보다 작으면, 동등하게 95% 신뢰구간이 0을 포함하지 않으면 그 설명변수는 유의수준 5%에서 통계적으로 유의하다. 위 표에서 Newspaper는 $p = 0.669$이고 신뢰구간 $(-0.017, 0.011)$이 0을 포함하므로 유의하지 않다.

## 연습문제

**연습문제 1.**
설명변수가 $p = 4$개이고 $n = 50$인 다중회귀에서 개별 계수의 $t$ 검정과 전체 유의성 $F$ 검정의 자유도를 유도하라.

??? success "연습문제 1 풀이"

    - **개별 계수 $\beta_j$의 $t$ 검정:** $H_0: \beta_j = 0$ 아래에서 $t = \hat{\beta}_j / \text{SE}(\hat{\beta}_j) \sim t_{n-p-1} = t_{45}$이다.

    - **전체 유의성 $F$ 검정:** $H_0: \beta_1 = \beta_2 = \beta_3 = \beta_4 = 0$을 검정한다. $H_0$ 아래에서 $F = (\text{SSR}/p) / (\text{SSE}/(n-p-1)) \sim F_{p, n-p-1} = F_{4, 45}$이다.

---

**연습문제 2.**
어떤 회귀 출력에서 설명변수 $X_3$이 $\hat{\beta}_3 = 2.1$, $p = 0.04$였는데, 모형에 $X_4$를 넣으니 $\hat{\beta}_3$이 $0.3$으로 바뀌고 $p = 0.72$가 되었다. 이 현상을 설명하라.

??? success "연습문제 2 풀이"
    **다중공선성** 또는 **교란**의 결과이다. $X_4$를 넣으면

    1. $X_3$과 $X_4$가 상관되어 있다면 $X_4$가 이전에 $X_3$이 설명하던 변동을 "흡수"한다. ($X_4$를 고정한) $X_3$의 부분효과는 ($X_4$를 무시한) 주변효과보다 훨씬 작다.

    2. 다중공선성 때문에 $\hat{\beta}_3$의 표준오차가 커지고(VIF가 커지고) $t$ 통계량이 더욱 작아진다.

    이는 계수 추정값과 그 유의성이 모형에 어떤 다른 설명변수가 들어 있는지에 달려 있음을 보여준다. 유의하던 것이 유의하지 않게 바뀌었다는 것은 $X_3$의 겉보기 효과가 부분적으로(또는 상당 부분) $X_4$와의 상관에서 비롯되었음을 시사한다.

---

**연습문제 3.**
전체 모형의 $F$ 검정이 유의하더라도 어떤 개별 설명변수의 $t$ 검정이 유의하리라는 보장은 없다. 그 이유를 설명하고 개념적인 예를 구성하라.

??? success "연습문제 3 풀이"
    $F$ 검정은 결합가설 $H_0: \beta_1 = \cdots = \beta_p = 0$을 검정한다. 개별 $t$ 검정은 각 $\beta_j = 0$을 따로 검정한다. 설명변수들이 강하게 상관되어 있으면 둘은 어긋날 수 있다.

    **예:** 상관이 $r = 0.95$인 두 설명변수 $X_1$과 $X_2$가 모두 $Y$를 잘 예측한다고 하자. 둘이 함께 상당한 분산을 설명하므로 $F$ 검정은 유의하다. 그러나 개별적으로는 공유된 분산이 둘로 갈라지고 다중공선성으로 표준오차가 부풀려져($\text{VIF} = 1/(1-0.95^2) \approx 10.3$) 두 $t$ 검정 모두 유의하지 않을 수 있다. 어느 쪽도 상대를 넘어서는 기여를 별로 하지 못하지만, 함께 보면 분명히 중요하다.
