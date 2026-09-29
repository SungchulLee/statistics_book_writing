# 성능 척도

## 결정계수

<div class="defn" markdown>

### 정의 1. 결정계수 { .dfn }

$R^2$는 독립변수로부터 예측할 수 있는 종속변수 분산의 비율이다.

$$
R^2 = 1 - \frac{SS_{\text{Residual}}}{SS_{\text{Total}}}
$$

여기서

$$
\begin{array}{lll}
SS_{\text{Total}} &=& \displaystyle \sum_{i}\left(y_{i}-\bar{y}\right)^{2} \\[8pt]
SS_{\text{Residual}} &=& \displaystyle \sum_{i}\left(y_{i}-\hat{y}_{i}\right)^{2}
\end{array}
$$

절편이 있는 모형을 훈련자료에 적합했다면 $R^2$는 0과 1 사이의 값을 가지며 클수록 적합이 좋다. 그러나 $R^2$는 설명변수를 추가하면 예측력이 나아지지 않아도 항상 커지므로 과대적합으로 이어질 수 있다.

</div>

### 총제곱합의 분해

$y$의 전체 변동은 설명된 부분과 설명되지 않은 부분으로 깔끔하게 분해된다.

$$
\begin{array}{lll}
SS_{\text{Total}} &=& \displaystyle \sum_{i}\left(y_{i}-\bar{y}\right)^{2} \\[10pt]
&=& \displaystyle \sum_{i}\left(\left(y_{i}-\hat{y}_{i}\right) + \left(\hat{y}_{i}-\bar{y}\right)\right)^{2} \\[10pt]
&=& \displaystyle \sum_{i}\left(y_{i}-\hat{y}_{i}\right)^{2} + \sum_{i}\left(\hat{y}_{i}-\bar{y}\right)^{2} \\[10pt]
&=& \displaystyle SS_{\text{Residual}} + SS_{\text{Treatment}}
\end{array}
$$

여기서 $SS_{\text{Treatment}}$는 회귀모형이 설명하는 변동을 나타내며, 다른 절에서 쓰는 $\text{SSR}$(회귀제곱합)과 같은 양이다. 교차항은 OLS 추정의 성질에 의해 사라진다.

### 단순선형회귀에서의 해석

단순선형회귀에서 $SS_{\text{Treatment}}$는 상관계수로 표현할 수 있다.

$$
\begin{array}{lll}
SS_{\text{Treatment}} &=& \displaystyle \sum_{i}\left(\hat{y}_{i} - \bar{y}\right)^{2} \\[10pt]
&=& \displaystyle \beta^2 \sum_{i}\left(x_i - \bar{x}\right)^{2} \\[10pt]
&\approx& \displaystyle n\sigma_x^2\beta^2 \\[10pt]
&\approx& \displaystyle n\sigma_x^2\left(\rho\frac{\sigma_y}{\sigma_x}\right)^2 \\[10pt]
&=& \displaystyle n\sigma_y^2\rho^2
\end{array}
$$

따라서

$$
R^2 = \frac{SS_{\text{Treatment}}}{SS_{\text{Total}}} \approx \frac{n\sigma_y^2 \rho^2}{n\sigma_y^2} = \rho^2
$$

!!! note "사실은 정확한 등식이다"
    위 유도에서 $\approx$가 등장하는 것은 표본분산을 $n$으로 나누느냐 $n-1$로 나누느냐를 얼버무렸기 때문이다. 두 곳에서 같은 규약을 쓰면 인자가 상쇄되어 단순선형회귀에서는 $R^2 = \rho^2$이 **정확히** 성립한다. 첫 줄의 $\hat{y}_i - \bar{y} = \hat{\beta}(x_i - \bar{x})$도 근사가 아니라 정확한 등식이다(회귀직선이 평균점 $(\bar{x}, \bar{y})$를 지나기 때문이다).

단순선형회귀에서 $R^2$는 $x$와 $y$의 상관계수의 제곱이다.

!!! tip "참고"

    - [R-squared or coefficient of determination (Khan Academy)](https://www.khanacademy.org/math/ap-statistics/bivariate-data-ap/assessing-fit-least-squares-regression/v/r-squared-or-coefficient-of-determination)
    - [R-squared intuition (Khan Academy)](https://www.khanacademy.org/math/ap-statistics/bivariate-data-ap/assessing-fit-least-squares-regression/a/r-squared-intuition)

## 수정 결정계수

<div class="defn" markdown>

### 정의 2. 수정 결정계수 { .dfn }

수정 $R^2$는 설명변수의 개수를 반영하여 불필요한 복잡도에 벌점을 준다.

$$
\text{Adjusted } R^2 = 1 - \left(1 - R^2\right) \frac{n - 1}{n - p - 1}
$$

여기서 $n$은 표본크기이고 $p$는 (절편을 제외한) 설명변수의 개수이다.

</div>

### 조정 인자의 유도

조정은 원래의 제곱합을 자유도로 나눈 불편추정값으로 바꾸는 것이다.

$$
\text{Adjusted } R^2 = 1 - \frac{SS_{\text{Residual}} / (n - p - 1)}{SS_{\text{Total}} / (n - 1)}
$$

이렇게 하면 $SS_{\text{Residual}}$의 감소가 잃어버린 자유도를 정당화할 때에만 설명변수 추가가 수정 $R^2$를 높인다.

### 결정계수와의 주요 차이

- **모형 복잡도**: 수정 $R^2$는 설명변수의 개수를 반영하지만 $R^2$는 그렇지 않다.
- **모형 비교**: 설명변수 개수가 다른 모형들을 비교할 때는 수정 $R^2$가 낫다.
- **방향**: 모형을 개선하지 못하는 설명변수를 넣으면 수정 $R^2$는 줄어들 수 있지만 $R^2$는 커지기만 한다.

## 그 밖의 성능 척도

$$
\begin{array}{lll}
\text{MAE} && \displaystyle\frac{1}{n}\sum_{i=1}^n|y_i-\hat{y}_i| \\[10pt]
\text{MSE} && \displaystyle\frac{1}{n}\sum_{i=1}^n(y_i-\hat{y}_i)^2 \\[10pt]
\text{RMSE} && \displaystyle\sqrt{\frac{1}{n}\sum_{i=1}^n(y_i-\hat{y}_i)^2}
\end{array}
$$

- **MAE(평균절대오차)**: 예측값과 실제값의 절대차이의 평균이다. MSE보다 이상점에 덜 민감하다. 반응변수와 같은 단위로 오차를 제공한다.
- **MSE(평균제곱오차)**: 제곱차이의 평균이다. 큰 오차에 더 무거운 벌점을 준다. OLS의 손실함수로 쓰인다.
- **RMSE(제곱근평균제곱오차)**: MSE의 제곱근이다. 오차를 반응변수의 원래 단위로 되돌려 MSE보다 해석하기 쉽다.

<div class="codebox" markdown>

### 예제 1. 네 성능 측도 한자리에 { .eg }

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# 앞 절과 같은 광고 자료를 쓴다.
url = 'https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/Advertising.csv'
df = pd.read_csv(url, usecols=[1, 2, 3, 4])

df['TV:Radio'] = df['TV'] * df['Radio']

X = df[['TV', 'Radio', 'TV:Radio']]
y = df['Sales']

test_size_ratio = 0.3
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}\n")

# 아래 네 측도를 훈련과 시험에서 각각 잰다. 시험 쪽 값이 훈련 쪽보다
# 크게 나쁘면 과적합을 의심한다.
# R^2 — 반응의 분산 중 모형이 설명하는 몫. 단위가 없어 견주기 좋다.
print(f"Training R^2: {model.score(x_train, y_train)}")
print(f"Testing R^2: {model.score(x_test, y_test)}\n")

# MAE — 오차의 절댓값 평균. 단위가 반응과 같고 이상치에 덜 휘둘린다.
print(f"Training MAE: {metrics.mean_absolute_error(y_train, y_train_pred)}")
print(f"Testing MAE: {metrics.mean_absolute_error(y_test, y_test_pred)}\n")

# MSE — 오차의 제곱 평균. 큰 오차에 더 무거운 벌을 준다. 단위가 제곱이라
# 그대로 읽기는 어렵다.
print(f"Training MSE: {metrics.mean_squared_error(y_train, y_train_pred)}")
print(f"Testing MSE: {metrics.mean_squared_error(y_test, y_test_pred)}\n")

# RMSE — MSE 의 제곱근. 단위가 반응과 같아져 해석이 쉬워진다.
# 큰 오차를 무겁게 보되 읽기도 편해, 회귀에서 가장 널리 쓰인다.
print(f"Training RMSE: {np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))}")
print(f"Testing RMSE: {np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))}\n")
```

출력:

```
Intercept: 6.37486462995429
Coefficients: [0.02060952 0.04735462 0.00100684]

Training R^2: 0.9659030787012204
Testing R^2: 0.9673268969053402

Training MAE: 0.6344840392254547
Testing MAE: 0.730384235550869

Training MSE: 0.8947370334590617
Testing MSE: 0.8921262830343071

Training RMSE: 0.9459054040754085
Testing RMSE: 0.9445243686820934
```

훈련 $R^2$ 0.9659와 시험 $R^2$ 0.9673이 거의 같다. 두 값이 크게 벌어지면 과적합을 의심한다.

</div>

## 척도가 엇갈릴 때

위 예제에서는 네 척도가 모두 같은 방향을 가리켰다. 늘 그렇지는 않다.

![오차 분포가 다른 두 모형과, 척도에 따라 뒤집히는 순위](./img/metric_disagreement.png)

같은 자료를 맞힌 두 모형이 있다고 하자. 왼쪽 그림이 두 모형의 예측오차 분포다. 모형 A(파란색)는 정규분포 모양이어서 오차가 대개 작지만 가끔 아주 크다. 붉게 칠한 $|e| > 8$ 구간에 사는 점은 모형 A뿐이다. 모형 B(주황색)는 오차의 크기가 늘 $1$에서 $6$ 사이로 고르다. 크게 빗나가는 일도 없지만 정확히 맞히는 일도 없다.

오른쪽이 두 자로 잰 결과다. MAE로 재면 모형 A가 $3.19$로 모형 B의 $3.51$보다 낫다. 그런데 RMSE로 재면 모형 A가 $4.00$, 모형 B가 $3.82$로 순위가 **뒤집힌다.** 제곱이 큰 오차를 증폭하기 때문이다. 모형 A의 $|e| > 8$짜리 오차들은 개수로는 전체의 $4.6\%$에 지나지 않지만 MSE에서는 훨씬 큰 몫을 차지한다. 두 모형의 $\text{RMSE}/\text{MAE}$ 비를 내 보면 A가 $1.25$, B가 $1.09$로 이 차이가 그대로 드러난다.

어느 쪽이 "더 나은 모형"인가? 자료만으로는 답할 수 없다. 큰 오차가 특별히 비싼 문제라면 — 재고를 크게 잘못 잡으면 결품이 나거나 창고가 넘치는 경우, 구조물의 하중을 과소평가하면 붕괴하는 경우 — 모형 B를 골라야 한다. 오차의 비용이 크기에 비례할 뿐이라면 모형 A가 낫다. 곧 **척도를 고르는 일은 자료 분석의 문제가 아니라 문제 정의의 문제다.** 모형을 적합하기 전에 "빗나감의 비용이 오차 크기에 어떻게 달라지는가"를 먼저 정해야 하는 이유가 여기에 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
어떤 회귀모형이 검정자료에서 MSE = 16.0, MAE = 3.2이다. 두 번째 모형은 MSE = 14.5, MAE = 3.5이다. 어느 모형이 더 나은가? MSE와 MAE가 엇갈리는 것은 자료에 대해 무엇을 시사하는가?

</div>

??? success "풀이"
    선택은 응용 상황에 달려 있다. 모형 2는 MSE가 더 낮고(14.5 대 16.0), 모형 1은 MAE가 더 낮다(3.2 대 3.5).

    두 척도가 엇갈리는 것은 **오차 분포의 모양**이 다르다는 뜻이다. RMSE/MAE 비를 보면 뚜렷해진다.

    - 모형 1: $\text{RMSE} = \sqrt{16.0} = 4.0$, 비 $= 4.0/3.2 = 1.25$
    - 모형 2: $\text{RMSE} = \sqrt{14.5} = 3.81$, 비 $= 3.81/3.5 = 1.09$

    비가 클수록 오차 분포의 꼬리가 무겁다. 곧 **모형 1**이 전형적인 오차는 작지만 몇 개의 큰 오차를 갖고 있고, **모형 2**는 오차가 더 고르게 퍼져 있되 평균적인 오차 크기는 조금 더 크다. MSE는 오차를 제곱하므로 큰 오차에 더 큰 벌점을 주고, 그래서 큰 오차가 있는 모형 1의 MSE가 더 높게 나온 것이다.

    큰 오차의 비용이 크다면 모형 2(낮은 MSE)를, 전형적인 오차 크기가 더 중요하다면 모형 1(낮은 MAE)을 택한다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
어떤 모형이 훈련자료에서 $R^2 = 0.95$, 검정자료에서 $R^2 = 0.60$이다. 문제를 진단하고 대책을 제안하라.

</div>

??? success "풀이"
    훈련 $R^2$(0.95)와 검정 $R^2$(0.60)의 큰 격차는 **과대적합**을 나타낸다. 모형이 훈련자료에만 있는 패턴(잡음 포함)을 학습하여 새 자료에 일반화되지 않는 것이다.

    대책: (1) 설명변수를 빼거나, 정칙화(릿지/라쏘)를 쓰거나, 다항 차수를 낮추어 **모형 복잡도를 줄인다**. (2) 가능하면 **훈련자료를 늘린다**. (3) 모형 선택 과정에서 **교차검증**을 써서 표본 밖 성능을 더 신뢰성 있게 추정한다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
(절편이 있을 때) 훈련자료에서는 $R^2$가 항상 0과 1 사이인데도 검정자료에서는 음수가 될 수 있는 이유를 설명하라.

</div>

??? success "풀이"
    절편이 있는 훈련자료에서는 OLS가 $\text{SSE} \leq \text{SST}$를 보장하므로 $R^2 = 1 - \text{SSE}/\text{SST} \geq 0$이다.

    검정자료에서는 $R^2 = 1 - \sum(y_i - \hat{y}_i)^2 / \sum(y_i - \bar{y}_{\text{test}})^2$이다. 예측값 $\hat{y}_i$는 훈련 모형이 만들어 낸 것이므로 검정자료에서는 체계적으로 치우쳐 있을 수 있다. 모형의 예측이 모든 관측값에 대해 단순히 검정자료의 평균을 예측하는 것보다 나쁘다면 $\text{SSE} > \text{SST}$가 되어 $R^2 < 0$이다. 이는 모형이 단지 나쁜 정도가 아니라 아무 모형도 쓰지 않는 것보다 나쁘다는 뜻이다.

---

## 정리하며

성능 척도를 **한자리에** 모았다.

- **세 갈래다.** 설명력($R^2$, 조정 $R^2$), 절대 오차(MAE·MSE·RMSE), 상대 오차(MAPE·MASE).
- **하나로 충분한 경우는 없다.** $R^2$ 는 비율만, RMSE 는 크기만, MAPE 는 상대 크기만 말한다. **셋을 함께 보아야 모형을 제대로 안다.**
- **훈련 성능과 검정 성능을 구별한다.** 훈련자료의 값은 언제나 낙관적이며, 모형이 유연할수록 격차가 크다. 다음 절의 모형선택이 이 문제를 정면으로 다룬다.
- **척도의 선택이 모형의 선택을 바꾼다.** MAE 로 고른 모형과 RMSE 로 고른 모형이 다를 수 있으며, **무엇을 최소화할지가 문제 정의의 일부**다.
- **예측이 목적이면 성능 척도, 추론이 목적이면 계수와 신뢰구간**이다. 1장의 구분이 여기서 보고 방식의 차이로 나타난다.

다음 절부터 **모형선택**으로 넘어간다.
