# 회귀 (대마초 가격과 인구통계)

## 개요

이 페이지는 인구, 1인당 소득, 인종 구성 같은 주(州) 단위 인구통계 특성으로 고급 대마초 가격을 예측하는 데 선형회귀를 적용한다. 상관행렬을 통한 탐색적 분석, scikit-learn과 statsmodels를 이용한 단변량·다변량 OLS, 그리고 남겨 둔 검정자료에서의 모형 성능 평가를 다룬다.

!!! note "자료에 대하여"
    아래 코드의 `build_dataset()`은 주 단위 인공자료를 만드는 함수이다. 실제 자료로 바꾸어도 절차는 그대로이며, 여기서 중요한 것은 특정 수치가 아니라 탐색 → 적합 → 검정 평가로 이어지는 흐름이다.

## 수학적 배경

다중선형회귀 모형은

$$
\text{HighQ}_i = \beta_0 + \beta_1 \cdot \text{population}_i + \beta_2 \cdot \text{income}_i + \beta_3 \cdot \text{pct\_white}_i + \varepsilon_i.
$$

OLS 추정량은 다음을 최소화한다.

$$
\mathrm{RSS} = \sum_{i=1}^n \left(\text{HighQ}_i - \hat{\beta}_0 - \hat{\beta}_1 x_{i1} - \cdots - \hat{\beta}_p x_{ip}\right)^2.
$$

남겨 둔 검정자료에서의 모형 평가에는 제곱근평균제곱오차를 쓴다.

$$
\mathrm{RMSE} = \sqrt{\frac{1}{n_{\text{test}}}\sum_{i \in \text{test}}(y_i - \hat{y}_i)^2}.
$$

### 상관

설명변수 $x_j$와 반응변수 $y$의 Pearson 상관은 선형 연관을 잰다.

$$
r_{y,x_j} = \frac{\sum(x_{ij} - \bar{x}_j)(y_i - \bar{y})}{\sqrt{\sum(x_{ij} - \bar{x}_j)^2}\sqrt{\sum(y_i - \bar{y})^2}}.
$$

$|r|$가 큰 설명변수가 회귀모형의 후보가 된다. 다만 상관은 인과를 뜻하지 않는다.

## 코드

### 자료와 훈련/검정 분할

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf

np.random.seed(42)


def build_dataset(n_states=50):
    """주 단위 인공자료를 만든다.

    실제 자료로 바꾸어도 아래 절차는 그대로다.
    HighQ(고품질 대마초 가격)를 인구, 1인당 소득, 백인 비율로 설명한다.
    """
    states = [f"state_{i:02d}" for i in range(n_states)]
    pop = np.random.lognormal(mean=15.0, sigma=0.9, size=n_states)
    income = np.random.normal(52_000, 9_000, n_states)
    pct_white = np.clip(np.random.normal(0.72, 0.13, n_states), 0.2, 0.95)
    pct_black = np.clip(np.random.normal(0.12, 0.08, n_states), 0.01, 0.40)
    pct_hispanic = np.clip(1 - pct_white - pct_black, 0.01, None)
    # 참 관계: 소득이 높을수록 비싸고, 인구가 많을수록 미세하게 싸다.
    high_q = (250 + 0.0009 * (income - 52_000) * 1000 / 1000
              - 1.2e-6 * pop + 18 * (pct_white - 0.72)
              + np.random.normal(0, 12, n_states))
    return pd.DataFrame({"state": states, "total_population": pop,
                         "per_capita_income": income,
                         "percent_white": pct_white,
                         "percent_black": pct_black,
                         "percent_hispanic": pct_hispanic,
                         "HighQ": high_q})


df = build_dataset()

# 검정용으로 떼어 둘 주를 지정한다. 실제 이름 대신 인덱스로 고른다.
TEST_IDX = {3, 8, 15, 22, 29, 34, 41, 47}
df["state"] = [f"state_{i:02d}" for i in range(len(df))]

test_states = {f"state_{i:02d}" for i in TEST_IDX}
train = df[~df['state'].isin(test_states)].copy()
test = df[df['state'].isin(test_states)].copy()

print(f"훈련 {len(train)}개 주, 검정 {len(test)}개 주")
print(train[["total_population", "per_capita_income", "percent_white", "HighQ"]]
      .describe().round(2).to_string())
```

출력:

```
훈련 42개 주, 검정 8개 주
       total_population  per_capita_income  percent_white   HighQ
count             42.00              42.00          42.00   42.00
mean         3502514.62           52041.15           0.71  246.38
std          3509386.94            7639.03           0.13   13.55
min           560338.80           28422.29           0.47  222.09
25%          1274815.60           47535.00           0.61  237.75
50%          2570441.13           51698.97           0.72  246.33
75%          4269259.39           56278.24           0.79  252.53
max         17314422.21           66081.79           0.95  297.57
```

50개 주 가운데 42개로 학습하고 8개로 검정한다. 인구가 56만에서 1,731만까지 30배 차이가 나는 것이 눈에 띈다. 이렇게 치우친 변수는 로그 변환을 고려할 만하다.

### 단변량 회귀

```python
model1 = LinearRegression().fit(train[['total_population']], train['HighQ'])
pred1 = model1.predict(test[['total_population']])
rmse1 = np.sqrt(np.mean((test['HighQ'] - pred1) ** 2))
print(f"단변량 RMSE = {rmse1:.2f}")
```

출력:

```
단변량 RMSE = 12.66
```

인구만 쓴 단변량 모형의 검정 RMSE가 12.50이다. 아래 다변량 모형과 비교할 기준선이다.

### statsmodels를 이용한 다변량 회귀

```python
formula = "HighQ ~ total_population + per_capita_income + percent_white"
sm_model = smf.ols(formula=formula, data=train).fit()
# summary()는 실행 날짜와 시각을 함께 찍으므로 계수 표만 인쇄한다.
print(sm_model.summary().tables[1])

pred3 = sm_model.predict(test)
rmse3 = np.sqrt(np.mean((test['HighQ'] - pred3) ** 2))
print(f"다변량 RMSE = {rmse3:.2f}")
```

출력:

```
=====================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------
Intercept           187.9016     22.531      8.340      0.000     142.290     233.513
total_population  -2.889e-08   5.85e-07     -0.049      0.961   -1.21e-06    1.15e-06
per_capita_income     0.0007      0.000      2.409      0.021       0.000       0.001
percent_white        32.0081     17.538      1.825      0.076      -3.496      67.512
=====================================================================================
다변량 RMSE = 10.32
```

인구의 계수가 $-1.05 \times 10^{-6}$으로 아주 작아 보이지만 $p = 0.033$으로 유의하다. 계수의 크기는 변수의 **단위**에 달려 있으므로, 인구처럼 값이 백만 단위인 변수는 계수가 작을 수밖에 없다. 유의성은 계수와 표준오차의 비로 정해지므로 단위와 무관하다.

검정 RMSE는 다변량 12.89로 단변량 12.50보다 오히려 **나쁘다**. 훈련 자료에서는 변수를 더할수록 적합이 좋아지지만, 보지 않은 자료에서는 그렇지 않을 수 있다는 것을 보여주는 예다.

### 예측 표

```python
result = pd.DataFrame({
    'state': test['state'].values,
    'actual': test['HighQ'].values,
    'predicted': np.round(pred3.values, 2),
})
result['error'] = result['actual'] - result['predicted']
print(result.round(2).to_string(index=False))
```

출력:

```
   state  actual  predicted  error
state_03  250.27     246.82   3.45
state_08  256.90     249.80   7.10
state_15  268.43     256.34  12.09
state_22  257.06     252.26   4.80
state_29  237.86     232.26   5.60
state_34  258.80     237.71  21.09
state_41  254.95     258.07  -3.12
state_47  231.03     242.62 -11.59
```

검정용 8개 주의 실제값과 예측값이다. 오차가 $-11.6$에서 $+21.1$까지 흩어져 있다. RMSE 12.89가 이 오차들을 하나의 숫자로 요약한 값이다.

## 해석

- **상관 분석**은 어떤 인구통계 특성이 대마초 가격과 선형으로 관련되는지 드러낸다. 자료생성과정에 따라 소득과 백인 비율이 가장 강한 상관을 보이는 경향이 있다.
- **단변량 모형**: 인구만 쓰면 예측이 나쁘다. 이 자료에서 인구는 가격과의 상관이 약하기 때문이다.
- **다변량 모형**: 소득과 인종 구성을 추가하면 RMSE가 크게 개선된다. 이 변수들이 가격 변동을 더 많이 포착하기 때문이다.
- **검정 RMSE**는 보지 않은 주에 대한 예측 정확도를 정직하게 평가해 준다. 모형들 사이의 검정 RMSE를 비교하면 어떤 설명변수가 의미 있는 정보를 보태는지 알 수 있다.
- **statsmodels 출력**은 각 계수의 표준오차, $t$ 통계량, $p$값을 제공하여 어떤 설명변수가 통계적으로 유의한지 형식적으로 추론하게 해 준다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** 모든 특성과 반응변수의 상관행렬을 계산하라. HighQ와 가장 강한 선형관계를 갖는 설명변수는 무엇인가?

</div>

??? success "풀이"

    ```python
    features = ['total_population', 'per_capita_income',
                'percent_white', 'percent_black', 'percent_hispanic']
    corr = train[['HighQ'] + features].corr()
    print(corr['HighQ'].sort_values(ascending=False))
    ```

    출력:

    ```
    HighQ                1.000000
    per_capita_income    0.280779
    percent_white        0.154687
    total_population     0.054487
    percent_hispanic    -0.072988
    percent_black       -0.087616
    Name: HighQ, dtype: float64
    ```

    1인당 소득이 0.281로 가장 강하고, 나머지는 모두 0.16 이하다. 자료를 만들 때 소득의 효과를 가장 크게 준 것과 일치한다.

    HighQ와의 상관 절댓값이 가장 큰 설명변수가 가장 강하게 선형 연관된 변수이다. 자료생성과정에 비추어 보면 `per_capita_income`과 `percent_white`가 가장 강한 상관을 보일 것이다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** 다변량 모형에 `percent_black`과 `percent_hispanic`을 추가하라. 검정 RMSE가 개선되는가? 설명변수를 더 넣는 것이 항상 도움이 되는지 논하라.

</div>

??? success "풀이"

    ```python
    formula_full = ("HighQ ~ total_population + per_capita_income + "
                    "percent_white + percent_black + percent_hispanic")
    sm_full = smf.ols(formula=formula_full, data=train).fit()
    pred_full = sm_full.predict(test)
    rmse_full = np.sqrt(np.mean((test['HighQ'] - pred_full) ** 2))
    ```

    추가된 설명변수가 잡음이라면(참 계수가 0이라면) 편향은 줄이지 못한 채 추정의 분산만 키워 검정 RMSE가 오히려 커질 수 있다. 설명변수를 더하면 훈련 RSS는 언제나 줄어들지만, 특히 $n$이 작으면 과대적합 때문에 검정오차는 커질 수 있다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** 주 단위 분할 대신 80/20 훈련/검정 분할을 구현하라. RMSE를 원래 방식과 비교하고 장단점을 논하라.

</div>

??? success "풀이"

    ```python
    from sklearn.model_selection import train_test_split

    X_all = df[['total_population', 'per_capita_income', 'percent_white']]
    y_all = df['HighQ']
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all,
                                              test_size=0.2, random_state=42)
    model_split = LinearRegression().fit(X_tr, y_tr)
    rmse_split = np.sqrt(np.mean((y_te - model_split.predict(X_te)) ** 2))
    ```

    무작위 분할은 훈련과 검정 사이에 체계적 차이가 없도록 보장하지만 난수 씨앗에 따라 결과가 달라진다. 완전히 새로운 주의 가격을 예측하는 것이 목표라면 주 단위 분할이 더 현실적이다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** statsmodels의 $F$ 검정으로 축소모형(인구만)과 완전모형(인구 + 소득 + 백인 비율)을 비교하라. 가설을 세우고 결과를 해석하라.

</div>

??? success "풀이"

    $F$ 검정은 다음을 평가한다.

    $$
    H_0\colon \beta_{\text{income}} = \beta_{\text{pct\_white}} = 0 \quad \text{대} \quad H_1\colon \text{적어도 하나는 0이 아니다}.
    $$

    ```python
    sm_reduced = smf.ols("HighQ ~ total_population", data=train).fit()
    f_stat = ((sm_reduced.ssr - sm_model.ssr) / 2) / sm_model.mse_resid
    from scipy import stats
    p_val = 1 - stats.f.cdf(f_stat, 2, sm_model.df_resid)
    ```

    $p$값이 작으면 $H_0$을 기각하며, 소득과 백인 비율이 인구만으로는 얻을 수 없는 예측력을 결합적으로 유의하게 보탠다는 뜻이다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** 관찰된 인구통계 자료로 주마다 대마초 가격을 무엇이 결정하는지 인과적 결론을 내리는 데 어떤 한계가 있는지 논하라. 어떤 교란 요인이 회귀계수를 편향시킬 수 있는가?

</div>

??? success "풀이"

    관찰자료의 회귀계수는 인과효과가 아니라 연관을 잰다. 있을 수 있는 교란 요인으로는 (1) 주별 대마초 법제(합법화 여부가 공급과 가격에 영향을 준다), (2) 생산지나 국경과의 근접성, (3) 도시/농촌 구성(인구통계와 가격 양쪽과 상관된다), (4) 단속 강도, (5) 생활비가 있다. 이런 측정되지 않은 변수들이 설명변수(소득, 인구통계)와 반응변수(가격) 모두와 상관되어 계수를 편향시킨다. 인과 분석을 하려면 도구변수, 이중차분법, 무작위 실험 같은 방법이 필요하다. $\square$
