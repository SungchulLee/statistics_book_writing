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
df = build_dataset()  # synthetic state-level data

TEST_STATES = {"iowa", "kentucky", "missouri", "nevada",
               "wyoming", "south dakota", "new jersey", "colorado_extra"}
train = df[~df['state'].isin(TEST_STATES)].copy()
test = df[df['state'].isin(TEST_STATES)].copy()
```

### 단변량 회귀

```python
model1 = LinearRegression().fit(train[['total_population']], train['HighQ'])
pred1 = model1.predict(test[['total_population']])
rmse1 = np.sqrt(np.mean((test['HighQ'] - pred1) ** 2))
```

### statsmodels를 이용한 다변량 회귀

```python
formula = "HighQ ~ total_population + per_capita_income + percent_white"
sm_model = smf.ols(formula=formula, data=train).fit()
print(sm_model.summary())

pred3 = sm_model.predict(test)
rmse3 = np.sqrt(np.mean((test['HighQ'] - pred3) ** 2))
```

### 예측 표

```python
result = pd.DataFrame({
    'state': test['state'].values,
    'actual': test['HighQ'].values,
    'predicted': np.round(pred3.values, 2),
})
result['error'] = result['actual'] - result['predicted']
```

## 해석

- **상관 분석**은 어떤 인구통계 특성이 대마초 가격과 선형으로 관련되는지 드러낸다. 자료생성과정에 따라 소득과 백인 비율이 가장 강한 상관을 보이는 경향이 있다.
- **단변량 모형**: 인구만 쓰면 예측이 나쁘다. 이 자료에서 인구는 가격과의 상관이 약하기 때문이다.
- **다변량 모형**: 소득과 인종 구성을 추가하면 RMSE가 크게 개선된다. 이 변수들이 가격 변동을 더 많이 포착하기 때문이다.
- **검정 RMSE**는 보지 않은 주에 대한 예측 정확도를 정직하게 평가해 준다. 모형들 사이의 검정 RMSE를 비교하면 어떤 설명변수가 의미 있는 정보를 보태는지 알 수 있다.
- **statsmodels 출력**은 각 계수의 표준오차, $t$ 통계량, $p$값을 제공하여 어떤 설명변수가 통계적으로 유의한지 형식적으로 추론하게 해 준다.

## 연습문제

**연습문제 1.** 모든 특성과 반응변수의 상관행렬을 계산하라. HighQ와 가장 강한 선형관계를 갖는 설명변수는 무엇인가?

??? success "풀이"

    ```python
    features = ['total_population', 'per_capita_income',
                'percent_white', 'percent_black', 'percent_hispanic']
    corr = train[['HighQ'] + features].corr()
    print(corr['HighQ'].sort_values(ascending=False))
    ```

    HighQ와의 상관 절댓값이 가장 큰 설명변수가 가장 강하게 선형 연관된 변수이다. 자료생성과정에 비추어 보면 `per_capita_income`과 `percent_white`가 가장 강한 상관을 보일 것이다. $\square$

---

**연습문제 2.** 다변량 모형에 `percent_black`과 `percent_hispanic`을 추가하라. 검정 RMSE가 개선되는가? 설명변수를 더 넣는 것이 항상 도움이 되는지 논하라.

??? success "풀이"

    ```python
    formula_full = ("HighQ ~ total_population + per_capita_income + "
                    "percent_white + percent_black + percent_hispanic")
    sm_full = smf.ols(formula=formula_full, data=train).fit()
    pred_full = sm_full.predict(test)
    rmse_full = np.sqrt(np.mean((test['HighQ'] - pred_full) ** 2))
    ```

    추가된 설명변수가 잡음이라면(참 계수가 0이라면) 편향은 줄이지 못한 채 추정의 분산만 키워 검정 RMSE가 오히려 커질 수 있다. 설명변수를 더하면 훈련 RSS는 언제나 줄어들지만, 특히 $n$이 작으면 과대적합 때문에 검정오차는 커질 수 있다. $\square$

---

**연습문제 3.** 주 단위 분할 대신 80/20 훈련/검정 분할을 구현하라. RMSE를 원래 방식과 비교하고 장단점을 논하라.

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

---

**연습문제 4.** statsmodels의 $F$ 검정으로 축소모형(인구만)과 완전모형(인구 + 소득 + 백인 비율)을 비교하라. 가설을 세우고 결과를 해석하라.

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

---

**연습문제 5.** 관찰된 인구통계 자료로 주마다 대마초 가격을 무엇이 결정하는지 인과적 결론을 내리는 데 어떤 한계가 있는지 논하라. 어떤 교란 요인이 회귀계수를 편향시킬 수 있는가?

??? success "풀이"

    관찰자료의 회귀계수는 인과효과가 아니라 연관을 잰다. 있을 수 있는 교란 요인으로는 (1) 주별 대마초 법제(합법화 여부가 공급과 가격에 영향을 준다), (2) 생산지나 국경과의 근접성, (3) 도시/농촌 구성(인구통계와 가격 양쪽과 상관된다), (4) 단속 강도, (5) 생활비가 있다. 이런 측정되지 않은 변수들이 설명변수(소득, 인구통계)와 반응변수(가격) 모두와 상관되어 계수를 편향시킨다. 인과 분석을 하려면 도구변수, 이중차분법, 무작위 실험 같은 방법이 필요하다. $\square$
