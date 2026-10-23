# 회귀 (대마초 가격과 인구통계)

## 개요

이 페이지는 인구, 1인당 소득, 인종 구성 같은 주(州) 단위 인구통계 특성으로 고급 대마초 가격을 예측하는 데 선형회귀를 적용한다. 상관행렬을 통한 탐색적 분석, scikit-learn과 statsmodels를 이용한 단변량·다변량 OLS, 그리고 남겨 둔 검정자료에서의 모형 성능 평가를 다룬다.

!!! note "자료에 대하여"
    아래 코드의 `build_dataset()`은 주 단위 인공자료를 만드는 함수이다. 실제 자료로 바꾸어도 절차는 그대로이며, 여기서 중요한 것은 특정 수치가 아니라 탐색 → 적합 → 검정 평가로 이어지는 흐름이다.

---

## 1. 수학적 배경

다중선형회귀 모형은

$$
\text{HighQ}_i = \beta_0 + \beta_1 \cdot \text{population}_i + \beta_2 \cdot \text{income}_i + \beta_3 \cdot \text{percent_white}_i + \varepsilon_i.
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

### 자료와 훈련/검정 분할

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> 자료 준비. `build_dataset()`으로 주 단위 인공자료 $50$개를 만들고 $8$개 주를 검정용으로 떼어 둔다.

**(1)** 이 자료는 **참 모형을 알고 만든 것**이다. `build_dataset()`의 식에서 참 절편과 세 참 계수, 그리고 오차의 표준편차를 읽어 적으시오. 식 안의 `* 1000 / 1000`은 무엇을 하는가.

**(2)** 인구의 최솟값이 $56$만, 최댓값이 $1731$만으로 $31$배 차이가 난다. 이 치우침을 **왜도**로 재고, 로그를 취하면 얼마가 되는지 확인하시오.

</div>

??? success "풀이"

    **(1) 해석적으로.** 자료를 만드는 줄은

    ```
    high_q = (250 + 0.0009 * (income - 52_000) * 1000 / 1000
              - 1.2e-6 * pop + 18 * (pct_white - 0.72)
              + np.random.normal(0, 12, n_states))
    ```

    이다. 세 계수는 각각 소득 $0.0009$, 인구 $-1.2\times10^{-6}$, 백인 비율 $18$이고 오차의 표준편차는 $12$다. 절편은 조금 생각해야 한다. 식이 중심화된 꼴로 적혀 있으므로 $\text{income} = 0$, $\text{pop} = 0$, $\text{pct_white} = 0$에서의 값을 계산해야 한다.

    $$
    \beta_0 = 250 - 0.0009 \times 52000 - 18 \times 0.72 = 250 - 46.8 - 12.96 = 190.24
    $$

    `* 1000 / 1000`은 **아무 일도 하지 않는다.** $1000$을 곱한 뒤 $1000$으로 나누니 항등이다. 단위를 바꾸려다 남은 흔적으로 보이며, 코드를 읽는 사람을 헷갈리게 할 뿐이다. 식의 뜻을 바꾸지 않으므로 위 참값들은 그대로다.

    **(2) 로그정규분포에서 뽑았으므로 오른쪽으로 치우쳤다.** `np.random.lognormal(mean=15.0, sigma=0.9)`가 만든 값이니 로그를 취하면 정규가 되어야 한다. 왜도로 확인한다.

    먼저 자료를 만든다.

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

    치우침을 수로 잰다.

    ```python
    # build_dataset 의 식에 적힌 참 모수를 꺼내 적는다.
    TRUE_INTERCEPT = 250 - 0.0009 * 52_000 - 18 * 0.72
    print(f"참 절편      = 250 - 0.0009*52000 - 18*0.72 = {TRUE_INTERCEPT:.4f}")
    print(f"참 b_pop     = -1.2e-06")
    print(f"참 b_income  =  0.0009")
    print(f"참 b_white   =  18")
    print(f"참 sigma     =  12")
    print()
    pop = train["total_population"]
    print(f"인구  최솟값 {pop.min():,.0f}  최댓값 {pop.max():,.0f}  최댓값/최솟값 = {pop.max() / pop.min():.1f}")
    print(f"      평균 {pop.mean():,.0f}  중앙값 {pop.median():,.0f}  (평균이 중앙값보다 크다)")
    print(f"      왜도 {pop.skew():.4f}")
    print(f"log(인구) 왜도 {np.log(pop).skew():.4f}")
    ```

    출력:

    ```
    참 절편      = 250 - 0.0009*52000 - 18*0.72 = 190.2400
    참 b_pop     = -1.2e-06
    참 b_income  =  0.0009
    참 b_white   =  18
    참 sigma     =  12

    인구  최솟값 560,339  최댓값 17,314,422  최댓값/최솟값 = 30.9
          평균 3,502,515  중앙값 2,570,441  (평균이 중앙값보다 크다)
          왜도 2.4519
    log(인구) 왜도 0.2292
    ```

    **왜도가 $2.45$ 에서 $0.23$ 으로 줄어든다.** 로그를 취하기 전에는 왜도가 $2$ 를 넘어 정규와 거리가 멀고, 취한 뒤에는 $0.23$ 으로 $0$ 에 가깝다. 자료를 로그정규분포에서 뽑았으니 당연한 결과다. 평균 $350$만이 중앙값 $257$만보다 큰 것도 같은 치우침의 다른 얼굴이다.

    **그렇다고 로그 변환이 반드시 옳은 것은 아니다.** 선형회귀의 네 가정 가운데 **설명변수의 분포에 대한 가정은 하나도 없다.** 가정은 모두 오차항에 대한 것이다. 치우친 설명변수가 문제가 되는 것은 (ㄱ) 극단값 하나가 지렛값으로 적합을 끌고 갈 때, (ㄴ) 참 관계가 수준이 아니라 로그에서 선형일 때다. 이 자료는 참 관계가 **인구의 수준에 선형**(`-1.2e-6 * pop`)이므로 로그를 취하면 오히려 모형이 틀려진다. 그러므로 본문의 "로그 변환을 고려할 만하다"는 **실제 자료를 다룰 때의 조언**으로 읽어야 하고, 이 인공자료에서는 적용되지 않는다. $\square$

### 단변량 회귀

<div class="exbox" markdown>

**보기 2.** <span class="diff easy" title="쉬움"></span> 단변량 모형. 인구 하나만 넣어 기준선을 잡고 검정 RMSE 를 잰다.

**(1)** 보기 1에서 읽은 참 모수를 쓰면 이 모형의 예측오차 표준편차를 **이론적으로** 계산할 수 있다. 세 설명변수가 서로 독립이라 보고, 인구만 쓴 모형이 남기는 오차의 표준편차와 세 변수를 다 쓴 모형의 그것을 각각 구하시오.

**(2)** 실제 검정 RMSE $12.66$과 (1)의 이론값을 견주시오. 검정자료가 $8$개뿐일 때 RMSE 의 상대 표준오차는 대략 얼마이며, 그 차이가 설명되는가.

</div>

??? success "풀이"

    **(1) 해석적으로.** 참 모형은

    $$
    \text{HighQ} = \beta_0 + \beta_{\text{pop}}\,\text{pop} + \beta_{\text{inc}}\,\text{inc} + \beta_{\text{w}}\,\text{w} + \varepsilon, \qquad \varepsilon \sim N(0, 12^2)
    $$

    이다. 세 설명변수가 독립이라 보면 분산이 더해진다.

    $$
    \operatorname{Var}(\text{HighQ}) = \beta_{\text{pop}}^2\sigma_{\text{pop}}^2 + \beta_{\text{inc}}^2\sigma_{\text{inc}}^2 + \beta_{\text{w}}^2\sigma_{\text{w}}^2 + 12^2
    $$

    인구만 설명변수로 쓰면 그 몫만 걷어 내고 나머지가 모두 오차로 남는다.

    $$
    \sigma^2_{\text{단변량}} = \beta_{\text{inc}}^2\sigma_{\text{inc}}^2 + \beta_{\text{w}}^2\sigma_{\text{w}}^2 + 12^2
    $$

    세 변수를 다 쓰면 남는 것은 $12$ 하나뿐이다. **참 모형을 정확히 적합해도 예측오차를 $12$ 아래로 내릴 수 없다.** 이것이 줄일 수 없는 오차다.

    **(2) 수치적으로.** 먼저 모형을 적합한다.

    ```python
    # 먼저 인구 하나만 넣은 단변량 모형으로 기준선을 잡는다.
    model1 = LinearRegression().fit(train[['total_population']], train['HighQ'])
    pred1 = model1.predict(test[['total_population']])
    rmse1 = np.sqrt(np.mean((test['HighQ'] - pred1) ** 2))
    print(f"단변량 RMSE = {rmse1:.2f}")
    ```

    출력:

    ```
    단변량 RMSE = 12.66
    ```

    이제 이론값과 견준다.

    ```python
    # 참 모수와 훈련자료의 표준편차로 이론 예측오차를 계산한다.
    sd_pop = train["total_population"].std()
    sd_income = train["per_capita_income"].std()
    sd_white = train["percent_white"].std()
    print(f"훈련자료 표준편차:  인구 {sd_pop:,.0f}   소득 {sd_income:,.0f}   백인비율 {sd_white:.4f}")
    print()
    print("각 설명변수가 1 표준편차 오를 때 HighQ 의 참 변화량")
    for name, beta, sd in [("인구", -1.2e-6, sd_pop), ("소득", 0.0009, sd_income),
                           ("백인비율", 18.0, sd_white)]:
        print(f"  {name:8s} {beta:+.2e} x {sd:12,.4f} = {beta * sd:+8.4f}")
    print()
    theory_total = np.sqrt((0.0009 * sd_income) ** 2 + (18 * sd_white) ** 2
                           + (1.2e-6 * sd_pop) ** 2 + 12.0 ** 2)
    theory_uni = np.sqrt(theory_total ** 2 - (1.2e-6 * sd_pop) ** 2)
    print(f"이론 sd(HighQ)                = {theory_total:.4f}   (관측 {train.HighQ.std():.4f})")
    print(f"인구만 쓴 모형의 이론 예측오차 = {theory_uni:.4f}")
    print(f"세 변수를 다 쓴 모형의 이론값  = 12.0000  (= 참 sigma)")
    print()
    print(f"실제 검정 RMSE                = {rmse1:.4f}")
    print(f"검정자료가 {len(test)}개이므로 RMSE 의 상대 표준오차 ~ 1/sqrt(2n) = {1 / np.sqrt(2 * len(test)):.4f}")
    print(f"이론값에서의 상대 차이        = {rmse1 / theory_uni - 1:+.4f}")
    ```

    출력:

    ```
    훈련자료 표준편차:  인구 3,509,387   소득 7,639   백인비율 0.1252

    각 설명변수가 1 표준편차 오를 때 HighQ 의 참 변화량
      인구       -1.20e-06 x 3,509,386.9355 =  -4.2113
      소득       +9.00e-04 x   7,639.0328 =  +6.8751
      백인비율     +1.80e+01 x       0.1252 =  +2.2541

    이론 sd(HighQ)                = 14.6316   (관측 13.5481)
    인구만 쓴 모형의 이론 예측오차 = 14.0124
    세 변수를 다 쓴 모형의 이론값  = 12.0000  (= 참 sigma)

    실제 검정 RMSE                = 12.6631
    검정자료가 8개이므로 RMSE 의 상대 표준오차 ~ 1/sqrt(2n) = 0.2500
    이론값에서의 상대 차이        = -0.0963
    ```

    **이론은 $14.01$, 실제는 $12.66$ 으로 $9.6\%$ 낮다.** 작아 보이는 어긋남이지만 방향을 판정하려면 불확실성을 알아야 한다. $n_{\text{test}} = 8$ 에서 RMSE 의 상대 표준오차는 대략 $1/\sqrt{2n} = 0.25$, 곧 **$25\%$** 다. $9.6\%$ 는 그 절반도 안 되므로 **표본변동으로 완전히 설명된다.** 이론과 자료가 어긋난다고 말할 근거가 없다.

    눈여겨볼 것은 **인구가 설명하는 몫이 보잘것없다**는 사실이다. 참 모형에서 인구가 $1$ 표준편차 오를 때 HighQ 는 $-4.21$ 달러 변하는데, 줄일 수 없는 오차의 표준편차가 $12$ 다. 분산으로 재면 인구의 몫이 전체의 $4.21^2/14.63^2 = 8.3\%$ 에 지나지 않는다. 그래서 인구만 쓴 모형의 이론 예측오차 $14.01$ 이 아무 설명변수도 안 쓴 $14.63$ 과 거의 다르지 않다. **인구는 참 계수가 $0$ 이 아니지만 쓸모가 거의 없는 변수다.**

    이론 sd(HighQ) $14.63$ 이 관측값 $13.55$ 보다 큰 것도 한마디 할 만하다. $n = 42$ 에서 표준편차의 상대 표준오차가 $1/\sqrt{2 \times 41} = 11\%$ 이고 차이가 $7.4\%$ 이므로 역시 표본변동 범위 안이다. 세 설명변수가 서로 완전히 독립이 아니라는 점(`pct_hispanic`을 만들면서 `pct_white`가 끌려 들어간다)도 조금 기여한다. $\square$

### 세 설명변수를 모두 넣은 모형

<div class="exbox" markdown>

**보기 3.** <span class="diff easy" title="쉬움"></span> 다변량 모형. 세 설명변수를 모두 넣고 `statsmodels`로 적합해 계수 표와 검정 RMSE 를 본다.

**(1)** 이 자료는 참 모수를 알고 만들었다(보기 1). 네 참 모수가 각각 **$95\%$ 신뢰구간 안에 들어오는지** 확인하시오. $95\%$ 신뢰구간이란 무엇을 뜻하며, 네 개가 모두 들어온 것은 얼마나 놀랄 일인가.

**(2)** 잔차 표준오차 $s$ 를 참 $\sigma = 12$ 와 견주고, 훈련 $R^2$ 를 이론값 $1 - 12^2/14.63^2$ 과 견주시오. 검정 RMSE 가 $12.66$ 에서 $10.32$ 로 줄어든 것은 유의한 개선인가.

</div>

??? success "풀이"

    **(1) 넷 다 들어와야 하는 일이고, 실제로 들어온다.** $95\%$ 신뢰구간의 뜻은 "같은 실험을 되풀이하면 구간이 참값을 덮는 비율이 $0.95$ 가 된다"이다. 네 모수가 각각 $0.95$ 확률로 덮이므로, **하나라도 벗어날 확률이 작지 않다.** 네 구간이 서로 독립이라면 모두 덮을 확률이 $0.95^4 = 0.8145$ 이고, 실제로는 상관되어 있어 그보다 높다. 그러므로 "넷 다 들어왔다"는 흔한 결과이고 모형이 특별히 훌륭하다는 뜻이 아니다. 거꾸로 하나가 벗어났더라도 그 자체로는 경고가 아니다.

    **(2) 해석적으로.** 잔차 표준오차 $s^2 = \text{RSS}/(n-p)$ 는 $\sigma^2$ 의 불편추정량이다. 자유도 $n - p = 42 - 4 = 38$ 에서 $s$ 의 상대 표준오차는 대략 $1/\sqrt{2 \times 38} = 11.5\%$ 다.

    이론 $R^2$ 는 보기 2의 분산분해에서 바로 나온다.

    $$
    R^2_{\text{이론}} = 1 - \frac{\sigma^2}{\operatorname{Var}(\text{HighQ})} = 1 - \frac{12^2}{14.6316^2} = 0.3274
    $$

    ```python
    # 변수를 셋으로 늘려 RMSE 가 실제로 줄어드는지 본다.
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

    이제 참값과 맞춰 본다.

    ```python
    # 참 모수가 95% 신뢰구간에 들어오는지 확인한다.
    truth = {"Intercept": 190.24, "total_population": -1.2e-6,
             "per_capita_income": 0.0009, "percent_white": 18.0}
    ci = sm_model.conf_int()
    print(f"{'항':20s}{'추정값':>14s}{'95% CI 하한':>15s}{'95% CI 상한':>15s}{'참값':>14s}  판정")
    for name, true_value in truth.items():
        low, high = ci.loc[name]
        verdict = "포함" if low <= true_value <= high else "벗어남"
        print(f"{name:20s}{sm_model.params[name]:14.6g}{low:15.6g}{high:15.6g}"
              f"{true_value:14.6g}  {verdict}")
    print()
    print(f"잔차 표준오차 s = {np.sqrt(sm_model.mse_resid):.4f}   (참 sigma = 12)")
    print(f"훈련 R^2        = {sm_model.rsquared:.4f}")
    print(f"  이론 R^2 = 1 - 12^2/14.6316^2 = {1 - 144 / 14.6316 ** 2:.4f}")
    print()
    print(f"단변량 검정 RMSE = {rmse1:.4f}   다변량 검정 RMSE = {rmse3:.4f}"
          f"   비 = {rmse3 / rmse1:.4f}")
    ```

    출력:

    ```
    항                              추정값      95% CI 하한      95% CI 상한            참값  판정
    Intercept                  187.902         142.29        233.513        190.24  포함
    total_population      -2.88857e-08   -1.21254e-06    1.15477e-06      -1.2e-06  포함
    per_capita_income      0.000688352    0.000109966     0.00126674        0.0009  포함
    percent_white              32.0081        -3.4962        67.5125            18  포함

    잔차 표준오차 s = 12.9413   (참 sigma = 12)
    훈련 R^2        = 0.1543
      이론 R^2 = 1 - 12^2/14.6316^2 = 0.3274

    단변량 검정 RMSE = 12.6631   다변량 검정 RMSE = 10.3194   비 = 0.8149
    ```

    **(1) 네 구간 모두 참값을 덮는다.** 다만 덮는 방식이 서로 다르다는 점을 보아야 한다. 인구의 참값 $-1.2\times10^{-6}$ 은 구간 $[-1.213\times10^{-6},\ 1.155\times10^{-6}]$ 의 **거의 왼쪽 끝**에 간신히 들어 있고, 백인 비율의 참값 $18$ 은 구간 $[-3.50,\ 67.51]$ 의 가운데쯤에 있지만 그 구간이 **$0$ 을 포함할 만큼 넓다.** 추정값 $32.01$ 이 참값 $18$ 의 $1.78$ 배인데도 유의하지 않다고 나오는 것이 이 넓이의 결과다.

    이것이 **표본이 $42$ 개뿐일 때 일어나는 일**이다. 참 효과가 분명히 있어도($\beta_{\text{w}} = 18 \ne 0$) 검정이 그것을 잡아내지 못한다. 거꾸로 추정값이 참값의 두 배 가까이 나와도 "틀렸다"고 말할 수 없다. 구간이 넓으면 **어느 쪽 결론도 내릴 수 없다.**

    **(2) $s = 12.94$ 가 참 $\sigma = 12$ 와 잘 맞는다.** 상대 차이 $7.8\%$ 로 $s$ 의 상대 표준오차 $11.5\%$ 안쪽이다. 추정이 제대로 되고 있다는 뜻이다.

    훈련 $R^2$ 는 사정이 다르다. $0.1543$ 으로 이론값 $0.3274$ 의 절반도 안 된다. $R^2$ 가 작은 쪽으로 어긋난 것은 **이 표본의 $\operatorname{Var}(\text{HighQ})$ 가 작게 나왔기 때문**이다(보기 2에서 관측 $13.55$ 대 이론 $14.63$). $R^2 = 1 - s^2/s_y^2$ 에서 분모가 작아지면 $R^2$ 가 내려간다. 실제로 $1 - 12.9413^2/13.5481^2 = 0.0876$ 인데 $R^2$ 정의의 분모가 $\text{TSS}/(n-1)$ 이 아니라 자유도 보정이 없는 쪽이라 $0.1543$ 이 나온다. 어느 쪽이든 **$R^2$ 가 $0.15$ 든 $0.33$ 든 이 모형은 가격 변동의 대부분을 설명하지 못한다.** 참 모형이 그렇게 만들어졌기 때문이다.

    **검정 RMSE 의 개선은 유의하다고 말할 수 없다.** $12.66 \to 10.32$ 로 $18.5\%$ 줄었고 이론은 $14.01 \to 12.00$ 으로 $14.4\%$ 줄 것이라 예측했으니 **방향과 크기가 모두 이론과 맞는다.** 그러나 $n_{\text{test}} = 8$ 에서 RMSE 의 상대 표준오차가 $25\%$ 이므로, 이 한 번의 비교만으로 개선을 선언하면 과하다. 두 RMSE 가 **같은 $8$ 개 주**에서 계산되어 서로 상관되어 있다는 점이 비교에 도움을 주지만, 그 상관을 쓰려면 짝지은 비교를 따로 해야 한다.

    요약하면 이렇다. **이론이 예측한 세 가지(계수의 포함, $s \approx \sigma$, RMSE 의 감소)가 모두 자료에서 확인되지만, $42$ 개 훈련·$8$ 개 검정이라는 크기로는 어느 것도 날카롭게 확인되지 않는다.** $\square$

### 계수의 크기는 무엇을 뜻하는가

![날것의 계수와 1표준편차 효과](./img/coef_scale_vs_effect.png)

위 출력표에서 세 계수는 $-2.889\times10^{-8}$, $0.000688$, $32.01$이다. 크기만 보면 백인 비율이 인구보다 **10억 배** 중요해 보인다. 왼쪽 그림이 그 착시다. 가로축이 로그 눈금이어야 세 막대를 한 화면에 그릴 수 있을 만큼 벌어져 있는데, 이 차이는 전적으로 **단위**에서 온다. 인구는 56만에서 1731만까지 가는 수이고, 백인 비율은 $0.47$에서 $0.95$ 사이의 수다. 같은 크기의 영향을 주더라도 값이 백만 단위인 변수는 계수가 백만분의 일 단위로 나올 수밖에 없다.

오른쪽이 같은 계수를 견줄 수 있게 옮긴 것이다. 각 설명변수가 **1 표준편차**만큼 오를 때 HighQ가 얼마나 변하는지, 곧 $\hat\beta_j \times \mathrm{SD}(x_j)$를 그렸다. 훈련자료에서 인구의 표준편차는 $3{,}509{,}387$, 소득은 $7{,}639$, 백인 비율은 $0.1252$다. 곱해 보면 인구 $-0.10$달러, 소득 $+5.26$달러, 백인 비율 $+4.01$달러가 된다. 세 수가 이제 같은 단위(달러)에 있고 크기도 비슷하다. **그리고 순위가 뒤집힌다.** 날것의 계수로는 백인 비율이 압도적으로 컸지만, 표준편차로 재면 소득이 가장 큰 효과를 갖는다.

$t$ 값이 이 재척도화와 **무관하다**는 점도 함께 보아 두자. $t_j = \hat\beta_j/\mathrm{SE}(\hat\beta_j)$인데 변수의 단위를 바꾸면 분자와 분모가 같은 비율로 변하므로 비는 그대로다. 실제로 $t$는 인구 $-0.05$, 소득 $+2.41$, 백인 비율 $+1.83$으로, 1표준편차 효과의 순위와 일치한다. 인구는 계수가 $0$과 구별되지 않고($p = 0.961$), 소득만 $5\%$ 수준에서 유의하다. 그러므로 회귀표를 읽을 때의 규칙은 이렇다. **유의성은 $t$와 $p$로 읽고, 중요도는 표준편차당 효과로 읽는다.** 계수 자체의 크기는 둘 중 어느 쪽도 말해 주지 않는다.

### 예측 표

<div class="exbox" markdown>

**보기 4.** <span class="diff easy" title="쉬움"></span> 주별 예측오차 보기. 검정용 $8$개 주의 실제값·예측값·오차를 한 표로 늘어놓는다.

**(1)** 표의 여덟 오차에서 RMSE 를 손으로 계산해 보기 3의 $10.32$ 와 맞추시오. **평균 오차**를 재고, 그것이 $0$ 이 아닌 것이 무엇을 뜻하는지 말하시오. $\text{RMSE}^2$ 를 치우침과 퍼짐으로 나누어 적으시오.

**(2)** 오차 하나가 $+21.09$ 로 유난히 크다. 이 주 하나가 RMSE 에 얼마나 기여하는지 재고, 빼면 RMSE 가 얼마가 되는지 보시오. "평균 오차 하나만 보면 놓치는 것"이 구체적으로 무엇인가.

</div>

??? success "풀이"

    **(1) 해석적으로.** RMSE 는 오차의 **이차 적률**이므로 치우침과 퍼짐이 섞여 들어간다. 표본평균 $\bar e$ 와 표본분산 $s_e^2$ 로 쓰면

    $$
    \text{RMSE}^2 = \frac{1}{n}\sum_i e_i^2 = \bar e^2 + \frac{1}{n}\sum_i (e_i - \bar e)^2 = \bar e^2 + \frac{n-1}{n}s_e^2
    $$

    이다. 앞항이 **치우침의 제곱**, 뒷항이 **퍼짐**이다. 평균 오차가 $0$ 이 아니면 모형이 검정자료에서 체계적으로 한쪽으로 빗나간다는 뜻이고, 그 몫이 RMSE 에 제곱으로 들어온다.

    훈련자료에서는 최소제곱의 정규방정식이 $\sum_i e_i = 0$ 을 보장하므로(A.4절, 그리고 단순회귀의 보기 2) **치우침이 정확히 $0$ 이다.** 검정자료에서는 그런 보장이 없다.

    **(2) 수치적으로.**

    ```python
    # 주별로 실제값과 예측값을 나란히 놓아 어느 주에서 크게 빗나갔는지 본다.
    # 평균 오차 하나만 보아서는 놓치는 것이 있다.
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

    검정용 8개 주의 실제값과 예측값이다. 오차가 $-11.6$ 에서 $+21.1$ 까지 흩어져 있다. 이제 이 여덟 수를 하나로 요약해 본다.

    ```python
    # 여덟 오차로 RMSE 를 손으로 계산하고, 치우침과 극단값의 몫을 본다.
    errors = result['error'].values
    print(f"오차 = {np.round(errors, 2)}")
    print(f"제곱합 = {np.sum(errors ** 2):.4f},  나누기 {len(errors)} = {np.mean(errors ** 2):.4f}")
    print(f"제곱근 = {np.sqrt(np.mean(errors ** 2)):.4f}   (보기 3 의 RMSE = {rmse3:.4f})")
    print()
    print(f"평균 오차(치우침) = {errors.mean():+.4f}")
    print(f"오차의 표준편차   = {errors.std(ddof=1):.4f}")
    print(f"평균 오차의 표준오차 = {errors.std(ddof=1) / np.sqrt(len(errors)):.4f}"
          f"   t = {errors.mean() / (errors.std(ddof=1) / np.sqrt(len(errors))):.4f}")
    print(f"RMSE^2 = 치우침^2 + (n-1)/n * 표준편차^2 = "
          f"{errors.mean() ** 2:.4f} + {(len(errors) - 1) / len(errors) * errors.var(ddof=1):.4f}"
          f" = {errors.mean() ** 2 + (len(errors) - 1) / len(errors) * errors.var(ddof=1):.4f}")
    print()
    worst = np.argmax(np.abs(errors))
    print(f"가장 큰 오차: {result['state'].iloc[worst]}  오차 {errors[worst]:+.2f}")
    print(f"  제곱합에서 차지하는 몫 = {errors[worst] ** 2 / np.sum(errors ** 2):.4f}")
    print(f"  이 주를 빼면 RMSE = {np.sqrt(np.mean(np.delete(errors, worst) ** 2)):.4f}")
    print(f"오차가 양수인 주 {np.sum(errors > 0)}개, 음수인 주 {np.sum(errors < 0)}개")
    ```

    출력:

    ```
    오차 = [  3.45   7.1   12.09   4.8    5.6   21.09  -3.12 -11.59]
    제곱합 = 851.9593,  나누기 8 = 106.4949
    제곱근 = 10.3196   (보기 3 의 RMSE = 10.3194)

    평균 오차(치우침) = +4.9279
    오차의 표준편차   = 9.6930
    평균 오차의 표준오차 = 3.4270   t = 1.4380
    RMSE^2 = 치우침^2 + (n-1)/n * 표준편차^2 = 24.2847 + 82.2102 = 106.4949

    가장 큰 오차: state_34  오차 +21.09
      제곱합에서 차지하는 몫 = 0.5222
      이 주를 빼면 RMSE = 7.6257
    오차가 양수인 주 6개, 음수인 주 2개
    ```

    **(1) 손계산과 보기 3의 값이 $10.3196$ 대 $10.3194$ 로 넷째 자리에서 어긋난다.** 이 어긋남에는 까닭이 있다. 표를 만들 때 `np.round(pred3.values, 2)` 로 예측값을 소수 둘째 자리에서 **잘랐기** 때문이다. 표의 오차는 반올림된 예측값으로 계산된 것이고 보기 3의 RMSE 는 자르지 않은 값으로 계산된 것이다. 어긋남 $0.0002$ 가 반올림 오차 $0.005$ 수준과 맞는다. **표에 찍힌 수로 다시 계산해 원래 값과 비교할 때는 반올림부터 확인해야 한다.**

    분해도 정확히 닫힌다. $\text{RMSE}^2 = 106.4949$ 가 치우침 제곱 $24.2847$ 과 퍼짐 $82.2102$ 의 합이다. 치우침이 전체의 $22.8\%$ 를 차지한다.

    **평균 오차가 $+4.93$ 이다.** 여덟 주 가운데 여섯 곳에서 모형이 실제보다 **낮게** 예측했다. 훈련자료에서라면 이런 일이 있을 수 없다. 정규방정식이 $\sum e_i = 0$ 을 강제하기 때문이다. 검정자료에서 치우침이 생긴 것은 **이 여덟 주가 우연히 평균보다 비싼 주들이었다**는 뜻으로 읽어야 한다. 실제로 $t = 4.9279/3.4270 = 1.44$ 로 유의하지 않다. $8$ 개로는 치우침이 체계적인지 우연인지 가릴 수 없다.

    **(2) state_34 하나가 제곱합의 $52.2\%$ 를 차지한다.** 여덟 개 가운데 하나가 절반을 넘게 가져간다. 이 주를 빼면 RMSE 가 $10.32$ 에서 $7.63$ 으로 **$26\%$ 떨어진다.** RMSE 가 오차를 제곱해 더하기 때문이고, $21.09$ 는 다른 오차들보다 두 배 이상 크다.

    그러므로 **"평균 오차 하나만 보면 놓치는 것"은 세 가지다.** 첫째, 오차가 어느 쪽으로 치우쳤는가. RMSE $10.32$ 만 보면 $+4.93$ 의 치우침이 보이지 않는다. 둘째, 그 오차가 고르게 퍼져 있는가 아니면 한 관측값에 몰려 있는가. 셋째, 어느 관측값이 문제인가. 세 번째가 실무에서 가장 값지다. state_34 를 지목할 수 있으면 그 주의 자료를 다시 들여다보거나 모형이 놓친 변수를 찾아볼 수 있다. **표를 늘어놓는 일은 요약값을 보완하는 것이 아니라 요약값이 가린 것을 되찾는 일이다.** $\square$

---

## 2. 해석

- **상관 분석**은 어떤 인구통계 특성이 대마초 가격과 선형으로 관련되는지 드러낸다. 자료생성과정에 따라 소득과 백인 비율이 가장 강한 상관을 보이는 경향이 있다.
- **단변량 모형**: 인구만 쓰면 예측이 나쁘다. 이 자료에서 인구는 가격과의 상관이 약하기 때문이다.
- **다변량 모형**: 소득과 인종 구성을 추가하면 RMSE가 크게 개선된다. 이 변수들이 가격 변동을 더 많이 포착하기 때문이다.
- **검정 RMSE**는 보지 않은 주에 대한 예측 정확도를 정직하게 평가해 준다. 모형들 사이의 검정 RMSE를 비교하면 어떤 설명변수가 의미 있는 정보를 보태는지 알 수 있다.
- **statsmodels 출력**은 각 계수의 표준오차, $t$ 통계량, $p$값을 제공하여 어떤 설명변수가 통계적으로 유의한지 형식적으로 추론하게 해 준다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span> 모든 특성과 반응변수의 상관행렬을 계산하라. HighQ와 가장 강한 선형관계를 갖는 설명변수는 무엇인가?

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

**연습문제 2.** <span class="diff med" title="중간"></span> 다변량 모형에 `percent_black`과 `percent_hispanic`을 추가하라. 검정 RMSE가 개선되는가? 설명변수를 더 넣는 것이 항상 도움이 되는지 논하라.

</div>

??? success "풀이"

    ```python
    formula_full = ("HighQ ~ total_population + per_capita_income + "
                    "percent_white + percent_black + percent_hispanic")
    sm_full = smf.ols(formula=formula_full, data=train).fit()
    pred_full = sm_full.predict(test)
    rmse_full = np.sqrt(np.mean((test['HighQ'] - pred_full) ** 2))
    ```

    추가된 설명변수가 잡음이라면(참 계수가 0이라면) 편향은 줄이지 못한 채 추정의 분산만 키워 검정 RMSE가 오히려 커질 수 있다. 설명변수를 더하면 훈련 RSS는 언제나 줄어들지만, 특히 $n$이 작으면 과적합 때문에 검정오차는 커질 수 있다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> 주 단위 분할 대신 80/20 훈련/검정 분할을 구현하라. RMSE를 원래 방식과 비교하고 장단점을 논하라.

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

**연습문제 4.** <span class="diff med" title="중간"></span> statsmodels의 $F$ 검정으로 축소모형(인구만)과 완전모형(인구 + 소득 + 백인 비율)을 비교하라. 가설을 세우고 결과를 해석하라.

</div>

??? success "풀이"

    $F$ 검정은 다음을 평가한다.

    $$
    H_0\colon \beta_{\text{income}} = \beta_{\text{percent_white}} = 0 \quad \text{대} \quad H_1\colon \text{적어도 하나는 0이 아니다}.
    $$

    ```python
    sm_reduced = smf.ols("HighQ ~ total_population", data=train).fit()
    f_stat = ((sm_reduced.ssr - sm_model.ssr) / 2) / sm_model.mse_resid
    from scipy import stats
    p_val = 1 - stats.f.cdf(f_stat, 2, sm_model.df_resid)
    ```

    $p$값이 작으면 $H_0$을 기각하며, 소득과 백인 비율이 인구만으로는 얻을 수 없는 예측력을 결합적으로 유의하게 보탠다는 뜻이다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> 관찰된 인구통계 자료로 주마다 대마초 가격을 무엇이 결정하는지 인과적 결론을 내리는 데 어떤 한계가 있는지 논하라. 어떤 교란 요인이 회귀계수를 편향시킬 수 있는가?

</div>

??? success "풀이"

    관찰자료의 회귀계수는 인과효과가 아니라 연관을 잰다. 있을 수 있는 교란 요인으로는 (1) 주별 대마초 법제(합법화 여부가 공급과 가격에 영향을 준다), (2) 생산지나 국경과의 근접성, (3) 도시/농촌 구성(인구통계와 가격 양쪽과 상관된다), (4) 단속 강도, (5) 생활비가 있다. 이런 측정되지 않은 변수들이 설명변수(소득, 인구통계)와 반응변수(가격) 모두와 상관되어 계수를 편향시킨다. 인과 분석을 하려면 도구변수, 이중차분법, 무작위 실험 같은 방법이 필요하다. $\square$

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span> "계수의 크기는 무엇을 뜻하는가" 절은 $t$ 값이 변수의 단위를 바꾸어도 그대로라고 했다. 이것을 끝까지 밀어 본다. `total_population` 을 사람 수가 아니라 **백만 명** 단위로 적어($10^6$ 으로 나눈다) 보기 3 과 똑같은 세 변수 모형을 다시 적합한다고 하자.

**(1)** 적합해 보기 전에 다음 열 가지 양이 어떻게 바뀔지 **예측**하시오. 인구의 계수, 그 표준오차, $t$ 값, $p$ 값, 인구의 $1$ 표준편차 효과, 절편, 소득의 계수, 훈련 $R^2$, 잔차 표준오차 $s$, 검정 RMSE. 바뀌는 것과 바뀌지 않는 것을 가르는 까닭을 한 문장으로 적으시오.

**(2)** 코드로 확인하고, 날것의 계수가 중요도를 말해 주지 못하는 이유를 이 결과로 설명하시오.

</div>

??? success "풀이"

    **(1) 해석적으로.** 인구 열만 $c = 10^6$ 으로 나누는 것은 설계행렬에 대각행렬을 곱하는 것이다. $\tilde X = XD$, $D = \operatorname{diag}(1, 1/c, 1, 1)$ 로 쓰면

    $$
    \tilde\beta = (\tilde X^\top \tilde X)^{-1}\tilde X^\top y = D^{-1}(X^\top X)^{-1}X^\top y = D^{-1}\hat\beta
    $$

    이고 공분산도 $\operatorname{Cov}(\tilde\beta) = D^{-1}\operatorname{Cov}(\hat\beta)D^{-1}$ 이다. 그러므로 인구의 계수와 표준오차는 **둘 다 $c$ 배**가 되고 $t_j = \tilde\beta_j/\mathrm{SE}(\tilde\beta_j)$ 에서는 $c$ 가 약분되어 $t$ 와 $p$ 가 그대로다. $1$ 표준편차 효과도 $\tilde\beta\cdot\mathrm{SD}(\tilde x) = c\hat\beta \cdot \mathrm{SD}(x)/c$ 로 그대로다. $D$ 의 다른 대각성분이 $1$ 이므로 절편과 소득 계수는 손대지 않는다.

    나머지는 더 센 이유로 변하지 않는다. $\tilde X = XD$ 이고 $D$ 가 가역이므로 **두 설계행렬의 열공간이 같다.** 적합값은 $y$ 를 그 열공간에 사영한 것이니 똑같고, 따라서 잔차·RSS·$R^2$·$s$ 가 모두 같다. 검정 예측값도 검정 설계행렬을 같은 $D$ 로 재므로 같고 RMSE 도 같다.

    가르는 선은 이것이다. **열공간(곧 적합값)만으로 정해지는 양은 모두 불변이고, 축의 좌표를 읽는 양 둘만 바뀐다.**

    **(2) 수치적으로.**

    ```python
    # 인구의 단위만 사람에서 백만 명으로 바꾸어 같은 모형을 다시 적합한다.
    train_m, test_m = train.copy(), test.copy()
    train_m["pop_mil"] = train_m["total_population"] / 1e6
    test_m["pop_mil"] = test_m["total_population"] / 1e6
    sm_m = smf.ols("HighQ ~ pop_mil + per_capita_income + percent_white",
                   data=train_m).fit()
    pred_m = sm_m.predict(test_m)
    rmse_m = np.sqrt(np.mean((test_m["HighQ"] - pred_m) ** 2))

    old, new = "total_population", "pop_mil"
    rows = [("인구 계수", sm_model.params[old], sm_m.params[new]),
            ("인구 표준오차", sm_model.bse[old], sm_m.bse[new]),
            ("인구 t 값", sm_model.tvalues[old], sm_m.tvalues[new]),
            ("인구 p 값", sm_model.pvalues[old], sm_m.pvalues[new]),
            ("인구 1 SD 효과", sm_model.params[old] * train[old].std(),
             sm_m.params[new] * train_m[new].std()),
            ("절편", sm_model.params["Intercept"], sm_m.params["Intercept"]),
            ("소득 계수", sm_model.params["per_capita_income"],
             sm_m.params["per_capita_income"]),
            ("훈련 R^2", sm_model.rsquared, sm_m.rsquared),
            ("잔차 표준오차 s", np.sqrt(sm_model.mse_resid), np.sqrt(sm_m.mse_resid)),
            ("검정 RMSE", rmse3, rmse_m)]
    print(f"{'양':18s}{'사람 단위':>16s}{'백만 명 단위':>16s}  같은가")
    for name, a, b in rows:
        print(f"{name:18s}{a:16.6g}{b:16.6g}  {'예' if np.isclose(a, b) else '아니오'}")
    print()
    print(f"계수의 비 = {sm_m.params[new] / sm_model.params[old]:.6g}")
    ```

    출력:

    ```text
    양                            사람 단위         백만 명 단위  같은가
    인구 계수                 -2.88857e-08      -0.0288857  아니오
    인구 표준오차                5.84695e-07        0.584695  아니오
    인구 t 값                  -0.0494031      -0.0494031  예
    인구 p 값                    0.960857        0.960857  예
    인구 1 SD 효과               -0.101371       -0.101371  예
    절편                         187.902         187.902  예
    소득 계수                  0.000688352     0.000688352  예
    훈련 R^2                    0.154334        0.154334  예
    잔차 표준오차 s                  12.9413         12.9413  예
    검정 RMSE                    10.3194         10.3194  예

    계수의 비 = 1e+06
    ```

    **열 가지 가운데 둘만 바뀌고, 그 둘은 정확히 $10^6$ 배가 된다.** 예측대로다.

    여기서 날것의 계수에 대한 결론이 나온다. 인구의 계수를 $-2.89\times10^{-8}$ 로도 $-0.0289$ 로도 적을 수 있고, 인구를 천 명 단위로 재면 $-2.89\times10^{-5}$ 가 된다. **단위를 고르는 사람이 계수의 크기를 마음대로 정할 수 있다.** 그런데도 모형의 예측은 한 치도 달라지지 않는다. 그러므로 계수의 크기를 변수들 사이에서 견주는 것은 애초에 뜻이 없는 일이고, 본문의 왼쪽 그림이 보여 준 "$10$ 억 배 차이"는 자료가 아니라 단위의 그림자다. 견줄 수 있는 것은 단위에 매이지 않는 양, 곧 $t$ 값과 $1$ 표준편차 효과뿐이다. $\square$

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span> 보기 4 에서 검정 오차의 평균이 $\bar e = +4.9279$ 였고 분해가 $\mathrm{RMSE}^2 = 106.4949 = 24.2847 + 82.2102$ 였다. 모형이 한쪽으로 치우쳤으니 예측값 전부에 상수 $c$ 를 더해 $\hat y_i + c$ 로 바꾸면 더 나아 보일 것이다.

**(1)** 새 RMSE 를 $c$ 의 함수로 적고, 그것을 최소로 하는 $c^\ast$ 와 그때의 RMSE 값을 보기 4 의 수치로 계산하시오.

**(2)** 그 값을 "이 모형의 검정 성능"으로 보고해도 되는가. 되지 않는다면, 같은 치우침 보정을 **정직하게** 하려면 어떤 자료로 $c$ 를 정해야 하고 그때 $c$ 는 얼마가 되는가. 또한 참 치우침이 $0$ 인 모형에서도 이 장난이 RMSE 를 평균 몇 퍼센트 낮추는지 $n_{\text{test}}$ 로 적으시오.

</div>

??? success "풀이"

    **(1) 보기 4 의 분해를 $c$ 에 대해 다시 하면 된다.** 오차가 $e_i - c$ 로 바뀌므로

    $$
    \mathrm{MSE}(c) = \frac{1}{n}\sum_i (e_i - c)^2 = (c - \bar e)^2 + \frac{1}{n}\sum_i (e_i - \bar e)^2
    $$

    이다. 뒷항은 $c$ 와 무관하므로 $c$ 에 대한 완전제곱식 하나만 남고, 최소는 $c^\ast = \bar e = +4.9279$ 에서 난다. 그때 남는 것은 보기 4 의 퍼짐 항뿐이다.

    $$
    \mathrm{RMSE}(c^\ast) = \sqrt{\frac{n-1}{n}s_e^2} = \sqrt{82.2102} = 9.0670
    $$

    원래 $10.3196$ 에서 $9.0670$ 으로 $12.1\%$ 내려간다. 제곱으로 재면 $106.4949 \to 82.2102$ 로 $22.8\%$ 다. 치우침 항 $24.2847$ 을 통째로 지운 것이니 당연하다.

    **(2) 보고할 수 없다.** $c^\ast$ 를 **검정자료의 $y$ 로 정했기** 때문이다. 그 순간 여덟 주는 더 이상 남겨 둔 자료가 아니고 $9.0670$ 은 검정오차가 아니라 훈련오차다. 실제로 $c$ 를 더하는 것은 절편을 $\hat\beta_0 + c$ 로 바꾸는 것이고, **절편 하나를 검정자료에 맞춰 적합한 것**과 똑같다.

    정직하게 하려면 $c$ 를 **훈련자료**로 정해야 한다. 그런데 훈련자료에서는 최소제곱의 정규방정식이 $\sum_i e_i = 0$ 을 강제하므로(보기 4 의 (1))

    $$
    c_{\text{정직}} = \bar e_{\text{훈련}} = 0
    $$

    이다. **아무 일도 할 수 없다.** 치우침 보정은 훈련자료에서는 할 것이 없고 검정자료에서는 해서는 안 되는 일이다.

    참 치우침이 $0$ 인 모형에서도 이 장난은 공짜로 이득을 준다. $\operatorname{E}[\bar e^2] = \sigma_e^2/n_{\text{test}}$ 이므로

    $$
    \frac{\operatorname{E}[\mathrm{MSE}(c^\ast)]}{\operatorname{E}[\mathrm{MSE}(0)]} = \frac{\sigma_e^2 - \sigma_e^2/n_{\text{test}}}{\sigma_e^2} = 1 - \frac{1}{n_{\text{test}}}
    $$

    이고 $n_{\text{test}} = 8$ 이면 MSE 가 평균 $12.5\%$, RMSE 가 $1 - \sqrt{0.875} = 6.5\%$ 내려간다. 자유모수 하나를 $8$ 개 점에 맞췄으니 그만큼 돌려받는 것이다.

    여기서 보기 4 의 치우침을 다시 읽을 수 있다. 관측된 MSE 감소 $22.8\%$ 는 공짜 몫 $12.5\%$ 보다 크지만 두 배가 안 되고, 보기 4 가 이미 $t = 1.44$ 로 유의하지 않다고 말했다. **치우침이 참으로 있다고 볼 근거가 없고, 그것을 고쳐 얻은 $9.07$ 은 성능이 아니라 $8$ 개에 모수 하나를 더 맞춘 대가다.** $\square$

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span> 보기 2 는 "참 모형을 정확히 적합해도 예측오차를 $12$ 아래로 내릴 수 없다"고 했다. 그러나 적합은 참 계수가 아니라 $42$ 개 주로 **추정한** 계수로 한다.

**(1)** 설계행렬 $X$ 는 고정, $\hat\beta$ 는 그 자료의 OLS 추정량이라 하고, 훈련에 쓰이지 않은 새 점 $(x_0, y_0)$ 를 생각하자. 모형이 옳을 때

$$
\operatorname{E}\!\left[(y_0 - x_0^\top\hat\beta)^2\right] = \sigma^2\left(1 + x_0^\top (X^\top X)^{-1} x_0\right)
$$

임을 보이시오.

**(2)** 훈련자료의 평균 지렛값이 언제나 정확히 $p/n$ 임을 쓰고, 검정점의 평균 지렛값을 그 값으로 어림하여 세 변수 모형의 검정 RMSE 가 겨냥하는 값을 $n = 42$, $p = 4$ 로 계산하시오. 실제로 여덟 검정 주의 평균 지렛값은 $0.1371$ 이다. 이 값을 쓰면 얼마가 되며, 관측값 $10.32$ 와 어긋나는가. 연습문제 2 에서 쓸모없는 변수 둘을 더 넣으면 이 값은 얼마가 되는가.

</div>

??? success "풀이"

    **(1)** 참 모형은 $y_0 = x_0^\top\beta + \varepsilon_0$ 이고 $\hat\beta = \beta + (X^\top X)^{-1}X^\top\varepsilon$ 이다. 여기서 $\varepsilon$ 은 훈련자료의 오차벡터이고 $\varepsilon_0$ 은 새 점의 오차다. 빼면

    $$
    y_0 - x_0^\top\hat\beta = \varepsilon_0 - x_0^\top (X^\top X)^{-1} X^\top \varepsilon
    $$

    이다. $x_0$ 가 훈련에 쓰이지 않았으므로 $\varepsilon_0$ 과 $\varepsilon$ 은 독립이고 둘 다 평균 $0$ 이다. 그러므로 분산이 더해진다. 뒷항의 분산은 $a = (X^\top X)^{-1}x_0$ 로 두면

    $$
    \operatorname{Var}\!\left(x_0^\top (X^\top X)^{-1} X^\top \varepsilon\right) = \sigma^2 a^\top X^\top X a = \sigma^2 x_0^\top (X^\top X)^{-1} x_0
    $$

    이다. 합하면 $\sigma^2 + \sigma^2 x_0^\top (X^\top X)^{-1}x_0$ 가 된다. $\square$

    **(2) 앞항 $\sigma^2$ 는 줄일 수 없는 오차, 뒷항은 추정의 대가다.** 훈련점의 지렛값에 대해서는 모자행렬 $H = X(X^\top X)^{-1}X^\top$ 의 대각합이 $\operatorname{tr}(H) = \operatorname{tr}\!\left((X^\top X)^{-1}X^\top X\right) = p$ 이므로 평균이 정확히 $p/n$ 이다. 검정점도 비슷한 자리에 있다고 보면 겨냥하는 값은

    $$
    \sigma\sqrt{1 + p/n} = 12\sqrt{1 + 4/42} = 12\sqrt{1.09524} = 12.5584
    $$

    이다. **$12$ 가 아니다.** 보기 2 의 $12$ 는 참 계수를 공짜로 안다고 할 때의 바닥이고, $42$ 개로 네 모수를 추정하는 대가가 $4.6\%$ 붙는다.

    실제 검정 주의 평균 지렛값 $0.1371$ 은 $p/n = 0.0952$ 보다 $44\%$ 크다. 이것이 요점이다. **검정점은 적합이 맞춰 준 점이 아니므로 평균적으로 훈련점보다 지렛값이 크다.** 이 값을 쓰면

    $$
    12\sqrt{1 + 0.1371} = 12.7962
    $$

    가 된다. 관측된 $10.32$ 는 이보다 $19.4\%$ 낮다. 어긋나는가. 보기 2 가 쓴 어림으로 $n_{\text{test}} = 8$ 에서 RMSE 의 상대 표준오차가 $25\%$ 이므로 **$19.4\%$ 는 그 안쪽이고 어긋남이 아니다.** 운 좋은 여덟 주였을 뿐이다.

    연습문제 2 처럼 `percent_black` 과 `percent_hispanic` 을 더 넣으면 $p = 6$ 이 되어

    $$
    12\sqrt{1 + 6/42} = 12.8285
    $$

    이다. 두 변수의 참 계수가 **정확히 $0$ 이어도** 바닥이 $12.5584$ 에서 $12.8285$ 로 올라간다. 연습문제 2 가 말로 한 "편향은 줄이지 못한 채 분산만 키운다"의 수치가 이것이다. 쓸모없는 변수 하나의 값은 $\sigma^2/n$, 곧 이 자료에서 예측오차 분산의 $1/42 = 2.4\%$ 씩이다. $\square$

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span> 보기 3 은 검정 RMSE 가 $12.66$ 에서 $10.32$ 로 줄어든 것을 "유의하다고 말할 수 없다"고 했다. 이 말을 수로 확인한다. $50$ 개 주의 설명변수는 그대로 두고 오차 $\varepsilon \sim N(0, 12^2)$ 만 $4000$ 번 다시 뽑아, 그때마다 같은 $42/8$ 분할로 (ㄱ) 인구만 쓴 모형과 (ㄴ) 세 변수 모형을 적합해 검정 RMSE 를 재라. 씨앗은 `np.random.default_rng(0)` 을 쓴다.

**(1)** 두 RMSE 의 제곱평균제곱근을 연습문제 8 의 공식값과 견주고, 상대 표준편차가 보기 2 의 어림 $1/\sqrt{2n_{\text{test}}} = 0.25$ 와 맞는지 보시오. 인구만 쓴 모형에서는 치우침 항이 더해지므로 공식을 그에 맞게 고쳐야 한다.

**(2)** 세 변수 모형이 인구만 쓴 모형보다 **더 나쁘게** 나오는 되풀이의 비율을 구하고, 관측된 비 $10.3194/12.6631 = 0.8149$ 가 그 분포에서 어디에 놓이는지 말하시오.

</div>

??? success "풀이"

    **설계를 고정하고 오차만 다시 뽑는다.** 그러면 인구만 쓴 모형이 빠뜨린 소득·백인 비율의 몫은 잡음이 아니라 **고정된 치우침**이 되고, 연습문제 8 의 공식에 그 항을 더해 주면 된다. 검정점 $i$ 의 치우침은 참 평균 $\mu$ 를 훈련 열공간에 사영한 것과의 차이다.

    $$
    \operatorname{E}[\text{검정 MSE}] = \sigma^2\left(1 + \bar h\right) + \frac{1}{n_{\text{test}}}\sum_{i \in \text{test}} b_i^2, \qquad b_i = \mu_i - x_i^\top (X^\top X)^{-1}X^\top \mu_{\text{train}}
    $$

    ```python
    # 참 모형에서 오차만 다시 뽑아 같은 42/8 분할로 두 모형을 되풀이 적합한다.
    rng = np.random.default_rng(0)
    cols = ["total_population", "per_capita_income", "percent_white"]
    Xall = df[cols].to_numpy()
    is_test = df["state"].isin(test_states).to_numpy()
    mu = 190.24 - 1.2e-6 * Xall[:, 0] + 0.0009 * Xall[:, 1] + 18 * Xall[:, 2]


    def design(which):
        return np.column_stack([np.ones(len(Xall)), Xall[:, which]])


    def test_rmse(y, A):
        """A 를 설계행렬로 42 개 주로 적합하고 8 개 주에서 RMSE 를 잰다."""
        beta = np.linalg.lstsq(A[~is_test], y[~is_test], rcond=None)[0]
        return np.sqrt(np.mean((y[is_test] - A[is_test] @ beta) ** 2))


    A1, A3 = design([0]), design([0, 1, 2])
    B = 4000
    r_uni, r_tri = np.empty(B), np.empty(B)
    for b in range(B):
        y = mu + rng.normal(0, 12, len(Xall))
        r_uni[b] = test_rmse(y, A1)
        r_tri[b] = test_rmse(y, A3)


    def exact(A):
        """고정 설계에서 E[검정 MSE] = sigma^2 (1 + h) + 치우침^2 을 그대로 계산한다."""
        M = np.linalg.inv(A[~is_test].T @ A[~is_test])
        h = np.einsum("ij,jk,ik->i", A[is_test], M, A[is_test]).mean()
        bias = mu[is_test] - A[is_test] @ (M @ A[~is_test].T @ mu[~is_test])
        return h, np.sqrt(np.mean(bias ** 2)), np.sqrt(144 * (1 + h) + np.mean(bias ** 2))


    print(f"{'모형':10s}{'RMSE 평균':>11s}{'제곱평균제곱근':>14s}{'상대 표준편차':>14s}"
          f"{'평균 지렛값':>12s}{'치우침':>9s}{'이론값':>9s}")
    for name, r, A in [("인구만", r_uni, A1), ("세 변수", r_tri, A3)]:
        h, bias, exact_rms = exact(A)
        print(f"{name:10s}{r.mean():11.4f}{np.sqrt(np.mean(r ** 2)):14.4f}"
              f"{r.std(ddof=1) / r.mean():14.4f}{h:12.4f}{bias:9.4f}{exact_rms:9.4f}")
    print(f"어림한 상대 표준오차 1/sqrt(2n) = {1 / np.sqrt(2 * 8):.4f}")
    print()
    print(f"세 변수 쪽이 더 나쁜 비율 P(RMSE_3 > RMSE_1) = {np.mean(r_tri > r_uni):.4f}")
    print(f"비 RMSE_3/RMSE_1 의 중앙값 = {np.median(r_tri / r_uni):.4f}"
          f"   5% 와 95% 분위 = {np.percentile(r_tri / r_uni, [5, 95]).round(4)}")
    print(f"관측된 비 {rmse3 / rmse1:.4f} 의 분위 = {np.mean(r_tri / r_uni <= rmse3 / rmse1):.4f}")
    ```

    출력:

    ```text
    모형            RMSE 평균       제곱평균제곱근       상대 표준편차      평균 지렛값      치우침      이론값
    인구만           14.8740       15.2928        0.2390      0.0557   9.1442  15.3506
    세 변수          12.3261       12.7282        0.2576      0.1371   0.0000  12.7960
    어림한 상대 표준오차 1/sqrt(2n) = 0.2500

    세 변수 쪽이 더 나쁜 비율 P(RMSE_3 > RMSE_1) = 0.1710
    비 RMSE_3/RMSE_1 의 중앙값 = 0.8190   5% 와 95% 분위 = [0.5766 1.2128]
    관측된 비 0.8149 의 분위 = 0.4913
    ```

    **(1) 공식과 모의가 맞는다.** 세 변수 모형은 치우침이 $0$(참 모형이므로)이고 제곱평균제곱근 $12.7282$ 가 연습문제 8 의 $12.7960$ 과 $0.5\%$ 안에서 일치한다. 평균 지렛값도 연습문제 8 에 적은 $0.1371$ 이 그대로 나온다. 인구만 쓴 모형은 치우침 $9.1442$ 가 붙어 $\sqrt{144 \times 1.0557 + 9.1442^2} = 15.3506$ 이 되고 모의가 $15.2928$ 을 준다.

    보기 2 의 이론값 $14.0124$ 가 이보다 작은 데에는 두 까닭이 있다. 추정의 대가 $\sqrt{1.0557}$ 이 $2.7\%$ 를 보태고, 더 큰 몫은 **이 여덟 주가 빠뜨린 신호가 큰 주들**이라는 데서 온다. 표본 전체로 재면 빠뜨린 신호의 크기가 $\sqrt{14.0124^2 - 144} = 7.2352$ 인데 검정 여덟 주에서는 $9.1442$ 다. 검정 집합을 고정해 놓고 보면 이런 치우침은 지워지지 않는다.

    상대 표준편차는 $0.2390$ 과 $0.2576$ 으로 어림 $0.25$ 를 잘 맞춘다. **$1/\sqrt{2n}$ 은 이 쪽에서 세 번 쓰였고 실제로 믿을 만하다.** 참고로 RMSE 의 평균은 제곱평균제곱근보다 작다($12.3261 < 12.7282$). 제곱근이 오목함수라서 생기는 차이이므로, 공식과 견줄 때는 제곱평균제곱근을 써야 한다.

    **(2) 세 변수 모형이 지는 일이 $17.1\%$ 나 일어난다.** 참 모형을 적합하는 쪽이 여섯 번에 한 번은 쓸모없는 단변량 모형보다 나쁘게 보인다는 뜻이다. 비의 $5\%$–$95\%$ 범위가 $[0.577,\ 1.213]$ 로 $1$ 을 한참 품고 있다.

    그리고 관측된 비 $0.8149$ 는 이 분포의 **$49$ 분위**, 곧 중앙값 $0.8190$ 바로 옆이다. 두 가지를 한꺼번에 말해 준다. 첫째, 보기 3 에서 본 개선은 **전혀 이상하지 않은, 가장 흔한 크기**다. 둘째, 그렇게 전형적인 개선조차 $8$ 개로는 선언할 수 없다. 같은 세계를 다시 살면 $17\%$ 의 확률로 반대 결론을 적었을 것이다. **검정 집합의 크기가 모형 비교의 해상도를 정한다.** $\square$

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span> 보기 3 의 풀이는 "실제로 $1 - 12.9413^2/13.5481^2 = 0.0876$ 인데 $R^2$ 정의의 분모가 자유도 보정이 없는 쪽이라 $0.1543$ 이 나온다"고 적었다. 이 $0.0876$ 의 정체를 밝힌다.

**(1)** $s^2 = \mathrm{RSS}/(n-p)$ 와 $s_y^2 = \mathrm{TSS}/(n-1)$ 에 대해

$$
1 - \frac{s^2}{s_y^2} = 1 - (1 - R^2)\frac{n-1}{n-p}
$$

임을 보이시오. 곧 보기 3 이 손으로 계산한 수는 **수정 결정계수**다. $n = 42$, $p = 4$, $R^2 = 0.1543$ 으로 두 식이 모두 $0.0876$ 을 주는지 확인하시오.

**(2)** 보기 3 은 훈련 $R^2 = 0.1543$ 을 모집단 양 $1 - \sigma^2/\operatorname{Var}(\text{HighQ}) = 0.3274$ 와 견주었다. $R^2$ 와 $R^2_{\text{adj}}$ 가운데 어느 쪽이 그 모집단 양을 겨냥하는가. 그 답을 쓰면 어긋남은 커지는가 작아지는가. 어긋남의 방향을 보기 2 의 어느 두 수치로 설명할 수 있는지 밝히고, 두 수치가 각각 몇 퍼센트 기여하는지 적으시오.

</div>

??? success "풀이"

    **(1)** $R^2 = 1 - \mathrm{RSS}/\mathrm{TSS}$ 이므로 $\mathrm{RSS}/\mathrm{TSS} = 1 - R^2$ 이다. 그대로 대입한다.

    $$
    1 - \frac{s^2}{s_y^2} = 1 - \frac{\mathrm{RSS}/(n-p)}{\mathrm{TSS}/(n-1)} = 1 - \frac{\mathrm{RSS}}{\mathrm{TSS}}\cdot\frac{n-1}{n-p} = 1 - (1 - R^2)\frac{n-1}{n-p}
    $$

    오른쪽이 수정 결정계수의 정의다. $\square$

    수치로 맞춰 본다. 오른쪽은

    $$
    1 - (1 - 0.1543)\cdot\frac{41}{38} = 1 - 0.8457 \times 1.078947 = 1 - 0.9124 = 0.0876
    $$

    이고 왼쪽은 $1 - 12.9413^2/13.5481^2 = 1 - 167.4738/183.5513 = 0.0876$ 이다. 같다. 그러므로 **보기 3 의 $0.0876$ 은 $R^2$ 를 잘못 계산한 수가 아니라 다른 통계량**이며, `statsmodels` 가 `rsquared_adj` 로 내주는 바로 그 값이다.

    **(2) 모집단 양을 겨냥하는 것은 $R^2_{\text{adj}}$ 다.** 겨냥하는 수가

    $$
    R^2_{\text{모집단}} = 1 - \frac{\sigma^2}{\operatorname{Var}(\text{HighQ})}
    $$

    처럼 두 **모분산의 비**로 적혀 있으니, 분자와 분모를 각각 불편추정량으로 바꾼 $1 - s^2/s_y^2$ 가 자연스러운 추정량이다. 반면 $R^2 = 1 - \mathrm{RSS}/\mathrm{TSS}$ 는 분자에서 자유도를 $p$ 만큼 깎지 않으므로 **위로 치우친다.** 그러므로 견주어야 할 짝은 $0.0876$ 대 $0.3274$ 이고, **어긋남은 $0.1543$ 대 $0.3274$ 보다 더 커진다.**

    방향은 보기 2 의 두 수치로 완전히 설명된다. 비 $s^2/s_y^2$ 가 겨냥하는 수는 $\sigma^2/\operatorname{Var}(\text{HighQ}) = 144/14.6316^2 = 0.6726$ 인데 실제로는 $167.4738/183.5513 = 0.9124$ 로 $35.7\%$ 크다. 두 쪽이 같은 방향으로 밀었기 때문이다.

    - **분자가 크게 나왔다.** $s = 12.9413$ 이 참 $\sigma = 12$ 보다 $7.8\%$ 크므로 분산으로는 $12.9413^2/144 = 1.1630$, 곧 $16.3\%$ 크다.
    - **분모가 작게 나왔다.** 관측 $s_y = 13.5481$ 이 이론 $14.6316$ 보다 $7.4\%$ 작으므로 분산으로는 $183.5513/214.0837 = 0.8574$, 곧 $14.3\%$ 작다.

    곱하면 $1.1630/0.8574 = 1.3565$ 로 관측된 $35.7\%$ 가 그대로 나온다. 두 어긋남은 각각 $11\%$ 남짓의 상대 표준오차 안쪽이라 따로 보면 아무 문제가 없는데(보기 2, 보기 3), **비를 만들면 어긋남이 합쳐져 $36\%$ 가 된다.**

    여기서 얻을 것은 이렇다. $R^2$ 는 두 분산의 비이므로 **분자와 분모의 표본변동이 모두 실려 들어온다.** $n = 42$ 에서 참값 $0.33$ 을 두고 $0.09$ 라는 점추정을 얻는 일이 아무렇지 않게 일어난다. 그러므로 작은 표본의 $R^2$ 를 소수 넷째 자리까지 적고 모형의 설명력을 논하는 것은 있는 정보보다 많은 것을 말하는 일이다. $\square$

---

## 정리하며

실제 자료로 **탐색부터 평가까지** 밟았다.

- **상관행렬을 먼저 본다.** 설명변수끼리 강하게 상관되어 있으면 다중공선성을 예상해야 하며, 12장의 열지도가 그 도구다.
- **단변량과 다변량 결과가 다르다.** 한 변수만 넣었을 때 유의하던 것이 다른 변수를 넣으면 사라지거나 부호가 바뀔 수 있다. **교란을 통제한 결과이기도 하고 공선성 때문이기도 하므로** 어느 쪽인지 판단해야 한다.
- **`sklearn` 과 `statsmodels` 의 역할이 다르다.** 앞의 것은 예측 성능에, 뒤의 것은 계수의 추론에 맞춰져 있다. 1장의 예측 대 추론 구분이 도구 선택으로 나타난 것이다.
- **남겨 둔 검정자료에서 평가한다.** 훈련자료의 $R^2$ 은 언제나 낙관적이다.
- **주 단위 자료라 생태학적 오류에 주의해야 한다.** 주 평균끼리의 관계가 개인 수준에서도 성립한다는 보장이 없다(12장).

다음 절부터 **추정과 추론**으로 넘어간다.
