# Patsy를 이용한 스플라인

## 개요

이 페이지는 patsy의 식 인터페이스로 B-스플라인과 자연 스플라인 기저행렬을 만들어 스플라인 회귀를 수행하는 방법을 보인다. King County 주택 자료를 써서 B-스플라인, 자유도를 달리한 자연 스플라인, 사용자 지정 매듭 배치를 비교하고, 이 유연한 방법들을 선형·다항 회귀와 대조한다.

## 수학적 배경

### B-스플라인(기저 스플라인)

매듭이 $\xi_1 < \cdots < \xi_K$인 $d$차 B-스플라인은 각 매듭에서 $(d-1)$번 연속미분 가능한 조각별 다항식이다. 회귀모형은

$$
f(x) = \sum_{m=1}^{K+d+1} \gamma_m B_{m,d}(x),
$$

여기서 $B_{m,d}$는 B-스플라인 기저함수이다. 기저함수의 개수는 $K + d + 1$이다(내부 매듭 + 차수 + 1).

### 자연 스플라인

자연 삼차 스플라인은 경계 매듭 바깥에서 $f(x)$가 선형이라는 제약을 추가한다. 이 제약이 자유도를 4만큼 줄이고(각 경계에서 $f'' = 0$과 $f''' = 0$, 두 경계에 각각 두 개) 외삽 거동을 훨씬 안정적으로 만든다.

### 자유도

스플라인의 자유도(df)가 유연성을 조절한다.

- B-스플라인: $\mathrm{df} = K + d + 1 - 1$ (기저함수 개수에서 절편을 뺀 것)
- df가 클수록 더 요동치는 적합이 가능하다
- 최적 df는 교차검증으로 고를 수 있다

### Patsy 문법

- `bs(x, df=4, degree=3)`: 자유도 4의 삼차 B-스플라인
- `bs(x, knots=[20, 40, 60])`: 내부 매듭을 지정한 B-스플라인
- `cr(x, df=4)`: 자유도 4의 자연 삼차 회귀 스플라인

## 자료

세 페이지가 공유하는 King County(시애틀) 주택 매매 자료를 읽는다.

```python
import pandas as pd

# "Practical Statistics for Data Scientists" 저장소의 자료. 탭으로 구분되어 있다.
url = ("https://raw.githubusercontent.com/gedeck/"
       "practical-statistics-for-data-scientists/master/data/house_sales.csv")
house = pd.read_csv(url, sep='\t')

print(f"{len(house)}건, 열 {house.shape[1]}개")
print(house[['AdjSalePrice', 'SqFtTotLiving', 'YrBuilt']].describe().round(1).to_string())
```

출력:

```
22687건, 열 22개
       AdjSalePrice  SqFtTotLiving  YrBuilt
count       22687.0        22687.0  22687.0
mean       565233.3         2080.2   1971.2
std        385402.9          913.7     30.3
min          3368.0          370.0   1900.0
25%        360563.0         1420.0   1950.0
50%        471315.0         1910.0   1977.0
75%        649411.0         2540.0   2000.0
max      11644855.0        10740.0   2015.0
```

주택 22,687건이다. 건축연도가 1900년부터 2015년까지 걸쳐 있어 스플라인으로 나이-가격 관계를 살피기에 적당하다.

## 코드

### B-스플라인 회귀

```python
import numpy as np
import pandas as pd
from patsy import dmatrix, build_design_matrices
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

age = 2024 - house['YrBuilt'].values
df = pd.DataFrame({'age': age, 'price': house['AdjSalePrice'].values})

# B-spline with df=4
bs_design = dmatrix("bs(age, df=4, degree=3, include_intercept=False) - 1",
                    {"age": df['age']}, return_type='dataframe')
bs_model = LinearRegression().fit(bs_design, df['price'])
bs_r2 = r2_score(df['price'], bs_model.predict(bs_design))
```

### 사용자 지정 매듭을 쓰는 B-스플라인

```python
knots_custom = [20, 40, 60]
bs_custom_design = dmatrix(
    f"bs(age, knots={knots_custom}, degree=3, include_intercept=False) - 1",
    {"age": df['age']}, return_type='dataframe'
)
bs_custom_model = LinearRegression().fit(bs_custom_design, df['price'])
```

### 자연 스플라인

```python
cs_design = dmatrix("cr(age, df=4) - 1",
                    {"age": df['age']}, return_type='dataframe')
cs_model = LinearRegression().fit(cs_design, df['price'])
cs_r2 = r2_score(df['price'], cs_model.predict(cs_design))
```

### 격자에서의 예측

```python
age_grid = np.linspace(df['age'].min(), df['age'].max(), 300)

# Reuse the ORIGINAL basis (same knots) via design_info
bs_grid = build_design_matrices([bs_design.design_info], {"age": age_grid})[0]
bs_pred = bs_model.predict(np.asarray(bs_grid))

cs_grid = build_design_matrices([cs_design.design_info], {"age": age_grid})[0]
cs_pred = cs_model.predict(np.asarray(cs_grid))
```

!!! warning "새 자료에 `dmatrix`를 다시 부르면 안 된다"
    식 문자열로 `dmatrix`를 다시 호출하면 patsy가 **새로 넘긴 자료로 매듭을 다시 계산한다**. 그러면 훈련에 쓴 기저와 다른 기저가 만들어져 계수가 엉뚱한 기저에 곱해진다. 반드시 원래 설계행렬의 `design_info`를 `build_design_matrices`에 넘겨 같은 매듭을 재사용해야 한다.

## 해석

- **B-스플라인**은 국소 제어를 제공한다. 각 기저함수가 좁은 구간에서만 0이 아니므로 한 구역의 적합이 멀리 떨어진 자료와 비교적 무관하다. 그래서 수치적으로 안정적이고 해석하기 좋다.
- **자연 스플라인**은 극단 매듭 바깥에서 적합을 선형으로 제약하므로 경계에서 더 안정적이다. 제약 없는 삼차 스플라인이 보이는 격렬한 외삽을 피한다.
- **자유도**: df를 키우면 훈련자료 적합은 좋아지지만 과대적합 위험이 커진다. 최적 df는 교차검증이나 정보기준으로 고른다.
- **매듭 배치**: 자동 배치(균등 분위수)가 대부분의 경우 잘 작동한다. 관계의 성격이 바뀌는 지점을 분야 지식으로 안다면 사용자 지정 매듭이 유용하다.
- **비교**: 스플라인은 하나의 전역적 모양을 강제하는 대신 국소적 유연성을 제공하므로, 복잡도가 같은 다항식보다 대체로 낫다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span> df를 3에서 8까지 바꿔 가며 B-스플라인을 적합하고 그 결과 곡선을 그려라. 어느 df에서 과대적합의 징후가 나타나기 시작하는가?

</div>

??? success "풀이"

    삼차 B-스플라인에서는 `df`가 `degree`보다 작을 수 없다. `df=2, degree=3`을 넣으면
    patsy가 `ValueError: df=2 is too small for degree=3`을 던지므로 3부터 시작한다.

    ```python
    for d in range(3, 9):
        design = dmatrix(f"bs(age, df={d}, degree=3, include_intercept=False) - 1",
                         {"age": df['age']}, return_type='dataframe')
        model = LinearRegression().fit(design, df['price'])
        grid_design = build_design_matrices([design.design_info],
                                           {"age": age_grid})[0]
        print(f"df={d}: R^2 = {model.score(design, df['price']):.4f}")
    ```

    출력:

    ```
    df=3: R^2 = 0.0292
    df=4: R^2 = 0.0346
    df=5: R^2 = 0.0351
    df=6: R^2 = 0.0349
    df=7: R^2 = 0.0353
    df=8: R^2 = 0.0354
    ```

    $R^2$가 df=4에서 0.0346까지 오른 뒤로는 거의 늘지 않는다(df=6에서는 오히려 미세하게 줄었는데, 매듭 위치가 달라지면서 생기는 일이다). 훈련 자료에 대한 적합이므로 이 값만으로는 과대적합을 판단할 수 없다.

    과대적합은 곡선의 모양에서 드러난다. 지나친 요동으로 나타나며, 특히 자료가 성긴 경계 근처에서 두드러진다. 이 자료에서는 대체로 df > 6에서 징후가 보이기 시작한다. 형식적으로 고르려면 교차검증을 써야 한다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 매듭이 $K$개인 자연 삼차 스플라인의 자유도가 (제약 없는 삼차 스플라인의 $K + 4$가 아니라) $K$인 이유를 설명하라. "잃어버린" 4개의 자유도는 어디로 가는가?

</div>

??? success "풀이"

    구간 $[a, b]$에 매듭 $\xi_1 < \cdots < \xi_K$가 있을 때(양 끝의 경계 매듭을 포함해 센다) 제약 없는 삼차 스플라인의 차원을 세어 보자. 매듭이 구간을 $K+1$개 조각으로 나누고 조각마다 삼차 다항식이므로 계수는 $4(K+1)$개이다. 각 내부 매듭에서 $f$, $f'$, $f''$의 연속성이 3개의 제약을 주므로 $3(K-2)$… 를 세는 대신, 널리 쓰이는 결과를 그대로 쓰면 **삼차 스플라인의 차원은 $K + 4$**이다.

    자연 스플라인은 여기에 경계 바깥에서 선형이라는 조건을 얹는다. 이는 양쪽 끝 각각에서

    $$
    f''(\xi_1) = 0, \quad f'''(\xi_1) = 0, \qquad f''(\xi_K) = 0, \quad f'''(\xi_K) = 0
    $$

    이라는 **4개의 제약**을 뜻한다. 따라서 차원은

    $$
    (K + 4) - 4 = K
    $$

    가 된다. 잃어버린 4개의 자유도는 양쪽 바깥 조각의 이차항과 삼차항 계수 네 개이며, 이들이 0으로 강제되어 그 구간에서 함수가 직선이 된다.

    !!! note "df 세는 방식의 차이"
        절편을 df에 포함하느냐에 따라 문헌마다 숫자가 하나씩 달라진다. patsy의 `cr(x, df=4)`나 R의 `ns(x, df=4)`는 절편을 뺀 기저 열 4개를 돌려주므로, 모형 절편을 더하면 전체 차원이 5가 된다. 어떤 규약을 쓰든 **자연 스플라인이 같은 매듭의 삼차 스플라인보다 정확히 4만큼 작다**는 점은 변하지 않는다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> 자료 범위 바깥(예: age = 150이나 age = 0)에서 B-스플라인 모형과 자연 스플라인 모형의 예측을 비교하라. 어느 쪽이 더 합리적으로 외삽하는가?

</div>

??? success "풀이"

    ```python
    age_extrap = np.array([0.0, 5.0, 145.0, 150.0])

    # 자연 스플라인: 선형으로 외삽한다.
    cs_extrap = build_design_matrices([cs_design.design_info],
                                      {"age": age_extrap})[0]
    print("Natural:", cs_model.predict(np.asarray(cs_extrap)).round(0))

    # B-스플라인: 매듭 바깥에서는 예외를 던진다.
    try:
        bs_extrap = build_design_matrices([bs_design.design_info],
                                          {"age": age_extrap})[0]
        print("B-spline:", bs_model.predict(np.asarray(bs_extrap)).round(0))
    except Exception as exc:
        print(f"B-spline: {type(exc).__name__}")
    ```

    출력:

    ```
    Natural: [772512. 741152. 775648. 801248.]
    B-spline: PatsyError
    ```

    자연 스플라인은 문제없이 계산되며 경계에서의 기울기를 그대로 이어 **선형으로** 외삽한다.

    B-스플라인 쪽은 값을 돌려주는 대신 **예외를 던진다**.

    ```text
    PatsyError: Error evaluating factor: NotImplementedError:
    some data points fall outside the outermost knots,
    and I'm not sure how to handle them.
    ```

    이 오류 자체가 답이다. B-스플라인 기저는 최외곽 매듭 바깥에서 정의되지 않으므로 patsy는 추측해서 값을 만들어 내는 대신 거부한다. 설령 삼차식을 그대로 연장하도록 구현했더라도 그 외삽값은 극단적이고 비현실적이었을 것이다. 자료 범위를 벗어나 예측해야 한다면 자연 스플라인을 써야 한다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> 적합 목적함수에 릿지 형태의 항 $\lambda \|\boldsymbol{\gamma}\|^2$을 더해 B-스플라인에 거칢 벌점을 구현하라. 이것은 평활 스플라인과 어떤 관계인가?

</div>

??? success "풀이"

    ```python
    from sklearn.linear_model import Ridge

    design = dmatrix("bs(age, df=8, degree=3, include_intercept=False) - 1",
                     {"age": df['age']}, return_type='dataframe')
    ridge_model = Ridge(alpha=1e4).fit(design, df['price'])
    ```

    $\lambda\|\boldsymbol{\gamma}\|^2$을 더하면 기저 계수가 0 쪽으로 축소되어 곡선이 매끄러워진다. 이는 2계 도함수에 벌점을 주는 평활 스플라인의 벌점 $\lambda\int [f'']^2$에 대한 근사이다. 정확한 평활 스플라인은 $D_{jk} = \int B_j''(t)B_k''(t)\,dt$인 벌점행렬 $\mathbf{D}$를 쓰지만, $\lambda\mathbf{I}$(릿지)가 더 간단한 근사를 제공한다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span> 매듭이 자료점에 놓인 자연 삼차 스플라인에 대해, 평활 스플라인 추정량이 두 번 미분 가능한 모든 함수 가운데 $\sum(y_i - f(x_i))^2 + \lambda\int[f''(t)]^2\,dt$를 최소화함을 증명하라.

</div>

??? success "풀이"

    비모수 회귀의 고전적 결과이다. 핵심은 매듭에서 주어진 값을 보간하는 모든 함수 $g$ 가운데 자연 삼차 스플라인이 $\int [g'']^2\,dt$(거칢)를 최소화한다는 것이다. 변분법적 논증으로 보인다. $f$를 자연 삼차 스플라인이라 하고 $g = f + h$로 쓰면

    $$
    \int [g'']^2 = \int [f'']^2 + 2\int f'' h'' + \int [h'']^2.
    $$

    부분적분과 자연 스플라인의 경계조건에 의해 교차항 $\int f'' h'' = 0$이므로 $\int[g'']^2 \geq \int[f'']^2$이다. 따라서 자연 삼차 스플라인이 임의의 $\lambda > 0$에 대해 벌점 목적함수의 유일한 최소화 함수이다. $\square$

---

## 정리하며

스플라인은 **조각별 다항을 매끄럽게 이어 붙인다.**

- **B-스플라인이 계산에 안정적인 기저다.** 매듭 사이에서 3차 다항이고 매듭에서 2계 도함수까지 연속이므로, 계단함수의 불연속 문제가 사라진다.
- **자연 스플라인은 양 끝에서 선형으로 제약한다.** 자료가 드문 경계에서 3차 다항이 폭주하는 것을 막으며, **외삽이 조금 덜 위험해진다.**
- **매듭의 개수가 유연성을 정한다.** 위치보다 개수가 중요하며, 보통 분위수에 균등 배치하고 자유도를 교차검증으로 고른다.
- **patsy 식으로 간단히 쓴다.** `bs(x, df=5)` 나 `cr(x, df=5)` 를 수식에 넣으면 기저행렬이 자동 생성되어 보통의 OLS 로 적합된다.
- **여전히 선형모형이다.** 기저함수가 비선형일 뿐 계수에 대해서는 선형이므로, 앞서 본 추론이 그대로 적용된다.

다음 절부터 **패키지 사용법**으로 13장을 마무리한다.
