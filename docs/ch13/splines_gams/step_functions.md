# 계단함수

## 개요

이 페이지는 비선형 모형화 기법으로서 계단함수 회귀(조각별 상수 적합)를 보인다. King County 주택 자료를 써서 구간화한 설명변수로 지시변수를 만들고, 계단함수 모형을 적합하고, 구간 개수를 달리해 비교하며, 계단함수를 선형·다항·스플라인 대안과 대조한다. 분류를 위한 로지스틱 회귀에 계단함수를 적용하는 방법도 다룬다.

## 수학적 배경

계단함수는 설명변수 $x$의 범위를 절단점 $c_1 < c_2 < \cdots < c_{K-1}$로 $K$개 구간으로 나누고 지시변수를 만든다.

$$
C_k(x) = \mathbf{1}(c_{k-1} < x \leq c_k), \qquad k = 1, \ldots, K.
$$

계단함수 회귀모형은

$$
y_i = \beta_0 + \beta_1 C_1(x_i) + \beta_2 C_2(x_i) + \cdots + \beta_K C_K(x_i) + \varepsilon_i.
$$

이는 각 구간 안에서 별도의 상수(구간 평균)를 적합하는 것과 같다. 구간 $k$에 속한 임의의 $x$에 대한 적합값은 (지시변수를 하나도 빼지 않고 모두 넣었을 때) 그 구간의 평균 반응 $\bar{y}_k$가 된다.

### 절단점 정하기

절단점은 다음 기준으로 정할 수 있다.

- **등간격 구간**: 범위를 $K$개의 같은 구간으로 나눈다
- **분위수 구간**(`pd.qcut`): 각 구간이 대략 같은 개수의 관측값을 담는다
- **분야 지식**: 의미 있는 문턱에 절단점을 둔다

## 자료

세 페이지가 공유하는 King County(시애틀) 주택 매매 자료를 읽는다.

<div class="codebox" markdown>

### 예제 1. 주택 자료 읽기 { .eg }

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

22,687건이다. 가격이 3,368달러에서 1,164만 달러까지 퍼져 있어 오른쪽으로 크게 치우친 자료다.

</div>

### pd.cut으로 계단함수 만들기

<div class="codebox" markdown>

#### 예제 2. 계단함수 회귀 { .eg }

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

age = 2024 - house['YrBuilt'].values
price = house['AdjSalePrice'].values
df = pd.DataFrame({'age': age, 'price': price})

# 계단함수는 연속변수를 구간으로 잘라 구간마다 상수를 맞추는 것이다.
# 가장 거친 비선형 모형이지만 해석이 쉽다는 장점이 있다.
knots = [0, 20, 40, 60, 80, 150]
df['age_bin'] = pd.cut(df['age'], bins=knots, include_lowest=True)

# 구간마다 가변수 하나씩을 만든다.
df_dummies = pd.get_dummies(df['age_bin'], drop_first=False)

X_step = df_dummies.values
step_model = LinearRegression()
step_model.fit(X_step, df['price'])

# 절편 없이 적합하면 각 계수가 곧 그 구간의 평균이 된다.
step_model_nc = LinearRegression(fit_intercept=False).fit(X_step, df['price'])
for interval, coef in zip(df_dummies.columns, step_model_nc.coef_):
    print(f"{str(interval):<18} 평균 가격 = {coef:>10,.0f}")
print(f"R^2 = {r2_score(df['price'], step_model.predict(X_step)):.4f}")
```

출력:

```
(-0.001, 20.0]     평균 가격 =    642,643
(20.0, 40.0]       평균 가격 =    621,917
(40.0, 60.0]       평균 가격 =    490,326
(60.0, 80.0]       평균 가격 =    482,741
(80.0, 150.0]      평균 가격 =    571,694
R^2 = 0.0283
```

구간별 평균 가격이 나이에 따라 단조롭지 않다. 20년 미만이 64만, 60~80년이 48만으로 가장 낮고, 80년이 넘으면 57만으로 다시 오른다.

**계단함수의 값어치가 여기 있다.** 선형 모형이라면 "나이가 들수록 싸진다" 같은 단조 관계만 잡아낼 수 있지만, 계단함수는 U자 모양을 그대로 담는다. 물론 $R^2 = 0.028$로 설명력 자체는 낮다.

</div>

!!! note "지시변수를 모두 넣으면 설계행렬의 계수가 부족해진다"
    `drop_first=False`로 만든 $K$개의 지시변수는 합이 1이므로 절편과 완전 공선이다. `LinearRegression`은 내부적으로 최소제곱 최소노름 해를 쓰므로 **예측값 자체는 정확하지만** 개별 계수를 해석할 수는 없다. 계수를 해석하려면 `drop_first=True`로 기준 구간을 하나 두거나, 절편 없이(`fit_intercept=False`) $K$개 지시변수만 쓰면 된다. 뒤쪽 방식에서는 각 계수가 그 구간의 평균이 된다(연습문제 4 참조).

### 구간 개수 비교

<div class="codebox" markdown>

#### 예제 3. 구간 수를 바꿔 가며 { .eg }

```python
# 구간을 몇 개로 나눌지가 이 방법의 유일한 조절값이다. 늘리면 R^2 는
# 반드시 오르지만 구간마다 자료가 줄어 추정이 불안해진다.
# qcut 은 개수가 고르게 들어가도록 분위수로 자른다.
results = []
for n_bins in [3, 4, 5, 6, 8, 10]:
    df['bin_temp'] = pd.qcut(df['age'], q=n_bins, duplicates='drop')
    X_temp = pd.get_dummies(df['bin_temp'], drop_first=False).values
    model = LinearRegression().fit(X_temp, df['price'])
    pred = model.predict(X_temp)
    r2 = r2_score(df['price'], pred)
    rmse = np.sqrt(mean_squared_error(df['price'], pred))
    results.append({'n_bins': n_bins, 'R2': r2, 'RMSE': rmse})

print(pd.DataFrame(results).round(4).to_string(index=False))
```

출력:

```
 n_bins     R2        RMSE
      3 0.0226 381016.7261
      4 0.0239 380754.8087
      5 0.0286 379852.0395
      6 0.0327 379038.4143
      8 0.0332 378949.6438
     10 0.0334 378904.4326
```

구간을 3개에서 10개로 늘려도 $R^2$가 0.023에서 0.033으로 밖에 오르지 않는다.

구간을 늘리면 모수가 늘어 훈련 자료에 대한 적합은 반드시 좋아진다. 그런데도 이만큼밖에 오르지 않는다는 것은 주택 나이 하나로 가격을 설명하는 데 한계가 있다는 뜻이다.

</div>

### 다른 방법과의 비교

<div class="codebox" markdown>

#### 예제 4. 직선·다항식과 견주기 { .eg }

```python
# 견줄 기준선 둘. 직선과 3차 다항식이다.
linear_model = LinearRegression()
linear_model.fit(df[['age']].values, df['price'])

X_poly = np.column_stack([df['age'] ** i for i in range(1, 4)])
poly_model = LinearRegression().fit(X_poly, df['price'])
```

</div>

## 해석

- **계단함수**는 조각별 상수이다. 한 구간 안의 모든 관측값에 대해 예측값이 같다. 그래서 구간 경계에서 불연속인 "계단" 모양의 적합값이 나온다.
- **장점**: 해석이 단순하고(예측값이 곧 구간 평균이다), 구현이 쉬우며, 국면 전환을 포착할 수 있다.
- **단점**: 경계에서 불연속이고(매끄러운 전이가 없고), 자유도를 낭비하며(구간마다 상수 하나에 모수 하나를 쓴다), 구간 배치에 민감하다.
- **비교**: 모수 개수가 같을 때 계단함수는 대체로 다항이나 스플라인 모형보다 $R^2$가 낮다. 스플라인과 다항식은 매끄러움을 통해 이웃 구역끼리 정보를 공유하기 때문이다.
- **분류**: 로그오즈가 특정 설명변수 값에서 급격히 변할 때 계단함수는 로지스틱 회귀에서도 잘 작동한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span> 분위수 기반 구간($K = 5$)으로 계단함수를 적합하고 등간격 구간의 구간 평균과 비교하라. 어느 방식이 더 균일한 예측 품질을 주는가?

</div>

??? success "풀이"

    ```python
    # 분위수로 나눈 구간 — 구간마다 개수가 고르다
    df['q_bin'] = pd.qcut(df['age'], q=5)
    q_means = df.groupby('q_bin')['price'].agg(['mean', 'count'])

    # 폭이 같은 구간 — 구간마다 개수가 들쭉날쭉하다
    df['w_bin'] = pd.cut(df['age'], bins=5)
    w_means = df.groupby('w_bin')['price'].agg(['mean', 'count'])
    ```

    분위수 구간은 각 구간의 관측값 개수를 대략 같게 만들어 구간 평균의 표준오차를 더 균일하게 한다. 등간격 구간은 극단 구간(예: 아주 오래된 집)에 관측값이 아주 적어 그 구간의 추정을 믿을 수 없게 될 수 있다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 구간 개수를 늘리면 훈련 $R^2$가 항상 커지거나(적어도 줄지 않고) 검정오차는 커질 수 있는 이유를 설명하라.

</div>

??? success "풀이"

    구간이 하나 늘 때마다 모수도 하나 늘어난다. 구간이 많아지면 모형이 훈련자료의 더 세밀한 패턴을 맞출 수 있어 훈련 RSS가 줄고 $R^2$가 커진다. 그러나 구간이 너무 많으면(특히 관측값이 적은 구간에서) 구간 평균이 참 조건부 기댓값의 잡음 섞인 추정이 된다. 표본 밖에서는 이 잡음 섞인 추정이 예측오차를 키운다. 극단적으로 $K = n$개 구간이면 훈련자료에서 $R^2 = 1$이지만 표본 밖 성능은 형편없다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> `pd.cut` 대신 `np.digitize`로 계단함수를 구현하라. 결과가 일치하는지 확인하라.

</div>

??? success "풀이"

    ```python
    bin_edges = [0, 20, 40, 60, 80, 150]
    bin_indices = np.digitize(df['age'], bin_edges)
    # 가변수 행렬을 만든다
    K = len(bin_edges) - 1
    X_digit = np.zeros((len(df), K))
    for k in range(K):
        X_digit[:, k] = (bin_indices == k + 1).astype(float)
    model_digit = LinearRegression().fit(X_digit, df['price'])
    ```

    두 방법이 같은 지시행렬을 만들므로 예측값이 일치한다. 다만 경계값 처리가 미묘하게 다르다. `np.digitize`는 기본적으로 왼쪽이 닫힌 구간 $[c_{k-1}, c_k)$을 쓰고 `pd.cut`은 오른쪽이 닫힌 구간 $(c_{k-1}, c_k]$을 쓴다. 절단점과 정확히 같은 값이 자료에 있다면 두 방법의 배정이 달라진다. `pd.cut`과 맞추려면 `np.digitize(..., right=True)`를 쓴다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> (모든 지시변수를 넣고 절편은 없는) 계단함수 모형에서 각 구간 계수의 OLS 추정값이 그 구간 반응변수의 표본평균과 같음을 수학적으로 보여라.

</div>

??? success "풀이"

    $\mathbf{X} = [C_1 \mid C_2 \mid \cdots \mid C_K]$라 하고 $C_k$를 구간 $k$의 지시벡터라 하자. 구간이 서로 겹치지 않으므로 $\mathbf{X}^\top\mathbf{X} = \mathrm{diag}(n_1, n_2, \ldots, n_K)$이고 $\mathbf{X}^\top\mathbf{y} = (\sum_{i \in B_1} y_i, \ldots, \sum_{i \in B_K} y_i)^\top$이다. 따라서

    $$
    \hat{\boldsymbol{\beta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y} = \left(\frac{\sum_{i \in B_1} y_i}{n_1}, \ldots, \frac{\sum_{i \in B_K} y_i}{n_K}\right)^\top = (\bar{y}_1, \ldots, \bar{y}_K)^\top.
    $$

    각 계수가 그 구간 안의 표본평균이다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> "비싼" 주택을 예측하는 계단함수 로지스틱 회귀를, 연속형 age를 쓰는 로지스틱 회귀와 비교하라. 어느 쪽의 분류 정확도가 높으며 그 이유는 무엇인가?

</div>

??? success "풀이"

    ```python
    from sklearn.linear_model import LogisticRegression

    # "비싼" 주택을 중앙값 위로 정의한다.
    df = pd.DataFrame({'age': 2024 - house['YrBuilt'].values,
                       'price': house['AdjSalePrice'].values})
    df['expensive'] = (df['price'] > df['price'].median()).astype(int)

    # 연속형 설명변수
    logit_cont = LogisticRegression().fit(df[['age']], df['expensive'])
    acc_cont = logit_cont.score(df[['age']], df['expensive'])

    # 계단함수 설명변수
    X_class = pd.get_dummies(pd.cut(df['age'], bins=5), drop_first=True).values
    logit_step = LogisticRegression().fit(X_class, df['expensive'])
    acc_step = logit_step.score(X_class, df['expensive'])

    print(f"연속형 정확도: {acc_cont:.4f}")
    print(f"계단함수 정확도: {acc_step:.4f}")
    ```

    출력:

    ```
    연속형 정확도: 0.5614
    계단함수 정확도: 0.5818
    ```

    계단함수 쪽이 정확도 0.582로 연속형의 0.561보다 조금 낫다. 주택 나이와 가격의 관계가 단조가 아니기 때문이다. 위 구간별 평균에서 보았듯 40~80년 된 집이 가장 싸고, 새 집과 아주 오래된 집이 비싸다.

    참 로그오즈가 age에 대해 대략 선형이면 연속형 모형이 더 적은 모수로 비슷하거나 더 나은 성능을 낼 수 있다. 로그오즈가 급격히 변한다면(예: 60년이 넘은 집은 비쌀 가능성이 훨씬 낮다면) 계단함수가 이를 더 잘 포착한다. 상대적 성능은 참 관계가 어떤 모양인지에 달려 있다.

    다만 두 정확도 모두 **훈련자료**에서 계산된 것임에 유의하라. 계단함수 쪽이 모수가 더 많으므로 훈련 정확도에서 유리할 수밖에 없다. 공정한 비교를 하려면 교차검증을 써야 한다. $\square$

---

## 정리하며

계단함수는 **연속변수를 구간으로 잘라** 조각별 상수로 적합한다.

- **가장 단순한 비선형 모형이다.** 지시변수를 만들어 넣으면 보통의 선형회귀가 되며, 구간마다 다른 상수를 갖는다.
- **장점은 해석의 편의다.** "소득 3000만원 미만 집단은 …"처럼 말하기 쉽고, 비선형성을 가정 없이 담는다.
- **대가가 크다.** 구간 경계에서 불연속이라 **경계를 살짝 넘는 두 관측이 전혀 다른 예측값**을 받는다. 실제 관계가 매끄럽다면 부자연스럽다.
- **경계 위치가 자의적이다.** 자료를 보고 정하면 선택 편향이 생기고, 균등 분위수로 정하는 것이 관례다.
- **스플라인이 이 문제를 고친다.** 조각별 다항이면서 경계에서 매끄럽게 이어 붙이며, 다음 절의 주제다.

다음 절 **스플라인 (patsy)** 으로 넘어간다.
