# 다중회귀 진단

## 개요

이 페이지는 다중선형회귀 진단을 종합적으로 따라가 본다. California Housing 자료로 모형을 적합하고, 분산팽창인자(VIF)로 다중공선성을 확인하고, 잔차 분석을 수행하고, Cook 거리로 영향점을 찾고, AIC와 BIC로 모형을 비교하며, 로그 변환·교호작용 항·다항회귀 같은 확장을 살펴본다.

## 수학적 배경

### 분산팽창인자

설명변수 $j$에 대해 VIF는 다른 설명변수와의 상관 때문에 $\hat{\beta}_j$의 분산이 얼마나 부풀려졌는지를 잰다.

$$
\mathrm{VIF}_j = \frac{1}{1 - R_j^2},
$$

여기서 $R_j^2$는 $x_j$를 나머지 모든 설명변수에 회귀시켰을 때의 $R^2$이다. VIF가 5–10을 넘으면 문제가 되는 다중공선성을 나타낸다.

### 잔차 진단

관측값 $i$의 표준화 잔차는

$$
r_i = \frac{e_i}{s\sqrt{1 - h_{ii}}},
$$

여기서 $h_{ii}$는 모자행렬 $\mathbf{H} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$의 $i$번째 대각원소이다.

### Cook 거리

Cook 거리는 $i$번째 관측값이 모든 적합값에 미치는 영향을 잰다.

$$
D_i = \frac{r_i^2}{k} \cdot \frac{h_{ii}}{1 - h_{ii}},
$$

여기서 $k$는 모수의 개수이다. $D_i > 4/n$인 관측값을 영향점으로 본다.

### 정보기준

$$
\mathrm{AIC} = n \ln(\mathrm{RSS}/n) + 2k, \qquad \mathrm{BIC} = n \ln(\mathrm{RSS}/n) + k \ln(n).
$$

$n > e^2 \approx 7.4$이면 BIC가 AIC보다 모형 복잡도에 더 무거운 벌점을 준다.

!!! note "statsmodels의 AIC와 위 공식은 상수만큼 다르다"
    위 공식은 모형 비교에 영향을 주지 않는 상수 $n\ln(2\pi) + n$을 뺀 간이형이다. `results.aic`는 $-2\ln L + 2k$를 그대로 계산하므로 절댓값이 다르다. 모형들 사이의 **차이**는 같으므로 비교 결과는 동일하다.

## 코드

### 적합과 VIF 계산

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.api import OLS, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['PRICE'] = housing.target

features = ['MedInc', 'AveRooms', 'AveOccup']
X = add_constant(df[features])
y = df['PRICE']

model = OLS(y, X).fit()

# Compute VIF for each feature (skip index 0: the constant)
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns[1:]
vif_data['VIF'] = [variance_inflation_factor(X.values, i)
                   for i in range(1, X.shape[1])]
print(vif_data)
```

출력:

```text
    Feature   VIF
0    MedInc  1.120
1  AveRooms  1.120
2  AveOccup  1.000
```

세 설명변수 모두 VIF가 1에 가까워 다중공선성 문제가 없다. 이 모형의 $R^2$는 $0.4808$이다.

### 잔차 분석

```python
y_pred = model.predict(X)
residuals = y - y_pred

# Residuals vs Fitted
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_pred, residuals, alpha=0.3, s=10)
axes[0].axhline(y=0, color='red', linestyle='--')
axes[0].set_xlabel('Fitted Values')
axes[0].set_ylabel('Residuals')

# Q-Q Plot
sm.qqplot(residuals, line='45', ax=axes[1])
plt.tight_layout()
plt.show()
```

이 잔차는 왜도 $1.256$, 첨도 $5.965$로 오른쪽으로 치우치고 꼬리가 두껍다. 반응변수 `PRICE`가 $5.00001$(50만 달러)에서 절단되어 있어(전체의 4.7%) 위쪽에 눈에 띄는 수평 띠가 나타나는 것도 확인할 수 있다.

### AIC와 BIC를 이용한 모형선택

```python
feature_sets = {
    'Model 1': ['MedInc'],
    'Model 2': ['MedInc', 'AveRooms'],
    'Model 3': ['MedInc', 'AveRooms', 'AveOccup'],
    'Model 4': list(housing.feature_names),
}

for name, feats in feature_sets.items():
    X_temp = add_constant(df[feats])
    m = OLS(y, X_temp).fit()
    print(f"{name}: AIC={m.aic:.1f}, BIC={m.bic:.1f}, "
          f"R2={m.rsquared:.4f}")
```

출력:

```text
Model 1: AIC=51249.3, BIC=51265.2, R2=0.4734
Model 2: AIC=51016.2, BIC=51040.1, R2=0.4794
Model 3: AIC=50962.2, BIC=50994.0, R2=0.4808
Model 4: AIC=45265.5, BIC=45337.0, R2=0.6062
```

여덟 개 특성을 모두 쓴 Model 4가 AIC와 BIC 모두에서 압도적으로 낫다($\Delta\text{AIC} \approx 5700$). $R^2$도 0.48에서 0.61로 크게 오른다.

## 해석

- **VIF**: 1에 가까우면 공선성이 거의 없다. VIF가 커지면 해당 계수의 표준오차가 부풀려져 유의성 검정을 믿을 수 없게 된다.
- **잔차-적합값**: 무작위로 흩어져 있으면 선형성과 상수분산 가정이 충족된다. 곡선이나 깔때기 같은 패턴은 모형 오설정이나 이분산을 시사한다.
- **Q-Q 그림**: 점들이 대각선을 따르면 잔차가 정규분포를 따른다. 꼬리에서 벗어나면 두껍거나 치우친 잔차 분포를 시사한다.
- **Cook 거리**: 이상점(큰 잔차)이면서 동시에 지렛대가 큰(설명변수 값이 특이한) 관측값은 Cook 거리가 크고 회귀 적합을 왜곡할 수 있다.
- **AIC/BIC**: 값이 작을수록 좋은 모형이다. AIC는 더 큰 모형을, BIC는 절약성을 선호하는 경향이 있다.

## 연습문제

**연습문제 1.** California Housing의 완전모형(8개 특성 전부)에서 각 설명변수의 VIF를 계산하라. 어느 설명변수가 높은 다중공선성을 보이는지 찾아라.

??? success "풀이"

    ```python
    X_full = add_constant(df[list(housing.feature_names)])
    for i, col in enumerate(X_full.columns):
        if col == 'const':
            continue          # VIF of the intercept is meaningless
        vif = variance_inflation_factor(X_full.values, i)
        print(f"{col}: VIF = {vif:.2f}")
    ```

    출력:

    ```text
    MedInc: VIF = 2.50
    HouseAge: VIF = 1.24
    AveRooms: VIF = 8.34
    AveBedrms: VIF = 6.99
    Population: VIF = 1.14
    AveOccup: VIF = 1.01
    Latitude: VIF = 9.30
    Longitude: VIF = 8.96
    ```

    VIF가 5를 넘는 것은 네 개이며 두 쌍으로 묶인다.

    - `AveRooms`(8.34)와 `AveBedrms`(6.99): 둘 다 방 수를 재므로 강하게 상관되어 있다.
    - `Latitude`(9.30)와 `Longitude`(8.96): 캘리포니아가 북서-남동으로 길게 뻗어 있어 두 좌표의 상관이 $r = -0.925$에 이른다.

    대책으로는 각 쌍에서 하나를 빼거나, 둘을 결합한 파생변수(예: 방당 침실 수, 위치 지표)를 만들거나, 정칙화를 쓰는 방법이 있다. 상수 열의 VIF는 의미가 없으므로 계산에서 제외해야 한다. $\square$

---

**연습문제 2.** 반응변수를 로그 변환한 $\ln(\text{PRICE})$로 모형을 적합하라. 잔차의 Q-Q 그림을 변환하지 않은 모형과 비교하고 어느 쪽이 정규성 가정을 더 잘 만족하는지 논하라.

??? success "풀이"

    ```python
    y_log = np.log(df['PRICE'])
    model_log = OLS(y_log, X).fit()
    sm.qqplot(model_log.resid, line='45')
    ```

    집값이 오른쪽으로 치우쳐 있으므로 로그 변환 모형의 잔차가 정규에 훨씬 가까워진다. 수치로 확인하면

    | 모형 | 잔차 왜도 | 잔차 첨도 | $R^2$ |
    |---|---|---|---|
    | PRICE | 1.256 | 5.965 | 0.4808 |
    | ln(PRICE) | 0.101 | 4.135 | 0.4430 |

    왜도가 $1.256$에서 $0.101$로 거의 사라져 Q-Q 그림에서 점들이 대각선에 훨씬 밀착한다. 첨도는 여전히 정규의 3보다 크지만 크게 개선되었다. 로그 변환이 큰 값을 압축하고 분산을 안정화하기 때문이다.

    다만 $R^2$가 0.481에서 0.443으로 떨어진 것을 두 모형의 우열로 읽어서는 안 된다. 반응변수의 척도가 달라졌으므로 두 $R^2$는 서로 비교할 수 있는 양이 아니다. $\square$

---

**연습문제 3.** 문턱값 $4/n$의 Cook 거리를 써서 설명변수 3개 모형에서 모든 영향점을 제거하라. $R^2$와 계수 추정값의 변화를 보고하라.

??? success "풀이"

    ```python
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    mask = cooks_d < 4 / len(y)
    X_clean = X[mask]
    y_clean = y[mask]
    model_clean = OLS(y_clean, X_clean).fit()
    print(f"Original R2: {model.rsquared:.4f}")
    print(f"Cleaned R2:  {model_clean.rsquared:.4f}")
    ```

    20,640개 가운데 470개(2.3%)가 제거되고 $R^2$는 $0.4808$에서 $0.5784$로 오른다. 계수도 크게 달라진다.

    | 항 | 원래 | 제거 후 |
    |---|---|---|
    | const | 0.6069 | 1.2564 |
    | MedInc | 0.4347 | 0.5188 |
    | AveRooms | $-0.0383$ | $-0.1399$ |
    | AveOccup | $-0.0042$ | $-0.1590$ |

    `AveOccup`의 계수는 38배가 되고 `AveRooms`도 3.7배가 된다. 소수의 극단적인 관측값이 이 두 계수를 거의 0으로 끌어내리고 있었다는 뜻이다.

    !!! warning "$R^2$가 올랐다고 좋아진 모형이 아니다"
        잔차가 큰 점들을 골라 지웠으니 $R^2$가 오르는 것은 당연하다. 이는 모형이 나아졌다는 증거가 아니라 자료를 바꾼 결과일 뿐이다. 이 470개 점이 기록 오류인지, 절단된 최고가 구간에 속한 정당한 관측값인지 먼저 조사해야 한다. 실제로 이 자료의 `PRICE`는 5.00001에서 절단되어 있으므로 상당수가 후자일 가능성이 높다. $\square$

---

**연습문제 4.** 정규 가능도에서 AIC 공식 $n\ln(\mathrm{RSS}/n) + 2k$가 왜 $-2\ln L + 2k$와 (상수를 빼면) 동등한지 설명하라.

??? success "풀이"

    정규 모형에서 최대화된 로그가능도는

    $$
    \ln L = -\frac{n}{2}\ln(2\pi\hat{\sigma}^2) - \frac{n}{2},
    $$

    여기서 $\hat{\sigma}^2 = \mathrm{RSS}/n$이다. 따라서

    $$
    -2\ln L = n\ln(2\pi) + n\ln(\mathrm{RSS}/n) + n.
    $$

    $2k$를 더하면 $\mathrm{AIC} = n\ln(\mathrm{RSS}/n) + 2k + \text{상수}$가 된다. 상수 $n\ln(2\pi) + n$은 모형에 의존하지 않으므로 모형 비교에서는 버릴 수 있다. $\square$

---

**연습문제 5.** 모형에 교호작용 항 $\text{MedInc} \times \text{AveRooms}$와 이차항 $\text{MedInc}^2$을 추가하라. AIC로 이 추가가 모형을 개선하는지 판정하라.

??? success "풀이"

    ```python
    df['MedInc_x_AveRooms'] = df['MedInc'] * df['AveRooms']
    df['MedInc_sq'] = df['MedInc'] ** 2
    X_ext = add_constant(df[features + ['MedInc_x_AveRooms', 'MedInc_sq']])
    model_ext = OLS(y, X_ext).fit()
    print(f"Base AIC: {model.aic:.1f}")
    print(f"Extended AIC: {model_ext.aic:.1f}")
    ```

    출력:

    ```text
    Base AIC: 50962.2
    Extended AIC: 50767.5
    ```

    확장 모형의 AIC가 $194.7$ 낮으므로 추가된 두 항이 늘어난 복잡도를 충분히 정당화한다($R^2$도 $0.4808$에서 $0.4858$로 오른다). 교호작용은 소득의 효과가 방 수에 의존하는지를, 이차항은 소득이 집값에 미치는 수익체감을 포착한다. $\square$
