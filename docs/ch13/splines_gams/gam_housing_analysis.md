# GAM 주택 분석

## 개요

이 페이지는 King County 주택 자료를 이용해 집값 예측에 일반화가법모형(GAM)을 적용한다. 선형회귀, 다항회귀, GAM(statsmodels와 pyGAM 양쪽)을 비교하고, 부분의존 그림을 시각화하며, RMSE와 $R^2$로 모형 성능을 평가한다.

!!! note "자료에 대하여"
    `house_98105`는 King County 주택 매매 자료 가운데 우편번호 98105 지역만 걸러 낸 pandas `DataFrame`이다. 아래 코드는 이 변수가 이미 만들어져 있다고 가정한다.

## 수학적 배경

일반화가법모형은 각 설명변수가 매끄러운 함수를 통해 비선형 효과를 갖도록 허용하여 선형회귀를 확장한다.

$$
y = \beta_0 + f_1(x_1) + f_2(x_2) + \cdots + f_p(x_p) + \varepsilon,
$$

여기서 각 $f_j$는 (보통 스플라인으로) 자료에서 추정한 매끄러운 함수이다. 각 설명변수가 반응변수에 독립적으로 기여하므로 "가법"이라 부른다.

### 평활 스플라인

각 $f_j$는 기저 전개, 흔히 B-스플라인으로 표현된다.

$$
f_j(x_j) = \sum_{m=1}^{M_j} \gamma_{jm}\, B_{jm}(x_j),
$$

여기서 $B_{jm}$은 기저함수이다. 매끄러움은 **평활 모수** $\lambda_j$로 조절되며 목적함수는 다음이 된다.

$$
\min_{\gamma} \sum_{i=1}^n \!\left(y_i - \beta_0 - \sum_{j=1}^p f_j(x_{ij})\right)^{\!2} + \sum_{j=1}^p \lambda_j \int [f_j''(t)]^2\, dt.
$$

벌점 $\lambda_j \int [f_j'']^2\,dt$가 $f_j$의 요동을 조절한다. $\lambda_j$가 클수록 더 매끄러운 곡선이 된다.

### 부분의존

$y$의 $x_j$에 대한 **부분의존**은 함수 $f_j(x_j)$이며, 다른 설명변수에 대해 평균을 낸 뒤 설명변수 $j$가 반응변수에 미치는 주변 효과를 보여준다.

## 자료

<div class="codebox" markdown>

### 예제 1. 주택 자료 읽기 { .eg }

```python
import pandas as pd

url = ("https://raw.githubusercontent.com/gedeck/"
       "practical-statistics-for-data-scientists/master/data/house_sales.csv")
house = pd.read_csv(url, sep='\t')
house_98105 = house.loc[house['ZipCode'] == 98105, :]

print(f"98105 지역 {len(house_98105)}건")
```

출력:

```
98105 지역 313건
```

98105 지역 313건이다. 아래에서 선형, 다항, 스플라인, GAM을 같은 자료에 적용해 비교한다.

</div>

### 선형 모형과 다항 모형

<div class="codebox" markdown>

#### 예제 2. 선형 모형과 다항 모형 { .eg }

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 세 모형을 차례로 세워 견준다: 선형 → 다항 → GAM.
predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade']
outcome = 'AdjSalePrice'

# 1) 선형 모형 — 기준선
X_linear = house_98105[predictors].assign(const=1)
result_linear = sm.OLS(house_98105[outcome], X_linear).fit()

# 2) 다항 모형 — 면적에 이차항을 더한다. 곡선을 담을 수는 있지만
# 다항식은 전 구간에 걸쳐 하나의 식이라, 한쪽 끝의 자료가 반대쪽 적합까지
# 흔든다는 약점이 있다.
formula_poly = ('AdjSalePrice ~ SqFtTotLiving + np.power(SqFtTotLiving, 2) + '
                'SqFtLot + Bathrooms + Bedrooms + BldgGrade')
result_poly = smf.ols(formula=formula_poly, data=house_98105).fit()
```

</div>

### statsmodels로 만드는 GAM

<div class="codebox" markdown>

#### 예제 3. statsmodels로 GAM 적합 { .eg }

```python
from statsmodels.gam.api import GLMGam, BSplines

# 3) GAM — 변수마다 매끄러운 함수를 따로 둔다. 스플라인은 구간마다
# 다른 다항식을 이어 붙이므로 다항식의 위 약점이 없다.
# 면적에만 자유도 10 을 주고 나머지는 3 으로 낮춰 두었다.
x_spline = house_98105[predictors]
bs = BSplines(x_spline, df=[10, 3, 3, 3, 3], degree=[3, 2, 2, 2, 2])
# alpha 는 매끄러움에 주는 벌점이다. 0 이면 벌점 없이 자유도대로 맞춘다.
alpha = np.array([0] * 5)

gam_sm = GLMGam.from_formula(
    'AdjSalePrice ~ SqFtTotLiving + SqFtLot + Bathrooms + Bedrooms + BldgGrade',
    data=house_98105, smoother=bs, alpha=alpha
)
res_sm = gam_sm.fit()
```

`alpha`를 모두 0으로 두었으므로 평활 벌점이 없는 회귀 스플라인이다. 매끄러움은 오직 `df`(기저함수의 개수)로만 조절된다.

</div>

### pyGAM으로 만드는 GAM

<div class="codebox" markdown>

#### 예제 4. pygam으로 GAM 적합 { .eg }

```python
from pygam import LinearGAM, s, l

X_gam = house_98105[predictors].values
y_gam = house_98105[outcome].values

# pygam 쪽은 항의 종류를 직접 고른다. s() 는 매끄러운 함수, l() 은 선형이다.
gam_py = LinearGAM(
    s(0, n_splines=12) +  # SqFtTotLiving: smooth
    l(1) +                 # SqFtLot: linear
    l(2) +                 # Bathrooms: linear
    l(3) +                 # Bedrooms: linear
    l(4)                   # BldgGrade: linear
)
# gridsearch 가 벌점 lambda 를 자동으로 골라 준다. statsmodels 쪽에서
# alpha 를 손으로 정한 것과 대비된다.
gam_py.gridsearch(X_gam, y_gam)
```

</div>

## 해석

- **선형 모형**: 각 설명변수가 가격에 일정한 주변 효과를 갖는다고 가정한다. 단순하지만 거주 면적과 가격 사이의 곡률을 놓칠 수 있다.
- **다항 모형**: 이차항으로 $\text{SqFtTotLiving}$의 곡률을 포착하지만 전역적인 모양을 강제한다(다항식이 모든 구간에 똑같이 적용된다).
- **GAM**: 다른 설명변수는 선형으로 두면서 $\text{SqFtTotLiving}$의 효과만 매끄러운 스플라인으로 자유롭게 변하도록 허용한다. 이 유연성이 대체로 적합을 개선한다.
- **부분의존 그림**은 각 $f_j$의 모양을 드러낸다. 부분의존이 거의 선형이면 선형항으로 충분하다는 뜻이고, 곡률이 있으면 매끄러운 항이 정당화된다.
- **유효 자유도**(EDF)는 각 매끄러운 항의 복잡도를 잰다. EDF가 클수록 요동이 심하고 과대적합 위험이 크다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span> 선형, 다항, GAM 모형의 $R^2$와 RMSE를 비교하라. 어느 모형이 가장 좋으며 그 개선은 실질적인가?

</div>

??? success "풀이"

    ```python
    from sklearn.metrics import mean_squared_error, r2_score

    models = {
        'Linear': result_linear.fittedvalues,
        'Polynomial': result_poly.fittedvalues,
        'GAM (pyGAM)': gam_py.predict(X_gam),
    }
    for name, pred in models.items():
        rmse = np.sqrt(mean_squared_error(y_gam, pred))
        r2 = r2_score(y_gam, pred)
        print(f"{name}: R2={r2:.4f}, RMSE={rmse:.0f}")
    ```

    출력:

    ```
    Linear: R2=0.7954, RMSE=176775
    Polynomial: R2=0.8058, RMSE=172241
    GAM (pyGAM): R2=0.8117, RMSE=169580
    ```

    선형 $R^2 = 0.795$, 다항 0.806, GAM 0.812로 조금씩 나아진다. RMSE로는 176,775달러에서 169,580달러로 4% 줄었다.

    개선폭이 크지 않다는 점이 오히려 유익한 결론이다. 이 자료에서 주택 가격과 설명변수의 관계는 대체로 선형에 가깝고, 비선형 모형이 가져오는 이득이 제한적이다. **더 유연한 모형이 언제나 크게 낫지는 않다.**

    GAM은 선형 모형보다 대체로 완만하게 개선되며 다항 모형과는 비슷한 성능을 보인다. 개선의 폭은 참 관계가 얼마나 비선형인가에 달려 있다.

    다만 이 세 값은 모두 **훈련자료**에서 계산된 것이므로, 더 유연한 모형이 유리할 수밖에 없다는 점에 유의하라. 공정한 비교를 하려면 교차검증이나 남겨 둔 검정자료에서 평가해야 한다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> pyGAM 설정을 바꾸어 모든 설명변수에 선형항 대신 매끄러운 스플라인을 쓰도록 하라. 적합이 개선되는가? 과대적합의 위험을 논하라.

</div>

??? success "풀이"

    ```python
    gam_all_smooth = LinearGAM(
        s(0) + s(1) + s(2) + s(3) + s(4)
    )
    gam_all_smooth.gridsearch(X_gam, y_gam)
    ```

    모든 설명변수에 매끄러운 항을 쓰면 유연성이 커져 훈련 $R^2$는 대체로 오르지만 과대적합할 수 있다. 유효 자유도가 늘어나고 모형이 신호가 아니라 잡음을 포착할 수 있다. 늘어난 유연성이 표본 밖 예측을 실제로 개선하는지는 교차검증으로 확인해야 한다.

    `Bathrooms`나 `Bedrooms`처럼 값이 몇 가지 정수뿐인 변수에 매끄러운 항을 쓰는 것은 특히 위험하다. 값의 종류가 적은 곳에 스플라인의 자유도를 쓰면 잡음을 외우기 쉽다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> GAM이 왜 "가법"이라 불리는지 설명하라. 가법 구조는 어떤 가정을 부과하며 언제 위배될 수 있는가?

</div>

??? success "풀이"

    GAM은 반응변수가 개별 매끄러운 함수들의 합이라고 가정한다: $y = \beta_0 + f_1(x_1) + \cdots + f_p(x_p) + \varepsilon$. 곧 각 설명변수의 효과가 다른 설명변수의 값과 무관하다는 뜻이다(교호작용 없음). 예를 들어 거주 면적이 가격에 미치는 효과가 건물 등급에 따라 달라진다면(교호작용) 이 가정이 위배된다. 텐서곱 평활 같은 확장으로 교호작용을 다룰 수 있다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> 평활 모수 $\lambda$는 편향-분산 절충을 조절한다. $\lambda \to 0$과 $\lambda \to \infty$일 때 어떻게 되는지 설명하라.

</div>

??? success "풀이"

    $\lambda \to 0$이면 벌점이 사라져 $f_j$가 자료를 보간한다(분산 큼, 편향 작음, 과대적합). $\lambda \to \infty$이면 벌점이 $f_j'' \equiv 0$을 강제하여 $f_j$가 선형이 된다(분산 작음, 편향 클 수 있음, 과소적합). 최적 $\lambda$는 이 양극단의 균형을 잡는다. pyGAM에서는 `gridsearch`가 일반화 교차검증(GCV)이나 비슷한 기준을 최소화하여 $\lambda$를 고른다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span> B-스플라인 기저를 쓰는 단변량 GAM의 벌점최소제곱 문제를 행렬로 표현하라. $\mathbf{B}$가 기저행렬이고 $\mathbf{D}$가 벌점행렬일 때 해가 $(B^\top B + \lambda D)^{-1} B^\top y$임을 보여라.

</div>

??? success "풀이"

    $f(x) = \sum_{m=1}^M \gamma_m B_m(x)$라 하면 $\mathbf{f} = \mathbf{B}\boldsymbol{\gamma}$이고 $\mathbf{B}$는 $n \times M$ 기저행렬이다. 거칢 벌점은 $\int [f'']^2\,dt = \boldsymbol{\gamma}^\top\mathbf{D}\boldsymbol{\gamma}$이며 $D_{jk} = \int B_j''(t) B_k''(t)\,dt$이다. 벌점 목적함수는

    $$
    (\mathbf{y} - \mathbf{B}\boldsymbol{\gamma})^\top(\mathbf{y} - \mathbf{B}\boldsymbol{\gamma}) + \lambda\,\boldsymbol{\gamma}^\top\mathbf{D}\boldsymbol{\gamma}.
    $$

    $\boldsymbol{\gamma}$에 대해 미분하여 0으로 두면

    $$
    -2\mathbf{B}^\top(\mathbf{y} - \mathbf{B}\boldsymbol{\gamma}) + 2\lambda\mathbf{D}\boldsymbol{\gamma} = \mathbf{0} \implies \hat{\boldsymbol{\gamma}} = (\mathbf{B}^\top\mathbf{B} + \lambda\mathbf{D})^{-1}\mathbf{B}^\top\mathbf{y}.
    $$

    이는 기저 계수에 대한 릿지 형태의 회귀이며 $\lambda\mathbf{D}$가 구조를 가진 벌점 역할을 한다. $\square$

---

## 정리하며

GAM 은 **각 변수의 효과를 매끄러운 함수로** 허용한다.

$$
y=\beta_0+f_1(x_1)+\cdots+f_p(x_p)+\varepsilon
$$

- **선형회귀와 완전 비선형 모형 사이에 있다.** 각 변수의 모양은 자유롭지만 **효과들이 여전히 더해진다**(가법성). 그래서 해석 가능성이 유지된다.
- **부분의존 그림이 해석의 도구다.** 변수 하나의 $f_j$ 를 그리면 "그 변수가 어떤 모양으로 영향을 주는가"가 직접 보인다. **선형회귀의 계수 하나가 곡선 하나로 바뀐 셈**이다.
- **매끄러움 정도를 정해야 한다.** 자유도나 벌점 모수를 교차검증으로 고르며, 너무 유연하면 과대적합한다.
- **교호작용은 기본적으로 없다.** 필요하면 명시적으로 넣어야 하며, 그러면 해석이 어려워진다.
- **선형·다항과 비교해 얻는 것과 잃는 것.** 예측 성능은 대개 오르고, 계수 하나로 요약하는 간결함은 잃는다.

다음 절 **계단함수**로 넘어간다.
