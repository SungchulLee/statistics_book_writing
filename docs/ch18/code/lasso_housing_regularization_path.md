# 주택가격 자료의 라쏘 정칙화 경로

## 개요

이 절에서는 실제 주택 자료(King County 주택 매매 자료)에 라쏘 회귀를 적용하여, 영모형
($\hat{\beta} = 0$)에서 OLS 해까지 이어지는 정칙화 경로 전체를 추적한다. 교차검증으로 최적
$\lambda$를 고르고, OLS 및 능형회귀와 성능을 비교하며, 라쏘의 변수선택에서 어떤 주택 특성이
살아남는지 해석한다.

!!! note "자료 파일"
    아래 코드는 `docs/data/house_sales.csv`(탭 구분)를 읽는다. 이 파일은 저장소에 포함되어
    있지 않으므로, King County 주택 매매 자료를 내려받아 해당 경로에 두어야 실행된다.

## 문제 설정

조정된 매매가를 주택 특성의 선형함수로 모형화한다.

$$
\text{AdjSalePrice} = \beta_0 + \beta_1 \cdot \text{SqFtTotLiving} + \beta_2 \cdot \text{SqFtLot} + \cdots + \varepsilon.
$$

설명변수에는 수치형 특성(연면적, 욕실 수, 건축연도)과 원-핫 부호화된 범주형 특성(부동산 유형)이
모두 포함된다. 적합 전에 모든 특성을 표준화한다.

## 코드: 자료 적재와 준비

```python
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV

DATA = Path(__file__).parent.parent.parent / 'data'
house = pd.read_csv(DATA / 'house_sales.csv', sep='\t')

predictors = [
    'SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms',
    'BldgGrade', 'PropertyType', 'NbrLivingUnits',
    'SqFtFinBasement', 'YrBuilt', 'YrRenovated', 'NewConstruction'
]
outcome = 'AdjSalePrice'

X = pd.get_dummies(house[predictors], drop_first=True)
X['NewConstruction'] = X['NewConstruction'].astype(int)
y = house[outcome]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
```

$L_1$ 벌점 $\lambda\|\beta\|_1$은 모든 계수를 동등하게 벌하므로 표준화가 필수적이다. 표준화하지
않으면 단위가 다른 변수들이 서로 다른 정도로 벌을 받게 된다.

## 코드: OLS 기준선

```python
ols_model = LinearRegression().fit(X_scaled, y)
ols_pred = ols_model.predict(X_scaled)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

ols_rmse = np.sqrt(mean_squared_error(y, ols_pred))
ols_r2 = r2_score(y, ols_pred)
n_nonzero_ols = np.sum(np.abs(ols_model.coef_) > 1e-8)
```

OLS는 모든 특성을 0이 아닌 계수로 유지하며, 정칙화하지 않은 기준선 역할을 한다.

## 코드: 정칙화 경로

로그 등간격 $\lambda$ 100개에 대해 라쏘를 적합하고 각 계수의 변화를 추적한다.

```python
import matplotlib.pyplot as plt

alphas = np.logspace(2, -2, 100)

lasso_coefs = []
for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_scaled, y)
    lasso_coefs.append(lasso.coef_)

lasso_coefs = np.array(lasso_coefs)
n_features_selected = (np.abs(lasso_coefs) > 1e-8).sum(axis=1)
```

정칙화 경로는 $\lambda$가 작아짐에 따라 특성들이 어떤 순서로 모형에 들어오는지 보여준다. 먼저
등장하는 특성일수록 강한 예측변수다.

## 코드: 교차검증으로 람다 선택

```python
lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_scaled, y)

lasso_pred = lasso_cv.predict(X_scaled)
lasso_rmse = np.sqrt(mean_squared_error(y, lasso_pred))
lasso_r2 = r2_score(y, lasso_pred)
n_nonzero_lasso = np.sum(np.abs(lasso_cv.coef_) > 1e-8)
```

5-겹 교차검증 절차가 각 $\lambda$를 평가하여 평균 MSE가 가장 낮은 값을 고른다.

## 코드: 모형 비교

```python
ridge_cv = RidgeCV(alphas=np.logspace(-2, 5, 100), cv=5)
ridge_cv.fit(X_scaled, y)

ridge_pred = ridge_cv.predict(X_scaled)
ridge_rmse = np.sqrt(mean_squared_error(y, ridge_pred))
ridge_r2 = r2_score(y, ridge_pred)
```

## 시각화

### 정칙화 경로 그림

왼쪽 패널은 $\log_{10}(\lambda)$에 대한 계수의 궤적을 그리고, 최적 $\lambda$ 위치에 세로
점선을 표시한다. 오른쪽 패널은 각 $\lambda$에서 활성(0이 아닌) 특성의 개수를 보여주어 모형
복잡도와 정칙화 사이의 절충을 드러낸다.

### 교차검증 오차 그림

$\pm 1$ 표준편차 띠를 곁들인 CV RMSE 곡선은 U자 모양이다. 최소점이 편향과 분산의 균형을
맞추는 최적 $\lambda$를 알려 준다.

### 모형 비교 그림

OLS, 능형회귀, 라쏘의 RMSE와 $R^2$ 막대그림에서 다음을 볼 수 있다.

- 세 모형 모두 이 자료에서는 비슷한 $R^2$를 낸다.
- 라쏘는 더 적은 특성으로 비슷한 예측정확도를 달성한다.

## 해석

주택 자료 분석에서 얻은 주요 결론은 다음과 같다.

- **변수선택.** 라쏘는 원래 특성 중 일부만 선택하고, 덜 중요한 예측변수(예: `YrRenovated`,
  `NbrLivingUnits`)의 계수를 0으로 만든다.
- **주요 예측변수.** `BldgGrade`(건물 등급)와 `SqFtTotLiving`(총 거주면적)이 대개 절댓값이 가장
  큰 계수를 가지며, 주택가격 예측에서의 중요성을 확인해 준다.
- **능형회귀 대 라쏘.** 능형회귀는 모든 특성을 연속적으로 축소하여 유지하고, 라쏘는 자동으로
  변수를 선택한다. 이 정도 크기의 자료에서는 예측정확도가 비슷하다.
- **실무적 이점.** 라쏘 모형이 더 해석하기 좋다. 13개가 넘는 작은 계수를 일일이 들여다보지
  않고도 어떤 특성이 가격 예측을 이끄는지 곧바로 알 수 있다.

!!! warning "표본 내 지표의 한계"
    위 코드는 훈련에 쓴 자료로 RMSE와 $R^2$를 계산한다. 표본 내 지표는 낙관적으로 편향되므로
    모형 비교의 근거로 삼기에는 부족하다. 연습문제 3에서 올바른 평가 전략을 다룬다.

## 연습문제

**연습문제 1.** 라쏘의 정칙화 경로는 왜 ($\lambda$의 함수로서) 조각별 선형인 반면 능형회귀의
경로는 매끄러운지 설명하라. 힌트: 각 방법의 KKT 조건을 생각해 보라.

??? success "연습문제 1 풀이"

    **라쏘:** 라쏘의 KKT(하위기울기) 조건은

    $$
    -\frac{1}{n}X_j^\top(y - X\hat{\beta}) + \lambda s_j = 0, \quad s_j \in \partial|\hat{\beta}_j|
    $$

    이다. 활성집합 $\mathcal{A} = \{j : \hat{\beta}_j \ne 0\}$ 위에서는 부호
    $s_j = \text{sign}(\hat{\beta}_j)$가 고정된다. 그러면 활성 계수들은 $\lambda$에 대한
    선형계를 풀게 되므로, 변수가 활성집합에 들어오거나 빠지는 분기점 사이에서
    $\hat{\beta}_{\mathcal{A}}(\lambda)$는 $\lambda$의 일차함수다. 여기서 조각별 선형 구조가
    나온다.

    **능형회귀:** 닫힌 형태의 해
    $\hat{\beta}^{\text{ridge}} = (X^\top X + \lambda I)^{-1}X^\top y$는 $\lambda$의
    일차함수인 행렬의 역행렬이므로 $\lambda$의 유리함수이고, 모든 $\lambda > 0$에서 매끄럽다
    (무한히 미분가능하다). 어떤 변수도 정확히 0이 되지 않으므로 분기점 자체가 없다. $\square$

---

**연습문제 2.** 주택 자료에서 `BldgGrade`와 `SqFtTotLiving`은 상관되어 있을 가능성이 크다. 이런
상관이 있을 때 변수선택에 라쏘를 쓰는 경우와 엘라스틱넷을 쓰는 경우가 어떻게 다른지 논하라.

??? success "연습문제 2 풀이"

    `BldgGrade`와 `SqFtTotLiving`이 강하게 상관되어 있으면 라쏘 해는 불안정하다. 자료가
    조금만 흔들려도 라쏘가 한 특성을 고르고 다른 특성을 버리는 결과가 뒤바뀔 수 있으며, 효과
    전체를 한 예측변수에 임의로 몰아줄 수도 있다.

    엘라스틱넷($0 < \alpha < 1$)은 그룹 성질을 갖는다. 강하게 상관된 두 예측변수가 모두 실제로
    관련 있다면 엘라스틱넷은 둘 다 포함하거나 둘 다 배제하는 경향이 있다. $L_2$ 성분이 해의
    유일성을 보장하고 계수 경로를 안정화하여 재현 가능한 변수선택을 낳는다.

    실무적으로는, 두 특성이 모두 중요하다는 배경지식이 있다면 엘라스틱넷이 낫다. 최대한의
    희소성이 목표이고 두 특성이 사실상 대체 가능하다면 라쏘가 하나만 고르는 것도 받아들일 만하다.
    $\square$

---

**연습문제 3.** 위 스크립트는 모형 비교에 표본 내 $R^2$와 RMSE를 쓴다. 이것이 왜 오도할 수
있는지 설명하고 더 나은 평가 전략을 제안하라.

??? success "연습문제 3 풀이"

    표본 내 지표는 훈련에 쓴 바로 그 자료로 모형을 평가하므로 낙관적으로 편향된다. OLS는
    모수가 가장 많아 선형모형 중 표본 내 잔차제곱합이 항상 가장 작고 표본 내 $R^2$가 가장 높다.
    그래서 실제로는 과적합하고 있어도 정칙화 방법과 비슷하거나 더 나아 보인다.

    **더 나은 전략:** 교차검증 지표를 쓴다. 공정한 비교를 위해서는,

    1. 모든 방법에 같은 $K$-겹 분할을 쓴다.
    2. 각 겹 안에서 훈련 겹의 통계량만으로 표준화한다.
    3. 남겨 둔 겹에서 계산한 CV RMSE와 CV $R^2$를 보고한다.

    또는 모형 선택과 훈련에 전혀 쓰지 않는 고정된 검정자료(예: 20%)를 떼어 둔다. 검정자료의
    RMSE는 일반화 성능의 불편 추정치를 준다.

    많은 자료에서 정칙화 방법은 표본 내 지표가 다소 나쁘더라도 CV 지표에서는 OLS를 능가한다.
    $\square$

---

**연습문제 4.** 주택 자료에 $N(0,1)$에서 뽑은, 반응변수와 아무 관계 없는 잡음 특성 50개를
추가한다고 하자. 라쏘의 최적 $\lambda$와 선택되는 특성 개수는 어떻게 바뀔 것으로 예상되는가?
실험을 수행하고 결과를 보고하라.

??? success "연습문제 4 풀이"

    ```python
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)
    # Assume X_scaled and y are already defined from the housing data

    # Add 50 noise features
    noise = np.random.randn(len(y), 50)
    noise_cols = [f'noise_{i}' for i in range(50)]
    X_aug = pd.concat([X_scaled.reset_index(drop=True),
                       pd.DataFrame(noise, columns=noise_cols)], axis=1)

    lasso_aug = LassoCV(n_alphas=100, cv=5, max_iter=10000)
    lasso_aug.fit(X_aug, y)

    n_nz = np.sum(np.abs(lasso_aug.coef_) > 1e-8)
    noise_selected = np.sum(np.abs(lasso_aug.coef_[-50:]) > 1e-8)

    print(f"Optimal lambda: {lasso_aug.alpha_:.4f}")
    print(f"Total nonzero:  {n_nz}")
    print(f"Noise features selected: {noise_selected}/50")
    ```

    **예상 결과:** 잡음 차원을 상쇄하기 위해 더 강한 정칙화가 필요하므로 최적 $\lambda$는
    커진다. 라쏘는 잡음 특성 50개의 대부분 또는 전부를 0으로 만들면서 실제로 예측력이 있는
    주택 특성은 유지해야 한다. 다만 몇 개의 잡음 특성은 통과할 수 있다(위양성). 이 자료는
    $n$이 $p$보다 훨씬 크므로 위양성이 많지는 않겠지만, $p$가 $n$에 가까워질수록 늘어난다.
    $\square$

---

**연습문제 5.** 라그랑주 모수 $\lambda$와, 동치인 제약형
$\min \|y - X\beta\|_2^2$ subject to $\|\beta\|_1 \le t$의 제약 경계 $t$ 사이의 관계를
유도하라. 구체적으로 $\lambda > 0$과 $t \in (0, \|\hat{\beta}^{\text{OLS}}\|_1)$ 사이에 일대일
감소 대응이 있음을 보여라.

??? success "연습문제 5 풀이"

    라쏘의 벌점형 문제와 제약형 문제는 라그랑주 쌍대성으로 연결된다.
    $f(t) = \min_{\|\beta\|_1 \le t} \|y - X\beta\|_2^2$라 하자. ($X$가 완전계수라는 가정
    아래) $\|y - X\beta\|_2^2$는 $\beta$에 대해 강볼록이고 제약 $\|\beta\|_1 \le t$는
    볼록이므로, KKT 조건에 의해 모수 $\lambda$의 벌점형 해는 어떤 $t(\lambda)$의 제약형 해와
    일치한다.

    KKT의 상보여유 조건 $\lambda(\|\hat{\beta}\|_1 - t) = 0$에 의해, $\lambda > 0$이면 제약이
    활성이다. 즉 $\|\hat{\beta}\|_1 = t$이다.

    $\lambda$가 커지면 벌점이 계수를 더 강하게 축소하므로 $\|\hat{\beta}(\lambda)\|_1$은
    감소한다. $t(\lambda) = \|\hat{\beta}(\lambda)\|_1$이므로 $t$는 $\lambda$의 감소함수다.

    - $\lambda = 0$일 때 $t = \|\hat{\beta}^{\text{OLS}}\|_1$.
    - $\lambda \to \infty$일 때 $t \to 0$.

    대응이 일대일인 근거는 라쏘 경로가 연속이고 $\|\hat{\beta}(\lambda)\|_1$이 $\lambda$에 대해
    (해가 0이 되기 전까지) 강감소한다는 데 있다.

    !!! warning "흔한 오해"
        $\|\hat{\beta}(\lambda)\|_1$은 $\lambda$에 대해 단조 비증가지만, **개별 계수의 절댓값은
        단조가 아니다.** 어떤 변수가 활성집합에 새로 들어오면 다른 변수의 계수가 오히려
        커지기도 한다. 그러므로 위 논증은 $L_1$ 노름 전체에 대해서만 성립하며, 좌표별 축소의
        단조성을 주장해서는 안 된다.
    $\square$
