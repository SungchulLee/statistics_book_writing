# 회귀 진단 (주택)

## 개요

이 페이지는 King County 주택 자료를 이용해 회귀 진단의 전체 흐름을 보인다. 이상점 탐지를 위한 스튜던트화 잔차, 영향 분석을 위한 Cook 거리와 모자값, Breusch-Pagan 검정을 통한 이분산 검정, 부분잔차 그림, 그리고 영향점 제거가 모형 추정에 미치는 영향을 다룬다.

!!! note "자료에 대하여"
    `house_98105`는 King County 주택 매매 자료 가운데 우편번호 98105 지역만 걸러 낸 부분자료를 담은 pandas `DataFrame`이다. 아래 코드는 이 변수가 이미 만들어져 있다고 가정한다. 다른 자료로 바꾸어도 진단 절차 자체는 그대로 적용된다.

## 수학적 배경

### 스튜던트화 잔차

관측값 $i$의 내부 스튜던트화 잔차는

$$
r_i = \frac{e_i}{s\sqrt{1 - h_{ii}}},
$$

여기서 $e_i = y_i - \hat{y}_i$는 원잔차이고 $h_{ii}$는 지렛대(모자행렬 $\mathbf{H} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$의 대각원소)이다. $|r_i| > 2.5$인 관측값은 잠재적 이상점이다.

### 모자값(지렛대)

지렛대 $h_{ii}$는 관측값 $i$가 설명변수 공간의 중심에서 얼마나 떨어져 있는지를 잰다. 지렛대가 크다는 것은 $\mathbf{x}_i$가 특이하다는 뜻이다. $h_{ii} > 2k/n$이거나 $h_{ii} > 3\bar{h}$($\bar{h} = k/n$)인 관측값을 높은 지렛대 점으로 본다.

### Cook 거리

Cook 거리는 잔차의 크기와 지렛대를 결합한다.

$$
D_i = \frac{1}{k}\,r_i^2\,\frac{h_{ii}}{1 - h_{ii}}.
$$

동등하게 $D_i$는 관측값 $i$를 뺐을 때 모든 적합값이 변하는 정도를 잰다. 흔한 문턱값은 $D_i > 4/n$이다.

### Breusch-Pagan 검정

이분산에 대한 Breusch-Pagan 검정은 제곱잔차를 설명변수에 회귀시킨다.

$$
e_i^2 = \gamma_0 + \gamma_1 x_{i1} + \cdots + \gamma_p x_{ip} + v_i.
$$

$H_0$(등분산) 아래에서 이 보조회귀의 검정통계량 $nR^2$은 $\chi^2_p$를 따른다.

## 자료

King County(시애틀) 주택 매매 자료에서 우편번호 98105 지역만 골라 쓴다.

<div class="codebox" markdown>

### 예제 1. 주택 자료 읽기 { .eg }

```python
import pandas as pd

# "Practical Statistics for Data Scientists" 저장소의 자료. 탭으로 구분되어 있다.
url = ("https://raw.githubusercontent.com/gedeck/"
       "practical-statistics-for-data-scientists/master/data/house_sales.csv")
house = pd.read_csv(url, sep='\t')
house_98105 = house.loc[house['ZipCode'] == 98105, :]

print(f"전체 {len(house)}건 중 98105 지역 {len(house_98105)}건")
print(house_98105[['AdjSalePrice', 'SqFtTotLiving', 'Bedrooms']].describe().round(1).to_string())
```

출력:

```
전체 22687건 중 98105 지역 313건
       AdjSalePrice  SqFtTotLiving  Bedrooms
count         313.0          313.0     313.0
mean       756277.3         2069.9       3.4
std        391463.3          905.4       1.1
min        119748.0          490.0       1.0
25%        521101.0         1410.0       3.0
50%        630041.0         1850.0       3.0
75%        851954.0         2570.0       4.0
max       3013254.0         5570.0       9.0
```

98105 지역 313건이다. 가격이 12만에서 301만 달러까지 25배 차이가 나고 표준편차가 평균의 절반이 넘는다. 이렇게 퍼진 자료에서는 이분산과 영향점이 함께 나타나기 쉽다.

</div>

### 기준 모형과 영향 진단

<div class="codebox" markdown>

#### 예제 2. 영향점 진단량 구하기 { .eg }

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

# assign(const=1) 이 절편 열을 만든다. statsmodels 의 OLS 는 절편을
# 자동으로 넣지 않는다.
predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade']
X = house_98105[predictors].assign(const=1)
y = house_98105['AdjSalePrice']

results = sm.OLS(y, X).fit()
influence = OLSInfluence(results)

# 세 진단량을 함께 본다. 스튜던트화 잔차는 그 점이 얼마나 벗어났는지,
# 지렛값은 설명변수 쪽에서 얼마나 외따로 있는지, Cook 거리는 그 둘을
# 합쳐 결론을 얼마나 흔드는지를 잰다.
studentized_resids = influence.resid_studentized_internal
hat_values = influence.hat_matrix_diag
cooks_dist, _ = influence.cooks_distance

print(results.summary().tables[1])
print(f"n = {len(y)}, 문턱 4/n = {4/len(y):.4f}")
print(f"Cook 거리 최댓값 = {cooks_dist.max():.4f} (관측 {cooks_dist.argmax()})")
print(f"문턱을 넘는 관측값 수 = {(cooks_dist > 4/len(y)).sum()}")
```

출력:

```
=================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
SqFtTotLiving   209.6023     24.408      8.587      0.000     161.574     257.631
SqFtLot          38.9333      5.330      7.305      0.000      28.445      49.421
Bathrooms      2282.2641      2e+04      0.114      0.909    -3.7e+04    4.16e+04
Bedrooms      -2.632e+04   1.29e+04     -2.043      0.042   -5.17e+04    -973.867
BldgGrade        1.3e+05   1.52e+04      8.533      0.000       1e+05     1.6e+05
const         -7.725e+05   9.83e+04     -7.861      0.000   -9.66e+05   -5.79e+05
=================================================================================
n = 313, 문턱 4/n = 0.0128
Cook 거리 최댓값 = 0.5608 (관측 152)
문턱을 넘는 관측값 수 = 20
```

계수를 보면 거주면적 1제곱피트당 210달러, 건물 등급 한 단계당 13만 달러다. 침실 수의 계수가 **음수**($-26{,}320$)인 것이 눈에 띄는데, 면적을 고정한 채 침실을 늘리면 방이 작아지므로 값이 떨어진다는 뜻이다. 다중회귀 계수를 "다른 변수를 고정한 채"로 읽어야 하는 이유다.

Cook 거리가 문턱 0.0128을 넘는 관측값이 20개이고, 그중 152번이 0.5608로 압도적이다.

</div>

### 영향점 제거의 효과

<div class="codebox" markdown>

#### 예제 3. 영향점을 빼고 다시 적합 { .eg }

```python
# 문턱을 넘는 관측값을 빼고 다시 적합해 계수가 얼마나 달라지는지 본다.
# 크게 달라진다면 결론이 몇 채의 집에 기대고 있다는 뜻이다.
threshold_cooks = 4 / len(y)
mask_keep = cooks_dist < threshold_cooks

X_filtered = X[mask_keep]
y_filtered = y[mask_keep]
results_filtered = sm.OLS(y_filtered, X_filtered).fit()

# 계수를 견준다
comparison = pd.DataFrame({
    'Original': results.params,
    'Filtered': results_filtered.params,
})
print(comparison.round(2).to_string())
print(f"\n제거된 관측값: {(~mask_keep).sum()}건")
print(f"R^2: {results.rsquared:.4f} -> {results_filtered.rsquared:.4f}")
```

출력:

```
                Original   Filtered
SqFtTotLiving     209.60     201.83
SqFtLot            38.93      42.59
Bathrooms        2282.26    1664.95
Bedrooms       -26320.27  -23734.89
BldgGrade      130000.10  111175.45
const         -772549.86 -644099.89

제거된 관측값: 20건
R^2: 0.7954 -> 0.8415
```

영향점 20건을 빼면 $R^2$가 0.795에서 0.842로 오르고 계수도 눈에 띄게 움직인다. BldgGrade의 계수가 13.0만에서 11.1만으로 14% 줄었다.

이만큼 움직인다는 것 자체가 보고해야 할 사실이다. 그렇다고 20건을 그냥 버려서는 안 된다. Cook 거리가 큰 관측값은 자료 오류일 수도, 정말로 특이한 거래(예: 재건축 예정 부지)일 수도 있으므로 개별적으로 확인해야 한다.

</div>

### 이분산 검정

<div class="codebox" markdown>

#### 예제 4. 등분산 검정 { .eg }

```python
from statsmodels.stats.diagnostic import het_breuschpagan

# 집값 자료는 비싼 집일수록 오차도 커지는 것이 보통이라, 등분산 검정이
# 기각되는 일이 흔하다. 그럴 때는 로그 변환이나 로버스트 표준오차로 간다.
bp_stat, bp_pval, _, _ = het_breuschpagan(results.resid, X)
print(f"Breusch-Pagan p-value: {bp_pval:.4f}")
```

출력:

```
Breusch-Pagan p-value: 0.0000
```

$p < 0.0001$로 등분산을 강하게 기각한다. 주택 가격 자료에서 흔한 일이다. 비싼 집일수록 가격의 변동폭도 커지기 때문이며, 로그 변환이나 로버스트 표준오차가 표준적인 처방이다.

</div>

## 해석

- **스튜던트화 잔차**: 값이 크면 모형이 그 관측값을 잘 예측하지 못한다는 뜻이다. 자료 입력 오류, 특이한 사례, 혹은 모형 오설정의 신호일 수 있다.
- **지렛대**: 지렛대가 큰 점이 반드시 해로운 것은 아니다. 잔차까지 클 때에만 영향점이 된다. 회귀 곡면 위에 놓인 높은 지렛대 점은 오히려 적합을 안정시킨다.
- **Cook 거리**: Cook 거리가 큰 점은 회귀 곡면 전체에 영향을 준다. 그 점들을 빼고 결과를 비교해 보면 결론이 몇몇 관측값에 민감한지 드러난다.
- **Breusch-Pagan 검정**: 유의한 결과($p < 0.05$)는 이분산을 나타내며 OLS 표준오차를 믿을 수 없음을 시사한다. 대책으로는 WLS, 로버스트 표준오차, 분산 안정화 변환이 있다.
- **부분잔차 그림**(CCPR 그림)은 다른 설명변수를 고려한 뒤 각 설명변수와 반응변수의 관계를 보여주어 잠재적 비선형성을 드러낸다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span> (하나 빼기 분산 추정을 쓰는) 외부 스튜던트화 잔차를 계산하여 내부 스튜던트화 잔차와 비교하라. 어떤 관측값에서 차이가 가장 큰가?

</div>

??? success "풀이"

    ```python
    ext_resids = influence.resid_studentized_external
    int_resids = influence.resid_studentized_internal
    discrepancy = np.abs(ext_resids - int_resids)
    top_idx = np.argsort(discrepancy)[-5:]
    ```

    차이가 가장 큰 곳은 잔차가 크면서 $h_{ii}$도 큰 관측값이다. 외부 잔차는 관측값 $i$를 빼고 계산한 $s_{(i)}$를 쓰는데, 관측값 $i$가 잔차분산에 강한 영향을 줄수록 $s$와 $s_{(i)}$의 차이가 커지기 때문이다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 모든 모자값의 합이 $k$(모수의 개수)임을 증명하라. 이는 평균 지렛대에 대해 무엇을 뜻하는가?

</div>

??? success "풀이"

    모자행렬은 $\mathbf{H} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$이다. $\mathbf{H}$가 멱등($\mathbf{H}^2 = \mathbf{H}$)이고 대칭이므로

    $$
    \sum_{i=1}^n h_{ii} = \operatorname{tr}(\mathbf{H}) = \operatorname{tr}(\mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top) = \operatorname{tr}((\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{X}) = \operatorname{tr}(\mathbf{I}_k) = k.
    $$

    따라서 평균 지렛대는 $\bar{h} = k/n$이고, 문턱값 $3\bar{h} = 3k/n$은 평균의 세 배가 넘는 지렛대를 가진 점을 찾아낸다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> $e_i^2$을 $\mathbf{X}$에 회귀시키고 $nR^2$을 계산하여 Breusch-Pagan 검정을 직접 구현하라. statsmodels 결과와 일치하는지 확인하라.

</div>

??? success "풀이"

    ```python
    resid_sq = results.resid ** 2
    aux_model = sm.OLS(resid_sq, X).fit()
    bp_stat_manual = len(y) * aux_model.rsquared
    from scipy import stats
    bp_pval_manual = 1 - stats.chi2.cdf(bp_stat_manual, X.shape[1] - 1)
    ```

    직접 계산한 값은 `het_breuschpagan`과 거의 일치한다(상수항 처리 같은 구현 세부에서 미세한 차이가 날 수 있다). 자유도는 상수를 제외한 설명변수의 개수이므로 `X.shape[1] - 1`을 쓴다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> 영향점을 제거하면 계수 추정값이 달라진다. 그런 관측값을 제거하는 것이 적절한 경우와 남겨 두어야 하는 경우를 논하라.

</div>

??? success "풀이"

    영향점이 명백히 잘못된 값(자료 입력 실수, 측정 실패)이거나 다른 모집단에서 온 것이라면 제거가 적절하다. 특이하긴 하지만 정당한 자료점이라면 남겨야 한다. 이 경우 민감도 분석 자체가 유익한 정보를 준다. 결과가 크게 달라진다면 결론이 취약하다는 뜻이다. 제거의 대안으로는 영향점을 완전히 배제하지 않고 가중치를 낮추는 로버스트 회귀(예: Huber나 bisquare 가중)가 있다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> 참 오차가 정규분포를 따르는데도 Q-Q 그림의 꼬리가 대각선에서 자주 벗어나는 이유를 설명하라. 표본크기는 이 현상에 어떤 영향을 주는가?

</div>

??? success "풀이"

    Q-Q 그림에서 극단의 순서통계량(가장 작은 잔차와 가장 큰 잔차)은 표집변동이 가장 크다. 오차가 정확히 정규여도 유한표본에서는 경험분포의 꼬리가 상당히 흔들린다. 관측값이 $n$개일 때 정규표본의 범위는 대략 $2\sqrt{2\ln n}\,\sigma$이지만 실제 범위는 크게 변동한다. $n$이 커지면 (큰 수의 법칙이 경험분위수를 안정시키므로) Q-Q 그림의 꼬리가 더 안정되고, 따라서 큰 표본의 Q-Q 그림이 꼬리 거동에 대해 더 믿을 만한 증거를 준다. $\square$

---

## 정리하며

실제 주택 자료로 **진단 전 과정**을 밟았다.

- **스튜던트화 잔차로 이상점을, 쿡 거리와 모자값으로 영향점을 찾는다.** 두 지표를 산점도로 함께 그리면 어느 점이 왜 위험한지가 보인다.
- **브로이시–페이건 검정이 이분산을 확인한다.** 주택 가격처럼 규모가 다양한 자료에서는 거의 언제나 기각되며, 그때 로그 변환이나 로버스트 표준오차로 간다.
- **부분잔차 그림이 개별 변수의 관계 모양을 드러낸다.** 전체 잔차 그림이 깨끗해 보여도 특정 변수와는 곡선 관계일 수 있다.
- **영향점을 뺀 전후를 비교한다.** 계수가 크게 달라지면 그 결론이 몇 개의 관측에 달려 있다는 뜻이며, **반드시 보고해야 할 사실**이다.
- **실제 자료는 언제나 가정을 어긴다.** 문제는 "가정이 성립하는가"가 아니라 **"위반의 정도가 결론을 바꾸는가"** 이며, 그 판단이 진단의 목적이다.

다음 절부터 **성능 척도**로 넘어간다.
