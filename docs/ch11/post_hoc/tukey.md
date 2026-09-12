# 사후비교: Tukey HSD

## 1. 일원배치 분산분석의 사후검정

**일원배치 분산분석**을 수행하면 집단 평균 사이에 유의한 차이가 있음을 알게 될 수 있다. 그러나 분산분석의 유의한 결과는 구체적으로 어느 집단이 서로 다른지는 알려주지 않는다. 이때 유의하게 다른 집단 쌍을 찾아내는 데 **사후검정**을 쓴다. 이 검정들은 여러 비교를 할 때 제1종 오류(거짓 양성)를 통제하도록 돕는다.

### A. 일원배치 분산분석의 사후검정 이해하기

일원배치 분산분석에서 사후검정은 다음일 때 수행한다:

1. 일원배치 분산분석이 집단 평균 사이에 통계적으로 유의한 차이를 나타냈을 때.
2. 구체적으로 어느 집단이 서로 다른지 알고자 할 때.

### B. 일원배치 분산분석의 사후검정 종류

일원배치 분산분석에서 가장 흔한 사후검정은 다음과 같다:

- **Tukey의 정직유의차(HSD)**: 가족단위 오류율을 통제하는 널리 쓰이는 방법. 집단 크기가 같을 때 적합하지만 약간의 불균형에도 쓸 수 있다.
- **Bonferroni 보정**: 유의수준을 비교 횟수로 나누는 보수적인 접근. 비교 횟수가 적거나 엄격한 오류 통제가 필요할 때 적합하다.
- **Scheffé 검정**: 유연하고 보수적인 검정으로, 쌍별 비교를 넘어선 복잡한 비교를 검정할 때 특히 유용하다.
- **Dunnett 검정**: 여러 처치군을 하나의 대조군과 비교할 때 쓴다.

### C. Python으로 사후검정 수행하기

#### 1단계: 일원배치 분산분석 수행

셋 이상의 집단이 있는 자료에서 평균 사이에 유의한 차이가 있는지 검정한다고 하자.

```python
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Load sample data
url = 'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/PlantGrowth.csv'
df = pd.read_csv(url, usecols=[1, 2])

# Perform one-way ANOVA
model = ols('weight ~ C(group)', data=df).fit()
anova_results = anova_lm(model)
print("One-Way ANOVA Results:")
print(anova_results)
```

출력:

```
One-Way ANOVA Results:
            df    sum_sq   mean_sq         F   PR(>F)
C(group)   2.0   3.76634  1.883170  4.846088  0.01591
Residual  27.0  10.49209  0.388596       NaN      NaN
```

전역 검정이 $p = 0.0159$로 기각한다. 이제 어느 쌍이 다른지 찾을 차례다.

#### 2단계: Tukey의 HSD를 이용한 사후검정

일원배치 분산분석이 유의하면 Tukey의 HSD로 어느 집단 쌍이 유의하게 다른지 찾을 수 있다.

```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Perform Tukey's HSD test
tukey_result = pairwise_tukeyhsd(endog=df['weight'], groups=df['group'], alpha=0.05)
print("Tukey's HSD Test Results:")
print(tukey_result)
```

출력:

```
Tukey's HSD Test Results:
Multiple Comparison of Means - Tukey HSD, FWER=0.05
===================================================
group1 group2 meandiff p-adj   lower  upper  reject
---------------------------------------------------
  ctrl   trt1   -0.371 0.3909 -1.0622 0.3202  False
  ctrl   trt2    0.494  0.198 -0.1972 1.1852  False
  trt1   trt2    0.865  0.012  0.1738 1.5562   True
---------------------------------------------------
```

세 비교 중 trt1 대 trt2 하나만 유의하다. `p-adj` 열은 이미 다중비교 보정을 마친 값이므로 그대로 0.05와 비교하면 된다.

이 출력은 각 집단 쌍의 비교 결과를 보여주며 다음을 포함한다:

- **meandiff**: 두 집단 평균의 차이.
- **p-adj**: 각 쌍별 비교의 조정 p-값.
- **reject**: 각 쌍에 대해 귀무가설(차이 없음)을 기각했는지를 나타내는 불리언.

#### 3단계: Bonferroni 보정 (Tukey HSD의 대안)

더 보수적인 접근으로, 유의수준을 비교 횟수로 나누는 Bonferroni 보정을 쓸 수 있다.

```python
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from scipy.stats import ttest_ind

# Define groups
groups = df['group'].unique()

# Perform pairwise t-tests and apply Bonferroni correction
p_values = []
comparisons = []

for group1, group2 in combinations(groups, 2):
    data1 = df[df['group'] == group1]['weight']
    data2 = df[df['group'] == group2]['weight']
    stat, p_val = ttest_ind(data1, data2)
    p_values.append(p_val)
    comparisons.append(f"{group1} vs {group2}")

# Apply Bonferroni correction
_, p_values_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')

# Display Bonferroni-corrected results
print("Bonferroni-Corrected Pairwise Comparisons:")
for comparison, p_val, p_val_corr in zip(comparisons, p_values, p_values_corrected):
    print(f"{comparison}: p-value = {p_val:.4f}, Bonferroni-corrected p-value = {p_val_corr:.4f}")
```

출력:

```
Bonferroni-Corrected Pairwise Comparisons:
ctrl vs trt1: p-value = 0.2490, Bonferroni-corrected p-value = 0.7471
ctrl vs trt2: p-value = 0.0469, Bonferroni-corrected p-value = 0.1406
trt1 vs trt2: p-value = 0.0075, Bonferroni-corrected p-value = 0.0226
```

Tukey와 결론은 같지만 보정 p-값이 다르다. trt1 대 trt2가 Tukey에서 0.012, Bonferroni에서 0.0226이다. Bonferroni가 더 보수적이기 때문이며, 비교 수가 늘수록 차이가 벌어진다.

보정 전 p-값이 Tukey의 `p-adj`와도 다르다는 점에 주의하라. 여기서는 쌍마다 두 집단의 자료만으로 $t$-검정을 하지만, Tukey는 세 집단 전체에서 얻은 합동 MSE를 쓴다. 자유도가 18 대 27로 달라진다.

#### 4단계: Scheffé 검정 (복잡한 비교용)

Scheffé 검정은 쌍별이 아닌 비교나 대비를 검정하는 데 적합하지만 복잡하고 기본적인 쌍별 비교에는 덜 쓰인다. `statsmodels`에는 Scheffé 검정이 직접 제공되지 않지만, 필요하다면 특히 더 진전된 비교를 위해 대비를 직접 구성할 수 있다.

## 2. scipy.stats.tukey_hsd

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd

def load_data():
    """
    Load and preprocess plant growth data for ANOVA and posthoc testing.
    """
    url = 'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/PlantGrowth.csv'
    df = pd.read_csv(url, usecols=[1, 2])

    grouped_data = df.groupby('group')
    data_ctrl = grouped_data.get_group('ctrl').weight
    data_trt1 = grouped_data.get_group('trt1').weight
    data_trt2 = grouped_data.get_group('trt2').weight
    data = (data_ctrl, data_trt1, data_trt2)

    total_samples = data_ctrl.shape[0] + data_trt1.shape[0] + data_trt2.shape[0]
    num_groups = len(data)
    df1 = num_groups - 1
    df2 = total_samples - num_groups

    return df, data, df1, df2

def perform_anova(data_ctrl, data_trt1, data_trt2):
    """
    Perform one-way ANOVA on the data groups.
    """
    statistic, p_value = stats.f_oneway(data_ctrl, data_trt1, data_trt2)
    print("\nOne-way ANOVA Results:")
    print(f"F-statistic = {statistic:.4f}")
    print(f"P-value = {p_value:.4f}\n")
    return statistic, p_value

def perform_tukey_hsd(data_ctrl, data_trt1, data_trt2, confidence_level=0.95):
    """
    Perform Tukey's HSD posthoc test and display confidence intervals.
    """
    result = stats.tukey_hsd(data_ctrl, data_trt1, data_trt2)
    print(result)

    print(f"\nTukey's HSD Pairwise Group Comparisons ({confidence_level:.0%} Confidence Interval)")
    print("Comparison    Lower CI   Upper CI")
    confidence_interval = result.confidence_interval(confidence_level=confidence_level)
    for ((i, j), low) in np.ndenumerate(confidence_interval.low):
        if i < j:
            high = confidence_interval.high[i, j]
            print(f" ({i} - {j})   {low:>10.3f}   {high:>9.3f}")
    print()

# Load data, perform ANOVA, and conduct Tukey's HSD posthoc tests
df, (data_ctrl, data_trt1, data_trt2), df1, df2 = load_data()

# Conduct one-way ANOVA
statistic, p_value = perform_anova(data_ctrl, data_trt1, data_trt2)

# Conduct Tukey's HSD posthoc tests at default 95% confidence level
perform_tukey_hsd(data_ctrl, data_trt1, data_trt2)

# Conduct Tukey's HSD posthoc tests at a 99% confidence level
perform_tukey_hsd(data_ctrl, data_trt1, data_trt2, confidence_level=0.99)
```

출력:

```
One-way ANOVA Results:
F-statistic = 4.8461
P-value = 0.0159

Tukey's HSD Pairwise Group Comparisons (95.0% Confidence Interval)
Comparison  Statistic  p-value  Lower CI  Upper CI
 (0 - 1)      0.371     0.391    -0.320     1.062
 (0 - 2)     -0.494     0.198    -1.185     0.197
 (1 - 0)     -0.371     0.391    -1.062     0.320
 (1 - 2)     -0.865     0.012    -1.556    -0.174
 (2 - 0)      0.494     0.198    -0.197     1.185
 (2 - 1)      0.865     0.012     0.174     1.556


Tukey's HSD Pairwise Group Comparisons (95% Confidence Interval)
Comparison    Lower CI   Upper CI
 (0 - 1)       -0.320       1.062
 (0 - 2)       -1.185       0.197
 (1 - 2)       -1.556      -0.174

Tukey's HSD Pairwise Group Comparisons (95.0% Confidence Interval)
Comparison  Statistic  p-value  Lower CI  Upper CI
 (0 - 1)      0.371     0.391    -0.320     1.062
 (0 - 2)     -0.494     0.198    -1.185     0.197
 (1 - 0)     -0.371     0.391    -1.062     0.320
 (1 - 2)     -0.865     0.012    -1.556    -0.174
 (2 - 0)      0.494     0.198    -0.197     1.185
 (2 - 1)      0.865     0.012     0.174     1.556


Tukey's HSD Pairwise Group Comparisons (99% Confidence Interval)
Comparison    Lower CI   Upper CI
 (0 - 1)       -0.515       1.257
 (0 - 2)       -1.380       0.392
 (1 - 2)       -1.751       0.021
```

`scipy.stats.tukey_hsd`는 statsmodels와 달리 대칭인 쌍을 모두 인쇄한다. `(0 - 1)`과 `(1 - 0)`이 부호만 반대인 같은 비교다.

집단 번호는 인자를 넘긴 순서(0 = ctrl, 1 = trt1, 2 = trt2)를 따른다. 이름이 아니라 번호로 나오므로 순서를 잘못 기억하면 결과를 거꾸로 읽게 된다.

결과 자체는 statsmodels의 Tukey HSD와 정확히 같다. `(1 - 2)`의 차이 $-0.865$, $p = 0.012$, 구간 $(-1.556, -0.174)$가 앞의 표와 일치한다.

### 출력 해석

이 표는 세 집단(집단 0, 1, 2로 표기)에 대한 Tukey의 HSD 쌍별 비교 결과를 95% 신뢰구간과 함께 보여준다.

**열 설명:**

1. **비교**: 비교하는 집단 쌍을 지표 번호로 나타낸다. 예를 들어 "(0 - 1)"은 집단 0과 집단 1의 비교를 뜻한다.

2. **통계량**: 해당 비교에서 두 집단 평균의 차이. 값이 양수이면 앞 집단의 평균이 더 크고, 음수이면 그 반대이다.

3. **p-값**: 그 비교에 대응하는 확률값. 평균 차이가 통계적으로 유의한지를 나타낸다. p-값이 작으면(보통 0.05 미만) 집단 사이에 통계적으로 유의한 차이가 있음을 시사한다.

4. **하한 CI**: 평균 차이에 대한 95% 신뢰구간의 하한. 이 구간이 0을 포함하지 않으면 차이가 통계적으로 유의하다고 본다.

5. **상한 CI**: 평균 차이에 대한 95% 신뢰구간의 상한.

**각 비교의 해석:**

- **(0 - 1)과 (1 - 0)**: 집단 0과 집단 1의 평균 차이는 0.371이고 p-값은 0.391이다. p-값이 크므로 차이가 통계적으로 유의하지 않으며, 신뢰구간(-0.320에서 1.062)도 0을 포함한다.

- **(0 - 2)와 (2 - 0)**: 집단 0과 집단 2의 평균 차이는 -0.494이고 p-값은 0.198이다. 역시 유의하지 않으며 신뢰구간(-1.185에서 0.197)이 0을 포함하여 유의한 차이가 없음을 시사한다.

- **(1 - 2)와 (2 - 1)**: 집단 1과 집단 2의 평균 차이는 -0.865이고 p-값은 0.012이다. p-값이 0.05보다 작아 통계적으로 유의한 차이를 나타낸다. 신뢰구간(-1.556에서 -0.174)이 0을 포함하지 않아 유의성을 다시 확인해 준다.

**요약:**

Tukey의 HSD 결과는 95% 신뢰수준에서 집단 1과 집단 2 사이에 통계적으로 유의한 차이가 있음을 시사한다. p-값이 작고 신뢰구간이 0을 포함하지 않기 때문이다. 집단 0과 1, 집단 0과 2 사이에는 유의한 차이가 없다.

### 위치로 본 해석

1. **집단 1(왼쪽)과 집단 2(오른쪽)**: 두 집단은 통계적으로 유의한 차이를 보인다. p-값이 작고(0.012) 신뢰구간(-1.556에서 -0.174)이 0을 포함하지 않는다. 평균값 면에서 서로 꽤 구별됨을 시사한다.

2. **집단 0(가운데)과 집단 1(왼쪽)**: 통계적으로 유의한 차이가 없다. p-값이 0.391이고 신뢰구간(-0.320에서 1.062)이 0을 포함한다.

3. **집단 0(가운데)과 집단 2(오른쪽)**: 마찬가지로 유의한 차이가 없다. p-값이 0.198이고 신뢰구간(-1.185에서 0.197)도 0을 포함한다.

**해석:**

- **집단 0(가운데)은 집단 1과 2 사이의 중간**으로 보인다. 어느 쪽과도 유의하게 다르지 않기 때문이다.
- **집단 1(왼쪽)과 집단 2(오른쪽)는 서로 유의하게 다르며**, 서로 구별되는 수준이나 조건을 대표함을 시사한다.
- 집단 0은 중간 혹은 과도기적 집단 역할을 하며 집단 1과도 집단 2와도 유의하게 다르지 않다.

### scipy.stats.tukey_hsd는 각 쌍에 대해 쌍별 t-검정을 하는가?

1. **목적과 쓰임의 맥락**: Tukey의 HSD 검정은 특별히 **사후**검정이다. 즉 분산분석이 집단 평균 사이에 통계적으로 유의한 차이가 있다고 판정한 뒤에 적용한다. "어느 집단 쌍의 평균이 유의하게 다른가?"라는 질문에 답한다. 분산분석과 무관하게 독립적으로 수행할 수 있는 t-검정과는 다르다.

2. **스튜던트화 범위 분포**: **스튜던트화 범위 분포**에 의존한다는 점이 핵심적인 차이이다. 이 분포는 (t-검정처럼 개별 쌍을 따로 평가하는 대신) 모든 집단 평균의 범위를 동시에 고려한다. 비교하는 집단의 수를 반영하여 가능한 모든 비교에 걸쳐 **가족단위 오류율**(FWER)을 통제한다.

3. **가족단위 오류율**: Tukey의 HSD가 하는 조정은 모든 비교에 걸쳐 제1종 오류를 적어도 한 번 범할 확률이 미리 정한 알파 수준(예: 0.05)을 넘지 않도록 보장한다. 반면 t-검정을 여러 번 수행하면 비교 횟수와 함께 오류율이 누적되어 제1종 오류의 가능성이 커진다.

    예를 들어 집단이 $m$개면 쌍별 비교의 수는 $\binom{m}{2} = \frac{m(m-1)}{2}$이다. $m = 5$이면 비교가 10개이다. 이를 각각 5% 유의수준에서 독립적으로 수행하면 전체 제1종 오류율이 40%를 넘을 수도 있다. Tukey의 HSD는 다중검정 보정을 넣어 이 부풀림을 막는다.

4. **임계값과 해석**: Tukey 검정은 모든 집단 비교에 일률적으로 적용되는 하나의 임계 차이값(HSD)을 계산한다. 두 집단 평균의 절대 차이가 이 HSD 값을 넘으면 차이가 유의하다고 본다. 이 균일한 문턱은 해석을 단순하게 하고, 각자 임계값을 갖는 개별 t-검정에서 생기는 변동을 피한다.

5. **보수성**: 가족단위 오류를 통제하기 때문에 Tukey의 HSD는 개별 t-검정보다 대체로 더 **보수적**이다. 제1종 오류의 위험을 줄이지만 제2종 오류(참 차이를 놓칠 위험)는 다소 커질 수 있다. 그래도 거짓 양성 통제가 우선인 연구에서는 이 맞바꿈이 흔히 받아들일 만하다.

**실무적 함의**: Tukey의 HSD는 균형 설계(집단 크기가 같은 경우)에 이상적이지만 약간의 조정으로 불균형 설계에도 적용할 수 있다. 그런 경우에는 정확도를 위해 Games-Howell 검정 같은 다른 사후검정이 선호될 수 있다.

## 3. 이원배치 분산분석의 사후검정

이원배치 분산분석의 사후검정은 유의한 주효과와 교호작용 효과를 자세히 살피는 데 꼭 필요하다.

### A. 이원배치 분산분석에서 사후검정을 언제 쓰는가

이원배치 분산분석에서 사후검정은 대체로 다음에 쓴다:

1. **주효과 조사**: 주효과 중 하나 또는 둘 다(예: 요인 A나 요인 B) 유의하면, 사후검정으로 그 요인의 어느 수준이 서로 유의하게 다른지 찾을 수 있다.
2. **교호작용 효과 검토**: 요인 A와 B 사이에 유의한 교호작용이 있으면, 사후검정으로 어떤 요인 수준 조합에서 유의한 차이가 나타나는지 판정할 수 있다.

### B. 이원배치 분산분석의 사후검정 종류

- **Tukey의 정직유의차(HSD)**: 가족단위 오류율을 통제하기 때문에 분산분석의 쌍별 비교에 널리 쓰인다.
- **Bonferroni 보정**: 유의수준을 비교 횟수로 나누는 더 보수적인 방법.
- **단순 효과 분석**: 교호작용 효과가 유의하면, 다른 요인의 각 수준에서 한 요인의 효과를 살피는 단순 효과 분석을 쓸 수 있다.

### C. Python으로 사후검정 수행하기

#### 1단계: 이원배치 분산분석 수행

```python
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Load the dataset
url = 'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/ToothGrowth.csv'
df = pd.read_csv(url, usecols=[1, 2, 3])

# Define and fit the two-way ANOVA model
model = ols('len ~ C(supp) + C(dose) + C(supp):C(dose)', data=df).fit()
anova_results = anova_lm(model)
print(anova_results)
```

출력:

```
                   df       sum_sq      mean_sq          F        PR(>F)
C(supp)           1.0   205.350000   205.350000  15.571979  2.311828e-04
C(dose)           2.0  2426.434333  1213.217167  91.999965  4.046291e-18
C(supp):C(dose)   2.0   108.319000    54.159500   4.106991  2.186027e-02
Residual         54.0   712.106000    13.187148        NaN           NaN
```

두 주효과와 교호작용이 모두 유의하다. 교호작용이 유의하다는 것은 주효과를 단독으로 해석하기 전에 조심하라는 신호다.

#### 2단계: 주효과에 대한 사후검정

```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Tukey's HSD for the main effect of dose
tukey_dose = pairwise_tukeyhsd(endog=df['len'], groups=df['dose'], alpha=0.05)
print("Post-Hoc Test for Dose:")
print(tukey_dose)

# Tukey's HSD for the main effect of supplement
tukey_supp = pairwise_tukeyhsd(endog=df['len'], groups=df['supp'], alpha=0.05)
print("Post-Hoc Test for Supplement:")
print(tukey_supp)
```

출력:

```
Post-Hoc Test for Dose:
Multiple Comparison of Means - Tukey HSD, FWER=0.05
===================================================
group1 group2 meandiff p-adj  lower   upper  reject
---------------------------------------------------
   0.5    1.0     9.13   0.0  5.9018 12.3582   True
   0.5    2.0   15.495   0.0 12.2668 18.7232   True
   1.0    2.0    6.365   0.0  3.1368  9.5932   True
---------------------------------------------------
Post-Hoc Test for Supplement:
Multiple Comparison of Means - Tukey HSD, FWER=0.05
=================================================
group1 group2 meandiff p-adj  lower  upper reject
-------------------------------------------------
    OJ     VC     -3.7 0.0604 -7.567 0.167  False
-------------------------------------------------
```

용량은 세 수준이 서로 모두 다르지만, 보충제는 $p = 0.060$으로 유의하지 않다. 분산분석표에서 `C(supp)`가 $p = 0.00023$이었던 것과 어긋나 보이는데, 이 Tukey가 용량을 무시하고 OJ 30개와 VC 30개를 통째로 비교하기 때문이다. 용량이 만드는 큰 변동이 잡음으로 남아 보충제의 차이를 덮는다.

#### 3단계: 교호작용 효과에 대한 사후검정

```python
# Create a combined factor for interaction analysis
df['supp_dose'] = df['supp'].astype(str) + "_" + df['dose'].astype(str)

# Perform Tukey's HSD on the interaction between supplement and dose
tukey_interaction = pairwise_tukeyhsd(endog=df['len'], groups=df['supp_dose'], alpha=0.05)
print("Post-Hoc Test for Interaction (Supplement x Dose):")
print(tukey_interaction)
```

출력:

```
Post-Hoc Test for Interaction (Supplement x Dose):
 Multiple Comparison of Means - Tukey HSD, FWER=0.05  
======================================================
group1 group2 meandiff p-adj   lower    upper   reject
------------------------------------------------------
OJ_0.5 OJ_1.0     9.47    0.0   4.6719  14.2681   True
OJ_0.5 OJ_2.0    12.83    0.0   8.0319  17.6281   True
OJ_0.5 VC_0.5    -5.25 0.0243 -10.0481  -0.4519   True
OJ_0.5 VC_1.0     3.54  0.264  -1.2581   8.3381  False
OJ_0.5 VC_2.0    12.91    0.0   8.1119  17.7081   True
OJ_1.0 OJ_2.0     3.36 0.3187  -1.4381   8.1581  False
OJ_1.0 VC_0.5   -14.72    0.0 -19.5181  -9.9219   True
OJ_1.0 VC_1.0    -5.93 0.0074 -10.7281  -1.1319   True
OJ_1.0 VC_2.0     3.44 0.2936  -1.3581   8.2381  False
OJ_2.0 VC_0.5   -18.08    0.0 -22.8781 -13.2819   True
OJ_2.0 VC_1.0    -9.29    0.0 -14.0881  -4.4919   True
OJ_2.0 VC_2.0     0.08    1.0  -4.7181   4.8781  False
VC_0.5 VC_1.0     8.79    0.0   3.9919  13.5881   True
VC_0.5 VC_2.0    18.16    0.0  13.3619  22.9581   True
VC_1.0 VC_2.0     9.37    0.0   4.5719  14.1681   True
------------------------------------------------------
```

같은 용량끼리 비교한 세 줄(`OJ_0.5 VC_0.5`, `OJ_1.0 VC_1.0`, `OJ_2.0 VC_2.0`)을 보면 차이가 각각 $-5.25$($p = 0.024$), $-5.93$($p = 0.007$), $-0.08$($p = 1.000$)이다. 낮은 용량에서는 OJ가 앞서지만 용량 2.0에서는 차이가 사라진다. 이것이 교호작용의 내용이다.

#### 4단계: 단순 효과 분석 (교호작용 사후검정의 대안)

교호작용 효과가 유의하면 **단순 효과 분석**으로 다른 요인의 각 수준에서 한 요인의 효과를 살펴 자세히 나눠 볼 수 있다.

```python
# Separate data by supplement type
oj_data = df[df['supp'] == 'OJ']
vc_data = df[df['supp'] == 'VC']

# Perform one-way ANOVA on dose within each supplement type
oj_model = ols('len ~ C(dose)', data=oj_data).fit()
vc_model = ols('len ~ C(dose)', data=vc_data).fit()

# Print ANOVA results for each supplement type
print("ANOVA for Dose within Supplement OJ:")
print(anova_lm(oj_model))

print("ANOVA for Dose within Supplement VC:")
print(anova_lm(vc_model))

# Tukey's HSD for dose within each supplement type
print("Tukey HSD for Dose within Supplement OJ:")
print(pairwise_tukeyhsd(endog=oj_data['len'], groups=oj_data['dose'], alpha=0.05))

print("Tukey HSD for Dose within Supplement VC:")
print(pairwise_tukeyhsd(endog=vc_data['len'], groups=vc_data['dose'], alpha=0.05))
```

출력:

```
ANOVA for Dose within Supplement OJ:
            df      sum_sq     mean_sq          F        PR(>F)
C(dose)    2.0  885.264667  442.632333  31.441504  8.887164e-08
Residual  27.0  380.105000   14.077963        NaN           NaN
ANOVA for Dose within Supplement VC:
            df       sum_sq     mean_sq          F        PR(>F)
C(dose)    2.0  1649.488667  824.744333  67.072379  3.357317e-11
Residual  27.0   332.001000   12.296333        NaN           NaN
Tukey HSD for Dose within Supplement OJ:
Multiple Comparison of Means - Tukey HSD, FWER=0.05 
====================================================
group1 group2 meandiff p-adj   lower   upper  reject
----------------------------------------------------
   0.5    1.0     9.47    0.0  5.3096 13.6304   True
   0.5    2.0    12.83    0.0  8.6696 16.9904   True
   1.0    2.0     3.36 0.1309 -0.8004  7.5204  False
----------------------------------------------------
Tukey HSD for Dose within Supplement VC:
Multiple Comparison of Means - Tukey HSD, FWER=0.05
===================================================
group1 group2 meandiff p-adj  lower   upper  reject
---------------------------------------------------
   0.5    1.0     8.79   0.0  4.9018 12.6782   True
   0.5    2.0    18.16   0.0 14.2718 22.0482   True
   1.0    2.0     9.37   0.0  5.4818 13.2582   True
---------------------------------------------------
```

단순 효과 분석이 교호작용을 가장 또렷하게 보여준다. OJ 안에서는 용량 1.0과 2.0의 차이가 $p = 0.131$로 유의하지 않은 반면, VC 안에서는 같은 비교가 $p < 0.001$로 강하게 유의하다.

즉 **용량을 0.5에서 1.0으로 올리는 것은 두 보충제 모두에서 효과가 있지만, 1.0에서 2.0으로 더 올리는 것은 VC에서만 효과가 있다.** 교호작용 항의 $p = 0.022$가 요약한 것이 이 이야기다.

### D. 단계 요약

1. **이원배치 분산분석 수행**: 유의한 주효과와 교호작용 효과를 파악한다.
2. **주효과의 사후검정**: 유의한 각 주효과에 대해 Tukey의 HSD나 다른 쌍별 검정을 쓴다.
3. **교호작용의 사후검정**: 교호작용 효과가 유의하면 결합된 요인 수준에 Tukey의 HSD를 적용하거나 단순 효과 분석을 수행한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
운동: HIIT $\bar Y = 8.8$, 근력 $6.4$, 요가 $4.4$, 각 $n = 5$, $\mathrm{MSW} = 1.43$. (a) 전체 분산분석. (b) Tukey HSD. (c) 해석.

</div>

??? success "풀이"
    (a) $\mathrm{SSB} = 5 \cdot [(8.8-6.53)^2 + (6.4-6.53)^2 + (4.4-6.53)^2] \approx 48.53$.

    $\mathrm{SSW} = 17.20$. $F = (48.53/2)/(17.20/12) = 24.27/1.43 \approx 16.93$.

    $F_{0.05, 2, 12} = 3.89$이므로 $H_0$을 기각한다.

    (b) HSD $= q_{0.05, 3, 12} \cdot \sqrt{\mathrm{MSW}/n} = 3.77 \cdot \sqrt{1.43/5} \approx 2.02$.

    쌍별: HIIT–근력 = 2.4 > 2.02 ✓; HIIT–요가 = 4.4 > 2.02 ✓; 근력–요가 = 2.0 < 2.02 ✗.

    (c) HIIT는 다른 두 방식보다 유의하게 낫다. 근력 대 요가는 유의하지 않아 둘을 구별할 수 없다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
**Bonferroni 쌍별 t-검정 대신 Tukey HSD를 쓰는 이유는?**

</div>

??? success "풀이"
    Tukey HSD는 평균의 쌍별 비교에 정확히 맞춰 보정된 **스튜던트화 범위** 분포를 쓴다.

    Bonferroni는 쌍별 $t$-검정에 Bonferroni 보정을 적용한다. $k = \binom{g}{2}$일 때 $\alpha/k$에서 검정한다.

    **비교:**

    - **Tukey:** 평균의 쌍별 비교에서 더 강력하며 가족단위 오류를 정확히 통제한다.
    - **Bonferroni:** 간단하고 매우 일반적이지만(어떤 검정에도 쓸 수 있지만) 비교가 많으면 보수적이다.

    균형 잡힌 일원배치 분산분석의 쌍별 비교에는 Tukey가 표준이다. 복잡한 대비나 혼합 설계에는 Bonferroni나 Scheffé 방법이 필요할 수 있다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
**다른 사후검정들.** Bonferroni, Scheffé, Dunnett을 간략히 설명하라.

</div>

??? success "풀이"
    **Bonferroni:** 각 비교를 $\alpha/k$에서 검정한다. 어떤 검정에도 쓸 수 있지만 보수적이다.

    **Scheffé:** 쌍별에 국한되지 않고 임의의 대비(평균의 선형결합)를 허용한다. 가장 보수적이지만 가장 일반적이다.

    **Dunnett:** 모든 집단을 하나의 대조군과 비교한다. 대조군과의 비교만 중요할 때 Tukey보다 강력하다.

    **Fisher의 LSD:** 분산분석의 합동분산을 쓰지만 다중검정 보정을 하지 않는다. 관대하므로 빠른 선별용으로만 유용하다.

    가설에 따라 고른다:

    - 모든 쌍별 비교: Tukey.
    - 모든 집단 대 대조군: Dunnett.
    - 임의의 선형 대비: Scheffé.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
**가족단위 오류와 비교단위 오류.**

</div>

??? success "풀이"
    **비교단위:** 개별 비교 하나의 제1종 오류율.

    **가족단위:** 한 가족에 속한 모든 비교 중 적어도 하나에서 오류를 범할 확률.

    보정하지 않으면 비교 횟수와 함께 가족단위 오류가 커진다. Tukey, Bonferroni 등은 가족단위를 통제한다.

    대안: **거짓발견율(FDR)**은 유의하다고 선언한 것 중 거짓 기각의 기대 비율을 통제한다. 덜 보수적이어서 비교가 아주 많을 때(예: 유전체학) 유용하다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
**스튜던트화 범위 분포.** 간단히 소개하라.

</div>

??? success "풀이"
    스튜던트화 범위 $q$: $H_0$ 아래에서 $(\max \bar Y_i - \min \bar Y_i)/\sqrt{\mathrm{MSW}/n}$의 분포.

    다음에 의존한다:

    - 집단의 수 $g$.
    - 집단 내 자유도($N - g$).

    표: 임계값 $q_{\alpha, g, df}$가 표로 제공된다.

    Tukey HSD와의 연결: HSD $= q_{\alpha} \sqrt{\mathrm{MSW}/n}$. 각 쌍별 차이를 HSD와 비교한다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
**불균형 설계**와 Tukey.

</div>

??? success "풀이"
    $n_i$가 서로 다르면 **Tukey-Kramer** 수정을 쓴다:

    $\mathrm{HSD}_{ij} = q_{\alpha} \sqrt{(\mathrm{MSW}/2)(1/n_i + 1/n_j)}$.

    각 쌍이 ($n_i, n_j$에 따라) 자신의 HSD 문턱을 갖는다. 보수적이지만 타당하다.

    순수한 Tukey는 $n$이 같다고 가정한다. Tukey-Kramer가 표준 확장이다. R의 `TukeyHSD`와 Python의 `statsmodels`는 크기가 다르면 Tukey-Kramer를 쓴다.

---

## 정리하며

사후검정은 **$F$ 가 기각한 뒤에** 어느 쌍이 다른지 찾는다.

- **투키 HSD 가 쌍별 비교의 표준이다.** 모든 쌍을 동시에 다루면서 FWER 을 정확히 $\alpha$ 로 통제하며, 스튜던트화 범위분포를 쓴다.
- **본페로니보다 검정력이 높다.** 모든 쌍을 비교한다는 사실을 이용해 상관 구조를 반영하기 때문이며, **쌍별 비교가 목적이라면 투키가 낫다.**
- **표본크기가 같을 때 가장 잘 작동한다.** 불균형이면 투키–크레이머 수정을 쓴다.
- **등분산을 가정한다.** 합동 MSE 를 쓰므로 분산이 다르면 게임스–하월로 가야 한다.
- **"$F$ 가 유의할 때만 사후검정"이라는 관행**이 널리 쓰이지만, 투키는 그 자체로 FWER 을 통제하므로 논리적으로 필수는 아니다. 다만 관행을 따르는 편이 보고에 안전하다.

다음 절 **Bonferroni와 Scheffé 방법**으로 넘어간다. 목적이 다르면 도구도 달라진다.
