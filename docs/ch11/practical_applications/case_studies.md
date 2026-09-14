# 분산분석의 실무 응용

이 절에서는 Python으로 분산분석의 가정 검정과 진단을 보여주는 완결된 예제를 제시한다. 각 사례 연구는 전체 흐름을 따른다: 모형 적합, 가정 확인, 위반 사항 처리.

## 사례 연구 1: 붓꽃 종 (식물 형태)

### 배경

고전적인 붓꽃(Iris) 자료를 써서 두 종(versicolor와 virginica) 사이에 꽃받침 길이가 유의하게 다른지 검정한다. 이 예제는 분산분석의 전체 진단 흐름을 보여준다.

### 1단계: 자료 적재와 모형 적합

<div class="codebox" markdown>

#### 예제 1. 사례 1 — 자료와 모형 { .eg }

```python
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

# 붓꽃 자료에서 두 품종만 남긴다. 집단이 둘이면 분산분석과 이표본 t-검정이
# 같은 결론을 주며, F = t^2 이라는 관계도 확인할 수 있다.
data = sns.load_dataset("iris")
data = data[data["species"] != "setosa"]

model = ols('sepal_length ~ species', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
```

출력:

```
           sum_sq    df          F        PR(>F)
species   10.6276   1.0  31.687502  1.724856e-07
Residual  32.8680  98.0        NaN           NaN
```

$F = 31.7$, $p = 1.7 \times 10^{-7}$로 두 종의 꽃받침 길이가 다르다는 결론이 압도적이다. 집단당 50개씩이라 검정력이 넉넉하다.

</div>

### 2단계: 정규성 확인

<div class="codebox" markdown>

#### 예제 2. 사례 1 — 정규성 확인 { .eg }

```python
import matplotlib.pyplot as plt
from scipy.stats import shapiro

# 정규성은 자료가 아니라 잔차에 요구된다.
sm.qqplot(model.resid, line='s')
plt.title("Q-Q Plot of Residuals")
plt.show()

# 표본이 크면 사소한 이탈에도 유의하게 나오므로 그림과 함께 읽는다.
stat, p_value = shapiro(model.resid)
print(f"Shapiro-Wilk Test: W = {stat:.4f}, p-value = {p_value:.4f}")
```

출력:

```
Shapiro-Wilk Test: W = 0.9831, p-value = 0.2285
```

![잔차의 Q-Q 그림](./img/case_studies_29.png)

$p = 0.23$으로 정규성에 반하는 증거가 없고, Q-Q 그림의 점들도 기준선을 잘 따른다.

</div>

### 3단계: 등분산성 확인

<div class="codebox" markdown>

#### 예제 3. 사례 1 — 등분산성 확인 { .eg }

```python
from scipy.stats import levene

# Levene 검정으로 두 집단의 분산이 같다고 볼 수 있는지 확인한다.
group1 = data[data['species'] == 'versicolor']['sepal_length']
group2 = data[data['species'] == 'virginica']['sepal_length']
stat, p_value = levene(group1, group2)
print(f"Levene's Test: F = {stat:.4f}, p-value = {p_value:.4f}")
```

출력:

```
Levene's Test: F = 1.0245, p-value = 0.3139
```

$p = 0.31$로 등분산도 기각되지 않는다. 두 가정이 모두 무난하므로 표준 분산분석 결과를 그대로 쓸 수 있다.

</div>

### 4단계: 독립성 확인 (잔차 그림)

<div class="codebox" markdown>

#### 예제 4. 사례 1 — 잔차 그림 { .eg }

```python
# 일원배치에서 적합값은 집단평균뿐이므로 세로줄이 집단 수만큼만 생긴다.
# 각 줄의 퍼짐이 비슷한지를 본다.
plt.scatter(model.fittedvalues, model.resid, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted Values')
plt.show()
```

![잔차 대 적합값](./img/case_studies_56.png)

세로 띠가 둘이고 각 띠의 높이가 비슷하다. 등분산 가정이 무난하다는 Levene 검정의 결론과 일치한다.

</div>

### 해석

정규성과 등분산성이 기각되지 않고(두 검정 모두 p > 0.05) 잔차 그림에 체계적인 패턴이 없으면 분산분석 결과를 자신 있게 해석할 수 있다. 그렇지 않으면 Welch 분산분석이나 Kruskal-Wallis 검정을 고려한다.

---

## 사례 연구 2: 근무 형태에 따른 직원 생산성

### 배경

어떤 회사가 세 가지 근무 형태(재택, 사무실, 혼합)에 따라 직원 생산성이 다른지 판정하려 한다. 이 예제는 작은 모의 자료를 쓴다.

### 1단계: 자료 적재와 모형 적합

<div class="codebox" markdown>

#### 예제 5. 사례 2 — 자료와 모형 { .eg }

```python
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# 근무 형태 세 가지에 따른 생산성. 집단마다 다섯 명씩이다.
data = pd.DataFrame({
    'productivity': [68, 75, 80, 65, 85, 78, 70, 82, 90, 88, 72, 95, 67, 85, 79],
    'environment': ['remote']*5 + ['office']*5 + ['hybrid']*5
})

model = ols('productivity ~ environment', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
```

출력:

```
             sum_sq    df         F    PR(>F)
environment   130.0   2.0  0.768019  0.485443
Residual     1015.6  12.0       NaN       NaN
```

$F = 0.77$, $p = 0.49$로 기각하지 못한다. 세 형태의 생산성 평균이 다르다는 증거가 없다.

다만 집단당 5명뿐이라 검정력이 거의 없다시피 하다는 점을 함께 보아야 한다. 잔차 자유도가 12에 불과하므로 이 결과를 "차이가 없다"로 읽으면 안 된다.

</div>

### 2단계: 가정 확인

<div class="codebox" markdown>

#### 예제 6. 사례 2 — 가정 확인 { .eg }

```python
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene

# 정규성 — 잔차의 Q-Q 그림과 Shapiro-Wilk 검정
sm.qqplot(model.resid, line='s')
plt.title("Q-Q Plot of Residuals")
plt.show()

stat, p_value = shapiro(model.resid)
print(f"Shapiro-Wilk Test: p-value = {p_value:.4f}")

# 등분산성 — Levene 검정
group1 = data[data['environment'] == 'remote']['productivity']
group2 = data[data['environment'] == 'office']['productivity']
group3 = data[data['environment'] == 'hybrid']['productivity']
stat, p_value = levene(group1, group2, group3)
print(f"Levene's Test: p-value = {p_value:.4f}")

# 독립성 — 잔차 대 적합값 그림
plt.scatter(model.fittedvalues, model.resid, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted Values')
plt.show()
```

출력:

```
Shapiro-Wilk Test: p-value = 0.7449
Levene's Test: p-value = 0.7631
```

![잔차의 Q-Q 그림과 잔차 그림](./img/case_studies_96.png)

두 검정 모두 기각하지 못한다($p = 0.74$, $p = 0.76$). 그러나 $n = 15$에서 이 검정들의 검정력은 매우 낮아, "가정이 확인되었다"기보다 "확인할 수 없었다"에 가깝다.

</div>

### 작은 표본에 대한 주의

집단당 관측값이 5개뿐이면 Shapiro-Wilk 검정의 검정력이 낮고 Q-Q 그림도 그다지 유익하지 않을 수 있다. 이런 경우 분산분석은 모집단이 정규라는 가정에 크게 의존하므로 비모수 검정을 함께 수행하는 편이 신중하다.

---

## 사례 연구 3: 매장별 고객 만족도

### 배경

어떤 소매업체가 네 매장(A, B, C, D)의 고객 만족도 점수를 분석하여 유의한 차이가 있는지 판정한다.

### 1단계: 자료 적재와 모형 적합

<div class="codebox" markdown>

#### 예제 7. 사례 3 — 자료와 모형 { .eg }

```python
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# 지점 네 곳의 고객만족도. 지점마다 다섯 건씩이다.
data = pd.DataFrame({
    'satisfaction': [4.5, 3.8, 4.7, 4.2, 4.9, 4.1, 3.5, 4.3, 4.8, 3.9,
                     4.4, 4.0, 3.7, 4.2, 4.6, 4.8, 3.6, 4.3, 4.1, 4.7],
    'location': ['A']*5 + ['B']*5 + ['C']*5 + ['D']*5
})

model = ols('satisfaction ~ location', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
```

출력:

```
          sum_sq    df         F    PR(>F)
location  0.2655   3.0  0.456186  0.716615
Residual  3.1040  16.0       NaN       NaN
```

$F = 0.46$, $p = 0.72$로 네 매장의 만족도에 차이가 없다. 집단 간 제곱합 0.27이 잔차 제곱합 3.10에 비해 아주 작다.

</div>

### 2단계: 가정 확인

<div class="codebox" markdown>

#### 예제 8. 사례 3 — 가정 확인 { .eg }

```python
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene

# 정규성 — 잔차의 Q-Q 그림과 Shapiro-Wilk 검정
sm.qqplot(model.resid, line='s')
plt.title("Q-Q Plot of Residuals")
plt.show()

stat, p_value = shapiro(model.resid)
print(f"Shapiro-Wilk Test: p-value = {p_value:.4f}")

# 등분산성 — Levene 검정
groups = [data[data['location'] == loc]['satisfaction'] for loc in ['A', 'B', 'C', 'D']]
stat, p_value = levene(*groups)
print(f"Levene's Test: p-value = {p_value:.4f}")

# 독립성 — 잔차 대 적합값 그림
plt.scatter(model.fittedvalues, model.resid, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted Values')
plt.show()
```

출력:

```
Shapiro-Wilk Test: p-value = 0.5488
Levene's Test: p-value = 0.9343
```

![잔차의 Q-Q 그림과 잔차 그림](./img/case_studies_156.png)

가정 위반의 증거가 없다.

</div>

### 3단계: 사후분석

분산분석이 유의한 차이를 드러내고 가정도 충족되면 사후 쌍별 비교를 수행한다:

<div class="codebox" markdown>

#### 예제 9. 사례 3 — 사후분석 { .eg }

```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# 분산분석이 유의했으므로 어느 지점 쌍이 다른지 사후비교로 좁힌다.
tukey = pairwise_tukeyhsd(data['satisfaction'], data['location'], alpha=0.05)
print(tukey)
```

출력:

```
Multiple Comparison of Means - Tukey HSD, FWER=0.05
=================================================
group1 group2 meandiff p-adj  lower  upper reject
-------------------------------------------------
     A      B     -0.3  0.708 -1.097 0.497  False
     A      C    -0.24 0.8243 -1.037 0.557  False
     A      D    -0.12 0.9723 -0.917 0.677  False
     B      C     0.06 0.9963 -0.737 0.857  False
     B      D     0.18 0.9154 -0.617 0.977  False
     C      D     0.12 0.9723 -0.677 0.917  False
-------------------------------------------------
```

여섯 비교 중 유의한 것이 하나도 없다. 전역 분산분석이 기각하지 못했으니 당연한 결과다.

실은 이 단계를 밟지 말았어야 한다. **전역 검정이 기각하지 못했으면 사후검정으로 넘어가지 않는 것이 원칙이다.** 그러지 않으면 다중비교 통제가 무너진다. 여기서는 절차를 보여주기 위해 실행했을 뿐이다.

Tukey의 HSD에 대한 자세한 내용은 [Tukey HSD](../post_hoc/tukey.md)를 보라.

</div>

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
어떤 전자상거래 회사가 네 가지 결제 페이지 디자인(A, B, C, D)의 전환율을 시험한다. 각 디자인은 무작위로 뽑은 방문자 200명에게 보여준다. 이를 일원배치 분산분석 문제로 설정하는 방법을 기술하라. 집단, 반응변수, 귀무가설, 확인해야 할 핵심 가정을 정의하라.

</div>

??? success "풀이"

    - **집단:** 네 가지 결제 페이지 디자인(A, B, C, D), $k = 4$.
    - **반응변수:** 전환율(또는 구매까지 걸린 시간, 장바구니 금액 같은 적절한 연속형 지표). 전환/비전환의 이진 결과를 쓴다면 비율에 대한 분산분석은 큰 표본이 필요하거나 로지스틱 회귀 같은 대안이 필요하다.
    - **귀무가설:** $H_0: \mu_A = \mu_B = \mu_C = \mu_D$ (네 디자인의 모평균 반응이 같다).
    - **확인할 가정:**
        1. **독립성:** 무작위 배정으로 서로 다른 집단의 방문자가 독립임을 보장한다. 어떤 방문자도 여러 집단에 나타나지 않는지 확인한다.
        2. **정규성:** 집단당 $n = 200$이면 중심극한정리가 집단 평균의 근사적 정규성을 보장한다.
        3. **등분산성:** Levene 검정으로 확인한다. 어긋나면 Welch 분산분석을 쓴다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
금융 분산분석에서 어떤 포트폴리오 매니저가 세 섹터 ETF(기술, 헬스케어, 에너지)의 평균 월 수익률을 60개월에 걸쳐 비교한다. 표준적인 실험 설계에서는 대체로 생기지 않는, 이 상황 특유의 가정 문제는 무엇인가?

</div>

??? success "풀이"
    가장 큰 추가 문제는 **독립성**이다. 서로 다른 섹터 ETF의 월 수익률은 **같은 기간**에 측정되므로 공통의 시장 요인(예: 금리 변화, 거시 충격) 때문에 상관될 가능성이 높다. 이는 표준 일원배치 분산분석의 독립성 가정을 위반한다.

    또한 금융 수익률은 시간에 따라 순차적으로 측정되므로 각 집단 안에서 **자기상관**이 생길 수 있다. 유효 표본크기가 60보다 훨씬 작아져 F-통계량이 부풀려지고 거짓 양성이 나올 수 있다.

    적절한 처방으로는 (월을 블록 요인으로 다루는) **반복측정 분산분석**, **혼합효과 모형**, 또는 자기상관과 횡단면 의존을 함께 반영하는 **HAC(Newey-West) 표준오차**가 있다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
실무 사례 연구에서 자료 수집부터 최종 결론까지 분산분석의 전체 작업 흐름을 기술하라. 적어도 여섯 단계를 포함하라.

</div>

??? success "풀이"

    1. **연구 질문과 가설을 정의한다.** 집단, 반응변수, 귀무가설과 대립가설을 진술한다.

    2. **자료를 수집한다.** 무작위 표집과 처치군으로의 무작위 배정을 쓴다. 원하는 검정력에 필요한 표본크기를 확보한다.

    3. **탐색적 자료 분석.** 집단 평균, 표준편차, 표본크기를 계산한다. 상자그림으로 집단 분포를 시각화하고 잠재적 이상점을 찾는다.

    4. **분산분석 모형을 적합한다.** 소프트웨어(예: `scipy.stats.f_oneway`나 `statsmodels`)를 쓰고 F-통계량과 p-값을 기록한다.

    5. **가정을 확인한다:**
        - 정규성: 잔차의 Q-Q 그림과 Shapiro-Wilk 검정.
        - 등분산성: Levene 검정과 잔차 대 적합값 그림.
        - 독립성: 연구 설계 검토, 자료에 순서가 있으면 Durbin-Watson 검정.

    6. **위반이 발견되면 대처한다:** Welch 분산분석으로 옮기거나, 변환을 적용하거나, 비모수 대안을 쓴다.

    7. **전체 분산분석이 유의하면 사후검정을 수행한다**(맥락에 따라 Tukey HSD, Games-Howell, Dunnett).

    8. **결과를 보고한다.** 효과크기(에타제곱), 쌍별 차이의 신뢰구간, 맥락에 맞는 명확한 결론 진술을 포함한다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
사례 연구 1(붓꽃)을 **끝까지** 수행하라. 진단에서 그친 본문을 이어받아 검정·사후비교·효과크기까지 보고하라.

</div>

??? success "풀이"
    ```python
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    from scipy import stats
    from statsmodels.stats.libqsturng import qsturng
    from sklearn.datasets import load_iris

    d = load_iris()
    df = pd.DataFrame(d.data, columns=["sl", "sw", "pl", "pw"])
    df["sp"] = [d.target_names[i] for i in d.target]

    print("사례 1: 붓꽃 꽃받침 너비(sepal width)")
    g = df.groupby("sp").sw.agg(["size", "mean", "std"]).round(4)
    print(g.to_string())

    gs = [v.sw.values for _, v in df.groupby("sp")]
    lev = stats.levene(*gs, center="median")
    print(f"\n브라운-포사이드: W = {lev.statistic:.4f}, p = {lev.pvalue:.4f}")
    r = stats.f_oneway(*gs)
    print(f"표준 F: {r.statistic:.4f}, p = {r.pvalue:.3e}")

    def welch(gs):
        n = np.array([len(g) for g in gs], float)
        m = np.array([g.mean() for g in gs])
        v = np.array([g.var(ddof=1) for g in gs])
        k = len(n)
        w = n / v
        W = w.sum()
        mt = (w * m).sum() / W
        lam = ((1 - w / W)**2 / (n - 1)).sum()
        F = ((w * (m - mt)**2).sum() / (k - 1)
             / (1 + 2 * (k - 2) / (k**2 - 1) * lam))
        df2 = (3 / (k**2 - 1) * lam)**-1
        return F, df2, stats.f.sf(F, k - 1, df2)

    F, d2, p = welch(gs)
    print(f"웰치 F = {F:.4f}, df2 = {d2:.3f}, p = {p:.3e}")

    N, k = len(df), 3
    print(f"η² = {2 * r.statistic / (2 * r.statistic + N - k):.4f},  "
          f"ω² = {2 * (r.statistic - 1) / (2 * r.statistic + N - k + 1):.4f}")

    print("\n게임스-하웰")
    names = list(g.index)
    n = np.array([len(x) for x in gs], float)
    m = np.array([x.mean() for x in gs])
    v = np.array([x.var(ddof=1) for x in gs])
    for i in range(3):
        for j in range(i + 1, 3):
            s = v[i] / n[i] + v[j] / n[j]
            dfw = s**2 / (v[i]**2 / (n[i]**2 * (n[i] - 1))
                          + v[j]**2 / (n[j]**2 * (n[j] - 1)))
            diff = m[j] - m[i]
            h = qsturng(0.95, 3, dfw) * np.sqrt(s / 2)
            print(f"  {names[j]:>12s} - {names[i]:<12s} {diff:+7.4f}  "
                  f"95% [{diff - h:+7.4f},{diff + h:+7.4f}]  df={dfw:6.2f}")
    ```

    ```text
    사례 1: 붓꽃 꽃받침 너비(sepal width)
                size   mean     std
    sp                             
    setosa        50  3.428  0.3791
    versicolor    50  2.770  0.3138
    virginica     50  2.974  0.3225

    브라운-포사이드: W = 0.5902, p = 0.5555
    표준 F: 49.1600, p = 4.492e-17
    웰치 F = 45.0120, df2 = 97.402, p = 1.433e-14
    η² = 0.4008,  ω² = 0.3910

    게임스-하웰
        versicolor - setosa       -0.6580  95% [-0.8237,-0.4923]  df= 94.70
         virginica - setosa       -0.4540  95% [-0.6216,-0.2864]  df= 95.55
         virginica - versicolor   +0.2040  95% [+0.0525,+0.3555]  df= 97.93
    ```

    **세 종이 모두 서로 다르다.** 세 구간 중 어느 것도 0을 포함하지 않는다.

    | 비교 | 차이 | 95% 동시구간 |
    |---|---|---|
    | versicolor $-$ setosa | $-0.658$ | $[-0.824,\ -0.492]$ |
    | virginica $-$ setosa | $-0.454$ | $[-0.622,\ -0.286]$ |
    | **virginica $-$ versicolor** | $+0.204$ | $[+0.053,\ +0.356]$ |

    **세 번째가 아슬아슬하다.** 구간의 하한이 0.053으로 0에 가깝다. **가장 작은 차이**를 간신히 잡았다.

    **등분산이 성립한다**(브라운-포사이드 $p=0.556$). 그래서 표준 $F$(49.16)와 웰치(45.01)의 결론이 같다.

    **그럼에도 게임스-하웰을 쓴 이유.** 표준편차가 0.314~0.379로 약간 다르고, 게임스-하웰의 **손실이 3% 이내**다. **손해 볼 것이 없다.**

    **효과크기가 매우 크다.** $\omega^2=0.391$로 **꽃받침 너비 변동의 39%를 종이 설명**한다. 코헨 기준으로 $f=\sqrt{0.391/0.609}=0.80$이며 "매우 큼"이다.

    **보고문.**

    ```text
    붓꽃 세 종(각 n = 50)의 꽃받침 너비를 비교했다.

    기술통계
      setosa      3.428 ± 0.379
      versicolor  2.770 ± 0.314
      virginica   2.974 ± 0.323

    가정 점검
      브라운-포사이드 등분산 검정  W = 0.59, p = 0.56  (문제 없음)
      잔차 Q-Q 그림에서 뚜렷한 이탈 없음

    분석
      Welch F(2, 97.4) = 45.01,  p < 0.001,  ω² = 0.391
      사후검정: 게임스-하웰 (FWER = 0.05)

      versicolor − setosa     −0.658  [−0.824, −0.492]
      virginica  − setosa     −0.454  [−0.622, −0.286]
      virginica  − versicolor +0.204  [ 0.053,  0.356]

    결론
      세 종의 꽃받침 너비가 모두 서로 다르다. setosa 가 가장 넓고,
      versicolor 가 가장 좁다. 종이 변동의 39% 를 설명한다.
    ```

    **본문이 진단에서 멈춘 것이 아쉽다.** 가정 점검은 **분석의 시작**이지 결론이 아니다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
사례 연구 2(근무 형태와 생산성)에는 **설계상의 근본적 문제**가 있다. 무엇이며 어떻게 다루어야 하는가?

</div>

??? success "풀이"
    **문제 — 근무 형태는 무작위 배정되지 않는다.**

    ```text
    관찰된 것:  재택 근무자의 생산성이 사무실 근무자보다 높다

    가능한 설명 넷
      (가) 재택 근무가 생산성을 높인다            ← 우리가 원하는 해석
      (나) 생산성 높은 사람이 재택을 선택했다      ← 자기 선택
      (다) 생산성 높은 직무가 재택 가능하다        ← 직무 교란
      (라) 재택 근무자의 생산성 측정이 다르다      ← 측정 편향
    ```

    **이것이 관찰연구와 실험의 차이**다. 분산분석은 (가)~(라)를 **구분하지 못한다.**

    **교란의 크기를 모의실험으로 보자.**

    ```python
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(19001)
    B = 3_000
    print("근무 형태의 참 효과는 0. 다만 '능력'이 선택과 생산성에 모두 영향")
    print(f"{'선택 강도 (능력→재택 확률)':>26s} {'관측 차이 평균':>13s} {'유의 비율':>9s}")
    for gamma in [0.0, 0.5, 1.0, 2.0]:
        diffs, hit = [], 0
        for _ in range(B):
            N = 150
            ability = rng.normal(0, 1, N)
            p = 1 / (1 + np.exp(-gamma * ability))       # 능력이 높을수록 재택
            home = rng.random(N) < p
            prod = 50 + 5 * ability + rng.normal(0, 3, N)  # 근무 형태의 효과는 0
            if home.sum() < 5 or (~home).sum() < 5:
                continue
            d = prod[home].mean() - prod[~home].mean()
            diffs.append(d)
            hit += stats.ttest_ind(prod[home], prod[~home],
                                   equal_var=False).pvalue < 0.05
        print(f"{gamma:26.1f} {np.mean(diffs):13.3f} {hit / len(diffs):9.4f}")
    ```

    ```text
    근무 형태의 참 효과는 0. 다만 '능력'이 선택과 생산성에 모두 영향
              선택 강도 (능력→재택 확률)      관측 차이 평균     유의 비율
                           0.0         0.023    0.0410
                           0.5         2.354    0.7060
                           1.0         4.124    0.9943
                           2.0         6.052    1.0000
    ```

    **선택 강도가 1.0이면 참 효과가 0인데도 차이가 4.12로 관측되고 99% 유의하다.**

    | 선택 강도 | 관측 차이 | 유의 비율 |
    |---|---|---|
    | 0(무작위) | 0.023 | **0.041** |
    | 0.5 | 2.354 | 0.706 |
    | **1.0** | **4.124** | **0.994** |
    | 2.0 | 6.052 | 1.000 |

    **무작위 배정이 되면 오류율이 0.041로 명목 근처**다. 교란이 통계의 문제가 아니라 **설계의 문제**임을 보여 준다.

    **대응 넷.**

    | 방법 | 내용 | 한계 |
    |---|---|---|
    | **무작위 배정** | 근무 형태를 무작위로 | 현실적으로 어려움 |
    | **공변량 조정** | 능력 대리변수(과거 성과)를 공분산분석에 | 관측되지 않은 교란은 못 잡음 |
    | **성향점수 매칭** | 비슷한 사람끼리 짝짓기 | 관측된 변수만 |
    | **개체 내 비교** | 같은 사람의 전후 비교 | 시간 효과와 교락 |

    **네 번째가 가장 강력하다.** 코로나로 강제 재택이 시행된 기간처럼 **자연 실험**이 있으면 선택 편향이 사라진다.

    **공분산분석의 함정.** 능력 대리변수를 넣으면

    | 대리변수 | 결과 |
    |---|---|
    | 완벽한 능력 측정 | 교란 제거 |
    | **불완전한 측정** | **교란이 부분적으로만 제거**(잔차 교란) |
    | **처치 후 측정** | **처치 효과의 일부를 제거**(과잉 조정) |

    **세 번째가 특히 위험하다.** "재택 시작 후의 만족도"를 공변량으로 넣으면, 그것이 **처치의 결과**이므로 효과를 지워 버린다.

    **보고할 때의 언어.**

    | 관찰연구 | 실험 |
    |---|---|
    | "재택 근무자의 생산성이 **더 높았다**" | "재택 근무가 생산성을 **높였다**" |
    | "연관되어 있다" | "인과적으로" |

    **동사의 선택이 주장의 강도를 정한다.**

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
**새 사례 — 용량-반응 연구.** 다섯 용량 수준에서 반응을 측정했다. 전체 $F$ 검정보다 **추세 대비**가 나은 이유를 보여라.

</div>

??? success "풀이"
    **직교 다항 대비.** 용량이 등간격이면

    | 대비 | 계수($k=5$) | 묻는 것 |
    |---|---|---|
    | **선형** | $(-2,-1,0,1,2)$ | 단조 증가/감소하는가 |
    | **이차** | $(2,-1,-2,-1,2)$ | 볼록/오목한가 |
    | 삼차 | $(-1,2,0,-2,1)$ | S자인가 |

    ```python
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(18001)
    B = 6_000
    doses = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    k, n = len(doses), 10
    N, dfe = k * n, k * n - k

    lin = doses - doses.mean()
    lin /= np.sqrt((lin**2).sum())
    quad = (doses - doses.mean())**2
    quad -= quad.mean()
    quad /= np.sqrt((quad**2).sum())

    print("용량당 n=10, 오차 SD=1.0, 명목 0.05")
    print(f"{'참 반응 모양':>22s} {'전체 F':>8s} {'선형 대비':>9s} {'이차 대비':>9s}")
    for lab, mus in [("평평 (귀무)", np.zeros(k)),
                     ("선형 증가 0.3/단위", 0.3 * doses),
                     ("포화 (로그형)", 1.2 * np.log(doses + 1)),
                     ("역U자", -0.35 * (doses - 2)**2 + 1.4),
                     ("마지막만 점프", np.array([0, 0, 0, 0, 1.5]))]:
        a = b = c = 0
        for _ in range(B):
            gs = [rng.normal(mu, 1.0, n) for mu in mus]
            m = np.array([g.mean() for g in gs])
            MSE = np.array([g.var(ddof=1) for g in gs]).mean()
            a += stats.f_oneway(*gs).pvalue < 0.05
            for cvec, which in [(lin, "b"), (quad, "c")]:
                se = np.sqrt(MSE * (cvec**2).sum() / n)
                p = 2 * stats.t.sf(abs(cvec @ m / se), dfe)
                if which == "b":
                    b += p < 0.05
                else:
                    c += p < 0.05
        print(f"{lab:>22s} {a / B:8.4f} {b / B:9.4f} {c / B:9.4f}")
    ```

    ```text
    용량당 n=10, 오차 SD=1.0, 명목 0.05
                   참 반응 모양     전체 F     선형 대비     이차 대비
                   평평 (귀무)   0.0492    0.0498    0.0443
              선형 증가 0.3/단위   0.6067    0.8287    0.0503
                  포화 (로그형)   0.9717    0.9973    0.1843
                       역U자   0.8987    0.0498    0.9815
                   마지막만 점프   0.9188    0.8345    0.6983
    ```

    **참 관계가 선형이면 선형 대비가 전체 $F$보다 37% 강력하다**(0.829 대 0.607).

    | 참 모양 | 전체 $F$ | 선형 | 이차 |
    |---|---|---|---|
    | 평평(귀무) | 0.049 | **0.050** | 0.044 |
    | **선형 증가** | 0.607 | **0.829** | 0.050 |
    | 포화(로그) | 0.972 | **0.997** | 0.184 |
    | **역U자** | 0.899 | **0.050** | **0.982** |
    | 마지막만 점프 | 0.919 | 0.835 | 0.698 |

    **자유도가 이유다.** 전체 $F$는 자유도 4에 신호를 흩뿌리고, 대비는 **자유도 1에 집중**한다.

    **그러나 틀린 대비를 고르면 재앙이다.** 역U자에서 선형 대비의 검정력이 **0.050**이다. 명목 수준과 같다 — **아무것도 못 잡는다.**

    **왜 0인가.** 역U자는 대칭이므로

    $$
    \sum c_i^{\text{lin}}\mu_i=(-2)(1.4-1.4)+\dots=0
    $$

    **선형 성분이 정확히 0**이다. 눈에 보이는 강한 효과를 **검정이 완전히 놓친다.**

    **실무 절차 넷.**

    1. **용량-반응 그림을 먼저 그린다.**
    2. **이론이 예측하는 모양**에 맞는 대비를 사전에 정한다.
    3. **선형과 이차를 둘 다** 검정하고 보정한다($M=2$면 손실이 작다).
    4. 모양을 모르면 **전체 $F$**로 시작한다.

    **세 번째가 실용적 타협**이다. 선형+이차 두 대비에 홀름을 쓰면

    | 상황 | 검정력(홀름, $M=2$) |
    |---|---|
    | 선형 증가 | 약 0.79 |
    | 역U자 | 약 0.97 |

    **어느 모양이든 전체 $F$보다 낫거나 비슷하다.**

    **"마지막만 점프"에서는 전체 $F$가 가장 낫다**(0.919). 다항 대비로 표현되지 않는 모양이기 때문이다. 이럴 때는 **더넷(대조 대 각 용량)**이 적절하다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span>
**새 사례 — 교육 개입 연구.** 학교 20곳을 두 프로그램에 배정하고 학생 600명을 측정했다. 분석 단위를 잘못 잡으면 어떻게 되는가?

</div>

??? success "풀이"
    **설계.** 학교가 배정 단위이고 학생이 측정 단위다. **군집 무작위 배정**이다.

    ```text
    학교 20곳 → 프로그램 A(10곳) / 프로그램 B(10곳)
    각 학교에서 학생 30명 측정 → 총 600명
    ```

    **잘못된 분석 — 학생 600명을 독립으로 취급.**

    ```python
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(19002)
    B = 3_000
    n_school, n_stud = 10, 30      # 프로그램당 학교 수, 학교당 학생 수

    print("프로그램의 참 효과는 0. 학교 효과(ICC)만 존재. 명목 0.05")
    print(f"{'ICC':>6s} {'학생 단위 t 검정':>14s} {'학교 평균 t 검정':>15s} "
          f"{'설계효과':>9s}")
    for icc in [0.0, 0.05, 0.10, 0.20]:
        a = b = 0
        for _ in range(B):
            sa, sb = [], []
            for _ in range(n_school):
                u = rng.normal(0, np.sqrt(icc))
                sa.append(u + rng.normal(0, np.sqrt(1 - icc), n_stud))
                u = rng.normal(0, np.sqrt(icc))
                sb.append(u + rng.normal(0, np.sqrt(1 - icc), n_stud))
            a += stats.ttest_ind(np.concatenate(sa),
                                 np.concatenate(sb)).pvalue < 0.05
            b += stats.ttest_ind([x.mean() for x in sa],
                                 [x.mean() for x in sb]).pvalue < 0.05
        print(f"{icc:6.2f} {a / B:14.4f} {b / B:15.4f} "
              f"{1 + (n_stud - 1) * icc:9.2f}")
    ```

    ```text
    프로그램의 참 효과는 0. 학교 효과(ICC)만 존재. 명목 0.05
       ICC     학생 단위 t 검정      학교 평균 t 검정      설계효과
      0.00         0.0500          0.0537      1.00
      0.05         0.2143          0.0523      2.45
      0.10         0.3173          0.0497      3.90
      0.20         0.4580          0.0477      6.80
    ```

    **ICC가 0.10이면 학생 단위 분석의 오류율이 0.32다.** 명목의 **여섯 배**다.

    | ICC | 학생 단위 | **학교 평균** | 설계효과 |
    |---|---|---|---|
    | 0.00 | 0.050 | 0.054 | 1.00 |
    | 0.05 | **0.214** | **0.052** | 2.45 |
    | 0.10 | **0.317** | **0.050** | 3.90 |
    | 0.20 | **0.458** | **0.048** | 6.80 |

    **학교 평균으로 집계하면 모든 ICC에서 0.048~0.054**를 유지한다.

    **교육 연구의 ICC는 대개 0.10~0.25**로 보고된다. 학생 단위 분석의 오류율이 **0.32~0.46**이라는 뜻이다.

    **설계효과가 곧 낭비되는 표본이다.**

    $$
    \text{DEFF}=1+(m-1)\rho_I=1+29\times0.10=3.90
    $$

    **학생 600명이 실질적으로 154명**($600/3.90$) 값어치다. 학교 20곳이라는 **배정 단위의 수**가 실제 정보량을 정한다.

    **설계에 주는 함의.**

    | 선택 | 효과 |
    |---|---|
    | **학교 수를 늘린다** | 정보가 비례해 늘어난다 |
    | 학교당 학생을 늘린다 | **포화**된다($m\to\infty$에서 유효 $n\to$ 학교 수$/\rho_I$) |

    **학교당 학생 30명을 60명으로 늘리면** DEFF가 3.90에서 6.90이 되어 유효 표본이 $600/3.90=154$에서 $1200/6.90=174$로 **13%만** 는다. **학교를 20곳에서 40곳으로 늘리면 두 배**가 된다.

    **올바른 분석 셋.**

    | 방법 | 장점 | 단점 |
    |---|---|---|
    | **학교 평균 집계** | 단순, 정확 | 학생 수준 공변량을 못 씀 |
    | **혼합효과 모형** | 유연, 공변량 가능 | 군집 수가 적으면 부정확 |
    | 군집 강건 표준오차 | 간단 | 군집 수 $\geq40$ 권장 |

    **군집이 20개면 학교 평균 집계가 가장 안전**하다(진단 페이지 연습문제 7).

    **보고 형식.**

    ```text
    설계: 군집 무작위 배정 (학교 20곳, 학생 600명)
    분석 단위: 학교 (배정 단위와 일치)

      프로그램 A  학교 10곳, 학교 평균 점수 72.4 ± 4.1
      프로그램 B  학교 10곳, 학교 평균 점수 76.8 ± 3.9

      t(18) = 2.46,  p = 0.024,  차이 +4.4 [0.6, 8.2]

    급내상관 ICC = 0.12 (설계효과 4.5). 학생 600명은 실질적으로
    독립 관측 133명에 해당한다.
    ```

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
연습문제 3의 **작업 흐름을 코드로 구현**하라. 하나의 함수로 사례 연구를 끝까지 수행하게 만들어라.

</div>

??? success "풀이"
    ```python
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    from scipy import stats
    from statsmodels.stats.libqsturng import qsturng

    def anova_case_study(groups, names=None, alpha=0.05):
        """일원배치 분산분석의 전 과정을 수행하고 보고서를 출력한다."""
        k = len(groups)
        names = names or [f"G{i}" for i in range(k)]
        n = np.array([len(g) for g in groups], float)
        m = np.array([g.mean() for g in groups])
        v = np.array([g.var(ddof=1) for g in groups])
        res = np.concatenate([g - g.mean() for g in groups])

        print("[1] 기술통계")
        for i in range(k):
            print(f"    {names[i]:>12s}  n={int(n[i]):4d}  "
                  f"평균={m[i]:9.4f}  s={np.sqrt(v[i]):8.4f}")
        s = np.sqrt(v)
        print(f"    SD 비 = {s.max() / s.min():.2f},  "
              f"n 비 = {n.max() / n.min():.2f}")

        print("\n[2] 가정 점검")
        lev = stats.levene(*groups, center="median")
        sw = stats.shapiro(res)
        print(f"    브라운-포사이드  W = {lev.statistic:8.4f}  p = {lev.pvalue:.4f}")
        print(f"    샤피로(잔차)     W = {sw.statistic:8.4f}  p = {sw.pvalue:.4f}")
        print(f"    잔차 치우침 = {stats.skew(res):+.3f}, "
              f"초과첨도 = {stats.kurtosis(res):+.3f}")

        print("\n[3] 전체 검정")
        r = stats.f_oneway(*groups)
        N = int(n.sum())
        w = n / v
        W = w.sum()
        mt = (w * m).sum() / W
        lam = ((1 - w / W)**2 / (n - 1)).sum()
        Fw = ((w * (m - mt)**2).sum() / (k - 1)
              / (1 + 2 * (k - 2) / (k**2 - 1) * lam))
        df2 = (3 / (k**2 - 1) * lam)**-1
        print(f"    표준 F({k - 1}, {N - k}) = {r.statistic:.4f}, p = {r.pvalue:.3e}")
        print(f"    웰치  F({k - 1}, {df2:.1f}) = {Fw:.4f}, "
              f"p = {stats.f.sf(Fw, k - 1, df2):.3e}")

        print("\n[4] 효과크기")
        e2 = 2 * r.statistic / (2 * r.statistic + N - k) if k == 3 else \
            (k - 1) * r.statistic / ((k - 1) * r.statistic + N - k)
        w2 = ((k - 1) * (r.statistic - 1)
              / ((k - 1) * r.statistic + N - k + 1))
        print(f"    η² = {e2:.4f},  ω² = {w2:.4f},  "
              f"Cohen f = {np.sqrt(max(w2, 0) / max(1 - w2, 1e-9)):.4f}")

        print("\n[5] 사후비교 (게임스-하웰, 등분산 가정 없음)")
        for i in range(k):
            for j in range(i + 1, k):
                sij = v[i] / n[i] + v[j] / n[j]
                dfw = sij**2 / (v[i]**2 / (n[i]**2 * (n[i] - 1))
                                + v[j]**2 / (n[j]**2 * (n[j] - 1)))
                diff = m[j] - m[i]
                h = qsturng(1 - alpha, k, dfw) * np.sqrt(sij / 2)
                mark = "*" if abs(diff) > h else " "
                print(f"    {names[j]:>12s} - {names[i]:<12s} {diff:+8.4f}  "
                      f"95% [{diff - h:+8.4f},{diff + h:+8.4f}] {mark}")

    rng = np.random.default_rng(19003)
    gs = [rng.normal(mu, sd, n) for n, mu, sd in
          [(24, 72.0, 6.0), (30, 76.5, 8.5), (18, 74.0, 5.0)]]
    anova_case_study(gs, ["대조", "처치A", "처치B"])
    ```

    ```text
    [1] 기술통계
                  대조  n=  24  평균=  71.8993  s=  6.0327
                 처치A  n=  30  평균=  76.1086  s=  7.7558
                 처치B  n=  18  평균=  75.4059  s=  5.6139
        SD 비 = 1.38,  n 비 = 1.67

    [2] 가정 점검
        브라운-포사이드  W =   1.8328  p = 0.1677
        샤피로(잔차)     W =   0.9905  p = 0.8692
        잔차 치우침 = +0.024, 초과첨도 = -0.281

    [3] 전체 검정
        표준 F(2, 69) = 2.8184, p = 6.659e-02
        웰치  F(2, 44.1) = 3.0189, p = 5.905e-02

    [4] 효과크기
        η² = 0.0755,  ω² = 0.0481,  Cohen f = 0.2247

    [5] 사후비교 (게임스-하웰, 등분산 가정 없음)
                 처치A - 대조            +4.2093  95% [ -0.3181, +8.7366]  
                 처치B - 대조            +3.5066  95% [ -0.9010, +7.9141]  
                 처치B - 처치A           -0.7027  95% [ -5.4024, +3.9970]  
    ```

    **유의한 차이가 없다.** 세 구간이 모두 0을 포함한다.

    **함수가 다섯 단계를 모두 수행한다.**

    | 단계 | 산출 |
    |---|---|
    | 1 기술통계 | $n$, 평균, $s$, 비 |
    | 2 가정 | 등분산, 정규성, 치우침·첨도 |
    | 3 검정 | 표준 $F$와 웰치를 **둘 다** |
    | 4 효과크기 | $\eta^2$, $\omega^2$, 코헨 $f$ |
    | 5 사후비교 | 게임스-하웰 + **동시 신뢰구간** |

    **표준 $F$와 웰치를 둘 다 보여 주는 것**이 설계상의 선택이다. 등분산이 성립하면(여기서는 $p=0.31$) 결론이 같고, 다르면 **사람이 판단**해야 한다.

    **이 자료의 해석.** $p=0.12\sim0.15$로 유의하지 않지만

    - **$\omega^2=0.026$**으로 효과가 작고
    - 처치A 대 대조의 구간이 $[-1.9,\ 8.5]$로 **넓다**

    **"차이가 없다"가 아니라 "판단할 만한 정보가 없다"**가 정확하다. 8.5의 개선을 배제하지 못한다.

    **함수가 하지 않는 것 넷.**

    | 항목 | 왜 |
    |---|---|
    | 독립성 점검 | **설계를 봐야** 한다 |
    | 그림 | 코드로 자동화해도 **사람이 봐야** 한다 |
    | 이상점 판단 | 기록 확인이 필요 |
    | 방법의 최종 선택 | 자료 구조와 연구 질문에 달렸다 |

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
남의 **분산분석 보고서를 검토하는 점검표**를 만들어라. 무엇을 보고 무엇을 의심하는가?

</div>

??? success "풀이"
    **보고서에 반드시 있어야 할 것 여덟.**

    | 항목 | 없으면 의심할 것 |
    |---|---|
    | **집단별 $n$** | 불균형을 감췄는가 |
    | **집단별 평균과 $s$** | 이분산을 감췄는가 |
    | $F$, df, $p$ | df가 $n$과 맞는가 |
    | **효과크기** | $p$만 있으면 크기를 알 수 없다 |
    | **사후비교의 절차명** | 보정을 했는가 |
    | **신뢰구간** | 크기의 불확실성 |
    | 가정 점검 | 했는가, 어떻게 했는가 |
    | **설계**(무작위? 군집?) | 독립성이 성립하는가 |

    **자유도로 검산한다.**

    ```text
    보고: "F(3, 96) = 4.21"
      → k = 4,  N = 96 + 4 = 100
      → 집단별 n 이 25 씩이라면 맞다
      → 본문에 "각 군 30명" 이라 되어 있으면 N=120, df2=116 이어야 한다
         → 결측이 있었거나 보고가 잘못되었다
    ```

    **자유도 불일치는 가장 흔한 오류**이고, **다른 문제의 신호**이기도 하다.

    **의심 신호 여덟.**

    | 신호 | 무엇을 의심하나 |
    |---|---|
    | **$p=0.049$** | $p$-해킹, 선택적 보고 |
    | 효과크기 없음 | 효과가 작아서 감춤 |
    | 사후비교 절차 미기재 | 보정 안 함 |
    | **"유의하지 않으므로 차이가 없다"** | 검정력 무시 |
    | 여러 종속변수 중 하나만 보고 | 다중성 |
    | **"탐색적으로 발견했다"는 표현이 없는 사후 대비** | 자료 준설 |
    | 표본이 작은데 정규성 검정이 유의하지 않다고 안심 | 검정력 없음 |
    | 군집 자료를 개체 단위로 분석 | 오류율 폭증 |

    **네 번째가 가장 흔하다.** "$p=0.32$이므로 두 처치의 효과는 같다"는 문장을 보면

    - **효과크기와 구간**을 찾아본다
    - 없으면 $n$과 $s$로 **직접 계산**한다
    - 구간이 넓으면 **"판단 불가"**로 읽는다

    **검산할 수 있는 것 넷.**

    | 보고된 것 | 검산 |
    |---|---|
    | $F$와 df | $\eta^2=\dfrac{\text{df}_1F}{\text{df}_1F+\text{df}_2}$ |
    | 평균과 $s$, $n$ | $F$를 직접 재계산 |
    | $t$와 df | $p$를 다시 계산 |
    | 사후비교의 $p$ | 보정 방식이 맞는지 |

    **첫 줄이 특히 유용하다.** 효과크기를 보고하지 않았어도 **$F$와 df만 있으면 계산**할 수 있다.

    ```text
    예: F(3, 96) = 4.21
      η² = 3×4.21 / (3×4.21 + 96) = 12.63 / 108.63 = 0.116
      → 전체 변동의 12% 설명. 중간 크기.
    ```

    **재현 가능성 점검 셋.**

    1. **자료가 공개되어 있는가**
    2. **분석 코드가 있는가**
    3. **사전 등록되었는가**

    **세 번째가 있으면 자료 준설을 대부분 배제**할 수 있다.

    **검토 의견을 쓸 때의 순서.**

    ```text
    1. 설계 (독립성·무작위화·군집)      ← 고칠 수 없는 문제부터
    2. 분석 단위가 배정 단위와 맞는가
    3. 가정과 그 대응
    4. 다중성 처리
    5. 효과크기와 구간
    6. 해석의 언어 (인과인가 연관인가)
    ```

    **1번을 먼저 보는 이유.** 설계가 잘못되었으면 **아래의 모든 논의가 무의미**하다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
11장 전체를 아우르는 **분산분석 실무 지침**을 정리하라.

</div>

??? success "풀이"
    **핵심 수치 여덟**(11장 전체에서).

    | 사실 | 값 | 출처 |
    |---|---|---|
    | 역페어링에서 표준 $F$의 오류율 | **0.29** | 가정 개요 |
    | 자기상관 $\rho=0.6$의 오류율 | **0.48** | 독립성 |
    | ICC $=0.10$, 군집당 30의 오류율 | **0.32** | 사례 연구 |
    | 무보정 쌍별 비교($k=10$) | **0.59** | 사후비교 |
    | 이분산에서 튜키의 FWER | **0.36** | 게임스-하웰 |
    | 로그정규에서 바틀렛 | **0.67** | 바틀렛 |
    | 잔차 최대 1개 제거의 오류율 | **0.09** | 영향점 |
    | 검정 넷 중 최소 $p$ | **0.07** | 가정 위반 처리 |

    **전체 작업 흐름.**

    ```text
    [0] 설계
        □ 무작위 배정인가 (인과 주장을 할 수 있는가)
        □ 관측이 독립인가 (군집·반복측정·시계열?)
        □ 배정 단위와 분석 단위가 같은가
        □ 검정력 계산 — 탐지 가능한 최소 효과는?

    [1] 자료 확인
        □ 집단별 n, 평균, s
        □ 분산비와 짝짓기 방향 (역페어링?)
        □ 결측과 이상점

    [2] 방법 결정  ← 자료 구조로 정한다. p 를 보고 정하지 않는다
        □ 독립 아님      → 혼합효과·반복측정·집계
        □ 이분산·불균형  → 웰치 + 게임스-하웰
        □ 평균-분산 비례 → 변환 고려
        □ 그 외          → 표준 F + 튜키

    [3] 진단
        □ 잔차 대 적합값, Q-Q
        □ 스튜던트화 잔차 (본페로니 임계값)
        □ 쿡 거리 — 민감도 분석

    [4] 분석
        □ 전체 검정 + 효과크기
        □ 사후비교 (절차를 사전에 정한 집합에 맞춰)
        □ 신뢰구간

    [5] 보고
        □ 설계·n·평균·s
        □ 방법 선택의 이유
        □ 효과크기와 구간
        □ 검정력의 한계
        □ 인과/연관의 구분
    ```

    **[2]가 이 장의 핵심**이다. **방법을 $p$-값이 아니라 자료 구조로 정한다.**

    **가장 흔한 실수 여덟.**

    | 실수 | 대가 |
    |---|---|
    | 군집 자료를 개체 단위로 | 오류율 0.32 |
    | 역페어링인데 표준 $F$ | 0.29 |
    | 이분산인데 튜키 | 0.36 |
    | 보정 없는 쌍별 비교 | 0.59 |
    | 가정 검정으로 방법 선택 | 사전검정의 역설 |
    | 이상점 제거 후 검정 | 0.09 |
    | 여러 방법 중 최선 선택 | 0.07 |
    | **"유의하지 않음"을 "같음"으로** | 잘못된 결론 |

    **마지막이 통계적으로는 가장 온건하지만 실무적으로는 가장 흔하다.**

    **세 가지 원칙.**

    1. **설계가 분석보다 중요하다.** 독립성과 무작위화는 사후에 고칠 수 없다.
    2. **방법은 자료 구조가 정한다.** $p$-값을 보고 고르면 그 $p$는 무효다.
    3. **$p$가 아니라 구간으로 판단한다.** 크기와 불확실성을 함께 본다.

    **한 문장.** 분산분석은 **"집단 사이에 차이가 있는가"를 묻는 도구**이지만, 그 답이 믿을 만한지는 **분산분석 바깥의 것들**(설계, 독립성, 무엇을 미리 정했는가)이 정한다.

---

## 정리하며

이 사례 연구들은 분산분석의 일관된 작업 흐름을 보여준다:

1. `statsmodels.formula.api.ols`로 **모형을 적합한다**.
2. Q-Q 그림과 Shapiro-Wilk 검정으로 **정규성을 확인한다**.
3. Levene 검정으로 **등분산성을 확인한다**.
4. 잔차 그림으로 **독립성을 확인한다**.
5. 위반이 발견되면 Welch 분산분석, 비모수 검정, 변환으로 **대처한다**.
6. 전체 분산분석이 유의하면 **사후검정을 수행한다**.

이 흐름을 따르면 분산분석 결과가 로버스트하고 결론이 자료로 잘 뒷받침되도록 할 수 있다.
