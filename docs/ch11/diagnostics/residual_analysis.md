# 잔차 분석

## 개요

잔차 분석은 분산분석의 결정적인 진단 도구이다. 잔차는 관측된 자료점과 모형의 예측값의 차이이다:

$$
e_{ij} = Y_{ij} - \hat{Y}_{ij} = Y_{ij} - \bar{Y}_{i\cdot}
$$

여기서 $Y_{ij}$는 집단 $i$의 $j$번째 관측값이고 $\bar{Y}_{i\cdot}$는 집단 $i$의 평균이다. 모형이 올바르게 설정되었다면 잔차는 0 주위에 무작위로 분포하며 분산이 일정하고 체계적인 패턴이 없어야 한다.

## 설정

<div class="codebox" markdown>

### 예제 1. 진단에 쓸 모형 준비 { .eg }

```python
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols

# 이 페이지의 진단은 모두 아래 모형 하나를 놓고 수행한다.
# 집단마다 표준편차를 1.0, 1.3, 1.6으로 다르게 주었고,
# 집단 C에 이상점을 하나 심어 두었다.
rng = np.random.default_rng(42)
n = 20
response = np.concatenate([
    rng.normal(10.0, 1.0, n),
    rng.normal(10.8, 1.3, n),
    rng.normal(12.0, 1.6, n),
])
response[-1] = 20.0                     # 마지막 관측값을 이상점으로 만든다
data = pd.DataFrame({
    "group": np.repeat(["A", "B", "C"], n),
    "response": response,
})
group1 = data.loc[data["group"] == "A", "response"]
group2 = data.loc[data["group"] == "B", "response"]
group3 = data.loc[data["group"] == "C", "response"]

model = ols("response ~ C(group)", data=data).fit()

print(data.groupby("group").response.agg(["count", "mean", "std"]).round(3))
print(f"\nF = {model.fvalue:.4f}, p = {model.f_pvalue:.4f}")
```

출력:

```
       count    mean    std
group                      
A         20   9.967  0.870
B         20  10.942  1.034
C         20  12.513  2.077

F = 16.1314, p = 0.0000
```

이상점 하나가 집단 C의 표준편차를 1.15에서 2.08로 키웠다. 아래 진단들이 이것을 잡아내는지 보라.

</div>

## 잔차 대 적합값 그림

가장 유익한 진단 그림은 잔차를 적합값에 대해 그린 것이다. 일원배치 분산분석에서 적합값은 곧 집단 평균이므로, 각 집단 평균 위치에 잔차가 수직 띠로 나타난다.

<div class="codebox" markdown>

### 예제 2. 잔차 대 적합값 그림 { .eg }

```python
import matplotlib.pyplot as plt

# 잔차 대 적합값 그림은 진단의 출발점이다. 점들이 0 선 둘레에 폭을 일정하게
# 유지하며 흩어져 있으면 좋다. 깔때기 모양이면 등분산이 깨진 것이고, 굽은
# 모양이면 모형이 놓친 구조가 남아 있다는 뜻이다.
plt.scatter(model.fittedvalues, model.resid, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Fitted Values")
plt.show()
```

![잔차 대 적합값](./img/residual_analysis_63.png)

세로 띠가 셋 있고, 그것이 집단 셋이다. 회귀분석의 잔차 그림처럼 연속적으로 퍼지지 않는 것은 적합값이 집단평균 세 개뿐이기 때문이다.

오른쪽 띠(집단 C)가 다른 둘보다 위아래로 넓고, 그 위에 7 남짓 떨어진 점 하나가 홀로 있다. 심어 둔 이상점이다.

</div>

### 패턴 알아보기

잔차 그림의 다음 패턴들은 특정한 가정 위반을 알려준다:

**깔때기 모양(이분산):**
넓어지거나 좁아지는 패턴은 독립변수의 수준에 따라 잔차의 분산이 다름을 나타낸다. 등분산성 가정을 위반하며 F-검정을 왜곡할 수 있다.

**곡률(비선형성):**
휘어진 패턴은 독립변수와 종속변수의 관계가 선형이 아님을 시사한다. 다항 항, 비선형 모형, 또는 자료 변환이 필요할 수 있다.

**군집(비독립성):**
잔차가 무리를 이루면 군집 안의 관측값이 상관되어 있어 독립성 가정을 위반함을 나타낼 수 있다. 계층적이거나 내포된 자료 구조에서 자주 나타난다.

**이상점:**
0선에서 멀리 떨어진 개별 점은 분석에 지나치게 큰 영향을 주는 이상점일 수 있다.

## 표준화 잔차

표준화 잔차는 각 잔차를 그 표준편차의 추정값으로 나누어 모든 잔차를 공통 척도에 놓는다:

$$
r_i = \frac{e_i}{\hat{\sigma}\sqrt{1 - h_{ii}}}
$$

여기서 $\hat{\sigma}$는 추정된 표준편차이고 $h_{ii}$는 관측값 $i$의 지렛값이다. 모형 가정 아래에서 표준화 잔차는 근사적으로 표준정규분포를 따라야 한다.

<div class="codebox" markdown>

### 예제 3. 표준화 잔차 { .eg }

```python
import numpy as np

# 표준화 잔차는 잔차를 그 표준오차로 나눈 것이다. 단위가 사라지므로
# 어느 자료에서든 ±2 를 같은 뜻으로 읽을 수 있다.
influence = model.get_influence()
standardized_resid = influence.resid_studentized_internal

plt.scatter(model.fittedvalues, standardized_resid, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.axhline(y=2, color='gray', linestyle=':', alpha=0.5)
plt.axhline(y=-2, color='gray', linestyle=':', alpha=0.5)
plt.xlabel("Fitted Values")
plt.ylabel("Standardized Residuals")
plt.title("Standardized Residuals vs. Fitted Values")
plt.show()
```

![표준화 잔차](./img/residual_analysis_100.png)

세로축이 표준편차 단위로 바뀌어 회색 기준선($\pm 2$)과 곧바로 비교할 수 있다. 이상점 하나가 4를 훌쩍 넘고, 나머지는 대부분 $\pm 2$ 안에 있다.

원래 잔차 그림과 모양이 거의 같아 보이지만 척도가 다르다. 표준화는 지렛값 $h_{ii}$도 함께 반영하므로, 설계가 불균형이면 두 그림의 모양이 눈에 띄게 달라진다.

$|r_i| > 2$인 관측값은 자세히 살펴볼 만하고, $|r_i| > 3$인 관측값은 이상점의 유력한 후보이다.

</div>

## 척도-위치 그림

척도-위치 그림은 $\sqrt{|r_i|}$를 적합값에 대해 그린 것으로 등분산성을 평가하는 데 유용하다. 추세선이 수평이고 점들이 고르게 퍼져 있으면 분산이 일정함을 나타낸다.

<div class="codebox" markdown>

### 예제 4. 척도-위치 그림 { .eg }

```python
# 척도-위치 그림은 부호를 없애고 크기만 본다. 제곱근을 씌우는 것은 큰 값이
# 그림을 독차지하지 않게 하려는 것이다. 추세선이 평평하면 등분산이다.
plt.scatter(model.fittedvalues, np.sqrt(np.abs(standardized_resid)), alpha=0.6)
plt.xlabel("Fitted Values")
plt.ylabel(r"$\sqrt{|\mathrm{Standardized\ Residuals}|}$")
plt.title("Scale-Location Plot")
plt.show()
```

![척도-위치 그림](./img/residual_analysis_122.png)

세로축이 $\sqrt{|r_i|}$라 부호가 사라지고 **크기만** 남는다. 그래서 "어느 쪽으로 벗어났는가"가 아니라 "얼마나 퍼져 있는가"에 집중할 수 있다.

세 띠의 높이를 비교하면 오른쪽 띠가 조금 더 위로 퍼져 있고 이상점이 2를 넘는다. 등분산이라면 세 띠의 평균 높이가 비슷해야 한다.

</div>

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
집단이 넷인 일원배치 분산분석에서 잔차 대 적합값 그림에 뚜렷한 깔때기 모양(적합값이 커질수록 잔차가 퍼짐)이 보인다. 어느 분산분석 가정이 위반되었는지 밝히고 문제를 다루는 두 가지 접근을 기술하라.

</div>

??? success "풀이"
    깔때기 모양은 **이분산**을 나타낸다. 적합값(집단 평균)이 커질수록 잔차의 분산이 커진다. 두 가지 접근:

    1. **Welch 분산분석:** 등분산을 가정하지 않고 집단마다 다른 분산을 반영하도록 자유도를 조정한다.

    2. **분산 안정화 변환:** 반응변수에 로그나 제곱근 변환을 적용한다. 분산이 평균에 비례하면(도수 자료나 양의 연속형 자료에서 흔하다) $\log(Y)$가 흔히 분산을 안정화한다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
학교 다섯 곳의 학생 시험 점수에 분산분석을 수행했다. 잔차 히스토그램에 하나의 종 모양 대신 두 개의 뚜렷한 봉우리가 보인다. 무엇을 뜻할 수 있으며 연구자는 어떻게 해야 하는가?

</div>

??? success "풀이"
    잔차 히스토그램의 **이봉** 형태는 모형에 포함되지 않은 **집단 변수나 중요한 공변량이 빠져 있음**을 시사한다. 예를 들어 각 학교 안에 점수가 체계적으로 다른 두 하위 집단(서로 다른 프로그램이나 학년의 학생)이 있을 수 있다.

    연구자는 추가 요인의 가능성을 조사하고, 그 변수를 포함하는 이원배치 분산분석이나 공분산분석을 적합하는 것을 고려해야 한다. 자연스러운 집단 구분을 찾아 모형에 넣으면 잔차분산이 줄고 모형이 개선된다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
분산분석에서 정규성을 평가할 때 원래 잔차보다 표준화 잔차가 선호되는 이유를 설명하라. 표준화 잔차는 어떻게 계산하는가?

</div>

??? success "풀이"
    원래 잔차 $e_i = Y_i - \hat{Y}_i$는 관측값의 지렛값에 따라 분산이 달라진다: $\text{Var}(e_i) = \sigma^2(1 - h_{ii})$이며 $h_{ii}$가 지렛값이다. 따라서 관측값들 사이에서 원래 잔차의 크기를 그대로 비교하면 오도할 수 있다.

    **표준화(내부 스튜던트화) 잔차**는 각 원래 잔차를 그 추정 표준편차로 나눈다:

    $$
    r_i = \frac{e_i}{\hat{\sigma}\sqrt{1 - h_{ii}}}
    $$

    모형 가정 아래에서 이 값들은 근사적으로 표준정규분포를 따르므로 관측값 사이에서 직접 비교할 수 있고, Q-Q 그림이나 $\pm 2$ 경험 법칙으로 평가하기 쉬워진다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
연습문제 3이 말한 표준화가 **왜 필요한지** 수치로 보여라. 분산분석 잔차의 분산은 일정한가?

</div>

??? success "풀이"
    **이론.** 잔차 벡터는 $\mathbf e=(\mathbf I-\mathbf H)\mathbf y$이므로

    $$
    \operatorname{Var}(e_i)=\sigma^2(1-h_{ii})
    $$

    **일원배치에서는 $h_{ii}=1/n_i$**이므로

    $$
    \operatorname{Var}(e_{ij})=\sigma^2\left(1-\frac{1}{n_i}\right)
    $$

    **작은 집단의 잔차가 체계적으로 작다.**

    ```python
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    from statsmodels.formula.api import ols

    rng = np.random.default_rng(9001)
    B = 30_000
    NS = [5, 15, 40]
    acc = {n: [] for n in NS}
    for _ in range(B):
        rows = [pd.DataFrame({"y": rng.normal(0, 1.0, n), "g": f"G{i}"})
                for i, n in enumerate(NS)]
        df = pd.concat(rows, ignore_index=True)
        r = ols("y ~ C(g)", data=df).fit().resid.values
        i = 0
        for n in NS:
            acc[n].append(r[i:i + n])
            i += n

    print("모든 집단의 참 σ = 1.0 (집단 크기만 다름)")
    print(f"{'집단 크기 n':>11s} {'잔차의 SD':>10s} {'이론 σ√(1-1/n)':>15s} {'h=1/n':>8s}")
    for n in NS:
        a = np.concatenate(acc[n])
        print(f"{n:11d} {a.std(ddof=0):10.4f} {np.sqrt(1 - 1 / n):15.4f} {1 / n:8.4f}")
    ```

    ```text
    모든 집단의 참 σ = 1.0 (집단 크기만 다름)
        집단 크기 n     잔차의 SD    이론 σ√(1-1/n)    h=1/n
              5     0.8960          0.8944   0.2000
             15     0.9658          0.9661   0.0667
             40     0.9880          0.9874   0.0250
    ```

    **이론값과 소수점 셋째 자리까지 일치한다.**

    | 집단 크기 | 잔차 SD | 참 $\sigma$ 대비 |
    |---|---|---|
    | 5 | 0.896 | $-10.4\%$ |
    | 15 | 0.966 | $-3.4\%$ |
    | 40 | 0.988 | $-1.2\%$ |

    **$n=5$ 집단의 잔차가 10% 작다.** 참 분산은 모두 같은데도 그렇다.

    **이것이 잔차 그림의 착시를 만든다.** 작은 집단이 **덜 퍼져 보여** "그 집단의 분산이 작다"고 오독하기 쉽다.

    **표준화 잔차가 이를 고친다.**

    $$
    r_{ij}=\frac{e_{ij}}{\hat\sigma\sqrt{1-1/n_i}}
    $$

    **그런데 왜 이 보정이 작은가.** $n\geq15$이면 3% 이하다. **표본이 아주 작은 집단이 있을 때만** 실질적으로 중요하다.

    **잔차 대신 집단별 표본표준편차 $s_i$를 직접 보는 것이 더 낫다.** $s_i$는 $\sigma_i$의 불편 추정에 가깝고 해석도 직접적이다.

    | 목적 | 볼 것 |
    |---|---|
    | 등분산 진단 | **집단별 $s_i$** |
    | 이상점 탐지 | **표준화 잔차**(연습문제 9) |
    | 정규성 진단 | 표준화 잔차의 Q-Q |
    | 그림의 인상 | 원 잔차도 무방($n$이 고르면) |

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
연습문제 2가 말한 **두 봉우리 잔차**의 원인을 네 가지 만들어 **구별할 수 있는 지표**를 찾아라.

</div>

??? success "풀이"
    ```python
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    from scipy import stats
    from statsmodels.formula.api import ols

    rng = np.random.default_rng(9002)
    cases = {}

    rows = []                                   # (1) 숨은 이분 잠재변수
    for g in range(3):
        h = rng.binomial(1, 0.5, 60)
        rows.append(pd.DataFrame({"y": 10 + 2 * g + 4 * h + rng.normal(0, 1, 60),
                                  "g": f"G{g}"}))
    cases["숨은 하위집단 (이분 잠재변수)"] = pd.concat(rows, ignore_index=True)

    rows = [pd.DataFrame({"y": 10 + 2 * g + rng.normal(0, 1, 60), "g": f"G{g}"})
            for g in range(3)]                  # (2) 문제 없음
    cases["문제 없음 (비교용)"] = pd.concat(rows, ignore_index=True)

    rows = [pd.DataFrame({"y": np.clip(10 + 2 * g + rng.normal(0, 3, 60), 8, 16),
                          "g": f"G{g}"}) for g in range(3)]    # (3) 절단
    cases["천장·바닥 효과 (절단)"] = pd.concat(rows, ignore_index=True)

    rows = [pd.DataFrame({"y": rng.binomial(1, 0.3 + 0.15 * g, 60).astype(float),
                          "g": f"G{g}"}) for g in range(3)]    # (4) 이분 반응
    cases["사실은 이분 반응"] = pd.concat(rows, ignore_index=True)

    def center_mass(r):
        """표준화 잔차가 0 근처에 얼마나 몰려 있는가. 정규면 약 0.383."""
        z = (r - r.mean()) / r.std(ddof=1)
        return np.mean(np.abs(z) < 0.5)

    print(f"{'상황':>26s} {'치우침':>7s} {'초과첨도':>8s} {'중심밀도':>9s} {'샤피로 p':>9s}")
    for lab, df in cases.items():
        r = ols("y ~ C(g)", data=df).fit().resid.values
        print(f"{lab:>26s} {stats.skew(r):7.3f} {stats.kurtosis(r):8.3f} "
              f"{center_mass(r):9.3f} {stats.shapiro(r).pvalue:9.4f}")
    print(f"\n  (정규 잔차라면 |z|<0.5 의 비율이 약 {2 * stats.norm.cdf(0.5) - 1:.3f})")
    ```

    ```text
                            상황     치우침     초과첨도      중심밀도     샤피로 p
             숨은 하위집단 (이분 잠재변수)  -0.289   -1.116     0.228    0.0000
                   문제 없음 (비교용)  -0.199   -0.027     0.400    0.1374
                 천장·바닥 효과 (절단)  -0.126   -0.925     0.278    0.0008
                     사실은 이분 반응   0.172   -1.778     0.000    0.0000

      (정규 잔차라면 |z|<0.5 의 비율이 약 0.383)
    ```

    **두 봉우리의 공통 신호는 **음의 초과첨도**다.**

    | 상황 | 초과첨도 | 중심밀도 | 해석 |
    |---|---|---|---|
    | 문제 없음 | $-0.03$ | **0.400** | 정규에 부합 |
    | 숨은 하위집단 | $-1.12$ | 0.228 | 두 봉우리 |
    | 절단 | $-0.93$ | 0.278 | 양 끝에 쌓임 |
    | **이분 반응** | $\mathbf{-1.78}$ | **0.000** | 두 점만 존재 |

    **"중심밀도"가 진단에 유용하다.** 정규라면 $|z|<0.5$인 관측이 38.3%인데, 두 봉우리면 **가운데가 비어** 그 비율이 떨어진다.

    **초과첨도가 음수인 것이 핵심**이다. 두꺼운 꼬리는 **양의** 초과첨도를 주지만, **두 봉우리는 음의** 초과첨도를 준다. 흔히 "첨도가 낮다 = 평평하다"로만 배우는데, **두 봉우리도 낮은 첨도**를 만든다.

    **네 원인을 구별하는 법.**

    | 원인 | 결정적 단서 | 처방 |
    |---|---|---|
    | **숨은 하위집단** | **집단 안에서도** 두 봉우리 | 잠재 변수를 찾아 모형에 추가 |
    | **절단** | 값이 **경계에 쌓임** | 토빗 모형, 절단 회귀 |
    | **이분 반응** | 값의 종류가 **둘뿐** | **로지스틱 회귀** |
    | 집단 평균 차이 | **잔차가 아니라 원자료**만 두 봉우리 | 문제 없음(정규성 페이지 연습문제 5) |

    **마지막 줄을 먼저 확인**해야 한다. 원자료 히스토그램이 두 봉우리여도 **잔차가 단봉이면 아무 문제가 없다.**

    **연습문제 2의 답.** 학교 다섯 곳의 시험 점수에서 **잔차가** 두 봉우리라면

    1. **학교 안에 또 다른 구분**이 있는지 본다(학년, 계열, 성별).
    2. 그 변수를 자료에서 찾을 수 있으면 **모형에 추가**한다.
    3. 찾을 수 없으면 **혼합 모형**(latent class)이나 **중앙값 기반 방법**을 고려한다.
    4. 점수가 상한·하한에 몰려 있지 않은지 확인한다(**천장 효과**).

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
잔차 그림의 **깔때기 모양이 착시일 수 있다**는 것을 보여라. 등분산인데도 집단별 산포가 얼마나 달라 보이는가?

</div>

??? success "풀이"
    ```python
    import numpy as np

    rng = np.random.default_rng(9003)
    B = 20_000
    print("모든 집단의 참 σ 가 같을 때, 표본 SD 의 최대/최소 비")
    print(f"{'k':>3s} {'집단당 n':>8s} {'비의 중앙값':>11s} {'90 백분위':>10s} "
          f"{'비>2 확률':>10s} {'비>3 확률':>10s}")
    for k, n in [(3, 5), (3, 10), (3, 20), (3, 50), (5, 10), (8, 10)]:
        r = []
        for _ in range(B // 4):
            s = np.array([rng.normal(0, 1, n).std(ddof=1) for _ in range(k)])
            r.append(s.max() / s.min())
        r = np.array(r)
        print(f"{k:3d} {n:8d} {np.median(r):11.3f} {np.quantile(r, 0.9):10.3f} "
              f"{(r > 2).mean():10.4f} {(r > 3).mean():10.4f}")
    ```

    ```text
    모든 집단의 참 σ 가 같을 때, 표본 SD 의 최대/최소 비
      k    집단당 n      비의 중앙값     90 백분위     비>2 확률     비>3 확률
      3        5       1.831      3.245     0.4092     0.1294
      3       10       1.475      2.067     0.1184     0.0078
      3       20       1.305      1.629     0.0112     0.0000
      3       50       1.176      1.347     0.0000     0.0000
      5       10       1.743      2.392     0.2756     0.0212
      8       10       1.991      2.693     0.4888     0.0504
    ```

    **$k=3$, $n=5$에서 표본 SD 비의 중앙값이 1.83이다.** 참 분산이 모두 같은데도 **절반의 자료에서 산포가 두 배 가까이 달라 보인다.**

    | 설계 | 중앙값 | 비$>2$ 확률 |
    |---|---|---|
    | $k=3$, $n=5$ | 1.83 | **0.409** |
    | $k=3$, $n=10$ | 1.48 | 0.118 |
    | $k=3$, $n=50$ | 1.18 | **0.000** |
    | $k=8$, $n=10$ | 1.99 | **0.489** |

    **집단 수가 많아도 착시가 커진다.** $k=8$, $n=10$에서 절반의 자료가 비 2배를 넘는다. **최댓값과 최솟값을 비교하는 것 자체가 다중성**이기 때문이다.

    **경험칙 "SD 비가 2 이하면 괜찮다"를 재해석해야 한다.**

    | $n$ | 비 2배가 뜻하는 것 |
    |---|---|
    | 5 | **우연으로 흔한 일**(41%) |
    | 10 | 가끔 있는 일(12%) |
    | 20 | **드문 일**(1%) |
    | 50 | **거의 없는 일**(0%) |

    **$n$이 크면 SD 비 2배는 진짜 신호**이고, $n$이 작으면 **아무 정보도 아니다.**

    **그림을 읽을 때의 지침 넷.**

    1. **집단당 $n$을 그림에 표시**한다. $n$을 모르면 산포를 해석할 수 없다.
    2. **작은 집단의 산포는 믿지 않는다.** $n=5$면 $s$의 95% 구간이 $[0.6s,\ 2.9s]$다.
    3. **경향을 본다.** 적합값이 커질수록 **단조롭게** 퍼지면 신호다. 한 집단만 튀면 우연일 수 있다.
    4. **검정과 병행**한다. 브라운-포사이드가 그림의 인상을 확인해 준다.

    **세 번째가 실용적으로 가장 중요하다.** 깔때기 모양의 핵심은 **"한 집단이 다르다"가 아니라 "적합값과 함께 단조롭게 변한다"**는 것이다. 그것이 평균-분산 관계의 신호이고, 변환으로 고칠 수 있는 유형이다(등분산성 페이지 연습문제 7·8).

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
**분산분석의 잔차 대 적합값 그림**은 회귀의 그것과 무엇이 다른가? 무엇을 볼 수 있고 무엇을 볼 수 없는가?

</div>

??? success "풀이"
    **결정적 차이 — 적합값이 $k$개뿐이다.**

    $$
    \hat y_{ij}=\bar y_{i\cdot}
    $$

    이므로 그림의 가로축에 **$k$개의 세로줄**만 나타난다. 연속형 설명변수가 있는 회귀와 달리 **가로축이 연속이 아니다.**

    | | 회귀 | **일원배치 분산분석** |
    |---|---|---|
    | 적합값의 개수 | $N$개(거의 모두 다름) | **$k$개** |
    | 그림의 모양 | 산점도 | **$k$개의 세로줄** |
    | 사실상 같은 것 | — | **집단별 점 그림** |

    **따라서 분산분석의 잔차 대 적합값 그림은 "집단별 상자그림"과 같은 정보**를 담는다. 가로축의 위치가 집단 평균의 크기를 반영한다는 점만 다르다.

    **볼 수 있는 것.**

    | 관측 | 해석 |
    |---|---|
    | 세로줄의 길이가 다름 | **이분산** |
    | 길이가 적합값과 함께 증가 | **평균-분산 관계** → 변환 |
    | 한 줄에만 먼 점 | 그 집단의 **이상점** |
    | 줄 안에서 위/아래 쏠림 | **치우침** |

    **볼 수 없는 것.**

    | 무엇 | 왜 |
    |---|---|
    | **비선형성** | 적합값이 $k$개뿐이라 곡선이 정의되지 않음 |
    | 독립성 | 순서 정보가 없음 |
    | 정규성의 세부 | Q-Q 그림이 필요 |

    **첫 줄이 중요하다.** 회귀에서 잔차 대 적합값 그림의 주된 용도는 **함수형 오설정 탐지**인데, **순수 범주형 분산분석에는 그런 문제가 없다**(선형성 페이지 연습문제 1). 적합값이 곧 집단 평균이므로 **모형은 항상 완벽하게 적합**한다.

    $$
    \sum_j e_{ij}=0\quad\text{(각 집단마다)}
    $$

    **각 세로줄의 평균이 정확히 0**이다. 이것은 자료가 아니라 **최소제곱의 항등식**이다.

    **공분산분석이면 이야기가 달라진다.** 연속형 공변량이 들어오면 적합값이 연속이 되고, **회귀의 진단이 그대로 필요**해진다.

    **그림을 고르는 지침.**

    | 목적 | 그림 |
    |---|---|
    | 이분산 | **집단별 상자그림** 또는 잔차 대 적합값 |
    | 평균-분산 관계 | **스프레드-레벨 그림**(로그-로그) |
    | 정규성 | **잔차의 Q-Q** |
    | 이상점 | **표준화 잔차 대 관측 번호** |
    | 독립성 | **잔차 대 수집 순서** |
    | 공변량의 함수형 | **성분+잔차 그림** |

    **잔차 대 적합값 하나로 모든 것을 보려는 것이 흔한 실수**다. 분산분석에서는 특히 **집단별 상자그림이 같은 정보를 더 읽기 쉽게** 준다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
**잔차는 서로 독립이 아니다.** 그 상관 구조를 구하고 수치로 확인하라. 진단에 어떤 영향을 주는가?

</div>

??? success "풀이"
    **유도.** 집단 $i$ 안에서 $e_{ij}=y_{ij}-\bar y_{i\cdot}$이므로

    $$
    \operatorname{Cov}(e_{ij},e_{ij'})
    =\operatorname{Cov}\!\left(y_{ij}-\bar y_i,\ y_{ij'}-\bar y_i\right)
    =0-\frac{\sigma^2}{n_i}-\frac{\sigma^2}{n_i}+\frac{\sigma^2}{n_i}
    =-\frac{\sigma^2}{n_i}
    $$

    이고 $\operatorname{Var}(e_{ij})=\sigma^2(1-1/n_i)$이므로

    $$
    \operatorname{Corr}(e_{ij},e_{ij'})=\frac{-\sigma^2/n_i}{\sigma^2(1-1/n_i)}=-\frac{1}{n_i-1}
    $$

    ```python
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    from statsmodels.formula.api import ols

    rng = np.random.default_rng(9001)
    B = 30_000
    NS = [5, 15, 40]
    acc = {n: [] for n in NS}
    for _ in range(B):
        rows = [pd.DataFrame({"y": rng.normal(0, 1.0, n), "g": f"G{i}"})
                for i, n in enumerate(NS)]
        df = pd.concat(rows, ignore_index=True)
        r = ols("y ~ C(g)", data=df).fit().resid.values
        i = 0
        for n in NS:
            acc[n].append(r[i:i + n])
            i += n

    a5 = np.array(acc[5])
    a40 = np.array(acc[40])
    print(f"집단 크기 5 의 e1, e2 상관  = {np.corrcoef(a5[:, 0], a5[:, 1])[0, 1]:.4f}"
          f"   (이론 -1/(n-1) = {-1 / 4:.4f})")
    print(f"집단 크기 40 의 e1, e2 상관 = "
          f"{np.corrcoef(a40[:, 0], a40[:, 1])[0, 1]:.4f}   (이론 {-1 / 39:.4f})")
    print(f"집단 크기 5 의 잔차합: 평균 {a5.sum(1).mean():.2e}, "
          f"SD {a5.sum(1).std():.2e}   (항등적으로 0)")
    ```

    ```text
    집단 크기 5 의 e1, e2 상관  = -0.2447   (이론 -1/(n-1) = -0.2500)
    집단 크기 40 의 e1, e2 상관 = -0.0230   (이론 -0.0256)
    집단 크기 5 의 잔차합: 평균 -1.53e-17, SD 7.67e-15   (항등적으로 0)
    ```

    **이론값과 일치한다.** 잔차합은 부동소수점 오차 수준($10^{-15}$)에서 정확히 0이다.

    | 집단 크기 | 잔차 사이 상관 |
    |---|---|
    | 5 | $-0.245$ |
    | 15 | $-0.071$ |
    | 40 | $-0.023$ |

    **상관이 음수인 이유.** 한 잔차가 크면 집단 평균이 그쪽으로 끌려가 **나머지 잔차가 반대편으로** 밀린다. 합이 0으로 고정되어 있기 때문이다.

    **진단에 주는 영향 셋.**

    | 영향 | 내용 |
    |---|---|
    | **정규성 검정** | 잔차가 독립이 아니므로 이론적으로 부정확 |
    | **이상점 검정** | 하나가 크면 나머지를 **밀어내 감춘다**(가림 현상) |
    | 더빈-왓슨 | 원래 음의 상관이 있어 $d$가 2보다 크게 나오는 경향 |

    **그런데 실무적으로는 대체로 무해하다.** 정규성 페이지 연습문제 9에서 확인했듯, 잔차에 대한 샤피로 검정의 기각률은 0.049~0.050으로 명목을 지킨다. **$n$이 작을 때만 약간 보수적**이다.

    **가림 현상이 진짜 문제다.** 작은 집단($n=5$)에 이상점이 둘 있으면

    - 집단 평균이 두 점 쪽으로 크게 끌려가고
    - 두 점의 잔차가 작아지며
    - **나머지 세 점이 "이상점"으로 표시**된다

    **처방 둘.**

    1. **집단별 중앙값과 MAD**로 이상점을 먼저 살핀다(최소제곱의 영향을 받지 않는다).
    2. **로버스트 추정**(절사평균, M-추정)으로 적합한 뒤 잔차를 본다.

    **더빈-왓슨의 편향.** 본문 예제에서 $d=2.11$이 나온 것도 이 음의 상관 때문일 수 있다. 자료를 독립으로 생성했는데도 $d>2$였다. **$k$개 집단, 집단당 $n$개면 $E[d]\approx2(1+\frac{1}{n-1})$** 수준의 상향 편향이 있다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
표준화가 **정규성 검정에는 거의 영향이 없다.** 그럼 왜 표준화 잔차를 쓰는가? 답을 수치로 보여라.

</div>

??? success "풀이"
    ```python
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    from scipy import stats
    from statsmodels.formula.api import ols

    rng = np.random.default_rng(9004)
    B = 8_000
    print("(가) 정규성 검정에는 차이가 거의 없다 (모두 정규, 명목 0.05)")
    print(f"{'집단 크기':>18s} {'원 잔차':>9s} {'표준화 잔차':>11s}")
    for ns in [[20, 20, 20], [5, 20, 60], [3, 10, 100]]:
        a = b = 0
        for _ in range(B):
            rows = [pd.DataFrame({"y": rng.normal(0, 1, n), "g": f"G{i}"})
                    for i, n in enumerate(ns)]
            df = pd.concat(rows, ignore_index=True)
            f = ols("y ~ C(g)", data=df).fit()
            a += stats.shapiro(f.resid).pvalue < 0.05
            b += stats.shapiro(
                f.get_influence().resid_studentized_internal).pvalue < 0.05
        print(f"{str(ns):>18s} {a / B:9.4f} {b / B:11.4f}")

    rng = np.random.default_rng(9005)
    B = 20_000
    NS = [4, 12, 40]
    cnt_raw = np.zeros(3)
    cnt_std = np.zeros(3)
    for _ in range(B):
        rows = [pd.DataFrame({"y": rng.normal(0, 1, n), "g": f"G{i}"})
                for i, n in enumerate(NS)]
        df = pd.concat(rows, ignore_index=True)
        f = ols("y ~ C(g)", data=df).fit()
        gi = np.repeat([0, 1, 2], NS)
        cnt_raw[gi[np.argmax(np.abs(f.resid.values))]] += 1
        cnt_std[gi[np.argmax(np.abs(
            f.get_influence().resid_studentized_internal))]] += 1

    print("\n(나) 가장 큰 |잔차| 가 어느 집단에서 나오는가")
    print(f"{'집단':>8s} {'n':>4s} {'기대 비율':>9s} {'원 잔차 최대':>12s} {'표준화 최대':>11s}")
    for i, n in enumerate(NS):
        print(f"{'G' + str(i):>8s} {n:4d} {n / sum(NS):9.4f} "
              f"{cnt_raw[i] / B:12.4f} {cnt_std[i] / B:11.4f}")
    ```

    ```text
    (가) 정규성 검정에는 차이가 거의 없다 (모두 정규, 명목 0.05)
                 집단 크기      원 잔차      표준화 잔차
          [20, 20, 20]    0.0512      0.0512
           [5, 20, 60]    0.0534      0.0516
          [3, 10, 100]    0.0511      0.0500

    (나) 가장 큰 |잔차| 가 어느 집단에서 나오는가
          집단    n     기대 비율      원 잔차 최대      표준화 최대
          G0    4    0.0714       0.0329      0.0664
          G1   12    0.2143       0.1900      0.2114
          G2   40    0.7143       0.7772      0.7221
    ```

    **(가) 정규성 검정의 기각률은 사실상 같다**(0.050~0.053 대 0.050~0.052). 집단 크기가 $(3,10,100)$으로 극단적이어도 차이가 없다.

    **왜 그런가.** 샤피로-윌크는 **표준화 잔차의 순위와 모양**에 의존하는데, 집단별로 상수를 곱하는 것이 **분포의 모양을 크게 바꾸지 않기** 때문이다.

    **(나)가 진짜 이유다.**

    | 집단 | $n$ | 기대 비율 | 원 잔차 | **표준화** |
    |---|---|---|---|---|
    | $G_0$ | 4 | 0.071 | **0.033** | **0.066** |
    | $G_1$ | 12 | 0.214 | 0.190 | 0.211 |
    | $G_2$ | 40 | 0.714 | **0.777** | 0.722 |

    **원 잔차를 쓰면 작은 집단이 절반만큼만 표시된다**(0.033 대 기대 0.071). 표준화하면 0.066으로 기대에 맞는다.

    **작은 집단의 이상점을 놓친다는 뜻**이다. $n=4$ 집단에서 잔차의 SD가 $\sigma\sqrt{1-1/4}=0.866\sigma$로 작으므로, **같은 크기의 이탈이 덜 눈에 띈다.**

    **그런데 작은 집단이야말로 이상점 하나의 영향이 크다.** $n=4$에서 한 점이 평균을 25% 끌어당긴다. **놓치면 안 되는 곳에서 놓치는 셈**이다.

    **정리 — 표준화의 용도.**

    | 용도 | 필요한가 |
    |---|---|
    | 정규성 검정 | **거의 무관**(원 잔차로도 충분) |
    | Q-Q 그림 | 무관(순위만 쓰므로) |
    | **이상점 탐지** | **필수** |
    | **관측 간 크기 비교** | **필수** |
    | 등분산 진단 | 오히려 **원 잔차나 $s_i$**가 낫다 |

    **마지막 줄이 역설적이다.** 표준화는 $\sqrt{1-h_{ii}}$로 나누어 **분산을 인위적으로 같게 만든다.** 이분산을 보려는데 그것을 지워 버리면 안 된다. **등분산 진단에는 집단별 $s_i$를 본다.**

    **실무 권고.** `statsmodels`에서

    ```text
    fit.resid                                   원 잔차
    fit.get_influence().resid_studentized_internal    표준화
    fit.get_influence().resid_studentized_external    삭제(이상점 탐지에 최선)
    ```

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
잔차 분석의 **전체 절차**를 정리하라.

</div>

??? success "풀이"
    **잔차의 성질 넷.** 일원배치에서

    | 성질 | 식 |
    |---|---|
    | 집단별 합이 0 | $\sum_j e_{ij}=0$ |
    | 분산이 일정하지 않음 | $\operatorname{Var}(e_{ij})=\sigma^2(1-1/n_i)$ |
    | 서로 독립이 아님 | $\operatorname{Corr}(e_{ij},e_{ij'})=-\dfrac{1}{n_i-1}$ |
    | 적합값이 $k$개뿐 | $\hat y_{ij}=\bar y_{i\cdot}$ |

    **네 성질이 모두 진단에 영향을 준다.**

    **그림별 용도.**

    | 그림 | 보는 것 | 주의 |
    |---|---|---|
    | **집단별 상자그림** | 이분산, 이상점, 치우침 | $n$을 함께 표시 |
    | 잔차 대 적합값 | 이분산의 **단조 경향** | 세로줄 $k$개뿐 |
    | **잔차의 Q-Q** | 정규성의 모양 | 표준화 잔차 사용 |
    | 스프레드-레벨 | **변환 지수** | 집단 4~5개 이상 필요 |
    | 잔차 대 수집 순서 | 독립성 | 순서를 기록해 두어야 |
    | 표준화 잔차 대 번호 | **이상점** | 본페로니 임계값 |

    **핵심 수치 넷.**

    | 사실 | 값 |
    |---|---|
    | $n=5$ 집단 잔차의 SD | $0.896\sigma$ |
    | $k=3$, $n=5$에서 SD 비 $>2$일 확률 | **0.409** |
    | 같은 집단 잔차의 상관 ($n=5$) | $-0.245$ |
    | 원 잔차가 작은 집단의 이상점을 놓치는 정도 | **절반** |

    **판독 절차.**

    ```text
    1. 집단별 n, 평균, s 를 표로 먼저 본다
         ↓
    2. 집단별 상자그림 — 산포·이상점·치우침
         ↓  (SD 비를 n 과 함께 해석 — 연습문제 6)
    3. 잔차의 Q-Q — 벗어남의 모양
         ↓
    4. 표준화 잔차의 최댓값 — 본페로니 임계값과 비교
         ↓
    5. 수집 순서가 있으면 순서에 대한 잔차
         ↓
    6. 이상한 것이 있으면 그 관측을 조사하고,
       빼고 다시 적합해 결론이 바뀌는지 확인
    ```

    **흔한 오독 다섯.**

    | 오독 | 바로잡기 |
    |---|---|
    | 작은 집단이 덜 퍼졌다 = 분산이 작다 | **잔차 분산이 원래 작다** |
    | SD 비 2배 = 이분산 | $n=5$면 **41%가 우연** |
    | 잔차가 두 봉우리 = 비정규 | **숨은 하위집단**일 수 있다 |
    | 원자료가 두 봉우리 = 문제 | **집단 평균 차이**면 정상 |
    | 잔차 그림 하나로 모든 진단 | 그림마다 **용도가 다르다** |

    **한 문장.** 잔차는 **오차의 추정값이지 오차 자체가 아니며**, 그 차이(분산이 일정하지 않고 서로 상관되어 있음)를 알아야 그림을 바르게 읽을 수 있다.

---

## 정리하며

잔차 분석은 분산분석을 위한 종합적인 시각 진단 틀을 제공한다. 잔차 그림을 살펴 정규성, 등분산성, 독립성, 선형성의 위반을 탐지하고, 분산분석 결과를 해석하기 전에 적절한 시정 조치를 취할 수 있다.
