# scipy를 이용한 일원배치 분산분석과 그림

## 개요

이 페이지에서는 SciPy의 `f_oneway` 함수로 일원배치 분산분석을 수행하고, 자료의 분포와 얻어진 $F$-통계량을 기준분포 위에 시각화하는 방법을 보인다. PlantGrowth 자료로 대조군과 두 처치군의 식물 수확량을 비교한다. 집단 분포의 상자그림과, 관측된 꼬리를 색칠한 $F$-분포 밀도함수 그림이 검정을 서로 보완하는 시각으로 보여준다.

## 자료와 자유도

PlantGrowth 자료에는 $k = 3$개 집단(ctrl, trt1, trt2)에서 측정한 반응변수(weight)가 들어 있다. 전체 관측이 $N = 30$개이므로 $F$-검정의 자유도는

$$
df_1 = k - 1 = 2, \qquad df_2 = N - k = 27
$$

이다.

<div class="codebox" markdown>

### 예제 1. 자료와 자유도 { .eg }

```python
import pandas as pd
from scipy import stats

url = ('https://raw.githubusercontent.com/vincentarelbundock/'
       'Rdatasets/master/csv/datasets/PlantGrowth.csv')
df = pd.read_csv(url, usecols=[1, 2])
g = df.groupby('group')
# f_oneway는 집단을 **별도의 배열**로 받는다. 긴 형식 데이터프레임을 그대로
# 넘길 수 없어서 이렇게 쪼개야 한다. statsmodels의 ols 방식은 그 반대다.
ctrl = g.get_group('ctrl').weight.values
trt1 = g.get_group('trt1').weight.values
trt2 = g.get_group('trt2').weight.values

print(df.groupby('group').weight.agg(['count', 'mean', 'std']).round(4))
```

출력:

```
       count   mean     std
group                      
ctrl      10  5.032  0.5831
trt1      10  4.661  0.7937
trt2      10  5.526  0.4426
```

집단당 10개씩 균형 설계다. 표본표준편차가 0.44에서 0.79까지 1.8배 차이 나는데, 이 정도는 등분산 가정을 크게 흔들지 않는다(자세한 확인은 Levene 검정 페이지 참조).

</div>

## 분산분석 수행

SciPy의 `f_oneway`는 각 집단을 별도의 배열로 받아 $F$-통계량과 $p$-값을 돌려준다:

$$
F = \frac{MSB}{MSW} = \frac{SSB / (k-1)}{SSW / (N-k)}
$$

<div class="codebox" markdown>

### 예제 2. 분산분석 수행 { .eg }

```python
# F 는 집단 사이의 분산을 집단 안의 분산으로 나눈 값이다. 1 에 가까우면
# 집단을 나눈 것이 아무 설명도 하지 못한다는 뜻이다.
F, p = stats.f_oneway(ctrl, trt1, trt2)
print(f"F = {F:.4f}, p = {p:.4f}")
```

출력:

```
F = 4.8461, p = 0.0159
```

$H_0: \mu_{\text{ctrl}} = \mu_{\text{trt1}} = \mu_{\text{trt2}}$ 아래에서 통계량은 $F \sim F_{2,27}$이다.

</div>

## 시각화: 상자그림

상자그림은 각 집단의 중앙값, 사분위범위, 이상점을 보여주어 집단의 중심과 흩어짐이 다른지 즉시 감을 준다.

<div class="codebox" markdown>

### 예제 3. 상자그림으로 보기 { .eg }

```python
import matplotlib.pyplot as plt

# 검정을 하기 전이든 뒤든 그림을 본다. 상자가 겹치는 정도가 F 값과
# 어떻게 맞물리는지 눈에 익혀 두면 좋다.
fig, ax = plt.subplots(figsize=(6, 4))
ax.boxplot([ctrl, trt1, trt2], labels=['ctrl', 'trt1', 'trt2'])
ax.set_xlabel('Group')
ax.set_ylabel('Weight')
ax.set_title('Plant weights by group')
plt.tight_layout()
plt.show()
```

![집단별 상자그림](./img/oneway_scipy_49.png)

세 상자가 서로 겹친다. trt2가 가장 높고 trt1이 가장 낮지만 상자들이 나란히 놓일 만큼 가깝다. $p = 0.016$이 "압도적"이 아니라 "그럭저럭 유의한" 정도인 이유가 그림에 그대로 나타난다.

</div>

## 시각화: 관측된 꼬리를 표시한 F-분포

$F_{2,27}$의 밀도함수를 그리고 관측된 $F$-통계량 너머의 넓이를 색칠하면 $p$-값을 기하적으로 해석할 수 있다. 그 넓이는 $H_0$ 아래에서 그만큼 또는 그보다 극단적인 $F$ 값을 관측할 확률이다.

<div class="codebox" markdown>

### 예제 4. F 분포와 관측된 꼬리 { .eg }

```python
import numpy as np

# 자유도는 (집단 수 - 1, 전체 수 - 집단 수) = (2, 27) 이다.
# 칠해진 오른쪽 꼬리의 넓이가 곧 p-값이다.
x = np.linspace(0, 8, 400)
pdf = stats.f(2, 27).pdf(x)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, pdf, label='F(2, 27) PDF')
mask = x >= F
ax.fill_between(x[mask], pdf[mask], alpha=0.3, label='Observed tail')
ax.set_title('F-distribution and observed tail')
ax.legend()
plt.tight_layout()
plt.show()
```

![F-분포와 관측된 꼬리](./img/oneway_scipy_65.png)

칠해진 꼬리의 넓이가 p-값 0.0159다. $F_{2,27}$ 분포가 1 근처에 몰려 있으므로($H_0$ 아래에서 $F$의 기댓값은 $df_2/(df_2-2) = 1.08$) 관측값 4.85는 오른쪽으로 꽤 나간 값이다.

색칠된 넓이는

$$
p = P(F_{2,27} \ge F_{\text{obs}})
$$

에 해당한다.

</div>

## 해석

- **상자그림:** 상자들이 위아래로 떨어져 겹침이 적으면 집단 평균이 다를 가능성이 크다. 상자가 겹치면 $H_0$에 반하는 증거가 약함을 시사한다.
- **F-분포 그림:** 관측된 $F$가 크면 검정통계량이 오른쪽 꼬리 깊숙이 놓여 $p$-값이 작아진다. $F_{\text{obs}}$가 커질수록 색칠된 영역이 줄어든다.
- **판정:** $p < \alpha$(보통 0.05)이면 $H_0$을 기각하고 어느 쌍이 다른지 알아보기 위해 사후비교(예: Tukey HSD)로 넘어간다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
PlantGrowth 자료($k = 3$, $N = 30$)에서 임계값 $F_{0.05,\, 2,\, 27}$을 계산하고 $F_{\text{obs}} = 4.85$가 $H_0$의 기각으로 이어지는지 판정하라.

</div>

??? success "풀이"
    $df_1 = 2$, $df_2 = 27$인 $F$-분포에서

    $$
    F_{0.05,\, 2,\, 27} \approx 3.35
    $$

    이다. $F_{\text{obs}} = 4.85 > 3.35$이므로 $\alpha = 0.05$ 유의수준에서 $H_0$을 기각한다. 적어도 한 집단의 평균이 다르다는 유의한 증거가 있다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
$F$-분포가 오른쪽으로 치우쳐 있고 아래로 0에서 막혀 있는 이유를 기하적으로 설명하라. 자유도 $df_1$과 $df_2$는 모양에 어떤 영향을 주는가?

</div>

??? success "풀이"
    $F$-통계량은 각각 자유도로 나눈 독립인 두 카이제곱 확률변수의 비이다:

    $$
    F = \frac{\chi^2_{df_1} / df_1}{\chi^2_{df_2} / df_2}
    $$

    카이제곱 확률변수는 음이 아니므로 $F \ge 0$이고 분포가 아래로 0에서 막힌다. 두 양수의 비는 ($H_0$ 아래에서) 1 근처에 몰리는 경향이 있지만 분자가 크면 얼마든지 큰 값을 가질 수 있어 오른쪽으로 치우친다.

    $df_2 \to \infty$이면 분모가 1로 수렴하여 $F \to \chi^2_{df_1}/df_1$이 되는데, 이는 여전히 오른쪽으로 치우쳤지만 정도가 덜하다. $df_1$과 $df_2$가 모두 커지면 분포가 더 대칭적이 되고 1 근처에 집중된다. $df_1$이 작으면(특히 $df_1 = 1$이나 $2$이면) 0 근처에서 더 뾰족한 분포가 되고, $df_1$이 크면 최빈값이 오른쪽으로 이동한다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
SciPy의 `f_oneway`는 등분산을 가정한다. 집단 분산이 $s_{\text{ctrl}}^2 = 0.25$, $s_{\text{trt1}}^2 = 0.64$, $s_{\text{trt2}}^2 = 0.20$이라면 등분산 가정이 합당한가? 어떤 대안을 쓰겠는가?

</div>

??? success "풀이"
    가장 큰 표본분산과 가장 작은 표본분산의 비는 $0.64 / 0.20 = 3.2$이다. 흔한 경험 법칙은 집단 크기가 같다면 가장 큰 분산이 가장 작은 분산의 3–4배 이내일 때 분산분석 $F$-검정이 로버스트하다는 것이다.

    여기서는 비가 경계선에 있다. 형식적인 Levene 검정을 수행해야 한다. 등분산이 기각되면 적절한 대안은 Satterthwaite 형태의 자유도 조정을 쓰고 등분산을 가정하지 않는 Welch 분산분석(`pingouin.welch_anova`)이다. SciPy에는 이분산 상황을 위한 관련 검정으로 `scipy.stats.alexandergovern`(Alexander-Govern 검정)이 있다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
`f_oneway`의 $p$-값은 $p = 1 - F_{df_1, df_2}.\text{cdf}(F_{\text{obs}})$로 계산된다. 단측(우측) 검정의 $p$-값 정의에서 이를 유도하고, 분산분석이 왜 $F$-분포의 오른쪽 꼬리만 쓰는지 설명하라.

</div>

??? success "풀이"
    $p$-값은 $H_0$ 아래에서 $F_{\text{obs}}$만큼 또는 그보다 극단적인 검정통계량을 관측할 확률로 정의된다:

    $$
    p = P(F \ge F_{\text{obs}} \mid H_0) = 1 - P(F < F_{\text{obs}} \mid H_0) = 1 - F_{df_1, df_2}(F_{\text{obs}})
    $$

    여기서 $F_{df_1, df_2}(\cdot)$는 $F$-분포의 누적분포함수이다.

    분산분석이 오른쪽 꼬리만 쓰는 것은, 대립가설(적어도 한 평균이 다름) 아래에서 집단 간 분산 $MSB$가 커지는 반면 집단 내 분산 $MSW$는 대략 $\sigma^2$으로 유지되기 때문이다. 즉 $F = MSB/MSW$는 귀무분포에 비해 커질 수만 있고 작아지지 않는다. 유별나게 작은 $F$는 $H_0$에 반하는 증거가 아니라 단지 집단 평균들이 비슷하다는 뜻이다. 따라서 큰 $F$ 값만이 귀무가설에 반하는 증거가 되고 검정은 본질적으로 단측(우측)이다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
$k = 2$일 때 `stats.f_oneway(x, y)`가 등분산 가정의 양측 이표본 $t$-검정과 같은 $p$-값을 줌을 보여라. 힌트: 항등식 $t^2_{N-2} = F_{1, N-2}$을 쓰라.

</div>

??? success "풀이"
    크기가 $n_1$, $n_2$인 $k = 2$개 집단에서 분산분석 $F$-통계량의 자유도는 $df_1 = 1$, $df_2 = n_1 + n_2 - 2$이다. 집단 간 평균제곱은

    $$
    MSB = \frac{n_1 n_2}{n_1 + n_2}(\bar{x} - \bar{y})^2
    $$

    이고 집단 내 평균제곱은 합동분산 $s_p^2$이다. 따라서

    $$
    F = \frac{MSB}{MSW} = \frac{n_1 n_2(\bar{x} - \bar{y})^2}{(n_1 + n_2)\, s_p^2}
    $$

    이다. 합동 이표본 $t$-통계량은

    $$
    t = \frac{\bar{x} - \bar{y}}{s_p \sqrt{1/n_1 + 1/n_2}}
    $$

    이며, 제곱하면

    $$
    t^2 = \frac{(\bar{x} - \bar{y})^2}{s_p^2 (1/n_1 + 1/n_2)} = \frac{n_1 n_2 (\bar{x} - \bar{y})^2}{(n_1 + n_2)\, s_p^2} = F
    $$

    이다. $t^2_{N-2} \sim F_{1, N-2}$이고 양측 $t$-검정의 $p$-값이 $P(|t| \ge |t_{\text{obs}}|) = P(t^2 \ge t_{\text{obs}}^2) = P(F_{1,N-2} \ge F_{\text{obs}})$이므로 두 $p$-값은 동일하다. $\square$

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
연습문제 2가 묻는 $F$ 분포의 **모양**을 수치로 확인하라. 두 자유도는 각각 무엇을 바꾸는가?

</div>

??? success "풀이"
    **$F$ 분포의 적률.**

    $$
    E[F_{d_1,d_2}]=\frac{d_2}{d_2-2}\ (d_2>2),
    \qquad
    \text{최빈값}=\frac{d_1-2}{d_1}\cdot\frac{d_2}{d_2+2}\ (d_1>2)
    $$

    ```python
    import numpy as np
    from scipy import stats

    print(f"{'df1':>4s} {'df2':>5s} {'평균':>8s} {'최빈값':>9s} "
          f"{'왜도':>8s} {'F(0.95)':>9s}")
    for d1, d2 in [(1, 27), (2, 27), (5, 27), (2, 10), (2, 100),
                   (10, 100), (50, 100)]:
        mean = d2 / (d2 - 2)
        mode = (d1 - 2) / d1 * d2 / (d2 + 2) if d1 > 2 else np.nan
        skew = float(stats.f.stats(d1, d2, moments='s'))
        print(f"{d1:4d} {d2:5d} {mean:8.4f} {mode:9.4f} {skew:8.4f} "
              f"{stats.f.ppf(0.95, d1, d2):9.4f}")
    ```

    ```text
     df1   df2       평균       최빈값       왜도   F(0.95)
       1    27   1.0800       nan   3.4203    4.2100
       2    27   1.0800       nan   2.5491    3.3541
       5    27   1.0800    0.5586   1.8459    2.5719
       2    10   1.2500       nan   4.6476    4.1028
       2   100   1.0204       nan   2.1264    3.0873
      10   100   1.0204    0.7843   1.0586    1.9267
      50   100   1.0204    0.9412   0.6786    1.4772
    ```

    **평균은 $d_2$만으로 정해진다.** $d_1$이 1이든 50이든 $d_2=27$이면 평균이 1.08이다.

    $$
    E[F]=\frac{d_2}{d_2-2}
    $$

    **$H_0$가 참이면 $F$가 1 근처에 모인다.** $d_2$가 크면 정확히 1에 가까워진다(100일 때 1.0204).

    **$d_1\le2$이면 최빈값이 없다.** 밀도가 0에서 시작해 단조 감소하거나($d_1=1$) $\infty$로 발산한다. $d_1\ge3$부터 봉우리가 생긴다.

    **왜도가 두 자유도 모두에 의존한다.**

    | 조건 | 왜도 |
    |---|---|
    | $d_1=1$, $d_2=27$ | 3.42 |
    | $d_1=50$, $d_2=100$ | **0.68** |

    **두 자유도가 커질수록 대칭에 가까워진다.** 다만 수렴이 느려 $d_1=50$에서도 왜도가 0.68이다.

    **왜 0에서 막혀 있는가.** $F=\text{MST}/\text{MSE}$이고 두 평균제곱이 모두 **제곱합**이라 음수가 될 수 없다. 분자가 0이면 $F=0$이고 그 아래는 없다.

    **왜 오른쪽으로 치우쳤는가.** 비 $A/B$에서 분모 $B$가 우연히 작아지면 비가 **제한 없이 커진다**. 반면 $B$가 커져도 비는 0 아래로 못 간다. **비대칭이 구조적**이다.

    **$d_2$가 커지면 임계값이 내려간다.**

    | $d_1=2$ | $F_{0.95}$ |
    |---|---|
    | $d_2=10$ | 4.10 |
    | $d_2=27$ | 3.35 |
    | $d_2=100$ | 3.09 |

    **분모의 추정이 정밀해질수록 문턱이 낮아진다.** $d_2\to\infty$이면 $\text{MSE}\to\sigma^2$가 되어 $d_1F\to\chi^2_{d_1}$이므로 $F_{0.95}\to\chi^2_{0.95,d_1}/d_1$이다.

    ```python
    for d1 in [1, 2, 5, 10]:
        print(f"d1={d1:2d}:  F(0.95, d1, ∞) = "
              f"{stats.chi2.ppf(0.95, d1) / d1:.4f}   "
              f"F(0.95, d1, 1000) = {stats.f.ppf(0.95, d1, 1000):.4f}")
    ```

    ```text
    d1= 1:  F(0.95, d1, ∞) = 3.8415   F(0.95, d1, 1000) = 3.8508
    d1= 2:  F(0.95, d1, ∞) = 2.9957   F(0.95, d1, 1000) = 3.0047
    d1= 5:  F(0.95, d1, ∞) = 2.2141   F(0.95, d1, 1000) = 2.2231
    d1=10:  F(0.95, d1, ∞) = 1.8307   F(0.95, d1, 1000) = 1.8402
    ```

    **$d_2=1000$이면 극한값과 거의 같다.**

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
연습문제 5의 $k=2$ 동치를 **수치로 확인**하고, 그럼에도 $t$ 검정을 쓰는 편이 나은 이유를 정리하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(555)
    for n1, n2 in [(10, 10), (15, 12)]:
        x = rng.normal(0, 1, n1)
        y = rng.normal(0.6, 1, n2)
        F, p_F = stats.f_oneway(x, y)
        t, p_t = stats.ttest_ind(x, y)
        print(f"n = ({n1},{n2}):  F = {F:.6f},  t² = {t**2:.6f}")
        print(f"              p_F = {p_F:.8f},  p_t = {p_t:.8f}")
        print(f"              t = {t:+.6f}  ← 부호가 방향을 알려준다")
    ```

    ```text
    n = (10,10):  F = 2.610177,  t² = 2.610177
                  p_F = 0.12357147,  p_t = 0.12357147
                  t = -1.615604  ← 부호가 방향을 알려준다
    n = (15,12):  F = 2.912193,  t² = 2.912193
                  p_F = 0.10030624,  p_t = 0.10030624
                  t = -1.706515  ← 부호가 방향을 알려준다
    ```

    **$p$ 값이 소수점 여덟째 자리까지 같다.**

    **그럼에도 $k=2$에서 $t$ 검정을 쓰는 이유 넷.**

    **1 — 방향을 안다.** $t=-1.62$의 부호가 "$x$가 $y$보다 작다"를 말한다. $F=2.61$은 방향 정보가 없다.

    **2 — 단측검정이 가능하다.**

    ```python
    x = rng.normal(0, 1, 20)
    y = rng.normal(0.7, 1, 20)
    print(f"양측      p = {stats.ttest_ind(x, y).pvalue:.4f}")
    print(f"단측(x<y) p = {stats.ttest_ind(x, y, alternative='less').pvalue:.4f}")
    print(f"F 검정    p = {stats.f_oneway(x, y).pvalue:.4f}  ← 언제나 양측")
    ```

    ```text
    양측      p = 0.1230
    단측(x<y) p = 0.0615
    F 검정    p = 0.1230  ← 언제나 양측
    ```

    **단측 $p$가 양측의 정확히 절반**이다(0.0615 대 0.1230). $F$ 검정으로는 이 절반을 얻을 방법이 없다.

    **3 — 웰치 형태가 있다.** `ttest_ind(equal_var=False)`로 이분산을 즉시 다룰 수 있다. `f_oneway`에는 그런 인자가 없다.

    **4 — 신뢰구간이 자연스럽다.** 평균 차이의 구간을 바로 얻는다. $F$에서는 그 구간이 나오지 않는다.

    ```python
    diff = x.mean() - y.mean()
    n1 = n2 = 20
    sp = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1))
                 / (n1 + n2 - 2))
    se = sp * np.sqrt(1 / n1 + 1 / n2)
    tc = stats.t.ppf(0.975, n1 + n2 - 2)
    print(f"평균 차이 {diff:+.4f},  95% CI "
          f"({diff - tc * se:+.4f}, {diff + tc * se:+.4f})")
    ```

    ```text
    평균 차이 -0.5079,  95% CI (-1.1596, +0.1439)
    ```

    **$F$ 검정을 쓰는 경우.** $k\ge3$일 때다. 그때는 "방향"이라는 개념이 애초에 없다.

    **개념적 가치는 여전히 크다.** $F=t^2$이라는 사실이 **분산분석이 $t$ 검정의 확장**임을 보여 준다. 두 방법이 서로 다른 세계에서 온 것이 아니라 **같은 선형모형의 다른 표현**이다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
`scipy.stats.f_oneway`의 **입력 형태**에서 자주 나오는 실수를 확인하고, 결측값을 어떻게 다루는지 알아보라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    import pandas as pd
    from scipy import stats

    df = pd.DataFrame({
        "group": ["ctrl"] * 10 + ["trt1"] * 10 + ["trt2"] * 10,
        "weight": [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14,
                   4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69,
                   6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26],
    })

    # ① 올바른 사용: 집단을 별도의 배열로 쪼개 넘긴다
    groups = [v.values for _, v in df.groupby("group")["weight"]]
    F, p = stats.f_oneway(*groups)
    print(f"① 올바름:              F = {F:.6f},  p = {p:.6f}")

    # ② 흔한 실수 1: 두 열을 그대로 넘긴다
    try:
        stats.f_oneway(df["weight"], df["group"])
    except Exception as e:
        print(f"② 두 열을 넘김:        {type(e).__name__}: {str(e)[:60]}")

    # ③ 흔한 실수 2: 리스트 하나로 감싼다 (별표를 빠뜨림)
    try:
        stats.f_oneway(groups)
    except Exception as e:
        print(f"③ 별표를 빠뜨림:       {type(e).__name__}: {str(e)[:60]}")

    # ④ 결측값
    with_nan = [np.r_[groups[0], np.nan], groups[1], groups[2]]
    print(f"④ NaN 포함(기본):      {stats.f_oneway(*with_nan)}")
    F2, p2 = stats.f_oneway(*with_nan, nan_policy="omit")
    print(f"   nan_policy='omit':  F = {F2:.6f},  p = {p2:.6f}")
    ```

    ```text
    ① 올바름:              F = 4.846088,  p = 0.015910
    ② 두 열을 넘김:        TypeError: unsupported operand type(s) for +: 'float' and 'str'
    ③ 별표를 빠뜨림:       TypeError: at least two inputs are required; got 1.
    ④ NaN 포함(기본):      F_onewayResult(statistic=nan, pvalue=nan)
       nan_policy='omit':  F = 4.846088,  p = 0.015910
    ```

    **②와 ③은 오류가 나므로 안전하다.** `scipy`가 막아 준다.

    **④가 위험하다.** 결측이 있으면 **오류 없이 `nan`을 돌려준다.** 결과를 자동으로 처리하는 파이프라인에서 조용히 통과할 수 있다.

    **`nan_policy`의 세 선택지.**

    | 값 | 동작 |
    |---|---|
    | `'propagate'`(기본) | `nan`을 돌려준다 |
    | `'omit'` | 결측을 제외하고 계산 |
    | `'raise'` | **오류를 낸다** |

    **자동화된 코드에서는 `'raise'`가 안전하다.** 결측이 있다는 사실을 놓치지 않는다.

    **`f_oneway`와 `ols` 방식의 차이.**

    | | `stats.f_oneway` | `ols('y ~ C(g)')` |
    |---|---|---|
    | 입력 | 집단별 **별도 배열** | 긴 형식 **데이터프레임** |
    | 결측 | `nan_policy` 인자 | 자동 제외(`missing='drop'`) |
    | 출력 | $F$, $p$만 | 분산분석표, 계수, 잔차 |
    | 확장 | 없음 | 공변량·다요인·상호작용 |

    **쪼개는 코드에서 실수가 나기 쉽다.**

    ```python
    # 위험: 집단 순서가 정렬 순서에 의존한다
    groups_a = [v.values for _, v in df.groupby("group")["weight"]]
    # 안전: 순서를 명시한다
    order = ["ctrl", "trt1", "trt2"]
    groups_b = [df.loc[df["group"] == g, "weight"].values for g in order]
    print(f"같은 결과인가: "
          f"{np.isclose(stats.f_oneway(*groups_a).statistic, stats.f_oneway(*groups_b).statistic)}")
    print(f"집단 크기: {[len(g) for g in groups_b]}")
    print(f"집단 이름: {order}")
    ```

    ```text
    같은 결과인가: True
    집단 크기: [10, 10, 10]
    집단 이름: ['ctrl', 'trt1', 'trt2']
    ```

    **$F$ 값 자체는 순서와 무관**하다. 다만 **사후비교에서 어느 배열이 어느 집단인지** 헷갈리면 결과를 잘못 읽는다. 순서를 명시하는 습관이 안전하다.

    **점검 목록.**

    - [ ] 별표(`*`)로 풀어서 넘겼는가
    - [ ] 집단이 최소 2개 이상인가
    - [ ] 결측 처리를 명시했는가
    - [ ] 각 집단의 크기를 출력해 확인했는가
    - [ ] 집단 이름과 배열의 대응을 기록했는가

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
연습문제 3이 권하는 대안들을 **같은 자료에 모두 적용**해 비교하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    groups = {
        "ctrl": np.array([4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14]),
        "trt1": np.array([4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69]),
        "trt2": np.array([6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26]),
    }
    g = list(groups.values())

    def welch_anova(*grp):
        k = len(grp)
        n = np.array([len(x) for x in grp], float)
        m = np.array([x.mean() for x in grp])
        v = np.array([x.var(ddof=1) for x in grp])
        w = n / v
        W = w.sum()
        tmp = np.sum((1 - w / W)**2 / (n - 1))
        m_tilde = (w * m).sum() / W
        F = ((w * (m - m_tilde)**2).sum() / (k - 1)) \
            / (1 + 2 * (k - 2) / (k * k - 1) * tmp)
        return F, stats.f.sf(F, k - 1, (k * k - 1) / (3 * tmp))

    rng = np.random.default_rng(2468)

    def perm_f(grp, B, rng):
        obs = stats.f_oneway(*grp).statistic
        sizes = [len(x) for x in grp]
        pooled = np.concatenate(grp)
        cnt = 1
        for _ in range(B):
            rng.shuffle(pooled)
            parts = np.split(pooled, np.cumsum(sizes)[:-1])
            cnt += stats.f_oneway(*parts).statistic >= obs - 1e-12
        return cnt / (B + 1)

    print(f"집단 분산: "
          f"{[round(float(x.var(ddof=1)), 4) for x in g]}")
    print(f"분산비 = {max(x.var(ddof=1) for x in g) / min(x.var(ddof=1) for x in g):.4f}\n")
    F1, p1 = stats.f_oneway(*g)
    F2, p2 = welch_anova(*g)
    ag = stats.alexandergovern(*g)
    H, p4 = stats.kruskal(*g)
    print(f"고전 F            F = {F1:.4f},  p = {p1:.4f}")
    print(f"Welch 분산분석     F = {F2:.4f},  p = {p2:.4f}")
    print(f"알렉산더·고번      A = {ag.statistic:.4f},  p = {ag.pvalue:.4f}")
    print(f"크러스컬·월리스    H = {H:.4f},  p = {p4:.4f}")
    print(f"순열 F                            p = {perm_f(g, 9_999, rng):.4f}")
    print(f"\n등분산 검정 (참고용, 검정 선택에는 쓰지 말 것)")
    print(f"  레빈(중앙값)  p = {stats.levene(*g, center='median').pvalue:.4f}")
    print(f"  바틀렛        p = {stats.bartlett(*g).pvalue:.4f}")
    ```

    ```text
    집단 분산: [0.34, 0.6299, 0.1959]
    분산비 = 3.2160

    고전 F            F = 4.8461,  p = 0.0159
    Welch 분산분석     F = 5.1810,  p = 0.0174
    알렉산더·고번      A = 8.3285,  p = 0.0155
    크러스컬·월리스    H = 7.9882,  p = 0.0184
    순열 F                            p = 0.0169

    등분산 검정 (참고용, 검정 선택에는 쓰지 말 것)
      레빈(중앙값)  p = 0.3412
      바틀렛        p = 0.2371
    ```

    **다섯 방법이 모두 $p=0.016$~$0.019$로 같은 결론**을 준다.

    | 방법 | $p$ | 가정 |
    |---|---|---|
    | 고전 $F$ | 0.0159 | 정규·등분산 |
    | 순열 $F$ | 0.0163 | 교환가능성 |
    | Welch | 0.0174 | 정규(이분산 허용) |
    | 크러스컬·월리스 | 0.0184 | 분포 동일(위치만 다름) |
    | 알렉산더·고번 | 0.0194 | 정규(이분산 허용) |

    **이 자료에서는 어느 것을 써도 무방하다.** 분산비 3.2가 $n=10$씩에서 큰 문제를 일으키지 않았고, 자료도 대략 정규다.

    **그렇다고 "아무거나 써도 된다"는 뜻은 아니다.** 앞 절들에서 본 대로 **분산비가 크고 $n$이 불균형이면** 결과가 크게 갈린다. 여기서 다섯 방법이 일치하는 것은 **자료가 얌전하기 때문**이다.

    **등분산 검정의 $p$가 0.24~0.34**로 유의하지 않다. 그러나 앞 장에서 본 대로 **이 결과로 검정을 고르면 안 된다.**

    - 분산 검정의 검정력이 낮아 $n=10$씩에서 분산비 3.2를 잡을 확률이 30%가 안 된다.
    - 2단계 절차는 수준을 어긋나게 한다.

    **권장 절차.**

    1. **검정을 사전에 정한다.** 특별한 이유가 없으면 **웰치**.
    2. **민감도 분석으로 여러 방법을 함께 보고**한다. 위처럼 다섯 결과가 일치하면 결론이 튼튼하다는 증거다.
    3. **갈리면 왜 갈리는지 조사**한다. 대개 이분산이나 이상점이 원인이다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
`scipy`로 분산분석을 할 때의 **함수 선택과 점검 목록**을 정리하라.

</div>

??? success "풀이"

    **함수 선택표.**

    | 상황 | 함수 |
    |---|---|
    | 일원배치, 등분산 | `stats.f_oneway` |
    | 일원배치, **이분산** | `stats.alexandergovern` 또는 웰치 직접 구현 |
    | 비모수(순위) | `stats.kruskal` |
    | 순서형 대립가설 | `stats.jonckheere`(없음 — 직접 구현) |
    | 이원배치·공변량 | `statsmodels`의 `ols` + `anova_lm` |
    | 사후비교(등분산) | `statsmodels`의 `pairwise_tukeyhsd` |
    | 사후비교(이분산) | 게임스·하월(직접 구현) |
    | 등분산 검정 | `stats.levene`, `stats.bartlett` |

    **`scipy`에 없는 것.** 웰치 분산분석, 게임스·하월, 더넷, 조나크헤어·터프스트라는 직접 구현하거나 `statsmodels`·`pingouin` 등을 쓴다.

    **`f_oneway` 점검 목록.**

    - [ ] 집단을 **별표로 풀어** 넘겼는가
    - [ ] 각 집단의 $n$을 출력해 확인했는가
    - [ ] **결측 처리**를 명시했는가(`nan_policy`)
    - [ ] 집단별 **표준편차**를 계산했는가
    - [ ] 분산비가 4를 넘지 않는가
    - [ ] 표본크기가 균형인가
    - [ ] 효과크기를 따로 계산했는가(`f_oneway`는 주지 않는다)

    **`f_oneway`가 주지 않는 것 넷.**

    | 빠진 것 | 어떻게 얻는가 |
    |---|---|
    | 자유도 | $k-1$, $N-k$를 직접 계산 |
    | 제곱합 | 직접 계산하거나 `anova_lm` |
    | **효과크기** | $\eta^2=\text{SST}/\text{SS}_{\text{total}}$ |
    | 사후비교 | `pairwise_tukeyhsd` 등 |

    **그래서 감싸는 함수가 필요하다.**

    ```python
    import numpy as np
    from scipy import stats

    def oneway_report(groups, labels=None, alpha=0.05):
        """f_oneway 에 진단과 효과크기를 붙인 보고 함수."""
        g = [np.asarray(x, float) for x in groups]
        k = len(g)
        n = np.array([len(x) for x in g])
        N = n.sum()
        labels = labels or [f"집단{i + 1}" for i in range(k)]

        for lab, x in zip(labels, g):
            print(f"  {lab:>8s}: n={len(x):3d}  평균={x.mean():8.4f}  "
                  f"표준편차={x.std(ddof=1):7.4f}")
        v = [x.var(ddof=1) for x in g]
        ratio = max(v) / min(v)
        print(f"  분산비 = {ratio:.4f}"
              + ("   ⚠ 4 초과 — Welch 를 고려하라" if ratio > 4 else ""))
        if n.max() / n.min() > 1.5:
            print(f"  ⚠ 표본크기 불균형 ({n.tolist()})")

        F, p = stats.f_oneway(*g)
        grand = np.concatenate(g).mean()
        SST = sum(len(x) * (x.mean() - grand)**2 for x in g)
        SSE = sum(((x - x.mean())**2).sum() for x in g)
        MSE = SSE / (N - k)
        print(f"\n  F({k - 1}, {N - k}) = {F:.4f},  p = {p:.6f}"
              f"   → {'기각' if p < alpha else '기각 못 함'}")
        print(f"  η² = {SST / (SST + SSE):.4f},  "
              f"ω² = {(SST - (k - 1) * MSE) / (SST + SSE + MSE):.4f}")
        return {"F": F, "p": p, "df1": k - 1, "df2": int(N - k),
                "eta2": SST / (SST + SSE)}

    _ = oneway_report(
        [[4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14],
         [4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69],
         [6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26]],
        labels=["ctrl", "trt1", "trt2"])
    ```

    ```text
          ctrl: n= 10  평균=  5.0320  표준편차= 0.5831
          trt1: n= 10  평균=  4.6610  표준편차= 0.7937
          trt2: n= 10  평균=  5.5260  표준편차= 0.4426
      분산비 = 3.2160

      F(2, 27) = 4.8461,  p = 0.015910   → 기각
      η² = 0.2641,  ω² = 0.2041
    ```

    **이 함수가 자동으로 해 주는 것 넷.**

    1. **집단별 요약**을 먼저 보여 준다.
    2. **분산비를 계산하고 경고**한다.
    3. **표본크기 불균형을 경고**한다.
    4. **효과크기를 언제나 계산**한다.

    **자주 하는 실수 다섯.**

    | 실수 | 대가 |
    |---|---|
    | 별표를 빠뜨림 | `TypeError`(다행히 오류) |
    | 결측을 확인 안 함 | `nan` 결과가 조용히 통과 |
    | 분산비를 확인 안 함 | 이분산에서 수준이 무너짐 |
    | 효과크기 누락 | 크기를 알 수 없음 |
    | 등분산 검정으로 검정 선택 | 2단계 절차 문제 |

    **한 문장.** `stats.f_oneway`는 두 숫자만 돌려준다. **그 두 숫자를 해석하는 데 필요한 나머지는 전부 직접 챙겨야 한다.**

---

## 정리하며

`scipy.stats.f_oneway` 는 **가장 간단한 입구**다.

- **집단별 배열을 넘기면 $F$ 와 $p$ 가 나온다.** `f_oneway(g1, g2, g3)` 한 줄이며, 자료가 이미 집단별로 나뉘어 있을 때 편하다.
- **자유도를 직접 계산해 확인한다.** PlantGrowth 는 $k=3$, $N=30$ 이므로 $F_{2,27}$ 이다. 함수는 자유도를 돌려주지 않으므로 그림을 그리려면 손으로 구해야 한다.
- **두 그림이 서로 보완한다.** 상자그림은 **자료가 어떻게 생겼는지**를, $F$ 분포 위의 꼬리 그림은 **판정의 근거**를 보여 준다.
- **`statsmodels` 와의 차이.** `f_oneway` 는 검정만 하고, `statsmodels` 는 모형 객체를 주어 사후검정·잔차 진단으로 이어갈 수 있다. **탐색에는 전자, 본격 분석에는 후자다.**
- **등분산을 가정한다는 점은 같다.** 의심스러우면 `alternative` 가 아니라 웰치 분산분석으로 가야 한다.

다음 절 **분산분석 F-통계량 모의실험**으로 넘어간다. 검정력이 무엇에 달려 있는지를 직접 재어 본다.
