# 분산분석 진단

## 개요

분산분석의 결과를 믿기 전에 몇 가지 핵심 가정을 확인해야 한다: 잔차의 정규성, 등분산성(집단 사이의 분산이 같음), 관측의 독립성, 그리고 영향점의 부재. 이 페이지는 완전한 진단 흐름을 따라가며 각 확인을 형식적 검정과 진단 그림으로 보이고, 가정이 어긋났을 때의 처방을 논한다.

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

## 진단 작업 흐름

전형적인 분산분석 진단 파이프라인은 적합된 모형 $y_{ij} = \mu + \alpha_i + \varepsilon_{ij}$의 잔차에 적용되는 네 단계로 이루어진다.

| 단계 | 질문 | 주요 도구 | 형식적 검정 |
|---|---|---|---|
| 1 | 잔차가 정규인가? | Q-Q 그림, 히스토그램 | Shapiro-Wilk |
| 2 | 집단 분산이 같은가? | 집단별 흩어짐 비교 | Levene, Bartlett |
| 3 | 잔차가 독립인가? | 잔차 대 적합값 | Durbin-Watson |
| 4 | 지나치게 영향력 있는 점이 있는가? | Cook의 거리 그림 | Cook의 $D$ 문턱 |

## 1단계: 정규성 확인

Shapiro-Wilk 검정은

$$
H_0: \text{the residuals come from a normal distribution}
$$

을 그렇지 않다는 대립가설에 대해 평가한다. Q-Q 그림은 표본 분위수가 이론적 정규 분위수와 맞는지 보여주어 검정을 보완한다.

<div class="codebox" markdown>

### 예제 2. 1단계 — 정규성 { .eg }

```python
from scipy.stats import shapiro
import statsmodels.api as sm

# 정규성은 자료가 아니라 잔차에 요구되는 가정이다. 집단마다 평균이 다르므로
# 자료 전체를 한 번에 검정하면 안 된다.
resid = model.resid
stat, p_value = shapiro(resid)
print(f"Shapiro-Wilk: W = {stat:.4f}, p = {p_value:.4f}")
sm.qqplot(resid, line='s')
```

출력:

```
Shapiro-Wilk: W = 0.8116, p = 0.0000
```

![잔차의 Q-Q 그림](./img/anova_diagnostics_74.png)

$p < 0.0001$로 정규성을 강하게 기각한다. Q-Q 그림의 오른쪽 끝에 크게 벗어난 점 하나가 보이는데, 설정에서 심어 둔 이상점이다.

**검정이 잡아낸 것은 "잔차가 정규가 아니다"이지만 실제 원인은 관측값 하나다.** 형식적 검정만 보면 분포 전체를 의심하게 되고, 그림을 함께 보아야 원인이 한 점이라는 것을 알 수 있다.

Shapiro-Wilk의 $p$-값이 작거나(예: $p < 0.05$) Q-Q 그림에 체계적인 곡률이 보이면 정규성이 의심스럽다. 처방으로는 자료 변환(로그, 제곱근)이나 Kruskal-Wallis 같은 비모수 검정으로의 전환이 있다.

</div>

## 2단계: 등분산성 확인

분산분석은 모든 집단이 공통 분산 $\sigma^2$을 공유한다고 가정한다. Levene 검정은 비정규성에 로버스트하고, Bartlett 검정은 자료가 정말 정규일 때 최적이지만 정규성 이탈에 민감하다.

집단이 $k$개일 때 Levene 검정통계량은

$$
W = \frac{(N - k)}{(k - 1)} \cdot \frac{\sum_{i=1}^{k} n_i (\bar{Z}_{i\cdot} - \bar{Z}_{\cdot\cdot})^2}{\sum_{i=1}^{k} \sum_{j=1}^{n_i} (Z_{ij} - \bar{Z}_{i\cdot})^2}
$$

이며 $Z_{ij} = |y_{ij} - \tilde{y}_{i}|$이고 $\tilde{y}_i$는 집단 중앙값이다.

<div class="codebox" markdown>

### 예제 3. 2단계 — 등분산성 { .eg }

```python
from scipy.stats import levene, bartlett

# 두 검정을 함께 돌려 결론이 갈리는지 본다. 갈린다면 정규성이 의심스럽다는
# 뜻이므로, 정규성을 덜 타는 Levene 쪽을 믿는다.
groups = [data[data['group'] == g]['response'].values for g in data['group'].unique()]
stat_lev, p_lev = levene(*groups)
stat_bart, p_bart = bartlett(*groups)
print(f"Levene:   W = {stat_lev:.4f}, p = {p_lev:.4f}")
print(f"Bartlett: chi2 = {stat_bart:.4f}, p = {p_bart:.4f}")
```

출력:

```
Levene:   W = 1.1666, p = 0.3188
Bartlett: chi2 = 16.6837, p = 0.0002
```

두 검정의 결론이 갈린다. Bartlett은 $p = 0.0002$로 등분산을 강하게 기각하고, Levene은 $p = 0.32$로 기각하지 못한다.

이것이 두 검정의 성격 차이를 보여주는 전형적인 예다. Bartlett은 정규성을 전제하므로 이상점 하나에 크게 흔들린다. Levene은 중앙값으로부터의 절대편차를 쓰므로 그 한 점에 덜 끌려간다. **자료에 이상점이 있을 때 Bartlett의 기각은 분산 차이의 증거가 아니라 이상점의 증거일 수 있다.**

등분산성이 기각되면 Welch 분산분석이나 이분산에 로버스트한 접근(HC3 공분산)을 써야 한다.

</div>

## 3단계: 독립성 확인

Durbin-Watson 통계량은 잔차의 1차 자기상관을 탐지한다:

$$
d = \frac{\sum_{t=2}^{n}(e_t - e_{t-1})^2}{\sum_{t=1}^{n} e_t^2}
$$

2에 가까운 값은 자기상관이 없음을, 0에 가까우면 양의 자기상관을, 4에 가까우면 음의 자기상관을 시사한다. 흔한 경험 법칙은 $d \in (1.5, 2.5)$이면 받아들일 만하다는 것이다.

<div class="codebox" markdown>

### 예제 4. 3단계 — 독립성 { .eg }

```python
from statsmodels.stats.stattools import durbin_watson

# 통계량은 0 에서 4 사이이고 2 가 무상관에 해당한다. 2 보다 뚜렷이 작으면
# 양의 자기상관, 크면 음의 자기상관이다.
dw = durbin_watson(model.resid)
print(f"Durbin-Watson: {dw:.4f}")
```

출력:

```
Durbin-Watson: 1.7586
```

1.76으로 경험칙의 범위 $(1.5, 2.5)$ 안에 있어 자기상관의 증거가 없다. 다만 여기서 "순서"는 자료프레임의 행 번호일 뿐이므로, 이 값이 의미를 가지려면 자료가 실제 수집 순서대로 정렬되어 있어야 한다.

잔차 대 적합값 산점도에는 알아볼 만한 패턴이 없어야 한다.

</div>

## 4단계: 영향점 찾기

Cook의 거리는 각 관측값이 적합 모형에 주는 영향을 잰다. 흔한 문턱은

$$
D_i > \frac{4}{n}
$$

이며 $n$은 전체 관측 수이다. 이 문턱을 넘는 관측값은 자료 입력 오류인지 아니면 정말로 특이한 조건인지 조사해야 한다.

<div class="codebox" markdown>

### 예제 5. 4단계 — 영향점 { .eg }

```python
# Cook 의 거리는 그 관측값 하나를 뺐을 때 적합값 전체가 얼마나 움직이는지를
# 잰다. 크다는 것은 결론이 그 한 점에 기대고 있다는 뜻이다.
influence = model.get_influence()
cooks_d = influence.cooks_distance[0]

# 4/n 은 널리 쓰이는 어림 기준일 뿐 검정이 아니다. 넘는 점은 지울 대상이
# 아니라 들여다볼 대상이다.
threshold = 4 / len(cooks_d)
flagged = np.where(cooks_d > threshold)[0]
print(f"threshold = {threshold:.4f}")
print(f"flagged observations = {flagged}")
print(f"max Cook's D = {cooks_d.max():.4f} (obs {cooks_d.argmax()})")
```

출력:

```
threshold = 0.0667
flagged observations = [52 59]
max Cook's D = 0.5058 (obs 59)
```

문턱을 넘는 관측값이 둘이고, 그중 압도적인 것이 마지막 관측값(59번)이다. Cook 거리 0.506은 문턱 0.067의 여덟 배에 가깝고 두 번째로 큰 값과도 크게 벌어져 있다. 설정에서 20.0으로 바꿔 심어 둔 바로 그 점이다.

Cook의 거리는 정규성 검정이나 등분산 검정과 달리 **어느 관측값이** 문제인지 짚어 준다. 진단의 순서를 이렇게 잡으면 좋다. 먼저 영향점을 찾고, 그것을 제거했을 때 결론이 바뀌는지 확인한 뒤, 남은 문제를 분포 가정의 문제로 다룬다.

</div>

## 전부 합치기

다음 함수는 어떤 일원배치 분산분석 설계에도 전체 파이프라인을 실행하고 2×2 진단 패널(Q-Q 그림, 잔차 히스토그램, 잔차 대 적합값, Cook의 거리)을 만든다.

<div class="codebox" markdown>

### 예제 6. 네 진단을 한꺼번에 { .eg }

```python
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

def run_full_diagnostics(data, response_col, group_col):
    """앞의 진단 넷을 한 번에 돌려 2x2 격자로 보여 준다.

    실제 분석에서는 이 네 그림을 늘 함께 본다. 하나만 보고 판단하면
    다른 쪽에서 드러날 문제를 놓치기 쉽다.
    """
    formula = f'{response_col} ~ {group_col}'
    model = ols(formula, data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # 왼쪽 위: Q-Q 그림 — 정규성
    sm.qqplot(model.resid, line='s', ax=axes[0, 0])
    # 오른쪽 위: 잔차 히스토그램 — 치우침과 봉우리
    axes[0, 1].hist(model.resid, bins=15, density=True, alpha=0.7, edgecolor='black')
    # 왼쪽 아래: 잔차 대 적합값 — 등분산성과 남은 구조
    axes[1, 0].scatter(model.fittedvalues, model.resid, alpha=0.6)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    # 오른쪽 아래: Cook 의 거리 — 영향점
    cooks_d = model.get_influence().cooks_distance[0]
    axes[1, 1].stem(range(len(cooks_d)), cooks_d, markerfmt=",")
    axes[1, 1].axhline(y=4 / len(cooks_d), color='r', linestyle='--')
    plt.tight_layout()
    plt.show()

run_full_diagnostics(data, "response", "C(group)")
```

출력:

```
              sum_sq    df          F    PR(>F)
C(group)   66.024886   2.0  16.131362  0.000003
Residual  116.649122  57.0        NaN       NaN
```

![분산분석 진단 패널](./img/anova_diagnostics_196.png)

네 그림을 한자리에 놓으면 이야기가 분명해진다. Q-Q 그림의 오른쪽 끝, 히스토그램의 오른쪽 꼬리, 잔차 그림의 위쪽 외딴 점, Cook 거리의 마지막 막대가 모두 **같은 관측값 하나**를 가리킨다.

</div>

## 해석

- **정규성:** Q-Q 그림이 기준선을 따르고 Shapiro-Wilk의 $p > 0.05$이면 정규성이 성립한다. 집단이 크고 균형 잡혀 있으면 중심극한정리 덕분에 약한 이탈은 덜 중요하다.
- **등분산성:** Levene의 $p > 0.05$이면 등분산 가정이 합당하다. 그렇지 않으면 Welch 분산분석을 쓴다.
- **독립성:** Durbin-Watson 값이 2 근처이고 잔차 그림에 패턴이 없으면 독립성을 뒷받침한다.
- **영향점:** Cook의 $D > 4/n$인 관측값은 살펴보아야 한다. 영향점 하나를 제거하고 분석을 다시 해 보면 결론이 그 관측값에 민감한지 알 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
어떤 연구자가 집단 $k = 4$개, 전체 관측 $n = 50$개로 일원배치 분산분석을 적합했다. Durbin-Watson 통계량은 $d = 0.85$이다. 무엇을 뜻하며 연구자는 무엇을 해야 하는가?

</div>

??? success "풀이"
    Durbin-Watson 통계량 $d = 0.85$는 받아들일 만한 범위 $(1.5, 2.5)$의 하한보다 한참 낮아 잔차 사이에 강한 양의 자기상관이 있음을 나타낸다. 연속한 잔차의 부호가 같은 경향이 있다는 뜻이며 분산분석의 독립성 가정을 위반한다.

    연구자는 자료 수집 과정을 조사해야 한다. 자료를 시간에 걸쳐 수집했다면 시계열 모형이나 반복측정 분산분석이 더 적절할 수 있다. 순서에 의미가 없다면 관측 순서를 다시 무작위화하고 재확인하여 자기상관이 정렬 때문에 생긴 인공물인지 밝힐 수 있다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
Levene 검정이 집단 평균이 아니라 집단 중앙값으로부터의 절대편차를 쓰는 이유를 설명하라. 평균을 쓰면 어떤 상황에서 오도하는 결과가 나오는가?

</div>

??? success "풀이"
    집단 중앙값을 쓰면 Levene 검정이 치우친 분포와 이상점에 로버스트해진다. 중앙값은 극단값에 저항적이므로 변환된 변수 $Z_{ij} = |y_{ij} - \tilde{y}_i|$는 $|y_{ij} - \bar{y}_i|$보다 비정규성의 영향을 덜 받는다.

    바탕 분포가 심하게 치우쳐 있거나 이상점을 포함하면 집단 평균이 극단값 쪽으로 끌려가 해당 집단의 절대편차가 부풀려진다. 그러면 검정이 등분산성을 거짓으로 기각하거나(제1종 오류 부풀림) 반대로 진짜 분산 차이를 가릴 수 있다. 중앙값 기반 형태(Brown-Forsythe 변형)는 더 넓은 범위의 분포 모양에서 명목 제1종 오류율을 유지한다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
Cook의 거리 문턱 $4/n$의 근거를 유도하라. 구체적으로 $D_i$가 근사적으로 $\text{Beta}\!\bigl(\tfrac{p}{2},\, \tfrac{n-p}{2}\bigr)$를 따른다고 할 때 $E[D_i] \approx p/n$임을 보이고, $4/n$이 왜 실용적인 단순화인지 설명하라.

</div>

??? success "풀이"
    관측값 $i$에 대한 Cook의 거리는 베타분포와 연결할 수 있다. 근사 $D_i \sim \text{Beta}(p/2,\, (n-p)/2)$ 아래에서 $\text{Beta}(\alpha, \beta)$ 확률변수의 기댓값은

    $$
    E[D_i] = \frac{\alpha}{\alpha + \beta} = \frac{p/2}{p/2 + (n-p)/2} = \frac{p}{n}
    $$

    이다. 집단이 $k$개인 일원배치 분산분석에서는 (절편을 포함하여) $p = k$이므로 $E[D_i] = k/n$이다. 문턱 $4/n$은 Cook의 거리가 평균의 약 네 배인 관측값을 고르는 것에 대략 대응하며, 영향점을 표시하는 표준적인 경험 법칙이다. $k$가 $n$에 비해 작으면 $4/n$과 $4k/n$의 크기가 비슷하므로 $4/n$이 편리한 단순화가 된다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
어떤 자료에 분산분석 진단을 수행했더니 정규성은 성립하지만 Levene 검정이 $p = 0.003$으로 등분산을 기각했다. Bartlett 검정은 $p = 0.001$이다. 표본크기는 $n_1 = 50$, $n_2 = 12$, $n_3 = 45$이다. 적절한 다음 단계와 구체적인 대안 분석을 기술하라.

</div>

??? success "풀이"
    Levene 검정과 Bartlett 검정이 모두 등분산성을 기각하므로, 등분산을 가정하는 고전적 분산분석 F-검정을 믿을 수 없다. 불균형 설계($n_2 = 12$가 다른 집단보다 훨씬 작다)가 문제를 키운다. 작은 집단의 분산이 크면 F-검정이 관대해지고(제1종 오류 부풀림), 작은 집단의 분산이 작으면 보수적이 된다.

    적절한 대안은 등분산을 가정하지 않고 Satterthwaite 형태의 근사로 자유도를 조정하는 Welch 분산분석이다. 사후 쌍별 비교에는 분산과 표본크기의 불균형을 함께 반영하는 Games-Howell이 Welch 분산분석의 자연스러운 짝이다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
Durbin-Watson 통계량이 $0 \le d \le 4$를 만족하고 $d = 2$가 잔차의 1차 자기상관이 0인 경우에 대응함을 증명하라.

</div>

??? success "풀이"
    Durbin-Watson 통계량은

    $$
    d = \frac{\sum_{t=2}^{n}(e_t - e_{t-1})^2}{\sum_{t=1}^{n} e_t^2}
    $$

    이다.

    **하한:** 모든 $t$에 대해 $(e_t - e_{t-1})^2 \ge 0$이므로 분자가 음이 아니다. 분모는 제곱합이므로 (모든 잔차가 0이 아닌 한) 양수이다. 따라서 $d \ge 0$이다.

    **상한:** 분자를 전개하면

    $$
    \sum_{t=2}^{n}(e_t - e_{t-1})^2 = \sum_{t=2}^{n} e_t^2 - 2\sum_{t=2}^{n} e_t e_{t-1} + \sum_{t=2}^{n} e_{t-1}^2
    $$

    이다. 첫째와 셋째 합은 각각 많아야 $\sum_{t=1}^{n} e_t^2$이고, Cauchy-Schwarz 부등식에 의해 $|\sum e_t e_{t-1}| \le \sum e_t^2$이므로 분자는 많아야 $4 \sum e_t^2$이 되어 $d \le 4$이다.

    **자기상관이 0인 경우:** 1차 자기상관 $\hat{\rho}_1 = \sum_{t=2}^{n} e_t e_{t-1} / \sum_{t=1}^{n} e_t^2 \approx 0$이면 교차항이 사라지고 전개식에 남은 두 합이 각각 대략 $\sum e_t^2$이 되어 $d \approx 2(1 - \hat{\rho}_1) \approx 2$가 된다. $\square$

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
연습문제 3이 유도한 문턱 $4/N$을 **실제로 시험**하라. 이상점이 전혀 없는 자료에서 이 문턱은 얼마나 자주 경보를 울리는가?

</div>

??? success "풀이"
    ```python
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    from statsmodels.formula.api import ols

    rng = np.random.default_rng(8001)
    B = 3_000
    print("모든 관측이 같은 정규분포 — 이상점이 하나도 없는 자료")
    print(f"{'k':>3s} {'집단당 n':>8s} {'N':>5s} {'4/N':>8s} "
          f"{'D>4/N 인 관측 비율':>16s} {'적어도 하나':>10s} {'D>1 비율':>9s}")
    for k, n in [(3, 10), (3, 20), (4, 15), (5, 30), (3, 50)]:
        N = k * n
        thr = 4 / N
        frac, any_, big = [], 0, 0
        for _ in range(B):
            rows = [pd.DataFrame({"y": rng.normal(0, 1, n), "g": f"G{g}"})
                    for g in range(k)]
            df = pd.concat(rows, ignore_index=True)
            D = ols("y ~ C(g)", data=df).fit().get_influence().cooks_distance[0]
            frac.append((D > thr).mean())
            any_ += (D > thr).any()
            big += (D > 1).any()
        print(f"{k:3d} {n:8d} {N:5d} {thr:8.4f} {np.mean(frac):16.4f} "
              f"{any_ / B:10.4f} {big / B:9.4f}")
    ```

    ```text
    모든 관측이 같은 정규분포 — 이상점이 하나도 없는 자료
      k    집단당 n     N      4/N    D>4/N 인 관측 비율     적어도 하나    D>1 비율
      3       10    30   0.1333           0.0560     0.9303    0.0000
      3       20    60   0.0667           0.0502     0.9933    0.0000
      4       15    60   0.0667           0.0526     0.9937    0.0000
      5       30   150   0.0267           0.0487     1.0000    0.0000
      3       50   150   0.0267           0.0473     1.0000    0.0000
    ```

    **$4/N$은 깨끗한 자료에서도 관측의 약 5%를 표시한다.**

    | $N$ | 표시되는 비율 | 적어도 하나 표시될 확률 |
    |---|---|---|
    | 30 | 0.056 | **0.930** |
    | 60 | 0.050 | **0.993** |
    | 150 | 0.047~0.049 | **1.000** |

    **$N=150$이면 이상점이 하나도 없어도 반드시 경보가 울린다.**

    **이것은 결함이 아니라 설계다.** $4/N$은

    $$
    E[D_i]\approx\frac{p}{N}
    $$

    의 **약 네 배**로 정한 값이다. "평균의 네 배"를 넘는 관측은 **어느 자료에나 5% 정도 있다.**

    **$D>1$은 전혀 나오지 않는다**(15,000번의 모의실험에서 0회). $D_i>1$은 **훨씬 엄격한 문턱**이며, 실제로 그것이 원래 쿡(1977)의 권고였다.

    | 문턱 | 근거 | 깨끗한 자료에서 |
    |---|---|---|
    | $D_i>4/N$ | 평균의 4배(경험칙) | **5% 표시** |
    | $D_i>1$ | 계수가 50% 신뢰영역만큼 이동 | **거의 0** |
    | $D_i>F_{0.5}(p,N-p)$ | 원래의 형식적 기준 | $D_i>1$과 비슷 |

    **그럼 $4/N$은 쓸모없는가.** 아니다. **용도가 다르다.**

    | 문턱 | 용도 |
    |---|---|
    | $4/N$ | **살펴볼 후보를 고르는 선별 도구** |
    | $D_i>1$ | **실제로 결론을 바꾸는 점을 찾는 도구** |

    **$4/N$이 표시한 점을 "이상점"이라 부르면 안 된다.** "상대적으로 영향이 큰 관측"일 뿐이다.

    **권장 절차 넷.**

    1. **$D_i$를 크기순으로 정렬**해 상위 몇 개를 본다(문턱보다 순위가 유용하다).
    2. **$D_i$ 그림**을 그려 **뚜렷하게 튀는 점**이 있는지 본다.
    3. 그런 점이 있으면 **빼고 다시 적합해** 결론이 바뀌는지 확인한다.
    4. **결론이 바뀌면 두 결과를 모두 보고**한다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span>
분산분석에서 쓰는 **네 종류의 잔차**를 한 자료에서 모두 계산하고, 영향 지표들 사이의 **대수적 관계**를 확인하라.

</div>

??? success "풀이"
    **네 종류.**

    | 이름 | 정의 |
    |---|---|
    | 원 잔차 | $e_i=y_i-\hat y_i$ |
    | **표준화(내적 스튜던트화)** | $r_i=\dfrac{e_i}{\hat\sigma\sqrt{1-h_{ii}}}$ |
    | **스튜던트화(외적, 삭제)** | $t_i=\dfrac{e_i}{\hat\sigma_{(i)}\sqrt{1-h_{ii}}}$ |
    | 예측 잔차 | $e_{(i)}=\dfrac{e_i}{1-h_{ii}}$ |

    ```python
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    from statsmodels.formula.api import ols

    rng = np.random.default_rng(8002)
    rows = [pd.DataFrame({"y": rng.normal(mu, 1.2, n), "g": f"G{g}"})
            for g, (n, mu) in enumerate([(8, 10.0), (12, 12.0), (6, 11.0)])]
    df = pd.concat(rows, ignore_index=True)
    df.loc[len(df) - 1, "y"] = 18.0          # 작은 집단에 이상점 하나

    fit = ols("y ~ C(g)", data=df).fit()
    inf = fit.get_influence()
    t = pd.DataFrame({
        "g": df.g, "y": df.y.round(3),
        "잔차": fit.resid.round(3),
        "h": inf.hat_matrix_diag.round(4),
        "표준화": inf.resid_studentized_internal.round(3),
        "스튜던트화(삭제)": inf.resid_studentized_external.round(3),
        "CookD": inf.cooks_distance[0].round(4),
        "DFFITS": inf.dffits[0].round(3)})
    print(t.tail(8).to_string(index=False))

    N, p = len(df), 3
    print(f"\nN = {N}, p = {p}")
    print(f"  h 의 값: {sorted(set(np.round(inf.hat_matrix_diag, 4)))}")
    print(f"  집단 크기: {df.g.value_counts().sort_index().to_dict()}")
    print(f"  Σh = {inf.hat_matrix_diag.sum():.4f}  (이론 p = {p})")
    print(f"\n문턱: 4/N = {4 / N:.4f},  DFFITS 2√(p/N) = {2 * np.sqrt(p / N):.4f}, "
          f"h 의 2p/N = {2 * p / N:.4f}")

    i = len(df) - 1
    r = inf.resid_studentized_internal[i]
    h = inf.hat_matrix_diag[i]
    te = inf.resid_studentized_external[i]
    print(f"\n관계 확인 (마지막 관측):")
    print(f"  Cook D = r²/p · h/(1-h) = {r**2 / p * h / (1 - h):.6f}   "
          f"실제 {inf.cooks_distance[0][i]:.6f}")
    print(f"  DFFITS = t·√(h/(1-h))  = {te * np.sqrt(h / (1 - h)):.6f}   "
          f"실제 {inf.dffits[0][i]:.6f}")
    ```

    ```text
     g      y     잔차      h    표준화  스튜던트화(삭제)  CookD  DFFITS
    G1 12.268 -0.238 0.0833 -0.169     -0.165 0.0009  -0.050
    G1 12.343 -0.164 0.0833 -0.116     -0.113 0.0004  -0.034
    G2 11.720 -1.141 0.1667 -0.847     -0.841 0.0478  -0.376
    G2 10.570 -2.291 0.1667 -1.701     -1.779 0.1929  -0.796
    G2 12.256 -0.605 0.1667 -0.449     -0.441 0.0134  -0.197
    G2 13.587  0.726 0.1667  0.539      0.531 0.0194   0.237
    G2 11.031 -1.830 0.1667 -1.359     -1.385 0.1230  -0.620
    G2 18.000  5.140 0.1667  3.816      6.162 0.9709   2.756

    N = 26, p = 3
      h 의 값: [0.0833, 0.125, 0.1667]
      집단 크기: {'G0': 8, 'G1': 12, 'G2': 6}
      Σh = 3.0000  (이론 p = 3)

    문턱: 4/N = 0.1538,  DFFITS 2√(p/N) = 0.6794, h 의 2p/N = 0.2308

    관계 확인 (마지막 관측):
      Cook D = r²/p · h/(1-h) = 0.970882   실제 0.970882
      DFFITS = t·√(h/(1-h))  = 2.755922   실제 2.755922
    ```

    **분산분석의 지렛값은 집단 크기의 역수다.** $h=0.0833,\ 0.125,\ 0.1667$이고 집단 크기가 $12,\ 8,\ 6$이다. **$h_{ii}=1/n_i$**가 정확히 성립한다.

    $$
    \sum_i h_{ii}=\sum_g n_g\cdot\frac{1}{n_g}=k=p
    $$

    **표에서 $\Sigma h=3.0000=p$**로 확인된다.

    **따라서 분산분석에서 지렛값은 진단 정보가 아니다.** 회귀와 달리 **$x$가 없으므로** 지렛값이 오직 집단 크기만 반영한다. **작은 집단의 관측이 자동으로 지렛값이 크다.**

    **표준화와 스튜던트화(삭제)의 차이가 이상점에서 극적이다.**

    | | 이상점 관측 | 두 번째로 큰 잔차 |
    |---|---|---|
    | 표준화 $r$ | **3.816** | $-1.701$ |
    | 스튜던트화 $t$ | **6.162** | $-1.779$ |

    **삭제 잔차가 훨씬 크다.** $\hat\sigma$에 이상점 자신이 들어가 있으면 분모가 부풀어 **이상점을 스스로 감춘다.** 삭제 잔차는 그 관측을 빼고 $\hat\sigma_{(i)}$를 계산하므로 감춰지지 않는다.

    **$r$은 $\sqrt{N-p}$를 넘을 수 없다**는 수학적 상한이 있다. 여기서는 $\sqrt{23}=4.80$이므로 3.816이 이미 상한에 가깝다. **$t$에는 그런 상한이 없다.**

    **대수적 관계가 정확히 성립한다.**

    $$
    D_i=\frac{r_i^2}{p}\cdot\frac{h_{ii}}{1-h_{ii}},
    \qquad
    \text{DFFITS}_i=t_i\sqrt{\frac{h_{ii}}{1-h_{ii}}}
    $$

    **두 지표는 같은 정보를 다르게 담는다.** $D_i$는 **계수 전체의 이동**, DFFITS는 **그 관측의 적합값 이동**을 재고, 부호의 유무가 다르다.

    **실무 권고.**

    | 목적 | 지표 |
    |---|---|
    | **이상점 탐지** | 스튜던트화(삭제) 잔차 |
    | **영향 탐지** | 쿡 거리 |
    | 방향까지 보기 | DFFITS |
    | 특정 계수에의 영향 | DFBETAS |
    | 정규성 진단 | **표준화 잔차**(등분산으로 보정됨) |

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
이상점을 **$|t_i|>2$ 같은 고정 문턱**으로 찾을 때의 다중성 문제를 재고, 올바른 보정을 제시하라.

</div>

??? success "풀이"
    ```python
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    from scipy import stats
    from statsmodels.formula.api import ols

    rng = np.random.default_rng(8003)
    B = 3_000
    print("자료에 이상점이 전혀 없을 때, 적어도 하나가 표시될 확률")
    print(f"{'N':>5s} {'|t*|>2':>9s} {'|t*|>3':>9s} {'본페로니 보정':>12s} {'보정 임계값':>11s}")
    for k, n in [(3, 10), (3, 20), (4, 25), (5, 40)]:
        N = k * n
        tb = stats.t.ppf(1 - 0.05 / (2 * N), N - k - 1)
        a = b = c = 0
        for _ in range(B):
            rows = [pd.DataFrame({"y": rng.normal(0, 1, n), "g": f"G{g}"})
                    for g in range(k)]
            df = pd.concat(rows, ignore_index=True)
            te = (ols("y ~ C(g)", data=df).fit()
                  .get_influence().resid_studentized_external)
            a += (np.abs(te) > 2).any()
            b += (np.abs(te) > 3).any()
            c += (np.abs(te) > tb).any()
        print(f"{N:5d} {a / B:9.4f} {b / B:9.4f} {c / B:12.4f} {tb:11.4f}")
    ```

    ```text
    자료에 이상점이 전혀 없을 때, 적어도 하나가 표시될 확률
        N    |t*|>2    |t*|>3      본페로니 보정      보정 임계값
       30    0.9340    0.1733       0.0533      3.5069
       60    0.9923    0.2417       0.0513      3.5322
      100    1.0000    0.3057       0.0453      3.6047
      200    1.0000    0.4857       0.0533      3.7314
    ```

    **$|t|>2$는 $N=100$ 이상이면 반드시 무언가를 표시한다.**

    | $N$ | $\|t\|>2$ | $\|t\|>3$ | **본페로니** |
    |---|---|---|---|
    | 30 | 0.934 | 0.173 | **0.053** |
    | 60 | 0.992 | 0.242 | **0.051** |
    | 100 | **1.000** | 0.306 | **0.045** |
    | 200 | **1.000** | 0.486 | **0.053** |

    **$|t|>3$도 $N=200$에서는 절반이 걸린다.**

    **본페로니 보정이 정확히 작동한다**(0.045~0.053). 임계값은

    $$
    t_{1-\alpha/(2N),\ N-p-1}
    $$

    이다. $N=30$에서 3.51, $N=200$에서 3.73으로 **$N$에 따라 커진다.**

    **왜 $N$이 커져도 임계값이 조금만 커지는가.** $t$ 분포의 꼬리가 지수적으로 얇아지므로, 분위수는 $\sqrt{2\ln N}$ 정도로 **아주 천천히** 자란다.

    **이것이 본페로니 이상점 검정**(외적 스튜던트화 잔차의 최댓값 검정)이며, 고전적 이름이 있다.

    | 이름 | 내용 |
    |---|---|
    | **본페로니 이상점 검정** | $\max_i\|t_i\|$를 보정 임계값과 비교 |
    | 그러브스 검정 | 같은 발상의 일표본 판 |
    | `car::outlierTest` (R) | 이 절차의 표준 구현 |

    **구현.**

    ```text
    p_i = 2 · P(t_{N-p-1} > |t_i|)          각 관측의 원 p-값
    p_i^adj = min(1, N · p_i)               본페로니 보정
    → 가장 작은 p_i^adj 만 보고
    ```

    **주의 — 이 검정은 "이상점이 하나"를 전제**한다. 여럿이면 **가림 현상(masking)**으로 놓칠 수 있다. 여러 이상점이 의심되면 **로버스트 회귀**(MM-추정 등)로 시작하는 것이 낫다.

    **실무 지침 셋.**

    1. **고정 문턱($|t|>2$, $|t|>3$)을 쓰지 않는다.**
    2. **본페로니 보정 임계값**을 쓰거나, 단순히 **가장 큰 것 하나만** 조사한다.
    3. 통계적 표시는 **조사의 시작점**이지 제거의 근거가 아니다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
연습문제 4의 상황($n=50,12,45$, 레빈 $p=0.003$, 바틀렛 $p=0.001$)을 **수치로 분석**하고 구체적인 대안을 제시하라.

</div>

??? success "풀이"
    **먼저 어느 방향의 불균형인지 물어야 한다.** 문제는 표본 크기만 주고 분산은 주지 않았다. **두 시나리오가 정반대의 결과**를 낳는다.

    ```python
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    from scipy import stats

    def welch_p(gs):
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
        return stats.f.sf(F, k - 1, (3 / (k**2 - 1) * lam)**-1)

    rng = np.random.default_rng(8004)
    B = 10_000
    NS = [50, 12, 45]
    print("n = (50, 12, 45), 모든 평균이 같음, 명목 0.05")
    print(f"{'시나리오':>28s} {'표준 F':>8s} {'웰치':>8s}")
    for lab, sds in [("역페어링: 작은 집단에 큰 분산 σ=(1,3,1)", [1, 3, 1]),
                     ("정페어링: 작은 집단에 작은 분산 σ=(3,1,3)", [3, 1, 3]),
                     ("큰 집단끼리 다름 σ=(1,1,3)", [1, 1, 3])]:
        a = b = 0
        for _ in range(B):
            gs = [rng.normal(0, s, n) for n, s in zip(NS, sds)]
            a += stats.f_oneway(*gs).pvalue < 0.05
            b += welch_p(gs) < 0.05
        print(f"{lab:>28s} {a / B:8.4f} {b / B:8.4f}")
    ```

    ```text
    n = (50, 12, 45), 모든 평균이 같음, 명목 0.05
                            시나리오     표준 F       웰치
     역페어링: 작은 집단에 큰 분산 σ=(1,3,1)   0.2656   0.0541
    정페어링: 작은 집단에 작은 분산 σ=(3,1,3)   0.0259   0.0515
             큰 집단끼리 다름 σ=(1,1,3)   0.0363   0.0500
    ```

    **표준 $F$의 오류율이 시나리오에 따라 0.026에서 0.266까지 요동친다.** 웰치는 모두 0.050~0.054다.

    | 시나리오 | 표준 $F$ | 웰치 |
    |---|---|---|
    | **역페어링**(작은 집단에 큰 분산) | **0.266** | 0.054 |
    | 정페어링 | **0.026** | 0.052 |
    | 큰 집단끼리 다름 | 0.036 | 0.050 |

    **$n_2=12$가 문제의 핵심**이다. 이 집단의 분산이 크면 오류율이 **명목의 다섯 배**, 작으면 절반 이하로 떨어진다.

    **세 번째 줄도 보수적이다**(0.036). 분산이 큰 집단($\sigma=3$)이 $n=45$로 크기 때문이다. **작은 집단에 어떤 분산이 붙는가가 모든 것을 정한다.**

    **권장 절차 다섯.**

    1. **집단별 $n_i$와 $s_i$ 표를 먼저 본다.** 어느 방향인지 확인한다.
    2. **웰치 분산분석**을 수행한다(모든 시나리오에서 안전).
    3. 유의하면 **게임스-하웰** 사후검정.
    4. **바틀렛의 $p=0.001$은 무시**한다. 정규성이 확인되었다 해도 바틀렛은 과민하다.
    5. **$n_2=12$의 $s_2$는 매우 부정확**하다는 점을 보고서에 밝힌다.

    **네 번째에 관해.** 문제는 "정규성은 성립한다"고 했으므로 바틀렛을 신뢰할 수 있지만, **레빈과 바틀렛이 같은 방향을 가리키므로** 굳이 구분할 필요가 없다. 두 검정 모두 이분산을 말한다.

    **다섯 번째가 자주 잊힌다.** $n=12$에서 $s^2$의 95% 구간은 대략

    $$
    \left[\frac{11s^2}{21.92},\ \frac{11s^2}{3.82}\right]=[0.50s^2,\ 2.88s^2]
    $$

    로 **거의 여섯 배의 폭**이다. "집단 2의 분산이 크다"는 결론 자체가 불확실하다.

    **추가 고려 — 변환.** 분산이 평균과 함께 커진다면 로그 변환이 이분산과 정규성을 함께 개선할 수 있다. **집단 평균 대 표준편차를 그려** 확인한다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
분산분석 진단의 **전체 점검표**를 만들어라.

</div>

??? success "풀이"
    **진단은 네 가정에 대응한다.**

    | 가정 | 도구 | 위반 시 오류율 |
    |---|---|---|
    | **독립성** | 설계 검토, 더빈-왓슨 | **0.48**($\rho=0.6$) |
    | 등분산 | 잔차 대 적합값, 브라운-포사이드 | 0.29(불균형) |
    | 정규성 | **Q-Q 그림**, 치우침·첨도 | 0.04(보수적) |
    | (이상점) | 스튜던트화 잔차, 쿡 거리 | 경우에 따라 |

    **점검표.**

    ```text
    [설계]
      □ 관측이 독립인가 (반복측정·군집·시계열?)
      □ 집단별 n 이 균형에 가까운가

    [잔차]
      □ 잔차 대 적합값 그림 — 깔때기 모양?
      □ Q-Q 그림 — 꼬리, 치우침, 계단?
      □ 관측 순서에 대한 잔차 — 추세·주기?

    [이상점·영향점]
      □ 스튜던트화(삭제) 잔차의 최댓값 — 본페로니 보정 임계값과 비교
      □ 쿡 거리 — 크기순 정렬, 뚜렷하게 튀는 점?
      □ 표시된 점을 빼고 다시 적합 — 결론이 바뀌는가?

    [검정]
      □ 브라운-포사이드 (참고용)
      □ 정규성 검정 (n 이 크면 참고만)
    ```

    **핵심 수치 넷.**

    | 사실 | 값 |
    |---|---|
    | 깨끗한 자료에서 $D>4/N$이 표시하는 비율 | **5%** |
    | $N=150$에서 적어도 하나 표시될 확률 | **1.000** |
    | $N=100$에서 $\|t^*\|>2$가 표시될 확률 | **1.000** |
    | 분산분석의 지렛값 | $h_{ii}=1/n_i$ |

    **문턱을 쓰는 원칙.**

    | 지표 | 선별용 | 판단용 |
    |---|---|---|
    | 쿡 거리 | $4/N$ | **$D_i>1$** |
    | 스튜던트화 잔차 | $\|t\|>2$ | **본페로니 임계값** |
    | 지렛값 | $2p/N$ | 분산분석에서는 **무의미** |

    **마지막 줄을 잊지 말자.** 분산분석에서 $h_{ii}=1/n_i$이므로, **작은 집단의 모든 관측이 자동으로 "높은 지렛값"**이다. 회귀의 지렛값 진단을 그대로 옮기면 안 된다.

    **이상점을 발견했을 때의 순서.**

    ```text
    1. 기록 오류인가?  ──→ 고치거나 결측 처리
    2. 다른 모집단에서 왔는가?  ──→ 제외하고 그 사실을 보고
    3. 그냥 극단값인가?  ──→ 남긴다
         ↓
    4. 있을 때와 없을 때의 결과를 모두 계산
         ↓
    5. 결론이 바뀌면 둘 다 보고, 바뀌지 않으면 그 사실을 보고
    ```

    **"쿡 거리가 커서 제거했다"는 정당한 이유가 아니다.** 통계적 표시는 **조사를 시작하라는 신호**일 뿐이다.

    **처방 대응표.**

    | 위반 | 처방 |
    |---|---|
    | 독립성 | **혼합효과 모형, 시계열 모형**(검정을 바꿔서는 안 됨) |
    | 등분산 | **웰치**, 변환 |
    | 정규성 | **순열검정**, 변환, 크러스컬-월리스 |
    | 이상점 | 조사 → 로버스트 방법 → 민감도 분석 |

    **한 문장.** 진단의 목적은 **가정을 통과시키는 것이 아니라**, 어떤 방법이 이 자료에 맞는지 **판단할 정보를 모으는 것**이다.

---

## 정리하며

진단을 **하나의 흐름**으로 묶었다.

- **네 가지를 차례로 본다.** 잔차의 정규성, 등분산성, 독립성, 영향점. 각각 형식적 검정과 그림을 함께 쓴다.
- **그림이 검정보다 정보가 많다.** 적합값 대 잔차 산점도 하나가 등분산성과 선형성을 동시에 보여 주고, Q-Q 그림이 정규성을, 순서 대 잔차 그림이 독립성의 실마리를 준다.
- **검정 결과를 판정 규칙으로 쓰지 말 것.** 표본이 크면 모든 가정 검정이 기각되고 작으면 아무것도 기각되지 않는다. **판단의 재료이지 판단 자체가 아니다.**
- **진단은 분석의 끝이 아니라 중간이다.** 문제를 찾으면 앞 절의 처방으로 돌아가고, 고친 뒤 다시 진단한다.
- **보고에 포함한다.** 어떤 가정을 어떻게 확인했고 무엇을 발견했는지 적는 것이 결과의 신뢰도를 뒷받침한다.

다음 절부터 **실무 응용**으로 넘어간다.
