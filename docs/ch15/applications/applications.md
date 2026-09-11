# 회귀와 분산분석에서의 응용


분산의 동일성 검정은 여러 통계 방법, 특히 회귀분석과 분산분석에서 결정적인 단계이다. 두 기법 모두 분산의 동질성(등분산성)에 대한 가정이 타당하고 신뢰할 만한 추론을 보장하는 근본적인 역할을 한다.

## 회귀에서의 분산 검정

회귀분석의 핵심 가정 가운데 하나가 **등분산성**이다. 잔차의 분산이 설명변수의 모든 수준에서 일정해야 한다. 이 가정이 위배되면 추론이 무효가 되어 잘못된 결론으로 이어질 수 있다.

!!! warning "이분산은 계수를 편향시키지 않는다"
    "이분산이 있으면 회귀계수 추정값이 편향된다"는 서술을 흔히 보는데 **틀렸다**. Gauss-Markov 정리의 불편성 부분은 등분산을 요구하지 않는다. $E[\varepsilon \mid X] = 0$이면 이분산이 있어도 OLS는 불편이고 일치성을 갖는다.

    이분산이 망가뜨리는 것은 **표준오차**이다. OLS 표준오차 공식 $s^2(\mathbf{X}'\mathbf{X})^{-1}$이 등분산을 가정하므로, 이분산 아래에서 이 값이 편향된다. 그 결과 $t$ 통계량, $p$값, 신뢰구간이 모두 틀린다.

    실무적 함의가 다르다. 계수가 편향된다면 결과를 폐기해야 하지만, 표준오차만 문제라면 **로버스트 표준오차로 고칠 수 있다.** 계수 추정값은 그대로 쓴다.

### 등분산성과 이분산

**등분산성:** 잔차의 분산이 독립변수의 모든 수준에서 일정하다.

$$
\text{Var}(\epsilon_i) = \sigma^2 \quad \text{(모든 } i \text{에 대해)}
$$

**이분산:** 잔차의 분산이 독립변수에 따라 달라진다.

$$
\text{Var}(\epsilon_i) = f(x_i)
$$

이분산은 회귀계수 추정의 비효율성과 잘못된 $p$값으로 이어져 가설검정에 영향을 준다. 따라서 회귀분석을 진행하기 전에 이분산을 검정하는 것이 중요하다.

### 등분산성 검정

등분산성을 확인하는 검정으로 **Breusch-Pagan 검정**과 **White 검정**이 있다. 이 검정들은 잔차와 설명변수의 관계를 조사한다.

### Breusch-Pagan 검정

Breusch-Pagan 검정은 잔차의 분산이 모형의 독립변수와 관련되어 있는지 검정하여 이분산을 탐지한다.

**가설:**

- $H_0$: 잔차가 등분산이다. 곧 분산이 일정하다: $\text{Var}(\epsilon_i) = \sigma^2$
- $H_1$: 잔차분산이 독립변수의 함수이다: $\text{Var}(\epsilon_i) = f(x_i)$

**Python 구현:**

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
import pandas as pd

# Sample data
data = pd.DataFrame({
    'X': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Y': [2, 4, 6, 8, 10, 9, 15, 16, 18, 20]
})

# Fit regression model
model = smf.ols('Y ~ X', data=data).fit()

# Perform Breusch-Pagan test
lm_stat, lm_p, f_stat, f_p = het_breuschpagan(model.resid, model.model.exog)

print(f"Breusch-Pagan LM = {lm_stat:.4f}, p = {lm_p:.4f}")
print(f"Breusch-Pagan F  = {f_stat:.4f}, p = {f_p:.4f}")
```

출력:

```text
Breusch-Pagan LM = 0.0803, p = 0.7769
Breusch-Pagan F  = 0.0648, p = 0.8055
```

**해석:**

- $p$값이 0.05보다 작으면 귀무가설을 기각하고 이분산이 존재한다고 결론짓는다.
- $p$값이 0.05보다 크면 귀무가설을 기각하지 못하고 잔차가 등분산과 일관된다고 본다.

여기서는 $p = 0.777$로 기각하지 못한다. 다만 $n = 10$으로 매우 작아 검정력이 사실상 없으므로, **"등분산성이 확인되었다"가 아니라 "이 자료로는 판정할 수 없다"**가 옳은 결론이다.

### 이분산의 해결책

이분산이 탐지되면 다음 해결책이 가능하다.

**1. 종속변수의 변환:** 로그나 제곱근 변환이 분산을 안정시킬 수 있다.

$$
Y' = \log(Y) \quad \text{또는} \quad Y' = \sqrt{Y}
$$

**2. 가중최소제곱(WLS):** 추정된 분산에 기반해 관측값에 가중치를 부여하여, 분산이 작은 관측값에 더 큰 비중을 준다.

$$
\hat{\beta} = (X^T W X)^{-1} X^T W Y
$$

**3. 로버스트 표준오차:** 회귀계수를 바꾸지 않으면서 이분산을 반영하는 로버스트 표준오차를 쓴다. 셋 중 가장 간단하고 가장 널리 쓰인다.

---

## 분산분석에서의 분산 검정

분산분석은 셋 이상 집단의 평균을 비교하여 유의한 차이가 있는지 판정한다. 핵심 가정은 집단들의 분산이 같다는 것(**분산의 동질성**)이다. 이 가정이 위배되면 분산분석 결과가 오도할 수 있다.

### 분산분석에서의 분산 동질성

**가설:**

- $H_0$: 집단들의 분산이 같다: $\sigma_1^2 = \sigma_2^2 = \dots = \sigma_k^2$
- $H_1$: 적어도 한 집단의 분산이 다르다: 적어도 한 쌍의 $i \neq j$에 대해 $\sigma_i^2 \neq \sigma_j^2$

### 분산분석에서의 Levene 검정

Levene 검정은 분산분석의 등분산 가정을 확인하는 데 자주 쓰인다. 분산이 같지 않으면(분산의 이질성) Welch 분산분석 같은 대안을 써야 한다.

**Python 구현:**

```python
import numpy as np
from scipy.stats import levene

# Group data (note: all three have the SAME variance)
group1 = [10, 12, 14, 16, 18]
group2 = [22, 24, 26, 28, 30]
group3 = [32, 34, 36, 38, 40]

print("variances:", [np.var(g, ddof=1) for g in (group1, group2, group3)])

# Perform Levene's test
test_stat, p_value = levene(group1, group2, group3)
print(f"Levene's test statistic: {test_stat}")
print(f"P-value: {p_value}")
```

출력:

```text
variances: [10.0, 10.0, 10.0]
Levene's test statistic: 0.0
P-value: 1.0
```

!!! note "이 예제 자료는 퇴화되어 있다"
    세 집단이 모두 등차수열 $\{a, a+2, a+4, a+6, a+8\}$의 형태이므로 **표본분산이 정확히 10으로 동일**하다. 중앙값으로부터의 절대편차도 세 집단 모두 $\{4, 2, 0, 2, 4\}$로 같다.

    그래서 Levene 통계량이 **정확히 0**, $p$값이 **정확히 1**이 된다. 검정을 시연하는 자료로는 적절하지 않다. 분산 차이가 있는 자료를 쓰려면 예컨대 `group3 = [26, 31, 36, 41, 46]`처럼 간격을 바꾸면 된다.

**해석:**

- $p$값이 0.05보다 작으면 귀무가설을 기각하고 분산이 같지 않다고 결론짓는다.
- $p$값이 0.05보다 크면 귀무가설을 기각하지 못한다. 이는 "분산이 같다"의 증명이 아니라 "다르다는 증거가 없다"는 뜻이다.

### 분산 이질성의 해결책

Levene 검정이 이분산을 시사하면 집단 간 등분산을 가정하지 않는 **Welch 분산분석**을 적용할 수 있다.

!!! danger "`anova_lm(..., robust='hc3')`은 Welch 분산분석이 아니다"
    다음 코드가 Welch 분산분석으로 소개되는 경우가 있으나 **틀렸다**.

    ```python
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    # 집단마다 분산이 다른 자료 (표준편차 1, 2, 4)
    rng = np.random.default_rng(0)
    data = pd.DataFrame({
        "Score": np.concatenate([rng.normal(10, 1, 20),
                                 rng.normal(12, 2, 20),
                                 rng.normal(14, 4, 20)]),
        "Group": np.repeat(["A", "B", "C"], 20),
    })

    # NOT Welch's ANOVA
    model = sm.formula.ols('Score ~ Group', data=data).fit()
    print(sm.stats.anova_lm(model, typ=2, robust='hc3'))
    ```

    출력:

    ```
                  sum_sq    df          F        PR(>F)
    Group     424.427668   2.0  30.736759  8.794817e-10
    Residual  393.541447  57.0        NaN           NaN
    ```

    표가 나오기는 하지만 이것은 Welch 분산분석이 아니다. 분모 자유도가 $57 = N - k$ 그대로이고, Welch라면 등분산이 깨진 만큼 자유도가 줄어들어야 한다.

    이것은 OLS 적합에 이분산 일치 공분산 행렬(HC3)을 적용한 **Wald 형태의 분산분석표**이다. Welch 분산분석과 다음 점에서 다르다.

    | | Welch 분산분석 | `anova_lm(robust='hc3')` |
    |---|---|---|
    | 집단평균의 가중 | 정밀도 $n_i/s_i^2$로 가중 | 가중하지 않음(OLS) |
    | 분모 자유도 | Welch-Satterthwaite 근사 | $N - k$ 그대로 |
    | 소표본 성질 | 잘 연구되어 있음 | 근사가 거칠 수 있음 |

    Welch 분산분석은 SciPy에 없으므로 직접 구현하거나(15.7절 [분산분석 사전검정](anova_pretest.md) 페이지 참조) `pingouin.welch_anova`를 쓴다.

**Welch 분산분석의 Python 구현:**

```python
import numpy as np
from scipy import stats

def welch_anova(groups):
    """Welch's one-way ANOVA. Returns (F, df1, df2, p)."""
    k = len(groups)
    n = np.array([len(g) for g in groups])
    m = np.array([np.mean(g) for g in groups])
    v = np.array([np.var(g, ddof=1) for g in groups])
    w = n / v
    m_w = np.sum(w * m) / np.sum(w)
    A = np.sum(w * (m - m_w) ** 2) / (k - 1)
    lam = np.sum((1 - w / np.sum(w)) ** 2 / (n - 1)) / (k ** 2 - 1)
    F = A / (1 + 2 * (k - 2) * lam)
    df2 = 1 / (3 * lam)
    return F, k - 1, df2, stats.f.sf(F, k - 1, df2)

groups = [[10, 12, 14], [22, 24, 26], [32, 34, 36]]
F, df1, df2, p = welch_anova(groups)
print(f"Welch ANOVA: F = {F:.4f}, df = ({df1}, {df2:.2f}), p = {p:.6f}")

# Classical one-way ANOVA for comparison
print(f"Classical ANOVA: {stats.f_oneway(*groups)}")
```

출력:

```
Welch ANOVA: F = 78.0000, df = (2, 4.00), p = 0.000625
Classical ANOVA: F_onewayResult(statistic=91.0, pvalue=3.2507247912312294e-05)
```

이 자료에서는 세 집단의 표본분산이 모두 4로 같아 Welch $F = 78.0$, 고전적 $F = 91.0$이다. 등분산일 때도 두 값이 다른 것은 Welch가 분모 자유도를 $4.00$으로 줄이기 때문이다. 표본이 작을수록 이 보정의 대가가 크다.

**해석:**

Welch 분산분석은 집단 간 등분산을 가정하지 않고 집단평균을 비교하는 F 통계량과 $p$값을 제공한다. $p$값이 0.05보다 작으면 집단평균에 유의한 차이가 있다고 결론짓는다.

---

## 흐름: 회귀와 분산분석에서의 분산 검정

세 처치집단의 평균을 분산분석으로 비교하고 자료에 회귀분석을 수행하는 상황을 생각하자.

1. **분산분석:** 처치집단에 Levene 검정(또는 Brown-Forsythe 검정)을 수행한다. 분산이 같으면 표준 분산분석으로, 아니면 Welch 분산분석으로 진행한다.
2. **회귀:** 회귀모형을 적합한 뒤 Breusch-Pagan 검정으로 이분산을 확인한다. 이분산이 있으면 로버스트 표준오차를 적용하거나 종속변수를 변환한다.

이 단계들이 모형의 가정을 충족시켜 더 정확하고 신뢰할 만한 통계적 추론으로 이어진다.

!!! tip "더 단순한 대안"
    위 흐름은 두 단계 절차의 문제(15.7절 [분산분석 사전검정](anova_pretest.md) 참조)를 안고 있다. 실무에서는 다음이 더 간단하고 안전하다.

    1. **분산분석:** 사전검정 없이 처음부터 Welch 분산분석을 쓴다.
    2. **회귀:** 사전검정 없이 처음부터 로버스트 표준오차(HC3)를 쓴다.

    두 경우 모두 가정이 성립할 때 잃는 것이 적고(검정력 2~3%p), 가정이 깨졌을 때 얻는 것이 크다. 검정 결과에 따라 방법을 바꾸는 자료 의존적 절차 자체를 없애는 것이 핵심이다.


## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
"이분산이 있으면 회귀계수가 편향된다"는 서술이 왜 틀렸는지 설명하고, 실제로 무엇이 편향되는지 밝혀라.

</div>

??? success "풀이"
    **OLS 추정량의 불편성.** $\hat{\boldsymbol{\beta}} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}$이고 $\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$이므로

    $$
    \hat{\boldsymbol{\beta}} = \boldsymbol{\beta} + (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\boldsymbol{\varepsilon}.
    $$

    $E[\boldsymbol{\varepsilon} \mid \mathbf{X}] = \mathbf{0}$이면

    $$
    E[\hat{\boldsymbol{\beta}} \mid \mathbf{X}] = \boldsymbol{\beta} + (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'E[\boldsymbol{\varepsilon}\mid\mathbf{X}] = \boldsymbol{\beta}.
    $$

    이 유도 어디에도 $\operatorname{Var}(\varepsilon_i)$이 등장하지 않는다. **불편성은 오차의 분산 구조와 무관하다.**

    **편향되는 것은 분산 추정량이다.** 참 공분산행렬은

    $$
    \operatorname{Var}(\hat{\boldsymbol{\beta}}) = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\boldsymbol{\Omega}\mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}, \quad \boldsymbol{\Omega} = \operatorname{diag}(\sigma_1^2, \ldots, \sigma_n^2)
    $$

    인데 OLS는 $\boldsymbol{\Omega} = \sigma^2 \mathbf{I}$를 가정하여

    $$
    \widehat{\operatorname{Var}}_{\text{OLS}}(\hat{\boldsymbol{\beta}}) = s^2(\mathbf{X}'\mathbf{X})^{-1}
    $$

    을 쓴다. $\boldsymbol{\Omega} \neq \sigma^2\mathbf{I}$이면 이 추정값이 참값과 다르며, 15.7절 [회귀에서의 분산 검정](regression_variance.md) 연습문제 2에서 보았듯 **어느 방향으로든** 틀릴 수 있다.

    **추가로 잃는 것: 효율성.** OLS는 여전히 불편이지만 최소분산은 아니다. 참 $\boldsymbol{\Omega}$를 알면 GLS가 더 작은 분산을 갖는다. 다만 실무에서 $\boldsymbol{\Omega}$를 모르는 경우가 많으므로, 효율성 손실을 감수하고 OLS + 로버스트 표준오차를 쓰는 것이 표준 관행이다.

    **왜 이 구분이 중요한가.** 계수가 편향된다고 믿으면 결과를 폐기하거나 모형을 다시 설정해야 한다고 생각하게 된다. 실제로는 **표준오차 계산 방식만 바꾸면 된다.** `cov_type='HC3'` 한 줄이면 해결된다. $\square$

<div class="drillbox" markdown>

**연습문제 2.**
본문의 Levene 예제 자료가 왜 퇴화되어 있는지 보이고, 분산 차이가 있는 자료로 바꾸어 검정을 다시 수행하라.

</div>

??? success "풀이"
    **퇴화의 이유.** 세 집단이 모두 공차 2인 등차수열이다.

    $$
    g_1 = \{10,12,14,16,18\}, \quad g_2 = \{22,24,26,28,30\}, \quad g_3 = \{32,34,36,38,40\}.
    $$

    등차수열의 분산은 공차와 항의 개수에만 의존하고 시작값과 무관하다. 공차 $d$, 항 $n = 5$이면

    $$
    s^2 = \frac{d^2 n(n+1)}{12} = \frac{4 \times 5 \times 6}{12} = 10.
    $$

    세 집단 모두 $s^2 = 10$이다.

    더 나아가 중앙값으로부터의 절대편차도 세 집단 모두 $\{4, 2, 0, 2, 4\}$로 **완전히 동일**하다. Levene 검정은 이 편차들의 집단평균을 비교하므로, 집단간 제곱합이 정확히 0이 되어 $W = 0$, $p = 1$이 나온다.

    **분산 차이를 만든 자료.**

    ```python
    import numpy as np
    from scipy.stats import levene, bartlett, fligner

    g1 = [10, 12, 14, 16, 18]      # d = 2, var = 10
    g2 = [22, 25, 28, 31, 34]      # d = 3, var = 22.5
    g3 = [32, 37, 42, 47, 52]      # d = 5, var = 62.5

    print("variances:", [round(np.var(g, ddof=1), 2) for g in (g1, g2, g3)])
    # 이름있는 튜플을 그대로 찍으면 유효숫자가 너무 많다. 자리수를 맞춰 출력한다.
    for name, res in [("Levene (median)", levene(g1, g2, g3)),
                      ("Levene (mean)  ", levene(g1, g2, g3, center='mean')),
                      ("Bartlett       ", bartlett(g1, g2, g3)),
                      ("Fligner-Killeen", fligner(g1, g2, g3))]:
        print(f"{name}: stat = {res.statistic:.4f}, p = {res.pvalue:.4f}")
    ```

    출력:

    ```text
    variances: [10.0, 22.5, 62.5]
    Levene (median): stat = 1.8947, p = 0.1927
    Levene (mean)  : stat = 1.8947, p = 0.1927
    Bartlett       : stat = 2.9323, p = 0.2308
    Fligner-Killeen: stat = 3.5932, p = 0.1659
    ```

    이제 검정통계량이 0이 아니고 $p$값도 1이 아니다. 다만 **어느 검정도 기각하지 못한다**($p$값이 0.17~0.23).

    분산비가 $62.5/10 = 6.25$배로 상당한데도 그렇다. 각 집단 $n = 5$, 총 15개 관측값으로는 검정력이 매우 낮기 때문이다. 15.4절 연습문제 4에서 총 30개로도 4배 차이를 겨우 탐지했음을 떠올리면 당연한 결과이다.

    (평균 중심과 중앙값 중심 Levene의 결과가 완전히 같다. 다섯 개 등차수열에서는 평균과 중앙값이 일치하기 때문이다.)

    **교육적 함의.** 예제 자료를 만들 때는 (1) 보이려는 현상이 실제로 존재하는지, (2) 그것을 탐지할 만한 표본크기인지 확인해야 한다. 등차수열처럼 규칙적인 자료는 의도치 않은 퇴화를 낳기 쉽다. $\square$

<div class="drillbox" markdown>

**연습문제 3.**
`sm.stats.anova_lm(model, typ=2, robust='hc3')`이 Welch 분산분석과 어떻게 다른지 구체적으로 설명하고, 두 결과를 비교하라.

</div>

??? success "풀이"
    **본질적 차이.** 두 절차 모두 이분산에 대응하지만 방식이 다르다.

    - **`anova_lm(robust='hc3')`:** OLS로 적합한 뒤, 계수의 공분산행렬만 HC3 샌드위치 추정량으로 바꾸어 Wald 검정을 수행한다. 집단평균의 **추정 방식은 그대로**이고, 분모 자유도도 $N - k$를 그대로 쓴다.
    - **Welch 분산분석:** 집단평균을 정밀도 $w_i = n_i/s_i^2$로 **가중**하여 결합하고, Welch-Satterthwaite 근사로 분모 자유도를 줄인다.

    ```python
    import numpy as np, pandas as pd
    import statsmodels.api as sm
    from scipy import stats

    data = pd.DataFrame({'Group': list('AAABBBCCC'),
                         'Score': [10, 12, 14, 22, 24, 26, 32, 34, 36]})
    m = sm.formula.ols('Score ~ Group', data=data).fit()
    print(sm.stats.anova_lm(m, typ=2, robust='hc3'))

    groups = [[10, 12, 14], [22, 24, 26], [32, 34, 36]]
    print(stats.f_oneway(*groups))
    ```

    출력:

    ```text
                  sum_sq   df          F    PR(>F)
    Group     485.333333  2.0  60.666667  0.000105
    Residual   24.000000  6.0        NaN       NaN
    F_onewayResult(statistic=91.0, pvalue=3.2507247912312294e-05)
    ```

    고전 분산분석의 $F = 91.0$이고 HC3 판은 $F = 60.67$이다. HC3 쪽이 더 보수적이다.

    **이 자료에서는 세 집단의 분산이 정확히 같다**($\{10,12,14\}$의 분산 = $\{22,24,26\}$의 분산 = $\{32,34,36\}$의 분산 = 4). 그러므로 Welch 분산분석은 고전 분산분석과 사실상 같은 결과를 낼 것이고, HC3 판만 다르게 나온다.

    **왜 HC3가 더 보수적인가.** HC3는 잔차를 $(1-h_{ii})$로 나누어 지렛대 보정을 한다. $n = 9$, 모수 3개로 자유도가 매우 적으므로 $h_{ii}$가 커서($1/3$) 보정이 크게 작용한다. 작은 표본에서 HC3는 알려진 대로 보수적이다.

    **결론.** 세 절차가 모두 다른 답을 준다. 이름이 "robust"라고 해서 Welch와 같은 것이 아니며, 무엇을 쓰는지 명확히 하고 그 소표본 성질을 알아야 한다. $n = 9$처럼 극단적으로 작은 자료에서는 어느 것도 신뢰하기 어렵고, 순열검정이 더 나은 선택이다. $\square$

<div class="drillbox" markdown>

**연습문제 4.**
본문의 두 단계 흐름 대신 "처음부터 Welch + HC3"를 쓰는 대안이 제시되었다. 각 방식의 장단점을 정리하고 언제 어느 쪽을 택할지 논하라.

</div>

??? success "풀이"

    | | 두 단계 (사전검정 후 선택) | 처음부터 로버스트 |
    |---|---|---|
    | 제1종 오류 조절 | 왜곡 가능(15.7절 표: 0.057) | 안정적(0.044) |
    | 검정력 (가정 성립 시) | 최대 | 2~3%p 손실 |
    | 자료 의존적 선택 | 있음 | 없음 |
    | 사전등록 가능성 | 어려움 | 쉬움 |
    | 보고의 투명성 | 낮음(선택 과정 서술 필요) | 높음 |
    | 계산 복잡도 | 검정 두 번 | 한 번 |

    **처음부터 로버스트를 택할 상황(대부분).**

    - 탐색적 분석이 아닌 확증적 분석
    - 사전등록된 연구
    - 가정의 성립 여부가 불확실한 관찰자료
    - 표본이 작아 사전검정의 검정력이 낮은 경우

    **두 단계를 택할 만한 상황(드묾).**

    - 표본이 매우 커서 사전검정의 검정력이 충분한 경우
    - 등분산성 자체가 과학적 관심사인 경우(이때는 사전검정이 아니라 **보고할 결과**이다)
    - 물리적·이론적 근거로 등분산을 강하게 기대할 수 있고 최대 검정력이 필요한 경우

    **핵심 원칙.** 통계적 절차의 선택은 **자료를 보기 전에** 정해야 한다. 자료를 보고 방법을 고르면, 아무리 각 방법이 개별적으로 타당해도 결합된 절차의 오류율이 통제되지 않는다.

    "처음부터 로버스트"의 진짜 장점은 검정력이나 크기 수치가 아니라 **이 선택 문제를 아예 없앤다는 것**이다. 자료를 보고 결정할 일이 없으면 자료 의존적 편향도 없다. $\square$
