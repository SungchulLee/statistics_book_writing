# 계수 검정 보기

## 개요

이 페이지는 statsmodels로 회귀계수의 가설검정을 수행하는 방법을 보인다. 점점 복잡해지는 세 보기를 통해 OLS 출력에서 $t$ 통계량, $p$값, 신뢰구간을 뽑아내는 법과, 설명변수가 하나일 때와 여럿일 때의 통계적 유의성을 해석하는 법을 다룬다.

---

## 1. 수학적 배경

선형모형 $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$에서 $j$번째 계수에 대한 가설검정은

$$
H_0\colon \beta_j = 0 \quad \text{대} \quad H_1\colon \beta_j \neq 0.
$$

$H_0$ 아래에서 검정통계량은 $t$ 분포를 따른다.

$$
t_j = \frac{\hat{\beta}_j}{\mathrm{SE}(\hat{\beta}_j)} \sim t_{n-k} \quad (H_0 \text{ 아래에서}).
$$

양측 $p$값은

$$
p = 2\,P(T_{n-k} > |t_j|).
$$

$p < \alpha$이면, 동등하게 $|t_j| > t^*_{n-k,\,\alpha/2}$이면, 동등하게 $\beta_j$의 $(1-\alpha)$ 수준 신뢰구간이 0을 포함하지 않으면 유의수준 $\alpha$에서 $H_0$을 기각한다.

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> 설명변수 두 개.

</div>

??? success "풀이"
    ```python
    import numpy as np
    import statsmodels.api as sm

    def print_summary(res):
        """summary()에서 실행 날짜와 시각만 지우고 인쇄한다(재현 가능한 출력을 위해)."""
        lines = []
        for line in str(res.summary()).split("\n"):
            if line.startswith(("Date:", "Time:")):
                lines.append(line[:19].ljust(38) + line[38:])
            else:
                lines.append(line)
        print("\n".join(lines))

    np.random.seed(0)
    X = np.random.rand(100, 2)
    y = 3 * X[:, 0] + 5 * X[:, 1] + np.random.randn(100)

    X_const = sm.add_constant(X)
    results = sm.OLS(y, X_const).fit()
    print_summary(results)
    ```

    출력:

    ```
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.743
    Model:                            OLS   Adj. R-squared:                  0.738
    Method:                 Least Squares   F-statistic:                     140.3
    Date:                                   Prob (F-statistic):           2.39e-29
    Time:                                   Log-Likelihood:                -133.53
    No. Observations:                 100   AIC:                             273.1
    Df Residuals:                      97   BIC:                             280.9
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.1145      0.259      0.442      0.659      -0.400       0.629
    x1             2.5737      0.332      7.742      0.000       1.914       3.233
    x2             5.0296      0.328     15.346      0.000       4.379       5.680
    ==============================================================================
    Omnibus:                        0.410   Durbin-Watson:                   2.045
    Prob(Omnibus):                  0.815   Jarque-Bera (JB):                0.502
    Skew:                           0.144   Prob(JB):                        0.778
    Kurtosis:                       2.807   Cond. No.                         5.58
    ==============================================================================

    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ```

요약표의 `P>|t|` 열이 각 계수에 대한 $H_0\colon \beta_j = 0$의 양측 p-값이다. 그 옆의 `[0.025 0.975]`가 95% 신뢰구간이고, 둘은 같은 정보를 다르게 표현한 것이다.

<div class="exbox" markdown>

**보기 2.** <span class="diff easy" title="쉬움"></span> p값과 신뢰구간 뽑아내기.

</div>

??? success "풀이"
    ```python
    np.random.seed(42)
    X = np.random.rand(100, 1)
    y = 2.5 * X[:, 0] + np.random.randn(100)

    X_const = sm.add_constant(X)
    results = sm.OLS(y, X_const).fit()

    p_values = results.pvalues
    confidence_intervals = results.conf_int()

    print("P-values:", p_values)
    print("95% CI:\n", confidence_intervals)
    ```

    출력:

    ```
    P-values: [2.09535133e-01 1.62497167e-09]
    95% CI:
     [[-0.12283207  0.55302438]
     [ 1.43199824  2.6484553 ]]
    ```

`pvalues`와 `conf_int()`로 요약표의 값을 배열로 꺼낸다. 절편의 p-값은 0.21로 유의하지 않고 기울기는 $1.6 \times 10^{-9}$로 강하게 유의하다.

<div class="exbox" markdown>

**보기 3.** <span class="diff easy" title="쉬움"></span> 여러 설명변수의 해석.

</div>

??? success "풀이"
    ```python
    np.random.seed(42)
    study_hours = np.random.rand(100) * 10
    sleep_hours = np.random.rand(100) * 8
    exam_scores = 5 + 2.5 * study_hours - 1.5 * sleep_hours + np.random.randn(100) * 2

    X = np.column_stack((study_hours, sleep_hours))
    X_const = sm.add_constant(X)
    results = sm.OLS(exam_scores, X_const).fit()

    conf = results.conf_int()   # ndarray when the input is a NumPy array
    for i, name in enumerate(["Intercept", "Study Hours", "Sleep Hours"]):
        pval = results.pvalues[i]
        ci = conf[i]
        status = "Significant" if pval < 0.05 else "Not Significant"
        print(f"{name}: coef={results.params[i]:.4f}, "
              f"p={pval:.4g} ({status}), "
              f"95% CI=({ci[0]:.4f}, {ci[1]:.4f})")
    ```

    출력:

    ```text
    Intercept: coef=4.8212, p=1.754e-15 (Significant), 95% CI=(3.8122, 5.8303)
    Study Hours: coef=2.4317, p=2.171e-58 (Significant), 95% CI=(2.2992, 2.5641)
    Sleep Hours: coef=-1.3202, p=3.633e-28 (Significant), 95% CI=(-1.4883, -1.1521)
    ```

!!! warning "`conf_int()`의 반환 형식은 입력에 따라 달라진다"
    `sm.OLS`에 NumPy 배열을 넘기면 `conf_int()`가 `ndarray`를 돌려주므로 `conf[i]`로 색인해야 한다. pandas `Series`/`DataFrame`을 넘겼을 때에만 `DataFrame`이 반환되어 `.iloc[i]`를 쓸 수 있다. 배열 입력에 `.iloc`를 쓰면 `AttributeError`가 난다.

---

## 2. 해석

![잡음 크기만 바꾸었을 때 계수 그림이 어떻게 달라지는가](./img/coefficient_plot.png)

보기 3의 결과를 표 대신 **계수 그림**으로 옮겨 보았다. 가로축이 계수의 값, 가로 막대가 95% 신뢰구간, 가운데 점이 추정값, 초록 마름모가 참값 $(5,\ 2.5,\ -1.5)$이다. 왼쪽 패널이 본문과 같은 잡음 $\sigma = 2$의 결과이고, 구간이 아주 짧아 점처럼 보인다. 공부 시간은 $(2.299,\ 2.564)$, 수면 시간은 $(-1.488,\ -1.152)$로 둘 다 참값을 담고 $0$에서 멀찍이 떨어져 있다. 회귀표를 수십 줄 읽는 것보다 이 한 장이 빠르다.

오른쪽은 **같은 설명변수, 같은 참 계수**에 잡음만 $\sigma = 16$으로 여덟 배 키운 결과다. 참값은 조금도 변하지 않았다. 그런데 결론이 크게 달라진다. 공부 시간은 추정값 $1.953$, 구간 $(0.893,\ 3.013)$, $p = 0.0004$로 여전히 유의하다. 수면 시간은 추정값 $-0.061$, 구간 $(-1.406,\ 1.283)$, $p = 0.93$으로 유의성을 잃었다. 절편도 $p = 0.38$로 함께 떨어졌다. 표준오차가 $0.085$에서 $0.678$로 여덟 배 커지면서 구간이 참값과 $0$을 한꺼번에 삼켜 버린 것이다.

이 대비가 마지막 해설의 요지다. 오른쪽 패널에서 수면 시간이 "유의하지 않다"는 판정을 받았지만, 참 효과는 여전히 $-1.5$다. 붉은 구간 $(-1.406,\ 1.283)$은 "효과가 $0$이다"라고 말하는 것이 아니라 **"$-1.4$일 수도 $+1.3$일 수도 있어 모르겠다"**고 말하고 있다. 초록 마름모가 구간 안에 있다는 사실이 그 점을 확인해 준다. $p$값만 보고했다면 이 구별이 통째로 사라졌을 것이다. 계수 그림이 회귀 결과를 보여 주는 표준 도구인 까닭이 여기에 있다. 유의성과 효과 크기와 정밀도를 한 장에 동시에 담기 때문이다.

- 보기 1에서는 두 설명변수($x_1$과 $x_2$)의 참 계수가 모두 0이 아니므로(3과 5) 두 $p$값이 모두 작고 둘 다 유의하다고 판정될 것으로 기대한다.
- 보기 2에서는 설명변수 하나의 참 계수가 2.5이고 잡음 표준편차가 1이다. 기울기의 $p$값이 매우 작아 선형관계를 확인해 줄 것이다.
- 보기 3에서는 "Study Hours"가 시험 점수에 양의 효과(계수 $\approx 2.5$)를, "Sleep Hours"가 음의 효과(계수 $\approx -1.5$)를 갖는다. 둘 다 통계적으로 유의해야 한다. 신뢰구간이 효과 크기의 그럴듯한 범위를 준다. 추정값이 참값과 조금 다른 것은($2.43$ 대 $2.5$, $-1.32$ 대 $-1.5$) 표집변동 때문이며, 참값이 신뢰구간 안에 들어 있는지 확인해 보면 좋다.
- 어떤 설명변수가 "유의하지 않다"($p > 0.05$)는 것이 효과가 0임을 증명하지는 않는다. 선택한 유의수준에서 $H_0$을 기각할 증거가 부족하다는 뜻일 뿐이다. 신뢰구간이 같은 정보를 더 유익하게 전달한다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> 보기 1에서 계수와 그 표준오차로부터 $\hat{\beta}_1$의 $t$ 통계량을 직접 계산하라. `results.tvalues[1]`의 값과 일치하는지 확인하라.

</div>

??? success "풀이"

    ```python
    t_manual = results.params[1] / results.bse[1]
    t_auto = results.tvalues[1]
    print(f"Manual t: {t_manual:.4f}")
    print(f"Auto t:   {t_auto:.4f}")
    print(f"Match: {np.isclose(t_manual, t_auto)}")
    ```

    출력:

    ```
    Manual t: 36.4271
    Auto t:   36.4271
    Match: True
    ```

    손으로 계산한 $t = \hat\beta/\text{SE}$가 statsmodels의 값과 정확히 같다.

    요약표가 $t_j = \hat{\beta}_j / \mathrm{SE}(\hat{\beta}_j)$를 그대로 계산하는 것이므로 두 값은 동일하다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 보기 2에서 잡음 표준편차를 1에서 5로 키우도록 고쳐라. 기울기의 $p$값과 신뢰구간은 어떻게 달라지는가?

</div>

??? success "풀이"

    ```python
    y_noisy = 2.5 * X[:, 0] + 5 * np.random.randn(100)
    results_noisy = sm.OLS(y_noisy, X_const).fit()
    print_summary(results_noisy)
    ```

    출력:

    ```
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.702
    Model:                            OLS   Adj. R-squared:                  0.696
    Method:                 Least Squares   F-statistic:                     114.3
    Date:                                   Prob (F-statistic):           3.10e-26
    Time:                                   Log-Likelihood:                -295.07
    No. Observations:                 100   AIC:                             596.1
    Df Residuals:                      97   BIC:                             603.9
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.0424      1.209      0.035      0.972      -2.358       2.443
    x1             2.4007      0.159     15.120      0.000       2.086       2.716
    x2             0.1402      0.201      0.696      0.488      -0.260       0.540
    ==============================================================================
    Omnibus:                        3.059   Durbin-Watson:                   2.164
    Prob(Omnibus):                  0.217   Jarque-Bera (JB):                2.645
    Skew:                          -0.207   Prob(JB):                        0.266
    Kurtosis:                       3.681   Cond. No.                         17.6
    ==============================================================================

    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ```

    잡음을 키운 자료에서는 같은 계수가 유의하지 않게 나온다. 계수의 유의성은 계수의 크기가 아니라 **계수와 표준오차의 비**로 정해지기 때문이다.

    잡음이 커지면 잔차 표준오차 $s$가 커져 $\mathrm{SE}(\hat{\beta}_1)$이 부풀려진다. $t$ 통계량이 작아지고 $p$값이 커지며 신뢰구간이 넓어진다. 잡음이 충분히 크면 참 효과가 0이 아닌데도 기울기가 더 이상 통계적으로 유의하지 않을 수 있다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> 보기 3에 시험 점수와 무관한 세 번째 설명변수 `caffeine = np.random.rand(100) * 5`를 추가하라. 그 $p$값에 대해 무엇을 기대하며 그 이유는 무엇인가?

</div>

??? success "풀이"

    caffeine이 시험 점수와 독립적으로 생성되었으므로 참 계수는 0이다. OLS 추정값 $\hat{\beta}_{\text{caffeine}}$은 0에 가깝고 $p$값은 클 것이다(대개 $> 0.05$). 다만 순전히 우연으로($\alpha = 0.05$ 수준에서 약 5%의 경우) $p$값이 0.05 아래로 떨어질 수 있는데, 그것이 제1종 오류이다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> 전체 모형 유의성의 $F$ 검정과 개별 $t$ 검정의 관계를 설명하라. 둘이 다른 결론을 줄 수 있는 경우는 언제인가?

</div>

??? success "풀이"

    $F$ 검정은 $H_0\colon \beta_1 = \beta_2 = \cdots = \beta_{k-1} = 0$(모든 기울기가 동시에 0)을 평가하고, 각 $t$ 검정은 계수 하나를 평가한다. 설명변수들이 무상관이면 개별 $t$ 검정들이 독립이고, 어느 $t$ 검정이 기각하면 사실상 $F$ 검정도 기각한다. 설명변수들이 상관되어 있으면 $F$ 검정은 기각하는데(모형 전체는 유용한데) 어느 개별 $t$ 검정도 기각하지 않는(나머지를 조정하고 나면 어느 설명변수도 유의하지 않은) 일이 가능하다. 다중공선성이 있을 때 일어나는 현상이다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> 설명변수가 하나인 단순선형회귀에서 $t_1^2 = F$임을 증명하라. 이 동등성이 깨지는 조건은 무엇인가?

</div>

??? success "풀이"

    단순회귀에서는 $k = 2$(절편 + 기울기)이고 $F$ 통계량은

    $$
    F = \frac{\mathrm{ESS}/1}{\mathrm{RSS}/(n-2)}.
    $$

    기울기의 $t$ 통계량은 $t_1 = \hat{\beta}_1 / \mathrm{SE}(\hat{\beta}_1)$이다. 대수적으로 $t_1^2 = \mathrm{ESS}/s^2 = F$임을 보일 수 있다. 이 동등성은 설명변수가 하나일 때(분자 자유도가 1일 때) 정확히 성립한다. 다중회귀에서는 $F$ 검정의 분자 자유도가 $k-1 > 1$이고 모든 기울기를 동시에 검정하므로 동등성이 깨진다. $\square$

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span> 연습문제 1 은 "보기 1 에서"라고 말하면서 `results.params[1] / results.bse[1]` 을 계산했는데, 그 출력은 보기 1 의 표에서 `x1` 행에 적힌 $t = 7.742$ 가 아니라 $36.4271$ 이다.

(a) 이 숫자가 실제로는 어느 모형의 어느 계수에 대한 $t$ 통계량인가. 왜 그렇게 되었는지 설명하라.

(b) (a) 의 답을 확인하라. 그 모형의 출력에 적힌 추정값과 95% 신뢰구간, 그리고 $t^*_{97,\,0.025} = 1.9847$ 만 써서 $36.4271$ 을 되살려 보라.

</div>

??? success "풀이"

    (a) 보기 1, 2, 3 의 코드 블록은 한 세션에서 차례로 돌아가며 블록마다 이름 `results` 를 다시 묶는다. 마지막으로 묶인 것은 보기 3 의 모형, 곧 시험 점수를 공부 시간과 수면 시간에 회귀시킨 적합이다. 따라서 `results.params[1] / results.bse[1]` 은 보기 1 의 `x1` 이 아니라 보기 3 의 **Study Hours** 에 대한 $t$ 통계량이다.

    같은 일이 연습문제 2 에서도 일어난다. 그 풀이는 `X[:, 0]` 과 `X_const` 를 그대로 쓰는데, 두 이름도 보기 3 에서 다시 묶여 있다. 그래서 설명변수가 하나뿐이었던 보기 2 를 고친다고 해 놓고도 출력에 `x1` 과 `x2` 두 행이 나온다. 출력 표에 `x2` 행이 있다는 것 자체가 그 증거다.

    (b) 보기 3 의 Study Hours 는 추정값 $\hat\beta = 2.4317$, 95% 신뢰구간 $(2.2992,\ 2.5641)$ 이다. 구간의 반폭이

    $$
    t^*_{97,\,0.025}\,\mathrm{SE}(\hat\beta) = \frac{2.5641 - 2.2992}{2} = 0.13245
    $$

    이므로

    $$
    \mathrm{SE}(\hat\beta) = \frac{0.13245}{1.9847} = 0.06674, \qquad
    t = \frac{2.4317}{0.06674} = 36.43.
    $$

    연습문제 1 의 출력 $36.4271$ 과 맞는다. 보기 1 의 `x1` 이었다면 같은 계산이 $2.5737 / 0.332 = 7.75$ 를 주어 표의 $7.742$ 와 맞았을 것이다.

    교훈은 두 가지다. 첫째, $t$ 통계량과 신뢰구간은 같은 정보를 담고 있어 어느 쪽에서든 다른 쪽을 복원할 수 있다. 둘째, 유의성을 읽을 때는 그 출력이 **어느 모형의** 출력인지를 먼저 확인해야 한다. 노트북에서 이름을 다시 묶는 습관은 숫자를 조용히 바꿔 놓는다. $\square$

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span> 해석 절은 보기 3 의 자료에 잡음만 $\sigma = 2$ 에서 $\sigma = 16$ 으로 키우면 공부 시간은 유의성을 지키고 수면 시간은 잃는다고 보고했다. 왜 하나만 떨어지는지를 숫자로 설명해 보자. 같은 설명변수 행렬에 잡음만 키우므로 표준오차는 $\sigma$ 에 비례해 $\mathrm{SE}(\sigma) = \mathrm{SE}(2) \cdot \sigma/2$ 로 커진다. 추정값 $\hat\beta_j$ 는 $\sigma = 2$ 에서의 값에 그대로 머문다고 가정하라.

(a) 계수 $j$ 가 유의수준 5% 에서 유의성을 잃기 시작하는 잡음 크기가

$$
\sigma^*_j = \frac{2\,\lvert \hat\beta_j \rvert}{(\text{95\% 신뢰구간의 반폭})}
$$

임을 보여라. 임계값 $t^*_{97,\,0.025}$ 가 이 식에서 사라지는 까닭도 밝혀라.

(b) 보기 3 의 출력으로 두 기울기의 $\sigma^*_j$ 를 구하고, $\sigma = 16$ 에서 수면 시간만 떨어진 것이 왜 우연이 아닌지 말하라.

(c) 오른쪽 패널의 수면 시간은 실제로 $p = 0.93$ 이었다. (b) 의 어림이 그 정도로 완전히 떨어진 것까지 설명하는가.

</div>

??? success "풀이"

    (a) 유의수준 5% 에서 유의하다는 것은 $\lvert \hat\beta_j \rvert / \mathrm{SE}(\sigma) > t^*_{97,\,0.025}$ 이다. $\mathrm{SE}(\sigma) = \mathrm{SE}(2)\,\sigma/2$ 를 넣으면

    $$
    \frac{\lvert \hat\beta_j \rvert}{\mathrm{SE}(2)\,\sigma/2} > t^*
    \iff \sigma < \frac{2\,\lvert \hat\beta_j \rvert}{t^*\,\mathrm{SE}(2)}.
    $$

    그런데 $\sigma = 2$ 에서의 신뢰구간 반폭이 바로 $t^*\,\mathrm{SE}(2)$ 다. 그래서 분모가 반폭으로 바뀌고 $t^*$ 는 따로 나타나지 않는다. 임계값이 사라진 것이 아니라 **반폭 안에 이미 들어 있다**. 같은 식을 $t$ 로 적으면 $\sigma^*_j = 2\lvert t_j \rvert / t^*$ 로, 문턱은 $\sigma = 2$ 에서의 $t$ 가 임계값보다 몇 배 컸는지를 그대로 잡음 배율로 옮긴 것이다.

    (b) 공부 시간: 반폭 $0.13245$, $\hat\beta = 2.4317$ 이므로

    $$
    \sigma^*_{\text{study}} = \frac{2 \times 2.4317}{0.13245} = 36.7.
    $$

    수면 시간: 반폭 $(-1.1521 + 1.4883)/2 = 0.16810$, $\lvert \hat\beta \rvert = 1.3202$ 이므로

    $$
    \sigma^*_{\text{sleep}} = \frac{2 \times 1.3202}{0.16810} = 15.7.
    $$

    $\sigma = 16$ 은 $15.7$ 을 가까스로 넘고 $36.7$ 에는 한참 못 미친다. 두 계수의 운명이 갈린 것은 잡음이 공부 시간보다 수면 시간을 더 괴롭혔기 때문이 아니라, 처음부터 수면 시간의 여유($\lvert t \rvert = 15.6$)가 공부 시간의 여유($\lvert t \rvert = 36.4$)의 절반이 못 되었기 때문이다. 잡음을 여덟 배로 키우면 두 $t$ 가 모두 여덟 분의 일로 줄어 $4.55$ 와 $1.95$ 가 된다. 앞의 것은 아직 $1.98$ 보다 크고 뒤의 것은 막 작아졌다.

    (c) 설명하지 못한다. $\sigma^*$ 는 추정값을 못 박아 둔 어림이고, 그 어림대로라면 $\sigma = 16$ 의 수면 시간은 문턱을 겨우 넘은 $p \approx 0.05$ 근처여야 한다. 실제 오른쪽 패널의 추정값은 $-1.320$ 이 아니라 $-0.061$ 로, 잡음을 키우면서 추정값까지 함께 흔들렸다. 표준오차만 여덟 배가 된 것이 아니라 추정오차도 여덟 배가 되어 참값 $-1.5$ 에서 거의 $0$ 까지 밀려난 것이다. 그래서 $p$ 가 $0.05$ 가 아니라 $0.93$ 이 되었다. 문턱 계산은 "어디쯤에서 위태로워지는가"를 알려 주는 데까지만 쓸 수 있고, 문턱 근처에서 한 번의 판정은 거의 동전 던지기다. $\square$

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span> 해석 절은 오른쪽 패널($\sigma = 16$)의 수면 시간을 두고, 붉은 구간 $(-1.406,\ 1.283)$ 이 효과가 $0$ 이라고 말하는 것이 아니라 "$-1.4$ 일 수도 $+1.3$ 일 수도 있어 모르겠다"고 말하고 있으며 "초록 마름모가 구간 안에 있다는 사실이 그 점을 확인해 준다"고 적었다. 수면 시간의 참 계수는 $-1.5$ 다.

(a) 적힌 숫자만으로 뒤 문장을 검사하라.

(b) 틀렸다면 무슨 일이 일어난 것인가. 같은 패널의 다른 두 구간 $(0.893,\ 3.013)$ 과 $(-4.503,\ 11.642)$ 는 각각 참값 $2.5$ 와 $5$ 를 어떻게 하고 있는가.

(c) 이 자리에서 그 단락의 요지인 "유의하지 않다는 것이 효과가 $0$ 이라는 뜻은 아니다" 는 그래도 살아남는가. 살아남는다면 어느 숫자가 그것을 받쳐 주는가.

</div>

??? success "풀이"

    (a) 구간의 아래끝이 $-1.406$ 이고 참값이 $-1.5$ 이므로 $-1.5 < -1.406$, 곧 참값은 구간 **밖**이다. 아래끝에서 $0.094$ 만큼 더 아래에 있다. 초록 마름모가 구간 안에 있다는 문장은 이 패널의 수면 시간에 대해서는 틀렸다.

    (b) 95% 신뢰구간이 참값을 놓친 것이다. 드문 일이 아니라 약속된 일이다. "95%" 는 같은 절차를 되풀이할 때 구간이 참값을 덮는 비율이 $0.95$ 라는 뜻이고, 나머지 $5\%$ 에서는 놓친다. 같은 패널의 다른 두 구간은 덮는다. 공부 시간은 $0.893 < 2.5 < 3.013$, 절편은 $-4.503 < 5 < 11.642$ 다. 세 구간 가운데 하나가 빗나간 표본을 만났을 뿐이다.

    (c) 살아남는다. 요지를 받치는 숫자는 마름모의 위치가 아니라 **구간의 폭**이다. $(-1.406,\ 1.283)$ 은 길이가 $2.689$ 로, $\sigma = 2$ 일 때의 $(-1.488,\ -1.152)$ 보다 여덟 배 넓고 부호마저 정하지 못하고 있다. $p = 0.93$ 은 "효과가 없다"가 아니라 "이 자료로는 $-1.4$ 와 $+1.3$ 사이의 어느 것도 배제하지 못한다"는 보고다. 참 효과 $-1.5$ 가 구간을 간신히 벗어났다는 사실은 이 결론을 흔들기는커녕 더 세게 만든다. 이 표본은 효과가 $0$ 이 아닐 뿐 아니라 구간이 담지 못할 만큼 크기까지 한데도 유의성 판정은 "모르겠다"로 나온 것이다.

    덧붙이면, 한 번의 그림에서 마름모가 구간 안에 있는지를 보고 안심하는 것은 쓸 수 있는 논법이 아니다. 실제 자료에서는 참값을 모르므로 마름모를 그릴 수조차 없다. 우리가 믿는 것은 이 한 구간이 아니라 구간을 만드는 **절차의 덮개율**이고, 그것은 연습문제 10 처럼 되풀이해서만 잴 수 있다. $\square$

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span> 연습문제 5 는 단순회귀에서 $t_1^2 = F$ 임을 보았다. 보기 1 은 기울기가 둘이고 $F = 140.3$, $t_1 = 7.742$, $t_2 = 15.346$ 이다. 자유도로 나눈 평균을 써 보면 $\tfrac{1}{2}(t_1^2 + t_2^2) = 147.72$ 로 $F$ 와 맞지 않는다. 보기 1 의 적합에서 두 기울기 추정값의 추정 상관은 $\widehat{\mathrm{Corr}}(\hat\beta_1, \hat\beta_2) = 0.0683$ 이다.

(a) 기울기가 둘인 모형에서 $H_0\colon \beta_1 = \beta_2 = 0$ 의 $F$ 통계량이 $r = \widehat{\mathrm{Corr}}(\hat\beta_1, \hat\beta_2)$ 로

$$
F = \frac{t_1^2 - 2r\,t_1 t_2 + t_2^2}{2\,(1 - r^2)}
$$

임을 유도하라.

(b) 숫자를 넣어 $140.3$ 이 나오는지 확인하라.

(c) 평균 공식 $\tfrac{1}{q}\sum_j t_j^2$ 이 맞는 것은 어떤 때인가. 연습문제 5 의 $t_1^2 = F$ 가 아무 조건 없이 성립한 까닭은 무엇인가. 그리고 이 식이 연습문제 4 의 "$F$ 는 기각하는데 어느 $t$ 도 기각하지 않는" 현상을 어디에서 허락하는가.

</div>

??? success "풀이"

    (a) $\hat{\boldsymbol\beta}_S = (\hat\beta_1, \hat\beta_2)^{\mathsf T}$ 와 그 추정공분산

    $$
    \widehat{\mathrm{Var}}(\hat{\boldsymbol\beta}_S) = \mathbf{V} =
    \begin{pmatrix} s_1^2 & r s_1 s_2 \\ r s_1 s_2 & s_2^2 \end{pmatrix},
    \qquad s_j = \mathrm{SE}(\hat\beta_j)
    $$

    에 대해 기울기 둘을 동시에 검정하는 $F$ 통계량은 제약 개수 $q = 2$ 로 나눈 이차형식

    $$
    F = \frac{1}{q}\,\hat{\boldsymbol\beta}_S^{\mathsf T}\,\mathbf{V}^{-1}\,\hat{\boldsymbol\beta}_S
    $$

    이다. $\det \mathbf{V} = s_1^2 s_2^2 (1 - r^2)$ 이므로

    $$
    \mathbf{V}^{-1} = \frac{1}{1 - r^2}
    \begin{pmatrix} 1/s_1^2 & -r/(s_1 s_2) \\ -r/(s_1 s_2) & 1/s_2^2 \end{pmatrix}.
    $$

    이것을 끼워 넣고 $t_j = \hat\beta_j / s_j$ 로 바꾸면

    $$
    \hat{\boldsymbol\beta}_S^{\mathsf T} \mathbf{V}^{-1} \hat{\boldsymbol\beta}_S
    = \frac{1}{1 - r^2}\left( \frac{\hat\beta_1^2}{s_1^2}
    - 2r\,\frac{\hat\beta_1}{s_1}\frac{\hat\beta_2}{s_2}
    + \frac{\hat\beta_2^2}{s_2^2} \right)
    = \frac{t_1^2 - 2r\,t_1 t_2 + t_2^2}{1 - r^2}
    $$

    이고, $q = 2$ 로 나누면 구하려던 식이다.

    (b) $t_1^2 = 59.94$, $t_2^2 = 235.50$, $t_1 t_2 = 118.81$ 이다. $r = 0.0683$ 이므로 $2r\,t_1t_2 = 16.24$, $1 - r^2 = 0.99533$ 이고

    $$
    F = \frac{59.94 - 16.24 + 235.50}{2 \times 0.99533}
    = \frac{279.20}{1.99066} = 140.25.
    $$

    보기 1 의 표에 적힌 $140.3$ 과 맞는다.

    (c) 평균 공식은 $r = 0$ 일 때, 곧 두 추정값이 무상관일 때만 맞는다. 그때 $\mathbf{V}$ 가 대각이 되어 교차항이 사라진다. 보기 1 은 $r = 0.0683$ 으로 $0$ 에 가깝기에 $140.25$ 와 $147.72$ 가 그만큼만 어긋났다. 연습문제 5 의 경우는 기울기가 하나뿐이어서 상관을 맺을 상대가 없고, $q = 1$ 이면 이차형식이 그대로 $t_1^2$ 이므로 동등성이 조건 없이 성립한다.

    또 이 식은 $r > 0$ 이면 교차항 $-2r t_1 t_2$ 가 (두 $t$ 의 부호가 같을 때) 분자를 깎아 $F$ 를 평균보다 작게 만든다는 것도 보여 준다. 두 추정값이 함께 흔들린다면 둘을 합친 증거는 각각을 더한 것보다 적다. 반대로 다중공선성이 심해 $\lvert r \rvert \to 1$ 이면 분모 $1 - r^2$ 가 $0$ 으로 가면서, 같은 부호의 작은 $t_1, t_2$ 와 $r < 0$ 의 조합만으로도 $F$ 가 얼마든지 커질 수 있다. 바로 연습문제 4 가 말한 현상이 대수적으로 앉아 있는 자리다. 개별 $t$ 는 다른 변수를 모두 조정한 뒤의 고유한 몫만 보는데, $F$ 는 그 몫들 사이의 상관까지 셈에 넣기 때문이다. $\square$

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span> 연습문제 3 은 무관한 설명변수 `caffeine` 의 $p$ 값이 크겠지만 "순전히 우연으로 약 5% 의 경우" $0.05$ 아래로 떨어진다고 했다. 그 $5\%$ 를 실제로 재라.

보기 3 의 설정에 `caffeine = rng.random(100) * 5` 를 넣은 회귀를 씨앗을 고정해 $5000$ 번 되돌려, `caffeine` 의 $p$ 값이 $0.05$ 아래인 비율과 그 계수의 95% 신뢰구간이 $0$ 을 덮는 비율을 세어라. 두 비율의 합이 왜 반드시 $100\%$ 가 되는지 밝히고, 측정값이 이름값 $5\%$ 와 맞는지 모의실험의 오차까지 셈하여 판정하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    import statsmodels.api as sm

    rng = np.random.default_rng(0)
    n, reps = 100, 5000
    rejected = 0
    covered = 0
    betas = []

    for _ in range(reps):
        study = rng.random(n) * 10
        sleep = rng.random(n) * 8
        caffeine = rng.random(n) * 5          # 점수와 무관한 설명변수
        scores = 5 + 2.5 * study - 1.5 * sleep + rng.standard_normal(n) * 2
        X_const = sm.add_constant(np.column_stack((study, sleep, caffeine)))
        res = sm.OLS(scores, X_const).fit()
        betas.append(res.params[3])
        if res.pvalues[3] < 0.05:
            rejected += 1
        lo, hi = res.conf_int()[3]
        if lo <= 0 <= hi:
            covered += 1

    print(f"반복 {reps}회")
    print(f"caffeine 의 p < 0.05 : {rejected}회 ({100 * rejected / reps:.2f}%)")
    print(f"95% 신뢰구간이 0 을 덮음 : {covered}회 ({100 * covered / reps:.2f}%)")
    print(f"beta_caffeine 평균 {np.mean(betas):.4f}, 표준편차 {np.std(betas):.4f}")
    ```

    출력:

    ```text
    반복 5000회
    caffeine 의 p < 0.05 : 255회 (5.10%)
    95% 신뢰구간이 0 을 덮음 : 4745회 (94.90%)
    beta_caffeine 평균 0.0006, 표준편차 0.1425
    ```

    두 수의 합이 $255 + 4745 = 5000$ 으로 정확히 모든 반복이다. 우연이 아니다. 1 절에서 적은 대로 $p < 0.05$ 와 "95% 신뢰구간이 $0$ 을 포함하지 않음" 은 같은 사건을 두 가지로 쓴 것이기 때문이다. 둘 다 $\lvert t \rvert > t^*_{96,\,0.025}$ 와 같다. 그러므로 한 비율은 다른 비율의 여사건이고 합이 $1$ 이 된다.

    측정값 $5.10\%$ 가 이름값 $5\%$ 와 맞는지 보려면 모의실험 자체의 오차를 셈해야 한다. 기각 여부가 베르누이 시행이므로 비율의 표준오차는

    $$
    \sqrt{\frac{0.05 \times 0.95}{5000}} = 0.00308,
    $$

    곧 $0.31$ 퍼센트포인트다. 측정값은 이름값에서 $0.10$ 퍼센트포인트, 표준오차의 $0.32$ 배만큼 떨어져 있다. 어긋났다고 볼 근거가 없다. 거꾸로 말하면 이 $5000$ 번으로는 참 기각률이 $4.5\%$ 인지 $5.5\%$ 인지도 가리지 못한다. 작은 확률을 재려면 반복이 더 필요하다.

    계수 쪽도 함께 보라. $\hat\beta_{\text{caffeine}}$ 의 평균은 $0.0006$ 으로 참값 $0$ 에서 사실상 벗어나지 않았고(비편향), 표본표준편차 $0.1425$ 가 그 추정값이 실제로 얼마나 흔들리는지를 말한다. $\sigma = 2$, $n = 100$ 에서 이만큼 흔들리는 추정값이 $255$ 번은 $0$ 에서 충분히 멀리 튀어 "유의"라는 딱지를 받았다는 것이다. 참 효과가 없는 변수를 모형에 더 넣을 때마다 이 $5\%$ 짜리 복권을 한 장 더 사는 셈이고, 설명변수를 스무 개 넣으면 그중 하나쯤이 유의하게 나오는 것이 오히려 정상이다. $\square$

---

## 정리하며

`statsmodels` 로 계수 검정을 **세 가지 상황**에서 수행했다.

- **출력에서 세 가지를 읽는다.** 계수(효과의 크기와 방향), 표준오차(정밀도), $p$ 값(유의성). **셋을 함께 보아야** 한다.
- **설명변수가 늘면 해석이 조건부가 된다.** "다른 변수를 고정한 채"라는 단서가 붙으며, 어떤 변수를 함께 넣었느냐에 따라 같은 변수의 계수가 달라진다.
- **참값을 아는 모의자료로 확인하는 것이 좋은 연습이다.** 계수 $3$ 과 $5$ 를 넣고 만든 자료에서 추정값이 그 근처에 오는지, 신뢰구간이 참값을 덮는지 확인할 수 있다.
- **표집변동을 기억한다.** 추정값이 참값과 정확히 같지 않은 것은 정상이며, **신뢰구간이 참값을 포함하는지**가 확인할 점이다.
- **유의하지 않다고 효과가 없는 것이 아니다.** 표본이 작거나 잡음이 크면 실재하는 효과도 못 잡는다.

다음 절부터 **가정과 진단**으로 넘어간다.
