# 계수 검정 예제

## 개요

이 페이지는 statsmodels로 회귀계수의 가설검정을 수행하는 방법을 보인다. 점점 복잡해지는 세 예제를 통해 OLS 출력에서 $t$ 통계량, $p$값, 신뢰구간을 뽑아내는 법과, 설명변수가 하나일 때와 여럿일 때의 통계적 유의성을 해석하는 법을 다룬다.

## 수학적 배경

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

## 코드

<div class="exbox" markdown>

**보기 1.** 설명변수 두 개.

</div>

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

**보기 2.** p값과 신뢰구간 뽑아내기.

</div>

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

**보기 3.** 여러 설명변수의 해석.

</div>

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

## 해석

- 보기 1에서는 두 설명변수($x_1$과 $x_2$)의 참 계수가 모두 0이 아니므로(3과 5) 두 $p$값이 모두 작고 둘 다 유의하다고 판정될 것으로 기대한다.
- 보기 2에서는 설명변수 하나의 참 계수가 2.5이고 잡음 표준편차가 1이다. 기울기의 $p$값이 매우 작아 선형관계를 확인해 줄 것이다.
- 보기 3에서는 "Study Hours"가 시험 점수에 양의 효과(계수 $\approx 2.5$)를, "Sleep Hours"가 음의 효과(계수 $\approx -1.5$)를 갖는다. 둘 다 통계적으로 유의해야 한다. 신뢰구간이 효과 크기의 그럴듯한 범위를 준다. 추정값이 참값과 조금 다른 것은($2.43$ 대 $2.5$, $-1.32$ 대 $-1.5$) 표집변동 때문이며, 참값이 신뢰구간 안에 들어 있는지 확인해 보면 좋다.
- 어떤 설명변수가 "유의하지 않다"($p > 0.05$)는 것이 효과가 0임을 증명하지는 않는다. 선택한 유의수준에서 $H_0$을 기각할 증거가 부족하다는 뜻일 뿐이다. 신뢰구간이 같은 정보를 더 유익하게 전달한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** 보기 1에서 계수와 그 표준오차로부터 $\hat{\beta}_1$의 $t$ 통계량을 직접 계산하라. `results.tvalues[1]`의 값과 일치하는지 확인하라.

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

**연습문제 2.** 보기 2에서 잡음 표준편차를 1에서 5로 키우도록 고쳐라. 기울기의 $p$값과 신뢰구간은 어떻게 달라지는가?

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

**연습문제 3.** 보기 3에 시험 점수와 무관한 세 번째 설명변수 `caffeine = np.random.rand(100) * 5`를 추가하라. 그 $p$값에 대해 무엇을 기대하며 그 이유는 무엇인가?

</div>

??? success "풀이"

    caffeine이 시험 점수와 독립적으로 생성되었으므로 참 계수는 0이다. OLS 추정값 $\hat{\beta}_{\text{caffeine}}$은 0에 가깝고 $p$값은 클 것이다(대개 $> 0.05$). 다만 순전히 우연으로($\alpha = 0.05$ 수준에서 약 5%의 경우) $p$값이 0.05 아래로 떨어질 수 있는데, 그것이 제1종 오류이다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** 전체 모형 유의성의 $F$ 검정과 개별 $t$ 검정의 관계를 설명하라. 둘이 다른 결론을 줄 수 있는 경우는 언제인가?

</div>

??? success "풀이"

    $F$ 검정은 $H_0\colon \beta_1 = \beta_2 = \cdots = \beta_{k-1} = 0$(모든 기울기가 동시에 0)을 평가하고, 각 $t$ 검정은 계수 하나를 평가한다. 설명변수들이 무상관이면 개별 $t$ 검정들이 독립이고, 어느 $t$ 검정이 기각하면 사실상 $F$ 검정도 기각한다. 설명변수들이 상관되어 있으면 $F$ 검정은 기각하는데(모형 전체는 유용한데) 어느 개별 $t$ 검정도 기각하지 않는(나머지를 조정하고 나면 어느 설명변수도 유의하지 않은) 일이 가능하다. 다중공선성이 있을 때 일어나는 현상이다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** 설명변수가 하나인 단순선형회귀에서 $t_1^2 = F$임을 증명하라. 이 동등성이 깨지는 조건은 무엇인가?

</div>

??? success "풀이"

    단순회귀에서는 $k = 2$(절편 + 기울기)이고 $F$ 통계량은

    $$
    F = \frac{\mathrm{ESS}/1}{\mathrm{RSS}/(n-2)}.
    $$

    기울기의 $t$ 통계량은 $t_1 = \hat{\beta}_1 / \mathrm{SE}(\hat{\beta}_1)$이다. 대수적으로 $t_1^2 = \mathrm{ESS}/s^2 = F$임을 보일 수 있다. 이 동등성은 설명변수가 하나일 때(분자 자유도가 1일 때) 정확히 성립한다. 다중회귀에서는 $F$ 검정의 분자 자유도가 $k-1 > 1$이고 모든 기울기를 동시에 검정하므로 동등성이 깨진다. $\square$
