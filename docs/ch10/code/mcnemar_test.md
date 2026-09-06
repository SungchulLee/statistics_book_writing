# McNemar 검정 (대응된 이진 자료)

## 개요

**McNemar 검정**은 대응된 이진 자료를 분석하는 데 쓰인다. 같은 대상을 두 조건에서 측정하는 전후 연구에서 흔히 나온다. 이 검정은 두 조건 사이에서 결과가 바뀐 대상인 **불일치 쌍**에 주목하여, 변화가 대칭인지 아니면 한쪽 방향의 변화가 유의하게 더 흔한지를 판정한다.

## 연구 설계

McNemar 검정은 다음일 때 적용한다:

- 각 대상을 **두 조건**에서 관측한다(예: 처치 전후, 두 진단검사).
- 각 조건에서의 결과가 **이진**이다(예: 양성/음성, 성공/실패).
- 관측이 **대응**되어 있다(같은 대상이 두 측정값을 모두 제공한다).

자료는 대응 도수의 $2 \times 2$ 표로 정리한다:

$$
\begin{array}{c|cc}
 & \text{After +} & \text{After -} \\
\hline
\text{Before +} & a & b \\
\text{Before -} & c & d
\end{array}
$$

- $a$: 두 시점 모두 양성(일치)
- $d$: 두 시점 모두 음성(일치)
- $b$: 이전 양성, 이후 음성(불일치)
- $c$: 이전 음성, 이후 양성(불일치)

## 가설

- **귀무가설** ($H_0$): 한 방향으로 바뀔 확률과 다른 방향으로 바뀔 확률이 같다. 즉 $P(b) = P(c)$.
- **대립가설** ($H_A$): 두 방향의 변화 확률이 다르다. 즉 $P(b) \ne P(c)$.

## 검정통계량

**연속성 보정**을 적용한 McNemar 통계량은

$$
\chi^2 = \frac{(|b - c| - 1)^2}{b + c}
$$

이고, **연속성 보정을 하지 않은** 형태는

$$
\chi^2 = \frac{(b - c)^2}{b + c}
$$

이다. $b + c$가 충분히 크면(보통 $b + c \ge 25$) $H_0$ 아래에서 이 통계량은 근사적으로 $\chi^2(1)$ 분포를 따른다.

## 코드

### 직접 구현

```python
import numpy as np
from scipy import stats

def mcnemar_test(table):
    """
    Perform McNemar's test on a 2x2 table of paired counts.

    Parameters
    ----------
    table : array-like, shape (2, 2)
        Contingency table where off-diagonal cells (b, c)
        represent discordant pairs:
            [[a, b],
             [c, d]]

    Returns
    -------
    statistic : float   McNemar chi-square statistic (continuity-corrected)
    p_value   : float   Two-sided p-value from chi-square(1)
    """
    table = np.asarray(table)
    b = table[0, 1]
    c = table[1, 0]
    # Continuity-corrected McNemar statistic
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = stats.chi2(1).sf(chi2)
    return chi2, p_value
```

### 검정 실행

```python
# Disease status before/after treatment
#                  After+   After-
# Before+           101      121
# Before-            59       33
table = np.array([[101, 121],
                  [ 59,  33]])

chi2, p = mcnemar_test(table)

print(f"McNemar chi2 = {chi2:.4f}")
print(f"p-value      = {p:.4e}")

if p < 0.05:
    print("Reject H0: significant change after treatment (alpha = 0.05).")
else:
    print("Fail to reject H0: no significant change (alpha = 0.05).")
```

**이 예제의 주요 값:**

- 불일치 쌍의 도수: $b = 121$, $c = 59$.
- 통계량은 양성에서 음성으로 바뀐 121명이 음성에서 양성으로 바뀐 59명과 유의하게 다른지를 검정한다.
- $\chi^2 = (|121-59| - 1)^2 / 180 = 61^2/180 \approx 20.67$이고 p-값은 약 $5.4 \times 10^{-6}$이다.

## 해석

McNemar 검정에서는 불일치 쌍만이 의미를 갖는다. 일치 쌍($a$와 $d$)은 변화의 차이에 대해 아무 정보도 주지 않는다.

질병 예제에서:

- $b = 121$명이 호전되었다(양성 → 음성).
- $c = 59$명이 악화되었다(음성 → 양성).
- 비대칭이 상당하다. 악화된 사람보다 호전된 사람이 훨씬 많다.
- p-값이 $0.05$보다 훨씬 작으므로 처치가 질병 상태에 통계적으로 유의한 변화를 만들었다고 결론짓는다.

## 연습문제

**1.** 어떤 진단 연구가 환자 200명에게 두 검사를 비교했다. 대응된 결과는 다음과 같다:

$$
\begin{pmatrix} 80 & 15 \\ 25 & 80 \end{pmatrix}
$$

(연속성 보정을 적용한) McNemar 통계량을 계산하고 $\alpha = 0.05$에서 두 검사가 유의하게 다른지 판정하라.

??? success "연습문제 1 풀이"

    불일치 도수는 $b = 15$, $c = 25$이다.

    $$
    \chi^2 = \frac{(|15 - 25| - 1)^2}{15 + 25} = \frac{(10 - 1)^2}{40} = \frac{81}{40} = 2.025
    $$

    $\text{df} = 1$에서 $\alpha = 0.05$의 임계값은 $3.841$이다. $2.025 < 3.841$이므로 $H_0$을 **기각하지 못한다**. 두 진단검사 사이에 유의한 차이가 없다. $\square$

---

**2.** 일치 쌍($a$와 $d$)이 McNemar 검정통계량에 기여하지 않는 이유를 설명하라.

??? success "연습문제 2 풀이"

    일치 쌍은 두 조건 사이에서 결과가 바뀌지 않은 대상(양성–양성 또는 음성–음성)이다. 이들은 어느 조건에서든 결과가 같으므로 두 조건이 다른지에 대한 증거를 주지 않는다. 관심 있는 질문은 *변화*가 대칭인가, 즉 실제로 바뀐 대상들 중에서 양방향으로 똑같이 바뀌었는가이다. 이 비대칭에 대한 정보는 불일치 쌍($b$와 $c$)만이 담고 있다. 일치 쌍을 넣으면 무관한 정보로 검정이 희석되어 검정력이 떨어진다. $\square$

---

**3.** $b = 30$, $c = 10$에 대해 연속성 보정을 한 경우와 하지 않은 경우의 McNemar 통계량을 계산하라. 두 값은 얼마나 차이 나는가?

??? success "연습문제 3 풀이"

    연속성 보정 없이:

    $$
    \chi^2 = \frac{(30 - 10)^2}{30 + 10} = \frac{400}{40} = 10.0
    $$

    연속성 보정을 적용하면:

    $$
    \chi^2 = \frac{(|30 - 10| - 1)^2}{30 + 10} = \frac{(19)^2}{40} = \frac{361}{40} = 9.025
    $$

    보정한 값이 $0.975$만큼 작다. 둘 다 매우 유의한 p-값을 주므로($\chi^2(1)$에서 $0.001$을 훨씬 밑돈다) 이 경우 보정이 결론을 바꾸지 않는다. 보정은 $b + c$가 작고 통계량이 임계값 근처일 때 더 중요해진다. $\square$

---

**4.** $b + c$가 작을 때(가령 25 미만) McNemar 검정의 카이제곱 근사가 좋지 않을 수 있다. 이항분포에 기반한 정확한 대안을 기술하라.

??? success "연습문제 4 풀이"

    $H_0$ 아래에서 각 불일치 쌍이 $b$형일 확률과 $c$형일 확률이 같으므로 $b \sim \text{Binomial}(b + c, 0.5)$이다. 양측검정의 정확한 p-값은

    $$
    p = 2 \cdot \min\bigl[P(X \le b),\; P(X \ge b)\bigr]
    $$

    이며 $X \sim \text{Binomial}(b + c, 0.5)$이다. Python에서는:

    ```python
    from scipy import stats
    n_discordant = b + c
    p_exact = stats.binomtest(b, n_discordant, 0.5).pvalue
    ```

    (예전 이름인 `stats.binom_test`는 최신 SciPy에서 제거되었다.)

    이 정확 이항검정은 분포에 대한 근사를 전혀 쓰지 않아 어떤 표본크기에서도 타당하다. 불일치 쌍의 수가 적을 때 권장되는 접근이다. $\square$

---

**5.** $H_0$(불일치 쌍 중 $P(b) = P(c) = 0.5$) 아래에서 보정하지 않은 McNemar 통계량이 $b + c \to \infty$일 때 $\chi^2(1)$로 분포수렴함을 증명하라.

??? success "연습문제 5 풀이"

    불일치 쌍의 총수를 $n = b + c$라 하자. $H_0$ 아래에서 $b \sim \text{Binomial}(n, 0.5)$이므로 $E[b] = n/2$, $\text{Var}(b) = n/4$이다.

    보정하지 않은 McNemar 통계량은

    $$
    \chi^2 = \frac{(b - c)^2}{b + c} = \frac{(b - (n - b))^2}{n} = \frac{(2b - n)^2}{n}
    $$

    으로 쓸 수 있다.

    이제 $Z = (b - n/2) / \sqrt{n/4} = (2b - n) / \sqrt{n}$이라 두면, 중심극한정리에 의해 $n \to \infty$일 때 $Z \xrightarrow{d} N(0, 1)$이다.

    통계량은

    $$
    \chi^2 = \frac{(2b - n)^2}{n} = \left(\frac{2b - n}{\sqrt{n}}\right)^2 = Z^2
    $$

    이다. $Z \xrightarrow{d} N(0,1)$이면 $Z^2 \xrightarrow{d} \chi^2(1)$이므로 McNemar 통계량은 $\chi^2(1)$로 분포수렴한다. $\square$
