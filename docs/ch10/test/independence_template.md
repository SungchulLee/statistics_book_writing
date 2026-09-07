# 독립성 검정 템플릿 함수

## 개요

이 페이지에서는 `scipy.stats.chi2_contingency` 위에 세운, 재사용 가능한 카이제곱 독립성 검정 템플릿 함수를 제시한다. SciPy 호출을 문서화된 함수로 감싸 두면 어떤 이원 분할표에도 손쉽게 검정을 적용할 수 있다. 이 함수는 $2 \times 2$ 표를 위한 Yates 연속성 보정도 선택적으로 제공한다.

## 가설

- **귀무가설** ($H_0$): 행 변수와 열 변수가 독립이다.
- **대립가설** ($H_A$): 행 변수와 열 변수 사이에 연관이 있다.

## 검정통계량

$$
\chi^2 = \sum_{i=1}^{r}\sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
$$

자유도는 $\text{df} = (r-1)(c-1)$이고, $E_{ij} = R_i C_j / n$은 독립 아래의 기대도수이다.

### Yates 연속성 보정

$2 \times 2$ 표에서는 선택적인 **Yates 보정**이 각 항을 다음과 같이 수정한다:

$$
\chi^2_{\text{Yates}} = \sum_{i=1}^{2}\sum_{j=1}^{2} \frac{(|O_{ij} - E_{ij}| - 0.5)^2}{E_{ij}}
$$

이 보정은 검정통계량을 조금 줄여, 칸 도수가 작을 때 검정을 더 보수적으로 만든다.

## 코드

### 템플릿 함수

```python
import numpy as np
from scipy import stats

def chi2_independence(observed: np.ndarray, correction: bool = False):
    """Run chi-square test of independence.

    Parameters
    ----------
    observed : np.ndarray
        2D contingency table of observed counts.
    correction : bool
        Yates' continuity correction (only applied to 2x2).
        Default False.

    Returns
    -------
    chi2, p, df, expected : tuple
        Test statistic, p-value, degrees of freedom,
        and expected counts.
    """
    return stats.chi2_contingency(observed, correction=correction)
```

### 사용 예

```python
observed = np.array([[30, 20, 10],
                     [12, 25, 18]], dtype=float)

chi2, p, df, exp = chi2_independence(observed, correction=False)
print(f"chi2 = {chi2:.3f}, p = {p:.4f}, df = {df}")
print("expected:\n", exp)
```

**출력:**

- $\chi^2 \approx 10.358$, $p \approx 0.0056$, $\text{df} = 2$
- 기대도수는 주변 합계로부터 자동으로 계산된다.

## 템플릿을 언제 어떻게 쓰는가

| 상황 | `correction` |
|----------|:------------:|
| $2 \times 2$보다 큰 표 | `False` (보정은 $2 \times 2$ 전용) |
| 모든 $E_{ij} \ge 5$인 $2 \times 2$ 표 | `False` (표준 검정으로 충분) |
| 일부 $E_{ij}$가 5에 가까운 $2 \times 2$ 표 | `True` (보수적 조정) |
| $E_{ij} < 5$인 칸이 있는 $2 \times 2$ 표 | Fisher의 정확검정을 고려 |

## 해석

예제 표에서 검정은 $\chi^2 = 10.358$, $\text{df} = 2$, $p = 0.0056$을 준다. $\alpha = 0.05$에서 $H_0$을 **기각하고** 행 변수와 열 변수가 독립이 아니라고 결론짓는다. 연관은 통계적으로 유의하다.

*어느* 칸이 유의성을 이끄는지 알아보려면 표준화 잔차 $(O_{ij} - E_{ij}) / \sqrt{E_{ij}}$를 살펴본다. 절댓값이 큰 잔차를 가진 칸이 검정통계량에 가장 많이 기여한다.

## 연습문제

**1.** 템플릿 함수를 표

$$
\begin{pmatrix} 50 & 50 \\ 50 & 50 \end{pmatrix}
$$

에 적용하라. 코드를 돌리기 전에 어떤 결과를 예상하는가? 확인해 보라.

??? success "풀이"

    주변 합계가 모두 같으므로 관측도수가 기대도수와 일치한다. 모든 칸에서 $O_{ij} = E_{ij}$이므로 $\chi^2 = 0$, $p = 1.0$이다. 이 표본에서 두 변수는 완전히 독립으로 보인다.

    ```python
    chi2, p, df, exp = chi2_independence(
        np.array([[50, 50], [50, 50]], dtype=float)
    )
    # chi2 = 0.0, p = 1.0, df = 1
    ```

    $\square$

---

**2.** $2 \times 2$ 표 $\begin{pmatrix} 10 & 5 \\ 3 & 12 \end{pmatrix}$에 대해 Yates 보정을 적용한 경우와 하지 않은 경우로 템플릿을 실행하라. 두 $\chi^2$ 값을 비교하고 차이를 설명하라.

??? success "풀이"

    행 합계: $R_1 = 15$, $R_2 = 15$. 열 합계: $C_1 = 13$, $C_2 = 17$. 총합: $n = 30$.

    $$
    E_{11} = \frac{15 \times 13}{30} = 6.5, \quad E_{12} = \frac{15 \times 17}{30} = 8.5
    $$

    $$
    E_{21} = 6.5, \quad E_{22} = 8.5
    $$

    Yates 보정 없이:

    $$
    \chi^2 = \frac{(10-6.5)^2}{6.5} + \frac{(5-8.5)^2}{8.5} + \frac{(3-6.5)^2}{6.5} + \frac{(12-8.5)^2}{8.5} = 1.885 + 1.441 + 1.885 + 1.441 = 6.652
    $$

    Yates 보정을 적용하면 각 분자가 $(|O_{ij} - E_{ij}| - 0.5)^2 = (3.5 - 0.5)^2 = 9$가 되어

    $$
    \chi^2_{\text{Yates}} = \frac{9}{6.5} + \frac{9}{8.5} + \frac{9}{6.5} + \frac{9}{8.5} = 1.385 + 1.059 + 1.385 + 1.059 = 4.887
    $$

    이다. Yates 보정 통계량이 더 작아 p-값이 더 커진다. 이 보정은 이산인 검정통계량에 연속인 $\chi^2$ 분포를 쓰는 데서 오는 근사를 보완한다. $\square$

---

**3.** 이 함수는 값 네 개를 돌려준다. 기대도수 출력만 써서 예제 표의 표준화 잔차를 계산하는 코드를 작성하라. 어느 칸이 카이제곱 통계량에 가장 많이 기여하는가?

??? success "풀이"

    ```python
    observed = np.array([[30, 20, 10],
                         [12, 25, 18]], dtype=float)
    _, _, _, expected = chi2_independence(observed)
    residuals = (observed - expected) / np.sqrt(expected)
    print(residuals)
    ```

    표준화 잔차는

    $$
    \begin{pmatrix} 1.73 & -0.72 & -1.21 \\ -1.80 & 0.75 & 1.26 \end{pmatrix}
    $$

    이다. 절댓값이 가장 큰 칸은 첫 번째 열의 두 칸(1.73과 −1.80)이며, 그다음이 세 번째 열(−1.21과 1.26)이다. 즉 첫 번째 집단은 범주 1에 과다 대표되고 두 번째 집단은 범주 3에 과다 대표된다. $\square$

---

**4.** Yates 보정 통계량이 언제나 보정하지 않은 통계량보다 작거나 같음을 증명하라.

??? success "풀이"

    임의의 칸 $(i,j)$에 대해 $d_{ij} = |O_{ij} - E_{ij}|$라 하자. 보정하지 않은 기여는 $d_{ij}^2 / E_{ij}$이고, Yates 보정 기여는 $(\max(d_{ij} - 0.5, 0))^2 / E_{ij}$이다.

    모든 $d_{ij} \ge 0$에 대해 $\max(d_{ij} - 0.5, 0) \le d_{ij}$이므로

    $$
    \frac{(\max(d_{ij} - 0.5, 0))^2}{E_{ij}} \le \frac{d_{ij}^2}{E_{ij}}
    $$

    이다. 모든 칸에 대해 합하면

    $$
    \chi^2_{\text{Yates}} = \sum_{i,j} \frac{(\max(|O_{ij} - E_{ij}| - 0.5, 0))^2}{E_{ij}} \le \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}} = \chi^2
    $$

    이 되어, 보정 통계량은 언제나 보정하지 않은 것보다 작거나 같고 따라서 검정이 더 보수적이 된다(p-값이 커진다). $\square$

---

**5.** 어떤 임상시험이 두 처치와 세 중증도 수준에 걸쳐 결과를 기록했다. 분할표는

$$
\begin{pmatrix} 45 & 30 & 25 \\ 35 & 40 & 25 \end{pmatrix}
$$

이다. 템플릿 함수로 $\alpha = 0.05$에서 독립성을 검정하라. 통계량, p-값, 결론을 보고하라.

??? success "풀이"

    ```python
    table = np.array([[45, 30, 25],
                      [35, 40, 25]], dtype=float)
    chi2, p, df, exp = chi2_independence(table, correction=False)
    ```

    행 합계: $R_1 = 100$, $R_2 = 100$. 열 합계: $C_1 = 80$, $C_2 = 70$, $C_3 = 50$. 총합: $n = 200$.

    기대도수:

    $$
    E = \begin{pmatrix} 40 & 35 & 25 \\ 40 & 35 & 25 \end{pmatrix}
    $$

    $$
    \chi^2 = \frac{(45-40)^2}{40} + \frac{(30-35)^2}{35} + \frac{(25-25)^2}{25} + \frac{(35-40)^2}{40} + \frac{(40-35)^2}{35} + \frac{(25-25)^2}{25}
    $$

    $$
    = 0.625 + 0.714 + 0 + 0.625 + 0.714 + 0 = 2.679
    $$

    $\text{df} = (2-1)(3-1) = 2$에서 p-값은 약 $0.262$이다. $p > 0.05$이므로 $H_0$을 **기각하지 못한다**. 처치와 중증도 수준 사이에 유의한 연관이 없다. $\square$
