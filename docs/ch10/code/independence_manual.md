# 독립성 검정 (수동 계산과 그림)

## 개요

이 페이지에서는 이원 분할표에 대한 카이제곱 독립성 검정을 **수동으로** 계산하는 과정을 따라간다. 주변 합계로부터 기대도수를 유도하고, NumPy와 SciPy로 검정통계량과 p-값을 계산하며, 기각역을 색칠한 $\chi^2$ 확률밀도함수를 그려 결과를 시각화한다. 수동 절차를 이해하면 `scipy.stats.chi2_contingency` 같은 상위 함수가 내부에서 무엇을 하는지 분명해진다.

## 가설

- **귀무가설** ($H_0$): 두 범주형 변수가 독립이다.
- **대립가설** ($H_A$): 두 범주형 변수가 독립이 아니다(서로 연관되어 있다).

## 기대도수

관측도수가 $O_{ij}$인 $r \times c$ 분할표에서 독립 아래의 기대도수는

$$
E_{ij} = \frac{R_i \cdot C_j}{n}
$$

이다. 여기서 $R_i = \sum_j O_{ij}$는 $i$번째 행 합계, $C_j = \sum_i O_{ij}$는 $j$번째 열 합계, $n$은 총합이다. 행렬로 쓰면

$$
E = \frac{\mathbf{r}\,\mathbf{c}^\top}{n}
$$

이며, $\mathbf{r}$과 $\mathbf{c}$는 각각 행 합계와 열 합계의 열벡터이다.

## 검정통계량

$$
\chi^2 = \sum_{i=1}^{r}\sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
$$

$H_0$ 아래에서 이 통계량은 근사적으로 자유도

$$
\text{df} = (r - 1)(c - 1)
$$

인 $\chi^2$ 분포를 따른다.

## 코드

### 기대도수 계산

```python
import numpy as np
from scipy import stats

def compute_expected(observed_counts: np.ndarray) -> np.ndarray:
    row_totals = observed_counts.sum(axis=1, keepdims=True)
    col_totals = observed_counts.sum(axis=0, keepdims=True)
    total = observed_counts.sum()
    return (row_totals @ col_totals) / total
```

`keepdims=True` 인자는 2차원 모양을 유지하여 행렬 곱 `row_totals @ col_totals`이 $r \times c$ 기대도수 행렬로 올바르게 계산되도록 한다.

### 전체 계산

```python
observed_counts = np.array([[934, 1070],
                            [113,   92],
                            [ 20,    8]], dtype=float)

expected_counts = compute_expected(observed_counts)
df = (observed_counts.shape[0] - 1) * (observed_counts.shape[1] - 1)

chi2 = np.sum((observed_counts - expected_counts)**2 / expected_counts)
p_value = stats.chi2(df).sf(chi2)

print(f"chi_squared_statistic = {chi2:.2f}")
print(f"p_value = {p_value:.2%}")
```

### 시각화

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 4))

x_left = np.linspace(0, chi2, 200)
y_left = stats.chi2(df).pdf(x_left)
ax.plot(x_left, y_left, linewidth=3)
x_fill = np.concatenate([[0], x_left, [chi2], [0]])
y_fill = np.concatenate([[0], y_left, [0], [0]])
ax.fill(x_fill, y_fill, alpha=0.1)

x_right = np.linspace(chi2, max(20, chi2 + 5), 200)
y_right = stats.chi2(df).pdf(x_right)
ax.plot(x_right, y_right, linewidth=3)
x_fill_r = np.concatenate([[chi2], x_right, [max(20, chi2 + 5)], [chi2]])
y_fill_r = np.concatenate([[0], y_right, [0], [0]])
ax.fill(x_fill_r, y_fill_r, alpha=0.1)

ax.annotate(f"p_value = {p_value:.2%}",
            xy=(chi2 * 0.8, y_left.max() * 0.15),
            xytext=(chi2 * 0.9 + 5, y_left.max() * 0.6),
            fontsize=12, arrowprops=dict(width=0.2, headwidth=8))

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_position("zero")
ax.spines["left"].set_position("zero")
plt.tight_layout()
plt.show()
```

## 해석

관측된 $3 \times 2$ 표에서 카이제곱 통계량은 약 $11.81$, 자유도는 $\text{df} = (3-1)(2-1) = 2$이다. p-값은 약 $0.0027$로 통상적인 문턱 $\alpha = 0.05$보다 훨씬 작다. 따라서 $H_0$을 **기각하고** 5% 수준에서 행 변수와 열 변수 사이에 연관이 있다고 결론짓는다.

그림을 보면 판정이 눈에 들어온다. 관측된 통계량이 오른쪽 꼬리 깊숙이 놓여 있어 색칠된 넓이(p-값)가 아주 작다. 다만 표본이 2,237명으로 크므로 유의성이 곧 실질적 중요성은 아니다. 이 표의 Cramér의 V는 $\sqrt{11.81/2237} \approx 0.073$에 불과해 연관 자체는 약하다.

## 연습문제

**1.** $2 \times 2$ 표

$$
\begin{pmatrix} 20 & 30 \\ 40 & 10 \end{pmatrix}
$$

에 대해 공식 $E_{ij} = R_i C_j / n$을 써서 기대도수 행렬을 손으로 계산하라.

??? success "연습문제 1 풀이"

    행 합계: $R_1 = 50$, $R_2 = 50$. 열 합계: $C_1 = 60$, $C_2 = 40$. 총합: $n = 100$.

    $$
    E_{11} = \frac{50 \times 60}{100} = 30, \quad E_{12} = \frac{50 \times 40}{100} = 20
    $$

    $$
    E_{21} = \frac{50 \times 60}{100} = 30, \quad E_{22} = \frac{50 \times 40}{100} = 20
    $$

    따라서 기대도수 행렬은

    $$
    E = \begin{pmatrix} 30 & 20 \\ 30 & 20 \end{pmatrix}
    $$

    이다. $\square$

---

**2.** 연습문제 1의 기대도수를 써서 카이제곱 통계량과 자유도를 계산하라. $\alpha = 0.05$에서 $H_0$을 기각하겠는가?

??? success "연습문제 2 풀이"

    $$
    \chi^2 = \frac{(20-30)^2}{30} + \frac{(30-20)^2}{20} + \frac{(40-30)^2}{30} + \frac{(10-20)^2}{20}
    $$

    $$
    = \frac{100}{30} + \frac{100}{20} + \frac{100}{30} + \frac{100}{20} = 3.333 + 5 + 3.333 + 5 = 16.667
    $$

    자유도: $\text{df} = (2-1)(2-1) = 1$.

    임계값은 $\chi^2_{0.05, 1} = 3.841$이다. $16.667 > 3.841$이므로 $H_0$을 **기각한다**. 두 변수는 유의하게 연관되어 있다. $\square$

---

**3.** `compute_expected` 함수에서 `keepdims=True`가 필요한 이유를 설명하라. 없으면 어떤 문제가 생기는가?

??? success "연습문제 3 풀이"

    `keepdims=True`가 없으면 `sum(axis=1)`은 모양이 $(r,)$인 1차원 배열을, `sum(axis=0)`은 모양이 $(c,)$인 1차원 배열을 준다. 1차원 배열 두 개의 행렬 곱 `@`는 원하는 $r \times c$ 외적 행렬이 아니라 스칼라(내적)를 준다.

    `keepdims=True`를 쓰면 모양이 각각 $(r, 1)$과 $(1, c)$가 된다. $(r, 1)$ 행렬과 $(1, c)$ 행렬의 곱은 $(r, c)$ 행렬이며, 이것이 기대도수에 필요한 외적이다. $\square$

---

**4.** 어떤 분할표에서든 기대도수의 합이 관측도수의 합과 같음, 즉 $\sum_{i,j} E_{ij} = n$임을 보여라.

??? success "연습문제 4 풀이"

    정의에 의해 $E_{ij} = R_i C_j / n$이다. 모든 칸에 대해 합하면

    $$
    \sum_{i=1}^{r}\sum_{j=1}^{c} E_{ij} = \sum_{i=1}^{r}\sum_{j=1}^{c} \frac{R_i C_j}{n} = \frac{1}{n}\sum_{i=1}^{r} R_i \sum_{j=1}^{c} C_j = \frac{1}{n} \cdot n \cdot n = n
    $$

    이다. 두 번째 단계는 $R_i$가 $j$에 의존하지 않아 안쪽 합에서 빼낼 수 있다는 사실을, 마지막 단계는 $\sum_i R_i = \sum_j C_j = n$을 이용한다. $\square$

---

**5.** 기대도수 표에서 자유로운 모수의 개수를 세어 카이제곱 독립성 검정의 자유도가 $(r-1)(c-1)$임을 증명하라.

??? success "연습문제 5 풀이"

    $H_0$(독립) 아래에서 칸 $(i,j)$의 결합확률은 $p_{ij} = p_{i\cdot} \cdot p_{\cdot j}$로 분해된다. 행 주변확률은 합이 1이어야 하므로 자유로운 모수가 $r - 1$개이고, 열 주변확률은 $c - 1$개이다. 따라서 독립 아래에서 자유로운 모수는 모두 $(r-1) + (c-1)$개이다.

    $r \times c$ 표에 대한 제약 없는 모형은 자유로운 칸 확률이 $rc - 1$개이다. 검정의 자유도는 그 차이이다:

    $$
    \text{df} = (rc - 1) - [(r - 1) + (c - 1)] = rc - 1 - r - c + 2 = rc - r - c + 1 = (r-1)(c-1)
    $$

    동등하게, (기대도수가 강제하는 대로) 행과 열의 주변 합계가 고정되면 표에서 자유롭게 변할 수 있는 칸의 수가 $(r-1)(c-1)$이다. $\square$
