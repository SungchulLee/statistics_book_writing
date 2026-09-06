# Pearson 상관계수

두 양적 변수를 분석할 때 자연스럽게 떠오르는 첫 질문은 "둘이 함께 움직이는 경향이 있는가?"이다. **Pearson 상관계수**는 두 변수 사이 *선형* 관계의 강도와 방향을 단위 없이 정확하게 재는 측도이다. 통계학에서 가장 널리 쓰이는 상관 측도이다.

---

## 모상관

분산이 유한한 두 확률변수 $X$와 $Y$에 대해 **모집단 Pearson 상관계수**는 다음으로 정의된다:

$$
\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \, \sigma_Y} = \frac{\mathbb{E}[(X - \mu_X)(Y - \mu_Y)]}{\sqrt{\mathbb{E}[(X - \mu_X)^2]} \; \sqrt{\mathbb{E}[(Y - \mu_Y)^2]}}
$$

여기서 $\mu_X = \mathbb{E}[X]$, $\mu_Y = \mathbb{E}[Y]$, $\sigma_X = \sqrt{\text{Var}(X)}$, $\sigma_Y = \sqrt{\text{Var}(Y)}$이다.

핵심 성질은 $\rho_{XY}$가 언제나 구간 $[-1, 1]$ 안에 있다는 것이다.

---

## 표본상관계수

짝지어진 관측값 $(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)$이 주어졌을 때 **표본 Pearson 상관계수**는

$$
r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2} \; \sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}
$$

이며 $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$와 $\bar{y} = \frac{1}{n}\sum_{i=1}^n y_i$는 표본평균이다. 계산에 편한 동등한 공식은

$$
r = \frac{n \sum x_i y_i - (\sum x_i)(\sum y_i)}{\sqrt{[n \sum x_i^2 - (\sum x_i)^2][n \sum y_i^2 - (\sum y_i)^2]}}
$$

이다. 표본상관 $r$은 모상관 $\rho_{XY}$의 일치추정량이다.

---

## 성질

Pearson 상관계수는 몇 가지 중요한 성질을 만족한다.

1. **유계성.** 어떤 자료에서든 $-1 \le r \le 1$이고, 분산이 유한한 어떤 확률변수 쌍에서든 $-1 \le \rho \le 1$이다.

2. **대칭성.** $r_{XY} = r_{YX}$이며 $\rho$도 마찬가지이다.

3. **평행이동 불변성.** 임의의 상수 $a$, $b$에 대해

    $$
    r_{X+a, \, Y+b} = r_{X, Y}
    $$

    이다. 어느 변수를 평행이동해도 상관은 변하지 않는다.

4. **축척 불변성.** 양의 상수 $a > 0$, $b > 0$에 대해

    $$
    r_{aX, \, bY} = r_{X, Y}
    $$

    이다. 양수 배로 축척을 바꾸어도 상관은 보존된다. $a < 0$이거나 $b < 0$이면 $r$의 부호가 뒤집힌다.

5. **완전 상관.** $|r| = 1$일 필요충분조건은 모든 자료점이 정확히 한 직선 위에 있는 것이다. 구체적으로 어떤 $b > 0$에 대해 $y_i = a + bx_i$이면 $r = 1$이고, $b < 0$이면 $r = -1$이다.

---

## 해석

$r$의 값은 선형 연관의 강도와 방향을 기술한다.

| $r$의 범위 | 해석 |
|:---:|:---|
| $0.7 \le r \le 1.0$ | 강한 양의 선형관계 |
| $0.3 \le r < 0.7$ | 중간 정도의 양의 선형관계 |
| $0 < r < 0.3$ | 약한 양의 선형관계 |
| $r = 0$ | 선형관계 없음 |
| $-0.3 < r < 0$ | 약한 음의 선형관계 |
| $-0.7 < r \le -0.3$ | 중간 정도의 음의 선형관계 |
| $-1.0 \le r \le -0.7$ | 강한 음의 선형관계 |

이 문턱들은 관례적인 지침이지 엄격한 규칙이 아니다. 맥락이 중요하다. 어떤 분야(예: 물리학)에서는 $r = 0.7$이 약하다고 여겨지지만 사회과학에서는 강하다고 볼 수 있다.

---

## 결정계수

Pearson 상관의 제곱 $r^2$을 **결정계수**라 한다. $Y$의 분산 중 $X$로 선형적으로 설명되는 비율을 나타낸다:

$$
r^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}
$$

여기서 $\hat{y}_i$는 $X$에 대한 $Y$의 단순선형회귀에서 나온 적합값이다.

예를 들어 $r = 0.80$이면 $r^2 = 0.64$이므로 $Y$ 변동의 64%가 $X$와의 선형관계로 설명된다.

---

## 선형성 가정

결정적인 한계는 Pearson의 $r$이 **선형** 연관만 잰다는 점이다. 두 변수가 강한 비선형 관계를 가지면서도 $r \approx 0$이 나올 수 있다.

!!! warning "Pearson의 r은 비선형 관계에서 오도할 수 있다"
    $X$가 $[-1, 1]$ 위에서 균등분포를 따르고 $Y = X^2$이라고 하자. 완벽한 결정적 관계가 있지만 관계가 대칭이고 비선형이므로 $r_{XY} = 0$이다. $r$을 해석하기 전에 반드시 산점도를 살펴보라.

Anscombe의 사중주가 유명한 예이다. $r$ 값은 거의 같지만 산점도 패턴이 아주 다른 자료 네 벌이다. 수치 요약과 함께 시각화가 왜 중요한지 잘 보여준다.

---

## 이상점에 대한 민감성

Pearson 상관은 평균과 표준편차에 기반하는데, 이들 자체가 극단값에 민감하므로 상관도 이상점에 민감하다. 이상점 하나가 $r$을 극적으로 키우거나 줄일 수 있다.

??? example "이상점이 상관에 미치는 영향"
    자료 $(1,1), (2,2), (3,3), (4,4), (5,5)$는 $r = 1.0$이다. 여기에 점 $(10, -5)$ 하나를 더하면 $r$이 약 $-0.65$로 바뀐다. 이상점 하나가 완전한 양의 상관을 중간 정도의 음의 상관으로 뒤집었다.

이상점이 있거나 자료가 정규가 아니면 [Spearman 상관](spearman.md)이나 [Kendall의 타우](kendall.md) 같은 순위 기반 대안을 고려하라.

---

## Python으로 계산하기

```python
import numpy as np
from scipy import stats

# Sample data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 12.3, 13.8, 16.1, 18.0, 20.2])

# Method 1: NumPy correlation matrix
corr_matrix = np.corrcoef(x, y)
r_numpy = corr_matrix[0, 1]
print(f"NumPy r = {r_numpy:.4f}")

# Method 2: SciPy (also returns the p-value)
r_scipy, p_value = stats.pearsonr(x, y)
print(f"SciPy r = {r_scipy:.4f}, p-value = {p_value:.6f}")
```

`scipy.stats.pearsonr` 함수는 표본상관과 함께 귀무가설 $H_0\!: \rho = 0$에 대한 양측 p-값을 돌려준다. 이 가설검정의 자세한 내용은 [Pearson의 r 검정](../correlation_test/test_pearson.md)을 보라.

---

## 요약

Pearson 상관계수 $r$은 두 변수 사이 선형관계의 강도와 방향을 수치화한다. $-1$(완전한 음의 관계)에서 $+1$(완전한 양의 관계)까지의 값을 가지며 $0$은 선형 연관이 없음을 뜻한다. 강력하고 널리 쓰이지만 $r$은 선형관계만 포착하고 이상점에 민감하다. $r$의 수치는 언제나 산점도와 함께 보아 선형모형이 적절한지 확인해야 한다.

## 연습문제

**연습문제 1.**
키–몸무게 자료를 써서 다음 각각에 대해 Pearson 상관계수를 계산하라:

1. 남성만
2. 여성만
3. 전체 자료

```python
import pandas as pd

url = "https://raw.githubusercontent.com/beccadsouza/Machine-Learning-Python/master/Datasets/height-weight.csv"
data = pd.read_csv(url)

# Your code here
```

전체 상관이 집단 내 상관과 다를 수 있는 이유를 논하라.

??? success "연습문제 1 풀이"

    남성과 여성에 대해 각각, 그리고 전체 자료에 대해 Pearson 상관을 계산하면 대체로 값이 다르게 나온다. 전체 자료에는 집단 간 변동이 포함되기 때문이다. 남성이 여성보다 키도 크고 몸무게도 무거운 경향이 있다면, 두 집단을 합칠 때 (집단 차이라는) 양의 공변동 원천이 하나 더 생겨 전체 상관이 집단 내 상관보다 커질 수 있다. 이는 생태학적 상관 효과의 한 예이다.

---

**연습문제 2.**
$\rho \in \{-0.99, -0.8, -0.5, 0, 0.5, 0.8, 0.99\}$에 대해 이변량 정규 표본을 생성하여 한 행의 부분그림들에 표시하는 함수를 작성하라. 각 부분그림의 제목에 표본 Pearson $r$도 출력하라.

??? success "연습문제 2 풀이"

    각 $\rho$에 대해 공분산행렬 $\Sigma = \begin{pmatrix} 1 & \rho \\ \rho & 1 \end{pmatrix}$로 `numpy.random.multivariate_normal`을 쓴다. 부분그림마다 $n = 200$개를 생성하고 `numpy.corrcoef`로 Pearson $r$을 계산한 뒤 `matplotlib.pyplot.subplots(1, 7)`로 표시한다. $|\rho|$가 커질수록 산점도가 점점 좁은 타원이 되고 $\pm 0.99$에서는 거의 직선으로 붕괴한다.

---

**연습문제 3.**
`scipy`를 쓰거나 직접 Anscombe의 사중주를 재현하라. 네 자료 각각에 대해 Pearson $r$을 계산하고, 산점도 패턴이 아주 다른데도 값이 거의 같음을 확인하라.

```python
# Hint: Anscombe's quartet is available in seaborn
import seaborn as sns
anscombe = sns.load_dataset("anscombe")
```

??? success "연습문제 3 풀이"

    Anscombe 사중주의 네 자료 모두 Pearson $r \approx 0.816$을 준다. 그러나 산점도는 아주 다른 관계를 드러낸다. 선형 추세, 휘어진 관계, 이상점 하나가 있는 완전한 선형 추세, 그리고 극단적인 점 하나가 상관을 만들어내는 자료이다. Pearson $r$만으로는 관계를 규정하기에 부족하며 시각적 검토가 언제나 필요함을 보여준다.
