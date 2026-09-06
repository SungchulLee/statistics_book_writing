# 회귀와 상관 그림

## 개요

이 페이지는 단순선형회귀 모형에서 Pearson 상관계수 $r$가 두 요인 — 참 회귀직선의 기울기와 오차항의 척도 — 에 어떻게 의존하는지 살펴본다. 기울기와 잡음 수준을 달리한 여덟 가지 설정을 통해 $r$가 언제 크고 언제 작은지에 대한 직관을 쌓는다.

---

## 선형모형

단순선형회귀를 생각하자.

$$
Y = \beta_1 + \beta_2 X + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2)
$$

$X$와 $Y$의 Pearson 상관은 기울기 $\beta_2$와 오차 척도 $\sigma$ 모두에 의존한다. 실제로 $X$의 분포를 고정하면 모집단 상관은 다음과 같다.

$$
\rho_{XY} = \frac{\beta_2 \, \sigma_X}{\sqrt{\beta_2^2 \, \sigma_X^2 + \sigma^2}}
$$

이 식은 두 가지 중요한 관계를 드러낸다.

1. **$|\beta_2|$가 커지면 $|\rho|$가 커진다**: 참 기울기가 가파를수록 잡음 대비 신호가 강하다.
2. **$\sigma$가 커지면 $|\rho|$가 작아진다**: 잡음이 커질수록 선형 신호가 묻힌다.

다만 이 두 문장은 **다른 하나를 고정했을 때**에만 성립한다. 뒤에서 보겠지만 결정적인 것은 $\beta_2$나 $\sigma$ 각각이 아니라 세 값 $\beta_2$, $\sigma_X$, $\sigma$가 결합해 만드는 신호 대 잡음비이다.

---

## 모의실험 설정

기울기와 오차 척도를 달리한 여덟 가지 설정에서 자료를 생성한다.

```python
import numpy as np

np.random.seed(42)
DATA_SIZE = 100

CONFIGS = [
    # (beta1, beta2, error_scale)
    (2, 0.05, 1),    # tiny slope, low noise
    (2, -0.6, 1),    # moderate negative slope
    (2, 1.0,  1),    # unit slope
    (2, 3.0,  1),    # steep slope, low noise
    (2, 3.0,  3),    # steep slope, moderate noise
    (2, 3.0, 10),    # steep slope, high noise
    (2, 3.0, 20),    # steep slope, very high noise
    (2, 3.0, 50),    # steep slope, extreme noise
]


def generate(beta1, beta2, error_scale, n=DATA_SIZE):
    x = np.random.randint(1, n, n).astype(float)
    y = beta1 + beta2 * x + error_scale * np.random.randn(n)
    r = np.corrcoef(x, y)[0, 1]
    return x, y, r
```

앞의 네 설정은 $\sigma = 1$로 고정하고 기울기를 바꾸며, 뒤의 네 설정은 $\beta_2 = 3$으로 고정하고 잡음을 키운다.

!!! note "$X$의 산포에 주목하라"
    `np.random.randint(1, n, n)`은 $1$부터 $99$까지의 정수를 뽑으므로 $\sigma_X \approx 28.6$으로 **매우 크다**. 위 공식에서 $\sigma_X$는 기울기와 곱해져 신호의 크기를 결정하므로, 이 큰 산포가 아래 결과 전체를 좌우한다.

---

## 패널 그림 그리기

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(20, 9))

for idx, (b1, b2, es) in enumerate(CONFIGS):
    row, col = divmod(idx, 4)
    ax = axes[row, col]
    x, y, r = generate(b1, b2, es)

    ax.scatter(x, y, alpha=0.5, s=15, edgecolors="grey")
    xs = np.sort(x)
    ax.plot(xs, b1 + b2 * xs, color="#FA954D", lw=2, alpha=0.8)
    ax.set_title(f"Y = {b1} + {b2}X + {es}u", fontsize=10)
    ax.annotate(f"r = {r:.3f}", xy=(0.05, 0.9),
                xycoords="axes fraction", fontsize=11,
                bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))

fig.suptitle("How Slope and Noise Affect Pearson Correlation",
             fontsize=14, y=1.01)
plt.tight_layout()
plt.show()
```

---

## 기울기를 바꿀 때(잡음 고정)

$\sigma = 1$로 고정하고 $\sigma_X \approx 28.6$일 때, 위 코드가 내놓는 표본 $r$와 공식이 주는 이론값 $\rho$는 다음과 같다.

| $\beta_2$ | 표본 $r$ | 이론 $\rho$ | 설명 |
|:---:|:---:|:---:|:---|
| 0.05 | 0.822 | 0.819 | 기울기 자체는 아주 작지만 $\beta_2 \sigma_X \approx 1.43$이 $\sigma = 1$보다 크므로 상관은 이미 높다 |
| $-0.6$ | $-0.998$ | $-0.998$ | 뚜렷한 하락 추세. 부호만 음수일 뿐 강도는 거의 완전하다 |
| 1.0 | 0.999 | 0.999 | 신호가 잡음을 압도한다 |
| 3.0 | 1.000 | 1.000 | 점들이 직선에 거의 붙어 있다 |

여기서 얻을 교훈이 중요하다. "$\beta_2 = 0.05$이니 기울기가 거의 평평하고 따라서 $r$도 0에 가까울 것"이라는 예상은 **틀린다**. 기울기가 작아도 $X$의 산포가 크면 $X$가 만들어 내는 $Y$의 변동 폭($\beta_2 \sigma_X$)은 잡음보다 클 수 있다. 그림에서 첫 패널의 세로축 눈금이 매우 좁다는 점을 확인하면 이해가 쉽다.

---

## 잡음을 바꿀 때(기울기 고정)

$\beta_2 = 3$으로 고정하면

| $\sigma$ | 표본 $r$ | 이론 $\rho$ | 설명 |
|:---:|:---:|:---:|:---|
| 1 | 1.000 | 1.000 | 잡음이 거의 없어 선형 패턴이 완벽하다 |
| 3 | 0.999 | 0.999 | 점이 조금 퍼지지만 추세는 그대로다 |
| 10 | 0.993 | 0.993 | 산점도의 띠가 눈에 띄게 두꺼워진다 |
| 20 | 0.977 | 0.974 | 흩어짐이 커지지만 추세는 여전히 명백하다 |
| 50 | 0.840 | 0.864 | 잡음이 극단적이지만 $\beta_2 \sigma_X \approx 85.7$이 $\sigma = 50$보다 여전히 크므로 상관은 강하게 남는다 |

$\sigma = 50$이라는 "극단적" 잡음조차 $r$를 0 근처로 끌어내리지 못한다. $r$를 0에 가깝게 만들려면 $\sigma$가 $\beta_2 \sigma_X \approx 85.7$보다 훨씬 커야 한다(예: $\sigma = 500$이면 $\rho \approx 0.169$).

---

## 해석

상관계수 $r$는 선형모형에서 **신호 대 잡음비**를 반영한다. 구체적으로 결정계수 $r^2$는 $X$가 설명하는 $Y$의 분산 비율이다.

$$
r^2 = \frac{\beta_2^2 \, \sigma_X^2}{\beta_2^2 \, \sigma_X^2 + \sigma^2}
$$

이는 신호 대 잡음비 $\text{SNR} = \beta_2^2 \sigma_X^2 / \sigma^2$로 다시 쓸 수 있다.

$$
r^2 = \frac{\text{SNR}}{1 + \text{SNR}}
$$

$\text{SNR} \to \infty$이면 $r^2 \to 1$이고, $\text{SNR} \to 0$이면 $r^2 \to 0$이다. 이 하나의 양이 기울기 효과와 잡음 효과를 통합한다. 앞의 두 표에서 본 "놀라운" 결과도 모두 이 식 하나로 설명된다. 기울기가 작아도 $\sigma_X$가 크면 SNR이 크고, 잡음이 커도 $\beta_2 \sigma_X$가 그보다 크면 SNR은 여전히 크다.

---

## 연습문제

**연습문제 1.**
모집단 상관 공식을 써서 $\beta_2 = 3$, $\sigma = 10$, $X \sim \text{Uniform}(1, 100)$일 때의 이론값 $\rho_{XY}$를 계산하라. 모의값과 비교하라. (힌트: $X \sim \text{Uniform}(a,b)$이면 $\text{Var}(X) = (b-a)^2/12$이다.)

??? success "연습문제 1 풀이"

    $X \sim \text{Uniform}(1, 100)$이면 분산은

    $$
    \sigma_X^2 = \frac{(100 - 1)^2}{12} = \frac{9801}{12} = 816.75, \quad \sigma_X \approx 28.579
    $$

    이론 상관은

    $$
    \rho = \frac{\beta_2 \sigma_X}{\sqrt{\beta_2^2 \sigma_X^2 + \sigma^2}} = \frac{3 \times 28.579}{\sqrt{9 \times 816.75 + 100}} = \frac{85.737}{\sqrt{7350.75 + 100}} = \frac{85.737}{\sqrt{7450.75}} \approx \frac{85.737}{86.318} \approx 0.9933
    $$

    ```python
    import numpy as np
    np.random.seed(42)
    x = np.random.uniform(1, 100, 1000)
    y = 2 + 3 * x + 10 * np.random.randn(1000)
    print(f"Simulated r = {np.corrcoef(x, y)[0, 1]:.4f}")   # 0.9935
    print("Theoretical rho = 0.9933")
    ```

    모의값 $0.9935$는 이론값 $0.9933$과 거의 일치한다. $\square$

---

**연습문제 2.**
$X \perp \varepsilon$인 $Y = \beta_1 + \beta_2 X + \varepsilon$에 대해, 공분산과 분산의 정의에서 출발하여 $\rho_{XY} = \frac{\beta_2 \sigma_X}{\sqrt{\beta_2^2 \sigma_X^2 + \sigma^2}}$를 유도하라.

??? success "연습문제 2 풀이"

    $Y = \beta_1 + \beta_2 X + \varepsilon$이고 $X \perp \varepsilon$이므로

    $$
    \text{Cov}(X, Y) = \text{Cov}(X, \beta_1 + \beta_2 X + \varepsilon) = \beta_2 \text{Var}(X) = \beta_2 \sigma_X^2
    $$

    여기서 공분산의 쌍선형성과 독립성에 의한 $\text{Cov}(X, \varepsilon) = 0$을 썼다.

    $$
    \text{Var}(Y) = \text{Var}(\beta_2 X + \varepsilon) = \beta_2^2 \sigma_X^2 + \sigma^2
    $$

    따라서

    $$
    \rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sqrt{\text{Var}(Y)}} = \frac{\beta_2 \sigma_X^2}{\sigma_X \sqrt{\beta_2^2 \sigma_X^2 + \sigma^2}} = \frac{\beta_2 \sigma_X}{\sqrt{\beta_2^2 \sigma_X^2 + \sigma^2}}
    $$

    $\square$

---

**연습문제 3.**
$\sigma_X$와 $\sigma$가 고정되어 있을 때 $r^2 = 0.5$가 되는(즉 신호가 분산의 정확히 절반을 설명하는) $\beta_2$를 구하라. 답을 $\sigma_X$와 $\sigma$로 표현하라.

??? success "연습문제 3 풀이"

    공식에 $r^2 = 0.5$를 대입하면

    $$
    \frac{\beta_2^2 \sigma_X^2}{\beta_2^2 \sigma_X^2 + \sigma^2} = 0.5
    $$

    양변에 분모를 곱하면

    $$
    2\beta_2^2 \sigma_X^2 = \beta_2^2 \sigma_X^2 + \sigma^2
    $$

    $$
    \beta_2^2 \sigma_X^2 = \sigma^2
    $$

    $$
    \beta_2 = \pm\frac{\sigma}{\sigma_X}
    $$

    기울기가 오차 표준편차와 설명변수 표준편차의 비와 같을 때 $Y$의 분산의 정확히 절반이 $X$로 설명된다. 신호와 잡음이 똑같이 기여하는 "손익분기점"이다. 앞 절의 설정에서는 $\sigma_X \approx 28.6$이므로 $\sigma = 1$일 때 손익분기 기울기는 $\beta_2 \approx 0.035$에 불과하다. $\beta_2 = 0.05$에서 이미 $r = 0.82$가 나온 이유가 여기 있다. $\square$

---

**연습문제 4.**
행은 서로 다른 기울기($\beta_2 \in \{0.5, 2, 5\}$), 열은 서로 다른 잡음 수준($\sigma \in \{1, 5, 20\}$)에 대응하는 $3 \times 3$ 패널 그림을 만들어라. 각 패널에 표본 $r$를 주석으로 달고 그 패턴을 논하라.

??? success "연습문제 4 풀이"

    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(42)
    slopes = [0.5, 2, 5]
    noises = [1, 5, 20]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i, b2 in enumerate(slopes):
        for j, sigma in enumerate(noises):
            ax = axes[i, j]
            x = np.random.uniform(1, 50, 100)
            y = 1 + b2 * x + sigma * np.random.randn(100)
            r = np.corrcoef(x, y)[0, 1]

            ax.scatter(x, y, s=10, alpha=0.5)
            xs = np.sort(x)
            ax.plot(xs, 1 + b2 * xs, 'r-', lw=2)
            ax.set_title(f"b2={b2}, sigma={sigma}\nr={r:.3f}", fontsize=9)

    plt.suptitle("Slope vs Noise: Effect on r", fontsize=14)
    plt.tight_layout()
    plt.show()
    ```

    표본 $r$(괄호 안은 $\sigma_X = 49/\sqrt{12} \approx 14.15$로 계산한 이론값):

    | $\beta_2 \backslash \sigma$ | 1 | 5 | 20 |
    |:---:|:---:|:---:|:---:|
    | 0.5 | 0.992 (0.990) | 0.857 (0.817) | 0.324 (0.333) |
    | 2 | 0.999 (0.999) | 0.987 (0.985) | 0.823 (0.817) |
    | 5 | 1.000 (1.000) | 0.997 (0.998) | 0.955 (0.962) |

    패턴은 신호 대 잡음비를 따른다. 행을 내려갈수록(기울기가 커질수록) $|r|$가 커지고, 열을 오른쪽으로 갈수록(잡음이 커질수록) $|r|$가 작아진다. 왼쪽 아래 칸(큰 기울기, 작은 잡음)은 $r = 1.000$으로 사실상 완전한 상관이다. 반대편인 오른쪽 위 칸(작은 기울기, 큰 잡음)은 $r = 0.324$로 가장 작지만 그래도 0은 아니다. $\beta_2 = 0.5$, $\sigma = 20$에서 $\text{SNR} = 0.25 \times 200.1 / 400 = 0.125$이고, 따라서 $r^2 = 0.125/1.125 = 0.111$, $|r| \approx 0.33$이다. 표의 아홉 칸 전부가 공식 하나로 예측된다. $\square$

---

**연습문제 5.**
$\text{SNR} = \beta_2^2 \sigma_X^2 / \sigma^2$일 때 $r^2 = \frac{\text{SNR}}{1 + \text{SNR}}$임을 증명하라. 이어서 $\text{SNR} = \frac{r^2}{1 - r^2}$임을 보이고 이 역관계를 해석하라.

??? success "연습문제 5 풀이"

    $\rho_{XY}$ 공식에서

    $$
    r^2 = \rho^2 = \frac{\beta_2^2 \sigma_X^2}{\beta_2^2 \sigma_X^2 + \sigma^2}
    $$

    분자와 분모를 $\sigma^2$로 나누면

    $$
    r^2 = \frac{\beta_2^2 \sigma_X^2 / \sigma^2}{\beta_2^2 \sigma_X^2 / \sigma^2 + 1} = \frac{\text{SNR}}{\text{SNR} + 1}
    $$

    이를 뒤집어 SNR에 대해 풀면

    $$
    r^2 (\text{SNR} + 1) = \text{SNR}
    $$

    $$
    r^2 = \text{SNR}(1 - r^2)
    $$

    $$
    \text{SNR} = \frac{r^2}{1 - r^2}
    $$

    해석: $r^2 / (1 - r^2)$는 설명된 분산 대 설명되지 않은 분산의 비이다. $r^2 = 0.5$이면 $\text{SNR} = 1$(신호와 잡음이 같다), $r^2 = 0.9$이면 $\text{SNR} = 9$(신호가 잡음보다 아홉 배 강하다)이다. 이 역공식은 관측된 상관을 신호 대 잡음비로 바꿀 때 유용하다. $\square$
