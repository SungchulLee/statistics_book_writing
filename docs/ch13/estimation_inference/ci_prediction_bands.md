# 신뢰띠와 예측띠

## 개요

이 페이지는 단순선형회귀에서 평균반응 $E[y \mid x]$의 신뢰구간과 새 관측값 $y \mid x$의 예측구간을 설명하고 구현한다. 알려진 모형에서 인공자료를 생성하고 OLS를 적합한 뒤 두 띠를 함께 그려, 회귀직선에 대한 불확실성과 개별 예측에 대한 불확실성이 어떻게 다른지 보인다.

## 수학적 배경

단순선형회귀 모형을 생각하자.

$$
y_i = \beta_0 + \beta_1 x_i + \varepsilon_i, \qquad \varepsilon_i \overset{\text{iid}}{\sim} N(0, \sigma^2).
$$

새 설명변수 값 $x_0$이 주어지면 적합값은 $\hat{y}_0 = \hat{\beta}_0 + \hat{\beta}_1 x_0$이다. 두 종류의 구간이 있다.

**$E[y \mid x_0]$의 신뢰구간**(평균반응):

$$
\hat{y}_0 \pm t^*_{n-2} \cdot s \sqrt{\frac{1}{n} + \frac{(x_0 - \bar{x})^2}{\sum_{i=1}^n (x_i - \bar{x})^2}}
$$

**$x_0$에서 새 $y$의 예측구간**:

$$
\hat{y}_0 \pm t^*_{n-2} \cdot s \sqrt{1 + \frac{1}{n} + \frac{(x_0 - \bar{x})^2}{\sum_{i=1}^n (x_i - \bar{x})^2}}
$$

예측구간은 평균 추정의 불확실성과 줄일 수 없는 잡음 $\sigma^2$을 모두 반영하므로 언제나 더 넓다.

### 자료 생성

<div class="codebox" markdown>

#### 예제 1. 자료 만들기 { .eg }

```python
import numpy as np

def generate_data(n, sigma, seed=0):
    """자료를 만든다. 참 모형은 y = 1 + 2x + 잡음 이다."""
    np.random.seed(seed)
    x = np.random.randn(n, 1)
    y = 1 + 2 * x + sigma * np.random.randn(n, 1)
    return x, y
```

</div>

### 회귀직선 추정

<div class="codebox" markdown>

#### 예제 2. 회귀직선 추정 { .eg }

```python
def estimate_regression_line(x, y):
    """상관계수 공식으로 기울기와 절편을 구한다.

    beta_hat = r * (s_y / s_x) 이다. 최소제곱해와 정확히 같은 값이지만,
    회귀계수가 상관계수를 척도만 바꿔 옮긴 것임이 이 꼴에서 드러난다.
    """
    x_bar = x.mean()
    y_bar = y.mean()
    s_x = x.std(ddof=1)
    s_y = y.std(ddof=1)
    r = np.corrcoef(np.concatenate([x, y], axis=1), rowvar=False)[1, 0]
    beta_hat = r * s_y / s_x
    y_hat = beta_hat * (x - x_bar) + y_bar
    return y_hat, beta_hat, y_bar, x_bar
```

</div>

### 잔차분산

<div class="codebox" markdown>

#### 예제 3. 잔차분산 구하기 { .eg }

```python
def calculate_residual_variance(y, y_hat, n):
    """잔차분산 s^2 을 구한다.

    n 이 아니라 n-2 로 나눈다. 절편과 기울기 둘을 자료에서 추정하느라
    자유도를 둘 잃었기 때문이다.
    """
    s_square = np.sum((y - y_hat) ** 2) / (n - 2)
    s = np.sqrt(s_square)
    return s_square, s
```

</div>

### 신뢰구간과 예측구간

<div class="codebox" markdown>

#### 예제 4. 두 구간 계산 { .eg }

```python
from scipy import stats

def confidence_intervals(x, y_hat, beta_hat, x_bar, y_bar, n, s):
    """평균반응의 신뢰구간과 개별관측의 예측구간을 함께 구한다.

    두 식의 차이는 근호 안의 1 뿐이다. 평균을 맞히는 데는 추정오차만 들지만,
    개별 관측을 맞히려면 잡음 자체의 분산이 더 얹힌다.
    """
    x0 = np.linspace(x.min(), x.max(), 20)
    y0_hat = beta_hat * (x0 - x_bar) + y_bar
    t_val = stats.t(n - 2).ppf(0.975)
    ss_x = np.sum((x - x_bar) ** 2)

    # 평균반응의 신뢰구간. x 의 평균에서 멀어질수록 넓어진다.
    margin = t_val * s * np.sqrt((1 / n) + (x0 - x_bar) ** 2 / ss_x)
    lower = y0_hat - margin
    upper = y0_hat + margin

    # 예측구간. 근호 안의 1 이 잡음 몫이며, n 을 키워도 사라지지 않는다.
    margin2 = t_val * s * np.sqrt(1 + (1 / n) + (x0 - x_bar) ** 2 / ss_x)
    lower2 = y0_hat - margin2
    upper2 = y0_hat + margin2

    return x0, lower, upper, lower2, upper2
```

</div>

## 해석

- **신뢰띠**(CI)는 참 회귀직선이 어디에 있는지에 대한 불확실성을 수량화한다. $x = \bar{x}$에서 가장 좁고 $x_0$이 자료의 중심에서 멀어질수록 넓어진다.
- **예측띠**(PI)는 새로운 관측값 하나가 어디에 떨어질지에 대한 불확실성을 수량화한다. 제곱근 안에 $+1$이 더 들어 있으므로 언제나 신뢰띠보다 넓다.
- $n \to \infty$이면 신뢰띠의 폭은 0으로 줄어들지만(직선이 완벽하게 추정된다) 예측띠는 $\hat{y}_0 \pm t^* \cdot s$로 수렴하며, 이는 줄일 수 없는 잡음을 반영한다.
- 두 띠의 "나비넥타이" 모양은 지렛대 원리를 보여준다. 예측은 관측된 설명변수 값들의 중심 근처에서 가장 믿을 만하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span> $n = 100$, $\sigma = 3$인 인공자료에서 $E[y \mid x_0 = 0]$의 90% 신뢰구간을 계산하고 95% 구간과 비교하라.

</div>

??? success "풀이"

    표준정규 설명변수에서 $x_0 = 0$은 대략 $\bar{x}$이다.

    ```python
    import numpy as np
    from scipy import stats

    n, s = 100, 3.0
    t_90 = stats.t(n - 2).ppf(0.95)
    t_95 = stats.t(n - 2).ppf(0.975)

    margin_90 = t_90 * s * np.sqrt(1 / n)
    margin_95 = t_95 * s * np.sqrt(1 / n)
    print(f"t_90 = {t_90:.4f}, margin_90 = {margin_90:.4f}")
    print(f"t_95 = {t_95:.4f}, margin_95 = {margin_95:.4f}")
    print(f"ratio = {margin_90 / margin_95:.4f}")
    ```

    출력:

    ```
    t_90 = 1.6606, margin_90 = 0.4982
    t_95 = 1.9845, margin_95 = 0.5953
    ratio = 0.8368
    ```

    90% 구간이 95% 구간의 0.837배로 좁다. 이 비는 $t$ 임계값의 비 $t_{0.95}/t_{0.975} = 1.6606/1.9845$와 정확히 같다. 신뢰수준만 바꾸면 구간의 **폭만** 비례해서 달라진다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 어떤 $x_0$에서든 예측구간이 신뢰구간보다 항상 넓은 이유를 대수적으로 설명하라.

</div>

??? success "풀이"

    신뢰구간 반폭의 제곱은 다음에 비례한다.

    $$
    \frac{1}{n} + \frac{(x_0 - \bar{x})^2}{S_{xx}},
    $$

    반면 예측구간은

    $$
    1 + \frac{1}{n} + \frac{(x_0 - \bar{x})^2}{S_{xx}}.
    $$

    예측구간에는 $1$이라는 항이 더 붙는다(표준화한 뒤 $\mathrm{Var}(\varepsilon_{\text{new}})/s^2 = 1$에 해당한다). 따라서 제곱근 안의 값이 예측구간에서 항상 더 크다. 임계값과 $s$가 같으므로 예측구간이 언제나 더 넓다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> $n = 30$, $\sigma = 1$인 모형의 띠를 그리도록 코드를 고쳐라. 원래의 $n = 100$, $\sigma = 3$인 경우와 띠의 폭을 비교하면 어떠한가?

</div>

??? success "풀이"

    ```python
    x30, y30 = generate_data(30, 1)
    y_hat30, beta30, ybar30, xbar30 = estimate_regression_line(x30, y30)
    _, s30 = calculate_residual_variance(y30, y_hat30, 30)
    ```

    신뢰구간의 폭은 $s/\sqrt{n}$에 의존한다. $n=30$, $\sigma=1$이면 $s \approx 1$이고 $s/\sqrt{30} \approx 0.18$인 반면, 원래는 $s \approx 3$이고 $s/\sqrt{100} = 0.3$이었다. 새 신뢰구간이 더 좁다. 예측구간의 폭은 $s$가 지배하므로 $\sigma=1$일 때의 예측구간은 $\sigma=3$일 때보다 훨씬 좁다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> $y_{\text{new}} = \beta_0 + \beta_1 x_0 + \varepsilon_{\text{new}}$이고 $\varepsilon_{\text{new}}$이 훈련자료와 독립일 때, 예측오차 $\hat{y}_0 - y_{\text{new}}$의 분산을 유도하라.

</div>

??? success "풀이"

    예측오차는 $\hat{y}_0 - y_{\text{new}} = (\hat{y}_0 - E[y \mid x_0]) - \varepsilon_{\text{new}}$이다. $\hat{y}_0$은 훈련자료에만 의존하고 $\varepsilon_{\text{new}}$은 그와 독립이므로

    $$
    \mathrm{Var}(\hat{y}_0 - y_{\text{new}}) = \mathrm{Var}(\hat{y}_0) + \mathrm{Var}(\varepsilon_{\text{new}}) = \sigma^2\!\left(\frac{1}{n} + \frac{(x_0 - \bar{x})^2}{S_{xx}}\right) + \sigma^2.
    $$

    $\sigma^2$을 묶어 내면

    $$
    \mathrm{Var}(\hat{y}_0 - y_{\text{new}}) = \sigma^2\!\left(1 + \frac{1}{n} + \frac{(x_0 - \bar{x})^2}{S_{xx}}\right).
    $$

    이것이 ($\sigma$를 $s$로 바꾸면) 예측구간 공식의 제곱근 안 표현과 정확히 일치한다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> $x_0 = \bar{x}$에서 평균의 신뢰구간이 $\bar{y} \pm t^*_{n-2} \cdot s / \sqrt{n}$으로 간단해짐을 보여라. 입문 통계학에서 배우는 모평균의 신뢰구간과 비교하라.

</div>

??? success "풀이"

    $x_0 = \bar{x}$에서는 $(x_0 - \bar{x})^2/S_{xx} = 0$이므로 신뢰구간이

    $$
    \hat{y}_0 \pm t^*_{n-2} \cdot s \sqrt{\frac{1}{n}} = \bar{y} \pm \frac{t^*_{n-2} \cdot s}{\sqrt{n}}
    $$

    가 된다. $\hat{y}_0 = \hat{\beta}_0 + \hat{\beta}_1 \bar{x} = \bar{y}$이기 때문이다. 이는 모평균의 신뢰구간 $\bar{y} \pm t^*_{n-1} \cdot s/\sqrt{n}$과 같은 형태이며, 다만 회귀에서는 모수를 두 개 추정하므로 자유도가 $n-1$이 아니라 $n-2$이다. $\square$

---

## 정리하며

두 띠를 **함께 그리면** 차이가 분명해진다.

- **신뢰 띠는 회귀직선의 위치에 대한 불확실성**이고, **예측 띠는 새 관측 하나에 대한 불확실성**이다.
- **예측 띠가 훨씬 넓다.** 차이가 $\sigma^2$ 만큼이며, $n$ 을 아무리 늘려도 이 폭은 줄지 않는다. **줄어드는 것은 신뢰 띠뿐이다.**
- **둘 다 $\bar x$ 에서 가장 좁다.** 모래시계 모양이며, 자료 중심에서 멀어질수록 추정이 불안해진다.
- **혼동이 실무에서 자주 일어난다.** "이 모형의 95% 구간"이라 하면 대개 예측 띠를 뜻해야 하는데 신뢰 띠를 그려 놓는 일이 흔하다. **개별 사례를 예측하는 문제라면 예측 띠가 옳다.**
- **띠가 자료 범위 안에서만 의미가 있다.** 밖으로 나가면 넓어지기는 하지만 모형 오지정의 위험은 반영하지 못한다.

다음 절 **RSS 곡면 시각화**로 넘어간다.
