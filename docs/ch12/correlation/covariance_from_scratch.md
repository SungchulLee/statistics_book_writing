# 공분산 밑바닥부터 만들기

## 개요

이 페이지는 공분산과 Pearson 상관계수를 제1원리에서 출발해 단계별로 만들어 본다. 공통의 거시경제 추세를 공유하는 두 주(州)의 가격 자료를 모의로 생성한 뒤, 각 양을 손으로 계산하고 라이브러리 구현과 대조해 확인하며, 상관된 시계열이 왜 인과를 뜻하지 않는지 보인다.

---

## 표본공분산

짝지어진 관측값 $(x_1, y_1), \ldots, (x_n, y_n)$에 대해 **표본공분산**은 다음과 같이 정의된다.

$$
\text{Cov}(X, Y) = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})
$$

여기서 $\bar{x}$와 $\bar{y}$는 표본평균이다. 분모의 $n - 1$(Bessel 보정)은 모집단 공분산의 불편추정량을 준다.

각 항 $(x_i - \bar{x})(y_i - \bar{y})$를 **편차곱**이라 한다.

- $x_i$와 $y_i$가 각자의 평균에서 **같은 방향**으로 벗어나면 양수이다.
- **반대 방향**으로 벗어나면 음수이다.

양의 곱이 우세하면 공분산이 양수가 되고, 이는 두 변수가 함께 움직이는 경향이 있음을 뜻한다.

---

## 공분산에서 Pearson 상관계수로

Pearson 상관계수는 공분산을 두 표준편차의 곱으로 표준화한 것이다.

$$
r = \frac{\text{Cov}(X, Y)}{s_X \, s_Y}
$$

여기서 $s_X = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2}$는 표본표준편차이다($s_Y$도 같다). 이 표준화 덕분에 $-1 \le r \le 1$이 보장된다.

---

## 단계별 구현

<div class="codebox" markdown>

### 예제 1. 공분산과 상관을 단계별로 구현 { .eg }

```python
import numpy as np

def covariance_step_by_step(x, y):
    """표본공분산을 구하고, 중간 계산인 편차까지 함께 돌려준다.

        편차를 돌려주는 까닭은 뒤에서 편차곱을 막대로 그려 보이기 위함이다.
        """
    n = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    x_dev = x - x_mean
    y_dev = y - y_mean
    # n이 아니라 n-1로 나눈다. 베셀 보정이며, 분산에서와 같은 이유다.
    # 편차를 참 평균이 아니라 표본평균에서 쟀기 때문에 자유도 하나를 잃는다.
    cov = np.sum(x_dev * y_dev) / (n - 1)
    return cov, x_dev, y_dev

def pearson_r_step_by_step(x, y):
    """피어슨 상관계수를 정의대로 구한다.

        공분산을 두 표준편차의 곱으로 나눈다. 이 나눗셈이 단위를 없애므로
        r 은 -1 과 1 사이에 갇힌 값이 된다.
        """
    cov, _, _ = covariance_step_by_step(x, y)
    # ddof=1로 맞춰야 한다. 공분산이 n-1로 나눈 값이므로
    # 표준편차도 같은 규약을 써야 두 n-1이 약분되어 r이 척도와 무관해진다.
    sx = x.std(ddof=1)
    sy = y.std(ddof=1)
    return cov / (sx * sy)
```

</div>

---

## 자료 생성

공통의 하락 추세를 공유하지만 잡음은 서로 독립인 두 주(CA와 NY)의 주간 가격을 모의로 만든다.

$$
\text{CA}_t = 248 + \text{trend}_t + \varepsilon_t^{(\text{CA})}, \qquad
\text{NY}_t = 350 + 0.8\,\text{trend}_t + \varepsilon_t^{(\text{NY})}
$$

여기서 $\text{trend}_t$는 48주에 걸쳐 0에서 $-12$까지 선형으로 감소하고, $\varepsilon_t^{(\text{CA})} \sim \mathcal{N}(0, 0.5^2)$, $\varepsilon_t^{(\text{NY})} \sim \mathcal{N}(0, 0.6^2)$이다.

<div class="codebox" markdown>

### 예제 2. 가격 자료 만들기 { .eg }

```python
np.random.seed(42)
WEEKS = 48

# 두 주의 가격에 공통으로 실릴 하락 추세. 공분산이 커지는 까닭이 바로 이것이다.
trend = np.linspace(0, -12, WEEKS)

CA = 248.0 + trend + np.random.normal(0, 0.5, WEEKS)
NY = 350.0 + trend * 0.8 + np.random.normal(0, 0.6, WEEKS)
```

</div>

---

## 계산과 검증

<div class="codebox" markdown>

### 예제 3. 라이브러리 결과와 맞춰 보기 { .eg }

```python
import pandas as pd

cov, x_dev, y_dev = covariance_step_by_step(CA, NY)
r = pearson_r_step_by_step(CA, NY)

print(f"CA mean     = {CA.mean():.4f}")
print(f"NY mean     = {NY.mean():.4f}")
print(f"Covariance  = {cov:.4f}")
print(f"Pearson r   = {r:.4f}")

# 직접 구한 값이 라이브러리와 맞는지 확인한다. pandas 의 cov 는 ddof=1 이므로
# 위 구현도 n-1 로 나눠야 값이 맞는다.
df = pd.DataFrame({"CA": CA, "NY": NY})
print(f"pandas cov  = {df['CA'].cov(df['NY']):.4f}")
print(f"pandas corr = {df['CA'].corr(df['NY']):.4f}")
print(f"numpy corr  = {np.corrcoef(CA, NY)[0, 1]:.4f}")
```

출력:

```text
CA mean     = 241.8974
NY mean     = 345.1893
Covariance  = 10.5691
Pearson r   = 0.9753
pandas cov  = 10.5691
pandas corr = 0.9753
numpy corr  = 0.9753
```

세 방법이 모두 같은 값을 내놓으므로 밑바닥부터 만든 구현이 옳음을 확인할 수 있다.

</div>

---

## 시각화

세 개의 패널이 이야기 전체를 들려준다.

<div class="codebox" markdown>

### 예제 4. 세 그림으로 이해하기 { .eg }

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# 왼쪽: 산점도와 회귀직선
axes[0].scatter(CA, NY, alpha=0.6, edgecolors="grey")
z = np.polyfit(CA, NY, 1)
axes[0].plot(np.sort(CA), np.polyval(z, np.sort(CA)),
             color="red", linewidth=2)
axes[0].set_xlabel("CA Price (\\$)")
axes[0].set_ylabel("NY Price (\\$)")
axes[0].set_title(f"Scatter (r = {r:.3f})")

# 가운데: 주마다의 편차곱. 공분산은 이 막대들의 평균이다.
# 파란 막대(양)가 빨간 막대(음)를 압도하면 공분산이 양이 된다.
products = x_dev * y_dev
colours = ["steelblue" if p > 0 else "salmon" for p in products]
axes[1].bar(range(WEEKS), products, color=colours, edgecolor="white")
axes[1].axhline(0, color="black", linewidth=0.5)
axes[1].set_xlabel("Week")
axes[1].set_ylabel("$(x - \\bar{x})(y - \\bar{y})$")
axes[1].set_title("Deviation Products")

# 오른쪽: 두 시계열. 함께 내려가는 모습이 위 편차곱의 부호를 설명한다.
weeks = np.arange(WEEKS)
axes[2].plot(weeks, CA, label="CA", marker="o", markersize=3)
axes[2].plot(weeks, NY, label="NY", marker="s", markersize=3)
axes[2].set_xlabel("Week")
axes[2].set_ylabel("Price (\\$)")
axes[2].set_title("Common Trend")
axes[2].legend()

plt.tight_layout()
plt.show()
```

![공분산의 시각적 분해](./img/covariance_from_scratch_129.png)

세 번째 그림에서 두 계열이 나란히 내려가는 것이 보인다. 이 공통 추세가 곧 상관의 원천이다.

</div>

---

## 해석

CA와 NY 가격 사이의 강한 양의 상관($r = 0.975$)은 전적으로 공유된 하락 추세, 곧 교란변수에서 비롯된다. 어느 주의 가격도 다른 주의 가격을 *일으키지* 않는다. **상관은 인과를 뜻하지 않는다**는 원리의 교과서적 예시이다.

편차곱 막대그림을 보면 48개 중 46개가 양수(파랑)이며, 그래서 공분산이 — 따라서 $r$가 — 강하게 양수가 된다. 시계열 패널에서는 두 계열이 함께 하락하는데, 이는 둘 사이의 인과 연결이 아니라 공통 추세가 이끄는 움직임이다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
자료 $(1, 2), (2, 4), (3, 5), (4, 4), (5, 5)$에 대해 공분산과 Pearson $r$를 손으로 계산하라. 편차곱을 포함한 모든 중간 단계를 보여라.

</div>

??? success "풀이"

    표본평균은 $\bar{x} = 3$, $\bar{y} = 4$이다.

    | $i$ | $x_i$ | $y_i$ | $x_i - \bar{x}$ | $y_i - \bar{y}$ | $(x_i - \bar{x})(y_i - \bar{y})$ |
    |:---:|:---:|:---:|:---:|:---:|:---:|
    | 1 | 1 | 2 | $-2$ | $-2$ | 4 |
    | 2 | 2 | 4 | $-1$ | 0 | 0 |
    | 3 | 3 | 5 | 0 | 1 | 0 |
    | 4 | 4 | 4 | 1 | 0 | 0 |
    | 5 | 5 | 5 | 2 | 1 | 2 |

    $$
    \text{Cov}(X, Y) = \frac{4 + 0 + 0 + 0 + 2}{5 - 1} = \frac{6}{4} = 1.5
    $$

    $$
    s_X = \sqrt{\frac{4 + 1 + 0 + 1 + 4}{4}} = \sqrt{2.5} \approx 1.5811
    $$

    $$
    s_Y = \sqrt{\frac{4 + 0 + 1 + 0 + 1}{4}} = \sqrt{1.5} \approx 1.2247
    $$

    $$
    r = \frac{1.5}{1.5811 \times 1.2247} \approx \frac{1.5}{1.9365} \approx 0.7746
    $$

    $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
표본공분산 $\frac{1}{n-1}\sum(x_i - \bar{x})(y_i - \bar{y})$가 모집단 공분산 $\text{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$의 불편추정량임을 증명하라.

</div>

??? success "풀이"

    $(X_1, Y_1), \ldots, (X_n, Y_n)$이 독립이고 동일한 분포를 따르며 $\mathbb{E}[X] = \mu_X$, $\mathbb{E}[Y] = \mu_Y$, $\text{Cov}(X, Y) = \sigma_{XY}$라 하자.

    합을 전개하면

    $$
    \sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y}) = \sum_{i=1}^n X_i Y_i - n\bar{X}\bar{Y}
    $$

    기댓값을 취하면

    $$
    \mathbb{E}\!\left[\sum_{i=1}^n X_i Y_i\right] = n(\sigma_{XY} + \mu_X \mu_Y)
    $$

    $$
    \mathbb{E}[n\bar{X}\bar{Y}] = n\!\left(\frac{\sigma_{XY}}{n} + \mu_X \mu_Y\right) = \sigma_{XY} + n\mu_X \mu_Y
    $$

    따라서

    $$
    \mathbb{E}\!\left[\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})\right] = n\sigma_{XY} + n\mu_X\mu_Y - \sigma_{XY} - n\mu_X\mu_Y = (n-1)\sigma_{XY}
    $$

    $n - 1$로 나누면

    $$
    \mathbb{E}\!\left[\frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})\right] = \sigma_{XY}
    $$

    이로써 분모가 $n - 1$인 표본공분산이 불편임이 확인된다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
$\text{Cov}(X, Y) = \mathbb{E}[XY] - \mathbb{E}[X]\,\mathbb{E}[Y]$임을 보여라. 이 항등식을 써서 $X$와 $Y$가 독립이면 $\text{Cov}(X, Y) = 0$임을 증명하라.

</div>

??? success "풀이"

    정의에서 출발한다.

    $$
    \text{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]
    $$

    전개하면

    $$
    = \mathbb{E}[XY - \mu_Y X - \mu_X Y + \mu_X \mu_Y]
    $$

    $$
    = \mathbb{E}[XY] - \mu_Y \mathbb{E}[X] - \mu_X \mathbb{E}[Y] + \mu_X \mu_Y
    $$

    $$
    = \mathbb{E}[XY] - \mu_X \mu_Y - \mu_X \mu_Y + \mu_X \mu_Y = \mathbb{E}[XY] - \mathbb{E}[X]\,\mathbb{E}[Y]
    $$

    $X \perp Y$이면 독립성에 의해 $\mathbb{E}[XY] = \mathbb{E}[X]\,\mathbb{E}[Y]$이므로

    $$
    \text{Cov}(X, Y) = \mathbb{E}[X]\,\mathbb{E}[Y] - \mathbb{E}[X]\,\mathbb{E}[Y] = 0
    $$

    주의: 역은 일반적으로 성립하지 않는다. 공분산이 0이어도 독립은 아니다(예: $X \sim \mathcal{N}(0,1)$이고 $Y = X^2$이면 $\text{Cov}(X, Y) = 0$이지만 분명히 종속이다). $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
CA와 NY 가격이 공통 성분을 갖지 않도록 모의실험을 고쳐라. 공분산과 $r$를 다시 계산하고, 공통 추세를 없앤 것이 결과를 어떻게 바꾸는지 설명하라.

</div>

??? success "풀이"

    먼저 흔한 오해를 짚고 넘어가자. 두 계열에 **기울기가 다른** 결정론적 선형 추세를 주는 것으로는 공통 성분이 사라지지 않는다.

    ```python
    import numpy as np

    np.random.seed(42)
    WEEKS = 48
    trend_CA = np.linspace(0, -12, WEEKS)
    trend_NY = np.linspace(0, -8, WEEKS)   # 기울기만 다른 추세

    CA = 248.0 + trend_CA + np.random.normal(0, 3, WEEKS)
    NY = 350.0 + trend_NY + np.random.normal(0, 3, WEEKS)

    print(f"Covariance = {np.cov(CA, NY)[0, 1]:.4f}")
    print(f"Pearson r  = {np.corrcoef(CA, NY)[0, 1]:.4f}")
    ```

    출력:

    ```
    Covariance = 10.5231
    Pearson r  = 0.5733
    ```

    잡음을 6배로 키웠는데도 $r = 0.573$으로 여전히 뚜렷하게 양수이다. 이유는 간단하다. 두 결정론적 직선 추세는 서로 상수배 관계이므로 **완전히 공선적**이다. 기울기가 다르다는 것은 독립이라는 뜻이 아니다. 상관이 낮아진 것은 공통 성분이 사라져서가 아니라 잡음이 커져 신호 대 잡음비가 낮아졌기 때문이다.

    공통 성분을 실제로 없애려면 한 계열에서 추세 자체를 빼야 한다.

    ```python
    np.random.seed(42)
    CA = 248.0 + np.linspace(0, -12, WEEKS) + np.random.normal(0, 0.5, WEEKS)
    NY = 350.0 + np.random.normal(0, 0.6, WEEKS)   # 추세 없음

    print(f"Covariance = {np.cov(CA, NY)[0, 1]:.4f}")
    print(f"Pearson r  = {np.corrcoef(CA, NY)[0, 1]:.4f}")
    ```

    출력:

    ```
    Covariance = 0.1348
    Pearson r  = 0.0659
    ```

    이제 $r = 0.066$으로 0에 가깝다(유한표본이므로 정확히 0은 아니다). 원래의 $r = 0.975$는 두 계열이 서로 영향을 주고받아서가 아니라 *공유된* 추세가 만들어 낸 것이었다. 이는 원래의 상관이 교란에 의한 인공물이었음을 다시 한번 확인해 준다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
공분산의 쌍선형성을 증명하라. 상수 $a, b, c, d$와 확률변수 $X, Y, W$에 대해

$$
\text{Cov}(aX + bY,\; cW + d) = ac\,\text{Cov}(X, W) + bc\,\text{Cov}(Y, W)
$$

</div>

??? success "풀이"

    항등식 $\text{Cov}(U, V) = \mathbb{E}[UV] - \mathbb{E}[U]\,\mathbb{E}[V]$를 쓴다.

    $$
    \text{Cov}(aX + bY,\; cW + d)
    = \mathbb{E}[(aX + bY)(cW + d)] - \mathbb{E}[aX + bY]\,\mathbb{E}[cW + d]
    $$

    첫째 항을 전개하면

    $$
    \mathbb{E}[(aX + bY)(cW + d)] = ac\,\mathbb{E}[XW] + ad\,\mathbb{E}[X] + bc\,\mathbb{E}[YW] + bd\,\mathbb{E}[Y]
    $$

    둘째 항을 전개하면

    $$
    -(a\,\mathbb{E}[X] + b\,\mathbb{E}[Y])(c\,\mathbb{E}[W] + d)
    = -ac\,\mathbb{E}[X]\mathbb{E}[W] - ad\,\mathbb{E}[X] - bc\,\mathbb{E}[Y]\mathbb{E}[W] - bd\,\mathbb{E}[Y]
    $$

    둘을 합치면

    $$
    = ac(\mathbb{E}[XW] - \mathbb{E}[X]\mathbb{E}[W]) + bc(\mathbb{E}[YW] - \mathbb{E}[Y]\mathbb{E}[W])
    $$

    $$
    = ac\,\text{Cov}(X, W) + bc\,\text{Cov}(Y, W)
    $$

    상수 $d$가 완전히 사라진다는 점에 주목하라. 확률변수에 상수를 더해도 다른 어떤 변수와의 공분산도 바뀌지 않는다는 사실을 반영한다. $\square$

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
연습문제 3의 항등식 $\operatorname{Cov}=E[XY]-E[X]E[Y]$를 **그대로 코드로 옮기면 위험하다.** 왜 그런지 보여라.

</div>

??? success "풀이"
    **문제는 재난적 상쇄(catastrophic cancellation)**다. $E[XY]$와 $E[X]E[Y]$가 둘 다 크고 거의 같으면, 그 차이에서 **유효숫자가 거의 전부 사라진다.**

    ```python
    import numpy as np

    rng = np.random.default_rng(25001)
    print(f"{'평균 이동량':>12s} {'단순 공식':>16s} {'2단계 공식':>16s} {'상대오차':>12s}")
    for shift in [0, 1e3, 1e6, 1e8, 1e9]:
        x = rng.standard_normal(1000) + shift
        y = rng.standard_normal(1000) * 2 + 0.7 * (x - shift) + shift
        n = len(x)
        naive = ((x * y).mean() - x.mean() * y.mean()) * n / (n - 1)
        two = ((x - x.mean()) * (y - y.mean())).sum() / (n - 1)
        print(f"{shift:12.0e} {naive:16.8f} {two:16.8f} "
              f"{abs(naive - two) / abs(two):12.2e}")
    ```

    ```text
          평균 이동량            단순 공식           2단계 공식         상대오차
           0e+00       0.71369665       0.71369665     1.56e-16
           1e+03       0.76347205       0.76347205     6.90e-11
           1e+06       0.64957536       0.64952755     7.36e-05
           1e+08       0.00000000       0.63626123     1.00e+00
           1e+09       0.00000000       0.76178758     1.00e+00
    ```

    **$10^8$을 더하면 단순 공식이 정확히 0을 낸다.** 참값은 0.636이다.

    | 이동량 | 상대오차 |
    |---|---|
    | 0 | $1.6\times10^{-16}$(기계 정밀도) |
    | $10^6$ | $7.4\times10^{-5}$ |
    | **$10^8$** | **100%** |

    **왜 $10^8$인가.** 배정밀도의 유효숫자가 약 16자리다. $x\approx10^8$이면 $xy\approx10^{16}$이고, 여기서 **0.6 정도의 차이를 읽어야 한다.** $10^{16}$ 대비 $10^0$이므로 **유효숫자가 전부 소진**된다.

    **단정밀도(float32)에서는 훨씬 빨리 무너진다.**

    ```python
    print("\nfloat32 에서")
    for shift in [0, 1e3, 1e5, 1e6]:
        x = (rng.standard_normal(1000) + shift).astype(np.float32)
        y = (rng.standard_normal(1000) * 2 + shift).astype(np.float32)
        naive = float((x * y).mean()) - float(x.mean()) * float(y.mean())
        two = float(((x - x.mean()) * (y - y.mean())).sum() / (len(x) - 1))
        true = float(np.cov(x.astype(np.float64), y.astype(np.float64))[0, 1])
        print(f"  shift={shift:8.0e}: 단순 {naive:14.6f}  2단계 {two:12.6f}  "
              f"참값 {true:10.6f}")
    ```

    ```text

    float32 에서
      shift=   0e+00: 단순       0.032305  2단계     0.032337  참값   0.032337
      shift=   1e+03: 단순       0.010370  2단계    -0.033220  참값  -0.033220
      shift=   1e+05: 단순       0.000549  2단계    -0.024096  참값  -0.024100
      shift=   1e+06: 단순   61440.000000  2단계    -0.031164  참값  -0.031155
    ```

    **$10^6$에서 단순 공식이 61440을 낸다.** 참값이 $-0.031$인데 **부호도 크기도 전부 틀렸다.**

    **$10^3$만 되어도 부호가 뒤집힌다**(0.010 대 $-0.033$).

    **해결책 셋.**

    | 방법 | 특징 |
    |---|---|
    | **2단계**(평균을 먼저 구해 뺀다) | 안정적, 자료를 **두 번** 읽어야 함 |
    | 이동 상수 $K$를 빼고 단순 공식 | 한 번만 읽음, $K$ 선택이 중요 |
    | **웰퍼드 온라인 갱신** | 안정적, **한 번만** 읽음 |

    **웰퍼드 알고리즘**이 최선이다.

    ```python
    def online_cov(xs, ys):
        """한 번의 통과로 안정적으로 공분산을 구한다 (웰퍼드)."""
        n = 0
        mx = my = C = 0.0
        for x, y in zip(xs, ys):
            n += 1
            dx = x - mx
            mx += dx / n
            my += (y - my) / n
            C += dx * (y - my)          # 갱신된 my 를 쓰는 것이 요점
        return C / (n - 1)

    rng = np.random.default_rng(25001)
    x = rng.standard_normal(10_000) + 1e8
    y = rng.standard_normal(10_000) + 0.5 * (x - 1e8) + 1e8
    print(f"\n  단순 공식: {(x * y).mean() - x.mean() * y.mean():.6f}")
    print(f"  2단계:     {((x - x.mean()) * (y - y.mean())).sum() / (len(x) - 1):.6f}")
    print(f"  웰퍼드:    {online_cov(x, y):.6f}")
    print(f"  np.cov:    {np.cov(x, y)[0, 1]:.6f}")
    ```

    ```text

      단순 공식: 0.000000
      2단계:     0.509465
      웰퍼드:    0.509465
      np.cov:    0.509465
    ```

    **웰퍼드가 `np.cov`와 소수점 여섯 자리까지 같다.** 단순 공식은 **정확히 0**을 낸다. 참값 0.509가 통째로 사라졌다.

    **갱신식의 핵심 — $dx$는 옛 평균으로, $y$의 편차는 새 평균으로 계산한다.** 순서를 바꾸면 편향이 생긴다.

    $$
    C_n=C_{n-1}+(x_n-\bar x_{n-1})(y_n-\bar y_n)
    $$

    **언제 문제가 되나.**

    | 자료 | 위험 |
    |---|---|
    | 유닉스 타임스탬프($\approx1.7\times10^9$) | **매우 높음** |
    | 주가(수천~수만) | 중간 |
    | 표준화된 자료 | 낮음 |
    | **센서 원시값**(float32) | **매우 높음** |

    **실무 지침.** `np.cov`와 `pandas`는 이미 안정적인 알고리즘을 쓴다. **직접 구현할 때만 주의**하면 되고, 그럴 일이 있다면 웰퍼드를 쓴다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span>
표본 공분산행렬은 **언제 역행렬을 갖지 못하는가?** 조건을 찾고 확인하라.

</div>

??? success "풀이"
    **핵심 사실.** 변수가 $p$개, 표본이 $n$개일 때

    $$
    \operatorname{rank}(S)\leq\min(n-1,\ p)
    $$

    **$n\leq p$이면 반드시 특이행렬**이다. 평균을 빼면서 자유도 하나를 잃기 때문에 $n-1$이다.

    ```python
    import numpy as np

    rng = np.random.default_rng(25002)
    print(f"{'p (변수 수)':>11s} {'n (표본)':>9s} {'계수(rank)':>11s} "
          f"{'최소 고윳값':>13s} {'역행렬':>8s}")
    for p, n in [(5, 100), (5, 10), (10, 10), (20, 15), (50, 30), (50, 60)]:
        X = rng.standard_normal((n, p))
        S = np.cov(X, rowvar=False)
        ev = np.linalg.eigvalsh(S)
        print(f"{p:11d} {n:9d} {np.linalg.matrix_rank(S):11d} {ev.min():13.2e} "
              f"{'가능' if ev.min() > 1e-10 else '불가':>8s}")
    ```

    ```text
       p (변수 수)    n (표본)    계수(rank)        최소 고윳값      역행렬
              5       100           5      7.03e-01       가능
              5        10           5      2.28e-01       가능
             10        10           9     -2.11e-16       불가
             20        15          14     -3.68e-16       불가
             50        30          29     -1.91e-15       불가
             50        60          50      5.15e-03       가능
    ```

    **$n\leq p$인 세 줄에서 계수가 정확히 $n-1$이다.**

    | $p$ | $n$ | 계수 | 예측 $\min(n-1,p)$ |
    |---|---|---|---|
    | 10 | 10 | **9** | 9 |
    | 20 | 15 | **14** | 14 |
    | 50 | 30 | **29** | 29 |

    **최소 고윳값이 $-2\times10^{-16}$ 같은 음수**로 나온다. 이론적으로는 정확히 0이지만 **수치 오차** 때문이다. 공분산행렬은 **반양정부호**이므로 음수 고윳값은 있을 수 없다.

    **$n>p$여도 안심할 수 없다.** $p=50$, $n=60$에서 최소 고윳값이 $5\times10^{-3}$으로 **매우 작다.** 역행렬의 성분이 거대해진다.

    **이것이 문제가 되는 곳 넷.**

    | 방법 | $S^{-1}$이 필요한 이유 |
    |---|---|
    | **마할라노비스 거리** | $\sqrt{(x-\mu)^TS^{-1}(x-\mu)}$ |
    | 선형판별분석(LDA) | 공통 공분산의 역행렬 |
    | **평균-분산 포트폴리오** | 최적 비중 $\propto S^{-1}\mu$ |
    | 가우스 그래프 모형 | **정밀도 행렬** $S^{-1}$ 자체가 대상 |

    **유전체·금융·이미지 자료에서 $p\gg n$이 흔하다.** 유전자 2만 개, 표본 100명이면 계수가 99다.

    **해결책 넷.**

    | 방법 | 내용 |
    |---|---|
    | **축소 추정**(레도이트-울프) | $\hat S=(1-\lambda)S+\lambda\cdot\text{목표}$ |
    | 정칙화 | $S+\epsilon I$ |
    | **희소 추정**(그래프 라소) | $S^{-1}$에 $L_1$ 벌점 |
    | 차원 축소 | 주성분 몇 개만 |

    **레도이트-울프 축소가 표준 도구**다. $\lambda$를 자료에서 최적으로 정한다.

    ```text
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(X)
    lw.covariance_       → 축소된 공분산행렬 (항상 양정부호)
    lw.precision_        → 그 역행렬
    lw.shrinkage_        → 자동으로 고른 λ
    ```

    **한 줄 요약.** $p$가 $n$에 가까워지면 **표본 공분산행렬을 믿지 않는다.** 고윳값이 위쪽은 과대, 아래쪽은 과소 추정되며, 이것이 축소 추정의 근거다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
공분산이 **분산 감소의 한계**를 어떻게 정하는지 보여라.

</div>

??? success "풀이"
    **동일 비중 $n$자산.** 각 분산이 $\sigma^2$, 모든 쌍 상관이 $\rho$면

    $$
    \operatorname{Var}\!\left(\frac1n\sum X_i\right)
    =\frac{\sigma^2}{n}+\frac{n-1}{n}\rho\sigma^2
    =\sigma^2\cdot\frac{1+(n-1)\rho}{n}
    $$

    **$n\to\infty$이면 $\rho\sigma^2$로 수렴**한다. 0이 아니다.

    ```python
    import numpy as np

    print("개별 분산 σ²=1, 모든 쌍 상관 ρ, 동일 비중 n 자산")
    print(f"{'ρ':>6s} {'n=10':>9s} {'n=50':>9s} {'n=200':>9s} {'n→∞':>9s}")
    for rho in [0.0, 0.1, 0.3, 0.5]:
        row = [(1 + (n - 1) * rho) / n for n in [10, 50, 200]]
        print(f"{rho:6.2f} {row[0]:9.4f} {row[1]:9.4f} {row[2]:9.4f} {rho:9.4f}")

    rng = np.random.default_rng(25002)
    print("\n모의실험으로 확인 (ρ=0.3)")
    for n in [10, 50, 200]:
        S = np.full((n, n), 0.3)
        np.fill_diagonal(S, 1.0)
        L = np.linalg.cholesky(S)
        r = (rng.standard_normal((200_000, n)) @ L.T).mean(1)
        print(f"  n={n:3d}: 모의 분산 {r.var(ddof=1):.4f}, "
              f"이론 {(1 + (n - 1) * 0.3) / n:.4f}")
    ```

    ```text
    개별 분산 σ²=1, 모든 쌍 상관 ρ, 동일 비중 n 자산
         ρ      n=10      n=50     n=200       n→∞
      0.00    0.1000    0.0200    0.0050    0.0000
      0.10    0.1900    0.1180    0.1045    0.1000
      0.30    0.3700    0.3140    0.3035    0.3000
      0.50    0.5500    0.5100    0.5025    0.5000

    모의실험으로 확인 (ρ=0.3)
      n= 10: 모의 분산 0.3696, 이론 0.3700
      n= 50: 모의 분산 0.3136, 이론 0.3140
      n=200: 모의 분산 0.3032, 이론 0.3035
    ```

    **이론과 모의실험이 소수점 셋째 자리까지 맞는다.**

    **$\rho$가 분산의 바닥을 정한다.**

    | $\rho$ | $n=200$ | 바닥($n\to\infty$) | 도달률 |
    |---|---|---|---|
    | 0.0 | 0.0050 | 0.0000 | — |
    | 0.1 | 0.1045 | 0.1000 | **96%** |
    | 0.3 | 0.3035 | 0.3000 | **99%** |
    | 0.5 | 0.5025 | 0.5000 | **99.5%** |

    **$\rho=0.3$이면 자산을 200개로 늘려도 분산이 0.30 아래로 못 간다.** 개별 분산의 30%가 **제거 불가능**하다.

    **이것이 금융의 "체계적 위험" 대 "개별 위험"**이다.

    $$
    \underbrace{\sigma^2\cdot\frac{1+(n-1)\rho}{n}}_{\text{전체}}
    =\underbrace{\rho\sigma^2}_{\text{체계적}}
    +\underbrace{\frac{(1-\rho)\sigma^2}{n}}_{\text{분산 가능}}
    $$

    **$n=10$에서 이미 상당히 도달한다.**

    | $n$ | $\rho=0.3$일 때 분산 | 남은 개선 여지 |
    |---|---|---|
    | 1 | 1.000 | — |
    | **10** | **0.370** | 0.070 |
    | 20 | 0.335 | 0.035 |
    | 50 | 0.314 | 0.014 |
    | 200 | 0.304 | 0.004 |

    **10~20개면 분산 효과의 90% 이상을 얻는다.** 종목을 200개로 늘리는 것은 **거의 의미가 없다.**

    **같은 구조가 통계학 곳곳에 나온다.**

    | 맥락 | 같은 식 |
    |---|---|
    | 군집 표본의 설계효과 | $1+(m-1)\rho_I$ |
    | 반복측정의 유효 표본수 | $n/(1+(m-1)\rho)$ |
    | **앙상블 학습의 오차** | 개별 모형의 상관이 바닥을 정한다 |

    **세 번째가 흥미롭다.** 랜덤 포레스트에서 나무를 아무리 늘려도 **나무들끼리의 상관**이 성능의 한계를 정한다. 그래서 **변수 무작위 선택**으로 상관을 일부러 낮춘다.

    **$\rho<0$이면 어떻게 되나.** 식이 여전히 성립하지만 $\rho\geq-1/(n-1)$이라는 제약이 붙는다. 상관행렬이 양정부호여야 하기 때문이다. $n=10$이면 $\rho\geq-0.111$이다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
공분산행렬의 **고유분해**가 무엇을 뜻하는지 수치로 보여라.

</div>

??? success "풀이"
    **고유분해.** 공분산행렬 $S$는 대칭 반양정부호이므로

    $$
    S=V\Lambda V^T,\qquad \Lambda=\operatorname{diag}(\lambda_1,\dots,\lambda_p)
    $$

    **$v_i$ 방향으로 자료를 투영하면 분산이 정확히 $\lambda_i$**가 된다.

    ```python
    import numpy as np

    rng = np.random.default_rng(25002)
    S = np.array([[4.0, 2.0], [2.0, 3.0]])
    ev, V = np.linalg.eigh(S)
    print(f"  공분산행렬 = {S.tolist()}")
    print(f"  고윳값 = {ev[::-1].round(4).tolist()}   (합 = {ev.sum():.1f} = 대각합)")
    print(f"  제1주성분 방향 = {V[:, -1].round(4).tolist()}")
    print(f"  제1주성분이 설명하는 비율 = {ev[-1] / ev.sum():.4f}")

    d = rng.multivariate_normal([0, 0], S, size=300_000)
    print(f"  그 방향으로 투영한 분산 = {(d @ V[:, -1]).var(ddof=1):.4f}  "
          f"(고윳값 {ev[-1]:.4f})")
    ```

    ```text
      공분산행렬 = [[4.0, 2.0], [2.0, 3.0]]
      고윳값 = [5.5616, 1.4384]   (합 = 7.0 = 대각합)
      제1주성분 방향 = [-0.7882, -0.6154]
      제1주성분이 설명하는 비율 = 0.7945
      그 방향으로 투영한 분산 = 5.5667  (고윳값 5.5616)
    ```

    **투영한 분산 5.5667이 고윳값 5.5616과 일치한다.**

    **세 가지 사실.**

    | 사실 | 확인 |
    |---|---|
    | **고윳값의 합 = 대각합** | $5.562+1.438=7.0=4+3$ |
    | **최대 고윳값 = 최대 분산 방향** | 5.562 |
    | 고유벡터는 **서로 직교** | $v_1\cdot v_2=0$ |

    **첫 번째가 "총분산"의 의미**다. 좌표를 회전해도 **전체 분산의 총량은 보존**된다.

    $$
    \sum_i\operatorname{Var}(X_i)=\sum_i\lambda_i
    $$

    **주성분분석은 이 회전을 찾는 일**이다. 분산이 큰 방향부터 정렬하면 **처음 몇 개로 대부분을 설명**할 수 있다.

    **여기서는 1개로 79.5%**를 설명한다.

    **부호에 주의.** 제1주성분 방향이 $(-0.788,-0.615)$로 나왔는데, $(+0.788,+0.615)$도 똑같이 옳다. **고유벡터의 부호는 임의**다.

    ```text
    라이브러리마다 부호가 다를 수 있다
      → 주성분 점수의 부호가 뒤집힌다
      → 해석할 때는 "어느 변수와 같은 방향인지"를 기준으로 삼는다
    ```

    **공분산 대 상관행렬.** 어느 것을 분해하느냐가 결과를 바꾼다.

    | 대상 | 언제 |
    |---|---|
    | **공분산행렬** | 변수들의 **단위가 같을 때** |
    | **상관행렬** | 단위가 다를 때(표준화한 것과 같다) |

    **단위가 다른데 공분산행렬을 쓰면** 척도가 큰 변수(예: 원 단위 소득)가 **제1주성분을 독점**한다. 이것이 주성분분석의 가장 흔한 실수다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
공분산의 **성질과 구현 지침**을 정리하라.

</div>

??? success "풀이"
    **정의와 기본 성질.**

    $$
    \operatorname{Cov}(X,Y)=E[(X-\mu_X)(Y-\mu_Y)]=E[XY]-E[X]E[Y]
    $$

    | 성질 | 내용 |
    |---|---|
    | 대칭 | $\operatorname{Cov}(X,Y)=\operatorname{Cov}(Y,X)$ |
    | 자기 자신 | $\operatorname{Cov}(X,X)=\operatorname{Var}(X)$ |
    | **쌍선형** | $\operatorname{Cov}(aX+bY,Z)=a\operatorname{Cov}(X,Z)+b\operatorname{Cov}(Y,Z)$ |
    | 상수 | $\operatorname{Cov}(X,c)=0$ |
    | **독립** $\Rightarrow$ 공분산 0 | 역은 **성립하지 않는다** |
    | 코시-슈바르츠 | $\lvert\operatorname{Cov}(X,Y)\rvert\leq\sigma_X\sigma_Y$ |

    **합의 분산.**

    $$
    \operatorname{Var}\!\left(\sum_i a_iX_i\right)
    =\sum_i a_i^2\operatorname{Var}(X_i)+2\sum_{i<j}a_ia_j\operatorname{Cov}(X_i,X_j)
    $$

    **핵심 수치 다섯.**

    | 사실 | 값 |
    |---|---|
    | 단순 공식이 무너지는 이동량(float64) | $\approx10^8$ |
    | float32에서 무너지는 이동량 | $\approx10^3$ |
    | 표본 공분산행렬의 계수 | $\min(n-1,p)$ |
    | 동일 비중 $n$자산의 분산 바닥 | $\rho\sigma^2$ |
    | $\rho=0.3$, $n=10$일 때 분산 | 0.370(바닥 0.300) |

    **구현 지침 넷.**

    ```text
    1. E[XY] - E[X]E[Y] 를 그대로 쓰지 않는다
    2. 자료를 두 번 읽을 수 있으면  → 2단계 공식
    3. 한 번만 읽어야 하면          → 웰퍼드 온라인 갱신
    4. 그냥 np.cov / pandas.cov 를 쓴다  ← 대부분의 경우 정답
    ```

    **자유도 선택.**

    | 코드 | 나누는 수 | 언제 |
    |---|---|---|
    | `np.cov(x, y)` | $n-1$ | **표본**(기본값) |
    | `np.cov(x, y, bias=True)` | $n$ | 모집단 전체 |
    | `pandas.DataFrame.cov()` | $n-1$ | 표본 |

    **`np.cov`의 기본값이 $n-1$**이고 `np.var`의 기본값이 $n$이라는 **비대칭**이 혼동을 낳는다.

    ```text
    np.var(x)             → n 으로 나눔   (ddof=0)
    np.cov(x, y)          → n-1 로 나눔  (ddof=1)
    np.cov(x, y)[0,0]  ≠  np.var(x)      ← 주의
    ```

    **흔한 실수 다섯.**

    | 실수 | 대가 |
    |---|---|
    | **단순 공식을 큰 값에 적용** | 부호까지 틀린다 |
    | 공분산 0을 독립으로 해석 | **비선형 의존**을 놓친다 |
    | $p\geq n$에서 $S^{-1}$을 구함 | 특이행렬 |
    | 단위가 다른데 공분산으로 PCA | 큰 척도 변수가 독점 |
    | `np.var`와 `np.cov`의 `ddof` 혼동 | 미세한 불일치 |

    **한 문장.** 공분산은 **두 변수의 동조를 원 단위로 재는 양**이며, 정의는 단순하지만 **수치적으로는 조심해서 계산해야 하고** 행렬로 모으면 $p$와 $n$의 관계가 그 쓸모를 정한다.

---

## 정리하며

공분산과 상관을 **제1원리에서** 만들어 보았다.

$$
\mathrm{Cov}(X,Y)=\frac{1}{n-1}\sum_i (x_i-\bar x)(y_i-\bar y),
\qquad r=\frac{\mathrm{Cov}(X,Y)}{s_X s_Y}
$$

- **$n-1$ 은 여기서도 베셀 보정이다.** 두 평균을 자료에서 추정했으므로 자유도를 잃는다(7장).
- **공분산은 단위에 의존한다.** 척도를 바꾸면 값이 바뀌므로 크기를 해석할 수 없고, 표준편차로 나눈 $r$ 만이 $[-1,1]$ 로 비교 가능해진다.
- **라이브러리와 대조해 검산한다.** `np.cov` 의 `ddof` 기본값이 1 이고 `np.var` 는 0 이라는 점을 다시 확인하게 된다.
- **공통 추세가 상관을 만든다.** 두 주의 가격이 같은 거시 요인에 노출되어 있으면 **인과관계가 전혀 없어도** 높은 상관이 나온다. 1장의 교란이 시계열에서 나타난 모습이다.
- **시계열 상관은 특히 조심해야 한다.** 둘 다 추세를 가지면 무관한 계열끼리도 높은 상관을 보이며, 이것이 허위상관의 전형이다.

다음 절 **회귀와 상관 그림**으로 넘어간다.
