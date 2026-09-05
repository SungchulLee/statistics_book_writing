# Matplotlib으로 기본 시각화하기

Matplotlib은 파이썬에서 그림을 그리는 토대가 되는 라이브러리이며, 이 책 전체에서 쓰는 출판 품질의 그림을 만들어낸다. 객체지향 API가 축, 눈금, 격자선, 주석 등 그림의 모든 요소를 정밀하게 제어하게 해준다. 통계 교재의 모든 그림은 특정한 논지를 뒷받침하므로 자동으로 생성할 것이 아니라 목적에 맞게 다듬어야 하기 때문에 이것이 중요하다.

## 정의

모든 Matplotlib 그림은 하나 이상의 **`Axes`** 객체(각각이 개별 패널이다)를 담고 있는 **`Figure`** 안에 존재한다. 이들을 만드는 권장 방식은 다음과 같다.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
# ... draw on ax ...
plt.show()
```

격자 배치의 경우:

```python
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
ax = axes[0, 1]    # row 0, column 1
```

`Figure`는 캔버스, 제목, 전체 배치를 담고, 각 `Axes`는 이름표·눈금·아티스트를 갖는 자기만의 좌표계다.

## 설명

### 두 가지 API

Matplotlib에는 두 개의 인터페이스가 있다.

1. **Pyplot(상태 기반)** — `plt.plot(x, y)`, `plt.title(...)`. 뒤에서 관리되는 전역 "현재 축"에 작용한다. 일회성 그림에는 편리하지만 여러 패널이 있는 그림에서는 모호하다.
2. **객체지향** — `ax.plot(x, y)`, `ax.set_title(...)`. 어느 `Axes`를 수정하는지 명시적이다. 패널이 둘 이상인 그림에는 이쪽이 낫다.

이 책은 객체지향 형태만 쓴다. pyplot 형태는 `plt.subplots`, `plt.show`, `plt.savefig`에만 남겨 둔다.

### 통계를 위한 핵심 그림 유형

| 그림 | 메서드 | 쓰임새 |
|---|---|---|
| 히스토그램 | `ax.hist(x, bins=...)` | 한 변수의 분포. `density=True`로 이론적 확률밀도함수를 겹쳐 그린다 |
| 산점도 | `ax.scatter(x, y)` | 두 변수의 관계, 회귀 진단 |
| 선 | `ax.plot(x, y)` | 시계열, 누적분포함수, 매끄러운 함수 곡선 |
| 상자그림 | `ax.boxplot(x_list)` | 집단별 다섯 수치 요약과 이상치 |
| 막대 | `ax.bar(categories, heights)` | 범주 간 비교 |
| Q-Q 그림 | `scipy.stats.probplot(x, plot=ax)` | 정규성 진단 |

### 사용자화의 핵심

```python
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Title")
ax.set_xlim(0, 10)
ax.set_ylim(-1, 1)
ax.grid(True, alpha=0.3)
ax.legend(loc="best", frameon=False)
fig.tight_layout()
fig.savefig("figure.png", dpi=150, bbox_inches="tight")
```

`tight_layout`은 여러 패널이 있는 그림에서 축 이름표가 잘리거나 겹치는 것을 막는다. `dpi=150`이면 화면과 대부분의 인쇄 용도에 충분하고, `dpi=300`은 최종 출판용이 아니라면 과하다.

### pandas와의 연동

DataFrame은 Matplotlib을 감싼 자체 그림 메서드를 갖고 있다.

```python
df["x"].plot.hist(bins=30)
df.plot.scatter(x="x", y="y", ax=ax)
df.boxplot(column="value", by="group", ax=ax)
```

빠르게 탐색할 때 편리하다. 최종 그림에서는 Matplotlib을 직접 호출하는 편이 더 세밀하게 제어할 수 있다.

## 예제

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

rng = np.random.default_rng(42)
data = rng.normal(loc=50, scale=10, size=500)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: histogram with theoretical density overlay
axes[0].hist(data, bins=30, density=True, alpha=0.5,
             edgecolor="black", label="Sample")
x = np.linspace(data.min(), data.max(), 200)
axes[0].plot(x, norm.pdf(x, loc=50, scale=10), "r-",
             lw=2, label="N(50, 10) PDF")
axes[0].set_xlabel("value"); axes[0].set_ylabel("density")
axes[0].set_title("Histogram with density overlay")
axes[0].legend()

# Right: Q-Q plot
from scipy.stats import probplot
probplot(data, dist="norm", plot=axes[1])
axes[1].set_title("Q-Q plot vs. Normal")

fig.tight_layout()
plt.show()
```

히스토그램은 적합도를 눈으로 확인하게 해주고, Q-Q 그림은 직선에서 벗어나는 정도를 보여줌으로써 분석적으로 확인하게 해준다. 통계적인 그림은 이 둘 중 하나 없이는 완성되는 일이 드물다.

## 연습문제

**연습문제 1.**
객체지향 API를 사용해 표준정규 표본 500개의 히스토그램을 구간 30개로 그리는 코드를 작성하라. 축 이름표, 제목, 그리고 표본평균 위치의 수직선을 포함하라.

??? success "연습문제 1 풀이"
    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    data = rng.standard_normal(500)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(data.mean(), color="red", lw=2, label=f"mean = {data.mean():.3f}")
    ax.set_xlabel("z")
    ax.set_ylabel("frequency")
    ax.set_title("500 standard normal samples")
    ax.legend()
    plt.show()
    ```

    `axvline`은 고정된 $x$ 좌표에 수직 참조선을 그리며, 평균·중앙값·임계값을 표시할 때 유용하다.

---

**연습문제 2.**
두 패널짜리 그림을 만들어라. 왼쪽에는 무작위 $(x, y)$ 쌍 100개의 산점도를, 오른쪽에는 $[0, 2\pi]$ 위의 곡선 $y = \sin(x)$을 그려라. 각 패널에 제목을 달고 "value"라는 하나의 $y$축 이름표를 공유하게 하라.

??? success "연습문제 2 풀이"
    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    x_s, y_s = rng.uniform(0, 10, 100), rng.uniform(-1, 1, 100)
    x_l = np.linspace(0, 2 * np.pi, 200)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    ax1.scatter(x_s, y_s, alpha=0.6)
    ax1.set_title("Random scatter")
    ax1.set_xlabel("x")
    ax2.plot(x_l, np.sin(x_l), color="blue")
    ax2.set_title("sin(x)")
    ax2.set_xlabel("x")
    fig.supylabel("value")
    fig.tight_layout()
    plt.show()
    ```

    `sharey=True`는 두 y축을 묶고, `fig.supylabel`은 그림 전체에 걸치는 하나의 y축 이름표를 추가한다.

---

**연습문제 3.**
여러 패널이 있는 그림에서 `plt.plot()`보다 `ax.plot()`이 선호되는 이유는 무엇인가? pyplot 형태가 조용히 엉뚱한 subplot에 그리게 되는 구체적인 예를 하나 들어라.

??? success "연습문제 3 풀이"
    `plt.plot()`은 전역 상태인 "현재" Axes를 대상으로 삼는다. 그림을 두 개 만드는 노트북 셀이나 여러 subplot을 갖는 그림 하나에서는, "현재" Axes가 가장 최근에 만들어지거나 활성화된 것 — 보통은 저자가 의도한 것이 아니라 **마지막** subplot — 이 된다.

    ```python
    fig, axes = plt.subplots(1, 2)
    axes[0].set_title("Panel A")        # explicit — correct
    plt.plot([1, 2, 3])                 # silently lands on axes[1] because it was created last
    ```

    객체지향 형태 `axes[0].plot([1, 2, 3])`는 대상을 명시하므로 이런 모호함을 완전히 없앤다.

---

**연습문제 4.**
$N(5, 4)$(평균 5, 분산 4)에서 뽑은 표본 1000개의 정규화된 히스토그램을 그리고 참된 밀도를 겹쳐 그려라. 경험적 곡선과 이론적 곡선이 일치함을 눈으로 확인하라.

??? success "연습문제 4 풀이"
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    rng = np.random.default_rng(42)
    data = rng.normal(loc=5, scale=2, size=1000)   # scale = sqrt(variance)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data, bins=30, density=True, alpha=0.5, edgecolor="black", label="Histogram")
    x = np.linspace(data.min(), data.max(), 200)
    ax.plot(x, norm.pdf(x, loc=5, scale=2), "r-", lw=2, label="N(5, 4) PDF")
    ax.set_xlabel("x"); ax.set_ylabel("density")
    ax.set_title("Histogram with true density overlay")
    ax.legend()
    plt.show()
    ```

    `density=True`는 막대 전체 넓이가 1이 되도록 히스토그램을 다시 크기 조정하여 확률밀도함수와 직접 비교할 수 있게 한다. 막대의 너비는 `bins`가 결정한다. 구간이 너무 적으면 구조를 감추고, 너무 많으면 없는 구조를 만들어낸다.

---

**연습문제 5.**
잔차 그림을 만들어라. $y = 1 + 2x + \varepsilon$에서 나온 잡음 섞인 점 50개에 최소제곱 직선을 적합한 뒤, 잔차를 적합값에 대해 그리고 0에 수평 참조선을 그어라. 선형성 가정이 위배되었음을 나타내는 패턴은 무엇인가?

??? success "연습문제 5 풀이"
    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, 50)
    y = 1 + 2 * x + rng.standard_normal(50)

    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    y_hat = X @ beta
    resid = y - y_hat

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(y_hat, resid, alpha=0.7)
    ax.axhline(0, color="red", lw=1)
    ax.set_xlabel("fitted value")
    ax.set_ylabel("residual")
    ax.set_title("Residuals vs. fitted")
    plt.show()
    ```

    제대로 지정된 선형모형은 뚜렷한 추세 없이 0 주위에 무작위로 흩어진 잔차를 만든다. 잔차 대 적합값 그림에서 **휘어진**(U자나 아치 모양) 패턴은 빠진 비선형 항을 알리는 신호이고, **깔때기** 모양은 분산이 일정하지 않음(이분산성)을 알리는 신호다. 제13장에서 이 진단들을 형식적으로 전개한다.

---

**연습문제 6.**
연습문제 5의 그림을 여백 없이 200 DPI PNG로 디스크에 저장하라. `bbox_inches="tight"` 인수는 무엇을 하며 언제 중요한가?

??? success "연습문제 6 풀이"
    ```python
    fig.savefig("residuals.png", dpi=200, bbox_inches="tight")
    ```

    `dpi=200`은 래스터화 해상도를 조절한다. `bbox_inches="tight"`는 가장자리의 빈 여백을 제외하도록 그림의 경계 상자를 다시 계산한다. 그림을 다른 문서(LaTeX, 워드, 슬라이드)에 끼워 넣을 때 가장 중요하다. 이 옵션이 없으면 savefig가 쓰이지 않은 공간까지 포함한 캔버스 전체를 저장하여, 삽입된 이미지에 보기 싫은 흰 테두리가 생긴다. `dpi=200, bbox_inches="tight"` 조합이 이 책에 실리는 그림의 권장 기본값이다.
