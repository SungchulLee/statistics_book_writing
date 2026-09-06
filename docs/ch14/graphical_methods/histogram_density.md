# 히스토그램과 밀도 그림

시각적 방법은 자료가 정규분포를 따르는지 평가하는 눈으로 보는 접근을 제공한다. 형식적 통계검정은 아니지만 자료의 분포를 이해하는 데 유용한 통찰을 준다.

## 개요

**히스토그램**은 자료의 분포를 그림으로 나타낸 것이다. 자료를 구간으로 나누고 각 구간에 자료점이 얼마나 자주 들어가는지 보여준다. 자료가 정규분포를 따르면 히스토그램이 익숙한 종 모양 곡선에 가까워야 한다. **밀도 그림**도 비슷하지만 분포를 나타내는 매끄러운 곡선을 제공한다.

## 정규 표본에 정규 확률밀도함수 겹치기

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_histogram_with_density(data, figsize=(12, 3)):
    """
    The histogram will show the frequency of data points,
    while the **kernel density estimate (KDE)** line will smooth the histogram
    to give a clearer idea of the data distribution.

    Parameters:
    - data (array-like): The input dataset to plot.
    - figsize (tuple): The size of the plot (width, height).

    Returns:
    - None: Displays the plot.
    """
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the histogram with the density curve (KDE)
    _, bins, _ = ax.hist(data, bins=20, density=True, alpha=0.5, label="Data Histogram")

    mu = data.mean()
    sigma = data.std()
    pdf = stats.norm(loc=mu, scale=sigma).pdf(bins)

    ax.plot(bins, pdf, "--r", label="Normal PDF")

    # Customize the appearance: remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Set plot title and labels
    ax.set_title('Histogram with Density Plot')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()

    plt.show()

if __name__ == "__main__":
    np.random.seed(0)
    sample_data = np.random.normal(loc=0, scale=1, size=1000)
    plot_histogram_with_density(sample_data)
```

자료가 정규분포에서 뽑혔을 때 히스토그램은 겹쳐 그린 정규 확률밀도함수 곡선과 잘 맞는다.

!!! note "`density=True`가 핵심이다"
    `ax.hist(..., density=True)`가 히스토그램의 전체 넓이를 1로 만들어 준다. 이렇게 해야 확률밀도함수 곡선과 같은 척도가 되어 비교가 의미를 갖는다. 빈도(도수)를 그대로 그리면 두 곡선의 척도가 완전히 달라 겹쳐 그리는 것이 무의미해진다.

## 지수 표본에 정규 확률밀도함수 겹치기

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_histogram_with_density(data, figsize=(12, 3)):
    """
    Plot histogram with a fitted normal PDF overlay.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _, bins, _ = ax.hist(data, bins=20, density=True, alpha=0.5, label="Data Histogram")

    mu = data.mean()
    sigma = data.std()
    pdf = stats.norm(loc=mu, scale=sigma).pdf(bins)
    ax.plot(bins, pdf, "--r", label="Normal PDF")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title('Histogram with Density Plot')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    np.random.seed(0)
    sample_data = np.random.exponential(scale=1, size=1000)
    plot_histogram_with_density(sample_data)
```

지수 자료에서는 히스토그램이 오른쪽으로 강하게 치우쳐 있어 대칭인 정규 확률밀도함수 곡선과 뚜렷이 맞지 않는다.

## 카이제곱 표본에 정규 확률밀도함수 겹치기

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_histogram_with_density(data, figsize=(12, 3)):
    """
    Plot histogram with a fitted normal PDF overlay.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _, bins, _ = ax.hist(data, bins=20, density=True, alpha=0.5, label="Data Histogram")

    mu = data.mean()
    sigma = data.std()
    pdf = stats.norm(loc=mu, scale=sigma).pdf(bins)
    ax.plot(bins, pdf, "--r", label="Normal PDF")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title('Histogram with Density Plot')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    np.random.seed(0)
    sample_data = np.random.chisquare(df=10, size=1000)
    plot_histogram_with_density(sample_data)
```

자유도가 중간 정도인 카이제곱 자료는 오른쪽으로 적당히 치우쳐 있다($\chi^2_{10}$의 왜도는 $\sqrt{8/10} = 0.894$이다). 정규 확률밀도함수가 대략은 맞지만 완벽하지 않으며, 시각적 점검을 넘어 형식적 검정이 필요한 이유를 보여준다.

## 연습문제

**연습문제 1.**
구간 폭의 선택이 히스토그램의 모습에 어떤 영향을 주는지 설명하라. 구간이 너무 적으면 어떻게 되는가? 너무 많으면?

??? success "연습문제 1 풀이"
    **구간이 너무 적으면(폭이 넓으면):** 히스토그램이 과도하게 평활된다. 이봉성, 치우침, 자료의 빈틈 같은 중요한 특징이 가려진다. 분포가 실제보다 단순해 보인다.

    **구간이 너무 많으면(폭이 좁으면):** 평활이 부족하다. 무작위 잡음이 들쭉날쭉한 톱니 모양을 만들어 바탕 모양을 가린다. 각 구간에 관측값이 적어 막대 높이를 믿을 수 없다.

    **구간 폭에 대한 지침:** 흔히 쓰는 규칙으로 Sturges 규칙($k = 1 + \log_2 n$), Freedman-Diaconis 규칙($h = 2 \times \text{IQR} \times n^{-1/3}$), Scott 규칙($h = 3.49s \times n^{-1/3}$)이 있다. Freedman-Diaconis 규칙이 이상점에 가장 로버스트하다.

---

**연습문제 2.**
히스토그램과 커널밀도추정(KDE)의 차이는 무엇인가? 각각의 장점을 하나씩 말하라.

??? success "연습문제 2 풀이"
    **히스토그램**은 자료를 구간으로 나누고 구간마다 관측값을 센다. 불연속(계단함수)이며 구간 경계에 의존한다.

    **KDE**는 각 자료점에 매끄러운 핵(예: Gauss)을 놓고 더하여 밀도의 매끄러운 연속 추정을 만든다.

    **히스토그램의 장점:** 해석이 더 단순하고, 도수와 빈도를 직접 보여주며, 자료의 빈틈이 눈에 보인다.

    **KDE의 장점:** 매끄럽고 연속이며, 임의로 정한 구간 경계에 의존하지 않고, 참 밀도의 모양을 더 잘 나타낸다. 구간화가 만들어 내는 가짜 봉우리나 골짜기를 피한다.

---

**연습문제 3.**
정규성 평가를 위해 히스토그램에 정규 밀도 곡선을 겹칠 때 히스토그램의 $y$축에 대해 무엇을 확인해야 하는가?

??? success "연습문제 3 풀이"
    히스토그램은 전체 넓이가 1이 되도록 **정규화**되어야 한다(밀도 척도). 이는 확률밀도함수의 성질과 일치한다. matplotlib에서는 보통 `density=True`로 설정하거나 적절한 구간 폭으로 상대도수를 쓴다.

    히스토그램이 원래의 도수(빈도)를 보여준다면, 적분값이 1인 정규 밀도 곡선은 완전히 다른 척도에 놓여 시각적 비교가 무의미해진다. 정규화한 뒤에는 각 막대의 높이가 추정된 밀도를 나타내므로 겹쳐 그린 정규 확률밀도함수와 직접 비교할 수 있다.

---

**연습문제 4.**
관측값 500개인 자료에 KDE를 겹친 히스토그램을 만드는 Python 코드를 작성하라.

??? success "연습문제 4 풀이"
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    rng = np.random.default_rng(42)
    data = rng.normal(5, 2, 500)

    fig, ax = plt.subplots()
    ax.hist(data, bins=30, density=True, alpha=0.5, edgecolor="black", label="Histogram")

    kde = gaussian_kde(data)
    x_grid = np.linspace(data.min() - 1, data.max() + 1, 200)
    ax.plot(x_grid, kde(x_grid), "r-", lw=2, label="KDE")

    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    plt.show()
    ```
