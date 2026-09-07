# 시각적 정규성 확인

## 개요

시각적 방법은 자료가 정규분포를 따르는지 평가하는 직관적인 첫 단계를 제공한다. 형식적 가설검정을 수행하기 전에 히스토그램, 밀도 겹쳐 그리기, 분위수-분위수(Q-Q) 그림을 눈으로 살피면 치우침, 두꺼운 꼬리, 다봉성 등 정규성에서의 이탈을 드러낼 수 있다. 이 방법들은 자료가 *이탈하는지*뿐 아니라 *어떻게* 이탈하는지를 보여주어 형식적 검정을 보완한다.

## 정규 밀도를 겹친 히스토그램

가장 단순한 시각적 확인은 관측 자료의 히스토그램을 그리고 평균과 분산을 표본 추정값에 맞춘 정규분포의 확률밀도함수를 겹쳐 그리는 것이다.

$X_1, X_2, \ldots, X_n$을 독립인 확률표본이라 하자. 표본평균과 표본표준편차는

$$
\bar{X} = \frac{1}{n}\sum_{i=1}^{n} X_i, \qquad S = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(X_i - \bar{X})^2}.
$$

적합된 정규 밀도는

$$
\hat{f}(x) = \frac{1}{S\sqrt{2\pi}} \exp\!\Bigl(-\frac{(x - \bar{X})^2}{2S^2}\Bigr).
$$

히스토그램 막대가 $\hat{f}$와 가깝게 맞으면 자료가 정규성과 일관된다.

### 코드

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

n = 100
data = np.random.normal(loc=0, scale=1, size=n)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(data, bins=15, density=True, alpha=0.6, edgecolor="black")
x_grid = np.linspace(data.min() - 0.5, data.max() + 0.5, 200)
ax.plot(x_grid, stats.norm.pdf(x_grid, data.mean(), data.std(ddof=1)),
        linewidth=2, label="Fitted Normal PDF")
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.set_title("Histogram with Normal Overlay")
ax.legend()
plt.tight_layout()
plt.show()
```

## 커널밀도추정

커널밀도추정(KDE)은 히스토그램을 매끄럽게 만들어 다봉성이나 비대칭을 발견하는 데 유용하다. Gauss 핵과 띠너비 $h$를 쓰면 KDE는

$$
\hat{f}_h(x) = \frac{1}{nh}\sum_{i=1}^{n} \phi\!\Bigl(\frac{x - X_i}{h}\Bigr),
$$

여기서 $\phi$는 표준정규 밀도이다. 이를 적합된 정규 곡선과 겹쳐 그린다. 체계적인 차이가 보이면 정규성에서의 이탈을 나타낸다.

## Q-Q 그림

분위수-분위수(Q-Q) 그림은 가장 정보량이 많은 단일 시각적 정규성 확인법이다. 각 순서통계량 $X_{(i)}$에 대해 대응하는 이론적 분위수를 계산하고

$$
q_i = \Phi^{-1}\!\Bigl(\frac{i - 0.5}{n}\Bigr),
$$

쌍 $(q_i,\, X_{(i)})$를 그린다. 정규성 아래에서는 점들이 대략 직선 위에 놓인다. 흔한 이탈에는 알아볼 수 있는 특징이 있다.

| 패턴 | 이탈 |
|---|---|
| S자 곡선 | 두꺼운 꼬리(고첨) |
| 아래로 볼록한 호(양 끝이 선 위로) | 오른쪽 치우침 |
| 위로 볼록한 호(양 끝이 선 아래로) | 왼쪽 치우침 |
| 계단 모양 | 이산성 또는 반올림 |

!!! note "치우침의 곡률 방향"
    오른쪽으로 치우친 자료를 $(q_i, X_{(i)})$로 그리면 곡선이 **아래로 볼록**(convex)해진다. 왼쪽 꼬리에서는 표본분위수가 기대보다 덜 음수라 선 위에 있고, 오른쪽 꼬리에서는 기대보다 더 양수라 역시 선 위에 있기 때문이다. 곧 양 끝이 모두 선 위로 올라간다. 왼쪽 치우침은 그 반대로 위로 볼록(concave)해진다.

## 해석

시각적 확인은 주관적이지만 매우 값지다. 뚜렷하게 이봉인 히스토그램, 두드러진 비대칭을 드러내는 KDE, 꼬리에서 급격히 휘는 Q-Q 그림은 모두 형식적 정규성 검정이 기각할 가능성이 높음을 알린다. 반대로 시각적 확인이 깨끗해 보이면 형식적 검정의 애매한 $p$값을 덜 걱정해도 된다. 균형 잡힌 평가를 위해 언제나 시각적 방법과 형식적 방법을 함께 쓰라.

## 연습문제

**연습문제 1.** 표준정규분포에서 관측값 $n = 200$개를 생성하라. 구간 20개의 히스토그램을 그리고 적합된 정규 밀도를 겹쳐라. 적합에 대해 논평하라.

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, size=200)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(data, bins=20, density=True, alpha=0.6, edgecolor="black")
    x_grid = np.linspace(-4, 4, 200)
    ax.plot(x_grid, stats.norm.pdf(x_grid, data.mean(), data.std(ddof=1)),
            linewidth=2)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Histogram with Normal Overlay")
    plt.tight_layout()
    plt.show()
    ```

    표준정규 추출값 $n = 200$개이면 히스토그램 막대가 종 모양의 적합 곡선을 가깝게 따라간다. 표집변동으로 인한 작은 이탈은 예상되는 일이다. $\square$

---

**연습문제 2.** $\text{Lognormal}(0, 0.6)$ 분포에서 관측값 $n = 300$개를 생성하라. 정규분포에 대한 Q-Q 그림을 만들고 관찰되는 모양을 기술하라.

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(1)
    data = rng.lognormal(0, 0.6, size=300)

    stats.probplot(data, dist="norm", plot=plt)
    plt.title("Q-Q Plot: Lognormal Data vs Normal")
    plt.tight_layout()
    plt.show()
    ```

    Q-Q 그림은 **아래로 볼록한**(convex, 위로 휘는) 곡선을 보인다. 자료의 위쪽 분위수가 이론적 정규분위수를 크게 넘어선다. 오른쪽 치우침의 전형적인 특징이다.

    수치로 확인해 보자. $\text{Lognormal}(0, 0.6)$의 1%, 50%, 99% 분위수는 각각 $e^{0.6 \times (-2.326)} = 0.248$, $1$, $e^{0.6 \times 2.326} = 4.04$이다. 중앙값에서 아래로는 $0.752$, 위로는 $3.04$ 떨어져 있어 위쪽이 네 배 넘게 길다. 왼쪽 아래 구간의 기울기는 작고 오른쪽 위 구간의 기울기는 크므로 곡선이 아래로 볼록해진다. $\square$

---

**연습문제 3.** Q-Q 그림이 정규성 이탈의 *유형*(예: 치우침 대 두꺼운 꼬리)을 드러낼 수 있는 반면 형식적 검정의 $p$값 하나로는 그럴 수 없는 이유를 설명하라.

??? success "풀이"

    형식적 정규성 검정은 정규성에서의 전반적 이탈을 재는 검정통계량과 $p$값 하나를 내놓는다. $p$값은 분포가 *어떻게* 벗어나는지에 대해 아무것도 말해 주지 않는다. 반면 Q-Q 그림은 모든 순서통계량을 그에 대응하는 이론값과 나란히 보여주므로, 이탈이 꼬리에서 일어나는지(두꺼운 꼬리는 S자를 만든다), 한쪽 꼬리에서만 일어나는지(치우침은 볼록하거나 오목한 호를 만든다), 중앙에서 일어나는지(다봉성은 계단이나 평평한 구역을 만든다) 볼 수 있다. 이 진단의 풍부함 때문에 형식적 검정과 함께 시각적 확인이 권장된다. $\square$

---

**연습문제 4.** 연속분포 $F$에서 크기 $n$인 확률표본을 뽑았을 때, $F = \Phi$라는 귀무가설 아래에서 Q-Q 그림의 $i$번째 순서통계량 플로팅 위치의 기댓값이 근사적으로 $\Phi^{-1}\!\bigl(\frac{i - 0.5}{n}\bigr)$임을 보여라.

??? success "풀이"

    $F = \Phi$ 아래에서 확률적분변환에 의해 $U_i = \Phi(X_i) \sim \text{Uniform}(0,1)$이다. 균등표본의 $i$번째 순서통계량 $U_{(i)}$는 $\mathbb{E}[U_{(i)}] = \frac{i}{n+1}$을 만족한다. $n$이 크면 Blom 근사가 이를 $p_i = \frac{i - 0.375}{n + 0.25} \approx \frac{i - 0.5}{n}$으로 바꾸며, 이는 경계 근처에서 균등 순서통계량의 편향을 보정한다. 여기에 $\Phi^{-1}$을 적용하면 이론적 분위수 $q_i = \Phi^{-1}(p_i)$를 얻는다. 자료가 정말로 $\Phi$에서 왔다면 정렬된 표본값 $X_{(i)}$가 $\mathbb{E}[X_{(i)}] \approx q_i$를 만족하므로 Q-Q 그림이 항등선 위에 놓일 것으로 기대된다. $\square$

---

**연습문제 5.** 자료 배열을 받아 (a) 정규 밀도를 겹친 히스토그램과 (b) Q-Q 그림을 나란히 배치한 그림을 만드는 Python 함수를 작성하라. 자유도 4인 $t$ 분포 자료로 시험하라.

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    def normality_panel(data):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram with normal overlay
        ax = axes[0]
        ax.hist(data, bins=30, density=True, alpha=0.6, edgecolor="black")
        x_grid = np.linspace(data.min() - 1, data.max() + 1, 300)
        ax.plot(x_grid,
                stats.norm.pdf(x_grid, data.mean(), data.std(ddof=1)),
                linewidth=2, label="Fitted Normal")
        ax.set_title("Histogram + Normal Overlay")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()

        # Q-Q plot
        ax = axes[1]
        stats.probplot(data, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot vs Normal")

        plt.tight_layout()
        plt.show()

    rng = np.random.default_rng(42)
    t_data = rng.standard_t(df=4, size=500)
    normality_panel(t_data)
    ```

    $t_4$ 자료에서 히스토그램은 정규 곡선보다 두꺼운 꼬리를 보인다($\pm 3$ 바깥에 질량이 더 많다). Q-Q 그림은 특징적인 S자를 보인다. 왼쪽 아래 점들은 선 아래로, 오른쪽 위 점들은 선 위로 휘어 초과첨도를 확인해 준다. $\square$
