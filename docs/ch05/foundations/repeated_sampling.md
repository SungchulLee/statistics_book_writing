# 반복추출 개념

## 개요

**표본분포**는 확률표본에 기반한 어떤 통계량의 확률분포이다. 같은 모집단에서 여러 개의 확률표본을 뽑아 각 표본마다 통계량(표본평균이나 표본비율 등)을 계산하면, 그 값들이 하나의 분포를 이룬다. 이 분포를 그 통계량의 **표본분포**라 한다.

$$
\left.
\begin{array}{ccccc}
\text{Population} &\rightarrow& \text{Sample } \mathbf{x}_1 &\rightarrow& \hat{\theta}(\mathbf{x}_1) \\
\\
\text{Population} &\rightarrow& \text{Sample } \mathbf{x}_2 &\rightarrow& \hat{\theta}(\mathbf{x}_2) \\
&\vdots& & & \\
\text{Population} &\rightarrow& \text{Sample } \mathbf{x}_n &\rightarrow& \hat{\theta}(\mathbf{x}_n) \\
&\vdots& & &
\end{array}
\right\}
\;\;
\begin{array}{c}
\text{Sampling Distribution:} \\
\text{Distribution of } \hat{\theta}(\mathbf{x}_1), \hat{\theta}(\mathbf{x}_2), \cdots, \hat{\theta}(\mathbf{x}_n), \cdots \\
\text{or Distribution of } \hat{\theta}(\mathbf{x})
\end{array}
$$

## 표본분포는 왜 중요한가?

표본분포는 표본을 바탕으로 모집단에 관한 결론을 내리는 추론통계학의 근본이다. 여러 표본에 걸친 통계량의 거동을 이해하면 다음을 할 수 있다:

- 표본에서 얻은 통계량으로 **모수를 추정**한다 (예: 평균, 분산).
- 추정량의 변동성을 파악하기 위해 **표준오차를 계산**한다.
- 추정의 불확실성을 정량화하기 위해 **신뢰구간을 구성**한다.
- 모수에 관해 근거 있는 판단을 내리기 위해 **가설검정을 수행**한다.

## 구별해야 할 세 가지 분포

### 모집단 분포, 표본 분포, 표본분포

**모집단 분포**는 모집단 전체에서 어떤 변수가 가질 수 있는 모든 값의 분포를 나타낸다. 표본을 뽑아 오는 모집단을 특징짓는 밑바탕의 분포이다.

**표본 분포**는 모집단에서 뽑은 특정한 하나의 표본 안에 있는 값들의 분포를 가리킨다. 표본은 모집단의 부분집합이며, 이를 사용해 모집단 전체에 관한 추론을 한다.

**표본분포**는 모집단에서 뽑은 같은 크기의 여러 표본으로부터 계산한 통계량(표본평균이나 표본비율 등)의 분포를 기술한다. 통계량의 변동성을 이해하게 해 주며 통계적 추론의 중심이 된다.

$$
\begin{array}{ccccccc}
\text{Population}
&\rightarrow&
\text{Sample } \mathbf{x}
&\rightarrow&
\text{Estimate } \hat{\theta}(\mathbf{x}) \\
\uparrow && \uparrow && \uparrow \\
\text{Population Distribution:} && \text{Sample Distribution:} && \text{Sampling Distribution:} \\
\text{Distribution of} && \text{Distribution of} && \text{Distribution of} \\
\text{Whole Population} && \text{Numbers in Particular Sample } \mathbf{x} && \text{Infinitely Many Estimates } \hat{\theta}(\mathbf{x}_i)
\end{array}
$$

## 모의실험 1: 균등 모집단

<div class="codebox" markdown>

### 예제 1. 균등 모집단에서 반복추출 { .eg }

```python
import matplotlib.pyplot as plt
import numpy as np

# 같은 결과가 다시 나오도록 난수 씨앗을 고정한다.
np.random.seed(1)

# 모집단 크기, 표본크기, 반복 횟수를 정한다.
sample_size = 5        # Size of a single random sample
n_samples = 10_000     # Number of samples to draw for the sampling distribution
n_population = 10_000  # Size of the population to simulate

def plot_distributions():
    """
    Generates a plot showing the population distribution, sample distribution,
    and sampling distribution.
    """
    # 균등분포에서 큰 모집단을 만든다. 10만 개면 사실상 무한 모집단으로 본다.
    population = np.random.uniform(size=(n_population,))

    # 모집단에서 표본 하나를 뽑는다. 가운데 패널에 그릴 자료다.
    single_sample = np.random.choice(population, size=sample_size, replace=False)

    # 같은 일을 여러 번 되풀이하며 그때마다 표본평균을 기록한다.
    sample_means = [
        np.mean(np.random.choice(population, size=sample_size, replace=False))
        for _ in range(n_samples)
    ]

    # 세 칸을 위아래로 놓는다. 모집단 → 표본 하나 → 표집분포 순이다.
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # 첫째 칸: 모집단의 모양.
    ax0.hist(population, bins=np.linspace(0, 1, 100))
    ax0.set_title('Population Distribution', fontsize=20)

    # 둘째 칸: 표본 하나를 점으로 흩뿌린다.
    ax1.scatter(single_sample, np.zeros_like(single_sample), s=100)
    ax1.set_title(f'Sample Distribution of {sample_size} Samples', fontsize=20)

    # 셋째 칸: 표본평균들의 히스토그램, 곧 표집분포다.
    ax2.hist(sample_means, bins=np.linspace(0, 1, 100))
    ax2.set_title('Sampling Distribution of $\\bar{X}$', fontsize=20)

    # 축 이름과 간격을 다듬는다.
    for ax in (ax0, ax1, ax2):
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_position('zero')
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_distributions()
```

![Population Distribution](./img/repeated_sampling_61.png)

**관찰.** 모집단이 균등분포(평평한 모양)임에도 $\bar{X}$의 표본분포는 종 모양이고 훨씬 좁게 모여 있다. 중심극한정리를 미리 엿보는 셈이다.

</div>

## 모의실험 2: 지수 모집단

<div class="codebox" markdown>

### 예제 2. 지수 모집단에서 반복추출 { .eg }

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# 같은 결과가 다시 나오도록 난수 씨앗을 고정한다.
np.random.seed(1)

# 모집단 크기, 표본크기, 반복 횟수를 정한다.
sample_size = 30
n_samples = 10_000
n_population = 10_000

def plot_distributions():
    """
    Generates a plot showing the population distribution, sample distribution,
    and sampling distribution for an exponential population.
    """
    # 지수분포에서 큰 모집단을 만든다. 오른쪽으로 길게 늘어진 모양이다.
    population = stats.expon().rvs((n_population,))

    # 모집단에서 표본 하나를 뽑는다. 가운데 패널에 그릴 자료다.
    single_sample = np.random.choice(population, size=sample_size, replace=False)

    # 같은 일을 여러 번 되풀이하며 그때마다 표본평균을 기록한다.
    sample_means = [
        np.mean(np.random.choice(population, size=sample_size, replace=False))
        for _ in range(n_samples)
    ]

    # 세 칸을 위아래로 놓는다. 모집단 → 표본 하나 → 표집분포 순이다.
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # 첫째 칸: 모집단의 모양.
    _, bins, _ = ax0.hist(population, bins=100)
    ax0.set_title('Population Distribution', fontsize=20)

    # 둘째 칸: 표본 하나를 점으로 흩뿌린다.
    ax1.scatter(single_sample, np.zeros_like(single_sample), s=100)
    ax1.set_title(f'Sample Distribution of {sample_size} Samples', fontsize=20)

    # 셋째 칸: 표본평균들의 히스토그램, 곧 표집분포다.
    ax2.hist(sample_means, bins=bins)
    ax2.set_title('Sampling Distribution of $\\bar{X}$', fontsize=20)

    # 축 이름과 간격을 다듬는다.
    for ax in (ax0, ax1, ax2):
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_position('zero')
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_distributions()
```

![Population Distribution](./img/repeated_sampling_124.png)

**관찰.** 지수 모집단은 오른쪽으로 심하게 치우쳐 있지만, $n = 30$일 때 $\bar{X}$의 표본분포는 근사적으로 정규분포이다. 중심극한정리가 작동하는 모습이다.

</div>

## 모의실험 3: 베르누이 모집단

<div class="codebox" markdown>

### 예제 3. 베르누이 모집단에서 반복추출 { .eg }

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# 같은 결과가 다시 나오도록 난수 씨앗을 고정한다.
np.random.seed(1)

# 모집단 크기, 표본크기, 반복 횟수를 정한다.
sample_size = 30
n_samples = 10_000
n_population = 10_000

def plot_distributions():
    """
    Generates a plot showing the population distribution, sample distribution,
    and sampling distribution for a Bernoulli population.
    """
    # 베르누이 모집단을 만든다. 값은 0과 1 둘뿐이다.
    population = stats.binom(n=1, p=0.3).rvs((n_population,))

    # 모집단에서 표본 하나를 뽑는다. 가운데 패널에 그릴 자료다.
    single_sample = np.random.choice(population, size=sample_size, replace=False)

    # 같은 일을 여러 번 되풀이하며 그때마다 표본평균을 기록한다.
    sample_means = [
        np.mean(np.random.choice(population, size=sample_size, replace=False))
        for _ in range(n_samples)
    ]

    # 세 칸을 위아래로 놓는다. 모집단 → 표본 하나 → 표집분포 순이다.
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # 첫째 칸: 모집단의 모양.
    _, bins, _ = ax0.hist(population, bins=100)
    ax0.set_title('Population Distribution', fontsize=20)

    # 둘째 칸: 표본 하나를 점으로 흩뿌린다.
    ax1.scatter(single_sample, np.zeros_like(single_sample), s=100)
    ax1.set_title(f'Sample Distribution of {sample_size} Samples', fontsize=20)

    # 셋째 칸: 표본평균들의 히스토그램, 곧 표집분포다.
    ax2.hist(sample_means, bins=10)
    ax2.set_title('Sampling Distribution of $\\bar{X}$', fontsize=20)

    # 축 이름과 간격을 다듬는다.
    for ax in (ax0, ax1, ax2):
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_position('zero')
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_distributions()
```

![Population Distribution](./img/repeated_sampling_188.png)

</div>

## 공 세 개에서 두 개를 뽑을 때의 표본분포

> **출처:** [Khan Academy — Introduction to Sampling Distributions](https://www.khanacademy.org/math/ap-statistics/sampling-distribution-ap/what-is-sampling-distribution/v/introduction-to-sampling-distributions)

<div class="probox" markdown>

**문제.** <span class="diff easy" title="쉬움"></span> 항아리에 1, 2, 3번이 매겨진 공 세 개가 있다. 모평균은 $\mu = 2$이다. 복원추출로 공 두 개를 뽑아 평균을 구한다. 이 표본평균의 분포, 즉 $\bar{X}$의 표본분포를 구하라.

</div>

??? success "풀이"
    동일한 확률을 갖는 $3^2 = 9$가지 결과가 있다:

    ```python
    import itertools as it
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    def main():
        sample_space = np.array([1, 2, 3])

        # 표본이 아주 작아 **가능한 모든 경우를 다 적을 수 있다.**
        # product(..., repeat=2) 가 복원추출로 두 개를 뽑는 3^2 = 9가지를 만든다.
        # 모의실험이 아니라 완전열거이므로 여기서 얻는 표집분포는 근사가 아니라 정확하다.
        columns = ["first", "second", "average"]
        df = pd.DataFrame(columns=columns)
        for first, second in it.product(sample_space, repeat=2):
            dg = pd.DataFrame([[first, second, (first + second) / 2]], columns=columns)
            df = pd.concat([df, dg], ignore_index=True)
        print(df, end="\n\n")

        fig, ax = plt.subplots(figsize=(12, 3))
        # 표본평균이 1, 1.5, 2, 2.5, 3 다섯 값만 가지므로 구간 경계를
        # 값에서 0.25씩 왼쪽으로 밀어 각 값이 자기 막대 가운데에 오게 한다.
        bins = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5]) - 0.25
        ax.hist(df.average, bins=bins, density=True, alpha=0.7)
        ax.set_title(r"Sampling Distribution of $\bar{X}$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.show()

    if __name__ == "__main__":
        main()
    ```

    출력:

    ```
      first second  average
    0     1      1      1.0
    1     1      2      1.5
    2     1      3      2.0
    3     2      1      1.5
    4     2      2      2.0
    5     2      3      2.5
    6     3      1      2.0
    7     3      2      2.5
    8     3      3      3.0
    ```

    ![반복추출 개념](./img/repeated_sampling_256.png)

    표본분포가 가질 수 있는 값은 $\{1.0, 1.5, 2.0, 2.5, 3.0\}$이고 확률은 $\{1/9, 2/9, 3/9, 2/9, 1/9\}$이다. 평균은 $E[\bar{X}] = 2 = \mu$이며, $\bar{X}$가 불편임을 확인해 준다.
## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
어떤 모집단의 평균이 100이고 표준편차가 20이다. 표본크기 $n$이 커질 때 (a) $\bar X$의 표본분포의 평균, (b) 표준오차, (c) 하나의 표본평균 $\bar x$는 각각 어떻게 되는가?

</div>

??? success "풀이"
    (a) **표본분포의 평균**은 모든 $n$에 대해 모평균 100이다. 표본크기와 무관하게 표본분포는 $\mu$를 중심으로 하며, $\bar X$는 *불편*이다.

    (b) **표준오차**는 $\sigma/\sqrt n = 20/\sqrt n$이다. $n$이 커질수록 줄어든다: $n = 4 \to \mathrm{SE} = 10$; $n = 100 \to \mathrm{SE} = 2$; $n = 10000 \to \mathrm{SE} = 0.2$.

    (c) **하나의 표본평균** $\bar x$는 약한 큰수의 법칙에 의해 100으로 확률수렴하고, 강한 큰수의 법칙에 의해 거의 확실하게 수렴한다. 표본분포가 좁아진다는 것은 개별 $\bar x$가 100에 가까울 가능성이 점점 커진다는 뜻이다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
**표준오차와 표준편차.** 예를 들어 그 구별을 설명하라. 표준오차는 왜 $\sigma$가 아니라 항상 $\sigma / \sqrt n$인가?

</div>

??? success "풀이"
    **표준편차 $\sigma$:** *모집단*(또는 하나의 표본)의 퍼짐을 잰다. 개별 관측값이 얼마나 흩어져 있는지를 기술한다.

    **표준오차 $\sigma/\sqrt n$:** *통계량의 표본분포*(보통 표본평균)의 퍼짐을 잰다. $\bar X$가 표본마다 얼마나 달라지는지를 기술한다.

    서로 다른 두 양이다. "SD = 5"라고 보고하는 것은 자료를 기술하는 것이고, "SE = 0.5"라고 보고하는 것은 추정값에 대한 불확실성을 기술하는 것이다.

    **평균에서 왜 $\sigma/\sqrt n$인가?** i.i.d. 자료에 대해 $\mathrm{Var}(\bar X) = \mathrm{Var}((1/n)\sum X_i) = (1/n^2) \cdot n\sigma^2 = \sigma^2/n$임을 떠올리자. 제곱근을 취하면 SE $= \sigma/\sqrt n$이다. $\sqrt n$ 비율은 "제곱근 개선의 법칙"이다. 표본을 네 배로 늘리면 표준오차가 절반이 된다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
**비복원추출.** 평균이 $\mu$, 분산이 $\sigma^2$인 크기 $N$의 유한모집단에서 $n$개를 *비복원*으로 뽑을 때 $\bar X$의 분산을 유도하라.

</div>

??? success "풀이"
    $\mathrm{Var}(\bar X) = \frac{\sigma^2}{n} \cdot \frac{N - n}{N - 1}$이며, 뒤의 인수를 **유한모집단 수정(FPC)** 계수라 한다.

    유도: $X_i$들은 더 이상 독립이 아니지만(두 번째 추출이 첫 번째에 의존한다) 교환 가능하다. 합의 분산을 계산하면:

    $\mathrm{Var}(\sum X_i) = n\sigma^2 + n(n-1) \mathrm{Cov}(X_1, X_2)$이다. 대칭성에 의해 $\sum_{i \ne j} \mathrm{Cov}(X_i, X_j) = -\mathrm{Var}(\sum X_i^{\text{total}})/(N-1)$이므로 $\mathrm{Cov}(X_1, X_2) = -\sigma^2/(N-1)$이다.

    따라서 $\mathrm{Var}(\sum X_i) = n\sigma^2(N-n)/(N-1)$이고, $n^2$으로 나누면 FPC 공식을 얻는다.

    **실무:** $n/N \le 0.05$이면 FPC가 1에 가까워 무시할 수 있다. $n/N$이 상당히 크면(감사, 재검표 등) FPC가 중요하며 표준오차 공식에 반드시 포함해야 한다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
**점근분포와 정확한 분포.** $\mathrm{Uniform}(0, 1)$에서 크기 $n = 5$인 표본을 뽑을 때 $\bar X$의 정확한 분포는 알려져 있다(척도조정된 Irwin-Hall 분포). 그 모양을 그려 보고 중심극한정리가 예측하는 정규근사와 비교하라.

</div>

??? success "풀이"
    **정확한 분포:** i.i.d. Uniform(0, 1) 다섯 개의 합은 $[0, 5]$ 위의 Irwin-Hall 분포이다. 4차 조각별 다항식이고 종 모양이며 2.5를 중심으로 대칭이다. $1/5$로 척도조정하면 $[0, 1]$ 위의 $\bar X$가 되고 $1/2$에서 정점을 이룬다.

    **중심극한정리 근사:** $\bar X \approx N(1/2, 1/(12 \cdot 5)) = N(0.5, 0.0167)$이고 표준편차는 $\approx 0.129$이다.

    **비교:** 정확한 분포는 $[0, 1]$에서 유계이지만 정규분포는 $\pm \infty$까지 뻗는다. 중앙에서는 둘이 거의 같고, 꼬리에서는 정확한 분포가 $[0, 1]$ 밖에서 확률 0인 반면 정규분포는 그곳에 작은 양의 확률(0 아래 또는 1 위로 약 0.001)을 준다.

    $n = 5$에서 이미 근사가 꽤 좋다. 균등분포의 대칭성과 유계 지지집합 덕분에 중심극한정리 수렴이 매우 빠르다. 반면 Exponential(1)에서 크기 5인 표본이라면 $\bar X$의 분포가 눈에 띄게 치우쳐 정규분포와 거리가 멀 것이다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
중심극한정리에서의 **수렴은 각 점에서의 수렴이 아니라 분포수렴이다.** 분포수렴을 정의하고 왜 이것이 중심극한정리에 알맞은 개념인지 설명하라.

</div>

??? success "풀이"
    **분포수렴:** $X_n \xrightarrow{d} X$일 필요충분조건은 $F_X$의 모든 연속점 $x$에서 $F_{X_n}(x) \to F_X(x)$인 것이다.

    이는 확률변수 자체가 아니라 *분포*에 관한 서술이다. $X_n$과 $X$가 같은 확률공간 위에 정의될 필요도 없다. CDF가 수렴하기만 하면 된다.

    **왜 중심극한정리에 알맞은가:** 중심극한정리는 $\sqrt n (\bar X_n - \mu)/\sigma$의 분포를 고정된 정규분포와 비교한다. $\bar X_n$의 실현값이 어떤 특정한 정규확률변수로 수렴한다고 주장하는 것이 아니라, 오직 그 *분포*가 정규분포에 가까워진다고 말하는 것이다. 실현되는 경로마다 극한 거동이 다르지만 안정되는 것은 분포이다.

    거의 확실한 수렴과의 구별: $\bar X_n \to \mu$는 거의 확실하게 성립하지만(각 경로가 수렴한다), $\sqrt n (\bar X_n - \mu)$는 어디에서도 거의 확실하게 수렴하지 *않는다*. 계속 요동치는 가운데 그 *분포*만 정규분포에 가까워진다. 더 강한 형태 없이 오직 분포수렴만 성립하는 것이다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
**모의실험.** 임의의 모집단에서 $\bar X$의 표본분포를 *시각화*하는 방법을 서술하라. $n = 5, 30, 100$에 대해 어떤 그림 세 개를 그리겠는가?

</div>

??? success "풀이"
    **절차:** 모집단에서 크기 $n$인 표본을 뽑아 $\bar X$를 계산한다. 이를 $B$번 반복한다(예: $B = 10000$). 그 결과 얻은 $B$개의 값이 표본분포를 근사한다.

    $n = 5, 30, 100$에 대한 **그림 세 개**:

    1. $\bar X$의 $B$개 값에 대한 **히스토그램**에 이론적 정규분포 $N(\mu, \sigma^2/n)$을 겹쳐 그린다. 중심극한정리 근사의 품질을 눈으로 확인한다.
    2. 이론적 정규분포에 대한 **Q-Q 그림**: 중심극한정리 근사가 잘 맞으면 점들이 직선 위에 놓인다.
    3. 세 히스토그램을 나란히(또는 위아래로) 놓는 **삼중 비교**: (a) 중심이 $\mu$에 머물고, (b) 퍼짐이 $\sigma/\sqrt n$로 줄어들며, (c) 모양이 점점 정규분포에 가까워짐을 시각적으로 확인한다.

    **이런 모의실험에서 얻는 주요 관찰:**

    - 대칭인 모집단(균등, 정규)에서는 $n = 5$만으로도 근사적 정규성이 충분할 수 있다.
    - 치우친 모집단(지수, 로그정규)에서는 관례적으로 $n = 30$을 기준으로 삼지만 여전히 치우침이 눈에 띌 수 있다.
    - 꼬리가 두꺼운 모집단(Cauchy, $t_1$)에서는 어떤 $n$으로도 정규성이 나타나지 않는다. 중심극한정리는 유한한 분산을 요구한다.

    이런 모의실험 기반 진단이 "$n \ge 30$" 같은 경험 법칙을 무턱대고 적용하는 것보다 훨씬 믿을 만하다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff easy" title="쉬움"></span>
모표준편차가 $\sigma = 20$일 때 $n = 25, 100, 400$에 대한 표준오차를 각각 구하라. 표준오차를 지금의 절반으로 줄이려면 표본을 몇 배로 늘려야 하는가? 이 관계가 조사 설계에 어떤 뜻을 갖는가?

</div>

??? success "풀이"
    $\operatorname{SE} = \sigma/\sqrt n$이므로

    | $n$ | 25 | 100 | 400 |
    |---|---|---|---|
    | $\operatorname{SE}$ | 4.0 | 2.0 | 1.0 |

    표본을 4배로 늘릴 때마다 표준오차가 절반이 된다. 일반적으로 $\operatorname{SE}$를 $1/k$로 줄이려면 표본을 $k^2$배로 늘려야 한다.

    **조사 설계의 함의.** 정밀도의 한계수익이 빠르게 줄어든다.

    - 100명에서 400명으로 늘리면 오차한계가 절반이 된다. 비용 대비 효과가 크다.
    - 1000명에서 4000명으로 늘려도 역시 절반이 되지만, 추가 비용은 훨씬 크다.
    - 10000명에서 오차한계를 10분의 1로 줄이려면 100만 명이 필요하다. 사실상 불가능하다.

    그래서 전국 여론조사가 대개 1000명 남짓에서 멈춘다. 그 지점에서 표집오차가 약 3%인데, 그보다 더 줄이려 해도 비용이 급증할뿐더러 **표집오차가 아닌 다른 오차**(무응답, 표집틀 오차, 질문 문항의 영향)가 이미 3%를 넘기 때문이다. 표본을 키워도 이 오차들은 줄지 않는다.

    **표본 1만 명인 편향된 조사보다 표본 1000명인 제대로 된 무작위 조사가 낫다.** 1936년 《리터러리 다이제스트》가 240만 명을 조사하고도 대통령 선거 예측에 실패한 반면 갤럽이 5만 명으로 맞힌 것이 고전적인 사례다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
모집단 분포를 모르고 표본 하나만 가지고 있을 때 표본분포를 어떻게 근사하겠는가? 그 방법이 왜 통하는지 설명하라.

</div>

??? success "풀이"
    **부트스트랩**을 쓴다. 절차는 이렇다.

    1. 관측된 표본 $x_1,\dots,x_n$에서 **복원추출**로 크기 $n$인 재표본을 뽑는다.
    2. 그 재표본에서 관심 통계량 $\hat\theta^*$를 계산한다.
    3. 1~2를 $B$번(보통 1000~10000번) 되풀이한다.
    4. 얻은 $B$개의 $\hat\theta^*$의 분포를 $\hat\theta$의 표본분포의 근사로 쓴다.

    **왜 통하는가.** 원래 하고 싶은 일은 참 모집단 $F$에서 표본을 반복해서 뽑아 $\hat\theta$의 흩어짐을 보는 것인데, $F$를 모르므로 할 수 없다. 부트스트랩은 $F$ 대신 **경험분포** $\hat F_n$(관측값 각각에 확률 $1/n$을 주는 분포)을 쓴다.

    $$
    \underbrace{F \to \hat\theta\text{의 분포}}_{\text{알고 싶은 것}} \quad\longleftrightarrow\quad \underbrace{\hat F_n \to \hat\theta^*\text{의 분포}}_{\text{계산할 수 있는 것}}
    $$

    글리벤코-칸텔리 정리에 따라 $\hat F_n$이 $F$로 (균등하게) 수렴하므로, $\hat\theta$가 $F$에 대해 충분히 매끄럽게 의존한다면 두 분포도 가까워진다. 재표집은 $\hat F_n$에서 표본을 뽑는 일을 컴퓨터로 실행한 것일 뿐이다.

    **장점.** 통계량의 표집분포를 해석적으로 구할 필요가 없다. 중앙값, 사분위수 범위, 상관계수, 두 추정값의 비처럼 공식이 어렵거나 없는 경우에도 그대로 쓸 수 있다.

    **한계.** $\hat F_n$이 $F$를 잘 대신해야 하므로 $n$이 너무 작으면 믿기 어렵다. 또 최댓값처럼 분포의 경계에 의존하는 통계량, 꼬리가 아주 두꺼워 적률이 없는 경우, 관측이 독립이 아닌 경우(시계열·군집자료)에는 그대로 쓰면 안 된다. 시계열에는 블록 부트스트랩처럼 구조를 반영한 변형이 필요하다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
$\text{Exp}(\lambda)$ 모집단의 왜도는 2이다. 표본평균의 왜도가 $2/\sqrt n$임을 보이고, "$n \ge 30$이면 정규근사가 통한다"는 규칙을 평가하라.

</div>

??? success "풀이"
    $X_i$가 독립이고 같은 분포를 따르며 3차 중심적률이 $\mu_3$, 분산이 $\sigma^2$일 때, 합의 3차 중심적률은 $n\mu_3$이고 분산은 $n\sigma^2$이므로

    $$
    \text{왜도}(\bar X) = \frac{n\mu_3}{(n\sigma^2)^{3/2}} = \frac{1}{\sqrt n}\cdot\frac{\mu_3}{\sigma^3} = \frac{\text{왜도}(X)}{\sqrt n}
    $$

    이다. 지수분포는 왜도가 2이므로 $\bar X$의 왜도가 $2/\sqrt n$이다. $\square$

    | $n$ | 5 | 30 | 100 | 400 |
    |---|---|---|---|---|
    | 왜도 | 0.894 | 0.365 | 0.200 | 0.100 |

    **규칙의 평가.** $n=30$에서 왜도가 여전히 0.37이다. 이는 무시할 만한 값이 아니다. 정규분포는 왜도가 0이고, 보통 $|{\text{왜도}}| < 0.5$면 "대략 대칭"으로 보지만 그것은 서술적 기준이지 추론의 정확성을 보장하는 기준이 아니다.

    실제로 문제가 되는 곳은 **꼬리**다. 지수 모집단에서 $n=30$일 때 명목 95% $t$ 신뢰구간의 실제 포함확률은 93% 안팎으로 떨어지고, 단측 검정의 오류율은 한쪽으로 치우친다. 중앙 부근의 확률은 잘 맞아도 꼬리에서 어긋나는 것이 중심극한정리 근사의 전형적인 실패 방식이다.

    **더 나은 규칙.** 필요한 $n$은 모집단의 왜도에 달려 있다. 위 식에서 $\bar X$의 왜도를 0.1 아래로 두려면

    $$
    n \ge \left(\frac{\text{왜도}(X)}{0.1}\right)^2
    $$

    이 필요하다. 지수분포(왜도 2)면 $n \ge 400$, 왜도가 5인 분포면 $n \ge 2500$이다. 왜도가 1 이하인 완만한 분포라면 $n=30$으로 충분하다.

    한마디로 **"$n \ge 30$"은 모집단이 얼마나 치우쳤는지를 전혀 보지 않는 규칙이라 믿을 수 없다.** 표본의 왜도를 실제로 재 보거나, 모의실험으로 포함확률을 확인하거나, 부트스트랩처럼 정규성을 덜 요구하는 방법을 쓰는 편이 낫다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
$\bar X$의 **정확한** 분포가 알려진 모집단을 세 가지 들고 각각의 분포를 적어라. 코시분포에서는 무슨 일이 일어나는가?

</div>

??? success "풀이"
    합에 대해 닫혀 있는(재생성) 분포족이면 표본평균의 정확한 분포를 알 수 있다.

    **(1) 정규.** $X_i \sim N(\mu,\sigma^2)$이면

    $$
    \bar X \sim N\!\left(\mu, \frac{\sigma^2}{n}\right)
    $$

    이다. 근사가 아니라 모든 $n$에서 정확하다. $t$ 검정이 작은 표본에서도 정확한 것이 이 덕분이다.

    **(2) 감마(지수 포함).** $X_i \sim \text{Gamma}(k,\theta)$이면 $\sum X_i \sim \text{Gamma}(nk,\theta)$이므로

    $$
    \bar X \sim \text{Gamma}\!\left(nk,\ \frac{\theta}{n}\right)
    $$

    이다. 지수분포는 $k=1$인 경우다.

    **(3) 포아송.** $X_i \sim \text{Poisson}(\lambda)$이면 $\sum X_i \sim \text{Poisson}(n\lambda)$이므로 $\bar X$는 그 값을 $n$으로 나눈 것이다(격자 $0, 1/n, 2/n, \dots$ 위의 이산분포).

    그 밖에 이항, 음이항, 카이제곱, 균등(어윈-홀)도 정확한 분포가 알려져 있다.

    **코시분포.** $X_i \sim \text{Cauchy}(0,1)$이면

    $$
    \bar X \sim \text{Cauchy}(0,1)
    $$

    이다. **$n$과 전혀 무관하게 같은 분포다.** 표본을 100만 개 모아도 표본평균의 분포가 관측값 하나의 분포와 똑같다. 정밀도가 조금도 좋아지지 않는다.

    특성함수로 보면 명확하다. 코시분포의 특성함수는 $\varphi(t) = e^{-|t|}$이고

    $$
    \varphi_{\bar X}(t) = \left\{\varphi(t/n)\right\}^n = \left(e^{-|t|/n}\right)^n = e^{-|t|}
    $$

    으로 $n$이 지워진다.

    원인은 코시분포에 **평균이 없다**는 것이다. $E|X| = \infty$이므로 큰수의 법칙도 중심극한정리도 적용되지 않는다. 표본을 늘려도 더 극단적인 값이 그만큼 자주 나타나 평균을 계속 흔든다.

    실무적 교훈은 이렇다. **중심극한정리는 공짜가 아니다.** 유한한 분산이라는 조건이 필요하고, 그 조건이 깨지면 "표본을 늘리면 나아진다"는 직관 자체가 성립하지 않는다. 꼬리가 두꺼운 자료에서는 평균 대신 중앙값이나 절사평균처럼 강건한 통계량을 써야 하며, 코시분포에서도 표본중앙값은 $n$과 함께 제대로 수렴한다.

---

## 정리하며

**표본분포**는 통계량 자체의 분포다. 이 개념이 5장 전체이자 뒤에 나올 모든 추론의 토대다.

- **세 분포를 혼동하지 말 것.** 모집단분포(개별 관측이 흩어진 모양), 하나의 표본의 분포(실제로 손에 쥔 자료), 통계량의 표본분포(같은 크기의 표본을 무한히 반복했을 때 $\hat\theta$ 가 흩어지는 모양)는 서로 다르다. **셋 중 실제로 관측되는 것은 두 번째뿐이다.**
- **모의실험이 보여 준 것.** 균등·지수·베르누이 어느 모집단에서 출발하든 $\bar X$ 의 표본분포는 종 모양으로 다가간다. 모집단 모양은 씻겨 나가고 남는 것은 중심과 폭이다.
- **중심은 $\mu$ 로 유지되고 폭은 $\sigma/\sqrt n$ 으로 줄어든다.** 앞의 것이 불편성이고 뒤의 것이 정밀도다.
- **유한모집단에서는 정확히 셀 수 있다.** 공 세 개에서 두 개를 뽑는 예처럼 가능한 모든 표본을 나열하면 표본분포를 근사가 아니라 그대로 얻는다. 개념을 확인하기에 가장 좋은 방법이다.

**"반복추출"은 사고실험이다.** 현실에서 표본은 하나뿐이며, 그 하나가 어느 정도 흔들릴 수 있었는지를 말해 주는 것이 표본분포다. 신뢰구간과 $p$ 값의 의미가 모두 이 가상의 반복 위에 서 있다.

다음 절 **금융위기와 중심극한정리의 실패**는 이 그림이 언제 무너지는지를 본다. 관측들이 서로 독립이 아니면 표본분포가 정규로 가지 않는다.
