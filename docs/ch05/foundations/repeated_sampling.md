# 반복추출 개념

## 개요

앞 절에서 표본을 다시 뽑을 때마다 평균이 달라지는 것을 보았다. 68.3이었다가 67.9였다가 69.1이었다. 그렇다면 다음 물음이 따라온다. **그 값들은 어떤 모양으로 흩어지는가.**

답을 얻는 방법은 원리만 보면 단순하다. 같은 모집단에서 같은 크기의 표본을 뽑아 통계량을 계산하는 일을 끝없이 되풀이하고, 그렇게 쌓인 값들이 이루는 분포를 보면 된다. 그 분포를 통계량의 **표본분포**라 부른다. 그림의 오른쪽 끝에 모이는 것이 그것이다.

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

이 그림 하나가 추론통계학 전체를 지탱한다. 표본에서 얻은 값으로 모평균이나 모분산을 추정할 수 있는 것도, 그 추정이 얼마나 정밀한지를 표준오차로 잴 수 있는 것도, 신뢰구간을 그리고 가설검정으로 판단을 내릴 수 있는 것도 모두 통계량이 어떻게 흩어지는지를 알기 때문이다. 표본분포를 모르면 손에 쥔 숫자 하나가 얼마나 믿을 만한지 말할 길이 없다.

## 이름이 닮은 세 분포

바로 여기서 독자가 가장 자주 걸려 넘어진다. 이름이 서로 닮은 분포가 셋 있는데 가리키는 대상이 전혀 다르기 때문이다. 하나씩 짚어 보자.

첫째는 **모집단분포**다. 모집단에 속한 개체들이 어떤 값을 어떤 비율로 갖는지를 나타낸다. 전국 성인 남성의 키가 어떻게 퍼져 있는가가 이것이다. 우리가 표본을 뽑든 말든 그 자리에 있고, 몇 번을 뽑아도 변하지 않으며, 그리고 우리는 이것을 직접 보지 못한다.

둘째는 **한 표본의 분포**다. 실제로 뽑아 온 값 $n$개가 이루는 분포이며, 히스토그램을 그릴 수 있는 것은 오직 이것뿐이다. 셋 가운데 유일하게 우리 손에 쥐어져 있는 것이 이것이고, 표본을 키우면 첫째, 곧 모집단분포의 모양을 점점 닮아 간다.

셋째가 **표본분포**다. 여기서 뽑히는 것은 개체가 아니라 **표본 전체**다. 같은 크기의 표본을 무한히 되풀이해 뽑고 그때마다 통계량 하나를 계산한다고 할 때, 그 통계량들이 이루는 분포가 표본분포다. 따라서 표본분포에서 점 하나는 관측값 하나가 아니라 **표본 하나**에 대응한다. 앞의 둘과 층이 다른 지점이 바로 여기다.

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

셋의 관계를 한 문장으로 줄이면 이렇다. **관측되는 것은 둘째뿐이고, 알고 싶은 것은 첫째이며, 그 사이를 이어 주는 것이 셋째다.** 표본분포는 현실에서 한 번도 관측되지 않는다. 표본을 무한히 되풀이해 뽑는 일은 실제로 일어나지 않기 때문이다. 그런데도 그것을 계산할 수 있고, 계산할 수 있기 때문에 하나뿐인 표본으로 모집단을 말할 수 있다.

퍼진 정도를 견주어 보면 층의 차이가 더 분명해진다. 모집단의 표준편차가 $\sigma$이면 한 표본 안의 값들도 대체로 그만큼 퍼져 있다. 그런데 표본평균들이 퍼진 정도는 $\sigma/\sqrt n$이어서 $n$이 커질수록 좁아진다. 같은 자료를 두고도 "개별 값이 얼마나 흩어져 있는가"와 "평균이 얼마나 흔들리는가"의 답이 다른 것이며, 5.3절에서 표준편차와 표준오차를 굳이 갈라 부르는 이유가 이것이다.

말로 나눈 셋을 그림 하나에 나란히 놓으면 훨씬 빨리 붙는다. 이어지는 세 모의실험은 구조가 모두 같다. 위 칸에 모집단분포를, 가운데 칸에 표본 하나를 점으로, 아래 칸에 표본평균 1만 개의 히스토그램을 그린다. 바뀌는 것은 오직 모집단의 모양뿐이다.

## 평평한 모집단에서 시작한다

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
    # 균등분포에서 큰 모집단을 만든다. 1만 개면 사실상 무한 모집단으로 본다.
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

세 칸을 위에서 아래로 훑어보라. 모집단은 0과 1 사이에 고르게 퍼진 평평한 모양이고, 가운데의 표본 다섯 개도 그 구간 아무 데나 흩어져 있다. 그런데 맨 아래 칸의 표본평균들은 평평하지도 않고 넓지도 않다. 종 모양이며 0.5 언저리에 훨씬 좁게 모여 있다. 모집단에서 물려받지 않은 모양이 통계량의 층에서 새로 생겨난 것이고, 중심극한정리를 미리 엿보는 셈이다.

</div>

## 치우친 모집단에서도 같은 일이 일어난다

균등분포는 대칭이라 쉬운 상대였을지 모른다. 이번에는 오른쪽으로 길게 늘어진 지수 모집단을 놓고 표본크기를 30으로 키워 본다. 바뀐 것은 이 두 가지뿐이다.

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

맨 위 칸의 모집단은 0 근처에 몰려 있고 오른쪽으로 꼬리가 길게 뻗어 누가 보아도 정규분포와 거리가 멀다. 그런데 $n = 30$에서 표본평균들은 다시 근사적으로 정규분포다. 치우친 모집단에서 출발했는데도 통계량의 층에서는 치우침이 대부분 씻겨 나갔다. 중심극한정리가 작동하는 모습이 이것이다.

</div>

## 값이 둘뿐인 모집단에서도 마찬가지다

마지막으로 연속도 아닌 모집단을 보자. 값이 0과 1 둘밖에 없고 1이 나올 확률이 0.3인 베르누이 모집단이다. 앞의 두 경우보다 모집단과 정규분포의 거리가 더 멀어 보인다.

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

위 칸의 모집단은 막대 두 개가 전부이고, 가운데 칸의 표본 30개도 0 아니면 1이라 점이 두 자리에만 겹쳐 찍힌다. 그런데 아래 칸의 표본평균들은 0.3 근처에 종 모양으로 모여 있다. 여기서도 모집단의 모양은 사라졌다. 다만 표본평균이 가질 수 있는 값이 $0/30, 1/30, 2/30, \dots$뿐이라 히스토그램이 여전히 띄엄띄엄하다. 모집단에서 물려받은 **이산성**만은 끝까지 남는 것이며, 5.5절에서 $\hat p$를 다룰 때 다시 문제가 된다.

</div>

## 유한모집단에서는 근사가 아니라 정확히 셀 수 있다

지금까지는 컴퓨터로 표본을 1만 번 뽑아 표본분포를 **근사**했다. 그런데 모집단이 아주 작으면 되풀이해 뽑을 필요조차 없다. 가능한 표본을 빠짐없이 적고 각각의 확률을 붙이면 표본분포가 그대로 나오기 때문이다. 개념을 처음 익힐 때 이만한 방법이 없다.

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

이 절에서 가져갈 것 하나만 고르라면 세 분포의 구별이다. 모집단분포는 개별 관측이 흩어진 모양이고, 한 표본의 분포는 실제로 손에 쥔 자료이며, 표본분포는 같은 크기의 표본을 무한히 되풀이했을 때 $\hat\theta$가 흩어지는 모양이다. 앞의 둘은 값 하나가 점 하나이지만 셋째는 표본 하나가 점 하나라는 점에서 층이 다르고, 셋 가운데 실제로 관측되는 것은 둘째뿐이다. 이 셋을 섞어 쓰는 순간 뒤의 모든 이야기가 흐려진다.

세 모의실험이 보여 준 것은 출발점이 무엇이든 $\bar X$의 표본분포가 종 모양으로 다가간다는 사실이다. 평평한 균등분포에서 출발하든, 오른쪽으로 길게 늘어진 지수분포에서 출발하든, 값이 둘뿐인 베르누이에서 출발하든 모집단의 모양은 씻겨 나가고 중심과 폭만 남는다. 중심은 $\mu$에 머물고 폭은 $\sigma/\sqrt n$으로 줄어드는데, 앞의 것이 불편성이고 뒤의 것이 정밀도다. 공 세 개에서 두 개를 뽑는 마지막 예처럼 모집단이 작을 때는 가능한 표본을 모두 나열해 이 분포를 근사가 아니라 그대로 얻을 수도 있다.

끝으로 "반복추출"이 사고실험이라는 점을 잊지 말아야 한다. 현실에서 표본은 하나뿐이고 아무도 1만 번 뽑지 않는다. 그 하나가 어느 정도까지 달라질 수 있었는지를 말해 주는 것이 표본분포이며, 신뢰구간과 $p$ 값의 의미가 모두 이 가상의 반복 위에 서 있다.

다음 절 **금융위기와 중심극한정리의 실패**는 이 그림이 언제 무너지는지를 본다. 관측들이 서로 독립이 아니면 표본분포가 정규로 가지 않는다.
