# 확률변수로서의 통계량

## 개요

**통계량**은 관측 자료의 임의의 함수이다. 자료가 확률추출에서 나오므로 통계량 자체도 **확률변수**이며, 그 값은 표본마다 달라진다. 이 점을 인식하는 것이 모든 표본분포 이론의 개념적 토대이다.

$$
\text{Population}
\;\xrightarrow{\text{draw sample}}\;
\mathbf{x} = (x_1, x_2, \dots, x_n)
\;\xrightarrow{\text{compute}}\;
T(\mathbf{x})
$$

표본을 뽑기 전에 $T(\mathbf{X})$는 확률변수이고, 표본을 관측한 뒤 $T(\mathbf{x})$는 실현된 하나의 수이다.

## 모집단에서 통계량까지

### 모집단, 표본, 통계량

| 개념 | 기호 | 설명 |
|---------|--------|-------------|
| 모집단 | — | 관심 대상이 되는 단위 전체의 모임 |
| 모수 | $\theta$ | 고정되어 있지만 미지인 모집단의 수치 요약 (예: $\mu$, $\sigma^2$, $p$) |
| 표본 | $\mathbf{X} = (X_1, \dots, X_n)$ | 모집단에서 뽑은 확률적 부분집합 |
| 통계량 | $T(\mathbf{X})$ | 표본의 임의의 함수 (미지 모수를 포함하지 않음) |
| 추정값 | $T(\mathbf{x})$ | 특정한 하나의 표본에 대한 통계량의 수치 |

### 핵심 구별

- **모수** $\theta$: 고정되어 있고 미지이며 모집단을 기술한다.
- **통계량** $T(\mathbf{X})$: 확률적이고 관측 가능하며 표본 자료로부터 계산된다.
- **추정량** $\hat{\theta}(\mathbf{X})$: $\theta$를 추정하기 위해 특별히 사용하는 통계량이다.

$$
\begin{array}{ccccc}
\text{Population}
&\rightarrow&
\text{Sample } \mathbf{X}
&\rightarrow&
\text{Statistic } T(\mathbf{X}) \\[6pt]
\text{Population}
&\rightarrow&
\text{Sample } \mathbf{X}
&\rightarrow&
\text{Estimator } \hat{\theta}(\mathbf{X})
\end{array}
$$

## 흔히 쓰는 통계량과 그 대상 모수

| 통계량 | 공식 | 대상 모수 |
|-----------|---------|-----------------|
| 표본평균 | $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$ | 모평균 $\mu$ |
| 표본분산 | $S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2$ | 모분산 $\sigma^2$ |
| 표본비율 | $\hat{p} = \frac{1}{n}\sum_{i=1}^n X_i$ (이항 자료) | 모비율 $p$ |
| 표본중앙값 | $\text{Med}(\mathbf{X})$ | 모집단 중앙값 |

이들 각각은 모집단 분포와 표본크기 $n$에 따라 분포가 결정되는 확률변수이다.

## 추정량과 그 성질

### 불편추정량

추정량 $\hat{\theta}$의 기댓값이 참 모수와 같으면 **불편**이라 한다:

$$
E[\hat{\theta}(\mathbf{X})] = \theta
$$

불편성은 무한히 반복추출할 때 추정량이 *평균적으로* 옳다는 뜻이다. $\theta$를 체계적으로 과대추정하지도 과소추정하지도 않는다.

**설정.** 0부터 32까지 번호가 매겨진 탁구공을 항아리에 넣는다. 모집단 중앙값은 16이다. 각 시행에서 5개를 비복원으로 뽑아 표본중앙값을 기록한다. 이를 50번 반복한다.

**질문.** 표본중앙값은 모집단 중앙값의 불편추정량인가?

**모의실험.**

<div class="codebox" markdown>

### 예제 1. 탁구공으로 보는 불편성 { .eg }

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
num_samples = 50

def main():
    # 모집단: 0부터 32까지 번호가 붙은 공 33개.
    # 개수가 홀수이므로 참 중앙값이 정확히 16으로 딱 떨어진다.
    balls = np.arange(33)
    print(f"Population median: {np.median(balls)}")

    # 크기 5짜리 표본을 50번 뽑아 그때마다 표본중앙값을 기록한다.
    # 표본이 달라지면 중앙값도 달라진다는 것,
    # 즉 **통계량이 확률변수라는 것**이 이 예제의 전부다.
    data = []
    for _ in range(num_samples):
        sample = np.random.choice(balls, size=5, replace=False)
        data.append(np.median(sample))

    print(f"Mean of sample medians: {np.mean(data):.2f}")

    # 값마다 몇 번 나왔는지 센다. 표본이 50개뿐이라 히스토그램보다
    # 점그림이 낫다(2.5절에서 본 대로 자료가 적을 때의 선택이다).
    data_dict = {}
    for num in data:
        data_dict[num] = data_dict.get(num, 0) + 1

    fig, ax = plt.subplots(figsize=(12, 3))
    # 같은 값을 세로로 쌓아 점그림을 만든다
    for num, freq in data_dict.items():
        ax.plot([num] * freq, range(1, freq + 1), 'ok')
    # 참 중앙값 16. 점들이 이 선 주위에 흩어지는지 확인한다.
    ax.plot([16, 16], [0, 5], "--r", alpha=0.3, label="True median")
    ax.legend()
    ax.set_title('Simulation-Based Distribution of Sample Median')
    ax.set_xlabel('Sample Median')
    ax.set_ylabel('Number of Samples')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position("zero")
    plt.show()

if __name__ == "__main__":
    main()
```

출력:

```
Population median: 16.0
Mean of sample medians: 16.44
```

![Simulation-Based Distribution of Sample Median](./img/statistics_as_rv_81.png)

**결론.** 표본중앙값의 표본분포는 근사적으로 대칭이고 참 중앙값 16을 중심으로 하며, 이는 표본중앙값이 모집단 중앙값의 불편추정량임을 시사한다.

</div>

## 최대가능도추정 (MLE)

### 소개

최대가능도추정(MLE)은 관측된 자료를 가장 그럴듯하게 만드는 값을 찾아 모수를 추정하는 방법이다. 일치성과 효율성을 비롯한 좋은 점근적 성질 때문에 자주 선호된다.

### 수학적 정식화

분포 $f(x \mid \theta)$에서 얻은 i.i.d. 관측값 $\mathbf{x} = (x_1, \dots, x_n)$이 주어졌을 때 MLE는:

$$
\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} \; L(\theta \mid \mathbf{x})
= \arg\max_{\theta} \prod_{i=1}^n f(x_i \mid \theta)
$$

계산의 편의를 위해 **로그가능도**를 최대화한다:

$$
\ell(\theta \mid \mathbf{x}) = \sum_{i=1}^n \log f(x_i \mid \theta)
$$

### 정규분포 모수의 MLE

$x^{(1)}, \dots, x^{(m)}$을 $N(\mu, \sigma^2)$에서 뽑은 i.i.d. 관측값이라 하자.

**가능도:**

$$
L(\mu, \sigma^2) = \prod_{i=1}^m \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x^{(i)} - \mu)^2}{2\sigma^2}\right)
$$

**로그가능도:**

$$
\ell(\mu, \sigma^2) = -\frac{1}{2\sigma^2}\sum_{i=1}^m (x^{(i)} - \mu)^2 - \frac{m}{2}\log\sigma^2 + \text{const.}
$$

**MLE 해:**

$$
\hat{\mu} = \frac{1}{m}\sum_{i=1}^m x^{(i)}, \qquad
\hat{\sigma}^2 = \frac{1}{m}\sum_{i=1}^m (x^{(i)} - \hat{\mu})^2
$$

!!! note
    $\sigma^2$의 MLE는 $m-1$이 아니라 $m$으로 나누므로 편향되어 있다. 불편추정량 $S^2$은 $m-1$로 나눈다(Bessel 수정).

### 베르누이 모수의 MLE

$x^{(1)}, \dots, x^{(m)}$을 $\text{Bernoulli}(p)$에서 뽑은 i.i.d. 관측값이라 하자.

**가능도:**

$$
L(p) = \prod_{i=1}^m p^{x^{(i)}}(1-p)^{1-x^{(i)}}
$$

**로그가능도:**

$$
\ell(p) = \sum_{i=1}^m \left[ x^{(i)} \log p + (1-x^{(i)})\log(1-p) \right]
$$

**MLE 해:**

$$
\hat{p} = \frac{1}{m}\sum_{i=1}^m x^{(i)}
$$

<div class="codebox" markdown>

#### 예제 2. 베르누이 모수의 최대가능도추정 { .eg }

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
p_true = 0.7
n_samples = 100

# 참 p = 0.7 인 동전을 100번 던진다. 물론 실제로는 이 값을 모른다.
coins = np.random.binomial(n=1, p=p_true, size=n_samples)

# p의 후보를 0.01부터 0.99까지 100개 늘어놓고 각각의 로그가능도를 잰다.
# 가능도는 "이 p라면 관측된 자료가 나올 확률이 얼마인가"이고,
# 각 던짐이 독립이므로 확률을 모두 곱해야 한다.
# 곱을 그대로 다루면 100번 곱하는 사이 값이 0으로 언더플로되므로
# 로그를 취해 **합**으로 바꾼다. 이것이 로그가능도를 쓰는 실용적 이유다.
#   앞면(coins=1)이면 log(p), 뒷면(coins=0)이면 log(1-p) 를 더한다.
ps = np.linspace(0.01, 0.99, 100)
log_likelihoods = np.array([
    np.sum(coins * np.log(p) + (1 - coins) * np.log(1 - p))
    for p in ps
])

# 최대가능도추정: 가능도를 가장 크게 만드는 p를 고른다.
# 로그는 단조증가 함수이므로 로그가능도를 최대화하는 것과 결과가 같다.
# 여기서는 격자에서 찾지만, 해석적으로 풀면 p-hat = 표본비율이 나온다.
idx = np.argmax(log_likelihoods)
mle_p = ps[idx]

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(ps, log_likelihoods, label="Log-likelihood")
ax.axvline(mle_p, color='r', linestyle='--', label=f"MLE: p = {mle_p:.2f}")
ax.legend(loc="lower right")
ax.set_xlabel("Probability (p)")
ax.set_ylabel("Log-likelihood")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
```

![확률변수로서의 통계량](./img/statistics_as_rv_192.png)

</div>

### 포획–재포획의 MLE

**포획–재포획법**은 두 단계의 표본추출로 모집단 크기 $N$을 추정한다:

1. $M$마리를 포획하여 표시한 뒤 놓아 준다.
2. $n$마리를 다시 포획하는데 그중 $m$마리가 표시되어 있다.

재포획에서 표시된 개체의 수는 **초기하분포**를 따른다:

$$
P(m \mid N) = \frac{\binom{M}{m}\binom{N-M}{n-m}}{\binom{N}{n}}
$$

$N$의 MLE는:

$$
\hat{N} = \frac{M \cdot n}{m}
$$

이는 비례 관계 $m/n \approx M/N$에서 따라 나온다.

**예.** $M = 50$마리의 물고기에 표시하고, 두 번째 표본 $n = 40$에서 $m = 10$마리가 표시되어 있었다면:

$$
\hat{N} = \frac{50 \times 40}{10} = 200
$$

<div class="codebox" markdown>

#### 예제 3. 포획-재포획의 최대가능도추정 { .eg }

```python
import matplotlib.pyplot as plt
from scipy import special

def prob(n, c, r, t):
    """포획-재포획의 초기하확률.

    n: 전체 개체수(우리가 추정하려는 미지수)
    c: 1차에서 잡아 표시한 수
    r: 2차에서 잡은 수
    t: 2차에서 잡힌 것 중 표시가 있던 수

    2차 표본 r마리를 고르는 모든 방법 중,
    표시된 것 t마리와 안 된 것 r-t마리를 고르는 방법의 비율이다.
    """
    return special.comb(n - c, r - t) * special.comb(c, t) / special.comb(n, r)

def capture_recapture(c=50, r=40, t=10):
    # 가능한 최소 개체수. 표시된 50마리와 2차에서 새로 잡힌 30마리는
    # 서로 다른 개체이므로 최소 50 + 40 - 10 = 80마리는 있어야 한다.
    min_n = c + r - t
    ns = range(min_n, 10 * min_n)

    # n을 바꿔 가며 가능도를 계산한다.
    # **n은 모수이지 확률변수가 아니다.** 자료 (c, r, t)는 고정해 두고
    # "어떤 n이 이 자료를 가장 그럴듯하게 만드는가"를 묻는 것이다.
    probs = [prob(n, c, r, t) for n in ns]

    mle_idx = probs.index(max(probs))
    mle_n = mle_idx + min_n
    # 직관적인 답 c*r/t = 50*40/10 = 200 과 비교해 보라.
    print(f"MLE of N: {mle_n}")
    return list(ns), probs, mle_n

ns, probs, mle_n = capture_recapture()

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(ns, probs, label='Likelihood')
ax.axvline(mle_n, color='r', linestyle='--', label=f'MLE: N = {mle_n}')
ax.set_xlabel('Population Size (N)')
ax.set_ylabel('Probability')
ax.set_title('Capture–Recapture: Likelihood vs Population Size')
ax.legend()
plt.show()
```

출력:

```
MLE of N: 199
```

![Capture–Recapture: Likelihood vs Population Size](./img/statistics_as_rv_278.png)

</div>

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
**포획–재포획.** $M = 50$마리의 물고기에 표시하여 놓아 주고, 나중에 $n = 40$마리를 잡았더니 $m = 10$마리가 표시되어 있었다. (a) 초기하 가능도 $P(m \mid N)$을 유도하라. (b) MLE $\hat N$을 구하라.

</div>

??? success "풀이"
    (a) $P(m \mid N) = \binom{M}{m} \binom{N - M}{n - m} / \binom{N}{n}$이며 초기하분포이다. 주어진 값을 넣으면 $P(N) = \binom{50}{10}\binom{N-50}{30}/\binom{N}{40}$.

    (b) 로그가능도를 미분하여 풀면 $\hat N = Mn/m = 50 \cdot 40 / 10 = 200$. 이 MLE는 "(모집단에서 표시된 수) × (잡은 총 수) / (잡힌 표시된 수)"라는 직관적인 형태이며, 비례 추론에 해당한다.

    포획–재포획은 야생동물 개체수 추정의 기본 방법이다. 변형(폐쇄/개방 모집단, 여러 번의 재포획, 표지 손실)을 통해 풍부한 추정량 계열이 만들어진다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
**베르누이의 MLE.** 동전을 100번 던져 앞면이 40번 나왔다. (a) 가능도 $L(p)$를 쓰라. (b) $\hat p_{\text{MLE}}$를 구하라.

</div>

??? success "풀이"
    (a) $L(p) = \binom{100}{40} p^{40}(1-p)^{60} \propto p^{40}(1-p)^{60}$ (상수 배수는 최대화에 영향을 주지 않는다).

    (b) 로그가능도: $\ell(p) = 40\ln p + 60\ln(1-p)$. 미분하면 $40/p - 60/(1-p) = 0 \Rightarrow p = 0.4$.

    $\hat p_{\text{MLE}} = 0.4$로 표본비율과 같다. 일반적으로 $X \sim \mathrm{Binomial}(n, p)$에 대해 MLE는 $\hat p = X/n$이다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
**정규분포 모수의 MLE.** i.i.d. $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$이 주어졌을 때 두 MLE를 모두 구하라.

</div>

??? success "풀이"
    로그가능도: $\ell(\mu, \sigma^2) = -(n/2)\ln(2\pi\sigma^2) - (1/(2\sigma^2))\sum(X_i - \mu)^2$.

    $\partial \ell/\partial \mu = (1/\sigma^2)\sum(X_i - \mu) = 0 \Rightarrow \hat\mu = \bar X$.

    $\partial \ell/\partial \sigma^2 = -n/(2\sigma^2) + (1/(2\sigma^4))\sum(X_i - \hat\mu)^2 = 0 \Rightarrow \hat\sigma^2 = (1/n)\sum(X_i - \bar X)^2$.

    **참고:** MLE는 $n - 1$이 아니라 $n$으로 나눈다. 따라서 $\hat\sigma^2_{\text{MLE}}$는 편향되어 있다: $\mathbb{E}[\hat\sigma^2] = ((n-1)/n)\sigma^2$. 불편추정을 하려면 $s^2 = \sum(X_i - \bar X)^2/(n - 1)$을 사용한다(Bessel 수정).

    MLE와 불편추정량의 구별은 반복해서 나타나는 주제이다. MLE는 점근적으로 최적이지만 유한표본에서는 편향될 수 있다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
**충분통계량.** 조건부분포 $X \mid T$가 $\theta$에 의존하지 않으면 통계량 $T(X)$가 $\theta$에 대해 **충분**하다고 한다. **Fisher-Neyman 인수분해 정리**를 서술하고, 이를 사용하여 $X_i \sim \mathrm{Poisson}(\lambda)$일 때 $\sum X_i$가 $\lambda$에 대해 충분함을 확인하라.

</div>

??? success "풀이"
    **Fisher-Neyman 인수분해 정리:** $T(X)$가 $\theta$에 대해 충분일 필요충분조건은 결합밀도가 다음과 같이 인수분해되는 것이다.

    $$
    f(x \mid \theta) = g(T(x), \theta) \cdot h(x)
    $$

    여기서 $g$는 $T(x)$를 통해서만 $\theta$에 의존하고 $h$는 $\theta$에 의존하지 않는다.

    **포아송의 경우:** $f(x_1, \ldots, x_n \mid \lambda) = \prod_i \frac{e^{-\lambda} \lambda^{x_i}}{x_i!} = e^{-n\lambda} \lambda^{\sum x_i} / \prod_i x_i!$.

    가능도가 $g(\sum x_i, \lambda) \cdot h(x) = (e^{-n\lambda} \lambda^{\sum x_i}) \cdot (1/\prod x_i!)$로 인수분해된다. 따라서 $T(X) = \sum X_i$는 충분통계량이다.

    **의의:** $\lambda$에 관해 $X_1, \ldots, X_n$이 담고 있는 정보가 모두 $\sum X_i$에 집약되어 있다. MLE는 자료에 오직 $T$를 통해서만 의존하며, 추론에 자료 전체가 필요하지 않다. 이것이 Rao-Blackwell 정리를 통한 효율적 추정의 토대이다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
**Fisher 정보량.** $\sigma^2$이 알려진 $X \sim N(\mu, \sigma^2)$에 대해 $\mu$에 관한 Fisher 정보량을 계산하라. 이 양이 왜 중요한가?

</div>

??? success "풀이"
    점수함수: $\partial \log f/\partial \mu = (x - \mu)/\sigma^2$.

    Fisher 정보량: $I(\mu) = \mathbb{E}\!\left[(\partial \log f / \partial \mu)^2\right] = \mathbb{E}[(X - \mu)^2/\sigma^4] = \sigma^2/\sigma^4 = 1/\sigma^2$.

    크기 $n$인 i.i.d. 표본에 대해서는 $I_n(\mu) = n/\sigma^2$이다.

    **왜 중요한가: Cramér-Rao 하한.** $\mu$의 임의의 불편추정량의 분산은 적어도 $1/I_n(\mu) = \sigma^2/n$이다. $\mathrm{Var}(\bar X) = \sigma^2/n$이 정확히 성립하므로 $\bar X$는 Cramér-Rao 하한을 달성하며 **효율적**이다. 어떤 불편추정량도 이보다 나을 수 없다.

    Fisher 정보량은 모수에 관해 표본이 담은 "정보량"을 정량화하고 추정량 분산의 하한을 준다. MLE의 점근이론, 실험설계, 정보기하학에서 쓰인다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
**MLE의 점근정규성.** 일반적인 결과를 서술하라: $\sqrt n (\hat\theta_{\text{MLE}} - \theta) \xrightarrow{d} N(0, 1/I(\theta))$이며 $I(\theta)$는 관측값 하나당 Fisher 정보량이다. 포아송에 대해 확인하라.

</div>

??? success "풀이"
    Poisson($\lambda$)에서 $\hat\lambda_{\text{MLE}} = \bar X$이다. 관측값 하나당 Fisher 정보량은 $I(\lambda) = 1/\lambda$이다($\partial \log f/\partial \lambda = X/\lambda - 1$이고 $\mathbb{E}[(X/\lambda - 1)^2] = \mathrm{Var}(X)/\lambda^2 = 1/\lambda$이므로).

    점근분포: $\sqrt n(\bar X - \lambda) \xrightarrow{d} N(0, \lambda)$이며, 이는 $1/I(\lambda) = \lambda$와 일치한다.

    중심극한정리로 직접 확인: $\mathrm{Var}(X_i) = \lambda$인 $\bar X = (1/n)\sum X_i$이므로 중심극한정리에 의해 $\sqrt n(\bar X - \lambda) \to N(0, \lambda)$. ✓

    **일반적 의의:** MLE는 점근적으로 정규분포를 따르며 그 분산은 관측값 하나당 Fisher 정보량의 역수이다. 이로부터 다음을 얻는다:

    - **점근 신뢰구간:** $\hat\theta \pm 1.96/\sqrt{n I(\hat\theta)}$ ($I(\theta)$의 대입추정값으로 $I(\hat\theta)$를 사용).
    - **점근 효율성:** MLE는 점근적으로 Cramér-Rao 하한을 달성한다.

    이 결과들이 현대 통계학에서 가능도 기반 추론이 중심적 위치를 차지하는 이유이다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
**적률법.** 표본에서 $\bar x = 4.2$, $s^2 = 8.4$를 얻었고 자료가 $\text{Gamma}(\text{형상}=k,\ \text{척도}=\theta)$에서 나왔다고 하자. 적률법으로 $k$와 $\theta$를 추정하라. 최대가능도추정과 견주면 어떤 장단점이 있는가?

</div>

??? success "풀이"
    감마분포의 적률은 $E[X] = k\theta$, $\operatorname{Var}(X) = k\theta^2$이다. 표본적률과 맞추면

    $$
    \hat k\hat\theta = 4.2, \qquad \hat k\hat\theta^2 = 8.4
    $$

    이고, 둘째 식을 첫째 식으로 나누면

    $$
    \hat\theta = \frac{s^2}{\bar x} = \frac{8.4}{4.2} = 2.0, \qquad \hat k = \frac{\bar x}{\hat\theta} = \frac{4.2}{2.0} = 2.1
    $$

    이다.

    **장점.** 계산이 한 줄이다. 감마분포의 최대가능도방정식은

    $$
    \ln\hat k - \psi(\hat k) = \ln\bar x - \overline{\ln x}
    $$

    꼴로 디감마함수가 들어 있어 수치적으로 풀어야 한다. 적률법 추정값은 그 반복의 좋은 출발점이 된다. 또 분포 전체를 가정하지 않고 적률만 맞추므로 모형이 조금 어긋나도 크게 망가지지 않는다.

    **단점.** 일반적으로 최대가능도추정보다 **비효율적**이다. 자료의 정보를 처음 몇 개의 적률로만 요약하기 때문이다. 고차 적률을 쓰면 표집변동이 커져 더 불안정해지고, 추정값이 모수공간 밖으로 나가는 일도 있다(예: 분산 추정이 음수). 또 충분통계량을 쓰지 않으므로 정보 손실이 생긴다.

    실무에서는 적률법을 **초기값이나 빠른 점검**으로 쓰고 최종 추정은 최대가능도로 하는 조합이 흔하다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
정규모집단에서 $\hat\sigma^2_c = c\sum_i(X_i-\bar X)^2$ 꼴의 추정량을 생각하자. 평균제곱오차를 최소로 하는 $c$를 구하고, $c = 1/(n-1)$(불편)과 $c = 1/n$(최대가능도)과 견주어라.

</div>

??? success "풀이"
    $W = \sum_i(X_i-\bar X)^2$로 두면 $W/\sigma^2 \sim \chi^2_{n-1}$이므로

    $$
    E[W] = (n-1)\sigma^2, \qquad \operatorname{Var}(W) = 2(n-1)\sigma^4
    $$

    이다. 따라서

    $$
    \text{MSE}(c) = \operatorname{Var}(cW) + \{E[cW]-\sigma^2\}^2 = \left[2(n-1)c^2 + \{(n-1)c-1\}^2\right]\sigma^4
    $$

    이다. $c$로 미분해 0으로 두면

    $$
    4(n-1)c + 2(n-1)\{(n-1)c-1\} = 0 \implies 2c + (n-1)c = 1 \implies c^* = \frac{1}{n+1}
    $$

    을 얻는다.

    **세 추정량.** MSE를 $\sigma^4$ 단위로 적으면

    | $c$ | 이름 | 편향 | MSE |
    |---|---|---|---|
    | $1/(n-1)$ | 불편 | $0$ | $\dfrac{2}{n-1}$ |
    | $1/n$ | 최대가능도 | $-\sigma^2/n$ | $\dfrac{2n-1}{n^2}$ |
    | $1/(n+1)$ | 최소 MSE | $-2\sigma^2/(n+1)$ | $\dfrac{2}{n+1}$ |

    $n=10$이면 각각 $0.2222$, $0.1900$, $0.1818$로, **불편추정량이 셋 중 가장 나쁘다.**

    **뜻.** 불편성은 좋은 성질이지만 최적성의 기준은 아니다. 편향을 조금 받아들이고 분산을 더 줄이면 전체 오차가 작아질 수 있으며, 이것이 **편향-분산 맞바꿈**이다. 능형회귀, 라소, 축소추정, 정칙화가 모두 같은 거래를 한다.

    그런데도 실무에서 $n-1$을 쓰는 이유가 있다. 첫째, 불편성이 여러 표본을 결합할 때 좋은 성질을 준다(분산분석에서 제곱평균들을 더하고 나눌 때 편향이 누적되지 않는다). 둘째, 최소 MSE인 $c=1/(n+1)$은 모집단이 정규일 때만 최적이라 일반성이 없다. 셋째, $n$이 크면 셋의 차이가 $O(1/n^2)$로 사라진다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
**MLE의 불변성.** $\hat\theta$가 $\theta$의 MLE이면 임의의 함수 $g$에 대해 $g(\hat\theta)$가 $g(\theta)$의 MLE임을 설명하라. $X_i \sim \text{Bernoulli}(p)$에서 오즈 $p/(1-p)$의 MLE를 구하라. 불편성에도 같은 성질이 있는가?

</div>

??? success "풀이"
    **불변성.** $g$가 일대일이면 모수를 $\eta = g(\theta)$로 바꿔 쓴 가능도가 $L^*(\eta) = L(g^{-1}(\eta))$이므로, $L$을 최대로 하는 $\hat\theta$에서 $L^*$가 최대가 되고 그 위치가 $\hat\eta = g(\hat\theta)$이다. $g$가 일대일이 아니면 유도가능도 $L^*(\eta) = \sup_{\theta:\,g(\theta)=\eta}L(\theta)$로 정의하면 같은 결론이 나온다.

    **오즈의 MLE.** $\hat p = \bar X$이므로

    $$
    \widehat{\text{오즈}} = \frac{\bar X}{1-\bar X}
    $$

    이다. 100번 중 40번 성공이면 $0.4/0.6 = 2/3$이다. 따로 최적화할 필요가 없다.

    **불편성에는 없다.** $E[\hat\theta] = \theta$라 해도 일반적으로 $E[g(\hat\theta)] \ne g(\theta)$이다. 옌센 부등식에 따라 $g$가 볼록이면 $E[g(\hat\theta)] \ge g(E[\hat\theta]) = g(\theta)$로 위로 치우친다.

    위 예에서도 $x/(1-x)$가 $(0,1)$에서 볼록이므로 오즈의 MLE는 오즈를 **과대추정**한다. 게다가 $\bar X = 1$이면 값이 무한대가 되어 버린다. 그래서 로그오즈를 다룰 때는 $\bar X$ 대신 $(k+0.5)/(n+1)$처럼 살짝 보정한 값(하딘-피터스 보정)을 쓰기도 한다.

    이 대비가 두 성질의 성격을 잘 보여 준다. **불변성은 "어떤 모수화로 문제를 적든 답이 같다"는 뜻**이고, 불편성은 특정 모수화에 묶여 있다. 표준편차의 불편추정량이 분산의 불편추정량의 제곱근이 아니라는 사실도 같은 이야기다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
**일치성과 불편성은 다르다.** (가) 불편이지만 일치가 아닌 추정량, (나) 편향되었지만 일치인 추정량의 예를 각각 들어라.

</div>

??? success "풀이"
    **(가) 불편이지만 일치가 아닌 경우.** $X_1,\dots,X_n \sim N(\mu,\sigma^2)$에서 $\hat\mu = X_1$(첫 관측값만 쓴다)을 생각하자. $E[X_1] = \mu$로 불편이지만, $n$이 아무리 커져도 분포가 $N(\mu,\sigma^2)$ 그대로라 $\mu$로 수렴하지 않는다. 일치가 아니다.

    자료를 버리는 극단적인 예지만 요점은 분명하다. **불편성은 표본크기가 커지는 것과 아무 상관이 없는, 한 표본크기에서의 성질이다.**

    **(나) 편향되었지만 일치인 경우.** 같은 설정에서 최대가능도 분산추정량

    $$
    \hat\sigma^2_{\text{MLE}} = \frac1n\sum_i (X_i-\bar X)^2
    $$

    은 $E[\hat\sigma^2_{\text{MLE}}] = \frac{n-1}{n}\sigma^2$로 편향되어 있다. 그러나 편향이 $-\sigma^2/n \to 0$이고 분산도 0으로 가므로 $\hat\sigma^2_{\text{MLE}} \xrightarrow{p} \sigma^2$이다. 일치추정량이다.

    다른 예로 연습문제 1의 포획-재포획 추정량 $\hat N = Mn/m$도 유한표본에서 편향되어 있지만 일치이다.

    **정리.** 두 성질은 서로 독립적이다.

    - **불편성**: $E[\hat\theta_n] = \theta$. 고정된 $n$에서의 성질이며, "평균적으로 맞다"는 뜻이다.
    - **일치성**: $\hat\theta_n \xrightarrow{p} \theta$. $n\to\infty$에서의 성질이며, "자료를 모으면 결국 맞는다"는 뜻이다.

    실무에서 더 중요한 쪽은 **일치성**이다. 일치가 아닌 추정량은 자료를 아무리 모아도 참값에 다가가지 않으므로 쓸 수 없다. 편향은 크기가 작고 $n$과 함께 사라지면 대개 감수할 만하며, 연습문제 8에서 보았듯 일부러 편향을 들여 오차를 줄이기도 한다.

    충분조건 하나를 기억해 두면 편하다. **편향과 분산이 모두 0으로 가면 일치이다**(MSE 수렴이 확률수렴을 함의하므로).

---

## 정리하며

| 개념 | 의미 |
|---------|---------|
| 통계량 | 표본의 임의의 함수. 자료를 관측하기 전에는 확률변수 |
| 추정량 | 모수를 추정하는 데 쓰이는 통계량 |
| 불편 | $E[\hat{\theta}] = \theta$ — 평균적으로 옳다 |
| MLE | 관측된 자료의 가능도를 최대화하는 모수값 |

통계량이 확률변수임을 이해하는 것이 추론통계학 전체로 들어가는 문이다. 신뢰구간, 가설검정, 예측구간은 모두 해당 통계량의 분포를 알거나 근사하는 데 의존한다.
