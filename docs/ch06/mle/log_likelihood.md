# 로그가능도 시각화

## 개요

**로그가능도함수**는 가능도에 로그를 취한 것으로 최대가능도추정의 주된 도구이다. 가능도 자체 대신 로그가능도로 작업하면 작은 확률을 많이 곱할 때 생기는 수치적 언더플로를 피할 수 있고 곱이 합으로 바뀌어 계산과 미분이 모두 간단해진다. 이 페이지에서는 Bernoulli 동전 던지기 예제로 로그가능도의 구성, 시각화, MLE 추출을 보인다.

## 가능도에서 로그가능도로

분포 $f(x; \theta)$에서 얻은 i.i.d. 관측값 $x_1, \ldots, x_n$이 주어졌을 때 **가능도**는:

$$
L(\theta) = \prod_{i=1}^n f(x_i; \theta)
$$

**로그가능도**는:

$$
\ell(\theta) = \log L(\theta) = \sum_{i=1}^n \log f(x_i; \theta)
$$

$\log$가 순증가함수이므로 MLE는 둘에 대해 같다:

$$
\hat{\theta}_{\text{MLE}} = \arg\max_\theta L(\theta) = \arg\max_\theta \ell(\theta)
$$

!!! warning "왜 가능도를 직접 쓰지 않는가?"
    $p = 0.7$인 $n = 100$개의 Bernoulli 관측값에서 가능도는 0과 1 사이 수 100개의 곱이다. 이 곱은 $10^{-30}$ 규모로 부동소수점 언더플로 문턱보다 훨씬 작다. 로그가능도는 로그확률의 합으로 작업하여 이를 피한다.

## Bernoulli 로그가능도

$X_i \sim \text{Bernoulli}(p)$에서 PMF는:

$$
f(x; p) = p^x(1-p)^{1-x}, \quad x \in \{0, 1\}
$$

관측값 하나의 로그확률은:

$$
\log f(x; p) = x\log p + (1-x)\log(1-p)
$$

$n$개 관측값에 대한 로그가능도는:

$$
\ell(p) = \sum_{i=1}^n [x_i \log p + (1 - x_i)\log(1-p)] = k\log p + (n-k)\log(1-p)
$$

여기서 $k = \sum_{i=1}^n x_i$는 성공 횟수이다.

## MLE의 유도

(로그가능도의 도함수인) 점수를 0으로 두면:

$$
\ell'(p) = \frac{k}{p} - \frac{n-k}{1-p} = 0
$$

$$
k(1-p) = (n-k)p \quad \Rightarrow \quad k = np \quad \Rightarrow \quad \hat{p}_{\text{MLE}} = \frac{k}{n}
$$

2계도함수가 최댓값임을 확인해 준다:

$$
\ell''(p) = -\frac{k}{p^2} - \frac{n-k}{(1-p)^2} < 0
$$

## 구현과 시각화

```python
import numpy as np

def compute_log_prob(coin, p):
    """Log-probability of a single Bernoulli outcome."""
    return coin * np.log(p) + (1 - coin) * np.log(1 - p)


def compute_log_likelihood(coins, p):
    """Log-likelihood for a sequence of Bernoulli trials."""
    return sum(compute_log_prob(coin, p) for coin in coins)


# Simulate coin flips
rng = np.random.default_rng(1)
p_true = 0.7
n_samples = 100
coins = rng.binomial(n=1, p=p_true, size=n_samples)

k = coins.sum()
print(f"Observed: {k} heads out of {n_samples} flips")
print(f"MLE: p_hat = {k / n_samples:.4f}")

# Evaluate log-likelihood over a grid
ps = np.linspace(0.01, 0.99, 200)
log_liks = np.array([compute_log_likelihood(coins, p) for p in ps])

# Find MLE numerically
idx = np.argmax(log_liks)
mle_p = ps[idx]
print(f"Grid-search MLE: p_hat = {mle_p:.4f}")
print(f"Max log-likelihood: {log_liks[idx]:.4f}")
```

출력:

```
Observed: 67 heads out of 100 flips
MLE: p_hat = 0.6700
Grid-search MLE: p_hat = 0.6699
Max log-likelihood: -63.4179
```

!!! note "로그가능도의 모양"
    Bernoulli 로그가능도는 $(0, 1)$에서 $p$에 대해 오목한 함수이므로 유일한 전역 최댓값이 보장된다. 이 오목성은 모든 $p \in (0, 1)$에서 $\ell''(p) < 0$이라는 사실에서 따라 나온다.

## 벡터화된 계산

로그가능도는 반복문 없이도 효율적으로 계산할 수 있다:

```python
import numpy as np

def log_likelihood_vectorized(coins, p):
    """Vectorized log-likelihood computation."""
    k = coins.sum()
    n = len(coins)
    return k * np.log(p) + (n - k) * np.log(1 - p)

# Compare
rng = np.random.default_rng(1)
coins = rng.binomial(1, 0.7, 100)
ps = np.linspace(0.01, 0.99, 200)

ll_vec = np.array([log_likelihood_vectorized(coins, p) for p in ps])
idx = np.argmax(ll_vec)
print(f"Vectorized MLE: p = {ps[idx]:.4f}")
```

출력:

```
Vectorized MLE: p = 0.6699
```

## 가능도와 로그가능도의 비교

로그변환이 왜 필수적인지 보이기 위해 원래 가능도 값을 살펴보자:

```python
import numpy as np

rng = np.random.default_rng(1)
coins = rng.binomial(1, 0.7, 100)
k = coins.sum()
n = len(coins)

p = 0.7
# 가능도를 곱으로 그대로 계산하면 100개의 작은 수를 곱하게 되어
# 값이 1e-28 까지 내려간다. n이 1000쯤 되면 아예 0으로 언더플로된다.
raw_likelihood = p**k * (1-p)**(n-k)

# 로그를 취하면 곱이 합이 되어 이 문제가 사라진다.
# log는 단조증가 함수이므로 **최대가 되는 지점은 바뀌지 않는다.**
# 로그가능도를 쓰는 이유가 이 두 가지다: 수치 안정성과 미분의 편리함.
log_likelihood = k * np.log(p) + (n-k) * np.log(1-p)

print(f"Raw likelihood at p=0.7: {raw_likelihood:.2e}")
print(f"Log-likelihood at p=0.7: {log_likelihood:.4f}")
```

출력:

```
Raw likelihood at p=0.7: 2.33e-28
Log-likelihood at p=0.7: -63.6283
```

원래 가능도는 천문학적으로 작은 수인 반면 로그가능도는 다루기 좋은 음수이다.

## 해석

- **로그가능도함수**는 확률의 곱을 합으로 바꾸어 수치적 안정성과 해석적 편의를 제공한다.
- **MLE**는 로그가능도 곡선의 봉우리에 있는 모수값이다.
- Bernoulli 자료에서 MLE $\hat{p} = k/n$(표본비율)은 해석적으로 구할 수 있지만, 로그가능도를 시각화하면 추론 지형의 전체 모양이 드러난다.
- MLE에서 로그가능도의 **곡률**은 Fisher 정보량과 관련되며 추정의 정밀도를 결정한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> $k = 14$번 성공한 $n = 20$번의 Bernoulli 시행에 대해 $p = 0.5, 0.6, 0.7, 0.8$에서 로그가능도를 계산하라. 어느 값의 로그가능도가 가장 높은가? MLE와 어떻게 비교되는가?

</div>

??? success "풀이"
    $\ell(p) = 14\log p + 6\log(1-p)$를 사용하여 계산하면:

    - $\ell(0.5) = 20 \ln 0.5 = -13.863$
    - $\ell(0.6) = 14 \ln 0.6 + 6 \ln 0.4 = -7.148 - 5.498 = -12.646$
    - $\ell(0.7) = 14 \ln 0.7 + 6 \ln 0.3 = -4.993 - 7.225 = -12.218$
    - $\ell(0.8) = 14 \ln 0.8 + 6 \ln 0.2 = -3.124 - 9.657 = -12.781$

    로그가능도가 가장 높은 것은 $p = 0.7$이며, 이것이 MLE $\hat{p} = 14/20 = 0.7$이다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> Bernoulli 모형의 로그가능도가 $p$에 대해 오목함을 보여라. 오목성이 임의의 임계점이 전역 최댓값임을 보장하는 이유는 무엇인가?

</div>

??? success "풀이"
    로그가능도의 2계도함수는:

    $$
    \ell''(p) = -\frac{k}{p^2} - \frac{n-k}{(1-p)^2}
    $$

    $k \geq 0$, $n - k \geq 0$, $p^2 > 0$, $(1-p)^2 > 0$이므로 두 항 모두 양수가 아니다. $0 < k < n$이면(성공과 실패가 적어도 하나씩 있으면) 두 항이 모두 엄격하게 음수이므로 모든 $p \in (0, 1)$에서 $\ell''(p) < 0$이다.

    2계도함수가 엄격하게 음수인 함수는 순오목이다. 구간에서 순오목인 함수에서는 임의의 임계점($\ell'(p) = 0$인 점)이 반드시 전역 최댓값이다. 오목성은 함수가 어디서나 아래로 휜다는 뜻이기 때문이다. 다른 국소 최댓값이나 안장점은 존재할 수 없다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> 수치 계산에서 $L(\theta)$ 대신 $\log L(\theta)$를 쓰는 것이 왜 필수적인지 설명하라. 컴퓨터에서 $L(\theta)$가 0으로 언더플로되는 구체적인 예를 들라.

</div>

??? success "풀이"
    IEEE 754 배정밀도 부동소수점의 최소 양수는 약 $5 \times 10^{-324}$이다. i.i.d. Bernoulli$(0.5)$ 관측값 $n = 1000$개를 생각하자. $p = 0.5$에서 가능도는:

    $$
    L(0.5) = 0.5^{1000} = 2^{-1000} \approx 9.3 \times 10^{-302}
    $$

    이는 표현 가능하지만, $n = 1100$이면 $2^{-1100} \approx 10^{-331}$로 최소값보다 작아 부동소수점에서 정확히 0.0으로 언더플로된다.

    로그가능도는 이를 피한다: $\ell(0.5) = -1100 \ln 2 \approx -762.5$로 완벽하게 표현 가능한 수이다. $n = 10^6$에서도 로그가능도는 수치적으로 안정하다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> $n$개의 관측값을 갖는 Poisson 분포에 대해 로그가능도 $\ell(\lambda)$를 쓰고 MLE를 유도하라. $\ell''(\hat{\lambda}) < 0$임을 확인하라.

</div>

??? success "풀이"
    Poisson PMF는 $f(x; \lambda) = e^{-\lambda}\lambda^x/x!$이므로:

    $$
    \ell(\lambda) = \sum_{i=1}^n [-\lambda + x_i \log\lambda - \log(x_i!)] = -n\lambda + \left(\sum x_i\right)\log\lambda - \sum\log(x_i!)
    $$

    점수: $\ell'(\lambda) = -n + \frac{\sum x_i}{\lambda} = 0$이므로 $\hat{\lambda} = \bar{X}$이다.

    2계도함수: $\ell''(\lambda) = -\frac{\sum x_i}{\lambda^2}$.

    $\hat{\lambda} = \bar{X}$에서 ($\bar{X} > 0$을 가정하면) $\ell''(\bar{X}) = -\frac{n\bar{X}}{\bar{X}^2} = -\frac{n}{\bar{X}} < 0$이다.

    로그가능도가 오목하고 MLE가 최댓값임이 확인된다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> 관측 Fisher 정보량은 $\hat{I}(\theta) = -\ell''(\hat{\theta})$이다. Bernoulli 모형에서 MLE에서의 관측 정보량이 $n/[\hat{p}(1-\hat{p})]$과 같음을 보여라. 이를 사용하여 $n = 100$, $k = 72$일 때 $p$에 대한 근사적인 95% 신뢰구간을 구성하라.

</div>

??? success "풀이"
    연습문제 2에서 $\ell''(p) = -k/p^2 - (n-k)/(1-p)^2$이다.

    $\hat{p} = k/n$에서:

    $$
    -\ell''(\hat{p}) = \frac{k}{\hat{p}^2} + \frac{n-k}{(1-\hat{p})^2} = \frac{n\hat{p}}{\hat{p}^2} + \frac{n(1-\hat{p})}{(1-\hat{p})^2} = \frac{n}{\hat{p}} + \frac{n}{1-\hat{p}} = \frac{n}{\hat{p}(1-\hat{p})}
    $$

    $n = 100, k = 72$이면 $\hat{p} = 0.72$이고 $\hat{I} = 100/(0.72 \times 0.28) = 495.87$이다.

    MLE의 근사 분산은 $1/\hat{I} = 0.72 \times 0.28/100 = 0.002016$이다.

    표준오차는 $\sqrt{0.002016} = 0.04490$이다.

    95% 신뢰구간은:

    $$
    0.72 \pm 1.96 \times 0.04490 = [0.632, 0.808]
    $$

    $\square$

---

## 정리하며

로그가능도는 최대가능도추정의 **실무 도구**다.

- **왜 로그인가.** 작은 확률을 수백 개 곱하면 언더플로가 나고, 곱은 미분하기 번거롭다. 로그를 취하면 합이 되어 둘 다 해결된다. 로그가 단조이므로 **최댓값의 위치는 바뀌지 않는다.**
- **그림에서 두 가지가 읽힌다.** 봉우리의 **위치**가 $\hat\theta$ 이고, 봉우리의 **뾰족함**이 정밀도다. 평평하면 여러 $\theta$ 가 비슷하게 그럴듯하다는 뜻이며, 그 곡률이 곧 관측 피셔 정보량이다.
- **베르누이 예에서 확인된다.** 앞면 비율이 $\hat p$ 에서 봉우리를 이루고, 시행 수가 늘수록 봉우리가 좁아진다. 자료가 많아질수록 $\theta$ 가 더 좁게 특정된다는 것을 눈으로 보는 셈이다.
- **가능도 자체를 그리면 봉우리만 보이고 나머지는 $0$ 에 붙어 버린다.** 로그 척도라야 모양 전체가 보인다.

다음 절 **기하분포와 포아송분포의 최대가능도**로 넘어간다. 두 이산분포를 같은 자료에 적합해 보고, 모형이 맞을 때와 틀릴 때의 차이를 본다.
