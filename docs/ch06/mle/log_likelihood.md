# 로그가능도 시각화

## 개요

**로그가능도함수**는 가능도에 로그를 취한 것으로 최대가능도추정의 주된 도구이다. 가능도 자체 대신 로그가능도로 작업하면 작은 확률을 많이 곱할 때 생기는 수치적 언더플로를 피할 수 있고 곱이 합으로 바뀌어 계산과 미분이 모두 간단해진다. 이 페이지에서는 베르누이 동전 던지기 예제로 로그가능도의 구성, 시각화, MLE 추출을 보인다.

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
    $p = 0.7$인 $n = 100$개의 베르누이 관측값에서 가능도는 0과 1 사이 수 100개의 곱이다. 이 곱은 $10^{-30}$ 규모로 부동소수점 언더플로 문턱보다 훨씬 작다. 로그가능도는 로그확률의 합으로 작업하여 이를 피한다.

## 베르누이 로그가능도

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

<div class="codebox" markdown>

### 예제 1. 로그가능도 구현과 시각화 { .eg }

```python
import numpy as np

def compute_log_prob(coin, p):
    """베르누이 시행 한 번의 로그확률."""
    return coin * np.log(p) + (1 - coin) * np.log(1 - p)


def compute_log_likelihood(coins, p):
    """베르누이 시행 여러 번의 로그가능도."""
    return sum(compute_log_prob(coin, p) for coin in coins)


# 동전을 n번 던진다. 참 p 는 우리가 모르는 값이라고 둔다.
rng = np.random.default_rng(1)
p_true = 0.7
n_samples = 100
coins = rng.binomial(n=1, p=p_true, size=n_samples)

k = coins.sum()
print(f"Observed: {k} heads out of {n_samples} flips")
print(f"MLE: p_hat = {k / n_samples:.4f}")

# 격자 위에서 로그가능도를 계산한다.
ps = np.linspace(0.01, 0.99, 200)
log_liks = np.array([compute_log_likelihood(coins, p) for p in ps])

# 수치 최적화로 MLE를 찾는다.
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

</div>

!!! note "로그가능도의 모양"
    베르누이 로그가능도는 $(0, 1)$에서 $p$에 대해 오목한 함수이므로 유일한 전역 최댓값이 보장된다. 이 오목성은 모든 $p \in (0, 1)$에서 $\ell''(p) < 0$이라는 사실에서 따라 나온다.

## 벡터화된 계산

로그가능도는 반복문 없이도 효율적으로 계산할 수 있다:

<div class="codebox" markdown>

### 예제 2. 로그가능도의 벡터화 { .eg }

```python
import numpy as np

def log_likelihood_vectorized(coins, p):
    """로그가능도를 벡터화해 한 번에 계산한다."""
    k = coins.sum()
    n = len(coins)
    return k * np.log(p) + (n - k) * np.log(1 - p)

# 두 값을 견준다.
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

</div>

## 가능도와 로그가능도의 비교

로그변환이 왜 필수적인지 보이기 위해 원래 가능도 값을 살펴보자:

<div class="codebox" markdown>

### 예제 3. 가능도와 로그가능도의 수치 비교 { .eg }

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

</div>

## 해석

- **로그가능도함수**는 확률의 곱을 합으로 바꾸어 수치적 안정성과 해석적 편의를 제공한다.
- **MLE**는 로그가능도 곡선의 봉우리에 있는 모수값이다.
- 베르누이 자료에서 MLE $\hat{p} = k/n$(표본비율)은 해석적으로 구할 수 있지만, 로그가능도를 시각화하면 추론 지형의 전체 모양이 드러난다.
- MLE에서 로그가능도의 **곡률**은 Fisher 정보량과 관련되며 추정의 정밀도를 결정한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> $k = 14$번 성공한 $n = 20$번의 베르누이 시행에 대해 $p = 0.5, 0.6, 0.7, 0.8$에서 로그가능도를 계산하라. 어느 값의 로그가능도가 가장 높은가? MLE와 어떻게 비교되는가?

</div>

??? success "풀이"
    $\ell(p) = 14\log p + 6\log(1-p)$를 사용하여 계산하면:

    - $\ell(0.5) = 20 \ln 0.5 = -13.863$
    - $\ell(0.6) = 14 \ln 0.6 + 6 \ln 0.4 = -7.148 - 5.498 = -12.646$
    - $\ell(0.7) = 14 \ln 0.7 + 6 \ln 0.3 = -4.993 - 7.225 = -12.218$
    - $\ell(0.8) = 14 \ln 0.8 + 6 \ln 0.2 = -3.124 - 9.657 = -12.781$

    로그가능도가 가장 높은 것은 $p = 0.7$이며, 이것이 MLE $\hat{p} = 14/20 = 0.7$이다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 베르누이 모형의 로그가능도가 $p$에 대해 오목함을 보여라. 오목성이 임의의 임계점이 전역 최댓값임을 보장하는 이유는 무엇인가?

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

**연습문제 4.** <span class="diff med" title="중간"></span> $n$개의 관측값을 갖는 포아송분포에 대해 로그가능도 $\ell(\lambda)$를 쓰고 MLE를 유도하라. $\ell''(\hat{\lambda}) < 0$임을 확인하라.

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

**연습문제 5.** <span class="diff med" title="중간"></span> 관측 Fisher 정보량은 $\hat{I}(\theta) = -\ell''(\hat{\theta})$이다. 베르누이 모형에서 MLE에서의 관측 정보량이 $n/[\hat{p}(1-\hat{p})]$과 같음을 보여라. 이를 사용하여 $n = 100$, $k = 72$일 때 $p$에 대한 근사적인 95% 신뢰구간을 구성하라.

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

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
로그가능도를 $\hat\theta$ 근처에서 2차까지 테일러 전개하면 무엇이 나오는가? 이 근사에서 왈드 신뢰구간이 어떻게 따라 나오는지 보이고, 근사가 나쁠 때의 신호를 적어라.

</div>

??? success "풀이"
    $\hat\theta$가 내부 최댓값이면 $\ell'(\hat\theta) = 0$이므로 1차항이 사라지고

    $$
    \ell(\theta) \approx \ell(\hat\theta) + \frac12\ell''(\hat\theta)(\theta-\hat\theta)^2 = \ell(\hat\theta) - \frac{\hat I}{2}(\theta-\hat\theta)^2
    $$

    이다($\hat I = -\ell''(\hat\theta)$는 관측정보량). 즉 **로그가능도가 봉우리 근처에서 아래로 볼록한 포물선**이고, 가능도 자체는

    $$
    L(\theta) \approx L(\hat\theta)\exp\left\{-\frac{\hat I}{2}(\theta-\hat\theta)^2\right\}
    $$

    로 **평균 $\hat\theta$, 분산 $1/\hat I$인 정규밀도 모양**이 된다.

    **왈드 구간.** 이 근사를 우도비 구간에 넣으면

    $$
    2\{\ell(\hat\theta)-\ell(\theta)\} \approx \hat I(\theta-\hat\theta)^2 \le \chi^2_{1,0.95} = 1.96^2
    $$

    이므로

    $$
    |\theta-\hat\theta| \le \frac{1.96}{\sqrt{\hat I}} \implies \hat\theta \pm 1.96\,\widehat{\operatorname{SE}}, \quad \widehat{\operatorname{SE}} = \frac{1}{\sqrt{\hat I}}
    $$

    가 나온다. **왈드 구간은 곧 "로그가능도를 포물선으로 본" 구간**이다.

    **근사가 나쁠 때의 신호.**

    - 로그가능도 곡선이 눈에 띄게 **비대칭**이다(한쪽이 급하고 다른 쪽이 완만하다).
    - $\hat\theta$가 모수공간의 **경계**에 있거나 가깝다($\hat p = 0$, $\hat\sigma^2 = 0$ 등).
    - 표본이 작다. 이차 근사의 오차는 $O(n^{-1/2})$이다.
    - 모수화가 부자연스럽다. 같은 모형이라도 $p$ 대신 로짓, $\sigma$ 대신 $\ln\sigma$로 두면 곡선이 훨씬 포물선에 가까워진다.

    **진단 방법이 간단하다.** 로그가능도 곡선을 그려 봉우리에 포물선을 겹쳐 그리면 된다. 두 곡선이 $\pm2$ 표준오차 범위에서 눈에 띄게 벌어지면 왈드 대신 우도비 구간을 써야 한다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
코시분포의 위치모수 $\theta$에 대한 로그가능도

$$
\ell(\theta) = -\sum_{i=1}^n \ln\left\{1+(x_i-\theta)^2\right\} + \text{상수}
$$

는 봉우리가 여럿일 수 있다. 왜 그런지 설명하고, 수치 최적화에서 무엇을 조심해야 하는지 적어라.

</div>

??? success "풀이"
    **왜 봉우리가 여럿인가.** 점수함수가

    $$
    \ell'(\theta) = \sum_i \frac{2(x_i-\theta)}{1+(x_i-\theta)^2}
    $$

    인데, 각 항 $\psi(r) = 2r/(1+r^2)$이 **유계이고 $|r|\to\infty$에서 0으로 되돌아간다**(재하강). 따라서 $\theta$가 어느 관측값 근처에 있으면 그 관측값만 강하게 끌어당기고 멀리 있는 관측값들은 거의 힘을 쓰지 못한다.

    관측값들이 서로 멀리 떨어져 있으면 **각 무리마다 국소 봉우리**가 생긴다. 실제로 $\ell'(\theta)=0$은 최대 $2n-1$개의 해를 가질 수 있다.

    정규분포와 대비하면 분명하다. 정규에서는 $\psi(r)=r$이 유계가 아니라 모든 관측값이 끝까지 끌어당기고, 점수방정식이 선형이라 해가 유일하다($\bar x$).

    **수치 최적화에서 조심할 것.**

    - **초기값을 잘 주어야 한다.** 표본중앙값이 좋은 출발점이다. 코시분포에서 중앙값은 일치추정량이고 계산도 간단하다. 표본평균은 절대 쓰면 안 된다(수렴하지 않는다).
    - **여러 초기값에서 돌린다.** 관측값들 자체나 분위수들을 초기값으로 삼아 여러 번 최적화하고 가장 큰 $\ell$을 고른다.
    - **격자 탐색으로 전역 모양을 먼저 본다.** 모수가 하나이므로 $\ell(\theta)$를 촘촘한 격자에서 그려 보는 것이 가장 확실하다.
    - **뉴턴법이 발산할 수 있다.** $\ell''$이 양수인 영역이 있으므로 갱신이 오르막이 아닐 수 있다. 신뢰영역법이나 감쇠 뉴턴법이 안전하다.

    **일반 교훈.** 재하강하는 $\psi$를 갖는 강건 추정량은 **강건성의 대가로 다봉성을 얻는다.** 후버 추정량처럼 $\psi$가 단조이면 로그가능도가 오목해 봉우리가 하나지만, 터키의 이중가중치나 $t$ 잡음처럼 재하강하는 것은 국소 최대가 생긴다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
**관측정보량** $\hat I = -\ell''(\hat\theta)$과 **기대정보량** $I(\theta) = E[-\ell''(\theta)]$의 차이를 설명하라. 표준오차 계산에 어느 쪽을 쓰는 것이 좋은가?

</div>

??? success "풀이"
    **차이.**

    - **관측정보량**은 손에 든 자료에서 계산한 실제 곡률이다. 확률변수이며 자료마다 다르다.
    - **기대정보량**은 그 기대값이다. $\theta$만의 함수이고 자료에 의존하지 않는다.

    큰수의 법칙에 따라 $\hat I/n \to I_1(\theta)$이므로 $n$이 크면 둘이 가까워진다.

    분포에 따라 아예 같은 경우도 있다. 지수분포에서 $\ell'' = -n/\lambda^2$은 자료에 의존하지 않으므로 관측정보량과 기대정보량이 정확히 같다. 베르누이에서는 $\hat I = n/\{\hat p(1-\hat p)\}$이고 $I(\hat p)$도 같은 값이라 역시 일치한다. **정준연결 지수족에서는 언제나 일치한다.**

    **어느 쪽을 쓸까.** 일반적으로 **관측정보량**이 권장되며, 이유는 세 가지다.

    1. **계산이 쉽다.** 기대값을 구할 필요 없이 최적화 과정에서 이미 얻은 헤시안을 그대로 쓴다.
    2. **조건성 원리에 맞는다.** 실제로 얻은 자료가 얼마나 정보를 담고 있는지를 반영한다. 에프런과 힝클리가 보인 대로, 부수통계량으로 조건화한 추론에 더 가깝다.
    3. **모형이 조금 틀렸을 때 더 낫다.** 기대정보량은 모형이 정확히 맞다는 가정에 더 의존한다.

    **예외.** 기대정보량이 유용한 경우도 있다. 피셔 점수법에서는 기대정보량이 언제나 양반정부호라 알고리즘이 안정적이다. 또 실험 설계 단계에서는 자료가 없으므로 기대정보량으로 표본크기를 계획한다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
혼합모형이나 잠재변수 모형에서는 $\ln\sum_k \exp(a_k)$ 꼴을 계산해야 한다. 이를 그대로 계산하면 왜 위험한지 설명하고 **logsumexp** 요령을 적어라.

</div>

??? success "풀이"
    **위험.** $a_k$가 로그가능도 값이면 $-1000$ 같은 큰 음수인 경우가 흔하다. 그대로 $\exp$를 취하면

    $$
    e^{-1000} = 0 \quad(\text{언더플로})
    $$

    이 되어 합이 0이 되고 $\ln 0 = -\infty$가 나온다. 반대로 $a_k$가 큰 양수이면 $\exp$가 오버플로해 `inf`가 된다.

    **logsumexp 요령.** $M = \max_k a_k$를 빼고 더한 뒤 되돌린다.

    $$
    \ln\sum_k e^{a_k} = M + \ln\sum_k e^{a_k-M}
    $$

    수학적으로 정확한 항등식이다($e^M$을 묶어 낸 것). 그런데 수치적으로는 완전히 다르다. 지수의 인수 $a_k - M$이 모두 0 이하이므로 $e^{a_k-M} \in (0,1]$이고, 적어도 하나(최댓값에 해당하는 항)는 정확히 1이다. **오버플로가 불가능하고, 합이 1 이상이라 언더플로도 무해하다.**

    ```python
    from scipy.special import logsumexp
    print(logsumexp([-1000, -1001, -1002]))   # -999.59...
    print(np.log(np.sum(np.exp([-1000, -1001, -1002]))))   # -inf
    ```

    **어디에 쓰이는가.**

    - **혼합모형의 로그가능도**: $\ell = \sum_i \ln\sum_k \pi_k f_k(x_i)$의 안쪽 합.
    - **소프트맥스와 교차엔트로피**: 분류 모형의 표준 구현이 모두 logsumexp를 쓴다.
    - **은닉 마르코프 모형의 전진-후진 알고리즘**, 신뢰전파, 베이즈망의 메시지 전달.
    - **중요도추출의 가중치 정규화**.

    같은 발상의 형제로 $\ln(1+x)$를 정확히 계산하는 `log1p`, $e^x-1$의 `expm1`, 로그 척도에서 두 값을 더하는 `logaddexp`가 있다. **가능도를 다루는 코드는 되도록 로그 척도를 떠나지 않는 것이 원칙이다.**

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
같은 모형에 $n=10$과 $n=100$인 자료를 적합해 로그가능도 곡선을 겹쳐 그렸다. 두 곡선에서 읽을 수 있는 것을 정리하고, 곡선을 어떻게 정규화해야 비교가 공정해지는지 적어라.

</div>

??? success "풀이"
    **곡선에서 읽는 것.**

    - **봉우리의 위치**가 $\hat\theta$다. 두 표본의 $\hat\theta$가 얼마나 다른지가 표집변동의 크기를 보여 준다.
    - **봉우리의 뾰족함**이 정보량이다. $n=100$ 곡선이 훨씬 가파르며, 곡률이 $n$에 비례하므로 폭이 $1/\sqrt n$로 줄어든다. 즉 $n$이 10배면 폭이 약 3.2분의 1이다.
    - **비대칭**이 남아 있는지. $n$이 커지면 곡선이 포물선에 가까워진다. $n=10$에서 눈에 띄던 비대칭이 $n=100$에서 거의 사라진다면, 왈드 근사를 작은 표본에 쓰면 안 된다는 신호다.

    **공정한 비교를 위한 정규화.**

    1. **세로축을 최댓값 기준으로 옮긴다.** $\ell(\theta)-\ell(\hat\theta)$를 그린다. 로그가능도의 절대 높이는 $n$에 비례해 커지므로(관측값마다 항이 하나씩 더해진다) 그대로 겹쳐 그리면 $n=100$ 곡선이 한참 아래에 놓여 모양을 볼 수 없다. 상대 로그가능도로 옮기면 두 곡선이 모두 0에서 시작한다.

    2. **관심이 "모양"이면 그대로, "정밀도"면 그대로 둔다.** 상대 로그가능도만 맞추면 $n=100$ 곡선이 훨씬 좁게 나오며, 이것이 곧 정밀도의 차이다. 반대로 두 곡선의 **모양**(비대칭 정도)만 비교하고 싶다면 가로축을 $\sqrt{\hat I}(\theta-\hat\theta)$로 표준화한다. 그러면 이차 근사가 완벽할 때 두 곡선이 같은 포물선 $-z^2/2$로 겹친다.

    3. **가로 범위를 표준오차 단위로 잡는다.** $\hat\theta \pm 4\widehat{\operatorname{SE}}$ 정도가 적당하다. 절대 범위로 잡으면 $n$이 클 때 곡선이 한 점으로 뭉개진다.

    **읽는 요령 하나.** 상대 로그가능도가 $-1.92$인 지점이 95% 우도비 구간의 양끝이다($\chi^2_{1,0.95}/2 = 1.92$). 이 수평선을 그려 두면 구간을 눈으로 바로 읽을 수 있고, 구간이 대칭인지도 함께 보인다.

---

## 정리하며

로그가능도는 최대가능도추정의 **실무 도구**다.

- **왜 로그인가.** 작은 확률을 수백 개 곱하면 언더플로가 나고, 곱은 미분하기 번거롭다. 로그를 취하면 합이 되어 둘 다 해결된다. 로그가 단조이므로 **최댓값의 위치는 바뀌지 않는다.**
- **그림에서 두 가지가 읽힌다.** 봉우리의 **위치**가 $\hat\theta$ 이고, 봉우리의 **뾰족함**이 정밀도다. 평평하면 여러 $\theta$ 가 비슷하게 그럴듯하다는 뜻이며, 그 곡률이 곧 관측 피셔 정보량이다.
- **베르누이 예에서 확인된다.** 앞면 비율이 $\hat p$ 에서 봉우리를 이루고, 시행 수가 늘수록 봉우리가 좁아진다. 자료가 많아질수록 $\theta$ 가 더 좁게 특정된다는 것을 눈으로 보는 셈이다.
- **가능도 자체를 그리면 봉우리만 보이고 나머지는 $0$ 에 붙어 버린다.** 로그 척도라야 모양 전체가 보인다.

다음 절 **기하분포와 포아송분포의 최대가능도**로 넘어간다. 두 이산분포를 같은 자료에 적합해 보고, 모형이 맞을 때와 틀릴 때의 차이를 본다.
