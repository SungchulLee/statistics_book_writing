# 베르누이분포의 MLE

## 개요

$x^{(i)}$를 $B(p)$에서 얻은 $m$개의 i.i.d. 표본이라 하자. 그러면 $p$는 다음 $\hat{p}$로 추정할 수 있다:

$$
\hat{p} = \frac{\sum_{i=1}^m x^{(i)}}{m}
$$

## 유도

### 자료

$$
\{x^{(i)} : i = 1, \ldots, m\}
$$

### 모형

$$
x^{(i)} \sim B(p)
$$

### 가능도함수

$$
L(p) = \prod_{i=1}^m p^{x^{(i)}} (1 - p)^{1 - x^{(i)}}
$$

### 로그가능도함수

$$
\ell(p) = \sum_{i=1}^m x^{(i)} \log(p) + (1 - x^{(i)}) \log(1 - p)
$$

### 비용함수

$$
J(p) = -\sum_{i=1}^m x^{(i)} \log(p) + (1 - x^{(i)}) \log(1 - p)
$$

!!! note "교차엔트로피와의 연결"
    비용함수 $J(p)$는 로지스틱 회귀와 신경망에서 쓰는 **이진 교차엔트로피 손실**과 정확히 같다. 베르누이분포의 MLE가 교차엔트로피 기반 학습의 이론적 토대이다.

### 최대가능도 원리

$$
\text{argmax}_{p}\; L
\quad\Leftrightarrow\quad
\text{argmax}_{p}\; \ell
\quad\Leftrightarrow\quad
\text{argmin}_{p}\; J
$$

### MLE 해

$$
\begin{array}{llcll}
\displaystyle\frac{\partial J}{\partial p} = 0
&\Rightarrow&
\displaystyle\sum_{i=1}^m \frac{x^{(i)}}{p} - \frac{1 - x^{(i)}}{1 - p} = 0
&\Rightarrow&
\displaystyle\hat{p} = \frac{\sum_{i=1}^m x^{(i)}}{m}
\end{array}
$$

## 로그가능도 곡선 그려 보기

<div class="codebox" markdown>

### 예제 1. 베르누이 로그가능도와 MLE { .eg }

```python
import numpy as np
import matplotlib.pyplot as plt

# 같은 결과가 다시 나오도록 난수 씨앗을 고정한다.
seed = 1
np.random.seed(seed)

# 앞면 확률과 던지는 횟수를 정한다.
p = 0.7
n_samples = 100

def load_data():
    """
    Simulate coin flips based on a binomial distribution.

    Returns:
    - numpy array: Array of coin flips (1 for heads, 0 for tails).
    """
    return np.random.binomial(n=1, p=p, size=(n_samples,))  # Shape (100,)

def compute_prob(coin, p):
    """
    Compute the probability of a single coin flip outcome.

    Parameters:
    - coin: Outcome of the coin flip (1 for heads, 0 for tails).
    - p: Probability of heads.

    Returns:
    - float: Probability of observing the outcome.
    """
    return p**coin * (1 - p)**(1 - coin)

def compute_log_prob(coin, p):
    """
    Compute the log-probability of a single coin flip outcome.

    Parameters:
    - coin: Outcome of the coin flip (1 for heads, 0 for tails).
    - p: Probability of heads.

    Returns:
    - float: Log-probability of observing the outcome.
    """
    return coin * np.log(p) + (1 - coin) * np.log(1 - p)

def compute_likelihood(coins, p):
    """
    Compute the joint probability of all coin flips for a given probability.

    Parameters:
    - coins: Array of coin flip outcomes.
    - p: Probability of heads.

    Returns:
    - float: Joint probability of observing all outcomes.
    """
    joint_prob = 1.0
    for coin in coins:
        joint_prob *= compute_prob(coin, p)
    return joint_prob

def compute_log_likelihood(coins, p):
    """
    Compute the log-likelihood of all coin flips for a given probability.

    Parameters:
    - coins: Array of coin flip outcomes.
    - p: Probability of heads.

    Returns:
    - float: Log-likelihood of observing all outcomes.
    """
    log_joint_prob = 0.0
    for coin in coins:
        log_joint_prob += compute_log_prob(coin, p)
    return log_joint_prob

# 동전을 n번 던진다. 참 p 는 우리가 모르는 값이라고 둔다.
coins = load_data()

# p 후보를 격자로 늘어놓는다. 이 중 로그가능도가 가장 큰 것을 고른다.
ps = np.linspace(0.01, 0.99, 100)

# 후보마다 로그가능도를 계산한다.
log_likelihood_list = [compute_log_likelihood(coins, p) for p in ps]
log_likelihood = np.array(log_likelihood_list)

# 로그가능도가 가장 큰 후보가 최대가능도추정값이다.
idx = np.argmax(log_likelihood)
mle_p = ps[idx]
log_likelihood_max = log_likelihood[idx]
print(f"MLE index: {idx}")
print(f"MLE probability (p): {mle_p:.4f}")
print(f"Max log-likelihood: {log_likelihood_max:.4f}\n")

# 로그가능도 곡선을 그리고 최댓값 자리를 표시한다.
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(ps, log_likelihood, label="Log-likelihood")
ax.plot([mle_p, mle_p], [0, log_likelihood_max], '--or', label="MLE")
ax.legend(loc="lower right")

# 축 이름과 범례를 다듬는다.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_position("zero")
ax.spines['left'].set_position("zero")
ax.set_xlabel("Probability (p)")
ax.set_ylabel("Log-likelihood")
plt.show()
```

출력:

```
MLE index: 74
MLE probability (p): 0.7425
Max log-likelihood: -57.3074
```

![베르누이분포의 MLE](./img/mle_bernoulli_70.png)

</div>

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
동전을 20번 던져 앞면 13번, 뒷면 7번이 나왔다. 로그가능도함수를 쓰고 MLE $\hat{p}$를 해석적으로 구하라.

</div>

??? success "풀이"
    $n = 20$번 중 $k = 13$번이 앞면이라 하자. 로그가능도는:

    $$
    \ell(p) = k \log p + (n-k) \log(1-p) = 13\log p + 7\log(1-p)
    $$

    도함수를 0으로 두면:

    $$
    \frac{d\ell}{dp} = \frac{13}{p} - \frac{7}{1-p} = 0
    $$

    $$
    13(1-p) = 7p \implies 13 - 13p = 7p \implies 13 = 20p \implies \hat{p} = \frac{13}{20} = 0.65
    $$

    2계도함수가 $-13/p^2 - 7/(1-p)^2 < 0$이므로 최댓값임이 확인된다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
베르누이의 MLE에서 표본크기나 관측된 자료와 무관하게 언제나 $\hat{p} = \bar{x}$(표본비율)이 MLE임을 보여라.

</div>

??? success "풀이"
    $k = \sum x_i$번 성공한 $n$번의 독립 베르누이 시행에 대해 로그가능도는:

    $$
    \ell(p) = k \log p + (n-k)\log(1-p)
    $$

    미분하여 0으로 두면:

    $$
    \frac{d\ell}{dp} = \frac{k}{p} - \frac{n-k}{1-p} = 0 \implies k(1-p) = (n-k)p \implies k = np
    $$

    $$
    \hat{p} = \frac{k}{n} = \frac{\sum x_i}{n} = \bar{x}
    $$

    이는 $0 \leq k \leq n$인 임의의 $k$와 $n$에 대해 성립한다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
베르누이 관측값 하나에 대한 Fisher 정보량을 계산하고 $\hat{p}$의 점근분산을 유도하라.

</div>

??? success "풀이"
    Bernoulli$(p)$ 관측값 하나에 대해 로그가능도는 $\ell(p) = x\log p + (1-x)\log(1-p)$이다. 2계도함수는:

    $$
    \frac{d^2\ell}{dp^2} = -\frac{x}{p^2} - \frac{1-x}{(1-p)^2}
    $$

    ($E[X] = p$를 사용하여) 기댓값에 음수를 취하면:

    $$
    I(p) = -E\!\left[\frac{d^2\ell}{dp^2}\right] = \frac{p}{p^2} + \frac{1-p}{(1-p)^2} = \frac{1}{p} + \frac{1}{1-p} = \frac{1}{p(1-p)}
    $$

    $n$개의 관측값에 기반한 $\hat{p}$의 점근분산은:

    $$
    \text{Var}(\hat{p}) \approx \frac{1}{nI(p)} = \frac{p(1-p)}{n}
    $$

    표본비율의 분산에 대한 익숙한 공식이다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
10번 던져 앞면이 0번 나오면 MLE는 $\hat{p} = 0$을 준다. 이것이 왜 문제인지 설명하고 대안적인 접근을 하나 서술하라.

</div>

??? success "풀이"
    MLE $\hat{p} = 0$은 이 동전에서 앞면이 결코 나올 수 없다는 뜻인데, 관측값 10개만으로 내리기에는 극단적인 결론이다. 문제는 특히 소표본에서 MLE가 비합리적인 경계값 추정치를 낼 수 있다는 것이다.

    한 가지 대안은 **라플라스 평활**(또는 균등 사전분포를 쓴 베이즈 접근)이다. 자료에 "가상의 성공" 하나와 "가상의 실패" 하나를 더하여 $\hat{p}_{\text{Laplace}} = (0+1)/(10+2) = 1/12 \approx 0.083$을 얻는다. 자료에 근거하면서도 0이라는 추정값을 피한다. 더 형식적으로 이는 Beta$(1,1)$(균등) 사전분포 아래의 사후평균에 해당하며 $\hat{p}_{\text{Bayes}} = (k+1)/(n+2)$를 준다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
연습문제 1의 자료($n=20$, 앞면 13번)로 $H_0: p=0.5$를 **왈드 검정**, **점수 검정**, **우도비 검정** 세 가지로 각각 수행하고 결과를 비교하라. 세 검정은 어떤 관계인가?

</div>

??? success "풀이"
    $\hat p = 0.65$이다.

    **왈드 검정.** MLE에서 평가한 정보량을 쓴다.

    $$
    z_W = \frac{\hat p - p_0}{\sqrt{\hat p(1-\hat p)/n}} = \frac{0.15}{0.1067} = 1.406, \qquad p\text{-값} = 0.160
    $$

    **점수 검정.** 귀무가설 아래의 정보량을 쓴다.

    $$
    z_S = \frac{\hat p-p_0}{\sqrt{p_0(1-p_0)/n}} = \frac{0.15}{0.1118} = 1.342, \qquad p\text{-값} = 0.180
    $$

    **우도비 검정.**

    $$
    \Lambda = 2\left\{\ell(\hat p)-\ell(p_0)\right\} = 2\left\{13\ln0.65+7\ln0.35-20\ln0.5\right\} = 1.828
    $$

    $\Lambda \sim \chi^2_1$이므로 $p\text{-값} = 0.176$이다.

    **비교.** 셋 다 5%에서 기각하지 못하고 $p$-값이 0.16~0.18로 비슷하다. 참고로 정확 이항검정은 0.263으로 꽤 다르다.

    **세 검정의 관계.** 모두 $n\to\infty$에서 **점근적으로 동등**하며 $H_0$ 아래에서 $\chi^2_1$을 따른다. 차이는 로그가능도 곡선을 어디서 재느냐에 있다.

    | 검정 | 재는 방식 | 기하적 그림 |
    |---|---|---|
    | 왈드 | $\hat\theta$와 $\theta_0$의 **거리**, $\hat\theta$에서의 곡률로 표준화 | 봉우리에서의 수평 거리 |
    | 점수 | $\theta_0$에서의 **기울기** | 귀무값에서 곡선이 얼마나 가파른가 |
    | 우도비 | $\ell(\hat\theta)-\ell(\theta_0)$, **높이 차이** | 봉우리와 귀무값의 수직 거리 |

    **실무적 권고는 우도비 검정이다.** 이유는 두 가지다. 첫째, **모수화에 불변**이다. $p$로 검정하든 로그오즈로 검정하든 같은 값이 나온다. 왈드는 그렇지 않아서 $p$로 하느냐 $\ln\{p/(1-p)\}$로 하느냐에 따라 결과가 달라진다. 둘째, 로그가능도가 이차식에서 멀어질수록 왈드가 부정확해지는데(경계 근처에서 특히 심하다) 우도비는 훨씬 잘 버틴다.

    점수 검정은 $\hat\theta$를 구하지 않아도 계산할 수 있다는 장점이 있어, 모형을 적합하기 전에 변수 추가의 효과를 미리 보는 데 쓰인다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
$n=10$, 앞면 0번인 경우로 돌아가자. 왈드 구간은 쓸 수 없다. **프로파일 가능도 구간**을 구하고 윌슨 구간·클로퍼-피어슨 구간과 견주어라.

</div>

??? success "풀이"
    **프로파일 가능도 구간.** 우도비 검정을 뒤집어

    $$
    \left\{p:\ 2\left\{\ell(\hat p)-\ell(p)\right\} \le \chi^2_{1,0.95} = 3.841\right\}
    $$

    을 구간으로 삼는다. $k=0$이면 $\ell(p) = 10\ln(1-p)$이고 $\hat p = 0$에서 $\ell(\hat p) = 0$이므로

    $$
    -20\ln(1-p) \le 3.841 \implies p \le 1-e^{-0.1921} = 0.1748
    $$

    이다. 구간은 $(0,\ 0.175)$다.

    **비교.**

    | 방법 | 95% 구간 |
    |---|---|
    | 왈드 | $[0,\ 0]$ — 쓸 수 없음 |
    | 프로파일 가능도 | $(0,\ 0.175)$ |
    | 윌슨 | $(0,\ 0.278)$ |
    | 클로퍼-피어슨 | $(0,\ 0.308)$ |
    | 3의 법칙 | $(0,\ 0.30)$ |

    **읽는 법.** 프로파일 가능도 구간이 가장 좁고 클로퍼-피어슨이 가장 넓다. 클로퍼-피어슨은 포함확률이 명목값 **이상**임을 보장하느라 보수적이고, 프로파일 구간은 카이제곱 근사에 기대므로 이 극단적인 상황에서는 다소 낙관적이다.

    **핵심은 왈드만 빼고 셋 다 쓸 만하다**는 점이다. 왈드가 무너지는 이유는 로그가능도를 $\hat p$ 근처에서 **이차함수로 근사**하는데, $\hat p = 0$이 모수공간의 경계라 그 근사가 성립하지 않기 때문이다. 실제로 $k=0$일 때 로그가능도 $10\ln(1-p)$는 $p=0$에서 기울기가 $-10$으로 0이 아니다. 봉우리가 아니라 절벽 끝인 셈이다.

    **일반 원리.** 추정값이 모수공간의 경계에 있거나 로그가능도가 심하게 비대칭이면 왈드 방법을 쓰면 안 된다. 프로파일 가능도나 정확 방법으로 옮겨야 한다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
로그오즈 $\theta = \ln\{p/(1-p)\}$의 MLE를 구하고 델타 방법으로 그 표준오차를 구하라. $\theta$ 척도의 왈드 구간을 $p$ 척도로 되돌린 것과 $p$ 척도에서 바로 만든 왈드 구간을 견주어라($n=20$, $k=13$).

</div>

??? success "풀이"
    **MLE.** 불변성에 따라

    $$
    \hat\theta = \ln\frac{\hat p}{1-\hat p} = \ln\frac{0.65}{0.35} = 0.6190
    $$

    **표준오차.** $g(p) = \ln\{p/(1-p)\}$의 도함수가 $g'(p) = 1/\{p(1-p)\}$이므로

    $$
    \operatorname{SE}(\hat\theta) \approx \frac{1}{\hat p(1-\hat p)}\sqrt{\frac{\hat p(1-\hat p)}{n}} = \frac{1}{\sqrt{n\hat p(1-\hat p)}} = \frac{1}{\sqrt{20(0.2275)}} = 0.4685
    $$

    **두 구간.**

    *($p$ 척도에서 바로)*

    $$
    0.65 \pm 1.96(0.1067) = (0.441,\ 0.859)
    $$

    *($\theta$ 척도에서 만들어 되돌림)*

    $$
    0.6190 \pm 1.96(0.4685) = (-0.299,\ 1.537)
    $$

    이고 로지스틱 변환 $p = e^\theta/(1+e^\theta)$로 되돌리면

    $$
    (0.426,\ 0.823)
    $$

    **다르다.** 두 번째 구간은 비대칭이며($\hat p$에서 아래로 0.224, 위로 0.173) 언제나 $(0,1)$ 안에 머문다. 첫 번째는 대칭이고 $\hat p$가 극단적이면 $[0,1]$을 벗어난다.

    **어느 쪽이 나은가.** 일반적으로 **로그오즈 척도에서 만드는 쪽**이 낫다. 로그가능도가 $\theta$ 척도에서 이차함수에 훨씬 가깝기 때문이고, 이것이 왈드 근사의 전제다. 모수공간이 $\mathbb{R}$ 전체로 펴져 경계 문제도 사라진다.

    **일반 교훈.** **왈드 구간은 모수화에 의존한다.** 분산, 비율, 위험비처럼 경계가 있거나 치우친 모수는 로그나 로짓 척도에서 구간을 만들고 되돌리는 것이 표준적인 관행이다. 반면 우도비 구간은 단조변환에 불변이라 이런 고민이 없다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
세 병원에서 각각 $n_i$번의 수술 중 $k_i$번 성공했다. (가) 세 병원의 성공률이 모두 다르다고 보는 모형과 (나) 모두 같다고 보는 모형의 MLE를 각각 구하고, 우도비 검정으로 두 모형을 비교하는 통계량과 자유도를 적어라.

</div>

??? success "풀이"
    **(가) 포화모형.** 각 병원이 독립이므로 로그가능도가 분리되고

    $$
    \hat p_i = \frac{k_i}{n_i}, \qquad i=1,2,3
    $$

    이다. 모수가 3개다.

    **(나) 축소모형.** 공통 $p$에 대해

    $$
    \ell(p) = \left(\sum_i k_i\right)\ln p + \left(\sum_i (n_i-k_i)\right)\ln(1-p)
    $$

    이므로

    $$
    \hat p = \frac{\sum_i k_i}{\sum_i n_i}
    $$

    로 전체를 합친 비율이다. 모수가 1개다.

    **우도비 통계량.**

    $$
    \Lambda = 2\left\{\sum_{i=1}^3\left(k_i\ln\hat p_i + (n_i-k_i)\ln(1-\hat p_i)\right) - \left(\sum_i k_i\ln\hat p + \sum_i(n_i-k_i)\ln(1-\hat p)\right)\right\}
    $$

    자유도는 모수 개수의 차이인 $3-1 = 2$이므로 $\Lambda \sim \chi^2_2$이다.

    **알아 둘 점.**

    - 이 $\Lambda$를 **이탈도(deviance)** 라 부르며, 일반화선형모형에서 내포된 모형을 비교하는 표준 도구다. 로지스틱 회귀에서 "병원"을 요인으로 넣은 모형과 절편만 있는 모형을 비교하는 것과 같다.
    - 같은 자료에 카이제곱 동질성 검정을 적용해도 되고, 두 통계량은 점근적으로 동등하다. 우도비 쪽을 $G^2$, 피어슨 쪽을 $X^2$이라 부른다.
    - 기대도수가 작으면 두 근사 모두 나빠진다. 그때는 정확검정이나 순열검정을 쓴다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
$\text{Beta}(a,b)$ 사전분포 아래에서 $p$의 **MAP 추정값**과 **사후평균**을 구하고, MLE와의 관계를 밝혀라. 세 추정값이 일치하는 조건은 무엇인가?

</div>

??? success "풀이"
    사후분포는 $\text{Beta}(a+k,\ b+n-k)$이다.

    **사후평균.**

    $$
    \hat p_{\text{mean}} = \frac{a+k}{a+b+n}
    $$

    **MAP.** $\text{Beta}(\alpha,\beta)$의 최빈값이 $\alpha,\beta>1$일 때 $(\alpha-1)/(\alpha+\beta-2)$이므로

    $$
    \hat p_{\text{MAP}} = \frac{a+k-1}{a+b+n-2}
    $$

    **MLE.** $\hat p = k/n$.

    **관계.**

    - $a=b=1$(균등 사전분포)이면 $\hat p_{\text{MAP}} = k/n = \hat p_{\text{MLE}}$다. **균등 사전분포의 MAP는 MLE와 같다.** 사후분포가 가능도에 비례하므로 당연하다. 반면 사후평균은 $(k+1)/(n+2)$로 다르다(라플라스 평활).
    - $a=b=1/2$(제프리스 사전분포)이면 $\hat p_{\text{MAP}} = (k-0.5)/(n-1)$, 사후평균은 $(k+0.5)/(n+1)$이다.
    - $n\to\infty$이면 셋 모두 $k/n$으로 수렴한다. 사전분포의 영향이 $O(1/n)$로 사라진다.

    **세 값이 모두 일치하는 조건.** MAP와 사후평균이 같으려면 사후분포가 대칭이어야 하고, 베타분포는 두 모수가 같을 때만 대칭이므로 $a+k = b+n-k$가 필요하다. 여기에 MLE까지 맞추려면 $a=b=1$이어야 하고, 그러면 $k = n-k$, 즉 $k = n/2$다. 즉 **균등 사전분포에서 정확히 절반이 성공했을 때**만 셋이 모두 $1/2$로 일치한다.

    **해석.** MAP는 "사후분포의 봉우리", 사후평균은 "사후분포의 무게중심"이다. 사후분포가 치우쳐 있으면 둘이 갈라지며, 치우침이 심한 경계 근처에서 차이가 크다. $k=0$, $n=10$, 균등 사전분포라면 MAP는 0이지만 사후평균은 $1/12$다. **MAP는 MLE의 문제(경계값)를 물려받지만 사후평균은 그렇지 않다.**

    어느 쪽을 쓸지는 손실함수가 정한다. 제곱오차 손실이면 사후평균, 0-1 손실이면 MAP, 절대오차 손실이면 사후중앙값이 최적이다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
설명변수 $x$가 있는 로지스틱 회귀에서 자료가 **완전히 분리**되면($x < c$인 모든 관측이 0, $x > c$인 모든 관측이 1) MLE가 존재하지 않는다. 왜 그런지 설명하고 대처법을 적어라.

</div>

??? success "풀이"
    **왜 존재하지 않는가.** 로지스틱 회귀에서 $P(Y=1\mid x) = \sigma(\beta_0+\beta_1x)$이다. 완전분리가 있으면 기울기 $\beta_1$을 키울수록 모든 관측의 예측확률이 관측된 값(0 또는 1)에 가까워지므로 가능도가 계속 커진다.

    $$
    \beta_1 \to \infty \implies \ell(\boldsymbol\beta) \to 0 \quad (\text{최댓값})
    $$

    그러나 이 상한은 **달성되지 않는다.** 로그가능도가 위로 유계이지만 최댓값이 무한대에서만 접근되므로 MLE가 존재하지 않는다. 수치 최적화는 수렴하지 않고 계수와 표준오차가 발산한다.

    **증상.** 소프트웨어가 "완전분리" 경고를 내거나, 계수 추정값이 $\pm20$ 이상으로 터무니없이 크고 표준오차가 더 크게 나온다($\hat\beta = 15$, $\operatorname{SE} = 3000$ 식). 왈드 검정의 $z = \hat\beta/\operatorname{SE}$가 0으로 가서 **무한히 강한 효과인데 유의하지 않다고 나오는** 역설적 결과가 생긴다.

    **대처.**

    - **펄스 보정(Firth 방법).** 가능도에 제프리스 사전분포에 해당하는 벌점 $|I(\boldsymbol\beta)|^{1/2}$을 곱한다. 유한한 추정값이 언제나 존재하고, 덤으로 소표본 편향도 줄여 준다. **현재 표준 권고**이며 R의 `logistf`, Python의 여러 구현이 있다.
    - **약한 정보 사전분포.** 계수에 $t_1(0, 2.5)$ 같은 사전분포를 두는 베이즈 접근이다. 실질적으로 펄스 방법과 비슷한 역할을 한다.
    - **벌점회귀.** 능형(L2)이나 라소(L1)를 쓰면 계수가 유한하게 눌린다.
    - **왈드 대신 프로파일 가능도 구간.** 계수가 발산해도 프로파일 구간은 한쪽이 무한인 형태로 의미 있는 답을 준다.

    **하지 말아야 할 것.** 문제를 일으키는 변수를 그냥 빼는 것은 최악이다. 완전분리는 **그 변수가 결과를 완벽히 예측한다**는 뜻이고, 이는 통계적 문제이기 이전에 정보다. 그 변수가 결과의 일부이거나 결과 이후에 측정된 것은 아닌지(정보 누출) 먼저 확인해야 한다.

---

## 정리하며

베르누이의 최대가능도추정량은 **표본비율**이다.

$$
\hat p = \frac{1}{m}\sum_{i=1}^m x^{(i)}
$$

- **유도가 가장 단순한 경우다.** 로그가능도가 $\left(\sum x_i\right)\log p+\left(m-\sum x_i\right)\log(1-p)$ 이고, 미분해 $0$ 으로 두면 곧바로 나온다.
- **직관과 일치한다.** 앞면이 7 번 나왔으면 $\hat p=0.7$ 이다. 최대가능도가 상식적인 답을 준다는 것이 이 예의 요점이다.
- **불편이며 효율적이다.** $\mathbb{E}[\hat p]=p$ 이고 분산 $p(1-p)/m$ 이 크라메르–라오 하한과 정확히 같다. **더 나은 불편추정량은 존재하지 않는다.**
- **$\sum x_i$ 가 충분통계량이다.** 어느 시행에서 성공했는지는 $p$ 에 관해 아무 정보도 주지 않으며, 성공 횟수만 알면 된다.
- **경계에서 주의가 필요하다.** 모두 성공이면 $\hat p=1$ 이고 표준오차 추정이 $0$ 이 된다. 소표본 비율 추정에서 이 문제가 실제로 생기며, 8장의 윌슨 구간 같은 대안이 그래서 나온다.

다음 절 **정규분포의 최대가능도**로 넘어간다. 모수가 둘이라 연립방정식을 풀어야 하고, 분산추정량에서 익숙한 $n$ 대 $n-1$ 문제가 다시 나타난다.
