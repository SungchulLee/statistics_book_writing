# 최대가능도 최적화 예제

## 개요

닫힌 형태의 해가 없으면 최대가능도추정에는 수치 최적화가 필요한 경우가 많다. 이 페이지에서는 실무에서 쓰는 주요 최적화 기법인 격자탐색, 기울기 기반 방법, 기댓값–최대화 알고리즘과 함께 수렴을 확인하는 진단과 출발값의 효과를 살펴본다.

## 최적화 문제

모수족 $f(x; \theta)$에서 얻은 관측 자료 $x_1, \ldots, x_n$이 주어졌을 때 MLE는 다음을 푼다:

$$
\hat{\theta}_{\text{MLE}} = \arg\max_\theta \ell(\theta) = \arg\max_\theta \sum_{i=1}^n \log f(x_i; \theta)
$$

동등하게 음의 로그가능도를 최소화한다:

$$
\hat{\theta}_{\text{MLE}} = \arg\min_\theta \left[-\ell(\theta)\right]
$$

!!! warning "실무적 고려사항"

    - 로그가능도에 국소 최댓값이 여럿 있을 수 있다(예: 혼합모형).
    - 모수공간에 제약이 있으면(예: $\sigma^2 > 0$) 재모수화하거나 제약 최적화를 써야 한다.
    - 출발값이 나쁘면 국소 최적점으로 수렴하거나 수치적으로 실패할 수 있다.

## 격자탐색

가장 단순한 최적화 전략은 후보값 격자에서 $\ell(\theta)$를 평가하는 것이다. 모수가 하나나 둘일 때 실행 가능하며 가능도 곡면을 시각화하는 데 유용하다.

<div class="codebox" markdown>

### 예제 1. 격자탐색으로 MLE 찾기 { .eg }

```python
import numpy as np
from scipy import stats

def grid_search_normal_mean(data, mu_grid):
    """mu 후보를 격자로 늘어놓고 로그가능도가 가장 큰 것을 고른다.

    가장 단순하고 가장 확실한 방법이다. 미분도 초기값도 필요 없고
    국소 최적에 갇히지도 않는다. 대신 격자 간격보다 정밀할 수 없고,
    모수가 d개면 격자점이 (격자 수)^d 로 폭발해 d가 3~4만 넘어도 못 쓴다.
    """
    sigma_hat = data.std(ddof=0)     # ddof=0 이 MLE 판본이다(n으로 나눔)

    # logpdf 를 더한다. pdf를 곱한 뒤 로그를 취하면 언더플로가 나므로
    # 처음부터 로그로 계산해 더하는 것이 정석이다.
    log_liks = np.array([
        np.sum(stats.norm.logpdf(data, loc=mu, scale=sigma_hat))
        for mu in mu_grid
    ])
    best_idx = np.argmax(log_liks)
    return mu_grid[best_idx], log_liks

# 보기.
rng = np.random.default_rng(42)
data = rng.normal(loc=5.0, scale=2.0, size=100)
mu_grid = np.linspace(3.0, 7.0, 500)
mu_hat, log_liks = grid_search_normal_mean(data, mu_grid)
print(f"Grid search MLE: mu_hat = {mu_hat:.4f}")
print(f"Closed-form MLE: mu_hat = {data.mean():.4f}")
```

출력:

```
Grid search MLE: mu_hat = 4.8998
Closed-form MLE: mu_hat = 4.8995
```

</div>

## 기울기 기반 최적화

모수가 여럿인 문제에서는 기울기 기반 방법이 필수적이다. (로그가능도의 기울기인) **점수함수**는:

$$
S(\theta) = \frac{\partial}{\partial\theta}\ell(\theta)
$$

MLE에서 점수는 0이다: $S(\hat{\theta}) = 0$.

### 제약 없는 최적화를 위한 재모수화

모수에 제약이 있으면(예: $\sigma^2 > 0$) 변환된 모수에 대해 최적화하는 것이 흔한 요령이다:

$$
\phi = \log(\sigma^2) \quad \Rightarrow \quad \sigma^2 = e^\phi
$$

이렇게 하면 제약 문제가 제약 없는 문제로 바뀐다.

<div class="codebox" markdown>

#### 예제 2. 재모수화로 제약 없애기 { .eg }

```python
import numpy as np
from scipy import optimize

def mle_normal_numerical(data):
    """재매개변수화를 써서 정규분포의 MLE를 수치적으로 찾는다."""
    def neg_log_lik(params):
        # sigma^2 을 직접 다루지 않고 log(sigma^2) 을 최적화한다.
        # 이유: sigma^2 > 0 이라는 제약을 최적화기에 알려 주기 어려운데,
        #       log를 쓰면 log_sigma2 가 어떤 실수든 exp를 거치며 자동으로 양수가 된다.
        #       제약 없는 최적화 문제로 바뀌므로 Nelder-Mead 같은 단순한 방법도 쓸 수 있다.
        mu, log_sigma2 = params
        sigma2 = np.exp(log_sigma2)
        n = len(data)
        # 음의 로그가능도. 최소화하는 것이 가능도를 최대화하는 것과 같다.
        return 0.5 * n * np.log(2 * np.pi * sigma2) + np.sum((data - mu) ** 2) / (2 * sigma2)

    # 출발점을 여러 개 시도한다.
    # 정규분포의 로그가능도는 볼록해서 사실 한 번이면 충분하지만,
    # 봉우리가 여럿인 문제에서는 이렇게 여러 곳에서 출발해
    # 가장 좋은 것을 골라야 국소 최적에 갇히지 않는다.
    # (numpy 배열에는 .median() 이 없으므로 np.median 을 쓴다.)
    best_result = None
    for mu0 in [0, data.mean(), np.median(data)]:
        for ls0 in [0, np.log(data.var())]:
            result = optimize.minimize(neg_log_lik, x0=[mu0, ls0], method="Nelder-Mead")
            # result.fun 이 그 출발점에서 도달한 최솟값이다
            if best_result is None or result.fun < best_result.fun:
                best_result = result

    mu_hat = best_result.x[0]
    sigma2_hat = np.exp(best_result.x[1])    # log에서 되돌린다
    return mu_hat, sigma2_hat

rng = np.random.default_rng(42)
data = rng.normal(5.0, 2.0, 100)
mu_hat, sigma2_hat = mle_normal_numerical(data)
print(f"Numerical MLE: mu = {mu_hat:.4f}, sigma^2 = {sigma2_hat:.4f}")
print(f"Closed-form:   mu = {data.mean():.4f}, sigma^2 = {np.mean((data - data.mean())**2):.4f}")
```

출력:

```
Numerical MLE: mu = 4.8995, sigma^2 = 2.3888
Closed-form:   mu = 4.8995, sigma^2 = 2.3888
```

</div>

## Newton-Raphson 방법

Newton-Raphson 방법은 (Hessian이라는) 2계 정보를 사용하여 더 빠르게 수렴한다:

$$
\theta^{(t+1)} = \theta^{(t)} - \left[\ell''(\theta^{(t)})\right]^{-1} \ell'(\theta^{(t)})
$$

다변량인 경우에는 다음과 같다:

$$
\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - \mathbf{H}^{-1}(\boldsymbol{\theta}^{(t)})\, \nabla\ell(\boldsymbol{\theta}^{(t)})
$$

여기서 $\mathbf{H}$는 로그가능도의 Hessian 행렬이다.

!!! info "Fisher 점수법"
    관측 Hessian을 그 기댓값 $-I(\theta)$(음의 Fisher 정보행렬)로 바꾸면 **Fisher 점수법** 알고리즘이 된다. MLE 근처에서는 Fisher 점수법과 Newton-Raphson이 비슷하게 거동한다.

## 출발값에 대한 민감도

볼록하지 않은 가능도(예: 혼합모형)에서는 최적화 결과가 출발점에 의존할 수 있다.

<div class="codebox" markdown>

### 예제 3. 출발값에 따른 수렴 위치 { .eg }

```python
import numpy as np
from scipy import optimize, stats

def mixture_log_likelihood(params, data):
    """성분 둘짜리 정규 혼합모형의 음의 로그가능도."""
    pi, mu1, mu2, sigma = params[0], params[1], params[2], np.exp(params[3])
    pi = 1 / (1 + np.exp(-pi))  # sigmoid transform for mixing weight
    ll = np.sum(np.log(
        pi * stats.norm.pdf(data, mu1, sigma) +
        (1 - pi) * stats.norm.pdf(data, mu2, sigma)
    ))
    return -ll

# 봉우리 둘짜리 혼합분포에서 자료를 만든다.
rng = np.random.default_rng(42)
n = 200
z = rng.binomial(1, 0.4, n)
data = np.where(z, rng.normal(0, 1, n), rng.normal(4, 1, n))

# 출발값을 바꿔 가며 어디로 수렴하는지 본다.
starts = [[0, -1, 5, 0], [0, 2, 2, 0], [0, 0, 3, 0.5]]
for i, x0 in enumerate(starts):
    result = optimize.minimize(mixture_log_likelihood, x0, args=(data,), method="Nelder-Mead")
    pi_hat = 1 / (1 + np.exp(-result.x[0]))
    print(f"Start {i+1}: pi={pi_hat:.3f}, mu1={result.x[1]:.3f}, "
          f"mu2={result.x[2]:.3f}, nll={result.fun:.2f}")
```

출력:

```
Start 1: pi=0.374, mu1=0.154, mu2=3.900, nll=402.58
Start 2: pi=0.626, mu1=3.900, mu2=0.154, nll=402.58
Start 3: pi=0.374, mu1=0.154, mu2=3.899, nll=402.58
```

</div>

## 해석

- **격자탐색**은 저차원 문제에서 믿을 만하며 가능도 곡면을 직접 시각화해 준다.
- **기울기 기반 방법**(Nelder-Mead, BFGS, Newton-Raphson)은 고차원으로 확장되지만 국소 최적점으로 수렴할 수 있다.
- **재모수화**는 제약 최적화를 제약 없는 최적화로 바꾸어 수치적 안정성을 높인다.
- **여러 출발값으로 재시작**하면 봉우리가 여럿인지 진단하는 데 도움이 된다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span> $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Exp}(\lambda)$에 대해 음의 로그가능도를 쓰고 MLE를 해석적으로 구하라. $\lambda \in [0.1, 5]$에서 격자탐색을 구현하여 답을 확인하라.

</div>

??? success "풀이"
    로그가능도는:

    $$
    \ell(\lambda) = n\log\lambda - \lambda\sum_{i=1}^n x_i
    $$

    $\ell'(\lambda) = n/\lambda - \sum x_i = 0$으로 두면 $\hat{\lambda}_{\text{MLE}} = n/\sum x_i = 1/\bar{X}$이다.

    ```python
    import numpy as np

    rng = np.random.default_rng(42)
    data = rng.exponential(scale=2.0, size=50)  # true lambda = 0.5
    lam_grid = np.linspace(0.1, 5, 1000)
    ll = len(data) * np.log(lam_grid) - lam_grid * data.sum()
    lam_hat_grid = lam_grid[np.argmax(ll)]
    lam_hat_exact = 1 / data.mean()
    print(f"Grid MLE:    {lam_hat_grid:.4f}")
    print(f"Analytic MLE: {lam_hat_exact:.4f}")
    ```

    출력:

    ```
    Grid MLE:    0.6003
    Analytic MLE: 0.5982
    ```

    두 값이 거의 일치해야 한다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 수치적 MLE에서 $\sigma^2$을 직접 최적화하는 대신 $\phi = \log(\sigma^2)$을 최적화하는 편이 나은 이유를 설명하라. 이 변환의 어떤 성질이 최적화기가 $\sigma^2 \leq 0$에서 평가하지 않도록 보장하는가?

</div>

??? success "풀이"
    지수함수 $\sigma^2 = e^\phi$는 $\phi \in \mathbb{R}$를 $\sigma^2 \in (0, \infty)$로 보낸다. 모든 실수 $\phi$에 대해 $e^\phi > 0$이므로 최적화기는 유효하지 않은(양수가 아닌) 분산을 만들 걱정 없이 $\mathbb{R}$ 전체를 탐색할 수 있다. 이 재모수화가 없으면 기울기 단계가 $\sigma^2$을 0 아래로 밀어낼 수 있고, (정규 로그가능도에 $\log(\sigma^2)$이 나오므로) 로그가능도가 정의되지 않게 된다. 이 변환은 곡률을 더 고르게 만들어 최적화 지형도 개선한다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> 관측값 $x$ 하나가 주어졌을 때 $\text{Binomial}(n, p)$의 $p$를 추정하는 Newton-Raphson 갱신식을 유도하라. $p^{(0)} = 0.5$에서 시작하여 $n = 20, x = 14$일 때 처음 두 번의 반복값을 계산하라.

</div>

??? success "풀이"
    (상수를 무시한) 로그가능도는:

    $$
    \ell(p) = x\log p + (n - x)\log(1 - p)
    $$

    점수: $\ell'(p) = x/p - (n-x)/(1-p)$.

    Hessian: $\ell''(p) = -x/p^2 - (n-x)/(1-p)^2$.

    Newton-Raphson 갱신: $p^{(t+1)} = p^{(t)} - \ell'(p^{(t)})/\ell''(p^{(t)})$.

    $n = 20, x = 14, p^{(0)} = 0.5$일 때:

    - $\ell'(0.5) = 14/0.5 - 6/0.5 = 28 - 12 = 16$
    - $\ell''(0.5) = -14/0.25 - 6/0.25 = -56 - 24 = -80$
    - $p^{(1)} = 0.5 - 16/(-80) = 0.5 + 0.2 = 0.7$

    $p^{(1)} = 0.7$에서:

    - $\ell'(0.7) = 14/0.7 - 6/0.3 = 20 - 20 = 0$

    따라서 $p^{(2)} = 0.7$이며, 이는 이미 MLE $\hat{p} = x/n = 14/20 = 0.7$이다. Newton-Raphson이 한 단계 만에 수렴했다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span> 성분이 둘인 정규 혼합에서 로그가능도가 위로 유계가 아님을 보여라(힌트: 한 성분의 분산을 어떤 자료점 주위에서 0으로 보내라). 실무에서 이것이 MLE를 무효화하지 않는 이유는 무엇인가?

</div>

??? success "풀이"
    혼합밀도 $\pi \cdot N(x_1, \sigma_1^2) + (1-\pi) \cdot N(\mu_2, \sigma_2^2)$을 생각하자. $\mu_1 = x_1$(어떤 자료점)로 두고 $\sigma_1 \to 0$으로 보내면 $x_1$에서 첫 성분의 밀도가 $1/\sigma_1 \to \infty$로 발산하여 로그가능도가 유계가 아니게 된다.

    실무에서 이것이 문제가 되지 않는 이유는:

    1. 이런 퇴화된 해는 자료점 하나에 과대적합한 것이어서 통계적으로 무의미하다.
    2. 혼합모형의 표준 방법인 EM 알고리즘은 합리적인 출발값에서 그런 퇴화된 해에 도달하지 못한다.
    3. 실무자는 최소 분산 제약을 두거나 벌점 가능도를 쓴다.
    4. *유용한* MLE는 전역 상한이 아니라 가능도의 국소 최댓값이다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> 베르누이분포의 모수 $p$를 추정하는 Fisher 점수법 알고리즘을 구현하라. Fisher 정보량은 $I(p) = 1/[p(1-p)]$이다. 참 $p = 0.3$인 $n = 50$개의 관측값에서 Newton-Raphson과 수렴 속도를 비교하라.

</div>

??? success "풀이"
    ```python
    import numpy as np

    rng = np.random.default_rng(42)
    n = 50
    data = rng.binomial(1, 0.3, n)
    x_sum = data.sum()

    # 뉴턴-랩슨: 관측정보량을 쓴다.
    p_nr = 0.5
    for i in range(10):
        score = x_sum / p_nr - (n - x_sum) / (1 - p_nr)
        hessian = -x_sum / p_nr**2 - (n - x_sum) / (1 - p_nr)**2
        p_nr = p_nr - score / hessian
        print(f"NR  iter {i+1}: p = {p_nr:.8f}")

    # 피셔 스코어링: 기대정보량을 쓴다.
    p_fs = 0.5
    for i in range(10):
        score = x_sum / p_fs - (n - x_sum) / (1 - p_fs)
        fisher_info = n / (p_fs * (1 - p_fs))
        p_fs = p_fs + score / fisher_info
        print(f"FS  iter {i+1}: p = {p_fs:.8f}")
    ```

    출력:

    ```
    NR  iter 1: p = 0.36000000
    NR  iter 2: p = 0.36000000
    NR  iter 3: p = 0.36000000
    NR  iter 4: p = 0.36000000
    NR  iter 5: p = 0.36000000
    NR  iter 6: p = 0.36000000
    NR  iter 7: p = 0.36000000
    NR  iter 8: p = 0.36000000
    NR  iter 9: p = 0.36000000
    NR  iter 10: p = 0.36000000
    FS  iter 1: p = 0.36000000
    FS  iter 2: p = 0.36000000
    FS  iter 3: p = 0.36000000
    FS  iter 4: p = 0.36000000
    FS  iter 5: p = 0.36000000
    FS  iter 6: p = 0.36000000
    FS  iter 7: p = 0.36000000
    FS  iter 8: p = 0.36000000
    FS  iter 9: p = 0.36000000
    FS  iter 10: p = 0.36000000
    ```

    둘 다 $\hat{p} = x_{\text{sum}}/n$으로 수렴한다. 베르누이에서는 관측 정보량과 기대 정보량이 밀접하게 연결되어 있어 수렴 속도가 거의 같다. 일반적으로는 관측 Hessian의 조건수가 나쁠 때 Fisher 점수법이 더 안정적일 수 있다. $\square$

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
수치 최적화가 수렴했다고 판정하는 기준을 세 가지 들고, 각각이 실패하는 경우를 하나씩 적어라.

</div>

??? success "풀이"
    **(1) 목적함수의 변화.** $|\ell^{(k+1)}-\ell^{(k)}| < \varepsilon$.

    - **실패**: 목적함수가 평평한 고원을 지날 때 변화가 작아 조기에 멈춘다. 실제로는 아직 봉우리에서 멀 수 있다.
    - 상대 기준 $|\Delta\ell|/(|\ell|+1) < \varepsilon$을 쓰는 편이 척도에 덜 민감하다.

    **(2) 모수의 변화.** $\|\boldsymbol\theta^{(k+1)}-\boldsymbol\theta^{(k)}\| < \varepsilon$.

    - **실패**: 보폭이 작게 잡혀 천천히 기어갈 때도 만족된다. 감쇠가 과한 알고리즘에서 흔하다.
    - 모수마다 척도가 다르면 상대 변화로 재야 한다.

    **(3) 기울기의 크기.** $\|\nabla\ell(\boldsymbol\theta^{(k)})\| < \varepsilon$.

    - **가장 원리적**이다. 내부 최댓값의 필요조건이 기울기 0이기 때문이다.
    - **실패**: 최댓값이 **경계**에 있으면 기울기가 0이 아니어서 영원히 만족되지 않는다. 또 기울기의 크기는 목적함수의 척도에 의존하므로 절대 기준을 정하기 어렵다.

    **권고.** 세 기준을 **모두** 요구하고, 여기에 다음을 더한다.

    - **헤시안 확인.** 수렴한 점에서 헤시안이 음정부호인지 본다. 그렇지 않으면 안장점이거나 수렴하지 않은 것이다. 표준오차를 구하려면 어차피 계산해야 한다.
    - **여러 초기값.** 서로 다른 출발점에서 같은 곳으로 가는지 본다. 다봉성을 잡아내는 가장 확실한 방법이다.
    - **최대 반복 횟수 도달 여부.** 수렴 플래그를 반드시 확인한다. 많은 라이브러리가 조용히 최대 횟수에서 멈추고 결과를 돌려준다.

    **실무의 사고.** `scipy.optimize.minimize`의 결과에서 `res.success`를 확인하지 않고 `res.x`만 쓰는 것이 흔한 실수다. 수렴하지 않은 값을 추정값으로 보고하게 된다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
모수에 제약이 있을 때($\sigma > 0$, $0<p<1$, $\sum\pi_k=1$) 이를 다루는 세 가지 방법을 비교하라.

</div>

??? success "풀이"
    **(1) 재모수화.** 제약을 자동으로 만족하는 변환을 쓴다.

    | 제약 | 변환 |
    |---|---|
    | $\sigma>0$ | $\sigma = e^\phi$ |
    | $0<p<1$ | $p = \sigma(\eta) = 1/(1+e^{-\eta})$ |
    | $\sum\pi_k=1$, $\pi_k>0$ | 소프트맥스 $\pi_k = e^{\eta_k}/\sum_j e^{\eta_j}$ |
    | 공분산행렬 양정부호 | 촐레스키 인자의 대각을 지수로 |

    - **장점**: 제약 없는 최적화기를 그대로 쓸 수 있다. 로그가능도가 새 척도에서 오히려 더 이차식에 가까워지는 경우도 많다.
    - **단점**: 최적해가 경계에 있으면 $\phi \to -\infty$로 발산한다. 표준오차를 원래 척도로 되돌리려면 델타 방법이 필요하다.

    **(2) 경계 제약 최적화.** `L-BFGS-B`처럼 상자 제약을 지원하는 알고리즘을 쓴다.

    - **장점**: 직관적이고, 경계해가 나와도 그 값을 그대로 준다.
    - **단점**: 선형·비선형 제약(예: $\sum\pi_k=1$)은 상자로 표현할 수 없다. 경계에서 멈추면 표준오차의 정규근사가 성립하지 않는다.

    **(3) 벌점이나 사전분포.** 목적함수에 $-\gamma/\sigma^2$ 같은 항을 더하거나 베이즈 사전분포를 둔다.

    - **장점**: 경계로 가는 퇴화를 근본적으로 막는다. 혼합모형의 분산 퇴화를 다루는 표준적인 방법이다.
    - **단점**: 목적함수가 달라지므로 더 이상 순수한 MLE가 아니다. 벌점의 세기를 정해야 한다.

    **권고.** 대부분의 경우 **재모수화**가 가장 깔끔하다. 최적해가 경계에 있을 가능성이 실질적이면(분산성분이 0일 수 있는 혼합효과 모형 등) 경계 제약 최적화를 쓰고, **경계해가 나왔을 때 표준적인 추론을 쓰면 안 된다**는 점을 기억한다. 그 경우 우도비 통계량의 극한분포가 $\chi^2$가 아니라 $\chi^2$들의 혼합이 된다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
같은 모형을 최적화하는데 초기값에 따라 다른 답이 나온다. 국소 최댓값이 여럿인지 확인하고 전역 최댓값을 찾는 전략을 적어라.

</div>

??? success "풀이"
    **확인.**

    - **여러 초기값에서 돌린다.** 수렴한 $\ell$ 값들을 히스토그램으로 보면 봉우리가 몇 개인지 드러난다. 값이 여러 무리로 갈리면 다봉성이다.
    - **모수가 1~2개면 격자로 그린다.** 가장 확실하다. 3개 이상이면 두 개씩 골라 프로파일 곡면을 그린다.
    - **수렴점 사이를 잇는 경로를 따라 $\ell$을 계산한다.** 두 해 사이에 골짜기가 있으면 서로 다른 봉우리다.

    **전역 최댓값을 찾는 전략.**

    - **다중 출발(multistart).** 모수공간에서 무작위로(또는 라틴 초입방 설계로) 초기값을 여러 개 뽑아 각각 국소 최적화하고 가장 좋은 것을 고른다. 가장 널리 쓰이고 병렬화가 쉽다.
    - **좋은 초기값을 설계한다.** 적률법 추정값, 더 단순한 모형의 해, 자료를 나눠 각각 적합한 값 등. 혼합모형이라면 $k$-평균 군집 결과로 초기화하는 것이 표준이다.
    - **전역 최적화 알고리즘.** `basinhopping`, `differential_evolution`, 시뮬레이티드 어닐링. 느리지만 넓게 훑는다.
    - **점진적 접근.** 단순한 모형에서 시작해 복잡도를 조금씩 올리며 직전 해를 초기값으로 쓴다. 성분 수를 1부터 늘려 가는 혼합모형 적합이 그 예다.
    - **결정론적 어닐링.** 목적함수를 처음에는 매끄럽게 만들었다가 점차 원래대로 되돌린다.

    **보고할 때.** 전역 최댓값을 찾았다고 **증명할 수는 없다.** 정직하게 하려면 몇 개의 초기값에서 출발했고 몇 개의 서로 다른 국소해를 찾았는지 기록한다. 국소해들의 $\ell$ 값이 비슷하면서 모수가 크게 다르다면, 그것은 **모형이 식별되지 않는다는 신호**일 수 있으므로 최적화 문제로만 볼 것이 아니다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
최적화 후에 표준오차를 얻으려면 헤시안이 필요하다. 최적화기가 돌려준 헤시안 근사를 그대로 써도 되는가? 주의할 점을 적어라.

</div>

??? success "풀이"
    **대체로 그대로 쓰면 안 된다.**

    **이유 1 — 준뉴턴 근사는 정확하지 않다.** BFGS가 유지하는 `hess_inv`는 **최적화를 안내하기 위한** 근사이며, 수렴 경로에서 본 기울기 변화로 쌓아 올린 것이다. 봉우리에서의 참 곡률과는 다를 수 있고, 특히 반복 횟수가 적으면 초기 근사(보통 단위행렬)의 흔적이 남는다.

    **이유 2 — 척도 문제.** 재모수화해서 최적화했다면 헤시안도 그 척도의 것이다. 원래 모수의 표준오차를 얻으려면 야코비안으로 변환해야 한다.

    $$
    \operatorname{Var}(\hat\theta) = \left(\frac{\partial\theta}{\partial\phi}\right)^2\operatorname{Var}(\hat\phi)
    $$

    이것을 잊는 것이 가장 흔한 실수다.

    **이유 3 — 목적함수의 부호와 상수.** 최소화를 위해 **음의** 로그가능도를 넘겼다면 그 헤시안이 곧 관측정보량이다. 부호를 한 번 더 뒤집으면 음수 분산이 나온다. 또 $-2\ell$을 최소화했다면 헤시안이 2배이므로 표준오차가 $\sqrt2$배 틀린다.

    **권장 절차.**

    1. 최적화는 아무 방법으로나 한다.
    2. 수렴한 $\hat{\boldsymbol\theta}$에서 헤시안을 **다시** 계산한다. 자동미분이 가장 좋고, 없으면 `numdifftools` 같은 고정밀 수치 미분을 쓴다.
    3. 그 헤시안이 양정부호인지 확인한다(음의 로그가능도 기준). 아니면 수렴하지 않았거나 안장점이다.
    4. 역행렬의 대각 제곱근이 표준오차다. 필요하면 델타 방법으로 원 척도로 옮긴다.
    5. 가능하면 **프로파일 가능도나 부트스트랩으로 교차 확인**한다. 왈드 표준오차와 크게 다르면 이차 근사가 나쁘다는 뜻이다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
관측이 $n = 10^7$개인 자료에 MLE를 적합하려 한다. 전체 자료를 한 번에 쓰는 방법이 비현실적일 때 쓸 수 있는 접근을 세 가지 적어라.

</div>

??? success "풀이"
    **(1) 확률적 경사법.** 매 단계에서 자료 전체가 아니라 무작위로 고른 작은 묶음(minibatch)으로 기울기를 근사한다.

    $$
    \boldsymbol\theta^{(k+1)} = \boldsymbol\theta^{(k)} + \eta_k\,\frac{n}{|B_k|}\sum_{i\in B_k}s(x_i;\boldsymbol\theta^{(k)})
    $$

    - 한 단계의 비용이 $n$과 무관하다.
    - 학습률 $\eta_k$가 $\sum\eta_k=\infty$, $\sum\eta_k^2<\infty$를 만족하면 수렴이 보장된다(로빈스-먼로).
    - 기계학습의 표준이며, Adam 같은 적응적 변형이 널리 쓰인다.
    - **단점**: 표준오차를 곧바로 주지 않고, 수렴 판정이 까다롭다.

    **(2) 충분통계량 활용.** 지수족이면 자료 전체가 아니라 **저차원 요약**만 있으면 된다. 정규분포는 $(\sum x_i, \sum x_i^2)$, 포아송은 $\sum x_i$, 로지스틱 회귀는 $X^\top\mathbf{y}$와 $X^\top X$면 충분하다.

    - 자료를 한 번만 훑으며 통계량을 누적하면 그 뒤로는 $n$과 무관한 비용으로 최적화한다.
    - 분산 처리도 쉽다. 각 노드에서 통계량을 계산해 더하기만 하면 된다.
    - **단점**: 지수족이 아니면 쓸 수 없다.

    **(3) 표본추출·분할정복.**

    - **부분표본.** $n=10^7$이면 $10^5$만 써도 표준오차가 $\sqrt{100}=10$배 커질 뿐이다. 이미 충분히 정밀하다면 나머지를 쓸 이유가 없다. **정밀도가 이미 실무 요구를 넘었는지** 먼저 따져 볼 일이다.
    - **분할정복.** 자료를 $K$조각으로 나눠 각각 $\hat\theta_j$를 구하고 평균한다. 앞서 본 정보량 가중을 쓰면 된다. 일차적으로는 전체 MLE와 점근적으로 동등하다.
    - **중요도 가중 부분표본.** 영향력이 큰 관측을 더 자주 뽑고 가중치로 보정한다. 희귀사건 자료에서 특히 효과적이다.

    **고르는 기준.** 모형이 지수족이면 (2)가 압도적으로 좋다. 그렇지 않고 정밀도가 이미 충분하면 (3)의 부분표본이 가장 간단하다. 복잡한 모형을 전체 자료로 적합해야 한다면 (1)을 쓰되, 표준오차는 부트스트랩이나 샌드위치로 따로 구한다.

---

## 정리하며

닫힌 형태가 없으면 **수치적으로** 봉우리를 찾는다.

- **최소화로 바꿔 푼다.** 대부분의 최적화 라이브러리가 최소화를 다루므로 $-\ell(\theta)$ 를 최소화하며, 로그를 쓰는 덕분에 곱이 합이 되고 언더플로를 피한다.
- **방법의 계층.** 격자탐색은 모수가 하나둘일 때 확실하지만 차원이 늘면 쓸 수 없다. 기울기·뉴턴 계열은 빠르지만 출발값에 민감하다. EM 알고리즘은 결측이나 잠재변수 구조가 있을 때 각 단계에서 가능도가 **줄지 않음**이 보장된다.
- **출발값이 결과를 바꾼다.** 가능도가 다봉이면 지역 최적에 갇힌다. 여러 출발값에서 돌려 같은 곳으로 모이는지 확인하는 것이 기본 진단이다.
- **수렴했다고 최적은 아니다.** 기울기가 $0$ 에 가까운 것은 필요조건일 뿐이며, 헤세행렬이 음정치인지도 확인해야 한다. 3장에서 본 정규혼합처럼 **가능도가 위로 유계가 아니면** 경계로 달아나는 해가 나온다.
- **모수를 변환해 제약을 없애는 것**이 실무의 요령이다. $\sigma>0$ 은 $\log\sigma$ 로, 확률 $p\in(0,1)$ 은 로짓으로 두면 무제약 최적화가 된다.

다음 절 **피셔 정보량 계산**에서 이 최적화 결과로부터 표준오차를 뽑아내는 코드를 본다.
