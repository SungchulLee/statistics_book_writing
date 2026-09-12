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

**예제 1.** 격자탐색으로 MLE 찾기

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

**예제 2.** 재모수화로 제약 없애기

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

**예제 3.** 출발값에 따른 수렴 위치

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

**연습문제 4.** <span class="diff hard" title="어려움"></span> 성분이 둘인 Gaussian 혼합에서 로그가능도가 위로 유계가 아님을 보여라(힌트: 한 성분의 분산을 어떤 자료점 주위에서 0으로 보내라). 실무에서 이것이 MLE를 무효화하지 않는 이유는 무엇인가?

</div>

??? success "풀이"
    혼합밀도 $\pi \cdot N(x_1, \sigma_1^2) + (1-\pi) \cdot N(\mu_2, \sigma_2^2)$을 생각하자. $\mu_1 = x_1$(어떤 자료점)로 두고 $\sigma_1 \to 0$으로 보내면 $x_1$에서 첫 성분의 밀도가 $1/\sigma_1 \to \infty$로 발산하여 로그가능도가 유계가 아니게 된다.

    실무에서 이것이 문제가 되지 않는 이유는:

    1. 이런 퇴화된 해는 자료점 하나에 과대적합한 것이어서 통계적으로 무의미하다.
    2. 혼합모형의 표준 방법인 EM 알고리즘은 합리적인 출발값에서 그런 퇴화된 해에 도달하지 못한다.
    3. 실무자는 최소 분산 제약을 두거나 벌점 가능도를 쓴다.
    4. *유용한* MLE는 전역 상한이 아니라 가능도의 국소 최댓값이다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> Bernoulli 분포의 모수 $p$를 추정하는 Fisher 점수법 알고리즘을 구현하라. Fisher 정보량은 $I(p) = 1/[p(1-p)]$이다. 참 $p = 0.3$인 $n = 50$개의 관측값에서 Newton-Raphson과 수렴 속도를 비교하라.

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

    둘 다 $\hat{p} = x_{\text{sum}}/n$으로 수렴한다. Bernoulli에서는 관측 정보량과 기대 정보량이 밀접하게 연결되어 있어 수렴 속도가 거의 같다. 일반적으로는 관측 Hessian의 조건수가 나쁠 때 Fisher 점수법이 더 안정적일 수 있다. $\square$

---

## 정리하며

닫힌 형태가 없으면 **수치적으로** 봉우리를 찾는다.

- **최소화로 바꿔 푼다.** 대부분의 최적화 라이브러리가 최소화를 다루므로 $-\ell(\theta)$ 를 최소화하며, 로그를 쓰는 덕분에 곱이 합이 되고 언더플로를 피한다.
- **방법의 계층.** 격자탐색은 모수가 하나둘일 때 확실하지만 차원이 늘면 쓸 수 없다. 기울기·뉴턴 계열은 빠르지만 출발값에 민감하다. EM 알고리즘은 결측이나 잠재변수 구조가 있을 때 각 단계에서 가능도가 **줄지 않음**이 보장된다.
- **출발값이 결과를 바꾼다.** 가능도가 다봉이면 지역 최적에 갇힌다. 여러 출발값에서 돌려 같은 곳으로 모이는지 확인하는 것이 기본 진단이다.
- **수렴했다고 최적은 아니다.** 기울기가 $0$ 에 가까운 것은 필요조건일 뿐이며, 헤세행렬이 음정치인지도 확인해야 한다. 3장에서 본 정규혼합처럼 **가능도가 위로 유계가 아니면** 경계로 달아나는 해가 나온다.
- **모수를 변환해 제약을 없애는 것**이 실무의 요령이다. $\sigma>0$ 은 $\log\sigma$ 로, 확률 $p\in(0,1)$ 은 로짓으로 두면 무제약 최적화가 된다.

다음 절 **피셔 정보량 계산**에서 이 최적화 결과로부터 표준오차를 뽑아내는 코드를 본다.
