# 기하과 포아송의 최대가능도

## 개요

최대가능도추정(MLE)은 관측된 자료에 모수적 모형을 맞추는 원리 있는 방법을 제공한다. 이 페이지에서는 두 가지 기본적인 이산분포인 기하과 포아송의 MLE를 유도하고 시연하며, 모수적 MLE 적합을 남겨 둔 검정 자료에서 비모수적 경험 PMF와 비교한다. 이 분석은 모형이 올바르게 설정되었을 때 모수적 모형이 왜 더 잘 일반화되는지를 부각한다.

## 기하분포의 MLE

### 모형

기하분포는 첫 실패 이전의 연속 성공 횟수를 모형화한다. 모수 $p$(각 시행의 성공확률)에 대해:

$$
P(X = k) = (1 - p)\, p^k, \quad k = 0, 1, 2, \ldots
$$

평균은 $E[X] = p / (1 - p)$이다.

### MLE 유도

관측값 $x_1, \ldots, x_n$이 주어졌을 때 로그가능도는:

$$
\ell(p) = \sum_{i=1}^n \left[ x_i \log p + \log(1 - p) \right] = n_s \log p + n \log(1 - p)
$$

여기서 $n_s = \sum_{i=1}^n x_i$는 전체 성공 횟수이다. 미분하여 0으로 두면:

$$
\frac{d\ell}{dp} = \frac{n_s}{p} - \frac{n}{1 - p} = 0
$$

풀면:

$$
\hat{p}_{\text{MLE}} = \frac{n_s}{n_s + n} = \frac{\bar{x}}{1 + \bar{x}}
$$

### 시연

<div class="codebox" markdown>

#### 예제 1. 기하분포의 MLE { .eg }

```python
import numpy as np

np.random.seed(42)

def geometric_mle_demo(n_train=1000, n_test=1000, p_true=0.12):
    """기하분포의 MLE — 모수적 적합과 비모수적 적합을 견준다."""
    # 자료를 만든다. 첫 실패가 나올 때까지의 성공 횟수다.
    train = np.random.geometric(1 - p_true, n_train) - 1  # 0-indexed
    test = np.random.geometric(1 - p_true, n_test) - 1

    # 최대가능도추정값.
    p_hat = train.mean() / (1 + train.mean())
    k_max = max(train.max(), test.max()) + 1
    k_vals = np.arange(k_max)

    # 모수적 적합: MLE를 넣은 확률질량함수.
    pmf_param = (1 - p_hat) * p_hat ** k_vals

    # 비모수적 적합: 자료의 상대도수를 그대로 쓴다.
    pmf_train = np.bincount(train, minlength=k_max) / n_train
    pmf_test = np.bincount(test, minlength=k_max) / n_test

    # 시험자료에서의 RMSE.
    err_param = np.sqrt(np.mean((pmf_param - pmf_test)**2))
    err_nonparam = np.sqrt(np.mean((pmf_train - pmf_test)**2))

    print(f"True p = {p_true:.3f}")
    print(f"MLE p_hat = {p_hat:.4f}")
    print(f"Test RMSE -- parametric: {err_param:.5f}")
    print(f"Test RMSE -- nonparametric: {err_nonparam:.5f}")

geometric_mle_demo()
```

출력:

```
True p = 0.120
MLE p_hat = 0.1220
Test RMSE -- parametric: 0.00996
Test RMSE -- nonparametric: 0.01164
```

</div>

### 로그가능도 곡면

로그가능도는 $p$에 대해 오목한 함수이며 유일한 전역 최댓값이 있음을 확인해 준다:

<div class="codebox" markdown>

#### 예제 2. 기하분포 로그가능도 곡면 { .eg }

```python
import matplotlib.pyplot as plt

# numpy의 geometric은 "첫 성공까지의 시행 수"(1부터)를 준다.
# 여기서 쓰는 판본은 "첫 성공 이전의 실패 수"(0부터)이므로 1을 뺀다.
# 성공확률을 1 - 0.12 로 준 것은, 이 절의 theta 가 numpy의 p와
# 서로 여집합 관계이기 때문이다(theta = 실패확률).
train = np.random.geometric(1 - 0.12, 1000) - 1

# 로그가능도 l(theta) = (실패 총횟수) log(theta) + (시행 수) log(1-theta).
# 관측이 1000개인데 계산에 들어가는 것은 이 두 숫자뿐이다.
# 이런 요약값을 충분통계량이라고 한다.
n_success = train.sum()
n_fail = len(train)

theta_grid = np.linspace(0.01, 0.99, 200)
ll = n_success * np.log(theta_grid) + n_fail * np.log(1 - theta_grid)

# 미분해서 0으로 두면 나오는 닫힌 해. 격자 탐색의 봉우리와 일치해야 한다.
p_hat = train.mean() / (1 + train.mean())

plt.plot(theta_grid, ll, "k-", lw=2)
plt.axvline(p_hat, color="red", linestyle="--", label=f"MLE = {p_hat:.3f}")
plt.axvline(0.12, color="blue", linestyle=":", label="True = 0.120")
plt.xlabel("theta")
plt.ylabel("Log-likelihood")
plt.title("Geometric: Log-Likelihood Surface")
plt.legend()
plt.show()
```

![Geometric: Log-Likelihood Surface](./img/geometric_poisson_mle_80.png)

</div>

## 포아송분포의 MLE

### 모형

포아송분포는 고정된 구간에서 일어나는 사건의 수를 모형화한다:

$$
P(X = k) = \frac{e^{-\lambda}\, \lambda^k}{k!}, \quad k = 0, 1, 2, \ldots
$$

평균과 분산이 모두 $\lambda$이다.

### MLE 유도

관측값 $x_1, \ldots, x_n$이 주어졌을 때 ($\lambda$에 의존하지 않는 상수를 제외한) 로그가능도는:

$$
\ell(\lambda) = \left(\sum_{i=1}^n x_i\right) \log \lambda - n\lambda
$$

미분하면:

$$
\frac{d\ell}{d\lambda} = \frac{\sum x_i}{\lambda} - n = 0
$$

풀면 잘 알려진 결과를 얻는다:

$$
\hat{\lambda}_{\text{MLE}} = \bar{x}
$$

포아송 비율 모수의 MLE는 단순히 표본평균이다.

### 시연

<div class="codebox" markdown>

#### 예제 3. 포아송분포의 MLE { .eg }

```python
from scipy import stats

def poisson_mle_demo(n_train=200, n_test=200, lam_true=4.5):
    """포아송분포의 MLE — 모수적 적합과 비모수적 적합을 견준다."""
    np.random.seed(42)
    train = np.random.poisson(lam_true, n_train)
    test = np.random.poisson(lam_true, n_test)

    lam_hat = train.mean()
    k_max = max(train.max(), test.max()) + 1
    k_vals = np.arange(k_max)

    # 모수적 적합: MLE를 넣은 확률질량함수.
    pmf_param = stats.poisson.pmf(k_vals, lam_hat)

    # 비모수적 적합: 자료의 상대도수를 그대로 쓴다.
    pmf_train = np.bincount(train, minlength=k_max) / n_train
    pmf_test = np.bincount(test, minlength=k_max) / n_test

    # 시험자료에서의 RMSE.
    err_param = np.sqrt(np.mean((pmf_param - pmf_test)**2))
    err_nonparam = np.sqrt(np.mean((pmf_train - pmf_test)**2))

    print(f"True lambda = {lam_true:.2f}")
    print(f"MLE lambda_hat = {lam_hat:.4f}")
    print(f"Test RMSE -- parametric: {err_param:.5f}")
    print(f"Test RMSE -- nonparametric: {err_nonparam:.5f}")

    # 아래 그림에서 다시 쓰도록 계산 결과를 돌려준다
    return k_vals, pmf_param, pmf_train, pmf_test

k_vals, pmf_param, pmf_train, pmf_test = poisson_mle_demo()
```

출력:

```
True lambda = 4.50
MLE lambda_hat = 4.4800
Test RMSE -- parametric: 0.01882
Test RMSE -- nonparametric: 0.03342
```

</div>

### 모수적 적합과 비모수적 적합의 비교

<div class="codebox" markdown>

#### 예제 4. 모수적 적합과 비모수적 적합의 비교 { .eg }

```python
import matplotlib.pyplot as plt

# 위 함수가 돌려준 값을 그대로 쓴다.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 왼쪽: 훈련자료에 대한 적합.
# 막대(모수적 MLE)와 점(경험적 PMF)이 잘 겹친다. 같은 자료로 맞췄으니 당연하다.
axes[0].bar(k_vals, pmf_param, color="white", edgecolor="black",
            lw=1.5, label="Parametric (MLE)")
axes[0].plot(k_vals, pmf_train, "ko", ms=5, label="Empirical (train)")
axes[0].set_title("Poisson: Train Fit")
axes[0].legend()

# 오른쪽: **보지 않은** 시험자료에 대한 적합. 여기가 진짜 시험이다.
# 막대는 왼쪽과 똑같다(모형은 훈련자료로만 맞췄으므로).
# 빨간 점만 새 자료로 바뀌었는데도 막대를 잘 따라간다.
# 반면 경험적 PMF는 훈련자료의 우연한 들쭉날쭉함까지 외웠기 때문에
# 새 자료에서는 오차가 더 크다. 위 출력의 RMSE 두 값이 그 차이다.
axes[1].bar(k_vals, pmf_param, color="white", edgecolor="black",
            lw=1.5, label="Parametric (MLE)")
axes[1].plot(k_vals, pmf_test, "ro", ms=5, label="Empirical (test)")
axes[1].set_title("Poisson: Test Fit")
axes[1].legend()

plt.tight_layout()
plt.show()
```

![Poisson: Train Fit](./img/geometric_poisson_mle_195.png)

</div>

## 해석

- **모수적 모형이 더 잘 일반화된다.** 가정한 모형족이 옳으면(자료가 실제로 기하이나 포아송분포에서 나왔으면) MLE에 기반한 모수적 PMF가 비모수적 경험 PMF보다 검정 자료의 RMSE가 대체로 작다. 모수적 모형은 함수 형태를 통해 모든 $k$ 값에 걸쳐 "힘을 빌려" 온다.
- **비모수적 모형은 더 유연하지만 잡음이 많다.** 경험 PMF는 훈련 자료에서 관측되지 않은 값에 확률 0을 부여한다. 훈련 표본이 작으면(포아송에서 $n = 200$) 이 이산성 효과가 꼬리에서 두드러진다.
- **두 분포 모두 로그가능도 곡면이 오목**하므로 기울기 기반 최적화가 유일한 전역 MLE로 수렴함이 보장된다.
- **모형 설정 오류의 위험.** 참 자료생성과정이 기하이나 포아송이 아니면 모수적 모형이 자료를 체계적으로 잘못 적합할 수 있고, 그때는 비모수적 접근이 더 나을 수 있다.

!!! warning "모수적 가정이 중요하다"
    모수적 적합이 검정 자료에서 더 나은 성능을 보이는 것은 모형이 올바르게 설정되었을 때의 이야기이다. 경험분포보다 모수적 모형을 믿기 전에 언제나 적합도를 확인하라(예: 카이제곱 검정, QQ 그림).

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span> 적률법의 관점에서 기하분포의 MLE를 직접 유도하라. 이 분포에서 MLE와 적률법 추정량이 일치함을 보여라.

</div>

??? success "풀이"
    ($P(X = k) = (1-p)p^k$인) $\text{Geometric}(p)$의 평균은:

    $$
    E[X] = \frac{p}{1 - p}
    $$

    $E[X] = \bar{x}$로 두고 $p$에 대해 풀면:

    $$
    \bar{x} = \frac{p}{1 - p} \implies \bar{x}(1 - p) = p \implies \bar{x} = p(1 + \bar{x})
    $$

    $$
    \hat{p}_{\text{MoM}} = \frac{\bar{x}}{1 + \bar{x}}
    $$

    이는 로그가능도를 최대화하여 얻은 $\hat{p}_{\text{MLE}} = \bar{x}/(1 + \bar{x})$와 동일하다. 단일모수 지수족 분포에서 MLE는 언제나 충분통계량의 함수이며, 적률방정식이 그 통계량만 포함하면 MLE와 적률법이 일치한다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 포아송의 MLE에서 $\hat{\lambda} = \bar{x}$가 임계점일 뿐 아니라 전역 최댓값임을 2계도함수 조건으로 확인하라.

</div>

??? success "풀이"
    로그가능도는:

    $$
    \ell(\lambda) = \left(\sum x_i\right) \log \lambda - n\lambda + C
    $$

    여기서 $C$는 $\lambda$에 의존하지 않는다. 1계도함수는:

    $$
    \frac{d\ell}{d\lambda} = \frac{\sum x_i}{\lambda} - n
    $$

    0으로 두면 $\hat{\lambda} = \bar{x}$이다.

    2계도함수는:

    $$
    \frac{d^2\ell}{d\lambda^2} = -\frac{\sum x_i}{\lambda^2}
    $$

    $\sum x_i \geq 0$이고 $\lambda > 0$이므로 모든 $\lambda > 0$에서 $d^2\ell/d\lambda^2 \leq 0$이다. $\sum x_i > 0$이면 부등호가 엄격하므로 $\hat{\lambda} = \bar{x}$가 전역 최댓값임이 확인된다. (모든 관측값이 0이면 $\hat{\lambda} = 0$이며 이는 경계에서의 최댓값이다.) $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> 포아송분포에서 i.i.d.로 $n = 500$개를 관측했고 표본평균이 $\bar{x} = 3.2$이다. Fisher 정보량을 사용하여 $\lambda$에 대한 근사적인 95% 신뢰구간을 구성하라.

</div>

??? success "풀이"
    포아송 관측값 하나에 대한 Fisher 정보량은:

    $$
    I(\lambda) = \frac{1}{\lambda}
    $$

    $n$개 관측값에서 전체 Fisher 정보량은 $nI(\lambda) = n/\lambda$이다. MLE의 점근정규성에 의해:

    $$
    \hat{\lambda} \dot{\sim} N\!\left(\lambda, \frac{1}{nI(\lambda)}\right) = N\!\left(\lambda, \frac{\lambda}{n}\right)
    $$

    $\hat{\lambda} = 3.2$, $n = 500$을 대입하면:

    $$
    \text{SE} = \sqrt{\frac{\hat{\lambda}}{n}} = \sqrt{\frac{3.2}{500}} = \sqrt{0.0064} = 0.08
    $$

    95% 신뢰구간은:

    $$
    \hat{\lambda} \pm 1.96 \cdot \text{SE} = 3.2 \pm 1.96(0.08) = 3.2 \pm 0.157 = [3.043, 3.357]
    $$

    $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> 모형이 올바르게 설정되었을 때 비모수적 (경험 PMF) 추정량이 불편인데도 모수적 MLE보다 검정 자료의 RMSE가 큰 이유를 설명하라. 편향–분산 맞바꿈은 어떤 역할을 하는가?

</div>

??? success "풀이"
    경험 PMF는 각 확률 $P(X = k)$를 비율 $\hat{p}_k = n_k / n$으로 따로따로 추정한다. 각 $\hat{p}_k$는 불편이고 분산은:

    $$
    \text{Var}(\hat{p}_k) = \frac{P(X = k)(1 - P(X = k))}{n}
    $$

    모수적 MLE는 모수 하나($p$나 $\lambda$)를 추정하고 그로부터 PMF 전체를 유도한다. $P(X = k)$의 모수적 추정량은 (비선형 대입 때문에) 유한표본에서 편향되지만, 각 $k$마다 하나씩이 아니라 자유도 하나만 추정하므로 분산이 훨씬 작다.

    전체 평균제곱오차는 다음과 같이 분해된다:

    $$
    \text{MSE} = \text{Bias}^2 + \text{Variance}
    $$

    비모수적 추정량은 편향이 0이지만 (특히 $n_k$가 작은 꼬리에서) 분산이 크다. 모수적 MLE는 설정이 올바르면 편향이 무시할 만하고, 함수 형태가 PMF의 모양을 제약하므로 분산이 작다. 모수적 모형이 유리한 편향–분산 맞바꿈을 달성하여 검정 자료에서 RMSE가 더 작아진다.

    모형이 잘못 설정되면 모수적 추정량에 사라지지 않는 편향이 생기고, 그때는 비모수적 추정량이 이길 수 있다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> 기하분포에서 나왔다고 믿는 다음 자료를 관측했다고 하자: $x = (0, 2, 1, 0, 3, 1, 0, 0, 1, 2)$. MLE $\hat{p}$를 계산하라. 그다음 $\hat{p}$, $p = 0.3$, $p = 0.7$에서 로그가능도를 계산하여 MLE가 가장 높은 로그가능도를 주는지 확인하라.

</div>

??? success "풀이"
    자료는 $x = (0, 2, 1, 0, 3, 1, 0, 0, 1, 2)$이고 $n = 10$, $\sum x_i = 10$이다.

    MLE는:

    $$
    \hat{p} = \frac{\bar{x}}{1 + \bar{x}} = \frac{1}{1 + 1} = 0.5
    $$

    로그가능도는 $n_s = 10$, $n = 10$일 때 $\ell(p) = n_s \log p + n \log(1 - p)$이다:

    $\hat{p} = 0.5$**에서**:

    $$
    \ell(0.5) = 10 \log(0.5) + 10 \log(0.5) = 20 \log(0.5) = -20 \times 0.6931 = -13.863
    $$

    $p = 0.3$**에서**:

    $$
    \ell(0.3) = 10\log(0.3) + 10\log(0.7) = 10(-1.2040) + 10(-0.3567) = -15.607
    $$

    $p = 0.7$**에서**:

    $$
    \ell(0.7) = 10\log(0.7) + 10\log(0.3) = 10(-0.3567) + 10(-1.2040) = -15.607
    $$

    실제로 $\ell(0.5) > \ell(0.3) = \ell(0.7)$이므로 MLE가 로그가능도를 최대화함이 확인된다. $n_s = n$일 때 로그가능도가 $\hat{p} = 0.5$를 중심으로 대칭이므로 $\ell(0.3) = \ell(0.7)$이라는 대칭성이 나타난다. $\square$

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
같은 계수 자료에 기하분포와 포아송분포를 모두 적합했다. 어느 쪽이 나은지 판단하는 방법을 세 가지 들고, 각각의 장단점을 적어라.

</div>

??? success "풀이"
    두 분포는 **내포 관계가 아니므로** 우도비 검정을 그대로 쓸 수 없다. 다음 세 가지가 표준적인 방법이다.

    **(1) 정보기준 비교.** 모수 개수가 둘 다 1개이므로 벌점이 같고, 결국 **최대 로그가능도를 직접 비교**하면 된다.

    $$
    \text{AIC} = -2\ell(\hat\theta)+2 \quad\Rightarrow\quad \ell_{\text{geom}} \gtrless \ell_{\text{pois}}
    $$

    - **장점**: 간단하고 내포 관계가 필요 없다.
    - **단점**: 차이가 얼마나 커야 의미 있는지에 대한 기준이 없다. 관례적으로 AIC 차이가 2 이상이면 주목할 만하다고 본다.

    **(2) 적합도 검정.** 관측 도수와 각 모형의 기대 도수를 카이제곱으로 비교한다.

    $$
    X^2 = \sum_k \frac{(O_k-E_k)^2}{E_k}, \qquad \text{자유도} = (\text{범주 수}) - 1 - 1
    $$

    - **장점**: 어느 구간에서 어긋나는지 잔차로 볼 수 있다. 두 모형 모두 나쁠 가능성도 잡아낸다.
    - **단점**: 기대도수가 5 미만인 범주를 합쳐야 하고, 그 방식에 따라 결과가 달라진다.

    **(3) 평균-분산 관계 진단.** 포아송은 분산 = 평균, 기하분포(실패 횟수 판본)는 분산 $= \frac{1-p}{p^2}$, 평균 $=\frac{1-p}{p}$이므로

    $$
    \frac{\operatorname{Var}}{E} = \frac1p > 1
    $$

    이다. 표본에서 $s^2/\bar x$를 계산해 1에 가까우면 포아송, 1보다 뚜렷이 크면 기하 쪽을 의심한다.

    - **장점**: 계산이 즉시 되고 해석이 직관적이다. 적합 전에 먼저 해 볼 수 있다.
    - **단점**: 이 비가 크다고 기하분포인 것은 아니다. 음이항이나 영과잉일 수도 있다.

    **권고.** 셋을 함께 쓴다. 진단으로 방향을 잡고, 정보기준으로 고르고, 적합도 검정과 잔차로 고른 모형이 실제로 맞는지 확인한다. **두 모형 중 나은 쪽을 고르는 것과 그 모형이 자료에 맞는 것은 다른 문제다.**

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
기하분포의 MLE $\hat\theta = 1/(1+\bar x)$(실패 횟수 판본)의 Fisher 정보량을 구하고 점근분산을 유도하라. $\theta$가 0 또는 1에 가까울 때 무슨 일이 일어나는가?

</div>

??? success "풀이"
    실패 횟수 판본의 PMF는 $P(X=k) = (1-\theta)^k\theta$이므로

    $$
    \ln f = k\ln(1-\theta) + \ln\theta
    $$

    이고

    $$
    \frac{\partial\ln f}{\partial\theta} = -\frac{k}{1-\theta}+\frac1\theta, \qquad \frac{\partial^2\ln f}{\partial\theta^2} = -\frac{k}{(1-\theta)^2}-\frac{1}{\theta^2}
    $$

    이다. $E[X] = (1-\theta)/\theta$를 넣어 기대값을 취하면

    $$
    I_1(\theta) = \frac{E[X]}{(1-\theta)^2}+\frac{1}{\theta^2} = \frac{1}{\theta(1-\theta)}+\frac{1}{\theta^2} = \frac{\theta+(1-\theta)}{\theta^2(1-\theta)} = \frac{1}{\theta^2(1-\theta)}
    $$

    이다. 따라서

    $$
    \operatorname{Var}(\hat\theta) \approx \frac{\theta^2(1-\theta)}{n}
    $$

    **경계에서의 거동.**

    - **$\theta \to 1$**(거의 언제나 첫 시행에 성공): $I_1 \to \infty$이고 분산이 0으로 간다. 관측값이 대부분 0이므로 $\theta$가 1에 가깝다는 것을 매우 확실히 알 수 있다. 다만 $\hat\theta$가 경계에 붙어 정규근사가 나빠진다. 모든 관측이 0이면 $\hat\theta = 1$로 경계값이 나온다.
    - **$\theta \to 0$**(성공이 매우 드묾): $I_1 \to \infty$이지만 $\operatorname{Var} \approx \theta^2/n \to 0$이다. 절댓값 분산은 작아도 **상대 분산**이

      $$
      \frac{\operatorname{SE}(\hat\theta)}{\theta} \approx \sqrt{\frac{1-\theta}{n}} \to \frac{1}{\sqrt n}
      $$

      으로 $\theta$와 무관하게 남는다. 즉 $\theta$의 **자릿수**를 알아내는 정밀도는 일정하다.

    **실무 권고.** $\theta$가 경계 근처면 로짓 척도 $\ln\{\theta/(1-\theta)\}$에서 추론하는 편이 낫다. 그 척도에서는 로그가능도가 훨씬 이차식에 가깝고 구간이 $(0,1)$을 벗어나지 않는다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
기하분포는 **무기억성**을 갖는 유일한 이산분포다. 이 사실이 계수 자료 모형 선택에서 어떤 신호로 쓰일 수 있는지 설명하라.

</div>

??? success "풀이"
    **무기억성.** $P(X \ge m+n \mid X \ge m) = P(X \ge n)$이다. "이미 $m$번 실패했다"는 사실이 앞으로 몇 번 더 실패할지에 대해 아무 정보도 주지 않는다.

    **위험함수로 보기.** 이산 위험함수를

    $$
    h(k) = P(X = k \mid X \ge k) = \frac{P(X=k)}{P(X\ge k)}
    $$

    로 정의하면, 기하분포에서는

    $$
    h(k) = \frac{(1-\theta)^k\theta}{(1-\theta)^k} = \theta
    $$

    로 **$k$에 무관한 상수**다.

    **진단으로 쓰는 법.** 자료에서 경험적 위험함수

    $$
    \hat h(k) = \frac{(\text{값이 정확히 } k\text{인 개수})}{(\text{값이 } k \text{ 이상인 개수})}
    $$

    를 계산해 $k$에 대해 그린다.

    | 모양 | 시사점 |
    |---|---|
    | 평평함 | 기하분포가 적절 |
    | 감소 | 대기가 길어질수록 성공이 어려워짐. 이질성(개체마다 $\theta$ 다름)의 신호. 베타-기하 혼합 검토 |
    | 증가 | 대기가 길어질수록 성공이 쉬워짐. 학습 효과나 소진 |

    **왜 감소가 이질성의 신호인가.** $\theta$가 개체마다 다르면 **$\theta$가 큰 개체가 먼저 빠져나간다.** 시간이 지날수록 남아 있는 집단은 $\theta$가 작은 개체들로 채워지고, 그 결과 전체 위험이 떨어진다. 개체 수준에서는 무기억성이 성립해도 집단 수준에서는 감소하는 위험이 관측되는 것이다.

    이는 생존분석의 **취약성(frailty)** 개념과 같은 현상이며, 개인 수준의 성질을 집단 수준 자료에서 읽을 때 주의해야 하는 대표적인 사례다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
포아송 모형을 적합했는데 표본분산이 표본평균의 세 배였다. 이를 무시하고 포아송 MLE의 표준오차를 그대로 쓰면 어떤 결과가 생기는가? 정량적으로 답하라.

</div>

??? success "풀이"
    **참 분산.** 실제 자료의 분산이 $3\lambda$라면

    $$
    \operatorname{Var}(\bar X) = \frac{3\lambda}{n}
    $$

    이다.

    **포아송 가정이 주는 값.** 포아송에서는 분산 = 평균이라고 보므로

    $$
    \widehat{\operatorname{Var}}(\bar X) = \frac{\hat\lambda}{n}
    $$

    으로 **참값의 3분의 1**이다.

    **결과.**

    - **표준오차가 $\sqrt3 = 1.73$배 과소평가**된다.
    - 신뢰구간의 폭이 참으로 필요한 것의 58%에 지나지 않는다. 명목 95% 구간의 실제 포함확률이 약 **74%**로 떨어진다($2\Phi(1.96/\sqrt3)-1 = 0.742$).
    - $z$ 통계량이 1.73배 부풀어 오른다. 참 $z$가 1.2인 효과가 2.08로 나와 유의해 보인다. 명목 5% 검정의 실제 제1종 오류율이 약 **26%**다($2\Phi(-1.96/\sqrt3) = 0.258$).

    **고치는 법.**

    - **준포아송.** 산포모수 $\hat\phi = X^2/(n-p)$를 추정해 모든 표준오차에 $\sqrt{\hat\phi}$를 곱한다. 계수 추정값은 그대로다. 여기서는 $\hat\phi \approx 3$이므로 표준오차를 1.73배 키운다.
    - **음이항.** 분산 $\mu + \mu^2/k$를 명시적으로 모형화한다. 준포아송이 분산을 $\phi\mu$(평균에 비례)로 두는 것과 달리 음이항은 이차 관계를 가정하므로, 어느 쪽이 자료에 맞는지 잔차로 확인해야 한다.
    - **강건 표준오차.** 샌드위치 추정량을 쓴다. 분산 구조를 특정하지 않아도 되지만 효율을 잃는다.

    **놓치기 쉬운 이유.** 과대산포가 있어도 계수 추정값은 그대로 나오고 적합도 그림도 그럴듯해 보인다. **표준오차만 조용히 틀린다.** 그래서 포아송 회귀를 적합하면 반드시 $\hat\phi$를 확인하는 것이 기본 절차다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
비모수적 경험 PMF 추정과 모수적 MLE의 차이를 **편향-분산** 관점에서 정리하고, 어느 쪽을 언제 써야 하는지 적어라.

</div>

??? success "풀이"
    **경험 PMF.** $\hat p_k = n_k/n$으로 각 값의 확률을 따로 추정한다.

    - **편향**: 0이다. $E[\hat p_k] = p_k$.
    - **분산**: $p_k(1-p_k)/n$으로 **값마다 독립적으로** 추정하므로, 값의 개수 $K$가 많으면 각 추정값이 적은 관측에 기댄다. 꼬리에서는 $n_k$가 0이나 1이라 $\hat p_k$가 거의 잡음이다.

    **모수적 MLE.** 분포족을 가정하고 모수 하나(또는 몇 개)만 추정한다.

    - **편향**: 모형이 틀리면 있다. 참 분포가 그 족에 없으면 아무리 $n$이 커도 남는다.
    - **분산**: 매우 작다. 모든 관측값이 **한 모수를 함께** 추정하는 데 쓰이므로, 꼬리의 확률도 중앙의 자료에서 얻은 정보로 채워진다.

    **맞바꿈.**

    $$
    \text{MSE} = \text{편향}^2 + \text{분산}
    $$

    에서 모수적 방법은 편향을 감수하고 분산을 크게 줄인다. **모형이 대략 맞으면 이 거래가 압도적으로 유리하다.** 연습문제 5에서 본 대로 경험 PMF가 불편인데도 검정 자료의 RMSE가 더 큰 이유가 이것이다.

    **고르는 기준.**

    | 상황 | 권장 |
    |---|---|
    | $n$이 작고 $K$가 큼 | 모수적 |
    | 꼬리 확률을 추정해야 함 | 모수적(외삽 가능) |
    | $n$이 아주 크고 $K$가 작음 | 비모수적 |
    | 모형이 미덥지 않음 | 비모수적 또는 유연한 모수족 |
    | 관측되지 않은 값의 확률이 필요 | 모수적(경험 PMF는 0을 준다) |

    **절충안도 있다.** 평활을 넣은 비모수 추정(커널 평활, 라플라스 보정), 모수가 여럿인 유연한 족(음이항, 영과잉, 혼합), 모수적 추정을 비모수적으로 보정하는 준모수 방법이 그런 예다. 실무에서는 **모수 모형으로 시작해 잔차로 어긋남을 확인하고, 어긋난 부분만 유연하게 만드는** 순서가 대체로 좋다.

---

## 정리하며

같은 계수 자료에 두 이산분포를 적합해 보고, **모수적 모형과 경험분포**를 비교했다.

- **두 최대가능도추정량 모두 표본평균에서 나온다.** 포아송은 $\hat\lambda=\bar X$ 이고, 기하분포는 $\hat p$ 가 $\bar X$ 의 함수다. 두 분포 모두 지수족이며 $\sum x_i$ 가 충분통계량이다.
- **남겨 둔 검정 자료에서 비교하는 것이 요점이다.** 경험 확률질량함수는 훈련자료를 그대로 외우므로 훈련에서는 완벽하지만, **관측되지 않은 값에 확률 $0$ 을 주어** 새 자료에서 무너진다.
- **모형이 맞으면 모수적 적합이 일반화에서 이긴다.** 분포의 모양을 가정한 덕분에 관측되지 않은 값에도 합리적인 확률을 배정한다. 이것이 모수적 모형의 본질적 이득이다.
- **모형이 틀리면 그 이득이 손해로 바뀐다.** 자료가 과산포되어 있는데 포아송을 쓰면 꼬리를 심하게 과소평가한다. 적합 뒤에 **표본분산과 표본평균을 비교하는 진단**이 반드시 따라야 하는 이유다.
- 1장의 편향–분산 절충이 분포 적합에서 나타난 모습이기도 하다. 모수적 모형은 편향을 들이고 분산을 줄인다.

**이것으로 6.2절이 끝난다.** 가능도의 개념에서 시작해 여러 분포의 최대가능도추정량을 유도하고, 점근 성질과 정보량, 수치 최적화까지 보았다.

다음 절 **적률법의 기초**로 넘어간다. 최대가능도보다 오래되고 계산이 간단한 또 하나의 추정 전략이다.
