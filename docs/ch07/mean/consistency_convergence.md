# 일치성과 수렴

## 개요

일치성이란 표본크기가 커질 때 추정량이 참 모수값으로 수렴한다는 뜻이다. 표본평균에서는 모집단의 평균이 유한하면 대수의법칙이 이를 보장한다. 중심극한정리는 나아가 이 수렴의 속도와 모양을 기술한다. 이 페이지에서는 이런 수렴 성질을 보이고, 수렴이 실패하는 경우(코시분포)를 살펴보며, 자기상관이나 금융에서의 추정 기간 같은 실무적 문제를 다룬다.

## 강대수의법칙

**강대수의법칙(SLLN)**은 $E[|X|] < \infty$인 i.i.d. 관측값에 대해 다음을 말한다:

$$\bar{X}_n \xrightarrow{\text{a.s.}} \mu \quad (n \to \infty)$$

즉 확률 1로 누적평균이 $\mu$로 수렴한다. 다음 모의실험은 네 가지 분포에서 독립적인 20개 수열의 누적평균을 그린다.

<div class="codebox" markdown>

### 예제 1. 여러 경로로 보는 강대수의 법칙 { .eg }

```python
import numpy as np
import matplotlib.pyplot as plt

def consistency_visualization(seed=42):
    """표본크기를 키워 가며 표본평균이 참 평균으로 수렴하는 경로를 그린다.

    네 모집단은 모양이 저마다 다르지만 강대수의 법칙은 유한한 평균만
    요구하므로 넷 다 같은 결론에 이른다.
    """
    rng = np.random.default_rng(seed)
    N = 10_000
    n_runs = 20

    distributions = {
        'Normal(5, 9)':    (lambda: rng.normal(5, 3, N), 5.0),
        'Exp(λ=0.5)':      (lambda: rng.exponential(2, N), 2.0),
        'Uniform(0, 10)':  (lambda: rng.uniform(0, 10, N), 5.0),
        'Chi²(df=5)':      (lambda: rng.chisquare(5, N), 5.0),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (name, (sampler, true_mu)) in zip(axes.flat, distributions.items()):
        # 같은 분포에서 20개의 **경로**를 그린다.
        # 큰수의 법칙은 "평균이 참값에 가까워진다"가 아니라
        # "거의 모든 경로가 참값으로 수렴한다"를 말하므로,
        # 한 경로가 아니라 여러 경로를 겹쳐 그려야 그 뜻이 드러난다.
        for _ in range(n_runs):
            data = sampler()
            # cumsum을 1, 2, 3, ... 으로 나누면 각 시점까지의 평균이 된다
            running_mean = np.cumsum(data) / np.arange(1, N + 1)
            ax.plot(running_mean, alpha=0.2, linewidth=0.5)
        ax.axhline(true_mu, color='red', linestyle='--', linewidth=2,
                   label=f'μ = {true_mu}')
        # 가로축을 로그로 둔다. n=1..100 구간의 극심한 흔들림과
        # n=1000 이후의 안정을 한 화면에 담으려면 로그가 필요하다.
        ax.set_xscale('log')
        ax.set_xlabel('n')
        ax.set_ylabel('X̄ₙ')
        ax.set_title(f'{name}: SLLN')
        ax.legend()
    plt.tight_layout()
    plt.show()
consistency_visualization()
```

![일치성과 수렴](./img/consistency_convergence_15.png)

</div>

!!! tip "그림에서 보이는 양상"
    모집단 분포와 무관하게, $n$이 커지면 20개의 표본경로가 모두 빨간 점선($\mu$)으로 수렴한다. 강대수의법칙이 작동하는 모습이다.

## 중심극한정리

**중심극한정리(CLT)**는 큰 $n$에서 $\bar{X}_n$의 분포를 기술한다:

$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} N(0, 1)$$

동등하게, 큰 $n$에 대해 $\bar{X}_n \approx N(\mu, \sigma^2/n)$이다. 분산이 유한한 **모든** 모집단 분포에서 성립한다.

<div class="codebox" markdown>

### 예제 2. 모집단 넷으로 보는 중심극한정리 { .eg }

```python
from scipy import stats

def clt_demonstration(seed=42):
    """모집단 넷과 표본크기 셋을 격자로 놓고 중심극한정리를 확인한다.

    대수의 법칙이 X-bar 가 어디로 가는지를 말한다면, 중심극한정리는 그
    주변에서 어떤 모양으로 흩어지는지를 말한다.
    """
    rng = np.random.default_rng(seed)
    n_sim = 20_000

    populations = {
        'Normal(5, 4)':     (lambda n: rng.normal(5, 2, n), 5.0, 4.0),
        'Exp(λ=0.5)':       (lambda n: rng.exponential(2, n), 2.0, 4.0),
        'Uniform(0, 10)':   (lambda n: rng.uniform(0, 10, n), 5.0, 100/12),
        'Bernoulli(0.3)':   (lambda n: rng.binomial(1, 0.3, n), 0.3, 0.21),
    }

    sample_sizes = [2, 5, 30]
    fig, axes = plt.subplots(len(populations), len(sample_sizes), figsize=(15, 12))

    # 행 = 모집단(넷), 열 = 표본 크기(2, 5, 30).
    # 가로로 읽으면 "n이 커지면 종 모양이 된다",
    # 세로로 읽으면 "모집단이 달라도 결과가 같다"가 보인다.
    # 다만 치우침이 심한 모집단일수록 정규가 되는 데 더 큰 n이 필요하다.
    # 베르누이(0.3) 행에서 n=2, 5 가 여전히 이산적인 것이 그 예다.
    for i, (pop_name, (sampler, mu, sigma2)) in enumerate(populations.items()):
        for j, n in enumerate(sample_sizes):
            x_bars = np.array([sampler(n).mean() for _ in range(n_sim)])
            ax = axes[i, j]
            ax.hist(x_bars, bins=60, density=True, alpha=0.6, color='steelblue')
            x = np.linspace(x_bars.min(), x_bars.max(), 200)
            # 참 모수로 계산한 이론적 표준오차. 표본에서 추정한 값이 아니다.
            # 붉은 곡선이 히스토그램과 얼마나 맞는지가 곧 근사의 품질이다.
            se = np.sqrt(sigma2 / n)
            ax.plot(x, stats.norm.pdf(x, mu, se), 'r-', linewidth=2)
            if i == 0:
                ax.set_title(f'n = {n}')
            if j == 0:
                ax.set_ylabel(pop_name)
    plt.suptitle('Central Limit Theorem')
    plt.tight_layout()
    plt.show()
clt_demonstration()
```

![Central Limit Theorem](./img/consistency_convergence_60.png)

</div>

!!! note "정규성으로의 수렴 속도"
    대칭인 분포(Normal, Uniform)는 정규성에 빨리 도달한다. 치우친 분포(Exponential, $p$가 0.5에서 먼 Bernoulli)는 더 큰 $n$이 필요하다. $n = 30$쯤이면 대부분의 분포에서 정규근사가 충분하다.

## 코시분포: 수렴의 실패

**코시분포**의 밀도는 $f(x) = \frac{1}{\pi(1 + x^2)}$이고 평균이 유한하지 않다($E[|X|] = \infty$). 그 결과 표본평균이 **수렴하지 않는다**:

$$\bar{X}_n \sim \text{Cauchy}(0, 1) \quad \text{모든 } n \text{에 대해}$$

코시 관측값을 더 많이 평균해도 전혀 나아지지 않는다.

<div class="codebox" markdown>

### 예제 3. 코시분포에서 무너지는 수렴 { .eg }

```python
def cauchy_failure(seed=42):
    """평균이 없는 분포에서는 대수의 법칙이 무너짐을 보인다.

    코시분포는 E[|X|] 가 무한이라 법칙의 전제부터 성립하지 않는다.
    표본을 아무리 늘려도 표본평균은 자리를 잡지 못한다.
    """
    rng = np.random.default_rng(seed)
    N = 10_000
    n_runs = 10

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 왼쪽: 정규분포. 경로들이 빠르게 0 으로 모여 붙는다.
    ax = axes[0]
    for _ in range(n_runs):
        data = rng.standard_normal(N)
        running_mean = np.cumsum(data) / np.arange(1, N + 1)
        ax.plot(running_mean, alpha=0.4, linewidth=0.7)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_ylim(-2, 2)
    ax.set_title('Normal: Converges')

    # 오른쪽: 코시분포. 잠잠하다가도 큰 값 하나가 나오면 평균이 통째로 튄다.
    # 이미 쌓인 n 개의 평균을 관측값 하나가 끌고 갈 만큼 꼬리가 두껍다.
    ax = axes[1]
    for _ in range(n_runs):
        data = rng.standard_cauchy(N)
        running_mean = np.cumsum(data) / np.arange(1, N + 1)
        ax.plot(running_mean, alpha=0.4, linewidth=0.7)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_title('Cauchy: Does NOT Converge')

    plt.suptitle('Consistency Failure: Cauchy (E[|X|] = ∞)')
    plt.tight_layout()
    plt.show()
cauchy_failure()
```

![Normal: Converges](./img/consistency_convergence_106.png)

</div>

!!! warning "대수의법칙에는 유한한 평균이 필요하다"
    코시의 표본평균은 $n$이 아무리 커도 불규칙하게 떠돈다. 그러나 표본 **중앙값**은 유한한 평균을 요구하지 않으므로 코시 위치모수에 대해 일치한다.

## 자기상관의 영향

관측값이 종속이면(예: AR(1) 과정 $X_t = \rho X_{t-1} + \epsilon_t$) 표본평균의 분산은 더 이상 $\sigma^2/n$이 아니다. 보정인자는 대략:

$$\text{Var}(\bar{X}) \approx \frac{\sigma^2}{n} \cdot \frac{1 + \rho}{1 - \rho}$$

양의 자기상관은 분산을 **부풀리고**, 음의 자기상관은 **줄인다**.

<div class="codebox" markdown>

### 예제 4. 자기상관이 표준오차에 미치는 영향 { .eg }

```python
def autocorrelation_effect(n=100, n_sim=30_000, seed=42):
    """관측값이 서로 독립이 아니면 sigma^2/n 공식이 얼마나 빗나가는지 본다.

    시계열 자료는 이웃한 값끼리 닮아 있다. 그 정도를 rho 로 조절해 가며
    표본평균의 실제 분산을 독립일 때의 값과 견준다.
    """
    rng = np.random.default_rng(seed)
    sigma = 1.0
    rho_values = [-0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 0.95]

    for rho in rho_values:
        x_bars = []
        # 잡음의 크기를 이렇게 잡아야 x 의 주변분산이 rho 와 무관하게 sigma^2 로
        # 유지된다. 그래야 달라진 것이 오직 상관뿐이라고 말할 수 있다.
        innov_sig = sigma * np.sqrt(max(1 - rho**2, 0.01))
        for _ in range(n_sim):
            # AR(1) 과정: 오늘 값은 어제 값의 rho 배에 새 잡음을 더한 것이다.
            x = np.zeros(n)
            x[0] = rng.normal(0, sigma)
            for t in range(1, n):
                x[t] = rho * x[t - 1] + rng.normal(0, innov_sig)
            x_bars.append(x.mean())

        # 비가 1 보다 크면 독립을 가정한 표준오차가 실제보다 작다는 뜻이다.
        # 곧 신뢰구간이 실제보다 좁게, 검정이 실제보다 후하게 나온다.
        var_emp = np.var(x_bars)
        var_iid = sigma**2 / n
        ratio = var_emp / var_iid
        print(f"ρ={rho:>5.2f}  Var(X̄)={var_emp:.6f}  σ²/n={var_iid:.6f}  Ratio={ratio:.2f}")
autocorrelation_effect()
```

출력:

```
ρ=-0.50  Var(X̄)=0.003387  σ²/n=0.010000  Ratio=0.34
ρ=-0.20  Var(X̄)=0.006796  σ²/n=0.010000  Ratio=0.68
ρ= 0.00  Var(X̄)=0.010030  σ²/n=0.010000  Ratio=1.00
ρ= 0.20  Var(X̄)=0.015012  σ²/n=0.010000  Ratio=1.50
ρ= 0.50  Var(X̄)=0.029373  σ²/n=0.010000  Ratio=2.94
ρ= 0.80  Var(X̄)=0.086491  σ²/n=0.010000  Ratio=8.65
ρ= 0.95  Var(X̄)=0.311792  σ²/n=0.010000  Ratio=31.18
```

</div>

!!! danger "금융 시계열"
    금융 수익률은 변동성에 양의 자기상관을 보이는 경우가 많다(수익률 자체에도 약한 자기상관이 있을 때가 있다). 이 종속성을 무시하면 표본평균의 불확실성을 낮춰 잡게 되어 신뢰구간이 너무 좁아지고 가설검정이 너무 관대해진다.

## 양의 수익률을 감지하기 위한 추정 기간

금융의 근본적인 난제: 기대초과수익률이 양수임을 감지하려면 몇 년치 자료가 필요한가? 주식 프리미엄이 3%이고 연간 변동성이 20%일 때, $T$년 후 $\mu > r_f$라고 올바르게 결론지을 확률은:

$$P(\bar{X}_T > r_f) = \mathcal{N}\left(\frac{\mu - r_f}{\sigma / \sqrt{T}}\right)$$

<div class="codebox" markdown>

### 예제 5. 초과수익을 확인하는 데 필요한 기간 { .eg }

```python
def estimation_horizon_analysis(seed=42):
    """주식이 무위험자산보다 낫다는 것을 확인하려면 몇 년치 자료가 필요한가.

    수익률의 표준오차는 sigma/sqrt(T) 로 줄지만 sigma 가 워낙 커서 잘 줄지
    않는다. 그래서 "장기적으로 주식이 낫다"는 말은 한 사람의 투자 기간
    안에서는 확인하기 어려운 주장이다.
    """
    mu_annual = 0.06     # 기대수익률 연 6%
    sigma_annual = 0.20  # 변동성 연 20%
    rf = 0.03            # 무위험이자율 연 3%

    # T 년치를 모았을 때 초과수익이 양수로 관측될 확률. 검정력에 해당한다.
    years = np.arange(1, 101)
    prob_detect = [stats.norm.cdf((mu_annual - rf) / (sigma_annual / np.sqrt(T)))
                   for T in years]

    for target in [0.80, 0.90, 0.95]:
        idx = np.argmax(np.array(prob_detect) >= target)
        print(f"  {target*100:.0f}% power: ~{years[idx]} years of data needed")
estimation_horizon_analysis()
```

출력:

```
  80% power: ~32 years of data needed
  90% power: ~73 years of data needed
  95% power: ~1 years of data needed
```

</div>

## 해석

- **일치성**은 표본평균이 결국 $\mu$에 가까워짐을 보장하지만 수렴 속도 $1/\sqrt{n}$은 느리다.
- **중심극한정리**는 표본분포의 모양(근사적으로 정규)을 주어, 모집단이 정규가 아니어도 추론을 가능하게 한다.
- 코시 예는 **대수의법칙과 중심극한정리에 유한한 적률이 필요하다**는 것을 극명하게 상기시킨다. 평균이 유한하지 않으면 표본평균은 쓸모가 없다.
- **자기상관**은 실제 자료에서 흔하며 유효표본크기를 크게 부풀리거나 줄일 수 있다.
- 금융에서는 기대수익률의 **신호 대 잡음 비**가 워낙 나빠서 실력과 운을 구분하려면 수십 년치 자료가 필요하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
약대수의법칙(WLLN)을 진술하고 강대수의법칙(SLLN)과 어떻게 다른지 설명하라. 각각에 필요한 최소한의 적률 조건은 무엇인가?

</div>

??? success "풀이"
    **WLLN:** $E[X_i] = \mu$이고 $\text{Var}(X_i) = \sigma^2 < \infty$인 i.i.d. $X_1, X_2, \ldots$에 대해:

    $$\bar{X}_n \xrightarrow{P} \mu \quad \text{(확률수렴)}$$

    즉 모든 $\epsilon > 0$에 대해 $n \to \infty$일 때 $P(|\bar{X}_n - \mu| > \epsilon) \to 0$이다.

    **SLLN:** 더 약한 조건 $E[|X_i|] < \infty$(1차 적률만 유한하고 분산은 필요 없음) 아래에서:

    $$\bar{X}_n \xrightarrow{\text{a.s.}} \mu \quad \text{(거의 확실한 수렴)}$$

    거의 확실한 수렴은 $P(\lim_{n\to\infty} \bar{X}_n = \mu) = 1$을 뜻한다.

    SLLN이 더 강하다: 거의 확실한 수렴은 확률수렴을 함의하지만 역은 성립하지 않는다. SLLN에는 $E[|X|] < \infty$만 필요한 반면, Chebyshev 부등식을 통한 WLLN의 간단한 증명에는 유한한 분산이 필요하다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
i.i.d. 코시 확률변수의 $\bar{X}_n$이 코시 관측값 하나와 같은 분포를 가짐을 보여라. (힌트: 특성함수를 쓰라.)

</div>

??? success "풀이"
    표준 코시 확률변수의 특성함수는 $\varphi_X(t) = e^{-|t|}$이다.

    i.i.d. Cauchy $X_1, \ldots, X_n$에 대해 $S_n = \sum X_i$의 특성함수는:

    $$\varphi_{S_n}(t) = \left(e^{-|t|}\right)^n = e^{-n|t|}$$

    $\bar{X}_n = S_n/n$의 특성함수는:

    $$\varphi_{\bar{X}_n}(t) = \varphi_{S_n}(t/n) = e^{-n|t/n|} = e^{-|t|}$$

    이는 정확히 표준 코시의 특성함수이다. 특성함수가 분포를 유일하게 결정하므로 모든 $n$에 대해 $\bar{X}_n \sim \text{Cauchy}(0,1)$이다.

    즉 표본평균이 관측값 하나보다 조금도 더 집중되어 있지 않다 — 평균을 내는 일이 무의미하다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
$|\rho| < 1$이고 $\epsilon_t \sim N(0, \sigma_\epsilon^2)$인 AR(1) 과정 $X_t = \rho X_{t-1} + \epsilon_t$에서 $\bar{X}_n$의 분산을 유도하고, 큰 $n$에서 $\frac{\sigma^2}{n}\cdot\frac{1+\rho}{1-\rho}$($\sigma^2 = \sigma_\epsilon^2/(1-\rho^2)$)에 근사함을 보여라.

</div>

??? success "풀이"
    정상 분산은 $\gamma_0 = \text{Var}(X_t) = \sigma_\epsilon^2/(1-\rho^2)$이다. 시차 $h$에서의 자기공분산은 $\gamma_h = \gamma_0 \rho^{|h|}$이다.

    $$\text{Var}(\bar{X}_n) = \frac{1}{n^2}\sum_{i=1}^n\sum_{j=1}^n \text{Cov}(X_i, X_j) = \frac{1}{n^2}\sum_{i=1}^n\sum_{j=1}^n \gamma_0 \rho^{|i-j|}$$

    $$= \frac{\gamma_0}{n^2}\left(n + 2\sum_{h=1}^{n-1}(n-h)\rho^h\right)$$

    큰 $n$에서 $\sum_{h=1}^{n-1}(n-h)\rho^h \approx n\sum_{h=1}^{\infty}\rho^h = \frac{n\rho}{1-\rho}$이다. 따라서:

    $$\text{Var}(\bar{X}_n) \approx \frac{\gamma_0}{n}\left(1 + \frac{2\rho}{1-\rho}\right) = \frac{\gamma_0}{n}\cdot\frac{1+\rho}{1-\rho}$$

    $\gamma_0 = \sigma^2$이므로 $\text{Var}(\bar{X}_n) \approx \frac{\sigma^2}{n}\cdot\frac{1+\rho}{1-\rho}$를 얻는다.

    $\rho = 0.8$이면 팽창인자가 $1.8/0.2 = 9$이므로 유효표본크기는 $n/9$에 불과하다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
어떤 펀드의 참 연간 기대초과수익률이 3%, 연간 변동성이 20%이다. 표본평균 초과수익률이 양수일 확률이 90%를 넘으려면 몇 년치 자료가 필요한가?

</div>

??? success "풀이"
    $\mu = 0.03$, $\sigma = 0.20$인 $\bar{X}_T \sim N(\mu, \sigma^2/T)$에서 $P(\bar{X}_T > 0) \geq 0.90$이 필요하다.

    $$P(\bar{X}_T > 0) = P\left(Z > \frac{-\mu}{\sigma/\sqrt{T}}\right) = \mathcal{N}\left(\frac{\mu\sqrt{T}}{\sigma}\right) \geq 0.90$$

    $\mathcal{N}^{-1}(0.90) = 1.282$이므로:

    $$\frac{0.03\sqrt{T}}{0.20} \geq 1.282 \implies \sqrt{T} \geq \frac{1.282 \times 0.20}{0.03} = 8.547 \implies T \geq 73.1$$

    따라서 약 **74년**치 자료가 필요하다. 금융에서 운용자의 실력과 운을 구분하는 일이 왜 그토록 어려운지를 극명하게 보여준다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
중심극한정리가 코시분포에 적용되지 않는 이유를 설명하라. 그렇다면 코시 표본평균에는 아무런 극한정리도 적용되지 않는가?

</div>

??? success "풀이"
    중심극한정리는 $\text{Var}(X) < \infty$를 요구한다. 코시분포는 분산이 유한하지 않으므로(사실 평균도 유한하지 않다) 중심극한정리가 적용되지 않는다.

    그러나 다른 극한정리는 적용된다. (안정분포에 대한) **일반화 중심극한정리**에 의해, i.i.d. 코시 변수의 정규화된 부분합은 코시분포로 수렴한다. 사실 이 결과는 정확하다: 모든 유한한 $n$에서 $\bar{X}_n$이 관측값 하나와 같은 코시분포를 갖는다.

    더 넓게 보면 코시분포는 안정지수 $\alpha = 1$인 **안정분포**족에 속한다. 지수 $\alpha < 2$인 안정분포에서는 중심극한정리가 실패하지만 일반화 중심극한정리가 비Gaussian 안정법칙으로의 수렴을 준다. 정규은 $\alpha = 2$인 특수한 경우이다. $\square$

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
**베리-에세인 정리**를 서술하고, 중심극한정리의 근사오차가 표본크기와 모집단 왜도에 어떻게 의존하는지 정량화하라.

</div>

??? success "풀이"
    **정리.** $E|X-\mu|^3 = \rho < \infty$이면 모든 $x$에서

    $$
    \left|P\!\left(\frac{\bar X_n-\mu}{\sigma/\sqrt n}\le x\right)-\Phi(x)\right| \le \frac{C\rho}{\sigma^3\sqrt n}
    $$

    이고 $C \le 0.4748$이다(현재 알려진 최선의 상수).

    **읽는 법.**

    - **오차가 $O(n^{-1/2})$**이다. 중심극한정리가 성립한다는 사실만으로는 $n$이 얼마여야 하는지 알 수 없는데, 이 정리가 그 속도를 준다.
    - **$\rho/\sigma^3$이 표준화된 3차 절대적률**이며, 왜도와 밀접하다. 대칭분포에서도 0이 아니지만(절댓값이므로), 치우친 분포에서 훨씬 크다.

    **수치 예.** 지수분포는 $\sigma=1/\lambda$이고 $E|X-\mu|^3 = 2.415/\lambda^3$이므로 $\rho/\sigma^3 = 2.415$다.

    | $n$ | 오차 상한 |
    |---|---|
    | 25 | 0.229 |
    | 100 | 0.115 |
    | 400 | 0.057 |
    | 10000 | 0.011 |

    **$n=100$에서도 상한이 0.115다.** 양쪽 꼬리에 각각 적용되므로 명목 5% 양측검정의 실제 오류율이 최악의 경우 28%까지 갈 수 있다는 뜻이다.

    **상한은 상한일 뿐.** 실제 오차는 대개 이보다 훨씬 작다. 그러나 **어느 $x$에서 최악이 되는지 모르므로** 보수적으로 읽어야 하고, 특히 꼬리에서 상대오차가 크다는 점이 중요하다. $\Phi(x)=0.001$인 영역에서 절대오차 0.01은 상대오차 1000%다.

    **함의.** "$n\ge30$" 같은 규칙은 **왜도를 전혀 보지 않는다.** 베리-에세인의 형태를 보면 필요한 $n$이 $(\rho/\sigma^3)^2$에 비례해야 하므로, 치우친 분포에서는 훨씬 커야 한다. 앞서 본 $n \ge (\text{왜도}/0.1)^2$ 규칙과 같은 이야기다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
시계열의 표본평균이 언제 수렴하는가? **에르고딕 정리**를 서술하고, 정상성만으로는 부족한 경우를 예로 들어라.

</div>

??? success "풀이"
    **에르고딕 정리(버코프).** $\{X_t\}$가 정상이고 **에르고딕**이며 $E|X_t|<\infty$이면

    $$
    \bar X_n \xrightarrow{\text{a.s.}} E[X_1]
    $$

    이다. 독립을 요구하지 않는다는 점이 큰수의 법칙과의 차이다.

    **에르고딕성이란.** 대략 "시간 평균이 공간 평균과 같다"는 성질이다. 형식적으로는 이동에 불변인 사건의 확률이 0 또는 1이어야 한다. **한 경로가 충분히 오래 관측되면 전체 분포를 훑는다**는 뜻이다.

    **정상성만으로는 부족한 예.**

    $$
    X_t = Z, \qquad Z\sim N(0,1)\ \text{한 번 뽑아 영원히 고정}
    $$

    - **정상이다.** 모든 $t$에서 $X_t\sim N(0,1)$이고 결합분포가 시간 이동에 불변이다.
    - **그러나 $\bar X_n = Z$로 $n$과 무관하다.** $E[X_1]=0$으로 수렴하지 않고 $Z$에 머문다.
    - **에르고딕이 아니다.** $\{Z>0\}$이 이동 불변 사건이고 확률이 1/2이다.

    직관적으로, 이 과정은 **경로 하나가 분포 전체를 훑지 못한다.** 한 번 뽑힌 $Z$에 영원히 갇힌다.

    **현실의 대응물.**

    - **확률적 추세.** 단위근 과정 $X_t = X_{t-1}+\varepsilon_t$는 정상도 에르고딕도 아니며, 표본평균이 수렴하지 않는다.
    - **체제 전환.** 흡수 상태가 있는 마르코프 연쇄. 한 체제에 갇히면 다른 체제를 영원히 보지 못한다.
    - **개체 간 이질성.** 각 개체가 고유한 수준을 갖고 그 수준이 변하지 않으면, 한 개체를 오래 관측해도 모집단 평균을 알 수 없다. 패널자료에서 고정효과를 다뤄야 하는 이유다.

    **실무 진단.** 앞서 본 **누적평균 그림**이 가장 유용하다. 수렴하지 않고 표류하면 에르고딕성이 의심스럽다. 자료를 전반부·후반부로 나눠 평균을 비교하는 것도 간단한 점검이다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
꼬리가 두꺼운 분포에서 표본평균의 수렴 속도가 어떻게 달라지는지, 파레토 $\alpha$로 나누어 정리하라.

</div>

??? success "풀이"
    파레토 $P(X>x)=x^{-\alpha}$($x\ge1$)를 기준으로 본다.

    | $\alpha$ | 평균 | 분산 | 수렴 여부 | 속도 | 극한분포 |
    |---|---|---|---|---|---|
    | $>2$ | 유한 | 유한 | ✓ | $n^{-1/2}$ | 정규 |
    | $(1,2]$ | 유한 | **무한** | ✓ | $n^{1-1/\alpha}$ | 안정분포 |
    | $\le1$ | **무한** | 무한 | ✗ | — | — |

    **$\alpha\in(1,2)$의 경우가 미묘하다.** 큰수의 법칙은 성립해 $\bar X_n\to\mu$이지만, 중심극한정리가 성립하지 않는다.

    $$
    n^{1-1/\alpha}(\bar X_n-\mu)\xrightarrow{d} S_\alpha
    $$

    으로 극한이 **지수 $\alpha$인 안정분포**다. $\alpha=1.5$면 속도가 $n^{1/3}$으로, 표준 $n^{1/2}$보다 훨씬 느리다.

    **실무적 귀결.**

    - **표준오차 $s/\sqrt n$이 무의미하다.** $s$ 자체가 수렴하지 않고 $n$과 함께 커진다.
    - **신뢰구간과 검정을 쓸 수 없다.** 극한분포가 정규가 아니고 꼬리가 두꺼우므로 $\pm1.96\operatorname{SE}$가 성립하지 않는다.
    - **부트스트랩도 실패한다.** 표준 부트스트랩이 일치하지 않으며, $m$-out-of-$n$ 같은 변형이 필요하다.

    **진단.**

    1. **로그-로그 생존함수 그림.** $\ln\hat S(x)$를 $\ln x$에 대해 그려 직선이면 거듭제곱 꼬리이고 기울기가 $-\alpha$다.
    2. **힐 추정량.** 상위 $k$개 관측값으로 $\alpha$를 추정한다. $k$에 대한 민감도를 힐 그림으로 확인한다.
    3. **누적 $s$ 그림.** 표본표준편차가 $n$과 함께 안정되는지 본다.

    **대처.** $\alpha\le2$로 판단되면 **평균 기반 추론을 포기**한다. 중앙값이나 분위수는 $\alpha$와 무관하게 $\sqrt n$ 속도로 정규수렴하므로 안전하다. 꼬리 자체가 관심이면 극단값 이론의 도구(일반화파레토 적합)를 쓴다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
**마팅게일 차분**에 대한 큰수의 법칙을 서술하고, 독립이 아닌데도 표본평균이 수렴하는 구조를 설명하라.

</div>

??? success "풀이"
    **마팅게일 차분 수열.** $\{D_t\}$가 필터레이션 $\{\mathcal{F}_t\}$에 대해

    $$
    E[D_t\mid\mathcal{F}_{t-1}] = 0
    $$

    을 만족하면 마팅게일 차분이라 한다. 독립보다 훨씬 약한 조건이다. **과거를 알아도 다음 값의 조건부 평균이 0**이면 된다.

    **큰수의 법칙.** $\sum_t E[D_t^2]/t^2 < \infty$이면

    $$
    \frac1n\sum_{t=1}^n D_t \xrightarrow{\text{a.s.}} 0
    $$

    이다. 분산이 유계이면 이 조건은 자동으로 만족된다.

    **중심극한정리도 성립한다.** 조건부 분산이 안정되고 린데베르그 조건이 성립하면

    $$
    \frac{1}{\sqrt n}\sum_t D_t \xrightarrow{d} N(0,\sigma^2)
    $$

    이다.

    **왜 독립이 아닌데 되는가.** 핵심은 **조건부 평균이 0**이라는 것이다. 그러면 서로 다른 시점의 항들이 **무상관**이 된다.

    $$
    E[D_sD_t] = E\left[D_s\,E[D_t\mid\mathcal{F}_{t-1}]\right] = 0 \quad (s<t)
    $$

    분산이 더해지는 데 필요한 것은 독립이 아니라 무상관이며, 마팅게일 구조가 그것을 보장한다.

    **어디에 나타나는가.**

    - **최대가능도의 점수함수.** 시계열 모형에서 점수 $s_t = \partial\ln f(x_t\mid\mathcal{F}_{t-1})/\partial\theta$가 마팅게일 차분이다. 그래서 시계열 MLE의 점근이론이 i.i.d.와 비슷한 모양을 갖는다.
    - **회귀 오차.** $E[\varepsilon_t\mid\mathcal{F}_{t-1}]=0$이면 최소제곱의 일치성이 따라 나온다. 완전 외생성보다 약한 조건이다.
    - **효율적 시장 가설.** 초과수익률이 마팅게일 차분이라는 것이 그 형식적 표현이다. 예측 가능한 부분이 없다는 뜻이다.
    - **확률적 경사법.** 갱신의 잡음이 마팅게일 차분이며, 그래서 수렴이 보장된다.

    **중요한 단서.** 마팅게일 차분은 **무상관이지만 독립은 아니다.** $D_t$의 **분산**이 과거에 의존할 수 있으며, 이것이 바로 GARCH 같은 변동성 군집 모형이다. 수익률이 예측 불가능하면서도 변동성은 예측 가능한 현상이 이 구조로 설명된다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
$\bar X_n$의 수렴을 실무에서 판정하는 절차를 정리하라. 어떤 그림과 수치를 보고, 어떤 결과이면 평균 기반 추론을 포기해야 하는가?

</div>

??? success "풀이"
    **절차.**

    1. **누적평균 그림.** $\bar x_k$를 $k$에 대해 그리고 $\pm2s/\sqrt k$ 띠를 얹는다. 띠 안에서 좁아지면 정상이다.

    2. **누적표준편차 그림.** $s_k$가 안정되는지 본다. **계속 커지면 2차 적률이 없다는 강한 신호**이며, 이 경우 $s/\sqrt n$ 자체가 무의미하다.

    3. **꼬리 진단.** 로그-로그 생존함수 그림의 기울기로 $\alpha$를 어림한다. $\hat\alpha\le2$이면 경고, $\le1$이면 평균이 아예 없다.

    4. **영향력 점검.** 상위 1%를 빼고 평균을 다시 계산한다. 크게 달라지면 소수가 지배하고 있다.

    5. **자기상관 점검.** ACF를 그린다. 유의한 자기상관이 있으면 유효 표본크기를 계산해 표준오차를 보정한다.

    6. **블록 비교.** 자료를 앞뒤로 나눠 평균을 비교한다. 크게 다르면 정상성이 의심스럽다.

    **포기해야 하는 경우.**

    | 관측 | 판단 | 대안 |
    |---|---|---|
    | $s_k$가 수렴하지 않음 | 분산 무한 | 중앙값, 분위수 |
    | 누적평균이 계단처럼 뜀 | 꼬리 극도로 두꺼움 | 절사평균, 강건 추정 |
    | 누적평균이 표류 | 비정상 | 차분, 국소 추정, 체제 모형 |
    | $\hat\alpha \le 1$ | 평균 자체가 없음 | 중앙값이 유일한 선택 |
    | 상위 1% 제거로 평균이 두 배 변함 | 소수가 지배 | 원인 확인 후 강건 방법 |

    **핵심 메시지.** **표본평균과 그 표준오차를 계산하는 것은 언제나 가능하지만, 그것이 의미를 갖는지는 별개의 문제다.** 위 진단은 각각 몇 줄이면 되며, 계산 비용에 견주어 얻는 안전이 크다. 특히 금융 수익률, 보험 청구액, 네트워크 트래픽, 소득처럼 꼬리가 두꺼운 것으로 알려진 자료에서는 습관으로 삼아야 한다.

---

## 정리하며

수렴을 **경로로 그려 보면** 정리가 말하는 바가 분명해진다.

- **강한 큰수의 법칙.** $\mathbb{E}|X|<\infty$ 이면 $\bar X_n\to\mu$ 가 거의 확실하게 성립한다. 누적평균 곡선 여러 개를 겹쳐 그리면 모두 한 값으로 빨려 들어가는 모습이 보인다.
- **중심극한정리가 속도와 모양을 더한다.** 큰수의 법칙이 "어디로"를 말하고, 중심극한정리가 "$\sqrt n$ 로 확대하면 정규"라고 말한다. 같은 현상의 두 배율이다.
- **코시분포에서 그림이 달라진다.** 누적평균이 정착하지 않고 이따금 크게 튄다. 평균이 정의되지 않으므로 수렴할 대상 자체가 없다.
- **자기상관이 있으면 느려진다.** 독립일 때의 $\sigma^2/n$ 이 $\frac{\sigma^2}{n}\cdot\frac{1+\phi}{1-\phi}$ 로 부풀며, 유효표본크기가 $n(1-\phi)/(1+\phi)$ 로 줄어든다. **금융 시계열에서 "관측 $n$ 개"가 곧 "정보 $n$ 개"가 아닌 이유다.**
- **추정 기간의 딜레마.** 길게 잡으면 표본이 늘어 표준오차가 줄지만 모수가 변했을 위험이 커진다. **편향과 분산의 절충이 기간 선택으로 나타난 것**이며, 이 장 뒤의 수익률 추정에서 다시 만난다.

다음 절 **로버스트 추정량 비교**로 7.1절을 마무리한다.
