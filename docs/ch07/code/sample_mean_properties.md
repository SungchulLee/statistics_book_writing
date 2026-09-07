# 표본평균의 성질

## 개요

표본평균 $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$은 통계학에서 가장 근본적인 추정량이다. 이 페이지에서는 모의실험으로 그 핵심 성질을 확인한다: 여러 분포에서의 불편성, 분산 공식 $\text{Var}(\bar{X}) = \sigma^2/n$, 평균제곱오차, $1/\sqrt{n}$ 속도의 표준오차 수렴, 다른 위치추정량 대비 효율, 그리고 역분산 가중.

## 불편성

표본평균은 바탕 분포와 무관하게 모평균에 대해 불편이다:

$$E[\bar{X}] = \mu$$

기댓값의 선형성에서 따라 나온다. 증명에는 각 $X_i$의 평균이 같은 $\mu$라는 것만 필요하고 분포에 대한 가정은 필요 없다.

다음 모의실험은 100,000번의 반복에서 $\bar{X}$를 계산하고 그 평균이 참 평균에 가까운지 확인하여 여섯 가지 분포에서 이를 검증한다.

```python
import numpy as np

def verify_unbiasedness(mu=10.0, sigma=3.0, n=20, n_sim=100_000, seed=42):
    rng = np.random.default_rng(seed)

    distributions = {
        f'Normal({mu}, {sigma}²)': (lambda: rng.normal(mu, sigma, n), mu),
        'Exp(λ=0.5)':             (lambda: rng.exponential(2, n), 2.0),
        'Poisson(3.7)':           (lambda: rng.poisson(3.7, n), 3.7),
        'Uniform(2, 8)':          (lambda: rng.uniform(2, 8, n), 5.0),
        'Bernoulli(0.4)':         (lambda: rng.binomial(1, 0.4, n), 0.4),
        'Chi²(df=5)':             (lambda: rng.chisquare(5, n), 5.0),
    }

    for name, (sampler, true_mu) in distributions.items():
        estimates = np.array([sampler().mean() for _ in range(n_sim)])
        bias = estimates.mean() - true_mu
        print(f"{name:<22} True μ={true_mu:.4f}  E[X̄]={estimates.mean():.4f}  Bias={bias:.6f}")
```

!!! tip "핵심"
    모든 편향이 (몬테카를로 잡음 범위 안에서) 무시할 만큼 작아, 시험한 모든 분포에서 $E[\bar{X}] = \mu$임이 확인된다.

## 분산과 평균제곱오차

분산이 $\sigma^2$인 i.i.d. 관측값에 대해:

$$\text{Var}(\bar{X}) = \frac{\sigma^2}{n}, \qquad \text{SE}(\bar{X}) = \frac{\sigma}{\sqrt{n}}$$

$\bar{X}$가 불편이므로 평균제곱오차는 분산과 같다:

$$\text{MSE}(\bar{X}) = \text{Bias}^2 + \text{Var}(\bar{X}) = 0 + \frac{\sigma^2}{n} = \frac{\sigma^2}{n}$$

```python
def verify_variance_and_mse(mu=10.0, sigma=3.0, n_sim=100_000, seed=42):
    rng = np.random.default_rng(seed)
    sample_sizes = [5, 10, 25, 50, 100, 500]

    for n in sample_sizes:
        samples = rng.normal(mu, sigma, (n_sim, n))
        x_bars = samples.mean(axis=1)

        var_xbar   = x_bars.var(ddof=0)
        theory_var = sigma**2 / n
        mse        = np.mean((x_bars - mu)**2)
        se         = x_bars.std(ddof=0)
        theory_se  = sigma / np.sqrt(n)

        print(f"n={n:>4}  Var(X̄)={var_xbar:.6f}  σ²/n={theory_var:.6f}  "
              f"MSE={mse:.6f}  SE={se:.6f}  σ/√n={theory_se:.6f}")
```

## 효율 비교

표본평균은 정규 자료에서 가장 효율적인 위치추정량이지만 꼬리가 두꺼운 분포에서는 그렇지 않다. $\bar{X}$ 대비 추정량 $T$의 **상대효율**은:

$$\text{RE}(T, \bar{X}) = \frac{\text{MSE}(\bar{X})}{\text{MSE}(T)}$$

$\text{RE} > 1$이면 대안 $T$가 *더* 효율적이다.

```python
from scipy import stats

def efficiency_comparison(n=30, n_sim=50_000, seed=42):
    rng = np.random.default_rng(seed)

    distributions = {
        'Normal(0,1)':           lambda: rng.standard_normal(n),
        't(df=3)':               lambda: rng.standard_t(3, n),
        'Contaminated Normal':   lambda: np.where(
            rng.uniform(0, 1, n) < 0.1,
            rng.normal(0, 10, n),
            rng.standard_normal(n)),
    }

    for dist_name, sampler in distributions.items():
        est = {'Mean': [], 'Median': [], 'Trim10%': [], 'Trim20%': []}
        for _ in range(n_sim):
            s = sampler()
            est['Mean'].append(np.mean(s))
            est['Median'].append(np.median(s))
            est['Trim10%'].append(stats.trim_mean(s, 0.1))
            est['Trim20%'].append(stats.trim_mean(s, 0.2))

        mse_mean = np.mean(np.array(est['Mean'])**2)
        print(f"\n{dist_name}:")
        for name, vals in est.items():
            mse = np.mean(np.array(vals)**2)
            re  = mse_mean / mse
            print(f"  {name:<12} MSE={mse:.6f}  Rel.Eff.={re:.4f}")
```

!!! note "평균이 지는 경우"
    $t(3)$이나 오염된 정규처럼 꼬리가 두꺼운 분포에서는 절사평균과 중앙값이 표본평균보다 평균제곱오차가 작다. 이상점에 민감한 평균은 이런 상황에서 비효율적이다.

## 역분산 가중평균

관측값의 분산 $\sigma_i^2$이 서로 다를 때 **최적 가중치**는 분산에 반비례한다:

$$w_i = \frac{1/\sigma_i^2}{\sum_{j=1}^k 1/\sigma_j^2}, \qquad \bar{X}_w = \sum_{i=1}^k w_i X_i$$

이 가중치는 기댓값이 참 평균이 되는 모든 가중평균 중에서 $\text{Var}(\bar{X}_w)$를 최소화한다.

```python
def weighted_mean_demo(mu=5.0, n_sim=50_000, seed=42):
    rng = np.random.default_rng(seed)
    sigmas = np.array([1.0, 2.0, 5.0, 10.0, 0.5])

    optimal_weights = 1 / sigmas**2
    optimal_weights /= optimal_weights.sum()

    unweighted, weighted = [], []
    for _ in range(n_sim):
        obs = rng.normal(mu, sigmas)
        unweighted.append(obs.mean())
        weighted.append(np.sum(optimal_weights * obs))

    unweighted = np.array(unweighted)
    weighted   = np.array(weighted)

    print(f"Unweighted:  Var={unweighted.var():.6f}  MSE={np.mean((unweighted-mu)**2):.6f}")
    print(f"IV-Weighted: Var={weighted.var():.6f}  MSE={np.mean((weighted-mu)**2):.6f}")
    print(f"Variance reduction: {(1 - weighted.var()/unweighted.var())*100:.1f}%")
```

## 표준오차의 수렴 속도

로그-로그 그래프에서 표준오차 $\text{SE}(\bar{X}) = \sigma/\sqrt{n}$은 기울기 $-1/2$인 직선으로 나타나며, $O(1/\sqrt{n})$ 수렴 속도를 확인해 준다.

```python
import matplotlib.pyplot as plt

def convergence_rate_plot(mu=5.0, sigma=3.0, n_sim=50_000, seed=42):
    rng = np.random.default_rng(seed)
    sample_sizes = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

    empirical_se = []
    for n in sample_sizes:
        estimates = np.array([rng.normal(mu, sigma, n).mean() for _ in range(n_sim)])
        empirical_se.append(estimates.std())

    theoretical_se = [sigma / np.sqrt(n) for n in sample_sizes]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(sample_sizes, empirical_se, 'bo-', label='Empirical SE')
    ax.loglog(sample_sizes, theoretical_se, 'r--', label='σ/√n', linewidth=2)
    ax.set_xlabel('Sample Size n')
    ax.set_ylabel('Standard Error')
    ax.set_title('Convergence Rate of Sample Mean')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()
```

## 해석

- **불편성**은 분포와 무관하다: 평균이 유한한 임의의 모집단에서 성립한다.
- **분산**은 $1/n$으로 줄어들므로 **표준오차**는 $1/\sqrt{n}$으로 줄어든다. 표준오차를 절반으로 줄이려면 관측값이 네 배 필요하다.
- 편향이 0이므로 **평균제곱오차가 곧 분산**이다. 편향–분산 맞바꿈의 가장 단순한 경우이다.
- **효율**은 모집단의 모양에 달려 있다. 정규 자료에서는 평균이 최적이고, 꼬리가 두꺼운 자료에서는 절사평균과 중앙값의 평균제곱오차가 더 작을 수 있다.
- **역분산 가중**은 정밀도가 다른 관측값을 결합하는 올바른 방법이며, 단순 평균에 비해 분산을 크게 줄일 수 있다.

## 연습문제

**연습문제 1.**
기댓값의 선형성만 써서, 평균이 유한한 임의의 분포에서 $E[\bar{X}] = \mu$임을 해석적으로 증명하라.

??? success "풀이"
    $X_1, \ldots, X_n$이 i.i.d.이고 $E[X_i] = \mu$라 하자. 그러면:

    $$E[\bar{X}] = E\left[\frac{1}{n}\sum_{i=1}^n X_i\right] = \frac{1}{n}\sum_{i=1}^n E[X_i] = \frac{1}{n} \cdot n\mu = \mu$$

    첫 등호는 $\bar{X}$의 정의, 둘째는 기댓값의 선형성, 셋째는 모든 $i$에 대해 $E[X_i] = \mu$라는 사실을 쓴다. $\square$

---

**연습문제 2.**
i.i.d. 관측값에 대해 $\text{Var}(\bar{X}) = \sigma^2/n$임을 보여라. 그다음 표준오차를 절반으로 줄이려면 왜 표본크기를 네 배로 해야 하는지 설명하라.

??? success "풀이"
    $\text{Var}(X_i) = \sigma^2$인 i.i.d. $X_1, \ldots, X_n$에 대해:

    $$\text{Var}(\bar{X}) = \text{Var}\left(\frac{1}{n}\sum_{i=1}^n X_i\right) = \frac{1}{n^2}\sum_{i=1}^n \text{Var}(X_i) = \frac{1}{n^2}\cdot n\sigma^2 = \frac{\sigma^2}{n}$$

    두 번째 단계는 독립성을 쓴다(합의 분산이 분산의 합이 된다). 표준오차는 $\text{SE} = \sigma/\sqrt{n}$이다. 이를 절반으로 줄이려면:

    $$\frac{\sigma}{\sqrt{n'}} = \frac{1}{2}\cdot\frac{\sigma}{\sqrt{n}} \implies \sqrt{n'} = 2\sqrt{n} \implies n' = 4n$$

    $\square$

---

**연습문제 3.**
표준편차가 $\sigma_1 = 1, \sigma_2 = 2, \sigma_3 = 5, \sigma_4 = 10, \sigma_5 = 0.5$인 출처에서 얻은 관측값 다섯 개를 생각하자. 최적 역분산 가중치와 가중평균의 분산을 계산하라. 가중하지 않은 평균의 분산과 비교하라.

??? success "풀이"
    정규화하지 않은 가중치는 $w_i^* = 1/\sigma_i^2$이다:

    $$w_1^* = 1, \quad w_2^* = 0.25, \quad w_3^* = 0.04, \quad w_4^* = 0.01, \quad w_5^* = 4$$

    합은 $W = 1 + 0.25 + 0.04 + 0.01 + 4 = 5.3$이다. 정규화한 가중치는 $w_i = w_i^*/W$이다.

    가중평균의 분산은:

    $$\text{Var}(\bar{X}_w) = \sum_{i=1}^5 w_i^2 \sigma_i^2 = \frac{1}{W^2}\sum_{i=1}^5 \frac{\sigma_i^2}{\sigma_i^4} = \frac{1}{W^2}\sum_{i=1}^5 \frac{1}{\sigma_i^2} = \frac{W}{W^2} = \frac{1}{W} = \frac{1}{5.3} \approx 0.1887$$

    가중하지 않은 평균의 분산은:

    $$\text{Var}(\bar{X}) = \frac{1}{25}\sum_{i=1}^5 \sigma_i^2 = \frac{1 + 4 + 25 + 100 + 0.25}{25} = \frac{130.25}{25} = 5.21$$

    역분산 가중평균의 분산이 약 $5.21/0.189 \approx 27.6$배 작다. $\square$

---

**연습문제 4.**
정규모집단에서 표본평균은 Cramer-Rao 하한 $\sigma^2/n$을 달성한다. 표본평균 대비 표본중앙값의 점근 상대효율이 $2/\pi \approx 0.637$임을 보여라.

??? success "풀이"
    $X_i \sim N(\mu, \sigma^2)$에서 표본평균의 분산은 $\sigma^2/n$이다. 표본중앙값 $\tilde{X}$의 점근분산은:

    $$\text{Var}(\tilde{X}) \approx \frac{1}{4n[f(\mu)]^2}$$

    여기서 $f$는 모집단 밀도이다. 정규분포에서 $f(\mu) = \frac{1}{\sigma\sqrt{2\pi}}$이므로:

    $$\text{Var}(\tilde{X}) \approx \frac{1}{4n \cdot \frac{1}{2\pi\sigma^2}} = \frac{2\pi\sigma^2}{4n} = \frac{\pi\sigma^2}{2n}$$

    점근 상대효율은:

    $$\text{ARE}(\tilde{X}, \bar{X}) = \frac{\text{Var}(\bar{X})}{\text{Var}(\tilde{X})} = \frac{\sigma^2/n}{\pi\sigma^2/(2n)} = \frac{2}{\pi} \approx 0.637$$

    모집단이 실제로 정규일 때 중앙값은 평균에 비해 자료의 약 36%를 "낭비"한다는 뜻이다. $\square$

---

**연습문제 5.**
$X_1 \sim N(\mu, 1)$과 $X_2 \sim N(\mu, 9)$를 독립적으로 관측한다고 하자. 분산을 최소화하는 가중추정량 $\hat{\mu} = aX_1 + bX_2$($a + b = 1$)를 구하라. 그 분산은 얼마인가?

??? success "풀이"
    $b = 1 - a$로 두면 분산은:

    $$\text{Var}(\hat{\mu}) = a^2 \cdot 1 + (1-a)^2 \cdot 9 = a^2 + 9(1-a)^2$$

    미분하여 0으로 놓으면:

    $$\frac{d}{da}\left[a^2 + 9(1-a)^2\right] = 2a - 18(1-a) = 2a - 18 + 18a = 20a - 18 = 0$$

    따라서 $a = 9/10$, $b = 1/10$이다. 이는 역분산 가중치와 일치한다: $w_1 \propto 1/1 = 1$, $w_2 \propto 1/9$를 정규화하면 $(9/10, 1/10)$이다.

    최소 분산은:

    $$\text{Var}(\hat{\mu}) = \left(\frac{9}{10}\right)^2 + 9\left(\frac{1}{10}\right)^2 = \frac{81}{100} + \frac{9}{100} = \frac{90}{100} = \frac{9}{10}$$

    가중하지 않은 평균과 비교하면 $\text{Var}\!\left(\frac{X_1+X_2}{2}\right) = \frac{1+9}{4} = 2.5$로 훨씬 크다. $\square$
