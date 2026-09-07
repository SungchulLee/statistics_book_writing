# 베리–에센 정리

## 개요

중심극한정리는 표준화된 표본평균이 정규분포로 수렴함을 보장하지만 그 수렴이 **얼마나 빠른지**에 대해서는 아무 말도 하지 않는다. **베리–에센 정리**는 유한한 표본 크기 $n$에 대해 근사 오차의 명시적인 상한을 제공하여 이 빈틈을 메운다.

---

## 진술

$X_1, X_2, \ldots, X_n$이 다음을 만족하는 i.i.d. 확률변수라 하자.

- 평균 $\mu = E[X_i]$
- 분산 $\sigma^2 = \text{Var}(X_i) > 0$
- 유한한 3차 절대적률 $\rho = E\left[|X_i - \mu|^3\right] < \infty$

$F_n(x) = P\left(\frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \leq x\right)$을 표준화된 표본평균의 누적분포함수라 하고 $\mathcal{N}(x)$를 표준정규분포의 누적분포함수라 하자. 그러면

$$
\sup_{x \in \mathbb{R}} \left|F_n(x) - \mathcal{N}(x)\right| \leq \frac{C \cdot \rho}{\sigma^3 \sqrt{n}}
$$

이며, 여기서 $C$는 보편 상수다. 알려진 최선의 값은 $C \leq 0.4748$이다(Shevtsova, 2011).

---

## 해석

이 정리는 **비점근적** 보장을 제공한다. 임의의 유한한 $n$에 대해 정규근사의 최대 오차가 $O(1/\sqrt{n})$으로 유계다. 핵심 함의는 다음과 같다.

- 근사 오차가 $1/\sqrt{n}$의 속도로 줄어든다.
- 3차 적률이 큰 분포(왜도가 크거나 꼬리가 두꺼운 분포)는 더 느리게 수렴한다.
- 비 $\rho / \sigma^3$이 원래 분포의 "비정규성"을 포착한다.

---

## 중심극한정리와의 관계

| 측면 | 중심극한정리 | 베리–에센 |
|:---|:---|:---|
| **진술** | $n \to \infty$일 때 $F_n(x) \to \mathcal{N}(x)$ | $\|F_n - \mathcal{N}\|_\infty \leq C\rho / (\sigma^3\sqrt{n})$ |
| **결과의 유형** | 점근적 | 비점근적(유한한 $n$) |
| **수렴 속도** | 명시하지 않음 | $O(1/\sqrt{n})$ |
| **가정** | 유한한 $\mu, \sigma^2$ | 유한한 $\mu, \sigma^2, \rho$ |

베리–에센 정리는 중심극한정리가 질적으로만 주장하는 바를 **정량화**한다.

---

## 예제

### 예: 공정한 동전 던지기

$X_i \sim \text{Bernoulli}(0.5)$에 대해 $\mu = 0.5$, $\sigma^2 = 0.25$, $\rho = E[|X_i - 0.5|^3] = 0.125$이다.

$$
\text{Bound} = \frac{0.4748 \times 0.125}{0.25^{3/2} \sqrt{n}} = \frac{0.4748}{n^{1/2}}
$$

$n = 100$이면 한계 $\approx 0.0475$이며, 이는 누적분포함수가 모든 점에서 정규 누적분포함수와 4.75% 이내라는 뜻이다.

### 예: 지수분포

$X_i \sim \text{Exponential}(1)$에 대해 $\mu = 1$, $\sigma^2 = 1$, $\rho = E[|X_i - 1|^3] = 2 + e^{-1} \approx 2.368$이다.

$$
\text{Bound} = \frac{0.4748 \times 2.368}{\sqrt{n}} \approx \frac{1.124}{\sqrt{n}}
$$

$n = 100$이면 한계 $\approx 0.112$이다. 한계가 더 큰 것은 지수분포의 왜도를 반영한다. 대칭인 베르누이보다 정규성으로 더 느리게 수렴한다.

---

## 파이썬으로 살펴보기

```python
import numpy as np
from scipy import stats

def berry_esseen_bound(sigma, rho, n, C=0.4748):
    """Compute the Berry-Esseen upper bound."""
    return C * rho / (sigma**3 * np.sqrt(n))

# Bernoulli(0.5)
sigma_b = np.sqrt(0.25)
rho_b = 0.125
print("=== Bernoulli(0.5) ===")
for n in [10, 30, 100, 1000]:
    bound = berry_esseen_bound(sigma_b, rho_b, n)
    print(f"n = {n:5d}: Berry–Esseen bound = {bound:.4f}")

print()

# Exponential(1)
sigma_e = 1.0
rho_e = 2.368
print("=== Exponential(1) ===")
for n in [10, 30, 100, 1000]:
    bound = berry_esseen_bound(sigma_e, rho_e, n)
    print(f"n = {n:5d}: Berry–Esseen bound = {bound:.4f}")
```

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def berry_esseen_visualization(dist_name, rvs_fn, mu, sigma, rho, sample_sizes):
    """Compare the actual CDF error with the Berry-Esseen bound."""
    C = 0.4748
    x_grid = np.linspace(-4, 4, 1000)
    n_sim = 50_000

    fig, axes = plt.subplots(1, len(sample_sizes), figsize=(12, 3),
                             sharey=True)
    fig.suptitle(f'Berry–Esseen: {dist_name}', fontsize=14)

    for ax, n in zip(axes, sample_sizes):
        np.random.seed(42)
        # Simulate standardized sample means
        samples = rvs_fn(size=(n_sim, n))
        x_bar = samples.mean(axis=1)
        z = (x_bar - mu) / (sigma / np.sqrt(n))

        # Empirical CDF vs normal CDF
        ecdf = np.array([np.mean(z <= x) for x in x_grid])
        ncdf = stats.norm.cdf(x_grid)
        actual_error = np.abs(ecdf - ncdf)
        max_error = actual_error.max()

        bound = C * rho / (sigma**3 * np.sqrt(n))

        ax.plot(x_grid, actual_error, lw=1.5, label=f'Actual max: {max_error:.4f}')
        ax.axhline(bound, color='r', linestyle='--', lw=1.5,
                   label=f'BE bound: {bound:.4f}')
        ax.set_title(f'n = {n}')
        ax.set_xlabel('x')
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)

    axes[0].set_ylabel('|Fₙ(x) − Φ(x)|')
    plt.tight_layout()
    plt.show()

# Exponential(1): skewed distribution
berry_esseen_visualization(
    'Exponential(1)',
    lambda size: np.random.exponential(1, size),
    mu=1.0, sigma=1.0, rho=2.368,
    sample_sizes=[5, 30, 100]
)
```

```python
import numpy as np
import matplotlib.pyplot as plt

def convergence_rate_comparison():
    """Compare convergence rates for different distributions."""
    C = 0.4748
    ns = np.arange(5, 501)

    distributions = {
        'Bernoulli(0.5)': {'sigma': np.sqrt(0.25), 'rho': 0.125},
        'Uniform(0,1)':   {'sigma': 1/np.sqrt(12), 'rho': 1/32},
        'Exponential(1)': {'sigma': 1.0,            'rho': 2.368},
    }

    fig, ax = plt.subplots(figsize=(12, 4))
    for name, params in distributions.items():
        bounds = C * params['rho'] / (params['sigma']**3 * np.sqrt(ns))
        ax.plot(ns, bounds, label=name, lw=2)

    ax.set_xlabel('Sample Size n')
    ax.set_ylabel('Berry–Esseen Bound')
    ax.set_title('Convergence Rate to Normal: Berry–Esseen Bounds')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()

convergence_rate_comparison()
```

---

## 핵심 요약

- 베리–에센 정리는 정규근사 오차에 대한 **유한표본** 한계 $O(1/\sqrt{n})$을 준다.
- 이 한계는 $\rho / \sigma^3$에 의존한다. 왜도가 크거나 꼬리가 두꺼운 분포일수록 더 느리게 수렴한다.
- 원하는 근사 정확도를 위해 "$n$이 얼마나 커야 하는가?"에 답함으로써 중심극한정리를 보완한다.
- 대칭 분포(예: Bernoulli(0.5))가 치우친 분포(예: 지수분포)보다 빠르게 수렴한다.

## 연습문제

**연습문제 1.**
베리–에센 정리는 $C \leq 0.4748$일 때 $\sup_x |F_n(x) - \mathcal{N}(x)| \leq \frac{C \rho}{\sigma^3 \sqrt{n}}$이라고 말한다. Bernoulli(0.5) 분포에서는 $\sigma^2 = 0.25$이고 $\rho = E[|X - \mu|^3] = 0.125$이다. 베리–에센 한계가 근사 오차 0.01 이하를 보장하려면 $n$이 얼마나 커야 하는가?

??? success "풀이"
    다음이 필요하다.

    $$
    \frac{C \rho}{\sigma^3 \sqrt{n}} \leq 0.01
    $$

    $C = 0.4748$, $\rho = 0.125$, $\sigma = 0.5$를 넣으면

    $$
    \frac{0.4748 \times 0.125}{0.5^3 \sqrt{n}} \leq 0.01
    $$

    $$
    \frac{0.05935}{0.125 \sqrt{n}} \leq 0.01
    $$

    $$
    \frac{0.4748}{\sqrt{n}} \leq 0.01
    $$

    $$
    \sqrt{n} \geq 47.48 \implies n \geq 2254.3
    $$

    이다. 따라서 Bernoulli(0.5)의 경우 정규근사 오차가 0.01 이하임을 보장하려면 $n \geq 2255$이면 충분하다.

---

**연습문제 2.**
$\mu = 1$, $\sigma = 1$, $\rho = E[|X-1|^3] \approx 2.368$인 Exponential(1) 분포를 생각하자. $n = 30$에서의 베리–에센 한계를 같은 표본 크기의 Bernoulli(0.5)와 비교하라. 어느 분포가 정규분포로 더 빨리 수렴하며 그 이유는 무엇인가?

??? success "풀이"
    **Bernoulli(0.5):** $\rho = 0.125$, $\sigma = 0.5$이므로

    $$
    \text{Bound} = \frac{0.4748 \times 0.125}{0.5^3 \sqrt{30}} = \frac{0.05935}{0.125 \times 5.477} = \frac{0.05935}{0.6847} \approx 0.0867
    $$

    **Exponential(1):** $\rho = 2.368$, $\sigma = 1$이므로

    $$
    \text{Bound} = \frac{0.4748 \times 2.368}{1^3 \sqrt{30}} = \frac{1.1243}{5.477} \approx 0.2053
    $$

    Bernoulli(0.5)의 한계(0.087)가 Exponential(1)의 한계(0.205)보다 훨씬 작다. Bernoulli(0.5)는 대칭이라($\rho/\sigma^3$이 작아서) 더 빨리 수렴하는 반면, Exponential(1)은 오른쪽으로 치우쳐 있고 $\sigma^3$에 비해 3차 절대적률이 크다.

---

**연습문제 3.**
중심극한정리가 이미 정규분포로의 수렴을 보장하는데도 베리–에센 정리가 필요한 이유를 설명하라.

??? success "풀이"
    중심극한정리는 **점근적** 결과다. $n \to \infty$일 때 $\bar{X}_n$이 정규분포로 분포수렴한다고 말할 뿐, 유한한 표본 크기 $n$에서 근사가 얼마나 좋은지에 대해서는 아무 말도 하지 않는다. 실무에서는 언제나 유한한 표본을 다루므로 $n = 30$이나 $n = 100$이나 $n = 10{,}000$이 "충분히 큰지"를 알아야 한다.

    베리–에센 정리는 정규근사의 최대 오차에 대한 명시적인 유한표본 **한계**를 제공하여 이 빈틈을 메운다. "내 특정 분포와 표본 크기에서 중심극한정리 근사는 얼마나 정확한가?"라는 실용적 질문에 답한다. 정규근사가 순진하게 짐작하는 것보다 훨씬 큰 표본을 요구할 수 있는, 치우쳤거나 꼬리가 두꺼운 분포에서 특히 중요하다.

---

**연습문제 4.**
베리–에센 한계는 $O(1/\sqrt{n})$으로 줄어든다. 근사 오차를 10분의 1로 줄이려면(예: 0.1에서 0.01로) 표본 크기를 몇 배로 늘려야 하는가?

??? success "풀이"
    한계가 $1/\sqrt{n}$에 비례하므로 한계를 10분의 1로 줄이려면

    $$
    \frac{1}{\sqrt{n_{\text{new}}}} = \frac{1}{10} \cdot \frac{1}{\sqrt{n_{\text{old}}}}
    $$

    $$
    \sqrt{n_{\text{new}}} = 10 \sqrt{n_{\text{old}}}
    $$

    $$
    n_{\text{new}} = 100 \cdot n_{\text{old}}
    $$

    이어야 한다. 근사 오차를 10분의 1로 줄이려면 표본 크기를 **100배**로 늘려야 한다. 이 $O(1/\sqrt{n})$의 수렴 속도는 비교적 느리며, 특히 치우친 분포에서 정확한 정규근사에 때때로 아주 큰 표본이 필요한 이유를 설명해 준다.

---

**연습문제 5.**
**에지워스 전개**가 중심극한정리를 2차까지 정밀화함을 보여라. $\gamma_1$이 왜도일 때 $\sqrt n (\bar X_n - \mu)/\sigma$의 누적분포함수를 $\Phi(x) + (\gamma_1/(6\sqrt n)) \phi(x) (1 - x^2) + O(1/n)$으로 근사할 수 있다. 이것이 순수한 중심극한정리보다 왜 더 정확한 근사인지 설명하라.

??? success "풀이"
    순수한 중심극한정리는 누적분포함수를 $\Phi(x)$로 근사하며 오차가 $O(1/\sqrt n)$, 즉 베리–에센의 속도다.

    **에지워스 전개**는 모집단 분포의 왜도 $\gamma_1$에 비례하는 보정항을 더한다.

    $$
    P\!\left(\frac{\sqrt n (\bar X_n - \mu)}{\sigma} \le x\right) = \Phi(x) - \frac{\gamma_1}{6\sqrt n}(x^2 - 1)\phi(x) + O(1/n)
    $$

    이렇게 하면 잔여 오차가 $O(1/\sqrt n)$에서 $O(1/n)$으로 줄어들어 한 자릿수만큼 더 촘촘해진다. 그다음 항들은 첨도($\gamma_2$)와 더 높은 누율을 포함한다.

    **실용적 쓰임:** 심하게 치우친 자료에서 $n$이 중간 정도일 때 에지워스 보정이 순수한 가우시안 근사보다 상당히 정확할 수 있다. 통계 소프트웨어의 여러 고정밀 근사가 에지워스나 관련된 안장점 보정을 사용한다.

    **단서:** 에지워스 전개는 꼬리에서 음의 "확률밀도"를 낼 수 있으므로 타당한 영역(대개 $|x| \le 2$ 정도) 밖에서 쓰면 안 된다.

---

**연습문제 6.**
**i.i.d.가 아닌** 합에 대한 베리–에센. 독립이지만 동일분포는 아닌 확률변수에 대한 일반화를 진술하라. 이것이 회귀와 시계열 분석에 왜 중요한가?

??? success "풀이"
    **일반화된 베리–에센:** $\mathbb{E}[X_i] = 0$, $\mathrm{Var}(X_i) = \sigma_i^2$, $\mathbb{E}[|X_i|^3] = \rho_i$, $B_n^2 = \sum_i \sigma_i^2$인 독립(동일분포일 필요는 없음) $X_i$에 대해

    $$
    \sup_x \left| P\!\left(\frac{S_n}{B_n} \le x\right) - \Phi(x) \right| \le \frac{C \sum_i \rho_i}{B_n^3}
    $$

    이다. 이 한계는 합의 표준편차의 세제곱 대비 3차 적률의 *합*에 의존한다. 어느 한 $X_i$의 $\rho_i$가 불균형하게 크면 한계가 느슨해지는데, 이는 큰 항 하나가 가우시안 수렴을 막을 수 있음을 반영한다.

    **응용통계에서의 중요성:**

    - **회귀**: 오차 $\varepsilon_i$의 분산이 서로 다를 수 있으므로(이분산성) $\hat\beta$에 대한 정규분포 기반 신뢰구간을 정당화하려면 i.i.d.가 아닌 중심극한정리가 필요하다.
    - **시계열**: 약한 종속(혼합) 열은 적절한 보정을 거쳐 중심극한정리를 만족한다.
    - **표본조사**: 층화표본은 분산이 서로 다른 층들의 독립적인 기여를 섞는다.

    현대 점근통계학은 i.i.d. 가정이 정확히 성립하는 일이 드문 실제 자료 상황에서 가우시안 중심극한정리를 정당화하기 위해 이런 일반화에 의존한다.
