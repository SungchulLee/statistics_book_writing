# X-bar의 표본분포 (Bernoulli)

## 개요

모집단이 이항 결과(성공/실패)로 이루어져 있으면 표본평균 $\bar{X}$는 표본에서 성공이 차지하는 비율, 즉 표본비율 $\hat{p}$과 같다. 이 페이지에서는 성공확률이 서로 다른 Bernoulli 모집단에서 뽑은 $\hat{p}$의 표본분포를 살펴본다. 중심극한정리에 의해 $n$이 크면 $\hat{p}$이 근사적으로 정규분포이며, 이를 모의실험으로 확인한다.

## 모집단 모형

각 관측값은 성공확률이 $p$인 Bernoulli 시행이다:

$$
X_i \sim \text{Bernoulli}(p), \qquad P(X_i = 1) = p, \quad P(X_i = 0) = 1 - p
$$

모평균과 모분산은:

$$
\mu = E[X_i] = p, \qquad \sigma^2 = \text{Var}(X_i) = p(1 - p)
$$

## 표본비율

크기 $n$인 표본에 대해 표본비율은:

$$
\hat{p} = \bar{X} = \frac{1}{n}\sum_{i=1}^n X_i
$$

그 표본분포는 다음을 만족한다:

$$
E[\hat{p}] = p, \qquad \text{Var}(\hat{p}) = \frac{p(1 - p)}{n}
$$

$\hat{p}$의 표준오차는:

$$
\text{SE}(\hat{p}) = \sqrt{\frac{p(1 - p)}{n}}
$$

## 정규근사

중심극한정리에 의해 $n$이 충분히 크면:

$$
\hat{p} \;\dot{\sim}\; N\!\left(p,\; \frac{p(1 - p)}{n}\right)
$$

!!! tip "경험 법칙"
    $\hat{p}$에 대한 정규근사는 대체로 $np \ge 10$이고 $n(1-p) \ge 10$일 때 믿을 만하다고 본다. 이 조건이 분포가 지나치게 치우치지 않도록 보장한다.

## 모의실험

다음 코드는 여러 $p$ 값에 대해 Bernoulli 모집단에서 크기 $n = 100$인 표본을 뽑아 $\hat{p}$의 표본분포를 모의실험하고 이론적 정규근사를 겹쳐 그린다.

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

np.random.seed(1)

n_population = 10_000
n_sample = 100
n_sim = 1_000
p_values = [0.4, 0.5, 0.6, 0.7]

fig, axes = plt.subplots(1, len(p_values), figsize=(14, 3.5))

# p를 0.4에서 0.7까지 바꿔 가며 네 패널을 그린다.
# 모집단은 0과 1뿐인 가장 비정규적인 분포인데도
# 표본비율의 표집분포는 어느 p에서나 종 모양이 된다.
for ax, p in zip(axes, p_values):
    population = stats.binom(n=1, p=p).rvs(n_population, random_state=1)

    # 0/1 자료의 평균이 곧 비율이므로 p-hat 은 표본평균의 한 경우다.
    p_hat_sims = np.array([
        np.random.choice(population, size=n_sample, replace=False).mean()
        for _ in range(n_sim)
    ])

    # Histogram of simulated values
    _, bins, _ = ax.hist(p_hat_sims, density=True, bins=15,
                         alpha=0.5, edgecolor="white",
                         label=r"simulated $\hat{p}$")

    # 정규근사를 겹쳐 그린다.
    # 베르누이의 분산이 p(1-p) 이므로 표준오차는 sqrt(p(1-p)/n) 이다.
    # 이 값은 p = 0.5 에서 최대가 되고 0이나 1에 가까울수록 작아진다.
    # 네 패널의 폭이 조금씩 다른 이유가 그것이다.
    se = np.sqrt(p * (1 - p) / n_sample)
    x_grid = np.linspace(bins[0], bins[-1], 200)
    pdf = stats.norm(loc=p, scale=se).pdf(x_grid)
    ax.plot(x_grid, pdf, "--r", lw=2, alpha=0.7, label="Normal approx.")
    ax.set_title(f"p = {p}")
    ax.set_xlabel(r"$\hat{p}$")

axes[0].set_ylabel("Density")
axes[-1].legend(fontsize=8)
plt.tight_layout()
plt.show()
```

## 해석

!!! note "주요 관찰"

    1. 네 가지 $p$ 값($0.4, 0.5, 0.6, 0.7$) 모두에서 $n = 100$일 때 모의실험한 $\hat{p}$의 표본분포가 정규근사와 잘 맞는다.
    2. 분포는 $p = 0.5$에서 가장 대칭이며(분산이 최대), $p$가 0이나 1로 갈수록 약간 더 치우친다.
    3. 퍼짐은 $p$에 따라 달라진다. 표준오차 $\sqrt{p(1-p)/n}$은 $p = 0.5$에서 최대이고 $p$가 $0.5$에서 멀어질수록 작아진다.
    4. $n = 100$이면 네 값 모두에서 $np \ge 10$이고 $n(1-p) \ge 10$이라는 경험 법칙이 충족되므로 정규근사가 잘 작동할 것으로 기대된다.

### 모비율에 따른 표준오차

| $p$ | $\text{SE}(\hat{p})$ |
|---|---|
| 0.4 | $\sqrt{0.24 / 100} = 0.0490$ |
| 0.5 | $\sqrt{0.25 / 100} = 0.0500$ |
| 0.6 | $\sqrt{0.24 / 100} = 0.0490$ |
| 0.7 | $\sqrt{0.21 / 100} = 0.0458$ |

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** $\text{Var}(X_i) = p(1-p)$에서 출발하여 정의로부터 $\hat{p}$의 분산을 유도하라.

</div>

??? success "풀이"
    $\hat{p} = \frac{1}{n}\sum_{i=1}^n X_i$이고 $X_i$들이 독립이므로:

    $$
    \text{Var}(\hat{p}) = \text{Var}\!\left(\frac{1}{n}\sum_{i=1}^n X_i\right) = \frac{1}{n^2} \sum_{i=1}^n \text{Var}(X_i) = \frac{1}{n^2} \cdot n \cdot p(1-p) = \frac{p(1-p)}{n}
    $$

    $\square$

<div class="drillbox" markdown>

**연습문제 2.** 어떤 여론조사가 유권자 $n = 400$명을 조사했다. 특정 후보를 지지하는 표본비율이 $\hat{p} = 0.53$이다. 참 비율 $p$에 대한 95% 신뢰구간을 구성하라.

</div>

??? success "풀이"
    정규근사를 사용하면 95% 신뢰구간은:

    $$
    \hat{p} \pm z_{0.025} \cdot \text{SE}(\hat{p})
    $$

    추정된 표준오차는:

    $$
    \widehat{\text{SE}} = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}} = \sqrt{\frac{0.53 \times 0.47}{400}} = \sqrt{\frac{0.2491}{400}} \approx 0.02495
    $$

    $z_{0.025} = 1.96$이므로:

    $$
    0.53 \pm 1.96 \times 0.02495 = 0.53 \pm 0.0489
    $$

    95% 신뢰구간은 약 $(0.481, 0.579)$이다. 이 구간이 0.5를 포함하므로 95% 수준에서 이 후보가 과반의 지지를 받는다고 결론지을 수 없다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** $p(1-p)$가 $p = 0.5$에서 최대이고 그 값이 $1/4$임을 보여라. 이것이 $\hat{p}$의 "최악의 경우" 표준오차가 $1/(2\sqrt{n})$임을 뜻하는 이유를 설명하라.

</div>

??? success "풀이"
    $p \in [0, 1]$에서 $g(p) = p(1-p) = p - p^2$이라 하자.

    $$
    g'(p) = 1 - 2p = 0 \implies p = \frac{1}{2}
    $$

    $g''(p) = -2 < 0$이므로 최대점이다. 최댓값은:

    $$
    g\!\left(\frac{1}{2}\right) = \frac{1}{2} \cdot \frac{1}{2} = \frac{1}{4}
    $$

    따라서 표준오차는 다음을 만족한다:

    $$
    \text{SE}(\hat{p}) = \sqrt{\frac{p(1-p)}{n}} \le \sqrt{\frac{1/4}{n}} = \frac{1}{2\sqrt{n}}
    $$

    이 상한은 표본크기를 계획할 때 유용하다. 미지의 $p$가 무엇이든 표준오차는 결코 $1/(2\sqrt{n})$을 넘지 않는다. 예를 들어 $\text{SE} \le 0.03$을 보장하려면 $n \ge 1/(4 \times 0.03^2) \approx 278$이 필요하다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** 참 $p$가 무엇이든 $\hat{p}$의 95% 오차한계가 최대 0.02가 되려면 $n$이 얼마나 커야 하는가?

</div>

??? success "풀이"
    오차한계는 $E = z_{0.025} \cdot \text{SE}(\hat{p}) = 1.96 \sqrt{p(1-p)/n}$이다.

    최악의 경우 $p(1-p) \le 1/4$를 사용하면:

    $$
    E \le 1.96 \cdot \frac{1}{2\sqrt{n}}
    $$

    $E \le 0.02$로 두면:

    $$
    1.96 \cdot \frac{1}{2\sqrt{n}} \le 0.02 \implies \sqrt{n} \ge \frac{1.96}{0.04} = 49 \implies n \ge 2401
    $$

    표본크기가 최소 $n = 2401$이면 오차한계 0.02 이하가 보장된다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** $p = 0.01$이고 $n = 100$일 때 $np$와 $n(1-p)$를 계산하라. 정규근사가 경험 법칙을 충족하는가? 이런 상황에 대한 대안을 제시하라.

</div>

??? success "풀이"
    계산하면:

    $$
    np = 100 \times 0.01 = 1, \qquad n(1-p) = 100 \times 0.99 = 99
    $$

    $np = 1 < 10$이므로 경험 법칙이 충족되지 **않으며** 정규근사를 신뢰할 수 없다. $n\hat{p} = \sum X_i$의 분포는 $\text{Binomial}(100, 0.01)$로 오른쪽으로 심하게 치우쳐 있고 0 근처에 몰려 있다.

    대안으로는 다음이 있다:

    - **정확한 binomial 방법**: 신뢰구간과 검정에 정확한 binomial 분포를 사용한다(예: Clopper–Pearson 구간).
    - **Poisson 근사**: $n$이 크고 $p$가 작으므로 $\sum X_i \approx \text{Poisson}(\lambda = np = 1)$이며 다루기가 더 간단한 경우가 많다.
    - **Wilson 구간**: $p$가 0이나 1에 가까울 때 Wald(정규 기반) 구간보다 잘 작동하도록 수정된 신뢰구간이다.

    일반적으로 관심 사건이 드물면 $n$을 크게 늘리거나 정규근사에 의존하지 않는 방법을 써야 한다. $\square$
