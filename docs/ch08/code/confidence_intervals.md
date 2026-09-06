# 신뢰구간 시연

## 개요

신뢰구간은 미지의 모수에 대해 그럴듯한 값들의 범위를 주며, 반복표본추출에서 지정된 신뢰수준으로 참 모수를 잡아내도록 구성된다. 이 페이지에서는 모평균 $\mu$, 비율 $p$, 분산 $\sigma^2$, 두 평균의 차이 $\mu_1 - \mu_2$에 대한 신뢰구간을 만드는 방법을 포함확률 모의실험, 표본크기 계산과 함께 시연한다.

## 모평균의 신뢰구간

### z-구간 (분산을 아는 경우)

모표준편차 $\sigma$를 알 때 $\mu$의 $(1-\alpha)100\%$ 신뢰구간은

$$
\bar{X} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}
$$

여기서 $z_{\alpha/2}$는 표준정규분포의 상위 $\alpha/2$ 분위수이다.

### t-구간 (분산을 모르는 경우)

$\sigma$를 모르고 표본표준편차 $s$로 대체하면 자유도 $n - 1$인 $t$-분포를 쓴다:

$$
\bar{X} \pm t_{\alpha/2,\, n-1} \cdot \frac{s}{\sqrt{n}}
$$

### Python 코드

```python
import numpy as np
from scipy import stats

data = np.array([120, 125, 118, 130, 122, 128, 115, 135, 121, 126,
                 119, 132, 124, 117, 129, 123, 131, 116, 127, 120])
n = len(data)
xbar = data.mean()
s = data.std(ddof=1)
alpha = 0.05

# z-interval (sigma known)
sigma_known = 6
z_crit = stats.norm.ppf(1 - alpha / 2)
me_z = z_crit * sigma_known / np.sqrt(n)
print(f"z-interval: ({xbar - me_z:.2f}, {xbar + me_z:.2f})")

# t-interval (sigma unknown)
t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
me_t = t_crit * s / np.sqrt(n)
print(f"t-interval: ({xbar - me_t:.2f}, {xbar + me_t:.2f})")

# Using scipy directly
ci = stats.t.interval(1 - alpha, df=n - 1, loc=xbar, scale=s / np.sqrt(n))
print(f"scipy t.interval: ({ci[0]:.2f}, {ci[1]:.2f})")
```

## 비율의 신뢰구간

비율 $\hat{p} = x / n$에 대한 **Wald 구간**은

$$
\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}
$$

**Wilson score 구간**은 포함확률을 개선하기 위해 중심과 너비를 조정한다:

$$
\frac{\hat{p} + \frac{z^2}{2n}}{1 + \frac{z^2}{n}}
\;\pm\;
\frac{z}{1 + \frac{z^2}{n}}
\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}
$$

**Agresti–Coull 구간**은 가상의 성공과 실패를 $z^2/2$개씩 더한 뒤, 보정된 개수 $\tilde{n} = n + z^2$과 $\tilde{p} = (x + z^2/2) / \tilde{n}$에 Wald 공식을 적용한다.

### Python 코드

```python
x, n = 84, 200
p_hat = x / n
z = stats.norm.ppf(1 - alpha / 2)

# Wald interval
me_wald = z * np.sqrt(p_hat * (1 - p_hat) / n)
print(f"Wald: ({p_hat - me_wald:.4f}, {p_hat + me_wald:.4f})")

# Wilson interval
denom = 1 + z**2 / n
center = (p_hat + z**2 / (2 * n)) / denom
me_wilson = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
print(f"Wilson: ({center - me_wilson:.4f}, {center + me_wilson:.4f})")

# Agresti-Coull interval
n_tilde = n + z**2
p_tilde = (x + z**2 / 2) / n_tilde
me_ac = z * np.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
print(f"Agresti-Coull: ({p_tilde - me_ac:.4f}, {p_tilde + me_ac:.4f})")
```

## 분산의 신뢰구간

자료가 정규모집단에서 나왔다는 가정 아래 추축량 $(n-1)S^2 / \sigma^2$은 자유도 $n - 1$인 카이제곱분포를 따른다. 그 결과 $\sigma^2$의 $(1-\alpha)100\%$ 신뢰구간은

$$
\left(\frac{(n-1)s^2}{\chi^2_{1-\alpha/2,\,n-1}},\;\;
      \frac{(n-1)s^2}{\chi^2_{\alpha/2,\,n-1}}\right)
$$

### Python 코드

```python
data = np.array([120, 125, 118, 130, 122, 128, 115, 135, 121, 126])
n = len(data)
s2 = data.var(ddof=1)

chi2_lower = stats.chi2.ppf(alpha / 2, df=n - 1)
chi2_upper = stats.chi2.ppf(1 - alpha / 2, df=n - 1)

ci_var = ((n - 1) * s2 / chi2_upper, (n - 1) * s2 / chi2_lower)
ci_sd = (np.sqrt(ci_var[0]), np.sqrt(ci_var[1]))

print(f"95% CI for sigma^2: ({ci_var[0]:.2f}, {ci_var[1]:.2f})")
print(f"95% CI for sigma:   ({ci_sd[0]:.2f}, {ci_sd[1]:.2f})")
```

## 평균 차이에 대한 이표본 신뢰구간

### Welch의 t-구간 (분산이 다른 경우)

크기 $n_1$과 $n_2$인 독립표본에 대해 $\mu_1 - \mu_2$의 신뢰구간은

$$
(\bar{X}_1 - \bar{X}_2) \;\pm\; t_{\alpha/2,\,\nu} \cdot \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}
$$

여기서 Satterthwaite 자유도는

$$
\nu = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}}
$$

### 합동 t-구간 (분산이 같은 경우)

등분산을 가정하면 합동분산은 $s_p^2 = [(n_1-1)s_1^2 + (n_2-1)s_2^2] / (n_1+n_2-2)$이고 구간은 다음이 된다:

$$
(\bar{X}_1 - \bar{X}_2) \;\pm\; t_{\alpha/2,\,n_1+n_2-2} \cdot s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}
$$

### Python 코드

```python
group_a = np.array([12, 15, 11, 14, 13, 16, 10, 15, 12, 14])
group_b = np.array([18, 20, 17, 19, 16, 21, 15, 20, 18, 17])

n1, n2 = len(group_a), len(group_b)
x1, x2 = group_a.mean(), group_b.mean()
s1, s2_val = group_a.std(ddof=1), group_b.std(ddof=1)

# Welch's t-interval
se = np.sqrt(s1**2 / n1 + s2_val**2 / n2)
df_welch = (s1**2 / n1 + s2_val**2 / n2)**2 / (
    (s1**2 / n1)**2 / (n1 - 1) + (s2_val**2 / n2)**2 / (n2 - 1)
)
t_crit = stats.t.ppf(1 - alpha / 2, df=df_welch)
diff = x1 - x2
ci_welch = (diff - t_crit * se, diff + t_crit * se)
print(f"Welch CI: ({ci_welch[0]:.2f}, {ci_welch[1]:.2f})")
```

## 포함확률 모의실험

포함확률 모의실험은 표본을 여러 번 뽑아 각각에서 신뢰구간을 만들고 참 모수를 담은 비율을 기록한다. 경험적 포함확률은 명목 신뢰수준에 가까워야 한다.

```python
np.random.seed(42)
mu_true, sigma_true = 100, 15
n_sim = 10_000

for n in [5, 10, 30, 100]:
    z_covers = 0
    t_covers = 0
    for _ in range(n_sim):
        sample = np.random.normal(mu_true, sigma_true, n)
        xbar = sample.mean()
        s = sample.std(ddof=1)

        # z-interval using s (common but incorrect)
        me_z = 1.96 * s / np.sqrt(n)
        if xbar - me_z <= mu_true <= xbar + me_z:
            z_covers += 1

        # t-interval (correct)
        t_c = stats.t.ppf(0.975, df=n - 1)
        me_t = t_c * s / np.sqrt(n)
        if xbar - me_t <= mu_true <= xbar + me_t:
            t_covers += 1

    print(f"n={n:>3}: z-coverage={z_covers/n_sim:.3f}  t-coverage={t_covers/n_sim:.3f}")
```

## 해석

- $n$이 작으면 $\sigma$ 자리에 $s$를 넣은 z-구간은 **포함확률이 부족하다**: 경험적 포함확률이 95% 아래로 떨어진다. $t$-구간은 $t$-분포의 더 큰 임계값을 써서 이를 바로잡는다.
- $n$이 커지면 $t$와 $z$의 임계값이 수렴하므로 두 구간의 성능이 비슷해진다.
- $n$이 작거나 $p$가 0 또는 1에 가까울 때는 Wilson과 Agresti–Coull 비율 구간이 Wald 구간보다 낫다.
- 카이제곱 분산 구간은 정규성 아래에서만 정확하다. 정규가 아닌 자료에는 붓스트랩 신뢰구간이 낫다.

## 표본크기의 결정

신뢰수준 $1 - \alpha$에서 오차한계 $E$ 이내로 $\mu$를 추정하려면 필요한 표본크기는

$$
n = \left\lceil \left(\frac{z_{\alpha/2} \cdot \sigma}{E}\right)^2 \right\rceil
$$

비율에 대해 보수적으로 $p = 0.5$를 택하면

$$
n = \left\lceil \left(\frac{z_{\alpha/2}}{2E}\right)^2 \right\rceil
$$

```python
sigma_est = 15
for E in [1, 2, 3, 5]:
    for conf in [0.90, 0.95, 0.99]:
        z = stats.norm.ppf(1 - (1 - conf) / 2)
        n_needed = int(np.ceil((z * sigma_est / E)**2))
        print(f"  E=+/-{E}, {conf*100:.0f}% conf -> n = {n_needed}")
```

## 연습문제

**연습문제 1.** 전구 36개의 확률표본에서 평균 수명이 1200시간이고 모표준편차는 $\sigma = 120$시간으로 알려져 있다. $z$-구간으로 $\mu$의 95% 신뢰구간을 구성하고 해석하라.

??? success "연습문제 1 풀이"

    표준오차는 $\text{SE} = 120 / \sqrt{36} = 20$이다. 임계값은 $z_{0.025} = 1.96$이다. 오차한계는 $1.96 \times 20 = 39.2$이다. 따라서 95% 신뢰구간은

    $$
    1200 \pm 39.2 = (1160.8,\; 1239.2)
    $$

    참 평균 수명이 1160.8시간과 1239.2시간 사이에 있다고 95% 신뢰한다. $\square$

---

**연습문제 2.** 유권자 400명을 조사했더니 220명이 어떤 안건을 지지한다. 참 비율 $p$에 대한 Wald와 Wilson 95% 신뢰구간을 계산하라. 어느 쪽을 선호하며 왜인가?

??? success "연습문제 2 풀이"

    여기서 $\hat{p} = 220/400 = 0.55$이고 $z_{0.025} = 1.96$이다.

    **Wald 구간:**

    $$
    \text{SE} = \sqrt{\frac{0.55 \times 0.45}{400}} = 0.02487
    $$

    $$
    0.55 \pm 1.96 \times 0.02487 = (0.5013,\; 0.5987)
    $$

    **Wilson 구간:** 분모 $1 + z^2/n = 1 + 3.8416/400 = 1.009604$이므로,

    $$
    \tilde{p} = \frac{0.55 + 3.8416/800}{1.009604} = \frac{0.554802}{1.009604} \approx 0.5495
    $$

    $$
    \text{반너비} = \frac{1.96\sqrt{0.55 \times 0.45/400 + 3.8416/640000}}{1.009604} \approx 0.0485
    $$

    $$
    \text{Wilson 신뢰구간} \approx (0.5010,\; 0.5980)
    $$

    $n$이 크고 $\hat{p}$가 극단적이지 않아 두 구간이 가깝다. 그래도 일반적으로는 $n$이 작거나 $\hat{p}$가 극단적일 때 포함 성질이 더 좋은 Wilson 구간이 낫다. $\square$

---

**연습문제 3.** 정규모집단에서 뽑은 측정값 15개에서 $s^2 = 25$를 얻었다. $\sigma^2$의 90% 신뢰구간을 구성하라. 그다음 그에 대응하는 $\sigma$의 구간을 유도하라.

??? success "연습문제 3 풀이"

    $n = 15$, $\text{df} = 14$, $\alpha = 0.10$일 때:

    $$
    \chi^2_{0.05,\,14} = 6.571, \quad \chi^2_{0.95,\,14} = 23.685
    $$

    $\sigma^2$의 90% 신뢰구간은

    $$
    \left(\frac{14 \times 25}{23.685},\; \frac{14 \times 25}{6.571}\right) = (14.78,\; 53.26)
    $$

    제곱근을 취하면 $\sigma$의 90% 신뢰구간은

    $$
    (\sqrt{14.78},\; \sqrt{53.26}) = (3.84,\; 7.30)
    $$

    $\square$

---

**연습문제 4.** $n \to \infty$일 때 $t$-구간 $\bar{X} \pm t_{\alpha/2,\,n-1} \cdot s/\sqrt{n}$이 $z$-구간 $\bar{X} \pm z_{\alpha/2} \cdot \sigma/\sqrt{n}$으로 수렴함을 증명하라.

??? success "연습문제 4 풀이"

    두 가지 수렴이 결합된다:

    1. **임계값.** $\text{df} = n - 1 \to \infty$일 때 $t_{n-1}$ 분포가 $N(0,1)$로 수렴한다. 따라서 $t_{\alpha/2,\,n-1} \to z_{\alpha/2}$이다.

    2. **표본표준편차.** 대수의법칙에 의해 $s^2 \xrightarrow{P} \sigma^2$이고, 연속사상정리에 의해 $s \xrightarrow{P} \sigma$이다.

    합치면 오차한계가

    $$
    t_{\alpha/2,\,n-1} \cdot \frac{s}{\sqrt{n}} \;\xrightarrow{P}\; z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}
    $$

    을 만족하므로 큰 $n$에서 두 구간을 구별할 수 없게 된다. $\square$

---

**연습문제 5.** $\sigma = 10$이라 가정할 때 99% 신뢰수준에서 모평균을 $\pm 2$ 단위 이내로 추정하려면 표본이 얼마나 커야 하는가?

??? success "연습문제 5 풀이"

    임계값은 $z_{0.005} = 2.576$이다. 필요한 표본크기는

    $$
    n = \left\lceil \left(\frac{2.576 \times 10}{2}\right)^2 \right\rceil = \left\lceil 12.88^2 \right\rceil = \left\lceil 165.87 \right\rceil = 166
    $$

    적어도 관측값 166개가 필요하다. $\square$
