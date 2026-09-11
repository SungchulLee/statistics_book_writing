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

# z-구간: sigma를 안다고 가정한다.
# ppf(1 - alpha/2)인 것에 주의하라. 양쪽 꼬리에 alpha/2씩 남겨야 하므로
# 필요한 것은 상위 alpha/2 분위수, 즉 왼쪽 누적확률 1 - alpha/2 지점이다.
sigma_known = 6
z_crit = stats.norm.ppf(1 - alpha / 2)
me_z = z_crit * sigma_known / np.sqrt(n)
print(f"z-interval: ({xbar - me_z:.2f}, {xbar + me_z:.2f})")

# t-구간: sigma를 모르고 s로 대신한다.
# s가 그 자체로 흔들리는 양이라 임계값을 z보다 키워 그 불확실성을 갚아 준다.
# 자유도는 n-1이다. 편차를 x_bar에서 재는 순간 자유도 하나를 잃기 때문이다.
t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
me_t = t_crit * s / np.sqrt(n)
print(f"t-interval: ({xbar - me_t:.2f}, {xbar + me_t:.2f})")

# scipy로 한 번에. scale에 s가 아니라 **표준오차** s/sqrt(n)을 넣어야 한다.
# 여기서 흔히 틀린다. loc/scale은 자료의 분포가 아니라 x_bar의 분포를 가리킨다.
ci = stats.t.interval(1 - alpha, df=n - 1, loc=xbar, scale=s / np.sqrt(n))
print(f"scipy t.interval: ({ci[0]:.2f}, {ci[1]:.2f})")
```

출력:

```
z-interval: (121.27, 126.53)
t-interval: (121.21, 126.59)
scipy t.interval: (121.21, 126.59)
```

$\sigma = 6$을 안다고 가정한 z-구간과, $s = 5.74$를 자료에서 추정해 쓴 t-구간의 너비가 거의 같다($\pm 2.63$ 대 $\pm 2.69$). $n = 20$에서는 $t_{0.025,\,19} = 2.093$이 $z_{0.025} = 1.960$과 크게 다르지 않고, $s$가 $\sigma$보다 조금 작게 나온 것이 임계값 차이를 거의 상쇄했기 때문이다. $n$이 작아지면 이 균형이 깨진다.

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

# Wald: 표준오차에 p_hat을 그냥 대입한다. 가장 간단하지만 가장 나쁘다.
# p_hat이 0이나 1이면 표준오차가 0이 되어 폭이 0인 구간이 나온다.
me_wald = z * np.sqrt(p_hat * (1 - p_hat) / n)
print(f"Wald: ({p_hat - me_wald:.4f}, {p_hat + me_wald:.4f})")

# Wilson: |p_hat - p| <= z*sqrt(p(1-p)/n) 을 p에 대한 이차부등식으로 풀어 얻는다.
# 표준오차에 미지의 p를 그대로 두고 풀었다는 것이 핵심이다.
# 그래서 중심이 p_hat이 아니라 p_hat과 0.5 사이로 조금 당겨진다.
denom = 1 + z**2 / n
center = (p_hat + z**2 / (2 * n)) / denom
me_wilson = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
print(f"Wilson: ({center - me_wilson:.4f}, {center + me_wilson:.4f})")

# Agresti-Coull: Wilson의 중심을 그대로 쓰되 너비는 Wald 공식으로 계산한다.
# 95%에서는 z^2 = 3.84 ~ 4 이므로 "성공 2개와 실패 2개를 더하고 Wald를 쓰라"는
# 손계산 규칙이 된다.
n_tilde = n + z**2
p_tilde = (x + z**2 / 2) / n_tilde
me_ac = z * np.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
print(f"Agresti-Coull: ({p_tilde - me_ac:.4f}, {p_tilde + me_ac:.4f})")
```

출력:

```
Wald: (0.3516, 0.4884)
Wilson: (0.3537, 0.4893)
Agresti-Coull: (0.3537, 0.4893)
```

$n = 200$이고 $\hat p = 0.42$로 극단적이지 않아 세 구간이 거의 겹친다. Wilson과 Agresti–Coull은 소수점 넷째 자리까지 같다. 두 구간의 중심이 $z^2$만큼 보정된 같은 값이고, 너비를 계산하는 방식만 다르기 때문이다. 차이는 $n$이 작거나 $\hat p$가 0 또는 1에 가까울 때 드러난다.

## 분산의 신뢰구간

자료가 정규모집단에서 나왔다는 가정 아래 추축량 $(n-1)S^2 / \sigma^2$은 자유도 $n - 1$인 카이제곱분포를 따른다. 그 결과 $\sigma^2$의 $(1-\alpha)100\%$ 신뢰구간은

$$
\left(\frac{(n-1)s^2}{\chi^2_{1-\alpha/2,\,n-1}},\;\;
      \frac{(n-1)s^2}{\chi^2_{\alpha/2,\,n-1}}\right)
$$

### Python 코드

```python
data = np.array([120, 125, 118, 130, 122, 128, 115, 135, 121, 126])   # 앞의 20개 중 10개
n = len(data)
s2 = data.var(ddof=1)

chi2_lower = stats.chi2.ppf(alpha / 2, df=n - 1)
chi2_upper = stats.chi2.ppf(1 - alpha / 2, df=n - 1)

# 분모에 들어가는 임계값이 **뒤바뀐다**. 자주 틀리는 자리다.
# (n-1)s²/σ² 이 chi2_lower 와 chi2_upper 사이에 있다는 부등식을
# σ² 에 대해 풀면 σ² 이 분모로 내려가면서 대소가 뒤집히기 때문이다.
ci_var = ((n - 1) * s2 / chi2_upper, (n - 1) * s2 / chi2_lower)
# 제곱근은 단조증가 함수라 양끝에 그대로 씌우면 σ의 구간이 된다.
ci_sd = (np.sqrt(ci_var[0]), np.sqrt(ci_var[1]))

print(f"95% CI for sigma^2: ({ci_var[0]:.2f}, {ci_var[1]:.2f})")
print(f"95% CI for sigma:   ({ci_sd[0]:.2f}, {ci_sd[1]:.2f})")
```

출력:

```
95% CI for sigma^2: (17.03, 119.98)
95% CI for sigma:   (4.13, 10.95)
```

$n = 10$에서 $\sigma^2$의 구간은 위쪽 끝이 아래쪽 끝의 일곱 배다. 카이제곱분포가 오른쪽으로 길게 늘어져 있어 구간이 $s^2$을 중심으로 대칭이 아니며, 분산은 평균보다 훨씬 추정하기 어렵다는 뜻이다.

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
s1, s2_val = group_a.std(ddof=1), group_b.std(ddof=1)   # 위의 분산 s2와 이름이 겹치지 않게

# 두 표본이 독립이므로 분산이 더해진다. 표준오차는 제곱해서 더한 뒤 제곱근이다.
se = np.sqrt(s1**2 / n1 + s2_val**2 / n2)

# Satterthwaite 자유도. 정수가 아니어도 된다.
# 두 분산이 같고 n도 같으면 n1+n2-2가 되고, 한쪽 분산이 압도하면
# 그쪽 표본의 자유도(n-1)로 줄어든다. 즉 "실효 표본크기"를 재는 양이다.
df_welch = (s1**2 / n1 + s2_val**2 / n2)**2 / (
    (s1**2 / n1)**2 / (n1 - 1) + (s2_val**2 / n2)**2 / (n2 - 1)
)
t_crit = stats.t.ppf(1 - alpha / 2, df=df_welch)
diff = x1 - x2
ci_welch = (diff - t_crit * se, diff + t_crit * se)
print(f"Welch CI: ({ci_welch[0]:.2f}, {ci_welch[1]:.2f})")
```

출력:

```
Welch CI: (-6.71, -3.09)
```

구간이 통째로 음수쪽에 있어 0을 담지 않는다. B군의 평균이 A군보다 3에서 7 정도 높다고 읽으며, 이는 유의수준 5%에서 $\mu_1 = \mu_2$를 기각하는 것과 같은 말이다(9장).

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

        # 흔하지만 틀린 방식: sigma를 모르면서 s를 넣고 임계값은 z를 쓴다.
        # 자료마다 흔들리는 s를 상수처럼 취급하는 셈이라 구간이 너무 좁아진다.
        me_z = 1.96 * s / np.sqrt(n)
        if xbar - me_z <= mu_true <= xbar + me_z:
            z_covers += 1

        # 올바른 방식: 같은 s를 쓰되 임계값을 t로 키운다.
        t_c = stats.t.ppf(0.975, df=n - 1)
        me_t = t_c * s / np.sqrt(n)
        if xbar - me_t <= mu_true <= xbar + me_t:
            t_covers += 1

    # 참값을 알고 있으니 "구간이 참값을 담았는가"를 그냥 세면 된다.
    # 이것이 신뢰수준의 정의다. 명목값 0.95에 얼마나 가까운지를 본다.
    print(f"n={n:>3}: z-coverage={z_covers/n_sim:.3f}  t-coverage={t_covers/n_sim:.3f}")
```

출력:

```
n=  5: z-coverage=0.876  t-coverage=0.953
n= 10: z-coverage=0.915  t-coverage=0.945
n= 30: z-coverage=0.937  t-coverage=0.945
n=100: z-coverage=0.946  t-coverage=0.949
```

$n = 5$에서 잘못된 z-구간의 포함확률은 95%가 아니라 87.6%다. 스무 번에 한 번 놓친다고 믿고 있지만 실제로는 여덟 번에 한 번 놓친다. $t$-구간은 같은 자료로 0.953을 낸다. 모의실험 오차는 $\sqrt{0.95 \times 0.05 / 10000} \approx 0.002$이므로 표의 셋째 자리 흔들림은 그 범위 안이다.

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
        # 자료를 모으기 전이라 s가 없으므로 t를 쓸 수 없다.
        # 그래서 표본크기 계산은 언제나 z와 sigma의 사전 추정값으로 한다.
        # 올림(ceil)은 부족한 쪽으로 내려가지 않기 위해서다.
        n_needed = int(np.ceil((z * sigma_est / E)**2))
        print(f"  E=+/-{E}, {conf*100:.0f}% conf -> n = {n_needed}")
```

출력:

```
  E=+/-1, 90% conf -> n = 609
  E=+/-1, 95% conf -> n = 865
  E=+/-1, 99% conf -> n = 1493
  E=+/-2, 90% conf -> n = 153
  E=+/-2, 95% conf -> n = 217
  E=+/-2, 99% conf -> n = 374
  E=+/-3, 90% conf -> n = 68
  E=+/-3, 95% conf -> n = 97
  E=+/-3, 99% conf -> n = 166
  E=+/-5, 90% conf -> n = 25
  E=+/-5, 95% conf -> n = 35
  E=+/-5, 99% conf -> n = 60
```

$E$가 분모에서 제곱되므로 오차한계를 절반으로 줄이려면 표본을 네 배 모아야 한다($E = 2$의 217개 대 $E = 1$의 865개). 반면 신뢰수준을 95%에서 99%로 올리는 값은 그보다 싸다(217개 → 374개). 정밀도가 신뢰수준보다 비싸다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** 전구 36개의 확률표본에서 평균 수명이 1200시간이고 모표준편차는 $\sigma = 120$시간으로 알려져 있다. $z$-구간으로 $\mu$의 95% 신뢰구간을 구성하고 해석하라.

</div>

??? success "풀이"

    표준오차는 $\text{SE} = 120 / \sqrt{36} = 20$이다. 임계값은 $z_{0.025} = 1.96$이다. 오차한계는 $1.96 \times 20 = 39.2$이다. 따라서 95% 신뢰구간은

    $$
    1200 \pm 39.2 = (1160.8,\; 1239.2)
    $$

    참 평균 수명이 1160.8시간과 1239.2시간 사이에 있다고 95% 신뢰한다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** 유권자 400명을 조사했더니 220명이 어떤 안건을 지지한다. 참 비율 $p$에 대한 Wald와 Wilson 95% 신뢰구간을 계산하라. 어느 쪽을 선호하며 왜인가?

</div>

??? success "풀이"

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

<div class="drillbox" markdown>

**연습문제 3.** 정규모집단에서 뽑은 측정값 15개에서 $s^2 = 25$를 얻었다. $\sigma^2$의 90% 신뢰구간을 구성하라. 그다음 그에 대응하는 $\sigma$의 구간을 유도하라.

</div>

??? success "풀이"

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

<div class="drillbox" markdown>

**연습문제 4.** $n \to \infty$일 때 $t$-구간 $\bar{X} \pm t_{\alpha/2,\,n-1} \cdot s/\sqrt{n}$이 $z$-구간 $\bar{X} \pm z_{\alpha/2} \cdot \sigma/\sqrt{n}$으로 수렴함을 증명하라.

</div>

??? success "풀이"

    두 가지 수렴이 결합된다:

    1. **임계값.** $\text{df} = n - 1 \to \infty$일 때 $t_{n-1}$ 분포가 $N(0,1)$로 수렴한다. 따라서 $t_{\alpha/2,\,n-1} \to z_{\alpha/2}$이다.

    2. **표본표준편차.** 대수의법칙에 의해 $s^2 \xrightarrow{P} \sigma^2$이고, 연속사상정리에 의해 $s \xrightarrow{P} \sigma$이다.

    합치면 오차한계가

    $$
    t_{\alpha/2,\,n-1} \cdot \frac{s}{\sqrt{n}} \;\xrightarrow{P}\; z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}
    $$

    을 만족하므로 큰 $n$에서 두 구간을 구별할 수 없게 된다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** $\sigma = 10$이라 가정할 때 99% 신뢰수준에서 모평균을 $\pm 2$ 단위 이내로 추정하려면 표본이 얼마나 커야 하는가?

</div>

??? success "풀이"

    임계값은 $z_{0.005} = 2.576$이다. 필요한 표본크기는

    $$
    n = \left\lceil \left(\frac{2.576 \times 10}{2}\right)^2 \right\rceil = \left\lceil 12.88^2 \right\rceil = \left\lceil 165.87 \right\rceil = 166
    $$

    적어도 관측값 166개가 필요하다. $\square$
