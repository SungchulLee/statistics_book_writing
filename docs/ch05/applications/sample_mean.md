# 평균의 표본분포

## 개요

**표본평균 $\bar{X}$의 표본분포**는 모집단에서 크기 $n$인 표본을 반복해서 뽑을 때 $\bar{X}$가 어떻게 달라지는지를 기술한다. 통계학에서 가장 중요한 표본분포 하나이며, $\mu$에 대한 신뢰구간, $t$ 검정, 응용통계학의 상당 부분을 떠받친다.

## 수학적 정의

$X_1, X_2, \dots, X_n$을 평균 $\mu$, 분산 $\sigma^2 < \infty$인 모집단에서 뽑은 i.i.d. 표본이라 하자. 표본평균은:

$$
\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i
$$

## 성질

### 기댓값 (불편성)

$$
E[\bar{X}] = \mu
$$

표본평균은 모평균의 **불편추정량**이다. 평균적으로 $\mu$를 과대추정하지도 과소추정하지도 않는다.

### 분산과 표준오차

$$
\text{Var}(\bar{X}) = \frac{\sigma^2}{n}, \qquad
\text{SE}(\bar{X}) = \frac{\sigma}{\sqrt{n}}
$$

$n$이 커질수록 표준오차가 작아진다. 표본이 클수록 $\mu$를 더 정밀하게 추정한다.

### 모양 (중심극한정리)

중심극한정리에 의해 $n$이 충분히 크면:

$$
\frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} N(0, 1)
$$

이므로 근사적으로:

$$
\bar{X} \sim N\!\left(\mu, \frac{\sigma^2}{n}\right)
$$

$\sigma^2 < \infty$이기만 하면 모집단의 모양과 무관하게 성립한다.

- **정규** 모집단에서는 **모든** $n$에 대해 정확하다.
- **치우쳤거나 꼬리가 두꺼운** 모집단에서는 더 큰 $n$이 필요하다.

## 표준화된 형태

| 상황 | 표준화된 통계량 | 분포 |
|----------|----------------------|--------------|
| 정규모집단, $\sigma$를 앎 | $\frac{\bar{X} - \mu}{\sigma/\sqrt{n}}$ | 정확히 $N(0, 1)$ |
| 정규모집단, $\sigma$를 모름 | $\frac{\bar{X} - \mu}{S/\sqrt{n}}$ | 정확히 $t_{n-1}$ |
| 임의의 모집단, 큰 $n$, $\sigma$를 앎 | $\frac{\bar{X} - \mu}{\sigma/\sqrt{n}}$ | 근사적으로 $N(0, 1)$ |
| 임의의 모집단, 큰 $n$, $\sigma$를 모름 | $\frac{\bar{X} - \mu}{S/\sqrt{n}}$ | 근사적으로 $N(0, 1)$ 또는 $t_{n-1}$ |

## 예: 표준오차 계산

<div class="probox" markdown>

**문제.** <span class="diff easy" title="쉬움"></span> 모집단이 $\mu = 100$, $\sigma = 4$이다. $n = 25$일 때:

$$
\text{SE}(\bar{X}) = \frac{4}{\sqrt{25}} = 0.8
$$

크기 25인 표본을 반복해서 뽑으면 표본평균들이 100 주위에 모이며 전형적인 편차는 0.8이다.

</div>

## 보기

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> 사과 무게. 사과 무게가 $N(150, 20^2)$이다. $n = 25$일 때 $P(\bar{X} > 155)$를 구하라.

</div>

??? success "풀이"

    $$
    \text{SE} = \frac{20}{\sqrt{25}} = 4, \qquad
    Z = \frac{155 - 150}{4} = 1.25
    $$

    $$
    P(\bar{X} > 155) = P(Z > 1.25) = 1 - \mathcal{N}(1.25) \approx 0.1056
    $$

    ```python
    from scipy import stats
    print(f"P(X_bar > 155) = {stats.norm.sf(1.25):.4f}")
    ```

    출력:

    ```
    P(X_bar > 155) = 0.1056
    ```
<div class="exbox" markdown>

**보기 2.** <span class="diff easy" title="쉬움"></span> 수면 시간. 평균 수면 시간이 7시간이고 $\sigma = 1.5$이다. $n = 49$일 때 $P(6.8 < \bar{X} < 7.2)$를 구하라.

</div>

??? success "풀이"

    $$
    \text{SE} = \frac{1.5}{\sqrt{49}} = 0.2143
    $$

    $$
    Z_1 = \frac{6.8 - 7}{0.2143} \approx -0.93, \qquad
    Z_2 = \frac{7.2 - 7}{0.2143} \approx 0.93
    $$

    $$
    P(6.8 < \bar{X} < 7.2) = \mathcal{N}(0.93) - \mathcal{N}(-0.93) \approx 0.6476
    $$

    ```python
    from scipy import stats
    print(f"P(6.8 < X_bar < 7.2) = {stats.norm.cdf(0.93) - stats.norm.cdf(-0.93):.4f}")
    ```

    출력:

    ```
    P(6.8 < X_bar < 7.2) = 0.6476
    ```
<div class="exbox" markdown>

**보기 3.** <span class="diff easy" title="쉬움"></span> 체중 (소표본, 정규모집단). 체중이 $N(70, 10^2)$이다. $n = 5$일 때 $P(\bar{X} > 72)$를 구하라.

</div>

??? success "풀이"
    모집단이 정규이므로 $n = 5$에서도 결과가 정확하다:

    $$
    \text{SE} = \frac{10}{\sqrt{5}} \approx 4.47, \qquad
    Z = \frac{72 - 70}{4.47} \approx 0.447
    $$

    $$
    P(\bar{X} > 72) \approx 0.3274
    $$

    ```python
    from scipy import stats
    print(f"P(X_bar > 72) = {stats.norm.sf(0.447):.4f}")
    ```

    출력:

    ```
    P(X_bar > 72) = 0.3274
    ```
<div class="exbox" markdown>

**보기 4.** <span class="diff easy" title="쉬움"></span> 물 부족. 평균 물 소비량이 2 L이고 $\sigma = 0.7$ L이다. 50명이 물 110 L를 가지고 여행할 때 물이 떨어질 확률을 구하라.

</div>

??? success "풀이"
    물이 떨어진다는 것은 $\bar{X} > 110/50 = 2.2$를 뜻한다:

    $$
    \text{SE} = \frac{0.7}{\sqrt{50}} \approx 0.0990, \qquad
    Z = \frac{2.2 - 2}{0.0990} \approx 2.020
    $$

    $$
    P(\bar{X} > 2.2) = P(Z > 2.020) \approx 0.0217
    $$
<div class="exbox" markdown>

**보기 5.** <span class="diff easy" title="쉬움"></span> 전구 (정규성 가정 없이). 전구 수명이 $\mu = 800$, $\sigma = 100$이다. $n = 5$일 때 정규성을 가정하지 않고 $P(\bar{X} > 810)$을 구하라.

</div>

??? success "풀이"
    $n = 5$이고 정규성 가정이 없으면 중심극한정리를 믿고 적용할 수 없다. 모집단 모양에 관한 추가 정보 없이는 이 확률을 **구할 수 없다**.
<div class="exbox" markdown>

**보기 6.** <span class="diff easy" title="쉬움"></span> 치우친 매출 (대표본). 일간 매출이 오른쪽으로 치우쳐 있고 $\mu = 2000$, $\sigma = 500$이다. $n = 100$일 때 $P(\bar{X} > 2100)$을 구하라.

</div>

??? success "풀이"
    모집단이 치우쳐 있지만 $n = 100$이면 중심극한정리를 쓰기에 충분히 크다:

    $$
    \text{SE} = \frac{500}{\sqrt{100}} = 50, \qquad
    Z = \frac{2100 - 2000}{50} = 2
    $$

    $$
    P(\bar{X} > 2100) = P(Z > 2) \approx 0.0228
    $$

    ```python
    from scipy import stats
    print(f"P(X_bar > 2100) = {stats.norm.sf(2):.4f}")
    ```

    출력:

    ```
    P(X_bar > 2100) = 0.0228
    ```
## 두 평균의 표본분포

### 분산을 알거나 표본이 큰 경우

$$
Z = \frac{(\bar{X}_1 - \bar{X}_2) - (\mu_1 - \mu_2)}{\sqrt{\sigma_1^2/n_1 + \sigma_2^2/n_2}} \sim N(0, 1)
$$

### 예: 두 교대조의 컵케이크

<div class="probox" markdown>

**문제.** <span class="diff easy" title="쉬움"></span> A 교대조: $\mu_A = 130$g, $\sigma_A = 4$g. B 교대조: $\mu_B = 125$g, $\sigma_B = 3$g. $n_A = n_B = 40$일 때 $P(|\bar{X}_A - \bar{X}_B| > 6)$을 구하라.

</div>

??? success "풀이"

    $$
    \text{SE} = \sqrt{\frac{16}{40} + \frac{9}{40}} = \sqrt{0.625} \approx 0.7906
    $$

    $$
    P(\bar{X}_A - \bar{X}_B > 6): \quad Z = \frac{6 - 5}{0.7906} \approx 1.265, \quad P = 0.1030
    $$

    $$
    P(\bar{X}_A - \bar{X}_B < -6): \quad Z = \frac{-6 - 5}{0.7906} \approx -13.91, \quad P \approx 0
    $$

    $$
    P(|\bar{X}_A - \bar{X}_B| > 6) \approx 0.1030
    $$

    ```python
    import numpy as np
    from scipy import stats

    se = np.sqrt(16/40 + 9/40)
    z_upper = (6 - 5) / se
    z_lower = (-6 - 5) / se
    prob = stats.norm.sf(z_upper) + stats.norm.cdf(z_lower)
    print(f"P(|X_bar_A - X_bar_B| > 6) = {prob:.4f}")
    ```

    출력:

    ```
    P(|X_bar_A - X_bar_B| > 6) = 0.1030
    ```
## 표본크기가 표준오차에 미치는 영향

| $n$ | SE ($\sigma = 50$일 때) |
|-----|------------------------|
| 25 | 10 |
| 100 | 5 |
| 400 | 2.5 |

$\text{SE} \propto 1/\sqrt{n}$이므로 $n$을 네 배로 늘리면 표준오차가 절반이 된다.

## 모의실험: X-bar의 표본분포

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(1)

# 1단계: 모집단을 만든다. 10만 개면 사실상 무한 모집단으로 취급할 수 있다.
population = stats.norm().rvs(100_000)
sample_size = 10       # 표본 하나의 크기
n_samples = 10_000     # 표본을 몇 번 되풀이해 뽑을 것인가

# 2단계: 표본을 1만 번 뽑고 그때마다 표본평균을 기록한다.
# replace=False 는 비복원추출. 모집단이 10만이고 표본이 10이라
# 복원이든 비복원이든 결과에 거의 차이가 없다.
#
# 현실에서는 표본을 한 번만 뽑으므로 표본평균도 하나뿐이다.
# 이 반복은 "만약 다시 뽑는다면 얼마가 나올까"를 눈으로 보기 위한 장치이며,
# 그렇게 만들어진 분포가 곧 **표집분포**다.
sample_means = [
    np.mean(np.random.choice(population, size=sample_size, replace=False))
    for _ in range(n_samples)
]

# 위아래 두 패널로 나눈다. 위는 모집단, 아래는 통계량의 표집분포다.
# 둘의 **가로 눈금이 다르다**는 점에 주의하라.
# 표집분포가 훨씬 좁으므로 같은 축에 그리면 한 점처럼 보인다.
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6))

ax0.hist(population, bins=100, density=True, alpha=0.5)
ax0.set_title('Population Distribution (Normal)', fontsize=16)

ax1.hist(sample_means, bins=100, density=True, alpha=0.5)
ax1.set_title(rf'Sampling Distribution of $\bar{{X}}$ (n = {sample_size})', fontsize=16)

for ax in (ax0, ax1):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
```

## 대학원 수준의 보충

- $\bar{X}$의 효율성은 $\text{Var}(\bar{X}) = \sigma^2/n$이라는 비율과 연결된다. 정규성 아래에서 Cramér–Rao 하한을 달성한다.
- **분산이 무한한** 모집단(예: Cauchy)에서는 중심극한정리가 적용되지 않으며 $\bar{X}$가 수렴하지 않을 수 있다.
- **Berry–Esseen 정리**는 중심극한정리의 수렴 속도를 정량화한다: $\rho = E[|X - \mu|^3]$일 때 $\sup_z |P(Z_n \leq z) - \mathcal{N}(z)| \leq C \cdot \rho / (\sigma^3 \sqrt{n})$.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
**표본평균의 평균과 표준오차.** $\mu = 75$, $\sigma = 18$인 모집단에서 $n = 9$인 표본을 뽑는다. (a) $\mathbb{E}[\bar X]$, (b) $\mathrm{SE}(\bar X)$를 계산하라.

</div>

??? success "풀이"
    (a) $\mathbb{E}[\bar X] = \mu = 75$. 표본평균은 $n$과 무관하게 불편이다.

    (b) $\mathrm{SE}(\bar X) = \sigma/\sqrt n = 18/3 = 6$. $n$을 네 배인 36으로 늘리면 표준오차가 절반인 3이 된다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
사과 무게가 $\mu = 150$ g, $\sigma = 20$ g인 정규분포를 따른다. $n = 25$인 표본에 대해 $P(\bar X > 155)$를 계산하라.

</div>

??? success "풀이"
    $\mathrm{SE} = 20/\sqrt{25} = 4$. $Z = (155 - 150)/4 = 1.25$.

    $P(\bar X > 155) = 1 - \Phi(1.25) = 0.1056$으로 약 10.6%이다.

    참고: 모집단의 정규성이 주어졌으므로 $\bar X$는 어떤 $n$에 대해서도 정확히 정규분포이다. 중심극한정리가 필요 없다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
학생들의 수면 시간이 $\mu = 7$시간, $\sigma = 1.5$시간이다. $n = 49$인 표본에 대해 $P(6.8 < \bar X < 7.2)$를 계산하라.

</div>

??? success "풀이"
    $\mathrm{SE} = 1.5/7 \approx 0.214$.

    $Z_1 = (6.8 - 7)/0.214 \approx -0.93$. $Z_2 = (7.2 - 7)/0.214 \approx 0.93$.

    $P(6.8 < \bar X < 7.2) = \Phi(0.93) - \Phi(-0.93) = 0.6476$으로 약 64.8%이다.

    ($n = 49 \ge 30$으로 정당화되는) 중심극한정리에 의해, 개별 수면 시간이 정규가 아니더라도 $\bar X$는 근사적으로 정규분포이다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
**소표본, 정규모집단.** 체중이 $X \sim N(70, 100)$ kg이다. $n = 5$인 표본에 대해 $P(\bar X > 72)$를 계산하라. $n$이 작은데도 왜 타당한가?

</div>

??? success "풀이"
    $\mathrm{SE} = 10/\sqrt 5 \approx 4.47$. $Z = (72 - 70)/4.47 \approx 0.447$.

    $P(\bar X > 72) = 1 - \Phi(0.447) \approx 0.327$로 약 32.7%이다.

    **타당성:** *모집단*이 정규이므로 $\bar X = (1/n)\sum X_i$는 어떤 $n$에 대해서도 *정확히* 정규분포이다(독립인 정규확률변수의 합은 정규분포이다). 중심극한정리가 필요 없으며 결과가 정확하다.

    정규성이 주어지지 않은 경우인 아래 **연습문제 5**와 대비된다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
**정규성 없는 전구.** 수명이 $\mu = 800$시간, $\sigma = 100$시간이고 분포는 알려져 있지 않다. $n = 5$인 표본에 대해 $P(\bar X > 810)$에 관해 무엇을 말할 수 있는가?

</div>

??? success "풀이"
    $\mathrm{SE} = 100/\sqrt 5 \approx 44.7$. 그대로 계산하면 $Z = 10/44.7 \approx 0.224$이므로 $P \approx 0.41$이다.

    **그러나:** $n = 5$는 중심극한정리를 쓰기에 너무 작고 모집단의 모양도 알려져 있지 않다. $\bar X$의 표본분포는 심하게 치우쳤거나(예: 수명이 지수분포일 때) 꼬리가 두꺼울 수 있다. 정규근사가 크게 빗나갈 수 있다.

    **Chebyshev 부등식을 이용한 보수적 한계:** $P(|\bar X - 800| \ge 10) \le \sigma^2/(n \cdot 10^2) = 10000/(5 \cdot 100) = 20$으로 아무 정보도 주지 못한다.

    실용적 결론: 분포 가정이 없거나 $n$이 더 크지 않으면 점 추정값을 포기하고 보수적 한계만 보고해야 한다. 실제 전구 수명은 흔히 Weibull 분포를 따르며, 이를 가정하면 정확한 계산이 가능하다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff easy" title="쉬움"></span>
**치우친 모집단 + 큰 $n$.** 일간 매출이 오른쪽으로 치우쳐 있고 $\mu = \$2000$, $\sigma = \$500$이다. $n = 100$인 표본에 대해 $P(\bar X > \$2100)$을 계산하라.

</div>

??? success "풀이"
    $\mathrm{SE} = 500/10 = 50$. $Z = (2100 - 2000)/50 = 2$.

    $P(\bar X > 2100) = 1 - \Phi(2) = 0.0228$로 약 2.3%이다.

    **왜 타당한가:** $n = 100$은 크므로 중심극한정리에 의해 모집단 모양과 무관하게 $\bar X$가 근사적으로 정규분포이다. 이 $n$에서는 모집단의 치우침이 $\bar X$의 분포와 무관하다.

    **주의:** 모집단의 치우침이 심하면(예: $\sigma_{\log} > 1$인 로그정규) $n = 100$에서도 $\bar X$에 치우침이 남을 수 있다. Berry-Esseen 한계에 따르면 정규근사의 오차는 $\sim \mathbb{E}|X|^3/(\sigma^3 \sqrt n)$이다. 약간 치우친 매출 자료라면 $n = 100$으로 충분하지만, 오른쪽으로 심하게 치우친 자료에서는 더 큰 $n$이나 붓스트랩이 안전하다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff easy" title="쉬움"></span>
**표본평균이 48과 52 사이일 확률.**
평균 50, 표준편차 12인 모집단에서 크기 36인 표본을 뽑을 때 표본평균이 48과 52 사이일 확률은?

</div>

??? success "풀이"
    $$\sigma_{\bar{X}} = \frac{12}{\sqrt{36}} = 2, \qquad Z_1 = -1, \quad Z_2 = 1$$

    $$P(48 < \bar{X} < 52) = \Phi(1) - \Phi(-1) = 0.6827$$

    약 **68.27%**이다. 정규분포에서 $\pm 1$ 표준편차 구간의 확률이라는 익숙한 값이다.

---

## 정리하며

| 성질 | 결과 |
|----------|--------|
| $E[\bar{X}]$ | $\mu$ (불편) |
| $\text{Var}(\bar{X})$ | $\sigma^2/n$ |
| $\text{SE}(\bar{X})$ | $\sigma/\sqrt{n}$ |
| 모양 | 정규 (모집단이 정규이면 정확, $n$이 크면 중심극한정리로 근사) |
| 핵심 통찰 | $n$이 클수록 → 표준오차가 작고 → 추정이 더 정밀하다 |
