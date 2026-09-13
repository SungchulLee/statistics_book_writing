# X̄의 표본분포 (Normal)

## 개요

모집단 자체가 정규분포를 따르면 표본평균 $\bar{X}$의 표본분포는 모든 표본크기 $n$에 대해 **정확히** 정규분포이다. 중심극한정리 근사가 필요 없다. 이 페이지에서는 이론과 모의실험으로 이 정확한 결과를 보인다. 정규모집단의 경우는 $t$ 검정과 신뢰구간을 비롯한 많은 고전적 추론 절차의 토대가 된다.

## 모집단 모형

모집단이 표준정규분포를 따른다고 하자:

$$
X \sim N(\mu, \sigma^2) = N(0, 1)
$$

여기서 $\mu = 0$, $\sigma^2 = 1$이다.

## 정확한 표본분포

확률표본 $X_1, X_2, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$에 대해 표본평균은 다음의 정확한 분포를 갖는다:

$$
\bar{X} \sim N\!\left(\mu, \frac{\sigma^2}{n}\right)
$$

!!! info "왜 정확한가"
    독립인 정규확률변수의 선형결합은 그 자체가 정규분포이다. $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$는 i.i.d. 정규확률변수의 선형결합이므로 $\bar{X}$는 정확히 정규분포이다. 점근적 논증이 전혀 필요 없다.

표준화하면:

$$
Z = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \sim N(0, 1)
$$

## 모의실험

다음 코드는 $N(0, 1)$ 모집단에서 크기 $n = 5$인 표본을 10,000개 뽑아 모집단, 하나의 표본, $\bar{X}$의 표본분포를 비교한다.

<div class="codebox" markdown>

### 예제 1. 정규 모집단에서 표본평균의 표집분포 { .eg }

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)

sample_size = 5
n_samples = 10_000
n_population = 10_000

# N(0,1)에서 큰 모집단을 만든다.
population = np.random.normal(loc=0, scale=1, size=n_population)

# 표본을 딱 하나 뽑는다. 현실에서 우리가 실제로 갖게 되는 것이 이것뿐이다.
# 아래 가운데 패널에 점 몇 개로 그려진다.
single_sample = np.random.choice(population, size=sample_size, replace=False)

# 표본을 되풀이해 뽑으며 표본평균을 기록한다. 이 값들의 분포가 표집분포다.
sample_means = [
    np.mean(np.random.choice(population, size=sample_size, replace=False))
    for _ in range(n_samples)
]

# 모집단과 표집분포를 나란히 그린다.
# 세 패널을 sharex=True 로 묶는 것이 이 그림의 핵심 장치다.
# 가로 눈금이 같아야 세 분포의 **퍼짐**을 직접 견줄 수 있다.
#   위   모집단      : 가장 넓다
#   가운데 표본 하나  : 모집단에서 뽑은 점 몇 개
#   아래  표집분포    : 눈에 띄게 좁다. 이 좁아짐이 sigma/sqrt(n) 이다.
fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

ax0.hist(population, bins=100, edgecolor="white")
ax0.set_title("Population Distribution N(0, 1)")

ax1.scatter(single_sample, np.zeros_like(single_sample), s=100)
ax1.set_title(f"Sample Distribution (n = {sample_size})")

ax2.hist(sample_means, bins=100, edgecolor="white")
ax2.set_title("Sampling Distribution of X-bar")

plt.tight_layout()
plt.show()
```

</div>

## 해석

!!! note "주요 관찰"

    1. **모집단 분포**는 종 모양(정규)이다.
    2. $\bar{X}$의 **표본분포**도 종 모양이며 같은 평균 $\mu = 0$을 중심으로 한다.
    3. 표본분포는 모집단보다 $\sqrt{n}$배 **좁다**. $n = 5$이면 $\sigma = 1$인 데 비해 표준오차가 $\sigma/\sqrt{5} \approx 0.447$이다.
    4. Uniform이나 Exponential의 경우와 달리 여기서 표본분포의 정규성은 근사가 아니라 **정확**하다.

### 퍼짐의 비교

| 분포 | 표준편차 |
|---|---|
| 모집단 $X$ | $\sigma = 1$ |
| $n = 5$일 때 $\bar{X}$ | $\sigma / \sqrt{5} \approx 0.447$ |
| $n = 25$일 때 $\bar{X}$ | $\sigma / \sqrt{25} = 0.200$ |
| $n = 100$일 때 $\bar{X}$ | $\sigma / \sqrt{100} = 0.100$ |

## 정확한 정규성의 증명

<div class="thmbox" markdown>

### 정리 1. { .thm }

$X_1, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$이면 $\bar{X} \sim N(\mu, \sigma^2/n)$이다.

</div>

??? proof "증명"

    $X_i$의 적률생성함수(MGF)는:

    $$
    M_{X_i}(t) = \exp\!\left(\mu t + \frac{\sigma^2 t^2}{2}\right)
    $$

    $X_i$들이 독립이므로:

    $$
    M_{S_n}(t) = \prod_{i=1}^n M_{X_i}(t) = \exp\!\left(n\mu t + \frac{n\sigma^2 t^2}{2}\right)
    $$

    $\bar{X} = S_n / n$의 MGF는:

    $$
    M_{\bar{X}}(t) = M_{S_n}(t/n) = \exp\!\left(\mu t + \frac{\sigma^2 t^2}{2n}\right)
    $$

    이는 $N(\mu, \sigma^2/n)$의 MGF이다. MGF가 분포를 유일하게 결정하므로 $\bar{X} \sim N(\mu, \sigma^2/n)$이다. $\square$

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> $X_1, \ldots, X_{25} \overset{\text{iid}}{\sim} N(100, 16)$일 때 $P(\bar{X} > 102)$를 구하라.

</div>

??? success "풀이"
    여기서 $\mu = 100$, $\sigma^2 = 16$, $n = 25$이다.

    $$
    \bar{X} \sim N\!\left(100, \frac{16}{25}\right) = N(100,\; 0.64)
    $$

    표준화하면:

    $$
    Z = \frac{102 - 100}{\sqrt{0.64}} = \frac{2}{0.8} = 2.5
    $$

    $$
    P(\bar{X} > 102) = P(Z > 2.5) = 1 - \mathcal{N}(2.5) \approx 1 - 0.9938 = 0.0062
    $$

    $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> $X \sim N(\mu_X, \sigma_X^2)$와 $Y \sim N(\mu_Y, \sigma_Y^2)$가 독립이면 $aX + bY \sim N(a\mu_X + b\mu_Y,\; a^2\sigma_X^2 + b^2\sigma_Y^2)$임을 증명하라.

</div>

??? success "풀이"
    $W = aX + bY$라 하자. $W$의 MGF는:

    $$
    M_W(t) = E[e^{t(aX + bY)}] = E[e^{taX}] \cdot E[e^{tbY}]
    $$

    이며 인수분해에는 독립성을 사용했다. 정규분포의 MGF를 대입하면:

    $$
    M_W(t) = \exp\!\left(a\mu_X t + \frac{a^2\sigma_X^2 t^2}{2}\right) \cdot \exp\!\left(b\mu_Y t + \frac{b^2\sigma_Y^2 t^2}{2}\right)
    $$

    $$
    = \exp\!\left((a\mu_X + b\mu_Y)t + \frac{(a^2\sigma_X^2 + b^2\sigma_Y^2)t^2}{2}\right)
    $$

    이는 $N(a\mu_X + b\mu_Y,\; a^2\sigma_X^2 + b^2\sigma_Y^2)$의 MGF이다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span> 어떤 기계가 병에 평균 500 ml, 표준편차 4 ml로 내용물을 채우며 충전량은 정규분포를 따른다. 품질관리에서 병 16개를 표본으로 뽑는다. 표본평균이 목표치에서 2 ml 이내일 확률은?

</div>

??? success "풀이"
    $n = 16$이고 $X_i \sim N(500, 16)$이라 하자.

    $$
    \bar{X} \sim N\!\left(500, \frac{16}{16}\right) = N(500, 1)
    $$

    $Z = (\bar{X} - 500)/1$일 때 $P(498 < \bar{X} < 502) = P(-2 < Z < 2)$를 구하면 된다.

    $$
    P(-2 < Z < 2) = \mathcal{N}(2) - \mathcal{N}(-2) = 2\mathcal{N}(2) - 1 \approx 2(0.9772) - 1 = 0.9544
    $$

    표본평균이 목표치에서 2 ml 이내일 확률은 약 95.44%이다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> 모집단이 정규일 때는 중심극한정리가 필요 없지만 모집단이 Exponential이나 Uniform일 때는 필수적인 이유를 설명하라. 정규가 아닌 경우 $n$이 커지면 표본분포에서 무엇이 달라지는가?

</div>

??? success "풀이"
    모집단이 정규이면 독립인 정규확률변수의 선형결합이 정규이므로 모든 $n$에서 $\bar{X}$가 정확히 정규분포이다. 이는 정규분포 MGF의 직접적인 성질(동등하게, 합성곱에 대해 닫혀 있다는 성질)이다.

    정규가 아닌 모집단(예: Exponential, Uniform)에서는 유한한 $n$에 대해 $\bar{X}$가 정확히 정규분포가 **아니다**. 그 분포는 $n$과 구체적인 모집단 모양에 의존한다. 다만 $n \to \infty$일 때 중심극한정리가 표준화된 $\bar{X}$의 $N(0,1)$로의 분포수렴을 보장한다.

    정규가 아닌 모집단에서 $n$이 커지면:

    - 표본분포가 더 대칭이 된다(왜도가 $\gamma_1/\sqrt{n}$로 감소한다).
    - 초과첨도가 ($1/n$의 비율로) 0을 향해 줄어든다.
    - 모양이 점점 정규분포에 가까워진다.

    수렴 속도는 모집단이 얼마나 "정규가 아닌지"에 달려 있다. 심하게 치우쳤거나 꼬리가 두꺼운 모집단은 더 큰 $n$을 요구한다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> 위의 모의실험 코드에서 $n$을 5에서 100으로 늘려라. 히스토그램 위에 이론적 밀도 $N(0, 1/100)$을 겹쳐 그리고, 10,000개 표본평균의 경험적 표준편차가 $1/\sqrt{100} = 0.1$에 가까운지 확인하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    np.random.seed(1)
    population = np.random.normal(loc=0, scale=1, size=10_000)

    sample_means = [
        np.mean(np.random.choice(population, size=100, replace=False))
        for _ in range(10_000)
    ]

    empirical_se = np.std(sample_means)
    theoretical_se = 1 / np.sqrt(100)

    print(f"Theoretical SE: {theoretical_se:.4f}")
    print(f"Empirical SE:   {empirical_se:.4f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(sample_means, bins=60, density=True, alpha=0.5, label="Simulated")
    x = np.linspace(-0.5, 0.5, 200)
    ax.plot(x, stats.norm.pdf(x, 0, theoretical_se), "r--", lw=2,
            label=f"N(0, {theoretical_se**2:.4f})")
    ax.legend()
    ax.set_title("Sampling Distribution of X-bar (n = 100)")
    plt.show()
    ```

    출력:

    ```
    Theoretical SE: 0.1000
    Empirical SE:   0.0993
    ```

    ![Sampling Distribution of X-bar (n = 100)](./img/xbar_normal_206.png)

    경험적 표준오차가 0.1에 가깝게 나오고 히스토그램이 $N(0, 0.01)$ 밀도와 사실상 완벽하게 일치하여 정확한 정규성 결과를 확인해 준다. $\square$

---

## 정리하며

모집단이 정규이면 $\bar X\sim N(\mu,\sigma^2/n)$ 이 **모든 $n$ 에서 정확**하다. 근사가 아니다.

- **이유는 정규분포가 덧셈에 대해 닫혀 있기 때문이다.** 독립인 정규의 합이 다시 정규이며, 적률생성함수를 곱해 보면 한 줄로 나온다(3장).
- **$n=2$ 에서도 성립한다.** 중심극한정리가 "$n$ 이 커지면"이라고 말하는 것과 달리 여기에는 조건이 없다.
- **이것이 고전적 추론의 토대다.** $t$ 검정, $t$ 신뢰구간, 분산분석이 모두 정규모집단 가정 위에서 정확한 분포를 갖는다. 그 가정이 깨지면 모든 것이 근사로 내려앉는다.
- **$\bar X$ 와 $S^2$ 이 독립**이라는 성질도 정규분포에서만 성립한다. 이것이 $t$ 통계량의 분자와 분모가 독립이 되는 근거이며, 다른 어떤 분포도 이 성질을 갖지 않는다.

다음 절 **베르누이 모집단**으로 넘어간다. 모집단이 $0$ 과 $1$ 뿐인 극단적인 경우이며, 표본평균이 곧 표본비율이 된다.
