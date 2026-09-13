# 검정력 분석과 표본크기

## 개요

검정력 분석은 의미 있는 효과를 지정된 확률로 탐지하는 데 필요한 표본크기를 정한다. 검정의 **검정력**은 $1-\beta$이며, 여기서 $\beta$는 제2종 오류(거짓인 귀무가설을 기각하지 못함)의 확률이다. 잘 설계된 연구는 유의수준 $\alpha$, 원하는 검정력, 관심 있는 최소 효과크기의 균형을 잡아 필요한 관측값 수를 계산한다.

## 핵심 공식

**이표본 t-검정**(집단 크기가 같고 $\sigma$를 아는 경우)에서 집단당 표본크기 $n$일 때 근사적인 검정력은

$$
\text{Power} = 1 - \mathcal{N}\!\left(z_{\alpha/2} - \frac{\delta}{\sigma\sqrt{2/n}}\right) + \mathcal{N}\!\left(-z_{\alpha/2} - \frac{\delta}{\sigma\sqrt{2/n}}\right),
$$

여기서 $\delta = \mu_1 - \mu_2$는 참 차이이고 $\mathcal{N}$은 표준정규 누적분포함수이다.

검정력 $1-\beta$를 달성하는 데 필요한 집단당 표본크기는

$$
n = \frac{2\,(z_{\alpha/2} + z_\beta)^2\,\sigma^2}{\delta^2}.
$$

$\bar{p} = (p_1+p_2)/2$로 $p_1$과 $p_2$를 비교하는 **두 비율 z-검정**에서는:

$$
n = \frac{\left(z_{\alpha/2}\sqrt{2\bar{p}(1-\bar{p})} + z_\beta\sqrt{p_1(1-p_1)+p_2(1-p_2)}\right)^2}{(p_1 - p_2)^2}.
$$

### 이표본 t-검정의 검정력

<div class="codebox" markdown>

#### 예제 1. 이표본 t-검정의 검정력 함수 { .eg }

```python
import numpy as np
from scipy import stats

def power_ttest(n, delta, sigma=1.0, alpha=0.05):
    """양측 이표본 t-검정의 검정력 (정규근사)."""
    # 두 집단의 차이라 분산이 두 번 들어간다. 그래서 sqrt(2/n)이다.
    se = sigma * np.sqrt(2 / n)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    z_effect = delta / se
    # 두 꼬리를 모두 세는 것이 정확하다. 두 번째 항은 참 효과가 양수인데도
    # 통계량이 반대쪽 꼬리로 넘어가 기각되는 경우이며, 보통 무시할 만큼 작다.
    power = (1 - stats.norm.cdf(z_crit - z_effect)
             + stats.norm.cdf(-z_crit - z_effect))
    return power

for n in [20, 50, 100]:
    print(f"n={n:>4} per group: power = {power_ttest(n, delta=0.5):.4f}")
```

출력:

```
n=  20 per group: power = 0.3526
n=  50 per group: power = 0.7054
n= 100 per group: power = 0.9424
```

$d = 0.5$에서 집단당 20명이면 검정력이 0.35에 불과하다. 실제로 효과가 있어도 세 번 중 두 번은 놓친다. 이 값이 정규근사라 $t$-분포를 쓰는 statsmodels의 결과보다 조금 낙관적이라는 점도 염두에 두라. 실제 검정력은 이보다 약간 낮다.

</div>

### 필요한 표본크기

<div class="codebox" markdown>

#### 예제 2. 필요한 표본크기 함수 { .eg }

```python
def sample_size_ttest(delta, sigma=1.0, alpha=0.05, power=0.80):
    """이표본 t-검정의 집단당 최소 표본크기."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    # 앞의 계수 2가 "두 집단"의 대가다. 일표본 공식과 여기서 갈린다.
    n = 2 * ((z_alpha + z_beta) * sigma / delta) ** 2
    return int(np.ceil(n))

def sample_size_proportion(p1, p2, alpha=0.05, power=0.80):
    """이표본 비율검정의 집단당 최소 표본크기."""
    p_bar = (p1 + p2) / 2
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    # 두 항의 제곱근 안이 서로 다르다는 점이 핵심이다.
    #   z_alpha 쪽: H0("두 비율이 같다") 아래의 분산이므로 합동비율 p_bar를 쓴다.
    #   z_beta  쪽: H1 아래의 분산이므로 p1과 p2를 각각 쓴다.
    # 검정력 계산은 두 가설 아래의 분포를 동시에 다루므로 이렇게 섞인다.
    numer = (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar))
             + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    n = numer / (p1 - p2) ** 2
    return int(np.ceil(n))
```

</div>

### 계산 예시

<div class="codebox" markdown>

#### 예제 3. 계산 예 { .eg }

```python
# 이표본 t-검정: 중간 크기 효과 (Cohen's d = 0.5)
delta = 0.5
n_req = sample_size_ttest(delta, sigma=1.0, alpha=0.05, power=0.80)
print(f"Required n per group: {n_req}")

# 비율 검정 (A/B 검정): 전환율 1.10% -> 1.21%, 즉 10% 상대 개선
p1, p2 = 0.0121, 0.011
n_prop = sample_size_proportion(p1, p2, alpha=0.05, power=0.80)
print(f"Required n per group: {n_prop:,}")
```

출력:

```
Required n per group: 63
Required n per group: 148,111
```

두 줄의 차이가 2,000배가 넘는다. 비율 쪽이 이렇게 커지는 이유는 두 가지가 겹쳐서다. 절대차가 0.0011로 아주 작고, 기저율 1.1%가 낮아 신호 대비 잡음이 나쁘다. 전환율을 10% 상대 개선하는 실험을 하려면 집단당 15만 명, 합쳐서 30만 명의 방문자가 필요하다는 뜻이다.

</div>

### 검정력 곡선

<div class="codebox" markdown>

#### 예제 4. 검정력 곡선 { .eg }

```python
import matplotlib.pyplot as plt

# 효과가 작을수록 같은 검정력에 필요한 표본이 가파르게 늘어난다.
# d=0.2 곡선이 0.8 에 닿는 자리를 d=0.8 곡선의 그것과 견주어 보면 된다.
ns = np.arange(10, 500)
fig, ax = plt.subplots(figsize=(10, 5))
for d, ls in [(0.2, '--'), (0.5, '-'), (0.8, ':')]:
    powers = [power_ttest(n, d) for n in ns]
    ax.plot(ns, powers, ls, label=f'd = {d}')

ax.axhline(0.80, color='grey', linestyle='-.', alpha=0.5, label='Power = 0.80')
ax.set_xlabel('Sample size per group (n)')
ax.set_ylabel('Power')
ax.set_title('Power Curves for Two-Sample t-Test')
ax.legend()
plt.tight_layout()
plt.show()
```

![Power Curves for Two-Sample t-Test](./img/power_analysis_114.png)

세 곡선이 회색 기준선(검정력 0.80)을 지나는 지점이 각 효과크기에 필요한 표본크기다. $d = 0.8$은 26 언저리에서, $d = 0.5$는 63에서, $d = 0.2$는 그래프 오른쪽 끝 근처인 393에서 지난다.

곡선의 모양도 읽어 둘 만하다. 검정력 0.9를 넘어서면 곡선이 거의 평평해진다. 그 구간에서는 표본을 더 모아도 얻는 것이 거의 없다. 반대로 $d = 0.2$ 곡선의 왼쪽 절반처럼 가파른 구간에서는 표본을 조금만 늘려도 검정력이 크게 오른다.

</div>

### 해석

- **작은 효과**($d=0.2$)는 검정력 80%를 달성하는 데 집단당 수백 명이 필요하다.
- **중간 효과**($d=0.5$)는 집단당 약 64명이 필요하다.
- **큰 효과**($d=0.8$)는 집단당 약 26명이면 된다.
- 비율 검정에서는 차이 $|p_1-p_2|$가 아주 작으면 수만 개의 관측값이 필요할 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> 어떤 연구자가 $\alpha=0.01$에서 두 집단 사이의 $\delta=0.3$ 표준편차만큼의 차이를 검정력 90%로 탐지하려 한다. 집단당 몇 명이 필요한가?

</div>

??? success "풀이"

    표본크기 공식을 쓴다:

    $$
    n = \frac{2(z_{\alpha/2} + z_\beta)^2}{\delta^2}.
    $$

    여기서 $z_{0.005} = 2.576$, $z_{0.10} = 1.282$, $\delta = 0.3$이므로:

    $$
    n = \frac{2(2.576 + 1.282)^2}{0.3^2} = \frac{2(3.858)^2}{0.09} = \frac{2 \times 14.884}{0.09} = \frac{29.768}{0.09} \approx 331.
    $$

    연구자에게는 집단당 적어도 331명이 필요하다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> ($\sigma$를 아는) 양측 일표본 z-검정에서 $\mu = \mu_0 + \delta$를 탐지하는 검정력을 다음과 같이 쓸 수 있음을 보여라.

$$
1 - \beta = \mathcal{N}\!\left(\frac{\delta\sqrt{n}}{\sigma} - z_{\alpha/2}\right) + \mathcal{N}\!\left(-\frac{\delta\sqrt{n}}{\sigma} - z_{\alpha/2}\right).
$$

</div>

??? success "풀이"

    $H_1\colon \mu = \mu_0 + \delta$ 아래에서 검정통계량 $Z = (\bar{X}-\mu_0)/(\sigma/\sqrt{n})$은 $N(\delta\sqrt{n}/\sigma,\,1)$을 따른다. $\lambda = \delta\sqrt{n}/\sigma$라 하자. 이 검정은 $|Z| > z_{\alpha/2}$일 때 기각하므로

    $$
    1 - \beta = P(Z > z_{\alpha/2}) + P(Z < -z_{\alpha/2}).
    $$

    $Z \sim N(\lambda, 1)$이므로:

    $$
    P(Z > z_{\alpha/2}) = P(Z - \lambda > z_{\alpha/2} - \lambda) = \mathcal{N}(\lambda - z_{\alpha/2}),
    $$

    $$
    P(Z < -z_{\alpha/2}) = P(Z - \lambda < -z_{\alpha/2} - \lambda) = \mathcal{N}(-z_{\alpha/2} - \lambda).
    $$

    둘을 더하면 원하는 결과를 얻는다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span> 어떤 A/B 검정이 전환율 $p_1 = 0.05$와 $p_2 = 0.04$를 비교한다. $\alpha = 0.05$에서 검정력 80%를 위한 집단당 표본크기를 계산하라.

</div>

??? success "풀이"

    $\bar{p} = (0.05 + 0.04)/2 = 0.045$, $z_{0.025}=1.96$, $z_{0.20}=0.842$이므로:

    $$
    n = \frac{\left(1.96\sqrt{2(0.045)(0.955)} + 0.842\sqrt{0.05(0.95) + 0.04(0.96)}\right)^2}{(0.05 - 0.04)^2}.
    $$

    각 부분을 계산하면 $2(0.045)(0.955) = 0.08595$이므로 $\sqrt{0.08595} \approx 0.2932$이다. 또 $0.0475 + 0.0384 = 0.0859$이므로 $\sqrt{0.0859}\approx 0.2931$이다.

    $$
    n = \frac{(1.96 \times 0.2932 + 0.842 \times 0.2931)^2}{0.0001} = \frac{(0.5747 + 0.2468)^2}{0.0001} = \frac{0.6749}{0.0001} \approx 6749.
    $$

    집단당 약 6749명이 필요하다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> Python으로 $\alpha = 0.05$에서 Cohen의 $d \in \{0.2, 0.5, 0.8\}$에 대해 일표본 t-검정의 검정력을 $n$(5부터 200까지)의 함수로 그려라. `statsmodels.stats.power.TTestPower`를 쓰라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.stats.power import TTestPower

    analysis = TTestPower()
    ns = np.arange(5, 201)

    fig, ax = plt.subplots(figsize=(9, 5))
    for d in [0.2, 0.5, 0.8]:
        powers = [analysis.power(effect_size=d, nobs=n, alpha=0.05) for n in ns]
        ax.plot(ns, powers, label=f'd = {d}')

    ax.axhline(0.80, color='grey', linestyle='--', label='80% power')
    ax.set_xlabel('Sample size n')
    ax.set_ylabel('Power')
    ax.set_title('Power Curves (One-Sample t-Test)')
    ax.legend()
    plt.tight_layout()
    plt.show()
    ```

    ![Power Curves (One-Sample t-Test)](./img/power_analysis_219.png)

    효과크기가 클수록 필요한 피험자가 적고, 곡선이 S자 모양으로 올라간다. 가파른 구간과 평평한 구간의 경계가 대략 검정력 0.9 근처다.

    앞의 이표본 곡선과 비교해 보라. 같은 $d$에서 일표본 쪽이 훨씬 왼쪽에 있다. $d = 0.5$에 일표본은 34명, 이표본은 집단당 64명(합계 128명)이 필요하다. 대응설계로 이표본 문제를 일표본 문제로 바꿀 수 있다면 그만큼 큰 이득이다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> 유의수준 $\alpha$와 효과크기 $\delta > 0$이 고정되어 있을 때 $n \to \infty$이면 이표본 z-검정의 검정력이 1에 다가감을 증명하라.

</div>

??? success "풀이"

    검정력은

    $$
    1 - \beta = \mathcal{N}\!\left(\frac{\delta}{\sigma\sqrt{2/n}} - z_{\alpha/2}\right) + \mathcal{N}\!\left(-\frac{\delta}{\sigma\sqrt{2/n}} - z_{\alpha/2}\right).
    $$

    $n \to \infty$이면 항 $\delta / (\sigma\sqrt{2/n}) = \delta\sqrt{n}/({\sigma\sqrt{2}}) \to \infty$이다. 따라서:

    - 첫째 항: $\mathcal{N}(\infty - z_{\alpha/2}) = \mathcal{N}(\infty) = 1$.
    - 둘째 항: $\mathcal{N}(-\infty - z_{\alpha/2}) = \mathcal{N}(-\infty) = 0$.

    그러므로 $1-\beta \to 1$이다. 자료가 충분하면 0이 아닌 어떤 고정된 효과도 결국 탐지됨을 확인해 준다. $\square$

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
검정력이 낮으면 유의한 결과의 **효과크기가 과장되고 부호까지 틀릴 수 있다.** 이를 정량화하라(M-오류와 S-오류).

</div>

??? success "풀이"
    **개념.** 겔먼과 칼린이 제안한 두 지표다.

    - **M-오류(과장 비)**: 유의한 결과만 모았을 때 $E[|\hat\theta|\mid\text{유의}]/|\theta|$.
    - **S-오류(부호 오류율)**: 유의한 결과의 부호가 참 부호와 반대일 확률.

    ```python
    import numpy as np
    from scipy import stats
    from scipy.optimize import brentq

    rng = np.random.default_rng(6)
    M = 400_000
    za = stats.norm.ppf(0.975)

    print(f"{'검정력':>8s} {'θ/SE':>7s} {'과장 비 M':>10s} {'부호 오류 S':>12s}")
    for pw in [0.06, 0.10, 0.20, 0.50, 0.80, 0.95]:
        # 목표 검정력을 주는 참 효과(표준오차 단위)
        nc = brentq(lambda c: stats.norm.sf(za - c)
                    + stats.norm.cdf(-za - c) - pw, 1e-9, 10)
        z = rng.normal(nc, 1, M)
        est = z[np.abs(z) > za]                      # 유의한 결과만
        print(f"{pw:8.0%} {nc:7.3f} {np.mean(np.abs(est)) / nc:10.3f} "
              f"{np.mean(est < 0):12.4f}")
    ```

    ```text
       검정력    θ/SE    과장 비 M    부호 오류 S
         6%   0.295      8.015       0.2030
        10%   0.652      3.710       0.0449
        20%   1.115      2.258       0.0051
        50%   1.960      1.407       0.0001
        80%   2.802      1.125       0.0000
        95%   3.605      1.030       0.0000
    ```

    **검정력 20%인 연구에서 유의하게 나온 효과는 평균적으로 참값의 2.3배**다. 검정력 10%면 3.7배, 6%면 8배다.

    **부호까지 틀릴 수 있다.** 검정력 6%이면 유의한 결과의 **20%가 참 효과와 반대 방향**이다. "여성에게 효과가 있다"가 실제로는 "해롭다"일 수 있다는 뜻이다.

    **왜 그런가.** 유의하려면 $|\hat\theta|>1.96\,\text{SE}$여야 한다. 참 효과가 SE보다 훨씬 작으면, **이 문턱을 넘는 표본은 우연히 크게 나온 것**뿐이다. 선택이 곧 과장을 만든다.

    **검정력 80% 이상이면 문제가 거의 없다.** 과장 비 1.13, 부호 오류 0이다. **이것이 "검정력 80%"라는 관례의 숨은 근거** 중 하나다.

    **실무적 함의.**

    1. **소규모 연구의 "놀라운 발견"을 의심한다.** 효과가 크게 보이는 것 자체가 검정력이 낮다는 증거일 수 있다.

    2. **후속 연구의 표본을 원 연구의 효과크기로 계산하면 안 된다.** 과장된 값을 쓰면 표본이 크게 모자란다. **문헌의 효과크기를 절반으로 할인**하는 것이 안전하다는 제안이 있다.

    3. **메타분석에서 소규모 연구가 큰 효과를 보이는 현상**(소규모 연구 효과)이 여기서 부분적으로 설명된다. 출판 편향과 겹친다.

    4. **설계 단계에서 M-오류를 계산**해 볼 가치가 있다. "이 설계로 유의한 결과가 나온다면 그것을 얼마나 믿을 수 있는가"를 미리 아는 것이다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
**검정력함수**를 그리고, 모수가 $H_0$에서 멀어질수록 어떻게 변하는지 여러 설계에서 비교하라.

</div>

??? success "풀이"
    **검정력함수.** $\pi(\theta)=P_\theta(\text{기각})$. $\theta\in\Theta_0$에서는 수준을, $\theta\in\Theta_1$에서는 검정력을 준다.

    ```python
    import numpy as np
    from scipy import stats

    za = stats.norm.ppf(0.975)
    deltas = np.array([0, 0.2, 0.5, 0.8, 1.0, 1.5])

    def power(d, n, alpha=0.05):
        z = stats.norm.ppf(1 - alpha / 2)
        nc = d * np.sqrt(n)
        return stats.norm.sf(z - nc) + stats.norm.cdf(-z - nc)

    print(f"{'δ':>5s} " + " ".join(f"{'n='+str(n):>9s}"
                                   for n in [10, 25, 50, 100]))
    for d in deltas:
        print(f"{d:5.1f} " + " ".join(f"{power(d, n):9.4f}"
                                      for n in [10, 25, 50, 100]))

    print("\nα 를 바꾸면 (n = 25)")
    print(f"{'δ':>5s} " + " ".join(f"{'α='+str(a):>9s}"
                                   for a in [0.10, 0.05, 0.01, 0.001]))
    for d in deltas:
        print(f"{d:5.1f} " + " ".join(f"{power(d, 25, a):9.4f}"
                                      for a in [0.10, 0.05, 0.01, 0.001]))
    ```

    ```text
        δ      n=10      n=25      n=50     n=100
      0.0    0.0500    0.0500    0.0500    0.0500
      0.2    0.0969    0.1701    0.2930    0.5160
      0.5    0.3526    0.7054    0.9424    0.9988
      0.8    0.7156    0.9793    0.9999    1.0000
      1.0    0.8854    0.9988    1.0000    1.0000
      1.5    0.9973    1.0000    1.0000    1.0000

    α 를 바꾸면 (n = 25)
        δ    α=0.1    α=0.05    α=0.01   α=0.001
      0.0    0.1000    0.0500    0.0100    0.0010
      0.2    0.2636    0.1701    0.0577    0.0110
      0.5    0.8038    0.7054    0.4698    0.2146
      0.8    0.9907    0.9793    0.9228    0.7610
      1.0    0.9996    0.9988    0.9923    0.9563
      1.5    1.0000    1.0000    1.0000    1.0000
    ```

    **네 가지 관찰.**

    1. **$\delta=0$에서 정확히 $\alpha$다.** 검정력함수가 수준을 포함한다.

    2. **U자 모양.** 양측검정이므로 $\delta<0$까지 그리면 $\delta=0$에서 최소인 대칭 U자다. **이 최솟값이 $\alpha$와 같다는 것이 불편성**이다.

    3. **$n$과 $\delta$가 $\delta\sqrt n$으로만 들어온다.** $\delta=0.5$, $n=25$의 검정력(0.705)과 $\delta=0.25$, $n=100$의 검정력이 같다. **검정력은 $\delta\sqrt n$ 하나의 함수**다.

    4. **$\alpha$를 낮추는 비용이 작은 효과에서 크다.** $\delta=0.2$에서 $\alpha$를 0.05에서 0.001로 낮추면 검정력이 0.170에서 0.011로 **15분의 1**이 된다. $\delta=1.0$에서는 0.999에서 0.956으로 거의 손해가 없다.

    **설계에 주는 지침.** 관심 있는 $\delta$ 근처에서 곡선이 **가파르게 오르는 구간**에 설계점을 두어야 한다. 너무 평평한 곳(검정력 0.1)이나 이미 포화된 곳(검정력 0.999)은 자원 낭비다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
검정력을 높이는 방법들을 **효과의 크기 순으로** 정렬하고, 각각의 비용을 적어라.

</div>

??? success "풀이"
    검정력은 $\delta\sqrt n/\sigma$에 의해 결정되므로, 이 값을 키우는 모든 것이 검정력을 높인다.

    ```python
    import numpy as np
    from scipy import stats

    def power(delta, n, sigma=1.0, alpha=0.05, rho=0.0, paired=False):
        se = sigma * np.sqrt(2 * (1 - rho) / n) if paired else sigma * np.sqrt(2 / n)
        z = stats.norm.ppf(1 - alpha / 2)
        nc = delta / se
        return stats.norm.sf(z - nc) + stats.norm.cdf(-z - nc)

    base = power(0.5, 30)
    print(f"기준: δ=0.5, σ=1, 집단당 n=30, α=0.05 → 검정력 {base:.4f}\n")
    cases = [
        ("n 을 2배(60)로", power(0.5, 60)),
        ("n 을 4배(120)로", power(0.5, 120)),
        ("σ 를 20% 줄임", power(0.5, 30, sigma=0.8)),
        ("σ 를 절반으로", power(0.5, 30, sigma=0.5)),
        ("δ 를 1.5배로(설계 강화)", power(0.75, 30)),
        ("α 를 0.10으로", power(0.5, 30, alpha=0.10)),
        ("대응설계 ρ=0.5", power(0.5, 30, paired=True, rho=0.5)),
        ("대응설계 ρ=0.8", power(0.5, 30, paired=True, rho=0.8)),
    ]
    for name, v in cases:
        print(f"{name:24s} {v:.4f}   (+{v - base:.4f})")
    ```

    ```text
    기준: δ=0.5, σ=1, 집단당 n=30, α=0.05 → 검정력 0.4907

    n 을 2배(60)로           0.7819   (+0.2912)
    n 을 4배(120)로          0.9721   (+0.4814)
    σ 를 20% 줄임            0.6775   (+0.1868)
    σ 를 절반으로             0.9721   (+0.4814)
    δ 를 1.5배로(설계 강화)     0.8276   (+0.3369)
    α 를 0.10으로            0.6149   (+0.1242)
    대응설계 ρ=0.5           0.7819   (+0.2912)
    대응설계 ρ=0.8           0.9911   (+0.5004)
    ```

    **효과와 비용의 정리.**

    | 방법 | 효과 | 비용·위험 |
    |---|---|---|
    | **$n$ 늘리기** | 확실하지만 $\sqrt n$ | 비용이 선형으로 증가. 4배 늘려야 효과가 2배 |
    | **$\sigma$ 줄이기** | $n$을 $1/k^2$배 줄이는 것과 동등 | 측정 개선, 표준화, 반복측정. **가장 저렴한 경우가 많다** |
    | **대응·블록화** | $\rho$가 크면 극적 | 설계 제약, 이월효과 위험 |
    | **공변량 보정** | $R^2$만큼 | 사전 계획 필요 |
    | **$\delta$ 키우기** | 직접적 | 용량을 높이면 부작용. **일반화 범위가 좁아짐** |
    | **$\alpha$ 올리기** | 작다 | 제1종 오류 증가. **대개 받아들여지지 않는다** |
    | **단측검정** | 작다 | 방향을 사전에 확정해야 함 |
    | **더 강력한 검정** | 상황에 따라 | 가정 확인 필요 |

    **주목할 점 셋.**

    1. **$\sigma$를 절반으로 줄이는 것이 $n$을 4배로 늘리는 것과 같다.** 측정 도구를 개선하거나 반복측정으로 평균을 내는 것이 **훨씬 싼 경우가 많다.** 그런데 실무에서는 거의 언제나 $n$부터 생각한다.

    2. **대응설계 $\rho=0.8$은 $n$ 4배보다도 낫다**(0.991 대 0.972). 설계의 힘이다.

    3. **$\alpha$를 올리는 것은 효과가 작다.** 0.05에서 0.10으로 두 배 올려도 검정력이 0.491에서 0.615로 오를 뿐이다. **가장 나쁜 거래**다.

    **실무 순서.** ① 측정 정밀도 개선 → ② 설계(대응·블록·공변량) → ③ 표본크기 → ④ 그 밖. 대부분의 연구가 ③에서 시작하는데, 순서가 거꾸로다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
**조건부 검정력**과 **예측 검정력**을 정의하고, 중간분석에서 무익성을 판단하는 데 어떻게 쓰이는지 설명하라.

</div>

??? success "풀이"
    **상황.** 계획 표본의 절반을 모았다. 지금까지의 결과로 "끝까지 가면 유의해질까"를 묻는다.

    **조건부 검정력(CP).** 현재까지의 자료를 조건으로, **특정 참 효과 아래에서** 최종 유의할 확률이다.

    $$
    \text{CP}(\delta)=P\left(\text{최종 유의}\ \middle|\ \text{현재 자료},\ \theta=\delta\right)
    $$

    ```python
    import numpy as np
    from scipy import stats

    za = stats.norm.ppf(0.975)
    t = 0.5                                  # 정보 비율(절반 모음)

    def cp(z_now, delta_over_se_final, t=0.5, za=za):
        """z_now: 현재 시점의 z 통계량 (정보비율 t)"""
        num = za - np.sqrt(t) * z_now - (1 - t) * delta_over_se_final
        return stats.norm.sf(num / np.sqrt(1 - t))

    print(f"{'현재 z':>8s} {'δ=계획값':>10s} {'δ=현재추정':>12s} {'δ=0':>8s}")
    for z_now in [0.0, 0.5, 1.0, 1.5, 2.0]:
        planned = za + stats.norm.ppf(0.90)          # 검정력 90% 설계
        print(f"{z_now:8.1f} {cp(z_now, planned):10.4f} "
              f"{cp(z_now, z_now / np.sqrt(t)):12.4f} {cp(z_now, 0.0):8.4f}")
    ```

    ```text
      현재 z    δ=계획값    δ=현재추정      δ=0
         0.0     0.3157       0.0028   0.0028
         0.5     0.5081       0.0382   0.0115
         1.0     0.6986       0.2201   0.0382
         1.5     0.8462       0.5903   0.1017
         2.0     0.9358       0.8903   0.2201
    ```

    **읽기.**

    - **현재 $z=0$이면** 계획한 효과가 참이어도 최종 유의 확률이 0.32에 불과하다. 절반의 정보를 이미 썼는데 아무 신호가 없기 때문이다.
    - **현재 추정값을 참으로 놓으면** 0.003이다. **거의 확실히 실패**한다.
    - **$\delta=0$ 가정**은 무익성의 최악 시나리오다.

    **예측 검정력(PP).** $\delta$를 하나로 고정하지 않고 **사후분포로 적분**한다.

    $$
    \text{PP}=\int\text{CP}(\delta)\,\pi(\delta\mid\text{현재 자료})\,d\delta
    $$

    - **장점**: $\delta$를 임의로 고르는 문제를 피한다. 현재 자료의 불확실성을 반영한다.
    - **단점**: 사전분포가 필요하다.

    **무익성 중단 규칙.** 흔히 "CP(현재 추정값) $<0.20$이면 중단"을 쓴다. 위 표에서 $z<1.0$이면 중단에 해당한다.

    **주의할 점 넷.**

    1. **어느 $\delta$를 쓰는지가 결정적이다.** 계획값을 쓰면 낙관적, 현재 추정값을 쓰면 비관적, $\delta=0$이면 극단적으로 비관적이다. **미리 정해 문서화**해야 한다.

    2. **검정력이 조금 떨어진다.** 무익성 중단을 넣으면 참 효과가 있는데도 일찍 멈출 확률이 생긴다. 설계 단계에서 이 손실을 반영해 표본을 조금 늘린다.

    3. **$\alpha$에는 영향이 거의 없다.** 무익성 중단은 **기각을 줄이는** 방향이므로 제1종 오류율을 높이지 않는다. 오히려 낮춘다.

    4. **구속력 여부.** 무익성 경계를 "반드시 따른다(binding)"로 하면 $\alpha$를 되찾을 수 있지만, 운영위원회의 재량이 사라진다. 실무에서는 비구속적으로 두는 경우가 많다.

    **윤리적 의미.** 실패할 시험을 계속하면 참가자를 불필요한 위험에 노출시키고 자원을 낭비한다. **무익성 중단은 통계적 장치이기 전에 윤리적 장치**다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
검정력 계산을 **모의실험으로** 하는 법을 보이고, 공식이 없는 상황에서 어떻게 쓰는지 예를 들어라.

</div>

??? success "풀이"
    **원리.** ① 대립가설 아래에서 자료를 생성하고 ② 계획한 분석을 그대로 수행하여 ③ 기각 비율을 센다.

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(101)
    M = 20_000

    def sim_power(gen, analyze, M=M, alpha=0.05):
        return np.mean([analyze(*gen()) < alpha for _ in range(M)])

    # 1) 공식이 있는 경우로 검증
    n, d = 30, 0.5
    gen1 = lambda: (rng.normal(d, 1, n), rng.normal(0, 1, n))
    ana1 = lambda x, y: stats.ttest_ind(x, y, equal_var=False).pvalue
    theo = stats.norm.sf(stats.norm.ppf(0.975) - d / np.sqrt(2 / n))
    print(f"t 검정  모의 {sim_power(gen1, ana1):.4f}   이론(정규근사) {theo:.4f}")

    # 2) 공식이 마땅치 않은 경우: 치우친 자료에 만-휘트니
    def gen2():
        x = rng.lognormal(0.3, 1.0, n)
        y = rng.lognormal(0.0, 1.0, n)
        return x, y
    ana2 = lambda x, y: stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
    ana3 = lambda x, y: stats.ttest_ind(x, y, equal_var=False).pvalue
    print(f"로그정규 — 만-휘트니 {sim_power(gen2, ana2):.4f}   "
          f"웰치 t {sim_power(gen2, ana3):.4f}")
    ```

    ```text
    t 검정  모의 0.4768   이론(정규근사) 0.4906
    로그정규 — 만-휘트니 0.1961   웰치 t 0.1422
    ```

    **첫 줄이 코드의 검증이다.** 모의실험 0.477이 정규근사 0.491에 가깝다. 차이는 $t$ 분포의 꼬리와 자유도를 정규근사가 무시한 데서 오며, $n$을 키우면 사라진다. **모의실험 코드를 짤 때 이런 대조를 반드시 먼저 해 본다.**

    **둘째 줄이 모의실험의 가치다.** 로그정규 자료에서 **만-휘트니가 웰치 $t$보다 검정력이 1.4배** 높다(0.196 대 0.142). 이런 비교는 공식으로 하기 어렵다. 두 방법 모두 검정력이 낮은 것은 로그정규의 변동이 커서 $n=30$으로는 부족하다는 뜻이며, **이 사실을 설계 전에 아는 것 자체가 모의실험의 가치**다.

    **모의실험이 필요한 상황.**

    | 상황 | 왜 공식이 없나 |
    |---|---|
    | 비정규 자료의 $t$/비모수 비교 | 비중심분포가 표준이 아님 |
    | 혼합모형, 반복측정 | 자유도와 상관구조가 복잡 |
    | 다단계·적응 설계 | 중단 규칙이 얽힘 |
    | 결측·탈락이 있는 설계 | 유효 표본이 확률적 |
    | 다중비교 보정 후 검정력 | 검정 간 상관 반영 |
    | 사용자 정의 통계량 | 분포가 알려지지 않음 |

    **설계 시 주의.**

    1. **분석 코드를 그대로 쓴다.** 실제로 쓸 코드를 모의실험에 넣어야 한다. 단순화하면 의미가 없다.
    2. **$M$과 MCSE를 보고한다.** 검정력 0.8 근처에서 $M=10{,}000$이면 MCSE가 0.004다.
    3. **여러 시나리오를 훑는다.** 효과크기, 분산, 결측률, 분포 형태.
    4. **가장 비관적인 시나리오를 반드시 포함**한다. 설계는 최악에 견뎌야 한다.
    5. **씨앗과 코드를 계획서에 첨부**한다.

<div class="drillbox" markdown>

**연습문제 11.** <span class="diff med" title="중간"></span>
검정력 분석을 **보고할 때** 무엇을 적어야 하는지 정리하고, 흔히 빠지는 항목을 지적하라.

</div>

??? success "풀이"
    **적어야 할 것 여덟.**

    1. **주 결과변수와 검정 방법.** 어느 검정의 검정력인가.
    2. **효과크기와 그 근거.** 숫자와 **왜 그 숫자인지**. 이것이 가장 중요하다.
    3. **변동성 가정.** $\sigma$, 기저율, 상관 등과 출처.
    4. **$\alpha$, 단측/양측, 다중비교 보정.**
    5. **목표 검정력.**
    6. **계산 결과.** 필요한 $n$ 또는 주어진 $n$에서의 검정력.
    7. **손실 요인.** 탈락률, 설계효과를 반영한 최종 모집 규모.
    8. **민감도.** 가정이 틀렸을 때의 결과.

    **효과크기의 근거 — 좋은 것과 나쁜 것.**

    | 근거 | 평가 |
    |---|---|
    | 임상적·실무적 최소 중요 차이 | **최선**. 의사결정과 직결 |
    | 선행 메타분석의 요약 효과 | 좋음. 다만 출판 편향에 주의 |
    | 예비연구의 관측 효과 | **위험**. 앞서 본 M-오류로 과장되어 있다 |
    | "코헨의 중간 효과 $d=0.5$" | 근거가 아니다 |
    | 예산에서 역산한 값 | 정직하다면 괜찮다. 다만 그렇게 밝혀야 |

    **흔히 빠지는 항목 다섯.**

    1. **단측/양측 표시.** 이것만으로 필요한 $n$이 20% 이상 달라진다.

    2. **다중비교 보정.** 주 결과가 둘 이상인데 보정 없이 계산한 경우가 흔하다.

    3. **탈락률.** 계산된 $n$을 그대로 모집 목표로 쓰면 부족해진다.

    4. **가정의 출처.** "$\sigma=10$으로 가정했다"만 있고 어디서 온 값인지 없다.

    5. **민감도.** 단 하나의 시나리오만 제시하는 경우가 대부분이다.

    **사후 검정력을 적지 않는다.** 앞서 본 대로 관측된 효과로 계산한 검정력은 $p$-값의 단조변환일 뿐이다. 대신 **관심 있는 효과크기에서의 설계 검정력**을 적는다.

    **좋은 보고의 예.**

    > 주 결과는 12주 시점의 통증 점수(0~100 VAS)다. 선행 메타분석(이 외, 2022, 7개 연구 $n=1{,}104$)에서 표준편차가 18점으로 보고되었다. 임상적으로 의미 있는 최소 차이는 10점이다(환자 대상 앵커 연구 근거). 양측 $\alpha=0.05$, 검정력 90%에서 집단당 69명이 필요하다(비중심 $t$ 분포로 계산). 예상 탈락률 20%를 반영해 집단당 87명, 총 174명을 모집한다.
    >
    > 민감도: 표준편차가 22점이면 집단당 103명(모집 129명)이 필요하다. 표본을 173명으로 고정할 경우 탐지 가능한 최소 차이는 10.9점이다. 계산 코드와 난수 씨앗은 부록 B에 있다.

    **한 문장 원칙.** **검정력 분석의 모든 입력값을 독자가 다시 계산할 수 있어야 한다.** 결과 숫자 하나만 적는 것은 계산을 안 한 것과 크게 다르지 않다.

---

## 정리하며

검정력 분석은 **설계 단계의 도구**다.

- **네 양이 서로 묶여 있다.** $\alpha$, 검정력 $1-\beta$, 효과크기 $\delta/\sigma$, 표본크기 $n$. **셋을 정하면 나머지 하나가 정해진다.**
- **표본크기 계산이 가장 흔한 용도다.** $\alpha=0.05$, 검정력 $0.80$ 을 정하고 의미 있는 최소 효과를 넣어 $n$ 을 푼다.
- **$n$ 이 효과크기의 제곱에 반비례한다.** 효과가 절반이면 표본은 네 배다. 작은 효과를 찾는 연구가 왜 대규모여야 하는지가 여기서 나온다.
- **가장 어려운 입력은 통계가 아니다.** "의미 있는 최소 효과"는 분야 지식이 정하며, 이 값 없이는 계산 자체가 불가능하다.
- **검정력 곡선을 그려 보는 것이 낫다.** 하나의 $n$ 을 고르기보다, 효과크기에 따라 검정력이 어떻게 변하는지 보면 설계의 여유를 판단할 수 있다.

다음 절 **제1종/제2종 오류의 시각화**에서 이 관계를 그림으로 확인한다.
