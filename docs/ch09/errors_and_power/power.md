# 검정력 분석

## 검정력의 정의

가설검정의 **검정력**은 거짓인 귀무가설을 올바르게 기각할 확률이다. 제2종 오류율의 여집합이다:

$$\text{Power} = 1 - \beta = P(\text{Reject } H_0 \mid H_a \text{ is true})$$

검정력이 높은 검정은 참 효과가 있을 때 그것을 탐지할 가능성이 크다. 연구자들은 보통 검정력 0.80(80%) 이상을 목표로 하며, 이는 참 효과를 탐지할 확률이 80%라는 뜻이다.

## 검정력에 영향을 주는 요인

검정력을 결정하는 핵심 요인은 넷이다.

### 1. 유의수준 (alpha)
$\alpha$를 키우면(예: 0.01에서 0.05로) $H_0$을 기각하기 쉬워져 검정력이 커진다. 그러나 제1종 오류의 위험도 함께 커진다.

### 2. 표본크기 (n)
표본이 클수록 검정통계량의 표준오차가 줄어 참 차이를 탐지하기 쉬워지므로 검정력이 커진다. 실무에서 검정력을 높이는 가장 현실적인 지렛대이다.

### 3. 효과크기

효과크기는 참 차이 또는 효과의 크기를 잰다. 효과가 클수록 탐지하기 쉬워 검정력이 커진다. 흔한 측도는:

- 평균 비교의 **Cohen의 $d$**: $d = \frac{\mu_1 - \mu_0}{\sigma}$
- 비율 비교의 **비율 차이**

### 4. 모집단의 변동성 (sigma)
모집단의 변동성이 작을수록 참 효과를 탐지하기 쉬워 검정력이 커진다. 연구자가 모집단 변동성을 통제하기는 대개 어렵지만, 더 나은 연구 설계로 측정오차는 줄일 수 있다.

## 실무에서의 검정력 분석

검정력 분석은 주로 두 가지 방식으로 쓰인다.

### 사전 검정력 분석 (표본크기의 결정)

연구를 수행하기 전에, 예상되는 효과크기를 원하는 검정력과 유의수준으로 탐지하는 데 필요한 최소 표본크기를 검정력 분석으로 정한다.

<div class="codebox" markdown>

#### 예제 1. 필요한 표본크기 구하기 { .eg }

```python
from scipy import stats
import numpy as np

def sample_size_z_test(effect_size, alpha=0.05, power=0.80, alternative='two-sided'):
    """일표본 z-검정에 필요한 표본크기."""
    if alternative == 'two-sided':
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    # z_beta는 ppf(power)이지 ppf(1 - power)가 아니다.
    # 검정력이 클수록 z_beta가 커져 n이 늘어나야 하기 때문이다.
    z_beta = stats.norm.ppf(power)
    # 두 임계값을 **더한다**. alpha는 H0 분포의 오른쪽 꼬리에서,
    # beta는 Ha 분포의 왼쪽 꼬리에서 재므로 둘 사이의 거리가 두 값의 합이다.
    n = ((z_alpha + z_beta) / effect_size) ** 2
    return int(np.ceil(n))

# 효과크기 0.5를 검정력 80%로 탐지하려면?
n = sample_size_z_test(effect_size=0.5)
print(f"Required sample size: {n}")
```

출력:

```
Required sample size: 32
```

$(1.96 + 0.8416)^2 / 0.5^2 = 31.4$를 올림한 값이다. 효과크기가 분모에서 제곱되므로 효과가 절반이면 표본은 네 배가 된다.

</div>

### 사후 검정력 분석

연구를 마친 뒤 관측된 효과크기, 표본크기, 유의수준으로 달성된 검정력을 계산할 수 있다. 그러나 유의하지 않은 결과에 대한 사후 검정력 분석은 p-값을 넘는 정보를 거의 주지 않으므로 일반적으로 권장되지 않는다.

## 검정력의 시각화

$H_0$과 $H_a$ 아래의 분포를 함께 그리면 검정력을 이해할 수 있다.

<div class="codebox" markdown>

### 예제 2. 검정력을 그림으로 보기 { .eg }

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_power(mu_0, mu_a, sigma, n, alpha=0.05):
    """단측 z 검정의 검정력을 그림으로 보인다.

        귀무분포와 대립분포를 겹쳐 그리고, 기각역에 해당하는 두 넓이를 칠한다.
        귀무 쪽 넓이가 alpha, 대립 쪽 넓이가 검정력이다.
        """
    se = sigma / np.sqrt(n)
    z_crit = stats.norm.ppf(1 - alpha)
    x_crit = mu_0 + z_crit * se

    x = np.linspace(mu_0 - 4*se, mu_a + 4*se, 300)

    # 귀무가설이 참일 때의 분포
    y_h0 = stats.norm(mu_0, se).pdf(x)
    # 대립가설이 참일 때의 분포. 중심만 오른쪽으로 옮겨 간다.
    y_ha = stats.norm(mu_a, se).pdf(x)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, y_h0, 'b-', label=f'$H_0$: $\\mu = {mu_0}$')
    ax.plot(x, y_ha, 'r-', label=f'$H_a$: $\\mu = {mu_a}$')

    # 귀무 분포에서 기각역에 해당하는 꼬리. 그 넓이가 유의수준 alpha 다.
    x_reject = x[x >= x_crit]
    ax.fill_between(x_reject, stats.norm(mu_0, se).pdf(x_reject), alpha=0.3, color='blue', label=f'$\\alpha$ = {alpha}')

    # 대립 분포에서 같은 기각역의 넓이. 그것이 검정력이다.
    ax.fill_between(x_reject, stats.norm(mu_a, se).pdf(x_reject), alpha=0.3, color='red', label=f'Power = {1 - stats.norm(mu_a, se).cdf(x_crit):.3f}')

    ax.axvline(x_crit, color='k', linestyle='--', label=f'Critical value = {x_crit:.2f}')
    ax.legend()
    ax.set_xlabel('Sample Mean')
    ax.set_ylabel('Density')
    ax.set_title('Power of a Hypothesis Test')
    plt.show()

plot_power(mu_0=50, mu_a=52, sigma=10, n=25)
```

![Power of a Hypothesis Test](./img/power_66.png)

파란 곡선이 $H_0$ 아래, 빨간 곡선이 $H_a$ 아래의 $\bar X$ 분포다. 검은 점선이 임계값 53.29이고, 그 오른쪽의 파란 영역이 $\alpha = 0.05$, 빨간 영역이 검정력이다.

이 설정에서 검정력은 0.259밖에 안 된다. 참 평균이 정말 52인데도 네 번 중 세 번은 $H_0$을 기각하지 못한다는 뜻이다. 두 곡선이 겹치는 정도가 곧 검정의 무력함이다. $n$을 키우면 두 곡선이 모두 좁아지면서 겹침이 줄고 빨간 영역이 커진다.

</div>

## 검정력, 표본크기, 효과크기의 관계

| 효과크기 | 필요한 $n$ (이표본, 집단당, 검정력 = 0.80, $\alpha$ = 0.05) |
|---|---|
| 작음 ($d = 0.2$) | ~393 |
| 중간 ($d = 0.5$) | ~64 |
| 큼 ($d = 0.8$) | ~26 |

이 값들은 작은 효과를 탐지하려면 왜 훨씬 큰 표본이 필요한지 보여준다.

## statsmodels를 이용한 검정력 분석

Statsmodels는 여러 검정 유형에 대한 검정력 분석 함수를 폭넓게 제공한다.

### 일표본 t-검정

<div class="codebox" markdown>

#### 예제 3. 일표본 t-검정의 검정력 { .eg }

```python
from statsmodels.stats.power import TTestPower
import numpy as np

analysis = TTestPower()

# 5점 차이를 탐지하려면 몇 명이 필요한가? (mu_0 = 0, mu_a = 5, sigma = 15)
# 검정력 계산에 들어가는 것은 delta도 sigma도 아니라 그 비뿐이다.
# 5점/15와 1점/3은 검정력 관점에서 완전히 같은 문제다.
effect_size = 5 / 15  # Cohen's d = 0.333

n_needed = analysis.solve_power(effect_size=effect_size, alpha=0.05,
                                power=0.80, alternative='two-sided')
print(f"One-sample t-test:")
print(f"  Effect size (Cohen's d): {effect_size:.3f}")
print(f"  Sample size needed for 80% power: {int(np.ceil(n_needed))}")

# 반대 방향의 질문: n이 정해져 있을 때 검정력은 얼마인가?
power = analysis.power(effect_size=effect_size, nobs=50, alpha=0.05,
                      alternative='two-sided')
print(f"  Power with n=50: {power:.3f}")
```

출력:

```
One-sample t-test:
  Effect size (Cohen's d): 0.333
  Sample size needed for 80% power: 73
  Power with n=50: 0.637
```

73명이 필요한데 50명만 모으면 검정력이 0.80에서 0.64로 떨어진다. 표본을 32% 줄였을 뿐인데 효과를 놓칠 확률은 20%에서 36%로 거의 두 배가 된다.

</div>

### 이표본 t-검정 (독립표본)

<div class="codebox" markdown>

#### 예제 4. 이표본 t-검정의 검정력 { .eg }

```python
from statsmodels.stats.power import TTestIndPower

# 두 집단을 견주는 t-검정의 검정력 분석
analysis = TTestIndPower()

# 두 집단의 크기를 같게 두는 설계다. 같은 총인원이면 이때 검정력이 가장 높다.
# 효과크기 Cohen 의 d = 0.5. 두 평균이 표준편차의 절반만큼 떨어진 경우다.
effect_size = 0.5

n_per_group = analysis.solve_power(effect_size=effect_size, alpha=0.05,
                                   power=0.80, ratio=1.0,
                                   alternative='two-sided')
print(f"\nTwo-sample t-test (equal n):")
print(f"  Effect size (Cohen's d): {effect_size:.3f}")
print(f"  Sample size per group for 80% power: {int(np.ceil(n_per_group))}")
print(f"  Total sample size: {2 * int(np.ceil(n_per_group))}")

# 집단 크기가 다른 경우 (예: 2:1 배분)
# ratio는 nobs2/nobs1이고, solve_power가 돌려주는 것은 **nobs1**이다.
# 따라서 두 번째 집단은 ratio를 곱해서 얻어야 한다. 나누면 거꾸로다.
ratio = 2.0
n_group1 = analysis.solve_power(effect_size=effect_size, alpha=0.05,
                                power=0.80, ratio=ratio,
                                alternative='two-sided')
n_group2 = n_group1 * ratio
n1, n2 = int(np.ceil(n_group1)), int(np.ceil(n_group2))
print(f"\nTwo-sample t-test (2:1 ratio):")
print(f"  Group 1 n: {n1}")
print(f"  Group 2 n: {n2}")
print(f"  Total sample size: {n1 + n2}")
```

출력:

```

Two-sample t-test (equal n):
  Effect size (Cohen's d): 0.500
  Sample size per group for 80% power: 64
  Total sample size: 128

Two-sample t-test (2:1 ratio):
  Group 1 n: 48
  Group 2 n: 96
  Total sample size: 144
```

배분이 균등에서 멀어질수록 **총** 표본이 늘어난다. 같은 검정력에 1:1은 128명, 1:2는 144명이 든다. 한쪽 집단을 모으기 쉽다고 해서 그쪽만 키우면 전체 비용이 오히려 커질 수 있다. 작은 쪽 집단이 병목이기 때문이다.

</div>

### 비율에 대한 검정 (A/B 검정)

<div class="codebox" markdown>

#### 예제 5. 비율 검정의 검정력 — A/B 검정 { .eg }

```python
import numpy as np
import statsmodels.stats.api as sms
from statsmodels.stats.power import NormalIndPower

# 전환율을 견주는 A/B 검정
# 대조군 전환율 1.1%
# 실험군 전환율 1.65%. 상대적으로는 50% 개선이지만 절대차는 0.55%p 다.
p0 = 0.011   # Control baseline
p1 = 0.0165  # Treatment goal

# 비율에서는 두 값의 차이 대신 arcsin 변환 후의 차이를 효과크기로 쓴다.
#   h = 2*arcsin(sqrt(p1)) - 2*arcsin(sqrt(p0)) — 비율을 각도로 바꿔 재는 효과크기다.
# 이렇게 하면 분산이 p에 의존하는 문제가 사라져 하나의 공식으로 처리된다.
# 0.011 대 0.0165는 절대차로 0.55%p뿐이지만 상대적으로는 50% 증가다.
effect_size = sms.proportion_effectsize(p1, p0)

# alternative='larger'는 단측이다. A/B 검정에서는 "개선되었는가"만 보는 일이 많다.
analysis = NormalIndPower()
n_ab = analysis.solve_power(effect_size=effect_size, alpha=0.05,
                            power=0.80, ratio=1.0, alternative='larger')

print(f"\nA/B Test (Proportions):")
print(f"  Control rate: {p0:.2%}")
print(f"  Treatment goal: {p1:.2%}")
print(f"  Effect size: {effect_size:.4f}")
print(f"  Sample size per group for 80% power: {int(np.ceil(n_ab))}")
```

출력:

```

A/B Test (Proportions):
  Control rate: 1.10%
  Treatment goal: 1.65%
  Effect size: 0.0475
  Sample size per group for 80% power: 5488
```

집단당 5,488명, 합쳐서 약 11,000명이 필요하다. 전환율이 낮으면 표본이 이렇게 커진다. 1.1%의 기저율에서는 집단당 5,488명이라도 전환이 60건 남짓에 불과하기 때문이다. 웹 실험이 몇 주씩 걸리는 이유가 여기 있다.

</div>

### 일원분산분석

<div class="codebox" markdown>

#### 예제 6. 일원분산분석의 검정력 { .eg }

```python
from statsmodels.stats.power import FTestAnovaPower

# 분산분석의 효과크기는 Cohen 의 f 다. t-검정의 d 와는 눈금이 다르므로
# 0.25 를 d 로 읽으면 안 된다. f 는 0.10 작음, 0.25 중간, 0.40 큼으로 본다.
analysis = FTestAnovaPower()

effect_size = 0.25   # 중간 크기 효과
k_groups = 4         # 비교할 집단 수

n_per_group = analysis.solve_power(effect_size=effect_size, alpha=0.05,
                                   power=0.80, k_groups=k_groups)
print(f"\nOne-way ANOVA (4 groups):")
print(f"  Effect size (Cohen's f): {effect_size:.3f}")
print(f"  Sample size per group for 80% power: {int(np.ceil(n_per_group))}")
print(f"  Total sample size: {k_groups * int(np.ceil(n_per_group))}")
```

출력:

```

One-way ANOVA (4 groups):
  Effect size (Cohen's f): 0.250
  Sample size per group for 80% power: 179
  Total sample size: 716
```

주의할 점은 Cohen의 $f$와 $d$가 다른 척도라는 것이다. $f = 0.25$는 ANOVA에서 "중간"이지만 $d = 0.5$와 같은 뜻이 아니다. 두 집단만 있을 때 $f = d/2$이므로 $f = 0.25$는 $d = 0.5$에 대응한다. 그런데도 집단당 179명이 필요한 것은 집단이 넷이라 비교해야 할 것이 많아졌기 때문이다.

</div>

### 검정력 곡선: 표본크기와 검정력의 관계

<div class="codebox" markdown>

#### 예제 7. 표본크기와 검정력의 곡선 { .eg }

```python
import matplotlib.pyplot as plt

# 효과크기를 세 가지로 두고 표본크기에 따른 검정력을 그린다.
# 세 곡선이 모두 위로 볼록하다는 점이 중요하다. 표본을 늘릴수록 얻는 것이
# 줄어들므로, 검정력 0.8 을 0.9 로 올리는 비용이 0.5 를 0.8 로 올리는 비용보다 크다.
fig, ax = plt.subplots(figsize=(10, 6))

analysis = TTestPower()
sample_sizes = np.arange(10, 200, 5)

for d in [0.2, 0.5, 0.8]:
    power_values = [analysis.power(effect_size=d, nobs=n, alpha=0.05)
                    for n in sample_sizes]
    ax.plot(sample_sizes, power_values, linewidth=2, label=f"d = {d:.1f}")

# 관례로 쓰는 두 기준선. 0.80 이 가장 흔하다.
ax.axhline(0.80, color='red', linestyle='--', linewidth=1, label='Power = 0.80')
ax.axhline(0.90, color='orange', linestyle='--', linewidth=1, label='Power = 0.90')

ax.set_xlabel('Sample Size (n)', fontsize=12)
ax.set_ylabel('Power', fontsize=12)
ax.set_title('Power Curves: Sample Size vs. Power\n(One-Sample t-Test, α = 0.05)')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1])

plt.tight_layout()
plt.show()
```

![Power Curves: Sample Size vs. Power](./img/power_224.png)

세 곡선 모두 처음에는 가파르게 오르다가 위로 갈수록 평평해진다. 이 평평해지는 구간이 실무에서 중요하다. $d = 0.5$에서 검정력 0.8에는 34명이면 되지만 0.9에는 44명이 필요하다. 30% 더 모아서 10%p를 얻는 셈이고, 0.95를 원하면 54명으로 더 늘어난다. 마지막 몇 %p가 가장 비싸다.

세 곡선의 간격도 눈여겨보라. $d = 0.8$은 15명이면 되고 $d = 0.5$는 34명, $d = 0.2$는 199명이다. 효과크기가 4분의 1로 줄면 표본은 열세 배가 된다. 작은 효과를 탐지하는 일은 표본이 곧 예산이다.

</div>

### 검정력 분석의 작업 흐름

<div class="codebox" markdown>

#### 예제 8. 연구 설계 작업 흐름 { .eg }

```python
def design_study(test_type, effect_size, alpha=0.05, power=0.80,
                 **kwargs):
    """연구 설계를 위한 검정력 분석을 한자리에 모은 함수.

    검정 종류마다 statsmodels의 클래스가 다르고 인자 이름도 다르다.
    그 차이를 여기서 흡수해 두면 설계 단계에서 시나리오를 바꿔 가며
    표본크기를 비교하기가 쉬워진다.

    test_type : 'one-sample', 'two-sample', 'anova', 'proportions'
    effect_size : 표준화된 효과크기 (종류마다 척도가 다르다: d, d, f)
    alpha : 유의수준
    power : 목표 검정력
    **kwargs : 검정별 추가 인자 (예: ANOVA의 k_groups)
    """
    results = {'test_type': test_type, 'alpha': alpha, 'power': power}

    if test_type == 'one-sample':
        analysis = TTestPower()
        n = analysis.solve_power(effect_size=effect_size, alpha=alpha,
                                power=power, alternative='two-sided')
        results['sample_size'] = int(np.ceil(n))

    elif test_type == 'two-sample':
        analysis = TTestIndPower()
        n = analysis.solve_power(effect_size=effect_size, alpha=alpha,
                                power=power, ratio=1.0, alternative='two-sided')
        results['sample_size_per_group'] = int(np.ceil(n))
        results['total_sample_size'] = 2 * int(np.ceil(n))

    elif test_type == 'anova':
        analysis = FTestAnovaPower()
        k = kwargs.get('k_groups', 3)
        n = analysis.solve_power(effect_size=effect_size, alpha=alpha,
                                power=power, k_groups=k)
        results['k_groups'] = k
        results['sample_size_per_group'] = int(np.ceil(n))
        results['total_sample_size'] = k * int(np.ceil(n))

    return results

# 사용 예
print("\n" + "="*60)
print("STUDY DESIGN: Two-Sample Comparison")
print("="*60)
design = design_study('two-sample', effect_size=0.5)
for key, value in design.items():
    print(f"{key:.<30} {value}")
```

출력:

```

============================================================
STUDY DESIGN: Two-Sample Comparison
============================================================
test_type..................... two-sample
alpha......................... 0.05
power......................... 0.8
sample_size_per_group......... 64
total_sample_size............. 128
```

이런 표를 연구계획서에 그대로 옮겨 적을 수 있다. 검정력 분석에서 정작 어려운 부분은 계산이 아니라 `effect_size`에 넣을 값을 정하는 일이다. 선행 연구, 예비조사, 또는 "이보다 작으면 실무적으로 의미가 없다"는 기준 중 하나를 근거로 삼아야 하며, 그 근거를 함께 적어 두는 것이 좋다.

</div>

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
어떤 검정의 $\alpha = 0.05$이고 검정력 $= 0.80$이다. 제1종 오류, 제2종 오류, 올바른 기각, 올바른 비기각의 확률은 각각 얼마인가?

</div>

??? success "풀이"

    - **제1종 오류** ($\alpha$): $P(\text{reject } H_0 \mid H_0 \text{ true}) = 0.05$
    - **제2종 오류** ($\beta$): $P(\text{fail to reject } H_0 \mid H_0 \text{ false}) = 1 - \text{power} = 1 - 0.80 = 0.20$
    - **올바른 기각(검정력)**: $P(\text{reject } H_0 \mid H_0 \text{ false}) = 0.80$
    - **올바른 비기각**: $P(\text{fail to reject } H_0 \mid H_0 \text{ true}) = 1 - \alpha = 0.95$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
어떤 연구자가 $\alpha = 0.05$(양측, 이표본 $t$-검정)에서 효과크기 $d = 0.3$을 검정력 90%로 탐지하려 한다. 공식 $n = 2(z_{\alpha/2} + z_\beta)^2/d^2$으로 집단당 필요한 표본크기를 추정하라.

</div>

??? success "풀이"
    $\alpha = 0.05$이면 $z_{0.025} = 1.96$이다. 검정력 $= 0.90$이면 $\beta = 0.10$이므로 $z_{0.10} = 1.282$이다.

    $$
    n = \frac{2(1.96 + 1.282)^2}{0.3^2} = \frac{2(3.242)^2}{0.09} = \frac{2 \times 10.511}{0.09} = \frac{21.022}{0.09} \approx 233.6
    $$

    올림하면 집단당 $n = 234$(총 468)이다. 작은 효과를 높은 검정력으로 탐지하려면 상당한 표본이 필요하다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
검정의 검정력을 높이는 방법 네 가지를 들라. 보통 어느 것이 가장 현실적인가?

</div>

??? success "풀이"

    1. **표본크기 $n$을 늘린다**: 자료가 많아지면 표준오차가 줄어 참 효과를 탐지하기 쉬워진다. 보통 가장 현실적인 방법이다.
    2. **$\alpha$를 키운다**: 유의수준을 덜 엄격하게 하면(예: 0.05 대신 0.10) 검정력이 커지지만 제1종 오류율도 커진다.
    3. **효과크기를 키운다**: 참 차이가 클수록 탐지하기 쉽다. 보통 연구자가 통제할 수 없지만, 더 나은 실험 설계(예: 더 극단적인 처리)로 어느 정도 가능할 때도 있다.
    4. **변동성 $\sigma$를 줄인다**: 더 정밀한 측정이나 더 동질적인 표본은 $\sigma$를 줄여 신호 대 잡음 비를 높인다. 더 나은 측정기기, 교란요인의 통제, 대응 설계 등으로 달성할 수 있다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
표본크기가 고정되어 있을 때 제1종 오류($\alpha$)와 제2종 오류($\beta$) 사이의 맞바꿈을 설명하라. 둘을 동시에 얼마든지 작게 만들 수 없는 이유는?

</div>

??? success "풀이"
    표본크기와 효과크기가 고정되어 있으면 직접적인 맞바꿈이 있다: $\alpha$를 줄이면($H_0$을 기각하기 어렵게 하면) $\beta$가 커지고(참 효과를 탐지하기 어려워지고), 그 반대도 마찬가지이다. 두 오류율 모두 $H_0$과 $H_1$ 아래 분포에 대한 기각 문턱의 위치에 달려 있기 때문이다.

    문턱을 옮겨 기각을 드물게 만들면(작은 $\alpha$) 동시에 대립가설을 탐지하기 어려워진다(큰 $\beta$). 둘을 동시에 줄이는 유일한 방법은 표본크기를 늘리거나(두 분포의 표준오차를 함께 줄인다) 효과크기를 키우는 것(두 분포를 더 멀리 떼어 놓는다)이다. 자료가 무한하면 $\alpha$와 $\beta$가 모두 0에 다가갈 수 있지만, 유한한 자료에서는 맞바꿈을 피할 수 없다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
**검정력의 네 요소**($\alpha$, $n$, 효과크기, 검정력) 중 셋을 알면 넷째가 정해진다. 각 방향의 계산을 모두 보여라.

</div>

??? success "풀이"
    **네 가지 방향.**

    ```python
    import numpy as np
    from scipy import stats
    from scipy.optimize import brentq

    def power_t2(n, d, alpha=0.05):
        """이표본 t 검정(집단당 n)의 정확한 검정력"""
        nu = 2 * n - 2
        lam = d * np.sqrt(n / 2)
        tc = stats.t.ppf(1 - alpha / 2, nu)
        return stats.nct.sf(tc, nu, lam) + stats.nct.cdf(-tc, nu, lam)

    # ① n 구하기
    n = next(m for m in range(4, 5000) if power_t2(m, 0.5) >= 0.80)
    print(f"① d=0.5, α=0.05, 검정력 0.80  →  집단당 n = {n}")

    # ② 검정력 구하기
    print(f"② d=0.5, α=0.05, n=40        →  검정력 = {power_t2(40, 0.5):.4f}")

    # ③ 탐지 가능한 최소 효과크기
    d = brentq(lambda x: power_t2(40, x) - 0.80, 0.01, 3.0)
    print(f"③ α=0.05, n=40, 검정력 0.80  →  d = {d:.4f}")

    # ④ 필요한 α
    a = brentq(lambda x: power_t2(40, 0.5, x) - 0.80, 1e-6, 0.5)
    print(f"④ d=0.5, n=40, 검정력 0.80   →  α = {a:.4f}")
    ```

    ```text
    ① d=0.5, α=0.05, 검정력 0.80  →  집단당 n = 64
    ② d=0.5, α=0.05, n=40        →  검정력 = 0.5981
    ③ α=0.05, n=40, 검정력 0.80  →  d = 0.6343
    ④ d=0.5, n=40, 검정력 0.80   →  α = 0.1672
    ```

    **네 계산의 쓰임이 각각 다르다.**

    | 방향 | 언제 쓰나 | 주의 |
    |---|---|---|
    | ① $n$ 구하기 | **설계 단계의 기본** | 효과크기의 근거가 관건 |
    | ② 검정력 구하기 | 표본이 제약될 때 | 낮으면 연구 재검토 |
    | ③ 최소 효과크기 | **예산이 먼저 정해진 경우** | 이 값이 실무적으로 의미 있는지 판단 |
    | ④ $\alpha$ 구하기 | 드물다 | 0.17은 받아들여지지 않는다 |

    **③이 특히 유용하다.** "집단당 40명밖에 못 모은다"면 "$d=0.63$ 이상만 탐지 가능"이라고 말할 수 있다. **그 크기의 효과가 그럴듯한지**를 영역 전문가와 논의하면, 연구를 할지 말지 판단할 수 있다.

    **④는 경고다.** 검정력 80%를 지키려면 $\alpha$를 0.17로 올려야 한다는 것은 **설계가 부족하다**는 신호다. $\alpha$를 올려 해결하려 들면 안 된다.

    **주의 — 사후 검정력은 이 네 가지에 속하지 않는다.** 관측된 효과크기를 ②에 넣는 것은 앞서 본 대로 $p$-값의 재포장일 뿐이다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
검정력이 **검정 방법에 따라** 얼마나 달라지는지 여러 분포에서 비교하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(42)
    n, M = 25, 10_000

    def run(gen, shift):
        cnt = np.zeros(3)
        for _ in range(M):
            x = gen(n)
            y = gen(n) + shift
            cnt[0] += stats.ttest_ind(x, y, equal_var=False).pvalue < 0.05
            cnt[1] += stats.mannwhitneyu(x, y).pvalue < 0.05
            # 부호검정 (중앙값 기준, 두 표본을 짝지어 비교하는 대신 대략적 대안)
            med = np.median(np.concatenate([x, y]))
            tab = [[np.sum(x > med), np.sum(x <= med)],
                   [np.sum(y > med), np.sum(y <= med)]]
            cnt[2] += stats.chi2_contingency(tab).pvalue < 0.05
        return cnt / M

    cases = [("정규", lambda m: rng.normal(0, 1, m), 0.8),
             ("t(3)", lambda m: rng.standard_t(3, m), 0.8),
             ("지수", lambda m: rng.exponential(1, m), 0.8),
             ("로그정규", lambda m: rng.lognormal(0, 1, m), 1.2)]
    print(f"{'분포':>8s} {'웰치 t':>9s} {'만-휘트니':>11s} {'중앙값검정':>11s}")
    for name, gen, sh in cases:
        r = run(gen, sh)
        print(f"{name:>8s} {r[0]:9.4f} {r[1]:11.4f} {r[2]:11.4f}")
    ```

    ```text
        분포     웰치 t     만-휘트니      중앙값검정
        정규    0.7854      0.7585      0.4788
      t(3)    0.4438      0.5716      0.3980
        지수    0.7939      0.9406      0.6787
    로그정규    0.6173      0.9597      0.7971
    ```

    **분포에 따라 순위가 바뀐다.**

    | 분포 | 최선 | 최악 |
    |---|---|---|
    | 정규 | **웰치 $t$**(0.785) | 중앙값검정(0.479) |
    | $t_3$ | **만-휘트니**(0.572) | 웰치 $t$(0.444) |
    | 지수 | **만-휘트니**(0.941) | 중앙값검정(0.679) |
    | 로그정규 | **만-휘트니**(0.960) | 웰치 $t$(0.617) |

    **로그정규에서 $t$ 검정이 0.617, 만-휘트니가 0.960이다.** 몇 개의 큰 값이 표본평균과 표준편차를 함께 부풀려 $t$ 통계량을 희석하기 때문이다. **표본크기로 환산하면 두 배 이상의 차이**다.

    **$t_3$에서도 만-휘트니가 앞선다**(0.572 대 0.444). 두꺼운 꼬리가 $t$ 검정의 분모를 키운다.

    **정규에서는 만-휘트니의 손실이 작다.** 0.785 대 0.759로 2.7%포인트다. 이론적으로 점근상대효율이 $3/\pi=0.955$이며, 표본크기로는 4.5% 손해에 해당한다.

    **중앙값검정은 언제나 나쁘다.** 순서 정보를 "중앙값보다 큰가"로 이분화해 버려 정보를 많이 잃는다.

    **실무 지침.**

    1. **정규성이 확실하면 $t$.** 다만 이득이 작다.
    2. **꼬리가 두껍거나 치우쳐 있으면 만-휘트니.** 손실이 작고 이득이 클 수 있다.
    3. **다만 추정 대상이 다르다.** 만-휘트니는 $P(X<Y)$에 대한 검정이며, 앞서 본 대로 평균 차이와 다르다.
    4. **자료를 보고 고르면 안 된다.** 앞서 본 2단계 절차의 문제가 여기에도 있다. **사전에 정한다.**

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
**검정력이 부족한 연구**를 하는 것이 왜 비윤리적일 수 있는지 논하고, 반론도 함께 검토하라.

</div>

??? success "풀이"
    **비윤리적이라는 논거.**

    1. **참가자를 헛되이 위험에 노출한다.** 결론을 낼 수 없는 연구에 환자를 참여시키는 것은 **위험은 지우고 이득은 없는** 거래다.

    2. **자원을 낭비한다.** 연구비, 시간, 시료는 유한하다.

    3. **거짓 정보를 생산한다.** 앞서 본 M-오류·S-오류 때문에, 검정력이 낮은 연구에서 나온 유의한 결과는 **과장되고 부호가 틀릴 수 있다.** 이것이 문헌에 남아 후속 연구를 오도한다.

    4. **거짓 음성이 발전을 막는다.** 효과가 있는 처리를 "효과 없음"으로 결론 내면 그 방향의 연구가 중단될 수 있다.

    5. **동의의 전제가 깨진다.** 참가자는 "의미 있는 연구에 기여한다"고 믿고 동의한다. 그 연구가 애초에 결론을 낼 수 없다면 동의의 근거가 부실하다.

    **반론과 그에 대한 재반론.**

    **반론 1 — "메타분석에 기여한다."**
    작은 연구들이 모이면 결론이 난다는 주장이다.

    - **일리가 있다.** 다만 **모든 연구가 보고되어야** 성립한다. 출판 편향이 있으면 메타분석이 오히려 왜곡된다.
    - **조건**: 사전등록하고, 결과와 무관하게 보고하며, 효과크기와 표준오차를 반드시 싣는다. 그러면 작은 연구도 정당하다.

    **반론 2 — "탐색적 연구는 다르다."**
    가설 생성이 목적이면 검정력 기준이 다르다는 주장이다.

    - **타당하다.** 다만 **탐색적임을 명시**해야 하고, 확증적 결론을 내지 말아야 한다.

    **반론 3 — "희귀질환에서는 불가피하다."**
    환자가 100명뿐인 질환에서 검정력 80%를 요구할 수 없다는 주장이다.

    - **타당하다.** 대신 **베이즈 방법, 단일군 설계, n-of-1 시험, 국제 공동연구** 등 대안을 적극적으로 검토해야 한다. "어쩔 수 없다"고 끝내면 안 된다.

    **반론 4 — "검정력 계산의 입력값도 추측이다."**
    효과크기를 모르는데 검정력을 계산하는 것이 형식적이라는 주장이다.

    - **일부 타당하다.** 그래서 **민감도 분석**과 **탐지 가능한 최소 효과** 보고가 중요하다. 계산이 불완전하다는 것이 계산을 안 해도 된다는 뜻은 아니다.

    **균형 잡힌 결론.**

    - **확증적 연구에서 검정력 부족은 정당화하기 어렵다.**
    - **탐색적·희귀질환 연구는 다른 기준**이 필요하되, 그 성격을 명시하고 결과를 전부 보고해야 한다.
    - **가장 중요한 것은 투명성이다.** 검정력이 낮다는 사실을 숨기는 것이 낮은 것 자체보다 나쁘다.

    **제도적 장치.** 여러 연구윤리위원회가 검정력 계산을 심사 요건으로 요구한다. 등록보고서 제도는 검정력이 확보된 연구만 사전 승인한다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
**여러 결과변수**가 있을 때 검정력을 어떻게 정의하고 계획하는지 설명하라.

</div>

??? success "풀이"
    **검정력의 정의가 여럿이 된다.**

    | 정의 | 뜻 | 언제 쓰나 |
    |---|---|---|
    | **개별 검정력** | 각 결과변수에서의 검정력 | 결과마다 독립적으로 판단 |
    | **분리 검정력(disjunctive)** | 적어도 하나에서 유의 | "어느 하나라도 효과가 있으면 성공" |
    | **결합 검정력(conjunctive)** | 모두에서 유의 | "모든 지표가 개선되어야 성공" |

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(8)
    M, k, n = 40_000, 3, 64
    d = 0.5

    print(f"{'ρ':>5s} {'개별':>8s} {'적어도 하나':>12s} {'모두':>8s} "
          f"{'본페로니 후 하나':>16s}")
    for rho in [0.0, 0.3, 0.6, 0.9]:
        S = rho * np.ones((k, k)) + (1 - rho) * np.eye(k)
        L = np.linalg.cholesky(S)
        z = rng.standard_normal((M, k)) @ L.T + d * np.sqrt(n / 2)
        p = 2 * stats.norm.sf(np.abs(z))
        sig = p < 0.05
        sig_b = p < 0.05 / k
        print(f"{rho:5.1f} {sig[:, 0].mean():8.4f} {sig.any(1).mean():12.4f} "
              f"{sig.all(1).mean():8.4f} {sig_b.any(1).mean():16.4f}")
    ```

    ```text
        ρ     개별      적어도 하나      모두     본페로니 후 하나
      0.0   0.8105       0.9932   0.5231           0.9631
      0.3   0.8115       0.9725   0.5853           0.9170
      0.6   0.8065       0.9382   0.6451           0.8563
      0.9   0.8092       0.8797   0.7304           0.7667
    ```

    **세 정의가 크게 다르다.** 개별 0.81인데 "모두"는 0.52, "적어도 하나"는 0.99다.

    **상관이 커지면 수렴한다.** $\rho=0.9$에서 세 값이 0.73~0.88로 가까워진다. 결과변수들이 사실상 같은 것을 재기 때문이다.

    **본페로니 보정 후에도 "적어도 하나"는 높다.** $\rho=0$에서 0.963이다. **보정의 대가가 생각보다 작다.**

    **계획 지침.**

    1. **주 결과변수를 하나로 정하는 것이 가장 깔끔하다.** 그러면 정의의 문제가 사라진다.

    2. **여럿이 불가피하면 어느 정의를 쓸지 사전에 명시**한다. "적어도 하나"인지 "모두"인지에 따라 필요한 표본이 크게 다르다.

    3. **"모두"를 요구하면 표본이 많이 든다.** $\rho=0$에서 결합 검정력 80%를 얻으려면 개별 검정력이 $0.8^{1/3}=0.928$이어야 하고, 그만큼 표본이 는다.

    4. **복합 결과변수를 고려한다.** 여러 지표를 하나로 합치면 문제가 사라지지만, 앞서 본 대로 해석이 흐려진다.

    5. **계층적 검정.** 순서를 정해 순차 검정하면 보정 없이 FWER이 유지된다.

    **주의.** 위 계산은 **모든 결과변수에서 효과가 같다**고 가정했다. 실제로는 일부만 효과가 있는 경우가 흔하고, 그때는 "적어도 하나"의 검정력이 크게 떨어진다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
**결측과 탈락**이 검정력에 미치는 영향을 계산하고, 설계에서 어떻게 대비하는지 정리하라.

</div>

??? success "풀이"
    **단순한 경우 — 완전 무작위 탈락.** 유효 표본이 $n(1-q)$로 줄고, 검정력이 그에 맞게 떨어진다.

    ```python
    import numpy as np
    from scipy import stats

    def power_t2(n, d=0.5, alpha=0.05):
        nu = max(2 * n - 2, 1)
        lam = d * np.sqrt(n / 2)
        tc = stats.t.ppf(1 - alpha / 2, nu)
        return stats.nct.sf(tc, nu, lam) + stats.nct.cdf(-tc, nu, lam)

    n0 = 64                                   # 계획 표본(검정력 0.80)
    print(f"{'탈락률':>8s} {'유효 n':>8s} {'실제 검정력':>12s} "
          f"{'보정 모집 n':>12s}")
    for q in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]:
        neff = int(np.floor(n0 * (1 - q)))
        nadj = int(np.ceil(n0 / (1 - q)))
        print(f"{q:8.0%} {neff:8d} {power_t2(neff):12.4f} {nadj:12d}")
    ```

    ```text
       탈락률    유효 n     실제 검정력    보정 모집 n
         0%       64       0.8015           64
         5%       60       0.7753           68
        10%       57       0.7538           72
        20%       51       0.7056           80
        30%       44       0.6402           92
        50%       32       0.5036          128
    ```

    **탈락률 20%면 검정력이 0.80에서 0.71로 떨어진다.** 회복하려면 집단당 80명을 모집해야 한다.

    **대응설계에서는 더 심각하다.** 앞서 본 대로 한쪽이 빠지면 쌍이 깨지므로, 개체 탈락률 $q$에서 완전한 쌍의 비율이 $(1-q)^2$다.

    ```python
    for q in [0.10, 0.20, 0.30]:
        print(f"개체 탈락 {q:.0%} → 쌍 손실 {1 - (1 - q)**2:.1%}, "
              f"필요 모집 배율 {1 / (1 - q)**2:.2f}배")
    ```

    ```text
    개체 탈락 10% → 쌍 손실 19.0%, 필요 모집 배율 1.23배
    개체 탈락 20% → 쌍 손실 36.0%, 필요 모집 배율 1.56배
    개체 탈락 30% → 쌍 손실 51.0%, 필요 모집 배율 2.04배
    ```

    **더 나쁜 경우 — 정보적 탈락.** 탈락이 결과와 관련되면 **검정력만이 아니라 편향**이 생긴다. 이때는 표본을 늘려도 해결되지 않는다.

    **설계에서의 대비.**

    | 항목 | 내용 |
    |---|---|
    | 표본 계산 | $n/(1-q)$ 또는 대응이면 $n/(1-q)^2$ |
    | $q$의 출처 | 유사 연구의 실제 탈락률. **낙관하지 않는다** |
    | 추적 계획 | 연락 수단 다중화, 방문 간격 단축, 보상 |
    | 기저 정보 | 탈락자의 특성을 반드시 확보 |
    | 분석 계획 | 혼합모형·다중대체를 **사전 명시** |
    | 민감도 | MNAR 시나리오에서의 결과 |
    | 중간 점검 | 실제 탈락률을 확인하고 필요시 모집 확대 |

    **흔한 실수.** 검정력 계산에서 나온 $n$을 **그대로 모집 목표로** 삼는 것이다. 임상시험 계획서를 검토할 때 가장 자주 발견되는 누락이다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
검정력 분석의 **한계**를 정리하라. 계산이 답해 주지 못하는 것은 무엇인가?

</div>

??? success "풀이"
    **답해 주지 못하는 것 여덟.**

    **1 — 효과크기가 얼마인가.** 검정력 계산은 효과크기를 **입력으로 받는다.** 그것을 모르면 계산 전체가 가정 위에 선다. 순환논법에 가깝다.

    **2 — 연구를 할 가치가 있는가.** 검정력 80%를 달성해도, 그 질문이 중요하지 않거나 이미 답이 알려져 있으면 무의미하다.

    **3 — 측정이 타당한가.** 앞서 본 제3종 오류의 영역이다. 검정력이 아무리 높아도 잘못된 것을 재면 소용없다.

    **4 — 표본이 대표적인가.** 검정력은 **내적 타당도**의 일부만 다룬다. 일반화 가능성은 전혀 다루지 않는다.

    **5 — 모형 가정이 맞는가.** 정규성·독립성·등분산을 가정하고 계산한 검정력은, 그 가정이 깨지면 달라진다. 앞서 로그정규에서 본 대로 절반이 될 수 있다.

    **6 — 비표집오차.** 무응답 편향, 측정오차, 누락변수 편향은 표본을 늘려도 줄지 않는다. **검정력을 높이는 것이 이들을 해결하지 못한다.**

    **7 — 여러 연구를 합친 결론.** 하나의 연구가 검정력 80%여도, 분야 전체의 신뢰도는 사전등록·보고 관행·재현 문화에 달려 있다.

    **8 — 결과를 어떻게 쓸 것인가.** 정책 결정, 임상 지침, 후속 연구의 방향은 통계 밖의 문제다.

    **검정력 계산의 진짜 가치.**

    이 모든 한계에도 검정력 분석이 중요한 이유는

    1. **자원 배분을 합리화한다.** "이 설계로 무엇을 알 수 있는가"를 사전에 묻게 한다.
    2. **효과크기를 명시하게 만든다.** 이 과정에서 "우리가 찾는 효과가 얼마나 큰가"를 영역 전문가와 논의하게 되며, **이 논의 자체가 계산 결과보다 값질 때가 많다.**
    3. **실현 불가능한 연구를 걸러 낸다.** "필요한 표본이 5,000명"이라는 계산은 계획을 바꾸라는 신호다.
    4. **투명성을 강제한다.** 가정을 문서에 적게 한다.

    **권고 세 가지.**

    - **하나의 숫자가 아니라 범위로 생각한다.** 여러 효과크기와 가정에서 계산해 표로 제시한다.
    - **"탐지 가능한 최소 효과"를 함께 보고**한다. 자원이 제한된 현실에서 가장 정직한 표현이다.
    - **검정력이 낮으면 낮다고 밝힌다.** 숨기는 것이 가장 나쁘다. 밝히면 독자가 결과를 올바로 할인해 읽을 수 있다.

    **한 문장.** **검정력 분석은 연구의 품질을 보장하지 않는다. 다만 품질을 논의할 언어를 준다.**

---

## 정리하며

- 검정력은 참 효과를 올바르게 탐지할 확률이다.
- 표본크기가 충분한지 확인하기 위해 연구 전에 항상 검정력 분석을 하라.
- 표본크기를 늘리는 것이 검정력을 높이는 가장 현실적인 방법이다.
- $\alpha$, $\beta$, 표본크기, 효과크기 사이에는 직접적인 맞바꿈이 있다.
- Statsmodels는 여러 검정 유형에 대한 편리한 검정력 분석 함수를 제공한다.
- 검정력 곡선으로 표본크기와 검정력의 관계를 시각화하라.
