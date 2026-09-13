# 가설검정 시연

## 개요

가설검정은 표본자료에 근거하여 모수에 관한 결정을 내리는 형식적인 틀이다. 이 절차는 귀무가설 $H_0$(기본 주장)을 대립가설 $H_1$에 맞세우고, 검정통계량을 계산한 뒤 그 표본분포로 p-값을 얻는다. 이 페이지에서는 평균과 비율에 대한 일표본·이표본·대응 검정을 검정력 분석 및 신뢰구간과 가설검정의 쌍대성과 함께 시연한다.

## 평균에 대한 일표본 검정

모표준편차 $\sigma$를 모를 때는 **t-검정**을 쓴다. 표본평균이 $\bar{X}$이고 표본표준편차가 $S$인 확률표본 $X_1, \dots, X_n$이 주어졌을 때 $H_0\colon \mu = \mu_0$의 검정통계량은

$$
T = \frac{\bar{X} - \mu_0}{S / \sqrt{n}} \sim t_{n-1}.
$$

<div class="codebox" markdown>

### 예제 1. 일표본 t-검정 { .eg }

```python
import numpy as np
from scipy import stats

# 공장은 평균 무게가 500g이라고 주장한다.
data = np.array([498, 495, 502, 497, 501, 499, 496, 503, 494, 500,
                 497, 502, 496, 501, 498, 499, 495, 503, 497, 500])
mu_0 = 500
alpha = 0.05

# scipy의 ttest_1samp는 기본이 **양측**이고 p-값도 양측이다.
t_stat, p_value = stats.ttest_1samp(data, mu_0)
print(f"t = {t_stat:.4f}, p-value = {p_value:.4f}")
print(f"Decision: {'Reject H0' if p_value < alpha else 'Fail to reject H0'}")
```

출력:

```
t = -2.1739, p-value = 0.0426
Decision: Reject H0
```

표본평균은 498.65로 주장값 500에서 1.35g 모자란다. 자료의 산포에 비하면 우연으로 보기 어려운 차이라 5% 수준에서 기각한다.

</div>

**단측** 검정 $H_1\colon \mu < \mu_0$에서는 검정통계량이 대립가설 방향에 있을 때 양측 p-값을 2로 나눈다:

<div class="codebox" markdown>

### 예제 2. 단측 p-값 구하기 { .eg }

```python
# 통계량이 대립가설 쪽(여기서는 음수)이면 양측 p-값을 반으로 나눈다.
# 반대쪽이면 1에서 그 절반을 빼야 한다. 이 두 번째 경우를 빠뜨리면
# 방향이 반대인 자료에 대해 0에 가까운 p-값을 보고하게 된다.
p_one_sided = p_value / 2 if t_stat < 0 else 1 - p_value / 2
print(f"one-sided p (H1: mu < 500) = {p_one_sided:.4f}")
```

출력:

```
one-sided p (H1: mu < 500) = 0.0213
```

</div>

### 직접 계산

<div class="codebox" markdown>

#### 예제 3. 검정통계량 직접 계산 { .eg }

```python
n = len(data)
xbar = data.mean()
s = data.std(ddof=1)
t_manual = (xbar - mu_0) / (s / np.sqrt(n))
# 양측 p-값은 "관측된 것만큼 극단적인" 확률이므로 한쪽 꼬리를 두 배 한다.
# -abs(t)를 넣어 왼쪽 꼬리를 재면 t의 부호와 무관하게 같은 식이 쓰인다.
p_manual = 2 * stats.t.cdf(-abs(t_manual), df=n - 1)
print(f"t = {t_manual:.4f}, p = {p_manual:.4f}")
```

출력:

```
t = -2.1739, p = 0.0426
```

scipy가 돌려준 값과 소수점 아래까지 같다.

</div>

## 비율에 대한 일표본 검정

$\hat{p} = x/n$으로 $H_0\colon p = p_0$을 검정할 때 Wald 검정통계량은

$$
Z = \frac{\hat{p} - p_0}{\sqrt{p_0(1 - p_0)/n}} \;\dot\sim\; N(0,1).
$$

<div class="codebox" markdown>

### 예제 4. 비율에 대한 일표본 검정 { .eg }

```python
from statsmodels.stats.proportion import proportions_ztest

x, n = 12, 200  # 200개 중 불량 12개
p_0 = 0.05
p_hat = x / n

# 표준오차에 p_hat이 아니라 **p_0**을 넣는다.
# 검정은 "H0가 참이라면"이라는 가정 아래에서의 확률을 재기 때문이다.
# 신뢰구간에서 p_hat을 넣었던 것과 여기서 갈린다.
z_stat = (p_hat - p_0) / np.sqrt(p_0 * (1 - p_0) / n)
p_value = 2 * stats.norm.sf(abs(z_stat))
print(f"수동:        z = {z_stat:.4f}, p = {p_value:.4f}")

# statsmodels는 기본적으로 p_hat 기반 표준오차를 쓴다(prop_var로 바꿀 수 있다).
# 그래서 같은 자료에서도 z가 조금 다르게 나온다.
z_sm, p_sm = proportions_ztest(x, n, value=p_0)
print(f"statsmodels: z = {z_sm:.4f}, p = {p_sm:.4f}")
```

출력:

```
수동:        z = 0.6489, p = 0.5164
statsmodels: z = 0.5955, p = 0.5515
```

두 결과가 다르다는 점이 중요하다. `proportions_ztest`는 표준오차를 $\hat p$로 계산하는 반면 위의 수동 계산은 $p_0$을 쓴다. 어느 쪽도 틀린 것은 아니지만, 교과서의 공식은 대개 $p_0$ 쪽이다. 어느 규약을 쓰는지 모르고 결과만 옮기면 보고한 z가 재현되지 않는다.

</div>

## 이표본 t-검정

두 모집단에서 얻은 독립표본에 대해 귀무가설은 $H_0\colon \mu_1 = \mu_2$이다. **Welch의 t-검정**(분산이 다른 경우)은

$$
T = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{S_1^2/n_1 + S_2^2/n_2}}
$$

를 쓰며 자유도는 Welch–Satterthwaite 근사로 계산한다.

<div class="codebox" markdown>

### 예제 5. 이표본 t-검정 { .eg }

```python
drug_a = np.array([5.2, 4.8, 6.1, 5.5, 4.9, 5.7, 5.3, 6.0, 5.1, 5.4])
drug_b = np.array([4.1, 3.8, 4.5, 4.2, 3.9, 4.6, 4.0, 4.3, 3.7, 4.4])

# scipy의 기본값은 equal_var=True(합동)이다. Welch를 쓰려면 명시해야 한다.
# 기본값을 그대로 두고 "Welch를 썼다"고 적는 실수가 흔하다.
t_welch, p_welch = stats.ttest_ind(drug_a, drug_b, equal_var=False)
t_pooled, p_pooled = stats.ttest_ind(drug_a, drug_b, equal_var=True)
print(f"Welch:  t = {t_welch:.4f}, p = {p_welch:.6f}")
print(f"Pooled: t = {t_pooled:.4f}, p = {p_pooled:.6f}")
```

출력:

```
Welch:  t = 7.4628, p = 0.000001
Pooled: t = 7.4628, p = 0.000001
```

$n_1 = n_2$이고 두 표본의 분산이 비슷하면 두 방법이 사실상 같은 답을 준다. 통계량은 아예 같고 자유도만 18과 17.8로 조금 다르다. 표본크기가 다르고 분산도 다를 때 비로소 둘이 갈라진다.

</div>

## 대응 t-검정

관측값이 자연스러운 쌍을 이룰 때(예: 전후 측정) 차이 $D_i = X_i^{(\text{after})} - X_i^{(\text{before})}$를 계산하고 $H_0\colon \mu_D = 0$을 검정한다:

$$
T = \frac{\bar{D}}{S_D / \sqrt{n}} \sim t_{n-1}.
$$

<div class="codebox" markdown>

### 예제 6. 대응 t-검정 { .eg }

```python
before = np.array([145, 150, 138, 155, 142, 148, 136, 152, 140, 146])
after  = np.array([138, 142, 130, 148, 135, 140, 132, 145, 134, 139])

t_stat, p_value = stats.ttest_rel(after, before)
print(f"ttest_rel:   t = {t_stat:.4f}, p = {p_value:.8f}")

# 대응검정은 차이에 대한 일표본 검정과 **같은 것**이다. 별개의 방법이 아니다.
diff = after - before
t_stat2, p_value2 = stats.ttest_1samp(diff, 0)
print(f"ttest_1samp: t = {t_stat2:.4f}, p = {p_value2:.8f}")

# 짝을 무시하고 독립 이표본으로 다루면 어떻게 되는지 비교해 본다.
t_ind, p_ind = stats.ttest_ind(after, before)
print(f"ttest_ind:   t = {t_ind:.4f}, p = {p_ind:.8f}")
```

출력:

```
ttest_rel:   t = -18.2253, p = 0.00000002
ttest_1samp: t = -18.2253, p = 0.00000002
ttest_ind:   t = -2.5841, p = 0.01871524
```

앞의 두 줄이 완전히 같다. 대응 $t$-검정은 별개의 방법이 아니라 차이에 대한 일표본 검정 그 자체다.

세 번째 줄이 이 예제의 핵심이다. 같은 자료를 짝만 무시하고 분석하면 $t$가 $-18.2$에서 $-2.6$으로, $p$가 $2 \times 10^{-8}$에서 0.019로 뛴다. 사람마다 혈압 수준이 136에서 155까지 흩어져 있어 그 개인차가 처리 효과를 덮어 버리기 때문이다. 여기서는 두 검정 모두 5% 수준에서 기각하지만, 효과가 조금만 작았다면 짝을 무시한 쪽은 놓쳤을 것이다.

</div>

## 검정력 분석

검정의 **검정력**은 $H_1$이 참일 때 $H_0$을 올바르게 기각할 확률이다:

$$
\text{Power} = 1 - \beta = P(\text{reject } H_0 \mid H_1 \text{ true}).
$$

<div class="codebox" markdown>

### 예제 7. 검정력 분석 { .eg }

```python
from statsmodels.stats.power import TTestPower, TTestIndPower

# 일표본: sigma=15에서 5점 차이를 탐지하려 한다.
# 검정력 계산에 들어가는 것은 delta도 sigma도 아니고 그 비(효과크기)뿐이다.
analysis = TTestPower()
effect_size = 5 / 15  # Cohen's d
n_needed = analysis.solve_power(effect_size=effect_size, alpha=0.05,
                                 power=0.80, alternative='two-sided')
print(f"one-sample n = {n_needed:.1f}")

# 이표본: 중간 크기 효과(d=0.5). ratio는 두 집단 크기의 비이고
# 돌려주는 n_each는 **집단당** 표본크기다. 전체가 아니다.
analysis2 = TTestIndPower()
n_each = analysis2.solve_power(effect_size=0.5, alpha=0.05, power=0.80,
                                ratio=1.0, alternative='two-sided')
print(f"two-sample n per group = {n_each:.1f}")
```

출력:

```
one-sample n = 72.6
two-sample n per group = 63.8
```

올림하면 일표본은 73개, 이표본은 집단당 64개(합계 128개)다. 효과크기가 0.33에서 0.5로 **커졌는데도** 전체 표본이 더 필요하다. 이표본 문제에서는 평균을 두 개 추정해야 해서 차이의 표준오차가 그만큼 커지기 때문이다.

</div>

## 신뢰구간과 검정의 쌍대성

수준 $\alpha$의 양측검정이 $H_0\colon \mu = \mu_0$을 기각하지 못할 필요충분조건은 $\mu_0$이 $100(1-\alpha)\%$ 신뢰구간 안에 있는 것이다.

<div class="codebox" markdown>

### 예제 8. 신뢰구간과 검정의 쌍대성 { .eg }

```python
data = np.array([52, 48, 55, 50, 47, 53, 49, 51, 54, 46])
n = len(data)
xbar = data.mean()
s = data.std(ddof=1)
alpha = 0.05

t_c = stats.t.ppf(1 - alpha / 2, df=n - 1)
me = t_c * s / np.sqrt(n)
ci = (xbar - me, xbar + me)
print(f"95% CI = ({ci[0]:.4f}, {ci[1]:.4f})\n")

# 여러 mu_0에 대해 검정을 반복하며 "기각 여부"와 "구간 포함 여부"를 나란히 본다.
# 두 열이 언제나 정확히 반대여야 한다. 그것이 쌍대성이다.
print(f"{'mu0':>5} {'p-value':>9} {'reject':>7} {'in CI':>6}")
for mu0 in [48, 49, 50, 51, 52, 53]:
    t_stat, p_val = stats.ttest_1samp(data, mu0)
    in_ci = ci[0] <= mu0 <= ci[1]
    reject = p_val < alpha
    print(f"{mu0:>5} {p_val:>9.4f} {str(reject):>7} {str(in_ci):>6}")
```

출력:

```
95% CI = (48.3341, 52.6659)

  mu0   p-value  reject  in CI
   48    0.0282    True  False
   49    0.1516   False   True
   50    0.6141   False   True
   51    0.6141   False   True
   52    0.1516   False   True
   53    0.0282    True  False
```

`reject` 열과 `in CI` 열이 여섯 줄 모두에서 정확히 반대다. 신뢰구간 $(48.33, 52.67)$ 밖에 있는 48과 53만 기각된다.

p-값이 구간의 중심 $\bar x = 50.5$를 기준으로 대칭인 것도 눈여겨볼 만하다. 49와 52가 둘 다 0.1516, 48과 53이 둘 다 0.0282다. 두 값이 $\bar x$에서 같은 거리에 있기 때문이다. 신뢰구간은 이런 검정들을 $\mu_0$에 대해 전부 돌려 놓고 기각되지 않는 값만 모아 놓은 것과 같다.

</div>

### 해석

쌍대성 원리는 다음을 말한다:

$$
\mu_0 \notin \left(\bar{X} \pm t_{\alpha/2,\,n-1}\,\frac{S}{\sqrt{n}}\right) \iff \text{reject } H_0\colon \mu = \mu_0 \text{ at level } \alpha.
$$

이는 통합된 관점을 준다: 신뢰구간을 구성하는 일은 가설검정의 가족을 뒤집는 일과 같다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> 전구 $n=16$개의 표본에서 $\bar{x}=1020$시간, $s=80$시간을 얻었다. $\alpha=0.05$에서 $H_0\colon \mu=1000$ 대 $H_1\colon \mu>1000$을 검정하라. 검정통계량, p-값, 판정을 제시하라.

</div>

??? success "풀이"

    검정통계량은

    $$
    T = \frac{1020 - 1000}{80/\sqrt{16}} = \frac{20}{20} = 1.0.
    $$

    $H_0$ 아래에서 $T \sim t_{15}$이다. 단측 p-값은

    $$
    p = P(T_{15} \geq 1.0) \approx 0.1667.
    $$

    $p = 0.1667 > 0.05$이므로 $H_0$을 기각하지 못한다. 평균 수명이 1000시간을 넘는다는 증거가 부족하다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span> 유권자 400명 조사에서 228명이 어떤 안건을 지지한다. 1% 유의수준에서 참 비율이 0.50을 넘는지 검정하라.

</div>

??? success "풀이"

    $\hat{p} = 228/400 = 0.57$이고 $H_0\colon p = 0.50$ 대 $H_1\colon p > 0.50$을 검정한다. 검정통계량은

    $$
    Z = \frac{0.57 - 0.50}{\sqrt{0.50 \times 0.50 / 400}} = \frac{0.07}{0.025} = 2.80.
    $$

    단측 p-값은 $P(Z \geq 2.80) \approx 0.0026 < 0.01$이므로 $H_0$을 기각한다. 유권자의 절반 넘는 비율이 이 안건을 지지한다는 강한 증거이다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span> 독립인 두 집단에서 $\bar{x}_1=75$, $s_1=10$, $n_1=20$과 $\bar{x}_2=70$, $s_2=12$, $n_2=25$를 얻었다. $\alpha=0.05$에서 $H_0\colon \mu_1 = \mu_2$의 Welch t-검정을 수행하라. Welch–Satterthwaite 자유도의 공식을 함께 쓰라.

</div>

??? success "풀이"

    검정통계량은

    $$
    T = \frac{75 - 70}{\sqrt{10^2/20 + 12^2/25}} = \frac{5}{\sqrt{5 + 5.76}} = \frac{5}{\sqrt{10.76}} \approx \frac{5}{3.280} \approx 1.524.
    $$

    Welch–Satterthwaite 자유도는

    $$
    \nu = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}} = \frac{(5 + 5.76)^2}{\frac{25}{19} + \frac{33.18}{24}} \approx \frac{115.78}{1.316 + 1.382} \approx 42.9.
    $$

    $\nu \approx 42$를 쓰면 양측 p-값이 약 0.135이다. $p > 0.05$이므로 $H_0$을 기각하지 못한다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> 신뢰구간과 검정의 쌍대성을 증명하라. 즉 수준 $\alpha$의 양측 일표본 t-검정에서 $H_0\colon \mu = \mu_0$을 기각하는 것이 $\mu_0$이 $100(1-\alpha)\%$ 신뢰구간 밖에 있는 것과 동등함을 보여라.

</div>

??? success "풀이"

    이 검정은 $|T| > t_{\alpha/2,\,n-1}$일 때 $H_0$을 기각한다. 즉,

    $$
    \left|\frac{\bar{X} - \mu_0}{S/\sqrt{n}}\right| > t_{\alpha/2,\,n-1}.
    $$

    이는 다음과 동등하다:

    $$
    |\bar{X} - \mu_0| > t_{\alpha/2,\,n-1}\,\frac{S}{\sqrt{n}},
    $$

    즉 $\mu_0 < \bar{X} - t_{\alpha/2,\,n-1}\,S/\sqrt{n}$이거나 $\mu_0 > \bar{X} + t_{\alpha/2,\,n-1}\,S/\sqrt{n}$이다. 다시 말해,

    $$
    \mu_0 \notin \left(\bar{X} - t_{\alpha/2,\,n-1}\,\frac{S}{\sqrt{n}},\; \bar{X} + t_{\alpha/2,\,n-1}\,\frac{S}{\sqrt{n}}\right).
    $$

    오른쪽의 구간이 바로 $\mu$의 $100(1-\alpha)\%$ 신뢰구간이다. 따라서 수준 $\alpha$에서 이 검정이 $H_0\colon \mu=\mu_0$을 기각할 필요충분조건은 $\mu_0$이 신뢰구간에 들어 있지 않은 것이다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> 일표본 z-검정의 검정력 공식을 써서, ($\sigma$를 아는) 양측검정에서 이동 $\delta$를 유의수준 $\alpha$에서 검정력 $1-\beta$로 탐지하는 데 필요한 표본크기가

$$
n = \left(\frac{(z_{\alpha/2} + z_\beta)\,\sigma}{\delta}\right)^2
$$

임을 보여라.

</div>

??? success "풀이"

    $H_0$ 아래에서 $Z = (\bar{X}-\mu_0)/(\sigma/\sqrt{n}) \sim N(0,1)$이고 $|Z|>z_{\alpha/2}$일 때 기각한다. $H_1\colon \mu = \mu_0 + \delta$ 아래에서 이 통계량은 $Z \sim N(\delta\sqrt{n}/\sigma,\,1)$을 따른다. ($\delta>0$일 때 지배적인 오른쪽 꼬리만 보면) 검정력은

    $$
    1 - \beta = P\!\left(Z > z_{\alpha/2}\right) = P\!\left(N(0,1) > z_{\alpha/2} - \frac{\delta\sqrt{n}}{\sigma}\right).
    $$

    이를 $1-\beta$와 같다고 놓으면

    $$
    z_{\alpha/2} - \frac{\delta\sqrt{n}}{\sigma} = -z_\beta,
    $$

    따라서

    $$
    \frac{\delta\sqrt{n}}{\sigma} = z_{\alpha/2} + z_\beta \implies \sqrt{n} = \frac{(z_{\alpha/2}+z_\beta)\,\sigma}{\delta}.
    $$

    양변을 제곱하면 원하는 공식을 얻는다. $\square$

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
**네이만-피어슨 보조정리**를 진술하고, 단순가설 대 단순가설에서 우도비 검정이 최강력임을 증명하라.

</div>

??? success "풀이"
    **진술.** $H_0:\theta=\theta_0$ 대 $H_1:\theta=\theta_1$(둘 다 단순가설)에서, 기각역

    $$
    C=\left\{\mathbf x:\ \frac{L(\theta_1;\mathbf x)}{L(\theta_0;\mathbf x)}>k\right\}
    $$

    가 $P_{\theta_0}(C)=\alpha$를 만족하면, $C$는 **수준 $\alpha$인 모든 검정 중 검정력이 최대**다.

    **증명.** $\phi$를 $C$의 지시함수, $\psi$를 다른 수준 $\alpha$ 검정의 검정함수라 하자($0\le\psi\le1$, $E_{\theta_0}[\psi]\le\alpha$).

    핵심은 다음 부등식이 **모든 $\mathbf x$에서** 성립한다는 것이다.

    $$
    \left\{\phi(\mathbf x)-\psi(\mathbf x)\right\}\left\{L_1(\mathbf x)-kL_0(\mathbf x)\right\}\ \ge\ 0
    $$

    - $L_1>kL_0$이면 $\phi=1\ge\psi$이므로 두 인수가 모두 $\ge0$.
    - $L_1<kL_0$이면 $\phi=0\le\psi$이므로 두 인수가 모두 $\le0$.
    - $L_1=kL_0$이면 둘째 인수가 0.

    양변을 적분하면

    $$
    \int(\phi-\psi)L_1\ \ge\ k\int(\phi-\psi)L_0=k\left\{\alpha-E_{\theta_0}[\psi]\right\}\ \ge\ 0
    $$

    이므로 $E_{\theta_1}[\phi]\ge E_{\theta_1}[\psi]$, 즉 $C$의 검정력이 더 크다. $\square$

    **직관.** 가능도비가 큰 표본부터 기각역에 넣는 것이 "**제1종 오류 예산 $\alpha$를 가장 효율적으로 쓰는**" 방법이다. 배낭 문제에서 가치/무게 비가 높은 것부터 담는 것과 같은 논리다.

    **예 — 정규평균.** $N(\mu,\sigma^2)$, $\sigma$ 기지, $H_0:\mu=0$ 대 $H_1:\mu=\delta>0$이면

    $$
    \frac{L_1}{L_0}=\exp\left\{\frac{\delta}{\sigma^2}\sum x_i-\frac{n\delta^2}{2\sigma^2}\right\}>k
    \iff \bar x>c
    $$

    로 **$\bar X$가 크면 기각**하는 통상의 $z$ 검정이 나온다. $\delta$의 구체적인 값이 $c$에 들어가지 않으므로, **모든 $\delta>0$에 대해 같은 검정**이다. 이 경우 검정이 **일양최강력(UMP)** 이 된다.

    **한계.**

    1. **양측 대립에서는 UMP가 없다.** $\delta>0$과 $\delta<0$에서 최강력 기각역의 방향이 반대라, 하나의 검정이 둘 다 최적일 수 없다. 이때는 **불편성**이나 **불변성**을 추가로 요구해 최적 검정을 특정한다.
    2. **복합가설·다차원 모수**에서는 일반적으로 UMP가 없다. 우도비 검정이 그 대안으로 쓰인다.
    3. **이산분포에서는 정확히 $\alpha$를 못 맞춘다.** 엄밀한 진술에는 경계에서의 무작위화가 들어간다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
**일반화 우도비검정**의 구성과 윌크스 정리를 설명하고, 자유도를 어떻게 세는지 예로 보여라.

</div>

??? success "풀이"
    **구성.** 모수공간 $\Theta$와 귀무공간 $\Theta_0\subset\Theta$에 대해

    $$
    \Lambda=\frac{\sup_{\theta\in\Theta_0}L(\theta)}{\sup_{\theta\in\Theta}L(\theta)}\in(0,1],
    \qquad
    G^2=-2\log\Lambda
    $$

    $\Lambda$가 작으면(제약을 걸었을 때 가능도가 많이 떨어지면) 기각한다.

    **윌크스 정리.** 정칙 조건 아래 $H_0$이 참이면

    $$
    G^2\ \xrightarrow{d}\ \chi^2_r,\qquad r=\dim\Theta-\dim\Theta_0
    $$

    **자유도는 "제약의 개수"** 다.

    **자유도 세기 — 네 예.**

    | 모형 | $\dim\Theta$ | $\dim\Theta_0$ | $r$ |
    |---|---|---|---|
    | $N(\mu,\sigma^2)$에서 $H_0:\mu=0$ | 2 | 1 | 1 |
    | $N(\mu,\sigma^2)$에서 $H_0:\mu=0,\sigma=1$ | 2 | 0 | 2 |
    | 다항 $k$범주에서 균등성 | $k-1$ | 0 | $k-1$ |
    | 회귀 $p$개 변수 중 $q$개가 0 | $p+1$ | $p+1-q$ | $q$ |

    **예 — 이항.** $H_0:p=0.5$, $n=30$, $k=22$.

    $$
    G^2=2\left\{k\log\frac{\hat p}{p_0}+(n-k)\log\frac{1-\hat p}{1-p_0}\right\}
    $$

    ```python
    import numpy as np
    from scipy import stats

    n, k, p0 = 30, 22, 0.5
    ph = k / n
    G2 = 2 * (k * np.log(ph / p0) + (n - k) * np.log((1 - ph) / (1 - p0)))
    print(f"G² = {G2:.4f}   df = 1   p = {stats.chi2.sf(G2, 1):.4f}")
    ```

    ```text
    G² = 6.7939   df = 1   p = 0.0091
    ```

    **주의할 정칙 조건.**

    1. **$\theta_0$가 모수공간의 내부점이어야 한다.** 분산성분처럼 $H_0:\sigma_b^2=0$이 경계면 극한분포가 $\chi^2$가 아니라 **혼합분포** $\tfrac12\chi^2_0+\tfrac12\chi^2_1$이다. 그냥 $\chi^2_1$을 쓰면 보수적이 된다.

    2. **모형이 겹쳐 있어야(nested) 한다.** 겹치지 않는 모형 비교에는 윌크스 정리가 적용되지 않는다. AIC나 복스 검정을 쓴다.

    3. **식별 가능해야 한다.** 혼합모형의 성분 수 검정처럼 $H_0$ 아래에서 일부 모수가 식별되지 않으면 극한분포가 표준이 아니다.

    4. **$n$이 충분해야 한다.** 소표본에서는 $\chi^2$ 근사가 나쁘다. **바틀렛 보정**이나 부트스트랩으로 개선한다.

    **왜 우도비를 쓰는가.**

    - **불변이다.** 모수를 다시 매개화해도 $\Lambda$가 변하지 않는다. 왈드 검정은 그렇지 않다.
    - **일반적이다.** 어떤 겹친 모형 쌍에도 적용된다.
    - **좋은 성질.** 넓은 조건에서 점근적으로 최적(국소최강력)이다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
**왈드·점수·우도비** 세 검정을 같은 자료에 적용하고, 왜 다른 답을 주는지 설명하라.

</div>

??? success "풀이"
    **세 검정의 기하.** 로그가능도 $\ell(\theta)$를 그렸을 때

    | 검정 | 무엇을 재는가 | 필요한 것 |
    |---|---|---|
    | **왈드** | $\hat\theta$와 $\theta_0$의 **가로 거리** | $\hat\theta$와 $\hat I(\hat\theta)$ |
    | **점수** | $\theta_0$에서의 **기울기** | $\theta_0$에서의 미분만 |
    | **우도비** | $\ell(\hat\theta)-\ell(\theta_0)$, **세로 거리** | 양쪽 모두 |

    ```python
    import numpy as np
    from scipy import stats

    n, k, p0 = 30, 22, 0.5
    ph = k / n

    wald = (ph - p0) / np.sqrt(ph * (1 - ph) / n)          # 분산을 p̂ 로
    score = (ph - p0) / np.sqrt(p0 * (1 - p0) / n)         # 분산을 p0 로
    G2 = 2 * (k * np.log(ph / p0) + (n - k) * np.log((1 - ph) / (1 - p0)))

    print(f"왈드   z = {wald:7.4f}   p = {2 * stats.norm.sf(abs(wald)):.4f}")
    print(f"점수   z = {score:7.4f}   p = {2 * stats.norm.sf(abs(score)):.4f}")
    print(f"우도비 G² = {G2:7.4f}   p = {stats.chi2.sf(G2, 1):.4f}")
    print(f"정확 이항        p = {stats.binomtest(k, n, p0).pvalue:.4f}")
    ```

    ```text
    왈드   z =  2.8900   p = 0.0039
    점수   z =  2.5560   p = 0.0106
    우도비 G² =  6.7939   p = 0.0091
    정확 이항        p = 0.0161
    ```

    **네 값이 0.0039에서 0.0161까지 네 배 차이**가 난다. 모두 같은 자료, 같은 가설이다.

    **왜 다른가.** 세 검정은 **점근적으로 동등**하지만 유한표본에서는 다르다.

    - **왈드**가 가장 작은 $p$를 준다. 분산을 $\hat p=0.733$에서 계산하는데, $\hat p(1-\hat p)=0.196$이 $p_0(1-p_0)=0.25$보다 작아 **표준오차를 과소평가**한다.
    - **점수**가 가장 보수적이다. $H_0$ 아래의 분산을 쓰므로 이 방향의 편향이 없다.
    - **우도비**가 중간이다.
    - **정확검정**이 가장 크다. 이산성의 보수성이 더해진다.

    **어느 것을 믿을 것인가.**

    1. **점수 검정이 일반적으로 가장 낫다.** $H_0$ 아래의 정보만 쓰므로 경계 근처에서 안정적이다. 비율에서는 이것이 윌슨 구간과 쌍대다.

    2. **왈드는 피한다.** 경계 근처에서 심각하게 나쁘고, **모수화에 의존**한다. $p$로 검정하느냐 $\log\frac p{1-p}$로 검정하느냐에 따라 답이 달라진다.

    3. **우도비는 불변이라 안전하다.** 계산이 조금 더 들지만 왈드보다 언제나 낫다.

    4. **$n$이 작으면 정확검정**을 쓴다.

    **왈드의 모수화 의존성 확인.**

    ```python
    eta = np.log(ph / (1 - ph))                   # 로짓 척도에서의 왈드
    se_eta = np.sqrt(1 / k + 1 / (n - k))
    w2 = eta / se_eta
    print(f"로짓 척도 왈드 z = {w2:.4f}   p = {2 * stats.norm.sf(abs(w2)):.4f}")
    ```

    ```text
    로짓 척도 왈드 z = 2.4502   p = 0.0143
    ```

    **같은 왈드 검정인데 $p$가 0.0039에서 0.0143으로 3.7배 바뀐다.** 점수와 우도비는 이런 일이 없다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
**순열검정**의 원리를 설명하고, 두 집단 비교에 적용하라. 어떤 가정이 필요한가?

</div>

??? success "풀이"
    **원리.** $H_0$이 "**두 집단의 분포가 같다**"면, 관측된 집단 표지는 자료와 무관하다. 따라서 표지를 무작위로 섞어 만든 통계량들이 $H_0$ 아래의 분포를 **정확히** 준다.

    ```python
    import numpy as np

    rng = np.random.default_rng(11)
    x = np.array([12.1, 15.3, 9.8, 14.2, 11.7, 13.5, 16.1, 10.9])
    y = np.array([17.2, 14.8, 19.3, 16.5, 15.9, 18.1, 13.7])
    obs = x.mean() - y.mean()

    pooled = np.concatenate([x, y])
    n1 = len(x)
    B = 99_999
    diffs = np.empty(B)
    for i in range(B):
        perm = rng.permutation(pooled)
        diffs[i] = perm[:n1].mean() - perm[n1:].mean()

    p_perm = (np.sum(np.abs(diffs) >= abs(obs)) + 1) / (B + 1)
    print(f"관측 차이 {obs:.4f}")
    print(f"순열검정 p = {p_perm:.4f}")

    from scipy import stats
    print(f"웰치 t 검정 p = {stats.ttest_ind(x, y, equal_var=False).pvalue:.4f}")
    ```

    ```text
    관측 차이 -3.5500
    순열검정 p = 0.0073
    웰치 t 검정 p = 0.0053
    ```

    **두 방법이 같은 결론을 준다**(0.0073 대 0.0053). 자료가 정규에 가까우면 대체로 그렇다. 순열검정의 장점은 **정규성을 가정하지 않고도** 이 결과를 얻는다는 점이다. 여기서 순열 쪽이 조금 큰 것은 순열분포의 이산성 때문이다.

    **필요한 가정 — 교환가능성.** $H_0$ 아래에서 관측값의 **순서를 바꿔도 결합분포가 변하지 않아야** 한다. 무작위 배정 실험에서는 설계가 이를 보장한다.

    **주의할 점 넷.**

    1. **$H_0$이 "분포가 같다"이지 "평균이 같다"가 아니다.** 두 집단의 분산이 다르면 평균이 같아도 교환가능성이 깨진다. 그때 순열검정을 "평균 차이 검정"으로 쓰면 **수준이 어긋난다.**

       - 대처: 스튜던트화한 통계량(웰치 $t$)을 순열분포에 넣으면 이 문제가 크게 완화된다.

    2. **$(b+1)/(B+1)$을 쓴다.** 앞서 본 이유 그대로이며, 순열검정에서는 **항등순열이 항상 유효한 순열**이라는 점에서 더 자연스럽다.

    3. **가능한 순열 수의 한계.** $n_1=8$, $n_2=7$이면 $\binom{15}8=6435$가지뿐이다. 도달 가능한 최소 $p$-값이 $1/6435=0.00016$이므로, 그보다 작은 $p$는 나올 수 없다. **소표본에서는 정확 열거**가 가능하고 더 낫다.

    4. **관측연구에서는 근거가 약하다.** 교환가능성이 설계로 보장되지 않으므로, 다른 검정과 같은 가정 문제를 안는다.

    **언제 쓰는가.**

    - **무작위 배정 실험**: 가장 자연스럽다. "무작위화 검정"이라 부르며, 분포가정이 전혀 필요 없다.
    - **비표준 통계량**: 중앙값 차이, 최댓값, 복잡한 요약통계의 분포를 이론적으로 구하기 어려울 때.
    - **소표본**: 점근이론을 믿을 수 없을 때.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
가설검정의 결과를 **논문에 어떻게 보고해야 하는지** 표준 양식을 만들어라.

</div>

??? success "풀이"
    **필수 요소 아홉.**

    1. **가설.** 무엇을 검정했는가. 단측인지 양측인지.
    2. **검정 방법.** 이름과 변형(웰치/합동, 정확/근사).
    3. **검정통계량과 자유도.** $t(28)=2.41$ 형태.
    4. **$p$-값.** 정확한 값. "$p<0.05$"가 아니라 "$p=0.023$".
    5. **효과크기.** 원 척도와 표준화 척도 둘 다면 더 좋다.
    6. **신뢰구간.** **가장 중요한 항목**이다.
    7. **표본크기.** 집단별로.
    8. **가정 확인.** 정규성, 등분산, 독립성을 어떻게 확인했는가.
    9. **다중비교 여부.** 몇 개의 검정을 했고 보정했는가.

    **좋은 보고의 예.**

    > 처리군($n=32$)의 평균 반응시간은 412 ms(SD 68), 대조군($n=30$)은 458 ms(SD 74)였다. 웰치의 이표본 $t$ 검정에서 $t(59.4)=-2.55$, 양측 $p=0.013$이었다. 평균 차이는 $-46$ ms(95% 신뢰구간 $-82$ ~ $-10$ ms), 코헨의 $d$는 $-0.65$(95% CI $-1.15$ ~ $-0.14$)다. 잔차의 Q-Q 그림에서 정규성 위배는 보이지 않았다. 이 분석은 사전등록된 주 결과이며, 부수 분석 세 건은 표 3에 별도로 보고한다.

    **나쁜 보고와 문제점.**

    | 서술 | 문제 |
    |---|---|
    | "두 군은 유의하게 달랐다($p<0.05$)" | 방향·크기·구간이 모두 없다 |
    | "$p=0.0000$" | 0이 될 수 없다. "$p<0.001$"로 |
    | "$t$ 검정 결과 차이가 있었다" | 어느 $t$ 검정인지, 통계량도 없음 |
    | "정규성을 만족했다($p=0.3$)" | 정규성 검정으로 정규성을 "확인"할 수 없다 |
    | "유의한 결과만 표에 실었다" | 선택적 보고. 전부 실어야 한다 |

    **자주 빠뜨리는 것.**

    - **효과의 방향.** "차이가 있었다"만 적고 어느 쪽이 큰지 안 적는 경우가 놀랍도록 많다.
    - **자유도.** 웰치는 소수 자유도가 나오는데 이를 적어야 어떤 검정인지 알 수 있다.
    - **제외된 관측값.** 몇 개를, 왜 뺐는지.
    - **탐색적 분석의 표시.** 사전 계획과 사후 분석을 구분하지 않으면 독자가 $p$-값을 올바로 해석할 수 없다.

    **한 줄 원칙.** **다른 연구자가 당신의 보고만 읽고 메타분석에 넣을 수 있어야 한다.** 그러려면 효과크기, 표준오차(또는 구간), 표본크기가 반드시 있어야 한다. $p$-값만으로는 불가능하다.

---

## 정리하며

가설검정의 전체 절차를 **한 페이지에서 훑었다.**

- **네 단계는 어느 검정이든 같다.** 가설을 세우고, 검정통계량을 계산하고, 그 표본분포에서 $p$ 값을 얻고, $\alpha$ 와 비교한다.
- **분포를 고르는 것이 자료가 아니라 가정이다.** $\sigma$ 를 알면 $z$, 모르면 $t_{n-1}$, 비율이면 정규근사, 분산이면 카이제곱이다.
- **일표본·이표본·대응이 같은 틀의 변주다.** 달라지는 것은 표준오차를 어떻게 만드느냐뿐이며, 대응 검정은 차이를 만들어 일표본으로 환원한다.
- **검정과 신뢰구간은 쌍대다.** 수준 $\alpha$ 양측검정이 $H_0:\theta=\theta_0$ 을 기각할 필요충분조건은 $\theta_0$ 이 $(1-\alpha)$ 신뢰구간 **밖**에 있는 것이다. 같은 계산을 다르게 보고할 뿐이다.
- **현대의 권고는 구간이 주, 검정이 보조다.** 구간은 효과의 크기와 정밀도를 함께 보여 주지만 검정은 이분법적 판정만 준다.

다음 절 **동전 던지기 모의실험**으로 넘어간다. 분포 공식을 전혀 쓰지 않고 $p$ 값을 얻는 방법이다.
