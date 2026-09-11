# 가설검정 시연

## 개요

가설검정은 표본자료에 근거하여 모수에 관한 결정을 내리는 형식적인 틀이다. 이 절차는 귀무가설 $H_0$(기본 주장)을 대립가설 $H_1$에 맞세우고, 검정통계량을 계산한 뒤 그 표본분포로 p-값을 얻는다. 이 페이지에서는 평균과 비율에 대한 일표본·이표본·대응 검정을 검정력 분석 및 신뢰구간과 가설검정의 쌍대성과 함께 시연한다.

## 평균에 대한 일표본 검정

모표준편차 $\sigma$를 모를 때는 **t-검정**을 쓴다. 표본평균이 $\bar{X}$이고 표본표준편차가 $S$인 확률표본 $X_1, \dots, X_n$이 주어졌을 때 $H_0\colon \mu = \mu_0$의 검정통계량은

$$
T = \frac{\bar{X} - \mu_0}{S / \sqrt{n}} \sim t_{n-1}.
$$

### 코드

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

**단측** 검정 $H_1\colon \mu < \mu_0$에서는 검정통계량이 대립가설 방향에 있을 때 양측 p-값을 2로 나눈다:

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

### 직접 계산

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

## 비율에 대한 일표본 검정

$\hat{p} = x/n$으로 $H_0\colon p = p_0$을 검정할 때 Wald 검정통계량은

$$
Z = \frac{\hat{p} - p_0}{\sqrt{p_0(1 - p_0)/n}} \;\dot\sim\; N(0,1).
$$

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

## 이표본 t-검정

두 모집단에서 얻은 독립표본에 대해 귀무가설은 $H_0\colon \mu_1 = \mu_2$이다. **Welch의 t-검정**(분산이 다른 경우)은

$$
T = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{S_1^2/n_1 + S_2^2/n_2}}
$$

를 쓰며 자유도는 Welch–Satterthwaite 근사로 계산한다.

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

## 대응 t-검정

관측값이 자연스러운 쌍을 이룰 때(예: 전후 측정) 차이 $D_i = X_i^{(\text{after})} - X_i^{(\text{before})}$를 계산하고 $H_0\colon \mu_D = 0$을 검정한다:

$$
T = \frac{\bar{D}}{S_D / \sqrt{n}} \sim t_{n-1}.
$$

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

## 검정력 분석

검정의 **검정력**은 $H_1$이 참일 때 $H_0$을 올바르게 기각할 확률이다:

$$
\text{Power} = 1 - \beta = P(\text{reject } H_0 \mid H_1 \text{ true}).
$$

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

## 신뢰구간과 검정의 쌍대성

수준 $\alpha$의 양측검정이 $H_0\colon \mu = \mu_0$을 기각하지 못할 필요충분조건은 $\mu_0$이 $100(1-\alpha)\%$ 신뢰구간 안에 있는 것이다.

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

### 해석

쌍대성 원리는 다음을 말한다:

$$
\mu_0 \notin \left(\bar{X} \pm t_{\alpha/2,\,n-1}\,\frac{S}{\sqrt{n}}\right) \iff \text{reject } H_0\colon \mu = \mu_0 \text{ at level } \alpha.
$$

이는 통합된 관점을 준다: 신뢰구간을 구성하는 일은 가설검정의 가족을 뒤집는 일과 같다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** 전구 $n=16$개의 표본에서 $\bar{x}=1020$시간, $s=80$시간을 얻었다. $\alpha=0.05$에서 $H_0\colon \mu=1000$ 대 $H_1\colon \mu>1000$을 검정하라. 검정통계량, p-값, 판정을 제시하라.

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

**연습문제 2.** 유권자 400명 조사에서 228명이 어떤 안건을 지지한다. 1% 유의수준에서 참 비율이 0.50을 넘는지 검정하라.

</div>

??? success "풀이"

    $\hat{p} = 228/400 = 0.57$이고 $H_0\colon p = 0.50$ 대 $H_1\colon p > 0.50$을 검정한다. 검정통계량은

    $$
    Z = \frac{0.57 - 0.50}{\sqrt{0.50 \times 0.50 / 400}} = \frac{0.07}{0.025} = 2.80.
    $$

    단측 p-값은 $P(Z \geq 2.80) \approx 0.0026 < 0.01$이므로 $H_0$을 기각한다. 유권자의 절반 넘는 비율이 이 안건을 지지한다는 강한 증거이다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** 독립인 두 집단에서 $\bar{x}_1=75$, $s_1=10$, $n_1=20$과 $\bar{x}_2=70$, $s_2=12$, $n_2=25$를 얻었다. $\alpha=0.05$에서 $H_0\colon \mu_1 = \mu_2$의 Welch t-검정을 수행하라. Welch–Satterthwaite 자유도의 공식을 함께 쓰라.

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

**연습문제 4.** 신뢰구간과 검정의 쌍대성을 증명하라. 즉 수준 $\alpha$의 양측 일표본 t-검정에서 $H_0\colon \mu = \mu_0$을 기각하는 것이 $\mu_0$이 $100(1-\alpha)\%$ 신뢰구간 밖에 있는 것과 동등함을 보여라.

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

**연습문제 5.** 일표본 z-검정의 검정력 공식을 써서, ($\sigma$를 아는) 양측검정에서 이동 $\delta$를 유의수준 $\alpha$에서 검정력 $1-\beta$로 탐지하는 데 필요한 표본크기가

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
