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

# Factory claims mean weight is 500g
data = np.array([498, 495, 502, 497, 501, 499, 496, 503, 494, 500,
                 497, 502, 496, 501, 498, 499, 495, 503, 497, 500])
mu_0 = 500
alpha = 0.05

# t-test (sigma unknown)
t_stat, p_value = stats.ttest_1samp(data, mu_0)
print(f"t = {t_stat:.4f}, p-value = {p_value:.4f}")
print(f"Decision: {'Reject H0' if p_value < alpha else 'Fail to reject H0'}")
```

**단측** 검정 $H_1\colon \mu < \mu_0$에서는 검정통계량이 대립가설 방향에 있을 때 양측 p-값을 2로 나눈다:

```python
p_one_sided = p_value / 2 if t_stat < 0 else 1 - p_value / 2
```

### 직접 계산

```python
n = len(data)
xbar = data.mean()
s = data.std(ddof=1)
t_manual = (xbar - mu_0) / (s / np.sqrt(n))
p_manual = 2 * stats.t.cdf(-abs(t_manual), df=n - 1)
```

## 비율에 대한 일표본 검정

$\hat{p} = x/n$으로 $H_0\colon p = p_0$을 검정할 때 Wald 검정통계량은

$$
Z = \frac{\hat{p} - p_0}{\sqrt{p_0(1 - p_0)/n}} \;\dot\sim\; N(0,1).
$$

```python
from statsmodels.stats.proportion import proportions_ztest

x, n = 12, 200  # 12 defectives in 200
p_0 = 0.05
p_hat = x / n

z_stat = (p_hat - p_0) / np.sqrt(p_0 * (1 - p_0) / n)
p_value = 2 * stats.norm.sf(abs(z_stat))

# Or using statsmodels
z_sm, p_sm = proportions_ztest(x, n, value=p_0)
```

## 이표본 t-검정

두 모집단에서 얻은 독립표본에 대해 귀무가설은 $H_0\colon \mu_1 = \mu_2$이다. **Welch의 t-검정**(분산이 다른 경우)은

$$
T = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{S_1^2/n_1 + S_2^2/n_2}}
$$

를 쓰며 자유도는 Welch–Satterthwaite 근사로 계산한다.

```python
drug_a = np.array([5.2, 4.8, 6.1, 5.5, 4.9, 5.7, 5.3, 6.0, 5.1, 5.4])
drug_b = np.array([4.1, 3.8, 4.5, 4.2, 3.9, 4.6, 4.0, 4.3, 3.7, 4.4])

# Welch's t-test (default: unequal variances)
t_welch, p_welch = stats.ttest_ind(drug_a, drug_b, equal_var=False)

# Pooled t-test (equal variances assumed)
t_pooled, p_pooled = stats.ttest_ind(drug_a, drug_b, equal_var=True)
```

## 대응 t-검정

관측값이 자연스러운 쌍을 이룰 때(예: 전후 측정) 차이 $D_i = X_i^{(\text{after})} - X_i^{(\text{before})}$를 계산하고 $H_0\colon \mu_D = 0$을 검정한다:

$$
T = \frac{\bar{D}}{S_D / \sqrt{n}} \sim t_{n-1}.
$$

```python
before = np.array([145, 150, 138, 155, 142, 148, 136, 152, 140, 146])
after  = np.array([138, 142, 130, 148, 135, 140, 132, 145, 134, 139])

t_stat, p_value = stats.ttest_rel(after, before)

# Equivalent to one-sample t-test on the differences
diff = after - before
t_stat2, p_value2 = stats.ttest_1samp(diff, 0)
```

## 검정력 분석

검정의 **검정력**은 $H_1$이 참일 때 $H_0$을 올바르게 기각할 확률이다:

$$
\text{Power} = 1 - \beta = P(\text{reject } H_0 \mid H_1 \text{ true}).
$$

```python
from statsmodels.stats.power import TTestPower, TTestIndPower

# One-sample: detect a 5-point difference with sigma=15
analysis = TTestPower()
effect_size = 5 / 15  # Cohen's d
n_needed = analysis.solve_power(effect_size=effect_size, alpha=0.05,
                                 power=0.80, alternative='two-sided')

# Two-sample: medium effect size
analysis2 = TTestIndPower()
n_each = analysis2.solve_power(effect_size=0.5, alpha=0.05, power=0.80,
                                ratio=1.0, alternative='two-sided')
```

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

# Test various mu_0 values
for mu0 in [48, 49, 50, 51, 52, 53]:
    t_stat, p_val = stats.ttest_1samp(data, mu0)
    in_ci = ci[0] <= mu0 <= ci[1]
    reject = p_val < alpha
    # reject <=> mu0 NOT in CI
```

### 해석

쌍대성 원리는 다음을 말한다:

$$
\mu_0 \notin \left(\bar{X} \pm t_{\alpha/2,\,n-1}\,\frac{S}{\sqrt{n}}\right) \iff \text{reject } H_0\colon \mu = \mu_0 \text{ at level } \alpha.
$$

이는 통합된 관점을 준다: 신뢰구간을 구성하는 일은 가설검정의 가족을 뒤집는 일과 같다.

## 연습문제

**연습문제 1.** 전구 $n=16$개의 표본에서 $\bar{x}=1020$시간, $s=80$시간을 얻었다. $\alpha=0.05$에서 $H_0\colon \mu=1000$ 대 $H_1\colon \mu>1000$을 검정하라. 검정통계량, p-값, 판정을 제시하라.

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

---

**연습문제 2.** 유권자 400명 조사에서 228명이 어떤 안건을 지지한다. 1% 유의수준에서 참 비율이 0.50을 넘는지 검정하라.

??? success "풀이"

    $\hat{p} = 228/400 = 0.57$이고 $H_0\colon p = 0.50$ 대 $H_1\colon p > 0.50$을 검정한다. 검정통계량은

    $$
    Z = \frac{0.57 - 0.50}{\sqrt{0.50 \times 0.50 / 400}} = \frac{0.07}{0.025} = 2.80.
    $$

    단측 p-값은 $P(Z \geq 2.80) \approx 0.0026 < 0.01$이므로 $H_0$을 기각한다. 유권자의 절반 넘는 비율이 이 안건을 지지한다는 강한 증거이다. $\square$

---

**연습문제 3.** 독립인 두 집단에서 $\bar{x}_1=75$, $s_1=10$, $n_1=20$과 $\bar{x}_2=70$, $s_2=12$, $n_2=25$를 얻었다. $\alpha=0.05$에서 $H_0\colon \mu_1 = \mu_2$의 Welch t-검정을 수행하라. Welch–Satterthwaite 자유도의 공식을 함께 쓰라.

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

---

**연습문제 4.** 신뢰구간과 검정의 쌍대성을 증명하라. 즉 수준 $\alpha$의 양측 일표본 t-검정에서 $H_0\colon \mu = \mu_0$을 기각하는 것이 $\mu_0$이 $100(1-\alpha)\%$ 신뢰구간 밖에 있는 것과 동등함을 보여라.

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

---

**연습문제 5.** 일표본 z-검정의 검정력 공식을 써서, ($\sigma$를 아는) 양측검정에서 이동 $\delta$를 유의수준 $\alpha$에서 검정력 $1-\beta$로 탐지하는 데 필요한 표본크기가

$$
n = \left(\frac{(z_{\alpha/2} + z_\beta)\,\sigma}{\delta}\right)^2
$$

임을 보여라.

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
