# 신뢰구간 ↔ 가설검정의 쌍대성

## 쌍대성 원리

신뢰구간과 가설검정 사이에는 깊은 연관이 있다. $(1 - \alpha) \times 100\%$ 신뢰구간과 유의수준 $\alpha$의 가설검정은 같은 동전의 양면이다:

> **수준 $\alpha$의 양측 가설검정이 $H_0: \theta = \theta_0$을 기각할 필요충분조건은 $\theta_0$이 $\theta$의 $(1-\alpha) \times 100\%$ 신뢰구간 밖에 있는 것이다.**

이 쌍대성 덕분에 신뢰구간을 살펴 가설검정을 수행할 수 있고, 그 반대도 가능하다.

## 쌍대성이 작동하는 방식

### 신뢰구간에서 가설검정으로

모수 $\theta$에 대한 $(1 - \alpha) \times 100\%$ 신뢰구간 $(L, U)$가 주어졌을 때:

- $\theta_0 \in (L, U)$이면 유의수준 $\alpha$에서 $H_0: \theta = \theta_0$을 기각하지 못한다.
- $\theta_0 \notin (L, U)$이면 유의수준 $\alpha$에서 $H_0: \theta = \theta_0$을 기각한다.

### 가설검정에서 신뢰구간으로

$(1 - \alpha) \times 100\%$ 신뢰구간은 유의수준 $\alpha$에서 가설검정 $H_0: \theta = \theta_0$이 기각되지 **않는** 모든 $\theta_0$의 집합이다.

$$CI_{1-\alpha} = \{\theta_0 : \text{fail to reject } H_0: \theta = \theta_0 \text{ at level } \alpha\}$$

## 예제

### 예제 1: 일표본 평균

$H_0: \mu = \mu_0$ 대 $H_a: \mu \neq \mu_0$의 일표본 z-검정에서:

- **검정**: $z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}$일 때 $|z| > z_{\alpha/2}$이면 $H_0$을 기각한다.
- **신뢰구간**: $\bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$

이 검정이 $H_0$을 기각할 필요충분조건은 $\mu_0$이 신뢰구간 밖에 있는 것이다.

**동등성의 증명:**

$$|z| > z_{\alpha/2} \iff \left|\frac{\bar{x} - \mu_0}{\sigma/\sqrt{n}}\right| > z_{\alpha/2} \iff \mu_0 \notin \left(\bar{x} - z_{\alpha/2}\frac{\sigma}{\sqrt{n}},\ \bar{x} + z_{\alpha/2}\frac{\sigma}{\sqrt{n}}\right)$$

### 예제 2: 두 품종의 배

Yuna가 Bosc 배와 Anjou 배의 열량을 비교한다. $\mu_{\text{Bosc}} - \mu_{\text{Anjou}}$의 99% 신뢰구간은 $4 \pm 6.44 = (-2.44, 10.44)$이다.

$\alpha = 0.01$에서 $H_0: \mu_{\text{Bosc}} = \mu_{\text{Anjou}}$(즉 $\mu_{\text{Bosc}} - \mu_{\text{Anjou}} = 0$)를 검정하면:

$0 \in (-2.44, 10.44)$이므로 $H_0$을 **기각하지 못한다**. 열량이 다르다고 결론지을 증거가 부족하다.

### 예제 3: 대면 수업과 온라인 수업

$p_{\text{in\_person}} - p_{\text{online}}$의 95% 신뢰구간이 $(-0.04, 0.14)$이다.

$\alpha = 0.05$에서 $H_0: p_{\text{in\_person}} = p_{\text{online}}$을 검정하면:

$0 \in (-0.04, 0.14)$이므로 $H_0$을 **기각하지 못한다**. 합격률에 유의한 차이가 없다.

## 단측검정과 신뢰구간

쌍대성은 단측 신뢰구간(신뢰한계)을 쓰면 단측검정으로도 확장된다:

- **상한 신뢰한계**: 신뢰수준 $1 - \alpha$에서 $\theta < U$는 검정 $H_0: \theta \geq \theta_0$ 대 $H_a: \theta < \theta_0$에 대응한다.
- **하한 신뢰한계**: 신뢰수준 $1 - \alpha$에서 $\theta > L$은 검정 $H_0: \theta \leq \theta_0$ 대 $H_a: \theta > \theta_0$에 대응한다.

## Python 예시

```python
import numpy as np
from scipy import stats

# Sample data
x_bar = 52
mu_0 = 50
sigma = 10
n = 25
alpha = 0.05

# Hypothesis test approach
z = (x_bar - mu_0) / (sigma / np.sqrt(n))
p_value = 2 * stats.norm.sf(abs(z))
reject_test = p_value <= alpha

# Confidence interval approach
z_crit = stats.norm.ppf(1 - alpha / 2)
ci_lower = x_bar - z_crit * sigma / np.sqrt(n)
ci_upper = x_bar + z_crit * sigma / np.sqrt(n)
reject_ci = mu_0 < ci_lower or mu_0 > ci_upper

print(f"Test: z = {z:.4f}, p-value = {p_value:.4f}, Reject = {reject_test}")
print(f"CI: ({ci_lower:.4f}, {ci_upper:.4f}), mu_0 outside CI = {reject_ci}")
print(f"Both methods agree: {reject_test == reject_ci}")
```

## 핵심 요약

- 양측검정에서 신뢰구간과 가설검정은 동등한 정보를 준다.
- 신뢰구간은 기각/비기각의 이분법적 판정만이 아니라 그럴듯한 값의 범위를 보여주므로 흔히 더 유익하다.
- 결과를 보고할 때는 p-값과 신뢰구간을 함께 제시하는 것이 좋다.
- 쌍대성은 양측검정에서 정확히 성립하며, 단측검정은 단측 신뢰한계에 대응한다.

## 연습문제

**연습문제 1.**
$\mu$의 95% 신뢰구간이 $(12.3, 18.7)$이다. 검정통계량을 계산하지 않고 $\alpha = 0.05$에서 $H_0: \mu = 10$의 양측검정 결과를 판단하라.

??? success "연습문제 1 풀이"
    $\mu_0 = 10$이 95% 신뢰구간 $(12.3, 18.7)$ **밖에** 있으므로 $\alpha = 0.05$에서 $H_0: \mu = 10$을 **기각한다**. 신뢰구간과 가설검정의 쌍대성에 의해 $(1-\alpha)$ 신뢰구간 밖의 어떤 값도 유의수준 $\alpha$에서 기각된다.

---

**연습문제 2.**
어떤 연구자가 $H_0: \mu = 50$의 양측검정을 수행하여 p-값 0.03을 얻었다. 50이 95%와 99% 신뢰구간의 안에 있는지 밖에 있는지 무엇을 결론지을 수 있는가?

??? success "연습문제 2 풀이"
    $p = 0.03 < 0.05$이므로 이 검정은 $\alpha = 0.05$에서 $H_0$을 기각한다. 쌍대성에 의해 $\mu_0 = 50$은 95% 신뢰구간 **밖에** 있다.

    $p = 0.03 > 0.01$이므로 $\alpha = 0.01$에서는 $H_0$을 기각하지 않는다. 쌍대성에 의해 $\mu_0 = 50$은 99% 신뢰구간 **안에** 있다.

---

**연습문제 3.**
수학적으로 동등한데도 신뢰구간이 가설검정보다 많은 정보를 주는 이유를 설명하라.

??? success "연습문제 3 풀이"
    가설검정은 가설의 값 $\mu_0$ 하나에 대해 기각이냐 비기각이냐라는 이분법적 판정을 낸다. 신뢰구간은 어떤 $\mu_0$ 값들이 기각되고 어떤 값들이 기각되지 않는지를 한꺼번에 보여준다. 다음을 제공한다:

    1. 효과의 **방향**(추정값이 $\mu_0$보다 위인가 아래인가?).
    2. 효과의 **크기**(추정값이 $\mu_0$에서 얼마나 먼가?).
    3. 추정의 **정밀도**(구간이 얼마나 넓은가?).

    예를 들어 신뢰구간 $(0.1, 15.2)$와 $(7.5, 7.8)$은 모두 5% 수준에서 $\mu_0 = 0$을 기각하지만, 앞의 것은 매우 불확실한 추정을, 뒤의 것은 7.65 근처의 정밀한 추정을 시사한다.

---

**연습문제 4.**
신뢰구간과 가설검정의 쌍대성이 단측검정에서도 성립하는가? 그렇다면 대응하는 신뢰한계는 무엇인가?

??? success "연습문제 4 풀이"
    쌍대성은 단측검정으로도 확장되지만, 대응하는 것은 양측 구간이 아니라 **단측 신뢰한계**이다.

    수준 $\alpha$에서 단측검정 $H_0: \mu \leq \mu_0$ 대 $H_1: \mu > \mu_0$에 대응하는 것은 하한 신뢰한계 $(\bar{x} - z_\alpha \cdot \text{SE},\; \infty)$이다. $\mu_0$이 이 하한 아래에 있을 때에만 $H_0$을 기각한다.

    마찬가지로 $H_0: \mu \geq \mu_0$ 대 $H_1: \mu < \mu_0$에 대응하는 것은 상한 신뢰한계 $(-\infty,\; \bar{x} + z_\alpha \cdot \text{SE})$이다. 단측 한계는 검정의 단측 성격을 반영하여 $z_{\alpha/2}$가 아니라 $z_\alpha$를 쓴다.
