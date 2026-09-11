# 두 표본평균 차의 표본분포

## 개요

두 모집단을 비교할 때는 흔히 차 $\bar{X}_1 - \bar{X}_2$를 살핀다. 이 차의 표본분포가 적절한 검정통계량, 신뢰구간 공식, 기준분포를 결정하며, 이는 모분산에 관해 무엇을 아는지와 표본크기에 따라 달라진다.

## 설정

$X_1^{(1)}, \dots, X_{n_1}^{(1)}$을 평균 $\mu_1$, 분산 $\sigma_1^2$인 모집단 1에서 뽑은 i.i.d. 표본이라 하고, $X_1^{(2)}, \dots, X_{n_2}^{(2)}$를 평균 $\mu_2$, 분산 $\sigma_2^2$인 모집단 2에서 뽑은 i.i.d. 표본이라 하자. 두 표본은 **독립**이라고 가정한다.

### 공통 성질 (모든 경우)

$$
E[\bar{X}_1 - \bar{X}_2] = \mu_1 - \mu_2
$$

$$
\text{Var}(\bar{X}_1 - \bar{X}_2) = \frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}
$$

## 경우 A: 모분산을 아는 경우

$\sigma_1^2$과 $\sigma_2^2$을 알 때:

$$
Z = \frac{(\bar{X}_1 - \bar{X}_2) - (\mu_1 - \mu_2)}{\sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}} \sim N(0, 1)
$$

**신뢰구간:**

$$
(\bar{X}_1 - \bar{X}_2) \pm z_{\alpha/2} \sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}
$$

## 경우 B: 표본크기가 큰 경우

$n_1$과 $n_2$가 모두 크면(중심극한정리가 적용된다) $\sigma_i^2$을 $S_i^2$으로 대체한다:

$$
Z = \frac{(\bar{X}_1 - \bar{X}_2) - (\mu_1 - \mu_2)}{\sqrt{\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}}} \approx N(0, 1)
$$

**신뢰구간:**

$$
(\bar{X}_1 - \bar{X}_2) \pm z_{\alpha/2} \sqrt{\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}}
$$

## 경우 C: 정규모집단, 등분산 (합동 t)

두 모집단이 모두 정규이고 $\sigma_1^2 = \sigma_2^2 = \sigma^2$일 때:

$$
T = \frac{(\bar{X}_1 - \bar{X}_2) - (\mu_1 - \mu_2)}{\sqrt{S_p^2\!\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}} \sim t_{n_1 + n_2 - 2}
$$

여기서 **합동분산**은:

$$
S_p^2 = \frac{(n_1 - 1)S_1^2 + (n_2 - 1)S_2^2}{n_1 + n_2 - 2}
= \frac{\sum_{i=1}^{n_1}(X_i^{(1)} - \bar{X}_1)^2 + \sum_{i=1}^{n_2}(X_i^{(2)} - \bar{X}_2)^2}{n_1 + n_2 - 2}
$$

**신뢰구간:**

$$
(\bar{X}_1 - \bar{X}_2) \pm t_{\alpha/2, \, n_1+n_2-2} \sqrt{S_p^2\!\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}
$$

## 경우 D: 정규모집단, 이분산 (Welch의 t)

두 모집단이 모두 정규이지만 $\sigma_1^2 \neq \sigma_2^2$일 때:

$$
T = \frac{(\bar{X}_1 - \bar{X}_2) - (\mu_1 - \mu_2)}{\sqrt{\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}}} \sim t_\nu
$$

여기서 **Welch–Satterthwaite 자유도**는:

$$
\nu = \frac{\left(\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}\right)^2}{\frac{\left(\frac{S_1^2}{n_1}\right)^2}{n_1} + \frac{\left(\frac{S_2^2}{n_2}\right)^2}{n_2}}
$$

**신뢰구간:**

$$
(\bar{X}_1 - \bar{X}_2) \pm t_{\alpha/2, \, \nu} \sqrt{\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}}
$$

## 경우 E: 보수적인 자유도

Welch 공식을 쓰기 번거로울 때는 보수적인(안전한) 대안을 쓴다:

$$
\text{df} = \min(n_1 - 1, \; n_2 - 1)
$$

이는 참 자유도를 언제나 과소추정하므로 신뢰구간이 더 넓어진다.

$$
T = \frac{(\bar{X}_1 - \bar{X}_2) - (\mu_1 - \mu_2)}{\sqrt{\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}}} \sim t_{\min(n_1-1, \, n_2-1)}
$$

## 선택 지침

| 조건 | 통계량 | 기준분포 |
|-----------|-----------|----------------------|
| $\sigma_1^2, \sigma_2^2$을 앎 | $Z$ | $N(0,1)$ |
| $n_1, n_2$가 큼 | $Z$ | $N(0,1)$ (근사) |
| 정규, $\sigma_1^2 = \sigma_2^2$ | 합동 $t$ | $t_{n_1+n_2-2}$ |
| 정규, $\sigma_1^2 \neq \sigma_2^2$ | Welch의 $t$ | $t_\nu$ (Satterthwaite) |
| 정규, 빠른 근사 | 보수적 $t$ | $t_{\min(n_1-1, n_2-1)}$ |

## 예: 두 교대조의 컵케이크

**문제.** 어떤 제과점에 두 교대조가 있다. A 교대조: $\mu_A = 130$g, $\sigma_A = 4$g. B 교대조: $\mu_B = 125$g, $\sigma_B = 3$g. $n_A = n_B = 40$일 때 $P(|\bar{X}_A - \bar{X}_B| > 6)$을 구하라.

**풀이.** $\sigma_A, \sigma_B$를 알고 있으므로 경우 A이다:

$$
\text{SE} = \sqrt{\frac{4^2}{40} + \frac{3^2}{40}} = \sqrt{\frac{16 + 9}{40}} = \sqrt{0.625} \approx 0.7906
$$

**상단꼬리:**

$$
Z = \frac{6 - (130 - 125)}{0.7906} = \frac{1}{0.7906} \approx 1.265
$$

$$
P(\bar{X}_A - \bar{X}_B > 6) = P(Z > 1.265) \approx 0.1030
$$

**하단꼬리:**

$$
Z = \frac{-6 - (130 - 125)}{0.7906} = \frac{-11}{0.7906} \approx -13.91
$$

$$
P(\bar{X}_A - \bar{X}_B < -6) \approx 0.0000
$$

**답:** $P(|\bar{X}_A - \bar{X}_B| > 6) \approx 0.1030$.

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

## 예: 차의 표준오차

**문제.** 모집단 A: $\mu_A = 100$, $\sigma_A = 15$, $n_A = 36$. 모집단 B: $\mu_B = 110$, $\sigma_B = 20$, $n_B = 49$. $\text{SE}(\bar{X}_A - \bar{X}_B)$를 구하라.

**풀이.**

$$
\text{SE} = \sqrt{\frac{15^2}{36} + \frac{20^2}{49}} = \sqrt{6.25 + 8.16} = \sqrt{14.41} \approx 3.80
$$

## 요약

| 경우 | 핵심 조건 | 분포 | 자유도 |
|------|--------------|-------------|-----|
| A | $\sigma$를 앎 | $Z$ | — |
| B | $n$이 큼 | $Z$ (근사) | — |
| C | 정규, $\sigma$가 같음 | $t$ (합동) | $n_1 + n_2 - 2$ |
| D | 정규, $\sigma$가 다름 | $t$ (Welch) | Satterthwaite |
| E | 정규, 빠른 근사 | $t$ (보수적) | $\min(n_1-1, n_2-1)$ |

모든 경우에 신뢰구간은 다음 형태를 취한다:

$$
(\bar{X}_1 - \bar{X}_2) \pm (\text{critical value}) \times \text{SE}
$$

임계값($z^*$ 또는 $t^*$)과 표준오차 공식의 선택이 경우에 따라 달라진다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
모집단 A: $\sigma_A = 15$, $n_A = 36$. 모집단 B: $\sigma_B = 20$, $n_B = 49$. $\mathrm{SE}(\bar X_A - \bar X_B)$를 계산하라.

</div>

??? success "풀이"
    독립성에 의해:

    $$
    \mathrm{Var}(\bar X_A - \bar X_B) = \sigma_A^2/n_A + \sigma_B^2/n_B = 225/36 + 400/49 = 6.25 + 8.16 = 14.41
    $$

    $\mathrm{SE} \approx 3.80$.

<div class="drillbox" markdown>

**연습문제 2.**
**차의 분포.** $\bar X_A, \bar X_B$가 독립이고 (근사적으로) 정규일 때 $\bar X_A - \bar X_B$의 분포를 유도하라.

</div>

??? success "풀이"
    독립인 정규확률변수의 선형결합은 정규분포이다. 따라서:

    $$
    \bar X_A - \bar X_B \sim N(\mu_A - \mu_B, \sigma_A^2/n_A + \sigma_B^2/n_B)
    $$

    $H_0: \mu_A = \mu_B$ 아래에서 중심화된 통계량은 $Z = (\bar X_A - \bar X_B)/\sqrt{\sigma_A^2/n_A + \sigma_B^2/n_B} \sim N(0, 1)$이다.

    $\sigma_A, \sigma_B$를 모르면 $s_A, s_B$로 대체하고 (자유도를 조정한 Welch의) $t$ 분포를 사용한다.

<div class="drillbox" markdown>

**연습문제 3.**
**가설검정.** 표본 A: $\bar x_A = 45$, $s_A = 10$, $n_A = 50$. 표본 B: $\bar x_B = 40$, $s_B = 12$, $n_B = 50$. $\alpha = 0.05$에서 $H_0: \mu_A = \mu_B$ 대 $H_1: \mu_A \ne \mu_B$를 검정하라.

</div>

??? success "풀이"
    $\mathrm{SE} = \sqrt{100/50 + 144/50} = \sqrt{4.88} \approx 2.21$.

    $t = (45 - 40)/2.21 \approx 2.26$.

    Welch 자유도: $\nu = (\mathrm{SE}^2)^2/[(s_A^2/n_A)^2/(n_A - 1) + (s_B^2/n_B)^2/(n_B - 1)] \approx (4.88)^2/[(2)^2/49 + (2.88)^2/49] = 23.8/0.250 \approx 95.5$. 95로 반올림한다.

    임계값: $t_{0.975, 95} \approx 1.985$.

    $|t| = 2.26 > 1.985$이므로 **$H_0$을 기각한다**. 5% 수준에서 평균에 차이가 있다는 증거이다.

    $p$ 값: $2 \cdot P(T_{95} > 2.26) \approx 2 \cdot 0.013 = 0.026$.

<div class="drillbox" markdown>

**연습문제 4.**
**신뢰구간.** 연습문제 3에 이어 $\mu_A - \mu_B$에 대한 95% 신뢰구간을 구성하라.

</div>

??? success "풀이"
    $\mathrm{CI} = (\bar x_A - \bar x_B) \pm t_{0.975, 95} \cdot \mathrm{SE} = 5 \pm 1.985 \cdot 2.21 = 5 \pm 4.39$.

    95% 신뢰구간: $(0.61, 9.39)$.

    0을 포함하지 *않으며*, 이는 $H_0$을 기각한 것과 일관된다. 자료는 A의 평균이 B보다 0.6에서 9.4단위 크다고 시사한다.

    신뢰구간은 대체로 $p$ 값보다 정보가 많다. "유의한가 아닌가"라는 이분법이 아니라 차이의 크기와 방향을 알려 준다.

<div class="drillbox" markdown>

**연습문제 5.**
**대응표본과 독립표본.** 두 표본 $t$ 검정은 표본의 독립성을 가정한다. 자료가 **대응**되어 있을 때(예: 같은 대상의 사전/사후 측정) 적절한 검정을 서술하고 대응 분석이 왜 검정력이 더 높은지 설명하라.

</div>

??? success "풀이"
    **대응표본 $t$ 검정:** 각 짝 $i = 1, \ldots, n$에 대해 차 $D_i = X_{1i} - X_{2i}$를 계산한다. $D_i$들에 대해 일표본 $t$ 검정으로 $H_0: \mathbb{E}[D] = 0$을 검정한다.

    검정통계량: 짝들이 i.i.d.라는 가정 아래 $H_0$에서 $t = \bar D/(s_D/\sqrt n) \sim t_{n-1}$이다.

    **왜 검정력이 더 높은가:** 짝짓기가 대상 간 변동성을 제거한다. (같은 대상의 반복측정에서 흔히 그렇듯) $\mathrm{Cov}(X_1, X_2) > 0$이면

    $$
    \mathrm{Var}(D) = \mathrm{Var}(X_1) + \mathrm{Var}(X_2) - 2\mathrm{Cov}(X_1, X_2) < \mathrm{Var}(X_1) + \mathrm{Var}(X_2)
    $$

    분산이 작아지면 ⟹ 표준오차가 작아지고 ⟹ 검정력이 높아진다.

    설계가 대응되어 있으면 언제나 대응 분석을 사용하라. 짝짓기를 무시하면 정보를 낭비하고 검정력을 잃는다.

<div class="drillbox" markdown>

**연습문제 6.**
**동등성 검정.** 두 평균이 허용범위 $\delta = 2$ 안에서 실질적으로 **동등한지** 검정한다. **TOST 절차**(두 개의 단측검정)의 귀무가설과 대립가설을 쓰라.

</div>

??? success "풀이"
    전통적인 검정은 $H_0: \mu_A = \mu_B$ 대 $H_1: \mu_A \ne \mu_B$이다. 기각하지 못했다고 해서 동등성이 증명되는 것은 아니다. 검정력이 부족했을 수도 있다.

    **동등성 검정(TOST):** 귀무가설과 대립가설을 맞바꾼다.

    $H_0: |\mu_A - \mu_B| \ge \delta$ (동등하지 않음) 대 $H_1: |\mu_A - \mu_B| < \delta$ (동등함).

    구현: 유의수준 $\alpha$의 단측검정 두 개를 수행한다.

    - $H_{01}: \mu_A - \mu_B \le -\delta$ 대 $H_{11}: \mu_A - \mu_B > -\delta$를 검정한다.
    - $H_{02}: \mu_A - \mu_B \ge \delta$ 대 $H_{12}: \mu_A - \mu_B < \delta$를 검정한다.
    - 두 단측검정이 모두 기각할 때에만 $H_0$을 기각한다(동등하다고 선언한다).

    신뢰구간 $(\bar X_A - \bar X_B) \pm t_{1-\alpha} \cdot \mathrm{SE}$가 온전히 $(-\delta, +\delta)$ 안에 들어가는 것과 동등하다.

    생물학적 동등성 시험(FDA는 두 제형이 $\pm 20\%$ 이내에서 동등함을 보이도록 요구한다), A/B 테스트의 "해가 없음" 확인, 그리고 "의미 있는 차이가 없다"가 원하는 결론인 모든 상황에서 쓰인다.
