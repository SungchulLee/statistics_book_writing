# 이표본 t-검정 (합동과 Welch)

## 개요

이표본 t-검정은 독립인 두 표본의 평균을 비교하여 유의하게 다른지 판단한다. A/B 검정, 임상시험, 실험 설계에서 널리 쓰인다.

## 가설

- **귀무가설** ($H_0$): $\mu_1 = \mu_2$ (평균이 같다)
- **대립가설** ($H_a$): $\mu_1 \neq \mu_2$ (평균이 다르다)

## 검정통계량

### 합동 t-검정 (등분산 가정)

두 집단의 모분산이 같다고 가정할 때:

$$
t = \frac{\bar{X}_1 - \bar{X}_2}{S_p\sqrt{1/n_1 + 1/n_2}}
$$

여기서 합동 표준편차는:

$$S_p^2 = \frac{(n_1-1)S_1^2 + (n_2-1)S_2^2}{n_1+n_2-2}$$

**자유도**: $df = n_1 + n_2 - 2$

### Welch t-검정 (분산이 다를 수 있는 경우)

분산이 다를 수 있으면 Welch 검정이 더 로버스트하다:

$$
t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{S_1^2/n_1 + S_2^2/n_2}}
$$

**자유도** (Satterthwaite 근사):

$$df = \frac{\left(\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}\right)^2}{\frac{(S_1^2/n_1)^2}{n_1-1} + \frac{(S_2^2/n_2)^2}{n_2-1}}$$

**참고**: Welch 검정은 등분산을 가정하지 않으면서 제1종 오류를 잘 통제하므로 일반적으로 선호된다.

## 실무적 고려사항

### 등분산 가정

합동 검정과 Welch 검정 중 하나를 고르기 전에 (Levene 검정 같은) 등분산 사전검정을 하고 싶을 수 있다. 그러나 현대의 통계 실무는 다음 이유로 Welch 검정을 기본으로 쓰기를 권한다:

1. 등분산 가정이 깨져도 로버스트하다
2. 실제로 분산이 같을 때도 합동 검정과 검정력이 거의 같다
3. 분산이 다를 때 더 나은 보호를 제공한다

### 효과크기

이표본 비교에서 **Cohen의 d**가 실질적 유의성을 잰다:

$$d = \frac{\bar{X}_1 - \bar{X}_2}{S_p}$$

해석:

- $|d| < 0.2$: 작은 효과
- $0.2 \leq |d| < 0.5$: 작은~중간 효과
- $0.5 \leq |d| < 0.8$: 중간 효과
- $|d| \geq 0.8$: 큰 효과

## 예제: 웹페이지 A/B 검정

새로 디자인한 웹페이지(페이지 B)에서 사용자가 기존 버전(페이지 A)보다 더 오래 머무는지 검정한다고 하자:

```python
import numpy as np
from scipy import stats

# Session times in seconds
page_a = np.array([185, 188, 142, 160, 161, 157, 182, 181, 159, 167])
page_b = np.array([173, 181, 182, 170, 169, 177, 168, 183, 169, 164])

# Welch's t-test (default: equal_var=False)
t_stat, p_value = stats.ttest_ind(page_a, page_b, equal_var=False)

print(f"Page A: mean = {page_a.mean():.2f}, std = {page_a.std(ddof=1):.2f}")
print(f"Page B: mean = {page_b.mean():.2f}, std = {page_b.std(ddof=1):.2f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value (two-sided): {p_value:.4f}")

# One-sided test: H_a: μ_B > μ_A
p_one_sided = p_value / 2 if page_b.mean() > page_a.mean() else 1 - p_value / 2
print(f"p-value (one-sided): {p_one_sided:.4f}")

# Effect size (Cohen's d)
pooled_std = np.sqrt(((len(page_a) - 1) * page_a.std(ddof=1)**2 +
                       (len(page_b) - 1) * page_b.std(ddof=1)**2) /
                      (len(page_a) + len(page_b) - 2))
cohens_d = (page_b.mean() - page_a.mean()) / pooled_std
print(f"Cohen's d: {cohens_d:.3f}")
```

## Python 구현

### scipy.stats 사용

```python
from scipy import stats

# Welch's t-test (recommended)
t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)

# Pooled t-test
t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=True)

# One-sided tests
if t_stat > 0:
    p_one_sided = p_value / 2  # Upper tail
else:
    p_one_sided = 1 - p_value / 2  # Lower tail
```

### statsmodels 사용

```python
import statsmodels.api as sm

# Welch's t-test with more details
t_stat, p_value, df = sm.stats.ttest_ind(group1, group2,
                                         usevar='unequal',
                                         alternative='two-sided')
```

## 가정

1. **독립성**: 각 집단 안의 관측값이 독립이다
2. **정규성**: 각 집단의 자료가 근사적으로 정규분포를 따른다(n > 30이면 덜 중요하다)
3. **확률표본추출**: 표본이 각 모집단에서 무작위로 뽑혔다

## 어느 검정을 언제 쓸 것인가

| 상황 | 쓸 검정 |
|----------|------------|
| 작은 표본이고 분산이 같아 보임 | 합동 t-검정 |
| 표본크기와 무관, 또는 분산이 불확실 | **Welch t-검정** |
| 정규가 아닌 자료, 작은 표본 | 순열검정 또는 Mann-Whitney U 검정 |
| 큰 표본 (n > 30) | 어느 쪽이든 (둘 다 잘 통한다) |

## 관련 검정

- **대응 t-검정**: 종속인 표본(짝지은 쌍)에 대해
- **Mann-Whitney U 검정**: 정규가 아닌 자료를 위한 비모수적 대안
- **순열검정**: 가정이 없는 재표본추출 접근
- **붓스트랩 신뢰구간**: 분포 가정 없이 신뢰구간을 구할 때

## 연습문제

**연습문제 1.**
프로그램 A: $\bar X_1 = 55\,000, s_1 = 7\,500, n_1 = 14$. 프로그램 B: $\bar X_2 = 60\,000, s_2 = 8\,000, n_2 = 16$. $\alpha = 0.05$에서 합동 $t$-검정을 하라.

??? success "풀이"
    $S_p^2 = (13 \cdot 56\,250\,000 + 15 \cdot 64\,000\,000)/28 \approx 60\,401\,786$.

    $t = -5000/\sqrt{60\,401\,786 \cdot (1/14 + 1/16)} = -5000/\sqrt{8\,089\,524} \approx -1.76$.

    임계값: $t_{0.025, 28} = \pm 2.048$. $|t| = 1.76 < 2.048$. **기각하지 못한다.**

---

**연습문제 2.**
Norway ($\bar X = 64.3, s = 18.2, n = 65$) 대 US ($\bar X = 53.4, s = 23.9, n = 75$)의 소득. Welch 검정을 하라.

??? success "풀이"
    $\mathrm{SE} = \sqrt{18.2^2/65 + 23.9^2/75} = \sqrt{5.10 + 7.62} = \sqrt{12.72} \approx 3.57$.

    $t = (64.3 - 53.4)/3.57 \approx 3.06$. 강하게 유의하다.

    Welch 자유도: $\nu \approx (12.72)^2/[(5.10)^2/64 + (7.62)^2/74] \approx 136$. 자유도가 이만큼 크면 사실상 $z$-검정이다.

    p-값 $\approx 0.003$. $H_0$을 기각한다. Norway의 소득이 유의하게 높다.

---

**연습문제 3.**
US ($\bar X = 25.5, s = 3.8, n = 108$) 대 Canada ($\bar X = 26.3, s = 3.2, n = 102$)의 초혼 연령. 합동 $t$-검정을 하라.

??? success "풀이"
    $S_p^2 = (107 \cdot 14.44 + 101 \cdot 10.24)/208 \approx 12.40$.

    $t = -0.8/\sqrt{12.40 \cdot (1/108 + 1/102)} = -0.8/\sqrt{0.2362} \approx -1.65$.

    임계값: $t_{0.025, 208} \approx \pm 1.97$. $|t| < 1.97$. 기각하지 못한다.

    p-값 $\approx 0.10$. 경계에 있어 5%에서는 유의하지 않지만 가깝다. 표본이 더 크면 실제 차이를 탐지할 수도 있다.

---

**연습문제 4.**
전기차 모델 A ($\bar X = 168, s = 5.4, n = 5$) 대 B ($\bar X = 172, s = 7.5, n = 5$). Welch 검정을 하라.

??? success "풀이"
    $\mathrm{SE} = \sqrt{29.16/5 + 56.25/5} = \sqrt{17.08} \approx 4.13$.

    $t = -4/4.13 \approx -0.97$. Welch 자유도: $\nu \approx (17.08)^2/[5.83^2/4 + 11.25^2/4] \approx 7.3$.

    임계값: $t_{0.025, 7} \approx 2.36$. $|t| < 2.36$. 기각하지 못한다.

    표본이 아주 작아 검정력이 낮다. 차이가 실재하더라도 결론지을 수 없다.

---

**연습문제 5.**
**Welch 대 합동.** 분산이 다를 때 합동 $t$-검정은 언제 실패하는가(잘못된 $\alpha$를 주는가)?

??? success "풀이"
    합동 $t$는 $\sigma_1 = \sigma_2$를 가정한다. 분산이 다르면 합동 표준오차 추정량이 편향되고 검정통계량이 정확한 $t$ 분포를 따르지 않는다.

    **실패 양상:**

    - **$n$도 다르고 $\sigma$도 다를 때:** 제1종 오류가 크게 부풀 수 있다. 작은 표본 쪽의 분산이 크면 $\alpha$가 명목 수준의 2~3배까지 커질 수 있다.
    - **$n$이 같을 때:** 합동 검정은 분산이 달라도 로버스트하다. $\alpha$가 명목값 근처에 머문다.

    **Welch 검정:** 등분산을 가정하지 않는다. 실제로 분산이 같을 때 검정력이 약간 낮을 뿐이다(효율 손실이 작다).

    **현대의 기본값:** Welch (R의 `t.test`, scipy의 `ttest_ind(equal_var=False)`). $\alpha$가 부풀 위험을 피한다.

---

**연습문제 6.**
이표본 $t$-검정의 **효과크기와 표본크기 계획.**

??? success "풀이"
    Cohen의 $d = (\mu_1 - \mu_2)/\sigma_{\text{pooled}}$. 연습문제 1에서: $d = -5000/7772 \approx -0.64$.

    이표본 $t$-검정의 (집단당) 표본크기:

    $$
    n = \frac{2(z_{\alpha/2} + z_\beta)^2}{d^2}
    $$

    $\alpha = 0.05$에서 검정력 80%이면 $n \approx 16/d^2$이다. $d = 0.5$(중간)이면 $n \approx 64$, $d = 0.8$(큼)이면 $n \approx 25$이다.

    작은 효과에는 큰 표본이 든다 — 사회과학과 의학 연구에서 흔한 축척이다. 연구를 시작하기 *전에* 검정력 분석을 하라.
