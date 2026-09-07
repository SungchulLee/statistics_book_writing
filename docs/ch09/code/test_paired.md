# 대응 평균 검정

## 개요

대응 t-검정은 같은 피험자에게서 얻은 관련된 두 측정값(예: 처리 전후)을 비교한다. 각 쌍에 대해 차이 $D_i = X_i - Y_i$를 계산하면 문제가 그 차이에 대한 일표본 t-검정으로 환원된다. 이 접근은 피험자 간 변동성을 제거하므로, 짝짓기가 의미 있을 때 독립 이표본 검정보다 통계적 검정력이 크다.

## 검정의 구성

**가설:** $\mu_D = E[D_i]$를 대응 차이의 모평균이라 하자.

- 양측: $H_0\colon \mu_D = \mu_{D_0}$ 대 $H_1\colon \mu_D \neq \mu_{D_0}$
- 단측: $H_0\colon \mu_D = \mu_{D_0}$ 대 $H_1\colon \mu_D > \mu_{D_0}$ (또는 $< \mu_{D_0}$)

보통 $\mu_{D_0} = 0$(차이 없음)이다.

**검정통계량:** 쌍이 $n$개이고 평균 차이가 $\bar{D}$, 차이의 표준편차가 $S_D$일 때,

$$
T = \frac{\bar{D} - \mu_{D_0}}{S_D / \sqrt{n}} \sim t_{n-1}.
$$

## 코드

```python
import math
from scipy.stats import t as tdist

def test_paired_mean(n, dbar, sd_d, mu_d0=0.0,
                     alt="two-sided", alpha=0.05):
    """
    Paired t-test: differences D = X - Y.
    Supply n, mean(D), sd(D).
    Returns (t, p, reject).
    """
    df = n - 1
    se = sd_d / math.sqrt(n)
    t = (dbar - mu_d0) / se
    if alt == "two-sided":
        p = 2 * min(tdist.cdf(t, df), 1 - tdist.cdf(t, df))
    elif alt == "less":
        p = tdist.cdf(t, df)
    else:
        p = 1 - tdist.cdf(t, df)
    return t, p, (p < alpha)
```

### 예제

```python
t_stat, p, reject = test_paired_mean(
    n=12, dbar=0.4, sd_d=1.1, mu_d0=0.0, alt="less"
)
print("t:", t_stat, "p:", p, "reject:", reject)
```

### 해석

$n=12$쌍, $\bar{D}=0.4$, $S_D=1.1$일 때 $H_1\colon \mu_D < 0$에 대한 검정통계량은

$$
T = \frac{0.4 - 0}{1.1/\sqrt{12}} = \frac{0.4}{0.3175} \approx 1.260.
$$

$T > 0$인데 왼쪽 꼬리를 검정하므로($H_1\colon \mu_D < 0$) p-값이 크고 $H_0$을 기각하지 못한다.

## 연습문제

**연습문제 1.** 환자 10명의 혈압을 새 약 투여 전후로 측정했다. 차이(투여 후 $-$ 투여 전)는 $-7, -8, -8, -7, -7, -8, -4, -7, -6, -7$이다. $\alpha = 0.05$에서 이 약이 혈압을 유의하게 낮추는지 검정하라.

??? success "풀이"

    계산하면 $\bar{D} = -6.9$, $S_D \approx 1.197$, $n = 10$이다.

    $H_0\colon \mu_D = 0$ 대 $H_1\colon \mu_D < 0$을 검정한다:

    $$
    T = \frac{-6.9 - 0}{1.197/\sqrt{10}} = \frac{-6.9}{0.3785} \approx -18.23.
    $$

    $\text{df} = 9$에서 $P(T_9 \leq -18.23) \approx 0$이다. $H_0$을 강하게 기각한다. 이 약이 혈압을 유의하게 낮춘다. $\square$

---

**연습문제 2.** 짝짓기가 적절할 때 대응 t-검정이 독립 이표본 t-검정보다 강력한 이유를 설명하라.

??? success "풀이"

    독립 이표본 t-검정에서 평균 차이의 분산은

    $$
    \text{Var}(\bar{X} - \bar{Y}) = \frac{\sigma_X^2}{n} + \frac{\sigma_Y^2}{n}.
    $$

    대응 검정에서 $\bar{D}$의 분산은

    $$
    \text{Var}(\bar{D}) = \frac{\sigma_D^2}{n} = \frac{\sigma_X^2 + \sigma_Y^2 - 2\rho\,\sigma_X\sigma_Y}{n},
    $$

    여기서 $\rho = \text{Corr}(X_i, Y_i)$이다. 짝짓기가 양의 상관을 만들면($\rho > 0$) $\bar{D}$의 분산이 작아져 검정통계량이 커지고 검정력이 높아진다. 감소량은 $2\rho\sigma_X\sigma_Y/n$에 비례한다. $\square$

---

**연습문제 3.** 대응 t-검정이 차이에 대한 일표본 t-검정과 대수적으로 동등함을 보여라.

??? success "풀이"

    $i = 1, \dots, n$에 대해 $D_i = X_i - Y_i$로 정의한다. 그러면 $\bar{D} = \bar{X} - \bar{Y}$이고

    $$
    S_D^2 = \frac{1}{n-1}\sum_{i=1}^n (D_i - \bar{D})^2.
    $$

    대응 t-통계량은

    $$
    T_{\text{paired}} = \frac{\bar{D} - 0}{S_D/\sqrt{n}}.
    $$

    표본 $D_1, \dots, D_n$에 귀무값 $\mu_0 = 0$으로 일표본 t-검정을 적용하면

    $$
    T_{\text{one-sample}} = \frac{\bar{D} - 0}{S_D/\sqrt{n}} = T_{\text{paired}}.
    $$

    검정통계량과 자유도($n-1$)가 모두 같으므로 두 검정은 동일하다. $\square$

---

**연습문제 4.** 어떤 연구가 피험자 8명의 반응시간(ms)을 카페인 조건과 위약 조건에서 측정했다. 대응 차이(카페인 $-$ 위약)는 $-15, -22, -8, -30, -12, -18, -25, -10$이다. 평균 차이의 99% 신뢰구간을 계산하라.

??? success "풀이"

    계산하면 $\bar{D} = (-15-22-8-30-12-18-25-10)/8 = -140/8 = -17.5$이다.

    $$
    S_D = \sqrt{\frac{\sum(D_i - \bar{D})^2}{7}} = \sqrt{\frac{(2.5)^2+(-4.5)^2+(9.5)^2+(-12.5)^2+(5.5)^2+(-0.5)^2+(-7.5)^2+(7.5)^2}{7}}
    $$

    $$
    = \sqrt{\frac{6.25+20.25+90.25+156.25+30.25+0.25+56.25+56.25}{7}} = \sqrt{\frac{416}{7}} \approx 7.71.
    $$

    99% 신뢰구간은 $\bar{D} \pm t_{0.005,7} \cdot S_D/\sqrt{8}$이다. $t_{0.005,7} = 3.499$이므로:

    $$
    -17.5 \pm 3.499 \times \frac{7.71}{\sqrt{8}} = -17.5 \pm 3.499 \times 2.727 = -17.5 \pm 9.54.
    $$

    99% 신뢰구간은 $(-27.04, -7.96)$이다. 0이 이 구간에 없으므로 $\alpha = 0.01$에서 $H_0\colon \mu_D = 0$을 기각한다. $\square$

---

**연습문제 5.** 대응 t-검정이 부적절한 조건은 무엇인가? 차이가 정규가 아닐 때의 대안을 제안하라.

??? success "풀이"

    대응 t-검정의 가정:

    1. 차이 $D_i$가 독립이다(서로 다른 피험자).
    2. 차이가 근사적으로 정규분포를 따른다(또는 $n$이 중심극한정리를 쓸 만큼 크다).

    다음의 경우 부적절하다:

    - 표본이 아주 작고 차이가 분명히 정규가 아닐 때(두꺼운 꼬리, 강한 치우침).
    - 쌍이 자연스럽게 짝지어져 있지 않아 상관 구조가 무의미할 때.

    비모수적 대안은 차이의 중앙값이 0인지 검정하는 **Wilcoxon 부호순위 검정**이다. 절대차이의 순위를 매기고 부호를 부여한 뒤 부호 있는 순위의 합을 검정통계량으로 쓴다. 차이의 분포가 중앙값을 중심으로 대칭이라는 더 약한 가정 아래에서 타당하다. $\square$
