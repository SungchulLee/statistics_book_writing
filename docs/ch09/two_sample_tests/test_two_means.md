# 이표본 평균 검정

## 개요

이표본 t-검정은 독립인 두 모집단의 평균을 비교한다. **합동 t-검정**은 등분산을 가정하고 두 표본분산을 하나의 추정값으로 합치며, **Welch t-검정**은 자유도를 조정하여 분산이 다른 경우를 허용한다. Welch 검정은 분산이 같을 때에도 잘 작동하므로 일반적으로 기본으로 권장된다.

## 검정의 구성

**가설:**

- 양측: $H_0\colon \mu_1 - \mu_2 = \delta_0$ 대 $H_1\colon \mu_1 - \mu_2 \neq \delta_0$
- 단측: $H_0\colon \mu_1 - \mu_2 = \delta_0$ 대 $H_1\colon \mu_1 - \mu_2 > \delta_0$

### Welch t-검정

$$
T = \frac{(\bar{X}_1 - \bar{X}_2) - \delta_0}{\sqrt{S_1^2/n_1 + S_2^2/n_2}}
$$

Welch–Satterthwaite 자유도:

$$
\nu = \frac{\left(\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}\right)^2}{\frac{(S_1^2/n_1)^2}{n_1-1} + \frac{(S_2^2/n_2)^2}{n_2-1}}.
$$

### 합동 t-검정

$\sigma_1^2 = \sigma_2^2 = \sigma^2$일 때 분산을 합친다:

$$
S_p^2 = \frac{(n_1-1)S_1^2 + (n_2-1)S_2^2}{n_1+n_2-2}, \qquad T = \frac{(\bar{X}_1 - \bar{X}_2) - \delta_0}{S_p\sqrt{1/n_1 + 1/n_2}} \sim t_{n_1+n_2-2}.
$$

## 코드

```python
import math
from scipy.stats import t as tdist

def test_diff_two_means(n1, m1, s1, n2, m2, s2, method="welch",
                        delta0=0.0, alt="two-sided", alpha=0.05):
    """H0: mu1 - mu2 = delta0.

    delta0을 0이 아닌 값으로 둘 수 있게 해 두었다.
    "차이가 있는가"가 아니라 "차이가 5 이상인가"를 묻는 동등성·비열등성
    검정에서 이 자리가 쓰인다.
    method='welch'(기본) 또는 'pooled'. 돌려주는 값은 (t, df, p, 기각 여부).
    """
    diff_hat = m1 - m2
    if method == "welch":
        # 두 분산을 따로 둔 채 더한다. 합동하지 않는다.
        se = math.sqrt(s1**2 / n1 + s2**2 / n2)
        num = (s1**2 / n1 + s2**2 / n2) ** 2
        den = (s1**2 / n1)**2 / (n1 - 1) + (s2**2 / n2)**2 / (n2 - 1)
        df = num / den
    else:
        # 합동: 두 분산이 같다고 보고 자유도로 가중평균한다.
        # 이 가정이 틀리면 표준오차가 편향되고 t가 t분포를 따르지 않는다.
        df = n1 + n2 - 2
        sp2 = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / df
        se = math.sqrt(sp2 * (1 / n1 + 1 / n2))

    t = (diff_hat - delta0) / se
    if alt == "two-sided":
        p = 2 * min(tdist.cdf(t, df), 1 - tdist.cdf(t, df))
    elif alt == "less":
        p = tdist.cdf(t, df)
    else:
        p = 1 - tdist.cdf(t, df)
    return t, df, p, (p < alpha)
```

<div class="codebox" markdown>

### 예제 1. 이표본 평균 검정 { .eg }

```python
t, df, p, reject = test_diff_two_means(
    n1=12, m1=0.0, s1=1.0, n2=10, m2=0.5, s2=1.5,
    method="welch", alt="greater"
)
print("t:", t, "df:", df, "p:", p, "reject:", reject)

# 같은 자료를 합동 t로도 해 본다. 자유도가 어떻게 달라지는지 보라.
t_p, df_p, p_p, reject_p = test_diff_two_means(
    n1=12, m1=0.0, s1=1.0, n2=10, m2=0.5, s2=1.5,
    method="pooled", alt="greater"
)
print("t:", t_p, "df:", df_p, "p:", p_p, "reject:", reject_p)
```

출력:

```
t: -0.9004503377814964 df: 15.195761856710394 p: 0.8090351110315042 reject: False
t: -0.9341987329938274 df: 20 p: 0.8193278973550725 reject: False
```

Welch 자유도가 15.20으로 합동의 20보다 작다. 분산이 1.0과 1.5로 다르고 표본크기도 12와 10으로 달라서 생기는 차이다. 정보량을 더 보수적으로 잡는 쪽이 Welch다.

p-값이 0.81로 1에 가깝다는 점도 읽어 두라. 자료가 대립가설과 **반대** 방향이기 때문이다. 단측검정에서 이런 p-값이 나오면 "증거가 약하다"가 아니라 "방향이 반대다"라는 뜻이다.

</div>

### 해석

$\bar{x}_1 = 0.0$, $s_1 = 1.0$, $n_1 = 12$와 $\bar{x}_2 = 0.5$, $s_2 = 1.5$, $n_2 = 10$으로 $H_0\colon \mu_1 - \mu_2 = 0$ 대 $H_1\colon \mu_1 - \mu_2 > 0$을 검정한다. $\bar{x}_1 - \bar{x}_2 = -0.5 < 0$이므로 검정통계량이 음수이다. 우측검정에서는 p-값이 1에 가까워지므로 $H_0$을 기각하지 못한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> 집단 A ($n_1=15$, $\bar{x}_1=82$, $s_1=6$)와 집단 B ($n_2=12$, $\bar{x}_2=76$, $s_2=8$). $\alpha=0.05$에서 $H_0\colon \mu_1 = \mu_2$의 Welch t-검정을 수행하라.

</div>

??? success "풀이"

    표준오차는

    $$
    SE = \sqrt{\frac{6^2}{15} + \frac{8^2}{12}} = \sqrt{2.4 + 5.333} = \sqrt{7.733} \approx 2.781.
    $$

    검정통계량은

    $$
    T = \frac{82 - 76}{2.781} = \frac{6}{2.781} \approx 2.157.
    $$

    Welch–Satterthwaite 자유도:

    $$
    \nu = \frac{7.733^2}{\frac{2.4^2}{14} + \frac{5.333^2}{11}} = \frac{59.80}{0.411 + 2.587} = \frac{59.80}{2.998} \approx 19.95.
    $$

    $\nu \approx 20$에서 양측 p-값은 약 0.044이다. $0.044 < 0.05$이므로 $H_0$을 기각한다. 두 평균이 유의하게 다르다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> $n_1 = n_2 = n$이고 $s_1 = s_2 = s$이면 합동 t-검정과 Welch t-검정의 검정통계량과 자유도가 같음을 보여라.

</div>

??? success "풀이"

    **합동:** $S_p^2 = \frac{(n-1)s^2 + (n-1)s^2}{2n-2} = s^2$. 표준오차는 $s\sqrt{2/n}$이고 $\text{df} = 2n-2$이다.

    **Welch:** $SE = \sqrt{s^2/n + s^2/n} = s\sqrt{2/n}$. 검정통계량이 같다. 자유도는:

    $$
    \nu = \frac{(s^2/n + s^2/n)^2}{\frac{(s^2/n)^2}{n-1} + \frac{(s^2/n)^2}{n-1}} = \frac{(2s^2/n)^2}{2(s^2/n)^2/(n-1)} = \frac{4s^4/n^2}{2s^4/(n^2(n-1))} = 2(n-1).
    $$

    이는 합동 자유도 $2n-2$와 같다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> 어떤 조건에서 Welch 검정보다 합동 t-검정을 선호하는가? 언제 잘못된 결론으로 이어질 수 있는가?

</div>

??? success "풀이"

    (F-검정이나 분야 지식 등으로) $\sigma_1^2 = \sigma_2^2$이라는 강한 사전 근거가 있을 때 합동 t-검정을 선호한다. 등분산 아래에서 합동 검정은 자유도가 조금 더 많아($n_1+n_2-2$ 대 $\nu_{\text{Welch}}$) 검정력이 미세하게 높다.

    그러나 분산이 다르면 표본크기와 분산의 관계에 따라 합동 검정의 제1종 오류율이 부풀거나(관대해지거나) 줄어들 수 있다(보수적이 될 수 있다). 구체적으로 표본이 작은 집단의 분산이 크면 합동 검정이 너무 자주 기각한다. Welch 검정은 분산이 달라도 로버스트하고 등분산일 때 검정력 손실도 미미하므로 더 안전한 기본값이다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span> 두 기계가 볼트를 생산한다. 기계 1: $n_1=20$, $\bar{x}_1=10.02$ mm, $s_1=0.05$. 기계 2: $n_2=25$, $\bar{x}_2=10.00$ mm, $s_2=0.04$. $\alpha = 0.01$에서 합동 t-검정으로 $H_0\colon \mu_1 - \mu_2 = 0$ 대 $H_1\colon \mu_1 - \mu_2 \neq 0$을 검정하라.

</div>

??? success "풀이"

    합동분산은

    $$
    S_p^2 = \frac{19(0.05)^2 + 24(0.04)^2}{43} = \frac{19(0.0025) + 24(0.0016)}{43} = \frac{0.0475 + 0.0384}{43} = \frac{0.0859}{43} \approx 0.001998.
    $$

    표준오차는

    $$
    SE = \sqrt{0.001998 \times (1/20 + 1/25)} = \sqrt{0.001998 \times 0.09} = \sqrt{0.0001798} \approx 0.01341.
    $$

    검정통계량은

    $$
    T = \frac{10.02 - 10.00}{0.01341} = \frac{0.02}{0.01341} \approx 1.491.
    $$

    $\text{df} = 43$에서 양측 p-값은 약 0.143이다. $0.143 > 0.01$이므로 $H_0$을 기각하지 못한다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> $\sigma_1^2 = \sigma_2^2 = \sigma^2$이라는 가정 아래에서 합동 분산추정량 $S_p^2$을 (편향 보정을 제외하면) $\sigma^2$의 최대가능도추정량으로 유도하라.

</div>

??? success "풀이"

    등분산 모형에서 $j=1,\dots,n_1$에 대해 $X_{1j} \overset{\text{iid}}{\sim} N(\mu_1, \sigma^2)$, $j=1,\dots,n_2$에 대해 $X_{2j} \overset{\text{iid}}{\sim} N(\mu_2, \sigma^2)$이고 두 집단은 독립이다. 로그가능도는

    $$
    \ell(\mu_1, \mu_2, \sigma^2) = -\frac{n_1+n_2}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\left[\sum_{j=1}^{n_1}(X_{1j}-\mu_1)^2 + \sum_{j=1}^{n_2}(X_{2j}-\mu_2)^2\right].
    $$

    $\mu_1, \mu_2$에 대해 최대화하면 $\hat{\mu}_i = \bar{X}_i$이다. 이를 대입하고 $\sigma^2$에 대해 최대화하면:

    $$
    \hat{\sigma}^2_{\text{MLE}} = \frac{\sum(X_{1j}-\bar{X}_1)^2 + \sum(X_{2j}-\bar{X}_2)^2}{n_1+n_2}.
    $$

    편향 보정을 적용하면($n_1+n_2$를 $n_1+n_2-2$로 바꾸면) 불편추정량을 얻는다:

    $$
    S_p^2 = \frac{(n_1-1)S_1^2 + (n_2-1)S_2^2}{n_1+n_2-2}. \quad \square
    $$

---

## 정리하며

합동과 웰치를 **나란히 구현**해 비교했다.

- **차이는 분모와 자유도 두 곳뿐이다.** 합동은 $s_p\sqrt{1/n_1+1/n_2}$ 에 자유도 $n_1+n_2-2$, 웰치는 $\sqrt{s_1^2/n_1+s_2^2/n_2}$ 에 새터스웨이트 자유도다.
- **웰치의 자유도는 정수가 아니다.** $t$ 분포는 실수 자유도를 받으므로 문제없으며, 반올림하면 결과가 미세하게 달라진다.
- **`scipy.stats.ttest_ind(..., equal_var=False)` 가 웰치**이고, 기본값 `True` 가 합동이다. **기본값이 합동이라는 점을 기억해야 한다.**
- **$\delta_0\ne0$ 을 검정하려면 직접 구현해야 한다.** `scipy` 는 $\delta_0=0$ 만 다루므로, 동등성 검정이나 비열등성 검정에서는 통계량을 손으로 만든다.
- **결론은 웰치를 기본으로.** 등분산일 때 손해가 미미하고 아닐 때 이득이 크다.

다음 절 **이표본 비율 검정**으로 넘어간다.
