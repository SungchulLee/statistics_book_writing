# 일표본 비율 검정

## 개요

일표본 비율 검정은 모비율 $p$가 가설의 값 $p_0$과 같은지 평가한다. **Wald z-검정**은 이항분포에 대한 정규근사를 쓰며 $np_0$과 $n(1-p_0)$이 모두 5 이상일 때 타당하다. 표본이 작으면 점근근사에 기대지 않는 **정확한 이항검정**이 대안이 된다.

## 검정의 구성

**가설:**

- 양측: $H_0\colon p = p_0$ 대 $H_1\colon p \neq p_0$
- 단측: $H_0\colon p = p_0$ 대 $H_1\colon p > p_0$ (또는 $H_1\colon p < p_0$)

**Wald z-검정:** $n$번 시행에서 성공이 $k$번이고 $\hat{p} = k/n$일 때 검정통계량은

$$
Z = \frac{\hat{p} - p_0}{\sqrt{p_0(1-p_0)/n}} \;\dot\sim\; N(0,1).
$$

귀무가설 아래에서 계산하므로 표준오차에 $\hat{p}$가 아니라 $p_0$을 쓴다는 점에 유의하라.

**정확한 이항검정:** 정규근사 없이 이항분포 $\text{Bin}(n, p_0)$에서 p-값을 직접 계산한다.

## 코드

```python
from scipy.stats import norm, binomtest
import math

def test_prop_one_sample(k, n, p0=0.5, method="wald",
                         alt="two-sided", alpha=0.05):
    """
    method='wald' (normal approx) or 'exact' (binomial).
    Returns (stat_or_None, pvalue, reject_bool, label).
    """
    phat = k / n
    if method == "exact":
        p = binomtest(k, n, p0, alternative=alt).pvalue
        return None, p, (p < alpha), "exact binomial"

    se0 = math.sqrt(p0 * (1 - p0) / n)
    z = (phat - p0) / se0
    if alt == "two-sided":
        p = 2 * min(norm.cdf(z), 1 - norm.cdf(z))
    elif alt == "less":
        p = norm.cdf(z)
    else:
        p = 1 - norm.cdf(z)
    return z, p, (p < alpha), "wald z-test"
```

### 예제

```python
stat, p, reject, label = test_prop_one_sample(
    k=12, n=50, p0=0.2, method="wald", alt="two-sided"
)
print(label, "stat:", stat, "p:", p, "reject:", reject)
```

### 해석

$n = 50$ 중 성공이 $k = 12$이어서 $\hat{p} = 0.24$인 자료로 $H_0\colon p = 0.2$를 검정한다. 검정통계량은

$$
Z = \frac{0.24 - 0.20}{\sqrt{0.20 \times 0.80 / 50}} = \frac{0.04}{0.0566} \approx 0.707.
$$

양측 p-값이 약 0.48이므로 $H_0$을 기각하지 못한다. 관측된 비율은 $p = 0.20$과 부합한다.

## 연습문제

**연습문제 1.** 제조품 $n = 200$개의 표본에서 18개가 불량이다. $\alpha = 0.01$에서 $H_0\colon p = 0.05$ 대 $H_1\colon p > 0.05$를 검정하라.

??? success "연습문제 1 풀이"

    $\hat{p} = 18/200 = 0.09$이다. 검정통계량은

    $$
    Z = \frac{0.09 - 0.05}{\sqrt{0.05 \times 0.95 / 200}} = \frac{0.04}{\sqrt{0.0002375}} = \frac{0.04}{0.01541} \approx 2.596.
    $$

    단측 p-값은 $P(Z \geq 2.596) \approx 0.0047$이다. $0.0047 < 0.01$이므로 $H_0$을 기각한다. 불량률이 5%를 넘는다는 유의한 증거가 있다. $\square$

---

**연습문제 2.** 동전을 100번 던져 앞면이 60번 나왔다. $\alpha = 0.05$에서 Wald 검정과 정확한 이항검정으로 이 동전이 공정한지 검정하라. p-값을 비교하라.

??? success "연습문제 2 풀이"

    **Wald 검정:** $\hat{p} = 0.60$, $p_0 = 0.50$.

    $$
    Z = \frac{0.60 - 0.50}{\sqrt{0.50 \times 0.50/100}} = \frac{0.10}{0.05} = 2.0.
    $$

    양측 p-값: $2 \times P(Z \geq 2.0) = 2(0.0228) = 0.0456$. $H_0$을 기각한다.

    **정확한 이항검정:** $p = 2 \times P(X \geq 60 \mid X \sim \text{Bin}(100, 0.5)) \approx 0.0569$. $H_0$을 기각하지 못한다.

    Wald 검정은 기각하고 정확검정은 기각하지 않는데, 이는 정규근사가 약간 관대해질 수 있음을 보여준다. 경계에 있는 경우에는 정확검정이 더 믿을 만하다. $\square$

---

**연습문제 3.** 이항분포에 중심극한정리를 적용하여 Wald 검정통계량을 유도하라.

??? success "연습문제 3 풀이"

    $X_1, \dots, X_n \overset{\text{iid}}{\sim} \text{Bernoulli}(p)$라 하자. 표본비율은 $\hat{p} = \bar{X} = \sum X_i / n$이다. $H_0\colon p = p_0$ 아래에서 $E[\hat{p}] = p_0$이고 $\text{Var}(\hat{p}) = p_0(1-p_0)/n$이다. 중심극한정리에 의해 $n \to \infty$일 때

    $$
    \frac{\hat{p} - p_0}{\sqrt{p_0(1-p_0)/n}} \xrightarrow{d} N(0,1)
    $$

    이다. 이 표준화된 양이 바로 Wald 검정통계량 $Z$이다. $np_0 \geq 5$이고 $n(1-p_0) \geq 5$이면 근사가 정확하다. $\square$

---

**연습문제 4.** 어떤 여론조사가 유권자 $n = 1000$명을 조사하여 540명이 한 후보를 지지함을 발견했다. $p$의 95% 신뢰구간을 구성하고 신뢰구간–검정 쌍대성으로 $H_0\colon p = 0.50$을 검정하라.

??? success "연습문제 4 풀이"

    $p$의 Wald 95% 신뢰구간은

    $$
    \hat{p} \pm z_{0.025}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}} = 0.54 \pm 1.96\sqrt{\frac{0.54 \times 0.46}{1000}} = 0.54 \pm 1.96(0.01577) = 0.54 \pm 0.0309.
    $$

    구간은 $(0.509, 0.571)$이다. $p_0 = 0.50$이 이 구간에 없으므로 $\alpha = 0.05$에서 $H_0\colon p = 0.50$을 기각한다. $\square$

---

**연습문제 5.** $H_0\colon p = p_0$ 대 $H_1\colon p > p_0$에 대한 수준 $\alpha$의 정확한 이항검정이 $k \geq c$일 때 기각함을 보여라. 여기서 $c$는 $P(X \geq c \mid X \sim \text{Bin}(n, p_0)) \leq \alpha$를 만족하는 가장 작은 정수이다.

??? success "연습문제 5 풀이"

    단측검정의 p-값은

    $$
    p\text{-value} = P(X \geq k \mid p = p_0) = \sum_{j=k}^{n} \binom{n}{j} p_0^j (1-p_0)^{n-j}.
    $$

    이 p-값이 $\alpha$ 이하일 때 $H_0$을 기각한다. 임계값 $c$는 다음을 만족하는 가장 작은 정수이다:

    $$
    P(X \geq c \mid p = p_0) = \sum_{j=c}^{n} \binom{n}{j} p_0^j (1-p_0)^{n-j} \leq \alpha.
    $$

    $P(X \geq k)$가 $k$의 감소하는 계단함수이므로, 관측값 $k \geq c$이면 p-값이 $\alpha$ 이하이고 $k < c$이면 $\alpha$를 넘는다. 이 검정은 이산적이므로 달성되는 유의수준이 $\alpha$보다 엄격히 작을 수 있다. $\square$
