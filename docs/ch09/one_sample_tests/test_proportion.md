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
    """method='wald'(정규근사) 또는 'exact'(이항).

    정확검정에는 검정통계량이 따로 없다. 이항분포에서 꼬리확률을
    바로 더하므로 표준화할 대상이 없기 때문이다. 그래서 None을 돌려준다.
    """
    phat = k / n
    if method == "exact":
        p = binomtest(k, n, p0, alternative=alt).pvalue
        return None, p, (p < alpha), "exact binomial"

    # 표준오차에 phat이 아니라 **p0**을 넣는다.
    # H0가 참이라는 가정 아래의 확률을 재는 것이 검정이기 때문이다.
    # 신뢰구간에서는 phat을 넣었다. 목적이 다르면 대입하는 값도 다르다.
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

<div class="codebox" markdown>

### 예제 1. 일표본 비율 검정 { .eg }

```python
stat, p, reject, label = test_prop_one_sample(
    k=12, n=50, p0=0.2, method="wald", alt="two-sided"
)
print(label, "stat:", stat, "p:", p, "reject:", reject)

# 같은 자료의 정확한 이항검정
stat_e, p_e, reject_e, label_e = test_prop_one_sample(
    k=12, n=50, p0=0.2, method="exact", alt="two-sided"
)
print(label_e, "stat:", stat_e, "p:", p_e, "reject:", reject_e)
```

출력:

```
wald z-test stat: 0.7071067811865471 p: 0.4795001221869537 reject: False
exact binomial stat: None p: 0.47974220659401984 reject: False
```

두 p-값이 0.4795와 0.4797로 거의 같다. $n p_0 = 10$과 $n(1-p_0) = 40$으로 정규근사의 경험칙을 만족하기 때문이다.

이 일치를 일반적인 것으로 받아들이면 곤란하다. $k = 2$, $n = 10$, $p_0 = 0.5$처럼 표본이 작고 비율이 극단적인 경우로 바꿔 보면 Wald가 0.0578, 정확검정이 0.1094로 두 배 가까이 벌어진다. 5% 기준을 놓고 결론이 갈리는 자리다. 이항분포를 직접 쓸 수 있을 때는 정확검정 쪽이 안전하다.

</div>

### 해석

$n = 50$ 중 성공이 $k = 12$이어서 $\hat{p} = 0.24$인 자료로 $H_0\colon p = 0.2$를 검정한다. 검정통계량은

$$
Z = \frac{0.24 - 0.20}{\sqrt{0.20 \times 0.80 / 50}} = \frac{0.04}{0.0566} \approx 0.707.
$$

양측 p-값이 약 0.48이므로 $H_0$을 기각하지 못한다. 관측된 비율은 $p = 0.20$과 부합한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** 제조품 $n = 200$개의 표본에서 18개가 불량이다. $\alpha = 0.01$에서 $H_0\colon p = 0.05$ 대 $H_1\colon p > 0.05$를 검정하라.

</div>

??? success "풀이"

    $\hat{p} = 18/200 = 0.09$이다. 검정통계량은

    $$
    Z = \frac{0.09 - 0.05}{\sqrt{0.05 \times 0.95 / 200}} = \frac{0.04}{\sqrt{0.0002375}} = \frac{0.04}{0.01541} \approx 2.596.
    $$

    단측 p-값은 $P(Z \geq 2.596) \approx 0.0047$이다. $0.0047 < 0.01$이므로 $H_0$을 기각한다. 불량률이 5%를 넘는다는 유의한 증거가 있다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** 동전을 100번 던져 앞면이 60번 나왔다. $\alpha = 0.05$에서 Wald 검정과 정확한 이항검정으로 이 동전이 공정한지 검정하라. p-값을 비교하라.

</div>

??? success "풀이"

    **Wald 검정:** $\hat{p} = 0.60$, $p_0 = 0.50$.

    $$
    Z = \frac{0.60 - 0.50}{\sqrt{0.50 \times 0.50/100}} = \frac{0.10}{0.05} = 2.0.
    $$

    양측 p-값: $2 \times P(Z \geq 2.0) = 2(0.0228) = 0.0456$. $H_0$을 기각한다.

    **정확한 이항검정:** $p = 2 \times P(X \geq 60 \mid X \sim \text{Bin}(100, 0.5)) \approx 0.0569$. $H_0$을 기각하지 못한다.

    Wald 검정은 기각하고 정확검정은 기각하지 않는데, 이는 정규근사가 약간 관대해질 수 있음을 보여준다. 경계에 있는 경우에는 정확검정이 더 믿을 만하다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** 이항분포에 중심극한정리를 적용하여 Wald 검정통계량을 유도하라.

</div>

??? success "풀이"

    $X_1, \dots, X_n \overset{\text{iid}}{\sim} \text{Bernoulli}(p)$라 하자. 표본비율은 $\hat{p} = \bar{X} = \sum X_i / n$이다. $H_0\colon p = p_0$ 아래에서 $E[\hat{p}] = p_0$이고 $\text{Var}(\hat{p}) = p_0(1-p_0)/n$이다. 중심극한정리에 의해 $n \to \infty$일 때

    $$
    \frac{\hat{p} - p_0}{\sqrt{p_0(1-p_0)/n}} \xrightarrow{d} N(0,1)
    $$

    이다. 이 표준화된 양이 바로 Wald 검정통계량 $Z$이다. $np_0 \geq 5$이고 $n(1-p_0) \geq 5$이면 근사가 정확하다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** 어떤 여론조사가 유권자 $n = 1000$명을 조사하여 540명이 한 후보를 지지함을 발견했다. $p$의 95% 신뢰구간을 구성하고 신뢰구간–검정 쌍대성으로 $H_0\colon p = 0.50$을 검정하라.

</div>

??? success "풀이"

    $p$의 Wald 95% 신뢰구간은

    $$
    \hat{p} \pm z_{0.025}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}} = 0.54 \pm 1.96\sqrt{\frac{0.54 \times 0.46}{1000}} = 0.54 \pm 1.96(0.01577) = 0.54 \pm 0.0309.
    $$

    구간은 $(0.509, 0.571)$이다. $p_0 = 0.50$이 이 구간에 없으므로 $\alpha = 0.05$에서 $H_0\colon p = 0.50$을 기각한다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** $H_0\colon p = p_0$ 대 $H_1\colon p > p_0$에 대한 수준 $\alpha$의 정확한 이항검정이 $k \geq c$일 때 기각함을 보여라. 여기서 $c$는 $P(X \geq c \mid X \sim \text{Bin}(n, p_0)) \leq \alpha$를 만족하는 가장 작은 정수이다.

</div>

??? success "풀이"

    단측검정의 p-값은

    $$
    p\text{-value} = P(X \geq k \mid p = p_0) = \sum_{j=k}^{n} \binom{n}{j} p_0^j (1-p_0)^{n-j}.
    $$

    이 p-값이 $\alpha$ 이하일 때 $H_0$을 기각한다. 임계값 $c$는 다음을 만족하는 가장 작은 정수이다:

    $$
    P(X \geq c \mid p = p_0) = \sum_{j=c}^{n} \binom{n}{j} p_0^j (1-p_0)^{n-j} \leq \alpha.
    $$

    $P(X \geq k)$가 $k$의 감소하는 계단함수이므로, 관측값 $k \geq c$이면 p-값이 $\alpha$ 이하이고 $k < c$이면 $\alpha$를 넘는다. 이 검정은 이산적이므로 달성되는 유의수준이 $\alpha$보다 엄격히 작을 수 있다. $\square$

---

## 정리하며

비율 검정에는 **근사와 정확, 두 갈래**가 있다.

- **왈드 $z$ 검정**은 $np_0\ge5$, $n(1-p_0)\ge5$ 일 때 쓴다. 계산이 간단하고 대표본에서 잘 맞는다.
- **정확 이항검정**은 근사를 쓰지 않고 이항분포에서 직접 확률을 더한다. $n$ 이 작거나 $p_0$ 이 극단적일 때 필요하며, `scipy.stats.binomtest` 가 제공한다.
- **정확 검정은 보수적이다.** 이산성 때문에 실제 제1종 오류율이 명목 $\alpha$ 보다 낮게 나오며, 그 대가로 검정력이 조금 떨어진다. **"정확"이 "최적"은 아니다.**
- **양측 정확 검정의 정의가 하나가 아니다.** 확률이 작은 쪽을 더하는 방식과 대칭으로 두 배 하는 방식이 있어 답이 미세하게 다를 수 있다. 라이브러리의 기본값을 확인해야 한다.
- **검정의 표준오차는 $p_0$ 로, 구간의 표준오차는 $\hat p$ 로** 만든다는 차이를 다시 확인하게 된다.

다음 절 **일표본 분산 검정**으로 넘어간다.
