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

## 보기

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> 일표본 평균. $H_0: \mu = \mu_0$ 대 $H_a: \mu \neq \mu_0$의 일표본 z-검정에서:

- **검정**: $z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}$일 때 $|z| > z_{\alpha/2}$이면 $H_0$을 기각한다.
- **신뢰구간**: $\bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$

</div>

??? success "풀이"
    이 검정이 $H_0$을 기각할 필요충분조건은 $\mu_0$이 신뢰구간 밖에 있는 것이다.

**동등성의 증명:**

$$|z| > z_{\alpha/2} \iff \left|\frac{\bar{x} - \mu_0}{\sigma/\sqrt{n}}\right| > z_{\alpha/2} \iff \mu_0 \notin \left(\bar{x} - z_{\alpha/2}\frac{\sigma}{\sqrt{n}},\ \bar{x} + z_{\alpha/2}\frac{\sigma}{\sqrt{n}}\right)$$

<div class="exbox" markdown>

**보기 2.** <span class="diff easy" title="쉬움"></span> 두 품종의 배. Yuna가 Bosc 배와 Anjou 배의 열량을 비교한다. $\mu_{\text{Bosc}} - \mu_{\text{Anjou}}$의 99% 신뢰구간은 $4 \pm 6.44 = (-2.44, 10.44)$이다.

$\alpha = 0.01$에서 $H_0: \mu_{\text{Bosc}} = \mu_{\text{Anjou}}$(즉 $\mu_{\text{Bosc}} - \mu_{\text{Anjou}} = 0$)를 검정하면:

</div>

??? success "풀이"
    $0 \in (-2.44, 10.44)$이므로 $H_0$을 **기각하지 못한다**. 열량이 다르다고 결론지을 증거가 부족하다.

<div class="exbox" markdown>

**보기 3.** <span class="diff easy" title="쉬움"></span> 대면 수업과 온라인 수업. $p_{\text{in\_person}} - p_{\text{online}}$의 95% 신뢰구간이 $(-0.04, 0.14)$이다.

$\alpha = 0.05$에서 $H_0: p_{\text{in\_person}} = p_{\text{online}}$을 검정하면:

</div>

??? success "풀이"
    $0 \in (-0.04, 0.14)$이므로 $H_0$을 **기각하지 못한다**. 합격률에 유의한 차이가 없다.

## 단측검정과 신뢰구간

쌍대성은 단측 신뢰구간(신뢰한계)을 쓰면 단측검정으로도 확장된다:

- **상한 신뢰한계**: 신뢰수준 $1 - \alpha$에서 $\theta < U$는 검정 $H_0: \theta \geq \theta_0$ 대 $H_a: \theta < \theta_0$에 대응한다.
- **하한 신뢰한계**: 신뢰수준 $1 - \alpha$에서 $\theta > L$은 검정 $H_0: \theta \leq \theta_0$ 대 $H_a: \theta > \theta_0$에 대응한다.

<div class="codebox" markdown>

### 예제 1. 신뢰구간과 검정이 같은 답을 준다 { .eg }

```python
import numpy as np
from scipy import stats

x_bar = 52
mu_0 = 50
sigma = 10
n = 25
alpha = 0.05

# 검정: mu_0를 중심에 놓고 x_bar가 얼마나 떨어져 있는지 잰다.
z = (x_bar - mu_0) / (sigma / np.sqrt(n))
p_value = 2 * stats.norm.sf(abs(z))
reject_test = p_value <= alpha

# 신뢰구간: x_bar를 중심에 놓고 mu_0가 안에 들어오는지 본다.
# 기준점만 바꿔 같은 부등식을 두 번 쓰는 셈이라 결론이 어긋날 수 없다.
z_crit = stats.norm.ppf(1 - alpha / 2)
ci_lower = x_bar - z_crit * sigma / np.sqrt(n)
ci_upper = x_bar + z_crit * sigma / np.sqrt(n)
reject_ci = mu_0 < ci_lower or mu_0 > ci_upper

print(f"Test: z = {z:.4f}, p-value = {p_value:.4f}, Reject = {reject_test}")
print(f"CI: ({ci_lower:.4f}, {ci_upper:.4f}), mu_0 outside CI = {reject_ci}")
print(f"Both methods agree: {reject_test == reject_ci}")
```

출력:

```
Test: z = 1.0000, p-value = 0.3173, Reject = False
CI: (48.0801, 55.9199), mu_0 outside CI = False
Both methods agree: True
```

두 접근이 같은 결론에 이른다. $z = 1$은 임계값 1.96에 못 미치고, 같은 이유로 $\mu_0 = 50$이 구간 $(48.08, 55.92)$ 안에 있다.

여기서 구간이 검정보다 하나 더 말해 준다는 점을 짚어 둘 만하다. 검정은 "50을 배제할 수 없다"까지만 말하지만, 구간은 48.08에서 55.92까지가 모두 배제되지 않는다고 말한다. 기각하지 못했다는 결과를 "차이가 없다"로 읽으면 안 되는 이유가 이것이다. 자료는 $\mu = 55$ 역시 배제하지 못한다.

</div>

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
$\mu$의 95% 신뢰구간이 $(12.3, 18.7)$이다. 검정통계량을 계산하지 않고 $\alpha = 0.05$에서 $H_0: \mu = 10$의 양측검정 결과를 판단하라.

</div>

??? success "풀이"
    $\mu_0 = 10$이 95% 신뢰구간 $(12.3, 18.7)$ **밖에** 있으므로 $\alpha = 0.05$에서 $H_0: \mu = 10$을 **기각한다**. 신뢰구간과 가설검정의 쌍대성에 의해 $(1-\alpha)$ 신뢰구간 밖의 어떤 값도 유의수준 $\alpha$에서 기각된다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
어떤 연구자가 $H_0: \mu = 50$의 양측검정을 수행하여 p-값 0.03을 얻었다. 50이 95%와 99% 신뢰구간의 안에 있는지 밖에 있는지 무엇을 결론지을 수 있는가?

</div>

??? success "풀이"
    $p = 0.03 < 0.05$이므로 이 검정은 $\alpha = 0.05$에서 $H_0$을 기각한다. 쌍대성에 의해 $\mu_0 = 50$은 95% 신뢰구간 **밖에** 있다.

    $p = 0.03 > 0.01$이므로 $\alpha = 0.01$에서는 $H_0$을 기각하지 않는다. 쌍대성에 의해 $\mu_0 = 50$은 99% 신뢰구간 **안에** 있다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
수학적으로 동등한데도 신뢰구간이 가설검정보다 많은 정보를 주는 이유를 설명하라.

</div>

??? success "풀이"
    가설검정은 가설의 값 $\mu_0$ 하나에 대해 기각이냐 비기각이냐라는 이분법적 판정을 낸다. 신뢰구간은 어떤 $\mu_0$ 값들이 기각되고 어떤 값들이 기각되지 않는지를 한꺼번에 보여준다. 다음을 제공한다:

    1. 효과의 **방향**(추정값이 $\mu_0$보다 위인가 아래인가?).
    2. 효과의 **크기**(추정값이 $\mu_0$에서 얼마나 먼가?).
    3. 추정의 **정밀도**(구간이 얼마나 넓은가?).

    예를 들어 신뢰구간 $(0.1, 15.2)$와 $(7.5, 7.8)$은 모두 5% 수준에서 $\mu_0 = 0$을 기각하지만, 앞의 것은 매우 불확실한 추정을, 뒤의 것은 7.65 근처의 정밀한 추정을 시사한다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
신뢰구간과 가설검정의 쌍대성이 단측검정에서도 성립하는가? 그렇다면 대응하는 신뢰한계는 무엇인가?

</div>

??? success "풀이"
    쌍대성은 단측검정으로도 확장되지만, 대응하는 것은 양측 구간이 아니라 **단측 신뢰한계**이다.

    수준 $\alpha$에서 단측검정 $H_0: \mu \leq \mu_0$ 대 $H_1: \mu > \mu_0$에 대응하는 것은 하한 신뢰한계 $(\bar{x} - z_\alpha \cdot \text{SE},\; \infty)$이다. $\mu_0$이 이 하한 아래에 있을 때에만 $H_0$을 기각한다.

    마찬가지로 $H_0: \mu \geq \mu_0$ 대 $H_1: \mu < \mu_0$에 대응하는 것은 상한 신뢰한계 $(-\infty,\; \bar{x} + z_\alpha \cdot \text{SE})$이다. 단측 한계는 검정의 단측 성격을 반영하여 $z_{\alpha/2}$가 아니라 $z_\alpha$를 쓴다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
쌍대성을 **일반적으로** 증명하라. 검정족에서 구간족을 만드는 법과 그 역을 모두 보여라.

</div>

??? success "풀이"
    **설정.** 모수 $\theta$, 자료 $\mathbf X$, 표본공간 $\mathcal X$, 모수공간 $\Theta$.

    **검정 → 구간.** 각 $\theta_0\in\Theta$에 대해 수준 $\alpha$ 검정의 **수용역** $A(\theta_0)\subset\mathcal X$가 주어졌다고 하자. 즉

    $$
    P_{\theta_0}\left\{\mathbf X\in A(\theta_0)\right\}\ge1-\alpha\quad\text{모든 }\theta_0
    $$

    이때

    $$
    C(\mathbf x)=\left\{\theta_0\in\Theta:\ \mathbf x\in A(\theta_0)\right\}
    $$

    로 두면 $C$가 $1-\alpha$ 신뢰집합이다.

    **증명.** 임의의 참값 $\theta$에 대해

    $$
    P_\theta\left\{\theta\in C(\mathbf X)\right\}
    =P_\theta\left\{\mathbf X\in A(\theta)\right\}\ge1-\alpha
    $$

    두 사건이 **정의상 같은 사건**이므로 확률이 같다. $\square$

    **구간 → 검정.** 반대로 $1-\alpha$ 신뢰집합 $C(\mathbf x)$가 주어지면

    $$
    A(\theta_0)=\left\{\mathbf x:\ \theta_0\in C(\mathbf x)\right\}
    $$

    로 두어 "$\theta_0\notin C(\mathbf x)$이면 기각"하는 검정을 만든다. 같은 계산으로 수준이 $\alpha$ 이하다. $\square$

    **핵심.** 두 방향의 증명이 **같은 한 줄**이다. 집합

    $$
    S=\left\{(\mathbf x,\theta):\ \mathbf x\in A(\theta)\right\}
    =\left\{(\mathbf x,\theta):\ \theta\in C(\mathbf x)\right\}
    $$

    의 두 단면일 뿐이기 때문이다.

    **주의할 점 넷.**

    1. **"$\ge1-\alpha$"가 양방향으로 보존된다.** 검정이 보수적이면($<\alpha$) 구간도 보수적이다($>1-\alpha$). 이산분포의 클로퍼-피어슨이 그 예다.

    2. **신뢰"집합"이지 "구간"이 아닐 수 있다.** $A(\theta)$의 모양에 따라 $C(\mathbf x)$가 연결되지 않거나 비어 있을 수 있다. 앞서 본 약한 도구변수의 경우다.

    3. **검정족의 성질이 구간에 옮겨 간다.** 검정이 불편이면 구간이 불편이고, 검정이 UMP이면 구간이 가장 짧다(적절한 의미에서).

    4. **모든 $\theta_0$에 대한 검정이 필요하다.** 하나의 $\theta_0$만으로는 구간을 만들 수 없다. 실무에서 "검정을 여러 번 해서 구간을 얻는다"는 표현이 여기서 나온다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
**$p$-값 함수**를 정의하고, 그것이 모든 신뢰수준의 구간을 한꺼번에 담고 있음을 보여라.

</div>

??? success "풀이"
    **정의.** $p(\theta_0)$를 "$H_0:\theta=\theta_0$의 $p$-값"으로 두면, $\theta_0$의 함수가 된다.

    **핵심 관계.**

    $$
    C_{1-\alpha}(\mathbf x)=\left\{\theta_0:\ p(\theta_0)>\alpha\right\}
    $$

    즉 **$p$-값 함수의 수평 절단이 신뢰구간**이다. $\alpha$를 바꾸면 다른 높이에서 자르는 것이고, 모든 신뢰수준의 구간이 이 하나의 함수에 들어 있다.

    ```python
    import numpy as np
    from scipy import stats

    n, xbar, s = 25, 103.2, 12.0
    se = s / np.sqrt(n)

    def pval(mu0):
        t = (xbar - mu0) / se
        return 2 * stats.t.sf(abs(t), n - 1)

    print(f"{'μ0':>7s} {'p-값':>9s} " + " ".join(f"{lv:>7.0%}" for lv in
                                                 [0.50, 0.80, 0.95, 0.99]))
    for mu0 in [96, 98, 100, 103.2, 106, 108, 110]:
        marks = []
        for lv in [0.50, 0.80, 0.95, 0.99]:
            marks.append(" 안 " if pval(mu0) > 1 - lv else " 밖 ")
        print(f"{mu0:7.1f} {pval(mu0):9.4f} " + " ".join(f"{m:>7s}" for m in marks))

    print()
    for lv in [0.50, 0.80, 0.95, 0.99]:
        t = stats.t.ppf(0.5 + lv / 2, n - 1)
        print(f"{lv:.0%} 구간  ({xbar - t * se:.3f}, {xbar + t * se:.3f})")
    ```

    ```text
         μ0      p-값     50%     80%     95%     99%
       96.0    0.0062     밖      밖      밖      밖 
       98.0    0.0404     밖      밖      밖      안 
      100.0    0.1949     밖      밖      안      안 
      103.2    1.0000     안      안      안      안 
      106.0    0.2548     밖      안      안      안 
      108.0    0.0569     밖      밖      안      안 
      110.0    0.0092     밖      밖      밖      밖 

    50% 구간  (101.556, 104.844)
    80% 구간  (100.037, 106.363)
    95% 구간  (98.247, 108.153)
    99% 구간  (96.487, 109.913)
    ```

    **표와 구간이 정확히 일치한다.** 예컨대 $\mu_0=98$은 $p=0.040$이므로 95% 구간(98.247~108.153) 밖이고 99% 구간(96.487~109.913) 안이다. $\mu_0=96$은 $p=0.0062$로 99% 구간에서도 밖이다.

    **$p$-값 함수의 성질.**

    1. **최댓값이 1이고 $\hat\theta$에서 달성된다.** 점추정값이 곡선의 꼭대기다.
    2. **연속이고 단봉**이면 절단이 항상 구간이 된다. 그렇지 않으면 연결되지 않은 집합이 나올 수 있다.
    3. **곡선의 기울기가 정밀도**를 나타낸다. 가파르면 좁은 구간, 완만하면 넓은 구간.

    **왜 유용한가.**

    - **문턱의 임의성이 사라진다.** 0.05만 보는 대신 전체 곡선을 본다.
    - **경계 근처가 보인다.** $\mu_0=108$이 $p=0.057$로 95% 구간 안이지만 **간신히** 들어왔음이 드러난다.
    - **비대칭이 보인다.** 대칭 $t$ 구간에서는 곡선도 대칭이지만, 비율이나 분산비에서는 비대칭이 곧바로 눈에 띈다.

    **관련 개념.** 이 곡선의 도함수가 **신뢰밀도**이며, 피셔의 신뢰분포(fiducial distribution)의 현대적 형태다. 베이즈 사후분포와 형태가 닮았지만 해석이 다르다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
검정과 구간이 **어긋나는** 실제 사례를 셋 들고, 각각의 원인과 해결책을 밝혀라.

</div>

??? success "풀이"
    **사례 1 — 두 비율 비교.** 검정은 합동 비율로, 구간은 개별 비율로 표준오차를 계산한다.

    ```python
    import numpy as np
    from scipy import stats

    k1, n1, k2, n2 = 30, 100, 18, 100
    p1, p2 = k1 / n1, k2 / n2
    pbar = (k1 + k2) / (n1 + n2)
    z = stats.norm.ppf(0.975)

    se_test = np.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2))
    se_ci = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    zt = (p1 - p2) / se_test
    print(f"검정: z = {zt:.4f}, p = {2 * stats.norm.sf(abs(zt)):.4f}")
    print(f"구간: ({p1 - p2 - z * se_ci:.4f}, {p1 - p2 + z * se_ci:.4f})")
    ```

    ```text
    검정: z = 1.9868, p = 0.0469
    구간: (0.0028, 0.2372)
    ```

    **여기서는 일관되지만** 두 표준오차가 다르므로(검정 0.0604, 구간 0.0598) 경계 근처에서 갈릴 수 있다.

    - **원인**: 검정은 $H_0$ 아래 분산, 구간은 $H_1$ 아래 분산.
    - **해결**: 점수 검정을 **역전**해 구간을 만든다(뉴콤·미텔후트의 점수 구간). 그러면 정의상 일치한다.

    **사례 2 — 왈드 대 우도비.** 같은 모수에 대해 왈드 구간과 우도비 검정을 쓰면 어긋난다.

    - **원인**: 앞서 본 대로 왈드는 모수화에 의존하고, 우도비는 불변이다. 로그가능도가 비대칭이면 차이가 크다.
    - **해결**: **프로파일 우도 구간**을 쓴다. 우도비 검정과 정확히 쌍대다.

    **사례 3 — 다중비교.** 검정에는 본페로니 보정을 하고 구간에는 하지 않는 경우.

    - **원인**: 두 절차의 수준이 다르다.
    - **해결**: 구간에도 같은 보정을 한다($1-\alpha/m$ 수준의 동시 구간).

    **사례 4 — 이산분포의 서로 다른 관행.** 검정은 정확검정, 구간은 왈드를 쓰면 크게 어긋난다.

    ```python
    n, k = 20, 3
    print(f"정확 이항검정 p (H0: p=0.5) = {stats.binomtest(k, n, 0.5).pvalue:.6f}")
    ph = k / n
    se = np.sqrt(ph * (1 - ph) / n)
    print(f"왈드 구간 ({ph - z * se:.4f}, {ph + z * se:.4f})")
    lo = stats.beta.ppf(0.025, k, n - k + 1)
    hi = stats.beta.ppf(0.975, k + 1, n - k)
    print(f"클로퍼-피어슨 구간 ({lo:.4f}, {hi:.4f})")
    ```

    ```text
    정확 이항검정 p (H0: p=0.5) = 0.002577
    왈드 구간 (-0.0065, 0.3065)
    클로퍼-피어슨 구간 (0.0321, 0.3789)
    ```

    두 구간 모두 0.5를 배제하지만, **왈드 구간은 하한이 음수**라 애초에 쓸 수 없는 형태다. 정확검정과 쌍대인 것은 클로퍼-피어슨이다.

    **일반 원칙.** **검정과 구간을 같은 원리로 만든다.** 점수 검정을 쓰면 점수 구간을, 우도비를 쓰면 프로파일 구간을, 정확검정을 쓰면 정확 구간을 쓴다. 섞으면 반드시 어딘가에서 어긋난다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
**동시 신뢰집합**과 다중검정의 쌍대성을 설명하고, 셰페 방법이 무엇을 보장하는지 밝혀라.

</div>

??? success "풀이"
    **개별 대 동시.** 여러 모수 $\theta_1,\dots,\theta_m$에 대해

    - **개별 구간**: 각각 $P(\theta_j\in C_j)\ge1-\alpha$.
    - **동시 구간**: $P(\theta_j\in C_j\ \text{모든 }j)\ge1-\alpha$.

    동시 구간은 **FWER을 통제하는 다중검정과 쌍대**다.

    **세 가지 구성.**

    | 방법 | 보장 범위 | 폭 |
    |---|---|---|
    | 본페로니 | 미리 정한 $m$개 | $z_{1-\alpha/(2m)}$ |
    | 투키 | 모든 **쌍별 차이** | 스튜던트화 범위 |
    | **셰페** | **모든 선형대비** | $\sqrt{(k-1)F_{k-1,\nu,1-\alpha}}$ |

    ```python
    import numpy as np
    from scipy import stats

    k, nu, alpha = 4, 36, 0.05        # 4개 집단, 오차 자유도 36
    print(f"개별 t        {stats.t.ppf(1 - alpha / 2, nu):.4f}")
    for m in [3, 6, 10]:
        print(f"본페로니 m={m:2d}  {stats.t.ppf(1 - alpha / (2 * m), nu):.4f}")
    print(f"셰페          "
          f"{np.sqrt((k - 1) * stats.f.ppf(1 - alpha, k - 1, nu)):.4f}")
    ```

    ```text
    개별 t        2.0281
    본페로니 m= 3  2.5110
    본페로니 m= 6  2.7920
    본페로니 m=10  2.9905
    셰페          2.9324
    ```

    **셰페가 보장하는 것.** 모든 선형대비 $\sum_j c_j\mu_j$($\sum c_j=0$)에 대해 **동시에** 구간이 유효하다. **무한히 많은 대비**를 자료를 본 뒤에 골라도 된다.

    **이것이 결정적인 장점이다.** 분산분석에서 $F$가 유의했을 때 "어느 대비 때문인가"를 자료를 보고 찾아도, 셰페 구간은 여전히 타당하다. 실제로

    > $F$ 검정이 수준 $\alpha$에서 유의하다 $\iff$ 셰페 구간이 0을 담지 않는 대비가 적어도 하나 존재한다

    는 **정확한 동치**가 성립한다. 셰페 방법은 $F$ 검정을 역전한 것이다.

    **대가.** 임계값이 크다. 위 예에서 2.93으로, 개별 $t$(2.03)의 1.45배다. **쌍별 비교만 필요하다면 투키가 훨씬 좁다.**

    **선택 지침.**

    | 상황 | 권장 |
    |---|---|
    | 미리 정한 소수의 비교 | **본페로니**(또는 홀름) |
    | 모든 쌍별 비교 | **투키** |
    | 자료를 보고 대비를 고름 | **셰페** |
    | 대조군과의 비교만 | **더넷** |

    **핵심 교훈.** **"자료를 보고 고른다"는 자유의 대가가 임계값의 크기로 정확히 나타난다.** 셰페(2.932)가 $m=9$쯤의 본페로니와 맞먹는다는 것은, 셰페가 대략 "아홉 개의 비교를 자유롭게 고를 권리"를 준다는 뜻으로 읽을 수 있다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
쌍대성이 **베이즈 신용구간**에서는 어떻게 달라지는지 설명하라.

</div>

??? success "풀이"
    **빈도주의 쌍대성.** 앞서 본 대로 "구간 밖 $\iff$ 기각"이 정의상 성립한다.

    **베이즈에는 그런 자동 대응이 없다.** 신용구간과 가설검정이 **서로 다른 원리**에서 나온다.

    - **신용구간**: 사후분포의 $1-\alpha$ 질량을 담는 영역.
    - **베이즈 검정**: 베이즈 인자나 사후 오즈로 두 모형을 비교.

    **왜 다른가 — 점 귀무가설의 문제.** $H_0:\theta=\theta_0$은 연속 사후분포에서 확률이 0이다. 따라서

    - **신용구간으로 판단**하려면 "$\theta_0$가 구간 안인가"를 묻는데, 이는 사실상 **구간 귀무가설** $|\theta-\theta_0|<\epsilon$을 검정하는 것이다.
    - **베이즈 인자로 판단**하려면 $H_0$에 **양의 사전확률**을 따로 부여해야 한다(스파이크 앤드 슬랩).

    **두 접근이 다른 답을 준다.**

    ```python
    import numpy as np
    from scipy import stats
    from scipy.special import betaln

    n, k = 100, 61
    # 1) 신용구간 (균등 사전분포)
    lo, hi = stats.beta.ppf([0.025, 0.975], k + 1, n - k + 1)
    print(f"95% 신용구간 ({lo:.4f}, {hi:.4f})   0.5 포함? "
          f"{'예' if lo <= 0.5 <= hi else '아니오'}")

    # 2) 베이즈 인자 (H0: p=0.5 에 사전확률 1/2)
    bf10 = np.exp(betaln(k + 1, n - k + 1) - betaln(1, 1) - n * np.log(0.5))
    print(f"BF10 = {bf10:.4f}   →  H0 의 사후확률 {1 / (1 + bf10):.4f}")

    # 3) 빈도주의
    print(f"정확 이항 p-값 {stats.binomtest(k, n, 0.5).pvalue:.4f}")
    ```

    ```text
    95% 신용구간 (0.5118, 0.6999)   0.5 포함? 아니오
    BF10 = 1.3924   →  H0 의 사후확률 0.4180
    정확 이항 p-값 0.0352
    ```

    **세 답이 갈린다.**

    - **신용구간**: 0.5를 배제한다(빈도주의 구간과 거의 같다).
    - **베이즈 인자**: $H_0$의 사후확률이 0.42로, **$H_0$를 배제할 근거가 약하다.**
    - **$p$-값**: 0.035로 기각.

    **왜 신용구간과 베이즈 인자가 다른가.** 신용구간은 **$H_0$에 특별한 지위를 주지 않는다.** $\theta=0.5$는 다른 값들과 똑같이 취급되어, 사후분포의 꼬리에 있으면 배제된다. 베이즈 인자는 $\theta=0.5$에 **사전확률 1/2을 몰아주므로** 그 값이 특별히 보호받는다.

    **어느 쪽이 옳은가 — 질문에 달렸다.**

    - "$\theta$가 얼마인가"라면 **신용구간**.
    - "$\theta$가 정확히 $\theta_0$인 모형이 그럴듯한가"라면 **베이즈 인자**. 다만 그런 질문이 실제로 의미 있는 경우는 생각보다 적다.

    **실무 권고.** 대부분의 상황에서 **점 귀무가설은 인위적**이다. "효과가 정확히 0"이 참일 가능성은 거의 없고, 진짜 질문은 "효과가 무시할 만한가"다. **신용구간과 실무적 문턱을 함께 쓰는 것**(ROPE, region of practical equivalence)이 가장 실용적이다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
쌍대성을 이용해 **검정만 있고 구간이 없는** 상황에서 구간을 만드는 법을 보여라. 순열검정을 예로 들어라.

</div>

??? success "풀이"
    **원리.** 각 $\theta_0$에 대해 검정을 수행하고, 기각되지 않는 $\theta_0$를 모은다. **검정이 있으면 구간은 언제나 만들 수 있다.**

    **순열검정의 경우.** 위치 이동 모형 $Y_j=X_i+\Delta$를 가정하면, $\Delta_0$를 검정하려면 **$Y$에서 $\Delta_0$를 빼고** 순열검정을 하면 된다.

    ```python
    import numpy as np

    rng = np.random.default_rng(77)
    x = np.array([12.1, 15.3, 9.8, 14.2, 11.7, 13.5, 16.1, 10.9])
    y = np.array([17.2, 14.8, 19.3, 16.5, 15.9, 18.1, 13.7])
    n1 = len(x)
    B = 4_999

    def perm_p(delta0):
        """H0: μ_y - μ_x = delta0 의 순열검정 p-값"""
        yy = y - delta0
        obs = yy.mean() - x.mean()
        pooled = np.concatenate([x, yy])
        cnt = 1
        for _ in range(B):
            pm = rng.permutation(pooled)
            if abs(pm[n1:].mean() - pm[:n1].mean()) >= abs(obs) - 1e-12:
                cnt += 1
        return cnt / (B + 1)

    grid = np.arange(0.0, 7.01, 0.25)
    ps = np.array([perm_p(d) for d in grid])
    inside = grid[ps > 0.05]
    print(f"관측 차이 {y.mean() - x.mean():.4f}")
    print(f"순열검정 역전 95% 구간  ({inside.min():.2f}, {inside.max():.2f})")

    from scipy import stats
    r = stats.ttest_ind(y, x, equal_var=False)
    se = np.sqrt(y.var(ddof=1) / len(y) + x.var(ddof=1) / n1)
    df = (y.var(ddof=1) / len(y) + x.var(ddof=1) / n1)**2 / (
        (y.var(ddof=1) / len(y))**2 / (len(y) - 1)
        + (x.var(ddof=1) / n1)**2 / (n1 - 1))
    h = stats.t.ppf(0.975, df) * se
    d0 = y.mean() - x.mean()
    print(f"웰치 t 구간             ({d0 - h:.2f}, {d0 + h:.2f})")
    ```

    ```text
    관측 차이 3.5500
    순열검정 역전 95% 구간  (1.25, 5.75)
    웰치 t 구간             (1.26, 5.84)
    ```

    **두 구간이 거의 같다**(1.25~5.75 대 1.26~5.84). 순열 구간이 격자 간격(0.25) 만큼 거칠 뿐이다.

    **장점.**

    1. **분포가정이 없다.** 정규성을 쓰지 않는다.
    2. **어떤 통계량에도 적용된다.** 중앙값 차이, 절사평균 차이, 최댓값 비.

    **주의할 점 넷.**

    1. **이동 모형이 필요하다.** "$\Delta$를 빼면 두 분포가 같아진다"는 가정이다. 분산이 다르면 성립하지 않는다.

    2. **계산이 무겁다.** 격자의 각 점에서 순열검정을 돌린다. 위 예에서 $29\times5000=145{,}000$번의 순열이다. **이분법 탐색**으로 경계만 찾으면 훨씬 빠르다.

    3. **격자의 해상도가 정밀도를 제한한다.** 경계 근처를 촘촘히 잡거나 이분법을 쓴다.

    4. **몬테카를로 오차.** $B$가 작으면 경계가 흔들린다. 구간의 끝을 정할 때는 $B$를 크게 한다.

    **같은 기법이 쓰이는 다른 곳.**

    - **앤더슨-루빈 구간**(약한 도구변수): 각 $\beta_0$에서 검정을 역전. 앞서 본 대로 빈 집합이나 무한 구간이 나올 수 있다.
    - **프로파일 우도 구간**: 우도비 검정의 역전.
    - **부트스트랩 검정의 역전**: 잘 쓰이지는 않지만 원리는 같다.
    - **선택 후 추론**: 조건부 검정을 역전해 구간을 만든다.

---

## 정리하며

- 양측검정에서 신뢰구간과 가설검정은 동등한 정보를 준다.
- 신뢰구간은 기각/비기각의 이분법적 판정만이 아니라 그럴듯한 값의 범위를 보여주므로 흔히 더 유익하다.
- 결과를 보고할 때는 p-값과 신뢰구간을 함께 제시하는 것이 좋다.
- 쌍대성은 양측검정에서 정확히 성립하며, 단측검정은 단측 신뢰한계에 대응한다.
