# p의 신뢰구간

## 일표본 비율 신뢰구간

많은 통계 문제에서 우리는 모비율 $p$ — 모집단에서 어떤 특성을 가진 개체의 비율 — 를 추정하는 데 관심이 있다. 예를 들어 특정 후보를 지지하는 유권자의 비율이나 한 배치에서 불량품의 비율 같은 것이다.

### 공식 (Wald z-구간)

모비율 $p$에 대한 신뢰구간의 일반형은

$$
\hat{p} \pm z_{\alpha/2} \times \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}
$$

여기서

- $\hat{p}$는 표본비율,
- $\alpha$는 유의수준($\text{유의수준} = 1 - \text{신뢰수준}$),
- $z_{\alpha/2}$는 $P(Z > z_{\alpha/2}) = \alpha/2$를 만족하는 표준정규분포의 임계값,
- $n$은 표본크기,
- $\sqrt{\hat{p}(1 - \hat{p})/n}$은 표본비율의 표준오차이다.

### 타당성 조건

$$
\hat{p}\pm z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
\quad\text{if}\quad
\begin{cases}
n\hat{p}\ge 10 \text{ 이고 } n(1-\hat{p})\ge 10 \text{ (중심극한정리)} \\
n \ge 30 \text{ (대수의법칙)} \\
n \le 0.1N \text{ (i.i.d.)}
\end{cases}
$$

비율의 표본분포가 근사적으로 정규가 되도록 표본크기 $n$이 충분히 커야 한다.

<div class="codebox" markdown>

#### 예제 1. 비율의 신뢰구간 계산 { .eg }

```python
import numpy as np
import scipy.stats as stats

n = 200          # 표본크기
x = 120          # 성공 횟수
confidence_level = 0.95

p_hat = x / n
z_critical = stats.norm.ppf(1 - (1 - confidence_level) / 2)
# 평균의 구간과 달리 여기서는 산포를 따로 추정하지 않는다.
# Bernoulli 분포에서는 분산 p(1-p)가 평균 p에 딸려 오기 때문이다.
# 그래서 p_hat 하나로 중심과 너비가 모두 정해진다.
standard_error = np.sqrt((p_hat * (1 - p_hat)) / n)
margin_of_error = z_critical * standard_error
confidence_interval = (p_hat - margin_of_error, p_hat + margin_of_error)

print(f"{confidence_interval = }")
```

출력:

```
confidence_interval = (0.5321048559554297, 0.6678951440445703)
```

타당성 조건 $n\hat p = 120 \ge 10$과 $n(1-\hat p) = 80 \ge 10$을 넉넉히 만족하므로 Wald 구간을 써도 되는 경우다.

</div>

---

## Wald z-구간의 대안

Wald 구간은 이런 형태의 구간을 정규근사로 형식화한 [Abraham Wald](https://en.wikipedia.org/wiki/Abraham_Wald)의 이름을 딴 것이다. 그러나 $n$이 작거나, $\hat{p}$가 0 또는 1에 가깝거나, $n\hat{p}$ 또는 $n(1-\hat{p})$가 10보다 작으면 **Wald 구간의 성능이 나쁘다**. 이런 경우에는 포함확률이 명목 수준보다 훨씬 낮아질 수 있다.

| 구간 | 공식의 종류 | $z$를 쓰는가? | 잘 통하는 경우 | 비고 |
|---|---|---|---|---|
| **Wald (z)** | $\hat{p} \pm z\sqrt{\hat{p}(1-\hat{p})/n}$ | 예 | $n$이 큰 경우 | 단순하지만 작은 표본에서 부정확 |
| **Wilson score** | z-검정을 뒤집어 유도 | 예 | 작은 $n$부터 큰 $n$까지 | 포함확률이 훨씬 좋음 |
| **Agresti–Coull** | 보정된 Wald(가상 관측값 추가) | 예 | 작은~중간 $n$ | 손쉬운 보완, Wilson에 근접한 성능 |
| **Clopper–Pearson** | 이항에 기반 | 아니오 | 작은 $n$ | 보수적이지만 정확 |

### Wilson score 구간

Wilson 구간은 비율에 대한 z-검정을 *뒤집어서* 얻는다:

$$
\frac{(\hat{p} - p)^2}{p(1-p)/n} = z_{\alpha/2}^2
$$

$p$에 대해 풀면:

$$
\text{CI} =
\frac{
\hat{p} + \frac{z^2}{2n} \pm
z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}
}{
1 + \frac{z^2}{n}
}
$$

구간의 중심은 $\hat{p}$가 **아니라** 0.5 쪽으로 **축소된 값**이다:

$$
\tilde{p} = \frac{\hat{p} + \frac{z^2}{2n}}{1 + \frac{z^2}{n}}
$$

이 구간은 $[0, 1]$ 안에 머물며, 작거나 치우친 표본에서 훨씬 좋은 성능을 보여 $n < 30$에서도 거의 명목 포함확률을 달성한다.

### Agresti–Coull 구간

Agresti와 Coull은 Wilson 공식을 "가상 관측값"을 더하는 것만으로 근사할 수 있음을 관찰했다. 95% 신뢰수준($z = 1.96 \approx 2$)에서는:

- 성공 2개와 실패 2개를 더한다 → 사실상 관측값 4개가 추가된다.
- 보정된 개수를 쓴다: $n' = n + 4$, $x' = x + 2$, $\tilde{p} = x'/n'$.
- 보정된 비율로 Wald 형태의 구간을 계산한다:

$$
\tilde{p} \pm z_{\alpha/2} \sqrt{\frac{\tilde{p}(1 - \tilde{p})}{n'}}
$$

포함확률이 Wilson에 매우 가깝고 설명하기도 손으로 계산하기도 쉽다. $n$이 크면 Wilson과 같아진다.

### 비교 정리

| 방법 | 중심 | 보정 | 성능 |
|---|---|---|---|
| **Wald (z)** | $\hat{p}$ | 없음 | 작은 표본·경계 근처에서 나쁨 |
| **Wilson score** | $\hat{p}$와 0.5의 가중평균 | 중심과 너비를 함께 조정 | 훌륭함 |
| **Agresti–Coull** | $(x+2)/(n+4)$ | 가상 자료 추가 | Wilson에 거의 맞먹음 |

---

## 보기

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> 유권자 지지율의 95% 신뢰구간. 유권자 200명의 확률표본에서 120명이 특정 후보를 지지한다고 답했다. 참 비율의 95% 신뢰구간을 구성하라.

</div>

??? success "풀이"

    $$
    \hat{p} = \frac{120}{200} = 0.60
    $$

    95% 신뢰수준에서 $z_{\alpha/2} \approx 1.96$이다.

    $$
    \text{SE} = \sqrt{\frac{0.60 \times 0.40}{200}} = \sqrt{0.0012} \approx 0.03464
    $$

    $$
    \text{ME} = 1.96 \times 0.03464 \approx 0.0679
    $$

    $$
    \boxed{(0.5321,\ 0.6679)}
    $$

    그 후보를 지지하는 유권자의 참 비율이 0.5321과 0.6679 사이에 있다고 95% 신뢰한다.
<div class="exbox" markdown>

**보기 2.** <span class="diff easy" title="쉬움"></span> 학교 재정 조사를 위한 표본크기. Della는 비율에 대해 95% 신뢰수준에서 오차한계를 $\pm 2\%$보다 작게 하려 한다. 필요한 최소 표본크기는?

</div>

??? success "풀이"
    최악의 표준오차는 ($\hat{p}(1-\hat{p})$를 최대화하는) $\hat{p} = 0.5$에서 나온다.

    ```python
    import scipy.stats as stats
    import numpy as np

    confidence_level = 0.95
    alpha = 1 - confidence_level
    z_star = stats.norm().ppf(1 - alpha / 2)
    margin_of_error_max = 0.02
    p_max = 0.5

    # p를 모르는 채로 표본크기를 정해야 하므로 최악을 가정한다.
    # p(1-p)는 p=0.5에서 최대(0.25)이니, 이 n이면 참 p가 무엇이든 안전하다.
    n = 1
    while True:
        n += 1
        me = z_star * np.sqrt(p_max * (1 - p_max) / n)
        if me <= margin_of_error_max:
            break
    print(f"{n = }")
    ```

    출력:

    ```
    n = 2401
    ```

    공식으로 계산하면 $n = (z_{0.025}/2E)^2 = 2400.91$이고 올림하여 2401이다($z = 1.96$으로 반올림해 쓰면 정확히 2401이 된다). 참 비율이 0.5에서 멀면 이만큼 필요하지 않다. 사전 정보로 $p \approx 0.1$을 쓸 수 있다면 같은 오차한계에 865개면 된다.
<div class="exbox" markdown>

**보기 3.** <span class="diff easy" title="쉬움"></span> 여성 아티스트의 노래 (99% 신뢰구간). Della는 노래를 500곡 넘게 가지고 있다. 무작위로 50곡을 골랐더니 20곡이 여성 아티스트의 노래였다. 99% 신뢰구간을 구성하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats

    confidence_level = 0.99
    alpha = 1 - confidence_level
    p_hat = 20 / 50
    n = 50

    z_star = stats.norm().ppf(1 - alpha / 2)
    margin_of_error = z_star * np.sqrt(p_hat * (1 - p_hat) / n)
    print(f"{p_hat} ± {margin_of_error:.3f}")
    ```

    출력:

    ```
    0.4 ± 0.178
    ```

    99% 신뢰구간은 대략 $(0.222, 0.578)$이다.

    여기서 한 가지 짚어 둘 것이 있다. 500곡 중 50곡을 뽑았으므로 $n/N = 0.1$이고, i.i.d. 근사가 아슬아슬한 경계에 있다. 유한모집단 수정을 넣으면 $\sqrt{(500-50)/499} = 0.950$이 곱해져 오차한계가 0.178에서 0.169로 줄어든다.
---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
차량 100대 중 74대가 검사를 통과했다. $p$의 95% 신뢰구간을 구하라.

</div>

??? success "풀이"
    $\hat p = 0.74$. $\mathrm{SE} = \sqrt{0.74 \cdot 0.26/100} \approx 0.0439$. 오차한계 = $1.96 \cdot 0.0439 \approx 0.086$.

    신뢰구간: $(0.654, 0.826)$.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
200명 중 120명이 A를 선호한다. $p$의 90% 신뢰구간을 구하라.

</div>

??? success "풀이"
    $\hat p = 0.60$. $z_{0.05} = 1.645$. 오차한계 = $1.645 \sqrt{0.60 \cdot 0.40/200} \approx 0.057$.

    신뢰구간: $(0.543, 0.657)$.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
유권자 500명 중 280명이 지지한다. (a) Wald 95% 신뢰구간. (b) Wilson 95% 신뢰구간. (c) $p > 0.5$라는 증거가 있는가?

</div>

??? success "풀이"
    (a) $\hat p = 0.56$. 오차한계 = $1.96 \cdot 0.0222 = 0.0435$. Wald 신뢰구간: $(0.517, 0.604)$.

    (b) Wilson 신뢰구간: 중심 $= (0.56 + 1.96^2/1000)/(1 + 1.96^2/500) \approx 0.560$. 신뢰구간 $\approx (0.516, 0.603)$. Wald와 거의 같다 — $n$이 크고 $\hat p$가 중간이면 Wilson과 Wald가 일치한다.

    (c) 신뢰구간 전체가 0.5보다 위에 있다: 5% 수준에서 과반이 이 안건을 지지한다는 증거이다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
**경계 근처에서의 Wilson 구간.** $n = 20$, $X = 2$(즉 $\hat p = 0.1$)에 대해 Wald와 Wilson 신뢰구간을 비교하라.

</div>

??? success "풀이"
    Wald: 오차한계 $= 1.96 \sqrt{0.1 \cdot 0.9/20} \approx 0.131$. Wald 신뢰구간: $(0.1 - 0.131, 0.1 + 0.131) = (-0.031, 0.231)$ — **0 아래로 뻗는다**.

    Wilson: 중심 $\approx (0.1 + 1.96^2/40)/(1 + 1.96^2/20) \approx 0.196/1.192 \approx 0.164$. 반너비: $1.96\sqrt{0.1 \cdot 0.9/20 + 1.96^2/1600}/1.192 \approx 0.137$. 신뢰구간 $\approx (0.028, 0.301)$ — $[0, 1]$ 안에 머문다.

    경계 근처에서는 Wilson이 훨씬 낫다. 이항 신뢰구간에는 Wilson을 기본으로 삼으라.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
**비율 신뢰구간을 위한 표본크기.** $\hat p$와 무관하게 95% 신뢰구간의 오차한계가 $0.03$ 이하가 되도록 하는 $n$을 구하라.

</div>

??? success "풀이"
    $\mathrm{ME} = 1.96\sqrt{\hat p(1-\hat p)/n} \le 0.03$.

    최악의 경우는 $\hat p = 1/2$에서 $\hat p(1-\hat p) = 0.25$일 때이다.

    $1.96 \sqrt{0.25/n} \le 0.03 \Rightarrow n \ge (1.96)^2 \cdot 0.25/(0.03)^2 = 0.9604/0.0009 \approx 1068$.

    참 $p$가 무엇이든 **$n = 1068$**이면 95% 신뢰수준에서 오차한계 $0.03$ 이하가 된다. 여론조사의 "$n \approx 1000$" 규칙이 여기서 나온다.

    $p$가 0.1이나 0.9 근처라고 짐작된다면 $p(1-p) = 0.09$이므로 $n \ge 385$면 된다. $p$의 대략적인 값을 알면 필요한 $n$을 약 64% 줄일 수 있다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
**연속성 보정.** 언제 신뢰구간을 개선하는가? 연습문제 1에 적용해 보라.

</div>

??? success "풀이"
    연속성 보정은 이항분포에 정규근사를 쓸 때 반너비에 $1/(2n)$을 더한다. 연습문제 1에서는 보정한 오차한계가 $\approx 0.086 + 1/200 = 0.091$이다.

    보정한 신뢰구간: $(0.649, 0.831)$, 보정하지 않은 것: $(0.654, 0.826)$. 약간 더 넓다.

    **언제 쓰는가:**

    - 연속성이 문제가 되는 작거나 중간 크기의 $n$.
    - 꼬리 확률이 중요할 때(단측 검정).
    - 이항분포의 이산성이 무시할 수 없을 때.

    현대의 실무: 보정 대신 정확한 방법(Clopper-Pearson)이나 Wilson 구간을 쓴다. 연속성 보정은 컴퓨터 이전 시대 근사의 유산이다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
비율의 오차한계를 3%포인트 이하로 하려면 $n$이 얼마여야 하는가? $p$를 모를 때, 사전정보가 있을 때, 그리고 신뢰수준을 바꿀 때를 각각 다루어라.

</div>

??? success "풀이"
    **기본식.** 오차한계 $E=z\sqrt{p(1-p)/n}$을 $n$에 대해 풀면

    $$
    n=\frac{z^2p(1-p)}{E^2}
    $$

    **$p$를 모를 때.** $p(1-p)$는 $p=0.5$에서 최대 0.25다. **가장 보수적인 선택**이다.

    ```python
    import numpy as np
    from scipy import stats

    E = 0.03
    print(f"{'신뢰수준':>8s} {'z':>7s} {'p=0.5':>8s} {'p=0.2':>8s} {'p=0.05':>8s}")
    for lv in [0.90, 0.95, 0.99]:
        z = stats.norm.ppf(0.5 + lv / 2)
        ns = [int(np.ceil(z**2 * p * (1 - p) / E**2)) for p in [0.5, 0.2, 0.05]]
        print(f"{lv:8.0%} {z:7.4f} {ns[0]:8d} {ns[1]:8d} {ns[2]:8d}")
    ```

    ```text
        신뢰수준       z    p=0.5    p=0.2   p=0.05
         90%  1.6449      752      481      143
         95%  1.9600     1068      683      203
         99%  2.5758     1844     1180      351
    ```

    **읽기.**

    - **95%에서 $p$를 모르면 $n=1068$.** 여론조사에서 "표본 1000명, 오차 $\pm3.1\%$포인트"라는 문구가 여기서 나온다.
    - **사전정보가 큰 차이를 만든다.** $p\approx0.2$임을 알면 683명으로 충분하다. 36% 절약이다.
    - **$p=0.05$처럼 극단이면 203명**이면 되지만, 이때는 정규근사 자체가 나빠 윌슨 기준으로 다시 계산해야 한다.

    **$E$에 대한 민감도.** $n\propto E^{-2}$이므로

    | $E$ | $n$ ($p=0.5$, 95%) |
    |---|---|
    | 5%p | 385 |
    | 3%p | 1,068 |
    | 2%p | 2,401 |
    | 1%p | 9,604 |

    **오차를 절반으로 줄이려면 표본이 네 배**다. 1%포인트를 노리면 1만 명이 필요하고, 이 지점에서 표집오차보다 무응답 편향이 훨씬 커지므로 **투자할 가치가 없다.**

    **주의.** 이 계산은 **단순임의추출**을 가정한다. 실제 조사는 층화·군집 설계를 쓰므로 다음 문제에서 볼 설계효과를 곱해야 한다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
학교 40곳에서 각각 25명씩 뽑아 1000명을 조사했다. 급내상관이 $\rho=0.05$일 때 **설계효과**를 계산하고, 신뢰구간을 어떻게 고쳐야 하는지 보여라.

</div>

??? success "풀이"
    **설계효과.** 크기 $m$인 군집을 뽑을 때

    $$
    \text{DEFF}=1+(m-1)\rho
    $$

    이고, **유효표본크기**는 $n_{\text{eff}}=n/\text{DEFF}$다.

    ```python
    import numpy as np
    from scipy import stats

    n, m, rho, phat = 1000, 25, 0.05, 0.42
    deff = 1 + (m - 1) * rho
    n_eff = n / deff
    z = stats.norm.ppf(0.975)

    se0 = np.sqrt(phat * (1 - phat) / n)
    se1 = np.sqrt(phat * (1 - phat) / n_eff)
    print(f"DEFF = 1 + ({m}-1)×{rho} = {deff:.2f}")
    print(f"유효표본크기 {n_eff:.1f}명 (실제 {n}명)")
    print(f"군집 무시  SE {se0:.5f}  구간 ({phat - z * se0:.4f}, {phat + z * se0:.4f})  "
          f"오차한계 {z * se0:.4f}")
    print(f"설계 반영  SE {se1:.5f}  구간 ({phat - z * se1:.4f}, {phat + z * se1:.4f})  "
          f"오차한계 {z * se1:.4f}")
    print(f"오차한계가 {se1 / se0:.2f}배로 커진다")
    ```

    ```text
    DEFF = 1 + (25-1)×0.05 = 2.20
    유효표본크기 454.5명 (실제 1000명)
    군집 무시  SE 0.01561  구간 (0.3894, 0.4506)  오차한계 0.0306
    설계 반영  SE 0.02315  구간 (0.3746, 0.4654)  오차한계 0.0454
    오차한계가 1.48배로 커진다
    ```

    **1000명을 조사했지만 정보량은 455명어치다.** 같은 학교 학생들이 서로 비슷하기 때문에, 한 학교에서 25명을 뽑아도 25명분의 독립정보를 얻지 못한다.

    **$\rho$가 작아도 $m$이 크면 심각하다.**

    | $\rho$ | $m=5$ | $m=25$ | $m=100$ |
    |---|---|---|---|
    | 0.01 | 1.04 | 1.24 | 1.99 |
    | 0.05 | 1.20 | 2.20 | 5.95 |
    | 0.10 | 1.40 | 3.40 | 10.9 |

    **$(m-1)\rho$가 관건**이다. 급내상관이 0.01처럼 작아 보여도 군집이 100명이면 유효표본이 절반이 된다.

    **설계 함의.** 같은 예산에서 **군집을 많이, 각 군집은 작게** 뽑는 것이 유리하다. 학교 40곳×25명보다 학교 100곳×10명이 낫다(DEFF가 2.20에서 1.45로 준다). 다만 학교를 추가하는 비용이 학생을 추가하는 비용보다 크므로, 최적 $m$은 비용비와 $\rho$로 결정된다.

    **보고.** 조사 결과를 낼 때 **설계효과나 유효표본크기를 함께** 적어야 한다. "$n=1000$, 오차 $\pm3.1\%$p"는 군집설계에서 거짓이다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
세 지역(가중치 0.5, 0.3, 0.2)에서 각각 $n_h=(300, 200, 200)$을 뽑아 $\hat p_h=(0.40, 0.55, 0.30)$을 얻었다. **층화 비율**의 95% 구간을 구하고, 층화를 무시했을 때와 비교하라.

</div>

??? success "풀이"
    **층화추정량.**

    $$
    \hat p_{\text{st}}=\sum_h W_h\hat p_h,\qquad
    \operatorname{Var}(\hat p_{\text{st}})=\sum_h W_h^2\frac{\hat p_h(1-\hat p_h)}{n_h}
    $$

    여기서 $W_h$는 **모집단에서의** 층 비중이다.

    ```python
    import numpy as np
    from scipy import stats

    W = np.array([0.5, 0.3, 0.2])
    nh = np.array([300, 200, 200])
    ph = np.array([0.40, 0.55, 0.30])
    z = stats.norm.ppf(0.975)

    p_st = (W * ph).sum()
    var_st = (W**2 * ph * (1 - ph) / nh).sum()
    se_st = np.sqrt(var_st)

    n = nh.sum()
    p_pool = (nh * ph).sum() / n                 # 층화를 무시하고 단순 합산
    se_pool = np.sqrt(p_pool * (1 - p_pool) / n)

    print(f"층화 추정 {p_st:.4f}  SE {se_st:.5f}  "
          f"구간 ({p_st - z * se_st:.4f}, {p_st + z * se_st:.4f})")
    print(f"단순 합산 {p_pool:.4f}  SE {se_pool:.5f}  "
          f"구간 ({p_pool - z * se_pool:.4f}, {p_pool + z * se_pool:.4f})")
    print(f"설계효과 DEFF = {var_st / (p_st * (1 - p_st) / n):.4f}")

    nh_prop = np.array([350, 210, 140])          # 비례배분이었다면
    var_prop = (W**2 * ph * (1 - ph) / nh_prop).sum()
    print(f"비례배분일 때 SE {np.sqrt(var_prop):.5f}  "
          f"DEFF = {var_prop / (p_st * (1 - p_st) / n):.4f}")
    ```

    ```text
    층화 추정 0.4250  SE 0.01880  구간 (0.3882, 0.4618)
    단순 합산 0.4143  SE 0.01862  구간 (0.3778, 0.4508)
    설계효과 DEFF = 1.0122
    비례배분일 때 SE 0.01837  DEFF = 0.9668
    ```

    **두 가지가 다르다.**

    1. **추정값이 다르다.** 0.4250 대 0.4143. 표본배분 $(300,200,200)$이 모집단 비중 $(0.5,0.3,0.2)$과 다르기 때문이다. **비례배분이 아니면 단순 합산은 편향된다.** 1지역이 모집단의 50%인데 표본에서는 42.9%뿐이라 과소대표되고, 비율이 가장 낮은 3지역이 20% 대신 28.6%로 과대대표되어 합산값이 아래로 끌린다.

    2. **표준오차도 다르다.** 이 배분에서는 DEFF가 **1.012로 1보다 크다.** 층화 자체는 분산을 줄이지만, **배분이 비례에서 벗어난 손실이 그 이득을 조금 넘어섰다.** 같은 700명을 비례배분 $(350,210,140)$으로 뽑았다면 DEFF가 0.967로 내려가 단순임의추출보다 유리해진다.

    **층화의 두 얼굴.**

    - **편향 제거.** 배분이 비례가 아닐 때 가중이 필수다. 이것은 언제나 이득이다.
    - **분산.** 비례배분이면 DEFF $\le1$이 보장되지만, 비례에서 벗어나면 1을 넘을 수 있다. **층별 추정을 위해 작은 층을 과대표집하면 전체 추정의 정밀도를 내주는 셈**이다.

    **군집과 반대 방향이다.** 앞 문제에서 군집은 DEFF를 1보다 크게 만들었다. 실제 조사는 **층화와 군집을 함께** 쓰므로 두 효과가 상쇄되며, 총 DEFF는 보통 1.5~3 사이다.

    **최적 배분.** 분산을 최소화하는 네이만 배분은 $n_h\propto W_h\sigma_h$다. 비율이라면 $\sigma_h=\sqrt{p_h(1-p_h)}$이므로 $p_h$가 0.5에 가까운 층에 더 많이 배분한다. 다만 층별 추정도 필요하다면 최소 표본수를 보장해야 한다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
**로짓 변환**으로 비율의 구간을 만드는 방법을 설명하고, $n=30$, $k=3$에서 윌슨·클로퍼-피어슨과 비교하라.

</div>

??? success "풀이"
    **착안.** $\hat p$는 $[0,1]$에 갇혀 있어 경계 근처에서 정규근사가 나쁘다. **로짓**

    $$
    \hat\eta=\log\frac{\hat p}{1-\hat p}
    $$

    은 실수 전체를 값으로 가지므로 정규근사가 낫다. 델타법으로

    $$
    \operatorname{Var}(\hat\eta)\approx\frac{1}{n\hat p(1-\hat p)}=\frac1k+\frac1{n-k}
    $$

    구간을 로짓 척도에서 만들고 **역로짓으로 되돌린다.**

    ```python
    import numpy as np
    from scipy import stats

    n, k = 30, 3
    z = stats.norm.ppf(0.975)
    ph = k / n

    # 로짓 구간
    eta = np.log(ph / (1 - ph))
    se_eta = np.sqrt(1 / k + 1 / (n - k))
    lo_l, hi_l = [1 / (1 + np.exp(-(eta + s * z * se_eta))) for s in (-1, 1)]

    # 왈드
    se = np.sqrt(ph * (1 - ph) / n)
    lo_w, hi_w = ph - z * se, ph + z * se

    # 윌슨
    d = 1 + z**2 / n
    c = (ph + z**2 / (2 * n)) / d
    h = z / d * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2))

    # 클로퍼-피어슨
    lo_c = stats.beta.ppf(0.025, k, n - k + 1)
    hi_c = stats.beta.ppf(0.975, k + 1, n - k)

    for name, lo, hi in [("왈드", lo_w, hi_w), ("로짓", lo_l, hi_l),
                         ("윌슨", c - h, c + h), ("클로퍼-피어슨", lo_c, hi_c)]:
        print(f"{name:12s} ({lo:.4f}, {hi:.4f})  폭 {hi - lo:.4f}")
    ```

    ```text
    왈드           (-0.0074, 0.2074)  폭 0.2147
    로짓           (0.0326, 0.2681)  폭 0.2355
    윌슨           (0.0346, 0.2562)  폭 0.2216
    클로퍼-피어슨      (0.0211, 0.2653)  폭 0.2442
    ```

    **읽기.**

    - **왈드는 하한이 음수**다. 확률이 음수일 수 없으므로 명백히 잘못이다.
    - **로짓 구간은 $(0,1)$ 안에 있다.** 역로짓이 항상 $(0,1)$로 사상하기 때문에 **구조적으로 보장**된다.
    - **로짓과 윌슨은 하한이 거의 같다**(0.033 대 0.035). 상한은 로짓이 0.268로 윌슨의 0.256보다 높아, 로짓 구간이 조금 더 넓고 오른쪽으로 더 뻗는다.
    - **클로퍼-피어슨이 가장 넓다.** 보수적인 정확 구간이다.

    **로짓의 치명적 결함.** $k=0$이나 $k=n$이면 $\hat\eta=\mp\infty$이고 분산도 무한대라 **구간을 만들 수 없다.** 흔한 대처는 **$k$와 $n-k$에 0.5씩 더하는** 것인데(하일랜드 수정), 임시방편이다.

    **어느 것을 쓰는가.** 단독 비율이라면 **윌슨**이 가장 낫다. 로짓이 유용한 곳은 **회귀모형**이다. 로지스틱회귀의 계수 구간을 만들고 되돌리면 자동으로 로짓 구간이 되며, 여러 변수를 함께 다룰 수 있다.

---

## 정리하며

비율의 왈드 구간은 익숙하지만 **실제 포함확률이 나쁘기로 유명하다.**

$$
\hat p \pm z_{\alpha/2}\sqrt{\frac{\hat p(1-\hat p)}{n}}
$$

- **표준오차가 모수에 의존한다.** $\sqrt{p(1-p)/n}$ 의 $p$ 를 $\hat p$ 로 대신하므로, 추정값이 나쁘면 구간의 폭도 함께 틀린다.
- **경계에서 무너진다.** $\hat p=0$ 이면 표준오차가 $0$ 이 되어 구간이 **한 점** $[0,0]$ 이 된다. 성공이 하나도 없었다고 $p=0$ 이라 단정하는 셈이다.
- **명목 $95\%$ 가 실제로는 훨씬 낮다.** $p$ 가 $0$ 이나 $1$ 에 가깝거나 $n$ 이 작으면 포함확률이 $80\%$ 대로 떨어지는 일이 흔하며, $n$ 을 늘려도 **매끄럽게 좋아지지 않고 톱니처럼 오르내린다**(이산성 때문이다).
- **대안이 있다.** **윌슨 점수구간**은 $\hat p$ 가 아니라 $p$ 를 기준으로 풀어 경계에서도 합리적인 답을 주고, **아그레스티–쿨 구간**은 성공과 실패에 각각 2 를 더하는 간단한 보정으로 비슷한 개선을 얻는다. **소표본이나 극단 비율에서는 왈드 대신 이쪽을 쓴다.**

다음 절 **$\sigma^2$ 의 신뢰구간**으로 넘어간다. 이번에는 분포가 비대칭이라 구간도 비대칭이 된다.
