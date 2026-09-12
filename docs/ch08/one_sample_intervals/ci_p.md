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

### Python 코드

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

---

## Wald z-구간의 대안

Wald 구간은 이런 형태의 구간을 정규근사로 형식화한 [Abraham Wald](https://en.wikipedia.org/wiki/Abraham_Wald)의 이름을 딴 것이다. 그러나 $n$이 작거나, $\hat{p}$가 0 또는 1에 가깝거나, $n\hat{p}$ 또는 $n(1-\hat{p})$가 10보다 작으면 **Wald 구간의 성능이 나쁘다**. 이런 경우에는 포함확률이 명목 수준보다 훨씬 낮아질 수 있다.

| 구간 | 공식의 종류 | $z$를 쓰는가? | 잘 통하는 경우 | 비고 |
|---|---|---|---|---|
| **Wald (z)** | $\hat{p} \pm z\sqrt{\hat{p}(1-\hat{p})/n}$ | 예 | $n$이 큰 경우 | 단순하지만 작은 표본에서 부정확 |
| **Wilson score** | z-검정을 뒤집어 유도 | 예 | 작은 $n$부터 큰 $n$까지 | 포함확률이 훨씬 좋음 |
| **Agresti–Coull** | 보정된 Wald(가상 관측값 추가) | 예 | 작은~중간 $n$ | 손쉬운 보완, Wilson에 근접한 성능 |
| **Clopper–Pearson** | Binomial에 기반 | 아니오 | 작은 $n$ | 보수적이지만 정확 |

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
