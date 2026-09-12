# Welch 분산분석의 제1종 오류와 검정력 모의실험

## 개요

Welch 분산분석은 집단 사이의 등분산을 가정하지 않는, 고전적 일원배치 분산분석의 대안이다. 이 페이지에서는 Monte Carlo 모의실험으로 이분산이고 집단 크기가 불균형한 조건에서 Welch 분산분석의 제1종 오류율과 검정력을 추정한다. 분산이 크게 달라도 Welch 분산분석이 명목 제1종 오류율을 유지하면서 집단 평균 차이를 탐지할 만한 검정력을 갖는다는 점을 보인다.

## 왜 Welch 분산분석인가

고전적 일원배치 분산분석은 등분산성 $\sigma_1^2 = \sigma_2^2 = \cdots = \sigma_k^2$을 가정한다. 이 가정이 어긋나고 표본크기도 다르면 고전적 $F$-검정의 제1종 오류율이 부풀려질 수 있다. Welch 분산분석은 가중된 형태를 쓴다:

$$
F_W = \frac{\sum_{i=1}^{k} w_i (\bar{y}_{i\cdot} - \tilde{y})^2 / (k-1)}{1 + \frac{2(k-2)}{k^2-1} \sum_{i=1}^{k} \frac{(1 - w_i/\sum w_j)^2}{n_i - 1}}
$$

여기서 $w_i = n_i / s_i^2$이고 $\tilde{y} = \sum w_i \bar{y}_i / \sum w_j$이다. 분모가 Satterthwaite 형태의 근사로 자유도를 조정하여 이분산에 로버스트한 검정을 만든다.

## 모의실험 설계

이 모의실험은 분산과 크기를 일부러 다르게 한 세 집단을 비교한다:

| 집단 | $n_i$ | $\sigma_i$ | $\mu_i$ (귀무) | $\mu_i$ (대립) |
|---|---|---|---|---|
| $G_1$ | 10 | 1.0 | 10.0 | 10.0 |
| $G_2$ | 18 | 3.0 | 10.0 | 10.0 |
| $G_3$ | 7 | 6.0 | 10.0 | 12.0 |

귀무가설 아래에서는 모든 평균이 같다. 대립가설 아래에서는 집단 $G_3$이 2만큼 위로 이동한다.

```python
import numpy as np
import pandas as pd
import pingouin as pg

rng = np.random.default_rng(0)

def simulate_once(null=True):
    ns = [10, 18, 7]
    sigmas = [1.0, 3.0, 6.0]
    means = [10.0, 10.0, 10.0] if null else [10.0, 10.0, 12.0]

    rows = []
    for i, (n, mu, sd) in enumerate(zip(ns, means, sigmas), start=1):
        x = rng.normal(mu, sd, size=n)
        rows += [{"Group": f"G{i}", "Values": v} for v in x]
    df = pd.DataFrame(rows)

    aov = pg.welch_anova(dv="Values", between="Group", data=df)
    # pingouin 0.6부터 열 이름이 "p-unc"에서 "p_unc"로 바뀌었다.
    # 두 이름을 모두 받아들여 버전에 무관하게 동작하도록 한다.
    col = "p_unc" if "p_unc" in aov.columns else "p-unc"
    return float(aov[col].iloc[0])

# 한 번 돌려 형태를 확인한다.
print(f"single run p-value (null) = {simulate_once(null=True):.4f}")
```

출력:

```
single run p-value (null) = 0.4545
```

## 모의실험 실행

각 시나리오(귀무와 대립)마다 많은 반복을 생성하여 $\alpha = 0.05$에서의 기각률을 추정한다.

```python
def run(n_sims=500, alpha=0.05):
    pvals_null = [simulate_once(null=True) for _ in range(n_sims)]
    pvals_alt  = [simulate_once(null=False) for _ in range(n_sims)]
    type1 = np.mean(np.array(pvals_null) < alpha)
    power = np.mean(np.array(pvals_alt)  < alpha)
    return type1, power

type1, power = run(n_sims=300, alpha=0.05)
print(f"Estimated Type I error: {type1:.3f}")
print(f"Estimated Power:        {power:.3f}")
```

출력:

```
Estimated Type I error: 0.047
Estimated Power:        0.100
```

제1종 오류가 0.047로 명목 0.05와 어긋나지 않는다(모의실험 표준오차 0.013). 분산비가 6배나 되고 표본크기도 10, 18, 7로 제각각인데도 Welch가 오류율을 지켜 낸다.

검정력 0.100은 처참하다. $G_3$을 2만큼 올렸지만 그 집단의 표준편차가 6이고 표본이 7개뿐이라 신호가 잡음에 묻힌다. **오류율을 지키는 것과 효과를 찾아내는 것은 다른 문제다.**

## 핵심 값

- **제1종 오류율:** $H_0$ 아래 모의실험에서 $p < \alpha$인 비율. 잘 보정된 검정이라면 대략 $\alpha = 0.05$가 나와야 한다.

$$
\widehat{\alpha} = \frac{1}{B} \sum_{b=1}^{B} \mathbf{1}(p_b < \alpha)
$$

- **검정력:** $H_A$ 아래 모의실험에서 $p < \alpha$인 비율.

$$
\widehat{\text{Power}} = \frac{1}{B} \sum_{b=1}^{B} \mathbf{1}(p_b < \alpha)
$$

여기서 $B$는 Monte Carlo 반복 횟수이다.

## 해석

- **제1종 오류 통제:** Welch 분산분석은 심한 이분산($\sigma_3 / \sigma_1 = 6$)에서도 경험적 제1종 오류를 명목 $\alpha = 0.05$ 가까이 유지한다. 반면 고전적 분산분석 $F$-검정은 가장 작은 집단($n_3 = 7$)의 분산이 가장 크므로 이 상황에서 제1종 오류가 부풀려진다.
- **검정력:** $G_3$의 $\Delta = 2$ 이동을 탐지할 검정력은 그 집단의 표본크기와 분산에 달려 있다. $n_3 = 7$, $\sigma_3 = 6$이면 신호 대 잡음 비가 $\Delta / \sigma_3 = 1/3$로 크지 않다. $n_3$이 커지거나 $\sigma_3$이 작아지거나 $\Delta$가 커지면 검정력이 커진다.
- **모의실험의 정밀도:** 반복 $B = 300$에서 추정된 제1종 오류의 표준오차는 대략 $\sqrt{0.05 \times 0.95 / 300} \approx 0.013$이다. 반복을 늘리면 이 폭이 좁아진다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
Monte Carlo 반복이 $B = 300$이고 참 제1종 오류율이 $\alpha = 0.05$일 때 추정된 제1종 오류율의 95% 신뢰구간을 계산하라. 이 구간의 폭을 절반으로 줄이려면 반복이 몇 번 필요한가?

</div>

??? success "풀이"
    추정된 제1종 오류는 표본비율 $\hat{p}$이고 표준오차는 $\text{SE} = \sqrt{\hat{p}(1-\hat{p})/B}$이다. $\hat{p} \approx 0.05$, $B = 300$이면

    $$
    \text{SE} = \sqrt{\frac{0.05 \times 0.95}{300}} \approx 0.0126
    $$

    이다. 95% 신뢰구간은 $0.05 \pm 1.96 \times 0.0126 \approx (0.025,\, 0.075)$이고 폭은 약 $0.049$이다.

    폭을 절반으로 줄이려면 정밀도를 두 배로 해야 하고, 그러려면 반복을 네 배로 늘려야 한다: $B = 4 \times 300 = 1200$.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
가장 작은 집단의 분산이 가장 클 때 고전적 분산분석 $F$-검정의 제1종 오류가 부풀려지는 이유를 설명하라. 가장 작은 집단의 분산이 가장 작으면 어떻게 되는가?

</div>

??? success "풀이"
    고전적 $F$-검정은 모든 집단을 합동하여 $MSW$를 추정한다. 가장 작은 집단의 분산이 가장 크면, 그 집단이 기여하는 관측값이 적기 때문에 합동 추정값이 그 큰 분산을 과소 반영한다. 그러면 $MSW$가 너무 작아져 $F$-통계량이 부풀려지고 기각이 너무 잦아진다(관대한 검정, 제1종 오류 부풀림).

    반대로 가장 작은 집단의 분산이 가장 작으면 합동 $MSW$가 그 집단의 유효 오차분산을 과대추정한다. 그러면 $F$-통계량이 지나치게 보수적이 되어 제1종 오류가 명목 수준 아래로 떨어진다. 검정력은 잃지만 거짓 양성이 늘지는 않는다. 이 비대칭성은 잘 알려진 결과이다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
$G_3$의 이동을 탐지하는 신호 대 잡음 비는 $\Delta/\sigma_3 = 2/6 \approx 0.33$이다. 이 세 집단 설계의 Cohen의 $f$를 계산하고 그 크기를 해석하라.

</div>

??? success "풀이"
    일원배치 분산분석의 Cohen의 $f$는

    $$
    f = \sqrt{\frac{\sum_{i=1}^{k} n_i (\mu_i - \bar{\mu})^2 / N}{\sigma_{\text{within}}^2}}
    $$

    로 정의된다. 대립가설 아래에서 $\mu_1 = \mu_2 = 10$, $\mu_3 = 12$이고 $n_1 = 10$, $n_2 = 18$, $n_3 = 7$, $N = 35$이므로

    $$
    \bar{\mu}_w = \frac{10 \times 10 + 18 \times 10 + 7 \times 12}{35} = \frac{364}{35} = 10.4
    $$

    $$
    \sum n_i(\mu_i - \bar{\mu}_w)^2 = 10(10 - 10.4)^2 + 18(10 - 10.4)^2 + 7(12 - 10.4)^2
    $$

    $$
    = 10(0.16) + 18(0.16) + 7(2.56) = 1.6 + 2.88 + 17.92 = 22.4
    $$

    이다. 분산이 서로 다르므로 합동분산 추정값을 쓴다: $\sigma_{\text{pool}}^2 = (9 \times 1 + 17 \times 9 + 6 \times 36)/32 = (9 + 153 + 216)/32 = 378/32 = 11.8125$.

    $$
    f = \sqrt{\frac{22.4 / 35}{11.8125}} = \sqrt{\frac{0.64}{11.8125}} = \sqrt{0.0542} \approx 0.233
    $$

    Cohen의 관례에서 $f = 0.10$은 작음, $f = 0.25$는 중간, $f = 0.40$은 큼이다. 이 효과크기($f \approx 0.23$)는 작음과 중간 사이이며, 모의실험에서 관측되는 중간 정도의 검정력을 설명해 준다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
세 집단의 분산을 모두 같게($\sigma_i = 3$) 하되 표본크기는 $n = (10, 18, 7)$로 불균형하게 유지하도록 모의실험 설계를 고쳐라. 고전적 분산분석과 Welch 분산분석 중 어느 쪽의 검정력이 높을지 예측하고 이유를 설명하라.

</div>

??? success "풀이"
    분산이 같으면 고전적 분산분석과 Welch 분산분석 모두 명목 수준에서 제1종 오류를 통제한다. 다만 고전적 분산분석은 정확한 $F_{k-1, N-k}$ 분포를 쓰는 반면 Welch 분산분석은 유효 자유도가 줄어든 근사 기준분포를 쓰므로, 고전적 쪽의 검정력이 약간 높다.

    Welch 분산분석은 분산을 따로 추정하고 자유도를 조정하는 대가로 약간의 검정력 손실을 치른다. 이 손실은 이분산에 대한 로버스트성의 "보험료"이다. 등분산이 성립하면 이 보험이 불필요하므로 고전적 검정이 (약간) 더 효율적이다. 실무에서 표본크기가 어느 정도 되면 검정력 차이가 작아 Welch 분산분석을 기본값으로 삼을 만하다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
등분산이고 표본크기가 같은 $H_0$ 아래에서 Welch의 $F_W$ 통계량이 고전적 분산분석 $F$-통계량으로 환원됨을 증명하라.

</div>

??? success "풀이"
    분산이 $\sigma_i^2 = \sigma^2$으로 같고 표본크기가 모든 $i$에서 $n_i = n$으로 같은 $H_0$ 아래에서 가중치는 $w_i = n / s_i^2$이다. 모든 $s_i^2$이 $\sigma^2$으로 수렴하면 가중치가 같아진다: 모든 $i$에서 $w_i = n / \sigma^2$.

    가중 전체평균은 $\tilde{y} = \sum w_i \bar{y}_i / \sum w_j = \bar{y}_{\cdot\cdot}$, 즉 가중하지 않은 전체평균이 된다.

    $F_W$의 분자는

    $$
    \frac{1}{k-1}\sum_{i=1}^{k} \frac{n}{\sigma^2} (\bar{y}_i - \bar{y}_{\cdot\cdot})^2 = \frac{n}{(k-1)\sigma^2} \sum_{i=1}^{k} (\bar{y}_i - \bar{y}_{\cdot\cdot})^2 = \frac{MSB}{\sigma^2}
    $$

    이 된다.

    분모의 보정항은 $\sum (1 - w_i / \sum w_j)^2 / (n_i - 1)$을 포함한다. 가중치가 같으면 $w_i / \sum w_j = 1/k$이므로 각 항이 $(1 - 1/k)^2 / (n-1)$이다. 그 합은 $k(k-1)^2/[k^2(n-1)]$이다. 전체 분모는 $1 + 2(k-2)/(k^2-1) \times (k-1)^2/[k(n-1)]$로 단순해지고 $n \to \infty$에서 1로 간다. 분산이 정확히 같은 경우 Welch 통계량은 고전적 $F$-통계량인 $MSB/MSW$로 단순해진다. $\square$

---

## 정리하며

웰치 분산분석의 이득을 **모의실험으로 직접 재었다.**

- **표준 $F$ 는 이분산 + 불균형에서 무너진다.** **작은 집단에 큰 분산**이 붙으면 제1종 오류율이 명목 $5\%$ 를 크게 넘고, 반대 조합에서는 지나치게 보수적이 된다.
- **웰치는 명목 수준을 지킨다.** 분산비가 커도 제1종 오류율이 $\alpha$ 근처에 머문다.
- **등분산일 때 웰치의 손해는 작다.** 자유도를 조금 잃을 뿐이며, **보험료가 싸다**는 8장·9장의 결론이 분산분석에서도 반복된다.
- **실무 권고가 단순해진다.** 등분산 검정으로 방법을 고르는 2단계 절차 대신 **웰치를 기본으로 쓴다.**
- **모의실험이 이 권고의 근거다.** 이론적 논증보다 재어 본 오류율이 설득력 있다.

다음 절 **이원배치 Welch 분산분석 (로버스트 HC3)** 으로 넘어간다.
