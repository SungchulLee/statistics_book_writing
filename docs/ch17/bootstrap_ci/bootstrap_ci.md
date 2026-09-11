# 붓스트랩 신뢰구간 방법 (코드)

## 개요

붓스트랩 신뢰구간은 분포 가정에 기대지 않고 모수 추정의 불확실성을 정량화하는 방법이다. 이 페이지에서는 세 가지 붓스트랩 신뢰구간 방법 — 백분위수, 기본(역백분위수), BCa(편향보정 가속) — 을 고전적 구간이 신뢰하기 어려울 수 있는 비정규(Poisson) 표본에 적용한다. 각 방법은 경험적 붓스트랩 분포를 서로 다른 방식으로 활용하며, 단순함과 정확성 사이에서 다른 절충을 제공한다.

## 붓스트랩 원리

관측 표본 $x_1, x_2, \ldots, x_n$이 주어졌을 때, 자료에서 **복원추출**로 반복 재표집하여 통계량 $\hat\theta = T(x_1, \ldots, x_n)$의 표집분포를 근사한다. 각 붓스트랩 반복 $\hat\theta^{*(b)}$($b = 1, \ldots, B$)는 원래 관측들로부터 균등하게 뽑은 크기 $n$의 재표본에서 계산된다.

## 백분위수법

백분위수법은 붓스트랩 분포의 분위수에서 신뢰한계를 직접 읽는다. $100(1-\alpha)$% 신뢰구간은

$$
\text{CI}_{\text{pct}} = \bigl[\hat\theta^*_{\alpha/2},\;\hat\theta^*_{1 - \alpha/2}\bigr]
$$

이다. 여기서 $\hat\theta^*_q$는 붓스트랩 분포의 $q$번째 분위수이다.

```python
def bootstrap_percentile_ci(data, statistic, n_boot=10_000, alpha=0.05, rng=None):
    """Percentile bootstrap CI."""
    rng = rng or np.random.default_rng(0)
    n = len(data)
    boot_stats = np.array([
        statistic(data[rng.integers(0, n, n)])
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_stats, 100 * alpha / 2)
    hi = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return lo, hi, boot_stats
```

단순하고 직관적이지만 붓스트랩 분포가 편향되거나 치우쳐 있으면 포함확률이 명목값에 못 미칠 수 있다.

## 기본(역백분위수)법

기본법은 붓스트랩 분포로 $\hat\theta - \theta$의 산포를 추정한 뒤 구간을 뒤집는다. $\hat\theta$를 표본통계량이라 할 때

$$
\text{CI}_{\text{basic}} = \bigl[2\hat\theta - \hat\theta^*_{1 - \alpha/2},\;2\hat\theta - \hat\theta^*_{\alpha/2}\bigr]
$$

```python
def bootstrap_basic_ci(data, statistic, boot_stats, alpha=0.05):
    """Basic (reverse-percentile) bootstrap CI."""
    theta_hat = statistic(data)
    lo = 2 * theta_hat - np.percentile(boot_stats, 100 * (1 - alpha / 2))
    hi = 2 * theta_hat - np.percentile(boot_stats, 100 * alpha / 2)
    return lo, hi
```

핵심 착상은 붓스트랩이 $\hat\theta$를 과대추정한다면 분위수를 $\hat\theta$에 대해 반사시켜 보정한다는 것이다.

## BCa법 (편향보정 가속)

BCa법은 붓스트랩 분포의 **편향**과 **왜도**를 모두 보정한다. 두 개의 보정계수를 도입한다.

- **편향보정** $z_0$: 붓스트랩 분포의 중심이 $\hat\theta$에서 얼마나 떨어져 있는지를 잰다.
- **가속** $a$: $\hat\theta$의 표준오차가 참 모수에 따라 변하는 속도를 담으며, 잭나이프로 추정한다.

조정된 백분위수 수준은

$$
\alpha_1 = \Phi\!\left(z_0 + \frac{z_0 + z_{\alpha/2}}{1 - a(z_0 + z_{\alpha/2})}\right), \qquad \alpha_2 = \Phi\!\left(z_0 + \frac{z_0 + z_{1 - \alpha/2}}{1 - a(z_0 + z_{1 - \alpha/2})}\right)
$$

이다. 여기서 $\Phi$는 표준정규 CDF, $z_q = \Phi^{-1}(q)$이며

$$
z_0 = \Phi^{-1}\!\left(\frac{1}{B}\sum_{b=1}^{B}\mathbf{1}(\hat\theta^{*(b)} < \hat\theta)\right)
$$

$$
a = \frac{\sum_{i=1}^{n}(\bar\theta_{(\cdot)} - \hat\theta_{(i)})^3}{6\left[\sum_{i=1}^{n}(\bar\theta_{(\cdot)} - \hat\theta_{(i)})^2\right]^{3/2}}
$$

이다. $\hat\theta_{(i)}$는 관측 $i$를 뺀 잭나이프 반복값이고 $\bar\theta_{(\cdot)}$는 잭나이프 반복값들의 평균이다.

```python
def bootstrap_bca_ci(data, statistic, boot_stats, alpha=0.05):
    """BCa (bias-corrected and accelerated) bootstrap CI."""
    n = len(data)
    theta_hat = statistic(data)

    # Bias correction factor z0
    z0 = stats.norm.ppf(np.mean(boot_stats < theta_hat))

    # Acceleration factor a -- jackknife estimate
    jack = np.array([statistic(np.delete(data, i)) for i in range(n)])
    jack_mean = jack.mean()
    a_num = np.sum((jack_mean - jack) ** 3)
    a_den = 6 * np.sum((jack_mean - jack) ** 2) ** 1.5
    a = a_num / a_den if a_den != 0 else 0.0

    # Adjusted percentiles
    z_alpha = stats.norm.ppf(alpha / 2)
    z_1alpha = stats.norm.ppf(1 - alpha / 2)

    p_lo = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
    p_hi = stats.norm.cdf(z0 + (z0 + z_1alpha) / (1 - a * (z0 + z_1alpha)))

    lo = np.percentile(boot_stats, 100 * p_lo)
    hi = np.percentile(boot_stats, 100 * p_hi)
    return lo, hi, z0, a
```

## Poisson 자료에 적용하기

참 비율 $\lambda = 3.5$인 Poisson 분포에서 $n = 80$개를 뽑는다. Poisson 분포는 이산이고 오른쪽으로 치우쳐 있어 붓스트랩 방법의 좋은 시험대이다.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
data = stats.poisson.rvs(3.5, size=80, random_state=42)
print(data.mean())          # 3.425

lo_p, hi_p, boots = bootstrap_percentile_ci(data, np.mean, rng=rng)
lo_b, hi_b = bootstrap_basic_ci(data, np.mean, boots)
lo_bca, hi_bca, z0, a = bootstrap_bca_ci(data, np.mean, boots)
print(z0, a)                # -0.0266  0.0126

for name, (lo, hi) in [("백분위수", (lo_p, hi_p)), ("기본", (lo_b, hi_b)),
                       ("BCa", (lo_bca, hi_bca))]:
    print(f"{name:>5}: [{lo:.4f}, {hi:.4f}]  폭 {hi - lo:.4f}")
```

출력:

```
3.425
-0.026573386823392654 0.012573560456423716
 백분위수: [3.0375, 3.8250]  폭 0.7875
   기본: [3.0250, 3.8125]  폭 0.7875
  BCa: [3.0375, 3.8250]  폭 0.7875
```

| 방법 | 하한 | 상한 | 폭 |
|---|---|---|---|
| 백분위수 | 3.0375 | 3.8250 | 0.7875 |
| 기본 | 3.0250 | 3.8125 | 0.7875 |
| BCa | 3.0375 | 3.8250 | 0.7875 |

세 구간이 거의 같다. $n = 80$으로 표본이 어느 정도 크고 표본평균이 잘 행동하기 때문이다. $z_0 = -0.027$과 $a = 0.013$이 모두 $0$에 가까워 BCa 보정이 사실상 작동하지 않았다.

**기본법이 백분위수법보다 정확히 $0.0125$만큼 왼쪽으로 옮겨져 있다.** 이는 붓스트랩 분포의 중심 $3.4375$가 $\hat\theta = 3.425$보다 $0.0125$ 크기 때문이다. 두 구간의 **폭은 항상 같다**. 기본법은 위치만 반사할 뿐 폭을 바꾸지 않는다.

## 해석

- **백분위수법**은 가장 단순하다. 붓스트랩 분포가 대략 대칭이고 편향이 없을 때 잘 작동한다.
- **기본법**은 분위수를 $\hat\theta$에 대해 반사시켜 위치 편향을 보정하지만 왜도는 다루지 않는다.
- **BCa**는 편향과 가속(왜도)을 모두 보정하므로 셋 중 포함확률이 가장 믿을 만하다. 대가는 잭나이프에 드는 추가 계산이다.

붓스트랩 분포가 대칭이고 $\hat\theta$를 중심으로 하면 세 방법이 거의 같은 결과를 준다. 차이는 작은 표본, 치우친 통계량(중앙값, 분산), 두꺼운 꼬리 자료에서 드러난다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** 비율 $\lambda = 1$인 지수분포에서 크기 $n = 30$인 표본을 생성하고 표본평균에 대한 세 가지 붓스트랩 $95$% 신뢰구간을 모두 계산하라. 어느 구간이 가장 넓은가? 왜 그런가?

</div>

??? success "풀이"

    ```python
    import numpy as np
    rng = np.random.default_rng(0)
    data = rng.exponential(scale=1.0, size=30)
    print(data.mean())      # 1.1845

    lo_p, hi_p, boots = bootstrap_percentile_ci(data, np.mean, rng=rng)
    lo_b, hi_b = bootstrap_basic_ci(data, np.mean, boots)
    lo_bca, hi_bca, z0, a = bootstrap_bca_ci(data, np.mean, boots)
    ```

    출력:

    ```
    1.1844536180110892
    ```

    | 방법 | 구간 | 폭 |
    |:---|:---|---:|
    | 백분위수 | $[0.7633,\ 1.6777]$ | 0.9145 |
    | 기본 | $[0.6912,\ 1.6056]$ | 0.9145 |
    | BCa | $[0.8264,\ 1.8135]$ | **0.9871** |

    ($z_0 = 0.068$, $a = 0.060$)

    **BCa가 가장 넓다**($0.9871$ 대 $0.9145$, $8$% 넓다).

    이유는 지수분포의 양의 왜도이다. $a = 0.060 > 0$이므로 조정된 백분위수 수준이 **둘 다 위로** 이동한다.

    $$
    \alpha_1 = \Phi\!\left(0.068 + \frac{0.068 - 1.96}{1 - 0.060(0.068-1.96)}\right) = \Phi(-1.633) = 0.051
    $$

    $$
    \alpha_2 = \Phi\!\left(0.068 + \frac{0.068 + 1.96}{1 - 0.060(0.068+1.96)}\right) = \Phi(2.383) = 0.991
    $$

    즉 $[2.5\%,\ 97.5\%]$ 대신 $[5.1\%,\ 99.1\%]$를 읽는다. 상한이 훨씬 오른쪽으로 밀리는 것이 폭 증가의 주된 원인이다.

    **백분위수와 기본의 폭이 정확히 같다는 점에 주목하라.** 이는 우연이 아니라 항상 성립한다.

    $$
    (2\hat\theta - \hat\theta^*_{\alpha/2}) - (2\hat\theta - \hat\theta^*_{1-\alpha/2}) = \hat\theta^*_{1-\alpha/2} - \hat\theta^*_{\alpha/2}
    $$

    기본법은 구간을 $\hat\theta$에 대해 **반사**할 뿐 폭을 바꾸지 않는다. 두 방법의 차이는 오직 위치이다. 여기서는 $0.072$만큼 왼쪽으로 옮겨져 있는데, 이는 붓스트랩 평균이 $\hat\theta$보다 $0.036$ 크기 때문이다($2 \times 0.036 = 0.072$).

<div class="drillbox" markdown>

**연습문제 2.** 붓스트랩 분포가 $\hat\theta$에 대해 정확히 대칭이고 편향이 없으면($z_0 = 0$, $a = 0$) BCa 구간이 백분위수 구간으로 환원됨을 보여라.

</div>

??? success "풀이"

    $z_0 = 0$이고 $a = 0$이면 조정된 백분위수 수준은

    $$
    \alpha_1 = \Phi\!\left(0 + \frac{0 + z_{\alpha/2}}{1 - 0}\right) = \Phi(z_{\alpha/2}) = \frac{\alpha}{2}
    $$

    $$
    \alpha_2 = \Phi\!\left(0 + \frac{0 + z_{1-\alpha/2}}{1 - 0}\right) = \Phi(z_{1-\alpha/2}) = 1 - \frac{\alpha}{2}
    $$

    가 된다. 이는 정확히 백분위수법이 쓰는 분위수 수준이다. 따라서

    $$
    \text{CI}_{\text{BCa}} = \bigl[\hat\theta^*_{\alpha/2},\;\hat\theta^*_{1 - \alpha/2}\bigr] = \text{CI}_{\text{pct}}
    $$

    $\square$

    Poisson 예제가 이 성질의 근사적 확인이다. $z_0 = -0.027$, $a = 0.013$으로 둘 다 $0$에 가까워 BCa 구간이 백분위수 구간과 소수 넷째 자리까지 일치했다.

<div class="drillbox" markdown>

**연습문제 3.** 기본 붓스트랩 구간이 적절한 분위수 $q$에 대해 $\hat\theta \pm (\hat\theta - \hat\theta^*_q)$ 형태로 쓰일 수 있음을 보이고, 이 방법을 "반사"법이라 부르는 이유를 기하적으로 설명하라.

</div>

??? success "풀이"

    기본 구간은

    $$
    \text{CI}_{\text{basic}} = \bigl[2\hat\theta - \hat\theta^*_{1-\alpha/2},\; 2\hat\theta - \hat\theta^*_{\alpha/2}\bigr]
    $$

    이다. 하한을 다시 쓰면

    $$
    2\hat\theta - \hat\theta^*_{1-\alpha/2} = \hat\theta - (\hat\theta^*_{1-\alpha/2} - \hat\theta)
    $$

    상한은

    $$
    2\hat\theta - \hat\theta^*_{\alpha/2} = \hat\theta + (\hat\theta - \hat\theta^*_{\alpha/2})
    $$

    이다.

    기하적으로, 붓스트랩 분포는 $\hat\theta^*$가 $\hat\theta$ 주위에서 어떻게 변하는지를 정량화한다. 기본법은 $\hat\theta$가 $\theta$ 주위에서 같은 방식으로 변한다고 가정하므로, 붓스트랩 분위수를 $\hat\theta$를 지나는 축에 대해 *반사*하여 $\theta$의 신뢰한계를 얻는다. "반사"라는 이름은 이 거울상 변환에서 왔다. $\square$

    **왜 반사가 필요한가.** 붓스트랩 분포가 오른쪽으로 치우쳐 있다고 하자. 그러면 $\hat\theta$도 $\theta$에 대해 오른쪽으로 치우쳐 있을 것이라 추론한다. 즉 $\hat\theta$가 $\theta$를 과대추정하는 경향이 있다.

    이때 옳은 대응은 신뢰구간을 **왼쪽으로** 옮기는 것이다. 백분위수법은 이를 하지 않고 붓스트랩 분포를 그대로 읽으므로 오른쪽으로 치우친 구간을 준다.

    !!! note "그런데 백분위수법이 더 나은 경우도 있다"
        위 논증은 설득력 있어 보이지만 항상 옳지는 않다. [백분위수법](./percentile.md) 연습문제 4에서, 치우친 추정량에 대해 백분위수 구간의 포함확률이 $0.932$인 반면 기본 구간은 $0.732$였다.

        이유는 백분위수법이 **변환 불변**이기 때문이다. $\hat\theta$의 백분위수 구간에 단조변환 $g$를 적용하면 정확히 $g(\hat\theta)$의 백분위수 구간이 된다. 기본법은 이 성질을 갖지 않는다.

<div class="drillbox" markdown>

**연습문제 4.** 포함확률 모의실험을 수행하라. $\chi^2(3)$ 분포에서 크기 $n = 20$인 표본을 $2{,}000$개 뽑는다. 각 표본에서 평균에 대한 백분위수·기본·BCa $95$% 신뢰구간을 계산하고, 참 평균 $\mu = 3$을 포함하는 비율을 보고하라. 어느 방법이 명목 $95$%에 가장 가까운가?

</div>

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats
    rng = np.random.default_rng(42)

    def run(n=20, M=2000, B=2000, df=3):
        cp = cbs = cb = ct = 0
        zl, zu = stats.norm.ppf(0.025), stats.norm.ppf(0.975)
        for _ in range(M):
            d = rng.chisquare(df, n); th = d.mean()
            b = d[rng.integers(0, n, (B, n))].mean(1)
            lo, hi = np.percentile(b, [2.5, 97.5])
            cp  += lo <= df <= hi
            cbs += (2*th - hi) <= df <= (2*th - lo)
            z0 = stats.norm.ppf(np.clip((b < th).mean(), 1e-6, 1-1e-6))
            jk = (d.sum() - d) / (n - 1); jm = jk.mean()        # 벡터화된 잭나이프
            num = ((jm-jk)**3).sum(); den = 6*(((jm-jk)**2).sum())**1.5
            a = num/den if den > 0 else 0.0
            p1 = stats.norm.cdf(z0 + (z0+zl)/(1 - a*(z0+zl)))
            p2 = stats.norm.cdf(z0 + (z0+zu)/(1 - a*(z0+zu)))
            l2, h2 = np.percentile(b, [100*p1, 100*p2]); cb += l2 <= df <= h2
            ct += stats.ttest_1samp(d, df).pvalue >= 0.05
        return cp/M, cbs/M, cb/M, ct/M
    ```

    | 방법 | $n = 20$ | $n = 50$ |
    |:---|---:|---:|
    | 백분위수 | 0.906 | 0.930 |
    | 기본 | 0.892 | 0.924 |
    | BCa | **0.914** | **0.932** |
    | $t$ 구간(비교용) | 0.928 | 0.937 |

    **BCa가 세 붓스트랩 방법 중 가장 낫다**($0.914$ 대 $0.906$, $0.892$). $\chi^2(3)$의 왜도가 $\sqrt{8/3} = 1.63$으로 크기 때문에 왜도 보정이 실제로 도움이 된다.

    **그러나 개선폭은 작다.** $0.906 \to 0.914$로 $0.8$%p이다. 흔히 "BCa가 훨씬 낫다"고 서술되지만 이 상황에서는 그렇지 않다.

    **셋 다 명목값에 못 미친다.** $n = 20$에서 $0.89$--$0.91$이고, $n = 50$에서도 $0.92$--$0.93$이다. 심지어 $t$ 구간도 $0.928$에 그친다. **자료 자체가 치우쳐 있어 $n = 20$으로는 어떤 방법도 $0.95$를 달성하지 못한다.**

    **기본법이 가장 나쁘다**($0.892$). 반사가 여기서는 잘못된 방향으로 작용한다. 연습문제 3의 주석에서 언급한 백분위수법의 변환 불변성이 치우친 자료에서 우위를 준다.

    !!! tip "이 결과를 어떻게 읽어야 하는가"
        "BCa를 쓰라"가 결론이 아니다. 더 정확한 결론은 세 가지이다.

        1. **BCa가 조금 낫지만 마법은 아니다.** 잭나이프에 $n$번의 추가 계산이 드는 것에 비해 $1$%p 개선이 항상 값진 것은 아니다.
        2. **표본크기가 근본 제약이다.** $n = 20$, 왜도 $1.63$이면 무엇을 해도 $0.91$ 근처이다. $n$을 늘리는 것이 방법을 바꾸는 것보다 효과가 크다.
        3. **붓스트랩-$t$를 시도해 보라.** 이 표에 없는 네 번째 방법이며, 치우친 자료에서 종종 가장 잘 작동한다([붓스트랩-$t$](./bootstrap_t.md) 참조). 대가는 이중 붓스트랩 또는 표준오차 공식이다.

<div class="drillbox" markdown>

**연습문제 5.** 잭나이프 가속계수 $a$는 잭나이프 값들의 3차 적률을 포함한다. 표집분포가 양으로 치우친 통계량에서 $a > 0$이 되는 이유를 직관적으로 설명하고, 이것이 BCa 구간을 백분위수 구간에 대해 어떻게 이동시키는지 서술하라.

</div>

??? success "풀이"

    가속계수는

    $$
    a = \frac{\sum_{i=1}^{n}(\bar\theta_{(\cdot)} - \hat\theta_{(i)})^3}{6\left[\sum_{i=1}^{n}(\bar\theta_{(\cdot)} - \hat\theta_{(i)})^2\right]^{3/2}}
    $$

    이다. 분자는 잭나이프 값들의 (정규화되지 않은) 3차 중심적률이다.

    $\hat\theta$의 표집분포가 양으로 치우쳐 있으면, $\hat\theta$를 낮추는 관측을 제거했을 때의 잭나이프 값들은 평균 아래에 몰리는 반면, $\hat\theta$를 높이는 관측(큰 이상값)을 제거했을 때는 소수의 잭나이프 값이 평균보다 훨씬 위에 놓인다. 이 비대칭이 양의 3차 적률을 만들어 $a > 0$이 된다.

    $a > 0$이면 조정된 백분위수 수준이 위로 이동한다. $\alpha_1$과 $\alpha_2$가 모두 커진다.

    - $z_0 + z_{\alpha/2} < 0$이므로 분모 $1 - a(z_0 + z_{\alpha/2}) > 1$이고, 비의 절댓값이 작아져 $\alpha_1$이 커진다.
    - $z_0 + z_{1-\alpha/2} > 0$이므로 분모 $1 - a(z_0 + z_{1-\alpha/2}) < 1$이고, 비가 커져 $\alpha_2$가 커진다.

    결과적으로 BCa 구간이 백분위수 구간에 비해 오른쪽으로 이동하면서 상단 꼬리가 넓어지고 하단 꼬리가 좁아진다. 양으로 치우친 통계량의 오른쪽 꼬리를 더 많이 담아 포함확률이 개선된다. $\square$

    **연습문제 1이 이를 수치로 보여준다.** 지수 자료에서 $a = 0.060$이고 $[2.5\%, 97.5\%] \to [5.1\%, 99.1\%]$로 이동했다. 하한이 $0.7633 \to 0.8264$로 올라가고 상한이 $1.6777 \to 1.8135$로 더 크게 올라갔다.

    !!! warning "$a$가 항상 신뢰할 만하지는 않다"
        잭나이프 추정값 $a$는 $n$개의 값에만 기반하므로 작은 $n$에서 불안정하다.

        더 근본적인 문제는 **매끄럽지 않은 통계량에서 잭나이프가 실패한다**는 것이다. 중앙값의 경우 관측 하나를 빼는 것이 중앙값을 거의 바꾸지 않거나(짝수 개에서 가운데가 아닌 관측을 뺄 때) 정확히 인접 관측으로 옮긴다. 그 결과 잭나이프 값들이 소수의 값에 몰려 3차 적률이 사실상 $0$이 된다.

        [BCa](./bca.md) 연습문제 3에서 확인했듯, 중앙값에 대한 $\hat{a}$는 평균 $0.00001$, 표준편차 $0.00089$로 **안정적으로 0**이다. 즉 왜도가 있어도 탐지하지 못한다. 매끄럽지 않은 통계량에는 BCa 대신 다른 방법을 고려해야 한다.
