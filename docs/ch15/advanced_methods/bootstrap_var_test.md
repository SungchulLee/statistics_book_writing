# 붓스트랩 분산 검정

## 개요

이 페이지는 두 집단의 분산이 같은지 검정하는 비모수 붓스트랩 접근을 시연한다. ($F$ 검정처럼) 분포 가정에 기대는 대신 관측 자료에서 재표집하여 분산비의 표집분포를 만든다. 붓스트랩은 신뢰구간과 근사적 $p$값을 모두 제공하므로 정규성이 의심스러울 때 유연한 대안이 된다.

---

## 분산비 통계량

독립인 두 표본 $\mathbf{x}_1 = (x_{11}, \ldots, x_{1,n_1})$과 $\mathbf{x}_2 = (x_{21}, \ldots, x_{2,n_2})$이 주어졌을 때 관심 통계량은 표본분산의 비이다.

$$
\hat\theta = \frac{S_1^2}{S_2^2}, \qquad S_i^2 = \frac{1}{n_i - 1}\sum_{j=1}^{n_i}(x_{ij} - \bar{x}_i)^2
$$

귀무가설 $H_0\colon \sigma_1^2 = \sigma_2^2$ 아래에서 참 비율은 $\theta = 1$이다.

---

## 붓스트랩 절차

비모수 붓스트랩은 정규성을 가정하지 않고 $\hat\theta$의 표집분포를 추정한다.

1. **재표집**: 각 붓스트랩 반복 $b = 1, \ldots, B$에 대해 $\mathbf{x}_1$에서 $n_1$개, $\mathbf{x}_2$에서 $n_2$개를 복원추출한다.
2. **계산**: $\hat\theta^{(b)} = S_1^{2(b)} / S_2^{2(b)}$을 계산한다.
3. **로그 변환**: 비는 오른쪽으로 치우쳐 있으므로 대칭성을 위해 로그 척도 $\log\hat\theta^{(b)}$에서 작업한다.
4. **신뢰구간**: 로그 척도의 백분위 신뢰구간은 $(q_{0.025}, q_{0.975})$이다. 지수를 취해 $\theta$의 신뢰구간을 얻는다.
5. **$p$값**: 붓스트랩 분포를 **귀무값 $\theta = 1$**(로그 척도에서 0)과 비교한다.

$$
p = 2 \min\!\Big(\frac{1}{B}\sum_{b=1}^B \mathbf{1}(\log\hat\theta^{(b)} \le 0),\;\; \frac{1}{B}\sum_{b=1}^B \mathbf{1}(\log\hat\theta^{(b)} \ge 0)\Big)
$$

!!! danger "$p$값을 관측 통계량과 비교하면 안 된다"
    문헌에서 다음과 같은 형태를 종종 볼 수 있으나 **완전히 잘못되었다**.

    $$
    p_{\text{잘못}} = 2 \min\!\Big(\tfrac{1}{B}\textstyle\sum_b \mathbf{1}(\log\hat\theta^{(b)} \le \log\hat\theta),\;\; \tfrac{1}{B}\textstyle\sum_b \mathbf{1}(\log\hat\theta^{(b)} \ge \log\hat\theta)\Big)
    $$

    각 집단 안에서 재표집하면 붓스트랩 분포가 **관측값 $\hat\theta$ 주위에 중심**을 갖는다. 그러면 $\log\hat\theta$가 붓스트랩 분포의 중앙값 근처에 있으므로 두 비율이 모두 $\approx 0.5$가 되고, **$p \approx 1$이 항상 나온다**.

    실제로 확인해 보면, 참 표준편차가 $1$과 $3$(분산비 9배)인 정규 자료 $n = 20$에서 이 공식의 $p$값은 200회 반복에서 $0.856$~$1.000$ 범위, 평균 $0.960$이었다. **분산이 9배 다른데도 절대 기각하지 않는다.**

    올바른 비교 대상은 관측값이 아니라 **귀무값 1**이다. 위 5단계의 공식이 그것이며, 이는 백분위 신뢰구간을 뒤집은 것과 동등하다(신뢰구간이 1을 포함하지 않을 때만 $p < \alpha$).

---

## 구현

```python
import numpy as np

def variance_ratio(x1, x2):
    return np.var(x1, ddof=1) / np.var(x2, ddof=1)

def bootstrap_varratio(x1, x2, B=2000, seed=None):
    rng = np.random.default_rng(seed)
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    n1, n2 = len(x1), len(x2)
    stat_obs = variance_ratio(x1, x2)

    boots = np.empty(B)
    for b in range(B):
        b1 = rng.choice(x1, size=n1, replace=True)
        b2 = rng.choice(x2, size=n2, replace=True)
        boots[b] = np.log(variance_ratio(b1, b2))

    # Percentile CI on log scale, then exponentiate
    lo, hi = np.percentile(boots, [2.5, 97.5])
    ci = (float(np.exp(lo)), float(np.exp(hi)))

    # Two-sided p-value: compare the bootstrap distribution to the
    # NULL value log(1) = 0, not to the observed statistic
    p_two = 2 * min(np.mean(boots <= 0.0), np.mean(boots >= 0.0))
    p_two = float(min(p_two, 1.0))

    return float(stat_obs), ci, p_two
```

---

## 예제

```python
x1 = np.array([12, 15, 14, 10, 13, 14, 12, 11], dtype=float)
x2 = np.array([22, 25, 20, 18, 24, 23, 19, 21], dtype=float)

theta_hat, ci, p = bootstrap_varratio(x1, x2, B=10000, seed=42)
print(f"Observed ratio: {theta_hat:.4f}")
print(f"95% Bootstrap CI: ({ci[0]:.4f}, {ci[1]:.4f})")
print(f"Bootstrap p-value: {p:.4f}")
```

출력:

```text
Observed ratio: 0.4732
95% Bootstrap CI: (0.1345, 1.6092)
Bootstrap p-value: 0.1892
```

95% 신뢰구간이 1을 포함하므로 등분산 귀무가설을 기각하지 못한다. $p = 0.189$가 이와 일관된 증거의 척도를 제공한다.

(같은 자료에 대한 F 검정의 $p$값은 $0.3448$이다. 붓스트랩이 F 검정보다 작은 $p$값을 냈지만 둘 다 기각하지 않는다.)

---

## 해석

- 붓스트랩은 $S^2$의 분포에 대해 모수적 가정을 하지 않는다. $F$ 검정이 실패하는 비정규, 치우침, 두꺼운 꼬리 자료에서도 타당하다.
- 붓스트랩 $p$값은 근사적이다. $B = 10{,}000$이면 몬테카를로 오차가 $1/\sqrt{B} \approx 0.01$ 규모이다.
- 작은 표본에서는 붓스트랩 분포가 거칠어질 수 있으며, BCa(편향보정·가속) 구간이 더 나을 수 있다.

!!! note "로그 척도는 백분위 구간에는 영향을 주지 않는다"
    3단계에서 로그 변환을 하지만, **백분위 방법은 단조변환에 불변**이므로 로그 척도의 백분위 구간에 지수를 취한 것과 비율 척도에서 직접 계산한 백분위 구간이 **정확히 같다**(연습문제 5에서 증명한다).

    위 $p$값 공식도 마찬가지이다. $\log$가 단조이므로 $\mathbf{1}(\log\hat\theta^{(b)} \le 0)$과 $\mathbf{1}(\hat\theta^{(b)} \le 1)$이 동일하다.

    그렇다면 로그가 왜 유용한가? **백분위 방법이 아닌 다른 방법에서** 유용하다.

    - 정규근사 구간 $\hat\theta \pm 1.96\,\text{SE}$는 비율 척도에서 음수 하한을 낼 수 있지만 로그 척도에서는 그런 문제가 없다.
    - 붓스트랩 $t$ 구간이나 BCa에서 로그 척도의 대칭성이 근사를 개선한다.
    - 붓스트랩 분포를 눈으로 볼 때 로그 척도가 훨씬 읽기 쉽다.

    백분위 방법만 쓴다면 로그 변환은 계산 단계에서 아무것도 바꾸지 않는다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** $\mathcal{N}(0, 1)$에서 크기 30인 표본 둘을 생성하라(등분산). $B = 5000$으로 붓스트랩 분산 검정을 실행하라. 95% 신뢰구간이 1을 포함하는가? 500회 반복하여 포함확률(참 비율 1을 포함하는 구간의 비율)을 추정하라.

</div>

??? success "풀이"

    ```python
    import numpy as np

    def boot_log_ratios(x1, x2, B, rng):
        n1, n2 = len(x1), len(x2)
        i1 = rng.integers(0, n1, (B, n1))
        i2 = rng.integers(0, n2, (B, n2))
        return np.log(x1[i1].var(axis=1, ddof=1) / x2[i2].var(axis=1, ddof=1))

    rng = np.random.default_rng(0)
    covers = 0
    for _ in range(500):
        x1 = rng.normal(0, 1, 30)
        x2 = rng.normal(0, 1, 30)
        bs = boot_log_ratios(x1, x2, 5000, rng)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        if np.exp(lo) <= 1.0 <= np.exp(hi):
            covers += 1

    print(f"Coverage: {covers/500:.3f}")
    ```

    출력:

    ```text
    Coverage: 0.920
    ```

    포함확률이 $0.920$으로 명목값 $0.95$보다 **낮다**. 몬테카를로 오차가 $\sqrt{0.92 \times 0.08/500} = 0.012$이므로 $0.95$와의 차이($0.030$)는 2.5 표준오차로 유의하다.

    **왜 부족한가.** 백분위 붓스트랩 구간은 두 가지 이유로 분산비에 대해 정확하지 않다.

    1. **편향.** 붓스트랩 분산 $S^{*2}$은 원표본의 경험분포에서 계산되므로 $\hat\sigma^2$을 향해 축소되는 경향이 있다.
    2. **꼬리 정보의 부족.** $n = 30$인 표본이 원분포의 꼬리를 충분히 담지 못하므로 붓스트랩 분포가 실제보다 좁아진다.

    **개선 방법.** BCa 구간(연습문제 4)이나 붓스트랩 $t$ 구간이 편향과 왜도를 보정하여 포함확률을 개선한다. 정규성이 성립한다면 물론 정확한 $F$ 기반 구간이 최선이다.

    **실무적 함의.** "붓스트랩은 가정이 없으니 언제나 정확하다"는 통념은 옳지 않다. 붓스트랩도 근사이며, 특히 분산처럼 고차 적률에 의존하는 통계량에서는 수렴이 느리다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** 붓스트랩 분포를 관측 통계량이 아니라 귀무값 1과 비교해야 하는 이유를 설명하고, 수정된 $p$값의 크기를 모의실험으로 확인하라.

</div>

??? success "풀이"

    **왜 귀무값과 비교하는가.** 가설검정의 $p$값은 "귀무가설이 참일 때 관측된 것만큼 극단적인 결과가 나올 확률"이다. 그러려면 **귀무가설 아래의 분포**가 필요하다.

    각 집단 안에서 재표집한 붓스트랩 분포는 귀무가설 분포가 아니라 **$\hat\theta$ 주위의 표집분포 추정값**이다. 이 분포에서 $\hat\theta$가 얼마나 극단적인지 묻는 것은 "표본평균이 표본평균에서 얼마나 떨어져 있는가"를 묻는 것과 같아 언제나 0에 가깝다.

    올바른 논리는 **신뢰구간의 반전**이다. $\theta$의 $100(1-\alpha)\%$ 신뢰구간이 1을 포함하지 않을 때 정확히 $p < \alpha$가 되도록 $p$값을 정의한다. 그것이 본문 5단계의 공식이다.

    ```python
    import numpy as np
    from scipy.stats import f as f_dist

    def p_vs_null(x1, x2, B, rng):
        n1, n2 = len(x1), len(x2)
        i1 = rng.integers(0, n1, (B, n1))
        i2 = rng.integers(0, n2, (B, n2))
        bs = np.log(x1[i1].var(axis=1, ddof=1) / x2[i2].var(axis=1, ddof=1))
        return min(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1.0)

    rng = np.random.default_rng(7)
    for name, gen in [("Normal", lambda n: rng.normal(0, 1, n)),
                      ("Exponential", lambda n: rng.exponential(1, n))]:
        for n in [20, 50]:
            rej = sum(p_vs_null(gen(n), gen(n), 1000, rng) < 0.05
                      for _ in range(2000))
            print(f"{name:12s} n={n}: size = {rej/2000:.4f}")
    ```

    출력:

    ```text
    Normal       n=20: size = 0.0725
    Normal       n=50: size = 0.0610
    Exponential  n=20: size = 0.0915
    Exponential  n=50: size = 0.0905
    ```

    | 자료 | $n=20$ | $n=50$ |
    |---|---|---|
    | Normal | 0.073 | 0.061 |
    | Exponential | 0.092 | 0.091 |

    수정된 $p$값은 실제로 작동한다(잘못된 공식은 크기가 사실상 0이었다). 다만 다소 자유주의적이다. 연습문제 1에서 본 포함확률 부족($0.920$)의 다른 얼굴이다.

    **F 검정과의 비교.** 같은 지수분포 $n = 20$ 설정에서 F 검정의 크기는 $0.272$이다. 붓스트랩의 $0.092$가 훨씬 낫지만 명목값의 두 배이므로 완벽하지는 않다.

    **검정력도 확인해 두자.** 정규 자료 $n = 30$, 표준편차 $1$ 대 $2$(분산비 4)에서 이 검정의 검정력은 $0.946$이다. 크기가 다소 부풀려진 만큼 검정력도 높게 나오므로, 엄밀한 비교에는 크기 보정이 필요하다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** 지수분포(강하게 오른쪽으로 치우침)에서 뽑은 자료로 붓스트랩 분산 검정과 고전적 $F$ 검정을 비교하라. $\text{Exp}(1)$에서 $n_1 = n_2 = 20$을 생성하고(등분산) 두 검정을 $\alpha = 0.05$에서 2,000회 실행하여 각각의 거짓 양성률을 보고하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    from scipy.stats import f as f_dist

    def variance_ratio(x1, x2):
        return np.var(x1, ddof=1) / np.var(x2, ddof=1)

    def bootstrap_pvalue(x1, x2, B, rng):
        """Two-sided bootstrap p-value: compare to the NULL value 1."""
        n1, n2 = len(x1), len(x2)
        i1 = rng.integers(0, n1, (B, n1))
        i2 = rng.integers(0, n2, (B, n2))
        bs = np.log(x1[i1].var(axis=1, ddof=1) / x2[i2].var(axis=1, ddof=1))
        return min(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1.0)

    rng = np.random.default_rng(42)
    rej_f, rej_boot = 0, 0
    n_sims = 2000

    for _ in range(n_sims):
        x1 = rng.exponential(1, 20)
        x2 = rng.exponential(1, 20)
        F = variance_ratio(x1, x2)
        p_f = 2 * min(f_dist.cdf(F, 19, 19), f_dist.sf(F, 19, 19))
        if p_f < 0.05:
            rej_f += 1
        if bootstrap_pvalue(x1, x2, 1000, rng) < 0.05:
            rej_boot += 1

    print(f"F-test false positive rate:    {rej_f/n_sims:.4f}")
    print(f"Bootstrap false positive rate: {rej_boot/n_sims:.4f}")
    ```

    출력:

    ```text
    F-test false positive rate:    0.2695
    Bootstrap false positive rate: 0.0890
    ```

    $F$ 검정의 거짓 양성률이 $0.270$으로 명목값의 **다섯 배**이다. 지수분포가 정규성을 심하게 위반하기 때문이다.

    붓스트랩 검정은 $0.089$로 훨씬 낫지만 **여전히 명목값의 두 배**이다. 어떤 특정한 분포 형태도 가정하지 않지만, 경험분포가 참 분포의 근사라는 가정은 여전히 필요하다. $n = 20$짜리 지수 표본은 오른쪽 꼬리를 제대로 담지 못한다.

    **비교 기준을 하나 더 두자.** 같은 조건에서 Brown-Forsythe 검정의 크기는 $0.048$이다(15.8절 [로버스트 분산 검정 비교](../robust_tests/robust_tests_comparison.md)). **치우친 자료에서는 붓스트랩보다 Brown-Forsythe가 낫다.**

    붓스트랩의 강점은 (1) 분산비 자체의 신뢰구간을 준다는 점과 (2) 임의의 통계량에 적용할 수 있다는 점이지, 크기 조절의 정확성이 아니다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** 백분위 붓스트랩 구간은 가장 단순한 변형이다. 로그 분산비에 대한 **BCa(편향보정·가속)** 붓스트랩 구간을 구현하라. `x1 = [12, 15, 14, 10, 13, 14, 12, 11]`과 `x2 = [22, 25, 20, 18, 24, 23, 19, 21]`에 두 방법을 적용하고 결과 구간을 비교하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats as sp_stats

    def variance_ratio(x1, x2):
        return np.var(x1, ddof=1) / np.var(x2, ddof=1)

    x1 = np.array([12, 15, 14, 10, 13, 14, 12, 11], dtype=float)
    x2 = np.array([22, 25, 20, 18, 24, 23, 19, 21], dtype=float)
    rng = np.random.default_rng(42)
    B = 10000
    n1, n2 = len(x1), len(x2)

    log_obs = np.log(variance_ratio(x1, x2))
    boots = np.array([np.log(variance_ratio(
        rng.choice(x1, n1, replace=True),
        rng.choice(x2, n2, replace=True))) for _ in range(B)])

    # Percentile
    lo_p, hi_p = np.percentile(boots, [2.5, 97.5])

    # BCa: bias correction
    z0 = sp_stats.norm.ppf(np.mean(boots < log_obs))

    # Acceleration (jackknife on group 1)
    jk = np.empty(n1)
    for i in range(n1):
        x1_jk = np.delete(x1, i)
        jk[i] = np.log(np.var(x1_jk, ddof=1) / np.var(x2, ddof=1))
    jk_mean = jk.mean()
    a_hat = np.sum((jk_mean - jk)**3) / (6 * np.sum((jk_mean - jk)**2)**1.5)

    z_lo, z_hi = sp_stats.norm.ppf(0.025), sp_stats.norm.ppf(0.975)
    alpha1 = sp_stats.norm.cdf(z0 + (z0 + z_lo) / (1 - a_hat * (z0 + z_lo)))
    alpha2 = sp_stats.norm.cdf(z0 + (z0 + z_hi) / (1 - a_hat * (z0 + z_hi)))
    lo_bca, hi_bca = np.percentile(boots, [100*alpha1, 100*alpha2])

    print(f"z0 = {z0:.4f}, a_hat = {a_hat:.4f}")
    print(f"BCa percentiles: {100*alpha1:.2f}%, {100*alpha2:.2f}%")
    print(f"Percentile CI (ratio): ({np.exp(lo_p):.3f}, {np.exp(hi_p):.3f})")
    print(f"BCa CI (ratio):        ({np.exp(lo_bca):.3f}, {np.exp(hi_bca):.3f})")
    ```

    출력:

    ```text
    z0 = 0.0175, a_hat = 0.0560
    BCa percentiles: 4.14%, 98.75%
    Percentile CI (ratio): (0.135, 1.609)
    BCa CI (ratio):        (0.160, 2.117)
    ```

    BCa 구간이 백분위 구간보다 **오른쪽으로 밀려 있다**. 하한이 $0.135 \to 0.160$, 상한이 $1.609 \to 2.117$이다.

    | | 하한 | 상한 | 폭(로그 척도) |
    |---|---|---|---|
    | 백분위 | 0.135 | 1.609 | 2.482 |
    | BCa | 0.160 | 2.117 | 2.584 |

    **보정의 내역.**

    - **편향 보정** $z_0 = 0.0175$가 매우 작다. 붓스트랩 분포의 중앙값이 $\hat\theta$와 거의 일치한다는 뜻이다.
    - **가속** $\hat{a} = 0.056$이 양수이다. 잭나이프 값들이 왼쪽으로 치우쳐 있어 통계량의 분산이 $\theta$가 커질수록 증가함을 시사한다.

    두 보정이 결합하여 백분위 지점을 $2.5\% \to 4.14\%$, $97.5\% \to 98.75\%$로 옮긴다.

    BCa 구간은 편향(붓스트랩 분포의 중앙값이 $\hat\theta$와 다를 수 있다)과 왜도(가속 인자 $\hat{a}$)를 모두 보정한다. 작은 표본에서 BCa와 백분위 구간의 차이가 눈에 띄게 클 수 있고, 일반적으로 BCa의 포함확률이 더 좋다.

    !!! warning "이 BCa 구현은 불완전하다"
        가속 인자를 **집단 1에 대해서만** 잭나이프로 계산했다. 두 표본 문제에서는 두 집단 모두에 대해 잭나이프를 수행하고 결합해야 이론적으로 옳다.

        엄밀한 구현은 $n_1 + n_2$개의 잭나이프 값을 모두 계산한 뒤

        $$
        \hat{a} = \frac{\sum_i (\bar{J} - J_i)^3}{6\left[\sum_i (\bar{J} - J_i)^2\right]^{3/2}}
        $$

        을 쓴다. 실무에서는 `scipy.stats.bootstrap(..., method='BCa')`를 쓰는 편이 안전하다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** 붓스트랩 백분위 구간이 변환 불변임을 증명하라. 곧 $(L, U)$가 $\theta$의 백분위 구간이면 임의의 단조증가함수 $g$에 대해 $(g(L), g(U))$가 $g(\theta)$의 백분위 구간임을 보여라.

</div>

??? success "풀이"

    $\hat\theta_1^*, \ldots, \hat\theta_B^*$을 붓스트랩 복제값이라 하자. 이들의 $(100\alpha/2)$번째와 $(100(1-\alpha/2))$번째 백분위수가 구간 $(L, U)$를 정의한다.

    $$
    L = \hat\theta^*_{(\lfloor B\alpha/2 \rfloor)}, \qquad U = \hat\theta^*_{(\lceil B(1-\alpha/2) \rceil)}
    $$

    여기서 $\hat\theta^*_{(k)}$는 $k$번째 순서통계량이다. 이제 변환된 복제값 $g(\hat\theta_1^*), \ldots, g(\hat\theta_B^*)$을 생각하자. $g$가 단조증가이므로 순서통계량이 일관되게 변환된다.

    $$
    g(\hat\theta^*)_{(k)} = g(\hat\theta^*_{(k)})
    $$

    따라서 $g(\theta)$의 백분위 구간은

    $$
    \bigl(g(\hat\theta^*_{(\lfloor B\alpha/2 \rfloor)}),\;\; g(\hat\theta^*_{(\lceil B(1-\alpha/2) \rceil)})\bigr) = (g(L),\; g(U))
    $$

    이것이 로그 척도에서 백분위 구간을 계산한 뒤 지수를 취하는 것이 비율 척도에서 직접 계산하는 것과 동등한 이유이다. 변환 불변성은 백분위 방법이 정규근사 구간에 비해 갖는 핵심 장점이다.

    **정규근사 구간과의 대비.** $\hat\theta \pm 1.96\,\widehat{\text{SE}}(\hat\theta)$ 형태의 구간은 변환 불변이 **아니다**. 로그 척도에서 계산한 뒤 지수를 취한 구간과 비율 척도에서 직접 계산한 구간이 다르다. 그래서 어느 척도에서 작업할지가 실질적인 선택이 된다.

    **BCa도 변환 불변이다.** $z_0$와 $\hat a$가 모두 순서에만 의존하는 양이므로, BCa 구간 역시 단조변환에 불변이다. 연습문제 4의 BCa 구간을 로그 척도에서 계산한 뒤 지수를 취하든 비율 척도에서 직접 계산하든 같은 결과가 나온다.

    **그렇다면 로그 척도는 언제 필요한가.** 붓스트랩 $t$ 구간처럼 표준오차 추정값으로 스튜던트화하는 방법, 정규근사, 그리고 붓스트랩 분포를 히스토그램으로 시각화할 때 필요하다. 백분위와 BCa만 쓴다면 척도 선택이 결과를 바꾸지 않는다. $\square$
