# σ₁² / σ₂²의 신뢰구간

## 두 분산의 비에 대한 신뢰구간

독립인 두 모집단의 변동성을 비교할 때는 비 $\theta = \sigma_1^2 / \sigma_2^2$의 신뢰구간을 구성한다. 이는 F-분포에 기반한다.

### 공식

$$
\left[\frac{s_1^2 / s_2^2}{F_{\alpha/2,\, n_1-1,\, n_2-1}},\;\; \frac{s_1^2 / s_2^2}{F_{1-\alpha/2,\, n_1-1,\, n_2-1}}\right]
$$

여기서

- $s_1^2$과 $s_2^2$은 (Bessel 수정을 한) 표본분산,
- $F_{\alpha/2, n_1-1, n_2-1}$과 $F_{1-\alpha/2, n_1-1, n_2-1}$은 자유도 $\text{df}_1 = n_1 - 1$, $\text{df}_2 = n_2 - 1$인 F-분포의 임계값이다.

### 표본분포

추축량은

$$
\frac{s_1^2 / \sigma_1^2}{s_2^2 / \sigma_2^2} \sim F_{n_1-1, \, n_2-1}
$$

이 결과는 두 모집단이 모두 정규분포를 따를 때 정확히 성립한다.

### 타당성 조건

$$
\sigma_1^2/\sigma_2^2 \text{에 대한 F-구간}
\quad\text{if}\quad
\begin{cases}
\text{두 모집단 분포가 모두 정규이다} \\
\text{각 집단에서 } n_i \le 0.1 N_i \text{ (i.i.d. 근사)}
\end{cases}
$$

!!! warning "정규성 요구"
    카이제곱 분산 신뢰구간과 마찬가지로 F-구간도 **정규성 아래에서만 정확하다**. 정규가 아닌 자료(치우쳤거나 꼬리가 두껍거나 이상점이 있는 자료)에서는 포함확률이 심각하게 어긋날 수 있다. 정규가 아닌 자료에는 붓스트랩이나 로버스트한 대안을 고려하라.

<div class="codebox" markdown>

#### 예제 1. 분산비의 신뢰구간 계산 { .eg }

```python
import numpy as np
from scipy.stats import f

n1, n2 = 15, 12
alpha = 0.05

# 참 분산비는 1/1.5² = 0.444다. 답을 알고 있는 상태에서 구간을 본다.
rng = np.random.default_rng(42)
x = rng.normal(loc=0, scale=1.0, size=n1)
y = rng.normal(loc=0, scale=1.5, size=n2)

s1_sq = x.var(ddof=1)
s2_sq = y.var(ddof=1)
rhat = s1_sq / s2_sq

# F분포는 두 자유도의 **순서**가 중요하다. dfn이 분자, dfd가 분모다.
# 두 표본의 크기가 다르므로 여기서 뒤바꾸면 조용히 틀린 답이 나온다.
df1, df2 = n1 - 1, n2 - 1
F_lo = f(dfn=df1, dfd=df2).ppf(alpha / 2.0)
F_hi = f(dfn=df1, dfd=df2).ppf(1 - alpha / 2.0)

# 여기서도 큰 임계값이 아래끝의 분모로 간다.
# (s1²/s2²)/(σ1²/σ2²) ~ F 를 σ1²/σ2² 에 대해 풀면 대소가 뒤집히기 때문이다.
ci_lower = rhat / F_hi
ci_upper = rhat / F_lo

print(f"95% CI for σ₁²/σ₂²: ({ci_lower:.4f}, {ci_upper:.4f})")
```

출력:

```
95% CI for σ₁²/σ₂²: (0.2396, 2.4903)
```

참값 0.444를 담기는 하지만 위끝이 아래끝의 열 배다. 구간이 1을 넉넉히 담고 있으므로 "두 분산이 같다"는 가설조차 배제하지 못한다. 실제로는 $\sigma_2$가 $\sigma_1$의 1.5배인데도 그렇다. 분산 하나를 추정하는 것도 어려운데 그 비를 추정하는 일은 훨씬 더 어렵다.

</div>

---

## 모의실험: 분산비 신뢰구간의 포함확률

<div class="codebox" markdown>

### 예제 2. F 구간의 포함확률 { .eg }

```python
#!/usr/bin/env python3
"""두 분산의 비에 대한 F 신뢰구간을 100번 만들어 포함확률을 확인한다.

두 표본분산의 비가 F 분포를 따른다는 사실에서 구간이 나온다. 이 방법은
정규성에 특히 민감해서, 모집단이 정규가 아니면 포함확률이 크게 어긋난다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f

rng_seed = 42        # 아래 그림을 재현하려면 고정한다
n_simulations = 100
n1, n2 = 15, 12
mu1, mu2 = 0.0, 0.0
sigma1, sigma2 = 1.0, 1.5
alpha = 0.05


def main():
    if rng_seed is not None:
        np.random.seed(rng_seed)

    theta_true = (sigma1**2) / (sigma2**2)
    df1, df2 = n1 - 1, n2 - 1

    lowers = np.empty(n_simulations)
    uppers = np.empty(n_simulations)
    centers = np.empty(n_simulations)

    F_lo = f(dfn=df1, dfd=df2).ppf(alpha / 2.0)
    F_hi = f(dfn=df1, dfd=df2).ppf(1 - alpha / 2.0)

    for i in range(n_simulations):
        x = np.random.normal(loc=mu1, scale=sigma1, size=n1)
        y = np.random.normal(loc=mu2, scale=sigma2, size=n2)
        s1_sq = x.var(ddof=1)
        s2_sq = y.var(ddof=1)
        rhat = s1_sq / s2_sq
        lowers[i] = rhat / F_hi
        uppers[i] = rhat / F_lo
        centers[i] = rhat

    covered = (lowers <= theta_true) & (theta_true <= uppers)
    n_fail = (~covered).sum()
    coverage_pct = 100.0 * covered.mean()

    fig, ax = plt.subplots(figsize=(12, 12))
    for i in range(n_simulations):
        color = "k" if covered[i] else "r"
        ax.plot([lowers[i], uppers[i]], [i, i], lw=2, color=color)
        ax.plot(centers[i], i, marker="o", ms=3, color=color)

    ax.axvline(theta_true, linestyle="--", linewidth=1.5, color="r")
    ax.set_title(
        f"{n_simulations} F-intervals for σ₁²/σ₂² | n1={n1}, n2={n2}, "
        f"df=({df1},{df2}), CL={int((1 - alpha) * 100)}% | "
        f"Fail={n_fail} (Coverage ≈ {coverage_pct:.1f}%)")
    ax.set_yticks([])
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("θ = σ₁² / σ₂²")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
```

![100 F-intervals for σ₁²/σ₂² | n1=15, n2=12, CL=95%](./img/ci_variance_ratio_74.png)

포함확률 97.0%로 명목값을 달성한다. 그런데 구간의 모양이 눈에 띈다. 대부분은 0 근처에서 2 언저리까지 뻗지만, 하나는 오른쪽 끝이 14를 넘는다. 그 표본에서 우연히 $s_1^2/s_2^2$이 크게 나오자 구간이 통째로 오른쪽으로 밀리며 길이까지 폭발한 것이다.

$\theta$의 척도가 비율이라 이런 일이 생긴다. 아래쪽으로는 0이라는 벽이 있어 눌리고 위쪽으로는 열려 있다. $\log \theta$로 보면 훨씬 대칭적인 그림이 된다.

100개 중 74개가 1을 담고 있다. 참 비율이 0.444, 즉 표준편차로 1.5배 차이인데도 $n_1 = 15$, $n_2 = 12$로는 네 번 중 세 번 등분산을 배제하지 못한다. **등분산 검정으로 Welch를 쓸지 합동 $t$를 쓸지 정하려는 시도가 위험한 이유**가 여기 있다. 검정이 등분산을 기각하지 못했다는 것은 분산이 같다는 뜻이 아니라 표본이 작다는 뜻일 때가 많다.

</div>

---

## 핵심 정리

- $\sigma_1^2 / \sigma_2^2$에 대한 F-구간은 **두 모집단이 모두 정규**일 것을 요구한다.
- 신뢰구간이 1을 포함하면 두 모분산이 다르다는 증거가 없다.
- F-분포는 비대칭이므로 신뢰구간이 점추정값을 중심으로 대칭이 아니다.
- 정규가 아닌 자료에는 붓스트랩 기반 대안을 고려하라.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
정규모집단에서 뽑은 독립인 두 표본에서 $s_1^2 = 25$ ($n_1 = 16$), $s_2^2 = 10$ ($n_2 = 21$)을 얻었다. $\sigma_1^2/\sigma_2^2$의 점추정값을 계산하고 $F_{15,20,0.025} = 0.392$, $F_{15,20,0.975} = 2.573$을 써서 95% 신뢰구간을 구성하라.

</div>

??? success "풀이"
    점추정값은:

    $$
    \frac{s_1^2}{s_2^2} = \frac{25}{10} = 2.5
    $$

    95% 신뢰구간은:

    $$
    \left(\frac{s_1^2/s_2^2}{F_{0.975}},\; \frac{s_1^2/s_2^2}{F_{0.025}}\right) = \left(\frac{2.5}{2.573},\; \frac{2.5}{0.392}\right) = (0.972,\; 6.378)
    $$

    구간이 1을 포함하므로 5% 수준에서 두 분산이 다르다고 결론지을 충분한 증거가 없다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
F에 기반한 신뢰구간이 점추정값을 중심으로 비대칭인 이유를 설명하라. 연습문제 1의 구간으로 예시하라.

</div>

??? success "풀이"
    F-분포는 오른쪽으로 치우쳐 있으므로(양수에서만 정의되고 오른쪽 꼬리가 더 길다) 임계값 $F_{\alpha/2}$와 $F_{1-\alpha/2}$가 1에서 같은 거리에 있지 않다. 연습문제 1에서 점추정값은 2.5이다. 신뢰구간의 하한은 $2.5/2.573 = 0.972$로 추정값보다 1.528 아래이고, 상한은 $2.5/0.392 = 6.378$로 추정값보다 3.878 위이다. 구간이 왼쪽보다 오른쪽으로 훨씬 멀리 뻗는데, 이는 F-분포의 치우침을 반영한다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
$\sigma_1^2/\sigma_2^2$의 신뢰구간이 $(1.5, 4.2)$이라면 두 모분산의 관계에 대해 무엇을 알 수 있는가?

</div>

??? success "풀이"
    구간 전체가 1보다 위에 있으므로 주어진 신뢰수준에서 $\sigma_1^2 > \sigma_2^2$이라고 결론지을 수 있다. 구체적으로 첫 번째 모집단의 분산이 두 번째 모집단 분산의 1.5배에서 4.2배 사이라고 신뢰한다. 예를 들어 (등분산을 가정하는) 합동 $t$-검정을 쓸지 Welch $t$-검정을 쓸지 결정할 때 이런 정보가 유용하다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
등분산에 대한 F-검정은 비정규성에 매우 민감한 것으로 알려져 있다. 이유를 설명하고 대안을 하나 들라.

</div>

??? success "풀이"
    F-분포의 유도는 두 모집단이 정확히 정규라고 가정한다. 정규성에서 조금만 벗어나도(약간의 치우침이나 두꺼운 꼬리만으로도) $S_1^2/S_2^2$의 분포가 F-분포에서 크게 벗어난다. 모집단의 첨도가 $S^2$의 분산에 직접 영향을 주므로, 꼬리가 두꺼운 모집단에서는 이 비가 F 이론이 예측하는 것보다 훨씬 넓게 퍼지고 제1종 오류율이 부풀려진다.

    대안으로 **Levene 검정**이 있다. 절대편차 $|X_{ij} - \bar{X}_j|$(Brown-Forsythe에서는 중앙값으로부터의 편차)에 일원분산분석을 적용하여 분산의 동일성을 검정한다. 이 방법은 비정규성에 훨씬 로버스트하다. 분산비에 대한 붓스트랩 신뢰구간도 또 다른 비모수적 대안이다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
$F$ 기반 분산비 구간이 **비정규성**에 얼마나 취약한지 모의실험으로 재고, $n$을 키우면 회복되는지 확인하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(23)
    M = 20_000
    cases = [("정규", lambda s: rng.normal(0, 1, s)),
             ("균등", lambda s: rng.uniform(-1.732, 1.732, s)),
             ("t(10)", lambda s: rng.standard_t(10, s)),
             ("t(5)", lambda s: rng.standard_t(5, s)),
             ("지수", lambda s: rng.exponential(1, s))]
    print(f"{'분포':>7s} {'n=20':>9s} {'n=50':>9s}")
    for name, gen in cases:
        row = []
        for n in [20, 50]:
            x, y = gen((M, n)), gen((M, n))
            r = x.var(1, ddof=1) / y.var(1, ddof=1)
            lo = r / stats.f.ppf(0.975, n - 1, n - 1)
            hi = r / stats.f.ppf(0.025, n - 1, n - 1)
            row.append(np.mean((lo <= 1) & (1 <= hi)))
        print(f"{name:>7s} {row[0]:9.4f} {row[1]:9.4f}")
    ```

    ```text
        분포      n=20      n=50
        정규    0.9473    0.9504
        균등    0.9944    0.9967
     t(10)    0.9075    0.9000
      t(5)    0.8373    0.8025
        지수    0.7323    0.7046
    ```

    **$n$을 키워도 회복되지 않는다.** 오히려 나빠진다. $t_5$에서 0.837 → 0.803, 지수에서 0.732 → 0.705.

    **앞서 본 단일 분산 구간과 같은 구조다.** $\log(S_1^2/S_2^2)$의 점근분산이

    $$
    \operatorname{Var}\left\{\log\frac{S_1^2}{S_2^2}\right\}\approx(\gamma_2+2)\left(\frac1{n_1}+\frac1{n_2}\right)
    $$

    로 초과첨도 $\gamma_2$에 비례하는데, $F$ 분포는 $\gamma_2=0$을 가정한다. 지수분포($\gamma_2=6$)에서는 **참 분산이 네 배**라 구간이 절반 폭밖에 안 된다.

    **꼬리가 얇으면 반대다.** 균등분포($\gamma_2=-1.2$)에서 0.994로 지나치게 보수적이다.

    **$t_{10}$조차 0.90이다.** 초과첨도가 1.0에 불과한, 정규에 꽤 가까운 분포인데도 5%포인트가 어긋난다. **$F$ 검정과 $F$ 구간이 "정규성에 극도로 민감하다"는 평판은 과장이 아니다.**

    **함의.** 앞 절에서 보았듯 등분산 사전검정에 $F$ 검정을 쓰면, 그 검정이 기각하는 이유가 **분산 차이인지 첨도인지 구별되지 않는다.** 이것이 사전검정을 권하지 않는 또 하나의 이유다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
분산비 구간의 **강건한 대안** 두 가지 — 로그 척도 부트스트랩과 첨도 보정 — 을 구현하고 $F$ 구간과 비교하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(77)
    n1 = n2 = 30
    M, B = 2_000, 999
    z = stats.norm.ppf(0.975)

    cases = [("정규", lambda s: rng.normal(0, 1, s)),
             ("t(5)", lambda s: rng.standard_t(5, s)),
             ("지수", lambda s: rng.exponential(1, s))]
    for name, gen in cases:
        hf = hb = hk = 0
        for _ in range(M):
            x, y = gen(n1), gen(n2)
            r = x.var(ddof=1) / y.var(ddof=1)

            lo = r / stats.f.ppf(0.975, n1 - 1, n2 - 1)
            hi = r / stats.f.ppf(0.025, n1 - 1, n2 - 1)
            hf += (lo <= 1 <= hi)

            bx = x[rng.integers(0, n1, (B, n1))]
            by = y[rng.integers(0, n2, (B, n2))]
            br = np.log(bx.var(1, ddof=1) / by.var(1, ddof=1))
            l, h = np.percentile(br, [2.5, 97.5])
            hb += (l <= 0 <= h)

            kk = (stats.kurtosis(x, bias=False) + stats.kurtosis(y, bias=False)) / 2
            sd = np.sqrt((kk + 2) * (1 / (n1 - 1) + 1 / (n2 - 1)))
            hk += (abs(np.log(r)) <= z * sd)

        print(f"{name:5s} F {hf / M:.4f}   부트스트랩(로그) {hb / M:.4f}   "
              f"첨도 보정 {hk / M:.4f}")
    ```

    ```text
    정규    F 0.9485   부트스트랩(로그) 0.9340   첨도 보정 0.9400
    t(5)  F 0.8100   부트스트랩(로그) 0.9110   첨도 보정 0.9015
    지수    F 0.7065   부트스트랩(로그) 0.9040   첨도 보정 0.8710
    ```

    **두 대안 모두 크게 낫다.** 지수분포에서 $F$는 0.707이지만 부트스트랩은 0.904, 첨도 보정은 0.871이다.

    **정규에서는 약간 손해다.** 0.949 → 0.934, 0.940. **모형을 덜 가정하는 대가**이며, 1~1.5%포인트에 불과하다.

    **부트스트랩이 조금 더 낫다.** 첨도를 명시적으로 추정하지 않고 재표본 분포가 스스로 말하게 하기 때문이다. 첨도 보정은 $\hat\gamma_2$의 추정오차가 크다는 약점을 그대로 안고 있다.

    **여전히 0.95에 못 미친다.** $n=30$에서 지수분포처럼 극단적인 경우에는 어떤 방법도 완전하지 않다. **BCa를 쓰면 조금 더 개선**되며, $n$이 커지면 부트스트랩은 제대로 수렴한다($F$와 달리).

    **로그 척도를 쓴 이유.**

    1. **분포가 대칭에 가까워진다.** $S_1^2/S_2^2$의 분포는 1에서 심하게 비대칭이다.
    2. **경계를 지킨다.** 지수를 취해 되돌리면 항상 양수다.
    3. **역수 대칭.** 두 집단의 순서를 바꾸면 $\log r$의 부호만 바뀐다. 구간도 부호를 뒤집은 것이 되어 **일관성**이 있다.

    **권고.** 분산비가 주 관심사라면 **정규성을 진단하고, 의심되면 부트스트랩**을 쓴다. $F$ 구간은 정규성이 뒷받침될 때만 보고한다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
$\sigma_1^2/\sigma_2^2$의 구간에서 $\sigma_1/\sigma_2$의 구간을 얻는 법을 보이고, **역수 대칭성**을 확인하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    s1sq, n1 = 25.0, 16
    s2sq, n2 = 10.0, 21
    r = s1sq / s2sq
    lo = r / stats.f.ppf(0.975, n1 - 1, n2 - 1)
    hi = r / stats.f.ppf(0.025, n1 - 1, n2 - 1)
    print(f"분산비 {r:.4f}      구간 ({lo:.4f}, {hi:.4f})")
    print(f"표준편차비 {np.sqrt(r):.4f}  구간 ({np.sqrt(lo):.4f}, {np.sqrt(hi):.4f})")

    # 순서를 바꾸면
    r2 = s2sq / s1sq
    lo2 = r2 / stats.f.ppf(0.975, n2 - 1, n1 - 1)
    hi2 = r2 / stats.f.ppf(0.025, n2 - 1, n1 - 1)
    print(f"\n역순 분산비 {r2:.4f}  구간 ({lo2:.4f}, {hi2:.4f})")
    print(f"원 구간의 역수      ({1 / hi:.4f}, {1 / lo:.4f})")
    ```

    ```text
    분산비 2.5000      구간 (0.9716, 6.8898)
    표준편차비 1.5811  구간 (0.9857, 2.6248)

    역순 분산비 0.4000  구간 (0.1452, 1.0293)
    원 구간의 역수      (0.1452, 1.0293)
    ```

    **두 가지가 확인된다.**

    1. **제곱근 변환.** $g(t)=\sqrt t$가 순증가이므로 양끝에 제곱근을 취하면 $\sigma_1/\sigma_2$의 구간이 되고 신뢰수준이 보존된다.

    2. **역수 대칭.** 집단의 순서를 바꿔 계산한 구간이 원 구간의 역수와 **정확히 같다.** $F_{\nu_2,\nu_1,\alpha}=1/F_{\nu_1,\nu_2,1-\alpha}$라는 $F$ 분포의 항등식 덕이다.

    **왜 역수 대칭이 중요한가.** "집단 1이 집단 2보다 분산이 크다"와 "집단 2가 집단 1보다 분산이 작다"는 같은 주장이다. **절차가 어느 집단을 분자에 두느냐에 따라 다른 답을 주면 안 된다.**

    **이 자료의 해석.** 분산비 구간 $(0.97,\ 6.89)$가 1을 **간신히** 담는다. 하한이 0.9716이므로 아슬아슬하게 등분산을 기각하지 못한다. 표준편차 척도로는 $(0.99,\ 2.62)$다.

    **비대칭이 극심하다.** 점추정값 2.5에서 하한까지 1.53, 상한까지 4.39로 **상한 쪽이 세 배 멀다.** 로그 척도에서 보면

    $$
    \log(0.9716)=-0.029,\quad \log 2.5=0.916,\quad \log(6.8898)=1.930
    $$

    으로 중심에서 $-0.945$와 $+1.014$다. 여전히 대칭이 아닌데, 이는 두 자유도($15$와 $20$)가 다르기 때문이다. $n_1=n_2$이면 로그 척도에서 정확히 대칭이 된다.

    **보고 권고.** 분산보다 **표준편차 비**로 보고하는 것이 읽기 쉽다. "집단 1의 표준편차가 집단 2의 1.58배(95% CI 0.99~2.62)"가 직관적이다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
분산비 구간을 목표 정밀도로 만들려면 **표본이 얼마나** 필요한가? 평균 비교와 비교하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    print(f"{'n(양군)':>8s} {'하한/r':>9s} {'상한/r':>9s} {'상한/하한':>10s}")
    for n in [10, 20, 30, 50, 100, 200, 500]:
        lo = 1 / stats.f.ppf(0.975, n - 1, n - 1)
        hi = 1 / stats.f.ppf(0.025, n - 1, n - 1)
        print(f"{n:8d} {lo:9.4f} {hi:9.4f} {hi / lo:10.2f}")

    print()
    for target in [10, 4, 2, 1.5]:
        n = next(m for m in range(5, 5000)
                 if (1 / stats.f.ppf(0.025, m - 1, m - 1))
                 / (1 / stats.f.ppf(0.975, m - 1, m - 1)) <= target)
        print(f"상한/하한 {target}배 이하 → 집단당 n = {n}")
    ```

    ```text
     n(양군)    하한/r    상한/r    상한/하한
          10    0.2484    4.0260      16.21
          20    0.3958    2.5265       6.38
          30    0.4760    2.1010       4.41
          50    0.5675    1.7622       3.11
         100    0.6728    1.4862       2.21
         200    0.7568    1.3214       1.75
         500    0.8389    1.1921       1.42

    상한/하한 10배 이하 → 집단당 n = 14
    상한/하한 4배 이하 → 집단당 n = 35
    상한/하한 2배 이하 → 집단당 n = 131
    상한/하한 1.5배 이하 → 집단당 n = 376
    ```

    **분산비는 매우 부정확하게 추정된다.** 집단당 30명이면 구간의 상한이 하한의 **4.4배**다. "분산비가 0.48배에서 2.1배 사이"라는 것은 사실상 아무 말도 하지 않는 것에 가깝다.

    **평균 비교와의 대비.**

    | 목표 | 분산비 | 평균 차이($\sigma$ 단위로 $\pm0.5$) |
    |---|---|---|
    | 집단당 $n$ | 131(2배 이내) | 32 |

    같은 수준의 "쓸 만한" 정밀도를 얻는 데 **분산이 네 배쯤 더 든다.**

    **왜 그런가.** $\log(S^2)$의 표준오차가 $\sqrt{2/(n-1)}$로 $\bar X$의 $\sigma/\sqrt n$보다 본질적으로 크고, 게다가 **분산비는 두 추정값의 불확실성을 모두 안는다.**

    **실무적 함의.**

    1. **등분산 검정이 소표본에서 무력한 이유**가 여기 있다. $n=20$에서 참 분산비가 2여도 구간이 $(0.79,\ 5.05)$라 1을 담는다.

    2. **분산을 비교하는 것이 주 목적이면** 표본을 크게 잡아야 한다. 부수적인 진단으로 쓰는 것과는 요구가 다르다.

    3. **웰치를 기본으로 쓰는 또 하나의 근거.** 등분산 여부를 확인할 힘이 없으므로, 확인에 의존하지 않는 절차를 쓰는 것이 합리적이다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
분산비가 **그 자체로 관심사**인 실무 상황을 셋 들고, 각각에서 무엇을 보고해야 하는지 적어라.

</div>

??? success "풀이"
    **상황 1 — 측정 시스템 분석.** 두 측정기기나 두 검사자의 **재현성**을 비교한다.

    - 같은 시료를 반복 측정해 각 기기의 반복성 분산을 구하고 비를 본다.
    - **보고**: 표준편차 비와 구간. "새 기기의 측정 표준편차가 기존의 0.72배(95% CI 0.55~0.94)"
    - **주의**: 측정값의 분포가 정규인지 확인한다. 아니면 부트스트랩.

    **상황 2 — 공정 개선의 검증.** 개선 전후의 변동을 비교한다.

    - **평균이 아니라 산포를 줄이는 것**이 목표인 경우가 많다. 규격 중심에 맞추는 것보다 흔들림을 줄이는 것이 어렵다.
    - **보고**: 분산비와 함께 **공정능력지수**의 변화. $C_p=(\text{USL}-\text{LSL})/(6\sigma)$이므로 $\sigma$가 0.72배면 $C_p$가 1.39배다.
    - **주의**: 개선 전후로 다른 요인이 바뀌지 않았는지.

    **상황 3 — 개체 내 변동과 개체 간 변동.** 급내상관계수(ICC)가 분산비의 함수다.

    $$
    \text{ICC}=\frac{\sigma_b^2}{\sigma_b^2+\sigma_w^2}=\frac{\lambda}{1+\lambda},
    \qquad \lambda=\frac{\sigma_b^2}{\sigma_w^2}
    $$

    ```python
    for lam in [0.5, 1, 2, 4, 9, 19]:
        print(f"σ_b²/σ_w² = {lam:4.1f}  →  ICC = {lam / (1 + lam):.3f}")
    ```

    ```text
    σ_b²/σ_w² =  0.5  →  ICC = 0.333
    σ_b²/σ_w² =  1.0  →  ICC = 0.500
    σ_b²/σ_w² =  2.0  →  ICC = 0.667
    σ_b²/σ_w² =  4.0  →  ICC = 0.800
    σ_b²/σ_w² =  9.0  →  ICC = 0.900
    σ_b²/σ_w² = 19.0  →  ICC = 0.950
    ```

    - **ICC 0.9를 넘으려면 분산비가 9 이상**이어야 한다. 신뢰도 기준이 왜 그렇게 엄격한지 보여 준다.
    - **보고**: ICC와 그 구간. $\lambda$의 구간에 $\lambda/(1+\lambda)$를 적용하면 되며, 단조증가 변환이라 신뢰수준이 보존된다.
    - **주의**: ICC는 **모집단의 이질성에 의존**한다. 동질적인 집단에서 잰 ICC는 낮게 나오며, 이것이 측정이 나쁘다는 뜻은 아니다.

    **공통 권고.**

    1. **표준편차 척도로 보고**한다. 분산은 단위가 제곱이라 읽기 어렵다.
    2. **구간을 반드시 함께** 적는다. 앞 문제에서 보았듯 점추정값만으로는 거의 정보가 없다.
    3. **정규성을 확인**하거나 강건한 방법을 쓴다.
    4. **실무적 문턱과 비교**한다. "분산비가 유의하게 1과 다르다"보다 "규격을 만족할 만큼 줄었는가"가 중요하다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
등분산 검정의 여러 방법($F$, 바틀렛, 레빈, 브라운-포사이드)을 비교하고, 실무 권고를 정리하라.

</div>

??? success "풀이"
    **네 방법의 구조.**

    | 방법 | 통계량의 바탕 | 정규성 의존 |
    |---|---|---|
    | $F$ | $S_1^2/S_2^2$ | **극도로 높음** |
    | 바틀렛 | 로그 분산의 가중합($k$개 집단) | **극도로 높음** |
    | 레빈 | $|x_{ij}-\bar x_i|$의 분산분석 | 중간 |
    | 브라운-포사이드 | $|x_{ij}-\text{med}_i|$의 분산분석 | **낮음** |

    **핵심 착안.** 레빈과 브라운-포사이드는 **절대편차를 새 자료로 보고 평균 비교 문제로 바꾼다.** 평균 비교는 중심극한정리의 보호를 받으므로 훨씬 강건해진다. 브라운-포사이드가 중앙값을 쓰는 것은 절대편차 자체의 치우침을 줄이기 위해서다.

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(404)
    n, M = 30, 3_000
    print(f"{'분포':>6s} {'F':>8s} {'바틀렛':>8s} {'레빈':>8s} {'BF':>8s}")
    for name, gen in [("정규", lambda s: rng.normal(0, 1, s)),
                      ("t(5)", lambda s: rng.standard_t(5, s)),
                      ("지수", lambda s: rng.exponential(1, s))]:
        rej = np.zeros(4)
        for _ in range(M):
            x, y = gen(n), gen(n)
            f = x.var(ddof=1) / y.var(ddof=1)
            pf = 2 * min(stats.f.cdf(f, n - 1, n - 1), stats.f.sf(f, n - 1, n - 1))
            pb = stats.bartlett(x, y).pvalue
            pl = stats.levene(x, y, center="mean").pvalue
            pv = stats.levene(x, y, center="median").pvalue
            rej += np.array([pf, pb, pl, pv]) < 0.05
        print(f"{name:>6s} " + " ".join(f"{v / M:8.4f}" for v in rej))
    ```

    ```text
      분포        F     바틀렛       레빈       BF
      정규   0.0527   0.0527   0.0517   0.0440
    t(5)   0.1667   0.1667   0.0470   0.0370
      지수   0.2790   0.2790   0.1330   0.0577
    ```

    **$H_0$가 참인데도 $F$와 바틀렛이 지수분포에서 28% 기각한다.** 명목 5%의 다섯 배가 넘는다. 두 집단일 때 $F$와 바틀렛은 사실상 같은 검정이라 값이 일치한다.

    **브라운-포사이드가 가장 강건하다.** 지수분포에서도 5.8%로 명목에 가깝고, $t_5$에서는 오히려 보수적이다(3.7%). 레빈(평균 기준)은 지수분포에서 13.3%로 중간이다.

    **실무 권고.**

    1. **등분산 검정 자체를 하지 않는 것이 첫째 권고다.** 앞 절에서 본 대로, 두 집단 평균 비교에는 웰치를 그냥 쓰면 된다.

    2. **꼭 해야 한다면 브라운-포사이드.** `scipy.stats.levene(..., center="median")`이 이것이다. $F$와 바틀렛은 **정규성이 확실한 경우에만** 쓴다.

    3. **분산 자체가 관심사라면** 검정이 아니라 **구간**을 보고한다. "등분산을 기각하지 못했다"는 진술은 정보가 거의 없다.

    4. **$k>2$ 집단이면** 브라운-포사이드나 웰치의 분산분석(`oneway.anova` 계열)을 쓴다.

    **더 근본적으로.** 이분산이 있다는 것은 **모형이 불완전하다**는 신호일 수 있다. 분산이 평균에 따라 변한다면(계수형 자료, 비율 자료), 그 구조를 반영하는 일반화선형모형이 이분산을 "고치는" 것이 아니라 **애초에 옳게 모형화**하는 길이다.

---

## 정리하며

두 분산은 **차가 아니라 비**로 비교한다.

$$
\left[\frac{s_1^2/s_2^2}{F_{\alpha/2,\,n_1-1,\,n_2-1}},\;\frac{s_1^2/s_2^2}{F_{1-\alpha/2,\,n_1-1,\,n_2-1}}\right]
$$

- **비를 쓰는 이유는 추축량이 만들어지기 때문이다.** $\frac{s_1^2/\sigma_1^2}{s_2^2/\sigma_2^2}\sim F_{n_1-1,n_2-1}$ 이며, 이 양의 분포가 미지 모수에 의존하지 않는다. 차에는 이런 성질이 없다.
- **분모의 임계값이 뒤바뀐다.** 분산 구간과 마찬가지로 큰 분위수가 하한에, 작은 분위수가 상한에 들어간다.
- **$1$ 을 포함하는지가 관심사다.** 포함하면 두 분산이 같다는 가설을 기각하지 못한다. 분산에서 $1$ 은 평균에서의 $0$ 에 해당한다.
- **$F$ 분포의 뒤집기 성질이 유용하다.** $F_{1-\alpha/2,d_1,d_2}=1/F_{\alpha/2,d_2,d_1}$ 이므로 표 한쪽만 있으면 된다.
- **정규성에 매우 민감하다.** 일표본 분산 구간보다도 더 취약하며, 두 모집단 모두 정규여야 한다. **실무에서는 레빈 검정이나 부트스트랩 쪽이 안전하다**(15장).

다음 절 **평균 차이 신뢰구간 모의실험**에서 여러 방법의 포함확률을 직접 재어 본다.
