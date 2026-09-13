# 분산 신뢰구간의 포함확률 모의실험

## 개요

이 페이지에서는 모분산 $\sigma^2$에 대한 카이제곱 신뢰구간을 시연하고 모의실험으로 그 포함확률을 평가한다. 카이제곱 분산 구간은 정규성 아래에서 정확하지만 바탕 모집단이 치우쳐 있거나 꼬리가 두꺼우면 실패할 수 있다. 모의실험은 정규 자료에서 명목 포함확률이 달성됨을 확인하고 정규성 가정에 대한 민감성을 부각한다.

## 분산에 대한 카이제곱 신뢰구간

### 추축량

$X_1, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$이면 통계량

$$
\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}
$$

이며, 여기서 $S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2$은 불편 표본분산이다.

### 신뢰구간

추축량을 뒤집으면 $\sigma^2$의 $(1-\alpha)100\%$ 신뢰구간을 얻는다:

$$
\left(\frac{(n-1)S^2}{\chi^2_{1-\alpha/2,\,n-1}},\;\;
      \frac{(n-1)S^2}{\chi^2_{\alpha/2,\,n-1}}\right)
$$

**더 큰** 카이제곱 분위수가 **아래쪽** 끝점에 나타나는데, 큰 수로 나누면 결과가 작아지기 때문이다.

$\sigma$의 신뢰구간은 제곱근을 취해 얻는다:

$$
\left(\sqrt{\frac{(n-1)S^2}{\chi^2_{1-\alpha/2,\,n-1}}},\;\;
      \sqrt{\frac{(n-1)S^2}{\chi^2_{\alpha/2,\,n-1}}}\right)
$$

### 결정적인 정규성 가정

이 구간은 **자료가 정규분포를 따를 때에만 정확하다**. (중심극한정리의 도움을 받는) 평균의 신뢰구간과 달리, 카이제곱 분산 구간은 $n$이 커져도 비정규성에 로버스트해지지 않는다. 치우치거나 꼬리가 두꺼운 자료에는 $\sigma^2$에 대한 붓스트랩 신뢰구간을 고려하라.

<div class="codebox" markdown>

#### 예제 1. 분산 신뢰구간 모의실험 { .eg }

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

np.random.seed(42)          # 아래 출력과 그림을 재현하려면 고정한다

n_simulations = 100
n_samples = 12
mu, sigma = 0.0, 2.0
alpha = 0.05

true_var = sigma**2
df = n_samples - 1
# 임계값은 표본과 무관하므로 반복문 밖에서 한 번만 구한다.
# 자유도만으로 정해지는 값이기 때문이다.
chi2_lo = chi2.ppf(alpha / 2, df=df)
chi2_hi = chi2.ppf(1 - alpha / 2, df=df)

lowers = np.empty(n_simulations)
uppers = np.empty(n_simulations)
centers = np.empty(n_simulations)

for i in range(n_simulations):
    x = np.random.normal(loc=mu, scale=sigma, size=n_samples)
    s2 = x.var(ddof=1)
    lowers[i] = df * s2 / chi2_hi
    uppers[i] = df * s2 / chi2_lo
    centers[i] = s2

covered = (lowers <= true_var) & (true_var <= uppers)
n_fail = int((~covered).sum())
coverage_pct = 100.0 * covered.mean()
print(f"Coverage: {coverage_pct:.1f}%, Failures: {n_fail}")
```

출력:

```
Coverage: 95.0%, Failures: 5
```

$n = 12$밖에 안 되지만 포함확률이 명목값과 정확히 맞는다. 근사가 아니라 정확한 분포 결과이기 때문이다. 단, 이것은 자료가 **정규**일 때의 이야기다(연습문제 3 참조).

</div>

### 구간의 시각화

<div class="codebox" markdown>

#### 예제 2. 구간 100개를 한 그림에 { .eg }

```python
# 구간 하나를 가로선 하나로 그린다. 참값을 담은 구간은 검정, 놓친 구간은
# 빨강이다. 세로 점선이 참값이고, 빨간 선이 몇 개인지 세는 것이 곧 포함확률을
# 재는 일이다. 구간마다 길이가 다른 까닭은 표본마다 s 가 다르기 때문이다.
fig, ax = plt.subplots(figsize=(12, 12))
for i in range(n_simulations):
    color = "k" if covered[i] else "r"
    ax.plot([lowers[i], uppers[i]], [i, i], lw=2, color=color)
    ax.plot(centers[i], i, marker="o", ms=3, color=color)

ax.axvline(true_var, linestyle="--", linewidth=1.5, color="r")
ax.set_title(f"{n_simulations} Chi-square Variance CIs | n={n_samples}, CL=95%")
ax.set_yticks([])
ax.set_xlabel("Variance")
plt.tight_layout()
plt.show()
```

![100 Chi-square Variance CIs | n=12, CL=95%](./img/ci_var_sim_81.png)

각 구간의 점이 $s^2$이다. 점이 구간 한가운데가 아니라 왼쪽에 치우쳐 있다는 것이 이 구간의 특징이다. 평균의 $t$-구간에서는 점이 언제나 정확히 가운데였다.

</div>

## 해석

- 자료가 실제로 정규분포에서 나오면 카이제곱 구간은 명시한 포함 수준을 달성한다. 모의실험에서 $n = 12$일 때 경험적 포함확률이 95%에 가까움이 확인된다.
- 카이제곱분포가 오른쪽으로 치우쳐 있으므로 구간은 **비대칭**이다. 상한이 하한보다 $S^2$에서 더 멀어지는 경향이 있다.
- 모집단이 정규가 아니면(예: 지수, 자유도가 작은 $t$, log-normal) 카이제곱 신뢰구간의 포함확률이 크게 부족하거나 과할 수 있다. $\sigma^2$에 대한 붓스트랩 백분위수나 BCa 구간이 더 로버스트한 대안이다.
- 양 끝점에 제곱근을 취해 표준편차 척도로 보고할 수도 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> 정규모집단에서 뽑은 크기 $n = 20$인 표본에서 $s^2 = 16$을 얻었다. $\sigma^2$과 $\sigma$의 95% 신뢰구간을 구성하라.

</div>

??? success "풀이"

    $\text{df} = 19$, $\alpha = 0.05$일 때:

    $$
    \chi^2_{0.025,\,19} = 8.907, \quad \chi^2_{0.975,\,19} = 32.852
    $$

    $\sigma^2$의 95% 신뢰구간은

    $$
    \left(\frac{19 \times 16}{32.852},\; \frac{19 \times 16}{8.907}\right) = \left(\frac{304}{32.852},\; \frac{304}{8.907}\right) = (9.25,\; 34.13)
    $$

    제곱근을 취하면 $\sigma$의 95% 신뢰구간은

    $$
    (\sqrt{9.25},\; \sqrt{34.13}) = (3.04,\; 5.84)
    $$

    $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> $X_1, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$일 때 $(n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$임을 증명하라.

</div>

??? success "풀이"

    $Z_i = (X_i - \mu)/\sigma$라 하면 $Z_i \overset{\text{iid}}{\sim} N(0,1)$이다. 그러면

    $$
    \sum_{i=1}^n Z_i^2 = \sum_{i=1}^n \frac{(X_i - \mu)^2}{\sigma^2} \sim \chi^2_n
    $$

    항등식 $X_i - \mu = (X_i - \bar{X}) + (\bar{X} - \mu)$로 분해하면:

    $$
    \sum_{i=1}^n (X_i - \mu)^2 = \sum_{i=1}^n (X_i - \bar{X})^2 + n(\bar{X} - \mu)^2
    $$

    $\sigma^2$으로 나누면:

    $$
    \chi^2_n = \frac{(n-1)S^2}{\sigma^2} + \frac{(\bar{X}-\mu)^2}{\sigma^2/n}
    $$

    둘째 항은 $Z = (\bar{X}-\mu)/(\sigma/\sqrt{n}) \sim N(0,1)$인 $Z^2$이므로 $\chi^2_1$을 따른다. (정규성 아래 $\bar{X}$와 $S^2$의 독립성, 즉 Cochran 정리에 의해) 두 항은 독립이다. 따라서 독립인 카이제곱 확률변수의 가법성에 의해

    $$
    \frac{(n-1)S^2}{\sigma^2} = \chi^2_n - \chi^2_1 \sim \chi^2_{n-1}
    $$

    이다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> 모의실험을 고쳐서 $\lambda = 1$인 지수분포(따라서 $\sigma^2 = 1$)와 $n = 20$으로 자료를 뽑아라. 경험적 포함확률을 보고하라. 카이제곱 구간이 실패하는 이유는?

</div>

??? success "풀이"

    ```python
    from scipy.stats import chi2
    np.random.seed(0)
    n, n_sim, true_var = 20, 10_000, 1.0
    df = n - 1
    chi2_lo = chi2.ppf(0.025, df)
    chi2_hi = chi2.ppf(0.975, df)
    covers = 0
    for _ in range(n_sim):
        x = np.random.exponential(1.0, size=n)
        s2 = x.var(ddof=1)
        lo = df * s2 / chi2_hi
        hi = df * s2 / chi2_lo
        if lo <= true_var <= hi:
            covers += 1
    print(f"Coverage: {100 * covers / n_sim:.1f}%")
    ```

    출력:

    ```
    Coverage: 72.7%
    ```

    95%를 훨씬 밑돈다. 스무 번에 한 번 놓치리라 믿었는데 네 번에 한 번 놓친다. 지수분포는 오른쪽으로 치우쳐 있고 초과첨도가 6이므로 $(n-1)S^2/\sigma^2$이 $\chi^2_{n-1}$을 따르지 않는다.

    앞 페이지의 평균 신뢰구간과 비교해 보라. 같은 지수자료, 비슷한 표본크기에서 $t$-구간의 포함확률은 90% 근처였고 $n$을 키우면 95%로 올라왔다. 여기서는 그렇지 않다. 이 코드에서 $n$만 바꿔 보면

    | $n$ | 20 | 100 | 1000 |
    |---|---|---|---|
    | 포함확률 | 72.7% | 68.4% | 67.2% |

    로 **오히려 나빠지면서** 67% 근처로 수렴한다.

    이유는 분명하다. 평균의 구간은 중심극한정리가 받쳐 주므로 $n$이 커지면 어떤 분포에서든 맞아 들어간다. 반면 카이제곱 추축량은 정규성 자체에 의존하며, 표본을 아무리 늘려도 모집단이 정규가 되지는 않는다. $\text{Var}(S^2)$은 첨도에 따라 달라지는데 카이제곱 구간은 정규분포의 첨도를 가정해 너비를 정하므로, 첨도가 큰 자료에서는 구간이 처음부터 너무 좁다. $n$을 키우면 그 "너무 좁음"이 더 또렷해질 뿐이다. **비정규 자료의 분산 구간에는 붓스트랩을 쓰라.** $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> $\sigma^2$의 카이제곱 신뢰구간이 $S^2$을 중심으로 대칭이 아님을 보이고, 그 이유를 기하적으로 설명하라.

</div>

??? success "풀이"

    하한은 $L = (n-1)S^2 / \chi^2_{1-\alpha/2}$이고 상한은 $U = (n-1)S^2 / \chi^2_{\alpha/2}$이다. $S^2$으로부터의 거리는:

    $$
    S^2 - L = S^2\!\left(1 - \frac{n-1}{\chi^2_{1-\alpha/2}}\right), \quad U - S^2 = S^2\!\left(\frac{n-1}{\chi^2_{\alpha/2}} - 1\right)
    $$

    $\chi^2_{n-1}$ 분포가 오른쪽으로 치우쳐 있으므로 $\chi^2_{\alpha/2} < n-1 < \chi^2_{1-\alpha/2}$이다($\chi^2_{n-1}$의 평균 $n-1$이 위쪽 분위수에 더 가깝다). 따라서 $(n-1)/\chi^2_{\alpha/2} > 1$이고 $(n-1)/\chi^2_{1-\alpha/2} < 1$인데, 전자가 1을 넘는 양이 후자가 1에 못 미치는 양보다 커서 $U - S^2 > S^2 - L$이 된다. 예컨대 $\text{df} = 19$에서는 $19/8.907 - 1 = 1.13$ 대 $1 - 19/32.852 = 0.42$이다. 카이제곱분포의 오른쪽 꼬리가 길어 분모의 작은 카이제곱 값이 큰 $\sigma^2$ 값을 만들기 때문에 신뢰구간의 위쪽 꼬리가 더 멀리 뻗는다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> $\sigma^2$의 95% 신뢰구간이 $(9.25, 34.13)$이라면 모표준편차가 6보다 작다고 주장할 수 있는가? 답을 정당화하라.

</div>

??? success "풀이"

    $\sigma$의 95% 신뢰구간은 $(\sqrt{9.25}, \sqrt{34.13}) = (3.04, 5.84)$이다. 구간 전체가 6보다 아래이므로 95% 신뢰수준에서 자료는 $\sigma < 6$과 부합한다. 동등하게, 값 $\sigma = 6$(즉 $\sigma^2 = 36$)은 $\sigma^2$의 95% 신뢰구간 $(9.25, 34.13)$ 밖에 있다. 따라서 5% 유의수준에서 $\sigma < 6$이라고 결론지을 충분한 증거가 있다. $\square$

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff easy" title="쉬움"></span>
$\sigma^2$의 구간과 $\sigma$의 구간이 **같은 포함확률**을 갖는 이유를 설명하고 모의실험으로 확인하라.

</div>

??? success "풀이"
    **이유.** $g(t)=\sqrt t$는 $(0,\infty)$에서 **순증가**하는 함수다. 따라서

    $$
    L\le\sigma^2\le U \iff \sqrt L\le\sigma\le\sqrt U
    $$

    로 두 사건이 **논리적으로 동일**하다. 확률도 같을 수밖에 없다.

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(42)
    n, M = 20, 40_000
    x = rng.normal(0, 2, (M, n))                  # σ = 2, σ² = 4
    s2 = x.var(1, ddof=1)
    lo = (n - 1) * s2 / stats.chi2.ppf(0.975, n - 1)
    hi = (n - 1) * s2 / stats.chi2.ppf(0.025, n - 1)

    print(f"σ² = 4 포함확률  {np.mean((lo <= 4) & (4 <= hi)):.6f}")
    print(f"σ  = 2 포함확률  "
          f"{np.mean((np.sqrt(lo) <= 2) & (2 <= np.sqrt(hi))):.6f}")
    ```

    ```text
    σ² = 4 포함확률  0.948075
    σ  = 2 포함확률  0.948075
    ```

    **소수 여섯째 자리까지 같다.** 우연이 아니라 **같은 사건**이기 때문이다.

    **일반 원리.** 모수 $\theta$의 구간에 **단조함수** $g$를 적용하면 $g(\theta)$의 구간이 되고, 신뢰수준이 그대로 보존된다. 이것을 **변환 등변성**이라 한다.

    **주의할 점 세 가지.**

    1. **감소함수면 양끝이 뒤바뀐다.** $\theta$의 구간 $(L,U)$에 $g(t)=1/t$를 적용하면 $(1/U,\ 1/L)$이다.

    2. **점추정값은 옮겨 가지 않는다.** $E[S^2]=\sigma^2$이지만 $E[S]\ne\sigma$다. 앞서 본 $c_4$ 보정이 필요한 이유다. **구간은 옮겨 가고 불편성은 옮겨 가지 않는다.**

    3. **왈드 구간은 등변이 아니다.** $\hat\theta\pm z\widehat{\operatorname{SE}}$에 $g$를 적용한 것과, $g(\hat\theta)\pm z\widehat{\operatorname{SE}}_g$는 다르다. 델타법으로 표준오차를 다시 계산하면 결과가 달라진다. **추축량이나 우도비로 만든 구간만이 등변**이다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
분산 구간의 **폭 자체가 확률변수**임을 확인하라. $n=20$인 정규 표본에서 폭의 분포를 조사하고, 이것이 왜 중요한지 설명하라.

</div>

??? success "풀이"
    **폭은 $S^2$에 비례한다.**

    $$
    \text{폭}=(n-1)S^2\left(\frac1{\chi^2_{\nu,\alpha/2}}-\frac1{\chi^2_{\nu,1-\alpha/2}}\right)
    $$

    따라서 폭의 분포는 $S^2$의 분포, 즉 **스케일된 카이제곱**이다.

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(42)
    n, M = 20, 40_000
    x = rng.normal(0, 2, (M, n))                  # σ² = 4
    s2 = x.var(1, ddof=1)
    lo = (n - 1) * s2 / stats.chi2.ppf(0.975, n - 1)
    hi = (n - 1) * s2 / stats.chi2.ppf(0.025, n - 1)
    w = hi - lo

    q = np.percentile(w, [10, 25, 50, 75, 90])
    print(f"폭의 십분위 {q[0]:.3f}  사분위 {q[1]:.3f}  중앙값 {q[2]:.3f}  "
          f"사분위 {q[3]:.3f}  구분위 {q[4]:.3f}")
    print(f"평균 {w.mean():.3f}   90%/10% 비 {q[4] / q[0]:.2f}")
    ```

    ```text
    폭의 십분위 3.822  사분위 4.764  중앙값 6.015  사분위 7.448  구분위 8.927
    평균 6.231   90%/10% 비 2.34
    ```

    **폭이 3.8에서 8.9까지 흔들린다.** 열 번 중 한 번은 폭이 8.9를 넘고, 또 한 번은 3.8보다 좁다. **같은 $n$, 같은 모집단인데 구간의 길이가 두 배 넘게 차이 난다.**

    **왜 중요한가.**

    1. **표본크기 계획이 어렵다.** "폭을 2 이하로"라는 목표를 세워도, $S^2$이 크게 나오면 달성하지 못한다. **확률적 보장**("90% 확률로 폭이 2 이하")으로 계획을 세워야 하며, 이를 위한 공식이 따로 있다.

    2. **좁은 구간이 좋은 소식이 아니다.** 폭이 작다는 것은 $S^2$이 작았다는 뜻이고, 그것은 **모분산을 과소추정**했을 가능성이 높다는 뜻이다. 좁은 구간과 낮은 하한이 함께 나타난다.

    3. **평균 폭만 보고하면 오해를 부른다.** 방법 비교에서 "평균 폭 6.24"만 적으면 이 변동이 숨는다. 사분위나 백분위를 함께 적는다.

    **평균과의 대비.** $t$ 구간의 폭도 $S$에 비례하므로 확률변수지만, $S$는 $S^2$보다 변동이 작다(제곱근이 변동을 압축한다). 그래서 **분산 구간의 폭이 평균 구간의 폭보다 훨씬 불안정**하다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
비정규 자료에서 카이제곱 분산 구간의 포함확률이 **$n$을 키워도 회복되지 않음**을 확인하고, 왜 그런지 설명하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(42)
    M = 20_000
    print(f"{'n':>6s} {'포함확률':>9s}")
    for n in [20, 50, 100, 500, 2000]:
        y = rng.exponential(1.0, (M, n))          # 분산 1, 초과첨도 6
        s2 = y.var(1, ddof=1)
        lo = (n - 1) * s2 / stats.chi2.ppf(0.975, n - 1)
        hi = (n - 1) * s2 / stats.chi2.ppf(0.025, n - 1)
        print(f"{n:6d} {np.mean((lo <= 1) & (1 <= hi)):9.4f}")
    ```

    ```text
         n     포함확률
        20    0.7296
        50    0.6993
       100    0.6970
       500    0.6772
      2000    0.6747
    ```

    **회복되기는커녕 오히려 나빠진다.** $n=20$에서 0.73, $n=2000$에서 0.67이다.

    **왜 그런가.** 평균의 $t$ 구간은 중심극한정리 덕에 $n$이 커지면 회복된다. 분산은 다르다.

    $$
    \sqrt n\left(S^2-\sigma^2\right)\ \xrightarrow{d}\ N\!\left(0,\ \mu_4-\sigma^4\right)
    $$

    로 **점근분산이 4차 적률에 의존**한다. 정규에서는 $\mu_4-\sigma^4=2\sigma^4$인데, 지수분포에서는 $\mu_4=9\sigma^4$이라 $8\sigma^4$로 **네 배**다.

    한편 카이제곱 구간은 $(n-1)S^2/\sigma^2\sim\chi^2_{n-1}$을 쓰는데, 이 분포의 분산이 $2(n-1)$이므로 **$2\sigma^4/n$짜리 변동만 반영**한다. 실제 변동이 $8\sigma^4/n$이므로 **구간의 폭이 필요한 것의 절반**이다.

    **핵심.** $n\to\infty$에서 이 비율은 변하지 않는다. $\sqrt{2/8}=0.5$가 그대로 남는다. **카이제곱 구간은 비정규 아래에서 점근적으로도 타당하지 않다.** 극한 포함확률을 직접 계산하면

    ```python
    from scipy import stats
    import numpy as np

    for kappa, name in [(0, "정규"), (6, "지수"), (3, "t(6)쯤")]:
        r = np.sqrt(2 / (kappa + 2))              # 반영된 폭 / 필요한 폭
        print(f"{name:6s} 초과첨도 {kappa}  극한 포함확률 "
              f"{2 * stats.norm.cdf(1.96 * r) - 1:.4f}")
    ```

    ```text
    정규    초과첨도 0  극한 포함확률 0.9500
    지수    초과첨도 6  극한 포함확률 0.6729
    t(6)쯤  초과첨도 3  극한 포함확률 0.7849
    ```

    **극한값 0.673이 모의실험의 0.675와 잘 맞는다.**

    **교훈.** "$n$이 크면 괜찮다"는 직관이 **모든 절차에 적용되지 않는다.** 중심극한정리가 보호하는 것은 평균이지 분산이 아니다. 분산·첨도·상관계수처럼 고차 적률에 의존하는 양은 **분포가정을 따로 확인**해야 한다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
$n=50$인 지수분포 자료에서 카이제곱·보넷·부트스트랩 세 구간의 포함확률을 비교하라. 무엇을 권하겠는가?

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    def bonett(x, alpha=0.05):
        n = len(x)
        z = stats.norm.ppf(1 - alpha / 2)
        trim = 1 / (2 * np.sqrt(max(n - 4, 1)))
        m = stats.trim_mean(x, min(trim, 0.49))
        xb = x.mean()
        g4 = n * np.sum((x - m)**4) / np.sum((x - xb)**2)**2
        se = np.sqrt((g4 - (n - 3) / n) / (n - 1))
        c = n / (n - z)
        s2 = x.var(ddof=1)
        return c * s2 * np.exp(-z * se), c * s2 * np.exp(z * se)

    rng = np.random.default_rng(13)
    n, M, B = 50, 3_000, 999
    hc = hb = hp = 0
    for _ in range(M):
        x = rng.exponential(1.0, n)               # 참 분산 1
        s2 = x.var(ddof=1)
        lo = (n - 1) * s2 / stats.chi2.ppf(0.975, n - 1)
        hi = (n - 1) * s2 / stats.chi2.ppf(0.025, n - 1)
        hc += (lo <= 1 <= hi)
        lb, hb_ = bonett(x)
        hb += (lb <= 1 <= hb_)
        idx = rng.integers(0, n, (B, n))
        star = x[idx].var(1, ddof=1)
        l, h = np.percentile(star, [2.5, 97.5])
        hp += (l <= 1 <= h)
    print(f"카이제곱 {hc / M:.4f}   보넷 {hb / M:.4f}   "
          f"부트스트랩 백분위 {hp / M:.4f}")
    ```

    ```text
    카이제곱 0.7113   보넷 0.8813   부트스트랩 백분위 0.8073
    ```

    **순위.** 보넷(0.881) > 부트스트랩 백분위(0.807) > 카이제곱(0.711).

    **부트스트랩이 기대만 못한 이유.** 백분위 부트스트랩은 **일차정확**($O(n^{-1/2})$ 오차)이라 치우친 통계량에서 수렴이 느리다. $S^2$의 표집분포는 지수분포 아래에서 크게 오른쪽으로 치우쳐 있고, 백분위법은 그 치우침을 **반대 방향으로** 반영하는 결함이 있다.

    **개선책.** BCa나 부트스트랩-$t$를 쓰면 **이차정확**($O(n^{-1})$)이 되어 훨씬 낫다. 가속상수 $a$가 왜도를 보정한다.

    **권고.**

    | 상황 | 권장 |
    |---|---|
    | 정규성이 확실 | 카이제곱(가장 효율적) |
    | 정규성이 의심 | **보넷** 또는 BCa 부트스트랩 |
    | 분포족을 알고 있음 | 그 족의 모수를 추정(가장 효율적) |
    | $n<20$ | 어느 것도 믿기 어렵다 — 결과를 조심스럽게 |

    **더 근본적인 질문.** 지수분포 자료에서 "분산"이 정말 관심사인가? 지수분포는 $\sigma=\mu$이므로 평균만 추정하면 분산이 따라온다. **분포를 모형화하는 것이 요약통계를 강건하게 추정하는 것보다 나은 경우가 많다.**

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
분산 구간의 모의실험을 설계할 때 쓸 **자료생성 분포 목록**을 만들고, 각각에서 참 분산이 얼마인지 정리하라. 설계상 주의점도 함께 적어라.

</div>

??? success "풀이"
    **분포 목록.** 첨도를 넓게 훑되 참 분산을 아는 것들로 고른다.

    ```python
    import numpy as np
    from scipy import stats

    rows = [
        ("균등(0,1)",        1 / 12,           -1.2),
        ("정규(0,1)",        1.0,               0.0),
        ("로지스틱(0,1)",     np.pi**2 / 3,      1.2),
        ("t(9) / √(9/7)",   1.0,               1.2),
        ("지수(1)",          1.0,               6.0),
        ("t(5) / √(5/3)",   1.0,               6.0),
        ("감마(2,1)",        2.0,               3.0),
        ("로그정규(0,1)",     np.exp(2) - np.exp(1), 110.9),
        ("오염 0.9N(0,1)+0.1N(0,9)", 0.9 + 0.1 * 9, 5.3),
    ]
    print(f"{'분포':26s} {'참 분산':>10s} {'초과첨도':>9s}")
    for name, var, kur in rows:
        print(f"{name:26s} {var:10.4f} {kur:9.1f}")
    ```

    ```text
    분포                            참 분산      초과첨도
    균등(0,1)                      0.0833      -1.2
    정규(0,1)                      1.0000       0.0
    로지스틱(0,1)                    3.2899       1.2
    t(9) / √(9/7)                 1.0000       1.2
    지수(1)                        1.0000       6.0
    t(5) / √(5/3)                 1.0000       6.0
    감마(2,1)                      2.0000       3.0
    로그정규(0,1)                    4.6708     110.9
    오염 0.9N(0,1)+0.1N(0,9)       1.8000       5.3
    ```

    **설계상 주의점.**

    1. **분산을 1로 정규화한다.** 분포마다 참 분산이 다르면 폭을 비교할 수 없다. $X/\sigma$로 표준화하면 **포함확률은 그대로, 폭은 비교 가능**해진다.

    2. **첨도가 지표다.** 위 표에서 보듯 **초과첨도가 같으면 포함확률도 거의 같다.** 지수와 $t_5$가 둘 다 6이고, 실제로 카이제곱 구간의 포함확률이 비슷하다. 왜도는 상대적으로 덜 중요하다.

    3. **첨도가 존재하지 않는 분포도 넣는다.** $t_4$ 이하는 4차 적률이 무한대라 점근이론이 아예 성립하지 않는다. 이런 극단도 한 번 확인해 둘 가치가 있다.

    4. **로그정규는 극단적이다.** 초과첨도 110.9로, 어떤 방법도 살아남지 못한다. **"이 방법은 로그정규에서도 잘 된다"는 주장은 의심**해야 한다.

    5. **오염 모형을 반드시 넣는다.** 실무의 비정규성은 매끄러운 두꺼운 꼬리보다 **소수의 이상치** 형태인 경우가 많다. 혼합분포가 그것을 흉내 낸다.

    6. **$n$의 격자.** 분산은 $n$에 민감하므로 $n\in\{10,20,50,100,500\}$ 정도를 훑는다. 앞 문제에서 보았듯 **큰 $n$에서 오히려 나빠질 수 있어**, 큰 $n$을 빼면 잘못된 안도감을 준다.

    7. **평가 지표.** 포함확률, 평균 폭, **좌우 누락의 균형**(구간이 참값 아래인지 위인지). 치우친 분포에서는 누락이 한쪽으로 쏠린다.

    8. **참 분산을 코드에서 계산하지 말고 손으로 확인한다.** 표본에서 추정한 값을 "참값"으로 쓰면 모의실험이 무의미해진다. 흔한 버그다.

---

## 정리하며

분산 구간의 포함확률은 **정규성이 있느냐 없느냐로 갈린다.**

- **정규 자료에서는 명목값을 정확히 달성한다.** $(n-1)S^2/\sigma^2\sim\chi^2_{n-1}$ 이 정확한 결과이므로 당연하다.
- **정규성이 깨지면 크게 어긋난다.** 치우치거나 꼬리가 두꺼운 모집단에서 포함확률이 명목값에 한참 못 미친다. 4장 연습문제에서 로그정규 자료의 실제 포함률이 $42\%$, 지수 자료가 $72\%$ 였다.
- **$n$ 을 늘려도 낫지 않는다.** 평균의 $t$ 구간은 중심극한정리 덕분에 비정규성이 씻겨 나가지만, 분산 구간은 **$n\to\infty$ 에서도 잘못된 분포를 쓰고 있다.** 이 차이가 결정적이다.
- **원인은 4차적률이다.** $S^2$ 의 실제 변동은 $\mu_4-\sigma^4$ 에 달려 있는데 카이제곱 구간은 $\gamma_2=0$ 을 가정해 $2\sigma^4$ 만 반영한다.
- **처방.** 먼저 첨도를 재고, $\hat\gamma_2$ 가 $1$ 을 넘으면 부트스트랩이나 첨도 보정 구간으로 옮긴다.

다음 절부터 **계산 실습**으로 넘어간다. 원자료나 요약통계량에서 구간을 실제로 뽑아내는 코드를 다룬다.
