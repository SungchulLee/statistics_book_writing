# 평균 차이 신뢰구간 모의실험

## 개요

독립인 두 모집단의 평균을 비교할 때는 $\Delta = \mu_1 - \mu_2$의 신뢰구간을 구성한다. 이 페이지에서는 네 가지 접근 — Welch의 $t$-구간(권장되는 기본값), 합동 $t$-구간, 그리고 $z$에 기반한 두 변형 — 의 포함확률을 모의실험한다. 모분산이 다를 수 있을 때 왜 Welch 방법이 선호되는지 보여준다.

## 네 가지 구간 방법

### Welch의 t-구간 (권장 기본값)

표본분산이 $s_1^2$, $s_2^2$이고 크기가 $n_1$, $n_2$인 독립표본에 대해:

$$
(\bar{X}_1 - \bar{X}_2) \pm t_{\alpha/2,\,\nu} \cdot \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}
$$

여기서 Satterthwaite 자유도는

$$
\nu = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}}
$$

Welch 방법은 등분산을 가정하지 **않으며** 이분산성에 로버스트하다.

### 합동 t-구간

$\sigma_1^2 = \sigma_2^2$을 가정한다. 합동분산은

$$
s_p^2 = \frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}
$$

이고 구간은

$$
(\bar{X}_1 - \bar{X}_2) \pm t_{\alpha/2,\,n_1+n_2-2} \cdot s_p\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}
$$

### 분산을 아는 z-구간

$\sigma_1^2$과 $\sigma_2^2$을 알 때:

$$
(\bar{X}_1 - \bar{X}_2) \pm z_{\alpha/2} \cdot \sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}
$$

### s를 대입한 z-구간

표본표준편차를 정규 임계값과 함께 쓰는 대표본 근사이다:

$$
(\bar{X}_1 - \bar{X}_2) \pm z_{\alpha/2} \cdot \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}
$$

<div class="codebox" markdown>

#### 예제 1. 두 평균 차이 구간 모의실험 { .eg }

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm

np.random.seed(42)          # 아래 출력과 그림을 재현하려면 고정한다

n_simulations = 100
# 표본크기도 분산도 서로 다르게 잡았다. 이런 설정에서 합동 t가 무너지고
# Welch가 버티는지 보려는 것이다.
n1, n2 = 12, 10
mu1, mu2 = 0.0, 0.5
sigma1, sigma2 = 1.0, 1.5
alpha = 0.05
method = "welch"  # 'welch' | 'pooled' | 'z_known' | 'z_plugin'

delta_true = mu1 - mu2
lowers = np.empty(n_simulations)
uppers = np.empty(n_simulations)
centers = np.empty(n_simulations)

for i in range(n_simulations):
    x = np.random.normal(mu1, sigma1, n1)
    y = np.random.normal(mu2, sigma2, n2)
    xbar, ybar = x.mean(), y.mean()
    s1, s2 = x.std(ddof=1), y.std(ddof=1)
    diff_hat = xbar - ybar
    centers[i] = diff_hat

    if method == "welch":
        se = np.sqrt(s1**2 / n1 + s2**2 / n2)
        num = (s1**2 / n1 + s2**2 / n2)**2
        den = (s1**2 / n1)**2 / (n1 - 1) + (s2**2 / n2)**2 / (n2 - 1)
        df = num / den
        crit = t.ppf(1 - alpha / 2, df=df)
    elif method == "pooled":
        df = n1 + n2 - 2
        sp2 = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / df
        se = np.sqrt(sp2 * (1 / n1 + 1 / n2))
        crit = t.ppf(1 - alpha / 2, df=df)
    elif method == "z_known":
        se = np.sqrt(sigma1**2 / n1 + sigma2**2 / n2)
        crit = norm.ppf(1 - alpha / 2)
    else:  # z_plugin
        se = np.sqrt(s1**2 / n1 + s2**2 / n2)
        crit = norm.ppf(1 - alpha / 2)

    lowers[i] = diff_hat - crit * se
    uppers[i] = diff_hat + crit * se

covered = (lowers <= delta_true) & (delta_true <= uppers)
coverage_pct = 100.0 * covered.mean()
print(f"{method} coverage: {coverage_pct:.1f}%")
```

출력:

```
welch coverage: 97.0%
```

같은 자료에 `method`만 바꾸면 합동 $t$ 95%, $z$-known 94%, $z$-plugin 93%가 나온다. 다만 100회짜리 모의실험의 표준오차가 2.2%p나 되므로 이 차이를 방법의 우열로 읽으면 안 된다. 방법 사이의 진짜 차이를 보려면 아래 연습문제 3처럼 10,000회가 필요하다.

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

ax.axvline(delta_true, linestyle="--", linewidth=1.5)
n_fail = int((~covered).sum())
ax.set_title(f"{n_simulations} {method} CIs | n1={n1}, n2={n2}, CL=95%")
ax.set_yticks([])
ax.set_xlabel("Difference of means")
plt.tight_layout()
plt.show()
```

![100 welch CIs | n1=12, n2=10, CL=95%](./img/ci_diff_means_sim_114.png)

참값은 $\mu_1 - \mu_2 = -0.5$(세로 점선)이다. 구간의 폭이 1.56에서 4.02까지 두 배 넘게 널을 뛰는데, $s_1$과 $s_2$ 두 개가 동시에 흔들리는 데다 $n_2 = 10$으로 작아 그 흔들림이 크기 때문이다.

100개 중 **89개가 0을 담고 있다**는 점도 눈여겨볼 만하다. 참 차이가 분명히 존재하는데도($-0.5$) 표본이 작아 "차이가 없다"는 값을 열에 아홉은 배제하지 못한다. 9장의 용어로 말하면 검정력이 낮은 상황이다. 구간이 참값을 잘 담는 것과 유용한 결론을 주는 것은 별개다.

</div>

## 해석

- **Welch의 $t$-구간**은 모분산이 같든 다르든 명목 포함확률을 달성한다. 이표본 문제의 권장 기본값이다.
- **합동 $t$-구간**은 $\sigma_1^2 = \sigma_2^2$일 때 잘 작동하지만 이 가정이 깨지면, 특히 표본크기도 다르면 포함확률이 부족하거나 과할 수 있다.
- **$z$-known** 구간은 정확하지만 두 모분산을 모두 알아야 하며 그런 경우는 드물다.
- **$z$-plugin** 구간은 $z$ 공식에 표본분산을 대입한다. 작은 표본에서는 포함확률이 부족하지만 $n_1, n_2 \to \infty$이면 올바른 수준으로 수렴한다.
- $\sigma_1 \ne \sigma_2$이고 $n_1 \ne n_2$이면 합동 구간이 상당히 관대해지거나 보수적이 될 수 있는 반면, Welch 방법은 Satterthwaite 자유도로 적응한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> 두 집단의 요약통계량이 다음과 같다: $n_1 = 15$, $\bar{x}_1 = 78$, $s_1 = 10$; $n_2 = 20$, $\bar{x}_2 = 72$, $s_2 = 12$. $\mu_1 - \mu_2$의 95% Welch 신뢰구간을 구성하라.

</div>

??? success "풀이"

    점추정값은 $\bar{x}_1 - \bar{x}_2 = 6$이다. 표준오차는

    $$
    \text{SE} = \sqrt{\frac{10^2}{15} + \frac{12^2}{20}} = \sqrt{\frac{100}{15} + \frac{144}{20}} = \sqrt{6.667 + 7.200} = \sqrt{13.867} = 3.724
    $$

    Satterthwaite 자유도:

    $$
    \nu = \frac{(6.667 + 7.200)^2}{\frac{6.667^2}{14} + \frac{7.200^2}{19}} = \frac{192.29}{\frac{44.44}{14} + \frac{51.84}{19}} = \frac{192.29}{3.174 + 2.728} = \frac{192.29}{5.903} \approx 32.6
    $$

    $\nu \approx 32.6$이고 $t_{0.025,32.6} \approx 2.036$이므로:

    $$
    6 \pm 2.036 \times 3.724 = 6 \pm 7.58
    $$

    95% 신뢰구간은 $(-1.58, 13.58)$이다. 구간이 0을 포함하므로 5% 수준에서 두 평균이 다르다고 결론지을 수 없다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span> 근사적인 $t$ 추축량의 처음 두 적률을 $t_\nu$ 분포의 적률에 맞추어 Satterthwaite 자유도를 유도하라.

</div>

??? success "풀이"

    분산이 다른 이표본 모형에서 추축량은

    $$
    T = \frac{(\bar{X}_1 - \bar{X}_2) - (\mu_1 - \mu_2)}{\sqrt{S_1^2/n_1 + S_2^2/n_2}}
    $$

    분모에는 독립인 두 카이제곱 확률변수의 가중합이 들어 있다. $V = S_1^2/n_1 + S_2^2/n_2$라 하자. 적률 맞추기로 $V$를 $c \cdot W$($W \sim \chi^2_\nu / \nu$, $\nu$는 유효 자유도)로 근사한다.

    $V$의 처음 두 적률을 맞추면:

    - $E[V] = \sigma_1^2/n_1 + \sigma_2^2/n_2$
    - $\operatorname{Var}(V) = \frac{2\sigma_1^4}{n_1^2(n_1-1)} + \frac{2\sigma_2^4}{n_2^2(n_2-1)}$

    축척된 $\chi^2_\nu$에 대해 $W = c \chi^2_\nu/\nu$이면 $E[W] = c$이고 $\operatorname{Var}(W) = 2c^2/\nu$이다. $c = E[V]$로 두고 분산을 맞추면:

    $$
    \frac{2(E[V])^2}{\nu} = \operatorname{Var}(V)
    \quad\Longrightarrow\quad
    \nu = \frac{2(E[V])^2}{\operatorname{Var}(V)} = \frac{(\sigma_1^2/n_1 + \sigma_2^2/n_2)^2}{\frac{\sigma_1^4}{n_1^2(n_1-1)} + \frac{\sigma_2^4}{n_2^2(n_2-1)}}
    $$

    $\sigma_i^2$을 $s_i^2$으로 바꾸면 실무에서 쓰는 Satterthwaite 공식이 된다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> $\sigma_1^2 = \sigma_2^2 = \sigma^2$이고 $n_1 = n_2 = n$일 때 Satterthwaite 자유도가 합동 자유도 $n_1 + n_2 - 2$로 간단해짐을 보여라. 분산이 같더라도 표본크기가 다르면 어떻게 되는가?

</div>

??? success "풀이"

    Satterthwaite 공식에 $\sigma_1^2 = \sigma_2^2 = \sigma^2$을 대입하면:

    $$
    \nu = \frac{(\sigma^2/n_1 + \sigma^2/n_2)^2}{\frac{\sigma^4}{n_1^2(n_1-1)} + \frac{\sigma^4}{n_2^2(n_2-1)}}
    = \frac{(1/n_1 + 1/n_2)^2}{\frac{1}{n_1^2(n_1-1)} + \frac{1}{n_2^2(n_2-1)}}
    $$

    $\sigma^4$이 약분된다. $n_1 = n_2 = n$이면:

    $$
    \nu = \frac{(2/n)^2}{2/[n^2(n-1)]} = \frac{4}{n^2} \cdot \frac{n^2(n-1)}{2} = 2(n-1) = n_1 + n_2 - 2
    $$

    즉 표본크기가 같으면 Satterthwaite 자유도가 정확히 합동 자유도와 일치한다.

    표본크기가 다르면 분산이 같더라도 이 등식은 성립하지 않는다. 예를 들어 $n_1 = 10$, $n_2 = 20$이면

    $$
    \nu = \frac{(0.1 + 0.05)^2}{\frac{0.01}{9} + \frac{0.0025}{19}} = \frac{0.0225}{0.0012427} \approx 18.1
    $$

    로 $n_1 + n_2 - 2 = 28$보다 작다. 등분산일 때 합동 방법이 자유도를 더 많이 쓰므로 약간 더 효율적인 것이며, Welch는 그 대가로 등분산 가정에서 벗어날 자유를 얻는다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> $\sigma_1 = 1$, $\sigma_2 = 3$, $n_1 = n_2 = 10$일 때 Welch 구간과 합동 구간의 포함확률을 비교하는 모의실험을 설계하라. 10,000회 반복하고 결과를 보고하라.

</div>

??? success "풀이"

    ```python
    from scipy.stats import t as t_dist, norm
    np.random.seed(0)
    n1 = n2 = 10
    sigma1, sigma2 = 1.0, 3.0
    mu1 = mu2 = 0.0
    delta = 0.0
    n_sim = 10_000
    alpha = 0.05
    welch_cov = pooled_cov = 0
    for _ in range(n_sim):
        x = np.random.normal(mu1, sigma1, n1)
        y = np.random.normal(mu2, sigma2, n2)
        s1, s2 = x.std(ddof=1), y.std(ddof=1)
        diff = x.mean() - y.mean()
        # Welch 방법 — 두 분산이 다를 수 있다고 본다
        se_w = np.sqrt(s1**2/n1 + s2**2/n2)
        num = (s1**2/n1 + s2**2/n2)**2
        den = (s1**2/n1)**2/9 + (s2**2/n2)**2/9
        df_w = num/den
        crit_w = t_dist.ppf(0.975, df_w)
        if diff - crit_w*se_w <= delta <= diff + crit_w*se_w:
            welch_cov += 1
        # 합동 방법 — 두 분산이 같다고 본다
        sp2 = (9*s1**2 + 9*s2**2)/18
        se_p = np.sqrt(sp2*(1/10+1/10))
        crit_p = t_dist.ppf(0.975, 18)
        if diff - crit_p*se_p <= delta <= diff + crit_p*se_p:
            pooled_cov += 1
    print(f"Welch: {100*welch_cov/n_sim:.1f}%")
    print(f"Pooled: {100*pooled_cov/n_sim:.1f}%")
    ```

    출력:

    ```
    Welch: 95.0%
    Pooled: 94.3%
    ```

    Welch는 명목값에 정확히 맞고 합동은 0.7%p 부족하다. 합동 구간이 등분산을 가정하는데 실제로는 $\sigma_2/\sigma_1 = 3$이기 때문이다.

    차이가 이 정도로 작은 것은 $n_1 = n_2$이기 때문이다. 표본크기가 같으면 $s_p^2$의 가중치가 반반이어서 $\bar X_1 - \bar X_2$의 참 분산 $\sigma_1^2/n + \sigma_2^2/n$을 우연히 잘 맞힌다. 표본크기까지 어긋나면 왜곡이 훨씬 커진다. 연습문제 5가 그 방향을 다룬다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> 표본크기와 분산의 관계에 따라 합동 $t$-구간이 관대해지기도 하고 보수적이 되기도 하는 이유를 설명하라.

</div>

??? success "풀이"

    합동 구간은 공통 분산추정값으로 $s_p^2 = [(n_1-1)s_1^2 + (n_2-1)s_2^2]/(n_1+n_2-2)$를 쓴다. $\sigma_1^2 \ne \sigma_2^2$일 때:

    - **큰** 표본이 분산이 **큰** 모집단에서 나오면 $s_p^2$이 $\bar{X}_1 - \bar{X}_2$의 유효 분산을 과대추정하여 신뢰구간이 **너무 넓어진다**(보수적, 포함확률 $> 1-\alpha$).
    - **큰** 표본이 분산이 **작은** 모집단에서 나오면 $s_p^2$이 유효 분산을 과소추정하여 신뢰구간이 **너무 좁아진다**(관대함, 포함확률 $< 1-\alpha$).

    이런 비대칭은 합동 추정값이 $s_1^2$과 $s_2^2$을 표본크기를 따라가는 자유도 $(n_i - 1)$로 가중하기 때문에 생긴다. $\bar{X}_1 - \bar{X}_2$의 실제 분산은 $\sigma_1^2/n_1 + \sigma_2^2/n_2$로, 분산과 표본크기가 어떻게 짝지어지는지에 달려 있다. Welch 방법은 두 분산을 따로 추정하고 그에 맞게 자유도를 조정하여 이 문제를 피한다. $\square$

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
합동 $t$ 구간과 웰치 구간의 포함확률을 **여러 $(n_1,n_2,\sigma_1,\sigma_2)$ 조합**에서 비교하라. 합동 구간이 언제 관대하고 언제 보수적인지 규칙을 찾아라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(17)
    M = 20_000
    print(f"{'n1':>4s} {'n2':>4s} {'σ1':>4s} {'σ2':>4s} "
          f"{'합동':>8s} {'웰치':>8s}")
    cases = [(10, 10, 1, 3), (10, 30, 1, 3), (30, 10, 1, 3),
             (10, 30, 3, 1), (20, 20, 1, 1), (5, 50, 1, 4)]
    for n1, n2, s1, s2 in cases:
        x = rng.normal(0, s1, (M, n1))
        y = rng.normal(0, s2, (M, n2))
        v1, v2 = x.var(1, ddof=1), y.var(1, ddof=1)
        d = x.mean(1) - y.mean(1)

        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        hp = stats.t.ppf(0.975, n1 + n2 - 2) * np.sqrt(sp2 * (1 / n1 + 1 / n2))

        se = np.sqrt(v1 / n1 + v2 / n2)
        df = (v1 / n1 + v2 / n2)**2 / ((v1 / n1)**2 / (n1 - 1)
                                       + (v2 / n2)**2 / (n2 - 1))
        hw = stats.t.ppf(0.975, df) * se

        print(f"{n1:4d} {n2:4d} {s1:4d} {s2:4d} "
              f"{np.mean(np.abs(d) <= hp):8.4f} {np.mean(np.abs(d) <= hw):8.4f}")
    ```

    ```text
      n1   n2   σ1   σ2       합동       웰치
      10   10    1    3   0.9417   0.9498
      10   30    1    3   0.9961   0.9528
      30   10    1    3   0.7898   0.9494
      10   30    3    1   0.7865   0.9465
      20   20    1    1   0.9528   0.9535
       5   50    1    4   1.0000   0.9496
    ```

    **웰치는 모든 경우에 0.95 근처다.** 0.9465~0.9535.

    **합동 구간의 규칙이 분명하다.**

    | 상황 | 합동 구간 | 방향 |
    |---|---|---|
    | $n_1=n_2$, $\sigma$ 다름 | 0.9417 | 약간 관대 |
    | **작은 표본이 큰 분산** | 0.9961, 1.0000 | **지나치게 보수적** |
    | **큰 표본이 큰 분산** | 0.7898, 0.7865 | **심각하게 관대** |
    | $\sigma$ 같음 | 0.9528 | 정확 |

    **규칙 한 줄.** **분산이 큰 쪽의 표본이 크면 합동 구간이 관대해지고(위험), 작으면 보수적이 된다.**

    **왜 그런가.** 합동분산 $S_p^2$은 자유도로 가중한다.

    $$
    S_p^2=\frac{(n_1-1)S_1^2+(n_2-1)S_2^2}{n_1+n_2-2}
    $$

    반면 $\bar X_1-\bar X_2$의 참 분산은 $\sigma_1^2/n_1+\sigma_2^2/n_2$로 **표본크기의 역수**로 가중한다. 두 가중이 반대 방향이다.

    - $n_1=30$, $\sigma_1=1$, $n_2=10$, $\sigma_2=3$이면 참 분산은 $1/30+9/10=0.933$인데, $S_p^2(1/n_1+1/n_2)\approx3.0\times0.133=0.40$으로 **절반 이하**를 쓴다. 구간이 너무 좁아 포함확률이 0.79로 무너진다.

    **$n_1=n_2$이면 안전하다.** 그때 $S_p^2(2/n)=(S_1^2+S_2^2)/n$이 되어 웰치와 분자가 같아진다. 자유도만 다르고, 그 차이가 작다(0.9417 대 0.9498).

    **결론.** **웰치를 기본값으로 쓴다.** 등분산이 확실하고 $n_1=n_2$일 때만 합동 구간이 약간 유리하며, 그 이득은 무시할 만하다. R의 `t.test`가 웰치를 기본으로 삼는 이유다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
등분산 **사전검정**(레빈 검정 등)을 해서 합동/웰치를 고르는 2단계 절차의 문제를 논하라.

</div>

??? success "풀이"
    **흔한 관행.** "레빈 검정이 유의하지 않으면 합동 $t$, 유의하면 웰치"를 쓴다. 통계 소프트웨어의 출력이 이를 부추긴다.

    **문제 1 — 조건부 절차의 수준이 명목과 다르다.** 최종 절차는 "자료에 따라 선택된 방법"이므로, 그 표집분포는 두 절차 중 어느 것도 아니다. 전체 제1종 오류율이 5%를 벗어난다.

    **문제 2 — 사전검정의 검정력이 낮다.** 등분산 검정은 $n$이 작을 때 힘이 없다. $n_1=n_2=10$에서 $\sigma_2/\sigma_1=2$여도 레빈 검정이 이를 잡아낼 확률은 절반이 안 된다. **가장 필요한 상황에서 작동하지 않는다.**

    **문제 3 — $F$ 검정은 정규성에 극도로 민감하다.** 앞서 보았듯 분산에 관한 절차는 첨도에 취약하다. 꼬리가 두꺼운 자료에서 $F$ 검정이 등분산을 기각하면, 그것이 분산 차이 때문인지 첨도 때문인지 알 수 없다.

    **문제 4 — 애초에 필요 없다.** 웰치가 등분산일 때도 거의 손해가 없다. 위 문제의 $(20,20,1,1)$ 행에서 합동 0.9528, 웰치 0.9535였다. **잃는 것이 없는데 왜 검정을 하는가.**

    **효율 손실의 크기.** $\sigma_1=\sigma_2$, $n_1=n_2=n$일 때 웰치의 자유도는 대략 $n-1$에서 $2n-2$ 사이에 있다. 자유도가 줄면 $t$ 임계값이 커지는데,

    ```python
    import numpy as np
    from scipy import stats

    print(f"{'n':>5s} {'합동 df':>8s} {'웰치 평균 df':>12s} {'폭 비':>8s}")
    rng = np.random.default_rng(1)
    for n in [5, 10, 20, 50]:
        x = rng.normal(0, 1, (20_000, n))
        y = rng.normal(0, 1, (20_000, n))
        v1, v2 = x.var(1, ddof=1), y.var(1, ddof=1)
        df = (v1 / n + v2 / n)**2 / ((v1 / n)**2 / (n - 1) + (v2 / n)**2 / (n - 1))
        tw = stats.t.ppf(0.975, df).mean()
        tp = stats.t.ppf(0.975, 2 * n - 2)
        print(f"{n:5d} {2 * n - 2:8d} {df.mean():12.1f} {tw / tp:8.4f}")
    ```

    ```text
        n   합동 df   웰치 평균 df     폭 비
        5        8         6.8   1.0373
       10       18        16.5   1.0074
       20       38        36.3   1.0017
       50       98        96.1   1.0003
    ```

    **$n=10$이면 웰치 구간이 0.7% 넓을 뿐**이고, $n=50$이면 0.03%다. $n=5$에서도 3.7%다. **이것이 웰치를 쓰는 전체 비용**이다.

    **권고.**

    1. **등분산 사전검정을 하지 않는다.**
    2. **언제나 웰치를 쓴다.** 이것이 현대적 표준이며, 여러 교과서와 논문이 명시적으로 권한다.
    3. **등분산 가정이 이론적으로 정당한 특수한 경우**(같은 측정기기, 같은 공정)에만 합동을 고려하고, 그때도 이득이 미미함을 안다.

    **일반 원리.** **자료를 보고 방법을 고르는 절차는 그 자체로 하나의 절차**이며, 구성요소의 성질을 물려받지 않는다. 정규성 검정 후 $t$/비모수를 고르는 관행도 같은 문제를 안고 있다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
웰치 구간이 **비정규성**에 얼마나 견디는지 모의실험으로 확인하라. 두 집단의 분포가 서로 다를 때 특히 주의할 점은 무엇인가?

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(4)
    M = 20_000

    def welch_cover(x, y, n1, n2, true=0.0):
        v1, v2 = x.var(1, ddof=1), y.var(1, ddof=1)
        se = np.sqrt(v1 / n1 + v2 / n2)
        df = (v1 / n1 + v2 / n2)**2 / ((v1 / n1)**2 / (n1 - 1)
                                       + (v2 / n2)**2 / (n2 - 1))
        h = stats.t.ppf(0.975, df) * se
        d = x.mean(1) - y.mean(1)
        below = np.mean(d + h < true)
        above = np.mean(d - h > true)
        return 1 - below - above, below, above

    cases = [
        ("정규 / 정규", lambda m: rng.normal(0, 1, m), lambda m: rng.normal(0, 1, m), 15, 15),
        ("지수 / 지수", lambda m: rng.exponential(1, m) - 1,
         lambda m: rng.exponential(1, m) - 1, 15, 15),
        ("지수 / 정규", lambda m: rng.exponential(1, m) - 1,
         lambda m: rng.normal(0, 1, m), 15, 15),
        ("지수 / 지수 (n=50)", lambda m: rng.exponential(1, m) - 1,
         lambda m: rng.exponential(1, m) - 1, 50, 50),
        ("t(3) / t(3)", lambda m: rng.standard_t(3, m),
         lambda m: rng.standard_t(3, m), 20, 20),
    ]
    for name, g1, g2, n1, n2 in cases:
        c, lo, hi = welch_cover(g1((M, n1)), g2((M, n2)), n1, n2)
        print(f"{name:20s} 포함 {c:.4f}   아래 누락 {lo:.4f}   위 누락 {hi:.4f}")
    ```

    ```text
    정규 / 정규          포함 0.9513   아래 누락 0.0239   위 누락 0.0248
    지수 / 지수          포함 0.9567   아래 누락 0.0209   위 누락 0.0224
    지수 / 정규          포함 0.9456   아래 누락 0.0406   위 누락 0.0138
    지수 / 지수 (n=50)    포함 0.9526   아래 누락 0.0242   위 누락 0.0233
    t(3) / t(3)         포함 0.9573   아래 누락 0.0222   위 누락 0.0205
    ```

    **포함확률은 대체로 잘 유지된다.** 최악이 0.9456이다. 앞서 분산 구간이 0.72로 무너진 것과 대조적이다.

    **그러나 꼬리의 균형이 깨진다.** "지수/정규" 조합에서

    - 아래 누락 4.1%, 위 누락 1.4%

    로 **한쪽이 세 배**다. 명목은 각각 2.5%여야 한다.

    **왜 그런가.** 두 집단의 **왜도가 다르면** 왜도가 상쇄되지 않는다. 같은 분포끼리면 $\bar X_1-\bar X_2$의 왜도가 서로 지워져 "지수/지수"에서 0.0209 대 0.0224로 균형이 잡힌다. **비정규성 자체보다 두 집단의 비대칭이 다른 것이 문제**다.

    **실무적 함의.**

    1. **양측 구간은 꽤 안전하다.** 두 꼬리의 오차가 부분적으로 상쇄된다.
    2. **단측 결론은 위험하다.** "처리군이 대조군보다 크다"를 주장할 때, 어느 쪽 꼬리인지에 따라 실제 오류율이 4%일 수도 1.4%일 수도 있다.
    3. **두 집단의 분포 모양을 비교**해야 한다. 히스토그램을 나란히 그려 왜도가 비슷한지 본다.

    **대처.** 왜도가 크게 다르면

    - **변환**으로 양쪽을 대칭화한다(같은 변환을 써야 한다).
    - **순열검정**이나 부트스트랩. 다만 순열검정은 $H_0$가 "두 분포가 같다"이므로, 분산이 다르면 평균 차이 검정으로 쓸 때 주의해야 한다.
    - **분포를 모형화**한다. 두 집단이 서로 다른 감마나 로그정규를 따른다면 그렇게 적합하는 것이 정직하다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
평균 차이 대신 **다른 대조**(비, 표준화 효과크기, 분위수 차이)에 관심이 있을 때 각각 구간을 어떻게 만드는지 정리하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(88)
    x = rng.lognormal(1.0, 0.5, 40)          # 집단 1
    y = rng.lognormal(0.7, 0.5, 45)          # 집단 2
    z = stats.norm.ppf(0.975)

    # 1) 평균 차이 (웰치)
    n1, n2 = len(x), len(y)
    v1, v2 = x.var(ddof=1), y.var(ddof=1)
    se = np.sqrt(v1 / n1 + v2 / n2)
    df = (v1 / n1 + v2 / n2)**2 / ((v1 / n1)**2 / (n1 - 1) + (v2 / n2)**2 / (n2 - 1))
    h = stats.t.ppf(0.975, df) * se
    print(f"평균 차이   {x.mean() - y.mean():7.4f}  "
          f"({x.mean() - y.mean() - h:7.4f}, {x.mean() - y.mean() + h:7.4f})")

    # 2) 로그 척도 차이 → 기하평균의 비
    lx, ly = np.log(x), np.log(y)
    sl = np.sqrt(lx.var(ddof=1) / n1 + ly.var(ddof=1) / n2)
    dfl = (lx.var(ddof=1) / n1 + ly.var(ddof=1) / n2)**2 / (
        (lx.var(ddof=1) / n1)**2 / (n1 - 1) + (ly.var(ddof=1) / n2)**2 / (n2 - 1))
    hl = stats.t.ppf(0.975, dfl) * sl
    dl = lx.mean() - ly.mean()
    print(f"기하평균 비 {np.exp(dl):7.4f}  "
          f"({np.exp(dl - hl):7.4f}, {np.exp(dl + hl):7.4f})")

    # 3) 코헨의 d (합동 표준편차 기준) + 부트스트랩 구간
    sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    dcoh = (x.mean() - y.mean()) / sp
    B = 20_000
    bx = x[rng.integers(0, n1, (B, n1))]
    by = y[rng.integers(0, n2, (B, n2))]
    spb = np.sqrt(((n1 - 1) * bx.var(1, ddof=1) + (n2 - 1) * by.var(1, ddof=1))
                  / (n1 + n2 - 2))
    db = (bx.mean(1) - by.mean(1)) / spb
    print(f"코헨의 d    {dcoh:7.4f}  "
          f"({np.percentile(db, 2.5):7.4f}, {np.percentile(db, 97.5):7.4f})")

    # 4) 중앙값 차이 (부트스트랩)
    mb = np.median(bx, 1) - np.median(by, 1)
    print(f"중앙값 차이 {np.median(x) - np.median(y):7.4f}  "
          f"({np.percentile(mb, 2.5):7.4f}, {np.percentile(mb, 97.5):7.4f})")
    ```

    ```text
    평균 차이    0.8968  ( 0.3329,  1.4608)
    기하평균 비  1.3590  ( 1.1250,  1.6418)
    코헨의 d     0.7099  ( 0.3063,  1.1280)
    중앙값 차이  0.6539  (-0.0641,  1.6675)
    ```

    **세 구간은 "차이 없음"을 배제하지만 중앙값 차이는 0을 담는다.** $n_1=40$, $n_2=45$에서 중앙값은 평균보다 표집변동이 커서 구간이 넓기 때문이다(폭 1.73 대 1.13). **어느 대조를 고르느냐가 결론을 바꿀 수 있다.**

    | 대조 | 무엇을 말하는가 | 언제 |
    |---|---|---|
    | 평균 차이 | 원 척도의 절대 차이 | 총량이 중요할 때 |
    | 기하평균 비 | **몇 배** 차이 | 곱셈적 구조, 치우친 자료 |
    | 코헨의 $d$ | 변동 대비 차이 | 척도가 임의적일 때, 메타분석 |
    | 중앙값 차이 | 전형적인 값의 차이 | 이상치가 있을 때 |

    **코헨의 $d$에 대한 경고.**

    1. **분모가 달라지면 값이 달라진다.** 합동 $S_p$, 대조군의 $S$, 잔차 표준편차 중 무엇을 쓰는지 명시해야 한다.
    2. **소표본에서 편향된다.** 헤지스의 보정 $g=d(1-3/(4(n_1+n_2)-9))$을 쓴다.
    3. **"0.2/0.5/0.8 = 작음/중간/큼"은 근거가 아니다.** 코헨 자신이 임시적 기준이라고 밝혔다. 분야의 맥락에서 해석해야 한다.
    4. **모집단의 이질성에 의존한다.** 같은 절대 효과라도 동질적인 집단에서는 $d$가 크게 나온다.

    **권고.** **원 척도의 차이를 우선 보고**하고, 효과크기는 보조로 덧붙인다. 척도에 실질적 의미가 있는데도 표준화 효과크기만 적으면 독자가 실무적 중요성을 판단할 수 없다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
두 집단 비교 모의실험을 설계할 때 **무엇을 격자로 훑어야 하는지** 정리하고, 흔히 빠뜨리는 조합을 지적하라.

</div>

??? success "풀이"
    **훑어야 할 축.**

    | 축 | 값 | 왜 |
    |---|---|---|
    | 표본크기 | $n\in\{5,10,20,50,100\}$ | 소표본에서 절차가 갈린다 |
    | **표본크기 비** | $n_1/n_2\in\{1,\ 2,\ 5\}$ | 불균형이 합동 구간을 무너뜨린다 |
    | 분산비 | $\sigma_1/\sigma_2\in\{1,\ 2,\ 4\}$ | 등분산 가정의 위배 정도 |
    | **분산비의 방향** | 큰 표본이 큰 분산 / 작은 분산 | **두 방향의 결과가 정반대** |
    | 분포 | 정규, 지수, $t_3$, 오염, 로그정규 | 왜도와 첨도 |
    | **두 집단의 분포 조합** | 같음 / 다름 | 왜도가 상쇄되는지 |
    | 참 차이 | 0(수준), 여러 값(검정력) | |
    | 신뢰수준 | 0.90, 0.95, 0.99 | 꼬리로 갈수록 근사가 나빠짐 |

    **흔히 빠뜨리는 조합 — 다섯.**

    1. **표본크기 비와 분산비의 교차.** $n_1=n_2$만 보면 합동 $t$가 멀쩡해 보인다. 앞서 본 대로 **불균형과 이분산이 함께** 있어야 문제가 드러난다. 게다가 방향까지 봐야 한다.

    2. **두 집단이 서로 다른 분포.** 둘 다 지수분포로 두면 왜도가 상쇄되어 절차가 실제보다 좋아 보인다. **한쪽만 치우친** 조합을 반드시 넣는다.

    3. **극단적 불균형.** $n_1=5$, $n_2=50$ 같은 경우. 실무에서 흔한데(희귀군 대 대조군) 모의실험에서는 빠지기 쉽다.

    4. **참 차이가 0이 아닌 경우.** 포함확률만 보면 $\delta=0$으로 충분해 보이지만, **구간의 폭과 검정력**은 $\delta$에 따라 달라진다. 특히 비율이나 분산비처럼 모수공간에 경계가 있으면 그렇다.

    5. **작은 신뢰수준.** 95%만 확인하고 99%를 건너뛰면, 꼬리에서 근사가 무너지는 것을 놓친다.

    **평가 지표도 여럿이어야 한다.**

    - **포함확률** — 기본.
    - **평균 폭과 폭의 분포** — 효율.
    - **좌우 누락의 균형** — 단측 결론의 타당성.
    - **실패율** — 계산이 불가능한 경우(자유도 정의 안 됨, 분산 0 등).

    **결과 제시.** 조합이 많으므로 표보다 **그림**이 낫다. 가로축에 $n$, 세로축에 포함확률, 선을 분산비로 나누고, 패널을 분포로 나누는 격자 그림이 표준적이다.

    **보고의 정직성.** 훑은 격자를 **모두** 보고한다. 유리한 조합만 골라 실으면 앞서 경고한 모의실험판 $p$-해킹이다. 결과가 많으면 본문에 요약하고 부록에 전체를 싣는다.

---

## 정리하며

네 가지 방법의 포함확률을 재어 보면 **웰치가 기본값인 이유**가 분명해진다.

- **분산이 다르고 표본크기도 다르면 합동 $t$ 가 무너진다.** 특히 **작은 표본에 큰 분산이 붙은 경우**가 최악으로, 포함확률이 명목값보다 크게 낮아진다. 반대 조합에서는 지나치게 보수적이 된다.
- **웰치는 거의 모든 조합에서 명목값을 지킨다.** 새터스웨이트 자유도가 두 표본의 불균형을 흡수하기 때문이다.
- **분산이 실제로 같을 때 웰치의 손해는 미미하다.** 자유도를 조금 잃을 뿐이며, 그 대가로 가정 하나를 통째로 없앤다. **보험료가 싸다.**
- **$z$ 기반 변형은 소표본에서 부족하다.** $s$ 를 대입하고 $z$ 임계값을 쓰면 구간이 좁아져 포함확률이 떨어진다. 일표본에서 본 것과 같은 현상이다.
- **실무 권고는 단순하다. 언제나 웰치를 쓴다.** 등분산 검정으로 방법을 고르는 2단계 절차는 쓰지 않는다.

다음 절부터 **대응표본**으로 넘어간다. 두 측정이 같은 대상에서 나온 경우이며, 독립을 가정할 수 없다.
