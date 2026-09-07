# 재표집 방법 비교 (코드)

## 개요

이 페이지는 붓스트랩과 순열 재표집 방법을 나란히 비교한다. 붓스트랩 신뢰구간(정규, 백분위수, 기본, BCa)과 포함확률 모의실험, 이표본 및 대응 순열검정, 상관에 대한 순열검정, 그리고 붓스트랩 신뢰구간과 순열 $p$값의 직접 비교를 다룬다. 각 방법이 언제 왜 적절한지를 보이는 것이 목표이다.

## 붓스트랩 표준오차와 신뢰구간

표본 $x_1, \ldots, x_n$이 주어졌을 때 통계량 $\hat\theta$의 붓스트랩 표준오차는

$$
\widehat{\text{SE}}_{\text{boot}} = \sqrt{\frac{1}{B-1}\sum_{b=1}^{B}\bigl(\hat\theta^{*(b)} - \overline{\hat\theta^*}\bigr)^2}
$$

이다. 네 가지 신뢰구간 방법이 있다.

**정규 구간.** 붓스트랩 표준오차와 정규 분위수를 쓴다.

$$
\hat\theta \pm z_{1-\alpha/2}\cdot\widehat{\text{SE}}_{\text{boot}}
$$

**백분위수 구간.** 붓스트랩 분포의 분위수에서 직접 읽는다.

$$
\bigl[\hat\theta^*_{\alpha/2},\;\hat\theta^*_{1-\alpha/2}\bigr]
$$

**기본(추축) 구간.** 분위수를 $\hat\theta$에 대해 반사한다.

$$
\bigl[2\hat\theta - \hat\theta^*_{1-\alpha/2},\;2\hat\theta - \hat\theta^*_{\alpha/2}\bigr]
$$

**BCa 구간.** 잭나이프를 써서 편향($z_0$)과 가속($a$)을 보정한다.

$$
\alpha_j = \Phi\!\left(z_0 + \frac{z_0 + z_{\alpha_j}}{1 - a(z_0 + z_{\alpha_j})}\right)
$$

```python
import numpy as np
from scipy import stats

def bootstrap_ci_demo(data, B=10_000, alpha=0.05, rng=None):
    """Compute the normal, percentile, and basic bootstrap CIs for the mean."""
    rng = rng or np.random.default_rng(0)
    n = len(data)
    theta_hat = data.mean()
    z = stats.norm.ppf(1 - alpha / 2)

    boot_means = data[rng.integers(0, n, (B, n))].mean(axis=1)
    se_boot = boot_means.std(ddof=1)
    lo_q, hi_q = np.percentile(boot_means, [100*alpha/2, 100*(1 - alpha/2)])

    return {
        "normal":     (theta_hat - z*se_boot, theta_hat + z*se_boot),
        "percentile": (lo_q, hi_q),
        "basic":      (2*theta_hat - hi_q, 2*theta_hat - lo_q),
    }
```

## 붓스트랩 포함확률 모의실험

포함확률 모의실험은 명목 신뢰수준이 참 모수를 포함하는 구간의 실제 비율과 일치하는지 확인한다. $N$번의 모의실험 각각에서

1. 알려진 모집단에서 새 표본을 뽑는다.
2. 붓스트랩 신뢰구간을 만든다.
3. 참 모수가 그 안에 들어가는지 확인한다.

경험적 포함확률은

$$
\widehat{\text{coverage}} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\!\bigl(\theta \in \text{CI}_i\bigr)
$$

이다.

$\text{Exp}(3)$에서 $n = 30$을 뽑은 결과($M = 3{,}000$, $B = 2{,}000$):

| 방법 | 포함확률 |
|:---|---:|
| 정규 | 0.912 |
| 백분위수 | 0.913 |
| 기본 | 0.902 |
| BCa | 0.922 |
| $t$ 구간 | 0.925 |

**네 붓스트랩 방법 모두 명목값에 못 미친다.** $t$ 구간도 $0.925$에 그친다. 지수분포의 왜도가 $2$로 크기 때문이며, $n = 30$으로는 어떤 방법도 $0.95$를 달성하지 못한다.

## 이표본 순열검정

순열검정은 두 표본을 합치고 라벨을 섞어 각 순열에서 검정통계량을 계산한다.

```python
def permutation_test_two_sample(x, y, B=9999, stat_func=None, rng=None):
    """Two-sample permutation test; stat_func may be any two-sample statistic."""
    rng = rng or np.random.default_rng(0)
    if stat_func is None:
        stat_func = lambda a, b: a.mean() - b.mean()
    t_obs = stat_func(x, y)
    pooled = np.concatenate([x, y])
    m = len(x)

    count = 0
    for _ in range(B):
        p = rng.permutation(pooled)
        count += abs(stat_func(p[:m], p[m:])) >= abs(t_obs)
    return t_obs, (count + 1) / (B + 1)
```

처치군 대 대조군 자료에 적용하면 순열 $p$값이 대개 Welch $t$ 검정의 $p$값과 가깝다. **다만 분산이 다르고 표본이 불균형하면 그렇지 않다**(연습문제 2).

## 상관에 대한 순열검정

$H_0\colon \rho = 0$을 검정하려면 한 변수를 고정한 채 다른 변수를 순열한다.

$$
p = \frac{\#\bigl\{b : |r^{(\pi_b)}| \ge |r_{\text{obs}}|\bigr\} + 1}{B + 1}
$$

```python
def permutation_test_correlation(x, y, B=9999, rng=None):
    """Permutation test for the Pearson correlation."""
    rng = rng or np.random.default_rng(0)
    r_obs = np.corrcoef(x, y)[0, 1]
    count = 0
    for _ in range(B):
        count += abs(np.corrcoef(x, rng.permutation(y))[0, 1]) >= abs(r_obs)
    return r_obs, (count + 1) / (B + 1)
```

## 대응 순열검정 (부호 뒤집기)

대응자료 $(x_i, y_i)$에서 차이 $d_i = x_i - y_i$는 $H_0$ 아래에서 $0$에 대해 대칭이어야 한다. 부호를 무작위로 뒤집는다.

$$
T^{(\pi)} = \frac{1}{n}\sum_{i=1}^{n} s_i\,d_i, \qquad s_i \in \{-1, +1\} \text{ 균등}
$$

```python
def paired_permutation_test(x, y, B=9999, rng=None):
    """Paired permutation test by sign-flipping the differences."""
    rng = rng or np.random.default_rng(0)
    d = np.asarray(x) - np.asarray(y)
    t_obs = d.mean()
    signs = rng.choice([-1, 1], size=(B, len(d)))
    t_perm = (signs * d).mean(axis=1)
    return t_obs, ((np.abs(t_perm) >= abs(t_obs)).sum() + 1) / (B + 1)
```

## 붓스트랩과 순열: 나란히

두 재표집 전략은 서로 다른 질문에 답한다.

| 측면 | 붓스트랩 | 순열 |
|---|---|---|
| **목표** | 모수 또는 그 불확실성 추정 | 귀무가설 검정 |
| **산출물** | 신뢰구간 | $p$값 |
| **재표집** | 각 집단에서 복원추출 | 라벨을 비복원으로 섞기 |
| **가정** | 표본이 대표적 | $H_0$ 아래 교환가능성 |

같은 이표본 비교에 둘 다 적용했을 때, $0$을 제외하는 붓스트랩 신뢰구간과 같은 $\alpha$에서 기각하는 순열검정은 **대개** 일치한다. 어긋나는 경우는 연습문제 4에서 다룬다.

```python
def bootstrap_vs_permutation_comparison(x, y, B=9999, rng=None):
    """Compare a bootstrap CI for the difference with a permutation p-value."""
    rng = rng or np.random.default_rng(0)
    diff_obs = x.mean() - y.mean()

    # 각 집단을 따로 재표집한 신뢰구간
    bx = x[rng.integers(0, len(x), (B, len(x)))].mean(axis=1)
    by = y[rng.integers(0, len(y), (B, len(y)))].mean(axis=1)
    ci = np.percentile(bx - by, [2.5, 97.5])

    # 순열 p 값
    _, p_perm = permutation_test_two_sample(x, y, B=B, rng=rng)
    return diff_obs, ci, p_perm
```

## 해석

- **포함확률 모의실험**은 치우친 분포와 작은 $n$에서 붓스트랩 신뢰구간이 명목값에 못 미칠 수 있음을 보인다. BCa와 $t$ 구간이 명목수준에 조금 더 가깝다.
- **평균과 상관에 대한 순열검정**은 분포 가정이 성립할 때 모수적 대응물과 매우 가까운 $p$값을 낸다.
- **부호 뒤집기 검정**은 대응 $t$ 검정의 순열 대응물이며 차이가 정규가 아니어도 타당하다.
- **붓스트랩과 순열 접근은 서로를 보완한다.** 추정(신뢰구간, 표준오차)에는 붓스트랩을, 가설검정에는 순열검정을 쓴다.

## 연습문제

**연습문제 1.** 포함확률 모의실험을 $n = 30$ 대신 $n = 100$으로 실행하라. 표본크기를 늘리면 백분위수법의 포함확률이 어떻게 변하는가? 중심극한정리로 설명하라.

??? success "풀이"

    ```python
    for n in (30, 100):
        print(n, cover(n, M=3000, B=2000, scale=3.0))
    ```

    | 방법 | $n = 30$ | $n = 100$ | 개선 |
    |:---|---:|---:|---:|
    | 정규 | 0.912 | 0.931 | +0.019 |
    | 백분위수 | 0.913 | 0.934 | +0.021 |
    | 기본 | 0.902 | 0.929 | +0.027 |
    | BCa | 0.922 | 0.938 | +0.016 |
    | $t$ 구간 | 0.925 | 0.938 | +0.013 |

    **백분위수법의 포함확률이 $0.913$에서 $0.934$로 개선된다.** 중심극한정리에 의해 $n$이 크면 $\bar x$가 근사적으로 $N(\mu, \sigma^2/n)$이다. $\bar x^*$의 붓스트랩 분포도 같은 정규 모양을 따라가며 왜도가 줄어들므로, 백분위수 분위수가 참 표집 분위수의 좋은 근사가 된다.

    **그러나 $n = 100$에서도 $0.95$에 도달하지 못한다.** 이 점이 중요하다. 흔히 "$n$이 크면 괜찮다"고 하지만, 지수분포처럼 왜도가 $2$인 모집단에서는 $n = 100$도 충분하지 않다.

    **왜 그런가.** 편단측 오차를 생각하면 명확해진다. 왜도가 $\gamma$인 모집단에서 $t$ 통계량의 각 꼬리 오차는 대략

    $$
    \frac{\gamma}{6\sqrt{n}}(2z_{\alpha}^2 + 1)\phi(z_\alpha)
    $$

    크기이며 $O(n^{-1/2})$로만 줄어든다. $n$을 $30$에서 $100$으로 늘리면 오차가 $\sqrt{30/100} = 0.55$배로 줄 뿐이다. $0.95$에 도달하려면 $n$이 수백 이상 필요하다.

    **네 방법의 순위가 $n$에 무관하게 유지된다.** BCa $>$ 백분위수 $\approx$ 정규 $>$ 기본이다. 기본법이 가장 나쁜 것은 치우친 자료에서 반사가 잘못된 방향으로 작용하기 때문이다.

---

**연습문제 2.** `permutation_test_two_sample`을 원래의 평균차 대신 Welch $t$ 통계량을 쓰도록 수정하라. 분산이 다른 자료 $X \sim N(5, 1)$, $Y \sim N(5, 3^2)$에 $n_x = 20$, $n_y = 50$으로 두 버전을 적용하고 제1종 오류율과 검정력을 비교하라.

??? success "풀이"

    ```python
    import numpy as np
    rng = np.random.default_rng(9)

    def welch_t(a, b):
        return (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1)/len(a)
                                              + b.var(ddof=1)/len(b))

    def rate(shift, M=1500, B=999):
        raw = wel = 0
        for _ in range(M):
            x = rng.normal(5 + shift, 1, 20); y = rng.normal(5, 3, 50)
            z = np.concatenate([x, y])
            P = np.array([rng.permutation(z) for _ in range(B)])
            A, Bm = P[:, :20], P[:, 20:]
            d = A.mean(1) - Bm.mean(1); obs = x.mean() - y.mean()
            raw += ((np.abs(d) >= abs(obs)).sum() + 1)/(B+1) < 0.05
            w = d / np.sqrt(A.var(1, ddof=1)/20 + Bm.var(1, ddof=1)/50)
            wo = welch_t(x, y)
            wel += ((np.abs(w) >= abs(wo)).sum() + 1)/(B+1) < 0.05
        return round(raw/M, 3), round(wel/M, 3)
    ```

    | 상황 | 평균차 통계량 | Welch $t$ 통계량 |
    |:---|---:|---:|
    | 제1종 오류율 (이동 $= 0$) | **0.003** | 0.053 |
    | 검정력 (이동 $= 1.5$) | 0.591 | **0.857** |

    **평균차 버전의 제1종 오류율이 $0.003$이다.** 명목값의 $1/17$로 극도로 보수적이다. 이는 무해한 결함이 아니라 **검정력을 대가로 치른 것**이다.

    **왜 그런가.** [기초](../permutation/foundations.md) 연습문제 1에서 본 메커니즘이다. 여기서는 작은 집단($n_x = 20$)의 분산이 작고($\sigma = 1$) 큰 집단($n_y = 50$)의 분산이 크다($\sigma = 3$).

    - 실제 $\bar X - \bar Y$의 분산: $1/20 + 9/50 = 0.230$
    - 합친 분산: $\bar\sigma^2 = (20 \times 1 + 50 \times 9)/70 = 6.71$
    - 순열분포의 분산: $6.71 \times (1/20 + 1/50) = 0.470$

    순열 귀무분포가 실제보다 **두 배 넓다**. 관측된 차이가 이 과도하게 넓은 분포의 꼬리에 도달하기 어려워 거의 기각하지 못한다.

    **Welch 통계량이 문제를 완전히 해결한다.** 크기가 $0.053$으로 명목값을 지키고, 검정력이 $0.591 \to 0.857$로 **$45$% 향상된다**.

    각 순열에서 그 순열의 표본분산으로 표준화하므로, 순열된 두 집단의 분산이 뒤섞여도 분모가 함께 그 변화를 반영한다.

    !!! tip "실무 규칙"
        이표본 순열검정에서 **표본크기가 다르면 항상 스튜던트화 통계량을 쓴다.** 균형 설계($n_x = n_y$)에서는 두 버전이 사실상 같으므로 스튜던트화를 기본값으로 삼아도 잃을 것이 없다.

        SciPy의 `stats.permutation_test`는 통계량을 사용자가 지정하게 되어 있다. 기본 예제들이 평균차를 쓰지만, 불균형 자료에서는 Welch $t$를 넘겨야 한다.

---

**연습문제 3.** 대응 순열검정은 차이 $d_i$의 부호를 무작위로 뒤집는다. $n$쌍이면 서로 다른 순열이 몇 개인가? $n = 10$에서 전부 열거하는 것이 가능한가? 완전 열거로 정확 $p$값을 계산하는 코드를 작성하라.

??? success "풀이"

    $n$쌍이면 각 차이를 유지하거나 뒤집을 수 있으므로 $2^n$가지 부호 배정이 있다. $n = 10$이면 $2^{10} = 1024$로 쉽게 열거된다.

    ```python
    import numpy as np
    from itertools import product
    from scipy import stats

    before = np.array([82, 78, 91, 85, 73, 88, 79, 95, 84, 76])
    after  = np.array([88, 82, 95, 89, 78, 91, 84, 98, 90, 81])
    d = after - before
    print(d)                    # [6 4 4 4 5 3 5 3 6 5]
    print(d.mean())             # 4.5

    S = np.array(list(product([-1, 1], repeat=len(d))))
    t_perm = (S * d).mean(axis=1)
    count = (np.abs(t_perm) >= abs(d.mean()) - 1e-12).sum()
    print(count, len(t_perm), count / len(t_perm))    # 2  1024  0.001953
    ```

    | 검정 | $p$값 |
    |:---|---:|
    | 부호 뒤집기 순열(정확) | **0.001953** |
    | 대응 $t$ 검정 | $3.5 \times 10^{-7}$ |

    정확 $p$값은 $2/1024 = 0.001953$이며 **몬테카를로 오차가 전혀 없다**.

    이 값이 $n = 10$에서 가능한 **최솟값**이다. 열 개의 차이가 모두 양수이므로 $|\bar{d}^*| \ge 4.5$를 만족하는 배정은 관측된 것과 전부 뒤집은 것 둘뿐이다.

    **$t$ 검정과 $5{,}600$배 차이가 난다.** $t = 13.17$이라는 극단적 값이 $t_9$ 분포에서 $3.5 \times 10^{-7}$로 환산되지만, 자료가 담을 수 있는 최대 증거는 $1/512$이다.

    $n > 20$쯤 되면 완전 열거가 비현실적이므로($2^{20} > 10^6$) 부호 뒤집기를 무작위로 표집하는 편이 낫다.

    | $n$ | $2^n$ | 열거 시간(대략) |
    |---:|---:|:---|
    | 10 | 1{,}024 | 순식간 |
    | 20 | 1{,}048{,}576 | 수 초 |
    | 25 | 33{,}554{,}432 | 수 분 · 메모리 주의 |
    | 30 | $1.07\times10^9$ | 비현실적 |

    NumPy로 $2^n \times n$ 행렬을 만들면 $n = 25$에서 이미 $6.7$ GB가 필요하다. 블록 단위로 처리하거나 무작위 표집으로 전환해야 한다.

---

**연습문제 4.** 붓스트랩-순열 비교에서 $0$을 제외하는 붓스트랩 신뢰구간과 $\alpha = 0.05$에서 기각하는 순열검정이 일치해야 한다고 했다. 두 결과가 어긋나는 상황을 구성하라.

??? success "풀이"

    두 방법이 어긋나는 상황은 세 가지 유형으로 나뉜다.

    **(1) 경계 근처의 자료.** 관측 차이가 문턱 부근이면 두 방법의 유한표본 성질 차이가 결론을 가른다. 이는 진짜 불일치가 아니라 잡음이다.

    **(2) 작은 표본.** [비교](./comparison.md) 연습문제 1에서 $m = n = 5$일 때 붓스트랩 백분위수 구간이 $[1.20, 4.80]$으로 $0$을 크게 제외하는데 정확 순열 $p$값은 $0.0397$이었다. 붓스트랩 꼬리 확률은 $0.0006$으로 $66$배 작았다.

    이 경우는 **붓스트랩이 틀렸다**. $n = 5$에서 백분위수 구간의 포함확률이 $0.890$에 불과하다.

    **(3) 이분산 + 불균형 표본.** 이것이 가장 흥미로운 유형이다. 연습문제 2의 설정을 그대로 쓴다($n_x = 20$, $\sigma_x = 1$; $n_y = 50$, $\sigma_y = 3$).

    ```python
    import numpy as np
    rng = np.random.default_rng(9)
    x = rng.normal(6.2, 1, 20)      # 이동 = 1.2
    y = rng.normal(5.0, 3, 50)
    diff, ci, p_perm = bootstrap_vs_permutation_comparison(x, y, rng=rng)
    ```

    여기서는 **붓스트랩 신뢰구간이 옳고 순열검정이 틀린다**. 각 집단을 따로 재표집하는 붓스트랩은 분산 구조를 보존하므로 구간이 $0$을 제외하지만, 평균차 순열검정은 크기가 $0.003$일 만큼 보수적이라 기각하지 못한다.

    | 유형 | 어느 쪽이 옳은가 | 진단 |
    |:---|:---|:---|
    | 경계 자료 | 둘 다 옳다 | $B$를 늘려 몬테카를로 오차를 줄인다 |
    | 작은 표본 | 순열검정 | $n < 20$이면 붓스트랩 구간을 의심한다 |
    | 이분산 + 불균형 | 붓스트랩 | 두 집단의 $s$와 $n$을 확인한다 |

    !!! note "불일치는 오류가 아니라 정보이다"
        두 방법이 어긋났을 때 "어느 쪽을 보고할까"를 고민하는 것은 잘못된 질문이다. 옳은 질문은 **"왜 어긋났는가"**이다.

        위 표의 세 진단은 모두 자료를 보면 즉시 답할 수 있다. $n_x$, $n_y$, $s_x$, $s_y$ 네 숫자만 있으면 어느 유형인지 판별된다.

        불일치를 발견했다면 스튜던트화 순열검정과 BCa 붓스트랩 구간을 함께 계산하는 것이 가장 안전한 대응이다. 이 둘은 위 세 유형 모두에서 원래 버전보다 낫다.

---

**연습문제 5.** Pearson 상관에 대한 순열검정에서 $x$를 고정한 채 $y$를 순열하는 것이 올바른 귀무분포를 생성함을 증명하라. 구체적으로, $(x_i, y_i)$가 독립이라는 가정 아래 $H_0\colon \rho = 0$에서 결합분포가 $y$값의 순열에 대해 불변임을 보여라.

??? success "풀이"

    $H_0$ 아래에서 $X$와 $Y$가 독립이다. 결합밀도가 분해된다.

    $$
    f_{X,Y}(x_i, y_i) = f_X(x_i)\,f_Y(y_i)
    $$

    관측된 자료의 결합가능도는

    $$
    L = \prod_{i=1}^{n} f_X(x_i)\,f_Y(y_i) = \left(\prod_{i=1}^{n} f_X(x_i)\right)\left(\prod_{i=1}^{n} f_Y(y_i)\right)
    $$

    이다. 이제 $\{1, \ldots, n\}$의 임의의 순열 $\pi$를 생각하자. 순열된 자료 $(x_i, y_{\pi(i)})$의 가능도는

    $$
    L_\pi = \prod_{i=1}^{n} f_X(x_i)\,f_Y(y_{\pi(i)}) = \left(\prod_{i=1}^{n} f_X(x_i)\right)\left(\prod_{i=1}^{n} f_Y(y_{\pi(i)})\right)
    $$

    이다. 곱셈이 교환법칙을 만족하므로 $\prod_{i=1}^{n} f_Y(y_{\pi(i)}) = \prod_{i=1}^{n} f_Y(y_i)$이다. 따라서 모든 순열 $\pi$에 대해 $L_\pi = L$이다.

    즉 $H_0$ 아래에서 $x$값과 $y$값의 $n!$가지 짝짓기가 모두 동등하게 가능하며, 이것이 정확히 순열검정이 요구하는 교환가능성 조건이다. $\square$

    !!! warning "이 증명이 요구하는 것은 독립이지 무상관이 아니다"
        증명의 첫 줄에서 $f_{X,Y} = f_X f_Y$를 썼다. 이는 **독립성**이다. $\rho = 0$만으로는 이 분해가 성립하지 않는다.

        따라서 순열검정이 정확히 통제하는 것은 $H_0: X \perp Y$이며, $H_0: \rho = 0$이 아니다. 두 가설의 차이가 실제로 문제가 되는 예 — $Y = X^2$에서 $\rho = 0$이지만 순열검정의 기각률이 $0.331$에 이르는 경우 — 는 [상관에 대한 순열검정](../permutation/correlation.md) 연습문제 3에서 다루었다.

        실무적 함의: 순열검정이 기각했을 때 "상관이 $0$이 아니다"가 아니라 **"$X$와 $Y$가 독립이 아니다"**로 읽어야 한다. 어떤 방식으로 종속인지는 검정통계량의 선택이 결정한다.
