# 일표본 평균 검정

## 개요

일표본 평균 검정은 모평균 $\mu$가 가설의 값 $\mu_0$과 같은지 판단한다. 모분산을 알 때는 **z-검정**을, 모르고 표본에서 추정할 때는 **t-검정**을 쓴다. 두 검정 모두 자료가 정규분포를 따르거나 표본이 중심극한정리를 적용할 만큼 크다는 가정에 기댄다.

## 검정의 구성

**가설:**

- 양측: $H_0\colon \mu = \mu_0$ 대 $H_1\colon \mu \neq \mu_0$
- 단측: $H_0\colon \mu = \mu_0$ 대 $H_1\colon \mu > \mu_0$ (또는 $H_1\colon \mu < \mu_0$)

**z-검정** ($\sigma$를 아는 경우): 검정통계량은

$$
Z = \frac{\bar{X} - \mu_0}{\sigma / \sqrt{n}} \sim N(0,1).
$$

**t-검정** ($\sigma$를 모르고 $S$로 추정하는 경우): 검정통계량은

$$
T = \frac{\bar{X} - \mu_0}{S / \sqrt{n}} \sim t_{n-1}.
$$

<div class="codebox" markdown>

### 예제 1. 일표본 평균 검정 계산기 { .eg }

```python
import math
from scipy.stats import t as tdist, norm

def test_mean_one_sample(xbar, n, mu0=0.0, sd=None, known_sigma=None,
                         alt="two-sided", alpha=0.05):
    """known_sigma를 주면 z-검정, 아니면 표본 sd로 t-검정.

    원자료가 아니라 요약통계량(xbar, n, sd)만 받는다.
    검정에 필요한 것이 그것뿐이기 때문이다.
    돌려주는 값은 (통계량, p-값, 기각 여부, 이름)이다.
    """
    if known_sigma is not None:
        se = known_sigma / math.sqrt(n)
        z = (xbar - mu0) / se
        if alt == "two-sided":
            # 작은 쪽 꼬리를 골라 두 배 한다. z의 부호를 따지지 않아도 되고
            # 어느 쪽으로 치우쳐도 같은 식이 쓰인다.
            p = 2 * min(norm.cdf(z), 1 - norm.cdf(z))
        elif alt == "less":
            p = norm.cdf(z)
        else:
            p = 1 - norm.cdf(z)
        return z, p, (p < alpha), "z-test"

    if sd is None:
        raise ValueError("Provide sd for t-test or known_sigma for z-test.")
    se = sd / math.sqrt(n)
    df = n - 1               # sd를 자료에서 추정했으므로 자유도 하나를 잃는다
    t = (xbar - mu0) / se
    if alt == "two-sided":
        p = 2 * min(tdist.cdf(t, df), 1 - tdist.cdf(t, df))
    elif alt == "less":
        p = tdist.cdf(t, df)
    else:
        p = 1 - tdist.cdf(t, df)
    return t, p, (p < alpha), f"t-test (df={df})"
```

</div>

<div class="codebox" markdown>

### 예제 2. 일표본 평균 검정 { .eg }

```python
stat, p, reject, label = test_mean_one_sample(
    xbar=3.2, n=25, mu0=3.0, sd=1.1, alt="greater"
)
print(label, "stat:", stat, "p:", p, "reject:", reject)

# 같은 자료를 sigma=1.1을 안다고 가정하고 z-검정으로도 해 본다.
stat_z, p_z, reject_z, label_z = test_mean_one_sample(
    xbar=3.2, n=25, mu0=3.0, known_sigma=1.1, alt="greater"
)
print(label_z, "stat:", stat_z, "p:", p_z, "reject:", reject_z)
```

출력:

```
t-test (df=24) stat: 0.9090909090909097 p: 0.18617076763866547 reject: False
z-test stat: 0.9090909090909097 p: 0.18165107044344886 reject: False
```

통계량은 같고 p-값만 다르다. 산포로 넣은 숫자가 1.1로 같으니 분자와 분모가 같을 수밖에 없고, 달라지는 것은 그 통계량을 어느 분포에 견주느냐뿐이다. $t_{24}$가 정규분포보다 꼬리가 두꺼워 같은 통계량에 더 큰 p-값을 준다. $\sigma$를 모른다는 사실의 값이 여기서는 0.0045만큼이다.

</div>

### 해석

이 예제에서는 $\bar{x} = 3.2$, $s = 1.1$, $n = 25$로 $H_0\colon \mu = 3.0$을 $H_1\colon \mu > 3.0$에 대해 검정한다. 검정통계량은

$$
T = \frac{3.2 - 3.0}{1.1/\sqrt{25}} = \frac{0.2}{0.22} \approx 0.909.
$$

자유도 24에서 단측 p-값은 약 0.186이다. 통상적인 $\alpha = 0.05$를 넘으므로 $H_0$을 기각하지 못한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> 관측값 $n = 36$개의 표본에서 $\bar{x} = 52$이고 모표준편차가 $\sigma = 6$으로 알려져 있다. $\alpha = 0.05$에서 $H_0\colon \mu = 50$ 대 $H_1\colon \mu \neq 50$을 검정하라.

</div>

??? success "풀이"

    z-검정통계량은

    $$
    Z = \frac{52 - 50}{6/\sqrt{36}} = \frac{2}{1} = 2.0.
    $$

    양측 p-값은 $2\,P(Z \geq 2.0) = 2(0.0228) = 0.0456$이다. $0.0456 < 0.05$이므로 $H_0$을 기각한다. $\mu \neq 50$이라는 유의한 증거가 있다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span> $n = 10$, $\bar{x} = 15.3$, $s = 2.5$일 때 $\alpha = 0.01$에서 $H_0\colon \mu = 14$ 대 $H_1\colon \mu > 14$를 검정하라.

</div>

??? success "풀이"

    t-검정통계량은

    $$
    T = \frac{15.3 - 14}{2.5/\sqrt{10}} = \frac{1.3}{0.7906} \approx 1.644.
    $$

    $\text{df} = 9$에서 단측 p-값은 $P(T_9 \geq 1.644) \approx 0.068$이다. $0.068 > 0.01$이므로 1% 수준에서 $H_0$을 기각하지 못한다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> $\sigma$를 모를 때 z-검정 대신 t-검정을 쓰는 이유를 설명하라. $n \to \infty$이면 t-분포는 어떻게 되는가?

</div>

??? success "풀이"

    $\sigma$를 모르면 $S$로 추정한다. $S$ 자체가 확률변수이므로 비 $(\bar{X}-\mu_0)/(S/\sqrt{n})$은 표준정규보다 꼬리가 두껍다. $t_{n-1}$ 분포가 이 추가 불확실성을 반영한다. $n \to \infty$이면 대수의법칙에 의해 $S \to \sigma$가 거의 확실하게 성립하므로 $S/\sqrt{n}$이 $\sigma/\sqrt{n}$처럼 행동하고 $t_{n-1} \to N(0,1)$이 된다. 형식적으로 $\nu \to \infty$일 때 $t_\nu \xrightarrow{d} N(0,1)$이다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> 유의수준 $\alpha$에서 단측 t-검정 $H_0\colon \mu = \mu_0$ 대 $H_1\colon \mu > \mu_0$의 기각역을 유도하라.

</div>

??? success "풀이"

    $H_0$ 아래에서 $T = (\bar{X} - \mu_0)/(S/\sqrt{n}) \sim t_{n-1}$이다. $T$가 클 때 $H_0$을 기각하고 $H_1\colon \mu > \mu_0$을 택한다. 기각역은

    $$
    T > t_{\alpha,\,n-1},
    $$

    여기서 $t_{\alpha,\,n-1}$은 $t_{n-1}$ 분포의 $(1-\alpha)$ 분위수, 즉 $P(T_{n-1} > t_{\alpha,\,n-1}) = \alpha$인 값이다. 동등하게 p-값 $P(T_{n-1} \geq t_{\text{obs}}) < \alpha$일 때 기각한다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span> 어떤 제조사가 강봉의 평균 인장강도가 적어도 5000 psi라고 주장한다. 강봉 $n = 20$개의 표본에서 $\bar{x} = 4917$, $s = 200$을 얻었다. $\alpha = 0.05$에서 이 주장이 뒷받침되는지 검정하라.

</div>

??? success "풀이"

    $H_0\colon \mu \geq 5000$ 대 $H_1\colon \mu < 5000$을 검정한다. 검정통계량은

    $$
    T = \frac{4917 - 5000}{200/\sqrt{20}} = \frac{-83}{44.72} \approx -1.856.
    $$

    $\text{df} = 19$에서 단측 p-값은 $P(T_{19} \leq -1.856) \approx 0.039$이다. $0.039 < 0.05$이므로 $H_0$을 기각한다. 평균 인장강도가 5000 psi보다 작다는 유의한 증거가 있어 제조사의 주장과 어긋난다. $\square$

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
일표본 평균 검정을 **부트스트랩**과 **순열**로 수행하는 법을 각각 설명하고, 두 방법의 차이를 밝혀라.

</div>

??? success "풀이"
    **핵심 차이.** 일표본 문제에서

    - **부트스트랩**은 재표본으로 **$\bar X$의 표집분포**를 근사한다. 가정: 관측값이 독립·동일분포.
    - **순열(부호 뒤집기)** 은 **대칭성**을 이용한다. $H_0$ 아래 $X_i-\mu_0$의 부호를 뒤집어도 분포가 같다는 가정이 필요하다.

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(19)
    x = np.array([12.4, 9.8, 14.1, 11.2, 13.7, 10.5, 15.3,
                  12.9, 11.8, 13.1, 10.2, 14.6])
    mu0 = 11.0
    n, B = len(x), 99_999

    # ① t 검정
    t_obs = (x.mean() - mu0) / (x.std(ddof=1) / np.sqrt(n))
    print(f"t({n-1}) = {t_obs:.4f},  p = {2 * stats.t.sf(abs(t_obs), n-1):.4f}")

    # ② 부트스트랩-t (자료를 mu0 로 중심 이동한 뒤 재표본)
    xc = x - x.mean() + mu0
    idx = rng.integers(0, n, (B, n))
    xs = xc[idx]
    tb = (xs.mean(1) - mu0) / (xs.std(1, ddof=1) / np.sqrt(n))
    print(f"부트스트랩 p = {(np.sum(np.abs(tb) >= abs(t_obs)) + 1) / (B + 1):.4f}")

    # ③ 부호 뒤집기 순열
    d = x - mu0
    s = rng.choice([-1.0, 1.0], (B, n))
    ts = (d * s).mean(1) / ((d * s).std(1, ddof=1) / np.sqrt(n))
    print(f"부호 순열 p  = {(np.sum(np.abs(ts) >= abs(t_obs)) + 1) / (B + 1):.4f}")

    # ④ 윌콕슨
    print(f"윌콕슨 p     = {stats.wilcoxon(d).pvalue:.4f}")
    ```

    ```text
    t(11) = 2.8271,  p = 0.0165
    부트스트랩 p = 0.0202
    부호 순열 p  = 0.0192
    윌콕슨 p     = 0.0269
    ```

    **네 방법이 모두 같은 결론을 준다**(0.017~0.027). 자료가 대칭에 가깝기 때문이다. 윌콕슨이 조금 큰 것은 순위만 쓰면서 정보를 일부 버리기 때문이다.

    **가정의 비교.**

    | 방법 | 가정 | 성질 |
    |---|---|---|
    | $t$ | 정규(또는 큰 $n$) | 정규에서 정확 |
    | 부트스트랩-$t$ | 독립·동일분포 | **이차정확**, 대칭 불필요 |
    | 부호 순열 | **$H_0$ 아래 대칭** | 정확(유한표본에서도) |
    | 윌콕슨 | **대칭** | 정확, 순위만 씀 |

    **결정적인 차이 — 대칭성.** 부호 뒤집기 순열은 "$X_i-\mu_0$의 분포가 0에 대해 대칭"을 가정한다. **치우친 자료에서는 평균 검정으로 부적절**하다.

    ```python
    rng = np.random.default_rng(21)
    M, n = 4_000, 20
    cnt_t = cnt_perm = 0
    for _ in range(M):
        y = rng.exponential(1.0, n)          # 평균 1, 오른쪽으로 치우침
        t = (y.mean() - 1.0) / (y.std(ddof=1) / np.sqrt(n))
        cnt_t += abs(t) > stats.t.ppf(0.975, n - 1)
        d = y - 1.0
        s = rng.choice([-1.0, 1.0], (999, n))
        ts = (d * s).mean(1) / ((d * s).std(1, ddof=1) / np.sqrt(n))
        cnt_perm += (np.sum(np.abs(ts) >= abs(t)) + 1) / 1000 < 0.05
    print(f"지수분포(n=20): t 검정 {cnt_t / M:.4f},  "
          f"부호 순열 {cnt_perm / M:.4f}")
    ```

    ```text
    지수분포(n=20): t 검정 0.0862,  부호 순열 0.0860
    ```

    **둘 다 0.086으로 어긋난다.** 부호 순열이 $t$보다 나을 것이 없다. **대칭성이 깨지면 순열의 "정확성"도 사라진다.**

    **권고.**

    - **대칭이 그럴듯하면** 부호 순열이나 윌콕슨이 좋다. 유한표본에서 정확하다.
    - **치우쳐 있고 평균이 관심사이면** 부트스트랩-$t$. 유일하게 이 상황을 제대로 다룬다.
    - **$n$이 크면** 모두 비슷해진다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
평균 검정에서 **이상치 하나**가 결론을 뒤집는 경우를 만들고, 강건한 대안과 비교하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    x = np.array([52.1, 48.7, 51.3, 49.8, 53.2, 50.4, 52.8,
                  49.1, 51.7, 50.9, 48.3, 52.5])
    mu0 = 50.0

    def report(v, tag):
        n = len(v)
        t = stats.ttest_1samp(v, mu0)
        w = stats.wilcoxon(v - mu0)
        k = (v > mu0).sum()
        sg = stats.binomtest(k, n, 0.5).pvalue
        tm = stats.trim_mean(v, 0.1)
        print(f"{tag:14s} 평균 {v.mean():7.3f}  t p={t.pvalue:.4f}  "
              f"윌콕슨 p={w.pvalue:.4f}  부호 p={sg:.4f}  "
              f"10% 절사평균 {tm:.3f}")

    report(x, "원자료")
    y = x.copy()
    y[0] = 20.0                       # 기록 오류 하나
    report(y, "이상치 1개")
    ```

    ```text
    원자료         평균  50.900  t p=0.0857  윌콕슨 p=0.1099  부호 p=0.3877  10% 절사평균 50.930
    이상치 1개      평균  48.225  t p=0.5101  윌콕슨 p=0.5186  부호 p=0.7744  10% 절사평균 50.550
    ```

    **$p$-값이 6배로 벌어진다.** $p=0.086$에서 $p=0.510$이 된다.

    **왜 그런가.** 이상치 하나가

    - **평균을 2.675 끌어내리고**($52.1\to20.0$이므로 $32.1/12=2.675$),
    - **동시에 $S$를 크게 키운다.**

    두 효과가 모두 $|t|$를 줄이는 방향이라 **결론이 극적으로 바뀐다.**

    **강건한 요약통계는 훨씬 덜 흔들린다.**

    | 지표 | 원자료 | 이상치 후 | 변화 |
    |---|---|---|---|
    | 평균 | 50.900 | 48.225 | $-2.675$ |
    | **10% 절사평균** | 50.930 | 50.550 | $-0.380$ |
    | $t$ 검정 $p$ | 0.086 | 0.510 | 6배 |
    | 윌콕슨 $p$ | 0.110 | 0.519 | 4.7배 |
    | 부호 $p$ | 0.388 | 0.774 | 2배 |

    **절사평균의 이동이 평균의 7분의 1이다**(0.38 대 2.68). 순위 기반 검정들도 $t$보다 덜 흔들리지만, $n=12$에서는 관측값 하나의 비중이 커서 완전히 면역은 아니다.

    **실무 절차.**

    1. **그림을 먼저 본다.** 점도표나 상자그림에서 20.0이 명백히 튄다.
    2. **원인을 확인한다.** 52.1을 20.0으로 잘못 입력했을 가능성이 높다(자릿수 실수).
    3. **민감도 분석을 보고한다.** "이상치를 포함하면 $p=0.51$, 제외하면 $p=0.09$"라고 명시한다.
    4. **결론이 관측값 하나에 좌우된다면** 그 사실 자체가 가장 중요한 발견이다.

    **하지 말 것.** 유의해지는 쪽을 골라 보고하기. 앞서 본 연구자 자유도의 전형이다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
일표본 검정에서 **$H_0$의 값 $\mu_0$를 어떻게 정하는지** 논하고, 잘못 정한 경우의 예를 들어라.

</div>

??? success "풀이"
    **$\mu_0$의 출처 — 다섯.**

    | 출처 | 예 | 신뢰도 |
    |---|---|---|
    | **명세·규격** | 병의 표시 용량 500 mL | 높음. 다툼의 여지가 적다 |
    | **이론값** | 물리 상수, 공정한 동전의 0.5 | 높음 |
    | **역사적 기준** | 작년 평균 고객 만족도 | 중간. 조건이 바뀌었을 수 있다 |
    | **외부 기준** | 전국 평균, 업계 표준 | 중간. 비교 가능성 확인 필요 |
    | **임의의 기준** | "0" | **낮다.** 대개 의미 없다 |

    **잘못 정한 경우 셋.**

    **1 — 의미 없는 $\mu_0=0$.** "학생들의 시험 점수 평균이 0인가"를 검정하는 것은 무의미하다. 점수가 0일 수 없기 때문이다. **기각은 당연하고 정보가 없다.**

    올바른 질문은 "기준 점수 70점을 넘는가" 같은 것이다.

    **2 — 자료에서 나온 $\mu_0$.** 같은 자료로 $\mu_0$를 정하고 검정하면 순환이다.

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(77)
    M, n = 20_000, 30

    cnt_ok = 0       # μ0 를 사전에 고정한 경우
    cnt_bad = 0      # μ0 를 같은 자료에서 고른 경우
    for _ in range(M):
        x = rng.normal(0, 1, n)
        cnt_ok += stats.ttest_1samp(x, 0.0).pvalue < 0.05
        mu0_bad = np.percentile(x, 25)          # 자료에서 고른 μ0
        cnt_bad += stats.ttest_1samp(x, mu0_bad).pvalue < 0.05
    print(f"사전 고정 μ0: 제1종 오류율 {cnt_ok / M:.4f}")
    print(f"자료에서 고른 μ0: 기각률 {cnt_bad / M:.4f}")
    ```

    ```text
    사전 고정 μ0: 제1종 오류율 0.0511
    자료에서 고른 μ0: 기각률 0.9769
    ```

    **자료에서 $\mu_0$를 고르면 거의 언제나 기각된다.** 극단적인 예지만, 원리는 흔한 실수와 같다.

    **3 — 대리 기준.** "이 약이 효과가 있는가"를 "$\mu=0$인가"로 바꾸는 것. 진짜 질문은 "임상적으로 의미 있는 효과가 있는가"이므로 $\mu_0$를 **최소 중요 차이**로 두는 것이 낫다.

    **더 나은 틀 — 구간 귀무가설.**

    $$
    H_0:\ |\mu-\mu_{\text{기준}}|\le\Delta
    $$

    앞서 본 대로 $n$이 커도 참인 $H_0$가 기각되지 않는다는 장점이 있다.

    **실무 권고.**

    1. **$\mu_0$를 자료를 보기 전에 정한다.** 사전등록에 적는다.
    2. **$\mu_0$의 근거를 밝힌다.** "규격서 3.2절" 같은 출처.
    3. **$\mu_0$가 임의적이면 검정 대신 추정**을 한다. 구간을 보고하고 독자가 판단하게 한다.
    4. **실무적 문턱이 있으면 그것을 $\mu_0$나 $\Delta$로** 쓴다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
일표본 평균 검정의 **점근적 근거**를 정리하라. 중심극한정리와 슬러츠키 정리가 어떻게 쓰이는가?

</div>

??? success "풀이"
    **목표.** $X_1,\dots,X_n$이 평균 $\mu$, 분산 $\sigma^2<\infty$인 임의 분포에서 나올 때

    $$
    T_n=\frac{\bar X_n-\mu}{S_n/\sqrt n}\ \xrightarrow{d}\ N(0,1)
    $$

    임을 보인다. **정규성 가정이 전혀 필요 없다.**

    **1단계 — 중심극한정리.**

    $$
    Z_n=\frac{\sqrt n(\bar X_n-\mu)}{\sigma}\ \xrightarrow{d}\ N(0,1)
    $$

    **2단계 — 대수의 법칙으로 $S_n$의 일치성.** $S_n^2\xrightarrow{p}\sigma^2$이고, 연속사상정리로 $S_n\xrightarrow{p}\sigma$, 따라서

    $$
    \frac{\sigma}{S_n}\ \xrightarrow{p}\ 1
    $$

    **3단계 — 슬러츠키 정리.** $Z_n\xrightarrow{d}Z$이고 $W_n\xrightarrow{p}c$이면 $Z_nW_n\xrightarrow{d}cZ$다. 여기서

    $$
    T_n=\frac{\sqrt n(\bar X_n-\mu)}{S_n}
    =\underbrace{\frac{\sqrt n(\bar X_n-\mu)}{\sigma}}_{\xrightarrow{d}N(0,1)}
    \times\underbrace{\frac{\sigma}{S_n}}_{\xrightarrow{p}1}
    \ \xrightarrow{d}\ N(0,1)\ \square
    $$

    **무엇이 필요하고 무엇이 필요 없는가.**

    | 필요 | 불필요 |
    |---|---|
    | **독립** | 정규성 |
    | 동일분포(또는 린데베르그 조건) | 대칭성 |
    | **유한한 분산** | 유한한 고차 적률 |

    **유한 분산이 결정적이다.** 코시분포처럼 분산이 없으면 중심극한정리가 성립하지 않고, $T_n$이 정규로 수렴하지 않는다.

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(31)
    M = 40_000
    print(f"{'분포':>8s} " + " ".join(f"{'n='+str(n):>8s}"
                                      for n in [10, 100, 1000]))
    for name, gen, ok in [("정규", lambda s: rng.normal(0, 1, s), True),
                          ("t(2.5)", lambda s: rng.standard_t(2.5, s), True),
                          ("코시", lambda s: rng.standard_cauchy(s), False)]:
        row = []
        for n in [10, 100, 1000]:
            x = gen((M, n))
            t = x.mean(1) / (x.std(1, ddof=1) / np.sqrt(n))
            row.append(np.mean(np.abs(t) > 1.96))
        print(f"{name:>8s} " + " ".join(f"{v:8.4f}" for v in row))
    ```

    ```text
        분포     n=10    n=100   n=1000
        정규   0.0838   0.0512   0.0512
    t(2.5)   0.0695   0.0477   0.0490
        코시   0.0407   0.0217   0.0209
    ```

    **$t_{2.5}$는 분산이 유한하므로 수렴한다.** $n=1000$에서 0.049다. 정규도 $n=100$부터 0.051로 자리 잡는다($n=10$에서 0.084인 것은 $z$ 임계값 1.96을 $t$ 임계값 대신 썼기 때문이다).

    **코시는 수렴하지 않는다.** $n$을 100배 늘려도 0.021~0.022에 머문다. 분산이 없어 정리가 적용되지 않는다.

    **수렴 속도 — 베리-에센.**

    $$
    \sup_z\left|P(Z_n\le z)-\Phi(z)\right|\le\frac{C\rho}{\sigma^3\sqrt n},
    \qquad \rho=E|X-\mu|^3
    $$

    **$n^{-1/2}$ 속도**이며, **3차 적률(왜도)** 이 상수에 들어간다. 앞서 왜도가 관건이라고 본 것의 이론적 근거다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
일표본 평균 검정을 수행하는 **완전한 절차**를 순서대로 정리하고, 각 단계에서 무엇을 확인하는지 적어라.

</div>

??? success "풀이"

    **0단계 — 자료를 보기 전.**

    - [ ] 연구 질문을 한 문장으로 적는다.
    - [ ] $\mu_0$와 그 근거를 정한다.
    - [ ] 단측/양측, $\alpha$를 정한다.
    - [ ] 표본크기와 검정력을 계산한다.
    - [ ] 이상치·결측 처리 규칙을 정한다.
    - [ ] 사전등록한다.

    **1단계 — 자료 확인.**

    - [ ] 관측값 수가 계획과 맞는가. 결측은 얼마인가.
    - [ ] **히스토그램과 점도표**를 그린다.
    - [ ] **Q-Q 그림**을 그린다.
    - [ ] 요약통계: $n$, $\bar x$, $s$, 중앙값, 사분위, 왜도.
    - [ ] 시간 순서나 수집 순서에 추세가 있는가(독립성).

    **2단계 — 가정 점검.**

    | 가정 | 확인 방법 | 위배 시 |
    |---|---|---|
    | 독립 | **설계**를 확인. 순서 그림 | 혼합모형, 군집 보정 |
    | 정규/왜도 | Q-Q 그림, $\hat\gamma_1$ | 변환, 부트스트랩, 비모수 |
    | 이상치 | 점도표, 영향도 | 강건 방법, 민감도 분석 |

    **3단계 — 검정 수행.**

    - [ ] 계획한 검정을 그대로 수행한다.
    - [ ] 검정통계량, 자유도, $p$-값을 기록한다.

    **4단계 — 효과크기와 구간.**

    - [ ] 원 척도의 차이와 95% 구간.
    - [ ] 표준화 효과크기와 구간.

    **5단계 — 민감도.**

    - [ ] 이상치 포함/제외.
    - [ ] 대안 검정($t$/윌콕슨/부트스트랩)의 결과.
    - [ ] 결측 처리 방식을 바꿔 본 결과.

    **6단계 — 보고.**

    > $n=25$개 표본의 평균 중량은 490 g(SD 18)으로, 표시 용량 500 g과 비교하는 양측 일표본 $t$ 검정에서 $t(24)=-2.78$, $p=0.010$이었다. 평균 차이는 $-10.0$ g(95% 신뢰구간 $-17.4$ ~ $-2.6$ g), 코헨의 $d$는 $-0.56$(95% CI $-0.97$ ~ $-0.13$)이다. Q-Q 그림에서 정규성 위배의 뚜렷한 증거는 없었고(왜도 0.21), 윌콕슨 부호순위 검정도 같은 결론을 주었다($p=0.013$). 규격상 허용 오차 $\pm5$ g을 구간의 상한이 넘으므로 공정 점검이 필요하다.

    **가장 자주 빠지는 세 가지.**

    1. **그림을 안 그린다.** 1단계를 건너뛰고 바로 검정한다.
    2. **구간을 보고하지 않는다.** $p$-값만 적는다.
    3. **실무적 문턱과 비교하지 않는다.** 통계적 유의성으로 끝낸다.

    **한 문장.** **검정은 여섯 단계 중 하나일 뿐이며, 앞뒤의 다섯 단계가 결론의 신뢰도를 결정한다.**

---

## 정리하며

평균 검정을 **코드로 구현**하며 실무의 세부를 확인했다.

- **$z$ 와 $t$ 의 분기는 한 줄이다.** $\sigma$ 가 주어졌는지로 갈리며, 나머지 계산은 동일하다. 함수 하나에 `sigma=None` 기본값을 두는 것이 깔끔하다.
- **단측·양측을 인자로 받는다.** `alternative` 를 `"two-sided"`, `"greater"`, `"less"` 로 두는 것이 `scipy` 의 관례이며, 이를 따르면 혼동이 줄어든다.
- **원자료와 요약통계량 둘 다 받도록 만든다.** $\bar x$, $s$, $n$ 만 있으면 검정이 가능하므로, 논문의 보고값만으로 재현할 수 있다.
- **`scipy.stats.ttest_1samp` 와 대조해 검산한다.** 직접 구현한 값이 라이브러리와 맞는지 확인하는 것이 이 절의 목적이며, 맞지 않으면 대개 자유도나 단측 처리에서 어긋난다.
- **작은 $p$ 값은 `sf` 로 계산한다.** `1 - cdf` 는 꼬리에서 정밀도를 잃는다(4장).

다음 절 **일표본 비율 검정**으로 넘어간다.
