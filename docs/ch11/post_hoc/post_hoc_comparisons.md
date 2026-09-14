# 사후비교 예제

## 개요

일원배치 분산분석이 귀무가설을 기각하면 적어도 한 집단의 평균이 나머지와 다르다는 것은 알 수 있지만 어느 쌍이 다른지는 알 수 없다. 사후비교 절차는 가족단위 오류율을 통제하면서 쌍별 검정을 수행하여 이 빈틈을 메운다. 이 페이지는 가장 흔한 방법인 Tukey의 HSD, Bonferroni 보정, Scheffé 방법을 예제와 함께 살펴본다.

## 다중비교 문제

집단이 $k$개면 쌍별 비교는 $\binom{k}{2}$개이다. 각 검정을 유의수준 $\alpha$에서 수행하면 전역 귀무가설 아래에서 거짓 기각이 적어도 하나 나올 확률은

$$
1 - (1 - \alpha)^{\binom{k}{2}}
$$

이다. $k = 5$이고 $\alpha = 0.05$이면 비교가 10개이므로 가족단위 오류율이 대략 $1 - 0.95^{10} \approx 0.40$이 된다. 사후 방법은 이 부풀려진 오류를 통제한다.

## Tukey의 정직유의차

Tukey의 HSD는

$$
|\bar{y}_{i\cdot} - \bar{y}_{j\cdot}| > q_{\alpha,\, k,\, N-k} \sqrt{\frac{MSW}{n}}
$$

일 때 집단 $i$와 $j$가 유의하게 다르다고 선언한다. 여기서 $q_{\alpha,k,N-k}$는 스튜던트화 범위 분포의 임계값, $MSW$는 집단 내 평균제곱, $n$은 (균형 설계에서) 공통 집단 크기이다.

<div class="codebox" markdown>

### 예제 1. Tukey HSD로 세 집단 견주기 { .eg }

```python
import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# 세 집단의 참 평균을 10, 12, 15 로 두고 표준편차는 모두 3.5 로 맞췄다.
# A와 B의 차이는 표준편차보다 작고 A와 C의 차이는 그보다 크다.
# Tukey 가 어느 쌍을 갈라내고 어느 쌍을 갈라내지 못하는지 보게 된다.
rng = np.random.default_rng(42)
n = 15
df = pd.DataFrame({
    "response": np.concatenate([
        rng.normal(10.0, 3.5, n),
        rng.normal(12.0, 3.5, n),
        rng.normal(15.0, 3.5, n),
    ]),
    "group": ["A"] * n + ["B"] * n + ["C"] * n,
})

# Tukey HSD 는 쌍 세 개를 한꺼번에 견주면서 전체 오류율을 0.05 로 묶는다.
# 쌍마다 t-검정을 따로 하면 이 통제가 무너진다.
print(pairwise_tukeyhsd(endog=df["response"], groups=df["group"], alpha=0.05))
```

출력:

```
Multiple Comparison of Means - Tukey HSD, FWER=0.05
===================================================
group1 group2 meandiff p-adj   lower  upper  reject
---------------------------------------------------
     A      B   2.1353 0.1115 -0.3876 4.6582  False
     A      C   5.4745    0.0  2.9516 7.9975   True
     B      C   3.3392 0.0069  0.8163 5.8621   True
---------------------------------------------------
```

참 평균이 10, 12, 15이고 표준편차가 3.5인 자료다. A와 C의 차이(참값 5)와 B와 C의 차이(참값 3)는 잡아내지만, A와 B의 차이(참값 2)는 $p = 0.11$로 놓친다. 집단당 15개로는 표준편차 3.5 대비 2의 차이를 가려내기 어렵다.

**사후검정이 유의하지 않다는 것이 차이가 없다는 뜻은 아니다.** 여기서는 참 차이가 분명히 존재하는데도 놓쳤다.

불균형 설계에서는 Tukey-Kramer 수정이 $\sqrt{MSW/n}$을 $\sqrt{MSW \cdot (1/n_i + 1/n_j)/2}$로 대체한다.

</div>

## Bonferroni 보정

Bonferroni 방법은 각 쌍별 $p$-값에 비교의 수 $m = \binom{k}{2}$를 곱해 조정한다:

$$
p_{\text{adj}} = \min(m \cdot p_{\text{raw}},\; 1)
$$

간단하고 널리 쓸 수 있지만 $m$이 크면 보수적일 수 있다. Holm-Bonferroni 단계적 하강 변형은 가족단위 오류율을 여전히 통제하면서 균일하게 더 강력하다.

## Scheffé 방법

Scheffé 절차는 쌍별 차이만이 아니라 가능한 모든 선형 대비에 대해 가족단위 오류율을 통제한다. 임계값은 $F$-분포에 근거한다:

$$
|\bar{y}_{i\cdot} - \bar{y}_{j\cdot}| > \sqrt{(k-1)\, F_{\alpha,\, k-1,\, N-k}} \cdot \sqrt{MSW \left(\frac{1}{n_i} + \frac{1}{n_j}\right)}
$$

Scheffé 방법은 더 넓은 대비의 족에 대해 오류를 통제하므로 쌍별 비교에서는 셋 중 가장 보수적이다.

## 방법 비교

| 방법 | 오류를 통제하는 대상 | 검정력 | 적합한 경우 |
|---|---|---|---|
| Tukey HSD | 모든 쌍별 비교 | 높음 | 모든 쌍별 비교가 관심사일 때 |
| Bonferroni | 미리 지정한 임의의 집합 | 중간 | 계획된 비교가 적을 때 |
| Scheffé | 가능한 모든 대비 | (쌍별에서는) 낮음 | 복잡한 대비가 관심사일 때 |

## 해석

- **Tukey HSD**는 모든 집단 평균 쌍을 비교하고 싶을 때의 기본 선택이다. 균형 설계에서는 정확하고 불균형 설계에서는 근사적이다(Tukey-Kramer).
- **Bonferroni**는 계획된 비교가 소수일 때 가장 유용하다. 비교 수가 늘수록 검정력이 떨어지기 때문이다.
- **Scheffé**는 임의의 선형 대비에 관심이 있을 때(예: 두 집단의 평균을 세 번째 집단과 비교할 때) 선택하는 방법이다. 순수하게 쌍별 비교만 한다면 Tukey HSD의 검정력이 더 높다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
집단 $k = 4$개, 집단당 $n_i = 15$인 일원배치 분산분석에서 $MSW = 12.3$을 얻었다. $\alpha = 0.05$의 스튜던트화 범위 임계값은 $q_{0.05,4,56} = 3.74$이다. Tukey HSD 문턱을 계산하고 평균 차이 $\bar{y}_1 - \bar{y}_3 = 3.5$가 유의한지 판정하라.

</div>

??? success "풀이"
    균형 설계의 Tukey HSD 문턱은

    $$
    \text{HSD} = q_{\alpha,k,N-k} \sqrt{\frac{MSW}{n}} = 3.74 \sqrt{\frac{12.3}{15}} = 3.74 \sqrt{0.82} = 3.74 \times 0.9055 \approx 3.39
    $$

    이다. $|\bar{y}_1 - \bar{y}_3| = 3.5 > 3.39$이므로 $\alpha = 0.05$ 수준에서 차이가 유의하다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
어떤 연구자가 ($k = 4$개 집단에서 나오는) $m = 6$개의 쌍별 비교를 가족단위 $\alpha = 0.05$의 Bonferroni 방법으로 수행하려 한다. 보정 전 $p$-값은 $0.003, 0.012, 0.041, 0.078, 0.210, 0.530$이다. Bonferroni에서 유의한 비교는 무엇인가? Holm-Bonferroni 단계적 하강 절차에서는 어떤 비교가 추가로 유의해지는가?

</div>

??? success "풀이"
    **Bonferroni:** 각 보정 전 $p$-값에 $m = 6$을 곱한다.

    | 보정 전 $p$ | Bonferroni $p_{\text{adj}}$ | 유의? |
    |---|---|---|
    | 0.003 | 0.018 | 예 |
    | 0.012 | 0.072 | 아니오 |
    | 0.041 | 0.246 | 아니오 |
    | 0.078 | 0.468 | 아니오 |
    | 0.210 | 1.000 | 아니오 |
    | 0.530 | 1.000 | 아니오 |

    Bonferroni에서는 첫 번째 비교만 유의하다.

    **Holm-Bonferroni:** 보정 전 $p$-값을 오름차순으로 정렬하고 $p_{(j)}$를 $\alpha / (m - j + 1)$과 비교한다:

    - $p_{(1)} = 0.003 < 0.05/6 = 0.00833$ — 기각한다.
    - $p_{(2)} = 0.012 < 0.05/5 = 0.01$이 아니다($0.012 > 0.01$) — 기각하지 못한다. 여기서 멈춘다.

    Holm-Bonferroni에서도 첫 번째 비교만 유의하다. 다만 두 번째 $p$-값이 조금만 더 작았다면(예: $0.009$) 그것도 기각되었을 것이며, 이는 Holm이 Bonferroni보다 엄밀하게 더 강력함을 보여준다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
$k = 2$일 때 Tukey HSD 검정이 이표본 $t$-검정과 동치임을 보여라. 구체적으로 $\nu = N - 2$일 때 $q_{\alpha,2,\nu}^2 = 2\, F_{\alpha,1,\nu}$임을 증명하라.

</div>

??? success "풀이"
    $k = 2$일 때 모수가 $(2, \nu)$인 스튜던트화 범위 분포는 $t$-분포와 $q_{2,\nu} = \sqrt{2}\, |t_\nu|$의 관계를 갖는다. 양변을 제곱하면 $q_{2,\nu}^2 = 2\, t_\nu^2$이다. $t_\nu^2 \sim F_{1,\nu}$이므로

    $$
    q_{\alpha,2,\nu}^2 = 2\, F_{\alpha,1,\nu}
    $$

    이다. Tukey HSD 검정은 $|\bar{y}_1 - \bar{y}_2| / \sqrt{MSW/n} > q_{\alpha,2,\nu}$일 때 기각하는데, 이는

    $$
    \frac{(\bar{y}_1 - \bar{y}_2)^2}{MSW/n} > 2\, F_{\alpha,1,\nu}
    $$

    와 동치이다. 좌변은 $2 F_{\text{obs}}$와 같고 $F_{\text{obs}} = MSB/MSW$는 $k = 2$일 때의 표준 분산분석 $F$-통계량이다. 따라서 Tukey 검정은 $F_{\text{obs}} > F_{\alpha,1,\nu}$일 때 기각하는 것으로 환원되며, 이것이 바로 이표본 $t$-검정(동등하게 분자 자유도 $k-1 = 1$인 $F$-검정)이다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
쌍별 비교에서 Scheffé 방법이 Tukey의 HSD보다 보수적이면서도 Tukey가 탐지할 수 없는 효과를 탐지할 수 있는 이유를 설명하라. Scheffé로는 검정할 수 있고 Tukey로는 할 수 없는 대비의 구체적인 예를 들어라.

</div>

??? success "풀이"
    Scheffé 방법은 쌍별 차이만이 아니라 $\sum c_i = 0$인 **가능한 모든 선형 대비** $\psi = \sum c_i \mu_i$에 대해 가족단위 오류율을 통제한다. 이 족이 쌍별 비교의 집합보다 훨씬 크므로 임계값이 커지고, 개별 쌍별 비교에서는 더 보수적이 된다.

    그러나 Scheffé 방법은 다음과 같은 복잡한 대비를 검정할 수 있다:

    $$
    \psi = \frac{\mu_1 + \mu_2}{2} - \mu_3
    $$

    이 대비는 집단 1과 2의 평균이 집단 3과 다른지를 묻는다. Tukey의 HSD는 이런 유형의 비교를 위해 설계되지 않았다. 예를 들어 세 가지 교수법을 비교하는 연구에서 강의 중심 두 방법의 평균 효과가 프로젝트 중심 방법과 다른지 검정하고 싶을 수 있다. Scheffé 방법은 이를 직접 다루지만 Tukey는 그럴 수 없다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
Bonferroni 보정이 가족단위 오류율을 수준 $\alpha$로 통제함을 증명하라. 즉 $m$개의 검정을 각각 수준 $\alpha/m$에서 수행하면 $P(H_0 \text{ 아래에서 거짓 기각이 적어도 하나}) \le \alpha$임을 보여라.

</div>

??? success "풀이"
    $R_j$를 $j$번째 귀무가설이 거짓으로 기각되는 사건이라 하자($j = 1, \ldots, m$). 각 검정을 수준 $\alpha/m$에서 수행하므로 $P(R_j) \le \alpha/m$이다. Boole의 부등식(합집합 상한)에 의해

    $$
    P\!\left(\bigcup_{j=1}^{m} R_j\right) \le \sum_{j=1}^{m} P(R_j) \le \sum_{j=1}^{m} \frac{\alpha}{m} = \alpha
    $$

    이다. 이는 검정들 사이의 의존 구조와 무관하게 성립한다. 이 상한은 기각 사건들이 서로 겹치지 않을 때 가장 빠듯하고, 검정들이 양의 상관을 가질수록(사건들이 많이 겹칠수록) 보수적이 된다. $\square$

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
쌍별 비교 절차 다섯의 **가족단위 오류율을 한 표에** 재라. 보정하지 않으면 $k$가 커질수록 얼마나 나빠지는가?

</div>

??? success "풀이"
    ```python
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    from scipy import stats
    from statsmodels.stats.libqsturng import qsturng

    def fwer_procs(gs, alpha=0.05):
        k = len(gs)
        n = np.array([len(g) for g in gs], float)
        m = np.array([g.mean() for g in gs])
        v = np.array([g.var(ddof=1) for g in gs])
        MSE = ((n - 1) * v).sum() / (n.sum() - k)
        dfe = int(n.sum() - k)
        ts = np.array([abs(m[i] - m[j]) / np.sqrt(MSE * (1 / n[i] + 1 / n[j]))
                       for i in range(k) for j in range(i + 1, k)])
        ps = 2 * stats.t.sf(ts, dfe)
        M = len(ts)

        srt = np.sort(ps)
        holm = any(srt[idx] < alpha / (M - idx)
                   for idx in range(next((i for i in range(M)
                                          if srt[i] >= alpha / (M - i)), M)))
        return {
            "보정 없음": (ps < alpha).any(),
            "본페로니": (ps < alpha / M).any(),
            "홀름": holm,
            "튜키": (ts * np.sqrt(2) > qsturng(1 - alpha, k, dfe)).any(),
            "셰페": (ts**2 > (k - 1) * stats.f.ppf(1 - alpha, k - 1, dfe)).any(),
        }

    rng = np.random.default_rng(12001)
    B = 4_000
    keys = ["보정 없음", "본페로니", "홀름", "튜키", "셰페"]
    print("완전 귀무, 정규·등분산, 명목 0.05")
    print(f"{'설계':>14s} " + " ".join(f"{s:>9s}" for s in keys))
    for k, n in [(3, 15), (4, 15), (6, 12), (10, 10)]:
        acc = {s: 0 for s in keys}
        for _ in range(B):
            r = fwer_procs([rng.normal(0, 1, n) for _ in range(k)])
            for s in keys:
                acc[s] += r[s]
        print(f"k={k}, n={n:<6d} " + " ".join(f"{acc[s] / B:9.4f}" for s in keys))
    ```

    ```text
    완전 귀무, 정규·등분산, 명목 0.05
                설계     보정 없음      본페로니        홀름        튜키        셰페
    k=3, n=15        0.1230    0.0387    0.0387    0.0470    0.0348
    k=4, n=15        0.2050    0.0450    0.0450    0.0537    0.0307
    k=6, n=12        0.3560    0.0333    0.0333    0.0475    0.0107
    k=10, n=10        0.5920    0.0328    0.0328    0.0460    0.0018
    ```

    **보정하지 않으면 $k=10$에서 오류율이 0.592다.** 자료에 아무 차이가 없는데도 **두 번 중 한 번 이상** "유의한 쌍"을 찾아낸다.

    | $k$ | 비교 수 | 보정 없음 | 본페로니 | **튜키** | 셰페 |
    |---|---|---|---|---|---|
    | 3 | 3 | 0.123 | 0.039 | **0.047** | 0.035 |
    | 4 | 6 | 0.205 | 0.045 | **0.054** | 0.031 |
    | 6 | 15 | 0.356 | 0.033 | **0.048** | 0.011 |
    | 10 | 45 | **0.592** | 0.033 | **0.046** | **0.002** |

    **튜키만 명목 0.05에 정확히 맞는다**(0.046~0.054). 나머지는 모두 보수적이다.

    **왜 튜키가 정확한가.** 스튜던트화 범위 분포는 **"$k$개 평균의 최댓값과 최솟값의 차이"**의 정확한 분포다. 쌍별 비교에서 가장 극단적인 것이 바로 그 범위이므로 **정확히 맞아떨어진다**(균형·등분산일 때).

    **본페로니가 보수적인 이유.** 본페로니 부등식

    $$
    P\left(\bigcup_i A_i\right)\leq\sum_i P(A_i)
    $$

    는 $A_i$가 **배반일 때만 등호**다. 쌍별 비교는 같은 평균들을 공유해 **강하게 상관**되어 있으므로 합집합 확률이 합보다 훨씬 작다.

    **셰페가 극단적으로 보수적이다.** $k=10$에서 0.0018로 명목의 **3%**에 불과하다. 셰페는 **모든 가능한 대비**(무한개)에 대해 동시에 통제하므로, 쌍별 비교만 볼 때는 지나치다.

    **홀름과 본페로니가 완전히 같다**(0.0387, 0.0450, …). **완전 귀무에서는 두 절차가 동일**하기 때문이다. 차이는 **일부 귀무가설이 거짓일 때** 나타난다(연습문제 7).

    **권고.**

    | 목적 | 절차 |
    |---|---|
    | 모든 쌍별 비교(균형·등분산) | **튜키** |
    | 미리 정한 소수의 비교 | **홀름**(본페로니보다 항상 낫다) |
    | 자료를 보고 고른 대비 | **셰페**(연습문제 8) |
    | 대조군과의 비교만 | **더넷** |
    | 이분산 | **게임스-하웰**(연습문제 9) |

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span>
연습문제 2의 **홀름 절차가 본페로니를 "지배"**한다고 한다. 검정력과 오류율을 모두 재어 확인하라.

</div>

??? success "풀이"
    ```python
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    from scipy import stats
    from statsmodels.stats.libqsturng import qsturng

    def proc_rejects(gs, alpha=0.05):
        """각 절차가 어느 쌍을 기각하는지."""
        k = len(gs)
        n = np.array([len(g) for g in gs], float)
        m = np.array([g.mean() for g in gs])
        v = np.array([g.var(ddof=1) for g in gs])
        MSE = ((n - 1) * v).sum() / (n.sum() - k)
        dfe = int(n.sum() - k)
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        ts = np.array([abs(m[i] - m[j]) / np.sqrt(MSE * (1 / n[i] + 1 / n[j]))
                       for i, j in pairs])
        ps = 2 * stats.t.sf(ts, dfe)
        M = len(pairs)
        rej = np.zeros(M, bool)
        for r, idx in enumerate(np.argsort(ps)):
            if ps[idx] < alpha / (M - r):
                rej[idx] = True
            else:
                break
        return pairs, {
            "본페로니": ps < alpha / M,
            "홀름": rej,
            "튜키": ts * np.sqrt(2) > qsturng(1 - alpha, k, dfe),
            "셰페": ts**2 > (k - 1) * stats.f.ppf(1 - alpha, k - 1, dfe),
        }

    procs = ["본페로니", "홀름", "튜키", "셰페"]
    B = 4_000

    rng = np.random.default_rng(12002)
    print("검정력: k=4, n=15, μ=(0,0,0,δ) — 참인 차이는 3쌍")
    print(f"{'δ':>5s} " + " ".join(f"{s + ' 전체':>11s}" for s in procs))
    for d in [0.8, 1.2, 1.6]:
        acc = {s: 0 for s in procs}
        for _ in range(B):
            pairs, out = proc_rejects([rng.normal(mu, 1, 15)
                                       for mu in [0, 0, 0, d]])
            idx = [i for i, p in enumerate(pairs) if p[1] == 3]
            for s in procs:
                acc[s] += out[s][idx].all()
        print(f"{d:5.1f} " + " ".join(f"{acc[s] / B:11.4f}" for s in procs))

    rng = np.random.default_rng(12003)
    print("\n부분 귀무에서의 FWER (참 차이가 없는 3쌍 중 하나라도 기각)")
    print(f"{'δ':>5s} " + " ".join(f"{s:>11s}" for s in procs))
    for d in [0, 2, 4, 8]:
        acc = {s: 0 for s in procs}
        for _ in range(B):
            pairs, out = proc_rejects([rng.normal(mu, 1, 15)
                                       for mu in [0, 0, 0, d]])
            idx = [i for i, p in enumerate(pairs) if p[1] != 3]
            for s in procs:
                acc[s] += out[s][idx].any()
        print(f"{d:5.1f} " + " ".join(f"{acc[s] / B:11.4f}" for s in procs))
    ```

    ```text
    검정력: k=4, n=15, μ=(0,0,0,δ) — 참인 차이는 3쌍
        δ     본페로니 전체       홀름 전체       튜키 전체       셰페 전체
      0.8      0.1067      0.1373      0.1250      0.0815
      1.2      0.4763      0.5417      0.5150      0.4200
      1.6      0.8682      0.8982      0.8838      0.8297

    부분 귀무에서의 FWER (참 차이가 없는 3쌍 중 하나라도 기각)
        δ        본페로니          홀름          튜키          셰페
      0.0      0.0255      0.0262      0.0323      0.0177
      2.0      0.0198      0.0398      0.0253      0.0112
      4.0      0.0210      0.0413      0.0255      0.0155
      8.0      0.0222      0.0475      0.0272      0.0127
    ```

    **홀름이 본페로니보다 검정력이 높다.**

    | $\delta$ | 본페로니 | **홀름** | 이득 |
    |---|---|---|---|
    | 0.8 | 0.107 | **0.137** | $+29\%$ |
    | 1.2 | 0.476 | **0.542** | $+14\%$ |
    | 1.6 | 0.868 | **0.898** | $+3\%$ |

    **그러면서 FWER을 넘지 않는다**(최대 0.0475).

    **"지배"의 정확한 의미.** 홀름이 기각하는 집합은 **언제나 본페로니가 기각하는 집합을 포함**한다.

    $$
    \text{가장 작은 }p\text{에 대해}:\quad
    \frac{\alpha}{M}\ \text{(본페로니)}\ =\ \frac{\alpha}{M-0}\ \text{(홀름의 첫 단계)}
    $$

    첫 단계가 같고, 이후 단계에서 홀름의 문턱이 **더 느슨**해진다. **본페로니가 기각한 것을 홀름이 기각하지 않는 경우는 없다.**

    **본페로니의 부분 귀무 FWER이 $\delta$와 무관하게 0.02 근처**인 것도 주목할 만하다. 홀름은 $\delta$가 커질수록 0.0475까지 올라가 **명목에 가까워진다.** 이것이 **검정력의 원천**이다.

    **홀름의 작동 원리.**

    ```text
    p(1) ≤ p(2) ≤ ... ≤ p(M) 으로 정렬
      p(1) < α/M     인가?  아니면 모두 기각하지 않고 종료
      p(2) < α/(M-1) 인가?  아니면 여기서 종료
      p(3) < α/(M-2) 인가?  ...
    ```

    **하나가 확실히 거짓이면 남은 가설의 수가 줄어드니 문턱을 완화해도 된다**는 논리다.

    **왜 모두가 홀름을 쓰지 않는가.**

    | 이유 | 설명 |
    |---|---|
    | 관례 | 본페로니가 더 오래되고 널리 알려짐 |
    | **신뢰구간** | 본페로니는 구간을 바로 주지만 **홀름은 어렵다** |
    | 설명의 단순함 | "$\alpha$를 $M$으로 나눴다"가 이해하기 쉽다 |

    **두 번째가 실질적 이유**다. 홀름은 **검정에는 좋지만 신뢰구간을 만들기 어렵다.** 구간이 필요하면 본페로니나 튜키를 쓴다.

    **연습문제 2의 답을 확인하면.** $p=(0.003,\,0.012,\,0.041,\,0.078,\,0.210,\,0.530)$, $M=6$, $\alpha=0.05$에서

    | 순위 | $p$ | 본페로니 $\alpha/6=0.0083$ | 홀름 문턱 |
    |---|---|---|---|
    | 1 | 0.003 | **기각** | $0.05/6=0.0083$ → **기각** |
    | 2 | 0.012 | 기각 안 함 | $0.05/5=0.0100$ → 기각 안 함 |

    **홀름도 두 번째에서 멈춘다.** 0.012가 0.010보다 크기 때문이다. **이 예에서는 두 절차의 결과가 같다.** 홀름이 항상 더 많이 기각하는 것은 아니다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
연습문제 4가 말한 **셰페의 진가**를 수치로 보여라. 자료를 보고 고른 대비에서 무슨 일이 일어나는가?

</div>

??? success "풀이"
    **셰페의 정리.** 모든 대비 $\psi=\sum c_i\mu_i$($\sum c_i=0$)에 대해 동시에

    $$
    P\left(\forall\psi:\ |\hat\psi-\psi|\leq\sqrt{(k-1)F_{\alpha}(k-1,\nu)}
    \cdot\operatorname{SE}(\hat\psi)\right)=1-\alpha
    $$

    **"모든 대비"가 핵심이다.** 자료를 보고 고른 대비도 포함된다.

    ```python
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    from scipy import stats
    from statsmodels.stats.libqsturng import qsturng

    rng = np.random.default_rng(12005)
    B = 4_000
    k, n = 5, 12
    dfe = k * (n - 1)
    tc = qsturng(0.95, k, dfe) / np.sqrt(2)
    sc = np.sqrt((k - 1) * stats.f.ppf(0.95, k - 1, dfe))
    bc = stats.t.ppf(1 - 0.05 / (2 * 10), dfe)

    a = b = c = d = 0
    for _ in range(B):
        gs = [rng.normal(0, 1, n) for _ in range(k)]
        m = np.array([g.mean() for g in gs])
        MSE = np.array([g.var(ddof=1) for g in gs]).mean()
        tmax = max(abs(m[i] - m[j]) / np.sqrt(2 * MSE / n)
                   for i in range(k) for j in range(i + 1, k))
        a += tmax > tc
        b += tmax > bc
        c += tmax > sc
        # 자료가 알려 주는 '최적' 대비: c ∝ (m - m̄)
        d += np.sqrt(n * ((m - m.mean())**2).sum() / MSE) > sc

    print("k=5, n=12, 완전 귀무. '가장 큰 차이를 내는 것'을 자료에서 찾아 검정")
    print(f"{'쌍별 최대 — 튜키 기준':>28s} {a / B:8.4f}")
    print(f"{'쌍별 최대 — 본페로니(10쌍)':>28s} {b / B:8.4f}")
    print(f"{'쌍별 최대 — 셰페 기준':>28s} {c / B:8.4f}")
    print(f"{'최적 대비 — 셰페 기준':>28s} {d / B:8.4f}")
    print(f"\n임계값: 튜키 {tc:.4f}, 본페로니 {bc:.4f}, 셰페 {sc:.4f}")

    rng = np.random.default_rng(12005)
    e = sum(stats.f_oneway(*[rng.normal(0, 1, n) for _ in range(k)]).pvalue < 0.05
            for _ in range(B))
    print(f"F 검정의 기각률(참고): {e / B:.4f}")
    ```

    ```text
    k=5, n=12, 완전 귀무. '가장 큰 차이를 내는 것'을 자료에서 찾아 검정
                   쌍별 최대 — 튜키 기준   0.0590
               쌍별 최대 — 본페로니(10쌍)   0.0435
                   쌍별 최대 — 셰페 기준   0.0195
                   최적 대비 — 셰페 기준   0.0542

    임계값: 튜키 2.8204, 본페로니 2.9247, 셰페 3.1873
    F 검정의 기각률(참고): 0.0542
    ```

    **마지막 두 줄이 셰페 정리의 핵심**이다.

    $$
    \text{최적 대비의 셰페 기각률}=0.0542
    =\text{전체 }F\text{ 검정의 기각률}
    $$

    **정확히 같다.** 이것은 우연이 아니라 **정리**다.

    > $F$ 검정이 유의하다 $\iff$ 셰페 기준으로 유의한 대비가 **적어도 하나 존재한다**

    **최적 대비는 $c_i\propto\bar y_i-\bar y_{\cdot\cdot}$**이고, 그 $t^2$이 정확히 $\text{SSB}/\text{MSE}=(k-1)F$다.

    **쌍별 비교에만 쓰면 셰페는 손해다**(0.0195). 쌍별 대비는 **모든 대비의 아주 작은 부분집합**인데, 셰페는 나머지 무한개까지 보호하느라 문턱을 높인다.

    | 대비의 집합 | 적합한 절차 | 그 절차의 FWER |
    |---|---|---|
    | 쌍별 $\binom k2$개 | **튜키** | 0.059 |
    | 미리 정한 $M$개 | **본페로니/홀름** | 0.044 |
    | **모든 대비(무한)** | **셰페** | **0.054** |

    **각 절차가 자기 영역에서 정확하다.** 영역을 넘어 쓰면 보수적이 된다.

    **셰페로만 검정할 수 있는 대비의 예.**

    | 대비 | 언제 쓰나 |
    |---|---|
    | $\dfrac{\mu_1+\mu_2}{2}-\dfrac{\mu_3+\mu_4+\mu_5}{3}$ | 두 처리군 대 세 대조군 |
    | $\mu_1-2\mu_2+\mu_3$ | 이차 추세 |
    | **자료를 보고 "이 묶음이 달라 보인다"** | **셰페만 가능** |

    **마지막이 결정적이다.** 자료를 본 뒤 대비를 고르면 튜키도 본페로니도 유효하지 않다. **몇 개를 고를지조차 자료가 정하기 때문**이다.

    **실무 지침.**

    1. **쌍별만 볼 것이면 튜키.**
    2. **사전에 정한 대비면 홀름.**
    3. **자료를 보고 대비를 고를 것이면 셰페.**
    4. **$F$가 유의하지 않으면 셰페로 유의한 대비는 존재하지 않는다.** 찾을 필요가 없다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
**이분산이 있으면** 튜키 HSD가 어떻게 되는가? 게임스-하웰과 비교하라.

</div>

??? success "풀이"
    ```python
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    from statsmodels.stats.libqsturng import qsturng

    def gh_any(gs, alpha=0.05):
        """게임스-하웰: 쌍마다 웰치형 표준오차와 자유도."""
        k = len(gs)
        n = np.array([len(g) for g in gs], float)
        m = np.array([g.mean() for g in gs])
        v = np.array([g.var(ddof=1) for g in gs])
        for i in range(k):
            for j in range(i + 1, k):
                s = v[i] / n[i] + v[j] / n[j]
                df = s**2 / (v[i]**2 / (n[i]**2 * (n[i] - 1))
                             + v[j]**2 / (n[j]**2 * (n[j] - 1)))
                if abs(m[i] - m[j]) / np.sqrt(s / 2) > qsturng(1 - alpha, k, df):
                    return True
        return False

    def tk_any(gs, alpha=0.05):
        """Tukey-Kramer: 합동 MSE 와 공통 자유도."""
        k = len(gs)
        n = np.array([len(g) for g in gs], float)
        m = np.array([g.mean() for g in gs])
        v = np.array([g.var(ddof=1) for g in gs])
        MSE = ((n - 1) * v).sum() / (n.sum() - k)
        qc = qsturng(1 - alpha, k, n.sum() - k)
        for i in range(k):
            for j in range(i + 1, k):
                if abs(m[i] - m[j]) / np.sqrt(MSE / 2 * (1 / n[i] + 1 / n[j])) > qc:
                    return True
        return False

    rng = np.random.default_rng(12006)
    B = 3_000
    print("완전 귀무에서의 FWER (k=4), 명목 0.05")
    print(f"{'설계':>30s} {'튜키':>8s} {'G-H':>8s}")
    for lab, ns, sds in [("등분산·균형 n=15, σ=1", [15] * 4, [1, 1, 1, 1]),
                         ("이분산·균형 σ=(1,1,1,4)", [15] * 4, [1, 1, 1, 4]),
                         ("역페어링 n=(6,15,15,30)", [6, 15, 15, 30], [4, 2, 1, 1]),
                         ("정페어링 n=(30,15,15,6)", [30, 15, 15, 6], [4, 2, 1, 1]),
                         ("극단 역페어링 n=(5,10,20,40)", [5, 10, 20, 40],
                          [5, 3, 2, 1])]:
        a = b = 0
        for _ in range(B):
            gs = [rng.normal(0, s, n) for n, s in zip(ns, sds)]
            a += tk_any(gs)
            b += gh_any(gs)
        print(f"{lab:>30s} {a / B:8.4f} {b / B:8.4f}")
    ```

    ```text
    완전 귀무에서의 FWER (k=4), 명목 0.05
                                설계       튜키      G-H
                  등분산·균형 n=15, σ=1   0.0510   0.0520
                이분산·균형 σ=(1,1,1,4)   0.1037   0.0510
               역페어링 n=(6,15,15,30)   0.2960   0.0480
               정페어링 n=(30,15,15,6)   0.0120   0.0523
            극단 역페어링 n=(5,10,20,40)   0.3620   0.0493
    ```

    **극단 역페어링에서 튜키의 FWER이 0.362다.** 명목의 **7배**다.

    | 설계 | 튜키 | **게임스-하웰** |
    |---|---|---|
    | 등분산·균형 | 0.051 | **0.052** |
    | 이분산·균형 | 0.104 | **0.051** |
    | 역페어링 | **0.296** | **0.048** |
    | 정페어링 | **0.012** | **0.052** |
    | 극단 역페어링 | **0.362** | **0.049** |

    **게임스-하웰은 다섯 상황 모두에서 0.048~0.052다.**

    **균형 설계인데도 튜키가 0.104로 무너진다**(둘째 줄). 이분산만으로도 두 배다. **균형이 이분산을 막아 주는 것은 전체 $F$ 검정이지 쌍별 비교가 아니다.**

    **왜 그런가.** 튜키는 **모든 쌍에 같은 $\text{MSE}$**를 쓴다. $\sigma=(1,1,1,4)$이면 합동 MSE가 $(1+1+1+16)/4=4.75$인데,

    - 세 작은 집단끼리의 비교: **참 분산은 1인데 4.75를 씀** → 지나치게 보수적
    - 큰 집단과의 비교: **참 분산은 8.5인데 4.75를 씀** → 지나치게 자유로움

    **후자가 FWER을 지배**한다. 가장 극단적인 쌍이 FWER을 정하기 때문이다.

    **정페어링의 0.012도 문제다.** 지나치게 보수적이라 **실재하는 차이를 못 잡는다.**

    **등분산일 때 게임스-하웰이 잃는 것은 얼마인가.**

    | 설계 | 튜키 검정력 | G-H 검정력 | 손실 |
    |---|---|---|---|
    | $n=10$, $\mu_4=1.5$ | 0.744 | 0.684 | $-8\%$ |
    | $n=20$, $\mu_4=1.0$ | 0.695 | 0.664 | $-5\%$ |
    | $n=30$, $\mu_4=0.8$ | 0.706 | 0.687 | $-3\%$ |

    **$n$이 커질수록 손실이 줄어든다.** $n\geq20$이면 5% 이하다.

    **권고 — 전체 검정과 사후검정의 가정을 일치시킨다.**

    | 전체 검정 | 사후검정 |
    |---|---|
    | 표준 $F$ | 튜키 HSD |
    | **웰치 $F$** | **게임스-하웰** |

    **웰치로 검정하고 튜키로 사후분석하는 것은 모순**이다. 앞에서 등분산을 가정하지 않다가 뒤에서 가정하는 셈이다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
사후검정 절차의 **선택 지침**을 정리하라.

</div>

??? success "풀이"
    **핵심 수치 다섯.**

    | 사실 | 값 |
    |---|---|
    | $k=10$에서 무보정 FWER | **0.592** |
    | 튜키의 FWER(균형·등분산) | 0.046~0.054 |
    | $k=10$에서 셰페의 FWER(쌍별) | **0.002** |
    | 역페어링에서 튜키의 FWER | **0.362** |
    | 더넷이 튜키보다 얻는 검정력 | $+18\%$ |

    **선택 흐름.**

    ```text
    무엇을 비교하려는가?
        │
        ├─ 모든 쌍 (k(k-1)/2 개)
        │     ├─ 등분산·균형 ──→ 튜키 HSD
        │     └─ 이분산      ──→ 게임스-하웰
        │
        ├─ 대조군과의 비교만 (k-1 개)
        │     └─→ 더넷 (튜키보다 검정력 18% 높다)
        │
        ├─ 미리 정한 소수의 대비
        │     └─→ 홀름 (본페로니를 지배)
        │
        └─ 자료를 보고 고른 대비
              └─→ 셰페 (다른 절차는 무효)
    ```

    **절차 비교표.**

    | 절차 | 통제 범위 | 등분산 | 정확도(쌍별) | 신뢰구간 |
    |---|---|---|---|---|
    | **튜키 HSD** | 모든 쌍 | 필요 | **정확** | **쉽다** |
    | **게임스-하웰** | 모든 쌍 | **불필요** | 정확 | 쉽다 |
    | 본페로니 | 지정한 $M$개 | 필요 | 보수적 | **쉽다** |
    | **홀름** | 지정한 $M$개 | 필요 | 보수적 | **어렵다** |
    | **더넷** | 대조군 대 나머지 | 필요 | **정확** | 쉽다 |
    | **셰페** | **모든 대비** | 필요 | 매우 보수적 | 쉽다 |

    **신뢰구간 열이 실무에서 중요하다.** $p$만 보고할 것이 아니라 **차이의 구간**을 보고해야 하는데, 홀름은 그것을 주지 못한다.

    **자주 하는 실수 다섯.**

    | 실수 | 대가 |
    |---|---|
    | 보정 없이 모든 쌍 비교 | $k=10$에서 FWER 0.59 |
    | 이분산인데 튜키 | FWER 0.36 |
    | 웰치 + 튜키 조합 | 가정이 모순 |
    | 쌍별인데 셰페 | 검정력 낭비 |
    | **자료를 보고 대비를 골라 튜키** | 통제가 무효 |

    **"$F$가 유의해야 사후검정을 한다"는 규칙은?**

    | 절차 | 보호된 $F$가 필요한가 |
    |---|---|
    | **튜키, 게임스-하웰, 더넷** | **불필요**(스스로 FWER을 통제) |
    | 보정 없는 LSD | **필요**(그러나 $k\geq4$에서 실패) |
    | 셰페 | 불필요(정리상 $F$와 동치) |

    **튜키를 쓸 때 $F$ 검정을 먼저 요구하면 오히려 보수적**이 된다. 두 관문을 통과해야 하기 때문이다.

    **보고 형식.**

    ```text
    전체 검정: Welch F(3, 21.4) = 5.82, p = 0.0045
    사후검정: 게임스-하웰 (등분산을 가정하지 않으므로)

      비교          차이     95% 구간        보정 p
      A vs B       2.31   [0.42, 4.20]     0.012
      A vs C      -0.14   [-2.03, 1.75]    0.998
      ...
    ```

    **차이·구간·보정된 $p$를 모두 적는다.** 그리고 **어느 절차를 왜 썼는지** 한 줄로 밝힌다.

    **한 문장.** 사후검정의 선택은 **"무엇을 비교할 것인가"를 먼저 정하는 데서 시작**하며, 그 집합을 자료를 보고 정하는 순간 **셰페 외의 모든 절차가 무효**가 된다.

---

## 정리하며

사후비교 방법들을 **한자리에서** 견주었다.

| 방법 | 보호 대상 | 보수성 | 등분산 가정 |
|---|---|---|---|
| 투키 HSD | 모든 쌍 | 중간 | 예 |
| 본페로니 | 계획된 $m$ 개 | 높음 | 예 |
| 셰페 | 모든 선형 대비 | 가장 높음 | 예 |
| 더넷 | 대조군과의 $k-1$ 개 | 낮음 | 예 |
| 게임스–하월 | 모든 쌍 | 중간 | **아니오** |

- **보호 범위가 좁을수록 검정력이 높다.** 필요한 만큼만 보호하는 것이 요령이며, 모든 쌍이 필요 없는데 투키를 쓰면 손해다.
- **먼저 물어야 할 것은 "무엇을 비교할 것인가"다.** 모든 쌍인가, 대조군과만인가, 사전에 정한 몇 개인가, 자료를 보고 떠오른 대비인가. **이 답이 방법을 정한다.**
- **$\binom k2$ 가 빠르게 커진다.** $k=5$ 면 10 개, $k=10$ 이면 45 개다. 보정 없이 두면 FWER 이 걷잡을 수 없다.
- **등분산 여부가 두 번째 갈림길이다.** 의심스러우면 게임스–하월로 간다.

다음 절부터 **웰치 분산분석**으로 넘어간다. 사후검정만이 아니라 전체 검정 자체를 등분산 없이 하는 방법이다.
