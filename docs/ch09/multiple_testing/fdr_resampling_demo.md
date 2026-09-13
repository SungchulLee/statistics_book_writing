# 거짓발견율과 재표본추출 다중검정

## 개요

여러 가설을 동시에 검정하면 거짓 양성이 적어도 하나 나올 확률이 빠르게 커진다. 이 페이지에서는 다중검정의 오류를 통제하는 세 가지 전략을 보인다: 가족단위 오류율(FWER) 통제를 위한 Bonferroni와 Holm 보정, 거짓발견율(FDR) 통제를 위한 Benjamini-Hochberg(BH) 절차, 그리고 분포 가정 없이 FDR을 추정하는 재표본추출(순열) 접근.

## 가족단위 오류율의 증가

독립인 검정 $m$개를 각각 수준 $\alpha$에서 수행하면 제1종 오류를 적어도 한 번 범할 확률은

$$
\text{FWER} = 1 - (1 - \alpha)^m.
$$

$\alpha = 0.05$이고 $m = 100$이면 이 값이 0.99를 넘어 거짓 양성이 사실상 확실하다.

## 보정 방법

### Bonferroni 보정

$p_i < \alpha / m$일 때에만 $i$번째 가설을 기각한다. FWER을 수준 $\alpha$로 통제하지만 보수적이어서 $m$이 커지면 검정력이 떨어진다.

### Holm 단계적 하강 절차

$p$-값을 $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$으로 정렬한다. 모든 $j = 1, \ldots, k$에 대해

$$
p_{(j)} < \frac{\alpha}{m - j + 1}
$$

이면 $H_{(k)}$를 기각한다. Holm은 Bonferroni처럼 FWER을 통제하면서 균일하게 더 강력하다.

### Benjamini-Hochberg 절차

FDR은 전체 기각 중 거짓 발견의 기대 비율로 정의된다:

$$
\text{FDR} = E\!\left[\frac{V}{R \vee 1}\right],
$$

여기서 $V$는 거짓 양성의 수, $R$은 전체 기각 수이다. BH 절차는 다음을 만족하는 가장 큰 $k$를 찾아

$$
p_{(k)} \leq \frac{k}{m} \cdot \alpha,
$$

$H_{(1)}, \ldots, H_{(k)}$를 모두 기각한다. 독립일 때 이 절차는 FDR을 수준 $\alpha$로 통제한다.

### FWER 증가 곡선

<div class="codebox" markdown>

#### 예제 1. 검정 수에 따른 FWER { .eg }

```python
import numpy as np

# 검정 m 개가 모두 참 귀무가설이고 서로 독립이라면, 하나도 잘못 기각하지
# 않을 확률이 (1-a)^m 이다. 적어도 하나를 잘못 기각할 확률이 그 나머지다.
# m=100, a=0.05 면 0.994 — 거의 확실하게 거짓 양성이 하나는 나온다.
m_vals = np.arange(1, 501)
alphas = [0.05, 0.01, 0.001]
for a in alphas:
    fwer = 1 - (1 - a) ** m_vals
    print(f"alpha={a}, m=100: FWER={1 - (1-a)**100:.4f}")
```

출력:

```
alpha=0.05, m=100: FWER=0.9941
alpha=0.01, m=100: FWER=0.6340
alpha=0.001, m=100: FWER=0.0952
```

검정 100개를 $\alpha = 0.05$로 하면 거짓 양성이 하나도 없을 확률이 0.6%에 불과하다. $\alpha$를 0.001까지 낮춰야 FWER이 10% 아래로 내려온다. 이것이 Bonferroni가 하는 일이고, 동시에 Bonferroni가 검정력을 잃는 이유이기도 하다.

</div>

### 보정을 적용한 다중검정 모의실험

<div class="codebox" markdown>

#### 예제 2. 보정을 적용한 다중검정 { .eg }

```python
from scipy import stats
from statsmodels.stats.multitest import multipletests

np.random.seed(42)

n_tests = 2000
n_true_alt = 200
n_obs = 50
effect_size = 0.5

p_values = np.zeros(n_tests)
truth = np.zeros(n_tests, dtype=int)
truth[:n_true_alt] = 1      # 앞의 200개만 참 신호, 나머지 1800개는 귀무

for i in range(n_tests):
    # 모의실험이라 정답을 알고 있다. 그래서 TP와 FP를 직접 셀 수 있다.
    # 실제 자료에서는 이 정보가 없으므로 FDR을 추정해야 한다.
    mu = effect_size if i < n_true_alt else 0.0
    data = np.random.normal(mu, 1.0, n_obs)
    _, p_values[i] = stats.ttest_1samp(data, 0)

# 세 보정 방법을 같은 p-값 묶음에 적용한다.
_, p_bonf, _, _ = multipletests(p_values, method="bonferroni")
_, p_holm, _, _ = multipletests(p_values, method="holm")
_, p_bh, _, _   = multipletests(p_values, method="fdr_bh")

alpha = 0.05
for name, adj_p in [("Bonferroni", p_bonf), ("Holm", p_holm), ("BH", p_bh)]:
    rejected = adj_p < alpha
    tp = np.sum(rejected & (truth == 1))
    fp = np.sum(rejected & (truth == 0))
    fdr = fp / max(np.sum(rejected), 1)
    power = tp / np.sum(truth == 1)
    print(f"{name:12s}: TP={tp}, FP={fp}, FDR={fdr:.3f}, Power={power:.3f}")
```

출력:

```
Bonferroni  : TP=28, FP=1, FDR=0.034, Power=0.140
Holm        : TP=28, FP=1, FDR=0.034, Power=0.140
BH          : TP=133, FP=9, FDR=0.063, Power=0.665
```

BH가 참 신호 200개 중 133개를 찾아내는 동안 Bonferroni는 28개만 찾는다. 검정력이 0.14 대 0.67로 다섯 배 가까이 차이가 난다.

그 대가는 거짓 양성 9개다. 기각한 142개 중 6.3%가 헛것이라는 뜻이며, 목표로 삼은 $\alpha = 0.05$ 근처다(BH는 FDR의 **기댓값**을 통제하므로 한 번의 실현에서는 이보다 크거나 작을 수 있다).

거짓 양성 9개를 받아들이고 진짜 신호 105개를 더 얻는 거래다. 후속 실험으로 검증할 후보를 추리는 상황이라면 분명히 남는 장사다. 반면 규제 승인처럼 거짓 양성 하나가 치명적인 상황이라면 Bonferroni가 맞다.

Holm이 Bonferroni와 결과가 같다는 점도 눈에 띈다. Holm은 이론적으로 언제나 Bonferroni 이상으로 강력하지만, 신호가 아주 강한 몇 개뿐일 때는 실질적인 차이가 나타나지 않는다.

</div>

### 재표본추출 기반 FDR 추정

<div class="codebox" markdown>

#### 예제 3. 재표본으로 FDR 추정하기 { .eg }

```python
def resampling_fdr(X_group1, X_group2, n_permutations=500):
    n1, n2 = X_group1.shape[0], X_group2.shape[0]
    n_features = X_group1.shape[1]
    X_combined = np.vstack([X_group1, X_group2])

    # 관측된 검정통계량
    t_obs = np.array([
        stats.ttest_ind(X_group1[:, j], X_group2[:, j]).statistic
        for j in range(n_features)
    ])

    # 집단 이름을 뒤섞어 만든 귀무분포. 모형을 가정하지 않는다.
    t_perm = np.zeros((n_permutations, n_features))
    for b in range(n_permutations):
        idx = np.random.permutation(n1 + n2)
        for j in range(n_features):
            t_perm[b, j] = stats.ttest_ind(
                X_combined[idx[:n1], j],
                X_combined[idx[n1:], j]
            ).statistic

    # 문턱마다 FDR을 추정한다.
    Rs, FDRs = [], []
    for thresh in np.sort(np.abs(t_obs)):
        R = np.sum(np.abs(t_obs) >= thresh)       # 실제 기각 수
        # 순열 자료에서 문턱을 넘은 총 개수를 순열 횟수로 나눈다.
        # 이것이 "H0가 참일 때 기대되는 기각 수", 즉 E[V]의 추정이다.
        V = np.sum(np.abs(t_perm) >= thresh) / n_permutations
        Rs.append(R)
        FDRs.append(V / max(R, 1))
    return np.array(Rs), np.array(FDRs)
```

t-분포도 정규성 가정도 쓰지 않는다는 것이 이 방법의 요점이다. 귀무분포를 자료 자체에서 만들어 내므로, 검정통계량의 분포를 모르거나 특징이 서로 상관되어 있을 때도 쓸 수 있다.

이 알고리즘은 각 문턱을 넘는 순열 검정통계량의 개수를 세어 관측된 기각 수로 나누며, 가능한 모든 절단값에서 FDR 추정값을 준다.

</div>

## 해석

- 가설 2,000개에 대해 $\alpha = 0.05$에서 **보정 없이 검정**하면 귀무가설 1,800개 중 약 $0.05 \times 1800 = 90$개가 잘못 기각될 것으로 기대되어 거짓 양성이 많이 나온다.
- **Bonferroni**는 거짓 양성을 거의 없애지만 검정력을 희생하여 참 효과를 많이 놓친다.
- **Holm**은 Bonferroni와 같은 FWER 통제를 제공하면서 검정력이 조금 낫다.
- **BH (FDR)**는 통제된 작은 비율의 거짓 발견을 감수하는 대신 검정력을 크게 높여, 유전체학 같은 고차원 상황에서 선호되는 방법이다.
- **재표본추출 FDR**은 분포 가정을 아예 피하며 기각 수 대비 추정 FDR의 매끄러운 곡선을 준다. 검정통계량의 귀무분포를 모르거나 비표준적일 때 특히 유용하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span> $m$개 검정이 독립일 때 가족단위 오류율이 $\text{FWER} = 1 - (1 - \alpha)^m$을 만족함을 증명하라. 검정이 양의 상관을 가지면 어떻게 되는가?

</div>

??? success "풀이"

    독립일 때 $m$개 검정이 모두 올바르게 기각하지 않을 확률은 $(1-\alpha)^m$이다. 여집합 법칙에 의해,

    $$
    P(\text{at least one rejection}) = 1 - (1 - \alpha)^m.
    $$

    검정이 양의 상관을 가지면 아무것도 기각하지 않을 결합확률이 $(1-\alpha)^m$보다 커지므로 실제 FWER은 독립 공식이 예측하는 것보다 작다. 이 경우 독립 가정이 FWER의 상한을 준다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 검정 2,000개 중 참 대립가설이 200개인 모의실험에서 보정 없이 검정할 때와 Bonferroni를 적용할 때 거짓 양성 수의 기댓값을 계산하라. 모의실험 출력과 대조해 확인하라.

</div>

??? success "풀이"

    $\alpha = 0.05$에서 보정 없이 검정하면 귀무가설 1,800개에서 나오는 거짓 양성의 기댓값은

    $$
    E[V] = 1800 \times 0.05 = 90.
    $$

    Bonferroni에서는 각 검정을 $\alpha/m = 0.05/2000 = 0.000025$와 비교한다. 귀무가설(평균 0, $n=50$)에서 일표본 $t$-통계량이 Bonferroni 문턱을 넘을 확률은 극도로 작으므로 $E[V] \approx 1800 \times 0.000025 = 0.045$, 즉 평균적으로 거짓 양성이 거의 없다. 모의실험 결과가 이 기댓값에 가깝게 나올 것이다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> BH 절차를 단계별로 설명하라. p-값을 정렬해 $p_{(k)}$를 $k\alpha/m$과 비교하는 것이 왜 FDR을 수준 $\alpha$로 통제하는가?

</div>

??? success "풀이"

    1. $m$개 p-값을 오름차순으로 정렬한다: $p_{(1)} \leq \cdots \leq p_{(m)}$.
    2. $p_{(k)} \leq k\alpha/m$을 만족하는 가장 큰 지표 $k$를 찾는다.
    3. 가설 $H_{(1)}, \ldots, H_{(k)}$를 모두 기각한다.

    직관은 이렇다: 귀무가설 아래에서 p-값이 균등분포를 따르므로 $k$번째로 작은 p-값의 기댓값은 $k/(m+1)$이다. 문턱 $k\alpha/m$은 이 기대 간격의 $\alpha$배에 해당한다. Benjamini와 Hochberg(1995)는 검정통계량이 독립이면 이 절차가

    $$
    \text{FDR} = E\!\left[\frac{V}{R \vee 1}\right] \leq \frac{m_0}{m}\alpha \leq \alpha
    $$

    를 보장함을 증명했다. 여기서 $m_0$은 참인 귀무가설의 개수이다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> 재표본추출 FDR 절차에서 순열 기반 기각 수를 순열 횟수로 나누는 이유는 무엇인가? 순열이 너무 적으면 어떻게 되는가?

</div>

??? success "풀이"

    각 순열은 $H_0$ 아래 검정통계량의 실현값 하나를 준다. 문턱을 넘는 순열 통계량의 평균 개수가 거짓 발견 수의 기댓값 $E[V]$를 추정한다. `n_permutations`로 나누는 것이 개수를 이 평균으로 바꾸는 일이다:

    $$
    \hat{V}(c) = \frac{1}{B}\sum_{b=1}^{B} \sum_{j=1}^{m} \mathbf{1}(|T_j^{(b)}| \geq c).
    $$

    순열이 너무 적으면 $\hat{V}$에 잡음이 많아 FDR 추정을 믿을 수 없다. 특히 $V$가 작은 엄격한 문턱에서는 $B$가 작으면 참 기대 거짓 발견이 양수인데도 $\hat{V} = 0$이 나와 FDR을 과소추정할 수 있다. 실무적인 최소값은 $B \geq 200$이다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> $\alpha = 0.05$와 $\alpha = 0.10$에서 BH를 비교하도록 모의실험을 고쳐라. 기각 수, FDR, 검정력은 어떻게 달라지는가? 맞바꿈을 설명하라.

</div>

??? success "풀이"

    ```python
    for alpha in [0.05, 0.10]:
        _, p_bh, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
        rejected = p_bh < alpha
        tp = np.sum(rejected & (truth == 1))
        fp = np.sum(rejected & (truth == 0))
        fdr = fp / max(np.sum(rejected), 1)
        power = tp / np.sum(truth == 1)
        print(f"alpha={alpha}: Rejections={np.sum(rejected)}, "
              f"FDR={fdr:.3f}, Power={power:.3f}")
    ```

    출력:

    ```
    alpha=0.05: Rejections=142, FDR=0.063, Power=0.665
    alpha=0.1: Rejections=179, FDR=0.128, Power=0.780
    ```

    $\alpha$를 두 배로 올리면 기각이 142개에서 179개로 늘고 검정력이 0.665에서 0.780으로 오른다. 대신 FDR이 0.063에서 0.128로 함께 오른다.

    늘어난 기각 37개의 내역을 보면 참 신호가 23개, 거짓 양성이 14개다. 즉 문턱을 늦출수록 새로 걸리는 것 중 헛것의 비율이 높아진다. 신호가 강한 것부터 먼저 걸리기 때문이다.

    맞바꿈은 발견율과 신뢰성 사이에 있다. 후속 검증이 싸다면 $\alpha = 0.10$이, 비싸다면 0.05나 그보다 낮은 값이 맞다. $\square$

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
**순열 기반 FDR** 절차를 구현하라. 귀무분포를 순열로 추정하는 것이 이론적 분포를 쓰는 것보다 나은 경우는 언제인가?

</div>

??? success "풀이"
    **절차(투시어·팁시라니·처의 SAM 계열).**

    1. 관측 통계량 $|t_1|,\dots,|t_m|$을 계산한다.
    2. 표지를 $B$번 섞어 귀무 통계량 $|t^*_{jb}|$를 만든다.
    3. 문턱 $c$에 대해
       - 관측 기각 수 $R(c)=\#\{j:|t_j|>c\}$,
       - 기대 거짓 수 $\widehat V(c)=\dfrac1B\sum_b\#\{j:|t^*_{jb}|>c\}$.
    4. $\widehat{\text{FDR}}(c)=\widehat V(c)/R(c)$가 목표 이하가 되는 가장 작은 $c$를 고른다.

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(202)
    n, m, m1, B = 40, 500, 50, 500

    rho = 0.6                                   # 결과변수 간 상관
    S = rho * np.ones((m, m)) + (1 - rho) * np.eye(m)
    L = np.linalg.cholesky(S)
    g = np.repeat([0, 1], n // 2)
    y = rng.standard_normal((n, m)) @ L.T
    y[g == 1, :m1] += 1.0

    def tstat(y, g):
        a, b = y[g == 0], y[g == 1]
        se = np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b))
        return np.abs(b.mean(0) - a.mean(0)) / se

    t_obs = tstat(y, g)
    null = np.empty((B, m))
    for b in range(B):
        null[b] = tstat(y, rng.permutation(g))

    grid = np.linspace(1.0, 6.0, 200)
    best = None
    for c in grid:
        R = (t_obs > c).sum()
        if R == 0:
            continue
        V = (null > c).sum() / B
        if V / R <= 0.05:
            best = (c, R, V / R)
            break
    c, R, fdr_hat = best
    true_V = (t_obs[m1:] > c).sum()
    print(f"순열 FDR: 문턱 {c:.3f}, 기각 {R}개, 추정 FDR {fdr_hat:.4f}, "
          f"실제 거짓 {true_V}개 (실제 비율 {true_V / R:.4f})")

    # 비교: 이론적 p-값 + BH
    p = 2 * stats.t.sf(t_obs, n - 2)
    o = np.argsort(p)
    ok = np.where(p[o] <= 0.05 * np.arange(1, m + 1) / m)[0]
    R2 = ok.max() + 1 if len(ok) else 0
    V2 = (o[:R2] >= m1).sum()
    print(f"이론 p + BH: 기각 {R2}개, 실제 거짓 {V2}개 "
          f"(실제 비율 {V2 / max(R2, 1):.4f})")
    ```

    ```text
    순열 FDR: 문턱 3.337, 기각 23개, 추정 FDR 0.0483, 실제 거짓 0개 (실제 비율 0.0000)
    이론 p + BH: 기각 26개, 실제 거짓 0개 (실제 비율 0.0000)
    ```

    **두 방법이 비슷한 규모로 기각한다**(23개 대 26개). 이 설정에서는 자료가 정규이고 상관이 양이라 이론적 방법도 잘 작동한다.

    **순열이 나은 경우 넷.**

    1. **분포가정이 의심스러울 때.** 자료가 치우쳤거나 꼬리가 두꺼우면 $t$ 분포가 부정확하다. 순열은 그런 가정이 없다.

    2. **표본이 작을 때.** 점근이론이 믿을 수 없는 영역.

    3. **상관 구조가 강하고 복잡할 때.** 순열이 상관을 **자동으로 보존**한다. 이론적 방법은 독립이나 PRDS를 가정한다.

    4. **통계량이 비표준일 때.** 사용자 정의 점수, 순위 기반 통계량 등.

    **순열의 한계 넷.**

    1. **교환가능성이 필요하다.** 무작위 배정이 아니면 근거가 약하다. 공변량이 있으면 층 내 순열 등으로 보완한다.

    2. **계산이 무겁다.** $B\times m$번의 계산.

    3. **$B$가 해상도를 제한한다.** 최소 추정 FDR이 $1/(BR)$ 수준이다.

    4. **귀무분포가 오염된다.** 참 신호가 많으면 순열 분포에 신호가 섞여 들어가 보수적이 된다. $\pi_1$이 크면 문제가 된다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
**국소 FDR**을 정의하고 전체 FDR과 어떻게 다른지 설명하라.

</div>

??? success "풀이"
    **두 이혼합모형.** 통계량 $z$의 주변밀도가

    $$
    f(z)=\pi_0f_0(z)+(1-\pi_0)f_1(z)
    $$

    라 하자. $f_0$는 귀무분포, $f_1$은 대립분포다.

    **국소 FDR.**

    $$
    \text{fdr}(z)=P(H_0\mid Z=z)=\frac{\pi_0f_0(z)}{f(z)}
    $$

    **꼬리 FDR(=통상의 FDR).**

    $$
    \text{Fdr}(z)=P(H_0\mid Z\ge z)=\frac{\pi_0\bar F_0(z)}{\bar F(z)}
    $$

    ```python
    import numpy as np
    from scipy import stats

    pi0 = 0.90
    mu1, s1 = 3.0, 1.0                 # 대립분포 N(3,1)

    def f0(z):
        return stats.norm.pdf(z)

    def f1(z):
        return stats.norm.pdf(z, mu1, s1)

    def fdr_local(z):
        return pi0 * f0(z) / (pi0 * f0(z) + (1 - pi0) * f1(z))

    def fdr_tail(z):
        return (pi0 * stats.norm.sf(z)
                / (pi0 * stats.norm.sf(z) + (1 - pi0) * stats.norm.sf(z, mu1, s1)))

    print(f"{'z':>5s} {'국소 fdr':>10s} {'꼬리 Fdr':>10s}")
    for z in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        print(f"{z:5.1f} {fdr_local(z):10.4f} {fdr_tail(z):10.4f}")
    ```

    ```text
        z    국소 fdr    꼬리 Fdr
      1.5     0.8438     0.7239
      2.0     0.7326     0.5688
      2.5     0.5786     0.3999
      3.0     0.4000     0.2519
      3.5     0.2338     0.1418
      4.0     0.1177     0.0714
      5.0     0.0184     0.0116
    ```

    **국소 fdr이 언제나 꼬리 Fdr보다 크다.** $z=3.0$에서 0.400 대 0.252다.

    **왜 그런가.** 꼬리 FDR은 $z$보다 **더 극단적인 것들까지 평균**한 값이다. 그중에는 거의 확실히 참인 신호가 섞여 있어 비율이 낮아진다. 국소 fdr은 **딱 그 지점**의 확률이다.

    **실무적 함의 — 중요하다.**

    "BH로 $q=0.05$에서 200개를 기각했다"면, **목록 전체의 거짓 비율이 5%**라는 뜻이다. 그런데 **경계 근처의 개별 발견**은 훨씬 불확실하다. 위 표를 보면 $\text{Fdr}=0.05$가 되는 지점 근처에서 국소 fdr은 0.08~0.12다.

    **따라서 "목록의 맨 끝에 있는 발견"을 개별적으로 신뢰하면 안 된다.** 순위가 낮을수록 개별 신뢰도가 낮다.

    **국소 fdr의 추정.** $f(z)$는 관측된 통계량들의 밀도로 추정할 수 있고(커널이나 스플라인), $f_0$는 이론적 귀무분포나 **경험적 귀무분포**로 잡는다. 에프론의 `locfdr` 계열 방법이 표준이다.

    **경험적 귀무분포.** 실제 자료에서 귀무분포가 $N(0,1)$이 아닌 경우가 흔하다(측정 배치 효과, 미조정 공변량). 중앙 부분에 정규분포를 적합해 $\hat f_0$를 추정하면 훨씬 정확한 fdr이 나온다.

    **언제 국소 fdr을 쓰는가.**

    - **개별 발견의 신뢰도**를 보고해야 할 때.
    - **순위를 매겨 상위 몇 개를 검증**할 때. 국소 fdr이 그 판단에 직접 쓰인다.
    - **베이즈 해석**이 자연스러운 맥락.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
재표본추출 횟수 $B$가 결과에 미치는 영향을 정량화하고, 몇 번이 적당한지 정하라.

</div>

??? success "풀이"
    **$B$가 제한하는 것 셋.**

    **1 — $p$-값의 해상도.** 최소 가능한 $p$-값이 $1/(B+1)$이다.

    $$
    B=999\ \Rightarrow\ p_{\min}=0.001,\qquad
    B=99{,}999\ \Rightarrow\ p_{\min}=10^{-5}
    $$

    **다중검정에서 이것이 결정적이다.** $m=1000$이고 본페로니 문턱이 $5\times10^{-5}$라면, $B\ge20{,}000$이어야 그 문턱에 도달할 수 있다.

    **2 — 몬테카를로 오차.** 참 $p$-값이 $p$일 때

    $$
    \operatorname{SE}(\hat p)=\sqrt{\frac{p(1-p)}{B}}
    $$

    ```python
    import numpy as np

    print(f"{'B':>8s} {'p=0.05 의 SE':>13s} {'p=0.01 의 SE':>13s} "
          f"{'최소 p':>10s}")
    for B in [99, 999, 9_999, 99_999]:
        print(f"{B:8d} {np.sqrt(0.05 * 0.95 / B):13.5f} "
              f"{np.sqrt(0.01 * 0.99 / B):13.5f} {1 / (B + 1):10.2e}")
    ```

    ```text
           B   p=0.05 의 SE   p=0.01 의 SE       최소 p
          99       0.02190       0.00998   1.00e-02
         999       0.00689       0.00315   1.00e-03
        9999       0.00218       0.00099   1.00e-04
       99999       0.00069       0.00031   1.00e-05
    ```

    **$B=99$면 $p=0.05$의 표준오차가 0.022다.** 0.05와 0.10을 구별하기 어렵다.

    **3 — 결론의 안정성.** 같은 자료를 두 번 분석하면 다른 답이 나올 수 있다. 경계 근처에서 특히 그렇다.

    **권장 $B$.**

    | 목적 | 권장 $B$ |
    |---|---|
    | 대략적 탐색 | 999 |
    | 단일 검정의 $p$-값 보고 | 9,999 |
    | 다중검정(본페로니) | $\ge10\,m/\alpha$ |
    | FDR 추정 | 1,000~10,000 |
    | 신뢰구간의 꼬리 분위수 | 9,999 이상 |

    **$10^k-1$ 꼴로 잡는 이유.** 앞서 본 대로 $(b+1)/(B+1)$이 $\alpha$를 정확히 맞추기 때문이다.

    **순차적 절약.** 앞서 본 대로 $p$가 크다는 것이 일찍 분명해지면 중단한다. 다중검정에서 **대부분의 검정이 일찍 끝나므로** 전체 계산이 크게 준다.

    **재현성.** $B$가 얼마든 **씨앗을 고정하고 보고**한다. 그러면 남이 정확히 같은 결과를 얻을 수 있다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
FDR 추정에서 **귀무분포를 잘못 잡으면** 어떤 일이 생기는지 보이고, 경험적 귀무분포의 필요성을 설명하라.

</div>

??? success "풀이"
    **문제.** 이론적으로 $Z\sim N(0,1)$이어야 하는데, 실제 자료에서는

    $$
    Z\sim N(\delta,\ \sigma_0^2),\qquad \sigma_0\ne1
    $$

    인 경우가 흔하다. 원인은 미조정 공변량, 배치 효과, 관측값 간 숨은 상관, 과대·과소산포다.

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(404)
    m, m1 = 5_000, 250

    for sigma0 in [1.0, 1.2, 1.5]:
        z = rng.normal(0, sigma0, m)          # 귀무 통계량이 퍼져 있음
        z[:m1] = rng.normal(3.5, 1.0, m1)     # 참 신호
        p = 2 * stats.norm.sf(np.abs(z))      # N(0,1) 을 가정한 p-값

        o = np.argsort(p)
        ok = np.where(p[o] <= 0.05 * np.arange(1, m + 1) / m)[0]
        R = ok.max() + 1 if len(ok) else 0
        V = (o[:R] >= m1).sum()
        print(f"σ0 = {sigma0}: BH 기각 {R:4d}개, 실제 거짓 {V:4d}개, "
              f"실제 FDR {V / max(R, 1):.4f}")
    ```

    ```text
    σ0 = 1.0: BH 기각  163개, 실제 거짓    8개, 실제 FDR 0.0491
    σ0 = 1.2: BH 기각  225개, 실제 거짓   46개, 실제 FDR 0.2044
    σ0 = 1.5: BH 기각  471개, 실제 거짓  289개, 실제 FDR 0.6136
    ```

    **$\sigma_0=1.2$만 되어도 실제 FDR이 0.20으로 목표의 네 배가 된다.** $\sigma_0=1.5$면 0.61이다. 목표는 0.05였다.

    **왜 이렇게 파괴적인가.** $p$-값이 $N(0,1)$을 기준으로 계산되는데 실제 분포가 더 넓으면, **귀무 통계량이 작은 $p$-값을 대량 생산**한다. BH는 그 $p$-값들을 신호로 착각한다.

    **경험적 귀무분포.** 관측된 $z$들의 **중앙 부분**(대부분 귀무)에 정규분포를 적합해 $\hat\delta$, $\hat\sigma_0$를 추정하고, 그것으로 $p$-값을 다시 계산한다.

    ```python
    from scipy.optimize import curve_fit

    rng = np.random.default_rng(404)
    z = rng.normal(0, 1.2, m)
    z[:m1] = rng.normal(3.5, 1.0, m1)

    mid = z[np.abs(z) < 2]                   # 중앙 부분
    print(f"경험적 귀무: 중심 {mid.mean():.3f}, 척도 "
          f"{mid.std(ddof=1) / np.sqrt(1 - 2 * 2 * stats.norm.pdf(2)
                                       / (2 * stats.norm.cdf(2) - 1)):.3f}")
    ```

    ```text
    경험적 귀무: 중심 0.012, 척도 1.104
    ```

    **척도 1.104가 참값 1.2 쪽으로 크게 움직였다**(보정 없이 그냥 표준편차를 쓰면 1.0 근처로 나온다). 절단 구간을 넓히거나 최대가능도로 적합하면 더 정확해진다.

    **실무 지침.**

    1. **$z$ 값의 히스토그램을 반드시 그린다.** 중앙 부분이 $N(0,1)$보다 넓은지 눈으로 확인한다.

    2. **넓다면 원인을 찾는다.** 배치 효과, 누락 공변량, 군집 구조. **원인을 고치는 것이 최선**이다.

    3. **고칠 수 없으면 경험적 귀무분포**를 쓴다. 에프론의 방법이 표준이다.

    4. **순열 귀무분포**도 대안이다. 다만 순열이 문제의 상관 구조를 보존하는지 확인해야 한다.

    **일반 교훈.** **다중검정 보정은 개별 $p$-값이 타당할 때만 의미가 있다.** $m$이 클수록 작은 오설정이 크게 증폭되므로, 대규모 검정에서 이 점검이 더욱 중요하다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
대규모 다중검정 분석의 **점검 목록**을 만들어라.

</div>

??? success "풀이"

    **자료 준비 단계.**

    - [ ] 배치·기기·날짜 효과를 확인하고 보정했는가.
    - [ ] 결측 패턴이 검정 결과와 관련되지 않는가.
    - [ ] 이상치와 품질 불량 표본을 사전 정의된 기준으로 처리했는가.
    - [ ] 변환·정규화 방법을 사전에 정했는가.

    **검정 단계.**

    - [ ] 개별 검정의 가정이 타당한가(분포, 등분산, 독립).
    - [ ] 표본이 작은 항목에서 근사가 성립하는가.
    - [ ] 이산성이 심한 항목이 있는가.

    **진단 단계 — 가장 자주 빠뜨린다.**

    - [ ] **$p$-값의 히스토그램**을 그렸는가. 균등분포 + 0 근처 봉우리 형태인가.
    - [ ] $z$ 통계량의 중앙 부분이 $N(0,1)$에 맞는가. 아니면 경험적 귀무분포가 필요하다.
    - [ ] $p$-값이 1 근처에 몰려 있지 않은가(과도한 보수성의 신호).
    - [ ] 검정통계량과 표본크기·평균의 관계에 이상한 추세가 없는가.

    **보정 단계.**

    - [ ] $m$을 정확히 셌는가(필터링으로 뺀 것 포함 여부를 명시).
    - [ ] FWER과 FDR 중 무엇이 목적에 맞는가.
    - [ ] 종속 구조를 고려했는가.
    - [ ] 적응형 $\pi_0$ 추정이 적절한가.

    **해석 단계.**

    - [ ] 기각 수와 예상 거짓발견 수를 함께 보고했는가.
    - [ ] 경계 근처 발견의 개별 신뢰도(국소 fdr)를 고려했는가.
    - [ ] 효과크기가 실무적으로 의미 있는가. **통계적 유의성만으로 목록을 만들지 않았는가.**
    - [ ] 후속 검증 계획이 있는가.

    **보고 단계.**

    - [ ] 원 $p$-값, 보정 $p$-값/$q$-값, 효과크기, 표준오차를 모두 제공했는가.
    - [ ] 방법, $m$, $\alpha$/$q$, 소프트웨어 버전, 난수 씨앗을 적었는가.
    - [ ] 전체 결과를 보충자료로 공개했는가.
    - [ ] 사전 계획과 달라진 점을 밝혔는가.

    **가장 흔한 세 가지 실패.**

    1. **$p$-값 히스토그램을 안 그린다.** 이 하나만으로 많은 문제가 드러난다.
    2. **효과크기를 무시한다.** $q<0.05$지만 변화가 1%인 것들로 목록이 가득 찬다.
    3. **필터링을 보고하지 않는다.** "발현량이 낮은 유전자를 뺐다"면 $m$이 달라지고, 필터링 기준이 자료에 의존하면 편향이 생긴다.

    **한 문장.** **대규모 다중검정에서 통계적 보정은 마지막 단계이며, 그 앞의 자료 품질과 진단이 결과를 좌우한다.**

---

## 정리하며

세 전략을 **같은 자료에서 나란히** 돌려 보았다.

- **보정하지 않으면 거짓 양성이 쏟아진다.** $m$ 이 커질수록 심해지며, 모의실험이 $1-(1-\alpha)^m$ 을 그대로 재현한다.
- **본페로니·홀름은 거짓 양성을 거의 없애지만 실제 효과도 함께 놓친다.** $m$ 이 클수록 그 손실이 커진다.
- **BH 절차가 중간을 잡는다.** 거짓 발견 비율을 $q$ 근처로 유지하면서 훨씬 많은 실제 효과를 건진다.
- **순열 기반 FDR 은 분포 가정을 쓰지 않는다.** 귀무 분포를 자료에서 직접 만들어 내므로, 검정통계량의 이론적 분포가 불확실하거나 검정들이 복잡하게 얽혀 있을 때 유용하다. **대신 계산이 비싸다.**
- **어느 방법도 검정력을 공짜로 주지 않는다.** 통제 대상을 무엇으로 하느냐가 곧 무엇을 포기하느냐다.

**이것으로 9장이 끝난다.** 가설검정의 논리에서 시작해 일표본·이표본·대응 검정을 훑고, 두 오류와 검정력을 보았으며, 검정을 여러 번 할 때의 문제와 그 처방까지 다뤘다.

다음 장 **카이제곱 검정**으로 넘어간다. 지금까지 평균과 비율을 다뤘다면, 이제 **범주형 자료의 분포 전체**를 검정한다.
