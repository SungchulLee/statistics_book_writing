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

## 코드

### FWER 증가 곡선

```python
import numpy as np

m_vals = np.arange(1, 501)
alphas = [0.05, 0.01, 0.001]
for a in alphas:
    fwer = 1 - (1 - a) ** m_vals
    print(f"alpha={a}, m=100: FWER={1 - (1-a)**100:.4f}")
```

### 보정을 적용한 다중검정 모의실험

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
truth[:n_true_alt] = 1

for i in range(n_tests):
    mu = effect_size if i < n_true_alt else 0.0
    data = np.random.normal(mu, 1.0, n_obs)
    _, p_values[i] = stats.ttest_1samp(data, 0)

# Apply corrections
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

### 재표본추출 기반 FDR 추정

```python
def resampling_fdr(X_group1, X_group2, n_permutations=500):
    n1, n2 = X_group1.shape[0], X_group2.shape[0]
    n_features = X_group1.shape[1]
    X_combined = np.vstack([X_group1, X_group2])

    # Observed test statistics
    t_obs = np.array([
        stats.ttest_ind(X_group1[:, j], X_group2[:, j]).statistic
        for j in range(n_features)
    ])

    # Permutation null distribution
    t_perm = np.zeros((n_permutations, n_features))
    for b in range(n_permutations):
        idx = np.random.permutation(n1 + n2)
        for j in range(n_features):
            t_perm[b, j] = stats.ttest_ind(
                X_combined[idx[:n1], j],
                X_combined[idx[n1:], j]
            ).statistic

    # Estimate FDR at each threshold
    Rs, FDRs = [], []
    for thresh in np.sort(np.abs(t_obs)):
        R = np.sum(np.abs(t_obs) >= thresh)
        V = np.sum(np.abs(t_perm) >= thresh) / n_permutations
        Rs.append(R)
        FDRs.append(V / max(R, 1))
    return np.array(Rs), np.array(FDRs)
```

이 알고리즘은 각 문턱을 넘는 순열 검정통계량의 개수를 세어 관측된 기각 수로 나누며, 가능한 모든 절단값에서 FDR 추정값을 준다.

## 해석

- 가설 2,000개에 대해 $\alpha = 0.05$에서 **보정 없이 검정**하면 귀무가설 1,800개 중 약 $0.05 \times 1800 = 90$개가 잘못 기각될 것으로 기대되어 거짓 양성이 많이 나온다.
- **Bonferroni**는 거짓 양성을 거의 없애지만 검정력을 희생하여 참 효과를 많이 놓친다.
- **Holm**은 Bonferroni와 같은 FWER 통제를 제공하면서 검정력이 조금 낫다.
- **BH (FDR)**는 통제된 작은 비율의 거짓 발견을 감수하는 대신 검정력을 크게 높여, 유전체학 같은 고차원 상황에서 선호되는 방법이다.
- **재표본추출 FDR**은 분포 가정을 아예 피하며 기각 수 대비 추정 FDR의 매끄러운 곡선을 준다. 검정통계량의 귀무분포를 모르거나 비표준적일 때 특히 유용하다.

## 연습문제

**연습문제 1.** $m$개 검정이 독립일 때 가족단위 오류율이 $\text{FWER} = 1 - (1 - \alpha)^m$을 만족함을 증명하라. 검정이 양의 상관을 가지면 어떻게 되는가?

??? success "풀이"

    독립일 때 $m$개 검정이 모두 올바르게 기각하지 않을 확률은 $(1-\alpha)^m$이다. 여집합 법칙에 의해,

    $$
    P(\text{at least one rejection}) = 1 - (1 - \alpha)^m.
    $$

    검정이 양의 상관을 가지면 아무것도 기각하지 않을 결합확률이 $(1-\alpha)^m$보다 커지므로 실제 FWER은 독립 공식이 예측하는 것보다 작다. 이 경우 독립 가정이 FWER의 상한을 준다. $\square$

---

**연습문제 2.** 검정 2,000개 중 참 대립가설이 200개인 모의실험에서 보정 없이 검정할 때와 Bonferroni를 적용할 때 거짓 양성 수의 기댓값을 계산하라. 모의실험 출력과 대조해 확인하라.

??? success "풀이"

    $\alpha = 0.05$에서 보정 없이 검정하면 귀무가설 1,800개에서 나오는 거짓 양성의 기댓값은

    $$
    E[V] = 1800 \times 0.05 = 90.
    $$

    Bonferroni에서는 각 검정을 $\alpha/m = 0.05/2000 = 0.000025$와 비교한다. 귀무가설(평균 0, $n=50$)에서 일표본 $t$-통계량이 Bonferroni 문턱을 넘을 확률은 극도로 작으므로 $E[V] \approx 1800 \times 0.000025 = 0.045$, 즉 평균적으로 거짓 양성이 거의 없다. 모의실험 결과가 이 기댓값에 가깝게 나올 것이다. $\square$

---

**연습문제 3.** BH 절차를 단계별로 설명하라. p-값을 정렬해 $p_{(k)}$를 $k\alpha/m$과 비교하는 것이 왜 FDR을 수준 $\alpha$로 통제하는가?

??? success "풀이"

    1. $m$개 p-값을 오름차순으로 정렬한다: $p_{(1)} \leq \cdots \leq p_{(m)}$.
    2. $p_{(k)} \leq k\alpha/m$을 만족하는 가장 큰 지표 $k$를 찾는다.
    3. 가설 $H_{(1)}, \ldots, H_{(k)}$를 모두 기각한다.

    직관은 이렇다: 귀무가설 아래에서 p-값이 균등분포를 따르므로 $k$번째로 작은 p-값의 기댓값은 $k/(m+1)$이다. 문턱 $k\alpha/m$은 이 기대 간격의 $\alpha$배에 해당한다. Benjamini와 Hochberg(1995)는 검정통계량이 독립이면 이 절차가

    $$
    \text{FDR} = E\!\left[\frac{V}{R \vee 1}\right] \leq \frac{m_0}{m}\alpha \leq \alpha
    $$

    를 보장함을 증명했다. 여기서 $m_0$은 참인 귀무가설의 개수이다. $\square$

---

**연습문제 4.** 재표본추출 FDR 절차에서 순열 기반 기각 수를 순열 횟수로 나누는 이유는 무엇인가? 순열이 너무 적으면 어떻게 되는가?

??? success "풀이"

    각 순열은 $H_0$ 아래 검정통계량의 실현값 하나를 준다. 문턱을 넘는 순열 통계량의 평균 개수가 거짓 발견 수의 기댓값 $E[V]$를 추정한다. `n_permutations`로 나누는 것이 개수를 이 평균으로 바꾸는 일이다:

    $$
    \hat{V}(c) = \frac{1}{B}\sum_{b=1}^{B} \sum_{j=1}^{m} \mathbf{1}(|T_j^{(b)}| \geq c).
    $$

    순열이 너무 적으면 $\hat{V}$에 잡음이 많아 FDR 추정을 믿을 수 없다. 특히 $V$가 작은 엄격한 문턱에서는 $B$가 작으면 참 기대 거짓 발견이 양수인데도 $\hat{V} = 0$이 나와 FDR을 과소추정할 수 있다. 실무적인 최소값은 $B \geq 200$이다. $\square$

---

**연습문제 5.** $\alpha = 0.05$와 $\alpha = 0.10$에서 BH를 비교하도록 모의실험을 고쳐라. 기각 수, FDR, 검정력은 어떻게 달라지는가? 맞바꿈을 설명하라.

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

    $\alpha$를 0.05에서 0.10으로 올리면 BH 문턱이 높아져 더 많은 가설이 기각된다. 검정력이 커지지만(참 효과를 더 많이 탐지하지만) FDR도 함께 올라간다(발견 중 잘못된 것의 비율이 커진다). 맞바꿈은 발견율과 신뢰성 사이에 있다: $\alpha$가 높으면 더 많은 효과를 찾지만 잘못된 단서도 늘어난다. $\square$
