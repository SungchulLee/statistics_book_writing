# 이산분포 모음

## 개요

이 페이지에서는 네 가지 주요 이산분포인 **Binomial**, **Poisson**, **Geometric**, **Hypergeometric** 분포를 실용적인 예와 함께 살펴본다. 각각은 서로 다른 유형의 계수 문제를 모형화하며, 언제 어느 것을 쓸지 아는 것이 응용통계학에서 필수적이다.

---

## 1. 이항분포

각각 성공확률이 $p$인 독립 베르누이 시행 $n$번에서의 성공 횟수:

$$
P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \qquad k = 0, 1, \ldots, n
$$

**예:** 어떤 은행원이 한 달에 50명의 대출 신청자를 만난다. 그중 30%는 신용이 나쁘다.

<div class="codebox" markdown>

### 예제 1. 이항분포 { .eg }

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

n, p = 50, 0.3                 # 시행 50번, 각 시행의 성공확률 0.3
k_vals = np.arange(0, n + 1)   # 가능한 성공 횟수 0~50
pmf = stats.binom.pmf(k_vals, n, p)

# pmf 는 "정확히" 그 값일 확률, cdf 는 "이하"일 확률이다.
# 이산분포에서 이 구분이 특히 중요하다. P(X <= 12) 에는 12가 포함된다.
print(f"P(X = 14) = {stats.binom.pmf(14, n, p):.4f}")
print(f"P(X <= 12) = {stats.binom.cdf(12, n, p):.4f}")
# 평균 np = 15, 분산 np(1-p) = 10.5
print(f"E[X] = {n*p:.1f}, Var(X) = {n*p*(1-p):.2f}")
```

출력:

```
P(X = 14) = 0.1189
P(X <= 12) = 0.2229
E[X] = 15.0, Var(X) = 10.50
```

</div>

---

## 2. 포아송분포

평균 비율이 $\lambda$일 때 고정된 구간에서 일어나는 희귀 사건의 수:

$$
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \qquad k = 0, 1, 2, \ldots
$$

**예:** 어떤 트레이더가 5년 동안 1200번 거래한다. 거래 한 번당 파산 확률이 1/1000이므로 $\lambda = 1.2$이다.

<div class="codebox" markdown>

### 예제 2. 포아송분포 { .eg }

```python
lam = 1.2
print(f"P(X = 2) = {stats.poisson.pmf(2, lam):.4f}")
print(f"P(X > 2) = {1 - stats.poisson.cdf(2, lam):.4f}")
```

출력:

```
P(X = 2) = 0.2169
P(X > 2) = 0.1205
```

</div>

---

## 3. 기하분포

독립 베르누이 시행에서 첫 성공 이전의 실패 횟수:

$$
P(X = k) = (1-p)^k \, p, \qquad k = 0, 1, 2, \ldots
$$

**예:** 성공확률이 $p = 0.3$이다. 첫 성공 이전의 기대 실패 횟수는 $(1-p)/p \approx 2.33$이다.

<div class="codebox" markdown>

### 예제 3. 기하분포 { .eg }

```python
p = 0.3
print(f"P(5 failures before 1st success) = {(1-p)**5 * p:.4f}")
print(f"E[failures] = {(1-p)/p:.2f}")
```

출력:

```
P(5 failures before 1st success) = 0.0504
E[failures] = 2.33
```

</div>

---

## 4. 초기하분포

성공이 $K$개 들어 있는 크기 $N$의 모집단에서 **비복원**으로 $n$개를 뽑을 때의 성공 개수:

$$
P(X = k) = \frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}}
$$

**예:** 모집단 $N = 100$, 불량품 $K = 20$, 추출 $n = 5$.

<div class="codebox" markdown>

### 예제 4. 초기하분포 { .eg }

```python
from scipy import special

N, K, n = 100, 20, 5
p2 = stats.hypergeom.pmf(2, N, K, n)
p2_manual = (special.comb(K, 2) * special.comb(N-K, n-2)) / special.comb(N, n)
print(f"P(X = 2) = {p2:.4f}  (manual = {p2_manual:.4f})")
print(f"E[X] = {n*K/N:.2f}")
```

출력:

```
P(X = 2) = 0.2073  (manual = 0.2073)
E[X] = 1.00
```

</div>

---

## 시각화

<div class="codebox" markdown>

### 예제 5. 네 이산분포 한눈에 보기 { .eg }

```python
# 네 이산분포를 2x2 격자에 나란히 놓는다.
# 서로 어떻게 다른지가 아니라 **무엇을 세는지가** 다르다는 데 주목하라.
#   Binomial      : 시행 수를 정해 놓고 성공 횟수를 센다 (복원추출)
#   Poisson       : 정해진 구간에서 사건 발생 횟수를 센다 (상한 없음)
#   Geometric     : 첫 성공까지 걸린 시행 수를 센다
#   Hypergeometric: 유한 모집단에서 비복원추출했을 때의 성공 횟수
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].bar(k_vals, pmf, color="steelblue", edgecolor="white", alpha=0.8)
axes[0, 0].axvline(n * p, color="red", linestyle="--", label=f"E[X] = {n*p:.0f}")
axes[0, 0].set_title("Binomial(n=50, p=0.3)")
axes[0, 0].set_xlabel("k")
axes[0, 0].legend()

pk = np.arange(0, 10)
axes[0, 1].bar(pk, stats.poisson.pmf(pk, lam), color="seagreen",
               edgecolor="white", alpha=0.8)
axes[0, 1].set_title(f"Poisson(λ={lam})")
axes[0, 1].set_xlabel("k")

gk = np.arange(0, 20)
# gk + 1 을 넣는 이유: scipy의 geom은 시행 번호(1부터)를 받는데
# 여기서는 실패 횟수(0부터)를 가로축으로 쓰고 싶기 때문이다.
axes[1, 0].bar(gk, stats.geom.pmf(gk + 1, 0.3), color="coral",
               edgecolor="white", alpha=0.8)
axes[1, 0].set_title("Geometric(p=0.3)")
axes[1, 0].set_xlabel("k (failures)")

hk = np.arange(0, n + 1)
# 초기하분포: 전체 N개 중 성공이 K개일 때, n개를 **비복원**으로 뽑아
# 성공이 몇 개 나오는가. 뽑을 때마다 남은 구성이 바뀌므로 시행이 독립이 아니다.
# 그 점이 이항분포와의 유일하면서도 결정적인 차이다.
axes[1, 1].bar(hk, stats.hypergeom.pmf(hk, N, K, n), color="mediumpurple",
               edgecolor="white", alpha=0.8)
axes[1, 1].set_title("Hypergeometric(N=100, K=20, n=5)")
axes[1, 1].set_xlabel("k (defectives)")

plt.tight_layout()
plt.show()
```

![Binomial(n=50, p=0.3)](./img/discrete_distributions_suite_95.png)

</div>

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
$np = \lambda$를 고정한 채 $n \to \infty$, $p \to 0$일 때 포아송분포가 이항분포의 극한임을 보여라.

</div>

??? success "풀이"
    $p = \lambda/n$인 $P(X = k) = \binom{n}{k}p^k(1-p)^{n-k}$에서 출발한다:

    $$
    \binom{n}{k}\left(\frac{\lambda}{n}\right)^k\left(1 - \frac{\lambda}{n}\right)^{n-k}
    $$

    $n \to \infty$일 때 $\binom{n}{k}/n^k \to 1/k!$, $(1-\lambda/n)^n \to e^{-\lambda}$, $(1-\lambda/n)^{-k} \to 1$이다. 따라서:

    $$
    P(X = k) \to \frac{\lambda^k}{k!} e^{-\lambda}
    $$

    이것이 Poisson PMF이다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
품질 검사원이 100개(불량 20개)로 이루어진 한 배치에서 5개를 뽑는다. 초기하분포(정확)와 이항분포(근사)로 $P(X = 2)$를 각각 구해 비교하라. 이항 근사는 언제 좋은가?

</div>

??? success "풀이"
    **Hypergeometric:** $P(X=2) = \binom{20}{2}\binom{80}{3}/\binom{100}{5} \approx 0.2075$.

    **Binomial** ($n=5$, $p=0.2$): $P(X=2) = \binom{5}{2}(0.2)^2(0.8)^3 = 10 \times 0.04 \times 0.512 = 0.2048$.

    $n/N = 5/100 = 5\%$로 작기 때문에 근사가 가깝다. 경험 법칙으로, 표본이 모집단의 5–10%보다 작으면 이항분포가 초기하분포를 잘 근사한다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
기하분포가 무기억성을 가짐을 증명하라: $P(X > s + t \mid X > s) = P(X > t)$.

</div>

??? success "풀이"
    여기서 $X$는 첫 성공 이전의 실패 횟수를 센다(0부터 시작). 그러면 $P(X \ge k) = (1-p)^k$이다. 따라서:

    $$
    P(X \ge s+t \mid X \ge s) = \frac{(1-p)^{s+t}}{(1-p)^s} = (1-p)^t = P(X \ge t)
    $$

    $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
초기하분포의 기댓값은 $E[X] = nK/N$이다. 이 결과를 유도하라.

</div>

??? success "풀이"
    $i$번째로 뽑은 물건이 불량이면 $X_i = 1$로 두고 $X = \sum_{i=1}^n X_i$로 쓰자. 대칭성에 의해 각 $i$에 대해 $P(X_i = 1) = K/N$이다(뽑힌 물건이 $N$개 중 어느 것일 확률이 모두 같다). 기댓값의 선형성에 의해:

    $$
    E[X] = \sum_{i=1}^n E[X_i] = n \cdot \frac{K}{N}
    $$

    참고: 비복원추출이므로 $X_i$들은 독립이 아니지만, 기댓값의 선형성에는 독립성이 필요하지 않다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
다음 각 상황에 알맞은 분포를 고르고 모수를 밝혀라.

1. 공장에서 하루 생산한 200개 중 불량이 몇 개인가(각 제품이 독립적으로 2% 확률로 불량).
2. 한 시간 동안 응급실에 몇 명이 오는가(평균 6명).
3. 면접에서 합격자가 나올 때까지 몇 명을 보아야 하는가(합격률 10%).
4. 로또 공 45개 중 당첨번호가 6개일 때, 내가 고른 6개 중 몇 개가 맞는가.

</div>

??? success "풀이"
    1. **이항분포** $\text{Binomial}(200, 0.02)$. 시행 수가 고정이고 각 시행이 독립이며 성공확률이 같다. 참고로 $n$이 크고 $p$가 작아 $\text{Poisson}(4)$로 근사해도 좋다.
    2. **포아송분포** $\text{Poisson}(6)$. 고정된 구간에서 상한 없이 세며, 도착이 서로 독립이다.
    3. **기하분포** $\text{Geometric}(0.1)$. 첫 성공까지의 대기 횟수다. 시행 번호를 세면 평균 10명, 실패 횟수를 세면 평균 9명이다.
    4. **초기하분포** $\text{Hypergeometric}(N=45, K=6, n=6)$. 비복원추출이고 모집단이 작아 이항근사를 쓸 수 없다($n/N = 0.133$).

    **고르는 순서.** 먼저 "시행 횟수가 정해져 있는가"를 묻는다. 정해져 있으면 이항 또는 초기하이고, 그다음 "복원인가"로 갈린다. 정해져 있지 않으면 "구간당 개수를 세는가(포아송)" 아니면 "성공까지 기다리는가(기하·음이항)"로 갈린다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
연습문제 4의 지시함수 표현을 이어받아 초기하분포의 분산

$$
\operatorname{Var}(X) = n\frac{K}{N}\left(1-\frac{K}{N}\right)\frac{N-n}{N-1}
$$

을 유도하라.

</div>

??? success "풀이"
    $p = K/N$으로 두면 $X_i \sim \text{Bernoulli}(p)$이므로 $\operatorname{Var}(X_i) = p(1-p)$이다. $X_i$들이 독립이 아니므로 공분산 항이 살아 있다.

    $$
    \operatorname{Var}(X) = \sum_i \operatorname{Var}(X_i) + \sum_{i \ne j}\operatorname{Cov}(X_i, X_j) = np(1-p) + n(n-1)\operatorname{Cov}(X_1, X_2)
    $$

    (대칭성으로 모든 쌍의 공분산이 같다.)

    $E[X_1X_2] = P(\text{첫째와 둘째가 모두 성공})$인데, 비복원이므로

    $$
    E[X_1X_2] = \frac{K}{N}\cdot\frac{K-1}{N-1}
    $$

    이다. 따라서

    $$
    \operatorname{Cov}(X_1,X_2) = \frac{K(K-1)}{N(N-1)} - \frac{K^2}{N^2} = \frac{K}{N}\cdot\frac{N(K-1) - K(N-1)}{N(N-1)} = -\frac{p(1-p)}{N-1}
    $$

    이다(분자가 $NK - N - KN + K = K - N$로 정리된다). **음수**라는 점이 핵심이다.

    대입하면

    $$
    \operatorname{Var}(X) = np(1-p) - \frac{n(n-1)p(1-p)}{N-1} = np(1-p)\left(1 - \frac{n-1}{N-1}\right) = np(1-p)\frac{N-n}{N-1}
    $$

    이다. $\square$

    이항분포와 **평균은 같은데 분산만 작다**는 사실이 이 유도에서 분명해진다. 차이는 오직 공분산에서 오고, 그 공분산이 음수인 것은 하나를 뽑으면 남은 것이 줄어드는 자기교정 때문이다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
네 분포의 **분산 대 평균 비**를 각각 구하고, 계수 자료를 보았을 때 이 비가 분포 선택에 어떤 단서를 주는지 정리하라.

</div>

??? success "풀이"

    | 분포 | 평균 | 분산 | 분산/평균 |
    |---|---|---|---|
    | 이항 | $np$ | $np(1-p)$ | $1-p < 1$ |
    | 포아송 | $\lambda$ | $\lambda$ | $1$ |
    | 기하(실패 횟수) | $(1-p)/p$ | $(1-p)/p^2$ | $1/p > 1$ |
    | 초기하 | $nK/N$ | $np(1-p)\frac{N-n}{N-1}$ | $(1-p)\frac{N-n}{N-1} < 1$ |

    **포아송의 비가 정확히 1**이고, 이것이 기준선이 된다.

    자료에서 $s^2/\bar x$를 계산해 보면 다음과 같이 읽는다.

    - **1보다 뚜렷이 작다(과소산포).** 개수에 상한이 있거나 자기교정이 있다는 신호다. 이항이나 초기하가 맞을 수 있고, 관측 단위 안에서 사건이 서로를 밀어내는 경우(영역 방어를 하는 동물의 분포 등)도 그렇다.
    - **1에 가깝다.** 포아송이 무난하다.
    - **1보다 크다(과대산포).** 가장 흔한 경우다. 개체마다 발생률이 다르거나(이질성), 사건이 뭉쳐서 일어나거나, 0이 지나치게 많다는 뜻이다. 음이항, 영과잉 모형을 고려한다.

    다만 이 비는 **진단의 출발점이지 결론이 아니다.** 설명변수를 넣은 회귀모형에서는 조건부 분산을 보아야 하며, 설명변수가 이질성을 흡수하면 겉보기 과대산포가 사라지기도 한다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
이항분포를 결과가 셋 이상인 경우로 넓힌 것이 **다항분포**이다. PMF를 적고, 각 성분의 주변분포와 두 성분의 공분산을 구하라.

</div>

??? success "풀이"
    $n$번의 독립 시행에서 각 시행이 범주 $1, \dots, m$ 중 하나를 확률 $p_1, \dots, p_m$($\sum p_j = 1$)으로 취할 때, 각 범주의 개수 $(X_1, \dots, X_m)$의 분포가 다항분포이다.

    $$
    P(X_1=k_1,\dots,X_m=k_m) = \frac{n!}{k_1!\cdots k_m!}p_1^{k_1}\cdots p_m^{k_m}, \qquad \sum_j k_j = n
    $$

    **주변분포.** 범주 $j$인지 아닌지만 보면 이항이므로

    $$
    X_j \sim \text{Binomial}(n, p_j), \qquad E[X_j] = np_j, \quad \operatorname{Var}(X_j) = np_j(1-p_j)
    $$

    이다.

    **공분산.** 지시함수로 쓰면 $X_j = \sum_i \mathbb{1}\{$시행 $i$가 범주 $j\}$이다. 같은 시행에서 두 범주가 동시에 일어날 수 없으므로 $E[\mathbb{1}_j\mathbb{1}_l] = 0$($j \ne l$)이고, 서로 다른 시행은 독립이라 공분산이 0이다. 따라서 한 시행당 공분산이 $0 - p_jp_l$이고

    $$
    \operatorname{Cov}(X_j, X_l) = -np_jp_l \quad (j \ne l)
    $$

    이다. 언제나 **음수**이며, $n$이 고정되어 있어 한 범주가 늘면 다른 범주가 줄어야 하기 때문이다.

    다항분포는 분할표 분석의 밑바탕이다. 카이제곱 적합도 검정의 통계량 $\sum (O_j - E_j)^2/E_j$가 근사적으로 $\chi^2_{m-1}$을 따르는데, 자유도가 $m$이 아니라 $m-1$인 이유가 바로 $\sum X_j = n$이라는 제약, 즉 위의 음의 공분산 구조 때문이다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
$n$번의 시행에서 성공확률 $p$가 시행마다 같지 않고 **묶음마다 다르다**고 하자. $P \sim \text{Beta}(a,b)$이고 $X \mid P = p \sim \text{Binomial}(n,p)$일 때 $X$의 평균과 분산을 구하고, 이항분포와 견주어라.

</div>

??? success "풀이"
    전체기대값 정리와 전체분산 정리를 쓴다. $\text{Beta}(a,b)$의 평균을 $\mu = a/(a+b)$, 분산을 $\sigma_P^2 = \mu(1-\mu)/(a+b+1)$이라 하자.

    **평균.**

    $$
    E[X] = E\{E[X \mid P]\} = E[nP] = n\mu
    $$

    이항분포와 같다.

    **분산.**

    $$
    \operatorname{Var}(X) = E\{\operatorname{Var}(X\mid P)\} + \operatorname{Var}\{E[X\mid P]\} = E[nP(1-P)] + \operatorname{Var}(nP)
    $$

    첫 항은 $n\{\mu - E[P^2]\} = n\{\mu(1-\mu) - \sigma_P^2\}$이고 둘째 항은 $n^2\sigma_P^2$이므로

    $$
    \operatorname{Var}(X) = n\mu(1-\mu) + n(n-1)\sigma_P^2 = n\mu(1-\mu)\left\{1 + \frac{n-1}{a+b+1}\right\}
    $$

    이다.

    **견주면.** 평균은 같은데 분산이 이항분포의 $\{1 + (n-1)/(a+b+1)\}$배다. 언제나 1보다 크므로 **과대산포**다. $a+b \to \infty$이면 $P$가 한 점으로 모여 이항분포로 돌아간다.

    이 분포를 **베타-이항분포**라 하며, 기하분포 절에서 본 감마-포아송 혼합(음이항)과 정확히 같은 구조다. **모수를 확률변수로 두고 섞으면 산포가 커진다.** 일반적으로 전체분산 정리의 둘째 항 $\operatorname{Var}\{E[X\mid P]\}$이 양수인 한 그렇다.

    실제 자료에서 이런 구조가 흔하다. 같은 농장의 닭들, 같은 학급의 학생들, 같은 환자에게서 여러 번 잰 값은 묶음 안에서 닮아 있다. 이를 무시하고 이항 모형을 쓰면 표준오차가 과소평가되어 잘못된 유의성이 나온다. 묶음 자료의 유효 표본크기가 관측 수보다 훨씬 작다는 사실을 반영하려면 베타-이항, 혼합효과 로지스틱, 일반화추정방정식 같은 도구가 필요하다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
`stats.hypergeom`의 인자는 `(M, n, N)`이다. 각각이 무엇을 뜻하는지 확인하고, 본문 예제의 "모집단 100개, 불량 20개, 추출 5개"를 어떻게 넘겨야 하는지 적어라. 어떤 혼동이 생기기 쉬운가?

</div>

??? success "풀이"
    SciPy의 규약은 다음과 같다.

    - `M` — 모집단 전체 크기 (이 책의 $N$)
    - `n` — 모집단 안의 성공 개수 (이 책의 $K$)
    - `N` — 뽑는 개수 (이 책의 $n$)

    따라서 본문 예제는 `stats.hypergeom(M=100, n=20, N=5)`이고, 위치 인자로는 `stats.hypergeom(100, 20, 5)`이다.

    **혼동의 원인은 이름이 뒤집혀 있다는 점이다.** 거의 모든 교재가 모집단 크기를 $N$, 표본 크기를 $n$으로 쓰는데 SciPy는 정확히 반대로 쓴다. `stats.hypergeom(100, 20, 5)`를 "$N=100$, $K=20$, $n=5$"라고 읽으면 우연히 맞지만, 키워드 인자로 `N=100`이라고 쓰면 "100개를 뽑는다"는 뜻이 되어 오류가 난다.

    ```python
    from scipy import stats
    rv = stats.hypergeom(100, 20, 5)     # 안전: 위치 인자로 순서를 지킨다
    print(rv.pmf(2), rv.mean(), rv.var())  # 0.2073  1.0  0.7677
    ```

    이런 함정이 SciPy 곳곳에 있다. 균등분포의 `scale`은 오른쪽 끝점이 아니라 폭이고, 로그정규분포의 `scale`은 평균이 아니라 $e^\mu$이며, 정규분포의 `scale`은 분산이 아니라 표준편차다. **처음 쓰는 분포는 반드시 `mean()`과 `var()`로 검산하는 습관**이 가장 확실한 방어다. 위에서 평균 $nK/N = 5 \times 0.2 = 1.0$이 맞게 나오는 것으로 인자를 제대로 넘겼음을 확인할 수 있다.

---

## 정리하며

네 이산분포는 각각 **다른 종류의 세기 문제**에 대응한다. 고르는 기준은 "무엇이 고정되어 있고 무엇이 확률변수인가"다.

| 분포 | 무엇을 세는가 | 고정된 것 |
|---|---|---|
| 이항 | 시행 $n$ 번의 성공 횟수 | 시행 수 $n$, 확률 $p$ |
| 포아송 | 일정 구간의 사건 수 | 평균 발생률 $\lambda$ |
| 기하 | 첫 성공까지의 시행 수 | 확률 $p$ |
| 초기하 | 비복원 추출에서의 성공 수 | 모집단 구성과 뽑는 수 |

- **이항과 초기하의 갈림길은 복원 여부다.** 모집단이 뽑는 수보다 훨씬 크면($n/N\le0.1$) 초기하가 이항에 가까워지므로 실무에서는 이항으로 근사한다.
- **이항과 포아송.** $n$ 이 크고 $p$ 가 작으며 $np=\lambda$ 가 중간 정도면 이항이 포아송으로 간다. 드문 사건을 셀 때 포아송이 등장하는 이유다.
- **기하분포의 무기억성.** 지금까지 실패했다는 사실이 앞으로 몇 번 더 해야 하는지에 아무 정보도 주지 않는다. 연속형 대응물이 지수분포다.
- **평균과 분산의 관계가 분포를 식별해 준다.** 이항은 분산이 평균보다 작고($np(1-p)<np$), 포아송은 같으며, 자료에서 분산이 평균보다 크면(과산포) 음이항 같은 다른 모형이 필요하다.

다음 절부터 **연속분포**로 넘어간다. 균등분포에서 시작해 지수·정규·$t$·카이제곱·$F$ 로 이어지며, 뒤의 셋은 추론에서 쓰이는 표집분포들이다.
