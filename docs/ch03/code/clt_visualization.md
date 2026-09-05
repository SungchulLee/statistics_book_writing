# 중심극한정리 다중 분포 시각화

## 개요

중심극한정리(CLT)는 모집단의 평균과 분산이 유한하기만 하면 모집단 분포가 무엇이든 $n$이 커질수록 표본평균의 표본분포가 정규분포로 수렴한다고 말한다. 이 절에서는 정규분포가 아닌 세 분포에서 표본을 반복해서 뽑고 그 결과로 얻은 $\bar{X}$의 표본분포를 그려 중심극한정리를 시각적으로 보인다.

---

## 설정

뚜렷하게 비정규인 모양을 갖는 세 모집단 분포를 사용한다.

| 분포 | 모양 | 평균 | 분산 |
|---|---|---|---|
| Uniform(2, 8) | 평평하고 대칭 | 5 | 3 |
| Beta(6, 2) | 왼쪽으로 치우침, $[0, 1]$에서 유계 | 0.75 | 0.0208 |
| Gamma(6, 1) | 오른쪽으로 치우침, 유계 아님 | 6 | 6 |

각 분포와 각 표본 크기 $n \in \{2, 10, 100\}$에 대해 독립적인 표본을 2000개 뽑아 각각의 표본평균을 계산하고 그 히스토그램을 그린다.

---

## 표집 절차

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
N_REPS = 2000

def sample_means(dist_rvs, sample_sizes, n_reps=N_REPS):
    """For each sample size, draw n_reps samples and return their means."""
    results = {}
    for n in sample_sizes:
        means = np.array([dist_rvs(n).mean() for _ in range(n_reps)])
        results[n] = means
    return results
```

핵심 발상: `results[n]`의 각 항목이 $\bar{X}_n$의 한 실현값이다. 이런 실현값 2000개를 그리면 $\bar{X}_n$의 **표본분포**를 근사하게 된다.

---

## 분포와 시각화

```python
sample_sizes = [2, 10, 100]

distributions = {
    "Uniform(2, 8)": {
        "rvs": lambda n: np.random.uniform(2, 8, n),
        "color": "tomato",
        "pop_x": np.linspace(2, 8, 200),
        "pop_pdf": lambda x: np.ones_like(x) / 6,
    },
    "Beta(6, 2)": {
        "rvs": lambda n: stats.beta.rvs(6, 2, size=n),
        "color": "seagreen",
        "pop_x": np.linspace(0, 1, 200),
        "pop_pdf": lambda x: stats.beta.pdf(x, 6, 2),
    },
    "Gamma(6, 1)": {
        "rvs": lambda n: stats.gamma.rvs(6, size=n),
        "color": "steelblue",
        "pop_x": np.linspace(0, 25, 200),
        "pop_pdf": lambda x: stats.gamma.pdf(x, 6),
    },
}

n_dists = len(distributions)
n_rows = 1 + len(sample_sizes)
fig, axes = plt.subplots(n_rows, n_dists, figsize=(6 * n_dists, 4 * n_rows))

for col, (name, d) in enumerate(distributions.items()):
    c = d["color"]

    # Row 0: population PDF
    ax = axes[0, col]
    ax.plot(d["pop_x"], d["pop_pdf"](d["pop_x"]), lw=3, color=c)
    ax.fill_between(d["pop_x"], d["pop_pdf"](d["pop_x"]), alpha=0.3, color=c)
    ax.set_title(name, fontsize=14, fontweight="bold")
    if col == 0:
        ax.set_ylabel("Population PDF", fontsize=11)

    # Rows 1–3: sampling distributions of the mean
    means_dict = sample_means(d["rvs"], sample_sizes)
    for row, n in enumerate(sample_sizes, start=1):
        ax = axes[row, col]
        ax.hist(means_dict[n], bins=30, color=c, alpha=0.5,
                edgecolor="white", density=True)
        ax.set_title(f"n = {n}", fontsize=11)
        if col == 0:
            ax.set_ylabel(f"Sampling Dist (n={n})", fontsize=10)

plt.suptitle("Central Limit Theorem: Sampling Distribution of x̄",
             fontsize=15, y=1.01)
plt.tight_layout()
plt.show()
```

그림은 $4 \times 3$ 격자로 되어 있다. 맨 윗줄이 세 모집단 분포를 보여주고, 나머지 줄이 $n = 2, 10, 100$에 대한 $\bar{X}$의 표본분포를 보여준다.

---

## 해석

### 눈여겨볼 점

- **맨 윗줄:** 세 모집단 분포가 눈에 띄게 비정규다. 각각 평평하고, 왼쪽으로 치우쳤고, 오른쪽으로 치우쳤다.
- **$n = 2$:** 표본분포가 여전히 모집단의 모양을 반영한다. 관측값 두 개로 평균을 내는 것으로는 원래 분포가 거의 매끄러워지지 않는다.
- **$n = 10$:** 히스토그램이 눈에 띄게 종 모양에 가까워지지만 (특히 감마분포에서) 왜도가 일부 남아 있을 수 있다.
- **$n = 100$:** 모집단 분포와 무관하게 세 표본분포가 모두 근사적으로 정규다. 중심극한정리가 작동하는 모습이다.

### 정량적 확인

중심극한정리는 $n$이 커질수록 다음을 예측한다.

$$
\text{std}(\bar{X}) \approx \frac{\sigma}{\sqrt{n}}
$$

$\sigma^2 = 3$인 Uniform(2, 8)의 경우:

| $n$ | 예측된 $\text{std}(\bar{X})$ | 모의실험 $\text{std}(\bar{X})$ |
|---|---|---|
| 2 | $\sqrt{3/2} \approx 1.225$ | $\approx 1.22$ |
| 10 | $\sqrt{3/10} \approx 0.548$ | $\approx 0.55$ |
| 100 | $\sqrt{3/100} \approx 0.173$ | $\approx 0.17$ |

모의실험의 표준편차가 $1/\sqrt{n}$ 예측과 잘 맞아떨어져, 표본분포가 예상된 속도로 좁아짐을 확인해 준다.

---

## 통계적 통찰

중심극한정리는 두 가지 조건을 요구한다.

1. 관측값이 **독립이고 동일한 분포를 따른다**.
2. 모집단의 **평균과 분산이 유한하다**.

둘 중 하나라도 어긋나면 중심극한정리는 적용되지 않는다.

- **종속인 자료**(예: 자기상관이 강한 시계열)는 더 느리게 수렴하거나 다른 극한으로 수렴할 수 있다.
- **무한한 분산**(예: 코시분포)이면 표본평균이 아예 안정되지 않는다.

!!! tip "수렴 속도"
    표본분포가 얼마나 빨리 정규가 되는지는 모집단 분포의 왜도에 달려 있다. 대칭 분포는 더 빨리 수렴하고, 심하게 치우친 분포는 더 큰 $n$이 필요할 수 있다. 베리–에센 정리가 이를 정량화한다. 근사 오차는 $O(1/\sqrt{n})$으로 유계다.

---

## 연습문제

**연습문제 1.**
$X_1, \ldots, X_n$이 i.i.d. Uniform(0, 1)이면 $\bar{X}$의 정확한 평균과 분산을 쓰라. $n = 12$일 때 $\bar{X}$의 표준편차는 얼마인가?

??? success "연습문제 1 풀이"
    Uniform(0, 1)에서 $\mu = 1/2$, $\sigma^2 = 1/12$이다.

    $$
    E[\bar{X}] = \mu = \frac{1}{2}, \qquad \text{Var}(\bar{X}) = \frac{\sigma^2}{n} = \frac{1}{12n}
    $$

    $n = 12$이면

    $$
    \text{Var}(\bar{X}) = \frac{1}{144}, \qquad \text{std}(\bar{X}) = \frac{1}{12} \approx 0.0833
    $$

    이다.

---

**연습문제 2.**
Gamma(2, 3) 분포($\mu = 6$, $\sigma^2 = 18$)에서 크기 $n$의 표본을 뽑는다고 하자. 중심극한정리 근사를 써서 $P(|\bar{X} - 6| < 0.5) \ge 0.95$가 되려면 $n$이 얼마나 커야 하는가?

??? success "연습문제 2 풀이"
    중심극한정리에 의해 $\bar{X} \approx N(\mu, \sigma^2/n)$이다. 다음이 필요하다.

    $$
    P\left(\left|\frac{\bar{X} - 6}{\sqrt{18/n}}\right| < \frac{0.5}{\sqrt{18/n}}\right) \ge 0.95
    $$

    이를 위해서는 $\frac{0.5}{\sqrt{18/n}} \ge 1.96$이어야 하므로

    $$
    \sqrt{18/n} \le \frac{0.5}{1.96} \approx 0.2551
    $$

    $$
    \frac{18}{n} \le 0.06506 \implies n \ge \frac{18}{0.06506} \approx 276.7
    $$

    이다. 따라서 $n \ge 277$이다.

---

**연습문제 3.**
중심극한정리가 코시분포에 적용되지 않는 이유를 설명하라. $n$이 커질 때 i.i.d. 코시 확률변수의 표본평균은 어떻게 되는가?

??? success "연습문제 3 풀이"
    코시분포의 밀도는 $f(x) = \frac{1}{\pi(1 + x^2)}$이다. 그 평균이 존재하지 않고(적분 $\int x f(x)\, dx$가 발산한다) 따라서 분산도 존재하지 않는다.

    중심극한정리는 유한한 평균과 분산을 요구하므로 적용되지 않는다. 실제로 i.i.d. 코시 $X_1, \ldots, X_n$에서 표본평균 $\bar{X}$는 관측값 하나와 같은 코시분포를 갖는다. 평균을 내도 퍼짐이 전혀 줄지 않는다. 코시분포가 지수 $\alpha = 1$인 안정분포이기 때문이다.

---

**연습문제 4.**
중심극한정리를 이용해 $\bar{X}$와 $s$(표본표준편차)에 근거한 모평균 $\mu$의 근사적 95% 신뢰구간을 유도하라.

??? success "연습문제 4 풀이"
    중심극한정리에 의해 $n$이 크면

    $$
    \frac{\bar{X} - \mu}{s / \sqrt{n}} \approx N(0, 1)
    $$

    이다. 95% 구간은 $|Z| \le 1.96$을 요구하므로

    $$
    P\left(-1.96 \le \frac{\bar{X} - \mu}{s/\sqrt{n}} \le 1.96\right) \approx 0.95
    $$

    이고, 정리하면

    $$
    \bar{X} - 1.96 \frac{s}{\sqrt{n}} \le \mu \le \bar{X} + 1.96 \frac{s}{\sqrt{n}}
    $$

    이다. 근사적 95% 신뢰구간은 $\bar{X} \pm 1.96 \, s / \sqrt{n}$이다.

---

**연습문제 5.**
$X_i$가 분산 $\sigma^2$인 i.i.d.일 때 $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$의 분산이 $\sigma^2 / n$임을 증명하라.

??? success "연습문제 5 풀이"
    독립인 확률변수에 대한 분산의 성질에 의해

    $$
    \text{Var}(\bar{X}_n) = \text{Var}\left(\frac{1}{n}\sum_{i=1}^n X_i\right) = \frac{1}{n^2} \sum_{i=1}^n \text{Var}(X_i) = \frac{1}{n^2} \cdot n\sigma^2 = \frac{\sigma^2}{n}
    $$

    이다. 두 번째 등호는 독립성(합의 분산이 분산의 합)과 각 $X_i$의 분산이 모두 $\sigma^2$이라는 사실을 쓴다. $\square$

---

**연습문제 6.**
이산분포에 대한 정규근사의 **연속성 수정**: 정숫값 $X$에 대해 $P(X \le k)$를 정규분포로 근사할 때 $\Phi((k + 0.5 - \mu)/\sigma)$를 쓴다. 왜 그런가?

??? success "연습문제 6 풀이"
    이산확률변수는 정수에 질량을 놓지만 연속 근사는 질량을 매끄럽게 퍼뜨린다. 수정하지 않으면 $P(X \le k) \approx \Phi((k - \mu)/\sigma)$가 사실상 $X = k$의 질량을 제외해 버린다.

    연속성 수정은 각 정수를 그 정수를 중심으로 하는 너비 1인 구간으로 취급한다. $P(X \le k)$에는 위쪽 끝점 $k + 0.5$를 쓴다.

    $$
    P(X \le k) \approx \Phi\!\left(\frac{k + 0.5 - \mu}{\sigma}\right)
    $$

    **개선:** 오차율이 $O(1/\sqrt n)$에서 $O(1/n)$으로 좋아진다.

    **예:** Binomial(20, 0.5)에서 $\mu = 10$, $\sigma \approx 2.236$이다. 정확한 값은 $P(X \le 12) = 0.8684$이다. 수정하지 않으면 $\Phi(0.894) = 0.814$(오차 0.054)이고, 수정하면 $\Phi(1.118) = 0.868$(오차 0.000)이다.

    특히 $n$이 크지 않을 때 이산에서 연속으로의 근사에는 언제나 연속성 수정을 적용하라.

---

**연습문제 7.**
**중심극한정리는 최댓값에 적용되지 않는다.** i.i.d. 표본의 최댓값 $M_n = \max_i X_i$이 가우시안 극한을 갖지 않음을 보여라. 그 극한분포는 무엇인가?

??? success "연습문제 7 풀이"
    $F_{M_n}(x) = F(x)^n$이다. $n \to \infty$이면 이것이 계단함수로 퇴화하며 가우시안이 아니다.

    적절히 척도를 다시 맞추면 퇴화하지 않는 극한을 얻는다. 수열 $a_n > 0$과 $b_n$에 대해

    $$
    P\!\left(\frac{M_n - b_n}{a_n} \le x\right) \to G(x)
    $$

    이다. **피셔–티펫–그네덴코 정리**에 의해 $G$는 세 가지 극단값 분포 중 하나여야 한다.

    - 꼬리가 얇은 $F$(정규, 지수)에 대해 **검벨**.
    - 꼬리가 두꺼운 $F$(파레토, $t$)에 대해 **프레셰**.
    - 받침이 유계인 $F$(균등)에 대해 **와이불**.

    **실용적 쓰임:** 극단값 이론은 최대 하천 수위(수문학), 최대 금융 손실(위험관리), 최소 수명 신뢰도 문제를 지배한다. 가우시안 중심극한정리는 합을 다루고 극단값 이론은 최댓값을 다룬다. 같은 자료의 서로 다른 통계량에 대한 별개의 점근 틀이다.
