# 비율의 표본분포

## 개요

**표본비율 $\hat{p}$의 표본분포**는 이항 모집단에서 확률표본을 반복해서 뽑을 때 성공 비율이 어떻게 달라지는지를 기술한다. 모비율에 관한 추론의 토대이며, 여론조사, 품질관리, 임상시험, A/B 테스트가 모두 이에 의존한다.

## 수학적 정의

$X_1, \dots, X_n$을 i.i.d. $\text{Bernoulli}(p)$라 하자. 여기서 $X_i = 1$(성공) 또는 $X_i = 0$(실패)이다. 표본비율은:

$$
\hat{p} = \frac{1}{n}\sum_{i=1}^n X_i = \frac{\text{number of successes}}{n}
$$

## 성질

### 기댓값 (불편성)

$$
E[\hat{p}] = p
$$

표본비율은 모비율의 **불편추정량**이다.

### 분산과 표준오차

$\text{Var}(X_i) = p(1-p)$이므로:

$$
\text{Var}(\hat{p}) = \frac{p(1-p)}{n}, \qquad
\text{SE}(\hat{p}) = \sqrt{\frac{p(1-p)}{n}}
$$

!!! note
    $\text{SE}(\bar{X}) = \sigma/\sqrt{n}$과 달리 $\hat{p}$의 표준오차는 모수 $p$ 자체에 의존한다. 실무에서는 $p$를 모르므로 $\hat{p}$로 대체한다:

    $$
    \widehat{\text{SE}}(\hat{p}) = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
    $$

### 모양 (정규근사)

중심극한정리에 의해 $n$이 충분히 크면:

$$
\frac{\hat{p} - p}{\sqrt{p(1-p)/n}} \xrightarrow{d} N(0, 1)
$$

정규근사가 타당하기 위한 **경험 법칙**:

$$
np \geq 5 \quad \text{and} \quad n(1-p) \geq 5
$$

성공 횟수와 실패 횟수가 모두 종 모양 근사를 쓰기에 충분히 크도록 보장한다.

## 예: 표준오차 계산

**문제.** 참 비율이 $p = 0.4$이고 표본크기가 $n = 100$이다.

$$
\text{SE}(\hat{p}) = \sqrt{\frac{0.4 \times 0.6}{100}} = \sqrt{0.0024} \approx 0.049
$$

크기 100인 표본을 반복해서 뽑으면 $\hat{p}$는 참값 $p = 0.4$ 주위로 대체로 0.049 정도 달라진다.

## 예제

### 예제 1: 브랜드 선호

**문제.** 어떤 모집단에서 60%가 브랜드 A를 선호한다. $n = 100$일 때 $P(\hat{p} > 0.65)$를 구하라.

**풀이.**

$$
\text{SE} = \sqrt{\frac{0.60 \times 0.40}{100}} \approx 0.049
$$

$$
Z = \frac{0.65 - 0.60}{0.049} \approx 1.02
$$

$$
P(\hat{p} > 0.65) = P(Z > 1.02) \approx 0.154
$$

```python
from scipy import stats
print(f"P(p_hat > 0.65) = {stats.norm.sf(1.02):.4f}")
```

### 예제 2: 소표본 — 정확값과 근사값

**문제.** 어떤 도시에서 30%가 대중교통을 선호한다. $n = 10$일 때 $P(\hat{p} > 0.35)$를 구하라.

**정확한 Binomial 계산.** $\hat{p} > 0.35$는 $X \geq 4$를 뜻하며 여기서 $X \sim \text{Binomial}(10, 0.3)$이다:

$$
P(X \geq 4) = 1 - P(X \leq 3)
$$

$$
P(X = 0) = 0.0282, \quad P(X = 1) = 0.1211, \quad P(X = 2) = 0.2335, \quad P(X = 3) = 0.2668
$$

$$
P(X \geq 4) = 1 - 0.6496 = 0.3504
$$

**정규근사.** 조건을 확인하면 $np = 3 < 5$이므로 정규근사가 미덥지 않다.

$$
\text{SE} = \sqrt{\frac{0.3 \times 0.7}{10}} \approx 0.1449, \qquad
Z = \frac{0.35 - 0.30}{0.1449} \approx 0.345
$$

$$
P(\hat{p} > 0.35) \approx P(Z > 0.345) \approx 0.365
$$

**비교:**

| 방법 | 결과 |
|--------|--------|
| 정확한 binomial | 0.3504 |
| 정규근사 | 0.3650 |

표본이 작은데도 근사가 꽤 가깝지만, $np < 5$일 때는 정확한 binomial 계산이 낫다.

```python
from scipy import stats

# Exact
exact = 1 - stats.binom(n=10, p=0.3).cdf(3)
print(f"Exact: {exact:.4f}")

# Normal approximation
approx = stats.norm.sf(0.345)
print(f"Normal approx: {approx:.4f}")
```

## 두 비율의 차

비율이 $p_1$과 $p_2$인 두 모집단에서 독립인 표본을 뽑으면:

$$
Z = \frac{(\hat{p}_1 - \hat{p}_2) - (p_1 - p_2)}{\sqrt{\frac{p_1(1-p_1)}{n_1} + \frac{p_2(1-p_2)}{n_2}}} \approx N(0, 1)
$$

**신뢰구간:**

$$
(\hat{p}_1 - \hat{p}_2) \pm z_{\alpha/2} \sqrt{\frac{\hat{p}_1(1-\hat{p}_1)}{n_1} + \frac{\hat{p}_2(1-\hat{p}_2)}{n_2}}
$$

## 모의실험: p-hat의 표본분포

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(1)

population = stats.binom(n=1, p=0.4).rvs(100_000)
sample_size = 1_000
n_samples = 10_000

sample_proportions = [
    np.mean(np.random.choice(population, size=sample_size, replace=False))
    for _ in range(n_samples)
]

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6))

ax0.hist(population, bins=3, density=True, alpha=0.5)
ax0.set_title('Population Distribution (Bernoulli, p = 0.4)', fontsize=16)

ax1.hist(sample_proportions, bins=50, density=True, alpha=0.5)
ax1.set_title(rf'Sampling Distribution of $\hat{{p}}$ (n = {sample_size})', fontsize=16)

for ax in (ax0, ax1):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
```

## 대학원 수준의 보충

- 표본이 작거나 비율이 극단적일 때($p$가 0이나 1에 가까울 때)는 **Binomial 분포**를 직접 사용해야 한다.
- **Wilson 구간**이 Wald 구간($\hat{p} \pm z^* \cdot \widehat{\text{SE}}$)보다 대체로 선호된다. 특히 $n$이 작거나 $p$가 극단적일 때 포함확률 성질이 더 좋다.
- **Agresti–Coull 구간**은 Wald 구간을 계산하기 전에 가상의 성공 2회와 실패 2회를 더하는 방식으로, 간단하면서도 포함확률을 개선한다.

## 요약

| 성질 | 결과 |
|----------|--------|
| $E[\hat{p}]$ | $p$ (불편) |
| $\text{Var}(\hat{p})$ | $p(1-p)/n$ |
| $\text{SE}(\hat{p})$ | $\sqrt{p(1-p)/n}$ |
| 정규근사가 타당한 조건 | $np \geq 5$이고 $n(1-p) \geq 5$ |
| $\bar{X}$와의 핵심 차이 | 표준오차가 모수 자체에 의존한다 |
| $n$이 작을 때 | 정규근사 대신 정확한 binomial을 사용 |

## 연습문제

**연습문제 1.**
모집단의 $p = 0.60$이고 표본 $n = 100$이다. $P(\hat p > 0.65)$를 계산하라.

??? success "풀이"
    $\mathrm{SE}(\hat p) = \sqrt{p(1-p)/n} = \sqrt{0.24/100} \approx 0.049$. $Z = (0.65 - 0.60)/0.049 \approx 1.02$.

    $P(\hat p > 0.65) = 1 - \Phi(1.02) \approx 0.154$로 약 15.4%이다.

    조건: $np = 60 \ge 10$이고 $n(1-p) = 40 \ge 10$이므로 정규근사가 타당하다.

---

**연습문제 2.**
**소표본의 문제.** 모집단의 $p = 0.30$이고 표본 $n = 10$이다. $P(\hat p > 0.35)$를 정확한 방법과 정규근사로 각각 계산하라. 여기서 정규근사가 통하는지 아니면 실패하는지 이유를 설명하라.

??? success "풀이"
    정확한 계산: $\hat p > 0.35$ ⟺ $X \ge 4$이며 $X \sim \mathrm{Binomial}(10, 0.3)$이다.

    $P(X < 4) = P(X = 0,1,2,3) = 0.028 + 0.121 + 0.233 + 0.267 = 0.650$이므로 $P(X \ge 4) = 0.350$이다.

    정규근사: $\mathrm{SE} = \sqrt{0.21/10} \approx 0.145$. $Z = 0.05/0.145 \approx 0.345$. $P(Z > 0.345) \approx 0.365$.

    차이: 정확값 0.350 대 정규근사 0.365로 약 1.5퍼센트포인트 차이가 난다. ($p = 0.3$에서 binomial이 그리 심하게 치우쳐 있지 않아) 정규근사가 그런대로 통하지만, $np = 3$이 작아 통상적인 경험 법칙($np \ge 10$)을 위반한다. 정확도를 높이려면 연속성 수정을 적용하거나 정확한 binomial을 사용하라.

---

**연습문제 3.**
**$\hat p$의 불편성을 증명하고 표준오차를 구하라.** $\mathbb{E}[\hat p] = p$이고 $\mathrm{Var}(\hat p) = p(1-p)/n$임을 보여라.

??? success "풀이"
    $X_i \sim \mathrm{Bernoulli}(p)$가 i.i.d.일 때 $X = \sum X_i$에 대해 $\hat p = X/n$이다.

    $\mathbb{E}[\hat p] = \mathbb{E}[X]/n = np/n = p$로 불편이다.

    $\mathrm{Var}(\hat p) = \mathrm{Var}(X)/n^2 = np(1-p)/n^2 = p(1-p)/n$.

    $\mathrm{SE}(\hat p) = \sqrt{p(1-p)/n}$.

    $\square$

    표준오차는 $p = 1/2$에서 최대가 된다(최악의 경우). $p = 1/2$일 때 $\mathrm{SE} = 1/(2\sqrt n)$이다.

---

**연습문제 4.**
**표본크기 설계.** $p$를 모를 때 95% 신뢰수준에서 오차한계 $\pm 3$퍼센트포인트로 $p$를 추정하려면 표본크기가 얼마여야 하는가?

??? success "풀이"
    오차한계: $\mathrm{ME} = z_{0.975} \cdot \mathrm{SE} = 1.96 \sqrt{p(1-p)/n} \le 0.03$.

    최악의 경우인 $p = 1/2$에서 $\mathrm{SE} = 1/(2\sqrt n)$이므로 $1.96/(2\sqrt n) \le 0.03 \Rightarrow \sqrt n \ge 1.96/0.06 \approx 32.67 \Rightarrow n \ge 1068$이다.

    이것이 여론조사에서 "$n \approx 1000$" 규칙이 나온 배경이다. $n = 1000$이면 $p$가 무엇이든 95% 신뢰수준의 오차한계가 최대 $\pm 3.1$퍼센트포인트이다.

    $p$가 0.5에서 멀다고 짐작되면(가령 $p \approx 0.1$) $p(1-p)$가 0.25 대신 0.09가 되어 $n \approx 0.09 \cdot 1068/0.25 \approx 385$면 충분하다. $p$의 대략적인 값을 알면 필요한 $n$이 줄어든다.

---

**연습문제 5.**
**Wilson 점수 구간.** 이항 비율에서 표준적인 Wald 신뢰구간 $\hat p \pm z \sqrt{\hat p(1-\hat p)/n}$보다 **Wilson 점수** 신뢰구간이 선호되는 이유는 무엇인가?

??? success "풀이"
    **Wald 신뢰구간의 문제:**

    - 특히 $p = 0$이나 $p = 1$ 근처에서 포함확률이 비대칭이다.
    - 0 아래나 1 위로 뻗는 구간이 나올 수 있다(예: $\hat p = 0.05, n = 50$이면 CI = $0.05 \pm 0.06 = (-0.01, 0.11)$).
    - 포함확률이 $n$에 따라 크게 요동쳐 명목 수준 $1 - \alpha$에서 멀어진다.

    **Wilson 점수 구간:**

    $$
    p_{\mathrm{Wilson}} = \frac{\hat p + z^2/(2n) \pm z\sqrt{\hat p(1-\hat p)/n + z^2/(4n^2)}}{1 + z^2/n}
    $$

    표준오차에서 $p$를 $\hat p$로 대체하는 대신 검정을 역으로 풀어 $|p - \hat p| \le z\sqrt{p(1-p)/n}$을 $p$에 대해 해결한 것이다.

    **장점:** 항상 $[0, 1]$ 안에 머물고, 경계 근처에서 포함확률이 훨씬 좋으며, 현대적 관행에서 권장된다(R의 `prop.test`, Python의 `statsmodels.stats.proportion.proportion_confint(method="wilson")`).

---

**연습문제 6.**
**비율의 차.** 독립인 두 표본에서 $n_1$로부터 $\hat p_1$을, $n_2$로부터 $\hat p_2$를 얻었다. $\hat p_1 - \hat p_2$의 표준오차를 유도하라.

??? success "풀이"
    독립성에 의해 $\mathrm{Var}(\hat p_1 - \hat p_2) = \mathrm{Var}(\hat p_1) + \mathrm{Var}(\hat p_2) = p_1(1-p_1)/n_1 + p_2(1-p_2)/n_2$이다.

    $\mathrm{SE}(\hat p_1 - \hat p_2) = \sqrt{p_1(1-p_1)/n_1 + p_2(1-p_2)/n_2}$.

    추론에서는($H_0: p_1 = p_2 = p$를 검정할 때) 표준오차에 합동추정량 $\hat p_{\text{pool}} = (X_1 + X_2)/(n_1 + n_2)$를 대입한다.

    신뢰구간에서는(동일성을 가정하지 않고 $p_1 - p_2$를 추정할 때) $\hat p_1$과 $\hat p_2$를 각각 대입한다:

    $$
    \mathrm{CI}: (\hat p_1 - \hat p_2) \pm z\sqrt{\hat p_1(1-\hat p_1)/n_1 + \hat p_2(1-\hat p_2)/n_2}
    $$

    이것이 A/B 테스트와 임상시험에서 쓰는 두 비율 $z$ 검정과 신뢰구간의 바탕이다.
