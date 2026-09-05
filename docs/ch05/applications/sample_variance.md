# S-squared의 표본분포

## 개요

**표본분산 $S^2$의 표본분포**는 모집단에서 표본을 반복해서 뽑을 때 확률표본으로부터 계산한 분산이 어떻게 거동하는지를 기술한다. 참 모분산 $\sigma^2$을 얼마나 정밀하게 추정할 수 있는지 이해하는 데 결정적인 개념이다.

## 수학적 정의

$X_1, X_2, \dots, X_n$을 평균 $\mu$, 분산 $\sigma^2$인 모집단에서 뽑은 i.i.d. 표본이라 하자. 표본분산은:

$$
S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2
$$

## 성질

### 기댓값 (불편성)

$$
E[S^2] = \sigma^2
$$

$n$이 아니라 $n-1$(자유도)로 나누는 이유가 바로 이 불편성이다. Bessel 수정은 표본으로부터 $\mu$를 추정하면서 잃어버린 자유도 하나를 보상한다.

### 카이제곱과의 연결 (정규모집단)

모집단이 **정규**이면 척도조정된 표본분산이 카이제곱 분포를 따른다:

$$
\frac{(n-1)S^2}{\sigma^2} = \sum_{i=1}^n \left(\frac{X_i - \bar{X}}{\sigma}\right)^2 \sim \chi^2_{n-1}
$$

동등하게:

$$
S^2 \sim \frac{\sigma^2}{n-1} \cdot \chi^2_{n-1}
$$

### S-squared의 분산

정규성 아래에서:

$$
\text{Var}(S^2) = \frac{2\sigma^4}{n-1}
$$

**유도.** $\text{Var}(\chi^2_{n-1}) = 2(n-1)$이므로:

$$
\text{Var}\!\left(\frac{(n-1)S^2}{\sigma^2}\right) = 2(n-1) \;\;\Longrightarrow\;\;
\text{Var}(S^2) = \frac{2\sigma^4}{n-1}
$$

### S-squared의 표준오차

$$
\text{SE}(S^2) = \sigma^2 \sqrt{\frac{2}{n-1}} \sim O\!\left(\frac{1}{\sqrt{n}}\right)
$$

## 수렴 속도

$\bar{X}$와 $S^2$은 점근적으로 같은 비율로 수렴한다:

| 추정량 | 표준오차 | 비율 |
|-----------|---------------|------|
| $\bar{X}$ | $\sigma / \sqrt{n}$ | $O(1/\sqrt{n})$ |
| $S^2$ | $\sigma^2 \sqrt{2/(n-1)}$ | $O(1/\sqrt{n})$ |

수렴 속도 면에서 둘 사이에 빠르고 느림의 차이는 **없다**.

## S-squared의 중대한 한계

$\bar{X}$와 $S^2$의 결정적 차이는 속도가 아니라 **분포에 대한 로버스트성**에 있다.

✅ **표본평균 $\bar{X}$**는 중심극한정리의 혜택을 본다. $n$이 충분히 크기만 하면 모집단 모양과 무관하게 $\bar{X}$의 근사적 정규성이 보장된다.

❌ **표본분산 $S^2$**은 다음에 의존한다:

$$
\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}
$$

이는 **정규성 아래에서만 성립한다**. 치우쳤거나 꼬리가 두껍거나 그 밖에 정규가 아닌 모집단에서는 이 카이제곱 결과가 더 이상 적용되지 않으며, 표본이 커도 $S^2$이 예측하기 어렵게 거동할 수 있다.

## 예제

### 예제 1: S-squared의 기댓값과 분산

**문제.** $N(\mu, 25)$에서 $n = 10$인 표본을 뽑는다. $E[S^2]$과 $\text{Var}(S^2)$을 구하라.

**풀이.** $Y \sim \chi^2_{n-1}$에 대해 $EY = n-1$이고 $\text{Var}(Y) = 2(n-1)$이다.

$$
E\!\left[\frac{(n-1)S^2}{\sigma^2}\right] = n - 1
\;\;\Longrightarrow\;\;
E[S^2] = \sigma^2 = 25
$$

$$
\text{Var}\!\left(\frac{(n-1)S^2}{\sigma^2}\right) = 2(n-1)
\;\;\Longrightarrow\;\;
\text{Var}(S^2) = \frac{2\sigma^4}{n-1} = \frac{2(25^2)}{9} = \frac{1250}{9} \approx 138.89
$$

### 예제 2: S-squared에 관한 확률 (정규모집단)

**문제.** $N(\mu, 25)$에서 $n = 10$인 표본을 뽑는다. $P(S^2 > 30)$을 구하라.

**풀이.**

$$
\frac{(n-1)S^2}{\sigma^2} = \frac{9 \times 30}{25} = 10.8
$$

$$
P(S^2 > 30) = P(\chi^2_9 > 10.8) \approx 0.2897
$$

```python
from scipy import stats

chi2_stat = 9 * 30 / 25
p_value = stats.chi2(df=9).sf(chi2_stat)
print(f"P(S^2 > 30) = {p_value:.4f}")
```

### 예제 3: 정규성 가정 없이

**문제.** 분산이 25인 모집단에서 $n = 10$인 표본을 뽑는다(정규성은 가정하지 않는다). $P(S^2 > 30)$에 관해 무엇을 말할 수 있는가?

**풀이.** 정규성이 없으면 $\frac{(n-1)S^2}{\sigma^2}$은 카이제곱 분포를 따르지 **않는다**. $E[S^2] = 25$임은 알지만, 모집단 모양에 관한 추가 정보 없이는 $P(S^2 > 30)$을 구할 수 없다.

$\text{Var}(S^2)$을 안다면 Chebyshev 부등식으로 한계를 줄 수 있겠지만, 그 값은 정규가 아닌 모집단의 고차 적률에 의존하며 우리는 그것을 알지 못한다.

## sigma-squared에 대한 신뢰구간

(정규성 아래에서) 카이제곱 추축량을 사용하면:

$$
P\!\left(\chi^2_{\alpha/2, \, n-1} \leq \frac{(n-1)S^2}{\sigma^2} \leq \chi^2_{1-\alpha/2, \, n-1}\right) = 1 - \alpha
$$

정리하면:

$$
\left[\frac{(n-1)S^2}{\chi^2_{1-\alpha/2, \, n-1}}, \;\; \frac{(n-1)S^2}{\chi^2_{\alpha/2, \, n-1}}\right]
$$

!!! note
    카이제곱 분포가 비대칭이므로 이 신뢰구간은 $S^2$을 중심으로 대칭이 **아니다**.

## 모의실험: S-squared의 표본분포

### 정규모집단

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(1)

population = stats.norm().rvs(100_000)
sample_size = 10
n_samples = 10_000

sample_vars = [
    np.var(np.random.choice(population, size=sample_size, replace=False), ddof=1)
    for _ in range(n_samples)
]

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6))

ax0.hist(population, bins=100, density=True, alpha=0.5)
ax0.set_title('Population Distribution (Normal)', fontsize=16)

ax1.hist(sample_vars, bins=100, density=True, alpha=0.5)
ax1.set_title(rf'Sampling Distribution of $S^2$ (n = {sample_size})', fontsize=16)

for ax in (ax0, ax1):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
```

### 소득 (치우친) 모집단

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(1)

url = 'https://raw.githubusercontent.com/gedeck/practical-statistics-for-data-scientists/master/data/loans_income.csv'
df = pd.read_csv(url)
population = df['x'].values
sample_size = 10
n_samples = 10_000

sample_vars = [
    np.var(np.random.choice(population, size=sample_size, replace=False), ddof=1)
    for _ in range(n_samples)
]

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6))

ax0.hist(population, bins=100, density=True, alpha=0.5)
ax0.set_title('Population Distribution (Income — Skewed)', fontsize=16)

ax1.hist(sample_vars, bins=100, density=True, alpha=0.5)
ax1.set_title(rf'Sampling Distribution of $S^2$ (n = {sample_size})', fontsize=16)

for ax in (ax0, ax1):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
```

## 요약

| 성질 | 결과 |
|----------|--------|
| $E[S^2]$ | $\sigma^2$ (언제나 불편) |
| $\text{Var}(S^2)$ | $2\sigma^4/(n-1)$ (정규성 아래에서) |
| 분포 | $\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}$ (정규성 아래에서만) |
| 로버스트성 | ❌ 중심극한정리 같은 보장이 없어 정규성 위반에 민감하다 |
| $\sigma^2$의 신뢰구간 | 카이제곱 분위수에 기반하며 비대칭이다 |

## 연습문제

**연습문제 1.**
**$S^2$의 평균과 분산.** $N(\mu, 25)$에서 $n = 10$인 표본을 뽑는다. (a) $\mathbb{E}[S^2]$, (b) $\mathrm{Var}(S^2)$을 계산하라.

??? success "연습문제 1 풀이"
    (a) $\mathbb{E}[S^2] = \sigma^2 = 25$ (Bessel 수정 덕분에 $S^2$이 불편이다).

    (b) 정규성 아래에서 $(n-1) S^2/\sigma^2 \sim \chi^2_{n-1}$이다. $\chi^2_{n-1}$의 분산은 $2(n-1)$이므로:

    $$
    \mathrm{Var}(S^2) = \frac{\sigma^4}{(n-1)^2} \cdot 2(n-1) = \frac{2\sigma^4}{n-1} = \frac{2 \cdot 625}{9} \approx 138.9
    $$

    $S^2$의 표준편차는 $\approx 11.8$로 변동성이 상당하다. $n = 10$에서 분산 추정값은 매우 불안정하다.

---

**연습문제 2.**
**$P(S^2 > 30)$.** 같은 설정으로 $n = 10$, $\sigma^2 = 25$, 정규모집단이다.

??? success "연습문제 2 풀이"
    $\chi^2 = (n-1)s^2/\sigma^2 = 9 \cdot 30/25 = 10.8$.

    $P(S^2 > 30) = P(\chi^2_9 > 10.8) \approx 0.290$으로 약 29%이다.

    참 분산이 25인데도 표본추출 변동성 때문에 표본분산이 30을 쉽게 넘을 수 있다. 소표본에서는 흔한 일이다.

---

**연습문제 3.**
**정규성 없이.** 모집단의 정규성을 가정하지 않으면 $P(S^2 > 30)$에 관해 무엇을 말할 수 있는가?

??? success "연습문제 3 풀이"
    정규성이 없으면 $(n-1)S^2/\sigma^2$은 $\chi^2_{n-1}$을 따르지 *않는다*. 카이제곱 결과는 정규분포에 특유한 것이다.

    $\mathbb{E}[S^2] = \sigma^2$은 여전히 성립하지만(불편성에는 정규성이 필요 없다), $S^2$의 분포는 상당히 다를 수 있다.

    $\mathrm{Var}(S^2)$을 안다면 **Chebyshev 한계**를 쓸 수 있지만, 일반적으로 $\mathrm{Var}(S^2)$은 모집단의 4차 적률(첨도)에 의존하며 이는 분포 모양에 민감하다.

    **실무:** 치우쳤거나 꼬리가 두꺼운 자료에서는 $S^2$의 분산이 카이제곱 공식이 시사하는 것보다 *크다*. 정규가 아닌 상황에서 $\sigma^2$에 관한 추론에는 붓스트랩이 권장되는 도구이다.

---

**연습문제 4.**
정규성 아래에서 **$S^2$의 카이제곱 분포를 증명하라.** 구체적으로 $X_1, \ldots, X_n$이 i.i.d. $N(\mu, \sigma^2)$이면 $(n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$임을 보여라.

??? success "연습문제 4 풀이"
    다음과 같이 분해한다:

    $$
    \frac{1}{\sigma^2}\sum_{i=1}^n (X_i - \mu)^2 = \frac{1}{\sigma^2}\sum_{i=1}^n (X_i - \bar X)^2 + \frac{n(\bar X - \mu)^2}{\sigma^2}
    $$

    좌변은 $\chi^2_n$이다(표준정규확률변수 $n$개의 제곱합). 우변의 두 번째 항은 $\chi^2_1$이다($\sqrt n(\bar X - \mu)/\sigma \sim N(0, 1)$의 제곱).

    ($\mathrm{span}\{\mathbf 1\}$과 그 직교여공간으로의 $X$의 직교사영에 적용한) **Cochran 정리**에 의해 우변의 두 항은 독립이다. 따라서:

    $$
    \chi^2_n = \frac{(n-1) S^2}{\sigma^2} + \chi^2_1
    $$

    이며 우변의 두 카이제곱은 독립이다. 적률생성함수의 성질에 의해 $(n-1) S^2/\sigma^2 \sim \chi^2_{n-1}$이다.

    $\square$

    이 유도가 $\sigma^2$에 관한 정규이론 추론 전체를 떠받치며, $t$ 분포와 $F$ 분포가 자연스럽게 나타나는 이유이기도 하다.

---

**연습문제 5.**
**$\sigma^2$의 신뢰구간.** 정규모집단에서 $n = 10$, $s^2 = 16$을 얻었다. $\sigma^2$에 대한 95% 신뢰구간을 구성하라.

??? success "연습문제 5 풀이"
    추축량: $(n-1)s^2/\sigma^2 \sim \chi^2_9$.

    95% 신뢰구간에는 $\chi^2_{0.025, 9} = 2.700$과 $\chi^2_{0.975, 9} = 19.023$을 사용한다:

    $$
    P(2.700 \le 9 s^2/\sigma^2 \le 19.023) = 0.95
    $$

    역으로 풀면:

    $$
    \frac{9 s^2}{19.023} \le \sigma^2 \le \frac{9 s^2}{2.700}
    $$

    $s^2 = 16$이면 신뢰구간은 $(9 \cdot 16/19.023, 9 \cdot 16/2.700) = (7.57, 53.33)$이다.

    넓고 비대칭이다. 자유도가 작으면 카이제곱이 치우쳐 있기 때문이다. $n = 10$에서는 분산이 거의 제약되지 않는다. $n$이 커지면 신뢰구간이 좁아진다.

---

**연습문제 6.**
**$S^2$과 $\sigma^2_{\text{MLE}}$.** $\sigma^2$의 MLE는 분모가 $n - 1$이 아니라 $n$이다. 편향, 분산, 평균제곱오차를 비교하라.

??? success "연습문제 6 풀이"
    $S^2 = \frac{1}{n-1}\sum(X_i - \bar X)^2$ (불편): $\mathbb{E}[S^2] = \sigma^2$, $\mathrm{Var}(S^2) = 2\sigma^4/(n-1)$.

    $\hat\sigma^2_{\text{MLE}} = \frac{1}{n}\sum(X_i - \bar X)^2 = \frac{n-1}{n} S^2$ (편향): $\mathbb{E}[\hat\sigma^2_{\text{MLE}}] = (n-1)\sigma^2/n$.

    $\mathrm{Var}(\hat\sigma^2_{\text{MLE}}) = ((n-1)/n)^2 \cdot 2\sigma^4/(n-1) = 2(n-1)\sigma^4/n^2$.

    $\mathrm{MSE}(\hat\sigma^2_{\text{MLE}}) = \mathrm{Var} + \mathrm{bias}^2 = 2(n-1)\sigma^4/n^2 + \sigma^4/n^2 = (2n-1)\sigma^4/n^2$.

    $\mathrm{MSE}(S^2) = \mathrm{Var}(S^2) = 2\sigma^4/(n-1)$.

    비를 비교하면 $\mathrm{MSE}(\hat\sigma^2_{\text{MLE}})/\mathrm{MSE}(S^2) = (2n-1)(n-1)/(2n^2)$로, 모든 $n \ge 2$에서 1보다 작다.

    **MLE는 편향되어 있지만 평균제곱오차가 더 작다.** 분산이 더 작은 것이 편향을 상쇄하고도 남기 때문이다. **편향–분산 맞바꿈**의 고전적인 예로, 전체 오차를 줄이기 위해 약간의 편향을 받아들이는 것이다.

    그럼에도 대부분의 소프트웨어가 (Bessel 수정된) $S^2$을 쓰는 이유는 불편성이 깔끔한 성질이고 $n \to \infty$이면 평균제곱오차의 차이가 사라지기 때문이다.
