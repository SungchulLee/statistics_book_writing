# 고급 기술 측도

## 개요

산술평균과 표본분산은 가장 흔한 요약통계량이지만 언제나 가장 적절한 것은 아니다. 이 절에서는 기본 도구를 확장하는 세 가지 주제를 다룬다.

1. **기하평균** — 투자 수익률처럼 곱셈적인 과정에 맞는 올바른 평균.
2. **체비쇼프 부등식** — 자료가 평균에서 얼마나 멀리 있을 수 있는지에 대한 분포무관 한계.
3. **모분산 대 표본분산** — $n - 1$로 나누는 것이 왜 불편추정량을 주는지 보이는 모의실험.

---

## 1. 기하평균 대 산술평균

### 문제

수익률의 열 $r_1, r_2, \ldots, r_T$에 대해 산술평균

$$
\bar{r} = \frac{1}{T} \sum_{t=1}^{T} r_t
$$

은 수익률이 변동하기만 하면 복리 성장률을 과대평가한다. 올바른 측도는 **기하평균**이다.

$$
r_g = \left(\prod_{t=1}^{T} (1 + r_t)\right)^{1/T} - 1
$$

### 코드

```python
import numpy as np
from scipy import stats

returns = np.array([0.36, 0.23, -0.48, -0.30, 0.15, 0.31])

arith_mean = np.mean(returns)
geo_mean = stats.mstats.gmean(1 + returns) - 1

print(f"Arithmetic mean: {arith_mean:.4f}  ({arith_mean*100:.2f}%)")
print(f"Geometric  mean: {geo_mean:.4f}  ({geo_mean*100:.2f}%)")
print(f"Compound value of \$1: \${np.prod(1 + returns):.4f}")
print(f"Using geo mean:       \${(1 + geo_mean)**len(returns):.4f}")
```

### 출력

```
Arithmetic mean: 0.0450  (4.50%)
Geometric  mean: -0.0139  (-1.39%)
Compound value of $1: $0.9196
Using geo mean:       $0.9196
```

### 해석

산술평균은 평균 수익률이 4.50%로 양수라고 말하지만, 실제로 투자한 \$1은 \$0.92로 줄어든다. 기하평균은 복리 수익률이 $-1.39\%$로 음수임을 올바르게 보고한다. 이 불일치는 산술평균이 복리의 비대칭성을 무시하기 때문에 생긴다. 50% 손실을 회복하려면 50%가 아니라 100%의 이익이 필요하다.

!!! warning "흔한 실수"
    복리 성장을 요약할 때 산술평균을 쓰지 마라. 곱셈적 과정에 대한 유일하게 올바른 요약은 기하평균이다.

---

## 2. 체비쇼프 부등식

### 진술

평균 $\mu$와 표준편차 $\sigma$가 유한한 **어떤** 분포에 대해서도, 평균에서 표준편차 $k$배 안에 있는 자료의 비율은 다음을 만족한다.

$$
P(|X - \mu| < k\sigma) \ge 1 - \frac{1}{k^2} \qquad \text{for } k > 1
$$

이 한계는 분포의 모양에 대해 아무 가정도 하지 않는다.

### 주요 값

| $k$ | $k\sigma$ 안에 있는 최소 비율 |
|---|---|
| 2 | $\ge 75\%$ |
| 3 | $\ge 88.9\%$ |
| 4 | $\ge 93.75\%$ |
| 5 | $\ge 96\%$ |

### 예: 키

키의 평균이 $\mu = 174$ cm이고 표준편차가 $\sigma = 4$ cm라고 하자. 모집단의 몇 퍼센트가 166 cm에서 182 cm 사이에 있는가?

$$
k = \frac{182 - 174}{4} = 2
$$

체비쇼프 부등식에 의해, 키 분포의 모양과 무관하게 모집단의 적어도 $1 - 1/4 = 75\%$가 이 범위에 들어간다.

### 코드

```python
import numpy as np
import matplotlib.pyplot as plt

def chebyshev(k):
    return 1 - 1 / k**2

z_vals = np.arange(1.1, 10, 0.1)
cheb_vals = [chebyshev(z) for z in z_vals]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(z_vals, cheb_vals, lw=2, color="seagreen")
ax.set_xlabel("k (standard deviations)")
ax.set_ylabel("Minimum fraction")
ax.set_title("Chebyshev's Inequality: 1 - 1/k²")
ax.axhline(0.75, color="grey", linestyle=":", alpha=0.5)
ax.annotate("k=2: ≥ 75%", (2, 0.75), fontsize=9,
            xytext=(4, 0.6), arrowprops=dict(arrowstyle="->"))
plt.tight_layout()
plt.show()
```

### 해석

체비쇼프의 한계는 **보수적**이다. 정규분포에서는 자료의 95%가 표준편차 2배 안에 있어 체비쇼프가 보장하는 75%를 훨씬 웃돈다. 이 한계의 힘은 보편성에 있다. 심하게 치우쳤거나 다봉인 분포를 포함해, 분산이 유한한 어떤 분포에도 적용된다.

---

## 3. 모분산 대 표본분산

### 편향 문제

소박한 분산 추정량은 $n$으로 나눈다.

$$
\hat{\sigma}^2_{\text{biased}} = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2
$$

$\bar{x}$가 $\mu$보다 표본점들에 더 가깝기 때문에, 이 추정량은 참된 모분산 $\sigma^2$을 체계적으로 과소추정한다. 베셀 보정은 $n - 1$을 쓴다.

$$
s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2
$$

### 모의실험

모집단에서 크기 100인 표본을 10,000번 뽑아 각 추정량의 평균을 참 분산과 비교하여 불편성을 확인한다.

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
population = np.random.normal(170, 10, 1000)
pop_var = np.var(population)

n_samples = 10000
sample_size = 100

biased_vars = np.empty(n_samples)
unbiased_vars = np.empty(n_samples)

for i in range(n_samples):
    sample = np.random.choice(population, size=sample_size, replace=False)
    biased_vars[i] = np.var(sample, ddof=0)
    unbiased_vars[i] = np.var(sample, ddof=1)

print(f"Population variance:          {pop_var:.4f}")
print(f"Mean of biased (ddof=0):      {biased_vars.mean():.4f}")
print(f"Mean of unbiased (ddof=1):    {unbiased_vars.mean():.4f}")
```

### 시각화

```python
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(biased_vars, bins=40, alpha=0.5, label="Biased (ddof=0)", density=True)
ax.hist(unbiased_vars, bins=40, alpha=0.5, label="Unbiased (ddof=1)", density=True)
ax.axvline(pop_var, color="red", linestyle="--", lw=2,
           label=f"True σ² = {pop_var:.1f}")
ax.set_xlabel("Variance estimate")
ax.set_title("Population vs Sample Variance")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
```

### 해석

편향된 추정량(ddof=0)의 히스토그램이 참 분산보다 살짝 왼쪽으로 이동해 있어 체계적인 과소추정을 확인해 준다. 불편추정량(ddof=1)은 참값을 중심으로 한다. $n = 100$에서는 차이가 작지만($100/99 \approx 1.01$배) 표본이 작아지면 상당해진다.

---

## 연습문제

**연습문제 1.**
어떤 투자의 연간 수익률이 $+20\%$, $-20\%$, $+20\%$, $-20\%$다. 산술평균과 기하평균을 계산하라. 투자한 \$1000의 최종 가치는 얼마인가?

??? success "풀이"
    산술평균: $(0.20 - 0.20 + 0.20 - 0.20)/4 = 0$. 산술평균은 성장이 없다고 말한다.

    기하평균:

    $$
    r_g = (1.20 \times 0.80 \times 1.20 \times 0.80)^{1/4} - 1 = (0.9216)^{1/4} - 1 \approx -0.0202
    $$

    최종 가치: $1000 \times 1.20 \times 0.80 \times 1.20 \times 0.80 = 1000 \times 0.9216 = \$921.60$.

    기하평균은 연 약 2%의 손실을 올바르게 나타내는 반면, 산술평균은 잘못되게 본전이라고 말한다.

---

**연습문제 2.**
체비쇼프 부등식을 증명하라. 마르코프 부등식에서 출발하라: 음이 아닌 확률변수 $Y$와 $a > 0$에 대해 $P(Y \ge a) \le E[Y]/a$.

??? success "풀이"
    $a = (k\sigma)^2 = k^2 \sigma^2$로 두고 $Y = (X - \mu)^2$에 마르코프 부등식을 적용한다.

    $$
    P\bigl((X - \mu)^2 \ge k^2 \sigma^2\bigr) \le \frac{E[(X - \mu)^2]}{k^2 \sigma^2} = \frac{\sigma^2}{k^2 \sigma^2} = \frac{1}{k^2}
    $$

    $(X - \mu)^2 \ge k^2 \sigma^2$은 $|X - \mu| \ge k\sigma$와 동치이므로

    $$
    P(|X - \mu| \ge k\sigma) \le \frac{1}{k^2}
    $$

    이고, 여집합을 취하면

    $$
    P(|X - \mu| < k\sigma) \ge 1 - \frac{1}{k^2}
    $$

    이다. $\square$

---

**연습문제 3.**
위의 모분산 모의실험에서 표본 크기 $n$이 커지면 ddof=0 추정량의 편향은 어떻게 되는가? 편향을 $n$과 $\sigma^2$의 함수로 나타내라.

??? success "풀이"
    편향된 추정량의 기댓값은

    $$
    E\left[\frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2\right] = \frac{n-1}{n} \sigma^2
    $$

    이므로 편향은

    $$
    \text{Bias} = \frac{n-1}{n}\sigma^2 - \sigma^2 = -\frac{\sigma^2}{n}
    $$

    이다. $n \to \infty$이면 편향은 0으로 간다. $n = 100$이면 편향이 $-\sigma^2/100$으로 참 분산의 1%에 불과하다. $n = 5$이면 편향이 $-\sigma^2/5 = -20\%$로 훨씬 크다.

---

**연습문제 4.**
어떤 자료의 평균이 50이고 표준편차가 5다. 체비쇼프 부등식을 이용해 구간 $[35, 65]$에 있는 자료의 최소 비율을 구하라. 그런 다음 자료가 정규분포를 따를 때의 비율과 비교하라.

??? success "풀이"
    구간 $[35, 65]$는 $[50 - 15, 50 + 15]$이므로 $k = 15/5 = 3$이다.

    체비쇼프에 의해 적어도 $1 - 1/9 \approx 88.9\%$다.

    정규분포에서는 $P(|Z| < 3) = P(-3 < Z < 3) \approx 0.9974$이므로 약 $99.7\%$다.

    정규분포는 체비쇼프의 보편적 하한이 요구하는 것보다 훨씬 많은 질량을 평균 근처에 몰아 놓는다. 그 차이($99.7\%$ 대 $88.9\%$)가 분포에 대해 아무 가정도 하지 않는 데 드는 비용을 보여준다.

---

**연습문제 5.**
음이 아닌 값들에 대해 기하평균이 언제나 산술평균 이하임을 보여라. 등호는 언제 성립하는가?

??? success "풀이"
    이것이 **산술–기하평균 부등식(AM-GM)** 이다. 음이 아닌 값 $a_1, \ldots, a_n$에 대해

    $$
    \frac{a_1 + a_2 + \cdots + a_n}{n} \ge (a_1 \cdot a_2 \cdots a_n)^{1/n}
    $$

    이다.

    **옌센 부등식을 이용한 증명:** 로그는 오목함수다. 옌센 부등식에 의해

    $$
    \log\left(\frac{1}{n}\sum_{i=1}^n a_i\right) \ge \frac{1}{n}\sum_{i=1}^n \log(a_i) = \log\left(\prod_{i=1}^n a_i\right)^{1/n}
    $$

    이다. $\log$가 순증가함수이므로 양변에 지수를 취하면 AM $\ge$ GM을 얻는다.

    등호는 $a_1 = a_2 = \cdots = a_n$일 때에 한해 성립한다. 엄격히 오목한 함수에 대해 옌센 부등식은 모든 값이 같지 않으면 엄격 부등식이기 때문이다. $\square$

---

**연습문제 6.**
양수 $a_1, \ldots, a_n$의 **조화평균**은 $H = n / \sum_i (1/a_i)$이다. 부등식 $H \le G \le A$(조화 $\le$ 기하 $\le$ 산술)를 증명하고, 조화평균이 *올바른* 평균인 경우를 설명하라.

??? success "풀이"
    **증명:** $1/a_1, \ldots, 1/a_n$에 AM-GM 부등식을 적용하면

    $$
    \frac{1}{n}\sum_i \frac{1}{a_i} \ge \left(\prod_i \frac{1}{a_i}\right)^{1/n} = \frac{1}{G}
    $$

    이다. 양변이 양수이므로 역수를 취하면 $H = n / \sum_i (1/a_i) \le G$이다. 연습문제 5의 $G \le A$와 결합하면 $H \le G \le A$를 얻는다.

    **조화평균이 올바른 경우:** 조화평균은 **비율**을 제대로 평균 낸다. 예를 들면 다음과 같다.

    - 서로 다른 속도로 같은 *거리*를 이동할 때의 평균 속도. 시속 60마일로 50마일, 시속 30마일로 50마일을 달리면 평균 속도는 $A(60, 30) = 45$가 아니라 $H(60, 30) = 40$마일이다. 느린 속도로 보내는 시간이 더 길기 때문에 산술평균은 과대평가한다.
    - 포트폴리오의 주가수익비율을 *이익*으로 가중할 때(조화평균 PER)와 *시가총액*으로 가중할 때(가중 산술평균 PER)의 차이.
    - $F_1$ 점수의 결합: $F_1 = H(\text{precision}, \text{recall}) = 2 PR / (P + R)$.

    일반 규칙: **역수**가 자연스럽게 더해지는 양(변화율, 단위당 빈도)을 평균 낼 때는 조화평균을 쓴다. 이를 산술평균과 혼동하면 체계적인 편향이 생긴다.

---

**연습문제 7.**
표본평균과 중앙값은 둘 다 대칭 분포의 "중심"을 추정한다. (a) 정규분포와 (b) 라플라스(이중지수) 분포 아래에서 **둘의 상대 효율을 비교하라**.

??? success "풀이"
    평균에 대한 중앙값의 상대 효율은 $\mathrm{ARE}(\tilde X, \bar X) = \mathrm{Var}(\bar X)/\mathrm{Var}(\tilde X)$이며, 값이 작을수록 중앙값의 효율이 낮다.

    **(a) 정규분포:** $\mathrm{ARE} = 2/\pi \approx 0.637$. 평균이 $\pi/2$배 더 효율적이다. 표본 크기 100에서 중앙값의 분산은 관측값 63개만으로 계산한 평균의 분산과 같다. 중앙값을 쓰는 데 드는 고전적인 가우시안 효율 비용이다.

    **(b) 라플라스 분포**(밀도가 $\propto e^{-|x|}$로 정규분포보다 꼬리가 두껍다): $\mathrm{ARE} = 2 > 1$. 중앙값이 평균보다 두 배 효율적이다. 라플라스 꼬리에서는 극단 관측값이 담고 있는 정보에 비해 영향력이 크므로 평균이 "낭비적"이다.

    **교훈:** 평균과 중앙값 사이의 선택은 강건성만의 문제가 아니라 추정량을 분포의 꼬리 행동에 맞추는 문제다. 정규성 아래에서는 평균이 최적(BLUE)이고, 꼬리가 더 두꺼운 분포에서는 중앙값(또는 다른 강건 추정량)이 훨씬 효율적일 수 있다. 이것이 (꼬리가 얇을 때 효율적인) 평균과 (꼬리가 두꺼울 때 효율적인) 중앙값 사이를 보간하는 M-추정량(Huber 1964)의 이론적 토대다.
