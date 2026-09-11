# Pearson의 r 검정 (상관에 대한 t-검정)

표본상관 $r$을 계산하면 관측된 선형 연관의 강도를 알 수 있지만, 그 값이 통계적으로 유의한지도 판정해야 한다. 즉 모상관이 0이라는 귀무가설에 반하는 증거를 주는지 확인해야 한다. Pearson 상관에 대한 표준 검정은 t-통계량을 쓰며 응용통계에서 가장 자주 수행되는 가설검정 중 하나이다.

---

## 가설

가장 흔한 검정은

$$
H_0\!: \rho = 0 \quad \text{vs} \quad H_1\!: \rho \neq 0
$$

이며 $\rho$는 모집단 Pearson 상관계수이다. 연관의 방향을 미리 예측한 경우에는 단측 대립가설 $H_1\!: \rho > 0$이나 $H_1\!: \rho < 0$도 쓴다.

---

## 검정통계량

$H_0\!: \rho = 0$이고 $(X, Y)$가 이변량 정규분포를 따른다는 가정 아래에서, 통계량

$$
t = \frac{r\sqrt{n-2}}{\sqrt{1 - r^2}}
$$

은 자유도 $n - 2$인 **Student t-분포**를 따른다. 여기서 $r$은 표본 Pearson 상관, $n$은 표본크기이다.

### 유도 개요

표본상관 $r$은 $Y$를 $X$에 표준화하여 회귀했을 때의 기울기로 쓸 수 있다. $H_0$ 아래에서 참 기울기가 0이므로 통상적인 회귀 t-검정이 적용된다. 모수 두 개(절편과 기울기)를 추정하므로 자유도가 $n - 2$이다.

---

## 판정 규칙

유의수준 $\alpha$의 양측검정에서:

- $|t| > t_{\alpha/2, \, n-2}$이면 $H_0$을 기각한다
- 동등하게 p-값이 $\alpha$보다 작으면 기각한다

p-값은

$$
p = 2 \cdot P(T_{n-2} > |t|)
$$

이며 $T_{n-2}$는 자유도 $n-2$인 $t$-분포 확률변수이다.

단측검정에서는:

- $H_1\!: \rho > 0$: $t > t_{\alpha, \, n-2}$이면 기각
- $H_1\!: \rho < 0$: $t < -t_{\alpha, \, n-2}$이면 기각

---

## 예제

어떤 연구자가 학생 $n = 25$명의 자료를 모아 공부 시간과 시험 점수 사이에 표본상관 $r = 0.45$를 얻었다.

$$
t = \frac{0.45\sqrt{25 - 2}}{\sqrt{1 - 0.45^2}} = \frac{0.45 \times 4.796}{\sqrt{0.7975}} = \frac{2.158}{0.8931} = 2.417
$$

자유도 $n - 2 = 23$에서 $\alpha = 0.05$ 양측검정의 임계값은 $t_{0.025, 23} = 2.069$이다. $|t| = 2.417 > 2.069$이므로 $H_0$을 기각하고 공부 시간과 시험 점수 사이에 통계적으로 유의한 양의 선형 상관이 있다고 결론짓는다.

---

## 모상관에 대한 신뢰구간

$\rho$에 대한 신뢰구간을 만들 때에는 (특히 $\rho$가 0에서 멀 때) $r$의 표본분포가 치우쳐 있으므로 **Fisher의 z 변환**을 쓴다:

1. 변환: $z = \text{arctanh}(r) = \frac{1}{2}\ln\!\left(\frac{1+r}{1-r}\right)$

2. $z$의 근사 표준오차는 $\text{SE}_z = \frac{1}{\sqrt{n-3}}$

3. 변환된 모수에 대한 신뢰구간을 만든다:

    $$
    z \pm z_{\alpha/2} \cdot \frac{1}{\sqrt{n-3}}
    $$

4. 각 끝점을 $\tanh$로 역변환하여 $\rho$에 대한 신뢰구간을 얻는다.

### 예제 (계속)

$r = 0.45$, $n = 25$일 때:

$$
z = \text{arctanh}(0.45) = 0.4847
$$

$$
\text{SE}_z = \frac{1}{\sqrt{22}} = 0.2132
$$

$\zeta = \text{arctanh}(\rho)$에 대한 95% 신뢰구간:

$$
0.4847 \pm 1.96 \times 0.2132 = (0.0668, \; 0.9026)
$$

역변환하면 $(\tanh(0.0668), \; \tanh(0.9026)) = (0.067, \; 0.717)$이다.

$\rho$에 대한 95% 신뢰구간은 약 $(0.07, 0.72)$이다.

---

## 0이 아닌 값에 대한 검정

$\rho_0 \neq 0$인 $H_0\!: \rho = \rho_0$을 검정할 때에는 위 t-검정을 쓸 수 없다(그 귀무분포가 $\rho = 0$에 의존한다). 대신 Fisher z 변환을 쓴다:

$$
Z = \frac{\text{arctanh}(r) - \text{arctanh}(\rho_0)}{1/\sqrt{n-3}}
$$

$H_0$ 아래에서 $Z$는 근사적으로 표준정규분포를 따른다.

---

## 가정

상관에 대한 t-검정은 다음을 요구한다:

1. **이변량 정규성**: $X$와 $Y$가 모두 정규분포를 따른다(적어도 $Y \mid X$의 조건부 분포가 분산이 일정한 정규여야 한다).
2. **무작위 표집**: 관측값들이 독립이다.
3. **선형성**: $X$와 $Y$의 관계가 선형이다.

!!! warning "표본이 크다고 비선형성이 해결되지는 않는다"
    $n$이 크면 아주 작은 $r$도 통계적으로 유의해진다. 유의한 p-값은 $\rho \neq 0$임을 알려줄 뿐 그 상관의 실질적 중요성에 대해서는 아무 말도 하지 않는다. p-값과 함께 항상 $r$을 보고하고 해석하라.

정규성이 어긋나면 순열검정이나 붓스트랩 방법이 비모수적 대안이 된다.

---

## Python으로 계산하기

```python
import numpy as np
from scipy import stats

# Sample data
np.random.seed(42)
n = 25
x = np.random.normal(5, 2, n)
y = 0.5 * x + np.random.normal(0, 1, n)

# Test H0: rho = 0
r, p_value = stats.pearsonr(x, y)
print(f"r = {r:.4f}")
print(f"p-value = {p_value:.4f}")

# Manual t-statistic
t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
p_manual = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))
print(f"t-statistic = {t_stat:.4f}")
print(f"Manual p-value = {p_manual:.4f}")

# Fisher z confidence interval
z = np.arctanh(r)
se_z = 1 / np.sqrt(n - 3)
ci_z = (z - 1.96 * se_z, z + 1.96 * se_z)
ci_r = (np.tanh(ci_z[0]), np.tanh(ci_z[1]))
print(f"95% CI for rho: ({ci_r[0]:.3f}, {ci_r[1]:.3f})")
```

출력:

```
r = 0.8074
p-value = 0.0000
t-statistic = 6.5638
Manual p-value = 0.0000
95% CI for rho: (0.605, 0.912)
```

scipy의 p-값과 $t = r\sqrt{(n-2)/(1-r^2)}$로 손계산한 값이 일치한다.

신뢰구간 $(0.605, 0.912)$가 $r = 0.807$을 중심으로 **대칭이 아니라는** 점을 눈여겨보라. 아래쪽으로 0.202, 위쪽으로 0.105다. $r$이 $\pm 1$이라는 벽에 갇혀 있어 분포가 치우치기 때문이며, 그래서 구간을 Fisher $z$ 척도에서 만든 뒤 되돌린다.

---

## 요약

Pearson의 $r$에 대한 t-검정은 관측된 표본상관이 $H_0\!: \rho = 0$에 반하는 통계적으로 유의한 증거를 주는지 판정한다. 이변량 정규성 아래에서 검정통계량 $t = r\sqrt{n-2}/\sqrt{1-r^2}$은 자유도 $n-2$인 $t$-분포를 따른다. $\rho$에 대한 신뢰구간은 $r$의 치우친 표본분포를 다루기 위해 Fisher의 z 변환을 쓴다. 이 검정은 이변량 정규성, 독립성, 선형관계를 요구한다. 이 가정들이 어긋나면 [Spearman 검정](test_spearman.md)이나 [Kendall 검정](test_kendall.md) 같은 순위 기반 검정이 선호된다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
단조이지만 비선형인 관계의 자료를 생성하라:

$$
y = e^{0.1x} + \epsilon, \quad x \sim U(0, 30), \quad \epsilon \sim N(0, 1)
$$

1. Pearson의 $r$, Spearman의 $\rho_s$, Kendall의 $\tau$를 계산하라
2. Spearman과 Kendall이 Pearson보다 이 관계를 더 잘 탐지하는 이유를 설명하라

</div>

??? success "풀이"

    $y = e^{0.1x}$는 단조증가하지만 선형이 아니라 지수적이다. Pearson의 $r$은 선형 연관을 재므로 이 휘어진 관계의 강도를 과소평가한다. Spearman의 $\rho_s$와 Kendall의 $\tau$는 순위를 써서 단조 연관을 재므로 함수 형태와 무관하게 관계를 포착한다. 선형 적합은 좋지 않아도 단조 구조가 순위에는 완벽히 보존되므로 두 순위 기반 측도 모두 Pearson의 $r$보다 1에 가까울 것이다.

<div class="drillbox" markdown>

**연습문제 2.**
나이–소득 자료를 쓴다:

```python
import numpy as np
from scipy import stats

age    = [18, 25, 57, 45, 26, 64, 37, 40, 24, 33]
income = [15000, 29000, 68000, 52000, 32000, 80000, 41000, 45000, 26000, 33000]

for name, fn in [("Pearson", stats.pearsonr),
                 ("Spearman", stats.spearmanr),
                 ("Kendall", stats.kendalltau)]:
    coef, p = fn(age, income)
    print(f"{name:<9} coef = {coef:.4f}  p = {p:.3e}")

# 이상점 추가: 젊은데 소득이 아주 높은 사람
age2 = age + [20]
income2 = income + [200000]
print()
for name, fn in [("Pearson", stats.pearsonr),
                 ("Spearman", stats.spearmanr),
                 ("Kendall", stats.kendalltau)]:
    coef, p = fn(age2, income2)
    print(f"{name:<9} coef = {coef:.4f}  p = {p:.3e}   (이상점 추가 후)")
```

출력:

```
Pearson   coef = 0.9923  p = 1.535e-08
Spearman  coef = 1.0000  p = 6.647e-64
Kendall   coef = 1.0000  p = 5.511e-07

Pearson   coef = 0.0301  p = 9.301e-01   (이상점 추가 후)
Spearman  coef = 0.5909  p = 5.558e-02   (이상점 추가 후)
Kendall   coef = 0.6727  p = 3.106e-03   (이상점 추가 후)
```

1. 세 상관계수와 그 p-값을 모두 계산하라
2. $\alpha = 0.01$에서 $H_0: \rho = 0$을 기각할 수 있는가?
3. 이상점(나이=20, 소득=200000)을 추가하고 다시 계산하라. 어느 검정이 가장 큰 영향을 받는가?

</div>

??? success "풀이"

    1. 세 상관계수 모두 나이와 소득 사이에 강한 양의 연관을 보인다. 실제로 Pearson $r = 0.9923$, Spearman $\rho_s = 1.000$, Kendall $\tau = 1.000$이다. 순위 기반 두 계수가 정확히 1인 것은 나이 순서와 소득 순서가 완벽히 일치하기 때문이다.

    2. $n = 10$에서 Pearson 검정의 p-값이 $1.5 \times 10^{-8}$이므로 1% 수준에서 $H_0$을 기각할 수 있다. 순위 기반 검정들도 유의하다.

    3. 이상점(나이=20, 소득=200000)을 추가하면 Pearson의 $r$이 $0.9923$에서 $0.0301$로 무너진다. 극단값에 민감하기 때문이다. 젊은 사람의 매우 높은 소득이 전체 추세와 모순되어 상관을 거의 0으로 끌어내린다. Spearman은 $0.5909$, Kendall은 $0.6727$로 훨씬 덜 영향받는다. 순위를 쓰므로 이상점이 소득에서는 극단적인 순위를 받지만 나이에서는 그렇지 않아 영향이 제한된다.

<div class="drillbox" markdown>

**연습문제 3.**
참 $\rho = 0.3$일 때:

1. $n \in \{10, 30, 100, 500, 1000\}$인 이변량 정규 표본을 생성하라
2. 각 $n$에서 Pearson의 $r$과 그 p-값을 계산하라
3. p-값 대 표본크기를 그리고 표본크기와 통계적 유의성의 관계를 논하라

```python
import numpy as np
from scipy import stats

np.random.seed(0)
rho = 0.3
sample_sizes = [10, 30, 100, 500, 1000]

cov = [[1, rho], [rho, 1]]
for n in sample_sizes:
    xy = np.random.multivariate_normal([0, 0], cov, size=n)
    r, p = stats.pearsonr(xy[:, 0], xy[:, 1])
    print(f"n = {n:>4}:  r = {r:+.4f}   p = {p:.4g}")
```

출력:

```
n =   10:  r = -0.0515   p = 0.8875
n =   30:  r = +0.4186   p = 0.02134
n =  100:  r = +0.2805   p = 0.00471
n =  500:  r = +0.2979   p = 1.049e-11
n = 1000:  r = +0.3143   p = 2.295e-24
```

</div>

??? success "풀이"

    참 $\rho = 0.3$을 고정하면 $n$이 커질수록 p-값이 대체로 작아진다. 위 출력에서 $n = 10$일 때는 표본상관이 $-0.05$로 부호마저 반대로 나왔고($p = 0.89$), $n = 1000$에서는 $r = 0.314$로 참값에 가까워지며 $p = 2 \times 10^{-24}$가 된다. $n = 10$에서는 p-값이 0.05를 훨씬 넘을 수 있다(실재하는 상관을 탐지하지 못한다). $n = 100$쯤이면 대체로 0.05 아래로 내려가고 $n = 1000$에서는 극도로 작아진다. 통계적 유의성이 효과크기와 표본크기 둘 다에 달려 있음을 보여준다. 중간 정도의 상관($\rho = 0.3$)도 표본이 작으면 유의하지 않고 표본이 크면 매우 유의해진다. p-값과 함께 효과크기를 보고해야 하는 이유이다.
