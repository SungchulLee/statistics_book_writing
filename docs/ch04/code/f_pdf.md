# F 분포 밀도함수

## 개요

분자 자유도 $d_1$과 분모 자유도 $d_2$를 갖는 **$F$ 분포**는 독립인 두 카이제곱 확률변수를 자유도로 나눈 뒤 그 비의 분포이다:

$$
F = \frac{U/d_1}{V/d_2}, \qquad U \sim \chi^2_{d_1},\; V \sim \chi^2_{d_2}
$$

분산분석의 F 검정, 분산 비교, 내포된 회귀모형 검정에서 핵심이 되는 분포이다.

---

## 주요 성질

| 성질 | 조건 | 값 |
|---|---|---|
| 지지집합 | — | $[0, \infty)$ |
| 평균 | $d_2 > 2$ | $\dfrac{d_2}{d_2 - 2}$ |
| 최빈값 | $d_1 > 2$ | $\dfrac{d_1 - 2}{d_1} \cdot \dfrac{d_2}{d_2 + 2}$ |
| 분산 | $d_2 > 4$ | $\dfrac{2d_2^2(d_1 + d_2 - 2)}{d_1(d_2-2)^2(d_2-4)}$ |

---

## 코드

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

d1, d2 = 5, 12
f_dist = stats.f(dfn=d1, dfd=d2)

x = np.linspace(f_dist.ppf(1e-6), f_dist.ppf(1 - 1e-6), 600)
y = f_dist.pdf(x)

mean = d2 / (d2 - 2)
mode = ((d1 - 2) / d1) * (d2 / (d2 + 2))

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(x, y, lw=2, label=f"F PDF (d1={d1}, d2={d2})")
ax.axvline(mean, linestyle='--', alpha=0.85, label=f"mean = {mean:.3f}")
ax.axvline(mode, linestyle=':', alpha=0.85, label=f"mode = {mode:.3f}")
ax.set_title("F Distribution — PDF")
ax.set_xlabel("x")
ax.set_ylabel("density")
ax.legend()
ax.grid(True, linestyle=":")
plt.tight_layout()
plt.show()
```

---

## 해석

$F$ 분포는 ($[0, \infty)$ 위에 놓이므로) 언제나 오른쪽으로 치우쳐 있다. $d_1$과 $d_2$가 크면 1 근처를 중심으로 하는 정규분포에 가까워진다. 평균은 1보다 크며($d_2/(d_2-2)$와 같다), 이는 분산비가 갖는 약간의 양의 편향을 반영한다.

---

## 연습문제

**연습문제 1.**
$F_{5, 12}$의 평균을 계산하고, $F$ 분포의 평균이 (존재할 때) 항상 1보다 큰 이유를 설명하라.

??? success "연습문제 1 풀이"
    $E[F] = d_2/(d_2 - 2) = 12/10 = 1.2$.

    평균이 1을 넘는 이유는 분모의 카이제곱이 $d_2$로 나누어지지만 평균에는 $d_2$만큼 기여하기 때문이다($E[\chi^2_{d_2}] = d_2$이므로). 비 $E[V/d_2] = 1$이지만 역수가 볼록함수이므로 Jensen 부등식에 의해 $E[d_1/U]$에 위쪽으로의 보정이 생긴다. 형식적으로는 $E[1/(V/d_2)] > 1/E[V/d_2] = 1$이다.

---

**연습문제 2.**
$T \sim t_\nu$이면 $T^2 \sim F_{1, \nu}$임을 보여라.

??? success "연습문제 2 풀이"
    정의에 의해 $T = Z/\sqrt{V/\nu}$이며 $Z \sim N(0,1)$과 $V \sim \chi^2_\nu$는 독립이다. 그러면:

    $$
    T^2 = \frac{Z^2}{V/\nu} = \frac{Z^2/1}{V/\nu}
    $$

    $Z^2 \sim \chi^2_1$이므로 이는 비 $(\chi^2_1/1)/(\chi^2_\nu/\nu)$이며, 정의에 의해 $F_{1,\nu}$를 따른다. $\square$

---

**연습문제 3.**
크기가 10인 세 집단으로 이루어진 일원배치 분산분석에서 $F$ 검정의 자유도는 얼마인가? 유의수준 5%에서 임계값은?

??? success "연습문제 3 풀이"

    - 집단 간 자유도: $d_1 = k - 1 = 2$
    - 집단 내 자유도: $d_2 = N - k = 30 - 3 = 27$

    임계값은 `stats.f.ppf(0.95, 2, 27) ≈ 3.354`이다.

    관측된 $F$ 통계량이 3.354를 넘으면 모든 집단 평균이 같다는 귀무가설을 기각한다.

---

**연습문제 4.**
$1/F_{d_1, d_2} \sim F_{d_2, d_1}$임을 보여라.

??? success "연습문제 4 풀이"
    $U \sim \chi^2_{d_1}$과 $V \sim \chi^2_{d_2}$가 독립일 때 $F = (U/d_1)/(V/d_2)$이면:

    $$
    \frac{1}{F} = \frac{V/d_2}{U/d_1}
    $$

    이는 독립인 두 카이제곱을 자유도로 나눈 비이며 분자의 자유도가 $d_2$, 분모의 자유도가 $d_1$이므로 $1/F \sim F_{d_2, d_1}$이다. $\square$
