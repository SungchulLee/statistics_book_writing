# 카이제곱 밀도함수

## 개요

자유도가 $k$인 **카이제곱 분포**는 독립인 표준정규확률변수 $k$개의 제곱합의 분포로 나타난다:

$$
Q = Z_1^2 + Z_2^2 + \cdots + Z_k^2, \qquad Z_i \overset{\text{iid}}{\sim} N(0,1)
$$

가설검정(적합도 검정, 독립성 검정)과 분산의 신뢰구간 구성에서 근본적인 역할을 한다.

---

## PDF

$$
f(x; k) = \frac{1}{2^{k/2}\,\Gamma(k/2)}\, x^{k/2 - 1}\, e^{-x/2}, \qquad x \ge 0
$$

| 성질 | 값 |
|---|---|
| 지지집합 | $[0, \infty)$ |
| 평균 | $k$ |
| 분산 | $2k$ |
| 최빈값 | $\max(k - 2,\, 0)$ |

---

## 코드

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

k = 5                      # 자유도. 표준정규 k개를 제곱해 더한 것의 분포다.
chi2 = stats.chi2(df=k)

# x 범위를 분위수로 정한다. 눈대중으로 (0, 20) 같은 범위를 쓰면
# 자유도가 바뀔 때마다 그림이 잘리거나 남는다.
# ppf(1e-6)부터 ppf(1-1e-6)까지 잡으면 어떤 k에서도 꼬리까지 알맞게 담긴다.
x = np.linspace(chi2.ppf(1e-6), chi2.ppf(1 - 1e-6), 600)
y = chi2.pdf(x)

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(x, y, lw=2, label=f"χ² PDF (k={k})")
# 평균은 정확히 k, 최빈값은 k-2 (k >= 2일 때).
# 둘이 다르다는 것이 곧 이 분포가 오른쪽으로 치우쳐 있다는 뜻이다.
ax.axvline(k, linestyle='--', alpha=0.8, label=f"mean = {k}")
ax.axvline(max(k - 2, 0), linestyle=':', alpha=0.8, label=f"mode = {max(k-2, 0)}")
ax.set_title("Chi-square Distribution — PDF")
ax.set_xlabel("x")
ax.set_ylabel("density")
ax.legend()
ax.grid(True, linestyle=":")
plt.tight_layout()
plt.show()
```

![Chi-square Distribution — PDF](./img/chi_square_pdf_32.png)

---

## 자유도에 따른 모양

- **$k = 1, 2$:** 오른쪽으로 심하게 치우치며 밀도가 0 또는 그 근처에서 정점을 이룬다.
- **$k \approx 10$:** 치우침이 중간 정도이고 종 모양에 가깝지만 비대칭이다.
- **큰 $k$:** 중심극한정리에 의해 $\chi^2_k \approx N(k, 2k)$이다.

!!! note "정규분포와의 연결"
    $\chi^2_k$는 i.i.d. 확률변수 $k$개(각각 $Z_i^2$)의 합이므로, 중심극한정리가 큰 $k$에서 근사적 정규성을 보장한다.

---

## 연습문제

**연습문제 1.**
$Q \sim \chi^2_k$에 대해 정의 $Q = \sum_{i=1}^k Z_i^2$로부터 $E[Q]$와 $\text{Var}(Q)$를 계산하라.

??? success "풀이"
    각 $Z_i^2$에 대해 $E[Z_i^2] = 1$이고 $\text{Var}(Z_i^2) = E[Z_i^4] - (E[Z_i^2])^2 = 3 - 1 = 2$이다.

    독립성에 의해:

    $$
    E[Q] = \sum_{i=1}^k E[Z_i^2] = k, \qquad \text{Var}(Q) = \sum_{i=1}^k \text{Var}(Z_i^2) = 2k
    $$

---

**연습문제 2.**
$X \sim \chi^2_m$과 $Y \sim \chi^2_n$이 독립이면 $X + Y \sim \chi^2_{m+n}$임을 보여라.

??? success "풀이"
    모든 $Z_i, W_j$가 독립인 $N(0,1)$일 때 $X = \sum_{i=1}^m Z_i^2$, $Y = \sum_{j=1}^n W_j^2$로 쓰자. 그러면:

    $$
    X + Y = \sum_{i=1}^m Z_i^2 + \sum_{j=1}^n W_j^2
    $$

    이는 독립인 표준정규확률변수 $m + n$개의 제곱합이므로 정의에 의해 $X + Y \sim \chi^2_{m+n}$이다. $\square$

---

**연습문제 3.**
PDF를 미분하여 0으로 두고, $k \ge 2$일 때 $\chi^2_k$의 최빈값이 $k - 2$임을 보여라.

??? success "풀이"
    PDF에 로그를 취하면 $\ln f(x) = \text{const} + (k/2 - 1)\ln x - x/2$이다. 미분하면:

    $$
    \frac{d}{dx}\ln f(x) = \frac{k/2 - 1}{x} - \frac{1}{2} = 0
    $$

    풀면 $x = k - 2$이다. $k \ge 2$이면 이 값은 음이 아니고 지지집합 $[0, \infty)$에 속하므로 최빈값은 $k - 2$이다. $k < 2$이면 $x > 0$에서 도함수가 항상 음수이므로 최빈값은 $x = 0$이다.

---

**연습문제 4.**
$N(\mu, \sigma^2)$에서 크기 $n = 25$인 확률표본을 뽑아 $s^2 = 12$를 얻었다. 카이제곱 분포를 사용하여 $\sigma^2$에 대한 95% 신뢰구간을 구성하라.

??? success "풀이"
    추축량은 $(n-1)s^2/\sigma^2 \sim \chi^2_{n-1}$이다. $n-1 = 24$이므로:

    $$
    P\!\left(\chi^2_{0.025} \le \frac{24 \cdot 12}{\sigma^2} \le \chi^2_{0.975}\right) = 0.95
    $$

    SciPy를 사용하면 $\chi^2_{0.025, 24} = 12.40$, $\chi^2_{0.975, 24} = 39.36$이다.

    $$
    \frac{24 \times 12}{39.36} \le \sigma^2 \le \frac{24 \times 12}{12.40}
    $$

    $$
    7.32 \le \sigma^2 \le 23.23
    $$
