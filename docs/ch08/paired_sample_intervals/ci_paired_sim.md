# 대응 평균 신뢰구간의 포함확률 모의실험

## 개요

같은 피험자에게서 두 측정값을 얻으면(예: 처리 전후) 각 쌍 안의 관측값이 상관된다. 평균 차이 $\mu_D = \mu_X - \mu_Y$에 대한 대응표본 신뢰구간은 문제를 차이 $D_i = X_i - Y_i$에 대한 일표본 구간으로 환원한다. 이 페이지에서는 대응 자료에 세 가지 방법을 적용해 포함확률을 모의실험하고, 짝 내 상관이 구간 너비에 어떤 영향을 주는지 보인다.

## 대응 신뢰구간

대응 관측값 $(X_1, Y_1), \ldots, (X_n, Y_n)$이 주어졌을 때 차이를 $D_i = X_i - Y_i$로 정의한다. 차이의 표본평균과 표본표준편차는

$$
\bar{D} = \frac{1}{n}\sum_{i=1}^n D_i, \quad S_D = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (D_i - \bar{D})^2}
$$

### t-구간 (기본)

$$
\bar{D} \pm t_{\alpha/2,\,n-1} \cdot \frac{S_D}{\sqrt{n}}
$$

### D의 분산을 아는 z-구간

$\sigma_D$를 아는 경우(실무에서는 드물다):

$$
\bar{D} \pm z_{\alpha/2} \cdot \frac{\sigma_D}{\sqrt{n}}
$$

차이의 참 분산은 $\sigma_D^2 = \sigma_X^2 + \sigma_Y^2 - 2\rho\,\sigma_X \sigma_Y$이며, 여기서 $\rho$는 짝 내 상관이다.

### s를 대입한 z-구간

$$
\bar{D} \pm z_{\alpha/2} \cdot \frac{S_D}{\sqrt{n}}
$$

$n$이 큰 경우의 근사이며 작은 표본에서는 포함확률이 부족하다.

## 상관의 역할

차이의 분산에 주목하라:

$$
\operatorname{Var}(D) = \sigma_X^2 + \sigma_Y^2 - 2\rho\,\sigma_X\sigma_Y
$$

$\rho > 0$(양의 짝 내 상관)이면 $D$의 분산이 $\sigma_X^2 + \sigma_Y^2$보다 **줄어든다**. 이것이 짝짓기의 통계적 이점이다: 피험자 간 변동성의 상당 부분이 상쇄되어 신뢰구간이 좁아진다.

## Python 코드

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm

n_simulations = 100
n = 12
mu_x, mu_y = 0.5, 0.0
sigma_x, sigma_y = 1.0, 1.2
rho = 0.6
alpha = 0.05
method = "t"  # 't' | 'z_known' | 'z_plugin'

rng = np.random.default_rng(None)

delta_true = mu_x - mu_y
var_d_true = sigma_x**2 + sigma_y**2 - 2 * rho * sigma_x * sigma_y
sigma_d_true = np.sqrt(var_d_true)

# Build covariance matrix and Cholesky factor
cov = rho * sigma_x * sigma_y
Sigma = np.array([[sigma_x**2, cov], [cov, sigma_y**2]])
L = np.linalg.cholesky(Sigma)

df = n - 1
t_star = t.ppf(1 - alpha / 2, df=df)
z_star = norm.ppf(1 - alpha / 2)

lowers = np.empty(n_simulations)
uppers = np.empty(n_simulations)
centers = np.empty(n_simulations)

for i in range(n_simulations):
    z_vals = rng.standard_normal(size=(2, n))
    xy = (L @ z_vals).T
    x = xy[:, 0] + mu_x
    y = xy[:, 1] + mu_y
    d = x - y
    dbar = d.mean()
    s_d = d.std(ddof=1)

    if method == "t":
        se, crit = s_d / np.sqrt(n), t_star
    elif method == "z_known":
        se, crit = sigma_d_true / np.sqrt(n), z_star
    else:
        se, crit = s_d / np.sqrt(n), z_star

    lowers[i] = dbar - crit * se
    uppers[i] = dbar + crit * se
    centers[i] = dbar

covered = (lowers <= delta_true) & (delta_true <= uppers)
coverage_pct = 100.0 * covered.mean()
print(f"Paired {method} coverage: {coverage_pct:.1f}%")
```

### 구간의 시각화

```python
fig, ax = plt.subplots(figsize=(12, 12))
for i in range(n_simulations):
    color = "k" if covered[i] else "r"
    ax.plot([lowers[i], uppers[i]], [i, i], lw=2, color=color)
    ax.plot(centers[i], i, marker="o", ms=3, color=color)

ax.axvline(delta_true, linestyle="--", linewidth=1.5, color="r")
n_fail = int((~covered).sum())
ax.set_title(f"{n_simulations} Paired {method} CIs | n={n}, rho={rho}, CL=95%")
ax.set_yticks([])
ax.set_xlabel("Mean difference")
plt.tight_layout()
plt.show()
```

## 해석

- 대응 $t$-구간은 $\sigma_D$의 추정을 올바르게 반영하므로 명목 95% 포함확률을 달성한다.
- 짝 내 상관 $\rho$가 클수록 $\sigma_D$가 줄어 신뢰구간이 좁아진다.
- 대입한 $z$-구간은 정규 임계값이 $t$ 임계값보다 작아 작은 $n$에서 포함확률이 부족하다.
- 짝짓기는 $\rho > 0$일 때 이롭다. $\rho \le 0$이면 짝짓기가 오히려 독립 이표본 설계보다 구간을 **넓힐** 수 있으므로 설계를 다시 생각해야 한다.

## 연습문제

**연습문제 1.** 환자 10명의 혈압을 투약 전후로 측정했다. 차이 $D_i$(투약 전 빼기 투약 후)는 5, 3, 8, 2, 6, 4, 7, 1, 5, 3이다. $\mu_D$의 95% $t$-구간을 구성하라.

??? success "풀이"

    요약통계량을 계산하면:

    $$
    \bar{D} = \frac{5+3+8+2+6+4+7+1+5+3}{10} = \frac{44}{10} = 4.4
    $$

    $$
    S_D = \sqrt{\frac{1}{9}\sum_{i=1}^{10}(D_i - 4.4)^2} = \sqrt{\frac{1}{9}(0.36+1.96+12.96+5.76+2.56+0.16+6.76+11.56+0.36+1.96)} = \sqrt{\frac{44.4}{9}} = \sqrt{4.933} \approx 2.221
    $$

    $\text{df} = 9$이고 $t_{0.025,9} = 2.262$이므로:

    $$
    4.4 \pm 2.262 \times \frac{2.221}{\sqrt{10}} = 4.4 \pm 2.262 \times 0.7024 = 4.4 \pm 1.589
    $$

    $\mu_D$의 95% 신뢰구간은 $(2.81, 5.99)$이다. 구간 전체가 양수이므로 이 약이 혈압을 낮추는 것으로 보인다. $\square$

---

**연습문제 2.** 대응 관측값에 대해 공식 $\sigma_D^2 = \sigma_X^2 + \sigma_Y^2 - 2\rho\,\sigma_X\sigma_Y$를 유도하라.

??? success "풀이"

    $D = X - Y$라 하자. 분산의 성질에 의해:

    $$
    \operatorname{Var}(D) = \operatorname{Var}(X - Y) = \operatorname{Var}(X) + \operatorname{Var}(Y) - 2\operatorname{Cov}(X,Y)
    $$

    상관계수의 정의에 의해 $\operatorname{Cov}(X,Y) = \rho\,\sigma_X\sigma_Y$이므로

    $$
    \sigma_D^2 = \sigma_X^2 + \sigma_Y^2 - 2\rho\,\sigma_X\sigma_Y
    $$

    를 얻는다. $\rho > 0$이면 빼지는 항 $2\rho\,\sigma_X\sigma_Y > 0$ 덕분에 $D$의 분산이, $X$와 $Y$가 독립일 때 나올 값인 $\sigma_X^2 + \sigma_Y^2$보다 작아진다. $\square$

---

**연습문제 3.** $\sigma_X = \sigma_Y = \sigma$이고 $\rho = 0.8$이라 하자. $n$쌍인 대응 설계에서 $\bar{D}$의 표준오차와, 집단당 관측값이 $n$개인 독립 이표본 설계에서 $\bar{X} - \bar{Y}$의 표준오차를 비교하라.

??? success "풀이"

    **대응 설계:** $\sigma_D^2 = \sigma^2 + \sigma^2 - 2(0.8)\sigma^2 = 2\sigma^2(1 - 0.8) = 0.4\sigma^2$. 표준오차는

    $$
    \text{SE}_{\text{paired}} = \frac{\sigma_D}{\sqrt{n}} = \frac{\sigma\sqrt{0.4}}{\sqrt{n}} = \frac{0.632\,\sigma}{\sqrt{n}}
    $$

    **독립 설계:** $\operatorname{Var}(\bar{X} - \bar{Y}) = \sigma^2/n + \sigma^2/n = 2\sigma^2/n$. 표준오차는

    $$
    \text{SE}_{\text{indep}} = \sqrt{\frac{2\sigma^2}{n}} = \frac{\sigma\sqrt{2}}{\sqrt{n}} = \frac{1.414\,\sigma}{\sqrt{n}}
    $$

    비는 $\text{SE}_{\text{paired}}/\text{SE}_{\text{indep}} = \sqrt{0.4}/\sqrt{2} = \sqrt{0.2} \approx 0.447$이다. 대응 설계가 표준오차를 절반 넘게 줄여 훨씬 좁은 신뢰구간을 준다. $\square$

---

**연습문제 4.** 어떤 $\rho$ 값에서 대응 설계가 독립 설계보다 나을 것이 없어지는가? $\rho < 0$이면 어떻게 되는가?

??? success "풀이"

    $\sigma_X = \sigma_Y = \sigma$일 때 대응 분산은 $\sigma_D^2 = 2\sigma^2(1-\rho)$이고, (집단당 $n$개인) 독립 설계에서 $\bar{X}-\bar{Y}$의 분산은 $2\sigma^2/n$이다. 표준오차를 비교하면:

    $$
    \text{SE}_{\text{paired}} = \frac{\sigma\sqrt{2(1-\rho)}}{\sqrt{n}}, \quad \text{SE}_{\text{indep}} = \frac{\sigma\sqrt{2}}{\sqrt{n}}
    $$

    $\sqrt{2(1-\rho)} = \sqrt{2}$일 때, 즉 $\rho = 0$일 때 둘이 같다. $\rho = 0$이면 짝 안의 측정값이 무상관이어서 짝짓기가 분산을 줄여 주지 않는다.

    $\rho < 0$이면 $2(1-\rho) > 2$이므로 $\text{SE}_{\text{paired}} > \text{SE}_{\text{indep}}$이다. 음의 짝 내 상관은 오히려 차이의 분산을 **키워** 대응 설계를 독립 설계보다 **못하게** 만든다. 실무에서 흔치는 않지만, 예컨대 짝지은 피험자들이 반대 방향으로 반응하는 경향이 있다면 생길 수 있다. $\square$

---

**연습문제 5.** 어떤 연구가 대응 관측값 $n = 15$쌍을 쓴다. 표본 평균 차이는 $\bar{D} = 2.3$, $S_D = 4.1$이다. 95% 신뢰구간에 0이 들어 있는지 확인하여 5% 수준에서 $\mu_D = 0$을 검정하라.

??? success "풀이"

    $\text{df} = 14$이고 $t_{0.025,14} = 2.145$이므로:

    $$
    \text{SE} = \frac{4.1}{\sqrt{15}} = \frac{4.1}{3.873} = 1.059
    $$

    $$
    \bar{D} \pm t_{0.025,14} \times \text{SE} = 2.3 \pm 2.145 \times 1.059 = 2.3 \pm 2.272
    $$

    95% 신뢰구간은 $(0.028, 4.572)$이다. $0$이 (아슬아슬하게) 이 구간 밖에 있으므로 5% 유의수준에서 $H_0: \mu_D = 0$을 기각한다. 자료는 참 평균 차이가 양수라는 증거를 준다. $\square$
