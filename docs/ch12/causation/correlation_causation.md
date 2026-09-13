# 상관과 인과

## 개요

이 페이지는 상관과 인과의 결정적인 차이를 탐구한다. 서로 다른 관계 유형에 대해 Pearson, Spearman, Kendall 상관계수를 비교하고, Fisher $z$ 변환으로 신뢰구간을 만들며, Simpson의 역설을 시연하고, 교란변수를 통제하는 부분상관을 계산하며, 다중검정이 어떻게 허위상관을 만들어 내는지 보인다. 관찰자료에서 타당한 결론을 이끌어 내려면 이 주제들을 이해해야 한다.

---

## 상관 측도 비교

짝지어진 관측값 $(x_1, y_1), \ldots, (x_n, y_n)$이 주어졌을 때, 세 가지 표준적인 상관 측도는 연관의 서로 다른 측면을 포착한다.

- **Pearson의 $r$**는 선형 연관을 측정한다.
- **Spearman의 $\rho_s$**는 순위에 적용한 Pearson의 $r$로, 단조 관계를 포착한다.
- **Kendall의 $\tau$**는 일치쌍과 불일치쌍의 개수를 센다.

관계가 선형이면 셋이 모두 일치한다. 단조이지만 비선형이면 Spearman과 Kendall이 Pearson보다 낫다. 이상점이 있으면 순위 기반 측도가 더 로버스트하다.

<div class="codebox" markdown>

**예제 1.** 관계의 모양에 따른 세 측도

```python
import numpy as np
from scipy import stats

np.random.seed(42)
n = 100

# 선형 관계. 세 측도가 모두 크게 나온다.
x_lin = np.random.normal(0, 1, n)
y_lin = 2 * x_lin + np.random.normal(0, 1, n)

# 단조이지만 곡선인 관계. Pearson 은 떨어지지만 순위를 쓰는 두 측도는 버틴다.
x_mono = np.random.uniform(0, 3, n)
y_mono = np.exp(x_mono) + np.random.normal(0, 2, n)

# 이차함수 관계. 관계는 아주 강한데 세 측도 모두 0 근처로 나온다.
# 상관계수가 0 이라는 말이 "관계가 없다"는 뜻이 아님을 보여 주는 자리다.
x_quad = np.random.normal(0, 2, n)
y_quad = x_quad**2 + np.random.normal(0, 1, n)

datasets = [
    ("Linear", x_lin, y_lin),
    ("Monotonic Nonlinear", x_mono, y_mono),
    ("Quadratic (r ~ 0)", x_quad, y_quad),
]

for name, x, y in datasets:
    r_p, _ = stats.pearsonr(x, y)
    r_s, _ = stats.spearmanr(x, y)
    r_k, _ = stats.kendalltau(x, y)
    print(f"{name:<25} r={r_p:.4f}  rho_s={r_s:.4f}  tau={r_k:.4f}")
```

출력:

```text
Linear                    r=0.8724  rho_s=0.8686  tau=0.6853
Monotonic Nonlinear       r=0.8676  rho_s=0.9032  tau=0.7402
Quadratic (r ~ 0)         r=0.1014  rho_s=-0.0229  tau=-0.0376
```

선형 관계에서는 Pearson이 가장 크고, 단조 비선형(지수) 관계에서는 Spearman과 Kendall이 Pearson을 앞선다(0.903 대 0.868). 이차 관계에서는 $y$가 사실상 $x$의 결정론적 함수인데도 Pearson의 $r$가 0에 가깝다. Pearson 상관이 낮다고 해서 "관계가 없다"는 뜻이 아님을 보여준다.

</div>

---

## Fisher z 변환과 신뢰구간

Pearson의 $r$는 표집분포가 치우쳐 있으며, 특히 $|\rho|$가 클 때 그렇다. **Fisher $z$ 변환**은 분산을 안정화한다.

$$
z = \operatorname{arctanh}(r) = \frac{1}{2}\ln\!\left(\frac{1+r}{1-r}\right)
$$

귀무가설 $\rho = \rho_0$ 아래에서 변환된 통계량은 근사적으로 정규분포를 따른다.

$$
z \;\dot\sim\; \mathcal{N}\!\left(\operatorname{arctanh}(\rho_0),\; \frac{1}{n-3}\right)
$$

$\rho$에 대한 $(1-\alpha)$ 신뢰구간은 변환을 되돌려 얻는다.

$$
\left(\tanh\!\bigl(z - z_{\alpha/2}\,\text{SE}\bigr),\;\; \tanh\!\bigl(z + z_{\alpha/2}\,\text{SE}\bigr)\right), \qquad \text{SE} = \frac{1}{\sqrt{n-3}}
$$

<div class="codebox" markdown>

**예제 2.** Fisher z 신뢰구간

```python
def fisher_z_ci(x, y, alpha=0.05):
    """Fisher z 변환으로 상관계수의 신뢰구간을 구한다.

    r 은 -1 과 1 사이에 갇혀 있어 분포가 치우친다. arctanh 를 씌우면
    그 눈금이 실수 전체로 펴지며 분포가 거의 정규가 되고, 표준오차도
    1/sqrt(n-3) 로 간단해진다. 구간을 만든 뒤 tanh 로 되돌린다.
    """
    n = len(x)
    r, p_val = stats.pearsonr(x, y)
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    z_lo, z_hi = z - z_crit * se, z + z_crit * se
    rho_lo, rho_hi = np.tanh(z_lo), np.tanh(z_hi)
    print(f"r = {r:.4f}, 95% CI for rho: ({rho_lo:.4f}, {rho_hi:.4f})")
    return rho_lo, rho_hi

np.random.seed(42)
x = np.random.normal(0, 1, 80)
y = 0.6 * x + np.random.normal(0, 0.8, 80)
fisher_z_ci(x, y)
```

출력:

```text
r = 0.6110, 95% CI for rho: (0.4519, 0.7324)
```

구간이 $\tanh$ 변환 때문에 점추정값 $0.611$을 중심으로 대칭이 아니라는 점에 주목하라. 위쪽 폭($0.121$)이 아래쪽 폭($0.159$)보다 좁다.

</div>

---

## Simpson의 역설

Simpson의 역설은 교란변수로 조건화한 뒤 연관의 방향이 뒤집힐 때 일어난다. 형식적으로 다음이 가능하다.

$$
r(X, Y) > 0 \qquad \text{그러나} \qquad r(X, Y \mid Z = z) < 0 \;\;\text{(모든 } z \text{에 대해)}
$$

이는 잠복변수 $Z$가 $X$와 $Y$ 모두와 양의 연관을 가질 때 일어난다. 집단 내 관계는 음수인데도 전체적으로는 허위의 양의 상관이 생긴다.

<div class="codebox" markdown>

**예제 3.** 심슨의 역설

```python
np.random.seed(42)
# 집단마다 x 의 평균이 커질수록 y 의 기준선도 함께 올라간다. 이것이 교란이다.
# 집단 안의 기울기는 셋 다 -0.5 로 음인데, 합쳐 놓으면 양이 된다.
groups = {"Group A": (50, 0.2, 2, -0.5),
          "Group B": (50, 0.5, 5, -0.5),
          "Group C": (50, 0.8, 8, -0.5)}

all_x, all_y = [], []
for name, (n, xm, yb, slope) in groups.items():
    x = np.random.normal(xm, 0.15, n)
    y = yb + slope * x + np.random.normal(0, 0.3, n)
    all_x.extend(x)
    all_y.extend(y)

all_x, all_y = np.array(all_x), np.array(all_y)
m_all, b_all = np.polyfit(all_x, all_y, 1)
r_overall, _ = stats.pearsonr(all_x, all_y)

print(f"Overall slope: {m_all:.2f} (positive)")
print(f"Within-group slope: -0.5 (negative)")
print(f"Overall r = {r_overall:.4f}")
```

출력:

```text
Overall slope: 6.50 (positive)
Within-group slope: -0.5 (negative)
Overall r = 0.8593
```

</div>

!!! warning "역설이 성립하려면 절편이 함께 움직여야 한다"
    핵심은 집단의 $x$ 평균이 커질수록 기준선 `yb`도 함께 커진다는 데 있다($0.2 \to 2$, $0.5 \to 5$, $0.8 \to 8$). 만약 기준선이 반대로 감소한다면($8, 5, 2$) 전체 기울기는 $-7.38$로 오히려 더 가파른 음수가 되어 역설이 일어나지 않는다. 집단 간 이동 방향이 집단 내 기울기와 **반대**일 때만 부호가 뒤집힌다.

전체 회귀직선의 기울기는 $+6.50$으로 양수인데, 집단마다의 회귀직선은 모두 기울기가 $-0.5$로 음수이다. 집단 변수를 무시하면 정반대의 결론에 이른다.

---

## 부분상관

부분상관은 교란변수 $Z$의 선형 효과를 $X$와 $Y$ 양쪽에서 제거한다. 1차 부분상관은 다음과 같다.

$$
r_{XY \cdot Z} = \frac{r_{XY} - r_{XZ}\,r_{YZ}}{\sqrt{(1 - r_{XZ}^2)(1 - r_{YZ}^2)}}
$$

동등하게, $X$를 $Z$에, $Y$를 $Z$에 회귀시킨 뒤 두 잔차의 Pearson 상관을 계산해도 된다.

<div class="codebox" markdown>

**예제 4.** 부분상관

```python
n = 200
np.random.seed(42)

# 앞과 같은 구조다. Z 가 X 와 Y 를 함께 끌어 상관을 만든다.
Z = np.random.normal(0, 1, n)
X = 0.7 * Z + np.random.normal(0, 0.5, n)
Y = 0.6 * Z + np.random.normal(0, 0.5, n)

r_xy, p_xy = stats.pearsonr(X, Y)
r_xz, _ = stats.pearsonr(X, Z)
r_yz, _ = stats.pearsonr(Y, Z)

r_xy_z = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2))

print(f"r(X, Y)    = {r_xy:.4f}  (appears significant)")
print(f"r(X,Y | Z) = {r_xy_z:.4f}  (nearly vanishes)")
```

출력:

```text
r(X, Y)    = 0.5699  (appears significant)
r(X,Y | Z) = -0.0201  (nearly vanishes)
```

$r(X, Y) = 0.570$은 $p \approx 1.3 \times 10^{-18}$로 압도적으로 유의하지만, 부분상관 $r_{XY \cdot Z} = -0.020$은 0에 가깝다. 겉보기 연관이 전적으로 교란변수 $Z$에서 비롯되었음이 드러난다.

</div>

---

## 다중검정이 만드는 허위상관

서로 독립인 변수 여러 개를 쌍마다 검정하면 순전히 우연으로 상당수의 "유의한" 상관이 나타난다. 변수가 $p$개면 쌍은 $\binom{p}{2}$개이다. 유의수준 $\alpha$에서 거짓양성의 기댓값은 다음과 같다.

$$
E[\text{거짓양성}] = \alpha \binom{p}{2}
$$

<div class="codebox" markdown>

**예제 5.** 다중검정이 만드는 허위상관

```python
def spurious_correlations_demo(n_vars=100, n_obs=30):
    """서로 완전히 무관한 변수 100개에서 유의한 상관이 몇 쌍이나 나오는지 센다.

    쌍이 4950개이므로 유의수준 5%에서 247쌍쯤은 그냥 나온다. 자료를 훑다가
    찾아낸 상관 하나를 그대로 보고하면 안 되는 까닭이 여기 있다.
    """
    np.random.seed(42)
    data = np.random.normal(0, 1, (n_obs, n_vars))
    n_pairs = n_vars * (n_vars - 1) // 2
    p_values = []
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            _, p = stats.pearsonr(data[:, i], data[:, j])
            p_values.append(p)
    p_values = np.array(p_values)
    n_sig = np.sum(p_values < 0.05)
    print(f"Pairs tested: {n_pairs}")
    print(f"Significant at 0.05: {n_sig} ({100*n_sig/n_pairs:.1f}%)")
    print(f"Expected false positives: {0.05 * n_pairs:.0f}")

spurious_correlations_demo()
```

출력:

```text
Pairs tested: 4950
Significant at 0.05: 240 (4.8%)
Expected false positives: 248
```

100개 변수가 모두 독립인데도 약 5%의 쌍이 유의하게 나타난다. 이것이 다중검정 문제이며, Bonferroni나 Benjamini--Hochberg 같은 보정이 필요하다.

</div>

---

## 독립인 두 상관의 비교

두 모집단 상관이 같은지, 곧 $H_0\colon \rho_1 = \rho_2$를 검정하려면 각각에 Fisher $z$ 변환을 적용한다.

$$
z = \frac{\operatorname{arctanh}(r_1) - \operatorname{arctanh}(r_2)}{\sqrt{\dfrac{1}{n_1-3} + \dfrac{1}{n_2-3}}}
$$

$H_0$ 아래에서 $z$는 근사적으로 표준정규분포를 따른다.

<div class="codebox" markdown>

**예제 6.** 두 상관의 비교

```python
def compare_two_correlations(r1, n1, r2, n2, alpha=0.05):
    """서로 독립인 두 표본의 상관계수가 다른지 검정한다.

    각각을 z 로 옮기면 차이의 분포가 정규가 되므로 z 검정을 쓸 수 있다.
    두 표본이 겹치지 않을 때만 이 방법이 맞다.
    """
    z1, z2 = np.arctanh(r1), np.arctanh(r2)
    se = np.sqrt(1/(n1 - 3) + 1/(n2 - 3))
    z_stat = (z1 - z2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    print(f"r1={r1:.4f} (n={n1}), r2={r2:.4f} (n={n2})")
    print(f"z = {z_stat:.4f}, p = {p_value:.4f}")

compare_two_correlations(r1=0.72, n1=100, r2=0.65, n2=120)
```

출력:

```text
r1=0.7200 (n=100), r2=0.6500 (n=120)
z = 0.9638, p = 0.3351
```

$p = 0.335$이므로 두 상관이 다르다는 증거가 없다. $0.72$와 $0.65$라는 차이는 이 정도 표본크기에서 우연히 생길 만하다.

</div>

---

## 해석

이 페이지의 예들은 통계학의 핵심 원리 하나를 보여준다. **상관은 인과를 뜻하지 않는다.** 구체적으로,

- Pearson $r$가 크다는 것은 선형 연관만 포착한다. 비선형 관계나 이상점이 있으면 오도할 수 있다.
- Simpson의 역설은 집계된 자료가 모든 하위집단에서 성립하는 연관의 부호를 뒤집을 수 있음을 보여준다.
- 부분상관은 겉보기 연관이 전적으로 교란변수 때문일 수 있음을 드러낸다.
- 다중검정은 순수한 잡음에서 허위의 "유의한" 상관을 만들어 낸다.

인과를 확립하려면 무작위 실험이 있거나, 도구변수나 유향비순환그래프 같은 신중히 정당화된 인과모형이 필요하다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span> $x \sim \text{Uniform}(-\pi, \pi)$, $\varepsilon \sim \mathcal{N}(0, 0.3^2)$인 모형 $y = \cos(x) + \varepsilon$에서 $n = 150$개의 관측값을 생성하라. Pearson의 $r$, Spearman의 $\rho_s$, Kendall의 $\tau$를 계산하라. $y$가 (잡음을 빼면) $x$의 결정론적 함수인데도 셋이 모두 0에 가까운 이유를 설명하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats

    np.random.seed(42)
    n = 150
    x = np.random.uniform(-np.pi, np.pi, n)
    y = np.cos(x) + np.random.normal(0, 0.3, n)

    r_p, _ = stats.pearsonr(x, y)
    r_s, _ = stats.spearmanr(x, y)
    r_k, _ = stats.kendalltau(x, y)
    print(f"Pearson r  = {r_p:.4f}")   # 0.0580
    print(f"Spearman   = {r_s:.4f}")   # 0.1024
    print(f"Kendall    = {r_k:.4f}")   # 0.0851
    ```

    출력:

    ```
    Pearson r  = 0.0580
    Spearman   = 0.1024
    Kendall    = 0.0851
    ```

    세 계수 모두 0.06~0.10으로 0에 가깝다. 관계가 없어서가 아니라 그 관계가 **단조가 아니기** 때문이다. 세 계수 어느 것도 U자 관계를 잡아내도록 만들어지지 않았다.

    $\cos$ 함수는 $[-\pi, \pi]$에서 원점에 대해 **우함수**이다. $[-\pi, 0]$에서 증가하고 $[0, \pi]$에서 감소하므로, 두 구간의 단조 성분이 정확히 상쇄된다. 실제로 $x$가 이 구간에서 균등분포이면

    $$
    \text{Cov}(x, \cos x) = \frac{1}{2\pi}\int_{-\pi}^{\pi} x \cos x \, dx = 0
    $$

    이다(피적분함수가 기함수이다). 세 계수는 모두 선형 연관이나 단조 연관을 측정하므로, 강한 결정론적 관계가 있음에도 0에 가깝다. 상관이 탐지하지 못하는 비단조 종속의 예이다.

    !!! note "$\sin$이었다면 결과가 다르다"
        같은 구간에서 $y = \sin(x)$로 바꾸면 $\sin$은 기함수이고 $x$와 순증가하는 성분을 가지므로 $r \approx 0.74$가 나온다. 비선형이라고 해서 상관이 자동으로 0이 되는 것이 아니라, 대칭성 때문에 상쇄가 일어나야 0이 된다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 한 연구자가 $n = 50$에서 $r = 0.45$를 얻고 모집단 상관이 "대략 0.45"라고 주장한다. Fisher $z$ 변환으로 99% 신뢰구간을 만들고, 이 점추정값이 그런 주장을 뒷받침할 만큼 정밀한지 평가하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats

    r = 0.45
    n = 50
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(0.995)
    lo = np.tanh(z - z_crit * se)
    hi = np.tanh(z + z_crit * se)
    print(f"99% CI: ({lo:.4f}, {hi:.4f})")
    ```

    출력:

    ```
    99% CI: (0.1085, 0.6965)
    ```

    99% 구간이 $(0.109, 0.697)$로 대단히 넓다. 표본이 작으면 상관계수의 불확실성이 이만큼 크다는 것을 구간이 보여준다. 점추정값만 보고하면 이 폭이 숨는다.

    99% 신뢰구간은 약 $(0.109, 0.697)$이다. 폭이 $0.59$에 이를 만큼 매우 넓다. 참 $\rho$는 약한 양의 상관부터 강한 양의 상관까지 어디든 될 수 있다. 관측값이 $n = 50$뿐일 때 $0.45$라는 점추정값은 결코 정밀하지 않으며, 구간 없이 점추정값만 보고하는 것은 오도하는 일이다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> 두 집단으로 Simpson의 역설 예를 구성하라. 집단 1은 $n = 60$이고 $x$ 평균이 1 근처, $y$ 절편이 2 근처이며, 집단 2는 $n = 60$이고 $x$ 평균이 4 근처, $y$ 절편이 10 근처라 하자. 두 집단 모두 집단 내 기울기는 $-1$로 둔다. 집단 내 기울기는 모두 음수인데 전체 기울기는 양수임을 확인하라.

</div>

??? success "풀이"

    ```python
    import numpy as np

    np.random.seed(0)
    # 1집단: 기준선도 낮고 x 도 작다
    x1 = np.random.normal(1, 0.3, 60)
    y1 = 2 - 1.0 * x1 + np.random.normal(0, 0.5, 60)

    # 2집단: 기준선도 높고 x 도 크다
    x2 = np.random.normal(4, 0.3, 60)
    y2 = 10 - 1.0 * x2 + np.random.normal(0, 0.5, 60)

    # 집단 안에서의 기울기
    m1, _ = np.polyfit(x1, y1, 1)
    m2, _ = np.polyfit(x2, y2, 1)

    # 합쳐 놓았을 때의 기울기 — 부호가 뒤집힌다
    x_all = np.concatenate([x1, x2])
    y_all = np.concatenate([y1, y2])
    m_all, _ = np.polyfit(x_all, y_all, 1)

    print(f"Group 1 slope: {m1:.3f}")
    print(f"Group 2 slope: {m2:.3f}")
    print(f"Overall slope: {m_all:.3f}")
    ```

    출력:

    ```text
    Group 1 slope: -1.130
    Group 2 slope: -1.119
    Overall slope: 1.531
    ```

    집단 내 기울기는 둘 다 $-1$ 근처(음수)인데 전체 기울기는 $+1.531$로 양수이다. $x$가 큰 집단(집단 2)이 절편 차이 덕분에 $y$도 크기 때문이다. 집단 중심은 $(1, 1)$과 $(4, 6)$이므로 두 중심을 잇는 직선의 기울기는 $(6-1)/(4-1) \approx 1.67$이고, 이 집단 간 이동이 집단 내 음의 기울기를 압도한다. 교란변수인 집단 소속이 역설을 만든다.

    절편을 반대로 주면(집단 1의 절편이 10, 집단 2의 절편이 2) 집단 간 이동 방향이 집단 내 기울기와 같은 방향이 되어 전체 기울기가 더 가파른 음수가 될 뿐, 역설은 일어나지 않는다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span> 부분상관 공식

$$
r_{XY \cdot Z} = \frac{r_{XY} - r_{XZ}\,r_{YZ}}{\sqrt{(1 - r_{XZ}^2)(1 - r_{YZ}^2)}}
$$

이 주어졌을 때 $|r_{XY \cdot Z}| \le 1$임을 증명하라. 어떤 조건에서 $r_{XY \cdot Z} = 0$인가?

</div>

??? success "풀이"

    $X$와 $Y$를 각각 $Z$에 회귀시킨 뒤의 잔차 $e_X = X - \hat{X}_{Z}$와 $e_Y = Y - \hat{Y}_{Z}$를 생각하자. 다음이 알려져 있다.

    $$
    r_{XY \cdot Z} = r(e_X, e_Y)
    $$

    $r(e_X, e_Y)$는 Pearson 상관이므로 Cauchy--Schwarz 부등식에 의해 $|r(e_X, e_Y)| \le 1$이다. 따라서 $|r_{XY \cdot Z}| \le 1$이다.

    $r_{XY \cdot Z} = 0$인 것은 잔차 $e_X$와 $e_Y$가 무상관인 것과 동치이고, 이는 다음일 때 일어난다.

    $$
    r_{XY} = r_{XZ} \cdot r_{YZ}
    $$

    이는 $X$와 $Y$ 사이의 주변상관 전부가 두 변수가 $Z$에 공통으로 선형 의존하는 데서 설명된다는 뜻이다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> 모의실험을 수행하라. 각각 $n = 25$개의 관측값을 가진 독립인 표준정규 변수 200개를 생성하고 $\binom{200}{2} = 19{,}900$개의 쌍별 Pearson 상관을 모두 계산하라. (a) $\alpha = 0.05$에서 유의한 것은 몇 개인가? (b) Bonferroni 보정을 적용하면 몇 개가 유의하게 남는가? (c) 이 맥락에서 제1종 오류와 제2종 오류의 상충관계를 논하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats

    np.random.seed(42)
    n_vars, n_obs = 200, 25
    data = np.random.normal(0, 1, (n_obs, n_vars))

    p_values = []
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            _, p = stats.pearsonr(data[:, i], data[:, j])
            p_values.append(p)

    p_values = np.array(p_values)
    n_pairs = len(p_values)
    n_sig = np.sum(p_values < 0.05)
    bonf_threshold = 0.05 / n_pairs
    n_sig_bonf = np.sum(p_values < bonf_threshold)

    print(f"Total pairs: {n_pairs}")
    print(f"Significant at 0.05: {n_sig}")
    print(f"Bonferroni threshold: {bonf_threshold:.2e}")
    print(f"Significant after Bonferroni: {n_sig_bonf}")
    ```

    출력:

    ```text
    Total pairs: 19900
    Significant at 0.05: 961
    Bonferroni threshold: 2.51e-06
    Significant after Bonferroni: 0
    ```

    (a) $961$개의 쌍이 우연히 유의하게 나타난다. 기댓값 $0.05 \times 19{,}900 = 995$와 잘 맞는다.

    (b) Bonferroni 보정 뒤(문턱값 $\approx 2.5 \times 10^{-6}$) 남는 것은 하나도 없다. 가장 작은 $p$값도 $7.1 \times 10^{-5}$로 문턱값보다 훨씬 크다. 모든 변수가 독립이므로 이것이 옳은 결과이다.

    (c) Bonferroni는 보수적이다. 집단별 오류율(FWER)을 통제하지만 검정력을 떨어뜨린다. 수천 번의 검정 가운데 진짜 상관이 몇 개 있다면 Bonferroni는 그것들을 놓칠 수 있다. 거짓발견율(FDR)을 통제하는 Benjamini--Hochberg 절차가 덜 보수적인 대안이다. $\square$

---

## 정리하며

상관을 다룰 때의 **실무 도구들**을 모았다.

- **세 계수를 함께 본다.** 피어슨·스피어만·켄달이 크게 갈리면 관계가 비선형이거나 이상치가 있다는 신호다.
- **피셔 $z$ 변환으로 신뢰구간을 만든다.** $r$ 의 표본분포가 비대칭이라 직접 구간을 만들 수 없고, $\text{arctanh}(r)$ 이 근사적으로 정규가 된다. **변환 후 구간을 만들고 되돌리는** 것이 표준 절차다.
- **부분상관이 교란을 통제한다.** $Z$ 의 효과를 뺀 뒤 $X$ 와 $Y$ 의 상관을 보며, **$Z$ 를 측정했을 때만 가능하다.**
- **심슨의 역설을 늘 의심한다.** 하위집단으로 나눠 보는 것이 기본 진단이다.
- **다중검정이 허위상관을 만든다.** 변수 $p$ 개의 상관행렬에는 $\binom p2$ 개의 상관이 있고, $p=20$ 이면 190 개다. 보정 없이 유의한 것만 골라 보고하면 **거의 확실히 잡음을 발견한다.** 9장의 문제가 상관 분석에서 나타난 형태다.

다음 절부터 **상관의 유의성 검정**으로 넘어간다.
