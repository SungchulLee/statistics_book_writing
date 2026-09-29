# 2차원 정규 조건부분포

## 개요

이변량 정규분포의 강력한 성질 하나는 **조건부분포도 정규분포**라는 것이다. $(a, b)^\top$가 상관계수 $\rho$인 표준 이변량 정규분포를 따르면:

$$
b \mid a = a_0 \;\sim\; N\!\left(\rho\, a_0,\; 1 - \rho^2\right)
$$

조건부 평균은 $a_0$의 선형함수이고, 조건부 분산 $1 - \rho^2$은 $a_0$에 의존하지 않고 오직 $\rho$에만 의존한다.

---

## 일반적인 경우

평균이 $\mu_a, \mu_b$, 분산이 $\sigma_a^2, \sigma_b^2$, 상관계수가 $\rho$인 이변량 정규분포에 대해:

$$
b \mid a = a_0 \;\sim\; N\!\left(\mu_b + \rho\frac{\sigma_b}{\sigma_a}(a_0 - \mu_a),\; \sigma_b^2(1 - \rho^2)\right)
$$

!!! tip "핵심 통찰"
    조건부 평균은 정확히 $a$에 대한 $b$의 **회귀직선**이다. 조건부 분산은 선형 관계를 반영하고 남은 잔차분산이다.

---

## 코드

<div class="codebox" markdown>

### 예제 1. 이변량 정규분포의 조건부분포 { .eg }

```python
import numpy as np
import matplotlib.pyplot as plt

def bivariate_gaussian_pdf(x, y, rho):
    """표준화된 이변량 정규분포의 밀도. 두 주변분포가 모두 N(0,1)이고
    상관계수만 rho 인 경우다."""
    return (np.exp(-(x**2 - 2*rho*x*y + y**2) / (2*(1 - rho**2)))
            / (2 * np.pi * np.sqrt(1 - rho**2)))


def conditional_pdf(x0, y, rho):
    """X = x0 으로 조건을 걸었을 때 Y의 분포.

    이변량 정규분포의 핵심 성질 두 가지가 여기 들어 있다.
      1. 조건부분포도 **정규분포**다. (다른 분포에서는 일반적으로 성립하지 않는다.)
      2. 조건부 평균은 x0에 **선형**으로 의존한다: mu = rho * x0.
         이것이 선형회귀가 왜 정규분포 가정과 잘 맞는지의 뿌리다.
      3. 조건부 분산 1 - rho^2 은 **x0에 의존하지 않는다.**
         어디를 잘라도 폭이 같다는 뜻이며, 회귀의 등분산 가정에 대응한다.
    """
    sigma_cond = np.sqrt(1 - rho**2)
    mu_cond = rho * x0
    return (1 / (np.sqrt(2*np.pi) * sigma_cond)
            * np.exp(-0.5 * ((y - mu_cond) / sigma_cond)**2))

x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)

rho_vals = [0.0, 0.5, 0.9]
cond_val = 1.0  # condition on a = 1

fig, axes = plt.subplots(len(rho_vals), 2, figsize=(10, 4 * len(rho_vals)))

for i, rho in enumerate(rho_vals):
    Z = bivariate_gaussian_pdf(X, Y, rho)
    mu_cond = rho * cond_val
    sigma_cond = np.sqrt(1 - rho**2)

    # 결합분포 등고선에 자른 면을 표시
    ax = axes[i, 0]
    ax.contourf(X, Y, Z, levels=20, cmap="Blues", alpha=0.4)
    ax.axvline(cond_val, color="red", linestyle="--", lw=2, label=f"a = {cond_val}")
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_title(f"Joint PDF (ρ = {rho})")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    # 조건부 밀도함수
    ax = axes[i, 1]
    cond_y = conditional_pdf(cond_val, y, rho)
    ax.plot(y, cond_y, "b-", lw=2)
    ax.axvline(mu_cond, color="red", linestyle="--", lw=1.5,
               label=f"E[b|a=1] = {mu_cond:.2f}")
    ax.fill_between(y, cond_y, alpha=0.15, color="blue")
    ax.set_xlabel("b")
    ax.set_ylabel("f(b | a = 1)")
    ax.set_title(f"Conditional PDF (σ = {sigma_cond:.3f})")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()
```

![2차원 정규조건부분포](./img/gaussian_2d_conditionals_30.png)

</div>

---

## 해석

| $\rho$ | $E[b \mid a=1]$ | $\text{Var}(b \mid a=1)$ | 효과 |
|---|---|---|---|
| 0 | 0 | 1 | $a$로 조건화해도 $b$에 대한 정보가 없다 |
| 0.5 | 0.5 | 0.75 | 불확실성이 중간 정도로 줄어든다 |
| 0.9 | 0.9 | 0.19 | $a$를 알면 $b$가 거의 결정된다 |

$|\rho| \to 1$일 때 조건부분포는 회귀직선 주위로 모이고 조건부 분산은 0에 가까워진다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
$\rho = 0.8$인 표준 이변량 정규분포에서 $a = 2$가 주어졌을 때 $b$의 조건부 평균과 분산을 계산하라.

</div>

??? success "풀이"
    $$
    E[b \mid a = 2] = \rho \cdot 2 = 0.8 \times 2 = 1.6
    $$

    $$
    \text{Var}(b \mid a = 2) = 1 - \rho^2 = 1 - 0.64 = 0.36
    $$

    따라서 $b \mid a = 2 \sim N(1.6, 0.36)$이고 조건부 표준편차는 $0.6$이다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
표준 이변량 정규분포의 조건부분포 공식 $b \mid a = a_0 \sim N(\rho a_0, 1 - \rho^2)$을 증명하라.

</div>

??? success "풀이"
    결합밀도는:

    $$
    f(a, b) = \frac{1}{2\pi\sqrt{1-\rho^2}}\exp\!\left(-\frac{a^2 - 2\rho ab + b^2}{2(1-\rho^2)}\right)
    $$

    $a$의 주변분포는 $f_a(a) = \frac{1}{\sqrt{2\pi}}e^{-a^2/2}$이다. 따라서 $b$에 대해 완전제곱식을 만들면:

    $$
    f(b \mid a = a_0) = \frac{f(a_0, b)}{f_a(a_0)} \propto \exp\!\left(-\frac{(b - \rho a_0)^2}{2(1-\rho^2)}\right)
    $$

    이는 $N(\rho a_0, 1-\rho^2)$의 핵이다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
조건부 평균 $E[b \mid a] = \rho \cdot a$와 $a$에 대한 $b$의 단순선형회귀 사이의 연결을 설명하라.

</div>

??? success "풀이"
    표준화된 변수(평균 0, 분산 1)에 대해 $a$에 대한 $b$의 단순선형회귀에서 회귀직선은 $\hat{b} = \rho \cdot a$이며 $\rho$는 상관계수이다. 이변량 정규분포에서는 이 회귀직선이 단지 최선의 선형 예측자인 데 그치지 않고 **조건부 기댓값** $E[b \mid a]$ 그 자체이다. 일반적으로 $E[Y \mid X]$는 비선형일 수 있지만, 이변량 정규분포에서는 정확히 선형이다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
$\rho = 0$이면 조건부분포는 무엇이 되는가? 이를 독립성 개념과 연결하라.

</div>

??? success "풀이"
    $\rho = 0$이면 $E[b \mid a = a_0] = 0$이고 $\text{Var}(b \mid a = a_0) = 1$이다. 조건부분포 $b \mid a \sim N(0, 1)$은 $a_0$에 전혀 의존하지 않으며, $b$의 주변분포와 같다.

    이것이 독립성의 정의이다. 모든 $a$에 대해 $f(b \mid a) = f(b)$인 것이다. 이변량 정규분포에서 $\rho = 0$은 독립성의 필요충분조건이다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
성인 남성의 키(cm)와 몸무게(kg)가 이변량 정규분포를 따르고 $\mu = (170, 65)$, $\sigma = (6, 8)$, $\rho = 0.6$이다. 키가 182cm인 사람의 몸무게에 대한 조건부분포와 95% 예측구간을 구하라.

</div>

??? success "풀이"
    일반 공식에 대입한다.

    $$
    E[W \mid H = 182] = 65 + 0.6\times\frac{8}{6}\times(182-170) = 65 + 0.8 \times 12 = 74.6
    $$

    $$
    \operatorname{Var}(W \mid H = 182) = 8^2(1-0.6^2) = 64 \times 0.64 = 40.96
    $$

    이므로 $W \mid H=182 \sim N(74.6,\ 6.4^2)$이다. 95% 예측구간은

    $$
    74.6 \pm 1.96 \times 6.4 = (62.1,\ 87.1)
    $$

    이다.

    세 가지를 확인해 둔다. 첫째, 기울기 $0.8$은 "키가 1cm 크면 평균 몸무게가 0.8kg 늘어난다"는 뜻이며, 이것이 몸무게를 키에 회귀한 최소제곱 기울기와 같다. 둘째, 조건부 표준편차 6.4는 주변 표준편차 8보다 작다. 키를 알면 불확실성이 20% 줄어든다. 셋째, **조건부 표준편차는 182라는 값에 의존하지 않는다.** 키가 155cm인 사람에게도 같은 6.4를 쓴다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
연습문제 5에서 키가 182cm(평균보다 $2\sigma$ 위)인데 예측된 몸무게는 74.6kg으로 평균보다 $1.2\sigma$ 위에 그친다. 이 현상을 설명하고 "평균으로의 회귀"라는 이름의 유래를 밝혀라.

</div>

??? success "풀이"
    표준화해서 보면 분명해진다. 조건부 평균 공식을 $z$ 척도로 쓰면

    $$
    E[z_W \mid z_H] = \rho\, z_H
    $$

    이다. $|\rho| < 1$이므로 **예측된 표준점수는 언제나 주어진 표준점수보다 0에 가깝다.** 키의 $z_H = 2$가 몸무게의 $z_W = 0.6 \times 2 = 1.2$로 줄어든 것이 그것이다.

    직관은 이렇다. 아주 키가 큰 사람의 몸무게에는 키가 주는 몫과 키와 무관한 몫이 함께 들어 있는데, 후자는 평균적으로 0이다. 따라서 극단적인 키에 대응하는 몸무게의 평균은 키만큼 극단적이지 않다.

    **이름의 유래.** 골턴이 부모와 자식의 키를 조사하다 발견했다. 키가 아주 큰 부모의 자식은 평균적으로 부모보다 작고, 아주 작은 부모의 자식은 부모보다 크다. 그는 이를 "평범으로의 회귀(regression towards mediocrity)"라 불렀고, **회귀분석이라는 이름 자체가 여기서 나왔다.** 오늘날 "회귀"라는 말에는 이 원래 뜻이 거의 남아 있지 않다.

    **회귀의 오류.** 이것을 인과로 읽으면 안 된다. 세대가 지날수록 키가 평균으로 수렴하는 것이 아니다. 방향을 뒤집어도 같은 일이 일어난다. 아주 키가 큰 자식의 부모 역시 평균적으로 자식보다 작다. 순전히 통계적 현상이다.

    실무의 함정으로 이어진다. 성적이 가장 나쁜 학급에 특별 프로그램을 넣고 이듬해 성적이 오르면 프로그램 덕분이라고 결론짓기 쉽지만, 아무것도 하지 않아도 평균으로 되돌아왔을 것이다. **극단값을 기준으로 집단을 고른 뒤 효과를 재려면 반드시 대조군이 필요하다.**

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
전체분산 정리 $\operatorname{Var}(b) = E\{\operatorname{Var}(b\mid a)\} + \operatorname{Var}\{E[b\mid a]\}$를 표준 이변량 정규분포에서 확인하고, 각 항이 회귀분석의 무엇에 대응하는지 밝혀라.

</div>

??? success "풀이"
    표준 이변량 정규분포에서 $\operatorname{Var}(b\mid a) = 1-\rho^2$은 상수이므로

    $$
    E\{\operatorname{Var}(b\mid a)\} = 1-\rho^2
    $$

    이고, $E[b\mid a] = \rho a$이며 $\operatorname{Var}(a) = 1$이므로

    $$
    \operatorname{Var}\{E[b\mid a]\} = \rho^2\operatorname{Var}(a) = \rho^2
    $$

    이다. 둘을 더하면

    $$
    (1-\rho^2) + \rho^2 = 1 = \operatorname{Var}(b)
    $$

    로 확인된다. $\square$

    **회귀분석과의 대응.** 이 항등식이 바로 제곱합의 분해다.

    | 전체분산 정리 | 회귀분석 |
    |---|---|
    | $\operatorname{Var}(b)$ | 총제곱합 SST |
    | $\operatorname{Var}\{E[b\mid a]\} = \rho^2$ | 회귀제곱합 SSR (설명된 몫) |
    | $E\{\operatorname{Var}(b\mid a)\} = 1-\rho^2$ | 잔차제곱합 SSE (남은 몫) |

    따라서

    $$
    R^2 = \frac{\text{설명된 분산}}{\text{총분산}} = \frac{\rho^2}{1} = \rho^2
    $$

    이다. **단순회귀에서 $R^2$이 상관계수의 제곱인 이유**가 여기 있다. $\rho = 0.6$이면 $R^2 = 0.36$으로, 몸무게 변동의 36%만 키로 설명되고 64%는 남는다. 상관이 0.6이면 꽤 강해 보이지만 설명력은 절반도 안 된다는 점을 기억할 일이다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
$b$를 $a$에 회귀한 직선과 $a$를 $b$에 회귀한 직선은 서로 다르다. 두 기울기를 구하고 그 곱이 $\rho^2$임을 보여라. 두 직선이 일치하는 경우는 언제인가?

</div>

??? success "풀이"
    조건부 평균 공식에서 $b$를 $a$에 회귀한 기울기는

    $$
    \beta_{b\cdot a} = \rho\frac{\sigma_b}{\sigma_a}
    $$

    이고, 역할을 바꾸면 $a$를 $b$에 회귀한 기울기는

    $$
    \beta_{a\cdot b} = \rho\frac{\sigma_a}{\sigma_b}
    $$

    이다. 곱하면 $\sigma$들이 약분되어

    $$
    \beta_{b\cdot a}\,\beta_{a\cdot b} = \rho^2
    $$

    이다. $\square$

    연습문제 5의 수치로는 $0.8 \times 0.45 = 0.36 = 0.6^2$이다.

    **왜 다른가.** $(a,b)$ 평면에서 보면 첫 직선은 세로 방향 거리(즉 $b$의 오차)를 최소로 하고, 둘째 직선은 가로 방향 거리를 최소로 한다. 무엇을 오차로 보느냐가 다르니 답도 다르다. 두 직선은 점 $(\mu_a, \mu_b)$에서 만나고, $a$-$b$ 평면에서 둘째 직선을 $b$에 대해 풀면 기울기가 $\sigma_b/(\rho\sigma_a)$로 첫 직선보다 가파르다. 즉 두 직선이 타원의 장축을 사이에 두고 벌어져 있다.

    **일치하는 경우**는 $\beta_{b\cdot a} = 1/\beta_{a\cdot b}$일 때이므로 $\rho^2 = 1$, 즉 $|\rho| = 1$뿐이다. 완전상관이면 모든 점이 한 직선 위에 있어 어느 방향으로 재든 오차가 0이다. 반대로 $\rho \to 0$이면 두 직선이 각각 수평선과 수직선으로 가며 직각으로 벌어진다.

    실무의 교훈은 분명하다. **"$x$와 $y$의 회귀직선"이라는 말은 어느 쪽을 반응변수로 두었는지 밝히지 않으면 뜻이 없다.** 둘을 대칭적으로 다루고 싶다면 회귀가 아니라 주성분분석의 첫 주축(직교 거리를 최소로 하는 직선)을 써야 한다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
다변량 정규분포 $\mathbf{X} = (\mathbf{X}_1, \mathbf{X}_2)$에서 조건부분포가

$$
\mathbf{X}_2 \mid \mathbf{X}_1 = \mathbf{x}_1 \sim N\!\left(\boldsymbol\mu_2 + \Sigma_{21}\Sigma_{11}^{-1}(\mathbf{x}_1-\boldsymbol\mu_1),\ \Sigma_{22} - \Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12}\right)
$$

임을 이변량 공식과 견주어 확인하고, 조건부 공분산행렬의 구조를 설명하라.

</div>

??? success "풀이"
    **이변량으로 특수화.** $\mathbf{X}_1 = a$, $\mathbf{X}_2 = b$인 1차원끼리라면 $\Sigma_{11} = \sigma_a^2$, $\Sigma_{21} = \rho\sigma_a\sigma_b$, $\Sigma_{22} = \sigma_b^2$이다. 대입하면

    $$
    \text{평균} = \mu_b + \rho\sigma_a\sigma_b\cdot\frac{1}{\sigma_a^2}(a_0-\mu_a) = \mu_b + \rho\frac{\sigma_b}{\sigma_a}(a_0-\mu_a)
    $$

    $$
    \text{분산} = \sigma_b^2 - \rho\sigma_a\sigma_b\cdot\frac{1}{\sigma_a^2}\cdot\rho\sigma_a\sigma_b = \sigma_b^2(1-\rho^2)
    $$

    으로 본문 공식과 정확히 일치한다.

    **조건부 공분산의 구조.** $\Sigma_{22\cdot1} = \Sigma_{22} - \Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12}$는 블록행렬 $\Sigma$에서 $\Sigma_{11}$에 대한 **슈어 보수**다. 읽는 법은 이렇다.

    - $\Sigma_{22}$는 $\mathbf{X}_2$의 원래 공분산.
    - $\Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12}$는 $\mathbf{X}_1$이 설명해 주는 몫. $\Sigma_{21}\Sigma_{11}^{-1}$이 회귀계수 행렬이므로, 이 항은 "회귀로 설명된 공분산"이다.
    - 그 차이가 남은 공분산이며, **언제나 양반정부호**다. 조건을 걸면 불확실성이 늘어날 수 없다는 뜻이다.

    그리고 $\mathbf{x}_1$에 전혀 의존하지 않는다. 정규분포의 매우 특별한 성질이며, 여기서 등분산성이 따라 나온다.

    **부분상관.** $\Sigma_{22\cdot1}$을 상관행렬로 바꾼 것이 $\mathbf{X}_1$을 통제한 **부분상관**이다. 즉 "다른 변수의 영향을 걷어 낸 뒤의 상관"이 조건부 공분산행렬에서 직접 읽힌다. 또 알려진 항등식으로 $\Sigma^{-1}$의 $(2,2)$ 블록이 $\Sigma_{22\cdot1}^{-1}$과 같은데, 그래서 **정밀도행렬의 비대각 성분이 0이라는 것이 나머지 변수를 모두 통제했을 때의 조건부 독립**과 같아진다. 가우스 그래프모형의 출발점이다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
연습문제 5의 상황에서 $P(W > 80)$과 $P(W > 80 \mid H = 182)$를 각각 구하고, 두 값이 크게 다른 것이 실무에서 어떤 오해를 낳는지 설명하라.

</div>

??? success "풀이"
    **주변확률.** $W \sim N(65, 8^2)$이므로

    $$
    P(W > 80) = P\!\left(Z > \frac{80-65}{8}\right) = P(Z > 1.875) = 0.0304
    $$

    **조건부확률.** $W \mid H=182 \sim N(74.6,\ 6.4^2)$이므로

    $$
    P(W > 80 \mid H = 182) = P\!\left(Z > \frac{80-74.6}{6.4}\right) = P(Z > 0.844) = 0.199
    $$

    3%에서 20%로 여섯 배 넘게 뛴다. 키를 알았다는 정보가 중심을 옮기고(65 → 74.6) 동시에 폭을 좁혔다(8 → 6.4).

    **낳는 오해.** 두 확률을 섞어 쓰는 것이 실무에서 가장 흔한 오류 중 하나다.

    - **기저율 무시.** "이 사람은 키가 182cm이니 몸무게가 80kg을 넘을 확률이 20%"는 맞지만, 여기서 "전체 인구의 20%가 80kg을 넘는다"로 넘어가면 틀린다. 조건이 걸린 집단은 전체의 일부일 뿐이다.
    - **조건의 방향 혼동.** $P(W>80\mid H=182)$와 $P(H=182\mid W>80)$은 전혀 다른 양이다. 의료검사에서 민감도 $P(\text{양성}\mid\text{질병})$와 양성예측도 $P(\text{질병}\mid\text{양성})$을 혼동하는 것이 같은 오류이며, 유병률이 낮으면 두 값이 극적으로 갈린다.
    - **예측구간을 신뢰구간으로 읽기.** 연습문제 5의 $(62.1, 87.1)$은 키가 182cm인 **한 사람**의 몸무게 구간이지, 그 집단의 평균 몸무게 구간이 아니다. 후자는 훨씬 좁다.

    조건부확률을 쓸 때는 언제나 "무엇을 알고 있다고 가정하는가"를 명시하는 습관이 안전하다.

---

## 정리하며

이변량 정규분포의 조건부분포도 **정규분포**이며, 이 사실 하나에서 회귀가 통째로 나온다.

$$
b \mid a = a_0 \;\sim\; N\!\left(\mu_b + \rho\frac{\sigma_b}{\sigma_a}(a_0 - \mu_a),\; \sigma_b^2(1-\rho^2)\right)
$$

- **조건부 평균이 곧 회귀직선이다.** 기울기 $\rho\,\sigma_b/\sigma_a$ 는 최소제곱 기울기와 정확히 같다. 3장에서 $\mathbb{E}[Y\mid X]$ 가 최적 예측임을 보았는데, 정규분포에서는 그 최적 예측이 마침 **선형**이다.
- **조건부 분산 $\sigma_b^2(1-\rho^2)$ 은 $a_0$ 에 의존하지 않는다.** 회귀분석이 가정하는 **등분산성**이 여기서는 결과로 따라 나온다.
- **$1-\rho^2$ 이 설명되지 않고 남은 비율**이다. 회귀의 $R^2=\rho^2$ 라는 익숙한 관계가 그대로 읽힌다.
- **조건을 걸면 분산이 줄어든다.** $|\rho|$ 가 클수록 많이 줄고, $\rho=0$ 이면 전혀 줄지 않는다. 정보를 얻는다는 것이 곧 불확실성이 준다는 뜻임을 정량적으로 보여 준다.

다음 절 **2차원 정규분포 고유분해**는 같은 분포를 좌표 없이 본다. 타원의 축이 어디를 향하고 얼마나 긴지를 공분산행렬의 고유벡터와 고윳값이 직접 알려 준다.
