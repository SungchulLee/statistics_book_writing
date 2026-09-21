# 로지스틱분포

## 개요

위치 $\mu$와 척도 $s$를 갖는 **로지스틱분포**의 PDF는 다음과 같다:

$$
f(x) = \frac{e^{-(x-\mu)/s}}{s\left(1 + e^{-(x-\mu)/s}\right)^2}
$$

| 성질 | 값 |
|---|---|
| 평균 | $\mu$ |
| 분산 | $s^2\pi^2/3$ |
| 지지집합 | $(-\infty, \infty)$ |

로지스틱분포는 정규분포와 비슷하지만 **꼬리가 더 두꺼워** 금융 수익률처럼 꼬리가 두꺼운 현상을 모형화하는 데 유용하다. CDF가 $F(x) = 1/(1 + e^{-(x-\mu)/s})$라는 편리한 닫힌 형태를 가지며, 이것이 기계학습 전반에서 쓰이는 로지스틱(시그모이드) 함수이다.

---

## 정규분포와의 비교

두 분포를 공정하게 비교하려면 분산을 맞춘다. 로지스틱분포의 척도가 $s$이면 분산은 $s^2\pi^2/3$이다. 이에 맞추는 정규분포는 $\sigma = s\pi/\sqrt{3}$이다.

<div class="codebox" markdown>

### 예제 1. 분산을 맞춘 로지스틱과 정규분포 비교 { .eg }

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

mu, s = 0, 1        # 위치모수와 척도모수. s는 표준편차가 아니다.
x = np.linspace(-8, 8, 400)

rv_logistic = stats.logistic(loc=mu, scale=s)

# 두 분포를 공정하게 비교하려면 **분산을 맞춰야** 한다.
# 로지스틱 분포의 분산은 s^2 * pi^2 / 3 이므로 표준편차는 s*pi/sqrt(3) ≈ 1.81s.
# 이 값을 정규분포의 sigma로 주면 두 곡선이 같은 퍼짐을 갖는다.
# 그래야 남는 차이가 오직 "꼬리의 두께"가 된다.
sigma = s * np.pi / np.sqrt(3)
rv_normal = stats.norm(loc=mu, scale=sigma)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(x, rv_logistic.pdf(x), label='Logistic(0, 1)')
ax.plot(x, rv_normal.pdf(x), '--', label=rf'Normal(0, {sigma:.2f}²) [same var]')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Logistic vs Normal Distribution — PDF')
ax.legend()
plt.tight_layout()
plt.show()
```

![Logistic vs Normal Distribution — PDF](./img/logistic_pdf_25.png)

분산을 맞췄는데도 두 곡선이 겹치지 않는다. 로지스틱 곡선이 중앙에서 더 높고($f(0) = 0.25$ 대 $0.220$), 그 바깥의 어깨 부분($|x|$가 대략 1.2에서 4.3 사이)에서는 더 낮으며, 다시 먼 꼬리에서 더 높다. 세 번 교차하는 이 모양이 첨도가 3보다 큰 분포의 전형이다. 같은 분산을 중앙과 먼 꼬리에 몰아 주고 중간을 비운 것이다.

</div>

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
로지스틱 CDF $F(x) = 1/(1 + e^{-(x-\mu)/s})$가 로지스틱 PDF의 부정적분임을 보여라.

</div>

??? success "풀이"
    $F(x) = (1 + e^{-(x-\mu)/s})^{-1}$을 미분하면:

    $$
    F'(x) = \frac{e^{-(x-\mu)/s}/s}{(1 + e^{-(x-\mu)/s})^2} = f(x)
    $$

    따라서 $F'(x) = f(x)$가 확인된다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
로지스틱 분산 $s^2\pi^2/3$을 계산하고, 같은 척도 모수 $\sigma = s$를 갖는 정규분포의 분산보다 큼을 확인하라.

</div>

??? success "풀이"
    척도가 $s$인 로지스틱분포의 분산은 $s^2\pi^2/3 \approx 3.29 s^2$이다. $\sigma = s$인 정규분포의 분산은 $s^2$이다. 따라서 로지스틱 분산이 $\pi^2/3 \approx 3.29$배 크며, 이는 꼬리가 더 두껍다는 사실과 일관된다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
로지스틱분포는 로지스틱 회귀에서 핵심적인 역할을 한다. CDF $F(x) = 1/(1 + e^{-x})$가 선형 예측자를 확률로 옮기는 연결함수 역할을 어떻게 하는지 설명하라.

</div>

??? success "풀이"
    로지스틱 회귀 모형은 $P(Y = 1 \mid \mathbf{x}) = \sigma(\mathbf{x}^\top\boldsymbol{\beta})$로 설정되며, $\sigma(z) = 1/(1+e^{-z})$가 로지스틱 CDF이다. 선형 예측자 $z = \mathbf{x}^\top\boldsymbol{\beta}$는 $(-\infty, \infty)$의 어떤 값이든 취할 수 있고, 로지스틱 함수가 이를 $(0, 1)$로 옮겨 유효한 확률을 만들어 준다. 그 역함수 $z = \ln(p/(1-p))$(로그 오즈 또는 로짓)가 확률 척도를 선형 척도로 이어 준다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
표준 로지스틱분포($\mu=0, s=1$)와 분산을 맞춘 정규분포에서 꼬리 확률 $P(|X| > 3)$을 비교하라. 어느 쪽의 꼬리 확률이 더 큰가?

</div>

??? success "풀이"
    **Logistic:** $P(|X| > 3) = 2 \cdot S(3) = 2/(1 + e^3) \approx 2 \times 0.0474 = 0.0949$.

    **분산을 맞춘 정규분포** ($\sigma = \pi/\sqrt{3} \approx 1.814$): $P(|X| > 3) = 2\mathcal{N}(-3/1.814) = 2\mathcal{N}(-1.654) \approx 2 \times 0.0491 = 0.0982$.

    이 문턱값에서는 꼬리 확률이 비슷하지만, 더 극단적인 값(예: $|X| > 5$)에서는 로지스틱 꼬리가 우세해진다. 로지스틱은 지수적으로($\sim e^{-|x|/s}$) 감소하는 반면 정규분포는 가우스 형태로($\sim e^{-x^2/(2\sigma^2)}$) 감소하기 때문이다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
표준 로지스틱분포 밀도의 최댓값을 구하고, 분산을 맞춘 정규분포의 최댓값과 견주어라. 어느 쪽이 더 뾰족한가?

</div>

??? success "풀이"
    로지스틱 밀도는 대칭이므로 $x = \mu = 0$에서 최대이고

    $$
    f(0) = \frac{e^{0}}{1 \cdot (1 + e^{0})^2} = \frac{1}{4} = 0.25
    $$

    이다. 분산을 맞춘 정규분포는 $\sigma = \pi/\sqrt{3} \approx 1.8138$이므로

    $$
    \phi(0) = \frac{1}{\sigma\sqrt{2\pi}} = \frac{1}{1.8138 \times 2.5066} \approx 0.220
    $$

    이다.

    분산이 같은데도 로지스틱 쪽이 중앙에서 더 높다. 로지스틱분포의 첨도는 4.2로 정규분포의 3보다 크며, 이런 분포는 "봉우리가 뾰족하고 꼬리가 두꺼운" 대신 중간 어깨가 얇다. 꼬리만 두껍고 봉우리는 낮을 것이라고 짐작하기 쉬운데, 분산이 고정되어 있으면 그럴 수 없다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
로지스틱분포의 분위수함수를 유도하고, 이를 이용해 $U \sim \text{Uniform}(0,1)$에서 로지스틱 난수를 만드는 방법을 적어라.

</div>

??? success "풀이"
    $u = 1/(1 + e^{-(x-\mu)/s})$를 $x$에 대해 풀면 $1/u = 1 + e^{-(x-\mu)/s}$이므로

    $$
    e^{-(x-\mu)/s} = \frac{1-u}{u} \implies x = \mu + s \ln\!\frac{u}{1-u}
    $$

    이다. 즉 $F^{-1}(u) = \mu + s\,\operatorname{logit}(u)$이다. 분위수함수가 바로 로짓 함수이다.

    따라서 $U \sim \text{Uniform}(0,1)$에 대해 $X = \mu + s \ln\{U/(1-U)\}$가 로지스틱분포를 따른다. 정규분포는 CDF도 그 역함수도 닫힌 형태가 없어 박스-뮐러 변환이나 수치적 역함수가 필요한데, 로지스틱은 로그 한 번으로 끝난다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
시그모이드 $\sigma(z) = 1/(1+e^{-z})$에 대해 $\sigma'(z) = \sigma(z)\{1 - \sigma(z)\}$임을 보이고, 이것이 로지스틱 회귀의 최대가능도 추정에서 어떤 이점을 주는지 설명하라.

</div>

??? success "풀이"
    $\sigma(z) = (1 + e^{-z})^{-1}$을 미분하면

    $$
    \sigma'(z) = \frac{e^{-z}}{(1+e^{-z})^2} = \frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}} = \sigma(z)\{1 - \sigma(z)\}
    $$

    이다. 도함수가 함숫값만으로 표현된다.

    로지스틱 회귀의 로그가능도는 $p_i = \sigma(\mathbf{x}_i^\top\boldsymbol\beta)$에 대해

    $$
    \ell(\boldsymbol\beta) = \sum_i \left\{ y_i \ln p_i + (1-y_i)\ln(1-p_i) \right\}
    $$

    이고, 연쇄법칙에서 나오는 $\sigma'$가 $\ln p_i$의 미분에서 나오는 $1/p_i$와 약분된다. 그 결과 점수함수가

    $$
    \frac{\partial \ell}{\partial \boldsymbol\beta} = \sum_i (y_i - p_i)\,\mathbf{x}_i
    $$

    라는 대단히 간결한 꼴이 된다. "관측값 빼기 적합값"에 설명변수를 가중한 형태다. 헤시안도 $-\sum_i p_i(1-p_i)\mathbf{x}_i\mathbf{x}_i^\top$로 항상 음반정부호이므로 로그가능도가 오목하고, 뉴턴-랩슨(IRLS)이 안정적으로 수렴한다. $\square$

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
로지스틱 회귀에서 어떤 설명변수의 계수가 $\hat\beta = 0.7$로 추정되었다. 이 값을 오즈의 말로 해석하라. 또 확률의 변화량으로 해석하면 왜 한 가지 숫자로 말할 수 없는가?

</div>

??? success "풀이"
    모형은 $\ln\{p/(1-p)\} = \beta_0 + \beta_1 x + \cdots$이므로 $x$가 1 늘면 로그 오즈가 $0.7$ 늘고, 오즈는

    $$
    e^{0.7} \approx 2.01
    $$

    배가 된다. "다른 변수를 고정했을 때 $x$가 한 단위 오르면 사건이 일어날 오즈가 약 두 배가 된다"가 정확한 해석이다.

    확률의 변화량으로 옮기면 사정이 달라진다. $\partial p/\partial x = \beta_1 p(1-p)$이므로 그 크기가 현재 확률에 달려 있다. $p = 0.5$ 근처에서는 $0.7 \times 0.25 = 0.175$로 크지만, $p = 0.05$ 근처에서는 $0.7 \times 0.0475 \approx 0.033$에 지나지 않는다. 로지스틱 회귀의 계수가 상수인 척도는 확률이 아니라 로그 오즈이며, 확률 척도에서의 효과는 어디서 재느냐에 따라 달라진다. 그래서 실무에서는 평균한계효과처럼 관측값 전체에서 평균낸 값을 따로 보고한다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
$G_1, G_2$가 독립이고 각각 표준 굼벨분포 $P(G \le g) = \exp(-e^{-g})$를 따를 때, $G_1 - G_2$가 표준 로지스틱분포를 따름을 보여라. 이 사실이 로지스틱 회귀에 어떤 근거를 주는가?

</div>

??? success "풀이"
    $G_2$로 조건을 걸고 적분한다. $g_2$의 밀도는 $e^{-g_2}\exp(-e^{-g_2})$이므로 $t = G_1 - G_2$에 대해

    $$
    P(G_1 - G_2 \le x) = \int_{-\infty}^{\infty} \exp\!\left(-e^{-(x + g_2)}\right) e^{-g_2}\exp\!\left(-e^{-g_2}\right) dg_2
    $$

    이다. $u = e^{-g_2}$로 치환하면 $du = -u\,dg_2$이고 적분 구간이 $(0, \infty)$로 바뀌어

    $$
    = \int_0^\infty \exp\!\left(-u e^{-x}\right)\exp(-u)\,du = \int_0^\infty e^{-u(1 + e^{-x})}\,du = \frac{1}{1 + e^{-x}}
    $$

    를 얻는다. 이것이 표준 로지스틱 CDF이다. $\square$

    **뜻.** 확률효용 모형에서 선택지 $j$의 효용을 $U_j = V_j + \varepsilon_j$로 두고 오차 $\varepsilon_j$가 독립인 굼벨분포를 따른다고 하면,

    $$
    P(\text{1번을 고름}) = P(U_1 > U_2) = P(\varepsilon_2 - \varepsilon_1 < V_1 - V_2) = \sigma(V_1 - V_2)
    $$

    가 된다. 로지스틱 회귀가 그저 편리한 곡선이 아니라 **효용을 최대로 하는 선택**이라는 행동 모형에서 유도된다는 뜻이다. 굼벨분포가 등장하는 이유는 그것이 최댓값의 극한분포이기 때문이고, 이 구조를 여러 선택지로 늘린 것이 다항 로짓 모형이다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
같은 자료에 로짓 모형과 프로빗 모형을 적합하면 로짓 계수가 프로빗 계수의 대략 1.6~1.8배로 나온다. 이유를 설명하라.

</div>

??? success "풀이"
    두 모형은 같은 잠재변수 구조 $Y = 1 \iff \mathbf{x}^\top\boldsymbol\beta^* + \varepsilon > 0$에서 오차의 분포만 다르게 둔 것이다. 로짓은 $\varepsilon \sim$ 표준 로지스틱, 프로빗은 $\varepsilon \sim N(0,1)$이다.

    잠재변수 모형에서 계수는 오차의 척도로 나눈 값으로만 식별된다. 즉 추정되는 것은 $\boldsymbol\beta^*/\sigma_\varepsilon$이다. 표준 로지스틱의 표준편차는

    $$
    \sqrt{\pi^2/3} = \frac{\pi}{\sqrt 3} \approx 1.814
    $$

    이고 표준정규의 표준편차는 1이다. 로짓 쪽이 오차를 1.814배 크게 잡아 놓았으므로, 같은 자료를 설명하려면 계수도 그만큼 크게 나와야 한다. 따라서

    $$
    \hat\beta_{\text{로짓}} \approx 1.814 \times \hat\beta_{\text{프로빗}}
    $$

    이 된다. 실제로는 두 분포의 모양이 꼬리에서 달라 완전히 비례하지는 않으며, 경험적으로 1.6~1.8 사이의 값이 관측된다. 중요한 것은 **두 모형의 계수를 직접 비교하면 안 된다**는 점이다. 비교하려면 오즈비나 한계효과, 예측확률처럼 척도에 의존하지 않는 양으로 옮겨야 한다.

---

## 정리하며

로지스틱분포는 정규분포와 모양이 닮았지만 **꼬리가 더 두껍고, 누적분포함수가 닫힌 형태**다.

- **평균 $\mu$, 분산 $s^2\pi^2/3$.** 정규분포와 공정하게 비교하려면 분산을 맞춰야 하며, 그때 $\sigma=s\pi/\sqrt3$ 이다.
- **$F(x)=1/(1+e^{-(x-\mu)/s})$ 가 곧 시그모이드 함수다.** 기계학습에서 로짓을 확률로 바꿀 때 쓰는 그 함수이며, 로지스틱 회귀의 이름이 여기서 왔다.
- **정규분포의 누적분포함수에는 닫힌 형태가 없다.** 이 한 가지 차이가 로지스틱분포를 계산에 편리하게 만들고, 프로빗 대신 로짓이 널리 쓰이는 실무적 이유가 된다.
- **꼬리가 두꺼워** 금융 수익률처럼 극단값이 정규분포보다 자주 나타나는 자료에 쓰인다. 다만 지수적으로 감소하므로 파레토류의 두꺼운 꼬리와는 다르다.

다음 절 **와이불분포**로 넘어간다. 지금까지의 분포들이 값의 분포를 그렸다면, 와이불은 **시간에 따라 위험이 어떻게 변하는가**를 모형화한다.
