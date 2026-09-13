# 균등 밀도함수

## 개요

구간 $[a, b]$ 위의 **연속 균등분포**는 구간의 모든 점에 동일한 확률밀도를 부여한다:

$$
f(x) = \frac{1}{b - a}, \qquad a \le x \le b
$$

| 성질 | 값 |
|---|---|
| 지지집합 | $[a, b]$ |
| 평균 | $(a + b)/2$ |
| 분산 | $(b - a)^2/12$ |
| CDF | $x \in [a, b]$에서 $(x - a)/(b - a)$ |

균등분포는 유계 구간 위의 최대 엔트로피 분포이다. 값이 어디에 놓일지에 대해 가장 적은 가정을 한다.

---

## SciPy 모수화

SciPy는 `stats.uniform(loc=a, scale=b-a)`를 사용하며, `loc`은 왼쪽 끝점이고 `scale`은 구간의 폭이다.

<div class="codebox" markdown>

### 예제 1. 구간에 따른 균등 밀도함수 { .eg }

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# 구간 [a, b]를 셋 준비한다. 폭이 1, 4, 4로 다르다.
intervals = [(0, 1), (-2, 2), (1, 5)]
x = np.linspace(-3, 6, 500)

fig, ax = plt.subplots(figsize=(12, 4))
for a, b in intervals:
    # scipy의 균등분포 매개변수화에 주의하라.
    # loc = 시작점 a, scale = **폭** (b가 아니라 b - a) 이다.
    # stats.uniform(1, 5) 는 [1, 5]가 아니라 [1, 6]을 뜻한다.
    rv = stats.uniform(loc=a, scale=b - a)
    # 밀도는 구간 안에서 1/(b-a)로 일정하고 밖에서는 0이다.
    # 폭이 좁을수록 높이가 높아진다. 전체 넓이가 언제나 1이어야 하기 때문이다.
    ax.plot(x, rv.pdf(x), label=f'Uniform({a}, {b})')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Uniform Distribution — PDF')
ax.legend()
ax.set_ylim(bottom=-0.05)
plt.tight_layout()
plt.show()
```

![Uniform Distribution — PDF](./img/uniform_pdf_26.png)

전체 넓이가 1이어야 하므로 구간이 넓어질수록 직사각형은 (더 넓어지는 대신) 더 낮아진다.

</div>

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
직접 적분하여 $X \sim \text{Uniform}(a, b)$의 평균과 분산을 유도하라.

</div>

??? success "풀이"
    **평균:**

    $$
    E[X] = \int_a^b \frac{x}{b-a}\,dx = \frac{1}{b-a}\cdot\frac{b^2 - a^2}{2} = \frac{a+b}{2}
    $$

    **2차 적률:**

    $$
    E[X^2] = \int_a^b \frac{x^2}{b-a}\,dx = \frac{b^3 - a^3}{3(b-a)} = \frac{a^2 + ab + b^2}{3}
    $$

    **분산:**

    $$
    \text{Var}(X) = E[X^2] - (E[X])^2 = \frac{a^2+ab+b^2}{3} - \frac{(a+b)^2}{4} = \frac{(b-a)^2}{12}
    $$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
$U \sim \text{Uniform}(0,1)$이면 $X = a + (b-a)U \sim \text{Uniform}(a,b)$임을 보여라.

</div>

??? success "풀이"
    $a \le x \le b$에서 $X$의 CDF는:

    $$
    P(X \le x) = P(a + (b-a)U \le x) = P\!\left(U \le \frac{x-a}{b-a}\right) = \frac{x-a}{b-a}
    $$

    이는 $\text{Uniform}(a, b)$의 CDF이다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
균등분포는 $[a, b]$ 위의 최대 엔트로피 분포이다. 이것이 무슨 뜻인지 밝히고 직관적으로 왜 타당한지 설명하라.

</div>

??? success "풀이"
    $[a, b]$를 지지집합으로 하는 모든 연속분포 중에서 균등분포가 미분 엔트로피 $h(X) = \ln(b - a)$를 최대로 한다. 최대 엔트로피란 최대 불확실성을 뜻한다. 값의 범위는 알지만 값이 어디에 놓일지에 대해서는 그 외에 아무것도 모르는 상태이다. 직관적으로, 균등하지 않은 밀도라면 어떤 부분 영역에 확률을 몰아 준다는 뜻이고, 이는 지지집합 외에 분포에 대해 더 많이 안다는 것을 함의한다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
$X \sim \text{Uniform}(0, 1)$일 때 $Y = -\ln(X)$의 분포를 구하고 어떤 분포인지 밝혀라.

</div>

??? success "풀이"
    $y > 0$에 대해:

    $$
    P(Y \le y) = P(-\ln X \le y) = P(X \ge e^{-y}) = 1 - e^{-y}
    $$

    이는 $\text{Exponential}(\lambda = 1)$의 CDF이다. 따라서 $Y = -\ln(U) \sim \text{Exp}(1)$이며, 이것이 지수분포에 대한 역변환 표본추출의 근거이다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
`stats.uniform(1, 5)`가 나타내는 분포의 구간, 평균, 분산을 구하라. $[1, 5]$ 위의 균등분포를 만들려면 어떻게 써야 하는가?

</div>

??? success "풀이"
    `loc = 1`, `scale = 5`이므로 구간은 $[1, 1+5] = [1, 6]$이다. 오른쪽 끝점이 5가 아니다.

    $$
    E[X] = \frac{1+6}{2} = 3.5, \qquad \operatorname{Var}(X) = \frac{(6-1)^2}{12} = \frac{25}{12} \approx 2.083
    $$

    $[1, 5]$를 원하면 폭이 $5 - 1 = 4$이므로 `stats.uniform(loc=1, scale=4)`라고 써야 한다. 그때 평균은 3, 분산은 $16/12 \approx 1.333$이다.

    SciPy의 모든 분포가 `loc`과 `scale`이라는 같은 이름으로 위치와 척도를 받기 때문에 생긴 일이다. 일관성을 지키려다 균등분포에서만 직관과 어긋나게 되었다. 코드에 `stats.uniform(loc=a, scale=b-a)`라고 뺄셈을 드러내 적으면 실수를 줄일 수 있다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
$U_1, \dots, U_n$이 독립이고 $\text{Uniform}(0,1)$을 따를 때 최댓값 $M = \max_i U_i$의 CDF와 밀도를 구하고 $E[M]$을 계산하라.

</div>

??? success "풀이"
    최댓값이 $m$ 이하이려면 모두가 $m$ 이하여야 한다. 독립이므로

    $$
    F_M(m) = P(U_1 \le m, \dots, U_n \le m) = m^n, \qquad 0 \le m \le 1
    $$

    이고 미분하면

    $$
    f_M(m) = n m^{n-1}
    $$

    이다. 이는 $\text{Beta}(n, 1)$의 밀도이다. 기대값은

    $$
    E[M] = \int_0^1 m \cdot n m^{n-1}\,dm = \frac{n}{n+1}
    $$

    이다.

    같은 방법으로 최솟값은 $P(\min > m) = (1-m)^n$에서 $\text{Beta}(1, n)$을 따르고 $E[\min] = 1/(n+1)$이다. 두 값이 $1/(n+1)$과 $n/(n+1)$로 대칭인 것이 자연스럽다. 일반적으로 $k$번째로 작은 값은 $\text{Beta}(k, n-k+1)$을 따르며 기대값이 $k/(n+1)$이다. $n$개의 균등난수가 구간 $[0,1]$을 $n+1$개의 평균적으로 같은 조각으로 나눈다는 그림이다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
$U_1, U_2$가 독립이고 $\text{Uniform}(0,1)$을 따를 때 $S = U_1 + U_2$의 밀도를 구하라.

</div>

??? success "풀이"
    합성곱 공식을 쓴다.

    $$
    f_S(s) = \int_{-\infty}^{\infty} f(u)\,f(s-u)\,du
    $$

    피적분함수가 0이 아니려면 $0 \le u \le 1$이면서 $0 \le s - u \le 1$, 즉 $s-1 \le u \le s$여야 한다.

    $0 \le s \le 1$이면 겹치는 구간이 $[0, s]$이므로 $f_S(s) = s$이고, $1 < s \le 2$이면 겹치는 구간이 $[s-1, 1]$이므로 $f_S(s) = 2 - s$이다. 정리하면

    $$
    f_S(s) = \begin{cases} s, & 0 \le s \le 1 \\ 2-s, & 1 < s \le 2 \\ 0, & \text{그 밖} \end{cases}
    $$

    으로 밑변 2, 높이 1인 이등변삼각형이다. 그래서 **삼각분포**라 부른다.

    평평하던 밀도 둘을 더했을 뿐인데 봉우리가 생겼다. 합이 1 근처가 되는 조합은 많고 0이나 2가 되는 조합은 드물기 때문이다. 셋을 더하면 이차식 세 조각으로 이루어진 더 매끄러운 종 모양이 되고, 계속 더하면 중심극한정리에 따라 정규분포로 다가간다. 실제로 균등난수 12개를 더하고 6을 빼면 평균 0, 분산 1인 정규분포의 꽤 좋은 근사가 되어, 옛 컴퓨터에서 정규난수를 만드는 데 쓰였다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
$X_1, \dots, X_n$이 $\text{Uniform}(0, \theta)$에서 나왔다. $\theta$의 최대가능도추정량을 구하고 그 편향을 계산한 뒤, 불편추정량으로 고쳐라.

</div>

??? success "풀이"
    가능도는 모든 $x_i$가 $[0,\theta]$ 안에 있을 때만 0이 아니고, 그때

    $$
    L(\theta) = \frac{1}{\theta^n}, \qquad \theta \ge \max_i x_i
    $$

    이다. $\theta$에 대해 감소하므로 허용되는 가장 작은 $\theta$에서 최대가 된다. 따라서

    $$
    \hat\theta_{\text{MLE}} = \max_i X_i
    $$

    이다. 미분해서 0으로 두는 방법이 통하지 않는 대표적인 예다. 최대가 지지집합의 경계에서 일어나기 때문이다.

    **편향.** 연습문제 6에서 $\max_i U_i$의 기대값이 $n/(n+1)$이었고 $X_i = \theta U_i$이므로

    $$
    E[\hat\theta_{\text{MLE}}] = \frac{n}{n+1}\theta, \qquad \text{편향} = -\frac{\theta}{n+1}
    $$

    이다. 항상 과소추정한다. 표본의 최댓값이 모집단의 최댓값을 넘을 수 없으니 당연한 일이다.

    **보정.**

    $$
    \tilde\theta = \frac{n+1}{n}\max_i X_i
    $$

    로 두면 불편이 된다. 이 추정량이 표본평균을 두 배 한 $2\bar X$보다 훨씬 낫다. $\operatorname{Var}(\tilde\theta) = \theta^2/\{n(n+2)\}$로 $n^{-2}$의 속도로 줄어드는 반면 $2\bar X$의 분산은 $\theta^2/(3n)$으로 $n^{-1}$로만 줄어든다. 보통의 $\sqrt n$ 속도보다 빠른 이 현상은 지지집합이 모수에 의존하는 비정칙 모형의 특징이며, 크라메르-라오 하한이 적용되지 않는 경우다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
연습문제 3의 주장을 엄밀히 증명하라. 즉 지지집합이 $[a,b]$인 임의의 밀도 $g$에 대해 $h(g) \le \ln(b-a)$이고 등호는 $g$가 균등밀도일 때만 성립함을 보여라.

</div>

??? success "풀이"
    $u(x) = 1/(b-a)$를 균등밀도라 하자. 쿨백-라이블러 발산은 항상 0 이상이므로

    $$
    0 \le D(g \,\|\, u) = \int_a^b g(x)\ln\frac{g(x)}{u(x)}\,dx = -h(g) - \int_a^b g(x)\ln u(x)\,dx
    $$

    이다. $\ln u(x) = -\ln(b-a)$는 상수이고 $\int_a^b g = 1$이므로

    $$
    -\int_a^b g(x)\ln u(x)\,dx = \ln(b-a)
    $$

    이다. 따라서 $0 \le -h(g) + \ln(b-a)$, 즉

    $$
    h(g) \le \ln(b-a) = h(u)
    $$

    이다. 등호는 $D(g\|u) = 0$일 때만 성립하고, 이는 (거의 어디서나) $g = u$와 동치이다. $\square$

    **왜 통하는가.** $\ln u$가 상수라는 점이 전부다. 그 덕분에 $\int g \ln u$가 $g$에 전혀 의존하지 않게 되고, 부등식 하나로 끝난다. 정규분포의 최대엔트로피성을 증명할 때는 $\ln\varphi$가 이차식이라 $\int g\ln\varphi$가 $g$의 처음 두 적률에만 의존했고, 그 둘이 제약으로 고정되어 있어 같은 논법이 통했다.

    이것이 일반적인 원리다. 제약이 $\int g\,T_j = c_j$ 꼴이면 최대엔트로피 분포는 $\exp(\sum\lambda_j T_j)$ 꼴의 지수족이 되며, 제약이 지지집합뿐이면 $T$가 없어 상수 밀도, 곧 균등분포가 된다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
측정값을 0.1 단위로 반올림해 기록했다. 반올림 오차의 분포와 그 분산을 구하고, 이것이 표본분산에 미치는 영향을 논하라.

</div>

??? success "풀이"
    참값을 $x$, 기록값을 $\tilde x$라 하면 오차 $e = \tilde x - x$는 $[-0.05, 0.05]$ 안에 있다. 참값이 눈금에 비해 매끄럽게 퍼져 있다면 오차는 그 구간 위의 균등분포로 볼 수 있다. 따라서

    $$
    E[e] = 0, \qquad \operatorname{Var}(e) = \frac{(0.1)^2}{12} = \frac{0.01}{12} \approx 0.000833
    $$

    이다. 눈금 폭을 $w$라 할 때 $w^2/12$이며, 이 값을 **양자화 잡음**이라 부른다.

    반올림 오차가 참값과 대략 독립이라고 보면 기록값의 분산이

    $$
    \operatorname{Var}(\tilde X) \approx \operatorname{Var}(X) + \frac{w^2}{12}
    $$

    로 부풀려진다. 이를 되돌리는 것이 **셰퍼드 보정** $s^2_{\text{보정}} = s^2 - w^2/12$이다.

    실제로 문제가 되는지는 비율에 달려 있다. 자료의 표준편차가 $s = 2$라면 $s^2 = 4$에 견주어 $0.00083$은 0.02%에 지나지 않아 무시해도 좋다. 그러나 $s = 0.15$처럼 눈금과 비슷한 규모라면 $s^2 = 0.0225$의 3.7%가 되어 무시할 수 없다. **눈금이 산포에 비해 성길 때만 보정을 생각하면 된다**는 것이 기준이고, 대략 $w < s/2$이면 안전하다.

    반대 방향의 활용도 있다. 디지털 신호처리에서는 이 잡음을 일부러 더한다. 반올림 오차가 신호와 상관될 때 생기는 체계적 왜곡을 없애려고 미세한 무작위 잡음(디더)을 섞어 오차를 진짜 균등분포로 만드는 기법이다.

---

## 정리하며

균등분포는 가장 단순한 연속분포이며, 단순하다는 것이 곧 쓸모다.

- **밀도가 상수** $1/(b-a)$ 이므로 확률이 곧 길이의 비다. 평균은 $(a+b)/2$, 분산은 $(b-a)^2/12$ 이며 **위치는 중점이, 퍼짐은 폭만이 결정한다.**
- **유계 구간 위의 최대 엔트로피 분포다.** 값이 어디 놓일지에 대해 가장 적은 가정을 하므로, 아는 것이 범위뿐일 때의 기본 선택이 된다.
- **SciPy 모수화에 주의하라.** `stats.uniform(loc=a, scale=b-a)` 에서 둘째 인자는 오른쪽 끝점이 아니라 **폭**이다. `uniform(0, 5)` 는 $[0,5]$ 가 맞지만 `uniform(1, 5)` 는 $[1,5]$ 가 아니라 $[1,6]$ 이다.
- $U(0,1)$ 은 모든 난수 생성의 출발점이다. 뒤의 **역변환 표본추출**에서 $F^{-1}(U)$ 로 임의의 분포를 만들어 내는 데 쓰인다.

다음 절 **지수분포**로 넘어간다. 균등분포가 "아무 데나 고르게"라면 지수분포는 "사건이 일어날 때까지 기다리는 시간"이며, 무기억성이라는 특이한 성질을 갖는다.
