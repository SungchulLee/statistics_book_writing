# 정규 누적분포함수와 분위수

## 개요

확률변수 $X$의 **누적분포함수**(CDF)는 $X$가 $x$ 이하의 값을 취할 확률을 준다:

$$
F(x) = P(X \le x) = \int_{-\infty}^{x} f(t)\,dt
$$

정규분포에서는 이 적분에 닫힌 형태의 표현이 없어 수치적으로 계산해야 한다. SciPy는 이를 위해 `stats.norm.cdf()`를 제공한다.

---

## CDF와 PDF를 함께 보기

(이중 y축을 써서) CDF와 PDF를 같은 그림에 그리면 둘의 관계가 분명해진다. 임의의 점에서의 CDF 값은 그 점 왼쪽의 PDF 아래 넓이와 같다.

<div class="codebox" markdown>

### 예제 1. 분포함수와 밀도함수를 두 축에 함께 보기 { .eg }

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

mu, sigma = 1, 2
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 400)

dist = stats.norm(loc=mu, scale=sigma)
y_cdf = dist.cdf(x)
y_pdf = dist.pdf(x)

fig, ax_cdf = plt.subplots(figsize=(12, 3))

# CDF는 왼쪽 축(0~1). PDF는 오른쪽 축(밀도).
# 두 함수의 눈금 규모가 달라 한 축에 그리면 한쪽이 납작해지므로 축을 나눈다.
# 이 절에서는 두 축의 관계가 고정되어 있어(CDF는 PDF의 적분) 안전한 사용이다.
ax_cdf.plot(x, y_cdf, lw=2, label="CDF P(X ≤ x)")
ax_cdf.set_xlabel("x")
ax_cdf.set_ylabel("P(X ≤ x)")
ax_cdf.set_ylim(-0.02, 1.02)

# 기준점 세 개를 표시한다: 평균에서 -1, 0, +1 표준편차.
# CDF 값이 각각 약 0.159, 0.500, 0.841 이 나온다.
# 0.841 - 0.159 = 0.682 가 곧 "68% 규칙"이다.
for xv in [mu - sigma, mu, mu + sigma]:
    yv = dist.cdf(xv)
    ax_cdf.axvline(xv, linestyle='--', color='gray', alpha=0.7)
    ax_cdf.text(xv, yv + 0.05, f"P(X≤{xv:.0f})={yv:.3f}",
                ha='center', fontsize=9)

# 오른쪽 축에 밀도함수
ax_pdf = ax_cdf.twinx()
ax_pdf.plot(x, y_pdf, lw=2, color='tab:red', label="PDF (density)")
ax_pdf.set_ylabel("Density", color='tab:red')

ax_cdf.set_title(f"Normal({mu}, {sigma}) — CDF with PDF Overlay")
plt.tight_layout()
plt.show()
```

![정규 누적분포함수와 분위수](./img/normal_cdf_19.png)

</div>

---

## 표준정규분포의 주요 CDF 값

| $x$ | $\mathcal{N}(x) = P(Z \le x)$ |
|---|---|
| $-1.96$ | $0.025$ |
| $-1$ | $0.159$ |
| $0$ | $0.500$ |
| $1$ | $0.841$ |
| $1.96$ | $0.975$ |

대칭성 $\mathcal{N}(-x) = 1 - \mathcal{N}(x)$ 덕분에 표의 절반만 있으면 된다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
$X \sim N(0, 1)$에 대해 CDF를 사용하여 $P(-1.96 \le X \le 1.96)$을 계산하라.

</div>

??? success "풀이"
    $$
    P(-1.96 \le X \le 1.96) = \mathcal{N}(1.96) - \mathcal{N}(-1.96) = 0.975 - 0.025 = 0.950
    $$

    이것이 95% 신뢰구간의 근거이다. 표준정규분포의 가운데 95%가 $\pm 1.96$ 사이에 놓인다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
표준정규 CDF의 대칭성 $\mathcal{N}(-x) = 1 - \mathcal{N}(x)$를 증명하라.

</div>

??? success "풀이"
    표준정규 PDF는 $\varphi(-t) = \varphi(t)$를 만족한다(0을 중심으로 대칭). 그러면:

    $$
    \mathcal{N}(-x) = \int_{-\infty}^{-x} \varphi(t)\,dt
    $$

    $u = -t$로 치환하면($du = -dt$):

    $$
    \mathcal{N}(-x) = \int_{\infty}^{x} \varphi(-u)(-du) = \int_x^{\infty} \varphi(u)\,du = 1 - \mathcal{N}(x)
    $$

    $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
$X \sim N(5, 9)$일 때 표준화하여 $P(X > 8)$을 구하라.

</div>

??? success "풀이"
    표준화하면 $Z = (X - 5)/3$이다. 그러면:

    $$
    P(X > 8) = P\!\left(Z > \frac{8-5}{3}\right) = P(Z > 1) = 1 - \mathcal{N}(1) \approx 1 - 0.8413 = 0.1587
    $$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
미적분학의 기본정리를 사용하여 $F'(x) = f(x)$(CDF의 도함수가 PDF임)를 보여라. 이것이 그래프에서 무엇을 뜻하는지 설명하라.

</div>

??? success "풀이"
    정의에 의해 $F(x) = \int_{-\infty}^x f(t)\,dt$이다. 미적분학의 기본정리에 의해:

    $$
    F'(x) = \frac{d}{dx}\int_{-\infty}^x f(t)\,dt = f(x)
    $$

    그래프로 보면, 임의의 점 $x$에서 CDF의 기울기가 그 점에서의 PDF 높이와 같다. PDF가 가장 높은 곳(최빈값)에서 CDF가 가장 가파르고, PDF가 0에 가까운 곳(꼬리)에서 CDF는 거의 평평하다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
시험 점수가 $N(70, 8^2)$을 따른다고 하자. 점수가 60점과 85점 사이일 확률을 구하라.

</div>

??? success "풀이"
    두 끝점을 표준화한다.

    $$
    z_1 = \frac{60-70}{8} = -1.25, \qquad z_2 = \frac{85-70}{8} = 1.875
    $$

    따라서

    $$
    P(60 < X < 85) = \mathcal{N}(1.875) - \mathcal{N}(-1.25) = 0.9696 - 0.1056 = 0.8640
    $$

    이다. 약 86.4%가 이 구간에 든다. SciPy로는 `stats.norm(70, 8).cdf(85) - stats.norm(70, 8).cdf(60)`이다.

    구간의 확률은 언제나 **CDF의 뺄셈**이라는 점을 기억한다. 연속분포이므로 등호를 넣든 빼든($60 \le X \le 85$이든 $60 < X < 85$이든) 값은 같다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
표준정규 CDF가 오차함수로

$$
\mathcal{N}(x) = \frac12\left[1 + \operatorname{erf}\!\left(\frac{x}{\sqrt2}\right)\right], \qquad \operatorname{erf}(u) = \frac{2}{\sqrt\pi}\int_0^u e^{-t^2}dt
$$

로 쓰임을 보여라.

</div>

??? success "풀이"
    대칭성에서 $\mathcal{N}(0) = 1/2$이므로

    $$
    \mathcal{N}(x) = \frac12 + \int_0^x \frac{1}{\sqrt{2\pi}}e^{-s^2/2}\,ds
    $$

    이다. $t = s/\sqrt2$로 치환하면 $s = \sqrt2\,t$, $ds = \sqrt2\,dt$이고 적분 상한이 $x/\sqrt2$가 되어

    $$
    \int_0^x \frac{1}{\sqrt{2\pi}}e^{-s^2/2}ds = \frac{\sqrt2}{\sqrt{2\pi}}\int_0^{x/\sqrt2} e^{-t^2}dt = \frac{1}{\sqrt\pi}\int_0^{x/\sqrt2}e^{-t^2}dt = \frac12\operatorname{erf}\!\left(\frac{x}{\sqrt2}\right)
    $$

    를 얻는다. 따라서 $\mathcal{N}(x) = \frac12 + \frac12\operatorname{erf}(x/\sqrt2)$이다. $\square$

    "닫힌 형태가 없다"는 말의 정확한 뜻은 초등함수로 쓸 수 없다는 것이며, $\operatorname{erf}$라는 이름을 붙인 특수함수로는 정확히 쓸 수 있다. `scipy.special.erf`가 이 함수이고, `stats.norm.cdf`는 실제로 이것을 불러 계산한다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
$X$가 연속이고 CDF가 $F$이며 $F$가 순증가할 때, $U = F(X)$가 $\text{Uniform}(0,1)$을 따름을 보여라. 이 사실의 쓸모를 두 가지 들어라.

</div>

??? success "풀이"
    $u \in (0,1)$에 대해 $F$가 순증가하므로 역함수 $F^{-1}$이 존재하고

    $$
    P(U \le u) = P(F(X) \le u) = P(X \le F^{-1}(u)) = F(F^{-1}(u)) = u
    $$

    이다. 이것이 $\text{Uniform}(0,1)$의 CDF이므로 $U \sim \text{Uniform}(0,1)$이다. $\square$

    **쓸모 1 — 난수 생성.** 뒤집어 읽으면 $U \sim \text{Uniform}(0,1)$일 때 $F^{-1}(U) \sim F$이다. 균등난수 하나로 어떤 분포의 난수든 만들 수 있다는 뜻이고, 이것이 역변환 추출법이다.

    **쓸모 2 — 적합도 검정.** 자료가 정말 $F$에서 나왔다면 $F(x_1), \dots, F(x_n)$은 균등분포처럼 보여야 한다. 이 균등성을 얼마나 어기는지 재는 것이 콜모고로프-스미르노프 검정이고, 잔차를 $F$로 통과시켜 균등성을 보는 방법은 시계열·생존분석의 모형 진단에서도 널리 쓰인다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
임의의 확률변수의 CDF가 (1) 비감소, (2) 우연속, (3) $F(-\infty)=0$, $F(\infty)=1$을 만족함을 정의에서 보여라. 또 연속분포에서 $P(X \le a) = P(X < a)$인 이유를 설명하라.

</div>

??? success "풀이"
    **(1) 비감소.** $a < b$이면 $\{X \le a\} \subseteq \{X \le b\}$이므로 확률의 단조성에 의해 $F(a) \le F(b)$이다.

    **(2) 우연속.** $x_n \downarrow x$인 수열에 대해 $\{X \le x_n\} \downarrow \{X \le x\}$이다. 집합열이 감소하며 교집합이 $\{X \le x\}$이므로 확률의 연속성(단조수렴)에 따라 $F(x_n) \to F(x)$이다.

    **(3) 극한.** $\{X \le -n\} \downarrow \emptyset$이고 $\{X \le n\} \uparrow \Omega$이므로 각각 $0$과 $1$로 간다.

    **연속분포에서의 등호.** 일반적으로 $P(X = a) = F(a) - F(a^-)$, 즉 $F$의 $a$에서의 도약 크기이다. $X$가 연속분포를 따르면 $F$가 연속이라 도약이 없으므로 $P(X=a) = 0$이고, 따라서 $P(X \le a) = P(X < a)$이다.

    이산분포에서는 사정이 다르다. 이항분포에서 $P(X \le 3)$과 $P(X < 3)$은 $P(X=3)$만큼 차이가 나며, 이 차이를 놓치는 것이 이산분포 검정에서 흔한 실수다. $\square$

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
기대값을 CDF로 쓴 항등식

$$
E[X] = \int_0^\infty \{1 - F(x)\}\,dx - \int_{-\infty}^0 F(x)\,dx
$$

를 보이고, 이를 $N(\mu, \sigma^2)$에 적용해 $E[X] = \mu$를 얻어라.

</div>

??? success "풀이"
    **항등식.** $X^+ = \max(X, 0)$에 대해 $X^+ = \int_0^\infty \mathbb{1}\{X > x\}\,dx$이다. 기대값을 취하고 토넬리 정리로 순서를 바꾸면

    $$
    E[X^+] = \int_0^\infty P(X > x)\,dx = \int_0^\infty \{1 - F(x)\}\,dx
    $$

    이다. 같은 방법으로 $X^- = \max(-X, 0)$에 대해

    $$
    E[X^-] = \int_0^\infty P(X < -x)\,dx = \int_{-\infty}^0 F(x)\,dx
    $$

    이다(마지막 등식은 $x \mapsto -x$ 치환과, 연속분포에서 $P(X<-x)=F(-x)$임을 쓴 것이다). $X = X^+ - X^-$이므로 두 식을 빼면 항등식을 얻는다. $\square$

    **정규분포에 적용.** 먼저 $\mu = 0$인 경우를 본다. 대칭성에서 $F(-x) = 1 - F(x)$이므로

    $$
    \int_{-\infty}^0 F(x)\,dx = \int_0^\infty F(-x)\,dx = \int_0^\infty \{1-F(x)\}\,dx
    $$

    이고 두 항이 정확히 상쇄되어 $E[X] = 0$이다. 일반적인 $\mu$에 대해서는 $X = \mu + (X - \mu)$이고 $X-\mu$가 평균 0인 정규분포이므로 $E[X] = \mu$이다.

    이 항등식이 유용한 것은 밀도를 몰라도 CDF나 생존함수만으로 기대값을 얻을 수 있기 때문이다. 생존분석에서 카플란-마이어 곡선 아래 넓이를 평균 생존시간으로 읽는 것이 정확히 이 항등식의 응용이다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
관측값 $62, 68, 71, 77, 86$이 $N(70, 8^2)$에서 나왔는지 보려 한다. 경험적 누적분포함수와 이론 CDF의 최대 수직 거리

$$
D = \max_i \max\left\{\frac{i}{n} - F(x_{(i)}),\; F(x_{(i)}) - \frac{i-1}{n}\right\}
$$

를 계산하라.

</div>

??? success "풀이"
    자료가 이미 정렬되어 있다. $n = 5$이고 각 점의 $F$ 값은 다음과 같다.

    | $i$ | $x_{(i)}$ | $F(x_{(i)})$ | $i/n - F$ | $F - (i-1)/n$ |
    |---|---|---|---|---|
    | 1 | 62 | 0.1587 | 0.0413 | 0.1587 |
    | 2 | 68 | 0.4013 | $-0.0013$ | 0.2013 |
    | 3 | 71 | 0.5497 | 0.0503 | 0.1497 |
    | 4 | 77 | 0.8092 | $-0.0092$ | 0.2092 |
    | 5 | 86 | 0.9772 | 0.0228 | 0.1772 |

    표의 모든 값 가운데 최대는 $i=4$ 행의 $0.2092$이므로 $D = 0.209$이다.

    경험적 누적분포함수는 계단함수라 각 관측점에서 위아래 두 값을 갖는다. 그래서 최대 거리를 찾을 때 계단의 위끝 $i/n$과 아래끝 $(i-1)/n$을 모두 확인해야 한다. 한쪽만 보면 $D$를 과소평가한다.

    이 $D$가 콜모고로프-스미르노프 검정통계량이다. `stats.kstest(x, stats.norm(70, 8).cdf)`가 같은 값과 함께 p-값 0.946을 준다. 자료가 다섯 개뿐이라 어지간한 어긋남으로는 정규성을 기각할 수 없다. 다만 이 검정은 모수를 **자료에서 추정하면 쓸 수 없다**. 그때는 릴리포스 검정처럼 보정된 임계값을 써야 한다.

---

## 정리하며

누적분포함수 $F(x)=P(X\le x)$ 는 밀도 아래 **왼쪽 넓이**다.

- **정규분포의 $F$ 에는 닫힌 형태가 없다.** 오차함수로 적거나 수치적으로 계산해야 하며, `stats.norm.cdf()` 가 그 일을 한다. 표준정규표가 존재했던 이유가 이것이다.
- **구간의 확률은 뺄셈이다.** $P(a\le X\le b)=F(b)-F(a)$ 이며, 연속분포라 등호의 포함 여부는 결과를 바꾸지 않는다.
- **$F$ 는 비감소이고 $0$ 에서 $1$ 로 간다.** 밀도와 함께 그려 보면 임의의 점에서 $F$ 의 값이 그 왼쪽 넓이와 같다는 관계가 눈에 들어온다.
- 밀도는 값이 $1$ 을 넘을 수 있지만 **$F$ 는 언제나 $[0,1]$ 안에 있다.** 확률인 쪽은 $F$ 다.

다음 절 **백분위점 함수**는 이 관계를 거꾸로 읽는다. "이 값 이하일 확률은?"이 아니라 **"이 확률을 주는 값은?"** 을 묻는 것이며, 신뢰구간의 임계값이 바로 그 답이다.
