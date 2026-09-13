# 로그정규 밀도함수

## 개요

$X \sim N(\mu, \sigma^2)$이면 $Y = e^X$는 모수 $\mu$와 $\sigma$를 갖는 **Log-정규분포**를 따른다. 동등하게, $\ln Y \sim N(\mu, \sigma^2)$이다.

$$
f(y) = \frac{1}{y\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(\ln y - \mu)^2}{2\sigma^2}\right), \qquad y > 0
$$

| 성질 | 값 |
|---|---|
| 지지집합 | $(0, \infty)$ |
| 평균 | $e^{\mu + \sigma^2/2}$ |
| 중앙값 | $e^{\mu}$ |
| 분산 | $(e^{\sigma^2} - 1)\,e^{2\mu + \sigma^2}$ |
| 최빈값 | $e^{\mu - \sigma^2}$ |

Log-정규분포는 자산 가격, 소득처럼 반드시 양수이고 오른쪽으로 치우친 양을 모형화하는 데 널리 쓰인다.

---

## SciPy 모수화

SciPy는 `stats.lognorm(s=sigma, scale=np.exp(mu))`를 사용하며, `s`가 형상 모수 $\sigma$이고 `scale`이 중앙값 $e^\mu$이다.

<div class="codebox" markdown>

### 예제 1. 로그정규분포의 SciPy 모수화 { .eg }

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# 로그정규분포: log(X)가 N(mu, sigma^2)을 따르는 분포다.
# mu와 sigma는 **로그를 취한 뒤의** 평균과 표준편차이지 X 자체의 것이 아니다.
mu = 0
sigmas = [0.5, 1.0, 1.5, 2.0]
x = np.linspace(0.001, 8, 500)     # X > 0 이므로 0에서 시작한다

fig, ax = plt.subplots(figsize=(12, 4))
for sigma in sigmas:
    # scipy의 매개변수화가 특히 헷갈리는 분포다.
    #   s     = 로그 척도의 표준편차 sigma
    #   scale = exp(mu)
    # 즉 loc가 아니라 scale에 exp(mu)를 넣어야 한다.
    rv = stats.lognorm(s=sigma, scale=np.exp(mu))
    # sigma가 커질수록 봉우리가 0쪽으로 밀리고 오른쪽 꼬리가 길어진다.
    ax.plot(x, rv.pdf(x), label=rf'$\sigma={sigma}$')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title(r'Log-Normal Distribution — PDF ($\mu=0$, varying $\sigma$)')
ax.legend()
ax.set_ylim(bottom=-0.02)
plt.tight_layout()
plt.show()
```

![Log-Normal Distribution — PDF ($\mu=0$, varying $\sigma$)](./img/lognormal_pdf_27.png)

$\sigma$가 커질수록 분포가 오른쪽으로 더 치우치고 최빈값은 0 쪽으로 이동한다.

</div>

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
$X \sim N(\mu, \sigma^2)$일 때 정규분포의 적률생성함수를 사용하여 $Y = e^X$의 $E[Y]$를 유도하라.

</div>

??? success "풀이"
    $X \sim N(\mu, \sigma^2)$의 MGF는 $M_X(t) = E[e^{tX}] = e^{\mu t + \sigma^2 t^2/2}$이다.

    $t = 1$로 두면:

    $$
    E[Y] = E[e^X] = M_X(1) = e^{\mu + \sigma^2/2}
    $$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
Log-정규분포의 중앙값이 $e^\mu$이며, 임의의 $\sigma > 0$에 대해 평균 $e^{\mu + \sigma^2/2}$보다 작음을 보여라.

</div>

??? success "풀이"
    중앙값 $m$은 $P(Y \le m) = 0.5$를 만족한다:

    $$
    P(e^X \le m) = P(X \le \ln m) = \mathcal{N}\!\left(\frac{\ln m - \mu}{\sigma}\right) = 0.5
    $$

    이려면 $(\ln m - \mu)/\sigma = 0$이어야 하므로 $m = e^\mu$이다.

    $\sigma^2/2 > 0$이므로 $e^{\mu + \sigma^2/2} > e^\mu$이며, 평균 > 중앙값임이 확인된다. 이는 오른쪽으로 치우침을 반영한다. 두꺼운 오른쪽 꼬리가 평균을 위로 끌어올린다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
주식 수익률이 (연율화된) $\mu = 0.05$, $\sigma = 0.2$인 Log-정규분포를 따른다면, 주가가 가치의 20% 넘게 하락할 확률은 얼마인가?

</div>

??? success "풀이"
    20% 손실은 $Y < 0.8$을 뜻한다(주가가 초기 가치의 80%보다 낮다):

    $$
    P(Y < 0.8) = P(X < \ln 0.8) = \mathcal{N}\!\left(\frac{\ln 0.8 - 0.05}{0.2}\right) = \mathcal{N}\!\left(\frac{-0.2231 - 0.05}{0.2}\right) = \mathcal{N}(-1.366) \approx 0.086
    $$

    20% 넘게 손실을 볼 확률은 약 8.6%이다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
독립인 Log-정규 확률변수들의 곱이 다시 Log-정규임을 증명하라.

</div>

??? success "풀이"
    $X_1 \sim N(\mu_1, \sigma_1^2)$과 $X_2 \sim N(\mu_2, \sigma_2^2)$이 독립일 때 $Y_1 = e^{X_1}$, $Y_2 = e^{X_2}$라 하자. 그러면:

    $$
    Y_1 Y_2 = e^{X_1 + X_2}
    $$

    독립인 정규확률변수의 합에 의해 $X_1 + X_2 \sim N(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$이므로, $Y_1 Y_2$는 모수가 $\mu_1 + \mu_2$와 $\sqrt{\sigma_1^2 + \sigma_2^2}$인 Log-정규분포이다.

    귀납법에 의해 독립인 Log-정규 확률변수들의 임의의 유한 곱은 Log-정규이다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
밀도함수를 미분해 로그정규분포의 최빈값이 $e^{\mu - \sigma^2}$임을 보여라. $\sigma$가 커지면 최빈값이 어디로 가는가?

</div>

??? success "풀이"
    로그를 취해 미분하는 편이 쉽다.

    $$
    \ln f(y) = -\ln y - \ln(\sigma\sqrt{2\pi}) - \frac{(\ln y - \mu)^2}{2\sigma^2}
    $$

    $y$로 미분하면

    $$
    \frac{d}{dy}\ln f(y) = -\frac{1}{y} - \frac{\ln y - \mu}{\sigma^2 y} = -\frac{1}{y}\left(1 + \frac{\ln y - \mu}{\sigma^2}\right)
    $$

    이다. $y > 0$이므로 괄호가 0이 될 때 극값을 가지며, 그때 $\ln y = \mu - \sigma^2$, 즉 $y = e^{\mu - \sigma^2}$이다. 괄호는 $y$의 증가함수이므로 도함수의 부호가 $+$에서 $-$로 바뀌어 이 점이 최대이다.

    $\sigma$가 커지면 $e^{\mu-\sigma^2}$는 0으로 빠르게 줄어든다. 중앙값 $e^\mu$는 그대로인데 봉우리는 0 쪽으로 밀리고 평균 $e^{\mu+\sigma^2/2}$는 오른쪽으로 달아난다. 세 중심이 벌어지는 이 현상이 예제 1의 그림에 그대로 나타나 있다. $\square$

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
$\operatorname{Var}(Y) = (e^{\sigma^2} - 1)e^{2\mu + \sigma^2}$임을 보이고, 변동계수가 $\sqrt{e^{\sigma^2}-1}$로 $\mu$와 무관함을 확인하라.

</div>

??? success "풀이"
    연습문제 1과 같이 정규분포의 적률생성함수 $M_X(t) = e^{\mu t + \sigma^2 t^2/2}$를 쓴다. $t = 2$에서

    $$
    E[Y^2] = E[e^{2X}] = M_X(2) = e^{2\mu + 2\sigma^2}
    $$

    이다. 따라서

    $$
    \operatorname{Var}(Y) = e^{2\mu + 2\sigma^2} - \left(e^{\mu + \sigma^2/2}\right)^2 = e^{2\mu+2\sigma^2} - e^{2\mu+\sigma^2} = e^{2\mu+\sigma^2}\left(e^{\sigma^2} - 1\right)
    $$

    이다.

    변동계수는 표준편차를 평균으로 나눈 값이므로

    $$
    \text{CV} = \frac{\sqrt{e^{2\mu+\sigma^2}(e^{\sigma^2}-1)}}{e^{\mu+\sigma^2/2}} = \sqrt{e^{\sigma^2}-1}
    $$

    가 되어 $\mu$가 사라진다. $\mu$는 척도 모수일 뿐이고 분포의 모양은 오직 $\sigma$가 정한다는 뜻이다. 소득이든 주가든 단위를 바꿔도 상대적 산포는 그대로라는 성질이 여기서 나온다. $\square$

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
평균이 10이고 표준편차가 4인 로그정규분포를 SciPy로 만들려 한다. `stats.lognorm`의 `s`와 `scale`에 각각 무엇을 넣어야 하는가? `stats.lognorm(s=4, scale=np.exp(10))`으로 쓰면 무슨 일이 일어나는가?

</div>

??? success "풀이"
    적률을 맞춰야 한다. $\text{CV} = 4/10 = 0.4$이므로 연습문제 6에서

    $$
    \sigma^2 = \ln\left(1 + \text{CV}^2\right) = \ln(1.16) \approx 0.14842, \qquad \sigma \approx 0.3853
    $$

    이고, 평균 조건 $e^{\mu + \sigma^2/2} = 10$에서

    $$
    \mu = \ln 10 - \frac{\sigma^2}{2} \approx 2.3026 - 0.0742 = 2.2284
    $$

    이다. 따라서 `stats.lognorm(s=0.3853, scale=np.exp(2.2284))`, 즉 `scale ≈ 9.2848`이다. 이 `scale`은 평균 10이 아니라 **중앙값** 9.285임에 주의한다.

    `stats.lognorm(s=4, scale=np.exp(10))`은 $\sigma = 4$, $\mu = 10$인 분포를 만든다. 그 평균은

    $$
    e^{10 + 16/2} = e^{18} \approx 6.6 \times 10^{7}
    $$

    로 의도한 10과 천만 배 넘게 어긋난다. `s`가 표준편차가 아니라 **로그 척도의** 표준편차이고, `scale`이 평균이 아니라 $e^\mu$라는 두 가지를 한꺼번에 틀린 결과다. 로그정규분포에서 가장 흔한 실수다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
$Y_1, \dots, Y_n$이 독립이고 같은 로그정규분포를 따를 때, 표본기하평균 $G_n = \left(\prod_i Y_i\right)^{1/n}$의 분포를 구하고 $n \to \infty$일 때의 극한을 말하라. 표본산술평균의 극한과 견주어라.

</div>

??? success "풀이"
    로그를 취하면

    $$
    \ln G_n = \frac{1}{n}\sum_{i=1}^n \ln Y_i
    $$

    이고 $\ln Y_i \sim N(\mu, \sigma^2)$이므로 $\ln G_n \sim N(\mu, \sigma^2/n)$이다. 지수를 되돌리면 $G_n$은 모수가 $\mu$와 $\sigma/\sqrt n$인 로그정규분포를 따른다.

    $n \to \infty$이면 $\sigma/\sqrt n \to 0$이므로 $G_n \xrightarrow{p} e^\mu$, 즉 **중앙값**으로 수렴한다. 반면 표본산술평균은 큰수의 법칙에 따라 **평균** $e^{\mu + \sigma^2/2}$로 수렴한다.

    두 평균이 서로 다른 곳으로 간다. 기하평균은 로그 척도에서 중심을 재므로 치우친 분포의 "전형적인 값"을 주고, 산술평균은 오른쪽 꼬리의 몇몇 큰 값에 끌려 올라간다. 소득 통계에서 중위소득과 평균소득이 크게 다른 까닭이 이것이며, 어느 쪽을 보고할지는 무엇을 알고 싶은가에 달려 있다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
$Z_1, Z_2, \dots$가 독립이고 같은 분포를 따르는 양의 확률변수이며 $E[\ln Z_i] = m$, $\operatorname{Var}(\ln Z_i) = v < \infty$라 하자. $P_n = \prod_{i=1}^n Z_i$의 극한 성질을 중심극한정리로 기술하고, 로그정규분포가 왜 그토록 자주 나타나는지 설명하라.

</div>

??? success "풀이"
    로그를 취하면 곱이 합이 된다.

    $$
    \ln P_n = \sum_{i=1}^n \ln Z_i
    $$

    $\ln Z_i$가 독립이고 같은 분포를 따르며 분산이 유한하므로 중심극한정리에 따라

    $$
    \frac{\ln P_n - nm}{\sqrt{nv}} \xrightarrow{d} N(0,1)
    $$

    이다. 지수를 되돌리면 $P_n$이 근사적으로 모수 $nm$과 $nv$인 로그정규분포를 따른다.

    핵심은 **$Z_i$의 분포가 무엇이든 상관없다**는 것이다. 유한한 로그분산만 있으면 된다. 중심극한정리가 "덧셈적으로 쌓이는 무작위 요인"을 정규분포로 몰아가듯, 그 로그판인 이 결과는 "곱셈적으로 쌓이는 무작위 요인"을 로그정규분포로 몰아간다.

    현실에서 많은 양이 곱셈적으로 자란다. 자산 가격은 일별 수익률 $(1+r_i)$의 곱이고, 생물의 크기는 성장률의 곱이며, 소득은 여러 배율 요인의 누적이다. 그래서 이 양들이 로그정규분포에 가까워진다.

    다만 주의할 점이 있다. 이 근사는 중앙 부근에서만 좋다. 꼬리에서는 수렴이 훨씬 느리고, 실제 자료의 극단값은 로그정규분포가 예측하는 것보다 자주 나타나는 경우가 많다. 금융에서 로그정규 모형이 폭락을 과소평가한다는 비판이 여기서 나온다. $\square$

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
반응변수에 로그를 씌워 $\ln Y = \mathbf{x}^\top\boldsymbol\beta + \varepsilon$, $\varepsilon \sim N(0, \sigma^2)$을 적합한 뒤 예측값에 지수를 취해 $\hat Y = e^{\mathbf{x}^\top\hat{\boldsymbol\beta}}$로 보고했다. 이 값이 무엇의 추정치인지 밝히고, $E[Y \mid \mathbf{x}]$를 원한다면 어떻게 고쳐야 하는지 적어라.

</div>

??? success "풀이"
    $\ln Y \mid \mathbf{x} \sim N(\mathbf{x}^\top\boldsymbol\beta, \sigma^2)$이므로 $Y \mid \mathbf{x}$는 로그정규분포를 따른다. 연습문제 2에 따라 그 중앙값이 $e^{\mathbf{x}^\top\boldsymbol\beta}$이다. 즉 지수를 그냥 되돌린 값은 **조건부 중앙값**의 추정치이지 조건부 평균이 아니다.

    조건부 평균은

    $$
    E[Y \mid \mathbf{x}] = e^{\mathbf{x}^\top\boldsymbol\beta + \sigma^2/2} = e^{\mathbf{x}^\top\boldsymbol\beta} \cdot e^{\sigma^2/2}
    $$

    이므로 $e^{\hat\sigma^2/2}$를 곱해 주어야 한다. $\hat\sigma^2$은 잔차 평균제곱이다. 예를 들어 $\hat\sigma^2 = 0.5$이면 보정계수가 $e^{0.25} \approx 1.284$로, 28%를 그냥 잃고 있었던 셈이다. 아무리 표본이 커져도 사라지지 않는 체계적 과소예측이다.

    다만 이 보정은 오차가 정규분포라는 가정에 기대고 있다. 그 가정이 미덥지 않으면 잔차의 경험분포를 그대로 쓰는 **두안(Duan)의 스미어링 추정량**

    $$
    \hat E[Y \mid \mathbf{x}] = e^{\mathbf{x}^\top\hat{\boldsymbol\beta}} \cdot \frac{1}{n}\sum_{i=1}^n e^{\hat\varepsilon_i}
    $$

    을 쓴다. 정규성이 성립하면 $\frac1n\sum e^{\hat\varepsilon_i} \approx e^{\hat\sigma^2/2}$이 되어 두 방법이 일치한다.

    가장 좋은 길은 아예 로그를 씌우지 않는 것이다. 로그연결함수를 쓰는 감마 일반화선형모형이나 포아송 유사가능도를 쓰면 평균을 직접 모형화하므로 역변환 문제가 생기지 않는다. $\square$

---

## 정리하며

로그정규분포는 **로그를 취하면 정규가 되는** 분포다. 덧셈적으로 누적되면 정규, **곱셈적으로 누적되면 로그정규**다.

- **세 중심이 모두 다르다.** 최빈값 $e^{\mu-\sigma^2}$ < 중앙값 $e^{\mu}$ < 평균 $e^{\mu+\sigma^2/2}$. 순서가 이렇게 고정되어 있고, $\sigma$ 가 클수록 벌어진다. **오른쪽 치우침의 교과서적 예다.**
- **중앙값 $e^\mu$ 만이 로그 척도에서의 평균에 대응한다.** 로그 자료의 평균을 되돌린 값은 평균이 아니라 중앙값이라는 점이 실무에서 자주 혼동된다.
- **지지집합이 $(0,\infty)$** 이라 자산 가격·소득·체류 시간처럼 반드시 양수인 양에 맞는다.
- **적률은 모두 유한하지만 적률생성함수가 존재하지 않는다.** 그래서 적률이 분포를 결정하지 못하며, 모든 적률이 같은 다른 분포가 존재한다(3장 적률생성함수 문서 연습문제 $6$).
- **SciPy 모수화가 까다롭다.** `stats.lognorm(s=sigma, scale=np.exp(mu))` 이며, `scale` 에 들어가는 것은 평균이 아니라 **중앙값** $e^\mu$ 다.

다음 절 **로지스틱분포**로 넘어간다. 정규분포와 모양은 비슷하되 꼬리가 두껍고, 무엇보다 누적분포함수가 닫힌 형태를 갖는다.
