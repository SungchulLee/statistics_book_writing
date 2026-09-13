# scipy.stats로 그리는 정규 밀도함수

## 개요

**정규(Gaussian) 분포**는 통계학에서 가장 중요한 연속분포이다. 확률밀도함수(PDF)는 익숙한 종 모양 곡선으로, 평균 $\mu$를 중심으로 대칭이고 퍼짐은 표준편차 $\sigma$가 결정한다.

$$
f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

확률질량의 약 99.7%가 $\mu \pm 3\sigma$ 안에 있다(68–95–99.7 규칙).

---

## scipy.stats 인터페이스

SciPy는 `stats.norm(loc=mu, scale=sigma)`로 정규분포를 나타내며, 이는 **고정된(frozen)** 분포 객체를 만든다. `loc` 모수가 $\mu$이고 `scale`이 $\sigma$이다($\sigma^2$이 아니다).

<div class="codebox" markdown>

### 예제 1. scipy.stats 로 정규 밀도함수 그리기 { .eg }

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

mu = 1       # mean
sigma = 2    # standard deviation

# 평균에서 좌우 3 표준편차까지. 확률의 99.7%가 이 안에 있다.
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)

# 밀도함수 계산
y = stats.norm(loc=mu, scale=sigma).pdf(x)

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(x, y)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title(f"Normal({mu}, {sigma}²) PDF")
plt.show()
```

![scipy.stats로 그리는 정규 밀도함수](./img/normal_pdf_19.png)

</div>

---

## 주요 성질

| 성질 | 값 |
|---|---|
| 지지집합 | $(-\infty, \infty)$ |
| 평균 | $\mu$ |
| 분산 | $\sigma^2$ |
| 최빈값 | $\mu$ |
| 왜도 | 0 |
| 첨도 (초과) | 0 |

정규분포는 대칭성(왜도 0)과 꼬리 두께(초과첨도 0)의 기준이 된다. 다른 모든 분포는 정규분포와 비교된다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
$X \sim N(1, 4)$에 대해 평균에서의 밀도 $f(1)$을 손으로 계산하라.

</div>

??? success "풀이"
    $\mu = 1$이고 $\sigma^2 = 4$($\sigma = 2$)이므로:

    $$
    f(1) = \frac{1}{2\sqrt{2\pi}} \exp(0) = \frac{1}{2\sqrt{2\pi}} \approx 0.1995
    $$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
정규 PDF의 적분이 1임을 보여라. (힌트: $I = \int_{-\infty}^{\infty} e^{-x^2/2}\,dx$에 대해 $I^2$을 극좌표에서 계산하라.)

</div>

??? success "풀이"
    $I = \int_{-\infty}^{\infty} e^{-x^2/2}\,dx$라 하자. 그러면:

    $$
    I^2 = \int_{-\infty}^{\infty}\int_{-\infty}^{\infty} e^{-(x^2+y^2)/2}\,dx\,dy
    $$

    극좌표 $x = r\cos\theta$, $y = r\sin\theta$로 바꾸면:

    $$
    I^2 = \int_0^{2\pi}\int_0^{\infty} e^{-r^2/2}\,r\,dr\,d\theta = 2\pi \cdot \left[-e^{-r^2/2}\right]_0^{\infty} = 2\pi
    $$

    따라서 $I = \sqrt{2\pi}$이고 $\frac{1}{\sigma\sqrt{2\pi}}\int_{-\infty}^{\infty} e^{-(x-\mu)^2/(2\sigma^2)}\,dx = 1$이다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
정규 PDF의 변곡점을 유도하라. 어떤 $x$ 값에서 곡률의 부호가 바뀌는가?

</div>

??? success "풀이"
    변곡점은 $f''(x) = 0$인 곳에서 생긴다. 계산하면:

    $$
    f'(x) = -\frac{x - \mu}{\sigma^2} f(x)
    $$

    $$
    f''(x) = \left(\frac{(x-\mu)^2}{\sigma^4} - \frac{1}{\sigma^2}\right) f(x)
    $$

    $f(x) > 0$임에 유의하여 $f''(x) = 0$으로 두면:

    $$
    (x - \mu)^2 = \sigma^2 \implies x = \mu \pm \sigma
    $$

    변곡점은 $x = \mu - \sigma$와 $x = \mu + \sigma$에 있으며, 평균에서 정확히 표준편차 하나만큼 떨어진 위치이다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
SciPy를 사용하여 표준정규분포에 대해 68–95–99.7 규칙을 수치적으로 확인하라.

</div>

??? success "풀이"
    ```python
    dist = stats.norm(0, 1)
    for k in [1, 2, 3]:
        prob = dist.cdf(k) - dist.cdf(-k)
        print(f"P(-{k} < Z < {k}) = {prob:.4f}")
    ```

    출력:

    ```
    P(-1 < Z < 1) = 0.6827
    P(-2 < Z < 2) = 0.9545
    P(-3 < Z < 3) = 0.9973
    ```

    즉

    - $P(-1 < Z < 1) = 0.6827$ (약 68%)
    - $P(-2 < Z < 2) = 0.9545$ (약 95%)
    - $P(-3 < Z < 3) = 0.9973$ (약 99.7%)

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
$N(1, 4)$를 만들려고 `stats.norm(loc=1, scale=4)`라고 썼다. 무엇이 잘못되었는가? 실제로 만들어진 분포에서 $f(1)$은 얼마인가?

</div>

??? success "풀이"
    `scale`은 표준편차 $\sigma$를 받는데 분산 $\sigma^2 = 4$를 넣었다. 올바른 코드는 `stats.norm(loc=1, scale=2)`이다.

    실제로 만들어진 것은 $\sigma = 4$, 즉 $N(1, 16)$이다. 그 봉우리 높이는

    $$
    f(1) = \frac{1}{4\sqrt{2\pi}} \approx 0.0997
    $$

    로 의도한 $0.1995$의 절반이다. 퍼짐이 두 배가 되었으니 높이는 절반이 된다.

    이 실수가 특히 위험한 것은 코드가 오류 없이 잘 돌아가고 그림도 그럴듯하게 나온다는 점이다. $N(\mu, \sigma^2)$이라는 수학 표기와 `scale=sigma`라는 코드 사이의 어긋남이 원인이므로, 분산을 다룰 때는 `scale=np.sqrt(var)`라고 명시적으로 적는 습관이 안전하다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
연습문제 1에서 얻은 $f(1) = 0.1995$는 확률이 아니다. 이를 확률로 바꾸려면 어떻게 해야 하는가? $P(0.99 < X < 1.01)$을 근사하고 정확한 값과 견주어라. 또 밀도값이 1을 넘는 예를 하나 들어라.

</div>

??? success "풀이"
    연속확률변수는 한 점을 가질 확률이 0이다. 밀도는 확률이 아니라 **단위 길이당 확률**이므로, 좁은 구간의 확률은 밀도에 구간 길이를 곱해 얻는다.

    $$
    P(0.99 < X < 1.01) \approx f(1) \times 0.02 = 0.19947 \times 0.02 = 0.0039894
    $$

    정확한 값은 `stats.norm(1,2).cdf(1.01) - stats.norm(1,2).cdf(0.99) = 0.0039894`로 소수점 일곱째 자리까지 일치한다. 구간이 짧아 그 안에서 밀도가 거의 상수이기 때문이다.

    밀도가 1을 넘는 예는 $\sigma$를 작게 잡으면 바로 나온다. $\sigma = 0.1$이면

    $$
    f(\mu) = \frac{1}{0.1\sqrt{2\pi}} \approx 3.989
    $$

    이다. 밀도의 **적분**이 1일 뿐 값 자체에는 상한이 없다. 밀도값을 확률로 읽어 "확률이 3.99"라고 말하는 것은 명백한 오류다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
$X \sim N(\mu, \sigma^2)$일 때 변수변환으로 $Z = (X - \mu)/\sigma$의 밀도를 구해 $Z \sim N(0,1)$임을 보여라.

</div>

??? success "풀이"
    $z = (x-\mu)/\sigma$는 $\sigma > 0$에서 단조증가하고 역변환이 $x = \mu + \sigma z$이므로 $dx/dz = \sigma$이다. 변수변환 공식에 따라

    $$
    f_Z(z) = f_X(\mu + \sigma z)\left|\frac{dx}{dz}\right| = \frac{1}{\sigma\sqrt{2\pi}}\exp\!\left(-\frac{(\mu + \sigma z - \mu)^2}{2\sigma^2}\right)\cdot \sigma
    $$

    이다. 지수 안이 $-\sigma^2 z^2/(2\sigma^2) = -z^2/2$로 정리되고 $\sigma$가 약분되어

    $$
    f_Z(z) = \frac{1}{\sqrt{2\pi}}e^{-z^2/2}
    $$

    를 얻는다. 이는 표준정규밀도이다. $\square$

    야코비안의 $\sigma$가 밀도 앞의 $1/\sigma$를 정확히 지운다는 점이 핵심이다. 이 덕분에 정규분포족 전체가 표준정규분포 하나의 위치·척도 변환으로 환원되고, 표 하나로 모든 정규분포를 다룰 수 있었다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
두 정규밀도의 곱 $f(x;\mu_1,\sigma_1^2)\,f(x;\mu_2,\sigma_2^2)$이 $x$의 함수로서 다시 정규밀도에 비례함을 보이고, 그 평균과 분산을 구하라.

</div>

??? success "풀이"
    $x$에 의존하는 부분만 보면 지수의 안이

    $$
    -\frac{(x-\mu_1)^2}{2\sigma_1^2} - \frac{(x-\mu_2)^2}{2\sigma_2^2}
    $$

    이다. $x$에 대한 이차식이므로 완전제곱으로 정리한다. $x^2$의 계수는 $-\frac12(1/\sigma_1^2 + 1/\sigma_2^2)$이고 $x$의 계수는 $\mu_1/\sigma_1^2 + \mu_2/\sigma_2^2$이므로,

    $$
    \frac{1}{\sigma_*^2} = \frac{1}{\sigma_1^2} + \frac{1}{\sigma_2^2}, \qquad \mu_* = \sigma_*^2\left(\frac{\mu_1}{\sigma_1^2} + \frac{\mu_2}{\sigma_2^2}\right)
    $$

    로 두면 지수가 $-(x - \mu_*)^2/(2\sigma_*^2)$ 더하기 $x$와 무관한 상수가 된다. 따라서 곱은 $N(\mu_*, \sigma_*^2)$의 밀도에 비례한다. $\square$

    정밀도(분산의 역수)가 더해지고, 평균은 정밀도를 가중치로 한 가중평균이 된다. 이것이 정규분포가 스스로에 대해 켤레인 이유다. $N(\mu_0,\tau^2)$을 사전분포로, 정규가능도를 관측으로 두면 사후분포가 다시 정규분포이고 그 중심이 사전평균과 표본평균의 정밀도 가중평균이 된다. 칼만 필터의 갱신식도 같은 계산이다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
평균이 $\mu$, 분산이 $\sigma^2$인 $\mathbb{R}$ 위의 모든 연속분포 가운데 미분엔트로피 $h(f) = -\int f \ln f$를 최대로 하는 것이 정규분포임을 보여라.

</div>

??? success "풀이"
    $g$를 평균 $\mu$, 분산 $\sigma^2$을 갖는 임의의 밀도라 하고 $\phi$를 $N(\mu,\sigma^2)$의 밀도라 하자. 쿨백-라이블러 발산은 항상 0 이상이므로

    $$
    0 \le D(g \,\|\, \phi) = \int g \ln \frac{g}{\phi} = -h(g) - \int g \ln \phi
    $$

    이다. 여기서 마지막 항을 계산한다.

    $$
    \ln\phi(x) = -\ln(\sigma\sqrt{2\pi}) - \frac{(x-\mu)^2}{2\sigma^2}
    $$

    이고 $g$가 평균 $\mu$, 분산 $\sigma^2$을 가지므로 $\int g(x)(x-\mu)^2 dx = \sigma^2$이다. 따라서

    $$
    -\int g \ln\phi = \ln(\sigma\sqrt{2\pi}) + \frac{\sigma^2}{2\sigma^2} = \ln(\sigma\sqrt{2\pi}) + \frac12
    $$

    인데, 이 값은 $g$에 의존하지 않으므로 $g = \phi$로 두어도 같다. 즉 이것이 곧 $h(\phi)$이다. 정리하면

    $$
    h(g) \le h(\phi) = \frac12\ln(2\pi e \sigma^2)
    $$

    이고, 등호는 $D(g\|\phi) = 0$, 즉 $g = \phi$일 때만 성립한다. $\square$

    핵심 요령은 $\int g \ln \phi$가 $\phi$의 로그가 이차식이라는 이유만으로 $g$의 처음 두 적률에만 의존한다는 점이다. 제약이 정확히 그 두 적률이므로 이 항이 $g$에 무관해진다.

    뜻은 이렇다. 평균과 분산만 알고 다른 것은 모를 때, 정규분포는 **그 둘 말고는 아무것도 가정하지 않은** 분포다. 최소제곱법이나 정규 오차 가정이 "가장 겸손한 선택"이라 불리는 근거가 여기에 있다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
정규 밀도의 반치전폭(밀도가 최댓값의 절반이 되는 두 점 사이의 거리)을 구하고, 연습문제 3에서 얻은 변곡점 사이의 거리와 견주어라.

</div>

??? success "풀이"
    최댓값은 $f(\mu) = 1/(\sigma\sqrt{2\pi})$이다. $f(x) = f(\mu)/2$가 되는 $x$를 찾으면

    $$
    \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) = \frac12 \implies \frac{(x-\mu)^2}{2\sigma^2} = \ln 2
    $$

    이므로 $x = \mu \pm \sigma\sqrt{2\ln 2}$이다. 두 점 사이의 거리는

    $$
    \text{FWHM} = 2\sigma\sqrt{2\ln 2} \approx 2.355\,\sigma
    $$

    이다.

    변곡점은 $\mu \pm \sigma$에 있으므로 그 사이 거리는 $2\sigma$이다. 반치전폭이 그보다 약 18% 넓다. 두 값 모두 $\sigma$에 정비례하므로 그림만 보고 $\sigma$를 눈대중할 수 있다. 봉우리 높이의 절반이 되는 폭을 재서 2.355로 나누면 된다. 분광학이나 신호처리에서 가우스 봉우리의 폭을 보고할 때 반치전폭을 쓰는 관행이 여기서 왔다.

---

## 정리하며

정규분포의 밀도는 모수 둘로 모양이 전부 정해진다. $\mu$ 가 위치를, $\sigma$ 가 퍼짐을 맡는다.

- **$\mu$ 에 대해 대칭이고 봉우리가 하나다.** 평균·중앙값·최빈값이 모두 $\mu$ 에서 만난다.
- **68–95–99.7 규칙.** 질량의 약 $68\%$ 가 $\mu\pm\sigma$, $95\%$ 가 $\mu\pm2\sigma$, $99.7\%$ 가 $\mu\pm3\sigma$ 안에 있다. 눈대중으로 확률을 가늠할 때 쓰는 기준이다.
- **`scale` 은 $\sigma$ 이지 $\sigma^2$ 이 아니다.** `stats.norm(loc=mu, scale=sigma)` 에 분산을 넣는 것이 가장 흔한 실수다.
- **고정(frozen) 객체로 쓰는 편이 낫다.** `rv = stats.norm(mu, sigma)` 로 한 번 만들어 두면 `rv.pdf`, `rv.cdf`, `rv.ppf` 를 모수 반복 없이 쓸 수 있다.

이어지는 네 절은 같은 분포를 **네 가지 창으로** 들여다본다. 누적분포함수(그 값 이하일 확률), 분위수함수(그 확률을 주는 값), 생존함수(그 값을 넘을 확률), 그리고 난수 생성이다.

다음 절 **정규분포 누적분포함수와 분위수**부터 시작한다.
