# 일치성과 점근정규성

## 개요

자료를 더 모을수록 추정값이 좋아지기를 기대한다. 그런데 어떤 추정 절차가 표본크기가 커질 때 실제로 참 모수값으로 수렴하는가? 일치성은 이 보장을 형식화한다. 일치추정량은 소표본에서 어떻게 거동하든 결국 참값 주위로 모여든다.

추정량 $\hat{\theta}_n$이 $\theta$로 확률수렴하면 $\theta$에 대해 **일치**한다고 한다:

$$
\hat{\theta}_n \xrightarrow{P} \theta \quad \text{as } n \to \infty
$$

이는 모든 $\varepsilon > 0$에 대해 $n \to \infty$일 때 $P(|\hat{\theta}_n - \theta| > \varepsilon) \to 0$임을 뜻한다.

!!! note "약한 일치성과 강한 일치성"

    위의 정의를 **약한 일치성**이라 부르기도 한다. 더 강한 개념인 **강한 일치성**은 거의 확실한 수렴 $P(\hat{\theta}_n \to \theta) = 1$을 요구한다. 강한 일치성은 약한 일치성을 함의하지만 그 역은 성립하지 않는다. 실무에서 흔히 쓰는 많은 추정량은 둘 다 만족한다.

## 일치성의 충분조건

정의에서 곧바로 일치성을 확인하려면 모든 $n$에 대해 $\hat{\theta}_n$의 분포 전체를 분석해야 해서 어려울 수 있다. 더 간단한 방법은 평균제곱오차를 쓰는 것이다. $\operatorname{Bias}(\hat{\theta}_n) = E[\hat{\theta}_n] - \theta$이고 $\operatorname{Var}(\hat{\theta}_n) = E\bigl[(\hat{\theta}_n - E[\hat{\theta}_n])^2\bigr]$임을 떠올리자.

일치성의 **충분조건**(필요조건은 아니다)은 $n \to \infty$일 때 편향과 분산이 모두 사라지는 것이다:

$$
\operatorname{Bias}(\hat{\theta}_n) \to 0 \quad \text{and} \quad \operatorname{Var}(\hat{\theta}_n) \to 0 \quad \text{as } n \to \infty
$$

$\operatorname{MSE}(\hat{\theta}_n) = \operatorname{Bias}^2(\hat{\theta}_n) + \operatorname{Var}(\hat{\theta}_n)$이므로 두 조건이 함께 성립하면 $\operatorname{MSE} \to 0$이 되고, 이는 다시 확률수렴을 함의하기 때문이다.

!!! example "표본평균은 모평균에 대해 일치한다"

    $X_1, \ldots, X_n$을 평균이 $\mu$이고 분산이 유한한 $\sigma^2$인 i.i.d. 확률변수라 하자. 표본평균 $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$는 $\operatorname{Bias}(\bar{X}_n) = 0$이고 $\operatorname{Var}(\bar{X}_n) = \sigma^2 / n \to 0$을 만족한다. 두 조건이 모두 성립하므로 $\bar{X}_n$은 $\mu$에 대해 일치한다.

## 점근정규성

일치성은 $\hat{\theta}_n$이 $\theta$로 수렴한다고 알려 주지만, 얼마나 빨리 집중되는지 또는 $n$이 클 때 어떤 분포를 따르는지는 말해 주지 않는다. 점근정규성은 이 두 물음에 답하며 대표본 추론의 토대를 제공한다.

추정량 $\hat{\theta}_n$이 다음을 만족하면 **점근적으로 정규**라고 한다:

$$
\sqrt{n}\,(\hat{\theta}_n - \theta) \xrightarrow{d} N(0,\, \sigma^2)
$$

여기서 $\sigma^2 > 0$은 추정량과 밑바탕 분포에 의존한다. 이 $\sigma^2$을 **점근분산**이라 한다. 표준적인 추정량 다수에서 $\sigma^2 = 1/I(\theta)$이며, $I(\theta)$는 관측값 하나당 Fisher 정보량이다.

이 결과가 대표본에서 정규분포에 기반한 신뢰구간을 사용하는 것을 정당화한다. 구체적으로 근사적인 $(1 - \alpha)$ 수준 신뢰구간은 다음 형태를 취한다:

$$
\hat{\theta}_n \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}
$$

여기서 $z_{\alpha/2}$는 표준정규 임계값이고 $\sigma / \sqrt{n}$은 점근 표준오차이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
$\text{Var}(X) = \sigma^2 < \infty$를 가정하고 Chebyshev 부등식을 사용하여 표본평균 $\bar{X}_n$이 $\mu = E[X]$의 일치추정량임을 증명하라.

</div>

??? success "풀이"
    Chebyshev 부등식에 의해 임의의 $\varepsilon > 0$에 대해:

    $$
    P(|\bar{X}_n - \mu| \geq \varepsilon) \leq \frac{\text{Var}(\bar{X}_n)}{\varepsilon^2} = \frac{\sigma^2}{n\varepsilon^2}
    $$

    $n \to \infty$일 때:

    $$
    P(|\bar{X}_n - \mu| \geq \varepsilon) \leq \frac{\sigma^2}{n\varepsilon^2} \to 0
    $$

    따라서 $\bar{X}_n \xrightarrow{p} \mu$이며, 이것이 일치성의 정의이다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
대칭분포에서는 표본중앙값도 모평균의 일치추정량이다. 왜 $\mu$로 수렴하는지 직관적으로 설명하고, 표본평균에 비해 표본중앙값이 갖는 장점을 하나 서술하라.

</div>

??? success "풀이"
    대칭분포에서는 모평균과 모집단 중앙값이 일치한다. 표본중앙값은 Glivenko-Cantelli 정리에 의해 모집단 중앙값으로 수렴한다(경험 CDF가 참 CDF로 균등수렴하므로 분위수도 수렴한다). 대칭분포에서는 모집단 중앙값이 $\mu$와 같으므로 표본중앙값은 $\mu$에 대해 일치한다.

    **중앙값의 장점:** 이상점에 로버스트하다. 꼬리가 두꺼운 분포(예: Cauchy)에서는 표본평균의 변동이 매우 커지고 일치하지조차 않을 수 있지만(Cauchy 분포에는 유한한 평균이 없다) 표본중앙값은 여전히 일치하고 안정적이다. 분산이 유한한 분포에서도 중앙값은 영향함수가 유계여서 극단 관측값 하나가 추정값을 크게 바꾸지 못한다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
$\hat{\theta}_n$이 $\theta$의 일치추정량이고 $g$가 연속함수이면 $g(\hat{\theta}_n)$이 $g(\theta)$의 일치추정량임을 보여라. 어떤 정리를 사용하는지 밝혀라.

</div>

??? success "풀이"
    이는 **연속사상정리**이다. $\hat{\theta}_n \xrightarrow{p} \theta$이고 $g$가 $\theta$에서 연속이면 $g(\hat{\theta}_n) \xrightarrow{p} g(\theta)$이다.

    **응용:** $S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2 \xrightarrow{p} \sigma^2$이므로(표본분산은 모분산에 대해 일치한다) 연속함수 $g(x) = \sqrt{x}$를 적용하면:

    $$
    S = \sqrt{S^2} \xrightarrow{p} \sqrt{\sigma^2} = \sigma
    $$

    따라서 표본표준편차는 $\sigma$의 일치추정량이다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
추정량 $\hat{\theta}_n$의 점근정규성을 정의하라. $\hat{\theta}_n$이 MLE이고 정칙 조건이 성립할 때 $\sqrt{n}(\hat{\theta}_n - \theta_0)$의 점근분포를 서술하라.

</div>

??? success "풀이"
    추정량 $\hat{\theta}_n$이 다음을 만족하면 **점근적으로 정규**이다:

    $$
    \sqrt{n}(\hat{\theta}_n - \theta_0) \xrightarrow{d} N(0, v^2)
    $$

    여기서 $v^2$은 어떤 분산이고 $\xrightarrow{d}$는 분포수렴을 나타낸다.

    표준적인 정칙 조건(모수공간이 열려 있고, 모형이 식별 가능하며, 로그가능도가 두 번 미분가능한 등) 아래에서 MLE는:

    $$
    \sqrt{n}(\hat{\theta}_{\text{MLE}} - \theta_0) \xrightarrow{d} N\!\left(0, \frac{1}{I(\theta_0)}\right)
    $$

    이며 $I(\theta_0) = -E\!\left[\frac{\partial^2}{\partial\theta^2}\log f(X;\theta_0)\right]$는 관측값 하나당 Fisher 정보량이다. 동등하게, $n$이 크면 $\hat{\theta}_{\text{MLE}} \approx N(\theta_0, 1/(nI(\theta_0)))$이다. 이 결과는 MLE가 점근적으로 Cramér-Rao 하한을 달성함을, 즉 점근적으로 효율적임을 함의한다.

---

## 정리하며

일치성은 추정량에 요구하는 **최소한의 조건**이다. 자료를 무한히 모으면 참값으로 가야 한다.

$$
\hat\theta_n \xrightarrow{P} \theta
$$

- **MSE 로 확인하는 것이 가장 쉽다.** 편향과 분산이 모두 $0$ 으로 가면 일치성이 따라온다. **충분조건이지 필요조건은 아니다** — MSE 가 무한해도 일치할 수 있다.
- **일치성은 속도를 말하지 않는다.** $n^{-1/2}$ 로 가든 $1/\log n$ 로 가든 둘 다 일치추정량이다. 실무에서 둘을 가르는 것은 **점근정규성과 그 속도**다.
- **점근정규성** $\sqrt n(\hat\theta_n-\theta)\xrightarrow{d}N(0,v)$ 이 신뢰구간과 검정을 가능하게 한다. $v$ 가 점근분산이며, 추정량을 비교하는 기준이 된다.
- **주의: 점근분산은 분산의 극한이 아니다.** 3장에서 보았듯 $\sqrt n(\hat\theta-\theta)$ 가 정규로 가면서도 실제 분산은 무한할 수 있다. 균등적분가능성이 따로 필요하다.
- **일치성만으로는 부족하다.** 언제나 $\hat\theta_n=\bar X_n+1/n$ 처럼 일치하지만 나쁜 추정량을 만들 수 있다.

다음 절 **효율성과 크라메르–라오 하한**으로 넘어간다. 불편추정량의 분산이 **얼마나 작아질 수 있는가**라는 물음에 정확한 답이 있다.
