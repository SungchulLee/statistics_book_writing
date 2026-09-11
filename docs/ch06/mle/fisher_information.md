# Fisher 정보량과 표준오차

## 왜 정보를 정량화하는가?

자료로부터 모수 $\theta$를 추정할 때 우리는 $\theta$를 얼마나 정밀하게 추정할 수 있는지 알고 싶다. 그 답은 자료가 $\theta$에 관해 얼마나 많은 "정보"를 담고 있는지에 달려 있다. Fisher 정보량은 이 개념을 형식화한다. 임의의 불편추정량이 도달할 수 있는 가장 날카로운 정밀도를 결정하고, 최대가능도추정값의 표준오차를 곧바로 제공한다. 요컨대 Fisher 정보량은 모수적 모형과 그로부터 이끌어 낼 수 있는 추론의 질을 잇는 다리이다.

## 점수함수

Fisher 정보량을 정의하기 위해 먼저 **점수함수**를 소개한다. $X$를 확률밀도함수 또는 확률질량함수가 $f(x; \theta)$인 확률변수라 하자. 점수함수는 로그가능도를 $\theta$에 대해 미분한 것이다:

$$
s(x; \theta) = \frac{\partial}{\partial \theta} \log f(x; \theta)
$$

점수함수는 로그가능도가 $\theta$의 변화에 얼마나 민감한지를 담아낸다. $\theta$가 변할 때 로그가능도가 급격히 달라지면 자료가 $\theta$에 관해 많은 정보를 담고 있는 것이고, 평평하면 정보가 거의 없는 것이다.

정칙 조건 아래에서, 구체적으로 지지집합 $\{x : f(x; \theta) > 0\}$이 $\theta$에 의존하지 않고 미분과 적분을 교환할 수 있을 때, 점수함수는 핵심적인 성질을 갖는다:

$$
E[s(X; \theta)] = 0
$$

즉 점수함수는 0을 중심으로 하며, 그 분산이 로그가능도 기울기의 전형적인 크기를 잰다.

## Fisher 정보량의 정의

관측값 하나에 대한 **Fisher 정보량**은 점수함수의 분산이다:

$$
I(\theta) = E\left[\left(\frac{\partial}{\partial \theta} \log f(X;\theta)\right)^2\right] = \text{Var}(s(X; \theta))
$$

같은 정칙 조건(미분과 적분의 교환) 아래에서 이는 로그가능도의 기대 곡률에 음수를 취한 것과 동등하다:

$$
I(\theta) = -E\left[\frac{\partial^2}{\partial\theta^2} \log f(X;\theta)\right]
$$

두 번째 형태가 계산하기 더 쉬운 경우가 많다. 기하적인 직관도 준다. 로그가능도가 급하게 휘어 있으면(큰 $I(\theta)$) 자료가 모수를 강하게 제약하는 것이고, 평평하면(작은 $I(\theta)$) 자료가 정보를 주지 못하는 것이다.

독립이고 동일한 분포를 따르는 $n$개의 관측값 $X_1, \ldots, X_n$에 대해 전체 Fisher 정보량은 가법적이다:

$$
I_n(\theta) = n \cdot I(\theta)
$$

## 예제

### Bernoulli 분포

$X \sim \text{Bernoulli}(p)$이면 $x \in \{0, 1\}$에서 $f(x; p) = p^x (1-p)^{1-x}$이다. 관측값 하나에 대한 로그가능도는

$$
\log f(x; p) = x \log p + (1-x) \log(1-p)
$$

2계도함수를 취하면:

$$
\frac{\partial^2}{\partial p^2} \log f(x; p) = -\frac{x}{p^2} - \frac{1-x}{(1-p)^2}
$$

$E[X] = p$이므로:

$$
I(p) = -E\left[-\frac{X}{p^2} - \frac{1-X}{(1-p)^2}\right] = \frac{p}{p^2} + \frac{1-p}{(1-p)^2} = \frac{1}{p} + \frac{1}{1-p} = \frac{1}{p(1-p)}
$$

Fisher 정보량은 $p$가 0이나 1에 가까울 때 가장 크고(그때 관측값 하나가 $p$에 관해 가장 많은 정보를 준다) $p = 1/2$에서 가장 작다.

### Normal 분포 (평균)

$\sigma^2$이 알려진 $X \sim N(\mu, \sigma^2)$이라 하자. 관측값 하나에 대한 로그가능도는

$$
\log f(x; \mu) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x - \mu)^2}{2\sigma^2}
$$

$\mu$에 대한 2계도함수는

$$
\frac{\partial^2}{\partial \mu^2} \log f(x; \mu) = -\frac{1}{\sigma^2}
$$

이는 ($x$에 의존하지 않는) 상수이므로

$$
I(\mu) = \frac{1}{\sigma^2}
$$

잡음 $\sigma^2$이 작아질수록 Fisher 정보량이 커지며, 이는 직관과 맞아떨어진다. 잡음이 적은 자료가 평균에 관해 더 많은 정보를 담는다.

### Poisson 분포

$X \sim \text{Poisson}(\lambda)$이면 $f(x; \lambda) = e^{-\lambda}\lambda^x / x!$이다. 로그가능도는

$$
\log f(x; \lambda) = -\lambda + x \log \lambda - \log(x!)
$$

2계도함수는

$$
\frac{\partial^2}{\partial \lambda^2} \log f(x; \lambda) = -\frac{x}{\lambda^2}
$$

$E[X] = \lambda$이므로:

$$
I(\lambda) = \frac{1}{\lambda}
$$

## Fisher 정보량으로부터의 표준오차

Fisher 정보량의 가장 중요한 응용 중 하나는 MLE의 표준오차를 제공하는 것이다. 정칙 조건 아래에서 MLE $\hat{\theta}_{\text{MLE}}$는 점근적으로 정규이다:

$$
\hat{\theta}_{\text{MLE}} \overset{d}{\to} N\left(\theta, \frac{1}{nI(\theta)}\right) \quad \text{as } n \to \infty
$$

실무에서는 (대입 원리에 따라) Fisher 정보량을 MLE에서 평가하여 표준오차를 추정한다:

$$
\widehat{\text{SE}}(\hat{\theta}_{\text{MLE}}) = \frac{1}{\sqrt{nI(\hat{\theta}_{\text{MLE}})}}
$$

이 근사는 표본크기가 커질수록 좋아지며 신뢰구간과 Wald 검정통계량 구성의 바탕이 된다.

!!! example "Bernoulli MLE의 표준오차"

    $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Bernoulli}(p)$에서 MLE는 $\hat{p} = \bar{X}$이다. 추정된 표준오차는

    $$
    \widehat{\text{SE}}(\hat{p}) = \frac{1}{\sqrt{n \cdot \frac{1}{\hat{p}(1-\hat{p})}}} = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
    $$

    표본비율의 표준오차에 대한 익숙한 공식이다.

## 흔한 Fisher 정보량 값 요약

| 분포 | 모수 | Fisher 정보량 $I(\theta)$ |
|---|---|---|
| $\text{Bernoulli}(p)$ | $p$ | $\dfrac{1}{p(1-p)}$ |
| $N(\mu, \sigma^2)$ ($\sigma^2$ 알려짐) | $\mu$ | $\dfrac{1}{\sigma^2}$ |
| $\text{Poisson}(\lambda)$ | $\lambda$ | $\dfrac{1}{\lambda}$ |
| $\text{Exp}(\lambda)$ | $\lambda$ | $\dfrac{1}{\lambda^2}$ |

이 표의 각 항목은 위의 예제에서 보인 2계도함수 방법으로 확인할 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
Pareto 분포: $x \ge 1$에서 $f(x; \alpha) = \alpha x^{-(\alpha+1)}$이다. (a) $\hat\alpha_{\text{MLE}}$를 유도하라. (b) $\mathbb{E}[X] = \alpha/(\alpha-1)$을 이용한 적률법. (c) Fisher 정보량과 CRLB. (d) MLE는 효율적인가?

</div>

??? success "풀이"
    (a) $\ell(\alpha) = n\ln\alpha - (\alpha + 1)\sum \ln x_i$. $\ell'(\alpha) = n/\alpha - \sum\ln x_i = 0 \Rightarrow \hat\alpha_{\text{MLE}} = n/\sum \ln x_i$.

    (b) $\bar x = \alpha/(\alpha - 1) \Rightarrow \hat\alpha_{\text{MoM}} = \bar x/(\bar x - 1)$.

    (c) $\partial^2 \ln f/\partial\alpha^2 = -1/\alpha^2 \Rightarrow I(\alpha) = 1/\alpha^2$. CRLB = $\alpha^2/n$.

    (d) $Y = \ln X \sim \mathrm{Exp}(\alpha)$이므로 $\sum \ln X_i \sim \mathrm{Gamma}(n, 1/\alpha)$이고 $\hat\alpha_{\text{MLE}} = n/\sum \ln X_i$는 $\mathbb{E}[\hat\alpha_{\text{MLE}}] = n\alpha/(n-1)$을 만족한다(위쪽으로 편향). 편향 보정한 $\tilde\alpha = (n-1)/n \cdot \hat\alpha_{\text{MLE}}$가 점근적으로 CRLB를 달성한다. MLE는 점근적으로 효율적이다.

<div class="drillbox" markdown>

**연습문제 2.**
Poisson$(\lambda)$: $f(x; \lambda) = \lambda^x e^{-\lambda}/x!$. (a) $I(\lambda)$를 계산하라. (b) CRLB. (c) $\bar X$가 효율적임을 보여라. (d) $g(\lambda) = e^{-\lambda}$에 대한 CRLB.

</div>

??? success "풀이"
    (a) $\ln f = x\ln\lambda - \lambda - \ln x!$. $\partial^2/\partial\lambda^2 = -x/\lambda^2$. $I(\lambda) = \mathbb{E}[X]/\lambda^2 = 1/\lambda$.

    (b) CRLB = $\lambda/n$.

    (c) $\mathrm{Var}(\bar X) = \lambda/n$ = CRLB이므로 효율적이다.

    (d) 델타 방법에 의한 CRLB: $\mathrm{Var}(\widehat{g}) \ge [g'(\lambda)]^2/(n I(\lambda)) = e^{-2\lambda} \cdot \lambda/n = \lambda e^{-2\lambda}/n$.

<div class="drillbox" markdown>

**연습문제 3.**
**Fisher 정보량을 정의하고** 두 정의의 동등성 $I(\theta) = \mathbb{E}[(\partial \log f/\partial\theta)^2] = -\mathbb{E}[\partial^2 \log f/\partial\theta^2]$을 증명하라.

</div>

??? success "풀이"
    **정의 1 (점수의 분산):** $I(\theta) = \mathbb{E}[(\partial \log f/\partial\theta)^2]$.

    **정의 2 (Hessian):** $I(\theta) = -\mathbb{E}[\partial^2 \log f/\partial\theta^2]$.

    **동등성 증명:** 항등식 $\int f \, dx = 1$을 한 번 미분하면 $\int \partial f/\partial\theta \, dx = 0 \Rightarrow \int f \cdot \partial \log f/\partial\theta \, dx = 0 \Rightarrow \mathbb{E}[\partial \log f/\partial\theta] = 0$이다.

    다시 미분하면 $\int [\partial^2 f/\partial\theta^2] dx = 0$이다. $\partial^2 f/\partial\theta^2 = f \partial^2 \log f/\partial\theta^2 + f (\partial \log f/\partial\theta)^2$으로 나타내면:

    $\mathbb{E}[\partial^2 \log f/\partial\theta^2] + \mathbb{E}[(\partial \log f/\partial\theta)^2] = 0$.

    따라서 $I(\theta) = \mathbb{E}[(\partial \log f/\partial\theta)^2] = -\mathbb{E}[\partial^2 \log f/\partial\theta^2]$이다. $\square$

    두 정의는 계산상 서로 바꿔 쓸 수 있으므로, 주어진 가능도에서 더 쉬운 쪽을 택하면 된다.

<div class="drillbox" markdown>

**연습문제 4.**
**Fisher 정보량의 가법성.** i.i.d. 자료에서 $I$가 관측값 하나당 Fisher 정보량일 때 $I_n(\theta) = n I(\theta)$이다. 이를 증명하고 해석하라.

</div>

??? success "풀이"
    결합 로그가능도: $\log L(\theta) = \sum_i \log f(X_i; \theta)$.

    점수: $\partial \log L/\partial\theta = \sum_i \partial \log f(X_i; \theta)/\partial\theta$. 독립 확률변수의 합의 분산은 분산의 합이므로:

    $\mathrm{Var}(\partial \log L/\partial\theta) = \sum_i \mathrm{Var}(\partial \log f(X_i; \theta)/\partial\theta) = n I(\theta)$.

    따라서 $I_n(\theta) = n I(\theta)$이다. $\square$

    **해석:** 정보는 표본크기에 선형으로 쌓인다. $n$을 두 배로 하면 Fisher 정보량이 두 배가 되고 CRLB가 절반이 되며 MLE의 점근분산이 좁아진다.

<div class="drillbox" markdown>

**연습문제 5.**
**정보량과 재모수화.** 미분가능한 $g$에 대해 $\eta = g(\theta)$일 때 $I(\eta) = I(\theta)/[g'(\theta)]^2$임을 보여라.

</div>

??? success "풀이"
    연쇄법칙에 의해 $\partial \log f/\partial\eta = (\partial \log f/\partial\theta) \cdot (\partial\theta/\partial\eta) = (\partial \log f/\partial\theta)/g'(\theta)$이다.

    분산을 취하면 $I(\eta) = \mathbb{E}[(\partial \log f/\partial\eta)^2] = \mathbb{E}[(\partial \log f/\partial\theta)^2]/[g'(\theta)]^2 = I(\theta)/[g'(\theta)]^2$이다.

    $\square$

    **함의:** Fisher 정보량은 **모수화에 의존한다**. 재모수화하면 정보량의 크기가 달라진다. 이것이 Jeffreys 사전분포 $\pi(\theta) \propto \sqrt{I(\theta)}$가 매력적인 이유 중 하나이다. (평평한 사전분포와 달리) 재모수화에 불변이기 때문이다.

<div class="drillbox" markdown>

**연습문제 6.**
**관측 정보량과 기대 정보량.** 둘을 정의하라. 언제 같고 언제 다른가?

</div>

??? success "풀이"
    **기대 (Fisher) 정보량:** $I(\theta) = -\mathbb{E}_\theta[\partial^2 \log f/\partial\theta^2]$. 참 분포 아래에서 기댓값을 취한다.

    **관측 정보량:** $J(\hat\theta) = -\partial^2 \log L/\partial\theta^2 \big|_{\hat\theta}$. *표본*의 MLE에서 평가하며 기댓값을 취하지 않는다.

    **일치:** $\theta = \hat\theta$에서 그리고 점근적으로 $J(\hat\theta)/n \to I(\theta)$이다. 기댓값의 의미에서 둘이 일치한다.

    **차이:**

    - 유한표본에서 관측 정보량은 실제 자료를 사용하므로 다른 점추정값을 줄 수 있다.
    - 기대 정보량은 분포에 대해 적분해야 하는데, 해석적으로 어렵거나 불가능할 수 있다.
    - 관측 정보량은 MLE에서 2계도함수만 계산하면 되므로 계산이 더 간단하다.

    **권고:** Efron과 Hinkley(1978)는 유한표본 추론에서 **관측 정보량**이 낫다고 주장한다(많은 경우 더 정확한 신뢰구간을 준다). 통계 소프트웨어(R, Python의 `statsmodels`)는 대개 관측 정보량을 기본으로 사용한다.

---

## 정리하며

피셔 정보량은 **자료가 모수에 관해 담은 정보의 양**이며, 추정의 정밀도를 직접 결정한다.

- **점수함수** $s(x;\theta)=\partial_\theta\log f(x;\theta)$ 는 로그가능도가 $\theta$ 에 얼마나 민감한지를 잰다. 정칙 조건 아래 $\mathbb{E}[s]=0$ 이므로 정보량은 점수의 **분산**이다.
- **두 가지 표현이 같다.**

$$
I(\theta)=\mathbb{E}\bigl[s(X;\theta)^2\bigr] = -\,\mathbb{E}\left[\frac{\partial^2}{\partial\theta^2}\log f(X;\theta)\right]
$$

  두 번째 것이 실무에서 계산하기 쉽고, **로그가능도의 곡률**이라는 해석을 준다. 봉우리가 뾰족할수록 정보가 많다.
- **가법적이다.** i.i.d. 관측 $n$ 개의 정보량은 $nI(\theta)$ 이며, 이것이 표준오차가 $1/\sqrt{nI(\theta)}$ 로 줄어드는 이유다.
- **실무의 표준오차는 관측정보로 계산한다.** 기댓값을 구하기 어려울 때 $\hat\theta$ 에서 평가한 $-\ell''(\hat\theta)$ 을 쓰며, 수치 최적화기가 돌려주는 헤세행렬이 바로 그것이다.
- **다모수에서는 행렬이 된다.** 정보행렬의 역행렬이 점근 공분산행렬이며, 대각원소의 제곱근이 각 모수의 표준오차다. **비대각 원소가 0 이 아니면 모수들이 얽혀 있어** 하나를 정밀하게 추정하기가 더 어려워진다.

다음 절 **최대가능도 최적화 예제**로 넘어간다. 닫힌 형태가 없을 때 실제로 봉우리를 찾는 방법이다.
