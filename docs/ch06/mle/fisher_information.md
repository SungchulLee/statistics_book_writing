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


<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> 베르누이분포. $X \sim \text{Bernoulli}(p)$이면 $x \in \{0, 1\}$에서 $f(x; p) = p^x (1-p)^{1-x}$이다. 관측값 하나에 대한 로그가능도는

</div>

??? success "풀이"
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

<div class="exbox" markdown>

**보기 2.** <span class="diff easy" title="쉬움"></span> 정규분포 (평균). $\sigma^2$이 알려진 $X \sim N(\mu, \sigma^2)$이라 하자. 관측값 하나에 대한 로그가능도는

</div>

??? success "풀이"
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

<div class="exbox" markdown>

**보기 3.** <span class="diff easy" title="쉬움"></span> 포아송분포. $X \sim \text{Poisson}(\lambda)$이면 $f(x; \lambda) = e^{-\lambda}\lambda^x / x!$이다. 로그가능도는

</div>

??? success "풀이"
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

**연습문제 1.** <span class="diff med" title="중간"></span>
파레토분포: $x \ge 1$에서 $f(x; \alpha) = \alpha x^{-(\alpha+1)}$이다. (a) $\hat\alpha_{\text{MLE}}$를 유도하라. (b) $\mathbb{E}[X] = \alpha/(\alpha-1)$을 이용한 적률법. (c) Fisher 정보량과 CRLB. (d) MLE는 효율적인가?

</div>

??? success "풀이"
    (a) $\ell(\alpha) = n\ln\alpha - (\alpha + 1)\sum \ln x_i$. $\ell'(\alpha) = n/\alpha - \sum\ln x_i = 0 \Rightarrow \hat\alpha_{\text{MLE}} = n/\sum \ln x_i$.

    (b) $\bar x = \alpha/(\alpha - 1) \Rightarrow \hat\alpha_{\text{MoM}} = \bar x/(\bar x - 1)$.

    (c) $\partial^2 \ln f/\partial\alpha^2 = -1/\alpha^2 \Rightarrow I(\alpha) = 1/\alpha^2$. CRLB = $\alpha^2/n$.

    (d) $Y = \ln X \sim \mathrm{Exp}(\alpha)$이므로 $\sum \ln X_i \sim \mathrm{Gamma}(n, 1/\alpha)$이고 $\hat\alpha_{\text{MLE}} = n/\sum \ln X_i$는 $\mathbb{E}[\hat\alpha_{\text{MLE}}] = n\alpha/(n-1)$을 만족한다(위쪽으로 편향). 편향 보정한 $\tilde\alpha = (n-1)/n \cdot \hat\alpha_{\text{MLE}}$가 점근적으로 CRLB를 달성한다. MLE는 점근적으로 효율적이다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
Poisson$(\lambda)$: $f(x; \lambda) = \lambda^x e^{-\lambda}/x!$. (a) $I(\lambda)$를 계산하라. (b) CRLB. (c) $\bar X$가 효율적임을 보여라. (d) $g(\lambda) = e^{-\lambda}$에 대한 CRLB.

</div>

??? success "풀이"
    (a) $\ln f = x\ln\lambda - \lambda - \ln x!$. $\partial^2/\partial\lambda^2 = -x/\lambda^2$. $I(\lambda) = \mathbb{E}[X]/\lambda^2 = 1/\lambda$.

    (b) CRLB = $\lambda/n$.

    (c) $\mathrm{Var}(\bar X) = \lambda/n$ = CRLB이므로 효율적이다.

    (d) 델타 방법에 의한 CRLB: $\mathrm{Var}(\widehat{g}) \ge [g'(\lambda)]^2/(n I(\lambda)) = e^{-2\lambda} \cdot \lambda/n = \lambda e^{-2\lambda}/n$.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
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

**연습문제 4.** <span class="diff med" title="중간"></span>
**Fisher 정보량의 가법성.** i.i.d. 자료에서 $I$가 관측값 하나당 Fisher 정보량일 때 $I_n(\theta) = n I(\theta)$이다. 이를 증명하고 해석하라.

</div>

??? success "풀이"
    결합 로그가능도: $\log L(\theta) = \sum_i \log f(X_i; \theta)$.

    점수: $\partial \log L/\partial\theta = \sum_i \partial \log f(X_i; \theta)/\partial\theta$. 독립 확률변수의 합의 분산은 분산의 합이므로:

    $\mathrm{Var}(\partial \log L/\partial\theta) = \sum_i \mathrm{Var}(\partial \log f(X_i; \theta)/\partial\theta) = n I(\theta)$.

    따라서 $I_n(\theta) = n I(\theta)$이다. $\square$

    **해석:** 정보는 표본크기에 선형으로 쌓인다. $n$을 두 배로 하면 Fisher 정보량이 두 배가 되고 CRLB가 절반이 되며 MLE의 점근분산이 좁아진다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
**정보량과 재모수화.** 미분가능한 $g$에 대해 $\eta = g(\theta)$일 때 $I(\eta) = I(\theta)/[g'(\theta)]^2$임을 보여라.

</div>

??? success "풀이"
    연쇄법칙에 의해 $\partial \log f/\partial\eta = (\partial \log f/\partial\theta) \cdot (\partial\theta/\partial\eta) = (\partial \log f/\partial\theta)/g'(\theta)$이다.

    분산을 취하면 $I(\eta) = \mathbb{E}[(\partial \log f/\partial\eta)^2] = \mathbb{E}[(\partial \log f/\partial\theta)^2]/[g'(\theta)]^2 = I(\theta)/[g'(\theta)]^2$이다.

    $\square$

    **함의:** Fisher 정보량은 **모수화에 의존한다**. 재모수화하면 정보량의 크기가 달라진다. 이것이 Jeffreys 사전분포 $\pi(\theta) \propto \sqrt{I(\theta)}$가 매력적인 이유 중 하나이다. (평평한 사전분포와 달리) 재모수화에 불변이기 때문이다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
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

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
모수가 여럿일 때 Fisher 정보**행렬**의 역행렬 대각성분이 표준오차를 준다. 2×2인 경우에 $\operatorname{Var}(\hat\theta_1) \ge 1/I_{11}$이 아니라 $\ge (I^{-1})_{11}$인 이유를 설명하라.

</div>

??? success "풀이"
    $2\times2$ 역행렬 공식에서

    $$
    (I^{-1})_{11} = \frac{I_{22}}{I_{11}I_{22}-I_{12}^2} = \frac{1}{I_{11}-I_{12}^2/I_{22}} \ge \frac{1}{I_{11}}
    $$

    이다. 등호는 $I_{12}=0$일 때만 성립한다.

    **뜻.** $1/I_{11}$은 **$\theta_2$를 안다고 가정했을 때**의 하한이다. 실제로는 $\theta_2$도 추정해야 하므로 $\theta_1$의 불확실성이 더 커진다. 그 증가분이 $I_{12}^2/I_{22}$이며, **두 모수가 얽혀 있을수록**(정보행렬의 비대각이 클수록) 커진다.

    직관적으로, 로그가능도 곡면이 기울어진 타원이면 $\theta_1$ 방향으로 자른 단면은 가파르지만 $\theta_2$를 자유롭게 두고 본 프로파일은 훨씬 완만하다. 그 완만함이 실제 불확실성이다.

    **극단적인 경우.** $I_{12}^2 \to I_{11}I_{22}$이면 행렬식이 0으로 가서 분산이 폭발한다. 이것이 **모수가 식별되지 않거나 다중공선성이 있는 상황**이다.

    **직교 모수화.** $I_{12}=0$이면 두 하한이 일치하고, 성가신 모수를 추정하는 대가가 사라진다. 정규분포의 $(\mu,\sigma^2)$이 그런 경우이며, 그래서 $\sigma$를 몰라도 $\hat\mu$의 점근분산이 커지지 않는다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
$I(\theta) = E\left[\left(\dfrac{\partial\ln f}{\partial\theta}\right)^2\right]$이라는 정의에서, Fisher 정보량이 **분포가 $\theta$에 얼마나 민감한가**를 재는 양임을 설명하라. 쿨백-라이블러 발산과의 관계를 밝혀라.

</div>

??? success "풀이"
    **민감도 해석.** 점수 $s(x;\theta) = \partial\ln f/\partial\theta$는 "$\theta$를 조금 바꿨을 때 이 관측값의 로그확률이 얼마나 달라지는가"이다. 정보량은 그 제곱의 기대값, 즉 **점수의 분산**($E[s]=0$이므로)이다.

    - 점수가 크게 흔들린다 = 관측값마다 "어느 $\theta$가 맞는지"에 대해 강한 의견을 낸다 = 자료가 $\theta$를 잘 구별해 준다.
    - 점수가 거의 0이다 = $\theta$를 바꿔도 자료의 확률이 별로 달라지지 않는다 = $\theta$를 알아내기 어렵다.

    **KL 발산과의 관계.** $\theta$ 근처에서 $\theta+\delta$로의 쿨백-라이블러 발산을 전개하면

    $$
    D_{\text{KL}}\left(f_\theta \,\|\, f_{\theta+\delta}\right) = \frac12 I(\theta)\,\delta^2 + O(\delta^3)
    $$

    이다. 1차항이 사라지는 것은 $D_{\text{KL}}$이 $\delta=0$에서 최솟값 0을 갖기 때문이고, 2차항의 계수가 정확히 $I(\theta)/2$다.

    **따라서 $I(\theta)$는 모수공간에 놓인 "거리"의 국소 척도다.** 정보량이 크면 $\theta$를 조금만 바꿔도 분포가 크게 달라지고, 그만큼 구별이 쉽다. 이 관점에서 $\sqrt{I(\theta)}\,d\theta$를 모수공간의 길이 요소로 삼는 것이 **정보기하학**이며, 그 길이로 잰 거리를 피셔-라오 거리라 한다.

    **부산물.** 제프리스 사전분포 $\pi(\theta) \propto \sqrt{I(\theta)}$가 여기서 나온다. 정보기하학적 부피 요소이므로 **모수화에 불변**이며, $\theta$로 적든 $g(\theta)$로 적든 같은 사전분포를 준다. 균등 사전분포가 그렇지 못한 것과 대비된다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
$g(\theta)$를 추정할 때의 크라메르-라오 하한이 $\{g'(\theta)\}^2/\{nI_1(\theta)\}$임을 설명하고, 포아송에서 $g(\lambda)=e^{-\lambda}$(사건이 0번 일어날 확률)의 하한을 구하라. $\hat g = e^{-\bar X}$가 이 하한을 달성하는가?

</div>

??? success "풀이"
    **하한.** $\hat g$가 $g(\theta)$의 불편추정량이면 크라메르-라오 부등식의 일반형에서

    $$
    \operatorname{Var}(\hat g) \ge \frac{\{g'(\theta)\}^2}{nI_1(\theta)}
    $$

    이다. 델타 방법의 결과와 같은 꼴이며, 이는 우연이 아니라 MLE의 점근효율성이 변환에 보존되기 때문이다.

    **포아송 적용.** $I_1(\lambda) = 1/\lambda$이고 $g(\lambda)=e^{-\lambda}$이므로 $g'(\lambda) = -e^{-\lambda}$이고

    $$
    \text{CRLB} = \frac{e^{-2\lambda}}{n/\lambda} = \frac{\lambda e^{-2\lambda}}{n}
    $$

    **$\hat g = e^{-\bar X}$는 달성하는가.** 점근적으로는 달성한다. 델타 방법으로

    $$
    \operatorname{Var}(e^{-\bar X}) \approx e^{-2\lambda}\operatorname{Var}(\bar X) = \frac{\lambda e^{-2\lambda}}{n}
    $$

    로 하한과 일치한다.

    그러나 **유한표본에서는 불편이 아니다.** $e^{-x}$가 볼록하므로 $E[e^{-\bar X}] > e^{-\lambda}$이다. 실제로 $T=\sum X_i \sim \text{Poisson}(n\lambda)$이고 적률생성함수를 쓰면

    $$
    E\left[e^{-T/n}\right] = \exp\left\{n\lambda\left(e^{-1/n}-1\right)\right\} \ne e^{-\lambda}
    $$

    이다.

    **불편추정량.** 레만-셰페로 구할 수 있다. $T$가 완비충분통계량이고

    $$
    \hat g_{\text{UMVUE}} = \left(1-\frac1n\right)^{T}
    $$

    이 불편임을 확인할 수 있다($E[a^T] = e^{n\lambda(a-1)}$에 $a = 1-1/n$을 넣으면 $e^{-\lambda}$). 이것이 유일한 UMVUE지만 **분산이 크라메르-라오 하한보다 크다.** $e^{-\lambda}$가 $\lambda$의 선형함수가 아니라서 하한이 달성 불가능한 예다.

    **교훈.** **크라메르-라오 하한은 언제나 달성되지는 않는다.** 하한을 달성하는 불편추정량이 존재하려면 점수함수가 $\hat g - g(\theta)$에 비례해야 하고, 이는 지수족에서 자연모수의 특정 함수일 때만 성립한다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
실험을 설계할 때 Fisher 정보량을 최대로 하도록 조건을 고르는 방식을 **최적 설계**라 한다. 로지스틱 회귀에서 용량 $x$를 어디에 배치해야 기울기를 가장 정밀하게 추정하는지 논하라.

</div>

??? success "풀이"
    로지스틱 회귀의 정보행렬은

    $$
    I(\boldsymbol\beta) = \sum_i p_i(1-p_i)\,\mathbf{x}_i\mathbf{x}_i^\top, \qquad p_i = \sigma(\beta_0+\beta_1x_i)
    $$

    이다. **가중치 $p(1-p)$가 $p=0.5$에서 최대**이고 $p$가 0이나 1에 가까우면 거의 0이 된다.

    **뜻.** 반응이 거의 확실한 용량(전부 죽거나 전부 사는 용량)에서 얻는 정보가 **거의 없다.** 그런 관측값은 "이미 아는 것"만 확인해 줄 뿐이다.

    **기울기 $\beta_1$을 겨냥한 최적 설계.** 위 가중치와 $\mathbf{x}\mathbf{x}^\top$의 퍼짐을 함께 고려해야 한다. $\operatorname{SE}(\hat\beta_1)$을 최소로 하려면 $x$가 널리 퍼져 있어야 하는데, 너무 멀리 가면 $p(1-p)$가 0이 되어 정보가 사라진다. 두 요구의 균형점이

    $$
    p = 0.176 \quad\text{과}\quad p = 0.824
    $$

    에 해당하는 두 용량에 표본을 반씩 배치하는 것으로 알려져 있다($\beta_0+\beta_1x = \pm1.5434$). 이것이 기울기에 대한 $D$-최적 설계다.

    **실무에서 그대로 쓰지 않는 이유.**

    - **모수를 알아야 설계할 수 있다.** 최적 용량이 $(\beta_0,\beta_1)$에 의존하는데, 그것이 바로 추정하려는 대상이다. 순환 문제다. 대처로 선행 정보를 쓰거나(국소 최적 설계), 사전분포를 두고 평균하거나(베이즈 최적 설계), 순차적으로 갱신한다(적응 설계).
    - **모형 진단이 불가능해진다.** 두 점만 쓰면 로지스틱 곡선의 모양이 맞는지 확인할 수 없다. 굽은 관계를 놓쳐도 알아챌 방법이 없다.
    - **다른 목적이 있을 수 있다.** $\text{LD}_{50}$(반수 치사량)을 추정하려면 $p=0.5$ 근처에 집중하는 것이 낫다. **무엇을 추정할 것인가에 따라 최적 설계가 달라진다.**

    **타협.** 실무에서는 최적점 근처에 대부분을 배치하되 중간과 양끝에도 일부를 두어 모형 검증의 여지를 남긴다. 최적 설계 이론은 **어디에 자원을 몰아야 하는지에 대한 지침**으로 쓰는 것이 적절하다.

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
