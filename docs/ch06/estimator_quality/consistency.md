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

    **중앙값의 장점:** 이상점에 로버스트하다. 꼬리가 두꺼운 분포(예: Cauchy)에서는 표본평균의 변동이 매우 커지고 일치하지조차 않을 수 있지만(코시분포에는 유한한 평균이 없다) 표본중앙값은 여전히 일치하고 안정적이다. 분산이 유한한 분포에서도 중앙값은 영향함수가 유계여서 극단 관측값 하나가 추정값을 크게 바꾸지 못한다.

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

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
**확률수렴**과 **거의 확실한 수렴**을 정의하고, 일치성이 보통 전자로 정의되는 이유를 설명하라. 둘의 관계는 무엇인가?

</div>

??? success "풀이"
    **확률수렴.** 모든 $\varepsilon>0$에 대해

    $$
    \lim_{n\to\infty}P\left(|\hat\theta_n-\theta|>\varepsilon\right) = 0
    $$

    "$n$이 크면 크게 빗나갈 **확률**이 작다"는 뜻이다.

    **거의 확실한 수렴.**

    $$
    P\left(\lim_{n\to\infty}\hat\theta_n = \theta\right) = 1
    $$

    "표본의 경로 하나하나가 거의 모두 수렴한다"는 뜻이다. 훨씬 강한 조건이다.

    **관계.** 거의 확실한 수렴 $\Rightarrow$ 확률수렴이고, 역은 성립하지 않는다. 고전적인 반례는 $[0,1]$ 위를 옮겨 다니며 폭이 줄어드는 지시함수열이다. 각 점에서 무한히 자주 1이 되므로 수렴하지 않지만, 1이 될 확률은 0으로 간다.

    **왜 확률수렴으로 정의하는가.**

    - **필요한 것이 그것뿐이다.** 실무의 질문은 "지금 내 표본에서 추정값이 참값 근처에 있을 가능성이 높은가"이지 "무한히 이어지는 표본열이 수렴하는가"가 아니다.
    - **증명하기 쉽다.** 체비쇼프 부등식으로 $\operatorname{Bias}\to0$, $\operatorname{Var}\to0$만 보이면 되고, 이는 보통 직접 계산할 수 있다. 거의 확실한 수렴은 보렐-칸텔리 보조정리 같은 더 강한 도구가 필요하다.
    - **점근이론과 잘 맞는다.** 슬러츠키 정리, 연속사상정리, 델타 방법이 모두 확률수렴을 전제로 서술된다.

    **덧붙임.** 거의 확실한 수렴이 성립하면 **강일치성**이라 부른다. 큰수의 법칙에도 약한 판본(확률수렴)과 강한 판본(거의 확실한 수렴)이 있고, 표본평균은 $E|X|<\infty$면 강일치다. 실무에서 둘의 차이가 결론을 바꾸는 경우는 드물다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
일치성의 **충분조건**으로 흔히 쓰는 "편향과 분산이 모두 0으로 가면 일치"를 증명하라. 이 조건이 필요조건은 아님을 반례로 보여라.

</div>

??? success "풀이"
    **충분조건 증명.** 마르코프 부등식에서

    $$
    P\left(|\hat\theta_n-\theta|>\varepsilon\right) \le \frac{E\left[(\hat\theta_n-\theta)^2\right]}{\varepsilon^2} = \frac{\operatorname{MSE}(\hat\theta_n)}{\varepsilon^2}
    $$

    이고 $\operatorname{MSE} = \operatorname{Bias}^2+\operatorname{Var} \to 0$이므로 우변이 0으로 간다. $\square$

    이를 **평균제곱수렴이 확률수렴을 함의한다**고 말한다.

    **필요조건이 아닌 반례.** 다음 추정량을 생각하자.

    $$
    \hat\theta_n = \begin{cases} \bar X_n & \text{확률 } 1-1/n \\ n & \text{확률 } 1/n\end{cases}
    $$

    ($\theta=0$이라 하자.)

    - **일치한다.** $P(|\hat\theta_n| > \varepsilon) \le P(|\bar X_n|>\varepsilon) + 1/n \to 0$이다.
    - **그러나 MSE가 발산한다.** $E[\hat\theta_n^2] \ge n^2\cdot\frac1n = n \to \infty$이다.

    아주 드물게(확률 $1/n$) 아주 크게(값 $n$) 빗나가는 것이 평균에는 치명적이지만 확률수렴에는 영향을 주지 않는다.

    **더 극단적인 예.** 코시분포에서 **표본중앙값**은 일치추정량이지만, 표본평균은 분산은커녕 평균도 없다. 적률이 존재하지 않아도 일치성은 얼마든지 성립한다.

    **교훈.** 편향과 분산이 0으로 가는지 보는 것은 **가장 쉬운 확인 방법**이지 유일한 방법이 아니다. 적률이 없는 상황에서는 다른 도구(직접 확률 계산, 큰수의 법칙, 극값 논증)가 필요하다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
일치성은 **점근적 성질**이라 유한표본의 성능에 대해 아무것도 보장하지 않는다. 일치하지만 실무에서 쓸 수 없는 추정량의 예를 들고, 무엇을 함께 보아야 하는지 적어라.

</div>

??? success "풀이"
    **예 1 — 수렴이 지나치게 느린 경우.** $\hat\theta_n = \bar X_n + \dfrac{10^6}{\ln n}$을 생각하자.

    - $\ln n \to \infty$이므로 두 번째 항이 0으로 가고, 일치추정량이다.
    - 그러나 $n = 10^6$이어도 두 번째 항이 $10^6/13.8 = 72{,}000$이다. 실무에서 완전히 쓸모없다.

    **예 2 — 첫 관측값만 무시하는 극단.** $\hat\theta_n = \bar X_n$이되 $n < 10^{10}$이면 0을 돌려주는 추정량도 형식적으로는 일치한다.

    **예 3 — 실제 사례.** 순간 커널 밀도추정에서 띠폭을 $h_n = n^{-1/100}$로 두면 이론적으로는 일치하지만 어떤 현실적인 $n$에서도 평활이 과하다.

    **함께 보아야 할 것.**

    - **수렴 속도.** $\hat\theta_n - \theta = O_p(n^{-1/2})$인지 $O_p(n^{-1/4})$인지가 실무 성능을 좌우한다. 비모수 방법이 모수 방법보다 느린 것이 그 예다.
    - **점근분산의 상수.** 같은 $n^{-1/2}$ 속도여도 상수가 10배 다르면 필요한 표본이 100배 다르다.
    - **유한표본 MSE.** 모의실험으로 실제 $n$에서 재 본다.
    - **점근근사가 언제부터 통하는가.** 앞 절에서 본 진단(프로파일 곡선, 부트스트랩, 포함확률 모의실험)을 쓴다.

    **한 문장으로.** **일치성은 최소한의 자격 요건이지 추천사가 아니다.** 일치하지 않는 추정량은 쓰면 안 되지만, 일치한다고 좋은 추정량인 것은 아니다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
표본크기가 커질 때 **모수의 개수도 함께 커지면** 일치성이 깨질 수 있다. 네이만-스콧 문제를 예로 들어 설명하라.

</div>

??? success "풀이"
    **설정.** $n$쌍의 관측을 얻는다.

    $$
    X_{i1}, X_{i2} \overset{\text{iid}}{\sim} N(\mu_i, \sigma^2), \qquad i=1,\dots,n
    $$

    각 쌍마다 고유한 평균 $\mu_i$가 있고(성가신 모수 $n$개), 공통 분산 $\sigma^2$이 관심 모수다.

    **MLE.** 로그가능도를 최대화하면 $\hat\mu_i = \bar X_i = (X_{i1}+X_{i2})/2$이고

    $$
    \hat\sigma^2_{\text{MLE}} = \frac{1}{2n}\sum_{i=1}^n\left\{(X_{i1}-\bar X_i)^2+(X_{i2}-\bar X_i)^2\right\} = \frac{1}{4n}\sum_i (X_{i1}-X_{i2})^2
    $$

    이다.

    **기대값.** $X_{i1}-X_{i2} \sim N(0, 2\sigma^2)$이므로 $E[(X_{i1}-X_{i2})^2] = 2\sigma^2$이고

    $$
    E[\hat\sigma^2_{\text{MLE}}] = \frac{2n\sigma^2}{4n} = \frac{\sigma^2}{2}
    $$

    **일치하지 않는다.** 큰수의 법칙에 따라 $\hat\sigma^2_{\text{MLE}} \xrightarrow{p} \sigma^2/2$로, **참값의 절반**에 수렴한다. $n\to\infty$여도 고쳐지지 않는다.

    **왜 그런가.** 각 쌍에서 자유도 2 중 1을 $\mu_i$ 추정에 쓰므로 잔차 자유도가 1뿐이다. 그런데 MLE는 2로 나눈다. 쌍이 늘어날수록 **정보와 성가신 모수가 같은 속도로 늘어** 자유도 손실이 희석되지 않는다.

    **고치는 법.**

    - **자유도 보정.** $2\hat\sigma^2_{\text{MLE}}$를 쓰면 불편이고 일치한다. 쌍당 잔차 자유도 1로 나눈 것이다.
    - **조건부 가능도.** $\bar X_i$로 조건을 걸면 $\mu_i$가 사라지고 $\sigma^2$만의 가능도가 남는다.
    - **주변 가능도(REML).** 차이 $X_{i1}-X_{i2}$의 분포만 쓴다. $\mu_i$가 들어 있지 않으므로 일치추정량을 준다.

    **일반적 교훈.** **모수의 개수가 표본크기와 함께 커지면 MLE의 표준이론이 무너진다.** 패널자료의 고정효과 모형, 사례-대조 짝지음 연구, 문항반응이론이 모두 이 구조이며, 그래서 조건부 가능도나 REML이 핵심 도구가 된다. 고차원 회귀($p/n$이 상수로 수렴)에서도 같은 종류의 문제가 나타난다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
표본평균이 일치추정량이려면 무엇이 필요한가? $E|X| = \infty$인 경우와 $E[X]$는 있지만 $\operatorname{Var}(X)=\infty$인 경우를 각각 논하라.

</div>

??? success "풀이"
    **필요충분조건.** 큰수의 법칙에 따라 $\bar X_n \xrightarrow{p} \mu$의 필요충분조건은

    $$
    E|X| < \infty
    $$

    이다. 분산은 필요 없다.

    **(1) $E|X| = \infty$인 경우.** 코시분포가 대표적이다. 앞 장에서 본 대로 $\bar X_n$의 분포가 $n$과 무관하게 $\text{Cauchy}(0,1)$이므로 어디에도 수렴하지 않는다. 표본평균이 **일치추정량이 아니다.**

    - 대안: 표본중앙값이 $\sqrt n$ 속도로 정규수렴한다. 절사평균도 잘 작동한다.
    - 진단: 표본평균의 궤적을 $n$에 대해 그리면 수렴하는 대신 계속 큰 점프가 일어난다.

    **(2) $E[X]$는 있지만 $\operatorname{Var}(X)=\infty$인 경우.** 파레토 $\alpha \in (1,2)$가 그렇다.

    - **일치성은 성립한다.** $E|X|<\infty$이므로 큰수의 법칙이 적용된다.
    - **그러나 중심극한정리가 성립하지 않는다.** 수렴 속도가 $\sqrt n$이 아니라 $n^{1-1/\alpha}$이고 극한분포가 정규가 아니라 안정분포다.
    - **표준오차를 쓸 수 없다.** $s/\sqrt n$이 수렴하지 않는다. $s$ 자체가 $n$과 함께 커진다.
    - 진단: $s$를 $n$에 대해 그려 본다. 안정되지 않고 계속 커지면 분산이 무한하다는 신호다.

    **정리.**

    | 조건 | 일치성 | 중심극한정리 | 실무 |
    |---|---|---|---|
    | $\operatorname{Var}<\infty$ | ✓ | ✓ | 표준 방법 |
    | $E|X|<\infty$, $\operatorname{Var}=\infty$ | ✓ | ✗ | 구간·검정 불가. 분위수 사용 |
    | $E|X|=\infty$ | ✗ | ✗ | 평균 자체가 무의미 |

    **실무 권고.** 꼬리가 두꺼워 보이는 자료에서는 **적률이 존재하는지 먼저 확인**한다. 로그-로그 생존함수 그림의 기울기로 꼬리 지수 $\alpha$를 어림하고, $\alpha \le 2$이면 평균 기반 추론을 접고 분위수나 강건 통계량으로 옮긴다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
$\hat\theta_n$이 일치추정량이면 $\hat\theta_n$의 **함수**도 일치한다는 결과(연속사상정리)를 이용해, 실무에서 자주 쓰는 유도 추정량 세 가지의 일치성을 논하라.

</div>

??? success "풀이"
    **연속사상정리.** $\hat\theta_n \xrightarrow{p}\theta$이고 $g$가 $\theta$에서 연속이면 $g(\hat\theta_n)\xrightarrow{p}g(\theta)$이다.

    **응용 1 — 표준편차.** $\hat\sigma^2 \xrightarrow{p}\sigma^2$이고 $\sqrt\cdot$가 $\sigma^2>0$에서 연속이므로 $\hat\sigma \xrightarrow{p}\sigma$다. **불편성은 물려받지 못하지만 일치성은 물려받는다.** 이것이 앞서 본 대비의 핵심이다.

    **응용 2 — 비율과 오즈.** $\hat p \xrightarrow{p} p$이고 $p \in (0,1)$에서 $p/(1-p)$가 연속이므로 오즈의 추정량도 일치한다. 다만 **$p=0$이나 $p=1$에서는 연속이 아니므로** 경계에서는 이 논증이 통하지 않는다. 실제로 $\hat p=1$이면 오즈가 무한대가 된다.

    **응용 3 — 회귀에서 파생된 양.** $\hat{\boldsymbol\beta}$가 일치하면 예측값 $\mathbf{x}_0^\top\hat{\boldsymbol\beta}$, $R^2$, 두 계수의 비 $\hat\beta_1/\hat\beta_2$(단 $\beta_2\ne0$)가 모두 일치한다. **비에서는 분모가 0이 아니어야** 한다는 조건이 결정적이며, 도구변수 추정에서 약한 도구 문제가 생기는 이유가 이것이다.

    **주의할 점 두 가지.**

    1. **연속성이 필요하다.** 불연속점에서는 성립하지 않는다. 예컨대 $g(\theta) = \mathbb{1}\{\theta>0\}$은 $\theta=0$에서 불연속이라, 참값이 정확히 0이면 $g(\hat\theta_n)$이 수렴하지 않는다. 모형 선택처럼 "유의하면 넣는다" 식의 불연속 규칙이 불안정한 이유와 맞닿아 있다.

    2. **일치성만 물려받는다.** 편향, 표준오차, 신뢰구간은 따로 계산해야 한다. $g(\hat\theta)$의 표준오차에는 델타 방법이 필요하고, 구간은 $g$가 단조이면 양끝을 옮기면 되지만 아니면 더 복잡하다.

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
