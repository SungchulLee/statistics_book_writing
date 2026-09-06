# MLE의 점근적 성질

## 점근이론이 중요한 이유

유한표본에서 MLE의 거동은 구체적인 모형과 자료에 달려 있다. 그러나 일반적인 정칙 조건 아래에서 MLE는 표본크기 $n$이 커질 때 강력한 성질 셋을 갖는다. 참 모수값으로 수렴하고(일치성), 근사적으로 정규분포를 따르게 되며(점근정규성), 가능한 최선의 정밀도를 달성한다(점근 효율성). 이 성질들이 MLE를 기본 추정 전략으로 삼는 것을 정당화하고, 가능도에 기반한 신뢰구간, 가설검정, 모형 비교 도구의 이론적 근거가 된다.

이 페이지에서 $\theta_0$은 **참 모수값**, 즉 실제로 자료를 만들어 낸 값을 나타낸다. $X_1, \ldots, X_n \overset{\text{iid}}{\sim} f(x; \theta_0)$을 가정한다.

## 정칙 조건

아래의 점근적 결과들은 일련의 정칙 조건에 의존한다. 이 조건들은 표준 이론이 적용될 만큼 가능도함수가 잘 행동하도록 보장한다.

1. **식별 가능성.** 서로 다른 모수값이 서로 다른 분포를 만든다. $\theta_1 \neq \theta_2$이면 적어도 하나의 $x$에서 $f(x; \theta_1) \neq f(x; \theta_2)$이다.

2. **공통 지지집합.** 집합 $\{x : f(x; \theta) > 0\}$이 $\theta$에 의존하지 않는다. 모수에 따라 지지집합이 바뀌는 $\text{Uniform}(0, \theta)$ 같은 분포는 제외된다.

3. **내부의 모수.** 참값 $\theta_0$이 경계가 아니라 모수공간 $\Theta$의 내부에 있다.

4. **매끄러움.** 로그밀도 $\log f(x; \theta)$가 $\theta$에 대해 적어도 세 번 연속미분가능하고, 도함수를 적분(또는 합) 기호 안으로 넘길 수 있다.

5. **양의 Fisher 정보량.** Fisher 정보량이 $I(\theta_0) > 0$이어서 자료가 실제로 $\theta$에 관한 정보를 담고 있다.

!!! warning "정칙 조건이 깨질 때"

    정칙 조건이 위배되면 표준 점근 결과가 완전히 무너질 수 있다. 예를 들어 $\text{Uniform}(0, \theta)$에서 $\theta$의 MLE는 $X_{(n)}$(표본 최댓값)인데, 통상적인 $1/\sqrt{n}$이 아니라 $1/n$의 속도로 수렴하고 극한분포가 정규가 아니라 지수분포이다.

## 일치성

좋은 추정량이라면 자료를 더 모을수록 참값으로 수렴해야 한다. **일치성**은 이 최소한의 요구를 형식화한다.

위의 정칙 조건 아래에서 MLE는 일치한다:

$$
\hat{\theta}_{\text{MLE}} \xrightarrow{P} \theta_0 \quad \text{as } n \to \infty
$$

이는 임의의 $\epsilon > 0$에 대해 다음이 성립함을 뜻한다:

$$
P(|\hat{\theta}_{\text{MLE}} - \theta_0| > \epsilon) \to 0 \quad \text{as } n \to \infty
$$

증명은 로그가능도비 $\ell(\theta) - \ell(\theta_0)$이 Kullback-Leibler 발산 $-\text{KL}(f_{\theta_0} \| f_\theta)$으로 균등수렴하고, 식별 가능성에 의해 이것이 $\theta = \theta_0$에서 유일하게 최대가 된다는 사실에 의존한다.

!!! tip "일치성은 필요하지만 충분하지는 않다"

    일치성만으로는 추정량이 얼마나 빨리 수렴하는지, 그 분포가 어떤 모양인지 알 수 없다. 일치하는 두 추정량이 유한표본에서 매우 다르게 거동할 수 있다. 다음의 두 성질이 속도와 분포 모양을 다룬다.

## 점근정규성

MLE의 점근적 성질 중 실무적으로 가장 중요한 것은 $n$이 클 때 표본분포가 근사적으로 정규가 된다는 점이다.

정칙 조건 아래에서 MLE는 다음을 만족한다:

$$
\sqrt{n}(\hat{\theta}_{\text{MLE}} - \theta_0) \xrightarrow{d} N\left(0, \frac{1}{I(\theta_0)}\right)
$$

동등하게, $n$이 크면:

$$
\hat{\theta}_{\text{MLE}} \overset{\text{approx}}{\sim} N\left(\theta_0, \frac{1}{nI(\theta_0)}\right)
$$

이 결과는 곧바로 실용적인 귀결을 낳는다. $\theta_0$에 대한 근사적인 $(1 - \alpha)$ 신뢰구간은

$$
\hat{\theta}_{\text{MLE}} \pm z_{\alpha/2} \cdot \frac{1}{\sqrt{nI(\hat{\theta}_{\text{MLE}})}}
$$

여기서 $z_{\alpha/2}$는 표준정규분포의 $(1 - \alpha/2)$ 분위수이다. $\theta_0$을 모르므로 실무에서는 Fisher 정보량을 $\hat{\theta}_{\text{MLE}}$에서 평가한다.

!!! example "Bernoulli 모수의 점근 신뢰구간"

    $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Bernoulli}(p)$에서 MLE는 $\hat{p} = \bar{X}$이고 Fisher 정보량은 $I(p) = 1/(p(1-p))$이다. 근사적인 95% 신뢰구간은

    $$
    \hat{p} \pm 1.96 \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
    $$

    $n = 100$이고 $\hat{p} = 0.3$이면 $0.3 \pm 1.96\sqrt{0.21/100} = 0.3 \pm 0.090$, 즉 근사적으로 $(0.210, 0.390)$이다.

## 점근 효율성

점근정규성은 MLE가 점근분산 $1/(nI(\theta_0))$으로 $1/\sqrt{n}$의 속도로 수렴한다고 알려 준다. 그런데 이 분산이 우리가 달성할 수 있는 최선인가? 답은 그렇다이다.

정칙추정량 중에서 MLE는 점근적으로 **Cramér-Rao 하한**을 달성한다:

$$
\text{Var}_{\text{asy}}(\hat{\theta}_{\text{MLE}}) = \frac{1}{nI(\theta_0)}
$$

이는 다른 어떤 정칙 일치추정량도 이보다 작은 점근분산을 가질 수 없음을 뜻한다. 일치하고 점근적으로 정규인 다른 추정량 $\tilde{\theta}$는 반드시 다음을 만족한다:

$$
\text{Var}_{\text{asy}}(\tilde{\theta}) \geq \frac{1}{nI(\theta_0)}
$$

등호는 $\tilde{\theta}$가 MLE와 점근적으로 동등할 때에만 성립한다.

!!! note "초효율성"

    모수공간의 고립된 점에서 MLE보다 분산이 작은 추정량을 구성할 수 있다(Hodges 추정량이 고전적인 예이다). 그러나 그런 추정량은 다른 점에서 반드시 분산이 더 커진다. MLE는 정칙추정량 부류 안에서 모수공간 전체에 걸쳐 일률적으로 효율적이다.

## 세 가지 성질 요약

| 성질 | 서술 | 실무적 함의 |
|---|---|---|
| 일치성 | $\hat{\theta}_{\text{MLE}} \xrightarrow{P} \theta_0$ | 추정값이 참값으로 수렴한다 |
| 점근정규성 | $\sqrt{n}(\hat{\theta}_{\text{MLE}} - \theta_0) \xrightarrow{d} N(0, 1/I(\theta_0))$ | 정규근사로 신뢰구간과 검정을 얻는다 |
| 점근 효율성 | $\text{Var}_{\text{asy}} = 1/(nI(\theta_0))$ | 어떤 정칙추정량도 더 나을 수 없다 |

이 세 결과를 종합하면, 정칙 조건이 성립하고 점근근사가 믿을 만할 만큼 표본이 크다면 MLE가 모수적 추정의 기본 선택이 된다.

## 연습문제

**연습문제 1.**
정칙 조건 아래에서 MLE가 갖는 세 가지 주요 점근적 성질인 일치성, 점근정규성, 점근 효율성을 서술하라.

??? success "연습문제 1 풀이"
    표준적인 정칙 조건 아래에서:

    1. **일치성:** $n \to \infty$일 때 $\hat{\theta}_{\text{MLE}} \xrightarrow{p} \theta_0$.

    2. **점근정규성:** $\sqrt{n}(\hat{\theta}_{\text{MLE}} - \theta_0) \xrightarrow{d} N(0, I(\theta_0)^{-1})$이며, $I(\theta_0)$은 관측값 하나에 대한 Fisher 정보량이다.

    3. **점근 효율성:** MLE는 점근적으로 Cramér-Rao 하한을 달성한다. 즉 어떤 일치추정량도 $1/(nI(\theta_0))$보다 작은 점근분산을 가질 수 없다.

    이 성질들 덕분에 MLE는 대표본 모수적 추정의 기본 선택이 된다. 다만 소표본에서는 MLE가 편향되거나 다른 추정량보다 효율이 낮을 수 있다.

---

**연습문제 2.**
비율 모수가 $\lambda$인 Exponential 분포에서 MLE는 $\hat{\lambda} = 1/\bar{X}$이다. Fisher 정보량 $I(\lambda)$를 계산하고 $\hat{\lambda}$의 점근분산이 $\lambda^2/n$임을 확인하라.

??? success "연습문제 2 풀이"
    관측값 하나에 대한 로그가능도는 $\ell(\lambda) = \log\lambda - \lambda x$이다. 2계도함수는:

    $$
    \frac{d^2\ell}{d\lambda^2} = -\frac{1}{\lambda^2}
    $$

    Fisher 정보량은:

    $$
    I(\lambda) = -E\!\left[\frac{d^2\ell}{d\lambda^2}\right] = \frac{1}{\lambda^2}
    $$

    MLE의 점근분산은:

    $$
    \text{Var}(\hat{\lambda}) \approx \frac{1}{nI(\lambda)} = \frac{\lambda^2}{n}
    $$

    따라서 $n$이 크면 $\hat{\lambda} \approx N(\lambda, \lambda^2/n)$이고 표준오차는 $\text{SE}(\hat{\lambda}) \approx \lambda/\sqrt{n}$이다.

---

**연습문제 3.**
불변성은 $\hat{\theta}$가 $\theta$의 MLE이면 임의의 함수 $g$에 대해 $g(\hat{\theta})$가 $g(\theta)$의 MLE라는 성질이다. 이를 이용하여 Exponential 분포의 평균 $1/\lambda$의 MLE를 구하라.

??? success "연습문제 3 풀이"
    비율 모수의 MLE는 $\hat{\lambda} = 1/\bar{X}$이다. 불변성에 의해 (모평균인) $g(\lambda) = 1/\lambda$의 MLE는:

    $$
    \widehat{1/\lambda} = g(\hat{\lambda}) = \frac{1}{\hat{\lambda}} = \frac{1}{1/\bar{X}} = \bar{X}
    $$

    Exponential 분포 평균의 MLE가 표본평균이라는 직관적인 결과가 확인된다. 불변성은 MLE를 처음부터 다시 유도하지 않고도 비선형 변환을 포함한 임의의 변환에 대해 작동하므로 강력하다.

---

**연습문제 4.**
MLE가 점근적으로는 불편인데도 유한표본에서 편향될 수 있는 이유를 설명하라. 구체적인 예를 들라.

??? success "연습문제 4 풀이"
    점근적 불편성은 $n \to \infty$일 때 $E[\hat{\theta}_n] \to \theta_0$임을 뜻하지만, 고정된 $n$에 대해서는 편향 $E[\hat{\theta}_n] - \theta_0$이 0이 아닐 수 있다.

    **예:** 정규분포의 분산에서 MLE는 $\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum(X_i - \bar{X})^2$이며 다음을 만족한다:

    $$
    E[\hat{\sigma}^2_{\text{MLE}}] = \frac{n-1}{n}\sigma^2 \neq \sigma^2
    $$

    편향은 $-\sigma^2/n$으로 $n \to \infty$일 때 사라지지만 모든 유한한 $n$에서는 0이 아니다. 이 때문에 불편추정량 $S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$이 Bessel 수정 $n - 1$을 사용한다.

    일반적으로 $\hat{\theta}$가 $\theta$의 MLE이고 $g$가 비선형이면 불변성에 의해 $g(\hat{\theta})$가 $g(\theta)$의 MLE이지만, Jensen 부등식에 의해 유한표본에서 $E[g(\hat{\theta})] \neq g(\theta)$가 되어 편향이 생긴다.
