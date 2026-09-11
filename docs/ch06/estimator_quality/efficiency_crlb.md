# 효율성과 Cramér-Rao 하한

## 왜 분산의 하한인가?

모수적 모형이 주어지면 모수 $\theta$에 대한 불편추정량이 여럿 존재할 수 있다. 어떤 것은 다른 것보다 분산이 작을 텐데, 자연스러운 물음이 떠오른다. 불편추정량의 분산은 얼마나 작아질 수 있는가? **Cramér-Rao 하한(CRLB)**은 임의의 불편추정량의 분산에 근본적인 바닥을 설정함으로써 이 물음에 답한다. 어떤 특정 추정량의 품질을 재는 기준선을 제공하므로 추정이론의 중심이 되는 결과이다.

## Fisher 정보량

CRLB를 서술하기 전에, 하나의 확률적 관측값이 미지의 모수에 관해 얼마나 많은 정보를 담고 있는지를 정량화하는 Fisher 정보량 개념이 필요하다.

$X$를 확률밀도함수 또는 확률질량함수가 $f(x; \theta)$인 확률변수라 하고 $\theta$를 관심 모수라 하자. **점수함수**는 로그가능도를 $\theta$에 대해 편미분한 것이다:

$$
s(X; \theta) = \frac{\partial}{\partial \theta} \log f(X; \theta)
$$

정칙 조건($f$의 지지집합이 $\theta$에 의존하지 않고 적분기호 아래에서의 미분이 타당함) 아래에서 점수함수의 평균은 0이다: $E[s(X; \theta)] = 0$.

관측값 하나에 대한 **Fisher 정보량**은 점수함수의 분산으로 정의된다:

$$
I(\theta) = E\left[\left(\frac{\partial}{\partial \theta} \log f(X;\theta)\right)^2\right]
$$

미분과 적분의 교환을 허용하는 같은 정칙 조건 아래에서, 이는 로그가능도의 2계도함수의 기댓값에 음수를 취한 것과 같다:

$$
I(\theta) = -E\left[\frac{\partial^2}{\partial\theta^2} \log f(X;\theta)\right]
$$

두 번째 형태가 계산하기 더 쉬운 경우가 많다. 직관적으로 Fisher 정보량이 크다는 것은 참 모수값 주위에서 로그가능도가 급하게 휘어 있다는 뜻이며, 그만큼 $\theta$를 정밀하게 추정하기 쉽다.

확률표본 $X_1, \ldots, X_n \overset{\text{iid}}{\sim} f(x; \theta)$에 대해 전체 Fisher 정보량은 $n I(\theta)$이며, 이는 독립인 관측값들이 정보에 가법적으로 기여한다는 사실을 반영한다.

## Cramér-Rao 부등식

Fisher 정보량을 손에 넣었으니 근본적인 결과를 서술할 수 있다.

<div class="thmbox" markdown>

### 정리 1. (Cramér-Rao 하한) { .thm }


$X_1, \ldots, X_n$을 다음 정칙 조건을 만족하는 밀도함수 또는 질량함수 $f(x; \theta)$를 갖는 i.i.d. 확률변수라 하자:

1. 지지집합 $\{x : f(x; \theta) > 0\}$이 $\theta$에 의존하지 않는다.
2. 도함수 $\frac{\partial}{\partial \theta} f(x; \theta)$와 $\frac{\partial^2}{\partial \theta^2} f(x; \theta)$가 존재하고 연속이다.
3. 미분과 적분(또는 합)을 교환할 수 있다.

그러면 임의의 불편추정량 $\hat{\theta} = \hat{\theta}(X_1, \ldots, X_n)$에 대해:

$$
\text{Var}(\hat{\theta}) \geq \frac{1}{n I(\theta)}
$$

</div>

$1 / (nI(\theta))$가 Cramér-Rao 하한이다. 어떤 불편추정량도 이 문턱 아래의 분산을 가질 수 없다.

!!! example "정규분포 평균의 CRLB"

    $\sigma^2$이 알려진 $X_1, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$이라 하자. 관측값 하나에 대한 로그가능도는

    $$
    \log f(x; \mu) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x - \mu)^2}{2\sigma^2}
    $$

    $\mu$에 대해 2계도함수를 취하면:

    $$
    \frac{\partial^2}{\partial \mu^2} \log f(x; \mu) = -\frac{1}{\sigma^2}
    $$

    따라서 $I(\mu) = 1/\sigma^2$이고, $n$개 관측값에 대한 CRLB는

    $$
    \text{Var}(\hat{\mu}) \geq \frac{1}{n \cdot (1/\sigma^2)} = \frac{\sigma^2}{n}
    $$

    $\text{Var}(\bar{X}) = \sigma^2 / n$이므로 표본평균 $\bar{X}$가 CRLB를 정확히 달성한다.

## 효율성

CRLB는 자연스럽게 불편추정량의 순위를 매기는 방법으로 이어진다. 하한을 달성하는 불편추정량은 가능한 한 최선이다. 자료에서 모든 정보를 뽑아내기 때문이다.

불편추정량 $\hat{\theta}$가 모든 $\theta$에 대해 다음을 만족하면 **효율적**이라 한다:

$$
\text{Var}(\hat{\theta}) = \frac{1}{n I(\theta)}
$$

불편추정량의 **효율**은 CRLB를 실제 분산으로 나눈 비이다:

$$
e(\hat{\theta}) = \frac{1 / (nI(\theta))}{\text{Var}(\hat{\theta})} \leq 1
$$

효율이 1이면 그 추정량은 효율적이다. 1보다 작은 값은 이론적 최적에 비해 얼마나 많은 분산이 "낭비"되었는지를 나타낸다.

!!! example "표본평균의 효율 (정규분포)"

    위의 예에서 $\bar{X}$의 분산은 $\sigma^2/n$이고 CRLB도 $\sigma^2/n$이므로

    $$
    e(\bar{X}) = \frac{\sigma^2/n}{\sigma^2/n} = 1
    $$

    표본평균은 (분산이 알려진) 정규분포 평균의 효율적 추정량이다.

!!! example "표본중앙값의 효율 (정규분포)"

    $X_1, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$에서 표본중앙값도 $\mu$의 불편추정량이지만 점근분산은 $\pi \sigma^2 / (2n)$이다. 따라서 점근 효율은

    $$
    e(\text{median}) = \frac{\sigma^2/n}{\pi\sigma^2/(2n)} = \frac{2}{\pi} \approx 0.637
    $$

    표본중앙값은 정규분포 자료에서 표본평균이 뽑아내는 정보의 약 64%만 사용한다.

## CRLB를 달성할 수 없는 경우

모든 모수적 모형에 효율적 추정량이 존재하는 것은 아니다. CRLB를 달성할 수 있을 필요충분조건은 점수함수를 다음 형태로 쓸 수 있는 것이다:

$$
\frac{\partial}{\partial \theta} \log f(x; \theta) = a(\theta)\left[T(x) - \theta\right]
$$

여기서 $a(\theta)$는 어떤 함수이고 $T(x)$는 통계량이다. 이 조건은 지수족 분포에서 만족되지만 다른 많은 모형에서는 성립하지 않는다.

!!! warning "균등분포"

    $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Uniform}(0, \theta)$라 하자. 지지집합 $\{x : 0 < x < \theta\}$가 $\theta$에 의존하므로 첫 번째 정칙 조건을 위반한다. CRLB가 적용되지 않으며, 실제로 MLE $\hat{\theta} = X_{(n)}$(표본 최댓값)의 분산은 $1/n^2$의 속도로 줄어들어 CRLB가 허용할 $1/n$ 속도보다 빠르다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
$X_1, \ldots, X_n \sim N(\mu, \sigma^2)$이고 추정량이 $\hat\mu_w = wX_1 + (1-w)\bar X$이다. (a) 불편인가? (b) 분산을 최소화하는 $w$는? (c) $w = 0$과 $w = 1$에서의 분산은?

</div>

??? success "풀이"
    (a) 선형성에 의해 $\mathbb{E}[\hat\mu_w] = w\mu + (1-w)\mu = \mu$이다. 모든 $w$에 대해 불편이다.

    (b) $\bar X_{-1}$을 ($X_1$과 독립인) $X_2, \ldots, X_n$의 평균이라 하고 $\bar X = X_1/n + (n-1)\bar X_{-1}/n$으로 쓰면:

    $\hat\mu_w = (w + (1-w)/n) X_1 + ((1-w)(n-1)/n) \bar X_{-1}$

    분산은 두 항 모두에 양의 계수를 가지며, $w$에 대해 미분하여 0으로 두면 $w^* = 0$을 얻는다.

    (c) $w = 0$: $\hat\mu = \bar X$, $\mathrm{Var} = \sigma^2/n$. $w = 1$: $\hat\mu = X_1$, $\mathrm{Var} = \sigma^2$. $w = 1$의 대가는 분산이 $n$배가 되는 것이다.

    표본평균이 관측값 하나에 의존하는 것보다 $n$배 효율적이다.

<div class="drillbox" markdown>

**연습문제 2.**
**Cramér-Rao 하한 (CRLB).** 불편추정량에 대한 CRLB를 서술하고 증명하라.

</div>

??? success "풀이"
    **CRLB:** $\hat\theta$가 밀도 $f(x; \theta)$에서 얻은 $n$개의 i.i.d. 관측값에 기반한 $\theta$의 불편추정량이면:

    $$
    \mathrm{Var}(\hat\theta) \ge \frac{1}{n I(\theta)}
    $$

    여기서 $I(\theta) = -\mathbb{E}[\partial^2 \log f/\partial \theta^2]$는 관측값 하나당 Fisher 정보량이다.

    **증명 개요:** Cauchy-Schwarz에 의해 $|\mathrm{Cov}(\hat\theta, S)|^2 \le \mathrm{Var}(\hat\theta) \mathrm{Var}(S)$이며, $S = \partial \log L/\partial \theta$는 **점수함수**이다. (불편성에서 $\mathbb{E}[\hat\theta] = \theta$를 미분하여) $\mathrm{Cov}(\hat\theta, S) = 1$이고 $\mathrm{Var}(S) = n I(\theta)$임을 계산한다. 따라서 $1 \le \mathrm{Var}(\hat\theta) \cdot n I(\theta)$이고 하한을 얻는다.

    등호는 $\hat\theta$가 점수함수의 선형함수일 때, 즉 추정량이 **효율적**일 때에만 성립한다.

<div class="drillbox" markdown>

**연습문제 3.**
**정규분포 평균의 효율성.** $X \sim N(\mu, \sigma^2)$에서 $\mu$를 추정할 때 $\bar X$가 CRLB를 달성함을 보여라.

</div>

??? success "풀이"
    정규분포 평균의 Fisher 정보량은 $I(\mu) = 1/\sigma^2$이다. CRLB는 $\mathrm{Var}(\hat\mu) \ge \sigma^2/n$이다.

    표본평균의 분산은 $\mathrm{Var}(\bar X) = \sigma^2/n$으로 CRLB를 정확히 달성한다.

    따라서 $\bar X$는 $\mu$에 대한 **일률최소분산불편추정량(UMVUE)**이다. 어떤 $\mu$에서도 이보다 분산이 작은 불편추정량은 없다.

    이는 강한 최적성 결과이다. $\bar X$는 단지 일치하고 불편인 데 그치지 않고, 정규 모형 아래에서 *최선의* 추정량이다.

<div class="drillbox" markdown>

**연습문제 4.**
**MLE의 점근 효율성.** MLE가 점근적으로 CRLB를 달성한다는 결과를 서술하고 설명하라.

</div>

??? success "풀이"
    **결과:** 정칙 조건 아래에서 $\hat\theta_{\text{MLE}}$는 분산이 CRLB와 같은 **점근적 정규**이다:

    $$
    \sqrt n (\hat\theta_{\text{MLE}} - \theta) \xrightarrow{d} N(0, 1/I(\theta))
    $$

    따라서 $\mathrm{Var}(\hat\theta_{\text{MLE}}) \to 1/(n I(\theta))$로, 관측값 하나당 Fisher 정보량의 역수가 된다. 이것이 불편추정량의 CRLB이다.

    **함의:** MLE는 **점근적으로 효율적**이다. 모든 일치추정량 중에서 MLE의 점근분산이 가장 작다.

    **주의:** 유한표본에서의 효율은 나쁠 수 있다. MLE는 편향될 수도, 존재하지 않을 수도, 느리게 수렴할 수도 있다. 점근 효율성은 장기적인 보장이지 유한표본에 대한 보장이 아니다.

    소표본에서는 다른 추정량(적률법, 편향 보정된 MLE, James-Stein 계열의 축소)이 MLE를 능가할 수 있다. $n$이 크면 MLE가 사실상 최적이다.

<div class="drillbox" markdown>

**연습문제 5.**
**비효율적인 불편추정량.** $N(\mu, 1)$에서 분산이 $1/n$(CRLB)보다 큰 $\mu$의 불편추정량을 구성하라. 그런 추정량이 존재하는 이유를 설명하라.

</div>

??? success "풀이"
    예: $\hat\mu = X_1$(첫 관측값만 사용). 불편이다: $\mathbb{E}[X_1] = \mu$. 분산: $\mathrm{Var}(X_1) = 1$.

    $n \ge 2$이면 $\mathrm{Var}(X_1) = 1 > 1/n =$ CRLB이다. 따라서 이 추정량은 $n$배 비효율적이다.

    **그런 추정량이 존재하는 이유:** CRLB는 *하한*이며 많은 추정량이 그 위에 있다. CRLB는 어떤 불편추정량도 그 **아래**로 갈 수 없다고 말할 뿐이다. ($X_2, \ldots, X_n$을 무시하는 $X_1$처럼) 정보를 버리는 추정량은 분명히 최적이 아니다.

    다른 예: $(X_1 + X_2)/2$(관측값 두 개만 사용), $X_{(n)}$(최댓값이며 특정 분포에서만 불편이다).

    **효율**은 불편추정량의 분산이 CRLB에 얼마나 가까운지를 잰다. $\bar X$의 효율은 1이고(CRLB 달성), 크기 $n$인 정규 표본에서 $X_1$의 효율은 $1/n$이다.

<div class="drillbox" markdown>

**연습문제 6.**
**실제 응용에서의 맞바꿈.** 언제 불편추정량보다 편향되어 있지만 평균제곱오차가 작은 추정량(예: 능형회귀, James-Stein)을 선호하겠는가?

</div>

??? success "풀이"
    **MSE 분해:** $\mathrm{MSE}(\hat\theta) = \mathrm{Var}(\hat\theta) + [\mathrm{Bias}(\hat\theta)]^2$.

    불편추정량은 $\mathrm{Bias} = 0$이므로 $\mathrm{MSE} = \mathrm{Var}$이고, CRLB가 이를 아래에서 제한한다.

    어느 정도의 편향을 받아들이면 편향의 제곱이 더하는 것보다 분산이 더 많이 줄어들 수 있고, 그러면 **평균제곱오차가 내려간다**.

    **예:**

    - **능형회귀:** 0 쪽으로 편향되지만 큰 계수에 벌점을 주어 분산을 줄인다. 설명변수가 상관된 상황에서 평균제곱오차가 더 작다.
    - **James-Stein** ($p \ge 3$개의 정규 평균 추정): 개별 표본평균을 0 쪽으로 축소하여 결합하며, 참 모수가 무엇이든 평균제곱오차에서 MLE를 지배한다.
    - **정보가 있는 사전분포를 쓴 MAP:** 사전 평균 쪽으로 편향되지만 정칙화 덕분에 분산이 작다. 사전분포가 대체로 옳으면 평균제곱오차가 더 작다.

    **선호할 때:**

    - 표본크기가 작을 때(높은 분산이 지배한다).
    - 자료에 비해 모수가 많을 때(고차원 회귀).
    - 예측이 목적일 때(불편성보다 평균제곱오차가 중요하다).
    - 사전 정보를 쓸 수 있을 때(베이즈 상황).

    **불편추정량을 선호할 때:**

    - 추정값의 해석이 중요할 때(예: 인과효과).
    - 표본크기가 클 때(분산은 줄지만 편향은 남는다).
    - 규제나 학문적 관행이 불편성을 요구할 때.

    편향–분산 맞바꿈은 근본적인 긴장이며, 현대 통계적 학습은 평균제곱오차에서 이기기 위해 편향을 받아들이는 일이 일상적이다.
