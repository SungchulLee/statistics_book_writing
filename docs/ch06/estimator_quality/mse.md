# 평균제곱오차

## 소개

**평균제곱오차(MSE)**는 통계적 추정량의 품질을 평가하는 데 가장 널리 쓰이는 기준이다. 추정량이 참 모수값에서 벗어난 제곱편차의 평균을 재며, 체계적 오차(편향)와 무작위 요동(분산)을 하나의 양에 함께 담는다.

평균제곱오차는 추정이론, 회귀분석, 그리고 통계학과 계량금융 전반의 많은 최적화 문제에서 기본 손실함수 역할을 한다.

## 평균제곱오차

<div class="defn" markdown>

**정의 1.** [평균제곱오차]

$\hat{\theta}$를 모수 $\theta$의 추정량이라 하자. **평균제곱오차**는:

$$\text{MSE}(\hat{\theta}) = E\left[(\hat{\theta} - \theta)^2\right]$$

이는 추정량과 참 모수의 차를 제곱한 값을 가능한 모든 표본에 걸쳐 평균한 것이다.

</div>

### 동등한 표현

평균제곱오차는 여러 동등한 방식으로 계산할 수 있다:

$$\text{MSE}(\hat{\theta}) = E[\hat{\theta}^2] - 2\theta E[\hat{\theta}] + \theta^2$$

$$= \text{Var}(\hat{\theta}) + [E[\hat{\theta}]]^2 - 2\theta E[\hat{\theta}] + \theta^2$$

$$= \text{Var}(\hat{\theta}) + (E[\hat{\theta}] - \theta)^2$$

$$= \text{Var}(\hat{\theta}) + [\text{Bias}(\hat{\theta})]^2$$

마지막 형태가 **편향–분산 분해**이다.

## 평균제곱오차의 성질

### 비음성

평균제곱오차는 언제나 음이 아니다: $\text{MSE}(\hat{\theta}) \geq 0$이며, 등호는 확률 1로 $\hat{\theta} = \theta$인 경우(추정량이 완벽한 경우)에만 성립한다.

### 불편추정량의 평균제곱오차

$\hat{\theta}$가 불편이면($\text{Bias}(\hat{\theta}) = 0$):

$$\text{MSE}(\hat{\theta}) = \text{Var}(\hat{\theta})$$

불편추정량에서는 평균제곱오차와 분산이 같다. 불편추정량을 평균제곱오차로 비교하는 것은 분산으로 비교하는 것과 동등하다.

### 일치성과 평균제곱오차

$n \to \infty$일 때 $\text{MSE}(\hat{\theta}_n) \to 0$이면 추정량이 **평균제곱오차 일치**라고 한다. 분해에 의해 다음 둘이 모두 필요하다:

- $\text{Bias}(\hat{\theta}_n) \to 0$
- $\text{Var}(\hat{\theta}_n) \to 0$

Chebyshev 부등식에 의해 평균제곱오차 일치성은 확률적 일치성($\theta$로의 확률수렴)을 함의한다.

## 추정량 사이의 평균제곱오차 비교

### 상대효율

추정량 $\hat{\theta}_1$의 $\hat{\theta}_2$에 대한 **상대효율**은:

$$\text{RE}(\hat{\theta}_1, \hat{\theta}_2) = \frac{\text{MSE}(\hat{\theta}_2)}{\text{MSE}(\hat{\theta}_1)}$$

$\text{RE} > 1$이면 $\hat{\theta}_1$이 더 효율적이다(평균제곱오차가 더 작다).

불편추정량에서는 다음과 같이 간단해진다:

$$\text{RE}(\hat{\theta}_1, \hat{\theta}_2) = \frac{\text{Var}(\hat{\theta}_2)}{\text{Var}(\hat{\theta}_1)}$$

### 허용성

다음을 만족하는 다른 추정량 $\hat{\theta}'$이 존재하면 추정량 $\hat{\theta}$는 평균제곱오차 아래에서 **비허용**이라 한다:

$$\text{MSE}(\hat{\theta}') \leq \text{MSE}(\hat{\theta}) \quad \text{for all } \theta$$

이때 적어도 하나의 $\theta$에서는 부등호가 엄격해야 한다. 비허용이 아닌 추정량을 **허용** 추정량이라 한다.

**James-Stein 결과:** $p \geq 3$인 다변량 정규분포의 평균 $\mu \in \mathbb{R}^p$를 추정할 때 표본평균 $\bar{X}$는 비허용이다. James-Stein 추정량이 평균제곱오차에서 일률적으로 그것을 지배한다.

## 보기

<div class="exbox" markdown>

**보기 1.** 표본평균의 평균제곱오차. $X_1, \ldots, X_n$을 평균 $\mu$, 분산 $\sigma^2$인 i.i.d. 확률변수라 하자. 표본평균은 $\bar{X} = \frac{1}{n}\sum X_i$이다.

</div>

**편향:** $E[\bar{X}] = \mu$이므로 $\text{Bias}(\bar{X}) = 0$ (불편).

**분산:** $\text{Var}(\bar{X}) = \sigma^2/n$.

**MSE:** $\text{MSE}(\bar{X}) = 0 + \sigma^2/n = \sigma^2/n$.

<div class="exbox" markdown>

**보기 2.** 소박한 분산추정량의 평균제곱오차. 소박한 분산추정량은 $\tilde{S}^2 = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2$이다.

정규모집단에 대해:

</div>

**편향:** $E[\tilde{S}^2] = \frac{n-1}{n}\sigma^2$이므로 $\text{Bias}(\tilde{S}^2) = -\sigma^2/n$.

**분산:** $\text{Var}(\tilde{S}^2) = \frac{2(n-1)}{n^2}\sigma^4$.

**MSE:**

$$\text{MSE}(\tilde{S}^2) = \frac{2(n-1)}{n^2}\sigma^4 + \frac{\sigma^4}{n^2} = \frac{2n-1}{n^2}\sigma^4$$

<div class="exbox" markdown>

**보기 3.** 편향 분산추정량과 불편 분산추정량의 비교. Bessel 수정된 추정량은 $S^2 = \frac{1}{n-1}\sum (X_i - \bar{X})^2$이다.

정규모집단에 대해:

</div>

**불편 $S^2$의 MSE:** $\text{MSE}(S^2) = \text{Var}(S^2) = \frac{2\sigma^4}{n-1}$

**편향 $\tilde{S}^2$의 MSE:** $\text{MSE}(\tilde{S}^2) = \frac{(2n-1)\sigma^4}{n^2}$

비교하면 $\frac{2n-1}{n^2}$ 대 $\frac{2}{n-1}$이다.

교차곱하면 $(2n-1)(n-1)$ 대 $2n^2$, 즉 $2n^2 - 3n + 1$ 대 $2n^2$이다.

$n > 0$에서 $-3n + 1 < 0$이므로 $\text{MSE}(\tilde{S}^2) < \text{MSE}(S^2)$이다.

**편향추정량이 불편추정량보다 평균제곱오차가 작다!** 편향–분산 맞바꿈을 구체적으로 보여 주는 예이다. ($c \cdot \sum(X_i - \bar{X})^2$ 형태의 추정량 중에서) 평균제곱오차를 최소화하는 최적 추정량은 $n$이나 $n-1$이 아니라 $n+1$로 나눈다.

<div class="exbox" markdown>

**보기 4.** 평균제곱오차 최적 분산추정량. 상수 $c > 0$에 대해 $\hat{\sigma}^2_c = \frac{1}{c}\sum_{i=1}^n(X_i - \bar{X})^2$을 생각하자.

</div>

정규모집단에 대해:

$$\text{MSE}(\hat{\sigma}^2_c) = \left(\frac{n-1}{c} - 1\right)^2 \sigma^4 + \frac{2(n-1)}{c^2}\sigma^4$$

$c$에 대해 미분하여 0으로 두면:

$$c^* = n + 1$$

따라서 평균제곱오차가 최적인 추정량은 $n+1$로 나눈다:

$$\hat{\sigma}^2_{n+1} = \frac{1}{n+1}\sum_{i=1}^n (X_i - \bar{X})^2$$

이는 편향되어 있지만($\sigma^2$을 과소추정한다) $\tilde{S}^2$($n$으로 나눔)과 $S^2$($n-1$로 나눔) 둘 다보다 평균제곱오차가 작다.

## 다른 손실함수와의 연결

### 평균절대오차 (MAE)

$$\text{MAE}(\hat{\theta}) = E\left[|\hat{\theta} - \theta|\right]$$

평균절대오차는 평균제곱오차보다 이상점에 덜 민감하다. 다만 평균제곱오차가 수학적으로 다루기 쉽고 편향–분산 분해와 직접 연결된다.

### 위험함수

의사결정이론에서 $\text{MSE}(\hat{\theta})$는 제곱오차 손실 $L(\hat{\theta}, \theta) = (\hat{\theta} - \theta)^2$ 아래에서 $\hat{\theta}$의 **위험**이다:

$$R(\hat{\theta}, \theta) = E[L(\hat{\theta}, \theta)] = \text{MSE}(\hat{\theta})$$

### Cramér-Rao 하한

불편추정량에서 평균제곱오차(= 분산)는 아래로 **Cramér-Rao 한계**에 의해 유계이다:

$$\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}$$

여기서 $I(\theta) = -E\left[\frac{\partial^2}{\partial\theta^2}\log f(X;\theta)\right]$는 Fisher 정보량이다. 이 한계를 달성하는 불편추정량을 **효율적**이라 한다.

## 금융에서의 평균제곱오차

평균제곱오차는 계량금융 전반에 나타난다:

- **예측 평가**: 수익률, 변동성, 위험 예측을 비교하는 표준 지표이다. RMSE = $\sqrt{\text{MSE}}$는 오차를 대상과 같은 단위로 나타낸다.
- **추적오차**: 포트폴리오 수익률과 벤치마크 사이의 평균제곱오차는 체계적 이탈(편향)과 무작위 이탈(분산)을 함께 담아낸다.
- **모형 보정**: 모형이 함의하는 옵션 가격과 시장에서 관측된 가격 사이의 평균제곱오차가 변동성 모형 보정의 목적함수가 된다.
- **회귀분석**: OLS는 표본 내 평균제곱오차인 $\sum(y_i - \hat{y}_i)^2/n$을 최소화한다.

## 요약

평균제곱오차는 추정량의 품질을 평가하는 근본 기준이다. 분산과 편향의 제곱으로 분해되면서 추정에 내재한 맞바꿈을 드러내고, 경쟁하는 추정량들 사이에서 고르는 원리적인 틀을 제공한다. 불편성은 바람직하지만, 평균제곱오차는 최선의 추정량이 전체 오차를 최소화하는 것임을 일깨워 준다. 약간의 편향이 큰 분산 감소를 얻어 낼 만한 값어치를 할 수 있다.

## 주요 공식

| 양 | 공식 |
|----------|---------|
| MSE | $E[(\hat{\theta} - \theta)^2]$ |
| 분해 | $\text{Var}(\hat{\theta}) + [\text{Bias}(\hat{\theta})]^2$ |
| $\bar{X}$의 MSE | $\sigma^2 / n$ |
| 상대효율 | $\text{MSE}(\hat{\theta}_2) / \text{MSE}(\hat{\theta}_1)$ |
| Cramér-Rao 한계 | $\text{Var}(\hat{\theta}) \geq 1/I(\theta)$ |

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
$X_i \sim \mathrm{Uniform}(0, \theta)$가 i.i.d.이다. 두 추정량 $\hat\theta_1 = 2\bar X$와 $\hat\theta_2 = ((n+1)/n) X_{(n)}$을 생각하자. (a) 둘 다 불편인가? (b) 분산은? (c) 어느 쪽이 평균제곱오차가 더 작은가?

</div>

??? success "풀이"
    (a) $\mathbb{E}[X] = \theta/2$이므로 $\mathbb{E}[\hat\theta_1] = \theta$이다.

    $\mathbb{E}[X_{(n)}] = n\theta/(n+1)$이므로 $\mathbb{E}[\hat\theta_2] = ((n+1)/n) \cdot n\theta/(n+1) = \theta$이다.

    둘 다 불편이다. ✓

    (b) $\mathrm{Var}(\hat\theta_1) = 4 \mathrm{Var}(\bar X) = 4 \theta^2/(12n) = \theta^2/(3n)$.

    $\mathrm{Var}(X_{(n)}) = n\theta^2/[(n+1)^2(n+2)]$이므로 $\mathrm{Var}(\hat\theta_2) = \theta^2/[n(n+2)]$.

    (c) 둘 다 불편이므로 MSE = 분산이다. $\hat\theta_2$의 분산은 $\theta^2/[n(n+2)] = O(1/n^2)$이고 $\hat\theta_1$은 $\theta^2/(3n) = O(1/n)$이다.

    **모든 $n \ge 2$에서 $\hat\theta_2$의 평균제곱오차가 더 작으며, 수렴 속도도 (초)선형으로 더 빠르다.** 주목할 만한 예이다. (최댓값에 기반한) MLE 계열 추정량이 $\sqrt n$ 일치가 아니라 $n$ 일치이다. 균등분포의 끝점을 다룰 때는 표본 극단값이 표본평균보다 훨씬 많은 정보를 담는다.

<div class="drillbox" markdown>

**연습문제 2.**
**$S^2$의 평균제곱오차 최적 척도.** $X_i \sim N(\mu, \sigma^2)$에서 $c \sum(X_i - \bar X)^2$ 형태의 $\sigma^2$ 추정량 중 평균제곱오차를 최소화하는 $c$를 구하라.

</div>

??? success "풀이"
    $T = \sum(X_i - \bar X)^2$이라 하자. 그러면 $T \sim \sigma^2 \chi^2_{n-1}$이므로 $\mathbb{E}[T] = (n-1)\sigma^2$, $\mathrm{Var}(T) = 2(n-1)\sigma^4$이다.

    $\hat\sigma^2_c = c T$에 대해:

    $\mathbb{E}[\hat\sigma^2_c] = c(n-1)\sigma^2$, $\mathrm{Bias} = (c(n-1) - 1)\sigma^2$, $\mathrm{Var} = 2 c^2 (n-1) \sigma^4$.

    $\mathrm{MSE}(c) = 2 c^2(n-1) \sigma^4 + (c(n-1) - 1)^2 \sigma^4$.

    최소화하면: $d \mathrm{MSE}/dc = 4c(n-1)\sigma^4 + 2(c(n-1) - 1)(n-1)\sigma^4 = 0$.

    $4c + 2(c(n-1) - 1) = 0 \Rightarrow 2c + c(n-1) = 1 \Rightarrow c(n+1) = 1 \Rightarrow c^* = 1/(n+1)$.

    따라서 **$\hat\sigma^2_{\text{MSE}} = (1/(n+1))\sum(X_i - \bar X)^2$**이 평균제곱오차를 최소화한다. MLE($c = 1/n$)와 불편추정량($c = 1/(n-1)$) 사이에 있으면서 더 큰 분모, 즉 더 강한 축소 쪽으로 기울어 있다.

    "불편"인 $s^2$의 해석이 더 깔끔하기 때문에 실무에서 잘 쓰이지는 않는다. 다만 평균제곱오차가 최적인 추정량이 MLE와도 불편추정량과도 다를 수 있음을 보여 준다.

<div class="drillbox" markdown>

**연습문제 3.**
**편향추정량의 평균제곱오차.** $\hat\theta$가 편향되어 있고 $\mathbb{E}[\hat\theta] = \theta + b/n$, $\mathrm{Var}(\hat\theta) = v/n$이다. 평균제곱오차와 그 점근 거동을 구하라.

</div>

??? success "풀이"
    $\mathrm{MSE} = \mathrm{Var} + \mathrm{Bias}^2 = v/n + b^2/n^2$.

    $n$이 크면 $\mathrm{MSE} \approx v/n + O(1/n^2)$이다. 분산이 지배하고 편향은 저차항으로만 기여한다.

    점근적 일치성: $\mathrm{MSE} \to 0 \Rightarrow \hat\theta \to \theta$가 $L^2$에서, 따라서 확률적으로도 성립한다. 유한표본에서 편향이 있어도 추정량은 일치한다.

    **통찰:** $O(1/n)$ 편향은 점근적으로 "보이지 않는다". 잡음보다 빨리 사라지기 때문이다. 이것이 (보통 $O(1/n)$ 편향을 갖는) MLE가 점근적으로 효율적인 이유이다.

    $O(1)$ 편향(상수 추정량 $\hat\theta = c$처럼)을 갖는 추정량은 일치하지 않는다. 평균제곱오차의 편향 항이 줄지 않기 때문이다.

<div class="drillbox" markdown>

**연습문제 4.**
**Cramér-Rao와 평균제곱오차.** 불편추정량에서는 MSE = 분산이고 CRLB가 분산 $\ge 1/(n I(\theta))$를 준다. 편향추정량에 대응하는 결과를 서술하라.

</div>

??? success "풀이"
    **편향추정량의 CRLB:** 편향 여부와 무관하게 임의의 추정량 $\hat\theta$에 대해:

    $$
    \mathrm{Var}(\hat\theta) \ge \frac{(1 + b'(\theta))^2}{n I(\theta)}
    $$

    여기서 $b(\theta) = \mathbb{E}[\hat\theta] - \theta$는 편향함수이다.

    불편추정량에서는 $b' = 0$이므로 표준 CRLB가 복원된다.

    편향추정량에서 $b'(\theta) = -1$이면(예: 상수 추정량 $\hat\theta = c$) 한계가 0이 되며, 분산이 0인 상수 추정량이 자명하게 이를 달성한다.

    더 일반적으로 편향 CRLB는 MSE = 분산 + 편향$^2 \ge (1 + b')^2/(nI) + b^2$을 준다.

    실용적 의미: 축소추정량(ridge, James-Stein)은 분산을 줄이려고 일부러 편향을 들여와 평균제곱오차를 불편추정량의 CRLB 아래로 낮춘다. 불편추정량에서는 증명 가능하게 불가능한 일이지만 편향추정량에서는 흔한 일이다.

<div class="drillbox" markdown>

**연습문제 5.**
**실용적인 축소 예제.** $n$명을 조사한 여론조사에서 $\hat p = X/n$을 얻었다. 어떤 목표값 $p_0$(예: 0.5)에 대해 축소추정량 $\hat p_{\text{shr}} = w \hat p + (1 - w) p_0$을 생각하자. (참 $p$가 $p_0$과 같다고 가정하고) 최적의 $w$를 구하라.

</div>

??? success "풀이"
    참 $p = p_0$이면 $\hat p_{\text{shr}}$의 편향은 $w p_0 + (1 - w) p_0 - p_0 = 0$이다. 분산은 $w^2 \mathrm{Var}(\hat p) = w^2 p_0(1 - p_0)/n$이다.

    MSE $= w^2 p_0(1-p_0)/n$이며 $w = 0$에서 최소가 된다(즉 언제나 $p_0$으로 추정한다).

    **더 현실적인 경우:** 참 $p$가 불확실하다($\mathbb{E}[p] = p_0$, $\mathrm{Var}(p) = \tau^2$인 확률변수). 편향의 제곱은 $(1 - w)^2(p - p_0)^2$이고, $p$에 대한 기대 평균제곱오차는:

    $w^2 p_0(1-p_0)/n + (1-w)^2 \tau^2$.

    최소화하면 $w^* = \tau^2/(\tau^2 + p_0(1-p_0)/n)$이다. $\tau^2$이 크면(사전 정보가 불확실하면) 1에 가까워 축소가 약해지고, $\tau^2$이 작으면(사전 정보가 확실하면) 0에 가까워 축소가 강해진다.

    이것이 **경험적 베이즈** 축소추정량이다. 선거 여론조사 통합, 작은 실험이 많은 A/B 테스트, James-Stein 계열의 다변량 축소에 쓰인다.

<div class="drillbox" markdown>

**연습문제 6.**
**모집단 분포에 따른 $\bar X$의 평균제곱오차.** 다음 각 경우에 $\mu$의 추정량으로서 $\bar X$의 평균제곱오차를 계산하라: (a) $N(\mu, \sigma^2)$; (b) $\mathrm{Exp}(1/\mu)$; (c) 2차 적률이 무한한 모집단.

</div>

??? success "풀이"
    (a) 정규분포: $\mathrm{MSE}(\bar X) = \mathrm{Var}(\bar X) = \sigma^2/n$ (불편). CRLB를 달성하며 UMVUE이다.

    (b) 평균이 $\mu$, 분산이 $\mu^2$인 지수분포: $\bar X$는 불편이고 $\mathrm{Var}(\bar X) = \mu^2/n$이므로 $\mathrm{MSE} = \mu^2/n$이다. CRLB와 비교하면 $I(\mu) = 1/\mu^2$이므로 CRLB $= \mu^2/n$이다. $\bar X$는 지수분포의 평균에 대해 효율적이다.

    (c) 분산이 무한한 경우(예: 형상모수가 $\le 2$인 Pareto): $\mathrm{Var}(\bar X) = \infty$이므로 $\mathrm{MSE} = \infty$이다.

    (유한한 평균이 없는) Cauchy 분포에서는 $\bar X$가 "중심"의 추정량으로서 의미조차 갖지 않는다. Cauchy 분포의 중앙값에 대해 일치하는 표본중앙값을 쓰는 편이 낫다.

    **일반적인 교훈:** $\bar X$는 표준적인 추정량이며 정규분포와 지수분포에서 최적이지만 꼬리가 두꺼운 모집단에서는 무너진다. 평균 기반 추정량을 쓰기 전에 언제나 적률이 유한한지 확인해야 한다.

---

## 정리하며

평균제곱오차는 편향과 분산을 **하나의 수**로 합친다.

$$
\mathrm{MSE}(\hat\theta) = \mathbb{E}\bigl[(\hat\theta-\theta)^2\bigr] = \mathrm{Var}(\hat\theta) + \bigl[\mathrm{Bias}(\hat\theta)\bigr]^2
$$

- **분해가 항등식이라는 점이 중요하다.** 근사도 가정도 없이 늘 성립하며, 대수적으로는 $\mathbb{E}[\hat\theta]$ 를 더하고 빼서 전개하면 교차항이 사라진다.
- **불편추정량 중에서는 MSE 가 곧 분산이다.** 그래서 불편이라는 조건을 걸고 나면 남는 문제는 분산 최소화뿐이며, 그것이 다음다음 절의 크라메르–라오 하한으로 이어진다.
- **편향을 허용하면 더 잘할 수 있다.** 분산추정량에서 $n+1$ 로 나누는 것이 $n-1$ 로 나누는 불편추정량보다 MSE 가 작다는 것이 이 절의 대표적인 예다. **불편성을 포기한 대가로 전체 오차가 줄어든다.**
- **제곱이라는 선택은 편의이자 제약이다.** 계산이 쉽고 분해가 깔끔하지만 큰 오차에 무겁게 벌점을 주므로, 이상치가 있는 상황에서는 절대오차 같은 다른 손실이 나을 수 있다.

다음 절 **일치성과 점근정규성**은 시선을 $n\to\infty$ 로 옮긴다. MSE 가 유한표본의 기준이라면 일치성은 극한의 최소 요건이다.
