# 편향–분산 맞바꿈

## 소개

모든 통계적 추정량은 근본적인 긴장에 놓인다. **단순함 대 유연함**이다. 단순한 추정량은 참 모수값을 체계적으로 빗나갈 수 있고(높은 편향), 유연한 추정량은 어떤 표본이 뽑혔느냐에 지나치게 민감할 수 있다(높은 분산). **편향–분산 맞바꿈**은 이 긴장을 형식화하며, 전체 추정오차를 최소화하려면 서로 경쟁하는 이 두 오차원을 균형 있게 다루어야 함을 드러낸다.

이 맞바꿈을 이해하는 것은 통계학, 기계학습, 계량금융 전반에서 추정량을 고르고 모형의 복잡도를 정하며 정칙화 전략을 설계하는 데 필수적이다.

## 정의

### 추정량의 편향

$\hat{\theta}$를 확률표본 $X_1, X_2, \ldots, X_n$에 기반한 모수 $\theta$의 추정량이라 하자. $\hat{\theta}$의 **편향**은 다음과 같이 정의된다:

$$\text{Bias}(\hat{\theta}) = E[\hat{\theta}] - \theta$$

$\text{Bias}(\hat{\theta}) = 0$이면, 즉 $E[\hat{\theta}] = \theta$이면 추정량이 **불편**이라고 한다.

**핵심:**

- 편향은 참 모수로부터의 체계적 이탈을 잰다
- 불편추정량은 가능한 모든 표본에 걸쳐 "평균적으로" 옳다
- 편향은 양수(과대추정)일 수도 음수(과소추정)일 수도 있다
- 어떤 추정량은 유한표본에서는 편향되어 있지만 $n \to \infty$일 때 점근적으로 불편일 수 있다

### 추정량의 분산

추정량 $\hat{\theta}$의 **분산**은 표본이 달라질 때 그것이 얼마나 요동치는지를 잰다:

$$\text{Var}(\hat{\theta}) = E\left[(\hat{\theta} - E[\hat{\theta}])^2\right]$$

**핵심:**

- 분산은 어떤 표본이 뽑혔느냐에 대한 추정량의 민감도를 담아낸다
- 분산이 크다는 것은 표본마다 추정량이 크게 달라진다는 뜻이다
- 분산은 표본크기 $n$이 커질수록 대체로 줄어든다
- 표준편차 $\text{SD}(\hat{\theta}) = \sqrt{\text{Var}(\hat{\theta})}$를 **표준오차**라 부른다

## 편향–분산 분해

추정량 $\hat{\theta}$의 **평균제곱오차(MSE)**는 편향 성분과 분산 성분으로 분해된다:

$$\text{MSE}(\hat{\theta}) = E\left[(\hat{\theta} - \theta)^2\right]$$

**유도:** $\mu = E[\hat{\theta}]$라 하자. 그러면:

$$\text{MSE}(\hat{\theta}) = E\left[(\hat{\theta} - \theta)^2\right]$$

$\mu$를 더하고 빼면:

$$= E\left[(\hat{\theta} - \mu + \mu - \theta)^2\right]$$

제곱을 전개하면:

$$= E\left[(\hat{\theta} - \mu)^2 + 2(\hat{\theta} - \mu)(\mu - \theta) + (\mu - \theta)^2\right]$$

$E[\hat{\theta} - \mu] = 0$이므로 교차항이 사라진다:

$$= E\left[(\hat{\theta} - \mu)^2\right] + (\mu - \theta)^2$$

따라서:

$$\boxed{\text{MSE}(\hat{\theta}) = \text{Var}(\hat{\theta}) + [\text{Bias}(\hat{\theta})]^2}$$

이것이 **편향–분산 분해**이다. 전체 오차(MSE)의 원천이 정확히 둘, 즉 분산(무작위 요동)과 편향의 제곱(체계적 오차)임을 보여 준다.

## 실제로 작동하는 맞바꿈

### 왜 맞바꿈이 존재하는가

많은 추정 문제에서 편향을 줄이면 분산이 커지고 그 반대도 마찬가지이다:

| 전략 | 편향에 대한 효과 | 분산에 대한 효과 |
|----------|---------------|-------------------|
| 더 유연한 모형 | ↓ 감소 | ↑ 증가 |
| 더 경직된 모형 | ↑ 증가 | ↓ 감소 |
| 더 큰 표본크기 | ↓ 감소 (대체로) | ↓ 감소 |
| 정칙화 | ↑ 증가 | ↓ 감소 |

### 고전적인 예: 모평균 추정

$X_1, \ldots, X_n \sim N(\mu, \sigma^2)$로부터 $\mu$를 추정하는 경우를 생각하자.

**추정량 1: 표본평균** $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$

- 편향: $E[\bar{X}] - \mu = 0$ (불편)
- 분산: $\text{Var}(\bar{X}) = \sigma^2/n$
- MSE: $\sigma^2/n$

**추정량 2: 축소추정량** $0 < \lambda < 1$에 대해 $\hat{\mu}_\lambda = \lambda \bar{X}$

- 편향: $E[\hat{\mu}_\lambda] - \mu = (\lambda - 1)\mu \neq 0$ (편향)
- 분산: $\text{Var}(\hat{\mu}_\lambda) = \lambda^2 \sigma^2/n$
- MSE: $\lambda^2 \sigma^2/n + (1-\lambda)^2 \mu^2$

어떤 $\lambda$ 값에서는 축소추정량이 편향되어 있음에도 불편인 표본평균보다 **평균제곱오차가 작을** 수 있다. 이것이 맞바꿈의 핵심이다. 작은 편향을 들여오면 분산을 크게 줄일 수 있고, 그 결과 추정의 정확도가 순전히 좋아진다.

### 최적의 축소

$\hat{\mu}_\lambda$의 평균제곱오차를 $\lambda$에 대해 최소화하면:

$$\frac{d}{d\lambda}\left[\lambda^2 \frac{\sigma^2}{n} + (1-\lambda)^2 \mu^2\right] = 0$$

$$2\lambda \frac{\sigma^2}{n} - 2(1-\lambda)\mu^2 = 0$$

$$\lambda^* = \frac{\mu^2}{\mu^2 + \sigma^2/n} = \frac{n\mu^2}{n\mu^2 + \sigma^2}$$

$|\mu|$가 $\sigma/\sqrt{n}$에 비해 작으면 최적의 $\lambda^*$가 1보다 상당히 작아지며, 이는 0 쪽으로 과감하게 축소하는 것이 최적임을 뜻한다.

## 기하적 해석

편향–분산 맞바꿈에는 직관적인 기하적 그림이 있다:

- **편향** = 추정량 분포의 중심에서 참값까지의 거리(체계적 이동)
- **분산** = 추정량 분포의 퍼짐(무작위 산포)
- **MSE** = 추정량에서 참값까지의 평균 제곱거리

다트판에 비유하면:

- **낮은 편향, 낮은 분산**: 다트가 과녁 중심 주위에 모여 있다 (이상적)
- **낮은 편향, 높은 분산**: 다트가 흩어져 있지만 중심을 기준으로 퍼져 있다
- **높은 편향, 낮은 분산**: 다트가 모여 있지만 중심에서 벗어나 있다
- **높은 편향, 높은 분산**: 다트가 흩어져 있고 중심에서도 벗어나 있다 (최악)

## 모형 선택에 대한 함의

### 과소적합과 과대적합

편향–분산 맞바꿈은 과소적합과 과대적합 개념과 직접 연결된다:

- **과소적합** (높은 편향): 모형이 너무 단순해서 참 관계를 담아내지 못한다. 모형의 복잡도를 높이면 편향이 줄지만 분산이 커질 수 있다.
- **과대적합** (높은 분산): 모형이 훈련 자료의 잡음까지 적합한다. 복잡도를 낮추거나 정칙화를 더하면 분산이 줄지만 편향이 커질 수 있다.

### "U자 모양"의 MSE 곡선

모형의 복잡도가 커질수록:

1. 편향은 단조 감소한다 (더 유연한 모형이 참값을 더 잘 근사한다)
2. 분산은 단조 증가한다 (더 유연한 모형이 자료에 더 민감하다)
3. MSE는 처음에는 감소하다가(편향 감소가 우세) 최솟값에 이른 뒤 증가한다(분산이 우세)

최적의 복잡도는 편향과 분산의 균형을 맞추는 MSE 최솟값에 있다.

## 금융과의 연결

계량금융에서 편향–분산 맞바꿈은 여러 맥락에서 나타난다:

- **포트폴리오 최적화**: 표본 공분산행렬(불편)과 축소추정량(편향되지만 분산이 작음) 중에서 고르는 문제이다. Ledoit-Wolf 축소추정량이 유명한 응용이다.
- **요인 모형**: 요인 개수를 정하는 문제이다. 너무 적으면 편향이 크고, 너무 많으면 추정된 적재값의 분산이 커진다.
- **변동성 추정**: EWMA(최근 자료 쪽으로 편향)와 역사적 변동성(불편이지만 분산이 큼) 사이의 선택이다.
- **위험 예측**: 더 복잡한 VaR 모형은 편향이 작을 수 있지만, 특히 자료가 제한적일 때 추정 분산이 커진다.

## 요약

편향–분산 맞바꿈은 기초 원리이다. 전체 추정오차(MSE)는 편향의 제곱과 분산으로 분해되며, 한쪽을 줄이면 흔히 다른 쪽이 커진다. 최선의 추정량이 반드시 불편인 것은 아니다. 올바른 균형점을 찾아 평균제곱오차를 최소화하는 것이 최선이다. 이 원리가 통계학과 계량금융 전반에서 추정량 선택, 정칙화, 모형 복잡도에 관한 결정을 이끈다.

## 주요 공식

| 양 | 공식 |
|----------|---------|
| 편향 | $\text{Bias}(\hat{\theta}) = E[\hat{\theta}] - \theta$ |
| 분산 | $\text{Var}(\hat{\theta}) = E[(\hat{\theta} - E[\hat{\theta}])^2]$ |
| MSE 분해 | $\text{MSE} = \text{Var}(\hat{\theta}) + [\text{Bias}(\hat{\theta})]^2$ |
| 불편 조건 | $E[\hat{\theta}] = \theta$ |

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
$\hat\theta$가 불편이고 $\mathrm{Var}(\hat\theta) = 0$이면 거의 확실하게 $\hat\theta = \theta$임을 증명하라.

</div>

??? success "풀이"
    $\mathrm{Var}(\hat\theta) = 0 \Rightarrow \hat\theta$는 퇴화되어 있다(거의 확실하게 상수이다). 이 상수를 $c$라 하자.

    불편성에 의해 $\mathbb{E}[\hat\theta] = \theta$이다. 그런데 $\mathbb{E}[c] = c$이므로 $c = \theta$이다.

    따라서 거의 확실하게 $\hat\theta = \theta$이다. $\square$

    이는 $\theta$가 퇴화되어 있지 않은 한 유한한 자료로 완벽하게 추정하는 것이 불가능함을 뜻한다. 어느 정도의 분산은 피할 수 없다.

<div class="drillbox" markdown>

**연습문제 2.**
**MSE의 편향–분산 분해.** $\mathrm{MSE}(\hat\theta) = \mathrm{Var}(\hat\theta) + [\mathrm{Bias}(\hat\theta)]^2$을 증명하라.

</div>

??? success "풀이"
    $\mathrm{MSE}(\hat\theta) = \mathbb{E}[(\hat\theta - \theta)^2]$이다.

    $\mathbb{E}[\hat\theta]$를 더하고 빼면:

    $\mathbb{E}[(\hat\theta - \mathbb{E}[\hat\theta] + \mathbb{E}[\hat\theta] - \theta)^2]$.

    전개하면:

    $= \mathbb{E}[(\hat\theta - \mathbb{E}[\hat\theta])^2] + 2(\mathbb{E}[\hat\theta] - \theta) \mathbb{E}[\hat\theta - \mathbb{E}[\hat\theta]] + (\mathbb{E}[\hat\theta] - \theta)^2$.

    ($\mathbb{E}[\hat\theta - \mathbb{E}[\hat\theta]] = 0$이므로) 가운데 항이 사라진다.

    $= \mathrm{Var}(\hat\theta) + \mathrm{Bias}(\hat\theta)^2$. $\square$

    이 분해가 모든 편향–분산 맞바꿈 논증의 토대이다.

<div class="drillbox" markdown>

**연습문제 3.**
**평균제곱오차가 더 작은 편향추정량.** $N(\mu, \sigma^2)$ 자료에 대해 $\hat\sigma^2_{\mathrm{MLE}} = (1/n)\sum(X_i - \bar X)^2$과 $s^2 = (1/(n-1))\sum(X_i - \bar X)^2$을 비교하라. 각각의 평균제곱오차를 계산하라.

</div>

??? success "풀이"
    $s^2$은 불편이다: $\mathbb{E}[s^2] = \sigma^2$, $\mathrm{Var}(s^2) = 2\sigma^4/(n-1)$, $\mathrm{MSE} = 2\sigma^4/(n-1)$.

    $\hat\sigma^2_{\mathrm{MLE}} = ((n-1)/n) s^2$: $\mathbb{E}[\hat\sigma^2] = (n-1)\sigma^2/n$이므로 편향 = $-\sigma^2/n$.

    $\mathrm{Var}(\hat\sigma^2_{\mathrm{MLE}}) = ((n-1)/n)^2 \cdot 2\sigma^4/(n-1) = 2(n-1)\sigma^4/n^2$.

    $\mathrm{MSE}(\hat\sigma^2_{\mathrm{MLE}}) = 2(n-1)\sigma^4/n^2 + \sigma^4/n^2 = (2n-1)\sigma^4/n^2$.

    **비:** 모든 $n \ge 2$에서 $\mathrm{MSE}_{\mathrm{MLE}}/\mathrm{MSE}_{s^2} = (2n-1)(n-1)/(2n^2) < 1$이다.

    MLE는 편향되어 있음에도 평균제곱오차가 *더 작다*. 편향–분산 맞바꿈의 교과서적인 예이다.

<div class="drillbox" markdown>

**연습문제 4.**
**점근적 불편성.** 점근적으로는 불편이지만 유한표본에서는 편향된 추정량을 정의하고 예를 들라.

</div>

??? success "풀이"
    **점근적으로 불편:** $n \to \infty$일 때 $\mathrm{Bias}(\hat\theta_n) \to 0$이다.

    **예:** $N(\mu, \sigma^2)$에서 $\sigma^2$의 MLE. 편향이 $-\sigma^2/n \to 0$이다. 유한한 $n$에서는 편향되어 있지만 $n$이 크면 그 크기가 사라진다.

    점근분산이 유한한 일치추정량은 *평균의 의미에서* 모두 점근적으로 불편이다. 그 역은 정확히 성립하지는 않는다. 점근적 불편성에 분산의 유계성이 더해지면 Chebyshev에 의해 일치성이 따라 나온다.

    **왜 중요한가:** 점근적 불편성은 약한 조건이어서 "나쁜" 추정량도 점근적으로 불편인 경우가 많다. 의미 있는 점근적 보장을 얻으려면 올바른 속도의 점근정규성이라는 더 강한 조건이 필요하다.

<div class="drillbox" markdown>

**연습문제 5.**
**편향 보정.** $X \sim N(\mu, \sigma^2)$에서 $\sigma$(표준편차)의 MLE는 $\hat\sigma_{\mathrm{MLE}} = \sqrt{(1/n)\sum(X_i - \bar X)^2}$이다. $\mathbb{E}[\hat\sigma_{\mathrm{MLE}}] < \sigma$임을 보이고 편향 보정을 제시하라.

</div>

??? success "풀이"
    $\hat\sigma^2_{\mathrm{MLE}} \sim (\sigma^2/n) \chi^2_{n-1}$일 때 $\hat\sigma_{\mathrm{MLE}} = \sqrt{\hat\sigma^2_{\mathrm{MLE}}}$이다.

    (오목함수 $\sqrt{\cdot}$에 대한) Jensen 부등식에 의해 $\mathbb{E}[\sqrt{\hat\sigma^2_{\mathrm{MLE}}}] \le \sqrt{\mathbb{E}[\hat\sigma^2_{\mathrm{MLE}}]} = \sigma\sqrt{(n-1)/n}$이다. 퇴화된 경우가 아니면 부등호가 엄격하다.

    더 정확히는 $\mathbb{E}[\hat\sigma_{\mathrm{MLE}}] = \sigma \sqrt{2/n} \Gamma(n/2)/\Gamma((n-1)/2)$이다.

    **편향 보정:** $c_n = \sqrt{2/n} \Gamma(n/2)/\Gamma((n-1)/2)$일 때 $\hat\sigma_{\mathrm{corrected}} = \hat\sigma_{\mathrm{MLE}}/c_n$으로 둔다.

    $n$이 크면 $c_n \approx \sqrt{(n-1)/n}$이므로 보정계수는 $\approx \sqrt{n/(n-1)}$이다. 1차 근사에서는 $\hat\sigma$ 대신 $s$를 쓰는 것과 같다.

    관리도 상수에 사용된다(예: Shewhart $\bar X$ 관리도의 $c_4$).

<div class="drillbox" markdown>

**연습문제 6.**
**분산과 일치성.** 편향이 0이지만 분산이 무한한 추정량의 예와, 분산은 유한하지만 일치하지 않는 추정량의 예를 각각 들라.

</div>

??? success "풀이"
    **편향 0, 분산 무한:** Cauchy 분포에서 $\hat\mu = X_1$. 평균은 정의되지 않지만 중앙값은 $\mu$이다. 더 일반적으로는 Cauchy 형태의 모형에서 얻은 MLE가 점근적으로는 불편이면서 분산이 무한할 수 있다.

    **분산 유한, 일치하지 않음:** i.i.d. $X_i \sim N(\theta, 1)$에서 $\theta$를 추정할 때 $\hat\theta = X_1$. 언제나 관측값 하나만 사용한다. 모든 $n$에 대해 분산이 1이고 0으로 가지 않는다. 따라서 일치하지 않는다($\hat\theta$가 $\theta$로 확률수렴하지 않는다).

    **교훈:** 일치성을 가지려면 추정량이 커지는 표본크기를 "활용"해야 한다. $\hat\theta = X_1$은 $n$과 무관하게 첫 관측값을 제외한 모든 것을 무시한다.
