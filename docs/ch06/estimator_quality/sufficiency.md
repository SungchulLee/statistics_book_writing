# 충분성과 최소충분성

## 개요

추정이론에서는 자료를 표본평균이나 표본분산 같은 요약통계량으로 압축하는 일이 일상적이다. 여기서 근본적인 물음이 생긴다. 이 요약은 원래 자료가 미지의 모수에 관해 담고 있던 정보를 모두 보존하는가? 그렇다면 자료 전체 대신 요약만 가지고 작업해도 잃는 것이 없다. 충분성은 이러한 무손실 자료 축약이라는 착상을 형식화한다.

$T$의 치역에 속하는 임의의 $t$에 대해 $T(\mathbf{X}) = t$가 주어졌을 때 $\mathbf{X}$의 조건부분포가 $\theta$에 의존하지 않으면, 통계량 $T(\mathbf{X})$가 모수 $\theta$에 대해 **충분**하다고 한다. 직관적으로, 충분통계량의 값을 알고 나면 자료에 남은 무작위성은 $\theta$에 관해 아무런 추가 정보도 담고 있지 않다.

## Fisher–Neyman 인수분해 정리

정의에서 곧바로 충분성을 확인하려면 조건부분포를 계산해야 해서 대수적으로 까다로울 수 있다. Fisher–Neyman 인수분해 정리는 훨씬 간단한 기준을 준다. 결합밀도를 두 조각으로 인수분해하기만 하면 된다.

**정리 (Fisher–Neyman 인수분해).** 통계량 $T(\mathbf{X})$가 $\theta$에 대해 충분일 필요충분조건은 표본공간의 모든 $\mathbf{x}$와 모수공간의 모든 $\theta$에 대해 결합밀도(또는 질량함수)를 다음과 같이 쓸 수 있는 것이다.

$$
f(\mathbf{x}; \theta) = g\bigl(T(\mathbf{x}),\, \theta\bigr) \cdot h(\mathbf{x})
$$

여기서 $g \geq 0$은 자료에 오직 $T(\mathbf{x})$를 통해서만 의존하고 $\theta$에 의존할 수 있는 함수이며, $h \geq 0$은 $\mathbf{x}$만의 함수로 $\theta$에 의존하지 않는다.

!!! example "Poisson 표본의 인수분해"

    $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Poisson}(\lambda)$라 하자. 결합 질량함수는

    $$
    f(\mathbf{x}; \lambda) = \prod_{i=1}^n \frac{e^{-\lambda} \lambda^{x_i}}{x_i!} = e^{-n\lambda}\, \lambda^{\sum_{i=1}^n x_i} \cdot \frac{1}{\prod_{i=1}^n x_i!}
    $$

    $g\bigl(\sum x_i,\, \lambda\bigr) = e^{-n\lambda}\, \lambda^{\sum x_i}$, $h(\mathbf{x}) = 1 / \prod x_i!$로 두면, 인수분해 정리에 의해 $T(\mathbf{X}) = \sum_{i=1}^n X_i$가 $\lambda$에 대해 충분함이 확인된다.

## 최소충분성

충분통계량은 유일할 필요가 없다. 실제로 자료 벡터 $\mathbf{X}$ 자체가 언제나 자명하게 충분하다. $T(\mathbf{X}) = \mathbf{X}$로 두고 $f(\mathbf{x};\theta) = f(\mathbf{x};\theta) \cdot 1$로 쓸 수 있기 때문이다. 그러나 이 극단적인 경우는 자료를 전혀 축약하지 못한다. 그래서 $\theta$에 관한 정보를 모두 보존하면서 중복 정보를 최대한 버리는, 가장 많이 압축된 충분통계량을 찾게 된다.

충분통계량 $T$가 다른 모든 충분통계량 $T'$에 대해 거의 확실하게 $T = g(T')$인 함수 $g$가 존재하면 $T$를 **최소충분**이라 한다. 다시 말해 $T$는 $\theta$에 관한 모든 정보를 보존하면서 가능한 최대의 자료 압축을 제공한다.

## Rao–Blackwell 정리

충분성은 단지 이론적 개념에 그치지 않고 추정량의 품질에 직접 영향을 준다. 충분통계량이 $\theta$에 관한 모든 정보를 담고 있으므로, 추정량을 그것으로 조건화해도 유용한 정보를 잃지 않으며 오히려 변동성을 줄일 수 있다. Rao–Blackwell 정리가 이를 정확하게 말해 준다.

**정리 (Rao–Blackwell).** $\hat{\theta}$가 $\theta$의 불편추정량이고 $T$가 $\theta$에 대한 충분통계량이면 $\tilde{\theta} = E[\hat{\theta} \mid T]$는 다음을 만족한다:

1. $\tilde{\theta}$도 $\theta$에 대해 불편이고,
2. $\operatorname{Var}(\tilde{\theta}) \leq \operatorname{Var}(\hat{\theta})$이며, 등호는 $\hat{\theta}$가 이미 거의 확실하게 $T$의 함수일 때에만 성립한다.

Rao–Blackwell 정리는 충분통계량으로 조건화함으로써 불편추정량을 언제나 개선할 수 있음을(적어도 나빠지지는 않음을) 보여 준다. 최소충분통계량과 함께 쓰면 이 절차는 자료에서 가능한 최대의 분산 감소를 뽑아낸다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
인수분해 정리를 사용하여 $\text{Poisson}(\theta)$에서 뽑은 확률표본 $X_1, \dots, X_n$에 대해 $\theta$의 충분통계량을 구하라.

</div>

??? success "풀이"
    결합 PMF는:

    $$
    f(\mathbf{x}; \theta) = \prod_{i=1}^n \frac{\theta^{x_i} e^{-\theta}}{x_i!} = \frac{\theta^{\sum x_i} e^{-n\theta}}{\prod x_i!}
    $$

    인수분해 정리에 따라 이를 $g(T(\mathbf{x}), \theta) \cdot h(\mathbf{x})$로 쓰면:

    $$
    g(T, \theta) = \theta^T e^{-n\theta}, \quad h(\mathbf{x}) = \frac{1}{\prod x_i!}, \quad T(\mathbf{x}) = \sum_{i=1}^n x_i
    $$

    따라서 $T = \sum_{i=1}^n X_i$가 $\theta$에 대해 충분하다. $\bar{X} = T/n$은 $T$의 일대일 함수이므로 마찬가지로 충분하다.

<div class="drillbox" markdown>

**연습문제 2.**
순서통계량 $(X_{(1)}, X_{(2)}, \dots, X_{(n)})$이 분포족과 무관하게 언제나 임의의 모수에 대해 충분함을 증명하라. 이것이 왜 "자명하게" 충분한 통계량인가?

</div>

??? success "풀이"
    표본의 결합밀도는 다음과 같이 쓸 수 있다:

    $$
    f(\mathbf{x}; \theta) = g(x_{(1)}, \dots, x_{(n)}; \theta) \cdot h(\mathbf{x})
    $$

    여기서 $g$는 순서통계량으로 표현한 결합밀도이고 $h(\mathbf{x}) = 1$이다. 다른 방식으로 보면, (관측값이 교환 가능하므로) 결합밀도는 $\mathbf{x}$에 오직 집합 $\{x_1, \dots, x_n\}$을 통해서만 의존하고, 순서통계량이 이 집합을 온전히 담아낸다.

    이것이 "자명하게" 충분한 이유는 순서통계량이 표본의 거의 모든 정보를 그대로 지니기 때문이다. 버리는 것은 어느 관측값이 먼저 왔는지라는 표지뿐이다. 유용한 충분통계량이라면 자료를 더 많이 축약해야 한다. **최소충분성** 개념이 이를 형식화한다. $\theta$에 관한 정보를 잃지 않으면서 최대로 압축하는, 가장 거친 충분통계량이다. $\square$

<div class="drillbox" markdown>

**연습문제 3.**
두 모수가 모두 미지인 $N(\mu, \sigma^2)$에서 뽑은 확률표본에 대해 $(\sum X_i, \sum X_i^2)$이 $(\mu, \sigma^2)$에 대해 결합충분임을 보여라.

</div>

??? success "풀이"
    결합밀도는:

    $$
    f(\mathbf{x}; \mu, \sigma^2) = (2\pi\sigma^2)^{-n/2} \exp\!\left(-\frac{1}{2\sigma^2}\sum_{i=1}^n(x_i - \mu)^2\right)
    $$

    지수부를 전개하면:

    $$
    \sum(x_i - \mu)^2 = \sum x_i^2 - 2\mu\sum x_i + n\mu^2
    $$

    따라서:

    $$
    f(\mathbf{x}; \mu, \sigma^2) = (2\pi\sigma^2)^{-n/2}\exp\!\left(-\frac{\sum x_i^2 - 2\mu\sum x_i + n\mu^2}{2\sigma^2}\right) \cdot 1
    $$

    전체 표현이 $\mathbf{x}$에 오직 $\sum x_i$와 $\sum x_i^2$을 통해서만 의존한다. 인수분해 정리에 의해 $T(\mathbf{x}) = (\sum X_i, \sum X_i^2)$이 $(\mu, \sigma^2)$에 대해 충분하다. $\square$

<div class="drillbox" markdown>

**연습문제 4.**
Rao-Blackwell 정리를 서술하고 그 실용적 의의를 설명하라. $\hat{\theta}$가 불편추정량이고 $T$가 충분통계량일 때 $\tilde{\theta} = E[\hat{\theta} \mid T]$는 $\hat{\theta}$와 어떻게 비교되는가?

</div>

??? success "풀이"
    **Rao-Blackwell 정리:** $\hat{\theta}$가 $\theta$의 임의의 불편추정량이고 $T$가 충분통계량이면 $\tilde{\theta} = E[\hat{\theta} \mid T]$도 불편이며 다음을 만족한다:

    $$
    \text{Var}(\tilde{\theta}) \leq \text{Var}(\hat{\theta})
    $$

    등호는 $\hat{\theta}$가 이미 $T$의 함수일 때에만 성립한다.

    **실용적 의의:** 이 정리는 임의의 불편추정량을 개선하는 체계적인 방법을 준다. 충분통계량으로 조건화하는 것이다. 그 결과 얻은 추정량은 (평균제곱오차의 관점에서) 적어도 같은 수준이고 흔히 엄격하게 더 낫다. Lehmann-Scheffé 정리($T$가 완비이고 충분이면 $\tilde{\theta}$가 유일한 최소분산불편추정량, 즉 UMVUE이다)와 결합하면 최적 추정에 이르는 구성적인 길이 열린다.
