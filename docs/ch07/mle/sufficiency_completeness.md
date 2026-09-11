# 충분성과 완비성

## 이 개념들이 중요한 이유

분포의 모수를 추정할 때 우리는 핵심적인 질문에 부딪힌다: 모수에 관한 정보를 하나도 잃지 않으면서 자료를 더 간단한 요약으로 줄일 수 있는가? **충분성**이 이 질문에 답한다. 관련 개념인 **완비성**은 충분통계량에 기반한 불편추정량이 (최소분산이라는 의미에서) 유일한 최선임을 보장한다. 충분성과 완비성을 함께 쓰면 Lehmann-Scheffé 정리로 이어지는데, 이 정리는 균일최소분산불편추정량(UMVUE)을 구성적으로 찾는 방법을 준다.

## 정규 모형의 충분통계량

통계량 $T(\mathbf{X})$는 $T(\mathbf{X})$가 주어졌을 때 자료 $\mathbf{X}$의 조건부분포가 $\theta$에 의존하지 않을 때 모수 $\theta$에 대해 **충분**하다고 한다. 직관적으로, 충분통계량의 값을 알고 나면 남은 자료는 $\theta$에 대한 추가 정보를 담고 있지 않다.

**인수분해 정리**(Fisher-Neyman)는 실용적인 판정법을 준다: $T(\mathbf{X})$가 $\theta$에 대해 충분할 필요충분조건은 결합밀도를

$$
f(\mathbf{x}; \theta) = g(T(\mathbf{x}), \theta) \cdot h(\mathbf{x})
$$

로 쓸 수 있는 것이다. 여기서 $g$는 자료에 오직 $T$를 통해서만 의존하고 $h$는 $\theta$에 의존하지 않는다.

정규 모형 $X_1, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$에서 결합밀도는

$$
f(\mathbf{x}; \mu, \sigma^2) = \left(\frac{1}{2\pi\sigma^2}\right)^{n/2} \exp\left(-\frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2\right)
$$

$\sum(x_i - \mu)^2 = \sum(x_i - \bar{x})^2 + n(\bar{x} - \mu)^2$을 써서 지수를 전개하면:

$$
f(\mathbf{x}; \mu, \sigma^2) = \underbrace{\left(\frac{1}{2\pi\sigma^2}\right)^{n/2} \exp\left(-\frac{\sum(x_i - \bar{x})^2 + n(\bar{x} - \mu)^2}{2\sigma^2}\right)}_{g\left((\bar{x},\, \sum(x_i - \bar{x})^2),\; \mu,\, \sigma^2\right)} \cdot \underbrace{1}_{h(\mathbf{x})}
$$

밀도는 자료에 오직 $\bar{x}$와 $\sum(x_i - \bar{x})^2$을 통해서만 의존한다. 인수분해 정리에 의해 $(\bar{X}, S^2)$ — 동등하게 $(\bar{X}, \sum(X_i - \bar{X})^2)$ — 은 $(\mu, \sigma^2)$에 대해 결합충분이다.

!!! tip "충분성이 실무에서 뜻하는 것"

    $\bar{X}$와 $S^2$을 계산하고 나면 자료가 $(\mu, \sigma^2)$에 대해 담고 있는 정보를 모두 포착한 것이다. 개별 관측값 $X_1, \ldots, X_n$은 이 모수들을 추정하는 데 추가 가치가 없다. 통계 요약이 흔히 표본평균과 표본분산만 보고하는 이유가 여기에 있다.

## 완비성

충분성은 통계량이 모든 정보를 포착한다는 것을 말해 준다. 그러나 충분통계량에 기반한 불편추정량이 여럿 존재할 수도 있다. **완비성**은 통계량의 불편인 함수가 유일함을 보장하여 이를 배제한다.

충분통계량 $T$는 모든 가측함수 $g$에 대해 다음이 성립할 때 **완비**라고 한다:

$$
E_\theta[g(T)] = 0 \quad \text{모든 } \theta \in \Theta \text{에 대해} \quad \Longrightarrow \quad P_\theta(g(T) = 0) = 1 \quad \text{모든 } \theta \in \Theta \text{에 대해}
$$

말로 하면: $T$에 기반한 0의 불편추정량은 항등적으로 0인 함수뿐이다. 즉 $T$에는 "낭비된" 정보가 없다는 뜻이다 — 모든 모수값에서 평균이 0이 되는 자명하지 않은 신호를 뽑아낼 수 없다.

정규 모형에서 통계량 $(\bar{X}, S^2)$은 충분할 뿐 아니라 완비이기도 하다. 정규족이 완전계수 지수족이고, 완전계수 지수족에서는 완비충분통계량이 존재한다는 사실에서 따라 나온다.

## Lehmann-Scheffé 정리

충분성과 완비성을 결합한 결실이 Lehmann-Scheffé 정리이며, 유일한 최량 불편추정량을 확인해 준다.

<div class="thmbox" markdown>

### 정리 1. (Lehmann-Scheffé) { .thm }


$T$가 $\theta$에 대한 완비충분통계량이라 하자. $h(T)$가 함수 $\tau(\theta)$의 임의의 불편추정량이면 — 즉 모든 $\theta$에 대해 $E_\theta[h(T)] = \tau(\theta)$이면 — $h(T)$는 $\tau(\theta)$의 유일한 **균일최소분산불편추정량(UMVUE)**이다.

</div>

증명은 두 가지 사실에 기댄다. 첫째, Rao-Blackwell 정리에 의해 임의의 불편추정량을 충분통계량으로 조건부기댓값을 취하면 분산이 커지지 않는다. 둘째, 완비성은 각 목표 $\tau(\theta)$에 대해 $T$의 불편인 함수가 하나뿐임을 보장하므로 Rao-Blackwell화한 추정량이 유일하다.

## 정규 모형에의 적용

정규 모형에서 $(\bar{X}, S^2)$이 $(\mu, \sigma^2)$에 대해 완비충분이므로, $(\bar{X}, S^2)$의 불편인 함수는 자동으로 UMVUE이다.

**$\mu$의 UMVUE:** 표본평균 $\bar{X}$는 충분통계량의 함수이고 $E[\bar{X}] = \mu$를 만족하므로 $\mu$의 유일한 UMVUE이다.

**$\sigma^2$의 UMVUE:** 표본분산 $S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2$은 충분통계량의 함수이고 $E[S^2] = \sigma^2$을 만족하므로 $\sigma^2$의 유일한 UMVUE이다.

!!! warning "MLE와 UMVUE"

    $\sigma^2$의 MLE는 $\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum(X_i - \bar{X})^2$으로 $n-1$이 아니라 $n$으로 나눈다. 이는 편향되어 있으므로 충분통계량의 함수임에도 UMVUE가 **아니다**. Lehmann-Scheffé 정리는 불편성을 요구한다 — 완비충분통계량의 편향된 함수는 UMVUE가 아니다.

!!! example "UMVUE 성질의 확인"

    $N(\mu, \sigma^2)$에서 관측값 $n = 20$개일 때:

    - $\bar{X}$는 분산이 $\sigma^2/20$인 $\mu$의 UMVUE이다. $\mu$의 다른 어떤 불편추정량도 분산이 $\sigma^2/20$보다 작을 수 없다.
    - $S^2$은 분산이 $2\sigma^4/19$인 $\sigma^2$의 UMVUE이다. MLE $\hat{\sigma}^2_{\text{MLE}} = (19/20)S^2$은 평균제곱오차가 더 작지만 편향되어 있으므로 UMVUE의 자격이 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
$\text{Bernoulli}(p)$에서 뽑은 확률표본에 대해 $T = \sum X_i$가 완비충분통계량임을 보여라.

</div>

??? success "풀이"
    **충분성:** 결합 PMF는 $p^{\sum x_i}(1-p)^{n - \sum x_i}$로, $\mathbf{x}$에 오직 $T = \sum x_i$를 통해서만 의존한다. 인수분해 정리에 의해 $T$는 충분하다.

    **완비성:** $T \sim \text{Binomial}(n, p)$이다. 완비성을 보이려면 모든 $p \in (0,1)$에 대해 $E[g(T)] = 0$이면 $g(T) = 0$ a.s.임을 보여야 한다.

    $$
    E[g(T)] = \sum_{t=0}^n g(t)\binom{n}{t}p^t(1-p)^{n-t} = (1-p)^n \sum_{t=0}^n g(t)\binom{n}{t}\left(\frac{p}{1-p}\right)^t = 0
    $$

    $r = p/(1-p) \in (0, \infty)$로 두면 이는 항등적으로 0인 $r$에 대한 $n$차 다항식이다. 모든 곳에서 0인 다항식은 계수가 모두 0이므로 모든 $t$에 대해 $g(t)\binom{n}{t} = 0$이고, 따라서 모든 $t$에 대해 $g(t) = 0$이다. $\square$

<div class="drillbox" markdown>

**연습문제 2.**
Lehmann-Scheffé 정리를 진술하고, 이를 이용해 Bernoulli 표본에서 $p(1-p)$의 UMVUE를 구하라.

</div>

??? success "풀이"
    **Lehmann-Scheffé 정리:** $T$가 완비충분통계량이고 $h(T)$가 $\tau(\theta)$의 불편추정량이면 $h(T)$는 $\tau(\theta)$의 유일한 UMVUE이다.

    $T = \sum X_i \sim \text{Bin}(n, p)$의 함수이면서 $\tau(p) = p(1-p)$에 대해 불편인 추정량을 찾는다.

    $h(T) = \frac{T(n - T)}{n(n-1)}$을 생각하자:

    $$
    E\!\left[\frac{T(n-T)}{n(n-1)}\right] = \frac{E[nT - T^2]}{n(n-1)} = \frac{n \cdot np - (np(1-p) + n^2p^2)}{n(n-1)}
    $$

    $$
    = \frac{n^2p - np + np^2 - n^2p^2}{n(n-1)} = \frac{np(n-1)(1-p)}{n(n-1)} = p(1-p)
    $$

    $T$가 완비충분이고 $h(T)$가 $p(1-p)$에 대해 불편이므로 Lehmann-Scheffé 정리에 의해 이것이 UMVUE이다. $\square$

<div class="drillbox" markdown>

**연습문제 3.**
완비성과, 충분통계량에 기반한 불편추정량의 유일성 사이의 관계를 설명하라.

</div>

??? success "풀이"
    충분통계량 $T$가 완비라는 것은 기댓값이 항등적으로 0인 자명하지 않은 $T$의 함수가 없다는 뜻이다: 모든 $\theta$에 대해 $E[g(T)] = 0$이면 $g(T) = 0$ a.s.이다.

    이는 불편추정량의 **유일성**을 보장한다: $h_1(T)$와 $h_2(T)$가 모두 $\tau(\theta)$에 대해 불편이면 모든 $\theta$에 대해 $E[h_1(T) - h_2(T)] = 0$이다. 완비성에 의해 $h_1(T) - h_2(T) = 0$ a.s.이므로 $h_1 = h_2$이다.

    완비성이 없으면 $T$의 불편인 함수가 여럿 존재할 수 있고, Rao-Blackwell 정리만으로는 유일한 최량 추정량이 보장되지 않는다. 완비성이 이 틈을 메워 (존재한다면) UMVUE를 유일하게 만든다.

<div class="drillbox" markdown>

**연습문제 4.**
Uniform$(0, \theta)$에서 $T = X_{(n)} = \max(X_1, \dots, X_n)$이 완비충분통계량임을 보여라. 그다음 Uniform$(\theta, \theta + 1)$에서 충분하지만 완비가 아닌 통계량의 예를 찾아라.

</div>

??? success "풀이"
    **Uniform$(0, \theta)$:** $T = X_{(n)}$의 밀도는 $0 < t < \theta$에서 $f_T(t) = nt^{n-1}/\theta^n$이다. 결합밀도가 $\theta^{-n}\mathbf{1}\{X_{(n)} \le \theta\}$이므로 인수분해 정리에 의해 $T$는 충분하다.

    완비성: 모든 $\theta > 0$에 대해 $E_\theta[g(T)] = \frac{n}{\theta^n}\int_0^\theta g(t)t^{n-1}\,dt = 0$이라 하자. 그러면 모든 $\theta$에 대해 $\int_0^\theta g(t)t^{n-1}\,dt = 0$이고, $\theta$에 대해 미분하면 (거의 모든 $\theta$에서) $g(\theta)\theta^{n-1} = 0$이므로 $g \equiv 0$ a.e.이다. 따라서 $T$는 **완비**이다.

    **Uniform$(\theta, \theta + 1)$:** 여기서는 $T = (X_{(1)}, X_{(n)})$이 충분하다. 범위 $R = X_{(n)} - X_{(1)}$의 분포는 $\theta$에 의존하지 않고(전체 자료를 $\theta$만큼 평행이동해도 $R$은 변하지 않는다) $E[R] = \frac{n-1}{n+1}$이다. 따라서

    $$
    g(X_{(1)}, X_{(n)}) = X_{(n)} - X_{(1)} - \frac{n-1}{n+1}
    $$

    은 모든 $\theta$에 대해 $E[g(T)] = 0$이지만 $g \neq 0$ a.s.이다. 즉 $T$는 충분하지만 완비가 아니다.

    교훈: 완비성은 충분통계량만의 성질이 아니라 통계 모형의 성질이다. 받침이 유계인 위치족은 흔히 완비성이 깨진다.

---

## 정리하며

충분성에 **완비성**을 더하면 최소분산 불편추정량을 찾는 길이 열린다.

- **충분성**은 정보를 잃지 않는 압축이고, 피셔–네이만 인수분해가 실용적 판정법이다. 정규 모형에서는 $(\sum X_i,\sum X_i^2)$ 이 충분통계량이다.
- **완비성**은 "$\mathbb{E}[g(T)]=0$ 이 모든 $\theta$ 에서 성립하면 $g\equiv0$"이라는 조건이다. 충분통계량의 함수 중 기댓값이 $0$ 인 것이 자명한 것뿐이라는 뜻이며, 그래서 **불편추정량이 유일하게 정해진다.**
- **라오–블랙웰.** 아무 불편추정량이나 잡아 충분통계량으로 조건을 걸면 분산이 줄거나 같다. **추정량을 개선하는 기계적인 절차**다.
- **레만–셰페.** 여기에 완비성을 더하면 그 결과가 **유일한 UMVUE** 가 된다. 정규분포에서 $\bar X$ 가 $\mu$ 의, $S^2$ 이 $\sigma^2$ 의 UMVUE 인 것이 이렇게 확인된다.
- **완비성이 깨지는 경우도 있다.** 지지집합이 모수에 의존하거나 모수공간이 제한되면 성립하지 않으며, 그때는 최적 불편추정량이 유일하지 않을 수 있다.

**"불편추정량 중 최선"이라는 개념이 이로써 완성된다.** 6.1절에서 크라메르–라오 하한으로 바닥을 알았다면, 여기서는 그 바닥에 닿는 추정량을 **만드는 방법**을 얻었다.

다음 절 **정규분포 최대가능도 (코드)**에서 지금까지의 결과를 수치로 확인한다.
