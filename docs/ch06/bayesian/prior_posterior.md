# 사전분포, 가능도, 사후분포

빈도주의 통계학에서 모수 $\theta$는 고정되어 있지만 미지인 상수로 다루어지며, 추론은 전적으로 자료의 표본분포에 의존한다. 베이즈 추론은 근본적으로 다른 접근을 취한다. $\theta$를 고유한 분포를 갖는 확률변수로 다루는 것이다. 자료를 관측하기 전에 $\theta$에 대한 믿음은 **사전분포**에 담긴다. 자료를 관측한 뒤에는 베이즈 정리가 이 믿음을 **사후분포**로 갱신하는 원리 있는 장치를 제공한다. 이 절에서는 베이즈 추론의 세 가지 핵심 요소와 그것들이 결합되는 방식을 소개한다.

## 베이즈 정리

관측된 자료 $\mathbf{x} = (x_1, \ldots, x_n)$이 주어졌을 때 $\theta$의 사후분포는 베이즈 정리로 주어진다:

$$
\pi(\theta \mid \mathbf{x}) = \frac{f(\mathbf{x} \mid \theta)\,\pi(\theta)}{f(\mathbf{x})} \propto f(\mathbf{x} \mid \theta)\,\pi(\theta)
$$

오른쪽의 비례 관계는 $\theta$에 의존하지 않는 분모 $f(\mathbf{x})$를 떨어뜨린 것이다. $\theta$에 의존하는 항만으로 사후분포의 분포족을 알아낼 수 있으므로 이 비례 형태만으로 충분한 경우가 많다.

## 구성 요소

**사전분포** $\pi(\theta)$: 자료를 보기 전 $\theta$에 대한 믿음을 담는다. 실제 분야 지식을 반영할 수도 있고(정보가 있는 사전분포) 의도적으로 판단을 유보할 수도 있다(무정보 또는 약한 정보의 사전분포). 사전분포의 선택은 베이즈 추론을 구별짓는 특징이자 자주 제기되는 비판 지점이기도 하다.

**가능도** $f(\mathbf{x} \mid \theta)$: 관측된 표본의 확률밀도(이산 자료에서는 확률질량)를 $\theta$의 함수로 본 것이다. $f(\mathbf{x} \mid \theta)$가 $\mathbf{x}$에 대한 밀도와 같은 표기를 쓰지만, 베이즈 맥락에서는 $\theta$에 대한 의존을 강조한다. 가능도는 각 후보 모수값이 자료를 얼마나 잘 설명하는지 알려 준다.

**사후분포** $\pi(\theta \mid \mathbf{x})$: 자료를 반영한 뒤의 $\theta$의 갱신된 분포이다. 사전분포와 가능도를 결합하며 베이즈 추론의 중심 대상이다. 모든 결론(점추정, 구간, 예측)이 사후분포에서 흘러나온다.

**주변가능도** $f(\mathbf{x})$: 사후분포의 적분이 1이 되도록 하는 정규화 상수이다:

$$
f(\mathbf{x}) = \int f(\mathbf{x} \mid \theta)\,\pi(\theta)\,d\theta
$$

이 적분은 사전분포로 가중하여 가능한 모든 모수값에 걸쳐 가능도를 평균한다. 이 적분을 계산하기 어려운 경우가 많으며, 그래서 (닫힌 형태의 사후분포를 주는) 켤레 사전분포와 (MCMC 같은) 계산 방법이 베이즈 실무의 필수 도구가 된다.

## 베이즈 점추정

완전한 사후분포가 추론 문제에 대한 온전한 베이즈적 답이다. 다만 실무자는 사후분포를 요약하는 하나의 수가 필요한 경우가 많다. 흔히 쓰는 각 점추정값은 서로 다른 손실함수를 최소화하는 것에 대응한다.

| 추정값 | 정의 | 최소화하는 손실 |
|---|---|---|
| 사후평균 | $E[\theta \mid \mathbf{x}]$ | 제곱오차 손실 |
| 사후중앙값 | $\pi(\theta \mid \mathbf{x})$의 중앙값 | 절대오차 손실 |
| MAP | $\pi(\theta \mid \mathbf{x})$의 최빈값 | (극한의 의미에서) 0-1 손실 |

!!! example "정규 사전분포와 정규 가능도"
    $\sigma^2$이 알려진 $X_1, \ldots, X_n \overset{iid}{\sim} N(\theta, \sigma^2)$이고 사전분포가 $\theta \sim N(\mu_0, \sigma_0^2)$이라 하자. 베이즈 정리를 적용하고 완전제곱식을 만들면 사후분포는

    $$
    \theta \mid \mathbf{x} \sim N\!\left(\frac{\sigma^2\,\mu_0 + n\,\sigma_0^2\,\bar{x}}{\sigma^2 + n\,\sigma_0^2},\; \frac{\sigma^2\,\sigma_0^2}{\sigma^2 + n\,\sigma_0^2}\right)
    $$

    사후평균은 사전평균 $\mu_0$과 표본평균 $\bar{x}$의 가중평균이며, 가중은 각각의 상대적 정밀도(분산의 역수)로 결정된다. $n$이 커지면 자료가 지배하여 사전분포와 무관하게 사후분포가 $\bar{x}$ 주위로 모인다. 사후분포가 대칭이고 단봉이므로 이 경우 평균, 중앙값, MAP가 모두 일치한다.

## 연습문제

**연습문제 1.**
자료 $\mathbf{x}$가 주어졌을 때 모수 $\theta$에 대한 베이즈 정리를 서술하라. 각 구성 요소(사전분포, 가능도, 사후분포, 주변가능도)를 찾아 이름을 붙여라.

??? success "풀이"
    베이즈 정리는 다음과 같다:

    $$
    \underbrace{\pi(\theta \mid \mathbf{x})}_{\text{posterior}} = \frac{\overbrace{L(\mathbf{x} \mid \theta)}^{\text{likelihood}} \cdot \overbrace{\pi(\theta)}^{\text{prior}}}{\underbrace{m(\mathbf{x})}_{\text{marginal likelihood}}}
    $$

    - **사전분포** $\pi(\theta)$: 자료를 보기 전 $\theta$의 분포로 사전 믿음을 담는다.
    - **가능도** $L(\mathbf{x} \mid \theta)$: $\theta$가 주어졌을 때 관측된 자료의 확률이다.
    - **주변가능도** $m(\mathbf{x}) = \int L(\mathbf{x} \mid \theta)\pi(\theta)\,d\theta$: 사후분포의 적분이 1이 되게 하는 정규화 상수이다.
    - **사후분포** $\pi(\theta \mid \mathbf{x})$: 자료를 관측한 뒤의 갱신된 $\theta$의 분포이다.

---

**연습문제 2.**
동전의 앞면 확률에 대한 사전분포가 $p \sim \text{Uniform}(0, 1)$이라 하자. 동전을 한 번 던져 앞면을 관측했다. 사후분포 $\pi(p \mid H)$를 계산하라.

??? success "풀이"
    사전분포는 $p \in (0, 1)$에서 $\pi(p) = 1$이다($\text{Beta}(1, 1)$과 같다). 앞면 한 번에 대한 가능도는 $L(H \mid p) = p$이다.

    베이즈 정리에 의해:

    $$
    \pi(p \mid H) = \frac{p \cdot 1}{\int_0^1 p \, dp} = \frac{p}{1/2} = 2p
    $$

    이는 $\text{Beta}(2, 1)$의 밀도이다. 사후평균은 $E[p \mid H] = 2/3$으로 사전평균 $1/2$에서 위로 이동했으며, 앞면을 관측했다는 증거를 반영한다.

---

**연습문제 3.**
사후최빈값(MAP 추정값)과 MLE의 관계를 설명하라. 어떤 조건에서 MAP 추정값이 MLE와 같아지는가?

??? success "풀이"
    **MAP(최대사후확률)** 추정값은 사후분포를 최대화한다:

    $$
    \hat{\theta}_{\text{MAP}} = \arg\max_\theta \bigl[L(\mathbf{x} \mid \theta) \cdot \pi(\theta)\bigr] = \arg\max_\theta \bigl[\log L(\mathbf{x} \mid \theta) + \log \pi(\theta)\bigr]
    $$

    **MLE**는 가능도만 최대화한다: $\hat{\theta}_{\text{MLE}} = \arg\max_\theta L(\mathbf{x} \mid \theta)$.

    사전분포가 평평하면(균등/무정보), 즉 $\pi(\theta) \propto c$(상수)이면 $\log \pi(\theta)$가 최적화에 영향을 주지 않는 상수가 되므로 MAP가 MLE와 같아진다. 또한 $n \to \infty$일 때 큰 표본에서는 가능도가 사전분포를 압도하므로 MAP가 MLE에 가까워진다.

---

**연습문제 4.**
사전분포 $\pi(\theta)$가 $\theta = 0$에 확률 0.7을, $\theta = 1$에 0.3을 부여한다. 가능도는 $P(X = 1 \mid \theta = 0) = 0.2$, $P(X = 1 \mid \theta = 1) = 0.9$를 만족한다. $X = 1$을 관측한 뒤 사후확률 $P(\theta = 0 \mid X = 1)$과 $P(\theta = 1 \mid X = 1)$을 계산하라.

??? success "풀이"
    이산 $\theta$에 대한 베이즈 정리에 의해:

    $$
    P(\theta = 0 \mid X = 1) = \frac{P(X = 1 \mid \theta = 0) \cdot P(\theta = 0)}{P(X = 1)}
    $$

    먼저 주변확률을 계산한다:

    $$
    P(X = 1) = 0.2 \times 0.7 + 0.9 \times 0.3 = 0.14 + 0.27 = 0.41
    $$

    그러면:

    $$
    P(\theta = 0 \mid X = 1) = \frac{0.14}{0.41} \approx 0.341
    $$

    $$
    P(\theta = 1 \mid X = 1) = \frac{0.27}{0.41} \approx 0.659
    $$

    $X = 1$이 $\theta = 1$ 아래에서 훨씬 더 그럴듯하므로, 이 관측이 사후분포를 $\theta = 1$ 쪽으로(사전 0.3에서 사후 0.659로) 이동시켰다.
