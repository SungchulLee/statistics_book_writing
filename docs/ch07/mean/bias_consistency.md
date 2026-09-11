# 표본평균의 편향과 일치성

## 들어가며

어떤 추정량에 대해서든 근본적인 질문 두 가지는 다음과 같다: (1) 참 모수를 체계적으로 과대 또는 과소추정하는가? (**편향**) (2) 표본크기가 커질 때 참값으로 수렴하는가? (**일치성**) 표본평균 $\bar{X}$에 대한 답은 안심할 만큼 단순하다 — 매우 온건한 조건 아래에서 불편이고 일치한다 — 그러나 그 정확한 진술과 함의는 꼼꼼히 살펴볼 가치가 있다.

## 표본평균의 편향

### 불편성

표본평균 $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$은 $\mu = E[X]$에 대해 **불편**이다:

$$E[\bar{X}] = \mu \quad \text{모든 } n \geq 1 \text{에 대해}$$

**증명:** 기댓값의 선형성에 의해:

$$E[\bar{X}] = E\left[\frac{1}{n}\sum_{i=1}^n X_i\right] = \frac{1}{n}\sum_{i=1}^n E[X_i] = \frac{1}{n} \cdot n\mu = \mu$$

이는 최소한의 조건 아래에서 성립한다:

- 관측값이 동일한 분포를 따를 필요가 없다(모든 $i$에 대해 $E[X_i] = \mu$이기만 하면 된다)
- 관측값이 독립일 필요가 없다
- 분포에 대한 가정이 필요 없다
- 임의의 표본크기 $n \geq 1$에서 유효하다

### 유한표본 편향이 0이다

(분산의 MLE 같은) 많은 추정량과 달리, 표본평균은 모든 유한 표본크기에서 **편향이 정확히 0**이다. 이는 강한 성질이다 — 대부분의 추정량은 유한표본에서 편향이 0이 아니고 점근적으로만 불편이다.

### 편향된 대안과의 비교

모평균의 추정량 중에는 의도적으로 편향된 것도 있다:

| 추정량 | 편향 | 평균제곱오차 |
|-----------|------|-----|
| $\bar{X}$ (표본평균) | $0$ | $\sigma^2/n$ |
| $\lambda\bar{X}$ (축소, $\lambda < 1$) | $(\lambda-1)\mu$ | $\lambda^2\sigma^2/n + (1-\lambda)^2\mu^2$ |
| $c$ (상수) | $c - \mu$ | $(c-\mu)^2$ |

편향–분산 맞바꿈에서 논의했듯이, 특히 $|\mu|$가 $\sigma/\sqrt{n}$에 비해 작을 때 편향추정량이 더 작은 평균제곱오차를 가질 수 있다.

## 일치성

### 확률수렴 의미의 일치성

표본평균은 $\mu$에 대해 **일치**한다:

$$\bar{X}_n \xrightarrow{p} \mu \quad (n \to \infty)$$

즉, 임의의 $\epsilon > 0$에 대해

$$\lim_{n \to \infty} P\left(|\bar{X}_n - \mu| > \epsilon\right) = 0$$

### Chebyshev 부등식을 통한 증명

$\bar{X}$의 알려진 분산과 함께 Chebyshev 부등식을 쓰면:

$$P(|\bar{X} - \mu| > \epsilon) \leq \frac{\text{Var}(\bar{X})}{\epsilon^2} = \frac{\sigma^2}{n\epsilon^2} \to 0$$

여기에는 $\sigma^2 < \infty$(유한한 분산)만 필요하다.

### 약대수의법칙(WLLN)을 통한 증명

$\bar{X}$의 일치성은 바로 **약대수의법칙**의 진술이다: $X_1, X_2, \ldots$가 i.i.d.이고 $E[X_i] = \mu$, $\text{Var}(X_i) = \sigma^2 < \infty$이면,

$$\bar{X}_n \xrightarrow{p} \mu$$

**Khintchine의 WLLN**은 조건을 약화한다: $E[|X|] < \infty$만 있으면 된다(분산이 유한할 필요가 없다).

### 거의 확실한 수렴 (강일치성)

같은 조건 아래에서 **강대수의법칙(SLLN)**은 더 강한 결과를 준다:

$$P\left(\lim_{n \to \infty} \bar{X}_n = \mu\right) = 1$$

즉 $\bar{X}_n \to \mu$가 확률수렴을 넘어 거의 확실하게 성립한다.

### 수렴 속도

$\bar{X}_n$은 얼마나 빨리 $\mu$로 수렴하는가?

**평균제곱오차의 수렴 속도:**

$$\text{MSE}(\bar{X}_n) = \frac{\sigma^2}{n} = O(1/n)$$

**표준오차의 수렴 속도:**

$$\text{SE}(\bar{X}_n) = \frac{\sigma}{\sqrt{n}} = O(1/\sqrt{n})$$

이 $O(1/\sqrt{n})$ 속도는 근본적이다 — 다음을 뜻한다:

- 정확도를 두 배로 하려면 자료가 4배 필요하다
- 정확도를 10배로 하려면 자료가 100배 필요하다
- 추가 가정 없이는 (일반적으로) 이 속도를 개선할 수 없다

### 평균제곱오차 일치성

$\text{MSE}(\hat{\theta}_n) \to 0$이면 추정량이 **평균제곱오차 일치**라고 한다. $\bar{X}$에 대해서는:

$$\text{MSE}(\bar{X}_n) = \underbrace{[\text{Bias}(\bar{X}_n)]^2}_{= 0} + \underbrace{\text{Var}(\bar{X}_n)}_{= \sigma^2/n \to 0} \to 0$$

평균제곱오차 일치성은 (Markov 부등식에 의해) 확률수렴 의미의 일치성을 함의한다.

## 일치성의 조건

### X-bar가 일치하는 경우
표본평균은 i.i.d. 가정을 여러 방식으로 완화해도 일치한다:

1. **독립이지만 동일 분포가 아닌 경우**: 모든 $i$에 대해 $E[X_i] = \mu$이고 $\frac{1}{n^2}\sum_{i=1}^n \text{Var}(X_i) \to 0$이면 $\bar{X}_n \xrightarrow{p} \mu$이다.

2. **종속 관측값**: 정상 에르고딕 과정에서는 에르고딕 정리에 의해 $\bar{X}_n \to \mu$가 거의 확실하게 성립한다.

3. **약종속 시계열**: 자기상관이 충분히 빨리 감쇠하면(예: $\sum_{k=0}^\infty |\rho_k| < \infty$) $\bar{X}_n$은 일치한다.

### X-bar가 일치하지 않는 경우
1. **무한한 분산** (예: 자유도 2인 Student-t): $\text{Var}(X)$가 존재하지 않더라도 $E[|X|] < \infty$이면 (Khintchine의 WLLN에 의해) $\bar{X}_n$은 여전히 일치한다.

2. **무한한 평균** (예: Cauchy): $E[X]$가 존재하지 않으므로 $\bar{X}$가 수렴할 $\mu$ 자체가 없다. 표본평균은 심하게 요동치며 수렴하지 않는다.

3. **비정상 자료**: 평균이 시간에 따라 변하면 $\bar{X}$는 변하는 평균들의 평균으로 수렴할 뿐, 어떤 하나의 "참"값으로 수렴하지 않는다.

## 점근분포

일치성을 넘어, 중심극한정리는 점근분포를 준다:

$$\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} N(0, \sigma^2)$$

이로부터 신뢰구간과 가설검정을 구성할 수 있다:

$$\bar{X} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}} \quad (\sigma \text{를 아는 경우})$$

$$\bar{X} \pm t_{n-1,\alpha/2} \frac{S}{\sqrt{n}} \quad (\sigma \text{를 모르는 경우})$$

## 금융과의 연결

평균의 편향과 일치성을 이해하는 것은 금융에서 결정적이다:

- **수익률 추정**: $\bar{X}$가 일치하기는 하지만 수렴 속도 $O(1/\sqrt{n})$은 실제 수익률 예측에 쓰기에는 너무 느리다. 30년치 월별 자료($n = 360$)에서도 $\text{SE} \approx \sigma_{\text{월별}} / 19$로 여전히 상당하다.

- **정상성에 대한 우려**: 금융 수익률 분포는 시간에 따라 변하므로(국면 전환, 구조적 단절) 정상성 가정이 깨진다. 과거 수익률의 표본평균이 *현재의* 기대수익률을 추정하지 못할 수 있다.

- **평균회귀 검정**: 자산가격이 평균회귀하는지 검정하려면 여러 종속 구조 아래에서 $\bar{X}$의 수렴 성질에 세심한 주의가 필요하다.

- **고빈도 추정**: 고빈도 자료에서는 미시구조 잡음이 편향을 만든다. 틱 단위 가격의 "실현" 평균은 매수–매도 호가 튐 효과로 편향된다.

## 요약

표본평균은 매우 온건한 조건 아래에서 불편이고(모든 $n$에서 편향이 0) 일치한다($n \to \infty$일 때 $\mu$로 수렴). 수렴 속도는 $O(1/\sqrt{n})$으로, 최적이지만 실무적으로는 느리다. 거의 확실한 수렴(SLLN)은 확률수렴(WLLN)보다 강한 보장을 준다. 이런 성질 덕분에 $\bar{X}$가 모평균의 기본 추정량이 되지만, 느린 수렴 속도와 분포 가정에 대한 민감성은 특히 금융 응용에서 반드시 인식해야 한다.

## 핵심 공식

| 성질 | 결과 | 조건 |
|----------|--------|-----------|
| 불편성 | $E[\bar{X}] = \mu$ | $E[X_i] = \mu$ |
| 일치성 (WLLN) | $\bar{X}_n \xrightarrow{p} \mu$ | i.i.d., $E[|X|] < \infty$ |
| 강일치성 (SLLN) | $\bar{X}_n \to \mu$ a.s. | i.i.d., $E[|X|] < \infty$ |
| 평균제곱오차 속도 | $O(1/n)$ | $\text{Var}(X) < \infty$ |
| 표준오차 속도 | $O(1/\sqrt{n})$ | $\text{Var}(X) < \infty$ |
| 중심극한정리 | $\sqrt{n}(\bar{X}-\mu)/\sigma \to N(0,1)$ | i.i.d., $\text{Var}(X) < \infty$ |

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
$X_1, \ldots, X_n$이 $\mathbb{E}[X_i] = \mu$를 만족한다. $\bar X$가 불편임을 보여라. 여기에 독립성이 필요한가?

</div>

??? success "풀이"
    선형성에 의해(독립성 불필요): $\mathbb{E}[\bar X] = (1/n)\sum \mathbb{E}[X_i] = \mu$. $\square$

    불편성에는 평균이 같다는 것만 필요하다. 추정량의 *분산*은 독립성에 의존하지만 *기댓값*은 그렇지 않다.

<div class="drillbox" markdown>

**연습문제 2.**
상관된 자료: $X_i$가 공통 평균 $\mu$, 분산 $\sigma^2$, 쌍별 상관계수 $\rho$를 갖는다. (a) $\mathrm{Var}(\bar X)$를 유도하라. (b) $\bar X$는 일치하는가? (c) 헤지펀드 20개로 이루어진 펀드, 변동성 15%, $\rho = 0.4$일 때 표준오차를 구하라.

</div>

??? success "풀이"
    (a) $\mathrm{Var}(\bar X) = (1/n^2)[n\sigma^2 + n(n-1)\rho\sigma^2] = \sigma^2[1 + (n-1)\rho]/n$.

    (b) $n \to \infty$일 때 $\mathrm{Var}(\bar X) \to \rho\sigma^2$(양의 극한). $\rho = 0$이 아닌 한 $\bar X$는 **일치하지 않는다**. 양의 상관은 줄일 수 없는 분산의 바닥을 만든다.

    (c) $\mathrm{Var}(\bar X) = 0.15^2 \cdot 8.6/20 = 0.00968$. $\mathrm{SE} \approx 0.098$ (9.8%). 유효 독립 표본크기가 $\approx 2.4$ — 상관이 높아 펀드 20개에서 얻는 이득이 거의 없다.

<div class="drillbox" markdown>

**연습문제 3.**
**일치성의 정의.** 약대수의법칙과 강대수의법칙을 진술하고, 각각으로부터 $\bar X$가 $\mu$에 대해 일치함을 보여라.

</div>

??? success "풀이"
    **일치성:** $n \to \infty$일 때 어떤 수렴 방식으로 $\hat\theta_n \to \theta$.

    **약일치성**(확률수렴): $\bar X \xrightarrow{P} \mu$. **WLLN**에서 따라 나온다: 평균이 유한한 i.i.d. 자료에서 $P(|\bar X - \mu| > \varepsilon) \to 0$.

    **강일치성**(거의 확실하게): 확률 1로 $\bar X \to \mu$. **SLLN**에서 따라 나온다.

    증명 개요(Chebyshev를 통한 WLLN): 분산 $\sigma^2$이 유한하면 $P(|\bar X - \mu| > \varepsilon) \le \mathrm{Var}(\bar X)/\varepsilon^2 = \sigma^2/(n\varepsilon^2) \to 0$. $\square$

<div class="drillbox" markdown>

**연습문제 4.**
중앙값의 **편향–분산 맞바꿈.** 정규 자료에서 표본중앙값이 $\mu$에 대해 **불편**이지만 그 **평균제곱오차가 점근적으로** 표본평균의 평균제곱오차를 **초과**함을 보여라.

</div>

??? success "풀이"
    정규분포처럼 대칭인 분포에서는 모평균 = 모중앙값이고 둘 다 $\mu$이다. 표본중앙값은 (표본분포의 대칭성에 의해) $\mu$에 대해 불편이다.

    표본중앙값의 점근분산은 $\pi\sigma^2/(2n)$이고 평균은 $\sigma^2/n$이다. 비는 $\pi/2 \approx 1.57$이다.

    MSE(중앙값) $>$ MSE(평균), 인자는 $\pi/2$ — 로버스트성의 대가이다.

    대칭이 아닌 분포에서는 모평균 $\ne$ 모중앙값이므로 둘은 서로 다른 양을 겨냥한다. 서로를 비교하지 말고 각자의 목표와 비교해야 한다.

<div class="drillbox" markdown>

**연습문제 5.**
**유한모집단 표본추출의 효과.** 크기 $N$, 평균 $\mu$인 유한모집단에서 비복원으로 $X_i$를 뽑는다. $\mathrm{Var}(\bar X)$를 유도하라.

</div>

??? success "풀이"
    $\mathrm{Var}(\bar X) = (\sigma^2/n) \cdot (N - n)/(N - 1)$.

    유도: 서로 다른 두 추출의 공분산은 $-\sigma^2/(N-1)$이다(비복원 추출). 모든 쌍과 개별 분산을 더하고 $n^2$으로 나눈다.

    인자 $(N-n)/(N-1)$이 **유한모집단 수정(FPC)**이다:

    - $n = N$ (전수조사): FPC = 0, $\mathrm{Var}(\bar X) = 0$. 모두를 측정했다.
    - $n \ll N$: FPC $\approx 1$, $\mathrm{Var}(\bar X) \approx \sigma^2/n$. 표준 공식.

    $n/N < 5\%$이면 흔히 무시한다. 감사, 재검표, 소규모 모집단 조사에서는 중요하다.

<div class="drillbox" markdown>

**연습문제 6.**
**적률법 추정량의 편향.** $[1, \infty)$ 위의 Pareto$(\alpha)$에서 $\mathbb{E}[X] = \alpha/(\alpha - 1)$이다. 적률법: $\hat\alpha = \bar X/(\bar X - 1)$. 불편인가? 일치하는가?

</div>

??? success "풀이"
    **일치성:** 대수의법칙에 의해 $\bar X \to \mu = \alpha/(\alpha - 1)$. 따라서 $\hat\alpha \to \mu/(\mu - 1) = \alpha$. 일치한다.

    **불편성:** $\mathbb{E}[\hat\alpha] = \mathbb{E}[\bar X/(\bar X - 1)]$. ($g(x) = x/(x-1)$이 $x > 1$에서 볼록이므로) Jensen 부등식에 의해 $\mathbb{E}[g(\bar X)] > g(\mathbb{E}[\bar X]) = \alpha$이다. 따라서 **적률법 추정량은 위쪽으로 편향**되어 있다.

    큰 $n$에서는 $\bar X \to \mu$이고 Jensen 간격 $\to 0$이므로 편향 $\to 0$이다. 점근적으로는 불편이지만 유한표본에서는 편향되어 있다.

    대부분의 적률법 추정량이 이 양상을 보인다: (대수의법칙에 의해) 일치하지만 유한표본에서는 편향된다. 편향은 $O(1/n)$으로, 표준편차의 $O(1/\sqrt n)$보다 빨리 사라진다.
