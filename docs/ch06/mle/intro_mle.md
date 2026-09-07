# 최대가능도추정 소개

## 개요

최대가능도추정(MLE)은 가능도함수를 최대화하여 통계모형의 모수를 추정하는 방법이다. 가능도함수는 특정 모수를 갖는 모형이 관측된 자료를 얼마나 잘 설명하는지를 잰다. MLE 추정값은 관측된 자료를 가장 그럴듯하게 만드는 모수이다.

더 쉽게 말하면 MLE는 관측된 자료를 가장 "그럴듯하게" 만드는 모수값을 찾는다. 적률법 같은 다른 방법에 비해 MLE는 일치성과 점근 효율성을 비롯한 좋은 성질 때문에 자주 선호된다.

## 수학적 정식화

MLE는 형식적으로 다음과 같이 정의된다:

$$
\hat{\theta}_{MLE} = \arg \max_{\theta} L(\theta \mid \mathbf{x}) = \arg \max_{\theta} \prod_{i=1}^n f(x_i \mid \theta)
$$

여기서 $\theta$는 추정하려는 모수이고, $L(\theta \mid \mathbf{x})$는 관측된 자료점 $\mathbf{x} = (x_1, x_2, \ldots, x_n)$에서 평가한 확률밀도함수(이산 자료에서는 확률질량함수)의 곱인 가능도함수이다.

## 로그가능도

계산의 편의를 위해 흔히 로그가능도함수를 사용한다:

$$
\log L(\theta \mid \mathbf{x}) = \sum_{i=1}^n \log f(x_i \mid \theta)
$$

이 변환은 확률의 곱을 로그확률의 합으로 바꾸어 최대화를 간단하게 만든다. $\log$가 단조증가함수이므로 로그가능도를 최대화하는 것은 가능도 자체를 최대화하는 것과 동등하다.

## 최대가능도 원리

MLE 틀에서의 핵심 동치 관계:

$$
\text{argmax}_{\theta}\; L
\quad\Leftrightarrow\quad
\text{argmax}_{\theta}\; \ell
\quad\Leftrightarrow\quad
\text{argmin}_{\theta}\; J
$$

여기서 $L$은 가능도, $\ell = \log L$은 로그가능도, $J = -\ell$은 비용함수(음의 로그가능도)이다.

## MLE의 성질

| 성질 | 설명 |
|----------|-------------|
| **일치성** | $n \to \infty$일 때 $\hat{\theta}_{MLE} \xrightarrow{P} \theta_0$ |
| **점근정규성** | $\sqrt{n}(\hat{\theta}_{MLE} - \theta_0) \xrightarrow{d} N(0, I(\theta_0)^{-1})$ |
| **점근 효율성** | 점근적으로 Cramér–Rao 하한을 달성한다 |
| **불변성** | $\hat{\theta}$가 $\theta$의 MLE이면 $g(\hat{\theta})$는 $g(\theta)$의 MLE이다 |

## 요약

MLE는 모수 추정에 대한 원리적이고 범용적인 접근을 제공한다. 다음과 자연스럽게 연결된다:

- 기계학습의 **비용함수** (분류에서 교차엔트로피 손실 = 음의 로그가능도)
- **베이즈 추론** (MLE는 평평한 사전분포를 쓴 MAP 추정값이다)
- **정보이론** (모형과 자료 사이의 KL 발산 최소화)

## 연습문제

**연습문제 1.**
일간 로그수익률이 $r_t \sim N(\mu, \sigma^2)$이다. $n = 252$, $\bar r = 0.0004$, $s = 0.015$일 때 (a) $\mu_{\text{ann}} = 252\mu$와 $\sigma_{\text{ann}} = \sigma\sqrt{252}$의 MLE. (b) 점근정규성을 이용한 95% 신뢰구간. (c) $\hat\mu$가 $\hat\sigma$보다 훨씬 불안정한 이유는?

??? success "풀이"
    (a) $\hat\mu_{\text{ann}} = 252 \cdot 0.0004 = 0.1008$ (10.08%). $\hat\sigma_{\text{ann}} = 0.015 \sqrt{252} \approx 0.2381$ (23.81%).

    (b) $\mathrm{SE}(\hat\mu) = s/\sqrt n = 0.015/\sqrt{252} \approx 0.000945$. $\mathrm{SE}(\hat\mu_{\text{ann}}) = 252 \cdot 0.000945 \approx 0.238$. 95% 신뢰구간: $(-0.366, 0.568)$.

    $\sigma$에 대해서는 점근 표준오차가 $\sigma/\sqrt{2n}$이므로 $\mathrm{SE}(\hat\sigma_{\text{ann}}) \approx 0.0106$이다. 95% 신뢰구간: $(0.217, 0.259)$.

    (c) $\hat\mu_{\text{ann}}$의 표준오차 ≈ 0.238로 점추정값 0.101보다 *크다*. 신뢰구간이 $-37\%$에서 $+57\%$까지 걸쳐 있다. 변동성의 신뢰구간은 (24% 주위로 대략 ±2%포인트로) 좁다. **1년 시계에서 추세는 절망적으로 불안정하지만 변동성은 잘 추정된다.** 미래 수익률을 예측하는 데 과거 표본평균에 의존하는 것은 금융에서 고전적인 함정이다.

---

**연습문제 2.**
**정규분포의 MLE 유도.** i.i.d. $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$이 주어졌을 때 $\hat\mu$와 $\hat\sigma^2$을 유도하라.

??? success "풀이"
    로그가능도: $\ell(\mu, \sigma^2) = -(n/2)\ln(2\pi\sigma^2) - (1/(2\sigma^2))\sum(X_i - \mu)^2$.

    $\partial\ell/\partial\mu = 0 \Rightarrow \hat\mu = \bar X$.

    $\partial\ell/\partial\sigma^2 = -n/(2\sigma^2) + \sum(X_i - \mu)^2/(2\sigma^4) = 0 \Rightarrow \hat\sigma^2_{\text{MLE}} = (1/n)\sum(X_i - \bar X)^2$.

    MLE가 $n - 1$이 아니라 $n$으로 나눔에 유의하라. 편향되어 있다($\mathbb{E}[\hat\sigma^2] = (n-1)\sigma^2/n$). 불편추정을 하려면 Bessel 수정을 사용한다.

---

**연습문제 3.**
**Bernoulli/Binomial의 MLE.** 표본이 $X \sim \mathrm{Binomial}(n, p)$이다. $\hat p_{\text{MLE}}$를 유도하라.

??? success "풀이"
    가능도: $L(p) = \binom{n}{X} p^X (1-p)^{n-X} \propto p^X(1-p)^{n-X}$.

    로그가능도: $\ell(p) = X\ln p + (n-X)\ln(1-p)$.

    $\ell'(p) = X/p - (n-X)/(1-p) = 0$.

    $X(1-p) = (n-X)p \Rightarrow X = np \Rightarrow \hat p = X/n$.

    MLE는 표본비율이다. 불편이다: $\mathbb{E}[\hat p] = p$.

---

**연습문제 4.**
**가능도와 확률.** MLE의 용어에서 "가능도"와 "확률"을 구별하라. 가능도가 $\theta$의 함수이면서도 $\theta$에 대한 확률밀도가 아닌 이유는?

??? success "풀이"
    **확률:** $P(X = x \mid \theta)$ — $\theta$를 고정한 $x$의 함수이다. $x$에 대해 합하거나 적분하면 1이 된다.

    **가능도:** $L(\theta) = P(X = x \mid \theta)$ — 수식은 같지만 *관측되어 고정된* $x$에 대해 $\theta$의 함수로 본 것이다. $\theta$에 대해 적분해도 1이 되지 않는다.

    **왜 $\theta$에 대한 밀도가 아닌가:** 빈도주의 통계학에서 $\theta$는 확률변수가 아니라 *모수*이다. 가능도는 $\theta$ 값들을 자료를 얼마나 잘 설명하는지로 순위 매길 뿐, 그것이 얼마나 확률적으로 있음 직한지로 매기지 않는다. MLE는 가능도를 최대화하는 $\theta$를 고르지만 이것이 "가장 확률이 높은 모수"인 것은 아니다.

    베이즈 추론에서는 $\theta$가 확률변수가 되고 *사후분포* $\pi(\theta \mid x) \propto L(\theta) \pi(\theta)$를 계산하는데, 이것은 $\theta$에 대한 밀도가 **맞다**. 가능도 자체는 그대로이고 달라지는 것은 틀이다.

---

**연습문제 5.**
**MLE의 점근정규성.** 그 결과를 서술하고 Bernoulli의 경우에 확인하라.

??? success "풀이"
    **MLE의 점근정규성:**

    $$
    \sqrt n(\hat\theta_{\text{MLE}} - \theta) \xrightarrow{d} N(0, 1/I(\theta))
    $$

    여기서 $I(\theta) = -\mathbb{E}[\partial^2 \log f/\partial\theta^2]$는 관측값 하나당 Fisher 정보량이다.

    **Bernoulli에서의 확인:** $\log f = x\log p + (1-x)\log(1-p)$. 2계도함수는 $-x/p^2 - (1-x)/(1-p)^2$이다. 기댓값을 취하면 $-p/p^2 - (1-p)/(1-p)^2 = -1/p - 1/(1-p) = -1/[p(1-p)]$.

    따라서 $I(p) = 1/[p(1-p)]$이고 $\mathrm{Var}(\hat p) = p(1-p)/n$이다. 표본비율에 대한 중심극한정리와 곧바로 일치한다.

    함의: MLE는 점근적으로 **Cramér-Rao 하한**을 달성하며 점근적으로 효율적이다. 어떤 불편추정량도 이보다 작은 점근분산을 가질 수 없다.

---

**연습문제 6.**
**MLE가 실패하는 경우.** 다음 각각의 예를 하나씩 들라: (a) MLE가 존재하지 않는 경우; (b) MLE가 경계에 있는 경우; (c) MLE가 일치하지 않는 경우.

??? success "풀이"
    **(a) MLE가 존재하지 않는 경우:** $X \sim \mathrm{Uniform}(0, \theta)$에서 가능도는 $\theta \ge \max X_i$일 때 $1/\theta^n$이고 그 아래에서는 0이다. $\theta \to \infty$이면 $L \to 0$이고, $\theta \to \max X_i$이면 $L \to (1/\max X_i)^n$으로 *상한*에 이르지만 그 지점이 경계이다. 엄밀히 말해 내부에 최댓값이 없으며, MLE는 경계값 $\hat\theta = \max X_i$이다.

    **(b) MLE가 경계에 있는 경우:** 위의 균등분포 예이다. 경계에 있는 MLE는 비표준 점근분포를 가지며($N(0, 1/I)$가 아니다) 수렴 속도가 $\sqrt n$이 아니라 $n$이다.

    **(c) MLE가 일치하지 않는 경우:** Neyman-Scott 문제이다. $i = 1, \ldots, n$, $j = 1, 2$에 대해 관측값이 $X_{ij} \sim N(\mu_i, \sigma^2)$이다. 모수의 개수($\mu_i$들 + $\sigma^2$)가 표본크기와 함께 늘어난다. $\hat\sigma^2_{\text{MLE}} \to \sigma^2/2$로 $\sigma^2$이 아니다. 모수공간의 차원이 $n$에 비례해 커지므로 일치하지 않는다.

    이런 실패 양상들이 편향 보정 MLE, 벌점 가능도, 프로파일 가능도, (베이즈의) 주변가능도 같은 개선을 이끌어 냈다.
