# 연습문제: 일반적인 통계적 추정

## 개념

**연습문제 1.** $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Uniform}(0, \theta)$라 하자. $\theta$의 추정량 두 개를 생각한다.

- $\hat{\theta}_1 = 2\bar{X}$
- $\hat{\theta}_2 = \frac{n+1}{n}X_{(n)}$, 여기서 $X_{(n)} = \max(X_1, \ldots, X_n)$

**(a)** 두 추정량이 모두 불편임을 보여라.

**(b)** 각 추정량의 분산을 계산하라. (힌트: $X_{(n)}$의 PDF는 $0 \leq x \leq \theta$에서 $f_{X_{(n)}}(x) = \frac{n}{\theta^n}x^{n-1}$이다.)

**(c)** 어느 추정량의 MSE가 더 작은가? 답이 $n$에 의존하는가?

**연습문제 2.** $\hat{\theta}$가 $\theta$의 불편추정량이고 $\text{Var}(\hat{\theta}) = 0$이면 확률 1로 $\hat{\theta} = \theta$임을 증명하라.

**연습문제 3.** $X_1, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$라 하자. $0 \leq w \leq 1$에 대해 가중추정량 $\hat{\mu}_w = w X_1 + (1-w)\bar{X}$를 생각한다.

**(a)** 모든 $w$에 대해 $\hat{\mu}_w$가 불편임을 보여라.

**(b)** $\text{Var}(\hat{\mu}_w)$를 최소화하는 $w$를 구하라.

**(c)** $w = 0$일 때와 $w = 1$일 때 무슨 일이 일어나는가?

**연습문제 4.** $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Bernoulli}(\theta)$에서 베이즈에서 착상을 얻은 추정량으로 $\theta$를 추정한다.

$$
\hat{\theta}_B = \frac{\sum X_i + a}{n + a + b}
$$

**(a)** $\text{Bias}(\hat{\theta}_B)$와 $\text{Var}(\hat{\theta}_B)$를 계산하라.

**(b)** $a = b = \sqrt{n}/2$일 때 $\hat{\theta}_B$가 편향되어 있지만 일치함을 보여라.

**(c)** MSE를 구하고 $\theta = 0.5$, $n = 10$일 때 MLE $\hat{\theta} = \bar{X}$와 비교하라.

## 계산

**연습문제 5.** 기하분포 $P(X = k) = (1-p)^{k-1}p$, $k = 1, 2, \ldots$ 의 MLE를 유도하라.

**(a)** 표본 $x_1, \ldots, x_n$에 대한 로그가능도를 쓰라.

**(b)** $\hat{p}_{\text{MLE}}$를 구하라.

**(c)** $E[X] = 1/p$를 이용해 적률추정량 $\hat{p}_{\text{MoM}}$을 구하라.

**(d)** 둘이 같은가?

**연습문제 6.** $x \geq 1$에서 PDF가 $f(x; \alpha) = \alpha x^{-(\alpha+1)}$인 Pareto 분포에 대해:

**(a)** $\alpha$의 MLE를 구하라.

**(b)** $E[X] = \frac{\alpha}{\alpha - 1}$($\alpha > 1$)을 이용해 적률추정량을 구하라.

**(c)** Fisher 정보량 $I(\alpha)$와 크라메르-라오 하한(CRLB)을 계산하라.

**(d)** MLE가 효율적인가?

**연습문제 7.** 다음을 수행하는 모의실험을 작성하라.

**(a)** $\text{Gamma}(\alpha = 2, \beta = 3)$에서 크기 $n = 30$인 표본 10,000개를 생성한다.

**(b)** 각 표본에서 $\alpha$와 $\beta$의 MLE와 적률추정값을 계산한다.

**(c)** MLE와 적률추정량의 경험적 편향, 분산, MSE를 비교한다.

**(d)** $n = 5, 10, 30, 100, 500$에 대해 반복하고 MSE를 $n$의 함수로 그린다.

**연습문제 8 (Fisher 정보량).** Poisson 분포 $f(x; \lambda) = \frac{\lambda^x e^{-\lambda}}{x!}$에 대해:

**(a)** Fisher 정보량 $I(\lambda)$를 계산하라.

**(b)** $\lambda$ 추정에 대한 CRLB를 서술하라.

**(c)** MLE $\hat{\lambda} = \bar{X}$가 효율적임을 보여라.

**(d)** $g(\lambda) = e^{-\lambda} = P(X = 0)$을 추정할 때의 Fisher 정보량을 계산하고 이 함수에 대한 CRLB를 구하라.

## 응용

**연습문제 9 (금융).** 주식의 일별 로그수익률은 흔히 $r_t \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$로 모형화한다.

**(a)** 252거래일 자료에서 $\bar{r} = 0.0004$, $s = 0.015$일 때 연율 평균수익률 $\mu_{\text{annual}} = 252\mu$와 연율 변동성 $\sigma_{\text{annual}} = \sigma\sqrt{252}$의 MLE를 계산하라.

**(b)** MLE의 점근정규성을 이용해 두 값에 대한 근사 95% 신뢰구간을 구성하라.

**(c)** 일별 수익률의 표본평균은 잡음이 매우 크다. $\hat{\mu}_{\text{annual}}$의 표준오차를 계산하고, 변동성 추정에 비해 기대수익률 추정이 왜 어려운지 논하라.

**연습문제 10.** 보험 청구액은 흔히 Gamma 분포를 따른다.

**(a)** 다음 청구 자료(단위: 천)에 대해 Gamma 모수 $\alpha$와 $\beta$의 적률추정값을 계산하라: 2.1, 0.8, 3.5, 1.2, 5.7, 0.4, 2.8, 1.9, 4.3, 0.6.

**(b)** `scipy.stats.gamma.fit()`으로 MLE 추정값을 계산하라.

**(c)** 히스토그램에 밀도를 겹쳐 그려 적합된 분포를 시각적으로 비교하라.
