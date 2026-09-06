# μ와 σ²의 MLE

## 들어가며

**Gaussian(정규) 분포 모수의 최대가능도추정량**은 통계학에서 가장 중요한 결과에 속한다. $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$에 대해 MLE는 평균 $\mu$와 분산 $\sigma^2$ 모두에 대한 닫힌 형태의 추정량을 준다. 이 절에서는 이 추정량들을 유도하고, 성질을 분석하며, 그 결과를 더 넓은 추정이론과 연결한다.

## 정규 로그가능도

$N(\mu, \sigma^2)$에서 얻은 i.i.d. 표본 $x_1, \ldots, x_n$에 대해 로그가능도는:

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2$$

## MLE의 유도

### mu의 MLE
$\mu$에 대해 미분하면:

$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = \frac{n}{\sigma^2}(\bar{x} - \mu)$$

0으로 놓으면:

$$\bar{x} - \mu = 0 \implies \boxed{\hat{\mu}_{\text{MLE}} = \bar{X} = \frac{1}{n}\sum_{i=1}^n X_i}$$

평균의 MLE는 **표본평균**이다.

### sigma-squared의 MLE
($\sigma^2$을 하나의 변수로 보고) $\sigma^2$에 대해 미분하면:

$$\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^n (x_i - \mu)^2$$

0으로 놓고 $\hat{\mu} = \bar{x}$를 대입하면:

$$-\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^n (x_i - \bar{x})^2 = 0$$

$$\sigma^2 = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2$$

$$\boxed{\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2}$$

분산의 MLE는 $n-1$이 **아니라** $n$으로 나눈다.

### 확인: 2계 조건

$(\hat{\mu}, \hat{\sigma}^2)$에서 평가한 Hessian 행렬은:

$$H = \begin{pmatrix} -n/\hat{\sigma}^2 & 0 \\ 0 & -n/(2\hat{\sigma}^4) \end{pmatrix}$$

대각 성분이 모두 음수이므로 음정부호이며, 최댓값임이 확인된다.

## Gaussian MLE의 성질

### mu-hat = X-bar의 성질
| 성질 | 결과 |
|----------|--------|
| 편향 | $E[\hat{\mu}] = \mu$ (불편) |
| 분산 | $\text{Var}(\hat{\mu}) = \sigma^2/n$ |
| 분포 | 정확히 $\hat{\mu} \sim N(\mu, \sigma^2/n)$ |
| 효율성 | CRLB 달성, MVUE |
| 충분성 | ($\sigma^2$이 주어지면) $\mu$에 대해 충분 |
| 일치성 | $\hat{\mu} \xrightarrow{p} \mu$ |

### sigma-squared (MLE)의 성질
| 성질 | 결과 |
|----------|--------|
| 편향 | $E[\hat{\sigma}^2] = \frac{n-1}{n}\sigma^2$ (편향됨) |
| 편향의 크기 | $\text{Bias} = -\sigma^2/n$ |
| 분포 | $n\hat{\sigma}^2/\sigma^2 \sim \chi^2_{n-1}$ |
| 분산 | $\text{Var}(\hat{\sigma}^2) = \frac{2(n-1)}{n^2}\sigma^4$ |
| 평균제곱오차 | $\frac{2n-1}{n^2}\sigma^4$ |
| 일치성 | $\hat{\sigma}^2 \xrightarrow{p} \sigma^2$ |
| 점근적 불편성 | $n \to \infty$일 때 $E[\hat{\sigma}^2] \to \sigma^2$ |

### 독립성

$\hat{\mu}$과 $\hat{\sigma}^2$은 (Cochran 정리에 의해) **독립**이다. 정규분포에만 있는 특별한 성질이며 $t$-분포를 유도하는 데 결정적이다.

## Fisher 정보행렬

$(\mu, \sigma^2)$에 대한 Fisher 정보행렬은:

$$I(\mu, \sigma^2) = \begin{pmatrix} n/\sigma^2 & 0 \\ 0 & n/(2\sigma^4) \end{pmatrix}$$

비대각 성분이 0이라는 사실은 $\mu$와 $\sigma^2$이 서로 독립적인 정보를 담고 있음을 확인해 준다.

### Cramér-Rao 하한

$$\text{Var}(\hat{\mu}) \geq \frac{\sigma^2}{n}, \quad \text{Var}(\hat{\sigma}^2) \geq \frac{2\sigma^4}{n}$$

$\mu$의 MLE는 CRLB를 정확히 달성한다. $\sigma^2$의 MLE는 유한표본에서는 CRLB에 도달하지 *못하지만*(분산이 $2(n-1)\sigma^4/n^2 < 2\sigma^4/n$이다) 점근적으로는 도달한다.

## 다른 모수화: (mu, sigma)
$(\mu, \sigma^2)$ 대신 $(\mu, \sigma)$로 모수화하면 $\sigma$의 MLE는:

$$\hat{\sigma}_{\text{MLE}} = \sqrt{\hat{\sigma}^2_{\text{MLE}}} = \sqrt{\frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2}$$

이는 MLE의 **불변성**에서 따라 나온다: $\hat{\theta}$가 $\theta$의 MLE이면 $g(\hat{\theta})$는 $g(\theta)$의 MLE이다.

$\hat{\sigma}_{\text{MLE}}$은 $\sigma$에 대해 편향되어 있음에 유의하라(Jensen 부등식에 의해 $E[\sqrt{X}] < \sqrt{E[X]}$).

## 편향 보정 추정량

$\sigma^2$의 불편추정량은:

$$S^2 = \frac{n}{n-1}\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2$$

비교:

| 추정량 | 공식 | $E[\cdot]$ | 평균제곱오차 |
|-----------|---------|------------|-----|
| MLE | $\frac{1}{n}\sum(X_i - \bar{X})^2$ | $\frac{n-1}{n}\sigma^2$ | $\frac{2n-1}{n^2}\sigma^4$ |
| Bessel | $\frac{1}{n-1}\sum(X_i - \bar{X})^2$ | $\sigma^2$ | $\frac{2}{n-1}\sigma^4$ |
| 평균제곱오차 최적 | $\frac{1}{n+1}\sum(X_i - \bar{X})^2$ | $\frac{n-1}{n+1}\sigma^2$ | 최소 |

## 로그가능도 곡면

로그가능도함수 $\ell(\mu, \sigma^2)$은 $(\mu, \sigma^2)$ 평면 위의 곡면을 이룬다:

- $\sigma^2$을 고정하면 $\ell$은 $\mu$에 대해 위로 볼록한(아래로 열린) 포물선이며 $\bar{X}$에서 최대가 된다
- $\mu$를 고정하면 $\ell$은 $\sigma^2$의 오목함수이다
- 전역 최댓값은 $(\bar{X}, \hat{\sigma}^2)$에 있다
- 로그가능도가 일정한 등고선은 (큰 $n$에서 근사적으로) MLE를 중심으로 하는 타원이다

## 가능도로부터의 신뢰영역

### mu에 대해 (σ²를 아는 경우)

$$\bar{X} \pm z_{\alpha/2}\frac{\sigma}{\sqrt{n}}$$

### mu에 대해 (σ²를 모르는 경우)

$$\bar{X} \pm t_{n-1, \alpha/2}\frac{S}{\sqrt{n}}$$

여기서 $S = \sqrt{S^2}$이고 $t_{n-1}$은 자유도 $n-1$인 Student $t$-분포이다.

### sigma-squared에 대해

$$\left(\frac{(n-1)S^2}{\chi^2_{n-1, \alpha/2}}, \quad \frac{(n-1)S^2}{\chi^2_{n-1, 1-\alpha/2}}\right)$$

## 제약 아래에서의 MLE

### 평균을 아는 경우

$\mu = \mu_0$이 알려져 있으면 $\sigma^2$의 제약 MLE는:

$$\hat{\sigma}^2_{\mu_0} = \frac{1}{n}\sum_{i=1}^n (X_i - \mu_0)^2$$

($\mu$를 추정하는 경우와 달리) 이 추정량은 불편이다.

### 평균이 같은 경우 (합동분산)

공통 분산을 갖는 두 집단 $X_1, \ldots, X_{n_1} \sim N(\mu_1, \sigma^2)$과 $Y_1, \ldots, Y_{n_2} \sim N(\mu_2, \sigma^2)$에 대해 $\sigma^2$의 MLE는:

$$\hat{\sigma}^2_{\text{pooled}} = \frac{\sum(X_i - \bar{X})^2 + \sum(Y_j - \bar{Y})^2}{n_1 + n_2}$$

불편 버전은 $n_1 + n_2 - 2$로 나눈다.

## 금융과의 연결

- **수익률 모형화**: 로그수익률이 $r_t \sim N(\mu, \sigma^2)$이라는 가정이 많은 금융 모형의 토대이다. MLE $\hat{\mu} = \bar{r}$과 $\hat{\sigma}^2 = \frac{1}{n}\sum(r_t - \bar{r})^2$이 표준적인 추정값이다.

- **Black-Scholes**: 이 모형은 $\log(S_T/S_t) \sim N((\mu - \sigma^2/2)(T-t), \sigma^2(T-t))$를 가정한다. 과거 수익률로 구한 변동성의 MLE가 핵심 입력이다.

- **VaR 추정**: 정규성 아래에서 $\text{VaR}_\alpha = -(\hat{\mu} + z_\alpha \hat{\sigma})$로, Gaussian MLE를 직접 쓴다.

- **포트폴리오 이론**: Markowitz 최적화는 $\hat{\mu}$과 $\hat{\Sigma}$(표본평균 벡터와 공분산행렬)를 쓰는데, 이들이 다변량 Gaussian MLE이다.

- **정규성 검정**: Gaussian MLE를 쓰기 전에 정규분포가 적절한지 검정해야 한다. 금융 수익률은 흔히 두꺼운 꼬리를 보이므로 Gaussian MLE가 최적이 아니게 된다.

## 요약

Gaussian MLE — $\hat{\mu} = \bar{X}$과 $\hat{\sigma}^2 = \frac{1}{n}\sum(X_i - \bar{X})^2$ — 는 닫힌 형태이고 계산이 아주 쉬우며 훌륭한 성질을 갖는다. 평균추정량은 불편이고 효율적이며, 분산추정량은 편향되어 있지만 일치하고 불편 대안보다 평균제곱오차가 작다. (정규분포에만 있는) 이들의 독립성 덕분에 $t$와 $\chi^2$ 분포를 통한 정확한 추론이 가능하다. 이 추정량들은 고전적 통계추론의 토대이며 금융 모수추정의 출발점이다.

## 핵심 공식

| 양 | 공식 |
|----------|---------|
| $\hat{\mu}_{\text{MLE}}$ | $\bar{X}$ |
| $\hat{\sigma}^2_{\text{MLE}}$ | $\frac{1}{n}\sum(X_i - \bar{X})^2$ |
| $\mu$의 Fisher 정보량 | $I_n(\mu) = n/\sigma^2$ |
| $\sigma^2$의 Fisher 정보량 | $I_n(\sigma^2) = n/(2\sigma^4)$ |
| $\hat{\mu}$의 분포 | $N(\mu, \sigma^2/n)$ |
| $n\hat{\sigma}^2/\sigma^2$의 분포 | $\chi^2_{n-1}$ |
| $t$-통계량 | $(\bar{X}-\mu)/(S/\sqrt{n}) \sim t_{n-1}$ |

## 연습문제

**연습문제 1.**
$N(\mu, \sigma^2)$에 대해 $\hat\mu_{\text{MLE}} = \bar X$와 $\hat\sigma^2_{\text{MLE}} = (1/n)\sum(X_i - \bar X)^2$을 유도하고 2계 조건을 확인하라.

??? success "연습문제 1 풀이"
    로그가능도: $\ell(\mu, \sigma^2) = -(n/2)\ln(2\pi\sigma^2) - (1/(2\sigma^2))\sum(x_i - \mu)^2$.

    $\partial\ell/\partial\mu = (1/\sigma^2)\sum(x_i - \mu) = 0 \Rightarrow \hat\mu = \bar X$.

    $\partial\ell/\partial\sigma^2 = -n/(2\sigma^2) + (1/(2\sigma^4))\sum(x_i - \mu)^2 = 0$이고, $\hat\mu$을 대입하면 $\hat\sigma^2_{\text{MLE}} = (1/n)\sum(x_i - \bar X)^2$을 얻는다.

    **Hessian:** MLE에서 $\partial^2\ell/\partial\mu^2 = -n/\sigma^2 < 0$, $\partial^2\ell/\partial(\sigma^2)^2 = -n/(2\sigma^4) < 0$이고, 혼합편도함수는 기댓값이 0이다. 음정부호이므로 최댓값임이 확인된다.

---

**연습문제 2.**
$N(\mu, \sigma^2)$의 **Fisher 정보행렬.** 비대각 성분이 0임을 보이고 $\bar X$가 CRLB를 정확히 달성함을 확인하라.

??? success "연습문제 2 풀이"
    $I_{\mu\mu} = n/\sigma^2$, $I_{\sigma^2 \sigma^2} = n/(2\sigma^4)$, $I_{\mu \sigma^2} = \mathbb{E}[-(X-\mu)/\sigma^4] = 0$.

    Fisher 정보행렬이 대각이다: $\mu$와 $\sigma^2$은 **직교 모수**이다. 하나를 추정하는 것이 다른 하나를 추정하는 점근분산에 영향을 주지 않는다.

    $\mathrm{Var}(\bar X) = \sigma^2/n = 1/I_{\mu\mu}$ — $\bar X$는 점근적으로만이 아니라 임의의 $n$에서 CRLB를 **정확히** 달성한다. $\mu$의 MLE는 완전히 효율적이다.

    $\sigma^2$의 경우: $\mathrm{Var}(\hat\sigma^2_{\text{MLE}}) = 2(n-1)\sigma^4/n^2$, CRLB $= 2\sigma^4/n$. CRLB보다 약간 위이며 점근적으로 효율적이다.

---

**연습문제 3.**
**MLE의 불변성.** (a) $\sigma$, (b) $\mathrm{CV} = \sigma/\mu$, (c) 99번째 백분위수 $\mu + 2.326\sigma$의 MLE를 구하라.

??? success "연습문제 3 풀이"
    MLE 불변성에 의해 $\widehat{g(\theta)} = g(\hat\theta_{\text{MLE}})$:

    (a) $\hat\sigma = \sqrt{\hat\sigma^2_{\text{MLE}}}$.

    (b) $\widehat{\mathrm{CV}} = \hat\sigma/\bar X$.

    (c) $\widehat{q_{0.99}} = \bar X + 2.326 \hat\sigma$.

    **주의:** 불변성은 MLE 점추정값은 보존하지만 불편성은 보존하지 않는다. $\hat\sigma$은 편향되어 있으며(오목함수 $\sqrt{\cdot}$에 대한 Jensen 부등식), $c_4$ 상수로 편향을 보정할 수 있다.

---

**연습문제 4.**
**$\mu = 0$이라는 제약 아래의 MLE.** $\mu$가 0임을 알 때 $\hat\sigma^2$을 유도하라. 제약 없는 MLE와 분산을 비교하라.

??? success "연습문제 4 풀이"
    $\mu = 0$이면 $\hat\sigma^2_0 = (1/n) \sum X_i^2$이다. $\mathbb{E}[\hat\sigma^2_0] = (1/n) \cdot n\sigma^2 = \sigma^2$ — **불편**이다($\mu$를 추정한 것이 아니라 알고 있으므로 Bessel 수정이 필요 없다).

    $n\hat\sigma^2_0/\sigma^2 \sim \chi^2_n$이므로 $\mathrm{Var}(\hat\sigma^2_0) = 2\sigma^4/n$이다.

    제약 없는 경우: $\mathrm{Var}(\hat\sigma^2_{\text{MLE}}) = 2\sigma^4(n-1)/n^2$.

    제약 있는 추정량은 (자유도가 하나 더 많아) 분산이 약간 크지만 불편이다. $\mu$를 추정하는 "대가"는 자유도 $-1$이다.

---

**연습문제 5.**
**모수적 VaR.** 일별 수익률 252개에서 $\hat\mu = 0.0003$, $\hat\sigma = 0.012$를 얻었다. (a) 1일 99% VaR. (b) 제곱근 규칙에 의한 10일 VaR. (c) 참 초과첨도가 3이라면 정규 VaR는 위험을 과대평가하는가, 과소평가하는가?

??? success "연습문제 5 풀이"
    (a) $\mathrm{VaR}_{0.99}^{\text{1일}} = -(\hat\mu + z_{0.01} \hat\sigma) = -(0.0003 - 2.326 \cdot 0.012) = 0.0276$ (2.76% 손실).

    (b) $\mathrm{VaR}_{0.99}^{\text{10일}} = \sqrt{10} \cdot 0.0276 \approx 0.0873$ (8.73%). i.i.d.이고 추세가 0이라는 가정 아래에서 유효하다.

    (c) 두꺼운 꼬리(초과첨도 3 > 0)는 참 99번째 백분위수 손실이 정규 예측보다 *크다*는 뜻이다. 정규 VaR는 실제 위험을 **과소평가**한다. 보수적인 실무: 위험관리에는 $t$-분포 기반 VaR나 경험적 분위수를 쓰라.

---

**연습문제 6.**
**몬테카를로 검증.** $N(5, 9)$에서 $n = 20$인 표본 10000개를 모의실험하라. $\mathbb{E}[\hat\mu], \mathbb{E}[\hat\sigma^2_{\text{MLE}}], \mathbb{E}[S^2]$을 확인하라.

??? success "연습문제 6 풀이"
    ```python
    import numpy as np
    rng = np.random.default_rng(0)
    R, n, mu, var = 10_000, 20, 5.0, 9.0
    samples = rng.normal(mu, np.sqrt(var), (R, n))
    mu_hat = samples.mean(axis=1)
    sig2_mle = samples.var(axis=1, ddof=0)
    s2 = samples.var(axis=1, ddof=1)
    print(f"E[mu_hat]   = {mu_hat.mean():.4f}   (true {mu})")
    print(f"E[sig2_MLE] = {sig2_mle.mean():.4f} (true {(n-1)/n*var:.4f})")
    print(f"E[S^2]      = {s2.mean():.4f}       (true {var})")
    ```

    예상 결과:

    - $\mathbb{E}[\bar X] \approx 5.00$ (불편).
    - $\mathbb{E}[\hat\sigma^2_{\text{MLE}}] \approx 8.55 = (19/20) \cdot 9$ ($\sigma^2/n$만큼 아래로 편향).
    - $\mathbb{E}[S^2] \approx 9.00$ (Bessel 수정, 불편).

    이론적 결과가 확인되며 분산에 대한 MLE의 편향이 드러난다.
