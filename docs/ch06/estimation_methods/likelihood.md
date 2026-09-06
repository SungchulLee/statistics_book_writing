# 가능도함수

## 소개

**가능도함수**는 모수적 통계 추론의 초석이다. 관측된 자료가 주어졌을 때 가능도함수는 각 후보 모수값이 그 자료를 만들어 냈을 법한 정도를 잰다. 모수를 고정하고 결과에 확률을 부여하는 확률분포와 달리, 가능도함수는 자료를 고정하고 모수를 변수로 다룬다.

가능도함수 위에 세워진 최대가능도추정(MLE)은 통계학, 기계학습, 계량금융에서 가장 널리 쓰이는 추정 방법이다. 분포 모수를 적합하는 일에서 옵션 가격 모형을 보정하는 일까지 두루 쓰인다.

## 정의

### 가능도함수

$X_1, X_2, \ldots, X_n$을 확률밀도함수(또는 질량함수)가 $f(x; \theta)$인 분포에서 뽑은 확률표본이라 하고, $\theta \in \Theta$를 미지 모수(벡터일 수도 있다)라 하자. 자료 $x_1, x_2, \ldots, x_n$을 관측한 뒤 **가능도함수**는:

$$L(\theta) = L(\theta; x_1, \ldots, x_n) = \prod_{i=1}^n f(x_i; \theta)$$

**핵심 구별:** ($\theta$를 고정하고) $x$의 함수로 본 $f(x; \theta)$는 밀도이다. 같은 식을 (관측된 자료로 $x$를 고정하고) $\theta$의 함수로 본 것이 가능도이다.

**중요한 성질:**

- 가능도는 $\theta$에 대한 확률밀도가 **아니다**. $\Theta$에서 적분해도 1이 되지 않는다
- 가능도의 **비**만이 의미를 가지며 절대적인 척도는 임의적이다
- (충분성 원리에 의해) 가능도함수는 $\theta$에 관해 자료가 담고 있는 모든 정보를 요약한다

### 로그가능도함수

가능도가 여러 항의 곱이므로 거의 언제나 **로그가능도**로 작업하는 편이 편리하다:

$$\ell(\theta) = \log L(\theta) = \sum_{i=1}^n \log f(x_i; \theta)$$

**로그가능도의 장점:**

- 곱을 합으로 바꾼다 (계산이 안정적이고 해석적으로 더 단순하다)
- 최댓값의 위치가 보존된다 (로그는 단조증가함수이다)
- 정보이론적 양(KL 발산, 엔트로피)과 직접 연결된다
- 큰 표본에서 수치적 언더플로를 피한다

## 최대가능도추정

### 정의

**최대가능도추정량(MLE)**은 가능도함수를 최대화하는 $\theta$ 값이다:

$$\hat{\theta}_{\text{MLE}} = \arg\max_{\theta \in \Theta} L(\theta) = \arg\max_{\theta \in \Theta} \ell(\theta)$$

MLE는 "어떤 모수값이 관측된 자료를 가장 그럴듯하게 만드는가?"라는 물음에 답한다.

### MLE 찾기

로그가능도가 미분가능하면 MLE는 대개 **점수방정식**을 풀어 찾는다:

$$\frac{\partial \ell(\theta)}{\partial \theta} = 0$$

**점수함수** $s(\theta) = \frac{\partial \ell(\theta)}{\partial \theta}$는 로그가능도의 기울기이다. MLE는 $s(\hat{\theta}_{\text{MLE}}) = 0$을 만족한다.

벡터 모수 $\theta = (\theta_1, \ldots, \theta_k)^T$에 대해서는 다음 연립방정식을 푼다:

$$\frac{\partial \ell}{\partial \theta_j} = 0, \quad j = 1, \ldots, k$$

그 해가 (최솟값이나 안장점이 아니라) 최댓값인지 확인해야 하며, 보통 해에서 Hessian 행렬이 음의 정부호인지 살핀다.

## 예제

### 예제 1: Normal 분포 — 평균이 미지

$\sigma^2$이 알려진 $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$이라 하자. $\mu$의 MLE를 구한다.

**가능도:**

$$L(\mu) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)$$

**로그가능도:**

$$\ell(\mu) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2$$

**점수방정식:**

$$\frac{d\ell}{d\mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = 0$$

$$\sum_{i=1}^n x_i - n\mu = 0$$

$$\hat{\mu}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n x_i = \bar{x}$$

평균의 MLE는 표본평균이며 불편이고 효율적이다.

### 예제 2: Normal 분포 — 두 모수 모두 미지

$\mu$와 $\sigma^2$이 모두 미지인 $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$이라 하자.

**로그가능도:**

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2$$

**점수방정식:**

$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = 0 \implies \hat{\mu} = \bar{x}$$

$$\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^n (x_i - \mu)^2 = 0 \implies \hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2$$

**참고:** $\sigma^2$의 MLE는 $n-1$이 아니라 $n$으로 나눈다. 편향되어 있다: $E[\hat{\sigma}^2_{\text{MLE}}] = \frac{n-1}{n}\sigma^2$. MLE가 언제나 불편인 것은 아님을 보여 준다.

### 예제 3: Bernoulli 분포

$X_1, \ldots, X_n \sim \text{Bernoulli}(p)$라 하자.

**로그가능도:**

$$\ell(p) = \sum_{i=1}^n \left[x_i \log p + (1 - x_i)\log(1 - p)\right]$$

$$= k \log p + (n - k)\log(1 - p)$$

여기서 $k = \sum_{i=1}^n x_i$는 성공 횟수이다.

**점수방정식:**

$$\frac{d\ell}{dp} = \frac{k}{p} - \frac{n-k}{1-p} = 0$$

$$\hat{p}_{\text{MLE}} = \frac{k}{n} = \bar{x}$$

MLE는 표본비율이며 직관적으로 자연스럽고 불편이다.

### 예제 4: Exponential 분포

$x > 0$에서 밀도가 $f(x; \lambda) = \lambda e^{-\lambda x}$인 $X_1, \ldots, X_n \sim \text{Exp}(\lambda)$라 하자.

**로그가능도:**

$$\ell(\lambda) = n\log\lambda - \lambda \sum_{i=1}^n x_i$$

**점수방정식:**

$$\frac{d\ell}{d\lambda} = \frac{n}{\lambda} - \sum_{i=1}^n x_i = 0$$

$$\hat{\lambda}_{\text{MLE}} = \frac{n}{\sum_{i=1}^n x_i} = \frac{1}{\bar{x}}$$

**2계도함수 확인:** $\frac{d^2\ell}{d\lambda^2} = -n/\lambda^2 < 0$이므로 최댓값이다.

### 예제 5: Poisson 분포

$X_1, \ldots, X_n \sim \text{Poisson}(\lambda)$라 하자.

**로그가능도:**

$$\ell(\lambda) = \sum_{i=1}^n \left[x_i \log\lambda - \lambda - \log(x_i!)\right] = \left(\sum x_i\right)\log\lambda - n\lambda - \sum\log(x_i!)$$

**점수방정식:**

$$\frac{d\ell}{d\lambda} = \frac{\sum x_i}{\lambda} - n = 0$$

$$\hat{\lambda}_{\text{MLE}} = \bar{x}$$

## MLE의 성질

### 점근적 성질

정칙 조건 아래에서 MLE는 강력한 점근적 성질을 여럿 갖는다:

**1. 일치성:** $n \to \infty$일 때 $\hat{\theta}_{\text{MLE}} \xrightarrow{P} \theta_0$이며 $\theta_0$은 참 모수값이다.

**2. 점근정규성:**

$$\sqrt{n}(\hat{\theta}_{\text{MLE}} - \theta_0) \xrightarrow{d} N(0, I(\theta_0)^{-1})$$

여기서 $I(\theta_0)$은 Fisher 정보량이다. 동등하게:

$$\hat{\theta}_{\text{MLE}} \dot{\sim} N\left(\theta_0, \frac{1}{nI(\theta_0)}\right) \quad \text{for large } n$$

**3. 점근 효율성:** MLE는 점근적으로 Cramér-Rao 하한을 달성한다. 어떤 일치추정량도 더 작은 점근분산을 갖지 못한다.

**4. 불변성:** $\hat{\theta}$가 $\theta$의 MLE이면 임의의 함수 $g$에 대해 $g(\hat{\theta})$가 $g(\theta)$의 MLE이다. 매우 유용한 성질이다. 예를 들어 $\hat{\sigma}^2$이 $\sigma^2$의 MLE이면 $\sqrt{\hat{\sigma}^2}$이 $\sigma$의 MLE이다.

### Fisher 정보량

**Fisher 정보량**은 표본이 모수에 관해 얼마나 많은 정보를 담고 있는지를 정량화한다:

$$I(\theta) = -E\left[\frac{\partial^2 \ell(\theta)}{\partial \theta^2}\right] = E\left[\left(\frac{\partial \ell(\theta)}{\partial \theta}\right)^2\right]$$

$n$개의 i.i.d. 관측값에서 전체 Fisher 정보량은 $nI_1(\theta)$이며, $I_1(\theta)$는 관측값 하나에서 나오는 Fisher 정보량이다.

**핵심 역할:** Fisher 정보량이 MLE의 정밀도를 결정한다. 정보량이 클수록 가능도함수가 날카롭고 추정이 정밀하다.

### 관측 Fisher 정보량

실무에서는 기대 Fisher 정보량 대신 **관측 Fisher 정보량**을 자주 쓴다:

$$J(\hat{\theta}) = -\frac{\partial^2 \ell(\theta)}{\partial \theta^2}\bigg|_{\theta = \hat{\theta}}$$

기댓값을 계산할 필요가 없고 MLE에서 평가한다. 표준오차는 $\text{SE}(\hat{\theta}) = 1/\sqrt{J(\hat{\theta})}$로 추정한다.

## 가능도 기반 추론

### 가능도비

두 모수값 $\theta_0$과 $\theta_1$을 비교하는 **가능도비**는:

$$\Lambda = \frac{L(\theta_0)}{L(\theta_1)}$$

$\Lambda$가 작으면 자료가 $\theta_0$보다 $\theta_1$을 지지한다는 뜻이다.

### 가능도비 검정

$H_0: \theta = \theta_0$ 대 $H_1: \theta \neq \theta_0$을 검정할 때 **가능도비 검정통계량**은:

$$\Lambda = \frac{L(\theta_0)}{L(\hat{\theta}_{\text{MLE}})}$$

$H_0$과 정칙 조건 아래에서:

$$-2\log\Lambda = 2[\ell(\hat{\theta}_{\text{MLE}}) - \ell(\theta_0)] \xrightarrow{d} \chi^2_k$$

여기서 $k$는 제약된 모수의 개수이다. 응용통계학의 많은 가설검정이 이를 토대로 한다.

### 가능도로부터의 신뢰구간

**Wald 신뢰구간:** 점근정규성을 사용하면:

$$\hat{\theta} \pm z_{\alpha/2} \cdot \text{SE}(\hat{\theta})$$

여기서 $\text{SE}(\hat{\theta}) = 1/\sqrt{nI(\hat{\theta})}$ 또는 $1/\sqrt{J(\hat{\theta})}$이다.

**프로파일 가능도 구간:** 다음을 만족하는 $\theta$ 값의 집합이다:

$$2[\ell(\hat{\theta}) - \ell(\theta)] \leq \chi^2_{1, 1-\alpha}$$

소표본이거나 가능도가 치우쳐 있을 때 이 구간이 대체로 Wald 구간보다 정확하다.

## 계산 방법

### Newton-Raphson 방법

점수방정식을 해석적으로 풀 수 없으면 MLE를 수치적으로 찾는다. Newton-Raphson은 다음을 반복한다:

$$\theta^{(t+1)} = \theta^{(t)} - \left[\frac{\partial^2 \ell}{\partial \theta^2}\bigg|_{\theta^{(t)}}\right]^{-1} \frac{\partial \ell}{\partial \theta}\bigg|_{\theta^{(t)}}$$

이는 로그가능도에 이차근사를 반복적으로 맞추는 것과 동등하다.

### Fisher 점수법

관측 Hessian을 기대 Fisher 정보량으로 바꾼 변형이다:

$$\theta^{(t+1)} = \theta^{(t)} + I(\theta^{(t)})^{-1} \cdot s(\theta^{(t)})$$

관측 Hessian의 조건수가 나쁠 때 Fisher 점수법이 더 안정적이다.

### EM 알고리즘

잠재변수를 갖는 모형(예: 혼합모형, 은닉 마르코프 모형)에서는 **기댓값–최대화(EM) 알고리즘**이 다음을 번갈아 수행한다:

- **E 단계**: 현재 모수와 관측 자료가 주어졌을 때 기대 로그가능도를 계산한다
- **M 단계**: 이 기대 로그가능도를 최대화하여 모수를 갱신한다

EM 알고리즘은 매 단계마다 가능도가 단조 증가함을 보장하며, 국면전환 모형 같은 금융 응용에서 널리 쓰인다.

## 금융과의 연결

가능도함수와 MLE는 계량금융 전반에 스며 있다:

- **GARCH 모형**: 보통 정규 또는 Student-$t$ 혁신항을 가정하고 수익률의 조건부 로그가능도를 최대화하여 모수 $(\omega, \alpha, \beta)$를 추정한다.
- **옵션 가격결정**: Black-Scholes 내재변동성은 모형이 가정한 동학 아래에서 관측된 옵션 가격이 주어졌을 때 변동성의 MLE이다.
- **국면전환 모형**: Hamilton의 국면전환 모형은 전이확률과 국면별 모수에 대해 가능도를 최대화하는 데 EM 알고리즘을 사용한다.
- **위험 모형화**: Value-at-Risk와 기대손실 추정을 위해 손실 자료에 꼬리가 두꺼운 분포(Student-$t$, 일반화 Pareto)를 MLE로 적합한다.
- **기간구조 모형**: Vasicek, CIR, 아핀 기간구조 모형의 모수는 관측된 수익률곡선의 가능도를 최대화하여 보정한다.
- **코퓰러 모형**: 코퓰러 기반 포트폴리오 위험 모형의 의존성 모수는 유사최대가능도로 추정한다.

## MLE의 한계

MLE는 강력하지만 중요한 한계가 있다:

- **소표본 편향**: 소표본에서 MLE가 상당히 편향될 수 있다(예: 분산추정량이 $n-1$ 대신 $n$으로 나눈다)
- **경계 문제**: 참 모수가 모수공간의 경계에 있으면 표준 점근이론이 무너진다
- **모형 설정 오류**: MLE는 모형이 옳을 때 최적이다. 설정이 잘못되면 참 분포에서 모형족까지의 KL 발산을 최소화하는 모수값으로 수렴한다(준최대가능도 또는 유사최대가능도)
- **여러 봉우리를 갖는 가능도**: 특히 혼합모형에서 수치 최적화가 전역이 아닌 국소 최댓값을 찾을 수 있다
- **이상점에 대한 민감도**: 어떤 모형에서는 이상점 하나가 MLE를 크게 움직일 수 있다

## 요약

가능도함수는 관측된 자료를 모수값에 대한 지지의 척도로 바꾸고, MLE는 가장 많은 지지를 받는 값을 고른다. MLE는 일치하고, 점근적으로 효율적이며, 재모수화에 불변이다. 가능도비와 Fisher 정보량을 통해 가능도함수는 가설검정과 신뢰구간에 대한 통일된 틀도 제공한다. 이런 성질들 덕분에 MLE는 통계학과 계량금융 전반에서 기본 추정 방법이 된다.

## 주요 공식

| 양 | 공식 |
|----------|---------|
| 가능도 | $L(\theta) = \prod_{i=1}^n f(x_i; \theta)$ |
| 로그가능도 | $\ell(\theta) = \sum_{i=1}^n \log f(x_i; \theta)$ |
| 점수함수 | $s(\theta) = \partial \ell(\theta) / \partial \theta$ |
| Fisher 정보량 | $I(\theta) = -E[\partial^2 \ell / \partial \theta^2]$ |
| MLE의 점근분산 | $1 / [nI(\theta)]$ |
| 가능도비 검정 | $-2\log\Lambda \sim \chi^2_k$ |
| 불변성 | $\widehat{g(\theta)} = g(\hat{\theta}_{\text{MLE}})$ |

## 연습문제

**연습문제 1.**
PDF가 $f(x;\lambda) = \lambda e^{-\lambda x}$인 Exponential$(\lambda)$ 분포에서 얻은 확률표본 $x_1, \ldots, x_n$에 대해 로그가능도함수를 쓰고 MLE $\hat{\lambda}$를 구하라.

??? success "연습문제 1 풀이"
    가능도는:

    $$
    L(\lambda) = \prod_{i=1}^n \lambda e^{-\lambda x_i} = \lambda^n e^{-\lambda \sum x_i}
    $$

    로그가능도는:

    $$
    \ell(\lambda) = n \log \lambda - \lambda \sum_{i=1}^n x_i
    $$

    점수를 0으로 두면:

    $$
    \frac{d\ell}{d\lambda} = \frac{n}{\lambda} - \sum_{i=1}^n x_i = 0 \implies \hat{\lambda} = \frac{n}{\sum_{i=1}^n x_i} = \frac{1}{\bar{x}}
    $$

    2계도함수가 $-n/\lambda^2 < 0$이므로 최댓값이다.

---

**연습문제 2.**
MLE의 불변성을 사용하여, Exponential$(\lambda)$ 분포에서 $\hat{\lambda} = 1/\bar{x}$가 MLE일 때 평균 $\mu = 1/\lambda$의 MLE는 무엇인가?

??? success "연습문제 2 풀이"
    불변성에 의해 임의의 함수 $g(\theta)$의 MLE는 $g(\hat{\theta})$이다. $\mu = 1/\lambda = g(\lambda)$이므로:

    $$
    \hat{\mu} = g(\hat{\lambda}) = \frac{1}{\hat{\lambda}} = \frac{1}{1/\bar{x}} = \bar{x}
    $$

    모평균의 MLE는 단순히 표본평균이며 직관적이다.

---

**연습문제 3.**
Exponential$(\lambda)$ 분포의 관측값 하나에 대한 Fisher 정보량 $I(\lambda)$를 계산하라. MLE $\hat{\lambda}$의 점근분산은 무엇인가?

??? success "연습문제 3 풀이"
    관측값 하나에 대한 로그가능도는 $\ell(\lambda) = \log \lambda - \lambda x$이다. 2계도함수는:

    $$
    \frac{d^2 \ell}{d\lambda^2} = -\frac{1}{\lambda^2}
    $$

    Fisher 정보량은:

    $$
    I(\lambda) = -E\!\left[\frac{d^2 \ell}{d\lambda^2}\right] = \frac{1}{\lambda^2}
    $$

    $n$개의 관측값에 대해 $\hat{\lambda}$의 점근분산은:

    $$
    \text{Var}(\hat{\lambda}) \approx \frac{1}{n I(\lambda)} = \frac{\lambda^2}{n}
    $$

---

**연습문제 4.**
정규분포의 분산추정량을 예로 들어 MLE가 유한표본에서 편향될 수 있는 이유를 설명하라.

??? success "연습문제 4 풀이"
    $N(\mu, \sigma^2)$에서 뽑은 확률표본에 대해 $\sigma^2$의 MLE는:

    $$
    \hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2
    $$

    그런데 $E[\hat{\sigma}^2_{\text{MLE}}] = \frac{n-1}{n}\sigma^2 \neq \sigma^2$이다. MLE가 $n-1$ 대신 $n$으로 나누므로 $\sigma^2/n$만큼 아래쪽으로 편향된다. MLE가 성가신 모수의 추정을 보정하지 않고 가능도를 최대화하기 때문이다(여기서는 $\mu$를 $\bar{X}$로 추정하면서 자유도 하나를 "써 버린다").

    $n \to \infty$일 때 편향이 사라지므로 MLE는 **일치**하지만, 유한표본에서 분산 추정에는 불편추정량 $S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$이 선호된다.
