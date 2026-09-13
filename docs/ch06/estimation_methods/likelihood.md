# 가능도함수

## 소개

**가능도함수**는 모수적 통계 추론의 초석이다. 관측된 자료가 주어졌을 때 가능도함수는 각 후보 모수값이 그 자료를 만들어 냈을 법한 정도를 잰다. 모수를 고정하고 결과에 확률을 부여하는 확률분포와 달리, 가능도함수는 자료를 고정하고 모수를 변수로 다룬다.

가능도함수 위에 세워진 최대가능도추정(MLE)은 통계학, 기계학습, 계량금융에서 가장 널리 쓰이는 추정 방법이다. 분포 모수를 적합하는 일에서 옵션 가격 모형을 보정하는 일까지 두루 쓰인다.

## 가능도함수와 로그가능도


<div class="defn" markdown>

### 정의 1. 가능도함수 { .dfn }

$X_1, X_2, \ldots, X_n$을 확률밀도함수(또는 질량함수)가 $f(x; \theta)$인 분포에서 뽑은 확률표본이라 하고, $\theta \in \Theta$를 미지 모수(벡터일 수도 있다)라 하자. 자료 $x_1, x_2, \ldots, x_n$을 관측한 뒤 **가능도함수**는:

$$L(\theta) = L(\theta; x_1, \ldots, x_n) = \prod_{i=1}^n f(x_i; \theta)$$

**핵심 구별:** ($\theta$를 고정하고) $x$의 함수로 본 $f(x; \theta)$는 밀도이다. 같은 식을 (관측된 자료로 $x$를 고정하고) $\theta$의 함수로 본 것이 가능도이다.

**중요한 성질:**

- 가능도는 $\theta$에 대한 확률밀도가 **아니다**. $\Theta$에서 적분해도 1이 되지 않는다
- 가능도의 **비**만이 의미를 가지며 절대적인 척도는 임의적이다
- (충분성 원리에 의해) 가능도함수는 $\theta$에 관해 자료가 담고 있는 모든 정보를 요약한다

</div>

<div class="defn" markdown>

### 정의 2. 로그가능도함수 { .dfn }

가능도가 여러 항의 곱이므로 거의 언제나 **로그가능도**로 작업하는 편이 편리하다:

$$\ell(\theta) = \log L(\theta) = \sum_{i=1}^n \log f(x_i; \theta)$$

**로그가능도의 장점:**

- 곱을 합으로 바꾼다 (계산이 안정적이고 해석적으로 더 단순하다)
- 최댓값의 위치가 보존된다 (로그는 단조증가함수이다)
- 정보이론적 양(KL 발산, 엔트로피)과 직접 연결된다
- 큰 표본에서 수치적 언더플로를 피한다

</div>

## 최대가능도추정

<div class="defn" markdown>

### 정의 3. 최대가능도추정량 { .dfn }

**최대가능도추정량(MLE)**은 가능도함수를 최대화하는 $\theta$ 값이다:

$$\hat{\theta}_{\text{MLE}} = \arg\max_{\theta \in \Theta} L(\theta) = \arg\max_{\theta \in \Theta} \ell(\theta)$$

MLE는 "어떤 모수값이 관측된 자료를 가장 그럴듯하게 만드는가?"라는 물음에 답한다.

</div>

### MLE 찾기

로그가능도가 미분가능하면 MLE는 대개 **점수방정식**을 풀어 찾는다:

$$\frac{\partial \ell(\theta)}{\partial \theta} = 0$$

**점수함수** $s(\theta) = \frac{\partial \ell(\theta)}{\partial \theta}$는 로그가능도의 기울기이다. MLE는 $s(\hat{\theta}_{\text{MLE}}) = 0$을 만족한다.

벡터 모수 $\theta = (\theta_1, \ldots, \theta_k)^T$에 대해서는 다음 연립방정식을 푼다:

$$\frac{\partial \ell}{\partial \theta_j} = 0, \quad j = 1, \ldots, k$$

그 해가 (최솟값이나 안장점이 아니라) 최댓값인지 확인해야 하며, 보통 해에서 Hessian 행렬이 음의 정부호인지 살핀다.

## 보기

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> 정규분포 — 평균이 미지. $\sigma^2$이 알려진 $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$이라 하자. $\mu$의 MLE를 구한다.

</div>

??? success "풀이"
    **가능도:**

    $$L(\mu) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)$$

    **로그가능도:**

    $$\ell(\mu) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2$$

    **점수방정식:**

    $$\frac{d\ell}{d\mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = 0$$

    $$\sum_{i=1}^n x_i - n\mu = 0$$

    $$\hat{\mu}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n x_i = \bar{x}$$

    평균의 MLE는 표본평균이며 불편이고 효율적이다.
<div class="exbox" markdown>

**보기 2.** <span class="diff easy" title="쉬움"></span> 정규분포 — 두 모수 모두 미지. $\mu$와 $\sigma^2$이 모두 미지인 $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$이라 하자.

</div>

??? success "풀이"
    **로그가능도:**

    $$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2$$

    **점수방정식:**

    $$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = 0 \implies \hat{\mu} = \bar{x}$$

    $$\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^n (x_i - \mu)^2 = 0 \implies \hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2$$

    **참고:** $\sigma^2$의 MLE는 $n-1$이 아니라 $n$으로 나눈다. 편향되어 있다: $E[\hat{\sigma}^2_{\text{MLE}}] = \frac{n-1}{n}\sigma^2$. MLE가 언제나 불편인 것은 아님을 보여 준다.
<div class="exbox" markdown>

**보기 3.** <span class="diff easy" title="쉬움"></span> 베르누이분포. $X_1, \ldots, X_n \sim \text{Bernoulli}(p)$라 하자.

</div>

??? success "풀이"
    **로그가능도:**

    $$\ell(p) = \sum_{i=1}^n \left[x_i \log p + (1 - x_i)\log(1 - p)\right]$$

    $$= k \log p + (n - k)\log(1 - p)$$

    여기서 $k = \sum_{i=1}^n x_i$는 성공 횟수이다.

    **점수방정식:**

    $$\frac{d\ell}{dp} = \frac{k}{p} - \frac{n-k}{1-p} = 0$$

    $$\hat{p}_{\text{MLE}} = \frac{k}{n} = \bar{x}$$

    MLE는 표본비율이며 직관적으로 자연스럽고 불편이다.
<div class="exbox" markdown>

**보기 4.** <span class="diff easy" title="쉬움"></span> 지수분포. $x > 0$에서 밀도가 $f(x; \lambda) = \lambda e^{-\lambda x}$인 $X_1, \ldots, X_n \sim \text{Exp}(\lambda)$라 하자.

</div>

??? success "풀이"
    **로그가능도:**

    $$\ell(\lambda) = n\log\lambda - \lambda \sum_{i=1}^n x_i$$

    **점수방정식:**

    $$\frac{d\ell}{d\lambda} = \frac{n}{\lambda} - \sum_{i=1}^n x_i = 0$$

    $$\hat{\lambda}_{\text{MLE}} = \frac{n}{\sum_{i=1}^n x_i} = \frac{1}{\bar{x}}$$

    **2계도함수 확인:** $\frac{d^2\ell}{d\lambda^2} = -n/\lambda^2 < 0$이므로 최댓값이다.
<div class="exbox" markdown>

**보기 5.** <span class="diff easy" title="쉬움"></span> 포아송분포. $X_1, \ldots, X_n \sim \text{Poisson}(\lambda)$라 하자.

</div>

??? success "풀이"
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

- **GARCH 모형**: 보통 정규 또는 스튜던트-$t$ 혁신항을 가정하고 수익률의 조건부 로그가능도를 최대화하여 모수 $(\omega, \alpha, \beta)$를 추정한다.
- **옵션 가격결정**: Black-Scholes 내재변동성은 모형이 가정한 동학 아래에서 관측된 옵션 가격이 주어졌을 때 변동성의 MLE이다.
- **국면전환 모형**: Hamilton의 국면전환 모형은 전이확률과 국면별 모수에 대해 가능도를 최대화하는 데 EM 알고리즘을 사용한다.
- **위험 모형화**: Value-at-Risk와 기대손실 추정을 위해 손실 자료에 꼬리가 두꺼운 분포(스튜던트-$t$, 일반화 Pareto)를 MLE로 적합한다.
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

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
PDF가 $f(x;\lambda) = \lambda e^{-\lambda x}$인 Exponential$(\lambda)$ 분포에서 얻은 확률표본 $x_1, \ldots, x_n$에 대해 로그가능도함수를 쓰고 MLE $\hat{\lambda}$를 구하라.

</div>

??? success "풀이"
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

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
MLE의 불변성을 사용하여, Exponential$(\lambda)$ 분포에서 $\hat{\lambda} = 1/\bar{x}$가 MLE일 때 평균 $\mu = 1/\lambda$의 MLE는 무엇인가?

</div>

??? success "풀이"
    불변성에 의해 임의의 함수 $g(\theta)$의 MLE는 $g(\hat{\theta})$이다. $\mu = 1/\lambda = g(\lambda)$이므로:

    $$
    \hat{\mu} = g(\hat{\lambda}) = \frac{1}{\hat{\lambda}} = \frac{1}{1/\bar{x}} = \bar{x}
    $$

    모평균의 MLE는 단순히 표본평균이며 직관적이다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
Exponential$(\lambda)$ 분포의 관측값 하나에 대한 Fisher 정보량 $I(\lambda)$를 계산하라. MLE $\hat{\lambda}$의 점근분산은 무엇인가?

</div>

??? success "풀이"
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

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
정규분포의 분산추정량을 예로 들어 MLE가 유한표본에서 편향될 수 있는 이유를 설명하라.

</div>

??? success "풀이"
    $N(\mu, \sigma^2)$에서 뽑은 확률표본에 대해 $\sigma^2$의 MLE는:

    $$
    \hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2
    $$

    그런데 $E[\hat{\sigma}^2_{\text{MLE}}] = \frac{n-1}{n}\sigma^2 \neq \sigma^2$이다. MLE가 $n-1$ 대신 $n$으로 나누므로 $\sigma^2/n$만큼 아래쪽으로 편향된다. MLE가 성가신 모수의 추정을 보정하지 않고 가능도를 최대화하기 때문이다(여기서는 $\mu$를 $\bar{X}$로 추정하면서 자유도 하나를 "써 버린다").

    $n \to \infty$일 때 편향이 사라지므로 MLE는 **일치**하지만, 유한표본에서 분산 추정에는 불편추정량 $S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$이 선호된다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
$X_1,\dots,X_n \sim \text{Exp}(\lambda)$에서 우도비 검정으로 $H_0:\lambda=\lambda_0$을 검정하는 통계량을 유도하고, 그 기각역이 $\bar X$에 대해 어떤 모양인지 설명하라.

</div>

??? success "풀이"
    로그가능도가 $\ell(\lambda) = n\ln\lambda - \lambda\sum x_i = n\{\ln\lambda-\lambda\bar x\}$이므로

    $$
    \Lambda = 2\left\{\ell(\hat\lambda)-\ell(\lambda_0)\right\} = 2n\left\{\ln\frac{\hat\lambda}{\lambda_0}+\lambda_0\bar x - 1\right\}
    $$

    이다($\hat\lambda\bar x = 1$을 썼다). $u = \lambda_0\bar x$로 두면

    $$
    \Lambda = 2n\left(u - 1 - \ln u\right)
    $$

    **모양.** 함수 $h(u) = u-1-\ln u$는 $u=1$에서 최솟값 0을 갖고 양쪽으로 증가한다. 따라서 $\Lambda > c$인 기각역은

    $$
    \bar x < a \quad\text{또는}\quad \bar x > b
    $$

    꼴의 **양쪽 꼬리**이며, $a$와 $b$는 $h(a\lambda_0)=h(b\lambda_0)$을 만족한다.

    **핵심은 비대칭이다.** $h$가 $u=1$을 중심으로 대칭이 아니므로($u\to0$에서 발산, $u\to\infty$에서 선형) 두 끝점이 $1/\lambda_0$에서 같은 거리에 있지 않다. 아래쪽이 위쪽보다 가깝다.

    이것이 왈드 검정과의 차이다. 왈드는 $|\bar x - 1/\lambda_0|$을 대칭으로 재므로 다른 기각역을 준다. **우도비가 가능도의 실제 모양을 반영한다는 점에서 우월하다.**

    **정확한 검정.** $2n\lambda_0\bar X \sim \chi^2_{2n}$이라는 정확한 결과가 있으므로, $\chi^2$ 근사를 쓰지 않고 $\chi^2_{2n}$의 양쪽 꼬리로 정확한 임계값을 정할 수 있다. 소표본에서는 이쪽이 낫다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
가능도함수를 **그림으로** 보고할 때 무엇을 어떻게 그려야 하는지 정리하라. 점추정값과 표준오차만 보고하는 것에 견주어 어떤 이점이 있는가?

</div>

??? success "풀이"
    **무엇을 그리는가.**

    - 세로축은 **상대 로그가능도** $\ell(\theta)-\ell(\hat\theta)$. 최댓값을 0으로 맞춰야 표본크기가 다른 자료끼리 겹쳐 볼 수 있다.
    - 가로축은 모수. 범위는 $\hat\theta \pm 4\widehat{\operatorname{SE}}$ 정도.
    - 수평 기준선 $-1.92$($=\chi^2_{1,0.95}/2$)를 그어 95% 우도비 구간을 눈으로 읽게 한다. $-0.5$(1/1.65 가능도)와 $-3.32$(99%)를 함께 그리기도 한다.
    - 모수가 여럿이면 **프로파일** 로그가능도를 그린다.

    **이점.**

    - **비대칭이 보인다.** 점추정값과 표준오차는 대칭 정규분포를 암묵적으로 가정한다. 곡선은 실제 모양을 그대로 보여 준다.
    - **왈드 근사가 얼마나 나쁜지 알 수 있다.** 포물선을 겹쳐 그려 벌어짐을 확인한다.
    - **경계 문제가 드러난다.** $\hat\theta$가 경계에 있으면 곡선이 절벽처럼 끊긴다. 숫자만으로는 알 수 없다.
    - **다봉성이 드러난다.** 봉우리가 여럿이면 곡선에 그대로 나타난다. 점추정값 하나는 이를 완전히 감춘다.
    - **평평함이 보인다.** 넓은 범위에서 거의 평평하면 "자료가 이 모수에 대해 별로 말해 주지 않는다"는 뜻이며, 좁은 신뢰구간보다 훨씬 정직한 표현이다.

    **실무 권고.** 모수가 하나이거나 핵심 모수가 한둘이면 프로파일 가능도 곡선을 **부록에라도 넣는 것**이 좋다. 비용은 거의 없고 독자가 얻는 정보는 많다. 특히 소표본, 경계 근처, 비선형 모형에서 가치가 크다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
가능도를 이용한 **모형 비교**에서 내포된 모형과 내포되지 않은 모형을 어떻게 다르게 다루는지 설명하라.

</div>

??? success "풀이"
    **내포된 경우.** 모형 $M_0$이 $M_1$의 모수공간을 제약해 얻어지면(예: 일부 계수를 0으로 고정) **우도비 검정**을 쓸 수 있다.

    $$
    \Lambda = 2\left\{\ell_1(\hat{\boldsymbol\theta}_1)-\ell_0(\hat{\boldsymbol\theta}_0)\right\} \sim \chi^2_{p_1-p_0}
    $$

    자유도는 제약의 개수다. 이것이 회귀의 $F$ 검정, 분산분석, 일반화선형모형의 이탈도 검정의 바탕이다.

    **주의.** 제약이 모수공간의 **경계**에 있으면($\sigma^2 = 0$ 같은 경우) $\chi^2$ 근사가 틀린다. 그때 극한분포는 $\chi^2$들의 혼합이며, 예컨대 분산성분 하나를 0으로 두는 검정은 $\frac12\chi^2_0+\frac12\chi^2_1$을 따른다. $\chi^2_1$을 쓰면 보수적이 된다.

    **내포되지 않은 경우.** 우도비 검정을 쓸 수 없다. $\Lambda$가 $\chi^2$를 따를 이유가 없고 음수가 될 수도 있다.

    **대안.**

    - **정보기준.** AIC나 BIC로 비교한다. 내포 관계를 요구하지 않는 것이 가장 큰 장점이다. 다만 "유의성"이라는 개념이 없고 순위만 준다.
    - **복스-콕스류 검정.** 두 모형을 모두 포함하는 더 큰 모형을 인위적으로 만들어 각각을 내포 검정으로 돌린다. 결과가 "둘 다 기각", "둘 다 채택"으로 나올 수 있어 해석이 애매하다.
    - **부옹(Vuong) 검정.** 두 모형의 관측별 로그가능도 차이 $\ell_{1i}-\ell_{0i}$의 평균이 0인지 검정한다. 정규근사가 성립하며, 두 모형이 모두 틀려도 "어느 쪽이 참에 더 가까운가"를 물을 수 있다는 점이 개념적으로 깔끔하다.
    - **교차검증.** 예측 성능으로 직접 비교한다. 가정이 가장 적고, 예측이 목적이면 가장 적절하다.

    **실무 권고.** 내포되면 우도비 검정, 아니면 정보기준이나 교차검증. **AIC 차이를 $p$-값처럼 해석하지 말 것.** AIC는 검정이 아니라 순위 기준이다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
로그가능도를 계산할 때 흔히 저지르는 실수 세 가지를 들고 각각의 증상과 대처를 적어라.

</div>

??? success "풀이"
    **(1) 상수항을 빼먹거나 잘못 넣는다.** $-\frac n2\ln(2\pi)$나 $-\sum\ln x_i!$처럼 모수에 무관한 항이다.

    - **증상**: 최적화 결과는 정확하지만 AIC·BIC나 우도비가 틀린다. 특히 **모형마다 다른 상수를 쓰면** 비교가 무의미해진다.
    - **대처**: 모형을 비교할 계획이면 상수를 모두 포함한다. 최적화만 할 것이면 빼도 되지만, 나중에 비교하고 싶어질 때를 대비해 포함해 두는 편이 안전하다.

    **(2) 척도 변환의 야코비안을 빠뜨린다.** $y$ 대신 $\ln y$를 모형화하면 밀도가

    $$
    f_Y(y) = f_{\ln Y}(\ln y)\cdot\frac1y
    $$

    이므로 로그가능도에 $-\sum\ln y_i$가 추가된다.

    - **증상**: 변환한 모형과 변환하지 않은 모형의 AIC를 비교하면 항상 변환한 쪽이 이긴다(또는 진다). 반응변수를 바꾸면 가능도의 척도가 달라지기 때문이다.
    - **대처**: 반응변수 변환이 다른 모형끼리는 야코비안을 보정하지 않으면 비교하지 않는다.

    **(3) 로그 척도를 벗어난다.** 확률을 곱하거나 $\ln\sum\exp$를 직접 계산한다.

    - **증상**: 언더플로로 $-\infty$가 나오거나 오버플로로 `nan`이 나온다. $n$이 커질수록 심하다.
    - **대처**: `logsumexp`, `log1p`, `expm1`, `logaddexp`를 쓰고, 분포의 `logpdf`/`logpmf` 메서드를 직접 쓴다.

    **그 밖에.**

    - **부호 혼동.** 최소화 함수에 넘길 때 음의 로그가능도를 쓰는데, 반환값이나 헤시안의 부호를 한 번 더 뒤집는 실수.
    - **결측값.** `nan`이 하나만 섞여도 합 전체가 `nan`이 된다. 합산 전에 확인한다.
    - **지지집합 밖.** 최적화기가 $\lambda \le 0$ 같은 값을 시도하면 $\ln$이 `nan`을 낸다. 재모수화나 경계 지정으로 막는다.

    **공통 대처법.** **작은 자료로 손계산과 맞춰 본다.** $n=2$나 $n=3$짜리 자료에서 로그가능도를 손으로 계산해 코드 결과와 일치하는지 확인하면 대부분의 실수를 잡아낸다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
가능도는 **관측 방식**에 의존한다. 같은 현상이라도 자료를 어떻게 모았느냐에 따라 가능도가 달라지는 예를 두 가지 들어라.

</div>

??? success "풀이"
    **예 1 — 정지규칙.** 앞서 본 대로, 동전을 20번 던져 앞면 6번을 얻는 것과 앞면 6번이 나올 때까지 던져 20번이 걸리는 것은

    $$
    L_{\text{이항}} \propto p^6(1-p)^{14}, \qquad L_{\text{음이항}} \propto p^6(1-p)^{14}
    $$

    로 **비례상수만 다르다.** 가능도 자체는 사실상 같지만 표집분포는 다르므로 $p$-값이 달라진다.

    **예 2 — 절단과 중도절단.** 같은 수명 자료라도

    - 전부 고장 날 때까지 기다렸으면 $L = \prod f(t_i)$.
    - 시각 $C$에 중단했으면 $L = \prod f(t_i)^{\delta_i}S(C)^{1-\delta_i}$.
    - 고장 난 것만 기록했으면(절단) $L = \prod \dfrac{f(t_i)}{F(C)}$.

    **세 가능도가 근본적으로 다르다.** 특히 마지막의 분모를 빼먹으면 체계적 편향이 생긴다.

    **예 3 — 길이편향 표집.** 버스 정류장에 무작위로 도착해 다음 버스까지의 간격을 재면, **긴 간격이 더 자주 관측된다.** 관측될 확률이 길이에 비례하므로

    $$
    f_{\text{관측}}(x) = \frac{x\,f(x)}{E[X]}
    $$

    이다. 이를 무시하면 평균 간격을 과대추정한다. 유병 사례만 모은 역학 연구, 케이블 길이 표본조사에서 같은 문제가 생긴다.

    **공통 교훈.** **가능도는 "자료가 어떻게 생성되고 관측되었는가"의 완전한 진술이어야 한다.** 모형(분포)뿐 아니라 설계(언제 멈췄는가, 무엇이 관측되지 않았는가, 어떤 편향으로 표본에 들어왔는가)까지 담아야 한다. 이 부분을 빠뜨리는 것이 실무에서 가장 흔하고 가장 치명적인 오류다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
**벌점 가능도**($\ell(\theta)-\text{pen}(\theta)$)를 쓰는 세 가지 동기를 들고, 각각의 예를 적어라.

</div>

??? success "풀이"
    **동기 1 — 존재하지 않거나 퇴화하는 해를 막는다.**

    - **예**: 혼합모형의 분산 퇴화. $\sigma_j^2$에 역감마 사전분포에 해당하는 벌점을 주면 $\sigma_j\to0$이 막힌다.
    - **예**: 로지스틱 회귀의 완전분리. 펄스(Firth) 방법은 $\frac12\ln|I(\boldsymbol\beta)|$를 더해 유한한 해를 보장한다. 덤으로 소표본 편향까지 줄인다.

    **동기 2 — 과적합을 막고 예측을 개선한다.**

    - **예**: 능형회귀 $\text{pen} = \lambda\|\boldsymbol\beta\|_2^2$. 편향을 들여 분산을 줄인다.
    - **예**: 라소 $\text{pen} = \lambda\|\boldsymbol\beta\|_1$. 변수 선택까지 동시에 한다.
    - **예**: 평활 스플라인의 거칠기 벌점 $\lambda\int (f'')^2$. 함수의 매끄러움을 조절한다.

    **동기 3 — 모형 복잡도를 벌점해 모형을 고른다.**

    - **예**: AIC의 $-p$, BIC의 $-\frac p2\ln n$. 최대 로그가능도에서 모수 개수만큼 깎는 것이 벌점 가능도의 한 형태다.

    **공통 구조.** 벌점 가능도는 **MAP 추정과 같다.**

    $$
    \arg\max\left\{\ell(\theta)+\ln\pi(\theta)\right\}
    $$

    에서 $\ln\pi$가 벌점의 음수다. 능형은 정규 사전분포, 라소는 라플라스 사전분포, 펄스는 제프리스 사전분포에 대응한다. **벌점을 고르는 것이 곧 사전분포를 고르는 것**이라는 관점이 유용하다.

    **대가.** 추정량이 더 이상 불변이 아니고(벌점이 모수화에 의존한다), 표준적인 점근이론이 그대로 적용되지 않으며, 벌점의 세기 $\lambda$를 정해야 한다. $\lambda$는 보통 교차검증이나 정보기준으로 고르는데, 그 선택의 불확실성이 최종 추론에 반영되지 않는다는 점이 알려진 한계다.

---

## 정리하며

이 절은 가능도와 최대가능도추정을 **한자리에 모아** 정리했다.

- **가능도는 자료를 고정하고 모수를 변수로 본다.** $L(\theta)=\prod_i f(x_i;\theta)$ 이며, 확률분포와 식은 같고 읽는 방향이 반대다.
- **최대가능도는 $L$ 을 최대화하는 $\theta$ 를 고른다.** 실제로는 로그를 취해 합으로 바꾼 뒤 미분하거나 수치 최적화한다.
- **주요 성질을 한 줄로.** 불변성(변환해도 그대로), 점근 일치성·정규성·효율성, 그리고 충분통계량을 통해서만 자료에 의존한다는 점.
- **불편성은 보장하지 않는다.** 정규분포의 $\hat\sigma^2$ 이 반례이며, 불변성과 불편성은 양립하지 않는 요구다.
- **쓰이는 곳이 넓다.** 분포 적합, 회귀모형 추정, 옵션 가격 모형 보정까지 모수적 추론의 기본 도구다.

다음 절 **적률법 개관**에서 두 번째 전략을 같은 방식으로 정리한다.
