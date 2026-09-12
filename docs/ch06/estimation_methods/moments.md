# 적률법

## 소개

**적률법(MoM)**은 모수 추정에 대한 가장 오래되고 직관적인 접근 중 하나이다. 착상은 단순하다. (미지 모수의 함수인) 모집단 적률을 그 표본 대응물과 같다고 두고 모수에 대해 푸는 것이다. 이렇게 얻은 추정량은 계산하기 쉽고 흔히 닫힌 형태로 주어지며, MLE 같은 더 정교한 방법의 좋은 출발점이 된다.

MLE가 대체로 더 효율적이지만, 적률법은 실무에서 여전히 널리 쓰인다. 특히 가능도를 다루기 어려울 때, 빠른 예비 추정량으로, 그리고 실증금융과 계량경제학을 지배하는 일반화 적률법(GMM) 틀에서 그렇다.

## 모집단 적률과 표본 적률


<div class="defn" markdown>

### 정의 1. 모집단 적률 { .dfn }

분포가 $f(x; \theta)$인 확률변수 $X$의 $k$차 **모집단 적률**(원적률)은:

$$\mu_k' = E[X^k] = \int x^k f(x; \theta) \, dx$$

$k$차 **중심적률**은:

$$\mu_k = E[(X - \mu)^k]$$

여기서 $\mu = E[X] = \mu_1'$이다.

이 적률들은 미지 모수 $\theta$의 함수이다:

- $\mu_1'(\theta) = E_\theta[X]$ (평균)
- $\mu_2'(\theta) = E_\theta[X^2]$ (2차 원적률)
- $\mu_2(\theta) = \text{Var}_\theta(X)$ (분산)
- $\mu_3(\theta)$는 왜도와 관련된다
- $\mu_4(\theta)$는 첨도와 관련된다

</div>

<div class="defn" markdown>

### 정의 2. 표본 적률 { .dfn }

$k$차 **표본 적률**(원적률)은:

$$m_k' = \frac{1}{n}\sum_{i=1}^n X_i^k$$

$k$차 **표본 중심적률**은:

$$m_k = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^k$$

큰수의 법칙에 의해 $n \to \infty$일 때 $m_k' \xrightarrow{P} \mu_k'$이다.

</div>

## 적률법 절차

### 일반적인 방법

분포가 $p$개의 미지 모수 $\theta = (\theta_1, \ldots, \theta_p)^T$에 의존한다고 하자. 적률법은 다음과 같이 진행한다.

**1단계.** 처음 $p$개의 모집단 적률을 모수의 함수로 나타낸다:

$$\mu_k'(\theta) = E_\theta[X^k], \quad k = 1, 2, \ldots, p$$

**2단계.** 모집단 적률을 표본 적률과 같다고 둔다:

$$\mu_k'(\theta) = m_k', \quad k = 1, 2, \ldots, p$$

**3단계.** 미지수가 $p$개인 $p$개의 방정식을 풀어 $\hat{\theta}_1, \ldots, \hat{\theta}_p$를 얻는다.

그 결과 얻은 추정량 $\hat{\theta}_{\text{MoM}}$이 **적률법 추정량**이다.

### 중심적률을 쓰는 경우

원적률 대신(또는 원적률에 더하여) 중심적률을 맞추는 편이 편리할 때가 있다. 예를 들어 $\theta = (\mu, \sigma^2)$이면 다음과 같이 둘 수 있다:

- $E[X] = \bar{X}$ (1차 원적률)
- $\text{Var}(X) = m_2$ (2차 중심적률)

이는 동등하며 대수 계산을 간단하게 해 주는 경우가 많다.

## 보기

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> Normal 분포. $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$이라 하자. 미지 모수가 둘이므로 적률방정식도 둘이 필요하다.

</div>

??? success "풀이"
    **모집단 적률:**

    - $\mu_1' = E[X] = \mu$
    - $\mu_2' = E[X^2] = \sigma^2 + \mu^2$

    **적률방정식:**

    $$\mu = m_1' = \bar{X}$$

    $$\sigma^2 + \mu^2 = m_2' = \frac{1}{n}\sum X_i^2$$

    **해:**

    $$\hat{\mu}_{\text{MoM}} = \bar{X}$$

    $$\hat{\sigma}^2_{\text{MoM}} = m_2' - (m_1')^2 = \frac{1}{n}\sum X_i^2 - \bar{X}^2 = \frac{1}{n}\sum (X_i - \bar{X})^2$$

    $\hat{\sigma}^2_{\text{MoM}}$이 MLE와 마찬가지로 $n$으로 나누므로(편향) 이 경우 적률법과 MLE가 같은 추정량을 준다.
<div class="exbox" markdown>

**보기 2.** <span class="diff easy" title="쉬움"></span> Exponential 분포. $E[X] = 1/\lambda$인 $X_1, \ldots, X_n \sim \text{Exp}(\lambda)$라 하자. 모수가 하나이므로 적률방정식도 하나면 된다.

</div>

??? success "풀이"
    **적률방정식:**

    $$\frac{1}{\lambda} = \bar{X}$$

    **해:**

    $$\hat{\lambda}_{\text{MoM}} = \frac{1}{\bar{X}}$$

    이는 MLE와 일치한다.
<div class="exbox" markdown>

**보기 3.** <span class="diff easy" title="쉬움"></span> Gamma 분포. 밀도가 $f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}$인 $X_1, \ldots, X_n \sim \text{Gamma}(\alpha, \beta)$라 하자. 모수가 둘이므로 방정식도 둘이 필요하다.

</div>

??? success "풀이"
    **모집단 적률:**

    - $E[X] = \alpha/\beta$
    - $\text{Var}(X) = \alpha/\beta^2$

    **적률방정식:**

    $$\frac{\alpha}{\beta} = \bar{X}, \qquad \frac{\alpha}{\beta^2} = \frac{1}{n}\sum (X_i - \bar{X})^2$$

    **해:** 비 $\text{Var}(X)/E[X] = 1/\beta$로부터:

    $$\hat{\beta}_{\text{MoM}} = \frac{\bar{X}}{m_2}, \qquad \hat{\alpha}_{\text{MoM}} = \frac{\bar{X}^2}{m_2}$$

    여기서 $m_2 = \frac{1}{n}\sum(X_i - \bar{X})^2$이다.

    적률법 추정량은 닫힌 형태로 주어지는 반면 Gamma 분포의 MLE는 수치 최적화가 필요하다. 이것이 실무적으로 중요한 장점이다.
<div class="exbox" markdown>

**보기 4.** <span class="diff easy" title="쉬움"></span> Uniform 분포. $X_1, \ldots, X_n \sim \text{Uniform}(a, b)$라 하자. 모수가 둘이므로 방정식도 둘이 필요하다.

</div>

??? success "풀이"
    **모집단 적률:**

    - $E[X] = (a + b)/2$
    - $\text{Var}(X) = (b - a)^2/12$

    **적률방정식:**

    $$\frac{a + b}{2} = \bar{X}, \qquad \frac{(b-a)^2}{12} = m_2$$

    **해:**

    $$\hat{a}_{\text{MoM}} = \bar{X} - \sqrt{3 m_2}, \qquad \hat{b}_{\text{MoM}} = \bar{X} + \sqrt{3 m_2}$$

    **참고:** 이 추정값들이 자료의 범위 안쪽으로 들어올 수 있다(즉 $\hat{a} > \min(X_i)$이거나 $\hat{b} < \max(X_i)$일 수 있다). 이는 논리적으로 모순이다. 적률법이 분포의 지지집합 제약을 언제나 지키지는 않는다는 알려진 한계이다. MLE($\hat{a} = \min(X_i)$, $\hat{b} = \max(X_i)$)에는 이런 문제가 없다.
<div class="exbox" markdown>

**보기 5.** <span class="diff easy" title="쉬움"></span> Beta 분포. $X_1, \ldots, X_n \sim \text{Beta}(\alpha, \beta)$라 하자.

</div>

??? success "풀이"
    **모집단 적률:**

    - $E[X] = \frac{\alpha}{\alpha + \beta}$
    - $\text{Var}(X) = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$

    $\bar{x} = m_1'$과 $s^2 = m_2$를 표본평균과 표본분산이라 하고 풀면:

    $$\hat{\alpha}_{\text{MoM}} = \bar{x}\left(\frac{\bar{x}(1-\bar{x})}{s^2} - 1\right), \qquad \hat{\beta}_{\text{MoM}} = (1 - \bar{x})\left(\frac{\bar{x}(1-\bar{x})}{s^2} - 1\right)$$

    $s^2 < \bar{x}(1 - \bar{x})$가 필요하며, 웬만한 자료에서는 성립한다.
## 적률법 추정량의 성질

### 일치성

큰수의 법칙에 의해 표본 적률은 모집단 적률로 수렴한다. 적률을 모수로 보내는 함수가 연속이면 연속사상정리에 의해 적률법 추정량은 **일치**한다:

$$\hat{\theta}_{\text{MoM}} \xrightarrow{P} \theta_0 \quad \text{as } n \to \infty$$

### 점근정규성

정칙 조건 아래에서 적률법 추정량은 점근적으로 정규이다. 1차 적률로 추정한 단일 모수에 대해:

$$\sqrt{n}(\hat{\theta}_{\text{MoM}} - \theta_0) \xrightarrow{d} N(0, V)$$

점근분산 $V$는 적률을 모수로 보내는 함수에 델타 방법을 적용하여 결정된다.

### 효율성

적률법 추정량은 일반적으로 MLE보다 **덜 효율적**이다. 점근 상대효율은:

$$\text{ARE}(\hat{\theta}_{\text{MoM}}, \hat{\theta}_{\text{MLE}}) = \frac{\text{Var}_{\text{asymp}}(\hat{\theta}_{\text{MLE}})}{\text{Var}_{\text{asymp}}(\hat{\theta}_{\text{MoM}})} \leq 1$$

효율 손실은 (적률법 = MLE인 정규분포처럼) 무시할 만한 수준부터 (꼬리가 두꺼운 일부 분포처럼) 상당한 수준까지 다양하다.

### MLE와의 비교

| 성질 | 적률법 | MLE |
|----------|------------------|-----|
| 계산 | 흔히 닫힌 형태 | 수치 최적화가 필요할 수 있음 |
| 효율성 | (일반적으로) 덜 효율적 | 점근적으로 효율적 |
| 일치성 | 예 (정칙 조건 아래) | 예 (정칙 조건 아래) |
| 불변성 | 일반적으로 불변이 아님 | 재모수화에 불변 |
| 로버스트성 | 고차 적률의 이상점에 민감 | 모형 설정 오류에 민감 |
| 유일성 | 언제나 유일하지는 않음 | 대개 유일 (정칙 조건 아래) |

## 일반화 적률법 (GMM)

### 동기

많은 응용에서, 특히 계량경제학과 금융에서 **모수보다 적률 조건이 많다**. GMM은 이 여분의 조건을 최적으로 활용하도록 적률법을 확장한다.

### 설정

적률 조건이 $q$개인데 모수는 $p < q$개라고 하자. 참 모수 $\theta_0$에서 다음을 만족하는 **적률함수** $g(X_i, \theta)$를 정의한다:

$$E[g(X_i, \theta_0)] = 0$$

표본 대응물은:

$$\bar{g}_n(\theta) = \frac{1}{n}\sum_{i=1}^n g(X_i, \theta)$$

$q > p$이면 $q$개의 표본 적률을 모두 정확히 0으로 만들 수 없다. 대신 GMM은 이차형식을 최소화한다:

$$\hat{\theta}_{\text{GMM}} = \arg\min_\theta \bar{g}_n(\theta)^T W \bar{g}_n(\theta)$$

여기서 $W$는 양의 정부호 **가중행렬**이다.

### 최적 가중행렬

**효율적 GMM** 추정량은 최적 가중행렬을 사용한다:

$$W^* = \left[E[g(X_i, \theta_0) g(X_i, \theta_0)^T]\right]^{-1} = S^{-1}$$

여기서 $S$는 적률 조건의 장기 공분산행렬이다. 이것이 $W$의 모든 선택 중에서 가장 효율적인 GMM 추정량을 준다.

실무에서는 $S$를 모르므로 자료에서 추정하며, 보통 2단계 절차를 쓴다:

1. $W = I$(단위행렬)로 $\hat{\theta}^{(1)}$을 추정한다
2. 1단계의 잔차로 $\hat{S}$를 추정한다
3. $W = \hat{S}^{-1}$로 $\hat{\theta}^{(2)}$를 다시 추정한다

### Hansen의 J 검정

모수보다 적률 조건이 많으면($q > p$) **과대식별 제약**을 검정할 수 있다. **J 통계량**은:

$$J = n \cdot \bar{g}_n(\hat{\theta})^T \hat{S}^{-1} \bar{g}_n(\hat{\theta}) \xrightarrow{d} \chi^2_{q-p}$$

J 통계량이 크면 모형의 적률 조건이 자료와 양립하지 않음을 시사한다.

## 금융과의 연결

적률법과 GMM은 계량금융에서 폭넓게 쓰인다:

- **자산가격결정**: GMM은 자산가격결정 모형(CAPM, Fama-French)을 추정하고 검정하는 표준 방법이다. $m_t$가 확률적 할인요인일 때 Euler 방정식 조건 $E[m_t R_t - 1] = 0$이 적률 조건을 제공한다.
- **GARCH 추정**: 준최대가능도가 표준이지만, 적률법은 무조건분산과 제곱수익률의 자기상관을 맞추어 $(\omega, \alpha, \beta)$의 닫힌 형태 초기 추정값을 준다.
- **분포 적합**: 수익률 자료에 꼬리가 두꺼운 분포(Student-$t$, 안정분포)를 맞출 때, 가능도가 복잡하거나 평가가 느리면 적률법이 빠른 추정값을 준다.
- **수익률곡선 모형화**: 수익률 수준이나 변화의 적률을 사용한 아핀 기간구조 모형의 GMM 추정.
- **실현변동성**: 고빈도 수익률 적률에 기반한 적률법 추정량이 적분변동성과 그 성질을 추정하는 데 쓰인다.
- **포트폴리오 이론**: 표본 적률로 기대수익률과 공분산을 추정하는 것이 포트폴리오 최적화에 대한 가장 단순한 적률법 접근이다.

## 심화 주제

### L-적률법

**L-적률**은 순서통계량의 선형결합으로 통상적인 적률의 대안을 제공한다. 특징은:

- 통상적인 적률보다 이상점에 로버스트하다
- (꼬리가 두꺼운 분포에서 존재하지 않을 수 있는 통상적인 적률과 달리) 언제나 분포를 유일하게 결정한다
- 위험관리에서 극단값 분포를 적합할 때 특히 유용하다

### 고차 적률 맞추기

모수가 셋 이상인 분포에서는 고차 적률(왜도, 첨도)을 맞춘다:

- $\hat{\gamma} = m_3/m_2^{3/2}$가 모집단 왜도를 맞춘다
- $\hat{\kappa} = m_4/m_2^2$가 모집단 첨도를 맞춘다

다만 고차 표본 적률일수록 잡음이 커지며 이것이 실용적인 한계이다. $k$차 표본 적률의 추정 분산은 분포의 $2k$차 적률을 포함한다.

### 모의 적률법 (SMM)

이론적 적률 $\mu_k'(\theta)$를 해석적으로 계산할 수는 없지만 모형을 모의실험할 수 있을 때, **SMM**은 모집단 적률을 모의 적률로 대체한다:

1. 후보 $\theta$에 대해 모형에서 크기 $n$인 표본을 $S$개 모의실험한다
2. 평균 모의 적률 $\tilde{m}_k'(\theta)$를 계산한다
3. 표본 적률과 모의 적률 사이의 거리를 최소화하는 $\theta$를 고른다

SMM은 가능도를 다루기 어려운 복잡한 금융 모형(예: 행위자 기반 모형, 복잡한 파생상품 가격결정 모형)에 쓰인다.

## 요약

적률법은 표본 적률을 모집단 적률과 같다고 두어 추정량을 준다. 일반적으로 MLE보다 효율이 낮지만, 계산의 단순함, 닫힌 형태의 해, 그리고 과대식별 모형을 위한 강력한 GMM 확장을 제공한다. 금융에서 GMM은 자산가격결정 모형을 추정하고 검정하는 주력 도구가 되었고, 표준 적률법은 빠른 추정과 초기화에 여전히 유용하다.

## 주요 공식

| 양 | 공식 |
|----------|---------|
| $k$차 표본 적률 | $m_k' = \frac{1}{n}\sum_{i=1}^n X_i^k$ |
| 적률방정식 | $k = 1, \ldots, p$에 대해 $\mu_k'(\theta) = m_k'$ |
| GMM 목적함수 | $\min_\theta \bar{g}_n(\theta)^T W \bar{g}_n(\theta)$ |
| 최적 가중 | $W^* = S^{-1}$ |
| J 검정 | $J = n \bar{g}_n(\hat{\theta})^T \hat{S}^{-1} \bar{g}_n(\hat{\theta}) \sim \chi^2_{q-p}$ |

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
Uniform$(0, \theta)$ 분포에서 얻은 확률표본의 표본평균이 $\bar{x} = 3.5$이다. 적률법 추정량 $\hat{\theta}$를 구하라.

</div>

??? success "풀이"
    Uniform$(0, \theta)$의 1차 모집단 적률은:

    $$
    \mu_1' = E[X] = \frac{\theta}{2}
    $$

    1차 모집단 적률을 1차 표본 적률과 같다고 두면:

    $$
    \frac{\theta}{2} = \bar{x} = 3.5 \implies \hat{\theta} = 2\bar{x} = 7.0
    $$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
평균이 $\alpha\beta$이고 분산이 $\alpha\beta^2$인 Gamma$(\alpha, \beta)$ 분포에 대해 $\bar{x}$와 $s^2$으로 $\alpha$와 $\beta$의 적률법 추정량을 유도하라.

</div>

??? success "풀이"
    모집단 적률을 표본 적률과 같다고 두면:

    $$
    \alpha\beta = \bar{x} \quad \text{and} \quad \alpha\beta^2 = s^2
    $$

    두 번째 식을 첫 번째 식으로 나누면:

    $$
    \frac{\alpha\beta^2}{\alpha\beta} = \frac{s^2}{\bar{x}} \implies \hat{\beta} = \frac{s^2}{\bar{x}}
    $$

    다시 대입하면:

    $$
    \hat{\alpha} = \frac{\bar{x}}{\hat{\beta}} = \frac{\bar{x}^2}{s^2}
    $$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
정규분포 분산의 적률법 추정량은 $\hat{\sigma}^2_{\text{MoM}} = \frac{1}{n}\sum(X_i - \bar{X})^2$이고 불편추정량은 $n-1$로 나눈다. 적률법 추정량이 편향되어 있음을 보이고 그 편향을 계산하라.

</div>

??? success "풀이"
    적률법 추정량의 기댓값은:

    $$
    E[\hat{\sigma}^2_{\text{MoM}}] = E\!\left[\frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2\right] = \frac{1}{n} \cdot (n-1)\sigma^2 = \frac{n-1}{n}\sigma^2
    $$

    편향은:

    $$
    \text{Bias} = E[\hat{\sigma}^2_{\text{MoM}}] - \sigma^2 = \frac{n-1}{n}\sigma^2 - \sigma^2 = -\frac{\sigma^2}{n}
    $$

    적률법 추정량은 참 분산을 $\sigma^2/n$만큼 과소추정한다. 이 편향은 $n \to \infty$일 때 사라지므로 추정량은 점근적으로 불편이다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
적률법과 최대가능도추정을 비교하라. 어떤 상황에서 MLE보다 적률법을 선호하겠는가?

</div>

??? success "풀이"
    **적률법을 선호할 때:**

    - 가능도함수를 해석적으로 쓰기 어렵거나 불가능할 때(예: 복잡한 생성 모형).
    - 반복적인 MLE 최적화의 출발점으로 빠른 닫힌 형태의 추정값이 필요할 때.
    - 모수가 많아 MLE의 계산 비용이 크지만 합리적인 근사로 충분할 때.
    - 모형 설정 오류에 대한 로버스트성이 필요할 때(적절한 적률 조건을 쓴 GMM은 설정 오류 아래에서 MLE보다 로버스트할 수 있다).

    **MLE를 선호할 때:**

    - 효율성이 중요할 때. MLE는 점근적으로 Cramér-Rao 하한을 달성하지만 적률법은 일반적으로 덜 효율적이다.
    - 가능도를 다룰 수 있고 모형이 올바르게 설정되었다고 볼 때.
    - 소표본 성능이 중요할 때. MLE가 대체로 유한표본 성질이 더 좋다.
    - 재모수화에 대한 불변성이 필요할 때. $g(\theta)$의 MLE는 $g(\hat{\theta}_{\text{MLE}})$인데, 적률법에는 이 성질이 없다.

---

## 정리하며

적률법은 **모집단 적률을 표본 적률과 맞추어** 모수를 푼다.

$$
\mu_r'(\theta_1,\ldots,\theta_k) = \frac1n\sum_i X_i^r, \qquad r=1,\ldots,k
$$

- **가장 오래된 체계적 추정법**이며, 피어슨이 1894년에 도입했다. 최대가능도보다 30년 가까이 앞선다.
- **큰수의 법칙이 일치성을 준다.** 연속사상정리를 함께 쓰면 적률의 연속함수인 추정량도 참값으로 수렴한다.
- **닫힌 형태가 나오는 경우가 많아** 계산이 빠르고, 반복 최적화의 출발값으로 유용하다.
- **분포를 전부 지정하지 않아도 된다.** 필요한 적률이 존재하고 모수로 표현되기만 하면 된다.
- **효율성이 대체로 낮다.** 적률 몇 개만 쓰므로 가능도가 담은 정보를 다 쓰지 못하며, 추정값이 모수공간을 벗어나는 일도 있다.
- **GMM 으로 확장된다.** 계량경제학과 실증금융에서는 오히려 이쪽이 주류다.

다음 절 **추정 방법 비교**에서 두 전략을 모의실험으로 직접 견준다.
