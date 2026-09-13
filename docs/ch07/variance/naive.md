# 소박한 분산추정량

## 들어가며

**소박한 분산추정량**은 편차제곱합을 $n-1$이 아니라 $n$(표본크기)으로 나눈다. 표본평균으로부터의 편차제곱을 그냥 평균하는 것이니 가장 직관적인 접근이지만, 알고 보면 편향되어 있다. *왜* 편향되는지를 이해하면 추정의 본질에 대한 깊은 통찰을 얻고 Bessel 수정의 동기를 알게 된다.

<div class="defn" markdown>

### 정의 1. 소박한 분산추정량 { .dfn }

표본평균이 $\bar{X}$인 확률표본 $X_1, X_2, \ldots, X_n$이 주어졌을 때 **소박한 분산추정량**은:

$$\tilde{S}^2 = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2$$

이는 **모분산 공식을 표본에 적용한 것**, **편향 표본분산**, 또는 (정규모집단에서) **분산의 MLE**라고도 불린다.

</div>

## 편향의 유도

### 핵심 항등식

편향 계산의 바탕이 되는 근본적인 항등식은 다음과 같다:

$$\sum_{i=1}^n (X_i - \bar{X})^2 = \sum_{i=1}^n (X_i - \mu)^2 - n(\bar{X} - \mu)^2$$

??? proof "증명"

    $(X_i - \bar{X})^2 = (X_i - \mu - (\bar{X} - \mu))^2$을 전개하면:

    $$\sum_{i=1}^n (X_i - \bar{X})^2 = \sum_{i=1}^n(X_i - \mu)^2 - 2(\bar{X} - \mu)\sum_{i=1}^n(X_i - \mu) + n(\bar{X} - \mu)^2$$

    $\sum(X_i - \mu) = n(\bar{X} - \mu)$이므로 가운데 항은 $-2n(\bar{X} - \mu)^2$이 되어:

    $$= \sum_{i=1}^n(X_i - \mu)^2 - n(\bar{X} - \mu)^2$$

### 기댓값의 계산

기댓값을 취하면:

$$E\left[\sum_{i=1}^n (X_i - \bar{X})^2\right] = \sum_{i=1}^n E[(X_i - \mu)^2] - nE[(\bar{X} - \mu)^2]$$

$$= n\sigma^2 - n \cdot \frac{\sigma^2}{n} = n\sigma^2 - \sigma^2 = (n-1)\sigma^2$$

따라서:

$$E[\tilde{S}^2] = E\left[\frac{1}{n}\sum_{i=1}^n(X_i - \bar{X})^2\right] = \frac{n-1}{n}\sigma^2$$

### 편향

$$\text{Bias}(\tilde{S}^2) = E[\tilde{S}^2] - \sigma^2 = \frac{n-1}{n}\sigma^2 - \sigma^2 = -\frac{\sigma^2}{n}$$

소박한 추정량은 참 분산을 $(n-1)/n$배만큼 **과소추정**한다.

## 직관: 편향이 생기는 이유

편향은 제곱합에서 $\mu$ 대신 $\bar{X}$를 쓰기 때문에 생긴다. $\bar{X}$는 모든 상수 $c$에 대해 $\sum(X_i - c)^2$을 최소화하는 값이므로:

$$\sum_{i=1}^n (X_i - \bar{X})^2 \leq \sum_{i=1}^n (X_i - \mu)^2$$

$\bar{X}$로부터의 편차제곱합은 **항상** 참 평균 $\mu$로부터의 제곱합보다 작거나 같다. $\bar{X}$를 사용함으로써 변동성을 체계적으로 적게 세게 되고, 이것이 아래쪽 편향으로 이어진다.

다르게 보는 방법도 있다: $\bar{X}$를 계산하는 데 자료의 정보 한 조각이 "소모된다". $n$개의 편차 $(X_i - \bar{X})$는 $\sum(X_i - \bar{X}) = 0$을 만족하므로 자유롭게 변할 수 있는 것은 $n-1$개뿐이다. **자유도**가 $n$이 아니라 $n-1$이다.

## 성질

### S-tilde-squared의 분산
정규모집단에서:

$$\text{Var}(\tilde{S}^2) = \frac{2(n-1)}{n^2}\sigma^4$$

### S-tilde-squared의 평균제곱오차

$$\text{MSE}(\tilde{S}^2) = \text{Var}(\tilde{S}^2) + [\text{Bias}(\tilde{S}^2)]^2 = \frac{2(n-1)}{n^2}\sigma^4 + \frac{\sigma^4}{n^2} = \frac{2n-1}{n^2}\sigma^4$$

### 일치성

편향되어 있음에도 $\tilde{S}^2$은 **일치**한다:

$$\tilde{S}^2 = \frac{n-1}{n} S^2 \xrightarrow{p} \sigma^2$$

$(n-1)/n \to 1$이고 $S^2 \xrightarrow{p} \sigma^2$이기 때문이다.

### 점근적 동등성

큰 $n$에서 $\tilde{S}^2$과 $S^2$은 사실상 같다:

$$\tilde{S}^2 = \frac{n-1}{n}S^2 \approx S^2 \quad (\text{큰 } n \text{에 대해})$$

편향 $-\sigma^2/n \to 0$이고 비 $(n-1)/n \to 1$이다.

## 비교: n, n-1, n+1로 나누기
| 추정량 | 나누는 수 | 편향 | 평균제곱오차 (정규) | 비고 |
|-----------|---------|------|--------------|-------|
| $\tilde{S}^2$ | $n$ | $-\sigma^2/n$ | $\frac{2n-1}{n^2}\sigma^4$ | MLE, 편향됨 |
| $S^2$ | $n-1$ | $0$ | $\frac{2}{n-1}\sigma^4$ | 불편 (Bessel) |
| $\hat{S}^2$ | $n+1$ | $-\frac{2}{n+1}\sigma^2$ | $\frac{2(n-1)}{(n+1)^2}\sigma^4 + \frac{4}{(n+1)^2}\sigma^4$ | 평균제곱오차 최적 (정규) |

**놀라운 사실:** 모든 $n$에 대해 $\text{MSE}(\tilde{S}^2) < \text{MSE}(S^2)$이다. 편향추정량이 불편추정량보다 평균제곱오차가 작다! 편향–분산 맞바꿈의 교과서적 예이다.

## mu를 아는 경우
참 평균 $\mu$가 알려져 있다면(실무에서는 드물다) 다음을 쓸 수 있다:

$$\hat{\sigma}^2_\mu = \frac{1}{n}\sum_{i=1}^n (X_i - \mu)^2$$

이 추정량은 불편이며($E[\hat{\sigma}^2_\mu] = \sigma^2$) $S^2$보다 분산이 작다:

$$\text{Var}(\hat{\sigma}^2_\mu) = \frac{2\sigma^4}{n} < \frac{2\sigma^4}{n-1} = \text{Var}(S^2)$$

## MLE와의 연결

정규모집단에서 $\tilde{S}^2$은 $\sigma^2$의 MLE이다. MLE는 유한표본에서 편향되지만 점근적으로 불편이다. 이는 흔한 양상이다: MLE는 유한표본에서 편향되는 경우가 많지만 일치한다.

## 금융과의 연결

- **변동성 추정**: 일별 수익률의 실현분산은 ($\mu \approx 0$으로 두고) $\frac{1}{n}\sum r_i^2$ 공식을 쓰는데, 이는 평균을 0으로 놓은 소박한 추정량이다.
- **위험 지표**: 큰 표본($n > 250$ 일별 관측값)으로 위험관리용 포트폴리오 분산을 계산할 때는 $n$으로 나누느냐 $n-1$로 나누느냐의 차이가 무시할 만하다.
- **편향 보정**: 작은 표본(예: 몇 년치 월별 자료)에서는 편향이 실질적일 수 있으므로 Bessel 수정을 써야 한다.

## 요약

소박한 분산추정량 $\tilde{S}^2 = \frac{1}{n}\sum(X_i - \bar{X})^2$은 참 평균 대신 표본평균을 쓰면 변동성을 체계적으로 과소추정하기 때문에 $\sigma^2/n$만큼 아래로 편향된다. 이 편향에도 불구하고 불편추정량 $S^2$보다 평균제곱오차가 작고 일치한다. 또한 정규모집단에서 MLE이기도 하다. 큰 표본에서는 편향이 무시할 만하지만, 작은 표본에서는 Bessel 수정($n-1$로 나누기)이 표준이다.

## 핵심 공식

| 양 | 공식 |
|----------|---------|
| 소박한 추정량 | $\tilde{S}^2 = \frac{1}{n}\sum(X_i - \bar{X})^2$ |
| 기댓값 | $E[\tilde{S}^2] = \frac{n-1}{n}\sigma^2$ |
| 편향 | $-\sigma^2/n$ |
| 평균제곱오차 (정규) | $\frac{2n-1}{n^2}\sigma^4$ |
| 핵심 항등식 | $\sum(X_i - \bar{X})^2 = \sum(X_i - \mu)^2 - n(\bar{X}-\mu)^2$ |

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
$\mathbb{E}[(1/n)\sum(X_i - \bar X)^2] = (n-1)\sigma^2/n$임을 증명하라.

</div>

??? success "풀이"
    항등식 $\sum(X_i - \bar X)^2 = \sum(X_i - \mu)^2 - n(\bar X - \mu)^2$을 쓴다.

    $\mathbb{E}[\sum(X_i - \mu)^2] = n\sigma^2$. $\mathbb{E}[n(\bar X - \mu)^2] = n \cdot \sigma^2/n = \sigma^2$.

    $\mathbb{E}[\sum(X_i - \bar X)^2] = (n-1)\sigma^2$. $n$으로 나누면 $(n-1)\sigma^2/n$. $\square$

    소박한 추정량은 $\sigma^2/n$만큼 아래로 편향된다. Bessel 수정은 $n/(n-1)$을 곱해 이를 바로잡는다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
**평균을 아는 경우의 분산.** $\mu$가 알려져 있으면 $\hat\sigma^2 = (1/n)\sum(X_i - \mu)^2$이다. (a) 불편인가? (b) 정규성 아래에서의 분산. (c) $S^2$ 대비 효율 이득.

</div>

??? success "풀이"
    (a) $\mathbb{E}[\hat\sigma^2] = (1/n) \cdot n\sigma^2 = \sigma^2$. **불편**이다($\mu$를 추정하느라 소모한 자유도가 없으므로 Bessel 수정이 필요 없다).

    (b) $n\hat\sigma^2/\sigma^2 \sim \chi^2_n$(표준정규 제곱 $n$개의 합). $\mathrm{Var}(\hat\sigma^2) = 2\sigma^4/n$.

    (c) $\mathrm{Var}(S^2) = 2\sigma^4/(n-1)$(자유도가 하나 적다). 효율 이득: $(n-1)/n$.

    $n = 10$이면 이득이 10%이다($\mu$를 아는 것이 관측값 약 1개의 가치가 있다). $n = 100$이면 1%로 무시할 만하다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
**편향은 왜 생기는가?** 직관적으로, 소박한 추정량이 $\sigma^2$을 과소추정하는 이유는?

</div>

??? success "풀이"
    $\bar X$는 $c$에 대해 $\sum(X_i - c)^2$을 최소화한다. 따라서 항상 $\sum(X_i - \bar X)^2 \le \sum(X_i - \mu)^2$이다.

    기댓값을 취하면 $\mathbb{E}[\sum(X_i - \bar X)^2] \le \mathbb{E}[\sum(X_i - \mu)^2] = n\sigma^2$이다. 구체적으로는 정확히 $\sigma^2$만큼 작다($\mathbb{E}[n(\bar X - \mu)^2] = \sigma^2$).

    **개념적으로:** $\bar X$를 자료에 맞추었기 때문에 잔차 $(X_i - \bar X)$는 "참값으로부터의 편차보다 작다" — $\bar X$가 표본에 적응했기 때문이다. 이는 통계학 전반에서 자유도 보정을 낳는 것과 같은 "이중 계산" 문제이다(회귀의 $R^2$, AIC 벌점 등).

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
**추정량 $\hat\sigma^2_{c}$ 계열.** $c = n, n-1, n+1$에 대해 $\hat\sigma^2_c = (1/c)\sum(X_i - \bar X)^2$의 편향을 구하라.

</div>

??? success "풀이"
    $\mathbb{E}[\sum(X_i - \bar X)^2] = (n-1)\sigma^2$을 쓰면:

    - $c = n$ (MLE): 편향 = $(n-1)\sigma^2/n - \sigma^2 = -\sigma^2/n$. 과소추정.
    - $c = n - 1$ (불편): 편향 = 0.
    - $c = n + 1$ (평균제곱오차 최적): 편향 = $(n-1)\sigma^2/(n+1) - \sigma^2 = -2\sigma^2/(n+1)$. MLE보다 더 과소추정.

    맞바꿈: 나누는 수가 작을수록 편향은 작아지지만 분산은 커진다. 나누는 수가 클수록 편향은 커지지만 분산은 작아진다(더 강한 축소).

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
**평균을 아는 표본분산이 모르는 경우보다 낫다.** 구체적으로, 이 분산추정량은 보정 없이도 불편이다. 이것이 카이제곱분포를 보존함을 보여라.

</div>

??? success "풀이"
    $\mu$를 아는 경우: $(X_i - \mu)/\sigma \sim N(0, 1)$이므로 $(X_i - \mu)^2/\sigma^2 \sim \chi^2_1$이고 $\sum(X_i - \mu)^2/\sigma^2 \sim \chi^2_n$이다.

    $\mu$를 추정하는 경우: $\sum(X_i - \bar X)^2/\sigma^2 \sim \chi^2_{n-1}$이다($\bar X$ 때문에 자유도 하나를 잃는다).

    자유도 하나의 차이가 Bessel 수정에 담겨 있다. 더 깊은 이유는 $\bar X$로부터의 잔차가 합이 0이라는 제약을 받아 자유도 하나가 사라지기 때문이다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
**회귀 맥락에서의 소박한 분산.** 선형회귀가 $\mathrm{SSE}/n$이나 $\mathrm{SSE}/(n-1)$이 아니라 **잔차분산** $\hat\sigma^2 = \mathrm{SSE}/(n - p)$를 보고하는 이유는?

</div>

??? success "풀이"
    모수가 $p$개인 선형회귀는 $\mathrm{SSE} = \sum(Y_i - \hat Y_i)^2$으로부터 잔차분산을 추정한다.

    적합된 모수 하나마다 자유도 하나가 사라진다. (절편을 포함하여) 모수가 $p$개이면 잔차의 유효 자유도는 $n - p$이다.

    $\hat\sigma^2 = \mathrm{SSE}/(n - p)$는 불편이다: $\mathbb{E}[\mathrm{SSE}/\sigma^2] = n - p$.

    **경계 사례:**

    - $p = 1$ (절편만): Bessel 수정을 한 표본분산과 같아 $n - 1$로 나눈다.
    - $p = n$ (완벽한 적합): 남은 자유도가 없어 분산이 정의되지 않는다. 과적합을 반영한다.

    자유도 보정은 Bessel 수정을 다중모수 상황으로 일반화한 것이다 — 원리는 같고, 빼야 할 모수가 더 많을 뿐이다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
소박한 추정량 $\tilde S^2 = \frac1n\sum(X_i-\bar X)^2$은 편향되어 있지만 **최대가능도추정량**이다. 왜 MLE가 편향될 수 있는지, 그리고 이 편향이 어디서 오는지 자유도로 설명하라.

</div>

??? success "풀이"
    **MLE인 이유.** 정규 로그가능도를 $\sigma^2$으로 미분해 0으로 두면

    $$
    -\frac{n}{2\sigma^2}+\frac{\sum(x_i-\hat\mu)^2}{2\sigma^4} = 0 \implies \hat\sigma^2 = \frac1n\sum(x_i-\bar x)^2
    $$

    로 분모가 $n$이다.

    **왜 편향되는가.** MLE는 **가능도를 최대로 하는 값**을 고를 뿐, 불편성을 겨냥하지 않는다. 두 기준이 다르므로 일치할 이유가 없다.

    더 구체적으로, MLE는 $\mu$를 $\bar X$로 대체하는데 **$\bar X$가 바로 그 제곱합을 최소로 하는 값**이다. 따라서 $\sum(x_i-\bar x)^2$이 $\sum(x_i-\mu)^2$보다 체계적으로 작고, 그 차이가

    $$
    \sum_i(x_i-\mu)^2-\sum_i(x_i-\bar x)^2 = n(\bar x-\mu)^2
    $$

    이며 기대값이 $\sigma^2$이다. 자유도 하나를 잃은 것이다.

    $$
    E\left[\sum_i(X_i-\bar X)^2\right] = n\sigma^2-\sigma^2 = (n-1)\sigma^2
    $$

    **일반 원리.** **MLE는 모수를 추정하면서 "자료에 최적으로 맞추므로" 잔차를 과소평가한다.** 회귀에서 $\text{RSS}/n$이 $\sigma^2$을 과소평가하는 것, 훈련오차가 검정오차보다 낙관적인 것이 모두 같은 현상이다.

    **크기.** 편향이 $-\sigma^2/n$으로 $O(1/n)$이다. $p$개 모수를 추정하는 회귀에서는 $-p\sigma^2/n$으로 **모수가 많을수록 커진다.** $p/n=0.3$이면 30% 과소평가다.

    **그래도 MLE를 쓰는 경우.** 정보기준(AIC, BIC)을 계산할 때는 MLE를 써야 한다. 벌점 항이 바로 이 낙관성을 보정하도록 설계되었기 때문이며, 불편추정량을 넣으면 이중 보정이 된다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
평균 $\mu$를 아는 경우와 모르는 경우의 분산추정을 **정보량** 관점에서 비교하라. $\mu$를 모르는 대가가 얼마인가?

</div>

??? success "풀이"
    **$\mu$를 아는 경우.**

    $$
    \hat\sigma^2_{\text{known}} = \frac1n\sum_i(X_i-\mu)^2, \qquad \frac{n\hat\sigma^2_{\text{known}}}{\sigma^2}\sim\chi^2_n
    $$

    불편이고 $\operatorname{Var} = 2\sigma^4/n$이다.

    **$\mu$를 모르는 경우.**

    $$
    S^2 = \frac{1}{n-1}\sum_i(X_i-\bar X)^2, \qquad \operatorname{Var}(S^2) = \frac{2\sigma^4}{n-1}
    $$

    **대가.**

    $$
    \frac{\operatorname{Var}(S^2)}{\operatorname{Var}(\hat\sigma^2_{\text{known}})} = \frac{n}{n-1}
    $$

    으로 **관측 하나를 잃은 것과 같다.**

    | $n$ | 분산 증가 |
    |---|---|
    | 5 | 25% |
    | 10 | 11% |
    | 50 | 2% |
    | 100 | 1% |

    **정보량으로 보면.** 정규분포의 정보행렬이 대각이었음을 떠올리면

    $$
    I(\mu,\sigma^2) = \begin{pmatrix}1/\sigma^2 & 0\\0&1/(2\sigma^4)\end{pmatrix}
    $$

    이고 **비대각이 0**이다. 앞서 본 대로 이는 두 모수가 직교한다는 뜻이며, $\mu$를 모르는 것이 $\sigma^2$의 **점근분산**을 전혀 키우지 않음을 함의한다.

    실제로 $2\sigma^4/(n-1)$과 $2\sigma^4/n$의 차이는 $O(1/n^2)$로, 점근적으로는 같다. **직교성이 보장하는 것이 바로 이것이다.**

    **직교하지 않는 경우와의 대비.** 감마분포의 $(\alpha,\beta)$는 직교하지 않아서, 앞서 본 대로 $\beta$를 모르는 대가로 $\alpha$의 점근분산이 2.5배 이상 커졌다. **정규분포에서 그 대가가 $O(1/n)$에 그치는 것은 특별한 일**이다.

    **실무적 함의.** $\mu$를 아는 경우는 드물지만, 평균이 0으로 알려진 경우가 있다. 금융 수익률의 일별 평균, 잔차, 차분한 시계열이 그렇다. 그때는 $\sum x_i^2/n$을 쓰는 것이 옳고, 굳이 평균을 빼면 자유도를 공짜로 잃는다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
제곱합을 계산하는 세 가지 방법을 비교하라. 수치적으로 어느 것이 안전한가?

</div>

??? success "풀이"
    **방법 1 — 간편식.**

    $$
    \text{SS} = \sum_i x_i^2 - n\bar x^2
    $$

    한 번만 훑으면 되지만 **거의 같은 두 큰 수를 뺀다.** 앞서 본 대로 $x$의 평균이 표준편차에 비해 크면 유효숫자가 모두 날아간다. $x = 10^8+\{1,2,3\}$에서 참값 1 대신 0이 나온다.

    **방법 2 — 두 번 훑기.**

    $$
    \text{SS} = \sum_i(x_i-\bar x)^2
    $$

    평균을 먼저 구하고 편차를 제곱한다. **상쇄가 없어 안전**하다. NumPy의 `var`가 이 방식이며, 자료를 메모리에 담을 수 있으면 최선이다.

    **방법 3 — 웰퍼드 갱신.**

    $$
    \bar x_k = \bar x_{k-1}+\frac{x_k-\bar x_{k-1}}{k}, \qquad M_k = M_{k-1}+(x_k-\bar x_{k-1})(x_k-\bar x_k)
    $$

    한 번만 훑으면서도 안정적이다. 갱신량이 편차 규모라 상쇄가 없다.

    **비교.**

    | | 훑는 횟수 | 수치 안정성 | 스트리밍 |
    |---|---|---|---|
    | 간편식 | 1 | **나쁨** | 가능 |
    | 두 번 훑기 | 2 | 좋음 | 불가 |
    | 웰퍼드 | 1 | 좋음 | **가능** |

    **권고.** 자료를 저장할 수 있으면 두 번 훑기, 스트리밍이면 웰퍼드. **간편식은 쓰지 않는다.**

    **개선된 두 번 훑기.** 더 정확하게 하려면 보정항을 더한다.

    $$
    \text{SS} = \sum_i(x_i-\bar x)^2 - \frac{1}{n}\left\{\sum_i(x_i-\bar x)\right\}^2
    $$

    수학적으로 둘째 항은 0이지만, 부동소수점에서는 $\bar x$의 반올림오차를 보정해 준다.

    **병렬 계산.** 자료를 여러 조각으로 나눠 각각 $(n_j,\bar x_j, M_j)$를 계산한 뒤 채의 결합 공식으로 합칠 수 있다.

    $$
    M_{AB} = M_A+M_B+\frac{n_An_B}{n_A+n_B}(\bar x_A-\bar x_B)^2
    $$

    분산분석의 제곱합 분해와 같은 구조임을 알아볼 수 있다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
가중 자료에서 분산의 **불편추정량**은 무엇인가? 빈도 가중과 신뢰도 가중을 구분해 설명하라.

</div>

??? success "풀이"
    **두 종류의 가중치를 구분해야 한다.**

    **(1) 빈도 가중.** $w_i$가 "이 값이 $w_i$번 관측되었다"는 뜻이다. 정수이며, 실질 표본크기가 $\sum_i w_i$다.

    $$
    \bar x_w = \frac{\sum_i w_ix_i}{\sum_i w_i}, \qquad s_w^2 = \frac{\sum_i w_i(x_i-\bar x_w)^2}{\sum_i w_i-1}
    $$

    **분모가 $\sum w_i - 1$**이다. 자료를 풀어 쓴 것과 정확히 같으므로 자연스럽다.

    **(2) 신뢰도 가중.** $w_i$가 관측의 정밀도를 나타낸다(예: $w_i\propto1/\sigma_i^2$, 또는 표본조사의 설계 가중치). **실질 표본크기가 $\sum w_i$가 아니다.**

    이때 불편추정량은

    $$
    s_w^2 = \frac{\sum_i w_i(x_i-\bar x_w)^2}{\sum_i w_i-\dfrac{\sum_i w_i^2}{\sum_i w_i}}
    $$

    이다. 분모가 **유효 자유도**이며, 모든 $w_i$가 같으면 $n-1$로 돌아온다.

    **유효 표본크기.** 분모의 첫 항을 정리하면 커시의 유효 표본크기가 나온다.

    $$
    n_{\text{eff}} = \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2}
    $$

    가중치가 고르면 $n_{\text{eff}}=n$이고, 몇몇에 몰려 있으면 훨씬 작아진다. 가중치가 극단적으로 불균등한 조사에서 **유효 표본크기가 명목의 절반 이하**로 떨어지는 일이 흔하다.

    **실무의 함정.** 소프트웨어가 어느 가중치를 가정하는지 반드시 확인해야 한다.

    - NumPy의 `np.average(x, weights=w)`는 평균만 주고 분산은 주지 않는다.
    - `statsmodels`의 `DescrStatsW`는 `ddof`와 가중치 해석을 인자로 받는다.
    - R의 `weighted.mean`과 조사 패키지(`survey`)가 다른 규약을 쓴다.

    **잘못 쓰면** 표준오차가 크게 어긋난다. 설계 가중치를 빈도 가중으로 취급하면 유효 표본크기를 과대평가해 신뢰구간이 부당하게 좁아진다. 복잡한 조사 자료에서는 **전용 조사 분석 도구**를 쓰는 것이 안전하다.

---

## 정리하며

$n$ 으로 나누는 소박한 추정량은 **체계적으로 과소추정한다.**

$$
\mathbb{E}[\tilde S^2] = \frac{n-1}{n}\,\sigma^2 < \sigma^2
$$

- **원인은 $\mu$ 가 아니라 $\bar X$ 로부터의 편차를 재기 때문이다.** $\bar X$ 는 정의상 자료에 가장 가까운 점이므로 $\sum(X_i-\bar X)^2$ 이 $\sum(X_i-\mu)^2$ 보다 **언제나 작거나 같다.**
- **자유도로 읽으면 더 분명하다.** 편차 $n$ 개가 $\sum(X_i-\bar X)=0$ 이라는 제약 하나를 받으므로 자유롭게 움직이는 방향이 $n-1$ 개뿐이다.
- **편향의 크기는 $-\sigma^2/n$** 으로 $n$ 이 커지면 사라진다. $n=10$ 이면 $10\%$ 과소추정이고 $n=100$ 이면 $1\%$ 다. **소표본에서만 문제가 된다.**
- **이것이 정규분포에서 $\sigma^2$ 의 최대가능도추정량이다.** 최대가능도가 불편성을 보장하지 않는다는 사실의 가장 익숙한 예다.
- **$\mu$ 를 안다면 $n$ 으로 나누는 것이 옳다.** $\frac1n\sum(X_i-\mu)^2$ 은 불편이며, 문제는 나눗수가 아니라 **중심을 추정했다는 사실**에 있다.

다음 절 **베셀 수정**은 이 편향을 정확히 되돌리는 방법을 다룬다.
