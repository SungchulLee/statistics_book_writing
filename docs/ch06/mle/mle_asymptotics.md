# MLE의 점근적 성질

## 점근이론이 중요한 이유

유한표본에서 MLE의 거동은 구체적인 모형과 자료에 달려 있다. 그러나 일반적인 정칙 조건 아래에서 MLE는 표본크기 $n$이 커질 때 강력한 성질 셋을 갖는다. 참 모수값으로 수렴하고(일치성), 근사적으로 정규분포를 따르게 되며(점근정규성), 가능한 최선의 정밀도를 달성한다(점근 효율성). 이 성질들이 MLE를 기본 추정 전략으로 삼는 것을 정당화하고, 가능도에 기반한 신뢰구간, 가설검정, 모형 비교 도구의 이론적 근거가 된다.

이 페이지에서 $\theta_0$은 **참 모수값**, 즉 실제로 자료를 만들어 낸 값을 나타낸다. $X_1, \ldots, X_n \overset{\text{iid}}{\sim} f(x; \theta_0)$을 가정한다.

## 정칙 조건

아래의 점근적 결과들은 일련의 정칙 조건에 의존한다. 이 조건들은 표준 이론이 적용될 만큼 가능도함수가 잘 행동하도록 보장한다.

1. **식별 가능성.** 서로 다른 모수값이 서로 다른 분포를 만든다. $\theta_1 \neq \theta_2$이면 적어도 하나의 $x$에서 $f(x; \theta_1) \neq f(x; \theta_2)$이다.

2. **공통 지지집합.** 집합 $\{x : f(x; \theta) > 0\}$이 $\theta$에 의존하지 않는다. 모수에 따라 지지집합이 바뀌는 $\text{Uniform}(0, \theta)$ 같은 분포는 제외된다.

3. **내부의 모수.** 참값 $\theta_0$이 경계가 아니라 모수공간 $\Theta$의 내부에 있다.

4. **매끄러움.** 로그밀도 $\log f(x; \theta)$가 $\theta$에 대해 적어도 세 번 연속미분가능하고, 도함수를 적분(또는 합) 기호 안으로 넘길 수 있다.

5. **양의 Fisher 정보량.** Fisher 정보량이 $I(\theta_0) > 0$이어서 자료가 실제로 $\theta$에 관한 정보를 담고 있다.

!!! warning "정칙 조건이 깨질 때"

    정칙 조건이 위배되면 표준 점근 결과가 완전히 무너질 수 있다. 예를 들어 $\text{Uniform}(0, \theta)$에서 $\theta$의 MLE는 $X_{(n)}$(표본 최댓값)인데, 통상적인 $1/\sqrt{n}$이 아니라 $1/n$의 속도로 수렴하고 극한분포가 정규가 아니라 지수분포이다.

## 일치성

좋은 추정량이라면 자료를 더 모을수록 참값으로 수렴해야 한다. **일치성**은 이 최소한의 요구를 형식화한다.

위의 정칙 조건 아래에서 MLE는 일치한다:

$$
\hat{\theta}_{\text{MLE}} \xrightarrow{P} \theta_0 \quad \text{as } n \to \infty
$$

이는 임의의 $\epsilon > 0$에 대해 다음이 성립함을 뜻한다:

$$
P(|\hat{\theta}_{\text{MLE}} - \theta_0| > \epsilon) \to 0 \quad \text{as } n \to \infty
$$

증명은 로그가능도비 $\ell(\theta) - \ell(\theta_0)$이 Kullback-Leibler 발산 $-\text{KL}(f_{\theta_0} \| f_\theta)$으로 균등수렴하고, 식별 가능성에 의해 이것이 $\theta = \theta_0$에서 유일하게 최대가 된다는 사실에 의존한다.

!!! tip "일치성은 필요하지만 충분하지는 않다"

    일치성만으로는 추정량이 얼마나 빨리 수렴하는지, 그 분포가 어떤 모양인지 알 수 없다. 일치하는 두 추정량이 유한표본에서 매우 다르게 거동할 수 있다. 다음의 두 성질이 속도와 분포 모양을 다룬다.

## 점근정규성

MLE의 점근적 성질 중 실무적으로 가장 중요한 것은 $n$이 클 때 표본분포가 근사적으로 정규가 된다는 점이다.

정칙 조건 아래에서 MLE는 다음을 만족한다:

$$
\sqrt{n}(\hat{\theta}_{\text{MLE}} - \theta_0) \xrightarrow{d} N\left(0, \frac{1}{I(\theta_0)}\right)
$$

동등하게, $n$이 크면:

$$
\hat{\theta}_{\text{MLE}} \overset{\text{approx}}{\sim} N\left(\theta_0, \frac{1}{nI(\theta_0)}\right)
$$

이 결과는 곧바로 실용적인 귀결을 낳는다. $\theta_0$에 대한 근사적인 $(1 - \alpha)$ 신뢰구간은

$$
\hat{\theta}_{\text{MLE}} \pm z_{\alpha/2} \cdot \frac{1}{\sqrt{nI(\hat{\theta}_{\text{MLE}})}}
$$

여기서 $z_{\alpha/2}$는 표준정규분포의 $(1 - \alpha/2)$ 분위수이다. $\theta_0$을 모르므로 실무에서는 Fisher 정보량을 $\hat{\theta}_{\text{MLE}}$에서 평가한다.

!!! example "베르누이 모수의 점근 신뢰구간"

    $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Bernoulli}(p)$에서 MLE는 $\hat{p} = \bar{X}$이고 Fisher 정보량은 $I(p) = 1/(p(1-p))$이다. 근사적인 95% 신뢰구간은

    $$
    \hat{p} \pm 1.96 \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
    $$

    $n = 100$이고 $\hat{p} = 0.3$이면 $0.3 \pm 1.96\sqrt{0.21/100} = 0.3 \pm 0.090$, 즉 근사적으로 $(0.210, 0.390)$이다.

## 점근 효율성

점근정규성은 MLE가 점근분산 $1/(nI(\theta_0))$으로 $1/\sqrt{n}$의 속도로 수렴한다고 알려 준다. 그런데 이 분산이 우리가 달성할 수 있는 최선인가? 답은 그렇다이다.

정칙추정량 중에서 MLE는 점근적으로 **Cramér-Rao 하한**을 달성한다:

$$
\text{Var}_{\text{asy}}(\hat{\theta}_{\text{MLE}}) = \frac{1}{nI(\theta_0)}
$$

이는 다른 어떤 정칙 일치추정량도 이보다 작은 점근분산을 가질 수 없음을 뜻한다. 일치하고 점근적으로 정규인 다른 추정량 $\tilde{\theta}$는 반드시 다음을 만족한다:

$$
\text{Var}_{\text{asy}}(\tilde{\theta}) \geq \frac{1}{nI(\theta_0)}
$$

등호는 $\tilde{\theta}$가 MLE와 점근적으로 동등할 때에만 성립한다.

!!! note "초효율성"

    모수공간의 고립된 점에서 MLE보다 분산이 작은 추정량을 구성할 수 있다(Hodges 추정량이 고전적인 예이다). 그러나 그런 추정량은 다른 점에서 반드시 분산이 더 커진다. MLE는 정칙추정량 부류 안에서 모수공간 전체에 걸쳐 일률적으로 효율적이다.

## 세 가지 성질 요약

| 성질 | 서술 | 실무적 함의 |
|---|---|---|
| 일치성 | $\hat{\theta}_{\text{MLE}} \xrightarrow{P} \theta_0$ | 추정값이 참값으로 수렴한다 |
| 점근정규성 | $\sqrt{n}(\hat{\theta}_{\text{MLE}} - \theta_0) \xrightarrow{d} N(0, 1/I(\theta_0))$ | 정규근사로 신뢰구간과 검정을 얻는다 |
| 점근 효율성 | $\text{Var}_{\text{asy}} = 1/(nI(\theta_0))$ | 어떤 정칙추정량도 더 나을 수 없다 |

이 세 결과를 종합하면, 정칙 조건이 성립하고 점근근사가 믿을 만할 만큼 표본이 크다면 MLE가 모수적 추정의 기본 선택이 된다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
정칙 조건 아래에서 MLE가 갖는 세 가지 주요 점근적 성질인 일치성, 점근정규성, 점근 효율성을 서술하라.

</div>

??? success "풀이"
    표준적인 정칙 조건 아래에서:

    1. **일치성:** $n \to \infty$일 때 $\hat{\theta}_{\text{MLE}} \xrightarrow{p} \theta_0$.

    2. **점근정규성:** $\sqrt{n}(\hat{\theta}_{\text{MLE}} - \theta_0) \xrightarrow{d} N(0, I(\theta_0)^{-1})$이며, $I(\theta_0)$은 관측값 하나에 대한 Fisher 정보량이다.

    3. **점근 효율성:** MLE는 점근적으로 Cramér-Rao 하한을 달성한다. 즉 어떤 일치추정량도 $1/(nI(\theta_0))$보다 작은 점근분산을 가질 수 없다.

    이 성질들 덕분에 MLE는 대표본 모수적 추정의 기본 선택이 된다. 다만 소표본에서는 MLE가 편향되거나 다른 추정량보다 효율이 낮을 수 있다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
비율 모수가 $\lambda$인 지수분포에서 MLE는 $\hat{\lambda} = 1/\bar{X}$이다. Fisher 정보량 $I(\lambda)$를 계산하고 $\hat{\lambda}$의 점근분산이 $\lambda^2/n$임을 확인하라.

</div>

??? success "풀이"
    관측값 하나에 대한 로그가능도는 $\ell(\lambda) = \log\lambda - \lambda x$이다. 2계도함수는:

    $$
    \frac{d^2\ell}{d\lambda^2} = -\frac{1}{\lambda^2}
    $$

    Fisher 정보량은:

    $$
    I(\lambda) = -E\!\left[\frac{d^2\ell}{d\lambda^2}\right] = \frac{1}{\lambda^2}
    $$

    MLE의 점근분산은:

    $$
    \text{Var}(\hat{\lambda}) \approx \frac{1}{nI(\lambda)} = \frac{\lambda^2}{n}
    $$

    따라서 $n$이 크면 $\hat{\lambda} \approx N(\lambda, \lambda^2/n)$이고 표준오차는 $\text{SE}(\hat{\lambda}) \approx \lambda/\sqrt{n}$이다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
불변성은 $\hat{\theta}$가 $\theta$의 MLE이면 임의의 함수 $g$에 대해 $g(\hat{\theta})$가 $g(\theta)$의 MLE라는 성질이다. 이를 이용하여 지수분포의 평균 $1/\lambda$의 MLE를 구하라.

</div>

??? success "풀이"
    비율 모수의 MLE는 $\hat{\lambda} = 1/\bar{X}$이다. 불변성에 의해 (모평균인) $g(\lambda) = 1/\lambda$의 MLE는:

    $$
    \widehat{1/\lambda} = g(\hat{\lambda}) = \frac{1}{\hat{\lambda}} = \frac{1}{1/\bar{X}} = \bar{X}
    $$

    지수분포 평균의 MLE가 표본평균이라는 직관적인 결과가 확인된다. 불변성은 MLE를 처음부터 다시 유도하지 않고도 비선형 변환을 포함한 임의의 변환에 대해 작동하므로 강력하다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
MLE가 점근적으로는 불편인데도 유한표본에서 편향될 수 있는 이유를 설명하라. 구체적인 예를 들라.

</div>

??? success "풀이"
    점근적 불편성은 $n \to \infty$일 때 $E[\hat{\theta}_n] \to \theta_0$임을 뜻하지만, 고정된 $n$에 대해서는 편향 $E[\hat{\theta}_n] - \theta_0$이 0이 아닐 수 있다.

    **예:** 정규분포의 분산에서 MLE는 $\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum(X_i - \bar{X})^2$이며 다음을 만족한다:

    $$
    E[\hat{\sigma}^2_{\text{MLE}}] = \frac{n-1}{n}\sigma^2 \neq \sigma^2
    $$

    편향은 $-\sigma^2/n$으로 $n \to \infty$일 때 사라지지만 모든 유한한 $n$에서는 0이 아니다. 이 때문에 불편추정량 $S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$이 Bessel 수정 $n - 1$을 사용한다.

    일반적으로 $\hat{\theta}$가 $\theta$의 MLE이고 $g$가 비선형이면 불변성에 의해 $g(\hat{\theta})$가 $g(\theta)$의 MLE이지만, Jensen 부등식에 의해 유한표본에서 $E[g(\hat{\theta})] \neq g(\theta)$가 되어 편향이 생긴다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
$X_i \sim \text{Uniform}(0,\theta)$에서 $\hat\theta = \max_i X_i$는 **정칙 조건을 만족하지 않는다.** 어느 조건이 깨지는지 밝히고, 그 결과 점근분포가 어떻게 달라지는지 보여라.

</div>

??? success "풀이"
    **깨지는 조건.** 정칙 조건의 핵심은 **밀도의 지지집합이 모수에 의존하지 않는다**는 것이다. 여기서는 지지집합이 $[0,\theta]$로 $\theta$에 직접 의존한다.

    그 결과 미분과 적분을 맞바꿀 수 없게 되고, 점수함수의 기대값이 0이라는 기본 항등식 $E[\ell'(\theta)] = 0$이 무너진다. 실제로 로그가능도 $\ell(\theta) = -n\ln\theta$($\theta \ge x_{(n)}$)는 $\theta$에 대해 **감소**하며 도함수가 0이 되는 점이 없다. MLE는 미분이 아니라 제약의 경계 $\theta = x_{(n)}$에서 나온다.

    **점근분포.** $M = \max_i X_i$의 CDF가 $(m/\theta)^n$이므로, $t > 0$에 대해

    $$
    P\left\{n(\theta - M) > t\right\} = P\left(M < \theta-\frac tn\right) = \left(1-\frac{t}{n\theta}\right)^n \to e^{-t/\theta}
    $$

    이다. 따라서

    $$
    n(\theta-\hat\theta) \xrightarrow{d} \text{Exp}(1/\theta)
    $$

    **정칙 경우와의 대비.**

    | | 정칙 | 균등 |
    |---|---|---|
    | 수렴 속도 | $\sqrt n$ | $n$ (**더 빠름**) |
    | 극한분포 | 정규 | 지수 |
    | 대칭성 | 대칭 | 한쪽으로 치우침 |
    | 크라메르-라오 | 점근적으로 달성 | **적용 불가** |

    수렴이 $\sqrt n$보다 빠른 것을 **초일치성**이라 한다. 크라메르-라오 하한이 $\hat\theta$의 분산에 대해 아무것도 말해 주지 않는 이유가 여기 있다. 하한 자체가 정칙 조건 아래에서만 유도되기 때문이다.

    **실무적 함의.** 이런 모형에서는 표준 오차와 왈드 구간을 쓰면 안 된다. 정확한 분포를 직접 쓰거나, 부트스트랩을 쓴다면 $m$-out-of-$n$ 같은 변형이 필요하다. 경제학의 경계모형, 생존분석의 임계 모수, 극단값 모형에서 같은 문제가 나타난다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
**델타 방법**을 서술하고, MLE의 점근정규성과 결합해 $g(\hat\theta)$의 점근분포를 구하라. $\hat\lambda$의 점근분포에서 $\hat\mu = 1/\hat\lambda$의 점근분포를 유도해 확인하라.

</div>

??? success "풀이"
    **델타 방법.** $\sqrt n(\hat\theta-\theta) \xrightarrow{d} N(0, \sigma^2)$이고 $g$가 $\theta$에서 미분가능하며 $g'(\theta) \ne 0$이면

    $$
    \sqrt n\left\{g(\hat\theta)-g(\theta)\right\} \xrightarrow{d} N\!\left(0,\ \{g'(\theta)\}^2\sigma^2\right)
    $$

    **증명 개요.** $\hat\theta$ 근처에서 테일러 전개하면

    $$
    g(\hat\theta) = g(\theta) + g'(\theta)(\hat\theta-\theta) + O_p\!\left((\hat\theta-\theta)^2\right)
    $$

    이고 $\sqrt n$을 곱하면 나머지항이 $O_p(n^{-1/2}) \to 0$이므로 슬러츠키 정리로 결론이 따라 나온다.

    **MLE와 결합.** $\sigma^2 = 1/I_1(\theta)$이므로

    $$
    \sqrt n\left\{g(\hat\theta)-g(\theta)\right\} \xrightarrow{d} N\!\left(0,\ \frac{\{g'(\theta)\}^2}{I_1(\theta)}\right)
    $$

    이다. 이는 $g(\theta)$ 모수화에서의 크라메르-라오 하한과 정확히 같다. **MLE의 점근효율성은 변환에 대해 보존된다.**

    **확인.** 지수분포에서 $\hat\lambda$의 점근분산이 $\lambda^2/n$이고 $g(\lambda) = 1/\lambda$이므로 $g'(\lambda) = -1/\lambda^2$이다.

    $$
    \operatorname{Var}(\hat\mu) \approx \frac{1}{\lambda^4}\cdot\frac{\lambda^2}{n} = \frac{1}{n\lambda^2} = \frac{\mu^2}{n}
    $$

    앞서 직접 구한 $\operatorname{Var}(\bar X) = \mu^2/n$과 일치한다. ✓

    **주의할 점.** $g'(\theta) = 0$이면 1차항이 사라져 이 결과가 성립하지 않는다. 그때는 2차항이 주도해 극한분포가 정규가 아니라 카이제곱류가 된다. 예컨대 $\theta = 0$ 근처에서 $g(\theta)=\theta^2$의 분포가 그렇다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
모형이 **틀렸을 때**($\theta$ 값이 무엇이든 참 분포를 담지 못할 때) MLE는 무엇으로 수렴하는가? 표준오차는 어떻게 고쳐야 하는가?

</div>

??? success "풀이"
    **수렴 대상.** 참 분포를 $g$, 모형족을 $\{f(\cdot;\theta)\}$라 하자. 로그가능도의 평균이

    $$
    \frac1n\ell(\theta) \xrightarrow{p} E_g[\ln f(X;\theta)]
    $$

    이고, 이를 최대로 하는 $\theta$는

    $$
    \theta^* = \arg\max_\theta E_g[\ln f(X;\theta)] = \arg\min_\theta D_{\text{KL}}(g \,\|\, f_\theta)
    $$

    이다. 즉 MLE는 **참 분포에 쿨백-라이블러 발산 기준으로 가장 가까운 모형**으로 수렴한다. 이를 유사참값(pseudo-true value)이라 하며, 추정량을 **유사최대가능도추정량(QMLE)** 이라 부른다.

    **표준오차.** 모형이 맞으면 성립하던 정보량 등식

    $$
    A(\theta) := -E\left[\frac{\partial^2\ln f}{\partial\theta^2}\right] = E\left[\left(\frac{\partial\ln f}{\partial\theta}\right)^2\right] =: B(\theta)
    $$

    가 깨진다. 점근분포는

    $$
    \sqrt n(\hat\theta-\theta^*) \xrightarrow{d} N\!\left(0,\ A^{-1}BA^{-1}\right)
    $$

    이고, 이 $A^{-1}BA^{-1}$을 **샌드위치 분산** 또는 화이트의 강건 분산이라 한다. 모형이 맞으면 $A=B$가 되어 $A^{-1}$, 즉 보통의 $I^{-1}$로 돌아온다.

    **추정.**

    $$
    \hat A = -\frac1n\sum_i \frac{\partial^2\ln f(x_i;\hat\theta)}{\partial\theta^2}, \qquad \hat B = \frac1n\sum_i\left(\frac{\partial\ln f(x_i;\hat\theta)}{\partial\theta}\right)^2
    $$

    **실무.** 회귀에서 이분산이 있을 때 쓰는 강건 표준오차(HC0~HC3), 군집 표준오차, 일반화추정방정식의 표준오차가 모두 샌드위치 형태다. 계수 추정값은 그대로 두고 불확실성만 고쳐 주는 것이 특징이다.

    **주의.** 샌드위치 분산은 **표준오차만 고친다.** 점추정이 겨냥하는 $\theta^*$가 애초에 관심 모수가 아닐 수 있고, 우도비 검정도 더 이상 $\chi^2$를 따르지 않는다. "모형이 틀려도 강건 표준오차를 쓰면 된다"는 말은 지나친 단순화다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
MLE의 점근정규성을 점수함수의 테일러 전개로 유도하라. 어디서 큰수의 법칙이, 어디서 중심극한정리가 쓰이는가?

</div>

??? success "풀이"
    점수함수를 $U_n(\theta) = \ell'(\theta)$라 하자. $\hat\theta$가 내부 최댓값이면 $U_n(\hat\theta) = 0$이다. 참값 $\theta_0$ 둘레에서 전개하면

    $$
    0 = U_n(\hat\theta) = U_n(\theta_0) + U_n'(\tilde\theta)(\hat\theta-\theta_0)
    $$

    이고($\tilde\theta$는 $\theta_0$와 $\hat\theta$ 사이의 어떤 값), 정리하면

    $$
    \sqrt n(\hat\theta-\theta_0) = \frac{n^{-1/2}U_n(\theta_0)}{-n^{-1}U_n'(\tilde\theta)}
    $$

    **분자 — 중심극한정리.** $U_n(\theta_0) = \sum_i s(X_i;\theta_0)$는 독립인 점수의 합이다. 정칙 조건에서 $E[s] = 0$이고 $\operatorname{Var}(s) = I_1(\theta_0)$이므로

    $$
    \frac{1}{\sqrt n}U_n(\theta_0) \xrightarrow{d} N\!\left(0,\ I_1(\theta_0)\right)
    $$

    **분모 — 큰수의 법칙.** $-n^{-1}U_n'(\theta_0) = -n^{-1}\sum_i s'(X_i;\theta_0)$ 역시 독립인 항의 평균이므로

    $$
    -\frac1n U_n'(\theta_0) \xrightarrow{p} -E[s'] = I_1(\theta_0)
    $$

    이다. $\hat\theta$의 일치성($\hat\theta\xrightarrow{p}\theta_0$)에서 $\tilde\theta \xrightarrow{p}\theta_0$이고, 적당한 연속성 조건 아래 $-n^{-1}U_n'(\tilde\theta)$도 같은 극한을 갖는다.

    **결합 — 슬러츠키.** 분포수렴하는 분자를 확률수렴하는 상수로 나누면

    $$
    \sqrt n(\hat\theta-\theta_0) \xrightarrow{d} \frac{N(0, I_1)}{I_1} = N\!\left(0,\ \frac{1}{I_1(\theta_0)}\right)
    $$

    를 얻는다. $\square$

    **구조를 읽으면.** 점근분산의 분자에 있는 $I_1$은 **점수의 변동**(중심극한정리)에서 오고, 분모의 $I_1^2$은 **로그가능도의 곡률**(큰수의 법칙)에서 온다. 둘이 같은 $I_1$인 것이 우연이 아니라 정보량 등식이며, 앞 연습문제에서 본 대로 모형이 틀리면 이 일치가 깨져 샌드위치 형태가 남는다.

    **증명에 필요한 조건.** (1) $\hat\theta$가 내부에 있을 것(경계면 실패), (2) 세 번 미분 가능하고 3계 도함수가 유계일 것(테일러 전개의 나머지 통제), (3) 미분과 적분을 맞바꿀 수 있을 것(지지집합이 모수에 무관), (4) $\hat\theta$가 일치할 것, (5) $I_1(\theta_0)$이 유한하고 양수일 것.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
"점근적으로 정규"라는 말이 실무에서 얼마나 믿을 만한지 판단하려 한다. 어떤 요인들이 필요한 $n$을 좌우하는지 정리하고, 실제로 확인하는 방법을 적어라.

</div>

??? success "풀이"
    **필요한 $n$을 좌우하는 요인.**

    - **모수공간의 경계까지의 거리.** $\hat p$가 0이나 1에 가깝거나 분산성분이 0에 가까우면 아무리 $n$이 커도 정규근사가 나쁘다. 중요한 것은 $n$ 자체가 아니라 **$\hat\theta$가 경계에서 몇 표준오차 떨어져 있는가**다.
    - **유효 정보량.** 이항에서는 $n$이 아니라 $np(1-p)$가, 생존분석에서는 관측 수가 아니라 **사건 수**가, 포아송에서는 총 계수가 기준이다.
    - **모수화.** 로그가능도가 이차식에 얼마나 가까운지가 핵심이며, 이는 모수화에 따라 크게 달라진다. $\sigma$보다 $\ln\sigma$, $p$보다 로짓이 훨씬 빨리 정규에 다가간다.
    - **모수의 개수.** $p$가 $n$에 비해 크면 점근이론이 통하지 않는다. 대략 $n/p \ge 10$은 되어야 한다는 경험칙이 있다.
    - **성가신 모수의 수.** 관측 단위마다 모수가 하나씩 생기는 구조(네이만-스콧 문제)에서는 $n\to\infty$여도 일치성조차 없다.

    **확인하는 방법.**

    1. **로그가능도 곡선(또는 프로파일 곡선)을 그린다.** 포물선에서 얼마나 벗어나는지 눈으로 본다. 가장 값싸고 정보가 많은 진단이다.
    2. **왈드 구간과 우도비 구간을 둘 다 계산해 견준다.** 둘이 크게 다르면 이차 근사가 나쁘다는 직접적인 증거다.
    3. **부트스트랩한다.** $\hat\theta$의 부트스트랩 분포를 정규밀도와 겹쳐 그리거나 Q-Q 그림을 그린다.
    4. **모의실험한다.** 추정값을 참값으로 놓고 자료를 생성해 신뢰구간의 실제 포함확률을 세어 본다. 가장 확실한 방법이며, 계산 비용이 감당되면 언제나 권할 만하다.

    **결론.** "$n$이 충분히 크면 된다"는 말은 **구체적인 상황에서 확인해야 하는 주장**이지 자동으로 적용되는 보증이 아니다. 위 진단 중 하나만 해 보아도 대부분의 사고를 막을 수 있다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
MLE의 점근분산이 크라메르-라오 하한과 같다는 사실을 근거로 "MLE는 언제나 최선"이라고 말할 수 있는가? 반례를 들어 논하라.

</div>

??? success "풀이"
    **말할 수 없다.** 네 가지 유보가 있다.

    **(1) 점근적일 뿐이다.** 유한표본에서는 다른 추정량이 나을 수 있다. 정규분포의 $\hat\sigma^2_{\text{MLE}} = \frac1n\sum(x_i-\bar x)^2$은 편향되어 있고, 평균제곱오차 기준으로는 $\frac{1}{n+1}\sum(x_i-\bar x)^2$이 더 낫다. MLE도 불편추정량도 최적이 아니다.

    **(2) 불편추정량 안에서만의 최적이다.** 크라메르-라오는 불편추정량의 분산에 대한 하한이다. 편향을 허용하면 평균제곱오차가 더 작은 추정량이 있을 수 있다.

    가장 유명한 반례가 **제임스-스타인 추정량**이다. $\mathbf{X}\sim N_d(\boldsymbol\mu, I)$에서 $d \ge 3$이면

    $$
    \hat{\boldsymbol\mu}_{\text{JS}} = \left(1-\frac{d-2}{\|\mathbf{X}\|^2}\right)\mathbf{X}
    $$

    가 MLE $\mathbf{X}$를 **모든 $\boldsymbol\mu$에서** 평균제곱오차로 이긴다. MLE가 허용 불가능(inadmissible)하다는 뜻이다. 서로 무관해 보이는 여러 평균을 원점 쪽으로 함께 축소하는 것이 이득이라는 결과라 발표 당시 충격을 주었고, 오늘날 축소 추정과 계층모형의 출발점이 되었다.

    **(3) 정칙 조건이 필요하다.** 연습문제 5의 균등분포에서는 MLE의 수렴 속도가 $n$이라 크라메르-라오가 아예 적용되지 않고, 하한을 훨씬 넘어선다.

    **(4) 모형이 맞다는 전제가 있다.** 모형이 틀리면 MLE는 유사참값으로 수렴하며, 그 값이 관심 모수가 아닐 수 있다. 이상치가 있는 자료에서 정규 MLE(표본평균)는 강건 추정량보다 훨씬 나쁘다.

    **정리.** MLE가 널리 쓰이는 이유는 "최선"이어서가 아니라 **일반적이고, 자동이며, 정칙 조건 아래 점근적으로 효율적**이기 때문이다. 표본이 작거나, 모수가 많거나, 모형이 미덥지 않거나, 모수공간에 경계가 있으면 대안을 함께 살펴야 한다.

---

## 정리하며

정칙 조건 아래에서 최대가능도추정량은 세 가지 성질을 갖는다.

$$
\hat\theta_n \xrightarrow{P}\theta_0, \qquad
\sqrt n(\hat\theta_n-\theta_0)\xrightarrow{d} N\!\left(0,\;\frac{1}{I(\theta_0)}\right)
$$

- **일치성 · 점근정규성 · 점근 효율성.** 참값으로 가고, 정규분포로 가며, 그 분산이 크라메르–라오 하한과 같다. **점근적으로는 더 나은 추정량이 없다.**
- **이 셋이 최대가능도를 기본 전략으로 삼는 근거**이며, 가능도비 검정·왈드 검정·정보기준이 모두 여기서 나온다.
- **정칙 조건이 장식이 아니다.** 식별 가능성, $\theta$ 에 의존하지 않는 지지집합, 내부의 참값, 매끄러움, 양의 정보량. 이 중 하나만 깨져도 결론이 무너진다.
- **깨지는 실제 사례들.** $\text{Uniform}(0,\theta)$ 는 지지집합이 $\theta$ 에 의존해 수렴 속도가 $n^{-1}$ 로 더 빠르고 극한이 정규가 아니다. 분산 모수를 $0$ 근처에서 추정하면 참값이 경계에 놓여 표준 이론이 적용되지 않는다. 정규혼합의 가능도는 위로 유계가 아니다(3장 참조).
- **"점근적으로"가 "당신의 $n$ 에서"를 뜻하지는 않는다.** 소표본에서는 편향이 $O(1/n)$ 으로 남아 있으며, 정규분포의 $\hat\sigma^2$ 이 그 예다.

다음 절 **피셔 정보량과 표준오차**는 위 공식의 $I(\theta_0)$ 을 실제로 계산하고 표준오차로 바꾸는 법을 다룬다.
