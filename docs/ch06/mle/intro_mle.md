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

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
일간 로그수익률이 $r_t \sim N(\mu, \sigma^2)$이다. $n = 252$, $\bar r = 0.0004$, $s = 0.015$일 때 (a) $\mu_{\text{ann}} = 252\mu$와 $\sigma_{\text{ann}} = \sigma\sqrt{252}$의 MLE. (b) 점근정규성을 이용한 95% 신뢰구간. (c) $\hat\mu$가 $\hat\sigma$보다 훨씬 불안정한 이유는?

</div>

??? success "풀이"
    (a) $\hat\mu_{\text{ann}} = 252 \cdot 0.0004 = 0.1008$ (10.08%). $\hat\sigma_{\text{ann}} = 0.015 \sqrt{252} \approx 0.2381$ (23.81%).

    (b) $\mathrm{SE}(\hat\mu) = s/\sqrt n = 0.015/\sqrt{252} \approx 0.000945$. $\mathrm{SE}(\hat\mu_{\text{ann}}) = 252 \cdot 0.000945 \approx 0.238$. 95% 신뢰구간: $(-0.366, 0.568)$.

    $\sigma$에 대해서는 점근 표준오차가 $\sigma/\sqrt{2n}$이므로 $\mathrm{SE}(\hat\sigma_{\text{ann}}) \approx 0.0106$이다. 95% 신뢰구간: $(0.217, 0.259)$.

    (c) $\hat\mu_{\text{ann}}$의 표준오차 ≈ 0.238로 점추정값 0.101보다 *크다*. 신뢰구간이 $-37\%$에서 $+57\%$까지 걸쳐 있다. 변동성의 신뢰구간은 (24% 주위로 대략 ±2%포인트로) 좁다. **1년 시계에서 추세는 절망적으로 불안정하지만 변동성은 잘 추정된다.** 미래 수익률을 예측하는 데 과거 표본평균에 의존하는 것은 금융에서 고전적인 함정이다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
**정규분포의 MLE 유도.** i.i.d. $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$이 주어졌을 때 $\hat\mu$와 $\hat\sigma^2$을 유도하라.

</div>

??? success "풀이"
    로그가능도: $\ell(\mu, \sigma^2) = -(n/2)\ln(2\pi\sigma^2) - (1/(2\sigma^2))\sum(X_i - \mu)^2$.

    $\partial\ell/\partial\mu = 0 \Rightarrow \hat\mu = \bar X$.

    $\partial\ell/\partial\sigma^2 = -n/(2\sigma^2) + \sum(X_i - \mu)^2/(2\sigma^4) = 0 \Rightarrow \hat\sigma^2_{\text{MLE}} = (1/n)\sum(X_i - \bar X)^2$.

    MLE가 $n - 1$이 아니라 $n$으로 나눔에 유의하라. 편향되어 있다($\mathbb{E}[\hat\sigma^2] = (n-1)\sigma^2/n$). 불편추정을 하려면 Bessel 수정을 사용한다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
**Bernoulli/이항의 MLE.** 표본이 $X \sim \mathrm{Binomial}(n, p)$이다. $\hat p_{\text{MLE}}$를 유도하라.

</div>

??? success "풀이"
    가능도: $L(p) = \binom{n}{X} p^X (1-p)^{n-X} \propto p^X(1-p)^{n-X}$.

    로그가능도: $\ell(p) = X\ln p + (n-X)\ln(1-p)$.

    $\ell'(p) = X/p - (n-X)/(1-p) = 0$.

    $X(1-p) = (n-X)p \Rightarrow X = np \Rightarrow \hat p = X/n$.

    MLE는 표본비율이다. 불편이다: $\mathbb{E}[\hat p] = p$.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
**가능도와 확률.** MLE의 용어에서 "가능도"와 "확률"을 구별하라. 가능도가 $\theta$의 함수이면서도 $\theta$에 대한 확률밀도가 아닌 이유는?

</div>

??? success "풀이"
    **확률:** $P(X = x \mid \theta)$ — $\theta$를 고정한 $x$의 함수이다. $x$에 대해 합하거나 적분하면 1이 된다.

    **가능도:** $L(\theta) = P(X = x \mid \theta)$ — 수식은 같지만 *관측되어 고정된* $x$에 대해 $\theta$의 함수로 본 것이다. $\theta$에 대해 적분해도 1이 되지 않는다.

    **왜 $\theta$에 대한 밀도가 아닌가:** 빈도주의 통계학에서 $\theta$는 확률변수가 아니라 *모수*이다. 가능도는 $\theta$ 값들을 자료를 얼마나 잘 설명하는지로 순위 매길 뿐, 그것이 얼마나 확률적으로 있음 직한지로 매기지 않는다. MLE는 가능도를 최대화하는 $\theta$를 고르지만 이것이 "가장 확률이 높은 모수"인 것은 아니다.

    베이즈 추론에서는 $\theta$가 확률변수가 되고 *사후분포* $\pi(\theta \mid x) \propto L(\theta) \pi(\theta)$를 계산하는데, 이것은 $\theta$에 대한 밀도가 **맞다**. 가능도 자체는 그대로이고 달라지는 것은 틀이다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
**MLE의 점근정규성.** 그 결과를 서술하고 베르누이의 경우에 확인하라.

</div>

??? success "풀이"
    **MLE의 점근정규성:**

    $$
    \sqrt n(\hat\theta_{\text{MLE}} - \theta) \xrightarrow{d} N(0, 1/I(\theta))
    $$

    여기서 $I(\theta) = -\mathbb{E}[\partial^2 \log f/\partial\theta^2]$는 관측값 하나당 Fisher 정보량이다.

    **베르누이에서의 확인:** $\log f = x\log p + (1-x)\log(1-p)$. 2계도함수는 $-x/p^2 - (1-x)/(1-p)^2$이다. 기댓값을 취하면 $-p/p^2 - (1-p)/(1-p)^2 = -1/p - 1/(1-p) = -1/[p(1-p)]$.

    따라서 $I(p) = 1/[p(1-p)]$이고 $\mathrm{Var}(\hat p) = p(1-p)/n$이다. 표본비율에 대한 중심극한정리와 곧바로 일치한다.

    함의: MLE는 점근적으로 **Cramér-Rao 하한**을 달성하며 점근적으로 효율적이다. 어떤 불편추정량도 이보다 작은 점근분산을 가질 수 없다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
**MLE가 실패하는 경우.** 다음 각각의 예를 하나씩 들라: (a) MLE가 존재하지 않는 경우; (b) MLE가 경계에 있는 경우; (c) MLE가 일치하지 않는 경우.

</div>

??? success "풀이"
    **(a) MLE가 존재하지 않는 경우:** $X \sim \mathrm{Uniform}(0, \theta)$에서 가능도는 $\theta \ge \max X_i$일 때 $1/\theta^n$이고 그 아래에서는 0이다. $\theta \to \infty$이면 $L \to 0$이고, $\theta \to \max X_i$이면 $L \to (1/\max X_i)^n$으로 *상한*에 이르지만 그 지점이 경계이다. 엄밀히 말해 내부에 최댓값이 없으며, MLE는 경계값 $\hat\theta = \max X_i$이다.

    **(b) MLE가 경계에 있는 경우:** 위의 균등분포 예이다. 경계에 있는 MLE는 비표준 점근분포를 가지며($N(0, 1/I)$가 아니다) 수렴 속도가 $\sqrt n$이 아니라 $n$이다.

    **(c) MLE가 일치하지 않는 경우:** Neyman-Scott 문제이다. $i = 1, \ldots, n$, $j = 1, 2$에 대해 관측값이 $X_{ij} \sim N(\mu_i, \sigma^2)$이다. 모수의 개수($\mu_i$들 + $\sigma^2$)가 표본크기와 함께 늘어난다. $\hat\sigma^2_{\text{MLE}} \to \sigma^2/2$로 $\sigma^2$이 아니다. 모수공간의 차원이 $n$에 비례해 커지므로 일치하지 않는다.

    이런 실패 양상들이 편향 보정 MLE, 벌점 가능도, 프로파일 가능도, (베이즈의) 주변가능도 같은 개선을 이끌어 냈다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
**불변성**을 정확히 서술하고, $g$가 일대일이 아닐 때도 성립하도록 하려면 정의를 어떻게 확장해야 하는지 설명하라. 실무에서 이 성질이 왜 유용한가?

</div>

??? success "풀이"
    **일대일인 경우.** $\eta = g(\theta)$로 모수화를 바꾸면 가능도가 $L^*(\eta) = L(g^{-1}(\eta))$이므로, $L$이 $\hat\theta$에서 최대이면 $L^*$는 $g(\hat\theta)$에서 최대다. 따라서

    $$
    \hat\eta = g(\hat\theta)
    $$

    **일대일이 아닌 경우.** $g$가 예를 들어 $\theta^2$처럼 여러 $\theta$를 같은 $\eta$로 보내면 $L^*$를 그대로 정의할 수 없다. **유도가능도**로 확장한다.

    $$
    L^*(\eta) = \sup_{\theta:\ g(\theta)=\eta} L(\theta)
    $$

    이렇게 두면 $L^*$를 최대로 하는 $\eta$가 여전히 $g(\hat\theta)$이므로 불변성이 유지된다.

    **왜 유용한가.**

    - **다시 최적화할 필요가 없다.** $\lambda$의 MLE를 구했으면 평균 $1/\lambda$, 중앙값 $\ln2/\lambda$, 생존확률 $e^{-\lambda t}$의 MLE가 모두 대입만으로 나온다.
    - **모수화를 자유롭게 고를 수 있다.** 수치 최적화를 하기 좋은 척도(경계가 없는 척도)에서 풀고 원하는 척도로 되돌려도 답이 같다. $\sigma$ 대신 $\ln\sigma$, $p$ 대신 로짓을 쓰는 것이 표준적인 관행인 이유다.
    - **보고 단위를 바꿔도 결론이 같다.** 위험비로 보고하든 로그위험비로 보고하든 추정값이 일관된다.

    **다만 불편성은 물려받지 못한다.** 앞서 본 대로 $E[g(\hat\theta)] \ne g(E[\hat\theta])$이며, 표준오차도 단순히 변환되지 않아 델타 방법이 필요하다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
닫힌 해가 없는 모형에서 MLE를 구하는 **뉴턴-랩슨**과 **피셔 점수법**을 비교하라. 각각의 갱신식과 장단점을 적어라.

</div>

??? success "풀이"
    점수함수 $U(\boldsymbol\theta) = \partial\ell/\partial\boldsymbol\theta$를 0으로 만드는 것이 목표다. $\hat{\boldsymbol\theta}$ 근처에서 테일러 전개하면 갱신식이 나온다.

    **뉴턴-랩슨.** **관측정보량**(헤시안의 음수)을 쓴다.

    $$
    \boldsymbol\theta^{(k+1)} = \boldsymbol\theta^{(k)} + \left\{-\frac{\partial^2\ell}{\partial\boldsymbol\theta\partial\boldsymbol\theta^\top}\right\}^{-1}U(\boldsymbol\theta^{(k)})
    $$

    **피셔 점수법.** 관측정보량을 그 기대값인 **피셔 정보량**으로 바꾼다.

    $$
    \boldsymbol\theta^{(k+1)} = \boldsymbol\theta^{(k)} + I(\boldsymbol\theta^{(k)})^{-1}U(\boldsymbol\theta^{(k)})
    $$

    **비교.**

    | | 뉴턴-랩슨 | 피셔 점수법 |
    |---|---|---|
    | 수렴 속도 | 이차(가까이서 매우 빠름) | 대체로 일차~이차 |
    | 안정성 | 헤시안이 양정부호가 아니면 발산 가능 | $I$가 언제나 양반정부호라 안정적 |
    | 계산 | 2계 도함수 필요 | 기대값을 미리 구해 두면 단순해지는 경우 많음 |
    | 정준연결 GLM | — | **둘이 완전히 일치** |

    피셔 점수법이 더 안정적인 이유는 $I(\boldsymbol\theta) = E[-\partial^2\ell/\partial\boldsymbol\theta^2]$가 언제나 양반정부호여서 갱신 방향이 반드시 오르막이기 때문이다. 관측 헤시안은 봉우리에서 멀 때 음정부호가 아닐 수 있어 엉뚱한 방향으로 뛴다.

    **실무.** 일반화선형모형에서 피셔 점수법을 정리하면 **반복 가중최소제곱(IRLS)** 이 된다. 각 단계가 가중최소제곱 문제라 기존 선형대수 코드를 그대로 쓸 수 있고, 이것이 GLM 소프트웨어의 표준 구현이다.

    **표준오차에는 어느 정보량을 쓸까.** 관측정보량 $-\ell''(\hat\theta)$을 쓰는 것이 일반적으로 권장된다. 자료가 실제로 담고 있는 곡률을 반영하고, 모형이 조금 틀렸을 때도 더 나은 근사를 준다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
자료 일부가 결측이거나 잠재변수가 있을 때 쓰는 **EM 알고리즘**의 두 단계를 설명하고, 왜 각 반복에서 가능도가 줄지 않는지 밝혀라.

</div>

??? success "풀이"
    관측자료를 $\mathbf{y}$, 결측(또는 잠재)자료를 $\mathbf{z}$라 하자. 완전자료 로그가능도 $\ell_c(\boldsymbol\theta;\mathbf{y},\mathbf{z})$는 다루기 쉬운데 관측자료 가능도

    $$
    \ell(\boldsymbol\theta;\mathbf{y}) = \ln\int f(\mathbf{y},\mathbf{z};\boldsymbol\theta)\,d\mathbf{z}
    $$

    는 적분 때문에 다루기 어렵다.

    **E 단계.** 현재 추정값 $\boldsymbol\theta^{(k)}$ 아래에서 완전자료 로그가능도의 조건부 기대값을 구한다.

    $$
    Q(\boldsymbol\theta \mid \boldsymbol\theta^{(k)}) = E_{\mathbf{z}\mid\mathbf{y},\,\boldsymbol\theta^{(k)}}\left[\ell_c(\boldsymbol\theta;\mathbf{y},\mathbf{z})\right]
    $$

    **M 단계.** $Q$를 최대로 하는 $\boldsymbol\theta^{(k+1)}$을 구한다.

    **왜 가능도가 줄지 않는가.** 다음 분해가 열쇠다.

    $$
    \ell(\boldsymbol\theta) = Q(\boldsymbol\theta\mid\boldsymbol\theta^{(k)}) - H(\boldsymbol\theta\mid\boldsymbol\theta^{(k)}), \qquad H(\boldsymbol\theta\mid\boldsymbol\theta^{(k)}) = E\left[\ln f(\mathbf{z}\mid\mathbf{y};\boldsymbol\theta)\right]
    $$

    젠슨 부등식(또는 KL 발산의 비음수성)에서 모든 $\boldsymbol\theta$에 대해

    $$
    H(\boldsymbol\theta\mid\boldsymbol\theta^{(k)}) \le H(\boldsymbol\theta^{(k)}\mid\boldsymbol\theta^{(k)})
    $$

    이다. M 단계가 $Q(\boldsymbol\theta^{(k+1)}) \ge Q(\boldsymbol\theta^{(k)})$를 보장하므로

    $$
    \ell(\boldsymbol\theta^{(k+1)}) - \ell(\boldsymbol\theta^{(k)}) = \underbrace{\left\{Q^{(k+1)}-Q^{(k)}\right\}}_{\ge 0} + \underbrace{\left\{H^{(k)}-H^{(k+1)}\right\}}_{\ge 0} \ge 0
    $$

    이다. $\square$

    **성질.** 단조증가가 보장되어 대단히 안정적이지만 수렴이 **일차**라 뉴턴법보다 느리다. 또 전역 최대를 보장하지 않으므로(정규혼합처럼 봉우리가 여럿인 경우) 여러 초기값에서 돌려 봐야 한다. 표준오차를 곧바로 주지 않는다는 점도 단점이라, 별도의 정보량 계산이나 부트스트랩이 필요하다.

    **쓰임.** 혼합모형, 은닉 마르코프 모형, 결측자료, 요인분석, 중도절단 자료, $t$ 잡음 회귀가 모두 EM으로 적합된다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
최대 로그가능도 $\ell(\hat{\boldsymbol\theta})$는 모수를 늘릴수록 반드시 커진다. 그런데도 모형 선택에 가능도를 쓸 수 있는 이유를 **AIC**로 설명하고, BIC와 비교하라.

</div>

??? success "풀이"
    **문제.** 모형 A가 모형 B에 내포되면 $\ell_A \ge \ell_B$가 언제나 성립한다. 더 넓은 모수공간에서 최대화하기 때문이다. 따라서 최대 로그가능도만 비교하면 **언제나 가장 복잡한 모형이 이긴다.**

    **AIC.** 아카이케는 모수 개수 $p$로 벌점을 준다.

    $$
    \text{AIC} = -2\ell(\hat{\boldsymbol\theta}) + 2p
    $$

    작을수록 좋다. 벌점 $2p$의 근거는 이렇다. 같은 자료로 적합하고 평가하면 로그가능도가 낙관적으로 나오는데, 그 낙관의 크기(기대 과적합량)가 근사적으로 $p$이고, $-2$배를 하면 $2p$가 된다.

    AIC가 겨냥하는 것은 **미래 자료에 대한 예측 성능**, 정확히는 참 분포와 적합된 모형 사이의 쿨백-라이블러 발산이다.

    **BIC.**

    $$
    \text{BIC} = -2\ell(\hat{\boldsymbol\theta}) + p\ln n
    $$

    $n \ge 8$이면 $\ln n > 2$이므로 **AIC보다 강하게 벌점**하고, 더 단순한 모형을 고른다. 근거도 다르다. BIC는 베이즈 주변가능도의 근사이며 **참 모형을 찾는 것**을 겨냥한다.

    **비교.**

    | | AIC | BIC |
    |---|---|---|
    | 벌점 | $2p$ | $p\ln n$ |
    | 목표 | 예측 정확도 | 참 모형 식별 |
    | 참 모형이 후보에 있을 때 | 일치성 없음(과적합 경향) | 일치성 있음 |
    | 참 모형이 후보에 없을 때 | 최선의 근사 모형을 고름 | 지나치게 단순한 모형을 고를 수 있음 |

    **실무.** 예측이 목적이면 AIC, 설명이나 변수 선택이 목적이면 BIC가 흔히 권장된다. 다만 어느 쪽도 만능이 아니며, 예측이 진짜 목적이라면 **교차검증**이 더 직접적이다. 실제로 일탈 하나 빼기 교차검증이 AIC와 점근적으로 동등하다는 결과가 있다.

    **주의할 점.** AIC나 BIC 값은 **같은 자료, 같은 반응변수**에 적합한 모형끼리만 비교할 수 있다. 반응변수를 변환하면($y$ 대 $\ln y$) 가능도의 척도가 달라져 비교가 무의미해진다. 야코비안을 보정하면 비교할 수 있지만 흔히 놓치는 함정이다.

---

## 정리하며

MLE는 모수 추정에 대한 원리적이고 범용적인 접근을 제공한다. 다음과 자연스럽게 연결된다:

- 기계학습의 **비용함수** (분류에서 교차엔트로피 손실 = 음의 로그가능도)
- **베이즈 추론** (MLE는 평평한 사전분포를 쓴 MAP 추정값이다)
- **정보이론** (모형과 자료 사이의 KL 발산 최소화)
