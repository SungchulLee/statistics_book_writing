# 지수분포의 MLE

## 동기

지수분포는 포아송 과정에서 연속한 사건 사이의 대기 시간을 모형화한다. 예를 들어 고객 도착 사이의 시간, 부품 고장 사이의 시간, 방사성 붕괴 사이의 시간이 그렇다. 관측된 대기 시간으로부터 비율 모수 $\lambda$를 추정하는 것은 기본적인 응용 문제이다. 지수의 경우 깔끔한 닫힌 형태의 MLE가 나오며, 동시에 비선형 변환이 추정량에 어떻게 편향을 들여오는지도 보여 준다.

## 설정

비율 모수가 $\lambda > 0$인 지수분포에서 독립적으로 뽑은 확률표본 $X_1, X_2, \ldots, X_n$을 생각하자. 각 관측값의 밀도는

$$
f(x; \lambda) = \lambda e^{-\lambda x}, \quad x > 0
$$

목표는 관측된 자료를 가장 그럴듯하게 만드는 $\lambda$ 값을 찾는 것이다.

## 유도

관측값이 독립이므로 결합밀도는 개별 밀도의 곱이다. 로그를 취하면 이 곱이 합으로 바뀌어 로그가능도함수를 얻는다:

$$
\ell(\lambda) = \sum_{i=1}^n \log f(x_i; \lambda) = n \log \lambda - \lambda \sum_{i=1}^n x_i
$$

로그가능도를 최대화하는 $\lambda$ 값을 찾기 위해 $\lambda$에 대해 미분하고 결과를 0으로 둔다:

$$
\frac{d\ell}{d\lambda} = \frac{n}{\lambda} - \sum_{i=1}^n x_i = 0
$$

$\lambda$에 대해 풀면 최대가능도추정량을 얻는다:

$$
\hat{\lambda}_{\text{MLE}} = \frac{n}{\sum_{i=1}^n x_i} = \frac{1}{\bar{X}}
$$

이 임계점이 실제로 최댓값임을 확인하기 위해 2계도함수를 살펴본다:

$$
\frac{d^2\ell}{d\lambda^2} = -\frac{n}{\lambda^2} < 0 \quad \text{for all } \lambda > 0
$$

2계도함수가 모수공간 전체에서 엄격하게 음수이므로 임계점 $\hat{\lambda}_{\text{MLE}} = 1/\bar{X}$는 로그가능도의 전역 최댓값이다.

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> 고객 대기 시간. 고객 도착 사이의 대기 시간(분)을 다섯 번 관측하여 $x_1 = 2.1$, $x_2 = 0.8$, $x_3 = 1.5$, $x_4 = 3.2$, $x_5 = 1.4$를 얻었다고 하자. 표본평균은

</div>

??? success "풀이"
    $$
    \bar{x} = \frac{2.1 + 0.8 + 1.5 + 3.2 + 1.4}{5} = 1.8 \text{ minutes}
    $$

    따라서 비율 모수의 MLE는

    $$
    \hat{\lambda} = \frac{1}{\bar{x}} = \frac{1}{1.8} \approx 0.556 \text{ arrivals per minute}
    $$

## 성질

MLE를 손에 넣었으니 이제 그 통계적 성질을 살펴보자. 평균적으로 참 모수를 맞히는지, 그리고 표본크기가 커질 때 $\lambda$를 얼마나 정밀하게 추정하는지를 본다.

**편향.** MLE $\hat{\lambda} = 1/\bar{X}$는 위쪽으로 편향되어 있다. 함수 $g(x) = 1/x$가 $(0, \infty)$에서 볼록하므로 Jensen 부등식에 의해

$$
E\!\left[\frac{1}{\bar{X}}\right] > \frac{1}{E[\bar{X}]} = \frac{1}{1/\lambda} = \lambda
$$

따라서 MLE는 평균적으로 $\lambda$를 과대추정한다. 다만 이 편향은 $n \to \infty$일 때 사라지며 추정량은 일치한다.

**Fisher 정보량.** 관측값 하나당 Fisher 정보량은

$$
I_1(\lambda) = -E\!\left[\frac{d^2 \log f(X;\lambda)}{d\lambda^2}\right] = -E\!\left[-\frac{1}{\lambda^2}\right] = \frac{1}{\lambda^2}
$$

크기 $n$인 표본에서 전체 Fisher 정보량은 $I_n(\lambda) = n / \lambda^2$이다.

**점근분산.** MLE의 일반적인 점근이론에 의해 $\hat{\lambda}$의 분산은 근사적으로

$$
\text{Var}(\hat{\lambda}) \approx \frac{1}{I_n(\lambda)} = \frac{\lambda^2}{n}
$$

표본크기가 커질수록 추정량이 정밀해지며 표준오차는 $\lambda / \sqrt{n}$에 비례한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
($x > 0$에서 밀도가 $f(x;\lambda) = \lambda e^{-\lambda x}$인) $\text{Exp}(\lambda)$에서 얻은 i.i.d. 표본 $x_1, \dots, x_n$에 대해 $\lambda$의 MLE를 유도하라.

</div>

??? success "풀이"
    로그가능도는:

    $$
    \ell(\lambda) = \sum_{i=1}^n \bigl(\log\lambda - \lambda x_i\bigr) = n\log\lambda - \lambda\sum_{i=1}^n x_i
    $$

    도함수를 0으로 두면:

    $$
    \frac{d\ell}{d\lambda} = \frac{n}{\lambda} - \sum_{i=1}^n x_i = 0 \implies \hat{\lambda} = \frac{n}{\sum_{i=1}^n x_i} = \frac{1}{\bar{X}}
    $$

    2계도함수가 $-n/\lambda^2 < 0$이므로 최댓값임이 확인된다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
어떤 콜센터에 걸려 오는 전화의 도착 간 시간이 지수분포를 따른다. 도착 간 시간 50개의 표본에서 $\bar{x} = 4.2$분이었다. 비율 $\lambda$의 MLE를 구하고 점근정규성을 이용해 근사적인 95% 신뢰구간을 구성하라.

</div>

??? success "풀이"
    MLE는 $\hat{\lambda} = 1/\bar{x} = 1/4.2 \approx 0.2381$(분당 통화 수)이다.

    점근분산은 $\hat{\lambda}^2/n = 0.2381^2/50 = 0.001133$이므로 $\text{SE}(\hat{\lambda}) = \sqrt{0.001133} \approx 0.03367$이다.

    95% 신뢰구간은:

    $$
    \hat{\lambda} \pm 1.96 \times \text{SE}(\hat{\lambda}) = 0.2381 \pm 0.0660 = (0.172, 0.304)
    $$

    평균 도착 간 시간 $1/\lambda$로 나타내면 MLE는 4.2분이고, 불변성에 의해 $1/\lambda$의 구간은 근사적으로 $(1/0.304, 1/0.172) = (3.29, 5.81)$분이다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
MLE $\hat{\lambda} = 1/\bar{X}$가 유한표본에서 $\lambda$에 대해 편향되어 있음을 보여라. $n = 2$일 때의 정확한 편향을 계산하라.

</div>

??? success "풀이"
    합 $S = \sum X_i \sim \text{Gamma}(n, \lambda)$이므로 $\bar{X} = S/n$이고 MLE는 $\hat{\lambda} = n/S$이다.

    Gamma$(n, \lambda)$ 분포에서 $n > 1$일 때 $E[1/S] = \lambda/(n-1)$이다(감마 확률변수의 역수에 관한 알려진 결과이다). 따라서:

    $$
    E[\hat{\lambda}] = E\!\left[\frac{n}{S}\right] = \frac{n\lambda}{n-1}
    $$

    편향은:

    $$
    \text{Bias}(\hat{\lambda}) = \frac{n\lambda}{n-1} - \lambda = \frac{\lambda}{n-1}
    $$

    $n = 2$일 때 $\text{Bias} = \lambda/(2-1) = \lambda$이다. MLE는 $\lambda$를 $n/(n-1)$배만큼 과대추정한다. 편향 보정된 추정량은 $\tilde{\lambda} = (n-1)/\sum X_i$이다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
MLE $\hat{\lambda} = 1/\bar{X}$를 $\lambda$의 적률법 추정량과 비교하라. 둘은 같은가?

</div>

??? success "풀이"
    지수분포에서 $E[X] = 1/\lambda$이다. 적률법은 $\bar{X} = 1/\hat{\lambda}$로 두므로 $\hat{\lambda}_{\text{MOM}} = 1/\bar{X}$를 준다.

    이는 MLE와 동일하다. 지수분포가 단일모수 지수족이기 때문에 생기는 일치이다. 그런 족에서는 점수방정식 $\partial\ell/\partial\lambda = 0$과 1차 적률방정식 $\bar{X} = E_\lambda[X]$가 같은 추정량을 준다.

    두 추정량 모두 유한표본 편향 $\lambda/(n-1)$과 점근분산 $\lambda^2/n$을 공유한다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
평균수명 $\mu = 1/\lambda$의 MLE를 구하고, 그것이 $\lambda$의 MLE와 달리 **불편**임을 보여라. 두 결과가 모순이 아닌 이유는 무엇인가?

</div>

??? success "풀이"
    **MLE.** 불변성에 따라

    $$
    \hat\mu = \frac{1}{\hat\lambda} = \bar X
    $$

    이다. 따로 최적화할 필요가 없다.

    **불편성.** $E[X_i] = 1/\lambda = \mu$이므로

    $$
    E[\hat\mu] = E[\bar X] = \mu
    $$

    로 정확히 불편이다.

    **모순이 아닌 이유.** 불변성은 **MLE에 대한 성질**이지 불편성에 대한 성질이 아니다. $g$가 비선형이면 $E[g(\hat\theta)] \ne g(E[\hat\theta])$이므로, 한 모수화에서 불편이어도 다른 모수화에서는 아니다.

    여기서 $g(x) = 1/x$는 볼록함수이므로 옌센 부등식에서

    $$
    E[\hat\lambda] = E\!\left[\frac{1}{\bar X}\right] > \frac{1}{E[\bar X]} = \lambda
    $$

    로 $\hat\lambda$가 위로 치우친다. 실제로 연습문제 3에서 $E[\hat\lambda] = \frac{n}{n-1}\lambda$였다.

    **교훈.** "MLE는 편향되어 있다"는 말은 **어느 모수화에서인지**를 밝혀야 뜻이 있다. 같은 모형을 비율로 적느냐 평균으로 적느냐에 따라 답이 달라진다. 이 점에서 불편성은 다소 자의적인 기준이며, 모수화에 불변인 불변성·일치성·점근효율성이 더 근본적인 성질로 여겨지는 이유다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
지수분포의 피셔 정보량 $I_1(\lambda)$을 구하고, MLE가 크라메르-라오 하한을 점근적으로 달성함을 확인하라. $\mu = 1/\lambda$ 모수화에서는 어떻게 되는가?

</div>

??? success "풀이"
    **$\lambda$ 모수화.** $\ln f(x;\lambda) = \ln\lambda - \lambda x$이므로

    $$
    \frac{\partial\ln f}{\partial\lambda} = \frac1\lambda - x, \qquad \frac{\partial^2\ln f}{\partial\lambda^2} = -\frac{1}{\lambda^2}
    $$

    이고

    $$
    I_1(\lambda) = -E\left[-\frac{1}{\lambda^2}\right] = \frac{1}{\lambda^2}
    $$

    이다. 2계도함수가 $x$에 의존하지 않아 기대값을 취할 것도 없다.

    크라메르-라오 하한은 $1/\{nI_1(\lambda)\} = \lambda^2/n$이고, MLE의 점근분산도 $\lambda^2/n$이므로 **점근적으로 하한을 달성**한다. 유한표본에서는 편향이 있어 정확히 하한에 닿지는 않는다.

    **$\mu$ 모수화.** 정보량은 모수화에 따라 변환된다. $\lambda = 1/\mu$이고 $d\lambda/d\mu = -1/\mu^2$이므로

    $$
    I_1(\mu) = I_1(\lambda)\left(\frac{d\lambda}{d\mu}\right)^2 = \mu^2\cdot\frac{1}{\mu^4} = \frac{1}{\mu^2}
    $$

    이다. 하한이 $\mu^2/n$이고 실제로 $\operatorname{Var}(\bar X) = \operatorname{Var}(X)/n = \mu^2/n$이므로, **$\mu$ 모수화에서는 유한표본에서도 하한을 정확히 달성**한다.

    두 모수화 모두 $I_1 = 1/(\text{모수})^2$ 꼴이 되는 것은 우연이 아니다. 지수분포에서 $\lambda$와 $\mu$가 모두 척도 모수이고, **척도 모수의 정보량은 언제나 모수의 제곱에 반비례**한다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
연습문제 2의 자료($n=50$, $\bar x = 4.2$분)로 $H_0: \lambda = 0.2$(평균 5분)를 우도비 검정으로 판정하고, 우도비 기반 신뢰구간을 구하라. 왈드 구간과 어떻게 다른가?

</div>

??? success "풀이"
    $\hat\lambda = 1/4.2 = 0.2381$이다.

    **우도비 통계량.** $\ell(\lambda) = n\ln\lambda - \lambda\sum x_i = n\{\ln\lambda - \lambda\bar x\}$이므로

    $$
    \Lambda = 2n\left\{\ln\frac{\hat\lambda}{\lambda_0} - (\hat\lambda-\lambda_0)\bar x\right\} = 2n\left\{\ln\frac{\hat\lambda}{\lambda_0} - \frac{\hat\lambda-\lambda_0}{\hat\lambda}\right\}
    $$

    ($\bar x = 1/\hat\lambda$를 썼다.) 값을 넣으면

    $$
    \Lambda = 100\left\{\ln\frac{0.2381}{0.2} - \frac{0.0381}{0.2381}\right\} = 100\{0.1744-0.1600\} = 1.44
    $$

    $\chi^2_1$에서 $p\text{-값} = 0.230$으로 기각하지 못한다.

    **우도비 구간.** $\Lambda(\lambda) \le 3.841$인 $\lambda$를 모으면 수치적으로

    $$
    (0.178,\ 0.310)
    $$

    을 얻는다.

    **왈드 구간.** $\operatorname{SE}(\hat\lambda) \approx \hat\lambda/\sqrt n = 0.2381/7.071 = 0.03367$이므로

    $$
    0.2381 \pm 1.96(0.03367) = (0.172,\ 0.304)
    $$

    **차이.** 왈드 구간은 $\hat\lambda$를 중심으로 대칭이지만 우도비 구간은 위로 조금 더 길다($-0.060$, $+0.072$). 로그가능도가 $\lambda$의 대칭함수가 아니기 때문이다. $n=50$이라 차이가 크지 않지만, $n$이 작으면 뚜렷해진다.

    **어느 쪽이 나은가.** 우도비 구간이다. 모수화에 불변이고($\mu$ 척도에서 만들어 되돌려도 같다), 로그가능도의 비대칭을 그대로 반영하며, 포함확률이 명목값에 더 가깝다. 다만 수치적으로 풀어야 해서 왈드보다 계산이 번거롭다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
수명 자료가 정말 지수분포를 따르는지 확인하려 한다. 와이불 모형 $S(t) = \exp\{-(t/\theta)^k\}$에 포함시켜 $H_0: k=1$을 검정하는 방법을 설명하라. 자유도는 얼마인가?

</div>

??? success "풀이"
    **내포 관계.** 와이불에서 $k=1$이면

    $$
    S(t) = e^{-t/\theta}
    $$

    로 정확히 지수분포다(비율 $\lambda = 1/\theta$). 즉 **지수분포는 와이불의 부분모형**이다.

    **우도비 검정.**

    1. 와이불 모형을 적합해 $(\hat k, \hat\theta)$와 최대 로그가능도 $\ell_1$을 얻는다.
    2. $k=1$로 고정한 지수 모형을 적합해 $\ell_0$를 얻는다. 이때 $\hat\theta = \bar x$이다.
    3. $\Lambda = 2(\ell_1-\ell_0)$를 계산한다.

    모수 개수가 2와 1이므로 **자유도는 1**이고 $\Lambda \sim \chi^2_1$이다. $\Lambda > 3.841$이면 지수 가정을 기각한다.

    **다른 방법들.**

    - **왈드 검정**: $(\hat k-1)/\operatorname{SE}(\hat k)$. 계산은 쉽지만 $k$의 로그가능도가 비대칭이라 $\ln k$ 척도에서 하는 편이 낫다.
    - **그림 진단**: 앞서 본 와이불 확률지에서 $\ln\{-\ln \hat S(t)\}$를 $\ln t$에 대해 그린다. 지수분포이면 **기울기가 1인 직선**이 되므로, 기울기가 1에서 얼마나 벗어나는지를 눈으로 본다.
    - **위험함수 진단**: 지수분포는 위험이 상수다. 시간 구간별로 위험을 추정해 평평한지 본다.

    **주의.** $H_0$를 기각하지 못했다고 지수 모형이 옳은 것은 아니다. 검정력이 낮으면 $k=1.3$ 같은 실질적 차이도 놓친다. 그리고 와이불에 포함되지 않는 다른 이탈(욕조곡선, 혼합분포)은 이 검정으로 전혀 잡히지 않는다. **모형 검정은 특정 대안에 대한 것일 뿐 "모형이 옳다"를 보증하지 않는다.**

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
$n$개 중 $d$개만 고장 나고 나머지는 시각 $C$에 중도절단된 자료에서 $\lambda$의 MLE와 그 관측정보량을 구하라. 신뢰구간을 만들 때 $d$가 작으면 무엇을 조심해야 하는가?

</div>

??? success "풀이"
    **가능도.** 고장은 밀도로, 중도절단은 생존확률로 들어간다. $t_i$를 관측시간, $\delta_i$를 고장 지시자라 하면

    $$
    L(\lambda) = \prod_i \left(\lambda e^{-\lambda t_i}\right)^{\delta_i}\left(e^{-\lambda t_i}\right)^{1-\delta_i} = \lambda^d\exp\!\left(-\lambda T\right), \qquad T = \sum_i t_i
    $$

    이다($d = \sum\delta_i$는 고장 수, $T$는 총 노출시간).

    **MLE.**

    $$
    \ell(\lambda) = d\ln\lambda - \lambda T, \qquad \ell'(\lambda) = \frac d\lambda - T = 0 \implies \hat\lambda = \frac{d}{T}
    $$

    **관측정보량.**

    $$
    -\ell''(\lambda) = \frac{d}{\lambda^2} \implies I_{\text{obs}}(\hat\lambda) = \frac{d}{\hat\lambda^2} = \frac{T^2}{d}
    $$

    이므로

    $$
    \operatorname{SE}(\hat\lambda) \approx \frac{\hat\lambda}{\sqrt d}
    $$

    이다.

    **핵심 관찰.** 표준오차가 $n$이 아니라 **$d$(고장 수)** 에만 의존한다. 부품 1000개를 시험해도 고장이 5개뿐이면 정밀도는 고장 5개만큼이다. 시험 기간을 늘려 $T$를 키워도 $\hat\lambda$가 작아질 뿐 상대 정밀도는 나아지지 않는다.

    **$d$가 작을 때 조심할 것.**

    - **왈드 구간이 무너진다.** $d < 10$이면 로그가능도가 심하게 비대칭이라 대칭 구간의 포함확률이 크게 어긋난다. $d=0$이면 $\hat\lambda = 0$이고 표준오차도 0이 되어 아예 쓸 수 없다.
    - **정확 구간을 쓴다.** $2\lambda T \sim \chi^2_{2d}$라는 결과를 이용해

      $$
      \left(\frac{\chi^2_{2d,\,0.025}}{2T},\ \frac{\chi^2_{2d+2,\,0.975}}{2T}\right)
      $$

      로 만든다. $d=0$이어도 상한 $\chi^2_{2,0.975}/(2T) = 7.38/(2T)$가 나온다.
    - **$\ln\lambda$ 척도의 왈드**도 대안이다. $\operatorname{SE}(\ln\hat\lambda) = 1/\sqrt d$로 간단하고, 되돌리면 비대칭 구간이 나온다.

    **설계상의 함의.** 신뢰성 시험의 표본크기 계획은 "몇 개를 시험할까"가 아니라 **"고장을 몇 개나 볼 것인가"** 로 세워야 한다. 가속수명시험처럼 고장을 일부러 앞당기는 기법이 쓰이는 이유다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
지수 모형을 쓰던 분석자가 자료에 이상치가 섞여 있음을 발견했다. MLE $\hat\lambda = 1/\bar X$가 이상치에 얼마나 취약한지 따져 보고 대안을 제시하라.

</div>

??? success "풀이"
    **취약성.** $\hat\lambda = 1/\bar X$이므로 $\bar X$의 취약성을 그대로 물려받는다. 관측값 하나가 $x_{(n)} \to \infty$이면

    $$
    \bar X \to \infty \implies \hat\lambda \to 0
    $$

    으로, **관측값 하나가 추정값을 0까지 끌고 갈 수 있다.** 붕괴점이 0이다.

    수치로 보면 $n=20$, 참 $\lambda=1$인 자료에 $x=50$인 값 하나가 섞이면 $\bar x$가 대략 $1 \to 3.45$로 뛰고 $\hat\lambda$가 $1 \to 0.29$로 떨어진다. 평균수명을 세 배 넘게 과대추정하는 셈이다.

    **대안.**

    - **중앙값 기반 추정.** 지수분포의 중앙값이 $\ln2/\lambda$이므로

      $$
      \tilde\lambda = \frac{\ln 2}{\tilde X}
      $$

      로 둔다. 붕괴점이 0.5로 대단히 강건하다. 대가는 효율로, 앞서 본 대로 점근상대효율이 $(\ln2)^2 = 0.48$이다.
    - **절사평균 기반.** 상위 $\alpha$를 잘라 낸 평균을 쓰되, 절단 때문에 생기는 편향을 보정한다. 중앙값보다 효율이 높으면서 어느 정도 강건하다.
    - **모형을 바꾼다.** 이상치가 오염이 아니라 **꼬리가 실제로 두껍다**는 신호일 수 있다. 와이불($k<1$)이나 로그정규, 파레토를 적합해 보고 우도비나 AIC로 비교한다.
    - **오염 혼합 모형.** $(1-\varepsilon)\text{Exp}(\lambda) + \varepsilon\,h(x)$ 꼴로 두고 EM 알고리즘으로 적합한다. 이상치를 버리지 않고 모형 안에서 다룬다.

    **먼저 할 일.** 이상치가 **기록 오류인지 진짜 관측인지** 확인하는 것이다. 측정 단위 혼동, 입력 실수, 장비 오작동이라면 수정하거나 제외하는 것이 옳다. 그러나 자료가 실제로 그렇게 생겼다면 이상치를 지우는 것은 결론을 조작하는 일이다. **어느 쪽이든 무엇을 왜 했는지 기록해야 하며, 이상치를 포함한 분석과 제외한 분석을 함께 보고하는 것이 안전하다.**

---

## 정리하며

지수분포의 최대가능도추정량은 표본평균의 **역수**다.

$$
\hat\lambda = \frac{1}{\bar X}
$$

- **유도는 간단하다.** 로그가능도 $n\log\lambda-\lambda\sum x_i$ 를 미분하면 $\hat\lambda=n/\sum x_i$ 가 나온다.
- **불변성이 작동한다.** 평균 $\mu=1/\lambda$ 의 최대가능도추정량은 $\bar X$ 이고, $\lambda$ 의 추정량은 그 역수다. **최대가능도는 변환에 대해 불변**이며, 이것이 최대가능도의 큰 장점이다.
- **그런데 그 불변성이 편향을 만든다.** $\bar X$ 는 불편이지만 $1/\bar X$ 는 그렇지 않다. 실제로 $\mathbb{E}[\hat\lambda]=\frac{n}{n-1}\lambda$ 이므로 **참값을 체계적으로 과대추정**한다. 옌센 부등식이 그 방향을 설명해 준다.
- **$n\le2$ 이면 분산이 무한하다.** 3장에서 본 대로, 점근적으로는 얌전한 정규분포로 가면서도 소표본의 실제 변동은 점근 공식이 말하는 것보다 훨씬 크다.
- **$\sum x_i$ 가 충분통계량**이고 감마분포를 따르므로, 정확한 분포를 써서 편향 보정과 정확 신뢰구간을 만들 수 있다.

다음 절 **포획–재포획법**으로 넘어간다. 모집단의 크기 자체를 추정하는 문제이며, 최대가능도가 직관적인 비례식으로 귀결되는 재미있는 예다.
