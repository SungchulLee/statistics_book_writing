# 포아송분포의 MLE

## 동기

포아송분포는 계수 자료의 표준 모형이다. 시간당 받는 이메일 수, 제품 하나당 결함 수, 교차로에서 연간 발생하는 사고 수 등이 그렇다. 관측된 계수로부터 비율 모수 $\lambda$를 추정하는 것은 가장 흔한 통계 작업 중 하나이다. 포아송의 MLE는 표본평균으로 밝혀지며, 모든 불편추정량 중에서 가능한 최선의 정밀도를 달성한다.

## 설정

비율 모수가 $\lambda > 0$인 포아송분포에서 독립적으로 뽑은 확률표본 $X_1, X_2, \ldots, X_n$을 생각하자. 각 관측값은 다음 확률질량함수를 가지고 $\{0, 1, 2, \ldots\}$의 값을 취한다:

$$
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots
$$

목표는 관측된 계수를 가장 그럴듯하게 만드는 $\lambda$ 값을 찾는 것이다.

## 유도

관측값이 독립이므로 결합 PMF는 개별 질량함수의 곱이다. 로그를 취하면 로그가능도를 얻는다:

$$
\ell(\lambda) = \left(\sum_{i=1}^n x_i\right) \log \lambda - n\lambda - \sum_{i=1}^n \log(x_i!)
$$

마지막 항은 $\lambda$에 의존하지 않으므로 최적화에서 아무 역할도 하지 않는다. 로그가능도를 최대화하기 위해 $\lambda$에 대해 미분하고 결과를 0으로 둔다:

$$
\frac{d\ell}{d\lambda} = \frac{\sum_{i=1}^n x_i}{\lambda} - n = 0
$$

$\lambda$에 대해 풀면 최대가능도추정량을 얻는다:

$$
\hat{\lambda}_{\text{MLE}} = \frac{\sum_{i=1}^n x_i}{n} = \bar{X}
$$

이 임계점이 최댓값임을 확인하기 위해 2계도함수를 살펴본다:

$$
\frac{d^2\ell}{d\lambda^2} = -\frac{\sum_{i=1}^n x_i}{\lambda^2} \leq 0
$$

관측값 중 적어도 하나가 양수이면 엄격하게 음수이며, $\lambda > 0$일 때 거의 언제나 그러하다. 따라서 이 임계점은 로그가능도의 전역 최댓값이다.

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> 회로기판의 결함 수. 품질검사원이 회로기판 6개의 결함 수를 세어 $x_1 = 3$, $x_2 = 1$, $x_3 = 4$, $x_4 = 0$, $x_5 = 2$, $x_6 = 2$를 관측했다고 하자. 표본평균은

</div>

??? success "풀이"
    $$
    \bar{x} = \frac{3 + 1 + 4 + 0 + 2 + 2}{6} = 2.0 \text{ defects per board}
    $$

    따라서 비율 모수의 MLE는 기판당 $\hat{\lambda} = 2.0$개의 결함이다.

## 성질

MLE를 구했으니 이제 그 통계적 특성을 살펴보자. 불편인지, 얼마나 정밀한지, 다른 불편추정량이 더 나을 수 있는지를 본다.

**불편성.** 지수분포의 MLE와 달리 포아송의 MLE는 불편이다. $E[\bar{X}] = E[X_1] = \lambda$이므로 표본크기와 무관하게 평균적으로 참 모수를 맞힌다.

**Fisher 정보량.** 관측값 하나당 Fisher 정보량은

$$
I_1(\lambda) = -E\!\left[\frac{d^2 \log f(X;\lambda)}{d\lambda^2}\right] = -E\!\left[-\frac{X}{\lambda^2}\right] = \frac{E[X]}{\lambda^2} = \frac{1}{\lambda}
$$

크기 $n$인 표본에서 전체 Fisher 정보량은 $I_n(\lambda) = n / \lambda$이다.

**효율성.** $\lambda$의 임의의 불편추정량의 분산에 대한 Cramér-Rao 하한(CRLB)은

$$
\text{Var}(\hat{\lambda}) \geq \frac{1}{I_n(\lambda)} = \frac{\lambda}{n}
$$

$\text{Var}(\bar{X}) = \text{Var}(X_1)/n = \lambda/n$이므로 MLE가 이 한계를 정확히 달성한다. $\lambda$의 어떤 불편추정량도 이보다 작은 분산을 가질 수 없으며, 따라서 $\bar{X}$는 포아송 비율에 대한 일률최소분산불편추정량(UMVUE)이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
확률표본 $X_1, \dots, X_n \overset{\text{iid}}{\sim} \text{Poisson}(\lambda)$에 대해 $\lambda$의 MLE를 유도하라.

</div>

??? success "풀이"
    로그가능도는:

    $$
    \ell(\lambda) = \sum_{i=1}^n \bigl(x_i \log\lambda - \lambda - \log(x_i!)\bigr) = \left(\sum x_i\right)\log\lambda - n\lambda - \sum\log(x_i!)
    $$

    도함수를 0으로 두면:

    $$
    \frac{d\ell}{d\lambda} = \frac{\sum x_i}{\lambda} - n = 0 \implies \hat{\lambda} = \frac{\sum x_i}{n} = \bar{X}
    $$

    2계도함수가 ($\sum x_i > 0$일 때) $-\sum x_i / \lambda^2 < 0$이므로 최댓값임이 확인된다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
어떤 병원 응급실이 24시간 동안 시간당 도착 환자 수를 기록했고 총합이 168명이다. 시간당 도착률 $\lambda$의 MLE를 구하고 근사적인 95% 신뢰구간을 구성하라.

</div>

??? success "풀이"
    $n = 24$시간이고 $\sum x_i = 168$이므로:

    $$
    \hat{\lambda} = \frac{168}{24} = 7.0 \text{ arrivals per hour}
    $$

    포아송 관측값 하나에 대한 Fisher 정보량이 $I(\lambda) = 1/\lambda$이므로 점근분산은:

    $$
    \text{Var}(\hat{\lambda}) \approx \frac{1}{nI(\lambda)} = \frac{\lambda}{n} \approx \frac{7.0}{24} = 0.2917
    $$

    표준오차는 $\text{SE} = \sqrt{0.2917} \approx 0.5401$이다.

    95% 신뢰구간은:

    $$
    7.0 \pm 1.96 \times 0.5401 = 7.0 \pm 1.059 = (5.94, 8.06)
    $$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
MLE $\hat{\lambda} = \bar{X}$가 불편임을 보이고 그 정확한 분산을 계산하라.

</div>

??? success "풀이"
    각 $X_i$가 $E[X_i] = \lambda$를 만족하므로:

    $$
    E[\hat{\lambda}] = E[\bar{X}] = \frac{1}{n}\sum_{i=1}^n E[X_i] = \lambda
    $$

    따라서 $\hat{\lambda}$는 불편이다.

    포아송분포에서 $\text{Var}(X_i) = \lambda$이고 $X_i$들이 독립이므로:

    $$
    \text{Var}(\hat{\lambda}) = \text{Var}(\bar{X}) = \frac{\lambda}{n}
    $$

    이는 정확히 Cramér-Rao 하한 $1/(nI(\lambda)) = \lambda/n$이므로 $\hat{\lambda} = \bar{X}$는 $\lambda$에 대한 UMVUE(일률최소분산불편추정량)이다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
포아송의 MLE $\hat{\lambda} = \bar{X}$는 (척도를 제외하면) 충분통계량이기도 하다. Rao-Blackwell 정리를 사용하여 다른 어떤 불편추정량도 더 작은 분산을 가질 수 없는 이유를 설명하라.

</div>

??? success "풀이"
    (인수분해 정리에 의해) $\lambda$에 대한 충분통계량은 $T = \sum X_i$이다. MLE $\hat{\lambda} = T/n$은 이미 $T$의 함수이다.

    Rao-Blackwell 정리에 의해 $T$로 조건화한 임의의 불편추정량은 원래보다 분산이 크지 않다. $\hat{\lambda}$가 이미 $T$의 함수이므로, 다른 어떤 불편추정량을 $T$로 조건화해도 $\hat{\lambda}$보다 나아질 수 없다.

    나아가 포아송 족은 완비 지수족이므로 Lehmann-Scheffé 정리가 $\hat{\lambda} = T/n$이 유일한 UMVUE임을 보장한다. 그 분산 $\lambda/n$이 Cramér-Rao 하한과 같으므로 (선형추정량뿐 아니라) 어떤 불편추정량도 이보다 나을 수 없음이 확인된다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
관측 단위마다 **노출량**이 다른 경우를 생각하자. 지역 $i$의 인구가 $t_i$이고 사건 수가 $X_i \sim \text{Poisson}(\lambda t_i)$일 때 $\lambda$의 MLE를 구하라. 각 지역의 비율 $X_i/t_i$를 단순평균한 것과 어떻게 다른가?

</div>

??? success "풀이"
    로그가능도는 상수를 빼고

    $$
    \ell(\lambda) = \sum_i \left\{x_i\ln(\lambda t_i) - \lambda t_i\right\} = \left(\sum_i x_i\right)\ln\lambda - \lambda\sum_i t_i + \text{상수}
    $$

    이므로

    $$
    \ell'(\lambda) = \frac{\sum_i x_i}{\lambda} - \sum_i t_i = 0 \implies \hat\lambda = \frac{\sum_i x_i}{\sum_i t_i}
    $$

    이다. **총 사건 수를 총 노출량으로 나눈 값**이며, 이를 조율비율(pooled rate)이라 한다.

    **단순평균과의 차이.** 단순평균은

    $$
    \frac{1}{k}\sum_i \frac{x_i}{t_i}
    $$

    로 각 지역에 같은 가중치를 준다. MLE는 노출량으로 가중한 평균이다.

    $$
    \hat\lambda = \sum_i \frac{t_i}{\sum_j t_j}\cdot\frac{x_i}{t_i}
    $$

    **인구가 큰 지역의 비율이 더 정확하므로** 그쪽에 큰 가중치를 주는 것이 옳다. 실제로 $\operatorname{Var}(X_i/t_i) = \lambda/t_i$이므로 정밀도가 $t_i$에 비례하고, 앞서 본 정밀도 가중 원리와 정확히 일치한다.

    단순평균을 쓰면 인구 100명인 마을의 요동치는 비율이 인구 100만 도시의 안정된 비율과 같은 무게를 갖게 되어 추정이 불안정해진다.

    **일반화.** 이 구조가 **포아송 회귀의 오프셋**이다. $\ln E[X_i] = \ln t_i + \mathbf{x}_i^\top\boldsymbol\beta$로 두면 $\ln t_i$가 계수 1로 고정된 항(오프셋)이 되고, 모형이 계수(count)가 아니라 비율(rate)을 설명하게 된다. 관찰 기간, 인구, 노출 면적이 다른 자료에서 반드시 필요한 장치다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
0을 관측할 수 없는 자료(예: 병원에 온 환자만 기록하므로 방문 횟수가 1 이상)의 분포는 **영절단 포아송**이다. 그 PMF를 쓰고 MLE 방정식을 유도하라. $\bar x = 2.5$일 때 $\hat\lambda$를 어림하라.

</div>

??? success "풀이"
    **PMF.** $P(X \ge 1) = 1-e^{-\lambda}$로 나누어 정규화한다.

    $$
    P(X=k) = \frac{e^{-\lambda}\lambda^k}{k!\,(1-e^{-\lambda})}, \qquad k = 1,2,\dots
    $$

    **평균.**

    $$
    E[X] = \frac{\lambda}{1-e^{-\lambda}}
    $$

    보통의 포아송보다 크다. 0을 잘라 냈으니 당연하다.

    **MLE 방정식.** 로그가능도가

    $$
    \ell(\lambda) = \left(\sum_i x_i\right)\ln\lambda - n\lambda - n\ln(1-e^{-\lambda}) + \text{상수}
    $$

    이므로

    $$
    \ell'(\lambda) = \frac{n\bar x}{\lambda} - n - \frac{ne^{-\lambda}}{1-e^{-\lambda}} = 0
    $$

    정리하면

    $$
    \frac{\hat\lambda}{1-e^{-\hat\lambda}} = \bar x
    $$

    이다. **적률방정식과 같은 꼴**(관측 평균 = 모형 평균)이며, 닫힌 해가 없어 수치적으로 풀어야 한다.

    **$\bar x = 2.5$일 때.** $g(\lambda) = \lambda/(1-e^{-\lambda})$가 증가함수이고 $g(2.2) = 2.474$, $g(2.3) = 2.556$이므로 그 사이에서 $\hat\lambda \approx 2.23$이다.

    **주의할 점.** 절단을 무시하고 보통의 포아송 MLE $\hat\lambda = \bar x = 2.5$를 쓰면 **$\lambda$를 12% 과대추정**한다. 절단이 심할수록($\lambda$가 작을수록) 오차가 커진다. $\bar x = 1.2$라면 $\hat\lambda \approx 0.376$으로 세 배 넘게 차이 난다.

    같은 발상이 절단 정규분포, 절단 회귀(토빗 모형), 길이편향 표집 보정에 쓰인다. **표본이 어떻게 관측되었는지가 가능도에 반영되어야 한다**는 것이 요점이다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
지역 A에서 10년간 45건, 지역 B에서 10년간 30건의 사고가 났다. 두 지역의 사고율이 같은지 우도비 검정으로 판정하라. 조건부 이항검정과 견주어라.

</div>

??? success "풀이"
    **MLE.** 각각 $\hat\lambda_A = 4.5$, $\hat\lambda_B = 3.0$건/년이고, 귀무가설 아래의 공통 비율은 $\hat\lambda_0 = 75/20 = 3.75$다.

    **우도비 통계량.** 로그가능도에서 $\lambda$에 의존하는 부분만 남기면

    $$
    G = 2\left\{x_A\ln\hat\lambda_A + x_B\ln\hat\lambda_B - (x_A+x_B)\ln\hat\lambda_0\right\}
    $$

    ($-\lambda t$ 항들은 $\sum\hat\lambda_i t_i = \hat\lambda_0 t$로 상쇄된다.) 값을 넣으면

    $$
    G = 2\left\{45\ln4.5 + 30\ln3.0 - 75\ln3.75\right\} = 3.020
    $$

    이고 $\chi^2_1$에서 $p\text{-값} = 0.082$다. 5%에서 기각하지 못한다.

    **조건부 이항검정.** 총합 $x_A+x_B = 75$를 고정하면 $X_A \mid 75 \sim \text{Binomial}(75,\ t_A/(t_A+t_B)) = \text{Binomial}(75, 0.5)$이다. 정확 이항검정의 양측 $p$-값은 **0.105**다.

    **비교.** 두 방법이 같은 방향의 결론을 주지만 $p$-값이 0.082 대 0.105로 꽤 다르다. 우도비는 카이제곱 근사라 다소 낙관적이고, 이항 정확검정은 이산성 때문에 보수적이다.

    **어느 쪽을 쓸 것인가.** 사건 수가 충분히 많으면(각각 10건 이상) 둘이 가까워지므로 계산이 편한 쪽을 쓴다. 사건 수가 적으면 **정확검정**이 안전하다. 조건부 접근의 또 다른 장점은 성가신 모수(전체 사건률)가 조건화로 사라진다는 점이며, 노출 기간이 달라도 $t_A/(t_A+t_B)$를 성공확률로 두면 그대로 통한다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
계수 자료에 포아송 MLE를 적합한 뒤 가정을 점검하려 한다. **이탈도**와 **피어슨 카이제곱**을 정의하고, 과대산포를 어떻게 진단하는지 적어라.

</div>

??? success "풀이"
    적합값을 $\hat\mu_i$라 하자(단순 포아송이면 모두 $\bar x$, 회귀모형이면 $\exp(\mathbf{x}_i^\top\hat{\boldsymbol\beta})$).

    **이탈도.** 포화모형(각 관측에 모수 하나)과의 우도비다.

    $$
    D = 2\sum_i\left\{x_i\ln\frac{x_i}{\hat\mu_i} - (x_i-\hat\mu_i)\right\}
    $$

    ($x_i=0$이면 첫 항을 0으로 본다.)

    **피어슨 카이제곱.**

    $$
    X^2 = \sum_i \frac{(x_i-\hat\mu_i)^2}{\hat\mu_i}
    $$

    분모가 $\hat\mu_i$인 것은 포아송에서 분산 = 평균이기 때문이다.

    **진단.** 모형이 맞으면 둘 다 근사적으로 $\chi^2_{n-p}$를 따르므로

    $$
    \hat\phi = \frac{X^2}{n-p} \approx 1
    $$

    이어야 한다. 이 값이 1보다 뚜렷이 크면 **과대산포**다.

    | $\hat\phi$ | 판단 |
    |---|---|
    | $\approx 1$ | 포아송 적합 |
    | $1.5\sim2$ | 경미한 과대산포. 준포아송으로 표준오차만 보정 |
    | $> 2$ | 음이항이나 영과잉 모형 검토 |

    **왜 중요한가.** 과대산포를 놓치면 **표준오차를 $\sqrt{\hat\phi}$배만큼 과소평가**한다. $\hat\phi = 4$면 표준오차가 절반으로 나와 $z$ 통계량이 두 배가 되고, 없는 유의성이 쏟아진다. 계수 추정값 자체는 대체로 일치하므로 문제를 알아채기 어렵다.

    **주의.** 이탈도를 적합도 검정에 쓰려면 $\hat\mu_i$가 충분히 커야 한다($\ge5$ 정도). 계수가 작으면 이탈도의 $\chi^2$ 근사가 나쁘고, 특히 0이 많은 자료에서는 믿을 수 없다. 그때는 잔차 그림이나 예측분포와 관측분포의 비교(루트그램)가 더 유용하다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
설명변수가 있는 **포아송 회귀** $\ln\mu_i = \mathbf{x}_i^\top\boldsymbol\beta$의 로그가능도를 쓰고 점수방정식을 유도하라. 왜 로그연결함수를 쓰며, 닫힌 해가 없는 이유는 무엇인가?

</div>

??? success "풀이"
    **로그가능도.** $\mu_i = \exp(\mathbf{x}_i^\top\boldsymbol\beta)$이므로

    $$
    \ell(\boldsymbol\beta) = \sum_i \left\{x_i\ln\mu_i - \mu_i - \ln x_i!\right\} = \sum_i\left\{x_i\,\mathbf{x}_i^\top\boldsymbol\beta - e^{\mathbf{x}_i^\top\boldsymbol\beta}\right\} + \text{상수}
    $$

    **점수방정식.** $\boldsymbol\beta$로 미분하면

    $$
    \frac{\partial\ell}{\partial\boldsymbol\beta} = \sum_i\left(x_i - e^{\mathbf{x}_i^\top\boldsymbol\beta}\right)\mathbf{x}_i = \sum_i (x_i-\mu_i)\,\mathbf{x}_i = \mathbf{0}
    $$

    **"관측값 빼기 적합값"에 설명변수를 가중한 형태**로, 로지스틱 회귀의 점수방정식과 똑같은 모양이다. 지수족에 정준연결함수를 쓰면 언제나 이 꼴이 나온다.

    **헤시안.**

    $$
    \frac{\partial^2\ell}{\partial\boldsymbol\beta\partial\boldsymbol\beta^\top} = -\sum_i \mu_i\,\mathbf{x}_i\mathbf{x}_i^\top
    $$

    $\mu_i > 0$이므로 언제나 음반정부호다. **로그가능도가 오목하므로 국소 최대가 곧 전역 최대**이고, 뉴턴-랩슨(IRLS)이 안정적으로 수렴한다.

    **왜 로그연결인가.**

    1. **모수공간을 지킨다.** $\mu_i = e^{\eta_i} > 0$이 자동으로 보장된다. 항등연결 $\mu_i = \mathbf{x}_i^\top\boldsymbol\beta$를 쓰면 음수 평균이 나올 수 있다.
    2. **포아송의 정준연결이다.** 지수족 표준형 $\exp\{x\theta - b(\theta)\}$에서 $\theta = \ln\mu$이므로, 이를 선형으로 두면 위와 같은 깔끔한 점수방정식과 오목성이 따라온다.
    3. **곱셈적 해석.** $\ln\mu = \beta_0+\beta_1x$이면 $x$가 1 늘 때 $\mu$가 $e^{\beta_1}$배가 된다. 비율의 상대적 변화로 읽히며, 이는 계수 자료에 자연스러운 해석이다.

    **닫힌 해가 없는 이유.** 점수방정식에 $\boldsymbol\beta$가 지수함수 **안**에 들어 있다. 선형모형이라면 $\sum(y_i-\mathbf{x}_i^\top\boldsymbol\beta)\mathbf{x}_i=0$이 $\boldsymbol\beta$에 대한 선형방정식이라 정규방정식으로 풀리지만, 여기서는 $e^{\mathbf{x}_i^\top\boldsymbol\beta}$ 때문에 초월방정식이 된다. 일반화선형모형 대부분이 같은 이유로 반복 알고리즘을 필요로 한다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
포아송 자료에 $\sqrt{X}$ 변환을 쓰면 분산이 안정된다고 알려져 있다. 델타 방법으로 확인하고, 그럼에도 요즘은 변환보다 포아송 회귀를 권하는 이유를 적어라.

</div>

??? success "풀이"
    **분산안정화 확인.** $g(x) = \sqrt x$이므로 $g'(\lambda) = 1/(2\sqrt\lambda)$이고 $\operatorname{Var}(X) = \lambda$이므로

    $$
    \operatorname{Var}(\sqrt X) \approx \left(\frac{1}{2\sqrt\lambda}\right)^2\lambda = \frac14
    $$

    로 **$\lambda$에 무관한 상수**가 된다. $\square$

    일반 원리로 $\operatorname{Var}(X) = V(\mu)$일 때 $g(\mu) = \int d\mu/\sqrt{V(\mu)}$가 분산을 안정시키며, 포아송은 $V(\mu)=\mu$라 제곱근이 나온다. 이항의 $\arcsin\sqrt{\hat p}$, 상관계수의 $\operatorname{arctanh}$도 같은 계산이다.

    실무에서는 $\lambda$가 작을 때 근사가 나빠져 **안스콤 변환** $2\sqrt{X+3/8}$이나 프리먼-튜키 변환 $\sqrt X + \sqrt{X+1}$을 쓴다.

    **그럼에도 변환을 권하지 않는 이유.**

    1. **해석이 망가진다.** $\sqrt{X}$의 평균은 $\sqrt{E[X]}$가 아니다(옌센 부등식). 변환 척도에서 얻은 결과를 되돌리면 편향이 생기고, 회귀계수를 원래 척도의 효과로 읽을 수 없다.
    2. **0을 다루기 어렵다.** 계수 자료에는 0이 많은데 $\sqrt0 = 0$이 경계에 붙어 정규성이 깨진다. 로그 변환이라면 아예 정의되지 않아 $\ln(x+1)$ 같은 임시방편을 쓰게 되고, 더하는 상수에 따라 결과가 달라진다.
    3. **한 가지 문제만 고친다.** 변환은 등분산을 겨냥하지만 정규성과 선형성까지 동시에 만족시킨다는 보장이 없다. 세 가지를 한 변환으로 맞추는 것은 대개 불가능하다.
    4. **일반화선형모형이 더 낫다.** 포아송 회귀는 **원래 척도에서 평균을 모형화**하면서 분산구조를 따로 지정한다. 변환할 필요가 없고, 계수가 비율비로 바로 해석되며, 과대산포도 준포아송이나 음이항으로 자연스럽게 확장된다.

    분산안정화 변환은 컴퓨터가 귀하던 시절에 선형모형 도구를 계수 자료에 억지로 맞추려던 장치였다. **원리를 아는 것은 여전히 유용하지만, 분석의 기본 도구로 쓸 이유는 없다.**

---

## 정리하며

포아송의 최대가능도추정량도 **표본평균**이다.

$$
\hat\lambda = \bar X
$$

- **로그가능도가 $\left(\sum x_i\right)\log\lambda - n\lambda - \sum\log(x_i!)$** 이고, 마지막 항은 $\lambda$ 와 무관해 버려도 된다.
- **불편이고 효율적이다.** $\mathbb{E}[\bar X]=\lambda$ 이며 분산 $\lambda/n$ 이 크라메르–라오 하한 $1/(nI(\lambda))=\lambda/n$ 과 일치한다. **모든 불편추정량 중 최선이다.**
- **평균과 분산이 같다는 것이 포아송의 지문이다.** 자료에서 표본분산이 표본평균보다 뚜렷이 크면(과산포) 포아송 가정이 틀린 것이며, 음이항 같은 모형이 필요하다. 이 진단은 적합 뒤에 반드시 해야 한다.
- **$\sum x_i$ 가 충분통계량**이고, 포아송의 가법성 덕분에 그 합 자체가 다시 $\text{Poisson}(n\lambda)$ 를 따른다.
- 관측이 모두 $0$ 이면 $\hat\lambda=0$ 이 되어 신뢰구간을 만들 수 없다. 드문 사건에서 실제로 마주치는 경계 문제다.

다음 절 **지수분포의 최대가능도**로 넘어간다. 닫힌 형태가 나오지만, 이번에는 **비선형 변환이 편향을 들여오는** 모습을 함께 보게 된다.
