# Poisson 분포의 MLE

## 동기

Poisson 분포는 계수 자료의 표준 모형이다. 시간당 받는 이메일 수, 제품 하나당 결함 수, 교차로에서 연간 발생하는 사고 수 등이 그렇다. 관측된 계수로부터 비율 모수 $\lambda$를 추정하는 것은 가장 흔한 통계 작업 중 하나이다. Poisson의 MLE는 표본평균으로 밝혀지며, 모든 불편추정량 중에서 가능한 최선의 정밀도를 달성한다.

## 설정

비율 모수가 $\lambda > 0$인 Poisson 분포에서 독립적으로 뽑은 확률표본 $X_1, X_2, \ldots, X_n$을 생각하자. 각 관측값은 다음 확률질량함수를 가지고 $\{0, 1, 2, \ldots\}$의 값을 취한다:

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

## 예제

품질검사원이 회로기판 6개의 결함 수를 세어 $x_1 = 3$, $x_2 = 1$, $x_3 = 4$, $x_4 = 0$, $x_5 = 2$, $x_6 = 2$를 관측했다고 하자. 표본평균은

$$
\bar{x} = \frac{3 + 1 + 4 + 0 + 2 + 2}{6} = 2.0 \text{ defects per board}
$$

따라서 비율 모수의 MLE는 기판당 $\hat{\lambda} = 2.0$개의 결함이다.

## 성질

MLE를 구했으니 이제 그 통계적 특성을 살펴보자. 불편인지, 얼마나 정밀한지, 다른 불편추정량이 더 나을 수 있는지를 본다.

**불편성.** Exponential 분포의 MLE와 달리 Poisson의 MLE는 불편이다. $E[\bar{X}] = E[X_1] = \lambda$이므로 표본크기와 무관하게 평균적으로 참 모수를 맞힌다.

**Fisher 정보량.** 관측값 하나당 Fisher 정보량은

$$
I_1(\lambda) = -E\!\left[\frac{d^2 \log f(X;\lambda)}{d\lambda^2}\right] = -E\!\left[-\frac{X}{\lambda^2}\right] = \frac{E[X]}{\lambda^2} = \frac{1}{\lambda}
$$

크기 $n$인 표본에서 전체 Fisher 정보량은 $I_n(\lambda) = n / \lambda$이다.

**효율성.** $\lambda$의 임의의 불편추정량의 분산에 대한 Cramér-Rao 하한(CRLB)은

$$
\text{Var}(\hat{\lambda}) \geq \frac{1}{I_n(\lambda)} = \frac{\lambda}{n}
$$

$\text{Var}(\bar{X}) = \text{Var}(X_1)/n = \lambda/n$이므로 MLE가 이 한계를 정확히 달성한다. $\lambda$의 어떤 불편추정량도 이보다 작은 분산을 가질 수 없으며, 따라서 $\bar{X}$는 Poisson 비율에 대한 일률최소분산불편추정량(UMVUE)이다.

## 연습문제

**연습문제 1.**
확률표본 $X_1, \dots, X_n \overset{\text{iid}}{\sim} \text{Poisson}(\lambda)$에 대해 $\lambda$의 MLE를 유도하라.

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

---

**연습문제 2.**
어떤 병원 응급실이 24시간 동안 시간당 도착 환자 수를 기록했고 총합이 168명이다. 시간당 도착률 $\lambda$의 MLE를 구하고 근사적인 95% 신뢰구간을 구성하라.

??? success "풀이"
    $n = 24$시간이고 $\sum x_i = 168$이므로:

    $$
    \hat{\lambda} = \frac{168}{24} = 7.0 \text{ arrivals per hour}
    $$

    Poisson 관측값 하나에 대한 Fisher 정보량이 $I(\lambda) = 1/\lambda$이므로 점근분산은:

    $$
    \text{Var}(\hat{\lambda}) \approx \frac{1}{nI(\lambda)} = \frac{\lambda}{n} \approx \frac{7.0}{24} = 0.2917
    $$

    표준오차는 $\text{SE} = \sqrt{0.2917} \approx 0.5401$이다.

    95% 신뢰구간은:

    $$
    7.0 \pm 1.96 \times 0.5401 = 7.0 \pm 1.059 = (5.94, 8.06)
    $$

---

**연습문제 3.**
MLE $\hat{\lambda} = \bar{X}$가 불편임을 보이고 그 정확한 분산을 계산하라.

??? success "풀이"
    각 $X_i$가 $E[X_i] = \lambda$를 만족하므로:

    $$
    E[\hat{\lambda}] = E[\bar{X}] = \frac{1}{n}\sum_{i=1}^n E[X_i] = \lambda
    $$

    따라서 $\hat{\lambda}$는 불편이다.

    Poisson 분포에서 $\text{Var}(X_i) = \lambda$이고 $X_i$들이 독립이므로:

    $$
    \text{Var}(\hat{\lambda}) = \text{Var}(\bar{X}) = \frac{\lambda}{n}
    $$

    이는 정확히 Cramér-Rao 하한 $1/(nI(\lambda)) = \lambda/n$이므로 $\hat{\lambda} = \bar{X}$는 $\lambda$에 대한 UMVUE(일률최소분산불편추정량)이다. $\square$

---

**연습문제 4.**
Poisson의 MLE $\hat{\lambda} = \bar{X}$는 (척도를 제외하면) 충분통계량이기도 하다. Rao-Blackwell 정리를 사용하여 다른 어떤 불편추정량도 더 작은 분산을 가질 수 없는 이유를 설명하라.

??? success "풀이"
    (인수분해 정리에 의해) $\lambda$에 대한 충분통계량은 $T = \sum X_i$이다. MLE $\hat{\lambda} = T/n$은 이미 $T$의 함수이다.

    Rao-Blackwell 정리에 의해 $T$로 조건화한 임의의 불편추정량은 원래보다 분산이 크지 않다. $\hat{\lambda}$가 이미 $T$의 함수이므로, 다른 어떤 불편추정량을 $T$로 조건화해도 $\hat{\lambda}$보다 나아질 수 없다.

    나아가 Poisson 족은 완비 지수족이므로 Lehmann-Scheffé 정리가 $\hat{\lambda} = T/n$이 유일한 UMVUE임을 보장한다. 그 분산 $\lambda/n$이 Cramér-Rao 하한과 같으므로 (선형추정량뿐 아니라) 어떤 불편추정량도 이보다 나을 수 없음이 확인된다.
