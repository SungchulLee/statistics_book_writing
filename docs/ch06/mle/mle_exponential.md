# Exponential 분포의 MLE

## 동기

Exponential 분포는 Poisson 과정에서 연속한 사건 사이의 대기 시간을 모형화한다. 예를 들어 고객 도착 사이의 시간, 부품 고장 사이의 시간, 방사성 붕괴 사이의 시간이 그렇다. 관측된 대기 시간으로부터 비율 모수 $\lambda$를 추정하는 것은 기본적인 응용 문제이다. Exponential의 경우 깔끔한 닫힌 형태의 MLE가 나오며, 동시에 비선형 변환이 추정량에 어떻게 편향을 들여오는지도 보여 준다.

## 설정

비율 모수가 $\lambda > 0$인 Exponential 분포에서 독립적으로 뽑은 확률표본 $X_1, X_2, \ldots, X_n$을 생각하자. 각 관측값의 밀도는

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

$$
\bar{x} = \frac{2.1 + 0.8 + 1.5 + 3.2 + 1.4}{5} = 1.8 \text{ minutes}
$$

따라서 비율 모수의 MLE는

$$
\hat{\lambda} = \frac{1}{\bar{x}} = \frac{1}{1.8} \approx 0.556 \text{ arrivals per minute}
$$

</div>

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
    Exponential 분포에서 $E[X] = 1/\lambda$이다. 적률법은 $\bar{X} = 1/\hat{\lambda}$로 두므로 $\hat{\lambda}_{\text{MOM}} = 1/\bar{X}$를 준다.

    이는 MLE와 동일하다. Exponential 분포가 단일모수 지수족이기 때문에 생기는 일치이다. 그런 족에서는 점수방정식 $\partial\ell/\partial\lambda = 0$과 1차 적률방정식 $\bar{X} = E_\lambda[X]$가 같은 추정량을 준다.

    두 추정량 모두 유한표본 편향 $\lambda/(n-1)$과 점근분산 $\lambda^2/n$을 공유한다.

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
