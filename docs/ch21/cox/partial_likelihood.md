# 부분가능도와 콕스 모형

모수적 생존 모형은 분석자가 사건시간의 분포를 지정할 것을 요구한다. 고른 분포가 틀리면 공변량
효과에 대한 추론이 편향될 수 있다. Cox(1972)가 도입한 **콕스 비례위험 모형**은 이 지정을 아예
피한다. **기저위험함수**를 전혀 지정하지 않은 채 공변량이 위험률에 미치는 효과를 모형화한다.

이 준모수적 접근은 생존분석에서 가장 널리 쓰이는 회귀 모형이다. 이 절에서는 콕스 모형을
정의하고, 공변량 효과를 추정하는 부분가능도를 유도하며, 기저 누적위험에 대한 브레슬로
추정량을 소개한다.

## 콕스 비례위험 모형

콕스 모형은 공변량 벡터 $\mathbf{x}_i = (x_{i1}, \ldots, x_{ip})^\top$를 갖는 대상 $i$의
위험함수를 다음과 같이 설정한다.

$$
h(t \mid \mathbf{x}_i) = h_0(t) \exp(\boldsymbol{\beta}^\top \mathbf{x}_i)
$$

여기서,

- $h_0(t)$는 **기저위험함수**로, 모든 공변량이 0일 때의 위험이다. 임의의, 지정되지 않은 음이
  아닌 함수다.
- $\boldsymbol{\beta} = (\beta_1, \ldots, \beta_p)^\top$는 회귀계수 벡터다.
- $\exp(\boldsymbol{\beta}^\top \mathbf{x}_i)$는 공변량 프로파일 $\mathbf{x}_i$에 결부된
  **상대위험**이다.

이 모형을 "비례위험"이라 부르는 이유는 임의의 두 대상 $i$와 $j$의 위험비가 시간에 걸쳐
일정하기 때문이다.

$$
\frac{h(t \mid \mathbf{x}_i)}{h(t \mid \mathbf{x}_j)} = \frac{\exp(\boldsymbol{\beta}^\top \mathbf{x}_i)}{\exp(\boldsymbol{\beta}^\top \mathbf{x}_j)} = \exp\!\bigl(\boldsymbol{\beta}^\top (\mathbf{x}_i - \mathbf{x}_j)\bigr)
$$

이 비는 $t$에 의존하지 않는다.

!!! note "준모수적 성격"

    콕스 모형은 준모수적이다. 공변량 효과 $\boldsymbol{\beta}$는 유한차원 모수이지만 기저위험
    $h_0(t)$는 무한차원의 성가신 모수다. 부분가능도는 $\boldsymbol{\beta}$의 추정에서
    $h_0(t)$를 소거한다.

## 부분가능도

서로 다른 $K$개의 사건시간을 크기순으로 $t_{(1)} < \cdots < t_{(K)}$라 하고, $t_{(j)}$에서
사건을 겪는 대상을 $i_j$라 하자. 시점 $t_{(j)}$의 **위험집합**은

$$
\mathcal{R}_j = \{i : t_i \geq t_{(j)}\}
$$

로, $t_{(j)}$ 직전까지 관측 중인 모든 대상의 집합이다.

사건시간 $t_{(j)}$에서 위험집합 안에 정확히 하나의 사건이 일어난다는 조건 아래, 그 사건을
겪는 대상이 $i_j$일 조건부확률은

$$
\frac{h(t_{(j)} \mid \mathbf{x}_{i_j})}{\sum_{l \in \mathcal{R}_j} h(t_{(j)} \mid \mathbf{x}_l)} = \frac{h_0(t_{(j)}) \exp(\boldsymbol{\beta}^\top \mathbf{x}_{i_j})}{\sum_{l \in \mathcal{R}_j} h_0(t_{(j)}) \exp(\boldsymbol{\beta}^\top \mathbf{x}_l)} = \frac{\exp(\boldsymbol{\beta}^\top \mathbf{x}_{i_j})}{\sum_{l \in \mathcal{R}_j} \exp(\boldsymbol{\beta}^\top \mathbf{x}_l)}
$$

이다. 비에서 기저위험 $h_0(t_{(j)})$가 소거된다. **부분가능도**는 모든 사건시간에 걸친 이
조건부확률들의 곱이다.

$$
PL(\boldsymbol{\beta}) = \prod_{j=1}^{K} \frac{\exp(\boldsymbol{\beta}^\top \mathbf{x}_{i_j})}{\sum_{l \in \mathcal{R}_j} \exp(\boldsymbol{\beta}^\top \mathbf{x}_l)}
$$

**부분 로그가능도**는

$$
\ell_P(\boldsymbol{\beta}) = \sum_{j=1}^{K} \left[\boldsymbol{\beta}^\top \mathbf{x}_{i_j} - \ln\!\left(\sum_{l \in \mathcal{R}_j} \exp(\boldsymbol{\beta}^\top \mathbf{x}_l)\right)\right]
$$

이다.

## 추정

MLE $\hat{\boldsymbol{\beta}}$는 $\ell_P(\boldsymbol{\beta})$를 최대화한다. 점수와 정보는
다음과 같다.

**점수벡터:**

$$
U(\boldsymbol{\beta}) = \sum_{j=1}^{K} \left[\mathbf{x}_{i_j} - \bar{\mathbf{x}}_j(\boldsymbol{\beta})\right]
$$

여기서 $\bar{\mathbf{x}}_j(\boldsymbol{\beta})$는 위험집합에 속한 공변량의 가중평균이다.

$$
\bar{\mathbf{x}}_j(\boldsymbol{\beta}) = \frac{\sum_{l \in \mathcal{R}_j} \mathbf{x}_l \exp(\boldsymbol{\beta}^\top \mathbf{x}_l)}{\sum_{l \in \mathcal{R}_j} \exp(\boldsymbol{\beta}^\top \mathbf{x}_l)}
$$

**관측정보:**

$$
\mathcal{I}(\boldsymbol{\beta}) = \sum_{j=1}^{K} \mathbf{V}_j(\boldsymbol{\beta})
$$

여기서 $\mathbf{V}_j$는 $t_{(j)}$의 위험집합에 속한 공변량의 가중 공분산행렬이다.

뉴턴-랩슨 반복은

$$
\boldsymbol{\beta}^{(m+1)} = \boldsymbol{\beta}^{(m)} + \mathcal{I}(\boldsymbol{\beta}^{(m)})^{-1} U(\boldsymbol{\beta}^{(m)})
$$

이다.

## 동점 사건시간 다루기

같은 시점에 여러 사건이 일어나면 부분가능도를 수정해야 한다. 흔한 접근은 다음과 같다.

- **브레슬로 근사**(대부분의 소프트웨어 기본값): 동점 사건들이 같은 위험집합 안에서 순차적으로
  일어난 것처럼 취급한다. 빠르지만 근사적이다.
- **에프론 근사**: 동점 사건의 가능한 순서들에 대해 평균하여 브레슬로보다 정확하다.
- **정확 부분가능도**: 가능한 모든 순서를 열거한다. 동점이 많으면 계산 비용이 크다.

!!! tip "어떤 동점 처리 방법을 쓸 것인가"

    동점이 드물면 브레슬로와 에프론이 거의 같은 결과를 준다. 동점이 흔하면(예: 이산 사건시간)
    에프론을 권한다. 정확 방법은 동점이 많은 작은 자료에만 쓴다.

    R의 `survival` 패키지는 에프론을, `lifelines`와 SAS는 브레슬로를 기본으로 쓴다. 동점이
    많은 자료에서 두 소프트웨어의 결과가 다르게 나오면 이 차이를 먼저 의심하라.

## 기저위험의 브레슬로 추정량

$\hat{\boldsymbol{\beta}}$를 추정한 뒤 기저 누적위험은 **브레슬로 추정량**으로 추정할 수 있다.

$$
\hat{H}_0(t) = \sum_{j:\, t_{(j)} \leq t} \frac{d_j}{\sum_{l \in \mathcal{R}_j} \exp(\hat{\boldsymbol{\beta}}^\top \mathbf{x}_l)}
$$

여기서 $d_j$는 $t_{(j)}$의 사건 수다. 기저 생존함수는

$$
\hat{S}_0(t) = \exp\!\bigl(-\hat{H}_0(t)\bigr)
$$

이고, 공변량 $\mathbf{x}$를 갖는 대상의 생존함수는

$$
\hat{S}(t \mid \mathbf{x}) = \hat{S}_0(t)^{\exp(\hat{\boldsymbol{\beta}}^\top \mathbf{x})}
$$

이다.

!!! warning "브레슬로 추정량은 계단함수다"

    넬슨-알렌 추정량과 마찬가지로 $\hat{H}_0(t)$는 관측된 사건시간에서만 뛴다. 사건시간
    사이에서는 추정치가 일정하다. 필요하면 평활 기법으로 연속 추정치를 만들 수 있지만
    계단함수가 표준 출력이다.

    $\boldsymbol{\beta} = \mathbf{0}$이면 브레슬로 추정량이 정확히 넬슨-알렌 추정량으로
    환원된다는 점도 확인하라. 분모의 $\exp(\hat{\boldsymbol{\beta}}^\top\mathbf{x}_l)$이
    모두 1이 되어 $\sum_l 1 = n_j$가 되기 때문이다.


## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
콕스 비례위험 모형의 부분가능도를 쓰고 왜 "부분"이라 불리는지 설명하라.

</div>

??? success "풀이"
    사건시간을 크기순으로 $t_{(1)} < t_{(2)} < \dots < t_{(D)}$라 하고 대응하는 대상을
    $j_1, j_2, \dots, j_D$라 하면 부분가능도는

    $$
    L(\boldsymbol{\beta}) = \prod_{i=1}^D \frac{\exp(\mathbf{x}_{j_i}^T\boldsymbol{\beta})}{\sum_{k \in R(t_{(i)})} \exp(\mathbf{x}_k^T\boldsymbol{\beta})}
    $$

    이다. 여기서 $R(t_{(i)})$는 시점 $t_{(i)}$의 위험집합(아직 관측 중인 모든 대상)이다.

    "부분"이라 불리는 이유는 사건의 **순서**만 쓰고 실제 사건시간이나 기저위험 $h_0(t)$를
    쓰지 않기 때문이다. $h_0(t)$를 소거함으로써 콕스 모형은 기저위험함수를 지정하지 않아도
    되고, 그래서 준모수적이다.

    더 정확히 말하면 이것은 완전한 가능도가 아니라 **조건부확률들의 곱**이다. 각 인자는
    "이 시점에 사건이 하나 일어났다는 조건에서 그것이 대상 $i_j$일 확률"이며, 사건시간의
    주변분포에 대한 정보는 버린다. 그럼에도 Cox는 이 부분가능도를 통상의 가능도처럼 다루어
    얻은 추정량이 일치성과 점근 정규성을 갖는다는 것을 보였고, 이후 계수과정 이론으로
    엄밀하게 증명되었다.

<div class="drillbox" markdown>

**연습문제 2.**
어떤 콕스 모형에서 처리 지시자의 계수가 $\hat{\beta} = 0.5$다. 이를 위험비로 해석하라.

</div>

??? success "풀이"
    위험비는 $\text{HR} = e^{\hat{\beta}} = e^{0.5} = 1.649$다.

    해석: 비례위험 가정 아래에서 처리군의 위험(사건의 순간 위험)이 모든 시점에서 대조군의
    1.649배다. 달리 말해 처리가 위험을 약 65% 높인다.

    사건이 사망이라면 $\text{HR} = 1.649$는 처리군이 임의의 시점에서 대조군보다 순간 사망
    위험이 65% 높다는 뜻이다. $\text{HR} < 1$이면 처리가 보호적이다.

    !!! warning "위험비는 생존시간의 비가 아니다"
        $\text{HR} = 1.649$가 "처리군이 1.649배 빨리 죽는다"거나 "생존시간이 $1/1.649$배"라는
        뜻이 **아니다.** 위험비는 순간 위험률의 비이며, 생존시간에 미치는 영향은 기저위험의
        모양에 달려 있다. 생존시간에 직접적인 곱셈 해석을 원한다면 AFT 모형(21.3절)을 써야
        한다. 와이불의 경우 두 해석이 연결되는데, 형상이 $k$이면 위험비 $\text{HR}$은 생존시간
        비 $\text{HR}^{-1/k}$에 대응한다.

<div class="drillbox" markdown>

**연습문제 3.**
비례위험 가정을 설명하라. 수학적으로 어떻게 표현되는가?

</div>

??? success "풀이"
    비례위험 가정은 임의의 두 대상 사이의 위험비가 시간에 걸쳐 일정하다는 것이다.

    $$
    \frac{h(t \mid \mathbf{x}_i)}{h(t \mid \mathbf{x}_j)} = \frac{h_0(t)\exp(\mathbf{x}_i^T\boldsymbol{\beta})}{h_0(t)\exp(\mathbf{x}_j^T\boldsymbol{\beta})} = \exp\!\left((\mathbf{x}_i - \mathbf{x}_j)^T\boldsymbol{\beta}\right)
    $$

    기저위험 $h_0(t)$가 소거되어 비가 시간과 무관해진다. 비가 시간에 따라 변하면(예: 처리가
    초기에는 효과적이지만 그 효과가 사라지면) 비례위험 가정이 위배되고 콕스 모형이 잘못
    지정된 것이다. 쇤펠트 잔차와 로그-로그 생존 그림으로 이 가정을 점검한다.

<div class="drillbox" markdown>

**연습문제 4.**
콕스 모형은 왜 절단된 관측치를 다룰 수 있는가? 절단된 대상이 부분가능도에 어떻게 들어가는지
설명하라.

</div>

??? success "풀이"
    절단된 대상은 사건으로서가 아니라 **위험집합**을 통해 부분가능도에 기여한다. 각 사건시간
    $t_{(i)}$에서 위험집합 $R(t_{(i)})$는 아직 살아 있고 관측 중인 모든 대상을 포함한다.
    나중에 사건을 겪을 대상과 나중에 절단될 대상 모두가 들어간다.

    절단된 대상은 부분가능도의 **분모**에 기여하지만(절단되기 전까지는 "위험에 있었다")
    분자에는 결코 나타나지 않는다(관측된 사건이 없었다). 이는 절단이 무정보라는 가정, 즉
    절단의 이유가 사건 위험과 무관하다는 가정 아래에서 타당하다.

    절단된 관측치를 버리지도, 사건시간을 대체하지도 않는 이 우아한 처리가 콕스 모형의 핵심
    장점이다. 카플란-마이어가 절단된 대상을 위험집합에서만 빼는 방식과 정확히 같은 착상이며,
    콕스 모형은 여기에 공변량 가중을 더한 것이라고 볼 수 있다.
