# 뉴턴-랩슨과 반복재가중최소제곱

## 반복 알고리즘이 필요한 이유

선형회귀에서 최소제곱 추정량은 닫힌 형태의 해
$\hat{\boldsymbol{\theta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$를 갖는다.
로지스틱 회귀에는 그런 호사가 없다. 로그가능도가 모수에 대해 비선형이므로 최대가능도 추정치는
반복적 수치 최적화로 찾아야 한다. 이 절에서는 가장 널리 쓰이는 두 알고리즘, **뉴턴-랩슨**과
**반복재가중최소제곱(IRLS)**을 전개한다.

## 로지스틱 회귀의 로그가능도

[가능도 절](../logistic_regression/likelihood.md)에서 보았듯이, $y_i \in \{0,1\}$인 독립
관측치 $(y_i, \mathbf{x}_i)$ $n$개에 대한 로그가능도는

$$
\ell(\boldsymbol{\theta})
= \sum_{i=1}^{n}\bigl[y_i\,\mathbf{x}_i^T\boldsymbol{\theta}
  - \log\bigl(1 + e^{\mathbf{x}_i^T\boldsymbol{\theta}}\bigr)\bigr]
$$

이며, $\boldsymbol{\theta} \in \mathbb{R}^{p}$는 절편과 기울기 계수를 모은 것이다. 관측치 $i$의
예측확률은 $\hat{p}_i = \sigma(\mathbf{x}_i^T\boldsymbol{\theta})$이고 $\sigma$는 시그모이드
함수다.

## 점수방정식

**점수**(로그가능도의 기울기)는

$$
\mathbf{s}(\boldsymbol{\theta})
= \frac{\partial \ell}{\partial \boldsymbol{\theta}}
= \sum_{i=1}^{n}(y_i - \hat{p}_i)\,\mathbf{x}_i
= \mathbf{X}^T(\mathbf{y} - \hat{\mathbf{p}})
$$

이다. 여기서 $\mathbf{X}$는 $n \times p$ 계획행렬이고
$\hat{\mathbf{p}} = (\hat{p}_1, \ldots, \hat{p}_n)^T$이다. 점수를 0으로 놓으면 $p$개의 비선형
방정식계를 얻는다. $\hat{\mathbf{p}}$가 시그모이드를 통해 $\boldsymbol{\theta}$에 의존하므로
닫힌 형태의 해가 없다.

## 헤세행렬

로그가능도의 이차도함수가 **헤세행렬**이다.

$$
\mathbf{H}(\boldsymbol{\theta})
= \frac{\partial^2 \ell}{\partial \boldsymbol{\theta}\,\partial \boldsymbol{\theta}^T}
= -\sum_{i=1}^{n}\hat{p}_i(1-\hat{p}_i)\,\mathbf{x}_i\mathbf{x}_i^T
= -\mathbf{X}^T\mathbf{W}\mathbf{X}
$$

여기서 $\mathbf{W} = \operatorname{diag}\bigl(\hat{p}_1(1-\hat{p}_1),\ldots,
\hat{p}_n(1-\hat{p}_n)\bigr)$는 $n \times n$ 대각 가중행렬이다. $0 < \hat{p}_i < 1$인 한
$\mathbf{W}$의 대각원소는 모두 양수이므로, $\mathbf{X}$가 완전열계수이면 헤세행렬은 음정치다.
그러면 로그가능도가 **강오목**이므로 국소최대점은 유일한 전역최대점이 된다.

!!! note "완전열계수 가정이 필요하다"
    $\mathbf{X}$의 열이 일차종속이면 $\mathbf{X}v = 0$인 $v \ne 0$이 존재하여
    $v^T\mathbf{H}v = 0$이 된다. 즉 헤세행렬은 음반정치일 뿐이고 로그가능도는 오목이지만
    강오목은 아니다. 이때 최대점은 유일하지 않고, $\mathbf{X}^T\mathbf{W}\mathbf{X}$가
    특이행렬이 되어 뉴턴 단계 자체가 정의되지 않는다.

## 뉴턴-랩슨 알고리즘

뉴턴-랩슨은 다음 반복으로 점수방정식의 근을 찾는다.

$$
\boldsymbol{\theta}^{(t+1)}
= \boldsymbol{\theta}^{(t)}
  - \bigl[\mathbf{H}(\boldsymbol{\theta}^{(t)})\bigr]^{-1}
    \mathbf{s}(\boldsymbol{\theta}^{(t)})
$$

위에서 유도한 점수와 헤세행렬을 대입하면

$$
\boldsymbol{\theta}^{(t+1)}
= \boldsymbol{\theta}^{(t)}
  + \bigl(\mathbf{X}^T\mathbf{W}^{(t)}\mathbf{X}\bigr)^{-1}
    \mathbf{X}^T\bigl(\mathbf{y} - \hat{\mathbf{p}}^{(t)}\bigr)
$$

이며, $\mathbf{W}^{(t)}$와 $\hat{\mathbf{p}}^{(t)}$는 현재 반복값
$\boldsymbol{\theta}^{(t)}$에서 평가한 것이다.

### 수렴

로그가능도가 강오목이므로 뉴턴-랩슨은 어떤 출발점에서든 유일한 MLE로 수렴한다. 최적점 근처에서
수렴은 **이차적**이다. 즉 반복마다 정확한 자릿수가 대략 두 배가 된다. 실무에서는 3회에서 8회면
충분한 경우가 많다.

## 피셔 점수법과 IRLS

**피셔 점수법**은 관측 헤세행렬을 그 기댓값으로 대체한다. 로지스틱 회귀에서는 두 양이 일치한다.

$$
-\mathbf{H}(\boldsymbol{\theta})
= \mathbf{X}^T\mathbf{W}\mathbf{X}
= \mathcal{I}(\boldsymbol{\theta})
$$

여기서 $\mathcal{I}(\boldsymbol{\theta})$는 **피셔 정보행렬**이다. 따라서 로지스틱 회귀에서는
피셔 점수법과 뉴턴-랩슨이 같은 반복열을 만든다.

### IRLS 해석

뉴턴-랩슨 갱신식을 다시 쓰면

$$
\boldsymbol{\theta}^{(t+1)}
= \bigl(\mathbf{X}^T\mathbf{W}^{(t)}\mathbf{X}\bigr)^{-1}
  \mathbf{X}^T\mathbf{W}^{(t)}\mathbf{z}^{(t)}
$$

이고, **작업 반응변수**는

$$
\mathbf{z}^{(t)}
= \mathbf{X}\boldsymbol{\theta}^{(t)}
  + \bigl(\mathbf{W}^{(t)}\bigr)^{-1}
    \bigl(\mathbf{y} - \hat{\mathbf{p}}^{(t)}\bigr)
$$

이다. 각 반복은 가중치 $\mathbf{W}^{(t)}$와 반응변수 $\mathbf{z}^{(t)}$를 갖는 **가중최소제곱**
문제를 푼다. 가중치가 단계마다 바뀌므로 이 절차를 **반복재가중최소제곱**이라 부른다.

## 알고리즘 요약

| 단계 | 내용 |
|---|---|
| 0 | $\boldsymbol{\theta}^{(0)}$을 초기화한다(예: 전부 0) |
| 1 | $\hat{\mathbf{p}}^{(t)} = \sigma(\mathbf{X}\boldsymbol{\theta}^{(t)})$를 계산한다 |
| 2 | $\mathbf{W}^{(t)} = \operatorname{diag}(\hat{p}_i(1-\hat{p}_i))$를 만든다 |
| 3 | 작업 반응변수 $\mathbf{z}^{(t)}$를 계산한다 |
| 4 | $\boldsymbol{\theta}^{(t+1)} = (\mathbf{X}^T\mathbf{W}^{(t)}\mathbf{X})^{-1}\mathbf{X}^T\mathbf{W}^{(t)}\mathbf{z}^{(t)}$를 푼다 |
| 5 | $\lVert\boldsymbol{\theta}^{(t+1)} - \boldsymbol{\theta}^{(t)}\rVert < \varepsilon$이 될 때까지 1--4를 반복한다 |

??? example "손으로 따라가는 예: 첫 두 반복"
    설명변수 하나(와 절편)를 갖는 관측치 $n=4$를 생각하자.

    | $i$ | $x_i$ | $y_i$ |
    |---|---|---|
    | 1 | 1.0 | 1 |
    | 2 | 2.0 | 1 |
    | 3 | 3.0 | 0 |
    | 4 | 4.0 | 0 |

    **반복 0.** $\boldsymbol{\theta}^{(0)} = (0, 0)^T$에서 출발한다. 그러면 모든 $i$에 대해
    $\hat{p}_i = 0.5$이고 $w_i = 0.25$이다.

    점수는 $\mathbf{X}^T(\mathbf{y} - \hat{\mathbf{p}})$이고
    $\mathbf{y} - \hat{\mathbf{p}} = (0.5, 0.5, -0.5, -0.5)^T$이다.

    **반복 1.** 가중최소제곱 문제를 풀면 $\boldsymbol{\theta}^{(1)} = (4.0,\, -1.6)^T$를 얻는다.
    $x_i = 1, 2$인 관측치의 $\hat{p}_i$는 올라가고 $x_i = 3, 4$인 관측치의 $\hat{p}_i$는
    내려간다.

    !!! warning "이 예제는 수렴하지 않는다"
        위 자료는 $x = 2.5$를 기준으로 **완전히 분리 가능**하므로 MLE가 존재하지 않는다.
        반복을 계속하면 다음과 같이 계수가 한없이 커진다.

        | $t$ | $\theta_0^{(t)}$ | $\theta_1^{(t)}$ |
        |---|---|---|
        | 1 | $4.00$ | $-1.60$ |
        | 2 | $7.11$ | $-2.84$ |
        | 5 | $21.58$ | $-8.63$ |
        | 10 | $46.68$ | $-18.67$ |
        | 20 | $96.68$ | $-38.67$ |

        반복마다 $\theta_0$이 약 $5$씩, $\theta_1$이 약 $-2$씩 커진다. 이차 수렴이 아니라
        발산이며, 아래 "분리와 비수렴" 절에서 설명하는 그대로다. 비분리 자료에서 알고리즘이
        어떻게 행동하는지는 연습문제 3에서 다룬다.

## 분리와 비수렴

두 범주가 **완전히 분리**되어 있으면 — 특성공간의 어떤 초평면이 모든 훈련점을 정확히 분류하면 —
MLE가 존재하지 않는다. $\lVert\boldsymbol{\theta}\rVert \to \infty$일 때 로그가능도가 상한에
접근할 뿐이고, 뉴턴-랩슨은 발산한다. 계수가 한없이 커지고 표준오차가 매우 커지는 것이 증상이다.

!!! warning "분리 탐지하기"
    최신 소프트웨어(R의 `glm`, 파이썬의 `statsmodels` 등)는 분리를 감지하면 경고를 낸다.
    해결책으로는 작은 능형 벌점을 더하거나(파스의 벌점가능도), 정확 조건부 로지스틱 회귀를 쓰는
    방법이 있다.

## 경사하강과의 관계

경사하강은 고정된 보폭 $\eta$로
$\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} + \eta\,\mathbf{s}(\boldsymbol{\theta}^{(t)})$
를 갱신한다. 뉴턴-랩슨은 헤세행렬의 역이 정해 주는 적응적 보폭을 쓰는 경사하강으로 볼 수 있다.
이 곡률 정보 덕분에 빠르게 수렴하지만, 매 단계 $p \times p$ 헤세행렬을 만들고 역을 구해야 하는
비용이 든다. $p$가 큰 고차원 문제에서는 L-BFGS 같은 준뉴턴법이 헤세행렬을 근사하여 계산 비용을
줄인다.


## 연습문제

**연습문제 1.**
뉴턴-랩슨 갱신식으로부터 IRLS의 작업 반응변수
$\mathbf{z}^{(t)} = \mathbf{X}\boldsymbol{\theta}^{(t)} + (\mathbf{W}^{(t)})^{-1}(\mathbf{y} - \hat{\mathbf{p}}^{(t)})$
를 유도하라.

??? success "풀이"

    표기를 줄이기 위해 위첨자 $(t)$를 생략하고 $\mathbf{W}$, $\hat{\mathbf{p}}$,
    $\boldsymbol{\theta}$로 쓴다. 뉴턴 갱신은

    $$
    \boldsymbol{\theta}^{+}
    = \boldsymbol{\theta} + (\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^T(\mathbf{y} - \hat{\mathbf{p}})
    $$

    이다. 첫 항 앞에 항등원 $(\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}(\mathbf{X}^T\mathbf{W}\mathbf{X})$
    를 끼워 넣으면

    $$
    \boldsymbol{\theta}^{+}
    = (\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}
      \bigl[\mathbf{X}^T\mathbf{W}\mathbf{X}\boldsymbol{\theta} + \mathbf{X}^T(\mathbf{y} - \hat{\mathbf{p}})\bigr]
    $$

    이고, 대괄호 안에서 $\mathbf{X}^T\mathbf{W}$를 묶어 내면

    $$
    = (\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^T\mathbf{W}
      \underbrace{\bigl[\mathbf{X}\boldsymbol{\theta} + \mathbf{W}^{-1}(\mathbf{y} - \hat{\mathbf{p}})\bigr]}_{\mathbf{z}}
    $$

    를 얻는다. 이는 반응변수 $\mathbf{z}$와 가중치 $\mathbf{W}$를 갖는 가중최소제곱의
    정규방정식이다.

    $\mathbf{W}$가 가역이어야 하는데, $0 < \hat{p}_i < 1$인 한 대각원소가 모두 양수이므로
    성립한다. $\hat{p}_i$가 0이나 1에 수치적으로 도달하면 $\mathbf{W}^{-1}$이 발산하며, 이것이
    분리가 있을 때 IRLS 구현이 깨지는 지점이다. $\square$

---

**연습문제 2.**
로지스틱 회귀에서 관측정보와 기대정보가 일치하는 이유를 설명하라. 이 성질이 성립하지 않는
이항 모형의 예를 하나 들어라.

??? success "풀이"

    관측정보는 $-\mathbf{H}(\boldsymbol{\theta})$이고 기대정보는
    $\mathcal{I}(\boldsymbol{\theta}) = \mathbb{E}[-\mathbf{H}(\boldsymbol{\theta})]$이다.
    로지스틱 회귀에서

    $$
    -\mathbf{H} = \sum_i \hat{p}_i(1-\hat{p}_i)\mathbf{x}_i\mathbf{x}_i^T
    $$

    인데, 이 식에는 $y_i$가 **전혀 나타나지 않는다.** $\mathbf{x}_i$를 조건부로 고정하면
    $-\mathbf{H}$는 상수이므로 기댓값을 취해도 그대로다. 따라서 두 정보가 같다.

    이는 우연이 아니라 **로짓이 이항분포의 정준연결**이기 때문이다. 정준연결을 쓰는 일반화선형
    모형에서는 언제나 관측정보와 기대정보가 일치하고, 뉴턴-랩슨과 피셔 점수법이 같아진다.

    **성립하지 않는 예:** 프로빗 모형 $P(Y=1) = \Phi(\mathbf{x}^T\boldsymbol{\theta})$은
    정준연결을 쓰지 않는다. 그 로그가능도의 이차도함수에는 $y_i$가 남아 있어 관측정보와
    기대정보가 다르며, 뉴턴-랩슨과 피셔 점수법이 서로 다른 반복열을 만든다. 여담이지만 이 경우
    피셔 점수법이 더 안정적인 경우가 많다. 기대정보는 항상 양반정치이지만 관측정보는 최적점에서
    멀리 떨어진 곳에서 그렇지 않을 수 있기 때문이다. $\square$

---

**연습문제 3.**
설명변수 $x = (1,2,3,4,5,6)$과 반응변수 $y = (0,0,1,0,1,1)$에 대해 $\boldsymbol{\theta}^{(0)} = \mathbf{0}$
에서 출발하는 IRLS를 구현하라. 반복마다 $\lVert\boldsymbol{\theta}^{(t)} - \hat{\boldsymbol{\theta}}\rVert$
를 기록하고 수렴이 이차적임을 확인하라.

??? success "풀이"

    ```python
    import numpy as np

    x = np.array([1., 2., 3., 4., 5., 6.])
    y = np.array([0., 0., 1., 0., 1., 1.])
    X = np.column_stack([np.ones(6), x])

    theta = np.zeros(2)
    history = [theta.copy()]
    for t in range(10):
        p_hat = 1 / (1 + np.exp(-X @ theta))
        W = p_hat * (1 - p_hat)
        H = X.T @ (X * W[:, None])          # X^T W X
        s = X.T @ (y - p_hat)               # score
        theta = theta + np.linalg.solve(H, s)
        history.append(theta.copy())

    theta_hat = theta
    for t, th in enumerate(history):
        print(f"{t}  {th}  err = {np.linalg.norm(th - theta_hat):.3e}")
    ```

    MLE는 $\hat{\boldsymbol{\theta}} = (-4.2491,\; 1.2140)^T$이고 오차의 변화는 다음과 같다.

    | $t$ | $\boldsymbol{\theta}^{(t)}$ | $\lVert\boldsymbol{\theta}^{(t)} - \hat{\boldsymbol{\theta}}\rVert$ |
    |---|---|---|
    | 0 | $(0,\ 0)$ | $4.42$ |
    | 1 | $(-2.800,\ 0.800)$ | $1.51$ |
    | 2 | $(-3.884,\ 1.110)$ | $3.80 \times 10^{-1}$ |
    | 3 | $(-4.221,\ 1.206)$ | $2.88 \times 10^{-2}$ |
    | 4 | $(-4.2489,\ 1.21398)$ | $1.74 \times 10^{-4}$ |
    | 5 | $(-4.24910,\ 1.214028)$ | $6.37 \times 10^{-9}$ |
    | 6 | $(-4.24910,\ 1.214028)$ | $9.2 \times 10^{-16}$ |

    이차 수렴이 뚜렷하다. 오차의 지수가 $-1 \to -2 \to -4 \to -9 \to -16$으로 대략 두 배씩
    커진다. 반복 6회 만에 배정도 부동소수점의 한계에 도달한다.

    첫 단계에서 $(-2.8,\ 0.8)$이라는 깔끔한 값이 나오는 것도 우연이 아니다.
    $\boldsymbol{\theta}^{(0)} = \mathbf{0}$이면 모든 $\hat{p}_i = 0.5$이고 $w_i = 1/4$로
    **상수**이므로, 첫 IRLS 단계는 가중치 없는 최소제곱
    $4(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T(\mathbf{y} - \tfrac12\mathbf{1})$로 환원된다.
    $\square$

---

**연습문제 4.**
반복당 계산 비용을 뉴턴-랩슨과 경사하강에 대해 $n$과 $p$로 나타내라. $n = 10^6$, $p = 10^4$일 때
어느 쪽을 쓰겠는가?

??? success "풀이"

    **뉴턴-랩슨 한 반복:**

    | 연산 | 비용 |
    |---|---|
    | $\hat{\mathbf{p}} = \sigma(\mathbf{X}\boldsymbol{\theta})$ | $O(np)$ |
    | $\mathbf{X}^T\mathbf{W}\mathbf{X}$ 구성 | $O(np^2)$ |
    | 그 계를 풀기(촐레스키) | $O(p^3)$ |
    | 합계 | $O(np^2 + p^3)$ |

    **경사하강 한 반복:** $\mathbf{X}^T(\mathbf{y} - \hat{\mathbf{p}})$만 필요하므로 $O(np)$.

    $n = 10^6$, $p = 10^4$이면 뉴턴 한 반복은 $np^2 = 10^{14}$번의 곱셈이 필요하다. 게다가
    $\mathbf{X}^T\mathbf{W}\mathbf{X}$만으로도 $p^2 = 10^8$개의 배정도 실수, 즉 800MB의
    메모리를 쓴다. 반복 몇 번이면 되더라도 현실적이지 않다.

    경사하강 한 반복은 $np = 10^{10}$이라 $10{,}000$배 싸다. 반복이 수천 번 필요해도 총 비용이
    뉴턴법 한 반복보다 작다.

    **실무적 답:** 이 규모에서는 확률적 경사하강이나 L-BFGS를 쓴다. L-BFGS는 최근 $m$개(보통
    $m \approx 10$)의 기울기 차이만으로 헤세행렬의 역을 암묵적으로 근사하므로, $p^2$ 크기의
    행렬을 만들지 않고도 곡률 정보를 일부 활용한다. 저장공간은 $O(mp)$이고 반복당 비용은
    $O(np + mp)$다. scikit-learn의 `LogisticRegression`이 기본 해법기로 L-BFGS를 쓰는 이유가
    바로 이것이다. $\square$

---

**연습문제 5.**
IRLS의 작업 반응변수 $\mathbf{z}^{(t)}$의 각 성분은 어떤 의미를 갖는가?
$\hat{p}_i$가 0이나 1에 가까울 때 $z_i$에 무슨 일이 일어나는지 설명하고, 실무적 대응책을
제시하라.

??? success "풀이"

    성분별로 쓰면

    $$
    z_i = \mathbf{x}_i^T\boldsymbol{\theta} + \frac{y_i - \hat{p}_i}{\hat{p}_i(1-\hat{p}_i)}
    $$

    이다. 첫 항은 현재의 로짓이고 둘째 항은 잔차를 확률 척도에서 로짓 척도로 옮긴 것이다. 실제로
    둘째 항은 $\hat p$에서 평가한 $\operatorname{logit}$의 도함수가 $1/[p(1-p)]$이므로, 관측된
    잔차를 로짓 척도로 선형근사한 값이다. 즉 $\mathbf{z}$는 "로짓 척도에서 본, 우리가 맞히고자
    하는 반응변수"다.

    $\hat{p}_i \to 0$ 또는 $1$이면 분모 $\hat{p}_i(1-\hat{p}_i) \to 0$이므로 $z_i$가 발산한다.
    동시에 가중치 $w_i = \hat{p}_i(1-\hat{p}_i)$도 0으로 간다. 가중최소제곱에서 이 관측치가
    기여하는 양은 $w_i z_i^2$ 꼴이므로 곱은 유한하게 남지만, $z_i$와 $w_i$를 따로 계산하면
    큰 수와 작은 수를 곱하는 전형적인 수치 문제가 생긴다.

    **실무적 대응:**

    1. 가중치에 하한을 둔다. 예컨대 $w_i \leftarrow \max(w_i, 10^{-10})$으로 잘라 낸다.
       R의 `glm`이 쓰는 방식이다.
    2. $\mathbf{z}$를 명시적으로 만들지 않고 뉴턴 형태
       $\boldsymbol{\theta}^{+} = \boldsymbol{\theta} + (\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^T(\mathbf{y}-\hat{\mathbf{p}})$
       를 그대로 쓴다. 대수적으로 동등하면서 $\mathbf{W}^{-1}$이 등장하지 않는다.
    3. 애초에 원인이 분리라면(연습문제와 위 예제 참조) 수치적 처방이 아니라 모형을 고쳐야 한다.
       벌점을 넣거나 문제의 변수를 제거하라.

    $\square$
