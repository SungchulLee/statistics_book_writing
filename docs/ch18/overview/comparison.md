# 능형 대 라쏘 대 엘라스틱넷

이 절은 세 가지 주요 정칙화 방법 — 능형회귀(L2), 라쏘(L1), 엘라스틱넷(L1+L2) — 을 체계적으로 비교한다. 각 방법은 축소, 희소성, 안정성, 계산비용 사이에서 서로 다른 절충을 한다.

## 벌점과 목적함수의 비교

세 방법 모두 다음 형태의 벌점최소제곱을 최소화한다.

$$
\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \left\{\frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda\, P(\boldsymbol{\beta})\right\}
$$

차이는 벌점 $P(\boldsymbol{\beta})$에 있다.

| 방법 | 벌점 $P(\boldsymbol{\beta})$ | 제약 모양 |
|---|---|---|
| 능형 | $\frac{1}{2}\|\boldsymbol{\beta}\|_2^2$ | 구 (L2 공) |
| 라쏘 | $\|\boldsymbol{\beta}\|_1$ | 마름모 (L1 공) |
| 엘라스틱넷 | $\alpha\|\boldsymbol{\beta}\|_1 + (1-\alpha)\frac{1}{2}\|\boldsymbol{\beta}\|_2^2$ | 둥근 마름모 |

## 성질의 종합 비교

| 성질 | 능형 | 라쏘 | 엘라스틱넷 |
|---|---|---|---|
| 변수선택 | 없음 | 있음 | 있음 |
| 정확한 0 | 결코 없음 | 있음 | 있음 |
| 축소 유형 | 비례 | 평행이동(연성 문턱) | 결합 |
| 다중공선성 처리 | 좋음(계수 분배) | 나쁨(하나만 선택) | 좋음(그룹 효과) |
| 해의 유일성 | 언제나 ($\lambda > 0$) | $p > n$이면 아님 | 언제나 ($\alpha < 1$) |
| 닫힌 형태 해 | 있음 | 직교 $\mathbf{X}$에서만 | 없음 |
| 최대 선택 변수 수 ($p > n$) | 전부 $p$개 | 최대 $n$개 | 제한 없음 |
| 베이즈 사전분포 | 가우스 $N(0, \tau^2)$ | Laplace$(0, b)$ | 혼합 |
| 계산 | 행렬 역변환 | 좌표하강 | 좌표하강 |
| 초모수 | $\lambda$ | $\lambda$ | $\lambda, \alpha$ |
| 유효자유도 공식 | $\sum d_j^2/(d_j^2 + \lambda)$ | 0이 아닌 계수의 개수 | 근사 |

## 계수 경로의 거동

**능형 경로.** 모든 $\lambda > 0$에서 모든 계수가 0이 아니다. 큰 $\lambda$에서 0 근처에서 출발해 OLS 값으로 연속적으로 커진다. 경로가 매끄러운 곡선이며 어떤 계수도 정확히 0에 닿지 않는다.

**라쏘 경로.** $\lambda$가 줄면서 계수가 하나씩 모형에 들어온다. 경로가 조각별 선형이다. 각 꺾인점에서 새 변수가 활성집합에 들어오거나 (드물게) 빠진다.

**엘라스틱넷 경로.** 계수가 점진적으로 들어온다는 점에서 라쏘와 비슷하지만 L2 성분 때문에 경로가 더 매끄럽다. 상관된 변수가 함께 들어오는 경향이 있고, 자료의 교란에 대해 더 안정적이다.

## 직교 설계에서의 축소 거동

$\mathbf{X}^\top\mathbf{X} = n\mathbf{I}$일 때 세 방법 모두 닫힌 형태 해를 가지며 축소 방식이 드러난다.

$$
\hat{\beta}_j^{\text{ridge}} = \frac{\hat{\beta}_j^{\text{OLS}}}{1 + \lambda}, \qquad
\hat{\beta}_j^{\text{lasso}} = \text{sign}(\hat{\beta}_j^{\text{OLS}})\,\max\bigl(|\hat{\beta}_j^{\text{OLS}}| - \lambda,\; 0\bigr)
$$

$$
\hat{\beta}_j^{\text{EN}} = \frac{\text{sign}(\hat{\beta}_j^{\text{OLS}})\,\max\bigl(|\hat{\beta}_j^{\text{OLS}}| - \alpha\lambda,\; 0\bigr)}{1 + (1-\alpha)\lambda}
$$

능형은 상수로 나누고(비례 축소), 라쏘는 상수를 빼고 0에서 자른다(평행이동 축소). 엘라스틱넷은 $\alpha\lambda$를 뺀 뒤 $1 + (1-\alpha)\lambda$로 나눈다.

## 편향-분산 프로파일

**능형.** 모든 계수에 중간 정도의 편향. 분산 감소는 $\mathbf{X}^\top\mathbf{X}$의 작은 고유벡터 방향에서 가장 크다.

**라쏘.** 제거된 계수에는 편향이 없다(참값이 0이면 정확하다). 그러나 남은 계수는 연성 문턱 때문에 편향된다. 분산 감소는 잡음이 큰 계수 추정을 제거하는 데서 온다.

**엘라스틱넷.** 둘의 프로파일을 결합한다. L2 성분이 모든 계수의 분산을 줄이고 L1 성분이 무관한 것을 제거한다.

## 상황별 성능

!!! note "보편적으로 최선인 방법은 없다"
    어떤 방법도 모든 상황에서 우월하지 않다. 최선의 선택은 참 자료생성과정, 설명변수의 상관 구조, $p/n$ 비, 분석 목표에 달려 있다.

**상황 1: 작은 효과가 많고 희소성이 없다.** 모든 변수가 조금씩 기여한다. 모든 변수를 남기고 축소를 최적으로 분배하는 **능형**이 가장 좋다.

**상황 2: 큰 효과가 몇 개뿐이고 실제로 희소하다.** 소수의 변수만 유의미하고 강한 상관이 없다. 유의미한 변수를 찾고 나머지를 0으로 만드는 **라쏘**가 가장 좋다.

**상황 3: 집단을 이루는 효과와 희소성.** 상관된 변수 집단 여럿이 유의미하다. 집단을 함께 선택하면서 희소성을 유지하는 **엘라스틱넷**이 가장 좋다.

**상황 4: 고차원, $p \gg n$.** 라쏘의 $n$개 제한이 실제로 작동할 수 있으므로 **엘라스틱넷**이 선호된다.

## 계산비용

| 방법 | $\lambda$당 비용 | 경로 ($M$개 $\lambda$) | LOOCV 지름길 |
|---|---|---|---|
| 능형 | $O(p^3)$ 또는 $O(np^2)$ | $O(Mp^3)$ | 있음(닫힌 형태) |
| 라쏘 | $O(np \times \text{반복})$ | 온기 시작으로 $O(Mnp)$ | 없음 |
| 엘라스틱넷 | $O(np \times \text{반복})$ | 온기 시작으로 $O(Mnp)$ | 없음 |

## 요약

능형회귀는 매끄럽고 희소하지 않은 축소로 다중공선성을 잘 다룬다. 라쏘는 L1 벌점으로 희소해를 주지만 상관된 변수에서 불안정하고 $p > n$일 때 $n$개로 제한된다. 엘라스틱넷은 두 벌점을 결합해 희소성, 상관 변수의 집단화, 해의 유일성, 선택 개수 제한 없음을 모두 달성한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
직교 설계($\mathbf{X}^\top\mathbf{X} = n\mathbf{I}$)에서 능형과 라쏘의 닫힌 형태 해를 유도하라. 라쏘는 정확한 0을 만들고 능형은 만들지 못하는 이유를 기하적으로 설명하라.

</div>

??? success "풀이"
    **유도.** 직교 설계에서 목적함수가 좌표별로 분리된다. $z_j = \frac{1}{n}\mathbf{x}_j^\top\mathbf{y} = \hat\beta_j^{\text{OLS}}$라 두면 각 좌표의 문제는

    $$
    \min_{\beta_j}\left\{\tfrac12(\beta_j - z_j)^2 + \lambda P(\beta_j)\right\}
    $$

    이다.

    **능형** ($P = \frac12\beta_j^2$): 미분하여 $\beta_j - z_j + \lambda\beta_j = 0$이므로 $\hat\beta_j = z_j/(1+\lambda)$.

    **라쏘** ($P = |\beta_j|$): $\beta_j \neq 0$이면 $\beta_j - z_j + \lambda\,\text{sign}(\beta_j) = 0$이므로 $\hat\beta_j = z_j - \lambda\,\text{sign}(\beta_j)$. $\beta_j = 0$이 최적일 조건은 부분미분에서 $|z_j| \le \lambda$. 둘을 합치면 연성 문턱이 된다.

    **기하적 이유.** 핵심은 $\beta_j = 0$에서 벌점의 도함수다.

    | 벌점 | $\beta_j \to 0^+$에서의 도함수 |
    |:---|---:|
    | L2 ($\frac12\beta_j^2$) | $\beta_j \to 0$ |
    | L1 ($|\beta_j|$) | $1$ (0이 아니다) |

    L2 벌점의 도함수가 0에서 사라지므로, $z_j$가 아무리 작아도 0에서 벗어나는 것이 언제나 이득이다. L1 벌점은 0에서도 기울기 $\lambda$를 유지하므로, 자료가 주는 이득 $|z_j|$가 $\lambda$를 넘지 못하면 0에 머무는 것이 최적이다.

    제약 형태로 보면 이는 [기하적 해석](../lasso/geometry.md)에서 본 "꼭짓점" 논증과 같은 이야기다. 매끄러운 구는 접선 조건을 **등식**으로, 뾰족한 마름모는 **부등식**으로 만든다.

<div class="drillbox" markdown>

**연습문제 2.**
$y = 3x_1 - 2x_2 + 0.5x_3 + \varepsilon$($\varepsilon \sim N(0,1)$)에서 $n = 100$개를 생성하고 잡음변수 17개($x_4, \ldots, x_{20}$)를 더하라.

**(a)** OLS, 능형, 라쏘, 엘라스틱넷을 적합하고 계수 추정값을 비교하라.

**(b)** 어느 방법이 참 변수 3개를 올바르게 식별하는가?

**(c)** 5-겹 교차검증으로 각 방법의 최적 $\lambda$를 고르고 검정 MSE를 보고하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from sklearn.linear_model import (LinearRegression, RidgeCV,
                                      LassoCV, ElasticNetCV)
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(0)
    n, p = 100, 20
    X = rng.normal(size=(n, p)); X -= X.mean(0); X /= X.std(0)
    beta = np.zeros(p); beta[:3] = [3, -2, 0.5]
    y = X @ beta + rng.normal(0, 1, n)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    models = {
        'OLS':   LinearRegression(),
        'Ridge': RidgeCV(alphas=np.logspace(-3, 3, 100)),
        'Lasso': LassoCV(cv=5, random_state=0, max_iter=50000),
        'ENet':  ElasticNetCV(l1_ratio=[.1,.5,.7,.9,.95,1], cv=5,
                              random_state=0, max_iter=50000),
    }
    for name, m in models.items():
        m.fit(Xtr, ytr)
        mse = ((yte - m.predict(Xte))**2).mean()
        nz = (np.abs(m.coef_) > 1e-8).sum()
        print(name, mse, nz)
    ```

    출력:

    ```
    OLS 1.7921322992720217 20
    Ridge 1.7989353465017053 20
    Lasso 1.259628016657471 9
    ENet 1.259628016657471 9
    ```

    **(a), (b)** 기대되는 양상은 다음과 같다.

    | 방법 | 0이 아닌 계수 | 참 변수 3개 포함 | $x_3$($\beta = 0.5$) |
    |:---|---:|:---|:---|
    | OLS | 20 | 포함(모두 포함하므로) | 추정값에 잡음이 크다 |
    | 능형 | 20 | 포함 | 축소되어 작다 |
    | 라쏘 | 소수 | 대체로 포함 | **놓칠 수 있다** |
    | 엘라스틱넷 | 라쏘보다 조금 많음 | 대체로 포함 | 라쏘보다 잘 잡는다 |

    **$x_3$의 계수 $0.5$가 관건이다.** 잡음 표준편차가 1이고 $n = 100$이므로 그 계수의 표준오차가 약 $0.1$이다. 신호 대 잡음비가 5로 검출은 가능하지만, 라쏘의 문턱 $\lambda$가 $0.5$에 가까우면 함께 잘려 나간다.

    **(c)** 검정 MSE는 네 방법이 대체로 비슷하며, 잡음변수가 17개뿐이라 OLS도 크게 나쁘지 않다. **차이가 극적으로 벌어지려면 $p$가 $n$에 가까워야 한다.** $p = 20$, $n = 100$은 $p/n = 0.2$로 OLS가 아직 견딜 만한 영역이다([과적합과 편향-분산 절충](../motivation/overfitting.md) 연습문제 3 참조).

    이 실험의 교훈은 **정칙화의 이득이 $p/n$과 신호의 희소성에 강하게 의존한다**는 것이다. 이 설정에서 "정칙화가 별로 도움이 안 된다"는 결과가 나와도 그것이 정칙화의 실패는 아니다.

<div class="drillbox" markdown>

**연습문제 3.**
**금융 응용.** Fama-French 요인과 거시경제 변수로 월별 주식수익률을 예측하는 상황을 생각하자. 예측변수 $p = 50$개, 월별 관측 $n = 120$개인 자료를 모의생성한다.

**(a)** 이 상황에서 정칙화가 필수적인 이유는 무엇인가?

**(b)** 이동창 교차검증(60개월로 훈련, 다음 달 예측, 한 달씩 이동)으로 OLS, 능형회귀, 라쏘, 엘라스틱넷의 표본외 $R^2$를 비교하라.

**(c)** 라쏘가 고르는 예측변수는 무엇인가? 이동창을 옮겨도 안정적인가?

</div>
