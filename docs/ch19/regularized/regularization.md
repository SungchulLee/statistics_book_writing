# 로지스틱 회귀의 L1·L2 정칙화

## 동기

설명변수의 개수 $p$가 표본크기 $n$에 비해 크면 로지스틱 회귀의 최대가능도 추정량은 과적합할 수
있다. 훈련 범주를 최대한 날카롭게 가르기 위해 계수가 커지고, 그 결과 예측확률이 극단으로 몰려
일반화 성능이 나빠진다. **정칙화**는 로그가능도에 벌점을 더해 계수를 0 쪽으로 축소함으로써,
편향을 조금 늘리는 대가로 분산을 크게 줄인다.

## 벌점 로그가능도

정칙화 추정량은

$$
\ell_{\text{pen}}(\boldsymbol{\theta})
= \ell(\boldsymbol{\theta}) - \lambda\,\Omega(\boldsymbol{\theta})
$$

를 최대화한다. 여기서 $\ell(\boldsymbol{\theta})$는 벌점 없는 로그가능도, $\lambda \ge 0$은
벌점의 강도, $\Omega(\boldsymbol{\theta})$는 벌점함수다. 절편 $\theta_0$에는 **벌점을 주지
않고** 기울기 계수 $\theta_1, \ldots, \theta_{p-1}$만 축소한다.

## L2 벌점(능형 로지스틱 회귀)

L2 벌점은 계수 제곱의 합이다.

$$
\Omega_{\text{L2}}(\boldsymbol{\theta})
= \frac{1}{2}\sum_{j=1}^{p-1}\theta_j^2
= \frac{1}{2}\lVert\boldsymbol{\theta}_{-0}\rVert_2^2
$$

벌점 로그가능도는

$$
\ell_{\text{ridge}}(\boldsymbol{\theta})
= \sum_{i=1}^{n}\bigl[y_i\,\mathbf{x}_i^T\boldsymbol{\theta} - \log(1+e^{\mathbf{x}_i^T\boldsymbol{\theta}})\bigr] - \frac{\lambda}{2}\lVert\boldsymbol{\theta}_{-0}\rVert_2^2
$$

가 된다. 능형 정칙화는 모든 계수를 0 쪽으로 축소하지만 어느 것도 정확히 0으로 만들지는 않는다.
설명변수들이 상관되어 있을 때(다중공선성) 추정치를 안정화하므로 특히 유용하다.

### 베이즈 해석

L2 벌점은 각 계수에 대한 **정규 사전분포** $\theta_j \sim \mathcal{N}(0, 1/\lambda)$에
대응한다. 따라서 능형 추정량은 이 사전분포 아래의 **사후최빈값(MAP)** 추정치다.

## L1 벌점(라쏘 로지스틱 회귀)

L1 벌점은 절댓값의 합이다.

$$
\Omega_{\text{L1}}(\boldsymbol{\theta})
= \sum_{j=1}^{p-1}|\theta_j|
= \lVert\boldsymbol{\theta}_{-0}\rVert_1
$$

벌점 로그가능도는

$$
\ell_{\text{lasso}}(\boldsymbol{\theta})
= \sum_{i=1}^{n}\bigl[y_i\,\mathbf{x}_i^T\boldsymbol{\theta} - \log(1+e^{\mathbf{x}_i^T\boldsymbol{\theta}})\bigr] - \lambda\,\lVert\boldsymbol{\theta}_{-0}\rVert_1
$$

이다. L1 벌점은 **희소한** 해를 만든다. $\lambda$가 충분히 크면 일부 계수가 정확히 0이 된다.
그래서 L1 로지스틱 회귀는 분류기이자 동시에 변수선택기다.

### 베이즈 해석

L1 벌점은 독립인 **라플라스 사전분포** $\theta_j \sim \text{Laplace}(0, 1/\lambda)$에 대응한다.
0에서의 뾰족한 봉우리와 두꺼운 꼬리가 희소성을 유도한다.

## 엘라스틱넷

엘라스틱넷은 두 벌점을 결합한다.

$$
\Omega_{\text{EN}}(\boldsymbol{\theta})
= \alpha\,\lVert\boldsymbol{\theta}_{-0}\rVert_1 + \frac{1-\alpha}{2}\lVert\boldsymbol{\theta}_{-0}\rVert_2^2
$$

여기서 $\alpha \in [0,1]$이 배합을 조절한다. $\alpha = 1$이면 라쏘, $\alpha = 0$이면 능형이
된다. 엘라스틱넷은 L1의 희소성을 물려받으면서 상관된 설명변수를 순수 라쏘보다 잘 다룬다.

## scikit-learn의 C 모수

scikit-learn의 `LogisticRegression`은 벌점 강도를 $C = 1/\lambda$로 모수화한다. $C$가
**클수록** 정칙화가 약하고(벌점 없는 MLE에 가깝고), **작을수록** 정칙화가 강하다.

| `LogisticRegression` 인자 | 대응 |
|---|---|
| `penalty='l2', C=1.0` | $\lambda = 1$인 능형 |
| `penalty='l1', C=0.1, solver='liblinear'` | $\lambda = 10$인 라쏘 |
| `penalty='elasticnet', l1_ratio=0.5, C=1.0, solver='saga'` | $\alpha = 0.5$, $\lambda = 1$인 엘라스틱넷 |

!!! tip "특성 표준화"
    벌점은 모든 계수를 동등하게 취급하므로, 정칙화 모형을 적합하기 전에 설명변수를
    **표준화**(평균 0, 분산 1)해야 한다. 그러지 않으면 척도가 큰 특성의 계수가 더 심하게
    벌을 받아 결과가 왜곡된다.

## 최적화

L2 벌점 로그가능도는 여전히 강오목이므로 헤세행렬에 $\lambda\mathbf{I}$를 더해 뉴턴-랩슨이나
IRLS로 최대화할 수 있다. L1 벌점 목적함수는 $\theta_j = 0$에서 미분 불가능하므로 전용
알고리즘이 필요하다.

- **좌표하강:** 연성 문턱을 이용해 계수를 하나씩 갱신한다. `glmnet`과 scikit-learn의 `saga`
  해법기가 쓰는 방식이다.
- **근접기울기법:** 매끄러운 부분에 대한 기울기 단계와 L1 항에 대한 근접 연산자를 결합한다.

## 정칙화 경로

$\lambda$ 격자를 큰 값에서 작은 값으로 훑으며 모형을 적합하면 **정칙화 경로**가 나온다.
$\lambda$가 줄어들면서 각 계수가 곡선을 그린다. L2 경로는 매끄럽고, L1 경로는 계수가 하나씩
모형에 들어오는 꺾인 모양이다.

!!! warning "로지스틱 라쏘 경로는 조각별 선형이 아니다"
    최소제곱 라쏘의 경로는 조각별 선형이지만, **로지스틱** 라쏘의 경로는 그렇지 않다. 조각별
    선형성은 손실이 조각별 이차이고 벌점이 조각별 선형일 때에만 성립한다(Rosset & Zhu, 2007).
    로지스틱 손실은 이차가 아니므로, 활성집합이 변하지 않는 구간 안에서도 계수가 $\lambda$에
    대해 **곡선**을 그린다. 연습문제 5에서 수치로 확인한다. 꺾이는 지점(활성집합이 바뀌는 곳)이
    있다는 점만 두 경우가 같다.

최적 $\lambda$는 교차검증으로 고른다.

??? example "교차검증으로 C 고르기"
    $n = 500$, $p = 20$인 자료에 5-겹 교차검증을 적용하면,

    | $C$ | 평균 CV 정확도 | 0이 아닌 계수 |
    |---|---|---|
    | 0.01 | 0.72 | 3 |
    | 0.1 | 0.81 | 8 |
    | 1.0 | 0.84 | 15 |
    | 10.0 | 0.83 | 19 |
    | 100.0 | 0.82 | 20 |

    CV 정확도는 $C = 1.0$에서 가장 높다. 이보다 $C$를 키우면 설명변수만 늘어날 뿐 성능은
    개선되지 않고 오히려 조금 나빠진다.


## 연습문제

**연습문제 1.**
L2 벌점이 정규 사전분포 $\theta_j \sim \mathcal{N}(0, 1/\lambda)$의 MAP 추정에 대응함을
유도하라. L1 벌점의 경우 라플라스 척도모수는 얼마인가?

??? success "연습문제 1 풀이"

    사전분포 $\pi(\boldsymbol{\theta})$ 아래의 로그사후는

    $$
    \log p(\boldsymbol{\theta}\mid \text{data}) = \ell(\boldsymbol{\theta}) + \log\pi(\boldsymbol{\theta}) + \text{const}
    $$

    이다. MAP 추정은 이를 최대화한다.

    **정규 사전분포.** $\theta_j \sim \mathcal{N}(0, \sigma_0^2)$이면

    $$
    \log\pi(\boldsymbol{\theta}) = -\frac{1}{2\sigma_0^2}\sum_j \theta_j^2 + \text{const}
    $$

    이므로 벌점 $-\frac{\lambda}{2}\sum_j\theta_j^2$과 비교하면
    $\frac{1}{\sigma_0^2} = \lambda$, 즉 $\sigma_0^2 = 1/\lambda$이다.

    **라플라스 사전분포.** $\theta_j \sim \text{Laplace}(0, b)$의 밀도는
    $\frac{1}{2b}e^{-|\theta_j|/b}$이므로

    $$
    \log\pi(\boldsymbol{\theta}) = -\frac{1}{b}\sum_j|\theta_j| + \text{const}
    $$

    이고, 벌점 $-\lambda\sum_j|\theta_j|$과 비교하면 $b = 1/\lambda$이다.

    !!! note "배율에 주의"
        이 대응은 목적함수를 **표본 크기로 나누지 않았을 때** 성립한다. 18장처럼 손실을
        $\frac{1}{2n}\|y-X\beta\|^2$로 쓰면 $\lambda$의 의미가 달라지고 척도모수도 그에 맞게
        바뀐다. 사전분포의 척도를 읽을 때는 반드시 목적함수의 배율을 먼저 확인하라.

    두 경우 모두 $\lambda \to 0$이면 사전분포가 평평해져 MLE로 돌아가고, $\lambda \to \infty$
    이면 사전분포가 0에 집중되어 모든 계수가 0이 된다. $\square$

---

**연습문제 2.**
왜 절편에는 벌점을 주지 않는가? 절편에도 벌점을 주면 무슨 일이 일어나는가?

??? success "연습문제 2 풀이"

    두 가지 이유가 있다.

    **첫째, 평행이동 불변성.** 반응변수의 기저율은 모형이 반드시 맞혀야 하는 값이지 축소할
    대상이 아니다. 절편에 벌점을 주면 예측확률이 $0.5$ 쪽으로 끌려간다. $\theta_0 \to 0$이면
    $\sigma(0) = 0.5$이기 때문이다. 사건이 드문 자료($\bar y = 0.02$)에서 이는 심각한 편향이다.

    **둘째, 표준화와의 정합성.** 설명변수를 표준화하면 각 $x_j$의 평균이 0이 되어, 절편은
    "평균적인 관측치의 로그오즈"라는 자연스러운 의미를 갖는다. 이 값은 자료의 단위나 중심화
    방식과 무관하게 정해져야 하는데, 벌점을 주면 다른 계수들이 축소되는 정도에 따라 절편까지
    끌려다니게 된다.

    **절편에 벌점을 주면:** $\lambda \to \infty$일 때 모든 계수가 0이 되어 모든 예측확률이
    정확히 $0.5$가 된다. 벌점을 주지 않으면 $\hat\theta_0 \to \operatorname{logit}(\bar y)$가
    되어 영모형, 즉 절편만 있는 모형으로 수렴한다. 후자가 옳은 극한이다.

    **scikit-learn에서는 해법기에 따라 다르다.** `lbfgs`, `newton-cg`, `sag`, `saga`는
    관례대로 절편에 벌점을 주지 않지만, **`liblinear`는 절편에도 벌점을 준다**(절편을 합성
    특성으로 다루기 때문이며, `intercept_scaling` 인자가 그 영향을 완화하기 위해 있다).
    $\bar y = 0.115$인 자료에 $C = 10^{-6}$(사실상 무한대의 벌점)을 걸고 확인해 보면,

    | 해법기 | $\hat\theta_0$ |
    |---|---|
    | `lbfgs` | $-2.0406$ |
    | `liblinear` | $-0.0002$ |

    이고 $\operatorname{logit}(\bar y) = -2.0407$이다. `lbfgs`는 올바른 극한인 영모형으로
    수렴하지만 `liblinear`는 절편까지 0으로 눌러 모든 예측확률을 $0.5$로 만든다. L1 벌점을
    쓸 때 `liblinear`를 고르는 경우가 많으므로 특히 주의해야 한다. $\square$

---

**연습문제 3.**
scikit-learn의 $C$와 이 절의 $\lambda$ 사이의 관계를 유도하라. $n$이 커지면 같은 $C$가
정칙화의 강도 면에서 어떻게 달라지는가?

??? success "연습문제 3 풀이"

    scikit-learn은

    $$
    \min_{\boldsymbol{\theta}}\ \frac{1}{2}\lVert\boldsymbol{\theta}\rVert_2^2 + C\sum_{i=1}^n \text{loss}_i(\boldsymbol{\theta})
    $$

    를 최소화한다. 양변을 $C$로 나누면

    $$
    \min_{\boldsymbol{\theta}}\ \sum_{i=1}^n \text{loss}_i(\boldsymbol{\theta}) + \frac{1}{2C}\lVert\boldsymbol{\theta}\rVert_2^2
    $$

    이 되어 이 절의 표기와 비교하면 $\lambda = 1/C$이다.

    **$n$과의 관계가 중요하다.** 위 식에서 손실은 **합**이지 평균이 아니다. 따라서 $n$이
    커지면 손실 항이 $n$에 비례해 커지는 반면 벌점은 그대로다. 즉 같은 $C$라도 $n$이 크면
    상대적으로 정칙화가 약해진다.

    실무적 함의는 두 가지다.

    - 표본 크기가 다른 자료 사이에서 $C$ 값을 그대로 옮겨 쓰면 안 된다. $n$이 10배 커지면
      같은 상대적 정칙화를 얻기 위해 $C$를 10분의 1로 줄여야 한다.
    - 교차검증에서 겹의 크기가 전체 자료와 다르므로 미세한 불일치가 생긴다. 겹마다 훈련
      표본이 $n(K-1)/K$이므로 최적 $C$가 전체 자료에 대한 최적값과 조금 다르다. $K$가 크면
      무시할 만하다.

    `glmnet`은 손실을 평균으로 쓰므로 이 문제가 없다. 두 도구의 $\lambda$를 비교할 때는
    $n$ 배율을 반드시 맞춰야 한다. $\square$

---

**연습문제 4.**
L2 벌점 로지스틱 회귀에서는 자료가 완전히 분리 가능해도 해가 존재하고 유일함을 보여라.

??? success "연습문제 4 풀이"

    벌점 목적함수를 최소화 형태로 쓰면

    $$
    f(\boldsymbol{\theta}) = -\ell(\boldsymbol{\theta}) + \frac{\lambda}{2}\lVert\boldsymbol{\theta}_{-0}\rVert_2^2
    $$

    이다.

    **존재성.** $-\ell(\boldsymbol{\theta}) \ge 0$이고
    $\frac{\lambda}{2}\lVert\boldsymbol{\theta}_{-0}\rVert^2 \to \infty$이므로,
    $\lVert\boldsymbol{\theta}_{-0}\rVert \to \infty$일 때 $f \to \infty$다. 즉 $f$는
    강제적(coercive)이고 연속이므로 최소점이 존재한다. 분리 가능한 자료에서 벌점 없는 MLE가
    존재하지 않았던 이유가 $\lVert\boldsymbol{\theta}\rVert \to \infty$를 막을 것이 없었기
    때문인데, 벌점이 정확히 그 역할을 한다.

    **유일성.** $-\ell$의 헤세행렬은 $A^\top B A$로 양반정치이고, 벌점 항의 헤세행렬은
    $\lambda I$(절편 성분은 0)이다. 기울기 좌표에 대해서는 합이 양정치이므로 그 방향으로
    강볼록이다. 절편이 벌점을 받지 않아 엄밀한 유일성에는 $A$의 첫 열에 대한 조건이 필요하지만,
    절편 열이 다른 열들과 일차독립인 통상적인 경우에는 해가 유일하다.

    이것이 실무적으로 매우 중요한 이유는, 분리는 드문 사건이 아니기 때문이다. $p$가 $n$에
    가깝거나 범주형 변수의 수준이 세분화되어 있으면 흔히 발생한다. 작은 $\lambda$ 하나만 넣어도
    문제가 사라진다. $\square$

---

**연습문제 5.**
로지스틱 라쏘 경로가 조각별 선형이 **아님**을 수치로 확인하라. 활성집합이 변하지 않는
$\lambda$ 구간에서 계수의 증분이 일정한지 살펴보라.

??? success "연습문제 5 풀이"

    ```python
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(5)
    n, p = 200, 3
    X = rng.normal(size=(n, p))
    z = 1.5 * X[:, 0] - 1.0 * X[:, 1]          # x3 is noise
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    for lam in np.arange(0.04, 0.17, 0.02):
        m = LogisticRegression(penalty='l1', C=1 / (lam * n),
                               solver='liblinear', max_iter=10000,
                               tol=1e-10).fit(X, y)
        print(f"lambda={lam:.2f}  {np.round(m.coef_[0], 5)}")
    ```

    | $\lambda$ | $\hat\theta_1$ | $\hat\theta_2$ | $\Delta\hat\theta_1$ |
    |---|---|---|---|
    | 0.04 | $0.91379$ | $-0.59897$ | |
    | 0.06 | $0.75432$ | $-0.47237$ | $-0.15947$ |
    | 0.08 | $0.61828$ | $-0.36294$ | $-0.13604$ |
    | 0.10 | $0.49857$ | $-0.26526$ | $-0.11971$ |
    | 0.12 | $0.39045$ | $-0.17568$ | $-0.10812$ |
    | 0.14 | $0.29051$ | $-0.09154$ | $-0.09994$ |
    | 0.16 | $0.19604$ | $-0.01074$ | $-0.09447$ |

    이 구간 내내 활성집합은 $\{x_1, x_2\}$로 고정되어 있다($x_3$은 $\lambda \ge 0.04$에서 계속
    0이다). 경로가 조각별 선형이라면 $\lambda$ 간격이 일정하므로 증분
    $\Delta\hat\theta_1$도 일정해야 한다. 그러나 실제 증분은 $-0.159$에서 $-0.094$로 단조롭게
    줄어든다. 경로가 **휘어 있다.**

    최소제곱 라쏘라면 이 구간에서 증분이 정확히 일정하다. 차이는 손실함수에서 온다. 제곱손실의
    KKT 조건은 활성 좌표에 대해 $\lambda$에 관한 **선형계**를 주지만, 로지스틱 손실의 조건은
    $\hat p_i$를 통해 $\boldsymbol{\theta}$에 비선형으로 의존한다.

    **실무적 함의:** LARS처럼 꺾인 지점만 계산해 경로 전체를 얻는 알고리즘은 로지스틱 회귀에
    쓸 수 없다. `glmnet`이 $\lambda$ 격자를 잡고 온기 시작으로 각 점을 따로 푸는 이유가
    이것이다. $\square$
