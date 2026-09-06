# 정칙화 로지스틱 회귀 실습


## 개요

정칙화 로지스틱 회귀는 로그가능도에 벌점항을 더해 과적합을 막고 일반화 성능을 높인다. 특히
특성의 개수가 표본크기에 비해 클 때 효과가 크다. 이 절에서는 L2(능형), L1(라쏘), 엘라스틱넷
벌점과 그것이 계수 추정에 미치는 영향, 그리고 scikit-learn에서의 실제 구현을 다룬다.

## 벌점 없는 로지스틱 회귀

표준 로지스틱 회귀 모형은 로그가능도

$$
\ell(\boldsymbol\beta) = \sum_{i=1}^{n}
  \bigl[y_i \log \hat{p}_i + (1 - y_i)\log(1 - \hat{p}_i)\bigr]
$$

를 최대화한다. 여기서 $\hat{p}_i = \sigma(\mathbf{x}_i^T \boldsymbol\beta)$이고
$\sigma(z) = 1/(1+e^{-z})$이다.

## L2 정칙화(능형)

능형 로지스틱 회귀는 제곱 노름 벌점을 더한다.

$$
\hat{\boldsymbol\beta}_{\text{ridge}}
  = \arg\max_{\boldsymbol\beta}\;
    \ell(\boldsymbol\beta) - \frac{\lambda}{2}\|\boldsymbol\beta\|_2^2
$$

$C = 1/\lambda$를 쓰는 scikit-learn의 표기로는 동등하게

$$
\hat{\boldsymbol\beta}_{\text{ridge}}
  = \arg\min_{\boldsymbol\beta}\;
    -\ell(\boldsymbol\beta) + \frac{1}{2C}\|\boldsymbol\beta\|_2^2
$$

이다. L2 벌점은 모든 계수를 0 쪽으로 축소하지만 어느 것도 정확히 0으로 만들지 않는다.

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

np.random.seed(42)
n, p = 200, 20
X = np.random.randn(n, p)
true_beta = np.zeros(p)
true_beta[:5] = [1.5, -1.0, 0.8, -0.5, 0.3]
logit = X @ true_beta
prob = 1 / (1 + np.exp(-logit))
y = np.random.binomial(1, prob)

ridge_model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                                  max_iter=1000)
ridge_model.fit(X, y)
print("Ridge coefficients:", np.round(ridge_model.coef_[0], 3))
```

앞 다섯 개 계수는 $(1.346,\ -1.162,\ 0.955,\ -0.593,\ -0.031)$이다. 참값
$(1.5, -1.0, 0.8, -0.5, 0.3)$과 비교하면 강한 신호 네 개는 잘 잡아냈지만 가장 약한 신호
$0.3$은 부호까지 틀렸다. 잡음변수 15개의 계수는 절댓값이 최대 $0.398$로, 0이 아니지만 작다.

## L1 정칙화(라쏘)

라쏘 로지스틱 회귀는 제곱 벌점 대신 절댓값 노름을 쓴다.

$$
\hat{\boldsymbol\beta}_{\text{lasso}}
  = \arg\min_{\boldsymbol\beta}\;
    -\ell(\boldsymbol\beta) + \frac{1}{C}\|\boldsymbol\beta\|_1
$$

L1 벌점은 **희소성**을 유도한다. 충분히 작은 계수는 정확히 0으로 밀려나 자동으로 변수선택이
이루어진다.

```python
lasso_model = LogisticRegression(penalty='l1', C=1.0, solver='saga',
                                  max_iter=5000)
lasso_model.fit(X, y)
print("Lasso coefficients:", np.round(lasso_model.coef_[0], 3))
print(f"Non-zero coefficients: {np.sum(lasso_model.coef_[0] != 0)} / {p}")
```

앞 다섯 개는 $(1.340,\ -1.146,\ 0.940,\ -0.579,\ 0)$으로, 가장 약한 신호가 정확히 0이 되었다.
전체로는 20개 중 **17개**가 0이 아니다. 즉 $C = 1.0$에서는 아직 벌점이 약해 잡음변수 대부분이
살아남는다.

## 엘라스틱넷

엘라스틱넷은 배합모수 $\alpha \in [0,1]$(scikit-learn에서는 `l1_ratio`)로 L1과 L2 벌점을
결합한다.

$$
\text{Penalty} = \frac{1-\alpha}{2}\|\boldsymbol\beta\|_2^2 + \alpha\,\|\boldsymbol\beta\|_1
$$

$\alpha = 0$이면 능형, $\alpha = 1$이면 라쏘가 된다. 상관된 특성 집단이 있을 때 유용하다. L1
단독이라면 각 집단에서 하나만 고르지만, L2 성분이 상관된 설명변수들끼리 가중치를 나누어 갖도록
유도하기 때문이다.

```python
enet_model = LogisticRegression(penalty='elasticnet', C=1.0,
                                 solver='saga', l1_ratio=0.5,
                                 max_iter=5000)
enet_model.fit(X, y)
print("Elastic Net coefficients:", np.round(enet_model.coef_[0], 3))
```

앞 다섯 개는 $(1.342,\ -1.153,\ 0.947,\ -0.585,\ -0.012)$이고 0이 아닌 계수는 19개다. 예상대로
능형(20개)과 라쏘(17개) 사이에 놓인다.

## 정칙화 강도의 영향

$C$가 커지면(정칙화가 약해지면) 추정치가 벌점 없는 MLE에 가까워지고, $C$가 작아지면
(정칙화가 강해지면) 계수가 0 쪽으로 축소된다.

```python
import matplotlib.pyplot as plt

C_values = np.logspace(-3, 3, 50)
coefs = []

for C in C_values:
    model = LogisticRegression(penalty='l2', C=C, solver='lbfgs',
                                max_iter=2000)
    model.fit(X, y)
    coefs.append(model.coef_[0])

coefs = np.array(coefs)

plt.figure(figsize=(10, 5))
for j in range(p):
    plt.plot(np.log10(C_values), coefs[:, j],
             linewidth=2 if j < 5 else 0.8,
             alpha=1.0 if j < 5 else 0.3)
plt.xlabel('log10(C)')
plt.ylabel('Coefficient value')
plt.title('Ridge Logistic Regression: Coefficient Paths')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
```

## 교차검증으로 C 조율하기

scikit-learn은 $C$ 격자 위에서 교차검증을 수행하는 `LogisticRegressionCV`를 제공한다.

```python
from sklearn.linear_model import LogisticRegressionCV

model_cv = LogisticRegressionCV(
    Cs=20, penalty='l2', cv=5, scoring='accuracy',
    solver='lbfgs', max_iter=2000
)
model_cv.fit(X, y)
print(f"Best C: {model_cv.C_[0]:.4f}")
print(f"Best CV accuracy: {model_cv.scores_[1].mean(axis=0).max():.4f}")
```

결과는 최적 $C = 1.6238$, 교차검증 정확도 $0.7450$이다.

!!! note "`scores_`의 키에 주의"
    `model_cv.scores_`는 범주 이름표를 키로 하는 딕셔너리다. 위 코드의 `scores_[1]`은 이름표가
    정수 `1`인 범주를 가리키므로, 이름표가 문자열(`'yes'`)이거나 다른 값이면 `KeyError`가 난다.
    범주에 무관하게 쓰려면 `next(iter(model_cv.scores_.values()))`를 쓰라. 이항 분류에서는
    두 범주의 점수 배열이 어차피 같다.

## 해석

- **능형(L2)**은 모든 특성이 기여할 것으로 기대되고 다중공선성이 있을 때 선호된다. 계수 추정을
  안정화한다.
- **라쏘(L1)**는 희소한 모형이 필요할 때 선호된다. 무관한 특성의 계수를 0으로 만들어 변수선택을
  수행한다.
- **엘라스틱넷**은 절충안으로, 특성이 상관되어 있으면서도 희소성이 필요할 때 유용하다.
- 정칙화 강도 $C$(또는 $\lambda = 1/C$)가 편향-분산 절충을 조절한다. $C$가 작을수록 편향은
  커지고 분산은 작아진다.

## 연습문제

**연습문제 1.**
$n = 200$, $p = 50$이고 처음 5개 특성만 참 계수가 0이 아닌 자료를 생성하라.
$C \in \{0.01, 0.1, 1.0, 10.0\}$에 대해 L1 정칙화 로지스틱 회귀를 적합하고, 각 $C$에서 0이
아닌 추정 계수의 개수를 보고하라.

??? success "연습문제 1 풀이"

    ```python
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    np.random.seed(42)
    n, p = 200, 50
    X = np.random.randn(n, p)
    true_beta = np.zeros(p)
    true_beta[:5] = [2.0, -1.5, 1.0, -0.8, 0.5]
    logit = X @ true_beta
    prob = 1 / (1 + np.exp(-logit))
    y = np.random.binomial(1, prob)

    for C in [0.01, 0.1, 1.0, 10.0]:
        model = LogisticRegression(penalty='l1', C=C, solver='saga',
                                    max_iter=5000)
        model.fit(X, y)
        nnz = np.sum(model.coef_[0] != 0)
        print(f"C = {C:5.2f}: {nnz} non-zero coefficients out of {p}")
    ```

    | $C$ | 0이 아닌 계수 | 그중 참 신호 | 그중 잡음 |
    |---|---|---|---|
    | 0.01 | 0 | 0 | 0 |
    | 0.10 | 4 | **4** | **0** |
    | 1.00 | 36 | 5 | 31 |
    | 10.00 | 48 | 5 | 43 |

    $C$가 커질수록(벌점이 약할수록) 0이 아닌 계수가 늘어난다. $C = 0.01$에서는 벌점이 너무 강해
    모든 계수가 0인 영모형이 되고, $C = 10$에서는 사실상 벌점 없는 적합에 가까워 잡음변수 43개가
    함께 들어온다.

    가장 흥미로운 지점은 $C = 0.1$이다. 정확히 4개가 선택되었고 **모두 참 신호이며 위양성이
    하나도 없다.** 놓친 하나는 가장 약한 신호 $\beta_5 = 0.5$다. 이것이 라쏘의 전형적인
    행동이다. 위양성을 0으로 유지할 만큼 벌점을 강하게 걸면 약한 참 신호도 함께 잘려 나간다.
    희소성과 검정력 사이의 절충은 피할 수 없다. $\square$

---

**연습문제 2.**
능형 벌점 $\|\boldsymbol\beta\|_2^2$이 베이즈 로지스틱 회귀에서 각 $\beta_j$에 독립인
$N(0, \sigma^2)$ 사전분포를 두는 것과 동등함을 보여라($\sigma^2 = C$).

??? success "연습문제 2 풀이"

    베이즈 로지스틱 회귀에서 사후분포는 가능도와 사전분포의 곱에 비례한다.

    $$
    p(\boldsymbol\beta \mid \mathbf{y})
      \propto \prod_{i=1}^n p(y_i \mid \mathbf{x}_i, \boldsymbol\beta)
      \;\cdot\; \prod_{j=1}^p \frac{1}{\sqrt{2\pi\sigma^2}}
      \exp\Bigl(-\frac{\beta_j^2}{2\sigma^2}\Bigr)
    $$

    로그를 취하고 상수를 무시하면

    $$
    \log p(\boldsymbol\beta \mid \mathbf{y})
      = \ell(\boldsymbol\beta) - \frac{1}{2\sigma^2}\sum_{j=1}^p \beta_j^2 + \text{const}
    $$

    이다. 이는 $\lambda = 1/\sigma^2$, 동등하게 $C = \sigma^2$인 능형 목적함수와 정확히 같다.
    따라서 로그사후를 최대화하는 것(MAP 추정)은 능형 로지스틱 회귀와 동일하다.

    !!! note "MAP는 사후분포의 요약 하나일 뿐이다"
        이 동등성은 **점추정**에 대한 것이다. 능형 추정치는 사후 최빈값이지 사후 평균이 아니며,
        정칙화 로지스틱 회귀는 사후분포의 폭에 대해 아무것도 알려 주지 않는다. 이것이
        정칙화 모형에서 표준오차와 신뢰구간이 그대로 유효하지 않은 이유다. 불확실성이
        필요하다면 실제로 사후분포를 표집(MCMC)하거나 붓스트랩을 써야 한다. $\square$

---

**연습문제 3.**
L1 벌점은 희소한 해를 만드는데 L2 벌점은 그렇지 않은 이유를 기하학적으로 설명하라.

??? success "연습문제 3 풀이"

    L1의 제약영역은 마름모(2차원에서는 집합
    $\{(\beta_1, \beta_2) : |\beta_1| + |\beta_2| \leq t\}$)로, 좌표축 위에 꼭짓점이 있다.
    L2의 제약영역은 원($\beta_1^2 + \beta_2^2 \leq t^2$)으로 어디서나 매끄럽다.

    로그가능도의 등고선은 대개 타원이다. 제약 최적해는 등고선이 제약영역에 처음 닿는 곳이다.
    마름모라면 이 접점이 좌표가 정확히 0인 꼭짓점에서 일어날 가능성이 훨씬 크다. 원이라면
    좌표축 위의 점에서 접하려면 특별한 정렬이 필요한데, 일반적인 자료에서 그럴 확률은 0이다.

    이 기하학적 논증은 고차원으로 일반화된다. $p$차원에서 L1 공은 $2^p$개의 꼭짓점을 가지며,
    접점은 대개 여러 좌표가 0이 되는 꼭짓점이나 면 위에 놓인다.

    **부분미분으로 본 같은 이야기.** $\beta_j = 0$에서 $|\beta_j|$의 부분미분은 구간
    $[-1, 1]$ 전체이므로, 최적성 조건 $|\partial\ell/\partial\beta_j| \le \lambda$를 만족하는
    한 $\beta_j = 0$이 최적으로 **유지된다.** 반면 $\beta_j^2$의 도함수는 $\beta_j = 0$에서
    정확히 0이므로, 기울기가 조금이라도 0이 아니면 곧바로 0에서 벗어난다. 꼭짓점의 뾰족함이
    곧 부분미분의 구간이다. $\square$

---

**연습문제 4.**
연습문제 1의 자료에 `LogisticRegressionCV`를 `penalty='l1'`, `solver='saga'`, 5-겹
교차검증으로 적용해 최적 $C$를 찾아라. 선택된 $C$와 그때의 교차검증 정확도를 보고하라.

??? success "연습문제 4 풀이"

    ```python
    from sklearn.linear_model import LogisticRegressionCV

    model_cv = LogisticRegressionCV(
        Cs=20, penalty='l1', cv=5, scoring='accuracy',
        solver='saga', max_iter=5000
    )
    model_cv.fit(X, y)
    print(f"Best C: {model_cv.C_[0]:.4f}")

    best_idx = np.argmax(model_cv.scores_[1].mean(axis=0))
    best_acc = model_cv.scores_[1].mean(axis=0)[best_idx]
    print(f"Best CV accuracy: {best_acc:.4f}")
    print(f"Non-zero coefficients: "
          f"{np.sum(model_cv.coef_[0] != 0)} / {p}")
    ```

    선택된 $C = 0.0886$, 교차검증 정확도 $0.8000$, 0이 아닌 계수는 **4개**이고 모두 참
    신호다(위양성 0개).

    이는 연습문제 1의 $C = 0.1$ 결과와 사실상 같은 지점이다. 눈여겨볼 것은, 교차검증이
    **정확도**를 기준으로 골랐는데도 여기서는 매우 희소한 모형을 선택했다는 점이다. 18장의
    회귀 예제에서 교차검증이 늘 지나치게 조밀한 모형을 고르던 것과 대비된다.

    차이의 원인은 기준의 성질이다. 정확도는 **계단함수**라 예측 이름표가 바뀌지 않는 한
    잡음변수를 하나 더 넣어도 값이 전혀 변하지 않는다. 반면 이탈도나 로그손실은 연속적이라
    잡음변수를 넣어 훈련 적합을 조금이라도 개선하면 값이 미세하게 좋아진다. 즉 `scoring`을
    `'neg_log_loss'`로 바꾸면 더 조밀한 모형이 선택될 가능성이 높다. **채점 기준이 곧 선택
    기준이다.** $\square$

---

**연습문제 5.**
엘라스틱넷 벌점

$$
\alpha\|\boldsymbol\beta\|_1 + \frac{1-\alpha}{2}\|\boldsymbol\beta\|_2^2
$$

이 임의의 $\alpha \in [0,1]$에 대해 $\boldsymbol\beta$의 볼록함수임을 증명하라.

??? success "연습문제 5 풀이"

    $\|\boldsymbol\beta\|_1 = \sum_j |\beta_j|$와
    $\|\boldsymbol\beta\|_2^2 = \sum_j \beta_j^2$은 모두 $\boldsymbol\beta$의 볼록함수다.
    L1 노름은 볼록함수 $|\beta_j|$들의 합이므로 볼록이고, L2 노름의 제곱은 헤세행렬이
    $2I$로 양정치이므로 강볼록이다.

    볼록함수들의 음이 아닌 가중합은 볼록이다. $\alpha \geq 0$이고 $(1-\alpha)/2 \geq 0$이므로
    엘라스틱넷 벌점

    $$
    P(\boldsymbol\beta) = \alpha\|\boldsymbol\beta\|_1 + \frac{1-\alpha}{2}\|\boldsymbol\beta\|_2^2
    $$

    은 볼록이다. 나아가 $\alpha < 1$이면 L2 항이 $P$를 강볼록으로 만들어 벌점 손실의 최소점이
    유일함을 보장한다.

    !!! note "$\alpha = 1$일 때도 유일할 수 있다"
        $\alpha = 1$(순수 라쏘)이면 벌점 자체는 강볼록이 아니지만, 목적함수 전체의 유일성은
        손실 항의 곡률에도 달려 있다. 계획행렬이 완전열계수이고 분리가 없으면 $-\ell$이
        강볼록이므로 라쏘 해도 유일하다. $p > n$이거나 열이 중복될 때 비로소 유일성이 깨진다.
        $\square$
