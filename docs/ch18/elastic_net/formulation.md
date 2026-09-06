# 엘라스틱넷의 정식화

능형회귀는 상관된 설명변수를 잘 다루지만 변수선택을 하지 못한다. 라쏘는 변수선택을 하지만 상관된 변수에서 어려움을 겪으며, 집단에서 하나를 임의로 고르고 나머지를 버린다. **엘라스틱넷**(Zou and Hastie, 2005)은 두 벌점을 결합하여 라쏘의 희소성과 능형의 안정성을 함께 물려받는다.

## 엘라스틱넷 목적함수

$$
\hat{\boldsymbol{\beta}}_{\text{EN}} = \arg\min_{\boldsymbol{\beta}} \left\{\frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda\bigl[\alpha\|\boldsymbol{\beta}\|_1 + (1 - \alpha)\tfrac{1}{2}\|\boldsymbol{\beta}\|_2^2\bigr]\right\}
$$

- $\lambda \geq 0$은 전체 정칙화 강도다.
- $\alpha \in [0, 1]$은 L1과 L2의 균형을 잡는 **혼합모수**다.
- $\|\boldsymbol{\beta}\|_2^2$ 앞의 $1/2$은 좌표하강 갱신을 단순하게 만드는 관례다.

## 특수한 경우

| $\alpha$ | 방법 | 벌점 |
|---|---|---|
| $\alpha = 0$ | 능형회귀 | $\frac{\lambda}{2}\|\boldsymbol{\beta}\|_2^2$ |
| $\alpha = 1$ | 라쏘 | $\lambda\|\boldsymbol{\beta}\|_1$ |
| $0 < \alpha < 1$ | 엘라스틱넷 | $\lambda\bigl[\alpha\|\boldsymbol{\beta}\|_1 + (1-\alpha)\tfrac{1}{2}\|\boldsymbol{\beta}\|_2^2\bigr]$ |

$\alpha$가 1에 가까우면 더 희소한 모형(라쏘에 가까움), 0에 가까우면 더 조밀하고 축소가 강한 모형(능형에 가까움)이 된다.

## 다른 매개화

두 개의 정칙화 모수를 따로 쓰는 정식화도 있다.

$$
\hat{\boldsymbol{\beta}}_{\text{EN}} = \arg\min_{\boldsymbol{\beta}} \left\{\frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda_1\|\boldsymbol{\beta}\|_1 + \lambda_2\|\boldsymbol{\beta}\|_2^2\right\}
$$

두 매개화는 $\lambda_1 = \lambda\alpha$, $\lambda_2 = \lambda(1 - \alpha)/2$로 연결된다. $(\lambda, \alpha)$ 매개화가 더 흔한데, 전체 벌점 강도와 벌점 혼합을 분리하여 $\alpha$를 고정한 채 $\lambda$에 대해 교차검증할 수 있기 때문이다.

## 제약최적화 형태

$$
\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 \quad \text{제약} \quad \alpha\|\boldsymbol{\beta}\|_1 + (1-\alpha)\tfrac{1}{2}\|\boldsymbol{\beta}\|_2^2 \leq t
$$

2차원에서 이 제약영역은 **모서리가 둥근 마름모**다. L1 마름모의 꼭짓점은 남아 있어(희소성 가능) 평평한 변이 곡선으로 바뀐다(안정성 제공). $\alpha \to 0$이면 원에, $\alpha \to 1$이면 마름모에 가까워진다.

!!! note "엘라스틱넷 제약의 기하"
    엘라스틱넷 제약집합은 L1 공에서 꼭짓점(정확한 0을 만든다)을, L2 공에서 매끄러움(상관된 변수에 안정성을 준다)을 물려받는다. 이 기하적 결합이 다중공선성을 다루면서 변수선택을 수행하는 능력의 핵심이다.

## 엘라스틱넷의 좌표하강

표준화된 설명변수에서 $\beta_j$의 갱신은

$$
\hat{\beta}_j \leftarrow \frac{S_{\alpha\lambda}(z_j)}{1 + (1-\alpha)\lambda}
$$

이다. 분자가 L1 연성 문턱(라쏘 성분)을, 분모가 L2 축소(능형 성분)를 담당한다. $\alpha = 1$이면 분모가 1이 되어 라쏘 갱신으로, $\alpha = 0$이면 분자에 문턱이 없어져 능형 갱신으로 환원된다.

## 강볼록성

엘라스틱넷 목적함수는 $\alpha < 1$이면(L2 성분이 있으면) $p > n$이더라도 **강볼록**이다. L2 벌점이 헤세 행렬을 양정치로 만들기 때문이다. 반면 순수 라쏘($\alpha = 1$)는 $p > n$일 때 볼록이지만 강볼록은 아니어서 해가 유일하지 않을 수 있다.

!!! tip "강볼록성의 실무적 함의"
    $\alpha < 1$에서 해가 유일하다는 성질 덕분에, $p > n$일 때 엘라스틱넷의 해 경로가 라쏘 경로보다 안정적이고 재현 가능하다. 고차원 상황에서 순수 라쏘보다 엘라스틱넷을 선호하는 이유 중 하나다.

## $\alpha$의 선택

1. **격자탐색과 교차검증.** $(\lambda, \alpha)$의 2차원 격자를 $K$-겹 교차검증으로 평가한다. 가장 철저하지만 계산이 비싸다.
2. **$\alpha$를 고정하고 $\lambda$만 최적화.** 영역 지식에 근거해 $\alpha$를 정하고($\alpha = 0.5$ 등) $\lambda$만 교차검증한다. 빠르고 대개 충분하다.

흔한 선택은 $\alpha = 0.5$(L1과 L2 동등), $\alpha = 0.9$(대체로 라쏘에 약간의 능형 안정화), $\alpha = 0.1$(대체로 능형에 약간의 라쏘 희소성)이다.

## 요약

엘라스틱넷은 혼합모수 $\alpha$를 통해 L1과 L2 벌점을 결합하며 능형($\alpha = 0$)과 라쏘($\alpha = 1$) 사이를 보간한다. L1 성분이 희소성을, L2 성분이 강볼록성과 상관된 변수에 대한 안정성을 제공한다. 좌표하강 갱신이 두 벌점을 자연스럽게 담아내며, 제약집합은 L1에서 꼭짓점을, L2에서 매끄러움을 물려받은 둥근 마름모다.

## 연습문제

**연습문제 1.**
scikit-learn의 `alpha`와 `l1_ratio`가 이 페이지의 $\lambda$, $\alpha$와 어떻게 대응하는지 확인하고, `l1_ratio`를 0에 보내면 정말 능형회귀와 일치하는지 검증하라.

??? success "연습문제 1 풀이"
    scikit-learn의 `ElasticNet` 목적함수는

    $$
    \frac{1}{2n}\|\mathbf{y}-\mathbf{X}\boldsymbol\beta\|^2
    + \texttt{alpha}\cdot\texttt{l1\_ratio}\,\|\boldsymbol\beta\|_1
    + \frac{\texttt{alpha}(1-\texttt{l1\_ratio})}{2}\|\boldsymbol\beta\|_2^2
    $$

    이므로 대응은 $\texttt{alpha} = \lambda$, $\texttt{l1\_ratio} = \alpha$이다. **이름이 뒤바뀌어 있으므로 주의해야 한다.**

    한편 `Ridge`의 목적함수는 $\|\mathbf{y}-\mathbf{X}\boldsymbol\beta\|^2 + \texttt{alpha}\|\boldsymbol\beta\|_2^2$로 $1/(2n)$ 배율이 없다. 따라서 `ElasticNet(alpha=a, l1_ratio=0)`에 대응하는 능형은 $\texttt{alpha}_{\text{ridge}} = n\,a\,(1-\texttt{l1\_ratio})$이다.

    ```python
    import numpy as np
    from sklearn.linear_model import ElasticNet, Ridge
    n, a, l1 = 60, 0.5, 1e-8
    be = ElasticNet(alpha=a, l1_ratio=l1, fit_intercept=False,
                    max_iter=500000, tol=1e-14).fit(X, y).coef_
    br = Ridge(alpha=n*a*(1-l1), fit_intercept=False).fit(X, y).coef_
    print(np.abs(be - br).max())      # 7.3e-09
    ```

    **최대 차이가 $7.3\times10^{-9}$로 사실상 일치한다.** 배율 관계가 정확함이 확인된다.

    실무적 함의: **`Ridge(alpha=1)`과 `Lasso(alpha=1)`의 벌점 강도는 전혀 다르다.** 두 방법을 같은 격자에서 비교하려면 배율 $n$을 반드시 고려해야 한다.

---

**연습문제 2.**
$\alpha < 1$이면 $p > n$에서도 해가 유일하다는 강볼록성 주장이 실제 선택 개수에 어떻게 나타나는지 확인하라.

??? success "연습문제 2 풀이"
    $n = 40$, $p = 200$, 참 변수 60개인 자료에서 세 방법을 비교한다.

    ```python
    import numpy as np
    from sklearn.linear_model import Lasso, ElasticNet
    rng = np.random.default_rng(77)
    n, p = 40, 200
    X = rng.normal(size=(n, p)); X -= X.mean(0); X /= X.std(0)
    beta = np.zeros(p); beta[:60] = 1.0
    y = X @ beta + rng.normal(0, 1, n); y -= y.mean()
    ```

    | 방법 | 선택된 변수 수 |
    |:---|---:|
    | 라쏘 ($\alpha = 1$) | **39** |
    | 엘라스틱넷 ($\alpha = 0.5$) | **89** |
    | 엘라스틱넷 ($\alpha = 0.1$) | **167** |

    ($n = 40$, $p = 200$, 참 변수 60개)

    **라쏘가 정확히 $n - 1 = 39$개에서 멈춘다.** 참 변수가 60개인데 그중 대부분을 놓친다. 이것이 "$p > n$에서 라쏘는 최대 $n$개"라는 한계의 직접적 관측이다.

    **엘라스틱넷은 이 벽을 넘는다.** $\alpha = 0.5$에서 89개, $\alpha = 0.1$에서 167개를 고른다. L2 항이 목적함수를 강볼록으로 만들어 해가 $n$차원 부분공간에 갇히지 않기 때문이다.

    다만 $\alpha = 0.1$의 167개는 참 변수 60개를 크게 넘는다. **한계를 없앤 대가로 희소성을 잃는다.** 참 변수 개수를 대략 안다면 그에 맞춰 $\alpha$를 조정해야 한다.
