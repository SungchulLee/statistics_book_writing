# 부분집합 선택과 단계적 선택

## 개요

이 페이지는 선형회귀의 세 가지 특성선택 전략 — 최량 부분집합 선택, 전진 단계적 선택, 후진 단계적 선택 — 을 보인다. 설명변수 8개(참으로 관련 있는 것 4개, 잡음 4개)인 인공자료로 훈련 RSS, 검증 RSS, 선택된 특성 집합을 비교하여 전수 탐색과 탐욕 알고리즘의 절충을 살펴본다.

## 수학적 배경

### 최량 부분집합 선택

각 모형 크기 $k = 1, \ldots, p$에 대해 최량 부분집합 선택은 가능한 $\binom{p}{k}$개의 $k$ 변수 모형을 모두 평가하여 훈련 RSS가 가장 낮은 것을 고른다. 평가하는 모형의 총 개수는 $\sum_{k=1}^p \binom{p}{k} = 2^p - 1$이므로, 이 방법은 $p$가 작을 때에만(보통 $p \leq 20$) 계산이 가능하다.

### 전진 단계적 선택

영모형(절편만)에서 시작하여 RSS를 가장 크게 줄이는 설명변수를 탐욕적으로 더한다.

1. $\mathcal{S} = \emptyset$에서 시작한다
2. $k = 1, \ldots, p$에 대해: $j^* = \arg\min_{j \notin \mathcal{S}} \mathrm{RSS}(\mathcal{S} \cup \{j\})$를 찾고 $\mathcal{S} \leftarrow \mathcal{S} \cup \{j^*\}$로 둔다

이는 $p + (p-1) + \cdots + 1 = p(p+1)/2$개의 모형만 평가하므로 $2^p$보다 훨씬 적다.

### 후진 단계적 선택

완전모형에서 시작하여 제거했을 때 RSS 증가가 가장 작은 설명변수를 탐욕적으로 뺀다.

1. $\mathcal{S} = \{1, \ldots, p\}$에서 시작한다
2. $k = p-1, \ldots, 1$에 대해: $j^* = \arg\min_{j \in \mathcal{S}} \mathrm{RSS}(\mathcal{S} \setminus \{j\})$를 찾고 $\mathcal{S} \leftarrow \mathcal{S} \setminus \{j^*\}$로 둔다

### 최적 크기 고르기

훈련 RSS는 $k$에 따라 언제나 줄어들므로, 최적 $k$는 남겨 둔 자료에서 계산한 기준(검증 RSS, 교차검증, AIC, BIC)을 최소화하여 고른다.

### 자료 생성

<div class="codebox" markdown>

**예제 1.** 실험용 자료

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# 앞의 넷만 참 계수가 0 이 아니다. 세 방법이 이 넷을 찾아내는지 견준다.
np.random.seed(42)
n, p = 200, 8
X = np.random.randn(n, p)
true_beta = np.array([3.0, 1.5, -2.0, 0.8, 0, 0, 0, 0])
y = X @ true_beta + np.random.normal(0, 2, n)
names = [f"x{i+1}" for i in range(p)]
```

</div>

### 최량 부분집합 선택

<div class="codebox" markdown>

**예제 2.** 최적 부분집합 선택

```python
from itertools import combinations

def best_subset(X, y, max_k=None):
    """모든 부분집합을 다 따져 크기별 최선을 찾는다.

    크기 k 마다 가능한 조합을 남김없이 본다. 답은 확실하지만 부분집합이
    2^p 개라 변수가 스물만 넘어도 감당할 수 없다.
    """
    n, p = X.shape
    if max_k is None:
        max_k = p
    results = {}
    for k in range(1, max_k + 1):
        best_rss, best_features = np.inf, None
        for combo in combinations(range(p), k):
            model = LinearRegression().fit(X[:, combo], y)
            rss = np.sum((y - model.predict(X[:, combo])) ** 2)
            if rss < best_rss:
                best_rss, best_features = rss, combo
        results[k] = {"features": best_features, "rss": best_rss}
    return results
```

</div>

### 전진 단계적 선택

<div class="codebox" markdown>

**예제 3.** 전진 단계선택

```python
def forward_stepwise(X, y):
    """빈 모형에서 시작해 RSS 를 가장 많이 줄이는 변수를 하나씩 더한다.

    따지는 모형이 p(p+1)/2 개로 줄어 훨씬 빠르다. 다만 한 번 들어간 변수는
    빠지지 않으므로 최적 부분집합을 놓칠 수 있다.
    """
    n, p = X.shape
    selected, remaining = [], list(range(p))
    results = {}
    for k in range(1, p + 1):
        best_rss, best_feature = np.inf, None
        for f in remaining:
            trial = selected + [f]
            model = LinearRegression().fit(X[:, trial], y)
            rss = np.sum((y - model.predict(X[:, trial])) ** 2)
            if rss < best_rss:
                best_rss, best_feature = rss, f
        selected.append(best_feature)
        remaining.remove(best_feature)
        results[k] = {"features": tuple(selected), "rss": best_rss}
    return results
```

</div>

### 후진 단계적 선택

<div class="codebox" markdown>

**예제 4.** 후진 단계선택

```python
def backward_stepwise(X, y):
    """전체 모형에서 시작해 RSS 를 가장 적게 늘리는 변수를 하나씩 뺀다.

    시작점이 전체 모형이므로 n > p 여야 쓸 수 있다. 변수가 관측보다 많으면
    전진선택으로 가야 한다.
    """
    n, p = X.shape
    current = list(range(p))
    results = {}
    model = LinearRegression().fit(X, y)
    results[p] = {"features": tuple(current),
                  "rss": np.sum((y - model.predict(X)) ** 2)}
    for k in range(p - 1, 0, -1):
        best_rss, best_remove = np.inf, None
        for f in current:
            trial = [x for x in current if x != f]
            model = LinearRegression().fit(X[:, trial], y)
            rss = np.sum((y - model.predict(X[:, trial])) ** 2)
            if rss < best_rss:
                best_rss, best_remove = rss, f
        current.remove(best_remove)
        results[k] = {"features": tuple(current), "rss": best_rss}
    return results
```

</div>

## 해석

- **최량 부분집합**은 크기별 전역 최적 모형을 반드시 찾아내지만 $p > 20$이면 계산이 불가능하다(지수적 증가).
- **전진 단계적**은 탐욕적 근사이므로 전역 최적 모형을 놓칠 수 있지만 $O(p^2)$ 시간에 끝난다. 한번 들어간 설명변수를 다시 뺄 수 없다.
- **후진 단계적**은 완전모형에서 시작하므로 전진 선택과 다른 해를 낼 수 있다. 처음 완전모형을 적합하려면 $n > p$가 필요하다.
- 신호가 강하고 참 모형이 탐색 경로 위에 있으면 세 방법 모두 같은 최적 $k$에서 일치한다.
- **검증 RSS**가 필수적이다. 훈련 RSS는 $k$에 따라 언제나 줄어들므로 모형 크기를 고르는 데 쓸 수 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span> 세 방법을 모두 실행하고 (검증 RSS로 정한) 최적 $k$에서 선택된 특성을 비교하라. 모두 참 설명변수 4개를 찾아내는가?

</div>

??? success "풀이"

    ```python
    n_train = 140
    X_tr, X_val = X[:n_train], X[n_train:]
    y_tr, y_val = y[:n_train], y[n_train:]

    best = best_subset(X_tr, y_tr)
    fwd = forward_stepwise(X_tr, y_tr)
    bwd = backward_stepwise(X_tr, y_tr)

    # Find optimal k by validation RSS for each method
    for method_name, res in [("Best", best), ("Fwd", fwd), ("Bwd", bwd)]:
        val_rss = [np.sum((y_val - LinearRegression().fit(X_tr[:, res[k]["features"]],
                   y_tr).predict(X_val[:, res[k]["features"]])) ** 2)
                   for k in range(1, 9)]
        opt_k = np.argmin(val_rss) + 1
        print(f"{method_name}: k={opt_k}, features={res[opt_k]['features']}")
    ```

    출력:

    ```
    Best: k=4, features=(0, 1, 2, 3)
    Fwd: k=4, features=(0, 2, 1, 3)
    Bwd: k=4, features=(0, 1, 2, 3)
    ```

    최적 부분집합, 전진선택, 후진제거가 모두 같은 변수 집합 $\{0,1,2,3\}$을 골랐다. 전진선택은 넣는 **순서**만 다르다.

    세 방법이 언제나 일치하지는 않는다. 최적 부분집합은 $2^p$개를 모두 보지만 단계적 방법은 탐욕적이라, 변수들이 서로 얽혀 있으면 갈릴 수 있다.

    세 방법 모두 $k = 4$에서 특성 $\{x_1, x_2, x_3, x_4\}$를 고르며 검증 RSS 곡선도 동일하다.

    | $k$ | 1 | 2 | 3 | **4** | 5 | 6 | 7 | 8 |
    |---|---|---|---|---|---|---|---|---|
    | 검증 RSS | 729.3 | 416.6 | 300.9 | **259.5** | 265.7 | 264.6 | 266.7 | 267.3 |

    $k = 4$까지는 검증 RSS가 가파르게 줄다가 그 뒤로는 오히려 조금 늘어난다. 잡음 설명변수 네 개를 넣어도 아무 도움이 되지 않고 오히려 손해임을 보여준다. 신호가 충분히 강하고 참 설명변수들이 처음 네 단계에 모두 들어오므로 탐욕적 방법도 전역 최적과 같은 답을 낸다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 참 설명변수 4개는 그대로 두고 $p$를 8에서 20으로 늘려라. 최량 부분집합을 여전히 계산할 수 있는가? 전진 선택은 어떻게 작동하는가?

</div>

??? success "풀이"

    $p = 20$이면 최량 부분집합은 $2^{20} - 1 = 1{,}048{,}575$개의 모형을 평가해야 하므로 계산이 비싸지만 아직은 가능하다. $p = 30$ 이상이면 비현실적이 된다. 전진 선택은 여전히 빠르고($O(p^2)$개 모형) 참 설명변수의 효과가 충분히 강하면 그것들을 찾아낸다. 탐욕적이라는 성질 때문에 우연히 반응변수와 상관된 잡음 설명변수를 일찍 고를 수도 있지만, 참 효과가 강하면 그럴 가능성은 낮다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> 같은 $k$에서 전진과 후진 단계적 선택이 서로 다른 특성을 고르는 예를 구성하라. 자료의 어떤 성질이 이 차이를 만드는가?

</div>

??? success "풀이"

    설명변수들이 상관되어 있을 때 일어난다. 예를 들어 $x_5$가 $x_1$·$x_2$와 중간 정도로 상관되어 있다면, $x_5$가 일부 신호를 담고 있으므로 전진 선택이 ($x_2$보다 먼저) $x_5$를 일찍 넣을 수 있다. 후진 선택은 모든 설명변수에서 시작하는데, 완전모형 안에서는 $x_2$가 더 유용하므로 $x_2$보다 $x_5$를 먼저 뺄 수 있다. 이 차이는 두 알고리즘의 탐욕적이고 경로 의존적인 성질에서 온다. 추가/제거의 순서가 이미 모형에 들어 있는 다른 설명변수에 달려 있기 때문이다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> 설명변수가 $k$개일 때 최량 부분집합 선택의 훈련 RSS가 전진 단계적 선택의 것보다 작거나 같음을 증명하라.

</div>

??? success "풀이"

    최량 부분집합은 $\binom{p}{k}$개의 부분집합을 모두 탐색하여 RSS가 가장 낮은 것을 고른다. 전진 선택은 탐욕적 경로를 통해 하나의 특정한 $k$ 변수 모형을 만든다. 전진 선택이 만든 모형도 최량 부분집합이 고려한 $\binom{p}{k}$개 부분집합 가운데 하나이므로 최량 부분집합의 RSS가 그보다 크지 않다.

    $$
    \mathrm{RSS}_{\text{best}}(k) = \min_{\mathcal{S}: |\mathcal{S}|=k} \mathrm{RSS}(\mathcal{S}) \leq \mathrm{RSS}(\mathcal{S}_{\text{fwd}}(k)) = \mathrm{RSS}_{\text{fwd}}(k).
    $$

    탐욕적 경로가 우연히 전역 최적을 찾았을 때 등호가 성립한다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> 검증 RSS 대신 AIC로 모형선택을 구현하라. AIC로 최적 $k$를 정하고 검증 방식과 비교하라.

</div>

??? success "풀이"

    ```python
    def aic(n, rss, k):
        return n * np.log(rss / n) + 2 * (k + 1)  # +1 for intercept

    fwd = forward_stepwise(X_tr, y_tr)
    aic_vals = []
    for k in range(1, 9):
        aic_vals.append(aic(n_train, fwd[k]['rss'], k))
    opt_k_aic = np.argmin(aic_vals) + 1
    print("AIC per k: [" + ", ".join(f"{v:.2f}" for v in aic_vals) + "]")
    print(f"Optimal k (AIC): {opt_k_aic}")
    ```

    출력:

    ```text
    AIC per k: [310.90, 267.13, 204.61, 178.62, 178.86, 179.83, 181.65, 183.59]
    Optimal k (AIC): 4
    ```

    AIC도 검증 RSS와 마찬가지로 $k = 4$(참 모형 크기)를 고른다. 다만 $k = 4$와 $k = 5$의 차이가 $0.24$에 지나지 않아, AIC의 약한 벌점 때문에 $k = 5$가 선택될 뻔했다는 점에 주목할 만하다. 검증 RSS는 편향이 작지만 변동이 크고(특정 분할에 의존한다), AIC는 자료를 나누지 않아 안정적이지만 점근 근사에 기댄다. 신호가 뚜렷하면 두 방법은 대체로 일치한다. $\square$

---

## 정리하며

세 가지 선택 전략을 **같은 자료에서** 견주었다.

- **관련 변수 4개와 잡음 4개**로 만든 자료에서 각 방법이 참 변수들을 찾아내는지 본다.
- **훈련 RSS 는 언제나 변수를 더할수록 줄어든다.** 그래서 **훈련 RSS 로 모형 크기를 고를 수 없다.** 검증 RSS 가 U 자를 그리며 최적점을 알려 준다.
- **최량 부분집합이 훈련 RSS 는 가장 낮지만** 검증에서 단계적 방법을 크게 앞서지는 않는다. **전수 탐색의 이득이 생각보다 작다**는 것이 실무적 교훈이다.
- **전진과 후진이 다른 답을 줄 수 있다.** 탐욕적이라 경로에 의존하며, 어느 쪽이 옳다고 말할 수 없다.
- **잡음 변수가 선택되는 일이 흔하다.** $p$ 가 크고 $n$ 이 작을수록 심하며, 이것이 선택 후 추론이 위험한 이유다.

다음 절부터 **스플라인과 GAM**으로 넘어간다. 비선형 관계를 다루는 유연한 모형들이다.
