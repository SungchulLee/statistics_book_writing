# 모형선택 비교

## 개요

이 페이지는 세 가지 기준 — Akaike 정보기준(AIC), Bayes 정보기준(BIC), 교차검증(CV) — 을 이용한 모형선택을 보인다. 설명변수 8개 중 3개만 실제로 관련 있는 인공자료로 전진 선택을 수행하고, 각 기준이 어떤 모형 크기를 최적으로 판정하는지 비교한다.

## 수학적 배경

### Akaike 정보기준 (AIC)

$$
\mathrm{AIC} = n \ln\!\left(\frac{\mathrm{RSS}}{n}\right) + 2k,
$$

여기서 $k$는 (절편을 포함한) 추정 모수의 개수이다. AIC는 표본 밖 예측오차를 추정하며 모수 하나당 $2$의 벌점으로 복잡도를 억제한다.

### Bayes 정보기준 (BIC)

$$
\mathrm{BIC} = n \ln\!\left(\frac{\mathrm{RSS}}{n}\right) + k \ln(n).
$$

BIC는 모수 하나당 $\ln(n)$의 벌점을 쓰며, $n > e^2 \approx 7.4$이면 이 값이 2를 넘는다. 따라서 표본이 중간 이상이면 BIC가 더 작은 모형을 선호한다.

### 교차검증 MSE

예측오차의 $K$-겹 교차검증 추정값은

$$
\mathrm{CV}(K) = \frac{1}{K}\sum_{k=1}^{K} \mathrm{MSE}_k, \qquad \mathrm{MSE}_k = \frac{1}{|V_k|}\sum_{i \in V_k}(y_i - \hat{y}_i^{(-k)})^2,
$$

여기서 $\hat{y}_i^{(-k)}$는 겹 $k$ 없이 훈련한 모형이 내놓은 관측값 $i$의 예측값이다.

### 정보기준 함수

<div class="codebox" markdown>

#### 예제 1. AIC와 BIC 구현 { .eg }

```python
import numpy as np

def aic(n, rss, k):
    """AIC. 모수 하나당 벌점이 2 다."""
    return n * np.log(rss / n) + 2 * k

def bic(n, rss, k):
    """BIC. 벌점이 log(n) 이라 n>7 부터 AIC 보다 무겁다.

    그래서 BIC 는 대개 더 작은 모형을 고른다. AIC 는 예측을 잘하는 모형을,
    BIC 는 참 모형을 찾는 것을 겨냥한다고 흔히 말한다.
    """
    return n * np.log(rss / n) + k * np.log(n)
```

</div>

### 교차검증 MSE

<div class="codebox" markdown>

#### 예제 2. 교차검증 구현 { .eg }

```python
def cv_mse(X, y, folds=5):
    """k겹 교차검증으로 예측오차를 추정한다.

    AIC·BIC 가 공식으로 벌점을 매긴다면, 교차검증은 실제로 떼어 놓은
    자료에서 재 본다. 가정이 적은 대신 계산이 많이 든다.
    """
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // folds
    mses = []
    for k in range(folds):
        val_idx = indices[k * fold_size:(k + 1) * fold_size]
        train_idx = np.setdiff1d(indices, val_idx)
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        beta = np.linalg.lstsq(X_tr, y_tr, rcond=None)[0]
        pred = X_va @ beta
        mses.append(np.mean((y_va - pred) ** 2))
    return np.mean(mses)
```

</div>

### 전진 선택

<div class="codebox" markdown>

#### 예제 3. 전진선택으로 변수 고르기 { .eg }

```python
# 여덟 변수 중 앞의 셋만 실제로 쓰이고 나머지 다섯은 계수가 0 이다.
# 전진선택이 그 셋을 먼저 집어내는지 보는 것이 이 실험의 목적이다.
np.random.seed(42)
n, p_total = 200, 8
X_raw = np.random.randn(n, p_total)
beta_true = np.array([3.0, -2.0, 1.5, 0, 0, 0, 0, 0])
y = X_raw @ beta_true + np.random.randn(n) * 2

remaining = list(range(p_total))
selected = []
aic_history, bic_history = [], []

# 전진선택: 매 단계에서 AIC 를 가장 많이 낮추는 변수를 하나씩 더한다.
# 모든 부분집합을 따지면 2^8 개지만, 이렇게 하면 훨씬 적게 본다.
# 대신 최적 부분집합을 놓칠 수 있다.
for step in range(p_total):
    best_score, best_j = np.inf, None
    for j in remaining:
        cols = selected + [j]
        X_cand = np.column_stack([np.ones(n), X_raw[:, cols]])
        beta = np.linalg.lstsq(X_cand, y, rcond=None)[0]
        rss = np.sum((y - X_cand @ beta) ** 2)
        score = aic(n, rss, len(cols) + 1)
        if score < best_score:
            best_score, best_j = score, j
    selected.append(best_j)
    remaining.remove(best_j)

    X_sel = np.column_stack([np.ones(n), X_raw[:, selected]])
    beta = np.linalg.lstsq(X_sel, y, rcond=None)[0]
    rss = np.sum((y - X_sel @ beta) ** 2)
    k = len(selected) + 1
    aic_history.append(aic(n, rss, k))
    bic_history.append(bic(n, rss, k))
```

</div>

### 결과

선택 순서(0부터 시작하는 색인)는 `[0, 1, 2, 4, 7, 3, 5, 6]`으로, 참으로 관련 있는 세 설명변수 0, 1, 2가 정확히 먼저 뽑혔다.

| 모형 크기 | AIC | BIC | 5-겹 CV MSE |
|---|---|---|---|
| 1 | 464.79 | 471.38 | 10.1104 |
| 2 | 371.54 | 381.44 | 6.3024 |
| **3** | **262.06** | **275.25** | **3.7603** |
| 4 | 262.67 | 279.16 | 3.7891 |
| 5 | 264.07 | 283.86 | 3.8011 |
| 6 | 266.02 | 289.10 | 3.8225 |
| 7 | 267.99 | 294.38 | 3.8682 |
| 8 | 269.99 | 299.67 | 3.8933 |

세 기준 모두 크기 3에서 최솟값을 갖는다. 잡음 설명변수 다섯 개가 올바르게 배제되었다.

크기 3과 크기 4의 AIC 차이가 $0.61$에 지나지 않는다는 점에 주목하라. AIC 벌점이 상대적으로 약하기 때문에, 네 번째 설명변수가 우연히 조금만 더 큰 적합 개선을 냈다면 AIC는 그것을 골랐을 수 있다. 반면 BIC 차이는 $3.91$로 훨씬 뚜렷하다. AIC보다 BIC가 절약적인 모형을 더 확실하게 선호한다는 사실을 그대로 보여준다.

## 해석

- **AIC**는 벌점($2k$)이 비교적 약해 조금 더 큰 모형을 고르는 경향이 있다. 예측 정확도를 겨냥한다.
- **BIC**는 $k \ln(n)$이 표본크기와 함께 커지므로 더 작은 모형을 고르는 경향이 있다. 일치성을 가져 (참 모형이 후보에 있다면) $n \to \infty$일 때 참 모형을 고른다.
- **교차검증**은 점근이론에 기대지 않고 표본 밖 예측오차를 직접 추정한다. 계산 비용이 크지만 분포 가정을 덜 요구한다.
- 이 예에서는 참으로 관련 있는 설명변수가 3개이며, 세 방법 모두 크기 3의 모형을 찾아내어 잡음 설명변수가 올바르게 배제됨을 확인해 준다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span> 각 단계에서 다음 설명변수를 정할 때 AIC 대신 BIC를 써서 전진 선택 절차를 수행하라. 선택되는 설명변수의 순서가 달라지는가?

</div>

??? success "풀이"

    안쪽 반복문의 `score = aic(n, rss, len(cols) + 1)`을 `score = bic(n, rss, len(cols) + 1)`로 바꾼다.

    선택 **순서는 바뀌지 않는다**. 한 단계 안에서 후보들은 모두 같은 $k$를 가지므로 AIC든 BIC든 벌점항이 상수이고, 결국 RSS를 가장 크게 줄이는 변수를 고르게 되어 두 기준이 동일한 선택을 한다. 달라지는 것은 **멈추는 지점**이다. BIC는 벌점이 더 강해 더 적은 설명변수에서 멈춘다. 위 표에서도 크기 4 이후 BIC의 증가폭이 AIC보다 훨씬 가파르다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 잡음 수준을 $\sigma = 2$에서 $\sigma = 5$로 키워라. 각 기준이 고르는 최적 모형 크기는 어떻게 달라지는가?

</div>

??? success "풀이"

    잡음이 커지면 신호 대 잡음비가 낮아진다. 참 설명변수를 넣고 뺄 때의 RSS 차이가 전체 RSS에 비해 작아진다. AIC와 BIC가 더 적은 설명변수를 고를 수 있고(3개 대신 1–2개), 교차검증 MSE 곡선은 평평해져 최솟값이 덜 뚜렷해진다. 잡음 설명변수와 참 설명변수를 구별하기 어려워진다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> 10-겹 교차검증을 구현하여 5-겹과 비교하라. $K$의 선택에서 편향-분산 절충을 논하라.

</div>

??? success "풀이"

    ```python
    cv5 = [cv_mse(np.column_stack([np.ones(n), X_raw[:, selected[:s]]]), y, folds=5)
           for s in range(1, p_total + 1)]
    cv10 = [cv_mse(np.column_stack([np.ones(n), X_raw[:, selected[:s]]]), y, folds=10)
            for s in range(1, p_total + 1)]
    ```

    $K$가 커지면 각 훈련집합의 크기가 $n$에 가까워져 편향이 줄지만, 겹들이 더 많이 겹쳐 분산이 커진다. $K = 5$는 편향이 조금 크지만 분산이 작고, $K = 10$은 편향이 작지만 분산이 크다. 극단인 $K = n$(LOOCV)은 거의 불편이지만 분산이 클 수 있다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span> Bayes 모형비교의 관점에서 BIC 벌점 $k\ln(n)$을 유도하라. 왜 벌점이 $n$에 의존하는가?

</div>

??? success "풀이"

    Bayes 모형선택에서는 모수공간에 대해 적분하여 주변가능도 $p(\mathbf{y} \mid M)$을 계산한다. 이 적분에 Laplace 근사를 쓰면

    $$
    \ln p(\mathbf{y} \mid M) \approx \ln p(\mathbf{y} \mid \hat{\boldsymbol{\theta}}, M) - \frac{k}{2}\ln(n) + O(1).
    $$

    양변에 $-2$를 곱하면 $\mathrm{BIC} = -2\ln L + k\ln(n)$을 얻는다. 벌점이 $n$에 의존하는 것은, 자료가 많아질수록 사후분포가 더 좁게 집중되어 모수 추가에 드는 "부피" 비용이 $n$에 대해 로그로 커지기 때문이다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> 참 모형에 관련 설명변수가 3개 있다고 하자. $n \to \infty$일 때 $P(\text{BIC가 참 모형을 고른다}) \to 1$이지만 AIC는 그렇지 않음을 증명하라.

</div>

??? success "풀이"

    AIC에서 모수 하나를 더할 때의 벌점은 $n$과 무관하게 $2$이다. $n \to \infty$일 때 무관한 설명변수를 넣어 얻는 $n\ln(\mathrm{RSS}/n)$의 감소량은 $\chi^2_1$ 확률변수(평균 1)로 수렴하므로 2를 넘을 확률이 0이 아니다. 따라서 AIC는 점근적으로 과대적합한다.

    BIC에서는 벌점이 $\ln(n) \to \infty$인 반면, 무관한 설명변수를 넣어 얻는 개선은 여전히 유계이다($\chi^2_1$로 수렴한다). 따라서 $n$이 충분히 크면 벌점이 지배하여 무관한 설명변수가 확률 1로 배제된다. 이것이 BIC의 일치성이다. $\square$

---

## 정리하며

세 기준을 **같은 자료에서** 비교했다.

- **설명변수 8개 중 3개만 실제로 관련 있는 자료**로 전진 선택을 돌리면 각 기준이 고르는 모형 크기가 갈린다.
- **BIC 가 가장 작은 모형을 고른다.** 벌점이 무거워 참 모형 크기에 가장 가깝게 간다.
- **AIC 는 조금 큰 모형을 고르는 경향이 있다.** 예측을 겨냥하므로 약간의 과대적합을 감수한다.
- **교차검증은 둘 사이 어딘가에 놓이며 변동이 있다.** 겹 나누기에 따라 결과가 달라지므로 여러 번 반복해 평균을 보는 것이 좋다.
- **세 기준이 일치하면 안심할 수 있고 갈리면 그 사실을 보고한다.** "최선의 모형" 하나를 고집하기보다 **여러 기준이 지지하는 후보들**을 제시하는 편이 정직하다.

다음 절 **다항 모형선택 교차검증**으로 넘어간다.
