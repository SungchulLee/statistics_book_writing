# 정칙화 모형의 정보기준

교차검증은 $\lambda$ 선택의 믿을 만한 방법이지만 겹마다 모형을 여러 번 적합해야 한다. 정보기준은 해석적 대안을 제공한다. 훈련오차에 복잡도 벌점을 더해 한 번의 적합만으로 예측오차를 추정한다. 정칙화 모형에서 핵심 난점은 적절한 복잡도 척도를 정의하는 것이다. 계수가 연속적으로 축소되므로 "모수의 개수"가 의미 있는 개념이 아니기 때문이다.

## AIC와 BIC 복습

로그가능도가 $\ell(\hat{\boldsymbol{\beta}})$이고 추정 모수가 $d$개인 모형에서

$$
\text{AIC} = -2\ell(\hat{\boldsymbol{\beta}}) + 2d, \qquad
\text{BIC} = -2\ell(\hat{\boldsymbol{\beta}}) + d\log n
$$

이다. $\sigma^2$이 알려진 가우스 선형모형에서 $-2\ell = n\log(2\pi\sigma^2) + \text{RSS}/\sigma^2$이므로, 상수를 무시하면

$$
\text{AIC} = \frac{\text{RSS}}{\sigma^2} + 2d, \qquad
\text{BIC} = \frac{\text{RSS}}{\sigma^2} + d\log n
$$

이다. 여기서 $\text{RSS} = \sum_i (y_i - \hat y_i)^2$은 잔차제곱**합**이다.

!!! warning "배율에 주의"
    문헌에 따라 $\text{RSS}$를 잔차제곱합이 아니라 평균제곱잔차 $\frac{1}{n}\sum_i(y_i-\hat y_i)^2$로 쓰기도 한다. 그 규약에서는 첫 항이 $n\,\text{RSS}/\sigma^2$이 된다.

    **어느 규약이든 두 항의 상대적 크기가 맞아야 한다.** 첫 항에 $n$을 잘못 곱하면 복잡도 벌점 $2d$가 상대적으로 $n$배 작아져 사실상 무시되고, AIC가 언제나 가장 복잡한 모형을 고르게 된다.

AIC는 예측오차 최소화를 겨냥하며 점근적으로 LOOCV와 동등하다. BIC는 모형선택 일치성을 겨냥한다. $n > 8$이면 $\log n > 2$이므로 BIC가 더 무거운 벌점을 주어 단순한 모형을 선호한다.

## 유효자유도

정칙화 모형에서는 정수 모수 개수 $d$를 **유효자유도** $\text{df}(\lambda)$로 대체한다.

### 능형회귀

$$
\text{df}_{\text{ridge}}(\lambda) = \text{tr}\bigl[\mathbf{X}(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\bigr] = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda}
$$

각 항이 $j$번째 성분의 축소인자다. $\lambda \to 0$에서 $p$로, $\lambda \to \infty$에서 $0$으로 간다. 이는 모자 행렬의 대각합이며 선형 평활자의 자유도에 대한 표준 정의다.

### 라쏘

$$
\text{df}_{\text{lasso}}(\lambda) = |\hat{S}(\lambda)| = \text{0이 아닌 계수의 개수}
$$

Zou, Hastie, Tibshirani(2007)가 Stein의 불편위험추정(SURE)으로 증명한 이 놀라운 결과는, 라쏘의 유효자유도가 선택된 변수의 개수와 정확히 같음을 말한다. $\mathbf{X}$에 대한 온건한 조건과 가우스 오차 아래에서 성립한다.

!!! note "라쏘의 자유도는 불연속이다"
    $\text{df}(\lambda)$가 연속적으로 변하는 능형과 달리, 라쏘의 자유도는 계수가 활성집합에 들어오거나 나갈 때마다 1씩 뛴다. 이 불연속성이 변수선택의 이산적 성격을 반영한다.

### 엘라스틱넷

엘라스틱넷의 유효자유도에는 이만큼 단순한 공식이 없다. 선택된 변수에 대해 L2 벌점을 반영한 근사를 쓴다.

$$
\text{df}_{\text{EN}}(\lambda, \alpha) \approx \text{tr}\bigl[\mathbf{X}_{\hat{S}}(\mathbf{X}_{\hat{S}}^\top\mathbf{X}_{\hat{S}} + \lambda(1-\alpha)\mathbf{I})^{-1}\mathbf{X}_{\hat{S}}^\top\bigr]
$$

## 정칙화 모형의 AIC와 BIC

$$
\text{AIC}(\lambda) = \frac{\text{RSS}(\lambda)}{\hat{\sigma}^2} + 2\,\text{df}(\lambda), \qquad
\text{BIC}(\lambda) = \frac{\text{RSS}(\lambda)}{\hat{\sigma}^2} + \text{df}(\lambda)\log n
$$

$\hat{\sigma}^2$은 오차분산의 추정값이다($p < n$이면 전체 OLS 모형에서, 아니면 저차원 모형에서 얻는다). $\lambda$ 격자에서 AIC나 BIC를 최소화하는 $\lambda$를 고른다.

!!! warning "$\sigma^2$의 추정"
    정보기준은 $\sigma^2$의 추정값을 요구한다. $p > n$이면 OLS 잔차분산을 쓸 수 없다. 저차원 모형을 쓰거나, 척도 라쏘(scaled lasso)를 쓰거나, 교차검증으로 $\sigma^2$을 따로 추정하는 방법이 있다.

## 정보기준과 교차검증의 비교

| 기준 | 계산 | 이론 | 실무 |
|---|---|---|---|
| AIC | $\lambda$당 1회 적합 | 예측에 점근적으로 최적 | 예측 중심 선택에 좋음 |
| BIC | $\lambda$당 1회 적합 | 모형선택에 일치 | 참 모형이 희소할 때 좋음 |
| $K$-겹 CV | $\lambda$당 $K$회 적합 | 분포무관, 유한표본 | 가장 견고, 표준 선택 |
| GCV | $\lambda$당 1회 적합 | LOOCV와 점근적으로 동등 | 능형에 효율적 |

정보기준은 빠르지만 가우스 가정에 의존하고 $\sigma^2$ 추정값이 필요하다. 교차검증은 더 견고하지만 계산이 무겁다. 실무에서는 CV가 기본 선택이며, 계산 자원이 제한적일 때나 검산 용도로 정보기준을 쓴다.

## 실무 권고

1. 계산 제약이 심하지 않으면 **CV를 주 방법으로** 쓴다.
2. 예측이 목표이고 가우스 가정이 합리적이면 **AIC**를 쓴다.
3. 참 희소 모형의 식별이 목표면 **BIC**를 쓴다.
4. **능형회귀**에는 GCV가 CV의 효율적 해석적 대안이다.
5. **라쏘와 엘라스틱넷**에서는 $\text{df} = |\hat{S}|$ 공식 덕분에 정칙화 경로를 따라 AIC와 BIC를 쉽게 계산할 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
라쏘 경로를 따라 AIC와 BIC를 계산하여 각각이 고르는 모형을 비교하라. 참 변수 5개, 잡음변수 15개, $n = 100$인 자료를 쓴다.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from sklearn.linear_model import lasso_path, LinearRegression
    rng = np.random.default_rng(55)
    n, p = 100, 20
    X = rng.normal(size=(n, p)); X -= X.mean(0); X /= X.std(0)
    beta = np.zeros(p); beta[:5] = [3, -2, 1.5, 1, -1]
    y = X @ beta + rng.normal(0, 1, n); y -= y.mean()

    # 전체 OLS 잔차로 sigma^2 추정 (p < n 이므로 가능)
    b_ols = LinearRegression(fit_intercept=False).fit(X, y).coef_
    s2 = ((y - X @ b_ols)**2).sum() / (n - p)          # 0.9808 (참값 1.0)

    alphas, coefs, _ = lasso_path(X, y, n_alphas=300, eps=1e-4)
    rss = ((y[:, None] - X @ coefs)**2).sum(0)
    df  = (np.abs(coefs) > 1e-10).sum(0)
    aic = rss/s2 + 2*df
    bic = rss/s2 + df*np.log(n)
    ```

    | 기준 | 고른 $\lambda$ | $\text{df}$ |
    |:---|---:|---:|
    | AIC | 0.0852 | **10** |
    | BIC | 0.0852 | **10** |
    | 참값 | — | **5** |

    **두 기준이 같은 $\lambda$를 골랐고, 둘 다 참 변수 수의 두 배를 선택한다.**

    BIC의 벌점이 더 무거운데도($\log 100 = 4.61 > 2$) 같은 답이 나온 것은, 이 자료에서 RSS 곡선이 $\text{df} = 10$ 부근에서 가파르게 꺾이기 때문이다. 벌점의 차이가 그 꺾임을 넘어서지 못했다.

    **10개 대 5개의 과다선택이 더 중요한 관찰이다.** 이는 잘 알려진 현상으로, 라쏘의 축소가 남은 계수를 작게 만들어 RSS를 부풀리고, 그 손실을 메우려 모형이 변수를 더 넣기 때문이다.

    대응책은 두 가지다. **사후 라쏘**로 축소를 되돌린 뒤 정보기준을 계산하거나, 확장 BIC(EBIC)처럼 $p$가 클 때 벌점을 더 키운 변형을 쓴다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
"AIC는 LOOCV와 점근적으로 동등하다"는 주장의 의미를 정확히 서술하고, 유한표본에서 두 방법이 갈릴 수 있는 이유를 설명하라.

</div>

??? success "풀이"
    **점근적 동등성의 정확한 의미.** 선형 평활자에서 LOOCV 오차를 1차 근사하면

    $$
    \text{CV}_{\text{LOO}} = \frac{1}{n}\sum_i \left(\frac{y_i-\hat y_i}{1-H_{ii}}\right)^2
    \approx \frac{\text{RSS}}{n}\left(1 + \frac{2\,\text{tr}(\mathbf{H})}{n}\right)
    = \frac{\text{RSS}}{n} + \frac{2\,\text{df}\cdot\text{RSS}}{n^2}
    $$

    를 얻는다($H_{ii}$가 작을 때 $(1-H_{ii})^{-2} \approx 1 + 2H_{ii}$). $\text{RSS}/n \approx \sigma^2$으로 두면

    $$
    n\,\text{CV}_{\text{LOO}}/\sigma^2 \approx \frac{\text{RSS}}{\sigma^2} + 2\,\text{df} = \text{AIC}
    $$

    이다. 즉 **AIC는 LOOCV의 2차 테일러 근사**다.

    **유한표본에서 갈리는 이유는 세 가지다.**

    | 근사 단계 | 언제 깨지는가 |
    |:---|:---|
    | $(1-H_{ii})^{-2} \approx 1 + 2H_{ii}$ | 지렛값이 클 때, 즉 $p/n$이 클 때 |
    | $\text{RSS}/n \approx \sigma^2$ | $\hat\sigma^2$ 추정이 나쁠 때 |
    | 개별 $H_{ii}$를 평균 $\text{df}/n$으로 대체 | 지렛값이 불균등할 때(이상값, 불균형 설계) |

    **실무적 함의:** $p/n$이 작고($< 0.1$) 설계가 균형 잡혀 있으면 AIC와 LOOCV가 거의 같은 답을 준다. 이때는 훨씬 싼 AIC를 쓰는 것이 합리적이다.

    $p/n$이 크거나 지렛값이 불균등하면 두 방법이 갈리며, 이때는 근사를 쓰지 않는 교차검증이 옳다. 특히 **가우스 가정이 의심스러우면 AIC의 근거 자체가 무너지므로** 교차검증이 유일한 선택이다.

---

## 정리하며

정보기준은 모수 개수를 유효자유도로 대체하여 정칙화 모형으로 확장된다. 능형에서 $\text{df}(\lambda)$는 축소인자의 합이고, 라쏘에서는 0이 아닌 계수의 개수다. AIC는 최적 예측을, BIC는 모형선택 일치성을 겨냥하며 둘 다 오차분산 추정값을 요구한다. 교차검증이 여전히 가장 견고하고 널리 쓰이지만, 정보기준은 정칙화 경로를 탐색할 때 특히 유용한 계산 효율적 대안이다.
