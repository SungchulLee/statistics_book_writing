# 능형 자취와 λ의 선택

능형 추정량은 편향과 분산의 절충을 조절하는 정칙화 모수 $\lambda$에 의존한다. $\lambda$가 너무 작으면 정칙화가 부족하고, 너무 크면 모든 계수를 0 쪽으로 지나치게 축소한다. 이 절에서는 $\lambda$를 고르는 세 가지 접근 — 능형 자취(시각적 진단), 교차검증(표준적인 자료 기반 방법), 일반화 교차검증(효율적인 해석적 근사) — 을 제시한다.

## 능형 자취

**능형 자취**는 각 계수 $\hat{\beta}_j(\lambda)$를 $\lambda$(또는 $\log\lambda$)의 함수로 그린 그림이다. 정칙화 강도에 따라 능형 해가 어떻게 변하는지 시각적으로 요약한다.

$\lambda$가 0에서 커질 때

- $\lambda = 0$에서 계수는 OLS 값과 같다.
- 처음에는 계수가 빠르게 변할 수 있으며, 특히 다중공선성으로 부풀려진 계수가 그렇다.
- 결국 계수가 안정되고 매끄럽게 0으로 수렴한다.

능형 자취는 계수가 안정되었지만(격렬한 변동이 사라졌지만) 아직 0에 지나치게 가까워지지는 않은 $\lambda$를 찾는 데 도움을 준다. 이 안정 구간이 정칙화가 선형종속의 영향을 성공적으로 제거하면서 과도한 편향은 들이지 않은 지점이다.

!!! note "능형 자취의 해석"
    계수 경로가 평평해지고 대체로 나란히 가는 구간을 찾는다. 정칙화에 의한 분산 감소가 이미 확보되었고 $\lambda$를 더 키우면 편향만 늘어나는 "팔꿈치" 구간이다. 능형회귀 초기 응용에서는 능형 자취가 $\lambda$를 고르는 주된 방법이었다.

## K-겹 교차검증

교차검증은 $\lambda$를 고르는 자료 기반의 객관적 방법을 제공한다. 각 후보 $\lambda$에 대해 검정오차를 추정한다.

**알고리즘.** $\lambda$의 격자 $\lambda_1 < \cdots < \lambda_M$이 주어졌을 때

1. 자료를 크기가 비슷한 $K$개의 겹으로 무작위 분할한다.
2. 각 $\lambda_m$과 각 겹 $k$에 대해, 겹 $k$를 뺀 자료로 능형 모형을 적합하고 겹 $k$의 반응을 예측하여 예측오차를 계산한다.
3. 겹에 걸쳐 평균낸다.

$$
\text{CV}(\lambda_m) = \frac{1}{K}\sum_{k=1}^K \text{MSE}_k(\lambda_m)
$$

4. $\hat{\lambda} = \arg\min_{\lambda_m} \text{CV}(\lambda_m)$을 고른다.

CV 오차 곡선은 대개 U자 모양이다. 아주 작은 $\lambda$에서 높고(과적합), 최솟값까지 감소했다가, 큰 $\lambda$에서 다시 증가한다(과소적합).

## 1-표준오차 규칙

CV 곡선의 최솟값이 $\lambda_{\min}$을 정하지만 CV 추정값 자체에 잡음이 있다. **1-표준오차 규칙**은 더 보수적인 선택을 제공한다.

$$
\lambda_{1\text{SE}} = \max\bigl\{\lambda : \text{CV}(\lambda) \leq \text{CV}(\lambda_{\min}) + \text{SE}(\lambda_{\min})\bigr\}
$$

더 복잡한 모형을 지지하는 증거가 통계적으로 뚜렷하지 않을 때 더 단순한 모형(더 강한 정칙화)을 택하는 규칙이다.

!!! tip "1-SE 규칙을 언제 쓰는가"
    예측 정확도의 미미한 개선보다 해석 가능성이나 안정성이 중요할 때 $\lambda_{1\text{SE}}$를 쓴다. 예측 정확도가 주된 목표라면 $\lambda_{\min}$을 쓴다.

## 하나 남기기 교차검증

능형회귀에서 LOOCV는 모형을 $n$번 재적합하지 않아도 되는 닫힌 형태의 지름길을 갖는다.

$$
\text{CV}_{\text{LOO}}(\lambda) = \frac{1}{n}\sum_{i=1}^n\left(\frac{y_i - \hat{y}_i(\lambda)}{1 - h_{ii}(\lambda)}\right)^2
$$

여기서 $h_{ii}(\lambda)$는 **모자 행렬**

$$
\mathbf{H}(\lambda) = \mathbf{X}(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top
$$

의 $i$번째 대각원소다. 이 공식은 $\lambda$당 한 번의 적합만 요구하므로 능형회귀에서 LOOCV가 계산적으로 효율적이다.

## 일반화 교차검증

**일반화 교차검증**(GCV)은 개별 지렛값 $h_{ii}(\lambda)$를 그 평균으로 대체한다.

$$
\text{GCV}(\lambda) = \frac{\text{RSS}(\lambda)/n}{\bigl(1 - \text{df}(\lambda)/n\bigr)^2}
$$

여기서 $\text{df}(\lambda) = \text{tr}[\mathbf{H}(\lambda)] = \sum_j d_j^2/(d_j^2 + \lambda)$이다. GCV의 장점은 자료의 직교 회전에 대해 불변이고, $n$개의 지렛값 대신 유효자유도 하나로 계산되며, 일정 조건에서 예측에 대해 점근적으로 최적이라는 점이다.

## 실무적 고려

**$\lambda$ 격자.** 로그 격자를 쓴다. 예를 들어 $10^{-4}$부터 $10^{4}$까지 100개 점을 잡으면 여러 자릿수에 걸쳐 충분한 해상도를 얻는다.

**표준화.** 능형 해를 계산하기 전에 설명변수를 표준화(평균 0, 분산 1)해야 벌점 $\lambda\|\boldsymbol{\beta}\|_2^2$이 모든 계수를 같은 척도에서 벌한다.

| 방법 | 계산량 | 장점 | 단점 |
|---|---|---|---|
| 능형 자취 | 시각적 판단 | 직관적, 계수 경로를 드러냄 | 주관적, 자동화되지 않음 |
| $K$-겹 CV | $K \times M$번 적합 | 표준적, 잘 이해됨 | 계산이 무거움 |
| LOOCV (닫힌 형태) | $M$번 적합 | 정확, 분할 난수 없음 | 개별 관측에 과적합할 수 있음 |
| GCV | 해석적 공식 | 효율적, 회전 불변 | 근사(지렛값을 평균냄) |

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
$\lambda_{\min}$과 $\lambda_{1\text{SE}}$가 실제로 얼마나 다른지, 그리고 유효자유도로 환산하면 어떤 의미인지 확인하라. $\rho = 0.9$인 표준화된 설명변수 15개 중 4개만 참인 자료($n = 80$)에서 10-겹 교차검증을 수행하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    rng = np.random.default_rng(31)

    n, p, rho = 80, 15, 0.9
    S = rho*np.ones((p, p)) + (1-rho)*np.eye(p)
    X = rng.normal(size=(n, p)) @ np.linalg.cholesky(S).T
    X -= X.mean(0); X /= X.std(0)
    beta = np.zeros(p); beta[:4] = [3, -2, 1.5, 1]
    y = X @ beta + rng.normal(0, 1, n); y -= y.mean()

    lams = np.logspace(-2, 3, 60)
    kf = KFold(10, shuffle=True, random_state=0)
    mu, se = [], []
    for lam in lams:
        errs = [((y[te] - X[te] @ Ridge(alpha=lam, fit_intercept=False)
                  .fit(X[tr], y[tr]).coef_)**2).mean() for tr, te in kf.split(X)]
        mu.append(np.mean(errs)); se.append(np.std(errs, ddof=1)/np.sqrt(10))
    mu, se = np.array(mu), np.array(se)
    i = mu.argmin(); j = np.max(np.where(mu <= mu[i] + se[i])[0])
    ```

    | 규칙 | $\lambda$ | CV 오차 | $\text{df}(\lambda)$ |
    |:---|---:|---:|---:|
    | $\lambda_{\min}$ | **1.081** | 1.172 | **12.52** |
    | $\lambda_{1\text{SE}}$ | **7.610** | 1.386 | **6.88** |

    **$\lambda$가 7배 커지고 유효자유도는 절반이 된다.** CV 오차는 $1.172 \to 1.386$으로 18% 나빠지지만, 그 차이는 표준오차 $0.226$ 안에 있다.

    **1-SE 규칙이 사는 것과 파는 것.** 파는 것은 CV 오차 18%다. 사는 것은 모형 복잡도의 절반이다. 자료가 $p = 15$ 중 참 변수 $4$개인 상황임을 감안하면 $\text{df} = 6.88$이 $12.52$보다 진실에 가깝다.

    **표준오차 $0.226$이 평균 $1.172$의 19%라는 점에 주목하라.** CV 추정 자체가 그만큼 불안정하다는 뜻이며, 그래서 최솟값을 곧이곧대로 믿지 않는 규칙이 필요하다. 겹 수를 늘리거나 교차검증을 반복하면 이 표준오차가 줄어든다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
LOOCV의 닫힌 형태 지름길이 실제로 $n$번 재적합한 결과와 일치하는지 확인하라. 왜 능형에는 이 지름길이 있고 라쏘에는 없는가?

</div>

??? success "풀이"
    ```python
    import numpy as np
    from sklearn.linear_model import Ridge

    def loocv_shortcut(X, y, lam):
        n, p = X.shape
        A = np.linalg.inv(X.T @ X + lam*np.eye(p))
        H = X @ A @ X.T
        yhat = H @ y; h = np.diag(H)
        return np.mean(((y - yhat)/(1 - h))**2)

    def loocv_brute(X, y, lam):
        n = len(y); errs = []
        for i in range(n):
            m = np.ones(n, bool); m[i] = False
            b = Ridge(alpha=lam, fit_intercept=False).fit(X[m], y[m]).coef_
            errs.append((y[i] - X[i] @ b)**2)
        return np.mean(errs)

    for lam in (0.1, 1.0, 10.0):
        assert np.isclose(loocv_shortcut(X, y, lam), loocv_brute(X, y, lam))
    ```

    두 값이 기계 정밀도까지 일치한다.

    **왜 능형에는 지름길이 있는가.** 능형은 **선형 평활자**다. 적합값이 $\hat{\mathbf{y}} = \mathbf{H}(\lambda)\mathbf{y}$ 형태이고 $\mathbf{H}$가 $\mathbf{y}$에 의존하지 않는다. 이 성질이 항등식

    $$
    y_i - \hat{y}_i^{(-i)} = \frac{y_i - \hat{y}_i}{1 - H_{ii}}
    $$

    를 성립시킨다.

    **왜 라쏘에는 없는가.** 라쏘 해는 $\mathbf{y}$의 **비선형** 함수다. 활성집합(0이 아닌 계수의 집합)이 $\mathbf{y}$에 의존하므로, 관측 하나를 빼면 활성집합 자체가 바뀔 수 있다. $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$로 쓸 수 있는 고정된 $\mathbf{H}$가 존재하지 않는다.

    실무적 결과: `RidgeCV`는 기본적으로 LOOCV를 쓰고 매우 빠르다. `LassoCV`는 $K$-겹 교차검증을 실제로 수행해야 하므로 훨씬 느리다. 대신 라쏘는 경로 알고리즘(LARS나 좌표하강의 온기 시작)으로 전체 $\lambda$ 격자를 한 번에 계산하여 이를 상쇄한다.

---

## 정리하며

능형회귀의 $\lambda$ 선택은 적합도와 복잡도의 균형을 요구한다. 능형 자취는 계수 안정성에 대한 시각적 직관을, 교차검증은 검정오차의 자료 기반 추정을 제공하며, 1-표준오차 규칙은 간결성을 선호한다. GCV는 효율적인 해석적 근사를 제공한다. 실무에서는 로그 격자 위의 $K$-겹 교차검증에 1-SE 규칙을 결합하는 방식이 가장 널리 쓰인다.
