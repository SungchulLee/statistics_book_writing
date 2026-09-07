# 능형회귀 (L2 정칙화)


## 동기: OLS의 문제

보통최소제곱(OLS)은 잔차제곱합을 최소화한다.

$$
\hat{\boldsymbol{\beta}}_{\text{OLS}} = \arg\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2
$$

닫힌 형태 해는 $\hat{\boldsymbol{\beta}}_{\text{OLS}} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}$이다.

이 추정량은 **불편**이며 Gauss-Markov 정리에 의해 모든 선형 불편추정량 중 최소분산(BLUE)이다. 그러나 다음 상황에서 성능이 나빠질 수 있다.

1. **다중공선성.** 설명변수가 강하게 상관되면 $\mathbf{X}^\top\mathbf{X}$가 거의 특이행렬이 된다. 자료의 작은 변화가 $\hat{\boldsymbol{\beta}}$의 큰 변화를 낳아 분산이 부풀려진다.
2. **고차원 상황.** $p$가 $n$에 가깝거나 넘으면 OLS가 정의되지 않거나 심하게 과적합한다.
3. **예측 정확도.** 편향-분산 절충(6장)에 따르면 약간의 편향을 들여와 분산을 크게 줄이면 MSE와 예측 정확도가 개선될 수 있다.

## 능형회귀의 정식화

**능형회귀**(Hoerl and Kennard, 1970)는 OLS 목적함수에 L2 벌점을 더한다.

$$
\hat{\boldsymbol{\beta}}_{\text{ridge}} = \arg\min_{\boldsymbol{\beta}} \left\{ \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda \|\boldsymbol{\beta}\|^2 \right\}
$$

여기서 $\lambda \geq 0$이 **정칙화 모수**(조정모수, 벌점강도)이고 $\|\boldsymbol{\beta}\|^2 = \sum_{j=1}^p \beta_j^2$이다.

!!! note "중심화와 척도조정"
    절편 $\beta_0$은 보통 **벌점하지 않는다.** 실무에서는 능형회귀를 적용하기 전에 $\mathbf{y}$를 중심화하고 $\mathbf{X}$의 각 열을 평균 0, 분산 1로 표준화하여 벌점이 모든 계수를 동등하게 다루도록 한다.

## 닫힌 형태 해

$$
\hat{\boldsymbol{\beta}}_{\text{ridge}} = (\mathbf{X}^\top\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}
$$

*유도.* 목적함수의 기울기를 0으로 두면

$$
-2\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) + 2\lambda\boldsymbol{\beta} = \mathbf{0}
\;\Longrightarrow\;
(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})\boldsymbol{\beta} = \mathbf{X}^\top\mathbf{y}
$$

$\lambda > 0$에서 $\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}$가 양정치이므로($\mathbf{X}^\top\mathbf{X}$가 특이행렬이어도) 해가 언제나 존재하고 유일하다.

## 기하적 해석

능형회귀는 **제약최적화**로 동등하게 표현된다.

$$
\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 \quad \text{제약} \quad \|\boldsymbol{\beta}\|^2 \leq t
$$

- OLS 목적함수는 계수공간에서 타원형 등고선을 정의한다
- 제약 $\|\boldsymbol{\beta}\|^2 \leq t$는 **구**($p$차원에서는 초구)이다
- 능형 해는 가장 작은 타원 등고선이 구에 닿는 점이다

이 구 제약은 모든 계수를 0 쪽으로 축소하지만 일반적으로 어떤 것도 정확히 0으로 만들지 **않는다.** 자세한 논의는 [기하적 해석](geometry.md)을 보라.

## SVD 해석

특이값분해 $\mathbf{X} = \mathbf{U}\mathbf{D}\mathbf{V}^\top$($d_1 \geq \cdots \geq d_p \geq 0$)을 쓰면

$$
\hat{\boldsymbol{\beta}}_{\text{ridge}} = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda} \cdot \frac{\mathbf{u}_j^\top \mathbf{y}}{d_j}\,\mathbf{v}_j
$$

이며 OLS는

$$
\hat{\boldsymbol{\beta}}_{\text{OLS}} = \sum_{j=1}^p \frac{\mathbf{u}_j^\top \mathbf{y}}{d_j}\,\mathbf{v}_j
$$

이다. 인자 $\frac{d_j^2}{d_j^2 + \lambda}$가 0과 1 사이의 **축소인자**다. 특이값이 작은 성분(불안정한 방향)이 가장 많이 축소되며, 바로 그곳이 OLS의 분산이 큰 곳이므로 능형회귀가 추정을 안정시킨다.

## 능형회귀의 편향과 분산

$$
\text{Bias}(\hat{\boldsymbol{\beta}}_{\text{ridge}}) = -\lambda(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\boldsymbol{\beta}
$$

$$
\text{Cov}(\hat{\boldsymbol{\beta}}_{\text{ridge}}) = \sigma^2 (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{X}(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}
$$

$\lambda$가 커지면 편향은 커지고 분산은 작아지며, 전체 MSE를 최소화하는 최적 $\lambda^*$가 존재한다.

**정리 (Hoerl and Kennard, 1970).** $\text{MSE}(\hat{\boldsymbol{\beta}}_{\text{ridge}}) < \text{MSE}(\hat{\boldsymbol{\beta}}_{\text{OLS}})$가 되는 $\lambda > 0$이 언제나 존재한다.

## $\lambda$의 선택: 교차검증

최적 $\lambda$는 알려져 있지 않으므로 자료에서 추정해야 한다. 표준적인 접근은 **$k$-겹 교차검증**이다.

1. 자료를 $k$개 겹으로 나눈다(대개 $k = 5$ 또는 $10$).
2. 격자 위의 각 후보 $\lambda$에 대해, 각 겹을 검정집합으로 삼아 예측오차를 기록하고 평균낸다.
3. $\hat{\lambda} = \arg\min_\lambda \text{CV}(\lambda)$를 고른다.

!!! tip "1-표준오차 규칙"
    CV 오차가 최소인 $\lambda$ 대신, 최솟값에서 1 표준오차 이내에 있는 가장 큰 $\lambda$를 고른다. 예측 성능은 비슷하면서 더 간결한 모형을 얻는다.

능형회귀의 LOOCV에는 닫힌 형태 지름길이 있다.

$$
\text{CV}_{\text{LOO}}(\lambda) = \frac{1}{n}\sum_{i=1}^n \left(\frac{y_i - \hat{y}_i(\lambda)}{1 - h_{ii}(\lambda)}\right)^2
$$

여기서 $h_{ii}(\lambda) = [\mathbf{X}(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top]_{ii}$이다. 자세한 내용은 [능형 자취와 $\lambda$의 선택](lambda_selection.md)을 보라.

## 베이즈 MAP로서의 능형회귀

능형회귀는 가우스 사전분포를 둔 베이즈 선형모형의 **사후최빈값**(MAP) 추정과 동등하다.

$$
\boldsymbol{\beta} \sim N(\mathbf{0}, \tau^2 \mathbf{I}), \quad \mathbf{y} \mid \boldsymbol{\beta} \sim N(\mathbf{X}\boldsymbol{\beta}, \sigma^2\mathbf{I})
$$

$$
\hat{\boldsymbol{\beta}}_{\text{MAP}} = \Bigl(\mathbf{X}^\top\mathbf{X} + \frac{\sigma^2}{\tau^2}\mathbf{I}\Bigr)^{-1}\mathbf{X}^\top\mathbf{y}
$$

$\lambda = \sigma^2/\tau^2$로 두면 능형 해가 된다. 자세한 내용은 [베이즈 해석](bayesian.md)을 보라.

## 유효자유도

$$
\text{df}(\lambda) = \text{tr}\left[\mathbf{X}(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\right] = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda}
$$

$\lambda \to 0$에서 $\text{df} \to p$(OLS), $\lambda \to \infty$에서 $\text{df} \to 0$(상수모형)이다. 서로 다른 $\lambda$의 모형 복잡도를 직접 비교할 수 있게 해준다.

## 주요 성질 요약

| 성질 | 값 |
|---|---|
| 벌점 | $\lambda\sum_{j=1}^p \beta_j^2$ (L2) |
| 해 | 닫힌 형태: $(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$ |
| 변수선택 | 없음(축소하되 0으로 만들지 않음) |
| 다중공선성 처리 | 가능 |
| $p > n$에서 작동 | 가능 |
| 베이즈 해석 | $\boldsymbol{\beta}$에 대한 가우스 사전분포 |
| 기하적 제약 | $\ell_2$ 공(구) |

## 연습문제

**연습문제 1.**
능형회귀의 최적화 문제와 닫힌 형태 해를 쓰라. $\lambda$는 해에 어떤 영향을 주는가?

??? success "풀이"
    최적화 문제는

    $$
    \hat{\boldsymbol{\beta}}_{\text{ridge}} = \arg\min_{\boldsymbol{\beta}} \left\{ \lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert^2 + \lambda \lVert \boldsymbol{\beta} \rVert^2 \right\}
    $$

    이고 닫힌 형태 해는

    $$
    \hat{\boldsymbol{\beta}}_{\text{ridge}} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}
    $$

    이다. $\lambda \to 0$이면 $\hat{\boldsymbol{\beta}}_{\text{ridge}} \to \hat{\boldsymbol{\beta}}_{\text{OLS}}$, $\lambda \to \infty$이면 $\hat{\boldsymbol{\beta}}_{\text{ridge}} \to \mathbf{0}$이다. $\lambda$가 편향-분산 절충을 조절하며, 클수록 편향이 커지고 분산이 줄어든다.

---

**연습문제 2.**
능형회귀가 편향되어 있음을 보이고, 편향을 $\lambda$와 참 $\boldsymbol{\beta}$로 유도하라.

??? success "풀이"
    $$
    E[\hat{\boldsymbol{\beta}}_{\text{ridge}}] = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{X}\boldsymbol{\beta} = \mathbf{W}\boldsymbol{\beta} \neq \boldsymbol{\beta}
    $$

    여기서 $\mathbf{W} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{X}$이다. 편향은

    $$
    \text{Bias} = (\mathbf{W} - \mathbf{I})\boldsymbol{\beta} = -\lambda(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\boldsymbol{\beta}
    $$

    이다. 편향은 $\lambda$와 $\boldsymbol{\beta}$의 크기에 따라 커진다. 그럼에도 분산 감소가 편향의 제곱을 넘어서면 전체 MSE는 OLS보다 작아질 수 있다. $\square$

    **편향의 방향이 $-\boldsymbol{\beta}$ 쪽이라는 점에 주목하라.** 즉 편향은 언제나 계수를 0 쪽으로 당긴다. $\boldsymbol{\beta}$가 클수록 편향도 크며, 이것이 참 계수가 큰 상황에서 $\lambda$를 작게 잡아야 하는 이유다([과적합과 편향-분산 절충](../motivation/overfitting.md) 연습문제 2 참조).

---

**연습문제 3.**
능형회귀가 다중공선성에 도움이 되는 이유를 설명하라. $\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}$의 조건수는 어떻게 되는가?

??? success "풀이"
    다중공선성은 $\mathbf{X}^\top\mathbf{X}$가 0에 가까운 고윳값을 가져 거의 특이행렬이 되고 조건수 $\kappa = d_1/d_p$가 커진다는 뜻이다.

    $\lambda\mathbf{I}$를 더하면 모든 고윳값이 $\lambda$만큼 이동한다. 새 조건수는

    $$
    \kappa_{\text{ridge}} = \frac{d_1 + \lambda}{d_p + \lambda} < \frac{d_1}{d_p} = \kappa_{\text{OLS}}
    $$

    이다. 최소 고윳값이 0에 가까운 $d_p$에서 $d_p + \lambda$로 올라가 수치 안정성이 크게 개선되고 $\hat{\boldsymbol{\beta}}$의 분산이 줄어든다.

    구체적인 수치는 [불량조건 계획행렬](../motivation/ill_conditioning.md) 연습문제 1에 있다. $\rho = 0.999$인 등상관 구조에서 $\lambda = 1$이 조건수를 $7993 \to 8.98$로 890배 줄인다.

---

**연습문제 4.**
표준화된 자료에 $\lambda = 1$로 능형회귀를 적합했더니 $\hat{\beta}_1 = 0.45$, $\hat{\beta}_2 = 0.38$을 얻었다. OLS는 $\hat{\beta}_1 = 1.2$, $\hat{\beta}_2 = -0.8$을 준다. 이 차이를 해석하라.

??? success "풀이"
    OLS 추정값($1.2$와 $-0.8$)은 크기가 크고 부호가 반대인데, 이는 다중공선성의 전형적 특징이다. 설명변수가 상관되어 있어 OLS가 서로를 부분적으로 상쇄하는 불안정하고 부풀려진 계수를 낸다.

    능형회귀($0.45$와 $0.38$)는 두 계수를 모두 0 쪽으로 축소하고 크기를 비슷하게 만든다. $\hat{\beta}_2$의 부호가 $-0.8$에서 $+0.38$로 뒤집힌 것은, OLS의 음수 계수가 진짜 음의 관계가 아니라 선형종속의 산물이었음을 시사한다. 능형회귀는 약간의 편향을 대가로 더 안정적이고 해석 가능한 추정값을 낸다.

    !!! warning "부호 반전을 '능형이 고쳤다'로 읽을 때의 주의"
        능형회귀가 부호를 "바로잡았다"고 단정할 수는 없다. 능형은 계수를 0 쪽으로 당기므로, 참값이 실제로 음수여도 크기가 작으면 양수 쪽으로 넘어갈 수 있다.

        확실히 말할 수 있는 것은 **자료가 두 계수를 개별적으로 결정하지 못한다**는 사실뿐이다. 계수의 합 $\hat\beta_1 + \hat\beta_2$는 OLS에서 $0.4$, 능형에서 $0.83$으로 그나마 안정적이다. 다중공선성이 있을 때는 개별 계수가 아니라 안정적인 조합을 해석해야 한다([다중공선성과 정칙화](../motivation/multicollinearity.md) 연습문제 2 참조).
