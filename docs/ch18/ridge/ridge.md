# 능형회귀 (L2 정칙화)


## 1. 동기: OLS의 문제

보통최소제곱(OLS)은 잔차제곱합을 최소화한다.

$$
\hat{\boldsymbol{\beta}}_{\text{OLS}} = \arg\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2
$$

닫힌 형태 해는 $\hat{\boldsymbol{\beta}}_{\text{OLS}} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}$이다.

이 추정량은 **불편**이며 Gauss-Markov 정리에 의해 모든 선형 불편추정량 중 최소분산(BLUE)이다. 그러나 다음 상황에서 성능이 나빠질 수 있다.

1. **다중공선성.** 설명변수가 강하게 상관되면 $\mathbf{X}^\top\mathbf{X}$가 거의 특이행렬이 된다. 자료의 작은 변화가 $\hat{\boldsymbol{\beta}}$의 큰 변화를 낳아 분산이 부풀려진다.
2. **고차원 상황.** $p$가 $n$에 가깝거나 넘으면 OLS가 정의되지 않거나 심하게 과적합한다.
3. **예측 정확도.** 편향-분산 절충(5장)에 따르면 약간의 편향을 들여와 분산을 크게 줄이면 MSE와 예측 정확도가 개선될 수 있다.

---

## 2. 능형회귀의 정식화

**능형회귀**(Hoerl and Kennard, 1970)는 OLS 목적함수에 L2 벌점을 더한다.

$$
\hat{\boldsymbol{\beta}}_{\text{ridge}} = \arg\min_{\boldsymbol{\beta}} \left\{ \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda \|\boldsymbol{\beta}\|^2 \right\}
$$

여기서 $\lambda \geq 0$이 **정칙화 모수**(조정모수, 벌점강도)이고 $\|\boldsymbol{\beta}\|^2 = \sum_{j=1}^p \beta_j^2$이다.

!!! note "중심화와 척도조정"
    절편 $\beta_0$은 보통 **벌점하지 않는다.** 실무에서는 능형회귀를 적용하기 전에 $\mathbf{y}$를 중심화하고 $\mathbf{X}$의 각 열을 평균 0, 분산 1로 표준화하여 벌점이 모든 계수를 동등하게 다루도록 한다.

---

## 3. 닫힌 형태 해

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

---

## 4. 기하적 해석

능형회귀는 **제약최적화**로 동등하게 표현된다.

$$
\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 \quad \text{제약} \quad \|\boldsymbol{\beta}\|^2 \leq t
$$

- OLS 목적함수는 계수공간에서 타원형 등고선을 정의한다
- 제약 $\|\boldsymbol{\beta}\|^2 \leq t$는 **구**($p$차원에서는 초구)이다
- 능형 해는 가장 작은 타원 등고선이 구에 닿는 점이다

이 구 제약은 모든 계수를 0 쪽으로 축소하지만 일반적으로 어떤 것도 정확히 0으로 만들지 **않는다.** 자세한 논의는 [기하적 해석](geometry.md)을 보라.

![공선성 때문에 평평해진 최소제곱 골짜기와 벌점을 더해 되살아난 그릇 모양](./img/rss_valley.png)

벌점이 하는 일을 목적함수의 모양으로 보면 가장 빠르다. 상관이 $0.99$인 설명변수 두 개, $n = 50$, 참 계수 $(1, 1)$인 자료에서 두 목적함수의 등고선을 그렸다. 왼쪽 최소제곱 목적함수는 **골짜기**다. $\beta_1 + \beta_2$가 일정한 대각선 방향으로는 값이 거의 변하지 않아 등고선이 길게 늘어졌다. $\mathbf{X}^\top\mathbf{X}$의 고윳값이 $121.03$과 $0.48$로 조건수가 $252$인데, 등고선 타원의 장축과 단축 길이의 비가 정확히 그 제곱근인 $16$배다.

골짜기가 평평하다는 것은 **자료가 바닥의 위치를 모른다**는 뜻이다. 이 표본에서 OLS가 찍은 바닥은 $(0.36,\ 1.72)$로 참값 $(1, 1)$에서 꽤 멀다. 자료를 다시 뽑으면 이 점은 같은 대각선을 따라 멀리 미끄러진다. 골짜기 바닥이 평평하므로 미끄러져도 목적함숫값이 거의 오르지 않기 때문이다.

오른쪽은 같은 자료에 $20\|\boldsymbol{\beta}\|^2$을 더한 것뿐이다. 등고선이 **그릇**이 되었다. 벌점항 $20(\beta_1^2 + \beta_2^2)$의 등고선은 원점을 중심으로 한 원이므로, 이것을 더하면 모든 방향의 곡률이 $20$씩 올라간다. 고윳값이 $121.03 \to 141.03$, $0.48 \to 20.48$로 바뀌어 조건수가 $252$에서 $6.9$로 떨어진다. 큰 쪽은 거의 그대로이고 작은 쪽만 40배 커졌다는 점이 핵심이다. **벌점은 자료가 이미 잘 아는 방향은 건드리지 않고, 모르는 방향에만 곡률을 빌려준다.**

그릇의 바닥은 $(0.878,\ 0.908)$로 참값 $(1, 1)$ 바로 옆이다. 물론 이 한 표본의 결과일 뿐이고, 편향이 들어갔으므로 평균적으로는 참값보다 원점 쪽에 놓인다. 그러나 **바닥이 한 점으로 확실히 정해진다**는 성질은 표본과 무관하게 언제나 성립한다. $\mathbf{X}^\top\mathbf{X}$가 특이행렬이어서 왼쪽 골짜기가 아예 평평한 직선이 되어 버리는 $p > n$ 상황에서도 마찬가지다.

---

## 5. SVD 해석

특이값분해 $\mathbf{X} = \mathbf{U}\mathbf{D}\mathbf{V}^\top$($d_1 \geq \cdots \geq d_p \geq 0$)을 쓰면

$$
\hat{\boldsymbol{\beta}}_{\text{ridge}} = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda} \cdot \frac{\mathbf{u}_j^\top \mathbf{y}}{d_j}\,\mathbf{v}_j
$$

이며 OLS는

$$
\hat{\boldsymbol{\beta}}_{\text{OLS}} = \sum_{j=1}^p \frac{\mathbf{u}_j^\top \mathbf{y}}{d_j}\,\mathbf{v}_j
$$

이다. 인자 $\frac{d_j^2}{d_j^2 + \lambda}$가 0과 1 사이의 **축소인자**다. 특이값이 작은 성분(불안정한 방향)이 가장 많이 축소되며, 바로 그곳이 OLS의 분산이 큰 곳이므로 능형회귀가 추정을 안정시킨다.

---

## 6. 능형회귀의 편향과 분산

$$
\text{Bias}(\hat{\boldsymbol{\beta}}_{\text{ridge}}) = -\lambda(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\boldsymbol{\beta}
$$

$$
\text{Cov}(\hat{\boldsymbol{\beta}}_{\text{ridge}}) = \sigma^2 (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{X}(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}
$$

$\lambda$가 커지면 편향은 커지고 분산은 작아지며, 전체 MSE를 최소화하는 최적 $\lambda^*$가 존재한다.

**정리 (Hoerl and Kennard, 1970).** $\text{MSE}(\hat{\boldsymbol{\beta}}_{\text{ridge}}) < \text{MSE}(\hat{\boldsymbol{\beta}}_{\text{OLS}})$가 되는 $\lambda > 0$이 언제나 존재한다.

---

## 7. λ의 선택: 교차검증

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

---

## 8. 베이즈 MAP로서의 능형회귀

능형회귀는 가우스 사전분포를 둔 베이즈 선형모형의 **사후최빈값**(MAP) 추정과 동등하다.

$$
\boldsymbol{\beta} \sim N(\mathbf{0}, \tau^2 \mathbf{I}), \quad \mathbf{y} \mid \boldsymbol{\beta} \sim N(\mathbf{X}\boldsymbol{\beta}, \sigma^2\mathbf{I})
$$

$$
\hat{\boldsymbol{\beta}}_{\text{MAP}} = \Bigl(\mathbf{X}^\top\mathbf{X} + \frac{\sigma^2}{\tau^2}\mathbf{I}\Bigr)^{-1}\mathbf{X}^\top\mathbf{y}
$$

$\lambda = \sigma^2/\tau^2$로 두면 능형 해가 된다. 자세한 내용은 [베이즈 해석](bayesian.md)을 보라.

---

## 9. 유효자유도

$$
\text{df}(\lambda) = \text{tr}\left[\mathbf{X}(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\right] = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda}
$$

$\lambda \to 0$에서 $\text{df} \to p$(OLS), $\lambda \to \infty$에서 $\text{df} \to 0$(상수모형)이다. 서로 다른 $\lambda$의 모형 복잡도를 직접 비교할 수 있게 해준다.

---

## 10. 주요 성질 요약

| 성질 | 값 |
|---|---|
| 벌점 | $\lambda\sum_{j=1}^p \beta_j^2$ (L2) |
| 해 | 닫힌 형태: $(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$ |
| 변수선택 | 없음(축소하되 0으로 만들지 않음) |
| 다중공선성 처리 | 가능 |
| $p > n$에서 작동 | 가능 |
| 베이즈 해석 | $\boldsymbol{\beta}$에 대한 가우스 사전분포 |
| 기하적 제약 | $\ell_2$ 공(구) |

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
능형회귀의 최적화 문제와 닫힌 형태 해를 쓰라. $\lambda$는 해에 어떤 영향을 주는가?

</div>

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

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
표준화된 자료에 $\lambda = 1$로 능형회귀를 적합했더니 $\hat{\beta}_1 = 0.45$, $\hat{\beta}_2 = 0.38$을 얻었다. OLS는 $\hat{\beta}_1 = 1.2$, $\hat{\beta}_2 = -0.8$을 준다. 이 차이를 해석하라.

</div>

??? success "풀이"
    OLS 추정값($1.2$와 $-0.8$)은 크기가 크고 부호가 반대인데, 이는 다중공선성의 전형적 특징이다. 설명변수가 상관되어 있어 OLS가 서로를 부분적으로 상쇄하는 불안정하고 부풀려진 계수를 낸다.

    능형회귀($0.45$와 $0.38$)는 두 계수를 모두 0 쪽으로 축소하고 크기를 비슷하게 만든다. $\hat{\beta}_2$의 부호가 $-0.8$에서 $+0.38$로 뒤집힌 것은, OLS의 음수 계수가 진짜 음의 관계가 아니라 선형종속의 산물이었음을 시사한다. 능형회귀는 약간의 편향을 대가로 더 안정적이고 해석 가능한 추정값을 낸다.

    !!! warning "부호 반전을 '능형이 고쳤다'로 읽을 때의 주의"
        능형회귀가 부호를 "바로잡았다"고 단정할 수는 없다. 능형은 계수를 0 쪽으로 당기므로, 참값이 실제로 음수여도 크기가 작으면 양수 쪽으로 넘어갈 수 있다.

        확실히 말할 수 있는 것은 **자료가 두 계수를 개별적으로 결정하지 못한다**는 사실뿐이다. 계수의 합 $\hat\beta_1 + \hat\beta_2$는 OLS에서 $0.4$, 능형에서 $0.83$으로 그나마 안정적이다. 다중공선성이 있을 때는 개별 계수가 아니라 안정적인 조합을 해석해야 한다([다중공선성과 정칙화](../motivation/multicollinearity.md) 연습문제 3 참조).

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
능형회귀가 편향되어 있음을 보이고, 편향을 $\lambda$와 참 $\boldsymbol{\beta}$로 유도하라.

</div>

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

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
능형회귀가 다중공선성에 도움이 되는 이유를 설명하라. $\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}$의 조건수는 어떻게 되는가?

</div>

??? success "풀이"
    다중공선성은 $\mathbf{X}^\top\mathbf{X}$가 0에 가까운 고윳값을 가져 거의 특이행렬이 되고 조건수 $\kappa = d_1/d_p$가 커진다는 뜻이다.

    $\lambda\mathbf{I}$를 더하면 모든 고윳값이 $\lambda$만큼 이동한다. 새 조건수는

    $$
    \kappa_{\text{ridge}} = \frac{d_1 + \lambda}{d_p + \lambda} < \frac{d_1}{d_p} = \kappa_{\text{OLS}}
    $$

    이다. 최소 고윳값이 0에 가까운 $d_p$에서 $d_p + \lambda$로 올라가 수치 안정성이 크게 개선되고 $\hat{\boldsymbol{\beta}}$의 분산이 줄어든다.

    구체적인 수치는 [불량조건 계획행렬](../motivation/ill_conditioning.md) 연습문제 1에 있다. $\rho = 0.999$인 등상관 구조에서 $\lambda = 1$이 조건수를 $7993 \to 8.98$로 890배 줄인다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
4절은 상관 $0.99$인 자료에서 "$\mathbf{X}^\top\mathbf{X}$의 고윳값이 $121.03$과 $0.48$로 조건수가 $252$인데, 등고선 타원의 장축과 단축 길이의 비가 정확히 그 제곱근인 $16$배다"라고 적었다. 왜 조건수가 아니라 그 **제곱근**인지 밝히고 수를 확인하라.

1. RSS를 $\hat{\boldsymbol{\beta}}_{\text{OLS}}$ 중심의 이차형식으로 쓰고, 등고선 타원의 반축 길이를 $\mathbf{X}^\top\mathbf{X}$의 고윳값으로 표현하라.
2. 축 길이의 비가 $16$에 맞는지 계산하라.
3. $\lambda = 20$을 더한 오른쪽 그릇의 등고선은 축비가 얼마인가.
4. 축비를 정확히 $2$로 만드는 $\lambda$를 구하라.
5. 유한한 $\lambda$로 등고선을 **완전한 원**으로 만들 수 있는가. 그 답이 연습문제 4의 결론에 무엇을 덧붙이는가.

</div>

??? success "풀이"
    **1.** $\hat{\boldsymbol{\beta}} = \hat{\boldsymbol{\beta}}_{\text{OLS}}$라 쓰면 정규방정식 $\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}) = \mathbf{0}$ 때문에 교차항이 사라져

    $$
    \lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert^2
    = \lVert \mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}} \rVert^2
      + (\boldsymbol{\beta} - \hat{\boldsymbol{\beta}})^\top \mathbf{X}^\top\mathbf{X} (\boldsymbol{\beta} - \hat{\boldsymbol{\beta}})
    $$

    가 된다. 따라서 등고선 $\{\boldsymbol{\beta} : \text{RSS}(\boldsymbol{\beta}) = \text{RSS}_{\min} + c\}$는

    $$
    (\boldsymbol{\beta} - \hat{\boldsymbol{\beta}})^\top \mathbf{X}^\top\mathbf{X} (\boldsymbol{\beta} - \hat{\boldsymbol{\beta}}) = c
    $$

    이다. $\mathbf{X}^\top\mathbf{X} = \sum_j \mu_j \mathbf{v}_j\mathbf{v}_j^\top$로 고유분해하고 $w_j = \mathbf{v}_j^\top(\boldsymbol{\beta} - \hat{\boldsymbol{\beta}})$로 좌표를 바꾸면 $\sum_j \mu_j w_j^2 = c$, 곧 반축 길이가 $\sqrt{c/\mu_j}$인 타원이다. **길이가 고윳값의 역수의 제곱근에 비례하므로** 축 길이의 비는

    $$
    \frac{\sqrt{c/\mu_2}}{\sqrt{c/\mu_1}} = \sqrt{\frac{\mu_1}{\mu_2}} = \sqrt{\kappa}
    $$

    가 된다. 고윳값은 곡률이고 축 길이는 그 역수의 제곱근이라는 것이 전부다.

    **2.** $\mu_1/\mu_2 = 121.03/0.48 = 252.146$이고 $\sqrt{252.146} = 15.879$다. 4절의 "$16$배"가 맞다.

    **3.** 벌점을 더한 목적함수도 $\boldsymbol{\beta}$의 이차함수이고, 이차항의 행렬이 $\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}$다. 고윳값이 $141.03$과 $20.48$이므로

    $$
    \sqrt{\frac{141.03}{20.48}} = \sqrt{6.8862} = 2.624
    $$

    배다. 장단축 비가 $15.9 \to 2.6$으로 떨어졌다. 등고선이 "골짜기"에서 "그릇"으로 보이는 것이 이 수다.

    **4.** $\dfrac{121.03 + \lambda}{0.48 + \lambda} = 2^2 = 4$를 풀면 $121.03 + \lambda = 1.92 + 4\lambda$, 곧

    $$
    \lambda = \frac{121.03 - 4 \times 0.48}{3} = \frac{119.11}{3} = 39.70
    $$

    이다. 축비를 $15.9$에서 $2.6$으로 줄이는 데는 $\lambda = 20$이면 되지만 거기서 다시 $2$로 줄이려면 $\lambda$를 두 배로 늘려야 한다. **벌점을 키워 얻는 조건수 개선은 수확이 체감한다.**

    **5.** 불가능하다. 축비가 $1$이려면 $121.03 + \lambda = 0.48 + \lambda$여야 하는데 이것은 어떤 $\lambda$에서도 성립하지 않는다. 축비 $\sqrt{(\mu_1+\lambda)/(\mu_2+\lambda)}$는 $\lambda$에 대해 **엄격히 감소하면서 $1$로 수렴할 뿐** $1$에 도달하지 못한다.

    이것이 연습문제 4에 덧붙이는 것은 다음이다. 연습문제 4는 $\kappa_{\text{ridge}} < \kappa_{\text{OLS}}$를 보였지만, 같은 계산은 $\kappa_{\text{ridge}}$가 $\lambda$에 대해 **단조감소**임도 말해 준다. 그러므로 "조건수를 가장 작게"가 목표라면 답은 $\lambda = \infty$이고 그때의 해는 $\hat{\boldsymbol{\beta}} = \mathbf{0}$이다. **조건수는 $\lambda$를 고르는 기준이 될 수 없다.** $\lambda$를 고르는 일에는 7절의 교차검증처럼 예측오차를 보는 기준이 필요하다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
9절의 유효자유도를 4절의 자료에 적용하라. $\mathbf{X}^\top\mathbf{X}$의 고윳값이 $d_1^2 = 121.03$, $d_2^2 = 0.48$이다.

1. $\lambda = 20$에서 방향별 축소인자 $d_j^2/(d_j^2+\lambda)$와 $\text{df}(20)$을 구하라.
2. 4절의 능형 해는 $(0.878,\ 0.908)$로 **두 계수가 모두 0이 아니다.** 그런데 $\text{df}(20) < 1$이다. 모순이 아닌 까닭을 설명하라.
3. $\text{df}(\lambda) = 1$이 되는 $\lambda$를 구하고, 일반적인 $p = 2$에서 그 값이 $d_1 d_2$임을 보여라.
4. 라쏘에서는 0이 아닌 계수의 개수가 자유도의 불편추정이 된다. 그 셈법을 능형에 그대로 쓰면 $\lambda = 20$에서 몇이 되고 참값과 몇 배 차이가 나는가.

</div>

??? success "풀이"
    **1.** 축소인자는

    $$
    \frac{121.03}{121.03 + 20} = \frac{121.03}{141.03} = 0.8582, \qquad
    \frac{0.48}{0.48 + 20} = \frac{0.48}{20.48} = 0.0234
    $$

    이고 유효자유도는 $\text{df}(20) = 0.8582 + 0.0234 = 0.8816$이다.

    **2.** 모순이 아니다. $\text{df}(\lambda)$는 **0이 아닌 계수의 개수가 아니라 방향마다의 축소인자를 더한 값**이다. 큰 고윳값 방향(설명변수의 합 방향)은 $85.8\%$가 살아남지만, 작은 고윳값 방향(차 방향, 곧 평평한 골짜기 방향)은 $2.3\%$만 남는다. 둘째 방향이 사실상 지워졌으므로 "자유롭게 움직이는 방향"은 하나에도 못 미친다.

    좌표 $(\beta_1, \beta_2)$가 둘 다 0이 아닌 것과는 아무 상관이 없다. 두 계수는 거의 $\beta_1 \approx \beta_2$라는 **한 방향** 위에서 함께 결정되고 있고, 실제로 $0.878$과 $0.908$이 거의 같다는 사실이 그 증거다. 자유도는 좌표를 세지 않고 방향을 센다.

    **3.** $a = d_1^2$, $b = d_2^2$라 쓰고 $\dfrac{a}{a+\lambda} + \dfrac{b}{b+\lambda} = 1$을 풀면

    $$
    a(b+\lambda) + b(a+\lambda) = (a+\lambda)(b+\lambda)
    \;\Longrightarrow\;
    2ab + (a+b)\lambda = ab + (a+b)\lambda + \lambda^2
    $$

    에서 $\lambda^2 = ab$, 곧

    $$
    \lambda = \sqrt{ab} = d_1 d_2 = \sqrt{\det(\mathbf{X}^\top\mathbf{X})}
    $$

    이다. 수를 넣으면 $\lambda = \sqrt{121.03 \times 0.48} = \sqrt{58.094} = 7.622$다. 확인하면 $121.03/128.652 + 0.48/8.102 = 0.9408 + 0.0592 = 1.0000$이다. **$p=2$에서 자유도를 정확히 하나 남기는 $\lambda$는 두 특잇값의 기하평균이다.**

    **4.** 0이 아닌 계수를 세면 $2$인데 참값은 $0.8816$이므로 $2.27$배 과대평가다. 능형에서 그 셈법이 틀리는 까닭은 능형이 계수를 **버리지 않고 줄이기만** 하기 때문이다. 줄어든 계수는 자유도를 온전히 하나씩 쓰지 않는다. 라쏘에서는 0이 된 계수가 실제로 모형에서 빠지므로 개수 셈이 뜻을 가진다. 정보기준(AIC·BIC)에 벌점모형을 넣을 때 이 차이를 지키지 않으면 능형의 복잡도를 $2$배 넘게 부풀려 모형을 잘못 고른다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span>
10절의 요약표는 능형의 변수선택을 "없음(축소하되 0으로 만들지 않음)"이라 적고 4절은 "**일반적으로** 어떤 것도 정확히 0으로 만들지 않는다"라고 적었다. 그 "일반적으로"가 꼭 필요한 말인지 가려라. 중심화·척도조정된 자료의 충분통계량이

$$
\mathbf{X}^\top\mathbf{X} = \begin{pmatrix} 1 & 0.8 \\ 0.8 & 1 \end{pmatrix}, \qquad
\mathbf{X}^\top\mathbf{y} = \begin{pmatrix} 5 \\ 2 \end{pmatrix}
$$

라 하자.

1. $\hat{\boldsymbol{\beta}}_{\text{ridge}}(\lambda)$를 $\lambda$의 함수로 닫힌 꼴로 구하라.
2. $\lambda = 0,\ 0.5,\ 1,\ 2,\ 10,\ 100$에서 두 계수를 표로 적어라.
3. $\hat{\beta}_2(\lambda) = 0$이 되는 $\lambda > 0$을 정확히 구하고, 같은 자료에서 OLS의 $\hat{\beta}_2$는 0이 아님을 확인하라.
4. 그럼에도 "능형은 변수선택을 하지 않는다"는 말이 여전히 옳은 까닭을 라쏘의 연성문턱 $\operatorname{sign}(\hat\beta_j)(\lvert\hat\beta_j\rvert - \lambda)_+$와 견주어 설명하라.
5. $\lvert\hat\beta_2(\lambda)\rvert$가 $\lambda$에 대해 단조가 **아님**을 보이고 최댓값과 그때의 $\lambda$를 구하라.

</div>

??? success "풀이"
    **1.** $\mathbf{A}_\lambda = \mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I} = \begin{pmatrix} 1+\lambda & 0.8 \\ 0.8 & 1+\lambda \end{pmatrix}$의 행렬식은 $D(\lambda) = (1+\lambda)^2 - 0.64$이고

    $$
    \mathbf{A}_\lambda^{-1} = \frac{1}{D(\lambda)}\begin{pmatrix} 1+\lambda & -0.8 \\ -0.8 & 1+\lambda \end{pmatrix}
    $$

    이다. 따라서

    $$
    \hat\beta_1(\lambda) = \frac{5(1+\lambda) - 0.8 \times 2}{D(\lambda)} = \frac{5\lambda + 3.4}{(1+\lambda)^2 - 0.64},
    \qquad
    \hat\beta_2(\lambda) = \frac{2(1+\lambda) - 0.8 \times 5}{D(\lambda)} = \frac{2\lambda - 2}{(1+\lambda)^2 - 0.64}
    $$

    이다.

    **2.**

    | $\lambda$ | $D(\lambda)$ | $\hat\beta_1(\lambda)$ | $\hat\beta_2(\lambda)$ | $\lVert\hat{\boldsymbol{\beta}}(\lambda)\rVert$ |
    |---:|---:|---:|---:|---:|
    | $0$ | $0.36$ | $9.4444$ | $-5.5556$ | $10.9573$ |
    | $0.5$ | $1.61$ | $3.6646$ | $-0.6211$ | $3.7169$ |
    | $1$ | $3.36$ | $2.5000$ | $\mathbf{0}$ | $2.5000$ |
    | $2$ | $8.36$ | $1.6029$ | $0.2392$ | $1.6206$ |
    | $10$ | $120.36$ | $0.4437$ | $0.1496$ | $0.4682$ |
    | $100$ | $10200.36$ | $0.0494$ | $0.0194$ | $0.0530$ |

    **3.** 분자 $2\lambda - 2$가 0이 되는 곳이므로 $\lambda = 1$이다. 그때

    $$
    \hat{\boldsymbol{\beta}}_{\text{ridge}}(1) = \frac{1}{3.36}\begin{pmatrix} 2 \times 5 - 0.8 \times 2 \\ -0.8 \times 5 + 2 \times 2 \end{pmatrix}
    = \frac{1}{3.36}\begin{pmatrix} 8.4 \\ 0 \end{pmatrix}
    = \begin{pmatrix} 2.5 \\ 0 \end{pmatrix}
    $$

    로 둘째 계수가 **정확히** 0이다. 한편 OLS는 $\det(\mathbf{X}^\top\mathbf{X}) = 1 - 0.64 = 0.36$에서

    $$
    \hat{\boldsymbol{\beta}}_{\text{OLS}} = \frac{1}{0.36}\begin{pmatrix} 5 - 1.6 \\ 2 - 4 \end{pmatrix}
    = \begin{pmatrix} 9.4444 \\ -5.5556 \end{pmatrix}
    $$

    이고 $\hat\beta_2^{\text{OLS}} = -5.5556 \neq 0$이다. 곧 **능형이 양의 $\lambda$에서 계수를 정확히 0으로 만드는 일은 일어날 수 있다.** 4절의 "일반적으로"는 꼭 필요한 말이다. 일이 벌어지는 조건은 $\hat\beta_2$의 분자가 0인 것, 곧 $\lambda = \rho c_1/c_2 - 1$이 양수인 것이다($c = \mathbf{X}^\top\mathbf{y}$). 여기서는 $0.8 \times 5/2 - 1 = 1$이었다.

    **4.** 그래도 "변수선택을 하지 않는다"는 말은 옳다. 차이는 **0이 되는 $\lambda$들의 집합의 크기**다.

    - 능형: $\hat\beta_2(\lambda) = 0$인 $\lambda$는 $\lambda = 1$ **한 점**뿐이다. $\lambda = 0.99$에서는 $-0.0060$, $\lambda = 1.01$에서는 $+0.0059$다. 경로가 0을 **지나갈** 뿐이다.
    - 라쏘: 직교설계에서 해는 $\operatorname{sign}(\hat\beta_j)(\lvert\hat\beta_j\rvert-\lambda)_+$이므로 $\lambda \geq \lvert\hat\beta_j\rvert$인 **구간 전체**에서 0이다. 경로가 0에 **머문다.**

    실무적으로도 둘은 다르다. 교차검증으로 고른 $\hat\lambda$가 하필 그 한 점을 집을 확률은 0이고, 설령 집었더라도 $\lambda$를 조금 흔들면 계수가 **부호를 바꾸어 다시 들어온다.** 그러니 이 0은 "변수를 뺀다"는 결정이 아니다. 라쏘의 0은 구간 안에서 흔들어도 0으로 남으므로 결정이다.

    **5.** $\hat\beta_2$를 미분한다. $u = 2\lambda - 2$, $v = (1+\lambda)^2 - 0.64$라 하면

    $$
    u'v - uv' = 2\bigl[(1+\lambda)^2 - 0.64\bigr] - (2\lambda-2) \cdot 2(1+\lambda)
    = -2\lambda^2 + 4\lambda + 4.72
    $$

    이다. $\lambda^2 - 2\lambda - 2.36 = 0$에서 $\lambda = 1 + \sqrt{3.36} = 2.8330$이고(음의 근은 버린다), 그 왼쪽에서 도함수가 양, 오른쪽에서 음이므로 최댓값이다. 값은

    $$
    \hat\beta_2(2.8330) = \frac{3.6661}{14.0521} = 0.2609
    $$

    이고 그때 $\hat\beta_1 = 1.25$다(분자와 분모가 정확히 $1{:}0.8$로 맞아떨어진다). 정리하면 $\hat\beta_2$는 $\lambda$가 커질 때 $-5.5556 \to 0 \to +0.2609 \to 0^+$으로 **한 번 올라갔다 내려온다.** $\lvert\hat\beta_2\rvert$도 $5.5556 \to 0 \to 0.2609 \to 0$으로 단조가 아니다.

    연습문제 2의 경고가 여기서 수로 확인된다. 다중공선성이 있으면 **개별 계수의 부호조차 $\lambda$의 함수**이고, 어느 한 $\lambda$에서 본 부호를 "진짜 부호"로 읽을 근거가 없다. 반면 두 계수의 합 $\hat\beta_1 + \hat\beta_2$는 $3.889 \to 3.043 \to 2.500 \to 1.842 \to 0.593 \to 0.069$로 깔끔하게 줄어든다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
연습문제 7에서 개별 계수는 $\lambda$에 대해 단조가 아니었다. 그렇다면 4절이 벌점꼴과 제약꼴 $\min_{\boldsymbol\beta} \lVert\mathbf{y}-\mathbf{X}\boldsymbol\beta\rVert^2$ s.t. $\lVert\boldsymbol\beta\rVert^2 \leq t$를 "동등하게" 묶는 근거는 무엇인가.

1. $0 \leq \lambda_1 < \lambda_2$이면 $\lVert\hat{\boldsymbol\beta}(\lambda_2)\rVert \leq \lVert\hat{\boldsymbol\beta}(\lambda_1)\rVert$임을 증명하라. (두 $\lambda$의 최적성 부등식 두 개를 더하는 것으로 끝난다.)
2. 같은 두 부등식에서 RSS가 $\lambda$에 대해 비감소임을 보여라.
3. 연습문제 7의 표로 1을 확인하라.
4. 1이 제약꼴과의 동등성에 어떻게 쓰이는지 적어라.
5. 1의 부등호는 **등호가 될 수 있는가.** 될 수 있다면 정확히 어떤 자료에서인가.

</div>

??? success "풀이"
    $R(\boldsymbol\beta) = \lVert\mathbf{y}-\mathbf{X}\boldsymbol\beta\rVert^2$, $N(\boldsymbol\beta) = \lVert\boldsymbol\beta\rVert^2$라 줄여 쓰고 $\hat{\boldsymbol\beta}_k = \hat{\boldsymbol\beta}(\lambda_k)$라 하자.

    **1.** $\hat{\boldsymbol\beta}_1$이 $\lambda_1$ 문제의 최소점이고 $\hat{\boldsymbol\beta}_2$가 $\lambda_2$ 문제의 최소점이므로

    $$
    R(\hat{\boldsymbol\beta}_1) + \lambda_1 N(\hat{\boldsymbol\beta}_1) \leq R(\hat{\boldsymbol\beta}_2) + \lambda_1 N(\hat{\boldsymbol\beta}_2), \qquad
    R(\hat{\boldsymbol\beta}_2) + \lambda_2 N(\hat{\boldsymbol\beta}_2) \leq R(\hat{\boldsymbol\beta}_1) + \lambda_2 N(\hat{\boldsymbol\beta}_1)
    $$

    이다. 두 식을 더하면 $R$ 항이 모두 지워지고

    $$
    \lambda_1 N(\hat{\boldsymbol\beta}_1) + \lambda_2 N(\hat{\boldsymbol\beta}_2) \leq \lambda_1 N(\hat{\boldsymbol\beta}_2) + \lambda_2 N(\hat{\boldsymbol\beta}_1)
    \;\Longrightarrow\;
    (\lambda_2 - \lambda_1)\bigl[N(\hat{\boldsymbol\beta}_2) - N(\hat{\boldsymbol\beta}_1)\bigr] \leq 0
    $$

    이 남는다. $\lambda_2 > \lambda_1$이므로 $N(\hat{\boldsymbol\beta}_2) \leq N(\hat{\boldsymbol\beta}_1)$이다. $\square$

    **2.** 이제 첫째 부등식에서 $\lambda_1 N(\hat{\boldsymbol\beta}_2) \leq \lambda_1 N(\hat{\boldsymbol\beta}_1)$을 쓰면

    $$
    R(\hat{\boldsymbol\beta}_1) + \lambda_1 N(\hat{\boldsymbol\beta}_1) \leq R(\hat{\boldsymbol\beta}_2) + \lambda_1 N(\hat{\boldsymbol\beta}_2) \leq R(\hat{\boldsymbol\beta}_2) + \lambda_1 N(\hat{\boldsymbol\beta}_1)
    $$

    에서 $R(\hat{\boldsymbol\beta}_1) \leq R(\hat{\boldsymbol\beta}_2)$다. **노름은 줄고 잔차제곱합은 는다** — 편향을 사서 분산을 파는 거래가 목적함수의 두 항에 그대로 나타난다. $\square$

    **3.** 연습문제 7 표의 마지막 열이 $10.9573 \to 3.7169 \to 2.5000 \to 1.6206 \to 0.4682 \to 0.0530$으로 단조감소한다. 개별 계수 $\hat\beta_2$는 올라갔다 내려왔지만 **벡터의 길이는 한 번도 늘지 않았다.** 벌점이 조이는 것은 좌표가 아니라 길이다.

    **4.** 1과 2는 $\lambda \mapsto t(\lambda) = \lVert\hat{\boldsymbol\beta}(\lambda)\rVert^2$가 연속이고 감소한다는 것을 준다. $t(0) = \lVert\hat{\boldsymbol\beta}_{\text{OLS}}\rVert^2$이고 $\lambda \to \infty$에서 $t(\lambda) \to 0$이므로, $0 < t \leq \lVert\hat{\boldsymbol\beta}_{\text{OLS}}\rVert^2$인 모든 $t$에 대해 $t(\lambda) = t$인 $\lambda$가 존재한다. 그 $\lambda$의 능형 해는 제약 $N(\boldsymbol\beta) \leq t$를 등호로 만족하고 KKT 조건의 승수가 바로 $\lambda$이므로 제약꼴의 해다. 역방향도 같은 대응으로 간다.

    이 단조성이 없으면 "$\lambda$ 하나 ↔ $t$ 하나"의 대응이 깨져 4절의 "동등하게"가 성립하지 않는다. $t > \lVert\hat{\boldsymbol\beta}_{\text{OLS}}\rVert^2$이면 제약이 작동하지 않아 해는 OLS이고 대응되는 $\lambda$는 $0$이다.

    **5.** $\lambda > 0$이면 등호는 $\mathbf{X}^\top\mathbf{y} = \mathbf{0}$일 때만 가능하다. 실제로 $N(\hat{\boldsymbol\beta}_1) = N(\hat{\boldsymbol\beta}_2)$라 하면 1의 두 부등식에서 벌점항이 서로 지워져 $R(\hat{\boldsymbol\beta}_1) = R(\hat{\boldsymbol\beta}_2)$이고, 따라서 $\hat{\boldsymbol\beta}_2$도 $\lambda_1$ 문제의 최소점이 된다. $\lambda_1 > 0$에서 목적함수가 강볼록하여 최소점이 유일하므로 $\hat{\boldsymbol\beta}_1 = \hat{\boldsymbol\beta}_2 =: \hat{\boldsymbol\beta}$다. 두 정규방정식

    $$
    (\mathbf{X}^\top\mathbf{X} + \lambda_1\mathbf{I})\hat{\boldsymbol\beta} = \mathbf{X}^\top\mathbf{y} = (\mathbf{X}^\top\mathbf{X} + \lambda_2\mathbf{I})\hat{\boldsymbol\beta}
    $$

    을 빼면 $(\lambda_2-\lambda_1)\hat{\boldsymbol\beta} = \mathbf{0}$, 곧 $\hat{\boldsymbol\beta} = \mathbf{0}$이고 $\mathbf{X}^\top\mathbf{y} = \mathbf{0}$이다. 반대로 $\mathbf{X}^\top\mathbf{y} = \mathbf{0}$이면 모든 $\lambda$에서 해가 $\mathbf{0}$이라 등호다.

    곧 **설명변수가 반응변수와 하나도 상관되어 있지 않은 자료를 빼면 능형 해의 길이는 $\lambda$에 대해 강감소다.** $\lambda_1 = 0$에서 같은 논법을 쓰려면 $\mathbf{X}^\top\mathbf{X}$가 정칙이어야 한다는 조건이 하나 더 붙는다($p > n$에서는 $\lambda = 0$의 해가 유일하지 않다).

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
7절은 LOOCV 지름길 공식을 증명 없이 썼다. 작은 자료에서 손으로 확인하고 일반적으로 증명하라. 절편 없는 $p = 1$ 모형 $y_i = \beta x_i + \varepsilon_i$에 $\mathbf{x} = (1, 2, 3)^\top$, $\mathbf{y} = (2, 2, 5)^\top$, $\lambda = 1$을 쓴다.

1. $\hat\beta(1)$, 적합값, 잔차, $h_{ii}(1)$을 구하라. $\sum_i h_{ii}$가 9절의 $\text{df}(1)$과 같은지 확인하라.
2. 지름길 공식으로 $\text{CV}_{\text{LOO}}(1)$을 계산하라.
3. 자료를 하나씩 빼고 세 번 다시 적합하여 예측오차를 직접 구하고 2와 비교하라.
4. 일반적으로 $\hat y_i^{(-i)} = \dfrac{\hat y_i - h_{ii}y_i}{1 - h_{ii}}$임을 증명하여 공식을 유도하라.
5. 보정 없이 $\frac{1}{n}\sum_i r_i^2$을 쓰면 얼마이고, 왜 그것이 예측오차의 과소평가인가.

</div>

??? success "풀이"
    **1.** $\mathbf{x}^\top\mathbf{x} = 1 + 4 + 9 = 14$, $\mathbf{x}^\top\mathbf{y} = 2 + 4 + 15 = 21$이므로

    $$
    \hat\beta(1) = \frac{21}{14 + 1} = \frac{21}{15} = 1.4
    $$

    이고 $\hat{\mathbf{y}} = (1.4,\ 2.8,\ 4.2)^\top$, 잔차 $\mathbf{r} = (0.6,\ -0.8,\ 0.8)^\top$이다. 레버리지는 $h_{ii} = x_i^2/(\mathbf{x}^\top\mathbf{x}+\lambda) = x_i^2/15$이므로

    $$
    h_{11} = \tfrac{1}{15} = 0.0667, \qquad h_{22} = \tfrac{4}{15} = 0.2667, \qquad h_{33} = \tfrac{9}{15} = 0.6000
    $$

    이고 합은 $14/15 = 0.9333$이다. 한편 9절의 공식은 $p=1$, $d_1^2 = 14$에서 $\text{df}(1) = 14/(14+1) = 14/15$를 준다. **같다.** $\text{df}(\lambda) = \text{tr}\,\mathbf{H}(\lambda) = \sum_i h_{ii}(\lambda)$이므로 당연하지만, 유효자유도가 "레버리지의 합"이라는 또 하나의 읽기를 준다.

    **2.** 보정계수 $1/(1-h_{ii})$는 $15/14$, $15/11$, $15/6$이다.

    $$
    \text{CV}_{\text{LOO}}(1) = \frac{1}{3}\left[\left(\frac{9}{14}\right)^2 + \left(-\frac{12}{11}\right)^2 + 2^2\right]
    = \frac{1}{3}\bigl[0.41327 + 1.19008 + 4\bigr] = \frac{5.60335}{3} = 1.86778
    $$

    **3.** 세 번 다시 적합한다. 벌점은 그대로 $\lambda = 1$이다.

    | 뺀 관측 | 남은 자료의 $\mathbf{x}^\top\mathbf{x}$ | $\mathbf{x}^\top\mathbf{y}$ | $\hat\beta^{(-i)}$ | 예측 $x_i\hat\beta^{(-i)}$ | 오차 $y_i - \hat y_i^{(-i)}$ |
    |---:|---:|---:|---:|---:|---:|
    | $i=1$ | $4+9=13$ | $4+15=19$ | $19/14 = 1.35714$ | $1.35714$ | $0.64286 = 9/14$ |
    | $i=2$ | $1+9=10$ | $2+15=17$ | $17/11 = 1.54545$ | $3.09091$ | $-1.09091 = -12/11$ |
    | $i=3$ | $1+4=5$ | $2+4=6$ | $6/6 = 1.00000$ | $3.00000$ | $2.00000$ |

    세 오차가 2에서 쓴 $r_i/(1-h_{ii})$와 분수까지 **똑같다.** 평균제곱은 역시 $1.86778$이다.

    **4.** $\mathbf{A}_\lambda = \mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I}$, $\mathbf{H} = \mathbf{X}\mathbf{A}_\lambda^{-1}\mathbf{X}^\top$라 하자. 핵심은 **능형이 선형 평활자**라는 것이다. $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$이고 $\mathbf{H}$는 $\mathbf{y}$에 의존하지 않는다.

    $i$를 고정하고 $\hat{\boldsymbol\beta}^{(-i)}$를 $\sum_{k\neq i}(y_k - \mathbf{x}_k^\top\boldsymbol\beta)^2 + \lambda\lVert\boldsymbol\beta\rVert^2$의 최소점, $z = \mathbf{x}_i^\top\hat{\boldsymbol\beta}^{(-i)} = \hat y_i^{(-i)}$라 쓴다. 이제 $y_i$만 $z$로 바꾼 자료 $\mathbf{y}^\ast$를 생각하자. 그 전체자료 목적함수는

    $$
    \sum_{k\neq i}(y_k - \mathbf{x}_k^\top\boldsymbol\beta)^2 + (z - \mathbf{x}_i^\top\boldsymbol\beta)^2 + \lambda\lVert\boldsymbol\beta\rVert^2
    $$

    이고, $\boldsymbol\beta = \hat{\boldsymbol\beta}^{(-i)}$에서 추가된 항의 기울기 $-2(z - \mathbf{x}_i^\top\hat{\boldsymbol\beta}^{(-i)})\mathbf{x}_i$가 $z$의 정의에 의해 $\mathbf{0}$이다. 따라서 전체 기울기가 제외자료 목적함수의 기울기와 같고 그것은 $\mathbf{0}$이다. $\lambda > 0$에서 목적함수가 강볼록하므로

    $$
    \hat{\boldsymbol\beta}(\mathbf{y}^\ast) = \hat{\boldsymbol\beta}^{(-i)}
    $$

    이다. **$i$번째 관측을 자기 예측값으로 바꿔 놓으면 그 관측은 적합에 아무 영향을 주지 않는다.** 이제 $\mathbf{H}$의 선형성을 쓴다. $\mathbf{y}^\ast = \mathbf{y} + (z - y_i)\mathbf{e}_i$이므로

    $$
    z = \mathbf{x}_i^\top\hat{\boldsymbol\beta}(\mathbf{y}^\ast) = (\mathbf{H}\mathbf{y}^\ast)_i = \hat y_i + h_{ii}(z - y_i)
    $$

    이다. $z$에 대해 풀면 $z(1 - h_{ii}) = \hat y_i - h_{ii}y_i$, 곧

    $$
    \hat y_i^{(-i)} = z = \frac{\hat y_i - h_{ii}y_i}{1 - h_{ii}}, \qquad
    y_i - \hat y_i^{(-i)} = \frac{y_i - \hat y_i}{1 - h_{ii}}
    $$

    이고 제곱하여 평균하면 7절의 공식이다($\lambda>0$이면 $h_{ii} < 1$이라 분모가 0이 아니다). $\square$

    증명이 쓴 성질은 둘뿐이다 — 적합값이 $\mathbf{y}$의 선형함수이고, 목적함수가 관측별 항의 합이라는 것. 라쏘는 앞의 것을 만족하지 않으므로 같은 지름길이 없다.

    **5.** 보정 없는 값은

    $$
    \frac{1}{3}(0.6^2 + 0.8^2 + 0.8^2) = \frac{1.64}{3} = 0.54667
    $$

    로 LOOCV $1.86778$의 $29\%$다. 같은 관측을 적합에도 평가에도 썼기 때문이다. 각 관측은 자기 적합값을 자기 쪽으로 끌어당기고, 그 끌어당기는 세기가 바로 $h_{ii}$다. 세 번째 관측은 $h_{33} = 0.6$으로 영향력이 가장 크고, 보정계수가 $2.5$배, 제곱하면 $6.25$배다. 잔차 $0.8$이 실제 예측오차 $2.0$으로 되살아난다. **레버리지가 큰 점에서 훈련잔차가 가장 심하게 거짓말을 한다.**

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
6절은 MSE를 줄이는 $\lambda$가 존재한다고만 말하고, 8절은 $\lambda = \sigma^2/\tau^2$라고 말한다. 두 $\lambda$가 서로 무슨 관계인지 직교설계에서 끝까지 계산하라. $\mathbf{X}^\top\mathbf{X} = \mathbf{I}_p$(곧 $d_j = 1$), $\boldsymbol\varepsilon \sim N(\mathbf{0}, \sigma^2\mathbf{I}_n)$, $S = \lVert\boldsymbol\beta\rVert^2$으로 둔다.

1. 능형 해가 $\hat{\boldsymbol\beta}_{\text{OLS}}/(1+\lambda)$임을 보이고 $g(\lambda) = E\lVert\hat{\boldsymbol\beta}_{\text{ridge}} - \boldsymbol\beta\rVert^2$을 $\lambda$의 함수로 적어라.
2. $g$를 최소화하는 $\lambda^\ast$, 그 최솟값 $g(\lambda^\ast)$, OLS에 대한 상대이득 $1 - g(\lambda^\ast)/g(0)$을 구하라.
3. 8절의 사전분포 $\boldsymbol\beta \sim N(\mathbf{0}, \tau^2\mathbf{I})$ 아래에서 $E[S]$를 구하고, 그것을 $\lambda^\ast$에 꽂아 넣으면 정확히 $\sigma^2/\tau^2$이 됨을 보여라.
4. $\sigma^2 = 1$, $p = 10$에서 $\tau^2 = 0.1,\ 1,\ 10$의 $\lambda$, $E[S]$, 상대이득을 표로 적어라.
5. 사전분포가 틀렸을 때의 손해를 재라. $\tau^2 = 10$으로 믿어 $\lambda = 0.1$을 썼는데 참값이 $S = 1$이었다면 MSE는 얼마인가. $\lambda^\ast$를 썼을 때와 OLS와 비교하라.

</div>

??? success "풀이"
    **1.** $\mathbf{X}^\top\mathbf{X} = \mathbf{I}_p$이면 $\mathbf{A}_\lambda = (1+\lambda)\mathbf{I}_p$이므로

    $$
    \hat{\boldsymbol\beta}_{\text{ridge}} = \frac{\mathbf{X}^\top\mathbf{y}}{1+\lambda} = \frac{\hat{\boldsymbol\beta}_{\text{OLS}}}{1+\lambda}
    $$

    로 **모든 좌표가 같은 비율로** 축소된다. $\hat{\boldsymbol\beta}_{\text{OLS}} = \boldsymbol\beta + \mathbf{X}^\top\boldsymbol\varepsilon \sim N(\boldsymbol\beta, \sigma^2\mathbf{I}_p)$이므로 편향은 $-\lambda\boldsymbol\beta/(1+\lambda)$, 좌표별 분산은 $\sigma^2/(1+\lambda)^2$이고

    $$
    g(\lambda) = \frac{\lambda^2 S}{(1+\lambda)^2} + \frac{p\sigma^2}{(1+\lambda)^2} = \frac{\lambda^2 S + p\sigma^2}{(1+\lambda)^2}
    $$

    이다. 앞항이 편향제곱, 뒤항이 분산이다.

    **2.** $g'(\lambda) = 0$을 풀면

    $$
    2\lambda S(1+\lambda)^2 - 2(1+\lambda)(\lambda^2 S + p\sigma^2) = 0
    \;\Longrightarrow\;
    \lambda S(1+\lambda) = \lambda^2 S + p\sigma^2
    \;\Longrightarrow\;
    \lambda S = p\sigma^2
    $$

    에서

    $$
    \lambda^\ast = \frac{p\sigma^2}{S}
    $$

    이다. 꽂아 넣으면 $1 + \lambda^\ast = (S + p\sigma^2)/S$이고

    $$
    g(\lambda^\ast) = \frac{p^2\sigma^4/S + p\sigma^2}{(S+p\sigma^2)^2/S^2}
    = \frac{p\sigma^2(p\sigma^2+S)}{S} \cdot \frac{S^2}{(S+p\sigma^2)^2}
    = \frac{p\sigma^2 S}{S + p\sigma^2}
    $$

    이다. $g(0) = p\sigma^2$이 OLS의 MSE이므로

    $$
    1 - \frac{g(\lambda^\ast)}{g(0)} = 1 - \frac{S}{S+p\sigma^2} = \frac{p\sigma^2}{S + p\sigma^2}
    $$

    가 상대이득이다. 이것이 $0$보다 크다는 것이 직교설계에서의 Hoerl-Kennard 정리다(존재성의 증명 자체는 [과적합과 편향-분산 절충](../motivation/overfitting.md) 연습문제 4에 있다). 두 조화평균 꼴 $g(\lambda^\ast) = (1/S + 1/p\sigma^2)^{-1}$이 눈에 잘 들어온다 — **최적 능형의 MSE는 신호 $S$와 잡음 $p\sigma^2$ 중 작은 쪽에 묶인다.**

    **3.** $\beta_j \sim N(0, \tau^2)$이 독립이므로 $E[S] = \sum_j E[\beta_j^2] = p\tau^2$이다. $\lambda^\ast$에 넣으면

    $$
    \lambda^\ast\big|_{S = E[S]} = \frac{p\sigma^2}{p\tau^2} = \frac{\sigma^2}{\tau^2}
    $$

    로 8절의 $\lambda$와 **정확히 일치한다.** 곧 MAP의 $\lambda = \sigma^2/\tau^2$는 MSE 최적 $\lambda^\ast = p\sigma^2/S$에서 알 수 없는 $S$를 사전평균 $p\tau^2$로 바꿔 넣은 꽂아넣기 값이다. 두 절이 서로 다른 말을 하는 것이 아니라, 8절이 6절의 최적값을 **사전분포에 대해 평균한** 꼴로 주고 있다.

    **4.** $p\sigma^2 = 10$이다.

    | $\tau^2$ | $\lambda = \sigma^2/\tau^2$ | $E[S] = p\tau^2$ | $g(\lambda^\ast)$ | $g(0)$ | 상대이득 |
    |---:|---:|---:|---:|---:|---:|
    | $0.1$ | $10$ | $1$ | $0.9091$ | $10$ | $0.909$ |
    | $1$ | $1$ | $10$ | $5.0000$ | $10$ | $0.500$ |
    | $10$ | $0.1$ | $100$ | $9.0909$ | $10$ | $0.091$ |

    **신호가 잡음에 비해 약할 때($E[S] \ll p\sigma^2$) 정칙화가 거의 모든 것을 벌고, 신호가 강할 때는 거의 아무것도 벌지 못한다.** 1절이 든 세 상황 — 다중공선성, $p$가 $n$에 가까움, 잡음이 큼 — 은 모두 $p\sigma^2$를 $S$에 비해 키우는 조건이다.

    **5.** $S = 1$, $p\sigma^2 = 10$인데 $\lambda = 0.1$을 썼다면

    $$
    g(0.1) = \frac{0.01 \times 1 + 10}{1.1^2} = \frac{10.01}{1.21} = 8.2727
    $$

    이다. 같은 자료의 최적값은 $\lambda^\ast = 10/1 = 10$이고 $g(10) = (100 + 10)/121 = 0.9091$, OLS는 $g(0) = 10$이다. 쓸 수 있었던 이득 $10 - 0.9091 = 9.0909$ 가운데 실제로 거둔 것은 $10 - 8.2727 = 1.7273$, 곧 $19\%$뿐이다. $\lambda$가 $100$배 작았던 대가로 MSE가 $9.1$배 커졌고 OLS보다 $17\%$ 나은 데서 멈췄다.

    배울 것은 두 가지다. 첫째, $\lambda^\ast = p\sigma^2/S$는 **알 수 없는 $S$에 의존**하므로 계산할 수 없다. 베이즈 경로는 그 미지를 $S$에서 $\tau^2$으로 옮겼을 뿐이고, 그래서 결국 7절의 교차검증처럼 자료에서 $\lambda$를 추정해야 한다. 둘째, $\lambda^\ast$가 $S$에 **반비례**하므로 참 계수가 클수록 $\lambda$를 작게 잡아야 한다 — 연습문제 3에서 편향이 $-\boldsymbol\beta$ 방향이고 $\boldsymbol\beta$에 비례해 커진다고 한 것과 같은 말이다.

---

## 정리하며

능형회귀는 **$L_2$ 벌점**으로 계수를 축소한다.

$$
\hat{\boldsymbol\beta}_{\text{ridge}}=(\mathbf X^\top\mathbf X+\lambda\mathbf I)^{-1}\mathbf X^\top\mathbf y
$$

- **대각선에 $\lambda$ 를 더하는 것이 전부다.** $\mathbf X^\top\mathbf X$ 가 특이하거나 거의 특이해도 역행렬이 존재하게 되며, **$p>n$ 에서도 해가 유일하다.**
- **13장의 RSS 곡면 그림이 설명해 준다.** 공선성으로 골짜기가 평평해졌을 때 벌점이 그릇 모양을 되살린다.
- **편향을 들여 분산을 줄인다.** 계수가 $0$ 쪽으로 당겨져 편향이 생기지만 분산이 더 크게 줄어 MSE 가 내려간다. 1장의 편향–분산 절충 그대로다.
- **어느 것도 정확히 $0$ 이 되지 않는다.** 변수선택은 하지 못하며, 그것이 라쏘와의 결정적 차이다.
- **표준화가 필수다.** 벌점이 척도에 의존하므로 설명변수를 표준화하지 않으면 단위가 큰 변수만 덜 벌점을 받는다. **절편은 벌점에서 제외한다.**

다음 절 **능형회귀 보기**로 넘어간다.
