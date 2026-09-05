# 표본분포 (일반 최소제곱)

앞 절의 단순회귀 결과는 행렬대수를 이용해 일반적인 다중회귀 모형으로 확장된다. 일반적인 경우에 최소제곱추정량 $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$은 다변량 정규벡터이고, 잔차제곱합을 $\sigma^2$으로 나눈 것은 카이제곱을 따르며, 이 둘은 독립이다. 정규성, 카이제곱, 독립성이라는 이 세 사실이 결합되어 개별 계수에 대한 t-통계량과 계수 집합을 검정하는 F-통계량이 나온다. 증명은 이 장에서 앞서 전개한 사영행렬과 이차형식 이론에 기댄다.

## 일반 선형모형

행렬 형태의 모형은

$$
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}
$$

이며, 여기서

- $\mathbf{y} \in \mathbb{R}^n$은 반응벡터다.
- $\mathbf{X} \in \mathbb{R}^{n \times p}$는 $\operatorname{rank}(\mathbf{X}) = p$인(완전 열계수, $p \leq n$) 고정된 계획행렬이다.
- $\boldsymbol{\beta} \in \mathbb{R}^p$는 미지의 모수벡터다.
- $\boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2\mathbf{I}_n)$: 오차는 평균 0, 공통 분산 $\sigma^2$을 갖는 독립 동일분포 정규확률변수다.

이 가정들로부터 $\mathbf{y} \sim N(\mathbf{X}\boldsymbol{\beta}, \sigma^2\mathbf{I}_n)$이다.

## 최소제곱추정량

최소제곱추정량은 $\lVert\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\rVert^2$을 최소화한다.

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
$$

이것은 $\mathbf{y}$의 선형함수다. 행렬 $(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$가 $\mathbf{y}$를 $\hat{\boldsymbol{\beta}}$으로 보낸다.

## 최소제곱추정량의 표본분포

!!! tip "정리 — 최소제곱추정량의 분포"
    정규오차를 갖는 일반 선형모형 아래에서

    $$
    \hat{\boldsymbol{\beta}} \sim N\!\left(\boldsymbol{\beta},\; \sigma^2(\mathbf{X}^T\mathbf{X})^{-1}\right)
    $$

**증명.**

*선형성:* $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$은 $\mathbf{y}$의 선형변환이다.

*평균:*

$$
E[\hat{\boldsymbol{\beta}}] = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T E[\mathbf{y}] = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \boldsymbol{\beta}
$$

따라서 $\hat{\boldsymbol{\beta}}$은 **불편**이다.

*공분산행렬:*

$$
\operatorname{Var}(\hat{\boldsymbol{\beta}}) = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T \operatorname{Var}(\mathbf{y})\, \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}
$$

$$
= (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T (\sigma^2\mathbf{I}_n)\, \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1} = \sigma^2(\mathbf{X}^T\mathbf{X})^{-1}
$$

*정규성:* 다변량 정규벡터의 선형변환은 다변량 정규분포를 따른다. $\square$

$j$번째 계수의 분산은 $\operatorname{Var}(\hat{\beta}_j) = \sigma^2[(\mathbf{X}^T\mathbf{X})^{-1}]_{jj}$이다.

## 적합값과 잔차

적합값과 잔차는

$$
\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}, \qquad \mathbf{e} = (\mathbf{I} - \mathbf{H})\mathbf{y} = \mathbf{M}\mathbf{y}
$$

이며, 여기서 $\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$는 모자 행렬이고 $\mathbf{M} = \mathbf{I} - \mathbf{H}$는 잔차생성행렬이다.

이들의 분포는 다음과 같다.

$$
\hat{\mathbf{y}} \sim N(\mathbf{X}\boldsymbol{\beta},\; \sigma^2\mathbf{H})
$$

$$
\mathbf{e} \sim N(\mathbf{0},\; \sigma^2\mathbf{M})
$$

$\mathbf{M}\mathbf{X} = \mathbf{0}$이므로 $\mathbf{e}$의 평균은 $\boldsymbol{\beta}$와 무관하게 0이지만, 공분산행렬 $\sigma^2\mathbf{M}$은 특이행렬이므로(계수 $n - p$) 잔차들끼리는 서로 독립이 아니다.

## 잔차제곱합의 분포

!!! tip "정리 — SSE의 카이제곱분포"
    잔차제곱합

    $$
    \text{SSE} = \mathbf{e}^T\mathbf{e} = \mathbf{y}^T\mathbf{M}\mathbf{y}
    $$

    은

    $$
    \frac{\text{SSE}}{\sigma^2} \sim \chi^2_{n-p}
    $$

    를 만족한다.

**증명.** $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$으로 쓰자. $\mathbf{M}\mathbf{X} = \mathbf{0}$이므로

$$
\text{SSE} = \boldsymbol{\varepsilon}^T\mathbf{M}\boldsymbol{\varepsilon}
$$

이다. $\mathbf{z} = \boldsymbol{\varepsilon}/\sigma \sim N(\mathbf{0}, \mathbf{I}_n)$으로 두면 $\text{SSE}/\sigma^2 = \mathbf{z}^T\mathbf{M}\mathbf{z}$이다. $\mathbf{M}$이 $\operatorname{rank}(\mathbf{M}) = n - p$인 대칭 멱등행렬이므로 기본 카이제곱 정리에 의해 $\mathbf{z}^T\mathbf{M}\mathbf{z} \sim \chi^2_{n-p}$이다. $\square$

따라서 $\sigma^2$의 불편추정량은

$$
s^2 = \frac{\text{SSE}}{n - p}
$$

이다.

## 추정량과 SSE의 독립성

!!! tip "정리 — 독립성"
    $\hat{\boldsymbol{\beta}}$과 $\text{SSE}$는 독립이다.

**증명.** $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$이 $\mathbf{y}$의 선형함수이고 $\text{SSE} = \mathbf{y}^T\mathbf{M}\mathbf{y}$이 이차형식이므로, 선형 부분과 이차 부분이 서로 직교하는 사영에 관여한다는 사실에서 독립성이 따라온다. 형식적으로는

$$
\operatorname{Cov}(\hat{\boldsymbol{\beta}}, \mathbf{e}) = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\operatorname{Var}(\mathbf{y})\mathbf{M} = \sigma^2(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{M} = \mathbf{0}
$$

인데, $\mathbf{X}^T\mathbf{M} = \mathbf{X}^T(\mathbf{I} - \mathbf{H}) = \mathbf{X}^T - \mathbf{X}^T = \mathbf{0}$이기 때문이다. 결합 정규성 아래에서 공분산이 0이면 독립이다. $\square$

## 개별 계수에 대한 t-통계량

$\hat{\beta}_j$의 정규분포, $\text{SSE}/\sigma^2$의 카이제곱분포, 그리고 이 둘의 독립성을 결합하면 t-분포가 나온다.

!!! tip "정리 — 계수 검정을 위한 t-분포"
    각 $j = 1, \dots, p$에 대해

    $$
    T_j = \frac{\hat{\beta}_j - \beta_j}{\text{SE}(\hat{\beta}_j)} \sim t_{n-p}
    $$

    이며, 여기서 $\text{SE}(\hat{\beta}_j) = s\sqrt{[(\mathbf{X}^T\mathbf{X})^{-1}]_{jj}}$이다.

**증명 개요.** 표준화된 추정량 $(\hat{\beta}_j - \beta_j)/(\sigma\sqrt{[(\mathbf{X}^T\mathbf{X})^{-1}]_{jj}}) \sim N(0,1)$이다. $\sigma$를 $s = \sqrt{\text{SSE}/(n-p)}$로 바꾸면 분모에 $\sqrt{\chi^2_{n-p}/(n-p)}$가 들어간 비가 만들어진다. (위 독립성 정리에 의한) 분자와 분모의 독립성이 $t_{n-p}$ 분포를 준다. $\square$

## 여러 계수를 검정하는 F-통계량

계수의 부분집합이 동시에 0인지 검정하려면, 계수 $q$인 $q \times p$ 행렬 $\mathbf{C}$에 대해 $H_0: \mathbf{C}\boldsymbol{\beta} = \mathbf{0}$을 검정하는 것을 생각한다.

!!! tip "정리 — 선형가설에 대한 F-분포"
    $H_0: \mathbf{C}\boldsymbol{\beta} = \mathbf{0}$ 아래에서

    $$
    F = \frac{(\mathbf{C}\hat{\boldsymbol{\beta}})^T[\mathbf{C}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{C}^T]^{-1}(\mathbf{C}\hat{\boldsymbol{\beta}})/q}{s^2} \sim F_{q,\,n-p}
    $$

**증명 개요.** $H_0$ 아래에서 $\mathbf{C}\hat{\boldsymbol{\beta}} \sim N(\mathbf{0}, \sigma^2\mathbf{C}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{C}^T)$이다. 분자의 이차형식을 $\sigma^2$으로 나누면 (정규벡터 이차형식에 대한 카이제곱 정리에 의해) $\chi^2_q$이다. 이는 ($\text{SSE}/(n-p)$에 근거한) $s^2$과 독립이다. 독립인 두 카이제곱 변수를 각자의 자유도로 나눈 비는 $F_{q,\,n-p}$이다. $\square$

### 특수한 경우: 전체 F-검정

($\mathbf{X}$의 첫 열이 절편이라 가정하고) $H_0: \beta_1 = \beta_2 = \cdots = \beta_{p-1} = 0$(모든 기울기가 0)을 검정하면 전체 F-통계량을 얻는다.

$$
F = \frac{\text{SSR}/(p - 1)}{\text{SSE}/(n - p)} = \frac{\text{MSR}}{\text{MSE}}
$$

여기서 $\text{SSR} = \hat{\mathbf{y}}^T\hat{\mathbf{y}} - n\bar{Y}^2$이 회귀제곱합이다. $H_0$ 아래에서 $F \sim F_{p-1,\,n-p}$이다.

## 가우스–마르코프 정리

정규성을 가정하지 않아도 최소제곱추정량은 최적성을 갖는다.

!!! tip "정리 — 가우스–마르코프"
    가정 $E[\boldsymbol{\varepsilon}] = \mathbf{0}$과 $\operatorname{Var}(\boldsymbol{\varepsilon}) = \sigma^2\mathbf{I}_n$ 아래에서(정규성은 필요 없다) 최소제곱추정량 $\hat{\boldsymbol{\beta}}$은 **최량선형불편추정량(BLUE)** 이다. 즉 $\boldsymbol{\beta}$의 모든 선형불편추정량 가운데 최소제곱추정량이 (행렬 순서의 의미에서) 가장 작은 분산을 갖는다.

**증명 개요.** $\tilde{\boldsymbol{\beta}} = \mathbf{A}\mathbf{y}$을 임의의 선형불편추정량이라 하자. 불편성은 $\mathbf{A}\mathbf{X} = \mathbf{I}_p$를 요구한다. $\mathbf{D}\mathbf{X} = \mathbf{0}$인 $\mathbf{D}$에 대해 $\mathbf{A} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T + \mathbf{D}$로 쓰면

$$
\operatorname{Var}(\tilde{\boldsymbol{\beta}}) = \sigma^2\mathbf{A}\mathbf{A}^T = \sigma^2(\mathbf{X}^T\mathbf{X})^{-1} + \sigma^2\mathbf{D}\mathbf{D}^T
$$

이다. $\mathbf{D}\mathbf{D}^T \succeq 0$이므로 양반정치 순서에서 $\operatorname{Var}(\tilde{\boldsymbol{\beta}}) \succeq \operatorname{Var}(\hat{\boldsymbol{\beta}})$이다. $\square$

## 핵심 분포 요약

| 양 | 분포 | 자유도 |
|---|---|---|
| $\hat{\boldsymbol{\beta}}$ | $N(\boldsymbol{\beta}, \sigma^2(\mathbf{X}^T\mathbf{X})^{-1})$ | — |
| $\text{SSE}/\sigma^2$ | $\chi^2_{n-p}$ | $n - p$ |
| $T_j = (\hat{\beta}_j - \beta_j)/\text{SE}(\hat{\beta}_j)$ | $t_{n-p}$ | $n - p$ |
| $F = \text{MSR}/\text{MSE}$ ($H_0$ 아래) | $F_{q,\,n-p}$ | $q$와 $n-p$ |

이 모든 분포는 세 가지 재료에 의존한다. (1) $\hat{\boldsymbol{\beta}}$이 정규벡터 $\mathbf{y}$의 선형함수이고, (2) $\mathbf{M}$이 대칭 멱등이며, (3) $\mathbf{X}^T\mathbf{M} = \mathbf{0}$이므로 $\hat{\boldsymbol{\beta}}$과 $\text{SSE}$가 독립이라는 것이다.

## 요약

최소제곱의 행렬 정식화는 단순회귀의 표본분포 이론을 하나의 통합된 틀로 압축한다. 추정량 $\hat{\boldsymbol{\beta}}$은 공분산이 $\sigma^2(\mathbf{X}^T\mathbf{X})^{-1}$인 다변량 정규분포를 따르고, 잔차제곱합 $\text{SSE}/\sigma^2$은 자유도 $n - p$인 카이제곱을 따르며, 이 둘은 독립이다. 최소제곱의 직교사영 구조와 오차의 정규성에서 따라오는 이 세 사실이 개별 계수에 대한 모든 표준 t-검정과 계수 집합에 대한 F-검정을 만들어낸다. 나아가 가우스–마르코프 정리는 정규성 가정이 없어도 최소제곱이 선형불편추정량 가운데 최적임을 보여준다.

## 연습문제

**연습문제 1.**
관측값이 $n = 30$개, (절편을 포함해) 모수가 $p = 4$개이고 $\text{SSE} = 52$인 다중회귀 모형 $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$을 생각하자. $s^2$을 계산하고 개별 계수 검정에 쓰이는 t-통계량의 자유도를 구하라.

??? success "연습문제 1 풀이"
    불편 분산추정량은

    $$
    s^2 = \frac{\text{SSE}}{n - p} = \frac{52}{30 - 4} = \frac{52}{26} = 2.0
    $$

    이다.

    개별 계수에 대한 t-통계량은 각자의 귀무가설 아래에서 $t_{n-p} = t_{26}$ 분포를 따른다. 각 t-통계량의 자유도는 $26$이다.

---

**연습문제 2.**
$\mathbf{M} = \mathbf{I} - \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$일 때 최소제곱 잔차벡터 $\mathbf{e} = \mathbf{M}\mathbf{y}$이 $\mathbf{X}^T\mathbf{e} = \mathbf{0}$을 만족함을 증명하라.

??? success "연습문제 2 풀이"
    직접 계산한다.

    $$
    \mathbf{X}^T\mathbf{e} = \mathbf{X}^T\mathbf{M}\mathbf{y} = \mathbf{X}^T\bigl(\mathbf{I} - \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\bigr)\mathbf{y}
    $$

    $$
    = \mathbf{X}^T\mathbf{y} - \mathbf{X}^T\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y} = \mathbf{X}^T\mathbf{y} - \mathbf{X}^T\mathbf{y} = \mathbf{0}
    $$

    이는 잔차가 $\mathbf{X}$의 모든 열에 직교함을 보여주며, 이것이 정규방정식의 행렬 형태다. $\square$

---

**연습문제 3.**
예측변수가 $p = 3$개인(절편을 더해 $p = 4$인) 모형에서 일반 F-검정으로 $H_0: \beta_2 = \beta_3 = 0$을 검정하려 한다. $H_0$ 아래에서 F-통계량의 분포를 진술하고, $q$를 밝히며, 이 맥락에서 "제약된" SSE와 "제약 없는" SSE가 무엇을 뜻하는지 설명하라.

??? success "연습문제 3 풀이"
    제약이 $q = 2$개다(두 계수를 0으로 둔다). F-통계량은

    $$
    F = \frac{(\text{SSE}_R - \text{SSE}_U)/q}{\text{SSE}_U/(n-p)} \sim F_{q,\, n-p} = F_{2,\, n-4}
    $$

    이다. 여기서 $\text{SSE}_R$은 $X_2$와 $X_3$을 제외한(절편과 $X_1$만 적합한) 제약 모형의 잔차제곱합이고, $\text{SSE}_U$는 네 모수를 모두 갖는 완전(제약 없는) 모형의 잔차제곱합이다. F-통계량은 $X_2$와 $X_3$을 포함해서 줄어든 SSE가 잡음 수준 $s^2 = \text{SSE}_U/(n-4)$에 비해 충분히 큰지를 잰다.

---

**연습문제 4.**
(정규성 없이) 가우스–마르코프 가정 아래에서 $\operatorname{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2(\mathbf{X}^T\mathbf{X})^{-1}$임을 보여라. (힌트: $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$에 $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$을 대입하라.)

??? success "연습문제 4 풀이"
    $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$을 대입하면

    $$
    \hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T(\mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}) = \boldsymbol{\beta} + (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\boldsymbol{\varepsilon}
    $$

    이다. $\hat{\boldsymbol{\beta}} - \boldsymbol{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\boldsymbol{\varepsilon}$이므로 ($\mathbf{X}$를 고정된 것으로 다루면)

    $$
    \operatorname{Var}(\hat{\boldsymbol{\beta}}) = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T \operatorname{Var}(\boldsymbol{\varepsilon})\, \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}
    $$

    이다. $\operatorname{Var}(\boldsymbol{\varepsilon}) = \sigma^2\mathbf{I}$이면

    $$
    = \sigma^2 (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1} = \sigma^2(\mathbf{X}^T\mathbf{X})^{-1}
    $$

    이다. 정규성 가정은 전혀 필요하지 않았고 $E[\boldsymbol{\varepsilon}] = \mathbf{0}$과 $\operatorname{Var}(\boldsymbol{\varepsilon}) = \sigma^2\mathbf{I}$만 썼다. $\square$

---

**연습문제 5.**
오차가 정규일 때 $\hat{\boldsymbol{\beta}}$과 $\text{SSE}$가 왜 독립인지 개념적으로 설명하라. 이를 가능하게 하는 모자 행렬의 기하적 성질은 무엇인가?

??? success "연습문제 5 풀이"
    핵심적인 기하적 성질은 모자 행렬 $\mathbf{H}$와 잔차생성행렬 $\mathbf{M} = \mathbf{I} - \mathbf{H}$가 서로 직교하는 부분공간 위로 사영한다는 점이다. 구체적으로 $\hat{\boldsymbol{\beta}}$은 $\mathbf{y}$에 오직 $\mathbf{H}\mathbf{y}$($\mathbf{X}$의 열공간 위로의 사영)를 통해서만 의존하고, $\text{SSE} = \mathbf{y}^T\mathbf{M}\mathbf{y}$은 오직 $\mathbf{M}\mathbf{y}$(직교여공간 위로의 사영)를 통해서만 의존한다.

    $\mathbf{H}\mathbf{M} = \mathbf{0}$이므로 벡터 $\mathbf{H}\mathbf{y}$와 $\mathbf{M}\mathbf{y}$는 무상관이다. 정규성 가정 아래에서 무상관인 정규확률벡터는 독립이다. 이 직교 분해가 t-통계량과 F-통계량이 앞서 진술한 분포를 갖는 기하적 이유다.
