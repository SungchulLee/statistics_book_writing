# 그람 행렬

벡터들의 모임을 행렬 $\mathbf{X}$의 열로 배열하면, 곱 $\mathbf{X}^T\mathbf{X}$는 모든 쌍의 내적을 하나의 행렬에 모아 담는다. 이 **그람 행렬(Gram matrix)** 은 언제나 대칭이고 양반정치이며, 열벡터들의 기하(각도와 길이)를 부호화한다. 회귀에서 그람 행렬 $\mathbf{X}^T\mathbf{X}$는 정규방정식에 등장하고, 그 가역성이 최소제곱해의 유일성을 결정하며, 그 고윳값이 추정량의 수치적 안정성을 좌우한다. 그람 행렬을 이해하면 회귀해의 존재에 대한 대수적 조건과 예측변수들의 일차독립성이라는 기하적 개념이 연결된다.

## 정의

!!! info "정의 — 그람 행렬"
    $\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_p \in \mathbb{R}^n$을 벡터들의 모임이라 하고, 이 벡터들을 열로 갖는 행렬을 $\mathbf{X} = (\mathbf{v}_1 \mid \mathbf{v}_2 \mid \cdots \mid \mathbf{v}_p) \in \mathbb{R}^{n \times p}$이라 하자. **그람 행렬**은

    $$
    \mathbf{G} = \mathbf{X}^T\mathbf{X} \in \mathbb{R}^{p \times p}
    $$

    이며, 그 $(i,j)$ 성분은 내적 $[\mathbf{G}]_{ij} = \mathbf{v}_i^T\mathbf{v}_j$이다.

대각 성분 $g_{ii} = \mathbf{v}_i^T\mathbf{v}_i = \lVert\mathbf{v}_i\rVert^2$는 벡터 길이의 제곱이고, 비대각 성분 $g_{ij} = \mathbf{v}_i^T\mathbf{v}_j$는 두 벡터가 얼마나 같은 방향을 향하는지를 잰다.

## 대칭성과 양반정치성

!!! tip "정리 — 그람 행렬은 대칭 양반정치다"
    임의의 행렬 $\mathbf{X} \in \mathbb{R}^{n \times p}$에 대해 그람 행렬 $\mathbf{G} = \mathbf{X}^T\mathbf{X}$는

    1. **대칭이다**: $\mathbf{G}^T = (\mathbf{X}^T\mathbf{X})^T = \mathbf{X}^T\mathbf{X} = \mathbf{G}$
    2. **양반정치다**: 모든 $\mathbf{y} \in \mathbb{R}^p$에 대해 $\mathbf{y}^T\mathbf{G}\mathbf{y} \geq 0$

**양반정치성의 증명.** 임의의 $\mathbf{y} \in \mathbb{R}^p$에 대해

$$
\mathbf{y}^T\mathbf{G}\mathbf{y} = \mathbf{y}^T\mathbf{X}^T\mathbf{X}\mathbf{y} = (\mathbf{X}\mathbf{y})^T(\mathbf{X}\mathbf{y}) = \lVert\mathbf{X}\mathbf{y}\rVert^2 \geq 0
$$

이다. 이차형식이 $\mathbf{X}\mathbf{y}$의 유클리드 노름의 제곱과 같으므로 언제나 음이 아니다. $\square$

## 그람 행렬이 언제 양정치인가

!!! tip "정리 — 그람 행렬의 양정치성"
    그람 행렬 $\mathbf{G} = \mathbf{X}^T\mathbf{X}$가 양정치일 필요충분조건은 $\mathbf{X}$가 완전 열계수를 갖는 것이다(즉 $\operatorname{rank}(\mathbf{X}) = p$).

**증명.** 위 계산에서 $\mathbf{y}^T\mathbf{G}\mathbf{y} = \lVert\mathbf{X}\mathbf{y}\rVert^2$이다. 이것이 모든 $\mathbf{y} \neq \mathbf{0}$에 대해 엄격히 양수일 필요충분조건은 모든 $\mathbf{y} \neq \mathbf{0}$에 대해 $\mathbf{X}\mathbf{y} \neq \mathbf{0}$인 것이고, 이는 $\ker(\mathbf{X}) = \{\mathbf{0}\}$, 즉 $\mathbf{X}$가 완전 열계수를 갖는 것과 동치다. $\square$

**회귀에 대한 귀결.** 최소제곱추정량 $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$이 존재하고 유일할 필요충분조건은 $\mathbf{X}^T\mathbf{X}$가 양정치인 것이며, 이는 예측변수 열들이 일차독립일 때에 한해 성립한다.

## 그람 행렬의 고윳값

$\mathbf{G} = \mathbf{X}^T\mathbf{X}$가 대칭 양반정치이므로 그 고윳값 $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_p \geq 0$은 모두 음이 아니다. 이 고윳값들은 $\mathbf{X}$의 **특이값**의 제곱이다. $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$가 특이값분해(SVD)라면

$$
\mathbf{X}^T\mathbf{X} = \mathbf{V}\boldsymbol{\Sigma}^T\boldsymbol{\Sigma}\mathbf{V}^T = \mathbf{V}\operatorname{diag}(\sigma_1^2, \dots, \sigma_p^2)\mathbf{V}^T
$$

이므로 $\lambda_i = \sigma_i^2$이다.

### 조건수

$\mathbf{X}^T\mathbf{X}$의 **조건수**는

$$
\kappa(\mathbf{X}^T\mathbf{X}) = \frac{\lambda_{\max}}{\lambda_{\min}} = \frac{\sigma_{\max}^2}{\sigma_{\min}^2}
$$

이다. 조건수가 크면 $\mathbf{X}^T\mathbf{X}$가 거의 특이($\mathbf{X}$의 열들이 거의 일차종속)라는 뜻이고, 최소제곱해가 수치적으로 불안정해진다. 이런 상황을 **다중공선성**이라 한다.

## 예

계획행렬이 (간단히 하기 위해 절편을 무시하고) 두 개의 예측변수 열을 갖는다고 하자.

$$
\mathbf{X} = \begin{pmatrix} 1 & 2 \\ 3 & 1 \\ 2 & 4 \end{pmatrix}
$$

그람 행렬은

$$
\mathbf{X}^T\mathbf{X} = \begin{pmatrix} 1 & 3 & 2 \\ 2 & 1 & 4 \end{pmatrix}\begin{pmatrix} 1 & 2 \\ 3 & 1 \\ 2 & 4 \end{pmatrix} = \begin{pmatrix} 14 & 13 \\ 13 & 21 \end{pmatrix}
$$

이다.

**대칭성 확인:** 이 행렬은 분명히 대칭이다($g_{12} = g_{21} = 13$).

**양정치성 확인:** $\det(\mathbf{X}^T\mathbf{X}) = 14 \cdot 21 - 13^2 = 294 - 169 = 125 > 0$이고 $g_{11} = 14 > 0$이므로 $\mathbf{X}^T\mathbf{X} \succ 0$이다. 이는 $\mathbf{X}$의 두 열이 일차독립임을 확인해 준다.

**해석:** 성분 $g_{11} = 14 = \lVert\mathbf{v}_1\rVert^2$은 첫 번째 예측변수의 제곱합이다. 성분 $g_{12} = 13 = \mathbf{v}_1^T\mathbf{v}_2$는 두 예측변수가 얼마나 같은 방향을 향하는지를 잰다. 예측변수들이 직교했다면 이 성분이 0이 되고 그람 행렬은 대각행렬이 되었을 것이다.

## 두 개의 그람 행렬

임의의 $\mathbf{X} \in \mathbb{R}^{n \times p}$에 대해 사실 그람 행렬은 두 개다.

| 행렬 | 크기 | 성분 | 회귀에서의 역할 |
|---|---|---|---|
| $\mathbf{X}^T\mathbf{X}$ | $p \times p$ | 열(예측변수) 벡터들의 내적 | 정규방정식 |
| $\mathbf{X}\mathbf{X}^T$ | $n \times n$ | 행(관측) 벡터들의 내적 | 모자 행렬, 커널 방법 |

둘은 0이 아닌 고윳값을 공유한다($\mathbf{A}^T\mathbf{A}$와 $\mathbf{A}\mathbf{A}^T$의 일반적 성질). $n \gg p$일 때는 $p \times p$ 그람 행렬로 작업하는 편이 훨씬 효율적이고, $p \gg n$인 고차원 상황에서는 $n \times n$ 쪽이 선호된다(이것이 "커널 트릭"이다).

## 그람 행렬과 직교성

그람 행렬은 열벡터들의 직교 구조를 부호화한다.

- $\mathbf{X}$의 열들이 **직교**일 필요충분조건은 $\mathbf{X}^T\mathbf{X}$가 대각행렬인 것이다.
- $\mathbf{X}$의 열들이 **정규직교**일 필요충분조건은 $\mathbf{X}^T\mathbf{X} = \mathbf{I}$인 것이다.

계획행렬의 열들이 직교하면 정규방정식이 분리되고 최소제곱추정량이 각 예측변수 $j$에 대해 독립적으로 $\hat{\beta}_j = \mathbf{v}_j^T\mathbf{y}/\lVert\mathbf{v}_j\rVert^2$로 간단해진다. 예측변수를 (그람–슈미트나 QR 분해로) 직교화하면 회귀의 계산과 해석이 단순해지는 이유가 이것이다.

## 통계와의 연결

### 정규방정식

최소제곱의 정규방정식은 $\mathbf{X}^T\mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}^T\mathbf{y}$이다. 그람 행렬 $\mathbf{X}^T\mathbf{X}$가 이 연립방정식의 계수행렬이다. ($\mathbf{X}$가 완전 열계수를 가질 때의) 양정치성이 유일한 해를 보장한다.

### 최소제곱추정량의 분산

$\operatorname{Var}(\boldsymbol{\varepsilon}) = \sigma^2\mathbf{I}$인 표준 선형모형 $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$ 아래에서

$$
\operatorname{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2(\mathbf{X}^T\mathbf{X})^{-1}
$$

이다. $(\mathbf{X}^T\mathbf{X})^{-1}$의 고윳값은 $1/\lambda_i$이므로, 그람 행렬의 고윳값이 작으면 추정량의 분산이 커진다. 이것이 다중공선성이 표준오차를 부풀리는 이유에 대한 대수적 설명이다.

### 표본 공분산행렬

$\mathbf{X}$가 중심화된 자료행렬(관측값에서 열 평균을 뺀 것)일 때 표본 공분산행렬은 $\mathbf{S} = \frac{1}{n-1}\mathbf{X}^T\mathbf{X}$로, 크기가 조정된 그람 행렬이다.

## 요약

그람 행렬 $\mathbf{X}^T\mathbf{X}$는 $\mathbf{X}$의 열들 사이의 모든 쌍의 내적을 대칭 양반정치행렬에 모은다. 열들이 일차독립일 때에 한해 양정치가 되며, 이것이 최소제곱해가 유일하기 위한 조건이다. 그람 행렬의 고윳값은 조건수를 통해 회귀의 수치적 안정성을 좌우하고, 그 비대각 구조가 예측변수들 사이의 상관을 잰다.

## 연습문제

**연습문제 1.**
$\mathbf{X} = \begin{pmatrix} 1 & 2 \\ 1 & 3 \\ 1 & 5 \end{pmatrix}$이라 하자. 그람 행렬 $\mathbf{X}^T\mathbf{X}$를 계산하고 그것이 대칭이며 양정치임을 확인하라.

??? success "풀이"
    $$
    \mathbf{X}^T\mathbf{X} = \begin{pmatrix} 1 & 1 & 1 \\ 2 & 3 & 5 \end{pmatrix}\begin{pmatrix} 1 & 2 \\ 1 & 3 \\ 1 & 5 \end{pmatrix} = \begin{pmatrix} 3 & 10 \\ 10 & 38 \end{pmatrix}
    $$

    대칭성은 $(\mathbf{X}^T\mathbf{X})^T = \mathbf{X}^T\mathbf{X}$로부터 곧바로 따라온다. 양정치성의 경우 선행 소행렬식이 $3 > 0$이고 $\det = 3 \times 38 - 10^2 = 114 - 100 = 14 > 0$이므로 $\mathbf{X}^T\mathbf{X}$는 양정치다. 동등하게 $\mathbf{X}$의 계수가 2이므로(두 열이 일차독립이므로) $\mathbf{X}^T\mathbf{X}$가 양정치다.

---

**연습문제 2.**
임의의 실행렬 $\mathbf{X}$에 대해 $\mathbf{X}^T\mathbf{X}$가 언제나 양반정치이고, 양정치일 필요충분조건이 $\mathbf{X}$가 완전 열계수를 갖는 것임을 증명하라.

??? success "풀이"
    임의의 벡터 $\mathbf{v} \neq \mathbf{0}$에 대해

    $$
    \mathbf{v}^T(\mathbf{X}^T\mathbf{X})\mathbf{v} = (\mathbf{X}\mathbf{v})^T(\mathbf{X}\mathbf{v}) = \lVert \mathbf{X}\mathbf{v} \rVert^2 \geq 0
    $$

    이다. 이것이 0일 필요충분조건은 $\mathbf{X}\mathbf{v} = \mathbf{0}$, 즉 $\mathbf{v} \in \ker(\mathbf{X})$인 것이다. $\mathbf{X}$가 완전 열계수를 가지면 $\ker(\mathbf{X}) = \{\mathbf{0}\}$이므로 모든 $\mathbf{v} \neq \mathbf{0}$에 대해 $\mathbf{v}^T(\mathbf{X}^T\mathbf{X})\mathbf{v} > 0$이고 이것이 양정치성이다. 역으로 $\mathbf{X}$가 완전 열계수를 갖지 않으면 $\mathbf{X}\mathbf{v} = \mathbf{0}$인 $\mathbf{v} \neq \mathbf{0}$이 존재하여 이차형식이 0이 된다. $\square$

---

**연습문제 3.**
$\mathbf{X}$의 열들이 거의 공선적일 때 $\mathbf{X}^T\mathbf{X}$의 조건수가 최소제곱추정값의 안정성과 어떻게 관련되는지 설명하라. 조건수를 고윳값으로 나타내면 무엇인가?

??? success "풀이"
    $\mathbf{X}^T\mathbf{X}$의 조건수는 $\kappa = \lambda_{\max}/\lambda_{\min}$이며, 여기서 $\lambda_{\max}$와 $\lambda_{\min}$은 최대·최소 고윳값이다.

    열들이 거의 공선적이면 $\lambda_{\min}$이 0에 가까워져 $\kappa$가 매우 커진다. $\operatorname{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2(\mathbf{X}^T\mathbf{X})^{-1}$이고 $(\mathbf{X}^T\mathbf{X})^{-1}$의 고윳값이 $1/\lambda_i$이므로, $\lambda_{\min}$이 작으면 그에 대응하는 고유벡터 방향으로 분산 $\sigma^2/\lambda_{\min}$이 커진다. 조건수가 크다는 것은 또한 $\mathbf{y}$의 작은 섭동이 $\hat{\boldsymbol{\beta}}$을 크게 변화시킨다는 뜻이며, 추정값이 수치적으로 불안정해진다.

---

**연습문제 4.**
표본 공분산행렬 $\mathbf{S} = \frac{1}{n-1}\mathbf{X}_c^T\mathbf{X}_c$($\mathbf{X}_c$는 평균 중심화된 자료행렬)가 양반정치임을 보여라. 어떤 조건에서 양정치가 되는가?

??? success "풀이"
    $\mathbf{S} = \frac{1}{n-1}\mathbf{X}_c^T\mathbf{X}_c$는 그람 행렬의 양의 스칼라배이므로 양반정치성을 물려받는다.

    $$
    \mathbf{v}^T\mathbf{S}\mathbf{v} = \frac{1}{n-1}\lVert \mathbf{X}_c \mathbf{v} \rVert^2 \geq 0
    $$

    $\mathbf{S}$가 양정치일 필요충분조건은 $\mathbf{X}_c$가 완전 열계수를 갖는 것이며, 이를 위해서는 $n - 1 \geq p$가 필요하다(중심화가 계수를 많아야 1만큼 줄이기 때문이다). 실무적으로는 표본 공분산행렬이 가역이려면 변수보다 관측값이 많아야 한다는($n > p$) 뜻이다. $p > n$이면 $\mathbf{S}$가 특이행렬이 되어 정칙화나 차원축소 같은 기법이 필요하다. $\square$
