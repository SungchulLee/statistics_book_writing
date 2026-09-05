# 양정치행렬

대칭행렬이 0이 아닌 모든 벡터 $\mathbf{x}$에 대해 이차형식 $\mathbf{x}^T\mathbf{A}\mathbf{x}$을 엄격히 양수로 만들 때 양정치라고 한다. 이 조건은 양의 실수에 대응하는 행렬판이며, $\mathbf{A}$가 가역이고 유일한 촐레스키 분해를 가지며 진짜 내적을 정의함을 보장한다. 통계에서 양정치성은 잘 정의된 공분산행렬(가역이며 유한한 밀도의 다변량 정규분포로 이어진다)과 퇴화한 것을 가르는 기준이다. 이 절에서는 정의, 동치인 특성화들, 그리고 촐레스키 분해를 다룬다.

## 정의

!!! info "정의 — 양정치와 양반정치"
    대칭행렬 $\mathbf{A} \in \mathbb{R}^{n \times n}$이

    - 모든 $\mathbf{x} \neq \mathbf{0}$에 대해 $\mathbf{x}^T\mathbf{A}\mathbf{x} > 0$이면 **양정치**($\mathbf{A} \succ 0$).
    - 모든 $\mathbf{x}$에 대해 $\mathbf{x}^T\mathbf{A}\mathbf{x} \geq 0$이면 **양반정치**($\mathbf{A} \succeq 0$).
    - $-\mathbf{A} \succ 0$이면 **음정치**($\mathbf{A} \prec 0$).
    - $\mathbf{x}^T\mathbf{A}\mathbf{x}$가 양수와 음수를 모두 취하면 **부정치**.

!!! warning "대칭성을 전제한다"
    대칭이 아닌 행렬에 대해서도 양정치성을 정의하는 문헌이 있지만, 이 책에서는(그리고 통계학의 거의 전부에서는) 양정치성이 언제나 대칭행렬을 가리킨다.

## 동치인 특성화

대칭행렬 $\mathbf{A} \in \mathbb{R}^{n \times n}$에 대해 다음 조건들은 서로 동치다.

!!! tip "정리 — 양정치성의 동치 조건"
    다음은 서로 동치다.

    1. $\mathbf{A} \succ 0$ (이차형식 조건).
    2. $\mathbf{A}$의 모든 고윳값이 엄격히 양수다: $i = 1, \dots, n$에 대해 $\lambda_i > 0$.
    3. 모든 **선행 주소행렬식**이 양수다: $k = 1, \dots, n$에 대해 $\det(\mathbf{A}_k) > 0$이며, 여기서 $\mathbf{A}_k$는 왼쪽 위 $k \times k$ 부분행렬이다.
    4. $\mathbf{A}$가 **촐레스키 분해**를 갖는다: 대각 성분이 양수인 유일한 하삼각행렬 $\mathbf{L}$에 대해 $\mathbf{A} = \mathbf{L}\mathbf{L}^T$.
    5. $\mathbf{A} = \mathbf{B}^T\mathbf{B}$인 가역행렬 $\mathbf{B}$가 존재한다.

### 증명 개요 (고윳값에 의한 특성화)

스펙트럼 정리에 의해 직교행렬 $\mathbf{Q}$에 대해 $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$이다. $\mathbf{z} = \mathbf{Q}^T\mathbf{x}$로 두면($\mathbf{Q}$가 직교행렬이므로 전단사다)

$$
\mathbf{x}^T\mathbf{A}\mathbf{x} = \mathbf{z}^T\boldsymbol{\Lambda}\mathbf{z} = \sum_{i=1}^n \lambda_i z_i^2
$$

이다. 이것이 모든 $\mathbf{z} \neq \mathbf{0}$에 대해 양수일 필요충분조건은 모든 $\lambda_i > 0$인 것이다. $\square$

**양반정치성**의 경우 위 조건들에서 고윳값과 선행 소행렬식에 대한 조건이 "$\geq 0$"으로 완화되고, 조건 (5)에서는 $\mathbf{B}$가 계수 부족이어도 된다.

## 촐레스키 분해

!!! info "정의 — 촐레스키 분해"
    양정치행렬 $\mathbf{A}$의 **촐레스키 분해**는 유일한 인수분해

    $$
    \mathbf{A} = \mathbf{L}\mathbf{L}^T
    $$

    이며, 여기서 $\mathbf{L}$은 대각 성분이 엄격히 양수인 하삼각행렬이다.

촐레스키 분해는 양수의 제곱근을 취하는 것에 대응하는 행렬판이다. 수치적으로 안정적이고 대략 $n^3/3$번의 연산이 필요하며(일반적인 $\mathbf{LU}$ 분해의 절반 비용), 양정치행렬이 관여하는 연립방정식을 푸는 데 선호되는 방법이다.

### 예

행렬

$$
\mathbf{A} = \begin{pmatrix} 4 & 2 \\ 2 & 5 \end{pmatrix}
$$

에 대해,

**고윳값 확인:** 고윳값은 $\lambda^2 - 9\lambda + 16 = 0$을 만족하므로 $\lambda = (9 \pm \sqrt{17})/2$이다. 둘 다 양수이므로(대략 6.56과 2.44) $\mathbf{A} \succ 0$이다.

**선행 소행렬식:** $\det(\mathbf{A}_1) = 4 > 0$이고 $\det(\mathbf{A}_2) = 16 > 0$이다. 둘 다 양수이므로 양정치성이 확인된다.

**촐레스키 분해:** $\mathbf{L}$에 대해 푼다.

$$
\begin{pmatrix} 4 & 2 \\ 2 & 5 \end{pmatrix} = \begin{pmatrix} l_{11} & 0 \\ l_{21} & l_{22} \end{pmatrix}\begin{pmatrix} l_{11} & l_{21} \\ 0 & l_{22} \end{pmatrix}
$$

$l_{11}^2 = 4$에서 $l_{11} = 2$, $l_{21}l_{11} = 2$에서 $l_{21} = 1$, $l_{21}^2 + l_{22}^2 = 5$에서 $l_{22} = 2$이다. 따라서

$$
\mathbf{L} = \begin{pmatrix} 2 & 0 \\ 1 & 2 \end{pmatrix}
$$

이다.

## 성질

### 양정치행렬은 가역이다

$\mathbf{A} \succ 0$이면 모든 고윳값이 양수이므로 $\det(\mathbf{A}) = \prod_i \lambda_i > 0$이다. 따라서 $\mathbf{A}$는 가역이고 $\mathbf{A}^{-1}$도 양정치다(그 고윳값이 $1/\lambda_i > 0$이므로).

### 합과 스칼라배

$\mathbf{A} \succ 0$이고 $\mathbf{B} \succ 0$이면 $\mathbf{A} + \mathbf{B} \succ 0$이고, 임의의 $c > 0$에 대해 $c\mathbf{A} \succ 0$이다. 양정치행렬 전체의 집합은 열린 볼록뿔을 이룬다.

### 합동변환은 양정치성을 보존한다

$\mathbf{A} \succ 0$이고 $\mathbf{B}$가 $\operatorname{rank}(\mathbf{B}) = m$인 $n \times m$ 행렬이면 $\mathbf{B}^T\mathbf{A}\mathbf{B} \succ 0$이다($\mathbb{R}^{m \times m}$에서).

**증명.** $\mathbb{R}^m$의 임의의 $\mathbf{y} \neq \mathbf{0}$에 대해 $\mathbf{x} = \mathbf{B}\mathbf{y}$로 두자. $\mathbf{B}$가 완전 열계수를 가지므로 $\mathbf{x} \neq \mathbf{0}$이고, 따라서 $\mathbf{y}^T(\mathbf{B}^T\mathbf{A}\mathbf{B})\mathbf{y} = \mathbf{x}^T\mathbf{A}\mathbf{x} > 0$이다. $\square$

이 결과가 $\mathbf{X}$가 완전 열계수를 가질 때 $\mathbf{X}^T\mathbf{X}$가 양정치인 이유를 설명한다. $\mathbf{I}_n \succ 0$을 $\mathbf{X}$로 합동변환한 것이기 때문이다.

### 슈어 여인수

대칭 양정치행렬을

$$
\mathbf{M} = \begin{pmatrix} \mathbf{A} & \mathbf{B} \\ \mathbf{B}^T & \mathbf{C} \end{pmatrix} \succ 0
$$

으로 분할하면 **슈어 여인수** $\mathbf{S} = \mathbf{C} - \mathbf{B}^T\mathbf{A}^{-1}\mathbf{B}$도 양정치다. 슈어 여인수는 다변량 정규분포의 조건부 분산에 등장한다.

## 통계와의 연결

### 공분산행렬

공분산행렬 $\boldsymbol{\Sigma} = E[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^T]$는 언제나 양반정치다. 양정치일 필요충분조건은 $\mathbf{X}$의 어떤 성분도 나머지 성분들의 정확한 선형결합이 아닌 것이다. $\boldsymbol{\Sigma} \succ 0$이면 다변량 정규밀도가 잘 정의된다.

$$
f(\mathbf{x}) = \frac{1}{(2\pi)^{p/2}|\boldsymbol{\Sigma}|^{1/2}}\exp\!\Bigl(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})\Bigr)
$$

$\boldsymbol{\Sigma}$의 양정치성이 $|\boldsymbol{\Sigma}| > 0$을 보장하여(밀도가 유한하다) 지수부가 언제나 음수가 되게 한다(밀도가 모든 방향으로 감쇠한다).

### 최소제곱해의 존재

최소제곱추정량 $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$은 $\mathbf{X}^T\mathbf{X}$가 가역이어야 한다. $\mathbf{X}^T\mathbf{X}$는 언제나 양반정치이므로, 가역(양정치)이 되는 것은 정확히 $\mathbf{X}$가 완전 열계수를 가질 때다.

### 마할라노비스 거리

마할라노비스 거리 $d^2 = (\mathbf{x} - \boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})$은 $\boldsymbol{\Sigma}^{-1}$에 대한 이차형식이고, $\boldsymbol{\Sigma}$가 양정치이면 $\boldsymbol{\Sigma}^{-1}$도 양정치다. 이 덕분에 $d^2 \geq 0$이고 등호는 $\mathbf{x} = \boldsymbol{\mu}$에서만 성립하여, 진짜 거리에 준하는 측도가 된다.

## 요약

양정치성은 이차형식이 엄격히 양수임을 보장하는 행렬의 성질이며, 이는 가역성, 잘 정의된 밀도, 유일한 최소제곱해로 이어진다. 핵심적인 동치 특성화들 — 양의 고윳값, 양의 선행 소행렬식, 촐레스키 분해의 존재 — 은 서로 다른 계산적·이론적 도구를 제공한다. 통계에서 $\boldsymbol{\Sigma}$의 양정치성이 다변량 정규분포를 떠받치고, $\mathbf{X}^T\mathbf{X}$의 양정치성이 최소제곱추정량의 존재를 보장한다.

## 연습문제

**연습문제 1.**
선행 소행렬식 기준을 이용해 $\mathbf{A} = \begin{pmatrix} 4 & 2 \\ 2 & 3 \end{pmatrix}$이 양정치인지 판정하라.

??? success "연습문제 1 풀이"
    선행 소행렬식은 다음과 같다.

    - 첫 번째 선행 소행렬식: $a_{11} = 4 > 0$
    - 두 번째 선행 소행렬식: $\det(\mathbf{A}) = 4 \times 3 - 2 \times 2 = 12 - 4 = 8 > 0$

    모든 선행 소행렬식이 엄격히 양수이므로 $\mathbf{A}$는 양정치다. 동등하게 고윳값은 $\lambda = \frac{7 \pm \sqrt{49 - 32}}{2} = \frac{7 \pm \sqrt{17}}{2}$이며 둘 다 양수다.

---

**연습문제 2.**
$\mathbf{A}$가 양정치이면 모든 대각 성분 $a_{ii} > 0$임을 증명하라.

??? success "연습문제 2 풀이"
    $\mathbf{e}_i$를 $i$번째 표준기저벡터라 하자($i$번째 자리가 $1$이고 나머지는 $0$). $\mathbf{e}_i \neq \mathbf{0}$이고 $\mathbf{A}$가 양정치이므로

    $$
    \mathbf{e}_i^T \mathbf{A} \mathbf{e}_i > 0
    $$

    이다. 그런데 $\mathbf{e}_i^T \mathbf{A} \mathbf{e}_i = a_{ii}$이므로 모든 $i$에 대해 $a_{ii} > 0$이다. 역은 성립하지 않음에 유의하라. 대각 성분이 양수라고 양정치성이 보장되지는 않는다(예를 들어 $\begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix}$은 대각 성분이 양수지만 고윳값이 $3$과 $-1$이다). $\square$

---

**연습문제 3.**
$\mathbf{A} = \begin{pmatrix} 4 & 6 \\ 6 & 13 \end{pmatrix}$의 촐레스키 분해 $\mathbf{A} = \mathbf{L}\mathbf{L}^T$를 구하라.

??? success "연습문제 3 풀이"
    $\mathbf{L}\mathbf{L}^T = \mathbf{A}$가 되는 하삼각행렬 $\mathbf{L} = \begin{pmatrix} l_{11} & 0 \\ l_{21} & l_{22} \end{pmatrix}$을 찾는다.

    $l_{11}^2 = 4$에서 $l_{11} = 2$.

    $l_{21} l_{11} = 6$에서 $l_{21} = 3$.

    $l_{21}^2 + l_{22}^2 = 13$에서 $9 + l_{22}^2 = 13$이므로 $l_{22} = 2$.

    $$
    \mathbf{L} = \begin{pmatrix} 2 & 0 \\ 3 & 2 \end{pmatrix}
    $$

    확인: $\mathbf{L}\mathbf{L}^T = \begin{pmatrix} 4 & 6 \\ 6 & 13 \end{pmatrix} = \mathbf{A}$.

---

**연습문제 4.**
어떤 공분산행렬 $\boldsymbol{\Sigma}$의 고윳값이 $\lambda_1 = 0.01$과 $\lambda_2 = 100$이다. $\boldsymbol{\Sigma}$는 양정치인가? $\boldsymbol{\Sigma}^{-1}$을 계산할 때의 실무적 함의를 논하라.

??? success "연습문제 4 풀이"
    그렇다. 두 고윳값이 모두 엄격히 양수이므로 $\boldsymbol{\Sigma}$는 양정치다. 그러나 조건수가 $\kappa = \lambda_{\max}/\lambda_{\min} = 100/0.01 = 10{,}000$으로 매우 크다.

    실무적 함의는 다음과 같다.

    - **수치적 불안정:** $\boldsymbol{\Sigma}^{-1}$을 계산할 때의 부동소수 오차가 조건수만큼 증폭된다. $\boldsymbol{\Sigma}^{-1}$의 고윳값 $1/\lambda_1 = 100$이 상당히 손상될 수 있다.
    - **거의 특이:** 가장 작은 고윳값 방향으로 자료가 거의 공선적이며, 이는 두 변수가 거의 완벽하게 상관되어 있다는 뜻이다.
    - **대응책:** 명시적인 역행렬 대신 촐레스키 분해를 쓰거나, 정칙화(능형회귀)를 적용하거나, 적절한 경우 유사역행렬을 쓴다.

---

**연습문제 5.**
$\boldsymbol{\Sigma}$가 양정치이면 마할라노비스 거리 $d^2(\mathbf{x}) = (\mathbf{x} - \boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})$이 0일 필요충분조건이 $\mathbf{x} = \boldsymbol{\mu}$임을 증명하라.

??? success "연습문제 5 풀이"
    $\boldsymbol{\Sigma}$가 양정치이므로 $\boldsymbol{\Sigma}^{-1}$도 양정치다(그 고윳값이 $1/\lambda_i > 0$이다).

    $\mathbf{z} = \mathbf{x} - \boldsymbol{\mu}$로 두면 $d^2 = \mathbf{z}^T\boldsymbol{\Sigma}^{-1}\mathbf{z}$이다.

    $\boldsymbol{\Sigma}^{-1}$의 양정치성에 의해 모든 $\mathbf{z}$에 대해 $\mathbf{z}^T\boldsymbol{\Sigma}^{-1}\mathbf{z} \geq 0$이고, 등호는 $\mathbf{z} = \mathbf{0}$일 때에 한해 성립한다.

    따라서 $d^2 = 0$일 필요충분조건은 $\mathbf{x} - \boldsymbol{\mu} = \mathbf{0}$, 즉 $\mathbf{x} = \boldsymbol{\mu}$이다. 이는 마할라노비스 거리가 (정치성을 만족하는) 제대로 된 거리에 준하는 측도임을 확인해 준다. $\square$
