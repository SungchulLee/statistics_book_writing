# 양정치행렬

대칭행렬이 0이 아닌 모든 벡터 $\mathbf{x}$에 대해 이차형식 $\mathbf{x}^T\mathbf{A}\mathbf{x}$을 엄격히 양수로 만들 때 양정치라고 한다. 이 조건은 양의 실수에 대응하는 행렬판이며, $\mathbf{A}$가 가역이고 유일한 촐레스키 분해를 가지며 진짜 내적을 정의함을 보장한다. 통계에서 양정치성은 잘 정의된 공분산행렬(가역이며 유한한 밀도의 다변량 정규분포로 이어진다)과 퇴화한 것을 가르는 기준이다. 이 절에서는 정의, 동치인 특성화들, 그리고 촐레스키 분해를 다룬다.

## 정의

<div class="defn" markdown>

**정의 1.** [양정치와 양반정치]

대칭행렬 $\mathbf{A} \in \mathbb{R}^{n \times n}$이

- 모든 $\mathbf{x} \neq \mathbf{0}$에 대해 $\mathbf{x}^T\mathbf{A}\mathbf{x} > 0$이면 **양정치**($\mathbf{A} \succ 0$).
- 모든 $\mathbf{x}$에 대해 $\mathbf{x}^T\mathbf{A}\mathbf{x} \geq 0$이면 **양반정치**($\mathbf{A} \succeq 0$).
- $-\mathbf{A} \succ 0$이면 **음정치**($\mathbf{A} \prec 0$).
- $\mathbf{x}^T\mathbf{A}\mathbf{x}$가 양수와 음수를 모두 취하면 **부정치**.

</div>

!!! warning "대칭성을 전제한다"
    대칭이 아닌 행렬에 대해서도 양정치성을 정의하는 문헌이 있지만, 이 책에서는(그리고 통계학의 거의 전부에서는) 양정치성이 언제나 대칭행렬을 가리킨다.

## 동치인 특성화

대칭행렬 $\mathbf{A} \in \mathbb{R}^{n \times n}$에 대해 다음 조건들은 서로 동치다.

<div class="thmbox" markdown>

### 정리 1. 양정치성의 동치 조건 { .thm }

다음은 서로 동치다.

1. $\mathbf{A} \succ 0$ (이차형식 조건).
2. $\mathbf{A}$의 모든 고윳값이 엄격히 양수다: $i = 1, \dots, n$에 대해 $\lambda_i > 0$.
3. 모든 **선행 주소행렬식**이 양수다: $k = 1, \dots, n$에 대해 $\det(\mathbf{A}_k) > 0$이며, 여기서 $\mathbf{A}_k$는 왼쪽 위 $k \times k$ 부분행렬이다.
4. $\mathbf{A}$가 **촐레스키 분해**를 갖는다: 대각 성분이 양수인 유일한 하삼각행렬 $\mathbf{L}$에 대해 $\mathbf{A} = \mathbf{L}\mathbf{L}^T$.
5. $\mathbf{A} = \mathbf{B}^T\mathbf{B}$인 가역행렬 $\mathbf{B}$가 존재한다.

</div>

??? proof "증명 개요 (고윳값에 의한 특성화)"


    스펙트럼 정리에 의해 직교행렬 $\mathbf{Q}$에 대해 $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$이다. $\mathbf{z} = \mathbf{Q}^T\mathbf{x}$로 두면($\mathbf{Q}$가 직교행렬이므로 전단사다)

    $$
    \mathbf{x}^T\mathbf{A}\mathbf{x} = \mathbf{z}^T\boldsymbol{\Lambda}\mathbf{z} = \sum_{i=1}^n \lambda_i z_i^2
    $$

    이다. 이것이 모든 $\mathbf{z} \neq \mathbf{0}$에 대해 양수일 필요충분조건은 모든 $\lambda_i > 0$인 것이다. $\square$

    **양반정치성**의 경우 위 조건들에서 고윳값과 선행 소행렬식에 대한 조건이 "$\geq 0$"으로 완화되고, 조건 (5)에서는 $\mathbf{B}$가 계수 부족이어도 된다.

## 촐레스키 분해

<div class="defn" markdown>

**정의 2.** [촐레스키 분해]

양정치행렬 $\mathbf{A}$의 **촐레스키 분해**는 유일한 인수분해

$$
\mathbf{A} = \mathbf{L}\mathbf{L}^T
$$

이며, 여기서 $\mathbf{L}$은 대각 성분이 엄격히 양수인 하삼각행렬이다.

</div>

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

??? proof "증명"

    $\mathbb{R}^m$의 임의의 $\mathbf{y} \neq \mathbf{0}$에 대해 $\mathbf{x} = \mathbf{B}\mathbf{y}$로 두자. $\mathbf{B}$가 완전 열계수를 가지므로 $\mathbf{x} \neq \mathbf{0}$이고, 따라서 $\mathbf{y}^T(\mathbf{B}^T\mathbf{A}\mathbf{B})\mathbf{y} = \mathbf{x}^T\mathbf{A}\mathbf{x} > 0$이다. $\square$

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

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
선행 소행렬식 기준을 이용해 $\mathbf{A} = \begin{pmatrix} 4 & 2 \\ 2 & 3 \end{pmatrix}$이 양정치인지 판정하라.

</div>

??? success "풀이"
    선행 소행렬식은 다음과 같다.

    - 첫 번째 선행 소행렬식: $a_{11} = 4 > 0$
    - 두 번째 선행 소행렬식: $\det(\mathbf{A}) = 4 \times 3 - 2 \times 2 = 12 - 4 = 8 > 0$

    모든 선행 소행렬식이 엄격히 양수이므로 $\mathbf{A}$는 양정치다. 동등하게 고윳값은 $\lambda = \frac{7 \pm \sqrt{49 - 32}}{2} = \frac{7 \pm \sqrt{17}}{2}$이며 둘 다 양수다.

<div class="drillbox" markdown>

**연습문제 2.**
$\mathbf{A}$가 양정치이면 모든 대각 성분 $a_{ii} > 0$임을 증명하라.

</div>

??? success "풀이"
    $\mathbf{e}_i$를 $i$번째 표준기저벡터라 하자($i$번째 자리가 $1$이고 나머지는 $0$). $\mathbf{e}_i \neq \mathbf{0}$이고 $\mathbf{A}$가 양정치이므로

    $$
    \mathbf{e}_i^T \mathbf{A} \mathbf{e}_i > 0
    $$

    이다. 그런데 $\mathbf{e}_i^T \mathbf{A} \mathbf{e}_i = a_{ii}$이므로 모든 $i$에 대해 $a_{ii} > 0$이다. 역은 성립하지 않음에 유의하라. 대각 성분이 양수라고 양정치성이 보장되지는 않는다(예를 들어 $\begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix}$은 대각 성분이 양수지만 고윳값이 $3$과 $-1$이다). $\square$

<div class="drillbox" markdown>

**연습문제 3.**
$\mathbf{A} = \begin{pmatrix} 4 & 6 \\ 6 & 13 \end{pmatrix}$의 촐레스키 분해 $\mathbf{A} = \mathbf{L}\mathbf{L}^T$를 구하라.

</div>

??? success "풀이"
    $\mathbf{L}\mathbf{L}^T = \mathbf{A}$가 되는 하삼각행렬 $\mathbf{L} = \begin{pmatrix} l_{11} & 0 \\ l_{21} & l_{22} \end{pmatrix}$을 찾는다.

    $l_{11}^2 = 4$에서 $l_{11} = 2$.

    $l_{21} l_{11} = 6$에서 $l_{21} = 3$.

    $l_{21}^2 + l_{22}^2 = 13$에서 $9 + l_{22}^2 = 13$이므로 $l_{22} = 2$.

    $$
    \mathbf{L} = \begin{pmatrix} 2 & 0 \\ 3 & 2 \end{pmatrix}
    $$

    확인: $\mathbf{L}\mathbf{L}^T = \begin{pmatrix} 4 & 6 \\ 6 & 13 \end{pmatrix} = \mathbf{A}$.

<div class="drillbox" markdown>

**연습문제 4.**
어떤 공분산행렬 $\boldsymbol{\Sigma}$의 고윳값이 $\lambda_1 = 0.01$과 $\lambda_2 = 100$이다. $\boldsymbol{\Sigma}$는 양정치인가? $\boldsymbol{\Sigma}^{-1}$을 계산할 때의 실무적 함의를 논하라.

</div>

??? success "풀이"
    그렇다. 두 고윳값이 모두 엄격히 양수이므로 $\boldsymbol{\Sigma}$는 양정치다. 그러나 조건수가 $\kappa = \lambda_{\max}/\lambda_{\min} = 100/0.01 = 10{,}000$으로 매우 크다.

    실무적 함의는 다음과 같다.

    - **수치적 불안정:** $\boldsymbol{\Sigma}^{-1}$을 계산할 때의 부동소수 오차가 조건수만큼 증폭된다. $\boldsymbol{\Sigma}^{-1}$의 고윳값 $1/\lambda_1 = 100$이 상당히 손상될 수 있다.
    - **거의 특이:** 가장 작은 고윳값 방향으로 자료가 거의 공선적이며, 이는 두 변수가 거의 완벽하게 상관되어 있다는 뜻이다.
    - **대응책:** 명시적인 역행렬 대신 촐레스키 분해를 쓰거나, 정칙화(능형회귀)를 적용하거나, 적절한 경우 유사역행렬을 쓴다.

<div class="drillbox" markdown>

**연습문제 5.**
$\boldsymbol{\Sigma}$가 양정치이면 마할라노비스 거리 $d^2(\mathbf{x}) = (\mathbf{x} - \boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})$이 0일 필요충분조건이 $\mathbf{x} = \boldsymbol{\mu}$임을 증명하라.

</div>

??? success "풀이"
    $\boldsymbol{\Sigma}$가 양정치이므로 $\boldsymbol{\Sigma}^{-1}$도 양정치다(그 고윳값이 $1/\lambda_i > 0$이다).

    $\mathbf{z} = \mathbf{x} - \boldsymbol{\mu}$로 두면 $d^2 = \mathbf{z}^T\boldsymbol{\Sigma}^{-1}\mathbf{z}$이다.

    $\boldsymbol{\Sigma}^{-1}$의 양정치성에 의해 모든 $\mathbf{z}$에 대해 $\mathbf{z}^T\boldsymbol{\Sigma}^{-1}\mathbf{z} \geq 0$이고, 등호는 $\mathbf{z} = \mathbf{0}$일 때에 한해 성립한다.

    따라서 $d^2 = 0$일 필요충분조건은 $\mathbf{x} - \boldsymbol{\mu} = \mathbf{0}$, 즉 $\mathbf{x} = \boldsymbol{\mu}$이다. 이는 마할라노비스 거리가 (정치성을 만족하는) 제대로 된 거리에 준하는 측도임을 확인해 준다. $\square$

<div class="drillbox" markdown>

**연습문제 6.**
$\mathbf{A}$가 양정치이면 $\mathbf{A}^{-1}$도 양정치임을 보여라. 또 $\mathbf{A}^{1/2}$(제곱근 행렬)이 존재함을 보여라.

</div>

??? success "풀이"
    **역행렬.** $\mathbf{A}$가 양정치이면 가역이다. 임의의 $\mathbf{x} \neq \mathbf{0}$에 대해 $\mathbf{y} = \mathbf{A}^{-1}\mathbf{x}$로 두면 $\mathbf{y} \neq \mathbf{0}$이고

    $$
    \mathbf{x}^T\mathbf{A}^{-1}\mathbf{x} = (\mathbf{A}\mathbf{y})^T\mathbf{A}^{-1}(\mathbf{A}\mathbf{y}) = \mathbf{y}^T\mathbf{A}\mathbf{y} > 0
    $$

    이다($\mathbf{A}$의 대칭성을 썼다). 고윳값으로 보면 더 분명하다. $\mathbf{A}$의 고윳값이 $\lambda_i > 0$이면 $\mathbf{A}^{-1}$의 고윳값은 $1/\lambda_i > 0$이다.

    **제곱근.** 스펙트럼 분해 $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$에서 모든 $\lambda_i > 0$이므로 $\sqrt{\lambda_i}$가 실수로 정의된다.

    $$
    \mathbf{A}^{1/2} = \mathbf{Q}\boldsymbol{\Lambda}^{1/2}\mathbf{Q}^T,
    \qquad \boldsymbol{\Lambda}^{1/2} = \operatorname{diag}(\sqrt{\lambda_1}, \dots, \sqrt{\lambda_p})
    $$

    로 두면 $(\mathbf{A}^{1/2})^2 = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T = \mathbf{A}$이고, $\mathbf{A}^{1/2}$ 자신도 대칭 양정치다.

    ```python
    import numpy as np

    A = np.array([[4., 2.], [2., 3.]])
    lam, Q = np.linalg.eigh(A)

    A_half = Q @ np.diag(np.sqrt(lam)) @ Q.T
    print("A^(1/2) =\n", A_half.round(6))
    print("제곱하면 A 인가:", np.allclose(A_half @ A_half, A))
    print("A^-1 의 고윳값:", np.linalg.eigvalsh(np.linalg.inv(A)).round(6))
    ```

    출력:

    ```
    A^(1/2) =
     [[1.919366 0.562169]
     [0.562169 1.638281]]
    제곱하면 A 인가: True
    A^-1 의 고윳값: [0.179806 0.695194]
    ```

    제곱근 행렬은 **백색화**에 쓰인다. $\mathbf{X} \sim (\boldsymbol{\mu}, \boldsymbol{\Sigma})$일 때 $\boldsymbol{\Sigma}^{-1/2}(\mathbf{X}-\boldsymbol{\mu})$의 공분산은 $\mathbf{I}$가 된다. 마할라노비스 거리가 이 변환 뒤의 유클리드 거리와 같다는 것도 여기서 나온다. $\square$

<div class="drillbox" markdown>

**연습문제 7.**
$\mathbf{A}$, $\mathbf{B}$가 양정치이면 $\mathbf{A} + \mathbf{B}$도 양정치임을 보여라. 그렇다면 곱 $\mathbf{A}\mathbf{B}$는 어떠한가?

</div>

??? success "풀이"
    **합.** 임의의 $\mathbf{x} \neq \mathbf{0}$에 대해

    $$
    \mathbf{x}^T(\mathbf{A}+\mathbf{B})\mathbf{x} = \underbrace{\mathbf{x}^T\mathbf{A}\mathbf{x}}_{>0} + \underbrace{\mathbf{x}^T\mathbf{B}\mathbf{x}}_{>0} > 0
    $$

    이고 $(\mathbf{A}+\mathbf{B})^T = \mathbf{A}+\mathbf{B}$이므로 양정치다. 같은 논법으로 양정치 + 양반정치도 양정치다.

    **곱은 그렇지 않다.** 문제는 $\mathbf{A}\mathbf{B}$가 **대칭이 아닐 수 있다**는 데 있다. 양정치성은 대칭행렬에 대해 정의되므로 곱은 애초에 후보가 되지 못한다.

    ```python
    import numpy as np

    A = np.array([[2., 1.], [1., 2.]])
    B = np.array([[3., -1.], [-1., 1.]])
    AB = A @ B

    print("A, B 양정치:", np.linalg.eigvalsh(A).min() > 0, np.linalg.eigvalsh(B).min() > 0)
    print("A+B 의 고윳값:", np.linalg.eigvalsh(A + B).round(4))
    print("\nAB =\n", AB)
    print("AB 가 대칭인가:", np.allclose(AB, AB.T))
    print("AB 의 고윳값:", np.linalg.eigvals(AB).round(4))
    ```

    출력:

    ```
    A, B 양정치: True True
    A+B 의 고윳값: [3. 5.]

    AB =
     [[ 5. -1.]
     [ 1.  1.]]
    AB 가 대칭인가: False
    AB 의 고윳값: [4.7321 1.2679]
    ```

    다만 곱의 고윳값은 모두 양수다. $\mathbf{A}\mathbf{B}$가 $\mathbf{A}^{1/2}\mathbf{B}\mathbf{A}^{1/2}$(대칭 양정치)와 닮았기 때문이다. **고윳값은 양수지만 대칭이 아니므로 양정치행렬은 아니다.** $\square$

<div class="drillbox" markdown>

**연습문제 8.**
세 변수의 상관계수가 모두 $\rho$로 같다고 하자. 이 행렬이 올바른 상관행렬이 되기 위한 $\rho$의 범위를 구하라. $\rho = -0.8$은 가능한가?

</div>

??? success "풀이"
    상관행렬은 반드시 **양반정치**여야 한다. 등상관행렬 $\mathbf{R} = (1-\rho)\mathbf{I} + \rho\mathbf{J}$의 고윳값은 잘 알려져 있다.

    - $1 + (p-1)\rho$ (중복도 1, 고유벡터 $\mathbf{1}$)
    - $1 - \rho$ (중복도 $p-1$)

    $p = 3$이면 고윳값이 $1 + 2\rho$와 $1 - \rho$(중복도 2)다. 둘 다 음이 아니려면

    $$
    1 + 2\rho \ge 0 \;\text{ 그리고 }\; 1 - \rho \ge 0
    \quad\Longrightarrow\quad -\tfrac{1}{2} \le \rho \le 1
    $$

    이다. 따라서 **$\rho = -0.8$은 불가능하다.**

    ```python
    import numpy as np

    for rho in (-0.8, -0.6, -0.5, 0.5, 0.9):
        R = np.full((3, 3), rho)
        np.fill_diagonal(R, 1.)
        ev = np.linalg.eigvalsh(R)
        print(f"rho={rho:>5}:  최소 고윳값 {ev.min():+.4f}   "
              f"{'유효' if ev.min() >= -1e-12 else '유효하지 않음'}")
    ```

    출력:

    ```
    rho= -0.8:  최소 고윳값 -0.6000   유효하지 않음
    rho= -0.6:  최소 고윳값 -0.2000   유효하지 않음
    rho= -0.5:  최소 고윳값 -0.0000   유효
    rho=  0.5:  최소 고윳값 +0.5000   유효
    rho=  0.9:  최소 고윳값 +0.1000   유효
    ```

    **직관.** 셋이 서로 강하게 음의 상관을 갖는 것은 불가능하다. $X_1$이 $X_2$와 반대로 움직이고 $X_2$가 $X_3$과 반대로 움직이면, $X_1$과 $X_3$은 오히려 **같이** 움직이는 경향이 생긴다. 변수가 많아질수록 제약은 더 강해져 하한이 $-1/(p-1)$이 된다.

    **실무적 함의.** 전문가에게 상관계수를 개별적으로 물어 행렬을 채우면 양반정치가 깨지기 쉽다. 그런 행렬로는 모의실험도 마할라노비스 거리 계산도 할 수 없다. 가장 가까운 양반정치행렬로 보정하는 절차가 따로 필요한 이유다. $\square$

<div class="drillbox" markdown>

**연습문제 9.**
촐레스키 분해 $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^T$를 이용해 공분산이 $\boldsymbol{\Sigma}$인 확률벡터를 생성하는 방법을 설명하고 수치로 확인하라.

</div>

??? success "풀이"
    $\mathbf{Z} \sim (\mathbf{0}, \mathbf{I})$이고 $\mathbf{X} = \mathbf{L}\mathbf{Z}$로 두면

    $$
    \operatorname{Var}(\mathbf{X}) = \mathbf{L}\operatorname{Var}(\mathbf{Z})\mathbf{L}^T = \mathbf{L}\mathbf{I}\mathbf{L}^T = \mathbf{L}\mathbf{L}^T = \boldsymbol{\Sigma}
    $$

    이다. 평균을 $\boldsymbol{\mu}$로 옮기려면 $\mathbf{X} = \boldsymbol{\mu} + \mathbf{L}\mathbf{Z}$로 두면 된다.

    ```python
    import numpy as np

    Sigma = np.array([[4., 2.], [2., 3.]])
    L = np.linalg.cholesky(Sigma)

    rng = np.random.default_rng(0)
    Z = rng.normal(size=(200_000, 2))
    X = Z @ L.T                       # 행 벡터 규약이라 L^T 를 곱한다

    print("L =\n", L.round(6))
    print("\n목표 Sigma =\n", Sigma)
    print("모의실험 공분산 =\n", np.cov(X, rowvar=False).round(3))
    ```

    출력:

    ```
    L =
     [[2.       0.      ]
     [1.       1.414214]]

    목표 Sigma =
     [[4. 2.]
     [2. 3.]]
    모의실험 공분산 =
     [[4.013 1.996]
     [1.996 2.997]]
    ```

    표본공분산이 목표와 소수점 둘째 자리까지 맞는다.

    **왜 촐레스키인가.** $\boldsymbol{\Sigma}^{1/2}$(연습문제 6)을 써도 되지만, 촐레스키는 고윳값 분해보다 약 두 배 빠르고 삼각행렬이라 곱셈도 싸다. 양정치성이 보장될 때 표준적인 선택이다.

    또한 촐레스키는 **양정치성 판정**에도 쓰인다. `np.linalg.cholesky`는 행렬이 양정치가 아니면 예외를 던지므로, 고윳값을 다 구하는 것보다 빠른 검사가 된다. $\square$

<div class="drillbox" markdown>

**연습문제 10.**
$\mathbf{X}$가 완전 열계수를 갖지 않아 $\mathbf{X}^T\mathbf{X}$가 특이행렬일 때, $\lambda > 0$에 대해 $\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I}$는 언제나 양정치임을 보여라.

</div>

??? success "풀이"
    임의의 $\mathbf{v} \neq \mathbf{0}$에 대해

    $$
    \mathbf{v}^T(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})\mathbf{v}
    = \underbrace{\lVert\mathbf{X}\mathbf{v}\rVert^2}_{\ge 0} + \lambda\underbrace{\lVert\mathbf{v}\rVert^2}_{>0}
    > 0
    $$

    이다. 첫 항이 0이 되더라도($\mathbf{v} \in \ker(\mathbf{X})$) 둘째 항이 엄격히 양수이므로 전체가 양수다. 고윳값으로 보면 $\mathbf{X}^T\mathbf{X}$의 고윳값 $\lambda_i \ge 0$이 모두 $\lambda_i + \lambda > 0$으로 밀려 올라간다.

    ```python
    import numpy as np

    X = np.array([[1., 2.], [2., 4.], [3., 6.]])     # 두 열이 공선 (계수 1)
    G = X.T @ X
    print("rank(X) =", np.linalg.matrix_rank(X))
    print("X^T X 의 고윳값:", np.linalg.eigvalsh(G).round(6))

    for lam in (0., 0.5, 2.):
        ev = np.linalg.eigvalsh(G + lam * np.eye(2))
        print(f"lambda={lam}: 최소 고윳값 = {ev.min():.6f}")
    ```

    출력:

    ```
    rank(X) = 1
    X^T X 의 고윳값: [-0. 70.]
    lambda=0.0: 최소 고윳값 = -0.000000
    lambda=0.5: 최소 고윳값 = 0.500000
    lambda=2.0: 최소 고윳값 = 2.000000
    ```

    $\mathbf{X}^T\mathbf{X}$의 최소 고윳값이 0이라 역행렬이 없지만, $\lambda$를 더하면 곧바로 양정치가 되어 역행렬이 존재한다.

    **이것이 능형회귀가 언제나 풀리는 이유다.** 보통최소제곱은 $p > n$이거나 예측변수가 완전히 공선이면 해가 유일하지 않지만,

    $$
    \hat{\boldsymbol{\beta}}_{\text{ridge}} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}
    $$

    는 임의의 $\lambda > 0$에서 언제나 존재하고 유일하다. 능형회귀가 원래 다중공선성을 다루려고 고안된 것도 이 때문이다(18장). $\square$

---

## 정리하며

양정치성은 이차형식이 엄격히 양수임을 보장하는 행렬의 성질이며, 이는 가역성, 잘 정의된 밀도, 유일한 최소제곱해로 이어진다. 핵심적인 동치 특성화들 — 양의 고윳값, 양의 선행 소행렬식, 촐레스키 분해의 존재 — 은 서로 다른 계산적·이론적 도구를 제공한다. 통계에서 $\boldsymbol{\Sigma}$의 양정치성이 다변량 정규분포를 떠받치고, $\mathbf{X}^T\mathbf{X}$의 양정치성이 최소제곱추정량의 존재를 보장한다.
