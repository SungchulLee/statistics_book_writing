# 대각화 가능 행렬의 대각형

행렬에 적용할 수 있는 모든 닮음변환 가운데 가장 유용한 결과는 대각행렬이다. 대각화 가능한 행렬은 $\mathbf{A} = \mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^{-1}$로 분해되며, 이 덕분에 거듭제곱, 지수함수, 이차형식의 계산이 간단해진다. 통계에서 공분산행렬은 대칭이므로 언제나 대각화 가능하고, 그 대각형이 주성분을 드러낸다. 이 절에서는 대각화 가능성을 정의하고 핵심 존재 정리를 진술한 뒤 계산 방법을 보인다.

## 정의

!!! info "정의 — 대각화 가능 행렬"
    정사각행렬 $\mathbf{A} \in \mathbb{R}^{n \times n}$이 대각행렬과 닮았으면 **대각화 가능(diagonalizable)** 하다고 한다. 즉 가역행렬 $\mathbf{P} \in \mathbb{R}^{n \times n}$과 대각행렬 $\boldsymbol{\Lambda} = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$이 존재하여

    $$
    \mathbf{A} = \mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^{-1}
    $$

    이 성립한다는 뜻이다. 동등하게 $\boldsymbol{\Lambda} = \mathbf{P}^{-1}\mathbf{A}\mathbf{P}$이다.

$\mathbf{P}$의 열은 $\mathbf{A}$의 고유벡터이고, $\boldsymbol{\Lambda}$의 대각 성분은 그에 대응하는 고윳값이다. $\mathbf{P} = (\mathbf{v}_1 \mid \mathbf{v}_2 \mid \cdots \mid \mathbf{v}_n)$으로 쓰면 분해 $\mathbf{A}\mathbf{P} = \mathbf{P}\boldsymbol{\Lambda}$는 각 $i$에 대해 $\mathbf{A}\mathbf{v}_i = \lambda_i \mathbf{v}_i$인 것과 동등하다.

## 언제 행렬이 대각화 가능한가

!!! tip "정리 — 대각화 가능성의 판정"
    행렬 $\mathbf{A} \in \mathbb{R}^{n \times n}$이 대각화 가능할 필요충분조건은 일차독립인 고유벡터를 $n$개 갖는 것이다.

**증명 개요.** $\mathbf{A}$가 일차독립인 고유벡터 $\mathbf{v}_1, \dots, \mathbf{v}_n$을 가지면 이들을 $\mathbf{P}$의 열로 놓는다. 그러면 $\mathbf{P}$는 (열들이 일차독립이므로) 가역이고, 고윳값 관계에 의해

$$
\mathbf{A}\mathbf{P} = \mathbf{A}(\mathbf{v}_1 \mid \cdots \mid \mathbf{v}_n) = (\lambda_1\mathbf{v}_1 \mid \cdots \mid \lambda_n\mathbf{v}_n) = \mathbf{P}\boldsymbol{\Lambda}
$$

이다. 양변 왼쪽에 $\mathbf{P}^{-1}$을 곱하면 $\boldsymbol{\Lambda} = \mathbf{P}^{-1}\mathbf{A}\mathbf{P}$를 얻는다.

역으로 $\mathbf{A} = \mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^{-1}$이면 $\mathbf{A}\mathbf{P} = \mathbf{P}\boldsymbol{\Lambda}$이므로 $\mathbf{P}$의 각 열은 $\mathbf{A}$의 고유벡터다. $\mathbf{P}$가 가역이므로 이 $n$개의 고유벡터는 일차독립이다. $\square$

### 충분조건

대각화 가능성을 보장하는 중요한 충분조건이 몇 가지 있다.

- **서로 다른 고윳값.** $\mathbf{A}$가 서로 다른 고윳값을 $n$개 가지면 대응하는 고유벡터들이 일차독립이므로 $\mathbf{A}$는 대각화 가능하다.
- **대칭행렬.** 모든 실대칭행렬은 대각화 가능하다(스펙트럼 정리). 나아가 고유벡터를 정규직교로 고를 수 있으므로 $\mathbf{P}$가 직교행렬이 된다.
- **대수적 중복도와 기하적 중복도가 일치.** 각 고윳값 $\lambda_i$에 대해 기하적 중복도(고유공간의 차원)가 대수적 중복도(특성다항식의 근으로서의 중복도)와 같다.

## 거듭제곱과 지수

대각형은 행렬의 거듭제곱을 극적으로 단순화한다.

$$
\mathbf{A}^k = \mathbf{P}\boldsymbol{\Lambda}^k\mathbf{P}^{-1} = \mathbf{P}\operatorname{diag}(\lambda_1^k, \dots, \lambda_n^k)\mathbf{P}^{-1}
$$

마찬가지로 행렬 지수는

$$
e^{\mathbf{A}} = \mathbf{P}\operatorname{diag}(e^{\lambda_1}, \dots, e^{\lambda_n})\mathbf{P}^{-1}
$$

이다. 더 일반적으로, 고윳값들 위에서 정의된 임의의 함수 $f$에 대해

$$
f(\mathbf{A}) = \mathbf{P}\operatorname{diag}\!\bigl(f(\lambda_1), \dots, f(\lambda_n)\bigr)\mathbf{P}^{-1}
$$

이다.

## 예 — 대각화 가능한 행렬

다음을 생각하자.

$$
\mathbf{A} = \begin{pmatrix} 2 & 1 \\ 0 & 3 \end{pmatrix}
$$

특성다항식은 $\det(\mathbf{A} - \lambda\mathbf{I}) = (2 - \lambda)(3 - \lambda) = 0$이므로 서로 다른 고윳값 $\lambda_1 = 2$와 $\lambda_2 = 3$을 얻는다.

$\lambda_1 = 2$에 대해: $(\mathbf{A} - 2\mathbf{I})\mathbf{v} = \mathbf{0}$에서 $\mathbf{v}_1 = (1, 0)^T$.

$\lambda_2 = 3$에 대해: $(\mathbf{A} - 3\mathbf{I})\mathbf{v} = \mathbf{0}$에서 $\mathbf{v}_2 = (1, 1)^T$.

$\mathbf{P} = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$, $\mathbf{P}^{-1} = \begin{pmatrix} 1 & -1 \\ 0 & 1 \end{pmatrix}$로 두면

$$
\mathbf{P}^{-1}\mathbf{A}\mathbf{P} = \begin{pmatrix} 2 & 0 \\ 0 & 3 \end{pmatrix} = \boldsymbol{\Lambda}
$$

임을 확인할 수 있다.

이 분해를 쓰면 $\mathbf{A}^{10} = \mathbf{P}\operatorname{diag}(2^{10}, 3^{10})\mathbf{P}^{-1} = \mathbf{P}\operatorname{diag}(1024, 59049)\mathbf{P}^{-1}$이다.

## 예 — 대각화 불가능한 행렬

행렬

$$
\mathbf{A} = \begin{pmatrix} 2 & 1 \\ 0 & 2 \end{pmatrix}
$$

은 대수적 중복도가 2인 중복 고윳값 $\lambda = 2$를 갖지만, 고유공간 $\ker(\mathbf{A} - 2\mathbf{I}) = \operatorname{span}\{(1, 0)^T\}$의 차원은 1이다(기하적 중복도 1). 일차독립인 고유벡터가 하나뿐이므로 $\mathbf{A}$는 대각화 가능하지 않다. 이런 행렬에는 대신 조르당 표준형이 필요하다.

## 통계와의 연결

대각화는 여러 핵심 통계 방법을 떠받치는 계산 엔진이다.

- **주성분분석.** 표본 공분산행렬 $\mathbf{S}$는 대칭이므로 대각화 가능하다: $\mathbf{S} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$. $\mathbf{Q}$의 열은 주성분 방향이고 $\boldsymbol{\Lambda}$는 각 성분이 설명하는 분산을 담는다.

- **이차형식.** $\mathbf{A}$가 고유분해 $\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$를 갖는 대칭행렬이면

$$
\mathbf{x}^T\mathbf{A}\mathbf{x} = \mathbf{z}^T\boldsymbol{\Lambda}\mathbf{z} = \sum_{i=1}^n \lambda_i z_i^2
$$

이다. 여기서 $\mathbf{z} = \mathbf{Q}^T\mathbf{x}$이다. 이는 이차형식을 가중된 제곱합으로 분리해 주며, 카이제곱분포를 유도하는 데 필수적이다.

- **행렬의 역.** $\boldsymbol{\Sigma} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$가 양정치일 때 $\boldsymbol{\Sigma}^{-1} = \mathbf{Q}\boldsymbol{\Lambda}^{-1}\mathbf{Q}^T = \mathbf{Q}\operatorname{diag}(1/\lambda_1, \dots, 1/\lambda_n)\mathbf{Q}^T$이며, 이는 계산 효율이 좋고 수치적으로도 안정적이다.

## 요약

행렬이 일차독립인 고유벡터를 $n$개 온전히 가질 때 대각화 가능하며, 그때 분해 $\mathbf{A} = \mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^{-1}$이 성립한다. 이 분해는 행렬 연산을 고윳값에 대한 스칼라 연산으로 환원한다. 모든 공분산행렬을 포함한 대칭행렬은 언제나 대각화 가능하며, 그래서 고유분해가 통계 이론의 기본 도구가 된다. 대각화되지 않는 행렬에는 다음에 다룰 조르당 표준형이 필요하다.

## 연습문제

**연습문제 1.**
행렬 $\mathbf{A} = \begin{pmatrix} 4 & 1 \\ 0 & 3 \end{pmatrix}$의 고윳값과 고유벡터, 그리고 행렬 $\mathbf{P}$와 $\boldsymbol{\Lambda}$를 구해 대각화하라.

??? success "풀이"
    특성다항식은 $\det(\mathbf{A} - \lambda\mathbf{I}) = (4-\lambda)(3-\lambda) = 0$이므로 고윳값은 $\lambda_1 = 4$와 $\lambda_2 = 3$이다.

    $\lambda_1 = 4$에 대해: $(\mathbf{A} - 4\mathbf{I})\mathbf{v} = \begin{pmatrix} 0 & 1 \\ 0 & -1 \end{pmatrix}\mathbf{v} = \mathbf{0}$이므로 $\mathbf{v}_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$.

    $\lambda_2 = 3$에 대해: $(\mathbf{A} - 3\mathbf{I})\mathbf{v} = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}\mathbf{v} = \mathbf{0}$이므로 $\mathbf{v}_2 = \begin{pmatrix} -1 \\ 1 \end{pmatrix}$.

    따라서

    $$
    \mathbf{P} = \begin{pmatrix} 1 & -1 \\ 0 & 1 \end{pmatrix}, \quad \boldsymbol{\Lambda} = \begin{pmatrix} 4 & 0 \\ 0 & 3 \end{pmatrix}, \quad \mathbf{A} = \mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^{-1}
    $$

---

**연습문제 2.**
$\mathbf{A}$가 $\mathbf{A} = \mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^{-1}$로 대각화 가능하면 임의의 양의 정수 $k$에 대해 $\mathbf{A}^k = \mathbf{P}\boldsymbol{\Lambda}^k\mathbf{P}^{-1}$임을 증명하라.

??? success "풀이"
    귀납법으로 진행한다. 기저 단계 $k = 1$은 정의에 의해 성립한다.

    $\mathbf{A}^k = \mathbf{P}\boldsymbol{\Lambda}^k\mathbf{P}^{-1}$이라고 가정하자. 그러면

    $$
    \mathbf{A}^{k+1} = \mathbf{A}^k \cdot \mathbf{A} = \mathbf{P}\boldsymbol{\Lambda}^k\mathbf{P}^{-1} \cdot \mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^{-1} = \mathbf{P}\boldsymbol{\Lambda}^k\boldsymbol{\Lambda}\mathbf{P}^{-1} = \mathbf{P}\boldsymbol{\Lambda}^{k+1}\mathbf{P}^{-1}
    $$

    이다. 핵심은 $\mathbf{P}^{-1}\mathbf{P} = \mathbf{I}$로 상쇄되는 것이다. $\boldsymbol{\Lambda}^k = \operatorname{diag}(\lambda_1^k, \dots, \lambda_n^k)$이므로 행렬의 거듭제곱 계산이 고윳값의 스칼라 거듭제곱 계산으로 환원된다. $\square$

---

**연습문제 3.**
$\boldsymbol{\Sigma}$가 고윳값 $\lambda_1 = 5$, $\lambda_2 = 2$를 갖는 $2 \times 2$ 공분산행렬이라 하자. $\boldsymbol{\Sigma}$를 명시적으로 계산하지 않고 $\operatorname{tr}(\boldsymbol{\Sigma})$, $\det(\boldsymbol{\Sigma})$, 그리고 $\boldsymbol{\Sigma}^{-1}$의 고윳값을 구하라.

??? success "풀이"
    대각합은 고윳값의 합이므로

    $$
    \operatorname{tr}(\boldsymbol{\Sigma}) = \lambda_1 + \lambda_2 = 5 + 2 = 7
    $$

    행렬식은 고윳값의 곱이므로

    $$
    \det(\boldsymbol{\Sigma}) = \lambda_1 \cdot \lambda_2 = 5 \times 2 = 10
    $$

    $\boldsymbol{\Sigma}^{-1}$의 고윳값은 $\boldsymbol{\Sigma}$ 고윳값의 역수다.

    $$
    \lambda_1(\boldsymbol{\Sigma}^{-1}) = \frac{1}{5} = 0.2, \quad \lambda_2(\boldsymbol{\Sigma}^{-1}) = \frac{1}{2} = 0.5
    $$

---

**연습문제 4.**
대각화 가능하지 않은 $2 \times 2$ 실행렬의 예를 들어라. 일차독립인 고유벡터가 두 개보다 적음을 보여 대각화할 수 없음을 증명하라.

??? success "풀이"
    $\mathbf{A} = \begin{pmatrix} 2 & 1 \\ 0 & 2 \end{pmatrix}$를 생각하자. 특성다항식은 $(2 - \lambda)^2 = 0$이므로 $\lambda = 2$가 유일한 고윳값이다(대수적 중복도 2).

    $\lambda = 2$의 고유공간은 다음 행렬의 영공간이다.

    $$
    \mathbf{A} - 2\mathbf{I} = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}
    $$

    이 행렬의 계수는 1이므로 영공간의 차원은 1이다(기하적 중복도 1). 상수배를 무시하면 고유벡터는 $\mathbf{v} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$ 하나뿐이다.

    $\mathbf{P}$를 만들려면 일차독립인 고유벡터가 2개 필요한데 1개뿐이므로 이 행렬은 대각화 가능하지 않다. $\square$

---

**연습문제 5.**
모든 실대칭행렬이 대각화 가능한 이유와, 대각화하는 행렬을 직교행렬로 고를 수 있는 이유를 설명하라. 이 성질이 공분산행렬에 왜 중요한가?

??? success "풀이"
    스펙트럼 정리는 모든 실대칭행렬이 (중복도를 세어) $n$개의 실수 고윳값과 $n$개의 정규직교 고유벡터를 온전히 가짐을 보장한다. 구체적으로, 서로 다른 고윳값에 대응하는 고유벡터는 직교하고, 중복 고윳값의 경우 그 고유공간을 그람–슈미트로 정규직교화할 수 있다. 이 고유벡터들을 $\mathbf{Q}$의 열로 배열하면 직교행렬($\mathbf{Q}^T\mathbf{Q} = \mathbf{I}$)이 되므로 $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$이다.

    공분산행렬 $\boldsymbol{\Sigma}$에 대해 이 스펙트럼 분해가 주성분분석(PCA)의 토대다. 고유벡터가 주성분 방향을 주고, 고윳값이 각 성분이 설명하는 분산을 주며, $\mathbf{Q}$의 직교성은 주성분들이 서로 무상관임을 뜻한다. 이 분해는 계산도 단순하게 만든다: $\boldsymbol{\Sigma}^{-1} = \mathbf{Q}\boldsymbol{\Lambda}^{-1}\mathbf{Q}^T$이고 $\boldsymbol{\Sigma}^{1/2} = \mathbf{Q}\boldsymbol{\Lambda}^{1/2}\mathbf{Q}^T$이다.
