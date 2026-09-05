# 닮은 행렬

하나의 선형변환은 기저를 어떻게 잡느냐에 따라 여러 다른 행렬로 기술될 수 있다. 같은 변환을 나타내는 두 행렬을 **닮았다(similar)** 고 한다. 닮음을 알아보면 변환의 본질적 성질을 바꾸지 않고도 복잡한 행렬을 대각형이나 조르당형 같은 더 간단한 행렬로 바꿔 놓을 수 있다. 회귀와 다변량 통계에서 닮음은 공분산행렬의 대각화, 주성분 좌표로의 회전, 이차형식의 단순화 등 모든 기저변환 논증의 바탕에 깔려 있다.

## 정의

!!! info "정의 — 닮은 행렬"
    두 정사각행렬 $\mathbf{A}, \mathbf{B} \in \mathbb{R}^{n \times n}$이 **닮았다**는 것은 가역행렬 $\mathbf{P} \in \mathbb{R}^{n \times n}$이 존재하여

    $$
    \mathbf{B} = \mathbf{P}^{-1}\mathbf{A}\mathbf{P}
    $$

    가 성립한다는 뜻이다. 동등하게 $\mathbf{A} = \mathbf{P}\mathbf{B}\mathbf{P}^{-1}$이다. 행렬 $\mathbf{P}$를 **기저변환행렬**이라 한다.

기하적 직관은 간단하다. $\mathbf{A}$가 표준기저에 대해 선형사상 $T: \mathbb{R}^n \to \mathbb{R}^n$을 나타내고 $\mathbf{P}$의 열들이 새 기저의 벡터들이라면, $\mathbf{B} = \mathbf{P}^{-1}\mathbf{A}\mathbf{P}$는 같은 사상 $T$를 새 기저로 표현한 행렬이다.

## 닮음은 동치관계다

닮음은 $n \times n$ 행렬 전체를 동치류로 분할한다.

**반사적.** 모든 행렬은 $\mathbf{P} = \mathbf{I}$를 통해 자기 자신과 닮았다.

$$
\mathbf{A} = \mathbf{I}^{-1}\mathbf{A}\mathbf{I}
$$

**대칭적.** $\mathbf{B} = \mathbf{P}^{-1}\mathbf{A}\mathbf{P}$이면 $\mathbf{Q} = \mathbf{P}^{-1}$에 대해 $\mathbf{A} = \mathbf{Q}^{-1}\mathbf{B}\mathbf{Q}$이다.

**추이적.** $\mathbf{B} = \mathbf{P}^{-1}\mathbf{A}\mathbf{P}$이고 $\mathbf{C} = \mathbf{Q}^{-1}\mathbf{B}\mathbf{Q}$이면

$$
\mathbf{C} = \mathbf{Q}^{-1}\mathbf{P}^{-1}\mathbf{A}\mathbf{P}\mathbf{Q} = (\mathbf{P}\mathbf{Q})^{-1}\mathbf{A}(\mathbf{P}\mathbf{Q})
$$

이므로 $\mathbf{C}$는 $\mathbf{R} = \mathbf{P}\mathbf{Q}$를 통해 $\mathbf{A}$와 닮았다.

## 닮음에 대한 불변량

닮음이 중요한 핵심 이유는 많은 중요한 행렬 관련 양이 **불변량**이라는 데 있다. 즉 한 닮음류에 속한 모든 행렬에 대해 같은 값을 갖는다.

!!! tip "정리 — 닮음 불변량"
    $\mathbf{B} = \mathbf{P}^{-1}\mathbf{A}\mathbf{P}$이면 $\mathbf{A}$와 $\mathbf{B}$는 다음 성질을 공유한다.

    1. **고윳값**(대수적 중복도 포함)
    2. **특성다항식**: $\det(\mathbf{B} - \lambda\mathbf{I}) = \det(\mathbf{A} - \lambda\mathbf{I})$
    3. **대각합**: $\operatorname{tr}(\mathbf{B}) = \operatorname{tr}(\mathbf{A})$
    4. **행렬식**: $\det(\mathbf{B}) = \det(\mathbf{A})$
    5. **계수**: $\operatorname{rank}(\mathbf{B}) = \operatorname{rank}(\mathbf{A})$
    6. **최소다항식**

### 증명 개요 (특성다항식)

$\mathbf{B}$의 특성다항식은

$$
\det(\mathbf{B} - \lambda\mathbf{I}) = \det(\mathbf{P}^{-1}\mathbf{A}\mathbf{P} - \lambda\mathbf{P}^{-1}\mathbf{I}\mathbf{P})
$$

이다. 왼쪽에서 $\mathbf{P}^{-1}$을, 오른쪽에서 $\mathbf{P}$를 묶어내면

$$
= \det\!\bigl(\mathbf{P}^{-1}(\mathbf{A} - \lambda\mathbf{I})\mathbf{P}\bigr) = \det(\mathbf{P}^{-1})\,\det(\mathbf{A} - \lambda\mathbf{I})\,\det(\mathbf{P})
$$

이다. $\det(\mathbf{P}^{-1})\det(\mathbf{P}) = 1$이므로 두 특성다항식이 같다. 고윳값은 특성다항식의 근이므로 고윳값도 일치한다. 대각합은 (중복도를 포함한) 고윳값의 합이고 행렬식은 그 곱이므로 둘 다 불변이다. $\square$

### 증명 개요 (계수)

가역인 $\mathbf{P}$에 대해 사상 $\mathbf{x} \mapsto \mathbf{P}\mathbf{x}$는 $\mathbb{R}^n$ 위의 전단사다. 따라서 $\dim(\operatorname{col}(\mathbf{B})) = \dim(\operatorname{col}(\mathbf{P}^{-1}\mathbf{A}\mathbf{P})) = \dim(\operatorname{col}(\mathbf{A}))$이다. $\square$

## 불변량이 아닌 성질

모든 행렬 성질이 닮음에서 보존되는 것은 아니다. 특히 다음과 같다.

- **대칭성**은 불변이 아니다. 대칭행렬이 대칭이 아닌 행렬과 닮을 수 있다(기저변환행렬 $\mathbf{P}$가 직교행렬일 필요는 없다).
- **양정치성**은 일반적인 닮음에서 불변이 아니다. 다만 양정치성의 고윳값에 의한 특성화는 불변이다.
- **개별 성분**은 당연히 바뀐다.

기저변환행렬 $\mathbf{P}$를 직교행렬($\mathbf{P}^T = \mathbf{P}^{-1}$)로 제한하면, 그 결과인 **직교닮음** $\mathbf{B} = \mathbf{P}^T\mathbf{A}\mathbf{P}$는 대칭성을 보존한다. 스펙트럼 정리가 직교대각화를 내놓는 이유가 여기에 있다.

## 예

다음 행렬을 생각하자.

$$
\mathbf{A} = \begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix}
$$

$\det(\mathbf{A} - \lambda\mathbf{I}) = (4 - \lambda)(3 - \lambda) - 2 = \lambda^2 - 7\lambda + 10 = (\lambda - 5)(\lambda - 2) = 0$에서 고윳값 $\lambda_1 = 5$, $\lambda_2 = 2$를 얻는다.

고유벡터: $\lambda_1 = 5$에 대해 $(\mathbf{A} - 5\mathbf{I})\mathbf{v} = \mathbf{0}$을 풀면 $\mathbf{v}_1 = (1, 1)^T$이고, $\lambda_2 = 2$에 대해 $(\mathbf{A} - 2\mathbf{I})\mathbf{v} = \mathbf{0}$을 풀면 $\mathbf{v}_2 = (1, -2)^T$이다.

$\mathbf{P} = \begin{pmatrix} 1 & 1 \\ 1 & -2 \end{pmatrix}$로 두면

$$
\mathbf{P}^{-1}\mathbf{A}\mathbf{P} = \begin{pmatrix} 5 & 0 \\ 0 & 2 \end{pmatrix} = \boldsymbol{\Lambda}
$$

이다. 대각행렬 $\boldsymbol{\Lambda}$는 $\mathbf{A}$와 닮았고, 두 행렬이 $\operatorname{tr} = 7$, $\det = 10$, 고윳값 $\{5, 2\}$를 공유함을 확인할 수 있다.

## 통계와의 연결

닮은 행렬은 다변량 통계 전반에 등장한다.

- **공분산행렬의 스펙트럼 분해.** $\boldsymbol{\Sigma} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$이면 $\boldsymbol{\Sigma}$는 (직교행렬 $\mathbf{Q}$를 통해) $\boldsymbol{\Lambda}$와 닮았다. 고유기저에서 작업하면 계산이 간단해진다. $\operatorname{tr}(\boldsymbol{\Sigma}) = \sum_i \lambda_i$가 총분산을 주고, $\det(\boldsymbol{\Sigma}) = \prod_i \lambda_i$가 일반화 분산을 측정한다.

- **이차형식의 단순화.** 마할라노비스 거리 $(\mathbf{x} - \boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})$는 $\boldsymbol{\Sigma}^{-1}$이 대각이 되는 고유기저로 옮겨서 분석할 수 있다. 이것이 정규확률벡터의 이차형식이 카이제곱분포를 따름을 유도하는 근거다.

- **모자 행렬 대각합의 불변성.** 회귀에서 예측변수를 어떻게 코딩하거나 척도를 바꾸든 $\operatorname{tr}(\mathbf{H}) = p$인데, 재매개변수화가 $\mathbf{X}^T\mathbf{X}$의 닮음변환에 해당하기 때문이다.

## 요약

두 행렬이 서로 다른 기저에서 같은 선형변환을 나타낼 때 이 둘은 닮았다. 닮은 행렬은 좌표계의 선택에서만 다를 뿐, 고윳값·대각합·행렬식·계수·특성다항식 등 변환의 본질적 성질을 모두 공유한다. 다음 주제인 대각화가 가장 중요한 특수한 경우다. 행렬이 대각이 되는 기저를 찾아 계산과 해석을 단순하게 만드는 것이다.

## 연습문제

**연습문제 1.**
$\mathbf{B} = \mathbf{P}^{-1}\mathbf{A}\mathbf{P}$가 되는 가역행렬 $\mathbf{P}$를 찾아 $\mathbf{A} = \begin{pmatrix} 1 & 2 \\ 0 & 3 \end{pmatrix}$와 $\mathbf{B} = \begin{pmatrix} 3 & 0 \\ 0 & 1 \end{pmatrix}$가 닮았음을 보여라.

??? success "연습문제 1 풀이"
    두 행렬 모두 고윳값이 $\lambda_1 = 1$과 $\lambda_2 = 3$이다($\mathbf{B}$는 이 값들을 대각에 갖는 대각행렬이고, $\mathbf{A}$는 이 값들을 대각에 갖는 상삼각행렬이다).

    $\mathbf{A}$의 고유벡터는 $\lambda = 1$에 대해 $\mathbf{v}_1 = (1, 0)^T$이고, $\lambda = 3$에 대해서는 $(A - 3I)\mathbf{v} = 0$을 풀어 $\mathbf{v}_2 = (1, 1)^T$이다.

    $\mathbf{B}$의 대각이 $\{3, 1\}$ 순서이므로 $\lambda = 3$의 고유벡터가 첫 열에 오도록 고유벡터를 열로 배열한다: $\mathbf{P} = \begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix}$, $\mathbf{P}^{-1} = \begin{pmatrix} 0 & 1 \\ 1 & -1 \end{pmatrix}$.

    확인: $\mathbf{P}^{-1}\mathbf{A}\mathbf{P} = \begin{pmatrix} 0 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} 1 & 2 \\ 0 & 3 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix} = \begin{pmatrix} 3 & 0 \\ 0 & 1 \end{pmatrix} = \mathbf{B}$.

---

**연습문제 2.**
닮은 행렬은 행렬식이 같고 대각합도 같음을 증명하라.

??? success "연습문제 2 풀이"
    $\mathbf{B} = \mathbf{P}^{-1}\mathbf{A}\mathbf{P}$이면

    $$
    \det(\mathbf{B}) = \det(\mathbf{P}^{-1}\mathbf{A}\mathbf{P}) = \det(\mathbf{P}^{-1})\det(\mathbf{A})\det(\mathbf{P}) = \frac{1}{\det(\mathbf{P})}\det(\mathbf{A})\det(\mathbf{P}) = \det(\mathbf{A})
    $$

    이다. 대각합의 경우 순환 성질 $\operatorname{tr}(\mathbf{X}\mathbf{Y}\mathbf{Z}) = \operatorname{tr}(\mathbf{Z}\mathbf{X}\mathbf{Y})$를 쓰면

    $$
    \operatorname{tr}(\mathbf{B}) = \operatorname{tr}(\mathbf{P}^{-1}\mathbf{A}\mathbf{P}) = \operatorname{tr}(\mathbf{A}\mathbf{P}\mathbf{P}^{-1}) = \operatorname{tr}(\mathbf{A})
    $$

    이다. $\square$

---

**연습문제 3.**
고윳값, 대각합, 행렬식이 모두 같지만 닮지는 않은 두 개의 $2 \times 2$ 행렬의 예를 들어라.

??? success "연습문제 3 풀이"
    $\mathbf{A} = \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix}$와 $\mathbf{B} = \begin{pmatrix} 2 & 1 \\ 0 & 2 \end{pmatrix}$를 생각하자.

    둘 다 고윳값이 $\lambda = 2$(대수적 중복도 2)이고 $\operatorname{tr} = 4$, $\det = 4$이다.

    그러나 $\mathbf{A} = 2\mathbf{I}$는 모든 행렬과 교환되므로 임의의 가역 $\mathbf{P}$에 대해 $\mathbf{P}^{-1}\mathbf{A}\mathbf{P} = \mathbf{P}^{-1}(2\mathbf{I})\mathbf{P} = 2\mathbf{I} = \mathbf{A}$이다. $\mathbf{B} \neq \mathbf{A}$이므로 어떤 닮음변환도 $\mathbf{A}$를 $\mathbf{B}$로 바꿀 수 없다. 차이는 $\mathbf{A}$가 대각화 가능한 반면(기하적 중복도 2) $\mathbf{B}$는 그렇지 않다는 데 있다(기하적 중복도 1).

---

**연습문제 4.**
닮음 개념을 이용해, 회귀모형을 재매개변수화해도(예: 예측변수를 중심화해도) $\operatorname{tr}(\mathbf{H})$나 $\mathbf{X}^T\mathbf{X}$의 고윳값이 바뀌지 않는 이유를 설명하라.

??? success "연습문제 4 풀이"
    재매개변수화는 어떤 가역행렬 $\mathbf{C}$에 대해 $\mathbf{X}$를 $\mathbf{X}\mathbf{C}$로 바꾸는 것에 해당한다. 모자 행렬은 다음과 같이 변환된다.

    $$
    \mathbf{H}' = \mathbf{X}\mathbf{C}(\mathbf{C}^T\mathbf{X}^T\mathbf{X}\mathbf{C})^{-1}\mathbf{C}^T\mathbf{X}^T = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T = \mathbf{H}
    $$

    모자 행렬은 대각합만이 아니라 완전히 불변이다. 그람 행렬 $\mathbf{X}^T\mathbf{X}$와 $\mathbf{C}^T\mathbf{X}^T\mathbf{X}\mathbf{C}$는 합동변환으로 연결된다. ($\mathbf{C}$가 직교행렬이 아닌 한) 표준적인 닮음은 아니지만 핵심 성질을 공유한다. 계수가 같고 적합값이 동일하다. $\operatorname{tr}(\mathbf{H}) = p$의 불변성은 재매개변수화로 모수의 개수가 바뀌지 않는다는 사실을 반영한다.
