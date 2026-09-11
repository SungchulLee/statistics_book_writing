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

<div class="thmbox" markdown>

### 정리 1. 닮음 불변량 { .thm }

$\mathbf{B} = \mathbf{P}^{-1}\mathbf{A}\mathbf{P}$이면 $\mathbf{A}$와 $\mathbf{B}$는 다음 성질을 공유한다.

1. **고윳값**(대수적 중복도 포함)
2. **특성다항식**: $\det(\mathbf{B} - \lambda\mathbf{I}) = \det(\mathbf{A} - \lambda\mathbf{I})$
3. **대각합**: $\operatorname{tr}(\mathbf{B}) = \operatorname{tr}(\mathbf{A})$
4. **행렬식**: $\det(\mathbf{B}) = \det(\mathbf{A})$
5. **계수**: $\operatorname{rank}(\mathbf{B}) = \operatorname{rank}(\mathbf{A})$
6. **최소다항식**

</div>

??? proof "증명 개요 (특성다항식)"


    $\mathbf{B}$의 특성다항식은

    $$
    \det(\mathbf{B} - \lambda\mathbf{I}) = \det(\mathbf{P}^{-1}\mathbf{A}\mathbf{P} - \lambda\mathbf{P}^{-1}\mathbf{I}\mathbf{P})
    $$

    이다. 왼쪽에서 $\mathbf{P}^{-1}$을, 오른쪽에서 $\mathbf{P}$를 묶어내면

    $$
    = \det\!\bigl(\mathbf{P}^{-1}(\mathbf{A} - \lambda\mathbf{I})\mathbf{P}\bigr) = \det(\mathbf{P}^{-1})\,\det(\mathbf{A} - \lambda\mathbf{I})\,\det(\mathbf{P})
    $$

    이다. $\det(\mathbf{P}^{-1})\det(\mathbf{P}) = 1$이므로 두 특성다항식이 같다. 고윳값은 특성다항식의 근이므로 고윳값도 일치한다. 대각합은 (중복도를 포함한) 고윳값의 합이고 행렬식은 그 곱이므로 둘 다 불변이다. $\square$

??? proof "증명 개요 (계수)"


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

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
$\mathbf{B} = \mathbf{P}^{-1}\mathbf{A}\mathbf{P}$가 되는 가역행렬 $\mathbf{P}$를 찾아 $\mathbf{A} = \begin{pmatrix} 1 & 2 \\ 0 & 3 \end{pmatrix}$와 $\mathbf{B} = \begin{pmatrix} 3 & 0 \\ 0 & 1 \end{pmatrix}$가 닮았음을 보여라.

</div>

??? success "풀이"
    두 행렬 모두 고윳값이 $\lambda_1 = 1$과 $\lambda_2 = 3$이다($\mathbf{B}$는 이 값들을 대각에 갖는 대각행렬이고, $\mathbf{A}$는 이 값들을 대각에 갖는 상삼각행렬이다).

    $\mathbf{A}$의 고유벡터는 $\lambda = 1$에 대해 $\mathbf{v}_1 = (1, 0)^T$이고, $\lambda = 3$에 대해서는 $(A - 3I)\mathbf{v} = 0$을 풀어 $\mathbf{v}_2 = (1, 1)^T$이다.

    $\mathbf{B}$의 대각이 $\{3, 1\}$ 순서이므로 $\lambda = 3$의 고유벡터가 첫 열에 오도록 고유벡터를 열로 배열한다: $\mathbf{P} = \begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix}$, $\mathbf{P}^{-1} = \begin{pmatrix} 0 & 1 \\ 1 & -1 \end{pmatrix}$.

    확인: $\mathbf{P}^{-1}\mathbf{A}\mathbf{P} = \begin{pmatrix} 0 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} 1 & 2 \\ 0 & 3 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix} = \begin{pmatrix} 3 & 0 \\ 0 & 1 \end{pmatrix} = \mathbf{B}$.

<div class="drillbox" markdown>

**연습문제 2.**
닮은 행렬은 행렬식이 같고 대각합도 같음을 증명하라.

</div>

??? success "풀이"
    $\mathbf{B} = \mathbf{P}^{-1}\mathbf{A}\mathbf{P}$이면

    $$
    \det(\mathbf{B}) = \det(\mathbf{P}^{-1}\mathbf{A}\mathbf{P}) = \det(\mathbf{P}^{-1})\det(\mathbf{A})\det(\mathbf{P}) = \frac{1}{\det(\mathbf{P})}\det(\mathbf{A})\det(\mathbf{P}) = \det(\mathbf{A})
    $$

    이다. 대각합의 경우 순환 성질 $\operatorname{tr}(\mathbf{X}\mathbf{Y}\mathbf{Z}) = \operatorname{tr}(\mathbf{Z}\mathbf{X}\mathbf{Y})$를 쓰면

    $$
    \operatorname{tr}(\mathbf{B}) = \operatorname{tr}(\mathbf{P}^{-1}\mathbf{A}\mathbf{P}) = \operatorname{tr}(\mathbf{A}\mathbf{P}\mathbf{P}^{-1}) = \operatorname{tr}(\mathbf{A})
    $$

    이다. $\square$

<div class="drillbox" markdown>

**연습문제 3.**
고윳값, 대각합, 행렬식이 모두 같지만 닮지는 않은 두 개의 $2 \times 2$ 행렬의 예를 들어라.

</div>

??? success "풀이"
    $\mathbf{A} = \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix}$와 $\mathbf{B} = \begin{pmatrix} 2 & 1 \\ 0 & 2 \end{pmatrix}$를 생각하자.

    둘 다 고윳값이 $\lambda = 2$(대수적 중복도 2)이고 $\operatorname{tr} = 4$, $\det = 4$이다.

    그러나 $\mathbf{A} = 2\mathbf{I}$는 모든 행렬과 교환되므로 임의의 가역 $\mathbf{P}$에 대해 $\mathbf{P}^{-1}\mathbf{A}\mathbf{P} = \mathbf{P}^{-1}(2\mathbf{I})\mathbf{P} = 2\mathbf{I} = \mathbf{A}$이다. $\mathbf{B} \neq \mathbf{A}$이므로 어떤 닮음변환도 $\mathbf{A}$를 $\mathbf{B}$로 바꿀 수 없다. 차이는 $\mathbf{A}$가 대각화 가능한 반면(기하적 중복도 2) $\mathbf{B}$는 그렇지 않다는 데 있다(기하적 중복도 1).

<div class="drillbox" markdown>

**연습문제 4.**
닮음 개념을 이용해, 회귀모형을 재매개변수화해도(예: 예측변수를 중심화해도) $\operatorname{tr}(\mathbf{H})$나 $\mathbf{X}^T\mathbf{X}$의 고윳값이 바뀌지 않는 이유를 설명하라.

</div>

??? success "풀이"
    재매개변수화는 어떤 가역행렬 $\mathbf{C}$에 대해 $\mathbf{X}$를 $\mathbf{X}\mathbf{C}$로 바꾸는 것에 해당한다. 모자 행렬은 다음과 같이 변환된다.

    $$
    \mathbf{H}' = \mathbf{X}\mathbf{C}(\mathbf{C}^T\mathbf{X}^T\mathbf{X}\mathbf{C})^{-1}\mathbf{C}^T\mathbf{X}^T = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T = \mathbf{H}
    $$

    모자 행렬은 대각합만이 아니라 완전히 불변이다. 그람 행렬 $\mathbf{X}^T\mathbf{X}$와 $\mathbf{C}^T\mathbf{X}^T\mathbf{X}\mathbf{C}$는 합동변환으로 연결된다. ($\mathbf{C}$가 직교행렬이 아닌 한) 표준적인 닮음은 아니지만 핵심 성질을 공유한다. 계수가 같고 적합값이 동일하다. $\operatorname{tr}(\mathbf{H}) = p$의 불변성은 재매개변수화로 모수의 개수가 바뀌지 않는다는 사실을 반영한다.

<div class="drillbox" markdown>

**연습문제 5.**
$\mathbf{B} = \mathbf{P}^{-1}\mathbf{A}\mathbf{P}$이면 모든 자연수 $k$에 대해 $\mathbf{B}^k = \mathbf{P}^{-1}\mathbf{A}^k\mathbf{P}$임을 보여라. 이를 이용해 본문의 $\mathbf{A} = \begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix}$에 대해 $\mathbf{A}^{10}$을 구하라.

</div>

??? success "풀이"
    가운데의 $\mathbf{P}\mathbf{P}^{-1} = \mathbf{I}$가 차례로 소거된다.

    $$
    \mathbf{B}^k = (\mathbf{P}^{-1}\mathbf{A}\mathbf{P})(\mathbf{P}^{-1}\mathbf{A}\mathbf{P})\cdots(\mathbf{P}^{-1}\mathbf{A}\mathbf{P})
    = \mathbf{P}^{-1}\mathbf{A}^k\mathbf{P}
    $$

    거꾸로 읽으면 $\mathbf{A}^k = \mathbf{P}\mathbf{B}^k\mathbf{P}^{-1}$이다. $\mathbf{B} = \boldsymbol{\Lambda} = \operatorname{diag}(5, 2)$가 대각이면 $\boldsymbol{\Lambda}^{10} = \operatorname{diag}(5^{10}, 2^{10})$이므로 거듭제곱이 곱셈 한 번으로 끝난다. **대각화가 유용한 가장 실용적인 이유다.**

    ```python
    import numpy as np

    A = np.array([[4., 1.], [2., 3.]])
    P = np.array([[1., 1.], [1., -2.]])          # 고유벡터를 열로
    Lam = np.diag([5., 2.])

    A10_direct = np.linalg.matrix_power(A, 10)
    A10_via_diag = P @ np.diag([5.**10, 2.**10]) @ np.linalg.inv(P)

    print("A^10 =\n", A10_direct)
    print("일치하는가:", np.allclose(A10_direct, A10_via_diag))
    ```

    출력:

    ```
    A^10 =
     [[6510758. 3254867.]
     [6509734. 3255891.]]
    일치하는가: True
    ```

    직접 $10$번 곱한 결과와 대각화를 거친 결과가 같다. 행렬이 커지고 $k$가 커질수록 이 차이가 결정적이 된다. $\square$

<div class="drillbox" markdown>

**연습문제 6.**
다항식 $f(x) = c_0 + c_1 x + \cdots + c_m x^m$에 대해, $\mathbf{A}$와 $\mathbf{B}$가 닮았으면 $f(\mathbf{A})$와 $f(\mathbf{B})$도 같은 $\mathbf{P}$로 닮았음을 보여라.

</div>

??? success "풀이"
    연습문제 5에서 $\mathbf{B}^k = \mathbf{P}^{-1}\mathbf{A}^k\mathbf{P}$이고, $k = 0$일 때도 $\mathbf{B}^0 = \mathbf{I} = \mathbf{P}^{-1}\mathbf{I}\mathbf{P}$로 성립한다. 따라서

    $$
    f(\mathbf{B}) = \sum_{k=0}^{m} c_k \mathbf{B}^k
    = \sum_{k=0}^{m} c_k \mathbf{P}^{-1}\mathbf{A}^k \mathbf{P}
    = \mathbf{P}^{-1}\!\left(\sum_{k=0}^{m} c_k \mathbf{A}^k\right)\!\mathbf{P}
    = \mathbf{P}^{-1} f(\mathbf{A}) \mathbf{P}
    $$

    이다. 가운데 등식에서 $\mathbf{P}^{-1}$과 $\mathbf{P}$를 합의 밖으로 묶어낼 수 있는 것이 핵심이다.

    **따름정리.** $f(\mathbf{A}) = \mathbf{O}$이면 $f(\mathbf{B}) = \mathbf{P}^{-1}\mathbf{O}\mathbf{P} = \mathbf{O}$이다. 곧 **최소다항식이 닮음 불변량**이라는 사실(본문 정리의 6번)이 여기서 따라 나온다.

    이 성질은 다항식을 넘어 수렴하는 멱급수에도 그대로 확장된다. 예컨대 행렬 지수함수는 $e^{\mathbf{B}} = \mathbf{P}^{-1}e^{\mathbf{A}}\mathbf{P}$를 만족한다. $\square$

<div class="drillbox" markdown>

**연습문제 7.**
대칭행렬이 대칭이 아닌 행렬과 닮을 수 있음을 구체적인 예로 보여라.

</div>

??? success "풀이"
    $\mathbf{A} = \begin{pmatrix} 1 & 0 \\ 0 & 2 \end{pmatrix}$(대칭)과 $\mathbf{P} = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$(직교가 **아닌** 가역행렬)을 잡으면

    $$
    \mathbf{P}^{-1}\mathbf{A}\mathbf{P}
    = \begin{pmatrix} 1 & -1 \\ 0 & 1 \end{pmatrix}
      \begin{pmatrix} 1 & 0 \\ 0 & 2 \end{pmatrix}
      \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}
    = \begin{pmatrix} 1 & -1 \\ 0 & 2 \end{pmatrix}
    $$

    이다. 오른쪽은 대칭이 아니지만 고윳값은 여전히 $\{1, 2\}$다.

    ```python
    import numpy as np

    A = np.diag([1., 2.])
    P = np.array([[1., 1.], [0., 1.]])
    B = np.linalg.inv(P) @ A @ P

    print("B =\n", B)
    print("B가 대칭인가:", np.allclose(B, B.T))
    print("A의 고윳값:", np.sort(np.linalg.eigvals(A)))
    print("B의 고윳값:", np.sort(np.linalg.eigvals(B)))
    ```

    출력:

    ```
    B =
     [[ 1. -1.]
     [ 0.  2.]]
    B가 대칭인가: False
    A의 고윳값: [1. 2.]
    B의 고윳값: [1. 2.]
    ```

    **대칭성은 닮음 불변량이 아니다.** 기저를 직교가 아닌 방향으로 비틀면 대칭성이 깨진다. 반면 고윳값은 그대로다. $\square$

<div class="drillbox" markdown>

**연습문제 8.**
$\mathbf{Q}$가 직교행렬이고 $\mathbf{A}$가 대칭이면 $\mathbf{Q}^T\mathbf{A}\mathbf{Q}$도 대칭임을 보여라. 연습문제 7과 견주어 무엇이 달라졌는지 설명하라.

</div>

??? success "풀이"
    $\mathbf{Q}$가 직교이므로 $\mathbf{Q}^{-1} = \mathbf{Q}^T$이고, 따라서 $\mathbf{Q}^T\mathbf{A}\mathbf{Q}$는 닮음변환이다. 전치를 취하면

    $$
    (\mathbf{Q}^T\mathbf{A}\mathbf{Q})^T = \mathbf{Q}^T \mathbf{A}^T (\mathbf{Q}^T)^T = \mathbf{Q}^T\mathbf{A}\mathbf{Q}
    $$

    이다($\mathbf{A}^T = \mathbf{A}$를 썼다). 곧 대칭이다.

    연습문제 7과의 차이는 **$\mathbf{P}$에 건 제약** 하나뿐이다. 일반적인 가역행렬에서는 $\mathbf{P}^{-1} \neq \mathbf{P}^T$이므로 위 계산의 마지막 단계가 성립하지 않는다.

    이것이 **직교닮음**을 따로 구분하는 이유다. 스펙트럼 정리가 대칭행렬에 대해 $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$를 보장할 때, 대각화가 하필 직교행렬로 이루어진다는 점이 결정적이다. 그 덕분에 공분산행렬을 대각화해도 대칭성과 양정치성이 함께 보존된다. $\square$

<div class="drillbox" markdown>

**연습문제 9.**
무작위로 뽑은 가역행렬 $\mathbf{P}$로 닮음변환을 만들어, 성분은 완전히 달라지지만 고윳값·대각합·행렬식·계수는 보존됨을 수치로 확인하라.

</div>

??? success "풀이"
    ```python
    import numpy as np

    rng = np.random.default_rng(0)
    A = np.array([[4., 1., 0.],
                  [2., 3., 1.],
                  [0., 1., 5.]])

    P = rng.normal(size=(3, 3))
    print("P가 가역인가 (det):", round(np.linalg.det(P), 4))

    B = np.linalg.inv(P) @ A @ P

    print("\nA =\n", np.round(A, 3))
    print("B =\n", np.round(B, 3))

    print("\n고윳값 A:", np.sort(np.linalg.eigvals(A).real).round(6))
    print("고윳값 B:", np.sort(np.linalg.eigvals(B).real).round(6))
    print("대각합 :", round(np.trace(A), 6), round(np.trace(B), 6))
    print("행렬식 :", round(np.linalg.det(A), 6), round(np.linalg.det(B), 6))
    print("계수   :", np.linalg.matrix_rank(A), np.linalg.matrix_rank(B))
    ```

    출력:

    ```
    P가 가역인가 (det): 0.4433

    A =
     [[4. 1. 0.]
     [2. 3. 1.]
     [0. 1. 5.]]
    B =
     [[ 6.627  1.644  0.05 ]
     [-2.824  0.822 -0.021]
     [-0.935 -1.815  4.55 ]]

    고윳값 A: [1.78568  4.539189 5.675131]
    고윳값 B: [1.78568  4.539189 5.675131]
    대각합 : 12.0 12.0
    행렬식 : 46.0 46.0
    계수   : 3 3
    ```

    성분은 서로 아무 관련이 없어 보이지만 네 가지 불변량은 소수점 아래까지 일치한다.

    수치적으로 한 가지 주의할 점이 있다. $\mathbf{P}$가 특이행렬에 가까우면 $\mathbf{P}^{-1}$의 성분이 커져 반올림 오차가 증폭된다. 그래서 실무에서는 **직교행렬**을 기저변환에 쓴다. $\mathbf{Q}^{-1} = \mathbf{Q}^T$이므로 역행렬을 계산할 필요조차 없고 수치적으로도 안정하다. $\square$

<div class="drillbox" markdown>

**연습문제 10.**
주성분분석은 공분산행렬 $\boldsymbol{\Sigma}$를 직교행렬 $\mathbf{Q}$로 대각화한다: $\boldsymbol{\Lambda} = \mathbf{Q}^T\boldsymbol{\Sigma}\mathbf{Q}$. 이때 **총분산**이 보존되는 이유를 닮음 불변량으로 설명하고 수치로 확인하라.

</div>

??? success "풀이"
    총분산은 각 변수의 분산을 모두 더한 값, 곧 $\operatorname{tr}(\boldsymbol{\Sigma})$로 정의된다. $\boldsymbol{\Lambda}$는 $\boldsymbol{\Sigma}$와 닮았고 대각합은 닮음 불변량이므로

    $$
    \operatorname{tr}(\boldsymbol{\Sigma}) = \operatorname{tr}(\boldsymbol{\Lambda}) = \sum_{i=1}^{p} \lambda_i
    $$

    이다. 곧 **주성분의 분산을 모두 더하면 원래 변수들의 분산 총합과 같다.** "첫 두 주성분이 총분산의 $85\%$를 설명한다"는 익숙한 표현이 성립하는 근거가 바로 이 등식이다.

    ```python
    import numpy as np

    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 3)) @ np.array([[2., 1., 0.],
                                              [0., 1., 1.],
                                              [0., 0., 3.]])
    Sigma = np.cov(X, rowvar=False)

    lam, Q = np.linalg.eigh(Sigma)          # 대칭행렬이므로 eigh
    lam = lam[::-1]                          # 큰 것부터

    print("각 변수의 분산:", np.diag(Sigma).round(4))
    print("총분산 tr(Sigma):", round(np.trace(Sigma), 6))
    print("고윳값의 합    :", round(lam.sum(), 6))
    print("설명 비율(%)   :", (lam / lam.sum() * 100).round(1))
    print("첫 두 성분 누적(%):", round(lam[:2].sum() / lam.sum() * 100, 1))
    ```

    출력:

    ```
    각 변수의 분산: [ 4.0421  2.0308 10.0525]
    총분산 tr(Sigma): 16.125337
    고윳값의 합    : 16.125337
    설명 비율(%)   : [62.9 32.7  4.3]
    첫 두 성분 누적(%): 95.7
    ```

    대각합은 같지만 **대각 성분의 분포는 완전히 달라진다.** 원래 변수들은 분산을 고만고만하게 나눠 갖는 반면, 주성분은 앞쪽에 몰아준다. 회전이 하는 일이 바로 이것이다. 총량은 그대로 두고 **배분만 바꾼다.**

    행렬식도 불변이므로 $\det(\boldsymbol{\Sigma}) = \prod_i \lambda_i$가 성립한다. 이 값을 **일반화 분산**이라 하며, 다변량 정규분포의 밀도에 그대로 등장한다. $\square$

---

## 정리하며

두 행렬이 서로 다른 기저에서 같은 선형변환을 나타낼 때 이 둘은 닮았다. 닮은 행렬은 좌표계의 선택에서만 다를 뿐, 고윳값·대각합·행렬식·계수·특성다항식 등 변환의 본질적 성질을 모두 공유한다. 다음 주제인 대각화가 가장 중요한 특수한 경우다. 행렬이 대각이 되는 기저를 찾아 계산과 해석을 단순하게 만드는 것이다.
