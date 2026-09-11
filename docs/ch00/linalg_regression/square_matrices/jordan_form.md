# 조르당 표준형

모든 행렬이 대각화되는 것은 아니다. 어떤 행렬이 일차독립인 고유벡터를 온전히 갖추지 못하면, 가능한 최선의 닮음 축약이 **조르당 표준형(Jordan canonical form)**(조르당 정규형이라고도 한다)이다. 조르당 형은 대각행렬 $\boldsymbol{\Lambda}$를 거의 대각인 행렬 — 주대각에 고윳값이 있고 그 바로 위 대각에 1(또는 0)이 있는 행렬 — 로 대체한다. 통계학자에게 조르당 형은 일상적인 자료 분석에 좀처럼 등장하지 않지만(공분산행렬은 대칭이라 언제나 대각화 가능하므로), 특정 행렬 구조가 왜 다르게 행동하는지를 이해하는 이론적 토대를 제공하고 닮음에 의한 정사각행렬의 분류를 완성한다.

## 조르당 블록

!!! info "정의 — 조르당 블록"
    고윳값 $\lambda$에 대응하는 크기 $k$의 **조르당 블록**은 다음 $k \times k$ 상삼각행렬이다.

    $$
    \mathbf{J}_k(\lambda) = \begin{pmatrix} \lambda & 1 & 0 & \cdots & 0 \\ 0 & \lambda & 1 & \cdots & 0 \\ \vdots & & \ddots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda & 1 \\ 0 & 0 & \cdots & 0 & \lambda \end{pmatrix}
    $$

    대각 성분은 모두 $\lambda$, 바로 위 대각 성분은 모두 1이고 나머지는 모두 0이다.

$1 \times 1$ 조르당 블록 $\mathbf{J}_1(\lambda) = (\lambda)$는 단순한 스칼라다. 모든 조르당 블록이 $1 \times 1$이면 조르당 형은 대각이고 그 행렬은 대각화 가능하다.

## 조르당 표준형 정리

<div class="thmbox" markdown>

### 정리 1. 조르당 표준형 { .thm }

모든 정사각행렬 $\mathbf{A} \in \mathbb{R}^{n \times n}$(또는 $\mathbb{C}^{n \times n}$)은 **조르당 행렬**

$$
\mathbf{J} = \begin{pmatrix} \mathbf{J}_{k_1}(\lambda_1) & & \\ & \ddots & \\ & & \mathbf{J}_{k_r}(\lambda_r) \end{pmatrix}
$$

과 닮았다. 여기서 $k_1 + k_2 + \cdots + k_r = n$이다. 즉 $\mathbf{A} = \mathbf{P}\mathbf{J}\mathbf{P}^{-1}$인 가역행렬 $\mathbf{P}$가 존재한다. 조르당 형은 블록의 순서를 제외하면 유일하다.

</div>

고윳값 $\lambda_1, \dots, \lambda_r$이 서로 달라야 할 필요는 없다. 같은 고윳값이 크기가 다른 여러 블록에 나타날 수 있다.

## 대각화 가능성과의 관계

조르당 형은 대각화 가능성을 깔끔하게 특성화한다.

- $\mathbf{A}$가 대각화 가능할 필요충분조건은 모든 조르당 블록이 $1 \times 1$인 것이다.
- 동등하게, $\mathbf{A}$가 대각화 가능할 필요충분조건은 각 고윳값 $\lambda$에 대해 기하적 중복도($\ker(\mathbf{A} - \lambda\mathbf{I})$의 차원)가 대수적 중복도(특성다항식의 근으로서 $\lambda$의 중복도)와 같은 것이다.
- 기하적 중복도가 대수적 중복도보다 엄격히 작으면, 그 고윳값에 대한 조르당 블록 중 적어도 하나는 크기가 1보다 크다.

## 조르당 블록의 거듭제곱

조르당 블록의 거듭제곱을 계산해 보면 대각화되지 않는 행렬이 왜 다르게 행동하는지 드러난다. 조르당 블록 $\mathbf{J}_k(\lambda)$에 대해

$$
[\mathbf{J}_k(\lambda)^m]_{ij} = \begin{cases} \binom{m}{j-i}\lambda^{m-(j-i)} & \text{if } j \geq i \text{ and } j - i \leq m \\ 0 & \text{otherwise} \end{cases}
$$

이다. 특히 $|\lambda| < 1$이면 $m \to \infty$일 때 거듭제곱 $\mathbf{J}_k(\lambda)^m \to \mathbf{0}$이지만 수렴 속도가 블록 크기 $k$에 달려 있다. $|\lambda| = 1$이고 $k > 1$이면 (이항계수 $\binom{m}{j-i}$가 커지므로) 거듭제곱이 다항식적으로 증가하며, 이는 유계로 남는 대각인 경우와 다르다.

## 예

앞 절에 나온 대각화 불가능한 행렬을 생각하자.

$$
\mathbf{A} = \begin{pmatrix} 2 & 1 \\ 0 & 2 \end{pmatrix}
$$

이 행렬은 대수적 중복도 2, 기하적 중복도 1인 고윳값 $\lambda = 2$를 갖는다. 이 행렬은 이미 조르당 형이다. 하나의 $2 \times 2$ 조르당 블록 $\mathbf{J}_2(2)$이다.

거듭제곱은

$$
\mathbf{A}^m = \begin{pmatrix} 2^m & m \cdot 2^{m-1} \\ 0 & 2^m \end{pmatrix}
$$

이다. $(1,2)$ 성분에서 $2^{m-1}$에 곱해진 다항식 인자 $m$에 주목하라. 고윳값이 2인 대각행렬이라면 $\mathbf{D}^m = \operatorname{diag}(2^m, 2^m)$으로 다항식적 증가 없이 순수한 지수적 행동만 보였을 것이다.

## 더 큰 예

다음을 생각하자.

$$
\mathbf{A} = \begin{pmatrix} 3 & 1 & 0 \\ 0 & 3 & 0 \\ 0 & 0 & 5 \end{pmatrix}
$$

고윳값은 $\lambda_1 = 3$(대수적 중복도 2)과 $\lambda_2 = 5$(대수적 중복도 1)이다. $\lambda_1 = 3$의 고유공간은 $\ker(\mathbf{A} - 3\mathbf{I}) = \operatorname{span}\{(1, 0, 0)^T\}$로 기하적 중복도가 1이며(대수적 중복도 2보다 작다), 조르당 형은

$$
\mathbf{J} = \begin{pmatrix} 3 & 1 & 0 \\ 0 & 3 & 0 \\ 0 & 0 & 5 \end{pmatrix} = \begin{pmatrix} \mathbf{J}_2(3) & \\ & \mathbf{J}_1(5) \end{pmatrix}
$$

이다. 이 경우 $\mathbf{A}$는 이미 조르당 형이다. 조르당 형은 고윳값 3에 대한 $2 \times 2$ 블록 하나와 고윳값 5에 대한 $1 \times 1$ 블록 하나를 갖는다.

## 일반화 고유벡터

$\mathbf{A} = \mathbf{P}\mathbf{J}\mathbf{P}^{-1}$에서 기저변환행렬 $\mathbf{P}$의 열을 **일반화 고유벡터**라 한다. 조르당 블록 $\mathbf{J}_k(\lambda)$에 대응하는 일반화 고유벡터 $\mathbf{v}_1, \dots, \mathbf{v}_k$는 다음을 만족한다.

$$
(\mathbf{A} - \lambda\mathbf{I})\mathbf{v}_1 = \mathbf{0}, \quad (\mathbf{A} - \lambda\mathbf{I})\mathbf{v}_2 = \mathbf{v}_1, \quad \dots, \quad (\mathbf{A} - \lambda\mathbf{I})\mathbf{v}_k = \mathbf{v}_{k-1}
$$

첫 번째 벡터 $\mathbf{v}_1$은 보통의 고유벡터이고, 나머지 $\mathbf{v}_2, \dots, \mathbf{v}_k$는 **조르당 사슬**을 이루는 일반화 고유벡터다.

## 통계와의 연결

응용통계에서 마주치는 행렬 — 공분산행렬, 모자 행렬, 잔차생성행렬 — 이 모두 대칭이므로 조르당 형 자체는 좀처럼 등장하지 않지만, 이 이론은 간접적으로 중요하다.

- **닮음 이론의 완결성.** 조르당 형은 모든 정사각행렬이 본질적으로 유일한 "표준" 행렬과 닮았음을 보장한다. 이는 대각합, 행렬식, 고윳값 같은 행렬의 성질이 닮음류를 완전히 특성화한다는 일반적 주장을 정당화한다.

- **안정성 분석.** VAR(벡터자기회귀) 과정 같은 시계열 모형에서 동반행렬의 고윳값이 정상성을 결정한다. 조르당 형은 경계에서 무슨 일이 일어나는지 밝혀준다. $1 \times 1$보다 큰 조르당 블록을 갖는 단위근 고윳값은 일정한 수준이 아니라 다항식 추세를 만들어낸다.

- **행렬 함수.** 공식 $f(\mathbf{A}) = \mathbf{P}f(\mathbf{J})\mathbf{P}^{-1}$은 $\mathbf{A}$가 대각화 불가능할 때에도 스칼라 함수를 행렬에 적용한다는 개념을 확장해 준다. $f(\mathbf{J}_k(\lambda))$를 계산하려면 $\lambda$에서 평가한 $f$의 $k-1$계까지의 도함수가 필요하다.

## 요약

조르당 표준형은 정사각행렬에 대한 가장 일반적인 닮음 축약이다. 모든 행렬은 조르당 블록들로 이루어진 블록대각행렬과 닮았다. 모든 블록이 $1 \times 1$이면 그 행렬은 대각화 가능하고, 그렇지 않으면 더 큰 블록의 초대각 1들이 고유벡터 개수의 "부족분"을 포착한다. 통계를 지배하는 대칭행렬(공분산행렬, 사영행렬)에서는 조르당 형이 언제나 대각으로 환원되지만, 조르당 이론은 이론적 그림을 완성하며 시계열 동반행렬처럼 대칭이 아닌 상황에서 필요하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
특성다항식이 $(\lambda - 2)^2(\lambda - 5)$이고 고윳값 $\lambda = 2$의 기하적 중복도가 1인 $3 \times 3$ 행렬의 조르당 표준형을 써라.

</div>

??? success "풀이"
    $\lambda = 2$는 대수적 중복도가 2이지만 기하적 중복도가 1이므로 하나의 $2 \times 2$ 조르당 블록을 만든다. 고윳값 $\lambda = 5$는 대수적·기하적 중복도가 모두 1이므로 $1 \times 1$ 블록이 된다. 조르당 형은

    $$
    \mathbf{J} = \begin{pmatrix} 2 & 1 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 5 \end{pmatrix}
    $$

    이다. $(1,2)$ 자리의 초대각 1이 부족분을 나타낸다. $\lambda = 2$에 대해 고유벡터가 하나 모자란 것이다.

<div class="drillbox" markdown>

**연습문제 2.**
조르당 블록의 거듭제곱 공식을 이용해 조르당 블록 $\mathbf{J}_2(3) = \begin{pmatrix} 3 & 1 \\ 0 & 3 \end{pmatrix}$에 대해 $\mathbf{J}^3$을 계산하라.

</div>

??? success "풀이"
    $2 \times 2$ 조르당 블록 $\mathbf{J}_2(\lambda)$에 대해 거듭제곱 공식은

    $$
    \mathbf{J}_2(\lambda)^k = \begin{pmatrix} \lambda^k & k\lambda^{k-1} \\ 0 & \lambda^k \end{pmatrix}
    $$

    이다. $\lambda = 3$, $k = 3$이면

    $$
    \mathbf{J}_2(3)^3 = \begin{pmatrix} 27 & 3 \cdot 9 \\ 0 & 27 \end{pmatrix} = \begin{pmatrix} 27 & 27 \\ 0 & 27 \end{pmatrix}
    $$

    직접 곱해서 확인할 수 있다. $\mathbf{J}^2 = \begin{pmatrix} 9 & 6 \\ 0 & 9 \end{pmatrix}$이고, 그다음 $\mathbf{J}^3 = \mathbf{J}^2 \cdot \mathbf{J} = \begin{pmatrix} 27 & 27 \\ 0 & 27 \end{pmatrix}$이다.

<div class="drillbox" markdown>

**연습문제 3.**
어떤 행렬이 대각화 가능할 필요충분조건이 그 조르당 형의 모든 조르당 블록이 $1 \times 1$인 것임을 증명하라.

</div>

??? success "풀이"
    ($\Rightarrow$) $\mathbf{A}$가 대각화 가능하면 $\boldsymbol{\Lambda}$가 대각인 $\mathbf{A} = \mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^{-1}$이 성립한다. 대각행렬은 각 블록이 $1 \times 1$인(초대각의 1이 없는) 조르당 형이다.

    ($\Leftarrow$) 모든 조르당 블록이 $1 \times 1$이면 조르당 형은 $\mathbf{J} = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$으로 대각행렬이다. $\mathbf{J}$가 대각인 $\mathbf{A} = \mathbf{P}\mathbf{J}\mathbf{P}^{-1}$이므로 $\mathbf{A}$는 대각화 가능하다.

    조르당 블록이 모두 $1 \times 1$인 것은 정확히 모든 고윳값에서 기하적 중복도가 대수적 중복도와 같을 때다. $\square$

<div class="drillbox" markdown>

**연습문제 4.**
고윳값이 $\lambda_1 = 1$, $\lambda_2 = 2$, $\lambda_3 = 3$(모두 서로 다름)인 $3 \times 3$ 행렬이 다른 성질과 무관하게 반드시 대각화 가능한 이유를 설명하라.

</div>

??? success "풀이"
    서로 다른 고윳값에 대응하는 고유벡터는 언제나 일차독립이다. 이 행렬은 $3 \times 3$이고 서로 다른 고윳값을 3개 가지므로 일차독립인 고유벡터를 3개 갖는다.

    일차독립인 고유벡터가 $n = 3$개이므로 이들로 만든 행렬 $\mathbf{P}$는 가역이고 $\mathbf{A} = \mathbf{P}\operatorname{diag}(1, 2, 3)\mathbf{P}^{-1}$이다.

    동등하게, 각 고윳값의 대수적 중복도가 1이므로 기하적 중복도도 1이다(기하적 중복도는 언제나 1 이상이고 대수적 중복도 이하이므로). 따라서 모든 조르당 블록이 $1 \times 1$이고 이 행렬은 대각화 가능하다.

<div class="drillbox" markdown>

**연습문제 5.**
VAR(1) 모형 $\mathbf{y}_t = \mathbf{A}\mathbf{y}_{t-1} + \boldsymbol{\varepsilon}_t$에서 $\mathbf{A}$의 모든 고윳값이 $|\lambda_i| < 1$을 만족하면 그 과정은 정상적이다. $\mathbf{A}$가 $2 \times 2$ 조르당 블록을 갖는 단위 고윳값($|\lambda| = 1$)을 가질 때 조르당 형이 무엇을 드러내는지 설명하라.

</div>

??? success "풀이"
    $\lambda = 1$이 $2 \times 2$ 조르당 블록을 가지면 $\mathbf{J}_2(1) = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$이다. 거듭제곱 공식에 의해

    $$
    \mathbf{J}_2(1)^k = \begin{pmatrix} 1 & k \\ 0 & 1 \end{pmatrix}
    $$

    이다. 비대각 성분이 $k$에 따라 선형으로 증가하므로 충격반응이 감쇠하지 않고 한없이 커진다. 이 과정은 일정한 수준이 아니라 선형(다항식) 추세를 보인다.

    이에 비해 $1 \times 1$ 블록을 갖는 단위 고윳값은 $\lambda^k = 1$이 되어 결정론적 추세 없는 단위근(확률보행) 행동을 만든다. 이렇게 조르당 형은 서로 다른 유형의 비정상성 — 단위근(확률보행)과 결정론적 추세 — 을 구별해 준다.

<div class="drillbox" markdown>

**연습문제 6.**
고윳값 $\lambda$에 대한 조르당 블록의 **개수**는 기하적 중복도와 같고, 블록의 **크기**는 $\operatorname{rank}(\mathbf{A}-\lambda\mathbf{I})^j$가 더 이상 줄지 않을 때까지의 변화로 알 수 있다. 본문의 $\mathbf{A} = \begin{pmatrix} 3 & 1 & 0 \\ 0 & 3 & 0 \\ 0 & 0 & 5 \end{pmatrix}$로 확인하라.

</div>

??? success "풀이"
    블록 개수가 기하적 중복도인 이유는 간단하다. 각 조르당 블록은 보통의 고유벡터를 **정확히 하나씩** 기여하므로(사슬의 첫 벡터), 블록의 개수가 곧 일차독립인 고유벡터의 개수다.

    크기는 계수의 변화로 읽는다. $\lambda$에 대한 크기 $\ge j$인 블록의 개수는

    $$
    \operatorname{rank}(\mathbf{A}-\lambda\mathbf{I})^{j-1} - \operatorname{rank}(\mathbf{A}-\lambda\mathbf{I})^{j}
    $$

    이다.

    ```python
    import numpy as np

    A = np.array([[3., 1., 0.],
                  [0., 3., 0.],
                  [0., 0., 5.]])
    N = A - 3 * np.eye(3)

    ranks = [np.linalg.matrix_rank(np.linalg.matrix_power(N, j)) for j in range(4)]
    print("rank (A-3I)^j, j=0..3:", ranks)
    for j in range(1, 4):
        print(f"  크기 >= {j} 인 블록 개수 = {ranks[j-1] - ranks[j]}")
    ```

    출력:

    ```
    rank (A-3I)^j, j=0..3: [3, 2, 1, 1]
      크기 >= 1 인 블록 개수 = 1
      크기 >= 2 인 블록 개수 = 1
      크기 >= 3 인 블록 개수 = 0
    ```

    $\operatorname{rank}$가 $3 \to 2 \to 1 \to 1$로 변한다. 크기 $\ge 1$인 블록이 $3-2=1$개, 크기 $\ge 2$인 블록이 $2-1=1$개, 크기 $\ge 3$인 블록이 $1-1=0$개다. 곧 $\lambda = 3$에 대해 **크기 2짜리 블록 하나**가 있다.

    이 계수 계산은 조르당 형을 직접 구하지 않고도 블록 구조를 알아내는 표준적인 방법이다. $\square$

<div class="drillbox" markdown>

**연습문제 7.**
$\mathbf{N} = \mathbf{J}_k(\lambda) - \lambda\mathbf{I}$가 **멱영행렬**임을 보여라. 곧 $\mathbf{N}^k = \mathbf{O}$이고 $\mathbf{N}^{k-1} \neq \mathbf{O}$이다.

</div>

??? success "풀이"
    $\mathbf{N}$은 초대각만 1이고 나머지가 0인 행렬이다. 이 행렬은 기저벡터를 한 칸씩 **밀어내는** 사상이다.

    $$
    \mathbf{N}\mathbf{e}_1 = \mathbf{0}, \quad \mathbf{N}\mathbf{e}_2 = \mathbf{e}_1, \quad \dots, \quad \mathbf{N}\mathbf{e}_k = \mathbf{e}_{k-1}
    $$

    따라서 $\mathbf{N}^j\mathbf{e}_i = \mathbf{e}_{i-j}$($i > j$일 때)이고 $i \le j$이면 $\mathbf{0}$이다. $j = k$이면 모든 기저벡터가 $\mathbf{0}$으로 가므로 $\mathbf{N}^k = \mathbf{O}$이고, $j = k-1$이면 $\mathbf{N}^{k-1}\mathbf{e}_k = \mathbf{e}_1 \neq \mathbf{0}$이다.

    ```python
    import numpy as np

    k = 4
    N = np.diag(np.ones(k - 1), 1)          # 초대각만 1
    for j in range(1, k + 1):
        M = np.linalg.matrix_power(N, j)
        print(f"N^{j} 이 영행렬인가: {np.allclose(M, 0)}   (0 아닌 성분 {int((M!=0).sum())}개)")
    ```

    출력:

    ```
    N^1 이 영행렬인가: False   (0 아닌 성분 3개)
    N^2 이 영행렬인가: False   (0 아닌 성분 2개)
    N^3 이 영행렬인가: False   (0 아닌 성분 1개)
    N^4 이 영행렬인가: True   (0 아닌 성분 0개)
    ```

    $\mathbf{N}^4$에서 처음으로 영행렬이 된다.

    **왜 중요한가.** 조르당 블록은 $\mathbf{J}_k(\lambda) = \lambda\mathbf{I} + \mathbf{N}$으로 **대각 부분과 멱영 부분의 합**으로 쪼개지고, 두 부분이 교환한다. 그래서 이항정리를 쓸 수 있다.

    $$
    \mathbf{J}_k(\lambda)^m = (\lambda\mathbf{I}+\mathbf{N})^m = \sum_{j=0}^{k-1}\binom{m}{j}\lambda^{m-j}\mathbf{N}^j
    $$

    $\mathbf{N}^k = \mathbf{O}$이라 합이 $j = k-1$에서 끊긴다. 본문의 거듭제곱 공식이 바로 이것이다. $\square$

<div class="drillbox" markdown>

**연습문제 8.**
$\mathbf{A} = \begin{pmatrix} 2 & 1 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 2 \end{pmatrix}$와 $\mathbf{B} = 2\mathbf{I}_3$은 특성다항식이 같다. 두 행렬이 닮지 않았음을 **최소다항식**으로 구별하라.

</div>

??? success "풀이"
    두 행렬 모두 특성다항식이 $(\lambda-2)^3$이므로 특성다항식만으로는 구별되지 않는다. 최소다항식은 각 고윳값에 대해 **가장 큰 조르당 블록의 크기**를 지수로 갖는다.

    ```python
    import numpy as np

    A = np.array([[2., 1., 0.], [0., 2., 0.], [0., 0., 2.]])
    B = 2 * np.eye(3)

    for name, M in (("A", A), ("B", B)):
        N = M - 2 * np.eye(3)
        k = 1
        while not np.allclose(np.linalg.matrix_power(N, k), 0):
            k += 1
        print(f"{name}: (M - 2I)^{k} = O  ->  최소다항식 (x-2)^{k}")
        print(f"   기하적 중복도 = {3 - np.linalg.matrix_rank(N)}")
    ```

    출력:

    ```
    A: (M - 2I)^2 = O  ->  최소다항식 (x-2)^2
       기하적 중복도 = 2
    B: (M - 2I)^1 = O  ->  최소다항식 (x-2)^1
       기하적 중복도 = 3
    ```

    $\mathbf{A}$는 $(\lambda-2)^2$, $\mathbf{B}$는 $(\lambda-2)^1$이다. 최소다항식이 다르므로 두 행렬은 닮지 않았다(최소다항식은 닮음 불변량이다).

    조르당 형으로 보면 명백하다. $\mathbf{A}$는 블록이 $\{2, 1\}$ 크기로 둘이고 $\mathbf{B}$는 $\{1,1,1\}$로 셋이다. 기하적 중복도가 각각 2와 3으로 다르다.

    **교훈.** 대각합·행렬식·특성다항식이 모두 같아도 닮았다는 보장은 없다. 닮음류를 완전히 결정하는 것은 **조르당 구조**다. $\square$

<div class="drillbox" markdown>

**연습문제 9.**
조르당 블록의 행렬 지수함수 $e^{\mathbf{J}_2(\lambda)}$를 구하고, 미분방정식 $\mathbf{y}' = \mathbf{A}\mathbf{y}$의 해가 대각화 불가능한 경우에 어떤 모양이 되는지 설명하라.

</div>

??? success "풀이"
    $\mathbf{J}_2(\lambda) = \lambda\mathbf{I} + \mathbf{N}$이고 두 항이 교환하며 $\mathbf{N}^2 = \mathbf{O}$이므로

    $$
    e^{\mathbf{J}_2(\lambda)t} = e^{\lambda t}e^{\mathbf{N}t}
    = e^{\lambda t}(\mathbf{I} + t\mathbf{N})
    = e^{\lambda t}\begin{pmatrix} 1 & t \\ 0 & 1 \end{pmatrix}
    $$

    이다($e^{\mathbf{N}t}$의 급수가 두 항에서 끊긴다).

    ```python
    import numpy as np
    from scipy.linalg import expm

    J = np.array([[2., 1.], [0., 2.]])
    print("expm(J) =\n", expm(J).round(6))
    print("e^2 =", round(np.exp(2), 6), "  t=1 이므로 (1,2) 성분도 e^2")
    ```

    출력:

    ```
    expm(J) =
     [[7.389056 7.389056]
     [0.       7.389056]]
    e^2 = 7.389056   t=1 이므로 (1,2) 성분도 e^2
    ```

    **미분방정식에서의 의미.** 해는 $\mathbf{y}(t) = e^{\mathbf{A}t}\mathbf{y}(0)$이다. 대각화 가능하면 성분이 모두 $e^{\lambda_i t}$의 결합이지만, 크기 $k$인 조르당 블록이 있으면

    $$
    t^{j}e^{\lambda t}, \qquad j = 0, 1, \dots, k-1
    $$

    꼴의 항이 나타난다. **다항식이 지수함수에 곱해진다.**

    물리적으로는 임계감쇠(critically damped) 진동이 이런 모양이다. $\lambda$가 중근일 때 해가 $e^{\lambda t}$와 $te^{\lambda t}$의 결합이 되는 것이 바로 조르당 블록 때문이다. $\square$

<div class="drillbox" markdown>

**연습문제 10.**
$\mathbf{C} = \begin{pmatrix} 5 & 1 \\ -1 & 3 \end{pmatrix}$의 조르당 형과 조르당 사슬(일반화 고유벡터)을 구하라.

</div>

??? success "풀이"
    **고윳값.** $\det(\mathbf{C}-\lambda\mathbf{I}) = (5-\lambda)(3-\lambda)+1 = \lambda^2-8\lambda+16 = (\lambda-4)^2$이므로 $\lambda = 4$가 중근이다.

    **기하적 중복도.** $\mathbf{C}-4\mathbf{I} = \begin{pmatrix} 1 & 1 \\ -1 & -1 \end{pmatrix}$의 계수가 1이므로 고유공간이 1차원이다. 대수적 중복도 2보다 작으므로 **대각화 불가능**이고 조르당 형은 $\mathbf{J}_2(4)$다.

    **사슬.** 먼저 보통의 고유벡터 $\mathbf{v}_1$을 구한다. $(\mathbf{C}-4\mathbf{I})\mathbf{v} = \mathbf{0}$에서 $v_1 + v_2 = 0$이므로 $\mathbf{v}_1 = (1,-1)^T$로 잡는다. 다음으로 $(\mathbf{C}-4\mathbf{I})\mathbf{v}_2 = \mathbf{v}_1$을 푼다.

    $$
    \begin{pmatrix} 1 & 1 \\ -1 & -1 \end{pmatrix}\mathbf{v}_2 = \begin{pmatrix} 1 \\ -1 \end{pmatrix}
    \;\Longrightarrow\; v_{21}+v_{22} = 1
    $$

    이므로 예컨대 $\mathbf{v}_2 = (1, 0)^T$다. $\mathbf{P} = (\mathbf{v}_1 \mid \mathbf{v}_2)$로 두면 $\mathbf{P}^{-1}\mathbf{C}\mathbf{P} = \mathbf{J}_2(4)$다.

    ```python
    import sympy as sp

    C = sp.Matrix([[5, 1], [-1, 3]])
    x = sp.symbols('lambda')
    print("특성다항식:", sp.factor(C.charpoly(x).as_expr()))

    P, J = C.jordan_form()
    print("J =", J.tolist())
    print("P =", P.tolist())
    print("P J P^-1 = C 인가:", sp.simplify(P * J * P.inv() - C) == sp.zeros(2, 2))
    ```

    출력:

    ```
    특성다항식: (lambda - 4)**2
    J = [[4, 1], [0, 4]]
    P = [[1, 1], [-1, 0]]
    P J P^-1 = C 인가: True
    ```

    `sympy`가 돌려준 $\mathbf{P}$는 위에서 손으로 고른 것과 열이 다를 수 있다. 사슬의 두 번째 벡터는 $\mathbf{v}_1$의 상수배만큼 자유롭기 때문이며, 어느 쪽을 골라도 같은 $\mathbf{J}$를 준다.

    **주의.** 조르당 형은 수치적으로 **불안정**하다. 성분을 아주 조금만 흔들어도 중근이 서로 다른 두 근으로 갈라져 대각화 가능해진다. 그래서 실제 계산에서는 조르당 형 대신 슈어 분해나 특이값분해를 쓴다. 조르당 형은 이론을 설명하는 도구다. $\square$

