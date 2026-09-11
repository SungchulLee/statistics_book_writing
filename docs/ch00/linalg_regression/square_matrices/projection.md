# 사영행렬

사영은 $\mathbb{R}^n$의 모든 벡터를 여집합 부분공간 $\mathcal{W}$ 방향을 따라 부분공간 $\mathcal{V}$ 위로 "눌러 붙인다". 대수적으로 사영은 멱등인 선형변환이다. 두 번 적용해도 한 번 적용한 것과 같은 결과가 나온다. 여집합 부분공간이 $\mathcal{V}$의 직교여공간일 때 그 사영을 **직교사영**이라 하며 다음 절에서 다룬다. 이 절에서는 일반적인(빗각) 경우를 다루는데, 직교성 제약을 붙이기 전에 사영을 이해하는 개념적 틀을 제공한다.

## 정의

!!! info "정의 — 사영(일반)"
    정사각행렬 $\mathbf{P} \in \mathbb{R}^{n \times n}$이

    $$
    \mathbf{P}^2 = \mathbf{P}
    $$

    를 만족하면 **사영행렬**(또는 **사영자**)이라 한다. 즉 $\mathbf{P}$는 멱등이다.

사영행렬에는 두 부분공간이 딸려 있다.

- **치역**(열공간) $\mathcal{V} = \operatorname{col}(\mathbf{P})$: 벡터가 사영되어 놓이는 부분공간.
- **영공간** $\mathcal{W} = \ker(\mathbf{P})$: 사영이 눌러 없애는 방향의 부분공간.

## 공간의 분해

<div class="thmbox" markdown>

### 정리 1. 직합 분해 { .thm }

$\mathbf{P}$가 사영이면 $\mathbb{R}^n = \operatorname{col}(\mathbf{P}) \oplus \ker(\mathbf{P})$이고, 모든 $\mathbf{x} \in \mathbb{R}^n$에 대해

$$
\mathbf{x} = \mathbf{P}\mathbf{x} + (\mathbf{I} - \mathbf{P})\mathbf{x}
$$

이며, 여기서 $\mathbf{P}\mathbf{x} \in \operatorname{col}(\mathbf{P})$이고 $(\mathbf{I} - \mathbf{P})\mathbf{x} \in \ker(\mathbf{P})$이다.

</div>

??? proof "증명"

    분해 $\mathbf{x} = \mathbf{P}\mathbf{x} + (\mathbf{I} - \mathbf{P})\mathbf{x}$는 자명하게 참이다. 주장된 부분공간 소속을 확인한다.

    - 정의에 의해 $\mathbf{P}\mathbf{x} \in \operatorname{col}(\mathbf{P})$이다.
    - $\mathbf{P}\bigl((\mathbf{I} - \mathbf{P})\mathbf{x}\bigr) = (\mathbf{P} - \mathbf{P}^2)\mathbf{x} = (\mathbf{P} - \mathbf{P})\mathbf{x} = \mathbf{0}$이므로 $(\mathbf{I} - \mathbf{P})\mathbf{x} \in \ker(\mathbf{P})$이다.

    합이 직합임을 보이려면, $\mathbf{v} \in \operatorname{col}(\mathbf{P}) \cap \ker(\mathbf{P})$라 하자. 그러면 어떤 $\mathbf{u}$에 대해 $\mathbf{v} = \mathbf{P}\mathbf{u}$이고 $\mathbf{P}\mathbf{v} = \mathbf{0}$이다. 그런데 $\mathbf{P}\mathbf{v} = \mathbf{P}^2\mathbf{u} = \mathbf{P}\mathbf{u} = \mathbf{v}$이므로 $\mathbf{v} = \mathbf{0}$이다. $\square$

## 여집합 사영

$\mathbf{P}$가 멱등이므로 여집합 행렬 $\mathbf{I} - \mathbf{P}$도 멱등이다.

$$
(\mathbf{I} - \mathbf{P})^2 = \mathbf{I} - 2\mathbf{P} + \mathbf{P}^2 = \mathbf{I} - \mathbf{P}
$$

여집합 사영 $\mathbf{I} - \mathbf{P}$는 $\operatorname{col}(\mathbf{P})$ 방향을 따라 $\ker(\mathbf{P})$ 위로 사영한다. 두 사영이 함께 임의의 벡터를 서로 여집합인 두 부분공간의 성분으로 분해한다.

**계수 관계:**

$$
\operatorname{rank}(\mathbf{P}) + \operatorname{rank}(\mathbf{I} - \mathbf{P}) = n
$$

## 고윳값과 대각합

모든 사영이 멱등이므로 그 고윳값은 $\{0, 1\}$로 제한되고

$$
\operatorname{tr}(\mathbf{P}) = \operatorname{rank}(\mathbf{P}) = \dim(\operatorname{col}(\mathbf{P}))
$$

이다. 대각합은 $\mathbf{P}$가 사영하는 부분공간의 차원을 센다.

## 빗각 사영 대 직교사영

사영 $\mathbf{P}$가 $\mathbf{P} = \mathbf{P}^T$이면(사영이 대칭이면) **직교사영**이라 하고, 그렇지 않으면 **빗각 사영**이라 한다.

| 성질 | 직교사영 | 빗각 사영 |
|---|---|---|
| $\mathbf{P}^2 = \mathbf{P}$ | 예 | 예 |
| $\mathbf{P}^T = \mathbf{P}$ | 예 | 아니오 |
| $\ker(\mathbf{P}) \perp \operatorname{col}(\mathbf{P})$ | 예 | 아니오 |
| $\lVert\mathbf{x} - \mathbf{P}\mathbf{x}\rVert$을 최소화 | 예 | (일반적으로) 아니오 |

통계에서 자연스럽게 등장하는 사영은 거의 모두 직교사영이다(모자 행렬, 잔차생성행렬, 중심화행렬). 빗각 사영은 도구변수 추정이나 일반화 최소제곱에서 나타날 수 있다.

## 예 — 빗각 사영

$\mathbb{R}^2$에서 $\mathcal{W} = \operatorname{span}\{(1, 1)^T\}$ 방향을 따라 $\mathcal{V} = \operatorname{span}\{(1, 0)^T\}$ 위로 사영하는 경우를 생각하자.

임의의 벡터 $\mathbf{x} = (x_1, x_2)^T$는 $\alpha = x_1 - x_2$, $\beta = x_2$에 대해 $\mathbf{x} = \alpha(1, 0)^T + \beta(1, 1)^T$로 분해된다. $\mathcal{W}$ 방향을 따라 $\mathcal{V}$ 위로 사영하면 $\mathcal{V}$ 성분만 남는다.

$$
\mathbf{P}\mathbf{x} = \alpha\begin{pmatrix}1 \\ 0\end{pmatrix} = \begin{pmatrix}x_1 - x_2 \\ 0\end{pmatrix}
$$

행렬 형태로는

$$
\mathbf{P} = \begin{pmatrix} 1 & -1 \\ 0 & 0 \end{pmatrix}
$$

이다.

**확인:** $\mathbf{P}^2 = \begin{pmatrix} 1 & -1 \\ 0 & 0 \end{pmatrix}\begin{pmatrix} 1 & -1 \\ 0 & 0 \end{pmatrix} = \begin{pmatrix} 1 & -1 \\ 0 & 0 \end{pmatrix} = \mathbf{P}$. 이 행렬은 멱등이지만 대칭이 아니므로($\mathbf{P} \neq \mathbf{P}^T$) 빗각 사영이다.

## 예 — 1차원에서의 직교사영

$\mathcal{V}^\perp = \operatorname{span}\{(0, 1)^T\}$ 방향을 따라 $\mathcal{V} = \operatorname{span}\{(1, 0)^T\}$ 위로 사영하는 행렬은

$$
\mathbf{P} = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}
$$

이다. 이것은 멱등이면서 대칭이므로 직교사영이다. 두 번째 성분을 떨어뜨려 $(x_1, x_2)^T$를 $(x_1, 0)^T$로 보낸다.

## 유일성

<div class="thmbox" markdown>

### 정리 2. 사영의 유일성 { .thm }

직합 분해 $\mathbb{R}^n = \mathcal{V} \oplus \mathcal{W}$가 주어지면 $\operatorname{col}(\mathbf{P}) = \mathcal{V}$이고 $\ker(\mathbf{P}) = \mathcal{W}$인 사영 $\mathbf{P}$가 유일하게 존재한다.

</div>

이는 목표 부분공간 $\mathcal{V}$만 지정해서는 사영이 유일하게 결정되지 않음을 뜻한다. 눌러 없애는 방향 $\mathcal{W}$도 함께 지정해야 한다. $\mathcal{W} = \mathcal{V}^\perp$일 때 그 사영은 직교사영이 되고, 이 특수한 경우에는 $\mathcal{V}$만으로 사영이 결정된다.

## 통계와의 연결

### 적합값과 잔차

선형회귀에서 분해 $\mathbf{y} = \mathbf{H}\mathbf{y} + (\mathbf{I} - \mathbf{H})\mathbf{y} = \hat{\mathbf{y}} + \mathbf{e}$는 사영 분해다. 모자 행렬 $\mathbf{H}$는 $\mathbf{X}$의 열공간 위로 사영하고, $\mathbf{I} - \mathbf{H}$는 그 직교여공간(잔차공간) 위로 사영한다.

### 분산분석 분해

분산분석의 총제곱합 분해는 모형 부분공간과 오차 부분공간 위로의 직교사영이 관여하는 이차형식들의 합으로 쓸 수 있다. 직교성 덕분에 (정규성 아래에서) 제곱합들이 서로 독립이 되며, 이것이 F-검정의 근거다.

## 요약

사영행렬은 $\mathbb{R}^n$을 자신의 열공간과 영공간의 직합으로 분해하는 멱등행렬이다. 모든 벡터가 각 부분공간의 성분으로 쪼개지고, 사영을 두 번 적용해도 새로운 일은 일어나지 않는다. 대각합은 목표 부분공간의 차원과 같다. 사영이 대칭이기까지 하면 직교사영이 되며, 회귀와 분산분석에서 가장 자주 등장하는 것이 바로 이 유형이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
$\mathbf{P} = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$이 사영행렬임을 확인하라. 어느 부분공간 위로 사영하는가? 여집합 사영 $\mathbf{I} - \mathbf{P}$는 무엇인가?

</div>

??? success "풀이"
    멱등성 확인:

    $$
    \mathbf{P}^2 = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} = \mathbf{P}
    $$

    $\mathbf{P}$의 열공간은 $\text{span}\{(1, 0)^T\}$이므로 $\mathbf{P}$는 $x_1$축 위로 사영한다. 여집합 사영은

    $$
    \mathbf{I} - \mathbf{P} = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}
    $$

    이며 $x_2$축 위로 사영한다. $\mathbf{P}$는 대칭이기도 하므로 이것은 직교사영이다.

<div class="drillbox" markdown>

**연습문제 2.**
$\mathbf{P}$가 멱등이면 $\operatorname{rank}(\mathbf{P}) = \operatorname{tr}(\mathbf{P})$임을 증명하라.

</div>

??? success "풀이"
    $\mathbf{P}$가 멱등이므로 고윳값은 0 아니면 1이다($\mathbf{P}\mathbf{v} = \lambda\mathbf{v}$이면 $\mathbf{P}^2\mathbf{v} = \lambda^2\mathbf{v} = \lambda\mathbf{v}$이므로 $\lambda^2 = \lambda$이고 $\lambda \in \{0, 1\}$이다).

    계수는 0이 아닌 고윳값의 개수와 같고, 이는 1인 고윳값의 개수다. 대각합은 모든 고윳값의 합인데, 나머지가 0이므로 이 또한 1인 고윳값의 개수다.

    따라서 $\operatorname{rank}(\mathbf{P}) = \operatorname{tr}(\mathbf{P})$이다. $\square$

<div class="drillbox" markdown>

**연습문제 3.**
빗각(직교가 아닌) 사영행렬의 예를 들어라. 멱등이지만 대칭이 아님을 확인하라.

</div>

??? success "풀이"
    $\mathbf{P} = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}$을 생각하자.

    멱등성:

    $$
    \mathbf{P}^2 = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix} = \mathbf{P}
    $$

    그러나 $\mathbf{P}^T = \begin{pmatrix} 1 & 0 \\ 1 & 0 \end{pmatrix} \neq \mathbf{P}$이므로 $\mathbf{P}$는 대칭이 아니다.

    이 행렬은 $(1, 0)^T + \ker(\mathbf{P}) = (1, 0)^T + \text{span}\{(-1, 1)^T\}$ 방향을 따라 $\text{span}\{(1, 0)^T\}$ 위로 사영한다. 사영 방향이 목표 부분공간에 대해 빗각이다(수직이 아니다).

<div class="drillbox" markdown>

**연습문제 4.**
회귀 분해 $\mathbf{y} = \mathbf{H}\mathbf{y} + (\mathbf{I} - \mathbf{H})\mathbf{y}$에서 $\mathbf{H}\mathbf{y}$와 $(\mathbf{I} - \mathbf{H})\mathbf{y}$가 직교인 이유를 설명하라. 멱등성 외에 어떤 성질이 추가로 필요한가?

</div>

??? success "풀이"
    $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$와 $\mathbf{e} = (\mathbf{I} - \mathbf{H})\mathbf{y}$의 직교성에는 $\mathbf{H}$가 (멱등일 뿐 아니라) 대칭이어야 한다. 대칭성이 있으면

    $$
    \hat{\mathbf{y}}^T\mathbf{e} = \mathbf{y}^T\mathbf{H}^T(\mathbf{I} - \mathbf{H})\mathbf{y} = \mathbf{y}^T\mathbf{H}(\mathbf{I} - \mathbf{H})\mathbf{y} = \mathbf{y}^T(\mathbf{H} - \mathbf{H}^2)\mathbf{y} = \mathbf{0}
    $$

    이다. $\mathbf{H}$가 멱등이지만 대칭이 아니라면(빗각 사영이라면) 분해 $\mathbf{y} = \mathbf{H}\mathbf{y} + (\mathbf{I} - \mathbf{H})\mathbf{y}$는 여전히 성립하지만 두 성분이 직교하지는 않는다. 최소제곱의 모자 행렬은 멱등이면서 대칭이며, 그 덕분에 제곱합 분해와 피타고라스 정리가 작동한다.

<div class="drillbox" markdown>

**연습문제 5.**
$\mathcal{V} = \operatorname{col}(\mathbf{V})$ 위로, $\mathcal{W}$ 방향을 따라 사영하는 행렬을 만드는 일반 공식은 $\mathbf{P} = \mathbf{V}(\mathbf{U}^T\mathbf{V})^{-1}\mathbf{U}^T$이다. 여기서 $\mathbf{U}$의 열들은 $\mathcal{W}$의 직교여공간을 편다. 이 $\mathbf{P}$가 사영임을 보이고, 본문의 빗각 사영 예를 이 공식으로 재현하라.

</div>

??? success "풀이"
    **멱등성.** 가운데에서 $\mathbf{U}^T\mathbf{V}$와 그 역행렬이 상쇄된다.

    $$
    \mathbf{P}^2 = \mathbf{V}(\mathbf{U}^T\mathbf{V})^{-1}\underbrace{\mathbf{U}^T\mathbf{V}}_{}(\mathbf{U}^T\mathbf{V})^{-1}\mathbf{U}^T
    = \mathbf{V}(\mathbf{U}^T\mathbf{V})^{-1}\mathbf{U}^T = \mathbf{P}
    $$

    **치역과 영공간.** $\mathbf{P}\mathbf{x}$는 언제나 $\mathbf{V}$의 열들의 일차결합이므로 $\operatorname{col}(\mathbf{P}) \subseteq \mathcal{V}$이다. 또 $\mathbf{P}\mathbf{x} = \mathbf{0}$일 필요충분조건은 $\mathbf{U}^T\mathbf{x} = \mathbf{0}$이므로 $\ker(\mathbf{P}) = \operatorname{col}(\mathbf{U})^\perp = \mathcal{W}$이다.

    본문의 예는 $\mathcal{V} = \operatorname{span}\{(1,0)^T\}$, $\mathcal{W} = \operatorname{span}\{(1,1)^T\}$이었다. $\mathcal{W}^\perp = \operatorname{span}\{(1,-1)^T\}$이므로 $\mathbf{U} = (1,-1)^T$로 둔다.

    ```python
    import numpy as np

    V = np.array([[1.], [0.]])      # 사영해서 놓일 부분공간
    U = np.array([[1.], [-1.]])     # 눌러 없앨 방향 span{(1,1)} 의 직교여공간

    P = V @ np.linalg.inv(U.T @ V) @ U.T
    print("P =\n", P)
    print("멱등인가:", np.allclose(P @ P, P))
    print("대칭인가:", np.allclose(P, P.T))
    print("(1,1) 을 보내면:", (P @ np.array([1., 1.])).round(10))
    ```

    출력:

    ```
    P =
     [[ 1. -1.]
     [ 0.  0.]]
    멱등인가: True
    대칭인가: False
    (1,1) 을 보내면: [0. 0.]
    ```

    본문의 $\mathbf{P} = \begin{pmatrix} 1 & -1 \\ 0 & 0 \end{pmatrix}$이 그대로 나오고, 눌러 없애기로 한 방향 $(1,1)^T$이 실제로 $\mathbf{0}$으로 간다.

    **직교사영은 특수한 경우다.** $\mathcal{W} = \mathcal{V}^\perp$로 두면 $\mathbf{U} = \mathbf{V}$가 되어 $\mathbf{P} = \mathbf{V}(\mathbf{V}^T\mathbf{V})^{-1}\mathbf{V}^T$, 곧 모자 행렬의 꼴이 된다. $\square$

<div class="drillbox" markdown>

**연습문제 6.**
모든 사영행렬은 대각화 가능하며, 적당한 기저에서 $\operatorname{diag}(1, \dots, 1, 0, \dots, 0)$과 닮았음을 보여라($1$이 $r = \operatorname{rank}(\mathbf{P})$개).

</div>

??? success "풀이"
    본문의 직합 분해에서 $\mathbb{R}^n = \operatorname{col}(\mathbf{P}) \oplus \ker(\mathbf{P})$이다.

    - $\mathbf{v} \in \operatorname{col}(\mathbf{P})$이면 $\mathbf{v} = \mathbf{P}\mathbf{u}$이므로 $\mathbf{P}\mathbf{v} = \mathbf{P}^2\mathbf{u} = \mathbf{P}\mathbf{u} = \mathbf{v}$, 곧 **고윳값 1**의 고유벡터다.
    - $\mathbf{w} \in \ker(\mathbf{P})$이면 $\mathbf{P}\mathbf{w} = \mathbf{0}$, 곧 **고윳값 0**의 고유벡터다.

    $\operatorname{col}(\mathbf{P})$의 기저 $r$개와 $\ker(\mathbf{P})$의 기저 $n - r$개를 합치면 $\mathbb{R}^n$ 전체의 기저를 이루고, 그 전부가 고유벡터다. 고유벡터로 이루어진 기저가 존재하므로 $\mathbf{P}$는 대각화 가능하고, 이 기저를 열로 갖는 $\mathbf{S}$에 대해

    $$
    \mathbf{S}^{-1}\mathbf{P}\mathbf{S} = \begin{pmatrix} \mathbf{I}_r & \mathbf{O} \\ \mathbf{O} & \mathbf{O} \end{pmatrix}
    $$

    이다.

    **따름정리.** 대각합이 닮음 불변량이므로 $\operatorname{tr}(\mathbf{P}) = r = \operatorname{rank}(\mathbf{P})$가 곧바로 나온다(연습문제 2의 다른 증명이다).

    **주의.** 사영은 **언제나** 대각화 가능하다. 빗각 사영도 그렇다. 빗각 사영에서 부족한 것은 대각화 가능성이 아니라 **고유벡터들의 직교성**이다. $\square$

<div class="drillbox" markdown>

**연습문제 7.**
$\operatorname{col}(\mathbf{I} - \mathbf{P}) = \ker(\mathbf{P})$이고 $\ker(\mathbf{I} - \mathbf{P}) = \operatorname{col}(\mathbf{P})$임을 보여라. 이로부터 $\operatorname{rank}(\mathbf{P}) + \operatorname{rank}(\mathbf{I} - \mathbf{P}) = n$을 유도하라.

</div>

??? success "풀이"
    **$\operatorname{col}(\mathbf{I} - \mathbf{P}) \subseteq \ker(\mathbf{P})$:** $\mathbf{P}(\mathbf{I} - \mathbf{P})\mathbf{x} = (\mathbf{P} - \mathbf{P}^2)\mathbf{x} = \mathbf{0}$이다.

    **$\ker(\mathbf{P}) \subseteq \operatorname{col}(\mathbf{I} - \mathbf{P})$:** $\mathbf{P}\mathbf{x} = \mathbf{0}$이면 $\mathbf{x} = \mathbf{x} - \mathbf{P}\mathbf{x} = (\mathbf{I} - \mathbf{P})\mathbf{x}$이므로 $\mathbf{x} \in \operatorname{col}(\mathbf{I} - \mathbf{P})$이다.

    두 포함이 함께 등호를 준다. $\mathbf{I} - \mathbf{P}$ 역시 사영이므로 $\mathbf{P}$와 $\mathbf{I} - \mathbf{P}$의 역할을 바꾸면 두 번째 등식도 같은 논법으로 나온다.

    계수-퇴화차수 정리에서

    $$
    \operatorname{rank}(\mathbf{I} - \mathbf{P}) = \dim\ker(\mathbf{P}) = n - \operatorname{rank}(\mathbf{P})
    $$

    이므로 두 계수의 합이 $n$이다.

    ```python
    import numpy as np

    P = np.array([[1., -1.], [0., 0.]])          # 본문의 빗각 사영
    I = np.eye(2)
    print("rank(P) =", np.linalg.matrix_rank(P),
          " rank(I-P) =", np.linalg.matrix_rank(I - P),
          " 합 =", np.linalg.matrix_rank(P) + np.linalg.matrix_rank(I - P))
    print("tr(P) =", np.trace(P), " tr(I-P) =", np.trace(I - P))
    ```

    출력:

    ```
    rank(P) = 1  rank(I-P) = 1  합 = 2
    tr(P) = 1.0  tr(I-P) = 1.0
    ```

    **회귀에서의 의미.** $\operatorname{rank}(\mathbf{H}) = p$이고 $\operatorname{rank}(\mathbf{I} - \mathbf{H}) = n - p$인데, 이 $n - p$가 바로 **잔차의 자유도**다. $s^2 = \lVert\mathbf{e}\rVert^2/(n-p)$에서 $n-p$로 나누는 이유가 여기에 있다. $\square$

<div class="drillbox" markdown>

**연습문제 8.**
$\mathbf{P}_1$과 $\mathbf{P}_2$가 사영일 때 $\mathbf{P}_1 + \mathbf{P}_2$가 사영이 될 필요충분조건이 $\mathbf{P}_1\mathbf{P}_2 = \mathbf{P}_2\mathbf{P}_1 = \mathbf{O}$임을 보여라.

</div>

??? success "풀이"
    전개하면

    $$
    (\mathbf{P}_1 + \mathbf{P}_2)^2 = \mathbf{P}_1^2 + \mathbf{P}_1\mathbf{P}_2 + \mathbf{P}_2\mathbf{P}_1 + \mathbf{P}_2^2
    = (\mathbf{P}_1 + \mathbf{P}_2) + (\mathbf{P}_1\mathbf{P}_2 + \mathbf{P}_2\mathbf{P}_1)
    $$

    이므로 멱등일 필요충분조건은 $\mathbf{P}_1\mathbf{P}_2 + \mathbf{P}_2\mathbf{P}_1 = \mathbf{O}$이다.

    이 조건에서 각각이 $\mathbf{O}$임을 끌어낼 수 있다. 위 식의 왼쪽에 $\mathbf{P}_1$을 곱하면 $\mathbf{P}_1\mathbf{P}_2 + \mathbf{P}_1\mathbf{P}_2\mathbf{P}_1 = \mathbf{O}$이고, 오른쪽에 곱하면 $\mathbf{P}_1\mathbf{P}_2\mathbf{P}_1 + \mathbf{P}_2\mathbf{P}_1 = \mathbf{O}$이다. 두 식을 빼면 $\mathbf{P}_1\mathbf{P}_2 = \mathbf{P}_2\mathbf{P}_1$이고, 합이 $\mathbf{O}$이므로 $2\mathbf{P}_1\mathbf{P}_2 = \mathbf{O}$, 곧 둘 다 $\mathbf{O}$다.

    **기하적 의미.** $\mathbf{P}_1\mathbf{P}_2 = \mathbf{O}$은 한 사영의 치역이 다른 사영의 영공간에 들어간다는 뜻이다. 직교사영이라면 **두 목표 부분공간이 서로 직교**한다는 말과 같다.

    ```python
    import numpy as np

    P1 = np.diag([1., 0., 0.])
    P2 = np.diag([0., 1., 0.])       # 서로 직교하는 축 위로의 사영
    S = P1 + P2
    print("P1 P2 = 0 인가:", np.allclose(P1 @ P2, 0))
    print("합이 멱등인가 :", np.allclose(S @ S, S), " rank:", np.linalg.matrix_rank(S))

    P3 = np.array([[.5, .5, 0], [.5, .5, 0], [0, 0, 0]])   # 직교하지 않는 사영
    print("\nP1 P3 = 0 인가:", np.allclose(P1 @ P3, 0))
    print("합이 멱등인가 :", np.allclose((P1 + P3) @ (P1 + P3), P1 + P3))
    ```

    출력:

    ```
    P1 P2 = 0 인가: True
    합이 멱등인가 : True  rank: 2

    P1 P3 = 0 인가: False
    합이 멱등인가 : False
    ```

    이 성질이 분산분석의 제곱합 분해를 떠받친다. 총제곱합이 여러 성분으로 **깔끔하게 쪼개지려면** 대응하는 사영들이 서로 직교해야 한다. $\square$

<div class="drillbox" markdown>

**연습문제 9.**
중심화행렬 $\mathbf{C} = \mathbf{I} - \frac{1}{n}\mathbf{J}$($\mathbf{J}$는 모든 성분이 1인 $n \times n$ 행렬)가 직교사영임을 보이고, 무엇 위로 사영하는지 밝혀라. $\operatorname{tr}(\mathbf{C})$는 얼마인가?

</div>

??? success "풀이"
    $\mathbf{J} = \mathbf{1}\mathbf{1}^T$이고 $\mathbf{1}^T\mathbf{1} = n$이므로 $\mathbf{J}^2 = \mathbf{1}(\mathbf{1}^T\mathbf{1})\mathbf{1}^T = n\mathbf{J}$이다. 따라서

    $$
    \mathbf{C}^2 = \mathbf{I} - \frac{2}{n}\mathbf{J} + \frac{1}{n^2}\mathbf{J}^2
    = \mathbf{I} - \frac{2}{n}\mathbf{J} + \frac{1}{n}\mathbf{J} = \mathbf{C}
    $$

    이고, $\mathbf{J}^T = \mathbf{J}$이므로 $\mathbf{C}^T = \mathbf{C}$다. 멱등이면서 대칭이므로 **직교사영**이다.

    $\mathbf{C}\mathbf{x} = \mathbf{x} - \bar{x}\mathbf{1}$이므로 $\mathbf{C}$는 $\mathbf{1}$과 직교인 부분공간, 곧 **성분의 합이 0인 벡터들의 공간** 위로 사영한다. 눌러 없애는 방향은 $\operatorname{span}\{\mathbf{1}\}$이다.

    대각합은 $\operatorname{tr}(\mathbf{I}) - \frac{1}{n}\operatorname{tr}(\mathbf{J}) = n - \frac{n}{n} = n - 1$이다.

    ```python
    import numpy as np

    n = 5
    C = np.eye(n) - np.ones((n, n)) / n
    x = np.array([2., 4., 4., 4., 6.])

    print("멱등:", np.allclose(C @ C, C), " 대칭:", np.allclose(C, C.T))
    print("tr(C) =", round(np.trace(C), 6), " (= n - 1)")
    print("Cx      =", (C @ x).round(10))
    print("x - 평균 =", x - x.mean())
    print("C1 =", (C @ np.ones(n)).round(10), " (1 은 영공간에 있다)")
    ```

    출력:

    ```
    멱등: True  대칭: True
    tr(C) = 4.0  (= n - 1)
    Cx      = [-2.  0.  0.  0.  2.]
    x - 평균 = [-2.  0.  0.  0.  2.]
    C1 = [0. 0. 0. 0. 0.]  (1 은 영공간에 있다)
    ```

    **자유도의 기원.** $\operatorname{tr}(\mathbf{C}) = n - 1$이 표본분산에서 $n-1$로 나누는 이유의 기하적 설명이다. 중심화된 잔차 벡터는 $n$차원이 아니라 $n-1$차원 부분공간에 놓인다. $\square$

<div class="drillbox" markdown>

**연습문제 10.**
빗각 사영은 거리를 최소화하지 않는다. 본문의 $\mathbf{P} = \begin{pmatrix} 1 & -1 \\ 0 & 0 \end{pmatrix}$과 같은 부분공간 위로의 직교사영을 $\mathbf{x} = (0, 1)^T$에 적용해 $\lVert\mathbf{x} - \mathbf{P}\mathbf{x}\rVert$를 비교하라.

</div>

??? success "풀이"
    두 사영 모두 치역이 $\operatorname{span}\{(1,0)^T\}$로 같지만 눌러 없애는 방향이 다르다.

    - 직교사영 $\mathbf{P}_\perp = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$: $\mathbf{P}_\perp\mathbf{x} = (0,0)^T$, 거리 $\lVert(0,1)^T\rVert = 1$.
    - 빗각 사영 $\mathbf{P}$: $\mathbf{P}\mathbf{x} = (-1, 0)^T$, 거리 $\lVert(1,1)^T\rVert = \sqrt{2} \approx 1.414$.

    ```python
    import numpy as np

    x = np.array([0., 1.])
    P_orth = np.array([[1., 0.], [0., 0.]])
    P_obl = np.array([[1., -1.], [0., 0.]])

    for name, P in [("직교", P_orth), ("빗각", P_obl)]:
        proj = P @ x
        print(f"{name}: Px = {proj},  거리 = {np.linalg.norm(x - proj):.4f}")
    ```

    출력:

    ```
    직교: Px = [0. 0.],  거리 = 1.0000
    빗각: Px = [-1.  0.],  거리 = 1.4142
    ```

    빗각 사영의 거리가 더 크다. **치역이 같아도 어느 방향으로 누르느냐가 거리를 바꾼다.**

    직교사영만이 $\lVert\mathbf{x} - \mathbf{v}\rVert$를 $\mathbf{v} \in \mathcal{V}$ 위에서 최소화한다. 최소제곱이 잔차 제곱합을 최소화하는 추정량을 주는 것도, 모자 행렬이 **직교**사영이기 때문이다. 빗각 사영을 쓰면 여전히 $\mathcal{V}$ 안의 어떤 점을 얻지만 그 점은 가장 가까운 점이 아니다. $\square$

