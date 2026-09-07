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

!!! tip "정리 — 직합 분해"
    $\mathbf{P}$가 사영이면 $\mathbb{R}^n = \operatorname{col}(\mathbf{P}) \oplus \ker(\mathbf{P})$이고, 모든 $\mathbf{x} \in \mathbb{R}^n$에 대해

    $$
    \mathbf{x} = \mathbf{P}\mathbf{x} + (\mathbf{I} - \mathbf{P})\mathbf{x}
    $$

    이며, 여기서 $\mathbf{P}\mathbf{x} \in \operatorname{col}(\mathbf{P})$이고 $(\mathbf{I} - \mathbf{P})\mathbf{x} \in \ker(\mathbf{P})$이다.

**증명.** 분해 $\mathbf{x} = \mathbf{P}\mathbf{x} + (\mathbf{I} - \mathbf{P})\mathbf{x}$는 자명하게 참이다. 주장된 부분공간 소속을 확인한다.

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

!!! tip "정리 — 사영의 유일성"
    직합 분해 $\mathbb{R}^n = \mathcal{V} \oplus \mathcal{W}$가 주어지면 $\operatorname{col}(\mathbf{P}) = \mathcal{V}$이고 $\ker(\mathbf{P}) = \mathcal{W}$인 사영 $\mathbf{P}$가 유일하게 존재한다.

이는 목표 부분공간 $\mathcal{V}$만 지정해서는 사영이 유일하게 결정되지 않음을 뜻한다. 눌러 없애는 방향 $\mathcal{W}$도 함께 지정해야 한다. $\mathcal{W} = \mathcal{V}^\perp$일 때 그 사영은 직교사영이 되고, 이 특수한 경우에는 $\mathcal{V}$만으로 사영이 결정된다.

## 통계와의 연결

### 적합값과 잔차

선형회귀에서 분해 $\mathbf{y} = \mathbf{H}\mathbf{y} + (\mathbf{I} - \mathbf{H})\mathbf{y} = \hat{\mathbf{y}} + \mathbf{e}$는 사영 분해다. 모자 행렬 $\mathbf{H}$는 $\mathbf{X}$의 열공간 위로 사영하고, $\mathbf{I} - \mathbf{H}$는 그 직교여공간(잔차공간) 위로 사영한다.

### 분산분석 분해

분산분석의 총제곱합 분해는 모형 부분공간과 오차 부분공간 위로의 직교사영이 관여하는 이차형식들의 합으로 쓸 수 있다. 직교성 덕분에 (정규성 아래에서) 제곱합들이 서로 독립이 되며, 이것이 F-검정의 근거다.

## 요약

사영행렬은 $\mathbb{R}^n$을 자신의 열공간과 영공간의 직합으로 분해하는 멱등행렬이다. 모든 벡터가 각 부분공간의 성분으로 쪼개지고, 사영을 두 번 적용해도 새로운 일은 일어나지 않는다. 대각합은 목표 부분공간의 차원과 같다. 사영이 대칭이기까지 하면 직교사영이 되며, 회귀와 분산분석에서 가장 자주 등장하는 것이 바로 이 유형이다.

## 연습문제

**연습문제 1.**
$\mathbf{P} = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$이 사영행렬임을 확인하라. 어느 부분공간 위로 사영하는가? 여집합 사영 $\mathbf{I} - \mathbf{P}$는 무엇인가?

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

---

**연습문제 2.**
$\mathbf{P}$가 멱등이면 $\operatorname{rank}(\mathbf{P}) = \operatorname{tr}(\mathbf{P})$임을 증명하라.

??? success "풀이"
    $\mathbf{P}$가 멱등이므로 고윳값은 0 아니면 1이다($\mathbf{P}\mathbf{v} = \lambda\mathbf{v}$이면 $\mathbf{P}^2\mathbf{v} = \lambda^2\mathbf{v} = \lambda\mathbf{v}$이므로 $\lambda^2 = \lambda$이고 $\lambda \in \{0, 1\}$이다).

    계수는 0이 아닌 고윳값의 개수와 같고, 이는 1인 고윳값의 개수다. 대각합은 모든 고윳값의 합인데, 나머지가 0이므로 이 또한 1인 고윳값의 개수다.

    따라서 $\operatorname{rank}(\mathbf{P}) = \operatorname{tr}(\mathbf{P})$이다. $\square$

---

**연습문제 3.**
빗각(직교가 아닌) 사영행렬의 예를 들어라. 멱등이지만 대칭이 아님을 확인하라.

??? success "풀이"
    $\mathbf{P} = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}$을 생각하자.

    멱등성:

    $$
    \mathbf{P}^2 = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix} = \mathbf{P}
    $$

    그러나 $\mathbf{P}^T = \begin{pmatrix} 1 & 0 \\ 1 & 0 \end{pmatrix} \neq \mathbf{P}$이므로 $\mathbf{P}$는 대칭이 아니다.

    이 행렬은 $(1, 0)^T + \ker(\mathbf{P}) = (1, 0)^T + \text{span}\{(-1, 1)^T\}$ 방향을 따라 $\text{span}\{(1, 0)^T\}$ 위로 사영한다. 사영 방향이 목표 부분공간에 대해 빗각이다(수직이 아니다).

---

**연습문제 4.**
회귀 분해 $\mathbf{y} = \mathbf{H}\mathbf{y} + (\mathbf{I} - \mathbf{H})\mathbf{y}$에서 $\mathbf{H}\mathbf{y}$와 $(\mathbf{I} - \mathbf{H})\mathbf{y}$가 직교인 이유를 설명하라. 멱등성 외에 어떤 성질이 추가로 필요한가?

??? success "풀이"
    $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$와 $\mathbf{e} = (\mathbf{I} - \mathbf{H})\mathbf{y}$의 직교성에는 $\mathbf{H}$가 (멱등일 뿐 아니라) 대칭이어야 한다. 대칭성이 있으면

    $$
    \hat{\mathbf{y}}^T\mathbf{e} = \mathbf{y}^T\mathbf{H}^T(\mathbf{I} - \mathbf{H})\mathbf{y} = \mathbf{y}^T\mathbf{H}(\mathbf{I} - \mathbf{H})\mathbf{y} = \mathbf{y}^T(\mathbf{H} - \mathbf{H}^2)\mathbf{y} = \mathbf{0}
    $$

    이다. $\mathbf{H}$가 멱등이지만 대칭이 아니라면(빗각 사영이라면) 분해 $\mathbf{y} = \mathbf{H}\mathbf{y} + (\mathbf{I} - \mathbf{H})\mathbf{y}$는 여전히 성립하지만 두 성분이 직교하지는 않는다. 최소제곱의 모자 행렬은 멱등이면서 대칭이며, 그 덕분에 제곱합 분해와 피타고라스 정리가 작동한다.
