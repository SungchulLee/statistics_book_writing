# 직교사영행렬

직교사영행렬은 모든 벡터를 어떤 부분공간 안에서 가장 가까운 점으로 보내며, 여기서 "가깝다"는 유클리드 거리로 잰다. 이것이 통상 최소제곱의 기하학적 내용이다. 적합값 벡터 $\hat{\mathbf{y}}$는 반응벡터 $\mathbf{y}$를 계획행렬 $\mathbf{X}$의 열공간 위로 직교사영한 것이고, 잔차벡터 $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$는 그 열공간에 수직이다. 직교사영은 멱등이면서 대칭인 것으로 특징지어지며, 그 명시적 공식 $\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$이 회귀의 모자 행렬이다.

## 정의

!!! info "정의 — 직교사영행렬"
    정사각행렬 $\mathbf{P} \in \mathbb{R}^{n \times n}$이 멱등이면서 대칭이면 **직교사영행렬**이라 한다.

    $$
    \mathbf{P}^2 = \mathbf{P} \quad \text{and} \quad \mathbf{P}^T = \mathbf{P}
    $$

대칭성 조건이 직교사영을 빗각 사영과 구별해 준다. 어떤 사영이 직교사영일 필요충분조건은 영공간이 열공간의 직교여공간인 것이다: $\ker(\mathbf{P}) = \operatorname{col}(\mathbf{P})^\perp$.

## 최선근사 성질

직교사영을 규정하는 기하적 성질은 가장 가까운 점 문제를 푼다는 것이다.

!!! tip "정리 — 최선근사"
    $\mathbf{P}$가 부분공간 $\mathcal{V} \subseteq \mathbb{R}^n$ 위로의 직교사영이라 하자. 임의의 $\mathbf{x} \in \mathbb{R}^n$에 대해 벡터 $\mathbf{P}\mathbf{x}$는 $\mathbf{x}$까지의 거리를 최소화하는 $\mathcal{V}$의 유일한 원소다.

    $$
    \mathbf{P}\mathbf{x} = \arg\min_{\mathbf{v} \in \mathcal{V}} \lVert\mathbf{x} - \mathbf{v}\rVert
    $$

**증명 개요.** $\mathbf{v} \in \mathcal{V}$를 임의로 잡자. 그러면

$$
\lVert\mathbf{x} - \mathbf{v}\rVert^2 = \lVert(\mathbf{x} - \mathbf{P}\mathbf{x}) + (\mathbf{P}\mathbf{x} - \mathbf{v})\rVert^2
$$

이다. ($\mathbf{P}$가 직교사영이므로) $\mathbf{x} - \mathbf{P}\mathbf{x} \in \mathcal{V}^\perp$이고 $\mathbf{P}\mathbf{x} - \mathbf{v} \in \mathcal{V}$이므로 이 두 벡터는 직교한다. 피타고라스 정리에 의해

$$
\lVert\mathbf{x} - \mathbf{v}\rVert^2 = \lVert\mathbf{x} - \mathbf{P}\mathbf{x}\rVert^2 + \lVert\mathbf{P}\mathbf{x} - \mathbf{v}\rVert^2 \geq \lVert\mathbf{x} - \mathbf{P}\mathbf{x}\rVert^2
$$

이다. 등호는 $\mathbf{v} = \mathbf{P}\mathbf{x}$일 때에 한해 성립한다. $\square$

바로 이것이 최소제곱이 잔차제곱합을 최소화하는 이유다. $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$가 $\operatorname{col}(\mathbf{X})$ 안에서 $\mathbf{y}$에 가장 가까운 점이다.

## 열공간 위로의 직교사영 공식

!!! tip "정리 — 사영 공식"
    $\mathbf{X} \in \mathbb{R}^{n \times p}$가 완전 열계수를 갖는다고 하자($\operatorname{rank}(\mathbf{X}) = p$). $\operatorname{col}(\mathbf{X})$ 위로의 직교사영은

    $$
    \mathbf{P}_{\mathbf{X}} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T
    $$

    이다.

**증명.** 두 가지 규정 성질을 확인한다.

*멱등성:*

$$
\mathbf{P}_{\mathbf{X}}^2 = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T = \mathbf{P}_{\mathbf{X}}
$$

*대칭성:*

$$
\mathbf{P}_{\mathbf{X}}^T = (\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T)^T = \mathbf{X}((\mathbf{X}^T\mathbf{X})^{-1})^T\mathbf{X}^T = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T = \mathbf{P}_{\mathbf{X}}
$$

마지막 단계에서는 $(\mathbf{X}^T\mathbf{X})^{-1}$이 대칭이라는 사실을 썼다(대칭행렬의 역행렬은 대칭이다). $\square$

## 성질

### 고윳값

직교사영의 고윳값은 (멱등성에서 물려받아) 0과 1이며,

$$
\operatorname{tr}(\mathbf{P}_{\mathbf{X}}) = \operatorname{rank}(\mathbf{P}_{\mathbf{X}}) = p
$$

이다.

### 여집합 사영

행렬 $\mathbf{M} = \mathbf{I} - \mathbf{P}_{\mathbf{X}}$는 $\operatorname{col}(\mathbf{X})^\perp$ 위로의 직교사영이다.

$$
\mathbf{M}^2 = \mathbf{M}, \quad \mathbf{M}^T = \mathbf{M}, \quad \operatorname{tr}(\mathbf{M}) = n - p
$$

### 사영들의 직교성

$\mathbf{P}_{\mathbf{X}}$의 치역과 $\mathbf{M}$의 치역은 서로 직교여공간이므로

$$
\mathbf{P}_{\mathbf{X}}\mathbf{M} = \mathbf{M}\mathbf{P}_{\mathbf{X}} = \mathbf{0}
$$

이다.

### 열공간의 특성화

벡터 $\mathbf{v}$가 $\operatorname{col}(\mathbf{X})$에 속할 필요충분조건은 $\mathbf{P}_{\mathbf{X}}\mathbf{v} = \mathbf{v}$이고, $\mathbf{v}$가 $\operatorname{col}(\mathbf{X})$에 직교할 필요충분조건은 $\mathbf{P}_{\mathbf{X}}\mathbf{v} = \mathbf{0}$이다.

## 하나의 벡터 위로의 사영

부분공간이 1차원일 때, 즉 0이 아닌 벡터 $\mathbf{u}$에 대해 $\mathcal{V} = \operatorname{span}\{\mathbf{u}\}$일 때 사영 공식은

$$
\mathbf{P}_{\mathbf{u}} = \frac{\mathbf{u}\mathbf{u}^T}{\mathbf{u}^T\mathbf{u}}
$$

으로 간단해진다. 이것은 임의의 벡터 $\mathbf{x}$를 $\mathbf{u}$ 위로 사영한다.

$$
\mathbf{P}_{\mathbf{u}}\mathbf{x} = \frac{\mathbf{u}^T\mathbf{x}}{\mathbf{u}^T\mathbf{u}}\,\mathbf{u}
$$

스칼라 $\frac{\mathbf{u}^T\mathbf{x}}{\mathbf{u}^T\mathbf{u}}$가 $\mathbf{u}$ 위로 사영된 $\mathbf{x}$의 계수다.

## 예 — 중심화행렬

**중심화행렬**은

$$
\mathbf{C} = \mathbf{I}_n - \frac{1}{n}\mathbf{1}_n\mathbf{1}_n^T
$$

이며, 여기서 $\mathbf{1}_n = (1, \dots, 1)^T$이다. 이 행렬은 다음을 만족한다.

- $\mathbf{C}^2 = \mathbf{C}$ (멱등)
- $\mathbf{C}^T = \mathbf{C}$ (대칭)
- $\operatorname{tr}(\mathbf{C}) = n - 1$

따라서 $\mathbf{C}$는 계수 $n - 1$인 직교사영이다. $\mathbf{1}_n$에 직교하는 부분공간(성분의 합이 0인 벡터들의 부분공간) 위로 사영한다. 임의의 자료벡터 $\mathbf{x}$에 대해

$$
\mathbf{C}\mathbf{x} = \mathbf{x} - \bar{x}\mathbf{1}_n
$$

이며, 여기서 $\bar{x} = \frac{1}{n}\sum_i x_i$는 표본평균이다. 자료를 중심화하는 것은 하나의 직교사영이다.

## 예 — 모자 행렬

완전 열계수를 갖는 $\mathbf{X} \in \mathbb{R}^{n \times p}$에 대한 선형모형 $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$에서 **모자 행렬**은

$$
\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T
$$

이다. 이것이 $\operatorname{col}(\mathbf{X})$ 위로의 직교사영이다. 이름은 $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$라는 사실에서 왔다. 모자 행렬이 "$\mathbf{y}$에 모자를 씌우는" 것이다.

회귀에서의 핵심 성질:

- $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$ (적합값)
- $\mathbf{e} = (\mathbf{I} - \mathbf{H})\mathbf{y}$ (잔차)
- $\hat{\mathbf{y}} \perp \mathbf{e}$ (적합값과 잔차의 직교성)
- $\operatorname{tr}(\mathbf{H}) = p$ (모형 자유도)
- $\operatorname{tr}(\mathbf{I} - \mathbf{H}) = n - p$ (잔차 자유도)

## 회귀에서의 피타고라스 정리

직교성 $\hat{\mathbf{y}} \perp \mathbf{e}$가 피타고라스 정리를 준다.

$$
\lVert\mathbf{y}\rVert^2 = \lVert\hat{\mathbf{y}}\rVert^2 + \lVert\mathbf{e}\rVert^2
$$

모형이 절편을 포함하고 평균을 기준으로 측정하면 이것이 분산분석 분해가 된다.

$$
\text{SST} = \text{SSR} + \text{SSE}
$$

여기서 SST는 총제곱합, SSR은 회귀제곱합, SSE는 오차제곱합이다. 그러면 $R^2$ 계수는 $R^2 = \text{SSR}/\text{SST} = \lVert\hat{\mathbf{y}}\rVert^2 / \lVert\mathbf{y}\rVert^2$이며, 이는 $\mathbf{y}$와 $\hat{\mathbf{y}}$ 사이 각의 코사인의 제곱, 즉 모형이 얼마나 잘 맞는지에 대한 기하적 측도다.

## 요약

직교사영행렬은 벡터를 부분공간 안에서 가장 가까운 점으로 보내는 대칭 멱등행렬이다. 공식 $\mathbf{P}_{\mathbf{X}} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$이 회귀의 모자 행렬이고, 여집합 사영 $\mathbf{I} - \mathbf{P}_{\mathbf{X}}$이 잔차를 만들어낸다. 적합값과 잔차의 직교성이 제곱합의 피타고라스 분해를 낳으며, 이것이 분산분석과 $R^2$의 기하학적 토대다.

## 연습문제

**연습문제 1.**
$\mathbf{X} = \begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{pmatrix}$이라 하자. 모자 행렬 $\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$를 계산하고 $\operatorname{tr}(\mathbf{H}) = 2$임을 확인하라.

??? success "연습문제 1 풀이"
    먼저 다음을 계산한다.

    $$
    \mathbf{X}^T\mathbf{X} = \begin{pmatrix} 3 & 6 \\ 6 & 14 \end{pmatrix}, \quad (\mathbf{X}^T\mathbf{X})^{-1} = \frac{1}{6}\begin{pmatrix} 14 & -6 \\ -6 & 3 \end{pmatrix}
    $$

    그러면

    $$
    \mathbf{H} = \mathbf{X} \cdot \frac{1}{6}\begin{pmatrix} 14 & -6 \\ -6 & 3 \end{pmatrix} \cdot \mathbf{X}^T = \frac{1}{6}\begin{pmatrix} 5 & 2 & -1 \\ 2 & 2 & 2 \\ -1 & 2 & 5 \end{pmatrix}
    $$

    이다. 대각합은 $\operatorname{tr}(\mathbf{H}) = (5 + 2 + 5)/6 = 12/6 = 2$로 $\mathbf{X}$의 열 개수($p = 2$)와 같다. 이는 일반적인 결과 $\operatorname{tr}(\mathbf{H}) = \operatorname{rank}(\mathbf{H}) = p$를 확인해 준다.

---

**연습문제 2.**
$\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$, $\mathbf{e} = (\mathbf{I} - \mathbf{H})\mathbf{y}$일 때 피타고라스 분해 $\lVert \mathbf{y} \rVert^2 = \lVert \hat{\mathbf{y}} \rVert^2 + \lVert \mathbf{e} \rVert^2$를 증명하라.

??? success "연습문제 2 풀이"
    $\mathbf{y} = \hat{\mathbf{y}} + \mathbf{e}$이므로

    $$
    \lVert \mathbf{y} \rVert^2 = (\hat{\mathbf{y}} + \mathbf{e})^T(\hat{\mathbf{y}} + \mathbf{e}) = \lVert \hat{\mathbf{y}} \rVert^2 + 2\hat{\mathbf{y}}^T\mathbf{e} + \lVert \mathbf{e} \rVert^2
    $$

    이다. 교차항은 다음과 같이 사라진다.

    $$
    \hat{\mathbf{y}}^T\mathbf{e} = (\mathbf{H}\mathbf{y})^T(\mathbf{I} - \mathbf{H})\mathbf{y} = \mathbf{y}^T\mathbf{H}^T(\mathbf{I} - \mathbf{H})\mathbf{y} = \mathbf{y}^T\mathbf{H}(\mathbf{I} - \mathbf{H})\mathbf{y}
    $$

    $\mathbf{H}(\mathbf{I} - \mathbf{H}) = \mathbf{H} - \mathbf{H}^2 = \mathbf{H} - \mathbf{H} = \mathbf{0}$이므로 교차항이 0이 되어 $\lVert \mathbf{y} \rVert^2 = \lVert \hat{\mathbf{y}} \rVert^2 + \lVert \mathbf{e} \rVert^2$를 얻는다. $\square$

---

**연습문제 3.**
$\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$가 $\text{col}(\mathbf{X})$ 안에서 $\mathbf{y}$에 가장 가까운 점임을 증명하여, 모자 행렬 $\mathbf{H}$가 $\lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert^2$을 최소화함을 보여라.

??? success "연습문제 3 풀이"
    $\mathbf{z} = \mathbf{X}\boldsymbol{\beta}$을 $\text{col}(\mathbf{X})$의 임의의 벡터라 하자. $\lVert \mathbf{y} - \hat{\mathbf{y}} \rVert \leq \lVert \mathbf{y} - \mathbf{z} \rVert$를 보이면 된다.

    $\mathbf{y} - \mathbf{z} = (\mathbf{y} - \hat{\mathbf{y}}) + (\hat{\mathbf{y}} - \mathbf{z})$로 쓰자. $\mathbf{y} - \hat{\mathbf{y}} = \mathbf{e} \in \text{col}(\mathbf{X})^\perp$이고 $\hat{\mathbf{y}} - \mathbf{z} \in \text{col}(\mathbf{X})$이므로 이 두 벡터는 직교한다. 피타고라스 정리에 의해

    $$
    \lVert \mathbf{y} - \mathbf{z} \rVert^2 = \lVert \mathbf{y} - \hat{\mathbf{y}} \rVert^2 + \lVert \hat{\mathbf{y}} - \mathbf{z} \rVert^2 \geq \lVert \mathbf{y} - \hat{\mathbf{y}} \rVert^2
    $$

    이다. 등호는 $\mathbf{z} = \hat{\mathbf{y}}$일 때에 한해 성립하므로 $\hat{\mathbf{y}}$이 유일한 최근접점임이 확인된다. $\square$

---

**연습문제 4.**
$\mathbf{y}$와 그 사영 $\hat{\mathbf{y}}$ 사이의 각이라는 관점에서 $R^2 = \lVert \hat{\mathbf{y}} \rVert^2 / \lVert \mathbf{y} \rVert^2$의 기하적 해석을 설명하라. $R^2 = 1$은 기하적으로 무엇을 뜻하는가?

??? success "연습문제 4 풀이"
    $\theta$를 $\mathbb{R}^n$에서 $\mathbf{y}$와 $\hat{\mathbf{y}}$ 사이의 각이라 하자. $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$가 $\mathbf{y}$를 $\text{col}(\mathbf{X})$ 위로 사영한 것이므로

    $$
    \cos\theta = \frac{\hat{\mathbf{y}}^T\mathbf{y}}{\lVert \hat{\mathbf{y}} \rVert \lVert \mathbf{y} \rVert} = \frac{\lVert \hat{\mathbf{y}} \rVert^2}{\lVert \hat{\mathbf{y}} \rVert \lVert \mathbf{y} \rVert} = \frac{\lVert \hat{\mathbf{y}} \rVert}{\lVert \mathbf{y} \rVert}
    $$

    이다. 따라서 $R^2 = \cos^2\theta$이다. 즉 반응벡터와 그것을 모형 부분공간 위로 사영한 벡터 사이 각의 코사인의 제곱이다.

    $R^2 = 1$은 $\cos^2\theta = 1$, 즉 $\theta = 0$을 뜻하므로 반응벡터 $\mathbf{y}$가 정확히 $\text{col}(\mathbf{X})$ 안에 놓인다. 기하적으로 자료가 잔차 없이 모형에 완벽히 들어맞는다는 뜻이다.
