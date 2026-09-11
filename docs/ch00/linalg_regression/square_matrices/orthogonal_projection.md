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

<div class="thmbox" markdown>

### 정리 1. 최선근사 { .thm }

$\mathbf{P}$가 부분공간 $\mathcal{V} \subseteq \mathbb{R}^n$ 위로의 직교사영이라 하자. 임의의 $\mathbf{x} \in \mathbb{R}^n$에 대해 벡터 $\mathbf{P}\mathbf{x}$는 $\mathbf{x}$까지의 거리를 최소화하는 $\mathcal{V}$의 유일한 원소다.

$$
\mathbf{P}\mathbf{x} = \arg\min_{\mathbf{v} \in \mathcal{V}} \lVert\mathbf{x} - \mathbf{v}\rVert
$$

</div>

??? proof "증명 개요"

    $\mathbf{v} \in \mathcal{V}$를 임의로 잡자. 그러면

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

<div class="thmbox" markdown>

### 정리 2. 사영 공식 { .thm }

$\mathbf{X} \in \mathbb{R}^{n \times p}$가 완전 열계수를 갖는다고 하자($\operatorname{rank}(\mathbf{X}) = p$). $\operatorname{col}(\mathbf{X})$ 위로의 직교사영은

$$
\mathbf{P}_{\mathbf{X}} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T
$$

이다.

</div>

??? proof "증명"

    두 가지 규정 성질을 확인한다.

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

<div class="drillbox" markdown>

**연습문제 1.**
$\mathbf{X} = \begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{pmatrix}$이라 하자. 모자 행렬 $\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$를 계산하고 $\operatorname{tr}(\mathbf{H}) = 2$임을 확인하라.

</div>

??? success "풀이"
    먼저 다음을 계산한다.

    $$
    \mathbf{X}^T\mathbf{X} = \begin{pmatrix} 3 & 6 \\ 6 & 14 \end{pmatrix}, \quad (\mathbf{X}^T\mathbf{X})^{-1} = \frac{1}{6}\begin{pmatrix} 14 & -6 \\ -6 & 3 \end{pmatrix}
    $$

    그러면

    $$
    \mathbf{H} = \mathbf{X} \cdot \frac{1}{6}\begin{pmatrix} 14 & -6 \\ -6 & 3 \end{pmatrix} \cdot \mathbf{X}^T = \frac{1}{6}\begin{pmatrix} 5 & 2 & -1 \\ 2 & 2 & 2 \\ -1 & 2 & 5 \end{pmatrix}
    $$

    이다. 대각합은 $\operatorname{tr}(\mathbf{H}) = (5 + 2 + 5)/6 = 12/6 = 2$로 $\mathbf{X}$의 열 개수($p = 2$)와 같다. 이는 일반적인 결과 $\operatorname{tr}(\mathbf{H}) = \operatorname{rank}(\mathbf{H}) = p$를 확인해 준다.

<div class="drillbox" markdown>

**연습문제 2.**
$\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$, $\mathbf{e} = (\mathbf{I} - \mathbf{H})\mathbf{y}$일 때 피타고라스 분해 $\lVert \mathbf{y} \rVert^2 = \lVert \hat{\mathbf{y}} \rVert^2 + \lVert \mathbf{e} \rVert^2$를 증명하라.

</div>

??? success "풀이"
    $\mathbf{y} = \hat{\mathbf{y}} + \mathbf{e}$이므로

    $$
    \lVert \mathbf{y} \rVert^2 = (\hat{\mathbf{y}} + \mathbf{e})^T(\hat{\mathbf{y}} + \mathbf{e}) = \lVert \hat{\mathbf{y}} \rVert^2 + 2\hat{\mathbf{y}}^T\mathbf{e} + \lVert \mathbf{e} \rVert^2
    $$

    이다. 교차항은 다음과 같이 사라진다.

    $$
    \hat{\mathbf{y}}^T\mathbf{e} = (\mathbf{H}\mathbf{y})^T(\mathbf{I} - \mathbf{H})\mathbf{y} = \mathbf{y}^T\mathbf{H}^T(\mathbf{I} - \mathbf{H})\mathbf{y} = \mathbf{y}^T\mathbf{H}(\mathbf{I} - \mathbf{H})\mathbf{y}
    $$

    $\mathbf{H}(\mathbf{I} - \mathbf{H}) = \mathbf{H} - \mathbf{H}^2 = \mathbf{H} - \mathbf{H} = \mathbf{0}$이므로 교차항이 0이 되어 $\lVert \mathbf{y} \rVert^2 = \lVert \hat{\mathbf{y}} \rVert^2 + \lVert \mathbf{e} \rVert^2$를 얻는다. $\square$

<div class="drillbox" markdown>

**연습문제 3.**
$\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$가 $\text{col}(\mathbf{X})$ 안에서 $\mathbf{y}$에 가장 가까운 점임을 증명하여, 모자 행렬 $\mathbf{H}$가 $\lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert^2$을 최소화함을 보여라.

</div>

??? success "풀이"
    $\mathbf{z} = \mathbf{X}\boldsymbol{\beta}$을 $\text{col}(\mathbf{X})$의 임의의 벡터라 하자. $\lVert \mathbf{y} - \hat{\mathbf{y}} \rVert \leq \lVert \mathbf{y} - \mathbf{z} \rVert$를 보이면 된다.

    $\mathbf{y} - \mathbf{z} = (\mathbf{y} - \hat{\mathbf{y}}) + (\hat{\mathbf{y}} - \mathbf{z})$로 쓰자. $\mathbf{y} - \hat{\mathbf{y}} = \mathbf{e} \in \text{col}(\mathbf{X})^\perp$이고 $\hat{\mathbf{y}} - \mathbf{z} \in \text{col}(\mathbf{X})$이므로 이 두 벡터는 직교한다. 피타고라스 정리에 의해

    $$
    \lVert \mathbf{y} - \mathbf{z} \rVert^2 = \lVert \mathbf{y} - \hat{\mathbf{y}} \rVert^2 + \lVert \hat{\mathbf{y}} - \mathbf{z} \rVert^2 \geq \lVert \mathbf{y} - \hat{\mathbf{y}} \rVert^2
    $$

    이다. 등호는 $\mathbf{z} = \hat{\mathbf{y}}$일 때에 한해 성립하므로 $\hat{\mathbf{y}}$이 유일한 최근접점임이 확인된다. $\square$

<div class="drillbox" markdown>

**연습문제 4.**
$\mathbf{y}$와 그 사영 $\hat{\mathbf{y}}$ 사이의 각이라는 관점에서 $R^2 = \lVert \hat{\mathbf{y}} \rVert^2 / \lVert \mathbf{y} \rVert^2$의 기하적 해석을 설명하라. $R^2 = 1$은 기하적으로 무엇을 뜻하는가?

</div>

??? success "풀이"
    $\theta$를 $\mathbb{R}^n$에서 $\mathbf{y}$와 $\hat{\mathbf{y}}$ 사이의 각이라 하자. $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$가 $\mathbf{y}$를 $\text{col}(\mathbf{X})$ 위로 사영한 것이므로

    $$
    \cos\theta = \frac{\hat{\mathbf{y}}^T\mathbf{y}}{\lVert \hat{\mathbf{y}} \rVert \lVert \mathbf{y} \rVert} = \frac{\lVert \hat{\mathbf{y}} \rVert^2}{\lVert \hat{\mathbf{y}} \rVert \lVert \mathbf{y} \rVert} = \frac{\lVert \hat{\mathbf{y}} \rVert}{\lVert \mathbf{y} \rVert}
    $$

    이다. 따라서 $R^2 = \cos^2\theta$이다. 즉 반응벡터와 그것을 모형 부분공간 위로 사영한 벡터 사이 각의 코사인의 제곱이다.

    $R^2 = 1$은 $\cos^2\theta = 1$, 즉 $\theta = 0$을 뜻하므로 반응벡터 $\mathbf{y}$가 정확히 $\text{col}(\mathbf{X})$ 안에 놓인다. 기하적으로 자료가 잔차 없이 모형에 완벽히 들어맞는다는 뜻이다.

<div class="drillbox" markdown>

**연습문제 5.**
영이 아닌 벡터 $\mathbf{a}$ 하나가 펼치는 직선 위로의 직교사영이 $\mathbf{P} = \dfrac{\mathbf{a}\mathbf{a}^T}{\mathbf{a}^T\mathbf{a}}$임을 보이고, 이것이 절편 없는 단순회귀와 어떻게 연결되는지 설명하라.

</div>

??? success "풀이"
    사영 공식 $\mathbf{P}_{\mathbf{X}} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$에서 $\mathbf{X} = \mathbf{a}$($n \times 1$)로 두면 $\mathbf{X}^T\mathbf{X} = \mathbf{a}^T\mathbf{a}$가 스칼라이므로 역행렬이 곧 역수다.

    $$
    \mathbf{P} = \mathbf{a}\,(\mathbf{a}^T\mathbf{a})^{-1}\mathbf{a}^T = \frac{\mathbf{a}\mathbf{a}^T}{\mathbf{a}^T\mathbf{a}}
    $$

    멱등성은 $\mathbf{P}^2 = \dfrac{\mathbf{a}(\mathbf{a}^T\mathbf{a})\mathbf{a}^T}{(\mathbf{a}^T\mathbf{a})^2} = \mathbf{P}$, 대칭성은 $(\mathbf{a}\mathbf{a}^T)^T = \mathbf{a}\mathbf{a}^T$에서 나온다. 계수가 1이므로 $\operatorname{tr}(\mathbf{P}) = 1$이다.

    $\mathbf{P}\mathbf{y} = \mathbf{a}\dfrac{\mathbf{a}^T\mathbf{y}}{\mathbf{a}^T\mathbf{a}}$로 쓰면 계수가 바로 읽힌다.

    $$
    \hat{\beta} = \frac{\mathbf{a}^T\mathbf{y}}{\mathbf{a}^T\mathbf{a}} = \frac{\sum_i a_i y_i}{\sum_i a_i^2}
    $$

    이것이 **절편 없는 단순회귀** $y_i = \beta a_i + \varepsilon_i$의 최소제곱추정량이다.

    ```python
    import numpy as np

    a = np.array([1., 2., 2.])
    y = np.array([3., 1., 4.])

    P = np.outer(a, a) / (a @ a)
    print("P =\n", P.round(4))
    print("멱등:", np.allclose(P @ P, P), " 대칭:", np.allclose(P, P.T),
          " tr:", round(np.trace(P), 6))
    print("beta_hat =", round(a @ y / (a @ a), 6))
    print("P y =", (P @ y).round(6), " = beta_hat * a =", (a @ y / (a @ a) * a).round(6))
    ```

    출력:

    ```
    P =
     [[0.1111 0.2222 0.2222]
     [0.2222 0.4444 0.4444]
     [0.2222 0.4444 0.4444]]
    멱등: True  대칭: True  tr: 1.0
    beta_hat = 1.444444
    P y = [1.444444 2.888889 2.888889]  = beta_hat * a = [1.444444 2.888889 2.888889]
    ```

    벡터 하나 위로의 사영은 앞으로 계속 쓰인다. 중심화행렬의 여집합 $\frac{1}{n}\mathbf{J}$가 $\mathbf{1}$ 위로의 사영이고, 그람–슈미트의 각 단계도 이 형태다. $\square$

<div class="drillbox" markdown>

**연습문제 6.**
모자 행렬의 대각 성분 $h_{ii}$를 **지렛대**라 한다. $0 \le h_{ii} \le 1$이고 $\sum_i h_{ii} = p$임을 보여라. $h_{ii} = 1$이면 무엇을 뜻하는가?

</div>

??? success "풀이"
    $\mathbf{H}$가 대칭 멱등이므로 $\mathbf{H} = \mathbf{H}^2 = \mathbf{H}^T\mathbf{H}$이고, 따라서

    $$
    h_{ii} = [\mathbf{H}^T\mathbf{H}]_{ii} = \sum_j h_{ji}^2 = h_{ii}^2 + \sum_{j \neq i} h_{ji}^2
    $$

    이다. 오른쪽의 합이 음이 아니므로 $h_{ii} \ge h_{ii}^2$, 곧 $h_{ii}(1 - h_{ii}) \ge 0$이고 $0 \le h_{ii} \le 1$이다. 합은 $\sum_i h_{ii} = \operatorname{tr}(\mathbf{H}) = p$다.

    **$h_{ii} = 1$인 경우.** 위 식에서 $j \neq i$인 모든 $h_{ji} = 0$이어야 한다. 그러면 $\hat{y}_i = \sum_j h_{ij}y_j = y_i$이므로 **그 관측값은 정확히 맞춰진다.** 잔차가 항상 0이고, 그 점은 회귀선을 자기 쪽으로 완전히 끌어당긴다.

    ```python
    import numpy as np

    rng = np.random.default_rng(0)
    n = 40
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    h = np.diag(H)

    print(f"지렛대 최소 {h.min():.4f}, 최대 {h.max():.4f}")
    print(f"합 = {h.sum():.6f}  (= p = {X.shape[1]})")
    print(f"평균 = {h.mean():.4f}  (= p/n = {X.shape[1]/n:.4f})")
    ```

    출력:

    ```
    지렛대 최소 0.0285, 최대 0.2371
    합 = 3.000000  (= p = 3)
    평균 = 0.0750  (= p/n = 0.0750)
    ```

    지렛대의 평균이 $p/n$이므로 **$2p/n$을 넘는 점을 주의해서 보라**는 실무 규칙이 여기서 나온다. 2장의 이상치·지렛대점 논의와 이어진다. $\square$

<div class="drillbox" markdown>

**연습문제 7.**
$\operatorname{col}(\mathbf{X}_1) \subseteq \operatorname{col}(\mathbf{X}_2)$이고 각각의 사영을 $\mathbf{H}_1$, $\mathbf{H}_2$라 하자. $\mathbf{H}_2\mathbf{H}_1 = \mathbf{H}_1\mathbf{H}_2 = \mathbf{H}_1$임을 보여라.

</div>

??? success "풀이"
    임의의 $\mathbf{x}$에 대해 $\mathbf{H}_1\mathbf{x} \in \operatorname{col}(\mathbf{X}_1) \subseteq \operatorname{col}(\mathbf{X}_2)$이다. 사영은 자기 치역의 벡터를 그대로 두므로 $\mathbf{H}_2(\mathbf{H}_1\mathbf{x}) = \mathbf{H}_1\mathbf{x}$이고, 따라서 $\mathbf{H}_2\mathbf{H}_1 = \mathbf{H}_1$이다.

    두 행렬 모두 대칭이므로 전치를 취하면

    $$
    \mathbf{H}_1 = \mathbf{H}_1^T = (\mathbf{H}_2\mathbf{H}_1)^T = \mathbf{H}_1^T\mathbf{H}_2^T = \mathbf{H}_1\mathbf{H}_2
    $$

    이다.

    **F 검정의 근거.** 이 성질에서 $\mathbf{H}_2 - \mathbf{H}_1$도 사영임이 따라 나온다.

    $$
    (\mathbf{H}_2 - \mathbf{H}_1)^2 = \mathbf{H}_2 - \mathbf{H}_2\mathbf{H}_1 - \mathbf{H}_1\mathbf{H}_2 + \mathbf{H}_1 = \mathbf{H}_2 - \mathbf{H}_1
    $$

    게다가 $(\mathbf{H}_2 - \mathbf{H}_1)(\mathbf{I} - \mathbf{H}_2) = \mathbf{O}$이므로 두 사영이 직교한다. 그래서 큰 모형과 작은 모형의 제곱합 차이 $\lVert(\mathbf{H}_2 - \mathbf{H}_1)\mathbf{y}\rVert^2$과 잔차제곱합 $\lVert(\mathbf{I}-\mathbf{H}_2)\mathbf{y}\rVert^2$이 (정규성 아래에서) 독립인 카이제곱이 되고, 그 비가 $F$ 분포를 따른다. **내포모형 F 검정의 기하가 바로 이것이다.** $\square$

<div class="drillbox" markdown>

**연습문제 8.**
부분공간 $\mathcal{V}$가 주어지면 그 위로의 직교사영행렬은 **유일**함을 보여라.

</div>

??? success "풀이"
    $\mathbf{P}_1$과 $\mathbf{P}_2$가 모두 $\mathcal{V}$ 위로의 직교사영이라 하자. 연습문제 7의 논법을 양방향으로 쓴다. 두 치역이 같으므로 $\mathbf{P}_2\mathbf{P}_1 = \mathbf{P}_1$이고 $\mathbf{P}_1\mathbf{P}_2 = \mathbf{P}_2$다.

    대칭성에서

    $$
    \mathbf{P}_1 = (\mathbf{P}_2\mathbf{P}_1)^T = \mathbf{P}_1^T\mathbf{P}_2^T = \mathbf{P}_1\mathbf{P}_2 = \mathbf{P}_2
    $$

    이므로 둘은 같다.

    **왜 중요한가.** 사영 공식 $\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$은 겉보기에 $\mathbf{X}$에 의존하지만, 실제로는 **$\operatorname{col}(\mathbf{X})$에만 의존한다.** 같은 열공간을 주는 다른 계획행렬(예: 예측변수를 재척도화하거나 선형결합한 것)을 써도 모자 행렬은 똑같다. 회귀에서 적합값 $\hat{\mathbf{y}}$과 $R^2$이 모수화 방식에 영향받지 않는 이유가 이것이다. 반면 계수 $\hat{\boldsymbol{\beta}}$은 모수화에 따라 달라진다.

    빗각 사영에서는 이 유일성이 성립하지 않는다. 치역이 같아도 눌러 없애는 방향이 다르면 다른 사영이다. $\square$

<div class="drillbox" markdown>

**연습문제 9.**
$\mathbf{M}_1 = \mathbf{I} - \mathbf{H}_1$을 $\mathbf{X}_1$에 대한 잔차생성행렬이라 하자. $\mathbf{y}$를 $[\mathbf{X}_1, \mathbf{x}_2]$에 회귀했을 때 $\mathbf{x}_2$의 계수가, $\mathbf{M}_1\mathbf{y}$를 $\mathbf{M}_1\mathbf{x}_2$에 회귀한 계수와 같음을 수치로 확인하라.

</div>

??? success "풀이"
    이것이 **Frisch–Waugh–Lovell 정리**다. 다중회귀의 한 계수는 "다른 변수들의 영향을 걷어낸 뒤"의 단순회귀 계수와 같다.

    ```python
    import numpy as np

    rng = np.random.default_rng(0)
    n = 40
    x1 = np.ones(n)
    x2 = rng.normal(size=n)
    x3 = 0.5 * x2 + rng.normal(size=n)          # x2 와 상관된 변수
    y = 1 + 2 * x2 - 1.5 * x3 + rng.normal(size=n)

    # (1) 전체 다중회귀
    X = np.column_stack([x1, x2, x3])
    beta_full = np.linalg.lstsq(X, y, rcond=None)[0]

    # (2) x1, x2 의 영향을 걷어낸 뒤 단순회귀
    X1 = np.column_stack([x1, x2])
    M1 = np.eye(n) - X1 @ np.linalg.inv(X1.T @ X1) @ X1.T
    y_res, x3_res = M1 @ y, M1 @ x3
    beta_fwl = (x3_res @ y_res) / (x3_res @ x3_res)

    print("다중회귀의 x3 계수 :", round(beta_full[2], 8))
    print("FWL 로 구한 계수   :", round(beta_fwl, 8))
    ```

    출력:

    ```
    다중회귀의 x3 계수 : -1.44939995
    FWL 로 구한 계수   : -1.44939995
    ```

    두 값이 소수점 여덟째 자리까지 같다.

    **해석적 의미가 크다.** 다중회귀의 계수 $\hat\beta_3$은 "$x_3$이 한 단위 늘 때 $y$의 변화"가 아니라 "**다른 변수로 설명되지 않는 부분의** $x_3$이 한 단위 늘 때, **다른 변수로 설명되지 않는 부분의** $y$의 변화"다. 이것이 다중회귀 계수를 "다른 변수를 통제했을 때의 효과"로 읽는 근거이며, 13장 편회귀그림의 바탕이기도 하다. $\square$

<div class="drillbox" markdown>

**연습문제 10.**
계획행렬 하나를 잡아 $\mathbf{H}$의 성질(대칭·멱등·대각합), $\hat{\mathbf{y}} \perp \mathbf{e}$, 피타고라스 분해, 그리고 $R^2 = \cos^2\theta$(연습문제 4)를 모두 수치로 확인하라.

</div>

??? success "풀이"
    ```python
    import numpy as np

    rng = np.random.default_rng(1)
    n, p = 30, 3
    X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
    y = X @ np.array([1., 2., -1.]) + rng.normal(size=n)

    H = X @ np.linalg.inv(X.T @ X) @ X.T
    y_hat = H @ y
    e = y - y_hat

    print("대칭   :", np.allclose(H, H.T))
    print("멱등   :", np.allclose(H @ H, H))
    print("tr(H)  :", round(np.trace(H), 6), " (= p =", p, ")")
    print("y_hat . e :", round(float(y_hat @ e), 12), " (직교)")

    print("\n||y||^2          =", round(y @ y, 6))
    print("||y_hat||^2 + ||e||^2 =", round(y_hat @ y_hat + e @ e, 6))

    cos2 = (y_hat @ y_hat) / (y @ y)
    print("\ncos^2(theta) =", round(cos2, 6))
    print("R^2 (원점 기준) =", round((y_hat @ y_hat) / (y @ y), 6))
    ```

    출력:

    ```
    대칭   : True
    멱등   : True
    tr(H)  : 3.0  (= p = 3 )
    y_hat . e : -0.0  (직교)

    ||y||^2          = 158.337856
    ||y_hat||^2 + ||e||^2 = 158.337856

    cos^2(theta) = 0.885583
    R^2 (원점 기준) = 0.885583
    ```

    네 성질이 모두 확인된다. 잔차와 적합값의 내적은 반올림 오차 수준에서 0이다.

    **주의.** 여기서 계산한 $R^2 = \lVert\hat{\mathbf{y}}\rVert^2/\lVert\mathbf{y}\rVert^2$은 **원점을 기준으로 한** 값이다. 보고되는 통상의 $R^2$은 평균을 빼고 계산한 $1 - \lVert\mathbf{e}\rVert^2/\lVert\mathbf{y} - \bar{y}\mathbf{1}\rVert^2$이며, 이는 $\mathbf{1}$ 위로의 사영을 먼저 걷어낸 뒤 같은 논리를 적용한 것이다. 절편이 있는 모형에서 두 값이 다르다는 점에 유의하라. $\square$

