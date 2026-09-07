# 대칭행렬

$\mathbf{A} = \mathbf{A}^T$를 만족하는 대칭행렬은 통계에서 가장 중요한 단일 행렬 부류다. 모든 공분산행렬이 대칭이다. 모든 모자 행렬이 대칭이다. 정규방정식에 등장하는 그람 행렬 $\mathbf{X}^T\mathbf{X}$도 대칭이다. 스펙트럼 정리는 대칭행렬이 실수 고윳값과 정규직교 고유기저를 가짐을 보장하며, 따라서 직교변환으로 대각화된다. 이 특수한 구조가 주성분분석, 이차형식의 카이제곱분포, 신뢰타원체의 기하학을 떠받친다.

## 정의

!!! info "정의 — 대칭행렬"
    정사각행렬 $\mathbf{A} \in \mathbb{R}^{n \times n}$이

    $$
    \mathbf{A} = \mathbf{A}^T
    $$

    를 만족하면 **대칭**이라 한다. 동등하게 모든 $i, j$에 대해 $a_{ij} = a_{ji}$이다.

대칭행렬은 대각과 그 위쪽 성분만으로 결정된다. $n^2$개의 성분 중 $n(n+1)/2$개만 자유롭다.

## 스펙트럼 정리

스펙트럼 정리는 대칭행렬에 관한 가장 중요한 결과다.

!!! tip "정리 — 스펙트럼 정리 (실대칭행렬)"
    $\mathbf{A} \in \mathbb{R}^{n \times n}$이 대칭이라 하자. 그러면

    1. $\mathbf{A}$의 모든 고윳값이 **실수**다.
    2. **서로 다른** 고윳값에 대응하는 고유벡터는 **직교**한다.
    3. $\mathbf{A}$는 **스펙트럼 분해**

    $$
    \mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T
    $$

    를 갖는다. 여기서 $\mathbf{Q}$는 정규직교 고유벡터를 열로 갖는 직교행렬($\mathbf{Q}^T\mathbf{Q} = \mathbf{I}$)이고 $\boldsymbol{\Lambda} = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$이다.

### 증명 개요 (고윳값이 실수임)

$\lambda \in \mathbb{C}$가 고유벡터 $\mathbf{v} \ne \mathbf{0}$을 갖는 고윳값이라 하자. 그러면 $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$이다. $\mathbf{A}$가 실대칭이므로 켤레전치를 취하면

$$
\overline{\mathbf{v}}^T \mathbf{A} = \overline{\lambda}\, \overline{\mathbf{v}}^T
$$

이다. 오른쪽에 $\mathbf{v}$를 곱하면

$$
\lambda\, \overline{\mathbf{v}}^T \mathbf{v} = \overline{\mathbf{v}}^T \mathbf{A}\mathbf{v} = \overline{\lambda}\, \overline{\mathbf{v}}^T \mathbf{v}
$$

이다. $\overline{\mathbf{v}}^T \mathbf{v} = \|\mathbf{v}\|^2 > 0$이므로 $\lambda = \overline{\lambda}$, 즉 $\lambda \in \mathbb{R}$이다. $\square$

### 증명 개요 (고유벡터의 직교성)

$\alpha \ne \beta$에 대해 $\mathbf{A}\mathbf{u} = \alpha\mathbf{u}$, $\mathbf{A}\mathbf{v} = \beta\mathbf{v}$라 하자. 그러면

$$
\alpha\, \mathbf{u}^T\mathbf{v} = (\mathbf{A}\mathbf{u})^T \mathbf{v} = \mathbf{u}^T \mathbf{A}^T \mathbf{v} = \mathbf{u}^T \mathbf{A}\mathbf{v} = \beta\, \mathbf{u}^T \mathbf{v}
$$

이다. 따라서 $(\alpha - \beta)\mathbf{u}^T\mathbf{v} = 0$이고 $\alpha \ne \beta$이므로 $\mathbf{u}^T\mathbf{v} = 0$이다. $\square$

중복 고윳값의 경우 각 고유공간 안에서 그람–슈미트를 적용하면 정규직교기저를 얻는다. 이 기저들을 이어 붙이면 $\mathbf{Q}$가 만들어진다.

## 외적 형태

스펙트럼 분해를 열 단위로 쓰면 **외적 형태**를 얻는다.

$$
\mathbf{A} = \sum_{i=1}^n \lambda_i\, \mathbf{q}_i \mathbf{q}_i^T
$$

각 $\mathbf{q}_i \mathbf{q}_i^T$는 $\mathbf{q}_i$ 위로의 계수 1인 직교사영자다. 대칭행렬은 고윳값으로 가중된 1차원 조각들로 지어진다. 주성분분석에서 공분산행렬을 주성분들의 합으로 제시하는 것과 같은 발상이다.

## 성질

### 직교대각화

실행렬이 **직교**대각화 가능할($\mathbf{Q}$가 직교행렬인 $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$) 필요충분조건은 그것이 대칭인 것이다. 이는 대각화 가능성보다 엄격히 강한 조건이다. 대칭이 아닌 많은 행렬이 대각화 가능하지만, *직교* 대각화를 허용하는 것은 대칭행렬뿐이다.

### 역행렬과 행렬 함수

$\mathbf{A}$가 대칭이고 가역이면 $\mathbf{A}^{-1}$도 대칭이다: $(\mathbf{A}^{-1})^T = (\mathbf{A}^T)^{-1} = \mathbf{A}^{-1}$. 스펙트럼 분해를 쓰면

$$
\mathbf{A}^k = \mathbf{Q}\boldsymbol{\Lambda}^k\mathbf{Q}^T, \quad f(\mathbf{A}) = \mathbf{Q}\operatorname{diag}\!\bigl(f(\lambda_1), \dots, f(\lambda_n)\bigr)\mathbf{Q}^T
$$

이다. 특히 모든 $\lambda_i \ge 0$일 때 $\mathbf{A}^{1/2} = \mathbf{Q}\operatorname{diag}(\sqrt{\lambda_1}, \dots, \sqrt{\lambda_n})\mathbf{Q}^T$가 유일한 대칭 양반정치 제곱근이다.

### 레일리 몫

고윳값이 $\lambda_1 \le \cdots \le \lambda_n$인 대칭행렬 $\mathbf{A}$에 대해 **레일리 몫** $R(\mathbf{x}) = \mathbf{x}^T \mathbf{A}\mathbf{x} / \mathbf{x}^T\mathbf{x}$는

$$
\lambda_1 = \min_{\mathbf{x} \ne \mathbf{0}} R(\mathbf{x}), \qquad \lambda_n = \max_{\mathbf{x} \ne \mathbf{0}} R(\mathbf{x})
$$

를 만족하며, 극값은 대응하는 고유벡터에서 달성된다. 이것이 주성분 유도를 이끄는 변분적 특성화다.

### 이차형식

변수변환 $\mathbf{z} = \mathbf{Q}^T \mathbf{x}$는 임의의 이차형식을 대각화한다.

$$
\mathbf{x}^T \mathbf{A}\mathbf{x} = \mathbf{z}^T \boldsymbol{\Lambda}\mathbf{z} = \sum_{i=1}^n \lambda_i z_i^2
$$

회전된 독립 좌표들의 가중 제곱합이며, 정규벡터 이차형식의 카이제곱분포로 건너가는 다리다.

## 예

$$
\boldsymbol{\Sigma} = \begin{pmatrix} 5 & 2 \\ 2 & 2 \end{pmatrix}
$$

특성다항식: $(5 - \lambda)(2 - \lambda) - 4 = \lambda^2 - 7\lambda + 6 = (\lambda - 6)(\lambda - 1)$이므로 $\lambda_1 = 6$, $\lambda_2 = 1$이다.

정규화된 고유벡터: $\mathbf{q}_1 = (2, 1)^T / \sqrt{5}$, $\mathbf{q}_2 = (-1, 2)^T / \sqrt{5}$.

스펙트럼 분해:

$$
\boldsymbol{\Sigma} = \frac{1}{5}\begin{pmatrix} 2 & -1 \\ 1 & 2 \end{pmatrix}\begin{pmatrix} 6 & 0 \\ 0 & 1 \end{pmatrix}\begin{pmatrix} 2 & 1 \\ -1 & 2 \end{pmatrix}
$$

검산: $\operatorname{tr}(\boldsymbol{\Sigma}) = 7 = 6 + 1$이고 $\det(\boldsymbol{\Sigma}) = 6 = 6 \cdot 1$이다. 총분산은 7이고, 그중 6단위가 첫 번째 주축을 따라, 1단위가 두 번째 주축을 따라 몰려 있다.

## 통계와의 연결

### 공분산행렬

확률벡터 $\mathbf{X} \in \mathbb{R}^p$에 대해 $\boldsymbol{\Sigma} = \mathbb{E}[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^T]$는 대칭이고 양반정치다. 그 스펙트럼 분해가 주성분 방향과 각 방향의 분산을 정의한다.

### 정규방정식

$\mathbf{X}^T \mathbf{X}$는 대칭이다. 그 고윳값이 최소제곱해의 조건수를 좌우한다. 고윳값이 여러 자릿수에 걸쳐 퍼져 있으면(다중공선성) 해가 $\mathbf{y}$의 섭동에 민감해진다.

### 신뢰타원체

$\boldsymbol{\beta} \sim N(\hat{\boldsymbol{\beta}}, \boldsymbol{\Sigma})$일 때 수준집합 $\{\boldsymbol{\beta} : (\boldsymbol{\beta} - \hat{\boldsymbol{\beta}})^T \boldsymbol{\Sigma}^{-1} (\boldsymbol{\beta} - \hat{\boldsymbol{\beta}}) \le c\}$은 타원체이며, 그 축은 $\boldsymbol{\Sigma}$의 고유벡터 방향을 향하고 길이는 $\sqrt{\lambda_i}$에 비례한다.

## 요약

대칭행렬은 실수 고윳값과 직교하는 고유벡터를 가지며 직교대각화 $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$를 허용한다. 이 구조는 $\mathbf{A}$의 거듭제곱, 역행렬, 함수를 고윳값에 대한 스칼라 연산으로 환원한다. 공분산행렬, 그람 행렬, 사영행렬이 모두 대칭이므로, 스펙트럼 정리는 주성분분석과 이차형식과 회귀 이론의 일꾼이 된다.

## 연습문제

**연습문제 1.**
$\mathbf{A} = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$의 고윳값, 고유벡터, 스펙트럼 분해를 구하라. $\boldsymbol{\Sigma} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$를 수치적으로 확인하라.

??? success "풀이"
    특성방정식: $(2 - \lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = (\lambda - 1)(\lambda - 3) = 0$. 고윳값은 $\lambda_1 = 1, \lambda_2 = 3$이다.

    $\lambda_1 = 1$에 대해: $(\mathbf{A} - \mathbf{I})\mathbf{v} = \mathbf{0}$에서 $\mathbf{v}_1 = (1, -1)^T / \sqrt{2}$.
    $\lambda_2 = 3$에 대해: $(\mathbf{A} - 3\mathbf{I})\mathbf{v} = \mathbf{0}$에서 $\mathbf{v}_2 = (1, 1)^T / \sqrt{2}$.

    $$
    \mathbf{A} = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & 3 \end{pmatrix}\begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}
    $$

---

**연습문제 2.**
대칭행렬 $\mathbf{A}$에 대해 서로 다른 고윳값에 대응하는 고유벡터가 직교함을 증명하라.

??? success "풀이"
    $\alpha \ne \beta$에 대해 $\mathbf{A}\mathbf{u} = \alpha\mathbf{u}$, $\mathbf{A}\mathbf{v} = \beta\mathbf{v}$라 하자. $\mathbf{u}^T\mathbf{A}\mathbf{v}$를 두 가지 방식으로 계산한다.

    - $\mathbf{u}^T(\beta\mathbf{v}) = \beta\, \mathbf{u}^T\mathbf{v}$로 계산.
    - $\mathbf{A} = \mathbf{A}^T$를 써서 $(\mathbf{A}\mathbf{u})^T \mathbf{v} = (\alpha\mathbf{u})^T\mathbf{v} = \alpha\, \mathbf{u}^T\mathbf{v}$로 계산.

    따라서 $\alpha\, \mathbf{u}^T\mathbf{v} = \beta\, \mathbf{u}^T\mathbf{v}$, 즉 $(\alpha - \beta)\mathbf{u}^T\mathbf{v} = 0$이다. $\alpha \ne \beta$이므로 $\mathbf{u}^T\mathbf{v} = 0$이다. $\square$

---

**연습문제 3.**
$\mathbf{A}$가 스펙트럼 분해 $\mathbf{A} = \sum_i \lambda_i \mathbf{q}_i \mathbf{q}_i^T$를 갖는 대칭행렬이라 하자. $\mathbf{A}^2 = \sum_i \lambda_i^2 \mathbf{q}_i \mathbf{q}_i^T$임을 보이고, 이것이 왜 "$\mathbf{A}$가 멱등일 필요충분조건은 모든 고윳값이 0 또는 1인 것"임을 확인해 주는지 설명하라.

??? success "풀이"
    정규직교성 $\mathbf{q}_i^T \mathbf{q}_j = \delta_{ij}$를 쓰면

    $$
    \mathbf{A}^2 = \Bigl(\sum_i \lambda_i \mathbf{q}_i \mathbf{q}_i^T\Bigr)\Bigl(\sum_j \lambda_j \mathbf{q}_j \mathbf{q}_j^T\Bigr) = \sum_{i,j} \lambda_i \lambda_j (\mathbf{q}_i^T \mathbf{q}_j)\mathbf{q}_i \mathbf{q}_j^T = \sum_i \lambda_i^2 \mathbf{q}_i \mathbf{q}_i^T
    $$

    이다. $\mathbf{A}$가 멱등일 필요충분조건은 $\mathbf{A}^2 = \mathbf{A}$, 즉 모든 $i$에 대해 $\lambda_i^2 = \lambda_i$인 것이다. 동등하게 $\lambda_i \in \{0, 1\}$이다. $\square$

---

**연습문제 4.**
**레일리 몫.** $\mathbf{A}$가 최소·최대 고윳값이 $\lambda_\min, \lambda_\max$인 대칭행렬이라 하자. 모든 0이 아닌 $\mathbf{x} \in \mathbb{R}^n$에 대해

$$
\lambda_\min \le \frac{\mathbf{x}^T\mathbf{A}\mathbf{x}}{\mathbf{x}^T\mathbf{x}} \le \lambda_\max
$$

임을 증명하라.

??? success "풀이"
    $\mathbf{x}$를 고유기저로 전개한다: $c_i = \mathbf{q}_i^T \mathbf{x}$에 대해 $\mathbf{x} = \sum_i c_i \mathbf{q}_i$. 정규직교성을 쓰면

    $$
    \mathbf{x}^T\mathbf{A}\mathbf{x} = \sum_i \lambda_i c_i^2, \qquad \mathbf{x}^T\mathbf{x} = \sum_i c_i^2
    $$

    이다. 따라서 레일리 몫은 가중치 $c_i^2 / \sum_j c_j^2$를 갖는 고윳값들의 볼록결합이다. 수들의 볼록결합은 언제나 그 최솟값과 최댓값 사이에 있다.

    $$
    \lambda_\min = \lambda_\min \sum_i \frac{c_i^2}{\sum_j c_j^2} \le \sum_i \lambda_i \frac{c_i^2}{\sum_j c_j^2} \le \lambda_\max
    $$

    $\lambda_\min$에서 등호는 $\mathbf{x}$가 $\lambda_\min$의 고유벡터일 때 성립하고 $\lambda_\max$도 마찬가지다. $\square$

---

**연습문제 5.**
모든 대칭 양반정치행렬 $\mathbf{A}$가 $\mathbf{A}^{1/2} \mathbf{A}^{1/2} = \mathbf{A}$를 만족하는 유일한 대칭 양반정치 제곱근 $\mathbf{A}^{1/2}$을 가짐을 보여라.

??? success "풀이"
    **존재성:** $\lambda_i \ge 0$인 스펙트럼 분해 $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$로부터

    $$
    \mathbf{A}^{1/2} := \mathbf{Q}\operatorname{diag}(\sqrt{\lambda_1}, \dots, \sqrt{\lambda_n})\mathbf{Q}^T
    $$

    로 정의한다. 이것은 대칭이고(대칭 조각들의 곱 $\mathbf{Q}\mathbf{D}\mathbf{Q}^T$) 양반정치다($\sqrt{\lambda_i} \ge 0$). 직접 확인하면 $\mathbf{A}^{1/2}\mathbf{A}^{1/2} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T = \mathbf{A}$이다.

    **유일성:** $\mathbf{B}$가 $\mathbf{B}^2 = \mathbf{A}$인 대칭 양반정치행렬이라 하자. $\mu_i \ge 0$인 $\mathbf{M} = \operatorname{diag}(\mu_i)$로 $\mathbf{B} = \mathbf{Q}'\mathbf{M}\mathbf{Q}'^T$와 같이 대각화한다. 그러면 $\mathbf{B}^2 = \mathbf{Q}'\mathbf{M}^2 \mathbf{Q}'^T = \mathbf{A}$이므로 $\mathbf{B}$는 $\mu_i^2 = \lambda_i$인 고윳값과 $\mathbf{A}$의 고유벡터를 공유한다. $\mu_i \ge 0$이므로 $\mu_i = \sqrt{\lambda_i}$가 강제되어 위 공식이 복원된다. $\square$

    통계적 쓰임: **마할라노비스 백색화 변환** $\mathbf{Z} = \boldsymbol{\Sigma}^{-1/2}(\mathbf{X} - \boldsymbol{\mu})$는 $\boldsymbol{\Sigma}^{-1/2}$이 존재할 때 공분산이 단위행렬인 벡터를 만들어낸다.

---

**연습문제 6.**
대칭이 아니면서도 대각화 가능한 행렬의 예를 하나 들어라(즉 대칭성은 대각화 가능성의 충분조건이지 필요조건은 아니다). 그리고 *중복* 고윳값의 고유벡터에 그람–슈미트를 적용해 명시적인 정규직교기저를 얻는 대칭행렬의 예도 하나 들어라.

??? success "풀이"
    **대칭이 아니지만 대각화 가능:**

    $$
    \mathbf{A} = \begin{pmatrix} 1 & 1 \\ 0 & 2 \end{pmatrix}
    $$

    고윳값 $1, 2$가 서로 다르므로 고유벡터가 일차독립이고 $\mathbf{A}$는 대각화 가능하다. 그러나 $\mathbf{A} \ne \mathbf{A}^T$이므로 대칭은 아니다.

    **중복 고윳값을 갖는 대칭행렬:**

    $$
    \mathbf{A} = \begin{pmatrix} 2 & 0 & 0 \\ 0 & 1 & 1 \\ 0 & 1 & 1 \end{pmatrix}
    $$

    고윳값: $\lambda_1 = 2$(고유벡터 $(1,0,0)^T$), $\lambda_2 = 2$(고유벡터 $(0,1,1)^T/\sqrt{2}$), $\lambda_3 = 0$(고유벡터 $(0,1,-1)^T/\sqrt{2}$).

    고윳값 $2$는 중복도가 2이고 고유공간은 $\operatorname{span}\{(1,0,0)^T, (0,1,1)^T\}$이다. 이 두 벡터는 이미 직교하므로(그람–슈미트가 필요 없다) 정규화하면 정규직교기저를 얻는다. 세 번째 고유벡터와 함께 쌓으면 스펙트럼 정리가 약속한 직교행렬 $\mathbf{Q}$가 만들어진다.
