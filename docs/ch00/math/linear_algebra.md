# 선형대수 표기와 관례

선형대수는 다변량 통계의 언어다. 회귀, 다변량 분석, 차원축소, 그리고 현대 추정 이론에 등장하는 거의 모든 양이 벡터나 행렬 표현이다. 이 절은 책 전체에서 쓰는 표기를 정하고, 가장 자주 되풀이되는 연산과 항등식을 복습한다.

## 정의

### 벡터와 행렬

벡터는 열벡터 $\mathbf{x} \in \mathbb{R}^n$이며 굵은 소문자로 쓴다. 행렬은 $\mathbf{A} \in \mathbb{R}^{m \times n}$이며 굵은 대문자로 쓴다. 전치는 $\mathbf{A}^T$, 역행렬은(존재할 때) $\mathbf{A}^{-1}$로 나타낸다. 단위행렬은 $\mathbf{I}_n$, 영행렬은 $\mathbf{0}$이다.

### 핵심 연산

$$
\mathbf{x}^T \mathbf{y} = \sum_{i=1}^n x_i y_i, \qquad \|\mathbf{x}\| = \sqrt{\mathbf{x}^T \mathbf{x}}, \qquad [\mathbf{A}\mathbf{B}]_{ij} = \sum_{\ell} a_{i\ell} b_{\ell j}
$$

**외적** $\mathbf{x} \mathbf{y}^T \in \mathbb{R}^{n \times m}$은 계수가 많아야 1이다. **대각합**은 $\mathrm{tr}(\mathbf{A}) = \sum_i a_{ii}$이다. **행렬식** $\det(\mathbf{A})$는 부호를 가진 부피 배율을 나타내며, $\mathbf{A}$가 가역일 때에 한해 0이 아니다.

### 계획행렬

**계획행렬(design matrix)** $\mathbf{X} \in \mathbb{R}^{n \times p}$는 $p$개 예측변수에 대한 $n$개의 관측을 행 방향으로 쌓은 것이다. $i$번째 행에는 관측 $i$의 예측변수 값이 들어 있다. $\mathbf{X}$의 열공간은 $p$개 열의 모든 선형결합으로 이루어진 집합, 즉 최소제곱으로 도달할 수 있는 적합값의 공간이다.

### 계수, 영공간, 네 가지 기본 부분공간

$\mathbf{A} \in \mathbb{R}^{m \times n}$에 대해:

- $\mathrm{Col}(\mathbf{A}) \subseteq \mathbb{R}^m$ (열공간)
- $\mathrm{Null}(\mathbf{A}) \subseteq \mathbb{R}^n$ (우영공간)
- $\mathrm{Col}(\mathbf{A}^T) \subseteq \mathbb{R}^n$ (행공간)
- $\mathrm{Null}(\mathbf{A}^T) \subseteq \mathbb{R}^m$ (좌영공간)

이고, $\mathrm{rank}(\mathbf{A}) + \dim \mathrm{Null}(\mathbf{A}) = n$(계수–퇴화차수 정리)이며 $\mathrm{Col}(\mathbf{A}^T) \perp \mathrm{Null}(\mathbf{A})$이다.

## 설명

### 이 책에서 끊임없이 쓰이는 항등식

- $(\mathbf{A}\mathbf{B})^T = \mathbf{B}^T \mathbf{A}^T$
- $(\mathbf{A}\mathbf{B})^{-1} = \mathbf{B}^{-1} \mathbf{A}^{-1}$
- $\mathrm{tr}(\mathbf{A}\mathbf{B}) = \mathrm{tr}(\mathbf{B}\mathbf{A})$ (순환 성질)
- 정사각 $\mathbf{A}$에 대해 $\mathrm{tr}(\mathbf{A}) = \sum_i \lambda_i(\mathbf{A})$, $\det(\mathbf{A}) = \prod_i \lambda_i(\mathbf{A})$
- $\mathbf{X}$가 확률벡터일 때 $\mathrm{Cov}(\mathbf{A} \mathbf{X}) = \mathbf{A}\, \mathrm{Cov}(\mathbf{X})\, \mathbf{A}^T$
- $\mathbb{E}[\mathbf{X}^T \mathbf{A} \mathbf{X}] = \mathrm{tr}(\mathbf{A}\, \mathrm{Cov}(\mathbf{X})) + \boldsymbol{\mu}^T \mathbf{A} \boldsymbol{\mu}$ (이차형식의 기댓값)

### 고윳값과 스펙트럼 정리

$\mathbf{A} \in \mathbb{R}^{n \times n}$에 대해 $\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$가 고윳값–고유벡터 쌍을 정의한다. **대칭** $\mathbf{A}$(모든 공분산행렬, 모든 $\mathbf{X}^T \mathbf{X}$, 모든 모자 행렬이 여기 해당한다)에 대해 **스펙트럼 정리**는

$$
\mathbf{A} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^T
$$

을 보장한다. 여기서 $\mathbf{Q}$는 직교행렬($\mathbf{Q}^T \mathbf{Q} = \mathbf{I}$)이고 $\boldsymbol{\Lambda}$는 실수 고윳값으로 이루어진 대각행렬이다. 이것이 주성분분석, 다변량 정규 이론, 이차형식의 카이제곱분포를 떠받치는 원동력이다.

### 양(반)정치성

모든 $\mathbf{x}$에 대해 $\mathbf{x}^T \mathbf{A} \mathbf{x} \ge 0$이면 $\mathbf{A}$가 **양반정치(positive semidefinite)** 라 하고($\mathbf{A} \succeq 0$), 이는 모든 고윳값이 $\ge 0$인 것과 동치다. **양정치(positive definite)** ($\mathbf{A} \succ 0$)는 두 부등식을 모두 엄격 부등식으로 바꾼 것이다. 공분산행렬은 언제나 양반정치이며, 어떤 변수도 다른 변수들의 결정론적 선형결합이 아닐 때 양정치가 된다. 양정치성은 $(\mathbf{X}^T \mathbf{X})^{-1}$이 존재하고 최소제곱해가 유일하게 정해지기 위해 필요한 바로 그 조건이다.

### 사영과 모자 행렬

**최소제곱의 모자 행렬(hat matrix)**

$$
\mathbf{H} = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T
$$

은 $\mathbf{y} \in \mathbb{R}^n$을 $\mathbf{X}$의 열공간 위로 직교사영한다. 이 행렬을 규정하는 성질은 다음과 같다.

- **대칭성**: $\mathbf{H}^T = \mathbf{H}$.
- **멱등성**: $\mathbf{H}^2 = \mathbf{H}$.
- **대각합 = 계수**: $\mathrm{tr}(\mathbf{H}) = p$ (모형의 자유도).
- **고윳값은 0 또는 1**: 1이 $p$개(열공간), 0이 $n - p$개(잔차공간).

여집합 사영자 $\mathbf{M} = \mathbf{I} - \mathbf{H}$는 잔차공간 위로 사영하며 $\mathrm{tr}(\mathbf{M}) = n - p$이다. 이것이 모든 $t$-검정과 $F$-검정에 등장하는 잔차 자유도다.

### 최소제곱을 위한 행렬 미적분

$\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$을 최소화하기 위해 기울기를 0으로 두면 정규방정식 $\mathbf{X}^T \mathbf{X} \boldsymbol{\beta} = \mathbf{X}^T \mathbf{y}$를 얻고, 따라서

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

이다. 이 하나의 공식과 그 뒤에 있는 사영 해석이 제13장 회귀 내용 전체의 바탕이 된다.

## 예제

```python
import numpy as np

rng = np.random.default_rng(42)
n, p = 50, 3
X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
beta_true = np.array([2.0, -1.0, 0.5])
y = X @ beta_true + rng.standard_normal(n) * 0.5

# === OLS via normal equations ===
beta_hat = np.linalg.solve(X.T @ X, X.T @ y)
print("True beta:", beta_true)
print("OLS beta: ", beta_hat.round(4))

# === Hat matrix properties ===
H = X @ np.linalg.inv(X.T @ X) @ X.T
print(f"Symmetric:  {np.allclose(H, H.T)}")
print(f"Idempotent: {np.allclose(H @ H, H)}")
print(f"tr(H) = {np.trace(H):.1f}  (should equal p = {p})")

# === Spectral decomposition of X'X (symmetric, PD here) ===
eigvals, eigvecs = np.linalg.eigh(X.T @ X)
print(f"Eigenvalues: {eigvals.round(2)}")
print(f"Condition number: {eigvals.max() / eigvals.min():.2f}")
reconstruction = eigvecs @ np.diag(eigvals) @ eigvecs.T
print(f"Spectral reconstruction matches X'X: {np.allclose(reconstruction, X.T @ X)}")
```

출력:

```
True beta: [ 2.  -1.   0.5]
OLS beta:  [ 1.9711 -0.95    0.4159]
Symmetric:  True
Idempotent: True
tr(H) = 3.0  (should equal p = 3)
Eigenvalues: [20.34 38.64 51.  ]
Condition number: 2.51
Spectral reconstruction matches X'X: True
```

## 연습문제

**연습문제 1.**
다음 계획행렬에 대해

$$
\mathbf{X} = \begin{pmatrix} 1 & 2 \\ 1 & 4 \\ 1 & 6 \end{pmatrix}, \qquad \mathbf{y} = \begin{pmatrix} 5 \\ 9 \\ 13 \end{pmatrix}
$$

**(a)** $\mathbf{X}^T \mathbf{X}$와 $\mathbf{X}^T \mathbf{y}$를 계산하라.
**(b)** 정규방정식을 풀어 $\hat{\boldsymbol{\beta}}$를 구하라.
**(c)** 적합값 $\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}}$와 잔차를 계산하라.

??? success "풀이"
    (a) $\mathbf{X}^T \mathbf{X} = \begin{pmatrix} 3 & 12 \\ 12 & 56 \end{pmatrix}$, $\mathbf{X}^T \mathbf{y} = \begin{pmatrix} 27 \\ 124 \end{pmatrix}$.

    (b) $\det(\mathbf{X}^T \mathbf{X}) = 168 - 144 = 24$이므로

    $$
    (\mathbf{X}^T \mathbf{X})^{-1} = \frac{1}{24}\begin{pmatrix} 56 & -12 \\ -12 & 3 \end{pmatrix}, \quad \hat{\boldsymbol{\beta}} = \frac{1}{24}\begin{pmatrix} 24 \\ 48 \end{pmatrix} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}
    $$

    (c) $\hat{\mathbf{y}} = (5, 9, 13)^T = \mathbf{y}$이고 잔차는 모두 0이다. 세 점이 한 직선 위에 있으므로 적합이 정확하다.

---

**연습문제 2.**
대각합의 순환 성질을 증명하라: $\mathbf{A} \in \mathbb{R}^{m \times n}$, $\mathbf{B} \in \mathbb{R}^{n \times m}$에 대해 $\mathrm{tr}(\mathbf{A}\mathbf{B}) = \mathrm{tr}(\mathbf{B}\mathbf{A})$.

??? success "풀이"
    직접 계산한다.

    $$
    \mathrm{tr}(\mathbf{A}\mathbf{B}) = \sum_{i=1}^m [\mathbf{A}\mathbf{B}]_{ii} = \sum_{i=1}^m \sum_{j=1}^n a_{ij} b_{ji} = \sum_{j=1}^n \sum_{i=1}^m b_{ji} a_{ij} = \sum_{j=1}^n [\mathbf{B}\mathbf{A}]_{jj} = \mathrm{tr}(\mathbf{B}\mathbf{A})
    $$

    순서를 바꾸는 데에는 합이 유한하다는 사실만 쓰였다. $\square$

---

**연습문제 3.**
$\mathbf{X} \in \mathbb{R}^{n \times p}$가 완전 열계수를 갖는다고 하자($\mathrm{rank}(\mathbf{X}) = p \le n$). $\mathbf{H} = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T$가 대칭이고 멱등이며 대각합이 $p$임을 증명하라.

??? success "풀이"
    **대칭성:** $(\mathbf{X}^T \mathbf{X})^{-1}$은 대칭이므로(대칭행렬의 역행렬)

    $$
    \mathbf{H}^T = \left(\mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T\right)^T = \mathbf{X}\left((\mathbf{X}^T \mathbf{X})^{-1}\right)^T \mathbf{X}^T = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T = \mathbf{H}
    $$

    **멱등성:**

    $$
    \mathbf{H}^2 = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1}\underbrace{\mathbf{X}^T \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1}}_{= \mathbf{I}_p}\mathbf{X}^T = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1}\mathbf{X}^T = \mathbf{H}
    $$

    **대각합:** 순환 성질에 의해

    $$
    \mathrm{tr}(\mathbf{H}) = \mathrm{tr}\!\left(\mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T\right) = \mathrm{tr}\!\left((\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{X}\right) = \mathrm{tr}(\mathbf{I}_p) = p
    $$

    $\square$

---

**연습문제 4.**
$\mathbf{A} \in \mathbb{R}^{n \times n}$이 대칭이고 스펙트럼 분해가 $\mathbf{A} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^T$라 하자. $\mathrm{tr}(\mathbf{A}) = \sum_i \lambda_i$이고 $\det(\mathbf{A}) = \prod_i \lambda_i$임을 증명하라.

??? success "풀이"
    순환 성질과 $\mathbf{Q}^T \mathbf{Q} = \mathbf{I}$를 쓰면

    $$
    \mathrm{tr}(\mathbf{A}) = \mathrm{tr}(\mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^T) = \mathrm{tr}(\boldsymbol{\Lambda} \mathbf{Q}^T \mathbf{Q}) = \mathrm{tr}(\boldsymbol{\Lambda}) = \sum_i \lambda_i
    $$

    이다. 행렬식의 경우, $\det$는 곱셈적이고 직교행렬 $\mathbf{Q}$에 대해 $\det(\mathbf{Q}) = \pm 1$이므로

    $$
    \det(\mathbf{A}) = \det(\mathbf{Q}) \det(\boldsymbol{\Lambda}) \det(\mathbf{Q}^T) = \det(\mathbf{Q})^2 \prod_i \lambda_i = \prod_i \lambda_i
    $$

    이다. $\square$

---

**연습문제 5.**
$\mathbf{X}$가 평균 $\boldsymbol{\mu}$, 공분산 $\boldsymbol{\Sigma}$인 확률벡터라 하자. 대칭행렬 $\mathbf{A}$에 대해

$$
\mathbb{E}[\mathbf{X}^T \mathbf{A} \mathbf{X}] = \mathrm{tr}(\mathbf{A} \boldsymbol{\Sigma}) + \boldsymbol{\mu}^T \mathbf{A} \boldsymbol{\mu}
$$

임을 보여라.

??? success "풀이"
    $\mathbb{E}[\mathbf{Z}] = 0$이고 $\mathrm{Cov}(\mathbf{Z}) = \boldsymbol{\Sigma}$인 $\mathbf{Z}$를 써서 $\mathbf{X} = \boldsymbol{\mu} + \mathbf{Z}$로 놓자. 전개하면($\mathbf{A}$의 대칭성을 이용한다)

    $$
    \mathbf{X}^T \mathbf{A} \mathbf{X} = \boldsymbol{\mu}^T \mathbf{A} \boldsymbol{\mu} + 2 \boldsymbol{\mu}^T \mathbf{A} \mathbf{Z} + \mathbf{Z}^T \mathbf{A} \mathbf{Z}
    $$

    이다. 기댓값을 취하면 $\mathbb{E}[\mathbf{Z}] = 0$이므로 가운데 항이 사라진다. 마지막 항의 경우 $\mathbf{Z}^T \mathbf{A} \mathbf{Z}$가 스칼라이므로 자기 자신의 대각합과 같고,

    $$
    \mathbb{E}[\mathbf{Z}^T \mathbf{A} \mathbf{Z}] = \mathbb{E}[\mathrm{tr}(\mathbf{Z}^T \mathbf{A} \mathbf{Z})] = \mathbb{E}[\mathrm{tr}(\mathbf{A} \mathbf{Z} \mathbf{Z}^T)] = \mathrm{tr}(\mathbf{A}\, \mathbb{E}[\mathbf{Z} \mathbf{Z}^T]) = \mathrm{tr}(\mathbf{A} \boldsymbol{\Sigma})
    $$

    이다. 여기에 결정론적인 항을 더하면 결과를 얻는다. $\square$

---

**연습문제 6.**
$\mathbf{X}^T \mathbf{X}$가 특이행렬이면(즉 $\mathbf{X}$가 완전 열계수를 갖지 않으면) 최소제곱추정량 $\hat{\boldsymbol{\beta}}$이 유일하게 정해지지 않는 이유를 서로 보완적인 두 방식으로 설명하라. (a) 대수적으로, (b) 기하적으로.

??? success "풀이"
    **(a) 대수적으로:** 특이성은 $\det(\mathbf{X}^T \mathbf{X}) = 0$을 뜻하므로 $(\mathbf{X}^T \mathbf{X})^{-1}$이 존재하지 않는다. 정규방정식 $\mathbf{X}^T \mathbf{X} \boldsymbol{\beta} = \mathbf{X}^T \mathbf{y}$은 (우변이 $\mathbf{X}^T \mathbf{X}$의 열공간 안에 있으므로) 해를 갖지만 무한히 많다. $\boldsymbol{\beta}^*$가 해이고 $\mathbf{v} \in \mathrm{Null}(\mathbf{X})$이면 $\mathbf{X}\mathbf{v} = \mathbf{0}$이므로 $\boldsymbol{\beta}^* + \mathbf{v}$도 이 방정식을 만족한다.

    **(b) 기하적으로:** $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$는 여전히 $\mathbf{y}$를 $\mathrm{Col}(\mathbf{X})$ 위로 직교사영한 유일한 벡터다. 그러나 $\mathbf{X}$의 열들이 선형종속이면 그 사영을 열들의 선형결합으로 나타내는 방법이 무한히 많고, 각각이 타당한 $\hat{\boldsymbol{\beta}}$을 준다. 적합값은 식별되지만 계수는 식별되지 않는다. 해결책은 종속인 열을 제거하거나, 능형회귀($\mathbf{X}^T \mathbf{X}$에 $\lambda \mathbf{I}$를 더해 가역성을 회복한다), 또는 유사역행렬(최소 노름 해를 준다)을 쓰는 것이다. $\square$

---

**연습문제 7.**
계획행렬

$$
\mathbf{X} = \begin{pmatrix} 1 & 2 & 3 \\ 1 & 3 & 4 \\ 1 & 4 & 5 \\ 1 & 5 & 6 \end{pmatrix}
$$

에 대해 네 가지 기본 부분공간의 차원을 구하고, 계수–퇴화차수 정리와 $\mathrm{Col}(\mathbf{X}^T) \perp \mathrm{Null}(\mathbf{X})$을 확인하라. 이 자료로 회귀를 돌리면 어떤 일이 일어나는가?

??? success "풀이"
    셋째 열이 첫째와 둘째 열의 합이므로 $\mathrm{rank}(\mathbf{X}) = 2$이다.

    | 부분공간 | 차원 | 사는 곳 |
    |---|---|---|
    | $\mathrm{Col}(\mathbf{X})$ | $2$ | $\mathbb{R}^4$ |
    | $\mathrm{Null}(\mathbf{X})$ | $3 - 2 = 1$ | $\mathbb{R}^3$ |
    | $\mathrm{Col}(\mathbf{X}^T)$ (행공간) | $2$ | $\mathbb{R}^3$ |
    | $\mathrm{Null}(\mathbf{X}^T)$ | $4 - 2 = 2$ | $\mathbb{R}^4$ |

    ```python
    import numpy as np

    X = np.array([[1., 2., 3.],
                  [1., 3., 4.],
                  [1., 4., 5.],
                  [1., 5., 6.]])          # 3열 = 1열 + 2열

    r = np.linalg.matrix_rank(X)
    U, s, Vt = np.linalg.svd(X)
    print(f"rank = {r},  특잇값 = {s.round(6)}")

    null_basis = Vt[r:]                    # 0 특잇값에 딸린 오른쪽 특이벡터
    row_basis = Vt[:r]
    print(f"영공간 기저: {null_basis.round(4)}")
    print(f"X v = {(X @ null_basis.T).ravel().round(12)}")
    print(f"rank + nullity = {r} + {null_basis.shape[0]} = {r + null_basis.shape[0]} = 열 개수")
    print(f"행공간 ⟂ 영공간: {np.allclose(row_basis @ null_basis.T, 0)}")
    ```

    출력:

    ```
    rank = 2,  특잇값 = [11.982576  0.646436  0.      ]
    영공간 기저: [[-0.5774 -0.5774  0.5774]]
    X v = [0. 0. 0. 0.]
    rank + nullity = 2 + 1 = 3 = 열 개수
    행공간 ⟂ 영공간: True
    ```

    세 번째 특잇값이 정확히 $0$이라는 것이 계수 결손의 신호다. 영공간 기저 $\propto (1, 1, -1)$은 "첫째 열 더하기 둘째 열 빼기 셋째 열 = 0"이라는 종속관계를 그대로 읽어 준다.

    **회귀를 돌리면.** $\mathbf{X}^T\mathbf{X}$가 특이행렬이라 $\hat{\boldsymbol{\beta}}$이 유일하지 않다(연습문제 6). $\boldsymbol{\beta}^*$가 해이면 임의의 $c$에 대해 $\boldsymbol{\beta}^* + c(1,1,-1)^T$도 해이며, 넷 모두 **똑같은 적합값**을 준다. `np.linalg.lstsq`는 오류를 내지 않고 최소 노름 해 하나를 조용히 돌려주므로 더 위험하다. 계수의 부호나 크기를 해석하려 들면 아무 의미 없는 수를 해석하게 된다.

    실무에서 이 상황은 완전한 다중공선성으로 나타난다. 가장 흔한 원인은 범주형 변수를 원-핫 인코딩하면서 기준 범주를 빼지 않은 것이다(더미변수 함정). $\square$

---

**연습문제 8.**
$\mathrm{Cov}(\mathbf{A}\mathbf{X}) = \mathbf{A}\,\mathrm{Cov}(\mathbf{X})\,\mathbf{A}^T$를 증명하고, 이로부터 공분산행렬이 항상 양반정치임을 보여라. 세 자산 포트폴리오에 적용해 분산투자 효과를 확인하라.

??? success "풀이"
    $\boldsymbol{\mu} = \mathbb{E}[\mathbf{X}]$라 하면 $\mathbb{E}[\mathbf{A}\mathbf{X}] = \mathbf{A}\boldsymbol{\mu}$이므로

    $$
    \mathrm{Cov}(\mathbf{A}\mathbf{X}) = \mathbb{E}\!\left[\mathbf{A}(\mathbf{X}-\boldsymbol{\mu})(\mathbf{X}-\boldsymbol{\mu})^T\mathbf{A}^T\right]
    = \mathbf{A}\,\mathbb{E}\!\left[(\mathbf{X}-\boldsymbol{\mu})(\mathbf{X}-\boldsymbol{\mu})^T\right]\mathbf{A}^T
    = \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T
    $$

    이다. $\mathbf{A}$가 상수라 기댓값 밖으로 빠져나온 것이 전부다.

    **양반정치성.** $\mathbf{A}$ 자리에 행벡터 $\mathbf{c}^T$를 넣으면 좌변이 스칼라 확률변수의 분산이므로

    $$
    \mathbf{c}^T\boldsymbol{\Sigma}\mathbf{c} = \operatorname{Var}(\mathbf{c}^T\mathbf{X}) \ge 0
    $$

    이고, 이것이 모든 $\mathbf{c}$에 대해 성립한다. 즉 **공분산행렬의 양반정치성은 "분산은 음수가 될 수 없다"를 행렬로 옮겨 적은 것일 뿐이다.** 등호는 $\mathbf{c}^T\mathbf{X}$가 상수일 때, 곧 변수들 사이에 완전한 선형관계가 있을 때만 성립한다. $\square$

    ```python
    import numpy as np

    Sigma = np.array([[0.04,  0.006,  0.000],
                      [0.006, 0.09,  -0.012],
                      [0.000, -0.012, 0.16]])

    w = np.full(3, 1 / 3)
    print(f"개별 표준편차:        {np.sqrt(np.diag(Sigma)).round(3)}")
    print(f"개별 표준편차의 평균: {np.sqrt(np.diag(Sigma)).mean():.4f}")
    print(f"동일가중 포트폴리오:   분산 {w @ Sigma @ w:.6f}, 표준편차 {np.sqrt(w @ Sigma @ w):.4f}")

    Si = np.linalg.inv(Sigma)
    one = np.ones(3)
    w_mv = Si @ one / (one @ Si @ one)          # 최소분산 포트폴리오
    print(f"\n최소분산 가중치: {w_mv.round(4)}")
    print(f"최소분산 표준편차: {np.sqrt(w_mv @ Sigma @ w_mv):.4f}")
    ```

    출력:

    ```
    개별 표준편차:        [0.2 0.3 0.4]
    개별 표준편차의 평균: 0.3000
    동일가중 포트폴리오:   분산 0.030889, 표준편차 0.1758

    최소분산 가중치: [0.5721 0.2561 0.1718]
    최소분산 표준편차: 0.1563
    ```

    동일가중 포트폴리오의 표준편차 $0.1758$은 개별 표준편차의 평균 $0.3$보다 훨씬 작다. **이것이 분산투자다.** 비대각 성분이 작거나 음수라서 $\mathbf{w}^T\boldsymbol{\Sigma}\mathbf{w}$가 대각 성분의 가중평균보다 작아지는 것이다.

    최소분산 포트폴리오는 여기서 더 나아가 $0.1563$까지 낮추는데, 변동성이 가장 낮은 첫 자산에 가장 큰 비중($0.57$)을 준다. 상관이 완전하다면($\rho = 1$) 비대각 성분이 커져 이런 이득이 사라진다. 위기 때 상관이 $1$로 몰리면서 분산투자 효과가 증발하는 현상이 바로 이 계산에 들어 있다.

---

**연습문제 9.**
행렬 미적분으로 정규방정식을 유도하라. 필요한 기울기 규칙 $\nabla_{\mathbf{x}}(\mathbf{a}^T\mathbf{x}) = \mathbf{a}$와 $\nabla_{\mathbf{x}}(\mathbf{x}^T\mathbf{A}\mathbf{x}) = 2\mathbf{A}\mathbf{x}$($\mathbf{A}$ 대칭)를 먼저 증명하고, 헤세행렬을 확인해 얻은 점이 정말 최소점임을 보여라.

??? success "풀이"
    **기울기 규칙.** $\mathbf{a}^T\mathbf{x} = \sum_j a_j x_j$이므로 $\partial/\partial x_k = a_k$, 곧 $\nabla = \mathbf{a}$이다. 이차형식은 $\mathbf{x}^T\mathbf{A}\mathbf{x} = \sum_{i,j}a_{ij}x_ix_j$이므로

    $$
    \frac{\partial}{\partial x_k} = \sum_j a_{kj}x_j + \sum_i a_{ik}x_i = [\mathbf{A}\mathbf{x}]_k + [\mathbf{A}^T\mathbf{x}]_k
    $$

    이고, $\mathbf{A}$가 대칭이면 $2\mathbf{A}\mathbf{x}$가 된다.

    **유도.** 목적함수를 전개하면

    $$
    S(\boldsymbol{\beta}) = \|\mathbf{y}-\mathbf{X}\boldsymbol{\beta}\|^2
    = \mathbf{y}^T\mathbf{y} - 2\boldsymbol{\beta}^T\mathbf{X}^T\mathbf{y} + \boldsymbol{\beta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}
    $$

    이다($\mathbf{y}^T\mathbf{X}\boldsymbol{\beta}$가 스칼라라 전치와 같으므로 두 교차항이 합쳐졌다). 위의 두 규칙을 쓰면

    $$
    \nabla S = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{0}
    \quad\Longrightarrow\quad
    \mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}
    $$

    **최소점 확인.** 헤세행렬은 $\nabla^2 S = 2\mathbf{X}^T\mathbf{X}$이고, 임의의 $\mathbf{v} \ne \mathbf{0}$에 대해

    $$
    \mathbf{v}^T(\mathbf{X}^T\mathbf{X})\mathbf{v} = \|\mathbf{X}\mathbf{v}\|^2 \ge 0
    $$

    이다. $\mathbf{X}$가 완전 열계수이면 $\mathbf{X}\mathbf{v} \ne \mathbf{0}$이라 엄격한 양정치가 되어 $S$가 순볼록이고, 정류점은 유일한 전역 최소점이다. $\square$

    ```python
    import numpy as np

    rng = np.random.default_rng(1)
    n, p = 40, 3
    X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
    y = X @ np.array([1., 2., -1.]) + rng.normal(size=n)

    beta = np.array([0.3, -0.7, 1.1])          # 아무 점에서나
    S = lambda b: ((y - X @ b) ** 2).sum()

    grad_analytic = -2 * X.T @ y + 2 * X.T @ X @ beta
    h = 1e-6
    grad_numeric = np.array([(S(beta + h * e) - S(beta - h * e)) / (2 * h)
                             for e in np.eye(p)])
    print("해석적 기울기:", grad_analytic.round(6))
    print("수치적 기울기:", grad_numeric.round(6))

    hess = 2 * X.T @ X
    print(f"\n헤세행렬 고윳값: {np.linalg.eigvalsh(hess).round(3)}  (모두 > 0)")
    print(f"정류점 = OLS 해: "
          f"{np.allclose(np.linalg.solve(X.T @ X, X.T @ y), np.linalg.lstsq(X, y, rcond=None)[0])}")
    ```

    출력:

    ```
    해석적 기울기: [ -69.943347 -130.877197   77.167931]
    수치적 기울기: [ -69.943347 -130.877197   77.167931]

    헤세행렬 고윳값: [44.717 64.702 84.593]  (모두 > 0)
    정류점 = OLS 해: True
    ```

    수치미분이 해석적 기울기와 소수점 여섯째 자리까지 맞는다. 새 손실함수를 유도했을 때 이런 대조는 언제나 해 볼 만한 값싼 점검이다.

---

**연습문제 10.**
**(하나 빼기 항등식)** 관측 $i$를 빼고 적합한 회귀의 예측오차가

$$
y_i - \mathbf{x}_i^T\hat{\boldsymbol{\beta}}_{(i)} = \frac{e_i}{1 - h_{ii}}
$$

임이 알려져 있다. 여기서 $e_i$는 통상 잔차이고 $h_{ii}$는 지렛값이다. 이 공식을 수치로 확인하고, $h_{ii}$가 $1$에 가까우면 어떤 일이 벌어지는지 설명하라.

??? success "풀이"
    ```python
    import numpy as np

    rng = np.random.default_rng(7)
    n, p = 12, 3
    X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
    y = rng.normal(size=n)

    XtXi = np.linalg.inv(X.T @ X)
    H = X @ XtXi @ X.T
    e = y - X @ (XtXi @ X.T @ y)
    h = np.diag(H)

    loo = np.empty(n)                       # 무식하게: n 번 다시 적합
    for i in range(n):
        keep = np.ones(n, bool)
        keep[i] = False
        loo[i] = y[i] - X[i] @ np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]

    print(f"공식과 실제 재적합의 최대 차이: {np.abs(loo - e / (1 - h)).max():.2e}")
    print(f"지렛값 합 = {h.sum():.4f}  (= p = {p})")
    print(f"\nSSE   = {(e ** 2).sum():.4f}   (같은 자료로 평가)")
    print(f"PRESS = {(loo ** 2).sum():.4f}   (빼놓은 자료로 평가)")
    ```

    출력:

    ```
    공식과 실제 재적합의 최대 차이: 4.55e-15
    지렛값 합 = 3.0000  (= p = 3)

    SSE   = 7.3015   (같은 자료로 평가)
    PRESS = 13.1653   (빼놓은 자료로 평가)
    ```

    **왜 중요한가.** $n$번 다시 적합하는 대신 **한 번의 적합에서 나온 $e_i$와 $h_{ii}$만으로** 하나 빼기 교차검증을 끝낼 수 있다. 회귀에서 LOOCV가 사실상 공짜인 이유다.

    **$h_{ii} \to 1$일 때.** 분모가 $0$으로 가면서 예측오차가 폭발한다. 뜻은 이렇다. 관측 $i$가 계획공간에서 워낙 외따로 떨어져 있어 그 점을 빼면 **그 위치를 짚어 줄 다른 관측이 없다.** 실제로 $h_{ii} = 1$이면 회귀선이 그 점을 정확히 통과해 $e_i = 0$이 되고, 모형은 그 점에 대해 아무것도 검증하지 못한다.

    이 때문에 $h_{ii} > 2p/n$인 관측을 높은 지렛점으로 보아 따로 살핀다. 쿡의 거리도 $e_i$와 $h_{ii}$를 같은 방식으로 결합해 만든다.

    위 출력에서 PRESS가 SSE의 두 배쯤 된다는 점도 눈여겨보라. 같은 자료로 적합하고 같은 자료로 평가하면 성능이 낙관적으로 나온다는 사실이 한 줄로 드러난다. $\square$

