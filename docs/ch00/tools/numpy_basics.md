# NumPy 배열

## 개요

**NumPy**(Numerical Python)는 파이썬 수치 계산의 토대가 되는 라이브러리다. 그 핵심은 `ndarray`로, 연속된 메모리에 저장되고 C, 포트란, 그리고 (가능한 경우) BLAS/LAPACK으로 작성된 루틴으로 다루어지는 동질적이고 크기가 고정된 다차원 배열이다. NumPy에 힘을 실어주는 성질은 세 가지다.

1. **벡터화**: 산술 연산이 배열 전체에 한 번에 적용되므로 파이썬 반복문이 사라진다.
2. **브로드캐스팅**: 모양이 다른 배열들이 자동 규칙으로 맞춰져, 표준화를 `(X - X.mean(axis=0)) / X.std(axis=0)`처럼 간결하게 쓸 수 있다.
3. **메모리 지역성**: `float64` 100만 개짜리 1차원 배열은 연속된 8MB 메모리를 차지하므로 벡터 연산이 캐시 친화적이다.

pandas, SciPy, scikit-learn, statsmodels, Matplotlib 등 모든 과학용 파이썬 라이브러리가 `ndarray`를 자료 교환 형식으로 삼아 그 위에 세워져 있다. 따라서 NumPy를 익히는 것이 나머지 생태계를 쓰기 위한 선수 조건이다.

```python
import numpy as np
```

## 배열 만들기

### 파이썬 리스트로부터

```python
# 1-D array
a = np.array([1, 2, 3, 4, 5])
print(a)          # [1 2 3 4 5]
print(a.shape)    # (5,)
print(a.dtype)    # int64

# 2-D array (matrix)
M = np.array([[1, 2, 3],
              [4, 5, 6]])
print(M.shape)    # (2, 3)
```

### 내장 생성자로

```python
np.zeros((3, 4))          # 3×4 matrix of zeros
np.ones((2, 2))           # 2×2 matrix of ones
np.full((3, 3), 7)        # 3×3 matrix filled with 7
np.eye(4)                 # 4×4 identity matrix
np.arange(0, 10, 2)       # array([0, 2, 4, 6, 8])
np.linspace(0, 1, 5)      # array([0., 0.25, 0.5, 0.75, 1.])
```

`arange`는 파이썬의 `range`를 본뜬 것이다(간격 기반이라 끝점을 지나칠 수 있다). `linspace`는 끝점을 포함하는 구간에 정해진 개수의 점을 고르게 배치하므로, 어떤 정의역 위에서 함수를 그릴 때는 보통 이쪽이 낫다.

### 난수 배열 (현대적 API)

예전의 `np.random.*` 함수도 여전히 작동하지만, NumPy 1.17에서 도입된 **`Generator`** API가 선호된다. 더 빠르고, 병렬 스트림을 지원하며, 전역 상태 변경으로부터 상태를 격리한다.

```python
rng = np.random.default_rng(seed=42)         # reproducible generator
rng.standard_normal((3, 3))                  # 3×3 standard normal draws
rng.uniform(0, 1, size=(2, 5))               # 2×5 Uniform(0,1) draws
rng.integers(0, 10, size=6)                  # 6 ints in [0, 10)
```

## 배열 속성

| 속성 | 설명 | 예 |
|---|---|---|
| `a.shape` | 차원의 모양 | `(2, 3)` |
| `a.ndim` | 차원의 수 | `2` |
| `a.size` | 전체 원소 개수 | `6` |
| `a.dtype` | 원소의 자료형 | `float64` |
| `a.nbytes` | 바이트 단위 메모리 사용량 | `48` |

`a.shape`의 곱은 언제나 `a.size`와 같다. `a.nbytes`는 `a.size * a.dtype.itemsize`와 같다.

## 인덱싱과 슬라이싱

### 1차원

```python
a = np.array([10, 20, 30, 40, 50])

a[0]        # 10        — first element
a[-1]       # 50        — last element
a[1:4]      # [20 30 40] — slice (start inclusive, stop exclusive)
a[::2]      # [10 30 50] — every other element
a[::-1]     # [50 40 30 20 10] — reversed
```

### 2차원

```python
M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

M[0, 1]       # 2          — row 0, col 1
M[1, :]       # [4 5 6]    — entire row 1
M[:, 2]       # [3 6 9]    — entire col 2
M[:2, :2]     # [[1 2],    — upper-left 2×2 sub-matrix
              #  [4 5]]
```

!!! note "뷰와 복사본"
    기본 슬라이싱(`a[1:4]`, `M[:2, :2]`)은 같은 메모리를 가리키는 **뷰**를 반환하므로, 슬라이스를 수정하면 원본이 바뀐다. 불리언 인덱싱과 팬시 인덱싱은 **복사본**을 반환한다. 헷갈릴 때는 `arr.copy()`로 의도를 분명히 하라.

### 불리언(팬시) 인덱싱

```python
a = np.array([3, 1, 4, 1, 5, 9])

mask = a > 3
print(mask)       # [False False  True False  True  True]
print(a[mask])    # [4 5 9]

# Combine masks with bit-wise &, |, ~
print(a[(a > 2) & (a < 6)])    # [3 4 5]
```

불리언 인덱싱은 반복문 없이 자료를 걸러내는 자연스러운 방법이다.

## 벡터화 연산

NumPy는 명시적 반복문 없이 원소별 산술을 수행하며, 이는 순수 파이썬보다 빠르면서도 읽기 좋다.

```python
a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])

a + 10        # [11 12 13 14 15]
a * 2         # [ 2  4  6  8 10]
a ** 2        # [ 1  4  9 16 25]
a + b         # [11 22 33 44 55]
a * b         # [ 10  40  90 160 250]
np.sqrt(a)    # [1.    1.414 1.732 2.    2.236]
```

### 왜 더 빠른가

```python
import time

size = 1_000_000
py_list = list(range(size))
np_arr  = np.arange(size)

t0 = time.perf_counter()
[x ** 2 for x in py_list]
print(f"Python list: {time.perf_counter() - t0:.4f} s")

t0 = time.perf_counter()
np_arr ** 2
print(f"NumPy array: {time.perf_counter() - t0:.4f} s")
```

배열 연산에서 NumPy는 보통 **10–100배 빠르다**. 안쪽 반복문이 C로 되어 있고 자료가 연속으로 저장되어 SIMD 명령과 캐시 친화적 접근이 가능하기 때문이다.

## 브로드캐스팅

브로드캐스팅은 중간 복사본을 만들지 않고 모양이 다른 배열끼리 산술 연산을 수행하는 NumPy의 기제다. 규칙은 다음과 같다.

1. 배열의 차원 수가 다르면 작은 쪽 모양 앞에 `1`을 채워 넣는다.
2. 어떤 차원에서 두 모양이 같거나 **또는** 둘 중 하나가 1이면 그 차원은 호환된다. 브로드캐스트 결과는 더 큰 크기를 갖는다.

```python
# Scalar broadcast (shape () broadcasts to anything)
a = np.array([1, 2, 3])
a + 100            # [101 102 103]

# Column vector (3,1) + row vector (3,) → (3,3)
col = np.array([[1], [2], [3]])
row = np.array([10, 20, 30])
print(col + row)
# [[11 21 31]
#  [12 22 32]
#  [13 23 33]]
```

### 통계적 응용: 표준화

```python
rng = np.random.default_rng(42)
X = rng.standard_normal((100, 5))            # (n=100) × (p=5)

X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
print(X_std.mean(axis=0).round(8))           # ≈ zeros
print(X_std.std(axis=0, ddof=1).round(8))    # ≈ ones
```

브로드캐스팅이 없다면 열마다 반복문을 돌려야 한다. 브로드캐스팅이 있으면 "각 열을 중심화한 뒤 그 표준편차로 나눈다"는 통계적 아이디어가 코드로 그대로 옮겨진다.

## 집계

```python
a = np.array([4, 1, 7, 3, 9, 2])

a.sum()        # 26
a.mean()       # 4.333...
a.std()        # 2.687...  (default ddof=0)
a.std(ddof=1)  # 2.943...  (Bessel-corrected sample std)
a.var()        # 7.222...
a.min()        # 1
a.max()        # 9
a.argmin()     # 1   — index of min
a.argmax()     # 4   — index of max
np.median(a)   # 3.5
```

### 축을 따라 집계하기

2차원 배열에서 `axis=0`은 *행*을 접어 열별 결과를 주고, `axis=1`은 *열*을 접어 행별 결과를 준다.

```python
M = np.array([[1, 2, 3],
              [4, 5, 6]])

M.sum(axis=0)    # [5 7 9]   — column sums
M.sum(axis=1)    # [6 15]    — row sums
M.mean(axis=0)   # [2.5 3.5 4.5]
```

!!! warning "기본값은 `ddof=0`"
    NumPy의 `var`와 `std`는 기본값이 `ddof=0`이다($n$으로 나누는 모분산). 베셀 보정을 적용한 표본분산은 `ddof=1`로 $n - 1$로 나눈다. pandas의 기본값은 `ddof=1`이다. 한 분석 안에서 둘을 섞어 쓰는 것은 미묘한 하나 차이 버그의 고전적인 원천이다.

## 선형대수

```python
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[2, 3],
              [0, 1]])

A @ B                  # preferred — matrix multiplication
np.matmul(A, B)
np.dot(A, B)           # identical for 2-D arrays

A.T                    # transpose
np.linalg.det(A)       # -2.0
np.linalg.inv(A)       # inverse (use solve() when possible)
np.linalg.eig(A)       # eigenvalues + eigenvectors
np.linalg.eigh(A)      # for symmetric/Hermitian — faster and stable

b = np.array([5, 11])
np.linalg.solve(A, b)  # [1. 2.] — solves Ax = b
```

### 통계적 응용: 최소제곱

최소제곱추정량 $\hat{\boldsymbol\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$는 코드로 그대로 옮겨진다.

```python
rng = np.random.default_rng(0)
n, p = 50, 3
X = np.column_stack([np.ones(n), rng.standard_normal((n, p))])
beta_true = np.array([2, -1, 0.5, 3])
y = X @ beta_true + rng.standard_normal(n) * 0.5

beta_hat = np.linalg.solve(X.T @ X, X.T @ y)
print(beta_hat.round(3))     # ≈ [2., -1., 0.5, 3.]
```

`np.linalg.inv(X.T @ X) @ X.T @ y`보다 `np.linalg.solve(X.T @ X, X.T @ y)`를 쓰라. 역행렬을 만드는 것보다 방정식을 푸는 편이 수치적으로 더 안정적이고 빠르다. 더 나은 선택은 `np.linalg.lstsq(X, y, rcond=None)`으로, 계수가 부족한 $\mathbf{X}$도 특이값분해로 처리한다.

## 모양 바꾸기와 쌓기

```python
a = np.arange(12)
M = a.reshape(3, 4)        # 3×4 view
flat = M.ravel()           # back to 1-D view

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
np.vstack([v1, v2])        # [[1 2 3], [4 5 6]]
np.hstack([v1, v2])        # [1 2 3 4 5 6]
np.column_stack([v1, v2])  # [[1 4], [2 5], [3 6]]
```

## 요약

| 개념 | 핵심 |
|---|---|
| `ndarray` | 동질적이고 크기가 고정된 $N$차원 연속 메모리 블록 |
| 벡터화 | 명시적 반복문을 배열 수준의 표현식으로 대체 |
| 브로드캐스팅 | 두 개의 단순한 규칙에 따른 자동 모양 확장 |
| 집계 | `axis` 매개변수를 갖는 `sum`, `mean`, `std`, `var`, `ddof`에 유의 |
| 선형대수 | 행렬곱은 `@`, `inv`보다 `solve`, 대칭행렬에는 `eigh` |
| 난수 생성 | `default_rng(seed)`가 재현 가능한 현대적 인터페이스 |

## 연습문제

**연습문제 1.**
다음에 대해 최소제곱 정규방정식의 해를 손으로, 그리고 NumPy로 확인하라.

$$
\mathbf{X} = \begin{pmatrix} 1 & 2 \\ 1 & 4 \\ 1 & 6 \end{pmatrix}, \qquad \mathbf{y} = (5, 9, 13)^T
$$

$\mathbf{X}^T\mathbf{X}$, $\mathbf{X}^T\mathbf{y}$, $\hat{\boldsymbol\beta}$, 그리고 잔차 벡터를 계산하라.

??? success "풀이"
    ```python
    import numpy as np

    X = np.array([[1, 2], [1, 4], [1, 6]])
    y = np.array([5, 9, 13])

    XtX = X.T @ X
    Xty = X.T @ y
    beta_hat = np.linalg.solve(XtX, Xty)
    y_hat = X @ beta_hat
    resid = y - y_hat

    print("X^T X =", XtX, sep="\n")
    print("X^T y =", Xty)
    print("beta_hat =", beta_hat)
    print("residuals =", resid)
    ```

    기대되는 결과: $\hat{\boldsymbol\beta} = (1, 2)^T$이고 잔차는 모두 0이다(세 점이 완전히 한 직선 위에 있다).

---

**연습문제 2.**
파이썬 `for` 반복문 없이 $1000 \times 5$ 크기의 표준정규 난수 행렬을 만들고, 각 열을 표본평균 0, 표본분산 1이 되도록 표준화하라. `mean(axis=0)`과 `var(axis=0, ddof=1)`로 확인하라.

??? success "풀이"
    ```python
    rng = np.random.default_rng(42)
    X = rng.standard_normal((1000, 5))
    X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    print(X_std.mean(axis=0).round(8))           # ≈ zeros
    print(X_std.var(axis=0, ddof=1).round(8))    # ≈ ones
    ```

    브로드캐스팅이 열 평균의 행벡터(모양 `(5,)`)를 `X`(모양 `(1000, 5)`)에서 빼고, 마찬가지로 열 표준편차의 행벡터로 나눈다.

---

**연습문제 3.**
`X`의 행들 사이의 유클리드 거리로 이루어진 $n \times n$ 행렬을 반환하는 벡터화된 함수 `pairwise_distances(X)`를 작성하라. 브로드캐스팅과 `np.sqrt`만 쓰고 명시적 반복문은 쓰지 마라.

??? success "풀이"
    ```python
    def pairwise_distances(X):
        # X has shape (n, p). diff has shape (n, n, p) after broadcasting:
        # X[:, None, :] is (n, 1, p), X[None, :, :] is (1, n, p)
        diff = X[:, None, :] - X[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1))

    X = np.array([[0, 0], [3, 4], [6, 8]])
    print(pairwise_distances(X))
    # [[ 0.  5. 10.]
    #  [ 5.  0.  5.]
    #  [10.  5.  0.]]
    ```

    두 번의 브로드캐스트 단계가 크기 1인 차원을 끼워 넣어 각 행을 다른 모든 행과 짝지어 준다. 결과는 대각이 0인 대칭행렬이다.

---

**연습문제 4.**
`np.linalg.solve(A, b)`가 `np.linalg.inv(A) @ b`보다 더 정확한 결과를 주는 이유는 무엇인가? $\mathbf{A}$가 거의 특이인 예를 만들어 두 답을 비교하라.

??? success "풀이"
    `solve`는 $\mathbf{A}$를 한 번 분해하고(부분 피벗을 쓰는 LU 분해) $\mathbf{b}$에 대해 후진 대입을 수행할 뿐, $\mathbf{A}^{-1}$을 명시적으로 만들지 않는다. 역행렬을 만들면 모든 성분에 $1/\det(\mathbf{A})$가 곱해지므로 $\det(\mathbf{A})$가 작을 때 반올림 오차가 증폭된다. 또한 역슬래시 방식의 루틴은 불필요한 $O(n^3)$ 행렬곱을 피한다.

    ```python
    A = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-12]])  # near-singular
    b = np.array([2.0, 2.0 + 1e-12])

    x_solve = np.linalg.solve(A, b)
    x_inv = np.linalg.inv(A) @ b
    print("solve:", x_solve)
    print("inv:  ", x_inv)
    ```

    둘 다 $(1, 1)^T$에 가까워야 하지만 `inv` 쪽 오차가 눈에 띄게 크다. 진짜로 특이인 행렬에서는 `solve`가 예외를 일으키는 반면 `inv`는 쓰레기 값을 돌려줄 수 있다.

---

**연습문제 5.**
모양이 $(5, 1)$과 $(1, 3)$인 두 배열이 있다. 이들의 원소별 곱의 모양은 무엇인가? 모양이 $(5,)$와 $(3,)$이라면 어떻게 되는가? 연산이 되는가?

??? success "풀이"
    브로드캐스팅 규칙을 따르면, 모양을 오른쪽에 맞춰 정렬하고 크기가 1인 차원을 확장한다. $(5, 1)$과 $(1, 3)$은 $(5, 3)$으로 브로드캐스트된다. 곱하면 외적 형태의 $5 \times 3$ 행렬이 나온다.

    $(5,)$와 $(3,)$의 경우, 오른쪽에 맞춰 정렬하면 마지막 축의 크기가 각각 $5$와 $3$이다. 둘이 같지도 않고 어느 쪽도 1이 아니므로 브로드캐스팅이 **실패하여** `ValueError`가 난다. $5 \times 3$ 외적을 얻으려면 축을 명시적으로 끼워 넣어야 한다: `a[:, None] * b[None, :]` 또는 `np.outer(a, b)`.

---

**연습문제 6.**
기본값인 `np.var(x)`는 $n$으로 나누고 `np.var(x, ddof=1)`은 $n - 1$로 나눈다. `x`가 i.i.d. 표본일 때 $\mathrm{Var}(X)$의 불편추정량은 어느 쪽인가? $N(0, 1)$에서 크기 $n = 5$인 표본을 $10^4$번 뽑아 반복에 걸친 `var(ddof=0)`과 `var(ddof=1)`의 평균을 비교하여 편향을 실증적으로 보여라.

??? success "풀이"
    `ddof=1`이 불편이다. $n - 1$로 나누면 $\mathbb{E}[S^2] = \sigma^2$이다. `ddof=0`은 $\sigma^2$을 $(n-1)/n$배만큼 과소추정한다.

    ```python
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((10_000, 5))
    print("Mean of var(ddof=0):", samples.var(axis=1, ddof=0).mean())  # ≈ 0.80
    print("Mean of var(ddof=1):", samples.var(axis=1, ddof=1).mean())  # ≈ 1.00
    ```

    $n = 5$일 때 모분산 방식의 분모는 평균적으로 $\approx 4/5 = 0.8$을 내놓는데, 이는 예측된 편향 계수와 정확히 일치한다. 베셀 보정을 적용한 쪽은 예상대로 $1.0$ 근처를 맴돈다. 이 편향은 $n$이 작을 때 가장 중요하며, $n$이 수천이면 차이는 무시할 만하다.
