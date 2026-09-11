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

출력:

```
[1 2 3 4 5]
(5,)
int64
(2, 3)
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

출력:

```
[False False  True False  True  True]
[4 5 9]
[3 4 5]
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
t_list = time.perf_counter() - t0

t0 = time.perf_counter()
np_arr ** 2
t_numpy = time.perf_counter() - t0

# 초 단위 값은 기계마다 다르므로 출력에는 배수만 싣는다(아래 주의 참조).
print(f"NumPy가 최소 10배 이상 빠른가? {t_list > 10 * t_numpy}")
```

출력:

```
NumPy가 최소 10배 이상 빠른가? True
```

!!! note "왜 초 단위를 출력하지 않는가"
    이 비교의 요점은 "몇 초"가 아니라 **얼마나 빠른가**이다. `time.perf_counter()`가
    재는 값은 기계·부하·파이썬 버전에 따라 달라져 재현되지 않으므로, 초 단위 값은
    `t_list`와 `t_numpy`에 담아 두기만 하고 출력에서는 뺐다. 덕분에 이 블록의
    출력은 언제 실행해도 같다.

    참고로 이 책을 쓰며 여러 번 실행했을 때 `t_list`는 $0.05$–$0.09$초,
    `t_numpy`는 $0.001$–$0.003$초로 비는 대략 $30$배에서 $70$배 사이였다.
    직접 `print(t_list, t_numpy)`로 확인해 보라.

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

출력:

```
[[11 21 31]
 [12 22 32]
 [13 23 33]]
```

### 통계적 응용: 표준화

```python
rng = np.random.default_rng(42)
X = rng.standard_normal((100, 5))            # (n=100) × (p=5)

X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
print(X_std.mean(axis=0).round(8))           # ≈ zeros
print(X_std.std(axis=0, ddof=1).round(8))    # ≈ ones
```

출력:

```
[-0.  0. -0. -0. -0.]
[1. 1. 1. 1. 1.]
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

출력:

```
[ 1.95  -0.918  0.427  3.029]
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

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
다음에 대해 최소제곱 정규방정식의 해를 손으로, 그리고 NumPy로 확인하라.

$$
\mathbf{X} = \begin{pmatrix} 1 & 2 \\ 1 & 4 \\ 1 & 6 \end{pmatrix}, \qquad \mathbf{y} = (5, 9, 13)^T
$$

$\mathbf{X}^T\mathbf{X}$, $\mathbf{X}^T\mathbf{y}$, $\hat{\boldsymbol\beta}$, 그리고 잔차 벡터를 계산하라.

</div>

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

    출력:

    ```
    X^T X =
    [[ 3 12]
     [12 56]]
    X^T y = [ 27 124]
    beta_hat = [1. 2.]
    residuals = [0. 0. 0.]
    ```

    기대되는 결과: $\hat{\boldsymbol\beta} = (1, 2)^T$이고 잔차는 모두 0이다(세 점이 완전히 한 직선 위에 있다).

<div class="drillbox" markdown>

**연습문제 2.**
파이썬 `for` 반복문 없이 $1000 \times 5$ 크기의 표준정규 난수 행렬을 만들고, 각 열을 표본평균 0, 표본분산 1이 되도록 표준화하라. `mean(axis=0)`과 `var(axis=0, ddof=1)`로 확인하라.

</div>

??? success "풀이"
    ```python
    rng = np.random.default_rng(42)
    X = rng.standard_normal((1000, 5))
    X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    print(X_std.mean(axis=0).round(8))           # ≈ zeros
    print(X_std.var(axis=0, ddof=1).round(8))    # ≈ ones
    ```

    출력:

    ```
    [-0.  0.  0. -0. -0.]
    [1. 1. 1. 1. 1.]
    ```

    브로드캐스팅이 열 평균의 행벡터(모양 `(5,)`)를 `X`(모양 `(1000, 5)`)에서 빼고, 마찬가지로 열 표준편차의 행벡터로 나눈다.

<div class="drillbox" markdown>

**연습문제 3.**
`X`의 행들 사이의 유클리드 거리로 이루어진 $n \times n$ 행렬을 반환하는 벡터화된 함수 `pairwise_distances(X)`를 작성하라. 브로드캐스팅과 `np.sqrt`만 쓰고 명시적 반복문은 쓰지 마라.

</div>

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

    출력:

    ```
    [[ 0.  5. 10.]
     [ 5.  0.  5.]
     [10.  5.  0.]]
    ```

    두 번의 브로드캐스트 단계가 크기 1인 차원을 끼워 넣어 각 행을 다른 모든 행과 짝지어 준다. 결과는 대각이 0인 대칭행렬이다.

<div class="drillbox" markdown>

**연습문제 4.**
`np.linalg.solve(A, b)`가 `np.linalg.inv(A) @ b`보다 더 정확한 결과를 주는 이유는 무엇인가? $\mathbf{A}$가 거의 특이인 예를 만들어 두 답을 비교하라.

</div>

??? success "풀이"
    `solve`는 $\mathbf{A}$를 한 번 분해하고(부분 피벗을 쓰는 LU 분해) $\mathbf{b}$에 대해 후진 대입을 수행할 뿐, $\mathbf{A}^{-1}$을 명시적으로 만들지 않는다. 역행렬을 만들면 모든 성분에 $1/\det(\mathbf{A})$가 곱해지므로 $\det(\mathbf{A})$가 작을 때 반올림 오차가 증폭된다. 또한 역슬래시 방식의 루틴은 불필요한 $O(n^3)$ 행렬곱을 피한다.

    차이를 보려면 정말로 조건이 나쁜 행렬이 필요하다. 힐베르트 행렬 $H_{ij} = 1/(i+j-1)$이 표준적인 예다.

    ```python
    import numpy as np
    from scipy.linalg import hilbert

    for n in (10, 12, 14):
        A = hilbert(n)
        x_true = np.ones(n)
        b = A @ x_true                       # 정답이 (1,...,1) 이 되도록 만든다

        err_solve = np.abs(np.linalg.solve(A, b) - x_true).max()
        err_inv = np.abs(np.linalg.inv(A) @ b - x_true).max()
        print(f"n={n:>3}  조건수 {np.linalg.cond(A):.2e}   "
              f"solve 오차 {err_solve:.3e}   inv 오차 {err_inv:.3e}")
    ```

    출력:

    ```
    n= 10  조건수 1.60e+13   solve 오차 4.556e-04   inv 오차 2.376e-02
    n= 12  조건수 1.64e+16   solve 오차 3.233e-01   inv 오차 1.800e+01
    n= 14  조건수 2.43e+17   solve 오차 2.472e+01   inv 오차 5.714e+03
    ```

    $n = 12$에서 `inv` 쪽 오차가 `solve`의 약 $56$배다. $n = 14$에 이르면 두 방법 모두 답을 잃지만, 그때도 `inv`가 두 자릿수 더 나쁘다.

    조건수가 $10^{16}$을 넘으면 배정밀도의 상대정밀도($\approx 2\times10^{-16}$)를 다 써 버린 것이라 **어떤 알고리즘도 정확한 답을 줄 수 없다.** 이럴 때는 알고리즘을 바꾸는 대신 문제를 바꾸어야 한다. 회귀라면 변수를 중심화·척도화하거나, 능형 벌점을 넣거나, `np.linalg.lstsq`의 SVD 기반 절단을 쓰는 것이다.

    진짜로 특이인 행렬에서는 `solve`가 `LinAlgError` 를 일으켜 문제를 알려 주는 반면, `inv`는 경고만 내거나 쓰레기 값을 조용히 돌려줄 수 있다. **실패가 드러나는 쪽이 낫다.**

<div class="drillbox" markdown>

**연습문제 5.**
모양이 $(5, 1)$과 $(1, 3)$인 두 배열이 있다. 이들의 원소별 곱의 모양은 무엇인가? 모양이 $(5,)$와 $(3,)$이라면 어떻게 되는가? 연산이 되는가?

</div>

??? success "풀이"
    브로드캐스팅 규칙을 따르면, 모양을 오른쪽에 맞춰 정렬하고 크기가 1인 차원을 확장한다. $(5, 1)$과 $(1, 3)$은 $(5, 3)$으로 브로드캐스트된다. 곱하면 외적 형태의 $5 \times 3$ 행렬이 나온다.

    $(5,)$와 $(3,)$의 경우, 오른쪽에 맞춰 정렬하면 마지막 축의 크기가 각각 $5$와 $3$이다. 둘이 같지도 않고 어느 쪽도 1이 아니므로 브로드캐스팅이 **실패하여** `ValueError`가 난다. $5 \times 3$ 외적을 얻으려면 축을 명시적으로 끼워 넣어야 한다: `a[:, None] * b[None, :]` 또는 `np.outer(a, b)`.

<div class="drillbox" markdown>

**연습문제 6.**
기본값인 `np.var(x)`는 $n$으로 나누고 `np.var(x, ddof=1)`은 $n - 1$로 나눈다. `x`가 i.i.d. 표본일 때 $\mathrm{Var}(X)$의 불편추정량은 어느 쪽인가? $N(0, 1)$에서 크기 $n = 5$인 표본을 $10^4$번 뽑아 반복에 걸친 `var(ddof=0)`과 `var(ddof=1)`의 평균을 비교하여 편향을 실증적으로 보여라.

</div>

??? success "풀이"
    `ddof=1`이 불편이다. $n - 1$로 나누면 $\mathbb{E}[S^2] = \sigma^2$이다. `ddof=0`은 $\sigma^2$을 $(n-1)/n$배만큼 과소추정한다.

    ```python
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((10_000, 5))
    print("Mean of var(ddof=0):", samples.var(axis=1, ddof=0).mean())  # ≈ 0.80
    print("Mean of var(ddof=1):", samples.var(axis=1, ddof=1).mean())  # ≈ 1.00
    ```

    출력:

    ```
    Mean of var(ddof=0): 0.8003156389266377
    Mean of var(ddof=1): 1.000394548658297
    ```

    $n = 5$일 때 모분산 방식의 분모는 평균적으로 $\approx 4/5 = 0.8$을 내놓는데, 이는 예측된 편향 계수와 정확히 일치한다. 베셀 보정을 적용한 쪽은 예상대로 $1.0$ 근처를 맴돈다. 이 편향은 $n$이 작을 때 가장 중요하며, $n$이 수천이면 차이는 무시할 만하다.

<div class="drillbox" markdown>

**연습문제 7.**
NumPy에서 어떤 연산은 **뷰(view)** 를, 어떤 연산은 **복사본(copy)** 을 돌려준다. 슬라이싱과 팬시 인덱싱이 어느 쪽인지 확인하고, 이 차이가 만들어 내는 버그를 보여라. 어느 쪽인지 확실히 알아내는 방법은 무엇인가?

</div>

??? success "풀이"
    ```python
    import numpy as np

    a = np.arange(10)
    b = a[2:5]                 # 슬라이싱 → 뷰
    b[0] = 999
    print("슬라이스 수정 후 원본:", a)

    c = np.arange(10)
    d = c[[2, 3, 4]]           # 팬시 인덱싱 → 복사본
    d[0] = 999
    print("팬시 수정 후 원본:  ", c)

    e = np.arange(6).reshape(2, 3)
    print(f"\ne[0:1] 은 뷰인가? {e[0:1].base is not None}")
    print(f"e[[0]] 은 복사본인가? {e[[0]].base is None}")
    ```

    출력:

    ```
    슬라이스 수정 후 원본: [  0   1 999   3   4   5   6   7   8   9]
    팬시 수정 후 원본:   [0 1 2 3 4 5 6 7 8 9]

    e[0:1] 은 뷰인가? True
    e[[0]] 은 복사본인가? True
    ```

    **규칙.** 기본 슬라이싱(`a[2:5]`, `a[::2]`, `a.T`, `a.reshape(...)`)은 같은 메모리를 가리키는 뷰를 준다. 팬시 인덱싱(정수 배열이나 불리언 마스크)은 언제나 새 메모리를 할당한다.

    `.base` 속성이 판정 도구다. 뷰이면 원본 배열을 가리키고, 복사본이면 `None`이다.

    **어떤 버그가 생기는가.** 자료의 일부를 떼어 전처리한다고 하자.

    ```python
    import numpy as np

    data = np.arange(10.0)
    train = data[:7]           # 뷰!
    train -= train.mean()      # 제자리 연산이 원본까지 바꾼다
    print("원본이 오염되었다:", data)
    ```

    출력:

    ```
    원본이 오염되었다: [-3. -2. -1.  0.  1.  2.  3.  7.  8.  9.]
    ```

    `train -= ...`은 **제자리(in-place)** 연산이라 뷰가 가리키는 원본 메모리를 직접 고친다. 뒤에서 `data`로 검정 자료를 만들면 이미 오염된 값을 쓰게 되며, 오류 없이 조용히 틀린 결과가 나오므로 찾기가 어렵다.

    안전하게 쓰려면 의도를 명시하라. 원본을 지키려면 `train = data[:7].copy()`, 뷰를 쓸 때는 `train = train - train.mean()`처럼 새 배열을 만드는 형태로 쓴다. 뷰 자체는 결함이 아니라 큰 배열을 복사 없이 다루게 해 주는 기능이다. 위험한 것은 **뷰와 제자리 연산의 조합**이다. $\square$

<div class="drillbox" markdown>

**연습문제 8.**
불리언 마스킹과 팬시 인덱싱으로 (a) 조건부 평균, (b) `np.where`를 이용한 절단, (c) 반복문 없는 부트스트랩을 구현하라. 부트스트랩 표준오차를 이론값 $s/\sqrt{n}$과 비교하라.

</div>

??? success "풀이"
    ```python
    import numpy as np

    rng = np.random.default_rng(1)
    data = rng.gamma(2, 2, size=200)

    # (a) 불리언 마스킹
    mask = data > 6
    print(f"6 초과 개수 {mask.sum()},  그 조건부 평균 {data[mask].mean():.4f}")

    # (b) np.where 로 절단(winsorize)
    print(f"6 에서 자른 뒤 평균 {np.where(data > 6, 6, data).mean():.4f}"
          f"   (원래 평균 {data.mean():.4f})")

    # (c) 부트스트랩: (B, n) 인덱스 행렬을 한 번에 만든다
    B, n = 10_000, len(data)
    idx = rng.integers(0, n, size=(B, n))
    boot = data[idx].mean(axis=1)               # 팬시 인덱싱이 (B, n) 배열을 만든다

    print(f"\n부트스트랩 SE  {boot.std(ddof=1):.4f}")
    print(f"이론 SE s/sqrt(n) {data.std(ddof=1) / np.sqrt(n):.4f}")
    print(f"95% 백분위수 신뢰구간 {np.percentile(boot, [2.5, 97.5]).round(4)}")
    ```

    출력:

    ```
    6 초과 개수 28,  그 조건부 평균 8.2255
    6 에서 자른 뒤 평균 3.4430   (원래 평균 3.7545)

    부트스트랩 SE  0.1774
    이론 SE s/sqrt(n) 0.1760
    95% 백분위수 신뢰구간 [3.4231 4.1138]
    ```

    부트스트랩 표준오차 $0.1774$가 이론값 $0.1760$과 잘 맞는다. 감마분포는 오른쪽으로 치우쳐 있지만 $n = 200$이면 중심극한정리가 이미 충분히 작동한다.

    **핵심 기법은 (c)다.** `rng.integers(0, n, size=(B, n))`이 $B$번의 복원추출을 한꺼번에 만들고, `data[idx]`가 팬시 인덱싱으로 $(B, n)$ 배열을 채운 뒤 `axis=1` 평균이 $B$개의 통계량을 한 번에 준다. 파이썬 반복문이 하나도 없다.

    **메모리에 주의하라.** 이 방식은 $B \times n \times 8$바이트를 쓴다. 여기서는 $16$ MB로 괜찮지만 $n = 10^5$, $B = 10^4$이면 $8$ GB가 되어 터진다. 그럴 때는 부트스트랩을 덩어리로 나누어 돌린다.

    (b)의 절단이 평균을 $3.755$에서 $3.443$으로 끌어내린 것도 눈여겨보라. 오른쪽 꼬리를 자르면 치우친 분포의 평균이 눈에 띄게 내려간다. $\square$

<div class="drillbox" markdown>

**연습문제 9.**
분산을 계산하는 두 공식

$$
\text{(A)}\;\; \frac{1}{n}\sum x_i^2 - \bar{x}^2,
\qquad
\text{(B)}\;\; \frac{1}{n}\sum (x_i - \bar{x})^2
$$

은 수학적으로 같지만 부동소수점에서는 같지 않다. 자료에 큰 상수를 더해 가며 두 방법을 비교하고, 무슨 일이 일어나는지 설명하라.

</div>

??? success "풀이"
    ```python
    import numpy as np

    rng = np.random.default_rng(0)
    for offset in (0, 1e6, 1e8, 1e9):
        x = rng.normal(0, 1, 10_000) + offset
        naive = (x ** 2).mean() - x.mean() ** 2        # (A)
        two_pass = ((x - x.mean()) ** 2).mean()        # (B)
        print(f"offset {offset:>8.0e}:  (A) {naive:>18.10f}   (B) {two_pass:.10f}")
    ```

    출력:

    ```
    offset    0e+00:  (A)       0.9961574236   (B) 0.9961574236
    offset    1e+06:  (A)       0.9877929688   (B) 0.9880233132
    offset    1e+08:  (A)      -4.0000000000   (B) 0.9983070355
    offset    1e+09:  (A)     128.0000000000   (B) 1.0323651510
    ```

    자료를 평행이동해도 분산은 변하지 않아야 하는데, (A)는 무너진다. offset이 $10^8$일 때 **분산이 $-4$로 음수가 나오고**, $10^9$에서는 $128$이 된다. (B)는 내내 $1$ 근처를 지킨다.

    **원인은 상쇄(catastrophic cancellation)다.** offset이 $10^8$일 때

    $$
    \frac{1}{n}\sum x_i^2 \approx 10^{16}, \qquad \bar{x}^2 \approx 10^{16}
    $$

    이고 두 값의 차이는 $1$ 정도다. 배정밀도는 유효숫자를 약 $16$자리 갖는데, $10^{16}$ 크기의 수에서 마지막 유효숫자가 이미 $1$ 단위다. 즉 **답 전체가 반올림 오차 안에 잠긴다.** 크기가 거의 같은 두 큰 수를 뺄 때마다 일어나는 일이며, 유효숫자가 한꺼번에 날아간다.

    (B)는 먼저 빼기 때문에 $x_i - \bar{x}$가 $O(1)$이고, 제곱과 합산이 모두 작은 수 위에서 이루어진다.

    **실무 지침.**

    - 분산·공분산·회귀는 **언제나 중심화한 뒤** 계산하라. `np.var`는 내부적으로 (B)를 쓰므로 그냥 쓰면 된다.
    - 직접 구현할 일이 있으면 웰포드 알고리즘을 쓰라. 한 번만 훑으면서도 수치적으로 안정하다.
    - 이 문제는 자료의 **변동 대비 평균이 클 때**(연도, 타임스탬프, 큰 화폐 단위) 실제로 나타난다. 회귀에서 예측변수를 중심화하라는 조언은 해석의 편의만이 아니라 수치적 필요이기도 하다. $\square$

<div class="drillbox" markdown>

**연습문제 10.**
`np.einsum`을 이용해 중간 행렬 전체를 만들지 않고 (a) 마할라노비스 거리와 (b) 모자 행렬의 대각 성분 $h_{ii}$를 계산하라. 왜 이것이 중요한가?

</div>

??? success "풀이"
    두 양 모두 큰 행렬의 **대각 성분만** 필요로 한다. 행렬 전체를 만들었다가 대각만 꺼내는 것은 $n^2$개를 계산해 $n$개만 쓰는 낭비다.

    ```python
    import numpy as np

    rng = np.random.default_rng(1)
    n, p = 100, 4
    X = rng.normal(size=(n, p))

    # (a) 마할라노비스 거리
    mu, S = X.mean(0), np.cov(X.T)
    Si = np.linalg.inv(S)
    d_einsum = np.einsum('ij,jk,ik->i', X - mu, Si, X - mu)
    d_loop = np.array([(x - mu) @ Si @ (x - mu) for x in X])
    print(f"(a) 반복문과 일치: {np.allclose(d_einsum, d_loop)},  평균 {d_einsum.mean():.4f}  (≈ p = {p})")

    # (b) 지렛값
    XtXi = np.linalg.inv(X.T @ X)
    h_einsum = np.einsum('ij,jk,ik->i', X, XtXi, X)
    H = X @ XtXi @ X.T
    print(f"(b) diag(H) 와 일치: {np.allclose(h_einsum, np.diag(H))},  합 {h_einsum.sum():.6f}  (= p)")
    ```

    출력:

    ```
    (a) 반복문과 일치: True,  평균 3.9600  (≈ p = 4)
    (b) diag(H) 와 일치: True,  합 4.000000  (= p)
    ```

    **첨자 표기를 읽는 법.** `'ij,jk,ik->i'`는 $\sum_j \sum_k A_{ij} B_{jk} A_{ik}$를 뜻한다. 출력에 $i$만 남았으므로 관측마다 스칼라 하나가 나온다. $i$가 세 인자에 모두 나타나면서 출력에도 있다는 것이 "각 행을 자기 자신과만 짝지어라"라는 지시이며, 이것이 비대각 성분을 아예 계산하지 않게 해 준다.

    마할라노비스 거리의 평균이 $p$에 가까운 것은 우연이 아니다. 표본공분산으로 표준화했으므로 $\mathbb{E}[(\mathbf{x}-\boldsymbol{\mu})^T\mathbf{S}^{-1}(\mathbf{x}-\boldsymbol{\mu})] \approx p$이다. 지렛값의 합이 정확히 $p$인 것과 같은 종류의 항등식이다.

    **왜 중요한가: 메모리다.** $n = 5000$이면 모자 행렬은

    $$
    5000^2 \times 8\ \text{바이트} = 200\ \text{MB}
    $$

    인데 정작 필요한 대각 성분은 $40$ KB다. $n = 10^5$이면 $80$ GB가 되어 아예 불가능하다. `einsum`은 대각 성분만 직접 계산하므로 메모리가 $O(n)$이다.

    회귀 진단에서 $h_{ii}$는 늘 필요한 값이므로($\mathrm{Var}(e_i) = \sigma^2(1-h_{ii})$, 쿡의 거리, 하나 빼기 잔차) 큰 자료에서 이 계산법을 아는 것이 실제로 도움이 된다. $\square$

---

## 정리하며

| 개념 | 핵심 |
|---|---|
| `ndarray` | 동질적이고 크기가 고정된 $N$차원 연속 메모리 블록 |
| 벡터화 | 명시적 반복문을 배열 수준의 표현식으로 대체 |
| 브로드캐스팅 | 두 개의 단순한 규칙에 따른 자동 모양 확장 |
| 집계 | `axis` 매개변수를 갖는 `sum`, `mean`, `std`, `var`, `ddof`에 유의 |
| 선형대수 | 행렬곱은 `@`, `inv`보다 `solve`, 대칭행렬에는 `eigh` |
| 난수 생성 | `default_rng(seed)`가 재현 가능한 현대적 인터페이스 |
