# 파이썬과 주피터 기초

파이썬은 이 책 전체에서 사용하는 주된 계산 언어다. 세 가지 이유로 파이썬을 골랐다. 통계적 아이디어를 전면에 남겨 두는 마찰 없는 문법, 수치 라이브러리의 성숙한 생태계(NumPy, SciPy, pandas, statsmodels, scikit-learn, PyMC), 그리고 코드·출력·수식·서술이 한 문서에 함께 사는 주피터 노트북의 문예적 프로그래밍 작업 흐름이다. 이 절에서는 환경 설정, 패키지 관리, 통계 스크립트에서 가장 자주 쓰는 언어 기능, 그리고 이 책의 모든 코드 예제가 따르는 관례를 다룬다.

## 파이썬 환경

### 이 책에서 "파이썬"이 뜻하는 것

**파이썬**은 동적 타입이고 가비지 컬렉션을 하며 인터프리터로 실행되는 언어로, 일급 함수와 "건전지 포함" 표준 라이브러리를 갖추고 있다. 통계 작업 흐름은 다음에 의존한다.

- CPython 인터프리터(3.11 이상 권장).
- 과학 스택: NumPy(배열), SciPy(알고리즘, 분포, 최적화), pandas(데이터프레임), Matplotlib(그림), statsmodels(회귀, 시계열), scikit-learn(기계학습).
- 패키지/환경 관리자: `conda`(Anaconda/Miniconda) 또는 `pip` + `venv` / `uv`.

**Anaconda** 배포판은 파이썬과 1,500개 이상의 과학 패키지, 그리고 `conda` 패키지/환경 관리자를 함께 묶어 제공하므로 처음 시작하는 사용자에게 저항이 가장 적은 길이다. **주피터 노트북**(과 그 후속인 JupyterLab)은 마크다운 서술, 파이썬 코드, 그림, LaTeX 수식이 공존하는 셀 기반 대화형 인터페이스를 제공한다.

### 재현 가능한 환경

프로젝트마다 특정 파이썬 버전과 패키지 버전을 고정해야 한다. conda를 쓴다면 다음과 같다.

```bash
conda create --name stats_env python=3.11
conda activate stats_env
conda install numpy scipy pandas matplotlib statsmodels
conda env export > environment.yml
```

`environment.yml` 파일을 형상관리에 커밋해 두면 다른 컴퓨터에서도 분석을 재현할 수 있다.

## 설명

### 내장 컨테이너와 각각의 쓰임새

| 컨테이너 | 변경 가능 | 순서 있음 | 쓰임새 |
|---|---|---|---|
| `list`  | ✓ | ✓ | 범용 열(예: 벡터화하기 전의 값들의 열) |
| `tuple` | ✗ | ✓ | 고정 크기 레코드(예: 함수에서 여러 값 반환) |
| `dict`  | ✓ | ✓ (삽입 순서) | 키로 조회 — 매개변수 묶음, JSON 형태의 자료 |
| `set`   | ✓ | ✗ | 포함 여부 검사, 중복 제거 |

기본은 리스트다. 키가 정수 인덱스가 아니라 의미 있는 이름일 때는 사전을 쓴다. 튜플은 가벼운 반환값으로 빛을 발한다. `mean, std = summarize(data)`처럼 결과를 풀어낼 수 있다.

### 컴프리헨션

리스트 컴프리헨션은 "어떤 모임의 각 $x$에 대해 $f(x)$를 계산하되 필요하면 조건으로 걸러낸다"는 패턴을 한 줄로 표현한다.

```python
data = [-2, 3, 0, 5, -1, 4]
squares = [x**2 for x in data if x > 0]
print(squares)
```

출력:

```
[9, 25, 16]
```

사전(`{k: f(k) for k in keys}`), 집합(`{f(x) for x in xs}`), 제너레이터(`(f(x) for x in xs)`)에도 같은 형태가 있다. 제너레이터는 게으르게 값을 내놓으므로 열이 크거나 무한할 때 중요하다.

### 함수와 독스트링

```python
def sample_mean(data):
    """Return the arithmetic mean of an iterable of numbers."""
    return sum(data) / len(data)
```

독스트링은 함수의 계약이다. 무엇을 계산하고, 무엇을 기대하며, 무엇을 반환하는지 밝힌다. `def` 줄 바로 뒤의 삼중 따옴표 문자열은 `help(fn)`으로 볼 수 있고 자동 문서화의 근거가 된다.

람다(`lambda x: x**2`)는 이름 없는 단일 표현식 함수로, `map`, `filter`, `sorted(..., key=...)`의 인수로 쓰기 편하다. 표현식 하나보다 길어지는 것은 이름 있는 `def`로 쓰는 편이 낫다.

### 모듈 구조와 `__main__` 가드

모든 파이썬 파일은 모듈이다. 어떤 파일을 안전하게 **임포트 가능**(다른 곳에서 함수를 재사용)하면서 동시에 **실행 가능**(직접 호출하면 시연을 수행)하게 만들려면 다음 형태를 쓴다.

```python
"""Module docstring describing what this script does."""

# === Imports ===
import numpy as np

# === Function definitions ===
def my_function(x):
    return x + 1

# === Demonstration / entry point ===
if __name__ == "__main__":
    print(my_function(3))
```

출력:

```
4
```

`if __name__ == "__main__":` 가드는 시연 블록이 직접 호출(`python my_script.py`)할 때만 실행되고, 다른 모듈이 `import my_script`할 때는 실행되지 않도록 보장한다. 이 책의 모든 `.py` 파일이 따르는 교육용 방식이다.

### 표준 임포트 별칭

거의 모든 노트북의 첫 셀은 다음과 같다.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
```

이 별칭들(`np`, `pd`, `plt`, `stats`)은 사실상의 표준이며 가독성을 높인다.

### 주피터 필수 사항

`jupyter notebook`이나 `jupyter lab`으로 실행한다. 명령 모드(`Esc`를 눌러 진입)의 주요 단축키는 다음과 같다.

- `Shift+Enter` — 현재 셀 실행.
- `A` / `B` — 위 / 아래에 셀 삽입.
- `M` / `Y` — 셀을 마크다운 / 코드로 변환.
- `D D` — 현재 셀 삭제.

마크다운 셀은 `$...$`(인라인)과 `$$...$$`(디스플레이) 사이의 LaTeX를 지원하므로, 노트북은 통계적 논증을 전개하기에 자연스러운 자리가 된다.

## 예제

```python
"""Demonstrate core Python idioms used in statistics."""

# === Compute a summary without NumPy ===
def summarize(data):
    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / (n - 1)
    return {"mean": mean, "variance": variance, "n": n}

# === Comprehensions and unpacking ===
values = [4, 7, 13, 2, 9]
squares = [v ** 2 for v in values]
above_average = [v for v in values if v > sum(values) / len(values)]

result = summarize(values)
mean, variance, n = result["mean"], result["variance"], result["n"]

print(f"Values:           {values}")
print(f"Squares:          {squares}")
print(f"Above average:    {above_average}")
print(f"Mean = {mean:.3f}, Var = {variance:.3f}, n = {n}")

# === Hand-off to NumPy ===
import numpy as np
data = np.array(values)
print(f"NumPy mean:       {data.mean():.3f}")
print(f"NumPy var (n-1):  {data.var(ddof=1):.3f}")
```

출력:

```
Values:           [4, 7, 13, 2, 9]
Squares:          [16, 49, 169, 4, 81]
Above average:    [13, 9]
Mean = 7.000, Var = 18.500, n = 5
NumPy mean:       7.000
NumPy var (n-1):  18.500
```

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
수의 리스트를 받아 `"mean"`, `"variance"`, `"n"`, `"se_mean"`(평균의 표준오차 $s / \sqrt{n}$)을 키로 갖는 사전을 반환하는 함수 `summary_stats(data)`를 NumPy를 쓰지 않고 작성하라.

</div>

??? success "풀이"
    ```python
    from math import sqrt

    def summary_stats(data):
        n = len(data)
        mean = sum(data) / n
        variance = sum((x - mean) ** 2 for x in data) / (n - 1)
        se_mean = sqrt(variance / n)
        return {"mean": mean, "variance": variance, "n": n, "se_mean": se_mean}

    print(summary_stats([2, 4, 4, 4, 5, 5, 7, 9]))
    # {'mean': 5.0, 'variance': 4.0, 'n': 8, 'se_mean': 0.7071...}
    ```

    출력:

    ```
    {'mean': 5.0, 'variance': 4.571428571428571, 'n': 8, 'se_mean': 0.7559289460184544}
    ```

    분산은 불편추정을 위해 $n - 1$을 쓴다(베셀 보정). 평균의 표준오차는 $s = \sqrt{s^2}$일 때 $s/\sqrt{n}$이다.

<div class="drillbox" markdown>

**연습문제 2.**
원소별 산술 연산에서 파이썬 `list`와 NumPy `ndarray`의 차이를 설명하라. `[1, 2, 3] * 2`와 `np.array([1, 2, 3]) * 2`가 각각 무엇을 만들어내는지, 그리고 그 이유를 보여라.

</div>

??? success "풀이"
    ```python
    import numpy as np

    print([1, 2, 3] * 2)            # [1, 2, 3, 1, 2, 3]
    print(np.array([1, 2, 3]) * 2)   # [2 4 6]
    ```

    출력:

    ```
    [1, 2, 3, 1, 2, 3]
    [2 4 6]
    ```

    파이썬 `list`는 `*`를 원소별 산술이 아니라 반복 이어붙이기로 정의한다. NumPy `ndarray`는 `*`를 재정의해 스칼라를 모든 원소에 브로드캐스트한다. 벡터화 연산(`arr + 1`, `arr ** 2`, `np.sqrt(arr)`)은 더 빠르고(컴파일된 C/BLAS 루틴에 위임된다) 더 짧다(명시적 반복문이 없다). 수치 작업에서 파이썬 `for` 반복문에 손을 뻗는다면 거의 언제나 그 일이 NumPy에 속한다는 신호다.

<div class="drillbox" markdown>

**연습문제 3.**
$1 \le i < j \le 5$인 모든 쌍 $(i, j)$를 하나의 표현식으로 생성하는 리스트 컴프리헨션을 작성하라. 그 개수가 $\binom{5}{2}$와 같음을 확인하라.

</div>

??? success "풀이"
    ```python
    pairs = [(i, j) for i in range(1, 6) for j in range(i + 1, 6)]
    print(pairs)
    print(f"Count: {len(pairs)}, C(5,2) = {5 * 4 // 2}")
    # [(1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)]
    # Count: 10, C(5,2) = 10
    ```

    출력:

    ```
    [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
    Count: 10, C(5,2) = 10
    ```

    이중 `for` 컴프리헨션은 바깥 반복에서 `i`를, 안쪽 반복에서 `j`를 훑으며, 중첩된 `for` 문과 같은 일을 표현식 형태로 한다.

<div class="drillbox" markdown>

**연습문제 4.**
`if __name__ == "__main__":`이 무엇을 하는지 설명하라. 이를 빠뜨렸을 때 임포트 시 원치 않는 동작이 생기는 예를 하나 들어라.

</div>

??? success "풀이"
    `__name__`은 모듈의 특수 속성이다. 파일이 스크립트로 실행될 때(`python myscript.py`)는 `"__main__"`과 같고, 그 밖의 경우에는 모듈의 임포트 이름(`mypackage.myscript`)과 같다.

    `if __name__ == "__main__":` 안의 코드는 직접 호출할 때만 실행되고 임포트 시에는 건너뛴다. 이것이 중요한 이유는 세 가지다.

    1. **임포트 시 부작용 없음**: 최상위의 `print` 문이나 비용이 큰 계산이 모듈을 불러올 때마다 실행되어 버린다.
    2. **재사용성**: 함수와 클래스는 임포트할 수 있게 두고 시연 코드는 감춘다.
    3. **관례**: 이 책의 모든 스크립트가 이 패턴을 따른다.

    **가드가 없을 때의 실패 예:**
    ```python
    # bad_module.py
    def util(x): return x + 1
    print("Loading...")            # runs every time someone imports bad_module
    print(util(10))
    ```

    출력:

    ```
    Loading...
    11
    ```
    `import bad_module`을 하는 다른 모듈은 모두 이 출력을 유발한다. 잘해야 지저분함이고, 나쁘면 부작용에서 비롯된 오류다.

<div class="drillbox" markdown>

**연습문제 5.**
큰 부동소수를 많이 더하면 산술평균이 넘칠 수 있다. 다음 갱신 규칙을 이용해 한 번의 순회로 계산하는 온라인 평균을 작성하라.

$$
\bar{x}_n = \bar{x}_{n-1} + \frac{x_n - \bar{x}_{n-1}}{n}
$$

가우스분포에서 뽑은 $10^6$개 값의 리스트에서 `sum(data)/len(data)`와 수치적으로 비교하고, 온라인 형태가 언제 더 나은지 설명하라.

</div>

??? success "풀이"
    ```python
    import numpy as np

    def online_mean(data):
        m = 0.0
        for n, x in enumerate(data, start=1):
            m += (x - m) / n
        return m

    rng = np.random.default_rng(0)
    data = rng.normal(1e6, 1, size=10**6).tolist()

    naive = sum(data) / len(data)
    online = online_mean(data)
    print(f"naive  = {naive:.6f}")
    print(f"online = {online:.6f}")
    ```

    출력:

    ```
    naive  = 1000000.000999
    online = 1000000.000999
    ```

    여기서는 둘이 여러 자리까지 일치하지만, 온라인 형태에는 실질적인 장점이 두 가지 있다. (1) 스트림을 처리하므로 자료가 메모리에 다 들어가지 않을 때 유용하고, (2) `n = 10^6, mean = 10^6`인 경우 $10^{12}$에 가까운 합을 누적하지 않아도 되는데, 그런 합은 64비트 부동소수에서 정밀도를 잃을 수 있다. 이 스트리밍 알고리즘은 분산(웰퍼드), 회귀, 분위수 추정으로 확장된다.

<div class="drillbox" markdown>

**연습문제 6.**
이름으로 접근하는 분포 매개변수를 저장할 때 (3.7 이후의) 파이썬 `dict`가 `(키, 값)` 튜플의 `list`보다 나은 이유는 무엇인가? 점근적 복잡도와 코드의 명료성 측면에서 논하라.

</div>

??? success "풀이"
    **복잡도**: 사전은 해시 테이블이다. 키에 의한 조회, 삽입, 삭제가 기대 시간 $O(1)$이다. $(k, v)$ 튜플의 리스트는 선형 탐색이 필요해 조회마다 $O(n)$이다. 항목이 몇 개뿐인 매개변수 묶음에서는 점근 복잡도보다 상수가 더 중요하지만, 자료가 커지면 원리는 그대로 적용된다.

    **명료성**: `params["mu"]`는 무엇을 읽는지 정확히 말해준다. `next(v for k, v in params if k == "mu")`도 같은 일을 하지만 의도를 흐리고 더 깨지기 쉽다. 이름 기반 접근은 묶음을 함수에 넘길 때 `**kwargs` 언패킹과도 깔끔하게 어울린다.

    ```python
    params = {"loc": 0.0, "scale": 1.0}
    samples = rng.normal(size=100, **params)  # equivalent to loc=0.0, scale=1.0
    ```

    튜플 리스트 형태는 키가 중복될 수 있거나 삽입 순서만이 의미를 갖는 경우에만 적절하다.

<div class="drillbox" markdown>

**연습문제 7.**
다음 함수는 호출할 때마다 새 리스트를 돌려줄 것처럼 보이지만 그렇지 않다. 무슨 일이 일어나는지 설명하고 고쳐라.

```python
def collect(x, acc=[]):
    acc.append(x)
    return acc
```

</div>

??? success "풀이"
    ```python
    def collect(x, acc=[]):
        acc.append(x)
        return acc

    print(collect(1), collect(2), collect(3))
    print("함수 객체가 들고 있는 기본값:", collect.__defaults__)
    ```

    출력:

    ```
    [1, 2, 3] [1, 2, 3] [1, 2, 3]
    함수 객체가 들고 있는 기본값: ([1, 2, 3],)
    ```

    세 번의 호출이 모두 같은 리스트를 돌려준다.

    **원인.** 기본값은 함수가 **정의될 때 한 번만** 평가되어 함수 객체에 붙어 있다(`__defaults__`로 직접 볼 수 있다). 호출할 때마다 새로 만들어지지 않는다. 리스트는 가변이므로 `append`가 그 하나뿐인 객체를 계속 키운다.

    **고치는 법.** 감시값으로 `None`을 쓴다.

    ```python
    def collect(x, acc=None):
        if acc is None:
            acc = []
        acc.append(x)
        return acc

    print(collect(1), collect(2), collect(3))
    ```

    출력:

    ```
    [1] [2] [3]
    ```

    이제 매 호출마다 새 리스트가 만들어진다.

    **왜 통계 코드에서 특히 위험한가.** 모의실험 결과를 모으는 함수에 이 버그가 있으면 오류 없이 조용히 **이전 실험 결과가 섞여 든다.** 반복 횟수가 맞지 않거나 분산이 이상하게 작아지는 식으로 나타나는데, 원인을 찾기가 매우 어렵다.

    같은 함정이 `dict`, `set`, NumPy 배열 등 모든 가변 객체에 적용된다. 반대로 `int`, `str`, `tuple`, `None` 같은 불변 객체는 기본값으로 안전하다. **기본값으로 쓸 것은 불변 객체뿐이라고 외워 두면 된다.** $\square$

<div class="drillbox" markdown>

**연습문제 8.**
`0.1 + 0.2 == 0.3`은 거짓이다. 이유를 설명하고, 부동소수를 비교하는 올바른 방법과 반올림 오차가 누적되는 예를 보여라.

</div>

??? success "풀이"
    ```python
    import math

    print(f"0.1 + 0.2 == 0.3  →  {0.1 + 0.2 == 0.3}")
    print(f"실제 값: {0.1 + 0.2!r}")
    print(f"차이:     {0.1 + 0.2 - 0.3:.3e}")
    print(f"math.isclose: {math.isclose(0.1 + 0.2, 0.3)}")

    s = 0.0
    for _ in range(10):
        s += 0.1
    print(f"\n0.1 을 열 번 더하면: {s!r}   (== 1.0 인가: {s == 1.0})")
    print(f"math.fsum 으로:      {math.fsum([0.1] * 10)!r}")
    ```

    출력:

    ```
    0.1 + 0.2 == 0.3  →  False
    실제 값: 0.30000000000000004
    차이:     5.551e-17
    math.isclose: True

    0.1 을 열 번 더하면: 0.9999999999999999   (== 1.0 인가: False)
    math.fsum 으로:      1.0
    ```

    **원인.** 배정밀도는 수를 이진 분수로 저장한다. $0.1$은 $2$의 거듭제곱의 유한 합으로 나타낼 수 없어($10$진법에서 $1/3$을 유한소수로 못 쓰는 것과 같다) 가장 가까운 표현 가능한 값으로 반올림된다. $0.1$, $0.2$, $0.3$ 각각의 반올림 오차가 서로 다르게 쌓이면서 앞의 둘의 합이 셋째와 정확히 같아지지 않는다.

    **올바른 비교.** `==` 대신 허용오차를 쓴다.

    - 스칼라: `math.isclose(a, b)` — 상대오차와 절대오차를 함께 본다
    - 배열: `np.isclose(a, b)`, `np.allclose(a, b)`
    - 검정 코드: `np.testing.assert_allclose(actual, desired, rtol=1e-7)`

    절대 허용오차만 쓰면 안 된다. $10^{-9}$짜리 두 수를 비교할 때와 $10^{9}$짜리 두 수를 비교할 때 의미 있는 허용오차가 전혀 다르기 때문이다. `isclose` 계열이 상대오차를 기본으로 쓰는 이유다.

    **누적.** $0.1$을 열 번 더하면 $1.0$이 아니라 $0.9999999999999999$가 된다. 오차가 매 덧셈마다 조금씩 쌓인 것이다. `math.fsum`은 중간 오차를 추적해 정확한 $1.0$을 준다.

    **통계 코드에서 부딪히는 곳.**

    - **누적확률**: 확률질량함수를 더해 나갈 때 마지막 합이 정확히 $1$이 아닐 수 있다. 반복문 조건을 `while total < 1.0`으로 두면 무한루프가 될 수 있다.
    - **격자 생성**: `np.arange(0, 1, 0.1)`은 부동소수 누적 때문에 원소 개수가 예상과 다를 수 있다. 개수를 지정하는 `np.linspace(0, 1, 11)`을 쓰라.
    - **분산이 음수로 나오는 경우**: NumPy 연습문제 9에서 본 상쇄 오차가 같은 뿌리에서 나온다. $\square$

<div class="drillbox" markdown>

**연습문제 9.**
리스트 컴프리헨션 `[f(x) for x in ...]`과 생성자 표현식 `(f(x) for x in ...)`의 차이를 메모리 관점에서 설명하라. 모의실험에서 어느 쪽을 언제 써야 하는가?

</div>

??? success "풀이"
    ```python
    import sys

    lst = [x * x for x in range(1_000_000)]      # 전부 만들어 메모리에 올린다
    gen = (x * x for x in range(1_000_000))      # 만드는 방법만 들고 있는다

    print(f"리스트: {sys.getsizeof(lst) / 1e6:.3f} MB")
    print(f"생성자: {sys.getsizeof(gen)} 바이트")
    print(f"합은 같은가: {sum(lst) == sum(x * x for x in range(1_000_000))}")
    ```

    출력:

    ```
    리스트: 8.449 MB
    생성자: 200 바이트
    합은 같은가: True
    ```

    리스트는 $8$ MB를 쓰는데 생성자는 $200$ 바이트다. 생성자는 값을 저장하지 않고 **다음 값을 만드는 방법**만 들고 있다가 요청받을 때마다 하나씩 내놓는다.

    **어느 쪽을 쓸 것인가.**

    | 생성자가 나은 경우 | 리스트가 나은 경우 |
    |---|---|
    | 한 번만 훑고 버린다(`sum`, `max`, `any`) | 여러 번 훑어야 한다 |
    | 자료가 메모리보다 크다 | 인덱싱·슬라이싱이 필요하다 |
    | 조기 종료할 수 있다(`next`, `any`) | `len()`이 필요하다 |
    | 무한 스트림 | 정렬해야 한다 |

    **결정적인 함정.** 생성자는 **한 번만 소비할 수 있다.**

    ```python
    gen = (x * x for x in range(5))
    print(f"첫 번째 sum: {sum(gen)}")
    print(f"두 번째 sum: {sum(gen)}   ← 이미 소진되어 0")
    ```

    출력:

    ```
    첫 번째 sum: 30
    두 번째 sum: 0   ← 이미 소진되어 0
    ```

    두 번째 호출이 오류 없이 $0$을 준다. 부트스트랩 결과를 생성자에 담고 평균과 표준편차를 잇따라 계산하면, 표준편차가 조용히 빈 자료에서 계산된다. **통계량을 두 개 이상 계산할 것이라면 리스트로 구체화하라.**

    실무에서는 모의실험 결과를 대개 NumPy 배열로 모은다. 메모리도 리스트보다 효율적이고(파이썬 객체가 아니라 원시 수치를 저장한다) 벡터화 연산도 쓸 수 있기 때문이다. 생성자는 자료가 정말 클 때, 또는 파일을 한 줄씩 읽어 들일 때 쓴다. $\square$

<div class="drillbox" markdown>

**연습문제 10.**
재현 가능한 난수를 만드는 올바른 방법을 보여라. `np.random.seed`의 전역 상태가 왜 문제인지 설명하고, 병렬 모의실험을 위한 **독립적인** 난수 스트림을 만들어라.

</div>

??? success "풀이"
    ```python
    import numpy as np

    # 같은 씨앗 → 같은 결과. 재현성의 기본
    a = np.random.default_rng(42).normal(size=3)
    b = np.random.default_rng(42).normal(size=3)
    print(f"같은 씨앗이 같은 값을 주는가: {np.allclose(a, b)}")

    # 흔한 실수: 병렬 작업마다 같은 씨앗을 준다
    naive = [np.random.default_rng(42).normal(size=2).round(4).tolist()
             for _ in range(3)]
    print(f"\n작업마다 같은 씨앗 → {naive}")
    print("  세 스트림이 완전히 동일하다. 독립 반복이 아니다.")

    # 올바른 방법: SeedSequence 로 갈래를 친다
    children = np.random.SeedSequence(42).spawn(3)
    print("\nspawn 으로 만든 독립 스트림:")
    for i, child in enumerate(children):
        print(f"  작업 {i}: {np.random.default_rng(child).normal(size=3).round(4)}")
    ```

    출력:

    ```
    같은 씨앗이 같은 값을 주는가: True

    작업마다 같은 씨앗 → [[0.3047, -1.04], [0.3047, -1.04], [0.3047, -1.04]]
      세 스트림이 완전히 동일하다. 독립 반복이 아니다.

    spawn 으로 만든 독립 스트림:
      작업 0: [0.4183 0.6056 0.0288]
      작업 1: [ 1.2545  0.6063 -1.3402]
      작업 2: [-1.6494  1.406   0.7235]
    ```

    **`np.random.seed`의 문제.** 이것은 인터프리터 전체가 공유하는 **전역** 상태를 건드린다.

    - 내가 부른 라이브러리 함수가 전역 난수를 쓰면 내 결과가 조용히 달라진다. 반대로 내 코드가 씨앗을 다시 심으면 남의 결과를 망가뜨린다.
    - 어느 코드가 몇 번 난수를 뽑았는지에 따라 결과가 달라지므로, 상관없어 보이는 곳을 고쳐도 재현이 깨진다.
    - 스레드 안전하지 않다.

    `default_rng`가 돌려주는 `Generator` 객체는 자기만의 상태를 들고 다닌다. 함수에 인자로 넘기면 그 함수가 쓰는 난수가 명시적으로 드러난다. **`rng`를 인자로 받는 함수로 쓰라.**

    **씨앗 하나를 여러 작업에 재사용하지 마라.** 위 출력에서 보듯 세 "반복"이 글자 그대로 같은 수를 낸다. 이렇게 모의실험을 돌리면 반복 간 변동이 $0$이 되어 표준오차가 터무니없이 작게 나온다. 작업 번호를 씨앗에 더하는 방식(`default_rng(42 + i)`)도 널리 쓰이지만 스트림이 겹치지 않는다는 보장이 없다. `SeedSequence.spawn`은 통계적으로 독립인 스트림을 보장하는 방법이며, 이것이 권장되는 방식이다.

    **재현 가능한 논문을 위한 점검표.**

    1. 씨앗을 코드에 명시하고 논문에 적는다.
    2. 전역 상태 대신 `Generator` 객체를 넘긴다.
    3. 라이브러리 버전을 기록한다(난수 알고리즘이 판올림에서 바뀔 수 있다).
    4. 병렬 작업에는 `spawn`을 쓴다. $\square$

---

## 정리하며

이 절은 뒤에 나오는 모든 코드가 기댈 **작업 환경과 관용구**를 정했다.

- **환경.** Anaconda 로 파이썬과 과학 패키지를 한꺼번에 들여오고, 주피터에서 서술·코드·그림·수식을 한 문서에 담는다. 환경을 파일로 고정해 두는 것이 재현성의 첫걸음이다.
- **컨테이너 고르기.** 순서가 필요하면 `list`, 바뀌지 않는 레코드면 `tuple`, 키로 찾으면 `dict`, 포함 여부만 보면 `set` 이다. 중복 제거와 포함 검사에서 `set` 이 압도적으로 빠르다.
- **컴프리헨션**은 짧고 빠르지만, 수치 계산에서는 다음 절의 **벡터화**가 다시 한 단계 빠르다.
- **`if __name__ == "__main__":` 가드**와 독스트링. 이 책의 모든 `.py` 예제가 이 꼴을 지키므로, 모듈로 불러다 써도 실행 코드가 튀어나오지 않는다.
- **표준 임포트 별칭.** `np`, `pd`, `plt`, `stats` 는 관례이며 이 책 전체에서 예외 없이 같은 뜻이다.

**언어는 도구일 뿐이고, 통계의 내용은 다음 세 절에 있다.** NumPy 가 수치 계산을, pandas 가 이름표 붙은 자료를, Matplotlib 이 그림을 맡는다.

다음 절 **NumPy 배열**부터 시작한다. 반복문을 배열 연산으로 바꾸는 사고방식이 요점이며, 앞 절에서 적은 벡터·행렬 연산이 그대로 한 줄의 코드가 된다.
