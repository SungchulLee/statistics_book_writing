# pandas로 자료 다루기

pandas는 구조화된 자료를 불러오고, 정제하고, 변환하고, 요약하기 위한 이름표가 붙은 자료구조 — `Series`(1차원)와 `DataFrame`(2차원) — 를 제공한다. 원본 파일(CSV, parquet, SQL)과 NumPy의 배열 수준 계산이나 statsmodels·scikit-learn의 통계 모형 사이를 잇는 다리다. 이 책의 거의 모든 실증적 작업 흐름은 pandas를 임포트하는 것으로 시작해, 마지막 요약표를 그려낸 뒤에야 끝난다.

## 정의

**`Series`** 는 1차원의 이름표 붙은 배열, 즉 값과 인덱스의 조합이다. **`DataFrame`** 은 2차원 표로, 각 열이 하나의 `Series`이며 서로 자료형이 다를 수 있지만 모두 같은 행 인덱스를 공유한다. 개념적으로 DataFrame은 열들의 사전이고, 기계적으로는 각 열이 NumPy 배열로 뒷받침된다.

떠올릴 만한 심상은 NumPy 위에 얹은 SQL이다. 배열 층이 속도를 주고, 이름표 층이 통계 분석과 잘 맞는 선택·정렬·그룹화·조인·피벗 의미론을 준다.

## 설명

### 불러오기와 살펴보기

```python
import pandas as pd

df = pd.read_csv("data.csv")
df.head()        # first 5 rows
df.tail()        # last 5 rows
df.shape         # (n_rows, n_cols)
df.dtypes        # type of each column
df.info()        # dtypes, non-null counts, memory usage
df.describe()    # numeric summary: count, mean, std, min, quartiles, max
```

`read_csv`는 `parse_dates`, `dtype`, `na_values`, `usecols`, `chunksize`를 받는다. 자료 품질 문제는 대부분 나중이 아니라 불러오는 시점에 처리하는 것이 가장 좋다.

### 행과 열 선택하기

반드시 구별해야 할 서로 독립적인 연산이 셋 있다.

| 형태 | 의미 |
|---|---|
| `df["col"]`, `df[["col1", "col2"]]` | 이름표로 열 선택 |
| `df.loc[row_label, col_label]` | 두 축 모두 이름표 기반 |
| `df.iloc[row_pos, col_pos]` | 두 축 모두 정수 위치 기반 |
| `df[df["x"] > 5]` | 어떤 열에 대한 불리언 행 필터 |

`loc`과 `iloc`은 행 인덱스가 기본값 `0, 1, 2, ...`가 아닐 때 정확히 갈린다. 인덱스를 정렬했거나 설정했거나 걸러낸 뒤에는 언제나 명시적인 형태를 쓰라.

### 결측값 정제

```python
df.isna().sum()             # count of missing per column
df.dropna()                 # drop rows with any NaN
df.dropna(subset=["x"])     # drop rows with NaN in 'x' only
df.fillna(df.median(numeric_only=True))   # impute with column median
```

"옳은" 대체 전략이란 없다. 무엇을 고를지(삭제, 평균, 중앙값, 모형 기반, 다중대체)는 결측 기제에 달려 있다. pandas는 도구를 줄 뿐 결정은 사용자에게 맡긴다.

### 그룹화: 분할–적용–결합

pandas에서 가장 강력한 하나의 패턴이다.

```python
df.groupby("treatment")["outcome"].agg(["count", "mean", "std"])
```

`groupby`는 `"treatment"`의 서로 다른 값에 따라 자료를 분할하고, 각 그룹 안에서 `"outcome"`에 지정된 집계를 적용한 뒤, 결과를 깔끔한 DataFrame으로 결합한다. 탐색적 분석과 확증적 분석의 일꾼이다. 여러 키(`groupby(["a", "b"])`)와 사용자 정의 집계(`agg(my_func)`)로 이 패턴을 일반화할 수 있다.

### 기술통계와 공식의 대응

| pandas 호출 | 계산하는 것 | ddof 기본값 |
|---|---|---|
| `df["x"].mean()` | $\bar{x}$ | 해당 없음 |
| `df["x"].var()` | $s^2$ | **`ddof=1`** |
| `df["x"].std()` | $s$ | **`ddof=1`** |
| `df["x"].quantile(0.5)` | 표본 중앙값 | 해당 없음 |
| `df.corr()` | 피어슨 상관행렬 | 해당 없음 |
| `df.cov()` | 표본 공분산행렬 | `ddof=1` |

기본값이 `ddof=0`인 NumPy와 다르다는 점에 유의하라. pandas와 NumPy의 결과가 $(n-1)/n$배만큼 어긋난다면 이유는 바로 이것이다.

## 예제

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# Build a DataFrame mixing categorical and numeric data
df = pd.DataFrame({
    "group": rng.choice(["A", "B", "C"], size=200),
    "value": rng.normal(50, 10, size=200),
    "score": rng.integers(0, 100, size=200),
})

# Per-group summary
summary = df.groupby("group")["value"].agg(["count", "mean", "std"]).round(2)
print(summary)

# Boolean filtering
high = df[df["value"] > 60]
print(f"\nValues > 60: {len(high)} / {len(df)}")

# Correlation between numeric columns
print("\nCorrelation:\n", df.corr(numeric_only=True).round(3))

# Pivot: mean value by group × score quartile
df["score_q"] = pd.qcut(df["score"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
print("\nPivot:\n", df.pivot_table(values="value", index="group",
                                   columns="score_q", aggfunc="mean").round(1))
```

## 연습문제

**연습문제 1.**
학생 다섯 명에 대해 `name`(문자열), `score`(정수), `passed`(불리언) 열을 갖는 `DataFrame`을 만들어라. `passed`가 `True`이고 **또한** `score`가 80보다 큰 행만 보이도록 걸러라.

??? success "풀이"
    ```python
    import pandas as pd
    df = pd.DataFrame({
        "name":   ["Alice", "Bob", "Carol", "Dave", "Eve"],
        "score":  [92, 65, 88, 73, 95],
        "passed": [True, False, True, False, True],
    })
    print(df[df["passed"] & (df["score"] > 80)])
    ```
    피연산자가 파이썬 스칼라가 아니라 pandas 불리언 Series이므로 `and`가 아니라 비트 연산자 `&`를 써야 한다. 각 비교식은 괄호로 묶어라. 연산자 우선순위상 `&`가 `>`보다 높기 때문이다.

---

**연습문제 2.**
다음 DataFrame이 주어졌을 때
```python
df = pd.DataFrame({"group": ["A","A","B","B","B"], "x": [1, 3, 2, 8, 5]})
```
각 그룹에 대해 개수, 표본평균, 표본분산, 그리고 최댓값에서 최솟값을 뺀 값을 계산하라. 하나의 DataFrame으로 반환하라.

??? success "풀이"
    ```python
    summary = df.groupby("group")["x"].agg(
        n="count",
        mean="mean",
        var="var",
        range=lambda s: s.max() - s.min(),
    )
    print(summary)
    ```
    `agg`는 출력 열 이름을 문자열 집계 이름이나 호출 가능 객체에 대응시키는 키워드 인수를 받는다. 람다를 쓰면 이름 있는 함수를 따로 정의하지 않고 "범위"를 하나의 표현식으로 나타낼 수 있다.

---

**연습문제 3.**
`df.loc[]`과 `df.iloc[]`의 차이를 설명하라. 같은 DataFrame에 대해 둘이 서로 다른 행을 반환하는 예를 제시하라.

??? success "풀이"
    `loc`은 **이름표 기반**이다. `df.loc[0]`은 인덱스 이름표가 `0`인 행을 반환한다. `iloc`은 **정수 위치 기반**이다. `df.iloc[0]`은 인덱스 이름표와 무관하게 첫 번째 행을 반환한다.

    ```python
    df = pd.DataFrame({"A": [10, 20, 30]}, index=[2, 0, 1])
    print(df.loc[0])    # row labeled 0   → A = 20
    print(df.iloc[0])   # row at position 0 → A = 10
    ```

    인덱스가 (기본값인) `RangeIndex(0, n)`이고 행 순서가 바뀌지 않았다면 둘은 일치한다. `df.sort_values()`, `df.set_index()`, 불리언 필터링을 거치고 나면 둘이 갈라지며, 이 둘을 조용히 혼동하는 것이 흔한 버그의 원천이다.

---

**연습문제 4.**
다음 CSV 형식 문자열을 불러와 결측값을 열 중앙값으로 채우고 상관행렬을 계산하라.

```text
x,y,z
1.0,2.0,3.0
2.0,,4.0
3.0,6.0,
4.0,8.0,6.0
5.0,10.0,7.0
```

??? success "풀이"
    ```python
    from io import StringIO
    csv = """x,y,z
    1.0,2.0,3.0
    2.0,,4.0
    3.0,6.0,
    4.0,8.0,6.0
    5.0,10.0,7.0"""

    df = pd.read_csv(StringIO(csv))
    df = df.fillna(df.median(numeric_only=True))
    print(df.corr().round(3))
    ```

    중앙값 대체는 평균 대체보다 이상치에 강건하지만, 결측이 정보를 담고 있을 때는 여전히 분산과 상관을 왜곡한다. 실제 분석에서는 모형 기반 대체나 다중대체가 낫다. 제12장을 보라.

---

**연습문제 5.**
`pd.Series.var()`의 기본값은 `ddof=1`이고 NumPy의 `np.var()`는 `ddof=0`이다. 값 다섯 개짜리 Series를 만들어 두 방식으로 분산을 계산하고, 어느 쪽이 불편추정량이며 그 이유가 무엇인지 설명하라.

??? success "풀이"
    ```python
    import numpy as np
    import pandas as pd
    s = pd.Series([2, 4, 4, 4, 5])
    print("pandas s.var()  :", s.var())            # ddof=1, divides by n-1=4
    print("numpy  np.var() :", np.var(s.values))   # ddof=0, divides by n=5
    ```

    `ddof=1`이면 $S^2 = \frac{1}{n-1}\sum(x_i - \bar x)^2$이 불편이다: $\mathbb{E}[S^2] = \sigma^2$. `ddof=0`이면 $\tilde S^2 = \frac{1}{n}\sum(x_i - \bar x)^2$인데, 이는 정규 자료에 대한 **최대가능도** 분산이지만 $(n-1)/n$배만큼 아래로 편향된다. 두 라이브러리의 기본값이 서로 반대라는 점은 늘 혼란의 원천이므로, 어느 쪽이 쓰이고 있는지 항상 확인하라.

---

**연습문제 6.**
두 DataFrame을 `df1.merge(df2, on="id", how="left")`로 조인한다. `how="left"`가 무엇을 하는지, 결과의 행 수를 무엇이 결정하는지, 그리고 병합 직후에 반드시 실행해야 할 진단 하나를 설명하라.

??? success "풀이"
    `how="left"`는 `df1`의 모든 행을 남기고 `id`를 기준으로 `df2`의 대응되는 열을 붙인다. `df2`에 대응이 없는 `df1`의 행은 새 열에 `NaN`이 들어간다. `df1`에 대응이 없는 `df2`의 행은 버려진다.

    출력 행 수가 `df1`의 행 수와 같아지는 것은 `df2`에서 `id`가 유일할 **때에 한해서**다. `df2`에서 `id`가 반복되면 `df1`의 각 행이 여러 `df2` 행과 대응되어 결과의 행 수가 `df1`보다 많아진다. 흔히 겪는 뜻밖의 일이다.

    병합 직후 실행할 진단:

    ```python
    assert df2["id"].is_unique, "right-side join key not unique — row count will inflate"
    merged = df1.merge(df2, on="id", how="left", indicator=True)
    print(merged["_merge"].value_counts())   # left_only / both / right_only
    ```

    `indicator=True` 플래그는 각 행이 어디서 왔는지 알려주는 `_merge` 열을 추가하여, 조용히 실패한 조인을 드러낸다.
