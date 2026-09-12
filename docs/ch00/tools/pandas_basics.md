# pandas로 자료 다루기

pandas는 구조화된 자료를 불러오고, 정제하고, 변환하고, 요약하기 위한 이름표가 붙은 자료구조 — `Series`(1차원)와 `DataFrame`(2차원) — 를 제공한다. 원본 파일(CSV, parquet, SQL)과 NumPy의 배열 수준 계산이나 statsmodels·scikit-learn의 통계 모형 사이를 잇는 다리다. 이 책의 거의 모든 실증적 작업 흐름은 pandas를 임포트하는 것으로 시작해, 마지막 요약표를 그려낸 뒤에야 끝난다.

<div class="defn" markdown>

**정의 1.** [Series와 DataFrame]

**`Series`** 는 1차원의 이름표 붙은 배열, 즉 값과 인덱스의 조합이다. **`DataFrame`** 은 2차원 표로, 각 열이 하나의 `Series`이며 서로 자료형이 다를 수 있지만 모두 같은 행 인덱스를 공유한다. 개념적으로 DataFrame은 열들의 사전이고, 기계적으로는 각 열이 NumPy 배열로 뒷받침된다.

떠올릴 만한 심상은 NumPy 위에 얹은 SQL이다. 배열 층이 속도를 주고, 이름표 층이 통계 분석과 잘 맞는 선택·정렬·그룹화·조인·피벗 의미론을 준다.

</div>

## 설명

### 불러오기와 살펴보기

```python
import numpy as np
import pandas as pd

# 실제 작업에서는 df = pd.read_csv("data.csv")로 파일을 읽는다.
# 여기서는 결과를 바로 볼 수 있도록 작은 표를 직접 만든다.
rng = np.random.default_rng(0)
n = 12
demo = pd.DataFrame({
    "treatment": rng.choice(["control", "drug"], size=n),
    "x": rng.normal(10, 2, size=n).round(2),
    "outcome": rng.normal(50, 8, size=n).round(1),
})
demo.loc[[2, 7], "x"] = np.nan        # 결측값 두 개를 일부러 넣는다

print(demo.head())        # first 5 rows
print(demo.shape)         # (n_rows, n_cols)
print(demo.dtypes)        # type of each column
print(demo.describe().round(2))   # count, mean, std, min, quartiles, max
```

출력:

```
treatment      x  outcome
0      drug  12.61     53.3
1      drug  11.89     58.3
2      drug    NaN     49.0
3   control   7.47     60.9
4   control   8.75     44.7
(12, 3)
treatment     object
x            float64
outcome      float64
dtype: object
           x  outcome
count  10.00    12.00
mean    9.05    50.98
std     2.13     5.90
min     5.35    42.60
25%     7.77    45.90
50%     8.83    51.30
75%     9.90    54.28
max    12.61    60.90
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
print(demo.isna().sum())              # count of missing per column
print(len(demo.dropna()))             # drop rows with any NaN
print(len(demo.dropna(subset=["x"]))) # drop rows with NaN in 'x' only

filled = demo.fillna(demo.median(numeric_only=True))   # impute with median
print(filled["x"].isna().sum(), filled["x"].median())
```

출력:

```
treatment    0
x            2
outcome      0
dtype: int64
10
10
0 8.83
```

"옳은" 대체 전략이란 없다. 무엇을 고를지(삭제, 평균, 중앙값, 모형 기반, 다중대체)는 결측 기제에 달려 있다. pandas는 도구를 줄 뿐 결정은 사용자에게 맡긴다.

### 그룹화: 분할–적용–결합

pandas에서 가장 강력한 하나의 패턴이다.

```python
print(demo.groupby("treatment")["outcome"].agg(["count", "mean", "std"]).round(2))
```

출력:

```
count   mean   std
treatment                    
control        6  51.75  6.69
drug           6  50.22  5.52
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

<div class="codebox" markdown>

### 예제 1. pandas로 자료 다루기 { .eg }

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

출력:

```
count   mean    std
group                     
A         58  50.17  10.20
B         72  49.72   9.83
C         70  49.24   9.98

Values > 60: 31 / 200

Correlation:
        value  score
value  1.000 -0.111
score -0.111  1.000

Pivot:
 score_q    Q1    Q2    Q3    Q4
group                          
A        55.4  50.1  48.6  46.9
B        50.5  53.6  48.0  48.0
C        48.8  48.0  50.1  50.0
```

</div>

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
학생 다섯 명에 대해 `name`(문자열), `score`(정수), `passed`(불리언) 열을 갖는 `DataFrame`을 만들어라. `passed`가 `True`이고 **또한** `score`가 80보다 큰 행만 보이도록 걸러라.

</div>

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

    출력:

    ```
    name  score  passed
    0  Alice     92    True
    2  Carol     88    True
    4    Eve     95    True
    ```
    피연산자가 파이썬 스칼라가 아니라 pandas 불리언 Series이므로 `and`가 아니라 비트 연산자 `&`를 써야 한다. 각 비교식은 괄호로 묶어라. 연산자 우선순위상 `&`가 `>`보다 높기 때문이다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
다음 DataFrame이 주어졌을 때
```python
df = pd.DataFrame({"group": ["A","A","B","B","B"], "x": [1, 3, 2, 8, 5]})
```
각 그룹에 대해 개수, 표본평균, 표본분산, 그리고 최댓값에서 최솟값을 뺀 값을 계산하라. 하나의 DataFrame으로 반환하라.

</div>

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

    출력:

    ```
    n  mean  var  range
    group                     
    A      2   2.0  2.0      2
    B      3   5.0  9.0      6
    ```
    `agg`는 출력 열 이름을 문자열 집계 이름이나 호출 가능 객체에 대응시키는 키워드 인수를 받는다. 람다를 쓰면 이름 있는 함수를 따로 정의하지 않고 "범위"를 하나의 표현식으로 나타낼 수 있다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
`df.loc[]`과 `df.iloc[]`의 차이를 설명하라. 같은 DataFrame에 대해 둘이 서로 다른 행을 반환하는 예를 제시하라.

</div>

??? success "풀이"
    `loc`은 **이름표 기반**이다. `df.loc[0]`은 인덱스 이름표가 `0`인 행을 반환한다. `iloc`은 **정수 위치 기반**이다. `df.iloc[0]`은 인덱스 이름표와 무관하게 첫 번째 행을 반환한다.

    ```python
    df = pd.DataFrame({"A": [10, 20, 30]}, index=[2, 0, 1])
    print(df.loc[0])    # row labeled 0   → A = 20
    print(df.iloc[0])   # row at position 0 → A = 10
    ```

    출력:

    ```
    A    20
    Name: 0, dtype: int64
    A    10
    Name: 2, dtype: int64
    ```

    인덱스가 (기본값인) `RangeIndex(0, n)`이고 행 순서가 바뀌지 않았다면 둘은 일치한다. `df.sort_values()`, `df.set_index()`, 불리언 필터링을 거치고 나면 둘이 갈라지며, 이 둘을 조용히 혼동하는 것이 흔한 버그의 원천이다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
다음 CSV 형식 문자열을 불러와 결측값을 열 중앙값으로 채우고 상관행렬을 계산하라.

```text
x,y,z
1.0,2.0,3.0
2.0,,4.0
3.0,6.0,
4.0,8.0,6.0
5.0,10.0,7.0
```

</div>

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

    출력:

    ```
    x      y      z
    x  1.000  0.906  1.000
    y  0.906  1.000  0.906
    z  1.000  0.906  1.000
    ```

    중앙값 대체는 평균 대체보다 이상치에 강건하지만, 결측이 정보를 담고 있을 때는 여전히 분산과 상관을 왜곡한다. 실제 분석에서는 모형 기반 대체나 다중대체가 낫다. 제12장을 보라.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
`pd.Series.var()`의 기본값은 `ddof=1`이고 NumPy의 `np.var()`는 `ddof=0`이다. 값 다섯 개짜리 Series를 만들어 두 방식으로 분산을 계산하고, 어느 쪽이 불편추정량이며 그 이유가 무엇인지 설명하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    import pandas as pd
    s = pd.Series([2, 4, 4, 4, 5])
    print("pandas s.var()  :", s.var())            # ddof=1, divides by n-1=4
    print("numpy  np.var() :", np.var(s.values))   # ddof=0, divides by n=5
    ```

    출력:

    ```
    pandas s.var()  : 1.2
    numpy  np.var() : 0.96
    ```

    `ddof=1`이면 $S^2 = \frac{1}{n-1}\sum(x_i - \bar x)^2$이 불편이다: $\mathbb{E}[S^2] = \sigma^2$. `ddof=0`이면 $\tilde S^2 = \frac{1}{n}\sum(x_i - \bar x)^2$인데, 이는 정규 자료에 대한 **최대가능도** 분산이지만 $(n-1)/n$배만큼 아래로 편향된다. 두 라이브러리의 기본값이 서로 반대라는 점은 늘 혼란의 원천이므로, 어느 쪽이 쓰이고 있는지 항상 확인하라.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
두 DataFrame을 `df1.merge(df2, on="id", how="left")`로 조인한다. `how="left"`가 무엇을 하는지, 결과의 행 수를 무엇이 결정하는지, 그리고 병합 직후에 반드시 실행해야 할 진단 하나를 설명하라.

</div>

??? success "풀이"
    `how="left"`는 `df1`의 모든 행을 남기고 `id`를 기준으로 `df2`의 대응되는 열을 붙인다. `df2`에 대응이 없는 `df1`의 행은 새 열에 `NaN`이 들어간다. `df1`에 대응이 없는 `df2`의 행은 버려진다.

    출력 행 수가 `df1`의 행 수와 같아지는 것은 `df2`에서 `id`가 유일할 **때에 한해서**다. `df2`에서 `id`가 반복되면 `df1`의 각 행이 여러 `df2` 행과 대응되어 결과의 행 수가 `df1`보다 많아진다. 흔히 겪는 뜻밖의 일이다.

    병합 직후 실행할 진단:

    ```python
    import pandas as pd

    df1 = pd.DataFrame({"id": [1, 2, 3, 4], "x": [10, 20, 30, 40]})
    df2 = pd.DataFrame({"id": [2, 3, 5], "y": ["b", "c", "e"]})

    assert df2["id"].is_unique, "right-side join key not unique — row count will inflate"
    merged = df1.merge(df2, on="id", how="left", indicator=True)
    print(merged)
    print(merged["_merge"].value_counts())   # left_only / both / right_only
    ```

    출력:

    ```
    id   x    y     _merge
    0   1  10  NaN  left_only
    1   2  20    b       both
    2   3  30    c       both
    3   4  40  NaN  left_only
    _merge
    left_only     2
    both          2
    right_only    0
    Name: count, dtype: int64
    ```

    `indicator=True` 플래그는 각 행이 어디서 왔는지 알려주는 `_merge` 열을 추가하여, 조용히 실패한 조인을 드러낸다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
`SettingWithCopyWarning`은 pandas에서 가장 자주 마주치면서 가장 자주 무시되는 경고다. 이 경고를 재현하고, 무엇을 경고하는 것인지 설명하고, 올바른 대안을 제시하라.

</div>

??? success "풀이"
    ```python
    import pandas as pd
    import warnings

    df = pd.DataFrame({"g": ["a", "a", "b"], "x": [1., 2., 3.]})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sub = df[df.g == "a"]        # 걸러낸 결과 — 복사본일 수도, 뷰일 수도
        sub["x"] = 0                 # 여기서 경고
        print("발생한 경고:", [w.category.__name__ for w in caught])

    print("\n원본은 바뀌지 않았다:")
    print(df.to_string(index=False))
    ```

    출력:

    ```
    발생한 경고: ['SettingWithCopyWarning']

    원본은 바뀌지 않았다:
    g   x
    a 1.0
    a 2.0
    b 3.0
    ```

    **무엇을 경고하는가.** `df[df.g == "a"]`가 복사본을 돌려줄지 뷰를 돌려줄지 pandas 자신도 확실히 말할 수 없다. 따라서 `sub["x"] = 0`이 원본 `df`까지 바꿀지 아닐지가 정해져 있지 않다. 경고는 "이 대입이 의도한 곳에 갔는지 나는 모른다"는 뜻이다.

    위 실행에서는 원본이 그대로였지만, **그 결과에 의존해서는 안 된다.** 같은 코드가 다른 자료 모양이나 다른 pandas 버전에서 반대로 동작할 수 있다.

    **올바른 대안 두 가지.** 무엇을 하려는지에 따라 갈린다.

    ```python
    import pandas as pd

    df = pd.DataFrame({"g": ["a", "a", "b"], "x": [1., 2., 3.]})

    # (1) 원본을 정말로 고치려는 경우 — .loc 으로 한 번에
    df.loc[df.g == "a", "x"] = 0
    print("원본을 고친 경우:")
    print(df.to_string(index=False))

    # (2) 부분집합만 따로 다루려는 경우 — 복사본임을 명시
    df2 = pd.DataFrame({"g": ["a", "a", "b"], "x": [1., 2., 3.]})
    sub = df2[df2.g == "a"].copy()
    sub["x"] = 0
    print("\n복사본을 고친 경우 — 원본은 그대로:")
    print(df2.to_string(index=False))
    ```

    출력:

    ```
    원본을 고친 경우:
    g   x
    a 0.0
    a 0.0
    b 3.0

    복사본을 고친 경우 — 원본은 그대로:
    g   x
    a 1.0
    a 2.0
    b 3.0
    ```

    규칙은 간단하다. **연쇄 대입(`df[...][...] = ...`)을 쓰지 마라.** 원본을 고치려면 `.loc[행조건, 열] = 값` 한 번으로 끝내고, 부분집합을 따로 가지고 놀 것이라면 `.copy()`를 명시하라.

    이것은 앞 절 NumPy 연습문제의 뷰/복사본 문제와 정확히 같은 구조다. 다만 pandas는 어느 쪽인지조차 보장하지 않아 한 겹 더 나쁘다. (pandas 3.0의 Copy-on-Write 방식은 "언제나 복사본처럼 동작한다"로 규칙을 통일해 이 모호함을 없앤다.) $\square$

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
`groupby`로 심슨의 역설을 재현하라. 두 치료법의 성공률을 층별로, 그리고 전체로 계산하고 왜 순서가 뒤집히는지 설명하라. `transform`과 `agg`의 차이도 함께 보여라.

</div>

??? success "풀이"
    ```python
    import pandas as pd

    df = pd.DataFrame(
        [("A", "작은 결석",  81,  87),
         ("A", "큰 결석",   192, 263),
         ("B", "작은 결석", 234, 270),
         ("B", "큰 결석",    55,  80)],
        columns=["치료법", "결석 크기", "성공", "전체"])
    df["성공률"] = df["성공"] / df["전체"]

    print("층별 성공률")
    print(df.to_string(index=False))

    total = df.groupby("치료법")[["성공", "전체"]].sum()
    total["성공률"] = total["성공"] / total["전체"]
    print("\n전체 성공률")
    print(total.to_string())
    ```

    출력:

    ```
    층별 성공률
    치료법 결석 크기  성공  전체      성공률
      A 작은 결석  81  87 0.931034
      A  큰 결석 192 263 0.730038
      B 작은 결석 234 270 0.866667
      B  큰 결석  55  80 0.687500

    전체 성공률
          성공   전체       성공률
    치료법                    
    A    273  350  0.780000
    B    289  350  0.825714
    ```

    **역설.** 치료법 A가 작은 결석에서도($93.1\%$ 대 $86.7\%$), 큰 결석에서도($73.0\%$ 대 $68.8\%$) 더 낫다. 그런데 전체를 합치면 A가 $78.0\%$로 B의 $82.6\%$보다 나쁘다.

    **왜 뒤집히는가.** 층에 따라 성공률 자체가 크게 다르고(작은 결석이 훨씬 잘 낫는다), 두 치료법이 **층에 배정된 비율이 정반대**이기 때문이다. A는 환자의 $75\%$가 어려운 큰 결석이고, B는 $77\%$가 쉬운 작은 결석이다. 전체 성공률은 층별 성공률의 가중평균인데, 그 **가중치가 치료법마다 다르다.**

    실제 자료에서 이런 배정은 우연이 아니다. 의사가 심한 환자에게 더 강한 치료법 A를 쓴 것이며, 결석 크기가 **교란변수**다. 층별 비교가 옳은 비교이고 전체 비교는 오해를 부른다. 무작위배정이 하는 일이 바로 층 구성을 두 군에서 같게 만드는 것이다.

    **`agg` 대 `transform`.**

    ```python
    import pandas as pd

    d = pd.DataFrame({"g": list("aabbb"), "x": [1., 3., 2., 8., 5.]})

    print("agg — 그룹당 한 행:")
    print(d.groupby("g")["x"].agg(["mean", "std"]).to_string())

    d["z"] = d.groupby("g")["x"].transform(lambda s: (s - s.mean()) / s.std(ddof=1))
    print("\ntransform — 원래 행 수를 유지하며 그룹별로 표준화:")
    print(d.to_string(index=False))
    ```

    출력:

    ```
    agg — 그룹당 한 행:
       mean       std
    g                
    a   2.0  1.414214
    b   5.0  3.000000

    transform — 원래 행 수를 유지하며 그룹별로 표준화:
    g   x         z
    a 1.0 -0.707107
    a 3.0  0.707107
    b 2.0 -1.000000
    b 8.0  1.000000
    b 5.0  0.000000
    ```

    `agg`는 그룹을 하나의 값으로 **줄이고**($5$행 → $2$행), `transform`은 결과를 원래 행에 **되돌려 붙인다**($5$행 → $5$행). 그룹 내 표준화, 그룹 평균 대비 편차, 그룹 평균으로 결측 채우기처럼 "그룹 통계량을 원래 자료에 다시 쓰는" 작업은 모두 `transform`이다. $\square$

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
같은 자료를 넓은 형식과 긴 형식으로 오가는 방법을 보여라. 반복측정 자료를 `melt`로 긴 형식으로 바꾸고 `pivot`으로 되돌린 뒤, 통계 도구들이 왜 긴 형식을 요구하는지 설명하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    import pandas as pd

    wide = pd.DataFrame({"id": [1, 2, 3],
                         "t0": [5., 6., 7.],
                         "t1": [6., 8., 7.],
                         "t2": [9., 9., 8.]})
    print("넓은 형식 — 시점마다 열 하나")
    print(wide.to_string(index=False))

    long = wide.melt(id_vars="id", var_name="시점", value_name="측정")
    print(f"\n긴 형식 — 관측마다 행 하나  {wide.shape} → {long.shape}")
    print(long.to_string(index=False))

    back = long.pivot(index="id", columns="시점", values="측정")
    print(f"\n되돌리기 성공: {np.allclose(back.values, wide.set_index('id').values)}")
    ```

    출력:

    ```
    넓은 형식 — 시점마다 열 하나
     id  t0  t1  t2
      1 5.0 6.0 9.0
      2 6.0 8.0 9.0
      3 7.0 7.0 8.0

    긴 형식 — 관측마다 행 하나  (3, 4) → (9, 3)
     id 시점  측정
      1 t0 5.0
      2 t0 6.0
      3 t0 7.0
      1 t1 6.0
      2 t1 8.0
      3 t1 7.0
      1 t2 9.0
      2 t2 9.0
      3 t2 8.0

    되돌리기 성공: True
    ```

    **왜 긴 형식인가.** 넓은 형식에서는 "시점"이라는 변수가 **열 이름 속에 숨어** 있다. `t0`, `t1`, `t2`는 값이지 변수명이 아니다. 긴 형식에서는 시점이 값을 갖는 어엿한 열이 되므로 다음이 가능해진다.

    - `statsmodels`의 수식 표기: `측정 ~ 시점` 또는 반복측정 혼합모형 `측정 ~ 시점 + (1|id)`
    - `groupby("시점")`으로 시점별 요약
    - seaborn의 `x="시점", y="측정", hue=...` 같은 매핑

    이것이 해들리 위컴의 **정돈된 자료(tidy data)** 원칙이다. 각 변수는 하나의 열, 각 관측은 하나의 행. 통계 도구 대부분이 이 형식을 전제한다.

    넓은 형식이 유리한 곳도 있다. 사람이 표로 읽기에 좋고, 상관행렬 계산(`wide.corr()`)이나 대응표본 $t$ 검정처럼 짝지어진 열이 필요한 계산에는 넓은 형식이 자연스럽다. **분석 단계마다 필요한 형식이 다르므로 두 방향 변환을 모두 익혀 두어야 한다.**

    실무 주의사항. `pivot`은 `index`와 `columns`의 조합이 유일할 것을 요구하며, 중복이 있으면 오류를 낸다. 중복을 집계로 처리하려면 `pivot_table(aggfunc=...)`을 쓴다. 오류가 나는 쪽이 낫다. 중복이 있다는 것은 보통 자료 구조를 잘못 이해했다는 신호이기 때문이다. $\square$

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
`NaN`의 동작 방식을 정확히 파악하라. `count`, `mean`, `sum`, `groupby`가 결측을 각각 어떻게 다루는지 확인하고, 문자열 열을 `category`로 바꾸었을 때의 메모리 이득을 측정하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    import pandas as pd

    s = pd.Series([1., np.nan, 3.])
    print(f"NaN == NaN 인가: {np.nan == np.nan}")
    print(f"count {s.count()}   mean {s.mean()}   sum {s.sum()}   (결측을 건너뛴다)")
    print(f"mean(skipna=False): {s.mean(skipna=False)}")

    all_nan = pd.Series([np.nan, np.nan])
    print(f"\n전부 결측일 때  sum = {all_nan.sum()}   mean = {all_nan.mean()}")

    g = pd.DataFrame({"k": ["a", None, "a"], "v": [1., 2., 3.]})
    print(f"\ngroupby 기본값(dropna=True):\n{g.groupby('k')['v'].sum().to_string()}")
    print(f"\ngroupby(dropna=False):\n{g.groupby('k', dropna=False)['v'].sum().to_string()}")
    ```

    출력:

    ```
    NaN == NaN 인가: False
    count 2   mean 2.0   sum 4.0   (결측을 건너뛴다)
    mean(skipna=False): nan

    전부 결측일 때  sum = 0.0   mean = nan

    groupby 기본값(dropna=True):
    k
    a    4.0

    groupby(dropna=False):
    k
    a      4.0
    NaN    2.0
    ```

    **함정 세 가지.**

    첫째, `NaN != NaN`이므로 `df[df.x == np.nan]`은 **언제나 빈 결과**를 준다. 결측을 찾으려면 `df.x.isna()`를 써야 한다.

    둘째, 전부 결측인 Series의 합이 `NaN`이 아니라 **$0.0$**이다. 빈 합의 항등원이 $0$이라는 규약 때문인데, 그룹별 합계를 낼 때 "자료가 하나도 없었다"와 "합이 정말 0이다"가 구별되지 않는다. `mean`은 올바르게 `NaN`을 준다.

    셋째, `groupby`가 기본적으로 결측 키를 **말없이 버린다.** 위에서 `v = 2.0`인 행이 사라져 총합이 $6$이 아니라 $4$가 된다. 그룹 합계가 전체 합계와 안 맞는다면 이것부터 의심하라. `dropna=False`로 결측을 하나의 그룹으로 남길 수 있다.

    **메모리: `category` 자료형.**

    ```python
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    s = pd.Series(rng.choice(["서울", "부산", "대구"], 100_000))

    mb_object = s.memory_usage(deep=True) / 1e6
    mb_category = s.astype("category").memory_usage(deep=True) / 1e6
    print(f"object   {mb_object:.3f} MB")
    print(f"category {mb_category:.3f} MB   ({mb_object / mb_category:.0f}배 절약)")
    ```

    출력:

    ```
    object   7.000 MB
    category 0.100 MB   (70배 절약)
    ```

    `object` 자료형은 문자열마다 파이썬 객체 포인터를 저장하지만, `category`는 범주 목록을 한 번만 두고 나머지는 작은 정수 코드로 저장한다. 서로 다른 값의 개수가 전체 행 수보다 훨씬 적을 때 효과가 크다.

    다만 `deep=True`가 필수다. 이것 없이 재면 `object` 열은 포인터 크기만 세어 실제 사용량을 크게 과소평가한다.

    `category`는 메모리 외에 의미도 담는다. 2장에서 보듯 순서형 범주로 지정하면 크기 비교가 가능해지고, `groupby`가 관측되지 않은 범주까지 유지할 수 있다. $\square$

---

## 정리하며

pandas 는 NumPy 배열에 **이름표**를 붙인 것이다. 그 하나의 차이가 실제 자료를 다루는 일을 가능하게 만든다.

- **`Series` 와 `DataFrame`.** 값에 인덱스가 따라붙고, 열마다 자료형이 달라도 된다. 연산할 때 인덱스를 기준으로 정렬되므로 행 순서를 맞추는 실수가 줄어든다.
- **선택의 세 갈래.** 이름표면 `.loc`, 정수 위치면 `.iloc`, 조건이면 불리언 색인이다. 이 셋을 섞어 쓰다 생기는 혼동이 초보자의 오류 대부분이다.
- **결측값.** `NaN` 은 전염된다. 무엇으로 채울지, 아니면 버릴지를 **정하는 것 자체가 분석의 일부**이며 조용히 넘어가서는 안 된다.
- **분할–적용–결합.** `groupby` 한 줄이 집단별 요약을 만든다. 뒤에 나올 분산분석과 층화 분석이 모두 이 구조다.
- **`ddof` 를 확인하라.** pandas 의 `var`·`std`·`cov` 는 기본이 `ddof=1`(표본, $n-1$ 로 나눔)이고 NumPy 는 기본이 `ddof=0`(모집단, $n$ 으로 나눔)이다. **같은 자료에서 두 라이브러리가 다른 값을 준다.** 7장의 베셀 보정이 바로 이 차이를 다룬다.

다음 절 **Matplotlib으로 기본 시각화하기**는 여기까지 정리한 자료를 그림으로 옮긴다. 수를 보기 전에 그림을 보는 습관이 2장 전체의 전제다.
