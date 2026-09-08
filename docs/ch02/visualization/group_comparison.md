# 집단 비교

## 개요

효과적인 자료 시각화에는 흔히 집단 간 분포, 도수, 관계를 비교하는 일이 필요하다. 이 절에서는 집단 비교를 위한 주요 시각화 도구인 산점도, 선그림, 막대그림, 원그래프, 쌍그림, 줄기잎그림, 점그림, 도수분포표, 모자이크 그림을 다룬다.

---

## 1. 선그림

선그림은 자료점을 순서대로 이어 그리므로 시계열과 추세를 보이는 데 이상적이다.

### 주가 예제

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 재현 가능하도록 주가를 모의생성한다. 실제 자료를 쓰는 법은 아래에 있다.
# 로그수익률을 정규분포에서 뽑고 누적합의 지수를 취하면
# 실제 주가와 비슷한 기하 브라운 운동 경로가 나온다.
rng = np.random.default_rng(42)
dates = pd.bdate_range("2023-01-01", "2023-12-31")   # 거래일만(주말 제외)
returns = rng.normal(0.0004, 0.02, len(dates))        # 일평균 0.04%, 일변동성 2%
price = 20000 * np.exp(np.cumsum(returns))            # 시작가 20,000원
s = pd.Series(price, index=dates, name="Close")

print(f"거래일 수: {len(s)}")
print(f"기간: {s.index.min().date()} ~ {s.index.max().date()}")
print(f"시작가 {s.iloc[0]:,.0f}  종료가 {s.iloc[-1]:,.0f}")
print(f"최저 {s.min():,.0f}  최고 {s.max():,.0f}")

fig, ax = plt.subplots(figsize=(12, 3))

# 선그림의 핵심: 점을 시간 순서대로 이어 그린다.
# 이 "이어 그리기" 때문에 추세와 변동이 눈에 들어온다.
ax.plot(s.index, s.values, color="blue", lw=1.2, label="Close")

# 특정 날짜를 강조하고 싶으면 그 점 하나만 따로 찍는다.
mark = pd.Timestamp("2023-10-19")
ax.plot([mark], [s.loc[mark]], "or", ms=8,
        label=f"{mark.date()}: {s.loc[mark]:,.0f}")

ax.set_xlabel("Date")
ax.set_ylabel("Price (KRW)")
ax.set_title("Simulated Daily Closing Price, 2023")
ax.legend()
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()
```

출력:

```
거래일 수: 260
기간: 2023-01-02 ~ 2023-12-29
시작가 20,130  종료가 17,330
최저 16,117  최고 22,786
```

![모의생성한 일별 종가 선그림](./img/gc_lineplot_timeseries.png)

선그림이 하는 일이 여기 다 있다. 260개의 점을 시간 순서로 이었을 뿐인데 **추세**(연중 하락)와 **변동성**(오르내림의 폭)이 한눈에 들어온다. 같은 260개 값을 히스토그램으로 그리면 이 두 가지가 모두 사라진다. 순서 정보가 버려지기 때문이다.

!!! tip "실제 주가로 바꾸려면"
    `yfinance`로 실제 자료를 받아 같은 그림을 그릴 수 있다.

    ```python
    import yfinance as yf

    data = yf.download('019170.KS', start='2023-01-01', end='2023-12-31')
    s = data['Close']
    s.index = s.index.tz_localize(None)
    # 이후 그리는 코드는 위와 같다
    ```

    다만 이 방식은 **네트워크와 외부 서비스에 의존한다.** 요청 제한에 걸리거나 종목 코드가 바뀌면 실행되지 않고, 자료가 갱신되면 그림도 달라진다. 교재의 예제를 모의자료로 둔 이유가 이것이다. 결과가 언제 실행해도 같아야 검증할 수 있다.

---

## 2. 산점도

산점도는 두 연속변수 사이의 관계를 보여준다. Matplotlib은 기능이 서로 다른 두 가지 방법을 제공한다.

### `ax.plot` 대 `ax.scatter`

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(0)
num_samples = 10
x = stats.norm().rvs(size=num_samples)
noise = 0.7 * stats.norm().rvs(size=num_samples)
y = 1 + 2 * x + noise

fig, (ax_plot, ax_scatter) = plt.subplots(1, 2, figsize=(12, 3))

point_sizes = 100 * stats.norm().rvs(size=num_samples) ** 2
color_values = stats.uniform().rvs(size=num_samples)

# ax.plot: 마커 속성이 모든 점에 똑같이 적용된다.
#   markersize=10  모든 점의 크기가 10
#   mec/mfc/mew    테두리색(red) / 채움색(blue) / 테두리굵기(3)
ax_plot.plot(x, y, 'o', markersize=10, mec="red", mfc="blue", mew=3)
ax_plot.set_title("Standard Plot\nFixed Marker Size")

# ax.scatter: 점마다 다른 값을 줄 수 있다.
#   s=배열  점마다 크기가 다르다  -> 세 번째 변수를 크기로 표현
#   c=배열  점마다 색이 다르다    -> 네 번째 변수를 색으로 표현
ax_scatter.scatter(x, y, s=point_sizes, c=color_values)
ax_scatter.set_title("Scatter Plot\nVariable Marker Size")

for ax in (ax_plot, ax_scatter):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine].set_visible(False)

plt.show()
```

![ax.plot과 ax.scatter의 비교](./img/gc_plot_vs_scatter.png)

**핵심 차이:** `ax.plot`은 마커의 크기와 색이 일정하여 단순한 점 표시에 이상적이다. `ax.scatter`는 각 점마다 크기와 색을 달리할 수 있어 자료의 차원을 추가로 시각화할 수 있다.

같은 10개 점을 그렸는데 오른쪽 그림은 **네 개의 변수**를 담는다. 가로축, 세로축, 점의 크기, 점의 색이다. 다만 크기와 색은 위치보다 읽기 어려우므로 보조 정보에만 쓰는 것이 좋다.

---

## 3. 막대그림

### 단일 집단 막대그림

```python
import matplotlib.pyplot as plt
import pandas as pd

data = {
    'Courses': ('Language', 'History', 'Geometry', 'Chemistry', 'Physics'),
    'Number of Teachers': (7, 3, 9, 1, 2)
}
df = pd.DataFrame(data).set_index('Courses')

fig, ax = plt.subplots(figsize=(12, 3))

# x           막대의 가로 위치 (0, 1, 2, ... 로 두고 눈금에 이름을 붙인다)
# height      막대의 높이 = 나타내려는 값
# tick_label  각 위치에 표시할 범주 이름
# width       막대 폭. 1.0이면 서로 붙고, 0.5면 절반 간격이 생긴다
ax.bar(x=range(len(df)), height=df["Number of Teachers"],
       tick_label=df.index, width=0.5)
ax.set_xlabel('Courses')
ax.set_ylabel('Number of Teachers')
ax.set_title("Favorite Courses of Teachers")
# 위·오른쪽 테두리를 지우면 자료 자체에 눈이 집중된다
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()
```

![단일 집단 막대그림](./img/gc_bar_simple.png)

막대그림에서 **세로축은 반드시 0에서 시작해야 한다.** 막대의 길이로 크기를 비교하는 그림이므로, 축을 잘라 내면 차이가 실제보다 크게 보인다. 선그림에서는 축을 잘라도 되지만 막대그림에서는 안 되는 이유다.

### 묶음 막대그림

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = {
    'Student': ['Brandon', 'Vanessa', 'Daniel', 'Kevin', 'William'],
    'Midterm': [85, 60, 60, 65, 100],
    'Final': [90, 90, 65, 80, 95]
}
df = pd.DataFrame(data).set_index('Student')

positions = np.arange(len(df))   # 학생마다 기준 위치 0, 1, 2, 3, 4
width = 0.3                      # 막대 하나의 폭

fig, ax = plt.subplots(figsize=(12, 3))

# 묶음 막대의 요령: 기준 위치에서 좌우로 반 폭씩 밀어 놓는다.
#   중간고사는 왼쪽(-width/2), 기말고사는 오른쪽(+width/2)
# 이렇게 하면 두 막대가 겹치지 않으면서 같은 학생끼리 붙어 있게 된다.
ax.bar(positions - width / 2, df['Midterm'], width=width, label="Midterm")
ax.bar(positions + width / 2, df['Final'], width=width, label="Final")
ax.set_xticks(positions)
ax.set_xticklabels(df.index)
ax.set_xlabel("Student")
ax.set_ylabel("Scores")
ax.set_title("Midterm and Final Scores")
ax.legend(title="Exam Type")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()
```

![묶음 막대그림](./img/gc_bar_grouped.png)

**묶음 막대는 개별 값을 비교할 때** 쓴다. 학생별로 중간고사와 기말고사를 나란히 놓아, Vanessa가 60에서 90으로 크게 올랐다는 사실이 곧바로 보인다.

### 분할(누적) 막대그림

분할 막대그림은 각 범주의 구성을 보여준다.

```python
import matplotlib.pyplot as plt
import numpy as np

labels = ("Yes", "No")
counts = (np.array([95, 90, 40]), np.array([5, 10, 60]))
age_groups = ("Adults", "Children", "Infants")

fig, ax = plt.subplots(figsize=(6, 3))

# 누적 막대의 요령: bottom 인자로 "이 막대가 어디서부터 시작할지"를 준다.
# 첫 막대는 0에서 시작하고, 다음 막대는 앞선 막대들의 합에서 시작한다.
bottom = np.zeros(3)

for label, count in zip(labels, counts):
    ax.bar(np.arange(3), count, width=0.5, bottom=bottom,
           tick_label=age_groups, label=label)
    bottom += count          # 다음 막대의 출발점을 위로 올린다

ax.set_title("Has Antibodies?")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# bbox_to_anchor로 범례를 그림 바깥 오른쪽에 내보낸다
ax.legend(title="Response", loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
plt.show()
```

![분할(누적) 막대그림](./img/gc_bar_stacked.png)

**누적 막대는 구성비를 볼 때** 쓴다. 세 집단 모두 막대의 전체 높이가 100으로 같아, 영아 집단만 항체 보유 비율이 40%로 낮다는 점이 바로 드러난다.

다만 한계가 있다. **맨 아래 조각을 뺀 나머지는 비교하기 어렵다.** 아래쪽 조각들은 시작점이 같아 길이를 견주기 쉽지만, 위쪽 조각은 시작점이 제각각이라 눈으로 길이를 비교하기가 힘들다. 여러 조각을 정확히 비교해야 한다면 묶음 막대가 낫다.

---

## 4. 원그래프

원그래프는 전체에 대한 비율을 보여준다. 범주 수가 적을 때 가장 잘 작동한다.

```python
import matplotlib.pyplot as plt

labels = 'Apples', 'Bananas', 'Cherries', 'Dates'
sizes = [215, 130, 245, 210]     # 개수. 합이 800이며 자동으로 백분율로 환산된다
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0, 0)         # 첫 조각만 0.1만큼 바깥으로 밀어 강조

fig, ax = plt.subplots()
ax.pie(sizes,
       explode=explode,
       labels=labels,
       colors=colors,
       autopct='%1.1f%%',        # 각 조각에 백분율 표시 (소수점 한 자리)
       shadow=True,
       startangle=140,           # 첫 조각이 시작하는 각도
       radius=1.5,
       counterclock=True)        # 반시계 방향으로 배치
ax.axis('equal')                 # 가로세로 비를 맞춰 원이 찌그러지지 않게 한다
ax.set_title('Fruit Distribution in Basket')
plt.show()
```

![원그래프](./img/gc_pie.png)

형식 문자열 `autopct='%1.1f%%'`는 각 조각에 백분율을 소수점 한 자리까지 표시한다. 마지막 `%%`는 퍼센트 기호 자체를 뜻한다.

!!! warning "원그래프는 웬만하면 쓰지 않는 편이 낫다"
    이 그림에서 Apples(26.9%)와 Dates(26.3%) 중 어느 쪽이 큰지 **각도만 보고** 판단할 수 있는가? 거의 불가능하다. 백분율 숫자를 읽어야 알 수 있다.

    사람은 **길이는 잘 비교하지만 각도와 넓이는 잘 비교하지 못한다.** 같은 자료를 막대그림으로 그리면 네 값의 순서와 차이가 즉시 보인다. 숫자를 적어 넣어야만 읽히는 그림이라면 그림의 역할을 못 하고 있는 셈이다.

    원그래프가 그나마 통하는 경우는 **범주가 두셋뿐이고 "절반쯤인가"처럼 대략적인 몫만 전달할 때**다. 그 밖에는 막대그림을 권한다.

---

## 5. 쌍그림

쌍그림은 모든 변수 쌍에 대한 산점도의 행렬을 만들고 대각선에는 히스토그램을 놓는다. 다변량 탐색에 매우 유용하다.

```python
import seaborn as sns
import pandas as pd

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url, index_col='PassengerId')

# 쌍그림은 수치형 변수만 받으므로 성별을 0/1로 부호화한다
df['Sex_int'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)

# 변수 k개를 주면 k x k 격자가 만들어진다.
#   대각선   그 변수 하나의 분포 (히스토그램)
#   비대각선 두 변수의 산점도
# 결측이 있는 행(Age가 비어 있는 177명)은 자동으로 빠진다.
sns.pairplot(df[["Survived", "Age", "Sex_int"]])
```

![쌍그림](./img/gc_pairplot.png)

세 변수의 격자에서 읽을 것이 몇 가지 있다.

- **Survived와 Sex_int의 산점도**는 네 귀퉁이에 점이 몰린 모양이다. 두 변수가 모두 0/1이기 때문인데, 왼쪽 위(여성·생존)와 오른쪽 아래(남성·사망)가 짙다. 성별과 생존이 강하게 얽혀 있다는 신호다.
- **Age의 히스토그램**(가운데 대각선)은 20–30대에 봉우리가 있고 오른쪽으로 약간 치우쳐 있다.
- **Age와 Survived**는 뚜렷한 관계가 보이지 않는다. 다만 이런 산점도는 한쪽이 0/1일 때 겹침이 심해 읽기 어려우므로, 앞 절의 바이올린 그림이나 상자그림이 더 낫다.

**쌍그림은 결론을 내는 도구가 아니라 훑어보는 도구다.** 어느 쌍을 더 들여다볼지 고르는 데 쓰고, 고른 뒤에는 그 쌍에 맞는 그림을 따로 그린다.

---

## 6. 줄기잎그림

줄기잎그림은 분포의 모양을 보여주면서 개별 자료값을 그대로 보존한다.

직접 만들어 보면 원리가 분명해진다. 각 값을 **줄기**(십의 자리)와 **잎**(일의 자리)으로 쪼개어, 같은 줄기끼리 한 줄에 모으면 된다.

```python
from collections import defaultdict

def stem_leaf(data, stem_unit=10):
    """줄기잎그림을 문자열로 만든다.

    stem_unit=10 이면 십의 자리가 줄기, 일의 자리가 잎이 된다.
    예: 65 -> 줄기 6, 잎 5
    """
    buckets = defaultdict(list)
    for v in sorted(data):                 # 잎이 오름차순이 되도록 먼저 정렬
        buckets[int(v) // stem_unit].append(int(v) % stem_unit)

    lines = ["줄기 | 잎", "-----+" + "-" * 20]
    # 값이 없는 줄기도 건너뛰지 않고 빈 줄로 남긴다.
    # 그래야 막대 길이가 도수에 비례해 분포 모양이 제대로 보인다.
    for stem in range(min(buckets), max(buckets) + 1):
        leaves = " ".join(str(leaf) for leaf in buckets.get(stem, []))
        lines.append(f"{stem:4d} | {leaves}")
    lines.append(f"\n줄기 단위 = {stem_unit}   (줄기 6, 잎 5  ->  65)")
    return "\n".join(lines)

scores = [65, 93, 45, 73, 99, 70, 88, 46, 75, 34, 83, 100, 88, 72, 70]
print(stem_leaf(scores))
```

출력:

```
줄기 | 잎
-----+--------------------
   3 | 4
   4 | 5 6
   5 |
   6 | 5
   7 | 0 0 2 3 5
   8 | 3 8 8
   9 | 3 9
  10 | 0

줄기 단위 = 10   (줄기 6, 잎 5  ->  65)
```

옆으로 누운 히스토그램처럼 읽으면 된다. 70대가 다섯 명으로 가장 많고, 50대는 한 명도 없다.

**히스토그램과 다른 점은 원자료가 남아 있다는 것이다.** 잎을 읽으면 70, 70, 72, 73, 75라는 실제 점수를 그대로 복원할 수 있다. 히스토그램은 구간에 몇 개인지만 알려 주고 값은 버린다. 자료가 수십 개 이하일 때 줄기잎그림이 유용한 이유다.

!!! note "`stemgraphic` 패키지"
    `stemgraphic` 라이브러리를 쓰면 그림 형태의 줄기잎그림을 얻을 수 있다.

    ```python
    import stemgraphic
    fig, ax = stemgraphic.stem_graphic(scores, scale=10)
    ```

    다만 별도 설치가 필요하고, 위 코드처럼 직접 만들면 줄기와 잎을 나누는 원리가 그대로 드러난다.

---

## 7. 점그림과 도수분포표

### 점그림

```python
import matplotlib.pyplot as plt

data = [5, 7, 5, 9, 7, 7, 6, 9, 9, 9, 10, 12, 12, 7]

# 값마다 몇 번 나왔는지 센다
age_freq = {}
for age in data:
    age_freq[age] = age_freq.get(age, 0) + 1

fig, ax = plt.subplots(figsize=(12, 3))

# 점그림의 요령: 같은 값을 세로로 쌓는다.
#   x = [age] * freq        가로 위치는 모두 같다
#   y = 1, 2, ..., freq     세로로 한 칸씩 올라간다
for age, freq in age_freq.items():
    ax.plot([age] * freq, range(1, freq + 1), 'ok')

ax.set_xlabel('Ages')
ax.set_ylabel('Number of Students')
ax.set_title("Ages of Students in Class")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticks([0, 1, 2, 3, 4])
ax.spines["bottom"].set_position("zero")   # 가로축을 y=0에 붙인다
plt.show()
```

![점그림](./img/gc_dotplot.png)

7세가 네 명, 9세가 네 명으로 가장 많고 나머지는 한둘씩이다. **자료가 적을 때는 점그림이 히스토그램보다 낫다.** 구간을 어떻게 나눌지 정할 필요가 없고, 점의 개수를 직접 셀 수 있기 때문이다.

---

## 8. 이원 도수분포표

### 도수분포표

```python
import pandas as pd

# 네 조합의 개수를 그대로 펼쳐 원자료 형태로 만든다.
#   SUV·사고 28명 / 비SUV·사고 35명 / SUV·무사고 97명 / 비SUV·무사고 104명
data = {'SUV':      28*['yes'] + 35*['no']  + 97*['yes'] + 104*['no'],
        'Accident': 28*['yes'] + 35*['yes'] + 97*['no']  + 104*['no']}
df = pd.DataFrame(data)

# crosstab이 두 범주형 변수를 교차하여 도수를 센다
dg = pd.crosstab(df.SUV, df.Accident, rownames=['SUV'], colnames=['Accident'])

dg.loc['TOTAL', :] = dg.sum()        # 열 방향 합계를 맨 아래 줄에 추가
dg.loc[:, 'TOTAL'] = dg.sum(axis=1)  # 행 방향 합계를 맨 오른쪽 열에 추가
dg = dg.astype(int)                  # 합계를 더하며 실수가 되었으므로 정수로 되돌린다
print(dg)
```

출력:

```
Accident   no  yes  TOTAL
SUV
no        104   35    139
yes        97   28    125
TOTAL     201   63    264
```

행과 열의 이름이 알파벳 순으로 정렬되어 `no`가 먼저 온다는 점에 주의하라. 오른쪽 아래 264는 전체 인원이다.

### 상대도수분포표

```python
# 모든 칸을 전체 인원(오른쪽 아래 칸)으로 나누면 비율이 된다
dh = dg / dg.loc['TOTAL', 'TOTAL']
print(dh)
```

출력:

```
Accident        no       yes     TOTAL
SUV
no        0.393939  0.132576  0.526515
yes       0.367424  0.106061  0.473485
TOTAL     0.761364  0.238636  1.000000
```

이 표에서 세 가지 확률을 곧바로 읽을 수 있다.

- **결합확률** $P(\text{SUV}, \text{사고}) = 0.106$ — 가운데 칸
- **주변확률** $P(\text{SUV}) = 0.473$ — 오른쪽 끝 열
- **조건부확률** $P(\text{사고} \mid \text{SUV}) = 28/125 = 0.224$ — 칸을 그 행의 합으로 나눈 값

마지막이 3장 조건부확률의 정의 $P(A \mid B) = P(A \cap B)/P(B)$를 표에서 실행한 것이다. $0.106/0.473 = 0.224$로 같은 값이 나온다.

비교해 보면 비SUV의 사고 비율은 $35/139 = 0.252$로 SUV보다 오히려 높다. 다만 이 표만으로 "SUV가 더 안전하다"고 말할 수는 없다. 1장에서 본 교란요인 — 주행거리, 운전자 연령, 도로 유형 — 이 통제되지 않았기 때문이다.

### 확률과의 연결

이원 도수분포표는 결합분포, 주변분포, 조건부분포와 직접 연결된다.

$$
\begin{array}{ll}
\text{Chain rule:} & p(x, y) = p(x) \, p(y|x) \\
\text{Marginalization:} & p(x) = \sum_y p(x, y) \\
\text{Conditioning:} & p(y|x) = \frac{p(x, y)}{p(x)}
\end{array}
$$

---

## 9. 자료 유형과 적절한 시각화

$$
\text{Data} \begin{cases}
\text{Categorical Data: Pie Chart, Bar Chart, Mosaic Plot, \ldots} \\
\text{Quantitative Data: Histogram, Box Plot, Stem Plot, Time Plot, \ldots}
\end{cases}
$$

## 요약

집단 비교의 과제가 다르면 필요한 시각화 도구도 다르다. 막대그림과 원그래프는 범주형 자료에 맞고, 히스토그램·상자그림·바이올린 그림은 연속분포의 모양을 드러내며, 산점도와 쌍그림은 두 변수의 관계를 노출하고, 도수분포표는 시각화와 확률을 잇는다. 알맞은 도구의 선택은 자료의 유형, 집단의 수, 그리고 강조하고 싶은 비교의 측면에 달려 있다.

## 연습문제

**연습문제 1.**
어떤 이원 도수분포표가 환자 200명에 대해 다음과 같은 도수를 보여준다.

|  | 처리 A | 처리 B | 합계 |
|:---|:---:|:---:|:---:|
| 호전됨 | 60 | 40 | 100 |
| 호전 안 됨 | 40 | 60 | 100 |
| 합계 | 100 | 100 | 200 |

(처리 A, 호전됨)의 결합상대도수와 처리 A가 주어졌을 때 호전될 조건부확률을 계산하라.

??? success "풀이"
    (처리 A, 호전됨)의 **결합상대도수**는

    $$
    \frac{60}{200} = 0.30
    $$

    이다. 처리 A가 주어졌을 때 호전될 **조건부확률**은

    $$
    P(\text{Improved} \mid \text{Treatment A}) = \frac{60}{100} = 0.60
    $$

    이다.

    비교하자면 처리 B가 주어졌을 때 호전될 조건부확률은 $40/100 = 0.40$이다. 처리 A의 호전율이 더 높아 보인다.

---

**연습문제 2.**
다음 각 상황에서 가장 적절한 그림 유형을 밝히고 그 선택을 정당화하라. (a) 환자 세 집단의 나이 분포 비교, (b) 12개월에 걸친 주가 변화 표시, (c) 경쟁하는 다섯 브랜드의 시장 점유율 표시.

??? success "풀이"
    **(a) 환자 세 집단의 나이 분포 비교:** **나란히 놓은 상자그림**이나 **바이올린 그림**. 상자그림은 집단 간 중심, 퍼짐, 이상치를 압축적으로 비교해 준다. 분포가 다봉일 수 있거나 전체 모양이 중요하다면 바이올린 그림이 낫다.

    **(b) 12개월에 걸친 주가:** x축에 시간, y축에 가격을 둔 **선그림**. 잇는 선이 시간적 연속성과 추세를 강조하므로 시계열 자료에는 선그림이 표준적인 선택이다.

    **(c) 다섯 브랜드의 시장 점유율:** **원그래프**나 **막대그림**. 범주 수가 적고 전체에 대한 비율을 보일 때는 원그래프가 적절하다. 다만 사람이 원의 각도보다 막대의 길이를 더 정확히 판단하므로 크기를 정확히 비교하기 쉬운 막대그림이 흔히 선호된다.

---

**연습문제 3.**
분할(누적) 막대그림이 세 부서의 응답 구성("동의", "중립", "반대")을 보여준다. 어떤 상황에서 누적 막대그림이 묶음 막대그림보다 유익하며, 반대의 경우는 언제인가?

??? success "풀이"
    **누적 막대그림**은 집단 간 **합계**를 비교하고 각 범주가 그 합계에 얼마나 기여하는지 보는 것이 주된 관심일 때 더 유익하다. 각 막대의 전체 크기와 각 구성요소의 비율을 쉽게 볼 수 있다.

    **묶음(나란히 놓은) 막대그림**은 집단 간 **개별 범주**를 비교하는 것이 주된 관심일 때 더 유익하다. 예를 들어 어느 부서의 "동의" 수가 가장 많은지 알고 싶다면, 같은 응답 범주의 막대가 서로 옆에 놓이므로 묶음 막대그림이 비교를 쉽게 만든다.

    요컨대 구성이 초점이면 누적을, 범주 수준의 직접 비교가 초점이면 묶음을 쓴다.

---

**연습문제 4.**
두 변수의 산점도가 강한 곡선(이차) 관계를 보이는데 피어슨 상관계수는 0에 가깝다. 이런 일이 왜 생기는지, 그리고 산점도와 함께 상관계수를 요약으로 쓰는 것에 대해 무엇을 시사하는지 설명하라.

??? success "풀이"
    피어슨 상관은 **선형** 연관만을 잰다. $X$와 $Y$의 관계가 대칭적인 곡선이면(예: 0을 중심으로 한 $Y = X^2$) 양의 절반과 음의 절반이 상쇄되어, $X$와 $Y$가 강하게 관련되어 있는데도 상관이 0 근처가 된다.

    이는 요약으로서의 상관에 언제나 산점도가 따라야 하는 이유를 보여준다. 산점도는 관계의 모양(선형, 곡선, 뭉침)을 드러내는 반면 상관계수는 선형 성분만 포착한다. 상관계수만 믿으면 두 변수가 무관하다는 잘못된 결론에 이를 수 있다.

---

**연습문제 5.**
앤스컴의 4중주는 요약통계량(평균, 분산, 상관, 회귀직선)이 **동일**하면서도 산점도는 극적으로 다른 네 자료로 이루어져 있다. 이것이 탐색적 자료분석에 대해 어떤 교훈을 주는가?

??? success "풀이"
    앤스컴(1973)은 각각 11개 점으로 이루어진 네 자료를 만들었는데, 소수점 둘째 자리까지 다음 통계량이 같다.

    - $x$의 평균 = 9, $y$의 평균 = 7.5.
    - $x$의 분산 = 11, $y$의 분산 = 4.12.
    - 상관 $r$ = 0.816.
    - 최소제곱 회귀: $y = 3 + 0.5x$.

    그런데 산점도는 다음과 같다.

    - 자료 1: 잡음이 있지만 대체로 선형인 관계.
    - 자료 2: 깔끔한 이차 곡선.
    - 자료 3: 완벽한 선형 추세에 이상치 하나가 회귀선을 밀어낸 형태.
    - 자료 4: 대부분의 점이 $x = 8$에 있고, $x = 19$의 영향력 있는 점 하나가 상관 전체를 좌우한다.

    **교훈:** 요약통계량은 아무리 포괄적이어도 질적으로 다른 자료 구조를 감출 수 있다. 같은 경고가 현대의 변형인 **데이터사우루스 도즌**(Matejka & Fitzmaurice 2017)에도 적용된다. 공룡 실루엣을 포함해 13가지 전혀 다른 모양이 동일한 요약통계량을 공유한다. **언제나 자료를 그려라.** 특히 요약통계량을 보고하거나 해석하기 전에 그래야 한다. 투키의 권고: "잘못된 질문에 대한 정확한 답보다, 흔히 모호하더라도 올바른 질문에 대한 근사적인 답이 훨씬 낫다."

---

**연습문제 6.**
하나의 그림에서 여러 집단을 비교할 때는 **다중비교 보정**이 필요해진다. 5개 집단의 쌍별 비교를 $\alpha = 0.05$로 수행한다면 적어도 하나의 거짓양성이 나올 가족단위 확률은 얼마이며, 어떻게 보정하겠는가?

??? success "풀이"
    집단이 5개이면 쌍별 비교의 수는 $\binom{5}{2} = 10$이다. (대략적인 근사로) 독립을 가정하면 10번의 검정에서 거짓양성이 *하나도* 없을 확률은 $(1 - 0.05)^{10} \approx 0.599$이다. 따라서 가족단위 오류율은 대략

    $$
    P(\text{at least one false positive}) \approx 1 - 0.599 = 0.401
    $$

    이다. 허위 유의성이 나올 확률이 40%로, 명목상의 5%보다 훨씬 높다.

    **보정 방법:**

    - **본페로니:** 각 쌍별 비교를 $\alpha/10 = 0.005$로 검정한다. 간단하고 보수적이다.
    - **투키의 HSD**(정직 유의차): 분산분석 이후 평균의 쌍별 비교에 대해 가족단위 오류를 정확히 통제한다. 본페로니보다 검정력이 높다.
    - **홀름–본페로니:** $p$-값을 정렬해 $\alpha$ 기준을 낮춰 가며 차례로 기각하는 단계적 절차. 본페로니보다 일률적으로 검정력이 높다.
    - **벤자미니–호크버그(FDR 통제):** 거짓양성이 하나라도 나올 확률이 아니라, 유의하다고 선언된 검정 가운데 거짓인 것의 *기대 비율*을 통제한다. 탐색적으로 검정을 많이 수행할 때 적절하다.

    시각화 측면: 하나의 그림에서 여러 집단을 비교할 때 모든 쌍별 $p$-값을 표기하지 *마라*. 관심 있는 비교 몇 개를 미리 지정하거나, 먼저 총괄 검정(분산분석, 크러스컬–월리스)을 하고 그것이 유의할 때만 쌍별 비교를 하라.
