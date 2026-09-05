# 집단 비교

## 개요

효과적인 자료 시각화에는 흔히 집단 간 분포, 도수, 관계를 비교하는 일이 필요하다. 이 절에서는 집단 비교를 위한 주요 시각화 도구인 산점도, 선그림, 막대그림, 원그래프, 쌍그림, 줄기잎그림, 점그림, 도수분포표, 모자이크 그림을 다룬다.

---

## 1. 선그림

선그림은 자료점을 순서대로 이어 그리므로 시계열과 추세를 보이는 데 이상적이다.

### 주가 예제

```python
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def download_stock_prices(ticker, start='2023-01-01', end='2023-12-31'):
    return yf.download(ticker, start=start, end=end)

def display_stock_prices(data, ticker, ticker_name):
    data.index = data.index.tz_localize(None)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(data['Close'], label=ticker_name, color='blue')

    date_to_mark = pd.to_datetime('2023-10-19').tz_localize(None)
    if date_to_mark in data.index:
        ax.plot([date_to_mark], [data.loc[date_to_mark, 'Close']],
                'or', label=f'{ticker_name} on {date_to_mark.date()}')

    ax.set_xlabel('Date')
    ax.set_ylabel(f'{ticker_name} Price (KRW)')
    ax.set_title(f'{ticker_name} Prices in 2023')
    ax.legend()
    plt.show()

ticker = '019170.KS'
data = download_stock_prices(ticker)
display_stock_prices(data, ticker, "Shinpoong")
```

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

# ax.plot: Fixed marker properties
ax_plot.plot(x, y, 'o', markersize=10, mec="red", mfc="blue", mew=3)
ax_plot.set_title("Standard Plot\nFixed Marker Size")

# ax.scatter: Variable marker properties
ax_scatter.scatter(x, y, s=point_sizes, c=color_values)
ax_scatter.set_title("Scatter Plot\nVariable Marker Size")

for ax in (ax_plot, ax_scatter):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine].set_visible(False)

plt.show()
```

**핵심 차이:** `ax.plot`은 마커의 크기와 색이 일정하여 단순한 점 표시에 이상적이다. `ax.scatter`는 각 점마다 크기와 색을 달리할 수 있어 자료의 차원을 추가로 시각화할 수 있다.

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
ax.bar(x=range(len(df)), height=df["Number of Teachers"],
       tick_label=df.index, width=0.5)
ax.set_xlabel('Courses')
ax.set_ylabel('Number of Teachers')
ax.set_title("Favorite Courses of Teachers")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()
```

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

positions = np.arange(len(df))
width = 0.3

fig, ax = plt.subplots(figsize=(12, 3))
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

### 분할(누적) 막대그림

분할 막대그림은 각 범주의 구성을 보여준다.

```python
import matplotlib.pyplot as plt
import numpy as np

labels = ("Yes", "No")
counts = (np.array([95, 90, 40]), np.array([5, 10, 60]))
age_groups = ("Adults", "Children", "Infants")

fig, ax = plt.subplots(figsize=(6, 3))
bottom = np.zeros(3)

for label, count in zip(labels, counts):
    ax.bar(np.arange(3), count, width=0.5, bottom=bottom,
           tick_label=age_groups, label=label)
    bottom += count

ax.set_title("Has Antibodies?")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(title="Response", loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
plt.show()
```

---

## 4. 원그래프

원그래프는 전체에 대한 비율을 보여준다. 범주 수가 적을 때 가장 잘 작동한다.

```python
import matplotlib.pyplot as plt

labels = 'Apples', 'Bananas', 'Cherries', 'Dates'
sizes = [215, 130, 245, 210]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0, 0)

fig, ax = plt.subplots()
ax.pie(sizes, explode=explode, labels=labels, colors=colors,
       autopct='%1.1f%%', shadow=True, startangle=140,
       radius=1.5, counterclock=True)
ax.axis('equal')
ax.set_title('Fruit Distribution in Basket')
plt.show()
```

형식 문자열 `autopct='%1.1f%%'`는 각 조각에 백분율을 소수점 한 자리까지 표시한다.

---

## 5. 쌍그림

쌍그림은 모든 변수 쌍에 대한 산점도의 행렬을 만들고 대각선에는 히스토그램을 놓는다. 다변량 탐색에 매우 유용하다.

```python
import seaborn as sns
import pandas as pd

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url, index_col='PassengerId')
df['Sex_int'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)

sns.pairplot(df[["Survived", "Age", "Sex_int"]])
```

---

## 6. 줄기잎그림

줄기잎그림은 분포의 모양을 보여주면서 개별 자료값을 그대로 보존한다.

```python
import stemgraphic

data = [65, 93, 45, 73, 99, 70, 88, 46, 75, 34, 83, 100, 88, 72, 70]
fig, ax = stemgraphic.stem_graphic(data, scale=10,
                                    title="Stem-and-Leaf Plot of Student Scores")
```

---

## 7. 점그림과 도수분포표

### 점그림

```python
import matplotlib.pyplot as plt

data = [5, 7, 5, 9, 7, 7, 6, 9, 9, 9, 10, 12, 12, 7]

age_freq = {}
for age in data:
    age_freq[age] = age_freq.get(age, 0) + 1

fig, ax = plt.subplots(figsize=(12, 3))
for age, freq in age_freq.items():
    ax.plot([age] * freq, range(1, freq + 1), 'ok')

ax.set_xlabel('Ages')
ax.set_ylabel('Number of Students')
ax.set_title("Ages of Students in Class")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticks([0, 1, 2, 3, 4])
ax.spines["bottom"].set_position("zero")
plt.show()
```

---

## 8. 이원 도수분포표

### 도수분포표

```python
import pandas as pd

data = {'SUV': 28*['yes'] + 35*['no'] + 97*['yes'] + 104*['no'],
        'Accident': 28*['yes'] + 35*['yes'] + 97*['no'] + 104*['no']}
df = pd.DataFrame(data)

dg = pd.crosstab(df.SUV, df.Accident, rownames=['SUV'], colnames=['Accident'])
dg.loc['TOTAL', :] = dg.sum()
dg.loc[:, 'TOTAL'] = dg.sum(axis=1)
dg = dg.astype(int)
print(dg)
```

### 상대도수분포표

```python
dh = dg / dg.loc['TOTAL', 'TOTAL']
print(dh)
```

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

??? success "연습문제 1 풀이"
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

??? success "연습문제 2 풀이"
    **(a) 환자 세 집단의 나이 분포 비교:** **나란히 놓은 상자그림**이나 **바이올린 그림**. 상자그림은 집단 간 중심, 퍼짐, 이상치를 압축적으로 비교해 준다. 분포가 다봉일 수 있거나 전체 모양이 중요하다면 바이올린 그림이 낫다.

    **(b) 12개월에 걸친 주가:** x축에 시간, y축에 가격을 둔 **선그림**. 잇는 선이 시간적 연속성과 추세를 강조하므로 시계열 자료에는 선그림이 표준적인 선택이다.

    **(c) 다섯 브랜드의 시장 점유율:** **원그래프**나 **막대그림**. 범주 수가 적고 전체에 대한 비율을 보일 때는 원그래프가 적절하다. 다만 사람이 원의 각도보다 막대의 길이를 더 정확히 판단하므로 크기를 정확히 비교하기 쉬운 막대그림이 흔히 선호된다.

---

**연습문제 3.**
분할(누적) 막대그림이 세 부서의 응답 구성("동의", "중립", "반대")을 보여준다. 어떤 상황에서 누적 막대그림이 묶음 막대그림보다 유익하며, 반대의 경우는 언제인가?

??? success "연습문제 3 풀이"
    **누적 막대그림**은 집단 간 **합계**를 비교하고 각 범주가 그 합계에 얼마나 기여하는지 보는 것이 주된 관심일 때 더 유익하다. 각 막대의 전체 크기와 각 구성요소의 비율을 쉽게 볼 수 있다.

    **묶음(나란히 놓은) 막대그림**은 집단 간 **개별 범주**를 비교하는 것이 주된 관심일 때 더 유익하다. 예를 들어 어느 부서의 "동의" 수가 가장 많은지 알고 싶다면, 같은 응답 범주의 막대가 서로 옆에 놓이므로 묶음 막대그림이 비교를 쉽게 만든다.

    요컨대 구성이 초점이면 누적을, 범주 수준의 직접 비교가 초점이면 묶음을 쓴다.

---

**연습문제 4.**
두 변수의 산점도가 강한 곡선(이차) 관계를 보이는데 피어슨 상관계수는 0에 가깝다. 이런 일이 왜 생기는지, 그리고 산점도와 함께 상관계수를 요약으로 쓰는 것에 대해 무엇을 시사하는지 설명하라.

??? success "연습문제 4 풀이"
    피어슨 상관은 **선형** 연관만을 잰다. $X$와 $Y$의 관계가 대칭적인 곡선이면(예: 0을 중심으로 한 $Y = X^2$) 양의 절반과 음의 절반이 상쇄되어, $X$와 $Y$가 강하게 관련되어 있는데도 상관이 0 근처가 된다.

    이는 요약으로서의 상관에 언제나 산점도가 따라야 하는 이유를 보여준다. 산점도는 관계의 모양(선형, 곡선, 뭉침)을 드러내는 반면 상관계수는 선형 성분만 포착한다. 상관계수만 믿으면 두 변수가 무관하다는 잘못된 결론에 이를 수 있다.

---

**연습문제 5.**
앤스컴의 4중주는 요약통계량(평균, 분산, 상관, 회귀직선)이 **동일**하면서도 산점도는 극적으로 다른 네 자료로 이루어져 있다. 이것이 탐색적 자료분석에 대해 어떤 교훈을 주는가?

??? success "연습문제 5 풀이"
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

??? success "연습문제 6 풀이"
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
