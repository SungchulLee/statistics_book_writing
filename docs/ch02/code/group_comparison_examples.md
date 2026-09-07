# 집단 비교 예제

## 개요

집단 간 분포를 비교하는 것은 기술통계의 핵심 과제다. 항공사, 우편번호, 신용등급 같은 범주형 변수로 자료를 나누었을 때 자연스럽게 떠오르는 질문은 이것이다. 집단들이 중심, 퍼짐, 모양에서 다른가?

시각적 집단 비교를 위한 두 가지 표준 도구는 **상자그림**과 **바이올린 그림**이다. 상자그림은 중앙값, 사분위수, 이상치를 압축적으로 요약하고, 바이올린 그림은 상자그림이 감추는 다봉성 같은 특징을 드러내며 분포의 전체 모양을 보여준다.

---

## 예제 1: 항공사별 지연

### 설정

항공편 지연은 승객의 이동 계획과 항공사 운영에 중요하다. 여기서는 네 항공사의 일간 지연율 분포를 비교한다. 각 항공사의 지연은 서로 다른 모양 및 척도 모수를 갖는 감마분포에서 뽑았으며, 이는 서로 다른 운영 특성을 반영한다.

### 코드

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
airlines = ['American', 'Delta', 'Southwest', 'United']
n_obs = 100

data_list = []
params = {
    'American':  (2, 3),      # shape, scale — more frequent delays
    'Delta':     (1.5, 2.5),  # moderate
    'Southwest': (1.2, 2),    # fewer delays
    'United':    (1.8, 3.2),  # high variability
}
for airline in airlines:
    shape, scale = params[airline]
    delays = np.random.gamma(shape=shape, scale=scale, size=n_obs)
    for delay in delays:
        data_list.append({'airline': airline, 'pct_carrier_delay': delay})

airline_stats = pd.DataFrame(data_list)
```

### 상자그림으로 보기

```python
fig, ax = plt.subplots(figsize=(8, 5))
airline_stats.boxplot(by='airline', column='pct_carrier_delay', ax=ax)
ax.set_xlabel('Airline')
ax.set_ylabel('Daily % of Delayed Flights')
ax.set_title('Airline Delay Comparison: Boxplots')
plt.suptitle('')
plt.tight_layout()
plt.show()
```

상자그림 읽기:

- **상자**는 $Q_1$에서 $Q_3$까지 뻗는다(지연의 가운데 50%).
- 상자 안의 **선**이 중앙값이다.
- **수염**은 $1.5 \times \text{IQR}$ 안의 가장 극단적인 관측값까지 뻗는다.
- 수염 너머의 점들이 **이상치**다.

### 바이올린 그림으로 보기

```python
fig, ax = plt.subplots(figsize=(8, 5))
sns.violinplot(data=airline_stats, x='airline', y='pct_carrier_delay',
               ax=ax, inner='quartile', color='lightblue')
ax.set_xlabel('Airline')
ax.set_ylabel('Daily % of Delayed Flights')
ax.set_title('Airline Delay Comparison: Violin Plots')
plt.tight_layout()
plt.show()
```

바이올린 그림 읽기:

- **넓은 부분**은 그 지연 수준에 관측값이 많음을 나타낸다.
- **좁은 부분**은 관측값이 적음을 나타낸다.
- **이봉** 모양(혹이 둘)은 전형적인 지연 시나리오가 둘임을 시사한다.
- **치우친** 모양은 지연 분포가 비대칭임을 나타낸다.

### 해석

항공사끼리 비교하면 위치와 퍼짐 모두에서 차이가 드러난다. 중앙값이 높고 상자가 넓은 항공사는 체계적으로 지연이 더 심하다. 바이올린 그림은 뉘앙스를 더한다. 두 항공사의 중앙값이 비슷해도 모양이 매우 다를 수 있는데, 상자그림만으로는 이를 볼 수 없다.

---

## 예제 2: 우편번호별 주택 가치

### 설정

부동산 투자자는 동네가 주택 가치에 어떤 영향을 주는지 알고 싶어 한다. 여기서는 워싱턴주 킹 카운티의 우편번호 네 곳에 대해 모의 생성한 과세평가 주택 가치를 비교한다.

### 코드

```python
np.random.seed(123)
zip_codes = [98188, 98105, 98108, 98126]
n_homes = 150

data_list = []
for zip_code in zip_codes:
    base_price = 300_000 if zip_code in [98105, 98108] else 450_000
    prices = np.random.normal(base_price, 100_000, n_homes)
    prices = np.clip(prices, 50_000, 2_000_000)
    for price in prices:
        data_list.append({'ZipCode': str(zip_code), 'TaxAssessedValue': price})

housing = pd.DataFrame(data_list)

fig, ax = plt.subplots(figsize=(8, 5))
housing.boxplot(by='ZipCode', column='TaxAssessedValue', ax=ax)
ax.set_xlabel('Zip Code')
ax.set_ylabel('Tax Assessed Value (\$)')
ax.set_title('Housing Values Across Neighborhoods')
plt.suptitle('')
plt.tight_layout()
plt.show()
```

### 해석

기준 가격이 높은 우편번호는 상자가 위쪽으로 이동해 나타난다. 상자의 폭이 비슷하면 변동성이 비슷하다는 뜻이다. 비싼 동네의 이상치는 고급 부동산을 나타낼 수 있는데, 이들이 평균을 부풀리면서도 중앙값은 비교적 안정적으로 남겨 둔다.

---

## 예제 3: 대출 신용등급별 소득

### 설정

대부자는 대출 등급(A = 최상, G = 최하)에 걸친 소득 분포를 살펴 신용 위험을 평가한다. 등급이 낮을수록 소득이 낮고 더 넓게 퍼지는 경향이 있다.

### 코드

```python
np.random.seed(456)
grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
n_per_grade = 100

data_list = []
for grade in grades:
    grade_idx = ord(grade) - ord('A')
    base_income = 80_000 - grade_idx * 8_000
    income_std = 15_000 + grade_idx * 5_000
    incomes = np.random.normal(base_income, income_std, n_per_grade)
    incomes = np.clip(incomes, 10_000, 200_000)
    for income in incomes:
        data_list.append({'grade': grade, 'income': income})

loans = pd.DataFrame(data_list)

fig, ax = plt.subplots(figsize=(10, 5))
sns.violinplot(data=loans, x='grade', y='income', ax=ax, color='lightgreen')
ax.set_xlabel('Loan Grade (A=best, G=worst)')
ax.set_ylabel('Annual Income (\$)')
ax.set_title('Income Distribution by Credit Grade')
plt.tight_layout()
plt.show()
```

### 해석

- **A등급** 차입자는 소득이 높고 더 집중되어 있어 부도 위험이 낮다.
- **G등급** 차입자는 소득이 낮고 더 넓게 퍼져 있어 부도 위험이 높다.
- 바이올린 모양이 A에서 G로 갈수록 점점 넓어져 소득의 불확실성이 커짐을 보여준다.

---

## 상자그림 대 바이올린 그림

| 기능 | 상자그림 | 바이올린 그림 |
|---|---|---|
| 사분위수 표시 | 예 | 예(`inner='quartile'` 사용 시) |
| 이상치 표시 | 예(개별 점) | 아니오(매끄럽게 지워짐) |
| 전체 모양 표시 | 아니오 | 예 |
| 다봉성 드러냄 | 아니오 | 예 |
| 압축성 | 높음 | 보통 |

!!! tip "권장 실무"
    빠른 요약에는 상자그림을, 분포의 모양이 중요할 때는 바이올린 그림을 쓴다. 판단이 서지 않으면 둘을 나란히 보여준다.

---

## 연습문제

**연습문제 1.**
중앙값 선이 $Q_3$보다 $Q_1$에 가깝고 위쪽 수염이 아래쪽 수염보다 긴 상자그림이 있다. 이 분포의 모양이 어떠할지 서술하라.

??? success "풀이"
    이 분포는 **오른쪽으로 치우쳐** 있다. 중앙값이 $Q_1$에 가깝다는 것은 상자의 위쪽 절반이 더 넓다는 뜻이며, 자료가 오른쪽으로 더 뻗어 있음을 나타낸다. 위쪽 수염이 더 긴 것은 극단값이 높은 쪽에 더 흔함을 확인해 준다.

---

**연습문제 2.**
중앙값과 IQR이 동일한 두 집단이 있는데, A 집단의 바이올린 그림은 봉우리가 하나이고 B 집단은 둘이다. 두 분포는 어떻게 다르며, 어떤 요약통계량이 이 차이를 포착하지 못하는가?

??? success "풀이"
    A 집단은 **단봉**이고 B 집단은 **이봉**이다. 평균, 중앙값, 분산, IQR은 두 집단에서 모두 비슷하거나 동일할 것이다. 중심과 퍼짐에 근거한 표준 요약통계량은 봉우리 수를 포착하지 못한다. 바이올린 그림, 히스토그램, 밀도추정에서 보이는 전체 모양만이 그 차이를 드러낸다.

---

**연습문제 3.**
위의 항공사 지연 자료에서 아메리칸의 지연 중앙값이 6.0%, IQR이 4.5%이고 사우스웨스트의 지연 중앙값이 2.4%, IQR이 2.0%라고 하자. 어떤 관리자가 "아메리칸의 지연이 정확히 2.5배 나쁘다"고 주장한다. 이 주장을 비판하라.

??? success "풀이"
    중앙값의 비는 $6.0 / 2.4 = 2.5$이므로 중앙값에 대해서는 그 주장이 성립한다. 그러나 "2.5배 나쁘다"는 지나친 단순화다. IQR의 비는 $4.5 / 2.0 = 2.25$로 퍼짐은 다른 배율로 차이가 난다. 게다가 모양도 다를 수 있다(아메리칸의 Gamma(2, 3)이 사우스웨스트의 Gamma(1.2, 2)보다 더 대칭적이다). 하나의 배율로는 중심, 퍼짐, 모양의 차이를 동시에 담아낼 수 없다.

---

**연습문제 4.**
상자그림은 보여주는데 바이올린 그림은 보여주지 못하는 이상치가 있을 수 있는 이유를 설명하라. 어떤 상황에서 바이올린 그림이 분포의 꼬리에 대해 오도할 수 있는가?

??? success "풀이"
    상자그림은 $1.5 \times \text{IQR}$ 너머의 개별 관측값을 이산적인 점으로 표시한다. 바이올린 그림은 자료를 매끄럽게 만드는 커널밀도추정을 쓴다. 꼬리에 관측값이 적으면 KDE 추정값이 아주 낮아 바이올린이 가는 선으로 좁아지므로 극단값이 보이지 않게 된다. 다음과 같을 때 오도한다.

    - 표본 크기가 작고 개별 이상치가 중요할 때.
    - 극단 관측값이 중요한 정보를 담는, 꼬리가 두꺼운 분포(예: 코시분포)일 때.
    - KDE의 대역폭이 너무 커서 꼬리가 지나치게 매끄러워질 때.

---

**연습문제 5.**
어떤 자료에서든 관측값의 적어도 50%가 상자그림의 상자 안(즉 $Q_1$과 $Q_3$ 사이)에 놓임을 증명하라.

??? success "풀이"
    정의에 의해 $Q_1$은 25번째 백분위수이고 $Q_3$은 75번째 백분위수다. 그 사이에 있는 자료의 비율은

    $$
    P(Q_1 \le X \le Q_3) = 0.75 - 0.25 = 0.50
    $$

    이므로 관측값의 적어도 50%가 상자 안에 들어간다. 이는 사분위수의 정의에서 곧바로 따라오므로 모양과 무관하게 어떤 분포에서도 성립한다. $\square$

---

**연습문제 6.**
**벌떼 그림**(또는 흔들림을 준 스트립 그림)은 겹쳐 그려지는 것을 피하기 위해 수직으로 흔들림을 주면서 모든 관측값을 가로축을 따라 보여준다. 벌떼 그림이 상자그림이나 바이올린 그림보다 나은 때는 언제인가?

??? success "풀이"
    벌떼 그림은 *모든 개별 관측값*을 보여주며, 다음과 같을 때 가치가 있다.

    - **표본 크기가 작을 때**($n < 50$): 관측값이 적으면 요약통계량이 잡음에 흔들리므로 상자그림과 바이올린이 오도할 수 있다. 각 점을 보여주면 보는 사람이 자료를 직접 판단할 수 있다.
    - **관측값의 정확한 개수가 중요할 때**: 임상시험이나 실험 연구에서는 집단별 개수가 이야기의 일부다. 벌떼 그림은 이를 한눈에 보여준다.
    - **이상치를 시각적으로 주변화해서는 안 될 때**: 상자그림의 "이상치"는 "이건 무시하라"는 인상을 주는 고립된 점이다. 벌떼 그림은 모든 관측값을 동등하게 다룬다.
    - **집단 안에 다봉성이나 뭉침이 있을 때**: 벌떼 그림에서 빈틈이나 덩어리로 보이며, 매끄러워진 바이올린에서는 가려지기도 한다.

    다음과 같을 때는 벌떼 그림이 적절하지 *않다*.

    - 표본 크기가 아주 클 때($n > 1000$): 흔들림으로도 겹침을 막을 수 없어 그림이 뭉개진다.
    - 여러 집단을 비교할 때: 벌떼 그림은 상자그림보다 가로 공간을 더 차지한다.
    - 이상치 식별이 목표일 때: 상자그림이 이상치를 시각적으로 더 두드러지게 만든다.

    흔히 쓰는 복합 그림은 벌떼 그림 위에 상자그림을 겹치는 것으로, 개별 자료와 요약을 함께 준다. Seaborn의 `sns.stripplot(..., dodge=True) + sns.boxplot(...)`이 이를 만들어낸다.

---

**연습문제 7.**
**바이올린 그림의 분할 모드**(집단당 반쪽 바이올린)는 짝지은 집단 비교를 시각적으로 선명하게 만든다. 분할 바이올린이 보는 사람을 **오도하는** 경우를 설명하고 그런 경우의 대안을 제안하라.

??? success "풀이"
    분할 바이올린은 다음과 같을 때 오도한다.

    - **집단 간 표본 크기가 크게 다를 때**: 각 반쪽이 같은 시각적 폭을 차지하도록 정규화되어 불균형이 감춰진다. 관측값 10개인 집단이 1000개인 집단만큼 시각적 무게를 갖는 것처럼 보인다.
    - **분포의 모양이 매우 다를 때**: 눈이 반대쪽 반쪽을 대칭으로 읽어, 밀도가 서로 무관한데도 거울상 분포처럼 보일 수 있다.
    - **기준 축이 임의적일 때**: 어느 집단을 어느 쪽에 두느냐가 "높음"과 "낮음"의 해석에 영향을 준다. 관례적인 순서가 없다면(예: "처리 A" 대 "처리 B") 분할 바이올린이 근거 없는 해석을 유도할 수 있다.

    **대안:** 각 아래에 표본 크기를 표기한($n = 10$ 대 $n = 1000$) 전체 바이올린을 나란히 놓는다. 분할 바이올린의 시각적 우아함을 포기하는 대신 자료에 대해 정직해진다.

    짝지은 자료(같은 대상을 두 번 측정)의 경우 분할이든 나란히든 바이올린은 짝지음을 포착하지 못한다. **짝 그림**을 쓰라. 각 대상의 두 측정값을 선으로 잇는다. 이렇게 하면 개별 변화와 대상 내 변동의 크기가 보이는데, 짝지은 설계에서 실제로 관심 있는 양이 바로 그것이다.
