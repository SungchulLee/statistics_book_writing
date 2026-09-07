# 히스토그램과 밀도 그림

## 개요

**히스토그램**은 탐색적 자료분석에서 가장 기본적인 도구 중 하나다. 연속변수의 범위를 같은 너비의 구간(bin)으로 나누고, 각 구간에 들어가는 관측값의 개수나 밀도를 직사각형 막대로 표시한다. 전체 넓이가 1이 되도록 정규화하면 히스토그램은 **밀도 그림** — 밑바탕의 확률밀도함수를 추정하는 매끄러운 곡선 — 을 근사한다.

$$
\text{Histogram height (density)} = \frac{\text{count in bin}}{\text{total count} \times \text{bin width}}
$$

히스토그램은 중심, 퍼짐, 왜도, 봉우리 수, 빈틈, 이상치 등 분포의 특징을 한눈에 드러낸다.

## 밀도를 겹쳐 그린 기본 히스토그램

다음 예제는 정규분포에서 표본 10,000개를 뽑아 `density=True`로 히스토그램을 그리고 적합된 정규 확률밀도함수를 겹쳐 그린다.

```python
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

samples = 10_000
x = stats.norm(loc=5, scale=10).rvs(samples)

fig, ax = plt.subplots(figsize=(12, 3))
_, bins, _ = ax.hist(x, bins=100, density=True)

x_mean = x.mean()
x_std = x.std(ddof=1)
pdf = stats.norm(loc=x_mean, scale=x_std).pdf(bins)

ax.plot(bins, pdf, 'r-', linewidth=2)
plt.show()
```

**핵심 사항:**

- `density=True`는 전체 넓이가 1이 되도록 히스토그램을 정규화하여 y축이 원자료의 개수가 아니라 확률밀도를 나타내게 한다.
- 빨간 곡선은 표본평균과 표본표준편차로 적합한 정규분포의 확률밀도함수다.
- 표본 10,000개와 구간 100개로 그리면 히스토그램이 이론적 밀도를 아주 가깝게 따라간다.

## 실제 자료의 히스토그램: 소득 분포

소득 자료는 히스토그램의 모양이 중요한 해석적 의미를 담는 오른쪽으로 치우친 분포의 고전적인 예다.

```python
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

def plot_loan_income_distribution():
    url = 'https://raw.githubusercontent.com/gedeck/practical-statistics-for-data-scientists/master/data/loans_income.csv'
    df = pd.read_csv(url)

    mean_income = df['x'].mean()
    std_dev_income = df['x'].std()

    fig, ax = plt.subplots(figsize=(15, 4))
    _, bins, _ = ax.hist(df['x'], bins=30, density=True,
                         color='skyblue', label='Income histogram')

    norm_pdf = stats.norm(loc=mean_income, scale=std_dev_income).pdf(bins)
    ax.plot(bins, norm_pdf, "--r", label='Normal distribution')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Loan Income Distribution with Normal Fit')
    ax.set_xlabel('Income')
    ax.set_ylabel('Density')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    plot_loan_income_distribution()
```

히스토그램과 정규곡선이 어긋나는 모습이 오른쪽 치우침을 드러낸다. 고소득자의 긴 꼬리가 적합된 정규분포를 오른쪽으로 끌어당긴다.

## 여러 패널의 히스토그램: 주택 자료

자료에 수치형 특성이 많을 때는 히스토그램을 격자로 배열하면 모든 변수를 한꺼번에 빠르게 훑어볼 수 있다.

```python
import matplotlib.pyplot as plt
import os
import pandas as pd
import tarfile
import urllib.request

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path)

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

fetch_housing_data()
df = load_housing_data()

fig, axes = plt.subplots(3, 3, figsize=(12, 9))
df.hist(bins=50, ax=axes)

for ax in axes.reshape((-1,)):
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.show()
```

## 범주형에 가까운 자료의 히스토그램: 타이타닉

범주형 변수와 수치형 변수가 섞인 자료에서도 히스토그램은 각 열의 분포를 시각화하는 데 도움이 된다.

```python
import matplotlib.pyplot as plt
import pandas as pd

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url, index_col='PassengerId')

fig, axes = plt.subplots(1, 5, figsize=(12, 3))
titles = ("Sex", "Survived", "Age", "Pclass", "Age")

for ax, title in zip(axes, titles):
    ax.hist(df[title], density=True, edgecolor='black', alpha=0.7)
    ax.set_title(title)

plt.tight_layout()
plt.show()
```

## 사용자화한 히스토그램: 도수분포표에서 밀도 히스토그램으로

자료가 구간 너비가 서로 다른 도수분포표로 주어질 때는, 각 막대의 높이가 아니라 **넓이**가 백분율을 나타내도록 막대 높이를 조정해야 한다.

$$
\text{height}_i = \frac{\text{percent}_i}{\text{width}_i}
$$

| 소득 수준 (\$) | 백분율 |
|---|---|
| 0 – 1,000 | 1 |
| 1,000 – 2,000 | 2 |
| 2,000 – 3,000 | 3 |
| 3,000 – 4,000 | 4 |
| 4,000 – 5,000 | 5 |
| 5,000 – 6,000 | 5 |
| 6,000 – 7,000 | 5 |
| 7,000 – 10,000 | 15 |
| 10,000 – 15,000 | 26 |
| 15,000 – 25,000 | 26 |
| 25,000 – 50,000 | 8 |

```python
import matplotlib.pyplot as plt

def compute_bins_widths_heights():
    bins = [0, 1_000, 2_000, 3_000, 4_000, 5_000,
            6_000, 7_000, 10_000, 15_000, 25_000, 50_000]
    widths = [right - left for left, right in zip(bins[:-1], bins[1:])]
    percents = [1, 2, 3, 4, 5, 5, 5, 15, 26, 26, 8]
    heights = [p / w for w, p in zip(widths, percents)]
    return bins, widths, heights

def draw_line(start, end, ax):
    ax.plot([start[0], end[0]], [start[1], end[1]], '-k')

def draw_box(x_left, x_right, height, ax):
    draw_line([x_left, 0], [x_right, 0], ax)
    draw_line([x_right, 0], [x_right, height], ax)
    draw_line([x_right, height], [x_left, height], ax)
    draw_line([x_left, height], [x_left, 0], ax)

def main():
    bins, widths, heights = compute_bins_widths_heights()
    fig, ax = plt.subplots(figsize=(12, 3))
    for x_left, x_right, height in zip(bins[:-1], bins[1:], heights):
        draw_box(x_left, x_right, height, ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position("zero")
    plt.show()

if __name__ == "__main__":
    main()
```

## 구간 개수 정하기

구간의 개수는 해석에 깊은 영향을 미친다. 구간이 너무 적으면 지나치게 매끄러워져 구조가 가려지고, 너무 많으면 잡음이 생긴다. 흔한 지침으로는 스터지스 규칙($k = 1 + \log_2 n$), 제곱근 규칙($k = \lceil\sqrt{n}\rceil$), 그리고 IQR로 구간 너비를 정하는 프리드먼–다이어코니스 규칙이 있다. Matplotlib의 `bins='auto'`는 자료에 적응하는 전략을 적용한다.

## 요약

히스토그램과 밀도 그림은 어떤 연속변수든 탐색의 최전선에 있다. 대칭인지 치우쳤는지, 단봉인지 다봉인지, 꼬리가 두꺼운지 얇은지 등 분포의 모양을 드러내어 이후의 모든 모형화와 추론 결정을 이끈다.

## 연습문제

**연습문제 1.**
어떤 연구자가 시험 점수 20개를 모았다: $55, 62, 67, 70, 71, 73, 74, 75, 76, 78, 80, 81, 83, 85, 87, 88, 90, 92, 95, 98$.

**(a)** 스터지스 규칙에 따르면 히스토그램의 구간은 몇 개여야 하는가?
**(b)** 전체 범위에 같은 너비의 구간 4개를 적용하라. 구간 경계와 도수를 제시하라.
**(c)** 구간이 너무 적으면 왜 구조가 가려지고 너무 많으면 왜 없는 구조가 만들어지는지 설명하라.

??? success "풀이"
    (a) 스터지스 규칙: $k = \lceil \log_2 n \rceil + 1$. $n = 20$이므로 $k = \lceil 4.32 \rceil + 1 = 6$.

    (b) 범위 $= 43$, 구간 너비 $= 43/4 = 10.75$:

    - $[55, 65.75)$: 2개 (55, 62)
    - $[65.75, 76.5)$: 7개 (67, 70, 71, 73, 74, 75, 76)
    - $[76.5, 87.25)$: 6개 (78, 80, 81, 83, 85, 87)
    - $[87.25, 98]$: 5개 (88, 90, 92, 95, 98)

    (c) 구간이 너무 적으면 서로 다른 특징이 하나로 합쳐진다. 두 최빈값이 같은 구간에 들어가면 이봉 분포가 단봉으로 보일 수 있다. 구간이 너무 많으면 참된 밀도를 반영하지 않는, 표집 잡음에서 비롯된 봉우리와 골이 생긴다. "적절한" 개수는 (지나친 매끄러움에서 오는) 편향과 (잡음 섞인 구간에서 오는) 분산 사이의 균형을 잡는 것이며, 이는 비모수 밀도추정의 밑바탕에 있는 것과 같은 절충이다.

---

**연습문제 2.**
**스터지스 규칙**($k = 1 + \log_2 n$), **제곱근 규칙**($k = \lceil \sqrt n \rceil$), **프리드먼–다이어코니스 규칙**(구간 너비 $h = 2 \cdot \mathrm{IQR}/n^{1/3}$)을 비교하라. 각각은 언제 실패하는가?

??? success "풀이"
    **스터지스:** 자료가 대략 정규라고 가정하며 $n$이 클 때 구간 수를 과소하게 잡는다. $n = 1024$에서 구간이 10개뿐이라 큰 표본에서는 지나치게 매끄러워진다. 치우쳤거나 꼬리가 두꺼운 자료에서 실패한다.

    **제곱근:** 간단하고 중간 정도의 $n$에는 합리적이지만 자료의 퍼짐을 무시한다. 희소한 자료는 구간을 과하게 나누고 조밀한 자료는 덜 나누는 경향이 있다.

    **프리드먼–다이어코니스:** (이상치에 강건한) IQR을 쓰고 $n^{-1/3}$로 축소되어 히스토그램 MISE에 대해 점근적으로 최적인 속도를 갖는다. 대체로 가장 좋은 기본값이다. IQR이 0이거나 아주 작을 때(예: 이산적인 값이 대부분인 자료) 실패하며, 그런 경우에는 표준편차를 쓰는 스콧 규칙으로 돌아간다.

    현대적 실무: 프리드먼–다이어코니스와 스터지스를 결합해 둘 중 큰 쪽을 고르는 Matplotlib의 `bins='auto'`를 쓴다.

---

**연습문제 3.**
`density=True`일 때 히스토그램 아래 전체 넓이가 1임을 보여라. 히스토그램을 이론적 확률밀도함수와 비교하려면 왜 이 정규화가 필요한가?

??? success "풀이"
    구간의 너비를 $w_1, \ldots, w_k$, 도수를 $c_1, \ldots, c_k$($\sum c_i = n$)라 하자. 밀도로 정규화하면 구간 $i$의 높이는 $h_i = c_i / (n w_i)$이다. 전체 넓이는

    $$
    \sum_i w_i \cdot h_i = \sum_i w_i \cdot \frac{c_i}{n w_i} = \frac{1}{n}\sum_i c_i = 1
    $$

    이다. 임의의 확률밀도함수 $f$는 $\int f(x)\,dx = 1$을 만족한다. 정규화하지 않으면 히스토그램 높이가 도수 단위(합이 1이 아니라 $n$)여서 $f$와 직접 겹쳐 그리면 $n \cdot w$배만큼 어긋난다. 밀도 정규화는 둘을 같은 척도($x$ 단위당 확률)에 놓아 직접적인 시각적 비교와 적합도 평가를 가능하게 한다.

---

**연습문제 4.**
$N(0, 1)$에서 뽑은 i.i.d. 표본 1000개의 히스토그램이 $[-4, 4]$ 위에 같은 너비의 구간 30개를 쓴다. (a) 0을 포함하는 구간의 기대 도수를 추정하라. (b) 그 도수의 표준편차를 추정하라.

??? success "풀이"
    구간 너비 $w = 8/30 \approx 0.267$이다. 0을 포함하는 구간은 $[-w/2, w/2] = [-0.133, 0.133]$이다.

    (a) 관측값 하나가 이 구간에 들어갈 확률: $P(-0.133 < Z < 0.133) \approx 2 \cdot 0.133 \cdot \phi(0) \approx 2 \cdot 0.133 \cdot 0.399 \approx 0.106$. 기대 도수 $\approx 1000 \times 0.106 = 106$.

    (b) 도수는 이항분포를 따른다: $\mathrm{Var} = np(1-p) = 1000 \cdot 0.106 \cdot 0.894 \approx 95$이므로 표준편차 $\approx 9.7$.

    이 중앙 구간에서 상대적 잡음(표준편차/평균)은 $\approx 9\%$로, 히스토그램이 밀도를 충실히 따라갈 만큼 작다. $p \approx 0.001$인 꼬리 구간에서는 기대 도수가 1뿐이고 표준편차도 $\approx 1$이라 상대적 잡음이 100%다. 히스토그램의 꼬리가 들쭉날쭉해 보이고 밀도추정이 꼬리에서 다른 처리를 필요로 하는 이유가 이것이다.

---

**연습문제 5.**
**커널밀도추정(KDE)** 은 각 관측값을 커널함수 $K_h(x - x_i)$로 바꾸어 히스토그램을 매끄럽게 만든다. KDE 공식을 쓰라. 시각화에서 KDE가 히스토그램보다 대체로 선호되는 이유는 무엇인가?

??? success "풀이"
    KDE는

    $$
    \hat f(x) = \frac{1}{n h} \sum_{i=1}^n K\!\left(\frac{x - x_i}{h}\right)
    $$

    이며, 여기서 $K$는 적분값이 1인 커널함수(보통 가우시안: $K(u) = \frac{1}{\sqrt{2\pi}}e^{-u^2/2}$)이고 $h > 0$은 대역폭이다.

    **히스토그램에 대한 장점:**

    - **매끄러움**: KDE는 연속 곡선을 만들어 읽기 쉽고 여러 그림 사이에서 비교하기 좋다.
    - **구간 경계 인공물 없음**: 히스토그램의 모양은 구간 경계가 이동함에 따라 불연속적으로 바뀌지만, KDE는 그런 이동에 불변이다.
    - 매끄러운 밀도에 대한 **더 나은 수렴 속도**: 가우시안 커널의 MISE가 최적으로 $O(n^{-4/5})$인 반면 히스토그램은 $O(n^{-2/3})$이다.
    - **적응적 대역폭 방법**(실버만 규칙, 플러그인 선택자)이 매끄러움 선택을 자동화한다.

    **단점:** KDE는 지나치게 매끄럽게 만들어(최빈값을 감추어) 버리거나 덜 매끄럽게 만들어(가짜 봉우리를 만들어) 버릴 수 있다. 또 있을 법하지 않은 영역에 0이 아닌 밀도를 줄 수도 있다(예: 소득 자료에서 음수 값). 후자는 경계 보정 KDE로 다룬다.

---

**연습문제 6.**
다음 각각의 히스토그램이 어떤 모양일지 그려보고, 각각이 어떤 분포적 특징을 드러내는지 밝혀라. (a) 성인의 키, (b) 연간 가구소득, (c) 선진국의 사망 연령, (d) 전화번호 각 자리 숫자의 합.

??? success "풀이"
    (a) **성인의 키**: 대체로 대칭인 종 모양이며 약간 이봉일 수도 있다(남성과 여성의 최빈값이 다르다). 성별을 조건부로 하면 대략적인 정규성을, 조건 없이는 혼합 구조를 드러낸다.

    (b) **연간 가구소득**: 위쪽 꼬리가 긴, 강하게 오른쪽으로 치우친 분포. 평균 $\gg$ 중앙값. 흔히 로그정규분포나 파레토분포로 적합한다. 두꺼운 위쪽 꼬리를 통해 경제적 불평등을 드러낸다.

    (c) **사망 연령**(선진국): 이봉이다. 0 근처에 작은 봉우리(영아 사망)가 있고 70–80대에 큰 봉우리가 있다. 서로 경쟁하는 사망 원인(생애 초기 대 노화 관련)을 드러낸다. 의료가 개선되면서 영아 봉우리는 줄고 노년 봉우리는 오른쪽으로 이동했다.

    (d) **전화번호 자릿수 합**: 대략 종 모양이다(중심극한정리의 작동). 자릿수 합은 거의 독립인 균등한 자릿수들의 합이므로 그 분포가 정규에 가까워진다. 일상의 자료에서 중심극한정리를 드러낸다.

    이들을 함께 보면 히스토그램의 모양이 요약통계량만으로는 놓치는 *질적* 정보를 담고 있음을 알 수 있다. 언제나 먼저 그리고, 요약은 그다음이다.
