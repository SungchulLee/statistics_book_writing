# 경험적 누적분포함수와 분위수

## 개요

**경험적 누적분포함수(ECDF)** 와 **분위수**는 히스토그램의 구간 너비 민감성을 피하면서 분포를 서로 보완적으로 바라보는 두 관점을 제공한다. ECDF는 각 자료값을 그 값 이하인 관측값의 비율로 보내어 계단함수를 만들며, 표본 크기가 커지면 참된 누적분포함수로 수렴한다. 분위수는 이 관계를 뒤집어 "자료의 주어진 비율이 어떤 값 아래에 떨어지는가?"에 답한다.

## 경험적 누적분포함수

표본 $x_1, x_2, \ldots, x_n$에 대해 ECDF는

$$
\hat{F}(t) = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}(x_i \le t)
$$

로 정의되며, 여기서 $\mathbf{1}(\cdot)$은 지시함수다. 핵심 성질은 다음과 같다.

- $\hat{F}$는 0에서 1까지 값을 갖는 비감소 계단함수다.
- 각 계단의 높이는 $1/n$이다(값이 겹치면 그 배수).
- 글리벤코–칸텔리 정리에 의해 $\hat{F}$는 참된 누적분포함수 $F$로 거의 확실하게 균등수렴한다.

### ECDF 대 이론적 누적분포함수

ECDF를 모수적 누적분포함수와 비교하는 것은 분포 가정을 평가하는 강력한 진단이다.

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

np.random.seed(1)
x = 4 + np.random.normal(0, 1.5, 100)

loc = x.mean()
scale = x.std()

x.sort()
cdf = stats.norm(loc=loc, scale=scale).cdf(x)

fig, ax = plt.subplots(figsize=(12, 3))
ax.ecdf(x, ls="-", c="r", label="Empirical CDF")
ax.plot(x, cdf, "-b", label="Theoretical CDF")
ax.legend()
plt.show()
```

경험적 곡선과 이론적 곡선이 가깝게 겹치면 모수 모형이 잘 맞는 것이다. 체계적으로 벗어나면 왜도, 두꺼운 꼬리, 또는 다봉성을 나타낸다.

### 누적분포함수와 확률밀도함수 나란히 보기

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

loc = 1
scale = 2
normal = stats.norm(loc=loc, scale=scale)

x = np.linspace(loc - 3 * scale, loc + 3 * scale, 1_000)
pdf = normal.pdf(x)
cdf = normal.cdf(x)

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(x, pdf, "-b", label="PDF")
ax.plot(x, cdf, "-r", label="CDF")
ax.legend()
plt.show()
```

확률밀도함수는 밀도가 어디에 몰려 있는지 보여주고, 누적분포함수는 누적 확률을 보여준다. 둘을 함께 보면 분포의 완전한 그림이 나온다.

## 분위수, 백분위수, 사분위수

### 백분위수

$p$번째 **백분위수** $P_p$는 자료의 $p\%$가 그 아래에 떨어지는 값이다. 누적상대도수 그래프에서 y축의 높이 $p/100$을 읽어 수평으로 곡선까지 이동하면 x축에서 백분위수를 얻는다.

### 사분위수

세 개의 사분위수가 자료를 네 등분한다.

$$
\begin{array}{llll}
\text{First Quartile} & Q_1 &=& P_{25} \\
\text{Second Quartile} & Q_2 &=& P_{50} \\
\text{Third Quartile} & Q_3 &=& P_{75} \\
\end{array}
$$

### 십분위수

$$
\begin{array}{llll}
D_1 = P_{10}, \quad D_2 = P_{20}, \quad \ldots, \quad D_9 = P_{90}
\end{array}
$$

### 중앙값과의 관계

$$
\text{Median} = Q_2 = D_5 = P_{50}
$$

## 파이썬에서 분위수 계산하기

흔히 쓰는 세 가지 방법이 모두 같은 결과를 낸다.

```python
import pandas as pd
import numpy as np
from scipy import stats

data = {'x': [4, 4, 6, 7, 10, 11, 12, 14, 15]}
df = pd.DataFrame(data)

# pandas: q in [0, 1]
print(f"{df.x.quantile(0.75) = }")

# numpy: q in [0, 100]
print(f"{np.percentile(df.x.values, 75) = }")

# scipy: q in [0, 100]
print(f"{stats.scoreatpercentile(df.x.values, 75) = }")
```

## 예: 스타벅스 음료의 당 함량

영양학자들이 스타벅스 음료 32종의 당 함량(그램)을 측정했다. 누적상대도수 그래프를 이용하면 다음과 같다.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 55, 5)
y = [0, 0.1, 0.1, 0.2, 0.3, 0.5, 0.6, 0.6, 0.8, 0.9, 1.0]

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(x, y, '-o')
ax.set_xlabel("Sugar Content (g)")
ax.set_ylabel("Cumulative Relative Frequency")
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.grid()
plt.show()
```

**질문과 답:**

1. 당이 15그램인 커피는 대략 **20번째 백분위수**에 해당한다.
2. **중앙값**(50번째 백분위수)은 대략 **25그램**이다.
3. $Q_1 \approx 17.5$ g, $Q_3 \approx 38.5$ g이므로 $\text{IQR} = Q_3 - Q_1 \approx 21$ g이다.

## 다섯 수치 요약

다섯 수치 요약은 분포의 핵심 분위수를 담는다.

$$
\text{Min} \quad Q_1 \quad \text{Median} \quad Q_3 \quad \text{Max}
$$

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.array([1, 2, 0, 0, 0, 1, 3, 1, 2, 1, 2, 4, 5, -1, -2, 0, 8])

quantiles = {"Min": 0, "Q1": 0.25, "Median": 0.5, "Q3": 0.75, "Max": 1}

for label, q in quantiles.items():
    print(f"{label:6} : {np.quantile(data, q)}")

fig, ax = plt.subplots(figsize=(2, 3))
ax.boxplot(data)
ax.set_title("Boxplot of Data")
plt.show()
```

## Q-Q 그림: 분위수 대 분위수 비교

**Q-Q 그림**은 관측 자료의 분위수를 이론적 분포의 분위수와 비교한다. 자료가 기준 분포를 따르면 점들이 대각선 기준선을 따라 놓인다.

### 정규분포에 대한 Q-Q 그림

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_qq(data, dist="norm", sparams=(), figsize=(12, 3)):
    fig, ax = plt.subplots(figsize=figsize)
    stats.probplot(data, dist=dist, sparams=sparams, plot=ax)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title('Q-Q Plot')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Ordered Values')
    plt.show()

np.random.seed(0)
sample_data = np.random.normal(loc=0, scale=1, size=1000)
plot_qq(sample_data, dist="norm")
```

### 지수분포에 대한 Q-Q 그림

```python
np.random.seed(0)
sample_data = np.random.exponential(scale=1, size=1000)
plot_qq(sample_data, dist="expon")
```

### 카이제곱분포에 대한 Q-Q 그림

```python
np.random.seed(0)
sample_data = np.random.chisquare(df=10, size=1000)
plot_qq(sample_data, dist="chi2", sparams=(10,))
```

### 진단적 활용: 카이제곱 자료를 정규 Q-Q 그림에 그리기

카이제곱 자료를 정규분포 기준으로 그리면 체계적인 휘어짐이 오른쪽 치우침을 드러내어, 정규 모형이 부적절함을 확인해 준다.

```python
np.random.seed(0)
sample_data = np.random.chisquare(df=10, size=1000)
plot_qq(sample_data, dist="norm")  # Systematic departure from the line
```

## 요약

ECDF와 분위수는 구간을 나눌 필요 없이 경험적 분포를 정확하게 표현한다. ECDF는 분포를 비교하거나 적합도를 평가하는 데 이상적이고, 분위수와 다섯 수치 요약은 간결한 수치 요약을 제공한다. Q-Q 그림은 이 개념들을 확장하여 분포 가정을 확인하는 강력한 시각적 진단 도구가 된다.

## 연습문제

**연습문제 1.**
자료 $\{2, 5, 5, 7, 10\}$에 대해 (a) ECDF $\hat F(x)$를 구간별 함수로 쓰라. (b) $\hat F(5)$와 $\hat F(6)$을 계산하라. (c) 50번째 백분위수(중앙값)를 구하라.

??? success "연습문제 1 풀이"
    (a) $n = 5$이므로

    $$
    \hat F(x) = \begin{cases} 0 & x < 2 \\ 1/5 & 2 \le x < 5 \\ 3/5 & 5 \le x < 7 \\ 4/5 & 7 \le x < 10 \\ 1 & x \ge 10 \end{cases}
    $$

    각 고유한 값에서 $1/n$만큼 뛰어오르며, 5는 두 번 나오므로 5에서의 도약은 $2/5$다.

    (b) $\hat F(5) = 3/5 = 0.6$(5 이하인 값이 셋), $\hat F(6) = 3/5 = 0.6$(5와 7 사이에 값이 없다).

    (c) 50번째 백분위수는 $\hat F(x) \ge 0.5$인 가장 작은 $x$이므로 $x = 5$다.

---

**연습문제 2.**
**글리벤코–칸텔리 정리**를 진술하고, ECDF를 참된 누적분포함수의 추정량으로 쓰는 것에 대해 이 정리가 무엇을 말해주는지 해석하라.

??? success "연습문제 2 풀이"
    $X_1, X_2, \ldots$가 누적분포함수 $F$를 갖는 i.i.d.이고 $\hat F_n$이 경험적 누적분포함수라 하자. 글리벤코–칸텔리 정리는

    $$
    \sup_x |\hat F_n(x) - F(x)| \xrightarrow{\text{a.s.}} 0 \quad \text{as } n \to \infty
    $$

    임을 말한다. 이 수렴은 점별이 아니라 모든 $x$에 걸쳐 *균등*하다. 바로 이 덕분에 ECDF가 범용 분포 추정량 역할을 할 수 있다. $F$의 어떤 연속적인 통계량(중앙값, IQR, 왜도 등)이든 $\hat F_n$의 대응되는 대입 통계량으로 일치성 있게 추정할 수 있다.

    **드보레츠키–키퍼–울포위츠(DKW) 부등식**이 그 속도를 정량화한다: $P(\sup_x |\hat F_n - F| > \varepsilon) \le 2 e^{-2n\varepsilon^2}$. $n = 100$이면 최악의 경우 차이가 확률 $\ge 0.96$으로 $\le 0.1$이다. 비모수 추정량치고는 빠르다.

---

**연습문제 3.**
분포 요약으로서 **ECDF**와 **히스토그램**을 비교하라. 각각의 장점을 두 가지씩 들어라.

??? success "연습문제 3 풀이"
    **ECDF의 장점:** (1) 구간을 나누지 않아 임의의 구간 너비를 고를 필요가 없다. (2) 모든 관측값을 정확히 사용한다. (3) 모수적 속도 $O(1/\sqrt{n})$로 균등수렴한다. (4) 두 분포를 비교하기 쉽다(ECDF 두 개를 겹쳐 그리거나 K-S 거리를 계산).

    **히스토그램의 장점:** (1) 보통의 독자에게 더 직관적이다 — "자료가 주로 어디에 있는가?" (2) ECDF가 y축 $[0, 1]$ 범위에 걸쳐 평평하게 만들어 버리는 밀도(봉우리, 최빈값, 빈틈)를 강조한다. (3) 다봉성을 즉시 드러낸다. (4) 논문과 대시보드에서 표준이다.

    실무적으로는 적합도 검정과 분포 비교에는 **ECDF**를, 모양을 시각적으로 전달하는 데는 **히스토그램**(또는 커널밀도추정)을 쓴다.

---

**연습문제 4.**
**분위수의 해석.** 어떤 표준화 시험이 한 학생을 85번째 백분위수라고 보고한다. 이것이 정확히 무슨 뜻인지 진술하라. 이것과 "시험에서 85%를 받았다"의 차이를 논하라.

??? success "연습문제 4 풀이"
    85번째 백분위수라는 것은 **응시자의 85%가 이 학생의 점수 이하를 받았다**는 뜻이다(15%가 더 높은 점수를 받았다). 백분위수는 기준 모집단에 대한 *순위 기반* 측도다.

    "시험에서 85%를 받았다"는 것은 *절대적* 성취 측도로, 정답을 맞힌 문항의 비율이다. 이 둘은 서로 무관하다.

    - 아주 어려운 시험에서 85%를 받은 학생은 99번째 백분위수일 수 있다(대부분이 더 못했으므로).
    - 아주 쉬운 시험에서 85%를 받은 학생은 30번째 백분위수일 수 있다(대부분이 더 잘했으므로).

    백분위 순위는 특정 시험판의 난이도에 불변이므로 표준화 시험에서 흔히 쓰인다. 연도나 시험 형식을 넘나드는 비교는 재규준화를 거친 뒤 원점수가 아니라 백분위수를 사용한다.

---

**연습문제 5.**
**콜모고로프–스미르노프 통계량** $D_n = \sup_x |\hat F_n(x) - F_0(x)|$는 자료가 지정된 분포 $F_0$에서 왔는지를 검정한다. 이것이 왜 자연스러운 검정통계량이며, 그 귀무분포는 $F_0$에 어떻게 의존하는가?

??? success "연습문제 5 풀이"
    **자연스러운 선택인 이유:** 글리벤코–칸텔리는 귀무가설($F = F_0$) 아래에서 $D_n \to 0$을 보장한다. 대립가설($F \ne F_0$) 아래에서는 $D_n$이 $\sup_x |F(x) - F_0(x)| > 0$으로 수렴한다. 따라서 $D_n$은 어떤 연속인 대립가설에 대해서도 귀무가설과 대립가설을 분리한다.

    **귀무분포:** $F_0$이 완전히 지정되어 있으면 $D_n$의 분포는 $F_0$ 자체가 아니라 오직 $n$에만 의존한다. 이것이 K-S의 **분포무관(distribution-free)** 성질이다. 귀무가설 아래에서 $F_0(X)$가 $[0, 1]$ 위의 균등분포를 따르므로, 원래의 $F_0$이 무엇이든 $D_n$은 사실상 균등분포로부터의 차이를 재는 셈이다. 덕분에 어떤 연속인 기준 분포에도 같은 임계값을 쓸 수 있다.

    **단서:** $F_0$의 모수를 같은 자료에서 추정했다면(예: 표본에서 추정한 $\hat\mu, \hat\sigma$로 정규성을 검정하는 경우) 이 검정은 더 이상 분포무관이 아니다. 그런 상황에서는 **릴리포스 검정**이나 **샤피로–윌크** 검정이 적절하다.

---

**연습문제 6.**
**Q-Q 그림**은 자료의 분위수를 기준 분포의 분위수와 비교한다. 다음 패턴들을 해석하라. (a) 점들이 직선 위에 놓인다. (b) S자 곡선. (c) 아래로 볼록한 체계적 휘어짐. (d) 꼬리에서만 크게 벗어남.

??? success "연습문제 6 풀이"
    (a) **직선**(기준과 일치): 자료가 기준 분포로 잘 근사된다(보통 표준정규분포이며, 중심화와 척도 조정을 거쳤을 수 있다). 기울기는 자료의 표준편차, 절편은 평균이다.

    (b) **S자 모양**(천천히, 그다음 빠르게, 다시 천천히 상승): 자료의 **꼬리가 기준보다 얇다**. 극단값이 더 적다는 뜻이다. 가운데가 기준의 가운데보다 가파르다. 꼬리가 얇은 분포(예: 균등분포나 절단된 분포)를 나타낸다.

    (c) **아래로 볼록한 휘어짐**(오른쪽 위에서 더 가파름): 자료가 **오른쪽으로 치우쳐** 있다. 위쪽 꼬리가 기준보다 길다. 소득, 계수 자료, 로그정규 또는 지수 표본을 정규 기준에 대해 그릴 때 흔하다.

    (d) **꼬리에서만 벗어남**: 자료의 대부분은 잘 모형화되지만 극단 관측값이 맞지 않는다. 꼬리가 두껍거나(기준보다 극단값이 많음, 예: $t$ 분포) 오염 과정에서 온 이상치일 수 있다. 여러 표본을 살펴보거나 본체에 대해서만 강건 분석을 수행하여 구별한다.

    Q-Q 그림은 적합도 $p$-값 하나보다 더 유익하다. 모형이 실패했는지 *여부*만이 아니라 *어디서* 실패하는지를 보여주기 때문이다.
