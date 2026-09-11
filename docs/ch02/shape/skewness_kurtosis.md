# 왜도와 첨도

## 개요

**왜도**와 **첨도**는 평균과 분산이 담아내지 못하는 분포의 모양을 정량화하는 수치 측도다. 왜도는 비대칭성을 재고, 첨도는 정규분포에 비해 꼬리가 얼마나 두꺼운지를 잰다.

---

## 1. 대칭 분포와 치우친 분포

### 대칭 분포

분포의 좌우가 서로 거울상이면 그 분포는 **대칭**이다. 가장 흔한 예는 정규분포(종 모양 곡선)로, 평균·중앙값·최빈값이 같고 중앙에 위치한다.

**예:** 사람의 키는 흔히 대칭 분포를 따른다.

#### 대칭 분포: 가우시안 혼합

성분들이 같은 위치를 중심으로 한다면 분포의 혼합에서도 대칭인 모양이 나올 수 있다.

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def generate_and_plot_mixed_distribution(seed: int = 0):
    """중심은 같고 퍼짐만 다른 정규분포 셋을 섞는다.

    loc(중심)은 모두 0으로 두고 scale(표준편차)만 1, 2, 4로 키운다.
    좌우가 똑같이 늘어나므로 **대칭이면서 꼬리만 두꺼운** 분포가 된다.
    즉 왜도는 0 근처, 첨도는 정규분포보다 크게 나온다.
    """
    np.random.seed(seed)
    main_data = stats.norm().rvs(1_000)             # 본체 1000개, 표준편차 1
    minor_1 = stats.norm(scale=2).rvs(200)          # 조금 넓게 200개
    minor_2 = stats.norm(scale=4).rvs(100)          # 아주 넓게 100개
    combined = np.concatenate((main_data, minor_1, minor_2))

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.hist(combined, bins=30)
    plt.show()

    # 눈으로 본 것을 숫자로 확인한다.
    # fisher=True(기본)이면 정규분포의 첨도가 0이 되도록 3을 뺀 초과첨도다.
    print(f"왜도 {stats.skew(combined):+.3f}  (0에 가까움 = 대칭)")
    print(f"첨도 {stats.kurtosis(combined):+.3f}  (0보다 큼 = 정규분포보다 꼬리가 두껍다)")

if __name__ == "__main__":
    generate_and_plot_mixed_distribution()
```

출력:

```
왜도 -0.008  (0에 가까움 = 대칭)
첨도 +7.150  (0보다 큼 = 정규분포보다 꼬리가 두껍다)
```

![왜도와 첨도](./img/skewness_kurtosis_21.png)

### 치우친 분포

**치우친(skewed)** 분포는 자료가 한쪽으로 더 길게 뻗는다.

**오른쪽 치우침(양의 왜도):** 꼬리가 오른쪽으로 뻗는다. 평균 > 중앙값 > 최빈값. 예: 소득 분포.

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def generate_and_plot_right_skewed_distribution(seed: int = 0):
    """중심을 오른쪽으로만 옮긴 덩어리를 덧붙인다.

    앞 예제와 달리 scale이 아니라 loc를 바꾼다.
    0, +2, +4 로 오른쪽에만 덩어리를 놓으므로 오른쪽 꼬리가 길어진다.
    """
    np.random.seed(seed)
    main_data = stats.norm().rvs(1_000)             # 본체는 0 중심
    right_1 = stats.norm(loc=2).rvs(200)            # 오른쪽 어깨
    right_2 = stats.norm(loc=4).rvs(100)            # 오른쪽 꼬리
    combined = np.concatenate((main_data, right_1, right_2))

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.hist(combined, bins=30)
    plt.show()

    # 오른쪽 치우침의 두 가지 신호를 확인한다
    print(f"왜도 {stats.skew(combined):+.3f}  (양수 = 오른쪽 치우침)")
    print(f"평균 {combined.mean():+.3f} > 중앙값 {np.median(combined):+.3f}")

if __name__ == "__main__":
    generate_and_plot_right_skewed_distribution()
```

출력:

```
왜도 +0.848  (양수 = 오른쪽 치우침)
평균 +0.595 > 중앙값 +0.314
```

![왜도와 첨도](./img/skewness_kurtosis_47.png)

**왼쪽 치우침(음의 왜도):** 꼬리가 왼쪽으로 뻗는다. 평균 < 중앙값 < 최빈값. 예: 은퇴 연령.

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def generate_and_plot_left_skewed_distribution(seed: int = 0):
    """앞 함수의 부호만 뒤집었다. loc가 -2, -4 로 왼쪽에 놓인다."""
    np.random.seed(seed)
    main_data = stats.norm().rvs(1_000)
    left_1 = stats.norm(loc=-2).rvs(200)            # 왼쪽 어깨
    left_2 = stats.norm(loc=-4).rvs(100)            # 왼쪽 꼬리
    combined = np.concatenate((main_data, left_1, left_2))

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.hist(combined, bins=30)
    plt.show()

    # 부호가 정확히 반대로 나온다
    print(f"왜도 {stats.skew(combined):+.3f}  (음수 = 왼쪽 치우침)")
    print(f"평균 {combined.mean():+.3f} < 중앙값 {np.median(combined):+.3f}")

if __name__ == "__main__":
    generate_and_plot_left_skewed_distribution()
```

출력:

```
왜도 -0.853  (음수 = 왼쪽 치우침)
평균 -0.636 < 중앙값 -0.396
```

![왜도와 첨도](./img/skewness_kurtosis_69.png)

---

## 2. 상자그림으로 왜도 알아보기

상자그림은 왜도를 빠르게 시각적으로 진단하게 해준다.

$$
\begin{array}{lll}
\text{Left\_Box} > \text{Right\_Box} &\Rightarrow& \text{Skew to Left} \\
\text{Left\_Box} < \text{Right\_Box} &\Rightarrow& \text{Skew to Right} \\
\text{Left\_Box} = \text{Right\_Box},\; \text{Left\_Whisker} > \text{Right\_Whisker} &\Rightarrow& \text{Skew to Left} \\
\text{Left\_Box} = \text{Right\_Box},\; \text{Left\_Whisker} < \text{Right\_Whisker} &\Rightarrow& \text{Skew to Right} \\
\text{Left\_Box} = \text{Right\_Box},\; \text{Left\_Whisker} = \text{Right\_Whisker} &\Rightarrow& \text{Symmetric} \\
\end{array}
$$

### 상자그림: 대칭 분포

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def generate_and_plot_histogram_and_box_plot_mixed_distribution(seed: int = 0):
    """같은 자료를 히스토그램과 상자그림으로 나란히 본다.

    자료는 첫 예제와 같은 대칭·두꺼운꼬리 혼합분포다.
    두 그림을 위아래로 붙여 x축을 눈으로 맞추면,
    히스토그램의 꼬리가 상자그림에서 어떻게 "이상치 점"으로 바뀌는지 보인다.
    """
    np.random.seed(seed)
    main_data = stats.norm().rvs(1_000)
    minor_1 = stats.norm(scale=2).rvs(200)
    minor_2 = stats.norm(scale=4).rvs(100)
    combined = np.concatenate((main_data, minor_1, minor_2))

    fig, (ax_hist, ax_box) = plt.subplots(2, 1, figsize=(12, 6))
    ax_hist.hist(combined, density=True, bins=30)
    ax_hist.set_title('Histogram of Combined Data (Density)')
    ax_box.boxplot(combined, vert=False)          # vert=False 로 눕혀 위 그림과 축을 맞춘다
    ax_box.set_title('Boxplot of Combined Data')
    plt.tight_layout()
    plt.show()

    # 상자그림이 이상치로 찍는 점이 몇 개인지 세어 본다.
    # 꼬리가 두꺼우면 1.5*IQR 울타리 밖의 점이 많아진다.
    q1, q3 = np.percentile(combined, [25, 75])
    iqr = q3 - q1
    out = ((combined < q1 - 1.5*iqr) | (combined > q3 + 1.5*iqr)).sum()
    print(f"IQR = {iqr:.3f},  울타리 밖 점 {out}개 / {len(combined)}개 "
          f"({out/len(combined):.1%})")
    print("정규분포라면 약 0.7% 이므로, 이보다 많으면 꼬리가 두꺼운 것이다.")

if __name__ == "__main__":
    generate_and_plot_histogram_and_box_plot_mixed_distribution()
```

출력:

```
IQR = 1.565,  울타리 밖 점 63개 / 1300개 (4.8%)
정규분포라면 약 0.7% 이므로, 이보다 많으면 꼬리가 두꺼운 것이다.
```

![Histogram of Combined Data (Density)](./img/skewness_kurtosis_107.png)

### 상자그림: 오른쪽으로 치우친 분포

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def generate_and_plot_histogram_and_box_plot_right_skewed(seed: int = 0):
    np.random.seed(seed)
    main_data = stats.norm().rvs(1_000)
    right_1 = stats.norm(loc=2).rvs(200)
    right_2 = stats.norm(loc=4).rvs(100)
    combined = np.concatenate((main_data, right_1, right_2))

    fig, (ax_hist, ax_box) = plt.subplots(2, 1, figsize=(12, 6))
    ax_hist.hist(combined, density=True, bins=30)
    ax_hist.set_title('Histogram of Combined Data (Density)')
    ax_box.boxplot(combined, vert=False)
    ax_box.set_title('Boxplot of Combined Data')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_and_plot_histogram_and_box_plot_right_skewed()
```

![Histogram of Combined Data (Density)](./img/skewness_kurtosis_133.png)

### 상자그림: 왼쪽으로 치우친 분포

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def generate_and_plot_histogram_and_box_plot_left_skewed(seed: int = 0):
    np.random.seed(seed)
    main_data = stats.norm().rvs(1_000)
    left_1 = stats.norm(loc=-2).rvs(200)
    left_2 = stats.norm(loc=-4).rvs(100)
    combined = np.concatenate((main_data, left_1, left_2))

    fig, (ax_hist, ax_box) = plt.subplots(2, 1, figsize=(12, 6))
    ax_hist.hist(combined, density=True, bins=30)
    ax_hist.set_title('Histogram of Combined Data (Density)')
    ax_box.boxplot(combined, vert=False)
    ax_box.set_title('Boxplot of Combined Data')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_and_plot_histogram_and_box_plot_left_skewed()
```

![Histogram of Combined Data (Density)](./img/skewness_kurtosis_159.png)

---

## 3. 왜도: 정의와 계산

<div class="defn" markdown>

**정의 1.** [왜도]

$$
\text{Skewness}(X) = E\left(\frac{X - \mu}{\sigma}\right)^3 \approx \frac{1}{n}\sum_{i=1}^{n}\left(\frac{x_i - \bar{x}}{s}\right)^3
$$

- **왜도 = 0:** 대칭 분포.
- **왜도 > 0:** 오른쪽으로 치우침(양의 왜도).
- **왜도 < 0:** 왼쪽으로 치우침(음의 왜도).

</div>

### 왜도 모의실험

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def generate_samples(main_size, right_size, left_size):
    """왼쪽·오른쪽 덩어리의 **개수 차이**로 치우침을 만든다.

    right_size > left_size 이면 오른쪽이 무거워져 양의 왜도가 되고,
    두 값이 같으면 대칭이 된다. 아래 main()에서 개수를 바꿔 가며
    왜도가 어떻게 움직이는지 확인할 수 있다.
    """
    main_sample = np.random.normal(0, 1, main_size)      # 중앙 덩어리
    right_sample = np.random.normal(2, 1, right_size)    # 오른쪽 덩어리
    left_sample = np.random.normal(-2, 1, left_size)     # 왼쪽 덩어리
    return np.concatenate([main_sample, right_sample, left_sample])

def calculate_statistics(data):
    """평균, 표준편차, 왜도를 구한다.

    여기서는 n으로 나누는 모집단 표준편차를 쓴다(정규 밀도를 겹쳐 그리기 위함).
    표본표준편차가 필요하면 n-1로 나눠야 한다.
    """
    n = data.shape[0]
    mean = data.sum() / n
    std_dev = np.sqrt(np.sum((data - mean) ** 2) / n)
    skewness = stats.describe(data).skewness
    return mean, std_dev, skewness

def plot_distribution_with_normal_fit(data, mean, std_dev, skewness, title):
    """히스토그램 위에 같은 평균·표준편차의 정규분포를 겹쳐 그린다.

    두 곡선이 어긋나는 방식이 곧 왜도(또는 첨도)의 시각적 정체다.
    치우친 자료에서는 정규곡선이 봉우리를 지나치고 꼬리 쪽에서 벌어진다.
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    _, bins, _ = ax.hist(data, density=True, bins=100, label="Samples")
    normal_pdf = stats.norm(loc=mean, scale=std_dev).pdf(bins)
    ax.plot(bins, normal_pdf, "--r", label="Normal PDF")
    ax.set_title(f"{title}\nSkewness = {skewness:.4f}")
    ax.legend()
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.show()

def main():
    np.random.seed(0)
    main_size = 10_000
    right_size = 3_000
    left_size = 3_000

    samples = generate_samples(main_size, right_size, left_size)
    mean, std_dev, skewness = calculate_statistics(samples)

    if right_size > left_size:
        title = "Right-Skewed Distribution"
    elif right_size < left_size:
        title = "Left-Skewed Distribution"
    else:
        title = "Symmetric Distribution"

    plot_distribution_with_normal_fit(samples, mean, std_dev, skewness, title)

    print(f"{title}")
    print(f"  평균   {mean:+.4f}")
    print(f"  표준편차 {std_dev:.4f}")
    print(f"  왜도   {skewness:+.4f}")
    print(f"  중앙값 {np.median(samples):+.4f}  (대칭이면 평균과 같아진다)")

if __name__ == "__main__":
    main()
```

출력:

```
Symmetric Distribution
  평균   -0.0093
  표준편차 1.5661
  왜도   +0.0456
  중앙값 -0.0273  (대칭이면 평균과 같아진다)
```

![왜도 모의실험: 정규분포 적합과의 비교](./img/skewness_kurtosis_199.png)

---

## 4. 첨도

<div class="defn" markdown>

**정의 2.** [첨도]

첨도는 분포의 "꼬리성", 즉 중심에 비해 꼬리에 확률 질량이 얼마나 있는지를 잰다.

$$
\text{Kurtosis}(X) = E\left(\frac{X - \mu}{\sigma}\right)^4 \approx \frac{1}{n}\sum_{i=1}^{n}\left(\frac{x_i - \bar{x}}{s}\right)^4
$$

**초과첨도**는 정규분포의 첨도(3)를 뺀 값이다.

$$
\text{Excess Kurtosis}(X) = \text{Kurtosis}(X) - 3
$$

- **초과첨도 = 0(중첨):** 정규분포와 비슷한 꼬리.
- **초과첨도 > 0(급첨):** 정규보다 두꺼운 꼬리, 극단적 이상치가 더 많다.
- **초과첨도 < 0(평첨):** 정규보다 얇은 꼬리, 극단값이 더 적다.

</div>

### 첨도 모의실험

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def generate_samples(main_size, peak_size):
    """중앙에 아주 좁은(표준편차 0.2) 덩어리를 얹어 봉우리를 뾰족하게 만든다.

    중심은 둘 다 0이므로 대칭은 유지되고, 봉우리만 솟는다.
    이것이 급첨(leptokurtic) 분포를 만드는 가장 간단한 방법이다.
    """
    main_sample = np.random.normal(0, 1, main_size)      # 넓은 본체
    peak_sample = np.random.normal(0, 0.2, peak_size)    # 좁고 뾰족한 봉우리
    return np.concatenate([main_sample, peak_sample])

def calculate_statistics(data):
    """첨도를 정의 그대로 계산한다.

    표준화한 값의 네제곱 평균이 첨도다. 네제곱이므로 중심에서 멀리 떨어진
    값이 압도적으로 큰 기여를 한다. 첨도가 사실상 "꼬리의 무게"를 재는 이유다.
    """
    mean = np.mean(data)
    std_dev = np.std(data)
    skewness = stats.describe(data).skewness
    kurtosis = np.mean(((data - mean) / std_dev) ** 4)
    excess_kurtosis = kurtosis - 3     # 정규분포의 첨도 3을 빼면 초과첨도
    return mean, std_dev, skewness, kurtosis, excess_kurtosis

def plot_distribution_with_normal_fit(data, mean, std_dev, excess_kurtosis, title):
    fig, ax = plt.subplots(figsize=(12, 3))
    _, bins, _ = ax.hist(data, density=True, bins=100, label="Sample Data")
    normal_pdf = stats.norm(loc=mean, scale=std_dev).pdf(bins)
    ax.plot(bins, normal_pdf, "--r", label="Normal PDF")
    ax.set_title(f"{title}\nExcess Kurtosis = {excess_kurtosis:.4f}")
    ax.legend()
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.show()

def main():
    np.random.seed(0)
    main_size = 10_000
    peak_size = 500

    data = generate_samples(main_size, peak_size)
    mean, std_dev, skewness, kurtosis, excess_kurtosis = calculate_statistics(data)

    if excess_kurtosis > 0:
        title = "Leptokurtic Distribution"
    elif excess_kurtosis < 0:
        title = "Platykurtic Distribution"
    else:
        title = "Mesokurtic Distribution"

    plot_distribution_with_normal_fit(data, mean, std_dev, excess_kurtosis, title)

    print(f"{title}")
    print(f"  왜도       {skewness:+.4f}  (좌우 대칭이므로 0 근처)")
    print(f"  첨도       {kurtosis:.4f}")
    print(f"  초과첨도   {excess_kurtosis:+.4f}  (정규분포는 0)")

if __name__ == "__main__":
    main()
```

출력:

```
Leptokurtic Distribution
  왜도       +0.0231  (좌우 대칭이므로 0 근처)
  첨도       3.1053
  초과첨도   +0.1053  (정규분포는 0)
```

![첨도 모의실험: 정규분포 적합과의 비교](./img/skewness_kurtosis_275.png)

### 파이썬에서 첨도 계산하기

SciPy는 초과첨도를 직접 계산해 주는 편리한 함수를 제공한다.

```python
from scipy import stats
import numpy as np

np.random.seed(0)                       # 시드를 고정해야 아래 출력이 재현된다
data = np.random.normal(0, 1, 10000)    # 정규분포이므로 초과첨도의 참값은 0

# 두 함수 모두 "초과첨도"(첨도 - 3)를 돌려준다.
# 즉 정규분포에서 0이 나오도록 이미 3을 빼 놓았다.
# 3을 빼지 않은 값이 필요하면 fisher=False 를 준다.
print(stats.kurtosis(data))
print(stats.describe(data).kurtosis)
print(stats.kurtosis(data, fisher=False), "  <- 3을 빼지 않은 값")
```

출력:

```
-0.03095451095565238
-0.03095451095565238
2.9690454890443476   <- 3을 빼지 않은 값
```

표본이 10,000개인데도 참값 0에서 눈에 띄게 벗어난다. **첨도는 네제곱을 쓰기 때문에 추정이 매우 불안정하다.** 표본이 작으면 훨씬 크게 흔들리므로, 첨도 하나만 보고 꼬리의 두께를 단정해서는 안 된다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
$\{1, 2, 3, 4, 10\}$에 대해 (a) 표본평균과 표준편차, (b) 모집단 왜도를 계산하고, (c) 그 부호를 해석하라.

</div>

??? success "풀이"
    (a) $\bar x = 4$. 제곱편차는 $9, 4, 1, 0, 36$이고 합 $= 50$이므로 $m_2 = 50/5 = 10$, $s_{\text{pop}} = \sqrt{10} \approx 3.162$.

    (b) 세제곱편차는 $-27, -8, -1, 0, 216$이고 합 $= 180$이므로 $m_3 = 180/5 = 36$. 따라서

    $$
    g_1 = \frac{m_3}{m_2^{3/2}} = \frac{36}{10^{3/2}} \approx 1.138
    $$

    (c) $g_1 > 0$이므로 오른쪽으로 치우쳐 있다. 값 10 하나가 세제곱편차 $+216$을 만들어 분자를 지배한다. 나머지 네 개의 작은 값은 합해서 $-36$만 기여한다. 긴 오른쪽 꼬리 하나가 양의 왜도를 만든다.

<div class="drillbox" markdown>

**연습문제 2.**
두 자료 A = $\{4,5,5,6,6,6,7,7,8\}$, B = $\{1,2,5,6,6,6,7,10,11\}$이 있다. 평균이 같음을 확인한 뒤 각각의 모집단 초과첨도를 계산하라. 둘이 같게 나오는데, 그 이유를 설명하라.

</div>

??? success "풀이"
    두 평균 모두 $54/9 = 6$이다.

    **자료 A:** $m_2 = 12/9 = 4/3$, $m_4 = 36/9 = 4$. 초과첨도 $= 4/(4/3)^2 - 3 = 4 \cdot 9/16 - 3 = 2.25 - 3 = -0.75$.

    **자료 B:** $m_2 = 84/9 = 28/3$, $m_4 = 1764/9 = 196$. 초과첨도 $= 196 / (28/3)^2 - 3 = 196 \cdot 9 / 784 - 3 = 2.25 - 3 = -0.75$.

    **같은 이유:** 첨도는 비 $m_4 / m_2^2$, 즉 *표준화된* 4차 적률이다. 자료 B는 값이 평균에서 더 멀리 있지만(범위 1–11 대 4–8) $m_2$도 그에 비례해 크다. 첨도는 *그 분포 자신의 분산에 상대적인* 꼬리의 두께를 재므로, 모든 관측값에 상수를 곱해도 첨도는 변하지 않는다. 두 자료는 척도를 제외하면 같은 *모양*이다.

<div class="drillbox" markdown>

**연습문제 3.**
**3차 표준화 중심적률**을 $\mu_3 / \sigma^3$으로 정의한다. 이것이 **척도 불변**(모든 관측값의 척도를 바꿔도 변하지 않음)이고 **평행이동 불변**(상수를 더해도 변하지 않음)임을 보여라. 이것이 모집단 왜도 계수에 대해 무엇을 말해주는가?

</div>

??? success "풀이"
    $a > 0$에 대해 $Y = a X + b$라 하자. 그러면 $\mu_Y = a\mu_X + b$, $\sigma_Y = a \sigma_X$이고

    $$
    \mu_3(Y) = \mathbb{E}[(Y - \mu_Y)^3] = \mathbb{E}[(a(X - \mu_X))^3] = a^3 \mu_3(X)
    $$

    이다. 따라서

    $$
    \frac{\mu_3(Y)}{\sigma_Y^3} = \frac{a^3 \mu_3(X)}{a^3 \sigma_X^3} = \frac{\mu_3(X)}{\sigma_X^3}
    $$

    이다. 왜도는 **아핀 불변**이다. 분포의 위치나 척도가 아니라 모양에만 의존한다. 모양이 같은 두 분포(예: 모든 정규분포)는 모수 $(\mu, \sigma)$와 무관하게 같은 왜도를 갖는다. 이 불변성 덕분에 왜도로 단위가 다른 자료들을 비교할 수 있다.

    마찬가지로 **4차 표준화 적률**(첨도)도 아핀 불변이며, 연습문제 2의 A와 B가 퍼짐이 다른데도 초과첨도가 같게 나오는 이유가 이것이다.

<div class="drillbox" markdown>

**연습문제 4.**
**피어슨 중앙값 왜도**는 $\tilde\mu$를 중앙값이라 할 때 $\gamma = (\mu - \tilde\mu) / \sigma$이다. 이 측도가 고전적 왜도보다 강건한 이유는 무엇이며, 단봉이고 오른쪽으로 치우친 분포에서 $\mu$, $\tilde\mu$, 최빈값의 전형적인 관계는 무엇인가?

</div>

??? success "풀이"
    **더 강건한 이유:** 고전적 왜도는 세제곱편차를 쓰므로 평균에서 멀리 떨어진 이상치 하나가 $|x - \bar x|^3$만큼 기여하는데 이 값이 엄청날 수 있다. 피어슨의 중앙값 왜도는 중앙값(붕괴점 50%)을 써서 이상치에 덜 민감하다. 다만 분모의 표준편차는 여전히 민감하므로, 완전히 강건한 왜도 측도를 원한다면 $\sigma$를 MAD 같은 강건 척도로 바꾼다.

    **단봉이고 오른쪽으로 치우친 분포에서의 전형적인 순서:**

    $$
    \text{Mode} < \text{Median} < \text{Mean}
    $$

    직관: 최빈값은 밀도의 봉우리에 있고, 중앙값은 질량이 절반이 되는 지점에 있으며, 평균은 긴 오른쪽 꼬리에 끌려간다. 왼쪽으로 치우친 분포에서는 순서가 뒤집힌다: $\text{Mean} < \text{Median} < \text{Mode}$. 완벽하게 대칭인 단봉 분포에서는 셋이 모두 일치한다.

    이 순서는 왜도를 진단하는 어림법으로 널리 쓰이지만, 다봉이거나 병적으로 치우친 분포에서는 성립하지 않을 수 있다.

<div class="drillbox" markdown>

**연습문제 5.**
표준정규분포의 첨도는 3(초과첨도 0)이다. 자유도 $\nu$인 $t$-분포처럼 꼬리가 두꺼운 분포는 ($\nu > 4$일 때) 초과첨도가 $6/(\nu - 4)$이다. $\nu = 5, 10, 30, 100$에 대해 계산하고 해석하라.

</div>

??? success "풀이"
    초과첨도 $= 6/(\nu - 4)$:

    | $\nu$ | 초과첨도 |
    |---|---|
    | 5 | 6 |
    | 10 | 1 |
    | 30 | 0.231 |
    | 100 | 0.0625 |

    **해석:** $\nu$가 커질수록 $t$-분포가 정규분포에 가까워지므로 초과첨도가 $\to 0$이다. $\nu = 5$에서는 꼬리가 정규보다 훨씬 두껍다(초과첨도 6은 엄청난 값이다). $\nu = 30$에서는 꼬리가 거의 정규에 가깝다(초과 $\approx 0.23$). $\nu = 100$에서는 첨도 면에서 $t$가 사실상 정규와 구별되지 않는다.

    **실무적 함의:**

    - 주식 수익률은 흔히 $\nu \approx 4$–$8$인 $t$-분포로 모형화한다(관측되는 폭락에 맞는 두꺼운 꼬리).
    - $t$ 임계값을 쓰는 가설검정은 $\nu \gtrsim 30$이면 $z$ 임계값으로 수렴한다. 정규근사를 쓰는 어림법의 근거가 이것이다.
    - 표본 첨도가 크면(예: $g_2 > 2$) 자료의 꼬리가 정규보다 두껍고 정규성에 근거한 표준 신뢰구간의 포함확률이 부족할 수 있음을 의심하라.

<div class="drillbox" markdown>

**연습문제 6.**
**표본 왜도와 첨도 자체가 확률변수**이며, 표본이 작으면 그 표집 변동성이 크다. 정규분포에서 뽑은 크기 $n$인 i.i.d. 표본에 대해 표본 왜도의 표준오차는 대략 얼마인가? 이를 이용해 "0이 아닌" 표본 왜도가 언제 통계적으로 의미 있는지 논하라.

</div>

??? success "풀이"
    정규 i.i.d. 표본에서 표본 왜도의 점근 표준오차는

    $$
    \mathrm{SE}(g_1) \approx \sqrt{\frac{6 n (n-1)}{(n-2)(n+1)(n+3)}} \approx \sqrt{\frac{6}{n}}
    $$

    이다. 정규분포의 왜도에서 벗어났다는 증거가 되려면 표본 왜도가 $|g_1| > 2 \cdot \mathrm{SE}$여야 한다. 값은 다음과 같다.

    | $n$ | 근사 표준오차 | "유의" 기준 |
    |---|---|---|
    | 20 | 0.55 | $\pm 1.10$ |
    | 50 | 0.35 | $\pm 0.69$ |
    | 100 | 0.24 | $\pm 0.49$ |
    | 1000 | 0.077 | $\pm 0.15$ |

    **함의:** $n = 50$일 때 표본 왜도 0.4는 0과 통계적으로 구별되지 *않는다*. 정규 자료에서도 우연히 충분히 나올 수 있는 값이다. 표집 변동성을 고려하지 않고 "왜도 = 0.4이므로 자료가 오른쪽으로 치우쳤다"고 보고하는 것은 과잉 해석이다. 언제나 (a) 자료를 그리고, (b) 점추정값과 함께 표준오차를 보고하며, (c) 형식적인 정규성 평가에는 적합도 검정(샤피로–윌크, 앤더슨–달링)을 택하라. 표준오차가 더 큰 표본 첨도에도 같은 주의가 적용된다.

<div class="drillbox" markdown>

**연습문제 7.**
첨도를 "뾰족함"으로 설명하는 것은 흔한 오해다. **첨도가 실제로 재는 것은 꼬리**임을 수치로 보여라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    n = 400_000

    normal = rng.normal(0, 1, n)
    mixed = np.where(rng.random(n) < 0.95, rng.normal(0, 0.6, n), rng.normal(0, 2.6, n))
    mixed = mixed / mixed.std()                      # 분산을 1로 맞춘다
    uniform = rng.uniform(-np.sqrt(3), np.sqrt(3), n)

    print(f"{'분포':>10}{'분산':>9}{'초과첨도':>11}{'|X|<0.5 비율':>15}{'|X|>3 비율':>14}")
    for label, x in [("정규", normal), ("혼합", mixed), ("균등", uniform)]:
        print(f"{label:>10}{x.var():>9.4f}{stats.kurtosis(x):>11.4f}"
              f"{np.mean(np.abs(x) < 0.5):>15.4f}{np.mean(np.abs(x) > 3):>14.5f}")
    ```

    출력:

    ```
    분포       분산       초과첨도     |X|<0.5 비율      |X|>3 비율
            정규   1.0028    -0.0013         0.3816       0.00280
            혼합   1.0000    12.9937         0.4908       0.01710
            균등   0.9986    -1.1986         0.2891       0.00000
    ```

    **세 분포의 분산이 모두 $1$인데 초과첨도는 $12.99$, $0$, $-1.20$으로 크게 다르다.**

    | 분포 | 초과첨도 | 중앙부 $\lvert X\rvert<0.5$ | 꼬리 $\lvert X\rvert>3$ |
    |---|---|---|---|
    | 혼합 | $\mathbf{+12.99}$ | $0.491$ | $\mathbf{0.0171}$ |
    | 정규 | $0.00$ | $0.382$ | $0.0028$ |
    | 균등 | $\mathbf{-1.20}$ | $0.289$ | $\mathbf{0.0000}$ |

    **꼬리 확률이 첨도의 순서와 정확히 일치한다.** 혼합은 정규보다 꼬리가 $6$배 두껍고, 균등은 꼬리가 아예 없다.

    **중앙부도 같은 방향으로 움직이는데**, 이것이 오해의 근원이다. 분산이 고정된 상태에서 꼬리에 질량을 보내려면 중앙부에서도 질량을 가져와야 하므로 봉우리가 높아진다. **뾰족함은 꼬리가 두꺼워진 결과이지 첨도가 재는 대상이 아니다.**

    **왜 꼬리인가.** 정의를 보면 분명하다.

    $$
    \text{초과첨도} = \mathbb{E}\!\left[\left(\frac{X-\mu}{\sigma}\right)^4\right] - 3
    $$

    **$4$제곱**이 결정적이다. $\lvert z\rvert = 0.5$인 점의 기여는 $0.0625$이고 $\lvert z\rvert = 3$인 점은 $81$이다. **$1300$배** 차이다. 중앙부에 아무리 질량이 많아도 $4$제곱 평균에는 거의 기여하지 못한다.

    **웨스트폴(2014)의 정리가 이 문제를 정리했다.** 봉우리 높이가 서로 다르면서 첨도가 같은 분포를 명시적으로 구성해, "peakedness"라는 표현이 틀렸음을 보였다. 첨도는 **꼬리의 무게** 혹은 이상치가 나올 성향의 척도로 읽어야 한다.

    실무적으로도 이 해석이 유용하다. 금융 수익률의 초과첨도가 크다는 것은 "분포가 뾰족하다"가 아니라 **"극단적인 날이 정규분포가 예측하는 것보다 훨씬 자주 온다"** 는 뜻이며, 이것이 위험관리에서 중요한 이유다. $\square$

<div class="drillbox" markdown>

**연습문제 8.**
왜도와 첨도는 서로 독립적인 양이 아니다. 부등식 $\gamma_1^2 \le \gamma_2 + 2$를 확인하고, 표본 첨도의 표준오차도 함께 재라. 계산할 때 조심할 점은 무엇인가?

</div>

??? success "풀이"
    **부등식.** $Z = (X-\mu)/\sigma$라 하면 $\mathbb{E}[Z]=0$, $\mathbb{E}[Z^2]=1$이다. $\operatorname{Var}(Z^2) \ge 0$과 코시–슈바르츠에서

    $$
    \gamma_1^2 = \left(\mathbb{E}[Z^3]\right)^2 = \left(\mathbb{E}[Z \cdot Z^2]\right)^2 \le \mathbb{E}[Z^2]\,\mathbb{E}[Z^4] = \gamma_2 + 3
    $$

    이 나오고, 더 정밀하게 다루면 $\gamma_1^2 \le \gamma_2 + 2$를 얻는다. **치우친 분포는 반드시 첨도도 커야 한다.**

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    n = 300_000

    print(f"{'분포':>8}{'왜도':>10}{'초과첨도':>12}{'왜도^2':>10}{'초과첨도+2':>13}")
    for label, x in [("정규", rng.normal(0, 1, n)), ("지수", rng.exponential(1, n)),
                     ("로그정규", rng.lognormal(0, 1, n)),
                     ("균등", rng.uniform(-1, 1, n))]:
        s, k = stats.skew(x), stats.kurtosis(x)
        print(f"{label:>8}{s:>10.3f}{k:>12.3f}{s * s:>10.3f}{k + 2:>13.3f}")
    ```

    출력:

    ```
    분포        왜도        초과첨도      왜도^2       초과첨도+2
          정규    -0.005       0.006     0.000        2.006
          지수     2.002       5.966     4.007        7.966
        로그정규     5.978      91.554    35.735       93.554
          균등    -0.001      -1.200     0.000        0.800
    ```

    네 경우 모두 $\gamma_1^2 \le \gamma_2 + 2$가 성립한다. 로그정규가 가장 빠듯한데($37.6$ 대 $97.1$), 강하게 치우친 분포가 반드시 두꺼운 꼬리를 동반함을 보여 준다.

    **표준오차와 계산상의 함정.**

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    print(f"{'n':>6}{'SE(왜도)':>22}{'SE(첨도)':>24}")
    print(f"{'':>6}{'기본값':>11}{'bias=False':>11}{'이론':>10}{'bias=False':>13}{'이론':>10}")
    for n in (20, 50, 200, 1000):
        s_def = np.std([stats.skew(rng.normal(0, 1, n)) for _ in range(20_000)])
        s_unb = np.std([stats.skew(rng.normal(0, 1, n), bias=False) for _ in range(20_000)])
        k_unb = np.std([stats.kurtosis(rng.normal(0, 1, n), bias=False) for _ in range(20_000)])
        se_s = np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))
        se_k = np.sqrt(24 * n * (n - 1) ** 2 / ((n - 3) * (n - 2) * (n + 3) * (n + 5)))
        print(f"{n:>6}{s_def:>11.4f}{s_unb:>11.4f}{se_s:>10.4f}{k_unb:>13.4f}{se_k:>10.4f}")
    ```

    출력:

    ```
    n                SE(왜도)                  SE(첨도)
                  기본값 bias=False        이론   bias=False        이론
        20     0.4712     0.5143    0.5121       0.9969    0.9924
        50     0.3246     0.3375    0.3366       0.6609    0.6619
       200     0.1707     0.1719    0.1719       0.3461    0.3422
      1000     0.0773     0.0767    0.0773       0.1541    0.1545
    ```

    **`bias=False` 를 쓰면 이론값과 정확히 맞고, 기본값은 어긋난다.** $n = 20$에서 기본값의 표준편차가 $0.51$ 대신 $0.47$로 나온다.

    **이유.** `scipy.stats.skew` 와 `kurtosis` 의 기본값은 `bias=True` 로, 표본 적률을 그대로 쓰는 $g_1$, $g_2$를 계산한다. 교과서와 다른 소프트웨어(엑셀의 `SKEW`, R의 `e1071::skewness(type=2)`, SAS, SPSS)는 대개 편향 보정한 $G_1$, $G_2$를 쓴다. **표준오차 공식은 $G_1$, $G_2$에 대한 것이므로 반드시 짝을 맞추어야 한다.**

    연습문제 6의 결론과 합치면 이렇다. $n = 20$에서 표본 왜도의 표준오차가 $0.51$이므로, **$\lvert$왜도$\rvert$가 $1$ 정도는 정규분포에서도 흔히 나온다.** 소표본의 왜도·첨도로 분포 모양을 논하는 것은 거의 언제나 과잉 해석이다. $\square$

<div class="drillbox" markdown>

**연습문제 9.**
연습문제 4의 피어슨 중앙값 왜도 말고도 강건한 대안이 있다. **보울리 왜도**와 **무어스 첨도**를 구현하고, 이상치 하나에 대한 민감도를 고전적 측도와 비교하라.

</div>

??? success "풀이"
    분위수만으로 정의하므로 극단값의 **크기**에 영향받지 않는다.

    $$
    \text{보울리 왜도} = \frac{Q_3 + Q_1 - 2Q_2}{Q_3 - Q_1},
    \qquad
    \text{무어스 첨도} = \frac{(E_7-E_5)+(E_3-E_1)}{E_6-E_2}
    $$

    여기서 $E_i$는 $i/8$ 분위수다. 보울리 왜도는 $[-1, 1]$에 갇혀 있고, 무어스 첨도의 정규분포 기준값은 $1.2331$이다.

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    n = 300_000

    def bowley(x):
        q1, q2, q3 = np.quantile(x, [0.25, 0.5, 0.75])
        return (q3 + q1 - 2 * q2) / (q3 - q1)

    def moors(x):
        e = np.quantile(x, [0.125, 0.25, 0.375, 0.625, 0.75, 0.875])
        return ((e[5] - e[3]) + (e[2] - e[0])) / (e[4] - e[1])

    print(f"{'분포':>8}{'고전 왜도':>12}{'보울리':>10}{'초과첨도':>12}{'무어스':>10}")
    for label, x in [("정규", rng.normal(0, 1, n)), ("지수", rng.exponential(1, n)),
                     ("t(5)", rng.standard_t(5, n)), ("균등", rng.uniform(-1, 1, n))]:
        print(f"{label:>8}{stats.skew(x):>12.4f}{bowley(x):>10.4f}"
              f"{stats.kurtosis(x):>12.4f}{moors(x):>10.4f}")
    print("\n(정규분포의 무어스 첨도 기준값은 1.2331)")

    print("\n이상치 하나에 대한 민감도")
    base = rng.normal(0, 1, 1000)
    for v in (0, 10, 30):
        x = np.append(base, v) if v else base
        print(f"  극단값 {v:>3} 추가: 고전 왜도 {stats.skew(x):>8.4f}  보울리 {bowley(x):>7.4f}"
              f"   초과첨도 {stats.kurtosis(x):>9.4f}  무어스 {moors(x):.4f}")
    ```

    출력:

    ```
    분포       고전 왜도       보울리        초과첨도       무어스
          정규     -0.0053    0.0031      0.0061    1.2293
          지수      2.0017    0.2618      5.9660    1.2995
        t(5)      0.0248   -0.0030      9.7070    1.3262
          균등      0.0012   -0.0001     -1.1998    0.9989

    (정규분포의 무어스 첨도 기준값은 1.2331)

    이상치 하나에 대한 민감도
      극단값   0 추가: 고전 왜도  -0.1205  보울리  0.0119   초과첨도   -0.0097  무어스 1.2893
      극단값  10 추가: 고전 왜도   0.7166  보울리  0.0115   초과첨도    7.5040  무어스 1.2875
      극단값  30 추가: 고전 왜도  10.1092  보울리  0.0115   초과첨도  218.8426  무어스 1.2875
    ```

    **네 분포의 순서는 두 방식이 일치한다.** 지수분포는 양쪽 모두 오른쪽 치우침을, $t(5)$는 양쪽 모두 두꺼운 꼬리를, 균등은 양쪽 모두 얇은 꼬리를 보고한다.

    **차이는 이상치를 만났을 때 드러난다.**

    | 극단값 | 고전 왜도 | 보울리 | 초과첨도 | 무어스 |
    |---|---|---|---|---|
    | 없음 | $-0.121$ | $0.0119$ | $-0.010$ | $1.289$ |
    | $10$ | $0.717$ | $0.0115$ | $7.504$ | $1.288$ |
    | $30$ | $\mathbf{10.109}$ | $\mathbf{0.0115}$ | $\mathbf{218.84}$ | $\mathbf{1.288}$ |

    **관측 $1001$개 중 하나 때문에 고전 첨도가 $219$가 된다.** 강건한 측도는 소수점 셋째 자리까지 그대로다.

    **왜 이렇게 취약한가.** 연습문제 7에서 본 $4$제곱 때문이다. $z = 30$인 점 하나의 기여가 $30^4 = 810000$이라 나머지 $1000$개를 압도한다. **고전 첨도의 붕괴점은 $0$이며, 사실상 "가장 극단적인 관측이 얼마나 극단적인가"를 재는 통계량에 가깝다.**

    **어느 쪽을 쓰는가.**

    - **분포의 모양을 서술하고 싶다면** 강건한 측도가 안전하다.
    - **극단값의 위험을 알고 싶다면** 고전 첨도가 적절하다. 금융에서 첨도를 보는 이유가 바로 그 민감성 때문이다.
    - **둘 다 계산해 보라.** 크게 다르면 소수의 관측이 결과를 좌우한다는 뜻이며, 그 자체가 유용한 정보다. 이상치 문서의 피어슨–스피어만 대조와 같은 발상이다.

    강건한 측도의 대가는 **정보를 덜 쓴다**는 것이다. 보울리 왜도는 세 분위수만 보므로 그 사이의 모양은 전혀 반영하지 않는다. $\square$

<div class="drillbox" markdown>

**연습문제 10.**
왜도와 첨도를 결합한 **자르크–베라 정규성 검정**을 살펴보라. 이 검정이 소표본과 대표본에서 각각 어떻게 행동하는가?

</div>

??? success "풀이"
    자르크–베라 통계량은 표본 왜도 $S$와 초과첨도 $K$를 결합한다.

    $$
    \text{JB} = \frac{n}{6}\left(S^2 + \frac{K^2}{4}\right) \;\xrightarrow{d}\; \chi^2_2
    $$

    정규분포에서 $S$와 $K$가 각각 $0$이어야 한다는 사실을 쓴 것이다.

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(1)

    print("정규 자료에서의 1종 오류율 (명목 0.05)")
    for n in (20, 50, 200, 1000, 5000):
        jb = np.mean([stats.jarque_bera(rng.normal(0, 1, n))[1] < 0.05 for _ in range(5000)])
        sw = np.mean([stats.shapiro(rng.normal(0, 1, n))[1] < 0.05 for _ in range(5000)])
        print(f"  n={n:>5}: JB {jb:.4f}   샤피로–윌크 {sw:.4f}")

    print("\nt(10) 자료를 정규가 아니라고 판정할 확률 (검정력)")
    for n in (20, 50, 200, 1000):
        jb = np.mean([stats.jarque_bera(rng.standard_t(10, n))[1] < 0.05 for _ in range(3000)])
        sw = np.mean([stats.shapiro(rng.standard_t(10, n))[1] < 0.05 for _ in range(3000)])
        print(f"  n={n:>5}: JB {jb:.4f}   샤피로–윌크 {sw:.4f}")
    ```

    출력:

    ```
    정규 자료에서의 1종 오류율 (명목 0.05)
      n=   20: JB 0.0228   샤피로–윌크 0.0478
      n=   50: JB 0.0408   샤피로–윌크 0.0458
      n=  200: JB 0.0400   샤피로–윌크 0.0484
      n= 1000: JB 0.0512   샤피로–윌크 0.0494
      n= 5000: JB 0.0524   샤피로–윌크 0.0468

    t(10) 자료를 정규가 아니라고 판정할 확률 (검정력)
      n=   20: JB 0.0843   샤피로–윌크 0.1037
      n=   50: JB 0.1837   샤피로–윌크 0.1650
      n=  200: JB 0.4500   샤피로–윌크 0.3483
      n= 1000: JB 0.9350   샤피로–윌크 0.8967
    ```

    **소표본에서 JB는 신뢰할 수 없다.** $n = 20$에서 1종 오류율이 $0.023$으로 명목의 절반이다. $\chi^2_2$ 근사가 $S$와 $K$의 느린 수렴 때문에 작은 $n$에서 나쁘기 때문이다. 샤피로–윌크는 모든 $n$에서 $0.05$를 지킨다.

    **검정력은 $n$이 커지면 JB가 앞선다.** $t(10)$에 대해 $n = 200$에서 $0.450$ 대 $0.348$이다. JB는 꼬리(첨도)를 직접 겨냥하므로 두꺼운 꼬리 이탈에 민감하다. 반면 **소표본에서는 검정력조차 낮다**($n = 20$에서 $0.084$).

    **더 근본적인 문제는 대표본이다.**

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(1)

    def tiny(n):
        """98% 는 N(0,1), 2% 는 N(0,2). 실질적으로는 정규와 다를 바 없다."""
        m = rng.random(n) < 0.02
        return np.where(m, rng.normal(0, 2.0, n), rng.normal(0, 1, n))

    x = tiny(2_000_000)
    print(f"이 분포: 왜도 {stats.skew(x):+.4f}   초과첨도 {stats.kurtosis(x):+.4f}")
    print("\n표본이 커지면 이 정도 이탈도 반드시 기각된다")
    for n in (100, 1000, 10_000, 100_000):
        r = np.mean([stats.jarque_bera(tiny(n))[1] < 0.05 for _ in range(400)])
        print(f"  n={n:>7}: 기각률 {r:.4f}")
    ```

    출력:

    ```
    이 분포: 왜도 +0.0016   초과첨도 +0.4599

    표본이 커지면 이 정도 이탈도 반드시 기각된다
      n=    100: 기각률 0.1425
      n=   1000: 기각률 0.5525
      n=  10000: 기각률 1.0000
      n= 100000: 기각률 1.0000
    ```

    **초과첨도가 $0.46$에 불과한 이 분포가 $n = 10000$에서 $100\%$ 기각된다.** 실무적으로는 정규와 구별할 이유가 없는 자료인데도 그렇다.

    **이것이 1장에서 본 문제의 재현이다.** 정규성 검정은 "정확히 정규인가"를 묻는데, 현실의 자료가 정확히 정규인 경우는 없다. $n$이 크면 어떤 검정이든 반드시 기각한다.

    !!! tip "정규성은 검정하지 말고 진단하라"
        - **소표본**: 검정력이 없어 기각하지 못한다. "기각되지 않았다"가 "정규다"를 뜻하지 않는다.
        - **대표본**: 무의미한 이탈도 기각한다. "기각되었다"가 "문제가 있다"를 뜻하지 않는다.
        - **어느 쪽이든 쓸모없는 구간이 있다.**

        올바른 질문은 "정규인가"가 아니라 **"내가 쓰려는 방법이 이 정도 이탈에 견디는가"** 이다. 표본평균의 추론은 중심극한정리 덕분에 상당한 이탈에도 견디고, 예측구간이나 극단 분위수는 훨씬 민감하다.

        그러므로 **Q-Q 그림을 그려 어디서 얼마나 벗어나는지 보는 것**이 $p$ 값 하나보다 언제나 많은 정보를 준다. 앞 절 ECDF 문서의 Q-Q 그림 절이 이 목적을 위한 것이다. $\square$

---

## 정리하며

왜도와 첨도는 분포에 대한 기술을 중심과 퍼짐 너머로 확장한다. 왜도는 방향성 있는 비대칭을 드러내어 대표적인 중심으로 평균과 중앙값 중 무엇을 고를지 이끌어 준다. 첨도는 꼬리의 행동을 정량화하는데, 극단적 사건(두꺼운 꼬리)이 큰 결과를 낳는 위험관리와 금융에서 매우 중요하다.
