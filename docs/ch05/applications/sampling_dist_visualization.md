# 표본분포 시각화: 표본크기의 효과

## 개요

이 절에서는 표본크기가 커질수록 표본평균의 **표본분포**가 어떻게 더 좁아지는지 보인다. 현실적인 소득 자료를 사용하여 모집단 분포, 표본 분포, 표본분포라는 세 가지 구별을 시각화한다.

## 세 가지 분포

표본을 반복해서 뽑고 통계량을 계산할 때 서로 다른 세 분포를 만나게 된다:

1. **모집단 분포**: 모집단 전체에 있는 모든 값의 분포
2. **표본 분포**: 특정한 하나의 표본 안에 있는 값들의 분포
3. **표본분포**: 여러 표본으로부터 계산한 통계량(예: 표본평균)의 분포

이는 **중심극한정리**의 핵심이다. 표본크기가 커지면 모집단 분포의 모양과 무관하게 평균의 표본분포가 정규분포에 가까워진다.

## 대출 소득 자료를 이용한 실증

다음 코드는 실제 소득 자료를 사용하여 표본크기가 커질 때 표본분포가 어떻게 좁아지는지 보인다:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(seed=1)

# Load income data (or use simulated data with similar properties)
# loans_income is a Series of income values
# For demonstration, we'll create synthetic data with similar characteristics
np.random.seed(1)
# Simulate left-skewed income distribution (like real loan data)
loans_income = np.random.exponential(scale=50000, size=10000) + 20000
loans_income = pd.Series(loans_income)

# Create three datasets:
# 1. A sample of 1000 individual income values from the population
sample_data = pd.DataFrame({
    'income': loans_income.sample(1000),
    'type': 'Population Sample\n(n=1000)',
})

# 2. Sampling distribution when drawing samples of size 5
# (Draw 1000 samples, compute mean of each)
sample_mean_05 = pd.DataFrame({
    'income': [loans_income.sample(5).mean() for _ in range(1000)],
    'type': 'Sampling Distribution\n(Mean of 5)',
})

# 3. Sampling distribution when drawing samples of size 20
sample_mean_20 = pd.DataFrame({
    'income': [loans_income.sample(20).mean() for _ in range(1000)],
    'type': 'Sampling Distribution\n(Mean of 20)',
})

# Combine all three
results = pd.concat([sample_data, sample_mean_05, sample_mean_20], ignore_index=True)

print("Summary of the three distributions:")
print(results.groupby('type')['income'].agg(['count', 'mean', 'std', 'min', 'max']))
print()

# Visualize all three distributions
g = sns.FacetGrid(results, col='type', col_wrap=1, height=2.5, aspect=2.5)
g.map(plt.hist, 'income', bins=40, range=[0, 200000], color='steelblue', edgecolor='black')
g.set_axis_labels('Income ($)', 'Frequency')
g.set_titles('{col_name}')

# Adjust layout
for ax in g.axes.flat:
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()
```

## 시각화 해석

### 모집단 표본 (위 패널)

모집단에서 나온 실제 소득 값의 분포를 보여 준다. 고소득자의 긴 꼬리를 갖는 **오른쪽으로 치우친** 분포로, 실제 소득 자료에서 전형적이다.

### n=5인 표본분포 (가운데 패널)

5명만 뽑아 평균 소득을 계산할 때, 이 1000개 표본평균의 분포는:

- 모집단보다 더 **좁게 모여** 있다
- 더 **대칭적이다**(정규분포 모양에 가까워진다)
- 여전히 모집단의 오른쪽 치우침이 조금 남아 있다

표본이 작으면 개별 극단값이 평균에 크게 영향을 미치기 때문이다.

### n=20인 표본분포 (아래 패널)

20명이라는 더 큰 표본에서는:

- 참 모평균 주위로 **더욱 좁게 모인다**
- 훨씬 더 **종 모양**이 된다(정규분포에 가까워진다)
- 이 관계는 표준오차 $SE = \frac{\sigma}{\sqrt{n}}$로 정량화된다

## 주요 관찰

### 표본크기가 커지면 표준오차가 줄어든다

**표준오차**(표본분포의 표준편차)는 $\sqrt{n}$에 반비례한다:

$$SE(\bar{X}) = \frac{\sigma}{\sqrt{n}}$$

우리 모의실험을 비교하면:

- $n = 5$일 때: $SE \approx \frac{\sigma}{\sqrt{5}} \approx 0.447\sigma$
- $n = 20$일 때: $SE \approx \frac{\sigma}{\sqrt{20}} \approx 0.224\sigma$

$n=20$의 표준오차는 $n=5$의 대략 절반이며, 그만큼 추정이 정밀해진다.

```python
# Verify standard error relationship
pop_std = loans_income.std()
se_5 = pop_std / np.sqrt(5)
se_20 = pop_std / np.sqrt(20)

print(f"Population standard deviation: ${pop_std:,.0f}")
print(f"SE for n=5:  ${se_5:,.0f}")
print(f"SE for n=20: ${se_20:,.0f}")
print(f"Ratio SE(5)/SE(20): {se_5/se_20:.2f}")
```

### 정규성으로의 수렴

중심극한정리에 따르면 모집단 분포의 모양과 무관하게 $n$이 커질수록 평균의 표본분포가 정규분포에 가까워진다. 소득이 오른쪽으로 치우쳐 있어도 표본분포는 점점 정규분포에 가까워진다.

### 실무적 함의

1. **표본크기 설계**: 불확실성을 절반으로 줄이려면 표본크기를 네 배로 늘려야 한다($\sqrt{4} = 2$이므로).
2. **신뢰구간**: 표본분포가 좁아지면 신뢰구간도 좁아진다.
3. **가설검정**: 표본이 클수록 참 효과를 탐지하는 통계적 검정력이 커진다.

## 정량적 비교

```python
import numpy as np
import pandas as pd

# Quantify the effect
np.random.seed(1)
loans_income = np.random.exponential(scale=50000, size=10000) + 20000

sample_means_5 = np.array([np.mean(np.random.choice(loans_income, 5)) for _ in range(1000)])
sample_means_20 = np.array([np.mean(np.random.choice(loans_income, 20)) for _ in range(1000)])

print("Sampling Distribution Comparison:")
print(f"{'Statistic':<20} {'n=5':<20} {'n=20':<20}")
print("-" * 60)
print(f"{'Mean':<20} ${sample_means_5.mean():>18,.0f} ${sample_means_20.mean():>18,.0f}")
print(f"{'Std Dev':<20} ${sample_means_5.std():>18,.0f} ${sample_means_20.std():>18,.0f}")
print(f"{'25th percentile':<20} ${np.percentile(sample_means_5, 25):>18,.0f} ${np.percentile(sample_means_20, 25):>18,.0f}")
print(f"{'75th percentile':<20} ${np.percentile(sample_means_5, 75):>18,.0f} ${np.percentile(sample_means_20, 75):>18,.0f}")
print(f"{'IQR':<20} ${np.percentile(sample_means_5, 75) - np.percentile(sample_means_5, 25):>18,.0f} ${np.percentile(sample_means_20, 75) - np.percentile(sample_means_20, 25):>18,.0f}")
```

## 요약

표본분포는 다음을 보여 준다:

- **통계적 정밀도**가 $1/\sqrt{n}$로 좋아진다
- 표본크기가 커질수록 참 모수 주위로의 **집중**이 강해진다
- 모집단이 정규가 아니어도 **정규성**이 나타난다(중심극한정리)
- 표본크기와 추정 정확도 사이의 **실무적 맞바꿈**이 있다

이 근본 개념이 신뢰구간, 가설검정, 그리고 표본평균에 기반한 모든 통계적 추론의 바탕이 된다.

## 연습문제

**연습문제 1.**
모집단이 치우쳐 있고 $\mu = 50$, $\sigma = 10$이다. 표본 $n = 100$일 때 (a) $\bar X$의 표본분포의 대략적인 모양은? (b) 그 평균과 표준오차는?

??? success "연습문제 1 풀이"
    (a) ($n$이 크므로) 중심극한정리에 의해 모집단이 치우쳐 있어도 $\bar X$는 근사적으로 **정규**이다.

    (b) 평균: $\mu = 50$. 표준오차: $\sigma/\sqrt n = 10/10 = 1$.

    따라서 $\bar X \approx N(50, 1)$이다. 밑바탕 모집단의 치우침이 표본분포에 (작은) 잔여 치우침으로 옮겨 오지만, $n = 100$에서는 중심극한정리가 만들어 내는 정규성이 압도한다.

---

**연습문제 2.**
**표본크기와 수렴 속도.** 지수 모집단(왜도 2)에서 $\bar X$의 표본분포는 어느 $n$에서 "근사적으로 정규"가 되는가? Berry-Esseen을 사용하여 정당화하라.

??? success "연습문제 2 풀이"
    Berry-Esseen 한계: $\rho = \mathbb{E}|X - \mu|^3$일 때 $\sup_x |F_{\bar X_n}(x) - \Phi((x - \mu)/(\sigma/\sqrt n))| \le C \cdot \rho/(\sigma^3 \sqrt n)$이다.

    $\mathrm{Exp}(1)$에서 $\rho \approx 2.0$, $\sigma = 1$이므로 한계는 $0.5 \cdot 2 / \sqrt n = 1/\sqrt n$이다.

    최대 KS 거리 $\le 0.05$로 "근사적으로 정규"라 하려면 $\sqrt n \ge 1/0.05 = 20$이므로 $n \ge 400$이다.

    $\le 0.10$이라면 $n \ge 100$이다.

    **실무에서는:** 치우침이 약하면 $n = 30$으로 충분하고, 중간 정도(예: 지수분포)면 $n = 100$, 치우침이나 꼬리가 심하면 $n = 1000$ 이상이 필요하다. 언제나 그림으로 확인하라.

---

**연습문제 3.**
**대안으로서의 붓스트랩.** 모집단 모양을 모르고 $n$이 중간 정도일 때 붓스트랩은 표본분포를 비모수적으로 추정해 준다. 그 절차를 서술하라.

??? success "연습문제 3 풀이"
    1. 미지의 모집단에서 뽑은 표본 $X_1, \ldots, X_n$이 주어진다.
    2. 각각 크기 $n$인 붓스트랩 표본을 복원추출로 $B$개 뽑는다.
    3. 각 붓스트랩 표본에 대해 $\bar X^*_b$를 계산한다.
    4. 모음 $\{\bar X^*_1, \ldots, \bar X^*_B\}$이 $\bar X$의 표본분포를 근사한다.

    이 분포를 사용하여:

    - 표준오차 추정: $\{\bar X^*_b\}$의 표본표준편차.
    - 신뢰구간 구성: 95% **백분위수 구간**을 위해 $\{\bar X^*_b\}$의 2.5 백분위수와 97.5 백분위수를 사용한다.

    붓스트랩은 중심극한정리에 기반한 정규근사가 놓치는 치우침, 두꺼운 꼬리 등의 특징을 포착한다. $n$이 중심극한정리를 쓰기에는 작고 정확한 소표본 추론을 하기에는 클 때 특히 유용하다.

---

**연습문제 4.**
**중심극한정리 시각화.** $n = 1, 5, 30, 100$인 지수 표본에 대해 중심극한정리를 보여 주는 일련의 그림을 서술하라.

??? success "연습문제 4 풀이"
    각 $n$마다 $\mathrm{Exp}(1)$에서 크기 $n$인 표본을 많이(가령 $B = 10000$개) 생성하고 각각에 대해 $\bar X_n$을 계산한 뒤 히스토그램을 그린다.

    예상되는 양상:

    - $n = 1$: 히스토그램이 지수분포처럼 보인다(오른쪽으로 치우치고 0에서 정점).
    - $n = 5$: 여전히 눈에 띄게 치우쳐 있지만 덜하다. 단봉이면서 오른쪽 꼬리가 길다.
    - $n = 30$: 평균 1, 표준편차 $1/\sqrt{30} \approx 0.18$인 근사적 정규분포. 오른쪽 치우침이 약간 남는다.
    - $n = 100$: 뚜렷한 정규분포 모양이고 표준편차 $\approx 0.1$.

    각 히스토그램에 $N(1, 1/n)$ 밀도를 겹쳐 그리면 중심극한정리의 수렴이 눈에 보인다. 종 모양이 좁아지고 치우침이 사라지는 것이 그 이야기를 들려준다.

    유용한 두 번째 그림: 각 $n$에서 $\bar X_n$에 대한 정규 Q-Q 그림. 점들이 점점 대각선 위에 놓인다.

---

**연습문제 5.**
**모분산의 효과.** $\mu = 50$인 모집단에 대해 $\sigma = 5, 10, 50$일 때 $n = 100$에서 $\bar X$의 표본분포를 비교하라.

??? success "연습문제 5 풀이"
    세 표본분포 모두 근사적으로 $N(\mu, \sigma^2/n) = N(50, \sigma^2/100)$이다:

    - $\sigma = 5$: $\bar X \approx N(50, 0.25)$, 표준편차 = 0.5.
    - $\sigma = 10$: $\bar X \approx N(50, 1.0)$, 표준편차 = 1.
    - $\sigma = 50$: $\bar X \approx N(50, 25)$, 표준편차 = 5.

    모두 50을 중심으로 하며 표준오차는 $\sigma$에 비례해 커진다. 모집단의 변동성이 클수록 표본분포도 그에 비례해 넓어진다.

    **표본크기 설계에 대한 함의:** 목표 정밀도 $\mathrm{SE} = \sigma_{\text{target}}$을 달성하려면 $n = (\sigma/\sigma_{\text{target}})^2$이 필요하다. 변동성이 큰 모집단에서 같은 정밀도를 얻으려면 훨씬 큰 표본이 필요하다.

---

**연습문제 6.**
**중앙값의 표본분포.** 평균의 표본분포와 모양, 표준오차 공식, 로버스트성을 간단히 대비하라.

??? success "연습문제 6 풀이"
    밀도가 $f$이고 중앙값이 $m$인 모집단에서 크기 $n$인 표본을 뽑을 때:

    - **모양:** 점근적으로 정규이다(정칙 조건 아래에서 중앙값도 자신의 중심극한정리를 갖는다).
    - **표준오차 공식:** $\mathrm{SE}(\tilde X) \approx 1/(2 f(m) \sqrt n)$. 중앙값에서의 밀도에 의존하며, $f(m)$이 크면 표준오차가 작다.

    **비교:**

    | 통계량 | 편향 | 분산 | 로버스트성 |
    |---|---|---|---|
    | 평균 | 0 | $\sigma^2/n$ | 이상점에 민감 |
    | 중앙값 | 0 | $1/(4 n f(m)^2)$ | 로버스트 |

    정규 자료에서는 $\mathrm{Var}(\tilde X)/\mathrm{Var}(\bar X) = \pi/2 \approx 1.57$로 중앙값이 덜 효율적이다(효율이 약 64%).

    꼬리가 두꺼운 자료(예: $t_3$이나 Laplace)에서는 중앙값이 평균보다 *더* 효율적이다. 이상점 때문에 평균의 분산이 부풀어 오르기 때문이다.

    평균과 중앙값의 선택은 자료에 맞춰야 한다. 깨끗하고 대칭인 자료에는 평균을, 이상점이 잦거나 꼬리가 두꺼운 자료에는 중앙값을 쓴다.
