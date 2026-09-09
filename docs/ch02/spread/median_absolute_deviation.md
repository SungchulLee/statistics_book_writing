# 중앙값 절대편차 (MAD)

## 개요

**중앙값 절대편차(MAD)** 는 자료가 중앙값 주위로 얼마나 퍼져 있는지를 재는 강건한 흩어짐 측도다. 분산이나 표준편차와 달리 MAD는 이상치에 저항하므로, 치우쳤거나 오염된 자료를 기술할 때 중앙값과 짝을 이루는 이상적인 측도다.

---

## 정의

MAD는 세 단계로 계산한다.

1. 중앙값 $M = \text{median}(x_1, x_2, \ldots, x_n)$을 구한다.
2. 각 관측값에 대해 절대편차 $d_i = |x_i - M|$을 계산한다.
3. 이 편차들의 중앙값을 구한다: $\text{MAD} = \text{median}(d_1, d_2, \ldots, d_n)$.

$$
\text{MAD} = \text{median}(|x_i - \text{median}(x)|)
$$

### 표준화 상수

MAD를 (특히 정규분포 자료에서) 표준편차와 직접 비교할 수 있게 하려면 표준화 상수로 나눈다.

$$
\text{Standardized MAD} = \frac{\text{MAD}}{0.6745} \approx 1.4826 \times \text{MAD}
$$

$0.6745 = \Phi^{-1}(0.75)$는 표준정규분포의 75번째 백분위수다. 정규분포 $N(\mu, \sigma^2)$에서 모집단 MAD가 정확히 $0.6745\,\sigma$이므로, 거꾸로 $0.6745$로 **나누어야** 표준화된 MAD가 $\sigma$의 추정값이 된다(연습문제 2에서 유도한다).

!!! warning "곱하는 것이 아니라 나눈다"
    $\sigma \approx 1.4826 \times \text{MAD}$이지 $0.6745 \times \text{MAD}$가 아니다. MAD는 언제나 $\sigma$보다 **작으므로**($0.6745$배), 이를 $\sigma$와 비교 가능하게 만들려면 키워야 한다.

    $\sigma = 15$인 정규자료 200만 개에서 확인하면 $\text{MAD} = 10.117$이고 $\text{MAD}/0.6745 = 15.000$인 반면 $0.6745 \times \text{MAD} = 6.824$로 크게 어긋난다.

    아래 코드와 R의 `mad()`, `statsmodels`의 `robust.scale.mad`가 모두 나누는 쪽을 쓴다.

---

## 예: 미국 주별 인구

주별 인구 자료로 MAD를 계산하고 표준편차와 비교한다.

```python
import pandas as pd
from statsmodels import robust

# Load state data
# 미국 50개 주의 인구와 살인율. 오른쪽으로 크게 치우친 전형적인 자료다.
url = ('https://raw.githubusercontent.com/gedeck/practical-statistics-for-data-scientists/master/data/state.csv')
state = pd.read_csv(url)

# Standard deviation (sensitive to outliers)
std_dev = state['Population'].std()
print(f"Standard Deviation: {std_dev:,.0f}")

# MAD using statsmodels
mad = robust.scale.mad(state['Population'])
print(f"MAD (standardized): {mad:,.0f}")

# Manual calculation
median_pop = state['Population'].median()
abs_deviations = abs(state['Population'] - median_pop)
mad_manual = abs_deviations.median()
mad_standardized = mad_manual / 0.6744897501960817
print(f"MAD (manual calc): {mad_standardized:,.0f}")
```

**출력:**
```
Standard Deviation: 6,848,235
MAD (standardized): 3,849,876
MAD (manual calc): 3,849,876
```

캘리포니아의 극단적인 인구(중앙값 440만 명에 비해 3700만 명)가 표준편차에 큰 영향을 주어 값을 끌어올린다. 중앙값으로부터의 편차에 근거하는 MAD는 이 이상치의 영향을 덜 받는다.

---

## MAD가 강건한 이유

이 두 측도에 이상치가 미치는 영향을 살펴보자.

```python
import pandas as pd
import numpy as np
from statsmodels import robust

# Original state population data
# 미국 50개 주의 인구와 살인율. 오른쪽으로 크게 치우친 전형적인 자료다.
url = ('https://raw.githubusercontent.com/gedeck/practical-statistics-for-data-scientists/master/data/state.csv')
state = pd.read_csv(url)
original_std = state['Population'].std()
original_mad = robust.scale.mad(state['Population'])

# Introduce extreme outliers
population_with_outliers = pd.concat([
    state['Population'],
    pd.Series([100_000_000, 150_000_000])  # Two fictional giant states
])

outlier_std = population_with_outliers.std()
outlier_mad = robust.scale.mad(population_with_outliers)

print("Impact of Outliers:")
print(f"  Std Dev: {original_std:,.0f} → {outlier_std:,.0f} ({100 * (outlier_std - original_std) / original_std:.1f}% increase)")
print(f"  MAD:     {original_mad:,.0f} → {outlier_mad:,.0f} ({100 * (outlier_mad - original_mad) / original_mad:.1f}% increase)")
```

출력:

```
Impact of Outliers:
  Std Dev: 6,848,235 → 24,537,372 (258.3% increase)
  MAD:     3,849,876 → 4,273,462 (11.0% increase)
```

극단적인 이상치 두 개를 추가하면 표준편차는 극적으로 커지지만 MAD는 거의 변하지 않는다. 이것이 MAD의 강건성을 보여준다.

---

## 강건성의 성질

MAD는 다음과 같은 성질을 갖는 **강건한** 통계량이다.

- **붕괴점:** 표준편차가 0%인 데 비해, MAD는 자료의 최대 50%가 임의로 오염되어도 신뢰성을 잃지 않는다.
- **영향함수:** 유계다. 극단적인 이상치 하나가 미치는 영향이 제한된다.
- **효율:** 정규분포 자료에서 MAD는 표준편차의 약 64% 효율을 갖는다. MAD가 얻는 막대한 강건성을 생각하면 이 효율 손실은 작다.

---

## 비교: 표준편차 대 MAD

| 특성 | 표준편차 | MAD |
|---|---|---|
| 이상치에 대한 민감성 | 높음 | 낮음 |
| 모든 자료점 사용 | 예 | 예 |
| 붕괴점 | 0% | 50% |
| 계산 복잡도 | $O(n)$ | $O(n \log n)$ (정렬 때문) |
| 해석 용이성 | 대부분의 분석가에게 익숙 | 덜 익숙 |
| 효율(정규 자료) | 100% | 64% |

---

## MAD를 쓸 때

**치우친 분포:** 소득, 자산, 그 밖에 오른쪽으로 치우친 금융 자료
**이상치가 많은 자료:** 센서 측정값, 천문 관측
**강건 추정:** 모든 자료점을 똑같이 신뢰할 수 없을 때
**비정규 자료:** 꼬리가 두껍거나 다봉인 분포

---

## 실용적 예: 금융 수익률

주식시장 분석에서 MAD가 표준편차보다 대표성이 클 수 있다.

```python
import pandas as pd
from statsmodels import robust

# Hypothetical daily stock returns
returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005, 0.03, -0.02,
                      0.01, -0.01, 0.005, -0.015, 0.02, -0.50])  # One crash day

print(f"Standard Deviation: {returns.std():.4f}")
print(f"MAD (standardized): {robust.scale.mad(returns):.4f}")

# The crash day (-0.50) inflates std dev much more than MAD
```

출력:

```
Standard Deviation: 0.1407
MAD (standardized): 0.0222
```

폭락한 하루(-0.50)가 표준편차를 크게 키워 전형적인 일간 변동성을 과장할 수 있다. MAD는 일상적인 변동에 대해 더 선명한 그림을 준다.

---

## 파이썬에서 MAD 계산하기

### statsmodels 사용 (권장)

```python
from statsmodels import robust
import pandas as pd

data = pd.Series([1, 2, 3, 4, 5, 100])  # Last value is an outlier
mad = robust.scale.mad(data)
print(f"MAD: {mad:.2f}")
```

출력:

```
MAD: 2.22
```

### 직접 계산

```python
import pandas as pd
import numpy as np

data = pd.Series([1, 2, 3, 4, 5, 100])
median = data.median()
abs_dev = abs(data - median)
mad = abs_dev.median()
mad_standardized = mad / 0.6744897501960817  # Standardize for normal data
print(f"MAD (standardized): {mad_standardized:.2f}")
```

출력:

```
MAD (standardized): 2.22
```

---

## 요약

중앙값 절대편차는 이상치가 있는 상황에서 자료의 퍼짐을 재는 강력한 도구다. 흩어짐을 (그 자체가 강건한) 중앙값으로부터의 편차에 근거해 계산함으로써, MAD는 분산과 표준편차가 따라올 수 없는 수준의 안정성을 얻는다. 치우친 자료, 이상치, 비정규 분포가 관여하는 분석이라면 중앙값과 MAD를 짝짓는 것이 평균과 표준편차보다 더 믿을 만한 요약을 준다.

## 연습문제

**연습문제 1.**
어떤 품질관리 공정이 지름 측정값 10개(mm)를 기록했다: $10.1, 10.0, 9.9, 10.2, 10.0, 9.8, 10.1, 10.0, 15.3, 10.0$. (a) $s$를 계산하라. (b) MAD를 계산하라. (c) 척도를 맞춘 MAD($1.4826 \cdot \text{MAD}$)를 계산하라. 비교하고 설명하라.

??? success "풀이"
    (a) $\bar x = 10.54$. 제곱편차의 합 $= 25.284$이며, 이 중 $(15.3 - 10.54)^2 = 22.66$ 항이 약 90%를 차지한다. $s^2 = 25.284/9 = 2.809$, $s \approx 1.676$.

    (b) 정렬하면 $9.8, 9.9, 10.0, 10.0, 10.0, 10.0, 10.1, 10.1, 10.2, 15.3$이고 중앙값 $= 10.0$이다. 절대편차를 정렬하면 $0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.2, 0.2, 5.3$이므로 MAD $= (0.1 + 0.1)/2 = 0.1$이다.

    (c) 척도를 맞춘 MAD $= 1.4826 \times 0.1 \approx 0.148$.

    표준편차($1.68$)가 척도를 맞춘 MAD($0.15$)의 11배가 넘는다. 이상치 15.3 하나가 표준편차를 엄청나게 부풀리는 반면 MAD는 사실상 건드리지 못한다. 이 자료에서는 MAD가 전형적인 퍼짐을 훨씬 정직하게 재는 측도다.

---

**연습문제 2.**
정규분포 아래에서 MAD를 $\sigma$와 같게 만드는 일치성 상수 $1/\Phi^{-1}(0.75) \approx 1.4826$을 유도하라.

??? success "풀이"
    $X \sim N(\mu, \sigma^2)$에서 중앙값은 $\mu$이므로 $|X - \mu|/\sigma$는 **반정규(half-normal)** 분포를 따른다. MAD에 $c$를 곱했을 때 $c \cdot \text{MAD} = \sigma$가 되는 $c$를 찾고자 한다.

    대칭성에 의해 $P(|X - \mu| \le m) = P(-m \le X - \mu \le m) = 2\Phi(m/\sigma) - 1$이다. 중앙값의 정의에 따라 이를 0.5로 두면

    $$
    2\Phi(m/\sigma) - 1 = 0.5 \implies \Phi(m/\sigma) = 0.75 \implies m/\sigma = \Phi^{-1}(0.75) \approx 0.6745
    $$

    이다. 따라서 모집단 MAD는 $0.6745 \sigma$이다. 척도 상수 $c = 1/0.6745 \approx 1.4826$이 MAD를 $\sigma$로 되돌린다. 대부분의 소프트웨어 라이브러리(R의 `mad()`, statsmodels의 `robust.scale.mad`)가 이 상수를 자동으로 적용하는 이유가 이것이다.

---

**연습문제 3.**
추정량의 **붕괴점**은 그 추정량을 참값에서 임의로 멀리 보낼 수 있게 되기까지 임의의 값으로 바꿔야 하는 자료의 비율이다. MAD의 붕괴점이 50%이고 표준편차의 붕괴점이 0%임을 보여라.

??? success "풀이"
    **표준편차의 붕괴점 0%:** 유한한 값들로 이루어진 크기 $n$의 표본을 생각하자. 관측값 하나 $x_i$를 값 $M$으로 바꾼다. 새 평균은 $M/n$처럼 커지지만 새 표준편차는 $M/\sqrt{n}$처럼 커진다. $M \to \infty$이면 둘 다 한없이 커진다. 따라서 자료의 $1/n$(0이 아닌 가장 작은 비율)만 바꿔도 표준편차를 임의로 크게 만들 수 있다. $1/n \to 0$이므로 붕괴점은 0이다.

    **MAD의 붕괴점 50%:** 표본의 중앙값은 값 하나를 바꿀 때마다 순위 위치가 많아야 하나씩 움직인다. 값의 절반보다 적게 바꾸면 중앙값은 여전히 원래의 "가운데" 자료에 묶여 있다. 마찬가지로 절대편차 $|x_i - \text{median}|$의 중앙값도 자료의 본체에 의존한다. 적어도 $\lceil n/2 \rceil$개를 바꿔야만 중앙값(따라서 MAD)을 임의의 위치로 옮길 수 있다. 따라서 큰 $n$에 대해 붕괴점은 $\lfloor n/2 \rfloor / n \approx 0.5$다.

    이것은 합리적인 위치/척도 추정량이 가질 수 있는 최대 붕괴점이다. 이론적 상한인 50%이며, 중앙값과 MAD(그리고 몇몇 M-추정량)만이 여기에 도달한다.

---

**연습문제 4.**
정규성 아래에서 MAD는 표준편차보다 **효율**이 낮아 가우시안 효율이 약 37%다. 통계적 효율을 정의하고, 그럼에도 많은 응용 맥락에서 MAD를 선호하는 것을 정당화하는 편향–분산 절충을 설명하라.

??? success "풀이"
    추정량 $\hat\theta$의 기준 추정량 $\hat\theta^*$에 대한 **효율**은 두 추정량의 점근분산의 비다. 정규성 아래에서 $\sigma_{\text{eff(MAD)}} \approx 0.37 \cdot \sigma_{\text{eff(SD)}}$이며, 자료가 정말로 정규일 때 MAD의 분산이 표준편차의 약 $1/0.37 \approx 2.7$배라는 뜻이다.

    **절충 관계:**

    - 자료가 *정확히* 정규라면 표준편차는 정보를 낭비하지 않아 효율이 100%이고, MAD는 자료를 낭비해 정밀도가 떨어진다.
    - 자료가 *오염되어* 있다면 — 이상치가 아주 조금만 있어도 — 이상치가 제곱항으로 기여하므로 표준편차의 분산이 폭증한다. MAD의 분산은 거의 그대로다.

    거의 결코 정확히 정규가 아닌 실제 자료에서는, MAD의 낮은 가우시안 효율이라는 비용을 오염에 대한 둔감함이 충분히 상쇄하고도 남는다. 일반적인 설계 원칙은 이렇다. **가정된 정규성이 조금만 어긋나도 파국적으로 실패하는 대가를 치르면서까지 최선의 경우에 최적화하지 마라.** 이것이 "강건통계"의 핵심이다.

---

**연습문제 5.**
이상치 탐지를 위한 **수정 Z-점수**는 $M_i = 0.6745 \cdot (x_i - \tilde x) / \text{MAD}$이다. 이상치 탐지에서 이것이 고전적인 Z-점수 $Z_i = (x_i - \bar x) / s$보다 선호되는 이유는 무엇인가?

??? success "풀이"
    고전적인 Z-점수는 (이상치에 민감한) $\bar x$와 (매우 민감한) $s$를 쓴다. 이상치가 둘 다 부풀려 스스로를 **가린다**. 표시되어야 할 바로 그 점이 평균과 표준편차를 자기 쪽으로 끌어당겼기 때문에 $|Z|$가 작아진다.

    수정 Z-점수는 중앙값(붕괴점 50%)과 MAD(붕괴점 50%)를 쓴다. 이상치는 둘 중 어느 쪽에도 무시할 만한 영향만 주므로, 진짜 이상치에 대해서는 Z와 비슷한 이 통계량이 크게 유지된다. 계수 $0.6745$는 수정 Z-점수를 정규성 아래의 고전적 Z와 비슷한 척도로 맞춘다. 즉 깨끗한 자료에서 $|M_i| > 3.5$가 $|Z_i| > 3$과 대략 같은 꼬리 희귀도에 대응한다.

    Iglewicz and Hoaglin(1993)은 이상치 표시 기준으로 $|M_i| > 3.5$를 권장하여, 고전적인 $|Z| > 3$ 규칙의 강건한 대안을 제공했다.

---

**연습문제 6.**
MAD는 여러 강건 척도 추정량 중 하나다. 이를 **사분위범위**(IQR) 및 **Qn 추정량**(Rousseeuw–Croux)과 간략히 비교하라. 각각은 언제 고르겠는가?

??? success "풀이"
    **MAD:** $|x_i - \tilde x|$의 중앙값. 붕괴점 50%, 가우시안 효율 37%. 간단하고 널리 구현되어 있으며 기본적인 강건 척도다.

    **IQR:** $Q_3 - Q_1$. 붕괴점 25%(사분위수 하나만 오염시키면 된다). 개념적으로 더 간단하지만 붕괴점이 낮다. 정규 자료에서 $1.349 \sigma$를 통해 $\sigma$와 척도가 맞는다. 상자그림의 사실상 표준이다.

    **Qn 추정량**(Rousseeuw and Croux 1993): 차이 $|x_i - x_j|$에 근거한 강건 척도로, 모든 쌍의 차이의 제1사분위수에 정규화 상수를 곱해 계산한다. 붕괴점 50%, 가우시안 효율 82%로 MAD보다 약 2배 낫다. 대가는 MAD의 $O(n)$에 비해 $O(n \log n)$의 계산량이다.

    **선택 기준:**

    - **MAD**: 기본적인 강건 척도. 간단하고 빠르며 잘 알려져 있다.
    - **IQR**: 상자그림과 빠른 기술적 요약. 높은 붕괴점이 필수인 경우에는 부적합하다.
    - **Qn**: 높은 가우시안 효율이 중요하고 계산 비용을 감당할 수 있는 큰 표본. 본격적인 강건 추정의 최신 표준이다.
