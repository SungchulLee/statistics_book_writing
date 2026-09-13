# 꼬리가 두꺼운 분포

## 개요

바탕 분포의 꼬리가 두꺼우면 극단 관측값 때문에 표본평균의 성능이 나빠질 수 있다. 자산 수익률이 정규분포보다 훨씬 두꺼운 꼬리를 보이는 금융에서 특히 그렇다.

## 특징

꼬리가 두꺼운 분포는 꼬리가 지수보다 느리게 감쇠한다. 예를 들면:

- **스튜던트 $t$ 분포** (자유도가 작을 때)
- **코시분포** (평균과 분산이 정의되지 않음)
- **파레토분포** (거듭제곱 법칙 꼬리)
- **로그정규분포** (오른쪽으로 치우침)
- **금융 수익률** (주식, 외환, 원자재 시장에서 경험적으로 관찰됨)

## 첨도와 꼬리의 무게

꼬리의 무게는 흔히 **첨도**로 정량화한다. 정규분포의 첨도는 3(초과첨도 0)이다. 초과첨도가 1보다 큰 분포는 꼬리가 두껍다고 본다.

$$\text{초과첨도} = E\left[\left(\frac{X - \mu}{\sigma}\right)^4\right] - 3$$

**예시**:

- 정규분포: 초과첨도 = 0
- 자유도 5인 스튜던트 $t$: 초과첨도 ≈ 6 (훨씬 두꺼운 꼬리)
- 실제 주식 수익률: 초과첨도는 보통 3–10 (빈도와 자산에 따라 다름)

## 표본평균에 미치는 영향

자료가 꼬리가 두꺼운 분포에서 나오면:

1. **$\bar{X}$의 분산이 커진다**: 표준오차가 정규근사가 예측하는 것보다 크다
2. **정규성으로의 수렴이 느리다**: 중심극한정리는 여전히 성립하지만 수렴이 느리다. $n = 30$으로는 부족할 수 있다
3. **이상점의 영향이 과도하다**: 극단 관측값 하나가 표본평균을 크게 옮길 수 있다
4. **신뢰구간이 불확실성을 과소평가한다**: 정규이론에 기반한 구간이 너무 좁아 포함확률이 명목값에 못 미친다

## 금융에서의 맥락: 자산 수익률

경험적 증거는 금융 수익률이 두꺼운 꼬리를 보인다는 것을 일관되게 확인해 준다:

- **일별 주식 수익률**: 초과첨도 3–6 (전형적으로)
- **일중 수익률**: 꼬리가 더 두껍다
- **원자재 가격**: 공급 충격 때 꼬리가 매우 두껍다
- **외환**: 초과첨도가 중간 정도

### 금융 수익률의 꼬리가 두꺼운 이유

1. **드문 사건이 뭉쳐서 일어난다**: 시장 폭락과 급등은 고르게가 아니라 물결처럼 온다
2. **변동성 군집**: 변동성이 높은 시기에 큰 움직임이 더 많이 몰린다
3. **정보 비대칭**: 갑작스러운 뉴스가 불연속적인 점프를 만든다
4. **레버리지와 마진콜**: 하락 움직임을 증폭할 수 있다

### 위험관리에 대한 함의

수익률의 꼬리가 두꺼운데 정규성을 가정하면:

- 99번째 백분위수에서의 **VaR**가 심각하게 과소평가된다
- **기대부족액(CVaR)**이 과소평가된다
- **헤지 비율**이 너무 작아 포지션이 충분히 보호되지 않는다
- 스트레스 시기에 **자본 요구량**이 부족해진다

**예시**: 정규분포는 5% 손실이 확률 0.01%로 일어난다고 예측한다. 꼬리가 두꺼운 금융 수익률에서는 이 손실이 확률 0.1%로 일어날 수 있다 — 10배의 과소평가다!

## 시각적 비교

<div class="codebox" markdown>

### 예제 1. 정규와 두꺼운 꼬리를 네 그림으로 { .eg }

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# 두 자료는 중심도 척도모수도 같다. 다른 것은 꼬리의 두께뿐이다.
normal_returns = np.random.normal(loc=0, scale=0.02, size=5000)
heavy_tailed_returns = stats.t.rvs(df=6, scale=0.02, size=5000)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 윗줄: 히스토그램. 가운데만 보면 두 분포는 거의 구별되지 않는다.
# 차이는 그림의 양끝, 자료가 드문 자리에 숨어 있다.
ax = axes[0, 0]
ax.hist(normal_returns, bins=50, alpha=0.6, label='Normal', color='blue', density=True)
x = np.linspace(-0.08, 0.08, 200)
ax.plot(x, stats.norm.pdf(x, 0, 0.02), 'b-', linewidth=2, label='Normal PDF')
ax.set_title('Normal Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('Return')
ax.set_ylabel('Density')
ax.legend()
ax.spines[['top', 'right']].set_visible(False)

ax = axes[0, 1]
ax.hist(heavy_tailed_returns, bins=50, alpha=0.6, label='Heavy-tailed', color='red', density=True)
x = np.linspace(-0.08, 0.08, 200)
ax.plot(x, stats.t.pdf(x, df=6, loc=0, scale=0.02), 'r-', linewidth=2, label="Student's t PDF")
ax.set_title("Heavy-Tailed (Student's t) Distribution", fontsize=12, fontweight='bold')
ax.set_xlabel('Return')
ax.set_ylabel('Density')
ax.legend()
ax.spines[['top', 'right']].set_visible(False)

# 아랫줄: Q-Q 그림. 히스토그램이 감추는 꼬리를 드러내려고 쓴다.
# 두꺼운 꼬리는 양끝이 직선에서 S 자로 벌어지는 모습으로 나타난다.
ax = axes[1, 0]
stats.probplot(normal_returns, dist="norm", plot=ax)
ax.set_title('Q-Q Plot: Normal Data', fontsize=12, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)

ax = axes[1, 1]
stats.probplot(heavy_tailed_returns, dist="norm", plot=ax)
ax.set_title('Q-Q Plot: Heavy-Tailed Data', fontsize=12, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()

# 초과첨도는 정규분포를 0 으로 두고 잰 꼬리의 두께다. 자유도 6 인 t 는
# 이론값이 3 이고, 정규 쪽은 표본 흔들림만큼만 0 에서 벗어난다.
print("Normal Distribution:")
print(f"  Excess Kurtosis: {stats.kurtosis(normal_returns):.2f}")
print()
print("Heavy-Tailed (t) Distribution:")
print(f"  Excess Kurtosis: {stats.kurtosis(heavy_tailed_returns):.2f}")
```

출력:

```
Normal Distribution:
  Excess Kurtosis: 0.04

Heavy-Tailed (t) Distribution:
  Excess Kurtosis: 1.78
```

![Normal Distribution](./img/heavy_tails_67.png)

</div>

## 평균의 로버스트한 대안

꼬리가 두꺼운 자료에서는 다음 대안을 고려하라.

### 1. 중앙값
- **로버스트성**: 극단값의 영향을 받지 않는다
- **효율**: 정규 자료에서는 평균보다 효율이 낮지만, 꼬리가 두꺼운 자료에서는 비슷하다
- **추론**: 표준오차와 신뢰구간에는 붓스트랩을 쓴다

<div class="codebox" markdown>

#### 예제 2. 중앙값의 표준오차를 붓스트랩으로 { .eg }

```python
import numpy as np
from sklearn.utils import resample

data = stats.t.rvs(df=5, size=100)
original_median = np.median(data)

# 중앙값에는 표본평균의 sigma/sqrt(n) 같은 간단한 표준오차 공식이 없다.
# 대신 자료에서 복원추출로 재표본을 1000번 만들어 그때마다 중앙값을 구한다.
# 그 1000개의 표준편차가 곧 중앙값의 표준오차 추정값이다.
bootstrap_medians = [np.median(resample(data)) for _ in range(1000)]
se_median = np.std(bootstrap_medians)
print(f"Median: {original_median:.4f} ± {se_median:.4f}")
```

출력:

```
Median: 0.2379 ± 0.1549
```

</div>

### 2. 절사평균
평균을 계산하기 전에 양쪽 꼬리에서 일정 비율을 제거한다:

$$\bar{X}_{\text{trim}, \alpha} = \frac{1}{n(1-2\alpha)} \sum_{i=\lceil n\alpha \rceil}^{\lfloor n(1-\alpha) \rfloor} X_{(i)}$$

여기서 $X_{(i)}$는 순서통계량이고 $\alpha$는 절사비율이다(예: 10%이면 0.1).

<div class="codebox" markdown>

#### 예제 3. 절사평균 { .eg }

```python
import numpy as np
from scipy import stats
from scipy.stats import trim_mean

np.random.seed(42)
data = stats.t.rvs(df=5, size=100)      # 자유도 5의 t분포. 꼬리가 두껍다.

# proportiontocut=0.1 은 **양쪽 각각** 10%를 잘라 낸다는 뜻이다.
# 즉 전체의 20%가 버려지고 가운데 80%만 평균에 들어간다.
mean_trim10 = trim_mean(data, 0.1)
print(f"표본평균     {np.mean(data):7.4f}")
print(f"10% 절단평균 {mean_trim10:7.4f}")
```

출력:

```
표본평균      0.0299
10% 절단평균 -0.1366
```

</div>

### 3. 윈저화 평균
극단값을 버리는 대신 $\alpha$-분위수로 대체한다:

<div class="codebox" markdown>

#### 예제 4. 윈저화 평균 { .eg }

```python
import numpy as np
from scipy import stats

def winsorize_mean(data, alpha=0.1):
    """꼬리를 잘라 내는 대신 분위수 값으로 **바꿔치기**한 뒤 평균을 낸다.

    절단평균과의 차이는 표본 크기다.
    절단은 관측값을 버려 n이 줄지만, 윈저화는 값만 바꾸고 개수는 그대로 둔다.
    "극단값도 방향 정보는 담고 있다"고 볼 때 윈저화가 낫다.
    """
    lower = np.quantile(data, alpha)
    upper = np.quantile(data, 1 - alpha)
    winsorized = np.clip(data, lower, upper)   # 범위 밖 값을 경계로 눌러 준다
    return np.mean(winsorized)

np.random.seed(42)
data = stats.t.rvs(df=5, size=100)
print(f"표본평균       {np.mean(data):7.4f}")
print(f"윈저화 평균    {winsorize_mean(data, alpha=0.1):7.4f}")
```

출력:

```
표본평균        0.0299
윈저화 평균    -0.1292
```

</div>

### 4. M-추정량 (Huber 추정량)
작은 오차에서는 이차식, 큰 오차에서는 절댓값으로 넘어가는 손실함수를 써서 극단값의 가중치를 매끄럽게 낮춘다:

<div class="codebox" markdown>

#### 예제 5. Huber M-추정량 { .eg }

```python
import numpy as np
from scipy import stats
# Huber 추정량은 scipy가 아니라 statsmodels에 있다.
from statsmodels.robust.scale import huber

np.random.seed(42)
data = stats.t.rvs(df=5, size=100)

# 위치와 척도를 **동시에** 반복 추정해 돌려준다.
# 조율모수 t의 기본값은 1.5이며, 이는 표준화 잔차가 1.5를 넘는 관측값부터
# 가중치를 낮추기 시작한다는 뜻이다. 작을수록 로버스트하지만 효율이 떨어진다.
loc, scale = huber(data)
print(f"Huber 위치추정: {loc:.4f}")
print(f"Huber 척도추정: {scale:.4f}")
```

출력:

```
Huber 위치추정: -0.1366
Huber 척도추정: 1.0812
```

</div>

## 추정량의 비교

<div class="codebox" markdown>

### 예제 6. 다섯 추정량 견주기 { .eg }

```python
import numpy as np
from scipy import stats
from scipy.stats import trim_mean
from statsmodels.robust.scale import huber

np.random.seed(42)
# 자유도 5의 t분포. 평균은 0이지만 꼬리가 정규분포보다 훨씬 두껍다.
data = stats.t.rvs(df=5, loc=0, scale=1, size=500)

# 다섯 추정량 모두 같은 모수(위치 0)를 겨냥한다.
# n = 500 으로 넉넉하므로 다섯 값이 모두 0 근처에 모인다.
# 이들의 차이는 한 번의 값이 아니라 **되풀이했을 때의 흩어짐**에서 드러난다.
print("Estimator Comparison (Population mean = 0):")
print(f"  Sample mean:         {np.mean(data):7.4f}")
print(f"  Median:              {np.median(data):7.4f}")
print(f"  10% Trimmed mean:    {trim_mean(data, 0.1):7.4f}")
print(f"  Winsorized mean:     {winsorize_mean(data, 0.1):7.4f}")
print(f"  Huber's estimator:   {huber(data)[0]:7.4f}")
```

출력:

```
Estimator Comparison (Population mean = 0):
  Sample mean:         -0.0009
  Median:               0.0058
  10% Trimmed mean:    -0.0406
  Winsorized mean:     -0.0318
  Huber's estimator:   -0.0369
```

</div>

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코시분포의 PDF는 $f(x) = \frac{1}{\pi(1+x^2)}$이다. $\int_{-\infty}^{\infty} |x| f(x)\,dx$가 발산함을 보여 평균이 존재하지 않음을 증명하라.

</div>

??? success "풀이"
    $$
    \int_{-\infty}^{\infty} |x| f(x)\,dx = \frac{2}{\pi}\int_0^{\infty} \frac{x}{1+x^2}\,dx
    $$

    치환 $u = 1 + x^2$, $du = 2x\,dx$를 쓰면:

    $$
    = \frac{2}{\pi} \cdot \frac{1}{2}\int_1^{\infty} \frac{du}{u} = \frac{1}{\pi}\left[\log u\right]_1^{\infty} = \frac{1}{\pi}(\infty - 0) = \infty
    $$

    $E[|X|] = \infty$이므로 평균 $E[X]$가 존재하지 않는다. 적분이 로그 속도로 발산하므로, 아주 큰 표본에서 평균을 내도 값이 안정되지 않는다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
자유도 $\nu$인 스튜던트-$t$ 분포는 $\nu > 2$일 때만 분산이 유한하다. $\nu = 3$이면 분산은 $\sigma^2 = \nu/(\nu-2) = 3$이다. $t_3$에서 뽑은 관측값 $n = 100$개와 $N(0,3)$에서 뽑은 경우를 $\bar{X}$의 표준오차 관점에서 비교하라.

</div>

??? success "풀이"
    $N(0,3)$에서 표준오차는 $\text{SE} = \sqrt{3/100} = \sqrt{0.03} \approx 0.173$이다. 모집단이 정규이므로 중심극한정리가 완벽하게 적용된다.

    $t_3$에서도 모분산이 3이므로 이론적 표준오차는 같다: $\text{SE} = \sqrt{3/100} \approx 0.173$. 그러나 $t_\nu$의 초과첨도는 $\nu > 4$일 때 $6/(\nu-4)$이고, $2 < \nu \le 4$에서는 4차 적률이 존재하지 않아 초과첨도가 무한하다. $\nu = 3$이 바로 그 경우이다.

    실제로 $t_3$에서 얻은 표본평균은 이따금 나타나는 극단 관측값이 $\bar{X}$를 크게 밀어내기 때문에 표준오차 공식이 예측하는 것보다 훨씬 크게 요동친다. $t_3$에서는 중심극한정리의 수렴이 매우 느려 $n = 100$에서도 $\bar{X}$의 정규근사가 나쁘다 — 정규성에 기반한 신뢰구간의 포함확률이 명목 수준을 크게 밑돈다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
정규 자료에서는 표본평균이 표준적인 선택인데도, 코시분포의 중심 추정에서는 왜 중앙값이 더 나은지 설명하라.

</div>

??? success "풀이"
    코시 자료의 **표본평균**은 수렴하지 않는다: 주목할 만한 성질에 의해, i.i.d. 코시 관측값 $n$개의 표본평균은 $n$과 무관하게 같은 코시분포를 따른다. 평균이 존재하지 않아 대수의법칙이 적용되지 않으므로 평균을 내도 변동성이 전혀 줄지 않는다.

    반면 **표본중앙값**은 코시분포의 위치모수에 대해 일치하며 점근분산이 $\pi^2/(4n)$으로 표준적인 $1/n$ 속도로 줄어든다. 중앙값은 꼬리의 극단 관측값에 영향받지 않으므로, 평균을 무용지물로 만드는 두꺼운 꼬리에 로버스트하다. 특히 코시분포에서 중앙값은 위치모수의 MLE이다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
어떤 위험관리자가 정규 모형으로 일별 포트폴리오 손실의 99번째 백분위수를 추정하여 \$233만을 얻었다. 손실의 참 분포가 같은 척도의 $t_5$ 분포를 따른다면 참 99번째 백분위수는 얼마나 더 큰가?

</div>

??? success "풀이"
    표준정규의 99번째 백분위수는 $z_{0.99} = 2.326$이다. $t_5$ 분포의 99번째 백분위수는 $t_{5, 0.99} \approx 3.365$이다.

    비는:

    $$
    \frac{t_{5,0.99}}{z_{0.99}} = \frac{3.365}{2.326} \approx 1.447
    $$

    $t_5$ 모형에서의 참 99번째 백분위수는 약 44.7% 더 크다: $\$233\text{만} \times 1.447 \approx \$337\text{만}$. 정규 모형이 꼬리 위험을 이렇게 과소평가하는 것은 금융 위험관리에서 잘 알려진 위험 요인이며 2008년 금융위기의 한 원인이기도 했다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
**꼬리 지수** $\alpha$를 힐 추정량으로 추정하는 방법을 적고, 상위 $k$개를 몇 개로 잡을지 정하는 문제를 논하라.

</div>

??? success "풀이"
    **힐 추정량.** 정렬한 자료 $x_{(1)}\ge\cdots\ge x_{(n)}$(내림차순)에서 상위 $k$개를 써서

    $$
    \hat\alpha_{\text{Hill}} = \left\{\frac1k\sum_{i=1}^k \ln\frac{x_{(i)}}{x_{(k+1)}}\right\}^{-1}
    $$

    **근거.** 꼬리가 $P(X>x)\approx Cx^{-\alpha}$이면 문턱 $u$를 넘는 초과분의 로그가 근사적으로 $\text{Exp}(\alpha)$를 따른다. 그 평균의 역수가 $\alpha$의 MLE다.

    **$k$를 정하는 문제 — 전형적인 편향-분산 맞바꿈.**

    - **$k$가 작으면**: 진짜 꼬리만 쓰므로 편향이 작지만, 관측이 적어 **분산이 크다**. $\operatorname{SE}(\hat\alpha)\approx\alpha/\sqrt k$이므로 $k=20$이면 상대오차가 22%다.
    - **$k$가 크면**: 분산은 작지만 꼬리가 아닌 본체까지 포함해 **편향이 크다.** 대개 $\alpha$를 과대추정한다.

    **실무의 방법.**

    1. **힐 그림.** $\hat\alpha(k)$를 $k$에 대해 그린다. **평평한 구간**이 있으면 그 영역의 값을 쓴다. 평평한 구간이 없으면 거듭제곱 꼬리 가정 자체가 의심스럽다.
    2. **자동 선택.** 점근 MSE를 최소로 하는 $k^*$를 추정하는 방법들이 있다(이중 부트스트랩, 드 한-펠트). 다만 실무에서는 불안정하다는 평가가 많다.
    3. **경험 규칙.** $k\approx\sqrt n$이나 $k\approx0.05n$을 출발점으로 삼고 힐 그림으로 확인한다.

    **주의할 점.**

    - **힐 추정량은 $\alpha>0$을 전제한다.** 꼬리가 지수적으로 줄면(정규, 지수) $\hat\alpha$가 $k$에 따라 계속 커지며 수렴하지 않는다. 이것 자체가 진단 정보다.
    - **문턱 선택이 결론을 좌우한다.** $\hat\alpha$가 1.8이냐 2.2냐에 따라 "분산이 있다/없다"가 갈리므로, **$k$에 대한 민감도를 반드시 보고**해야 한다.
    - **자료가 독립이 아니면** 표준오차가 과소평가된다. 금융 시계열은 변동성 군집 때문에 극단값이 뭉쳐 나타난다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
**극단값 이론**의 두 접근(블록 최댓값, 임계값 초과)을 설명하고, 꼬리 위험 추정에 어느 쪽이 적합한지 논하라.

</div>

??? success "풀이"
    **접근 1 — 블록 최댓값(GEV).** 자료를 블록(연, 월)으로 나눠 각 블록의 최댓값을 모으고, **일반화극단값분포**를 적합한다.

    $$
    G(x) = \exp\left[-\left\{1+\xi\frac{x-\mu}{\sigma}\right\}^{-1/\xi}\right]
    $$

    - $\xi>0$: 프레셰(두꺼운 꼬리, $\alpha=1/\xi$).
    - $\xi=0$: 굼벨(지수 꼬리).
    - $\xi<0$: 와이불(위가 막힘).

    **근거.** 피셔-티펫-그네덴코 정리에 따라 최댓값의 극한분포가 이 셋뿐이다. 중심극한정리의 극단값 판이다.

    **접근 2 — 임계값 초과(POT/GPD).** 문턱 $u$를 넘는 초과분 $X-u$에 **일반화파레토분포**를 적합한다.

    $$
    H(y) = 1-\left(1+\frac{\xi y}{\beta}\right)^{-1/\xi}
    $$

    **근거.** 피컨즈-발케마-데 한 정리에 따라 $u$가 크면 초과분의 조건부분포가 GPD에 수렴한다.

    **비교.**

    | | 블록 최댓값 | 임계값 초과 |
    |---|---|---|
    | 자료 사용 | 블록당 하나만 | 문턱 넘는 것 모두 |
    | 효율 | 낮음 | **높음** |
    | 조율 모수 | 블록 크기 | 문턱 $u$ |
    | 해석 | "연 최대" 자연스러움 | 임의 분위수 |

    **꼬리 위험 추정에는 POT가 적합하다.** 이유는

    - **자료를 훨씬 많이 쓴다.** 연 최대만 쓰면 10년 자료에서 관측이 10개뿐이지만, POT는 상위 5%를 쓰면 126개다.
    - **VaR와 ES가 바로 나온다.** GPD를 적합하면 $q>u$인 분위수가

      $$
      \text{VaR}_q = u+\frac{\beta}{\xi}\left[\left\{\frac{n}{N_u}(1-q)\right\}^{-\xi}-1\right]
      $$

      로 닫힌 형태다. ES도 $\text{VaR}/(1-\xi)+(\beta-\xi u)/(1-\xi)$로 간단하다.

    **핵심 장점.** **관측 범위를 넘어서는 외삽**이 가능하다. 10년 자료로 100년 재현 수준을 추정할 수 있다. 경험적 분위수로는 불가능한 일이다. 물론 외삽이므로 불확실성이 크고, 모형 가정에 기댄다.

    **문턱 선택.** 힐 그림과 같은 문제다. 평균초과함수 그림(mean excess plot)이 문턱 위에서 직선이 되는 지점을 찾는 것이 표준적인 방법이다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
꼬리가 두꺼운 자료에서 **표본크기를 늘리는 것**이 얼마나 도움이 되는지 $\alpha$별로 정량화하라.

</div>

??? success "풀이"
    **평균 추정의 수렴 속도.**

    $$
    \bar X_n-\mu = O_p\!\left(n^{-(1-1/\alpha)}\right) \quad (1<\alpha<2), \qquad O_p(n^{-1/2}) \quad(\alpha>2)
    $$

    | $\alpha$ | 속도 | 표준오차를 절반으로 줄이려면 |
    |---|---|---|
    | $>2$ | $n^{-1/2}$ | 4배 |
    | 1.8 | $n^{-0.444}$ | 4.8배 |
    | 1.5 | $n^{-1/3}$ | **8배** |
    | 1.2 | $n^{-1/6}$ | **64배** |
    | $\le1$ | 수렴 안 함 | 불가능 |

    **$\alpha=1.5$면 8배, $\alpha=1.2$면 64배**가 필요하다. $\alpha\to1$에서 발산한다.

    **분위수 추정은 다르다.** 표본 분위수는 $\alpha$와 무관하게 $n^{-1/2}$로 수렴한다(그 점의 밀도가 양수이기만 하면). **꼬리가 두꺼워도 중앙값은 정상 속도로 개선된다.**

    **극단 분위수는 또 다르다.** $q$가 1에 가까우면 그 근처의 관측이 적어 정밀도가 떨어진다. $\text{VaR}_{0.99}$를 추정하려면 상위 1%에 관측이 충분해야 하므로, 앞서 본 대로 **실제로 관측된 초과 횟수**가 정밀도를 정한다.

    $$
    \frac{\operatorname{SE}}{\text{VaR}} \approx \frac{1}{\sqrt{n(1-q)}}
    $$

    $n=1000$, $q=0.99$면 초과가 10개이므로 상대오차가 32%다.

    **실무적 결론.**

    - **평균 기반 추론은 꼬리가 두꺼우면 자료를 늘려도 잘 나아지지 않는다.** 방법을 바꾸는 것이 자료를 늘리는 것보다 효과적이다.
    - **분위수로 옮기면 정상 속도를 회복**한다.
    - **극단 분위수는 모형(EVT)에 기대야** 한다. 순수 경험적 방법으로는 관측 범위를 넘을 수 없다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
꼬리가 두꺼운 자료에 **로그 변환**을 적용하면 무엇이 해결되고 무엇이 해결되지 않는가?

</div>

??? success "풀이"
    **해결되는 것.**

    - **거듭제곱 꼬리가 지수 꼬리가 된다.** $P(X>x)=Cx^{-\alpha}$이면 $Y=\ln X$에 대해

      $$
      P(Y>y) = Ce^{-\alpha y}
      $$

      로 **지수 꼬리**다. 모든 적률이 존재하고 중심극한정리가 적용된다.

    - **적률이 생긴다.** $\alpha\le2$라 분산이 없던 자료도 로그를 취하면 분산이 유한하다.

    - **곱셈적 구조가 덧셈적이 된다.** 소득이나 자산가치처럼 곱셈적으로 자라는 양에 자연스럽다.

    **해결되지 않는 것.**

    1. **추정 대상이 바뀐다.** $E[\ln X]\ne\ln E[X]$이므로, 로그 척도의 평균을 되돌리면 **기하평균(중앙값)** 이지 산술평균이 아니다. 앞서 본 스미어링 보정 문제다.

    2. **0과 음수를 다룰 수 없다.** $\ln(x+c)$로 이동하면 $c$의 선택이 결과를 좌우하고, 그 선택에 원리적 근거가 없다.

    3. **양쪽 꼬리를 다루지 못한다.** 로그는 오른쪽 꼬리만 압축한다. 수익률처럼 양쪽이 두꺼우면 도움이 안 된다.

    4. **꼬리 지수가 작으면 역부족일 수 있다.** $\alpha$가 아주 작으면 $\ln X$의 분산이 $1/\alpha^2$로 커져, 지수 꼬리이긴 해도 실무적으로 여전히 다루기 어렵다.

    5. **해석이 어려워진다.** 로그 척도의 회귀계수는 탄력성으로 읽히는데, 청중이 그것을 이해해야 한다.

    **대안들.**

    - **박스-콕스 변환.** $\lambda$를 자료에서 정해 로그($\lambda=0$)와 항등($\lambda=1$) 사이를 연속적으로 오간다. 다만 $\lambda$ 추정의 불확실성이 뒤의 추론에 반영되지 않는다는 문제가 있다.
    - **분위수로 옮긴다.** 변환 없이 중앙값과 분위수를 직접 보고한다.
    - **일반화선형모형.** 로그연결함수를 쓰되 평균을 직접 모형화한다. 역변환 편향이 생기지 않는다.
    - **꼬리를 모형화한다.** 변환으로 감추는 대신 EVT로 직접 다룬다.

    **권고.** **로그 변환은 "곱셈적 구조"라는 실질적 근거가 있을 때 쓴다.** 단지 "꼬리가 두꺼워서" 쓰면 추정 대상이 바뀐다는 대가를 치른다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
두꺼운 꼬리가 **분산투자와 포트폴리오 이론**에 미치는 영향을 논하라. 평균-분산 최적화가 언제 무너지는가?

</div>

??? success "풀이"
    **평균-분산 이론의 전제.** 마코위츠의 틀은 **분산이 위험의 충분한 요약**이라고 가정한다. 이는 (가) 수익률이 정규이거나 (나) 효용함수가 이차식일 때 정당화된다.

    **꼬리가 두꺼우면.**

    1. **분산이 위험을 대표하지 못한다.** 같은 분산이라도 꼬리가 두꺼우면 극단 손실 확률이 훨씬 크다. 앞서 본 대로 $t_5$의 99% ES가 정규보다 23% 크다.

    2. **$\alpha\le2$면 분산 자체가 없다.** 최적화 문제가 정의되지 않는다. 개별 주식 수준에서는 $\alpha$가 3~4 정도로 추정되어 분산은 존재하지만, 일부 자산이나 고빈도 자료에서는 2에 가까워진다.

    3. **분산투자의 이득이 줄어든다.** $\alpha<2$이면 독립인 $n$개를 섞어도 포트폴리오의 꼬리 지수가 그대로 $\alpha$다. 대수법칙이 작동하지 않아 **"많이 모으면 안전해진다"는 논리가 깨진다.**

    4. **공분산 추정이 불안정하다.** 꼬리가 두꺼우면 표본공분산의 분산이 커지고, 앞서 본 차원 문제와 겹쳐 최적화 결과가 극단적인 가중치를 낸다.

    5. **꼬리의존성.** 개별 자산이 정규여도 코퓰라가 꼬리의존을 가지면 동시 폭락이 일어난다. 상관계수만으로는 잡히지 않는다.

    **대안.**

    - **위험측도를 바꾼다.** 분산 대신 ES를 최소로 한다(CVaR 최적화). 선형계획으로 풀리며, 정합적 위험측도라는 이론적 근거도 있다.
    - **분포를 바꾼다.** $t$ 코퓰라나 다변량 $t$로 모형화한다.
    - **강건 최적화.** 모수의 불확실성 집합을 두고 최악의 경우를 최적화한다.
    - **공분산을 축소한다.** 르두아-울프나 요인모형.
    - **가중치에 제약을 건다.** 공매도 금지, 최대 비중 제한. 이론적 최적성은 잃지만 표본 밖 성능이 대개 낫다.

    **가장 중요한 것.** **분산투자가 무엇을 지워 주고 무엇을 지워 주지 못하는지 구분**하는 것이다. 개별 위험은 지워지고 체계적 위험은 남으며, 꼬리의존이 있으면 위기 시에 분산 효과가 사라진다. 평상시 자료로 추정한 상관계수를 위기에 적용하면 안 된다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
꼬리가 두꺼운 자료를 분석하는 **표준 작업 흐름**을 정리하라. 진단부터 보고까지 순서대로 적어라.

</div>

??? success "풀이"
    **1단계 — 시각화.**

    - 히스토그램(선형·로그 세로축), 상자그림, Q-Q 그림(정규·$t$·지수).
    - **로그-로그 생존함수 그림**이 핵심이다. 직선이면 거듭제곱 꼬리, 기울기가 $-\alpha$.

    **2단계 — 적률 존재 여부 확인.**

    - 누적평균과 누적 $s$ 그림.
    - 힐 그림으로 $\hat\alpha$ 추정. $\hat\alpha\le2$면 분산 없음, $\le1$이면 평균 없음.
    - **여기서 이후 전략이 갈린다.**

    **3단계 — 방법 선택.**

    | $\hat\alpha$ | 중심 | 산포 | 추론 |
    |---|---|---|---|
    | $>4$ | 평균 | $s$ | 표준 |
    | $2\sim4$ | 평균(주의) 또는 절사평균 | $s$(느림) | 부트스트랩 |
    | $1\sim2$ | 중앙값, 절사평균 | IQR, MAD | 분위수 기반 |
    | $\le1$ | 중앙값만 | IQR | 순서통계량 구간 |

    **4단계 — 꼬리가 관심이면 EVT.**

    - 문턱을 정하고(평균초과함수 그림) GPD를 적합.
    - $\hat\xi$의 신뢰구간과 문턱 민감도를 확인.
    - VaR·ES를 계산하되 **외삽의 불확실성을 명시.**

    **5단계 — 검증.**

    - 부트스트랩으로 추정값의 안정성 확인($m$-out-of-$n$이 필요할 수 있음).
    - 자료를 반으로 나눠 결론이 일관되는지.
    - 극단값 몇 개를 빼고 다시 계산해 영향력 확인.

    **6단계 — 보고.**

    - **$\hat\alpha$와 그 불확실성**을 보고한다. 이것이 독자가 결과를 해석하는 열쇠다.
    - **어떤 중심·산포 측도를 왜 썼는지** 밝힌다.
    - 평균을 보고한다면 **그것이 소수의 관측값에 얼마나 의존하는지** 함께 적는다.
    - 분포 그림을 반드시 포함한다. 요약통계만으로는 꼬리를 전달할 수 없다.

    **가장 흔한 실수.** 진단 없이 평균과 표준편차를 계산하고 $\pm1.96\operatorname{SE}$ 구간을 붙이는 것이다. **1단계와 2단계에 드는 비용은 몇 줄이며, 그것이 나머지 전부를 구한다.**

---

## 정리하며

꼬리가 두꺼운 분포는 통계적 추론에 상당한 어려움을 준다:

- 표본평균이 비효율적이고 불안정하다
- 정규이론 신뢰구간이 너무 좁다
- 정규성에 기반한 위험 측도가 위험할 정도로 낙관적이다

특히 금융 자료에서는 중앙값, 절사평균, M-추정량 같은 로버스트한 대안이 더 믿을 만한 추론을 준다. (분포 가정이 필요 없는) 붓스트랩 방법은 이런 추정량 어느 것에 대해서든 신뢰구간을 만드는 데 이상적이다.
